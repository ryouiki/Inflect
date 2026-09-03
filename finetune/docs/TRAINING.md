# Training, resume, evaluation, and export

## Warm-start behavior

`--base` accepts a released Micro or Nano checkpoint, or a checkpoint this
toolkit produced earlier. Either way it is an inference generator, not a
training snapshot. The trainer:

1. builds the compatible training form of Micro or Nano;
2. copies all compatible released generator weights;
3. migrates text embeddings by symbol identity;
4. deterministically initializes newly added symbols;
5. initializes the training-only posterior encoder and discriminator;
6. creates fresh public optimizers, schedulers, scaler, and RNG state;
7. records hashes for the base model, prepared data, symbols, and options.

This begins a new adaptation run. It does not continue the private release run.

## Chaining two runs

Moving language and voice at the same time asks every part of a small generator
to move at once. Splitting the work is one way to reduce that:

```bash
# 1. a language base
inflect-adapt train --base owensong/Inflect-Micro-v2 \
  --dataset prepared/ja-base --output runs/ja-base
inflect-adapt export --checkpoint runs/ja-base/checkpoints/adaptation-final.pth \
  --prepared-dataset prepared/ja-base \
  --package-template micro \
  --format pytorch --output exports/ja-base

# 2. the target voice, warm-started from it
inflect-adapt train --base exports/ja-base \
  --dataset prepared/ja-voice --output runs/ja-voice
```

A base checkpoint may use a larger symbol inventory than the release, as long
as it keeps the released symbols at their released positions. Embedding rows
are matched by symbol string, so:

- symbols present in both are copied, wherever they sit in either inventory;
- symbols only the new dataset uses are initialized;
- symbols only the base had are dropped.

That last case is correct — the new dataset does not use them — but it discards
weights the earlier run trained. `compatibility-report.json` records
`base_symbol_count` and `discarded_base_symbols` so it is visible. If you did
not expect a symbol to be dropped, prepare the second dataset with
`--base-symbols exports/ja-base/symbols.json` so both inventories agree.

Resume is a different thing and still refuses a changed base: chaining starts a
new run, with a new output directory and a new run identity.

Both datasets must still be single-speaker. Preparation rejects a manifest with
more than one speaker value, so a language base built from a multi-speaker
corpus is not supported today — the first stage has to be one speaker whose
recordings you are willing to move away from in the second.

## Stages

The generic public schedule has three stages:

- `posterior_warmup`: establish the training-only posterior path;
- `linguistic_adaptation`: adapt timing, text, latent, and acoustic behavior;
- `decoder_polish`: optionally unfreeze the waveform decoder at a lower
  learning-rate multiplier.

Stage boundaries are explicit and resumable. `--decoder-unfreeze-step none`
keeps the decoder frozen.

## Presets and overrides

| Preset | Intended starting point |
| --- | --- |
| `balanced` | General CUDA starting point |
| `micro-12gb` | Conservative Micro setup near the 12 GB class |
| `nano-8gb` | Small-batch Nano setup near the 8 GB class |

Actual memory depends on clip length, model, batch size, validation, PyTorch,
CUDA, and allocator behavior. Preset names are not hardware guarantees.

Every explicitly supplied CLI flag overrides its preset, even when the value
matches a library default:

```bash
inflect-adapt train \
  --base owensong/Inflect-Micro-v2 \
  --dataset prepared/es \
  --preset micro-12gb \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --output runs/es-micro
```

If memory is exhausted, reduce the batch size and increase accumulation. Do not
compare two runs as equivalent if precision, segment length, effective batch,
or optimization settings differ.

## Outputs

```text
runs/es-micro/
  run-identity.json
  training-options.json
  compatibility-report.json
  metrics.jsonl
  training-summary.json
  checkpoints/
    adaptation-step-00001000.pth
    adaptation-final.pth
    latest.pth
  exports/
    model-step-00001000.pth
    model.pth
  validation/
    step-00000500.json
    step-00000500.wav
```

Training checkpoints contain the public training state needed for exact
same-run resume. Files in `exports/` are lightweight generator checkpoints,
not complete deployment packages.

## Resume identity

Resume validates before mutating model or optimizer state. It rejects
incompatible toolkit schema, run ID, base model, prepared dataset, symbols,
frontend, model shapes, optimizer schema, or relevant options.

```bash
inflect-adapt train \
  --base owensong/Inflect-Micro-v2 \
  --dataset prepared/es \
  --preset micro-12gb \
  --output runs/es-micro \
  --resume runs/es-micro/checkpoints/latest.pth
```

Do not bypass a rejection. Start a new output directory when the data or
configuration changes.

## Checkpoint selection

Validation intervals create fixed-seed held-out synthesis and loss
diagnostics. The trainer intentionally does not create `best.pth`, because no
single training loss reliably identifies the best-sounding TTS checkpoint.

Declare a selection rule before inspecting the final test set. Consider:

- intelligibility on difficult held-out text;
- clipping, silence, duration, and truncated endings;
- pronunciation and phoneme coverage;
- voice consistency;
- buzz, metallic resonance, sibilance, thinness, and transients;
- blinded listening by fluent speakers.

Predicted MOS, ASR WER, and training loss are useful diagnostics, not complete
quality measures.

## Deployment exports

PyTorch:

```bash
inflect-adapt export \
  --checkpoint runs/es-micro/checkpoints/adaptation-final.pth \
  --prepared-dataset prepared/es \
  --package-template micro \
  --format pytorch \
  --output exports/es-micro
```

ONNX:

```bash
inflect-adapt export \
  --checkpoint runs/es-micro/checkpoints/adaptation-final.pth \
  --prepared-dataset prepared/es \
  --package-template micro \
  --format onnx \
  --output exports/es-micro-onnx
```

`--package-template` resolves like `--base` — `micro`, `nano`, a Hugging Face
repository ID, or a local release directory. The package needs that runtime,
and the exporter cannot infer it: a training checkpoint deliberately omits its
base model from the saved options, because those options are hashed into the
run identity that guards resume. Without it a verified export fails rather than
writing an unusable package.

The exporter carries the exact language/frontend contract into the package.
eSpeak packages use the configured language, prephonemized packages require
phoneme input, and custom packages require a source-hash-matching
`--frontend-hook`.

The exporter removes training-only state and verifies strict inference loading.
When ONNX is requested, it exports separate duration and decode graphs and
compares them with PyTorch on fixed inputs. Quantized export is not included.

## Evaluation

```bash
inflect-adapt evaluate \
  --model-dir exports/es-micro \
  --manifest prepared/es/validation.jsonl \
  --output evaluations/es-micro
```

Evaluation writes audio, per-item diagnostics, aggregate JSON, and a readable
summary. Supply `--transcript-evaluator module:callable` to integrate a
project-specific ASR or metric implementation.

The diagnostics include three pitch observables per clip: `f0_median_hz`,
`f0_iqr_semitones`, and `voiced_frame_fraction`. Read the first two together.
A change that only moves the median has a degenerate solution where the contour
flattens, which measures as success and sounds worse, and the interquartile
range in semitones is what shows the contour still moving. The search range is
60 Hz to 1 kHz by default; a ceiling set near a target's own high notes reports
a falling contour that is an artifact of the setting. A clip with no voiced
frames reports no pitch rather than a pitch of zero, and the aggregate skips it
instead of averaging the zero in.

A row that carries an `audio` field is read from disk and no model is loaded,
which is what makes the prepared `validation.jsonl` the real-audio anchor: the
same code path measures the recording and the render, so the two numbers are
comparable. To render instead, pass a copy of those rows with `audio` and
`phonemes` removed. Passing `validation.jsonl` itself while varying
`--checkpoint` scores the same recordings every time.

## Blind listening

Automatic diagnostics screen candidates; they do not rank them. Two examples
build the round that does:

```bash
python examples/build_blind_ab_page.py \
  --system step6000=evaluations/es-micro-step6000 \
  --system step8000=evaluations/es-micro-step8000 \
  --anchor evaluations/es-micro-val-real-anchor \
  --must-include-ids review/high-register-ids.txt \
  --rows 32 --output listening/round1

python examples/tally_verdict.py \
  --mapping listening/round1/mapping.json \
  --verdict round1-verdict.json
```

The page relabels every row with fresh random letters, levels every clip to one
RMS by pure gain, forces the real-audio anchor onto each row, asks for a
description rather than a bare number, and requires free text on what the defect
sounded like. Which letter was which system lives only in `mapping.json`, so
scoring and tallying are separate steps; a letter tallied directly is noise.
`--catch-rows` puts one system on a row twice, and the two scores it earns for
byte-identical audio are that round's noise floor.

Score one round at a time. Absolute scores drift between sessions, so only
contrasts inside a single page are comparable, and none of this is MOS.

## Clean-environment test

Before release:

1. build and install the wheel in a clean environment;
2. download only the public base and adapted artifacts;
3. verify `checksums.sha256`;
4. run inference without the training checkout;
5. test ordinary and difficult text;
6. test the target frontend mode;
7. repeat ONNX parity on the release artifact if ONNX is published.
