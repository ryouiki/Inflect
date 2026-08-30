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
# 1. a language base from a multi-speaker corpus
inflect-adapt train --base owensong/Inflect-Micro-v2 \
  --dataset prepared/ja-base --output runs/ja-base
inflect-adapt export --checkpoint runs/ja-base/checkpoints/adaptation-final.pth \
  --prepared-dataset prepared/ja-base \
  --package-template <released Micro directory> \
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
  --format pytorch \
  --output exports/es-micro
```

ONNX:

```bash
inflect-adapt export \
  --checkpoint runs/es-micro/checkpoints/adaptation-final.pth \
  --prepared-dataset prepared/es \
  --format onnx \
  --output exports/es-micro-onnx
```

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

## Clean-environment test

Before release:

1. build and install the wheel in a clean environment;
2. download only the public base and adapted artifacts;
3. verify `checksums.sha256`;
4. run inference without the training checkout;
5. test ordinary and difficult text;
6. test the target frontend mode;
7. repeat ONNX parity on the release artifact if ONNX is published.
