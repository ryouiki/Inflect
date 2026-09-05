# Training, resume, evaluation, and export

## Warm-start behavior

`--base` accepts a released Micro or Nano checkpoint, or a checkpoint this
toolkit produced earlier. Either way it is an inference generator, not a
training snapshot. The trainer:

1. builds the compatible training form of Micro or Nano;
2. copies all compatible released generator weights;
3. migrates text embeddings by symbol identity;
4. deterministically initializes newly added symbols;
5. initializes the training-only posterior encoder and discriminator, unless
   `--posterior-init inherit` reads a posterior sidecar from the base;
6. creates fresh public optimizers, schedulers, scaler, RNG state, and, when
   `--generator-ema-decay` is set, an averaged copy of the generator;
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

By default the second run initializes its own posterior encoder, paying that
cost twice. Exporting the first run with `--include-posterior` writes a
`posterior.pth` sidecar beside `model.pth`, and `--posterior-init inherit`
reads it, so the second run continues the posterior the first one trained. The
sidecar's hash joins the run identity, so swapping it is caught exactly like
swapping the base checkpoint. Inherit deliberately: a posterior whose latents
had already drifted hands that drift to the next run, and the `z_dc_rms` column
will show it from the first step.

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

A frozen decoder does not mean a quiet decoder. The discriminator scores the
audio it produces, and those gradients travel back through it into the
posterior encoder and the flow. For the first two stages the model is therefore
being pushed to satisfy a freshly initialized critic by moving its latents,
with the one component that could answer properly held still. That is the
mechanism behind the frame-rate comb described below, and the controls in this
section exist to interrupt it. All of them are off by default.

`--adversarial-gating` holds the generator's adversarial and feature-matching
terms at exactly zero for as long as the decoder is frozen, then raises them to
full weight linearly over `--adversarial-ramp-steps`, counted from the unfreeze
step. The discriminator trains throughout, so the gated window is its warm-up
rather than lost time. Gating with `--decoder-unfreeze-step none` is accepted
and warned about: it means adaptation with no adversarial term at all.

`--decoder-lr-warmup-steps` eases the decoder in after it unfreezes, scaling
its learning rate by `min(1, (step - unfreeze) / warmup)`. The optimizer starts
that parameter group with empty moment estimates, so without a warm-up its
first update is the largest it will ever take, on the component whose released
weights are the most valuable thing in the run. The scale is zero on the first
step by design: the moments fill and the weights do not move. The learning rate
recorded in `metrics.jsonl` and in the checkpoint is the unscaled nominal one,
and `decoder_lr_scale` reports the factor separately.

`--decoder-polish-mode recon` changes what the polish stage is. It trains only
the decoder, against reconstruction losses with no discriminator at all, and
asks a single question: can the decoder render the latents it is already given,
cleanly. The posterior encoder and the flow are held, so those latents stop
moving while it answers. Because the linguistic stage ends where the polish
stage begins, a run that needs more language adaptation should move
`--decoder-unfreeze-step` later rather than shorten the polish. The mode
cannot be turned off partway through a run; going back to an adversarial polish
means a new chained run.

`--stft-loss-weight` adds a multi-resolution linear STFT reconstruction term
alongside the mel loss, at three resolutions with 512, 1024 and 2048-point
transforms. An 80-band mel averages over bands hundreds of hertz wide in the
top octaves and barely charges for a narrow comb sitting there; the 2048-point
resolution has 11.7 Hz bins at 24 kHz and does. The term is the mean over
resolutions, following Parallel WaveGAN, so a weight quoted for the summed
convention is worth three times as much here.

`--decoder-proximal-weight` holds the decoder near the weights the run started
from, measuring each tensor's squared drift relative to that tensor's own
squared norm. The normalisation is what makes it steer. An unnormalised mean
over millions of elements stays near zero even for a drift that ruins the
render, so it reads as satisfied while nothing is being held.

`--decoder-freeze-upsamplers` holds the transposed convolutions and the input
convolution while the residual stack and the output convolution train. The
upsamplers are where the frame grid enters the waveform. It is a gradient mask
inside the existing decoder parameter group rather than a fourth group, so the
shape of saved optimizer state does not change.

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
not complete deployment packages. With `--generator-ema-decay` set, each
`model-step-*.pth` is joined by a `model-ema-step-*.pth`, and the run ends with
both `model.pth` and `model-ema.pth`; the averaged copy is a second candidate
for the listening round at almost no cost.

`metrics.jsonl` carries one row per optimizer step. Alongside the loss terms it
records `adversarial_weight` and `decoder_lr_scale`, which say what the
schedule was doing, and `z_dc_rms` and `z_rms`, which say how far the latents
have moved. A term the schedule switched off is written as `null`, never as
`0.0`: a zero would read as measured and negligible, which is a different
claim. Each entry in `validation/step-*.json` carries the two cheap comb
screens for that clip, as a trend to watch rather than a gate.

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

The options are part of that identity, so a release that adds an option field
makes every earlier checkpoint unresumable. A checkpoint written before the
controls in this section existed fails with `identity fields differ:
['options']`. That is the guard working, not a bug in it, and the way forward
is a new run or a chained one rather than a way around it.

Nothing about the gate, the ramp or the warm-up is stored in the checkpoint,
because each is a function of the step count, and the proximal anchor is
re-derived from the base the run started from. The averaged generator is the
only new state, and it rides in the checkpoint under `generator_ema`. A run
that asks for an average and resumes from a checkpoint written without one is
rejected rather than silently restarted from the base weights.

## Checkpoint selection

Validation intervals create fixed-seed held-out synthesis and loss
diagnostics. The trainer intentionally does not create `best.pth`, because no
single training loss reliably identifies the best-sounding TTS checkpoint.

Declare a selection rule before inspecting the final test set. Consider:

- intelligibility on difficult held-out text;
- clipping, silence, duration, and truncated endings;
- pronunciation and phoneme coverage;
- voice consistency;
- buzz, metallic resonance, sibilance, thinness, and transients, for which
  the frame-grid observables below give a machine-readable first pass;
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
project-specific ASR or metric implementation;
`examples/transcript_evaluator_asr.py` is a working Japanese and Korean one,
scoring CER on kana and on jamo respectively. It reads its model from
`INFLECT_ASR_MODEL_DIR` with `local_files_only`, because an ASR model arriving
by surprise during evaluation is a different measurement than the one you meant
to make.

The diagnostics include three pitch observables per clip: `f0_median_hz`,
`f0_iqr_semitones`, and `voiced_frame_fraction`. Read the first two together.
A change that only moves the median has a degenerate solution where the contour
flattens, which measures as success and sounds worse, and the interquartile
range in semitones is what shows the contour still moving. The search range is
60 Hz to 1 kHz by default; a ceiling set near a target's own high notes reports
a falling contour that is an artifact of the setting. A clip with no voiced
frames reports no pitch rather than a pitch of zero, and the aggregate skips it
instead of averaging the zero in.

Four more observables per clip look for a comb on the decoder's frame grid,
which is the sample rate over the hop, so 93.75 Hz for Micro at 24 kHz. The hop
comes from the model config unless `--hop-length` says otherwise.

| Observable | What it is |
| --- | --- |
| `grid_tone_excess_db` | On-grid power over off-grid power above 2 kHz, in dB. Zero by construction for real speech. |
| `fold_periodic_excess_db` | The clip folded into hop-length frames, relative to the floor an uncorrelated signal would give. `fold_periodic_db` is the same measure without that correction and moves with clip length. |
| `steady_tone_artifact_score` | Summed prominence of spectral peaks that are both steady across frames and above 1200 Hz. |
| `f0_grid_deviation_hz` | Distance from the measured pitch to the nearest small multiple of the grid. A tracker fed a ringing render reports the comb as the voice. |

Measured on one speaker, comparing 40 real recordings with 40 renders from a
run a listener rejected for ringing:

| Observable | Real, p50 and max | Ringing, p50 and max |
| --- | --- | --- |
| `grid_tone_excess_db` | -0.13 and 3.50 | 8.15 and 9.77 |
| `fold_periodic_excess_db` | -0.16 and 5.19 | 4.17 and 8.35 |
| `steady_tone_artifact_score` | 0.00 and 0.00 | 29.9 and 66.3 |

At the shipped thresholds the grid-tone and steady-tone screens flagged none of
the recordings and every one of the renders. The fold measure overlaps the two
populations, so it corroborates rather than accuses. The aggregate reports
`clips_grid_tone_flagged`, `clips_fold_periodic_flagged`,
`clips_steady_tone_flagged` and `clips_f0_locked_to_frame_grid`.

Two cautions. A pitch that happens to sit on the grid raises the grid-tone
excess legitimately: a static 250 Hz tone scores +7.19 dB because every third
harmonic lands on 750 Hz. Real speech pitch moves, so this did not appear in
the recordings above, and it is why the pitch deviation is reported next to the
excess rather than the excess alone. And these thresholds come from one speaker
pair on one render channel, so a clip sitting just inside one has proven
nothing. A screen that cannot measure a clip reports nothing and raises no
flag, rather than treating silence as innocence.

## Ringing at multiples of the frame rate

Two adaptation rounds were rejected by a listener for a steady comb of tones at
multiples of 93.75 Hz, audible even in the silence between words. The decoder
upsamples with transposed convolutions and no anti-imaging filter, so that grid
is where images land once anything upstream goes wrong.

The first investigation blamed the latents. The released checkpoint carries no
posterior encoder, so a fresh one starts every run, the latent channel-mean RMS
moved from the released 0.74 to 1.42 and 1.51 on the failing runs, and feeding
those latents to the released decoder rang harder than feeding them to the
adapted one. That reading did not survive being acted on, which is the reason
this section reads as it does.

Two of the controls below were built from it and tried on a Korean corpus, at
10,000 steps each, against a run made without them. Neither helped, and the
drift itself turned out not to track the artifact. Read what follows as a
record of what was measured, not as a recipe.

```bash
inflect-adapt train \
  --base owensong/Inflect-Micro-v2 \
  --dataset prepared/ko \
  --preset balanced \
  --adversarial-gating \
  --adversarial-ramp-steps 1000 \
  --decoder-lr-warmup-steps 300 \
  --decoder-polish-mode recon \
  --stft-loss-weight 1.0 \
  --decoder-proximal-weight 0.1 \
  --generator-ema-decay 0.999 \
  --output runs/ko-micro
```

Measured at the endpoint, over 40 held-out clips scored the same way:

| Run | Grid-tone excess dB, p50 | Steady-tone score, p50 | Tracked pitch, p50 |
| --- | --- | --- | --- |
| the speaker's own recordings | -0.13 | 0.00 | 363.9 Hz |
| no controls | 8.15 | 29.9 | 362.7 Hz |
| gating, ramp, learning-rate warm-up | 8.27 | 27.1 | 376.9 Hz |
| the above plus a reconstruction polish | 11.53 | 7.4 | 93.8 Hz |

Gating changed nothing. It also raised the latent drift rather than lowering
it, from 1.190 to 1.528 against the released decoder's own 0.737, because the
adversarial term had been pulling the latents back toward what the
discriminator accepts and gating removed that force for 3000 steps. The
reconstruction polish was worse still: its median pitch collapsed onto the
comb frequency itself, 93.76 Hz against the speaker's 364 Hz, and 31 of the 40
clips had their pitch tracked to the grid. Its steady-tone score fell while
every other measure rose, which is why more than one screen is reported.

The reconstruction polish deserves a specific warning. Its own losses improved
throughout: the mel term fell from 0.936 to 0.726. The comb rose at the same
time, on the very latents the term was fitting. Reconstruction losses,
including the multi-resolution STFT term at weight 1.0, are not sensitive to
this artifact, so optimizing them harder is not a way out of it.

What survives is the measurement. The screens detect the artifact without a
listening round, they dated its arrival in these runs to between steps 500 and
1000, and with every control at its default the loss and the schedule are
unchanged. Watch `z_dc_rms` in `metrics.jsonl` and the grid-tone excess in
`validation/step-*.json` while a run is still cheap to abandon. The cause
remains open: the latent drift the diagnosis blamed turns out not to track the
artifact, since runs ending at 1.190 and 1.528 produced 8.15 and 8.27 dB of
comb.

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
