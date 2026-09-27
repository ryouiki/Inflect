# Inflect adaptation toolkit

Public tools for warm-starting Inflect v2 with user-owned speech data. One run
creates one fixed voice for one configured language.

This is a generic compatible trainer, not the private process used to create
the official Inflect v2 checkpoints. It does not include the private corpus,
corpus-generation process, original curriculum, hyperparameter search, or
internal checkpoint-selection process.

## What is implemented

- JSONL and CSV manifests with strict path and audio validation
- deterministic 24 kHz mono preparation
- leakage-safe train/validation splitting by transcript and recording group
- eSpeak, prephonemized, bundled, and explicit custom Python frontends
- Japanese and Korean frontends that add no symbols to the released inventory
- symbol-aware embedding migration from Micro or Nano
- staged generator/discriminator training with AMP and accumulation
- opt-in controls for the frame-rate comb: adversarial gating, decoder
  learning-rate warm-up, reconstruction-only polish, multi-resolution STFT
  loss, decoder proximal anchor, upsampler freeze, generator averaging
- automatic frame-grid artifact screens on every evaluated clip
- atomic checkpoints and strict same-run resume validation
- held-out waveform diagnostics and optional transcript evaluators
- inference-only PyTorch and ONNX packages
- language-aware deployment frontends with no silent English fallback

The software path has been tested end to end, including a real Nano CUDA
training step, resume, strict PyTorch load, and ONNX Runtime parity. New
language and new voice quality remain experimental: data quality, phoneme
coverage, and fluent-speaker review determine whether an adaptation is useful.

Read [CONTRACT.md](CONTRACT.md) and [scope](docs/SCOPE.md) before starting.

## Install

From this directory:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Activate on PowerShell with:

```powershell
.\.venv\Scripts\Activate.ps1
```

For ONNX export and parity checks:

```bash
python -m pip install -e ".[onnx]"
```

Every command is available as either `inflect-adapt` or
`python -m inflect_finetune`.

## 1. Create a manifest

JSONL and CSV are supported. A minimal JSONL row is:

```json
{"audio":"session01/000001.wav","text":"Buenos dias.","speaker":"voice-a","session":"session01"}
```

`audio` and `text` are required. `speaker`, `session`, `group_id`, `id`, and
`phonemes` are documented in [MANIFESTS.md](docs/MANIFESTS.md).

Use one consenting speaker per dataset. Put clips cut from the same source
recording in the same `group_id` or `session` so they cannot leak across
splits.

## 2. Prepare and audit

For an eSpeak-supported language:

```bash
inflect-adapt prepare \
  --manifest data/metadata.jsonl \
  --audio-root data/audio \
  --language es \
  --frontend espeak \
  --output prepared/es

inflect-adapt audit --dataset prepared/es
```

Preparation writes converted audio, ordered symbols, frontend identity, hashes,
and deterministic nonempty train/validation splits. Do not train until audit
passes and a fluent speaker has inspected representative normalized text and
phonemes.

Some languages ship a bundled frontend instead. Japanese needs one because
eSpeak cannot read kanji, and Korean because eSpeak merges the tense/plain
consonant contrast:

```bash
python -m pip install ".[ja]"

inflect-adapt prepare \
  --manifest data/metadata.jsonl \
  --audio-root data/audio \
  --language ja \
  --frontend ja-openjtalk \
  --output prepared/ja

inflect-adapt audit --dataset prepared/ja --require-no-new-symbols
```

If neither eSpeak nor a bundled frontend is suitable, use prephonemized rows or
the documented [custom frontend hook](docs/CUSTOM_G2P.md). See
[languages and symbols](docs/LANGUAGES.md) for the bundled list.

## 3. Train

Start with Micro unless minimum footprint is the primary goal:

```bash
inflect-adapt train \
  --base owensong/Inflect-Micro-v2 \
  --dataset prepared/es \
  --preset micro-12gb \
  --output runs/es-micro
```

Available presets are `balanced`, `micro-12gb`, and `nano-8gb`. They are
starting points, not memory or quality guarantees. CLI flags override preset
values exactly:

```bash
inflect-adapt train \
  --base nano \
  --dataset prepared/es \
  --preset nano-8gb \
  --batch-size 1 \
  --gradient-accumulation-steps 12 \
  --output runs/es-nano
```

The public release checkpoint contains inference weights only. Training-only
posterior and discriminator components are initialized by this toolkit, and
new symbol embeddings are initialized deterministically. Chaining a second run
onto an export made with `--include-posterior` is the one exception, and it
inherits from that export, never from the release.

Adaptations of this model family have produced a steady comb of tones at
multiples of the frame rate. The controls for it are off by default, so the
command above behaves as it always has; `docs/TRAINING.md` explains what each
one does and `docs/TROUBLESHOOTING.md` describes the symptom.

```bash
inflect-adapt train \
  --base owensong/Inflect-Micro-v2 \
  --dataset prepared/es \
  --preset micro-12gb \
  --adversarial-gating \
  --decoder-lr-warmup-steps 300 \
  --generator-ema-decay 0.999 \
  --output runs/es-micro
```

## 4. Resume safely

```bash
inflect-adapt train \
  --base owensong/Inflect-Micro-v2 \
  --dataset prepared/es \
  --preset micro-12gb \
  --output runs/es-micro \
  --resume runs/es-micro/checkpoints/latest.pth
```

Resume is accepted only for the same run identity. Changes to the base model,
prepared data, symbols, frontend, public optimizer schema, or relevant
configuration are rejected. The options are part of that identity, so this
release's new option fields make checkpoints written by earlier versions
unresumable, even at their defaults.

The final resumable checkpoint is
`runs/es-micro/checkpoints/adaptation-final.pth`. Step checkpoints and held-out
audio are written at the configured intervals. The toolkit does not label a
checkpoint "best"; select one using a declared held-out process.

## 5. Export

Export the selected training checkpoint together with the exact prepared
frontend metadata:

```bash
inflect-adapt export \
  --checkpoint runs/es-micro/checkpoints/adaptation-final.pth \
  --prepared-dataset prepared/es \
  --package-template micro \
  --format pytorch \
  --output exports/es-micro
```

For ONNX:

```bash
inflect-adapt export \
  --checkpoint runs/es-micro/checkpoints/adaptation-final.pth \
  --prepared-dataset prepared/es \
  --package-template micro \
  --format onnx \
  --output exports/es-micro-onnx
```

For a custom frontend, also pass the same source file:

```bash
inflect-adapt export \
  --checkpoint runs/custom/checkpoints/adaptation-final.pth \
  --prepared-dataset prepared/custom \
  --frontend-hook my_frontend.py \
  --package-template micro \
  --format onnx \
  --output exports/custom
```

`--package-template` supplies the runtime the package must carry, and a
verified export requires it: a training checkpoint does not record which base
model produced it. It resolves like `--base`, so `micro`, `nano`, a Hugging
Face repository ID, or a local release directory all work.

Export strips posterior, discriminator, optimizer, scheduler, scaler, RNG, and
other training-only state. It writes `frontend.json`, `symbols.json`,
checksums, a report, a deployable runtime, and optional ONNX graphs. Adapted
checkpoints without frontend metadata are rejected instead of silently using
the release English frontend.

## 6. Evaluate the exported package

Use original text for eSpeak/custom packages or include `phonemes` in each
evaluation row for a prephonemized package:

```bash
inflect-adapt evaluate \
  --model-dir exports/es-micro \
  --manifest prepared/es/validation.jsonl \
  --output evaluations/es-micro \
  --max-samples 100
```

The report covers waveform duration, silence, clipping, peak, RMS, DC offset,
and non-finite values. An optional transcript-evaluator hook can add ASR or
other metrics. These diagnostics do not replace blind listening or
fluent-speaker review.

## Documentation

- [Supported scope](docs/SCOPE.md)
- [Manifests and split safety](docs/MANIFESTS.md)
- [Data quality](docs/DATA_QUALITY.md)
- [Languages and symbols](docs/LANGUAGES.md)
- [Training, checkpoints, evaluation, and export](docs/TRAINING.md)
- [Custom G2P/frontend hooks](docs/CUSTOM_G2P.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Consent and responsible use](docs/RESPONSIBLE_USE.md)
- [Multilingual extension roadmap](docs/MULTILINGUAL_ROADMAP.md)

## Release gate

Before publishing an adapted checkpoint:

1. Re-run preparation and audit from the source manifest.
2. Confirm groups and duplicate transcripts do not cross the split.
3. Compare several checkpoints on untouched held-out text.
4. Load the exported package in a clean environment.
5. Run its checksum, inference, and ONNX parity checks where applicable.
6. Have fluent speakers review pronunciation and naturalness.
7. Publish data provenance, consent, language, voice, frontend, base model,
   toolkit version, evaluation method, and limitations.

Passing the software checks does not establish naturalness, voice identity, or
acceptable pronunciation.
