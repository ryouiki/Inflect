# Deployment

Inflect v2 ships as two self-contained Hugging Face model repositories with the
same Python and command-line interface:

- [Inflect-Micro-v2](https://huggingface.co/owensong/Inflect-Micro-v2) for the
  stronger quality/footprint balance.
- [Inflect-Nano-v2](https://huggingface.co/owensong/Inflect-Nano-v2) when model
  size is the primary constraint.

The GitHub repository contains integration examples and documentation. The
weights, frontend, runtime module, evaluation artifacts, and integrity hashes
live in each Hugging Face package.

## Install

```bash
git clone https://github.com/owenawsong/Inflect.git
cd Inflect
python -m pip install -r requirements.txt
```

The runtime requires `espeak-ng`. The `espeakng-loader` dependency supplies a
portable binary for common environments; system installations are also
supported by `phonemizer`.

## Download and synthesize

```bash
python examples/download_and_speak.py \
  --model micro \
  --text "A complete local voice can fit almost anywhere." \
  --output inflect.wav
```

Use `--model nano` for the smaller package and `--device cuda` when CUDA is
available. The first run downloads and caches the selected Hugging Face model.

## Compare both models

```bash
python examples/compare_models.py \
  --text "Use the same prompt and seed for a direct comparison." \
  --output-dir comparison
```

This writes `inflect-micro-v2.wav` and `inflect-nano-v2.wav` with identical
runtime controls.

## Runtime controls

| Control | Typical value | Meaning |
| --- | ---: | --- |
| `speed` | `1.0` | Speaking-rate multiplier. The validated range is 0.5-2.0. |
| `variation` | `0.667` | Stochastic delivery strength from 0.0-1.0. |
| `seed` | `7` | Reproduces the same stochastic generation. |
| `device` | `cpu` | `cpu` or `cuda`. |

Long input is split at punctuation-aware boundaries, generated one chunk at a
time, and joined with bounded pauses and edge fades. This limits memory use,
but it is not acoustic streaming and does not provide measured
time-to-first-audio.

## Measured CPU profile

On the managed Hugging Face CPU Upgrade reference (8 vCPU, 32 GB RAM, four
framework threads), steady-state end-to-end throughput was:

| Model | Median RTF | Throughput |
| --- | ---: | ---: |
| Inflect-Micro-v2 | 0.1593 | 6.28x real time |
| Inflect-Nano-v2 | 0.0933 | 10.72x real time |

The protocol used 100 fixed Modern400 prompts and three passes. Pass 1 built
caches and was excluded; passes 2 and 3 formed the steady-state pool. These
numbers include text processing and waveform synthesis. Results will vary with
hardware, thread settings, text length, and software versions.

## Export formats

The validated release is PyTorch FP32. ONNX, FP16, integer-quantized, Core ML,
TFLite, and GGUF packages are not currently claimed. A smaller file is not a
valid release unless matched listening and intelligibility checks show that it
preserves the model.
