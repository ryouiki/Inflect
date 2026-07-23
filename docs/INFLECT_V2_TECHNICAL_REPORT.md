# Inflect v2 Technical Report

## Abstract

Inflect v2 is a pair of complete, fixed-voice English text-to-speech models
designed for local deployment. Inflect-Nano-v2 contains 3,966,721 inference
parameters and Inflect-Micro-v2 contains 9,356,513. Both map normalized text to
a 24 kHz waveform without an external vocoder, server model, or inference-time
teacher. This report describes the deployable system, frozen evaluation
protocol, measured results, and release limitations. It intentionally excludes
private corpus-generation and full optimization infrastructure.

## 1. Motivation

Many compact TTS comparisons omit the waveform decoder or rely on a hosted
component. Inflect reports complete inference parameters and weight size for
the entire text-to-waveform path. The design target is practical offline speech
on constrained hardware, not general-purpose voice cloning or multilingual
synthesis.

## 2. System

Inflect v2 is a compact VITS-family end-to-end generator. Its deployed path
contains an English normalization and phoneme frontend, transformer text
encoder, stochastic duration predictor, monotonic alignment machinery, latent
speech generator, residual coupling flows, and an integrated alias-reduced
neural waveform decoder. Micro allocates more capacity to hidden and decoder
width; Nano preserves the same public API with a narrower network.

| Model | Complete parameters | FP32 weights | Output |
| --- | ---: | ---: | --- |
| Inflect-Nano-v2 | 3,966,721 | 15.97 MB | 24 kHz mono |
| Inflect-Micro-v2 | 9,356,513 | 37.53 MB | 24 kHz mono |

Long input is split at punctuation-aware boundaries, generated per chunk, and
joined with bounded pauses and edge fades. This controls memory but is not
equivalent to one globally planned long-form generation.

## 3. Evaluation

The release separates five questions:

1. Human preference: anonymous randomized pairwise choices.
2. Predicted naturalness: UTMOS22 on 500 matched unseen prompts per voice.
3. Intelligibility: 400 matched unseen prompts decoded by Qwen3-ASR,
   Nemotron 3.5, and Whisper large-v3.
4. Deployment footprint: complete FP32 inference weights.
5. Runtime: 48 matched prompts on an AMD Ryzen 9 3900X, 12 configured CPU
   threads, three warmups, one isolated process, and no visible GPU.

Headline semantic WER is the equal-weight corpus-level mean of Qwen3-ASR and
Nemotron 3.5. Whisper remains in the raw audit but is excluded consistently
from the headline after insertion-heavy failures on a subset of Supertonic
8-step clips. UTMOS22 is a learned predictor, not human MOS. Community
preference is descriptive evidence, not a formal listening-panel estimate.

## 4. Results

| Model | Community preference | UTMOS22 | Two-ASR semantic WER | CPU throughput |
| --- | ---: | ---: | ---: | ---: |
| Inflect-Micro-v2 | 66.2% | 4.395 | 3.99% | 1.58x real time |
| Inflect-Nano-v2 | 63.9% | 4.386 | 4.21% | 1.59x real time |

The community rates are normalized from appearances in a multi-system blind
study, with ties counting as half a win. Raw reports, hypotheses, prompt
manifests, intervals, runtime rows, and integrity hashes ship with each model
package under `evaluation/final/`.

## 5. Limitations

The release is English-only and contains one fixed male voice. It does not
support zero-shot cloning, selectable speakers, female voices, multilingual
synthesis, or streaming. Uncommon names, abbreviations, homographs, numbers,
and distribution-shifted phrasing can be flatter or less stable. Long-form
chunk boundaries can be audible. The package is not validated for medical,
legal, emergency, or accessibility-critical communication.

Only PyTorch FP32 is a validated release format. ONNX, reduced precision,
integer quantization, Core ML, TFLite, and GGUF are not claimed.

## 6. Reproducibility and Scope

Inflect v2 is open-weight. It publishes deployable weights, inference and
frontend code, samples, evaluation prompts, benchmark outputs, and file hashes.
The private corpus-generation pipeline, source reference material, filtering
infrastructure, and full optimization recipe are outside the release. This
boundary should be stated explicitly in any derived report or model card.

## Citation

```bibtex
@software{song2026inflectv2,
  author = {Owen Song},
  title = {Inflect v2: Complete Local Text-to-Waveform TTS at 3.97M and 9.36M Parameters},
  year = {2026},
  url = {https://github.com/owenawsong/Inflect}
}
```
