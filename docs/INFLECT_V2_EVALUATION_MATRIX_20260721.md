# Inflect v2 Evaluation Matrix

This document defines the frozen evidence used by the Inflect v2 model cards.
Metrics remain separate; no aggregate score is used to hide a weak dimension.

## Frozen Evidence

| Dimension | Protocol | Release use |
| --- | --- | --- |
| Intelligibility | 400 matched unseen prompts; Qwen3-ASR, Nemotron 3.5, Whisper large-v3 | Two-ASR consensus headline plus complete three-ASR audit |
| Predicted quality | UTMOS22 on 500 matched unseen outputs per voice; 10,000 bootstrap samples | Diagnostic only, not human MOS |
| Human preference | Anonymous randomized pairwise choices; ties count as half a win | Descriptive community evidence |
| Runtime | Ryzen 9 3900X, 12 configured threads, 48 matched prompts, 3 warmups, isolated processes | Named-host engineering comparison |
| Footprint | Exact deployable FP32 weights | Complete model size, waveform decoder included |
| Integrity | Per-file SHA-256 manifest plus clean package validation | Release gate |

Modern400 combines 200 fixed modern/stress prompts with 200 deterministic
FLEURS `en_us` test prompts. Exact-text exclusion was checked against 87,362
training transcripts. All ASR inputs are resampled to 16 kHz and scored with
the same disclosed English normalizer.

## Headline Results

| Model | Community preference | UTMOS22 | Qwen3-ASR WER | Nemotron WER | Two-ASR mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Inflect-Micro-v2 | 66.2% | 4.395 | 2.52% | 5.45% | 3.99% |
| Inflect-Nano-v2 | 63.9% | 4.386 | 2.79% | 5.63% | 4.21% |

Whisper large-v3 scores are retained in the raw audit. It is excluded
consistently from the headline because insertion-heavy failures on a subset of
otherwise intelligible Supertonic 8-step clips made its system comparison
unreliable. This decision was not made selectively for Inflect.

## Matched CPU Runtime

| System | Throughput | RTF | p95 latency |
| --- | ---: | ---: | ---: |
| Inflect-Micro-v2 | 1.58x | 0.635 | 7.281 s |
| Inflect-Nano-v2 | 1.59x | 0.630 | 7.207 s |
| KittenTTS Nano, two-voice mean | 7.85x | 0.127 | 1.074 s |
| Piper Low, two-voice mean | 18.58x | 0.054 | 0.364 s |
| Supertonic 3, James, 3-step | 5.51x | 0.181 | 1.494 s |
| Supertonic 3, James, 8-step | 2.51x | 0.398 | 3.181 s |

Runtime includes text processing and waveform generation. Inflect uses its
released PyTorch path; competitors use their published ONNX Runtime paths.
The comparison measures packages on one host, not architecture in isolation.

## Interpretation Rules

- UTMOS22 is a learned quality predictor, not formal MOS.
- ASR WER measures transcript recoverability, not naturalness.
- Community preference is normalized by appearances and is not a population estimate.
- Voice variants sharing weights are averaged for family-level competitor plots.
- Confidence intervals and per-system rows remain in the model packages.
- A chart must link back to the exact raw report from which it was rendered.

## Not Claimed

TTSDS2, SpeechBERTScore, formal MUSHRA/MOS, streaming TTFA, multilingual
evaluation, female-voice evaluation, and quantized-runtime evaluation are not
part of v2.0.0. They must not be implied by launch copy.

