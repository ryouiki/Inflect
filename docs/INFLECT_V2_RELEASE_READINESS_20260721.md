# Inflect v2 Release Readiness

Target release: 2026-07-24.

## Frozen Models

| Model | Checkpoint | Complete parameters | FP32 bytes | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Inflect-Micro-v2 | 186,000 | 9,356,513 | 37,529,995 | `3eede065c9ccfa88ade0a5a9a5c23de34afcbbb32213e59aad44d5cf100fdee8` |
| Inflect-Nano-v2 | 148,000 | 3,966,721 | 15,971,083 | `bfca468489c9069361d6b87b295a17eef611af8ff09854ce0b80305d9122f5b3` |

Both totals cover the complete text-to-waveform inference model, including the
integrated 24 kHz waveform decoder.

## Public Contract

- PyTorch FP32 CPU and CUDA inference.
- One fixed English male voice per model.
- Python API and CLI with identical public controls.
- Deterministic seeds, speaking speed, and delivery variation.
- Punctuation-aware long-text chunking.
- Five held-out samples with exact transcripts.
- Frozen evaluation prompts, reports, figures, and integrity manifest.

## Evidence

| Model | Community preference | UTMOS22 | Two-ASR semantic WER | Ryzen 3900X throughput |
| --- | ---: | ---: | ---: | ---: |
| Inflect-Micro-v2 | 66.2% | 4.395 | 3.99% | 1.58x real time |
| Inflect-Nano-v2 | 63.9% | 4.386 | 4.21% | 1.59x real time |

See [the evaluation matrix](INFLECT_V2_EVALUATION_MATRIX_20260721.md) for
protocol boundaries and competitor results.

## Known Weaknesses

- Numbers, abbreviations, uncommon names, and context-sensitive homographs.
- Flatter or less stable delivery on distribution-shifted phrasing.
- Audible transitions can occur between long-text chunks.
- Stochastic settings can alter timing and pronunciation.
- CPU speed is real time on the measured host but trails optimized ONNX competitors.

## Explicitly Unsupported

- female voices, selectable speakers, or zero-shot voice cloning;
- multilingual synthesis;
- streaming or measured time-to-first-audio;
- validated ONNX, FP16/BF16, integer quantization, Core ML, TFLite, or GGUF;
- medical, legal, emergency, or accessibility-critical use.

## Final Gate

The release is ready only when both private Hub repositories pass a clean
download test, file hashes match, the Python API and CLI synthesize valid
24 kHz WAV output, the Space generates with both frozen revisions, all relative
model-card links resolve, and visibility remains private until the owner
explicitly publishes.

