# Inflect v2 Release Notes

Inflect v2 introduces two complete compact English text-to-speech systems:

- **Inflect-Micro-v2:** 9,356,513 parameters and 37.53 MB FP32 weights.
- **Inflect-Nano-v2:** 3,966,721 parameters and 15.97 MB FP32 weights.

Both checkpoints include the full 24 kHz text-to-waveform path and integrated
waveform decoder.

## Included

- one fixed male English voice;
- self-contained PyTorch FP32 inference;
- shared Python API and CLI;
- deterministic seed, delivery variation, and speaking-speed controls;
- punctuation-aware long-text chunking;
- held-out samples with exact transcripts;
- Modern400 ASR, UTMOS22, blind-preference, footprint, and runtime evidence;
- SHA-256 integrity manifest, Apache-2.0 license, and third-party notices.

## Frozen results

| Model | Community preference | UTMOS22 | Semantic WER | Managed CPU throughput |
| --- | ---: | ---: | ---: | ---: |
| Inflect-Micro-v2 | 66.2% | 4.395 | 3.99% | 6.28x real time |
| Inflect-Nano-v2 | 63.9% | 4.386 | 4.21% | 10.72x real time |

The CPU result uses 100 fixed prompts on the managed 8-vCPU reference, four
framework threads, one excluded cache-building pass, and two pooled
steady-state passes. UTMOS22 is predicted quality, not human MOS. Community
preference is descriptive pairwise evidence, not a formal population estimate.

## Limitations

- English only and one fixed voice.
- No voice cloning, selectable speakers, female voices, or multilingual speech.
- No acoustic streaming.
- Uncommon names, numbers, abbreviations, and distribution-shifted text can be weaker.
- Long passages are chunked and can expose boundary differences.
- No validated ONNX, reduced-precision, integer-quantized, mobile, or GGUF release.
