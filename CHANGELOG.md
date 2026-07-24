# Changelog

## 2.0.0 - 2026-07-24

### Added

- Inflect-Micro-v2, a complete 9.36M-parameter English text-to-waveform model.
- Inflect-Nano-v2, a complete 3.97M-parameter English text-to-waveform model.
- Shared self-contained Python API and CLI for CPU and CUDA.
- Integrated 24 kHz waveform generation with no external vocoder.
- Deterministic seeds, speaking speed, and delivery-variation controls.
- Punctuation-aware long-text chunking.
- Modern400 multi-ASR evaluation, UTMOS22 reports, blind preference evidence,
  managed CPU measurements, and integrity manifests.
- GitHub download-and-synthesize and matched-comparison examples.

### Release boundary

- One fixed English male voice per model.
- PyTorch FP32 is the validated format.
- Voice cloning, selectable speakers, multilingual synthesis, acoustic
  streaming, and quantized/mobile exports are not claimed.

## 1.x

Inflect Nano v1 remains available as a legacy release for reproducibility.
Inflect v2 is recommended for new integrations.
