---
title: Inflect v2 — Tiny Local TTS
emoji: 🔊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.11.0
python_version: 3.12
app_file: app.py
pinned: false
license: apache-2.0
short_description: Run complete 3.96M and 9.36M text-to-waveform models live.
models:
  - owensong/Inflect-Micro-v2
  - owensong/Inflect-Nano-v2
tags:
  - text-to-speech
  - local-tts
  - edge-ai
---

# Inflect v2 — Tiny Local TTS

Live text-to-waveform inference for both Inflect v2 release models:

- [Inflect-Micro-v2](https://huggingface.co/owensong/Inflect-Micro-v2): 9.36M parameters
- [Inflect-Nano-v2](https://huggingface.co/owensong/Inflect-Nano-v2): 3.96M parameters

Every result is synthesized live from text. There is no reference audio,
prerecorded fallback, or inference-time teacher model. Use the Compare tab to
run the same text, speed, variation, and seed through both checkpoints.

Long input is split automatically at sentence and punctuation boundaries, then
assembled into one downloadable WAV. ZeroGPU still applies a per-generation
time limit, so submit book-length or document-length text in smaller sections.
