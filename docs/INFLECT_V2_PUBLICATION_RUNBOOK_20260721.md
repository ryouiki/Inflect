# Inflect v2 Publication Runbook

Target release: 2026-07-24.

## Frozen Artifacts

| Artifact | Revision |
| --- | --- |
| Inflect-Micro-v2 model SHA-256 | `3eede065c9ccfa88ade0a5a9a5c23de34afcbbb32213e59aad44d5cf100fdee8` |
| Inflect-Micro-v2 Hub commit/tag | `d0385b51a2232abbd52602a92e0496a0df2bca25` / `v2.0.0` |
| Inflect-Nano-v2 model SHA-256 | `bfca468489c9069361d6b87b295a17eef611af8ff09854ce0b80305d9122f5b3` |
| Inflect-Nano-v2 Hub commit/tag | `752c3f31a1140cd0fa4c1a6d62d346496cfd2a6d` / `v2.0.0` |
| Private Space commit | `44574cfbd452645f2b5894759bfdcc9793dbe485` |
| GitHub release branch | `codex/marketing-readme-renovation` |

The private model packages passed their 88-file manifest validators, were
downloaded back from Hugging Face, and synthesized valid 24 kHz output. The
private Space `/compare` endpoint returned valid Micro and Nano WAVs.

## Owner Gate

Before changing visibility:

1. Listen to conversational, punctuation, numbers, uncommon names, technical,
   and one long-form prompt in the private Space.
2. Check both model cards in desktop and mobile Hub layouts.
3. Play all five embedded samples on each card.
4. Confirm the GitHub release PR contains no private data or credentials.
5. Merge the release PR.

## Publication Order

1. Change both model repositories to public in the same release window.
2. Verify signed-out downloads of `config.json`, `model.pth`, and one sample.
3. Run one anonymous clean CPU synthesis for each model.
4. Change the Space to public.
5. Test Micro, Nano, comparison, pitch, speed, variation, and long text.
6. Verify links from both model cards, GitHub, collection, and Space.
7. Publish launch posts only after all signed-out checks pass.

Visibility is the only intended model/Space change during the owner gate.
Do not replace the frozen checkpoints without rerunning package and evaluation
validation.

## Supported Claims

- Complete 24 kHz text-to-waveform models at 3.96M and 9.36M parameters.
- Integrated waveform decoder with no external vocoder download.
- One fixed English male voice per model.
- Local PyTorch FP32 CPU and CUDA inference.
- Frozen human-preference, UTMOS22, multi-ASR, footprint, and CPU evidence.
- Punctuation-aware chunked long-text synthesis.

## Unsupported Claims

Do not claim female voices, selectable speakers, voice cloning, multilingual
synthesis, streaming, Raspberry Pi/ESP32 speed, validated ONNX or quantized
formats, universal superiority, or safety-critical use.

## Rollback

If a public package fails, make the affected surface private, preserve the
failing commit and report, fix on a new commit, rerun clean-download validation,
and only then republish. Never overwrite evidence or silently swap weights.

