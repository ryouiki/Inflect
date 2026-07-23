# Publishing Inflect v2

Inflect v2 is an open-weight release split across GitHub and Hugging Face. Keep
the deployable product public while retaining private corpus construction,
reference material, filtering infrastructure, and the full optimization recipe.

## Release Surfaces

| Surface | Contents |
| --- | --- |
| GitHub | Public source, evaluation/release tooling, architecture notes, documentation |
| Hugging Face models | Frozen weights, self-contained inference, samples, raw evaluation, integrity manifest |
| Hugging Face Space | Private preflight or public interactive inference after owner approval |
| Local only | Raw checkpoints, generated corpora, private references, credentials, rented-host state |

## Frozen v2 Artifacts

- [Inflect-Micro-v2](https://huggingface.co/owensong/Inflect-Micro-v2):
  9,356,513 complete inference parameters, 37.53 MB FP32 weights.
- [Inflect-Nano-v2](https://huggingface.co/owensong/Inflect-Nano-v2):
  3,966,721 complete inference parameters, 15.97 MB FP32 weights.
- [Inflect v2 playground](https://huggingface.co/spaces/owensong/Inflect-v2):
  exact frozen Micro and Nano checkpoints.

Both model packages include their integrated 24 kHz waveform decoder. They do
not download an external vocoder or inference-time teacher.

## Visibility Rule

Preparing and uploading a private release candidate is allowed. Changing a
model, Space, collection, or repository from private to public is a separate
owner action and must never be bundled into routine release preparation.

## Required Gates

1. Freeze exact checkpoints and record full SHA-256 hashes.
2. Validate inference from a clean downloaded package.
3. Verify the Python API, CLI, deterministic seed, long-text path, and WAV output.
4. Render fixed held-out samples with exact transcripts.
5. Publish matched intelligibility, predicted-quality, human-preference, footprint,
   and named-hardware runtime evidence.
6. Check every chart against its raw report and document exclusions.
7. Confirm licensing, third-party notices, limitations, and responsible-use text.
8. Upload privately, download again, and repeat smoke tests.
9. Only then make visibility changes and publish launch posts.

## Supported Release Formats

The v2.0.0 package supports PyTorch FP32 inference on CPU and CUDA. FP16/BF16,
ONNX, integer quantization, Core ML, TFLite, and GGUF are not release formats
until they pass the same intelligibility and listening gates as FP32. GGUF is
not a natural container for this convolution-heavy VITS-family waveform model.

## Do Not Publish

- credentials, tokens, SSH material, or rented-instance connection details;
- private reference voices or source-speaker material;
- raw generated corpora or unfinished checkpoints;
- claims of female voices, voice cloning, multilingual support, streaming, or
  validated quantization;
- teacher or corpus-generation internals outside the documented open-weight scope.

## Release Order

1. Finish and validate the private Hugging Face packages.
2. Tag the exact model commits as `v2.0.0`.
3. Push the reviewed GitHub release branch.
4. Verify model cards, audio, charts, and the Space in the Hub UI.
5. Owner changes model and Space visibility.
6. Publish the release and announcement copy.
