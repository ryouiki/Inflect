# Examples

These examples use the exact self-contained packages published on Hugging
Face. Model weights are downloaded into the Hugging Face cache rather than
committed to GitHub.

## One model

```bash
python -m pip install -r requirements.txt
python examples/download_and_speak.py \
  --model micro \
  --text "A complete local voice can fit almost anywhere." \
  --output inflect.wav
```

Options:

```text
--model micro|nano
--device cpu|cuda
--speed 1.0
--variation 0.667
--seed 7
```

The model repository is cached after the first download. Long text is handled
by the punctuation-aware runtime included with the selected package.

## Direct comparison

```bash
python examples/compare_models.py \
  --text "Render both models with exactly the same controls." \
  --output-dir comparison
```

This writes:

```text
comparison/inflect-micro-v2.wav
comparison/inflect-nano-v2.wav
```

The comparison runs each self-contained model in its own Python process. This
avoids module-name collisions between two model packages while preserving the
same text, seed, speed, and variation.
