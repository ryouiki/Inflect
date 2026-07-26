# Custom G2P and frontend hooks

Use a custom frontend when eSpeak does not provide acceptable normalization or
phonemization. A frontend controls normalization, punctuation, word
boundaries, and the exact symbol stream consumed by the generator.

## Hook contract

Pass a trusted factory as either `module:callable` or `file.py:function`:

```bash
python -m inflect_finetune prepare \
  --manifest data/metadata.jsonl \
  --audio-root data/audio \
  --language my-language \
  --frontend custom \
  --frontend-hook ./my_frontend.py:create_frontend \
  --output prepared/my-language
```

The factory may accept no arguments or the keyword argument `language`. It
must return an object with:

```python
normalize(text: str) -> str
phonemize(normalized_text: str) -> str
symbols() -> list[str]
metadata() -> dict
```

`symbols()` must return a nonempty ordered list of unique, one-character
strings. `metadata()` must contain `name`, `version`, `language`, and
`configuration`.

The hook is trusted Python code and executes during preparation. Do not run a
hook from an untrusted source.

## Reproducibility checks

Preparation calls normalization and phonemization twice and rejects
nondeterministic output. It also rejects empty text, control characters, and
symbols not declared by the hook.

The prepared dataset stores hashes of the hook source and declared metadata.
Export requires the matching hook source for custom frontends, verifies those
hashes, and copies the hook into the deployment package. It will not silently
replace a custom or non-English frontend with the release English frontend.

## Frontend validation

Before training, test:

- ordinary sentences and every target phoneme;
- punctuation and sentence boundaries;
- numbers, dates, currencies, and abbreviations;
- names, loanwords, and mixed scripts;
- unsupported and empty input;
- repeated calls for exact determinism.

Have fluent speakers inspect both normalized text and phonemes. A technically
valid symbol stream can still encode the wrong pronunciation.

## Deployment requirements

An adapted package must include the exact frontend needed for inference.
Third-party dictionaries or models remain subject to their own licenses and
must be packaged or documented separately. Do not describe an export as
self-contained if its frontend requires an external artifact.
