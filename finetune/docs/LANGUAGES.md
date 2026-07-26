# Languages and frontend configuration

## One configured language per checkpoint

An adaptation run produces a checkpoint tied to the language and symbol
inventory recorded in its prepared dataset. It does not make the original
checkpoint multilingual, and it does not expose a runtime language selector.

For example, separate English and Spanish runs produce separate English and
Spanish checkpoints. Combining their files does not produce a bilingual model.

## eSpeak frontend

The built-in frontend uses `phonemizer` with eSpeak NG:

```bash
python -m inflect_finetune prepare \
  --manifest data/metadata.jsonl \
  --audio-root data/audio \
  --language es \
  --frontend espeak \
  --output prepared/es
```

Use an eSpeak language or voice code appropriate to the transcripts and
recordings. Language codes, stress behavior, punctuation preservation, and
normalization must be recorded in `dataset.json`.

Before training:

1. Confirm eSpeak supports the requested code in the installed version.
2. Inspect normalized text and phonemes for a representative sample.
3. Ask a fluent speaker to review names, loanwords, abbreviations, and numbers.
4. Run `audit` and resolve unknown symbols.
5. Verify validation phonemes are covered by training.

## Symbol inventory migration

The base checkpoint and prepared language may use different symbols. Migration
must:

- copy shape-compatible generator weights;
- match text embedding rows by symbol string, never numeric position;
- deterministically initialize newly added symbol rows;
- record copied, new, and unused symbols;
- reject ambiguous or duplicate symbol definitions;
- preserve the prepared symbol order in the exported package.

New symbol initialization makes training possible; it does not provide a
pronunciation. The corpus must teach the acoustic realization and timing.

## Normalization is language-specific

Numbers, currencies, dates, abbreviations, casing, punctuation, and symbols
cannot be normalized reliably with one universal rule set. Users must verify
that normalized text matches what the speaker says.

Do not reuse English-specific normalization for a different language without
review. If the built-in frontend cannot represent the intended reading, use a
custom frontend rather than editing prepared phonemes by hand.

## Quality expectations

Availability of an eSpeak voice is not evidence that Inflect will train well on
that language. Languages with substantially different phonology, writing
systems, timing, or prosody may require more data, frontend work, model
capacity, and optimization changes.

Nano has less capacity and less tolerance for poor coverage than Micro. Use
Micro for the first adaptation attempt unless footprint is the primary
constraint and the Micro workflow has already been validated.

Every release should state:

- language and regional variety;
- frontend name and version;
- symbol inventory;
- speaker and corpus provenance;
- training and validation quantities;
- known pronunciation limitations;
- whether fluent speakers evaluated held-out output.
