# Manifests and prepared datasets

## Source formats

UTF-8 JSON Lines and CSV are supported. Every row requires:

| Field | Type | Meaning |
| --- | --- | --- |
| `audio` | string | Relative path below `--audio-root` |
| `text` | string | Exact transcript of the recording |

Supported optional fields:

| Field | Meaning |
| --- | --- |
| `id` | Stable user identifier preserved in prepared metadata |
| `speaker` | Consistency metadata; all nonempty values must identify one speaker |
| `group_id` | Strongest split boundary, such as source recording or chapter |
| `session` | Split boundary used when `group_id` is absent |
| `phonemes` | Required on every row with `--frontend prephonemized` |

JSONL example:

```json
{"audio":"session01/000001.wav","text":"Buenos dias.","speaker":"voice-a","session":"session01"}
{"audio":"session02/000002.wav","text":"El tren llega a las nueve.","speaker":"voice-a","session":"session02"}
```

CSV example:

```csv
audio,text,speaker,session
session01/000001.wav,Buenos dias.,voice-a,session01
session02/000002.wav,El tren llega a las nueve.,voice-a,session02
```

## Path and content rules

- Paths must stay below the declared audio root.
- Absolute paths and `..` traversal are rejected.
- Missing, duplicate, undecodable, empty, or non-finite audio is rejected.
- Audio is converted deterministically to 24 kHz mono WAV.
- Reused audio content is rejected, even if it appears under another filename.
- Transcripts should describe exactly what is spoken.
- Empty transcripts and unsupported frontend output are rejected.

Do not place comments, timestamps, markup, or speaker directions in `text`
unless the chosen frontend intentionally handles them.

## Leakage-safe splitting

Preparation creates deterministic, nonempty train and validation splits from
the recorded split seed.

Rows connected by the following relationships stay in the same split:

- identical normalized transcript;
- the same `group_id`;
- the same `session` when no `group_id` is present.

`speaker` is not a split group. A fixed-voice corpus normally has the same
speaker on every row, so treating it as a group would make a validation split
impossible. Multiple nonempty speaker values are rejected instead.

Use `group_id` for the strongest leakage boundary. For example, clips cut from
one long recording should share a `group_id`, even if they have different row
IDs.

## Prepared layout

`prepare` writes:

```text
prepared/
  dataset.json
  symbols.json
  train.jsonl
  validation.jsonl
  audio/
  preparation_report.json
```

Prepared rows contain the converted audio path, original and normalized text,
phoneme string, duration, split metadata, and any supported source identifiers.
`dataset.json` records the language, frontend identity, hashes, speaker,
split configuration, counts, and diagnostics. `symbols.json` contains the
ordered inventory used to migrate embedding rows by symbol identity.

Prepared data is immutable input. Correct the source data or frontend and run
`prepare` again instead of editing prepared files by hand.
