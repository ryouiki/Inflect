# Inflect adaptation toolkit contract

This directory is a public, generic warm-start adaptation toolkit. It is not
the private recipe used to produce the Inflect v2 release checkpoints.

## Supported result

The toolkit produces a separate, fixed-voice checkpoint for one configured
language. The dataset speaker becomes the checkpoint voice. It does not add
runtime-selectable speakers or languages to the original checkpoint.

## Public workflow

```text
python -m inflect_finetune prepare ...
python -m inflect_finetune audit ...
python -m inflect_finetune train ...
python -m inflect_finetune export ...
```

Every command must support `--help`, fail with an actionable message, and write
machine-readable reports alongside human-readable summaries.

## Prepared dataset layout

```text
prepared/
  dataset.json
  symbols.json
  train.jsonl
  validation.jsonl
  audio/
```

Each JSONL row contains:

```json
{
  "audio": "audio/example.wav",
  "text": "Original transcript.",
  "normalized_text": "Normalized transcript.",
  "phonemes": "phoneme string",
  "duration_seconds": 2.34
}
```

Prepared rows may also preserve `id`, `speaker`, `group_id`, and
`group_field`. Source manifests may provide `session` as a split boundary and
`phonemes` for prephonemized preparation.

`dataset.json` records the language, sample rate, frontend, source-manifest
hash, frontend source/metadata hashes where applicable, the bundled frontend
name where one was used, speaker, split seed, row counts, and aggregate
diagnostics. `symbols.json` records the ordered
symbol inventory and its relationship to the base inventory.

## Checkpoint migration

Checkpoint migration copies all shape-compatible generator weights. Text
embedding rows are copied by symbol string, not by numeric index. Newly added
symbols receive deterministic initialization. A base checkpoint may carry a
larger inventory than the release as long as the released symbols keep their
released positions; rows it holds that the new dataset does not use are dropped
and reported. Training-only discriminators,
optimizers, and schedulers are initialized from public generic defaults.

## Safety boundaries

Public code must not contain or infer:

- private corpus paths, transcripts, or generated audio;
- teacher-model names or corpus-generation prompts;
- internal filtering thresholds unrelated to generic audio validity;
- the original base-model curriculum, search history, or checkpoint ranking;
- credentials, rental identifiers, or private storage locations.

## Validation gates

At minimum, automated tests must cover:

- manifest parsing and path traversal rejection;
- deterministic splitting;
- group, duplicate-audio, and duplicate-transcript leakage prevention;
- audio-format validation;
- language/frontend failures;
- phoneme coverage and unknown-symbol reporting;
- embedding migration by symbol identity;
- checkpoint save/resume behavior;
- inference-only export loadability;
- language-aware frontend packaging and refusal of silent English fallback;
- ONNX Runtime parity when ONNX export is requested.

A release may call language adaptation experimental until at least one
non-English end-to-end run passes preparation, training, export, and inference.
