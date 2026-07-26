# Troubleshooting

Start with the first failing stage. Do not work around preparation or audit
errors by editing prepared files.

## Command or module not found

Install from the `finetune` directory in an activated environment:

```bash
python -m pip install -e .
python -m inflect_finetune --help
```

If the editable install is not wanted, build and install the wheel instead.

## Manifest rejected

Check that:

- the file is UTF-8 JSONL or CSV;
- every non-empty JSONL line is one JSON object;
- `audio` and `text` are strings;
- audio paths are relative to `--audio-root`;
- paths do not contain traversal outside the root;
- referenced files exist;
- JSON quoting and escaping are valid.

## Audio cannot be decoded

Preserve the original file, decode it with a trusted audio tool, and convert a
copy to a standard uncompressed WAV. Confirm sample rate, channels, duration,
and finite samples. Do not rename a compressed file to `.wav`.

## eSpeak or phonemizer failure

Confirm the installed eSpeak NG library is available, the language code exists,
and `phonemizer` can process a short sentence independently. Record package and
eSpeak versions when reporting the problem.

If only specific text fails, reduce it to the smallest reproducible input and
inspect unsupported symbols, language mixing, and normalization.

## Unknown symbols

Do not delete unknown symbols from prepared text. Determine whether:

- the frontend emitted an undeclared symbol;
- Unicode normalization produced a different code point;
- the base inventory lacks a valid target-language phoneme;
- punctuation or a word-boundary marker was omitted from the inventory.

Correct the frontend or symbol declaration, rerun `prepare`, and audit again.
New valid symbols should be initialized and learned; they should not be mapped
silently to unrelated base symbols.

## Out of GPU memory

Use a smaller validated memory preset, reduce the configured batch workload,
shorten training segments only if the trainer supports that change, or increase
gradient accumulation. Restart after clearing the failed process.

Do not compare runs as equivalent if batch semantics, segment length, precision,
or optimizer behavior changed.

## Resume rejected

Compare the recorded toolkit version, base checkpoint, dataset hash, symbols
hash, frontend metadata, model configuration, and optimizer schema. Resume
rejection usually means the run inputs changed.

Start a new run rather than forcing incompatible state to load.

## Training loss improves but audio worsens

Stop and inspect held-out audio. Common failures include decoder buzz, metallic
resonance, excessive sibilance, clipped endings, duration collapse, speaker
drift, and overfitting.

Select checkpoints using matched held-out listening and diagnostics, not loss
alone. More steps can make adaptation worse.

## Output is intelligible but pronunciation is wrong

Verify the source transcript, normalized text, phonemes, symbol coverage, and
training examples for the affected sound. If the frontend is wrong, fix it and
re-prepare the corpus. Training longer will not reliably correct a systematically
wrong phoneme sequence.

## Adapted voice sounds unlike the speaker

Check speaker consistency, recording conditions, data quantity and coverage,
and whether the run changed language and voice simultaneously. Speaker
similarity is not guaranteed by the fixed-voice architecture.

## ONNX model fails to parse or load

Confirm the file is fully downloaded and its checksum matches. Test with the
documented ONNX Runtime and ONNX versions. Re-export from the same inference
checkpoint and validate every graph before upload.

An ONNX file existing on disk is not evidence that export succeeded.

## Reporting an issue

Include:

- exact command;
- toolkit commit or version;
- operating system, Python, PyTorch, CUDA, eSpeak, and ONNX Runtime versions;
- base model and checksum;
- redacted manifest schema and failing row;
- prepared dataset and symbols hashes;
- complete error text;
- minimal reproducible input.

Do not attach private speech data without the speaker's permission.
