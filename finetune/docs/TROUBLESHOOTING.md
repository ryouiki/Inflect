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

## Bundled frontend failure

A bundled frontend needs its extra installed, for example `pip install ".[ja]"`
for `ja-openjtalk`. It also requires the language it declares, so
`--frontend ja-openjtalk` must be paired with `--language ja`.

If preparation fails on specific rows, reduce the transcript to the smallest
failing text. A row that is only punctuation, brackets, or a bare long-vowel
mark has no reading and is rejected rather than silently dropped; fix the
transcript instead of the frontend.

Export recovers the bundled hook from the prepared dataset. If it still asks
for `--frontend-hook`, the dataset was prepared with `--frontend custom` and
your own file, or `--prepared-dataset` was not passed.

A hook source hash mismatch at export means the toolkit version changed after
preparation. Re-prepare with the current version rather than overriding it.

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
or optimizer behavior changed. The mel projection and the waveform transform now
run in full precision regardless of the surrounding cast, which fixed a
half-precision window that made the transform return complex32 and quantized the
mel target as well as the prediction. Mixed-precision loss values are therefore
not comparable across that change; the largest observed difference in a log-mel
value is 0.025, and runs with `--no-amp` are unaffected.

## Resume rejected

Compare the recorded toolkit version, base checkpoint, dataset hash, symbols
hash, frontend metadata, model configuration, and optimizer schema. Resume
rejection usually means the run inputs changed.

Note that this release adds option fields for its own reasons, so every
checkpoint written before it is unresumable even at their defaults.

`identity fields differ: ['options']` after an upgrade usually means the run
inputs did not change and the toolkit did. The public options are part of the
run identity, so a release that adds an option field makes every checkpoint
written before it unresumable, including at that field's default. This is the
guard doing its job. Finish the work in a new run, or export the interrupted
one and chain from it.

Start a new run rather than forcing incompatible state to load.

## Training loss improves but audio worsens

Stop and inspect held-out audio. Common failures include decoder buzz, metallic
resonance, excessive sibilance, clipped endings, duration collapse, speaker
drift, and overfitting. One of those has a mechanical explanation and automatic
detection; see the section on ringing below.

Select checkpoints using matched held-out listening and diagnostics, not loss
alone. More steps can make adaptation worse.

## A steady tone or hum behind the voice

The giveaway is that it does not stop when the speaking does. A comb of tones at
multiples of the sample rate over the frame hop sits under the whole render,
including the silence between words, and on headphones it localizes as a single
pitch behind the voice. For Micro at 24 kHz with hop 256 that is 93.75 Hz and
its multiples.

Confirm it from an evaluation report rather than by ear. `grid_tone_excess_db`
is zero by construction for real speech and measured 8.15 dB at the median on a
rejected run against -0.13 dB on the speaker's own recordings.
`steady_tone_artifact_score` separated the same two sets completely, 29.9
against 0.00. `clips_f0_locked_to_frame_grid` counts clips where the pitch
tracker reported the comb as the voice; one failing checkpoint scored 134 of
160 there. The training run also writes the two cheap screens into each
`validation/step-*.json`, so the arrival can be dated to within a validation
interval instead of discovered at the end.

The cause found by investigation was drift in the latents, not in the decoder
weights. The released checkpoint carries no posterior encoder, so a fresh one
starts every run, and while the decoder is frozen the adversarial gradients
still reach the posterior encoder through it with nothing anchoring where they
go. They do not reach the flow, which the KL term updates instead. Feeding the drifted latents to the released decoder
rang harder than feeding them to the adapted one. That was read at the time as
ruling the decoder out, which it does not: the reverse case, released latents
through the adapted decoder, was never run, so how the two interact is still
open. The upsampler has no anti-imaging filter, which is why this particular
grid is where the energy lands.

There is no known fix, and the search for one is mid-flight. The two training
controls that exist for this were tried on one corpus at 10,000 steps each and
did not work: gating the adversarial term left the artifact where it was, and a
reconstruction-only decoder polish made it substantially worse, collapsing the
median tracked pitch onto the comb frequency. But every one of those runs
trained against a mel loss that floored its two sides differently, so what
those numbers measure is those controls under a broken objective.

That defect is fixed, and the controlled comparison has been run. Against
signal power it looked like progress: the broadband floor fell 4.64 dB and the
comb 3.16 dB, on 40 and 37 of 40 sentences. Measured absolutely the same
sentences say something else. On-grid power rose 0.99 dB, with the fixed arm
lower on only 9 of 40, while total power rose 3.97 dB on 39 of 40 and the
renders came out shorter with less silence in them. The comb did not shrink;
the render got louder. Both arms then failed listening outright at 1,500 steps,
scored "not a human voice" on every row, so the round is a usability rejection
rather than a deferral. The mel fix stays in: it is a defect fix on its own
terms, and this experiment did not resolve the artifact by itself.

That episode is the reason this page keeps three readings of the comb. The
ratio, the level and the absolute power ranked the two arms three different
ways, and the listening test agreed with none of them about which was better.
Screens eliminate; the listener decides.

Crossing latents against decoders has since narrowed where to look. The same
twelve sentences were rendered through three decoders with the latents held
fixed. Latents from the released prior path came out clean through all three,
including two decoders that had trained for 7000 steps; latents from an
adapted run rang through all three, including the released decoder. That held
on 12 of 12 sentences in every decoder. So adapted decoder weights are not
necessary for the comb, and the latents are where it enters. That is not the
same as clearing the decoder: this architecture may still be what turns those
particular latents into a comb, and the released decoder has no anti-imaging
filter. Which property of the latents matters is not settled either: removing
their per-channel offset helps on one screen and hurts on the other, matching
their scale helps modestly, and smoothing them in time helps on neither.

Naming the screen matters when reporting that grid. Decoder training looks
progressively worse on the ratio, 6.33 then 8.01 then 9.89 dB across the
released, control and reconstruction-polish decoders, and progressively better
on the relative level over the same three, -44.48 then -46.47 then -47.66 dB.
Which of those is the worse sound has not been listened to.

The listening round that followed reframed the problem. At 1,500 steps neither
arm produced speech at all, only ringing at the tempo of the sentence, while a
10,000-step control was scored as awkward but recognisable speech. What is
failing is the formation of speech, not a residual noise on top of it.

Splitting the two paths on existing checkpoints then separated the two
problems outright. A 10,000-step run reconstructs and infers as recognisable
speech while carrying 8.1 dB of comb excess; the 1,500-step arms carry the
same 7.7 to 8.6 dB and are not speech at all. **How much comb a render has
does not predict whether speech formed in it.** Treat them as two problems.

The same split found a working path that later broke. At step 500, with the
prior, flow and decoder all still frozen, inference produced clean speech,
scored "speech but awkward" with no defect at all, in a voice the listener
described as an English-speaking man reading Korean text. A thousand steps of
linguistic adaptation later the same path is not speech and is the most
smeared track on the page. Both 1,500-step paths also hand their pitch to the
comb, tracked at 93.75 Hz on 34 and 39 of 40 sentences, while every render
scored as speech tracks near the speaker's own 360 Hz. Pitch lock is not
necessary for a render to fail, but nothing that locked has passed.

Four things were tried and measured and did not work, under that caveat.
Gating the generator's adversarial term while the decoder is frozen leaves the
artifact unchanged and raises the latent drift, because that term had been
pulling the latents back. A reconstruction-only decoder polish raises the
artifact while its own reported losses fall. Freezing the decoder for the whole
run does not prevent the comb either; an ablation that never unfroze it showed
the comb by step 1000 with 94 per cent of frames locked to the grid. Unfreezing
earlier was worse early and no better at the end.
Restoring the released decoder at export time makes it louder, not quieter.

The cause is therefore still open. The latent drift the investigation first
blamed does not track the artifact on its own: two runs ending at drift 1.190
and 1.528 produced 8.15 and 8.27 dB of comb, which rules out a single scalar
mean as the controlling variable and rules out nothing else about the latents.
Both the training path and the inference path ring, which shows a mismatch
between them is not the whole story rather than showing there is none. Treat
the screens as the reliable part and the explanations as provisional.

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
