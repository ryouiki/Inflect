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

## Preparation refused: conversion clipped

`prepare` stops without writing a dataset when conversion (the input gain, then
resampling) pushed any row past full scale. Read `<output>.clipping-report.json` beside the requested output
directory: it lists the clipped rows, the largest pre-clip peak, and a
recommended `--input-gain-db`. Prepare again with that gain; the same output
directory may be reused. One gain applies to every row, so the level
relationship between rows is kept. A nonzero gain is a different dataset for run
identity and draws a different validation split. See
[data quality](DATA_QUALITY.md).

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
multiples of the sample rate over the frame hop sits under the render, and on
headphones it localizes as a single pitch behind the voice. For Micro at 24 kHz
with hop 256 that is 93.75 Hz and its multiples (187.5 Hz, 281.25 Hz and so
on). The decoder's upsampler has no anti-imaging filter, which is why this is
the grid where the energy lands when the comb appears.

Confirm the tone's presence from an evaluation report rather than by ear, and
leave the verdict to the listener. Screens eliminate; they do not decide.

### The screens

- `grid_tone_excess_db`: the mean power per bin (Welch, 8192-sample window) in
  bins within 4.4 Hz of any grid multiple, over the mean power per bin in the
  rest of the band from 2 kHz to about 11.9 kHz, in dB. It needs at least 8192
  samples (about 341 ms at 24 kHz); a shorter clip reports nothing. It is
  positive when a comb is present, but it is not zero by construction for real
  speech, so read it against the same sentence's recording.
- `grid_tone_level_db` and `off_grid_level_db`: the same two band powers against
  the clip's own signal power. Compare these across renders. Their absolute
  offset depends on the window settings.
- `fold_periodic_excess_db`: the clip folded into hop-length frames, relative to
  the floor an uncorrelated signal would give. It corroborates the others
  rather than accusing on its own.
- `steady_tone_artifact_score`: summed prominence of spectral peaks that are
  steady across frames and above 1200 Hz. Real recordings score low because
  their prominent lines move with the voice. It costs more per clip, so the
  training-time validation leaves it out; the offline evaluation reports it by
  default (`--no-steady-tone-screen` turns it off).
- `f0_grid_deviation_hz` and `clips_f0_locked_to_frame_grid`: a pitch tracker
  fed a ringing render reports the comb as the voice. A clip counts as locked
  when its median tracked pitch lies within 1.5 Hz of 93.75, 187.5 or 281.25 Hz
  (`f0_grid_deviation_hz(max_multiple=3)`). The deviation is reported next to
  the excess because a pitch that happens to sit on the grid raises the excess
  legitimately.
- The inter-word ratio: the same grid power ratio measured on windows with no
  active frame, to ask whether the comb continues between words.

The training run writes the cheap screens (`grid_tone_excess_db` and
`fold_periodic_excess_db`, with the level readings) into each
`validation/step-*.json`, so the arrival of a comb can be dated to within a
validation interval instead of discovered at the end. They come from one
rendered clip, so read them as a trend to watch, not as a gate. The gate is the
offline evaluation over a held-out set, against the real recordings of the same
sentences.

A screen that cannot measure a clip reports nothing and raises no flag. Silence
is not innocence. The flag thresholds (defaults in `evaluation.py`) were set on
a small set of speakers on one render channel, so a clip that sits just inside
one has proven nothing.

*(Correction, 2026-10-03: the lock count takes a clip whose median tracked
pitch lies within 1.5 Hz of any of the first three multiples, and the two upper
multiples fall inside real voices. A lock at 187.5 or 281.25 Hz can be the
speaker's own pitch and not a comb. Read a lock count together with the median
pitch it was taken from, and recount by multiple when the pitch range of the
speaker or corpus reaches the upper multiples. Lock counts recorded earlier with
this instrument count matches to any of the first three multiples, so where the
pitch range reaches the upper multiples they can include the speaker's own
pitch rather than a comb.)*

### Reading limits

**Short windows.** On windows of a few hundred milliseconds a real recording can
read clearly above zero on the excess, so a threshold that works on whole clips
is not a line on windows. Compare a window with the same window of a recording,
not with a whole-clip threshold.

**Between words, and the quantisation floor.** Windows with no active frame sit
close to the 16-bit quantisation level, so the inter-word ratio there is partly
set by quantisation. Compare the float output with the stored file on the
windows a number rests on before reading it. In fully active windows the two
agree; in windows with little or no speech the float output can read higher
than the stored file, by an amount that varies from track to track, and windows
with only a few active frames are not clean either. Quantisation lowers the
measured ratio there; it does not create the comb. So a comb seen between words
is a property of the model, but its size there is a measurement about the
stored file as much as about the model.

**A ratio versus an absolute level.** The excess is a ratio whose denominator is
the render's own broadband floor, which makes it a good detector and an unsafe
comparator. When one render has both a quieter comb and a much quieter floor,
the ratio can rank the render whose comb is quieter worse. Every screen above
is also invariant to pure gain, so none of them says how much comb there is.
Three readings answer three questions. The excess ratio asks whether the
artifact is present. The level against signal power asks what share of the
render is comb. The level plus the clip's `rms_dbfs` (the absolute mean power
of the selected bins, not an isolated ringing component) asks how much comb
there is as rendered, and it is worth computing whenever two renders sit at
different loudness. When one render is louder, shorter or has less silence in
it, these readings can disagree about which of two arms rings less; length,
pause share and spectral content move total power on their own, and no
normalisation removes them. So name the reading when reporting a comparison.
None of the three replaces listening.

### Which path rings

A checkpoint renders two ways. Reconstruction runs a real recording through the
posterior encoder into the decoder. Inference runs text through the prior
encoder, the duration predictor and the flow into the same decoder. The decoder
is the only module the two share. Reconstruction uses the posterior and the
decoder together, so a ringing reconstruction cannot single out either of them.
It also inherits the recording's timing and contour, so a listener may be able
to tell which tracks are reconstructions, and the comparison is not fully blind
to the path. Ringing has been observed on both paths, so a mismatch between the
training path and the inference path is not the whole story, but it is not
cleared either.

### What is known about the cause

The cause is open. Treat the screens as the reliable part and the explanations
as provisional.

The first explanation was drift in the latents. The released checkpoint carries
no posterior encoder, so a fresh one starts every run, and while the decoder is
frozen the adversarial gradients still reach the posterior encoder through it,
with nothing anchoring where they go. They do not reach the flow, which the KL
term updates instead. The size of that drift, taken as a single scalar, was not
found to be the controlling variable. Which property of the latents matters, if
any, is unsettled: per-channel offset, scale and smoothing in time have been
probed, and the screens did not agree about them.

Crossing latents against decoders (the same sentences through several decoders
with the latents held fixed) is the way to separate where the comb enters. The
crossings run so far were read on the ratio screen without listening, and
swapping the latents also swapped the speaker, so they isolate neither a
property of the latents nor clear the decoder. Which of the two the comb enters
through is not settled, and a crossing that implicates one does not clear the
other.

Comb size alone does not say whether speech formed, so the two are treated as
separate problems and speech is judged by a listener. Forming speech and
finishing quality are separate milestones too: a run that has formed speech has
not necessarily reached usable quality. Hearing the right language is a third
question. Report speech, language and defect separately, and do not promote
"speech" to "usable".

Equal steps are not equal epochs. The same number of updates is a different
number of passes over corpora of different size, and the starting weights, the
data volume and the target pitch range differ as well, so equal step counts
across conditions are not a like-for-like comparison.

### Training controls

Every control below is off by default, so a command that sets none of them
behaves as it always has. `TRAINING.md` says what each one does; this section
says when each one is a reasonable thing to try. No evidence yet favours one
over another.

- `--adversarial-gating` holds the generator's adversarial and feature-matching
  terms at zero while the decoder is frozen, then ramps them in over
  `--adversarial-ramp-steps` from the unfreeze step. `--warmup-adversarial-gating`
  holds the same terms at zero for the posterior warm-up alone and restores full
  weight when linguistic adaptation starts. Try them when you want the posterior
  encoder not to be pushed by a freshly initialised critic through a frozen
  decoder. Gating removes one candidate factor; it has not been shown to prevent
  the comb.
- `--decoder-lr-warmup-steps` eases the decoder in after it unfreezes, so its
  first update is not the largest it will ever take. Try it when the comb first
  appears around the unfreeze step.
- `--decoder-polish-mode recon` trains only the decoder in the polish stage,
  against reconstruction losses with no discriminator, with the posterior
  encoder and the flow held, so the latents stop moving while the decoder
  answers for them. The `posterior_decoder` variants also train the posterior
  encoder (`posterior_decoder_recon` without the discriminator), so they do not
  hold the latents still.
- `--stft-loss-weight` adds a multi-resolution STFT term. In a synthetic
  comb-injection check it charged for a narrow comb more heavily than the
  80-band mel does; that is a property of that check, not a general one.
- `--decoder-proximal-weight` holds the decoder near its starting weights.
  `--decoder-freeze-upsamplers` holds the transposed convolutions, where the
  grid enters the waveform, and the input convolution.
- `--generator-ema-decay` keeps an averaged copy, a second candidate for the
  listening round at almost no cost.
- `--posterior-init inherit` starts from the posterior encoder of a chained
  export instead of a fresh one.
- `--discriminator-update-order first` steps the discriminator before the
  generator's adversarial terms are computed; neither order is recommended over
  the other (see `TRAINING.md`).

None of them is a known fix, and there is no recommended remedy. A control that
moves a screen has not thereby changed what a listener hears, and the earliest
trials of some controls were screen-only and ran before the mel-loss fix, so
they do not rule a control out. Use them one at a time, as candidates for a
blind listening round, with the pass bar declared before the run. Prove from the
logs that the manipulation took effect (the gated rows carry a zero adversarial
weight with null generator-side losses and a finite discriminator loss),
because a null result from a control that never switched on says nothing. A
restore of one module leaves a mixture of weights that never trained together,
so a variant that does not recover does not clear its module. Watch the screens
in `validation/step-*.json` and `z_dc_rms` in `metrics.jsonl` while a run is
still cheap to abandon.

### Comparing runs

Repeating a run with the same nominal settings does not reproduce the trained
weights: two runs with the same seed, corpus and early settings were found not
to end with the same trained weights, while the frozen parts matched. What
produces the variation has not been isolated: nothing here separates GPU kernel
nondeterminism from mixed precision from anything else, and no experiment to do
so is planned. Rendering is a separate matter: the render check re-renders a
checkpoint at a fixed seed and expects identical bytes.

So a contrast between two runs carries this variation alongside whatever was
manipulated. Its size is unmeasured, and a single repeat pair would be one
observation of it rather than a bound. Do not read a gain of zero as evidence
that the variation is small. A verdict that rests on a gain of one grade or less
is weakened by it, not supported. An interrupted and resumed run is also not an
uninterrupted run of the same length: the loader's shuffle position is not
checkpointed, so the data order restarts (see the branching notes in
`TRAINING.md`).

### Lessons for listening rounds

- Screens eliminate; the blind listener decides. A screen can separate two runs
  that a listener cannot, and the reverse.
- An unlocked pitch is not speech. It says the comb has not captured the
  tracker and nothing more, and a render can fail as speech while unlocked. A
  lock that releases is "released", not "speech". An aggregate over many
  sentences is a fact about those sentences, so do not read it alongside grades
  given to the few sentences that were heard.
- What a page cannot separate is the page's limit, not "no effect". One
  listener, a few sentences and ordinal scales of three to five steps can fail
  to put two tracks in different grades while a smaller difference, or one the
  axes do not ask about, remains open.
- A duplicate of one track, placed on the page under another name, shows what
  the same audio scores. If using the duplicate's score in place of the
  original's changes how a comparison is classified (for example because the
  two scores fall on both sides of the speech boundary), that comparison is
  undecided and is reported as such. A split that changes no classification is
  noted but decides nothing.
- Declare the bar before listening, and do not add gains from separate pages
  together: a bar met on one page and missed on another is two readings, not a
  sum.

The detailed numbers and listening results of the private runs behind this
section are kept in the private work records (`<work-dir>`). This page keeps the
definitions of the screens, the code defaults, and the lessons.

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
