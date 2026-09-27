# Data quality

Adaptation quality is constrained by the corpus. More hours do not compensate
for inaccurate transcripts, inconsistent speakers, clipping, or poor phoneme
coverage.

## Required properties

Use recordings that are:

- legally usable for training and redistribution under the intended terms;
- spoken by one consenting speaker for a fixed-voice checkpoint;
- paired with manually verified transcripts;
- mono or safely convertible to mono;
- consistently recorded, without changing microphones or aggressive effects;
- free from clipping, dropouts, corruption, background music, and overlapping
  speakers;
- trimmed without cutting initial consonants, breaths needed for natural
  phrasing, or sentence endings;
- diverse in phonemes, word positions, sentence lengths, punctuation, and
  prosodic patterns.

Keep raw source audio unchanged. Preparation should write converted copies under
the prepared dataset rather than destructively replacing source files.

## Coverage matters

Inspect the audit report for:

- phonemes absent or rare in training;
- symbols introduced by only one transcript;
- validation phonemes not represented in training;
- repeated sentence templates;
- narrow pitch or duration distributions;
- unusually long or short clips;
- names, numbers, abbreviations, and punctuation patterns relevant to the
  intended use.

An eSpeak frontend producing a phoneme does not mean the model has enough data
to learn it. Newly initialized symbol rows need repeated, acoustically clear
examples in varied contexts.

## Recording consistency

Room tone, microphone frequency response, denoising, compression, and loudness
changes can become part of the learned voice. Avoid mixing studio audio,
telephone audio, and heavily processed clips unless that variation is an
intentional target and is evaluated.

Do not apply strong denoising or de-essing blindly. Processing can create
musical noise, phase artifacts, dull consonants, or unstable sibilance that a
small decoder reproduces prominently.

## Automated checks are diagnostics

The preparation and audit stages report:

- decode and sample-rate failures;
- channel count and duration;
- peak level, silence, and non-finite samples;
- clipping, separately for what the recordings arrived with
  (`source_clipped_files`) and what the conversion introduced
  (`output_clipped_files`);
- duplicate audio content and transcripts crossing split boundaries;
- transcript and frontend failures;
- unknown symbols and phoneme coverage;
- split statistics and source-manifest hash.

Configured duration and structural thresholds are recorded. A clip passing
automated checks does not prove that its transcript, speaker identity,
pronunciation, or audio quality is correct.

`prepare` refuses a dataset whose conversion clipped any row. Resampling rings
above the source peak, so a corpus mastered near full scale — anything limited
at a fraction of a decibel below it — loses samples to the peak limit as a
matter of course, and the loss is not visible in the source-side number. The
verdict is taken from each row's peak after resampling and before the clip
(`pre_clip_peak` and `output_clipped_samples` per row in
`preparation_report.json`), never from the written file, whose peak of 1.0
cannot say whether anything was cut. There is no tolerance: a peak 0.04 dB over
full scale is refused like any other.

A refused run writes no dataset. It leaves `<output>.clipping-report.json`
beside the requested directory, with the clipped rows and a recommended gain,
and the error names that gain. The fix is `--input-gain-db`: one gain over the
whole corpus, applied before resampling, chosen so the largest pre-clip peak
lands at −1 dBFS. The recommendation is `G_new = G + 20·log10(T/P)`, where `G`
is the gain of the refused run, `P` its largest pre-clip peak and `T` the
−1 dBFS target, rounded down to 0.1 dB so the rounding never costs headroom.
The same output directory may be reused; a later successful run removes the
refusal report it leaves behind. Lowering only the rows that clipped would
change the level relationship between rows, which is a property of the corpus
rather than of those rows, so per-row gain is not offered.

The default gain is 0. It leaves the samples untouched and adds nothing to
`dataset.json`, which stays byte-identical to one prepared before the option
existed, so an existing run's identity survives preparing the same manifest
again. A nonzero
gain is recorded in `dataset.json` under `audio_processing`, with the peak
limit, and `dataset.json` is part of a run's identity: a different gain is a
different dataset for checkpoints and exports. The measured peak is a float
that would move with any resampler change, so it is kept out of the hashed file,
in `preparation_report.json` under `conversion`, and in
`PREPARATION_REPORT.txt`.

A nonzero gain changes every row's audio bytes, and the split ranks rows by a
hash that includes them, so the validation split is drawn again. Evaluation on a
gained dataset's validation rows is not comparable row for row with an ungained
one prepared from the same manifest.

Clipping the recordings arrived with is still counted on the untouched source
(`source_clipped_files`) and does not disappear when the gain lowers it; a copy
attenuated outside the tool reports its own, lower count. `audit` rejects a
dataset whose `output_clipped_files` is nonzero, and warns without rejecting
when a dataset prepared before this count existed cannot answer.

## Manual review

Listen to a random sample, every flagged clip, and all validation clips. Check
the beginning and ending of each clip, consonant clarity, background sound,
speaker consistency, and transcript agreement.

Before a long run, train a short smoke run and listen to held-out synthesis.
Stop if the model develops severe buzz, metallic resonance, clipped endings,
identity collapse, unintelligible phonemes, or unstable duration.

## Data volume

This toolkit does not promise a minimum number of minutes or hours that will
work for every language and speaker. Required data depends on phoneme coverage,
recording consistency, desired quality, distance from the base language and
voice, and which parts of the generator must adapt.

Any future presets describing data volume must be validated experimentally and
must not be interpreted as quality guarantees.
