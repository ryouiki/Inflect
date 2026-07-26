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
- peak level, clipping indicators, silence, and non-finite samples;
- duplicate audio content and transcripts crossing split boundaries;
- transcript and frontend failures;
- unknown symbols and phoneme coverage;
- split statistics and source-manifest hash.

Configured duration and structural thresholds are recorded. A clip passing
automated checks does not prove that its transcript, speaker identity,
pronunciation, or audio quality is correct.

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
