# Inflect v2 Evaluation

Inflect v2 is evaluated as a complete text-to-waveform system. The release does
not use one score as a substitute for listening, intelligibility, footprint,
and runtime.

## Headline results

| Model | Community preference ↑ | UTMOS22 ↑ | Semantic WER ↓ | FP32 weights ↓ |
| --- | ---: | ---: | ---: | ---: |
| **Inflect-Micro-v2** | **66.2%** | **4.395** | **3.99%** | 37.53 MB |
| **Inflect-Nano-v2** | **63.9%** | **4.386** | **4.21%** | **15.97 MB** |

These metrics are intentionally separate. Community preference is not MOS,
UTMOS is not a human panel, WER does not measure naturalness, and file size
does not imply speed.

## 1. Community blind listening

The public study used hidden model identities, randomized left/right order,
and prompt-level pairwise decisions. Participants could select either clip or
a tie. Preference rate is normalized by appearances; a tie counts as half a
win.

The released leaderboard records 147 pairwise decisions across Inflect and
established compact TTS baselines. The number is disclosed because sample size
is necessary to interpret uncertainty. The study is useful descriptive
evidence, not a controlled demographic panel or formal MOS experiment.

Controls:

- model names were hidden during each decision;
- left/right assignment was randomized;
- systems were compared on matched text;
- participants could stop rather than being forced through every pair;
- no account, address, or identity was attached to a vote.

Human results should be read with the raw win/loss/tie counts and confidence
intervals included in the Hugging Face evaluation package.

## 2. Predicted naturalness

UTMOS22 was run on 500 matched unseen prompts per evaluated voice. It estimates
perceived quality from audio, but remains a learned predictor with its own
domain biases.

Appropriate use:

- compare matched render sets;
- detect large quality regressions;
- report alongside human listening and signal diagnostics.

Inappropriate use:

- call the value human MOS;
- optimize a checkpoint only for the predictor;
- compare values from incompatible preprocessing without disclosure.

## 3. Intelligibility

The Modern400 suite contains 400 held-out prompts spanning conversational
English, punctuation, names, abbreviations, dates, currency, measurements, and
other normalization stress cases.

The headline semantic WER is the equal-weight corpus-level mean of:

- Qwen3-ASR;
- Nemotron 3.5 ASR.

Whisper large-v3 remains in the raw audit. It is excluded consistently from the
headline aggregate because it produced insertion-heavy failures on a subset of
Supertonic 8-step clips that contradicted the other recognizers and direct
listening. This exclusion rule is applied at the protocol level, not only to
Inflect.

Semantic scoring normalizes equivalent written and spoken forms where the
meaning is unchanged, such as `12.6` and `twelve point six`. This avoids
penalizing a correct utterance solely for ASR formatting.

WER still has limits:

- an ASR model can hide or introduce pronunciation errors;
- low WER does not prove natural timing or pleasant audio;
- difficult proper nouns can dominate a short set;
- acoustic artifacts may not change the transcript.

## 4. Complete footprint

Parameter and file-size claims cover the entire learned inference path:

| Model | Parameters | FP32 weights | External vocoder |
| --- | ---: | ---: | --- |
| Inflect-Micro-v2 | 9,356,513 | 37.53 MB | No |
| Inflect-Nano-v2 | 3,966,721 | 15.97 MB | No |

Training-only discriminators are excluded because they are not loaded for
synthesis. The text frontend is code rather than a second learned model.

## 5. CPU runtime

The public deployment profile measures Inflect under one managed environment:

- Hugging Face CPU Upgrade;
- 8 vCPU and 32 GB RAM;
- four framework threads;
- 100 fixed Modern400 prompts;
- three complete passes;
- pass 1 excluded as cache building;
- passes 2 and 3 pooled for steady-state results;
- end-to-end text processing and waveform synthesis included.

| Model | Median RTF ↓ | Throughput ↑ |
| --- | ---: | ---: |
| Inflect-Micro-v2 | 0.1593 | 6.28x real time |
| Inflect-Nano-v2 | 0.0933 | 10.72x real time |

`RTF = synthesis_seconds / generated_audio_seconds`; values below 1.0 are
faster than real time. Throughput is `1 / RTF`.

Direct competitor throughput is not used as a headline claim because the
systems use different frameworks, threading behavior, generation steps, and
warmup requirements. Those directional measurements can still be inspected in
the raw benchmark reports when the environment and caveats are retained.

## Comparator policy

The comparison set was chosen to avoid weak straw-man baselines:

- Piper low voices represent a widely deployed compact local baseline;
- KittenTTS Nano represents a popular sub-20M compact neural TTS family;
- Supertonic 3 at 3 and 8 steps represents a modern fast small-model system;
- Inflect Nano v1 is retained only as a historical internal baseline.

Piper and Kitten voice-level results remain visible where voice variation
matters. Aggregate summaries may average the two tested voices, but raw
voice-level rows are preserved.

## Reproducibility artifacts

Each Hugging Face package includes, under `evaluation/`:

- prompt manifests and hashes;
- generated sample references;
- ASR hypotheses and normalized references;
- corpus-level and category-level reports;
- bootstrap intervals where available;
- UTMOS summaries;
- runtime rows and environment metadata;
- SHA-256 integrity manifests.

The GitHub repository provides the interpretation and integration layer. The
model repositories are the source of exact checkpoint-specific evidence.

## Known weak cases

Both models can be weaker on unfamiliar proper nouns, abbreviations, ambiguous
numeric strings, homographs, extreme punctuation, and text far outside the
release distribution. Nano has less capacity and can sound thinner or flatter
on difficult prompts. Long-form chunk boundaries can expose changes in rhythm.

Claims should be updated only after a matched rerun. Hand-picked clips are
useful demonstrations, not substitutes for the frozen suites above.
