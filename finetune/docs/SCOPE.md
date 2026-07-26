# Scope and support status

## What this toolkit produces

This toolkit warm-starts a new Inflect-compatible generator from an official
Inflect v2 checkpoint and a user-supplied, single-speaker corpus. One run
produces one fixed voice for one configured language.

It does not add a speaker or language selector to the original model. Separate
voices or languages require separate prepared datasets, training runs, and
exports.

## Capability status

| Capability | Status |
| --- | --- |
| Prepare JSONL or CSV data | Implemented and tested |
| Resample and validate 24 kHz mono audio | Implemented and tested |
| Leakage-safe train/validation splitting | Implemented and tested |
| eSpeak language frontend | Implemented |
| Prephonemized input | Implemented |
| Explicit custom Python frontend | Implemented and source-hashed |
| Warm-start Micro or Nano | Implemented; Nano CUDA smoke-tested |
| Strict same-run resume | Implemented and tested |
| Fixed-voice adaptation | Experimental quality |
| New-language adaptation | Experimental quality |
| PyTorch deployment export | Implemented and load-tested |
| ONNX deployment export | Implemented; Nano parity-tested |
| Runtime-selectable voices or languages | Not supported |
| Zero-shot or few-shot cloning | Not supported |
| Quantized export | Not included |

The software path has been exercised end to end with a real prepared dataset,
a CUDA training step, resume, PyTorch export, and ONNX Runtime parity. This does
not prove that an arbitrary corpus or language will produce a good voice.
Fluent-speaker review and held-out listening remain required.

## Why adaptation is difficult

Language adaptation changes the symbol inventory, pronunciation rules, timing,
and acoustic distribution. Voice adaptation changes pitch range, formants,
speaking rate, recording conditions, and waveform statistics. Changing both at
once asks every part of a very small generator to move.

A model may remain intelligible while becoming thin, buzzy, metallic,
sibilant, unstable, or unlike the target speaker. Low training loss is not
evidence that adaptation succeeded.

## Public toolkit versus private release process

This package contains a generic compatible trainer: manifest readers, audio
checks, frontend adapters, symbol migration, generic losses and optimizer
defaults, evaluation utilities, checkpointing, and inference-only export.

It does not publish or reconstruct:

- private corpora, paths, transcripts, or generated audio;
- private corpus-generation or filtering methods;
- the exact curriculum used for the official checkpoints;
- private hyperparameter searches or failed experiments;
- internal checkpoint-ranking and release-selection procedures;
- credentials, rental identifiers, or private storage locations.

The released generator is an initialization point, not a resumable copy of the
original training run. It does not include the posterior encoder,
discriminators, optimizers, schedulers, manifests, or RNG state needed to
continue that run.

## Claims for adapted checkpoints

Do not describe an adapted checkpoint as an official Inflect voice or
language. Do not claim language support only because a frontend can emit its
phonemes. Publish the language, data provenance, consent basis, base model,
toolkit version, frontend, known limitations, and evaluation method.
