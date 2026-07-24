# Contributing to Inflect

Contributions are welcome. For a speech model, the bar is reproducible evidence
rather than a single favorable clip.

## Good Contributions

Useful contributions usually fall into one of these categories:

- reproducible benchmark improvements
- clearer docs
- safer data preparation
- better evaluation prompts
- runtime stability fixes
- small, well-scoped runtime experiments
- bug reports with audio examples and exact commands

## Before Opening a PR

Please make sure:

- the change is small enough to review
- generated artifacts are not committed
- private reference voices are not committed
- checkpoints are not committed
- local absolute paths are not added to public docs
- the README does not claim unreleased model quality

## Runtime and evaluation reports

For model or inference changes, include:

- base variant
- changed variant
- exact command
- prompt set
- model package and commit tested
- checkpoint used
- listening notes
- objective metrics if available

Minimum listening notes:

- voice consistency
- pacing
- skipped words
- glitches
- long-prompt behavior

## Commit Scope

Keep PRs focused.

Good:

- "Add ASR pseudo-label filter"
- "Improve README and media kit"
- "Add duration-ratio check to benchmark"

Bad:

- one PR containing docs, checkpoints, generated audio, unrelated training changes, and local state files

## Local artifacts

Do not commit:

- `outputs/`
- `.blind_ab_state*/`
- `reference_voices/`
- local third-party checkouts
- checkpoints
- full generated datasets
- private audio

## Public release boundary

The release is open-weight. A contribution may improve public inference,
evaluation, examples, documentation, or integration without requiring the
private corpus-construction pipeline. Do not open issues requesting private
reference material, generated corpora, credentials, or undisclosed training
infrastructure.

## Project tone

Inflect should be ambitious without exaggerating. Distinguish measured results
from estimates, human preference from predicted quality, and complete
text-to-waveform parameters from training-only modules.
