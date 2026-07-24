# Project Structure

Inflect combines a polished public v2 release surface with an active research
history. The two are intentionally separated so users have a clear integration
path without erasing reproducible experimentation.

## Public release surface

| Path | Purpose |
| --- | --- |
| `README.md` | Family overview, results, quickstart, and release scope |
| `requirements.txt` | Dependencies for the GitHub integration examples |
| `examples/` | Download-and-synthesize and matched comparison scripts |
| `docs/DEPLOYMENT.md` | Installation, controls, long text, CPU profile, formats |
| `docs/EVALUATION.md` | Frozen protocols, metrics, and comparison policy |
| `docs/ARCHITECTURE.md` | Shipped v2 inference architecture |
| `docs/INFLECT_V2_TECHNICAL_REPORT.md` | Concise technical report |
| `assets/` | Model covers and evidence visuals |
| `.github/workflows/` | Public-surface consistency checks |

Runnable weights, model-local inference code, checkpoint configuration,
evaluation rows, and integrity hashes live in:

- `owensong/Inflect-Micro-v2` on Hugging Face;
- `owensong/Inflect-Nano-v2` on Hugging Face.

Large weights are deliberately not duplicated in GitHub.

## Research implementation

| Path | Purpose |
| --- | --- |
| `inflect/` | Inflect-native model and experimental modules |
| `scripts/` | Training, evaluation, release, and research utilities |
| `configs/` | Experiment and model configurations |
| `docs/` | Public docs plus dated architecture and experiment records |

Dated internal reports are historical records, not necessarily current
deployment instructions. The root README and the five public documents linked
from it define the v2 release contract.

## Local-only artifacts

Generated datasets, checkpoints, reference audio, caches, rental-instance
transfers, and listening galleries must remain outside the public Git history.
The repository `.gitignore` and CI checks provide guardrails, but contributors
must still inspect staged files before every commit.

## Version policy

Inflect v2 is the active family. Inflect Nano v1 remains available as a legacy
Hugging Face release and in repository history for reproducibility. New users
should not combine v1 checkpoints or runtime code with the v2 packages.
