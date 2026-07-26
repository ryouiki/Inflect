"""Adapter example for an ASR system that the user installed separately.

This file intentionally does not import or download an ASR model. Replace the
body with a call to an already-configured local or hosted transcription system.
"""

from pathlib import Path
from typing import Any


def evaluate_transcript(
    audio_path: Path,
    reference_text: str,
    sample_rate: int,
) -> dict[str, Any]:
    raise RuntimeError(
        "Configure your own transcript evaluator in "
        "finetune/examples/transcript_evaluator_plugin.py. The adaptation "
        "toolkit never downloads a heavyweight ASR model automatically."
    )
