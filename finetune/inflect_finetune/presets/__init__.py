"""Wheel-safe built-in adaptation presets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "learning_rate_g": 0.0001,
        "learning_rate_d": 0.0001,
        "max_steps": 20_000,
        "num_workers": 4,
        "amp": True,
        "checkpoint_interval": 1_000,
        "validation_interval": 500,
    },
    "micro-12gb": {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate_g": 0.00008,
        "learning_rate_d": 0.00008,
        "max_steps": 20_000,
        "num_workers": 3,
        "amp": True,
        "checkpoint_interval": 1_000,
        "validation_interval": 500,
    },
    "nano-8gb": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate_g": 0.00008,
        "learning_rate_d": 0.00008,
        "max_steps": 20_000,
        "num_workers": 2,
        "amp": True,
        "checkpoint_interval": 1_000,
        "validation_interval": 500,
    },
}


def available_presets() -> tuple[str, ...]:
    return tuple(sorted(_PRESETS))


def load_packaged_preset(name: str) -> dict[str, Any]:
    try:
        return deepcopy(_PRESETS[name])
    except KeyError as error:
        raise KeyError(name) from error


__all__ = ["available_presets", "load_packaged_preset"]
