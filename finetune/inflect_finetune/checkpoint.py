"""Strict public-checkpoint migration and toolkit checkpoint I/O."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from . import __version__

RELEASE_FORMAT = "inflect_vits_inference_checkpoint_v1"
TRAINING_FORMAT = "inflect_adaptation_training_checkpoint_v1"
INFERENCE_FORMAT = "inflect_vits_inference_checkpoint_v1"
RUN_IDENTITY_FORMAT = "inflect_adaptation_run_identity_v1"
EMBEDDING_KEY = "enc_p.emb.weight"
FRESH_PREFIXES = ("enc_q.",)


@dataclass(frozen=True)
class CompatibilityReport:
    source_path: str
    source_format: str
    source_tensor_count: int
    source_parameter_count: int
    copied_tensor_count: int
    copied_parameter_count: int
    exact_tensor_count: int
    migrated_embedding_rows: int
    initialized_embedding_rows: int
    fresh_tensor_count: int
    fresh_parameter_count: int
    fresh_prefixes: tuple[str, ...]
    verified_equal_after_copy: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_release_checkpoint(path: str | Path) -> tuple[dict, Mapping[str, torch.Tensor]]:
    source = Path(path).resolve()
    payload = _torch_load(source)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), Mapping):
        raise ValueError(f"{source} is not an Inflect inference checkpoint.")
    checkpoint_format = str(payload.get("format", ""))
    if checkpoint_format != RELEASE_FORMAT:
        raise ValueError(
            f"Unsupported checkpoint format {checkpoint_format!r}; expected {RELEASE_FORMAT!r}."
        )
    state = payload["model"]
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("The release model state contains non-tensor entries.")
    return payload, state


def _symbol_seed(seed: int, symbol: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{symbol}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF


def _initialize_embedding_row(row: torch.Tensor, symbol: str, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_symbol_seed(seed, symbol))
    values = torch.empty(row.shape, dtype=torch.float32, device="cpu")
    values.normal_(mean=0.0, std=float(row.shape[-1]) ** -0.5, generator=generator)
    row.copy_(values.to(dtype=row.dtype, device=row.device))


def _occurrence_indices(symbols: Sequence[str]) -> dict[tuple[str, int], int]:
    """Map repeated symbols by occurrence so release duplicate rows stay distinct."""

    counts: defaultdict[str, int] = defaultdict(int)
    indices: dict[tuple[str, int], int] = {}
    for index, symbol in enumerate(symbols):
        occurrence = counts[symbol]
        indices[(symbol, occurrence)] = index
        counts[symbol] += 1
    return indices


def warm_start_from_release(
    model: nn.Module,
    checkpoint_path: str | Path,
    base_symbols: Sequence[str],
    target_symbols: Sequence[str],
    *,
    initialization_seed: int = 1234,
) -> CompatibilityReport:
    """Load all released tensors exactly, allowing only a fresh enc_q.

    The embedding may grow for a new language. Existing rows are mapped by
    symbol identity and remain bit-identical; only genuinely new rows are
    initialized.
    """

    source_path = Path(checkpoint_path).resolve()
    payload, source = load_release_checkpoint(source_path)
    target = model.state_dict()
    source_keys = set(source)
    target_keys = set(target)

    unexpected = sorted(source_keys - target_keys)
    if unexpected:
        raise RuntimeError(f"Release checkpoint has unexpected generator keys: {unexpected}")
    fresh = sorted(target_keys - source_keys)
    invalid_fresh = [key for key in fresh if not key.startswith(FRESH_PREFIXES)]
    if invalid_fresh:
        raise RuntimeError(
            "Training model contains non-release keys outside the allowed training-only "
            f"prefixes {FRESH_PREFIXES}: {invalid_fresh}"
        )
    if EMBEDDING_KEY not in source or EMBEDDING_KEY not in target:
        raise RuntimeError(f"Both states must contain {EMBEDDING_KEY}.")
    if len(base_symbols) != source[EMBEDDING_KEY].shape[0]:
        raise RuntimeError(
            "Base symbol inventory length does not match the released embedding rows."
        )
    if len(target_symbols) != target[EMBEDDING_KEY].shape[0]:
        raise RuntimeError(
            "Target symbol inventory length does not match the training embedding rows."
        )

    mismatched = []
    for key in sorted(source_keys - {EMBEDDING_KEY}):
        if source[key].shape != target[key].shape or source[key].dtype != target[key].dtype:
            mismatched.append(
                (
                    key,
                    tuple(source[key].shape),
                    tuple(target[key].shape),
                    source[key].dtype,
                    target[key].dtype,
                )
            )
    if mismatched:
        raise RuntimeError(f"Released non-embedding tensors do not match exactly: {mismatched}")

    migrated = target[EMBEDDING_KEY].clone()
    base_index = _occurrence_indices(base_symbols)
    target_index = _occurrence_indices(target_symbols)
    target_occurrences: defaultdict[str, int] = defaultdict(int)
    copied_rows = 0
    initialized_rows = 0
    for row_index, symbol in enumerate(target_symbols):
        occurrence = target_occurrences[symbol]
        target_occurrences[symbol] += 1
        source_index = base_index.get((symbol, occurrence))
        if source_index is None:
            _initialize_embedding_row(
                migrated[row_index],
                f"{symbol}\0occurrence={occurrence}",
                initialization_seed,
            )
            initialized_rows += 1
        else:
            migrated[row_index].copy_(source[EMBEDDING_KEY][source_index])
            copied_rows += 1

    loaded = dict(target)
    for key, value in source.items():
        if key != EMBEDDING_KEY:
            loaded[key] = value
    loaded[EMBEDDING_KEY] = migrated
    model.load_state_dict(loaded, strict=True)

    verified = model.state_dict()
    unequal = [
        key
        for key in source_keys - {EMBEDDING_KEY}
        if not torch.equal(verified[key].cpu(), source[key].cpu())
    ]
    for identity, source_index in base_index.items():
        if identity in target_index:
            if not torch.equal(
                verified[EMBEDDING_KEY][target_index[identity]].cpu(),
                source[EMBEDDING_KEY][source_index].cpu(),
            ):
                unequal.append(f"{EMBEDDING_KEY}[{identity!r}]")
    if unequal:
        raise RuntimeError(f"Warm-start verification failed for released tensors: {unequal}")

    source_parameters = sum(tensor.numel() for tensor in source.values())
    fresh_parameters = sum(target[key].numel() for key in fresh)
    return CompatibilityReport(
        source_path=str(source_path),
        source_format=str(payload["format"]),
        source_tensor_count=len(source),
        source_parameter_count=source_parameters,
        copied_tensor_count=len(source),
        copied_parameter_count=source_parameters,
        exact_tensor_count=len(source) - 1,
        migrated_embedding_rows=copied_rows,
        initialized_embedding_rows=initialized_rows,
        fresh_tensor_count=len(fresh),
        fresh_parameter_count=fresh_parameters,
        fresh_prefixes=FRESH_PREFIXES,
        verified_equal_after_copy=True,
    )


def cpu_compatibility_report(
    model: nn.Module,
    checkpoint_path: str | Path,
    base_symbols: Sequence[str],
    target_symbols: Sequence[str],
    *,
    initialization_seed: int = 1234,
) -> CompatibilityReport:
    """Run strict migration on CPU and return its machine-readable audit."""

    model.cpu()
    return warm_start_from_release(
        model,
        checkpoint_path,
        base_symbols,
        target_symbols,
        initialization_seed=initialization_seed,
    )


def _atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_identity(
    *,
    run_id: str,
    base_root: str | Path,
    prepared_dir: str | Path,
    options: Mapping[str, Any],
    optimizer_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a public, path-independent identity for one adaptation run."""

    base = Path(base_root).resolve()
    prepared = Path(prepared_dir).resolve()
    required = {
        "base checkpoint": base / "model.pth",
        "base config": base / "config.json",
        "dataset metadata": prepared / "dataset.json",
        "training split": prepared / "train.jsonl",
        "validation split": prepared / "validation.jsonl",
        "symbol inventory": prepared / "symbols.json",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot establish a resumable run identity; required public inputs are missing: "
            + "; ".join(missing)
        )
    if not run_id or not isinstance(run_id, str):
        raise ValueError("run_id must be a non-empty string.")
    identity = {
        "format": RUN_IDENTITY_FORMAT,
        "toolkit_version": __version__,
        "run_id": run_id,
        "base": {
            "identity": base.name,
            "checkpoint_sha256": sha256_file(required["base checkpoint"]),
            "config_sha256": sha256_file(required["base config"]),
        },
        "prepared_dataset": {
            "dataset_json_sha256": sha256_file(required["dataset metadata"]),
            "train_jsonl_sha256": sha256_file(required["training split"]),
            "validation_jsonl_sha256": sha256_file(required["validation split"]),
        },
        "symbols_sha256": sha256_file(required["symbol inventory"]),
        "options": dict(options),
        "optimizer_schema": dict(optimizer_schema),
    }
    # Enforce JSON stability at construction time instead of failing at save.
    return json.loads(_canonical_json(identity))


def validate_run_identity(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    source: str = "resume checkpoint",
) -> None:
    if not isinstance(actual, Mapping):
        raise ValueError(f"{source} does not contain run identity metadata.")
    if actual.get("format") != RUN_IDENTITY_FORMAT:
        raise ValueError(f"{source} has an unsupported run identity format.")
    if _canonical_json(actual) != _canonical_json(expected):
        differing = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise ValueError(
            f"{source} belongs to a different adaptation run; identity fields differ: "
            f"{differing}"
        )


def write_run_identity(path: str | Path, identity: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(identity, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def load_run_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read run identity marker {source}: {error}") from error
    if not isinstance(payload, dict) or payload.get("format") != RUN_IDENTITY_FORMAT:
        raise ValueError(f"{source} is not an Inflect adaptation run identity marker.")
    return payload


def copy_checkpoint_alias(source: str | Path, destination: str | Path) -> Path:
    """Atomically refresh a stable checkpoint copy without symlink assumptions."""

    source_path = Path(source).resolve()
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(f"Checkpoint alias source does not exist: {source_path}")
    if source_path == destination_path.resolve():
        return destination_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(destination_path.name + ".tmp")
    shutil.copyfile(source_path, temporary)
    os.replace(temporary, destination_path)
    return destination_path


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    path: str | Path,
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scheduler_g: Any,
    scheduler_d: Any,
    scaler: Any,
    step: int,
    epoch: int,
    stage: str,
    options: Mapping[str, Any],
    symbols: Sequence[str],
    compatibility: CompatibilityReport,
    run_identity: Mapping[str, Any],
    latest_path: str | Path | None = None,
) -> Path:
    destination = Path(path)
    payload = {
        "format": TRAINING_FORMAT,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scheduler_g": scheduler_g.state_dict(),
        "scheduler_d": scheduler_d.state_dict(),
        "scaler": scaler.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "stage": str(stage),
        "options": dict(options),
        "symbols": list(symbols),
        "compatibility": compatibility.to_dict(),
        "run_identity": dict(run_identity),
        "rng_state": capture_rng_state(),
    }
    _atomic_torch_save(payload, destination)
    if latest_path is not None:
        copy_checkpoint_alias(destination, latest_path)
    return destination


def resume_training_checkpoint(
    path: str | Path,
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scheduler_g: Any,
    scheduler_d: Any,
    scaler: Any,
    expected_symbols: Sequence[str],
    expected_run_identity: Mapping[str, Any],
) -> tuple[int, int, str]:
    payload = _torch_load(Path(path))
    if not isinstance(payload, dict) or payload.get("format") != TRAINING_FORMAT:
        raise ValueError(
            "Resume requires an Inflect adaptation training checkpoint, not a base release "
            "checkpoint."
        )
    validate_run_identity(
        payload.get("run_identity"),
        expected_run_identity,
        source="resume checkpoint",
    )
    if tuple(payload.get("symbols", ())) != tuple(expected_symbols):
        raise ValueError("Resume checkpoint symbol inventory differs from the prepared dataset.")
    stage = payload.get("stage")
    if not isinstance(stage, str) or not stage:
        raise ValueError("Resume checkpoint does not record the active warm-start stage.")
    required_state = {
        "generator",
        "discriminator",
        "optimizer_g",
        "optimizer_d",
        "scheduler_g",
        "scheduler_d",
        "scaler",
        "step",
        "epoch",
    }
    missing_state = sorted(required_state - payload.keys())
    if missing_state:
        raise ValueError(f"Resume checkpoint is missing mutable state: {missing_state}")

    # Identity, symbols, stage, and payload shape are validated before any
    # live module, optimizer, scaler, scheduler, or RNG state is mutated.
    generator.load_state_dict(payload["generator"], strict=True)
    discriminator.load_state_dict(payload["discriminator"], strict=True)
    optimizer_g.load_state_dict(payload["optimizer_g"])
    optimizer_d.load_state_dict(payload["optimizer_d"])
    scheduler_g.load_state_dict(payload["scheduler_g"])
    scheduler_d.load_state_dict(payload["scheduler_d"])
    scaler.load_state_dict(payload["scaler"])
    if "rng_state" in payload:
        restore_rng_state(payload["rng_state"])
    return int(payload["step"]), int(payload["epoch"]), stage


def save_inference_checkpoint(
    path: str | Path,
    *,
    generator: nn.Module,
    iteration: int,
    learning_rate: float,
) -> Path:
    """Save deployable state; training-only posterior tensors are excluded."""

    state = {
        key: value.detach().cpu()
        for key, value in generator.state_dict().items()
        if not key.startswith(FRESH_PREFIXES)
    }
    deployable_parameters = sum(tensor.numel() for tensor in state.values())
    payload = {
        "format": INFERENCE_FORMAT,
        "model": state,
        "iteration": int(iteration),
        "learning_rate": float(learning_rate),
        "deployable_parameters": deployable_parameters,
    }
    destination = Path(path)
    _atomic_torch_save(payload, destination)
    return destination
