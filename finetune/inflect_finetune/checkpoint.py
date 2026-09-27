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
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch import nn

from . import __version__

RELEASE_FORMAT = "inflect_vits_inference_checkpoint_v1"
TRAINING_FORMAT = "inflect_adaptation_training_checkpoint_v1"
INFERENCE_FORMAT = "inflect_vits_inference_checkpoint_v1"
RUN_IDENTITY_FORMAT = "inflect_adaptation_run_identity_v1"
POSTERIOR_FORMAT = "inflect_vits_posterior_v1"
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
    base_symbol_count: int
    discarded_base_symbols: tuple[str, ...]
    fresh_tensor_count: int
    fresh_parameter_count: int
    fresh_prefixes: tuple[str, ...]
    verified_equal_after_copy: bool
    # fresh_prefixes stays the truth about which tensors the release omits; these two
    # record where a fresh posterior's values came from, a seed or an earlier run.
    posterior_source: str = "fresh"
    posterior_tensor_count: int = 0

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
    posterior_state: Mapping[str, torch.Tensor] | None = None,
) -> CompatibilityReport:
    """Load all released tensors exactly, allowing only a fresh enc_q.

    The embedding may grow for a new language. Existing rows are mapped by
    symbol identity and remain bit-identical; only genuinely new rows are
    initialized.

    `posterior_state` inherits a previous run's posterior instead of paying for a
    freshly seeded one again. It is held to the model's own posterior key set
    exactly: a sidecar that differs describes a different architecture, and
    accepting it partially would yield a run that reports itself warm-started
    while part of it is not.
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

    if posterior_state is not None:
        unpaired = sorted(set(posterior_state) ^ set(fresh))
        if unpaired:
            raise RuntimeError(
                "Posterior sidecar does not describe this model's posterior; keys present on "
                f"only one side: {unpaired}"
            )
        posterior_mismatched = [
            (
                key,
                tuple(posterior_state[key].shape),
                tuple(target[key].shape),
                posterior_state[key].dtype,
                target[key].dtype,
            )
            for key in fresh
            if posterior_state[key].shape != target[key].shape
            or posterior_state[key].dtype != target[key].dtype
        ]
        if posterior_mismatched:
            raise RuntimeError(
                f"Posterior sidecar tensors do not match the model posterior: "
                f"{posterior_mismatched}"
            )

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
    if posterior_state is not None:
        live = model.state_dict()
        with torch.no_grad():
            for key, value in posterior_state.items():
                live[key].copy_(value)

    verified = model.state_dict()
    unequal = [
        key
        for key in source_keys - {EMBEDDING_KEY}
        if not torch.equal(verified[key].cpu(), source[key].cpu())
    ]
    if posterior_state is not None:
        unequal.extend(
            key
            for key, value in posterior_state.items()
            if not torch.equal(verified[key].cpu(), value.cpu())
        )
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
        base_symbol_count=len(base_symbols),
        # A base whose inventory extends the release one can carry rows the new
        # dataset does not use. Dropping them is correct, but it must be visible:
        # those are trained weights the next stage will not inherit.
        discarded_base_symbols=tuple(
            sorted(
                symbol
                for symbol, occurrence in base_index
                if (symbol, occurrence) not in target_index
            )
        ),
        fresh_tensor_count=len(fresh),
        fresh_parameter_count=fresh_parameters,
        fresh_prefixes=FRESH_PREFIXES,
        verified_equal_after_copy=True,
        posterior_source="fresh" if posterior_state is None else "sidecar",
        posterior_tensor_count=0 if posterior_state is None else len(posterior_state),
    )


def cpu_compatibility_report(
    model: nn.Module,
    checkpoint_path: str | Path,
    base_symbols: Sequence[str],
    target_symbols: Sequence[str],
    *,
    initialization_seed: int = 1234,
    posterior_state: Mapping[str, torch.Tensor] | None = None,
) -> CompatibilityReport:
    """Run strict migration on CPU and return its machine-readable audit."""

    model.cpu()
    return warm_start_from_release(
        model,
        checkpoint_path,
        base_symbols,
        target_symbols,
        initialization_seed=initialization_seed,
        posterior_state=posterior_state,
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
    posterior_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a public, path-independent identity for one adaptation run.

    A run that inherits a posterior sidecar pins it here, so swapping the sidecar
    between runs is as fatal on resume as swapping model.pth. A run without one
    records no posterior field at all: adding a null would change the identity of
    every default run.
    """

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
    if posterior_path is not None:
        required["posterior sidecar"] = Path(posterior_path).resolve()
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
    if posterior_path is not None:
        identity["base"]["posterior_sha256"] = sha256_file(required["posterior sidecar"])
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
    generator_ema: Mapping[str, torch.Tensor] | None = None,
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
    # Only a run that averages the generator adds the key, so a run that does not
    # keeps writing exactly the payload it wrote before the option existed.
    if generator_ema is not None:
        payload["generator_ema"] = {
            key: value.detach().cpu() for key, value in generator_ema.items()
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
    generator_ema: MutableMapping[str, torch.Tensor] | None = None,
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
    if generator_ema is not None:
        required_state.add("generator_ema")
    missing_state = sorted(required_state - payload.keys())
    if missing_state == ["generator_ema"]:
        raise ValueError(
            "Resume checkpoint carries no averaged generator state because it predates the "
            "generator_ema_decay option; resume without generator_ema_decay, or start a new run."
        )
    if missing_state:
        raise ValueError(f"Resume checkpoint is missing mutable state: {missing_state}")
    if generator_ema is not None:
        unpaired = sorted(set(payload["generator_ema"]) ^ set(generator_ema))
        if unpaired:
            raise ValueError(
                "Resume checkpoint averaged generator state does not match the live averaged "
                f"generator; keys present on only one side: {unpaired}"
            )

    # Identity, symbols, stage, and payload shape are validated before any
    # live module, optimizer, scaler, scheduler, or RNG state is mutated.
    generator.load_state_dict(payload["generator"], strict=True)
    discriminator.load_state_dict(payload["discriminator"], strict=True)
    optimizer_g.load_state_dict(payload["optimizer_g"])
    optimizer_d.load_state_dict(payload["optimizer_d"])
    scheduler_g.load_state_dict(payload["scheduler_g"])
    scheduler_d.load_state_dict(payload["scheduler_d"])
    scaler.load_state_dict(payload["scaler"])
    # A caller without averaging ignores a saved average, so an EMA checkpoint still
    # resumes into a run that turned the option off.
    if generator_ema is not None:
        with torch.no_grad():
            for key, value in payload["generator_ema"].items():
                generator_ema[key].copy_(value)
    if "rng_state" in payload:
        restore_rng_state(payload["rng_state"])
    return int(payload["step"]), int(payload["epoch"]), stage


def save_posterior_sidecar(
    path: str | Path,
    *,
    generator: nn.Module | None = None,
    state: Mapping[str, torch.Tensor] | None = None,
    iteration: int,
) -> Path:
    """Write the training-only posterior beside an export, for a later run to inherit.

    model.pth stays strictly inference-only, so a chained run otherwise starts from
    a freshly seeded posterior every time and pays that cost again.

    A caller holding a live module passes `generator`; the export path holds
    tensors read out of a saved payload and passes `state`. Both go through one
    writer so the format has a single owner.
    """

    if (generator is None) == (state is None):
        raise ValueError("Pass exactly one of generator or state.")
    source = generator.state_dict() if generator is not None else state
    posterior = {
        key: value.detach().cpu()
        for key, value in source.items()
        if key.startswith(FRESH_PREFIXES)
    }
    if not posterior:
        raise ValueError(
            f"The source holds no tensors under {FRESH_PREFIXES}; it has no posterior to "
            "export."
        )
    destination = Path(path)
    _atomic_torch_save(
        {"format": POSTERIOR_FORMAT, "model": posterior, "iteration": int(iteration)},
        destination,
    )
    return destination


def load_posterior_sidecar(path: str | Path) -> dict[str, torch.Tensor]:
    source = Path(path).resolve()
    payload = _torch_load(source)
    if not isinstance(payload, dict) or str(payload.get("format", "")) != POSTERIOR_FORMAT:
        raise ValueError(
            f"{source} is not an Inflect posterior sidecar; expected format "
            f"{POSTERIOR_FORMAT!r}."
        )
    state = payload.get("model")
    if state is None or not isinstance(state, Mapping):
        raise ValueError(f"{source} does not contain a posterior state mapping.")
    foreign = sorted(
        str(key)
        for key, value in state.items()
        if not isinstance(key, str)
        or not key.startswith(FRESH_PREFIXES)
        or not torch.is_tensor(value)
    )
    if foreign:
        raise ValueError(
            f"{source} holds entries that are not posterior tensors under {FRESH_PREFIXES}: "
            f"{foreign}"
        )
    return dict(state)


def save_inference_checkpoint(
    path: str | Path,
    *,
    generator: nn.Module,
    iteration: int,
    learning_rate: float,
    state: Mapping[str, torch.Tensor] | None = None,
) -> Path:
    """Save deployable state; training-only posterior tensors are excluded.

    `state` exports weights that are not the module's live ones — an averaged copy
    of the generator kept alongside it — through exactly this filtering and payload
    shape, so both candidates of a run are comparable file for file.
    """

    deployable = {
        key: value.detach().cpu()
        for key, value in (generator.state_dict() if state is None else state).items()
        if not key.startswith(FRESH_PREFIXES)
    }
    deployable_parameters = sum(tensor.numel() for tensor in deployable.values())
    payload = {
        "format": INFERENCE_FORMAT,
        "model": deployable,
        "iteration": int(iteration),
        "learning_rate": float(learning_rate),
        "deployable_parameters": deployable_parameters,
    }
    destination = Path(path)
    _atomic_torch_save(payload, destination)
    return destination
