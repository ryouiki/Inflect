"""Inference-only PyTorch and optional ONNX exports for adapted checkpoints."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import shutil
import sys
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .reporting import file_record, make_report, sha256_file, status, write_checksums, write_json


_TRAINING_ONLY_PREFIXES = (
    "enc_q.",
    "discriminator.",
    "discriminators.",
    "disc.",
    "mpd.",
    "net_d.",
    "optimizer.",
    "optim.",
    "scheduler.",
    "scaler.",
    "amp_scaler.",
)

_TRAINING_ONLY_TOP_LEVEL_FIELDS = frozenset(
    {
        "compatibility",
        "discriminator",
        "discriminators",
        "disc",
        "epoch",
        "mpd",
        "net_d",
        "optim",
        "optimizer",
        "optimizer_d",
        "optimizer_g",
        "options",
        "rng_state",
        "scaler",
        "scheduler",
        "scheduler_d",
        "scheduler_g",
        "stage",
        "step",
    }
)


def _is_training_only_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in _TRAINING_ONLY_TOP_LEVEL_FIELDS or any(
        lowered == prefix[:-1] or lowered.startswith(prefix)
        for prefix in _TRAINING_ONLY_PREFIXES
    )


@dataclass(slots=True)
class ExportOptions:
    """Inputs for :func:`export_checkpoint`.

    ``package_template`` may point at a released Inflect package. Its public
    runtime and inference files are copied into the result and used for strict
    model-load verification. The source checkpoint may be either an inference
    checkpoint or a training checkpoint containing ``model``/``state_dict``.
    """

    checkpoint: str | Path
    output_dir: str | Path
    config: str | Path | None = None
    symbols: str | Path | Sequence[str] | Mapping[str, Any] | None = None
    package_template: str | Path | None = None
    prepared_dataset: str | Path | Mapping[str, Any] | None = None
    frontend_hook: str | Path | None = None
    include_onnx: bool = False
    onnx_opset: int = 17
    model_name: str | None = None
    source_revision: str | None = None
    overwrite: bool = False
    verify: bool = True


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}: {exc}") from exc


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_config(
    options: ExportOptions,
    checkpoint: Path,
    package_template: Path | None,
) -> tuple[dict[str, Any], Path | None]:
    candidates = []
    if options.config is not None:
        candidates.append(Path(options.config))
    candidates.extend((checkpoint.parent / "config.json", checkpoint.parent.parent / "config.json"))
    if package_template is not None:
        candidates.append(package_template / "config.json")
    for candidate in candidates:
        if candidate.is_file():
            config = _load_json(candidate)
            if not isinstance(config, dict):
                raise ValueError(f"Config must contain a JSON object: {candidate}")
            return config, candidate.resolve()
    raise FileNotFoundError(
        "No config.json was found. Pass ExportOptions(config=...) or place it beside "
        "the checkpoint."
    )


def _symbols_from_payload(payload: Any) -> list[str]:
    if isinstance(payload, (list, tuple)):
        symbols = list(payload)
    elif isinstance(payload, Mapping):
        for key in ("symbols", "ordered_symbols", "inventory"):
            if key in payload:
                return _symbols_from_payload(payload[key])
        raise ValueError("symbols.json must contain 'symbols', 'ordered_symbols', or 'inventory'.")
    else:
        raise ValueError("Symbols must be an ordered JSON list or an object containing one.")
    if not symbols or any(not isinstance(symbol, str) or not symbol for symbol in symbols):
        raise ValueError("Every symbol must be a non-empty string.")
    # The published v2 inventory contains a legacy duplicate apostrophe row.
    # Preserve ordered inventories exactly because embedding migration is by
    # symbol identity and numeric compatibility depends on every row.
    return symbols


def _load_symbols(
    options: ExportOptions,
    checkpoint: Path,
    checkpoint_payload: Mapping[str, Any],
    package_template: Path | None,
) -> tuple[list[str], Path | None]:
    source = options.symbols
    if source is None:
        for key in ("symbols", "symbol_inventory"):
            if key in checkpoint_payload:
                return _symbols_from_payload(checkpoint_payload[key]), None
        candidates = [
            checkpoint.parent / "symbols.json",
            checkpoint.parent.parent / "symbols.json",
        ]
        if package_template is not None:
            candidates.append(package_template / "symbols.json")
        for candidate in candidates:
            if candidate.is_file():
                return _symbols_from_payload(_load_json(candidate)), candidate.resolve()
        if package_template is not None:
            symbols_py = package_template / "runtime" / "text" / "symbols.py"
            if symbols_py.is_file():
                namespace: dict[str, Any] = {}
                source = symbols_py.read_text(encoding="utf-8")
                exec(compile(source, str(symbols_py), "exec"), namespace)
                return _symbols_from_payload(namespace["symbols"]), symbols_py.resolve()
        raise FileNotFoundError(
            "No ordered symbol inventory was found. Pass ExportOptions(symbols=...) or place "
            "symbols.json beside the checkpoint."
        )
    if isinstance(source, (str, Path)):
        path = Path(source)
        return _symbols_from_payload(_load_json(path)), path.resolve()
    return _symbols_from_payload(source), None


def _extract_state(
    payload: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[str]]:
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must contain a mapping.")
    state: Any = payload
    for key in ("model", "state_dict", "generator", "net_g"):
        if key in payload and isinstance(payload[key], Mapping):
            state = payload[key]
            break
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint does not contain a model state dictionary.")
    tensors: dict[str, torch.Tensor] = {}
    stripped: list[str] = []
    for raw_key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(raw_key)
        for prefix in ("module.", "model.", "generator."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if _is_training_only_name(key):
            stripped.append(key)
            continue
        tensors[key] = value.detach().cpu().contiguous()
    if not tensors:
        raise ValueError("Checkpoint state dictionary contains no tensors.")
    metadata = {str(key): value for key, value in payload.items() if value is not state}
    return tensors, metadata, sorted(stripped)


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as safe_error:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            raise safe_error
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping: {path}")
    return payload


def _resolve_package_template(
    options: ExportOptions,
    checkpoint_payload: Mapping[str, Any],
) -> Path | None:
    if options.package_template is not None:
        candidate: str | Path | None = options.package_template
    else:
        training_options = checkpoint_payload.get("options")
        candidate = (
            training_options.get("base_model")
            if isinstance(training_options, Mapping)
            else None
        )
    if candidate is None:
        return None
    try:
        from .modeling import resolve_base_model

        return resolve_base_model(candidate)
    except FileNotFoundError:
        path = Path(candidate).expanduser()
        if path.is_dir():
            return path.resolve()
        if options.package_template is not None:
            raise
        return None


def _is_unmodified_template_checkpoint(checkpoint: Path, template: Path | None) -> bool:
    template_checkpoint = template / "model.pth" if template is not None else None
    return bool(
        template_checkpoint
        and template_checkpoint.is_file()
        and sha256_file(checkpoint) == sha256_file(template_checkpoint)
    )


def _prepared_dataset_candidates(
    options: ExportOptions,
    checkpoint: Path,
) -> list[Path]:
    candidates: list[Path] = []
    if isinstance(options.symbols, (str, Path)):
        symbols_path = Path(options.symbols).expanduser()
        candidates.append(symbols_path.parent / "dataset.json")
    candidates.extend(
        (
            checkpoint.parent / "dataset.json",
            checkpoint.parent.parent / "dataset.json",
            checkpoint.parent.parent / "prepared" / "dataset.json",
        )
    )
    return candidates


def _load_prepared_dataset(
    options: ExportOptions,
    checkpoint: Path,
    package_template: Path | None,
) -> tuple[dict[str, Any], Path | None, bool]:
    source = options.prepared_dataset
    if isinstance(source, Mapping):
        return dict(source), None, False
    candidates: list[Path] = []
    if source is not None:
        explicit = Path(source).expanduser()
        candidates.append(explicit / "dataset.json" if explicit.is_dir() else explicit)
    else:
        candidates.extend(_prepared_dataset_candidates(options, checkpoint))
    for candidate in candidates:
        if candidate.is_file():
            payload = _load_json(candidate)
            if not isinstance(payload, dict):
                raise ValueError(f"Prepared dataset metadata must be an object: {candidate}")
            return payload, candidate.resolve(), False
    if source is not None:
        raise FileNotFoundError(
            "Prepared dataset metadata does not exist. Pass a prepared directory or "
            f"dataset.json, not {source!r}."
        )
    if _is_unmodified_template_checkpoint(checkpoint, package_template):
        return {
            "format": "inflect_release_frontend_compatibility_v1",
            "language": "en-us",
            "sample_rate": 24000,
            "frontend": {
                "type": "espeak",
                "language": "en-us",
                "preserve_punctuation": True,
                "with_stress": True,
            },
        }, None, True
    raise ValueError(
        "A non-template checkpoint requires prepared dataset frontend metadata. Pass "
        "ExportOptions(prepared_dataset=...) or, through the existing CLI, pass "
        "--symbols PREPARED_DIR/symbols.json so sibling dataset.json can be resolved. "
        "Export refuses to reuse the release English frontend for an adapted checkpoint."
    )


def _validate_prepared_symbols(
    dataset_path: Path | None,
    symbols: Sequence[str],
) -> dict[str, Any]:
    if dataset_path is None:
        return status(
            True,
            "Prepared symbol inventory check skipped for inline or release metadata",
            skipped=True,
        )
    symbols_path = dataset_path.parent / "symbols.json"
    if not symbols_path.is_file():
        raise FileNotFoundError(
            f"Prepared dataset is missing its ordered symbol inventory: {symbols_path}"
        )
    prepared_symbols = _symbols_from_payload(_load_json(symbols_path))
    return status(
        list(prepared_symbols) == list(symbols),
        "Export symbols exactly match the prepared dataset inventory",
        prepared_symbol_count=len(prepared_symbols),
        export_symbol_count=len(symbols),
    )


def _custom_hook_contract(
    options: ExportOptions,
    frontend: Mapping[str, Any],
    language: str,
    symbols: Sequence[str],
) -> tuple[dict[str, Any], Path]:
    hook_metadata = frontend.get("hook")
    if not isinstance(hook_metadata, Mapping):
        raise ValueError(
            "Custom prepared frontend metadata has no reproducibility record. "
            "Re-run dataset preparation with an explicit custom hook."
        )
    if options.frontend_hook is None:
        raise ValueError(
            "Custom frontend export requires ExportOptions(frontend_hook=PATH_TO_HOOK.py). "
            "The hook is copied into the package; export will never substitute English."
        )
    hook_file = Path(options.frontend_hook).expanduser().resolve()
    if not hook_file.is_file() or hook_file.suffix.lower() != ".py":
        raise ValueError(f"Custom frontend hook must be an existing .py file: {hook_file}")
    expected_source_hash = hook_metadata.get("source_sha256")
    actual_source_hash = sha256_file(hook_file)
    if expected_source_hash != actual_source_hash:
        raise ValueError(
            "Custom frontend hook source does not match prepared dataset metadata: "
            f"expected {expected_source_hash}, got {actual_source_hash}."
        )
    identity = str(hook_metadata.get("identity", ""))
    _, separator, factory_name = identity.rpartition(":")
    if not separator or not factory_name.isidentifier():
        raise ValueError(
            "Prepared custom frontend identity does not name a valid top-level factory."
        )
    from .frontend import (
        FrontendOptions,
        custom_frontend_metadata,
        custom_frontend_symbols,
    )

    frontend_options = FrontendOptions(
        mode="custom",
        language=language,
        hook=f"{hook_file}:{factory_name}",
    )
    current = custom_frontend_metadata(frontend_options)
    assert current is not None
    for field in (
        "source_sha256",
        "metadata_sha256",
        "factory_invocation",
        "declared_metadata",
        "declared_symbol_count",
    ):
        if current.get(field) != hook_metadata.get(field):
            raise ValueError(
                f"Custom frontend {field} differs from prepared dataset metadata."
            )
    declared_symbols = custom_frontend_symbols(frontend_options) or ()
    missing_symbols = sorted(set(declared_symbols).difference(symbols))
    if missing_symbols:
        raise ValueError(
            "Custom frontend declares symbols absent from the prepared/export "
            f"inventory: {missing_symbols[:16]!r}."
        )
    return {
        "path": "frontend_hook.py",
        "factory": factory_name,
        "source_sha256": actual_source_hash,
        "metadata_sha256": current["metadata_sha256"],
        "declared_metadata": current["declared_metadata"],
    }, hook_file


def _deployment_frontend_contract(
    options: ExportOptions,
    checkpoint: Path,
    package_template: Path | None,
    config: Mapping[str, Any],
    symbols: Sequence[str],
) -> tuple[dict[str, Any], Path | None, list[dict[str, Any]]]:
    dataset, dataset_path, release_compatibility = _load_prepared_dataset(
        options,
        checkpoint,
        package_template,
    )
    if dataset.get("format") not in {
        "inflect_prepared_dataset_v1",
        "inflect_release_frontend_compatibility_v1",
    }:
        raise ValueError(
            "Unsupported prepared dataset format. Expected inflect_prepared_dataset_v1."
        )
    language = dataset.get("language")
    frontend = dataset.get("frontend")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("Prepared dataset must declare a non-empty language.")
    if not isinstance(frontend, Mapping):
        raise ValueError("Prepared dataset must contain a frontend metadata object.")
    mode = frontend.get("type", frontend.get("mode"))
    if mode not in {"espeak", "prephonemized", "custom"}:
        raise ValueError(
            "Prepared frontend type must be espeak, prephonemized, or custom."
        )
    frontend_language = frontend.get("language", language)
    if frontend_language != language:
        raise ValueError(
            "Prepared dataset language and frontend language differ: "
            f"{language!r} != {frontend_language!r}."
        )
    sample_rate = dataset.get("sample_rate")
    configured_rate = config.get("data", {}).get("sampling_rate")
    if int(sample_rate or 0) != int(configured_rate or 0):
        raise ValueError(
            "Prepared dataset sample rate does not match model config: "
            f"{sample_rate!r} != {configured_rate!r}."
        )
    symbol_check = _validate_prepared_symbols(dataset_path, symbols)
    if not symbol_check["ok"]:
        raise ValueError(symbol_check["message"])
    hook_contract: dict[str, Any] | None = None
    hook_file: Path | None = None
    if mode == "custom":
        hook_contract, hook_file = _custom_hook_contract(
            options,
            frontend,
            language,
            symbols,
        )
    elif options.frontend_hook is not None:
        raise ValueError("frontend_hook may only be supplied for a custom frontend.")
    dataset_hash = (
        sha256_file(dataset_path)
        if dataset_path is not None
        else _canonical_json_sha256(dataset)
    )
    contract = {
        "format": "inflect_deployment_frontend_v1",
        "mode": mode,
        "language": language,
        "preserve_punctuation": bool(frontend.get("preserve_punctuation", True)),
        "with_stress": bool(frontend.get("with_stress", True)),
        "accepts_prephonemized_input": True,
        "prepared_frontend": {
            "type": mode,
            "language": language,
            "preserve_punctuation": bool(
                frontend.get("preserve_punctuation", True)
            ),
            "with_stress": bool(frontend.get("with_stress", True)),
            "custom_hook": (
                {
                    "source_kind": frontend["hook"].get("source_kind"),
                    "source_sha256": frontend["hook"].get("source_sha256"),
                    "metadata_sha256": frontend["hook"].get("metadata_sha256"),
                    "factory_invocation": frontend["hook"].get(
                        "factory_invocation"
                    ),
                    "declared_metadata": frontend["hook"].get(
                        "declared_metadata"
                    ),
                    "declared_symbol_count": frontend["hook"].get(
                        "declared_symbol_count"
                    ),
                }
                if mode == "custom"
                else None
            ),
        },
        "prepared_dataset": {
            "format": dataset["format"],
            "dataset_json_sha256": dataset_hash,
            "source_manifest_sha256": dataset.get("source_manifest_sha256"),
            "release_compatibility": release_compatibility,
        },
        "custom_hook": hook_contract,
    }
    checks = [
        symbol_check,
        status(
            True,
            "Deployment frontend is explicit and language-aware",
            mode=mode,
            language=language,
            accepts_prephonemized_input=True,
            release_compatibility=release_compatibility,
        ),
    ]
    return contract, hook_file, checks


def _validate_config_and_symbols(
    config: Mapping[str, Any],
    symbols: Sequence[str],
    state: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    data = config.get("data")
    model = config.get("model")
    if not isinstance(data, Mapping) or not isinstance(model, Mapping):
        raise ValueError("Config must contain 'data' and 'model' objects.")
    sample_rate = data.get("sampling_rate")
    duplicates = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
    checks.append(
        status(
            True,
            "ordered symbol inventory preserved exactly",
            duplicate_symbols=duplicates,
            note=(
                "Duplicate rows are retained for published-checkpoint compatibility."
                if duplicates
                else None
            ),
        )
    )
    checks.append(
        status(
            isinstance(sample_rate, int) and sample_rate > 0,
            "sampling_rate is a positive integer",
            value=sample_rate,
        )
    )
    embedding = state.get("enc_p.emb.weight")
    if embedding is None:
        checks.append(status(False, "enc_p.emb.weight is missing from the checkpoint"))
    else:
        checks.append(
            status(
                embedding.ndim == 2 and embedding.shape[0] == len(symbols),
                "embedding rows match the ordered symbol inventory",
                embedding_shape=list(embedding.shape),
                symbol_count=len(symbols),
            )
        )
        expected_hidden = model.get("hidden_channels")
        checks.append(
            status(
                embedding.ndim == 2 and embedding.shape[1] == expected_hidden,
                "embedding width matches model.hidden_channels",
                embedding_width=embedding.shape[1] if embedding.ndim == 2 else None,
                configured_hidden_channels=expected_hidden,
            )
        )
    conv_pre = state.get("dec.conv_pre.weight")
    expected_inter = model.get("inter_channels")
    if conv_pre is not None:
        checks.append(
            status(
                conv_pre.ndim == 3 and conv_pre.shape[1] == expected_inter,
                "decoder input width matches model.inter_channels",
                decoder_shape=list(conv_pre.shape),
                configured_inter_channels=expected_inter,
            )
        )
    rates = model.get("upsample_rates")
    hop_length = data.get("hop_length")
    if isinstance(rates, list) and all(isinstance(item, int) for item in rates):
        product = 1
        for rate in rates:
            product *= rate
        checks.append(
            status(
                product == hop_length,
                "decoder upsample product matches data.hop_length",
                upsample_product=product,
                hop_length=hop_length,
            )
        )
    return checks


def _copy_public_runtime(template: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    runtime = template / "runtime"
    if not runtime.is_dir():
        raise FileNotFoundError(f"Package template has no runtime directory: {runtime}")
    shutil.copytree(
        runtime,
        destination / "runtime",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    copied.extend(path for path in (destination / "runtime").rglob("*") if path.is_file())
    for name in (
        "inference.py",
        "inflect_vits_frontend.py",
        "inflect_nano_v2_frontend.py",
        "requirements.txt",
        "requirements-tested.txt",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
    ):
        source = template / name
        if source.is_file():
            target = destination / name
            shutil.copy2(source, target)
            copied.append(target)
    return copied


_DEPLOYMENT_FRONTEND_SOURCE = r'''
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PACKAGE_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))

from text.symbols import symbols


class DeploymentFrontendError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrontendOutput:
    raw_text: str
    normalized_text: str
    phoneme_text: str


def _load_contract() -> dict[str, Any]:
    path = PACKAGE_ROOT / "frontend.json"
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentFrontendError(f"Could not load frontend contract: {exc}") from exc
    if contract.get("format") != "inflect_deployment_frontend_v1":
        raise DeploymentFrontendError("Unsupported or missing deployment frontend contract.")
    if contract.get("mode") not in {"espeak", "prephonemized", "custom"}:
        raise DeploymentFrontendError("Deployment frontend mode is invalid.")
    if not isinstance(contract.get("language"), str) or not contract["language"].strip():
        raise DeploymentFrontendError("Deployment frontend language is missing.")
    return contract


CONTRACT = _load_contract()
SYMBOLS = frozenset(symbols)
_ESPEAK_BACKEND: Any = None
_CUSTOM_FRONTEND: Any = None


def _normalize_generic(text: str) -> str:
    if not isinstance(text, str):
        raise DeploymentFrontendError("Text input must be a Unicode string.")
    if "\x00" in text:
        raise DeploymentFrontendError("Text input contains a null byte.")
    value = unicodedata.normalize("NFKC", text)
    value = "".join(" " if char in "\r\n\t" else char for char in value)
    value = "".join(
        char for char in value if not unicodedata.category(char).startswith("C")
    )
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise DeploymentFrontendError("Text input is empty after normalization.")
    return value


def _clean_phonemes(value: str) -> str:
    if not isinstance(value, str):
        raise DeploymentFrontendError("Phoneme input must be a Unicode string.")
    value = unicodedata.normalize("NFC", value)
    if "\x00" in value or any(char in "\r\n" for char in value):
        raise DeploymentFrontendError("Phoneme input contains unsupported controls.")
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise DeploymentFrontendError("Phoneme input is empty.")
    unknown = sorted(set(value).difference(SYMBOLS))
    if unknown:
        rendered = ", ".join(repr(char) for char in unknown[:16])
        raise DeploymentFrontendError(
            "Phoneme input contains symbols absent from this checkpoint: " + rendered
        )
    return value


def _configure_espeak() -> None:
    candidates = (
        Path("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"),
        Path("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1"),
        Path("/usr/lib64/libespeak-ng.so.1"),
    )
    system = next((path for path in candidates if path.is_file()), None)
    if system is not None:
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(system))
        return
    try:
        import espeakng_loader

        os.environ.setdefault(
            "PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path()
        )
        os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
        espeakng_loader.make_library_available()
        espeakng_loader.load_library()
    except (ImportError, OSError, RuntimeError) as exc:
        raise DeploymentFrontendError(
            "Could not initialize eSpeak NG. Install espeakng-loader or a system "
            "eSpeak NG library."
        ) from exc


def _espeak() -> Any:
    global _ESPEAK_BACKEND
    if _ESPEAK_BACKEND is not None:
        return _ESPEAK_BACKEND
    _configure_espeak()
    try:
        from phonemizer.backend import EspeakBackend

        _ESPEAK_BACKEND = EspeakBackend(
            language=CONTRACT["language"],
            preserve_punctuation=bool(CONTRACT["preserve_punctuation"]),
            with_stress=bool(CONTRACT["with_stress"]),
            language_switch="remove-flags",
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        raise DeploymentFrontendError(
            f"Could not create eSpeak frontend for {CONTRACT['language']!r}: {exc}"
        ) from exc
    return _ESPEAK_BACKEND


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _custom() -> Any:
    global _CUSTOM_FRONTEND
    if _CUSTOM_FRONTEND is not None:
        return _CUSTOM_FRONTEND
    record = CONTRACT.get("custom_hook")
    if not isinstance(record, dict):
        raise DeploymentFrontendError("Custom frontend package metadata is missing.")
    hook_path = (PACKAGE_ROOT / str(record.get("path", ""))).resolve()
    try:
        hook_path.relative_to(PACKAGE_ROOT)
    except ValueError as exc:
        raise DeploymentFrontendError("Custom hook path escapes the package.") from exc
    if hook_path.suffix.lower() != ".py" or not hook_path.is_file():
        raise DeploymentFrontendError(f"Packaged custom hook is missing: {hook_path.name}")
    digest = hashlib.sha256(hook_path.read_bytes()).hexdigest()
    if digest != record.get("source_sha256"):
        raise DeploymentFrontendError("Packaged custom hook hash verification failed.")
    spec = importlib.util.spec_from_file_location("_inflect_package_frontend", hook_path)
    if spec is None or spec.loader is None:
        raise DeploymentFrontendError("Could not load packaged custom frontend.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, str(record.get("factory", "")), None)
    if not callable(factory):
        raise DeploymentFrontendError("Packaged custom frontend factory is unavailable.")
    try:
        signature = inspect.signature(factory)
        try:
            signature.bind(language=CONTRACT["language"])
            implementation = factory(language=CONTRACT["language"])
        except TypeError:
            signature.bind()
            implementation = factory()
    except Exception as exc:
        raise DeploymentFrontendError(f"Custom frontend factory failed: {exc}") from exc
    for method in ("normalize", "phonemize", "metadata"):
        if not callable(getattr(implementation, method, None)):
            raise DeploymentFrontendError(
                f"Custom frontend does not provide callable {method}()."
            )
    metadata = implementation.metadata()
    if _canonical_hash(metadata) != record.get("metadata_sha256"):
        raise DeploymentFrontendError("Custom frontend metadata hash verification failed.")
    _CUSTOM_FRONTEND = implementation
    return implementation


def process_input(
    text: str | None = None,
    *,
    phonemes: str | None = None,
) -> FrontendOutput:
    raw_text = text or ""
    if phonemes is not None:
        normalized = _normalize_generic(text) if text and text.strip() else ""
        return FrontendOutput(raw_text, normalized, _clean_phonemes(phonemes))
    mode = CONTRACT["mode"]
    if mode == "prephonemized":
        raise DeploymentFrontendError(
            "This checkpoint uses a prephonemized frontend. Supply phonemes=... "
            "or use inference.py --phonemes."
        )
    if text is None:
        raise DeploymentFrontendError("Text input is required.")
    if mode == "custom":
        implementation = _custom()
        try:
            normalized = _normalize_generic(implementation.normalize(text))
            phoneme_text = implementation.phonemize(normalized)
        except Exception as exc:
            raise DeploymentFrontendError(f"Custom frontend failed: {exc}") from exc
    else:
        normalized = _normalize_generic(text)
        try:
            from phonemizer.separator import Separator

            phoneme_text = _espeak().phonemize(
                [normalized],
                separator=Separator(phone="", word=" ", syllable=""),
                strip=True,
                njobs=1,
            )[0]
        except (RuntimeError, ValueError, OSError) as exc:
            raise DeploymentFrontendError(
                f"eSpeak failed for language {CONTRACT['language']!r}: {exc}"
            ) from exc
    return FrontendOutput(raw_text, normalized, _clean_phonemes(phoneme_text))
'''


_VITS_FRONTEND_SOURCE = r'''
from __future__ import annotations

from deployment_frontend import FrontendOutput, process_input


VitsFrontendOutput = FrontendOutput


def run_vits_frontend(
    text: str | None = None,
    *,
    phonemes: str | None = None,
) -> VitsFrontendOutput:
    return process_input(text, phonemes=phonemes)


def run_vits_frontend_batch(
    texts: list[str],
    *,
    jobs: int = 1,
) -> list[VitsFrontendOutput]:
    del jobs
    return [process_input(text) for text in texts]
'''


_INFERENCE_SOURCE = r'''
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


PACKAGE_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PACKAGE_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

import commons
import utils
from inflect_vits_frontend import run_vits_frontend
from models import SynthesizerTrn
from text import cleaned_text_to_sequence
from text.symbols import symbols


def split_text(text: str, limit: int = 280) -> list[str]:
    normalized = " ".join(text.split())
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?;:。！？；：])\s*", normalized)
        if part.strip()
    ]
    chunks: list[str] = []
    for sentence in sentences or [normalized]:
        while len(sentence) > limit:
            search = sentence[: limit + 1]
            punctuation = max(search.rfind(mark) for mark in (",", ";", ":", "，", "；", "："))
            split_at = (
                punctuation + 1
                if punctuation >= limit // 2
                else sentence.rfind(" ", 0, limit + 1)
            )
            if split_at < limit // 2:
                split_at = limit
            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def boundary_pause_seconds(chunk: str) -> float:
    ending = chunk.rstrip()[-1:] if chunk.strip() else ""
    return {
        "?": 0.28, "？": 0.28, "!": 0.24, "！": 0.24,
        ".": 0.22, "。": 0.22, ";": 0.16, "；": 0.16,
        ":": 0.13, "：": 0.13, ",": 0.09, "，": 0.09,
    }.get(ending, 0.08)


def edge_fade(
    waveform: np.ndarray,
    sample_rate: int,
    milliseconds: float = 5.0,
) -> np.ndarray:
    frames = min(round(sample_rate * milliseconds / 1000.0), waveform.size // 2)
    if frames <= 0:
        return waveform
    output = waveform.copy()
    ramp = np.linspace(0.0, 1.0, frames, endpoint=True, dtype=np.float32)
    output[:frames] *= ramp
    output[-frames:] *= ramp[::-1]
    return output


class InflectTTS:
    def __init__(
        self,
        model_dir: str | Path = PACKAGE_ROOT,
        device: str = "cpu",
    ) -> None:
        self.root = Path(model_dir).resolve()
        self.device = torch.device(device)
        self.hps = utils.get_hparams_from_file(str(self.root / "config.json"))
        self.model = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model,
        ).to(self.device).eval()
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        try:
            root_logger.setLevel(logging.WARNING)
            utils.load_checkpoint(str(self.root / "model.pth"), self.model, None)
        finally:
            root_logger.setLevel(previous_level)
        self.sample_rate = int(self.hps.data.sampling_rate)

    def _tokens(
        self,
        text: str | None = None,
        *,
        phonemes: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = run_vits_frontend(text, phonemes=phonemes)
        sequence = cleaned_text_to_sequence(output.phoneme_text)
        if self.hps.data.add_blank:
            sequence = commons.intersperse(sequence, 0)
        if not sequence:
            raise ValueError("The deployment frontend produced no speakable tokens.")
        tokens = torch.LongTensor(sequence).to(self.device).unsqueeze(0)
        lengths = torch.LongTensor([tokens.size(1)]).to(self.device)
        return tokens, lengths

    @torch.inference_mode()
    def synthesize(
        self,
        text: str | None = None,
        *,
        phonemes: str | None = None,
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 0,
    ) -> tuple[int, np.ndarray]:
        if phonemes is None:
            normalized = " ".join((text or "").split())
            if not normalized:
                raise ValueError("Text must not be empty.")
            chunks: list[tuple[str | None, str | None]] = [
                (chunk, None) for chunk in split_text(normalized)
            ]
        else:
            chunks = [(text, phonemes)]
        if not 0.5 <= speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        if not 0.0 <= variation <= 1.0:
            raise ValueError("variation must be between 0.0 and 1.0")
        pieces: list[np.ndarray] = []
        for index, (chunk_text, chunk_phonemes) in enumerate(chunks):
            if index:
                previous = chunks[index - 1][0] or ""
                pieces.append(
                    np.zeros(
                        round(self.sample_rate * boundary_pause_seconds(previous)),
                        dtype=np.float32,
                    )
                )
            tokens, lengths = self._tokens(chunk_text, phonemes=chunk_phonemes)
            torch.manual_seed(seed + index)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed + index)
            waveform = self.model.infer(
                tokens,
                lengths,
                noise_scale=variation,
                noise_scale_w=0.8,
                length_scale=1.0 / speed,
                max_len=4000,
            )[0][0, 0].float().cpu().numpy()
            pieces.append(edge_fade(waveform, self.sample_rate))
        return self.sample_rate, np.clip(np.concatenate(pieces), -1.0, 1.0)

    def save(
        self,
        text: str | None,
        output: str | Path,
        **kwargs: object,
    ) -> Path:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sample_rate, waveform = self.synthesize(text, **kwargs)
        sf.write(destination, waveform, sample_rate)
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone Inflect synthesis.")
    parser.add_argument("--model-dir", type=Path, default=PACKAGE_ROOT)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--phonemes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--variation", type=float, default=0.667)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    engine = InflectTTS(args.model_dir, args.device)
    engine.save(
        args.text,
        args.output,
        phonemes=args.phonemes,
        speed=args.speed,
        variation=args.variation,
        seed=args.seed,
    )
    print(f"wrote {args.output} at {engine.sample_rate} Hz")


if __name__ == "__main__":
    main()
'''


def _write_deployment_runtime(
    output: Path,
    hook_file: Path | None,
) -> list[Path]:
    paths = [
        output / "deployment_frontend.py",
        output / "inflect_vits_frontend.py",
        output / "inference.py",
    ]
    paths[0].write_text(
        textwrap.dedent(_DEPLOYMENT_FRONTEND_SOURCE).lstrip(),
        encoding="utf-8",
    )
    paths[1].write_text(
        textwrap.dedent(_VITS_FRONTEND_SOURCE).lstrip(),
        encoding="utf-8",
    )
    paths[2].write_text(
        textwrap.dedent(_INFERENCE_SOURCE).lstrip(),
        encoding="utf-8",
    )
    if hook_file is not None:
        destination = output / "frontend_hook.py"
        shutil.copy2(hook_file, destination)
        paths.append(destination)
    return paths


def _verify_deployment_runtime(
    output: Path,
    expected_contract: Mapping[str, Any],
    symbols: Sequence[str],
) -> dict[str, Any]:
    """Import the packaged frontend and prove its explicit-phoneme bypass works."""

    source_paths = (
        output / "deployment_frontend.py",
        output / "inflect_vits_frontend.py",
        output / "inference.py",
    )
    for source in source_paths:
        compile(source.read_text(encoding="utf-8"), str(source), "exec")

    module_names = (
        "deployment_frontend",
        "inflect_vits_frontend",
        "text",
        "text.symbols",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(output))
        sys.path.insert(0, str(output / "runtime"))
        for name in module_names:
            sys.modules.pop(name, None)
        generated_module_name = f"_inflect_deployment_frontend_{abs(hash(output))}"
        spec = importlib.util.spec_from_file_location(
            generated_module_name,
            output / "deployment_frontend.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not import generated deployment_frontend.py.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[generated_module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(generated_module_name, None)
        if module.CONTRACT != dict(expected_contract):
            raise RuntimeError("Generated runtime loaded a different frontend contract.")
        probe = next(
            (
                symbol
                for symbol in symbols
                if len(symbol) == 1 and not symbol.isspace()
            ),
            None,
        )
        if probe is None:
            raise RuntimeError("No single-character symbol is available for bypass test.")
        result = module.process_input(phonemes=probe)
        if result.phoneme_text != probe:
            raise RuntimeError("Prephonemized runtime bypass changed the supplied input.")
    finally:
        sys.path[:] = old_path
        for name in module_names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]

    return status(
        True,
        "Generated deployment runtime imports its language contract and accepts "
        "prephonemized input without invoking a text frontend",
        mode=expected_contract["mode"],
        language=expected_contract["language"],
        compiled_files=[path.name for path in source_paths],
    )


@contextmanager
def _runtime_imports(runtime_root: Path) -> Iterator[tuple[Any, Any]]:
    old_path = list(sys.path)
    names = ("models", "commons", "utils", "modules", "attentions", "transforms")
    previous = {name: sys.modules.get(name) for name in names}
    try:
        sys.path.insert(0, str(runtime_root))
        for name in names:
            sys.modules.pop(name, None)
        models = importlib.import_module("models")
        commons = importlib.import_module("commons")
        yield models, commons
    finally:
        sys.path[:] = old_path
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


def _build_model(
    runtime_root: Path,
    config: Mapping[str, Any],
    symbol_count: int,
    *,
    inference_only: bool,
) -> tuple[nn.Module, Any]:
    train = config["train"]
    data = config["data"]
    model_config = dict(config["model"])
    model_config["inference_only"] = inference_only
    with _runtime_imports(runtime_root) as (models, commons):
        model = models.SynthesizerTrn(
            symbol_count,
            int(data["filter_length"]) // 2 + 1,
            int(train["segment_size"]) // int(data["hop_length"]),
            **model_config,
        ).cpu().eval()
    return model, commons


class _DurationGraph(nn.Module):
    def __init__(self, model: nn.Module, commons_module: Any) -> None:
        super().__init__()
        self.enc_p = model.enc_p
        self.dp = model.dp
        self.commons = commons_module

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        length_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, means, logs, text_mask = self.enc_p(tokens, lengths)
        log_duration = self.dp(hidden, text_mask, g=None)
        durations = torch.ceil(torch.exp(log_duration) * text_mask * length_scale)
        output_lengths = torch.clamp_min(torch.sum(durations, [1, 2]), 1).long()
        output_mask = torch.unsqueeze(
            self.commons.sequence_mask(output_lengths, None), 1
        ).to(text_mask.dtype)
        attention_mask = torch.unsqueeze(text_mask, 2) * torch.unsqueeze(output_mask, -1)
        attention = self.commons.generate_path(durations, attention_mask)
        expanded_means = torch.matmul(attention.squeeze(1), means.transpose(1, 2)).transpose(1, 2)
        expanded_logs = torch.matmul(attention.squeeze(1), logs.transpose(1, 2)).transpose(1, 2)
        return expanded_means, expanded_logs, output_mask


class _DecodeGraph(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.flow = model.flow
        self.decoder = model.dec

    def forward(
        self,
        means: torch.Tensor,
        logs: torch.Tensor,
        output_mask: torch.Tensor,
        noise: torch.Tensor,
        noise_scale: torch.Tensor,
    ) -> torch.Tensor:
        latent = means + noise * torch.exp(logs) * noise_scale
        decoded = self.flow(latent, output_mask, g=None, reverse=True)
        return self.decoder(decoded * output_mask, g=None)


def _export_onnx(
    model: nn.Module,
    commons_module: Any,
    destination: Path,
    *,
    opset: int,
) -> tuple[list[Path], dict[str, Any]]:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export was requested, but 'onnx' is not installed. "
            "Install the toolkit with the 'onnx' extra."
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    duration_path = destination / "duration.onnx"
    decode_path = destination / "decode.onnx"
    duration = _DurationGraph(model, commons_module).eval()
    decode = _DecodeGraph(model).eval()
    tokens = torch.tensor([[0, 1, 0, 2, 0, 3, 0]], dtype=torch.long)
    lengths = torch.tensor([tokens.shape[1]], dtype=torch.long)
    length_scale = torch.tensor(1.0, dtype=torch.float32)
    torch.onnx.export(
        duration,
        (tokens, lengths, length_scale),
        duration_path,
        input_names=["tokens", "lengths", "length_scale"],
        output_names=["m_p_exp", "logs_p_exp", "y_mask"],
        dynamic_axes={
            "tokens": {1: "text_len"},
            "m_p_exp": {2: "audio_frames"},
            "logs_p_exp": {2: "audio_frames"},
            "y_mask": {2: "audio_frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    # These tensors are traced again as decode-graph inputs. inference_mode()
    # marks them as inference tensors, which older supported PyTorch versions
    # cannot save for backward while constructing the second ONNX graph.
    with torch.no_grad():
        means, logs, output_mask = duration(tokens, lengths, length_scale)
    noise = torch.zeros_like(means)
    noise_scale = torch.tensor(0.667, dtype=torch.float32)
    with torch.no_grad():
        reference_waveform = decode(
            means,
            logs,
            output_mask,
            noise,
            noise_scale,
        ).detach().cpu().numpy()
    torch.onnx.export(
        decode,
        (means, logs, output_mask, noise, noise_scale),
        decode_path,
        input_names=["m_p_exp", "logs_p_exp", "y_mask", "zp_noise", "noise_scale"],
        output_names=["waveform"],
        dynamic_axes={
            "m_p_exp": {2: "audio_frames"},
            "logs_p_exp": {2: "audio_frames"},
            "y_mask": {2: "audio_frames"},
            "zp_noise": {2: "audio_frames"},
            "waveform": {2: "waveform_samples"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(duration_path))
    onnx.checker.check_model(onnx.load(decode_path))
    provider_check: dict[str, Any]
    try:
        import onnxruntime as ort

        sessions = [
            ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            for path in (duration_path, decode_path)
        ]
        duration_outputs = sessions[0].run(
            None,
            {
                "tokens": tokens.numpy(),
                "lengths": lengths.numpy(),
                "length_scale": length_scale.numpy(),
            },
        )
        onnx_noise = np.zeros_like(duration_outputs[0], dtype=np.float32)
        onnx_waveform = sessions[1].run(
            None,
            {
                "m_p_exp": duration_outputs[0],
                "logs_p_exp": duration_outputs[1],
                "y_mask": duration_outputs[2],
                "zp_noise": onnx_noise,
                "noise_scale": noise_scale.numpy(),
            },
        )[0]
        reference_duration = [
            value.detach().cpu().numpy() for value in (means, logs, output_mask)
        ]
        duration_errors = [
            float(np.max(np.abs(actual - expected)))
            for actual, expected in zip(duration_outputs[:2], reference_duration[:2])
        ]
        mask_exact = bool(np.array_equal(duration_outputs[2], reference_duration[2]))
        waveform_delta = np.abs(onnx_waveform - reference_waveform)
        waveform_max_error = float(np.max(waveform_delta))
        waveform_mean_error = float(np.mean(waveform_delta))
        correlation = float(
            np.corrcoef(onnx_waveform.reshape(-1), reference_waveform.reshape(-1))[0, 1]
        )
        parity_ok = (
            all(error <= 1.0e-4 for error in duration_errors)
            and mask_exact
            and waveform_max_error <= 5.0e-4
            and correlation >= 0.9999
        )
        provider_check = status(
            parity_ok,
            "ONNX Runtime executed both graphs with PyTorch parity",
            inputs=[[item.name for item in session.get_inputs()] for session in sessions],
            outputs=[[item.name for item in session.get_outputs()] for session in sessions],
            parity={
                "duration_max_abs_error": duration_errors,
                "duration_mask_exact": mask_exact,
                "waveform_max_abs_error": waveform_max_error,
                "waveform_mean_abs_error": waveform_mean_error,
                "waveform_correlation": correlation,
                "tolerances": {
                    "duration_max_abs_error": 1.0e-4,
                    "waveform_max_abs_error": 5.0e-4,
                    "waveform_correlation_minimum": 0.9999,
                },
            },
        )
    except ImportError:
        provider_check = status(
            True,
            "ONNX checker passed; ONNX Runtime load check skipped because it is not installed",
            skipped=True,
        )
    return [duration_path, decode_path], provider_check


def export_checkpoint(options: ExportOptions) -> dict[str, Any]:
    """Export and verify an adapted checkpoint.

    Returns the same report written to ``export_report.json``. Existing output
    directories are rejected unless ``overwrite=True``.
    """

    checkpoint = Path(options.checkpoint).resolve()
    output = Path(options.output_dir).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if output.exists() and any(output.iterdir()):
        if not options.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    payload = _load_checkpoint(checkpoint)
    state, source_metadata, stripped_keys = _extract_state(payload)
    package_template = _resolve_package_template(options, payload)
    config, config_source = _resolve_config(options, checkpoint, package_template)
    symbols, symbols_source = _load_symbols(
        options,
        checkpoint,
        payload,
        package_template,
    )
    checks = _validate_config_and_symbols(config, symbols, state)
    frontend_contract, frontend_hook, frontend_checks = _deployment_frontend_contract(
        options,
        checkpoint,
        package_template,
        config,
        symbols,
    )
    checks.extend(frontend_checks)
    remaining_training_keys = [key for key in state if _is_training_only_name(key)]
    omitted_training_fields = sorted(
        str(key) for key in payload if _is_training_only_name(str(key))
    )
    checks.append(
        status(
            not remaining_training_keys,
            "Inference state contains no posterior encoder, discriminator, optimizer, "
            "scheduler, or scaler tensors",
            stripped_tensor_count=len(stripped_keys),
            stripped_tensor_keys=stripped_keys,
            omitted_top_level_fields=omitted_training_fields,
            remaining_training_only_keys=remaining_training_keys,
        )
    )
    failed = [check for check in checks if not check["ok"]]
    if failed:
        raise ValueError("Checkpoint/config/symbol validation failed: " + "; ".join(
            check["message"] for check in failed
        ))

    export_config = copy.deepcopy(config)
    export_config.setdefault("model", {})["inference_only"] = True
    export_config["deployment_frontend"] = copy.deepcopy(frontend_contract)
    deployable_parameters = sum(tensor.numel() for tensor in state.values())
    inference_payload = {
        "format": "inflect_vits_inference_checkpoint_v1",
        "model": state,
        "iteration": int(
            source_metadata.get("iteration", source_metadata.get("step", 0)) or 0
        ),
        "learning_rate": 0.0,
        "deployable_parameters": deployable_parameters,
        "adaptation": {
            "model_name": options.model_name,
            "source_revision": options.source_revision,
            "symbol_count": len(symbols),
            "frontend": copy.deepcopy(frontend_contract),
        },
    }
    model_path = output / "model.pth"
    torch.save(inference_payload, model_path)
    config_path = write_json(output / "config.json", export_config)
    symbols_path = write_json(
        output / "symbols.json",
        {
            "format": "inflect_v2_symbol_inventory_v1",
            "symbols": symbols,
            "count": len(symbols),
        },
    )
    frontend_path = write_json(output / "frontend.json", frontend_contract)
    produced = [model_path, config_path, symbols_path, frontend_path]

    strict_load_check = status(
        True,
        "Tensor checkpoint reloaded and symbol/config consistency passed",
        strict_model_load=False,
    )
    model: nn.Module | None = None
    commons_module: Any = None
    if package_template is not None:
        template = package_template
        produced.extend(_copy_public_runtime(template, output))
        produced.extend(_write_deployment_runtime(output, frontend_hook))
        runtime_symbols = output / "runtime" / "text" / "symbols.py"
        if runtime_symbols.is_file():
            runtime_symbols.write_text(
                "# Generated by inflect-finetune. Keep this order unchanged.\n"
                f"symbols = {symbols!r}\n"
                'SPACE_ID = symbols.index(" ") if " " in symbols else -1\n',
                encoding="utf-8",
            )
        checks.append(
            _verify_deployment_runtime(output, frontend_contract, symbols)
        )
        model, commons_module = _build_model(
            output / "runtime",
            export_config,
            len(symbols),
            inference_only=True,
        )
        incompatible = model.load_state_dict(state, strict=True)
        strict_load_check = status(
            not incompatible.missing_keys and not incompatible.unexpected_keys,
            "Runtime model instantiated and checkpoint loaded strictly",
            strict_model_load=True,
            missing_keys=list(incompatible.missing_keys),
            unexpected_keys=list(incompatible.unexpected_keys),
        )
        checks.append(strict_load_check)
        training_model, _ = _build_model(
            output / "runtime",
            export_config,
            len(symbols),
            inference_only=False,
        )
        training_incompatible = training_model.load_state_dict(state, strict=False)
        training_missing = list(training_incompatible.missing_keys)
        training_unexpected = list(training_incompatible.unexpected_keys)
        checks.append(
            status(
                bool(training_missing)
                and all(key.startswith("enc_q.") for key in training_missing)
                and not training_unexpected,
                "Training-form generator differs only by intentionally omitted enc_q.* tensors",
                missing_keys=training_missing,
                unexpected_keys=training_unexpected,
            )
        )
        del training_model
    elif options.verify:
        raise ValueError(
            "Verified export requires a released package runtime. Pass the released "
            "directory as ExportOptions(package_template=...), or --package-template "
            "on the command line. A training checkpoint cannot supply it: its saved "
            "options omit the base model, because they are hashed into the run "
            "identity that guards resume."
        )

    onnx_report: dict[str, Any] = {"requested": options.include_onnx}
    if options.include_onnx:
        if model is None or commons_module is None:
            raise ValueError("ONNX export requires ExportOptions(package_template=...).")
        onnx_files, onnx_check = _export_onnx(
            model,
            commons_module,
            output / "onnx",
            opset=options.onnx_opset,
        )
        produced.extend(onnx_files)
        checks.append(onnx_check)
        onnx_report.update(
            {
                "opset": options.onnx_opset,
                "files": [file_record(path, relative_to=output) for path in onnx_files],
                "verification": onnx_check,
            }
        )

    reloaded = torch.load(model_path, map_location="cpu", weights_only=True)
    reload_state, _, reloaded_stripped = _extract_state(reloaded)
    checks.append(
        status(
            list(reload_state) == list(state)
            and all(torch.equal(reload_state[key], state[key]) for key in state),
            "Saved inference checkpoint reloads with exact tensor parity",
            tensor_count=len(state),
            training_only_tensors_after_reload=reloaded_stripped,
        )
    )
    if any(not check["ok"] for check in checks):
        raise RuntimeError("One or more export verification checks failed.")

    checksum_path = write_checksums(
        output / "checksums.sha256",
        list(dict.fromkeys(produced)),
        relative_to=output,
    )
    produced = list(dict.fromkeys(produced))
    produced.append(checksum_path)
    report = make_report(
        "export_report",
        ok=True,
        model_name=options.model_name,
        source={
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256_file(checkpoint),
            "config": config_source.name if config_source else None,
            "symbols": symbols_source.name if symbols_source else "checkpoint_or_inline",
            "revision": options.source_revision,
        },
        output_dir=".",
        deployable_parameters=deployable_parameters,
        stripped_training_tensors={
            "count": len(stripped_keys),
            "keys": stripped_keys,
            "top_level_fields": omitted_training_fields,
        },
        symbol_count=len(symbols),
        sample_rate=int(export_config["data"]["sampling_rate"]),
        deployment_frontend=frontend_contract,
        checks=checks,
        onnx=onnx_report,
        files=[file_record(path, relative_to=output) for path in produced],
    )
    write_json(output / "export_report.json", report)
    return report
