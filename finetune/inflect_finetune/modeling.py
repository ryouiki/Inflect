"""Construction helpers for public Inflect v2 warm-start training."""

from __future__ import annotations

import importlib
import json
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator, Sequence

import torch
from torch import nn

from . import monotonic_align as training_monotonic_align
from .symbols import BASE_SYMBOLS as RELEASE_BASE_SYMBOLS


BASE_SYMBOL_COUNT = 178
_RELEASE_DUPLICATE_POSITIONS = {
    symbol: tuple(
        index for index, value in enumerate(RELEASE_BASE_SYMBOLS) if value == symbol
    )
    for symbol, count in Counter(RELEASE_BASE_SYMBOLS).items()
    if count > 1
}
_RUNTIME_MODULE_NAMES = (
    "attentions",
    "commons",
    "inflect_alias_free",
    "models",
    "modules",
    "monotonic_align",
    "transforms",
    "utils",
    "text",
    "text.cleaners",
    "text.symbols",
)


@dataclass(frozen=True)
class RuntimeComponents:
    """Objects imported from one released model's self-contained runtime."""

    synthesizer_class: type[nn.Module]
    discriminator_class: type[nn.Module]
    commons: ModuleType
    base_symbols: tuple[str, ...]
    runtime_root: Path


@dataclass(frozen=True)
class ModelBundle:
    generator: nn.Module
    discriminator: nn.Module
    config: dict
    symbols: tuple[str, ...]
    base_symbols: tuple[str, ...]
    components: RuntimeComponents


def repository_root() -> Path:
    """Return the repository root without relying on the process CWD."""

    return Path(__file__).resolve().parents[2]


def resolve_base_model(model: str | Path) -> Path:
    """Resolve a local release, Hugging Face repository ID, or model shorthand."""

    candidate = Path(model).expanduser()
    if candidate.is_dir():
        resolved = candidate.resolve()
    else:
        model_text = str(model).strip()
        name = model_text.lower().replace("\\", "/").rstrip("/").split("/")[-1]
        aliases = {
            "micro": "Inflect-Micro-v2",
            "inflect-micro-v2": "Inflect-Micro-v2",
            "nano": "Inflect-Nano-v2",
            "inflect-nano-v2": "Inflect-Nano-v2",
        }
        release_name = aliases.get(name)
        checkout = (
            repository_root() / "release_assets" / "hf_clean_download" / release_name
            if release_name is not None
            else None
        )
        if checkout is not None and checkout.is_dir():
            resolved = checkout.resolve()
        else:
            repo_id = f"owensong/{release_name}" if release_name is not None else model_text
            revision = None
            if "@" in repo_id:
                repo_id, revision = repo_id.rsplit("@", 1)
            if "/" not in repo_id or not all(part for part in repo_id.split("/", 1)):
                raise FileNotFoundError(
                    f"Base model {model!r} is neither a local directory, a Micro/Nano "
                    "shorthand, nor a Hugging Face repository ID such as "
                    "'owensong/Inflect-Micro-v2'."
                )
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise RuntimeError(
                    "Downloading a base model requires huggingface-hub. Install the "
                    "toolkit dependencies or pass a local model directory."
                ) from exc
            resolved = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    allow_patterns=[
                        "config.json",
                        "model.pth",
                        "runtime/**",
                        "inference.py",
                        "inflect*_frontend.py",
                        "requirements*.txt",
                        "LICENSE",
                        "THIRD_PARTY_NOTICES.md",
                    ],
                )
            ).resolve()
    required = ("config.json", "model.pth", "runtime")
    missing = [name for name in required if not (resolved / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{resolved} is not a complete Inflect release directory; missing {missing}."
        )
    return resolved


def validate_release_compatible_symbols(
    symbols: Sequence[str],
    *,
    source: str,
) -> tuple[str, ...]:
    """Return an inventory that extends the release inventory without altering it.

    An adaptation may append symbols, but the released rows must keep their
    exact strings and positions. Embedding migration maps rows by symbol string,
    so a reordered or altered prefix would silently move pretrained weights onto
    other phonemes.

    Both a prepared dataset's ``symbols.json`` and a base model's runtime
    inventory are held to this rule, so a checkpoint adapted with extra symbols
    can serve as the base of a later run.
    """

    values = list(symbols)
    if not values or not all(isinstance(item, str) for item in values):
        raise ValueError(f"{source} must contain a non-empty list of symbol strings.")
    if values[0] != "_":
        raise ValueError(f"{source} must begin with '_' at index 0.")
    if len(values) < BASE_SYMBOL_COUNT:
        raise ValueError(
            f"{source} has {len(values)} symbols and must preserve the "
            f"{BASE_SYMBOL_COUNT}-symbol release prefix."
        )
    if tuple(values[:BASE_SYMBOL_COUNT]) != tuple(RELEASE_BASE_SYMBOLS):
        raise ValueError(
            f"The first {BASE_SYMBOL_COUNT} symbols in {source} must exactly preserve the "
            "published Inflect v2 release inventory and indices."
        )
    duplicate_positions = {
        symbol: tuple(index for index, value in enumerate(values) if value == symbol)
        for symbol, count in Counter(values).items()
        if count > 1
    }
    if duplicate_positions != _RELEASE_DUPLICATE_POSITIONS:
        raise ValueError(
            f"{source} contains duplicate/custom-added symbols outside the "
            "release-compatible duplicate apostrophe at base indices 174 and 176: "
            f"{duplicate_positions}"
        )
    return tuple(values)


def load_symbols(path: str | Path) -> tuple[str, ...]:
    """Load an ordered symbol inventory written by the preparation workflow."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = payload.get("symbols", payload.get("ordered_symbols"))
    else:
        values = None
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"{source} must contain a JSON symbol list or a 'symbols' list.")
    return validate_release_compatible_symbols(values, source=str(source))


@contextmanager
def _isolated_runtime_import(runtime_root: Path) -> Iterator[None]:
    """Temporarily isolate VITS' historical absolute module imports."""

    previous = {name: sys.modules.get(name) for name in _RUNTIME_MODULE_NAMES}
    for name in _RUNTIME_MODULE_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(runtime_root))
    try:
        yield
    finally:
        if sys.path and sys.path[0] == str(runtime_root):
            sys.path.pop(0)
        for name in _RUNTIME_MODULE_NAMES:
            sys.modules.pop(name, None)
        for name, module in previous.items():
            if module is not None:
                sys.modules[name] = module


def load_runtime_components(base_model: str | Path) -> RuntimeComponents:
    """Import training-capable classes from the exact released runtime."""

    model_root = resolve_base_model(base_model)
    runtime_root = model_root / "runtime"
    with _isolated_runtime_import(runtime_root):
        models = importlib.import_module("models")
        commons = importlib.import_module("commons")
        symbol_module = importlib.import_module("text.symbols")
        # The release bundle intentionally contains an inference-only stub.
        # Replace only that module global with the public training operation.
        models.monotonic_align = training_monotonic_align
        synthesizer = models.SynthesizerTrn
        discriminator = models.MultiPeriodDiscriminator
        symbols = tuple(symbol_module.symbols)
    # The base may be a release or a checkpoint this toolkit adapted, and an
    # adaptation is allowed to append symbols. What it may not do is disturb the
    # released prefix, because migration copies rows by symbol identity.
    try:
        symbols = validate_release_compatible_symbols(
            symbols, source=f"{runtime_root / 'text' / 'symbols.py'}"
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return RuntimeComponents(
        synthesizer_class=synthesizer,
        discriminator_class=discriminator,
        commons=commons,
        base_symbols=symbols,
        runtime_root=runtime_root,
    )


def load_release_config(base_model: str | Path) -> dict:
    model_root = resolve_base_model(base_model)
    payload = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    for section in ("train", "data", "model"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"Release config is missing the {section!r} object.")
    return payload


def build_training_models(
    base_model: str | Path,
    symbols: Sequence[str],
    *,
    seed: int = 1234,
) -> ModelBundle:
    """Construct the released generator in training mode plus a fresh MPD."""

    model_root = resolve_base_model(base_model)
    config = load_release_config(model_root)
    components = load_runtime_components(model_root)
    model_kwargs = dict(config["model"])
    model_kwargs["inference_only"] = False
    # These release fields are retained even where a runtime version consumes
    # them through **kwargs; they are part of the architecture contract.
    model_kwargs["n_layers_q"] = int(model_kwargs.get("n_layers_q", 3))
    model_kwargs["n_speakers"] = int(config["data"].get("n_speakers", 0))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        generator = components.synthesizer_class(
            len(symbols),
            int(config["data"]["filter_length"]) // 2 + 1,
            int(config["train"]["segment_size"]) // int(config["data"]["hop_length"]),
            **model_kwargs,
        )
        discriminator = components.discriminator_class(
            bool(model_kwargs.get("use_spectral_norm", False))
        )
    return ModelBundle(
        generator=generator,
        discriminator=discriminator,
        config=config,
        symbols=tuple(symbols),
        base_symbols=components.base_symbols,
        components=components,
    )


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def optimizer_parameters(module: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters remain after applying the freeze policy.")
    return parameters
