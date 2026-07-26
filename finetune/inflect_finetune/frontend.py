"""Configurable eSpeak, prephonemized, and explicit custom frontends."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Mapping, Sequence


class FrontendError(RuntimeError):
    """Raised when text normalization or phonemization cannot be completed."""


FrontendMode = Literal["espeak", "prephonemized", "custom"]


@dataclass(frozen=True)
class FrontendOptions:
    """Frontend configuration shared by preparation and future CLI commands."""

    mode: FrontendMode = "espeak"
    language: str = "en-us"
    preserve_punctuation: bool = True
    with_stress: bool = True
    hook: str | None = None

    def validate(self) -> None:
        """Validate options without loading eSpeak."""
        if self.mode not in {"espeak", "prephonemized", "custom"}:
            raise ValueError("mode must be 'espeak', 'prephonemized', or 'custom'.")
        if not self.language.strip():
            raise ValueError("language cannot be empty.")
        if self.mode == "custom" and not (self.hook and self.hook.strip()):
            raise ValueError(
                "Custom frontend mode requires an explicit hook in "
                "'module:callable' or 'file.py:function' form."
            )
        if self.mode != "custom" and self.hook:
            raise ValueError("frontend hook may only be supplied when mode is 'custom'.")


@dataclass(frozen=True)
class FrontendResult:
    """Normalized transcript and the character-level model input string."""

    raw_text: str
    normalized_text: str
    phonemes: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class CustomFrontend:
    """Validated custom frontend instance plus its reproducibility record."""

    identity: str
    source_kind: str
    source_sha256: str
    metadata_sha256: str
    invocation: str
    metadata: dict[str, Any]
    symbols: tuple[str, ...]
    implementation: Any

    def public_metadata(self) -> dict[str, Any]:
        """Return metadata safe to persist in a prepared dataset."""
        return {
            "identity": self.identity,
            "source_kind": self.source_kind,
            "source_sha256": self.source_sha256,
            "metadata_sha256": self.metadata_sha256,
            "factory_invocation": self.invocation,
            "declared_metadata": self.metadata,
            "declared_symbol_count": len(self.symbols),
        }


_CONFIGURED = False
_BACKENDS: dict[tuple[str, bool, bool], Any] = {}
_CUSTOM_FRONTENDS: dict[tuple[str, str], CustomFrontend] = {}
_IMPORT_PATH = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_CALLABLE_NAME = re.compile(r"^[A-Za-z_]\w*$")


def normalize_text(text: str) -> str:
    """Apply language-neutral Unicode and whitespace normalization.

    Language-specific number and abbreviation expansion should be supplied by
    a custom frontend rather than silently applying English rules.
    """
    if not isinstance(text, str):
        raise FrontendError(f"Transcript must be a string, got {type(text).__name__}.")
    if "\x00" in text:
        raise FrontendError("Transcript contains a null byte.")
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(
        " " if character in {"\r", "\n", "\t"} else character for character in normalized
    )
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("C")
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise FrontendError("Transcript is empty after Unicode normalization.")
    return normalized


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_hash(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FrontendError(
            f"Custom frontend metadata must be finite and JSON-serializable: {exc}"
        ) from exc
    return _sha256_bytes(encoded)


def _source_hash(factory: Any, module: ModuleType, explicit_file: Path | None) -> str:
    source_path = explicit_file
    if source_path is None:
        candidate = inspect.getsourcefile(factory) or getattr(module, "__file__", None)
        source_path = Path(candidate).resolve() if candidate else None
    if source_path and source_path.is_file():
        return _sha256_bytes(source_path.read_bytes())
    try:
        return _sha256_bytes(inspect.getsource(factory).encode("utf-8"))
    except (OSError, TypeError) as exc:
        raise FrontendError(
            "The custom frontend callable has no readable source to fingerprint. "
            "Use a Python module or .py file with available source."
        ) from exc


def _load_hook_callable(specification: str) -> tuple[Any, str, str, str]:
    source, separator, callable_name = specification.rpartition(":")
    source = source.strip()
    callable_name = callable_name.strip()
    if not separator:
        raise FrontendError(
            "Custom frontend hook must include ':' in "
            "'module:callable' or 'file.py:function' form."
        )
    if not source or not _CALLABLE_NAME.fullmatch(callable_name):
        raise FrontendError(
            "Custom frontend hook must name one top-level callable, for example "
            "'my_frontend:create_frontend' or './frontend.py:create_frontend'."
        )

    explicit_file: Path | None = None
    if source.lower().endswith(".py"):
        explicit_file = Path(source).expanduser().resolve()
        if not explicit_file.is_file():
            raise FrontendError(f"Custom frontend file does not exist: {explicit_file}")
        module_name = (
            "_inflect_custom_frontend_"
            + hashlib.sha256(str(explicit_file).encode("utf-8")).hexdigest()[:16]
        )
        module_spec = importlib.util.spec_from_file_location(module_name, explicit_file)
        if module_spec is None or module_spec.loader is None:
            raise FrontendError(f"Could not load custom frontend file: {explicit_file}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise FrontendError(
                f"Custom frontend file raised while loading: {explicit_file}: {exc}"
            ) from exc
        identity = f"file:{explicit_file.name}:{callable_name}"
        source_kind = "file"
    else:
        if not _IMPORT_PATH.fullmatch(source):
            raise FrontendError(
                "Custom frontend module must be a dotted Python import path, or the "
                "source must be a .py file."
            )
        try:
            module = importlib.import_module(source)
        except Exception as exc:
            raise FrontendError(
                f"Could not import custom frontend module '{source}': {exc}"
            ) from exc
        identity = f"{source}:{callable_name}"
        source_kind = "module"

    factory = getattr(module, callable_name, None)
    if not callable(factory):
        raise FrontendError(f"Custom frontend hook '{identity}' is not callable.")
    return factory, identity, source_kind, _source_hash(factory, module, explicit_file)


def _create_custom_frontend(options: FrontendOptions) -> CustomFrontend:
    options.validate()
    assert options.hook is not None
    cache_key = (options.hook, options.language)
    if cache_key in _CUSTOM_FRONTENDS:
        return _CUSTOM_FRONTENDS[cache_key]

    factory, identity, source_kind, source_sha256 = _load_hook_callable(options.hook)
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError) as exc:
        raise FrontendError(
            f"Could not inspect custom frontend factory '{identity}': {exc}"
        ) from exc
    try:
        signature.bind(language=options.language)
        invocation = "factory(language=<configured language>)"
    except TypeError:
        try:
            signature.bind()
        except TypeError as exc:
            raise FrontendError(
                f"Custom frontend factory '{identity}' must accept either no arguments "
                "or the keyword argument 'language'."
            ) from exc
        invocation = "factory()"
    try:
        implementation = (
            factory(language=options.language)
            if invocation.startswith("factory(language=")
            else factory()
        )
    except Exception as exc:
        raise FrontendError(
            f"Custom frontend factory '{identity}' failed: {exc}"
        ) from exc

    for method_name in ("normalize", "phonemize", "symbols", "metadata"):
        if not callable(getattr(implementation, method_name, None)):
            raise FrontendError(
                f"Custom frontend '{identity}' must provide callable "
                f"{method_name}(...)."
            )
    try:
        metadata = implementation.metadata()
        declared_symbols = implementation.symbols()
    except Exception as exc:
        raise FrontendError(
            f"Custom frontend '{identity}' failed while declaring metadata or symbols: {exc}"
        ) from exc
    if not isinstance(metadata, Mapping):
        raise FrontendError(
            f"Custom frontend '{identity}' metadata() must return a mapping."
        )
    metadata_dict = dict(metadata)
    required_metadata = ("name", "version", "language", "configuration")
    missing = [field for field in required_metadata if field not in metadata_dict]
    if missing:
        raise FrontendError(
            f"Custom frontend '{identity}' metadata() is missing: {', '.join(missing)}."
        )
    if not isinstance(declared_symbols, Sequence) or isinstance(
        declared_symbols, (str, bytes)
    ):
        raise FrontendError(
            f"Custom frontend '{identity}' symbols() must return an ordered sequence."
        )
    symbols = tuple(declared_symbols)
    if not symbols or any(
        not isinstance(symbol, str) or len(symbol) != 1 for symbol in symbols
    ):
        raise FrontendError(
            f"Custom frontend '{identity}' symbols() must contain one-character strings."
        )
    if len(symbols) != len(set(symbols)):
        raise FrontendError(f"Custom frontend '{identity}' symbols() contains duplicates.")
    normalized_symbols = [unicodedata.normalize("NFC", symbol) for symbol in symbols]
    if len(normalized_symbols) != len(set(normalized_symbols)):
        raise FrontendError(
            f"Custom frontend '{identity}' symbols() has Unicode normalization collisions."
        )

    result = CustomFrontend(
        identity=identity,
        source_kind=source_kind,
        source_sha256=source_sha256,
        metadata_sha256=_json_hash(metadata_dict),
        invocation=invocation,
        metadata=metadata_dict,
        symbols=symbols,
        implementation=implementation,
    )
    _CUSTOM_FRONTENDS[cache_key] = result
    return result


def _hook_text(value: Any, *, field: str, identity: str) -> str:
    if not isinstance(value, str):
        raise FrontendError(
            f"Custom frontend '{identity}' {field} must return a Unicode string."
        )
    normalized = unicodedata.normalize("NFC", value)
    if "\x00" in normalized or any(
        unicodedata.category(character).startswith("C") and character != "\t"
        for character in normalized
    ):
        raise FrontendError(
            f"Custom frontend '{identity}' {field} returned unsupported control characters."
        )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise FrontendError(
            f"Custom frontend '{identity}' {field} returned an empty string."
        )
    return normalized


def custom_frontend_metadata(options: FrontendOptions) -> dict[str, Any] | None:
    """Return a reproducibility record for an explicitly configured custom hook."""
    if options.mode != "custom":
        return None
    return _create_custom_frontend(options).public_metadata()


def custom_frontend_symbols(options: FrontendOptions) -> tuple[str, ...] | None:
    """Return the ordered symbol inventory declared by a custom frontend."""
    if options.mode != "custom":
        return None
    return _create_custom_frontend(options).symbols


def _configure_espeak() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    candidates = (
        Path("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"),
        Path("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1"),
        Path("/usr/lib64/libespeak-ng.so.1"),
    )
    system_library = next((candidate for candidate in candidates if candidate.is_file()), None)
    try:
        if system_library:
            os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(system_library))
        else:
            import espeakng_loader

            os.environ.setdefault(
                "PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path()
            )
            os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
            espeakng_loader.make_library_available()
            espeakng_loader.load_library()
    except (ImportError, OSError, RuntimeError) as exc:
        raise FrontendError(
            "Could not initialize eSpeak NG. Install the system eSpeak NG library or "
            "the declared espeakng-loader dependency."
        ) from exc
    _CONFIGURED = True


def _backend(options: FrontendOptions) -> Any:
    options.validate()
    _configure_espeak()
    key = (options.language, options.preserve_punctuation, options.with_stress)
    if key in _BACKENDS:
        return _BACKENDS[key]
    try:
        from phonemizer.backend import EspeakBackend

        backend = EspeakBackend(
            language=options.language,
            preserve_punctuation=options.preserve_punctuation,
            with_stress=options.with_stress,
            language_switch="remove-flags",
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        raise FrontendError(
            f"Could not create the eSpeak frontend for language '{options.language}'. "
            "Confirm the language with `espeak-ng --voices` and install phonemizer."
        ) from exc
    _BACKENDS[key] = backend
    return backend


def phonemize_text(text: str, options: FrontendOptions | None = None) -> str:
    """Phonemize normalized text into the character sequence consumed by Inflect."""
    options = options or FrontendOptions()
    if options.mode != "espeak":
        raise FrontendError("phonemize_text requires frontend mode 'espeak'.")
    normalized = normalize_text(text)
    try:
        from phonemizer.separator import Separator

        result = _backend(options).phonemize(
            [normalized],
            separator=Separator(phone="", word=" ", syllable=""),
            strip=True,
            njobs=1,
        )[0]
    except (RuntimeError, ValueError, OSError) as exc:
        raise FrontendError(
            f"eSpeak failed to phonemize text using language '{options.language}': {exc}"
        ) from exc
    result = unicodedata.normalize("NFC", result)
    result = re.sub(r"\s+", " ", result).strip()
    if not result:
        raise FrontendError(
            f"eSpeak produced an empty phoneme sequence for language '{options.language}'."
        )
    return result


def process_text(
    text: str,
    *,
    options: FrontendOptions | None = None,
    prephonemized: str | None = None,
) -> FrontendResult:
    """Normalize a transcript and obtain phonemes from eSpeak or supplied input."""
    options = options or FrontendOptions()
    options.validate()
    normalized = normalize_text(text)
    if options.mode == "prephonemized":
        if prephonemized is None:
            raise FrontendError(
                "Prephonemized mode requires a non-empty phoneme string in every manifest row."
            )
        phonemes = unicodedata.normalize("NFC", prephonemized)
        phonemes = re.sub(r"\s+", " ", phonemes).strip()
        if not phonemes:
            raise FrontendError("Prephonemized input is empty after normalization.")
        if "\x00" in phonemes or any(character in "\r\n" for character in phonemes):
            raise FrontendError("Prephonemized input contains an unsupported control character.")
    elif options.mode == "custom":
        custom = _create_custom_frontend(options)
        try:
            first_normalized = custom.implementation.normalize(text)
            second_normalized = custom.implementation.normalize(text)
        except Exception as exc:
            raise FrontendError(
                f"Custom frontend '{custom.identity}' normalize() failed: {exc}"
            ) from exc
        if first_normalized != second_normalized:
            raise FrontendError(
                f"Custom frontend '{custom.identity}' normalize() is not deterministic."
            )
        normalized = _hook_text(
            first_normalized, field="normalize()", identity=custom.identity
        )
        try:
            first_phonemes = custom.implementation.phonemize(normalized)
            second_phonemes = custom.implementation.phonemize(normalized)
        except Exception as exc:
            raise FrontendError(
                f"Custom frontend '{custom.identity}' phonemize() failed: {exc}"
            ) from exc
        if first_phonemes != second_phonemes:
            raise FrontendError(
                f"Custom frontend '{custom.identity}' phonemize() is not deterministic."
            )
        phonemes = _hook_text(
            first_phonemes, field="phonemize()", identity=custom.identity
        )
        undeclared = sorted(set(phonemes).difference(custom.symbols))
        if undeclared:
            raise FrontendError(
                f"Custom frontend '{custom.identity}' emitted undeclared symbols: "
                + ", ".join(repr(symbol) for symbol in undeclared)
            )
    else:
        phonemes = phonemize_text(normalized, options)
    return FrontendResult(raw_text=text, normalized_text=normalized, phonemes=phonemes)


def validate_frontend(options: FrontendOptions) -> None:
    """Fail early when a configured frontend cannot be initialized."""
    options.validate()
    if options.mode == "espeak":
        _backend(options)
    elif options.mode == "custom":
        _create_custom_frontend(options)
