from __future__ import annotations

import importlib
import importlib.util
import sys
import unicodedata
from pathlib import Path

import pytest

from inflect_finetune.cli import build_parser
from inflect_finetune.frontends import (
    REGISTRY,
    FrontendRegistryError,
    get,
    hook_path_for_record,
    is_registry_frontend,
    registry_names,
    registry_record,
    resolve,
)
from inflect_finetune.prepare import PrepareOptions
from inflect_finetune.symbols import BASE_SYMBOLS


def test_bundled_frontends_are_registered_and_exposed_by_the_cli() -> None:
    assert "ja-openjtalk" in registry_names()
    assert is_registry_frontend("ja-openjtalk")
    assert not is_registry_frontend("espeak")

    parser = build_parser()
    for name in registry_names():
        args = parser.parse_args(
            [
                "prepare",
                "--manifest",
                "metadata.jsonl",
                "--language",
                REGISTRY[name].language,
                "--frontend",
                name,
                "--output",
                "prepared",
            ]
        )
        assert args.frontend == name


def test_non_registry_frontends_pass_through_unchanged() -> None:
    espeak = resolve("espeak", "en-us")
    assert espeak.mode == "espeak"
    assert espeak.language == "en-us"
    assert espeak.hook is None

    custom = resolve("custom", "xx", hook="module:create_frontend")
    assert custom.mode == "custom"
    assert custom.hook == "module:create_frontend"


def test_registry_resolves_to_a_bundled_custom_frontend() -> None:
    options = resolve("ja-openjtalk", "ja")
    assert options.mode == "custom"
    assert options.language == "ja"
    assert options.hook is not None
    source, _, factory = options.hook.rpartition(":")
    assert factory == "create_frontend"
    assert Path(source).is_file()
    options.validate()


def test_registry_accepts_regional_variants_of_its_language() -> None:
    assert resolve("ja-openjtalk", "ja-JP").mode == "custom"


def test_registry_rejects_a_conflicting_or_wrong_configuration() -> None:
    with pytest.raises(FrontendRegistryError, match="supplies its own hook"):
        resolve("ja-openjtalk", "ja", hook="other.py:create_frontend")
    with pytest.raises(FrontendRegistryError, match="configured for language"):
        resolve("ja-openjtalk", "en-us")
    with pytest.raises(FrontendRegistryError, match="Unknown bundled frontend"):
        get("ja-nonexistent")


def test_registry_record_and_hook_recovery_round_trip() -> None:
    record = registry_record("ja-openjtalk")
    assert record["name"] == "ja-openjtalk"
    assert record["language"] == "ja"
    assert record["toolkit_version"]

    recovered = hook_path_for_record(record)
    assert recovered is not None
    assert recovered == REGISTRY["ja-openjtalk"].module_file
    assert hook_path_for_record(None) is None
    assert hook_path_for_record({"name": "not-registered"}) is None
    assert hook_path_for_record({}) is None


def test_prepare_options_accept_a_bundled_frontend_without_a_hook() -> None:
    PrepareOptions(
        manifest_path=Path("metadata.jsonl"),
        output_dir=Path("prepared"),
        language="ja",
        frontend="ja-openjtalk",
    ).validate()

    with pytest.raises(ValueError, match="frontend_hook may only be used"):
        PrepareOptions(
            manifest_path=Path("metadata.jsonl"),
            output_dir=Path("prepared"),
            language="ja",
            frontend="ja-openjtalk",
            frontend_hook="other.py:create_frontend",
        ).validate()

    with pytest.raises(ValueError, match="bundled language frontend"):
        PrepareOptions(
            manifest_path=Path("metadata.jsonl"),
            output_dir=Path("prepared"),
            frontend="ja-nonexistent",
        ).validate()


def _language_module(name: str):
    """Import one bundled frontend's module by its registry entry."""
    return importlib.import_module(
        f"inflect_finetune.frontends.{REGISTRY[name].module_file.stem}"
    )


@pytest.mark.parametrize("name", registry_names())
def test_bundled_frontend_adds_no_symbols_to_the_released_inventory(name: str) -> None:
    """The whole point of a bundled frontend is that it costs no embedding rows.

    An extended inventory changes the symbol count, and a checkpoint whose
    inventory is no longer the release inventory cannot warm-start a later run.
    """
    declared = _language_module(name).DECLARED_SYMBOLS
    assert sorted(set(declared) - set(BASE_SYMBOLS)) == []


@pytest.mark.parametrize("name", registry_names())
def test_bundled_frontend_declares_a_valid_symbol_inventory(name: str) -> None:
    declared = _language_module(name).DECLARED_SYMBOLS
    assert declared
    assert len(declared) == len(set(declared))
    assert all(isinstance(symbol, str) and len(symbol) == 1 for symbol in declared)
    assert declared == tuple(sorted(declared))
    assert all(
        unicodedata.normalize("NFC", symbol) == symbol for symbol in declared
    )


@pytest.mark.parametrize(
    ("name", "dependency"),
    [("ja-openjtalk", "pyopenjtalk"), ("ko-g2pkk", "g2pkk")],
)
def test_a_missing_language_dependency_fails_before_any_data_is_touched(
    name: str,
    dependency: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation validates the frontend first; a missing extra must stop it there.

    Loading the engine lazily would let preparation start and fail part-way
    through a corpus instead.
    """
    from inflect_finetune import frontend as frontend_module

    monkeypatch.setitem(sys.modules, dependency, None)
    # validate_frontend reuses an already-constructed frontend, which would hide
    # the failure in a session where another test built it successfully.
    monkeypatch.setattr(frontend_module, "_CUSTOM_FRONTENDS", {})

    entry = REGISTRY[name]
    with pytest.raises(RuntimeError, match="requires"):
        _language_module(name).create_frontend(language=entry.language)
    with pytest.raises(frontend_module.FrontendError, match="requires"):
        frontend_module.validate_frontend(resolve(name, entry.language))


@pytest.mark.parametrize(
    ("name", "dependency"),
    [("ja-openjtalk", "pyopenjtalk"), ("ko-g2pkk", "g2pkk")],
)
def test_symbol_tables_are_readable_without_the_language_dependency(
    name: str,
    dependency: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mapping must be inspectable so its coverage stays checked in CI."""
    monkeypatch.setitem(sys.modules, dependency, None)
    # Load a private copy from the file. Reloading the shared module would
    # rebind its classes, and every later test comparing against the originals
    # would then see a different exception type.
    source = REGISTRY[name].module_file
    spec = importlib.util.spec_from_file_location(f"_probe_{source.stem}", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.DECLARED_SYMBOLS
    assert module.PUNCTUATION_MAP
