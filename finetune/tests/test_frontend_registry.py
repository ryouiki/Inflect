from __future__ import annotations

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
from inflect_finetune.frontends.ja_openjtalk import (
    ACCENT_FALL,
    ACCENT_PHRASE_BOUNDARY,
    ACCENT_RISE,
    DECLARED_SYMBOLS,
    PAUSE_SYMBOL,
    PHONE_TO_IPA,
    PUNCTUATION_MAP,
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


def test_japanese_mapping_stays_inside_the_released_symbol_inventory() -> None:
    """Japanese must add no embedding rows.

    A prepared inventory that extends the release inventory changes the symbol
    count, and a checkpoint whose inventory is no longer the release inventory
    cannot warm-start a later adaptation run.
    """
    base = set(BASE_SYMBOLS)
    emitted: set[str] = set()
    for phonemes in PHONE_TO_IPA.values():
        emitted.update(unicodedata.normalize("NFC", phonemes))
    emitted.update(PUNCTUATION_MAP.values())
    emitted.update({ACCENT_RISE, ACCENT_FALL, ACCENT_PHRASE_BOUNDARY, PAUSE_SYMBOL})

    assert sorted(emitted - base) == []


def test_japanese_declared_symbols_are_valid_and_complete() -> None:
    declared = set(DECLARED_SYMBOLS)
    assert len(DECLARED_SYMBOLS) == len(declared)
    assert all(isinstance(symbol, str) and len(symbol) == 1 for symbol in DECLARED_SYMBOLS)
    assert DECLARED_SYMBOLS == tuple(sorted(DECLARED_SYMBOLS))

    for phonemes in PHONE_TO_IPA.values():
        assert set(phonemes) <= declared
    assert set(PUNCTUATION_MAP.values()) <= declared
    assert {ACCENT_RISE, ACCENT_FALL, ACCENT_PHRASE_BOUNDARY, PAUSE_SYMBOL} <= declared
