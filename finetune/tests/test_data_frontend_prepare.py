from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from inflect_finetune.audit import AuditOptions, audit_dataset
from inflect_finetune.prepare import PreparationError, PrepareOptions, prepare_dataset


def _write_wav(path: Path, frequency: float) -> None:
    sample_rate = 24_000
    time = np.arange(sample_rate // 10, dtype=np.float32) / sample_rate
    waveform = 0.1 * np.sin(2.0 * np.pi * frequency * time)
    sf.write(path, waveform, sample_rate, subtype="PCM_16")


def _write_manifest(root: Path, rows: list[dict[str, str]]) -> Path:
    path = root / "metadata.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _source_dataset(
    root: Path,
    *,
    count: int,
    texts: list[str] | None = None,
    groups: list[tuple[str, str] | None] | None = None,
    speakers: list[str | None] | None = None,
) -> Path:
    root.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    for index in range(count):
        audio_name = f"audio-{index}.wav"
        _write_wav(root / audio_name, 180.0 + index * 37.0)
        row = {
            "audio": audio_name,
            "text": texts[index] if texts else f"Sentence {index}.",
            "phonemes": f"test {index}",
        }
        if groups and groups[index]:
            field, value = groups[index]
            row[field] = value
        if speakers and speakers[index]:
            row["speaker"] = speakers[index]
        rows.append(row)
    return _write_manifest(root, rows)


def _prepare(manifest: Path, output: Path, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "manifest_path": manifest,
        "output_dir": output,
        "frontend": "prephonemized",
        "validation_fraction": 0.34,
        "split_seed": 99,
    }
    values.update(overrides)
    return prepare_dataset(PrepareOptions(**values))


def _read_split(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_prepare_rejects_dataset_too_small_for_validation(tmp_path: Path) -> None:
    manifest = _source_dataset(tmp_path / "source", count=1)
    with pytest.raises(PreparationError, match="at least two usable rows"):
        _prepare(manifest, tmp_path / "prepared")


def test_group_aware_split_is_deterministic_and_keeps_groups_together(
    tmp_path: Path,
) -> None:
    manifest = _source_dataset(
        tmp_path / "source",
        count=6,
        groups=[
            None,
            None,
            ("session", "session-b"),
            ("session", "session-b"),
            ("group_id", "group-c"),
            ("group_id", "group-d"),
        ],
        speakers=["alice"] * 6,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_metadata = _prepare(manifest, first)
    second_metadata = _prepare(manifest, second)

    assert first_metadata["split"] == second_metadata["split"]
    first_splits = {
        split: _read_split(first / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    second_splits = {
        split: _read_split(second / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    assert first_splits == second_splits

    placements: dict[tuple[str, str], set[str]] = {}
    for split, rows in first_splits.items():
        for row in rows:
            if "group_id" in row:
                key = (str(row["group_field"]), str(row["group_id"]))
                placements.setdefault(key, set()).add(split)
    assert all(len(splits) == 1 for splits in placements.values())
    assert first_metadata["split"]["group_fields"] == [
        "group_id",
        "session",
    ]
    assert first_metadata["speaker"] == "alice"


def test_same_speaker_multiple_sessions_produces_nonempty_splits(
    tmp_path: Path,
) -> None:
    manifest = _source_dataset(
        tmp_path / "source",
        count=6,
        groups=[
            ("session", "session-a"),
            ("session", "session-a"),
            ("session", "session-b"),
            ("session", "session-b"),
            ("session", "session-c"),
            ("session", "session-c"),
        ],
        speakers=["one-speaker"] * 6,
    )
    output = tmp_path / "prepared"
    metadata = _prepare(manifest, output)
    assert metadata["speaker"] == "one-speaker"
    assert metadata["row_counts"]["train"] > 0
    assert metadata["row_counts"]["validation"] > 0
    for split in ("train", "validation"):
        assert {row["speaker"] for row in _read_split(output / f"{split}.jsonl")} == {
            "one-speaker"
        }


def test_prepare_rejects_multiple_speakers(tmp_path: Path) -> None:
    manifest = _source_dataset(
        tmp_path / "source",
        count=3,
        speakers=["alice", "bob", "alice"],
    )
    with pytest.raises(PreparationError, match="one consistent nonempty speaker"):
        _prepare(manifest, tmp_path / "prepared")


def test_normalized_duplicate_text_is_co_located(tmp_path: Path) -> None:
    manifest = _source_dataset(
        tmp_path / "source",
        count=4,
        texts=["Hello   World", "hello world", "Independent A", "Independent B"],
    )
    output = tmp_path / "prepared"
    _prepare(manifest, output, validation_fraction=0.5)
    placements: dict[str, set[str]] = {}
    for split in ("train", "validation"):
        for row in _read_split(output / f"{split}.jsonl"):
            key = " ".join(str(row["normalized_text"]).casefold().split())
            placements.setdefault(key, set()).add(split)
    assert placements["hello world"] in ({"train"}, {"validation"})


def test_prepare_rejects_duplicate_audio_content(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_wav(source / "first.wav", 220.0)
    (source / "second.wav").write_bytes((source / "first.wav").read_bytes())
    _write_wav(source / "third.wav", 330.0)
    manifest = _write_manifest(
        source,
        [
            {"audio": "first.wav", "text": "One", "phonemes": "one"},
            {"audio": "second.wav", "text": "Two", "phonemes": "two"},
            {"audio": "third.wav", "text": "Three", "phonemes": "three"},
        ],
    )
    with pytest.raises(PreparationError, match="Duplicate audio content"):
        _prepare(manifest, tmp_path / "prepared")


def _custom_hook_source() -> str:
    return """
class Frontend:
    def __init__(self, language):
        self.language = language

    def normalize(self, text):
        return " ".join(text.lower().split())

    def phonemize(self, normalized_text):
        return "ab"

    def symbols(self):
        return ["a", "b"]

    def metadata(self):
        return {
            "name": "test-frontend",
            "version": "1",
            "language": self.language,
            "configuration": {"case": "lower"},
        }

def create_frontend(*, language):
    return Frontend(language)
""".lstrip()


def test_custom_file_hook_is_hashed_and_recorded(tmp_path: Path) -> None:
    manifest = _source_dataset(tmp_path / "source", count=3)
    hook_path = tmp_path / "custom_frontend.py"
    hook_path.write_text(_custom_hook_source(), encoding="utf-8")
    output = tmp_path / "prepared"
    metadata = _prepare(
        manifest,
        output,
        frontend="custom",
        frontend_hook=f"{hook_path}:create_frontend",
    )

    hook = metadata["frontend"]["hook"]
    assert hook["identity"] == "file:custom_frontend.py:create_frontend"
    assert hook["source_kind"] == "file"
    assert hook["source_sha256"] == hashlib.sha256(hook_path.read_bytes()).hexdigest()
    assert len(hook["metadata_sha256"]) == 64
    assert hook["declared_metadata"]["language"] == "en-us"
    assert {row["phonemes"] for row in _read_split(output / "train.jsonl")} == {"ab"}


def test_custom_module_hook_is_supported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _source_dataset(tmp_path / "source", count=3)
    module_name = "temporary_inflect_frontend"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(_custom_hook_source(), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop(module_name, None)

    metadata = _prepare(
        manifest,
        tmp_path / "prepared",
        frontend="custom",
        frontend_hook=f"{module_name}:create_frontend",
    )
    hook = metadata["frontend"]["hook"]
    assert hook["identity"] == f"{module_name}:create_frontend"
    assert hook["source_kind"] == "module"
    assert hook["source_sha256"] == hashlib.sha256(module_path.read_bytes()).hexdigest()


def test_audit_detects_group_and_normalized_text_crossing_splits(
    tmp_path: Path,
) -> None:
    manifest = _source_dataset(tmp_path / "source", count=4)
    output = tmp_path / "prepared"
    _prepare(manifest, output, validation_fraction=0.5)
    train = _read_split(output / "train.jsonl")
    validation = _read_split(output / "validation.jsonl")
    train[0]["group_id"] = "leaked"
    train[0]["group_field"] = "session"
    validation[0]["group_id"] = "leaked"
    validation[0]["group_field"] = "session"
    validation[0]["normalized_text"] = train[0]["normalized_text"]
    (output / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in train), encoding="utf-8"
    )
    (output / "validation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in validation), encoding="utf-8"
    )

    report = audit_dataset(AuditOptions(prepared_dir=output, strict=False))
    assert not report["valid"]
    assert any("normalized transcript crosses" in error for error in report["errors"])
    assert any("session='leaked' crosses" in error for error in report["errors"])


def test_audit_treats_empty_validation_as_error(tmp_path: Path) -> None:
    manifest = _source_dataset(tmp_path / "source", count=3)
    output = tmp_path / "prepared"
    _prepare(manifest, output)
    validation = output / "validation.jsonl"
    validation.write_text("", encoding="utf-8")
    dataset_path = output / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["row_counts"]["validation"] = 0
    dataset["row_counts"]["total"] = dataset["row_counts"]["train"]
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    report = audit_dataset(AuditOptions(prepared_dir=output, strict=False))
    assert not report["valid"]
    assert any("no validation rows" in error for error in report["errors"])


def _extending_hook_source() -> str:
    """A frontend that emits a symbol outside the released inventory."""
    return """
class Frontend:
    def __init__(self, language):
        self.language = language

    def normalize(self, text):
        return " ".join(text.split())

    def phonemize(self, normalized_text):
        return "a\\uA71Cb"

    def symbols(self):
        return ["a", "b", "\\uA71C"]

    def metadata(self):
        return {
            "name": "extending-frontend",
            "version": "1",
            "language": self.language,
            "configuration": {},
        }

def create_frontend(*, language):
    return Frontend(language)
""".lstrip()


def _japanese_source(root: Path, texts: list[str]) -> Path:
    root.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    for index, text in enumerate(texts):
        audio_name = f"ja-{index}.wav"
        _write_wav(root / audio_name, 210.0 + index * 41.0)
        rows.append({"audio": audio_name, "text": text, "speaker": "voice-ja"})
    return _write_manifest(root, rows)


def test_bundled_japanese_frontend_prepares_without_new_symbols(tmp_path: Path) -> None:
    pytest.importorskip(
        "pyopenjtalk",
        reason="The Japanese frontend needs the 'ja' extra (pyopenjtalk-plus).",
    )
    manifest = _japanese_source(
        tmp_path / "source",
        [
            "こんにちは、今日はいい天気ですね。",
            "彼女は2026年8月30日に来ます。",
            "よろしくお願いします。",
            "本当に美味しいお茶でした。",
        ],
    )
    output = tmp_path / "prepared"
    metadata = _prepare(
        manifest,
        output,
        language="ja",
        frontend="ja-openjtalk",
    )

    # The bundled frontend resolves to the custom contract export understands,
    # while the registry name stays recorded for reproducibility.
    assert metadata["frontend"]["type"] == "custom"
    assert metadata["frontend"]["registry"]["name"] == "ja-openjtalk"
    assert metadata["frontend"]["registry"]["language"] == "ja"
    assert metadata["frontend"]["hook"]["declared_metadata"]["name"] == "ja-openjtalk"
    assert metadata["diagnostics"]["added_symbol_count"] == 0
    assert metadata["diagnostics"]["base_symbol_coverage_fraction"] == 1.0

    inventory = json.loads((output / "symbols.json").read_text(encoding="utf-8"))
    assert inventory["added_symbols"] == []
    assert inventory["total_size"] == inventory["base_size"]

    report = audit_dataset(
        AuditOptions(prepared_dir=output, require_no_new_symbols=True)
    )
    assert report["valid"]
    assert report["added_symbols"] == []
    assert report["required_no_new_symbols"] is True


def test_audit_can_require_that_the_released_inventory_is_not_extended(
    tmp_path: Path,
) -> None:
    manifest = _source_dataset(tmp_path / "source", count=3)
    hook_path = tmp_path / "extending_frontend.py"
    hook_path.write_text(_extending_hook_source(), encoding="utf-8")
    output = tmp_path / "prepared"
    metadata = _prepare(
        manifest,
        output,
        frontend="custom",
        frontend_hook=f"{hook_path}:create_frontend",
    )
    assert metadata["diagnostics"]["added_symbol_count"] == 1

    permissive = audit_dataset(AuditOptions(prepared_dir=output))
    assert permissive["valid"]
    assert permissive["added_symbols"] == ["ꜜ"]

    strict = audit_dataset(
        AuditOptions(prepared_dir=output, strict=False, require_no_new_symbols=True)
    )
    assert not strict["valid"]
    assert any("adds symbols" in error for error in strict["errors"])
