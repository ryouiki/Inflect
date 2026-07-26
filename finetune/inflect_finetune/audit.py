"""Integrity and coverage audits for prepared Inflect adaptation datasets."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import soundfile as sf

from .audio import AudioOptions, AudioValidationError, inspect_wav
from .manifest import ManifestError, resolve_audio_path
from .symbols import SymbolInventoryError, audit_symbol_coverage


class DatasetAuditError(RuntimeError):
    """Raised when strict auditing finds invalid prepared data."""


@dataclass(frozen=True)
class AuditOptions:
    """Options suitable for programmatic use and a future audit CLI."""

    prepared_dir: Path
    strict: bool = True
    duration_tolerance_seconds: float = 0.02

    def validate(self) -> None:
        """Validate audit settings."""
        if self.duration_tolerance_seconds < 0:
            raise ValueError("duration_tolerance_seconds cannot be negative.")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetAuditError(f"Required prepared dataset file is missing: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetAuditError(f"Invalid JSON in {path}: {exc}") from exc


def _jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetAuditError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise DatasetAuditError(f"Expected a JSON object at {path}:{line_number}.")
                yield line_number, value
    except FileNotFoundError as exc:
        raise DatasetAuditError(f"Required prepared split is missing: {path}") from exc


def _summary(report: dict[str, Any]) -> str:
    lines = [
        "Inflect dataset audit",
        "=====================",
        f"Status: {'PASS' if report['valid'] else 'FAIL'}",
        f"Rows checked: {report['row_counts']['total']}",
        f"Audio duration: {report['total_duration_seconds']:.2f} seconds",
        f"Errors: {len(report['errors'])}",
        f"Warnings: {len(report['warnings'])}",
    ]
    if report["errors"]:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {error}" for error in report["errors"])
    if report["warnings"]:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_split_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def audit_dataset(options: AuditOptions) -> dict[str, Any]:
    """Audit layout, WAV invariants, duration metadata, and symbol coverage."""
    options.validate()
    root = Path(options.prepared_dir).expanduser().resolve()
    if not root.is_dir():
        raise DatasetAuditError(f"Prepared dataset directory does not exist: {root}")

    dataset = _load_json(root / "dataset.json")
    symbols_payload = _load_json(root / "symbols.json")
    if not isinstance(dataset, dict):
        raise DatasetAuditError("dataset.json must contain a JSON object.")
    symbols = symbols_payload.get("symbols") if isinstance(symbols_payload, dict) else None
    if not isinstance(symbols, list):
        raise DatasetAuditError("symbols.json must contain a 'symbols' list.")

    sample_rate = dataset.get("sample_rate")
    if sample_rate != 24_000:
        raise DatasetAuditError(
            f"dataset.json sample_rate must be 24000, found {sample_rate!r}."
        )
    errors: list[str] = []
    warnings: list[str] = []
    seen_audio: set[str] = set()
    audio_hash_locations: dict[str, tuple[str, str]] = {}
    normalized_text_splits: dict[str, set[str]] = {}
    normalized_text_locations: dict[str, list[str]] = {}
    group_splits: dict[tuple[str, str], set[str]] = {}
    group_locations: dict[tuple[str, str], list[str]] = {}
    speakers: set[str] = set()
    phonemes: list[str] = []
    split_counts: dict[str, int] = {}
    total_duration = 0.0
    audio_options = AudioOptions(sample_rate=24_000)

    for split in ("train", "validation"):
        split_count = 0
        for line_number, row in _jsonl(root / f"{split}.jsonl"):
            split_count += 1
            location = f"{split}.jsonl:{line_number}"
            required = {
                "audio",
                "text",
                "normalized_text",
                "phonemes",
                "duration_seconds",
            }
            missing = sorted(required.difference(row))
            if missing:
                errors.append(f"{location} is missing fields: {', '.join(missing)}")
                continue
            text_fields = required - {"duration_seconds"}
            if not all(
                isinstance(row[field], str) and row[field] for field in text_fields
            ):
                errors.append(f"{location} contains an empty or non-string text field.")
                continue
            try:
                relative, audio_path = resolve_audio_path(
                    root, str(row["audio"]), location=location
                )
            except ManifestError as exc:
                errors.append(str(exc))
                continue
            if relative in seen_audio:
                errors.append(f"Prepared audio path appears more than once: {relative}")
                continue
            seen_audio.add(relative)
            try:
                info = inspect_wav(audio_path, audio_options)
            except AudioValidationError as exc:
                errors.append(str(exc))
                continue
            if info.samplerate != 24_000 or info.channels != 1:
                errors.append(
                    f"{relative} must be 24 kHz mono; found "
                    f"{info.samplerate} Hz and {info.channels} channel(s)."
                )
            audio_sha256 = _sha256(audio_path)
            recorded_sha256 = row.get("audio_sha256")
            if recorded_sha256 is not None and recorded_sha256 != audio_sha256:
                errors.append(
                    f"{location} audio_sha256 does not match the prepared WAV content."
                )
            prior_audio = audio_hash_locations.get(audio_sha256)
            if prior_audio is not None:
                prior_split, prior_location = prior_audio
                boundary = (
                    " across train and validation"
                    if prior_split != split
                    else " within the prepared dataset"
                )
                errors.append(
                    f"Duplicate audio content detected{boundary}: "
                    f"{prior_location} and {location}."
                )
            else:
                audio_hash_locations[audio_sha256] = (split, location)

            normalized_key = _normalized_split_key(row["normalized_text"])
            normalized_text_splits.setdefault(normalized_key, set()).add(split)
            normalized_text_locations.setdefault(normalized_key, []).append(location)
            group_id = row.get("group_id")
            group_field = row.get("group_field", "group_id")
            speaker = row.get("speaker")
            if speaker is not None:
                if not isinstance(speaker, str) or not speaker.strip():
                    errors.append(f"{location} has an empty or non-string speaker.")
                else:
                    speakers.add(speaker)
            if group_id is not None:
                if not isinstance(group_id, str) or not group_id.strip():
                    errors.append(f"{location} has an empty or non-string group_id.")
                elif group_field not in {"group_id", "session"}:
                    errors.append(
                        f"{location} has unsupported group_field {group_field!r}."
                    )
                else:
                    group_key = (str(group_field), group_id)
                    group_splits.setdefault(group_key, set()).add(split)
                    group_locations.setdefault(group_key, []).append(location)
            actual_duration = info.frames / info.samplerate
            try:
                recorded_duration = float(row["duration_seconds"])
            except (TypeError, ValueError):
                errors.append(f"{location} has a non-numeric duration_seconds value.")
                continue
            if abs(actual_duration - recorded_duration) > options.duration_tolerance_seconds:
                errors.append(
                    f"{location} duration differs from WAV metadata by "
                    f"{abs(actual_duration - recorded_duration):.4f}s."
                )
            total_duration += actual_duration
            phonemes.append(row["phonemes"])
        split_counts[split] = split_count

    try:
        coverage = audit_symbol_coverage(phonemes, symbols)
    except SymbolInventoryError as exc:
        errors.append(str(exc))
        coverage = None
    if coverage and coverage.unknown_counts:
        errors.append(
            "Phoneme text contains symbols absent from symbols.json: "
            + ", ".join(repr(symbol) for symbol in coverage.unknown_counts)
        )

    for normalized_key, splits in normalized_text_splits.items():
        if len(splits) > 1:
            locations = ", ".join(normalized_text_locations[normalized_key])
            errors.append(
                "The same normalized transcript crosses train and validation "
                f"({normalized_key!r}): {locations}."
            )
    for (group_field, group_id), splits in group_splits.items():
        if len(splits) > 1:
            locations = ", ".join(group_locations[(group_field, group_id)])
            errors.append(
                f"{group_field}={group_id!r} crosses train and validation: {locations}."
            )
    if len(speakers) > 1:
        errors.append(
            "Single-speaker adaptation requires one speaker value, but prepared rows "
            f"contain {len(speakers)} values: "
            + ", ".join(repr(speaker) for speaker in sorted(speakers))
        )
    dataset_speaker = dataset.get("speaker")
    if dataset_speaker is not None and dataset_speaker not in speakers:
        errors.append(
            "dataset.json speaker does not match the speaker value in prepared rows."
        )

    expected_counts = dataset.get("row_counts", {})
    if expected_counts.get("train") != split_counts["train"]:
        errors.append("dataset.json train row count does not match train.jsonl.")
    if expected_counts.get("validation") != split_counts["validation"]:
        errors.append("dataset.json validation row count does not match validation.jsonl.")
    if expected_counts.get("total") != sum(split_counts.values()):
        errors.append("dataset.json total row count does not match the prepared splits.")
    if not split_counts["train"]:
        errors.append(
            "The prepared dataset has no training rows; training-ready data requires "
            "nonempty train and validation splits."
        )
    if not split_counts["validation"]:
        errors.append(
            "The prepared dataset has no validation rows; training-ready data requires "
            "at least two independent rows or groups and a nonzero validation fraction."
        )
    if sum(split_counts.values()) < 2:
        errors.append(
            "The prepared dataset is too small for training: at least two usable rows "
            "that can occupy independent train and validation splits are required."
        )

    frontend = dataset.get("frontend")
    if isinstance(frontend, dict) and frontend.get("type") == "custom":
        hook = frontend.get("hook")
        if not isinstance(hook, dict):
            errors.append("Custom frontend metadata is missing its hook identity and hashes.")
        else:
            for field in ("identity", "source_sha256", "metadata_sha256"):
                value = hook.get(field)
                if not isinstance(value, str) or not value:
                    errors.append(f"Custom frontend hook metadata is missing '{field}'.")
            for field in ("source_sha256", "metadata_sha256"):
                value = hook.get(field)
                if isinstance(value, str) and not re.fullmatch(r"[0-9a-f]{64}", value):
                    errors.append(
                        f"Custom frontend hook metadata field '{field}' is not a SHA-256 hash."
                    )

    report = {
        "format": "inflect_dataset_audit_v1",
        "valid": not errors,
        "prepared_dir": str(root),
        "row_counts": {
            "train": split_counts["train"],
            "validation": split_counts["validation"],
            "total": sum(split_counts.values()),
        },
        "audio_files": len(seen_audio),
        "unique_audio_hashes": len(audio_hash_locations),
        "explicit_groups": len(group_splits),
        "normalized_text_keys": len(normalized_text_splits),
        "total_duration_seconds": round(total_duration, 6),
        "symbol_coverage": coverage.to_dict() if coverage else None,
        "errors": errors,
        "warnings": warnings,
    }
    (root / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "AUDIT_REPORT.txt").write_text(_summary(report), encoding="utf-8")
    if errors and options.strict:
        raise DatasetAuditError(
            f"Prepared dataset audit failed with {len(errors)} error(s). "
            f"See {root / 'audit.json'}."
        )
    return report
