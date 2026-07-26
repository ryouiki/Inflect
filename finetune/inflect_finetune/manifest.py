"""Strict CSV and JSONL manifest parsing with confined audio paths."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


class ManifestError(ValueError):
    """Raised when a source manifest is malformed or unsafe."""


@dataclass(frozen=True)
class ManifestRow:
    """Canonical source row consumed by dataset preparation."""

    index: int
    source_location: str
    audio_relative: str
    audio_path: Path
    text: str
    phonemes: str | None = None
    row_id: str | None = None
    speaker: str | None = None
    group_id: str | None = None
    group_field: str | None = None


_AUDIO_FIELDS = ("audio", "audio_path", "wav", "wav_path", "path")
_TEXT_FIELDS = ("text", "transcript", "sentence")
_PHONEME_FIELDS = ("phonemes", "phoneme_text", "phones")
_ID_FIELDS = ("id", "utt_id", "utterance_id", "name")
_GROUP_FIELDS = ("group_id", "session")
_SPEAKER_FIELDS = ("speaker",)
_MAX_MANIFEST_BYTES = 128 * 1024 * 1024
_MAX_LINE_BYTES = 4 * 1024 * 1024
_MAX_FIELD_CHARS = 1_000_000


def _first(row: Mapping[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return value
    return None


def _first_named(
    row: Mapping[str, Any], fields: tuple[str, ...]
) -> tuple[str | None, Any]:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return field, value
    return None, None


def resolve_audio_path(audio_root: Path, value: str, *, location: str) -> tuple[str, Path]:
    """Resolve a relative manifest path while rejecting traversal and symlink escapes."""
    if "\x00" in value:
        raise ManifestError(f"Audio path contains a null byte at {location}.")
    raw = Path(value.strip())
    if not value.strip():
        raise ManifestError(f"Audio path is empty at {location}.")
    if raw.is_absolute() or raw.drive or raw.root:
        raise ManifestError(
            f"Audio path must be relative to the configured audio root at {location}: {value!r}"
        )
    if any(part == ".." for part in raw.parts):
        raise ManifestError(f"Audio path traversal is not allowed at {location}: {value!r}")

    root = Path(audio_root).expanduser().resolve()
    candidate = (root / raw).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError(
            f"Audio path escapes the configured audio root at {location}: {value!r}"
        ) from exc
    if not candidate.is_file():
        raise ManifestError(f"Audio file referenced at {location} does not exist: {candidate}")
    return relative.as_posix(), candidate


def _validate_scalar(value: Any, field: str, location: str) -> str:
    if isinstance(value, (dict, list, tuple)):
        raise ManifestError(f"Field '{field}' must be a string at {location}.")
    text = str(value)
    if "\x00" in text:
        raise ManifestError(f"Field '{field}' contains a null byte at {location}.")
    if len(text) > _MAX_FIELD_CHARS:
        raise ManifestError(
            f"Field '{field}' exceeds {_MAX_FIELD_CHARS:,} characters at {location}."
        )
    return text.strip()


def _canonicalize(
    raw: Mapping[str, Any],
    *,
    index: int,
    location: str,
    audio_root: Path,
    require_phonemes: bool,
) -> ManifestRow:
    audio_value = _first(raw, _AUDIO_FIELDS)
    text_value = _first(raw, _TEXT_FIELDS)
    phone_value = _first(raw, _PHONEME_FIELDS)
    row_id_value = _first(raw, _ID_FIELDS)
    speaker_value = _first(raw, _SPEAKER_FIELDS)
    group_field, group_value = _first_named(raw, _GROUP_FIELDS)

    if audio_value is None:
        raise ManifestError(
            f"No audio field found at {location}; expected one of {', '.join(_AUDIO_FIELDS)}."
        )
    if text_value is None:
        raise ManifestError(
            f"No transcript field found at {location}; expected one of {', '.join(_TEXT_FIELDS)}."
        )
    if require_phonemes and phone_value is None:
        raise ManifestError(
            f"No phoneme field found at {location}; prephonemized mode expects one of "
            f"{', '.join(_PHONEME_FIELDS)}."
        )

    audio_text = _validate_scalar(audio_value, "audio", location)
    text = _validate_scalar(text_value, "text", location)
    phonemes = (
        _validate_scalar(phone_value, "phonemes", location)
        if phone_value is not None
        else None
    )
    row_id = (
        _validate_scalar(row_id_value, "id", location) if row_id_value is not None else None
    )
    speaker = (
        _validate_scalar(speaker_value, "speaker", location)
        if speaker_value is not None
        else None
    )
    group_id = (
        _validate_scalar(group_value, group_field or "group_id", location)
        if group_value is not None
        else None
    )
    if not text:
        raise ManifestError(f"Transcript is empty at {location}.")
    if require_phonemes and not phonemes:
        raise ManifestError(f"Phoneme text is empty at {location}.")
    relative, resolved = resolve_audio_path(audio_root, audio_text, location=location)
    return ManifestRow(
        index=index,
        source_location=location,
        audio_relative=relative,
        audio_path=resolved,
        text=text,
        phonemes=phonemes,
        row_id=row_id,
        speaker=speaker,
        group_id=group_id,
        group_field=group_field,
    )


def _jsonl_rows(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if len(raw_line) > _MAX_LINE_BYTES:
                raise ManifestError(
                    f"JSONL line {line_number} exceeds {_MAX_LINE_BYTES:,} bytes in {path}."
                )
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8")
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ManifestError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ManifestError(f"Expected a JSON object at {path}:{line_number}.")
            yield line_number, value


def _csv_rows(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ManifestError(f"CSV manifest has no header: {path}")
            normalized = [str(field).strip() for field in reader.fieldnames]
            if not all(normalized) or len(set(normalized)) != len(normalized):
                raise ManifestError(f"CSV header contains empty or duplicate fields: {path}")
            reader.fieldnames = normalized
            for line_number, row in enumerate(reader, 2):
                if None in row:
                    raise ManifestError(
                        f"CSV row has more values than header columns at {path}:{line_number}."
                    )
                yield line_number, row
    except UnicodeDecodeError as exc:
        raise ManifestError(f"CSV manifest must be UTF-8 encoded: {path}") from exc
    except csv.Error as exc:
        raise ManifestError(f"Invalid CSV manifest {path}: {exc}") from exc


def parse_manifest(
    path: Path,
    *,
    audio_root: Path | None = None,
    require_phonemes: bool = False,
) -> list[ManifestRow]:
    """Parse a UTF-8 CSV or JSONL manifest into validated canonical rows."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ManifestError(f"Manifest does not exist: {path}")
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ManifestError(
            f"Manifest exceeds the {_MAX_MANIFEST_BYTES // (1024 * 1024)} MiB safety limit: {path}"
        )
    root = Path(audio_root).expanduser().resolve() if audio_root else path.parent
    if not root.is_dir():
        raise ManifestError(f"Configured audio root is not a directory: {root}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        source_rows = _jsonl_rows(path)
    elif suffix == ".csv":
        source_rows = _csv_rows(path)
    else:
        raise ManifestError(
            f"Unsupported manifest extension '{path.suffix}'. Use .csv or .jsonl."
        )

    rows = [
        _canonicalize(
            raw,
            index=index,
            location=f"{path}:{line_number}",
            audio_root=root,
            require_phonemes=require_phonemes,
        )
        for index, (line_number, raw) in enumerate(source_rows)
    ]
    if not rows:
        raise ManifestError(f"Manifest contains no data rows: {path}")
    return rows
