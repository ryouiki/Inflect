"""Small, dependency-free helpers for reproducible public reports."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPORT_SCHEMA_VERSION = 1


def utc_now() -> str:
    """Return an RFC 3339 timestamp without host-specific locale data."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Hash a file without reading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: str | Path, *, relative_to: str | Path | None = None) -> dict[str, Any]:
    """Describe a file using a stable relative path, size, and SHA-256."""

    source = Path(path).resolve()
    if relative_to is None:
        display_path = source.name
    else:
        display_path = source.relative_to(Path(relative_to).resolve()).as_posix()
    return {
        "path": display_path,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: str | Path, payload: Any) -> Path:
    """Atomically write deterministic, UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def write_text(path: str | Path, text: str) -> Path:
    """Atomically write a human-readable UTF-8 report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text.rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def write_checksums(
    path: str | Path,
    files: Iterable[str | Path],
    *,
    relative_to: str | Path,
) -> Path:
    """Write a standard SHA-256 checksum list."""

    root = Path(relative_to).resolve()
    rows = []
    resolved = (Path(file).resolve() for file in files)
    for item in sorted(resolved, key=lambda value: value.as_posix()):
        rows.append(f"{sha256_file(item)}  {item.relative_to(root).as_posix()}")
    return write_text(path, "\n".join(rows))


def make_report(kind: str, **payload: Any) -> dict[str, Any]:
    """Create the shared envelope used by export and evaluation reports."""

    return {
        "format": f"inflect_adaptation_{kind}_v{REPORT_SCHEMA_VERSION}",
        "created_at": utc_now(),
        **payload,
    }


def status(ok: bool, message: str, **details: Any) -> dict[str, Any]:
    """Create a consistently shaped validation result."""

    return {
        "ok": bool(ok),
        "message": message,
        **details,
    }
