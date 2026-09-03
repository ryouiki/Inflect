"""Atomic preparation of public Inflect adaptation datasets."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .audio import AudioOptions, convert_wav
from .frontend import (
    custom_frontend_metadata,
    custom_frontend_symbols,
    process_text,
    validate_frontend,
)
from .frontends import is_registry_frontend, registry_names, registry_record, resolve
from .manifest import ManifestRow, parse_manifest
from .symbols import (
    BASE_SYMBOLS,
    audit_symbol_coverage,
    build_symbol_inventory,
    load_base_symbols,
    write_symbol_inventory,
)


class PreparationError(RuntimeError):
    """Raised when a source dataset cannot be prepared completely."""


@dataclass(frozen=True)
class PrepareOptions:
    """Options suitable for programmatic use and a future CLI."""

    manifest_path: Path
    output_dir: Path
    audio_root: Path | None = None
    language: str = "en-us"
    frontend: str = "espeak"
    frontend_hook: str | None = None
    validation_fraction: float = 0.05
    split_seed: int = 1337
    sample_rate: int = 24_000
    min_duration_seconds: float = 0.05
    max_duration_seconds: float | None = None
    base_symbols_path: Path | None = None

    def validate(self) -> None:
        """Validate preparation settings before any output is written."""
        bundled = is_registry_frontend(self.frontend)
        if not bundled and self.frontend not in {"espeak", "prephonemized", "custom"}:
            raise ValueError(
                "frontend must be 'espeak', 'prephonemized', 'custom', or a bundled "
                "language frontend: " + ", ".join(registry_names()) + "."
            )
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in the interval (0, 1).")
        if self.frontend == "custom" and not self.frontend_hook:
            raise ValueError("frontend_hook is required when frontend='custom'.")
        if self.frontend != "custom" and self.frontend_hook:
            raise ValueError("frontend_hook may only be used when frontend='custom'.")
        if self.sample_rate != 24_000:
            raise ValueError("Inflect prepared datasets must use a 24,000 Hz sample rate.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(row: ManifestRow) -> str:
    source = row.row_id or Path(row.audio_relative).stem or f"row-{row.index}"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", source).strip(".-_")[:48] or "utterance"
    suffix = hashlib.sha256(
        f"{row.index}\0{row.audio_relative}\0{row.text}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{row.index:07d}-{slug}-{suffix}.wav"


class _UnionFind:
    """Small deterministic union-find used to construct leakage-safe split units."""

    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, index: int) -> int:
        """Return a component root with path compression."""
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        """Join two components using the smaller root as a stable representative."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def _normalized_split_key(text: str) -> str:
    """Return the canonical key used to prevent transcript leakage."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def _split_components(
    prepared_rows: Sequence[dict[str, Any]],
    source_rows: Sequence[ManifestRow],
    *,
    fraction: float,
    seed: int,
) -> tuple[set[int], dict[str, Any]]:
    """Split deterministic atomic components linked by group or normalized text."""
    count = len(prepared_rows)
    if count < 2:
        raise PreparationError(
            "Training-ready preparation requires at least two usable rows so both "
            "train and validation are nonempty."
        )

    union_find = _UnionFind(count)
    first_group: dict[tuple[str, str], int] = {}
    first_text: dict[str, int] = {}
    for index, (prepared, source) in enumerate(zip(prepared_rows, source_rows)):
        if source.group_id:
            group_key = (source.group_field or "group_id", source.group_id)
            if group_key in first_group:
                union_find.union(index, first_group[group_key])
            else:
                first_group[group_key] = index
        text_key = _normalized_split_key(prepared["normalized_text"])
        if text_key in first_text:
            union_find.union(index, first_text[text_key])
        else:
            first_text[text_key] = index

    components: dict[int, list[int]] = {}
    for index in range(count):
        components.setdefault(union_find.find(index), []).append(index)
    if len(components) < 2:
        raise PreparationError(
            "The dataset is too small to create leakage-safe train and validation "
            "splits: all rows are connected by the same group or normalized transcript. "
            "Add at least one independent group with different text."
        )

    def component_rank(indices: list[int]) -> bytes:
        identities = [
            "\0".join(
                (
                    source_rows[index].group_field or "",
                    source_rows[index].group_id or "",
                    prepared_rows[index]["normalized_text"],
                    prepared_rows[index]["audio_sha256"],
                )
            )
            for index in indices
        ]
        payload = f"{seed}\0" + "\0".join(sorted(identities))
        return hashlib.sha256(payload.encode("utf-8")).digest()

    ranked = sorted(components.values(), key=component_rank)
    target = max(1, min(count - 1, int(round(count * fraction))))
    prefix_sizes: list[int] = []
    running = 0
    for component in ranked[:-1]:
        running += len(component)
        prefix_sizes.append(running)
    chosen_prefix = min(
        range(1, len(ranked)),
        key=lambda length: (
            abs(prefix_sizes[length - 1] - target),
            prefix_sizes[length - 1] > target,
            length,
        ),
    )
    validation_indices = {
        index for component in ranked[:chosen_prefix] for index in component
    }
    if not validation_indices or len(validation_indices) == count:
        raise PreparationError(
            "Could not produce nonempty train and validation splits. Add more independent "
            "groups or transcripts."
        )
    return validation_indices, {
        "strategy": "deterministic_group_aware_v1",
        "atomic_component_count": len(components),
        "explicit_group_count": len(first_group),
        "normalized_text_key_count": len(first_text),
        "target_validation_rows": target,
        "actual_validation_rows": len(validation_indices),
        "group_fields": [
            field
            for field in ("group_id", "session")
            if any(row.group_field == field for row in source_rows)
        ],
        "normalized_text_duplicates_co_located": True,
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _human_summary(dataset: dict[str, Any]) -> str:
    diagnostics = dataset["diagnostics"]
    return "\n".join(
        [
            "Inflect prepared dataset",
            "========================",
            f"Language: {dataset['language']}",
            f"Frontend: {dataset['frontend']}",
            f"Sample rate: {dataset['sample_rate']} Hz mono",
            f"Rows: {dataset['row_counts']['total']} "
            f"(train {dataset['row_counts']['train']}, "
            f"validation {dataset['row_counts']['validation']})",
            f"Audio duration: {diagnostics['total_duration_seconds']:.2f} seconds",
            f"Resampled files: {diagnostics['resampled_files']}",
            f"Downmixed files: {diagnostics['downmixed_files']}",
            f"Added symbols: {diagnostics['added_symbol_count']}",
            f"Base-symbol coverage before extension: "
            f"{diagnostics['base_symbol_coverage_fraction']:.6f}",
            "",
        ]
    )


def prepare_dataset(options: PrepareOptions) -> dict[str, Any]:
    """Prepare a manifest into the versioned public dataset layout.

    Output is assembled in a sibling staging directory and moved into place
    only after every row succeeds.
    """
    options.validate()
    manifest_path = Path(options.manifest_path).expanduser().resolve()
    output_dir = Path(options.output_dir).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise PreparationError(f"Output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PreparationError(
            f"Output directory is not empty: {output_dir}. Choose a new directory."
        )

    frontend_options = resolve(
        options.frontend,
        options.language,
        hook=options.frontend_hook,
    )
    validate_frontend(frontend_options)
    rows = parse_manifest(
        manifest_path,
        audio_root=options.audio_root,
        require_phonemes=options.frontend == "prephonemized",
    )
    speakers = sorted({row.speaker for row in rows if row.speaker})
    if len(speakers) > 1:
        raise PreparationError(
            "Single-speaker adaptation requires one consistent nonempty speaker value, "
            f"but the manifest contains {len(speakers)} values: "
            + ", ".join(repr(speaker) for speaker in speakers)
        )
    dataset_speaker = speakers[0] if speakers else None
    base_symbols = (
        load_base_symbols(options.base_symbols_path)
        if options.base_symbols_path
        else BASE_SYMBOLS
    )
    audio_options = AudioOptions(
        sample_rate=options.sample_rate,
        min_duration_seconds=options.min_duration_seconds,
        max_duration_seconds=options.max_duration_seconds,
    )

    stage = output_dir.with_name(f".{output_dir.name}.preparing-{uuid.uuid4().hex}")
    if stage.exists():
        raise PreparationError(f"Unexpected preparation staging collision: {stage}")
    (stage / "audio").mkdir(parents=True)

    prepared_rows: list[dict[str, Any]] = []
    audio_diagnostics: list[dict[str, Any]] = []
    seen_audio_hashes: dict[str, ManifestRow] = {}
    try:
        for row in rows:
            filename = _safe_name(row)
            destination = stage / "audio" / filename
            try:
                audio = convert_wav(row.audio_path, destination, audio_options)
                frontend = process_text(
                    row.text,
                    options=frontend_options,
                    prephonemized=row.phonemes,
                )
            except Exception as exc:
                raise PreparationError(f"Failed to prepare {row.source_location}: {exc}") from exc
            audio_sha256 = _sha256(destination)
            duplicate = seen_audio_hashes.get(audio_sha256)
            if duplicate is not None:
                raise PreparationError(
                    "Duplicate audio content detected after canonical 24 kHz conversion "
                    f"between {duplicate.source_location} and {row.source_location}. "
                    "Remove or replace one duplicate before training."
                )
            seen_audio_hashes[audio_sha256] = row
            prepared = {
                "audio": f"audio/{filename}",
                "audio_sha256": audio_sha256,
                "text": frontend.raw_text,
                "normalized_text": frontend.normalized_text,
                "phonemes": frontend.phonemes,
                "duration_seconds": round(audio.duration_seconds, 6),
            }
            if row.group_id:
                prepared["group_id"] = row.group_id
                prepared["group_field"] = row.group_field or "group_id"
            if row.speaker:
                prepared["speaker"] = row.speaker
            prepared_rows.append(prepared)
            audio_diagnostics.append(audio.to_dict())

        phoneme_texts = [row["phonemes"] for row in prepared_rows]
        base_coverage = audit_symbol_coverage(phoneme_texts, base_symbols)
        declared_symbols = custom_frontend_symbols(frontend_options) or ()
        inventory = build_symbol_inventory(
            phoneme_texts,
            base_symbols=base_symbols,
            extension_symbols=declared_symbols,
        )
        validation_indices, split_metadata = _split_components(
            prepared_rows,
            rows,
            fraction=options.validation_fraction,
            seed=options.split_seed,
        )
        train_rows = [
            row for index, row in enumerate(prepared_rows) if index not in validation_indices
        ]
        validation_rows = [
            row for index, row in enumerate(prepared_rows) if index in validation_indices
        ]
        _write_jsonl(stage / "train.jsonl", train_rows)
        _write_jsonl(stage / "validation.jsonl", validation_rows)
        write_symbol_inventory(stage / "symbols.json", inventory)

        # A bundled frontend is recorded as the custom frontend it resolves to,
        # so export and the deployment runtime keep their existing contract. The
        # registry name is preserved beside it for reproducibility.
        frontend_metadata: dict[str, Any] = {
            "type": frontend_options.mode,
            "language": options.language,
            "preserve_punctuation": frontend_options.preserve_punctuation,
            "with_stress": frontend_options.with_stress,
        }
        if is_registry_frontend(options.frontend):
            frontend_metadata["registry"] = registry_record(options.frontend)
        custom_metadata = custom_frontend_metadata(frontend_options)
        if custom_metadata is not None:
            frontend_metadata["hook"] = custom_metadata

        dataset = {
            "format": "inflect_prepared_dataset_v1",
            "language": options.language,
            "sample_rate": options.sample_rate,
            "channels": 1,
            "speaker": dataset_speaker,
            "frontend": frontend_metadata,
            "source_manifest_sha256": _sha256(manifest_path),
            "split_seed": options.split_seed,
            "validation_fraction": options.validation_fraction,
            "split": split_metadata,
            "row_counts": {
                "total": len(prepared_rows),
                "train": len(train_rows),
                "validation": len(validation_rows),
            },
            "diagnostics": {
                "total_duration_seconds": round(
                    sum(item["duration_seconds"] for item in audio_diagnostics), 6
                ),
                "min_duration_seconds": min(
                    item["duration_seconds"] for item in audio_diagnostics
                ),
                "max_duration_seconds": max(
                    item["duration_seconds"] for item in audio_diagnostics
                ),
                "resampled_files": sum(bool(item["resampled"]) for item in audio_diagnostics),
                "downmixed_files": sum(bool(item["downmixed"]) for item in audio_diagnostics),
                "source_clipped_files": sum(
                    item["source_clipped_fraction"] > 0 for item in audio_diagnostics
                ),
                # Clipping the conversion introduced, which is not the same
                # thing as clipping the recordings arrived with: a corpus
                # mastered near full scale loses samples to the resampler's
                # overshoot, and lowering the whole corpus by a few dB before
                # preparing is what removes it.
                "output_clipped_files": sum(
                    item["output_clipped_fraction"] > 0 for item in audio_diagnostics
                ),
                "base_symbol_coverage_fraction": base_coverage.coverage_fraction,
                "base_unknown_symbols": base_coverage.unknown_counts,
                "added_symbol_count": len(inventory.added_symbols),
            },
        }
        (stage / "dataset.json").write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report = {
            "format": "inflect_preparation_report_v1",
            "dataset": dataset,
            "audio": audio_diagnostics,
            "symbol_coverage_before_extension": base_coverage.to_dict(),
        }
        (stage / "preparation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (stage / "phoneme_coverage.json").write_text(
            json.dumps(
                {
                    "format": "inflect_phoneme_coverage_v1",
                    "base_inventory": base_coverage.to_dict(),
                    "added_symbols": list(inventory.added_symbols),
                    "adapted_inventory_size": len(inventory.symbols),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (stage / "PREPARATION_REPORT.txt").write_text(
            _human_summary(dataset), encoding="utf-8"
        )

        if output_dir.exists():
            output_dir.rmdir()
        stage.replace(output_dir)
        return dataset
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
