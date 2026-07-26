"""Lightweight held-out synthesis and signal diagnostics."""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import soundfile as sf
from scipy import signal

from .reporting import file_record, make_report, status, write_json, write_text


class Synthesizer(Protocol):
    def __call__(self, text: str | None = None, **kwargs: Any) -> Any: ...


class TranscriptEvaluator(Protocol):
    def __call__(
        self,
        audio_path: Path,
        reference_text: str,
        sample_rate: int,
    ) -> str | Mapping[str, Any]: ...


@dataclass(slots=True)
class EvaluationOptions:
    """Inputs for :func:`evaluate_checkpoint`.

    A custom ``synthesizer`` avoids coupling evaluation to a particular
    trainer. If omitted, ``model_dir/inference.py`` is loaded. A manifest can
    also reference existing audio, allowing diagnostics without model loading.
    No ASR model is installed or downloaded; transcript scoring only runs when
    the caller explicitly supplies ``transcript_evaluator``.
    """

    model_dir: str | Path | None
    manifest: str | Path
    output_dir: str | Path
    checkpoint: str | Path | None = None
    synthesizer: Synthesizer | None = None
    transcript_evaluator: TranscriptEvaluator | str | None = None
    device: str = "cpu"
    max_samples: int | None = None
    seed: int = 0
    speed: float = 1.0
    variation: float = 0.667
    overwrite: bool = False
    save_audio: bool = True
    clipping_threshold: float = 0.999
    silence_threshold_db: float = -50.0
    frame_ms: float = 25.0


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                text = row.get("normalized_text") or row.get("text")
                phonemes = (
                    row.get("phonemes")
                    or row.get("phoneme_text")
                    or row.get("phones")
                )
                has_text = isinstance(text, str) and bool(text.strip())
                has_phonemes = isinstance(phonemes, str) and bool(phonemes.strip())
                if not has_text and not has_phonemes:
                    raise ValueError(
                        f"line {line_number} has neither non-empty text nor phonemes"
                    )
                row["_line"] = line_number
                rows.append(row)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid evaluation manifest {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"Evaluation manifest is empty: {path}")
    return rows


def _safe_id(row: Mapping[str, Any], index: int) -> str:
    raw = str(row.get("id") or row.get("key") or f"sample-{index:04d}")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip(".-")
    return cleaned or f"sample-{index:04d}"


def _load_default_synthesizer(
    model_dir: Path,
    device: str,
    checkpoint: Path | None,
) -> Synthesizer:
    inference_path = model_dir / "inference.py"
    if not inference_path.is_file():
        raise FileNotFoundError(
            f"No inference.py was found in {model_dir}. Pass EvaluationOptions(synthesizer=...)."
        )
    module_name = f"_inflect_eval_inference_{abs(hash(inference_path))}"
    module_names = (
        "inference",
        "deployment_frontend",
        "inflect_vits_frontend",
        "models",
        "commons",
        "utils",
        "modules",
        "attentions",
        "transforms",
        "text",
        "text.symbols",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(model_dir))
        sys.path.insert(0, str(model_dir / "runtime"))
        for name in module_names:
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(module_name, inference_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import {inference_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        engine = module.InflectTTS(model_dir=model_dir, device=device)
    finally:
        sys.path[:] = old_path
        for name in module_names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]
    if checkpoint is not None and checkpoint.resolve() != (model_dir / "model.pth").resolve():
        from .exporting import _extract_state, _load_checkpoint

        state, _, _ = _extract_state(_load_checkpoint(checkpoint))
        incompatible = engine.model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Evaluation checkpoint did not load strictly: "
                f"missing={list(incompatible.missing_keys)}, "
                f"unexpected={list(incompatible.unexpected_keys)}"
            )
    return engine.synthesize


def _load_transcript_evaluator(
    value: TranscriptEvaluator | str | None,
) -> TranscriptEvaluator | None:
    if value is None or callable(value):
        return value
    if ":" not in value:
        raise ValueError("Transcript evaluator must use 'module_or_file:function' syntax.")
    source, function_name = value.rsplit(":", 1)
    path = Path(source)
    if path.is_file():
        spec = importlib.util.spec_from_file_location(
            f"_inflect_transcript_eval_{abs(hash(path.resolve()))}", path.resolve()
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import transcript evaluator: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        spec = importlib.util.find_spec(source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Transcript evaluator module is unavailable: {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    evaluator = getattr(module, function_name, None)
    if not callable(evaluator):
        raise ValueError(f"Transcript evaluator is not callable: {value}")
    return evaluator


def _invoke_synthesizer(
    synthesizer: Synthesizer,
    text: str | None,
    *,
    phonemes: str | None,
    seed: int,
    speed: float,
    variation: float,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    kwargs = {"seed": seed, "speed": speed, "variation": variation}
    if phonemes is not None:
        kwargs["phonemes"] = phonemes
    try:
        signature = inspect.signature(synthesizer)
        if not any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
            kwargs = {key: val for key, val in kwargs.items() if key in signature.parameters}
    except (TypeError, ValueError):
        pass
    result = synthesizer(text, **kwargs)
    metadata: dict[str, Any] = {}
    if isinstance(result, Mapping):
        sample_rate = result.get("sample_rate")
        waveform = result.get("waveform", result.get("audio"))
        metadata = dict(result.get("metadata", {}))
    elif isinstance(result, tuple) and len(result) >= 2:
        sample_rate, waveform = result[:2]
        if len(result) >= 3 and isinstance(result[2], Mapping):
            metadata = dict(result[2])
    else:
        raise TypeError(
            "Synthesizer must return (sample_rate, waveform) or a mapping with those values."
        )
    audio = np.asarray(waveform, dtype=np.float32).squeeze()
    if audio.ndim != 1:
        raise ValueError(f"Synthesizer returned non-mono audio with shape {audio.shape}.")
    return int(sample_rate), audio, metadata


def _load_existing_audio(path: Path) -> tuple[int, np.ndarray]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    return int(sample_rate), audio


def _leading_trailing_silence(mask: np.ndarray, sample_rate: int) -> tuple[float, float]:
    non_silent = np.flatnonzero(~mask)
    if not non_silent.size:
        duration = mask.size / sample_rate
        return duration, duration
    return non_silent[0] / sample_rate, (mask.size - 1 - non_silent[-1]) / sample_rate


def _signal_metrics(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    clipping_threshold: float,
    silence_threshold_db: float,
    frame_ms: float,
) -> dict[str, Any]:
    if waveform.size == 0:
        raise ValueError("Waveform is empty.")
    finite = np.isfinite(waveform)
    safe = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    absolute = np.abs(safe)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(safe))))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    silence_amplitude = 10.0 ** (silence_threshold_db / 20.0)
    sample_silence = absolute <= silence_amplitude
    leading, trailing = _leading_trailing_silence(sample_silence, sample_rate)
    frame_length = max(1, round(sample_rate * frame_ms / 1000.0))
    usable = safe[: (safe.size // frame_length) * frame_length]
    if usable.size:
        frame_rms = np.sqrt(np.mean(np.square(usable.reshape(-1, frame_length)), axis=1))
        silent_frames = float(np.mean(frame_rms <= silence_amplitude))
    else:
        silent_frames = float(np.mean(sample_silence))
    zero_crossing = (
        float(np.mean(np.signbit(safe[1:]) != np.signbit(safe[:-1])))
        if safe.size > 1
        else 0.0
    )
    nperseg = min(1024, safe.size)
    frequencies, spectrum = signal.welch(safe, fs=sample_rate, nperseg=nperseg)
    power_sum = float(np.sum(spectrum))
    if power_sum > 0:
        centroid = float(np.sum(frequencies * spectrum) / power_sum)
        high_frequency = frequencies >= min(8000, sample_rate * 0.4)
        high_ratio = float(np.sum(spectrum[high_frequency]) / power_sum)
    else:
        centroid = 0.0
        high_ratio = 0.0
    return {
        "sample_rate": sample_rate,
        "samples": int(safe.size),
        "duration_seconds": safe.size / sample_rate,
        "all_finite": bool(np.all(finite)),
        "non_finite_samples": int(np.count_nonzero(~finite)),
        "peak": peak,
        "rms": rms,
        "rms_dbfs": rms_dbfs,
        "dc_offset": float(np.mean(safe)),
        "clipped_fraction": float(np.mean(absolute >= clipping_threshold)),
        "silent_sample_fraction": float(np.mean(sample_silence)),
        "silent_frame_fraction": silent_frames,
        "leading_silence_seconds": leading,
        "trailing_silence_seconds": trailing,
        "zero_crossing_rate": zero_crossing,
        "crest_factor_db": 20.0 * math.log10(max(peak, 1e-12) / max(rms, 1e-12)),
        "spectral_centroid_hz": centroid,
        "high_frequency_energy_ratio": high_ratio,
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "duration_seconds",
        "peak",
        "rms_dbfs",
        "dc_offset",
        "clipped_fraction",
        "silent_sample_fraction",
        "silent_frame_fraction",
        "leading_silence_seconds",
        "trailing_silence_seconds",
        "zero_crossing_rate",
        "crest_factor_db",
        "spectral_centroid_hz",
        "high_frequency_energy_ratio",
        "characters_per_second",
        "words_per_second",
    )
    result: dict[str, Any] = {}
    for name in metric_names:
        values = [
            float(row["signal"][name] if name in row["signal"] else row[name])
            for row in rows
            if name in row.get("signal", {}) or name in row
        ]
        result[name] = {
            "mean": float(np.mean(values)) if values else None,
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
            "max": max(values) if values else None,
        }
    result["clips_with_clipping"] = sum(
        row["signal"]["clipped_fraction"] > 0 for row in rows
    )
    result["clips_all_silent"] = sum(
        row["signal"]["silent_sample_fraction"] >= 0.999 for row in rows
    )
    result["clips_with_non_finite_samples"] = sum(
        not row["signal"]["all_finite"] for row in rows
    )
    return result


def _run_transcript_evaluator(
    evaluator: TranscriptEvaluator,
    audio_path: Path,
    reference: str,
    sample_rate: int,
) -> dict[str, Any]:
    result = evaluator(audio_path, reference, sample_rate)
    if isinstance(result, str):
        return {"transcript": result}
    if isinstance(result, Mapping):
        return dict(result)
    raise TypeError("Transcript evaluator must return a string or mapping.")


def evaluate_checkpoint(options: EvaluationOptions) -> dict[str, Any]:
    """Synthesize/evaluate held-out rows and write JSON plus a short summary."""

    manifest = Path(options.manifest).resolve()
    output = Path(options.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not options.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "audio"
    rows = _read_manifest(manifest)
    if options.max_samples is not None:
        if options.max_samples <= 0:
            raise ValueError("max_samples must be positive when supplied.")
        rows = rows[: options.max_samples]

    model_dir = Path(options.model_dir).resolve() if options.model_dir is not None else None
    checkpoint = (
        Path(options.checkpoint).resolve() if options.checkpoint is not None else None
    )
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint}")
    synthesizer = options.synthesizer
    if synthesizer is None and not all(row.get("audio") for row in rows):
        if model_dir is None:
            raise ValueError("model_dir or a synthesizer is required for rows without audio.")
        synthesizer = _load_default_synthesizer(model_dir, options.device, checkpoint)
    transcript_evaluator = _load_transcript_evaluator(options.transcript_evaluator)
    if options.save_audio or transcript_evaluator is not None:
        audio_dir.mkdir(parents=True, exist_ok=True)

    evaluated: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample_id = _safe_id(row, index)
        raw_text = row.get("normalized_text") or row.get("text")
        text = str(raw_text).strip() if isinstance(raw_text, str) else ""
        raw_phonemes = (
            row.get("phonemes")
            or row.get("phoneme_text")
            or row.get("phones")
        )
        phonemes = (
            str(raw_phonemes).strip()
            if isinstance(raw_phonemes, str) and raw_phonemes.strip()
            else None
        )
        try:
            synthesis_metadata: dict[str, Any] = {}
            if synthesizer is not None:
                sample_rate, waveform, synthesis_metadata = _invoke_synthesizer(
                    synthesizer,
                    text or None,
                    phonemes=phonemes,
                    seed=options.seed + index,
                    speed=options.speed,
                    variation=options.variation,
                )
            else:
                source_audio = (manifest.parent / str(row["audio"])).resolve()
                try:
                    source_audio.relative_to(manifest.parent)
                except ValueError as exc:
                    raise ValueError(
                        f"Audio path escapes manifest directory: {row['audio']}"
                    ) from exc
                sample_rate, waveform = _load_existing_audio(source_audio)
            metrics = _signal_metrics(
                waveform,
                sample_rate,
                clipping_threshold=options.clipping_threshold,
                silence_threshold_db=options.silence_threshold_db,
                frame_ms=options.frame_ms,
            )
            duration = metrics["duration_seconds"]
            scoring_text = text or phonemes or ""
            words = len(scoring_text.split())
            result: dict[str, Any] = {
                "id": sample_id,
                "text": text,
                "input_mode": "prephonemized" if phonemes is not None else "text",
                "phoneme_characters": len(phonemes) if phonemes is not None else None,
                "seed": options.seed + index if synthesizer is not None else None,
                "signal": metrics,
                "characters_per_second": (
                    len(scoring_text) / duration if duration else None
                ),
                "words_per_second": words / duration if duration else None,
                "synthesis_metadata": synthesis_metadata,
            }
            destination = audio_dir / f"{sample_id}.wav"
            if options.save_audio or transcript_evaluator is not None:
                sf.write(destination, waveform, sample_rate, subtype="PCM_16")
                result["audio"] = file_record(destination, relative_to=output)
            if transcript_evaluator is not None:
                if not text:
                    raise ValueError(
                        "Transcript evaluation requires reference text even when "
                        "synthesis uses prephonemized input."
                    )
                result["transcript_evaluation"] = _run_transcript_evaluator(
                    transcript_evaluator,
                    destination,
                    str(row.get("text") or text),
                    sample_rate,
                )
            evaluated.append(result)
        except Exception as exc:
            failures.append(
                {
                    "id": sample_id,
                    "line": row.get("_line"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    checks = [
        status(bool(evaluated), "At least one held-out item evaluated successfully"),
        status(not failures, "All requested held-out items evaluated", failures=len(failures)),
    ]
    report = make_report(
        "evaluation_report",
        ok=bool(evaluated) and not failures,
        source={
            "manifest": manifest.name,
            "model_dir": model_dir.name if model_dir else None,
            "checkpoint": checkpoint.name if checkpoint else None,
            "device": options.device,
        },
        settings={
            "seed": options.seed,
            "speed": options.speed,
            "variation": options.variation,
            "clipping_threshold": options.clipping_threshold,
            "silence_threshold_db": options.silence_threshold_db,
            "frame_ms": options.frame_ms,
            "transcript_evaluator_enabled": transcript_evaluator is not None,
        },
        counts={"requested": len(rows), "evaluated": len(evaluated), "failed": len(failures)},
        checks=checks,
        aggregate=_aggregate(evaluated),
        items=evaluated,
        failures=failures,
    )
    write_json(output / "evaluation_report.json", report)
    summary = [
        "Inflect adaptation evaluation",
        f"Evaluated: {len(evaluated)}/{len(rows)}",
        f"Failures: {len(failures)}",
        f"Clips with clipping: {report['aggregate'].get('clips_with_clipping', 0)}",
        f"All-silent clips: {report['aggregate'].get('clips_all_silent', 0)}",
        "Transcript evaluator: "
        + ("enabled (caller supplied)" if transcript_evaluator else "disabled"),
        f"Result: {'PASS' if report['ok'] else 'FAIL'}",
    ]
    write_text(output / "evaluation_summary.txt", "\n".join(summary))
    return report
