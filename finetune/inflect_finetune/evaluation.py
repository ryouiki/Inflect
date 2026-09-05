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

from .grid_screens import f0_grid_deviation_hz, grid_comb_metrics, steady_tone_artifact_score
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
    # The pitch search range. The ceiling is deliberately well above a speaking
    # voice: a target whose questions end near 800 Hz reads as a falling contour
    # if the ceiling clips it, which is an artifact of the setting rather than
    # anything the model did.
    f0_min_hz: float = 60.0
    f0_max_hz: float = 1000.0
    # Screens for a comb at multiples of the frame rate. The grid is derived
    # from the hop, which is read from the model config when not given here.
    # The thresholds flag; they never select. A clip sitting just inside one
    # has proven nothing, and a listener still decides.
    #
    # Measured on 40 real recordings against 40 renders from a run a listener
    # rejected for ringing (p50 / max, then flagged clips at these defaults):
    #   grid_tone_excess_db         real -0.13 / 3.50   rings 8.15 / 9.77   0/40 vs 40/40
    #   fold_periodic_excess_db     real -0.16 / 5.19   rings 4.17 / 8.35   0/40 vs 29/40
    #   steady_tone_artifact_score  real  0.00 / 0.00   rings 29.9 / 66.3   0/40 vs 40/40
    # The fold measure overlaps, so it corroborates rather than accuses; its
    # threshold is set where a flag still means something. The other two
    # separate the two populations completely.
    hop_length: int | None = None
    grid_tone_flag_db: float = 4.0
    fold_periodic_excess_flag_db: float = 6.0
    f0_grid_lock_tolerance_hz: float = 1.5
    f0_grid_lock_max_multiple: int = 3
    # Costs 0.02-0.04 s per clip, cheap enough to leave on for the screen with
    # the cleanest separation of the three.
    steady_tone_screen: bool = True
    steady_tone_flag: float = 5.0


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


def _f0_metrics(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    f0_min_hz: float,
    f0_max_hz: float,
    silence_amplitude: float,
) -> dict[str, Any]:
    """Return the pitch observables, by normalized autocorrelation.

    Three numbers, and the second is the reason the first is reported at all: a
    register objective that only moves the median has a degenerate solution
    where the contour goes flat, which measures as success and sounds worse. The
    interquartile range in semitones is what shows the contour still moving, so
    the two are always read together.

    The estimator is deliberately plain — autocorrelation over Hann-windowed
    frames, sub-sample refinement, and a bias toward the shortest candidate
    period so a doubled lag cannot halve the reported pitch. It is a screen for
    register collapse and pitch flattening, not a pitch tracker.
    """
    if not 0 < f0_min_hz < f0_max_hz:
        raise ValueError("f0_min_hz must be positive and below f0_max_hz.")
    # Three periods of the lowest pitch, so the lowest lag still has support.
    frame_length = min(waveform.size, math.ceil(3.0 * sample_rate / f0_min_hz))
    hop_length = max(1, round(sample_rate * 0.010))
    min_lag = max(2, math.floor(sample_rate / f0_max_hz))
    max_lag = math.ceil(sample_rate / f0_min_hz)
    if frame_length <= max_lag or waveform.size < frame_length:
        return {"f0_median_hz": None, "f0_iqr_semitones": None, "voiced_frame_fraction": 0.0}

    window = np.hanning(frame_length)
    padded = int(1 << (2 * frame_length - 1).bit_length())
    starts = range(0, waveform.size - frame_length + 1, hop_length)
    frequencies: list[float] = []
    frames = 0
    for start in starts:
        frames += 1
        frame = waveform[start : start + frame_length]
        if float(np.sqrt(np.mean(np.square(frame)))) <= silence_amplitude:
            continue
        centred = (frame - float(np.mean(frame))) * window
        energy = float(np.dot(centred, centred))
        if energy <= 0:
            continue
        spectrum = np.fft.rfft(centred, n=padded)
        correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=padded)[: max_lag + 1]
        normalized = correlation / energy
        search = normalized[min_lag : max_lag + 1]
        if search.size == 0:
            continue
        best = float(np.max(search))
        # A periodic frame correlates with itself; an unvoiced one does not.
        if best < 0.45:
            continue
        # The shortest lag that is nearly as strong as the best one. Picking the
        # global maximum alone reports half the pitch whenever a multiple of the
        # period correlates marginally better.
        candidates = np.flatnonzero(search >= 0.85 * best)
        offset = int(candidates[0]) if candidates.size else int(np.argmax(search))
        lag = min_lag + offset
        if 0 < lag < max_lag:
            previous, current, following = (
                float(normalized[lag - 1]),
                float(normalized[lag]),
                float(normalized[lag + 1]),
            )
            denominator = previous - 2.0 * current + following
            if denominator != 0:
                lag += 0.5 * (previous - following) / denominator
        if lag > 0:
            frequency = sample_rate / lag
            if f0_min_hz <= frequency <= f0_max_hz:
                frequencies.append(frequency)

    if not frequencies:
        return {
            "f0_median_hz": None,
            "f0_iqr_semitones": None,
            "voiced_frame_fraction": 0.0,
        }
    values = np.asarray(frequencies, dtype=np.float64)
    lower, upper = (float(value) for value in np.percentile(values, [25.0, 75.0]))
    return {
        "f0_median_hz": float(np.median(values)),
        "f0_iqr_semitones": 12.0 * math.log2(upper / lower) if lower > 0 else 0.0,
        "voiced_frame_fraction": len(frequencies) / frames if frames else 0.0,
    }


def _grid_screens(
    waveform: np.ndarray,
    sample_rate: int,
    f0_median_hz: float | None,
    *,
    hop_length: int,
    grid_tone_flag_db: float,
    fold_periodic_excess_flag_db: float,
    f0_grid_lock_tolerance_hz: float,
    f0_grid_lock_max_multiple: int,
    steady_tone_screen: bool,
    steady_tone_flag: float,
) -> dict[str, Any]:
    """Measure the frame-rate comb and say which thresholds it crosses.

    The boolean keys are always present, and false when a value could not be
    measured, because the aggregate counts them by direct indexing.
    """

    screens = dict(
        grid_comb_metrics(waveform, sample_rate, hop_length=hop_length)
    )
    grid_hz = screens["frame_grid_hz"]
    deviation = f0_grid_deviation_hz(
        f0_median_hz, grid_hz, max_multiple=f0_grid_lock_max_multiple
    )
    score = (
        steady_tone_artifact_score(waveform, sample_rate) if steady_tone_screen else None
    )
    excess = screens["grid_tone_excess_db"]
    fold_excess = screens["fold_periodic_excess_db"]
    screens.update(
        {
            "f0_grid_deviation_hz": deviation,
            "steady_tone_artifact_score": score,
            "grid_tone_flagged": excess is not None and excess > grid_tone_flag_db,
            "fold_periodic_flagged": (
                fold_excess is not None and fold_excess > fold_periodic_excess_flag_db
            ),
            "f0_grid_locked": (
                deviation is not None and deviation <= f0_grid_lock_tolerance_hz
            ),
            "steady_tone_flagged": score is not None and score > steady_tone_flag,
        }
    )
    return screens


def _signal_metrics(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    clipping_threshold: float,
    silence_threshold_db: float,
    frame_ms: float,
    f0_min_hz: float = 60.0,
    f0_max_hz: float = 1000.0,
    hop_length: int = 256,
    grid_tone_flag_db: float = 4.0,
    fold_periodic_excess_flag_db: float = 6.0,
    f0_grid_lock_tolerance_hz: float = 1.5,
    f0_grid_lock_max_multiple: int = 3,
    steady_tone_screen: bool = True,
    steady_tone_flag: float = 5.0,
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
    pitch = _f0_metrics(
        safe,
        sample_rate,
        f0_min_hz=f0_min_hz,
        f0_max_hz=f0_max_hz,
        silence_amplitude=silence_amplitude,
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
        **pitch,
        **_grid_screens(
            safe,
            sample_rate,
            pitch["f0_median_hz"],
            hop_length=hop_length,
            grid_tone_flag_db=grid_tone_flag_db,
            fold_periodic_excess_flag_db=fold_periodic_excess_flag_db,
            f0_grid_lock_tolerance_hz=f0_grid_lock_tolerance_hz,
            f0_grid_lock_max_multiple=f0_grid_lock_max_multiple,
            steady_tone_screen=steady_tone_screen,
            steady_tone_flag=steady_tone_flag,
        ),
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
        "f0_median_hz",
        "f0_iqr_semitones",
        "voiced_frame_fraction",
        "grid_tone_excess_db",
        "fold_periodic_db",
        "fold_periodic_excess_db",
        "f0_grid_deviation_hz",
        "steady_tone_artifact_score",
        "characters_per_second",
        "words_per_second",
    )
    result: dict[str, Any] = {}
    for name in metric_names:
        # A metric can be genuinely unmeasurable for a row — an unvoiced clip has
        # no pitch — so a missing value is skipped rather than counted as zero.
        raw = [
            row["signal"][name] if name in row.get("signal", {}) else row.get(name)
            for row in rows
            if name in row.get("signal", {}) or name in row
        ]
        values = [float(value) for value in raw if value is not None]
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
    result["clips_grid_tone_flagged"] = sum(
        row["signal"]["grid_tone_flagged"] for row in rows
    )
    result["clips_fold_periodic_flagged"] = sum(
        row["signal"]["fold_periodic_flagged"] for row in rows
    )
    result["clips_f0_locked_to_frame_grid"] = sum(
        row["signal"]["f0_grid_locked"] for row in rows
    )
    result["clips_steady_tone_flagged"] = sum(
        row["signal"]["steady_tone_flagged"] for row in rows
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


def _resolve_hop_length(options: EvaluationOptions, model_dir: Path | None) -> int:
    """Take the frame hop from the package being evaluated, not from a guess.

    The comb the screens look for sits at multiples of sample rate over hop, so
    a hop that does not belong to this model would measure the wrong
    frequencies and quietly report nothing.
    """

    if options.hop_length is not None:
        return int(options.hop_length)
    if model_dir is not None:
        config_path = model_dir / "config.json"
        if config_path.is_file():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                return int(payload["data"]["hop_length"])
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError,
                    ValueError):
                pass
    return 256


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
    hop_length = _resolve_hop_length(options, model_dir)
    checkpoint = (
        Path(options.checkpoint).resolve() if options.checkpoint is not None else None
    )
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint}")
    # The source is chosen once for the whole manifest, so it has to be counted
    # from the rows as they arrive. A mixed manifest synthesizes every row and
    # ignores the audio fields of the ones that had them, and no count taken
    # afterwards could show that: it would report every row as synthesized,
    # which is true and is exactly the thing worth knowing about.
    rows_with_audio = sum(bool(row.get("audio")) for row in rows)
    synthesizer = options.synthesizer
    if synthesizer is None and rows_with_audio < len(rows):
        if model_dir is None:
            raise ValueError("model_dir or a synthesizer is required for rows without audio.")
        synthesizer = _load_default_synthesizer(model_dir, options.device, checkpoint)
    mode = "synthesis" if synthesizer is not None else "existing_audio"
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
                f0_min_hz=options.f0_min_hz,
                f0_max_hz=options.f0_max_hz,
                hop_length=hop_length,
                grid_tone_flag_db=options.grid_tone_flag_db,
                fold_periodic_excess_flag_db=options.fold_periodic_excess_flag_db,
                f0_grid_lock_tolerance_hz=options.f0_grid_lock_tolerance_hz,
                f0_grid_lock_max_multiple=options.f0_grid_lock_max_multiple,
                steady_tone_screen=options.steady_tone_screen,
                steady_tone_flag=options.steady_tone_flag,
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
        status(
            not 0 < rows_with_audio < len(rows),
            "The manifest asks for one source rather than a mixture",
            mode=mode,
            rows_with_audio=rows_with_audio,
            rows=len(rows),
        ),
    ]
    report = make_report(
        "evaluation_report",
        # `ok` deliberately does not consult `checks`. Measuring the reference
        # recordings through this path is a documented workflow, and the model
        # directory is a required flag, so a report that read the anchors is a
        # correct report; the mixture check is there to be read, not to fail a
        # run whose meaning was never in doubt.
        ok=bool(evaluated) and not failures,
        source={
            "manifest": manifest.name,
            # The model directory is recorded whether or not a model was
            # opened, so `mode` is the field that says which happened.
            "mode": mode,
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
            "f0_min_hz": options.f0_min_hz,
            "f0_max_hz": options.f0_max_hz,
            "hop_length": hop_length,
            "frame_grid_hz": None if not evaluated else evaluated[0]["signal"]["frame_grid_hz"],
            "grid_tone_flag_db": options.grid_tone_flag_db,
            "fold_periodic_excess_flag_db": options.fold_periodic_excess_flag_db,
            "f0_grid_lock_tolerance_hz": options.f0_grid_lock_tolerance_hz,
            "f0_grid_lock_max_multiple": options.f0_grid_lock_max_multiple,
            "steady_tone_screen": options.steady_tone_screen,
            "steady_tone_flag": options.steady_tone_flag,
            "transcript_evaluator_enabled": transcript_evaluator is not None,
        },
        counts={
            "requested": len(rows),
            "evaluated": len(evaluated),
            "failed": len(failures),
            "synthesized": len(evaluated) if synthesizer is not None else 0,
            "read_from_manifest_audio": 0 if synthesizer is not None else len(evaluated),
            "manifest_audio_ignored": rows_with_audio if synthesizer is not None else 0,
        },
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
        (
            f"Source: {mode}, synthesized {report['counts']['synthesized']}, "
            f"read from manifest audio {report['counts']['read_from_manifest_audio']}"
            + (
                f", manifest audio ignored {report['counts']['manifest_audio_ignored']}"
                if report["counts"]["manifest_audio_ignored"]
                else ""
            )
        ),
        f"Clips with clipping: {report['aggregate'].get('clips_with_clipping', 0)}",
        f"All-silent clips: {report['aggregate'].get('clips_all_silent', 0)}",
        (
            "Frame-grid comb flags (grid tone/fold/F0 lock/steady tone): "
            f"{report['aggregate'].get('clips_grid_tone_flagged', 0)}/"
            f"{report['aggregate'].get('clips_fold_periodic_flagged', 0)}/"
            f"{report['aggregate'].get('clips_f0_locked_to_frame_grid', 0)}/"
            f"{report['aggregate'].get('clips_steady_tone_flagged', 0)}"
            f" of {len(evaluated)}"
        ),
        (
            "Grid-tone excess dB p50/max: "
            f"{(report['aggregate'].get('grid_tone_excess_db') or {}).get('p50')}/"
            f"{(report['aggregate'].get('grid_tone_excess_db') or {}).get('max')}"
        ),
        "These flag; they do not select. A blind listening round decides.",
        "Transcript evaluator: "
        + ("enabled (caller supplied)" if transcript_evaluator else "disabled"),
        f"Result: {'PASS' if report['ok'] else 'FAIL'}",
    ]
    write_text(output / "evaluation_summary.txt", "\n".join(summary))
    return report
