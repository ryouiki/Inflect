"""Audio validation and deterministic conversion for Inflect adaptation datasets."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class AudioValidationError(ValueError):
    """Raised when an input recording cannot be used safely for preparation."""


@dataclass(frozen=True)
class AudioOptions:
    """Generic audio validity and conversion settings."""

    sample_rate: int = 24_000
    min_duration_seconds: float = 0.05
    max_duration_seconds: float | None = None
    max_channels: int = 8
    peak_limit: float = 1.0
    output_subtype: str = "PCM_16"

    def validate(self) -> None:
        """Validate option values before reading any source files."""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.min_duration_seconds < 0:
            raise ValueError("min_duration_seconds cannot be negative.")
        if (
            self.max_duration_seconds is not None
            and self.max_duration_seconds <= self.min_duration_seconds
        ):
            raise ValueError("max_duration_seconds must exceed min_duration_seconds.")
        if self.max_channels < 1:
            raise ValueError("max_channels must be at least one.")
        if not 0 < self.peak_limit <= 1:
            raise ValueError("peak_limit must be in the interval (0, 1].")


@dataclass(frozen=True)
class AudioDiagnostics:
    """Machine-readable diagnostics for one converted recording."""

    source_sample_rate: int
    source_channels: int
    source_frames: int
    output_sample_rate: int
    output_frames: int
    duration_seconds: float
    source_peak: float
    output_peak: float
    source_clipped_fraction: float
    resampled: bool
    downmixed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def inspect_wav(path: Path, options: AudioOptions | None = None) -> sf.SoundFile:
    """Validate WAV container metadata and return the libsndfile descriptor.

    The returned object is metadata only; callers do not need to close it.
    """
    options = options or AudioOptions()
    options.validate()
    path = Path(path)
    if not path.is_file():
        raise AudioValidationError(f"Audio file does not exist: {path}")
    if path.suffix.lower() not in {".wav", ".wave"}:
        raise AudioValidationError(
            f"Expected a WAV file but received '{path.name}'. Convert it to WAV first."
        )
    try:
        info = sf.info(path)
    except (RuntimeError, TypeError) as exc:
        raise AudioValidationError(f"Could not read WAV metadata for {path}: {exc}") from exc
    if info.format != "WAV":
        raise AudioValidationError(
            f"{path} has extension .wav but container format '{info.format}' is not WAV."
        )
    if info.frames <= 0:
        raise AudioValidationError(f"Audio file is empty: {path}")
    if info.samplerate < 1_000 or info.samplerate > 384_000:
        raise AudioValidationError(
            f"Unsupported sample rate {info.samplerate} Hz in {path}; "
            "expected a conventional audio sample rate."
        )
    if info.channels < 1 or info.channels > options.max_channels:
        raise AudioValidationError(
            f"Unsupported channel count {info.channels} in {path}; "
            f"the configured maximum is {options.max_channels}."
        )
    duration = info.frames / info.samplerate
    if duration < options.min_duration_seconds:
        raise AudioValidationError(
            f"{path} is only {duration:.3f}s; minimum is "
            f"{options.min_duration_seconds:.3f}s."
        )
    if options.max_duration_seconds is not None and duration > options.max_duration_seconds:
        raise AudioValidationError(
            f"{path} is {duration:.3f}s; maximum is "
            f"{options.max_duration_seconds:.3f}s."
        )
    return info


def _mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise AudioValidationError(f"Expected one- or two-dimensional audio, got {audio.shape}.")
    return np.mean(audio, axis=1, dtype=np.float64)


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(
        audio,
        up=target_rate // divisor,
        down=source_rate // divisor,
        padtype="line",
    )


def convert_wav(
    source: Path,
    destination: Path,
    options: AudioOptions | None = None,
) -> AudioDiagnostics:
    """Validate, downmix, resample, and write one canonical 24 kHz mono WAV."""
    options = options or AudioOptions()
    info = inspect_wav(source, options)
    try:
        audio, sample_rate = sf.read(
            source,
            dtype="float64",
            always_2d=True,
            fill_value=0.0,
        )
    except (RuntimeError, TypeError) as exc:
        raise AudioValidationError(f"Could not decode WAV audio from {source}: {exc}") from exc
    if sample_rate != info.samplerate:
        raise AudioValidationError(
            f"Metadata/decode sample-rate mismatch in {source}: "
            f"{info.samplerate} versus {sample_rate}."
        )
    if not np.isfinite(audio).all():
        raise AudioValidationError(f"Audio contains NaN or infinite samples: {source}")

    source_peak = float(np.max(np.abs(audio), initial=0.0))
    source_clipped_fraction = float(np.mean(np.abs(audio) >= 0.999))
    mono = _mono(audio)
    converted = _resample(mono, sample_rate, options.sample_rate)
    if not np.isfinite(converted).all() or converted.size == 0:
        raise AudioValidationError(f"Audio conversion produced invalid samples for {source}.")
    converted = np.clip(converted, -options.peak_limit, options.peak_limit)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        sf.write(
            temporary,
            converted,
            options.sample_rate,
            format="WAV",
            subtype=options.output_subtype,
        )
        temporary.replace(destination)
    except (OSError, RuntimeError, TypeError) as exc:
        temporary.unlink(missing_ok=True)
        raise AudioValidationError(f"Could not write prepared WAV {destination}: {exc}") from exc

    output_info = sf.info(destination)
    return AudioDiagnostics(
        source_sample_rate=sample_rate,
        source_channels=info.channels,
        source_frames=info.frames,
        output_sample_rate=options.sample_rate,
        output_frames=output_info.frames,
        duration_seconds=output_info.frames / options.sample_rate,
        source_peak=source_peak,
        output_peak=float(np.max(np.abs(converted), initial=0.0)),
        source_clipped_fraction=source_clipped_fraction,
        resampled=sample_rate != options.sample_rate,
        downmixed=info.channels != 1,
    )
