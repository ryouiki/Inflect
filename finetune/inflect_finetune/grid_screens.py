"""Numpy-only screens for hop-grid ringing in vocoder output.

A VITS-family decoder that overfits its upsampler can emit a steady comb at
multiples of ``sample_rate / hop_length`` (93.75 Hz for 24 kHz audio at a hop of
256), audible even in silent frames as a metallic ring. The measurements here
are the ones that separated the failing adaptation renders from real recordings
during that diagnosis. The module deliberately depends on nothing but numpy and
``scipy.signal`` so that both the trainer and the evaluator can import it
without pulling in torch or an audio backend.

Two screens are exposed. :func:`grid_comb_metrics` is cheap enough to run on
every validation clip; :func:`steady_tone_artifact_score` reproduces the slower
long-term-average-spectrum analysis and is meant to stay behind a flag.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

_GRID_LTAS_NPERSEG = 8192
_GRID_BAND_TOP_HZ = 11_900.0
_GRID_BAND_NYQUIST_FRACTION = 0.496
_MIN_FOLD_FRAMES = 8

_ARTIFACT_NPERSEG = 2048
_ARTIFACT_FRAME_SECONDS = 0.05
_ARTIFACT_HOP_SECONDS = 0.025
_ARTIFACT_NFFT = 4096
_ACTIVE_REF_PERCENTILE = 95.0
_ACTIVE_REL_DB = -35.0
_ACTIVE_TARGET_DBFS = -23.0
_LTAS_SMOOTH_BINS = 5
_NEIGHBOURHOOD_OCTAVES = 1.0 / 6.0
_NEIGHBOURHOOD_EXCLUDE_BINS = 2
_NEIGHBOURHOOD_MIN_HZ = 40.0
_PEAK_MIN_HZ = 60.0
_PEAK_MIN_PROMINENCE_DB = 3.0
_LTAS_MAX_PEAKS = 12
_FRAME_PEAK_TOPK = 10
_FRAME_ENVELOPE_MEDIAN_BINS = 61
_STEADY_TOLERANCE_OCTAVES = 1.0 / 48.0
_STEADY_MIN_TOLERANCE_HZ = 12.0
_STEADY_MIN_FRACTION = 0.40
_ARTIFACT_MIN_PEAK_HZ = 1200.0

_POWER_FLOOR = 1e-20
_RMS_FLOOR = 1e-12


def _as_mono_float64(waveform: np.ndarray) -> np.ndarray:
    """Collapse any channel axes and promote to float64.

    The metrics are ratios of mean powers, which float32 accumulation biases
    noticeably over a multi-second clip.
    """
    samples = np.asarray(waveform, dtype=np.float64)
    while samples.ndim > 1:
        samples = samples.mean(axis=-1)
    return np.ascontiguousarray(samples)


def _power_db(numerator: float, denominator: float) -> float | None:
    """Power ratio in dB, or ``None`` when either side carries no power."""
    if not numerator > 0.0 or not denominator > 0.0:
        return None
    return float(10.0 * np.log10(numerator / denominator))


def _db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, _POWER_FLOOR))


def grid_comb_metrics(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    hop_length: int,
    grid_tolerance_hz: float = 4.4,
    band_hz: tuple[float, float | None] = (2000.0, None),
) -> dict[str, float | None]:
    """Measure how much of a clip sits on the decoder's frame grid.

    ``grid_tone_level_db`` and ``off_grid_level_db`` report those two band
    powers against the clip's own signal power, which is what to compare when
    two renders differ in level or in noise floor.

    ``grid_tone_excess_db`` compares mean power in bins within
    ``grid_tolerance_hz`` of a grid multiple against every other bin in the
    band: it is 0 dB by construction for real speech and +6..+9 dB for the
    ringing renders. ``fold_periodic_db`` folds the waveform at the hop period
    and reports the surviving power, but its floor is ``-10*log10(frames)`` for
    an uncorrelated signal and therefore moves with clip length, so
    ``fold_periodic_excess_db`` restates it against that floor and is the value
    callers threshold on.

    Unmeasurable values are ``None`` rather than a number that could be mistaken
    for a passing score. Only caller bugs raise.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive.")
    samples = _as_mono_float64(waveform)
    if samples.size == 0:
        raise ValueError("waveform must contain at least one sample.")

    grid_hz = float(sample_rate) / float(hop_length)
    frames = int(samples.size // hop_length)
    metrics: dict[str, float | None] = {
        "frame_grid_hz": grid_hz,
        "grid_tone_excess_db": None,
        "grid_tone_level_db": None,
        "off_grid_level_db": None,
        "fold_periodic_db": None,
        "fold_periodic_excess_db": None,
        "fold_periodic_frames": frames,
    }

    if samples.size >= _GRID_LTAS_NPERSEG:
        freqs, psd = signal.welch(
            samples,
            fs=sample_rate,
            nperseg=_GRID_LTAS_NPERSEG,
            noverlap=_GRID_LTAS_NPERSEG // 2,
            window="hann",
        )
        low_hz = float(band_hz[0])
        high_hz = (
            float(band_hz[1])
            if band_hz[1] is not None
            else min(_GRID_BAND_TOP_HZ, _GRID_BAND_NYQUIST_FRACTION * sample_rate)
        )
        in_band = (freqs >= low_hz) & (freqs <= high_hz)
        band_freqs = freqs[in_band]
        band_psd = psd[in_band]
        deviation_hz = np.abs(band_freqs - np.round(band_freqs / grid_hz) * grid_hz)
        on_grid = deviation_hz <= grid_tolerance_hz
        if on_grid.any() and not on_grid.all():
            metrics["grid_tone_excess_db"] = _power_db(
                float(band_psd[on_grid].mean()), float(band_psd[~on_grid].mean())
            )
            # The excess is a ratio, and its denominator is the render's own
            # broadband floor. That makes it a good detector and a bad
            # comparator: two renders whose floors differ can be ranked
            # backwards by it, which happened on a real pair here. These two
            # report the same two band powers against the clip's own level
            # instead, so a caller comparing renders can see which term moved.
            signal_power = float(np.mean(samples**2))
            metrics["grid_tone_level_db"] = _power_db(
                float(band_psd[on_grid].mean()), signal_power
            )
            metrics["off_grid_level_db"] = _power_db(
                float(band_psd[~on_grid].mean()), signal_power
            )

    if frames >= _MIN_FOLD_FRAMES:
        fold = samples[: frames * hop_length].reshape(frames, hop_length).mean(axis=0)
        fold_db = _power_db(float(np.mean(fold**2)), float(np.mean(samples**2)))
        if fold_db is not None:
            metrics["fold_periodic_db"] = fold_db
            metrics["fold_periodic_excess_db"] = fold_db + 10.0 * float(np.log10(frames))

    return metrics


def f0_grid_deviation_hz(
    f0_hz: float | None, frame_grid_hz: float, *, max_multiple: int = 3
) -> float | None:
    """Distance from a measured pitch to the nearest low multiple of the grid.

    A pitch tracker fed a ringing render locks onto the comb instead of the
    voice and reports a "pitch" of 93.75 Hz; on the failing runs that happened
    on 134 of 160 clips, so a near-zero deviation is a strong screen.
    """
    if frame_grid_hz <= 0:
        raise ValueError("frame_grid_hz must be positive.")
    if max_multiple < 1:
        raise ValueError("max_multiple must be at least one.")
    if f0_hz is None or not f0_hz > 0:
        return None
    multiples = np.arange(1, max_multiple + 1, dtype=np.float64) * float(frame_grid_hz)
    return float(np.min(np.abs(multiples - float(f0_hz))))


def _active_frames(
    samples: np.ndarray, frame_length: int, hop_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping analysis frames plus a mask of the ones carrying speech.

    Activity is judged relative to the clip's own 95th-percentile frame level so
    that a quiet render is not screened out wholesale.
    """
    count = 1 + max(0, (samples.size - frame_length) // hop_length)
    index = np.arange(frame_length)[None, :] + hop_length * np.arange(count)[:, None]
    frames = samples[index]
    rms = np.sqrt(np.mean(frames**2, axis=1) + _RMS_FLOOR)
    rms_db = 20.0 * np.log10(rms + _RMS_FLOOR)
    reference_db = float(np.percentile(rms_db, _ACTIVE_REF_PERCENTILE))
    return frames, rms_db > (reference_db + _ACTIVE_REL_DB)


def _neighbourhood_median(ltas_db: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Median level over a 1/6-octave neighbourhood, excluding the bin itself.

    Subtracting this leaves the prominence of a narrow line without the broad
    spectral tilt of the voice.
    """
    out = np.zeros_like(ltas_db)
    total = len(freqs)
    exclude = _NEIGHBOURHOOD_EXCLUDE_BINS
    for i in range(total):
        centre_hz = max(float(freqs[i]), _NEIGHBOURHOOD_MIN_HZ)
        low_hz = centre_hz * 2.0 ** (-_NEIGHBOURHOOD_OCTAVES / 2.0)
        high_hz = centre_hz * 2.0 ** (_NEIGHBOURHOOD_OCTAVES / 2.0)
        low = int(np.searchsorted(freqs, low_hz))
        high = int(np.searchsorted(freqs, high_hz))
        low = max(0, min(low, i - exclude - 3))
        high = min(total, max(high, i + exclude + 4))
        segment = np.concatenate(
            [ltas_db[low : max(low, i - exclude)], ltas_db[min(total, i + exclude + 1) : high]]
        )
        out[i] = np.median(segment) if len(segment) else ltas_db[i]
    return out


def _frame_peak_freqs(frames: np.ndarray, sample_rate: int, frame_length: int) -> list[np.ndarray]:
    """Per-frame frequencies of the strongest lines in the whitened spectrum."""
    window = signal.get_window("hann", frame_length)
    spectra = _db(np.abs(np.fft.rfft(frames * window, n=_ARTIFACT_NFFT, axis=1)) ** 2)
    freqs = np.fft.rfftfreq(_ARTIFACT_NFFT, 1.0 / sample_rate)
    peaks: list[np.ndarray] = []
    for spectrum in spectra:
        residual = spectrum - signal.medfilt(spectrum, _FRAME_ENVELOPE_MEDIAN_BINS)
        maxima = (
            np.where((residual[1:-1] > residual[:-2]) & (residual[1:-1] >= residual[2:]))[0] + 1
        )
        maxima = maxima[freqs[maxima] > _PEAK_MIN_HZ]
        if len(maxima) == 0:
            peaks.append(np.array([]))
            continue
        order = maxima[np.argsort(residual[maxima])[::-1][:_FRAME_PEAK_TOPK]]
        peaks.append(freqs[order])
    return peaks


def _steadiness(peak_hz: float, frame_peaks: list[np.ndarray]) -> float:
    """Fraction of active frames in which ``peak_hz`` is itself a frame peak."""
    if not frame_peaks:
        return 0.0
    tolerance_hz = max(_STEADY_MIN_TOLERANCE_HZ, peak_hz * (2.0**_STEADY_TOLERANCE_OCTAVES - 1.0))
    hits = sum(
        1
        for found in frame_peaks
        if len(found) and bool(np.any(np.abs(found - peak_hz) <= tolerance_hz))
    )
    return hits / len(frame_peaks)


def steady_tone_artifact_score(waveform: np.ndarray, sample_rate: int) -> float | None:
    """Summed prominence (dB) of steady long-term spectral peaks above 1200 Hz.

    Real recordings score near 0 because their prominent lines move with the
    voice; a ringing render holds the same lines for most of the clip, so their
    prominences accumulate. This costs 0.3-1.5 s per clip, so callers keep it
    behind a flag and rely on :func:`grid_comb_metrics` for the always-on screen.

    ``None`` means the clip was too short or too quiet to measure, not that it
    passed.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    samples = _as_mono_float64(waveform)
    if samples.size == 0:
        raise ValueError("waveform must contain at least one sample.")

    frame_length = max(1, round(_ARTIFACT_FRAME_SECONDS * sample_rate))
    hop_length = max(1, round(_ARTIFACT_HOP_SECONDS * sample_rate))
    if samples.size < frame_length:
        return None

    frames, active = _active_frames(samples, frame_length, hop_length)
    if not active.any():
        return None

    gain = 10.0 ** (_ACTIVE_TARGET_DBFS / 20.0) / np.sqrt(np.mean(frames[active] ** 2) + _RMS_FLOOR)
    samples = samples * gain
    frames = frames * gain

    block_count = samples.size // frame_length
    blocks = samples[: block_count * frame_length].reshape(block_count, frame_length)
    block_db = 20.0 * np.log10(np.sqrt(np.mean(blocks**2, axis=1)) + _RMS_FLOOR)
    block_active = block_db > (
        float(np.percentile(block_db, _ACTIVE_REF_PERCENTILE)) + _ACTIVE_REL_DB
    )
    active_samples = blocks[block_active].reshape(-1) if block_active.any() else samples
    if active_samples.size < _ARTIFACT_NPERSEG:
        return None

    freqs, psd = signal.welch(
        active_samples,
        fs=sample_rate,
        window="hann",
        nperseg=_ARTIFACT_NPERSEG,
        noverlap=_ARTIFACT_NPERSEG // 2,
        detrend=False,
    )
    smoothed = np.convolve(_db(psd), np.ones(_LTAS_SMOOTH_BINS) / _LTAS_SMOOTH_BINS, mode="same")
    prominence = smoothed - _neighbourhood_median(smoothed, freqs)
    maxima = (
        np.where((prominence[1:-1] > prominence[:-2]) & (prominence[1:-1] >= prominence[2:]))[0] + 1
    )
    maxima = maxima[(freqs[maxima] > _PEAK_MIN_HZ) & (prominence[maxima] > _PEAK_MIN_PROMINENCE_DB)]
    maxima = maxima[np.argsort(prominence[maxima])[::-1][:_LTAS_MAX_PEAKS]]
    if len(maxima) == 0:
        return 0.0

    frame_peaks = _frame_peak_freqs(frames[active], sample_rate, frame_length)
    score = 0.0
    for index in maxima:
        peak_hz = float(freqs[index])
        if peak_hz <= _ARTIFACT_MIN_PEAK_HZ:
            continue
        if _steadiness(peak_hz, frame_peaks) >= _STEADY_MIN_FRACTION:
            score += float(prominence[index])
    return score
