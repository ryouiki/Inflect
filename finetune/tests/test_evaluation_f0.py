from __future__ import annotations

import math

import numpy as np
import pytest

from inflect_finetune.evaluation import _aggregate, _f0_metrics, _signal_metrics

SAMPLE_RATE = 24_000
SILENCE = 10.0 ** (-50.0 / 20.0)


def tone(frequency: float, seconds: float = 1.0, amplitude: float = 0.3) -> np.ndarray:
    time = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    # A couple of harmonics, so the frame looks like voiced speech rather than a
    # pure sinusoid, which is the easy case for autocorrelation.
    wave = (
        np.sin(2 * np.pi * frequency * time)
        + 0.5 * np.sin(4 * np.pi * frequency * time)
        + 0.25 * np.sin(6 * np.pi * frequency * time)
    )
    return (amplitude * wave / np.max(np.abs(wave))).astype(np.float64)


def glide(start: float, end: float, seconds: float = 1.0, amplitude: float = 0.3) -> np.ndarray:
    time = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    instantaneous = start * (end / start) ** (time / seconds)
    phase = 2 * np.pi * np.cumsum(instantaneous) / SAMPLE_RATE
    wave = np.sin(phase) + 0.5 * np.sin(2 * phase)
    return (amplitude * wave / np.max(np.abs(wave))).astype(np.float64)


def measure(waveform: np.ndarray, **overrides) -> dict:
    options = {"f0_min_hz": 60.0, "f0_max_hz": 1000.0, "silence_amplitude": SILENCE}
    options.update(overrides)
    return _f0_metrics(waveform, SAMPLE_RATE, **options)


@pytest.mark.parametrize("frequency", [110.0, 220.0, 440.0, 660.0, 880.0])
def test_a_steady_tone_reports_its_own_pitch(frequency):
    metrics = measure(tone(frequency))
    assert metrics["f0_median_hz"] == pytest.approx(frequency, rel=0.02)
    assert metrics["voiced_frame_fraction"] > 0.95


def test_the_ceiling_does_not_clip_a_high_female_register():
    """A 760 Hz endpoint is inside the range; a 650 Hz ceiling would fold it down."""
    high = tone(760.0)
    assert measure(high)["f0_median_hz"] == pytest.approx(760.0, rel=0.02)
    clipped = measure(high, f0_max_hz=650.0)
    assert clipped["f0_median_hz"] is None or clipped["f0_median_hz"] < 700.0


def test_a_flat_contour_and_a_moving_one_share_a_median_but_not_an_iqr():
    """The reason the median is never read alone.

    A register objective that only moves the median has a degenerate solution
    where the pitch goes flat, so the interquartile range is what distinguishes
    a live contour from a flattened one.
    """
    moving = measure(glide(220.0, 440.0))
    flat = measure(tone(311.0))  # the geometric middle of the glide
    assert moving["f0_median_hz"] == pytest.approx(flat["f0_median_hz"], rel=0.06)
    assert flat["f0_iqr_semitones"] == pytest.approx(0.0, abs=0.3)
    assert moving["f0_iqr_semitones"] > 4.0


def test_silence_has_no_pitch_rather_than_a_pitch_of_zero():
    silent = np.zeros(SAMPLE_RATE, dtype=np.float64)
    metrics = measure(silent)
    assert metrics["f0_median_hz"] is None
    assert metrics["f0_iqr_semitones"] is None
    assert metrics["voiced_frame_fraction"] == 0.0


def test_noise_is_mostly_unvoiced():
    generator = np.random.default_rng(20260904)
    noise = 0.2 * generator.standard_normal(SAMPLE_RATE)
    assert measure(noise)["voiced_frame_fraction"] < 0.5


def test_a_doubled_period_does_not_halve_the_reported_pitch():
    """Autocorrelation's classic failure: the octave below correlates too."""
    metrics = measure(tone(300.0))
    assert metrics["f0_median_hz"] == pytest.approx(300.0, rel=0.02)


def test_measurement_is_deterministic():
    waveform = glide(180.0, 300.0)
    first = measure(waveform)
    second = measure(waveform)
    assert first == second


def test_a_clip_too_short_to_hold_three_periods_reports_nothing():
    metrics = measure(tone(220.0, seconds=0.01))
    assert metrics["f0_median_hz"] is None
    assert metrics["voiced_frame_fraction"] == 0.0


def test_an_inverted_range_is_rejected():
    with pytest.raises(ValueError):
        measure(tone(220.0), f0_min_hz=800.0, f0_max_hz=200.0)


def test_signal_metrics_carries_the_pitch_observables():
    metrics = _signal_metrics(
        tone(240.0),
        SAMPLE_RATE,
        clipping_threshold=0.999,
        silence_threshold_db=-50.0,
        frame_ms=25.0,
    )
    assert metrics["f0_median_hz"] == pytest.approx(240.0, rel=0.02)
    assert metrics["f0_iqr_semitones"] is not None
    assert 0.0 <= metrics["voiced_frame_fraction"] <= 1.0
    assert math.isfinite(metrics["spectral_centroid_hz"])


def test_aggregate_skips_rows_whose_pitch_was_unmeasurable():
    voiced = _signal_metrics(
        tone(240.0),
        SAMPLE_RATE,
        clipping_threshold=0.999,
        silence_threshold_db=-50.0,
        frame_ms=25.0,
    )
    silent = _signal_metrics(
        np.zeros(SAMPLE_RATE, dtype=np.float64),
        SAMPLE_RATE,
        clipping_threshold=0.999,
        silence_threshold_db=-50.0,
        frame_ms=25.0,
    )
    assert silent["f0_median_hz"] is None
    aggregate = _aggregate([{"signal": voiced}, {"signal": silent}])
    assert aggregate["f0_median_hz"]["p50"] == pytest.approx(240.0, rel=0.02)
    assert aggregate["voiced_frame_fraction"]["max"] > 0.9
    assert aggregate["clips_all_silent"] == 1
