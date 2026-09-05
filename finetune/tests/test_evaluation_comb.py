"""Screens for the hop-grid comb that made the first adaptation unusable.

The failing renders rang at multiples of 93.75 Hz (24000 / 256, the decoder's
upsample grid). These tests fix the behaviour of the four measurements that
separated those renders from real recordings, and of their wiring into the
evaluator, so a later edit cannot quietly turn a screen into something that
always passes.

The synthetic fixtures below stand in for the two ends of that separation: an
impulse train is the comb in its purest form, and ``voiced`` is a broadband
harmonic signal whose partials avoid the grid, which is what real speech looks
like to these screens.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import pytest

from inflect_finetune.evaluation import _aggregate, _signal_metrics
from inflect_finetune.grid_screens import (
    f0_grid_deviation_hz,
    grid_comb_metrics,
    steady_tone_artifact_score,
)

SAMPLE_RATE = 24_000
HOP_LENGTH = 256
FRAME_GRID_HZ = SAMPLE_RATE / HOP_LENGTH  # 93.75
NOISE_SEED = 20260905

# Real audio, if this machine happens to be the one the diagnosis ran on.
ANCHOR_AUDIO = Path("/home/ysoya/inflect-work/prepared/ko-arona-v1b/audio")
RINGING_RENDERS = Path("/home/ysoya/inflect-work/evals/ko-arona-micro-direct-20260904-final-round/audio")


def comb(period: int = HOP_LENGTH, seconds: float = 2.0, amplitude: float = 0.5) -> np.ndarray:
    """An impulse train: a flat comb at every multiple of ``sample_rate / period``."""
    samples = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float64)
    samples[::period] = amplitude
    return samples


def noise(seconds: float = 2.0, amplitude: float = 0.2, seed: int = NOISE_SEED) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return amplitude * generator.standard_normal(int(seconds * SAMPLE_RATE))


def comb_in_noise(
    period: int = HOP_LENGTH,
    level_db: float = -10.0,
    seconds: float = 2.0,
    seed: int = NOISE_SEED,
) -> np.ndarray:
    """Noise plus an impulse train whose RMS sits ``level_db`` below the noise.

    The realistic shape of the artifact: the comb is well under the voice, and
    the screen still has to see it.
    """
    background = noise(seconds, seed=seed)
    train = comb(period, seconds, amplitude=1.0)
    scale = np.sqrt(np.mean(background**2)) * 10.0 ** (level_db / 20.0)
    mixed = background + train * (scale / np.sqrt(np.mean(train**2)))
    return 0.7 * mixed / np.max(np.abs(mixed))


def voiced(
    f0: float = 283.0,
    seconds: float = 2.0,
    top_hz: float = 11_900.0,
    amplitude: float = 0.3,
    aspiration_db: float = -6.0,
    seed: int = 7,
) -> np.ndarray:
    """A harmonic stack with a 1/k rolloff plus broadband aspiration noise.

    283 Hz is chosen because no partial of it inside the 2-11.9 kHz screening
    band comes within 14 Hz of a multiple of 93.75 Hz, well outside the 4.4 Hz
    on-grid tolerance. The aspiration noise is what makes the upper band
    broadband rather than a handful of isolated lines, which is the property
    that puts real speech at 0 dB of grid excess.
    """
    time = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    wave = np.zeros_like(time)
    harmonic = 1
    while f0 * harmonic <= top_hz:
        wave += np.sin(2 * np.pi * f0 * harmonic * time) / harmonic
        harmonic += 1
    wave = wave / np.max(np.abs(wave))
    generator = np.random.default_rng(seed)
    wave = wave + 10.0 ** (aspiration_db / 20.0) * generator.standard_normal(time.size)
    return amplitude * wave / np.max(np.abs(wave))


def tone(frequency: float, seconds: float = 2.0, amplitude: float = 0.3) -> np.ndarray:
    """A held tone with a couple of harmonics (same generator as test_evaluation_f0)."""
    time = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    wave = (
        np.sin(2 * np.pi * frequency * time)
        + 0.5 * np.sin(4 * np.pi * frequency * time)
        + 0.25 * np.sin(6 * np.pi * frequency * time)
    )
    return (amplitude * wave / np.max(np.abs(wave))).astype(np.float64)


def glide(start: float, end: float, seconds: float = 2.0, amplitude: float = 0.3) -> np.ndarray:
    """A sweeping tone (same generator as test_evaluation_f0)."""
    time = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    instantaneous = start * (end / start) ** (time / seconds)
    phase = 2 * np.pi * np.cumsum(instantaneous) / SAMPLE_RATE
    wave = np.sin(phase) + 0.5 * np.sin(2 * phase)
    return (amplitude * wave / np.max(np.abs(wave))).astype(np.float64)


def measure(waveform: np.ndarray, **overrides) -> dict:
    options = {"hop_length": HOP_LENGTH}
    options.update(overrides)
    return grid_comb_metrics(waveform, SAMPLE_RATE, **options)


def screen(waveform: np.ndarray, **overrides) -> dict:
    """`_signal_metrics` with the non-screen arguments it insists on."""
    options = {
        "clipping_threshold": 0.999,
        "silence_threshold_db": -50.0,
        "frame_ms": 25.0,
        "hop_length": HOP_LENGTH,
    }
    options.update(overrides)
    return _signal_metrics(waveform, SAMPLE_RATE, **options)


SCREEN_VALUE_KEYS = (
    "frame_grid_hz",
    "grid_tone_excess_db",
    "fold_periodic_db",
    "fold_periodic_excess_db",
    "fold_periodic_frames",
    "f0_grid_deviation_hz",
    "steady_tone_artifact_score",
)
SCREEN_FLAG_KEYS = (
    "grid_tone_flagged",
    "fold_periodic_flagged",
    "f0_grid_locked",
    "steady_tone_flagged",
)


def test_an_impulse_train_on_the_hop_grid_is_flagged_by_both_comb_screens():
    """The artifact in its purest form must be unmissable.

    Measured on a mathematically perfect 256-sample train: grid excess
    +324.9 dB (the off-grid bins hold only float noise) and fold excess
    +22.7 dB. The failing renders measured a far more modest +6..+9 dB, so the
    bounds here are only asked to be unambiguous.
    """
    metrics = measure(comb())
    assert metrics["frame_grid_hz"] == FRAME_GRID_HZ
    assert metrics["grid_tone_excess_db"] > 30.0
    assert metrics["fold_periodic_excess_db"] > 15.0


def test_a_comb_ten_db_under_the_noise_is_still_flagged():
    """The realistic case, and the one the thresholds were set for.

    Measured at -10 dB relative to the noise: grid excess +3.19 dB and fold
    excess +12.50 dB, both past the evaluator's default flags of 2.0 and 3.0.
    """
    metrics = measure(comb_in_noise(level_db=-10.0))
    assert metrics["grid_tone_excess_db"] > 2.5
    assert metrics["fold_periodic_excess_db"] > 8.0


@pytest.mark.parametrize("seed", [1, 2, 3, NOISE_SEED])
def test_white_noise_sits_at_chance_on_both_comb_screens(seed):
    """Neither screen may accuse a signal that has no structure at all.

    Across these four seeds the observed spread was -0.09..+0.05 dB of grid
    excess and -0.85..+0.45 dB of fold excess.
    """
    metrics = measure(noise(seed=seed))
    assert abs(metrics["grid_tone_excess_db"]) < 0.5
    assert abs(metrics["fold_periodic_excess_db"]) < 1.5


def test_only_the_excess_form_of_the_fold_screen_is_stable_across_clip_length():
    """Why two fold numbers are reported instead of one.

    The raw fold value's floor for an uncorrelated signal is -10*log10(frames),
    so it falls about 3 dB per doubling of the clip: measured -19.89, -22.58 and
    -25.68 dB over 1, 2 and 4 seconds of the same noise. Thresholding that raw
    number is how the diagnosis first mistook clip length for an artifact. The
    excess form restates it against the floor and stayed inside +/-0.21 dB.
    """
    full = noise(4.0)
    raw = {}
    excess = {}
    for seconds in (1.0, 2.0, 4.0):
        metrics = measure(full[: int(seconds * SAMPLE_RATE)])
        raw[seconds] = metrics["fold_periodic_db"]
        excess[seconds] = metrics["fold_periodic_excess_db"]
        assert metrics["fold_periodic_frames"] == int(seconds * SAMPLE_RATE) // HOP_LENGTH

    assert raw[1.0] > raw[2.0] > raw[4.0]
    assert raw[1.0] - raw[4.0] > 4.0  # measured 5.79 dB, i.e. two 3 dB steps
    assert max(abs(value) for value in excess.values()) < 0.6
    assert abs(excess[1.0] - excess[4.0]) < 0.5


def test_a_harmonic_voice_whose_partials_miss_the_grid_shows_no_grid_excess():
    """0 dB by construction for real speech is the whole basis of the screen.

    Measured -0.18 dB of grid excess and +0.09 dB of fold excess for the 283 Hz
    stack, which is the range real recordings landed in (-0.1..+0.2 dB).
    """
    metrics = measure(voiced())
    assert abs(metrics["grid_tone_excess_db"]) < 1.0
    assert abs(metrics["fold_periodic_excess_db"]) < 1.0
    assert not screen(voiced())["grid_tone_flagged"]


def test_the_grid_the_screen_looks_on_follows_the_hop_it_is_given():
    """The comb is only an artifact at the decoder's own frame rate.

    An 80 Hz train (period 300) is exactly on the grid for a hop of 300 and
    nowhere near it for a hop of 256: measured +31.75 dB against +0.07 dB.
    """
    eighty_hz = comb(period=300)
    on_grid = measure(eighty_hz, hop_length=300)
    off_grid = measure(eighty_hz, hop_length=HOP_LENGTH)
    assert on_grid["frame_grid_hz"] == 80.0
    assert off_grid["frame_grid_hz"] == FRAME_GRID_HZ
    assert on_grid["grid_tone_excess_db"] > 20.0
    assert off_grid["grid_tone_excess_db"] < 2.0  # under the default flag threshold


def test_a_clip_too_short_to_analyse_reports_nothing_rather_than_accusing():
    """A screen that cannot measure must not flag.

    1000 samples is below the 8192-point Welch segment and below the eight-frame
    minimum for the fold, and below the 50 ms frame the steady-tone screen needs.
    """
    short = comb()[:1000]
    metrics = measure(short)
    assert metrics["grid_tone_excess_db"] is None
    assert metrics["fold_periodic_db"] is None
    assert metrics["fold_periodic_excess_db"] is None
    assert metrics["fold_periodic_frames"] == 3
    assert steady_tone_artifact_score(short, SAMPLE_RATE) is None

    flags = screen(short, steady_tone_screen=True)
    for key in ("grid_tone_excess_db", "fold_periodic_excess_db", "steady_tone_artifact_score"):
        assert flags[key] is None
    for key in ("grid_tone_flagged", "fold_periodic_flagged", "steady_tone_flagged"):
        assert flags[key] is False
    # The pitch screen is the exception, and deliberately so: it needs three
    # periods of the lowest searchable pitch, not an 8192-point spectrum, so it
    # still measures here and still finds the comb at 93.75 Hz. A screen must
    # stay silent when it cannot measure, not when its neighbours cannot.
    assert flags["f0_grid_locked"] is True
    assert flags["f0_grid_deviation_hz"] < 1e-4


def test_the_f0_grid_screen_counts_a_locked_pitch_and_leaves_an_honest_one_alone():
    """A pitch tracker fed a ringing render reports the comb as the voice.

    The positive fixture is an impulse train rather than a 93.75 Hz sine because
    the estimator in `_f0_metrics` needs harmonic structure: a sine with two
    harmonics at that frequency reports no pitch at all, while the train is
    measured at 93.74999830 Hz.
    """
    assert f0_grid_deviation_hz(FRAME_GRID_HZ, FRAME_GRID_HZ) == 0.0
    assert f0_grid_deviation_hz(2 * FRAME_GRID_HZ, FRAME_GRID_HZ) == 0.0
    # 220 Hz sits between the second and third multiple, and the nearer one is
    # the second: 220 - 187.5 = 32.5, against 281.25 - 220 = 61.25.
    assert f0_grid_deviation_hz(220.0, FRAME_GRID_HZ) == pytest.approx(32.5)
    # max_multiple is honoured: the fourth multiple is out of reach by default.
    assert f0_grid_deviation_hz(375.0, FRAME_GRID_HZ, max_multiple=3) == pytest.approx(93.75)
    assert f0_grid_deviation_hz(375.0, FRAME_GRID_HZ, max_multiple=4) == 0.0
    assert f0_grid_deviation_hz(None, FRAME_GRID_HZ) is None
    assert f0_grid_deviation_hz(0.0, FRAME_GRID_HZ) is None

    ringing = screen(comb())
    assert ringing["f0_median_hz"] == pytest.approx(FRAME_GRID_HZ, abs=1e-4)
    assert ringing["f0_grid_deviation_hz"] < 1e-4
    assert ringing["f0_grid_locked"] is True

    # The estimator reads this fixture at 216.7 Hz, 29.2 Hz off the nearest
    # multiple, which is twenty times the lock tolerance.
    honest = screen(tone(220.0))
    assert honest["f0_grid_deviation_hz"] > 20.0
    assert honest["f0_grid_locked"] is False


def test_the_steady_tone_score_rewards_a_held_high_line_and_ignores_a_moving_one():
    """Steadiness above 1200 Hz is what separated the renders from real audio.

    Measured: a held 1433 Hz tone scores 154.6, a 800->3200 Hz glide scores
    exactly 0.0 because nothing holds still, and a held 300 Hz tone also scores
    exactly 0.0 because all of its partials are below the 1200 Hz gate.
    """
    assert steady_tone_artifact_score(glide(800.0, 3200.0), SAMPLE_RATE) == 0.0
    assert steady_tone_artifact_score(tone(300.0), SAMPLE_RATE) == 0.0
    assert steady_tone_artifact_score(tone(1433.0), SAMPLE_RATE) > 20.0


def test_measurement_is_deterministic():
    waveform = comb_in_noise()
    assert measure(waveform) == measure(waveform)
    assert steady_tone_artifact_score(waveform, SAMPLE_RATE) == steady_tone_artifact_score(
        waveform, SAMPLE_RATE
    )


def test_signal_metrics_carries_every_screen_and_never_drops_a_flag():
    """The aggregate indexes the flags directly, so they must always be there."""
    ringing = screen(comb(), steady_tone_screen=True)
    for key in SCREEN_VALUE_KEYS + SCREEN_FLAG_KEYS:
        assert key in ringing
    assert ringing["frame_grid_hz"] == FRAME_GRID_HZ
    assert math.isfinite(ringing["grid_tone_excess_db"])
    for key in SCREEN_FLAG_KEYS:
        assert isinstance(ringing[key], bool)

    unmeasurable = screen(comb()[:1000])
    assert unmeasurable["grid_tone_excess_db"] is None
    for key in SCREEN_VALUE_KEYS + SCREEN_FLAG_KEYS:
        assert key in unmeasurable
    for key in SCREEN_FLAG_KEYS:
        assert isinstance(unmeasurable[key], bool)


def test_aggregate_counts_the_four_screens_and_skips_what_could_not_be_measured():
    """One row trips the comb screens, one trips only the steady-tone screen.

    The unmeasurable row must not be averaged in as a zero, which would drag a
    ringing run's reported grid excess back under the flag.
    """
    ringing = screen(comb())
    held = screen(tone(1433.0), steady_tone_screen=True)
    unmeasurable = screen(comb()[:1000])

    assert unmeasurable["grid_tone_excess_db"] is None
    aggregate = _aggregate(
        [{"signal": ringing}, {"signal": held}, {"signal": unmeasurable}]
    )
    assert aggregate["clips_grid_tone_flagged"] == 1
    assert aggregate["clips_fold_periodic_flagged"] == 1
    assert aggregate["clips_steady_tone_flagged"] == 1
    # Two, not one: the truncated row is too short for the spectral screens but
    # not for the pitch tracker, and its pitch is the comb.
    assert aggregate["clips_f0_locked_to_frame_grid"] == 2

    # Two rows carry a grid excess, so the mean is theirs alone; a third zero
    # would pull it down by a third.
    assert aggregate["grid_tone_excess_db"]["mean"] == pytest.approx(
        (ringing["grid_tone_excess_db"] + held["grid_tone_excess_db"]) / 2.0
    )
    assert aggregate["grid_tone_excess_db"]["max"] == ringing["grid_tone_excess_db"]
    assert aggregate["fold_periodic_db"]["max"] == ringing["fold_periodic_db"]
    assert aggregate["fold_periodic_excess_db"]["max"] == ringing["fold_periodic_excess_db"]
    assert aggregate["f0_grid_deviation_hz"]["max"] == held["f0_grid_deviation_hz"]
    # The steady-tone screen runs by default, so two rows carry a score and the
    # truncated one is skipped rather than averaged in as a zero. The impulse
    # train scores nothing: its peaks are not steady, which is why the comb
    # needs the grid screen and not this one.
    assert ringing["steady_tone_artifact_score"] == 0.0
    assert unmeasurable["steady_tone_artifact_score"] is None
    assert aggregate["steady_tone_artifact_score"]["mean"] == pytest.approx(
        (ringing["steady_tone_artifact_score"] + held["steady_tone_artifact_score"]) / 2.0
    )
    assert aggregate["steady_tone_artifact_score"]["max"] == held["steady_tone_artifact_score"]


def test_the_flag_thresholds_come_from_the_options_and_are_not_hardcoded():
    """The same clip flags or does not, purely on the configured threshold.

    The fixture measures +3.19 dB of grid excess. It is deliberately a clip the
    shipped 4.0 dB default lets through: the thresholds were set from one
    speaker pair on one render channel, so a caller retuning them must actually
    change the verdict.
    """
    waveform = comb_in_noise(level_db=-10.0)
    assert screen(waveform)["grid_tone_excess_db"] == pytest.approx(3.19, abs=0.05)
    assert screen(waveform, grid_tone_flag_db=2.0)["grid_tone_flagged"] is True
    assert screen(waveform)["grid_tone_flagged"] is False
    assert screen(waveform, grid_tone_flag_db=5.0)["grid_tone_flagged"] is False
    assert screen(waveform, fold_periodic_excess_flag_db=20.0)["fold_periodic_flagged"] is False

    steady = screen(tone(1433.0), steady_tone_screen=True)
    assert steady["steady_tone_flagged"] is True
    assert screen(
        tone(1433.0), steady_tone_screen=True, steady_tone_flag=1000.0
    )["steady_tone_flagged"] is False
    # Turning the screen off removes the score, and with it the accusation.
    off = screen(tone(1433.0), steady_tone_screen=False)
    assert off["steady_tone_artifact_score"] is None
    assert off["steady_tone_flagged"] is False


def test_caller_mistakes_raise_instead_of_returning_a_passing_score():
    with pytest.raises(ValueError):
        measure(comb(), hop_length=0)
    with pytest.raises(ValueError):
        grid_comb_metrics(comb(), 0, hop_length=HOP_LENGTH)
    with pytest.raises(ValueError):
        measure(np.array([], dtype=np.float64))
    with pytest.raises(ValueError):
        f0_grid_deviation_hz(200.0, FRAME_GRID_HZ, max_multiple=0)
    with pytest.raises(ValueError):
        f0_grid_deviation_hz(200.0, 0.0)
    with pytest.raises(ValueError):
        steady_tone_artifact_score(comb(), 0)
    with pytest.raises(ValueError):
        steady_tone_artifact_score(np.array([], dtype=np.float64), SAMPLE_RATE)


# --------------------------------------------------------------------------
# Opt-in: the same separation on the real recordings and the real renders
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (ANCHOR_AUDIO.is_dir() and RINGING_RENDERS.is_dir()),
    reason="Needs the diagnosis working directory under /home/ysoya/inflect-work.",
)
def test_real_recordings_and_ringing_renders_separate_on_the_grid_screen():
    """The measurement that pinned the fault, against the audio that showed it.

    Medians over the first eight clips of each set, measured by hand: -0.07 dB
    for the anchor recordings and +7.99 dB for the ringing renders.
    """
    import soundfile as sf

    def median_grid_excess(directory: Path) -> float:
        values = []
        for path in sorted(glob.glob(str(directory / "*.wav")))[:8]:
            waveform, sample_rate = sf.read(path, dtype="float64", always_2d=False)
            excess = grid_comb_metrics(waveform, sample_rate, hop_length=HOP_LENGTH)[
                "grid_tone_excess_db"
            ]
            if excess is not None:
                values.append(excess)
        assert values, f"No measurable audio in {directory}"
        return float(np.median(values))

    anchor = median_grid_excess(ANCHOR_AUDIO)
    render = median_grid_excess(RINGING_RENDERS)
    assert abs(anchor) < 1.0
    assert render > 4.0
    assert render - anchor > 4.0
