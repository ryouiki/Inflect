from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from inflect_finetune.audio import AudioOptions, AudioValidationError, convert_wav, inspect_wav

SAMPLE_RATE = 48_000


def tone(seconds: float = 0.25, amplitude: float = 0.5) -> np.ndarray:
    time = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    return amplitude * np.sin(2.0 * np.pi * 220.0 * time)


@pytest.mark.parametrize(
    ("container", "subtype"),
    [("WAV", "PCM_16"), ("WAV", "PCM_24"), ("WAVEX", "PCM_24"), ("WAVEX", "PCM_16")],
)
def test_both_riff_wave_containers_are_accepted(tmp_path: Path, container, subtype) -> None:
    """WAVE_FORMAT_EXTENSIBLE is what sox and ffmpeg write above 16-bit.

    Rejecting it turns an ordinary 24-bit conversion step into a wall, and it is
    the same RIFF WAVE the plain header describes.
    """
    source = tmp_path / f"{container}-{subtype}.wav"
    sf.write(source, tone(), SAMPLE_RATE, format=container, subtype=subtype)
    assert sf.info(source).format == container
    assert inspect_wav(source).samplerate == SAMPLE_RATE
    diagnostics = convert_wav(source, tmp_path / f"out-{container}-{subtype}.wav")
    assert diagnostics.output_sample_rate == 24_000
    assert diagnostics.resampled is True


def test_a_file_that_is_not_riff_wave_is_still_refused(tmp_path: Path) -> None:
    """The check's purpose: another format wearing a .wav extension."""
    disguised = tmp_path / "actually-flac.wav"
    sf.write(disguised, tone(), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    with pytest.raises(AudioValidationError, match="not WAV"):
        inspect_wav(disguised)


def test_conversion_reports_the_clipping_it_had_to_do(tmp_path: Path) -> None:
    """Resampling overshoots, and the overshoot is clipped away silently.

    A corpus limited near full scale loses samples to that clip in the ordinary
    course of preparation, so the fraction is measured before the clip rather
    than left for the caller to infer from a peak of exactly 1.0.
    """
    source = tmp_path / "hot.wav"
    # Peak-limited just under full scale, the way a mastered corpus is, and with
    # energy well up the band so the anti-alias filter rings above 1.0. A
    # low-frequency tone does not reproduce it; speech does.
    limit = 0.9886
    generator = np.random.default_rng(20260904)
    samples = np.clip(3.0 * generator.standard_normal(SAMPLE_RATE // 2), -limit, limit)
    sf.write(source, samples, SAMPLE_RATE, subtype="PCM_24")
    diagnostics = convert_wav(source, tmp_path / "out.wav")
    assert diagnostics.source_peak == pytest.approx(limit, abs=1e-3)
    assert diagnostics.output_clipped_fraction > 0.0
    assert diagnostics.output_peak == pytest.approx(1.0, abs=1e-6)


def test_a_quiet_recording_reports_no_output_clipping(tmp_path: Path) -> None:
    source = tmp_path / "quiet.wav"
    sf.write(source, tone(amplitude=0.2), SAMPLE_RATE, subtype="PCM_16")
    diagnostics = convert_wav(source, tmp_path / "out.wav")
    assert diagnostics.output_clipped_fraction == 0.0
    assert diagnostics.output_peak < 1.0


def test_peak_limit_option_is_honoured(tmp_path: Path) -> None:
    source = tmp_path / "loud.wav"
    sf.write(source, tone(amplitude=0.9), SAMPLE_RATE, subtype="PCM_16")
    diagnostics = convert_wav(
        source, tmp_path / "out.wav", AudioOptions(peak_limit=0.5)
    )
    assert diagnostics.output_peak == pytest.approx(0.5, abs=1e-3)
    assert diagnostics.output_clipped_fraction > 0.0
