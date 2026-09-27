"""Clipping the conversion introduces is refused, measured before the clip.

The failure this guards against: a corpus limited just under full scale arrives
unclipped, the 48 -> 24 kHz resampler rings above 1.0, and the peak limit cuts
the overshoot. The written files then peak at exactly 1.0 and look like any
other loud recording, so the verdict has to come from the pre-clip samples.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from inflect_finetune.audio import AudioOptions, convert_wav
from inflect_finetune.audit import AuditOptions, audit_dataset
from inflect_finetune.cli import build_parser, main
from inflect_finetune.prepare import (
    RECOMMENDED_PEAK_DBFS,
    OutputClippingError,
    PrepareOptions,
    prepare_dataset,
    recommended_input_gain_db,
)

SOURCE_RATE = 48_000
LIMIT = 0.9886  # under the 0.999 source-clipping threshold, as a limited master is


def _hot(seed: int, seconds: float = 0.5) -> np.ndarray:
    """Peak-limited broadband material that only exceeds 1.0 once resampled."""
    generator = np.random.default_rng(seed)
    samples = 3.0 * generator.standard_normal(int(SOURCE_RATE * seconds))
    return np.clip(samples, -LIMIT, LIMIT)


def _quiet(frequency: float, rate: int = SOURCE_RATE) -> np.ndarray:
    time = np.arange(rate // 4, dtype=np.float64) / rate
    return 0.3 * np.sin(2.0 * np.pi * frequency * time)


def _corpus(root: Path, signals: list[np.ndarray], rate: int = SOURCE_RATE) -> Path:
    root.mkdir(parents=True)
    lines = []
    for index, signal in enumerate(signals):
        name = f"row-{index}.wav"
        sf.write(root / name, signal, rate, subtype="PCM_24")
        lines.append(
            json.dumps({"audio": name, "text": f"Row {index}.", "phonemes": f"row {index}"})
        )
    manifest = root / "metadata.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _conversion(output: Path) -> dict:
    report = json.loads((output / "preparation_report.json").read_text(encoding="utf-8"))
    return report["conversion"]


def _prepare(manifest: Path, output: Path, gain: float = 0.0) -> dict:
    return prepare_dataset(
        PrepareOptions(
            manifest_path=manifest,
            output_dir=output,
            frontend="prephonemized",
            validation_fraction=0.34,
            split_seed=7,
            input_gain_db=gain,
        )
    )


def test_the_regression_signal_is_in_range_until_it_is_resampled(tmp_path: Path) -> None:
    """The test input must reproduce the cause, not just be loud."""
    signal = _hot(1)
    assert np.max(np.abs(signal)) < 0.999
    assert np.max(np.abs(resample_poly(signal, 1, 2, padtype="line"))) > 1.0

    source = tmp_path / "hot.wav"
    sf.write(source, signal, SOURCE_RATE, subtype="PCM_24")
    diagnostics = convert_wav(source, tmp_path / "out.wav")
    assert diagnostics.source_clipped_fraction == 0.0
    assert diagnostics.pre_clip_peak > 1.0
    assert diagnostics.output_clipped_samples > 0
    assert diagnostics.output_peak == pytest.approx(1.0, abs=1e-6)


def test_prepare_refuses_a_clipped_conversion_and_keeps_the_evidence(
    tmp_path: Path,
) -> None:
    manifest = _corpus(tmp_path / "source", [_hot(seed) for seed in (1, 2, 3)])
    output = tmp_path / "prepared"
    with pytest.raises(OutputClippingError, match="--input-gain-db") as caught:
        _prepare(manifest, output)

    assert not output.exists(), "a clipped dataset must never look prepared"
    assert not list(tmp_path.glob(".prepared.preparing-*"))
    report = json.loads(caught.value.report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "rejected"
    assert report["output_clipped_files"] == 3
    assert report["source_clipped_files"] == 0
    assert report["max_pre_clip_peak"] > 1.0
    assert report["recommended_input_gain_db"] == caught.value.recommended_gain_db
    assert caught.value.recommended_gain_db == recommended_input_gain_db(
        0.0, report["max_pre_clip_peak"]
    )
    assert len(report["clipped_rows"]) == 3
    assert all(row["audio"].startswith("audio/") for row in report["clipped_rows"])


def test_the_recommended_gain_prepares_cleanly_and_passes_audit(tmp_path: Path) -> None:
    manifest = _corpus(tmp_path / "source", [_hot(seed) for seed in (1, 2, 3)])
    with pytest.raises(OutputClippingError) as caught:
        _prepare(manifest, tmp_path / "hot")
    gain = caught.value.recommended_gain_db
    assert gain < 0.0

    output = tmp_path / "attenuated"
    dataset = _prepare(manifest, output, gain=gain)
    assert dataset["diagnostics"]["output_clipped_files"] == 0
    assert dataset["audio_processing"]["input_gain_db"] == gain
    conversion = _conversion(output)
    assert conversion["output_clipped_samples"] == 0
    assert conversion["max_pre_clip_peak"] <= 10.0 ** (RECOMMENDED_PEAK_DBFS / 20.0)

    report = audit_dataset(AuditOptions(prepared_dir=output))
    assert report["valid"]
    assert not any("output_clipped_files" in warning for warning in report["warnings"])


def test_recommended_gain_follows_the_stated_formula() -> None:
    # G_new = G + 20 log10(T / P), rounded down to 0.1 dB.
    peak = 1.021
    exact = 0.0 + RECOMMENDED_PEAK_DBFS - 20.0 * math.log10(peak)
    assert recommended_input_gain_db(0.0, peak) == math.floor(exact * 10) / 10
    assert recommended_input_gain_db(0.0, peak) == -1.2
    # -1.509 exactly; rounding goes toward more attenuation, never less.
    assert recommended_input_gain_db(0.0, 1.0603) == -1.6
    # Already attenuated: the new gain is relative to the one that produced P.
    assert recommended_input_gain_db(-3.0, 1.05) == -4.5
    assert recommended_input_gain_db(-3.0, 1.05) < -3.0


def test_default_gain_leaves_ordinary_output_unchanged(tmp_path: Path) -> None:
    """Gain 0 must reproduce exactly what the conversion wrote before it existed."""
    signal = _quiet(220.0)
    source = tmp_path / "quiet.wav"
    sf.write(source, signal, SOURCE_RATE, subtype="PCM_24")
    decoded, _ = sf.read(source, dtype="float64")
    reference = tmp_path / "reference.wav"
    sf.write(
        reference,
        resample_poly(decoded, 1, 2, padtype="line"),
        24_000,
        format="WAV",
        subtype="PCM_16",
    )

    default = convert_wav(source, tmp_path / "default.wav")
    explicit = convert_wav(
        source, tmp_path / "explicit.wav", AudioOptions(input_gain_db=0.0)
    )
    expected = hashlib.sha256(reference.read_bytes()).hexdigest()
    for name in ("default.wav", "explicit.wav"):
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected
    assert default.input_gain_db == explicit.input_gain_db == 0.0
    assert default.output_clipped_samples == 0


def test_gain_applies_to_input_that_is_already_24_khz(tmp_path: Path) -> None:
    signal = _quiet(330.0, rate=24_000)
    source = tmp_path / "native.wav"
    sf.write(source, signal, 24_000, subtype="PCM_24")
    diagnostics = convert_wav(
        source, tmp_path / "out.wav", AudioOptions(input_gain_db=-6.0)
    )
    assert diagnostics.resampled is False
    written, _ = sf.read(tmp_path / "out.wav", dtype="float64")
    np.testing.assert_allclose(written, signal * 10.0 ** (-6.0 / 20.0), atol=1e-4)
    assert diagnostics.pre_clip_peak == pytest.approx(0.3 * 10.0 ** (-6.0 / 20.0), rel=1e-3)


def test_source_clipping_stays_reported_after_attenuation(tmp_path: Path) -> None:
    """Lowering the corpus removes our clipping, not the clipping it arrived with."""
    signal = np.clip(5.0 * _quiet(200.0), -1.0, 1.0)
    source = tmp_path / "clipped-at-source.wav"
    sf.write(source, signal, SOURCE_RATE, subtype="PCM_24")
    diagnostics = convert_wav(
        source, tmp_path / "out.wav", AudioOptions(input_gain_db=-6.0)
    )
    assert diagnostics.source_clipped_fraction > 0.0
    assert diagnostics.output_clipped_samples == 0


def test_a_different_gain_is_a_different_dataset(tmp_path: Path) -> None:
    """dataset.json is what checkpoints and exports hash; the gain must be in it."""
    manifest = _corpus(tmp_path / "source", [_quiet(f) for f in (200.0, 250.0, 300.0)])
    _prepare(manifest, tmp_path / "zero")
    _prepare(manifest, tmp_path / "lowered", gain=-1.0)
    zero = (tmp_path / "zero" / "dataset.json").read_bytes()
    lowered = (tmp_path / "lowered" / "dataset.json").read_bytes()
    assert zero != lowered
    assert "audio_processing" not in json.loads(zero)
    assert json.loads(lowered)["audio_processing"] == {"input_gain_db": -1.0, "peak_limit": 1.0}


# dataset.json as prepared before the gain option existed, in order.
PRE_GAIN_DATASET_KEYS = [
    "format", "language", "sample_rate", "channels", "speaker", "frontend",
    "source_manifest_sha256", "split_seed", "validation_fraction", "split",
    "row_counts", "diagnostics",
]
PRE_GAIN_DIAGNOSTICS_KEYS = [
    "total_duration_seconds", "min_duration_seconds", "max_duration_seconds",
    "resampled_files", "downmixed_files", "source_clipped_files",
    "output_clipped_files", "base_symbol_coverage_fraction", "base_unknown_symbols",
    "added_symbol_count",
]


def test_the_default_gain_adds_nothing_to_dataset_json(tmp_path: Path) -> None:
    """An existing dataset prepared again at gain 0 keeps its run identity.

    The pre-clip peak is a float that would move with any resampler change, so
    it is reported beside the dataset, not hashed into it.
    """
    manifest = _corpus(tmp_path / "source", [_quiet(f) for f in (200.0, 250.0, 300.0)])
    _prepare(manifest, tmp_path / "zero")
    _prepare(manifest, tmp_path / "negative-zero", gain=-0.0)
    zero = json.loads((tmp_path / "zero" / "dataset.json").read_text(encoding="utf-8"))
    assert list(zero) == PRE_GAIN_DATASET_KEYS
    assert list(zero["diagnostics"]) == PRE_GAIN_DIAGNOSTICS_KEYS
    for name in ("dataset.json", "train.jsonl", "validation.jsonl", "symbols.json"):
        assert (tmp_path / "zero" / name).read_bytes() == (
            tmp_path / "negative-zero" / name
        ).read_bytes()

    conversion = _conversion(tmp_path / "negative-zero")
    assert conversion["input_gain_db"] == 0.0
    assert math.copysign(1.0, conversion["input_gain_db"]) == 1.0
    assert conversion["max_pre_clip_peak"] == pytest.approx(0.3, rel=1e-2)
    summary = (tmp_path / "zero" / "PREPARATION_REPORT.txt").read_text(encoding="utf-8")
    assert "Input gain: +0.00 dB" in summary


def test_a_peak_just_over_full_scale_is_refused(tmp_path: Path) -> None:
    """No tolerance above the peak limit: +0.04 dB over is still a clip.

    The rows are already 24 kHz, so no resampler overshoot blurs the boundary;
    the gain alone places the peak at 0.995 or 1.005.
    """
    signal = np.zeros(6_000)
    signal[3_000] = 0.5  # exact in PCM_24
    signals = [np.roll(signal, shift) for shift in (0, 500, 1_000)]
    manifest = _corpus(tmp_path / "source", signals, rate=24_000)

    under = 20.0 * math.log10(0.995 / 0.5)
    _prepare(manifest, tmp_path / "under", gain=under)
    assert _conversion(tmp_path / "under")["max_pre_clip_peak"] == pytest.approx(0.995)

    over = 20.0 * math.log10(1.005 / 0.5)
    with pytest.raises(OutputClippingError) as caught:
        _prepare(manifest, tmp_path / "over", gain=over)
    report = json.loads(caught.value.report_path.read_text(encoding="utf-8"))
    assert report["output_clipped_files"] == 3
    assert report["output_clipped_samples"] == 3
    assert report["max_pre_clip_peak"] == pytest.approx(1.005)
    assert caught.value.recommended_gain_db == recommended_input_gain_db(over, 1.005)
    _prepare(manifest, tmp_path / "retry", gain=caught.value.recommended_gain_db)


def test_success_removes_the_earlier_refusal_for_the_same_output(tmp_path: Path) -> None:
    manifest = _corpus(tmp_path / "source", [_hot(seed) for seed in (1, 2, 3)])
    output = tmp_path / "prepared"
    with pytest.raises(OutputClippingError) as caught:
        _prepare(manifest, output)
    report_path = caught.value.report_path
    assert report_path == tmp_path / "prepared.clipping-report.json"
    assert report_path.is_file()

    _prepare(manifest, output, gain=caught.value.recommended_gain_db)
    assert (output / "dataset.json").is_file()
    assert not report_path.exists(), "a refusal must not outlive the dataset that replaced it"

    # Only a report prepare wrote is removed, and nothing found at that path can
    # fail a dataset that is already in place.
    gain = caught.value.recommended_gain_db
    for name, content in (
        ("other", '{"note": "not ours"}\n'),
        ("nested", "[" * 100_000 + "]" * 100_000),  # json.loads raises RecursionError
    ):
        foreign = tmp_path / f"{name}.clipping-report.json"
        foreign.write_text(content, encoding="utf-8")
        _prepare(manifest, tmp_path / name, gain=gain)
        assert (tmp_path / name / "dataset.json").is_file()
        assert foreign.read_text(encoding="utf-8") == content


def test_audit_rejects_a_recorded_clip_and_only_warns_when_unrecorded(
    tmp_path: Path,
) -> None:
    manifest = _corpus(tmp_path / "source", [_quiet(f) for f in (200.0, 250.0, 300.0)])
    output = tmp_path / "prepared"
    _prepare(manifest, output)
    dataset_path = output / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    dataset["diagnostics"]["output_clipped_files"] = 2
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    report = audit_dataset(AuditOptions(prepared_dir=output, strict=False))
    assert not report["valid"]
    assert any("Conversion clipped 2 row(s)" in error for error in report["errors"])

    del dataset["diagnostics"]["output_clipped_files"]
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    report = audit_dataset(AuditOptions(prepared_dir=output))
    assert report["valid"]
    assert any("output_clipped_files" in warning for warning in report["warnings"])


def test_prepare_cli_accepts_an_input_gain() -> None:
    base = ["prepare", "--manifest", "m.jsonl", "--output", "out"]
    assert build_parser().parse_args(base).input_gain_db == 0.0
    assert build_parser().parse_args(base + ["--input-gain-db", "-3"]).input_gain_db == -3.0


def test_prepare_cli_fails_on_a_clip_and_passes_the_gain_through(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _corpus(tmp_path / "source", [_hot(seed) for seed in (1, 2, 3)])
    output = tmp_path / "prepared"
    base = [
        "prepare", "--manifest", str(manifest), "--output", str(output),
        "--frontend", "prephonemized", "--validation-fraction", "0.34", "--split-seed", "7",
    ]
    assert main(base) == 1
    error = capsys.readouterr().err
    assert "--input-gain-db" in error
    assert not output.exists()
    report = json.loads((tmp_path / "prepared.clipping-report.json").read_text("utf-8"))
    gain = report["recommended_input_gain_db"]
    assert f"--input-gain-db {gain:.1f}" in error

    assert main(base + ["--input-gain-db", f"{gain:.1f}"]) == 0
    capsys.readouterr()
    dataset = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert dataset["audio_processing"]["input_gain_db"] == gain
    assert dataset["diagnostics"]["output_clipped_files"] == 0
