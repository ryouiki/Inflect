"""End-to-end coverage of the real `train_adaptation` step body.

Every other training test in this suite inspects one helper in isolation.
Nothing before this file ran the loop itself, so the step body, the stage
transitions, the metrics row and the checkpoint payload — all four of which the
frame-grid remedy rewrote — were only ever verified by hand.

Each test here starts from the stub release in `tests/_stub_runtime.py` and a
real prepared dataset, runs a handful of CPU steps, and reads the artifacts the
run leaves behind: `metrics.jsonl`, the checkpoint payloads, the validation
sidecars and `training-summary.json`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from _stub_runtime import (
    BASE_SYMBOLS,
    HOP_LENGTH,
    MIN_WAVEFORM_SECONDS,
    SAMPLING_RATE,
    build_stub_release,
)
from test_data_frontend_prepare import _prepare, _write_manifest

from inflect_finetune import training as training_module
from inflect_finetune.checkpoint import (
    cpu_compatibility_report,
    load_posterior_sidecar,
    resume_training_checkpoint,
    save_posterior_sidecar,
)
from inflect_finetune.modeling import build_training_models, optimizer_parameters
from inflect_finetune.training import (
    FROZEN_UPSAMPLER_PREFIXES,
    STAGE_ADAPT,
    STAGE_DECODER,
    STAGE_POSTERIOR,
    TrainingOptions,
    train_adaptation,
)

SEED = 1234
#: Long enough that the spectrogram has more frames than the row has tokens.
#: `tests/test_data_frontend_prepare._write_wav` writes a fixed 0.1 s, which is
#: nine frames against the thirteen tokens its phoneme strings expand to, and
#: `tests/_stub_runtime` asks for the other ordering so the duration term does
#: not dominate every loss in this file.
CLIP_SECONDS = 0.3
#: Two characters become five tokens once the blank symbol is interspersed.
PHONEMES = "ab"
SOURCE_ROWS = 4
SPEAKER = "stub-voice"

#: The exact column set of one `metrics.jsonl` row. Pinned so that adding or
#: renaming a column is a deliberate act with a test to update, not a silent
#: change to the file every run is read back from.
METRIC_KEYS = frozenset(
    {
        "step",
        "epoch",
        "stage",
        "loss_g",
        "loss_d",
        "loss_mel",
        "loss_duration",
        "loss_kl",
        "loss_generator",
        "loss_feature",
        "loss_stft",
        "loss_proximal",
        "adversarial_weight",
        "decoder_lr_scale",
        "lr",
        "z_dc_rms",
        "z_rms",
    }
)

#: The nine settings the remedy added. A checkpoint written before they existed
#: carries a run identity whose `options` map lacks exactly these.
NEW_OPTION_FIELDS = (
    "adversarial_gating",
    "adversarial_ramp_steps",
    "decoder_lr_warmup_steps",
    "decoder_polish_mode",
    "stft_loss_weight",
    "decoder_proximal_weight",
    "decoder_freeze_upsamplers",
    "posterior_init",
    "generator_ema_decay",
)


@dataclass(frozen=True)
class Corpus:
    """A stub release plus a prepared dataset the loop can train on."""

    base: Path
    prepared: Path
    symbols: tuple[str, ...]


def write_speech_wav(path: Path, frequency: float, seconds: float = CLIP_SECONDS) -> None:
    """Mirror `_write_wav` from the prepare tests, with a duration."""

    axis = np.arange(int(seconds * SAMPLING_RATE), dtype=np.float32) / SAMPLING_RATE
    waveform = 0.1 * np.sin(2.0 * np.pi * frequency * axis)
    sf.write(path, waveform, SAMPLING_RATE, subtype="PCM_16")


def build_corpus(root: Path) -> Corpus:
    """Write a stub release and a real prepared dataset under `root`."""

    assert CLIP_SECONDS > MIN_WAVEFORM_SECONDS, "clips must survive rand_slice_segments"
    source = root / "source"
    source.mkdir(parents=True)
    rows = []
    for index in range(SOURCE_ROWS):
        name = f"audio-{index}.wav"
        write_speech_wav(source / name, 180.0 + index * 37.0)
        rows.append(
            {
                "audio": name,
                "text": f"Sentence {index}.",
                "phonemes": PHONEMES,
                "speaker": SPEAKER,
            }
        )
    prepared = root / "prepared"
    metadata = _prepare(_write_manifest(source, rows), prepared)
    # Read back rather than assumed: the prepare defaults these tests borrow own
    # the split, and a run needs both splits nonempty.
    assert metadata["row_counts"] == {"total": 4, "train": 3, "validation": 1}
    inventory = json.loads((prepared / "symbols.json").read_text(encoding="utf-8"))
    return Corpus(
        base=build_stub_release(root / "release", BASE_SYMBOLS),
        prepared=prepared,
        symbols=tuple(inventory["symbols"]),
    )


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    """One release and dataset for the whole module; every run reads them only."""

    return build_corpus(tmp_path_factory.mktemp("corpus"))


def make_options(corpus: Corpus, output_dir: Path, **overrides: object) -> TrainingOptions:
    """Training options for a deterministic, CPU-only, few-step run.

    The intervals are larger than any `max_steps` here, so validation renders
    and intermediate checkpoints happen only in the tests that lower them.
    """

    values: dict[str, object] = {
        "base_model": corpus.base,
        "prepared_dir": corpus.prepared,
        "output_dir": output_dir,
        "device": "cpu",
        "amp": False,
        "seed": SEED,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_workers": 0,
        "max_steps": 2,
        "validation_interval": 1_000,
        "checkpoint_interval": 1_000,
        "log_interval": 1_000,
    }
    values.update(overrides)
    return TrainingOptions(**values)


def metric_rows(output_dir: Path) -> list[dict]:
    text = (output_dir / "metrics.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def load_payload(path: Path) -> dict:
    # The training payload carries RNG state, so it is not a weights-only load.
    return torch.load(path, map_location="cpu", weights_only=False)


def deployable_state(path: Path) -> dict[str, torch.Tensor]:
    return load_payload(path)["model"]


def optimizer_group(payload: dict, name: str) -> dict:
    """Find a saved generator parameter group by name rather than by position."""

    groups = [
        group for group in payload["optimizer_g"]["param_groups"] if group["name"] == name
    ]
    assert len(groups) == 1, f"expected exactly one {name!r} group"
    return groups[0]


def fresh_resume_target(corpus: Corpus, options: TrainingOptions) -> dict[str, object]:
    """Warm-start a second copy of everything `resume_training_checkpoint` mutates.

    `train_adaptation` keeps its model private, so proving a rejected resume
    touched nothing means making the same call where the model is visible. The
    keys are the resume function's own keyword names.
    """

    bundle = build_training_models(corpus.base, corpus.symbols, seed=options.seed)
    cpu_compatibility_report(
        bundle.generator,
        corpus.base / "model.pth",
        bundle.base_symbols,
        corpus.symbols,
        initialization_seed=options.seed,
    )
    optimizer_g = torch.optim.AdamW(
        training_module._generator_groups(bundle.generator, options),
        lr=options.learning_rate_g,
    )
    optimizer_d = torch.optim.AdamW(
        optimizer_parameters(bundle.discriminator), lr=options.learning_rate_d
    )
    return {
        "generator": bundle.generator,
        "discriminator": bundle.discriminator,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d,
        "scheduler_g": torch.optim.lr_scheduler.ExponentialLR(
            optimizer_g, gamma=options.lr_decay
        ),
        "scheduler_d": torch.optim.lr_scheduler.ExponentialLR(
            optimizer_d, gamma=options.lr_decay
        ),
        "scaler": training_module._grad_scaler(False),
    }


def test_a_default_two_step_run_logs_the_legacy_and_the_new_metric_columns(
    corpus: Corpus, tmp_path: Path
) -> None:
    """The row a default run writes, column for column.

    Nothing but the harness settings is overridden here, so both steps sit in
    the posterior warm-up the shipped defaults start with, the adversarial term
    is at full weight, and the two opt-in loss terms are absent rather than zero.
    """

    train_adaptation(make_options(corpus, tmp_path / "run"))
    rows = metric_rows(tmp_path / "run")

    assert [row["step"] for row in rows] == [1, 2]
    for row in rows:
        assert set(row) == METRIC_KEYS
        assert row["stage"] == STAGE_POSTERIOR
        assert row["adversarial_weight"] == 1.0
        assert row["decoder_lr_scale"] == 1.0
        assert row["loss_stft"] is None
        assert row["loss_proximal"] is None
        # The discriminator trains from the first step of a default run.
        assert isinstance(row["loss_d"], float)
        for key in ("loss_g", "loss_mel", "loss_duration", "loss_kl"):
            assert math.isfinite(row[key])
        for key in ("loss_generator", "loss_feature"):
            assert isinstance(row[key], float)
        assert set(row["lr"]) == {"posterior", "linguistic", "decoder"}
        assert math.isfinite(row["z_dc_rms"])
        assert math.isfinite(row["z_rms"])


def test_a_default_run_writes_no_averaged_generator_state_at_all(
    corpus: Corpus, tmp_path: Path
) -> None:
    """Averaging is opt-in, so its absence must not leave a null behind."""

    summary = train_adaptation(make_options(corpus, tmp_path / "run"))
    payload = load_payload(tmp_path / "run" / "checkpoints" / "adaptation-final.pth")

    assert "generator_ema" not in payload
    assert summary["generator_ema_checkpoint"] is None
    assert not (tmp_path / "run" / "exports" / "model-ema.pth").exists()


def gated_options(corpus: Corpus, output_dir: Path, **overrides: object) -> TrainingOptions:
    """Gating with a four-step ramp opening at the unfreeze step.

    The weight logged with step N was the one used at step N-1, so the rows read
    0, 0, 0 for steps 0-2 and then 0.25, 0.5 as the ramp opens.
    """

    return make_options(
        corpus,
        output_dir,
        adversarial_gating=True,
        posterior_warmup_steps=1,
        decoder_unfreeze_step=2,
        adversarial_ramp_steps=4,
        max_steps=5,
        **overrides,
    )


def test_gating_holds_the_generator_at_zero_while_the_discriminator_keeps_training(
    corpus: Corpus, tmp_path: Path
) -> None:
    """The gated window is the discriminator's warm-up, not a pause for it."""

    train_adaptation(gated_options(corpus, tmp_path / "run"))
    rows = metric_rows(tmp_path / "run")

    weights = [row["adversarial_weight"] for row in rows]
    assert weights == [0.0, 0.0, 0.0, 0.25, 0.5]
    partway = weights[3]
    assert 0.0 < partway < 1.0
    # This is the whole point of gating: the term the generator sees is zero
    # while the decoder is frozen, and the discriminator trains anyway.
    for row in rows:
        assert isinstance(row["loss_d"], float)
        assert math.isfinite(row["loss_d"])


def test_the_adversarial_terms_are_null_exactly_when_their_weight_is_zero(
    corpus: Corpus, tmp_path: Path
) -> None:
    """A gated step measures nothing, so it logs null rather than a zero."""

    train_adaptation(gated_options(corpus, tmp_path / "run"))
    rows = metric_rows(tmp_path / "run")

    observed = [
        (row["adversarial_weight"], row["loss_generator"], row["loss_feature"])
        for row in rows
    ]
    assert [weight for weight, _, _ in observed] == [0.0, 0.0, 0.0, 0.25, 0.5]
    for weight, generator, feature in observed:
        if weight == 0.0:
            assert generator is None
            assert feature is None
        else:
            assert isinstance(generator, float)
            assert isinstance(feature, float)


def test_recon_polish_trains_only_the_decoder_with_no_discriminator_at_all(
    corpus: Corpus, tmp_path: Path
) -> None:
    """The stage that used to crash: no discriminator, generator still stepping.

    Stepping an optimizer the gradient scaler recorded no inf check for is a
    hard error, so the discriminator guard has to follow the same condition as
    the backward pass rather than merely skipping wasted work.
    """

    options = make_options(
        corpus,
        tmp_path / "run",
        posterior_warmup_steps=0,
        decoder_unfreeze_step=0,
        decoder_polish_mode="recon",
        max_steps=2,
    )
    train_adaptation(options)
    rows = metric_rows(tmp_path / "run")

    assert len(rows) == 2
    for row in rows:
        assert row["stage"] == STAGE_DECODER
        assert row["loss_d"] is None
        assert row["adversarial_weight"] == 0.0
        assert row["loss_generator"] is None
        assert row["loss_feature"] is None
        # Only the decoder group is enabled, and a disabled group's rate is
        # exactly zero rather than merely small.
        assert row["lr"]["posterior"] == 0.0
        assert row["lr"]["linguistic"] == 0.0
        assert row["lr"]["decoder"] > 0.0
    assert rows[0]["loss_g"] != rows[1]["loss_g"]

    payload = load_payload(tmp_path / "run" / "checkpoints" / "adaptation-final.pth")
    assert payload["step"] == 2
    assert payload["stage"] == STAGE_DECODER
    released = deployable_state(corpus.base / "model.pth")
    trained = payload["generator"]
    moved = {key for key in released if not torch.equal(released[key], trained[key])}
    assert moved == {key for key in released if key.startswith("dec.")}


def test_the_stft_and_proximal_terms_are_logged_and_the_anchor_starts_at_zero(
    corpus: Corpus, tmp_path: Path
) -> None:
    """Both opt-in reconstruction terms, and where the proximal anchor is taken.

    The anchor is the decoder the run began from, and the decoder cannot move
    before its stage, so the first polish step's drift is exactly zero.
    """

    options = make_options(
        corpus,
        tmp_path / "run",
        posterior_warmup_steps=1,
        decoder_unfreeze_step=2,
        max_steps=4,
        stft_loss_weight=2.0,
        decoder_proximal_weight=1.0,
    )
    train_adaptation(options)
    rows = metric_rows(tmp_path / "run")

    assert [row["stage"] for row in rows] == [
        STAGE_POSTERIOR,
        STAGE_ADAPT,
        STAGE_DECODER,
        STAGE_DECODER,
    ]
    for row in rows:
        assert isinstance(row["loss_stft"], float)
        assert math.isfinite(row["loss_stft"])
        assert row["loss_stft"] > 0.0
    # The decoder is frozen before its stage, so there is nothing to anchor.
    assert rows[0]["loss_proximal"] is None
    assert rows[1]["loss_proximal"] is None
    assert rows[2]["loss_proximal"] == 0.0
    assert rows[3]["loss_proximal"] > 0.0


def test_the_decoder_warmup_scales_the_step_without_touching_the_saved_rate(
    corpus: Corpus, tmp_path: Path
) -> None:
    """The warm-up is a per-step scale, not a change to the schedule.

    The logged and saved learning rates therefore stay the nominal decayed ones,
    identical to a run with no warm-up at all; the warm-up itself is visible in
    `decoder_lr_scale`, whose product with that rate is the effective step.
    """

    stage_at_zero: dict[str, object] = {
        "posterior_warmup_steps": 0,
        "decoder_unfreeze_step": 0,
        "max_steps": 3,
    }
    train_adaptation(
        make_options(corpus, tmp_path / "warmup", decoder_lr_warmup_steps=2, **stage_at_zero)
    )
    train_adaptation(make_options(corpus, tmp_path / "plain", **stage_at_zero))
    warmup = metric_rows(tmp_path / "warmup")
    plain = metric_rows(tmp_path / "plain")

    # min(1, (step - unfreeze) / 2) over steps 0, 1, 2.
    assert [row["decoder_lr_scale"] for row in warmup] == [0.0, 0.5, 1.0]
    assert [row["decoder_lr_scale"] for row in plain] == [1.0, 1.0, 1.0]
    assert [row["lr"] for row in warmup] == [row["lr"] for row in plain]

    payload = load_payload(tmp_path / "warmup" / "checkpoints" / "adaptation-final.pth")
    assert optimizer_group(payload, "decoder")["lr"] == warmup[-1]["lr"]["decoder"]


def test_freezing_the_upsamplers_holds_them_bit_identical_while_the_rest_moves(
    corpus: Corpus, tmp_path: Path
) -> None:
    """The frame grid enters the waveform in `dec.ups`, so it can be held alone."""

    options = make_options(
        corpus,
        tmp_path / "run",
        posterior_warmup_steps=0,
        decoder_unfreeze_step=0,
        decoder_freeze_upsamplers=True,
        max_steps=2,
    )
    train_adaptation(options)

    released = deployable_state(corpus.base / "model.pth")
    trained = load_payload(
        tmp_path / "run" / "checkpoints" / "adaptation-final.pth"
    )["generator"]
    held = sorted(key for key in released if key.startswith(FROZEN_UPSAMPLER_PREFIXES))
    rest = sorted(
        key for key in released if key.startswith("dec.") and key not in set(held)
    )
    # Both prefixes must actually match something, or the test proves nothing.
    assert {key.rsplit(".", 1)[0] for key in held} >= {"dec.ups.0", "dec.conv_pre"}
    assert rest

    for key in held:
        assert torch.equal(released[key], trained[key]), key
    for key in rest:
        assert not torch.equal(released[key], trained[key]), key


def test_resuming_mid_ramp_recomputes_the_weight_from_the_step(
    corpus: Corpus, tmp_path: Path
) -> None:
    """No ramp state is saved, so the step alone has to reproduce the schedule."""

    settings: dict[str, object] = {
        "adversarial_gating": True,
        "posterior_warmup_steps": 1,
        "decoder_unfreeze_step": 2,
        "adversarial_ramp_steps": 4,
        "max_steps": 6,
        "checkpoint_interval": 3,
    }
    output_dir = tmp_path / "run"
    train_adaptation(make_options(corpus, output_dir, **settings))
    first = metric_rows(output_dir)
    assert [(row["step"], row["adversarial_weight"]) for row in first] == [
        (1, 0.0),
        (2, 0.0),
        (3, 0.0),
        (4, 0.25),
        (5, 0.5),
        (6, 0.75),
    ]

    # Resume needs the same run identity and a checkpoint inside this run's own
    # checkpoints directory, so the second call reuses the same output dir and
    # the same options.
    train_adaptation(
        make_options(
            corpus,
            output_dir,
            resume=output_dir / "checkpoints" / "adaptation-step-00000003.pth",
            **settings,
        )
    )
    resumed = metric_rows(output_dir)[len(first) :]

    assert [(row["step"], row["adversarial_weight"]) for row in resumed] == [
        (4, 0.25),
        (5, 0.5),
        (6, 0.75),
    ]
    assert [row["stage"] for row in resumed] == [STAGE_DECODER] * 3


def test_generator_averaging_exports_a_shadow_that_resume_restores(
    corpus: Corpus, tmp_path: Path
) -> None:
    """The averaged generator is a second candidate, and it survives a resume.

    The resumed run starts at `max_steps` and therefore takes no step at all, so
    the shadow it re-exports is purely the one it loaded. A shadow reset to the
    warm-started base would export the released weights instead.
    """

    settings: dict[str, object] = {
        "generator_ema_decay": 0.9,
        "posterior_warmup_steps": 0,
        "decoder_unfreeze_step": 0,
        "max_steps": 3,
    }
    output_dir = tmp_path / "run"
    summary = train_adaptation(make_options(corpus, output_dir, **settings))

    ema_path = output_dir / "exports" / "model-ema.pth"
    assert ema_path.is_file()
    assert summary["generator_ema_checkpoint"] == str(ema_path)
    reported = json.loads((output_dir / "training-summary.json").read_text(encoding="utf-8"))
    assert reported["generator_ema_checkpoint"] == str(ema_path)

    payload = load_payload(output_dir / "checkpoints" / "adaptation-final.pth")
    assert "generator_ema" in payload
    assert set(payload["generator_ema"]) == set(payload["generator"])

    live = deployable_state(output_dir / "exports" / "model.pth")
    shadow = deployable_state(ema_path)
    released = deployable_state(corpus.base / "model.pth")
    assert set(shadow) == set(live)
    # The shadow starts at the warm-started base and trails the live weights, so
    # every tensor that moved differs from both.
    assert all(not torch.equal(live[key], shadow[key]) for key in live)
    assert all(not torch.equal(released[key], shadow[key]) for key in released)
    kept = {key: value.clone() for key, value in shadow.items()}

    train_adaptation(
        make_options(
            corpus,
            output_dir,
            resume=output_dir / "checkpoints" / "adaptation-final.pth",
            **settings,
        )
    )
    assert len(metric_rows(output_dir)) == 3, "the resumed run must take no step"
    restored = deployable_state(ema_path)
    assert all(torch.equal(kept[key], restored[key]) for key in kept)


def test_the_validation_sidecar_carries_the_grid_comb_screens(
    corpus: Corpus, tmp_path: Path
) -> None:
    """A comb that appears at the unfreeze step is cheap to see during the run.

    The stub renders nonsense, so this pins the screens' presence and shape, not
    their values.
    """

    train_adaptation(
        make_options(corpus, tmp_path / "run", max_steps=1, validation_interval=1)
    )
    sidecar = tmp_path / "run" / "validation" / "step-00000001.json"
    assert (tmp_path / "run" / "validation" / "step-00000001.wav").is_file()
    screens = json.loads(sidecar.read_text(encoding="utf-8"))

    assert {
        "frame_grid_hz",
        "grid_tone_excess_db",
        "fold_periodic_db",
        "fold_periodic_excess_db",
    } <= set(screens)
    # 24000 / 256, the decoder's upsample grid and the comb's spacing.
    assert screens["frame_grid_hz"] == SAMPLING_RATE / HOP_LENGTH == 93.75
    for key in ("fold_periodic_db", "fold_periodic_excess_db"):
        assert isinstance(screens[key], float)
        assert math.isfinite(screens[key])
    # The stub's clip is far shorter than the screen's own analysis window, so
    # an unmeasurable band ratio is null rather than a passing number.
    assert screens["grid_tone_excess_db"] is None or isinstance(
        screens["grid_tone_excess_db"], float
    )


def test_inheriting_a_posterior_without_a_sidecar_fails_before_the_run_starts(
    corpus: Corpus, tmp_path: Path
) -> None:
    """The sidecar is resolved before the output directory is even created."""

    output_dir = tmp_path / "run"
    with pytest.raises(FileNotFoundError, match="needs a posterior sidecar"):
        train_adaptation(make_options(corpus, output_dir, posterior_init="inherit"))

    assert not output_dir.exists()


def test_an_inherited_posterior_is_reported_and_loaded_bit_exactly(
    tmp_path: Path,
) -> None:
    """A sidecar beside the base replaces the freshly seeded posterior exactly.

    This test builds its own release because it writes `posterior.pth` into it.
    The donor uses a different seed from the run, so an inherited posterior is
    distinguishable from the one this run would have seeded for itself, and
    recon polish freezes the posterior so the final checkpoint still holds the
    tensors that were loaded.
    """

    corpus = build_corpus(tmp_path / "corpus")
    donor = build_training_models(corpus.base, corpus.symbols, seed=777)
    cpu_compatibility_report(
        donor.generator,
        corpus.base / "model.pth",
        donor.base_symbols,
        corpus.symbols,
        initialization_seed=777,
    )
    sidecar = save_posterior_sidecar(
        corpus.base / "posterior.pth", generator=donor.generator, iteration=11
    )
    expected = load_posterior_sidecar(sidecar)
    assert expected

    output_dir = tmp_path / "run"
    summary = train_adaptation(
        make_options(
            corpus,
            output_dir,
            posterior_init="inherit",
            posterior_warmup_steps=0,
            decoder_unfreeze_step=0,
            decoder_polish_mode="recon",
            max_steps=2,
        )
    )

    assert summary["posterior_source"] == "sidecar"
    report = json.loads((output_dir / "compatibility-report.json").read_text(encoding="utf-8"))
    assert report["posterior_source"] == "sidecar"
    assert report["posterior_tensor_count"] == len(expected)
    # The sidecar is pinned into the run identity, so swapping it is as fatal on
    # resume as swapping model.pth.
    identity = json.loads((output_dir / "run-identity.json").read_text(encoding="utf-8"))
    assert "posterior_sha256" in identity["base"]

    trained = load_payload(output_dir / "checkpoints" / "adaptation-final.pth")["generator"]
    for key, value in expected.items():
        assert torch.equal(trained[key], value), key
    # Without inheritance this run would have seeded its own posterior instead.
    seeded = build_training_models(corpus.base, corpus.symbols, seed=SEED).generator
    fresh = seeded.state_dict()
    assert any(not torch.equal(fresh[key], value) for key, value in expected.items())


def test_a_checkpoint_predating_the_new_options_is_rejected_before_the_model_moves(
    corpus: Corpus, tmp_path: Path
) -> None:
    """Adding option fields changed the run identity, so older runs cannot resume.

    Simulated by stripping the new keys out of a real checkpoint's recorded
    options. The rejection has to name `options` — that is the only signal
    telling the operator which half of the identity moved — and it has to happen
    before any live state is loaded, or a refused resume would still leave the
    models half-overwritten.
    """

    output_dir = tmp_path / "run"
    options = make_options(corpus, output_dir, max_steps=2)
    train_adaptation(options)

    checkpoint = output_dir / "checkpoints" / "adaptation-final.pth"
    payload = load_payload(checkpoint)
    recorded = payload["run_identity"]["options"]
    assert set(NEW_OPTION_FIELDS) <= set(recorded)
    for field in NEW_OPTION_FIELDS:
        del recorded[field]
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=r"identity fields differ: \['options'\]"):
        train_adaptation(make_options(corpus, output_dir, max_steps=2, resume=checkpoint))

    # The loop hides its model, so the no-mutation half is asserted against a
    # second warm-started copy driven through the same resume call.
    marker = json.loads((output_dir / "run-identity.json").read_text(encoding="utf-8"))
    target = fresh_resume_target(corpus, options)
    generator = target["generator"]
    before = {key: value.clone() for key, value in generator.state_dict().items()}
    with pytest.raises(ValueError, match=r"identity fields differ: \['options'\]"):
        resume_training_checkpoint(
            checkpoint,
            expected_symbols=corpus.symbols,
            expected_run_identity=marker,
            **target,
        )
    after = generator.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)


# A checkpoint written exactly where the schedule changes stage is the case the
# resume path used to get wrong: the group the new stage enables was saved at
# zero while it was inactive, and nothing in the loop would ever raise it. The
# decoder case is the worst one, because `decoder_polish` is terminal and there
# is no later transition to recover from.
BOUNDARY_SCHEDULE = {
    "max_steps": 4,
    "checkpoint_interval": 1,
    "validation_interval": 1_000,
    "log_interval": 1_000,
}


def rates_after_resuming_at(
    corpus: Corpus, root: Path, boundary: int, **schedule: object
) -> tuple[list[dict], list[dict]]:
    """Return (uninterrupted rows, rows produced after resuming at `boundary`).

    Both runs use the same options, so any difference is the resume path alone.
    """

    settings = {**BOUNDARY_SCHEDULE, **schedule}
    plain = root / f"plain-{boundary}"
    train_adaptation(make_options(corpus, plain, **settings))

    resumed = root / f"resumed-{boundary}"
    train_adaptation(make_options(corpus, resumed, **settings))
    written = len(metric_rows(resumed))
    checkpoint = resumed / "checkpoints" / f"adaptation-step-{boundary:08d}.pth"
    train_adaptation(make_options(corpus, resumed, resume=checkpoint, **settings))
    return metric_rows(plain), metric_rows(resumed)[written:]


def test_resuming_at_the_decoder_unfreeze_step_restores_the_decoder_rate(
    corpus: Corpus, tmp_path: Path
) -> None:
    """The terminal stage has no later transition, so a zero here is permanent.

    Before the fix the decoder trained at rate zero for the whole polish stage
    while the run reported that it had polished.
    """

    plain, resumed = rates_after_resuming_at(
        corpus, tmp_path, 2, posterior_warmup_steps=1, decoder_unfreeze_step=2
    )
    expected = {row["step"]: row["lr"]["decoder"] for row in plain}
    assert resumed
    for row in resumed:
        assert row["stage"] == STAGE_DECODER
        assert row["lr"]["decoder"] > 0.0
        assert row["lr"]["decoder"] == pytest.approx(expected[row["step"]], abs=1e-18)


def test_resuming_at_the_posterior_boundary_restores_the_linguistic_rate(
    corpus: Corpus, tmp_path: Path
) -> None:
    plain, resumed = rates_after_resuming_at(
        corpus, tmp_path, 1, posterior_warmup_steps=1, decoder_unfreeze_step=2
    )
    expected = {row["step"]: row["lr"]["linguistic"] for row in plain}
    assert resumed
    for row in resumed:
        assert row["lr"]["linguistic"] > 0.0
        assert row["lr"]["linguistic"] == pytest.approx(expected[row["step"]], abs=1e-18)


def test_resuming_where_both_boundaries_coincide_restores_both_rates(
    corpus: Corpus, tmp_path: Path
) -> None:
    """One ablation ran with the warm-up and the unfreeze on the same step.

    That configuration strands two groups at once, so it is worth its own case.
    """

    plain, resumed = rates_after_resuming_at(
        corpus, tmp_path, 1, posterior_warmup_steps=1, decoder_unfreeze_step=1
    )
    assert resumed
    for group in ("linguistic", "decoder"):
        expected = {row["step"]: row["lr"][group] for row in plain}
        for row in resumed:
            assert row["lr"][group] > 0.0
            assert row["lr"][group] == pytest.approx(expected[row["step"]], abs=1e-18)


def test_resuming_again_from_a_boundary_checkpoint_still_restores_the_rate(
    corpus: Corpus, tmp_path: Path
) -> None:
    """A resume that lands on a boundary and then saves records the entered stage.

    Resuming from such a checkpoint sees a saved stage equal to the stage the
    options derive, so comparing those two would report no boundary and skip
    the reset. Comparing the entered stage against the previous one has no such
    hole. The checkpoint is built here by relabelling a real boundary
    checkpoint, which is exactly the state that sequence produces.
    """

    settings = {**BOUNDARY_SCHEDULE, "posterior_warmup_steps": 1, "decoder_unfreeze_step": 2}
    output_dir = tmp_path / "twice"
    train_adaptation(make_options(corpus, output_dir, **settings))
    boundary = output_dir / "checkpoints" / "adaptation-step-00000002.pth"
    assert load_payload(boundary)["stage"] == STAGE_ADAPT

    payload = load_payload(boundary)
    payload["stage"] = STAGE_DECODER
    relabelled = output_dir / "checkpoints" / "boundary-relabelled.pth"
    torch.save(payload, relabelled)

    written = len(metric_rows(output_dir))
    train_adaptation(make_options(corpus, output_dir, resume=relabelled, **settings))
    again = metric_rows(output_dir)[written:]
    assert again
    assert all(row["stage"] == STAGE_DECODER for row in again)
    assert all(row["lr"]["decoder"] > 0.0 for row in again)


def test_a_mid_stage_resume_keeps_the_decayed_rate(corpus: Corpus, tmp_path: Path) -> None:
    """Off a boundary the decay is part of the run and must survive the resume.

    This is the property the two runs that were actually resumed relied on.
    """

    settings = {
        "max_steps": 6,
        "checkpoint_interval": 1,
        "validation_interval": 1_000,
        "log_interval": 1_000,
        "posterior_warmup_steps": 1,
        "decoder_unfreeze_step": 2,
    }
    output_dir = tmp_path / "mid"
    train_adaptation(make_options(corpus, output_dir, **settings))
    rows = {row["step"]: row for row in metric_rows(output_dir)}
    checkpoint = output_dir / "checkpoints" / "adaptation-step-00000004.pth"

    written = len(metric_rows(output_dir))
    train_adaptation(make_options(corpus, output_dir, resume=checkpoint, **settings))
    after = metric_rows(output_dir)[written:]
    assert after
    # A reset would put the rate back at nominal, above the decayed value.
    nominal = 8.0e-5 * 0.1
    for row in after:
        assert row["lr"]["decoder"] < nominal
        assert row["lr"]["decoder"] == pytest.approx(rows[row["step"]]["lr"]["decoder"], abs=1e-18)


def test_validation_does_not_perturb_the_training_stream(
    corpus: Corpus, tmp_path: Path
) -> None:
    """Validation seeds the global generator, which training draws from too.

    Without the fork, how often validation ran became part of the trajectory,
    so a run could not be compared against one that only validated at a
    different interval.
    """

    settings = {"max_steps": 4, "checkpoint_interval": 1_000, "log_interval": 1_000}
    often = metric_rows_from(corpus, tmp_path / "often", validation_interval=1, **settings)
    never = metric_rows_from(corpus, tmp_path / "never", validation_interval=1_000, **settings)
    assert [row["step"] for row in often] == [row["step"] for row in never]
    for left, right in zip(often, never):
        for key in ("loss_g", "loss_d", "loss_mel", "loss_kl", "loss_duration"):
            assert left[key] == right[key], f"step {left['step']} diverged on {key}"


def metric_rows_from(corpus: Corpus, output_dir: Path, **overrides: object) -> list[dict]:
    train_adaptation(make_options(corpus, output_dir, **overrides))
    return metric_rows(output_dir)
