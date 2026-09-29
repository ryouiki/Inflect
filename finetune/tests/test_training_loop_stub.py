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

#: The ten settings the remedy added, and the discriminator update order added
#: after them. A checkpoint written before they existed carries a run identity
#: whose `options` map lacks these.
NEW_OPTION_FIELDS = (
    "adversarial_gating",
    "warmup_adversarial_gating",
    "adversarial_ramp_steps",
    "decoder_lr_warmup_steps",
    "decoder_polish_mode",
    "stft_loss_weight",
    "decoder_proximal_weight",
    "decoder_freeze_upsamplers",
    "posterior_init",
    "generator_ema_decay",
    "discriminator_update_order",
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


def test_warmup_gating_zeroes_the_generator_terms_only_while_the_posterior_warms(
    corpus: Corpus, tmp_path: Path
) -> None:
    """Two gated warm-up steps, then adaptation at full weight, D training throughout.

    The weight logged with step N was the one used at step N-1, so a warm-up of
    two steps puts the zeros on the first two rows and full weight on the rest.
    """

    train_adaptation(
        make_options(
            corpus,
            tmp_path / "run",
            warmup_adversarial_gating=True,
            posterior_warmup_steps=2,
            decoder_unfreeze_step=None,
            max_steps=4,
        )
    )
    rows = metric_rows(tmp_path / "run")

    assert [row["adversarial_weight"] for row in rows] == [0.0, 0.0, 1.0, 1.0]
    assert [row["stage"] for row in rows] == [
        "posterior_warmup",
        "posterior_warmup",
        "linguistic_adaptation",
        "linguistic_adaptation",
    ]
    for row in rows[:2]:
        assert row["loss_generator"] is None
        assert row["loss_feature"] is None
    for row in rows[2:]:
        assert isinstance(row["loss_generator"], float)
        assert isinstance(row["loss_feature"], float)
    # The point of the option: the generator stops hearing the critic, the
    # critic keeps training.
    for row in rows:
        assert isinstance(row["loss_d"], float)
        assert math.isfinite(row["loss_d"])


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


def instrumented_events(
    corpus: Corpus, output_dir: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object
) -> list[tuple]:
    """Run the loop and record, in order, what each update and each discriminator call saw.

    ("d_call", weights) for every call of the discriminator, with a copy of all
    its parameters at that moment; ("unscale" | "step", "d" | "g", grads) for every
    scaler.unscale_ and scaler.step, grads being a copy of the discriminator's
    gradients at that moment (None for the generator); ("clip", "d" | "g",
    pre-clip norm, limit) for every gradient clip; ("update",) for every
    scaler.update. The generator's optimizer is the one whose parameter groups
    are named. The stub runs without AMP, where unscale_ does nothing, so its
    calls are recorded rather than trusted to fail.
    """

    events: list[tuple] = []
    real_build = training_module.build_training_models
    real_scaler = training_module._grad_scaler
    real_clip = torch.nn.utils.clip_grad_norm_
    discriminator_ids: set[int] = set()

    def which(optimizer) -> str:
        return "g" if "name" in optimizer.param_groups[0] else "d"

    def build(*args: object, **kwargs: object):
        bundle = real_build(*args, **kwargs)
        discriminator_ids.update(id(p) for p in bundle.discriminator.parameters())

        def record(module: torch.nn.Module, _inputs: object) -> None:
            with torch.no_grad():
                weights = torch.cat([p.detach().flatten().clone() for p in module.parameters()])
            events.append(("d_call", weights))

        bundle.discriminator.register_forward_pre_hook(record)
        return bundle

    def scaler_factory(enabled: bool):
        scaler = real_scaler(enabled)
        step, update, unscale = scaler.step, scaler.update, scaler.unscale_

        def grads(optimizer):
            if which(optimizer) != "d":
                return None
            return torch.cat([
                p.grad.detach().flatten().clone()
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ])

        def counted_unscale(optimizer):
            result = unscale(optimizer)
            events.append(("unscale", which(optimizer), grads(optimizer)))
            return result

        def counted_step(optimizer, *args, **kwargs):
            events.append(("step", which(optimizer), grads(optimizer)))
            return step(optimizer, *args, **kwargs)

        def counted_update(*args, **kwargs):
            events.append(("update",))
            return update(*args, **kwargs)

        scaler.unscale_ = counted_unscale
        scaler.step = counted_step
        scaler.update = counted_update
        return scaler

    def counted_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        kind = "d" if id(parameters[0]) in discriminator_ids else "g"
        limit = args[0] if args else kwargs["max_norm"]
        norm = real_clip(parameters, *args, **kwargs)
        # The pre-clip norm and the limit ride along; shape() keeps two fields.
        events.append(("clip", kind, float(norm), float(limit)))
        return norm

    monkeypatch.setattr(training_module, "build_training_models", build)
    monkeypatch.setattr(training_module, "_grad_scaler", scaler_factory)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", counted_clip)
    settings = {"max_steps": 3, **overrides}
    train_adaptation(make_options(corpus, output_dir, **settings))
    return events


def shape(step: list[tuple]) -> list[tuple]:
    """A step's events without the weights the discriminator calls carry."""

    return [event[:1] if event[0] in ("d_call", "update") else event[:2] for event in step]


def split_steps(events: list[tuple]) -> list[list[tuple]]:
    """One list per training step, each ending at its scaler.update."""

    steps, current = [], []
    for event in events:
        current.append(event)
        if event[0] == "update":
            steps.append(current)
            current = []
    assert not current, "events after the last update"
    return steps


def test_the_joint_order_scores_both_discriminator_calls_with_the_same_weights(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The order every earlier run used: the generator's terms see the pre-step discriminator."""

    steps = split_steps(instrumented_events(corpus, tmp_path / "run", monkeypatch))

    assert len(steps) == 3
    for step in steps:
        assert shape(step) == [
            ("d_call",), ("d_call",),
            ("unscale", "g"), ("unscale", "d"), ("clip", "g"), ("clip", "d"),
            ("step", "d"), ("step", "g"), ("update",),
        ]
        before, after = (event[1] for event in step if event[0] == "d_call")
        assert torch.equal(before, after)


def test_the_first_order_scores_the_generator_with_the_discriminator_it_just_stepped(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VITS order: one discriminator step, then the generator's terms, one update in all.

    Counting the calls matters as much as their order: a loop that stepped the
    discriminator here and again in the step block would still show new
    weights on the second call.
    """

    steps = split_steps(
        instrumented_events(
            corpus, tmp_path / "run", monkeypatch, discriminator_update_order="first"
        )
    )

    assert len(steps) == 3
    for index, step in enumerate(steps):
        assert shape(step) == [
            ("d_call",), ("unscale", "d"), ("clip", "d"), ("step", "d"),
            ("d_call",), ("unscale", "g"), ("clip", "g"), ("step", "g"), ("update",),
        ]
        before, after = (event[1] for event in step if event[0] == "d_call")
        assert not torch.equal(before, after)
        if index + 1 < len(steps):
            # Nothing touches the discriminator between the generator's terms
            # and the next step's discriminator loss.
            assert torch.equal(after, steps[index + 1][0][1])


def test_the_first_order_refuses_gradient_accumulation(corpus: Corpus, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gradient_accumulation_steps=1"):
        train_adaptation(
            make_options(
                corpus,
                tmp_path / "run",
                discriminator_update_order="first",
                gradient_accumulation_steps=2,
            )
        )
    assert not (tmp_path / "run").exists()


def test_an_unknown_discriminator_update_order_is_refused(corpus: Corpus, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="discriminator_update_order must be one of"):
        train_adaptation(
            make_options(corpus, tmp_path / "run", discriminator_update_order="after")
        )


def test_the_update_order_is_inert_when_the_discriminator_does_not_train(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reconstruction-only polish has no discriminator pass, so "first" has nothing to move."""

    recon = {"decoder_polish_mode": "recon", "posterior_warmup_steps": 0, "decoder_unfreeze_step": 0}
    first = split_steps(
        instrumented_events(
            corpus, tmp_path / "first", monkeypatch, discriminator_update_order="first", **recon
        )
    )
    assert len(first) == 3
    for step in first:
        assert shape(step) == [("unscale", "g"), ("clip", "g"), ("step", "g"), ("update",)]
    train_adaptation(make_options(corpus, tmp_path / "joint", max_steps=3, **recon))
    assert (tmp_path / "first" / "metrics.jsonl").read_bytes() == (
        tmp_path / "joint" / "metrics.jsonl"
    ).read_bytes()


def _grad_norm_lines(run: Path) -> list[dict]:
    def refuse(constant: str) -> None:
        raise AssertionError(f"grad-norms.jsonl is not strict JSON: {constant}")

    text = (run / "grad-norms.jsonl").read_text(encoding="utf-8")
    return [json.loads(line, parse_constant=refuse) for line in text.splitlines()]


def _clip_norms(step: list[tuple]) -> dict[str, float]:
    return {event[1]: event[2] for event in step if event[0] == "clip"}


@pytest.mark.parametrize("order", ["joint", "first"])
def test_grad_norms_record_what_each_clip_returned(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: str
) -> None:
    """One line per optimizer step, numbered like metrics.jsonl, holding the clips' own norms."""

    run = tmp_path / "run"
    steps = split_steps(
        instrumented_events(corpus, run, monkeypatch, discriminator_update_order=order)
    )
    lines = _grad_norm_lines(run)
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [line["step"] for line in lines] == [row["step"] for row in metrics] == [1, 2, 3]
    for line, step in zip(lines, steps):
        norms = _clip_norms(step)
        assert set(line) == {"step", "generator", "discriminator"}
        assert line["generator"] == norms["g"]
        assert line["discriminator"] == norms["d"]


def test_grad_norms_hold_null_where_no_clip_ran(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Null means the module had no clip call that step; the generator always has one."""

    run = tmp_path / "run"
    recon = {"decoder_polish_mode": "recon", "posterior_warmup_steps": 0, "decoder_unfreeze_step": 0}
    steps = split_steps(instrumented_events(corpus, run, monkeypatch, **recon))
    lines = _grad_norm_lines(run)
    assert len(lines) == len(steps) == 3
    for line, step in zip(lines, steps):
        assert line["discriminator"] is None
        assert line["generator"] == _clip_norms(step)["g"]


def test_a_non_finite_norm_is_kept_as_a_string() -> None:
    record = training_module._grad_norm_record
    assert record(None) is None
    assert record(torch.tensor(2.5)) == 2.5
    assert [record(torch.tensor(value)) for value in (float("inf"), float("-inf"), float("nan"))] == [
        "inf", "-inf", "nan",
    ]


def _steps_with(corpus: Corpus, run: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object) -> list[list[tuple]]:
    return split_steps(instrumented_events(corpus, run, monkeypatch, **overrides))


def test_each_module_clips_at_its_own_limit(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defaults clip both at max_grad_norm; a generator limit moves only the generator's clip."""

    default = _steps_with(corpus, tmp_path / "default", monkeypatch)
    raised = _steps_with(corpus, tmp_path / "raised", monkeypatch, generator_max_grad_norm=500.0)
    for steps, expected in ((default, {"g": 10.0, "d": 10.0}), (raised, {"g": 500.0, "d": 10.0})):
        for step in steps:
            assert {event[1]: event[3] for event in step if event[0] == "clip"} == expected
    assert [shape(step) for step in raised] == [shape(step) for step in default]


@pytest.mark.parametrize("order", ["joint", "first"])
def test_clipping_off_skips_the_clip_and_keeps_the_gradients(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: str
) -> None:
    """Off is no clip call at all: the discriminator steps on the gradients unscale_ left, and the
    recorded norm is theirs. A tiny limit shows the check would catch a clip that changed them."""

    run = tmp_path / "off"
    steps = _steps_with(corpus, run, monkeypatch, discriminator_update_order=order,
                        discriminator_grad_clipping="off")
    lines = _grad_norm_lines(run)
    for step, line in zip(steps, lines):
        assert [event[1] for event in step if event[0] == "clip"] == ["g"]
        (after_unscale,) = [event[2] for event in step if event[0] == "unscale" and event[1] == "d"]
        (at_step,) = [event[2] for event in step if event[0] == "step" and event[1] == "d"]
        assert torch.equal(after_unscale, at_step)
        assert line["discriminator"] == pytest.approx(float(torch.linalg.vector_norm(at_step)), rel=1e-5)

    tiny = _steps_with(corpus, tmp_path / "tiny", monkeypatch, discriminator_update_order=order,
                       discriminator_max_grad_norm=1e-6)
    changed = [
        not torch.equal(
            next(e[2] for e in step if e[0] == "unscale" and e[1] == "d"),
            next(e[2] for e in step if e[0] == "step" and e[1] == "d"),
        )
        for step in tiny
    ]
    assert all(changed)


def test_the_generator_can_skip_its_clip_too(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    steps = _steps_with(corpus, run, monkeypatch, generator_grad_clipping="off")
    for step in steps:
        assert [event[1] for event in step if event[0] == "clip"] == ["d"]
    assert all(isinstance(line["generator"], float) for line in _grad_norm_lines(run))


def test_the_total_norm_matches_clip_grad_norm_with_or_without_get_total_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch>=2.2 is allowed; get_total_norm exists only in newer releases."""

    shapes = [(4, 3), (7,), (2, 2, 5), (1,)]

    def parameters() -> list[torch.nn.Parameter]:
        made = []
        for index, shape_ in enumerate(shapes):
            parameter = torch.nn.Parameter(torch.zeros(shape_))
            parameter.grad = torch.randn(shape_, generator=torch.Generator().manual_seed(index)) * 30
            made.append(parameter)
        made.append(torch.nn.Parameter(torch.zeros(3)))  # no grad: skipped by both
        return made

    reference = float(torch.nn.utils.clip_grad_norm_(parameters(), 1e12))
    kept = parameters()
    before = [p.grad.clone() for p in kept if p.grad is not None]
    assert float(training_module._total_grad_norm(kept)) == pytest.approx(reference, rel=1e-6)
    assert all(torch.equal(a, p.grad) for a, p in zip(before, [p for p in kept if p.grad is not None]))
    if hasattr(torch.nn.utils, "get_total_norm"):
        monkeypatch.delattr(torch.nn.utils, "get_total_norm")
    assert float(training_module._total_grad_norm(parameters())) == pytest.approx(reference, rel=1e-6)
    assert float(training_module._total_grad_norm([torch.nn.Parameter(torch.zeros(2))])) == 0.0
