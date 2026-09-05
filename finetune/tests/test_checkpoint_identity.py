from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from inflect_finetune.checkpoint import (
    POSTERIOR_FORMAT,
    CompatibilityReport,
    build_run_identity,
    load_posterior_sidecar,
    resume_training_checkpoint,
    save_inference_checkpoint,
    save_posterior_sidecar,
    save_training_checkpoint,
    sha256_file,
    validate_run_identity,
)
from inflect_finetune.training import (
    TrainingOptions,
    _establish_run_identity,
    _public_options,
    _validate_new_output_dir,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "Inflect-Micro-v2"
    base.mkdir(parents=True)
    (base / "model.pth").write_bytes(b"public checkpoint")
    (base / "config.json").write_text('{"model":"micro"}\n', encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "dataset.json").write_text('{"language":"en-us"}\n', encoding="utf-8")
    (prepared / "train.jsonl").write_text('{"audio":"a.wav"}\n', encoding="utf-8")
    (prepared / "validation.jsonl").write_text(
        '{"audio":"b.wav"}\n', encoding="utf-8"
    )
    (prepared / "symbols.json").write_text('["a","b"]\n', encoding="utf-8")
    return base, prepared


def _identity(tmp_path: Path, *, run_id: str = "run-a") -> dict:
    base, prepared = _inputs(tmp_path)
    return build_run_identity(
        run_id=run_id,
        base_root=base,
        prepared_dir=prepared,
        options={"max_steps": 10, "seed": 7},
        optimizer_schema={"generator": {"class": "torch.optim.AdamW"}},
    )


def _training_parts():
    generator = nn.Linear(2, 2)
    discriminator = nn.Linear(2, 1)
    optimizer_g = torch.optim.AdamW(generator.parameters(), lr=1.0e-4)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1.0e-4)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.99)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.99)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return (
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        scheduler_g,
        scheduler_d,
        scaler,
    )


def _report() -> CompatibilityReport:
    return CompatibilityReport(
        source_path="model.pth",
        source_format="inflect_vits_inference_checkpoint_v1",
        source_tensor_count=1,
        source_parameter_count=1,
        copied_tensor_count=1,
        copied_parameter_count=1,
        exact_tensor_count=1,
        migrated_embedding_rows=0,
        initialized_embedding_rows=0,
        base_symbol_count=178,
        discarded_base_symbols=(),
        fresh_tensor_count=0,
        fresh_parameter_count=0,
        fresh_prefixes=("enc_q.",),
        verified_equal_after_copy=True,
    )


def test_run_identity_hashes_every_public_input_and_is_path_independent(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path)

    assert identity["toolkit_version"] == "0.1.0"
    assert identity["base"]["identity"] == "Inflect-Micro-v2"
    assert len(identity["base"]["checkpoint_sha256"]) == 64
    assert len(identity["base"]["config_sha256"]) == 64
    assert len(identity["prepared_dataset"]["dataset_json_sha256"]) == 64
    assert len(identity["prepared_dataset"]["train_jsonl_sha256"]) == 64
    assert len(identity["prepared_dataset"]["validation_jsonl_sha256"]) == 64
    assert len(identity["symbols_sha256"]) == 64
    rendered = json.dumps(identity)
    assert str(tmp_path) not in rendered


def test_identity_mismatch_is_rejected_before_mutable_state_load(tmp_path: Path) -> None:
    source_parts = _training_parts()
    checkpoint = tmp_path / "checkpoint.pth"
    latest = tmp_path / "latest.pth"
    identity = _identity(tmp_path / "identity")
    save_training_checkpoint(
        checkpoint,
        generator=source_parts[0],
        discriminator=source_parts[1],
        optimizer_g=source_parts[2],
        optimizer_d=source_parts[3],
        scheduler_g=source_parts[4],
        scheduler_d=source_parts[5],
        scaler=source_parts[6],
        step=5,
        epoch=2,
        stage="posterior_warmup",
        options={"max_steps": 10},
        symbols=("a", "b"),
        compatibility=_report(),
        run_identity=identity,
        latest_path=latest,
    )
    target_parts = _training_parts()
    with torch.no_grad():
        target_parts[0].weight.fill_(42.0)
    before = target_parts[0].weight.detach().clone()
    wrong_identity = dict(identity)
    wrong_identity["run_id"] = "another-run"

    with pytest.raises(ValueError, match="different adaptation run"):
        resume_training_checkpoint(
            checkpoint,
            generator=target_parts[0],
            discriminator=target_parts[1],
            optimizer_g=target_parts[2],
            optimizer_d=target_parts[3],
            scheduler_g=target_parts[4],
            scheduler_d=target_parts[5],
            scaler=target_parts[6],
            expected_symbols=("a", "b"),
            expected_run_identity=wrong_identity,
        )

    assert torch.equal(target_parts[0].weight, before)
    assert latest.read_bytes() == checkpoint.read_bytes()
    assert sha256_file(latest) == sha256_file(checkpoint)


def test_valid_identity_resumes_stage_and_state(tmp_path: Path) -> None:
    source_parts = _training_parts()
    identity = _identity(tmp_path / "identity")
    checkpoint = tmp_path / "checkpoint.pth"
    save_training_checkpoint(
        checkpoint,
        generator=source_parts[0],
        discriminator=source_parts[1],
        optimizer_g=source_parts[2],
        optimizer_d=source_parts[3],
        scheduler_g=source_parts[4],
        scheduler_d=source_parts[5],
        scaler=source_parts[6],
        step=5,
        epoch=2,
        stage="linguistic_adaptation",
        options={"max_steps": 10},
        symbols=("a", "b"),
        compatibility=_report(),
        run_identity=identity,
    )
    target_parts = _training_parts()

    state = resume_training_checkpoint(
        checkpoint,
        generator=target_parts[0],
        discriminator=target_parts[1],
        optimizer_g=target_parts[2],
        optimizer_d=target_parts[3],
        scheduler_g=target_parts[4],
        scheduler_d=target_parts[5],
        scaler=target_parts[6],
        expected_symbols=("a", "b"),
        expected_run_identity=identity,
    )

    assert state == (5, 2, "linguistic_adaptation")
    assert torch.equal(target_parts[0].weight, source_parts[0].weight)


def test_new_run_output_rejects_unrelated_content(tmp_path: Path) -> None:
    output = tmp_path / "run"
    (output / "checkpoints").mkdir(parents=True)
    (output / "checkpoints" / ".gitkeep").touch()
    _validate_new_output_dir(output)
    (output / "old-result.pth").write_bytes(b"do not overwrite")

    with pytest.raises(ValueError, match="nonempty output directory"):
        _validate_new_output_dir(output)


def test_resume_is_bound_to_marker_and_its_checkpoints_directory(tmp_path: Path) -> None:
    base, prepared = _inputs(tmp_path)
    output = tmp_path / "run"
    options = TrainingOptions(
        base_model=base,
        prepared_dir=prepared,
        output_dir=output,
        max_steps=10,
    )
    identity, resume = _establish_run_identity(
        options=options,
        output_dir=output,
        base_root=base,
        prepared_dir=prepared,
    )
    assert resume is None
    inside = output / "checkpoints" / "latest.pth"
    inside.parent.mkdir()
    inside.write_bytes(b"checkpoint")

    resumed, checkpoint = _establish_run_identity(
        options=TrainingOptions(
            base_model=base,
            prepared_dir=prepared,
            output_dir=output,
            resume=inside,
            max_steps=10,
        ),
        output_dir=output,
        base_root=base,
        prepared_dir=prepared,
    )

    assert resumed == identity
    assert checkpoint == inside.resolve()
    outside = tmp_path / "foreign.pth"
    outside.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="inside this run"):
        _establish_run_identity(
            options=TrainingOptions(
                base_model=base,
                prepared_dir=prepared,
                output_dir=output,
                resume=outside,
                max_steps=10,
            ),
            output_dir=output,
            base_root=base,
            prepared_dir=prepared,
        )


def test_public_options_exclude_machine_paths() -> None:
    options = TrainingOptions(
        base_model=Path("private/base"),
        prepared_dir=Path("private/prepared"),
        output_dir=Path("private/output"),
        resume=Path("private/checkpoint.pth"),
    )
    payload = _public_options(options)

    assert not {"base_model", "prepared_dir", "output_dir", "preset", "resume"} & payload.keys()
    assert payload["posterior_warmup_steps"] == 500
    assert payload["decoder_lr_multiplier"] == 0.1


class _ToyGenerator(nn.Module):
    """One decoder-shaped tensor and one posterior-shaped one.

    That is all the posterior sidecar and the inference-export filter look at:
    both of them select purely on the ``enc_q.`` key prefix.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dec = nn.Linear(2, 2)
        self.enc_q = nn.Linear(2, 2)


# The nine settings the comb remedy added to TrainingOptions, and so to the
# public options block of every run identity.
_NEW_OPTION_FIELDS = (
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

_OPTIMIZER_SCHEMA = {"generator": {"class": "torch.optim.AdamW"}}


def _payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _write_training_checkpoint(
    path: Path,
    parts: tuple,
    identity: dict,
    *,
    generator_ema: dict[str, torch.Tensor] | None = None,
    stage: str = "decoder_polish",
) -> Path:
    return save_training_checkpoint(
        path,
        generator=parts[0],
        discriminator=parts[1],
        optimizer_g=parts[2],
        optimizer_d=parts[3],
        scheduler_g=parts[4],
        scheduler_d=parts[5],
        scaler=parts[6],
        step=5,
        epoch=2,
        stage=stage,
        options={"max_steps": 10},
        symbols=("a", "b"),
        compatibility=_report(),
        run_identity=identity,
        generator_ema=generator_ema,
    )


def _resume(path: Path, parts: tuple, identity: dict, **extra) -> tuple[int, int, str]:
    return resume_training_checkpoint(
        path,
        generator=parts[0],
        discriminator=parts[1],
        optimizer_g=parts[2],
        optimizer_d=parts[3],
        scheduler_g=parts[4],
        scheduler_d=parts[5],
        scaler=parts[6],
        expected_symbols=("a", "b"),
        expected_run_identity=identity,
        **extra,
    )


def test_public_options_pin_the_comb_remedy_settings_at_the_previous_behaviour() -> None:
    """These nine values are part of every run's identity, and they must stay these.

    Each one reproduces what runs did before it existed, so a default that moves
    changes what an unchanged command line trains, silently. Pinning them here
    also records the deliberate consequence of adding them at all: the options
    block of the identity payload grew, so a checkpoint written before these
    fields existed no longer matches and cannot resume.
    """

    payload = _public_options(
        TrainingOptions(base_model="nano", prepared_dir="prepared", output_dir="run")
    )

    assert set(_NEW_OPTION_FIELDS) <= payload.keys()
    assert payload["adversarial_gating"] is False
    assert payload["adversarial_ramp_steps"] == 1_000
    assert payload["decoder_lr_warmup_steps"] == 0
    assert payload["decoder_polish_mode"] == "adversarial"
    assert payload["stft_loss_weight"] == 0.0
    assert payload["decoder_proximal_weight"] == 0.0
    assert payload["decoder_freeze_upsamplers"] is False
    assert payload["posterior_init"] == "fresh"
    assert payload["generator_ema_decay"] == 0.0


def _identity_pair(tmp_path: Path) -> tuple[dict, dict]:
    """Two identities differing only in whether options carry the new fields."""

    base, prepared = _inputs(tmp_path)
    current = _public_options(
        TrainingOptions(base_model="nano", prepared_dir="prepared", output_dir="run")
    )
    legacy = {key: value for key, value in current.items() if key not in _NEW_OPTION_FIELDS}
    legacy_identity = build_run_identity(
        run_id="run-a",
        base_root=base,
        prepared_dir=prepared,
        options=legacy,
        optimizer_schema=_OPTIMIZER_SCHEMA,
    )
    current_identity = build_run_identity(
        run_id="run-a",
        base_root=base,
        prepared_dir=prepared,
        options=current,
        optimizer_schema=_OPTIMIZER_SCHEMA,
    )
    return legacy_identity, current_identity


def test_a_checkpoint_whose_options_predate_the_new_fields_cannot_resume(
    tmp_path: Path,
) -> None:
    """The identity change is a hard stop, not a warning, and it names the block.

    Everything else about the two identities is identical: the same run id, the
    same base, the same dataset. Only the options block differs, so 'options' is
    the whole diagnosis, and the rejection happens before any live state is
    touched — the distinctive weight below survives it.
    """

    legacy_identity, current_identity = _identity_pair(tmp_path / "identity")
    checkpoint = tmp_path / "checkpoint.pth"
    _write_training_checkpoint(checkpoint, _training_parts(), legacy_identity)
    target_parts = _training_parts()
    with torch.no_grad():
        target_parts[0].weight.fill_(-13.0)
    before = target_parts[0].weight.detach().clone()

    with pytest.raises(ValueError, match=r"identity fields differ: \['options'\]"):
        _resume(checkpoint, target_parts, current_identity)

    assert torch.equal(target_parts[0].weight, before)
    assert torch.equal(
        target_parts[2].state_dict()["param_groups"][0]["lr"] * torch.ones(1),
        torch.ones(1) * 1.0e-4,
    )


def test_a_run_without_averaging_writes_the_payload_key_set_it_wrote_before(
    tmp_path: Path,
) -> None:
    """The literal key set, so adding a payload key later has to be deliberate.

    A reader of these checkpoints keys off exactly these names; ``generator_ema``
    is absent unless the run actually kept an average.
    """

    identity = _identity(tmp_path / "identity")
    checkpoint = tmp_path / "checkpoint.pth"
    _write_training_checkpoint(checkpoint, _training_parts(), identity)

    assert set(_payload(checkpoint)) == {
        "format",
        "generator",
        "discriminator",
        "optimizer_g",
        "optimizer_d",
        "scheduler_g",
        "scheduler_d",
        "scaler",
        "step",
        "epoch",
        "stage",
        "options",
        "symbols",
        "compatibility",
        "run_identity",
        "rng_state",
    }


def test_the_averaged_generator_is_saved_and_restored_into_the_live_dictionary(
    tmp_path: Path,
) -> None:
    """The average is state, so it has to survive a resume like the weights do."""

    identity = _identity(tmp_path / "identity")
    source_parts = _training_parts()
    saved_average = {
        key: value.detach().clone() + 3.0
        for key, value in source_parts[0].state_dict().items()
    }
    checkpoint = tmp_path / "checkpoint.pth"
    _write_training_checkpoint(
        checkpoint, source_parts, identity, generator_ema=saved_average
    )
    assert "generator_ema" in _payload(checkpoint)

    target_parts = _training_parts()
    live_average = {
        key: torch.zeros_like(value) for key, value in target_parts[0].state_dict().items()
    }
    state = _resume(checkpoint, target_parts, identity, generator_ema=live_average)

    assert state == (5, 2, "decoder_polish")
    for key, value in saved_average.items():
        assert torch.equal(live_average[key], value)


def test_a_resume_without_averaging_ignores_a_saved_average(tmp_path: Path) -> None:
    """Turning the option off mid-run must not strand the run's own checkpoints."""

    identity = _identity(tmp_path / "identity")
    source_parts = _training_parts()
    checkpoint = tmp_path / "checkpoint.pth"
    _write_training_checkpoint(
        checkpoint,
        source_parts,
        identity,
        generator_ema=dict(source_parts[0].state_dict()),
    )
    target_parts = _training_parts()

    state = _resume(checkpoint, target_parts, identity)

    assert state == (5, 2, "decoder_polish")
    assert torch.equal(target_parts[0].weight, source_parts[0].weight)


def test_asking_for_an_average_a_checkpoint_never_wrote_names_the_option(
    tmp_path: Path,
) -> None:
    """Silently starting the average from the live weights would corrupt it."""

    identity = _identity(tmp_path / "identity")
    checkpoint = tmp_path / "checkpoint.pth"
    _write_training_checkpoint(checkpoint, _training_parts(), identity)
    target_parts = _training_parts()
    live_average = {
        key: torch.zeros_like(value) for key, value in target_parts[0].state_dict().items()
    }

    with pytest.raises(ValueError, match="generator_ema_decay"):
        _resume(checkpoint, target_parts, identity, generator_ema=live_average)


def test_an_export_of_a_supplied_state_uses_it_and_still_drops_the_posterior(
    tmp_path: Path,
) -> None:
    """The averaged generator has to leave through the same filter as the live one."""

    torch.manual_seed(20260905)
    generator = _ToyGenerator()
    average = {
        "dec.weight": torch.full((2, 2), 7.0),
        "dec.bias": torch.full((2,), 8.0),
        "enc_q.weight": torch.full((2, 2), 9.0),
        "enc_q.bias": torch.full((2,), 9.0),
    }
    destination = tmp_path / "model.pth"

    save_inference_checkpoint(
        destination, generator=generator, iteration=3, learning_rate=1.0e-4, state=average
    )
    payload = _payload(destination)

    assert set(payload["model"]) == {"dec.weight", "dec.bias"}
    assert torch.equal(payload["model"]["dec.weight"], average["dec.weight"])
    assert torch.equal(payload["model"]["dec.bias"], average["dec.bias"])
    # 2x2 weight plus a length-2 bias; the four posterior values are excluded.
    assert payload["deployable_parameters"] == 6


def test_a_posterior_sidecar_round_trips_the_posterior_and_nothing_else(
    tmp_path: Path,
) -> None:
    """model.pth stays inference-only, so the posterior travels beside it."""

    torch.manual_seed(20260905)
    generator = _ToyGenerator()
    sidecar = tmp_path / "posterior.pth"

    save_posterior_sidecar(sidecar, generator=generator, iteration=11)
    payload = _payload(sidecar)
    restored = load_posterior_sidecar(sidecar)

    assert payload["format"] == POSTERIOR_FORMAT
    assert payload["iteration"] == 11
    assert set(restored) == {"enc_q.weight", "enc_q.bias"}
    for key, value in restored.items():
        assert torch.equal(value, generator.state_dict()[key])


def test_a_sidecar_carrying_a_tensor_outside_the_posterior_is_rejected(
    tmp_path: Path,
) -> None:
    """Anything but enc_q would be loaded over released weights on inherit."""

    foreign = tmp_path / "foreign.pth"
    torch.save(
        {
            "format": POSTERIOR_FORMAT,
            "model": {"enc_q.weight": torch.zeros(2, 2), "dec.weight": torch.zeros(2, 2)},
            "iteration": 1,
        },
        foreign,
    )

    with pytest.raises(ValueError, match="not posterior tensors"):
        load_posterior_sidecar(foreign)


def test_an_inference_checkpoint_is_not_accepted_as_a_posterior_sidecar(
    tmp_path: Path,
) -> None:
    """The two files sit side by side under similar names; the format decides."""

    mislabelled = tmp_path / "model.pth"
    torch.save(
        {
            "format": "inflect_vits_inference_checkpoint_v1",
            "model": {"enc_q.weight": torch.zeros(2, 2)},
        },
        mislabelled,
    )

    with pytest.raises(ValueError, match="not an Inflect posterior sidecar"):
        load_posterior_sidecar(mislabelled)


def test_a_module_with_no_posterior_cannot_write_a_sidecar(tmp_path: Path) -> None:
    """An empty sidecar would report a run as inheriting a posterior it never had."""

    with pytest.raises(ValueError, match="no posterior to"):
        save_posterior_sidecar(
            tmp_path / "posterior.pth", generator=nn.Linear(2, 2), iteration=1
        )


def test_a_run_without_an_inherited_posterior_records_no_posterior_hash_at_all(
    tmp_path: Path,
) -> None:
    """A null key would change the identity of every default run for nothing."""

    identity = _identity(tmp_path)

    assert "posterior_sha256" not in identity["base"]
    assert "posterior_sha256" not in json.dumps(identity)


def test_an_inherited_posterior_is_pinned_by_hash_like_the_base_checkpoint(
    tmp_path: Path,
) -> None:
    """Swapping the sidecar between resumes changes the run as much as swapping model.pth."""

    base, prepared = _inputs(tmp_path)
    sidecar = base / "posterior.pth"
    sidecar.write_bytes(b"posterior of the previous run")

    def identity_now() -> dict:
        return build_run_identity(
            run_id="run-a",
            base_root=base,
            prepared_dir=prepared,
            options={"max_steps": 10, "posterior_init": "inherit"},
            optimizer_schema=_OPTIMIZER_SCHEMA,
            posterior_path=sidecar,
        )

    recorded = identity_now()
    assert recorded["base"]["posterior_sha256"] == sha256_file(sidecar)

    sidecar.write_bytes(b"posterior of some other run")
    with pytest.raises(ValueError, match=r"identity fields differ: \['base'\]"):
        validate_run_identity(recorded, identity_now())
