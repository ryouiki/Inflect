from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from inflect_finetune.checkpoint import (
    CompatibilityReport,
    build_run_identity,
    resume_training_checkpoint,
    save_training_checkpoint,
    sha256_file,
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
