"""Continuing one run's training checkpoint in a new run directory.

`--resume` is bound to its own run, so a finished run cannot be given more
steps, and chaining through an export throws away the optimizers, schedulers,
scaler and step. A branch keeps all of that. These tests run the real loop on
the stub release: a short parent, a branch that continues it, and the ways a
branch must refuse or stop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from test_checkpoint_identity import _inputs
from test_training_loop_stub import (
    SEED,
    build_corpus,
    make_options,
    metric_rows,
)

from inflect_finetune import training as training_module
from inflect_finetune.checkpoint import (
    build_run_identity,
    sha256_file,
    validate_branch_identity,
)
from inflect_finetune.cli import _run_train, build_parser
from inflect_finetune.training import (
    STAGE_DECODER,
    TrainingOptions,
    _public_options,
    train_adaptation,
)

#: The parent stops inside the decoder stage, as J1 did at 10000: no stage
#: boundary sits at the branch point or after it.
SETTINGS: dict[str, object] = {
    "posterior_warmup_steps": 1,
    "decoder_unfreeze_step": 2,
    "checkpoint_interval": 3,
}
PARENT_STEPS = 3
BRANCH_STEPS = 7


@pytest.fixture(scope="module")
def parent(tmp_path_factory: pytest.TempPathFactory):
    """One corpus and one finished parent run, read by every test here."""

    root = tmp_path_factory.mktemp("branch")
    corpus = build_corpus(root / "corpus")
    run = root / "parent"
    train_adaptation(make_options(corpus, run, max_steps=PARENT_STEPS, **SETTINGS))
    checkpoint = run / "checkpoints" / f"adaptation-step-{PARENT_STEPS:08d}.pth"
    assert checkpoint.is_file()
    return corpus, run, checkpoint


def _files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_a_branch_may_differ_from_its_parent_only_in_the_budget(tmp_path: Path) -> None:
    identities = []
    for name, run_id, options in (
        ("parent", "p", {"max_steps": 10, "seed": 7, "stft_loss_weight": 3.0}),
        ("child", "c", {"max_steps": 20, "seed": 7, "stft_loss_weight": 3.0}),
        ("changed", "c", {"max_steps": 20, "seed": 8, "stft_loss_weight": 3.0}),
    ):
        base, prepared = _inputs(tmp_path / name)
        identities.append(
            build_run_identity(
                run_id=run_id,
                base_root=base,
                prepared_dir=prepared,
                options=options,
                optimizer_schema={"generator": {"class": "torch.optim.AdamW"}},
            )
        )
    parent_identity, child, changed = identities

    assert validate_branch_identity(parent_identity, child) == ["options.max_steps", "run_id"]
    # A parent that was itself a branch carries lineage, not a setting.
    assert validate_branch_identity({**parent_identity, "branch": {"parent_step": 1}}, child) == [
        "options.max_steps",
        "run_id",
    ]
    with pytest.raises(ValueError, match=r"these differ as well: \['options\.seed'\]"):
        validate_branch_identity(parent_identity, changed)


def test_a_branch_continues_the_parent_exactly_and_can_itself_be_resumed(
    parent, tmp_path: Path
) -> None:
    corpus, parent_run, checkpoint = parent
    parent_files = _files(parent_run)
    branch = tmp_path / "branch"

    train_adaptation(
        make_options(corpus, branch, max_steps=BRANCH_STEPS, branch_from=checkpoint, **SETTINGS)
    )

    rows = metric_rows(branch)
    assert [row["step"] for row in rows] == list(range(PARENT_STEPS + 1, BRANCH_STEPS + 1))
    assert {row["stage"] for row in rows} == {STAGE_DECODER}

    check = json.loads((branch / "branch-check.json").read_text(encoding="utf-8"))
    assert check["passed"], check
    assert all(item["ok"] for item in check["items"].values())
    assert check["items"]["stage_schedule"]["branch_point_is_boundary"] is False
    assert check["items"]["stage_schedule"]["boundaries_ahead"] == []

    identity = json.loads((branch / "run-identity.json").read_text(encoding="utf-8"))
    lineage = identity["branch"]
    assert lineage["parent_step"] == PARENT_STEPS
    assert lineage["parent_checkpoint_sha256"] == sha256_file(checkpoint)
    assert lineage["differences"] == ["options.max_steps", "run_id"]
    assert lineage["not_inherited"] == ["dataloader shuffle order"]
    assert "scaler" in lineage["inherited"] and "rng_state" in lineage["inherited"]

    # The parent is read, never written.
    assert _files(parent_run) == parent_files

    # The learning rates stay on the uninterrupted run's decay curve: they are
    # a function of the step and the inherited scheduler, not of the data order.
    plain = tmp_path / "plain"
    train_adaptation(make_options(corpus, plain, max_steps=BRANCH_STEPS, **SETTINGS))
    plain_rates = {row["step"]: row["lr"] for row in metric_rows(plain)}
    for row in rows:
        assert row["lr"] == plain_rates[row["step"]]

    # A branch that stops midway has to be resumable like any run: its marker
    # carries the lineage and the resume path must carry it too.
    train_adaptation(
        make_options(
            corpus,
            branch,
            max_steps=BRANCH_STEPS,
            resume=branch / "checkpoints" / "adaptation-step-00000006.pth",
            **SETTINGS,
        )
    )
    assert [row["step"] for row in metric_rows(branch)][len(rows) :] == [BRANCH_STEPS]


def test_a_branch_that_changes_another_setting_is_refused_before_anything_is_written(
    parent, tmp_path: Path
) -> None:
    corpus, _, checkpoint = parent
    refused = tmp_path / "refused"

    with pytest.raises(ValueError, match=r"options\.seed"):
        train_adaptation(
            make_options(
                corpus,
                refused,
                max_steps=BRANCH_STEPS,
                branch_from=checkpoint,
                seed=SEED + 1,
                **SETTINGS,
            )
        )
    assert not refused.exists()


def test_a_branch_has_to_go_past_its_parent(parent, tmp_path: Path) -> None:
    corpus, _, checkpoint = parent
    with pytest.raises(ValueError, match="go past its parent"):
        train_adaptation(
            make_options(
                corpus, tmp_path / "short", max_steps=PARENT_STEPS, branch_from=checkpoint, **SETTINGS
            )
        )
    assert not (tmp_path / "short").exists()


def test_a_branch_that_did_not_land_stops_before_the_first_update(
    parent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading and then assuming is how a continuation goes wrong unnoticed."""

    corpus, _, checkpoint = parent
    real_resume = training_module.resume_training_checkpoint

    def resume_then_disturb(path, **kwargs):
        result = real_resume(path, **kwargs)
        with torch.no_grad():
            next(kwargs["generator"].parameters()).add_(1.0)
        return result

    monkeypatch.setattr(training_module, "resume_training_checkpoint", resume_then_disturb)
    broken = tmp_path / "broken"
    with pytest.raises(RuntimeError, match=r"stopping before the first update.*'generator'"):
        train_adaptation(
            make_options(corpus, broken, max_steps=BRANCH_STEPS, branch_from=checkpoint, **SETTINGS)
        )
    check = json.loads((broken / "branch-check.json").read_text(encoding="utf-8"))
    assert check["passed"] is False
    assert check["items"]["generator"]["ok"] is False
    assert not (broken / "metrics.jsonl").exists() or not metric_rows(broken)


def test_resume_and_branch_are_exclusive(parent, tmp_path: Path) -> None:
    corpus, _, checkpoint = parent
    with pytest.raises(ValueError, match="not both"):
        train_adaptation(
            make_options(
                corpus,
                tmp_path / "both",
                max_steps=BRANCH_STEPS,
                branch_from=checkpoint,
                resume=checkpoint,
                **SETTINGS,
            )
        )


def test_the_branch_flag_reaches_training_options_and_stays_out_of_the_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "inflect_finetune.training.train_adaptation",
        lambda options: captured.setdefault("options", options),
    )
    parent_checkpoint = tmp_path / "parent.pth"
    args = build_parser().parse_args(
        [
            "train",
            "--base",
            "nano",
            "--dataset",
            str(tmp_path / "prepared"),
            "--output",
            str(tmp_path / "run"),
            "--branch-from",
            str(parent_checkpoint),
        ]
    )
    _run_train(args)
    options = captured["options"]
    assert isinstance(options, TrainingOptions)
    assert Path(options.branch_from) == parent_checkpoint
    # A machine path, like resume: it must not enter the identity, or no run
    # could ever match the parent it branches from.
    assert "branch_from" not in _public_options(options)
