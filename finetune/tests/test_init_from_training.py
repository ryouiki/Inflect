"""Starting a new recipe from another run's weights, and training the acoustic path alone.

`--init-from` takes the generator and discriminator weights of a training
checkpoint (the generator alone with `init_from_discriminator='fresh'`) and
nothing else, so the optimizer starts empty at step 0. The
`posterior_decoder` polish mode trains the posterior encoder and the decoder
together and holds the text side. These tests run the real loop on the stub
release: a short parent, a run started from its weights, and the ways such a
run must refuse or stop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from test_training_loop_stub import build_corpus, make_options, metric_rows

from inflect_finetune import training as training_module
from inflect_finetune.checkpoint import sha256_file
from inflect_finetune.cli import _run_train, build_parser
from inflect_finetune.modeling import build_training_models, load_symbols
from inflect_finetune.training import (
    STAGE_DECODER,
    TrainingOptions,
    _adversarial_weight,
    _discriminator_active,
    _enabled_groups,
    _public_options,
    train_adaptation,
)

PARENT_SETTINGS: dict[str, object] = {
    "posterior_warmup_steps": 1,
    "decoder_unfreeze_step": 2,
    "checkpoint_interval": 3,
}
PARENT_STEPS = 3
#: The acoustic path from step 0: without these the new mode sits behind the
#: parent's warm-up and unfreeze steps and never runs.
ACOUSTIC: dict[str, object] = {
    "posterior_warmup_steps": 0,
    "decoder_unfreeze_step": 0,
    "decoder_polish_mode": "posterior_decoder",
    "decoder_lr_warmup_steps": 0,
    "adversarial_gating": False,
    "kl_loss_weight": 0.0,
    "duration_loss_weight": 0.0,
}
TEXT_SIDE = ("enc_p.", "dp.", "flow.")


@pytest.fixture(scope="module")
def parent(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("init")
    corpus = build_corpus(root / "corpus")
    run = root / "parent"
    train_adaptation(make_options(corpus, run, max_steps=PARENT_STEPS, **PARENT_SETTINGS))
    checkpoint = run / "checkpoints" / f"adaptation-step-{PARENT_STEPS:08d}.pth"
    assert checkpoint.is_file()
    return corpus, checkpoint


def _generator(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=False)["generator"]


def test_the_acoustic_mode_trains_the_posterior_and_decoder_with_the_discriminator_on() -> None:
    options = TrainingOptions(
        base_model="nano", prepared_dir=".", output_dir=".", **ACOUSTIC
    )
    assert _enabled_groups(options, STAGE_DECODER) == {"posterior", "decoder"}
    assert _discriminator_active(options, STAGE_DECODER)
    assert _adversarial_weight(options, 0, STAGE_DECODER) == 1.0


def test_a_run_starts_from_the_weights_alone_and_moves_only_the_acoustic_path(
    parent, tmp_path: Path
) -> None:
    corpus, checkpoint = parent
    run = tmp_path / "acoustic"
    train_adaptation(
        make_options(
            corpus, run, max_steps=3, checkpoint_interval=3, init_from=checkpoint, **ACOUSTIC
        )
    )
    check = json.loads((run / "init-check.json").read_text(encoding="utf-8"))
    assert check["passed"] is True
    assert check["items"]["generator"]["ok"] and check["items"]["discriminator"]["ok"]
    assert check["items"]["discriminator"]["source"] == "init_from"
    assert check["items"]["active_groups"]["trainable"] == ["decoder", "posterior"]
    assert check["items"]["active_groups"]["stage"] == STAGE_DECODER
    assert check["items"]["optimizer"]["entries"] == 0
    assert check["items"]["adversarial"]["discriminator_active"] is True

    identity = json.loads((run / "run-identity.json").read_text(encoding="utf-8"))
    assert identity["init"]["checkpoint_sha256"] == sha256_file(checkpoint)
    assert identity["init"]["parent_step"] == PARENT_STEPS
    assert "optimizer_g" in identity["init"]["not_inherited"]
    # The step counts from zero: nothing of the parent's position is taken.
    assert metric_rows(run)[0]["step"] == 1
    assert {row["stage"] for row in metric_rows(run)} == {STAGE_DECODER}

    before = _generator(checkpoint)
    after = _generator(run / "checkpoints" / "adaptation-step-00000003.pth")
    text_side = [name for name in before if name.startswith(TEXT_SIDE)]
    assert text_side and all(torch.equal(before[name], after[name]) for name in text_side)
    for prefix in ("enc_q.", "dec."):
        moved = [
            name
            for name in before
            if name.startswith(prefix) and not torch.equal(before[name], after[name])
        ]
        assert moved, f"no {prefix} tensor moved"


def test_a_run_started_from_weights_can_be_resumed(parent, tmp_path: Path) -> None:
    corpus, checkpoint = parent
    run = tmp_path / "resumable"
    train_adaptation(
        make_options(corpus, run, max_steps=3, checkpoint_interval=2, init_from=checkpoint, **ACOUSTIC)
    )
    rows = metric_rows(run)
    # The resume rebuilds the `init` block from the same checkpoint; a run that
    # stopped at step 2 carries on from there.
    train_adaptation(
        make_options(
            corpus,
            run,
            max_steps=3,
            checkpoint_interval=2,
            init_from=checkpoint,
            resume=run / "checkpoints" / "adaptation-step-00000002.pth",
            **ACOUSTIC,
        )
    )
    assert [row["step"] for row in metric_rows(run)][len(rows) :] == [3]


def test_a_run_whose_groups_are_not_the_declared_ones_stops_before_the_first_update(
    parent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean load says nothing about which groups will train."""

    corpus, checkpoint = parent
    real_apply = training_module._apply_stage

    def apply_everything(model, optimizer, options, stage, **kwargs):
        real_apply(model, optimizer, options, stage, **kwargs)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.requires_grad_(True)

    monkeypatch.setattr(training_module, "_apply_stage", apply_everything)
    broken = tmp_path / "broken"
    with pytest.raises(RuntimeError, match=r"stopping before the first update.*'active_groups'"):
        train_adaptation(make_options(corpus, broken, max_steps=3, init_from=checkpoint, **ACOUSTIC))
    check = json.loads((broken / "init-check.json").read_text(encoding="utf-8"))
    assert check["items"]["active_groups"]["trainable"] == ["decoder", "linguistic", "posterior"]
    assert not (broken / "metrics.jsonl").exists() or not metric_rows(broken)


def test_init_from_refuses_a_branch_and_an_inherited_posterior(parent, tmp_path: Path) -> None:
    corpus, checkpoint = parent
    with pytest.raises(ValueError, match="not both"):
        train_adaptation(
            make_options(
                corpus, tmp_path / "a", max_steps=5, init_from=checkpoint, branch_from=checkpoint
            )
        )
    with pytest.raises(ValueError, match="posterior_init='fresh'"):
        train_adaptation(
            make_options(
                corpus, tmp_path / "b", max_steps=5, init_from=checkpoint, posterior_init="inherit"
            )
        )


def _states(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone().cpu() for name, tensor in module.state_dict().items()}


def _same(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[k], right[k].cpu()) for k in left)


def _built_discriminator(corpus, seed: int) -> dict[str, torch.Tensor]:
    symbols = load_symbols(Path(corpus.prepared) / "symbols.json")
    return _states(build_training_models(corpus.base, symbols, seed=seed).discriminator)


def test_a_fresh_discriminator_starts_as_built_while_the_generator_comes_from_the_checkpoint(
    parent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, checkpoint = parent
    seen: dict[str, dict[str, torch.Tensor]] = {}
    real_check = training_module._init_check

    def capture(**kwargs):
        seen["generator"] = _states(kwargs["generator"])
        seen["discriminator"] = _states(kwargs["discriminator"])
        return real_check(**kwargs)

    monkeypatch.setattr(training_module, "_init_check", capture)
    run = tmp_path / "fresh"
    options = make_options(
        corpus, run, max_steps=2, init_from=checkpoint, init_from_discriminator="fresh", **ACOUSTIC
    )
    train_adaptation(options)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    built = _built_discriminator(corpus, options.seed)
    assert _same(seen["generator"], payload["generator"])
    assert _same(seen["discriminator"], built)
    assert not _same(seen["discriminator"], payload["discriminator"])

    check = json.loads((run / "init-check.json").read_text(encoding="utf-8"))
    assert check["passed"] is True
    assert check["items"]["discriminator"]["source"] == "fresh"
    assert check["items"]["discriminator"]["ok"] is True
    identity = json.loads((run / "run-identity.json").read_text(encoding="utf-8"))
    assert identity["init"]["inherited"] == ["generator"]
    assert identity["init"]["not_inherited"][0] == "discriminator"
    assert "optimizer_d" in identity["init"]["not_inherited"]


def test_a_fresh_discriminator_run_resumes_with_its_saved_discriminator(
    parent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume restores the discriminator it saved; it never builds a fresh one again."""

    corpus, checkpoint = parent
    run = tmp_path / "fresh-resume"
    settings: dict[str, object] = {
        "max_steps": 3,
        "checkpoint_interval": 2,
        "init_from": checkpoint,
        "init_from_discriminator": "fresh",
        **ACOUSTIC,
    }
    train_adaptation(make_options(corpus, run, **settings))
    rows = metric_rows(run)
    middle = run / "checkpoints" / "adaptation-step-00000002.pth"
    saved = torch.load(middle, map_location="cpu", weights_only=False)["discriminator"]
    assert not _same(saved, _built_discriminator(corpus, make_options(corpus, run).seed))

    loaded: dict[str, torch.Tensor] = {}
    real_resume = training_module.resume_training_checkpoint

    def capture(*args, **kwargs):
        result = real_resume(*args, **kwargs)
        loaded.update(_states(kwargs["discriminator"]))
        return result

    monkeypatch.setattr(training_module, "resume_training_checkpoint", capture)
    train_adaptation(make_options(corpus, run, resume=middle, **settings))
    assert _same(loaded, saved)
    assert [row["step"] for row in metric_rows(run)][len(rows) :] == [3]


def test_a_fresh_discriminator_needs_init_from_and_a_known_mode(parent, tmp_path: Path) -> None:
    corpus, checkpoint = parent
    with pytest.raises(ValueError, match="only applies to an init_from run"):
        train_adaptation(make_options(corpus, tmp_path / "a", init_from_discriminator="fresh"))
    with pytest.raises(ValueError, match="init_from_discriminator must be one of"):
        train_adaptation(
            make_options(
                corpus, tmp_path / "b", init_from=checkpoint, init_from_discriminator="random"
            )
        )


def test_the_init_flag_reaches_training_options_and_stays_out_of_the_options_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "inflect_finetune.training.train_adaptation",
        lambda options: captured.setdefault("options", options),
    )
    weights = tmp_path / "parent.pth"
    args = build_parser().parse_args(
        [
            "train",
            "--base",
            "nano",
            "--dataset",
            str(tmp_path / "prepared"),
            "--output",
            str(tmp_path / "run"),
            "--init-from",
            str(weights),
            "--decoder-polish-mode",
            "posterior_decoder",
        ]
    )
    _run_train(args)
    options = captured["options"]
    assert isinstance(options, TrainingOptions)
    assert Path(options.init_from) == weights
    assert options.decoder_polish_mode == "posterior_decoder"
    # A machine path: the weights are pinned by sha in the identity's own
    # `init` block instead.
    assert "init_from" not in _public_options(options)


def test_a_run_started_from_weights_can_be_branched_and_the_branch_resumed(
    parent, tmp_path: Path
) -> None:
    """The `init` lineage travels with a branch instead of blocking it."""

    corpus, checkpoint = parent
    first = tmp_path / "first"
    train_adaptation(
        make_options(corpus, first, max_steps=2, checkpoint_interval=2, init_from=checkpoint, **ACOUSTIC)
    )
    branch = tmp_path / "branch"
    first_checkpoint = first / "checkpoints" / "adaptation-step-00000002.pth"
    train_adaptation(
        make_options(
            corpus, branch, max_steps=5, checkpoint_interval=2, branch_from=first_checkpoint, **ACOUSTIC
        )
    )
    check = json.loads((branch / "branch-check.json").read_text(encoding="utf-8"))
    assert check["passed"] is True
    recorded = json.loads((first / "run-identity.json").read_text(encoding="utf-8"))
    identity = json.loads((branch / "run-identity.json").read_text(encoding="utf-8"))
    assert identity["init"] == recorded["init"]
    assert identity["branch"]["differences"] == ["options.max_steps", "run_id"]
    rows = metric_rows(branch)
    assert [row["step"] for row in rows] == [3, 4, 5]
    train_adaptation(
        make_options(
            corpus,
            branch,
            max_steps=5,
            checkpoint_interval=2,
            resume=branch / "checkpoints" / "adaptation-step-00000004.pth",
            **ACOUSTIC,
        )
    )
    assert [row["step"] for row in metric_rows(branch)][len(rows) :] == [5]


RECON_ONLY = {**ACOUSTIC, "decoder_polish_mode": "posterior_decoder_recon", "stft_loss_weight": 3.0}


def test_the_reconstruction_only_acoustic_mode_leaves_the_discriminator_alone() -> None:
    options = TrainingOptions(base_model="nano", prepared_dir=".", output_dir=".", **RECON_ONLY)
    assert _enabled_groups(options, STAGE_DECODER) == {"posterior", "decoder"}
    assert not _discriminator_active(options, STAGE_DECODER)
    assert _adversarial_weight(options, 0, STAGE_DECODER) == 0.0


def test_a_reconstruction_only_run_trains_on_mel_and_stft_alone(parent, tmp_path: Path) -> None:
    """No adversarial or feature term, no discriminator update, no discriminator schedule."""

    corpus, checkpoint = parent
    run = tmp_path / "recon-only"
    train_adaptation(
        make_options(corpus, run, max_steps=3, checkpoint_interval=3, init_from=checkpoint, **RECON_ONLY)
    )
    check = json.loads((run / "init-check.json").read_text(encoding="utf-8"))
    assert check["passed"] is True
    assert check["items"]["active_groups"]["trainable"] == ["decoder", "posterior"]
    assert check["items"]["adversarial"]["weight_at_step_0"] == 0.0
    assert check["items"]["adversarial"]["discriminator_active"] is False

    rows = metric_rows(run)
    assert rows
    for row in rows:
        assert row["adversarial_weight"] == 0.0
        assert row["loss_generator"] is None and row["loss_feature"] is None
        expected = 45.0 * row["loss_mel"] + 3.0 * row["loss_stft"]
        assert row["loss_g"] == pytest.approx(expected, rel=1e-4)

    parent_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload = torch.load(run / "checkpoints" / "adaptation-step-00000003.pth", map_location="cpu", weights_only=False)
    assert all(
        torch.equal(parent_payload["discriminator"][name], payload["discriminator"][name])
        for name in parent_payload["discriminator"]
    )
    # A fresh optimizer's schedule starts at 0 and, with no discriminator
    # update, the discriminator's stays there.
    assert payload["scheduler_d"]["last_epoch"] == 0
    assert payload["scheduler_g"]["last_epoch"] == 3
    for prefix in ("enc_q.", "dec."):
        assert any(
            not torch.equal(parent_payload["generator"][name], payload["generator"][name])
            for name in payload["generator"]
            if name.startswith(prefix)
        )
