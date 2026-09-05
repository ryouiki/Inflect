from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from inflect_finetune.cli import (
    _non_negative_float,
    _non_negative_int,
    _optional_step,
    _resolved_frontend_hook,
    _run_evaluate,
    _run_export,
    _run_train,
    _unit_interval,
    build_parser,
)
from inflect_finetune.frontends import REGISTRY


def test_cli_exposes_complete_workflow() -> None:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if action.dest == "command"
    )
    assert set(subparsers.choices) == {
        "prepare",
        "audit",
        "train",
        "evaluate",
        "export",
    }


def test_prepare_cli_maps_documented_arguments() -> None:
    args = build_parser().parse_args(
        [
            "prepare",
            "--manifest",
            "metadata.jsonl",
            "--audio-root",
            "audio",
            "--language",
            "fr-fr",
            "--output",
            "prepared/fr",
        ]
    )
    assert args.command == "prepare"
    assert args.manifest == Path("metadata.jsonl")
    assert args.audio_root == Path("audio")
    assert args.language == "fr-fr"
    assert args.output == Path("prepared/fr")


def test_prepare_cli_exposes_custom_frontend_hook() -> None:
    args = build_parser().parse_args(
        [
            "prepare",
            "--manifest",
            "metadata.jsonl",
            "--frontend",
            "custom",
            "--frontend-hook",
            "frontend.py:create_frontend",
            "--output",
            "prepared/custom",
        ]
    )
    assert args.frontend == "custom"
    assert args.frontend_hook == "frontend.py:create_frontend"


@pytest.mark.parametrize(("value", "expected"), [("none", None), ("off", None), ("0", 0)])
def test_optional_decoder_step(value: str, expected: int | None) -> None:
    assert _optional_step(value) == expected


def test_optional_decoder_step_rejects_negative_value() -> None:
    with pytest.raises(Exception, match="non-negative"):
        _optional_step("-1")


def test_train_cli_explicit_value_overrides_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_train(options: object) -> dict[str, object]:
        captured["options"] = options
        return {"ok": True}

    monkeypatch.setattr("inflect_finetune.training.train_adaptation", fake_train)
    args = build_parser().parse_args(
        [
            "train",
            "--base",
            "nano",
            "--dataset",
            str(tmp_path / "prepared"),
            "--output",
            str(tmp_path / "run"),
            "--preset",
            "balanced",
            "--batch-size",
            "2",
            "--no-amp",
        ]
    )

    assert _run_train(args) == {"ok": True}
    options = captured["options"]
    assert options.batch_size == 2
    assert options.gradient_accumulation_steps == 2
    assert options.amp is False


def test_train_cli_can_disable_decoder_unfreeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_train(options: object) -> dict[str, object]:
        captured["options"] = options
        return {"ok": True}

    monkeypatch.setattr("inflect_finetune.training.train_adaptation", fake_train)
    args = build_parser().parse_args(
        [
            "train",
            "--base",
            "micro",
            "--dataset",
            str(tmp_path / "prepared"),
            "--output",
            str(tmp_path / "run"),
            "--decoder-unfreeze-step",
            "none",
        ]
    )

    _run_train(args)
    assert captured["options"].decoder_unfreeze_step is None


def test_export_cli_exposes_language_frontend_inputs() -> None:
    args = build_parser().parse_args(
        [
            "export",
            "--checkpoint",
            "run/checkpoints/adaptation-final.pth",
            "--prepared-dataset",
            "prepared/es",
            "--frontend-hook",
            "frontend.py",
            "--output",
            "exports/es",
        ]
    )
    assert args.prepared_dataset == Path("prepared/es")
    assert args.frontend_hook == Path("frontend.py")


def _prepared_registry_dataset(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "format": "inflect_prepared_dataset_v1",
                "language": "ja",
                "frontend": {
                    "type": "custom",
                    "language": "ja",
                    "registry": {"name": "ja-openjtalk"},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "symbols.json").write_text(json.dumps({"symbols": ["_"]}), encoding="utf-8")
    return root


def _export_args(*extra: str):
    return build_parser().parse_args(
        ["export", "--checkpoint", "model.pth", "--output", "exports/ja", *extra]
    )


def test_prepare_cli_offers_the_bundled_language_frontends() -> None:
    args = build_parser().parse_args(
        [
            "prepare",
            "--manifest",
            "metadata.jsonl",
            "--language",
            "ja",
            "--frontend",
            "ja-openjtalk",
            "--output",
            "prepared/ja",
        ]
    )
    assert args.frontend == "ja-openjtalk"
    assert args.frontend_hook is None


def test_audit_cli_exposes_the_no_new_symbols_gate() -> None:
    assert not build_parser().parse_args(
        ["audit", "--dataset", "prepared/ja"]
    ).require_no_new_symbols
    assert build_parser().parse_args(
        ["audit", "--dataset", "prepared/ja", "--require-no-new-symbols"]
    ).require_no_new_symbols


def test_export_recovers_the_hook_of_a_bundled_frontend(tmp_path: Path) -> None:
    """Export needs the exact hook source; a bundled one ships with the toolkit."""
    prepared = _prepared_registry_dataset(tmp_path / "prepared")
    expected = REGISTRY["ja-openjtalk"].module_file

    assert _resolved_frontend_hook(_export_args("--prepared-dataset", str(prepared))) == expected
    assert (
        _resolved_frontend_hook(
            _export_args("--prepared-dataset", str(prepared / "dataset.json"))
        )
        == expected
    )
    assert (
        _resolved_frontend_hook(_export_args("--symbols", str(prepared / "symbols.json")))
        == expected
    )


def test_export_keeps_an_explicit_hook_and_stays_silent_without_one(tmp_path: Path) -> None:
    prepared = _prepared_registry_dataset(tmp_path / "prepared")
    explicit = _export_args(
        "--prepared-dataset", str(prepared), "--frontend-hook", "mine.py"
    )
    assert _resolved_frontend_hook(explicit) == Path("mine.py")
    assert _resolved_frontend_hook(_export_args()) is None
    assert (
        _resolved_frontend_hook(_export_args("--prepared-dataset", str(tmp_path / "absent")))
        is None
    )


def _captured_train_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str
) -> tuple[object, argparse.Namespace]:
    """Parse a train command line and return the TrainingOptions it produced."""

    captured: dict[str, object] = {}

    def fake_train(options: object) -> dict[str, object]:
        captured["options"] = options
        return {"ok": True}

    monkeypatch.setattr("inflect_finetune.training.train_adaptation", fake_train)
    args = build_parser().parse_args(
        [
            "train",
            "--base",
            "nano",
            "--dataset",
            str(tmp_path / "prepared"),
            "--output",
            str(tmp_path / "run"),
            *extra,
        ]
    )
    assert _run_train(args) == {"ok": True}
    return captured["options"], args


# (flag, value or None for a switch, TrainingOptions field, expected value)
_NEW_TRAIN_FLAGS = (
    ("--feature-loss-weight", "0.5", "feature_loss_weight", 0.5),
    ("--adversarial-gating", None, "adversarial_gating", True),
    ("--no-adversarial-gating", None, "adversarial_gating", False),
    ("--adversarial-ramp-steps", "250", "adversarial_ramp_steps", 250),
    ("--decoder-lr-warmup-steps", "300", "decoder_lr_warmup_steps", 300),
    ("--decoder-polish-mode", "recon", "decoder_polish_mode", "recon"),
    ("--stft-loss-weight", "2.5", "stft_loss_weight", 2.5),
    ("--decoder-proximal-weight", "0.25", "decoder_proximal_weight", 0.25),
    ("--decoder-freeze-upsamplers", None, "decoder_freeze_upsamplers", True),
    ("--no-decoder-freeze-upsamplers", None, "decoder_freeze_upsamplers", False),
    ("--posterior-init", "inherit", "posterior_init", "inherit"),
    ("--generator-ema-decay", "0.999", "generator_ema_decay", 0.999),
)

_NEW_TRAIN_FIELDS = frozenset(field for _, _, field, _ in _NEW_TRAIN_FLAGS)


@pytest.mark.parametrize(("flag", "value", "field", "expected"), _NEW_TRAIN_FLAGS)
def test_every_new_train_flag_reaches_training_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str | None,
    field: str,
    expected: object,
) -> None:
    """Declaring a flag on the parser is only half of wiring it up.

    `_run_train` copies flags into TrainingOptions through a hand-maintained
    `override_names` tuple listed separately from the parser entries. A flag
    missing from that tuple still parses cleanly and is then silently dropped,
    which is exactly what this test catches: every value here differs from the
    field's default, so a dropped name fails on the options assertion while the
    namespace assertion still passes.

    The two `--no-` forms land on their default value, so for those the
    namespace assertion is what proves the parser accepts the negated form.
    """

    extra = [flag] if value is None else [flag, value]
    options, args = _captured_train_options(monkeypatch, tmp_path, *extra)

    assert getattr(args, field) == expected
    assert getattr(options, field) == expected


def test_omitting_the_new_train_flags_leaves_every_field_at_its_dataclass_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remedy is opt-in: an unchanged command line must build unchanged options."""

    options, args = _captured_train_options(monkeypatch, tmp_path)

    # Every new flag defaults to argparse.SUPPRESS, so an omitted one is absent
    # from the namespace and is never handed to TrainingOptions at all.
    assert not _NEW_TRAIN_FIELDS & set(vars(args))
    assert options.feature_loss_weight == 1.0
    assert options.adversarial_gating is False
    assert options.adversarial_ramp_steps == 1_000
    assert options.decoder_lr_warmup_steps == 0
    assert options.decoder_polish_mode == "adversarial"
    assert options.stft_loss_weight == 0.0
    assert options.decoder_proximal_weight == 0.0
    assert options.decoder_freeze_upsamplers is False
    assert options.posterior_init == "fresh"
    assert options.generator_ema_decay == 0.0


@pytest.mark.parametrize("coerce", [_non_negative_int, _non_negative_float, _unit_interval])
def test_no_new_coercer_accepts_a_negative_value(coerce) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="value must be"):
        coerce("-1")


def test_every_new_coercer_accepts_its_zero_boundary() -> None:
    """Zero is the documented default of each setting these coerce."""

    assert _non_negative_int("0") == 0
    assert _non_negative_float("0") == 0.0
    assert _unit_interval("0") == 0.0


def test_the_unit_interval_excludes_one_because_a_decay_of_one_never_updates() -> None:
    assert _unit_interval("0.999") == 0.999
    with pytest.raises(argparse.ArgumentTypeError, match=r"value must be in \[0, 1\)"):
        _unit_interval("1.0")


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--decoder-polish-mode", "reconstruction"), ("--posterior-init", "warm")],
)
def test_an_unknown_enumerated_value_is_rejected_at_the_parser(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["train", "--base", "nano", "--dataset", "d", "--output", "o", flag, value]
        )


def _captured_evaluate_options(
    monkeypatch: pytest.MonkeyPatch, *extra: str
) -> object:
    captured: dict[str, object] = {}

    def fake_evaluate(options: object) -> dict[str, object]:
        captured["options"] = options
        return {"ok": True}

    monkeypatch.setattr("inflect_finetune.evaluation.evaluate_checkpoint", fake_evaluate)
    args = build_parser().parse_args(
        ["evaluate", "--model-dir", "model", "--manifest", "held-out.jsonl", "--output", "eval"]
        + list(extra)
    )
    assert _run_evaluate(args) == {"ok": True}
    return captured["options"]


def test_evaluate_cli_passes_the_comb_screen_controls_to_evaluation_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _captured_evaluate_options(
        monkeypatch, "--hop-length", "512", "--steady-tone-screen"
    )

    assert options.hop_length == 512
    assert options.steady_tone_screen is True


def test_evaluate_can_turn_the_steady_tone_screen_back_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negated form has to reach the options, or the screen cannot be skipped."""

    options = _captured_evaluate_options(monkeypatch, "--no-steady-tone-screen")

    assert options.steady_tone_screen is False


def test_evaluate_leaves_the_hop_to_the_model_config_and_screens_tones_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hop is read from the model config when unset; the tone screen is on."""

    options = _captured_evaluate_options(monkeypatch)

    assert options.hop_length is None
    assert options.steady_tone_screen is True


def _captured_export_options(monkeypatch: pytest.MonkeyPatch, *extra: str) -> object:
    captured: dict[str, object] = {}

    def fake_export(options: object) -> dict[str, object]:
        captured["options"] = options
        return {"ok": True}

    monkeypatch.setattr("inflect_finetune.exporting.export_checkpoint", fake_export)
    assert _run_export(_export_args(*extra)) == {"ok": True}
    return captured["options"]


def test_export_cli_can_select_the_averaged_generator_and_the_posterior_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _captured_export_options(
        monkeypatch, "--generator-state", "ema", "--include-posterior"
    )

    assert options.generator_state == "ema"
    assert options.include_posterior is True


def test_export_exports_the_live_generator_and_no_sidecar_unless_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _captured_export_options(monkeypatch)

    assert options.generator_state == "live"
    assert options.include_posterior is False


def test_export_rejects_a_generator_state_that_is_neither_live_nor_averaged() -> None:
    with pytest.raises(SystemExit):
        _export_args("--generator-state", "best")
