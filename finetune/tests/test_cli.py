from __future__ import annotations

from pathlib import Path

import pytest

from inflect_finetune.cli import _optional_step, _run_train, build_parser


def test_cli_exposes_complete_workflow() -> None:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if action.dest == "command"  # noqa: SLF001
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
    assert getattr(options, "batch_size") == 2
    assert getattr(options, "gradient_accumulation_steps") == 2
    assert getattr(options, "amp") is False


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
    assert getattr(captured["options"], "decoder_unfreeze_step") is None


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
