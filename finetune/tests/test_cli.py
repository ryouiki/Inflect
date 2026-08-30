from __future__ import annotations

import json
from pathlib import Path

import pytest

from inflect_finetune.cli import (
    _optional_step,
    _resolved_frontend_hook,
    _run_train,
    build_parser,
)
from inflect_finetune.frontends import REGISTRY


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
