"""Executable acceptance test for the public export/evaluation APIs.

Run from the ``finetune`` directory:

    python examples/self_test_export_eval.py \
        --package ../release_assets/hf_clean_download/Inflect-Nano-v2

The script uses only a released package and temporary synthetic metadata. It
does not require or inspect an adaptation corpus.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from inflect_finetune.checkpoint import capture_rng_state
from inflect_finetune.evaluation import EvaluationOptions, evaluate_checkpoint
from inflect_finetune.exporting import (
    ExportOptions,
    _build_model,
    _extract_state,
    _load_checkpoint,
    export_checkpoint,
)
from inflect_finetune.frontend import FrontendOptions, custom_frontend_metadata


def _symbols(package: Path) -> list[str]:
    source = package / "runtime" / "text" / "symbols.py"
    namespace: dict[str, Any] = {}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    return list(namespace["symbols"])


def _training_fixture(package: Path, destination: Path) -> int:
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    symbols = _symbols(package)
    deployable, _, _ = _extract_state(_load_checkpoint(package / "model.pth"))
    model, _ = _build_model(
        package / "runtime",
        config,
        len(symbols),
        inference_only=False,
    )
    incompatible = model.load_state_dict(deployable, strict=False)
    missing = list(incompatible.missing_keys)
    if not missing or not all(key.startswith("enc_q.") for key in missing):
        raise AssertionError(f"Training model had invalid missing keys: {missing}")
    if incompatible.unexpected_keys:
        raise AssertionError(
            f"Training model had unexpected deployable keys: {incompatible.unexpected_keys}"
        )
    torch.save(
        {
            "format": "inflect_vits_adaptation_training_checkpoint_v1",
            "generator": model.state_dict(),
            "discriminator": {"self_test": torch.zeros(1)},
            "optimizer_g": {"state": {}, "param_groups": []},
            "optimizer_d": {"state": {}, "param_groups": []},
            "scheduler_g": {"last_epoch": 0},
            "scheduler_d": {"last_epoch": 0},
            "scaler": {"scale": 1.0},
            "step": 17,
            "epoch": 1,
            "stage": "self-test",
            "options": {"base_model": str(package)},
            "symbols": symbols,
            "compatibility": {"self_test": True},
            "rng_state": capture_rng_state(),
        },
        destination,
    )
    return len(missing)


def _prepared_fixture(
    destination: Path,
    symbols: list[str],
    *,
    language: str,
    mode: str,
    hook_metadata: dict[str, Any] | None = None,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "symbols.json").write_text(
        json.dumps(
            {
                "format": "inflect_v2_symbol_inventory_v1",
                "symbols": symbols,
                "count": len(symbols),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    frontend: dict[str, Any] = {
        "type": mode,
        "language": language,
        "preserve_punctuation": True,
        "with_stress": True,
    }
    if hook_metadata is not None:
        frontend["hook"] = hook_metadata
    (destination / "dataset.json").write_text(
        json.dumps(
            {
                "format": "inflect_prepared_dataset_v1",
                "language": language,
                "sample_rate": 24000,
                "frontend": frontend,
                "source_manifest_sha256": "0" * 64,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _onnx_available() -> bool:
    return (
        importlib.util.find_spec("onnx") is not None
        and importlib.util.find_spec("onnxruntime") is not None
    )


def run(package: Path, *, include_onnx: bool) -> dict[str, Any]:
    package = package.resolve()
    required = ("config.json", "model.pth", "runtime", "inference.py")
    missing = [name for name in required if not (package / name).exists()]
    if missing:
        raise FileNotFoundError(f"Release package is missing: {missing}")

    with tempfile.TemporaryDirectory(prefix="inflect-export-eval-self-test-") as temporary:
        root = Path(temporary)
        checkpoint = root / "training.pth"
        expected_enc_q = _training_fixture(package, checkpoint)
        symbols = _symbols(package)

        try:
            export_checkpoint(
                ExportOptions(
                    checkpoint=checkpoint,
                    output_dir=root / "must-reject-missing-frontend",
                    package_template=package,
                )
            )
        except ValueError as exc:
            if "requires prepared dataset frontend metadata" not in str(exc):
                raise
        else:
            raise AssertionError(
                "Adapted checkpoint silently reused the release English frontend."
            )

        prepared = _prepared_fixture(
            root / "prepared-es",
            symbols,
            language="es",
            mode="espeak",
        )
        export_dir = root / "export"
        export_report = export_checkpoint(
            ExportOptions(
                checkpoint=checkpoint,
                output_dir=export_dir,
                package_template=package,
                prepared_dataset=prepared,
                include_onnx=include_onnx,
                verify=True,
            )
        )
        frontend_contract = json.loads(
            (export_dir / "frontend.json").read_text(encoding="utf-8")
        )
        if (
            frontend_contract["mode"] != "espeak"
            or frontend_contract["language"] != "es"
            or not frontend_contract["accepts_prephonemized_input"]
        ):
            raise AssertionError(
                f"Exported language frontend contract is invalid: {frontend_contract}"
            )
        payload = _load_checkpoint(export_dir / "model.pth")
        deployable, _, stripped_after_reload = _extract_state(payload)
        if stripped_after_reload or any(key.startswith("enc_q.") for key in deployable):
            raise AssertionError("Inference export retained enc_q.* tensors.")

        required_omissions = {
            "discriminator",
            "optimizer_g",
            "optimizer_d",
            "scheduler_g",
            "scheduler_d",
            "scaler",
            "rng_state",
        }
        omissions = set(
            export_report["stripped_training_tensors"]["top_level_fields"]
        )
        if not required_omissions <= omissions:
            raise AssertionError(
                f"Export report did not prove all omissions: {required_omissions - omissions}"
            )
        training_check = next(
            check for check in export_report["checks"] if "enc_q" in check["message"]
        )
        if (
            not training_check["ok"]
            or len(training_check["missing_keys"]) != expected_enc_q
            or training_check["unexpected_keys"]
        ):
            raise AssertionError(f"Training-form compatibility failed: {training_check}")

        manifest = root / "evaluation.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "id": "held-out-self-test",
                    "text": "The exported checkpoint produces a complete waveform.",
                    "phonemes": "test",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        evaluation_report = evaluate_checkpoint(
            EvaluationOptions(
                model_dir=export_dir,
                manifest=manifest,
                output_dir=root / "evaluation",
                max_samples=1,
            )
        )
        if not evaluation_report["ok"]:
            raise AssertionError(f"Evaluation failed: {evaluation_report['failures']}")
        if evaluation_report["items"][0]["input_mode"] != "prephonemized":
            raise AssertionError("Evaluation did not use the explicit phoneme bypass.")

        hook = Path(__file__).with_name("custom_frontend_hook.py")
        hook_options = FrontendOptions(
            mode="custom",
            language="x-example",
            hook=f"{hook}:create_frontend",
        )
        hook_metadata = custom_frontend_metadata(hook_options)
        assert hook_metadata is not None
        custom_prepared = _prepared_fixture(
            root / "prepared-custom",
            symbols,
            language="x-example",
            mode="custom",
            hook_metadata=hook_metadata,
        )
        try:
            export_checkpoint(
                ExportOptions(
                    checkpoint=checkpoint,
                    output_dir=root / "must-reject-missing-hook",
                    package_template=package,
                    prepared_dataset=custom_prepared,
                )
            )
        except ValueError as exc:
            if "frontend_hook" not in str(exc):
                raise
        else:
            raise AssertionError("Custom export succeeded without a package hook.")

        custom_export = root / "custom-export"
        custom_report = export_checkpoint(
            ExportOptions(
                checkpoint=checkpoint,
                output_dir=custom_export,
                package_template=package,
                prepared_dataset=custom_prepared,
                frontend_hook=hook,
                verify=True,
            )
        )
        packaged_hook = custom_export / "frontend_hook.py"
        if (
            not packaged_hook.is_file()
            or packaged_hook.read_bytes() != hook.read_bytes()
        ):
            raise AssertionError("Custom frontend hook was not copied byte-for-byte.")
        custom_manifest = root / "custom-evaluation.jsonl"
        custom_manifest.write_text(
            json.dumps({"id": "custom-hook-self-test", "text": "Test"}) + "\n",
            encoding="utf-8",
        )
        custom_evaluation = evaluate_checkpoint(
            EvaluationOptions(
                model_dir=custom_export,
                manifest=custom_manifest,
                output_dir=root / "custom-evaluation",
                max_samples=1,
            )
        )
        if not custom_evaluation["ok"]:
            raise AssertionError(
                f"Packaged custom frontend failed: {custom_evaluation['failures']}"
            )

        onnx_report = export_report["onnx"]
        onnx_verification = onnx_report.get("verification", {})
        return {
            "ok": True,
            "deployable_parameters": export_report["deployable_parameters"],
            "stripped_enc_q_tensors": export_report["stripped_training_tensors"][
                "count"
            ],
            "omitted_training_fields": sorted(omissions),
            "strict_runtime_load": any(
                check["ok"] and check.get("strict_model_load")
                for check in export_report["checks"]
            ),
            "training_model_missing_only_enc_q": training_check["ok"],
            "frontend": {
                "mode": frontend_contract["mode"],
                "language": frontend_contract["language"],
                "prephonemized_evaluation": True,
                "custom_hook_packaged": custom_report["deployment_frontend"]["mode"]
                == "custom",
                "missing_metadata_rejected": True,
                "missing_custom_hook_rejected": True,
            },
            "evaluation_duration_seconds": evaluation_report["items"][0]["signal"][
                "duration_seconds"
            ],
            "onnx": {
                "requested": onnx_report.get("requested", False),
                "status": onnx_report.get("status"),
                "message": onnx_report.get("message"),
                "parity": onnx_verification.get("parity"),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--onnx",
        choices=("auto", "on", "off"),
        default="auto",
        help="Auto runs ONNX parity only when both optional dependencies are installed.",
    )
    args = parser.parse_args()
    available = _onnx_available()
    if args.onnx == "on" and not available:
        raise SystemExit("ONNX self-test requested but onnx/onnxruntime are unavailable.")
    include_onnx = available if args.onnx == "auto" else args.onnx == "on"
    print(json.dumps(run(args.package, include_onnx=include_onnx), indent=2))


if __name__ == "__main__":
    main()
