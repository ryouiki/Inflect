"""Command-line interface for the public Inflect adaptation toolkit."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be in (0, 1)")
    return parsed


def _optional_step(value: str) -> int | None:
    if value.lower() in {"none", "never", "off"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("step must be non-negative or 'none'")
    return parsed


def _add_prepare(subparsers: Any) -> None:
    from .frontends import REGISTRY, registry_names

    bundled = registry_names()
    parser = subparsers.add_parser(
        "prepare",
        help="Validate and prepare user-owned speech data.",
        description=(
            "Convert a CSV or JSONL manifest into Inflect's versioned, 24 kHz "
            "prepared-dataset format."
        ),
    )
    parser.add_argument("--manifest", type=_path, required=True)
    parser.add_argument("--audio-root", type=_path)
    parser.add_argument("--language", default="en-us")
    parser.add_argument(
        "--frontend",
        choices=("espeak", "prephonemized", "custom") + bundled,
        default="espeak",
        help=(
            "Use eSpeak NG, manifest phonemes, an explicit custom frontend hook, "
            "or a bundled language frontend. Bundled: "
            + "; ".join(f"{name} ({REGISTRY[name].summary})" for name in bundled)
        ),
    )
    parser.add_argument(
        "--frontend-hook",
        help=(
            "Trusted Python factory in module:callable or file.py:function form. "
            "Required for --frontend custom; loading it executes that Python code."
        ),
    )
    parser.add_argument("--output", type=_path, required=True)
    parser.add_argument("--validation-fraction", type=_fraction, default=0.05)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--min-duration-seconds", type=float, default=0.05)
    parser.add_argument("--max-duration-seconds", type=float)
    parser.add_argument("--base-symbols", type=_path)
    parser.set_defaults(handler=_run_prepare)


def _add_audit(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "audit",
        help="Audit a prepared dataset before training.",
    )
    parser.add_argument("--dataset", type=_path, required=True)
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat all structural warnings selected by the auditor as fatal.",
    )
    parser.add_argument("--duration-tolerance-seconds", type=float, default=0.02)
    parser.add_argument(
        "--require-no-new-symbols",
        action="store_true",
        help=(
            "Fail when the prepared inventory extends the released symbol "
            "inventory. A checkpoint that keeps the released inventory can be "
            "reused as the base of a later adaptation run."
        ),
    )
    parser.set_defaults(handler=_run_audit)


def _add_train(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "train",
        help="Warm-start a fixed-voice, single-language checkpoint.",
        description=(
            "Run generic staged adaptation from a released Inflect generator. "
            "The public checkpoint has no posterior encoder, optimizer, or discriminator; "
            "those training-only components are initialized by this toolkit."
        ),
    )
    parser.add_argument(
        "--base",
        required=True,
        help="micro, nano, a local model directory, or a Hugging Face repo ID",
    )
    parser.add_argument("--dataset", type=_path, required=True)
    parser.add_argument("--output", type=_path, required=True)
    parser.add_argument("--preset", default="balanced")
    parser.add_argument("--resume", type=_path)
    parser.add_argument(
        "--device", default=argparse.SUPPRESS, help="auto, cpu, cuda, or cuda:N"
    )
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=_positive_int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=_positive_int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--num-workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=_positive_int, default=argparse.SUPPRESS)
    parser.add_argument("--learning-rate-g", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--learning-rate-d", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--posterior-warmup-steps", type=int, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--decoder-unfreeze-step",
        type=_optional_step,
        default=argparse.SUPPRESS,
        metavar="STEP|none",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Use automatic mixed precision on CUDA.",
    )
    parser.add_argument(
        "--checkpoint-interval", type=_positive_int, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--validation-interval", type=_positive_int, default=argparse.SUPPRESS
    )
    parser.add_argument("--log-interval", type=_positive_int, default=argparse.SUPPRESS)
    parser.set_defaults(handler=_run_train)


def _add_evaluate(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "evaluate",
        help="Synthesize held-out text and write machine-readable diagnostics.",
    )
    parser.add_argument("--model-dir", type=_path, required=True)
    parser.add_argument("--manifest", type=_path, required=True)
    parser.add_argument("--output", type=_path, required=True)
    parser.add_argument("--checkpoint", type=_path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=_positive_int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--variation", type=float, default=0.667)
    parser.add_argument(
        "--transcript-evaluator",
        help="Optional module:attribute callable that returns a transcript or metrics.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.set_defaults(handler=_run_evaluate)


def _add_export(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "export",
        help="Strip training state and create a verified inference package.",
    )
    parser.add_argument("--checkpoint", type=_path, required=True)
    parser.add_argument("--output", type=_path, required=True)
    parser.add_argument("--format", choices=("pytorch", "onnx"), default="pytorch")
    parser.add_argument("--config", type=_path)
    parser.add_argument("--symbols", type=_path)
    parser.add_argument(
        "--prepared-dataset",
        type=_path,
        help=(
            "Prepared directory or dataset.json containing the exact language "
            "frontend metadata for this checkpoint."
        ),
    )
    parser.add_argument(
        "--frontend-hook",
        type=_path,
        help=(
            "Matching trusted .py frontend source. Required only for exports "
            "prepared with --frontend custom."
        ),
    )
    parser.add_argument(
        "--package-template",
        type=_path,
        help="Released Micro/Nano directory whose public runtime should be copied.",
    )
    parser.add_argument("--onnx-opset", type=int, default=17)
    parser.add_argument("--model-name")
    parser.add_argument("--source-revision")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.set_defaults(handler=_run_export)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inflect-adapt",
        description=(
            "Prepare data, audit it, warm-start Inflect v2, evaluate held-out "
            "speech, and export inference-only packages."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable informational logging.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare(subparsers)
    _add_audit(subparsers)
    _add_train(subparsers)
    _add_evaluate(subparsers)
    _add_export(subparsers)
    return parser


def _run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    from .prepare import PrepareOptions, prepare_dataset

    return prepare_dataset(
        PrepareOptions(
            manifest_path=args.manifest,
            audio_root=args.audio_root,
            language=args.language,
            frontend=args.frontend,
            frontend_hook=args.frontend_hook,
            output_dir=args.output,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
            min_duration_seconds=args.min_duration_seconds,
            max_duration_seconds=args.max_duration_seconds,
            base_symbols_path=args.base_symbols,
        )
    )


def _run_audit(args: argparse.Namespace) -> dict[str, Any]:
    from .audit import AuditOptions, audit_dataset

    return audit_dataset(
        AuditOptions(
            prepared_dir=args.dataset,
            strict=args.strict,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
            require_no_new_symbols=args.require_no_new_symbols,
        )
    )


def _run_train(args: argparse.Namespace) -> dict[str, Any]:
    from .training import TrainingOptions, train_adaptation

    override_names = (
        "resume",
        "device",
        "seed",
        "batch_size",
        "gradient_accumulation_steps",
        "num_workers",
        "max_steps",
        "learning_rate_g",
        "learning_rate_d",
        "posterior_warmup_steps",
        "decoder_unfreeze_step",
        "amp",
        "checkpoint_interval",
        "validation_interval",
        "log_interval",
    )
    parsed = vars(args)
    overrides = {name: parsed[name] for name in override_names if name in parsed}
    options = TrainingOptions.from_preset(
        args.preset,
        base_model=args.base,
        prepared_dir=args.dataset,
        output_dir=args.output,
        **overrides,
    )
    return train_adaptation(options)


def _run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from .evaluation import EvaluationOptions, evaluate_checkpoint

    return evaluate_checkpoint(
        EvaluationOptions(
            model_dir=args.model_dir,
            manifest=args.manifest,
            output_dir=args.output,
            checkpoint=args.checkpoint,
            transcript_evaluator=args.transcript_evaluator,
            device=args.device,
            max_samples=args.max_samples,
            seed=args.seed,
            speed=args.speed,
            variation=args.variation,
            overwrite=args.overwrite,
            save_audio=args.save_audio,
        )
    )


def _prepared_dataset_json(args: argparse.Namespace) -> Path | None:
    """Return the prepared dataset.json named directly or by a sibling symbols file."""
    candidates: list[Path] = []
    if args.prepared_dataset is not None:
        prepared = Path(args.prepared_dataset).expanduser()
        candidates.append(prepared / "dataset.json" if prepared.is_dir() else prepared)
    if args.symbols is not None:
        candidates.append(Path(args.symbols).expanduser().parent / "dataset.json")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _resolved_frontend_hook(args: argparse.Namespace) -> Path | None:
    """Recover the hook file for a dataset prepared with a bundled frontend.

    Export needs the exact custom frontend source. When the dataset was prepared
    through the registry the file ships with the toolkit, so it is resolved here
    instead of asked for. An unreadable dataset falls through to export's own
    error, which names what is missing.
    """
    if args.frontend_hook is not None:
        return args.frontend_hook
    dataset_json = _prepared_dataset_json(args)
    if dataset_json is None:
        return None
    from .frontends import hook_path_for_record

    try:
        payload = json.loads(dataset_json.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    frontend = payload.get("frontend") if isinstance(payload, dict) else None
    if not isinstance(frontend, dict):
        return None
    return hook_path_for_record(frontend.get("registry"))


def _run_export(args: argparse.Namespace) -> dict[str, Any]:
    from .exporting import ExportOptions, export_checkpoint

    return export_checkpoint(
        ExportOptions(
            checkpoint=args.checkpoint,
            output_dir=args.output,
            config=args.config,
            symbols=args.symbols,
            prepared_dataset=args.prepared_dataset,
            frontend_hook=_resolved_frontend_hook(args),
            package_template=args.package_template,
            include_onnx=args.format == "onnx",
            onnx_opset=args.onnx_opset,
            model_name=args.model_name,
            source_revision=args.source_revision,
            overwrite=args.overwrite,
            verify=args.verify,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        report = args.handler(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.verbose:
            logging.exception("Command failed")
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
