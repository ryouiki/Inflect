"""Export a language-aware inference package through the public Python API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from inflect_finetune.exporting import ExportOptions, export_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-dataset", type=Path, required=True)
    parser.add_argument("--package-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--symbols", type=Path)
    parser.add_argument("--frontend-hook", type=Path)
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    symbols = args.symbols or args.prepared_dataset / "symbols.json"
    report = export_checkpoint(
        ExportOptions(
            checkpoint=args.checkpoint,
            output_dir=args.output,
            config=args.config,
            symbols=symbols,
            package_template=args.package_template,
            prepared_dataset=args.prepared_dataset,
            frontend_hook=args.frontend_hook,
            include_onnx=args.onnx,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
