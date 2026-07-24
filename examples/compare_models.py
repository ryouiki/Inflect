"""Render the same text and seed with Inflect Micro v2 and Nano v2."""

from __future__ import annotations

import argparse
from pathlib import Path

from download_and_speak import synthesize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        default="Small models can still speak clearly, naturally, and locally.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("inflect_compare"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--variation", type=float, default=0.667)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for model in ("micro", "nano"):
        output = args.output_dir / f"inflect-{model}-v2.wav"
        synthesize(
            model,
            args.text,
            output,
            device=args.device,
            speed=args.speed,
            variation=args.variation,
            seed=args.seed,
        )
        print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
