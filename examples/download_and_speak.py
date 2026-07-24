"""Download an Inflect v2 release package and synthesize a WAV file."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPOS = {
    "micro": "owensong/Inflect-Micro-v2",
    "nano": "owensong/Inflect-Nano-v2",
}


def synthesize(
    model: str,
    text: str,
    output: Path,
    *,
    device: str,
    speed: float,
    variation: float,
    seed: int,
) -> None:
    repo_id = MODEL_REPOS[model]
    model_dir = Path(snapshot_download(repo_id=repo_id))
    inference_script = model_dir / "inference.py"
    if not inference_script.is_file():
        raise FileNotFoundError(f"{repo_id} does not contain inference.py")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(inference_script),
        "--model-dir",
        str(model_dir),
        "--text",
        text,
        "--output",
        str(output.resolve()),
        "--device",
        device,
        "--speed",
        str(speed),
        "--variation",
        str(variation),
        "--seed",
        str(seed),
    ]
    subprocess.run(command, check=True, cwd=model_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_REPOS, default="micro")
    parser.add_argument(
        "--text",
        default="A complete local voice can fit almost anywhere.",
    )
    parser.add_argument("--output", type=Path, default=Path("inflect.wav"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--variation", type=float, default=0.667)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthesize(
        args.model,
        args.text,
        args.output,
        device=args.device,
        speed=args.speed,
        variation=args.variation,
        seed=args.seed,
    )
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
