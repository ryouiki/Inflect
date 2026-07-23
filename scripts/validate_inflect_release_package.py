from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(package: Path) -> int:
    manifest = json.loads((package / "release_manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = package / row["path"]
        if path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"size mismatch: {row['path']}")
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"hash mismatch: {row['path']}")
    return len(manifest["files"])


def waveform_hash(waveform: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(waveform, dtype=np.float32).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a standalone Inflect v2 release package.")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    package = args.package.resolve()
    manifest_files_verified = verify_manifest(package)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(package))
    spec = importlib.util.spec_from_file_location("inflect_release_inference", package / "inference.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load package inference module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    started = time.perf_counter()
    engine = module.InflectTTS(package, device=args.device)
    load_seconds = time.perf_counter() - started
    text = "Deterministic release packages should speak clearly every single time."
    first_sr, first = engine.synthesize(text, seed=17)
    second_sr, second = engine.synthesize(text, seed=17)
    different_sr, different = engine.synthesize(text, seed=18)
    slow_sr, slow = engine.synthesize(
        "Pacing controls should remain stable.", speed=0.8, seed=17
    )
    fast_sr, fast = engine.synthesize(
        "Pacing controls should remain stable.", speed=1.2, seed=17
    )
    chunked_text = " ".join(["This bounded paragraph checks punctuation-aware long text synthesis."] * 8)
    chunked_sr, chunked = engine.synthesize(chunked_text, seed=17)

    sample_rates = {first_sr, second_sr, different_sr, slow_sr, fast_sr, chunked_sr}
    if sample_rates != {engine.sample_rate}:
        raise RuntimeError(f"inconsistent sample rates returned: {sorted(sample_rates)}")

    if not np.array_equal(first, second):
        raise RuntimeError("same-seed synthesis is not deterministic")
    if np.array_equal(first, different):
        raise RuntimeError("different seeds produced identical waveforms")
    if slow.size <= fast.size:
        raise RuntimeError("speed control did not shorten faster synthesis")
    for name, waveform in {
        "default": first,
        "different_seed": different,
        "slow": slow,
        "fast": fast,
        "chunked": chunked,
    }.items():
        if not np.isfinite(waveform).all():
            raise RuntimeError(f"{name} contains non-finite samples")
        if float(np.max(np.abs(waveform))) > 1.0:
            raise RuntimeError(f"{name} exceeds normalized waveform range")
    try:
        engine.synthesize(text, speed=0.1)
    except ValueError:
        pass
    else:
        raise RuntimeError("invalid speed was not rejected")

    report = {
        "format": "inflect_v2_release_package_qa_v1",
        "package": package.name,
        "device": args.device,
        "sample_rate": engine.sample_rate,
        "manifest_files_verified": manifest_files_verified,
        "load_seconds": load_seconds,
        "deterministic_same_seed": True,
        "different_seed_changes_output": True,
        "default_seconds": first.size / engine.sample_rate,
        "slow_seconds": slow.size / engine.sample_rate,
        "fast_seconds": fast.size / engine.sample_rate,
        "chunked_characters": len(chunked_text),
        "chunked_seconds": chunked.size / engine.sample_rate,
        "default_waveform_sha256": waveform_hash(first),
        "peak": float(np.max(np.abs(first))),
        "rms": math.sqrt(float(np.mean(np.square(first, dtype=np.float64)))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
