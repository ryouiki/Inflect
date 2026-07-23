from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SKIP_FILES = {"release_manifest.json", "model-card-preview.html"}
SKIP_DIRECTORIES = {".cache", "__pycache__"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def refresh(package: Path) -> dict[str, object]:
    manifest_path = package / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = []
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.name in SKIP_FILES:
            continue
        relative = path.relative_to(package)
        if any(part in SKIP_DIRECTORIES for part in relative.parts):
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest["files"] = files
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "package": package.name,
        "files": len(files),
        "model_sha256": next(
            row["sha256"] for row in files if row["path"] == "model.pth"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh file sizes and SHA-256 hashes in an Inflect release package."
    )
    parser.add_argument("packages", nargs="+", type=Path)
    args = parser.parse_args()
    for package in args.packages:
        print(json.dumps(refresh(package.resolve())))


if __name__ == "__main__":
    main()
