"""Fail CI when the public Inflect v2 release surface becomes inconsistent."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "assets/inflect-micro-v2-cover.png",
    "assets/inflect-nano-v2-cover.png",
    "assets/evidence/cpu-throughput.svg",
    "docs/ARCHITECTURE.md",
    "docs/DEPLOYMENT.md",
    "docs/EVALUATION.md",
    "docs/README.md",
    "docs/INFLECT_V2_TECHNICAL_REPORT.md",
    "examples/download_and_speak.py",
    "examples/compare_models.py",
)
STALE_PUBLIC_TEXT = (
    "1.58x real time",
    "1.58× real time",
    "1.59x real time",
    "1.59× real time",
    "AMD Ryzen 9 3900X",
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_LINK = re.compile(r"(?:href|src)=\"([^\"]+)\"")


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing public release files:\n- " + "\n- ".join(missing))

    public_docs = [
        ROOT / "README.md",
        ROOT / "docs" / "DEPLOYMENT.md",
        ROOT / "docs" / "EVALUATION.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "INFLECT_V2_TECHNICAL_REPORT.md",
        ROOT / "docs" / "INFLECT_V2_RELEASE_NOTES_20260721.md",
    ]
    violations: list[str] = []
    for path in public_docs:
        text = path.read_text(encoding="utf-8")
        for stale in STALE_PUBLIC_TEXT:
            if stale in text:
                violations.append(f"{path.relative_to(ROOT)} contains {stale!r}")
        if re.search(r"\b[A-Za-z]:\\", text):
            violations.append(f"{path.relative_to(ROOT)} contains an absolute Windows path")

        for target in MARKDOWN_LINK.findall(text) + HTML_LINK.findall(text):
            target = target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            local_target = (path.parent / target).resolve()
            if not local_target.exists():
                violations.append(
                    f"{path.relative_to(ROOT)} links to missing local target {target!r}"
                )

    if violations:
        raise SystemExit("Stale public release claims:\n- " + "\n- ".join(violations))

    print(f"Validated {len(REQUIRED_FILES)} release files and {len(public_docs)} docs.")


if __name__ == "__main__":
    main()
