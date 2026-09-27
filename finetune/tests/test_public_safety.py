from __future__ import annotations

import re
from pathlib import Path

from inflect_finetune.exporting import _copy_public_runtime


TOOLKIT_ROOT = Path(__file__).resolve().parents[1]

PROHIBITED_PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    "private storage path": re.compile(r"Inflect-Storage", re.IGNORECASE),
    "rental instance detail": re.compile(r"\b(vast\.ai|instance[_ -]?id|ssh_host)\b", re.IGNORECASE),
    "credential assignment": re.compile(
        r"\b(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|WANDB_API_KEY)\s*=\s*[\"'][^\"']+",
        re.IGNORECASE,
    ),
}


EXCLUDED_DIRECTORIES = {"__pycache__", "build", "dist"}


def _is_published_surface(relative: Path) -> bool:
    """Return whether a file under the toolkit belongs to the public surface.

    A virtual environment, a build tree, or an editable-install artifact lives
    inside the toolkit without being part of what it publishes, and the
    documented install creates ``.venv`` in this directory, so those trees are
    skipped. Site-packages also carries files that are not valid UTF-8, which
    would fail the read below rather than the assertion.
    """
    return not any(
        part in EXCLUDED_DIRECTORIES or part.startswith(".") or part.endswith(".egg-info")
        for part in relative.parts[:-1]
    )


def public_text_files() -> list[Path]:
    extensions = {
        ".py",
        ".md",
        ".toml",
        ".json",
        ".jsonl",
        ".csv",
        ".yaml",
        ".yml",
        ".txt",
    }
    return [
        path
        for path in TOOLKIT_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
        and path.name != Path(__file__).name
        and _is_published_surface(path.relative_to(TOOLKIT_ROOT))
    ]


def test_public_toolkit_contains_no_private_infrastructure_references() -> None:
    findings: list[str] = []
    for path in public_text_files():
        text = path.read_text(encoding="utf-8")
        for label, pattern in PROHIBITED_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(TOOLKIT_ROOT)}: {label}")
    assert not findings, "\n".join(findings)


def test_contract_explicitly_separates_public_adaptation_from_base_recipe() -> None:
    contract = (TOOLKIT_ROOT / "CONTRACT.md").read_text(encoding="utf-8")
    assert "not" in contract.lower()
    assert "private recipe" in contract.lower()
    assert "dataset speaker becomes the checkpoint voice" in contract.lower()


def test_package_metadata_does_not_claim_unvalidated_universal_support() -> None:
    metadata = (TOOLKIT_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "generic language and fixed-voice adaptation" in metadata
    assert "universal" not in metadata


def test_public_runtime_export_omits_python_bytecode(tmp_path: Path) -> None:
    template = tmp_path / "template"
    runtime = template / "runtime"
    cache = runtime / "__pycache__"
    cache.mkdir(parents=True)
    (runtime / "models.py").write_text("MODEL = True\n", encoding="utf-8")
    (cache / "models.cpython-312.pyc").write_bytes(b"bytecode")

    destination = tmp_path / "export"
    destination.mkdir()
    copied = _copy_public_runtime(template, destination)

    assert (destination / "runtime" / "models.py").is_file()
    assert not (destination / "runtime" / "__pycache__").exists()
    assert all(path.suffix != ".pyc" for path in copied)
