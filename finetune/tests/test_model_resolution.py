from __future__ import annotations

from pathlib import Path

import huggingface_hub

from inflect_finetune.modeling import resolve_base_model


def test_hugging_face_repo_id_and_revision_are_resolved(
    tmp_path: Path, monkeypatch
) -> None:
    downloaded = tmp_path / "snapshot"
    (downloaded / "runtime").mkdir(parents=True)
    (downloaded / "config.json").write_text("{}\n", encoding="utf-8")
    (downloaded / "model.pth").write_bytes(b"checkpoint")
    observed: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        observed.update(kwargs)
        return str(downloaded)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    resolved = resolve_base_model("example/Inflect-Adapted@release-1")

    assert resolved == downloaded.resolve()
    assert observed["repo_id"] == "example/Inflect-Adapted"
    assert observed["revision"] == "release-1"
    assert "model.pth" in observed["allow_patterns"]
