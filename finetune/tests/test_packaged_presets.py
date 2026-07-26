from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from inflect_finetune.training import TrainingOptions, load_preset


TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
PRESET_NAMES = ("balanced", "micro-12gb", "nano-8gb")


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_named_preset_loads_from_the_python_package(name: str) -> None:
    payload = load_preset(name)

    assert payload["max_steps"] == 20_000
    assert payload["checkpoint_interval"] == 1_000
    assert TrainingOptions.from_preset(
        name,
        base_model="base",
        prepared_dir="prepared",
        output_dir="output",
    ).preset is None


def test_built_wheel_contains_and_can_import_packaged_presets(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(TOOLKIT_ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        assert "inflect_finetune/presets/__init__.py" in archive.namelist()

    script = (
        "import sys;"
        f"sys.path.insert(0, {str(wheels[0])!r});"
        "from inflect_finetune.presets import available_presets,load_packaged_preset;"
        "assert available_presets()==('balanced','micro-12gb','nano-8gb');"
        "assert load_packaged_preset('nano-8gb')['batch_size']==1"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
