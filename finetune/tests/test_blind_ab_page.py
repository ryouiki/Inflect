from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "build_blind_ab_page.py"


def load_example():
    spec = importlib.util.spec_from_file_location("_blind_ab_page", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def page_module():
    return load_example()


def write_wav(path: Path, amplitude: float, seconds: float = 0.25, sample_rate: int = 24_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    sf.write(str(path), (amplitude * np.sin(2 * np.pi * 220 * time)).astype(np.float32), sample_rate)


def evaluate_output(root: Path, name: str, identifiers: list[str], amplitude: float) -> Path:
    directory = root / name
    for identifier in identifiers:
        write_wav(directory / "audio" / f"{identifier}.wav", amplitude)
    return directory


def test_levelling_hits_the_target_rms_and_respects_the_peak_guard(page_module, tmp_path):
    quiet = tmp_path / "quiet.wav"
    write_wav(quiet, 0.01)
    samples, sample_rate = page_module.levelled(quiet)
    assert sample_rate == 24_000
    rms_dbfs = 20 * math.log10(float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))))
    peak_dbfs = 20 * math.log10(float(np.max(np.abs(samples))))
    assert rms_dbfs == pytest.approx(page_module.TARGET_RMS_DBFS, abs=0.05)
    assert peak_dbfs <= page_module.PEAK_GUARD_DBFS + 1e-6


def test_levelling_never_exceeds_the_peak_guard_even_when_that_lowers_rms(page_module, tmp_path):
    """A clip whose crest factor is high must be attenuated, not clipped."""
    loud = tmp_path / "loud.wav"
    sample_rate = 24_000
    samples = np.zeros(sample_rate // 4, dtype=np.float32)
    samples[::500] = 1.0  # very peaky, very low RMS
    sf.write(str(loud), samples, sample_rate)
    levelled, _ = page_module.levelled(loud)
    peak_dbfs = 20 * math.log10(float(np.max(np.abs(levelled))))
    assert peak_dbfs == pytest.approx(page_module.PEAK_GUARD_DBFS, abs=0.05)


def write_peaky_wav(path: Path, crest_db: float, sample_rate: int = 24_000) -> None:
    """Write a clip whose peak sits `crest_db` above its RMS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(sample_rate // 4, dtype=np.float64)
    # A sine at full scale has a 3.01 dB crest; spacing lone spikes raises it
    # without touching the peak, which is what a real transient does.
    spacing = max(1, round(10 ** (crest_db / 10)))
    samples[::spacing] = 1.0
    sf.write(str(path), samples.astype(np.float32), sample_rate)


def rms_dbfs(samples: np.ndarray) -> float:
    return 20 * math.log10(float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))))


def test_a_page_puts_every_track_at_one_rms_whatever_its_crest_factor(page_module, tmp_path):
    """Levelling clip by clip made the peaky ones quieter and nothing else."""
    root = tmp_path / "sources"
    for name, crest in (("flat", 6.0), ("mid", 14.0), ("peaky", 28.0)):
        write_peaky_wav(root / name / "audio" / "0001.wav", crest)
    output = tmp_path / "page"
    exit_code = page_module.main(
        [
            "--system", f"flat={root / 'flat'}",
            "--system", f"mid={root / 'mid'}",
            "--system", f"peaky={root / 'peaky'}",
            "--rows", "1",
            "--catch-rows", "0",
            "--output", str(output),
        ]
    )
    assert exit_code == 0

    levels = []
    for track in sorted((output / "tracks" / "0001").glob("*.wav")):
        samples, _ = sf.read(str(track), dtype="float32", always_2d=False)
        levels.append(rms_dbfs(samples))
        peak_dbfs = 20 * math.log10(float(np.max(np.abs(samples))))
        assert peak_dbfs <= page_module.PEAK_GUARD_DBFS + 0.05
    assert len(levels) == 3
    # PCM_16 quantisation is the only thing left between them.
    assert max(levels) - min(levels) < 0.05

    mapping = json.loads((output / "mapping.json").read_text(encoding="utf-8"))
    levelling = mapping["levelling"]
    assert levelling["target_rms_dbfs"] == pytest.approx(page_module.PEAK_GUARD_DBFS - 28.0, abs=0.3)
    assert levelling["target_rms_dbfs"] < page_module.TARGET_RMS_DBFS
    assert levelling["limited_by"] == ["0001/peaky"]
    assert levelling["peak_guard_fired"] == []
    assert levels[0] == pytest.approx(levelling["target_rms_dbfs"], abs=0.1)
    # The page carries the policy; naming the loudest system on it would tell
    # the listener which track to distrust.
    page = (output / "index.html").read_text(encoding="utf-8")
    assert "limited_by" not in page
    assert "peaky" not in page


def test_one_pathological_clip_stops_the_page_instead_of_quietly_dragging_it_down(
    page_module, tmp_path
):
    root = tmp_path / "sources"
    write_peaky_wav(root / "fine" / "audio" / "0001.wav", 8.0)
    write_peaky_wav(root / "spiky" / "audio" / "0001.wav", 40.0)
    output = tmp_path / "page"
    with pytest.raises(SystemExit) as failure:
        page_module.main(
            [
                "--system", f"fine={root / 'fine'}",
                "--system", f"spiky={root / 'spiky'}",
                "--rows", "1",
                "--catch-rows", "0",
                "--output", str(output),
            ]
        )
    message = str(failure.value)
    assert "0001/spiky" in message, "the operator has to be told which clip to look at"
    assert f"{page_module.LEVEL_FLOOR_DBFS:.1f}" in message
    assert not (output / "index.html").exists()


def test_the_catch_track_can_be_pinned_to_a_named_system(page_module, tmp_path):
    root = tmp_path / "sources"
    for name in ("early", "late"):
        for identifier in ("0001", "0002"):
            write_wav(root / name / "audio" / f"{identifier}.wav", 0.2)
    output = tmp_path / "page"
    exit_code = page_module.main(
        [
            "--system", f"early={root / 'early'}",
            "--system", f"late={root / 'late'}",
            "--rows", "2",
            "--catch-rows", "2",
            "--catch-system", "late",
            "--output", str(output),
        ]
    )
    assert exit_code == 0
    mapping = json.loads((output / "mapping.json").read_text(encoding="utf-8"))
    assert mapping["catch_system"] == "late"
    for row, letters in mapping["rows"].items():
        assert sorted(letters.values()) == ["early", "late", "late#catch"], row
        pair = [letter for letter, name in letters.items() if name.startswith("late")]
        first, second = (
            (output / "tracks" / row / f"{letter}.wav").read_bytes() for letter in pair
        )
        assert first == second, "a catch pair that is not byte-identical measures nothing"

    with pytest.raises(SystemExit, match="not one of"):
        page_module.main(
            [
                "--system", f"early={root / 'early'}",
                "--rows", "1",
                "--catch-system", "missing",
                "--output", str(tmp_path / "other"),
            ]
        )


def test_a_round_note_reaches_the_listener_without_touching_the_fixed_wording(page_module, tmp_path):
    """A round can ask one extra thing; the scale wording stays byte-identical."""
    root = tmp_path / "sources"
    for name in ("one", "two"):
        write_wav(root / name / "audio" / "0001.wav", 0.2)
    output = tmp_path / "page"
    note = "말소리인지, 언어인지, 알아들을 수 있는지를 구분해 적어 주십시오."
    assert page_module.main(
        [
            "--system", f"one={root / 'one'}",
            "--system", f"two={root / 'two'}",
            "--rows", "1",
            "--catch-rows", "0",
            "--note", note,
            "--output", str(output),
        ]
    ) == 0
    page = (output / "index.html").read_text(encoding="utf-8")
    assert note in page
    for option in page_module.QUALITY_AXIS["options"]:
        assert option in page
    assert json.loads((output / "mapping.json").read_text(encoding="utf-8"))["note"] == note


def test_letters_are_unique_per_row_and_reshuffled_between_rows(page_module):
    names = ["one", "two", "three", "real"]
    seed = b"\x01" * 32
    first = page_module.assign_letters(names, seed, "row-a")
    second = page_module.assign_letters(names, seed, "row-b")
    assert sorted(first) == sorted(names)
    assert len(set(first.values())) == len(names)
    assert first != second, "a letter must not mean the same system in every row"


def test_required_rows_come_first_and_a_missing_one_is_fatal(page_module):
    systems = {
        "a": {"x": Path("x"), "y": Path("y"), "z": Path("z")},
        "b": {"x": Path("x"), "y": Path("y"), "z": Path("z")},
    }
    rows = page_module.choose_rows(systems, ["z"], 2, b"\x02" * 32)
    assert rows[0] == "z" and len(rows) == 2
    with pytest.raises(SystemExit):
        page_module.choose_rows(systems, ["absent"], 2, b"\x02" * 32)


def test_page_seals_the_mapping_and_forces_the_anchor(page_module, tmp_path):
    identifiers = [f"{index:04d}" for index in range(1, 7)]
    first = evaluate_output(tmp_path, "step1000", identifiers, 0.2)
    second = evaluate_output(tmp_path, "step2000", identifiers, 0.3)
    anchor = evaluate_output(tmp_path, "anchor", identifiers, 0.4)
    required = tmp_path / "required.txt"
    required.write_text("0003\n# a comment\n", encoding="utf-8")
    output = tmp_path / "round"

    exit_code = page_module.main(
        [
            "--system", f"early={first}",
            "--system", f"late={second}",
            "--anchor", str(anchor),
            "--must-include-ids", str(required),
            "--rows", "4",
            "--catch-rows", "1",
            "--output", str(output),
        ]
    )
    assert exit_code == 0

    page = (output / "index.html").read_text(encoding="utf-8")
    # Naming the mapping in the instructions is the point; loading it is not.
    assert 'href="mapping.json"' not in page
    assert 'src="mapping.json"' not in page
    assert "fetch(" not in page and "XMLHttpRequest" not in page
    for occurrence in page.split("mapping.json")[:-1]:
        assert occurrence.endswith("<code>"), "mapping.json may only be named, never loaded"
    assert page_module.QUALITY_AXIS["options"][0] in page
    assert page_module.FREE_TEXT in page
    for option in page_module.DEFECT_AXIS["options"]:
        assert option in page, "a bare number must never stand in for a descriptive label"

    mapping = json.loads((output / "mapping.json").read_text(encoding="utf-8"))
    assert mapping["format"] == "inflect_listening_mapping_v1"
    assert mapping["required_ids"] == ["0003"]
    assert mapping["row_count"] == 4
    assert next(iter(mapping["rows"])) == "0003"
    for row, letters in mapping["rows"].items():
        assert page_module.ANCHOR_NAME in letters.values(), f"row {row} lost the real anchor"
        for letter in letters:
            assert (output / "tracks" / row / f"{letter}.wav").is_file()

    catch_row = mapping["catch_rows"][0]
    duplicated = [name for name in mapping["rows"][catch_row].values() if name.endswith("#catch")]
    assert duplicated, "the catch row must carry one system twice"
    base = duplicated[0].split("#", 1)[0]
    letters = {name: letter for letter, name in mapping["rows"][catch_row].items()}
    left = sf.read(str(output / "tracks" / catch_row / f"{letters[base]}.wav"))[0]
    right = sf.read(str(output / "tracks" / catch_row / f"{letters[duplicated[0]]}.wav"))[0]
    assert np.array_equal(left, right), "the catch pair must be byte-identical after levelling"


def test_a_non_empty_output_directory_is_refused(page_module, tmp_path):
    identifiers = ["0001", "0002"]
    system = evaluate_output(tmp_path, "only", identifiers, 0.2)
    output = tmp_path / "busy"
    output.mkdir()
    (output / "left-over.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        page_module.main(["--system", f"only={system}", "--rows", "2", "--output", str(output)])


TALLY = Path(__file__).resolve().parents[1] / "examples" / "tally_verdict.py"


def load_tally():
    spec = importlib.util.spec_from_file_location("_tally_verdict", TALLY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tally_joins_letters_through_the_mapping_and_reads_the_catch_pair(page_module, tmp_path, capsys):
    identifiers = [f"{index:04d}" for index in range(1, 5)]
    early = evaluate_output(tmp_path, "step1000", identifiers, 0.2)
    late = evaluate_output(tmp_path, "step2000", identifiers, 0.3)
    output = tmp_path / "round"
    assert page_module.main(
        [
            "--system", f"early={early}",
            "--system", f"late={late}",
            "--rows", "3",
            "--catch-rows", "1",
            "--output", str(output),
        ]
    ) == 0

    mapping = json.loads((output / "mapping.json").read_text(encoding="utf-8"))
    axes = mapping["axes"]
    best = axes["quality"]["options"][-1]
    worst = axes["quality"]["options"][0]
    clean = axes["defect"]["options"][0]

    # A verdict where 'late' is always the better track, scored per row by letter.
    rows: dict[str, dict] = {}
    for row, letters in mapping["rows"].items():
        tracks = {}
        for letter, name in letters.items():
            better = name.split("#", 1)[0] == "late"
            tracks[letter] = {
                "quality": best if better else worst,
                "defect": clean,
                "language": axes["language"]["options"][0],
            }
        winner = next(letter for letter, name in letters.items() if name == "late")
        loser = next(letter for letter, name in letters.items() if name == "early")
        rows[row] = {
            "tracks": tracks,
            "most_natural": winner,
            "most_blurred": loser,
            "comment": f"{row}: sibilance on the late track",
        }
    verdict = tmp_path / "verdict.json"
    verdict.write_text(
        json.dumps(
            {
                "format": "inflect_listening_verdict_v1",
                "page_key": mapping["page_key"],
                "axes": axes,
                "rows": rows,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tally = load_tally()
    tally_json = tmp_path / "tally.json"
    assert tally.main(["--mapping", str(output / "mapping.json"), "--verdict", str(verdict), "--output", str(tally_json)]) == 0
    printed = capsys.readouterr().out
    assert "most_natural: {'late': 3}" in printed
    assert "most_blurred: {'early': 3}" in printed
    assert "noise floor for this round: 0 option step(s)" in printed

    summary = json.loads(tally_json.read_text(encoding="utf-8"))
    assert summary["unanswered"] == 0
    assert summary["median_option_index"]["late"]["quality"] > summary["median_option_index"]["early"]["quality"]
    assert len(summary["comments"]) == 3


def test_tally_refuses_a_verdict_from_another_page(tmp_path):
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"page_key": "round-one", "rows": {}}), encoding="utf-8")
    verdict = tmp_path / "verdict.json"
    verdict.write_text(json.dumps({"page_key": "round-two", "rows": {}}), encoding="utf-8")
    tally = load_tally()
    with pytest.raises(SystemExit):
        tally.main(["--mapping", str(mapping), "--verdict", str(verdict)])
