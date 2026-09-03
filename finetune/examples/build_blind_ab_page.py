"""Build a blind listening page from `evaluate` outputs.

Every automatic metric in this toolkit is a screen. The judge is listening, and
listening needs a page that cannot tell the listener which system they are
hearing. This builds one from the `audio/` directories `evaluate` writes.

What the page enforces, and why each one exists:

* **Fresh random labels per row.** Letters come from `os.urandom`, and a letter
  means a different system in every row. Tallying letters across rows therefore
  tallies noise; the arm lives only in `mapping.json`.
* **A sealed mapping.** `mapping.json` sits at the output root and the page never
  links to it. Score first, join afterwards with `tally_verdict.py`.
* **A real-audio anchor.** Without a recording on the page there is no way to
  tell "both systems are poor" from "this material is hard". The anchor is
  forced onto every row it exists for.
* **One level.** Every clip is scaled by pure gain to the same RMS with a peak
  guard, so loudness cannot stand in for quality.
* **Descriptive labels, never bare numbers.** A number invites arithmetic on an
  ordinal scale; a description keeps the reading anchored. The wording is fixed
  in this file and must stay byte-identical across rounds, because only
  within-round contrasts are comparable.
* **Mandatory free text.** Asking what the defect *sounded like* has repeatedly
  handed over defects no pre-registered metric was watching for.
* **A forced blur attribution.** A defect-severity scale rewards a system that
  smears everything into smoothness, so each row also asks which track sounds
  most muffled or smeared.
* **A byte-identical catch row.** One row can carry the same audio under two
  letters. A listener who scores them far apart tells you the round's noise
  floor before you read anything else into it.

```
python examples/build_blind_ab_page.py \
  --system step6000=evaluations/step6000-valtext \
  --system step8000=evaluations/step8000-valtext \
  --anchor evaluations/val-real-anchor \
  --must-include-ids review/high-register-ids.txt \
  --rows 32 \
  --output listening/ja-round1
```

The page writes its verdict to a JSON file the listener downloads. Nothing is
uploaded and nothing leaves the machine.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

# Fixed wording. Editing these between rounds breaks the only comparison the
# protocol allows, so they are defined once and quoted in the verdict file.
QUALITY_AXIS = {
    "label": "자연성",
    "question": "사람이 말한 것처럼 들리는가",
    "options": [
        "1 · 사람 목소리로 들리지 않는다",
        "2 · 말이지만 전반적으로 어색하다",
        "3 · 알아들을 수 있으나 어색한 구간이 있다",
        "4 · 대체로 자연스럽고 어색함이 약간 남는다",
        "5 · 실제 녹음과 구분하기 어렵다",
    ],
}
DEFECT_AXIS = {
    "label": "결함",
    "question": "갈라짐·잡음·끊김·발음 뭉갬이 있는가",
    "options": [
        "0 · 없음",
        "1 · 주의해 들으면 있다",
        "2 · 뚜렷하다",
    ],
}
LANGUAGE_QUESTION = "이 언어의 발화로 들리는가 (정체성·품질과 무관하게)"
LANGUAGE_OPTIONS = ["예", "아니오", "판단 어려움"]
NATURAL_CHOICE = "가장 자연스러운 트랙"
BLUR_CHOICE = "가장 뭉개지거나 먹먹한 트랙"
FREE_TEXT = "무엇처럼 들렸는가 · 무엇이 문제였는가 (필수)"

TARGET_RMS_DBFS = -24.0
PEAK_GUARD_DBFS = -1.0
ANCHOR_NAME = "real"
_LETTERS = "ABCDEFGH"


def load_ids(directory: Path) -> dict[str, Path]:
    """Return the sample id to WAV mapping in an `evaluate` output directory."""
    audio = directory / "audio"
    if not audio.is_dir():
        raise SystemExit(f"{directory} has no audio/ directory; run evaluate with --save-audio")
    return {path.stem: path for path in sorted(audio.glob("*.wav"))}


def levelled(path: Path) -> tuple[np.ndarray, int]:
    """Return the clip scaled by pure gain to the target RMS, peak guarded."""
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
    if rms > 0:
        samples = samples * (10 ** (TARGET_RMS_DBFS / 20) / rms)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    ceiling = 10 ** (PEAK_GUARD_DBFS / 20)
    if peak > ceiling:
        samples = samples * (ceiling / peak)
    return samples.astype(np.float32), sample_rate


def choose_rows(
    systems: dict[str, dict[str, Path]],
    required: list[str],
    limit: int,
    seed_bytes: bytes,
) -> list[str]:
    """Return the row ids to score: every required id first, then a random fill."""
    shared = set.intersection(*(set(ids) for ids in systems.values()))
    missing = [identifier for identifier in required if identifier not in shared]
    if missing:
        raise SystemExit(
            "these required ids are not present in every system: " + ", ".join(missing)
        )
    rows = [identifier for identifier in required]
    remaining = sorted(shared - set(rows))
    # Deterministic given the seed, and the seed is recorded in the mapping.
    order = sorted(remaining, key=lambda identifier: _digest(seed_bytes, identifier))
    rows.extend(order[: max(0, limit - len(rows))])
    return rows


def _digest(seed_bytes: bytes, text: str) -> str:
    import hashlib

    return hashlib.sha256(seed_bytes + text.encode("utf-8")).hexdigest()


def assign_letters(names: list[str], seed_bytes: bytes, row: str) -> dict[str, str]:
    """Return system name to letter, shuffled per row."""
    order = sorted(names, key=lambda name: _digest(seed_bytes, f"{row}\0{name}"))
    if len(order) > len(_LETTERS):
        raise SystemExit(f"at most {len(_LETTERS)} tracks per row")
    return {name: _LETTERS[index] for index, name in enumerate(order)}


def render_page(page_key: str, rows: list[dict], axes: dict) -> str:
    """Return the standalone HTML page."""
    blocks: list[str] = []
    for row in rows:
        letters = sorted(row["letters"])
        tracks = "".join(
            f"""
        <div class="track">
          <div class="letter">{letter}</div>
          <audio controls preload="none" src="tracks/{html.escape(row['id'])}/{letter}.wav"></audio>
          <label>{html.escape(QUALITY_AXIS['label'])}
            <select data-row="{html.escape(row['id'])}" data-letter="{letter}" data-field="quality">
              <option value="">—</option>
              {''.join(f'<option>{html.escape(option)}</option>' for option in QUALITY_AXIS['options'])}
            </select>
          </label>
          <label>{html.escape(DEFECT_AXIS['label'])}
            <select data-row="{html.escape(row['id'])}" data-letter="{letter}" data-field="defect">
              <option value="">—</option>
              {''.join(f'<option>{html.escape(option)}</option>' for option in DEFECT_AXIS['options'])}
            </select>
          </label>
          <label>{html.escape(LANGUAGE_QUESTION)}
            <select data-row="{html.escape(row['id'])}" data-letter="{letter}" data-field="language">
              <option value="">—</option>
              {''.join(f'<option>{html.escape(option)}</option>' for option in LANGUAGE_OPTIONS)}
            </select>
          </label>
        </div>"""
            for letter in letters
        )
        options = "".join(f"<option>{letter}</option>" for letter in letters)
        blocks.append(
            f"""
      <section class="row" id="row-{html.escape(row['id'])}">
        <h2>{html.escape(row['id'])}</h2>
        {f'<p class="text">{html.escape(row["text"])}</p>' if row.get("text") else ''}
        <div class="tracks">{tracks}</div>
        <div class="forced">
          <label>{html.escape(NATURAL_CHOICE)}
            <select data-row="{html.escape(row['id'])}" data-field="most_natural">
              <option value="">—</option>{options}
            </select>
          </label>
          <label>{html.escape(BLUR_CHOICE)}
            <select data-row="{html.escape(row['id'])}" data-field="most_blurred">
              <option value="">—</option>{options}
            </select>
          </label>
        </div>
        <label class="free">{html.escape(FREE_TEXT)}
          <textarea data-row="{html.escape(row['id'])}" data-field="comment" rows="2"></textarea>
        </label>
      </section>"""
        )

    axes_json = json.dumps(axes, ensure_ascii=False)
    return f"""<!doctype html>
<meta charset="utf-8">
<title>blind listening · {html.escape(page_key)}</title>
<style>
 body {{ font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; padding: 24px; max-width: 1100px; color: #16181d; background: #fbfbfc; }}
 h1 {{ font-size: 20px; }}
 .note {{ background: #fff6e0; border: 1px solid #e6cf95; padding: 12px 14px; border-radius: 8px; }}
 .row {{ border-top: 1px solid #dcdfe5; padding: 18px 0; }}
 .text {{ font-size: 16px; background: #fff; border: 1px solid #e2e5ea; border-radius: 6px; padding: 8px 10px; }}
 .tracks {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
 .track {{ background: #fff; border: 1px solid #e2e5ea; border-radius: 8px; padding: 10px 12px; }}
 .letter {{ font-weight: 700; font-size: 18px; }}
 label {{ display: block; margin-top: 8px; font-size: 13px; color: #4a5160; }}
 select, textarea {{ width: 100%; font: inherit; margin-top: 3px; }}
 .forced {{ display: flex; gap: 16px; margin-top: 12px; }}
 .forced label {{ flex: 1; }}
 .bar {{ position: sticky; bottom: 0; background: #fbfbfc; border-top: 1px solid #dcdfe5; padding: 12px 0; }}
 button {{ font: inherit; padding: 8px 14px; }}
 #status {{ margin-left: 12px; color: #4a5160; }}
</style>
<h1>블라인드 청취 · {html.escape(page_key)}</h1>
<p class="note">
 라벨(A/B/C…)은 <b>행마다 다시 섞인다</b>. 라벨을 행 사이에서 합산하면 잡음을 합산하는 것이다 —
 귀속은 <code>mapping.json</code>에만 있고, 채점이 끝나기 전에는 열지 않는다.
 모든 트랙은 같은 RMS로 순수 게인 정렬됐다. 절대 점수는 <b>라운드 사이에 비교하지 않는다</b>;
 이 페이지 안의 대비만 읽는다. 자유기술은 필수다.
</p>
{''.join(blocks)}
<div class="bar">
 <button id="save">판정 JSON 내려받기</button>
 <span id="status"></span>
</div>
<script>
const PAGE_KEY = {json.dumps(page_key)};
const AXES = {axes_json};
const store = () => {{
  const state = {{}};
  document.querySelectorAll("[data-field]").forEach(element => {{
    const row = element.dataset.row, field = element.dataset.field, letter = element.dataset.letter || "";
    state[[row, letter, field].join("|")] = element.value;
  }});
  return state;
}};
const restore = () => {{
  let saved;
  try {{ saved = JSON.parse(localStorage.getItem(PAGE_KEY) || "{{}}"); }} catch (error) {{ saved = {{}}; }}
  document.querySelectorAll("[data-field]").forEach(element => {{
    const key = [element.dataset.row, element.dataset.letter || "", element.dataset.field].join("|");
    if (saved[key]) element.value = saved[key];
  }});
}};
const persist = () => {{
  try {{ localStorage.setItem(PAGE_KEY, JSON.stringify(store())); }} catch (error) {{}}
}};
document.addEventListener("input", persist);
document.addEventListener("change", persist);
restore();
document.getElementById("save").addEventListener("click", () => {{
  const rows = {{}};
  let missing = 0;
  document.querySelectorAll("section.row").forEach(section => {{
    const id = section.id.replace(/^row-/, "");
    const entry = {{ tracks: {{}} }};
    section.querySelectorAll("[data-field]").forEach(element => {{
      const field = element.dataset.field, letter = element.dataset.letter;
      if (letter) {{
        entry.tracks[letter] = entry.tracks[letter] || {{}};
        entry.tracks[letter][field] = element.value;
      }} else {{
        entry[field] = element.value;
      }}
      if (!element.value) missing += 1;
    }});
    rows[id] = entry;
  }});
  const verdict = {{ format: "inflect_listening_verdict_v1", page_key: PAGE_KEY, axes: AXES, unanswered: missing, rows }};
  const blob = new Blob([JSON.stringify(verdict, null, 1)], {{ type: "application/json" }});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = PAGE_KEY + "-verdict.json";
  link.click();
  document.getElementById("status").textContent = missing
    ? ("빈 항목 " + missing + "개가 있다. 그대로 내려받았다.")
    : "모든 항목이 채워졌다.";
}});
</script>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="An evaluate output directory to score, named for the mapping.",
    )
    parser.add_argument("--anchor", type=Path, help="Real-audio evaluate output; forced onto the page.")
    parser.add_argument("--must-include-ids", type=Path, help="File of sample ids, one per line.")
    parser.add_argument("--texts", type=Path, help="JSONL with id and text, shown beside the players.")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--catch-rows", type=int, default=1, help="Rows carrying one system twice.")
    parser.add_argument("--page-key", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    systems: dict[str, dict[str, Path]] = {}
    system_paths: dict[str, Path] = {}
    for entry in args.system:
        if "=" not in entry:
            raise SystemExit(f"--system expects NAME=DIR, got {entry!r}")
        name, _, directory = entry.partition("=")
        if name in systems or name == ANCHOR_NAME:
            raise SystemExit(f"duplicate or reserved system name {name!r}")
        system_paths[name] = Path(directory).resolve()
        systems[name] = load_ids(Path(directory))
    if args.anchor:
        systems[ANCHOR_NAME] = load_ids(args.anchor)

    texts: dict[str, str] = {}
    if args.texts:
        for line in args.texts.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                texts[str(record.get("id"))] = str(record.get("text", ""))

    required: list[str] = []
    if args.must_include_ids:
        required = [
            stripped
            for line in args.must_include_ids.read_text(encoding="utf-8").splitlines()
            if (stripped := line.strip()) and not stripped.startswith("#")
        ]

    seed_bytes = os.urandom(32)
    row_ids = choose_rows(systems, required, args.rows, seed_bytes)
    if not row_ids:
        raise SystemExit("no rows are shared by every system")

    output = args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    (output / "tracks").mkdir(parents=True, exist_ok=True)

    catch_rows = set(row_ids[: max(0, args.catch_rows)])
    scored_names = [name for name in systems if name != ANCHOR_NAME]
    mapping: dict[str, dict[str, str]] = {}
    page_rows: list[dict] = []
    for row in row_ids:
        names = list(systems)
        if row in catch_rows and scored_names:
            duplicate = min(scored_names, key=lambda name: _digest(seed_bytes, f"catch\0{row}\0{name}"))
            names.append(f"{duplicate}#catch")
        letters = assign_letters(names, seed_bytes, row)
        destination = output / "tracks" / row
        destination.mkdir(parents=True, exist_ok=True)
        for name, letter in letters.items():
            source_name = name.split("#", 1)[0]
            samples, sample_rate = levelled(systems[source_name][row])
            sf.write(str(destination / f"{letter}.wav"), samples, sample_rate, subtype="PCM_16")
        mapping[row] = {letter: name for name, letter in letters.items()}
        page_rows.append({"id": row, "letters": list(letters.values()), "text": texts.get(row, "")})

    page_key = args.page_key or output.name
    axes = {
        "quality": QUALITY_AXIS,
        "defect": DEFECT_AXIS,
        "language": {"question": LANGUAGE_QUESTION, "options": LANGUAGE_OPTIONS},
        "forced": {"most_natural": NATURAL_CHOICE, "most_blurred": BLUR_CHOICE},
        "free_text": FREE_TEXT,
        "levelling": {"target_rms_dbfs": TARGET_RMS_DBFS, "peak_guard_dbfs": PEAK_GUARD_DBFS},
    }
    (output / "index.html").write_text(render_page(page_key, page_rows, axes), encoding="utf-8")
    (output / "mapping.json").write_text(
        json.dumps(
            {
                "format": "inflect_listening_mapping_v1",
                "page_key": page_key,
                "systems": {name: str(path) for name, path in system_paths.items()},
                "anchor": str(args.anchor.resolve()) if args.anchor else None,
                "required_ids": required,
                "catch_rows": sorted(catch_rows),
                "row_count": len(row_ids),
                "axes": axes,
                "rows": mapping,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"rows:      {len(row_ids)} ({len(required)} required, {len(catch_rows)} catch)")
    print(f"tracks:    {len(systems)} systems" + (" including a real anchor" if args.anchor else ""))
    print(f"page:      {output / 'index.html'}")
    print(f"mapping:   {output / 'mapping.json'} — do not open before scoring")
    if not args.anchor:
        print("WARNING: no real-audio anchor on the page; 'both are poor' becomes unreadable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
