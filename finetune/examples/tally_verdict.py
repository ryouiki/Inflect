"""Join a listening verdict to its sealed mapping and tally by system.

A letter is a per-row label, so counting letters counts noise. The arm lives in
`mapping.json`, which is why scoring and tallying are separate steps and this
script is the only place the two meet.

What it reports, and what it refuses to report:

* Per system: the distribution of each descriptive axis, as counts and a median
  of the option index. Never a mean — the axes are ordinal.
* The forced choices, as counts per system.
* The catch pair, if the page carried one, as the option steps between two
  scores for byte-identical audio. That is a fact about that pair. It is not
  the round's noise floor and not a bound on how much any other contrast could
  have varied, so no other contrast is read against it.
* Free text, per row, because that column has repeatedly carried the defect no
  axis was watching.

Only the axes the page actually asked are counted, so a verdict sealed before
an axis existed still tallies the way it did on the day it was read.

It does not compare rounds. Absolute scores drift between sessions, so only
within-round contrasts mean anything, and a cross-round number would invite
exactly the comparison the protocol forbids.

```
python examples/tally_verdict.py \
  --mapping listening/ja-round1/mapping.json \
  --verdict ~/Downloads/ja-round1-verdict.json
```
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

KNOWN_AXES = ("quality", "defect", "language", "ringing")
FORCED = ("most_natural", "most_blurred")
ROW_TEXT = ("comment", "ring_detail", "ring_order")


def axis_names(axes: dict) -> tuple[str, ...]:
    """The axes this page actually carried, in a fixed order.

    Pages built before an axis existed have no cell for it, and counting one
    would report every track of every sealed verdict as unanswered. The page's
    own `axes` block is the authority on what was asked.
    """
    return tuple(axis for axis in KNOWN_AXES if axis in axes)


def option_index(axes: dict, axis: str, value: str) -> int | None:
    """Return the position of a chosen option, for a median over an ordinal axis."""
    options = axes.get(axis, {}).get("options") or []
    return options.index(value) if value in options else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write the tally as JSON as well.")
    args = parser.parse_args(argv)

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    verdict = json.loads(args.verdict.read_text(encoding="utf-8"))
    if mapping.get("page_key") != verdict.get("page_key"):
        raise SystemExit(
            f"page keys differ: mapping {mapping.get('page_key')!r} vs "
            f"verdict {verdict.get('page_key')!r}"
        )
    axes = mapping.get("axes", {})
    rows = mapping.get("rows", {})
    asked = axis_names(axes)

    scores: dict[str, dict[str, collections.Counter]] = collections.defaultdict(
        lambda: {axis: collections.Counter() for axis in asked}
    )
    ordinals: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: {axis: [] for axis in asked}
    )
    forced: dict[str, collections.Counter] = {name: collections.Counter() for name in FORCED}
    texts: dict[str, list[tuple[str, str]]] = {field: [] for field in ROW_TEXT}
    unanswered = 0

    for row, letters in rows.items():
        scored = verdict.get("rows", {}).get(row)
        if not scored:
            continue
        for letter, name in letters.items():
            track = scored.get("tracks", {}).get(letter, {})
            for axis in asked:
                value = track.get(axis, "")
                if not value:
                    unanswered += 1
                    continue
                scores[name][axis][value] += 1
                index = option_index(axes, axis, value)
                if index is not None:
                    ordinals[name][axis].append(index)
        for choice in FORCED:
            letter = scored.get(choice, "")
            if letter and letter in letters:
                forced[choice][letters[letter]] += 1
        for field in ROW_TEXT:
            written = (scored.get(field) or "").strip()
            if written:
                texts[field].append((row, written))

    print(f"page {mapping.get('page_key')} · rows scored {len(verdict.get('rows', {}))}/{len(rows)}")
    print(f"unanswered axis cells: {unanswered}")
    print()
    for name in sorted(scores):
        print(f"[{name}]")
        for axis in asked:
            counter = scores[name][axis]
            if not counter:
                continue
            values = ordinals[name][axis]
            median = f"median option index {statistics.median(values):.1f}" if values else ""
            print(f"  {axis:9} {dict(counter)} {median}")
        print()
    for choice in FORCED:
        if forced[choice]:
            print(f"{choice}: {dict(forced[choice])}")

    catch_rows = mapping.get("catch_rows") or []
    for row in catch_rows:
        letters = mapping["rows"].get(row, {})
        pairs = collections.defaultdict(list)
        for letter, name in letters.items():
            pairs[name.split("#", 1)[0]].append(letter)
        for name, group in pairs.items():
            if len(group) < 2:
                continue
            scored = verdict.get("rows", {}).get(row, {}).get("tracks", {})
            readings = {
                letter: scored.get(letter, {}).get("quality", "") for letter in sorted(group)
            }
            indexes = [
                option_index(axes, "quality", value) for value in readings.values() if value
            ]
            spread = (
                max(index for index in indexes if index is not None)
                - min(index for index in indexes if index is not None)
                if len([index for index in indexes if index is not None]) > 1
                else None
            )
            print()
            print(f"catch row {row} · {name} appears as {sorted(group)}")
            for letter, value in readings.items():
                print(f"  {letter}: {value or '—'}")
            if spread is not None:
                print(
                    f"  이 복제쌍의 옵션 단차: {spread}단 "
                    "(바이트 동일 오디오 · 라운드 전체의 잡음 바닥도 변동의 상한도 아니다)"
                )

    for field in ROW_TEXT:
        if texts[field]:
            print()
            print(field)
            for row, written in texts[field]:
                print(f"  {row}: {written}")

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "format": "inflect_listening_tally_v1",
                    "page_key": mapping.get("page_key"),
                    "rows_scored": len(verdict.get("rows", {})),
                    "rows_total": len(rows),
                    "unanswered": unanswered,
                    "scores": {
                        name: {axis: dict(counter) for axis, counter in axis_map.items()}
                        for name, axis_map in scores.items()
                    },
                    "median_option_index": {
                        name: {
                            axis: statistics.median(values)
                            for axis, values in axis_map.items()
                            if values
                        }
                        for name, axis_map in ordinals.items()
                    },
                    "forced": {choice: dict(counter) for choice, counter in forced.items()},
                    "catch_rows": catch_rows,
                    "comments": [{"row": row, "comment": comment} for row, comment in texts["comment"]],
                    "row_text": {
                        field: [{"row": row, "text": written} for row, written in entries]
                        for field, entries in texts.items()
                    },
                    "note": (
                        "Within-round contrasts only. Absolute scores are not comparable "
                        "across rounds and are not MOS."
                    ),
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
