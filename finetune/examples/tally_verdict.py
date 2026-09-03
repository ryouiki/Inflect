"""Join a listening verdict to its sealed mapping and tally by system.

A letter is a per-row label, so counting letters counts noise. The arm lives in
`mapping.json`, which is why scoring and tallying are separate steps and this
script is the only place the two meet.

What it reports, and what it refuses to report:

* Per system: the distribution of each descriptive axis, as counts and a median
  of the option index. Never a mean — the axes are ordinal.
* The forced choices, as counts per system.
* The catch pair, if the page carried one. Two scores for byte-identical audio
  are this round's noise floor; read every other contrast against it.
* Free text, grouped by system, because that column has repeatedly carried the
  defect no axis was watching.

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

AXES = ("quality", "defect", "language")
FORCED = ("most_natural", "most_blurred")


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

    scores: dict[str, dict[str, collections.Counter]] = collections.defaultdict(
        lambda: {axis: collections.Counter() for axis in AXES}
    )
    ordinals: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: {axis: [] for axis in AXES}
    )
    forced: dict[str, collections.Counter] = {name: collections.Counter() for name in FORCED}
    comments: list[tuple[str, str]] = []
    unanswered = 0

    for row, letters in rows.items():
        scored = verdict.get("rows", {}).get(row)
        if not scored:
            continue
        for letter, name in letters.items():
            track = scored.get("tracks", {}).get(letter, {})
            for axis in AXES:
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
        comment = (scored.get("comment") or "").strip()
        if comment:
            comments.append((row, comment))

    print(f"page {mapping.get('page_key')} · rows scored {len(verdict.get('rows', {}))}/{len(rows)}")
    print(f"unanswered axis cells: {unanswered}")
    print()
    for name in sorted(scores):
        print(f"[{name}]")
        for axis in AXES:
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
                    f"  noise floor for this round: {spread} option step(s) on byte-identical audio"
                )

    if comments:
        print()
        print("free text")
        for row, comment in comments:
            print(f"  {row}: {comment}")

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
                    "comments": [{"row": row, "comment": comment} for row, comment in comments],
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
