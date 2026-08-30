"""Dump normalized text and phonemes for fluent-speaker review.

A frontend can be deterministic, declare every symbol it emits, and add no new
embedding rows while still encoding the wrong pronunciation. Only a fluent
speaker can tell you that, and they cannot do it from IPA alone, so this writes
the reading beside the symbol stream.

It phonemizes through the same path `prepare` uses, so what a reviewer sees is
what training would receive.

```
python examples/frontend_review_dump.py \
  --sentences examples/japanese_review_suite.txt \
  --frontend ja-openjtalk \
  --language ja \
  --output review/ja.tsv
```

Review the bundled suite *and* a random sample of your own transcripts: the
suite covers frontend behavior, not your corpus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator

from inflect_finetune.frontend import FrontendError, process_text
from inflect_finetune.frontends import resolve
from inflect_finetune.symbols import BASE_SYMBOLS


COLUMNS = ("index", "source", "normalized", "reading", "phonemes", "note")


def read_sentences(path: Path) -> list[str]:
    """Return non-empty, non-comment lines in file order."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        stripped
        for line in lines
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]


def reading_lookup(language: str) -> Any:
    """Return a kana reader for Japanese, or None when one is unavailable."""
    if not (language == "ja" or language.startswith("ja-")):
        return None
    try:
        import pyopenjtalk
    except ImportError:
        return None

    def read(text: str) -> str:
        return str(pyopenjtalk.g2p(text, kana=True))

    return read


def review_rows(
    sentences: list[str],
    frontend: str,
    language: str,
) -> Iterator[dict[str, str]]:
    """Yield one review row per sentence, recording failures instead of raising."""
    options = resolve(frontend, language)
    read = reading_lookup(language)
    inventory = set(BASE_SYMBOLS)
    for index, sentence in enumerate(sentences, start=1):
        row = {"index": str(index), "source": sentence, "note": ""}
        try:
            result = process_text(sentence, options=options)
        except (FrontendError, ValueError, RuntimeError) as exc:
            row.update(normalized="", reading="", phonemes="", note=f"FAILED: {exc}")
            yield row
            continue
        row["normalized"] = result.normalized_text
        row["phonemes"] = result.phonemes
        try:
            row["reading"] = read(result.normalized_text) if read else ""
        except Exception as exc:  # noqa: BLE001 - reviewer aid, never fatal
            row["reading"] = ""
            row["note"] = f"reading unavailable: {exc}"
        outside = sorted(set(result.phonemes) - inventory)
        if outside:
            note = "outside the released inventory: " + " ".join(outside)
            row["note"] = f"{row['note']}; {note}" if row["note"] else note
        yield row


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a tab-separated file with literal tabs and newlines removed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            handle.write(
                "\t".join(
                    " ".join(str(row.get(column, "")).split()) for column in COLUMNS
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--frontend", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    sentences = read_sentences(args.sentences)
    if args.limit is not None:
        sentences = sentences[: args.limit]
    if not sentences:
        print(f"error: {args.sentences} contains no sentences", file=sys.stderr)
        return 1

    rows = list(review_rows(sentences, args.frontend, args.language))
    write_tsv(args.output, rows)

    failures = [row for row in rows if row["note"].startswith("FAILED")]
    flagged = [row for row in rows if row["note"] and not row["note"].startswith("FAILED")]
    print(f"sentences: {len(rows)}")
    print(f"failed:    {len(failures)}")
    print(f"flagged:   {len(flagged)}")
    print(f"written:   {args.output}")
    for row in failures[:10]:
        print(f"  {row['index']}: {row['source']} -> {row['note']}", file=sys.stderr)
    print(
        "\nA clean run is not a passing review. Have a fluent speaker read the "
        "'reading' column against 'source' and record the misreading rate."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
