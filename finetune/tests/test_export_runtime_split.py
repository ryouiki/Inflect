"""The sentence splitter that ships inside every exported package.

It runs before the language frontend, per chunk, so a boundary it invents is a
boundary the frontend can no longer undo. The Korean frontend reads "1.5" as
"일점오" correctly, and the Japanese one likewise, but only if the decimal
still reaches it in one piece. Each chunk is also synthesized separately, with
its own fade and its own noise seed, and joined with a pause sized by the mark
it ends on, so a spurious split is audible rather than merely untidy.

The function is exercised from the source string the exporter materialises,
which is the artifact that actually ships. Doing it that way needs no torch and
no export.
"""

from __future__ import annotations

import re
import textwrap

import pytest

from inflect_finetune.exporting import _INFERENCE_SOURCE


def load_split_text():
    """Execute just the splitter out of the runtime source the exporter writes."""

    namespace: dict[str, object] = {"re": re}
    source = textwrap.dedent(_INFERENCE_SOURCE)
    start = source.index("_SENTENCE_BOUNDARY = re.compile")
    end = source.index("def boundary_pause_seconds")
    exec(source[start:end], namespace)  # noqa: S102
    return namespace["split_text"]


def load_boundary_pause():
    namespace: dict[str, object] = {"re": re}
    source = textwrap.dedent(_INFERENCE_SOURCE)
    start = source.index("def boundary_pause_seconds")
    end = source.index("def edge_fade")
    exec(source[start:end], namespace)  # noqa: S102
    return namespace["boundary_pause_seconds"]


@pytest.fixture(scope="module")
def split_text():
    return load_split_text()


@pytest.mark.parametrize(
    "text",
    [
        "1.5초 남았습니다.",
        "3.14는 파이입니다.",
        "It costs 1.5 dollars.",
        "1.5秒待ってください。",
    ],
    ids=["korean-decimal", "korean-pi", "english-decimal", "japanese-decimal"],
)
def test_a_decimal_point_does_not_end_a_sentence(split_text, text: str) -> None:
    """Split here and the listener hears "one." then a pause then "five"."""

    assert split_text(text) == [text]


def test_a_version_number_survives_intact(split_text) -> None:
    assert split_text("버전 2.0.1 입니다.") == ["버전 2.0.1 입니다."]


def test_a_clock_time_does_not_end_a_sentence(split_text) -> None:
    """The colon is in the same position as the point, and had the same bug."""

    assert split_text("10:30에 만나요.") == ["10:30에 만나요."]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("終わり。2秒です。", ["終わり。", "2秒です。"]),
        ("네!3시에 만나요.", ["네!", "3시에 만나요."]),
        ("終わり！3回です。", ["終わり！", "3回です。"]),
        ("第1. 次へ。", ["第1.", "次へ。"]),
    ],
    ids=["japanese-period", "korean-bang", "japanese-bang", "ordered-list"],
)
def test_a_sentence_ending_before_a_digit_still_splits(
    split_text, text: str, expected: list[str]
) -> None:
    """The guard has to be narrow.

    Refusing to split before any digit is the obvious fix and it swallows the
    next sentence whenever that sentence opens with a number.
    """

    assert split_text(text) == expected


def test_a_japanese_period_without_a_space_still_splits(split_text) -> None:
    """Why the separator is optional in the first place.

    A CJK full stop carries no trailing space, so requiring whitespace after
    the mark collapses a paragraph into one chunk.
    """

    assert split_text("今日はいい天気です。明日も晴れるでしょう。それでは。") == [
        "今日はいい天気です。",
        "明日も晴れるでしょう。",
        "それでは。",
    ]
    assert split_text("끝났습니다.다음으로.") == ["끝났습니다.", "다음으로."]


def test_a_run_of_marks_stays_with_the_chunk_it_ends(split_text) -> None:
    """An ellipsis used to become chunks of bare dots, which have no phonemes."""

    assert split_text("잠깐...기다려요.") == ["잠깐...", "기다려요."]
    assert split_text("정말?! 그래요.") == ["정말?!", "그래요."]


def test_a_long_sentence_still_splits_on_length(split_text) -> None:
    """The length fallback is the function's original purpose."""

    text = "가" * 200 + " 그리고 " + "나" * 80 + " 끝입니다"
    chunks = split_text(text)
    assert len(chunks) == 2
    assert all(len(chunk) <= 280 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_a_digit_grouping_comma_is_not_a_length_split_point(split_text) -> None:
    """Defensive: the same class of defect one layer down.

    The length fallback prefers a comma, and a thousands separator is a comma.
    I could not construct an input where the unguarded version actually split
    there, because the space search usually wins first, so this pins the
    intent rather than a reproduced failure.
    """

    text = "가" * 289 + " 1,500원이 필요합니다"
    chunks = split_text(text)
    assert not any(chunk.endswith(",") for chunk in chunks)
    assert any("1,500" in chunk for chunk in chunks)


def test_short_and_degenerate_inputs_do_not_crash(split_text) -> None:
    assert split_text("안녕") == ["안녕"]
    assert split_text("") == []
    assert split_text("   ") == []


def test_the_pause_after_a_chunk_follows_its_final_mark() -> None:
    """A spurious "1." chunk drew the full sentence-final pause, the longest one."""

    boundary_pause_seconds = load_boundary_pause()
    assert boundary_pause_seconds("끝났습니다.") == pytest.approx(0.22)
    assert boundary_pause_seconds("정말?") == pytest.approx(0.28)
    assert boundary_pause_seconds("그리고") == pytest.approx(0.08)


def test_the_shipped_source_carries_the_guarded_pattern() -> None:
    """The exporter writes this source into each package, so the guard must be in it."""

    assert "_SENTENCE_BOUNDARY" in _INFERENCE_SOURCE
    assert r"(?!(?<=[0-9][.:])[0-9])" in _INFERENCE_SOURCE
