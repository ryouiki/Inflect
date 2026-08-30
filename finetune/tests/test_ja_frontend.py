from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "pyopenjtalk",
    reason="The Japanese frontend needs the 'ja' extra (pyopenjtalk-plus).",
)

from inflect_finetune.frontend import FrontendOptions, process_text  # noqa: E402
from inflect_finetune.frontends import resolve  # noqa: E402
from inflect_finetune.frontends.ja_openjtalk import (  # noqa: E402
    ACCENT_FALL,
    ACCENT_RISE,
    DECLARED_SYMBOLS,
    LEXICON_ENVIRONMENT_VARIABLE,
    JapaneseOpenJTalkFrontend,
    create_frontend,
)


def _phonemes(frontend: JapaneseOpenJTalkFrontend, text: str) -> str:
    return frontend.phonemize(frontend.normalize(text))


def _bare(phonemes: str) -> str:
    """Drop accent and boundary marks, leaving the reading."""
    return "".join(
        character
        for character in phonemes
        if character not in {ACCENT_RISE, ACCENT_FALL, " "}
    )


@pytest.fixture(scope="module")
def frontend() -> JapaneseOpenJTalkFrontend:
    return create_frontend(language="ja")


def test_kanji_readings_are_resolved(frontend: JapaneseOpenJTalkFrontend) -> None:
    """Lock the readings that a text-only frontend cannot produce.

    These are checked without accent marks so the assertion survives a
    dictionary revision that moves an accent nucleus.
    """
    reading = _bare(_phonemes(frontend, "抗うつ剤の対策について、痛み止め薬を飲みました。"))
    assert "kooɯtsɯzai" in reading
    assert "taisakɯ" in reading
    assert "itamidomejakɯ" in reading


def test_numbers_and_dates_are_read_as_words(frontend: JapaneseOpenJTalkFrontend) -> None:
    reading = _bare(_phonemes(frontend, "彼女は2026年8月30日に来ます。"))
    assert "niseɴnidʑɯɯɾokɯneɴ" in reading
    assert "hatɕiɡatsɯ" in reading
    assert "saɴdʑɯɯnitɕi" in reading


def test_pitch_accent_distinguishes_minimal_pairs(
    frontend: JapaneseOpenJTalkFrontend,
) -> None:
    """はし and あめ differ only by accent; the marks must carry that."""
    chopsticks = _phonemes(frontend, "箸")
    bridge = _phonemes(frontend, "橋")
    assert _bare(chopsticks) == _bare(bridge)
    assert chopsticks != bridge
    assert chopsticks.startswith(f"ha{ACCENT_FALL}")
    assert bridge.startswith(f"ha{ACCENT_RISE}")

    rain = _phonemes(frontend, "雨")
    candy = _phonemes(frontend, "飴")
    assert _bare(rain) == _bare(candy)
    assert rain != candy


def test_accent_marks_are_not_placed_inside_a_mora(
    frontend: JapaneseOpenJTalkFrontend,
) -> None:
    """A one-mora accent phrase must not be split between its consonant and vowel."""
    phonemes = _phonemes(frontend, "「本当に？」と彼は言った。")
    assert " to " in phonemes
    assert "t o" not in phonemes


def test_punctuation_survives_open_jtalk(frontend: JapaneseOpenJTalkFrontend) -> None:
    """Open JTalk collapses every mark into 'pau'; the writer's punctuation is kept."""
    assert _phonemes(frontend, "こんにちは。元気ですか？").endswith("?")
    assert "," in _phonemes(frontend, "えっと、そうですね。")
    assert "." in _phonemes(frontend, "はい。")
    assert _phonemes(frontend, "本当に！").endswith("!")


def test_brackets_are_removed_during_normalization(
    frontend: JapaneseOpenJTalkFrontend,
) -> None:
    normalized = frontend.normalize("「本当に？」と彼は言った。")
    assert "「" not in normalized and "」" not in normalized
    assert normalized == "本当に?と彼は言った。"


def test_output_is_deterministic_and_uses_only_declared_symbols(
    frontend: JapaneseOpenJTalkFrontend,
) -> None:
    declared = set(DECLARED_SYMBOLS)
    samples = [
        "こんにちは、今日はいい天気ですね。",
        "アロナです！よろしくお願いします。",
        "ジュース、コーヒー・お茶",
        "彼女は2026年8月30日に来ます。",
    ]
    for sample in samples:
        first = _phonemes(frontend, sample)
        second = _phonemes(frontend, sample)
        assert first == second
        assert first
        assert set(first) <= declared


def test_reading_lexicon_rewrites_text_and_changes_the_metadata_hash() -> None:
    plain = create_frontend(language="ja")
    overridden = JapaneseOpenJTalkFrontend(language="ja", lexicon={"鷹神": "たかかみ"})

    assert plain.metadata()["configuration"]["lexicon"] == {}
    assert overridden.metadata()["configuration"]["lexicon"] == {"鷹神": "たかかみ"}
    assert plain.metadata() != overridden.metadata()

    assert overridden.normalize("鷹神です。") == "たかかみです。"
    assert _bare(_phonemes(overridden, "鷹神です。")).startswith("takakami")


def test_lexicon_is_loaded_from_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(
        json.dumps({"鷹神": "たかかみ"}, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setenv(LEXICON_ENVIRONMENT_VARIABLE, str(lexicon))
    assert create_frontend(language="ja").normalize("鷹神") == "たかかみ"

    lexicon.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be a JSON object"):
        create_frontend(language="ja")


def test_registry_frontend_passes_the_toolkit_frontend_contract() -> None:
    """The bundled frontend must satisfy the same checks as any custom hook."""
    options = resolve("ja-openjtalk", "ja")
    assert isinstance(options, FrontendOptions)
    result = process_text("こんにちは、世界。", options=options)
    assert result.raw_text == "こんにちは、世界。"
    assert result.normalized_text == "こんにちは、世界。"
    assert set(result.phonemes) <= set(DECLARED_SYMBOLS)


def test_digit_grouping_commas_are_removed_before_reading(
    frontend: JapaneseOpenJTalkFrontend,
) -> None:
    """Open JTalk reads '3,000' as four separate digits unless the comma goes."""
    assert frontend.normalize("定価は3,000円でした。") == "定価は3000円でした。"
    assert frontend.normalize("1,234,567円") == "1234567円"
    assert "saɴzeɴ" in _bare(_phonemes(frontend, "定価は3,000円でした。"))

    # A comma that is not a digit separator still separates chunks.
    assert frontend.normalize("はい,そうです") == "はい,そうです"
    assert "," in _phonemes(frontend, "えっと、そうですね。")


def test_a_decimal_point_does_not_end_a_sentence(
    frontend: JapaneseOpenJTalkFrontend,
) -> None:
    reading = _bare(_phonemes(frontend, "ここから駅まで1.5キロあります。"))
    assert "iʔteɴɡo" in reading
    assert "." not in reading[:-1]

    # A period that is not inside a number still terminates a chunk.
    phonemes = _phonemes(frontend, "終わりました。次へ。")
    assert phonemes.count(".") == 2
    assert _phonemes(frontend, "第1.").endswith(".")
