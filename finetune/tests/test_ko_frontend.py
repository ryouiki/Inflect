from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "g2pkk",
    reason="The Korean frontend needs the 'ko' extra (g2pkk).",
)

from inflect_finetune.frontend import FrontendOptions, process_text  # noqa: E402
from inflect_finetune.frontends import resolve  # noqa: E402
from inflect_finetune.frontends.ko_g2pkk import (  # noqa: E402
    DECLARED_SYMBOLS,
    LEXICON_ENVIRONMENT_VARIABLE,
    KoreanFrontendError,
    KoreanG2pkkFrontend,
    create_frontend,
)


#: 평음 / 경음 / 격음. This contrast is the reason the frontend exists.
LARYNGEAL_PAIRS = [
    ("살", "쌀"),
    ("자다", "짜다"),
    ("불", "뿔"),
    ("불", "풀"),
    ("달", "딸"),
    ("달", "탈"),
    ("공", "콩"),
    ("방", "빵"),
    ("기", "끼"),
    ("기", "키"),
    ("정", "쩡"),
    ("정", "청"),
    ("사", "싸"),
]


def _phonemes(frontend: KoreanG2pkkFrontend, text: str) -> str:
    return frontend.phonemize(frontend.normalize(text))


@pytest.fixture(scope="module")
def frontend() -> KoreanG2pkkFrontend:
    return create_frontend(language="ko")


@pytest.mark.parametrize(("plain", "contrasting"), LARYNGEAL_PAIRS)
def test_the_laryngeal_contrast_survives(
    frontend: KoreanG2pkkFrontend,
    plain: str,
    contrasting: str,
) -> None:
    """eSpeak's Korean voice merges six of these pairs; this frontend must not.

    A merged pair is a phonemic collision the model can never recover from, so
    this is the regression lock that justifies not using eSpeak for Korean.
    """
    assert _phonemes(frontend, plain) != _phonemes(frontend, contrasting)


def test_korean_phonological_rules_are_applied(
    frontend: KoreanG2pkkFrontend,
) -> None:
    """g2pkk supplies the phonology; the mapping must carry it through."""
    # 비음화
    assert _phonemes(frontend, "국물") == "kuŋmul"
    assert _phonemes(frontend, "옵니다") == "omnita"
    # 유음화 — ㄹㄹ is a long lateral, not a flap after a lateral
    assert _phonemes(frontend, "신라면") == "sillamjʌn"
    # 겹받침 단순화
    assert _phonemes(frontend, "닭") == "tak"
    assert _phonemes(frontend, "값") == "kap"
    assert _phonemes(frontend, "여덟") == "jʌtʌl"
    # 구개음화
    assert _phonemes(frontend, "같이") == "katɕʰi"
    assert _phonemes(frontend, "굳이") == "kutɕi"
    # 격음화
    assert _phonemes(frontend, "좋다") == "tɕotʰa"
    assert _phonemes(frontend, "놓고") == "nokʰo"


def test_phonology_is_applied_one_eojeol_at_a_time(
    frontend: KoreanG2pkkFrontend,
) -> None:
    """Given a whole sentence, g2pkk applies liaison across word boundaries.

    It turns 오늘 날씨 into 오늘 랄씨 and 희망을 얘기 into 히망으 럐기 — different
    words, not a different register.
    """
    assert "nalsʼi" in _phonemes(frontend, "오늘 날씨가 좋다")
    assert "ɾalsʼi" not in _phonemes(frontend, "오늘 날씨가 좋다")

    weather = _phonemes(frontend, "희망을 얘기했다")
    assert "himaŋɯl" in weather
    assert _phonemes(frontend, "그녀는 옵니다").startswith("kɯnjʌnɯn")


def test_numbers_are_read_as_words(frontend: KoreanG2pkkFrontend) -> None:
    # A digit-grouping comma is left for g2pkk, which reads it correctly.
    assert frontend.normalize("정가는 3,000원") == "정가는 3,000원"
    assert "samtɕʰʌnwʌn" in _phonemes(frontend, "정가는 3,000원이다")
    # A decimal point is not read, so normalization spells it.
    assert frontend.normalize("무게는 1.5킬로") == "무게는 1점5킬로"
    assert "iltɕʌmo" in _phonemes(frontend, "무게는 1.5킬로")
    assert "itɕʰʌnisipjuŋnjʌn" in _phonemes(frontend, "2026년에 옵니다")
    assert "pʰʌsentʰɯ" in _phonemes(frontend, "30% 할인")


def test_latin_is_refused_even_when_it_leaves_no_trace(
    frontend: KoreanG2pkkFrontend,
) -> None:
    """g2pkk consumes some acronyms and reads them wrong.

    IT becomes the syllable 읻 and AI becomes 아이, with no Latin left behind, so
    checking the output would miss exactly the cases that matter.
    """
    for text in ("AI 기술", "IT 업계", "TV를 봤다", "IoT 장비", "Q&A 시간"):
        with pytest.raises(KoreanFrontendError, match="Latin letters or bare jamo"):
            _phonemes(frontend, text)


def test_bare_jamo_and_unreadable_symbols_are_refused(
    frontend: KoreanG2pkkFrontend,
) -> None:
    with pytest.raises(KoreanFrontendError, match="Latin letters or bare jamo"):
        _phonemes(frontend, "ㅋㅋㅋ 웃겨")
    # ℃ decomposes to °C under NFKC, so the Latin gate catches it.
    with pytest.raises(KoreanFrontendError):
        _phonemes(frontend, "기온 25℃")
    # A symbol g2pkk cannot read reaches the output check instead.
    with pytest.raises(KoreanFrontendError, match="unread"):
        _phonemes(frontend, "이메일@주소")


def test_a_reading_lexicon_resolves_the_refusal() -> None:
    plain = create_frontend(language="ko")
    with_readings = KoreanG2pkkFrontend(
        language="ko", lexicon={"AI": "에이아이", "TV": "티비"}
    )

    assert plain.metadata()["configuration"]["lexicon"] == {}
    assert with_readings.metadata() != plain.metadata()

    assert with_readings.normalize("AI 기술") == "에이아이 기술"
    assert _phonemes(with_readings, "AI 기술").startswith("eiai")
    assert _phonemes(with_readings, "TV를 봤다").startswith("tʰipi")


def test_lexicon_is_loaded_from_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(
        json.dumps({"AI": "에이아이"}, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setenv(LEXICON_ENVIRONMENT_VARIABLE, str(lexicon))
    assert create_frontend(language="ko").normalize("AI") == "에이아이"

    lexicon.write_text("[]", encoding="utf-8")
    with pytest.raises(KoreanFrontendError, match="must be a JSON object"):
        create_frontend(language="ko")


def test_punctuation_and_brackets(frontend: KoreanG2pkkFrontend) -> None:
    assert frontend.normalize("「정말?」이라고 물었다.") == "정말?이라고 물었다."
    assert _phonemes(frontend, "정말? 그래.").endswith(".")
    assert "?" in _phonemes(frontend, "정말? 그래.")
    assert "," in _phonemes(frontend, "네, 알겠습니다.")


def test_output_is_deterministic_and_uses_only_declared_symbols(
    frontend: KoreanG2pkkFrontend,
) -> None:
    declared = set(DECLARED_SYMBOLS)
    samples = [
        "안녕하세요, 오늘 날씨가 참 좋네요.",
        "저는 서울역에서 지하철을 기다리고 있습니다.",
        "값이 없는 걸 밟았다.",
        "국물 좀 드세요. 신라면 맛있어요.",
    ]
    for sample in samples:
        first = _phonemes(frontend, sample)
        second = _phonemes(frontend, sample)
        assert first == second
        assert first
        assert set(first) <= declared


def test_registry_frontend_passes_the_toolkit_frontend_contract() -> None:
    """The bundled frontend must satisfy the same checks as any custom hook."""
    options = resolve("ko-g2pkk", "ko")
    assert isinstance(options, FrontendOptions)
    result = process_text("안녕하세요, 반갑습니다.", options=options)
    assert result.raw_text == "안녕하세요, 반갑습니다."
    assert result.normalized_text == "안녕하세요, 반갑습니다."
    assert set(result.phonemes) <= set(DECLARED_SYMBOLS)
