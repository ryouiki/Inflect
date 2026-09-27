from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "transcript_evaluator_asr.py"


@pytest.fixture(scope="module")
def plugin():
    spec = importlib.util.spec_from_file_location("_transcript_evaluator_asr", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_contract_matches_what_evaluate_calls(plugin):
    import inspect

    from inflect_finetune.evaluation import TranscriptEvaluator  # noqa: F401

    for entry in (plugin.evaluate_japanese, plugin.evaluate_korean):
        parameters = list(inspect.signature(entry).parameters)
        assert parameters == ["audio_path", "reference_text", "sample_rate"]


def test_a_missing_model_directory_is_an_error_not_a_download(plugin, monkeypatch, tmp_path):
    monkeypatch.delenv(plugin.MODEL_ENVIRONMENT_VARIABLE, raising=False)
    with pytest.raises(RuntimeError, match="never"):
        plugin.evaluate_japanese(tmp_path / "clip.wav", "テスト", 24_000)


def test_identical_text_scores_zero_and_a_substitution_costs_one_edit(plugin):
    assert plugin.character_error_rate("アイウエオ", "アイウエオ") == 0.0
    assert plugin.character_error_rate("アイウエオ", "アイウエカ") == pytest.approx(0.2)
    assert plugin.character_error_rate("アイ", "") == pytest.approx(1.0)


def test_an_empty_reference_is_refused(plugin):
    with pytest.raises(ValueError):
        plugin.character_error_rate("", "アイ")


def test_korean_normalization_decomposes_to_jamo(plugin):
    jamo = plugin.normalize_korean("한글, 좋다!")
    assert jamo and not any("가" <= character <= "힣" for character in jamo)
    # One wrong onset costs one edit out of the jamo count, not a whole syllable.
    tense = plugin.normalize_korean("쌀")
    plain = plugin.normalize_korean("살")
    assert tense != plain
    assert plugin.character_error_rate(tense, plain) < 1.0


def test_korean_normalization_keeps_the_laryngeal_contrast_visible(plugin):
    pairs = [("살", "쌀"), ("불", "뿔"), ("자다", "짜다"), ("방", "빵")]
    for plain, tense in pairs:
        assert plugin.normalize_korean(plain) != plugin.normalize_korean(tense)


def test_japanese_normalization_reads_kanji_to_kana(plugin):
    pytest.importorskip("pyopenjtalk")
    reading = plugin.normalize_japanese("水を買わなくてはならないのです。")
    assert reading
    assert not any("一" <= character <= "鿿" for character in reading)
    assert all(0x30A1 <= ord(character) <= 0x30FA for character in reading), reading


def test_japanese_normalization_folds_spelling_differences(plugin):
    """The reason CER is compared on readings rather than on the writing."""
    pytest.importorskip("pyopenjtalk")
    assert plugin.normalize_japanese("わたしは") == plugin.normalize_japanese("私は")


def test_japanese_normalization_drops_punctuation_and_length_marks(plugin):
    pytest.importorskip("pyopenjtalk")
    assert "、" not in plugin.normalize_japanese("はい、そうです。")
    assert "ー" not in plugin.normalize_japanese("コーヒー")


def test_an_analyzer_that_returns_kanji_raises_instead_of_scoring_it(plugin, monkeypatch):
    """The silent-failure guard.

    An analyzer whose dictionary is missing has been observed to hand back the
    text it was given. Scoring that compares spellings while looking like a
    pronunciation score, so it has to fail loudly instead.
    """
    pyopenjtalk = pytest.importorskip("pyopenjtalk")
    monkeypatch.setattr(pyopenjtalk, "g2p", lambda text, kana=False: text)
    with pytest.raises(RuntimeError, match="kanji"):
        plugin.normalize_japanese("水を買う")
