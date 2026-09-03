"""Japanese and Korean CER plugins for `evaluate --transcript-evaluator`.

The toolkit never downloads an ASR model, and this does not change that: the
model comes from `INFLECT_ASR_MODEL_DIR`, is loaded with `local_files_only`, and
a missing value is an error rather than a fetch. Point it at a local directory
or at a repository already in your Hugging Face cache.

```bash
export INFLECT_ASR_MODEL_DIR=openai/whisper-large-v3-turbo   # already cached
inflect-adapt evaluate \
  --model-dir exports/ja-arona \
  --manifest manifests/eval/ja-val-text.jsonl \
  --transcript-evaluator examples/transcript_evaluator_asr.py:evaluate_japanese \
  --output evaluations/ja-arona
```

CER is compared in the script the language is actually pronounced in, not the
one it is written in. Japanese orthography gives the same reading several
spellings, so both sides are read to kana first; Korean syllables are
decomposed to jamo so a single wrong consonant costs one edit instead of a
whole syllable.

**This is a screen.** It orders candidates and catches gross mispronunciation.
It cannot hear a metallic transient, a flattened contour, or the wrong voice,
and a CER difference inside the noise of the ASR model is not a difference. The
judge is listening.

A normalizer that quietly gives back what it was handed is worse than one that
fails, because the score then looks fine — so the Japanese side raises when its
analyzer leaves kanji in the output rather than scoring the characters.
"""

from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

MODEL_ENVIRONMENT_VARIABLE = "INFLECT_ASR_MODEL_DIR"
LANGUAGE_ENVIRONMENT_VARIABLE = "INFLECT_ASR_LANGUAGE"

_PUNCTUATION = re.compile(
    r"[\s。、．，,.!！?？:：;；…‥・「」『』〈〉《》【】〔〕()\[\]{}<>\"'“”‘’~〜～\-—ー]"
)
_KANJI = re.compile(r"[一-鿿]")
_HIRAGANA_OFFSET = 0x60
_JAMO = re.compile(r"[ᄀ-ᇿㄱ-ㆎ]")


def _model_directory() -> str:
    value = os.environ.get(MODEL_ENVIRONMENT_VARIABLE, "").strip()
    if not value:
        raise RuntimeError(
            f"Set {MODEL_ENVIRONMENT_VARIABLE} to a local ASR model directory or to a "
            "repository already present in your Hugging Face cache. This plugin never "
            "downloads a model: an ASR arriving by surprise during evaluation is a "
            "different measurement than the one you meant to make."
        )
    return value


@lru_cache(maxsize=2)
def _pipeline(language: str) -> Any:
    """Return a cached transformers ASR pipeline restricted to local files."""
    # Resolved first: a missing model is the more useful thing to hear about, and
    # it is the error the caller can fix without installing anything.
    source = _model_directory()
    try:
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )
    except ImportError as exc:  # pragma: no cover - depends on the caller's environment
        raise RuntimeError(
            "This plugin needs transformers installed alongside the toolkit. It is "
            "deliberately not a toolkit dependency: the adaptation path does not "
            "require an ASR model."
        ) from exc

    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(source, local_files_only=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        generate_kwargs={"language": language, "task": "transcribe"},
    )


def to_katakana(text: str) -> str:
    """Return the text with hiragana folded to katakana."""
    return "".join(
        chr(ord(character) + _HIRAGANA_OFFSET) if 0x3041 <= ord(character) <= 0x3096 else character
        for character in text
    )


def normalize_japanese(text: str) -> str:
    """Return the kana reading, with punctuation and length marks removed.

    Raises when the analyzer leaves kanji behind. A normalizer that returns its
    input unchanged scores the spelling instead of the pronunciation, and the
    resulting number looks perfectly reasonable.
    """
    try:
        import pyopenjtalk
    except ImportError as exc:
        raise RuntimeError(
            "Japanese CER needs the 'ja' extra: pip install \".[ja]\""
        ) from exc
    stripped = unicodedata.normalize("NFKC", text).strip()
    if not stripped:
        return ""
    reading = str(pyopenjtalk.g2p(stripped, kana=True))
    reading = _PUNCTUATION.sub("", to_katakana(reading))
    if _KANJI.search(reading):
        raise RuntimeError(
            "The Japanese analyzer returned kanji instead of a reading for "
            f"{text!r}: {reading!r}. Scoring this would compare spellings, not "
            "pronunciations."
        )
    return reading


def normalize_korean(text: str) -> str:
    """Return the jamo decomposition, with punctuation removed.

    Syllable-level comparison charges a whole syllable for one wrong consonant,
    which flattens exactly the laryngeal contrast this language cares about.
    """
    stripped = unicodedata.normalize("NFKC", text).strip()
    decomposed = unicodedata.normalize("NFD", _PUNCTUATION.sub("", stripped))
    return "".join(character for character in decomposed if _JAMO.match(character))


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Return the edit distance from reference to hypothesis, over its length."""
    if not reference:
        raise ValueError("Reference text is empty after normalization.")
    if reference == hypothesis:
        return 0.0
    previous = list(range(len(hypothesis) + 1))
    for index, expected in enumerate(reference, start=1):
        current = [index]
        for position, produced in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[position] + 1,
                    current[position - 1] + 1,
                    previous[position - 1] + (expected != produced),
                )
            )
        previous = current
    return previous[-1] / len(reference)


def _evaluate(
    audio_path: Path,
    reference_text: str,
    sample_rate: int,
    *,
    language: str,
    normalize: Any,
    normalization_name: str,
) -> dict[str, Any]:
    transcript = str(_pipeline(language)(str(audio_path))["text"]).strip()
    reference = normalize(reference_text)
    hypothesis = normalize(transcript)
    return {
        "transcript": transcript,
        "cer": character_error_rate(reference, hypothesis),
        "reference_normalized": reference,
        "hypothesis_normalized": hypothesis,
        "normalization": normalization_name,
        "language": language,
        "backend": _model_directory(),
        "sample_rate": sample_rate,
        "note": "screen only; not a pass bar and not comparable across ASR models",
    }


def evaluate_japanese(
    audio_path: Path,
    reference_text: str,
    sample_rate: int,
) -> dict[str, Any]:
    """Kana-normalized CER against a local ASR model."""
    return _evaluate(
        audio_path,
        reference_text,
        sample_rate,
        language=os.environ.get(LANGUAGE_ENVIRONMENT_VARIABLE, "japanese"),
        normalize=normalize_japanese,
        normalization_name="katakana",
    )


def evaluate_korean(
    audio_path: Path,
    reference_text: str,
    sample_rate: int,
) -> dict[str, Any]:
    """Jamo-normalized CER against a local ASR model."""
    return _evaluate(
        audio_path,
        reference_text,
        sample_rate,
        language=os.environ.get(LANGUAGE_ENVIRONMENT_VARIABLE, "korean"),
        normalize=normalize_korean,
        normalization_name="jamo",
    )
