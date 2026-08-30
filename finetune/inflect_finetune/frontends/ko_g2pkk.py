"""Korean frontend built on g2pkk pronunciation output.

g2pkk applies Korean phonology — nasalization, lateralization, tensification,
coda neutralization, cluster simplification, and number reading — and returns
the result as Hangul. Because Hangul is featural, converting that surface
pronunciation to phonemes is a mechanical syllable decomposition.

eSpeak is deliberately not part of this chain. Its Korean voice collapses the
three-way laryngeal contrast: 살/쌀, 자다/짜다, 불/뿔, 방/빵, 정/쩡, and 사/싸
each come back as one phoneme string. That is a phonemic merger, not a nuance,
and no amount of training data can recover a distinction the frontend erased.

Every phoneme maps into the published Inflect v2 symbol inventory, so a Korean
dataset prepared with this frontend adds no new embedding rows.

Requires the ``ko`` extra:

```
python -m pip install ".[ko]"
```

The MeCab dictionary g2pkk depends on is a third-party artifact under its own
license. A package exported with this frontend is not self-contained.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping


FRONTEND_NAME = "ko-g2pkk"
FRONTEND_VERSION = "1"
IPA_MAPPING_VERSION = "1"
LEXICON_ENVIRONMENT_VARIABLE = "INFLECT_KO_LEXICON"

_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3
_NUCLEUS_COUNT = 21
_CODA_COUNT = 28

# Tense consonants take the modifier apostrophe and aspirated ones the modifier
# h. Both are in the released inventory, so the three-way contrast costs no new
# symbols.
ONSET_TO_IPA: tuple[str, ...] = (
    "k", "kʼ", "n", "t", "tʼ", "ɾ", "m", "p", "pʼ", "s", "sʼ",
    "",  # ㅇ is silent in onset position
    "tɕ", "tɕʼ", "tɕʰ", "kʰ", "tʰ", "pʰ", "h",
)

# ㅐ/ㅔ and ㅙ/ㅚ/ㅞ have merged for most modern speakers. ㅐ and ㅔ are kept
# apart because a merger cannot be undone later, while keeping them apart costs
# nothing if the speaker does merge them.
NUCLEUS_TO_IPA: tuple[str, ...] = (
    "a", "ɛ", "ja", "jɛ", "ʌ", "e", "jʌ", "je", "o", "wa", "wɛ",
    "we", "jo", "u", "wʌ", "we", "wi", "ju", "ɯ", "ɯi", "i",
)

# Coda position neutralizes to seven phonemes. g2pkk has normally applied this
# already; the full table keeps the mapping correct if it has not.
CODA_TO_IPA: tuple[str, ...] = (
    "", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l",
    "p", "l", "m", "p", "p", "t", "t", "ŋ", "t", "t", "k", "t", "p", "t",
)

PUNCTUATION_MAP: dict[str, str] = {
    ",": ",",
    "·": ",",
    ".": ".",
    "!": "!",
    "?": "?",
    ":": ":",
    ";": ";",
    "…": "…",
    "‥": "…",
}

# Brackets and quotation marks carry no reading.
DROPPED_CHARACTERS = frozenset(
    "「」『』〈〉《》【】〔〕"
    "()[]{}<>\"'“”‘’"
)

# Latin letters and bare jamo are the silent-error class: g2pkk turns IT into
# 읻 and AI into 아이 without leaving anything behind to detect, so they are
# rejected on the way in rather than looked for on the way out.
_UNREADABLE_INPUT = re.compile(r"[A-Za-zᄀ-ᇿ㄰-㆏ꥠ-꥿]")
# g2pkk reads a digit-grouping comma correctly (3,000 -> 삼천) but leaves a
# decimal point alone, giving "일.오" instead of "일점오".
_DECIMAL_POINT = re.compile(r"(?<=[0-9])\.(?=[0-9])")
_NUMERIC_MARKS = ".,"
_SENTENCE_MARKS = "".join(sorted(set(PUNCTUATION_MAP) - set(_NUMERIC_MARKS)))
_SPLIT_PATTERN = re.compile(
    "("
    f"(?<![0-9])[{re.escape(_NUMERIC_MARKS)}]"
    f"|[{re.escape(_NUMERIC_MARKS)}](?![0-9])"
    f"|[{re.escape(_SENTENCE_MARKS)}]"
    ")"
)


class KoreanFrontendError(RuntimeError):
    """Raised when Korean text cannot be converted into Inflect symbols."""


def _declared_symbols() -> tuple[str, ...]:
    characters: set[str] = set()
    for table in (ONSET_TO_IPA, NUCLEUS_TO_IPA, CODA_TO_IPA):
        for phonemes in table:
            characters.update(phonemes)
    characters.update(PUNCTUATION_MAP.values())
    characters.add(" ")
    return tuple(sorted(characters))


DECLARED_SYMBOLS: tuple[str, ...] = _declared_symbols()


def _load_environment_lexicon() -> dict[str, str]:
    location = os.environ.get(LEXICON_ENVIRONMENT_VARIABLE)
    if not location:
        return {}
    path = Path(location).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KoreanFrontendError(
            f"Could not read the Korean reading lexicon at {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and key and value
        for key, value in payload.items()
    ):
        raise KoreanFrontendError(
            f"The Korean reading lexicon at {path} must be a JSON object of "
            "non-empty surface/reading strings."
        )
    return {
        unicodedata.normalize("NFKC", key): unicodedata.normalize("NFKC", value)
        for key, value in payload.items()
    }


class KoreanG2pkkFrontend:
    """Deterministic Korean normalization and phonemization."""

    def __init__(
        self,
        language: str = "ko",
        lexicon: Mapping[str, str] | None = None,
    ) -> None:
        self.language = language
        self._lexicon = dict(lexicon) if lexicon is not None else _load_environment_lexicon()
        self._lexicon_pattern = self._build_lexicon_pattern()
        self._engine: Any | None = None
        # Load here so a missing dependency stops preparation during frontend
        # validation instead of part-way through a corpus. The module itself
        # stays importable without g2pkk so the symbol tables can be inspected.
        self._g2p()

    def _build_lexicon_pattern(self) -> re.Pattern[str] | None:
        if not self._lexicon:
            return None
        # Longest surface first so a longer entry is never shadowed by a
        # shorter one that happens to be a prefix.
        ordered = sorted(self._lexicon, key=lambda item: (-len(item), item))
        return re.compile("|".join(re.escape(surface) for surface in ordered))

    def _g2p(self) -> Any:
        if self._engine is None:
            try:
                from g2pkk import G2p
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise KoreanFrontendError(
                    "The Korean frontend requires g2pkk. Install the toolkit's "
                    "'ko' extra."
                ) from exc
            self._engine = G2p()
        return self._engine

    def normalize(self, text: str) -> str:
        """Return NFKC text with unreadable marks removed and readings applied."""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = "".join(
            character
            for character in normalized
            if character not in DROPPED_CHARACTERS
        )
        normalized = _DECIMAL_POINT.sub("점", normalized)
        if self._lexicon_pattern is not None:
            normalized = self._lexicon_pattern.sub(
                lambda match: self._lexicon[match.group(0)], normalized
            )
        return re.sub(r"\s+", " ", normalized).strip()

    def phonemize(self, text: str) -> str:
        """Return the Inflect symbol stream for already-normalized text."""
        self._reject_unreadable_input(text)
        parts: list[str] = []
        for piece in _SPLIT_PATTERN.split(text):
            if piece in PUNCTUATION_MAP:
                # Punctuation before any speech has nothing to attach to.
                if parts:
                    parts[-1] += PUNCTUATION_MAP[piece]
                continue
            chunk = piece.strip()
            if not chunk:
                continue
            phonemes = self._phonemize_chunk(chunk)
            if phonemes:
                parts.append(phonemes)
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    def _reject_unreadable_input(self, text: str) -> None:
        found = sorted(set(_UNREADABLE_INPUT.findall(text)))
        if not found:
            return
        raise KoreanFrontendError(
            "Normalized Korean text still contains Latin letters or bare jamo: "
            + ", ".join(repr(character) for character in found)
            + ". g2pkk reads some of these incorrectly without leaving a trace "
            "(IT becomes 읻, AI becomes 아이), so they are refused rather than "
            f"guessed. Supply readings through {LEXICON_ENVIRONMENT_VARIABLE} "
            "or write them in Hangul. Note that normalization is applied first, "
            "so a compatibility character may be reported in its decomposed "
            "form (℃ as °C, ㅋ as ᄏ)."
        )

    def _phonemize_chunk(self, chunk: str) -> str:
        engine = self._g2p()
        pieces: list[str] = []
        # One eojeol at a time. Given a whole sentence, g2pkk applies liaison
        # across word boundaries and produces different words: 오늘 날씨 becomes
        # 오늘 랄씨 and 희망을 얘기 becomes 히망으 럐기.
        for word in chunk.split():
            try:
                reading = engine(word)
            except Exception as exc:  # pragma: no cover - engine dependent
                raise KoreanFrontendError(
                    f"g2pkk failed on {word!r}: {exc}"
                ) from exc
            transcribed = self._transcribe(reading, word)
            if transcribed:
                pieces.append(transcribed)
        return " ".join(pieces)

    def _transcribe(self, reading: str, source: str) -> str:
        out: list[str] = []
        previous_coda = ""
        for character in reading:
            index = ord(character) - _HANGUL_BASE
            if not 0 <= index <= _HANGUL_LAST - _HANGUL_BASE:
                raise KoreanFrontendError(
                    f"g2pkk left {character!r} unread in {source!r} (pronunciation "
                    f"{reading!r}). Write it in Hangul or supply a reading through "
                    f"{LEXICON_ENVIRONMENT_VARIABLE}."
                )
            onset = ONSET_TO_IPA[index // (_NUCLEUS_COUNT * _CODA_COUNT)]
            nucleus = NUCLEUS_TO_IPA[(index % (_NUCLEUS_COUNT * _CODA_COUNT)) // _CODA_COUNT]
            coda = CODA_TO_IPA[index % _CODA_COUNT]
            # ㄹㄹ is a long lateral, not a flap after a lateral.
            if onset == "ɾ" and previous_coda == "l":
                onset = "l"
            out.append(onset + nucleus + coda)
            previous_coda = coda
        return "".join(out)

    def symbols(self) -> tuple[str, ...]:
        """Return every character this frontend can emit."""
        return DECLARED_SYMBOLS

    def metadata(self) -> dict[str, Any]:
        """Return the hashed reproducibility record for this configuration."""
        return {
            "name": FRONTEND_NAME,
            "version": FRONTEND_VERSION,
            "language": self.language,
            "configuration": {
                "engine": "g2pkk",
                "ipa_mapping_version": IPA_MAPPING_VERSION,
                "phonology_unit": "one eojeol at a time",
                "tense_marker": "ʼ",
                "aspirated_marker": "ʰ",
                "vowels": "ㅐ and ㅔ kept apart; ㅚ and ㅞ both /we/",
                "grouping_comma": "left for g2pkk to read",
                "decimal_point": "rewritten as 점",
                "latin_and_jamo": "refused; supply readings through the lexicon",
                "punctuation_map": dict(sorted(PUNCTUATION_MAP.items())),
                "dropped_characters": "".join(sorted(DROPPED_CHARACTERS)),
                "lexicon": dict(sorted(self._lexicon.items())),
                "declared_symbol_count": len(DECLARED_SYMBOLS),
            },
        }


def create_frontend(*, language: str = "ko") -> KoreanG2pkkFrontend:
    """Return the Korean frontend used by the ``ko-g2pkk`` registry entry."""
    return KoreanG2pkkFrontend(language=language)
