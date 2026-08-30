"""Japanese frontend built on Open JTalk full-context labels.

Open JTalk supplies the reading, the phoneme sequence, and the accent fields
that this frontend converts into the character stream Inflect consumes. eSpeak
is not usable for Japanese: it emits the literal English words "chinese letter"
for every kanji, so kanji text is lost entirely.

Every phoneme maps into the published Inflect v2 symbol inventory, so a
Japanese dataset prepared with this frontend adds no new embedding rows.

Requires the ``ja`` extra:

```
python -m pip install ".[ja]"
```

The Open JTalk dictionary is a third-party artifact under its own license. A
package exported with this frontend is not self-contained.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping


FRONTEND_NAME = "ja-openjtalk"
FRONTEND_VERSION = "1"
IPA_MAPPING_VERSION = "1"
ACCENT_RULE_ID = "openjtalk-fullcontext-a-fields-v1"
LEXICON_ENVIRONMENT_VARIABLE = "INFLECT_JA_LEXICON"

#: Emitted where Open JTalk reports a rise at the start of an accent phrase.
ACCENT_RISE = "↑"
#: Emitted where Open JTalk reports the fall after the accent nucleus.
ACCENT_FALL = "↓"
#: Emitted between accent phrases inside one punctuation-delimited chunk.
ACCENT_PHRASE_BOUNDARY = " "
#: Emitted for an Open JTalk ``pau`` that survives chunk splitting.
PAUSE_SYMBOL = ","

# Long vowels stay as repeated vowels ("koo", not "koː"). Japanese is mora
# timed and the duration predictor works per symbol, so one symbol per mora
# keeps the predicted duration unimodal.
PHONE_TO_IPA: dict[str, str] = {
    "a": "a",
    "i": "i",
    "u": "ɯ",
    "e": "e",
    "o": "o",
    # Devoiced vowels fold to their plain counterpart. Whether the distinction
    # is worth its own symbol is a listening question, not a guess.
    "A": "a",
    "I": "i",
    "U": "ɯ",
    "E": "e",
    "O": "o",
    "N": "ɴ",
    "cl": "ʔ",
    "b": "b",
    "by": "bʲ",
    "ch": "tɕ",
    "d": "d",
    "dy": "dʲ",
    "f": "ɸ",
    "g": "ɡ",
    "gy": "ɡʲ",
    "gw": "ɡʷ",
    "h": "h",
    "hy": "ç",
    "j": "dʑ",
    "k": "k",
    "ky": "kʲ",
    "kw": "kʷ",
    "m": "m",
    "my": "mʲ",
    "n": "n",
    "ny": "ɲ",
    "p": "p",
    "py": "pʲ",
    "r": "ɾ",
    "ry": "ɾʲ",
    "s": "s",
    "sh": "ɕ",
    "t": "t",
    "ts": "ts",
    "ty": "tʲ",
    "v": "v",
    "w": "w",
    "y": "j",
    "z": "z",
}

# Punctuation Open JTalk collapses into an undifferentiated ``pau``. The text
# is split on these instead so the writer's punctuation reaches the model.
PUNCTUATION_MAP: dict[str, str] = {
    "、": ",",  # 、
    "・": ",",  # ・
    ",": ",",
    "。": ".",  # 。
    ".": ".",
    "!": "!",
    "?": "?",
    ":": ":",
    ";": ";",
    "…": "…",  # …
    "‥": "…",  # ‥
}

# Brackets and quotation marks carry no reading. They are removed during
# normalization rather than handed to Open JTalk.
DROPPED_CHARACTERS = frozenset(
    "「」『』〈〉《》【】〔〕"
    "()[]{}<>\"'“”‘’"
)

# A Japanese mora is (C)V, the moraic nasal, or the geminate stop, so only
# these phones can end one. Accent marks are placed at mora boundaries; the
# widely copied label-level rule misfires inside a two-phone mora that is its
# own accent phrase (for example the one-mora particle と).
_MORA_FINAL_PHONES = frozenset("aiueoAIUEO") | {"N", "cl"}

_PHONE_PATTERN = re.compile(r"\-(.+?)\+")
_ACCENT_PATTERN = re.compile(r"/A:(-?\d+)\+(\d+)\+(\d+)")
_PHRASE_PATTERN = re.compile(r"/F:(\d+)_")
# A digit-grouping comma is followed by exactly three digits. Matching any
# digit instead would silently join an enumeration: 1,2,3 into 123.
_GROUPING_COMMA = re.compile(r"(?<=[0-9]),(?=[0-9]{3}(?![0-9]))")
# '.' also occurs inside numbers, where splitting would turn 1.5 into two
# sentences. It separates chunks only when a digit is not on both sides. Any
# comma that survives normalization is a list separator, so it always splits.
_NUMERIC_MARKS = "."
_SENTENCE_MARKS = "".join(sorted(set(PUNCTUATION_MAP) - set(_NUMERIC_MARKS)))
_SPLIT_PATTERN = re.compile(
    "("
    f"(?<![0-9])[{re.escape(_NUMERIC_MARKS)}]"
    f"|[{re.escape(_NUMERIC_MARKS)}](?![0-9])"
    f"|[{re.escape(_SENTENCE_MARKS)}]"
    ")"
)


class JapaneseFrontendError(RuntimeError):
    """Raised when Japanese text cannot be converted into Inflect symbols."""


def _declared_symbols() -> tuple[str, ...]:
    characters: set[str] = set()
    for phonemes in PHONE_TO_IPA.values():
        characters.update(phonemes)
    characters.update(PUNCTUATION_MAP.values())
    characters.update({ACCENT_RISE, ACCENT_FALL, ACCENT_PHRASE_BOUNDARY, PAUSE_SYMBOL})
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
        raise JapaneseFrontendError(
            f"Could not read the Japanese reading lexicon at {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and key and value
        for key, value in payload.items()
    ):
        raise JapaneseFrontendError(
            f"The Japanese reading lexicon at {path} must be a JSON object of "
            "non-empty surface/reading strings."
        )
    return {
        unicodedata.normalize("NFKC", key): unicodedata.normalize("NFKC", value)
        for key, value in payload.items()
    }


class JapaneseOpenJTalkFrontend:
    """Deterministic Japanese normalization and phonemization."""

    def __init__(
        self,
        language: str = "ja",
        lexicon: Mapping[str, str] | None = None,
    ) -> None:
        self.language = language
        self._lexicon = dict(lexicon) if lexicon is not None else _load_environment_lexicon()
        self._lexicon_pattern = self._build_lexicon_pattern()
        self._engine: Any | None = None
        # Load here, not on first use: preparation validates the frontend before
        # touching any data, and a missing dependency should stop it there
        # rather than part-way through a corpus. The module itself stays
        # importable without Open JTalk so the symbol tables can be inspected.
        self._open_jtalk()

    def _build_lexicon_pattern(self) -> re.Pattern[str] | None:
        if not self._lexicon:
            return None
        # Longest surface first so a longer entry is never shadowed by a
        # shorter one that happens to be a prefix.
        ordered = sorted(self._lexicon, key=lambda item: (-len(item), item))
        return re.compile("|".join(re.escape(surface) for surface in ordered))

    def _open_jtalk(self) -> Any:
        if self._engine is None:
            try:
                import pyopenjtalk
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise JapaneseFrontendError(
                    "The Japanese frontend requires Open JTalk. Install the "
                    "toolkit's 'ja' extra, which uses the prebuilt "
                    "pyopenjtalk-plus wheel."
                ) from exc
            self._engine = pyopenjtalk
        return self._engine

    def normalize(self, text: str) -> str:
        """Return NFKC text with unreadable marks removed and readings applied."""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = "".join(
            character
            for character in normalized
            if character not in DROPPED_CHARACTERS
        )
        # Open JTalk reads a digit-grouping comma as a break, turning 3,000 into
        # "three zero zero zero". Removing it restores the intended reading.
        normalized = _GROUPING_COMMA.sub("", normalized)
        if self._lexicon_pattern is not None:
            normalized = self._lexicon_pattern.sub(
                lambda match: self._lexicon[match.group(0)], normalized
            )
        return re.sub(r"\s+", " ", normalized).strip()

    def phonemize(self, text: str) -> str:
        """Return the Inflect symbol stream for already-normalized text."""
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
        result = " ".join(parts)
        return re.sub(r"\s+", " ", result).strip()

    def _phonemize_chunk(self, chunk: str) -> str:
        engine = self._open_jtalk()
        try:
            labels = engine.extract_fullcontext(chunk)
        except Exception as exc:  # pragma: no cover - engine dependent
            raise JapaneseFrontendError(
                f"Open JTalk failed on {chunk!r}: {exc}"
            ) from exc

        entries: list[tuple[str, int, int, int, int]] = []
        for label in labels:
            match = _PHONE_PATTERN.search(label)
            if match is None:
                raise JapaneseFrontendError(
                    f"Open JTalk returned an unreadable label for {chunk!r}."
                )
            phone = match.group(1)
            if phone in {"sil", "xx"}:
                continue
            if phone == "pau":
                entries.append((phone, 0, 0, 0, 0))
                continue
            accent = _ACCENT_PATTERN.search(label)
            phrase = _PHRASE_PATTERN.search(label)
            if accent is None or phrase is None:
                raise JapaneseFrontendError(
                    f"Open JTalk label for {phone!r} in {chunk!r} has no accent fields."
                )
            entries.append(
                (
                    phone,
                    int(accent.group(1)),
                    int(accent.group(2)),
                    int(accent.group(3)),
                    int(phrase.group(1)),
                )
            )

        pieces: list[str] = []
        for index, (phone, position, mora, remaining, phrase_moras) in enumerate(entries):
            if phone == "pau":
                pieces.append(PAUSE_SYMBOL)
                continue
            phonemes = PHONE_TO_IPA.get(phone)
            if phonemes is None:
                raise JapaneseFrontendError(
                    f"Open JTalk produced the unmapped phone {phone!r} for {chunk!r}. "
                    "Add it to PHONE_TO_IPA rather than dropping it."
                )
            pieces.append(phonemes)
            following = entries[index + 1] if index + 1 < len(entries) else None
            if following is None or following[0] == "pau":
                continue
            if phone not in _MORA_FINAL_PHONES:
                continue
            next_mora = following[2]
            if remaining == 1 and next_mora == 1:
                pieces.append(ACCENT_PHRASE_BOUNDARY)
            elif position == 0 and next_mora == mora + 1 and mora != phrase_moras:
                pieces.append(ACCENT_FALL)
            elif mora == 1 and next_mora == 2:
                pieces.append(ACCENT_RISE)
        return "".join(pieces).strip()

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
                "engine": "open_jtalk",
                "ipa_mapping_version": IPA_MAPPING_VERSION,
                "accent_rule": ACCENT_RULE_ID,
                "accent_rise": ACCENT_RISE,
                "accent_fall": ACCENT_FALL,
                "accent_phrase_boundary": "space",
                "long_vowels": "repeated vowel per mora",
                "devoiced_vowels": "folded to the plain vowel",
                "pause_symbol": PAUSE_SYMBOL,
                "punctuation_map": dict(sorted(PUNCTUATION_MAP.items())),
                "dropped_characters": "".join(sorted(DROPPED_CHARACTERS)),
                "lexicon": dict(sorted(self._lexicon.items())),
                "declared_symbol_count": len(DECLARED_SYMBOLS),
            },
        }


def create_frontend(*, language: str = "ja") -> JapaneseOpenJTalkFrontend:
    """Return the Japanese frontend used by the ``ja-openjtalk`` registry entry."""
    return JapaneseOpenJTalkFrontend(language=language)
