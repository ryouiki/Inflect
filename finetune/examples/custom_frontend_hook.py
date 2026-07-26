"""Minimal public custom-frontend example for Inflect adaptation exports."""

from __future__ import annotations

import re
import unicodedata


class ExampleFrontend:
    def __init__(self, language: str) -> None:
        self.language = language

    def normalize(self, text: str) -> str:
        value = unicodedata.normalize("NFKC", text)
        return re.sub(r"\s+", " ", value).strip()

    def phonemize(self, text: str) -> str:
        # Replace this illustrative mapping with a real language frontend.
        return " ".join(character.lower() for character in text if character.isalpha())

    def symbols(self) -> tuple[str, ...]:
        return tuple("abcdefghijklmnopqrstuvwxyz ")

    def metadata(self) -> dict[str, object]:
        return {
            "name": "public-example-character-frontend",
            "version": "1",
            "language": self.language,
            "configuration": {"mapping": "lowercase Unicode alphabetic characters"},
        }


def create_frontend(*, language: str) -> ExampleFrontend:
    return ExampleFrontend(language)
