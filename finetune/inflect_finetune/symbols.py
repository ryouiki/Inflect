"""Ordered Inflect symbol inventories and phoneme coverage reporting."""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


_PAD = "_"
_PUNCTUATION = ';:,.!?¡¿—…"«»“” '
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_LETTERS_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵ"
    "ɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞"
    "↓↑→↗↘'̩'ᵻ"
)
BASE_SYMBOLS: tuple[str, ...] = tuple(
    [_PAD] + list(_PUNCTUATION) + list(_LETTERS) + list(_LETTERS_IPA)
)


class SymbolInventoryError(ValueError):
    """Raised when a symbol inventory is malformed."""


@dataclass(frozen=True)
class CoverageReport:
    """Character coverage of phoneme strings against an ordered inventory."""

    total_characters: int
    covered_characters: int
    coverage_fraction: float
    unique_observed: int
    unknown_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class SymbolInventory:
    """An identity-preserving extension of a base symbol inventory."""

    symbols: tuple[str, ...]
    base_symbols: tuple[str, ...]
    added_symbols: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the public symbols.json contract."""
        return {
            "format": "inflect_symbol_inventory_v1",
            "symbols": list(self.symbols),
            "base_symbols": list(self.base_symbols),
            "base_size": len(self.base_symbols),
            "total_size": len(self.symbols),
            "added_symbols": list(self.added_symbols),
            "base_identity": [
                {
                    "symbol": symbol,
                    "base_index": position,
                    "adapted_index": position,
                }
                for position, symbol in enumerate(self.base_symbols)
            ],
        }


def _validate_symbols(
    symbols: Sequence[str],
    *,
    label: str,
    allow_duplicates: bool = False,
) -> tuple[str, ...]:
    result = tuple(symbols)
    if not result:
        raise SymbolInventoryError(f"{label} cannot be empty.")
    if any(not isinstance(symbol, str) or not symbol for symbol in result):
        raise SymbolInventoryError(f"{label} must contain non-empty strings.")
    if not allow_duplicates and len(set(result)) != len(result):
        duplicates = sorted(symbol for symbol, count in Counter(result).items() if count > 1)
        raise SymbolInventoryError(f"{label} contains duplicate symbols: {duplicates!r}")
    return result


def load_base_symbols(path: Path | None = None) -> tuple[str, ...]:
    """Load a JSON symbol list, or use the exact Inflect v2 release inventory."""
    if path is None:
        return BASE_SYMBOLS
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SymbolInventoryError(f"Could not read base symbols from {path}: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("symbols")
    if not isinstance(payload, list):
        raise SymbolInventoryError(
            f"Base symbol file {path} must be a JSON list or an object with 'symbols'."
        )
    return _validate_symbols(
        payload,
        label=f"base symbols in {path}",
        allow_duplicates=True,
    )


def audit_symbol_coverage(
    phoneme_texts: Iterable[str],
    symbols: Sequence[str],
) -> CoverageReport:
    """Measure character-level coverage against a supplied inventory."""
    inventory = set(
        _validate_symbols(symbols, label="symbols", allow_duplicates=True)
    )
    counts: Counter[str] = Counter()
    for text in phoneme_texts:
        if not isinstance(text, str):
            raise SymbolInventoryError("Phoneme entries must be strings.")
        counts.update(unicodedata.normalize("NFC", text))
    total = sum(counts.values())
    unknown = {symbol: counts[symbol] for symbol in sorted(counts) if symbol not in inventory}
    unknown_total = sum(unknown.values())
    covered = total - unknown_total
    return CoverageReport(
        total_characters=total,
        covered_characters=covered,
        coverage_fraction=(covered / total) if total else 1.0,
        unique_observed=len(counts),
        unknown_counts=unknown,
    )


def build_symbol_inventory(
    phoneme_texts: Iterable[str],
    *,
    base_symbols: Sequence[str] = BASE_SYMBOLS,
    extension_symbols: Sequence[str] = (),
) -> SymbolInventory:
    """Append declared then observed symbols while preserving base symbol indices."""
    base = _validate_symbols(
        base_symbols,
        label="base_symbols",
        allow_duplicates=True,
    )
    declared = _validate_symbols(
        extension_symbols,
        label="extension_symbols",
    ) if extension_symbols else ()
    observed: set[str] = set()
    for text in phoneme_texts:
        if not isinstance(text, str):
            raise SymbolInventoryError("Phoneme entries must be strings.")
        observed.update(unicodedata.normalize("NFC", text))
    declared_added = tuple(symbol for symbol in declared if symbol not in base)
    remaining = observed.difference(base).difference(declared_added)
    added = declared_added + tuple(
        sorted(remaining, key=lambda value: tuple(map(ord, value)))
    )
    return SymbolInventory(symbols=base + added, base_symbols=base, added_symbols=added)


def write_symbol_inventory(path: Path, inventory: SymbolInventory) -> None:
    """Write symbols.json deterministically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inventory.to_dict(), ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
