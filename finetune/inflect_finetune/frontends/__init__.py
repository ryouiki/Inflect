"""Named language frontends bundled with the adaptation toolkit.

A registry entry is a *name for a bundled custom frontend file*, not a new
frontend mode. :func:`resolve` returns the ordinary custom ``FrontendOptions``
that preparation, auditing, and export already understand, so the existing
source hashing, declared-symbol enforcement, determinism checks, hook
packaging, and refusal to fall back to the release English frontend all apply
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .. import __version__
from ..frontend import FrontendOptions


_BUNDLE_ROOT = Path(__file__).resolve().parent


class FrontendRegistryError(ValueError):
    """Raised when a named frontend cannot be resolved to a usable hook."""


@dataclass(frozen=True)
class RegistryEntry:
    """One bundled language frontend and the way to invoke it."""

    name: str
    module_file: Path
    factory: str
    language: str
    extras: tuple[str, ...]
    summary: str

    def hook(self) -> str:
        """Return the ``file.py:function`` hook accepted by the frontend loader."""
        return f"{self.module_file}:{self.factory}"

    def accepts_language(self, language: str) -> bool:
        """Accept the declared language and its regional variants."""
        value = language.strip()
        return value == self.language or value.startswith(f"{self.language}-")


REGISTRY: dict[str, RegistryEntry] = {
    "ja-openjtalk": RegistryEntry(
        name="ja-openjtalk",
        module_file=_BUNDLE_ROOT / "ja_openjtalk.py",
        factory="create_frontend",
        language="ja",
        extras=("ja",),
        summary=(
            "Japanese Open JTalk grapheme-to-phoneme with pitch-accent marks. "
            "Install the 'ja' extra."
        ),
    ),
    "ko-g2pkk": RegistryEntry(
        name="ko-g2pkk",
        module_file=_BUNDLE_ROOT / "ko_g2pkk.py",
        factory="create_frontend",
        language="ko",
        extras=("ko",),
        summary=(
            "Korean g2pkk phonology with a direct Hangul-to-phoneme mapping. "
            "Install the 'ko' extra."
        ),
    ),
}


def registry_names() -> tuple[str, ...]:
    """Return the bundled frontend names in a stable order."""
    return tuple(sorted(REGISTRY))


def is_registry_frontend(name: str) -> bool:
    """Return whether a ``--frontend`` value names a bundled frontend."""
    return name in REGISTRY


def get(name: str) -> RegistryEntry:
    """Return one registry entry or raise an actionable error."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise FrontendRegistryError(
            f"Unknown bundled frontend {name!r}. Available: "
            + ", ".join(registry_names())
        ) from None


def resolve(
    frontend: str,
    language: str,
    *,
    hook: str | None = None,
) -> FrontendOptions:
    """Return the frontend options for a mode name or a bundled frontend name.

    Non-registry values pass through unchanged so that ``espeak``,
    ``prephonemized``, and explicit ``custom`` hooks keep their exact behavior.
    """
    if not is_registry_frontend(frontend):
        return FrontendOptions(mode=frontend, language=language, hook=hook)
    entry = get(frontend)
    if hook:
        raise FrontendRegistryError(
            f"Bundled frontend {frontend!r} supplies its own hook; remove the "
            "explicit frontend hook or use --frontend custom instead."
        )
    if not entry.accepts_language(language):
        raise FrontendRegistryError(
            f"Bundled frontend {frontend!r} is configured for language "
            f"{entry.language!r}, but {language!r} was requested. "
            f"Pass --language {entry.language}."
        )
    if not entry.module_file.is_file():
        raise FrontendRegistryError(
            f"Bundled frontend {frontend!r} is missing its source file: "
            f"{entry.module_file}."
        )
    return FrontendOptions(mode="custom", language=language, hook=entry.hook())


def registry_record(name: str) -> dict[str, Any]:
    """Return the reproducibility record stored in a prepared ``dataset.json``."""
    entry = get(name)
    return {
        "name": entry.name,
        "factory": entry.factory,
        "language": entry.language,
        "extras": list(entry.extras),
        "summary": entry.summary,
        "toolkit_version": __version__,
    }


def hook_path_for_record(record: Mapping[str, Any] | None) -> Path | None:
    """Return the bundled hook file named by a prepared registry record.

    Export needs the exact hook source for a custom frontend. When a prepared
    dataset was produced through the registry, the file ships with the toolkit,
    so the path can be recovered instead of asked for.
    """
    if not isinstance(record, Mapping):
        return None
    name = record.get("name")
    if not isinstance(name, str) or not is_registry_frontend(name):
        return None
    module_file = get(name).module_file
    return module_file if module_file.is_file() else None
