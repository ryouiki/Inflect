"""Symbol inventory validation and embedding migration.

CONTRACT.md requires an automated check for "embedding migration by symbol
identity". These tests provide it, and cover the rule that lets a checkpoint
this toolkit adapted serve as the base of a later run.

`load_runtime_components` needs very little from a release runtime — the two
model classes, an assignable `monotonic_align`, `commons`, and
`text.symbols.symbols` — so the fixtures build a stub runtime rather than
requiring the real one. The stub proves the validation and migration logic; the
opt-in test at the end proves the same path against a real release.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from inflect_finetune.checkpoint import warm_start_from_release
from inflect_finetune.modeling import (
    BASE_SYMBOL_COUNT,
    build_training_models,
    load_runtime_components,
    validate_release_compatible_symbols,
)
from inflect_finetune.symbols import BASE_SYMBOLS


EMBEDDING_KEY = "enc_p.emb.weight"
EMBEDDING_WIDTH = 8

_MODELS_SOURCE = '''
from torch import nn

monotonic_align = None


class SynthesizerTrn(nn.Module):
    def __init__(self, n_vocab, spec_channels, segment_size, **kwargs):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.enc_p = nn.Module()
        self.enc_p.emb = nn.Embedding(n_vocab, 8)
        self.dec = nn.Linear(8, 4)
        if not kwargs.get("inference_only", False):
            self.enc_q = nn.Linear(4, 8)


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        self.use_spectral_norm = use_spectral_norm
        self.head = nn.Linear(2, 1)
'''

_CONFIG = {
    "train": {"segment_size": 8192},
    "data": {"filter_length": 1024, "hop_length": 256, "n_speakers": 0},
    "model": {"inference_only": True},
}


def _extended(count: int) -> list[str]:
    """Return the release inventory plus `count` distinct new symbols."""
    return list(BASE_SYMBOLS) + [chr(0xA71C + index) for index in range(count)]


def _write_runtime(root: Path, symbols: list[str]) -> None:
    runtime = root / "runtime"
    (runtime / "text").mkdir(parents=True)
    (runtime / "models.py").write_text(_MODELS_SOURCE, encoding="utf-8")
    (runtime / "commons.py").write_text("", encoding="utf-8")
    (runtime / "text" / "__init__.py").write_text("", encoding="utf-8")
    (runtime / "text" / "symbols.py").write_text(
        f"symbols = {symbols!r}\n", encoding="utf-8"
    )


def _write_checkpoint(root: Path, symbols: list[str], *, seed: int = 3) -> None:
    """Write an inference checkpoint whose embedding matches `symbols`."""
    generator = torch.Generator().manual_seed(seed)
    state = {
        EMBEDDING_KEY: torch.randn(
            len(symbols), EMBEDDING_WIDTH, generator=generator
        ),
        "dec.weight": torch.randn(4, 8, generator=generator),
        "dec.bias": torch.randn(4, generator=generator),
    }
    torch.save(
        {
            "format": "inflect_vits_inference_checkpoint_v1",
            "model": state,
            "iteration": 0,
        },
        root / "model.pth",
    )


def _base_model(root: Path, symbols: list[str]) -> Path:
    """Build a complete stub release directory for `symbols`."""
    root.mkdir(parents=True, exist_ok=True)
    _write_runtime(root, symbols)
    _write_checkpoint(root, symbols)
    (root / "config.json").write_text(json.dumps(_CONFIG) + "\n", encoding="utf-8")
    return root


def _migrate(base_root: Path, target_symbols: list[str]):
    """Warm-start a training model for `target_symbols` from `base_root`."""
    components = load_runtime_components(base_root)
    bundle = build_training_models(base_root, target_symbols)
    report = warm_start_from_release(
        bundle.generator,
        base_root / "model.pth",
        components.base_symbols,
        target_symbols,
    )
    return bundle, report


# --------------------------------------------------------------------------
# Inventory validation
# --------------------------------------------------------------------------


def test_the_release_inventory_is_accepted_unchanged() -> None:
    assert validate_release_compatible_symbols(
        list(BASE_SYMBOLS), source="test"
    ) == tuple(BASE_SYMBOLS)


def test_an_extended_inventory_is_accepted() -> None:
    """An adapted checkpoint may append symbols; that is what makes it reusable."""
    extended = _extended(2)
    assert len(
        validate_release_compatible_symbols(extended, source="test")
    ) == BASE_SYMBOL_COUNT + 2


@pytest.mark.parametrize(
    ("symbols", "message"),
    [
        (list(BASE_SYMBOLS)[:-1], "must preserve"),
        ([], "non-empty list"),
        (["x"] + list(BASE_SYMBOLS)[1:], "begin with '_'"),
        (
            list(BASE_SYMBOLS)[:100] + ["ǀ"] + list(BASE_SYMBOLS)[101:],
            "must exactly preserve",
        ),
        (list(BASE_SYMBOLS) + ["a"], "duplicate/custom-added"),
    ],
    ids=["too-short", "empty", "wrong-pad", "altered-prefix", "extra-duplicate"],
)
def test_an_inventory_that_disturbs_the_release_prefix_is_rejected(
    symbols: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_release_compatible_symbols(symbols, source="test")


def test_the_rejection_names_the_inventory_it_read() -> None:
    """A failure has to say which of the two inventories was wrong."""
    with pytest.raises(ValueError, match="prepared/ja/symbols.json"):
        validate_release_compatible_symbols(["a"], source="prepared/ja/symbols.json")


def test_a_runtime_inventory_is_held_to_the_same_rule(tmp_path: Path) -> None:
    release = _base_model(tmp_path / "release", list(BASE_SYMBOLS))
    assert len(load_runtime_components(release).base_symbols) == BASE_SYMBOL_COUNT

    adapted = _base_model(tmp_path / "adapted", _extended(2))
    assert len(load_runtime_components(adapted).base_symbols) == BASE_SYMBOL_COUNT + 2

    broken = _base_model(tmp_path / "broken", list(BASE_SYMBOLS)[:-1])
    with pytest.raises(RuntimeError, match="must preserve"):
        load_runtime_components(broken)


# --------------------------------------------------------------------------
# Embedding migration
# --------------------------------------------------------------------------


def test_release_rows_are_copied_bit_for_bit_and_new_rows_initialized(
    tmp_path: Path,
) -> None:
    base = _base_model(tmp_path / "base", list(BASE_SYMBOLS))
    target = _extended(2)
    bundle, report = _migrate(base, target)

    source = torch.load(base / "model.pth", map_location="cpu", weights_only=False)
    migrated = bundle.generator.state_dict()[EMBEDDING_KEY]

    assert report.migrated_embedding_rows == BASE_SYMBOL_COUNT
    assert report.initialized_embedding_rows == 2
    assert report.base_symbol_count == BASE_SYMBOL_COUNT
    assert report.discarded_base_symbols == ()
    assert torch.equal(
        migrated[:BASE_SYMBOL_COUNT], source["model"][EMBEDDING_KEY]
    )


def test_an_extended_base_carries_its_added_rows_forward(tmp_path: Path) -> None:
    """The chaining case: stage one taught extra symbols, stage two keeps them."""
    symbols = _extended(2)
    base = _base_model(tmp_path / "base", symbols)
    bundle, report = _migrate(base, symbols)

    source = torch.load(base / "model.pth", map_location="cpu", weights_only=False)
    assert report.migrated_embedding_rows == len(symbols)
    assert report.initialized_embedding_rows == 0
    assert report.base_symbol_count == len(symbols)
    assert torch.equal(
        bundle.generator.state_dict()[EMBEDDING_KEY], source["model"][EMBEDDING_KEY]
    )


def test_symbols_the_new_dataset_drops_are_reported(tmp_path: Path) -> None:
    """Dropping a trained row is correct here, but it must not be silent."""
    base = _base_model(tmp_path / "base", _extended(2))
    _, report = _migrate(base, list(BASE_SYMBOLS))

    assert report.base_symbol_count == BASE_SYMBOL_COUNT + 2
    assert report.discarded_base_symbols == ("ꜜ", "ꜝ")
    assert report.migrated_embedding_rows == BASE_SYMBOL_COUNT
    assert report.initialized_embedding_rows == 0


def test_rows_follow_the_symbol_string_not_its_position(tmp_path: Path) -> None:
    """Reordering the added symbols must not move their weights."""
    base = _base_model(tmp_path / "base", _extended(2))
    added = [chr(0xA71C), chr(0xA71D)]
    reordered = list(BASE_SYMBOLS) + [added[1], added[0]]
    bundle, report = _migrate(base, reordered)

    source = torch.load(base / "model.pth", map_location="cpu", weights_only=False)
    migrated = bundle.generator.state_dict()[EMBEDDING_KEY]

    assert report.initialized_embedding_rows == 0
    assert report.discarded_base_symbols == ()
    # The base held them in the opposite order, so the rows must be swapped.
    assert torch.equal(migrated[BASE_SYMBOL_COUNT], source["model"][EMBEDDING_KEY][-1])
    assert torch.equal(migrated[BASE_SYMBOL_COUNT + 1], source["model"][EMBEDDING_KEY][-2])


def test_new_row_initialization_is_deterministic(tmp_path: Path) -> None:
    base = _base_model(tmp_path / "base", list(BASE_SYMBOLS))
    target = _extended(2)
    first, _ = _migrate(base, target)
    second, _ = _migrate(base, target)

    assert torch.equal(
        first.generator.state_dict()[EMBEDDING_KEY],
        second.generator.state_dict()[EMBEDDING_KEY],
    )


def test_a_base_whose_inventory_contradicts_its_checkpoint_is_rejected(
    tmp_path: Path,
) -> None:
    """The length check in warm-start still guards a mismatched base."""
    base = _base_model(tmp_path / "base", list(BASE_SYMBOLS))
    _write_checkpoint(base, _extended(2))  # checkpoint now wider than the runtime

    with pytest.raises(RuntimeError, match="Base symbol inventory length"):
        _migrate(base, list(BASE_SYMBOLS))


# --------------------------------------------------------------------------
# Opt-in: the same path against a real release
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("INFLECT_TEST_BASE_MODEL"),
    reason="Set INFLECT_TEST_BASE_MODEL to a release directory, 'micro', or 'nano'.",
)
def test_a_real_release_chains_through_an_extended_inventory(tmp_path: Path) -> None:
    """Prove chaining against the real architecture, not the stub.

    Simulates a stage-one export by widening a real release's inventory and
    embedding, then warm-starts stage two from it.
    """
    from inflect_finetune.modeling import resolve_base_model

    release = resolve_base_model(os.environ["INFLECT_TEST_BASE_MODEL"])
    staged = tmp_path / "stage-one"
    shutil.copytree(release, staged)

    symbols = _extended(2)
    (staged / "runtime" / "text" / "symbols.py").write_text(
        f"symbols = {symbols!r}\n"
        'SPACE_ID = symbols.index(" ") if " " in symbols else -1\n',
        encoding="utf-8",
    )
    payload = torch.load(staged / "model.pth", map_location="cpu", weights_only=False)
    embedding = payload["model"][EMBEDDING_KEY]
    widened = torch.cat(
        [embedding, torch.zeros(2, embedding.shape[1], dtype=embedding.dtype)]
    )
    payload["model"][EMBEDDING_KEY] = widened
    torch.save(payload, staged / "model.pth")

    components = load_runtime_components(staged)
    assert len(components.base_symbols) == len(symbols)

    bundle = build_training_models(staged, symbols)
    report = warm_start_from_release(
        bundle.generator, staged / "model.pth", components.base_symbols, symbols
    )

    assert report.initialized_embedding_rows == 0
    assert report.migrated_embedding_rows == len(symbols)
    assert torch.equal(bundle.generator.state_dict()[EMBEDDING_KEY], widened)
