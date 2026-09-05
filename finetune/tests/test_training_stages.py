from __future__ import annotations

import logging
import re
from dataclasses import replace

import pytest

from inflect_finetune.training import (
    DECODER_POLISH_MODES,
    POSTERIOR_INITS,
    STAGE_ADAPT,
    STAGE_DECODER,
    STAGE_POSTERIOR,
    TrainingOptions,
    _stage_for_step,
    _validate_options,
)


def _options() -> TrainingOptions:
    return TrainingOptions(
        base_model="nano",
        prepared_dir="prepared",
        output_dir="run",
        posterior_warmup_steps=10,
        decoder_unfreeze_step=20,
    )


def test_stage_boundaries_select_the_stage_for_the_next_step() -> None:
    options = _options()

    assert _stage_for_step(options, 0) == STAGE_POSTERIOR
    assert _stage_for_step(options, 9) == STAGE_POSTERIOR
    assert _stage_for_step(options, 10) == STAGE_ADAPT
    assert _stage_for_step(options, 19) == STAGE_ADAPT
    assert _stage_for_step(options, 20) == STAGE_DECODER


def test_the_enumerated_settings_offer_exactly_the_values_the_cli_advertises() -> None:
    """The parser spells these choices out again, so the two lists must agree."""

    assert DECODER_POLISH_MODES == ("adversarial", "recon")
    assert POSTERIOR_INITS == ("fresh", "inherit")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"adversarial_ramp_steps": -1}, "adversarial_ramp_steps must be non-negative."),
        ({"decoder_lr_warmup_steps": -1}, "decoder_lr_warmup_steps must be non-negative."),
        ({"stft_loss_weight": -0.5}, "stft_loss_weight must be non-negative."),
        ({"decoder_proximal_weight": -1e-9}, "decoder_proximal_weight must be non-negative."),
        (
            {"decoder_polish_mode": "reconstruction"},
            "decoder_polish_mode must be one of ['adversarial', 'recon'].",
        ),
        ({"posterior_init": "warm"}, "posterior_init must be one of ['fresh', 'inherit']."),
        ({"generator_ema_decay": 1.0}, "generator_ema_decay must be at least 0 and below 1."),
        ({"generator_ema_decay": -0.001}, "generator_ema_decay must be at least 0 and below 1."),
    ],
)
def test_a_new_setting_outside_its_domain_is_rejected_by_name(
    overrides: dict[str, object], message: str
) -> None:
    """The whole message is pinned: a refactor that widens a domain has to say so here."""

    with pytest.raises(ValueError, match=re.escape(message)):
        _validate_options(replace(_options(), **overrides))


def test_the_zero_boundary_of_every_new_weight_and_step_count_is_accepted() -> None:
    """Zero is the documented default of these settings, so it cannot be rejected."""

    _validate_options(
        replace(
            _options(),
            adversarial_ramp_steps=0,
            decoder_lr_warmup_steps=0,
            stft_loss_weight=0.0,
            decoder_proximal_weight=0.0,
            generator_ema_decay=0.0,
        )
    )


def test_a_decay_just_below_one_is_accepted_because_only_one_itself_never_updates() -> None:
    _validate_options(replace(_options(), generator_ema_decay=0.9999))


def test_recon_polish_is_rejected_when_the_decoder_never_unfreezes() -> None:
    """The polish stage it names does not exist without an unfreeze step."""

    options = replace(_options(), decoder_unfreeze_step=None, decoder_polish_mode="recon")

    with pytest.raises(ValueError, match="only reachable when the decoder unfreezes"):
        _validate_options(options)


def test_a_decoder_lr_warmup_is_rejected_when_the_decoder_never_unfreezes() -> None:
    """The ramp is measured from the unfreeze step, so it would silently do nothing."""

    options = replace(_options(), decoder_unfreeze_step=None, decoder_lr_warmup_steps=1)

    with pytest.raises(ValueError, match="measured from decoder_unfreeze_step"):
        _validate_options(options)


def test_gating_without_a_decoder_unfreeze_is_allowed_and_warned_about(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """This combination is meaningful: adaptation with no adversarial term at all.

    It is the one new setting that survives `decoder_unfreeze_step=None`, so it
    is accepted rather than rejected, and the warning is the only signal that
    the discriminator is being trained and never used.
    """

    options = replace(_options(), decoder_unfreeze_step=None, adversarial_gating=True)

    with caplog.at_level(logging.WARNING, logger="inflect_finetune"):
        assert _validate_options(options) is None

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].name == "inflect_finetune"
    assert "adversarial_gating" in warnings[0].getMessage()
    assert "decoder_unfreeze_step" in warnings[0].getMessage()


def test_default_options_and_the_gated_recipe_both_validate_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The remedy's own recommended combination must not trip its own guards."""

    with caplog.at_level(logging.WARNING, logger="inflect_finetune"):
        _validate_options(_options())
        _validate_options(
            replace(
                _options(),
                adversarial_gating=True,
                adversarial_ramp_steps=1_000,
                decoder_lr_warmup_steps=300,
                decoder_polish_mode="recon",
                stft_loss_weight=2.0,
                decoder_proximal_weight=1.0,
                decoder_freeze_upsamplers=True,
                posterior_init="inherit",
                generator_ema_decay=0.999,
            )
        )

    assert [record.getMessage() for record in caplog.records] == []
