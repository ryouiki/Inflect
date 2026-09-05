"""Pure-tensor tests of the generator objective and the remedy schedules.

The comb at multiples of the frame rate was fixed by adding options, not by
changing the loss. So the first claim in this file is the one that matters
most: with every new option at its default, the assembled loss is the same
bits and the optimizer schedule is the same numbers as the runs that came
before. Everything else here pins one schedule or one term.
"""

from __future__ import annotations

import math
import types

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from inflect_finetune.training import (
    STAGE_ADAPT,
    STAGE_DECODER,
    STAGE_POSTERIOR,
    STAGES,
    TrainingOptions,
    _adversarial_weight,
    _apply_stage,
    _decoder_lr_scale,
    _decoder_reference,
    _discriminator_active,
    _ema_decay,
    _ema_state,
    _ema_update,
    _enabled_groups,
    _generator_groups,
    _generator_objective,
    _latent_statistics,
    _mel_from_spec,
    _mel_from_waveform,
    _multi_resolution_stft_loss,
    _proximal_loss,
    _scalar,
    _scaled_decoder_lr,
)
from inflect_finetune.training_data import (
    MAGNITUDE_FLOOR_POWER,
    AudioConfig,
    magnitude_spectrogram,
    spectrogram,
)

SAMPLE_RATE = 24_000
HOP_LENGTH = 256
# The released Inflect-Micro-v2 analysis front end, which is all
# _mel_from_spec and _mel_from_waveform read off a bundle.
MEL_DATA = {
    "filter_length": 1024,
    "hop_length": HOP_LENGTH,
    "win_length": 1024,
    "n_mel_channels": 80,
    "sampling_rate": SAMPLE_RATE,
    "mel_fmin": 0.0,
    "mel_fmax": 12_000.0,
}


def options(**overrides) -> TrainingOptions:
    """A TrainingOptions with the three required paths filled in.

    Nothing here resolves a preset or touches a filesystem, so the paths are
    never opened.
    """

    settings = {
        "base_model": "nano",
        "prepared_dir": "prepared",
        "output_dir": "run",
    }
    settings.update(overrides)
    return TrainingOptions(**settings)


def bundle() -> types.SimpleNamespace:
    return types.SimpleNamespace(config={"data": dict(MEL_DATA)})


def loss_terms() -> dict[str, torch.Tensor]:
    """Five distinct, long-mantissa loss values.

    Round numbers would let a term swap or a regrouping pass unnoticed; these
    are deliberately values whose sums depend on the order of addition.
    """

    generator = torch.Generator().manual_seed(4321)
    values = torch.rand(5, generator=generator, dtype=torch.float32)
    names = ("loss_generator", "loss_feature", "loss_mel", "loss_duration", "loss_kl")
    return {name: values[index].clone() for index, name in enumerate(names)}


def noise(seconds: float = 1.0, amplitude: float = 0.05) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260905)
    samples = int(seconds * SAMPLE_RATE)
    return amplitude * torch.randn(1, samples, generator=generator)


def grid_comb(seconds: float = 1.0, hop: int = HOP_LENGTH, level_db: float = -34.0) -> torch.Tensor:
    """An impulse every `hop` samples: the artifact this file is about.

    One impulse per decoder frame puts energy at every multiple of
    sample_rate / hop = 93.75 Hz, right across the top octaves where an
    80-band mel has bands hundreds of hertz wide.
    """

    samples = int(seconds * SAMPLE_RATE)
    comb = torch.zeros(1, samples)
    comb[:, ::hop] = 10.0 ** (level_db / 20.0)
    return comb


class _DecoderStub(nn.Module):
    """Module names matching the released HiFi-GAN V1 decoder's own layout."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_pre = nn.Conv1d(2, 2, 1)
        self.ups = nn.ModuleList([nn.Conv1d(2, 2, 1)])
        self.resblocks = nn.ModuleList([nn.Conv1d(2, 2, 1)])
        self.conv_post = nn.Conv1d(2, 2, 1)


class _GeneratorStub(nn.Module):
    """Enough of a generator for the parameter-group machinery.

    _generator_groups classifies by name prefix, so the five submodules below
    are what decide the three groups; nothing here is ever run forward.
    """

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1234)
        self.enc_p = nn.Conv1d(2, 2, 1)
        self.dp = nn.Conv1d(2, 2, 1)
        self.flow = nn.Conv1d(2, 2, 1)
        self.enc_q = nn.Conv1d(2, 2, 1)
        self.dec = _DecoderStub()


def _decoder_optimizer(learning_rate: float = 1.0e-3) -> tuple[nn.Parameter, torch.optim.Optimizer]:
    parameter = nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "name": "decoder", "lr": learning_rate, "lr_multiplier": 1.0}],
        lr=learning_rate,
    )
    return parameter, optimizer


def _toy_warmup_run(warmup_steps: int, steps: int = 10) -> dict:
    """Ten optimizer steps under an ExponentialLR, with and without warm-up."""

    torch.manual_seed(0)
    parameter, optimizer = _decoder_optimizer()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    settings = options(
        posterior_warmup_steps=0,
        decoder_unfreeze_step=0,
        decoder_lr_warmup_steps=warmup_steps,
    )
    applied: list[float] = []
    moved: list[bool] = []
    for step in range(steps):
        parameter.grad = torch.full((3,), 0.1)
        scale = _decoder_lr_scale(settings, step, STAGE_DECODER)
        with _scaled_decoder_lr(optimizer, scale):
            applied.append(optimizer.param_groups[0]["lr"])
            before = parameter.detach().clone()
            optimizer.step()
            moved.append(not torch.equal(before, parameter.detach()))
        scheduler.step()
    return {
        "nominal_lr": optimizer.param_groups[0]["lr"],
        "applied": applied,
        "moved": moved,
    }


def test_the_default_objective_is_bitwise_identical_to_the_expression_it_replaced() -> None:
    """The whole point of the remedy: a default run computes the old number.

    The right-hand side is the inline expression _generator_objective
    replaced, written out in its original left-to-right order. It cannot be
    tidied into `loss_generator + fw * lf + (mw * lm + dw * ld + kw * lk)` or
    any other grouping, because float addition is not associative: regrouping
    moves the last bits, and `torch.equal` is exactly the assertion that they
    do not move.
    """

    settings = options()
    terms = loss_terms()
    total = _generator_objective(settings, adversarial_weight=1.0, **terms)
    expected = (
        terms["loss_generator"]
        + settings.feature_loss_weight * terms["loss_feature"]
        + settings.mel_loss_weight * terms["loss_mel"]
        + settings.duration_loss_weight * terms["loss_duration"]
        + settings.kl_loss_weight * terms["loss_kl"]
    )
    assert torch.equal(total, expected)


@pytest.mark.parametrize(
    ("step", "weight"),
    [(0, 0.0), (2_999, 0.0), (3_000, 0.0), (3_500, 0.5), (4_000, 1.0), (5_000, 1.0)],
)
def test_gating_holds_the_adversarial_weight_at_zero_until_the_decoder_unfreezes(
    step, weight
) -> None:
    """Zero while the decoder is frozen, then a linear ramp over 1000 steps."""

    settings = options(
        adversarial_gating=True,
        decoder_unfreeze_step=3_000,
        adversarial_ramp_steps=1_000,
    )
    stage = STAGE_DECODER if step >= 3_000 else STAGE_ADAPT
    assert _adversarial_weight(settings, step, stage) == weight


def test_a_zero_length_ramp_reaches_full_weight_on_the_unfreeze_step() -> None:
    settings = options(
        adversarial_gating=True,
        decoder_unfreeze_step=3_000,
        adversarial_ramp_steps=0,
    )
    assert _adversarial_weight(settings, 2_999, STAGE_ADAPT) == 0.0
    assert _adversarial_weight(settings, 3_000, STAGE_DECODER) == 1.0


def test_gating_without_an_unfreeze_step_never_admits_an_adversarial_gradient() -> None:
    """A decoder that never unfreezes can never answer the discriminator."""

    settings = options(adversarial_gating=True, decoder_unfreeze_step=None)
    for step in (0, 500, 3_000, 20_000):
        assert _adversarial_weight(settings, step, STAGE_ADAPT) == 0.0


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("step", [0, 1, 499, 500, 2_999, 3_000, 3_500, 20_000])
def test_the_default_options_leave_the_adversarial_weight_at_one_in_every_stage(
    stage, step
) -> None:
    assert _adversarial_weight(options(), step, stage) == 1.0


@pytest.mark.parametrize(
    ("stage", "weight"),
    [(STAGE_POSTERIOR, 1.0), (STAGE_ADAPT, 1.0), (STAGE_DECODER, 0.0)],
)
def test_recon_polish_drops_the_adversarial_terms_in_the_polish_stage_only(
    stage, weight
) -> None:
    settings = options(decoder_polish_mode="recon", decoder_unfreeze_step=3_000)
    assert _adversarial_weight(settings, 3_500, stage) == weight


@pytest.mark.parametrize(
    ("mode", "stage", "active"),
    [
        ("recon", STAGE_POSTERIOR, True),
        ("recon", STAGE_ADAPT, True),
        ("recon", STAGE_DECODER, False),
        ("adversarial", STAGE_POSTERIOR, True),
        ("adversarial", STAGE_ADAPT, True),
        ("adversarial", STAGE_DECODER, True),
    ],
)
def test_recon_polish_idles_the_discriminator_only_during_the_polish_stage(
    mode, stage, active
) -> None:
    settings = options(decoder_polish_mode=mode, decoder_unfreeze_step=3_000)
    assert _discriminator_active(settings, stage) is active


def test_a_half_ramped_weight_scales_the_adversarial_pair_and_leaves_reconstruction_alone() -> None:
    """A non-unit feature weight makes the grouping of the scaling visible."""

    settings = options(feature_loss_weight=2.0)
    terms = loss_terms()
    total = _generator_objective(settings, adversarial_weight=0.5, **terms)
    expected = (
        0.5 * terms["loss_generator"]
        + (0.5 * settings.feature_loss_weight) * terms["loss_feature"]
        + settings.mel_loss_weight * terms["loss_mel"]
        + settings.duration_loss_weight * terms["loss_duration"]
        + settings.kl_loss_weight * terms["loss_kl"]
    )
    assert torch.equal(total, expected)
    reconstruction = _generator_objective(
        settings,
        adversarial_weight=0.0,
        loss_generator=None,
        loss_feature=None,
        loss_mel=terms["loss_mel"],
        loss_duration=terms["loss_duration"],
        loss_kl=terms["loss_kl"],
    )
    adversarial_part = 0.5 * terms["loss_generator"] + 1.0 * terms["loss_feature"]
    # rel=1e-6 is float32 round-off on a sum near 20, not a modelling tolerance.
    assert float(total - adversarial_part) == pytest.approx(float(reconstruction), rel=1e-6)


def test_a_fully_gated_objective_equals_the_reconstruction_only_sum() -> None:
    settings = options()
    terms = loss_terms()
    total = _generator_objective(
        settings,
        adversarial_weight=0.0,
        loss_generator=None,
        loss_feature=None,
        loss_mel=terms["loss_mel"],
        loss_duration=terms["loss_duration"],
        loss_kl=terms["loss_kl"],
    )
    expected = (
        settings.mel_loss_weight * terms["loss_mel"]
        + settings.duration_loss_weight * terms["loss_duration"]
        + settings.kl_loss_weight * terms["loss_kl"]
    )
    assert torch.equal(total, expected)


def test_the_stft_and_proximal_terms_are_added_only_when_supplied() -> None:
    settings = options(stft_loss_weight=2.5, decoder_proximal_weight=0.125)
    terms = loss_terms()
    base = _generator_objective(settings, adversarial_weight=1.0, **terms)
    loss_stft = torch.tensor(0.375)
    loss_proximal = torch.tensor(4.0)
    assert torch.equal(
        _generator_objective(settings, adversarial_weight=1.0, loss_stft=loss_stft, **terms),
        base + settings.stft_loss_weight * loss_stft,
    )
    assert torch.equal(
        _generator_objective(settings, adversarial_weight=1.0, loss_proximal=loss_proximal, **terms),
        base + settings.decoder_proximal_weight * loss_proximal,
    )
    both = _generator_objective(
        settings,
        adversarial_weight=1.0,
        loss_stft=loss_stft,
        loss_proximal=loss_proximal,
        **terms,
    )
    assert torch.equal(
        both,
        base
        + settings.stft_loss_weight * loss_stft
        + settings.decoder_proximal_weight * loss_proximal,
    )
    # Both weights are positive here, so an unconditional term would show up.
    assert torch.equal(base, _generator_objective(settings, adversarial_weight=1.0, **terms))


@pytest.mark.parametrize(
    ("step", "scale"),
    [(3_000, 0.0), (3_150, 0.5), (3_300, 1.0), (4_000, 1.0)],
)
def test_the_decoder_rate_warms_up_linearly_from_the_unfreeze_step(step, scale) -> None:
    settings = options(decoder_unfreeze_step=3_000, decoder_lr_warmup_steps=300)
    assert _decoder_lr_scale(settings, step, STAGE_DECODER) == scale


@pytest.mark.parametrize("stage", [STAGE_POSTERIOR, STAGE_ADAPT])
def test_the_decoder_rate_is_unscaled_outside_the_polish_stage(stage) -> None:
    """The scale only means anything once the decoder group is trainable."""

    settings = options(decoder_unfreeze_step=3_000, decoder_lr_warmup_steps=300)
    assert _decoder_lr_scale(settings, 3_150, stage) == 1.0


@pytest.mark.parametrize("step", [0, 3_000, 3_150, 20_000])
def test_a_disabled_warmup_leaves_the_decoder_rate_unscaled(step) -> None:
    settings = options(decoder_unfreeze_step=3_000, decoder_lr_warmup_steps=0)
    assert _decoder_lr_scale(settings, step, STAGE_DECODER) == 1.0


def test_the_warmup_scale_leaves_the_exponential_decay_chain_and_the_saved_rate_intact() -> None:
    """The warm-up is applied for one step and taken back off again.

    ExponentialLR is chainable: it reads the live rate and multiplies it. If
    the scale stayed on the group it would compound into the decay and be
    written into the checkpoint as the nominal rate. So a warmed run and an
    unwarmed one must end on the same nominal rate.
    """

    warmed = _toy_warmup_run(4)
    plain = _toy_warmup_run(0)
    # Measured identical; 1e-18 is margin on rates near 3.5e-4, not slack.
    assert warmed["nominal_lr"] == pytest.approx(plain["nominal_lr"], abs=1e-18)
    assert plain["applied"][0] == 1.0e-3
    assert warmed["applied"][0] == 0.0
    assert warmed["applied"][1] == pytest.approx(0.25 * plain["applied"][1], rel=1e-12)
    assert warmed["applied"][2] == pytest.approx(0.5 * plain["applied"][2], rel=1e-12)
    # From the warm-up step onward the scale is 1.0 and nothing is touched.
    assert warmed["applied"][4:] == plain["applied"][4:]
    # A rate of exactly zero is the point of the first step: Adam's moments
    # fill and the weights do not move.
    assert warmed["moved"][0] is False
    assert plain["moved"][0] is True
    assert all(warmed["moved"][1:])


def test_a_scale_of_one_touches_no_optimizer_group() -> None:
    _, optimizer = _decoder_optimizer()
    group = optimizer.param_groups[0]
    before = {key: value for key, value in group.items() if key != "params"}
    with _scaled_decoder_lr(optimizer, 1.0):
        assert {key: value for key, value in group.items() if key != "params"} == before
    assert {key: value for key, value in group.items() if key != "params"} == before


def test_the_decoder_rate_is_restored_even_when_the_step_raises() -> None:
    _, optimizer = _decoder_optimizer()
    group = optimizer.param_groups[0]
    with pytest.raises(RuntimeError, match="optimizer step failed"), _scaled_decoder_lr(
        optimizer, 0.25
    ):
        assert group["lr"] == 0.25e-3
        raise RuntimeError("optimizer step failed")
    assert group["lr"] == 1.0e-3


def test_the_stft_loss_is_zero_for_identical_waveforms_and_positive_for_a_perturbed_one() -> None:
    clean = noise()
    assert float(_multi_resolution_stft_loss(clean, clean)) == 0.0
    assert float(_multi_resolution_stft_loss(clean + grid_comb(), clean)) > 0.0


def test_the_stft_loss_stays_float32_under_autocast_and_matches_the_uncast_value() -> None:
    """A half-precision window makes torch.stft return complex32.

    The magnitudes would then carry more quantisation than the comb being
    measured, which is why the transform runs with autocast disabled.
    """

    clean = noise()
    combed = clean + grid_comb()
    plain = _multi_resolution_stft_loss(combed, clean)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = _multi_resolution_stft_loss(combed, clean)
    assert cast.dtype is torch.float32
    assert torch.equal(cast, plain)


def test_a_segment_shorter_than_every_resolution_is_rejected_by_length() -> None:
    with pytest.raises(ValueError, match="segment 100 samples"):
        _multi_resolution_stft_loss(torch.zeros(1, 100), torch.zeros(1, 100))


def test_the_stft_loss_charges_more_for_a_frame_grid_comb_than_the_mel_l1_does() -> None:
    """The reason the term exists at all.

    An 80-band mel averages over bands hundreds of hertz wide in the top
    octaves, so a narrow comb there is nearly free under a mel L1. Measured
    ratio for a -34 dB comb on 0.05 RMS noise is 3.88 (mel 0.0103, STFT
    0.0399); pinned as an inequality at 2x because the exact ratio depends on
    the noise floor, not on anything the remedy promises.
    """

    clean = noise()
    combed = clean + grid_comb()
    reference = bundle()
    mel_penalty = F.l1_loss(
        _mel_from_waveform(combed, reference), _mel_from_waveform(clean, reference)
    )
    stft_penalty = _multi_resolution_stft_loss(combed, clean)
    assert float(stft_penalty) > 2.0 * float(mel_penalty)


def test_the_proximal_loss_is_zero_at_the_reference() -> None:
    model = _GeneratorStub()
    reference = _decoder_reference(model)
    assert float(_proximal_loss(model, reference).detach()) == 0.0


def test_the_proximal_loss_is_one_per_doubled_tensor() -> None:
    """Relative drift: a doubled tensor has drifted by its own norm, exactly."""

    model = _GeneratorStub()
    reference = _decoder_reference(model)
    decoder_tensors = [name for name in reference if name.startswith("dec.")]
    assert len(decoder_tensors) == 8
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in reference:
                parameter.mul_(2.0)
    assert float(_proximal_loss(model, reference).detach()) == float(len(decoder_tensors))


def test_the_proximal_loss_is_insensitive_to_the_scale_of_the_anchor() -> None:
    """Normalising by each tensor's own norm is what makes the term steer.

    An unnormalised mean over millions of elements reads as satisfied for a
    drift that ruins the render, so the value must depend on the relative
    drift and not on how large the released weights happen to be.
    """

    model = _GeneratorStub()
    reference = _decoder_reference(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in reference:
                parameter.add_(0.25 * torch.ones_like(parameter))
    drift = float(_proximal_loss(model, reference).detach())
    assert drift > 0.0
    rescaled = {name: value * 4.0 for name, value in reference.items()}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in reference:
                parameter.mul_(4.0)
    # 4 is a power of two, so the rescaling is exact in float32 and so is this.
    assert float(_proximal_loss(model, rescaled).detach()) == drift


def test_the_proximal_loss_skips_frozen_parameters() -> None:
    """It measures only what the optimizer can actually move.

    With the upsamplers held, the anchor must not charge for tensors that
    cannot drift, or the term reports a constant the run cannot reduce.
    """

    model = _GeneratorStub()
    reference = _decoder_reference(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in reference:
                parameter.mul_(2.0)
            if name.startswith(("dec.ups.", "dec.conv_pre.")):
                parameter.requires_grad_(False)
    # Four of the eight decoder tensors are dec.ups.0.* and dec.conv_pre.*.
    assert float(_proximal_loss(model, reference).detach()) == 4.0


def test_the_proximal_loss_refuses_a_decoder_with_nothing_trainable() -> None:
    model = _GeneratorStub()
    reference = _decoder_reference(model)
    for name, parameter in model.named_parameters():
        if name in reference:
            parameter.requires_grad_(False)
    with pytest.raises(RuntimeError, match="no decoder parameter"):
        _proximal_loss(model, reference)


def _masked_latents() -> tuple[torch.Tensor, torch.Tensor]:
    """A two-item batch whose second item is half padding.

    Item 0 is 0.5 everywhere across 10 frames; item 1 is 2.0 for five frames
    and then padding carrying 7.0, which the mask must discard.
    """

    z = torch.zeros(2, 4, 10)
    z[0] = 0.5
    z[1, :, :5] = 2.0
    z[1, :, 5:] = 7.0
    mask = torch.ones(2, 1, 10)
    mask[1, :, 5:] = 0.0
    return z, mask


def test_latent_statistics_match_the_hand_computed_values_on_a_masked_batch() -> None:
    z, mask = _masked_latents()
    statistics = _latent_statistics(z, mask)
    # Per-channel time means are 0.5 (item 0) and 2.0 (item 1), eight in all,
    # so z_dc_rms = sqrt((4 * 0.25 + 4 * 4) / 8) = sqrt(17 / 8).
    assert statistics["z_dc_rms"] == pytest.approx(math.sqrt(17.0 / 8.0), rel=1e-6)
    # Masked energy is 40 * 0.25 + 20 * 4 = 90 over 15 frames x 4 channels.
    assert statistics["z_rms"] == pytest.approx(math.sqrt(1.5), rel=1e-6)
    assert set(statistics) == {"z_dc_rms", "z_rms"}


def test_latent_statistics_ignore_the_padded_frames() -> None:
    """Padding must not enter the drift reading the run is judged on."""

    z, mask = _masked_latents()
    louder = z.clone()
    louder[1, :, 5:] = 100.0
    assert _latent_statistics(louder, mask) == _latent_statistics(z, mask)


class _EmaStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.register_buffer("updates", torch.tensor(3, dtype=torch.long))


def test_the_ema_shadow_covers_the_whole_state_dict() -> None:
    model = _EmaStub()
    shadow = _ema_state(model)
    assert set(shadow) == set(model.state_dict())
    assert shadow["weight"].dtype is torch.float32
    assert shadow["updates"].dtype is torch.int64


@pytest.mark.parametrize(
    ("updates", "decay"),
    [(1, 2.0 / 11.0), (10, 0.55), (100, 101.0 / 110.0), (100_000, 0.999)],
)
def test_the_ema_decay_ramps_in_and_caps_at_the_configured_value(updates, decay) -> None:
    """Without the ramp the average stays anchored to the weights at step 0."""

    assert _ema_decay(0.999, updates) == pytest.approx(decay, rel=1e-12)
    assert _ema_decay(0.999, updates) <= 0.999


def test_an_ema_update_moves_the_shadow_by_exactly_the_decay() -> None:
    model = _EmaStub()
    shadow = _ema_state(model)
    with torch.no_grad():
        model.weight.fill_(3.0)
    _ema_update(shadow, model, 0.5)
    # 0.5 * 1.0 + 0.5 * 3.0, exact in binary floating point.
    assert torch.equal(shadow["weight"], torch.full((2,), 2.0))


def test_an_integer_buffer_is_copied_into_the_ema_rather_than_averaged() -> None:
    """Averaging a step counter would give a value it never held."""

    model = _EmaStub()
    shadow = _ema_state(model)
    with torch.no_grad():
        model.updates.fill_(9)
    _ema_update(shadow, model, 0.5)
    assert shadow["updates"].dtype is torch.int64
    assert torch.equal(shadow["updates"], torch.tensor(9, dtype=torch.long))


@pytest.mark.parametrize(
    ("mode", "stage", "enabled"),
    [
        ("adversarial", STAGE_POSTERIOR, {"posterior"}),
        ("adversarial", STAGE_ADAPT, {"posterior", "linguistic"}),
        ("adversarial", STAGE_DECODER, {"posterior", "linguistic", "decoder"}),
        ("recon", STAGE_POSTERIOR, {"posterior"}),
        ("recon", STAGE_ADAPT, {"posterior", "linguistic"}),
        ("recon", STAGE_DECODER, {"decoder"}),
    ],
)
def test_the_enabled_groups_follow_the_stage_and_the_polish_mode(mode, stage, enabled) -> None:
    settings = options(decoder_polish_mode=mode, decoder_unfreeze_step=3_000)
    assert _enabled_groups(settings, stage) == enabled


def test_an_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown training stage"):
        _enabled_groups(options(), "decoder_polish_v2")


def test_recon_polish_trains_only_the_decoder_group() -> None:
    """Nothing may move the latents while the decoder is asked to render them."""

    model = _GeneratorStub()
    settings = options(
        decoder_polish_mode="recon", decoder_unfreeze_step=3_000, learning_rate_g=1.0e-4
    )
    optimizer = torch.optim.AdamW(_generator_groups(model, settings), lr=0.0)
    _apply_stage(model, optimizer, settings, STAGE_DECODER)
    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert rates["posterior"] == 0.0
    assert rates["linguistic"] == 0.0
    assert rates["decoder"] == settings.learning_rate_g * settings.decoder_lr_multiplier
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {name for name in dict(model.named_parameters()) if name.startswith("dec.")}


def test_apply_stage_zeroes_the_rate_of_every_disabled_group() -> None:
    model = _GeneratorStub()
    settings = options(learning_rate_g=1.0e-4)
    optimizer = torch.optim.AdamW(_generator_groups(model, settings), lr=0.0)
    _apply_stage(model, optimizer, settings, STAGE_POSTERIOR)
    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert rates["posterior"] == settings.learning_rate_g * settings.posterior_lr_multiplier
    assert rates["linguistic"] == 0.0
    assert rates["decoder"] == 0.0
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"enc_q.weight", "enc_q.bias"}


def test_freezing_the_upsamplers_keeps_three_optimizer_groups() -> None:
    """The freeze is a gradient mask, not a fourth parameter group.

    A fourth group would change the shape of the saved optimizer state, so a
    checkpoint written with the option on could not be resumed with it off.
    """

    model = _GeneratorStub()
    settings = options(decoder_freeze_upsamplers=True, learning_rate_g=1.0e-4)
    groups = _generator_groups(model, settings)
    optimizer = torch.optim.AdamW(groups, lr=0.0)
    _apply_stage(model, optimizer, settings, STAGE_DECODER)
    assert len(optimizer.param_groups) == 3
    assert [group["name"] for group in optimizer.param_groups] == [
        "posterior",
        "linguistic",
        "decoder",
    ]
    decoder_group = next(g for g in optimizer.param_groups if g["name"] == "decoder")
    assert decoder_group["lr"] == settings.learning_rate_g * settings.decoder_lr_multiplier
    held = {name for name, p in model.named_parameters() if not p.requires_grad}
    assert held == {
        "dec.conv_pre.weight",
        "dec.conv_pre.bias",
        "dec.ups.0.weight",
        "dec.ups.0.bias",
    }
    # The rest of the decoder still trains inside the same group.
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(("dec.resblocks.", "dec.conv_post."))
    )


def test_the_upsamplers_train_when_the_freeze_is_off() -> None:
    model = _GeneratorStub()
    settings = options(decoder_freeze_upsamplers=False, learning_rate_g=1.0e-4)
    optimizer = torch.optim.AdamW(_generator_groups(model, settings), lr=0.0)
    _apply_stage(model, optimizer, settings, STAGE_DECODER)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_the_mel_projection_stays_float32_under_autocast_and_matches_the_uncast_value() -> None:
    """The mel target is quantised by a half-precision matmul, not just the
    prediction, so both sides of the L1 would move under autocast."""

    reference = bundle()
    generator = torch.Generator().manual_seed(11)
    spec = torch.rand(1, 513, 20, generator=generator).abs() + 1.0e-3
    plain = _mel_from_spec(spec, reference)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = _mel_from_spec(spec, reference)
    assert cast.dtype is torch.float32
    assert torch.equal(cast, plain)


def test_the_waveform_stft_stays_float32_under_autocast_and_matches_the_uncast_value() -> None:
    reference = bundle()
    waveform = noise(seconds=0.1)
    plain = _mel_from_waveform(waveform, reference)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = _mel_from_waveform(waveform, reference)
    assert cast.dtype is torch.float32
    assert torch.equal(cast, plain)


def test_scalar_maps_none_to_none_and_a_tensor_to_a_float() -> None:
    """A term the schedule switched off is logged as null, never as 0.0."""

    assert _scalar(None) is None
    value = _scalar(torch.tensor(0.5))
    assert value == 0.5
    assert isinstance(value, float)


# The reconstruction loss compares a target spectrogram the dataset produced
# against a spectrogram of the generated waveform. Those two were computed by
# different code with different magnitude floors, 1e-3 against 1e-5, so two
# identical waveforms scored ln(100) and the gradient in every quiet cell paid
# for filling it with noise. These tests pin the invariant whose absence let
# that survive from the toolkit's first commit.


def target_mel(waveform: torch.Tensor, reference) -> torch.Tensor:
    """The target side, through the dataset's own transform."""

    config = AudioConfig(
        sampling_rate=SAMPLE_RATE,
        filter_length=MEL_DATA["filter_length"],
        hop_length=HOP_LENGTH,
        win_length=MEL_DATA["win_length"],
        add_blank=True,
    )
    return _mel_from_spec(spectrogram(waveform, config)[None], reference)


def mel_l1(target: torch.Tensor, generated: torch.Tensor, **kwargs) -> float:
    reference = bundle()
    return float(
        torch.nn.functional.l1_loss(
            target_mel(target, reference).float(),
            _mel_from_waveform(generated[None], reference, **kwargs).float(),
        )
    )


@pytest.mark.parametrize(
    "signal",
    [
        torch.zeros(SAMPLE_RATE),
        0.1 * torch.sin(2 * torch.pi * 220.0 * torch.arange(SAMPLE_RATE) / SAMPLE_RATE),
        0.001 * torch.sin(2 * torch.pi * 220.0 * torch.arange(SAMPLE_RATE) / SAMPLE_RATE),
        noise(seconds=1.0)[0],
    ],
    ids=["silence", "loud-tone", "quiet-tone", "noise"],
)
def test_the_mel_loss_is_zero_for_identical_waveforms(signal: torch.Tensor) -> None:
    """The invariant whose absence hid a ln(100) bias for the toolkit's whole life."""

    assert mel_l1(signal, signal) == 0.0


def test_the_legacy_floor_scores_identical_waveforms_as_different() -> None:
    """What the failed runs trained against, kept only to make the two comparable.

    Silence lands on ln(100) exactly, because that is the ratio of the two
    floors and it survives the shared bank and the shared logarithm.
    """

    silence = torch.zeros(SAMPLE_RATE)
    assert mel_l1(silence, silence, legacy_floor=True) == pytest.approx(math.log(100.0), abs=1e-5)
    tone = 0.1 * torch.sin(2 * torch.pi * 220.0 * torch.arange(SAMPLE_RATE) / SAMPLE_RATE)
    assert mel_l1(tone, tone, legacy_floor=True) > 2.0


def test_the_mel_loss_rises_with_injected_noise_instead_of_falling() -> None:
    """Against a silent target the legacy floor has a minimum away from zero.

    Measured: 4.605 at no noise, falling to 0.543 at 1e-4 RMS. The unified
    transform is monotone from zero, which is what a reconstruction loss has
    to be for the thing it reconstructs to be the optimum.
    """

    torch.manual_seed(0)
    silence = torch.zeros(SAMPLE_RATE)
    unit = torch.randn(SAMPLE_RATE)
    levels = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    unified = [mel_l1(silence, level * unit) for level in levels]
    legacy = [mel_l1(silence, level * unit, legacy_floor=True) for level in levels]

    assert unified[0] == 0.0
    assert unified == sorted(unified)
    # The legacy objective prefers a noise floor to silence.
    assert min(legacy) < legacy[0]
    assert legacy[legacy.index(min(legacy))] < 1.0


def test_the_mel_gradient_points_away_from_noise() -> None:
    """The sign of the gradient is the part that steers training.

    Under the legacy floor a quiet existing noise is paid to grow.
    """

    torch.manual_seed(0)
    silence = torch.zeros(SAMPLE_RATE)
    unit = torch.randn(SAMPLE_RATE)

    def slope(**kwargs) -> float:
        amplitude = torch.tensor(1e-6, requires_grad=True)
        reference = bundle()
        loss = torch.nn.functional.l1_loss(
            target_mel(silence, reference).float(),
            _mel_from_waveform((amplitude * unit)[None], reference, **kwargs).float(),
        )
        loss.backward()
        return float(amplitude.grad)

    assert slope(legacy_floor=True) < 0.0
    assert slope() > 0.0


def test_the_shared_magnitude_spectrogram_is_the_dataset_transform() -> None:
    """One formula, one place, and the batch form is the single form stacked."""

    config = AudioConfig(
        sampling_rate=SAMPLE_RATE,
        filter_length=MEL_DATA["filter_length"],
        hop_length=HOP_LENGTH,
        win_length=MEL_DATA["win_length"],
        add_blank=True,
    )
    sizes = {"n_fft": MEL_DATA["filter_length"], "hop_length": HOP_LENGTH,
             "win_length": MEL_DATA["win_length"]}
    single = noise(seconds=0.4)[0]
    assert torch.equal(spectrogram(single, config), magnitude_spectrogram(single, **sizes))

    batch = torch.cat([noise(seconds=0.4), 0.2 * noise(seconds=0.4)])
    stacked = torch.stack([magnitude_spectrogram(row, **sizes) for row in batch])
    assert torch.equal(magnitude_spectrogram(batch, **sizes), stacked)

    # The floor is added in the power domain, so silence lands on its square root.
    floor = magnitude_spectrogram(torch.zeros(SAMPLE_RATE), **sizes)
    assert torch.allclose(floor, torch.full_like(floor, MAGNITUDE_FLOOR_POWER**0.5))
