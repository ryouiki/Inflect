"""Generic staged warm-start training for Inflect v2 adaptation."""

from __future__ import annotations

import json
import logging
import random
import uuid
from contextlib import contextmanager
from dataclasses import MISSING, asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import (
    CompatibilityReport,
    build_run_identity,
    cpu_compatibility_report,
    load_posterior_sidecar,
    load_run_identity,
    resume_training_checkpoint,
    save_inference_checkpoint,
    save_training_checkpoint,
    validate_run_identity,
    write_run_identity,
)
from .grid_screens import grid_comb_metrics
from .modeling import (
    ModelBundle,
    build_training_models,
    load_symbols,
    optimizer_parameters,
    resolve_base_model,
)
from .presets import available_presets, load_packaged_preset
from .training_data import AudioConfig, PreparedTTSDataset, create_dataloader


LOGGER = logging.getLogger("inflect_finetune")
STAGE_POSTERIOR = "posterior_warmup"
STAGE_ADAPT = "linguistic_adaptation"
STAGE_DECODER = "decoder_polish"
STAGES = (STAGE_POSTERIOR, STAGE_ADAPT, STAGE_DECODER)
DECODER_POLISH_MODES = ("adversarial", "recon")
POSTERIOR_INITS = ("fresh", "inherit")
FROZEN_UPSAMPLER_PREFIXES = ("dec.ups.", "dec.conv_pre.")
# Linear-frequency resolutions as (n_fft, hop). The 1024/256 pair matches the
# model's own analysis grid; 2048 gives 11.7 Hz bins at 24 kHz, fine enough to
# resolve a comb at multiples of the frame rate that an 80-band mel averages
# away; 512 keeps a short window for transients.
STFT_RESOLUTIONS = ((512, 128), (1024, 256), (2048, 512))


@dataclass(frozen=True)
class TrainingOptions:
    """Public, generic adaptation settings.

    These defaults are intentionally conservative starting points, not the
    private schedule used to create an Inflect release checkpoint.
    """

    base_model: str | Path
    prepared_dir: str | Path
    output_dir: str | Path
    preset: str | Path | None = None
    resume: str | Path | None = None
    device: str = "auto"
    seed: int = 1234
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_workers: int = 2
    max_steps: int = 20_000
    learning_rate_g: float = 8.0e-5
    learning_rate_d: float = 8.0e-5
    posterior_lr_multiplier: float = 1.0
    linguistic_lr_multiplier: float = 0.5
    decoder_lr_multiplier: float = 0.1
    posterior_warmup_steps: int = 500
    decoder_unfreeze_step: int | None = 3_000
    amp: bool = True
    adam_betas: tuple[float, float] = (0.8, 0.99)
    adam_eps: float = 1.0e-9
    weight_decay: float = 0.0
    lr_decay: float = 0.99999
    max_grad_norm: float = 10.0
    mel_loss_weight: float = 45.0
    kl_loss_weight: float = 1.0
    feature_loss_weight: float = 1.0
    duration_loss_weight: float = 1.0
    checkpoint_interval: int = 1_000
    validation_interval: int = 500
    log_interval: int = 25
    validation_seed: int = 7
    # Every field below defaults to the behaviour of the runs that preceded it,
    # so a caller that sets none of them gets the same loss and the same
    # schedule. They exist because those runs produced a steady comb at
    # multiples of the frame rate; docs/TROUBLESHOOTING.md explains when to
    # reach for which.
    adversarial_gating: bool = False
    adversarial_ramp_steps: int = 1_000
    decoder_lr_warmup_steps: int = 0
    decoder_polish_mode: str = "adversarial"
    stft_loss_weight: float = 0.0
    decoder_proximal_weight: float = 0.0
    decoder_freeze_upsamplers: bool = False
    posterior_init: str = "fresh"
    generator_ema_decay: float = 0.0

    @classmethod
    def from_preset(
        cls,
        preset: str | Path,
        *,
        base_model: str | Path,
        prepared_dir: str | Path,
        output_dir: str | Path,
        **overrides: Any,
    ) -> "TrainingOptions":
        payload = load_preset(preset)
        payload.update(overrides)
        return cls(
            base_model=base_model,
            prepared_dir=prepared_dir,
            output_dir=output_dir,
            # The preset and caller overrides are already merged above.
            preset=None,
            **payload,
        )

    def resolved(self) -> "TrainingOptions":
        if self.preset is None:
            return self
        payload = load_preset(self.preset)
        defaults = {
            field.name: field.default
            for field in fields(self)
            if field.default is not MISSING
        }
        explicit = asdict(self)
        # Dataclass values differing from defaults are treated as explicit
        # caller overrides; required paths are always preserved.
        for key, value in explicit.items():
            if key in {"base_model", "prepared_dir", "output_dir", "preset", "resume"}:
                continue
            if key not in defaults or value != defaults[key]:
                payload[key] = value
        allowed = {field.name for field in fields(self)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Preset contains unknown TrainingOptions: {unknown}")
        return replace(self, **payload, preset=None)


@dataclass
class TrainingState:
    step: int = 0
    epoch: int = 0
    stage: str = STAGE_POSTERIOR


def load_preset(preset: str | Path) -> dict[str, Any]:
    source = Path(preset)
    if source.is_file():
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        name = str(preset)
        try:
            payload = load_packaged_preset(name)
        except KeyError as error:
            raise FileNotFoundError(
                f"Unknown training preset {preset!r}. "
                f"Available: {list(available_presets())}"
            ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"Training preset {preset!r} must contain a JSON object.")
    return payload


def _validate_options(options: TrainingOptions) -> None:
    positive_ints = (
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "checkpoint_interval",
        "validation_interval",
        "log_interval",
    )
    for name in positive_ints:
        if int(getattr(options, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    if options.posterior_warmup_steps < 0:
        raise ValueError("posterior_warmup_steps must be non-negative.")
    if (
        options.decoder_unfreeze_step is not None
        and options.decoder_unfreeze_step < options.posterior_warmup_steps
    ):
        raise ValueError("decoder_unfreeze_step cannot precede posterior_warmup_steps.")
    for name in (
        "learning_rate_g",
        "learning_rate_d",
        "posterior_lr_multiplier",
        "linguistic_lr_multiplier",
        "decoder_lr_multiplier",
        "lr_decay",
    ):
        if float(getattr(options, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    for name in ("adversarial_ramp_steps", "decoder_lr_warmup_steps"):
        if int(getattr(options, name)) < 0:
            raise ValueError(f"{name} must be non-negative.")
    for name in ("stft_loss_weight", "decoder_proximal_weight"):
        if float(getattr(options, name)) < 0:
            raise ValueError(f"{name} must be non-negative.")
    if options.decoder_polish_mode not in DECODER_POLISH_MODES:
        raise ValueError(
            f"decoder_polish_mode must be one of {list(DECODER_POLISH_MODES)}."
        )
    if options.posterior_init not in POSTERIOR_INITS:
        raise ValueError(f"posterior_init must be one of {list(POSTERIOR_INITS)}.")
    if not 0.0 <= float(options.generator_ema_decay) < 1.0:
        raise ValueError("generator_ema_decay must be at least 0 and below 1.")
    if options.decoder_unfreeze_step is None:
        # Both settings are anchored to the unfreeze step. Accepting them
        # against a decoder that never unfreezes would silently do nothing.
        if options.decoder_polish_mode != "adversarial":
            raise ValueError(
                "decoder_polish_mode is only reachable when the decoder unfreezes; "
                "set decoder_unfreeze_step or leave the mode at 'adversarial'."
            )
        if options.decoder_lr_warmup_steps > 0:
            raise ValueError(
                "decoder_lr_warmup_steps is measured from decoder_unfreeze_step, "
                "which is disabled."
            )
    if options.adversarial_gating and options.decoder_unfreeze_step is None:
        LOGGER.warning(
            "adversarial_gating with no decoder_unfreeze_step trains the "
            "discriminator without ever using it; the generator never sees an "
            "adversarial gradient."
        )


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stage_for_step(options: TrainingOptions, step: int) -> str:
    if step < options.posterior_warmup_steps:
        return STAGE_POSTERIOR
    if options.decoder_unfreeze_step is not None and step >= options.decoder_unfreeze_step:
        return STAGE_DECODER
    return STAGE_ADAPT


def _generator_groups(model: nn.Module, options: TrainingOptions) -> list[dict[str, Any]]:
    named = dict(model.named_parameters())
    assignments: dict[str, list[nn.Parameter]] = {
        "posterior": [],
        "linguistic": [],
        "decoder": [],
    }
    for name, parameter in named.items():
        if name.startswith("enc_q."):
            assignments["posterior"].append(parameter)
        elif name.startswith("dec."):
            assignments["decoder"].append(parameter)
        elif name.startswith(("enc_p.", "dp.", "flow.")):
            assignments["linguistic"].append(parameter)
        else:
            raise RuntimeError(f"Unclassified generator parameter {name!r}.")
    if any(not values for values in assignments.values()):
        empty = [name for name, values in assignments.items() if not values]
        raise RuntimeError(f"Generator parameter groups are unexpectedly empty: {empty}")
    multipliers = {
        "posterior": options.posterior_lr_multiplier,
        "linguistic": options.linguistic_lr_multiplier,
        "decoder": options.decoder_lr_multiplier,
    }
    return [
        {
            "params": parameters,
            "name": name,
            "lr_multiplier": multipliers[name],
            "lr": 0.0,
        }
        for name, parameters in assignments.items()
    ]


def _enabled_groups(options: TrainingOptions, stage: str) -> set[str]:
    if stage not in STAGES:
        raise ValueError(f"Unknown training stage {stage!r}.")
    if stage == STAGE_DECODER and options.decoder_polish_mode == "recon":
        # Reconstruction polish asks one question: can the decoder render the
        # latents it is already given cleanly. Letting the posterior and the
        # flow keep moving would change those latents at the same time and
        # answer nothing.
        return {"decoder"}
    return {
        STAGE_POSTERIOR: {"posterior"},
        STAGE_ADAPT: {"posterior", "linguistic"},
        STAGE_DECODER: {"posterior", "linguistic", "decoder"},
    }[stage]


def _apply_stage(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    options: TrainingOptions,
    stage: str,
    *,
    reset_learning_rates: bool = True,
) -> None:
    enabled = _enabled_groups(options, stage)
    held: set[int] = set()
    if options.decoder_freeze_upsamplers:
        # The transposed convolutions are where the frame grid enters the
        # waveform. Holding them while the residual stack trains keeps the
        # optimizer's parameter groups, and so its saved state, the same shape
        # as a run that does not use the option.
        held = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if name.startswith(FROZEN_UPSAMPLER_PREFIXES)
        }
    for group in optimizer.param_groups:
        name = group["name"]
        active = name in enabled
        if not active:
            group["lr"] = 0.0
        elif reset_learning_rates:
            group["lr"] = options.learning_rate_g * float(group["lr_multiplier"])
        for parameter in group["params"]:
            trainable = active and id(parameter) not in held
            parameter.requires_grad_(trainable)
            if not trainable:
                parameter.grad = None
    optimizer.zero_grad(set_to_none=True)


def _adversarial_weight(options: TrainingOptions, step: int, stage: str) -> float:
    """Weight applied to the adversarial and feature-matching terms.

    A fresh discriminator meets a pretrained generator whose decoder is frozen
    for thousands of steps. Its gradients still reach the posterior encoder and
    the flow through that frozen decoder, and nothing else constrains where the
    latents go. Gating holds the term at exactly zero until the decoder can
    respond, then ramps it so the first real adversarial push is not a step
    change. The discriminator keeps training throughout: the gated window is
    its warm-up.
    """

    if stage == STAGE_DECODER and options.decoder_polish_mode == "recon":
        return 0.0
    if not options.adversarial_gating:
        return 1.0
    unfreeze = options.decoder_unfreeze_step
    if unfreeze is None or step < unfreeze:
        return 0.0
    if options.adversarial_ramp_steps <= 0:
        return 1.0
    return min(1.0, (step - unfreeze) / options.adversarial_ramp_steps)


def _decoder_lr_scale(options: TrainingOptions, step: int, stage: str) -> float:
    """Multiplier easing the decoder in after it unfreezes.

    Adam starts the decoder group with empty moment estimates, so without a
    warm-up its first update is the largest one it will ever take, on the
    component whose released weights are the most valuable thing in the run. A
    scale of zero on the first step is deliberate: the moments fill and the
    weights do not move.
    """

    if options.decoder_lr_warmup_steps <= 0 or stage != STAGE_DECODER:
        return 1.0
    unfreeze = options.decoder_unfreeze_step
    if unfreeze is None:
        return 1.0
    return min(1.0, (step - unfreeze) / options.decoder_lr_warmup_steps)


def _discriminator_active(options: TrainingOptions, stage: str) -> bool:
    return not (stage == STAGE_DECODER and options.decoder_polish_mode == "recon")


@contextmanager
def _scaled_decoder_lr(optimizer: torch.optim.Optimizer, scale: float):
    """Apply a warm-up scale for one optimizer step only.

    The scheduler is chainable: it reads the live learning rate and multiplies
    it. Scaling in place would compound into the decay, and would be written
    into the checkpoint as if it were the nominal rate. Restoring before the
    scheduler runs keeps both honest.
    """

    if scale >= 1.0:
        yield
        return
    groups = [group for group in optimizer.param_groups if group["name"] == "decoder"]
    saved = [group["lr"] for group in groups]
    for group in groups:
        group["lr"] = group["lr"] * scale
    try:
        yield
    finally:
        for group, learning_rate in zip(groups, saved):
            group["lr"] = learning_rate


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _discriminator_loss(
    real_outputs: Iterable[torch.Tensor], generated_outputs: Iterable[torch.Tensor]
) -> torch.Tensor:
    return sum(
        torch.mean((1.0 - real.float()) ** 2) + torch.mean(generated.float() ** 2)
        for real, generated in zip(real_outputs, generated_outputs)
    )


def _generator_loss(outputs: Iterable[torch.Tensor]) -> torch.Tensor:
    return sum(torch.mean((1.0 - output.float()) ** 2) for output in outputs)


def _feature_loss(
    real_maps: Iterable[Iterable[torch.Tensor]],
    generated_maps: Iterable[Iterable[torch.Tensor]],
) -> torch.Tensor:
    return 2.0 * sum(
        torch.mean(torch.abs(real.float().detach() - generated.float()))
        for real_group, generated_group in zip(real_maps, generated_maps)
        for real, generated in zip(real_group, generated_group)
    )


def _kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    value = logs_p.float() - logs_q.float() - 0.5
    value += 0.5 * ((z_p.float() - m_p.float()) ** 2) * torch.exp(-2.0 * logs_p.float())
    return torch.sum(value * mask.float()) / torch.sum(mask.float()).clamp_min(1.0)


def _multi_resolution_stft_loss(
    generated: torch.Tensor,
    real: torch.Tensor,
    resolutions: Iterable[tuple[int, int]] = STFT_RESOLUTIONS,
) -> torch.Tensor:
    """Linear-frequency reconstruction error the mel loss cannot see.

    An 80-band mel averages over bands that are hundreds of hertz wide in the
    top octaves, so a narrow comb sitting there costs almost nothing under a
    mel L1. Each resolution contributes spectral convergence plus a
    log-magnitude L1, and the result is the mean over resolutions, following
    Parallel WaveGAN. A weight quoted for the summed convention is worth three
    times as much here.

    Autocast is disabled inside: a half-precision window makes the transform
    return complex32, and the magnitudes then carry more quantisation than the
    artifact being measured.
    """

    with torch.autocast(device_type=generated.device.type, enabled=False):
        predicted = generated.float()
        target = real.float()
        total = predicted.new_zeros(())
        counted = 0
        for n_fft, hop in resolutions:
            if predicted.shape[-1] < n_fft:
                continue
            window = torch.hann_window(
                n_fft, device=predicted.device, dtype=predicted.dtype
            )
            magnitudes = [
                torch.stft(
                    signal,
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=n_fft,
                    window=window,
                    center=True,
                    return_complex=True,
                )
                .abs()
                .clamp_min(1.0e-7)
                for signal in (predicted, target)
            ]
            predicted_magnitude, target_magnitude = magnitudes
            convergence = torch.linalg.norm(
                target_magnitude - predicted_magnitude
            ) / torch.linalg.norm(target_magnitude).clamp_min(1.0e-7)
            magnitude = F.l1_loss(
                torch.log(predicted_magnitude), torch.log(target_magnitude)
            )
            total = total + convergence + magnitude
            counted += 1
        if not counted:
            raise ValueError(
                "The training segment is shorter than every STFT resolution; "
                f"segment {predicted.shape[-1]} samples."
            )
        return total / counted


def _decoder_reference(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot the decoder as the run received it, for the proximal anchor."""

    return {
        name: parameter.detach().clone().float()
        for name, parameter in model.named_parameters()
        if name.startswith("dec.")
    }


def _proximal_loss(
    model: nn.Module, reference: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Relative squared drift of the decoder from its starting weights.

    Normalising each tensor's drift by that tensor's own squared norm is what
    makes the penalty steer: an unnormalised mean over millions of elements
    stays near zero even for a drift that ruins the render, so it reads as
    satisfied while nothing is being held. Frozen parameters are skipped so
    the term measures only what the optimizer can actually move.
    """

    total: torch.Tensor | None = None
    for name, parameter in model.named_parameters():
        anchor = reference.get(name)
        if anchor is None or not parameter.requires_grad:
            continue
        drift = ((parameter.float() - anchor) ** 2).sum() / (anchor**2).sum().clamp_min(
            1.0e-12
        )
        total = drift if total is None else total + drift
    if total is None:
        raise RuntimeError(
            "The decoder proximal term was requested while no decoder parameter "
            "is trainable."
        )
    return total


def _latent_statistics(z: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Scale of the latents the decoder is being fed.

    The released decoder was trained on latents whose per-channel time mean has
    an RMS near 0.74. Adaptations that ring reach 1.4 to 1.5, and handing those
    latents to the released decoder rings harder than handing them to the
    adapted one, which is what identified the latents rather than the decoder
    weights as the defect. Recording it every step makes the drift visible
    while the run is still cheap to abandon.
    """

    with torch.no_grad():
        value = z.float() * mask.float()
        frames = mask.float().sum(dim=-1).clamp_min(1.0)
        channel_mean = value.sum(dim=-1) / frames
        total_frames = mask.float().sum().clamp_min(1.0)
        return {
            "z_dc_rms": float(torch.sqrt(torch.mean(channel_mean**2))),
            "z_rms": float(
                torch.sqrt(value.pow(2).sum() / (total_frames * value.shape[1]))
            ),
        }


def _ema_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone().float()
        if value.is_floating_point()
        else value.detach().clone()
        for key, value in model.state_dict().items()
    }


def _ema_decay(decay: float, updates: int) -> float:
    """Ramp the horizon in so the average is not anchored to step zero."""

    return min(decay, (1.0 + updates) / (10.0 + updates))


def _ema_update(
    ema: Mapping[str, torch.Tensor], model: nn.Module, decay: float
) -> None:
    with torch.no_grad():
        for key, value in model.state_dict().items():
            stored = ema[key]
            if stored.is_floating_point():
                stored.mul_(decay).add_(value.detach().float(), alpha=1.0 - decay)
            else:
                stored.copy_(value.detach())


def _scalar(value: torch.Tensor | None) -> float | None:
    return None if value is None else float(value.detach().cpu())


def _generator_objective(
    options: TrainingOptions,
    *,
    adversarial_weight: float,
    loss_generator: torch.Tensor | None,
    loss_feature: torch.Tensor | None,
    loss_mel: torch.Tensor,
    loss_duration: torch.Tensor,
    loss_kl: torch.Tensor,
    loss_stft: torch.Tensor | None = None,
    loss_proximal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assemble the generator loss.

    The full-weight branch is written out term by term rather than factored,
    because floating-point addition is not associative: grouping the
    reconstruction terms would change the result of a default run in the last
    bits, and the point of the branch is that it does not.
    """

    if loss_generator is not None and loss_feature is not None and adversarial_weight == 1.0:
        total = (
            loss_generator
            + options.feature_loss_weight * loss_feature
            + options.mel_loss_weight * loss_mel
            + options.duration_loss_weight * loss_duration
            + options.kl_loss_weight * loss_kl
        )
    elif loss_generator is not None and loss_feature is not None and adversarial_weight > 0.0:
        total = (
            adversarial_weight * loss_generator
            + (adversarial_weight * options.feature_loss_weight) * loss_feature
            + options.mel_loss_weight * loss_mel
            + options.duration_loss_weight * loss_duration
            + options.kl_loss_weight * loss_kl
        )
    else:
        total = (
            options.mel_loss_weight * loss_mel
            + options.duration_loss_weight * loss_duration
            + options.kl_loss_weight * loss_kl
        )
    if loss_stft is not None:
        total = total + options.stft_loss_weight * loss_stft
    if loss_proximal is not None:
        total = total + options.decoder_proximal_weight * loss_proximal
    return total


def _hz_to_mel(value: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + value / 700.0)


def _mel_to_hz(value: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)


def _mel_filterbank(
    *,
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    minimum = _hz_to_mel(torch.tensor(fmin, dtype=torch.float64))
    maximum = _hz_to_mel(torch.tensor(fmax, dtype=torch.float64))
    points = _mel_to_hz(torch.linspace(minimum, maximum, n_mels + 2))
    bins = torch.floor((n_fft + 1) * points / sample_rate).long()
    frequencies = n_fft // 2 + 1
    bank = torch.zeros(n_mels, frequencies, dtype=torch.float32)
    for index in range(n_mels):
        left, center, right = (int(value) for value in bins[index : index + 3])
        center = max(center, left + 1)
        right = max(right, center + 1)
        for column in range(left, min(center, frequencies)):
            bank[index, column] = (column - left) / (center - left)
        for column in range(center, min(right, frequencies)):
            bank[index, column] = (right - column) / (right - center)
    return bank.to(device=device, dtype=dtype)


def _mel_from_spec(spec: torch.Tensor, bundle: ModelBundle) -> torch.Tensor:
    """Project a linear spectrogram onto the mel scale in full precision.

    Under mixed precision the projection is a half-precision matmul and the
    logarithm that follows quantises the target the loss is measured against,
    not just the prediction. Both sides of that comparison run here, so the
    surrounding cast is switched off for it.
    """

    data = bundle.config["data"]
    with torch.autocast(device_type=spec.device.type, enabled=False):
        value = spec.float()
        bank = _mel_filterbank(
            n_fft=int(data["filter_length"]),
            n_mels=int(data["n_mel_channels"]),
            sample_rate=int(data["sampling_rate"]),
            fmin=float(data["mel_fmin"]),
            fmax=float(data["mel_fmax"]),
            device=value.device,
            dtype=value.dtype,
        )
        return torch.log(torch.matmul(bank, value).clamp_min(1.0e-5))


def _mel_from_waveform(waveform: torch.Tensor, bundle: ModelBundle) -> torch.Tensor:
    data = bundle.config["data"]
    n_fft = int(data["filter_length"])
    hop = int(data["hop_length"])
    win = int(data["win_length"])
    padding = (n_fft - hop) // 2
    with torch.autocast(device_type=waveform.device.type, enabled=False):
        signal = waveform.float()
        padded = F.pad(signal.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        # A half-precision window makes this transform return complex32, which
        # costs more accuracy than the artifacts the loss is meant to charge for.
        window = torch.hann_window(win, device=padded.device, dtype=padded.dtype)
        spectrum = torch.stft(
            padded,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            center=False,
            return_complex=True,
        ).abs().clamp_min(1.0e-5)
        return _mel_from_spec(spectrum, bundle)


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _public_options(options: TrainingOptions) -> dict[str, Any]:
    payload = asdict(options)
    for key in ("base_model", "prepared_dir", "output_dir", "preset", "resume"):
        payload.pop(key, None)
    return json.loads(json.dumps(payload, sort_keys=True))


def _optimizer_schema(options: TrainingOptions) -> dict[str, Any]:
    common = {
        "class": "torch.optim.AdamW",
        "betas": list(options.adam_betas),
        "eps": options.adam_eps,
        "weight_decay": options.weight_decay,
    }
    return {
        "generator": {
            **common,
            "base_learning_rate": options.learning_rate_g,
            # The group list is deliberately fixed. Freezing the upsamplers is
            # expressed as a gradient mask inside the decoder group, because a
            # fourth group would change the shape of the saved optimizer state.
            "parameter_groups": [
                {
                    "name": "posterior",
                    "lr_multiplier": options.posterior_lr_multiplier,
                },
                {
                    "name": "linguistic",
                    "lr_multiplier": options.linguistic_lr_multiplier,
                },
                {
                    "name": "decoder",
                    "lr_multiplier": options.decoder_lr_multiplier,
                },
            ],
            "decoder_lr_warmup_steps": options.decoder_lr_warmup_steps,
            "freeze_upsamplers": options.decoder_freeze_upsamplers,
        },
        "discriminator": {
            **common,
            "base_learning_rate": options.learning_rate_d,
        },
        "scheduler": {
            "class": "torch.optim.lr_scheduler.ExponentialLR",
            "gamma": options.lr_decay,
        },
        "gradient_accumulation_steps": options.gradient_accumulation_steps,
        "amp_requested": options.amp,
        "generator_ema_decay": options.generator_ema_decay,
    }


def _validate_new_output_dir(output_dir: Path) -> None:
    """Reject accidental reuse while permitting empty scaffolding."""

    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError(f"Training output path is not a directory: {output_dir}")
    allowed_directories = {"checkpoints", "exports", "validation"}
    unsafe: list[str] = []
    for entry in output_dir.iterdir():
        if entry.is_file():
            if entry.name != ".gitkeep":
                unsafe.append(entry.name)
            continue
        if not entry.is_dir() or entry.name not in allowed_directories:
            unsafe.append(entry.name)
            continue
        for nested in entry.rglob("*"):
            if nested.is_dir() or nested.name != ".gitkeep":
                unsafe.append(str(nested.relative_to(output_dir)))
    if unsafe:
        raise ValueError(
            "Refusing to start a new run in a nonempty output directory. "
            f"Unexpected entries: {sorted(unsafe)}"
        )


def _resume_checkpoint_for_run(output_dir: Path, resume: str | Path) -> Path:
    checkpoint = Path(resume).resolve()
    checkpoints_dir = (output_dir / "checkpoints").resolve()
    try:
        checkpoint.relative_to(checkpoints_dir)
    except ValueError as error:
        raise ValueError(
            "Resume checkpoint must be inside this run's checkpoints directory: "
            f"{checkpoints_dir}"
        ) from error
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    return checkpoint


def _posterior_sidecar_path(options: TrainingOptions, base_root: Path) -> Path | None:
    """Resolve the posterior to inherit, or nothing when starting fresh."""

    if options.posterior_init != "inherit":
        return None
    path = base_root / "posterior.pth"
    if not path.is_file():
        raise FileNotFoundError(
            "posterior_init='inherit' needs a posterior sidecar beside the base "
            "checkpoint. Export the previous run with --include-posterior to "
            f"produce one: {path}"
        )
    return path


def _establish_run_identity(
    *,
    options: TrainingOptions,
    output_dir: Path,
    base_root: Path,
    prepared_dir: Path,
    posterior_path: Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    marker_path = output_dir / "run-identity.json"
    public_options = _public_options(options)
    optimizer_schema = _optimizer_schema(options)
    if options.resume is None:
        _validate_new_output_dir(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        identity = build_run_identity(
            run_id=uuid.uuid4().hex,
            base_root=base_root,
            prepared_dir=prepared_dir,
            options=public_options,
            optimizer_schema=optimizer_schema,
            posterior_path=posterior_path,
        )
        write_run_identity(marker_path, identity)
        return identity, None

    if not marker_path.is_file():
        raise ValueError(
            "Resume requires run-identity.json in the existing output directory."
        )
    checkpoint = _resume_checkpoint_for_run(output_dir, options.resume)
    recorded = load_run_identity(marker_path)
    expected = build_run_identity(
        run_id=str(recorded.get("run_id", "")),
        base_root=base_root,
        prepared_dir=prepared_dir,
        options=public_options,
        optimizer_schema=optimizer_schema,
        posterior_path=posterior_path,
    )
    validate_run_identity(recorded, expected, source="output directory run marker")
    return expected, checkpoint


@torch.inference_mode()
def _validate(
    bundle: ModelBundle,
    loader: Any,
    device: torch.device,
    output_dir: Path,
    step: int,
    seed: int,
) -> dict[str, Any]:
    model = bundle.generator
    model.eval()
    batch = next(iter(loader))
    x, x_lengths = batch[0].to(device), batch[1].to(device)
    # Seeding here is what makes the validation clip comparable across steps,
    # but the seed lands on the global generator that training is drawing from.
    # Without the fork, how many times validation has run becomes part of the
    # training trajectory, so `validation_interval` silently changes the run
    # and two runs that differ only in that setting are not comparable.
    # `manual_seed` covers every CUDA device, so the fork has to as well.
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        output = model.infer(x[:1], x_lengths[:1], noise_scale=0.667, max_len=4000)[0]
    waveform = output[0, 0].float().cpu().numpy()
    sample_rate = int(bundle.config["data"]["sampling_rate"])
    sample_path = output_dir / "validation" / f"step-{step:08d}.wav"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(sample_path, np.clip(waveform, -1.0, 1.0), sample_rate)
    model.train()
    # One clip is a noisy sample, so these are a trend to watch rather than a
    # gate; the gate is the offline evaluation over a held-out set. They are
    # here because a comb that appears at the unfreeze step is cheap to see now
    # and expensive to discover after the run finishes.
    screens = grid_comb_metrics(
        waveform.astype(np.float64),
        sample_rate,
        hop_length=int(bundle.config["data"]["hop_length"]),
    )
    return {
        "step": step,
        "sample": str(sample_path),
        "samples": int(waveform.size),
        "seconds": waveform.size / sample_rate,
        "peak": float(np.max(np.abs(waveform))),
        **screens,
    }


def train_adaptation(options: TrainingOptions) -> dict[str, Any]:
    """Warm-start and train one fixed-voice, single-language checkpoint."""

    options = options.resolved()
    _validate_options(options)
    _seed_everything(options.seed)
    device = _device(options.device)
    output_dir = Path(options.output_dir).resolve()
    prepared_dir = Path(options.prepared_dir).resolve()
    symbols = load_symbols(prepared_dir / "symbols.json")
    base_root = resolve_base_model(options.base_model)
    posterior_path = _posterior_sidecar_path(options, base_root)
    run_identity, resume_checkpoint = _establish_run_identity(
        options=options,
        output_dir=output_dir,
        base_root=base_root,
        prepared_dir=prepared_dir,
        posterior_path=posterior_path,
    )

    bundle = build_training_models(base_root, symbols, seed=options.seed)
    posterior_state = (
        load_posterior_sidecar(posterior_path) if posterior_path is not None else None
    )
    compatibility = cpu_compatibility_report(
        bundle.generator,
        base_root / "model.pth",
        bundle.base_symbols,
        symbols,
        initialization_seed=options.seed,
        posterior_state=posterior_state,
    )
    compatibility.write(output_dir / "compatibility-report.json")
    bundle.generator.to(device)
    bundle.discriminator.to(device)
    # Both snapshots are taken from the warm-started base and before any resume
    # state is read, so the anchor is the decoder this run began from whether
    # that is step 0 or step 6000. This is why neither has to be persisted.
    decoder_reference = (
        _decoder_reference(bundle.generator)
        if options.decoder_proximal_weight > 0.0
        else None
    )
    generator_ema = (
        _ema_state(bundle.generator) if options.generator_ema_decay > 0.0 else None
    )

    generator_groups = _generator_groups(bundle.generator, options)
    optimizer_g = torch.optim.AdamW(
        generator_groups,
        lr=options.learning_rate_g,
        betas=options.adam_betas,
        eps=options.adam_eps,
        weight_decay=options.weight_decay,
    )
    optimizer_d = torch.optim.AdamW(
        optimizer_parameters(bundle.discriminator),
        lr=options.learning_rate_d,
        betas=options.adam_betas,
        eps=options.adam_eps,
        weight_decay=options.weight_decay,
    )
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=options.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=options.lr_decay)
    amp_enabled = bool(options.amp and device.type == "cuda")
    scaler = _grad_scaler(amp_enabled)

    state = TrainingState(stage=_stage_for_step(options, 0))
    _apply_stage(bundle.generator, optimizer_g, options, state.stage)
    if resume_checkpoint is not None:
        step, epoch, saved_stage = resume_training_checkpoint(
            resume_checkpoint,
            generator=bundle.generator,
            discriminator=bundle.discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            scaler=scaler,
            expected_symbols=symbols,
            expected_run_identity=run_identity,
            generator_ema=generator_ema,
        )
        expected_stage = _stage_for_step(options, step)
        previous_stage = _stage_for_step(options, max(step - 1, 0))
        if saved_stage not in {previous_stage, expected_stage}:
            raise ValueError(
                f"Resume stage {saved_stage!r} conflicts with options-derived stage "
                f"{expected_stage!r} at step {step}."
            )
        # Checkpoints record the stage that produced their current step. When a
        # checkpoint lands exactly on a stage boundary, resume must configure
        # the model for the next step rather than reject the valid checkpoint.
        state = TrainingState(step=step, epoch=epoch, stage=expected_stage)
        # A resume in the middle of a stage keeps the decayed rates the
        # checkpoint carries, because the decay is part of the run. A resume
        # that lands on a boundary has to do what an uninterrupted run does at
        # that step and reset them: the group the new stage enables was saved
        # at zero while it was inactive, and nothing else would ever raise it,
        # so the decoder could silently spend a whole polish stage frozen.
        # The comparison is options-derived rather than against the saved
        # stage, because a boundary checkpoint records the stage it is entering
        # and a second resume from it would then miss the boundary.
        _apply_stage(
            bundle.generator,
            optimizer_g,
            options,
            state.stage,
            reset_learning_rates=expected_stage != previous_stage,
        )

    audio = AudioConfig(
        sampling_rate=int(bundle.config["data"]["sampling_rate"]),
        filter_length=int(bundle.config["data"]["filter_length"]),
        hop_length=int(bundle.config["data"]["hop_length"]),
        win_length=int(bundle.config["data"]["win_length"]),
        add_blank=bool(bundle.config["data"].get("add_blank", True)),
    )
    train_dataset = PreparedTTSDataset(prepared_dir, "train", symbols, audio)
    validation_dataset = PreparedTTSDataset(prepared_dir, "validation", symbols, audio)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.num_workers,
        seed=options.seed,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(options.num_workers, 1),
        seed=options.seed,
        pin_memory=device.type == "cuda",
    )

    options_path = output_dir / "training-options.json"
    options_path.write_text(
        json.dumps(_public_options(options), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log_path = output_dir / "metrics.jsonl"
    bundle.generator.train()
    bundle.discriminator.train()
    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)
    micro_step = 0

    while state.step < options.max_steps:
        state.epoch += 1
        for batch in train_loader:
            desired_stage = _stage_for_step(options, state.step)
            if desired_stage != state.stage:
                state.stage = desired_stage
                _apply_stage(bundle.generator, optimizer_g, options, state.stage)
                optimizer_d.zero_grad(set_to_none=True)
                micro_step = 0

            x, x_lengths, spec, spec_lengths, waveform, _ = (
                tensor.to(device, non_blocking=device.type == "cuda") for tensor in batch
            )
            adversarial_weight = _adversarial_weight(options, state.step, state.stage)
            train_discriminator = _discriminator_active(options, state.stage)
            with _autocast(device, amp_enabled):
                generated, duration, _, ids_slice, _, z_mask, latent = bundle.generator(
                    x, x_lengths, spec, spec_lengths
                )
                z, z_p, m_p, logs_p, _, logs_q = latent
                real = bundle.components.commons.slice_segments(
                    waveform,
                    ids_slice * audio.hop_length,
                    int(bundle.config["train"]["segment_size"]),
                )

                loss_d = None
                if train_discriminator:
                    _set_requires_grad(bundle.discriminator, True)
                    real_scores, generated_scores, _, _ = bundle.discriminator(
                        real, generated.detach()
                    )
                    loss_d = _discriminator_loss(real_scores, generated_scores)

            if loss_d is not None:
                scaler.scale(loss_d / options.gradient_accumulation_steps).backward()

            _set_requires_grad(bundle.discriminator, False)
            with _autocast(device, amp_enabled):
                loss_feature = None
                loss_generator = None
                if adversarial_weight > 0.0:
                    generated_scores_g = bundle.discriminator(real, generated)
                    _, fake_scores, real_maps, generated_maps = generated_scores_g
                    loss_feature = _feature_loss(real_maps, generated_maps)
                    loss_generator = _generator_loss(fake_scores)
                target_mel = bundle.components.commons.slice_segments(
                    _mel_from_spec(spec, bundle),
                    ids_slice,
                    int(bundle.config["train"]["segment_size"]) // audio.hop_length,
                )
                generated_mel = _mel_from_waveform(generated.squeeze(1), bundle)
                loss_mel = F.l1_loss(target_mel.float(), generated_mel.float())
                loss_duration = duration.float().sum()
                loss_kl = _kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                loss_stft = (
                    _multi_resolution_stft_loss(generated.squeeze(1), real.squeeze(1))
                    if options.stft_loss_weight > 0.0
                    else None
                )
                loss_proximal = (
                    _proximal_loss(bundle.generator, decoder_reference)
                    if decoder_reference is not None and state.stage == STAGE_DECODER
                    else None
                )
                latent_statistics = _latent_statistics(z, z_mask)
                loss_g = _generator_objective(
                    options,
                    adversarial_weight=adversarial_weight,
                    loss_generator=loss_generator,
                    loss_feature=loss_feature,
                    loss_mel=loss_mel,
                    loss_duration=loss_duration,
                    loss_kl=loss_kl,
                    loss_stft=loss_stft,
                    loss_proximal=loss_proximal,
                )
            scaler.scale(loss_g / options.gradient_accumulation_steps).backward()
            _set_requires_grad(bundle.discriminator, True)
            micro_step += 1

            if micro_step % options.gradient_accumulation_steps:
                continue
            scaler.unscale_(optimizer_g)
            if train_discriminator:
                scaler.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in optimizer_g.param_groups
                    for parameter in group["params"]
                    if parameter.requires_grad
                ],
                options.max_grad_norm,
            )
            if train_discriminator:
                torch.nn.utils.clip_grad_norm_(
                    bundle.discriminator.parameters(), options.max_grad_norm
                )
                # Stepping an optimizer the scaler recorded no inf check for is
                # a hard error, so this has to follow the same condition as the
                # backward pass above rather than being merely wasteful.
                scaler.step(optimizer_d)
            decoder_lr_scale = _decoder_lr_scale(options, state.step, state.stage)
            with _scaled_decoder_lr(optimizer_g, decoder_lr_scale):
                scaler.step(optimizer_g)
            scaler.update()
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)
            scheduler_g.step()
            scheduler_d.step()
            state.step += 1
            if generator_ema is not None:
                _ema_update(
                    generator_ema,
                    bundle.generator,
                    _ema_decay(options.generator_ema_decay, state.step),
                )

            metrics = {
                "step": state.step,
                "epoch": state.epoch,
                "stage": state.stage,
                "loss_g": float(loss_g.detach().cpu()),
                "loss_d": _scalar(loss_d),
                "loss_mel": float(loss_mel.detach().cpu()),
                "loss_duration": float(loss_duration.detach().cpu()),
                "loss_kl": float(loss_kl.detach().cpu()),
                # A term the schedule switched off is null rather than zero: a
                # zero here would read as "measured and negligible".
                "loss_generator": _scalar(loss_generator),
                "loss_feature": _scalar(loss_feature),
                "loss_stft": _scalar(loss_stft),
                "loss_proximal": _scalar(loss_proximal),
                "adversarial_weight": adversarial_weight,
                "decoder_lr_scale": decoder_lr_scale,
                "lr": {
                    group["name"]: float(group["lr"]) for group in optimizer_g.param_groups
                },
                **latent_statistics,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            if state.step % options.log_interval == 0:
                LOGGER.info(
                    "step=%d stage=%s loss_g=%.4f loss_d=%.4f",
                    state.step,
                    state.stage,
                    metrics["loss_g"],
                    float("nan") if metrics["loss_d"] is None else metrics["loss_d"],
                )
            if state.step % options.validation_interval == 0:
                validation = _validate(
                    bundle,
                    validation_loader,
                    device,
                    output_dir,
                    state.step,
                    options.validation_seed,
                )
                (output_dir / "validation" / f"step-{state.step:08d}.json").write_text(
                    json.dumps(validation, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            if state.step % options.checkpoint_interval == 0:
                checkpoint_path = (
                    output_dir / "checkpoints" / f"adaptation-step-{state.step:08d}.pth"
                )
                save_training_checkpoint(
                    checkpoint_path,
                    generator=bundle.generator,
                    discriminator=bundle.discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    scheduler_g=scheduler_g,
                    scheduler_d=scheduler_d,
                    scaler=scaler,
                    step=state.step,
                    epoch=state.epoch,
                    stage=state.stage,
                    options=_public_options(options),
                    symbols=symbols,
                    compatibility=compatibility,
                    run_identity=run_identity,
                    latest_path=output_dir / "checkpoints" / "latest.pth",
                    generator_ema=generator_ema,
                )
                save_inference_checkpoint(
                    output_dir / "exports" / f"model-step-{state.step:08d}.pth",
                    generator=bundle.generator,
                    iteration=state.step,
                    learning_rate=optimizer_g.param_groups[0]["lr"],
                )
                if generator_ema is not None:
                    save_inference_checkpoint(
                        output_dir / "exports" / f"model-ema-step-{state.step:08d}.pth",
                        generator=bundle.generator,
                        iteration=state.step,
                        learning_rate=optimizer_g.param_groups[0]["lr"],
                        state=generator_ema,
                    )
            if state.step >= options.max_steps:
                break

    final_training = save_training_checkpoint(
        output_dir / "checkpoints" / "adaptation-final.pth",
        generator=bundle.generator,
        discriminator=bundle.discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        scaler=scaler,
        step=state.step,
        epoch=state.epoch,
        stage=state.stage,
        options=_public_options(options),
        symbols=symbols,
        compatibility=compatibility,
        run_identity=run_identity,
        latest_path=output_dir / "checkpoints" / "latest.pth",
        generator_ema=generator_ema,
    )
    final_inference = save_inference_checkpoint(
        output_dir / "exports" / "model.pth",
        generator=bundle.generator,
        iteration=state.step,
        learning_rate=optimizer_g.param_groups[0]["lr"],
    )
    final_ema = None
    if generator_ema is not None:
        final_ema = save_inference_checkpoint(
            output_dir / "exports" / "model-ema.pth",
            generator=bundle.generator,
            iteration=state.step,
            learning_rate=optimizer_g.param_groups[0]["lr"],
            state=generator_ema,
        )
    summary = {
        "step": state.step,
        "epoch": state.epoch,
        "stage": state.stage,
        "training_checkpoint": str(final_training),
        "inference_checkpoint": str(final_inference),
        "generator_ema_checkpoint": None if final_ema is None else str(final_ema),
        "posterior_source": compatibility.posterior_source,
        "compatibility_report": compatibility.to_dict(),
        "run_id": run_identity["run_id"],
    }
    (output_dir / "training-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = ["TrainingOptions", "train_adaptation"]
