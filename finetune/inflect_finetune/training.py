"""Generic staged warm-start training for Inflect v2 adaptation."""

from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import MISSING, asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import (
    CompatibilityReport,
    build_run_identity,
    cpu_compatibility_report,
    load_run_identity,
    resume_training_checkpoint,
    save_inference_checkpoint,
    save_training_checkpoint,
    validate_run_identity,
    write_run_identity,
)
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


def _apply_stage(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    options: TrainingOptions,
    stage: str,
    *,
    reset_learning_rates: bool = True,
) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unknown training stage {stage!r}.")
    enabled = {
        STAGE_POSTERIOR: {"posterior"},
        STAGE_ADAPT: {"posterior", "linguistic"},
        STAGE_DECODER: {"posterior", "linguistic", "decoder"},
    }[stage]
    for group in optimizer.param_groups:
        name = group["name"]
        active = name in enabled
        if not active:
            group["lr"] = 0.0
        elif reset_learning_rates:
            group["lr"] = options.learning_rate_g * float(group["lr_multiplier"])
        for parameter in group["params"]:
            parameter.requires_grad_(active)
            if not active:
                parameter.grad = None
    optimizer.zero_grad(set_to_none=True)


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
    data = bundle.config["data"]
    bank = _mel_filterbank(
        n_fft=int(data["filter_length"]),
        n_mels=int(data["n_mel_channels"]),
        sample_rate=int(data["sampling_rate"]),
        fmin=float(data["mel_fmin"]),
        fmax=float(data["mel_fmax"]),
        device=spec.device,
        dtype=spec.dtype,
    )
    return torch.log(torch.matmul(bank, spec).clamp_min(1.0e-5))


def _mel_from_waveform(waveform: torch.Tensor, bundle: ModelBundle) -> torch.Tensor:
    data = bundle.config["data"]
    n_fft = int(data["filter_length"])
    hop = int(data["hop_length"])
    win = int(data["win_length"])
    padding = (n_fft - hop) // 2
    padded = F.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    window = torch.hann_window(win, device=waveform.device, dtype=waveform.dtype)
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


def _establish_run_identity(
    *,
    options: TrainingOptions,
    output_dir: Path,
    base_root: Path,
    prepared_dir: Path,
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
    torch.manual_seed(seed)
    output = model.infer(x[:1], x_lengths[:1], noise_scale=0.667, max_len=4000)[0]
    waveform = output[0, 0].float().cpu().numpy()
    sample_rate = int(bundle.config["data"]["sampling_rate"])
    sample_path = output_dir / "validation" / f"step-{step:08d}.wav"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(sample_path, np.clip(waveform, -1.0, 1.0), sample_rate)
    model.train()
    return {
        "step": step,
        "sample": str(sample_path),
        "samples": int(waveform.size),
        "seconds": waveform.size / sample_rate,
        "peak": float(np.max(np.abs(waveform))),
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
    run_identity, resume_checkpoint = _establish_run_identity(
        options=options,
        output_dir=output_dir,
        base_root=base_root,
        prepared_dir=prepared_dir,
    )

    bundle = build_training_models(base_root, symbols, seed=options.seed)
    compatibility = cpu_compatibility_report(
        bundle.generator,
        base_root / "model.pth",
        bundle.base_symbols,
        symbols,
        initialization_seed=options.seed,
    )
    compatibility.write(output_dir / "compatibility-report.json")
    bundle.generator.to(device)
    bundle.discriminator.to(device)

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
        _apply_stage(
            bundle.generator,
            optimizer_g,
            options,
            state.stage,
            reset_learning_rates=False,
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

                _set_requires_grad(bundle.discriminator, True)
                real_scores, generated_scores, _, _ = bundle.discriminator(
                    real, generated.detach()
                )
                loss_d = _discriminator_loss(real_scores, generated_scores)

            scaler.scale(loss_d / options.gradient_accumulation_steps).backward()

            _set_requires_grad(bundle.discriminator, False)
            with _autocast(device, amp_enabled):
                generated_scores_g = bundle.discriminator(real, generated)
                _, fake_scores, real_maps, generated_maps = generated_scores_g
                target_mel = bundle.components.commons.slice_segments(
                    _mel_from_spec(spec, bundle),
                    ids_slice,
                    int(bundle.config["train"]["segment_size"]) // audio.hop_length,
                )
                generated_mel = _mel_from_waveform(generated.squeeze(1), bundle)
                loss_mel = F.l1_loss(target_mel.float(), generated_mel.float())
                loss_duration = duration.float().sum()
                loss_kl = _kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                loss_feature = _feature_loss(real_maps, generated_maps)
                loss_generator = _generator_loss(fake_scores)
                loss_g = (
                    loss_generator
                    + options.feature_loss_weight * loss_feature
                    + options.mel_loss_weight * loss_mel
                    + options.duration_loss_weight * loss_duration
                    + options.kl_loss_weight * loss_kl
                )
            scaler.scale(loss_g / options.gradient_accumulation_steps).backward()
            _set_requires_grad(bundle.discriminator, True)
            micro_step += 1

            if micro_step % options.gradient_accumulation_steps:
                continue
            scaler.unscale_(optimizer_g)
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
            torch.nn.utils.clip_grad_norm_(
                bundle.discriminator.parameters(), options.max_grad_norm
            )
            scaler.step(optimizer_d)
            scaler.step(optimizer_g)
            scaler.update()
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)
            scheduler_g.step()
            scheduler_d.step()
            state.step += 1

            metrics = {
                "step": state.step,
                "epoch": state.epoch,
                "stage": state.stage,
                "loss_g": float(loss_g.detach().cpu()),
                "loss_d": float(loss_d.detach().cpu()),
                "loss_mel": float(loss_mel.detach().cpu()),
                "loss_duration": float(loss_duration.detach().cpu()),
                "loss_kl": float(loss_kl.detach().cpu()),
                "lr": {
                    group["name"]: float(group["lr"]) for group in optimizer_g.param_groups
                },
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            if state.step % options.log_interval == 0:
                LOGGER.info(
                    "step=%d stage=%s loss_g=%.4f loss_d=%.4f",
                    state.step,
                    state.stage,
                    metrics["loss_g"],
                    metrics["loss_d"],
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
                )
                save_inference_checkpoint(
                    output_dir / "exports" / f"model-step-{state.step:08d}.pth",
                    generator=bundle.generator,
                    iteration=state.step,
                    learning_rate=optimizer_g.param_groups[0]["lr"],
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
    )
    final_inference = save_inference_checkpoint(
        output_dir / "exports" / "model.pth",
        generator=bundle.generator,
        iteration=state.step,
        learning_rate=optimizer_g.param_groups[0]["lr"],
    )
    summary = {
        "step": state.step,
        "epoch": state.epoch,
        "stage": state.stage,
        "training_checkpoint": str(final_training),
        "inference_checkpoint": str(final_inference),
        "compatibility_report": compatibility.to_dict(),
        "run_id": run_identity["run_id"],
    }
    (output_dir / "training-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = ["TrainingOptions", "train_adaptation"]
