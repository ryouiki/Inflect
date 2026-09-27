"""A tiny but complete stub Inflect release the real training loop can run.

`tests/test_symbol_inventory_migration.py` already writes a stub release, but its
`SynthesizerTrn` has no `forward` — enough to prove symbol migration, useless for
proving anything about an optimizer step. This module fills that gap: the release
it writes is loadable by `modeling.load_runtime_components`, warm-startable by
`checkpoint.warm_start_from_release`, and shaped closely enough to VITS that
`training.train_adaptation` runs end to end on CPU in seconds.

"Closely enough" means specifically:

* Every generator parameter falls under one of the five prefixes
  `training._generator_groups` classifies, and all five are populated, so the
  optimizer's parameter groups are exercised rather than skipped.
* `dec` carries the four real HiFi-GAN submodule names, and `dec.ups.0` is a
  `ConvTranspose1d` whose stride equals the hop, so one latent frame becomes
  exactly `hop` samples. That is the upsample grid whose comb the remedy hunts,
  and it is what an upsampler-freeze option has to be able to match by name.
* `enc_p`, `dp`, `flow` and `dec` are callable with the released signatures, so
  `exporting`'s `_DurationGraph`/`_DecodeGraph` can be built from this model too.

What it deliberately does not reproduce is the model: there is no attention, no
WaveNet stack, and no monotonic alignment search. Text is expanded to the
spectrogram length by a fixed positional mapping.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from inflect_finetune.checkpoint import RELEASE_FORMAT, save_inference_checkpoint
from inflect_finetune.modeling import build_training_models
from inflect_finetune.symbols import BASE_SYMBOLS as _RELEASE_BASE_SYMBOLS

BASE_SYMBOLS: tuple[str, ...] = _RELEASE_BASE_SYMBOLS

SAMPLING_RATE = 24_000
FILTER_LENGTH = 1024
HOP_LENGTH = 256
SEGMENT_SIZE = 2048
SEGMENT_FRAMES = SEGMENT_SIZE // HOP_LENGTH
INTER_CHANNELS = 8

#: Prefixes `training._generator_groups` classifies; the stub uses all five and
#: adds none of its own.
GENERATOR_PREFIXES = ("enc_p.", "dp.", "flow.", "enc_q.", "dec.")
#: Parameter prefix an upsampler-freeze option has to match.
UPSAMPLER_PREFIX = "dec.ups."
#: Submodule names the remedy's decoder controls refer to by name.
DECODER_SUBMODULES = ("dec.conv_pre", "dec.ups.0", "dec.resblocks.0", "dec.conv_post")

#: Shortest waveform whose magnitude spectrogram still has `SEGMENT_FRAMES`
#: frames. Below this `commons.rand_slice_segments` has nothing to slice and the
#: loop raises instead of training. A row with fewer frames than tokens still
#: trains, but the fixed alignment then leaves most tokens at zero duration and
#: the duration term dominates the loss, so prefer audio long enough that the
#: frame count exceeds the token count.
MIN_WAVEFORM_SAMPLES = SEGMENT_SIZE
MIN_WAVEFORM_SECONDS = MIN_WAVEFORM_SAMPLES / SAMPLING_RATE

_RELEASE_SEED = 20_240_917

_CONFIG: dict[str, object] = {
    "format": "inflect_v2_inference_config_v1",
    "train": {"segment_size": SEGMENT_SIZE},
    "data": {
        "text_cleaners": [],
        "max_wav_value": 32768.0,
        "sampling_rate": SAMPLING_RATE,
        "filter_length": FILTER_LENGTH,
        "hop_length": HOP_LENGTH,
        "win_length": FILTER_LENGTH,
        "n_mel_channels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": 12000.0,
        "add_blank": True,
        "n_speakers": 0,
        "cleaned_text": True,
    },
    "model": {
        "inter_channels": INTER_CHANNELS,
        "hidden_channels": 8,
        "upsample_initial_channel": 4,
        "upsample_rates": [HOP_LENGTH],
        "upsample_kernel_sizes": [HOP_LENGTH],
        "n_speakers": 0,
        "n_layers_q": 3,
        "use_spectral_norm": False,
        "use_sdp": False,
        "inference_only": True,
    },
}

# Copied from the released runtime so the slicing and mask semantics the loop
# depends on are the real ones rather than a re-derivation.
_COMMONS_SOURCE = '''"""Subset of the released VITS commons, copied verbatim."""

import torch
from torch.nn import functional as F


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path
'''

_MODELS_SOURCE = '''"""A VITS-shaped stub: released module names, negligible capacity."""

import torch
from torch import nn
from torch.nn import functional as F

import commons

# `modeling.load_runtime_components` replaces this with the toolkit's own
# implementation. The stub never calls it, but the attribute has to exist for
# that assignment to be meaningful.
monotonic_align = None

LRELU_SLOPE = 0.1

# A diverged duration head could otherwise ask `infer` for minutes of audio, and
# this fixture has to stay seconds-fast even mid-training.
MAX_INFER_FRAMES = 512


def _small_stats_init(conv):
    """Keep the prior/posterior log-variance near zero at initialisation.

    `training._kl_loss` exponentiates `-2 * logs_p`, so an untrained projection
    with default initialisation can hand the very first step a loss several
    orders of magnitude larger than any other term.
    """

    nn.init.normal_(conv.weight, 0.0, 0.05)
    nn.init.zeros_(conv.bias)


def uniform_alignment(x_lengths, y_lengths, x_mask, y_mask):
    """Stand in for the monotonic alignment search with a fixed mapping.

    Each spectrogram frame takes the token at the same relative position. The
    result is monotonic, sums to one frame per row and masks padding, which is
    everything the loop reads off `attn`.
    """

    t_x = x_mask.shape[-1]
    t_y = y_mask.shape[-1]
    frames = torch.arange(t_y, device=x_mask.device, dtype=torch.float32)
    ratio = x_lengths.to(torch.float32) / y_lengths.to(torch.float32).clamp_min(1.0)
    tokens = (frames.unsqueeze(0) * ratio.unsqueeze(1)).floor().to(torch.long)
    tokens = torch.minimum(tokens, (x_lengths - 1).clamp_min(0).unsqueeze(1))
    attn = F.one_hot(tokens, num_classes=t_x).to(y_mask.dtype)
    return (attn * y_mask.transpose(1, 2)).unsqueeze(1)


class TextEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels):
        super().__init__()
        self.out_channels = out_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)
        self.enc = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        _small_stats_init(self.proj)

    def forward(self, x, x_lengths):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(1)), 1).to(
            self.emb.weight.dtype
        )
        hidden = self.emb(x).transpose(1, 2) * x_mask
        hidden = torch.tanh(self.enc(hidden)) * x_mask
        stats = self.proj(hidden) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return hidden, m, logs, x_mask


class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        _small_stats_init(self.proj)

    def forward(self, y, y_lengths, g=None):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        hidden = self.pre(y) * y_mask
        hidden = torch.tanh(self.enc(hidden)) * y_mask
        stats = self.proj(hidden) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * y_mask
        return z, m, logs, y_mask


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, 3, padding=1)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask, g=None):
        # The release detaches here, so the duration term never reaches enc_p.
        x = torch.detach(x)
        x = torch.relu(self.conv_1(x * x_mask))
        return self.proj(x * x_mask) * x_mask


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        return torch.flip(x, [1])


class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.half_channels = channels // 2
        self.enc = nn.Conv1d(self.half_channels, self.half_channels, 1)
        nn.init.normal_(self.enc.weight, 0.0, 0.05)
        nn.init.zeros_(self.enc.bias)

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        shift = self.enc(x0 * x_mask) * x_mask
        x1 = x1 - shift if reverse else x1 + shift
        return torch.cat([x0, x1], 1) * x_mask


class ResidualCouplingBlock(nn.Module):
    """Exactly invertible, so `reverse=True` really undoes the forward pass."""

    def __init__(self, channels, n_flows=2):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        flows = reversed(self.flows) if reverse else self.flows
        for flow in flows:
            x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=commons.get_padding(kernel_size),
                )
            ]
        )

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(F.leaky_relu(x, LRELU_SLOPE))
        return x


class Generator(nn.Module):
    """One-stage HiFi-GAN: a single transposed conv striding by the hop.

    Keeping stride == hop is the point. One latent frame becomes exactly `hop`
    output samples, so the fixture reproduces the 24000/hop upsample grid whose
    periodic artefact the remedy exists to detect.
    """

    def __init__(self, initial_channel, channels, upsample_rate, upsample_kernel_size):
        super().__init__()
        self.conv_pre = nn.Conv1d(initial_channel, channels, 7, 1, padding=3)
        self.ups = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    channels,
                    channels,
                    upsample_kernel_size,
                    upsample_rate,
                    padding=(upsample_kernel_size - upsample_rate) // 2,
                )
            ]
        )
        self.resblocks = nn.ModuleList([ResBlock(channels)])
        self.conv_post = nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(commons.init_weights)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        for index, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            x = self.resblocks[index](x)
        return torch.tanh(self.conv_post(F.leaky_relu(x, LRELU_SLOPE)))


class DiscriminatorP(nn.Module):
    def __init__(self, period, use_spectral_norm=False):
        super().__init__()
        self.period = int(period)
        self.use_spectral_norm = bool(use_spectral_norm)
        self.convs = nn.ModuleList([nn.Conv2d(1, 4, (5, 1), (3, 1), padding=(2, 0))])
        self.conv_post = nn.Conv2d(4, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        batch, channels, length = x.shape
        if length % self.period:
            x = F.pad(x, (0, self.period - (length % self.period)), "reflect")
            length = x.shape[-1]
        x = x.view(batch, channels, length // self.period, self.period)
        feature_maps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        return torch.flatten(x, 1, -1), feature_maps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        self.use_spectral_norm = bool(use_spectral_norm)
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period, use_spectral_norm) for period in (2, 3)]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for discriminator in self.discriminators:
            y_d_r, fmap_r = discriminator(y)
            y_d_g, fmap_g = discriminator(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels=8,
        hidden_channels=8,
        upsample_initial_channel=4,
        upsample_rates=(256,),
        upsample_kernel_sizes=(256,),
        n_speakers=0,
        gin_channels=0,
        use_sdp=False,
        **kwargs,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = bool(use_sdp)
        self.inference_only = bool(kwargs.get("inference_only", False))

        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels)
        self.dec = Generator(
            inter_channels,
            upsample_initial_channel,
            int(upsample_rates[0]),
            int(upsample_kernel_sizes[0]),
        )
        if not self.inference_only:
            self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels)
        self.flow = ResidualCouplingBlock(inter_channels)
        self.dp = DurationPredictor(hidden_channels, hidden_channels)

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        if self.inference_only:
            raise RuntimeError("This stub was built inference-only and has no enc_q.")

        hidden, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths)
        z_p = self.flow(z, y_mask)

        attn = uniform_alignment(x_lengths, y_lengths, x_mask, y_mask)
        w = attn.sum(2)
        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(hidden, x_mask)
        l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice)
        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(
        self,
        x,
        x_lengths,
        sid=None,
        noise_scale=1.0,
        length_scale=1.0,
        noise_scale_w=1.0,
        max_len=None,
        **kwargs,
    ):
        # noise_scale_w is accepted and ignored: the stub duration predictor is
        # deterministic, exactly as the release is with use_sdp false.
        hidden, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        logw = self.dp(hidden, x_mask)
        w_ceil = torch.ceil(torch.exp(logw) * x_mask * float(length_scale))
        y_lengths = torch.clamp_min(w_ceil.sum([1, 2]), 1).to(torch.long)
        limit = MAX_INFER_FRAMES if max_len is None else min(int(max_len), MAX_INFER_FRAMES)
        y_lengths = y_lengths.clamp(max=max(int(limit), 1))
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)

        attn = uniform_alignment(x_lengths, y_lengths, x_mask, y_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * float(noise_scale)
        z = self.flow(z_p, y_mask, reverse=True)
        return self.dec(z * y_mask), attn, y_mask, (z, z_p, m_p, logs_p)
'''


def stub_config() -> dict[str, object]:
    """Return a fresh copy of the stub release config."""

    return copy.deepcopy(_CONFIG)


def extended_symbols(count: int) -> tuple[str, ...]:
    """Return the release inventory plus `count` distinct appended symbols.

    Appending is the one inventory change `validate_release_compatible_symbols`
    allows, so this is how a test builds a stub whose embedding has to grow.
    """

    return BASE_SYMBOLS + tuple(chr(0xA71C + index) for index in range(count))


def _write_runtime(root: Path, symbols: Sequence[str]) -> None:
    runtime = root / "runtime"
    (runtime / "text").mkdir(parents=True, exist_ok=True)
    (runtime / "models.py").write_text(_MODELS_SOURCE, encoding="utf-8")
    (runtime / "commons.py").write_text(_COMMONS_SOURCE, encoding="utf-8")
    (runtime / "text" / "__init__.py").write_text("", encoding="utf-8")
    (runtime / "text" / "symbols.py").write_text(
        f"symbols = {list(symbols)!r}\nSPACE_ID = symbols.index(' ')\n",
        encoding="utf-8",
    )


def build_stub_release(root: Path, symbols: Sequence[str] | None = None) -> Path:
    """Write a complete, loadable stub release directory and return its path.

    The released `model.pth` is produced by instantiating the stub through the
    toolkit's own `build_training_models` and dropping the `enc_q.*` tensors,
    rather than by writing hand-listed shapes. `warm_start_from_release` demands
    that the release keys equal the training keys minus the fresh prefixes, with
    identical shapes and dtypes; deriving the payload from the model is the only
    way that stays true as the stub architecture changes.
    """

    inventory = tuple(BASE_SYMBOLS if symbols is None else symbols)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _write_runtime(root, inventory)
    (root / "config.json").write_text(json.dumps(_CONFIG, indent=2) + "\n", encoding="utf-8")
    # resolve_base_model insists on model.pth before anything may be imported
    # from the directory, so a placeholder goes in first and is then replaced by
    # the real payload built from the instantiated model.
    torch.save({"format": RELEASE_FORMAT, "model": {}}, root / "model.pth")

    bundle = build_training_models(root, inventory, seed=_RELEASE_SEED)
    save_inference_checkpoint(
        root / "model.pth",
        generator=bundle.generator,
        iteration=0,
        learning_rate=0.0,
    )
    return root
