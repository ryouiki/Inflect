"""Prepared-dataset loading for Inflect adaptation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class AudioConfig:
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    add_blank: bool


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"Dataset audio path escapes the prepared directory: {relative!r}"
        ) from error
    if not candidate.is_file():
        raise FileNotFoundError(f"Prepared audio file does not exist: {candidate}")
    return candidate


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: each row must be a JSON object.")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no dataset rows.")
    return rows


MAGNITUDE_FLOOR_POWER = 1.0e-6


def magnitude_spectrogram(
    signal: torch.Tensor, *, n_fft: int, hop_length: int, win_length: int
) -> torch.Tensor:
    """The one magnitude-spectrogram definition this toolkit uses.

    The floor is added in the power domain and then square-rooted, which is
    the released VITS convention. That detail is load-bearing rather than
    cosmetic: clamping the magnitude instead puts the floor a hundred times
    lower, and while only one side of the reconstruction loss clamped, two
    identical waveforms scored ln(100) rather than zero and the objective paid
    for a broadband noise floor in every quiet cell. Both sides come through
    here so the two cannot drift apart again.

    Takes one waveform or a batch, and owns the reflect padding so that no
    caller applies it a second time.
    """

    batched = signal.dim() == 2
    frames = signal if batched else signal[None]
    padding = (n_fft - hop_length) // 2
    padded = F.pad(frames[:, None], (padding, padding), mode="reflect")[:, 0]
    window = torch.hann_window(win_length, device=padded.device, dtype=padded.dtype)
    spectrum = torch.stft(
        padded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = (spectrum.abs().square() + MAGNITUDE_FLOOR_POWER).sqrt()
    return magnitude if batched else magnitude[0]


def spectrogram(waveform: torch.Tensor, config: AudioConfig) -> torch.Tensor:
    """Match the released VITS magnitude-spectrogram convention."""

    padding = (config.filter_length - config.hop_length) // 2
    if waveform.numel() < 2:
        raise ValueError("Audio contains fewer than two samples.")
    if waveform.numel() <= padding:
        # A prepared row can be shorter than the reflect padding. A training
        # segment never is, which is why this guard belongs to the dataset
        # side rather than to the shared core: padding a segment there would
        # quietly change the signal the loss is computed on.
        waveform = F.pad(waveform, (0, padding + 1 - waveform.numel()))
    return magnitude_spectrogram(
        waveform,
        n_fft=config.filter_length,
        hop_length=config.hop_length,
        win_length=config.win_length,
    )


class PreparedTTSDataset(Dataset):
    """Load one split from the contract-defined prepared dataset."""

    def __init__(
        self,
        prepared_dir: str | Path,
        split: str,
        symbols: Sequence[str],
        audio_config: AudioConfig,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        self.root = Path(prepared_dir).resolve()
        self.rows = read_jsonl(self.root / f"{split}.jsonl")
        self.symbol_to_id = {symbol: index for index, symbol in enumerate(symbols)}
        self.audio_config = audio_config
        self.records = []
        unknown: set[str] = set()
        for index, row in enumerate(self.rows):
            for field in ("audio", "phonemes"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"{split}.jsonl row {index} has invalid {field!r}.")
            audio_path = _safe_child(self.root, row["audio"])
            missing = set(row["phonemes"]) - self.symbol_to_id.keys()
            unknown.update(missing)
            self.records.append((audio_path, row["phonemes"]))
        if unknown:
            rendered = " ".join(repr(item) for item in sorted(unknown))
            raise ValueError(
                f"{split}.jsonl contains symbols absent from symbols.json: {rendered}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_path, phonemes = self.records[index]
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        if sample_rate != self.audio_config.sampling_rate:
            raise ValueError(
                f"{audio_path} is {sample_rate} Hz; expected "
                f"{self.audio_config.sampling_rate} Hz. Re-run dataset preparation."
            )
        waveform = torch.from_numpy(audio.mean(axis=1).copy())
        if not torch.isfinite(waveform).all():
            raise ValueError(f"{audio_path} contains NaN or infinite samples.")
        peak = float(waveform.abs().max())
        if peak > 1.05:
            raise ValueError(f"{audio_path} exceeds normalized floating-point range ({peak:.3f}).")
        token_ids = [self.symbol_to_id[symbol] for symbol in phonemes]
        if self.audio_config.add_blank:
            expanded = [0] * (len(token_ids) * 2 + 1)
            expanded[1::2] = token_ids
            token_ids = expanded
        if not token_ids:
            raise ValueError(f"{audio_path} has an empty phoneme sequence.")
        tokens = torch.tensor(token_ids, dtype=torch.long)
        return tokens, spectrogram(waveform, self.audio_config), waveform[None]


class TTSCollate:
    def __call__(
        self, batch: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, ...]:
        rows = sorted(batch, key=lambda item: item[1].shape[-1], reverse=True)
        batch_size = len(rows)
        text_lengths = torch.tensor([row[0].numel() for row in rows], dtype=torch.long)
        spec_lengths = torch.tensor([row[1].shape[-1] for row in rows], dtype=torch.long)
        wav_lengths = torch.tensor([row[2].shape[-1] for row in rows], dtype=torch.long)
        text = torch.zeros(batch_size, int(text_lengths.max()), dtype=torch.long)
        spec = torch.zeros(
            batch_size, rows[0][1].shape[0], int(spec_lengths.max()), dtype=torch.float32
        )
        wav = torch.zeros(batch_size, 1, int(wav_lengths.max()), dtype=torch.float32)
        for index, (tokens, spectrum, waveform) in enumerate(rows):
            text[index, : tokens.numel()] = tokens
            spec[index, :, : spectrum.shape[-1]] = spectrum
            wav[index, :, : waveform.shape[-1]] = waveform
        return text, text_lengths, spec, spec_lengths, wav, wav_lengths


def create_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": TTSCollate(),
        "drop_last": False,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)
