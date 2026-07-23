from __future__ import annotations

import argparse
import io
import json
import os
import platform
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any, Callable

try:
    import psutil
except ImportError:  # pragma: no cover - minimal release environments may omit it
    psutil = None


def process_rss_bytes() -> int:
    if psutil is not None:
        return int(psutil.Process().memory_info().rss)
    try:
        import resource

        # Linux reports KiB while macOS reports bytes.
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError):
        return 0


def cpu_model() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def process_cpu_affinity_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def load_prompts(path: Path, limit: int) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{key: str(row[key]) for key in ("id", "category", "text")} for row in rows[:limit]]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timed_runs(
    prompts: list[dict[str, str]],
    synthesize: Callable[[str], float],
    synchronize: Callable[[], None] = lambda: None,
) -> tuple[list[dict[str, Any]], int]:
    for row in prompts[:3]:
        synthesize(row["text"])
    synchronize()
    peak_rss = process_rss_bytes()
    results = []
    for index, row in enumerate(prompts, 1):
        synchronize()
        started = time.perf_counter()
        audio_seconds = synthesize(row["text"])
        synchronize()
        latency = time.perf_counter() - started
        peak_rss = max(peak_rss, process_rss_bytes())
        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "characters": len(row["text"]),
                "audio_seconds": audio_seconds,
                "latency_seconds": latency,
                "rtf": latency / audio_seconds,
            }
        )
        print(f"runtime={index}/{len(prompts)} latency={latency:.4f}s audio={audio_seconds:.3f}s", flush=True)
    return results, peak_rss


def summarize(system: str, provider: str, load_seconds: float, rows: list[dict[str, Any]], peak_rss: int) -> dict[str, Any]:
    latencies = [float(row["latency_seconds"]) for row in rows]
    total_latency = sum(latencies)
    total_audio = sum(float(row["audio_seconds"]) for row in rows)
    return {
        "format": "inflect_tts_runtime_v2",
        "system": system,
        "provider": provider,
        "hardware": {
            "platform": platform.platform(),
            "processor": cpu_model(),
            "logical_cpus_visible": os.cpu_count(),
            "process_cpu_affinity_count": process_cpu_affinity_count(),
        },
        "runtime_protocol": {
            "python": platform.python_version(),
            "single_process": True,
            "concurrent_synthesis_requests": 1,
            "warmup_utterances": 3,
            "thread_policy": "runtime default",
            "environment_threads": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
        "utterances": len(rows),
        "load_seconds": load_seconds,
        "peak_process_rss_bytes_observed": peak_rss,
        "total_latency_seconds": total_latency,
        "total_audio_seconds": total_audio,
        "rtf": total_latency / total_audio,
        "realtime_multiple": total_audio / total_latency,
        "latency_mean_seconds": statistics.mean(latencies),
        "latency_median_seconds": statistics.median(latencies),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "rows": rows,
    }


def benchmark_inflect(args: argparse.Namespace, prompts: list[dict[str, str]]) -> tuple[str, float, list[dict[str, Any]], int]:
    import torch

    if args.model_dir is not None:
        sys.path.insert(0, str(args.model_dir.resolve()))
        from inference import InflectTTS

        if args.cpu_threads is not None:
            torch.set_num_threads(args.cpu_threads)
            torch.set_num_interop_threads(1)
        started = time.perf_counter()
        engine = InflectTTS(args.model_dir, device=args.device)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started

        def synthesize_package(text: str) -> float:
            sample_rate, waveform = engine.synthesize(
                text,
                speed=1.0,
                variation=0.667,
                seed=20260722,
            )
            return waveform.size / sample_rate

        sync = torch.cuda.synchronize if args.device.startswith("cuda") else (lambda: None)
        rows, peak = timed_runs(prompts, synthesize_package, sync)
        return str(engine.device), load_seconds, rows, peak

    sys.path.insert(0, str(args.vits_root.resolve()))
    import commons
    import utils
    from models import SynthesizerTrn
    from text import cleaned_text_to_sequence
    from text.symbols import symbols

    sys.path.insert(0, str(args.frontend_root.resolve()))
    from inflect_vits_frontend import run_vits_frontend

    device = torch.device(args.device)
    started = time.perf_counter()
    hps = utils.get_hparams_from_file(str(args.config))
    model = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device).eval()
    utils.load_checkpoint(str(args.checkpoint), model, None)
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started

    @torch.inference_mode()
    def synthesize(text: str) -> float:
        cleaned = run_vits_frontend(text).phoneme_text
        sequence = cleaned_text_to_sequence(cleaned)
        if hps.data.add_blank:
            sequence = commons.intersperse(sequence, 0)
        tokens = torch.LongTensor(sequence).to(device).unsqueeze(0)
        lengths = torch.LongTensor([tokens.size(1)]).to(device)
        torch.manual_seed(20260722)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(20260722)
        waveform = model.infer(
            tokens,
            lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0,
            max_len=4000,
        )[0][0, 0]
        return waveform.numel() / int(hps.data.sampling_rate)

    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    rows, peak = timed_runs(prompts, synthesize, sync)
    return str(device), load_seconds, rows, peak


def benchmark_kitten(args: argparse.Namespace, prompts: list[dict[str, str]]) -> tuple[str, float, list[dict[str, Any]], int]:
    import numpy as np
    from kittentts import KittenTTS

    voice = "Bruno" if args.system.endswith("bruno") else "Hugo"
    started = time.perf_counter()
    model = KittenTTS("KittenML/kitten-tts-nano-0.8")
    load_seconds = time.perf_counter() - started

    def synthesize(text: str) -> float:
        audio = np.asarray(model.generate(text, voice=voice, speed=1.0))
        return audio.size / 24000.0

    rows, peak = timed_runs(prompts, synthesize)
    return ",".join(model.model.session.get_providers()), load_seconds, rows, peak


def benchmark_piper(args: argparse.Namespace, prompts: list[dict[str, str]]) -> tuple[str, float, list[dict[str, Any]], int]:
    from piper.voice import PiperVoice

    started = time.perf_counter()
    voice = PiperVoice.load(args.piper_model)
    load_seconds = time.perf_counter() - started

    def synthesize(text: str) -> float:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            voice.synthesize_wav(text, handle)
        buffer.seek(0)
        with wave.open(buffer, "rb") as handle:
            return handle.getnframes() / handle.getframerate()

    rows, peak = timed_runs(prompts, synthesize)
    return ",".join(voice.session.get_providers()), load_seconds, rows, peak


def benchmark_supertonic(args: argparse.Namespace, prompts: list[dict[str, str]]) -> tuple[str, float, list[dict[str, Any]], int]:
    import onnxruntime as ort
    import soundfile as sf
    from supertonic import TTS

    steps = 3 if args.system.endswith("3step") else 8
    started = time.perf_counter()
    tts = TTS(auto_download=True)
    style = tts.get_voice_style(voice_name="M2")
    load_seconds = time.perf_counter() - started

    def synthesize(text: str) -> float:
        waveform, _ = tts.synthesize(
            text,
            voice_style=style,
            lang="en",
            total_steps=steps,
            speed=1.0,
        )
        # The SDK returns the same waveform that save_audio would write.
        sample_rate = int(getattr(tts, "sample_rate", 24000))
        try:
            samples = waveform.shape[-1]
        except AttributeError:
            samples = len(waveform)
        return samples / sample_rate

    rows, peak = timed_runs(prompts, synthesize)
    sessions = (
        tts.model.dp_ort,
        tts.model.text_enc_ort,
        tts.model.vector_est_ort,
        tts.model.vocoder_ort,
    )
    active_providers = sorted(
        {provider for session in sessions for provider in session.get_providers()}
    )
    return ",".join(active_providers), load_seconds, rows, peak


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched warm end-to-end TTS runtime benchmark.")
    parser.add_argument("--system", required=True, choices=(
        "inflect-micro-v2", "inflect-nano-v2",
        "kitten-nano-bruno", "kitten-nano-hugo",
        "piper-ryan-low", "piper-danny-low",
        "supertonic3-james-3step", "supertonic3-james-8step",
    ))
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vits-root", type=Path)
    parser.add_argument("--frontend-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Self-contained Inflect release directory. Preferred over raw training paths.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        help="Fixed PyTorch intra-op thread count for matched local CPU runs.",
    )
    parser.add_argument("--piper-model", type=Path)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts, args.limit)
    if args.system.startswith("inflect-"):
        provider, load_seconds, rows, peak = benchmark_inflect(args, prompts)
    elif args.system.startswith("kitten-"):
        provider, load_seconds, rows, peak = benchmark_kitten(args, prompts)
    elif args.system.startswith("piper-"):
        provider, load_seconds, rows, peak = benchmark_piper(args, prompts)
    else:
        provider, load_seconds, rows, peak = benchmark_supertonic(args, prompts)
    report = summarize(args.system, provider, load_seconds, rows, peak)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
