from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Model:
    repo: str
    sibling_repo: str
    title: str
    sibling_title: str
    role: str
    limit: str
    params: str
    params_short: str
    weights: str
    utmos: str
    utmos_ci: str
    qwen_wer: str
    nemo_wer: str
    whisper_wer: str
    consensus_wer: str
    asr_mean: str
    preference: str
    record: str
    rtf: str
    realtime: str
    median: str
    p95: str
    load: str
    latent: str
    hidden: str
    layers_heads: str
    ffn: str
    flow: str
    decoder: str


MODELS = (
    Model(
        repo="Inflect-Micro-v2",
        sibling_repo="Inflect-Nano-v2",
        title="Inflect-Micro-v2",
        sibling_title="Inflect-Nano-v2",
        role="quality-focused",
        limit="10M",
        params="9,356,513",
        params_short="9.36M",
        weights="37.53 MB",
        utmos="4.395",
        utmos_ci="4.381–4.408",
        qwen_wer="2.52%",
        nemo_wer="5.45%",
        whisper_wer="2.73%",
        consensus_wer="3.99%",
        asr_mean="3.57%",
        preference="66.2%",
        record="21 wins · 10 losses · 3 ties",
        rtf="0.635",
        realtime="1.58×",
        median="2.402 s",
        p95="7.281 s",
        load="2.774 s",
        latent="192",
        hidden="96",
        layers_heads="3 / 2",
        ffn="768",
        flow="4",
        decoder="320",
    ),
    Model(
        repo="Inflect-Nano-v2",
        sibling_repo="Inflect-Micro-v2",
        title="Inflect-Nano-v2",
        sibling_title="Inflect-Micro-v2",
        role="portability-focused",
        limit="4M",
        params="3,966,721",
        params_short="3.97M",
        weights="15.97 MB",
        utmos="4.386",
        utmos_ci="4.372–4.399",
        qwen_wer="2.79%",
        nemo_wer="5.63%",
        whisper_wer="2.65%",
        consensus_wer="4.21%",
        asr_mean="3.69%",
        preference="63.9%",
        record="22 wins · 12 losses · 2 ties",
        rtf="0.630",
        realtime="1.59×",
        median="1.797 s",
        p95="7.207 s",
        load="0.192 s",
        latent="128",
        hidden="72",
        layers_heads="3 / 2",
        ffn="384",
        flow="4",
        decoder="192",
    ),
)


TRANSCRIPTS = (
    ("Conversational", "conversational.wav", "It wasn't until later that I realized what had actually happened."),
    ("Punctuation", "punctuation.wav", "First, close the window; second, turn off the lamp; finally, lock the door."),
    ("Numbers", "numbers.wav", "The package weighs twelve point six kilograms and arrived on July twenty-first."),
    ("Names and places", "names_places.wav", "Gwendolyn photographed the eucalyptus trees outside Ljubljana."),
    ("Technical", "technical.wav", "The system runs on three core components that all have to stay in sync."),
)


def sample_rows(model: Model) -> str:
    base = f"https://huggingface.co/owensong/{model.repo}/resolve/main/samples/male"
    rows = []
    for label, filename, transcript in TRANSCRIPTS:
        url = f"{base}/{filename}"
        rows.append(
            f'| **{label}** | {transcript} | <audio controls preload="metadata" '
            f'src="{url}"></audio> |'
        )
    return "\n".join(rows)


def navigation(model: Model) -> str:
    sibling_asset = "nano" if model.repo == "Inflect-Micro-v2" else "micro"
    sibling_alt = "Inflect Nano v2" if sibling_asset == "nano" else "Inflect Micro v2"
    return f"""<p align="center">
  <a href="https://huggingface.co/spaces/owensong/Inflect-v2"><img alt="Live playground" src="assets/nav/playground.svg" width="168"></a>
  <a href="https://github.com/owenawsong/Inflect"><img alt="GitHub" src="assets/nav/github.svg" width="168"></a>
  <a href="https://huggingface.co/owensong/{model.sibling_repo}"><img alt="{sibling_alt}" src="assets/nav/{sibling_asset}.svg" width="168"></a>
  <a href="docs/FINETUNING.md"><img alt="Fine-tuning guide" src="assets/nav/finetuning.svg" width="168"></a>
  <a href="https://discord.gg/CVJYedvzvp"><img alt="Inflect Discord" src="assets/nav/discord.svg" width="168"></a>
</p>"""


def card(model: Model) -> str:
    return f"""---
license: apache-2.0
language:
  - en
pipeline_tag: text-to-speech
model_name: {model.title}
metrics:
  - wer
tags:
  - text-to-speech
  - speech-synthesis
  - local-tts
  - cpu
  - edge-ai
  - small-model
  - pytorch
  - vits
  - 24khz
thumbnail: assets/inflect-v2-repository-hero.png
inference: false
---

![Inflect v2](assets/inflect-v2-repository-hero.png)

<h1 align="center">{model.title}</h1>
<p align="center"><strong>Complete local text-to-waveform speech synthesis under {model.limit} parameters.</strong><br>
24 kHz English TTS with the neural waveform decoder already inside the model.</p>

{navigation(model)}

<p align="center"><strong>{model.params_short} complete parameters</strong> · <strong>{model.weights} FP32 weights</strong> · <strong>one fixed English voice</strong> · <strong>no external vocoder</strong></p>

---

> **Small enough to ship. Complete enough to stand alone.** Every published parameter is part of the local text-to-waveform path: text encoding, duration prediction, latent synthesis, and the integrated 24 kHz waveform decoder. There is no second vocoder download, server model, or external generation service.

<details>
<summary><strong>Explore this model card</strong></summary>

| Start here | Technical detail |
| --- | --- |
| [Listen](#listen) | [Architecture](#architecture-and-parameter-budget) |
| [Evaluation](#evaluation) | [Controls and long text](#controls-determinism-and-long-text) |
| [Choose Micro or Nano](#choose-the-right-inflect) | [Data and adaptation](#data-voice-and-adaptation-status) |
| [Run locally](#run-locally) | [Exports and quantization](docs/EXPORTS.md) |
| [Package map](#package-map) | [Evaluation and raw protocol](docs/EVALUATION.md) |
| [Limitations](#limitations) | [Deployment guide](docs/DEPLOYMENT.md) |

</details>

## Listen

These are held-out text generations, not reconstructions of training audio. Each transcript is shown exactly as passed to the public frontend.

| Test | Exact transcript | Generated audio |
| --- | --- | --- |
{sample_rows(model)}

## Evaluation

No single metric captures TTS quality. Inflect v2 reports **human preference**, **predicted naturalness**, **multi-ASR intelligibility**, **complete footprint**, and **runtime** separately rather than compressing them into one unverifiable score.

| Community preference ↑ | UTMOS22 ↑ | Two-ASR semantic WER ↓ | Complete FP32 weights ↓ |
| ---: | ---: | ---: | ---: |
| **{model.preference}** | **{model.utmos}** | **{model.consensus_wer}** | **{model.weights}** |

The headline row always refers to **{model.title}**. Detailed competitor results and protocol boundaries are kept visible below.

> **Comparator policy.** Inflect is tested against serious compact and on-device baselines, not deliberately weak systems: [KittenTTS Nano](https://huggingface.co/KittenML/kitten-tts-nano-0.8), [Piper Low](https://huggingface.co/rhasspy/piper-voices), and [Supertonic 3](https://huggingface.co/Supertone/supertonic-3). Under this package-level protocol, every comparator has a larger deployable weight footprint than both Inflect releases. That makes the comparison demanding; it does not imply that one metric establishes universal superiority.

### 1. Human blind preference

![Community blind listening](assets/evidence/human-preference.svg)

{model.title} recorded a **{model.preference} preference rate** ({model.record}) in the final anonymous community study. Systems were hidden, left/right order was randomized, and ties count as half a win. This is descriptive community evidence, not formal MOS.

### 2. Predicted naturalness versus footprint

![Predicted quality versus footprint](assets/evidence/quality-vs-footprint.svg)

The UTMOS22 run used 500 identical unseen prompts per voice. KittenTTS and Piper are equal-weight two-voice means; their observed voice ranges appear as whiskers. Supertonic 3-step is reported below the plotted range rather than flattening every other system.

**{model.title}: {model.utmos} UTMOS22**, 95% bootstrap CI **{model.utmos_ci}**. UTMOS22 is a learned predictor, not human MOS.

### 3. Intelligibility on unseen text

![Two-ASR semantic WER consensus](assets/evidence/asr-consensus.svg)

The headline score is the equal-weight mean of Qwen3-ASR and Nemotron 3.5 corpus WER for **every** system. Whisper is excluded consistently from the headline because it produced insertion-heavy hallucinations on a subset of otherwise intelligible Supertonic 8-step clips. It is not deleted: the complete three-ASR evidence remains below.

<details>
<summary><strong>Open the complete three-ASR audit</strong></summary>

![Semantic WER across Qwen3-ASR, Nemotron 3.5, and Whisper large-v3](assets/evidence/modern400-three-asr.svg)

| System / voice | Qwen3-ASR ↓ | Nemotron 3.5 ↓ | Whisper large-v3 ↓ |
| --- | ---: | ---: | ---: |
| **Inflect-Micro-v2** | **2.52%** | **5.45%** | **2.73%** |
| **Inflect-Nano-v2** | **2.79%** | **5.63%** | **2.65%** |
| KittenTTS Nano · Bruno | 2.15% | 3.96% | 2.17% |
| KittenTTS Nano · Hugo | 2.39% | 3.80% | 2.11% |
| Piper Low · Danny | 2.62% | 5.60% | 2.55% |
| Piper Low · Ryan | 2.81% | 5.51% | 2.87% |
| Supertonic 3 · M2 · 3-step | 3.03% | 6.04% | 3.22% |
| Supertonic 3 · M2 · 8-step | 2.05% | 3.56% | 8.08% |

For {model.title}, the individual results are **{model.qwen_wer} Qwen3-ASR**, **{model.nemo_wer} Nemotron 3.5**, and **{model.whisper_wer} Whisper large-v3**. The former three-model mean, **{model.asr_mean}**, is retained only as a descriptive audit value and is not used as the headline score.

</details>

<details>
<summary><strong>Open evaluator robustness and error-category diagnostics</strong></summary>

![ASR evaluator robustness](assets/evidence/asr-robustness.svg)

![Semantic WER by prompt category](assets/evidence/category-semantic-wer.svg)

These views are diagnostics, not additional leaderboards. They show where the
recognizers disagree and which prompt categories still produce recoverable
transcription errors.

</details>

### 4. Runtime status

The release supports CPU and CUDA inference through the same public API. A
previous local CPU table has been withdrawn because background host saturation
made its cross-system throughput figures non-reproducible. A clean-host rerun
will be published with the complete environment, thread policy, warmup count,
matched prompts, and raw per-utterance timings. Until then, this card makes no
headline CPU-speed claim.

### 5. Complete weight footprint

![Complete deployable model footprint](assets/evidence/model-footprint.svg)

Voice variants sharing the same weights are merged. Inflect totals include the integrated waveform decoder.

<details>
<summary><strong>Open the frozen evaluation protocol</strong></summary>

- Modern400 uses 400 identical unseen English prompts per system: 200 fixed modern/stress prompts plus 200 deterministic FLEURS `en_us` test prompts.
- Exact-text exclusion was checked against 87,362 training transcripts.
- All ASR inputs are resampled to 16 kHz and scored with the same disclosed English normalizer.
- UTMOS22 uses `tarepan/SpeechMOS` v1.2.0 on a separate 500-prompt generation set.
- Headline intervals use 10,000 bootstrap samples.
- The Modern400 corpus SHA-256 is `b7504ce2dce44a2da82770a6a5dfd2a034fe17e2113980f8a69663ade417a34c`.
- Prompts, hypotheses, compressed row-level reports, and summaries ship under `evaluation/final/`.
- Runtime is evaluated separately because framework, thread policy, compilation,
  and host load can dominate small-model comparisons.

</details>

---

## Choose the right Inflect

| | **Inflect-Nano-v2** | **Inflect-Micro-v2** |
| --- | ---: | ---: |
| Complete parameters | 3,966,721 | 9,356,513 |
| FP32 weights | 15.97 MB | 37.53 MB |
| Positioning | Smallest practical footprint | Strongest Inflect v2 quality |
| 24 kHz waveform decoder | Included | Included |
| Python API and frontend | Same | Same |

**{model.title}** is the {model.role} member of the family. Both models use the same public API and complete text-to-waveform packaging.

## Run locally

### Install

```bash
git clone https://huggingface.co/owensong/{model.repo}
cd {model.repo}
python -m pip install -r requirements.txt
```

### Python

```python
from inference import InflectTTS

tts = InflectTTS(".", device="cpu")
tts.save(
    "A small voice can still have something meaningful to say.",
    "sample.wav",
    speed=1.0,
    variation=0.667,
    seed=7,
)
```

### Download through the Hub

```python
import sys
from huggingface_hub import snapshot_download

model_dir = snapshot_download("owensong/{model.repo}")
sys.path.insert(0, model_dir)

from inference import InflectTTS

tts = InflectTTS(model_dir, device="cpu")
sample_rate, waveform = tts.synthesize("The complete model runs locally.")
```

The result is a 24 kHz mono `float32` waveform. Long input is split at punctuation-aware boundaries, synthesized chunk by chunk, and joined with controlled pauses.

## Why Inflect

| **Small enough to ship** | **Complete by design** |
| --- | --- |
| {model.weights} of FP32 model weights, with no server dependency. | Text frontend, acoustic generator, duration model, and waveform decoder ship together. |
| **Built for local inference** | **Measured, not hand-picked** |
| CPU-ready PyTorch runtime, deterministic seeds, and punctuation-aware long text. | Frozen prompts, raw hypotheses, intervals, hashes, and per-system reports are included. |

<details>
<summary id="architecture-and-parameter-budget"><strong>Architecture and parameter budget</strong></summary>

Inflect v2 is a parameter-efficient VITS-family end-to-end text-to-waveform generator with an English phoneme frontend, monotonic alignment, stochastic latent synthesis, residual coupling flow, and an integrated alias-reduced neural waveform decoder.

| Component | {model.title} |
| --- | ---: |
| Latent channels | {model.latent} |
| Text hidden channels | {model.hidden} |
| Encoder layers / heads | {model.layers_heads} |
| Feed-forward channels | {model.ffn} |
| Flow coupling blocks | {model.flow} |
| Initial decoder channels | {model.decoder} |
| Upsample rates | 8, 8, 2, 2 |
| Training segment | 16,384 samples |
| Output | 24 kHz mono waveform |

The release describes the deployable architecture. Private corpus-construction and optimization details are not part of this open-weight package.

</details>

<details>
<summary id="controls-determinism-and-long-text"><strong>Controls, determinism, and long text</strong></summary>

| Control | Default | Public range | Meaning |
| --- | ---: | ---: | --- |
| `speed` | `1.0` | `0.5–2.0` | Lower is slower; higher is faster. |
| `variation` | `0.667` | `0.0–1.0` | Lower is steadier; higher samples more latent variation. |
| `seed` | `0` | integer | Repeats the same stochastic sample on the same runtime stack. |

Long passages are punctuation-aware chunks, not one unlimited autoregressive pass. Chunk boundaries receive short pauses and edge fades. See [`docs/API.md`](docs/API.md) for waveform contracts and concurrency notes.

</details>

<details>
<summary id="data-voice-and-adaptation-status"><strong>Data, voice, and adaptation status</strong></summary>

The release contains one fixed synthetic English voice. The package does not redistribute a real-speaker recording corpus, does not claim the voice as the identity of a real person, and requires no reference audio or external model at inference.

This release is inference-first. New-voice and new-language adaptation are **not currently validated or supported**. A new voice would replace the fixed speaker rather than add a selectable speaker; language adaptation also requires rebuilding normalization, phonemes, symbols, embeddings, and training data. See [`docs/DATA_AND_VOICE.md`](docs/DATA_AND_VOICE.md) and [`docs/FINETUNING.md`](docs/FINETUNING.md).

</details>

## Package map

| Path | Purpose |
| --- | --- |
| `model.pth` | Inference-only generator checkpoint |
| `config.json` | Architecture and audio configuration; also the Hub download-count query file |
| `inference.py` | Public Python API and CLI |
| `inflect_vits_frontend.py` | English normalization, phonemization, and punctuation frontend |
| `runtime/` | Self-contained model implementation |
| `samples/` | Held-out example generations |
| `evaluation/final/` | Frozen benchmark prompts, reports, and protocol artifacts |
| `docs/` | API, deployment, evaluation, adaptation, and export documentation |
| `release_manifest.json` | File sizes and SHA-256 hashes |

## Limitations

- English only, with one fixed male voice. This is not zero-shot voice cloning.
- Unfamiliar phrasing can become flatter, less expressive, or less stable.
- Numbers, abbreviations, homographs, and uncommon names remain frontend- and context-sensitive.
- Long passages use punctuation-aware chunking; transitions can differ from a native long-form model pass.
- Stochastic variation can alter timing and pronunciation. Fix the seed for comparisons.
- UTMOS22 and ASR scores do not replace controlled human MOS or MUSHRA-style evaluation.
- Not validated for medical, legal, emergency, or accessibility-critical communication.

## Responsible use

Do not use the included voice to impersonate a real person, deceive listeners, or create fraudulent content. Disclose synthetic speech where the context could otherwise mislead. Users are responsible for applicable laws and the Apache-2.0 license.

## License, integrity, and attribution

Original Inflect code and weights are released under Apache-2.0. Bundled third-party components retain their own notices in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). `release_manifest.json` records packaged file sizes and SHA-256 hashes.

### Private training scope and contact

Inflect v2 is an **open-weight** release. Deployable weights, inference code, frontend code, evaluation prompts, and release reports are public. The training corpus-generation pipeline, private filtering infrastructure, and full optimization recipe are not part of the public package.

Owen Song may share additional technical context privately for credible research, collaboration, reproducibility, or deployment inquiries when the request has a clear purpose and does not conflict with licensing or data-provenance constraints.

- **Discord:** `b111ue` — fastest for informal technical questions
- **Community server:** [discord.gg/CVJYedvzvp](https://discord.gg/CVJYedvzvp)
- **Email:** [owen.aw.song@gmail.com](mailto:owen.aw.song@gmail.com) — preferred for professional inquiries

## Citation

```bibtex
@software{{song2026{model.repo.lower().replace("-", "")},
  author = {{Owen Song}},
  title = {{{model.title}: Complete Local Text-to-Waveform TTS Under {model.limit} Parameters}},
  year = {{2026}},
  url = {{https://huggingface.co/owensong/{model.repo}}}
}}
```

<p align="center"><sub>Designed and developed independently by Owen Song · open weights · Apache-2.0 · complete local text-to-waveform inference</sub></p>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    for model in MODELS:
        path = args.root / model.repo / "README.md"
        path.write_text(card(model), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
