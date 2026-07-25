from __future__ import annotations

import logging
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch

try:
    import spaces
except ImportError:  # Local smoke tests do not require the ZeroGPU shim.
    class _Spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorate(function):
                return function
            return decorate
    spaces = _Spaces()


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(ROOT))

import commons
import utils
from inflect_vits_frontend import run_vits_frontend
from models import SynthesizerTrn
from text import cleaned_text_to_sequence
from text.symbols import symbols


@dataclass(frozen=True)
class ModelSpec:
    label: str
    short_label: str
    params: str
    checkpoint: Path
    config: Path
    revision: str


SPECS = {
    "Inflect Micro v2": ModelSpec(
        label="Inflect Micro v2",
        short_label="Micro",
        params="9.36M",
        checkpoint=ROOT / "models" / "micro" / "model.pth",
        config=ROOT / "models" / "micro" / "config.json",
        revision="3eede065",
    ),
    "Inflect Nano v2": ModelSpec(
        label="Inflect Nano v2",
        short_label="Nano",
        params="3.96M",
        checkpoint=ROOT / "models" / "nano" / "model.pth",
        config=ROOT / "models" / "nano" / "config.json",
        revision="bfca4684",
    ),
}


def split_text(text: str, limit: int = 280) -> list[str]:
    normalized = " ".join(text.split())
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?;:])\s+", normalized)
        if part.strip()
    ]
    chunks: list[str] = []
    for sentence in sentences or [normalized]:
        while len(sentence) > limit:
            search = sentence[: limit + 1]
            punctuation = max(search.rfind(mark) for mark in (",", ";", ":"))
            split_at = (
                punctuation + 1
                if punctuation >= limit // 2
                else sentence.rfind(" ", 0, limit + 1)
            )
            if split_at < limit // 2:
                split_at = limit
            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def boundary_pause_seconds(chunk: str) -> float:
    ending = chunk.rstrip()[-1:] if chunk.strip() else ""
    return {
        "?": 0.28,
        "!": 0.24,
        ".": 0.22,
        ";": 0.16,
        ":": 0.13,
        ",": 0.09,
    }.get(ending, 0.08)


def edge_fade(waveform: np.ndarray, sample_rate: int, milliseconds: float = 5.0) -> np.ndarray:
    frames = min(round(sample_rate * milliseconds / 1000.0), waveform.size // 2)
    if frames <= 0:
        return waveform
    output = waveform.copy()
    ramp = np.linspace(0.0, 1.0, frames, endpoint=True, dtype=np.float32)
    output[:frames] *= ramp
    output[-frames:] *= ramp[::-1]
    return output


class Engine:
    def __init__(self, spec: ModelSpec) -> None:
        if not spec.config.is_file():
            raise FileNotFoundError(f"Missing release configuration for {spec.label}")
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(f"Missing release weights for {spec.label}")
        self.spec = spec
        self.hps = utils.get_hparams_from_file(str(spec.config))
        self.model = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model,
        ).eval()
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        try:
            root_logger.setLevel(logging.WARNING)
            utils.load_checkpoint(str(spec.checkpoint), self.model, None)
        finally:
            root_logger.setLevel(previous_level)
        self.device = torch.device("cpu")
        self.sample_rate = int(self.hps.data.sampling_rate)
        self.lock = threading.Lock()

    def tokens(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        phonemes = run_vits_frontend(text).phoneme_text
        sequence = cleaned_text_to_sequence(phonemes)
        if self.hps.data.add_blank:
            sequence = commons.intersperse(sequence, 0)
        if not sequence:
            raise ValueError("The phoneme frontend produced no speakable tokens.")
        tokens = torch.LongTensor(sequence).to(self.device).unsqueeze(0)
        lengths = torch.LongTensor([tokens.size(1)]).to(self.device)
        return tokens, lengths

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        speed: float,
        variation: float,
        pitch_steps: float,
        seed: int,
    ) -> tuple[int, np.ndarray]:
        chunks = split_text(text)
        waveforms: list[np.ndarray] = []
        with self.lock:
            self.device = torch.device("cuda")
            self.model.to(self.device)
            try:
                for index, chunk in enumerate(chunks):
                    if index:
                        waveforms.append(
                            np.zeros(
                                round(self.sample_rate * boundary_pause_seconds(chunks[index - 1])),
                                dtype=np.float32,
                            )
                        )
                    tokens, lengths = self.tokens(chunk)
                    torch.manual_seed(seed + index)
                    torch.cuda.manual_seed_all(seed + index)
                    waveform = self.model.infer(
                        tokens,
                        lengths,
                        noise_scale=variation,
                        noise_scale_w=0.8,
                        length_scale=1.0 / speed,
                        max_len=4000,
                    )[0][0, 0].float().cpu().numpy()
                    waveforms.append(edge_fade(waveform, self.sample_rate))
            finally:
                self.model.to("cpu")
                self.device = torch.device("cpu")
                torch.cuda.empty_cache()
        audio = np.concatenate(waveforms)
        if abs(pitch_steps) >= 0.01:
            audio = librosa.effects.pitch_shift(
                audio,
                sr=self.sample_rate,
                n_steps=pitch_steps,
                bins_per_octave=12,
                res_type="soxr_hq",
                scale=False,
            )
            audio = librosa.util.fix_length(audio, size=sum(len(part) for part in waveforms))
        return self.sample_rate, np.clip(audio, -1.0, 1.0)


ENGINES = {label: Engine(spec) for label, spec in SPECS.items()}


def validate(
    text: str,
    speed: float,
    variation: float,
    pitch_steps: float,
    seed: int,
) -> tuple[str, float, float, float, int]:
    text = " ".join((text or "").split())
    if not text:
        raise gr.Error("Enter something for Inflect to say.")
    return text, float(speed), float(variation), float(pitch_steps), int(seed)


@spaces.GPU(duration=120)
def synthesize_one(
    text: str,
    model_name: str,
    speed: float,
    variation: float,
    pitch_steps: float,
    seed: int,
):
    text, speed, variation, pitch_steps, seed = validate(
        text, speed, variation, pitch_steps, seed
    )
    engine = ENGINES[model_name]
    started = time.perf_counter()
    audio = engine.synthesize(text, speed, variation, pitch_steps, seed)
    seconds = len(audio[1]) / audio[0]
    wall = time.perf_counter() - started
    rtf = wall / seconds
    chunks = len(split_text(text))
    return audio, (
        f"**{model_name}** · fixed English voice  \n"
        f"`{engine.spec.params}` parameters · `{seconds:.2f}s` audio · "
        f"`{wall:.2f}s` generation · `RTF {rtf:.3f}` · "
        f"`{len(text)}` characters · `{chunks}` text chunk{'s' if chunks != 1 else ''} · "
        f"pitch `{pitch_steps:+.2f} st` · "
        f"seed `{seed}` · weights `{engine.spec.revision}`"
    )


@spaces.GPU(duration=120)
def synthesize_both(
    text: str,
    speed: float,
    variation: float,
    pitch_steps: float,
    seed: int,
):
    text, speed, variation, pitch_steps, seed = validate(
        text, speed, variation, pitch_steps, seed
    )
    started = time.perf_counter()
    micro = ENGINES["Inflect Micro v2"].synthesize(
        text, speed, variation, pitch_steps, seed
    )
    micro_wall = time.perf_counter() - started
    started = time.perf_counter()
    nano = ENGINES["Inflect Nano v2"].synthesize(
        text, speed, variation, pitch_steps, seed
    )
    nano_wall = time.perf_counter() - started
    micro_seconds = len(micro[1]) / micro[0]
    nano_seconds = len(nano[1]) / nano[0]
    return micro, nano, (
        f"Same text and seed `{seed}` · "
        f"Micro `{micro_wall:.2f}s / {micro_seconds:.2f}s audio` · "
        f"Nano `{nano_wall:.2f}s / {nano_seconds:.2f}s audio`"
    )


CSS = """
:root{color-scheme:dark;--paper:#111412;--panel:#181c19;--panel-2:#141815;--ink:#f3f4ef;--muted:#9ea8a0;--line:#303732;--signal:#2486ff;--signal-hover:#52a0ff;--signal-soft:#15263b}
html,body,.gradio-container,.dark{background:var(--paper)!important;color:var(--ink)!important;color-scheme:dark!important}
body,.gradio-container,.gradio-container button,.gradio-container input,.gradio-container textarea{font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important}
.gradio-container{width:min(100%,1640px)!important;max-width:none!important;margin:auto!important;padding:0 clamp(14px,2.2vw,34px) 54px!important;box-sizing:border-box!important}
.gradio-container>.main{padding:0!important}
.gradio-container *{box-sizing:border-box}
.hero-wrap,.hero-wrap.block,.hero-wrap>.wrap{padding:0!important;border:0!important;box-shadow:none!important;background:transparent!important}
.hero{padding:18px 2px 13px;border-bottom:1px solid var(--line);margin-bottom:2px}
.hero-top{display:flex;align-items:center;justify-content:space-between;gap:20px}
.eyebrow{color:var(--signal);font:700 11px/1 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.13em;text-transform:uppercase}
.links{display:flex;align-items:center;gap:18px;font-size:12px;font-weight:700}
.links a{color:var(--muted)!important;text-decoration:none}.links a:hover{color:var(--ink)!important}
.hero h1{max-width:1040px;margin:15px 0 8px;color:var(--ink);font:650 clamp(40px,4.5vw,56px)/.96 Inter,ui-sans-serif,sans-serif;letter-spacing:-.055em}
.hero>p{max-width:1000px;margin:0;color:var(--muted)!important;font-size:13px;line-height:1.48}
.model-facts{display:flex;align-items:stretch;margin-top:13px;border-top:1px solid var(--line)}
.model-fact{flex:1;padding:8px 24px 0 0;color:var(--muted)!important;font-size:11px;line-height:1.3}
.model-fact+.model-fact{padding-left:20px;border-left:1px solid var(--line)}
.model-fact strong{display:block;margin-bottom:2px;color:var(--ink);font-size:14px;line-height:1.2}
.zerogpu-note{max-width:1080px;margin:10px 0 0!important;padding:7px 11px;border-left:2px solid var(--signal);background:#151b20;color:var(--muted)!important;font-size:11px!important;line-height:1.4!important}
.workbench{padding:14px 0 4px!important;border:0!important;background:transparent!important}
.workbench.block,.workbench>.wrap{border:0!important;box-shadow:none!important;background:transparent!important}
.control-row{align-items:stretch!important;gap:20px!important;margin-top:6px!important;padding:13px 14px 12px!important;border:1px solid var(--line)!important;border-radius:3px!important;background:var(--panel-2)!important}
.control-row>div{min-width:0!important}
.control-row .form{height:100%!important;border:0!important;background:transparent!important}
.control-row label>span:first-child{font-size:12px!important;font-weight:750!important;letter-spacing:.015em!important}
.action-row{justify-content:center!important;gap:12px!important;margin:12px auto 0!important;max-width:840px!important}
.action-row button{min-height:48px!important}
.comparison{margin-top:22px!important;padding-top:20px!important;border-top:1px solid var(--line)!important}
.comparison-title{margin-bottom:9px!important}
.gradio-container .block,.gradio-container .form{border-radius:3px!important;background:var(--panel-2)!important;border-color:var(--line)!important;color:var(--ink)!important}
.gradio-container .prose,.gradio-container .markdown-body,.gradio-container label,.gradio-container span,.gradio-container p,.gradio-container h1,.gradio-container h2,.gradio-container h3{color:var(--ink)!important}
.gradio-container label{background:transparent!important}
.gradio-container .wrap{background:var(--panel-2)!important}
.gradio-container select{min-height:48px!important;background:#101411!important;color:var(--ink)!important;border-color:var(--line)!important;text-align:left!important}
.gradio-container input,.gradio-container textarea{background:#101411!important;color:var(--ink)!important;border-color:var(--line)!important}
.prompt-input textarea{font-size:16px!important;line-height:1.55!important;min-height:104px!important;padding:15px!important}
.gradio-container button:not(.primary){background:var(--panel-2)!important;color:var(--ink)!important;border-color:var(--line)!important}
.gradio-container button:not(.primary):hover{border-color:#527aa6!important;background:#1b242d!important}
button.primary{min-height:46px!important;border:0!important;border-radius:1px!important;background:var(--signal)!important;color:#fff!important;font-weight:800!important}
button.primary *{color:#fff!important}button.primary:hover{background:var(--signal-hover)!important}
.advanced{margin:10px 0 3px!important;background:#131814!important;border-left:2px solid #344138!important}
.advanced>button{font-size:12px!important;font-weight:750!important;letter-spacing:.01em!important}
.output-audio{margin-top:8px!important}
.outputmeta{padding:11px 13px!important;background:var(--signal-soft)!important;border:1px solid #294e78!important}
.outputmeta *{color:#dceaff!important}
.helper-copy{margin:8px 2px 4px!important;color:var(--muted)!important;font-size:12px!important}
.fineprint{color:var(--muted)!important;font-size:12px;line-height:1.55;border-top:1px solid var(--line);padding-top:16px;margin-top:22px}
footer{display:none!important}
@media(max-width:760px){
html,body{overflow-x:hidden!important}.gradio-container{width:100%!important;min-width:0!important;padding:0 14px 38px!important}
.hero{padding-top:22px}.hero-top{align-items:flex-start;flex-direction:column;gap:12px}.links{gap:14px;font-size:11px}
.hero h1{margin-top:18px;font-size:42px}.model-facts{display:grid;grid-template-columns:1fr 1fr}.model-fact{padding:11px 10px 9px 0}.model-fact+.model-fact{padding-left:10px}.model-fact:nth-child(3){padding-left:0;border-left:0;border-top:1px solid var(--line)}.model-fact:nth-child(4){border-top:1px solid var(--line)}
.workbench{min-width:0!important;padding-top:18px!important}.control-row,.action-row,.comparison .row{flex-direction:column!important;min-width:0!important}
.control-row>*,.action-row>*,.comparison .row>*{width:100%!important;min-width:0!important}.gradio-container .wrap,.gradio-container .form,.gradio-container .block{min-width:0!important}
}
"""


PROMPTS = [
    "Wait, are you actually being for real? I thought the train left twenty minutes ago!",
    "Does punctuation work? Yes: commas, semicolons, and periods should create sensible boundaries.",
    "Professor Nguyen mailed the fluorescent poster to Reykjavik on Thursday.",
    "Although the weather changed without warning, the musicians finished packing before anyone opened the enormous glass door.",
]

with gr.Blocks(title="Inflect v2 · local speech playground") as demo:
    gr.HTML("""
    <header class="hero">
      <div class="hero-top">
        <span class="eyebrow">Inflect v2 · live ZeroGPU synthesis</span>
        <nav class="links">
          <a href="https://huggingface.co/owensong/Inflect-Micro-v2">Micro model</a>
          <a href="https://huggingface.co/owensong/Inflect-Nano-v2">Nano model</a>
          <a href="https://github.com/owenawsong/Inflect">GitHub</a>
        </nav>
      </div>
      <h1>Small models.<br>Complete speech.</h1>
      <p>Generate 24 kHz English speech with the published Inflect-Micro-v2 and Inflect-Nano-v2 checkpoints. The full text-to-waveform path runs here: no reference clip, external vocoder, hosted teacher, or prerecorded result.</p>
      <div class="model-facts">
        <div class="model-fact"><strong>9.36M · Micro</strong>quality-first model</div>
        <div class="model-fact"><strong>3.96M · Nano</strong>footprint-first model</div>
        <div class="model-fact"><strong>24 kHz WAV</strong>complete waveform output</div>
        <div class="model-fact"><strong>Local stack</strong>text normalization through audio</div>
      </div>
      <p class="zerogpu-note"><strong>ZeroGPU note:</strong> each generation has a time limit. Inflect automatically splits text at sentence and punctuation boundaries, but very long documents should be submitted in smaller sections.</p>
    </header>
    """, elem_classes="hero-wrap")
    with gr.Column(elem_classes="workbench"):
        text = gr.Textbox(
            value=PROMPTS[0],
            lines=4,
            max_lines=14,
            label="Text",
            placeholder="Write a sentence, paragraph, or script...",
            elem_classes="prompt-input",
        )
        with gr.Row(equal_height=True, elem_classes="control-row"):
            model = gr.Dropdown(
                list(SPECS),
                value="Inflect Micro v2",
                label="Model",
                info="Micro prioritizes quality; Nano prioritizes footprint.",
                scale=4,
            )
            speed = gr.Slider(
                0.7,
                1.35,
                value=1.0,
                step=0.01,
                label="Speaking speed",
                info="1.00 is the trained default.",
                scale=4,
            )
            variation = gr.Slider(
                0.0,
                1.0,
                value=0.667,
                step=0.001,
                label="Delivery variation",
                info="Lower is steadier. Higher gives each take more variation.",
                scale=5,
            )
        with gr.Accordion(
            "Advanced controls",
            open=False,
            elem_classes="advanced",
        ):
            with gr.Row():
                pitch = gr.Slider(
                    -2.0,
                    2.0,
                    value=0.0,
                    step=0.25,
                    label="Pitch shift (semitones)",
                    info="Changes the finished voice pitch without changing speaking speed.",
                )
                seed = gr.Number(
                    value=0,
                    precision=0,
                    label="Repeatable seed",
                    info="Use the same seed and settings to reproduce a take.",
                )
        with gr.Row(elem_classes="action-row"):
            generate = gr.Button("Generate selected model", variant="primary")
            compare = gr.Button("Compare Micro and Nano")
        audio = gr.Audio(label="Generated 24 kHz WAV", elem_classes="output-audio")
        status = gr.Markdown("Ready.", elem_classes="outputmeta")
        gr.Markdown(
            "The first request can include ZeroGPU queue and model-transfer time. "
            "Use the audio player's menu to download the final WAV.",
            elem_classes="helper-copy",
        )
        gr.Examples(
            [[prompt] for prompt in PROMPTS],
            inputs=text,
            label="Try a prepared prompt",
        )
    with gr.Column(elem_classes="comparison"):
        gr.Markdown(
            "### Direct model comparison\n"
            "The same text, seed, speed, variation, and pitch are sent to both models.",
            elem_classes="comparison-title",
        )
        with gr.Row():
            micro_audio = gr.Audio(label="Micro v2 · 9.36M · quality first")
            nano_audio = gr.Audio(label="Nano v2 · 3.96M · footprint first")
        compare_status = gr.Markdown("", elem_classes="outputmeta")
    gr.HTML("""<p class="fineprint">Fixed English voice · English-only frontend · no voice cloning · no reference audio upload. Uncommon names, abbreviations, and long passages can still fail. Generated audio is returned to your browser; this interface does not ask for an account or personal information.</p>""")
    generate.click(synthesize_one, [text, model, speed, variation, pitch, seed], [audio, status], concurrency_limit=1, api_name="synthesize")
    text.submit(synthesize_one, [text, model, speed, variation, pitch, seed], [audio, status], concurrency_limit=1)
    compare.click(synthesize_both, [text, speed, variation, pitch, seed], [micro_audio, nano_audio, compare_status], concurrency_limit=1, api_name="compare")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        theme=gr.themes.Base(),
        css=CSS,
    )
