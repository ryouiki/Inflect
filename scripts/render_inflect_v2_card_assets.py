from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


BLUE = "#1769E0"
BLUE_LIGHT = "#6EA8F2"
BLUE_DARK = "#0B2A54"
INK = "#10253F"
MUTED = "#61738A"
GRID = "#DCE6F1"
PAPER = "#F8FBFF"
COMPETITOR = "#8B9AA9"

SYSTEM_LABELS = {
    "inflect-micro-v2": "Inflect-Micro-v2",
    "inflect-nano-v2": "Inflect-Nano-v2",
    "kitten-nano-bruno": "KittenTTS Nano · Bruno",
    "kitten-nano-hugo": "KittenTTS Nano · Hugo",
    "piper-ryan-low": "Piper Low · Ryan",
    "piper-danny-low": "Piper Low · Danny",
    "supertonic3-m2-3step": "Supertonic 3 · M2 · 3-step",
    "supertonic3-m2-8step": "Supertonic 3 · M2 · 8-step",
}

ASR_FAMILIES = {
    "inflect-micro-v2": ("Inflect-Micro-v2", ["inflect-micro-v2"]),
    "inflect-nano-v2": ("Inflect-Nano-v2", ["inflect-nano-v2"]),
    "kitten": (
        "KittenTTS Nano · two-voice mean",
        ["kitten-nano-bruno", "kitten-nano-hugo"],
    ),
    "piper": (
        "Piper Low · two-voice mean",
        ["piper-ryan-low", "piper-danny-low"],
    ),
    "supertonic-8": ("Supertonic 3 · M2 · 8-step", ["supertonic3-m2-8step"]),
    "supertonic-3": ("Supertonic 3 · M2 · 3-step", ["supertonic3-m2-3step"]),
}

UTMOS_FAMILIES = {
    "inflect-micro-v2": ("Inflect-Micro-v2", ["inflect-micro-v2"], 37.53),
    "inflect-nano-v2": ("Inflect-Nano-v2", ["inflect-nano-v2"], 15.97),
    "kitten": (
        "KittenTTS Nano · 2-voice mean",
        ["kitten-nano-bruno", "kitten-nano-hugo"],
        56.77,
    ),
    "piper": (
        "Piper Low · 2-voice mean",
        ["piper-ryan-low", "piper-danny-low"],
        63.10,
    ),
    "supertonic-8": (
        "Supertonic 3 · 8-step",
        ["supertonic3-james-8step"],
        398.08,
    ),
}

FOOTPRINTS = (
    ("Inflect-Nano-v2", 15.97, True),
    ("Inflect-Micro-v2", 37.53, True),
    ("KittenTTS Nano", 56.77, False),
    ("Piper Low", 63.10, False),
    ("Supertonic 3", 398.08, False),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def shell(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        f'<rect width="{width}" height="{height}" rx="18" fill="{PAPER}"/>',
        (
            "<style>"
            "text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#10253F}"
            ".title{font-size:28px;font-weight:750;letter-spacing:-.5px}"
            ".subtitle{font-size:14px;fill:#61738A}"
            ".label{font-size:14px;font-weight:600}"
            ".small{font-size:12px;fill:#61738A}"
            ".value{font-size:14px;font-weight:750}"
            ".grid{stroke:#DCE6F1;stroke-width:1}"
            "</style>"
        ),
        f'<rect x="0" y="0" width="8" height="{height}" rx="4" fill="{BLUE}"/>',
        f'<text class="title" x="56" y="52">{html.escape(title)}</text>',
        f'<text class="subtitle" x="56" y="78">{html.escape(subtitle)}</text>',
    ]


def write_svg(path: Path, parts: list[str]) -> None:
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def human_preference(data: dict[str, Any], out: Path) -> None:
    rows = []
    for row in data["systems"]:
        total = row["wins"] + row["losses"] + row["ties"]
        score = (row["wins"] + 0.5 * row["ties"]) / total
        rows.append((row["label"], score, row["wins"], row["losses"], row["ties"]))
    rows.sort(key=lambda item: item[1], reverse=True)
    parts = shell(
        1200,
        760,
        "Community blind listening",
        "Anonymous pairwise preference · hidden system names · randomized left/right order",
    )
    left, right, top, row_h = 330, 1080, 126, 61
    for tick in (0, 25, 50, 75, 100):
        x = left + (right - left) * tick / 100
        parts.extend(
            [
                f'<line class="grid" x1="{x}" y1="108" x2="{x}" y2="682"/>',
                f'<text class="small" x="{x}" y="104" text-anchor="middle">{tick}%</text>',
            ]
        )
    for index, (label, score, wins, losses, ties) in enumerate(rows):
        y = top + index * row_h
        color = BLUE if label.startswith("Inflect-") else COMPETITOR
        parts.extend(
            [
                f'<text class="label" x="56" y="{y + 20}">{html.escape(label)}</text>',
                f'<rect x="{left}" y="{y + 4}" width="{right-left}" height="18" rx="9" fill="#E9F0F7"/>',
                f'<rect x="{left}" y="{y + 4}" width="{(right-left)*score:.1f}" height="18" rx="9" fill="{color}"/>',
                f'<text class="value" x="{right + 18}" y="{y + 19}">{score*100:.1f}%</text>',
                f'<text class="small" x="56" y="{y + 39}">{wins}W · {losses}L · {ties}T</text>',
            ]
        )
    parts.append(
        '<text class="small" x="56" y="730">Ties count as half a win. This is descriptive community evidence, not a formal MOS study.</text>'
    )
    write_svg(out, parts)


def quality_footprint(report: dict[str, Any], out: Path) -> None:
    rows = []
    for family, (label, members, size) in UTMOS_FAMILIES.items():
        scores = [float(report["systems"][member]["mean"]) for member in members]
        rows.append(
            {
                "family": family,
                "label": label,
                "size": size,
                "score": sum(scores) / len(scores),
                "low": min(scores),
                "high": max(scores),
            }
        )
    parts = shell(
        1500,
        730,
        "Predicted quality versus model footprint",
        "UTMOS22 on 500 matched unseen prompts · complete deployable weight footprint",
    )
    x0, x1, y0, y1 = 100, 850, 610, 125
    min_x, max_x = math.log10(12), math.log10(500)
    min_y, max_y = 4.12, 4.46
    for size in (16, 32, 64, 128, 256, 512):
        x = x0 + (math.log10(size) - min_x) / (max_x - min_x) * (x1 - x0)
        parts.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y0}"/>',
                f'<text class="small" x="{x:.1f}" y="{y0+28}" text-anchor="middle">{size}</text>',
            ]
        )
    for tick in range(5):
        value = min_y + (max_y - min_y) * tick / 4
        y = y0 - (value - min_y) / (max_y - min_y) * (y0 - y1)
        parts.extend(
            [
                f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>',
                f'<text class="small" x="{x0-14}" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>',
            ]
        )
    ordered = sorted(rows, key=lambda row: (-row["score"], row["size"]))
    for index, row in enumerate(ordered, start=1):
        x = x0 + (math.log10(row["size"]) - min_x) / (max_x - min_x) * (x1 - x0)
        y = y0 - (row["score"] - min_y) / (max_y - min_y) * (y0 - y1)
        low = y0 - (row["low"] - min_y) / (max_y - min_y) * (y0 - y1)
        high = y0 - (row["high"] - min_y) / (max_y - min_y) * (y0 - y1)
        color = BLUE if row["family"].startswith("inflect-") else COMPETITOR
        if row["high"] != row["low"]:
            parts.append(
                f'<line x1="{x:.1f}" y1="{low:.1f}" x2="{x:.1f}" y2="{high:.1f}" stroke="{color}" stroke-width="3"/>'
            )
        parts.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="{color}"/>',
                f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" font-size="11" font-weight="800" fill="#FFFFFF">{index}</text>',
            ]
        )
    panel_x = 905
    parts.extend(
        [
            f'<rect x="{panel_x}" y="116" width="540" height="420" rx="12" fill="#FFFFFF" stroke="{GRID}"/>',
            f'<text class="small" x="{panel_x+22}" y="146">MODEL FAMILY</text>',
            f'<text class="small" x="{panel_x+420}" y="146" text-anchor="end">UTMOS22</text>',
            f'<text class="small" x="{panel_x+510}" y="146" text-anchor="end">MB</text>',
        ]
    )
    for index, row in enumerate(ordered, start=1):
        y = 178 + (index - 1) * 66
        color = BLUE if row["family"].startswith("inflect-") else COMPETITOR
        parts.extend(
            [
                f'<circle cx="{panel_x+22}" cy="{y-4}" r="11" fill="{color}"/>',
                f'<text x="{panel_x+22}" y="{y}" text-anchor="middle" font-size="10" font-weight="800" fill="#FFFFFF">{index}</text>',
                f'<text class="label" x="{panel_x+44}" y="{y}">{html.escape(row["label"])}</text>',
                f'<text class="value" x="{panel_x+420}" y="{y}" text-anchor="end">{row["score"]:.3f}</text>',
                f'<text class="small" x="{panel_x+510}" y="{y}" text-anchor="end">{row["size"]:.1f}</text>',
            ]
        )
    parts.extend(
        [
            f'<text class="small" x="{(x0+x1)/2}" y="675" text-anchor="middle">Model weight footprint (MB, logarithmic scale)</text>',
            '<text class="small" x="26" y="370" transform="rotate(-90 26 370)" text-anchor="middle">Predicted MOS · higher is better</text>',
            '<text class="small" x="56" y="710">UTMOS22 is an automated predictor, not human MOS. Family means use equal weighting across the named voices.</text>',
            '<text class="small" x="905" y="565">Supertonic 3-step: 2.471 UTMOS22 · below plotted range.</text>',
            '<text class="small" x="905" y="585">Supertonic UTMOS uses the James voice; family means use equal voice weighting.</text>',
        ]
    )
    write_svg(out, parts)


def two_asr_consensus(summary: dict[str, Any], out: Path) -> None:
    rows = []
    for family, (label, members) in ASR_FAMILIES.items():
        member_values = []
        for member in members:
            payload = summary["systems"][member]
            qwen = float(payload["qwen3_asr"]["semantic"]["wer"])
            nemo = float(payload["nemotron35_asr"]["semantic"]["wer"])
            member_values.append((qwen + nemo) / 2)
        rows.append((family, label, sum(member_values) / len(member_values)))
    rows.sort(key=lambda row: row[2])
    maximum = 0.05
    parts = shell(
        1200,
        590,
        "Intelligibility on unseen text",
        "Two-ASR consensus semantic WER · Qwen3-ASR + Nemotron 3.5 · 400 matched prompts",
    )
    left, right, top, row_h = 360, 1060, 132, 68
    for tick in range(6):
        value = maximum * tick / 5
        x = left + (right - left) * tick / 5
        parts.extend(
            [
                f'<line class="grid" x1="{x}" y1="108" x2="{x}" y2="515"/>',
                f'<text class="small" x="{x}" y="104" text-anchor="middle">{value*100:.0f}%</text>',
            ]
        )
    for index, (family, label, value) in enumerate(rows):
        y = top + index * row_h
        color = BLUE if family.startswith("inflect-") else COMPETITOR
        x = left + (right - left) * value / maximum
        parts.extend(
            [
                f'<text class="label" x="56" y="{y+6}">{html.escape(label)}</text>',
                f'<line x1="{left}" y1="{y}" x2="{x:.1f}" y2="{y}" stroke="{color}" stroke-width="4"/>',
                f'<circle cx="{x:.1f}" cy="{y}" r="8" fill="{color}"/>',
                f'<text class="value" x="{right+16}" y="{y+6}">{value*100:.2f}%</text>',
            ]
        )
    parts.append(
        '<text class="small" x="56" y="560">Lower is better. Whisper is excluded from this headline for every system; the complete three-ASR audit is reported separately.</text>'
    )
    write_svg(out, parts)


def marker(shape: str, x: float, y: float, color: str) -> str:
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>'
    if shape == "square":
        return f'<rect x="{x-6:.1f}" y="{y-6:.1f}" width="12" height="12" rx="2" fill="{color}"/>'
    return (
        f'<path d="M {x:.1f} {y-7:.1f} L {x+7:.1f} {y:.1f} '
        f'L {x:.1f} {y+7:.1f} L {x-7:.1f} {y:.1f} Z" fill="{color}"/>'
    )


def three_asr_detail(summary: dict[str, Any], out: Path) -> None:
    specs = (
        ("qwen3_asr", "Qwen3-ASR", BLUE, "circle"),
        ("nemotron35_asr", "Nemotron 3.5", BLUE_LIGHT, "square"),
        ("whisper_large_v3", "Whisper large-v3", BLUE_DARK, "diamond"),
    )
    order = list(SYSTEM_LABELS)
    parts = shell(
        1400,
        900,
        "Three-ASR intelligibility audit",
        "Semantic WER on the same 400 generated clips per system · lower is better",
    )
    left, right, top, row_h = 330, 965, 205, 76
    legend_x = left
    for _, label, color, shape in specs:
        parts.append(marker(shape, legend_x, 112, color))
        parts.append(f'<text class="small" x="{legend_x+16}" y="116">{label}</text>')
        legend_x += 210
    for tick in (0, 2, 4, 6, 8, 10):
        x = left + (right - left) * tick / 10
        parts.extend(
            [
                f'<line class="grid" x1="{x}" y1="160" x2="{x}" y2="805"/>',
                f'<text class="small" x="{x}" y="152" text-anchor="middle">{tick}%</text>',
            ]
        )
    parts.extend(
        [
            '<text class="small" x="1040" y="152">QWEN</text>',
            '<text class="small" x="1125" y="152">NEMO</text>',
            '<text class="small" x="1210" y="152">WHISPER</text>',
        ]
    )
    offsets = (-17, 0, 17)
    for index, system in enumerate(order):
        y = top + index * row_h
        payload = summary["systems"][system]
        parts.append(
            f'<text class="label" x="56" y="{y+5}">{html.escape(SYSTEM_LABELS[system])}</text>'
        )
        values = []
        for (key, _, color, shape), offset in zip(specs, offsets, strict=True):
            value = float(payload[key]["semantic"]["wer"])
            low, high = [float(v) for v in payload[key]["semantic_95ci"]]
            x = left + (right - left) * min(value, 0.10) / 0.10
            x_low = left + (right - left) * min(low, 0.10) / 0.10
            x_high = left + (right - left) * min(high, 0.10) / 0.10
            py = y + offset
            parts.extend(
                [
                    f'<line x1="{x_low:.1f}" y1="{py}" x2="{x_high:.1f}" y2="{py}" stroke="{color}" stroke-width="2"/>',
                    marker(shape, x, py, color),
                ]
            )
            values.append(value)
        for x, value in zip((1040, 1125, 1210), values, strict=True):
            parts.append(f'<text class="value" x="{x}" y="{y+5}">{value*100:.2f}%</text>')
    parts.extend(
        [
            '<rect x="1020" y="796" width="330" height="64" rx="10" fill="#EAF2FF"/>',
            '<text class="small" x="1038" y="821">Supertonic 8-step / Whisper is an insertion-heavy</text>',
            '<text class="small" x="1038" y="841">ASR outlier; Qwen, Nemotron, and listening disagree.</text>',
            '<text class="small" x="56" y="870">95% matched-prompt bootstrap intervals. Formatting-equivalent number forms are normalized before scoring.</text>',
        ]
    )
    write_svg(out, parts)


def footprint(out: Path) -> None:
    parts = shell(
        1200,
        500,
        "Complete model weight footprint",
        "One entry per deployable model family · lower is smaller",
    )
    left, right, top, row_h, maximum = 300, 1080, 122, 64, 420.0
    for tick in (0, 100, 200, 300, 400):
        x = left + (right - left) * tick / maximum
        parts.extend(
            [
                f'<line class="grid" x1="{x}" y1="104" x2="{x}" y2="410"/>',
                f'<text class="small" x="{x}" y="100" text-anchor="middle">{tick} MB</text>',
            ]
        )
    for index, (label, size, is_inflect) in enumerate(FOOTPRINTS):
        y = top + index * row_h
        color = BLUE if is_inflect else COMPETITOR
        width = (right - left) * size / maximum
        parts.extend(
            [
                f'<text class="label" x="56" y="{y+20}">{label}</text>',
                f'<rect x="{left}" y="{y+4}" width="{width:.1f}" height="22" rx="5" fill="{color}"/>',
                f'<text class="value" x="{left+width+12:.1f}" y="{y+21}">{size:.1f} MB</text>',
            ]
        )
    parts.append(
        '<text class="small" x="56" y="470">Voice variants sharing identical weights are merged. Inflect totals include the neural waveform decoder.</text>'
    )
    write_svg(out, parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modern-summary", type=Path, required=True)
    parser.add_argument("--utmos", type=Path, required=True)
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    modern = read_json(args.modern_summary)
    utmos = read_json(args.utmos)
    human = read_json(args.human)
    args.out.mkdir(parents=True, exist_ok=True)

    human_preference(human, args.out / "human-preference.svg")
    quality_footprint(utmos, args.out / "quality-vs-footprint.svg")
    two_asr_consensus(modern, args.out / "asr-consensus.svg")
    three_asr_detail(modern, args.out / "modern400-three-asr.svg")
    footprint(args.out / "model-footprint.svg")


if __name__ == "__main__":
    main()
