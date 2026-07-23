from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


LABELS = {
    "inflect-micro-v2": "Inflect-Micro-v2",
    "inflect-nano-v2": "Inflect-Nano-v2",
    "kitten-nano-bruno": "KittenTTS Nano · Bruno",
    "kitten-nano-hugo": "KittenTTS Nano · Hugo",
    "piper-ryan-low": "Piper · Ryan Low",
    "piper-danny-low": "Piper · Danny Low",
    "supertonic3-james-3step": "Supertonic 3 · James · 3-step",
    "supertonic3-james-8step": "Supertonic 3 · James · 8-step",
}
PLOT_LABELS = {
    "inflect-micro-v2": "Inflect-Micro-v2",
    "inflect-nano-v2": "Inflect-Nano-v2",
    "kitten-nano-bruno": "Kitten · Bruno",
    "kitten-nano-hugo": "Kitten · Hugo",
    "piper-ryan-low": "Piper · Ryan",
    "piper-danny-low": "Piper · Danny",
    "supertonic3-james-3step": "Supertonic · 3-step",
    "supertonic3-james-8step": "Supertonic · 8-step",
}
ORDER = list(LABELS)
SIZES_MB = {
    "inflect-micro-v2": 37.529995,
    "inflect-nano-v2": 15.971083,
    "kitten-nano-bruno": 56.767095,
    "kitten-nano-hugo": 56.767095,
    "piper-ryan-low": 63.104526,
    "piper-danny-low": 63.104526,
    "supertonic3-james-3step": 398.075273,
    "supertonic3-james-8step": 398.075273,
}
BLUE = "#1769E0"
NAVY = "#10253F"
MUTED = "#5E7085"
GRID = "#DCE6F1"
PAPER = "#F8FBFF"
COMPETITOR = "#8B9AA9"
FAMILY_SPECS = {
    "inflect-micro-v2": ("Inflect-Micro-v2", ["inflect-micro-v2"]),
    "inflect-nano-v2": ("Inflect-Nano-v2", ["inflect-nano-v2"]),
    "kitten-nano": ("KittenTTS Nano · 2-voice mean", ["kitten-nano-bruno", "kitten-nano-hugo"]),
    "piper-low": ("Piper Low · 2-voice mean", ["piper-ryan-low", "piper-danny-low"]),
    "supertonic3-3step": ("Supertonic 3 · 3-step", ["supertonic3-james-3step"]),
    "supertonic3-8step": ("Supertonic 3 · 8-step", ["supertonic3-james-8step"]),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_release_evidence(
    asr_report: dict[str, Any],
    utmos_report: dict[str, Any],
    runtime_dir: Path,
    expected_clips: int,
    expected_runtime_prompts: int,
    minimum_bootstrap_samples: int,
) -> None:
    expected = set(ORDER)
    for label, report in (("ASR", asr_report), ("UTMOS22", utmos_report)):
        systems = set(report.get("systems", {}))
        if systems != expected:
            raise RuntimeError(
                f"{label} system mismatch: missing={sorted(expected-systems)} "
                f"extra={sorted(systems-expected)}"
            )

    asr_names = sorted({str(row.get("asr")) for row in asr_report.get("rows", []) if row.get("asr")})
    if "whisper-large-v3" not in asr_names:
        raise RuntimeError("ASR evidence is missing the required Whisper-large-v3 results")
    asr_counts = {(system, asr): 0 for system in ORDER for asr in asr_names}
    for row in asr_report.get("rows", []):
        system = row.get("variant") or row.get("system") or row.get("model")
        identity = (system, str(row.get("asr")))
        if identity in asr_counts:
            asr_counts[identity] += 1
    bad_asr = {
        f"{system}/{asr}": count
        for (system, asr), count in asr_counts.items()
        if count != expected_clips
    }
    if bad_asr:
        raise RuntimeError(f"ASR row counts are not {expected_clips} per system/model: {bad_asr}")

    if int(utmos_report.get("bootstrap_samples", 0)) < minimum_bootstrap_samples:
        raise RuntimeError(
            f"UTMOS22 report must use at least {minimum_bootstrap_samples:,} bootstrap samples"
        )
    bad_utmos = {
        system: int(utmos_report["systems"][system].get("count", -1))
        for system in ORDER
        if int(utmos_report["systems"][system].get("count", -1)) != expected_clips
    }
    if bad_utmos:
        raise RuntimeError(f"UTMOS22 row counts are not {expected_clips} per system: {bad_utmos}")

    bad_runtime: dict[str, str] = {}
    for system in ORDER:
        path = runtime_dir / f"{system}_cpu.json"
        if not path.is_file():
            bad_runtime[system] = "missing"
            continue
        report = read_json(path)
        utterances = int(report.get("utterances", -1))
        rows = len(report.get("rows", []))
        if report.get("system") != system or utterances != expected_runtime_prompts or rows != expected_runtime_prompts:
            bad_runtime[system] = f"system={report.get('system')} utterances={utterances} rows={rows}"
    if bad_runtime:
        raise RuntimeError(f"CPU runtime reports failed the matched protocol: {bad_runtime}")


def shell(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#10253F}.title{font-size:28px;font-weight:700;letter-spacing:-.3px}.subtitle{font-size:14px;fill:#5E7085}.label{font-size:14px}.small{font-size:12px;fill:#5E7085}.value{font-size:14px;font-weight:700}.axis{stroke:#AFC0D2;stroke-width:1}.grid{stroke:#DCE6F1;stroke-width:1}.ci{stroke-width:2}</style>',
        f'<text class="title" x="56" y="52">{html.escape(title)}</text>',
        f'<text class="subtitle" x="56" y="78">{html.escape(subtitle)}</text>',
    ]


def write_svg(path: Path, parts: list[str]) -> None:
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def point_label(
    parts: list[str], *, x: float, y: float, dx: float, dy: float, label: str
) -> None:
    """Draw a direct label with a quiet leader line when it is moved off-point."""
    anchor = "start" if dx >= 0 else "end"
    label_x, label_y = x + dx, y + dy
    line_x = label_x - 5 if anchor == "start" else label_x + 5
    parts.extend(
        [
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{line_x:.1f}" y2="{label_y-4:.1f}" '
            f'stroke="{MUTED}" stroke-width="1"/>',
            f'<text class="label" x="{label_x:.1f}" y="{label_y:.1f}" '
            f'text-anchor="{anchor}">{html.escape(label)}</text>',
        ]
    )


def asr_values(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = {}
    for system, payload in report["systems"].items():
        result = payload["asr"]["whisper-large-v3"]
        values[system] = {"wer": float(result["semantic"]["wer"]), "ci": [float(v) for v in result["semantic_95ci"]], "categories": result["categories"]}
    return values


def utmos_values(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {system: {"mean": float(payload["mean"]), "ci": [float(v) for v in payload["mean_bootstrap_95_ci"]]} for system, payload in report["systems"].items()}


def family_rows(values: dict[str, dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    rows = []
    for family, (label, members) in FAMILY_SPECS.items():
        samples = [float(values[member][metric]) for member in members]
        rows.append(
            {
                "family": family,
                "label": label,
                "members": members,
                "value": sum(samples) / len(samples),
                "low": min(samples),
                "high": max(samples),
                "inflect": family.startswith("inflect-"),
            }
        )
    return rows


def human_figure(data: dict[str, Any], out: Path) -> None:
    rows = []
    for row in data["systems"]:
        total = row["wins"] + row["losses"] + row["ties"]
        score = (row["wins"] + 0.5 * row["ties"]) / total
        rows.append((row["label"], score, row["wins"], row["losses"], row["ties"]))
    rows.sort(key=lambda item: item[1], reverse=True)
    parts = shell(1200, 760, "Community blind listening", f'{data["total_pairwise_votes"]} anonymous pairwise decisions · ties count as half a win')
    left, right, top, row_h = 330, 1100, 126, 61
    for tick in (0, 25, 50, 75, 100):
        x = left + (right - left) * tick / 100
        parts.extend([f'<line class="grid" x1="{x:.1f}" y1="108" x2="{x:.1f}" y2="{top + row_h * len(rows) - 12}"/>', f'<text class="small" x="{x:.1f}" y="104" text-anchor="middle">{tick}%</text>'])
    for index, (label, score, wins, losses, ties) in enumerate(rows):
        y = top + index * row_h
        color = BLUE if label.startswith("Inflect-") else COMPETITOR
        parts.extend([
            f'<text class="label" x="56" y="{y + 20}">{html.escape(label)}</text>',
            f'<rect x="{left}" y="{y + 4}" width="{right-left}" height="18" fill="#E9F0F7"/>',
            f'<rect x="{left}" y="{y + 4}" width="{(right-left)*score:.1f}" height="18" fill="{color}"/>',
            f'<text class="value" x="{right + 18}" y="{y + 19}">{score*100:.1f}%</text>',
            f'<text class="small" x="56" y="{y + 39}">{wins}W · {losses}L · {ties}T</text>',
        ])
    parts.append('<text class="small" x="56" y="730">Descriptive community result, not a formal MOS study. Systems were hidden and left/right order was randomized.</text>')
    write_svg(out, parts)


def quality_size_figure(values: dict[str, dict[str, Any]], out: Path) -> None:
    rows = family_rows(values, "mean")
    plotted = [row for row in rows if row["family"] != "supertonic3-3step"]
    width, height = 1200, 720
    parts = shell(width, height, "Predicted quality versus model footprint", "UTMOS22 · 500 identical unseen prompts per voice · family means show the observed voice range")
    x0, x1, y0, y1 = 105, 1110, 610, 120
    min_x, max_x = math.log10(12), math.log10(500)
    ys = [row["value"] for row in plotted]
    min_y, max_y = min(ys) - 0.08, max(ys) + 0.08
    for size in (16, 32, 64, 128, 256, 512):
        x = x0 + (math.log10(size) - min_x) / (max_x - min_x) * (x1 - x0)
        parts.extend([f'<line class="grid" x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y0}"/>', f'<text class="small" x="{x:.1f}" y="{y0+28}" text-anchor="middle">{size}</text>'])
    for index in range(5):
        value = min_y + (max_y - min_y) * index / 4
        y = y0 - (value - min_y) / (max_y - min_y) * (y0 - y1)
        parts.extend([f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>', f'<text class="small" x="{x0-14}" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>'])
    offsets = {
        "inflect-nano-v2": (12, 30),
        "inflect-micro-v2": (12, -16),
        "kitten-nano": (14, -16),
        "piper-low": (14, 34),
        "supertonic3-8step": (-14, -16),
    }
    for row in plotted:
        value = row["value"]
        size = sum(SIZES_MB[member] for member in row["members"]) / len(row["members"])
        x = x0 + (math.log10(size) - min_x) / (max_x - min_x) * (x1 - x0)
        y = y0 - (value - min_y) / (max_y - min_y) * (y0 - y1)
        low = y0 - (row["low"] - min_y) / (max_y - min_y) * (y0 - y1)
        high = y0 - (row["high"] - min_y) / (max_y - min_y) * (y0 - y1)
        color = BLUE if row["inflect"] else COMPETITOR
        radius = 9 if row["inflect"] else 7
        dx, dy = offsets.get(row["family"], (10, -12))
        if len(row["members"]) > 1:
            parts.append(f'<line class="ci" stroke="{color}" x1="{x:.1f}" y1="{low:.1f}" x2="{x:.1f}" y2="{high:.1f}"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"/>')
        point_label(parts, x=x, y=y, dx=dx, dy=dy, label=row["label"])
    parts.extend([f'<text class="small" x="{(x0+x1)/2}" y="675" text-anchor="middle">Model weight footprint (MB, logarithmic scale)</text>', '<text class="small" x="28" y="365" transform="rotate(-90 28 365)" text-anchor="middle">Predicted MOS · higher is better</text>', '<text class="small" x="56" y="704">UTMOS22 is an automated predictor, not human MOS. Supertonic uses four ONNX weight files; Inflect includes its waveform decoder.</text>'])
    parts.append('<text class="small" x="1110" y="104" text-anchor="end">Supertonic 3-step: 2.471 UTMOS22 · below plotted range</text>')
    write_svg(out, parts)


def wer_figure(values: dict[str, dict[str, Any]], out: Path) -> None:
    systems = sorted((system for system in ORDER if system in values), key=lambda system: values[system]["wer"])
    # A failed synthesis can legitimately push corpus WER far above 100% through
    # insertions. Keep the useful comparison readable and mark those runs as
    # off-scale rather than silently clipping or flattening every other result.
    in_scale = [system for system in systems if values[system]["wer"] <= 1.0]
    max_value = max(values[system]["ci"][1] for system in in_scale) * 1.12
    parts = shell(1200, 720, "Intelligibility on unseen text", "Whisper-large-v3 semantic WER · 500 identical prompts per system · 95% bootstrap intervals")
    # Reserve a proper right-hand value column so long off-scale annotations
    # never get clipped by Hugging Face's SVG viewport.
    left, right, top, row_h = 330, 1005, 125, 63
    for tick in range(6):
        value = max_value * tick / 5
        x = left + (right-left) * tick / 5
        parts.extend([f'<line class="grid" x1="{x:.1f}" y1="108" x2="{x:.1f}" y2="{top+row_h*len(systems)-18}"/>', f'<text class="small" x="{x:.1f}" y="104" text-anchor="middle">{value*100:.1f}%</text>'])
    for index, system in enumerate(systems):
        y = top + index * row_h
        value, ci = values[system]["wer"], values[system]["ci"]
        color = BLUE if system.startswith("inflect-") else COMPETITOR
        parts.append(f'<text class="label" x="56" y="{y+11}">{html.escape(LABELS[system])}</text>')
        if value > 1.0:
            parts.extend([
                f'<line x1="{left:.1f}" y1="{y+7}" x2="{right-12:.1f}" y2="{y+7}" stroke="{color}" stroke-width="3" stroke-dasharray="7 7"/>',
                f'<path d="M {right-18:.1f} {y+1:.1f} L {right-4:.1f} {y+7:.1f} L {right-18:.1f} {y+13:.1f}" fill="none" stroke="{color}" stroke-width="3"/>',
                f'<text class="value" x="{right+16}" y="{y+12}">{value*100:.2f}% · off scale</text>',
            ])
            continue
        x = left + (right-left) * value / max_value
        low = left + (right-left) * ci[0] / max_value
        high = left + (right-left) * ci[1] / max_value
        parts.extend([f'<line x1="{low:.1f}" y1="{y+7}" x2="{high:.1f}" y2="{y+7}" stroke="{color}" stroke-width="3"/>', f'<line x1="{low:.1f}" y1="{y+1}" x2="{low:.1f}" y2="{y+13}" stroke="{color}"/>', f'<line x1="{high:.1f}" y1="{y+1}" x2="{high:.1f}" y2="{y+13}" stroke="{color}"/>', f'<circle cx="{x:.1f}" cy="{y+7}" r="7" fill="{color}"/>', f'<text class="value" x="{right+16}" y="{y+12}">{value*100:.2f}%</text>'])
    parts.append('<text class="small" x="56" y="690">Lower is better. WER may exceed 100% when repeated or mismatched speech creates many insertion errors; off-scale values remain reported exactly.</text>')
    write_svg(out, parts)


def asr_robustness_figure(report: dict[str, Any], out: Path) -> None:
    asr_names = sorted(
        set.intersection(
            *(set(report["systems"][system]["asr"]) for system in ORDER)
        )
    )
    if len(asr_names) < 2:
        return
    preferred = [name for name in ("whisper-large-v3", "wav2vec2-large-lv60k-960h") if name in asr_names]
    selected = preferred if len(preferred) == 2 else asr_names[:2]
    values = {
        (system, asr): float(report["systems"][system]["asr"][asr]["semantic"]["wer"])
        for system in ORDER
        for asr in selected
    }
    in_scale = [value for value in values.values() if value <= 1.0]
    maximum = max(in_scale) * 1.12 or 0.01
    parts = shell(
        1200,
        730,
        "Intelligibility across two ASR families",
        "Semantic WER on the same 500 generated clips per system · lower is better",
    )
    left, right, top, row_h = 330, 950, 145, 63
    legend = {
        selected[0]: (BLUE, "circle", "Whisper-large-v3" if selected[0] == "whisper-large-v3" else selected[0]),
        selected[1]: ("#E28B38", "square", "wav2vec2 LV-60K" if selected[1].startswith("wav2vec2") else selected[1]),
    }
    lx = left
    for asr in selected:
        color, shape, label = legend[asr]
        if shape == "circle":
            parts.append(f'<circle cx="{lx}" cy="104" r="6" fill="{color}"/>')
        else:
            parts.append(f'<rect x="{lx-6}" y="98" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text class="small" x="{lx+14}" y="108">{html.escape(label)}</text>')
        lx += 220
    for tick in range(6):
        value = maximum * tick / 5
        x = left + (right-left) * tick / 5
        parts.extend([
            f'<line class="grid" x1="{x:.1f}" y1="120" x2="{x:.1f}" y2="{top+row_h*len(ORDER)-18}"/>',
            f'<text class="small" x="{x:.1f}" y="136" text-anchor="middle">{value*100:.1f}%</text>',
        ])
    for index, system in enumerate(ORDER):
        y = top + index * row_h
        parts.append(f'<text class="label" x="56" y="{y+12}">{html.escape(LABELS[system])}</text>')
        value_labels = []
        for offset, asr in zip((-7, 9), selected, strict=True):
            value = values[(system, asr)]
            color, shape, short = legend[asr]
            py = y + offset
            value_labels.append(f'{short.split("-")[0]} {value*100:.2f}%')
            if value > 1.0:
                parts.extend([
                    f'<line x1="{left}" y1="{py}" x2="{right-12}" y2="{py}" stroke="{color}" stroke-width="2" stroke-dasharray="6 6"/>',
                    f'<path d="M {right-18} {py-5} L {right-5} {py} L {right-18} {py+5}" fill="none" stroke="{color}" stroke-width="2"/>',
                ])
            else:
                x = left + (right-left) * value / maximum
                if shape == "circle":
                    parts.append(f'<circle cx="{x:.1f}" cy="{py}" r="6" fill="{color}"/>')
                else:
                    parts.append(f'<rect x="{x-6:.1f}" y="{py-6}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text class="small" x="{right+18}" y="{y+1}">{html.escape(value_labels[0])}</text>')
        parts.append(f'<text class="small" x="{right+18}" y="{y+18}">{html.escape(value_labels[1])}</text>')
    parts.append('<text class="small" x="56" y="704">Large disagreement identifies ASR sensitivity rather than a hidden aggregate score; all hypotheses and per-clip errors ship in the raw reports.</text>')
    write_svg(out, parts)


def footprint_figure(out: Path) -> None:
    rows = []
    for family, (label, members) in FAMILY_SPECS.items():
        rows.append((family, label, sum(SIZES_MB[member] for member in members) / len(members)))
    rows.sort(key=lambda row: row[2])
    parts = shell(1200, 610, "Complete model weight footprint", "One entry per deployable model family · waveform decoder included where applicable")
    left, right, top, row_h, maximum = 330, 1100, 120, 64, 420.0
    for tick in (0, 100, 200, 300, 400):
        x = left + (right-left) * tick / maximum
        parts.extend([f'<line class="grid" x1="{x:.1f}" y1="102" x2="{x:.1f}" y2="{top+row_h*len(rows)-18}"/>', f'<text class="small" x="{x:.1f}" y="98" text-anchor="middle">{tick} MB</text>'])
    for index, (family, label, size) in enumerate(rows):
        y = top + index * row_h
        color = BLUE if family.startswith("inflect-") else COMPETITOR
        length = (right-left) * size / maximum
        parts.extend([f'<text class="label" x="56" y="{y+18}">{html.escape(label)}</text>', f'<rect x="{left}" y="{y+3}" width="{length:.1f}" height="22" fill="{color}"/>', f'<text class="value" x="{left+length+12:.1f}" y="{y+20}">{size:.1f} MB</text>'])
    parts.append('<text class="small" x="56" y="580">Voices that share identical model weights are merged. Supertonic step counts share the same 398.1 MB deployment footprint.</text>')
    write_svg(out, parts)


def runtime_figure(runtime_dir: Path, out: Path) -> None:
    rows = []
    for family, (label, members) in FAMILY_SPECS.items():
        reports = [read_json(runtime_dir / f"{member}_cpu.json") for member in members]
        multiple = sum(float(report["realtime_multiple"]) for report in reports) / len(reports)
        p95 = sum(float(report["latency_p95_seconds"]) for report in reports) / len(reports)
        rows.append((family, label, multiple, p95))
    if not rows:
        return
    rows.sort(key=lambda row: row[2], reverse=True)
    maximum = max(row[2] for row in rows) * 1.12
    parts = shell(
        1200,
        610,
        "Warm CPU synthesis throughput",
        "AMD Ryzen 9 3900X · 12 threads · 48 matched prompts · higher is faster",
    )
    left, right, top, row_h = 330, 1100, 120, 64
    for tick in range(6):
        value = maximum * tick / 5
        x = left + (right-left) * tick / 5
        parts.extend([f'<line class="grid" x1="{x:.1f}" y1="102" x2="{x:.1f}" y2="{top+row_h*len(rows)-18}"/>', f'<text class="small" x="{x:.1f}" y="98" text-anchor="middle">{value:.1f}×</text>'])
    for index, (family, label, multiple, p95) in enumerate(rows):
        y = top + index * row_h
        length = (right-left) * multiple / maximum
        color = BLUE if family.startswith("inflect-") else COMPETITOR
        parts.extend([f'<text class="label" x="56" y="{y+17}">{html.escape(label)}</text>', f'<rect x="{left}" y="{y+2}" width="{length:.1f}" height="22" fill="{color}"/>', f'<text class="value" x="{left+length+12:.1f}" y="{y+19}">{multiple:.2f}× real time</text>', f'<text class="small" x="56" y="{y+37}">mean p95 {p95:.3f} s</text>'])
    parts.append('<text class="small" x="56" y="580">1.0× is real time. Runtime includes text processing and waveform generation; two-voice families are equal-weight means.</text>')
    write_svg(out, parts)


def quality_speed_figure(values: dict[str, dict[str, Any]], runtime_dir: Path, out: Path) -> None:
    rows = []
    for family, (label, members) in FAMILY_SPECS.items():
        if family == "supertonic3-3step":
            continue
        runtimes = [read_json(runtime_dir / f"{member}_cpu.json") for member in members]
        speed = sum(float(report["realtime_multiple"]) for report in runtimes) / len(runtimes)
        quality = sum(float(values[member]["mean"]) for member in members) / len(members)
        rows.append((family, label, speed, quality))
    if not rows:
        return
    width, height = 1200, 760
    parts = shell(
        width,
        height,
        "Predicted quality and CPU throughput",
        "UTMOS22 versus warm end-to-end speed · Ryzen 9 3900X · 12 threads · 48 matched prompts",
    )
    x0, x1, y0, y1 = 105, 1110, 610, 120
    max_x = max(row[2] for row in rows) * 1.12
    ys = [row[3] for row in rows]
    min_y, max_y = min(ys) - 0.08, max(ys) + 0.08
    for tick in range(6):
        value = max_x * tick / 5
        x = x0 + (x1 - x0) * tick / 5
        parts.extend([f'<line class="grid" x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y0}"/>', f'<text class="small" x="{x:.1f}" y="{y0+28}" text-anchor="middle">{value:.1f}×</text>'])
    for tick in range(5):
        value = min_y + (max_y - min_y) * tick / 4
        y = y0 - (value - min_y) / (max_y - min_y) * (y0 - y1)
        parts.extend([f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>', f'<text class="small" x="{x0-14}" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>'])
    offsets = {
        "inflect-micro-v2": (12, -16),
        "inflect-nano-v2": (12, 30),
        "kitten-nano": (12, -42),
        "piper-low": (-12, 30),
        "supertonic3-8step": (-12, -42),
    }
    for family, label, speed, quality in rows:
        x = x0 + (x1 - x0) * speed / max_x
        y = y0 - (quality - min_y) / (max_y - min_y) * (y0 - y1)
        color = BLUE if family.startswith("inflect-") else COMPETITOR
        radius = 9 if family.startswith("inflect-") else 7
        dx, dy = offsets.get(family, (10, -12))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"/>')
        point_label(parts, x=x, y=y, dx=dx, dy=dy, label=label)
    parts.extend([
        f'<text class="small" x="{(x0+x1)/2}" y="685" text-anchor="middle">Audio seconds generated per wall-clock second · higher is faster</text>',
        '<text class="small" x="28" y="365" transform="rotate(-90 28 365)" text-anchor="middle">Predicted MOS · higher is better</text>',
        '<text class="small" x="56" y="738">A point farther right and higher is preferable. One host and isolated process per system; UTMOS22 is not human MOS.</text>',
        '<text class="small" x="1110" y="104" text-anchor="end">Supertonic 3-step: 2.471 UTMOS22 · below plotted range</text>',
    ])
    write_svg(out, parts)


def category_wer_figure(values: dict[str, dict[str, Any]], out: Path) -> None:
    systems = [system for system in ("inflect-micro-v2", "inflect-nano-v2") if system in values]
    if len(systems) != 2:
        return
    categories = sorted(set(values[systems[0]]["categories"]) & set(values[systems[1]]["categories"]))
    if not categories:
        return
    def category_wer(system: str, category: str) -> float:
        row = values[system]["categories"][category]
        if isinstance(row, dict):
            for key in ("semantic", "semantic_wer", "wer"):
                if key in row:
                    value = row[key]
                    return float(value["wer"] if isinstance(value, dict) else value)
        return float(row)
    rows = [(category, *(category_wer(system, category) for system in systems)) for category in categories]
    rows.sort(key=lambda row: max(row[1:]), reverse=True)
    maximum = max(max(row[1:]) for row in rows) * 1.12 or 0.01
    height = max(520, 160 + 62 * len(rows))
    parts = shell(1200, height, "Where transcription errors remain", "Semantic WER by prompt category · identical held-out text · Whisper-large-v3")
    left, right, top, row_h = 310, 1095, 125, 62
    for tick in range(6):
        value = maximum * tick / 5
        x = left + (right-left) * tick / 5
        parts.extend([f'<line class="grid" x1="{x:.1f}" y1="108" x2="{x:.1f}" y2="{top+row_h*len(rows)-15}"/>', f'<text class="small" x="{x:.1f}" y="104" text-anchor="middle">{100*value:.1f}%</text>'])
    for index, (category, micro, nano) in enumerate(rows):
        y = top + index * row_h
        parts.extend([
            f'<text class="label" x="56" y="{y+20}">{html.escape(category.replace("_", " ").title())}</text>',
            f'<rect x="{left}" y="{y+2}" width="{(right-left)*micro/maximum:.1f}" height="14" fill="{BLUE}"/>',
            f'<rect x="{left}" y="{y+22}" width="{(right-left)*nano/maximum:.1f}" height="14" fill="#78AEEF"/>',
            f'<text class="small" x="{right+12}" y="{y+14}">Micro {100*micro:.2f}%</text>',
            f'<text class="small" x="{right+12}" y="{y+34}">Nano {100*nano:.2f}%</text>',
        ])
    parts.append(f'<text class="small" x="56" y="{height-28}">Category results diagnose weaknesses; they are not pooled with UTMOS22 or human preference.</text>')
    write_svg(out, parts)


def family_average(systems: dict[str, dict[str, Any]], names: list[str], metric: str) -> float:
    return sum(float(systems[name][metric]) for name in names) / len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render publication-grade Inflect v2 benchmark figures.")
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--utmos", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-clips", type=int, default=500)
    parser.add_argument("--expected-runtime-prompts", type=int, default=50)
    parser.add_argument("--minimum-bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    asr_report = read_json(args.asr)
    utmos_report = read_json(args.utmos)
    validate_release_evidence(
        asr_report,
        utmos_report,
        args.runtime_dir,
        args.expected_clips,
        args.expected_runtime_prompts,
        args.minimum_bootstrap_samples,
    )
    asr = asr_values(asr_report)
    utmos = utmos_values(utmos_report)
    args.out.mkdir(parents=True, exist_ok=True)
    human_figure(read_json(args.human), args.out / "human-preference.svg")
    quality_size_figure(utmos, args.out / "quality-vs-footprint.svg")
    wer_figure(asr, args.out / "semantic-wer.svg")
    asr_robustness_figure(asr_report, args.out / "asr-robustness.svg")
    footprint_figure(args.out / "model-footprint.svg")
    runtime_figure(args.runtime_dir, args.out / "cpu-throughput.svg")
    quality_speed_figure(utmos, args.runtime_dir, args.out / "quality-vs-cpu-speed.svg")
    category_wer_figure(asr, args.out / "category-semantic-wer.svg")

    summary = {
        "format": "inflect_v2_release_evidence_v1",
        "systems": {system: {"label": LABELS[system], "weight_mb": SIZES_MB[system], "semantic_wer": asr.get(system), "utmos22": utmos.get(system)} for system in ORDER},
        "asr_robustness": {
            system: {
                asr_name: {
                    "wer": float(result["semantic"]["wer"]),
                    "ci": [float(value) for value in result["semantic_95ci"]],
                }
                for asr_name, result in asr_report["systems"][system]["asr"].items()
            }
            for system in ORDER
        },
        "family_averages": {
            "kitten": {
                "semantic_wer": family_average(asr, ["kitten-nano-bruno", "kitten-nano-hugo"], "wer"),
                "utmos22": family_average(utmos, ["kitten-nano-bruno", "kitten-nano-hugo"], "mean"),
            },
            "piper": {
                "semantic_wer": family_average(asr, ["piper-ryan-low", "piper-danny-low"], "wer"),
                "utmos22": family_average(utmos, ["piper-ryan-low", "piper-danny-low"], "mean"),
            },
        },
        "notes": ["Family averages are equal-weight macro averages over the two named voices.", "Supertonic 3-step and 8-step remain separate.", "UTMOS22 and WER are not combined into one score."],
    }
    (args.out / "evidence-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "systems": len(summary["systems"])}, indent=2))


if __name__ == "__main__":
    main()
