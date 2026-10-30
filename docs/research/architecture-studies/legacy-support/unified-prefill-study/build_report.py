#!/usr/bin/env python3
"""Build a self-contained HTML artifact for the unified prefill study."""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
REPORT = OUT / "UNIFIED_PREFIX_CACHE_PREFILL_STUDY.html"

COLORS = {
    "Controlled 5T factorial": "#46d6c3",
    "SWA and hybrid extensions": "#a78bfa",
    "Kimi K3 scaling": "#fb923c",
    "Controlled Kimi vs DeepSeek": "#f472b6",
    "Native sparse anchors": "#facc15",
    "DeepSeek scaling": "#60a5fa",
    "GLM scaling": "#34d399",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def f(value: str | float | None) -> float:
    if value in (None, "", "None"):
        return math.nan
    return float(value)


def fmt(value: str | float | None, digits: int = 1) -> str:
    value = f(value)
    return "—" if math.isnan(value) else f"{value:,.{digits}f}"


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)


def family_range_svg(summary: list[dict[str, str]]) -> str:
    width, height, left, right, top, row_h = 980, 330, 220, 40, 35, 38
    families = list(COLORS)
    values = [f(r["gb_1k_c1_latency_ms"]) for r in summary] + [f(r["rubin_1k_c1_latency_ms"]) for r in summary]
    x_max = max(values) * 1.03
    plot_w = width - left - right
    x = lambda v: left + plot_w * v / x_max
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Latency range by architecture family">']
    for tick in range(0, int(x_max) + 1, 200):
        xx = x(tick)
        parts += [f'<line x1="{xx:.1f}" y1="25" x2="{xx:.1f}" y2="305" class="grid"/>', f'<text x="{xx:.1f}" y="322" class="axis" text-anchor="middle">{tick}</text>']
    for i, family in enumerate(families):
        rows = [r for r in summary if r["family"] == family]
        y = top + i * row_h
        parts.append(f'<text x="{left-12}" y="{y+4}" class="label" text-anchor="end">{esc(family)} ({len(rows)})</text>')
        for system, dy, color, field in (
            ("GB300", -6, COLORS[family], "gb_1k_c1_latency_ms"),
            ("Rubin", 7, "#f8fafc", "rubin_1k_c1_latency_ms"),
        ):
            vals = [f(r[field]) for r in rows]
            q10, med, q90 = quantile(vals, .1), statistics.median(vals), quantile(vals, .9)
            parts += [
                f'<line x1="{x(q10):.1f}" y1="{y+dy}" x2="{x(q90):.1f}" y2="{y+dy}" stroke="{color}" stroke-width="5" stroke-linecap="round" opacity=".8"><title>{system}: P10 {q10:.1f}, median {med:.1f}, P90 {q90:.1f} ms</title></line>',
                f'<circle cx="{x(med):.1f}" cy="{y+dy}" r="5" fill="{color}" stroke="#07111f" stroke-width="1"><title>{system} median {med:.1f} ms</title></circle>',
            ]
    parts += [
        '<circle cx="760" cy="18" r="5" fill="#46d6c3"/><text x="772" y="22" class="axis">GB300</text>',
        '<circle cx="850" cy="18" r="5" fill="#f8fafc"/><text x="862" y="22" class="axis">Rubin projection</text>',
        '<text x="600" y="322" class="axis">1K extend latency (ms, lower is better)</text>',
        '</svg>',
    ]
    return "".join(parts)


def scatter_svg(summary: list[dict[str, str]]) -> str:
    width, height, left, right, top, bottom = 760, 600, 70, 30, 25, 65
    max_v = max(max(f(r["gb_1k_c1_latency_ms"]), f(r["rubin_1k_c1_latency_ms"])) for r in summary) * 1.04
    plot_w, plot_h = width-left-right, height-top-bottom
    x = lambda v: left + plot_w * v / max_v
    y = lambda v: top + plot_h * (1 - v / max_v)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="GB300 versus Rubin latency scatter">']
    for tick in range(0, int(max_v)+1, 200):
        parts += [
            f'<line x1="{x(tick):.1f}" y1="{top}" x2="{x(tick):.1f}" y2="{height-bottom}" class="grid"/>',
            f'<line x1="{left}" y1="{y(tick):.1f}" x2="{width-right}" y2="{y(tick):.1f}" class="grid"/>',
            f'<text x="{x(tick):.1f}" y="{height-bottom+22}" class="axis" text-anchor="middle">{tick}</text>',
            f'<text x="{left-10}" y="{y(tick)+4:.1f}" class="axis" text-anchor="end">{tick}</text>',
        ]
    parts.append(f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(max_v)}" y2="{y(max_v)}" stroke="#64748b" stroke-dasharray="6 6"/>')
    for row in summary:
        gx, ry = f(row["gb_1k_c1_latency_ms"]), f(row["rubin_1k_c1_latency_ms"])
        color = COLORS[row["family"]]
        title = f"{row['label']} | GB {gx:.1f} ms | Rubin {ry:.1f} ms | {gx/ry:.2f}x"
        parts.append(f'<circle cx="{x(gx):.1f}" cy="{y(ry):.1f}" r="3.4" fill="{color}" opacity=".72"><title>{esc(title)}</title></circle>')
    parts += [
        f'<text x="{left+plot_w/2}" y="{height-14}" class="axis">GB300 latency (ms)</text>',
        f'<text transform="translate(18 {top+plot_h/2}) rotate(-90)" class="axis">Rubin latency (ms)</text>',
        '</svg>',
    ]
    return "".join(parts)


def resource_svg(summary: list[dict[str, str]], selected: list[str]) -> str:
    rows = [next(r for r in summary if r["architecture_id"] == aid) for aid in selected]
    width, row_h, left, bar_w = 1000, 43, 285, 650
    height = 35 + len(rows) * row_h
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="GB300 resource shares">']
    for i, row in enumerate(rows):
        y = 22 + i * row_h
        shares = [
            (f(row["gb_1k_c1_attention_share_pct"]), "#60a5fa", "attention/cache"),
            (f(row["gb_1k_c1_dispatch_share_pct"]), "#f59e0b", "dispatch/fabric"),
            (f(row["gb_1k_c1_moe_compute_share_pct"]), "#34d399", "MoE compute/weights"),
        ]
        used = sum(v for v, _, _ in shares)
        shares.append((max(0, 100-used), "#64748b", "other"))
        parts.append(f'<text x="{left-12}" y="{y+15}" class="label" text-anchor="end">{esc(row["label"][:42])}</text>')
        cursor = left
        for value, color, label in shares:
            w = bar_w * value / 100
            parts.append(f'<rect x="{cursor:.1f}" y="{y}" width="{w:.1f}" height="22" fill="{color}"><title>{label}: {value:.1f}%</title></rect>')
            cursor += w
        parts.append(f'<text x="{left+bar_w+10}" y="{y+15}" class="axis">{f(row["gb_1k_c1_latency_ms"]):.0f} ms</text>')
    parts += [
        '<rect x="285" y="5" width="10" height="10" fill="#60a5fa"/><text x="300" y="14" class="axis">attention/cache</text>',
        '<rect x="410" y="5" width="10" height="10" fill="#f59e0b"/><text x="425" y="14" class="axis">dispatch/fabric</text>',
        '<rect x="535" y="5" width="10" height="10" fill="#34d399"/><text x="550" y="14" class="axis">MoE compute/weights</text>',
        '</svg>',
    ]
    return "".join(parts)


def curves_svg(results: list[dict[str, str]], selected: list[tuple[str, str]]) -> str:
    width, height, left, right, top, bottom = 980, 520, 72, 240, 25, 65
    plot_w, plot_h = width-left-right, height-top-bottom
    chunks = sorted({int(r["new_tokens_per_request"]) for r in results})
    lx0, lx1 = math.log2(min(chunks)), math.log2(max(chunks))
    x = lambda v: left + plot_w * (math.log2(v)-lx0)/(lx1-lx0)
    y = lambda v: top + plot_h * (1-v/1.05)
    palette = ["#46d6c3", "#f472b6", "#facc15", "#60a5fa", "#a78bfa", "#fb923c"]
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Normalized throughput saturation curves">']
    for value in (0, .25, .5, .75, .95, 1.0):
        parts += [f'<line x1="{left}" y1="{y(value):.1f}" x2="{left+plot_w}" y2="{y(value):.1f}" class="grid"/>', f'<text x="{left-10}" y="{y(value)+4:.1f}" class="axis" text-anchor="end">{value:.0%}</text>']
    for chunk in chunks:
        parts += [f'<line x1="{x(chunk):.1f}" y1="{top}" x2="{x(chunk):.1f}" y2="{top+plot_h}" class="grid"/>', f'<text x="{x(chunk):.1f}" y="{height-bottom+22}" class="axis" text-anchor="middle">{chunk if chunk<1024 else str(chunk//1024)+"K"}</text>']
    for i, (aid, short) in enumerate(selected):
        color = palette[i % len(palette)]
        for system, dash in (("GB300", ""), ("VR200 HBM+NVLink", "6 4")):
            points = sorted(
                (r for r in results if r["architecture_id"] == aid and r["system"] == system and int(r["cpr"]) == 1),
                key=lambda r: int(r["new_tokens_per_request"]),
            )
            roof = max(f(r["rack_input_tps"]) for r in results if r["architecture_id"] == aid and r["system"] == system)
            coords = " ".join(f'{x(int(r["new_tokens_per_request"])):.1f},{y(f(r["rack_input_tps"])/roof):.1f}' for r in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}" opacity=".9"><title>{esc(short)} · {system}</title></polyline>')
        ly = 42 + i*25
        parts += [f'<line x1="{left+plot_w+25}" y1="{ly}" x2="{left+plot_w+55}" y2="{ly}" stroke="{color}" stroke-width="3"/>', f'<text x="{left+plot_w+64}" y="{ly+4}" class="axis">{esc(short)}</text>']
    parts += [
        f'<line x1="{left+plot_w+25}" y1="{height-65}" x2="{left+plot_w+55}" y2="{height-65}" stroke="#e2e8f0" stroke-width="2"/><text x="{left+plot_w+64}" y="{height-61}" class="axis">GB300</text>',
        f'<line x1="{left+plot_w+25}" y1="{height-40}" x2="{left+plot_w+55}" y2="{height-40}" stroke="#e2e8f0" stroke-width="2" stroke-dasharray="6 4"/><text x="{left+plot_w+64}" y="{height-36}" class="axis">Rubin</text>',
        f'<text x="{left+plot_w/2}" y="{height-14}" class="axis">new tokens/request at c/r=1 (log₂)</text>',
        '<text transform="translate(18 270) rotate(-90)" class="axis">fraction of observed rack throughput roof</text>',
        '</svg>',
    ]
    return "".join(parts)


def dep_svg(dep_results: list[dict[str, str]], selected: list[tuple[str, str]]) -> str:
    width, height, left, right, top, bottom = 960, 500, 75, 210, 25, 60
    plot_w, plot_h = width-left-right, height-top-bottom
    deps = (8, 16, 32, 64)
    max_v = max(
        f(r["latency_ms"]) for r in dep_results
        if r["architecture_id"] in {x[0] for x in selected} and r["system"] == "GB300" and r["cpr"] == "1"
    ) * 1.08
    x = lambda d: left + plot_w * (math.log2(d)-3)/3
    y = lambda v: top + plot_h * (1-v/max_v)
    colors = ["#46d6c3", "#60a5fa", "#f472b6", "#facc15", "#a78bfa", "#fb923c"]
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="DEP sensitivity at concurrency per rank one">']
    for tick in range(0, int(max_v)+1, 100):
        parts += [f'<line x1="{left}" y1="{y(tick):.1f}" x2="{left+plot_w}" y2="{y(tick):.1f}" class="grid"/>', f'<text x="{left-10}" y="{y(tick)+4:.1f}" class="axis" text-anchor="end">{tick}</text>']
    for dep in deps:
        parts += [f'<line x1="{x(dep):.1f}" y1="{top}" x2="{x(dep):.1f}" y2="{top+plot_h}" class="grid"/>', f'<text x="{x(dep):.1f}" y="{height-bottom+23}" class="axis" text-anchor="middle">DEP{dep}</text>']
    for i, (aid, label) in enumerate(selected):
        color = colors[i]
        rows = sorted(
            (r for r in dep_results if r["architecture_id"] == aid and r["system"] == "GB300" and r["cpr"] == "1"),
            key=lambda r: int(r["dep"]),
        )
        coords = " ".join(f'{x(int(r["dep"])):.1f},{y(f(r["latency_ms"])):.1f}' for r in rows)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.7"><title>{esc(label)}</title></polyline>')
        for row in rows:
            oom = row["oom"] == "True"
            parts.append(f'<circle cx="{x(int(row["dep"])):.1f}" cy="{y(f(row["latency_ms"])):.1f}" r="5" fill="{color}" stroke="{"#fb923c" if oom else "#07111f"}" stroke-width="{3 if oom else 1}"><title>{esc(label)} · DEP{row["dep"]}: {f(row["latency_ms"]):.1f} ms{" · OOM" if oom else ""}</title></circle>')
        ly = 40+i*27
        parts += [f'<line x1="{left+plot_w+22}" y1="{ly}" x2="{left+plot_w+52}" y2="{ly}" stroke="{color}" stroke-width="3"/>', f'<text x="{left+plot_w+60}" y="{ly+4}" class="axis">{esc(label)}</text>']
    parts += [f'<text x="{left+plot_w/2}" y="{height-12}" class="axis">parallel group size · c/r=1</text>', '<text transform="translate(18 265) rotate(-90)" class="axis">GB300 1K extend latency (ms)</text>', '<text x="750" y="440" class="axis" fill="#fb923c">orange ring = OOM</text>', '</svg>']
    return "".join(parts)


def dep_exact_rows(dep_summary: list[dict[str, str]]) -> str:
    out = []
    for r in (x for x in dep_summary if x["study_kind"] == "direct AIC config"):
        best = int(r["gb_best_dep_cpr1"])
        gain = f(r["gb_dep64_c1_latency_ms"]) / f(r["gb_best_latency_cpr1_ms"]) if best else math.nan
        out.append(
            f'<tr><td>{esc(r["label"])}</td><td class="num">{best or "OOM"}</td>'
            f'<td class="num">{fmt(r["gb_best_latency_cpr1_ms"],1)}</td>'
            f'<td class="num">{fmt(r["gb_dep64_c1_latency_ms"],1)}</td><td class="num">{fmt(gain,2)}×</td>'
            f'<td class="num">{esc(r["gb_best_dep_global64"] or "—")}</td></tr>'
        )
    return "".join(out)


def table_rows(rows: list[dict[str, str]], exact_only: bool = False) -> str:
    body = []
    for row in rows:
        search = " ".join(str(row.get(k, "")) for k in ("label", "family", "body_id", "attention", "risk")).lower()
        body.append(
            f'<tr data-family="{esc(row["family"])}" data-risk="{esc(row["risk"])}" data-search="{esc(search)}">'
            f'<td class="sticky">{esc(row["label"])}</td><td>{esc(row["family"])}</td>'
            f'<td>{esc(row["attention"])}</td><td>{esc(row["risk"])}</td>'
            f'<td class="num">{fmt(row["total_t"],2)}</td><td class="num">{esc(row["depth"])}</td>'
            f'<td class="num">{esc(row["experts"])}/{esc(row["topk"])}</td>'
            f'<td class="num">{fmt(row["gb_1k_c1_latency_ms"],1)}</td>'
            f'<td class="num">{fmt(row["rubin_1k_c1_latency_ms"],1)}</td>'
            f'<td class="num">{fmt(row["rubin_latency_gain_1k_c1"],2)}×</td>'
            f'<td class="num">{esc(row["gb_chunk_c1_at_95pct"])}</td>'
            f'<td class="num">{esc(row["rubin_chunk_c1_at_95pct"])}</td>'
            f'<td class="num">{esc(row["gb_memory_max_cpr"])}</td>'
            f'<td>{esc(row["gb_1k_c1_dominant_resource"])}</td></tr>'
        )
    return "".join(body)


def main() -> None:
    summary = read_csv(OUT / "architecture_summary.csv")
    results = read_csv(OUT / "all_results.csv")
    dep_results = read_csv(OUT / "dep_sweep_results.csv")
    dep_summary = read_csv(OUT / "dep_sweep_summary.csv")
    metadata = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
    exact = [r for r in summary if r["study_kind"] == "direct AIC config"]
    gains = [f(r["rubin_latency_gain_1k_c1"]) for r in summary]
    gb_dom = Counter(r["gb_1k_c1_dominant_resource"] for r in summary)
    family_counts = Counter(r["family"] for r in summary)
    chunk_gb = Counter(r["gb_chunk_c1_at_95pct"] for r in summary)
    chunk_ru = Counter(r["rubin_chunk_c1_at_95pct"] for r in summary)

    selected_resources = [
        "exact__k3_2p8t", "exact__k3_5t_trend", "exact__k3_10t_trend",
        "exact__deepseek_v4_pro_native", "exact__deepseek_v4_pro_5t_expert",
        "exact__glm52_ratio_preserved_5t", "swa__gptoss_faithful",
        "factorial__latent_L61_balanced_m3072_s1__dsa1024",
    ]
    selected_curves = [
        ("exact__k3_2p8t", "K3 2.8T"),
        ("exact__k3_5t_trend", "Kimi-trend 5T"),
        ("exact__deepseek_v4_pro_5t_expert", "DS V4 5T"),
        ("exact__glm52_ratio_preserved_5t", "GLM 5T"),
        ("swa__gptoss_faithful", "GPT-OSS-like SWA"),
        ("factorial__latent_L61_balanced_m3072_s1__dsa1024", "5T latent+DSA"),
    ]
    selected_dep = [
        ("exact__deepseek_v4_pro_native", "DS V4 native"),
        ("exact__deepseek_v4_pro_5t_expert", "DS V4 5T"),
        ("exact__k3_2p8t", "K3 2.8T"),
        ("exact__k3_5t_expert", "K3 expert-only 5T"),
        ("exact__k3_5t_trend", "Kimi-trend 5T"),
        ("swa__gptoss_faithful", "GPT-OSS-like SWA"),
    ]
    gb_best_dep = Counter(r["gb_best_dep_cpr1"] for r in dep_summary)
    gb_best_g64 = Counter(r["gb_best_dep_global64"] for r in dep_summary)

    fam_cards = "".join(
        f'<div class="mini"><span style="background:{COLORS[name]}"></span><b>{count}</b><small>{esc(name)}</small></div>'
        for name, count in family_counts.items()
    )
    chunk_rows = "".join(
        f'<tr><td>{chunk}</td><td class="num">{chunk_gb.get(chunk,0)}</td><td class="num">{chunk_ru.get(chunk,0)}</td></tr>'
        for chunk in sorted(set(chunk_gb)|set(chunk_ru), key=lambda x: int(x))
    )

    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unified 256K-prefix prefill study</title>
<style>
:root{{--bg:#07111f;--panel:#0d1b2e;--panel2:#10243b;--text:#e7eef8;--muted:#9fb0c7;--line:#223b57;--cyan:#46d6c3;--orange:#fb923c}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 15% -10%,#15375a 0,transparent 35%),var(--bg);color:var(--text);font:15px/1.48 Inter,ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:38px 30px 80px}} h1{{font-size:clamp(35px,5vw,66px);line-height:1.02;max-width:1050px;margin:.1em 0 .25em;letter-spacing:-.045em}} h2{{font-size:28px;margin:60px 0 14px}} h3{{font-size:19px;margin:22px 0 9px}} p{{max-width:1050px;color:var(--muted)}} strong,b{{color:#fff}} .eyebrow{{color:var(--cyan);font-weight:800;letter-spacing:.14em;text-transform:uppercase}} .lede{{font-size:19px;max-width:1000px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:28px 0}} .card,.panel{{background:linear-gradient(145deg,rgba(16,36,59,.92),rgba(10,25,43,.92));border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 18px 40px #0004}} .card b{{display:block;font-size:31px;color:var(--cyan)}} .card small,.muted{{color:var(--muted)}} .families{{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}} .mini{{display:grid;grid-template-columns:10px auto;column-gap:9px;align-items:center;background:#0b192b;border:1px solid var(--line);border-radius:12px;padding:9px 12px}} .mini span{{width:9px;height:30px;border-radius:6px;grid-row:1/3}} .mini b{{font-size:18px}} .mini small{{color:var(--muted)}}
.callout{{border-left:4px solid var(--cyan);background:#0d2536;padding:17px 20px;border-radius:4px 12px 12px 4px;max-width:1100px}} .warn{{border-color:var(--orange)}} .grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:18px}} .viz{{overflow-x:auto}} svg{{width:100%;min-width:720px;height:auto}} svg .grid{{stroke:#26415f;stroke-width:1}} svg .axis{{fill:#9fb0c7;font-size:12px}} svg .label{{fill:#dce8f6;font-size:12px}}
.table-wrap{{overflow:auto;max-height:720px;border:1px solid var(--line);border-radius:14px;background:#091625}} table{{border-collapse:separate;border-spacing:0;width:100%;min-width:1300px}} th{{position:sticky;top:0;z-index:3;background:#10243b;color:#bcd0e8;text-align:left;font-size:12px;text-transform:uppercase;letter-spacing:.04em}} th,td{{padding:9px 11px;border-bottom:1px solid #18314b;white-space:nowrap}} td{{color:#c8d6e7}} tr:hover td{{background:#10243b}} .sticky{{position:sticky;left:0;z-index:1;background:#091625;max-width:390px;overflow:hidden;text-overflow:ellipsis}} th.sticky{{z-index:4;background:#10243b}} .num{{font-variant-numeric:tabular-nums;text-align:right}} .controls{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}} input,select{{background:#091625;color:#e7eef8;border:1px solid var(--line);border-radius:9px;padding:10px 12px;min-width:220px}} code{{background:#10243b;padding:.15em .35em;border-radius:5px}} a{{color:#72e6d5}} ul{{color:var(--muted);max-width:1050px}} .pill{{display:inline-block;border:1px solid #2b5774;border-radius:99px;padding:4px 9px;margin:2px;color:#bad2e9}} footer{{margin-top:55px;color:#7890aa}}
@media(max-width:700px){{main{{padding:24px 14px}}.grid2{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">AIC SOL · cached-prefix prefill · unified sweep</div>
<h1>256K resident cache + variable new-input prefill</h1>
<p class="lede">This supersedes the SWA-only artifact. Every prior architecture family now uses one workload contract: a 262,144-token resident prefix, 1–4,096 new tokens/request, concurrency/rank 1–64, DEP64, GB300 versus the controlled VR200-HBM + Rubin-NVLink system.</p>
<div class="cards"><div class="card"><b>{len(summary)}</b><small>architecture configurations</small></div><div class="card"><b>{len(results):,}</b><small>composed sweep points</small></div><div class="card"><b>{statistics.median(gains):.2f}×</b><small>median Rubin latency gain at 1K, c/r=1</small></div><div class="card"><b>1</b><small>c/r needed for ≥95% of the observed roof at a 1K chunk—all configs</small></div></div>
<div class="families">{fam_cards}</div>

<h2>Answer: when does prefill saturate?</h2>
<div class="callout"><strong>At the requested 1K new-token chunk, every one of the 506 configurations is already at or above 95% of its observed rack input-throughput roof at concurrency/rank 1.</strong> Raising concurrency does not materially increase steady-state input-token throughput; it mostly makes each synchronized pass longer. The saturation unit is tokens processed per rank: a single 1K request/rank is already a large GEMM/MoE batch.</div>
<div class="grid2"><div class="panel"><h3>New-token threshold at c/r=1</h3><p>The table counts architectures by the first chunk that reaches 95% of the maximum seen anywhere in the chunk × concurrency grid.</p><div class="table-wrap" style="max-height:390px"><table style="min-width:420px"><thead><tr><th>new tokens/request</th><th>GB300 configs</th><th>Rubin configs</th></tr></thead><tbody>{chunk_rows}</tbody></table></div></div>
<div class="panel"><h3>What resource is saturated at 1K?</h3><p>On GB300 the largest latency component is <b>MoE dispatch/fabric for {gb_dom['MoE dispatch/fabric']} configs</b>, attention/cache for {gb_dom['attention/cache']}, and MoE compute/weight access for {gb_dom['MoE compute/weights']}. “Saturated” is architecture-dependent: usually the all-to-all path is at its SOL floor; full/long-context attention designs can instead be attention/cache limited; the highest-active/deep bodies can be MoE compute limited.</p><p>Rubin’s faster memory and fabric move many designs toward compute: its median fixed-workload gain is {statistics.median(gains):.2f}×, with a {min(gains):.2f}–{max(gains):.2f}× range—not the full 2.75× HBM ratio once compute and non-HBM operations dominate.</p></div></div>

<h2>Is DEP64 optimal for prefill?</h2>
<div class="callout"><strong>Not at fixed concurrency/rank.</strong> At c/r=1, the best DEP is the smallest group that fits: DEP8 for {gb_best_dep['8']} configs, DEP16 for {gb_best_dep['16']}, and DEP32 for {gb_best_dep['32']}; the two 10T depth/trend configurations do not fit even at DEP64. All 448 factorial and all 43 SWA/hybrid architectures choose DEP16. But at fixed global concurrency=64 in one model instance, DEP64 wins for all {gb_best_g64['64']} fitting configurations, because DEP16 must process four requests/rank.</div>
<div class="grid2"><div class="panel"><h3>GB300 DEP curves at c/r=1</h3><p>Smaller groups reduce dispatch scope, while fewer ranks increase resident weights and expert work. The orange-ring points are latency projections that fail the HBM capacity check.</p><div class="viz">{dep_svg(dep_results, selected_dep)}</div></div><div class="panel"><h3>Two different deployment questions</h3><p><b>Several narrow replicas:</b> on 64 GPUs, four independent DEP16 groups at c/r=1 jointly serve 64 requests. They retain DEP16’s lower latency and higher normalized rack throughput, at the cost of four copies of the model across the allocation.</p><p><b>One replica serving 64 requests:</b> DEP16 requires c/r=4, DEP32 requires c/r=2, and DEP64 needs c/r=1. Here DEP64 is fastest because the smaller group places proportionally more attention and expert work on each GPU.</p><p>Example, DeepSeek V4 5T on GB300: DEP16/c/r1 is 83.3 ms versus DEP64/c/r1 at 169.1 ms. Dispatch falls from 125.4 to 29.8 ms, while MoE compute rises from 9.9 to 19.7 ms. At fixed global concurrency 64, however, DEP16/c/r4 is 293.3 ms versus DEP64/c/r1 at 169.1 ms.</p></div></div>
<h3>Direct configurations</h3><div class="table-wrap" style="max-height:520px"><table style="min-width:850px"><thead><tr><th>configuration</th><th>best DEP at c/r1</th><th>best ms</th><th>DEP64 ms</th><th>speedup</th><th>best DEP at global=64</th></tr></thead><tbody>{dep_exact_rows(dep_summary)}</tbody></table></div>

<h2>Architecture-family latency envelope</h2><p>Dots mark medians; bars span P10–P90 within each family. The 448-way factorial dominates the population, so family medians are shown separately rather than pooled.</p><div class="panel viz">{family_range_svg(summary)}</div>

<h2>GB300 versus Rubin</h2><p>Every point is one configuration at 1K new tokens and c/r=1. Points near the diagonal receive little benefit because compute or attention math dominates; points farther below it expose more HBM/fabric headroom.</p><div class="panel viz" style="max-width:1000px">{scatter_svg(summary)}</div>

<h2>Why the chunk threshold differs</h2><p>At tiny chunks, constant per-forward work—resident weights, expert dispatch setup, projection weights, and the 256K-prefix attention scan/index—dominates. Increasing new tokens amortizes that floor until tensor-core work, fabric bytes, or attention work grows linearly with the chunk. Rubin reaches its roof earlier because its fixed HBM/fabric floor is lower. A model with deeper/full attention often needs 256–512 new tokens; a lean sparse/linear design can reach the knee at 32–128.</p><div class="panel viz">{curves_svg(results, selected_curves)}</div>

<h2>Representative GB300 resource composition</h2><p>Shares are additive AIC operation groups at 1K and c/r=1. “Attention/cache” includes attention projections and cache/index reads; “MoE compute/weights” is expert execution/weight traffic; dispatch is the expert all-to-all path.</p><div class="panel viz">{resource_svg(summary, selected_resources)}</div>

<h2>Direct model and scaling configurations</h2><p>These 15 rows invoke AIC on a concrete model config or built-in model geometry. The native DeepSeek/GLM anchors and the Kimi scale-up series appear alongside the controlled ratio/MoE-normalized variants. The 448 factorial and 43 SWA rows below are controlled component compositions.</p><div class="table-wrap"><table><thead><tr><th class="sticky">configuration</th><th>family</th><th>attention</th><th>status</th><th>total T</th><th>layers</th><th>experts/top-k</th><th>GB ms</th><th>Rubin ms</th><th>gain</th><th>GB 95% chunk</th><th>Rubin 95% chunk</th><th>max c/r by HBM</th><th>GB bottleneck</th></tr></thead><tbody>{table_rows(exact)}</tbody></table></div>

<h2>All 506 configurations</h2><p>The table contains every direct, factorial, and SWA/hybrid config in one schema. “Max c/r by HBM” is the linear boundary inferred from AIC memory at c/r 1 and 2 for the 1K workload; zero means even c/r=1 is OOM. It is a capacity boundary, not a latency recommendation.</p>
<div class="controls"><input id="search" placeholder="Search architecture, body, attention…"><select id="family"><option value="">All families</option>{''.join(f'<option>{esc(x)}</option>' for x in COLORS)}</select><select id="risk"><option value="">All risk labels</option><option>native/scaling hypothesis</option><option>conservative</option><option>balanced</option><option>aggressive</option></select><span id="visible" class="pill"></span></div>
<div class="table-wrap"><table id="all"><thead><tr><th class="sticky">configuration</th><th>family</th><th>attention</th><th>risk/status</th><th>total T</th><th>layers</th><th>experts/top-k</th><th>GB ms</th><th>Rubin ms</th><th>gain</th><th>GB 95% chunk</th><th>Rubin 95% chunk</th><th>max c/r by HBM</th><th>GB bottleneck</th></tr></thead><tbody>{table_rows(summary)}</tbody></table></div>

<h2>Method and interpretation</h2><div class="grid2"><div class="panel"><h3>Direct versus composite</h3><ul><li><b>15 direct configs:</b> AIC <code>static_ctx</code> is run on the complete model geometry.</li><li><b>448 factorial configs:</b> 28 MoE bodies × 16 attention profiles. Native attention is subtracted and a prefix-aware KDA/MLA/V4/DSA component is added.</li><li><b>43 SWA/hybrid configs:</b> the common 61-layer 5T body is combined with full GQA, SWA window/ratio grids, GPT-OSS/Inkling-like shapes, and SWA+MLA/V4/DSA/KDA hybrids.</li><li>DSA top-k 512/1024/4096 uses AIC’s direct top-2048 context result scaled by the exact <code>ContextDSAModule</code> SOL-operation ratio; per-layer versus shared indexing is measured directly.</li></ul></div><div class="panel"><h3>Fixed contract</h3><ul><li>SGLang estimate backend, SOL database mode.</li><li>TP1, PP1, attention-DP64, MoE-EP64.</li><li>FP8 trunk, NVFP4 routed MoE, FP8 KV, BF16 FMHA.</li><li>GB300 usable modeled HBM: {metadata['capacity_gib']:.1f} GiB/rank. Rubin preserves capacity and GB300 compute, replacing HBM bandwidth with 22 TB/s and fabric with 1.8 TB/s.</li><li>No cache lookup/transfer, queueing, scheduler loss, prefill/decode interference, or framework efficiency discount.</li><li>Speculative decoding is excluded because this is prefill, not autoregressive decode.</li></ul></div></div>
<div class="callout warn"><strong>Read these as comparative SOL ceilings, not achieved online service numbers.</strong> A 1K synchronized extend pass is enough to fill the modeled processing pipeline, but production throughput can be lower from routing imbalance, kernel/utilization gaps, communication overlap limits, cache movement, and scheduler interference.</div>

<h2>Data and reproduction</h2><p><a href="all_results.csv">All 92,092 chunk/concurrency rows (CSV)</a> · <a href="architecture_summary.csv">Architecture summary (CSV)</a> · <a href="dep_sweep_results.csv">All 16,192 DEP rows (CSV)</a> · <a href="dep_sweep_summary.csv">DEP summary (CSV)</a> · <a href="metadata.json">Method metadata (JSON)</a></p><p><code>.venv-aic-kimi-k3/bin/python experiments/unified-prefill-study/run_unified_prefill.py</code><br><code>.venv-aic-kimi-k3/bin/python experiments/unified-prefill-study/sweep_dep.py</code><br><code>.venv-aic-kimi-k3/bin/python experiments/unified-prefill-study/build_report.py</code></p>
<footer>Generated from local AIC branch evidence. This unified report supersedes the narrower SWA-only prefill artifact.</footer>
<script>
const q=document.querySelector('#search'), fam=document.querySelector('#family'), risk=document.querySelector('#risk'), rows=[...document.querySelectorAll('#all tbody tr')], visible=document.querySelector('#visible');
function filter(){{let n=0; const needle=q.value.toLowerCase(); rows.forEach(r=>{{const show=(!needle||r.dataset.search.includes(needle))&&(!fam.value||r.dataset.family===fam.value)&&(!risk.value||r.dataset.risk===risk.value);r.style.display=show?'':'none';if(show)n++}});visible.textContent=n+' visible'}}
[q,fam,risk].forEach(x=>x.addEventListener('input',filter));filter();
</script></main></body></html>'''
    REPORT.write_text(document, encoding="utf-8")
    print(f"Wrote {REPORT} ({REPORT.stat().st_size/2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
