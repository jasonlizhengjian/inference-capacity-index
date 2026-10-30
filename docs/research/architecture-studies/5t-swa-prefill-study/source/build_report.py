#!/usr/bin/env python3
"""Build a self-contained HTML report for the cached-prefix prefill study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
DESTINATION = OUTPUT / "PREFIX_CACHE_PREFILL_STUDY.html"
CURVE_IDS = {
    "full_gqa",
    "gptoss_faithful",
    "common_r3_w512",
    "inkling_faithful",
    "swa5_mla",
    "swa5_dsa",
    "swa5_kda",
    "swa5_v4",
    "mla_all",
    "dsa2048_all",
    "kda_all",
    "v4_all",
}


def parsed(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for source in csv.DictReader(stream):
            row: dict[str, Any] = {}
            for key, value in source.items():
                if value == "":
                    row[key] = None
                elif value in {"True", "False"}:
                    row[key] = value == "True"
                else:
                    try:
                        number = float(value)
                        row[key] = int(number) if number.is_integer() else number
                    except ValueError:
                        row[key] = value
            rows.append(row)
    return rows


def reference_breakdowns() -> dict[str, dict[str, float]]:
    found: dict[str, dict[str, float]] = {}
    with (OUTPUT / "prefill_estimate_cache.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["cpr"] != 1 or not row["model"].endswith("common_global"):
                continue
            latency = row["latency"]
            attention_names = {
                "context_attention",
                "context_global_attn_norm",
                "context_global_qkv_gemm",
                "context_global_proj_gemm",
            }
            dispatch = latency["context_global_moe_pre_dispatch"] + latency["context_global_moe_post_dispatch"]
            moe = latency["context_global_moe"]
            attention = sum(latency.get(name, 0.0) for name in attention_names)
            body = row["context_latency_ms"] - attention
            found[row["system"]] = {
                "total_ms": row["context_latency_ms"],
                "body_ms": body,
                "attention_ms": attention,
                "dispatch_ms": dispatch,
                "moe_ms": moe,
                "body_other_ms": body - dispatch - moe,
            }
    return found


def main() -> None:
    summaries = parsed(OUTPUT / "architecture_summary.csv")
    all_rows = parsed(OUTPUT / "all_results.csv")
    curves = [
        {key: row[key] for key in (
            "architecture_id",
            "label",
            "system",
            "cpr",
            "extend_latency_ms",
            "input_tps_user",
            "rack_input_tps",
            "memory_gib",
            "oom",
            "body_ms",
            "attention_ms",
        )}
        for row in all_rows
        if row["architecture_id"] in CURVE_IDS
    ]
    metadata = json.loads((OUTPUT / "metadata.json").read_text(encoding="utf-8"))
    breakdowns = reference_breakdowns()
    saturation_summary = parsed(OUTPUT / "saturation_summary.csv")
    saturation_rows = parsed(OUTPUT / "saturation_results.csv")

    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cached-prefix prefill at 5T · 256K + 1K</title>
<style>
:root{--bg:#090c13;--panel:#111725;--panel2:#171f30;--ink:#f3f5f8;--muted:#9aa7ba;--line:#29334a;--cyan:#69d7e8;--amber:#f4b860;--violet:#aa8dff;--rose:#ee7da8;--red:#f07972;--green:#8fd17f;--blue:#79a8ff;--accent:#dbea7d}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 85% 0,#182847 0,transparent 32rem),radial-gradient(circle at 0 30%,#162834 0,transparent 32rem),var(--bg);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}.page{width:min(1480px,calc(100% - 32px));margin:auto;padding:24px 0 72px}.hero{padding:clamp(31px,5vw,70px);border:1px solid var(--line);border-radius:28px;background:linear-gradient(145deg,#172033f5,#0d121df5);box-shadow:0 28px 90px #0006;position:relative;overflow:hidden}.hero:after{content:"256K + 1K";position:absolute;right:-18px;bottom:-42px;font-size:clamp(70px,13vw,190px);font-weight:900;letter-spacing:-.08em;color:#69d7e808;white-space:nowrap}.eyebrow,.kicker{text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:800;color:var(--accent)}h1{font-size:clamp(43px,6.5vw,84px);line-height:.96;letter-spacing:-.06em;max-width:1050px;margin:14px 0 18px}.lede{font-size:clamp(17px,2vw,22px);color:#c5cddb;max-width:1020px}.chips,.controls,.legend{display:flex;flex-wrap:wrap;gap:8px}.chips{margin-top:28px}.chip,button,select,input{border:1px solid var(--line);border-radius:99px;background:#0c1220;color:var(--muted);padding:8px 12px;font:inherit;font-size:12px}button{cursor:pointer}button:hover,button.active{background:#25314a;color:var(--ink);border-color:#526582}.section{margin-top:62px}.head{display:grid;grid-template-columns:minmax(0,1fr) minmax(300px,600px);gap:28px;align-items:end;margin-bottom:22px}h2{margin:7px 0 0;font-size:clamp(29px,4vw,50px);line-height:1.05;letter-spacing:-.04em}h3{margin:0;font-size:19px}.copy{color:var(--muted);font-size:16px}.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{border:1px solid var(--line);border-radius:19px;background:#111725ed;box-shadow:0 14px 48px #0003}.kpi{padding:21px;min-height:170px}.kpi small{color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.kpi strong{display:block;font-size:clamp(28px,3vw,43px);line-height:1;margin:14px 0 9px;letter-spacing:-.045em}.kpi p{color:#76839a;font-size:12px}.cyan{color:var(--cyan)}.amber{color:var(--amber)}.violet{color:var(--violet)}.green{color:var(--green)}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:18px}table{width:100%;min-width:980px;border-collapse:collapse;background:#111725ed;font-variant-numeric:tabular-nums}table.compact{min-width:0}th,td{padding:12px 14px;border-bottom:1px solid #252f44;text-align:right}th:first-child,td:first-child{text-align:left}th{font-size:10px;text-transform:uppercase;letter-spacing:.075em;color:var(--muted);background:#171f30;white-space:nowrap}tr:last-child td{border:0}tbody tr:hover{background:#69d7e808}.note{margin-top:14px;padding:14px;border-left:3px solid var(--accent);background:#0c1220;color:#adbacd;font-size:13px}.equation{padding:20px;margin:16px 0;background:#0b111d;border-left:3px solid var(--cyan);font:600 17px/1.5 ui-monospace,monospace}.viz-grid{display:grid;grid-template-columns:1.15fr .85fr;gap:14px}.viz{padding:20px;overflow:hidden}.viz-head{display:flex;justify-content:space-between;gap:18px;align-items:start;margin-bottom:12px}.viz-head p{color:#77849a;font-size:12px;margin:4px 0 0}.chart{min-height:390px}.axis{fill:#7d899e;font-size:11px}.gridline{stroke:#283249;stroke-width:1}.legend{margin:9px 0}.legend button{padding:6px 9px}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.heatmap table{min-width:0}.heatmap td{text-align:center;font-weight:750;border:2px solid var(--panel);height:68px}.heatmap th{text-align:center}.heatmap th:first-child{text-align:left}.slo-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.slo{padding:22px}.slo strong{display:block;font-size:31px;margin:10px 0;color:var(--cyan)}.slo p{color:var(--muted)}.insights{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.insight{padding:24px;min-height:230px}.num{font:800 12px ui-monospace,monospace;color:var(--accent)}.insight h3{font-size:21px;margin:12px 0 10px}.insight p{color:var(--muted)}.limits{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.limit{padding:19px}.limit b{display:block;margin-bottom:5px}.limit span{color:var(--muted)}.filterbar{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:12px}.filterbar input{border-radius:10px;min-width:260px}.filterbar select{border-radius:10px}a{color:var(--cyan)}footer{margin-top:55px;border-top:1px solid var(--line);padding-top:20px;color:#748097;font-size:12px;display:flex;justify-content:space-between;gap:20px}code{color:#cbd5e5}@media(max-width:1050px){.grid4{grid-template-columns:1fr 1fr}.head,.viz-grid{grid-template-columns:1fr}.slo-grid,.insights{grid-template-columns:1fr 1fr}}@media(max-width:680px){.page{width:calc(100% - 16px);padding-top:8px}.hero{border-radius:18px}.grid4,.slo-grid,.insights,.limits{grid-template-columns:1fr}.chart{min-height:300px}footer{display:block}}
</style>
</head>
<body><main class="page">
<header class="hero"><div class="eyebrow">Separate prefill study · AIC speed of light · September 2026</div><h1>Cached-prefix prefill at 5T</h1><p class="lede">Every request starts with exactly 262,144 tokens of resident cache and extends it with exactly 1,024 new input tokens. The sweep mirrors the controlled decode architecture breadth, but uses AIC’s prefix-aware <code>static_ctx</code> path rather than generation.</p><div class="chips"><span class="chip">256K resident prefix</span><span class="chip">+1K new input</span><span class="chip">257K postfill context</span><span class="chip">43 architectures</span><span class="chip">DEP64</span><span class="chip">c/r 1–1,024</span><span class="chip">GB300 + projected Rubin</span><span class="chip">raw SOL</span></div></header>

<section class="section"><div class="head"><div><div class="kicker">Executive result</div><h2>One 1K extend already fills the machine with work</h2></div><p class="copy">At c/r=1 there are already 64 concurrent requests rack-wide and 65,536 new input tokens in flight. For this large chunk, AIC’s SOL latency is almost perfectly proportional to c/r, so higher batching adds almost no aggregate prefill throughput—it mainly adds latency and consumes cache capacity.</p></div><div class="grid4"><article class="card kpi"><small>All-global GQA · GB · c/r 1</small><strong class="amber" id="kpi-full">—</strong><p>Dense full-prefix attention is compute-bound and barely benefits from faster HBM.</p></article><article class="card kpi"><small>Fastest controlled profile · GB · c/r 1</small><strong class="cyan" id="kpi-fast">—</strong><p id="kpi-fast-note">—</p></article><article class="card kpi"><small>Best 500-ms concurrency · GB → Rubin</small><strong class="green" id="kpi-service">—</strong><p>Global concurrent requests meeting the chunk-latency target.</p></article><article class="card kpi"><small>Largest memory admission</small><strong class="violet" id="kpi-memory">—</strong><p id="kpi-memory-note">—</p></article></div></section>

<section class="section"><div class="head"><div><div class="kicker">Exact workload semantics</div><h2>This is an extend, not a fresh 257K prefill</h2></div><p class="copy">AIC receives <code>ISL=263,168</code> and <code>prefix=262,144</code>. GEMMs, MoE, and projections process only the 1,024 uncached query tokens; attention sees those queries against the resident prefix plus earlier tokens in the new chunk.</p></div><div class="slo-grid"><article class="card slo"><small>Per attention-DP rank</small><strong>c/r × 1,024</strong><p>new tokens processed in the synchronized forward pass.</p></article><article class="card slo"><small>Rack-wide concurrency</small><strong>64 × c/r</strong><p>independent requests because attention DP is 64.</p></article><article class="card slo"><small>Rack input throughput</small><strong>64·c/r·1,024 / latency</strong><p>new input tokens/s; this is static batch throughput, not an arrival-rate simulation.</p></article></div><div class="note">The cached prefix is assumed local and ready. Prefix lookup, hit probability, remote KV fetch, and P/D KV transfer are outside this study.</div></section>

<section class="section"><div class="head"><div><div class="kicker">Saturation boundary</div><h2>Chunk length matters more than request concurrency</h2></div><p class="copy">Here, “saturated” means reaching 95% of the largest observed SOL rack input throughput. A 1K chunk reaches that boundary at c/r=1 for every studied profile. Shorter chunks may need more request concurrency to amortize weights, but every extra request also brings a separate 256K cache, so concurrency cannot fully replace per-request query length.</p></div><div class="viz-grid"><article class="card viz"><div class="viz-head"><div><h3>Rack input throughput versus new tokens/request</h3><p>c/r=1 · prefix remains 256K · log-scaled chunk axis</p></div><div class="controls"><button data-sat-system="GB300" class="active">GB300</button><button data-sat-system="VR200 HBM+NVLink">Rubin</button></div></div><div class="legend" id="sat-legend"></div><div class="chart" id="sat-chart"></div></article><article class="card viz"><div class="viz-head"><div><h3>95% full-roof chunk boundary</h3><p>Minimum new tokens/request at c/r=1.</p></div></div><table class="compact"><thead><tr><th>Architecture</th><th>GB300</th><th>Rubin</th><th>1K c/r for 95%</th></tr></thead><tbody id="sat-table"></tbody></table><div class="note">For the sparse/recurrent 5:1 designs, the defensible 95% boundary is ≈512 new tokens/request on GB300 and ≈256 on Rubin. Full GQA reaches it sooner—128 and 64—because its 256K-prefix attention becomes compute-bound at a very small query length.</div></article></div><div class="viz-grid" style="margin-top:14px"><article class="card viz"><div class="viz-head"><div><h3>Concurrency needed for each fixed chunk</h3><p>Smallest c/r reaching 95% of that chunk shape’s own concurrency ceiling.</p></div></div><table class="compact"><thead><tr><th>New tokens / request</th><th>Full GQA · GB</th><th>SWA/DSA · GB</th><th>SWA/DSA · Rubin</th></tr></thead><tbody id="shape-sat-table"></tbody></table><div class="note">The low-token ceilings are much lower. For example, batching one-token extends to c/r=32 “saturates” that shape at only ≈17K rack input TPS for SWA/DSA, versus ≈370K for the requested 1K shape.</div></article><article class="card viz"><div class="viz-head"><div><h3>What is saturated at 1K, c/r=1?</h3><p>Share of modeled latency; each operation has its own roofline.</p></div></div><table class="compact"><thead><tr><th>Architecture</th><th>System</th><th>Dispatch / fabric</th><th>MoE compute</th><th>Attention</th><th>Primary limit</th></tr></thead><tbody id="resource-table"></tbody></table><div class="note">There is no single whole-pass bandwidth ceiling. Operations execute sequentially: expert dispatch is fabric-bound, routed expert GEMMs are tensor-compute-bound at 1K, and dense full-prefix attention is tensor-compute-bound. HBM dominates only the shorter-chunk weight/cache-loading regime.</div></article></div></section>

<section class="section"><div class="head"><div><div class="kicker">Representative endpoints</div><h2>Latency, serviceability, and memory tell different stories</h2></div><p class="copy">“500 ms max” is the largest memory-fitting c/r whose whole 1K chunk completes within 500 ms. “Memory max” is only a residency boundary. The two should not be conflated.</p></div><div class="table-wrap"><table><thead><tr><th>Architecture</th><th>KV GiB / req</th><th>GB c/r1 latency</th><th>GB input TPS/user</th><th>Rubin c/r1 latency</th><th>Rubin gain</th><th>500 ms c/r GB / Rubin</th><th>Memory max c/r</th><th>Memory max global</th><th>Latency at memory max GB / Rubin</th></tr></thead><tbody id="endpoint-table"></tbody></table></div></section>

<section class="section"><div class="head"><div><div class="kicker">Concurrency curves</div><h2>Latency rises; rack throughput plateaus</h2></div><div class="controls"><button data-system="GB300" class="active">GB300</button><button data-system="VR200 HBM+NVLink">Rubin projection</button><button data-max="16" class="active">c/r 16</button><button data-max="64">c/r 64</button><button data-max="256">c/r 256</button><button data-max="1024">c/r 1,024</button><button data-xscale="log" class="active">log x</button><button data-xscale="linear">linear x</button></div></div><article class="card viz"><div class="viz-head"><div><h3 id="curve-title">1K extend latency</h3><p>Solid through the modeled memory boundary; dashed afterward. Target lines appear in latency mode.</p></div><div class="controls"><button data-metric="extend_latency_ms" class="active">Latency</button><button data-metric="input_tps_user">Input TPS/user</button><button data-metric="rack_input_tps">Rack input TPS</button></div></div><div class="legend" id="curve-legend"></div><div class="chart" id="curve-chart"></div></article></section>

<section class="section"><div class="head"><div><div class="kicker">Mechanism comparison</div><h2>The long-range path still matters during extend</h2></div><p class="copy">At 5:1 local/global, sparse DSA, recurrent KDA, and V4 compression all remove most of the dense prefix-attention cost. Full MLA is not cheap here: it still performs full-context attention for every new query.</p></div><div class="table-wrap"><table><thead><tr><th>Architecture</th><th>Global mechanism</th><th>GB c/r1</th><th>GB c/r4</th><th>Attention share at c/r1</th><th>GB rack input TPS</th><th>Rubin gain</th><th>1,000 ms c/r GB / Rubin</th><th>Memory max c/r</th></tr></thead><tbody id="hybrid-table"></tbody></table></div></section>

<section class="section"><div class="head"><div><div class="kicker">SWA design space</div><h2>Global-layer frequency dominates window tuning again</h2></div><div class="controls" id="heat-controls"><button data-heat="gb_c1_latency_ms" class="active">GB c/r1 latency</button><button data-heat="gb_c4_latency_ms">GB c/r4 latency</button><button data-heat="gb_max_cpr_500ms">500-ms c/r</button><button data-heat="kv_gib_per_request">KV GiB/request</button></div></div><article class="card viz heatmap"><div class="viz-head"><div><h3 id="heat-title">GB300 1K-extend latency at c/r=1</h3><p>Rows vary local window; columns vary local layers per full-global layer.</p></div></div><div id="heatmap"></div><div class="note">With a 256K prefix, changing the number of full-context layers has much more effect than tuning a local window already thousands of times smaller than the prefix.</div></article></section>

<section class="section"><div class="head"><div><div class="kicker">Body decomposition</div><h2>Fast attention exposes the common MoE floor</h2></div><p class="copy">The common body is 61 layers, hidden 7,168, 1,216 routed experts, top-25, expert intermediate 3,072, zero shared experts, and DEP64. Every attention curve uses this exact body.</p></div><div class="viz-grid"><article class="card viz"><div class="viz-head"><div><h3>GB300 c/r=1: common body + attention</h3><p>Milliseconds for one 1K extend chunk.</p></div></div><div class="chart" id="decomp-chart"></div></article><article class="card viz"><div class="viz-head"><div><h3>All-global reference operation floor</h3><p>Common body after removing global attention.</p></div></div><table class="compact"><thead><tr><th>Component</th><th>GB300</th><th>Rubin projection</th></tr></thead><tbody id="body-table"></tbody></table><div class="note">GB300 spends about three quarters of the common-body time in MoE pre/post dispatch. Rubin’s doubled fabric halves that term, while the MoE GEMM stays compute-bound. Dense full-prefix attention also stays almost unchanged because it is compute-bound at this query shape.</div></article></div></section>

<section class="section"><div class="head"><div><div class="kicker">Capacity versus latency</div><h2>Compact state does not make huge batches useful</h2></div><p class="copy">KDA and V4 allow hundreds of resident requests per rank, but a 1K chunk is already large enough to saturate throughput at c/r=1. The additional cache slots are useful for residency and scheduling flexibility, not for one enormous synchronized prefill batch.</p></div><div class="table-wrap"><table><thead><tr><th>Architecture</th><th>250 ms max c/r GB / Rubin</th><th>500 ms</th><th>1,000 ms</th><th>2,000 ms</th><th>Memory max c/r</th><th>GB latency at max</th><th>Rubin latency at max</th></tr></thead><tbody id="capacity-table"></tbody></table></div></section>

<section class="section"><div class="head"><div><div class="kicker">All 43 profiles</div><h2>Full controlled sweep</h2></div><p class="copy">Search or filter the complete architecture set. Each summary comes from direct integer c/r evaluations through 1,024; the raw CSV retains all 88,064 composed points.</p></div><div class="filterbar"><input id="search" placeholder="Search architecture…"><select id="category"><option value="">All categories</option><option>control</option><option>ratio-window grid</option><option>faithful-inspired</option><option>cross-mechanism hybrid</option><option>mechanism control</option></select></div><div class="table-wrap"><table><thead><tr><th>Architecture</th><th>Category</th><th>Local / global</th><th>Mechanism</th><th>Window</th><th>KV GiB/req</th><th>GB c/r1 ms</th><th>Rubin c/r1 ms</th><th>500-ms c/r GB / R</th><th>Memory max c/r</th><th>Attention share</th></tr></thead><tbody id="all-table"></tbody></table></div></section>

<section class="section"><div class="head"><div><div class="kicker">Interpretation</div><h2>What changes relative to decode</h2></div><p class="copy">These are the robust architectural takeaways from the raw SOL model.</p></div><div class="insights"><article class="card insight"><span class="num">01</span><h3>No batching dividend remains</h3><p>At c/r=1 each rank already processes 1,024 new tokens. The large GEMMs and attention kernels are fully occupied, so doubling c/r nearly doubles latency and leaves rack input TPS nearly flat.</p></article><article class="card insight"><span class="num">02</span><h3>Dense attention is compute-bound</h3><p>The all-global attention primitive is about 191.5 ms on both systems at c/r=1. Faster Rubin HBM does not accelerate the quadratic query–prefix math when GB300 compute is intentionally held constant.</p></article><article class="card insight"><span class="num">03</span><h3>Rubin helps the MoE path</h3><p>When sparse/recurrent attention shrinks to a few percent of latency, Rubin’s doubled interconnect halves expert dispatch. The fixed-c/r gain rises to roughly 1.6×—still below the 2.75× HBM ratio.</p></article><article class="card insight"><span class="num">04</span><h3>Memory max is not service max</h3><p>A 5:1 SWA/V4 profile can admit hundreds of requests/rank, yet only a handful can share one 1K extend batch under a 500–1,000 ms chunk target.</p></article><article class="card insight"><span class="num">05</span><h3>DSA and KDA converge here</h3><p>At 5:1 they are almost tied on GB300 because both reduce the global attention term enough that the identical MoE body dominates. Their state capacity and modeling risk remain very different.</p></article><article class="card insight"><span class="num">06</span><h3>Speculation is absent by design</h3><p>Draft acceptance length accelerates decode verification, not target-model prefill. No speculative-decoding assumption or DSpark acceptance curve is applied to this report.</p></article></div></section>

<section class="section"><div class="head"><div><div class="kicker">Method and limits</div><h2>What these numbers do—and do not—mean</h2></div><p class="copy">This is a controlled operation-level study, not a serving benchmark. It is best read as a speed-of-light comparison of architecture scaling under an exact cached-prefix shape.</p></div><div class="limits"><article class="card limit"><b>Prefix-aware AIC path</b><span><code>static_ctx</code> receives total ISL and cached prefix separately. Only the 1K uncached portion enters token GEMMs; attention receives prefix=256K.</span></article><article class="card limit"><b>100% SOL latency</b><span>Every nonzero latency contribution in the 24,576 reference estimates is tagged SOL. Zero-time P2P entries do not affect the totals.</span></article><article class="card limit"><b>Additive architecture composites</b><span>GQA/SWA, Kimi MLA/KDA, V4, and GLM DSA context primitives are combined with the identical body. Hybrid checkpoints and kernels do not exist.</span></article><article class="card limit"><b>Explicit resident state</b><span>Memory uses postfill 257K FP8 KV/state, with local layers window-capped, plus AIC weights and common activation/other overhead.</span></article><article class="card limit"><b>Ideal MoE balance</b><span>Each 1K chunk creates enough routes to touch every local expert. AIC assumes even rank balance and does not model hot experts, token correlations, or stragglers.</span></article><article class="card limit"><b>Static batch, no scheduler</b><span>Reported rack throughput assumes a synchronized batch. Continuous arrivals, chunk interleaving, queueing, prefix-cache lookup, and prefill/decode interference are excluded.</span></article><article class="card limit"><b>Projected Rubin</b><span>The Rubin case substitutes 22-TB/s HBM and 1.8-TB/s fabric while retaining GB300 compute and the same 277.55-GiB/rank usable capacity.</span></article><article class="card limit"><b>No quality prediction</b><span>The simulator cannot determine whether aggressive SWA, DSA, KDA, or compressed hybrids preserve capability at 5T scale.</span></article></div></section>

<footer><span>Raw data: <a href="all_results.csv">all_results.csv</a> · <a href="architecture_summary.csv">architecture_summary.csv</a> · <a href="saturation_results.csv">saturation_results.csv</a> · <a href="metadata.json">metadata.json</a></span><span>43 architectures · 88,064 core points · 26,760 direct AIC references including saturation</span></footer>
</main>
<script>
const SUMMARY=__SUMMARY__, CURVES=__CURVES__, META=__META__, BREAKDOWNS=__BREAKDOWNS__, SAT_SUMMARY=__SAT_SUMMARY__, SAT_ROWS=__SAT_ROWS__;
const byId=Object.fromEntries(SUMMARY.map(x=>[x.architecture_id,x]));
const f=(n,d=1)=>Number(n).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d}),fi=n=>Math.round(n).toLocaleString();
const colors={full_gqa:'#738299',gptoss_faithful:'#f4b860',common_r3_w512:'#79a8ff',inkling_faithful:'#69d7e8',swa5_mla:'#ee7da8',swa5_dsa:'#f07972',swa5_kda:'#8fd17f',swa5_v4:'#aa8dff',mla_all:'#bd6c91',dsa2048_all:'#b95b55',kda_all:'#69a85f',v4_all:'#8873d1'};
const labels=Object.fromEntries(CURVES.map(r=>[r.architecture_id,r.label]));

const fastest=[...SUMMARY].sort((a,b)=>a.gb_c1_latency_ms-b.gb_c1_latency_ms)[0],full=byId.full_gqa;
const serviceBest=[...SUMMARY].sort((a,b)=>b.rubin_max_cpr_500ms-a.rubin_max_cpr_500ms||b.gb_max_cpr_500ms-a.gb_max_cpr_500ms)[0];
const memoryBest=[...SUMMARY].sort((a,b)=>b.gb_physical_max_cpr-a.gb_physical_max_cpr)[0];
document.getElementById('kpi-full').textContent=f(full.gb_c1_latency_ms,1)+' ms';
document.getElementById('kpi-fast').textContent=f(fastest.gb_c1_latency_ms,1)+' ms';
document.getElementById('kpi-fast-note').textContent=fastest.label+' · '+f(fastest.gb_c1_input_tps_user,0)+' input tokens/s/request';
document.getElementById('kpi-service').textContent=fi(64*serviceBest.gb_max_cpr_500ms)+' → '+fi(64*serviceBest.rubin_max_cpr_500ms);
document.getElementById('kpi-memory').textContent=fi(memoryBest.gb_physical_max_cpr)+' c/r';
document.getElementById('kpi-memory-note').textContent=memoryBest.label+' · '+fi(64*memoryBest.gb_physical_max_cpr)+' global requests, but '+f(memoryBest.gb_latency_at_physical_max_ms/1000,1)+' s GB latency';

const satById=Object.fromEntries(SAT_SUMMARY.map(x=>[x.architecture_id,x]));
const satIds=['full_gqa','gptoss_faithful','inkling_faithful','swa5_dsa','swa5_kda','swa5_v4'];
document.getElementById('sat-table').innerHTML=satIds.map(id=>{const r=satById[id];return `<tr><td>${r.label}</td><td>${fi(r.gb_chunk_at_95pct)} tokens</td><td>${fi(r.rubin_chunk_at_95pct)} tokens</td><td>${fi(r.gb_cpr_1k_at_95pct)} / ${fi(r.rubin_cpr_1k_at_95pct)}</td></tr>`}).join('');
let satSystem='GB300';const satActive=new Set(['full_gqa','inkling_faithful','swa5_dsa','swa5_v4']);
document.getElementById('sat-legend').innerHTML=satIds.map(id=>`<button data-sat-profile="${id}" class="${satActive.has(id)?'active':''}"><i style="background:${colors[id]}"></i>${satById[id].label}</button>`).join('');
document.querySelectorAll('[data-sat-profile]').forEach(b=>b.onclick=()=>{satActive.has(b.dataset.satProfile)?satActive.delete(b.dataset.satProfile):satActive.add(b.dataset.satProfile);b.classList.toggle('active');drawSaturation()});
document.querySelectorAll('[data-sat-system]').forEach(b=>b.onclick=()=>{satSystem=b.dataset.satSystem;document.querySelectorAll('[data-sat-system]').forEach(x=>x.classList.toggle('active',x===b));drawSaturation()});
function drawSaturation(){const host=document.getElementById('sat-chart'),W=760,H=390,m={l:76,r:20,t:18,b:48},iw=W-m.l-m.r,ih=H-m.t-m.b,rows=SAT_ROWS.filter(r=>r.system===satSystem&&r.cpr===1&&satActive.has(r.architecture_id)),ymax=Math.max(...rows.map(r=>r.rack_input_tps),1)*1.08,x=v=>m.l+Math.log2(v)/12*iw,y=v=>m.t+(1-v/ymax)*ih;let s=`<svg viewBox="0 0 ${W} ${H}" role="img">`;for(let i=0;i<=5;i++){const v=ymax*i/5,py=y(v);s+=`<line class="gridline" x1="${m.l}" y1="${py}" x2="${W-m.r}" y2="${py}"/><text class="axis" x="${m.l-8}" y="${py+4}" text-anchor="end">${f(v/1000,0)}K</text>`}for(const v of [1,4,16,64,256,1024,4096]){const px=x(v);s+=`<line class="gridline" x1="${px}" y1="${m.t}" x2="${px}" y2="${H-m.b}"/><text class="axis" x="${px}" y="${H-18}" text-anchor="middle">${fi(v)}</text>`}for(const id of satActive){const p=rows.filter(r=>r.architecture_id===id).sort((a,b)=>a.new_tokens_per_request-b.new_tokens_per_request);if(!p.length)continue;const d=p.map((r,i)=>`${i?'L':'M'}${x(r.new_tokens_per_request).toFixed(1)},${y(r.rack_input_tps).toFixed(1)}`).join('');s+=`<path d="${d}" fill="none" stroke="${colors[id]}" stroke-width="2.7"/>`;for(const r of p)s+=`<circle cx="${x(r.new_tokens_per_request)}" cy="${y(r.rack_input_tps)}" r="3" fill="${colors[id]}"><title>${r.label} · ${r.new_tokens_per_request} new · ${f(r.rack_input_tps/1000,1)}K tokens/s</title></circle>`}s+=`<text class="axis" x="${m.l+iw/2}" y="${H-2}" text-anchor="middle">new tokens per request (log scale)</text></svg>`;host.innerHTML=s}drawSaturation();
function shapeCpr(id,system,chunk){const rows=SAT_ROWS.filter(r=>r.architecture_id===id&&r.system===system&&r.new_tokens_per_request===chunk).sort((a,b)=>a.cpr-b.cpr),peak=Math.max(...rows.map(r=>r.rack_input_tps));return rows.find(r=>r.rack_input_tps>=.95*peak)?.cpr}
document.getElementById('shape-sat-table').innerHTML=[1,8,32,128,512,1024].map(chunk=>`<tr><td>${fi(chunk)}</td><td>${fi(shapeCpr('full_gqa','GB300',chunk))}</td><td>${fi(shapeCpr('swa5_dsa','GB300',chunk))}</td><td>${fi(shapeCpr('swa5_dsa','VR200 HBM+NVLink',chunk))}</td></tr>`).join('');
function primary(r){const vals=[['Fabric',r.dispatch_share_pct],['MoE tensor compute',r.moe_compute_share_pct],['Attention compute',r.attention_share_pct]];return vals.sort((a,b)=>b[1]-a[1])[0][0]}
const resourceIds=['full_gqa','inkling_faithful','swa5_dsa','swa5_v4'];
document.getElementById('resource-table').innerHTML=resourceIds.flatMap(id=>['GB300','VR200 HBM+NVLink'].map(system=>{const r=SAT_ROWS.find(x=>x.architecture_id===id&&x.system===system&&x.new_tokens_per_request===1024&&x.cpr===1);return `<tr><td>${r.label}</td><td>${system==='GB300'?'GB':'Rubin'}</td><td>${f(r.dispatch_share_pct,1)}%</td><td>${f(r.moe_compute_share_pct,1)}%</td><td>${f(r.attention_share_pct,1)}%</td><td>${primary(r)}</td></tr>`})).join('');

const endpointIds=['full_gqa','gptoss_faithful','inkling_faithful','common_r3_w512','swa5_mla','swa5_dsa','swa5_kda','swa5_v4','v4_all'];
document.getElementById('endpoint-table').innerHTML=endpointIds.map(id=>{const r=byId[id];return `<tr><td>${r.label}</td><td>${f(r.kv_gib_per_request,2)}</td><td>${f(r.gb_c1_latency_ms,1)} ms</td><td>${f(r.gb_c1_input_tps_user,0)}</td><td>${f(r.rubin_c1_latency_ms,1)} ms</td><td>${f(r.rubin_latency_gain_c1,2)}×</td><td>${fi(r.gb_max_cpr_500ms)} / ${fi(r.rubin_max_cpr_500ms)}</td><td>${fi(r.gb_physical_max_cpr)}</td><td>${fi(64*r.gb_physical_max_cpr)}</td><td>${f(r.gb_latency_at_physical_max_ms/1000,2)} / ${f(r.rubin_latency_at_physical_max_ms/1000,2)} s</td></tr>`}).join('');

const hybridIds=['common_r5_w512','swa5_mla','swa5_dsa','swa5_kda','swa5_v4','mla_all','dsa2048_all','kda_all','v4_all'];
document.getElementById('hybrid-table').innerHTML=hybridIds.map(id=>{const r=byId[id];return `<tr><td>${r.label}</td><td>${r.global_mechanism.toUpperCase()}</td><td>${f(r.gb_c1_latency_ms,1)} ms</td><td>${f(r.gb_c4_latency_ms,1)} ms</td><td>${f(r.attention_share_gb_c1,1)}%</td><td>${f(r.gb_best_rack_input_tps/1000,0)}K</td><td>${f(r.rubin_latency_gain_c1,2)}×</td><td>${fi(r.gb_max_cpr_1000ms)} / ${fi(r.rubin_max_cpr_1000ms)}</td><td>${fi(r.gb_physical_max_cpr)}</td></tr>`}).join('');

let system='GB300',maxC=16,metric='extend_latency_ms',xscale='log';
const active=new Set(['full_gqa','gptoss_faithful','inkling_faithful','swa5_dsa','swa5_kda','swa5_v4']);
document.getElementById('curve-legend').innerHTML=Object.keys(colors).map(id=>`<button data-profile="${id}" class="${active.has(id)?'active':''}"><i style="background:${colors[id]}"></i>${labels[id]}</button>`).join('');
document.querySelectorAll('[data-profile]').forEach(b=>b.onclick=()=>{active.has(b.dataset.profile)?active.delete(b.dataset.profile):active.add(b.dataset.profile);b.classList.toggle('active');drawCurve()});
document.querySelectorAll('[data-system]').forEach(b=>b.onclick=()=>{system=b.dataset.system;document.querySelectorAll('[data-system]').forEach(x=>x.classList.toggle('active',x===b));drawCurve()});
document.querySelectorAll('[data-max]').forEach(b=>b.onclick=()=>{maxC=+b.dataset.max;document.querySelectorAll('[data-max]').forEach(x=>x.classList.toggle('active',x===b));drawCurve()});
document.querySelectorAll('[data-metric]').forEach(b=>b.onclick=()=>{metric=b.dataset.metric;document.querySelectorAll('[data-metric]').forEach(x=>x.classList.toggle('active',x===b));drawCurve()});
document.querySelectorAll('[data-xscale]').forEach(b=>b.onclick=()=>{xscale=b.dataset.xscale;document.querySelectorAll('[data-xscale]').forEach(x=>x.classList.toggle('active',x===b));drawCurve()});
function drawCurve(){const host=document.getElementById('curve-chart'),W=1180,H=400,m={l:86,r:24,t:18,b:48},iw=W-m.l-m.r,ih=H-m.t-m.b;const rows=CURVES.filter(r=>r.system===system&&r.cpr<=maxC&&active.has(r.architecture_id));const vals=rows.map(r=>r[metric]),ymax=Math.max(...vals,1)*1.08,ymin=0,x=v=>m.l+(xscale==='log'?Math.log2(v)/Math.log2(maxC):v/maxC)*iw,y=v=>m.t+(1-(v-ymin)/(ymax-ymin))*ih;const title={extend_latency_ms:'1K extend latency',input_tps_user:'Input tokens/s per request',rack_input_tps:'Rack input tokens/s'}[metric];document.getElementById('curve-title').textContent=title+` · ${system==='GB300'?'GB300':'Rubin projection'}`;let s=`<svg viewBox="0 0 ${W} ${H}" role="img">`;for(let i=0;i<=5;i++){const v=ymax*i/5,py=y(v),lab=metric==='extend_latency_ms'?f(v,0)+' ms':f(v/1000,metric==='input_tps_user'?1:0)+'K';s+=`<line class="gridline" x1="${m.l}" y1="${py}" x2="${W-m.r}" y2="${py}"/><text class="axis" x="${m.l-9}" y="${py+4}" text-anchor="end">${lab}</text>`}const ticks=xscale==='log'?[1,2,4,8,16,32,64,128,256,512,1024].filter(v=>v<=maxC):[0,.25,.5,.75,1].map(v=>Math.max(1,Math.round(v*maxC)));for(const v of [...new Set(ticks)]){const px=x(v);s+=`<line class="gridline" x1="${px}" y1="${m.t}" x2="${px}" y2="${H-m.b}"/><text class="axis" x="${px}" y="${H-19}" text-anchor="middle">${fi(v)}</text>`}if(metric==='extend_latency_ms')for(const t of [250,500,1000,2000])if(t<ymax){s+=`<line x1="${m.l}" y1="${y(t)}" x2="${W-m.r}" y2="${y(t)}" stroke="#dbea7d" opacity=".45" stroke-dasharray="4 5"/><text x="${W-m.r-4}" y="${y(t)-5}" text-anchor="end" fill="#dbea7d" font-size="10">${fi(t)} ms</text>`}for(const id of active){const p=rows.filter(r=>r.architecture_id===id).sort((a,b)=>a.cpr-b.cpr);if(!p.length)continue;const cap=byId[id][system==='GB300'?'gb_physical_max_cpr':'rubin_physical_max_cpr'];const before=p.filter(r=>r.cpr<=cap),after=p.filter(r=>r.cpr>=cap);for(const [segment,dash,opacity] of [[before,'',1],[after,'6 5',.42]]){if(segment.length<2)continue;const d=segment.map((r,i)=>`${i?'L':'M'}${x(r.cpr).toFixed(1)},${y(r[metric]).toFixed(1)}`).join('');s+=`<path d="${d}" fill="none" stroke="${colors[id]}" stroke-width="2.5" stroke-dasharray="${dash}" opacity="${opacity}"/>`}if(cap<=maxC){const edge=p.find(r=>r.cpr===cap);if(edge)s+=`<circle cx="${x(edge.cpr)}" cy="${y(edge[metric])}" r="5" fill="${colors[id]}" stroke="#090c13" stroke-width="2"><title>${labels[id]} memory max c/r ${cap}</title></circle>`}}s+=`<text class="axis" x="${m.l+iw/2}" y="${H-2}" text-anchor="middle">concurrency / rank (${xscale} scale)</text></svg>`;host.innerHTML=s}drawCurve();

const heatDefs={gb_c1_latency_ms:['GB300 1K-extend latency at c/r=1',false,v=>f(v,1)+' ms'],gb_c4_latency_ms:['GB300 1K-extend latency at c/r=4',false,v=>f(v,0)+' ms'],gb_max_cpr_500ms:['Largest c/r under 500 ms',true,v=>fi(v)],kv_gib_per_request:['Postfill KV GiB per request',false,v=>f(v,2)]};let heatMetric='gb_c1_latency_ms';
function renderHeat(){const [title,highGood,fmt]=heatDefs[heatMetric];document.getElementById('heat-title').textContent=title;const vals=[];for(const w of [128,512,2048,4096])for(const r of [1,2,3,4,5,7])vals.push(byId[`common_r${r}_w${w}`][heatMetric]);const lo=Math.min(...vals),hi=Math.max(...vals);let h='<table><thead><tr><th>Window ↓ / local:global →</th>'+[1,2,3,4,5,7].map(r=>`<th>${r}:1</th>`).join('')+'</tr></thead><tbody>';for(const w of [128,512,2048,4096]){h+=`<tr><th>W${fi(w)}</th>`;for(const r of [1,2,3,4,5,7]){const v=byId[`common_r${r}_w${w}`][heatMetric],n=(v-lo)/(hi-lo||1),score=highGood?n:1-n,hue=5+score*165;h+=`<td style="background:hsla(${hue},58%,36%,${.18+.5*score})">${fmt(v)}</td>`}h+='</tr>'}h+='</tbody></table>';document.getElementById('heatmap').innerHTML=h}
document.querySelectorAll('[data-heat]').forEach(b=>b.onclick=()=>{heatMetric=b.dataset.heat;document.querySelectorAll('[data-heat]').forEach(x=>x.classList.toggle('active',x===b));renderHeat()});renderHeat();

function drawDecomp(){const ids=['full_gqa','gptoss_faithful','inkling_faithful','swa5_mla','swa5_dsa','swa5_kda','swa5_v4','v4_all'],rows=ids.map(id=>CURVES.find(r=>r.architecture_id===id&&r.system==='GB300'&&r.cpr===1)),W=760,H=390,m={l:190,r:45,t:18,b:38},iw=W-m.l-m.r,rowH=(H-m.t-m.b)/rows.length,max=Math.max(...rows.map(r=>r.body_ms+r.attention_ms))*1.08,x=v=>m.l+v/max*iw;let s=`<svg viewBox="0 0 ${W} ${H}">`;rows.forEach((r,i)=>{const y=m.t+i*rowH+7,h=21;s+=`<text class="axis" x="${m.l-9}" y="${y+15}" text-anchor="end">${r.label.replace(' · W512','')}</text><rect x="${m.l}" y="${y}" width="${x(r.body_ms)-m.l}" height="${h}" rx="4" fill="#40506c"/><rect x="${x(r.body_ms)}" y="${y}" width="${x(r.body_ms+r.attention_ms)-x(r.body_ms)}" height="${h}" rx="4" fill="${colors[r.architecture_id]||'#69d7e8'}"/><text x="${x(r.body_ms+r.attention_ms)+6}" y="${y+15}" fill="#c5cddb" font-size="11">${f(r.body_ms+r.attention_ms,1)}</text>`});s+=`<text class="axis" x="${m.l+iw/2}" y="${H-4}" text-anchor="middle">milliseconds · common body (slate) + attention (color)</text></svg>`;document.getElementById('decomp-chart').innerHTML=s}drawDecomp();
document.getElementById('body-table').innerHTML=['GB300','VR200 HBM+NVLink'].map(()=> '').join('');const gb=BREAKDOWNS.GB300,ru=BREAKDOWNS['VR200 HBM+NVLink'];document.getElementById('body-table').innerHTML=[['MoE dispatch',gb.dispatch_ms,ru.dispatch_ms],['Routed MoE compute',gb.moe_ms,ru.moe_ms],['Other body',gb.body_other_ms,ru.body_other_ms],['Common body total',gb.body_ms,ru.body_ms],['Dense global attention',gb.attention_ms,ru.attention_ms]].map(([l,a,b])=>`<tr><td>${l}</td><td>${f(a,2)} ms</td><td>${f(b,2)} ms</td></tr>`).join('');

const capacityIds=['full_gqa','gptoss_faithful','inkling_faithful','common_r3_w512','swa5_dsa','swa5_kda','swa5_v4','v4_all'];
document.getElementById('capacity-table').innerHTML=capacityIds.map(id=>{const r=byId[id],pair=t=>`${fi(r[`gb_max_cpr_${t}ms`])} / ${fi(r[`rubin_max_cpr_${t}ms`])}`;return `<tr><td>${r.label}</td><td>${pair(250)}</td><td>${pair(500)}</td><td>${pair(1000)}</td><td>${pair(2000)}</td><td>${fi(r.gb_physical_max_cpr)}</td><td>${f(r.gb_latency_at_physical_max_ms/1000,2)} s</td><td>${f(r.rubin_latency_at_physical_max_ms/1000,2)} s</td></tr>`}).join('');

let query='',category='';function renderAll(){const rows=SUMMARY.filter(r=>(!query||r.label.toLowerCase().includes(query))&&(!category||r.category===category)).sort((a,b)=>a.gb_c1_latency_ms-b.gb_c1_latency_ms);document.getElementById('all-table').innerHTML=rows.map(r=>`<tr><td>${r.label}</td><td>${r.category}</td><td>${r.local_layers} / ${r.global_layers}</td><td>${r.global_mechanism.toUpperCase()}</td><td>${r.window?fi(r.window):'—'}</td><td>${f(r.kv_gib_per_request,2)}</td><td>${f(r.gb_c1_latency_ms,1)}</td><td>${f(r.rubin_c1_latency_ms,1)}</td><td>${fi(r.gb_max_cpr_500ms)} / ${fi(r.rubin_max_cpr_500ms)}</td><td>${fi(r.gb_physical_max_cpr)}</td><td>${f(r.attention_share_gb_c1,1)}%</td></tr>`).join('')};document.getElementById('search').oninput=e=>{query=e.target.value.trim().toLowerCase();renderAll()};document.getElementById('category').onchange=e=>{category=e.target.value;renderAll()};renderAll();
</script></body></html>'''

    rendered = template.replace("__SUMMARY__", json.dumps(summaries, separators=(",", ":")))
    rendered = rendered.replace("__CURVES__", json.dumps(curves, separators=(",", ":")))
    rendered = rendered.replace("__META__", json.dumps(metadata, separators=(",", ":")))
    rendered = rendered.replace("__BREAKDOWNS__", json.dumps(breakdowns, separators=(",", ":")))
    rendered = rendered.replace("__SAT_SUMMARY__", json.dumps(saturation_summary, separators=(",", ":")))
    rendered = rendered.replace("__SAT_ROWS__", json.dumps(saturation_rows, separators=(",", ":")))
    DESTINATION.write_text(rendered, encoding="utf-8")
    print(f"Wrote {DESTINATION}")
    print(f"Embedded {len(summaries)} summaries and {len(curves)} curve points")


if __name__ == "__main__":
    main()
