#!/usr/bin/env python3
"""Build a self-contained visual analysis of the unlimited-memory experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
SOURCE = OUTPUT / "balanced_v4_unlimited_memory_curve.csv"
METADATA = OUTPUT / "balanced_v4_unlimited_memory_metadata.json"
DESTINATION = OUTPUT / "BLACKWELL_RUBIN_CONCURRENCY_ANALYSIS.html"


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        for source in csv.DictReader(stream):
            rows.append({
                "system": source["system"],
                "cpr": int(source["cpr"]),
                "global": int(source["global_concurrency"]),
                "tpot": float(source["sol_tpot_ms"]),
                "tpsUser": float(source["sol_tps_user"]),
                "rackTps": float(source["aggregate_tps_dep64"]),
                "memory": float(source["modeled_memory_gib_if_available"]),
                "nextn": int(source["best_nextn"]),
                "beyondCapacity": source["beyond_physical_capacity"] == "True",
            })
    return rows


def main() -> None:
    data = load_rows()
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    template = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>When More Memory Stops Buying Throughput</title>
  <style>
    :root {
      --ink: #eef5f7;
      --muted: #9fb1b7;
      --faint: #64777d;
      --panel: #101b1f;
      --panel-2: #142328;
      --line: #294047;
      --paper: #071014;
      --blackwell: #ff9b56;
      --rubin: #56d6d0;
      --capacity: #e7ca73;
      --good: #a5e483;
      --danger: #f18b85;
      --shadow: 0 18px 60px rgba(0,0,0,.26);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 92% 2%, rgba(86,214,208,.14), transparent 31rem),
        radial-gradient(circle at 4% 18%, rgba(255,155,86,.10), transparent 30rem),
        var(--paper);
      font: 15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .page { width: min(1420px, calc(100% - 34px)); margin: 0 auto; padding: 28px 0 72px; }
    .hero {
      position: relative;
      overflow: hidden;
      padding: clamp(30px, 5vw, 68px);
      border: 1px solid var(--line);
      border-radius: 26px;
      background: linear-gradient(145deg, rgba(20,35,40,.97), rgba(9,21,25,.92));
      box-shadow: var(--shadow);
    }
    .hero::after {
      content: "";
      position: absolute;
      width: 440px; height: 440px; right: -170px; top: -215px;
      border: 1px solid rgba(86,214,208,.28); border-radius: 50%;
      box-shadow: 0 0 0 48px rgba(86,214,208,.025), 0 0 0 96px rgba(86,214,208,.018);
    }
    .eyebrow { color: var(--rubin); text-transform: uppercase; letter-spacing: .14em; font-weight: 750; font-size: 12px; }
    h1 { max-width: 900px; margin: 14px 0 16px; font-size: clamp(39px, 6vw, 78px); line-height: .98; letter-spacing: -.055em; }
    .lede { max-width: 860px; margin: 0; color: #bfd0d5; font-size: clamp(17px, 2vw, 22px); }
    .chips { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 28px; }
    .chip { padding: 7px 11px; border-radius: 99px; border: 1px solid var(--line); background: #0b171b; color: #b7c6ca; font-size: 12px; }
    .section { margin-top: 62px; }
    .section-head { display: grid; grid-template-columns: minmax(0, 1fr) minmax(260px, 560px); gap: 30px; align-items: end; margin-bottom: 22px; }
    .kicker { color: var(--capacity); text-transform: uppercase; letter-spacing: .12em; font-size: 11px; font-weight: 800; }
    h2 { margin: 7px 0 0; font-size: clamp(28px, 4vw, 48px); line-height: 1.05; letter-spacing: -.035em; }
    h3 { margin: 0; font-size: 18px; letter-spacing: -.015em; }
    p { margin: 0; }
    .section-copy { color: var(--muted); font-size: 16px; }
    .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 13px; margin-top: 18px; }
    .kpi { min-height: 164px; padding: 20px; border: 1px solid var(--line); border-radius: 18px; background: rgba(16,27,31,.9); }
    .kpi .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .kpi .value { margin: 13px 0 7px; font-size: clamp(28px, 3.2vw, 43px); line-height: 1; font-variant-numeric: tabular-nums; letter-spacing: -.045em; }
    .kpi .note { color: var(--faint); font-size: 12px; }
    .bw { color: var(--blackwell); }
    .rb { color: var(--rubin); }
    .cap { color: var(--capacity); }
    .card { border: 1px solid var(--line); border-radius: 20px; background: rgba(16,27,31,.94); box-shadow: 0 12px 40px rgba(0,0,0,.14); }
    .controls { display: flex; flex-wrap: wrap; gap: 7px; }
    button {
      cursor: pointer; border: 1px solid var(--line); border-radius: 99px; padding: 8px 13px;
      color: var(--muted); background: #0c181c; font: inherit; font-size: 12px;
    }
    button:hover, button.active { color: var(--ink); background: #20343a; border-color: #45636b; }
    .chart-card { padding: 18px 18px 10px; position: relative; overflow: hidden; }
    .chart-title { display: flex; justify-content: space-between; gap: 20px; align-items: baseline; padding: 2px 4px 8px; }
    .chart-title span { color: var(--faint); font-size: 12px; }
    .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .chart-wide { grid-column: 1 / -1; }
    .chart { min-height: 310px; }
    svg { display: block; width: 100%; height: auto; overflow: visible; }
    .axis { fill: var(--faint); font-size: 11px; }
    .grid { stroke: #22363c; stroke-width: 1; }
    .capacity-zone { fill: rgba(231,202,115,.055); }
    .capacity-line { stroke: var(--capacity); stroke-width: 1.2; stroke-dasharray: 5 5; }
    .selected-line { stroke: rgba(238,245,247,.52); stroke-width: 1; stroke-dasharray: 2 4; }
    .legend { display: flex; flex-wrap: wrap; gap: 16px; padding: 4px 5px 9px; color: var(--muted); font-size: 12px; }
    .legend i { display: inline-block; width: 18px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
    .scrubber { padding: 24px; margin-top: 14px; }
    .scrubber-head { display: flex; justify-content: space-between; align-items: baseline; gap: 22px; }
    .scrubber-value { font-size: 23px; font-weight: 750; font-variant-numeric: tabular-nums; }
    input[type="range"] { width: 100%; margin: 23px 0 13px; accent-color: var(--rubin); }
    .presets { display: flex; gap: 7px; flex-wrap: wrap; }
    .readout { display: grid; grid-template-columns: 1.15fr 1fr 1fr; gap: 12px; margin-top: 17px; }
    .readout-cell { padding: 17px; border-radius: 14px; background: #0b171b; border: 1px solid #24393f; }
    .readout-cell small { color: var(--faint); text-transform: uppercase; letter-spacing: .07em; }
    .readout-cell strong { display: block; margin-top: 7px; font-size: 22px; }
    .mini-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 11px; }
    .mini-grid div { color: var(--muted); font-size: 12px; }
    .mini-grid b { display: block; color: var(--ink); font-size: 17px; font-variant-numeric: tabular-nums; }
    .status { display: inline-flex; align-items: center; gap: 7px; color: var(--good); font-size: 12px; }
    .status::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
    .status.over { color: var(--danger); }
    .explain-grid { display: grid; grid-template-columns: 1.05fr .95fr; gap: 14px; }
    .prose { padding: clamp(22px, 4vw, 38px); }
    .prose p + p { margin-top: 15px; }
    .equation {
      margin: 22px 0; padding: 22px; border-left: 3px solid var(--rubin); background: #0a161a;
      font: 600 clamp(17px, 2.5vw, 25px)/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .equation small { display: block; color: var(--muted); font: 12px/1.5 Inter, sans-serif; margin-top: 10px; }
    .flow { padding: 28px; display: grid; align-content: center; gap: 10px; }
    .flow-row { display: grid; grid-template-columns: 1fr 34px 1fr 34px 1fr; align-items: center; text-align: center; }
    .flow-box { min-height: 108px; display: grid; place-content: center; padding: 15px; border: 1px solid var(--line); border-radius: 14px; background: #0b171b; }
    .flow-box b { display: block; font-size: 23px; }
    .arrow { color: var(--capacity); font-size: 21px; }
    .flow-note { color: var(--muted); text-align: center; font-size: 13px; }
    .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 18px; }
    table { width: 100%; border-collapse: collapse; min-width: 760px; background: rgba(16,27,31,.94); font-variant-numeric: tabular-nums; }
    th, td { padding: 13px 16px; text-align: right; border-bottom: 1px solid #22363c; }
    th:first-child, td:first-child { text-align: left; }
    th { position: sticky; top: 0; color: var(--muted); background: #132329; font-size: 11px; text-transform: uppercase; letter-spacing: .07em; }
    tbody tr:hover { background: rgba(86,214,208,.04); }
    tbody tr:last-child td { border-bottom: 0; }
    .bar-list { display: grid; gap: 15px; padding: 28px; }
    .bar-row { display: grid; grid-template-columns: 90px 1fr 74px; gap: 12px; align-items: center; }
    .bar-track { height: 10px; background: #22353b; border-radius: 99px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--blackwell), var(--rubin)); }
    .insights { display: grid; grid-template-columns: repeat(3, 1fr); gap: 13px; }
    .insight { padding: 24px; min-height: 230px; }
    .insight .num { color: var(--capacity); font: 800 12px ui-monospace, monospace; }
    .insight h3 { margin: 13px 0 10px; font-size: 21px; }
    .insight p { color: var(--muted); }
    .callout { margin-top: 14px; padding: 22px 24px; border-radius: 17px; border: 1px solid rgba(231,202,115,.35); background: rgba(231,202,115,.055); color: #d9d1b1; }
    .limits { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
    .limit { padding: 20px; }
    .limit b { display: block; margin-bottom: 6px; }
    .limit span { color: var(--muted); }
    footer { margin-top: 58px; padding-top: 22px; border-top: 1px solid var(--line); color: var(--faint); font-size: 12px; display: flex; justify-content: space-between; gap: 25px; }
    code { color: #c8dbdf; }
    @media (max-width: 1000px) {
      .kpis { grid-template-columns: 1fr 1fr; }
      .charts, .explain-grid { grid-template-columns: 1fr; }
      .chart-wide { grid-column: auto; }
      .readout { grid-template-columns: 1fr; }
      .insights { grid-template-columns: 1fr 1fr; }
      .section-head { grid-template-columns: 1fr; }
    }
    @media (max-width: 650px) {
      .page { width: min(100% - 18px, 1420px); padding-top: 9px; }
      .hero { border-radius: 18px; }
      .kpis, .insights, .limits { grid-template-columns: 1fr; }
      .chart { min-height: 250px; }
      .flow-row { grid-template-columns: 1fr; gap: 8px; }
      .arrow { transform: rotate(90deg); }
      footer { display: block; }
    }
    @media print {
      body { background: #fff; color: #111; }
      .page { width: 100%; }
      .hero, .card, table { box-shadow: none; break-inside: avoid; }
      button, input[type="range"] { display: none; }
    }
  </style>
</head>
<body>
<main class="page">
  <header class="hero">
    <div class="eyebrow">AIC speed-of-light decode study · September 2026</div>
    <h1>When more memory stops buying throughput</h1>
    <p class="lede">A controlled Blackwell-versus-Rubin thought experiment for a balanced 5T V4-style model. The rack can hold more resident agents if memory capacity is relaxed—but each agent receives a thinner share of an eventually fixed decode budget.</p>
    <div class="chips">
      <span class="chip">DEP64</span><span class="chip">256K resident context/request</span>
      <span class="chip">decode only</span><span class="chip">raw AIC SOL</span>
      <span class="chip">no prefill</span><span class="chip">same modeled 277.5-GiB/rank admission limit</span>
    </div>
  </header>

  <section class="section">
    <div class="section-head">
      <div><div class="kicker">The result in four numbers</div><h2>Capacity is not throughput</h2></div>
      <p class="section-copy">Rubin raises the useful throughput ceiling, but neither system turns unlimited memory into unlimited token generation. The service-level target determines how many of the resident agents are actually useful.</p>
    </div>
    <div class="kpis">
      <article class="kpi"><div class="label">Physical model boundary</div><div class="value cap">191 c/r</div><div class="note">12,224 global requests at DEP64 · 276.8 GiB/rank modeled use</div></article>
      <article class="kpi"><div class="label">100 TPS/user concurrency lift</div><div class="value rb">2.26×</div><div class="note">GB300 35 c/r → Rubin 79 c/r</div></article>
      <article class="kpi"><div class="label">50 TPS/user concurrency lift</div><div class="value rb">1.83×</div><div class="note">GB300 87 c/r → Rubin 159 c/r</div></article>
      <article class="kpi"><div class="label">Asymptotic rack throughput lift</div><div class="value rb">1.55×</div><div class="note">≈330K TPS → ≈511K TPS at DEP64</div></article>
    </div>
  </section>

  <section class="section" id="curves">
    <div class="section-head">
      <div><div class="kicker">One experiment · three views</div><h2>Latency rises while throughput flattens</h2></div>
      <div class="controls" aria-label="Chart range">
        <button data-range="256">Capacity detail</button>
        <button data-range="512">Early plateau</button>
        <button data-range="2048" class="active">Full counterfactual</button>
      </div>
    </div>
    <div class="charts">
      <article class="card chart-card chart-wide">
        <div class="chart-title"><h3>SOL time per output token</h3><span>lower is better · linear scale</span></div>
        <div class="legend"><span><i style="background:var(--blackwell)"></i>GB300 (Blackwell)</span><span><i style="background:var(--rubin)"></i>Rubin HBM + fabric</span><span><i style="background:var(--capacity);height:1px"></i>physical capacity</span></div>
        <div class="chart" id="chart-tpot"></div>
      </article>
      <article class="card chart-card">
        <div class="chart-title"><h3>Per-user generation rate</h3><span>higher is better · logarithmic scale</span></div>
        <div class="chart" id="chart-user"></div>
      </article>
      <article class="card chart-card">
        <div class="chart-title"><h3>Aggregate DEP64 throughput</h3><span>higher is better · linear scale</span></div>
        <div class="chart" id="chart-rack"></div>
      </article>
    </div>

    <article class="card scrubber">
      <div class="scrubber-head"><div><h3>Inspect one concurrency point</h3><span class="status" id="capacity-status">Inside physical memory</span></div><div class="scrubber-value"><span id="selected-cpr">191</span> c/r</div></div>
      <input id="concurrency" type="range" min="1" max="2048" value="191" step="1" aria-label="Concurrency per rank">
      <div class="presets">
        <button data-cpr="35">GB 100 TPS</button><button data-cpr="79">Rubin 100 TPS</button>
        <button data-cpr="87">GB 50 TPS</button><button data-cpr="159">Rubin 50 TPS</button>
        <button data-cpr="191">Memory full</button><button data-cpr="512">512 c/r</button><button data-cpr="2048">2,048 c/r</button>
      </div>
      <div class="readout">
        <div class="readout-cell"><small>Workload at selected point</small><strong id="global-requests">12,224 agents</strong><div class="mini-grid"><div>Memory/rank<b id="memory">276.8 GiB</b></div><div>Rubin speedup<b id="speedup">1.65×</b></div><div>Spec depth<b id="spec">n0 / n0</b></div></div></div>
        <div class="readout-cell"><small class="bw">GB300 · Blackwell</small><strong id="gb-user">25.3 TPS/user</strong><div class="mini-grid"><div>TPOT<b id="gb-tpot">39.5 ms</b></div><div>Rack TPS<b id="gb-rack">309K</b></div><div>Share/user<b id="gb-share">—</b></div></div></div>
        <div class="readout-cell"><small class="rb">Rubin HBM + fabric</small><strong id="rb-user">41.8 TPS/user</strong><div class="mini-grid"><div>TPOT<b id="rb-tpot">23.9 ms</b></div><div>Rack TPS<b id="rb-rack">511K</b></div><div>Share/user<b id="rb-share">—</b></div></div></div>
      </div>
    </article>
  </section>

  <section class="section">
    <div class="section-head">
      <div><div class="kicker">Mechanism</div><h2>The plateau is mathematically expected</h2></div>
      <p class="section-copy">Once the token-proportional term dominates a decode step, doubling the batch doubles both useful tokens produced and elapsed step time. Those factors cancel in aggregate throughput.</p>
    </div>
    <div class="explain-grid">
      <article class="card prose">
        <p>In the high-concurrency region, the model is well approximated by a fixed component plus a component proportional to concurrency <em>C</em>:</p>
        <div class="equation">TPOT(C) ≈ a + bC<br>rack TPS(C) ≈ 64,000C / (a + bC)<small>Therefore rack TPS → 64,000 / b as C becomes large.</small></div>
        <p>The fitted high-concurrency slope is <span class="bw">0.194 ms/c-r on GB300</span> and <span class="rb">0.125 ms/c-r on Rubin</span>. This produces fitted ceilings of approximately 330K and 511K rack TPS.</p>
        <p>More capacity still increases the count of resident requests. It simply cannot prevent their individual generation rates from approaching zero as the fixed rack budget is shared more finely.</p>
      </article>
      <article class="card flow">
        <div class="flow-row">
          <div class="flow-box"><span>Concurrency</span><b>4×</b><small>512 → 2,048 c/r</small></div><div class="arrow">→</div>
          <div class="flow-box"><span>Per-user TPS</span><b>≈¼×</b><small>both systems</small></div><div class="arrow">→</div>
          <div class="flow-box"><span>Rack TPS</span><b>≈1×</b><small>already saturated</small></div>
        </div>
        <div class="flow-note">At DEP64 this changes resident requests from 32,768 to 131,072, but Rubin stays at ≈511K aggregate TPS and GB300 moves only from 325K to 329K.</div>
      </article>
    </div>
  </section>

  <section class="section">
    <div class="section-head">
      <div><div class="kicker">Concrete checkpoints</div><h2>What an agent actually experiences</h2></div>
      <p class="section-copy">Selected raw-SOL points make the trade-off tangible. Values beyond 191 c/r intentionally ignore physical memory admission.</p>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>Concurrency/rank</th><th>Global requests</th><th>Memory/rank</th><th>GB TPS/user</th><th>Rubin TPS/user</th><th>GB rack TPS</th><th>Rubin rack TPS</th><th>Rubin latency gain</th></tr></thead>
      <tbody id="checkpoint-body"></tbody>
    </table></div>
    <div class="callout"><b>The sharpest illustration:</b> from c/r 512 to 2,048, the request count grows 4×. GB300 per-user rate falls from 9.9 to 2.5 TPS while rack throughput rises only 1.1%. Rubin per-user rate falls from 15.6 to 3.9 TPS while rack throughput changes by less than 0.1%.</div>
  </section>

  <section class="section">
    <div class="section-head">
      <div><div class="kicker">Diminishing returns</div><h2>Rubin saturates earlier—and higher</h2></div>
      <p class="section-copy">The bars show how much of each fitted asymptotic throughput ceiling has already been reached. Reaching the ceiling is good for utilization, but it also means extra concurrency mostly becomes extra latency.</p>
    </div>
    <div class="explain-grid">
      <article class="card"><div class="bar-list" id="saturation-bars"></div></article>
      <article class="card prose">
        <p>Rubin is already at about <b>96% of its fitted ceiling by c/r 64</b> and effectively saturated by c/r 128. GB300 is around 82% at c/r 64 and approaches its lower ceiling more gradually.</p>
        <p>This is why Rubin can support many more agents at 50 or 100 TPS/user, but cannot support arbitrarily many at those rates. After saturation, each doubling of concurrency produces close to a halving of TPS/user.</p>
        <p>The Rubin/GB latency ratio also compresses—from 2.65× at c/r 1 to 1.56× at c/r 2,048—because the extreme-batch slope includes token-proportional work that is not accelerated by the nominal HBM-bandwidth ratio alone.</p>
      </article>
    </div>
  </section>

  <section class="section">
    <div class="section-head">
      <div><div class="kicker">Interpretation</div><h2>Five conclusions that survive the thought experiment</h2></div>
      <p class="section-copy">These conclusions are about the model’s structure and the shape of the SOL envelope. Production values require an efficiency discount and serving-system validation.</p>
    </div>
    <div class="insights">
      <article class="card insight"><span class="num">01</span><h3>“Maximum concurrency” needs an SLO</h3><p>Memory-full concurrency is 191 c/r here, but the useful boundary is only 35/79 c/r at 100 TPS/user or 87/159 c/r at 50 TPS/user for GB300/Rubin.</p></article>
      <article class="card insight"><span class="num">02</span><h3>Capacity-only scaling has weak late returns</h3><p>Past 191 c/r, extra memory admits more 256K contexts but adds little rack throughput. It is a resident-state benefit, not a proportional serving-capacity benefit.</p></article>
      <article class="card insight"><span class="num">03</span><h3>Bandwidth raises the ceiling</h3><p>Rubin’s meaningful advantage is its higher throughput envelope: about 511K versus 330K modeled rack TPS. This permits more concurrent agents at the same per-user rate.</p></article>
      <article class="card insight"><span class="num">04</span><h3>Equal-concurrency speedup is not constant</h3><p>The gain is largest when fixed weight-streaming pressure dominates. It shrinks at extreme batches as token-proportional compute, KV/attention work, and communication determine the slope.</p></article>
      <article class="card insight"><span class="num">05</span><h3>“Fill memory” is not a universal policy</h3><p>At memory-full c/r 191, Rubin provides 41.8 SOL TPS/user and GB300 25.3. Whether either is reasonable depends on the agent’s responsiveness target.</p></article>
      <article class="card insight"><span class="num">06</span><h3>Rubin is close to full utilization earlier</h3><p>For throughput maximization, Rubin needs far less than its full resident capacity. The remaining memory headroom is useful for context residency, burst absorption, or isolation—not more steady-state tokens.</p></article>
    </div>
  </section>

  <section class="section">
    <div class="section-head">
      <div><div class="kicker">Boundaries of the claim</div><h2>What this artifact does—and does not—establish</h2></div>
      <p class="section-copy">Treat the exact values as comparative SOL projections. Treat the plateau mechanism and the distinction between capacity and serviceable concurrency as the more robust conclusions.</p>
    </div>
    <div class="limits">
      <article class="card limit"><b>Raw speed of light</b><span>No framework overhead, queueing, scheduler loss, prefill interference, kernel gaps, or production efficiency multiplier is applied.</span></article>
      <article class="card limit"><b>Decode-only resident context</b><span>Every request is modeled with a fully resident 256K context. Prompt ingestion and cache creation are outside scope.</span></article>
      <article class="card limit"><b>Controlled hardware projection</b><span>The Rubin profile changes projected HBM bandwidth and fabric while retaining the controlled capacity assumption used in this study.</span></article>
      <article class="card limit"><b>Counterfactual admission</b><span>Points above 191 c/r require more memory than modeled hardware provides—2,284 GiB/rank by c/r 2,048—and use no speculative decoding.</span></article>
      <article class="card limit"><b>Speculation inside capacity</b><span>The fastest memory-fitting depth uses the shared DSpark acceptance-length assumption. At higher concurrency the selected optimum naturally falls back to n0.</span></article>
      <article class="card limit"><b>Model-specific result</b><span>This is the balanced 5T V4-style candidate at DEP64. Different active parameters, depth, attention sparsity, or sharding can change both slope and ceiling.</span></article>
    </div>
  </section>

  <footer><span>Generated from <code>balanced_v4_unlimited_memory_curve.csv</code> and its fitted high-concurrency metadata.</span><span>Balanced 5T V4 · DEP64 · 256K · AIC SOL</span></footer>
</main>

<script>
const DATA = __DATA__;
const META = __META__;
const COLORS = {"GB300":"#ff9b56", "VR200 HBM+NVLink":"#56d6d0"};
const SYSTEMS = ["GB300", "VR200 HBM+NVLink"];
const bySystem = Object.fromEntries(SYSTEMS.map(s => [s, DATA.filter(d => d.system === s).sort((a,b) => a.cpr-b.cpr)]));
const physicalMax = META.physical_max_cpr.GB300;
let maxX = 2048;
let selectedCpr = 191;

const fmt = (v, digits=1) => Number(v).toLocaleString(undefined, {maximumFractionDigits: digits, minimumFractionDigits: digits});
const fmtInt = v => Math.round(v).toLocaleString();
const fmtK = v => v >= 1e6 ? `${fmt(v/1e6,2)}M` : `${fmt(v/1000, v < 100000 ? 1 : 0)}K`;
function nearestRow(system, cpr) {
  return bySystem[system].reduce((best, row) => Math.abs(row.cpr-cpr) < Math.abs(best.cpr-cpr) ? row : best);
}
function niceTicks(max, count=5) {
  const raw = max / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const n = raw / power;
  const step = (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * power;
  const ticks = [];
  for (let x=0; x<=max; x+=step) ticks.push(x);
  return ticks;
}
function linePath(points, x, y) { return points.map((p,i) => `${i?'L':'M'}${x(p.cpr).toFixed(2)},${y(p).toFixed(2)}`).join(' '); }

function renderChart(id, metric, options={}) {
  const host = document.getElementById(id);
  const W=900, H=310, m={l:64,r:18,t:12,b:40}, iw=W-m.l-m.r, ih=H-m.t-m.b;
  const visible = Object.fromEntries(SYSTEMS.map(s => [s, bySystem[s].filter(d => d.cpr <= maxX)]));
  const all = SYSTEMS.flatMap(s => visible[s]);
  const x = v => m.l + (v/maxX)*iw;
  const log = options.log;
  let y, ticks;
  if (log) {
    const values = all.map(d => d[metric]);
    const lo = Math.max(.9, Math.min(...values)*.88), hi = Math.max(...values)*1.14;
    const logLo=Math.log10(lo), logHi=Math.log10(hi);
    y = d => m.t + (logHi-Math.log10(d[metric]))/(logHi-logLo)*ih;
    ticks = [1,2,5,10,20,50,100,200,500,1000].filter(t => t>=lo && t<=hi);
  } else {
    const high = Math.max(...all.map(d=>d[metric]))*1.08;
    y = d => m.t + (1-d[metric]/high)*ih;
    ticks = niceTicks(high, 5);
  }
  const yValue = value => {
    if (log) {
      const values = all.map(d => d[metric]);
      const lo = Math.max(.9, Math.min(...values)*.88), hi = Math.max(...values)*1.14;
      return m.t + (Math.log10(hi)-Math.log10(value))/(Math.log10(hi)-Math.log10(lo))*ih;
    }
    const high = Math.max(...all.map(d=>d[metric]))*1.08;
    return m.t + (1-value/high)*ih;
  };
  const xTicks = maxX===256 ? [0,64,128,191,256] : maxX===512 ? [0,128,256,384,512] : [0,256,512,1024,1536,2048];
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${options.label}">`;
  if (physicalMax < maxX) svg += `<rect class="capacity-zone" x="${x(physicalMax)}" y="${m.t}" width="${W-m.r-x(physicalMax)}" height="${ih}"/>`;
  ticks.forEach(t => {
    const py=yValue(t); const label=metric==='rackTps'?fmtK(t):fmt(t, t<10?1:0);
    svg += `<line class="grid" x1="${m.l}" y1="${py}" x2="${W-m.r}" y2="${py}"/><text class="axis" x="${m.l-9}" y="${py+4}" text-anchor="end">${label}</text>`;
  });
  xTicks.forEach(t => { const px=x(t); svg += `<line class="grid" x1="${px}" y1="${m.t}" x2="${px}" y2="${H-m.b}"/><text class="axis" x="${px}" y="${H-15}" text-anchor="middle">${fmtInt(t)}</text>`; });
  svg += `<line class="capacity-line" x1="${x(physicalMax)}" y1="${m.t}" x2="${x(physicalMax)}" y2="${H-m.b}"/>`;
  if (selectedCpr <= maxX) svg += `<line class="selected-line" x1="${x(selectedCpr)}" y1="${m.t}" x2="${x(selectedCpr)}" y2="${H-m.b}"/>`;
  SYSTEMS.forEach(s => {
    svg += `<path d="${linePath(visible[s],x,y)}" fill="none" stroke="${COLORS[s]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>`;
    const point=nearestRow(s,Math.min(selectedCpr,maxX));
    if(selectedCpr<=maxX) svg += `<circle cx="${x(point.cpr)}" cy="${y(point)}" r="5" fill="${COLORS[s]}" stroke="#071014" stroke-width="2"/>`;
  });
  svg += `<text class="axis" x="${m.l+iw/2}" y="${H-1}" text-anchor="middle">concurrency / rank</text>`;
  if (physicalMax < maxX) svg += `<text x="${x(physicalMax)+8}" y="${m.t+14}" fill="#e7ca73" font-size="11">counterfactual beyond memory</text>`;
  svg += `</svg>`;
  host.innerHTML=svg;
}

function renderCharts() {
  renderChart('chart-tpot','tpot',{label:'Time per output token versus concurrency per rank'});
  renderChart('chart-user','tpsUser',{label:'Tokens per second per user versus concurrency per rank',log:true});
  renderChart('chart-rack','rackTps',{label:'Aggregate rack throughput versus concurrency per rank'});
}

function setSelected(cpr) {
  selectedCpr = Number(cpr);
  const gb=nearestRow('GB300',selectedCpr), rb=nearestRow('VR200 HBM+NVLink',selectedCpr);
  selectedCpr=gb.cpr;
  document.getElementById('concurrency').value=selectedCpr;
  document.getElementById('selected-cpr').textContent=fmtInt(selectedCpr);
  document.getElementById('global-requests').textContent=`${fmtInt(gb.global)} agents`;
  document.getElementById('memory').textContent=`${fmt(gb.memory,1)} GiB`;
  document.getElementById('speedup').textContent=`${fmt(gb.tpot/rb.tpot,2)}×`;
  document.getElementById('spec').textContent=`n${gb.nextn} / n${rb.nextn}`;
  document.getElementById('gb-user').textContent=`${fmt(gb.tpsUser,1)} TPS/user`;
  document.getElementById('rb-user').textContent=`${fmt(rb.tpsUser,1)} TPS/user`;
  document.getElementById('gb-tpot').textContent=`${fmt(gb.tpot,1)} ms`;
  document.getElementById('rb-tpot').textContent=`${fmt(rb.tpot,1)} ms`;
  document.getElementById('gb-rack').textContent=fmtK(gb.rackTps);
  document.getElementById('rb-rack').textContent=fmtK(rb.rackTps);
  document.getElementById('gb-share').textContent=`${fmt(100*gb.rackTps/META.high_cpr_fits.GB300.asymptotic_dep64_tps,1)}% ceiling`;
  document.getElementById('rb-share').textContent=`${fmt(100*rb.rackTps/META.high_cpr_fits['VR200 HBM+NVLink'].asymptotic_dep64_tps,1)}% ceiling`;
  const status=document.getElementById('capacity-status');
  status.textContent=selectedCpr<=physicalMax?'Inside physical memory':'Counterfactual: exceeds memory';
  status.classList.toggle('over',selectedCpr>physicalMax);
  renderCharts();
}

document.querySelectorAll('[data-range]').forEach(button => button.addEventListener('click', () => {
  maxX=Number(button.dataset.range);
  document.querySelectorAll('[data-range]').forEach(b=>b.classList.toggle('active',b===button));
  renderCharts();
}));
document.getElementById('concurrency').addEventListener('input', e => setSelected(e.target.value));
document.querySelectorAll('[data-cpr]').forEach(button => button.addEventListener('click', () => setSelected(button.dataset.cpr)));

const checkpointCprs=[1,16,35,64,79,87,128,159,191,512,2048];
document.getElementById('checkpoint-body').innerHTML=checkpointCprs.map(c=>{
  const g=nearestRow('GB300',c), r=nearestRow('VR200 HBM+NVLink',c);
  const marker=c>physicalMax?' <span style="color:var(--danger)">†</span>':'';
  return `<tr><td>${fmtInt(c)}${marker}</td><td>${fmtInt(g.global)}</td><td>${fmt(g.memory,1)} GiB</td><td>${fmt(g.tpsUser,1)}</td><td>${fmt(r.tpsUser,1)}</td><td>${fmtK(g.rackTps)}</td><td>${fmtK(r.rackTps)}</td><td>${fmt(g.tpot/r.tpot,2)}×</td></tr>`;
}).join('');

const asym={GB300:META.high_cpr_fits.GB300.asymptotic_dep64_tps,Rubin:META.high_cpr_fits['VR200 HBM+NVLink'].asymptotic_dep64_tps};
const saturationCprs=[16,64,128,191,512,2048];
document.getElementById('saturation-bars').innerHTML=saturationCprs.map(c=>{
  const g=nearestRow('GB300',c), r=nearestRow('VR200 HBM+NVLink',c);
  const gp=100*g.rackTps/asym.GB300, rp=100*r.rackTps/asym.Rubin;
  return `<div><div class="bar-row"><b>c/r ${fmtInt(c)}</b><div><div class="bar-track"><div class="bar-fill" style="width:${Math.min(gp,100)}%;background:var(--blackwell)"></div></div><div class="bar-track" style="margin-top:5px"><div class="bar-fill" style="width:${Math.min(rp,100)}%;background:var(--rubin)"></div></div></div><span>${fmt(gp,0)}% / ${fmt(rp,0)}%</span></div></div>`;
}).join('') + `<div style="color:var(--faint);font-size:12px"><span class="bw">upper bar: GB300</span> · <span class="rb">lower bar: Rubin</span> · percentage of fitted throughput ceiling</div>`;

setSelected(191);
</script>
</body>
</html>
'''
    rendered = template.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    rendered = rendered.replace("__META__", json.dumps(metadata, separators=(",", ":")))
    DESTINATION.write_text(rendered, encoding="utf-8")
    print(f"Wrote {DESTINATION}")
    print(f"Embedded {len(data)} modeled points")


if __name__ == "__main__":
    main()
