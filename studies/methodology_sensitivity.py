"""Exact methodology sensitivity on the frozen four-model release; standard library only."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent_index.index import capacity

REFERENCE = "GB300 NVL72"
BASELINE = "log_50_400"
NAMES = {
    "GLM-5.2": "GLM-5.2", "Kimi-K3": "Kimi-K3",
    "Qwen-480B": "Qwen3-Coder 480B",
    "MAI2-informed-h128-w512-sh4352": "MAI-Base-2 2.8T proxy",
}


def combined_capacity(points, speed, prefill, ratio):
    c = capacity(points, speed)
    if ratio == 0:
        return c
    return c / (1 + ratio * speed * c / prefill) if c and prefill > 0 else 0.


def area(points, prefill, ratio, low, high, axis):
    """Exact integral of C/(1+r*s*C/P) against ds or ds/s, at every saved edge."""
    if not 0 < low < high or ratio < 0 or axis not in ("log", "linear"):
        raise ValueError("Invalid integration scenario")
    if ratio and prefill <= 0:
        return 0.
    edges = sorted({low, high, *(p["speed"] for p in points if low < p["speed"] < high)})
    terms = []
    for a, b in zip(edges, edges[1:]):
        c = capacity(points, (a+b)/2)
        if not c:
            continue
        k = ratio*c/prefill if ratio else 0.
        if axis == "log":
            terms.append(c*math.log1p((b-a)/(a*(1+k*b))))
        else:
            terms.append(c*math.log1p(k*(b-a)/(1+k*a))/k if k else c*(b-a))
    return math.fsum(terms)


def scenarios(models, ratio):
    equal = {m: 1/len(models) for m in models}
    def case(key, family, label, **overrides):
        return dict(id=key, family=family, label=label, axis="log", low=50.,
                    high=400., target=None, ratio=ratio, weights=equal.copy()) | overrides
    out = [case(BASELINE, "axis_band", "Headline: log 50–400")]
    for axis in ("log", "linear"):
        for high in (200, 300, 400, 800):
            key = f"{axis}_50_{high}"
            if key != BASELINE:
                out.append(case(key, "axis_band", f"{axis.title()} 50–{high}", axis=axis, high=float(high)))
    out.append(case("glm_kimi", "basket", "GLM + Kimi", weights={m: .5 for m in models[:2]}))
    for i, m in enumerate(models):
        out.append(case(f"only_{i}", "single_model", NAMES[m] + " only", weights={m: 1.}))
        out.append(case(f"without_{i}", "leave_one_out", "Without " + NAMES[m],
                        weights={x: 1/(len(models)-1) for x in models if x != m}))
    proxy = next(m for m in models if m.startswith("MAI2"))
    for percent in (0, 10, 25, 50):
        w = percent/100
        out.append(case(f"proxy_{percent}", "proxy_weight", f"Proxy weight {percent}%",
                        weights={m: w if m == proxy else (1-w)/(len(models)-1) for m in models}))
    for key, label, multiplier in (("decode_only", "Decode only", 0),
                                    ("half_input", "Half new-input/output ratio", .5),
                                    ("double_input", "Double new-input/output ratio", 2)):
        out.append(case(key, "prefill_ratio", label, ratio=ratio*multiplier))
    for speed in (50, 100, 200, 400):
        out.append(case(f"point_{speed}", "single_point", f"Capacity at {speed} TPS/user",
                        axis="point", low=None, high=None, target=float(speed)))
    return out


def evaluate(saved, cases):
    pref = {(p["hardware"], p["model"]): p["tokens_chip_second"] for p in saved["prefill_rates"]}
    rows, components = [], []
    for case in cases:
        weights = {m: w for m, w in case["weights"].items() if w > 0}
        if not math.isclose(sum(weights.values()), 1):
            raise ValueError("Basket weights must sum to one")
        def metric(h, m):
            points, p = saved["curves"][h][m], pref[h, m]
            if case["axis"] == "point":
                return combined_capacity(points, case["target"], p, case["ratio"])
            return area(points, p, case["ratio"], case["low"], case["high"], case["axis"])
        refs = {m: metric(REFERENCE, m) for m in weights}
        undefined = [m for m, value in refs.items() if value <= 0]
        for h in saved["curves"]:
            values = []
            for m, w in weights.items():
                value = metric(h, m)
                score = value/refs[m] if refs[m] > 0 else None
                components.append(dict(scenario=case["id"], hardware=h, model=NAMES[m],
                    weight=w, metric=value, GB300_metric=refs[m], model_GB300e=score))
                if score is not None:
                    values.append(w*score)
            rows.append(dict(scenario=case["id"], family=case["family"], hardware=h,
                score_GB300e=math.fsum(values) if not undefined else None,
                status="undefined GB300 reference: " + ", ".join(undefined) if undefined else "ok"))
    baseline = {r["hardware"]: r["score_GB300e"] for r in rows if r["scenario"] == BASELINE}
    def rank(value, peers):
        if value is None:
            return None
        return 1 + sum(v is not None and v > value and not math.isclose(v, value, rel_tol=1e-12, abs_tol=1e-15) for v in peers)
    base_ranks = {h: rank(v, baseline.values()) for h, v in baseline.items()}
    for case in cases:
        group = [r for r in rows if r["scenario"] == case["id"]]
        for r in group:
            value, base = r["score_GB300e"], baseline[r["hardware"]]
            r.update(baseline_GB300e=base,
                change_percent=100*(value/base-1) if value is not None and base else None,
                rank=rank(value, [x["score_GB300e"] for x in group]),
                baseline_rank=base_ranks[r["hardware"]])
            r["rank_change"] = r["baseline_rank"]-r["rank"] if r["rank"] is not None else None
    return rows, components


def summaries(rows, hardware, families=None):
    out = []
    for family in families or ["all"]:
        for h in hardware:
            subset = [r for r in rows if r["hardware"] == h and
                      (family == "all" or r["family"] == family or r["scenario"] == BASELINE)]
            valid = [r for r in subset if r["score_GB300e"] is not None]
            base = next(r for r in subset if r["scenario"] == BASELINE)
            lo, hi = min(valid, key=lambda r:r["score_GB300e"]), max(valid, key=lambda r:r["score_GB300e"])
            out.append(dict(family=family, hardware=h, baseline_GB300e=base["score_GB300e"],
                minimum_GB300e=lo["score_GB300e"], maximum_GB300e=hi["score_GB300e"],
                minimum_scenario=lo["scenario"], maximum_scenario=hi["scenario"],
                minimum_change_percent=lo["change_percent"], maximum_change_percent=hi["change_percent"],
                baseline_rank=base["rank"], best_rank=min(r["rank"] for r in valid),
                worst_rank=max(r["rank"] for r in valid), defined_scenarios=len(valid),
                undefined_scenarios=len(subset)-len(valid)))
    return out


def csv_file(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(output=ROOT/"results/methodology-sensitivity"):
    from rubin_release import report_inputs, BASE, REV
    source, output = BASE, Path(output)
    revised = report_inputs()
    saved = json.loads((revised/"curves_and_inputs.json").read_text())
    cases = scenarios(saved["models"], saved["ratio"])
    rows, components = evaluate(saved, cases)
    expected = {r["hardware"]:float(r["agent_index_GB300e"])
                for r in csv.DictReader((revised/"index.csv").open())}
    base = {r["hardware"]:r["score_GB300e"] for r in rows if r["scenario"] == BASELINE}
    if set(base) != set(expected):
        raise ValueError("Incomplete release hardware set")
    error = max(abs(base[h]/expected[h]-1) for h in base)
    if error > 1e-12:
        raise ValueError(f"Headline parity failed: {error}")
    hardware = sorted(base, key=lambda h:(h != REFERENCE, -base[h], h))
    lookup = {(r["hardware"],r["scenario"]):r for r in rows}
    matrix = [dict(hardware=h, **{c["id"]:lookup[h,c["id"]]["score_GB300e"] for c in cases}) for h in hardware]
    summary = summaries(rows, hardware)
    family_summary = summaries(rows, hardware, sorted({c["family"] for c in cases}))
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("scenario_results.csv", rows), ("per_model_sensitivity.csv", components),
                       ("score_matrix.csv", matrix), ("robustness.csv", summary),
                       ("family_robustness.csv", family_summary)):
        csv_file(output/name, data)
    (output/"scenarios.json").write_text(json.dumps(cases, indent=2, allow_nan=False)+"\n")
    provenance = dict(source_release="2026-10-15 + 2026-10-20-rubin", refreshed="2026-10-29", simulation_rerun=False,
        revision_manifest_sha256=hashlib.sha256((REV/"manifest.json").read_bytes()).hexdigest(),
        merged_inputs_sha256=hashlib.sha256((revised/"curves_and_inputs.json").read_bytes()).hexdigest(),
        expected_revised_index_sha256=hashlib.sha256((revised/"index.csv").read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest()
                       for n in ("manifest.json", "curves_and_inputs.json", "expected_index.csv")},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        capacity_code_sha256=hashlib.sha256((ROOT/"src/agent_index/index.py").read_bytes()).hexdigest(),
        hardware=len(hardware), scenarios=len(cases), score_cells=len(rows),
        headline_max_relative_error=error,
        undefined_scores=sum(r["score_GB300e"] is None for r in rows),
        integration="Exact saved-breakpoint integration; no interpolation or extrapolation.",
        normalization="Per-model GB300 denominator recomputed within every scenario; weighted arithmetic mean.",
        ranking="Competition rank; 1e-12 relative tie tolerance; positive rank_change is improvement.")
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    render(output, cases, rows, summary, hardware, provenance)
    print(json.dumps(provenance))


def render(output, cases, rows, summary, hardware, provenance):
    lookup = {(r["hardware"],r["scenario"]):r for r in rows}
    f = lambda v: "N/A" if v is None else f"{v:.3f}"
    headings = "| Accelerator | " + " | ".join(c["id"] for c in cases[:8]) + " |"
    table = [headings, "|---|"+"---:|"*8]
    table += ["| "+h+" | "+" | ".join(f(lookup[h,c["id"]]["score_GB300e"]) for c in cases[:8])+" |" for h in hardware]
    basket_ids = [BASELINE, "glm_kimi", "only_0", "only_1", "only_2", "only_3", "without_3"]
    basket_table = ["| Accelerator | Headline | GLM + Kimi | GLM only | Kimi only | Qwen only | MAI proxy only | Without proxy |",
                    "|---|"+"---:|"*len(basket_ids)]
    basket_table += ["| "+h+" | "+" | ".join(f(lookup[h,key]["score_GB300e"]) for key in basket_ids)+" |" for h in hardware]
    vr = {c["id"]:lookup["VR200",c["id"]]["score_GB300e"] for c in cases}
    text = f"""# Sensitivity to methodology choices

**Report hardware revision: 20 October 2026, refreshed 29 October.** The 15 October base is combined with revised Rubin inputs: 19.2 TB/s HBM and 1.5 TB/s outbound NVLink. Baseline VR200 = 2.177367 GB300e. [Methodology](../../METHODOLOGY.md) | [Snapshot guide](../../SNAPSHOTS.md) | [Pre-revision archive](../archives/2026-10-19-pre-rubin-refresh/README.md).

All 28 scenarios are re-integrated from saved native curves. No timings, workload, runtime, precision or non-Rubin inputs change. Scores for other chips remain unchanged; their ranks can change.

The headline is unchanged: equal GLM-5.2, Kimi-K3, Qwen3-Coder 480B and
MAI-Base-2 2.8T proxy weights; log 50–400 SOL TPS/user; separate prefill/decode
provisioning at the saved new-input/output ratio of 1.513342523752522.

This study re-integrates the **same frozen 23-accelerator, four-model release**
from 15 October 2026. No timing, precision, memory, topology, model-dimension or
layout-search assumptions change. Later standalone hardware additions, including
A100, are not silently merged into this frozen basket.

Open [the interactive tables](index.html), or download [all scores](score_matrix.csv).
The detailed [long table](scenario_results.csv) includes percentage and rank
changes; [per-model contributions](per_model_sensitivity.csv) expose architecture
dependence. [Family summaries](family_robustness.csv) keep distinct choices separate.

## What changes?

- **Axis × band:** both linear and log integration over 50–200, 50–300,
  50–400 and 50–800. Same-band comparisons isolate weighting.
- **Basket:** GLM + Kimi, every model alone, and every leave-one-out basket.
- **Proxy weight:** 0%, 10%, 25%, 50%; remaining weight shared equally.
- **Prefill:** decode-only, half, current and double the input/output ratio.
- **Point alternative:** capacity at 50, 100, 200 or 400 SOL TPS/user.

There are {len(cases)} named scenarios, including the headline and the explicit
25%-proxy control (which reproduces it). Basket, ratio and point scenarios hold
all other headline choices fixed. This is not a full factorial of every choice.

## Linear versus logarithmic weighting

Linear integrates capacity against **ds**: equal absolute speed intervals have
equal weight. Log integrates against **ds/s**: equal multiplicative intervals
have equal weight. Changing only a display axis does not change a score.

For VR200, log 50–400 gives **{vr[BASELINE]:.3f} GB300e**, while linear over the
same band gives **{vr['linear_50_400']:.3f}**. Log 50–200 gives
**{vr['log_50_200']:.3f}** and log 50–800 gives **{vr['log_50_800']:.3f}**.
These are sensitivity results, not a reason to pick the methodology that
produces a preferred accelerator ratio.

{chr(10).join(table)}

## Model-basket sensitivity

All columns below retain log 50–400 and the observed input/output ratio. Individual
models are architecture diagnostics, not a proposed single-model headline.
For VR200 the two-model GLM + Kimi basket gives **{vr['glm_kimi']:.3f}**, versus
**{vr[BASELINE]:.3f}** for the four-model headline. Removing the MAI proxy gives
**{vr['without_3']:.3f}**; increasing its weight to 50% gives **{vr['proxy_50']:.3f}**.
The per-model range is wider: **{min(vr['only_'+str(i)] for i in range(4)):.3f}–{max(vr['only_'+str(i)] for i in range(4)):.3f}**.
This shows why a basket score should not be presented as an architecture-independent
hardware multiplier. See the full tables for other accelerators and rank changes.

{chr(10).join(basket_table)}

## Exact calculation and interpretation

For each speed threshold s, C(s) is the highest requests/allocated-decode-chip
density among saved qualifying layouts and batches. P is the saved optimal
prefill tokens/chip-second and r is new input/output tokens.

`B(s) = C(s) / (1 + r*s*C(s)/P)`.

At r=0, B=C. With r>0 and no prefill throughput, B=0. Integrate B using every
saved speed breakpoint. On a segment [a,b] with constant C=c and k=r*c/P:

- Linear area: `c/k * log((1+k*b)/(1+k*a))`, with limit `c*(b-a)` at k=0.
- Log area: `c * log(b*(1+k*a)/(a*(1+k*b)))`.

Implementation uses stable `log1p` forms. Capacity is zero when no saved point
qualifies, with no interpolation or extension past maximum feasible speed.
Exactly at a saved speed, a point with that speed still qualifies; point
comparisons respect this equality. Endpoints have zero measure for AUC.

Normalize each model to **GB300 under the identical scenario**, then take the
stated weighted arithmetic mean. Single-point scores normalize capacity rather
than area. A positive-weight model with zero GB300 reference makes the whole
scenario score undefined: no epsilon and no silent reweighting.

The combined capacity is monotonic in decode capacity for fixed s, P and r,
so reselecting the maximum saved decode density at each speed remains optimal
under this provisioning equation. This is optimization over saved candidates,
not a fresh layout search. Separate fully utilized prefill pools and fractional
fleet-scale provisioning remain assumptions.

[Robustness ranges](robustness.csv) are min/max across these deliberately selected
scenarios, **not confidence intervals or probabilities**. Single-model endpoints
are architecture diagnostics, not equally plausible replacement baskets.
Use family-specific ranges for narrower conclusions. Scores are displayed to
three decimals; CSVs retain full precision. Ranks use unrounded scores with a
1e-12 relative tie tolerance; positive rank change means improvement.

## Reproduce and audit

Run `python3 reproduce.py sensitivity` from the repository root. Standard library
only; no AISim installation, network access, private traces or GPU required.
Outputs go to `results/methodology-sensitivity/`. Use
`--output docs/research/methodology-sensitivity` to refresh this checked-in snapshot.

Source: `studies/methodology_sensitivity.py`. [Provenance](provenance.json) records
input/code hashes and headline parity. All {len(hardware)} headline scores are
checked against the revised report snapshot; maximum relative error:
`{provenance['headline_max_relative_error']:.3g}`.

Changed context length, speculative step size, precision, memory guards, topology
or architecture dimensions generally require new timing/search runs and are
outside this re-integration study. Basket weights are model weights, not traffic
shares. No update to the public headline, Google Sheet or hosted explainer is
implied by these alternative scenarios.
"""
    (output/"README.md").write_text(text)
    data = json.dumps(dict(cases=cases, rows=rows, summary=summary, hardware=hardware),
                      allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inference Capacity Index · Methodology sensitivity</title>
<style>body{font:16px/1.6 system-ui;margin:30px auto;max-width:1500px;padding:0 22px;color:#203530;background:#f7f7f0}
h1{font-size:34px;line-height:1.15}a{color:#225f81}select{font:inherit;padding:8px}table{border-collapse:collapse;font-size:13px;white-space:nowrap;width:100%}
th,td{padding:9px 12px;border-bottom:1px solid #d6ddd6;text-align:right}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:#f7f7f0}
thead th{background:#e7eee4}td.base,th.base{background:#e7eee4}.scroll{overflow:auto;max-height:70vh;border:1px solid #d6ddd6;margin:15px 0 30px}
.note{padding:16px;background:#e7eee4;max-width:1100px}.muted{color:#596b63;font-size:14px}details{margin:22px 0}summary{cursor:pointer;font-weight:650}
label{display:inline-block;margin:10px 24px 10px 0}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #ba612b}
</style>
<h1>How sensitive is the index to methodological choices?</h1>
<p class="note"><strong>20 October 2026 report hardware revision, refreshed 29 October.</strong> Rubin uses 19.2 TB/s HBM and 1.5 TB/s outbound NVLink (baseline 2.177). All 28 scenarios use the revised curves; the other 22 hardware profiles and the workload are unchanged. <a href="../archives/2026-10-19-pre-rubin-refresh/README.md">Pre-revision archive</a>.</p>
<p>Same frozen simulations. Different model baskets, interactivity weights, speed bands and input provisioning.</p>
<p class="note"><b>The headline is unchanged.</b> Four equally weighted models, log 50–400 SOL TPS/user, workload-based prefill provisioning.
Every alternative recomputes its own per-model GB300 normalization. This uses the report’s revised 23-accelerator inputs; it is not a new timing run.</p>
<p><a href="README.md">Full method and interpretation</a> · <a href="score_matrix.csv">Full score CSV</a> · <a href="scenario_results.csv">Scores, changes and ranks</a> · <a href="per_model_sensitivity.csv">Per-model detail</a> · <a href="scenarios.json">Exact scenarios</a> · <a href="provenance.json">Provenance</a></p>
<label>Scenario family <select id="family"><option value="axis_band">Linear/log × speed band</option><option value="basket">GLM + Kimi</option><option value="single_model">Each model individually</option><option value="leave_one_out">Leave one model out</option><option value="proxy_weight">MAI-Base-2 proxy weight</option><option value="prefill_ratio">Input/output ratio</option><option value="single_point">Single speed threshold</option><option value="all">All scenarios</option></select></label>
<label>Show <select id="metric"><option value="score_GB300e">GB300-equivalent score</option><option value="change_percent">Change from headline (%)</option><option value="rank">Rank</option><option value="rank_change">Rank improvement</option></select></label>
<p class="muted">The highlighted first column is always the headline. Accelerator order stays fixed by headline score, with GB300 first.
At matched bounds, linear weights equal absolute intervals; log weights equal speed ratios. This changes integration, not just axis styling.</p>
<div class="scroll"><table id="matrix"></table></div>
<details><summary>Min/max across all selected scenarios—not uncertainty intervals</summary>
<p>These ranges include single-model extremes and single-speed alternatives. They are not confidence intervals.
For a more focused assessment, use <a href="family_robustness.csv">family-specific ranges</a>.
Raw ranks use unrounded values; tied ranks can occur.</p><div class="scroll"><table id="summary"></table></div></details>
<details><summary>What is held fixed?</summary><p>262,144 resident tokens, native nextn3/AL3 decode; 1,024-token cached-prefix prefill;
saved model dimensions, precision, topology, memory guards and admitted layouts. Capacity is reselected across saved points at each speed.
No interpolation, extrapolation, hardware retiming or traffic simulation. The input/output ratio baseline is 1.513342523752522.</p>
<p>At every speed, combined capacity is C/(1+r*s*C/P). We integrate exactly at saved breakpoints, normalize each model to GB300 for the
same scenario, and average using the stated basket weights. A zero positive-weight GB300 reference produces N/A, never an epsilon ratio.</p></details>
<script id="data" type="application/json">DATA_PLACEHOLDER</script>
<script>
const d=JSON.parse(document.getElementById('data').textContent),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const map=new Map(d.rows.map(r=>[r.hardware+'|'+r.scenario,r]));
function draw(){const family=document.getElementById('family').value,metric=document.getElementById('metric').value;
const cases=d.cases.filter(c=>c.id==='log_50_400'||family==='all'||c.family===family);
let h='<thead><tr><th>Accelerator</th>'+cases.map(c=>'<th class="'+(c.id==='log_50_400'?'base':'')+'">'+esc(c.label)+'</th>').join('')+'</tr></thead><tbody>';
for(const chip of d.hardware){h+='<tr><td>'+esc(chip)+'</td>';for(const c of cases){const r=map.get(chip+'|'+c.id),v=r[metric];
let val=v===null?'N/A':metric.includes('rank')?String(v):v.toFixed(metric==='change_percent'?1:3);
if(v!==null&&(metric==='rank_change'||metric==='change_percent')&&v>0)val='+'+val;
h+='<td class="'+(c.id==='log_50_400'?'base':'')+'" title="'+esc(r.status)+'">'+val+'</td>';}h+='</tr>';}
document.getElementById('matrix').innerHTML=h+'</tbody>';}
document.getElementById('family').addEventListener('change',draw);document.getElementById('metric').addEventListener('change',draw);draw();
document.getElementById('summary').innerHTML='<thead><tr><th>Accelerator</th><th>Headline</th><th>Minimum</th><th>Maximum</th><th>Best rank</th><th>Worst rank</th></tr></thead><tbody>'+
d.summary.map(r=>'<tr><td>'+esc(r.hardware)+'</td><td>'+r.baseline_GB300e.toFixed(3)+'</td><td>'+r.minimum_GB300e.toFixed(3)+'</td><td>'+r.maximum_GB300e.toFixed(3)+'</td><td>'+r.best_rank+'</td><td>'+r.worst_rank+'</td></tr>').join('')+'</tbody>';
</script></html>"""
    (output/"index.html").write_text(page.replace("DATA_PLACEHOLDER", data))


if __name__ == "__main__":
    main()
