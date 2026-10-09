"""Portable CSV/JSON/HTML publication using only the standard library."""
import csv
import hashlib
import html
import json
import math
from pathlib import Path

from .index import MODELS, REFERENCE, RELEASE, read_csv, load_catalog


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def chart(data, catalog, model):
    """Display exact step functions; score computation never samples this SVG."""
    width, height = 1020, 460
    left, right, top, bottom = 60, 810, 35, 410
    colors = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#5c3b99", "#6e6e00")
    selected = [REFERENCE, "VR200", "MI455X", "B300 · 800G", "Jalapeño", "TPU8i · corrected · hops"]
    ymax = max(p["capacity"] for h in selected for p in data["curves"][h][model]) * 1.05
    def x(s): return left + math.log(s / 50) / math.log(8) * (right-left)
    def y(c): return bottom - c / ymax * (bottom-top)
    items = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(model)} capacity frontier">',
             f'<text x="{left}" y="22">{html.escape(model)} — resident decoding requests / chip</text>']
    for tick in (50, 100, 200, 400):
        items.append(f'<path d="M{x(tick):.2f},{top} V{bottom}" stroke="#ddd"/><text x="{x(tick):.2f}" y="432">{tick}</text>')
    for i in range(6):
        v=ymax*i/5
        items.append(f'<path d="M{left},{y(v):.2f} H{right}" stroke="#eee"/><text x="5" y="{y(v):.2f}">{v:.1f}</text>')
    for i,h in enumerate(selected):
        points=data["curves"][h][model]
        edges=sorted({50,400,*(p["speed"] for p in points if 50<p["speed"]<400)})
        coords=[]
        for a,b in zip(edges,edges[1:]):
            c=max((p["capacity"] for p in points if p["speed"]>=(a+b)/2),default=0)
            coords.extend([(x(a),y(c)),(x(b),y(c))])
        path=" ".join(f'{"M" if j==0 else "L"}{a:.2f},{b:.2f}' for j,(a,b) in enumerate(coords))
        items.append(f'<path d="{path}" fill="none" stroke="{colors[i]}" stroke-width="2"/><text x="825" y="{55+i*25}" fill="{colors[i]}">{html.escape(h)}</text>')
    items.append('<text x="330" y="455">Required SOL TPS/user (logarithmic axis)</text></svg>')
    return "".join(items)


def publish(root, data, destination, validation):
    destination = Path(destination).resolve()
    # Never allow generated files to replace the frozen release or source code.
    for protected in ("reference", "configs", "src", "docs", "environment"):
        if destination.is_relative_to((root / protected).resolve()):
            raise ValueError(f"Output is inside protected {protected}/")
    if destination == root.resolve():
        raise ValueError("Use an output directory, not the repository root")
    destination.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(root)
    expected = {r["accelerator"]: r for r in read_csv(root / f"reference/{RELEASE}/expected_scores.csv")}
    rows=[]
    for hardware, r in data["results"].items():
        row=expected[catalog[hardware]["name"]].copy()
        row.update(agent_index_GB300e=r["combined"]["log400"],
                   GLM_5_2_GB300e=r["models"][MODELS[0]]["log400"]["score"],
                   Kimi_K3_GB300e=r["models"][MODELS[1]]["log400"]["score"])
        rows.append(row)
    rows.sort(key=lambda r:(r["accelerator"]!=catalog[REFERENCE]["name"],-r["agent_index_GB300e"]))
    write_csv(destination / "agent_index.csv", rows)
    (destination / "frontiers.json").write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")
    targets=[]
    for h,models in data["curves"].items():
        for model,points in models.items():
            for target in (50,100,200,400):
                winner=max((p for p in points if p["speed"]>=target),key=lambda p:(p["capacity"],p["speed"]),default=None)
                targets.append({"accelerator":catalog[h]["name"],"model":model,"required_SOL_TPS":target,
                                "agents_per_allocated_chip":winner["capacity"] if winner else 0,
                                "achieved_SOL_TPS":winner["speed"] if winner else "",
                                "requests":winner["requests"] if winner else "",
                                "chips":winner["chips"] if winner else "",
                                "layout":winner["layout"] if winner else "No qualifying candidate"})
    write_csv(destination / "capacity_targets.csv",targets)
    sensitivities=[{"accelerator":catalog[h]["name"],**data["results"][h]["combined"]} for h in data["results"]]
    write_csv(destination / "sensitivity_scores.csv",sensitivities)
    headers=("Accelerator","Combined GB300e","GLM","Kimi")
    table='<table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'
    for r in rows:
        table+='<tr><td>'+html.escape(r['accelerator'])+'</td>'+''.join(f'<td>{r[k]:.6f}</td>' for k in ('agent_index_GB300e','GLM_5_2_GB300e','Kimi_K3_GB300e'))+'</tr>'
    table+='</tbody></table>'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Agent capacity index — frozen release</title>
<style>body{font:16px/1.55 system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#213547}table{border-collapse:collapse;width:100%}th,td{text-align:right;padding:8px;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}svg{width:100%;font:13px system-ui}aside{background:#edf3f7;padding:20px}a{color:#00689d}</style>
<h1>Agent capacity index</h1><p>Frozen release: 9 October 2026. Log-capacity AUC over 50–400 SOL TPS/user; equal GLM-5.2 and Kimi-K3 weights; per-model GB300 normalization.</p>
<aside>256 Ki cached tokens; decode only; nextn3 / assumed AL3. These are conditional analytical hardware-resource projections, not measured production agent capacity. Each point counts resident decoding requests per allocated physical chip. Tool pauses, prefill, queueing and duty cycles are excluded. All hardware inputs and vendor assumptions are frozen, not newly verified.</aside>'''
    page+=table+''.join(chart(data,catalog,m) for m in MODELS)
    page+='''<p>Charts show six illustrative chips; CSV and JSON cover all 17. Step functions are not interpolated. Re-optimization is over saved configurations, not every possible implementation. Display rendering does not affect scoring.</p><p><a href="agent_index.csv">Complete CSV and hardware notes</a> · <a href="frontiers.json">Full frontier data</a> · <a href="capacity_targets.csv">Winning configurations at speed targets</a> · <a href="sensitivity_scores.csv">Band/weighting sensitivities</a> · <a href="validation.json">Validation</a></p></html>'''
    (destination / "REPORT.html").write_text(page)
    validation=dict(validation)
    validation['outputs_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir()) if p.name in ('agent_index.csv','frontiers.json','REPORT.html','capacity_targets.csv','sensitivity_scores.csv')}
    (destination / "validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    return destination
