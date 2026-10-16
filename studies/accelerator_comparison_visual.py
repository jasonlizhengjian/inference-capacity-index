"""Render frozen index comparisons; no AISim calls or score recomputation.

Power inputs: Epoch's dated ZIP snapshot, supplemented by the existing sourced
power ledger and the Maia/MTIA studies. Hardware timing inputs are not modified.
"""
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/methodology-current'
STUDY = ROOT / 'results/workload-capacity-oct15-log400'
LEDGER = ROOT / 'reference/2026-10-15/power_ledger.json'
EPOCH_URL = 'https://epoch.ai/data/machine-learning-hardware'
MAIA = 'https://azure.microsoft.com/en-us/blog/wp-content/uploads/2026/10/Maia-300-Specifications.webp'
META = 'https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/'
COLORS = {'NVIDIA':'#276b99','AMD':'#ad482d','Google':'#8b6b16','Microsoft':'#568564',
          'Meta':'#6862a5','OpenAI':'#087f82','Huawei':'#9c5c72'}
MATCHES = {
    'GB300 NVL72':'NVIDIA GB300 (Blackwell Ultra)',
    'GB200 NVL72':'NVIDIA GB200', 'B300 · 800G':'NVIDIA B300 (Blackwell Ultra)',
    'B200 · 800G':'NVIDIA B200', 'H100 SXM · FP8 · 400G':'NVIDIA H100 SXM5 80GB',
    'H200 SXM · FP8 · 400G':'NVIDIA H200 SXM', 'MI355X · 400G':'AMD Instinct MI355X',
    'TPU7x Ironwood · FP8':'Google TPU v7 Ironwood', 'Maia 200 rack':'Microsoft Maia 200',
}


def vendor(name):
    for prefix, company in [('Meta','Meta'),('Maia','Microsoft'),('TPU','Google'),
                             ('Ascend','Huawei'),('MI','AMD'),('Jalape','OpenAI')]:
        if name.startswith(prefix): return company
    return 'NVIDIA'


def label(name):
    return name.replace(' · corrected · hops',' · hop model').replace(' · FP4 equivalent',' · FP4 proxy')


def status(name):
    if name.startswith('Ascend'): return 'Provisional topology / silicon-resource scenario'
    if 'MTIA 500' in name: return 'Roadmap memory endpoint; not a confirmed SKU'
    if 'MTIA 450' in name: return 'Roadmap scenario'
    return 'Modeled hardware profile'


def svg_chart(rows, power=False):
    key = 'relative_score_per_rated_watt' if power else 'score'
    ordered = sorted((r for r in rows if r[key] is not None), key=lambda r:r[key], reverse=True)
    title = 'Capacity per rated watt' if power else 'Agent capacity index'
    sub = 'Selected vendors · published ratings and estimates · GB300 = 1' if power else 'All 23 hardware profiles · combined prefill + decode · GB300 = 1'
    width, left, plot, top, rowh = 1120, 340, 650, 126, 39
    bottom = top + rowh*len(ordered)
    height = bottom+128
    limit = 3.5
    ident = 'power-chart' if power else 'capacity-chart'
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{ident}-title {ident}-desc">',
             f'<title id="{ident}-title">{title}</title><desc id="{ident}-desc">Ranked bar chart. Higher is better. Values are model projections, not measured performance. A complete data table follows on the page.</desc>',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<g font-family="Arial, sans-serif" fill="#243448">',
             f'<text x="24" y="36" font-size="25" font-weight="700">{title}</text>',
             f'<text x="24" y="64" font-size="15">{sub}</text>']
    for tick in [0,.5,1,1.5,2,2.5,3,3.5]:
        x = left+plot*tick/limit
        parts += [f'<path d="M{x},{top-12} V{bottom}" stroke="{ "#3b4b5d" if tick==1 else "#e6e9ee"}" stroke-dasharray="{ "5 4" if tick==1 else "none"}"/>',
                  f'<text x="{x}" y="{top-25}" text-anchor="middle" font-size="13">{tick:g}{" · GB300" if tick==1 else ""}</text>']
    for i,r in enumerate(ordered):
        y = top+i*rowh
        name = label(r['hardware'])
        if r['manufacturer']=='Huawei': name += ' *'
        if 'MTIA 450' in name or 'MTIA 500' in name: name += ' †'
        value = r[key]
        assert 0 <= value <= limit
        if r['hardware']=='GB300 NVL72':
            parts.append(f'<rect x="12" y="{y-5}" width="1090" height="36" fill="#edf4fa" rx="4"/>')
        # Keep the baseline visible on top of the highlighted row.
        parts += [f'<text x="{left-15}" y="{y+15}" text-anchor="end" font-size="14" font-weight="{700 if r["hardware"]=="GB300 NVL72" else 400}">{html.escape(name)}</text>',
                  f'<rect x="{left}" y="{y}" width="{plot*value/limit:.3f}" height="22" rx="2" fill="{COLORS[r["manufacturer"]]}"/>',
                  f'<text x="{left+plot*value/limit+9:.3f}" y="{y+16}" font-size="14" font-weight="700">{value:.3f}</text>']
        if power:
            parts.append(f'<text x="1090" y="{y+16}" text-anchor="end" font-size="12" fill="#66717f">{r["watts"]:g} W{" ≈" if r["hardware"].startswith("TPU7") else ""}</text>')
    foot = ('Score × 1,400 W ÷ rated accelerator watts. Not measured energy efficiency or whole-system power.' if power else
            'Log-weighted 50–400 SOL TPS/user · equal GLM, Kimi, Qwen and MAI-2 2.8T weights.')
    parts += [f'<text x="24" y="{bottom+30}" font-size="13">{foot}</text>',
              f'<text x="24" y="{bottom+53}" font-size="13">† MTIA 450/500 are roadmap scenarios. {"≈ Ironwood watts are an Epoch estimate." if power else "* Huawei topology is under review; frozen scores retained."}</text>',
              f'<text x="24" y="{bottom+76}" font-size="12" fill="#66717f">October 15, 2026 MAI-2 basket · see linked methodology and power notes.</text>',
              '</g></svg>']
    return ''.join(parts)


def main():
    ledger = json.loads(LEDGER.read_text())
    saved_rows = {r['hardware']:r for r in ledger['rows']}
    rows = []
    for src in csv.DictReader((STUDY/'index.csv').open()):
        r = dict(saved_rows[src['hardware']])
        r['score'] = float(src['agent_index_GB300e'])
        r['relative_score_per_rated_watt'] = r['score']*1400/r['watts'] if r['watts'] else None
        rows.append(r)
    assert len(rows)==23 and len({r['hardware'] for r in rows})==23
    ref = next(r for r in rows if r['hardware']=='GB300 NVL72')
    assert ref['watts']==1400 and ref['score']==ref['relative_score_per_rated_watt']==1
    rows.sort(key=lambda r:r['score'],reverse=True)
    power_rows = [r for r in rows if r['manufacturer']!='Huawei']
    ranked = [r for r in power_rows if r['watts']]
    missing = [r for r in power_rows if not r['watts']]
    assert len(ranked)==15 and len(missing)==4
    provenance = dict(score_file='results/workload-capacity-oct15-log400/index.csv',
        score_sha256=hashlib.sha256((STUDY/'index.csv').read_bytes()).hexdigest(),
        epoch_archive_sha256=ledger['epoch_archive_sha256'],
        epoch_download_url='https://epoch.ai/data/ml_hardware.zip',epoch_retrieved='2026-10-15',
        note='Only chart arithmetic; no timing, model, hardware, layout, search or index changes.',rows=rows)
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'accelerator_comparison_data.json').write_text(json.dumps(provenance,indent=2)+'\n')
    with (OUT/'accelerator_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    first,second=svg_chart(rows),svg_chart(power_rows,True)
    (OUT/'accelerator_scores.svg').write_text(first)
    (OUT/'accelerator_scores_per_watt.svg').write_text(second)
    legend=''.join(f'<span><i style="background:{c}"></i>{n}</span>' for n,c in COLORS.items())
    table=''.join('<tr>'+''.join(f'<td>{v}</td>' for v in [html.escape(r['hardware']),r['manufacturer'],f'{r["score"]:.6f}',
         f'{r["watts"]:g}' if r['watts'] else '—', f'{r["relative_score_per_rated_watt"]:.6f}' if r['watts'] else '—',
         html.escape(r['power_basis']),f'<a href="{r["power_source"]}">Source</a>' if r['power_source'] else '—',html.escape(r['hardware_status'])])+'</tr>' for r in rows)
    absent=' · '.join(html.escape(label(r['hardware'])) for r in missing)
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Compare accelerators — Agent capacity index</title>
<style>body{margin:0;background:#f7f8fa;color:#243448;font:16px/1.6 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:30px 24px 70px}h1{font-size:clamp(30px,5vw,44px);line-height:1.15;margin:24px 0 16px}h2{font-size:25px;margin-bottom:8px}p{max-width:900px}a{color:#175f9e}nav{display:flex;gap:20px;flex-wrap:wrap}section{margin-top:45px;scroll-margin-top:20px}.intro{font-size:19px;max-width:850px}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;margin:24px 0 12px}.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px}.chart{background:white;border:1px solid #dbe1e8;border-radius:8px;overflow-x:auto}.chart svg{display:block;width:100%;min-width:760px}.note{border-left:3px solid #ae7b26;padding:10px 18px;background:#fff9ee;max-width:none}.small{font-size:14px;color:#526176}details{margin:20px 0;border:1px solid #dbe1e8;border-radius:6px;background:white;padding:14px 18px}summary{cursor:pointer;font-weight:600}code{font-size:14px}table{border-collapse:collapse;font-size:13px;min-width:1050px;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #e3e7ec;vertical-align:top}th{background:#edf2f7}.table{overflow:auto}.downloads{display:flex;gap:22px;flex-wrap:wrap;margin-top:12px}footer{margin-top:45px;border-top:1px solid #dbe1e8;padding-top:16px}</style></head><body><main>
<nav aria-label="Related explainers"><a href="REPORT.html">← How the index works</a><a href="PERFORMANCE_KNEES.html">Explore performance knees →</a></nav>
<h1>Compare the accelerators</h1><p class="intro">One chart for capacity per allocated accelerator. A second for capacity per rated watt. Both use the same current combined index, with GB300 as the reference.</p>
<nav aria-label="Chart sections"><a href="#capacity">Capacity scores</a><a href="#power">Power-normalized scores</a><a href="#data">Data &amp; sources</a></nav>
<p class="small">October 15 MAI-2 basket · log-weighted 50–400 SOL TPS/user · four equally weighted architectures: GLM-5.2, Kimi-K3, Qwen3-Coder 480B and MAI-2 2.8T.</p>
<section id="capacity"><h2>1. Capacity per allocated accelerator</h2><p>A score of 2 means twice GB300’s normalized area under the modeled agent-capacity curve—not twice the single-user token speed. Prefill and decode chip requirements are combined before model-level normalization.</p>
<div class="legend">LEGEND</div><div class="chart">FIRST</div><div class="downloads"><a href="accelerator_scores.svg" download>Download chart (SVG)</a><a href="accelerator_comparison.csv" download>Download data (CSV)</a></div>
<p class="note"><strong>Scenario labels matter.</strong> MTIA 450/500 are roadmap profiles; the two MTIA 500 rows are alternative memory endpoints. Huawei scores retain the earlier optimistic, flat-topology silicon-resource assumptions. New evidence supports a 64-chip locality boundary for Atlas 950; the current scores have not yet been rerun with that hierarchy. Exact 960 topology remains uncertain. See Huawei’s <a href="https://arxiv.org/html/2503.20377v3">UB-Mesh paper</a> and <a href="https://www.huawei.com/en/news/2026/3/mwc-superpod-ai">Atlas 950 announcement</a>.</p>
<details><summary>What is being held fixed?</summary><p>The four models receive equal weights. Across hardware, we use 256Ki context, nextn3/AL3 for decode, a 1Ki-new-token prefill objective and the same workload input/output ratio. The index adds the modeled accelerator requirements of separate prefill and decode pools, not an arithmetic blend of phase scores. Queueing and tool-wait duty cycles are excluded.</p><p>All results are analytical SOL projections, not measured serving performance. MAI-2 2.8T combines disclosed architecture with explicitly assumed dimensions and speculative decoding; it is not an exact Microsoft production implementation.</p><p><a href="REPORT.html">See the full methodology and configuration explorer.</a></p></details></section>
<section id="power"><h2>2. Capacity per rated accelerator watt</h2><p>For NVIDIA, AMD, Google TPU, Microsoft Maia, Meta MTIA and OpenAI Jalapeño. A value of 2 means twice GB300’s modeled index capacity per rated watt, using GB300’s 1,400 W rating.</p>
<p><code>Power-normalized score = capacity score × 1,400 W / accelerator watts</code></p>
<p class="note"><strong>This is a rated-power proxy, not measured energy efficiency.</strong> TDP, module power and SoC power have different boundaries. External switches, CPUs, cooling and facility overhead are not added. The same rating is charged to every allocated prefill/decode accelerator; we do not model workload-dependent power or power-cap performance.</p>
<div class="chart">SECOND</div><div class="downloads"><a href="accelerator_scores_per_watt.svg" download>Download power chart (SVG)</a></div>
<p><strong>Unranked: ABSENT.</strong> No adopted absolute wattage is available in the retained source set. These are missing values, not zero scores. Huawei is outside the requested power-comparison scope.</p>
<details><summary>Power sources and the Ironwood estimate</summary><p>Epoch AI’s <a href="https://epoch.ai/data/machine-learning-hardware">machine-learning hardware dataset</a>, retrieved October 15, 2026, supplies TDP for the matched NVIDIA profiles, MI355X, Maia 200 and TPU v7. We use only its power column; we do not replace any frozen timing or hardware-spec inputs.</p><p>Ironwood’s 960 W is an Epoch estimate inferred from Google’s relative FLOP-per-watt claim: (4,614 / 46) × 280 / 29.3 ≈ 959 W, rounded to 960 W. It is not an official absolute Google TDP.</p><p>For entries absent from Epoch, we retain sourced study values: Jalapeño 700 W; Maia 300 950 W SoC TDP; MTIA 400/450/500 at 1,200/1,400/1,700 W module power. Both MTIA 500 memory scenarios use the same roadmap power assumption. Maia’s SoC boundary and other vendors’ module/package boundaries are not proven identical.</p><p>Power-normalization divides the already computed index by a hardware-specific constant. It does not select different configurations or claim the modeled peak bandwidth and FLOPs are simultaneously sustainable at that rating.</p></details></section>
<section id="data"><h2>Data &amp; source ledger</h2><p class="small">More digits are available for audit, not as a claim of predictive precision. Model-basket changes can move scores even when hardware assumptions are unchanged.</p><div class="table"><table><thead><tr><th>Accelerator</th><th>Manufacturer</th><th>Combined score</th><th>Watts</th><th>Score/watt vs GB300</th><th>Power basis</th><th>Source</th><th>Profile note</th></tr></thead><tbody>TABLE</tbody></table></div><p><a href="accelerator_comparison_data.json" download>Download full ledger and input hashes (JSON)</a></p></section>
<footer class="small">Power dataset attribution: Epoch AI, “Data on machine learning hardware,” <a href="https://epoch.ai/data/machine-learning-hardware">epoch.ai</a>, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Chart arithmetic only; no new simulation runs. <a href="REPORT.html">Return to methodology</a>.</footer></main></body></html>'''
    for a,b in [('LEGEND',legend),('FIRST',first),('SECOND',second),('ABSENT',absent),('TABLE',table)]:
        assert page.count(a)==1
        page=page.replace(a,b)
    assert '/home/' not in page
    (OUT/'ACCELERATOR_COMPARISON.html').write_text(page)
    print(f'Rendered {len(rows)} scores and {len(ranked)} rated-power proxies; {len(missing)} selected-vendor watts missing. No simulations run.')


if __name__=='__main__': main()
