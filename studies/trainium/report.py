"""Publishable, offline summary from saved Trainium scenario frontiers.

python3 studies/trainium/report.py
Reads docs/research/trainium by default; --prepare imports the completed local
study and the public chip-sales snapshot. It never changes headline scores.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/trainium'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'studies')]
from log400_release import combined_area,REFERENCE
from agent_index.index import capacity


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def prepare():
    source=ROOT/'results/trainium-2026-10-22-audit'
    DOC.mkdir(parents=True,exist_ok=True)
    for name in ('overall.csv','per_model.csv','targets.csv','provenance.json'):
        if name.endswith('.csv'):
            with (source/name).open(newline='') as f:
                csv_write(DOC/name,list(csv.DictReader(f)))
        else:
            shutil.copyfile(source/name,DOC/name)
    shutil.copytree(source/'runs',DOC/'runs',dirs_exist_ok=True)
    shutil.copytree(source/'inputs/hardware',DOC/'hardware',dirs_exist_ok=True)
    # Only public aggregate shipment inputs: no raw/private trace data.
    data=ROOT.parent/'sites/inference-compute-growth/dist/data.json'
    d=json.loads(data.read_text());h100=d['scores']['NVIDIA H100 SXM5 80GB']
    def contribution(r):
        chip=d['chips'][r['chip']];s=d['scores']
        score=chip['score'] if chip['score'] is not None else chip['h100e']*h100
        if r['chip']=='B200':score=.4*s['NVIDIA B200']+.6*s['NVIDIA GB200']
        elif r['chip']=='B300':score=.4*s['NVIDIA B300 (Blackwell Ultra)']+.6*s['NVIDIA GB300 (Blackwell Ultra)']
        value=r['units']*score
        if r['chip']=='H100':value+=r.get('h200_eligible_units',0)*.5*(s['NVIDIA H200 SXM']-h100)
        return value
    quarters=[]
    for date in sorted({r['end'] for r in d['stock'] if r['end']>='2025-06-30'}):
        rs=[r for r in d['stock'] if r['end']==date]
        quarters.append(dict(date=date,baseline_capacity=sum(contribution(r) for r in rs),
            peak_compute=sum(r['h100e'] for r in rs),
            trainium2_units=sum(r['units'] for r in rs if r['chip']=='Trainium2'),
            trainium3_units=sum(r['units'] for r in rs if r['chip']=='Trainium3')))
    save(DOC/'growth_inputs.json',dict(source='https://epoch.ai/data/ai-chip-sales',
        sales_archive_sha256=d['sales_zip_sha256'],snapshot_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        old_scores={str(g):d['chips'][f'Trainium{g}']['h100e']*h100 for g in (2,3)},
        assumptions='60% shared NVL72 mix; 50% eligible H200; H100-scaled unscored chips. Shipment medians held fixed.',quarters=quarters))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    frozen=json.loads((ROOT/'reference/2026-10-15/curves_and_inputs.json').read_text())
    ref_pref={r['model']:r['tokens_chip_second'] for r in frozen['prefill_rates'] if r['hardware']==REFERENCE}
    runs=[json.loads(p.read_text()) for p in sorted((DOC/'runs').glob('*.json'))]
    assert len(runs)==8
    scores=list(csv.DictReader((DOC/'per_model.csv').open()))
    max_error=0.;unique_decode=unique_prefill=0
    for r in runs:
        assert max(r['parity'].values())<1e-10
        unique_decode+=max(c['candidates_decode'] for c in r['cases'].values())
        unique_prefill+=max(c['candidates_prefill'] for c in r['cases'].values())
        m=r['model'];denom=combined_area(frozen['curves'][REFERENCE][m],ref_pref[m],frozen['ratio'])
        for name,c in r['cases'].items():
            rate=c['prefill']['input_tokens_chip_second'] if c['prefill'] else 0
            score=combined_area(c['frontier'],rate,frozen['ratio'])/denom
            stored=float(next(v for v in scores if int(v['generation'])==r['generation'] and v['model']==m and v['scenario']==name)['score'])
            max_error=max(max_error,abs(score-stored));assert abs(score-stored)<1e-12
            rows=c['frontier']+([c['prefill']] if c['prefill'] else [])
            for row in rows:
                assert row['chips']==row['tp']*row['dp']*row['pp']
                assert row['requests']==row['batch']*row['dp']
                assert row['fits'] and row['memory']<=.9*(96 if r['generation']==2 else 144)+1e-9
                assert row['pp']==1
    totals=list(csv.DictReader((DOC/'overall.csv').open()))
    lookup={(int(r['generation']),r['scenario']):float(r['score']) for r in totals}
    growth=json.loads((DOC/'growth_inputs.json').read_text());old=growth['old_scores'];qs=growth['quarters']
    growth_cases=[('Existing fallback',float(old['2']),float(old['3'])),
        ('Trn2 16-chip / Trn3 64-chip detailed',lookup[2,'16_link_limited'],lookup[3,'64_detailed_tiers']),
        ('Trn2 64-chip link-limited / Trn3 64-chip detailed',lookup[2,'64_link_limited'],lookup[3,'64_detailed_tiers']),
        ('Trn2 64-chip link-limited / Trn3 144-chip aggregate-half',lookup[2,'64_link_limited'],lookup[3,'144_aggregate_half']),
        ('Injection ceilings, not preferred estimates',lookup[2,'64_injection_ceiling'],lookup[3,'144_injection_ceiling'])]
    impacts=[]
    for name,t2,t3 in growth_cases:
        values=[q['baseline_capacity']+q['trainium2_units']*(t2-float(old['2']))+q['trainium3_units']*(t3-float(old['3'])) for q in qs]
        impacts.append(dict(scenario=name,trainium2_score=t2,trainium3_score=t3,
            capacity_q2_2025=values[0],capacity_q2_2026=values[-1],annual_growth=values[-1]/values[0],
            endpoint_change_pct=100*(values[-1]/qs[-1]['baseline_capacity']-1)))
    csv_write(DOC/'growth_impact.csv',impacts)
    save(DOC/'validation.json',dict(models=frozen['models'],input_output_ratio=frozen['ratio'],
        hardware_model_pairs=len(runs),unique_decode_candidates=unique_decode,unique_prefill_candidates=unique_prefill,
        max_native_parity_error=max(max(r['parity'].values()) for r in runs),max_recomputed_score_error=max_error,
        tested='Saved-score replay, chip/request identities, 90% memory admission, and native operation/overlap parity. Not empirical hardware calibration.'))
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
    def chart():
        # Exact step envelope at every saved breakpoint, log-speed horizontal axis.
        selected=[(2,'64_link_limited','#b86528'),(3,'64_detailed_tiers','#258b82'),(3,'144_aggregate_half','#51519b')]
        panels=[]
        for model in frozen['models']:
            curves=[('GB300',frozen['curves'][REFERENCE][model],'#222')]
            for gen,case,color in selected:
                run=next(r for r in runs if r['generation']==gen and r['model']==model)
                curves.append((f'Trn{gen} {case}',run['cases'][case]['frontier'],color))
            maximum=max(capacity(p,50) for _,p,_ in curves)*1.05
            X=lambda s:55+440*math.log(s/50)/math.log(8)
            Y=lambda c:235-190*c/maximum
            svg=['<svg viewBox="0 0 520 285" role="img" aria-label="Decode capacity frontier">']
            for t in (50,100,200,400):svg.append(f'<path d="M{X(t)} 40V235" stroke="#ddd"/><text x="{X(t)}" y="257" text-anchor="middle">{t}</text>')
            for f in (0,.5,1):
                y=Y(maximum*f);svg.append(f'<path d="M55 {y}H495" stroke="#ddd"/><text x="48" y="{y+4}" text-anchor="end">{maximum*f:.1f}</text>')
            for label,points,color in curves:
                edges=sorted({50,400,*(p['speed'] for p in points if 50<p['speed']<400)})
                coords=[]
                for a,b in zip(edges,edges[1:]):
                    value=capacity(points,(a+b)/2);coords.extend([(X(a),Y(value)),(X(b),Y(value))])
                svg.append('<polyline fill="none" stroke="'+color+'" stroke-width="2.4" points="'+' '.join(f'{x:.3f},{y:.3f}' for x,y in coords)+'"/>')
            svg.append('<text x="275" y="282" text-anchor="middle">SOL tokens/s/user (log scale)</text></svg>')
            label='MAI-Base-2 proxy' if model.startswith('MAI2') else model
            panels.append('<section><h3>'+label+'</h3><p>Concurrent decode requests per allocated chip</p>'+''.join(svg)+'</section>')
        return '<div class="grid">'+''.join(panels)+'</div>'
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Trainium: inference-capacity scenarios</title><style>body{font:16px system-ui;max-width:1150px;margin:36px auto;padding:0 24px;color:#24353c;background:#fcfcfa}p,li{line-height:1.55}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}svg{width:100%;font:12px system-ui}section{border:1px solid #ddd;padding:15px}a{color:#156a79}@media(max-width:750px){.grid{grid-template-columns:1fr}}code{overflow-wrap:anywhere}</style>',
        '<h1>Trainium: inference-capacity scenarios</h1><p>22 October 2026 · Separate research extension. Existing published scores and growth figures are unchanged.</p>',
        '<p>Dedicated roofline projections are materially above the H100-FLOPs fallback. The large scale-up domain explains much of the difference—not superior per-chip compute. Network scenarios bracket unresolved AWS specification conflicts; they are not confidence intervals or measured Neuron performance.</p>',
        '<h2>1. Scores and topology sensitivity</h2><p>Four equally weighted models, exact log-speed AUC from 50–400 SOL TPS/user. Combined scores provision separate prefill resources at the frozen 1.5133 new-input/output ratio; the decode and prefill columns are standalone diagnostics, not fixed-weight components.</p>',
        '<div class="scroll">'+table(['Chip / scenario','Combined GB300e','Decode','Prefill'],[[f'Trn{r["generation"]}: {r["scenario"]}',*[f'{float(r[k]):.3f}' for k in ('score','decode_score','prefill_score')]] for r in totals])+'</div>',
        '<p>The 8-chip controls prohibit larger instances and PP; they are size controls, not identical-topology comparisons to GPU deployments. The Trn3 144-chip slow-tier case applies 64 GB/s to every group wider than four as a pessimistic placement scenario. It is not evidence that increasing domain size intrinsically hurts performance.</p>',
        '<h2>2. Decode frontiers</h2><p>Black: GB300. Orange: Trn2 64-chip link-limited. Teal: Trn3 64-chip detailed tiers. Purple: Trn3 144-chip aggregate-half. These plots are decode-only; the score above also accounts for prefill resources.</p>',chart(),
        '<h2>3. Effect on the growth estimate</h2><p>These are conditional replacements of Trainium2/3 scores for all their shipped units. Trainium1 stays on its existing fallback. Deployment mix and deployment delays are not inferred. Blackwell remains at the shared 60% NVL72 central scenario.</p>',
        table(['Scenario','Q2 2025 (m GB300e)','Q2 2026 (m GB300e)','One-year growth','End capacity change'],[[r['scenario'],f'{r["capacity_q2_2025"]/1e6:.3f}',f'{r["capacity_q2_2026"]/1e6:.3f}',f'{r["annual_growth"]:.3f}×',f'{r["endpoint_change_pct"]:+.2f}%'] for r in impacts]),
        '<p>The starting stock already contains 775,000 Trainium2 chips; the ending stock contains 2,005,000 Trainium2 and 95,000 Trainium3 chips. Raising their scores therefore raises both endpoints and slightly reduces the annual growth ratio.</p>',
        '<h2>4. Method, assumptions and saved configurations</h2><p>See <a href="README.md">the detailed method and resource sanity check</a>, <a href="per_model.csv">per-model scores</a>, <a href="targets.csv">winning layouts at 50/100/200/400 TPS</a>, <a href="growth_impact.csv">growth sensitivity</a>, <a href="provenance.json">provenance</a> and <a href="validation.json">validation</a>.</p>',
        f'<p>Completed {unique_decode:,} unique decode and {unique_prefill:,} prefill candidates, with communication scenarios evaluated on the same candidates. Native overlap and unmodified MoE formula parity are checked before the Trainium3 byte-accounting correction. All selected requests fit the same 90% memory policy.</p></html>']
    (DOC/'index.html').write_text('\n'.join(body))
    print(json.dumps(impacts,indent=2));print(DOC/'index.html')


if __name__=='__main__':main()
