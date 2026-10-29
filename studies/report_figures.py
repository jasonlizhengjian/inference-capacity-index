"""Render report Figures 1–4 from versioned saved data; no simulations.

Run from any directory with Python + matplotlib (numpy is its dependency).
The score rebuild is offline and standard-library only.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'studies')]
import reproduce
import rubin_release
from agent_index.index import capacity
from log400_release import combined_area

OUT = ROOT/'results/report-figures'
OUT.mkdir(parents=True, exist_ok=True)
reproduce.verify()
rubin_release.main(OUT/'scores')
FROZEN_PATH = OUT/'scores/curves_and_inputs.json'
FROZEN = json.loads(FROZEN_PATH.read_text())
MODELS = FROZEN['models']
NAMES = dict(zip(MODELS, ['GLM-5.2', 'Kimi-K3', 'Qwen3-Coder 480B', 'MAI-Base-2 proxy']))
RATIO = FROZEN['ratio']
COL = {'GB300': '#344958', 'VR200': '#008b87'}
KEY = {'GB300':'GB300 NVL72', 'VR200':'VR200'}
INPUT = ROOT/'docs/report/inputs/kimi_gb300_fixed_layout.csv'
PIN = json.loads((INPUT.parent/'provenance.json').read_text())
assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == PIN['csv_sha256']
POINTS = [{k:float(v) if k in ('speed','capacity','memory_gib') else int(v)
           for k,v in row.items()} for row in csv.DictReader(INPUT.open())]
# Verify chip counting and timing parity for matching saved frontier points.
assert len({q['batch'] for q in POINTS}) == len(POINTS)
for q in POINTS:
    assert tuple(q[k] for k in ('tp','dp','dcp','ep','etp','pp')) == (8,8,8,64,1,1)
    assert q['chips']==q['tp']*q['dp']*q['pp'] and q['requests']==q['batch']*q['dp']
    assert math.isclose(q['capacity'],q['requests']/q['chips'])
for q in FROZEN['curves']['GB300 NVL72']['Kimi-K3']:
    if (q['tp'],q['dp'],q['dcp'],q['ep'],q['etp'],q['pp'])==(8,8,8,64,1,1):
        match=next(r for r in POINTS if r['batch']==q['batch'])
        assert math.isclose(match['speed'],q['speed'],rel_tol=1e-12)

def prefill(h,m):
    return next(p['tokens_chip_second'] for p in FROZEN['prefill_rates']
                if p['hardware']==KEY[h] and p['model']==m)
def curve(h,m): return FROZEN['curves'][KEY[h]][m]
def value(points,s,p):
    c=capacity(points,float(s))
    return c/(1+RATIO*s*c/p) if c else 0.
def area(h,m): return combined_area(curve(h,m),prefill(h,m),RATIO)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
    'axes.labelsize':11, 'axes.titlesize':13, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.spines.left':False,
    'axes.edgecolor':'#cbd1d4', 'axes.labelcolor':'#263540', 'text.color':'#263540',
    'xtick.color':'#263540', 'ytick.color':'#263540', 'figure.facecolor':'white',
    'axes.facecolor':'white', 'svg.fonttype':'none', 'savefig.dpi':300})

def grid(points,lo=50,hi=400):
    xs=list(np.geomspace(lo,hi,1600))+[lo,hi]
    for p in points:
        if lo<p['speed']<hi: xs.extend([p['speed'],np.nextafter(p['speed'],np.inf)])
    return sorted(set(xs))
def csvout(name,rows):
    with (OUT/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader();w.writerows(rows)
def style(ax,ylabel='Concurrent requests per total chip'):
    ax.set_xscale('log',base=2);ax.set_xticks([50,100,200,400],[50,100,200,400])
    ax.set_xlim(50,400);ax.set_ylim(bottom=0)
    ax.set_xlabel('Required output speed per request (tokens/s)',labelpad=12)
    ax.set_ylabel(ylabel,labelpad=10);ax.grid(axis='y',color='#e7eaec',lw=.7)
    ax.set_axisbelow(True);ax.tick_params(length=0,pad=7)
def heading(fig,title,sub):
    fig.text(.065,.955,title,fontsize=20,weight='bold',va='top')
    fig.text(.065,.885,sub,fontsize=11.5,va='top')
def foot(fig,text): fig.text(.065,.035,text,fontsize=10,linespacing=1.45,va='bottom')
def save(fig,name,pdf=None):
    for t in fig.findobj(matplotlib.text.Text): assert '·' not in t.get_text()
    svg = OUT/(name+'.svg')
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(OUT/(name+'.png'))
    fig.savefig(OUT/(name+'_preview.png'),dpi=100)
    fig.savefig(OUT/(name+'_article_preview.png'),dpi=900/fig.get_figwidth())
    if pdf is not None: pdf.savefig(fig)
    plt.close(fig)

def batch_panel(ax, decode_only=False):
    layout=[p for p in POINTS if (p['tp'],p['dp'],p['dcp'])==(8,8,8)]
    p=prefill('GB300','Kimi-K3')
    # Connect real operating points, including outside endpoints for correct clipping.
    ordered=sorted(layout,key=lambda q:q['speed'])
    x=[q['speed'] for q in ordered]
    y=[q['capacity'] if decode_only else value([q],q['speed'],p) for q in ordered]
    ax.plot(x,y,color=COL['GB300'],lw=1.1)
    ax.scatter(x,y,s=7,color=COL['GB300'],alpha=.25,zorder=3)
    for batch in [64,32,16]:
        q=next(x for x in layout if x['batch']==batch)
        y=q['capacity'] if decode_only else value([q],q['speed'],p)
        ax.scatter(q['speed'],y,s=44,color=COL['GB300'],edgecolor='white',zorder=4)
        label=(f'{q["requests"]/q["chips"]:g} requests/chip' if decode_only
               else f'Batch/chip: {q["requests"]/q["chips"]:g}')
        ax.annotate(label, (q['speed'],y),
                    xytext=(12,13),textcoords='offset points',fontsize=11)
    style(ax, 'Concurrency (requests/decode chip)' if decode_only else 'Concurrent requests per total chip')
    ax.set_ylim(0,12)
    ax.set_xlabel('Output speed (tokens/s/request)' if decode_only else 'Output speed per request (tokens/s)',labelpad=12)

def area_figure(ratios,pdf):
    fig,ax=plt.subplots(figsize=(10.8,8.8))
    fig.subplots_adjust(left=.10,right=.96,top=.79,bottom=.22)
    fig.text(.065,.965,f'VR200’s capacity curve has {ratios["Kimi-K3"]:.2f} times\nthe area of GB300’s',fontsize=20,weight='bold',va='top')
    fig.text(.065,.865,'Modeled Kimi-K3 capacity per chip, including prefill and decode.',fontsize=11.5,va='top')
    for h in ['VR200','GB300']:
        g=grid(curve(h,'Kimi-K3'))
        y=[value(curve(h,'Kimi-K3'),s,prefill(h,'Kimi-K3')) for s in g]
        ax.fill_between(g,y,color=COL[h],alpha=.14)
        ax.plot(g,y,lw=1.6,color=COL[h],label=h)
    style(ax,'Concurrency (requests/chip)');ax.legend(frameon=False,fontsize=12)
    ax.set_xlabel('Required output speed (tokens/s/request)',labelpad=12)
    ax.text(.97,.55,f'VR200 area ÷ GB300 area\n≈ {ratios["Kimi-K3"]:.2f}',transform=ax.transAxes,
            ha='right',fontsize=15,weight='bold',color=COL['VR200'],linespacing=1.5)
    foot(fig,'The logarithmic speed axis gives equal weight to each doubling: 50–100, 100–200,\nand 200–400 tokens/s/request.')
    save(fig,'figure_3_model_areas',pdf)

def average_figure(ratios,ici,pdf):
    fig,ax=plt.subplots(figsize=(10.8,8.6))
    fig.subplots_adjust(left=.255,right=.94,top=.78,bottom=.29)
    fig.text(.065,.965,f'Averaging across four models gives VR200\na score of {ici:.2f} GB300e',fontsize=20,weight='bold',va='top')
    fig.text(.065,.86,'Each model’s ratio compares the area under VR200’s capacity curve with GB300’s.',fontsize=11.5,va='top')
    positions=[0,1,2,3,5]
    vals=list(ratios.values())+[ici]
    labels=[NAMES[m] for m in MODELS]+['ICI score']
    for pos,v in zip(positions,vals):
        ax.barh(pos,v,color=COL['VR200'],alpha=.85 if pos==5 else .48,height=.58)
        ax.text(v+.045,pos,f'{v:.2f}',va='center',weight='bold' if pos==5 else 'normal',fontsize=13)
    ax.axvline(1,color=COL['GB300'],ls='--',lw=1)
    ax.axhline(4,color='#cbd1d4',lw=.8)
    ax.set_yticks(positions,labels);ax.invert_yaxis();ax.set_xlim(0,3)
    ax.set_xlabel('Capacity relative to GB300 (GB300 = 1)',labelpad=12)
    ax.grid(axis='x',color='#e7eaec');ax.set_axisbelow(True);ax.tick_params(length=0,pad=8)
    fig.text(.255,.16,'('+ ' + '.join(f'{ratios[m]:.2f}' for m in MODELS)+f') ÷ 4 ≈ {ici:.2f}',fontsize=15,weight='bold')
    save(fig,'figure_4_model_average',pdf)


ratios={m:area('VR200',m)/area('GB300',m) for m in MODELS}
ici=sum(ratios.values())/4
assert math.isclose(ici,2.177366570368923,rel_tol=1e-12)
fig,ax=plt.subplots(figsize=(10.8,8))
fig.subplots_adjust(left=.10,right=.96,top=.78,bottom=.14)
heading(fig,'Serving more requests slows output for each request',
        'Modeled Kimi-K3 decode performance on GB300, varying batch size within one configuration.')
batch_panel(ax,decode_only=True)
save(fig,'figure_2_decode_tradeoff')
csvout('figure_2_decode_tradeoff',[dict(batch_per_replica=q['batch'],decode_chips=q['chips'],
    concurrent_requests=q['requests'],output_tokens_per_second_per_request=q['speed'],
    requests_per_decode_chip=q['capacity'],output_tokens_per_second_per_decode_chip=q['capacity']*q['speed']) for q in POINTS])
area_figure(ratios,None)
average_figure(ratios,ici,None)
csvout('figure_3_capacity_curves',[dict(hardware=h,required_output_tokens_per_second_per_request=s,
    requests_per_total_chip=value(curve(h,'Kimi-K3'),s,prefill(h,'Kimi-K3')))
    for h in ('GB300','VR200') for s in grid(curve(h,'Kimi-K3'))])
csvout('figure_4_model_ratios',[dict(model=NAMES[m],GB300_log_capacity_area=area('GB300',m),
    VR200_log_capacity_area=area('VR200',m),VR200_area_ratio=ratios[m],model_weight=.25,overall_ICI_GB300e=ici) for m in MODELS])

SOURCE=OUT/'scores/index.csv'
SELECTION = {
    'MI455X': 'AMD MI455X',
    'VR200': 'Nvidia VR200',
    'TPU8i · corrected · hops': 'Google TPU v8i',
    'GB300 NVL72': 'Nvidia GB300',
    'GB200 NVL72': 'Nvidia GB200',
    'TPU7x Ironwood · FP8': 'Google TPU v7',
    'Maia 200 rack': 'Microsoft Maia 200',
    'MI355X · 400G': 'AMD MI355X',
    'B300 · 800G': 'Nvidia B300',
    'H200 SXM · FP8 · 400G': 'Nvidia H200',
}
rows = []
for q in csv.DictReader(SOURCE.open()):
    if q['hardware'] not in SELECTION:
        continue
    rows.append(dict(chip=SELECTION[q['hardware']],
                     ici_relative_to_gb300=float(q['agent_index_GB300e']),
                     hardware_profile=q['hardware'],
                     score_snapshot='2026-10-20'))
rows.sort(key=lambda q: q['ici_relative_to_gb300'], reverse=True)
assert len(rows) == 10
assert next(q['ici_relative_to_gb300'] for q in rows if q['chip']=='Nvidia GB300') == 1

fig, ax = plt.subplots(figsize=(11,8.2))
fig.subplots_adjust(left=.30,right=.94,top=.78,bottom=.22)
heading(fig, 'Inference Capacity Index v1',
          'Modeled inference capacity per chip, relative to Nvidia GB300.')
ys=list(range(len(rows)))
for i,q in enumerate(rows):
    color = '#008b87' if q['chip']=='Nvidia VR200' else '#344958' if q['chip']=='Nvidia GB300' else '#8b9fae'
    ax.barh(i,q['ici_relative_to_gb300'],height=.62,color=color,
            edgecolor='white',linewidth=.6,zorder=3)
    ax.text(q['ici_relative_to_gb300']+.055,i,f'{q["ici_relative_to_gb300"]:.2f}',
            va='center',fontsize=11.5,color='#263540',weight='bold' if q['chip'] in ['Nvidia GB300','Nvidia VR200'] else 'normal')
ax.set_yticks(ys,[q['chip'] for q in rows],fontsize=11.5)
ax.invert_yaxis()
ax.set_xlim(0,3.25)
ax.set_xticks([0,.5,1,1.5,2,2.5,3],['0','0.5','1','1.5','2','2.5','3'])
ax.set_xlabel('Inference Capacity Index (GB300e)',labelpad=12)
ax.grid(axis='x',color='#e7eaec',lw=.7,zorder=0)
ax.set_axisbelow(True)
ax.axvline(1,color='#344958',lw=1,ls=(0,(3,3)),zorder=2)
ax.text(1,1.025,'GB300 = 1',transform=ax.get_xaxis_transform(),ha='center',fontsize=10,color='#344958')
ax.tick_params(length=0,pad=8)
ax.spines['bottom'].set_visible(False)
foot(fig, 'Includes prefill and decode across four models at output speeds of 50–400 tokens/s/request.')
save(fig,'figure_1_selected_chips')
csvout('figure_1_selected_chips',rows)

inputs=[ROOT/'reference/2026-10-15/manifest.json',ROOT/'reference/2026-10-20-rubin/manifest.json',INPUT,Path(__file__)]
provenance=dict(score_revision='2026-10-20-rubin',methodology='ICI v1, four models, log 50–400, separate prefill/decode provisioning',
    input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ('.csv','.png','.svg')},
    matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,
    VR200_ICI=ici,VR200_Kimi_ratio=ratios['Kimi-K3'],timings_rerun=False,
    note='Curve CSV samples are for plotting; scores use exact saved-breakpoint integration.')
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Report figures saved to',OUT)
