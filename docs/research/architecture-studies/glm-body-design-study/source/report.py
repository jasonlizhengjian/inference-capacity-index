"""Offline fixed-attention architecture study with controlled comparisons."""
import collections
import csv
import hashlib
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'output'
def read(name):
    rows=list(csv.DictReader((OUT/name).open()))
    for r in rows:
        for k,v in r.items():
            if v in ('True','False'):r[k]=v=='True'
            else:
                try:r[k]=float(v)
                except ValueError:pass
    return rows
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def cfg(r):return f"EP{int(r['world'])}/TP{int(r['tp'])}, c/group={int(r['cpr'])}, draft={int(r['nextn'])}"
def short(d):return f"{d['target_t']}T L{d['layers']} H{d['hidden']} M{d['expert_width']} E{d['experts']} K{d['active_experts']} S{d['shared_experts']}"
def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    defs=json.loads((HERE/'definitions.json').read_text());by={d['id']:d for d in defs};grid=read('grid.csv');slo=read('slo.csv');limits=read('memory_limits.csv');overlap=read('overlap_sensitivity.csv')
    assert len(limits)==len(defs)*2*3*4*3,'Incomplete search'
    assert set(r['id'] for r in limits)==set(by),'Stale model definitions'
    assert all(r['admit90'] and r['world']==r['dep']*r['tp'] for r in grid)
    groups=collections.defaultdict(list);sg=collections.defaultdict(list)
    for r in grid:groups[r['id'],r['system'],r['phase'],r['nextn']].append(r)
    for r in slo:sg[r['id'],r['system'],r['efficiency'],r['target']].append(r)
    best=[]
    for d in defs:
        for hw in ('GB300','VR200-dense'):
            for eff,target in ((1,50),(1,100),(.7,50),(.7,100)):
                for spec in (False,True):
                    q=[r for r in sg[d['id'],hw,eff,target] if bool(r['nextn'])==spec]
                    if not q:continue
                    r=max(q,key=lambda r:(r['requests_64gpu'],r['throughput_gpu']))
                    best.append(dict(id=d['id'],family=d['family'],system=hw,efficiency=eff,target=target,spec=spec,world=r['world'],tp=r['tp'],cpr=r['cpr'],nextn=r['nextn'],
                        max_requests64=r['requests_64gpu'],throughput_gpu=r['throughput_gpu']*eff,tps_user=r['tps_user']*eff,
                        max_memory_requests64=64*r['max_cpr']/r['tp'],memory_gib=r['memory_gib']))
    write('best_decode.csv',best)
    parts=['''<h1>GLM-style attention, varied model body</h1><p>87 controlled architecture variants · ~1–10T capacity anchors; main5T/10T designs · GB300 and Rubin proxy · September2026</p>
<div class="callout"><b>September9 extension:</b> <a href="../../glm-attention-dimension-study/output/GLM_ATTENTION_DIMENSIONS.html">Internal attention dimensions and coordinated scale-ups</a> adds58 variants and corrects decode index-reuse accounting. This page retains the older formula and results. Its0.7 efficiency scenario should no longer be read as a validated calibration for the corrected formula.</div>
<div class="callout"><b>Question:</b> with the attention design fixed, which costs come from expert activation, expert granularity, stored capacity, shared experts and sequential depth? These are inference-performance comparisons, not predictions of model quality or of proprietary architectures.</div>
<h2>What is fixed and what changes</h2>
<p>All variants use GLM DSA top2048,64 attention heads, fixed Q/KV low-rank and head dimensions, index reuse at the native four-layer frequency, and FP8 KV. Most controlled comparisons keep residual width6144 and depth78. When residual width changes, the attention input/output projections change too; when depth changes, the same attention block is repeated more times. Attention algorithm and inner dimensions are fixed, not total attention work.</p>
<p>Every request has262,144 cached tokens. Decode has no prefill work. The prefill supplement adds exactly1,024 new tokens. Routed experts use NVFP4; non-expert GEMMs use BF16 uniformly across variants. This consistent synthetic policy is not an exact copy of every native checkpoint's per-layer quantization exception. Model totals below are parameter-count estimates excluding small normalization/bias terms; targets refer to approximate routed-parameter budgets.</p>
<p>Hardware: inherited GB3008TB/s HBM, Rubin22TB/s; directional fabric0.9/1.8TB/s; Rubin dense compute proxy, not adaptive headline FLOPS. Memory admission uses90% of modeled capacity. The64-GPU budget is tiled with16/32/64-GPU worker groups. TP1/2/4/8 is searched; independent attention groups=world/TP. Mixed TP+DP uses AIC's SGLang analytical graph because its TRT-LLM selector rejects that layout; TP1 uses TRT-LLM. Mixed layouts are less directly grounded than the measured pure-attention-DP anchor.</p>
<p>Main charts are <b>SOL</b>. A separate0.7 efficiency scenario is provided, not silently applied. Speculation is explicitly a four-proposal/AL3.36 scenario borrowed from the earlier DSpark study, not a measured GLM acceptance forecast. The native calibration used three-proposal MTP; its efficiency must not be assumed invariant under this change.</p>
<h2>Controlled axes</h2>''']
    parts.append(table(['Family','Held fixed','Varied / interpretation'],[
        ['sparsity','Stored routed model, depth, width, expert size','K=8/16/32/64/128; active budget changes deliberately'],
        ['granularity','Exact routed total and active parameter budgets, depth/width; shared experts=0','M=1024/2048/4096/8192, E and K inversely scaled; router and fanout may differ'],
        ['geometry','Approximately fixed routed total and routed activation fraction1/32; fixed E per target','L=48/64/78/96/128 × H=4096/6144/8192/12288; expert M solves the budget, rounded128'],
        ['shared','Routed model, K/E, attention geometry','0/1/2/4 shared experts; adds always-active work'],
        ['capacity_fixed_active','Depth/width/expert size and K=8','Increase E to~1/3/5/10T; stored capacity grows without proportional active work'],
        ['capacity_fixed_ratio','Depth/width/expert size, approximately K/E=1/32','E and K grow together; stored and active budgets both grow']]))
    parts.append('<p><b>Implementation audit:</b> the native AIC GLM graph assumes one shared expert regardless of the config count. This study explicitly scales shared-expert operation cost and weight memory by S in a temporary adapter, restoring the original constructor after each query. S=0 genuinely removes shared work. The graph assumes ideal overlap of shared and routed branches; therefore shared latency can be hidden even while shared weight memory grows. All synthetic layers are MoE (first_k_dense_replace=0), matching the graph actually evaluated; this is not a claim that native GLM has no initial dense layers.</p>')
    parts.append('''<h2>Principal findings</h2><ul>
<li><b>Active parameters are not a sufficient decode-cost predictor.</b> For~5T at EP64 and4 requests/rank, moving K128→K8 reduces estimated active parameters from396B to42B, but improves GB300 decode only90→112 TPS/user (1.25×). At1 request/rank the improvement is104→187 (1.79×). At a global batch of256, K8 already touches about70% of1728 experts in expectation; high K touches almost all. Attention remains unchanged.</li>
<li><b>Expert granularity is a smaller SOL effect in this regime.</b> With exactly matched routed total and active budgets and no shared experts, M1024→M8192 improves GB300's four-requests/rank rate about91→97 TPS/user (7%). This does not establish equal real kernel efficiency for tiny and large experts.</li>
<li><b>Depth/width can move performance much more.</b> The~5T L48/H4096 geometry yields122 TPS/user versus93 at L78/H6144 and51 at L128/H12288, at fixed EP64 and4 requests/rank. Routed budgets are approximate, not exact:128-wide expert rounding can move capacity several percent; inspect manifest totals before treating a pair as strictly parameter-matched.</li>
<li><b>Capacity-only scaling is cheaper than proportional-active scaling.</b> At K8,~5T→~10T changes112→98 GB300 TPS/user. Keeping K/E near1/32 changes92→60. Capacity-only scaling still loses throughput because weight reuse falls and cache headroom shrinks.</li>
<li><b>Rubin improves absolute rates but does not remove these tradeoffs.</b> At the same fixed four-requests/rank shape, the native-like~5T and~10T cases give about250 and164 TPS/user, compared with92 and60 on GB300. These near-bandwidth-ratio gains are SOL projections, not measured Rubin efficiency.</li>
<li><b>Faster memory need not increase target-feasible concurrency when capacity already binds.</b> With the assumed speculation, the native-like~5T GB300 case reaches704 requests at96 TPS/user; Rubin reaches640 at271. The lower modeled Rubin count comes from the inherited conservative decimal288GB capacity assumption versus GB300's driver-visible277.55GiB, not a demonstrated real hardware capacity regression. In this regime bandwidth mainly buys faster users. Consult no-spec and0.7 scenarios separately.</li></ul>''')
    parts.append('''<h2>Fixed-layout, fixed-concurrency decode comparisons</h2><p>EP64, attention TP1, four requests per attention rank, <b>no speculation</b>. These isolate architectural changes without changing serving parallelism or acceptance assumptions. Hardware curves use the same configuration.</p>''')
    def fixed(d,h):
        return next((r for r in groups[d['id'],h,'decode',0] if r['world']==64 and r['tp']==1 and r['cpr']==4),None)
    def plot_family(family,xkey,xlabel,targets):
        for target in targets:
            ds=sorted((d for d in defs if d['family']==family and d['target_t']==target),key=lambda d:d[xkey])
            fig,ax=plt.subplots(figsize=(8.8,4.2))
            for hw,color in [('GB300','#bf6413'),('VR200-dense','#008774')]:
                q=[(d,fixed(d,hw)) for d in ds];q=[(d,r) for d,r in q if r]
                ax.plot([d[xkey] for d,r in q],[r['tps_user'] for d,r in q],'o-',color=color,label=hw)
            ax.set(xlabel=xlabel,ylabel='SOL output tokens/s/user',title=f'{family}: ~{target}T, EP64, 4 requests/rank, no speculation');ax.grid(alpha=.2);ax.legend();fig.tight_layout()
            name=f'{family}_{target}.svg';fig.savefig(OUT/name)
            if family=='granularity' and target==5:fig.savefig(OUT/'preview.png')
            plt.close(fig);parts.append(f'<img src="{name}" alt="{family} {target}T comparison">')
    plot_family('sparsity','active_experts','Active routed experts per MoE layer (K)',(5,10))
    plot_family('granularity','expert_width','Expert intermediate width (M); E and K inversely scaled',(5,10))
    plot_family('shared','shared_experts','Always-active shared experts per MoE layer',(5,10))
    parts.append('<h3>Shared experts are not free: overlap sensitivity</h3><p>The flat shared-expert curves above result from AIC taking max(routed branch, shared branch). That can hide shared execution time but not its memory footprint, and it does not model contention when both branches need HBM. The table compares ideal overlap with serialized branches at EP64,4 requests/rank,no speculation. Serialized execution is a sensitivity, not a prediction that the backend necessarily serializes them.</p>'+table(['Target','Hardware','Shared experts','Ideal-overlap ms','Serialized ms','Serialized throughput / ideal','GiB/rank'],[[r['target_t'],r['system'],int(r['shared_experts']),f"{r['ideal_overlap_ms']:.2f}",f"{r['serialized_ms']:.2f}",f"{r['throughput_ratio']:.3f}",f"{r['memory_gib']:.1f}"] for r in overlap if r['family']=='shared' and r['target_t'] in (5,10) and r['cpr']==4 and r['nextn']==0]))
    parts.append('<h2>How the sparsity benefit changes with concurrency</h2><p>Same EP64/TP1 layout and no speculation; each curve stops at its own admitted memory limit. K changes while stored capacity stays fixed. The benefit of low K can narrow as the batch collectively touches most experts, even though per-token active parameters remain different.</p>')
    for target in (5,10):
        for hw in ('GB300','VR200-dense'):
            fig,axes=plt.subplots(1,2,figsize=(11,4.2))
            for k,color in [(8,'#008774'),(32,'#3676b3'),(128,'#ba6317')]:
                d=next(d for d in defs if d['family']=='sparsity' and d['target_t']==target and d['active_experts']==k)
                rr=sorted((r for r in groups[d['id'],hw,'decode',0] if r['world']==64 and r['tp']==1),key=lambda r:r['cpr'])
                for ax,field in zip(axes,('tps_user','throughput_gpu')):
                    ax.plot([r['cpr'] for r in rr],[r[field] for r in rr],'o-',color=color,label=f'K={k}',markersize=3)
                    ax.set(xscale='log',xlabel='Concurrent requests / attention rank');ax.grid(alpha=.2);ax.legend()
            axes[0].set_ylabel('SOL output TPS/user');axes[1].set_ylabel('SOL output TPS/GPU')
            fig.suptitle(f'~{target}T · {hw} · EP64/TP1 · no speculation');fig.tight_layout();name=f'sparsity_batch_{target}_{hw}.svg';fig.savefig(OUT/name);plt.close(fig);parts.append(f'<img src="{name}" alt="Sparsity versus concurrency">')
    parts.append('<h2>Depth versus width, fixed approximate routed budget</h2><p>Each cell is SOL TPS/user in the same EP64,4-requests/rank,no-spec configuration. E and K are fixed within each target; expert width changes to hold the routed budget. Thus this is a controlled body-shape tradeoff, not depth-only scaling.</p>')
    for t in (5,10):
        for hw in ('GB300','VR200-dense'):
            ds=[d for d in defs if d['family']=='geometry' and d['target_t']==t]
            parts.append(f'<h3>~{t}T · {hw}</h3>'+table(['Layers / hidden width','4096','6144','8192','12288'],[[l]+[f"{fixed(next(d for d in ds if d['layers']==l and d['hidden']==h),hw)['tps_user']:.1f}" if fixed(next(d for d in ds if d['layers']==l and d['hidden']==h),hw) else 'Not admitted' for h in (4096,6144,8192,12288)] for l in (48,64,78,96,128)]))
    parts.append('<h2>Capacity-only versus proportional-active scaling</h2><p>Same fixed layout and concurrency. Distinguishes cost of storing more experts from cost of activating more parameters.</p>')
    rows=[]
    for d in defs:
        if not d['family'].startswith('capacity_'):continue
        rows.append([d['family'],f"{d['approximate_total_t']:.2f}T",d['experts'],d['active_experts'],f"{d['approximate_active_b']:.1f}B"]+[f"{fixed(d,h)['tps_user']:.1f}" if fixed(d,h) else 'Not admitted' for h in ('GB300','VR200-dense')])
    parts.append(table(['Scaling','Approx total','E','K','Approx active','GB300 TPS/user','Rubin TPS/user'],rows))
    parts.append('''<h2>Best concurrency at50/100 TPS/user</h2><p>Each row independently optimizes worker size and TP over the searched layouts. Concurrency is total requests across64 GPUs. Memory-max is for the selected layout, not the architecture's largest possible cache allocation. The expandable table contains all architectures, no-spec/spec variants and SOL/0.7 scenarios. Search boundaries are checked at adjacent integer concurrency values.</p>''')
    selected=[]
    for t in (5,10):
        for label,family,conditions in [('Native-like activation ratio','capacity_fixed_ratio',{}),('Moderately shallower','geometry',{'layers':64,'hidden':6144}),('Aggressive shallow/narrow','geometry',{'layers':48,'hidden':4096}),('Aggressive low active budget','sparsity',{'active_experts':16})]:
            d=next(d for d in defs if d['target_t']==t and d['family']==family and all(d[k]==v for k,v in conditions.items()))
            for hw in ('GB300','VR200-dense'):
                row=[label,short(d),f"{d['approximate_total_t']:.2f}T / {d['approximate_active_b']:.1f}B",hw]
                for spec,target in ((False,50),(True,50),(True,100)):
                    r=next((r for r in best if r['id']==d['id'] and r['system']==hw and r['efficiency']==1 and r['spec']==spec and r['target']==target),None)
                    row.append('No feasible point' if r is None else f"N={int(r['max_requests64'])}; EP{int(r['world'])}/TP{int(r['tp'])}; c/group={int(r['cpr'])}; memory Nmax={int(r['max_memory_requests64'])}")
                selected.append(row)
    parts.append('<h3>Candidate operating points to investigate</h3><p>These are chosen architectural hypotheses, not quality-ranked winners. “Native-like” refers to activation fraction and body shape, not a known trained5T/10T checkpoint. N is total concurrency across64 GPUs. Spec columns use the assumed DSpark acceptance, and all columns are SOL.</p>'+table(['Direction','Architecture','Approx total / active','Hardware','50 TPS no spec','50 TPS assumed spec','100 TPS assumed spec'],selected))
    parts.append('<details><summary>Show all optimized decode operating points</summary>'+table(['Architecture','Hardware','Efficiency','Target','Spec','EP/TP','Requests/group','Requests /64 GPUs','TPS/user','Output TPS/GPU','Memory-max requests','GiB/rank'],[[short(by[r['id']]),r['system'],r['efficiency'],r['target'],r['spec'],f"{int(r['world'])}/{int(r['tp'])}",int(r['cpr']),int(r['max_requests64']),f"{r['tps_user']:.1f}",f"{r['throughput_gpu']:.1f}",int(r['max_memory_requests64']),f"{r['memory_gib']:.1f}"] for r in best])+'</details>')
    parts.append('<h2>Prefill supplement</h2><p>Best sampled aggregate throughput for1K new tokens after256K cached. EP/TP layouts and batches through memory capacity are searched; <b>PCP is not included</b>, so this is not a new proof of globally optimal prefill. Speculation is off. Larger batches may add latency for negligible throughput.</p>')
    pf=[]
    for d in defs:
        for hw in ('GB300','VR200-dense'):
            q=groups[d['id'],hw,'prefill',0]
            if not q:continue
            peak=max(q,key=lambda r:r['throughput_gpu']);near=min((r for r in q if r['throughput_gpu']>=.95*peak['throughput_gpu']),key=lambda r:r['step_ms'])
            pf.append(dict(id=d['id'],system=hw,peak_input_tps_gpu=peak['throughput_gpu'],near_input_tps_gpu=near['throughput_gpu'],near_job_ms=near['step_ms'],near_config=cfg(near),requests64=near['requests_64gpu']))
    write('best_prefill.csv',pf)
    parts.append('<details><summary>Show prefill results</summary>'+table(['Architecture','Hardware','Peak input TPS/GPU','Near-peak input TPS/GPU','Near-peak1K job ms','Configuration','Requests /64 GPUs'],[[short(by[r['id']]),r['system'],f"{r['peak_input_tps_gpu']:.0f}",f"{r['near_input_tps_gpu']:.0f}",f"{r['near_job_ms']:.1f}",r['near_config'],int(r['requests64'])] for r in pf])+'</details>')
    parts.append('''<h2>How to interpret the mechanisms</h2><ul>
<li>Expected distinct expert reads follow E×[1−(1−K/E)^T], where T is tokens participating in the routed step. The model does not assume every token activates disjoint experts until exhaustion.</li>
<li>At low load, more/smaller experts can increase distinct destinations and router work even when active parameter FLOPS match. At high load, most expert weights are already touched; expert granularity then mainly changes kernel shape and communication. Real small-expert kernel penalties are not fully represented by SOL.</li>
<li>Reducing K lowers active expert work but leaves attention and other always-active work intact. Its benefit must plateau when those terms dominate.</li>
<li>Reducing depth removes sequential attention, projection and communication work; the budget-preserving larger experts add parallelizable work instead. This is not free model quality: training behavior and reasoning capability are not modeled.</li>
<li>Increasing E at fixed K can still increase expert-weight traffic across a batch because reuse falls, and it consumes memory that could hold more requests. Fixed active parameters do not guarantee fixed inference cost.</li>
<li>Shared experts are always active; they are replicated or sharded differently from routed experts. Their memory/bandwidth costs can matter even when their parameter share is small.</li></ul>
<h2>Confidence and boundaries</h2><p>The empirical GLM anchor was native-model GB300 attention-DP8, roughly10 live requests/rank and heterogeneous~163K mean context. It supported about0.69× SOL with a padding-aware approximation. This study changes capacity, EP size, expert shapes, depth, width, context and speculative width. Therefore0.7 is a scenario, not a newly validated efficiency coefficient. Rubin has no measured anchor here.</p>
<p>Quality is not held constant. The lowest-active, shallowest design is not automatically the most plausible frontier model. Shared-expert count, expert fragmentation, top-K router behavior and trainability may impose constraints not visible to AIC. Treat the fixed-ratio/native-like expert-width cases as more conservative architectural baselines and the very low-K/very shallow cases as aggressive performance hypotheses.</p>
<p>Routing uses the existing expected-occupancy analytical extension and includes modeled communication; it is not a fully calibrated rank-skew, kernel-startup or topology-contention simulator. Memory ceilings assume balanced placement. Sparse attention input/output projection dimensions are updated with H; head geometry and index selection remain fixed. Parameter counts include router, shared/dense MLP, embeddings and attention estimates but omit small terms.</p>
<h2>Full architecture manifest and data</h2>''')
    parts.append('<details><summary>Show all87 architecture definitions</summary>'+table(['ID','Family','L','H','Expert M','Total E/layer','Active K/layer','Shared','K/E','Routed total T','Routed active B','Approx total T','Approx active B'],[[d['id'],d['family'],d['layers'],d['hidden'],d['expert_width'],d['experts'],d['active_experts'],d['shared_experts'],f"{100*d['sparsity_fraction']:.3f}%",f"{d['routed_total_t']:.3f}",f"{d['routed_active_b']:.2f}",f"{d['approximate_total_t']:.3f}",f"{d['approximate_active_b']:.2f}"] for d in defs])+'</details>')
    files=['grid.csv','slo.csv','memory_limits.csv','best_decode.csv','best_prefill.csv','overlap_sensitivity.csv','manifest.json']
    parts.append('<ul>'+''.join(f'<li><a href="{n}">{n}</a></li>' for n in files)+'</ul><p><a href="../definitions.json">Architecture definitions</a> · <a href="../study.py">Sweep code</a> · <a href="../prepare.py">Configuration generator</a> · <a href="../../inferencex-agentx-grounding/output/MATCHED_DECODE_CALIBRATION.html">Native GLM calibration</a>. Full AIC operator ledgers are retained in aic_cache.jsonl.</p>')
    css='body{font:16px/1.55 system-ui;max-width:1450px;margin:35px auto;padding:0 24px;background:#f7fafb;color:#183043}h2{margin-top:38px}.callout{padding:20px;background:#e0f1ed;border-left:5px solid #008774}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:13px}td,th{padding:9px;border-bottom:1px solid #dae4ea;text-align:left}th{background:#e5eef2}img{max-width:100%;display:block;margin:22px auto}summary{cursor:pointer;font-weight:600}a{color:#007866}'
    (OUT/'GLM_BODY_DESIGN_STUDY.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GLM architecture body study</title><style>'+css+'</style><main>'+''.join(parts)+'</main></html>')
    sources=[HERE/'prepare.py',HERE/'study.py',HERE/'report.py',HERE/'overlap_sensitivity.py',HERE/'definitions.json',HERE.parent/'frontier-hardware-study/engine.py',HERE.parent/'frontier-hardware-study/output/hardware.json']+[OUT/f for f in files if f!='manifest.json']+[Path(d['model'])/'config.json' for d in defs]
    (OUT/'manifest.json').write_text(json.dumps({'models':len(defs),'grid_rows':len(grid),'slo_rows':len(slo),'layout_limits':len(limits),'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
    print(OUT/'GLM_BODY_DESIGN_STUDY.html')
if __name__=='__main__':main()
