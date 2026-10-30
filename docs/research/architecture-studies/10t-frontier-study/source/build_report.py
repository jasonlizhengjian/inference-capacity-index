"""Offline HTML report, paired 5T comparisons, provenance and validation."""
import collections
import csv
import hashlib
import html
import json
import math
import re
import statistics
import subprocess
from html.parser import HTMLParser
import study as st
import report as util
import matplotlib.pyplot as plt
P=st.OUT;e=st.e
util.P=P

def main():
    data=util.read('architecture_grid.csv');bounds=util.read('capacity_slo.csv');sat=util.read('prefill_saturation.csv')
    native=util.read('native_geometry_grid.csv');topology=util.read('topology_spec.csv')
    resources=util.read('resource_counterfactuals.csv');ranks=util.read('rank_balance.csv')
    definitions=json.loads((P/'definitions.json').read_text());by={d['aid']:d for d in definitions}
    meta=json.loads((P/'metadata.json').read_text());table=util.table
    # Match the exact old geometry and fixed concurrency, not an unrelated best model.
    oldpath=st.HERE.parent/'frontier-hardware-study/output'
    old={}
    for filename in ('unified_results.csv','parallelism_screen.csv'):
        for r in csv.DictReader((oldpath/filename).open()):
            old[(r['aid'],r['system'],int(r['dep']),r['phase'],int(r['cpr']))]=r
    matched=[]
    for r in data:
        a=old.get((r['parent_aid'],r['system'],r['dep'],r['phase'],r['cpr']))
        if a:
            matched.append(dict(aid=r['aid'],parent_aid=r['parent_aid'],scaling=r['scaling'],system=r['system'],dep=r['dep'],phase=r['phase'],cpr=r['cpr'],
                latency_5t_ms=float(a['latency_ms']),latency_10t_ms=r['latency_ms'],latency_ratio=r['latency_ms']/float(a['latency_ms']),
                memory_5t_gib=float(a['memory_gib']),memory_10t_gib=r['memory_gib'],admit90_5t=a['admit90']=='True',admit90_10t=r['admit90']))
    e.write_csv(P/'matched_5t_10t.csv',matched)
    sections=[]
    def add(t,b):sections.append((t,b))
    add('Scope and the two distinct 10T questions',f'''<p>This extends the <a href="../../frontier-hardware-study/output/FRONTIER_INFERENCE_STUDY.html">prior frontier study</a> with {meta['controlled_architectures']} controlled 10T variants and {meta['native_geometries']} native-graph/width geometries. All simulations are CPU-only AIC analytical estimates, not measured hardware.</p>
    <div class="cards"><div><b>Fixed active work</b><p>Double routed expert count, keep top-k, expert size and layer count unchanged. Routed capacity doubles; active routed parameters stay approximately fixed. Sparsity increases.</p></div><div><b>Preserved sparsity ratio</b><p>Double expert count <em>and</em> top-k. Routed capacity and active routed work both double. Layers, residual width and attention are unchanged.</p></div><div><b>Width/native scaling</b><p>Keep the previous 10T Kimi expert/trend anchors; add 10T DS/GLM and normalized controls. Native width geometries cover 7,168/12,288/16,384 residual width, 64/96 layers and ~100/200/400B active routed budgets.</p></div></div>
    <p>The 982 controlled variants cover all 448 previous MoE × attention combinations and all 43 SWA/hybrid designs under both scaling rules. “10T” is an approximate expert budget, not an assertion of exactly equal whole-model parameter counts or equal model quality. Controlled routed budgets are approximately 9.8T; actual counts are preserved in definitions.json. Attention, shared experts and dense parameters add on top. Depth-only scaling is not used as the main proposal.</p>
    <p><b>Workloads:</b> decode with an existing independent 256K cache per request, and prefill of 1K new input tokens after that cache. Cached-prefix prefill does not recompute the prefix or assume users share caches. Main curves/tables are <b>non-speculative</b>; speculative decode has its own section.</p>''')
    add('Main findings', '''<ul><li><b>10T capacity does not imply twice the latency.</b> For the 61-layer balanced controls on GB300, DEP64 and c/r=1, fixed-active scaling changes 1K prefill latency by only ~0.1–0.2% in the KDA/MLA and V4 examples. Ratio-preserved scaling increases it by ~35% and ~60% respectively. Both examples fit the memory budget at that concurrency.</li><li><b>Fixed active parameters do not guarantee unchanged decode.</b> At c/r=4, those same fixed-active examples increase decode latency by ~45–52%, because routing across a larger expert population reduces weight reuse. The exact penalty depends on concurrency and attention cost.</li><li><b>Pure DEP fit is not rack fit.</b> The 10T Kimi-trend anchor requires ~288.4 GiB/rank at pure DEP64, but ~126.1 GiB/rank with attention TP8 + DP8 in the same 64 GPUs at c/r=1. MLA cache remains replicated across TP in this AIC model; the reduction is not fictitious DCP.</li><li><b>Concurrency remains an SLO question.</b> More bandwidth helps latency, but weights consume more of the cache budget at 10T. Compare memory-full TPS and the 50/100-TPS boundary separately, using the tables below.</li></ul><p>These are conditional strategy-specific SOL estimates with analytical routing assumptions, not measurements or predictions of trained-model quality.</p>''')
    add('Hardware and interpretation', '''<p>Same hardware assumptions as the audited 5T study: GB300; GB300 with only Rubin HBM+fabric rates; and a Rubin dense-spec proxy. HBM-only is also retained in native routing sensitivity data.</p>'''+table(['Scenario','HBM TB/s','Directional NVLink TB/s','BF16 PF/s','FP8 PF/s','Dense FP4 PF/s','GiB capacity'],[
        ['GB300',8,.9,2.5,5,15,277.546875],['HBM+fabric',22,1.8,2.5,5,15,277.546875],['VR200-dense',22,1.8,4,17.5,35,288e9/2**30]])+'''<p>The Rubin proxy uses published <em>dense</em> tensor rates, not 50 PF/s adaptive-compression headline inference performance. It still inherits AIC GB300 kernel structure and unspecified device parameters; it is not a full native Rubin microarchitectural simulation. <a href="https://www.nvidia.com/en-gb/data-center/vera-rubin-nvl72/">NVIDIA specifications</a>, checked 2026-09-08; preliminary and subject to change.</p>
    <p>Admission is at 90% modeled capacity, including weights, cache and modeled runtime overhead. GB300 uses AIC driver-visible capacity while Rubin conservatively treats advertised 288 GB as decimal. Ratios near one can depend on that capacity convention; it is not evidence Rubin has less physical memory. No CPU offload, shared prefixes or invented larger rack is credited.</p>
    <p>Pure DEP screens use 16/32/64 GPUs. Mixed attention TP screens also stay ≤64 GPUs, hence within the 72-GPU limit. We do not claim an exhaustive 72-rank ragged-expert optimum. At a fixed 64-GPU budget, DEP16 means four replicas and DEP32 means two replicas. Concurrency/rank means per <em>attention-DP</em> rank; for TP&gt;1 this is a multi-GPU group.</p>''')
    parents=['factorial__direct_L61_balanced_m3072_s1__'+x for x in ('mla100','kda75_mla25','kda75_dsa25','dsa2048','v4_compressed','kda100')]+['swa__common_r5_w512','swa__swa5_dsa']
    short={p:('61L balanced '+p.split('__')[-1] if p.startswith('factorial') else p.split('__')[-1]) for p in parents}
    counts=[]
    for p in parents:
        for scaling in ('fixed_active','ratio_preserved'):
            d=by[scaling+'__'+p]
            counts.append([short[p],scaling,d['depth'],d['experts'],d['topk'],d['routed_total_t'],d['routed_active_b']])
    add('Expert layouts: active versus total, per layer',table(['Attention/body','10T rule','Layers','Total experts/layer','Selected experts/token/layer','Routed total T','Routed active B'],counts)+'''<p>Counts describe routed experts. Shared experts and dense/attention work are additional. The direct balanced body is held fixed within its six attention rows; SWA uses the existing SWA body, so cross-body comparisons are not perfectly all-else-equal.</p>''')
    paired=[]
    for p in parents:
        for scaling in ('fixed_active','ratio_preserved'):
            for h in ('GB300','VR200-dense'):
                q={r['phase']:r for r in matched if r['aid']==scaling+'__'+p and r['system']==h and r['dep']==64 and r['cpr']==4}
                if q:
                    a=q['decode'];b=q['prefill'];paired.append([short[p],scaling,h,a['latency_5t_ms'],a['latency_10t_ms'],a['latency_ratio'],b['latency_ratio'],a['memory_10t_gib'],a['admit90_10t'],b['admit90_10t']])
    add('Matched 5T → 10T: what actually gets slower?',table(['Architecture','10T rule','Hardware','5T decode ms','10T decode ms','Decode latency ratio','Prefill latency ratio','10T decode GiB/rank','10T decode admitted','10T prefill admitted'],paired)+'''<p>Same DEP64 and concurrency/rank=4, same attention and context, same precision. Ratios above one mean slower. Decode and prefill have separate admission flags because prefill workspace can be larger. Non-admitted points remain visible only as resource-shape diagnostics, not feasible configurations. At fixed active work, more experts reduce routing collisions, so expert weight traffic can still increase even though active FLOPs do not. Preserved sparsity doubles active expert work, not attention or every serial operation, so whole-step latency need not exactly double.</p>''')
    boundary_table=[]
    for p in parents:
        for scaling in ('fixed_active','ratio_preserved'):
            for h in ('GB300','VR200-dense'):
                q=[r for r in bounds if r['aid']==scaling+'__'+p and r['system']==h and r['dep']==64 and r['efficiency']==1]
                a=q[0]
                boundary_table.append([short[p],scaling,h,a['max_cpr_memory90'],a['tps_at_memory_max'],next(r['max_cpr_target'] for r in q if r['target']==50),next(r['max_cpr_target'] for r in q if r['target']==100)])
    add('Memory capacity and the 50/100 TPS targets',table(['Architecture','10T rule','Hardware','Max c/r by memory','TPS/user at max','Max c/r ≥50 TPS','Max c/r ≥100 TPS'],boundary_table)+'''<p>DEP64, no speculation. Boundaries were bracketed using memory growth and verified at integer neighbors; latency thresholds use integer bisection and endpoint verification. Zero means no admitted positive-concurrency point meets the criterion. Boundaries.csv-equivalent data are in capacity_slo.csv, covering all 982 variants and DEP16/32/64.</p>''')
    ratios=[]
    for scaling,eff,target in __import__('itertools').product(('fixed_active','ratio_preserved'),(1.,.5,.7),(50,100)):
        groups={}
        for r in bounds:
            if r['scaling']==scaling and r['efficiency']==eff and r['target']==target:
                a=groups.setdefault(r['aid'],{});a[r['system']]=max(a.get(r['system'],0),r['max_cpr_target'])
        pairs=[(x['GB300'],x['VR200-dense']) for x in groups.values()];v=sorted(b/a for a,b in pairs if a and b)
        ratios.append([scaling,eff,target,len(v),v[int(.1*(len(v)-1))] if v else None,statistics.median(v) if v else None,v[int(.9*(len(v)-1))] if v else None,sum(a==0 and b>0 for a,b in pairs),sum(a==0 and b==0 for a,b in pairs)])
    add('Rubin concurrency improvement and achievable scenarios',table(['10T rule','SOL throughput fraction','Target TPS','Both feasible','p10 ratio','Median ratio','p90 ratio','Newly feasible','Neither feasible'],ratios)+'''<p>Each architecture picks its best pure DEP16/32/64 configuration, normalized to 64 total GPUs. These are descriptive statistics of a constructed grid, not probabilities over real frontier models. Cases newly feasible on Rubin have no finite ratio and are counted separately.</p><p>0.5 and 0.7 are conditional throughput multipliers, not measured 10T efficiencies. They imply 2× and ~1.43× SOL latency at fixed concurrency. A tight target can amplify a hardware speed gain into a large concurrency ratio; a memory-limited workload can gain little or no concurrency. There is no universal Rubin agent multiplier.</p>''')
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for ax,p in zip(axes.flat,parents[:6]):
        for h,color in (('GB300','#b44b2b'),('VR200-dense','#147d70')):
            for scaling,style in (('fixed_active','--'),('ratio_preserved','-')):
                q=sorted([r for r in data if r['aid']==scaling+'__'+p and r['system']==h and r['dep']==64 and r['phase']=='decode'],key=lambda r:r['cpr'])
                ax.plot([r['cpr'] for r in q],[r['latency_ms'] for r in q],style,color=color,label=h+' '+scaling)
                f=[r for r in q if r['admit90']];ax.scatter([r['cpr'] for r in f],[r['latency_ms'] for r in f],color=color,s=14)
        ax.set_xscale('log',base=2);ax.set_title(short[p],fontsize=10);ax.set_xlabel('Concurrency / rank');ax.set_ylabel('TPOT ms');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=6);fig.tight_layout()
    add('Decode curves: active work versus capacity',util.svg(fig)+'''<p>DEP64, 256K cache, no speculation. Dots mark memory-admitted points; lines beyond dots are hypothetical capacity-unconstrained diagnostics. Dashed = fixed active budget; solid = preserved sparsity. We keep the same expert size/depth within each pair, avoiding a hidden depth penalty.</p>''')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,p in zip(axes.flat,[parents[1],parents[3],parents[4],parents[6]]):
        for h,color in (('GB300','#b44b2b'),('VR200-dense','#147d70')):
            for dep,style in ((16,'--'),(64,'-')):
                q=sorted([r for r in sat if r['aid']=='ratio_preserved__'+p and r['system']==h and r['dep']==dep],key=lambda r:r['chunk'])
                ax.plot([r['chunk'] for r in q],[r['tps_gpu'] for r in q],style,color=color,label=h+' DEP'+str(dep))
                f=[r for r in q if r['admit90']];ax.scatter([r['chunk'] for r in f],[r['tps_gpu'] for r in f],color=color,s=12)
        ax.set_xscale('log',base=2);ax.set_title(short[p],fontsize=10);ax.set_xlabel('New prefill tokens / request');ax.set_ylabel('Input tokens/s/GPU');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7);fig.tight_layout()
    sr=[]
    for p in parents:
        for h in ('GB300','VR200-dense'):
            for dep in (16,64):
                q=[r for r in sat if r['aid']=='ratio_preserved__'+p and r['system']==h and r['dep']==dep and r['admit90']]
                if q:
                    peak=max(r['tps_gpu'] for r in q);first=min(r['chunk'] for r in q if r['tps_gpu']>=.9*peak)
                    sr.append([short[p],h,dep,first,peak,max(r['chunk'] for r in q)])
    add('Cached-prefix prefill saturation',util.svg(fig)+table(['Architecture, ratio-preserved','Hardware','DEP','First ≥90% sampled peak: new tokens','Peak input TPS/GPU','Largest feasible chunk'],sr)+'''<p>Concurrency/rank=1, existing 256K cache. “Saturation” means 90% of the best memory-feasible sampled throughput, through 4,096 new tokens—not proof of the asymptotic ceiling. A larger prefill can improve throughput while still increasing completion latency. Dotted admitted points, not capacity-infeasible lines, determine the table.</p>''')
    rp=[]
    for aid in sorted({r['aid'] for r in resources}):
        for h in ('GB300','VR200-dense'):
            q={r['resource']:r for r in resources if r['aid']==aid and r['system']==h and r['dep']==64 and r['cpr']==1 and r['phase']=='prefill'}
            if q:
                a=q['HBM'];rp.append([by[aid]['label'],h,a['original_ms'],a['attention_ms'],a['moe_ms'],a['dispatch_ms']]+[100*q[k]['reduction'] for k in ('HBM','tensor','fabric')]+[a['admit90']])
    add('Bottlenecks: double one resource at a time',table(['Native architecture','Hardware','Prefill ms','Attention ms','MoE ms','Routing ms','2× HBM reduction %','2× tensor reduction %','2× fabric reduction %','Admitted'],rp)+'''<p>256K+1K, DEP64, concurrency/rank=1. Each counterfactual keeps capacity and all other resource rates unchanged. Reductions are not additive time shares. The raw dataset also covers decode and higher concurrency. It preserves compute- and memory-roof times for MoE separately, avoiding the assumption that an attention category is necessarily cache-bandwidth-bound.</p>''')
    top=[]
    for aid in [d['aid'] for d in definitions if d['kind']=='exact' and d['scaling']!='width_geometry']:
        for h in ('GB300','VR200-dense'):
            for spec in (False,True):
                q=[r for r in topology if r['aid']==aid and r['system']==h and r['status']=='ok' and r['admit90'] and r['world']==64 and r['cpr']==1 and r['phase']=='decode' and bool(r['nextn'])==spec]
                if q:
                    a=min(q,key=lambda r:r['latency_ms']);top.append([by[aid]['label'],h,spec,a['dep'],a['tp'],a['dep']*a['cpr'],a['nextn'],a['al'],1000/a['latency_ms'],a['memory_gib']])
                else:top.append([by[aid]['label'],h,spec]+['not admitted']*7)
    add('Can sharding recover usable 10T performance?',table(['Native architecture','Hardware','Speculation','Attention DP','Attention TP','Total requests','Draft tokens','AL','TPS/user','GiB/rank'],top)+'''<p>Fastest admitted screened configuration at <b>64 GPUs and concurrency per attention-DP rank=1</b>. Changing TP changes total request count; that count is shown explicitly, so this is not an equal-concurrency contest. The raw screen includes c/r=4, GPU groups16/32/64, TP1/2/4/8 and prefill as well as decode.</p><p>Pure DEP uses the analytical routed-traffic extension; mixed TP retains AIC native collectives. Thus a topology comparison includes an implementation asymmetry. A pure-DEP memory failure does not prove the model cannot fit with attention TP. Conversely, speculative decode does not inherently solve weight capacity and adds draft/state requirements.</p><p>Speculation uses AIC native graphs only, with DSpark AL assumptions: 2→2.51, 4→3.36, 8→4.00 emitted tokens per step. These are borrowed, not validated for 10T. We do not invent draft implementations for every arbitrary composed hybrid; speculative results on selected native/body graphs are explicitly narrower than the full architecture sweep. No DCP/Helix or cross-rack configuration is silently substituted.</p>''')
    nb=util.read('native_capacity_slo.csv');nbtable=[]
    selected_native=('exact__k3_10t_trend','exact__k3_10t_expert','ratio_preserved__exact__deepseek_v4_pro_5t_expert','ratio_preserved__exact__glm52_ratio_preserved_5t')
    for aid,h,eff,target in __import__('itertools').product(selected_native,('GB300','VR200-dense'),(1.,.5,.7),(50,100)):
        options=[r for r in nb if r['aid']==aid and r['system']==h and r['efficiency']==eff and r['target']==target]
        def best(spec):
            q=[r for r in options if bool(r['nextn'])==spec and r['max_requests_target']>0]
            return max(q,key=lambda r:r['max_requests_target']) if q else None
        a,b=best(False),best(True)
        def config(r):return f"DP{r['dep']:g}/TP{r['tp']:g}, c/r={r['max_cpr_target']:g}" if r else 'not feasible'
        nbtable.append([by[aid]['label'],h,eff,target,a['max_requests_target'] if a else 0,config(a),
            b['max_requests_target'] if b else 0,config(b),b['nextn'] if b else None,b['max_requests_memory90'] if b else None])
    add('Fixed GPU budget: maximum native-model concurrency',table(['Native model','Hardware','Efficiency','Target TPS','No-spec requests','No-spec config','Spec requests','Spec config','Draft tokens','Spec config memory-limit requests'],nbtable)+'''<p>Every row uses the same 64 physical GPUs, with an independent 256K cache per request. The objective is now <b>maximum total requests meeting the target</b>, not fastest single-user latency. Configurations explicitly report concurrency per attention-DP rank as well as total requests. Both memory and SLO boundaries are verified at adjacent integer concurrency values.</p><p>No-spec and speculative columns each optimize over TP1/2/4/8; the speculative column additionally selects draft depth2/4/8 with the borrowed DSpark ALs. A tie is not a unique optimum. “Memory-limit requests” is the capacity of the selected speculative recipe, not necessarily the recipe with largest possible cache capacity. 0.5/0.7 efficiencies remain conditional throughput factors. The full native_capacity_slo.csv covers all ten native/source 10T models and every tested recipe.</p><p>This closes the gap between a model fitting with sharding and how many active requests that sharding can actually serve under the projected SLO. It does not remove the mixed-TP communication-model caveat or establish real 10T acceptance rates.</p>''')
    add('Rank balance, routing and precision',table(['Native architecture','DEP','Mean-rank ms','Critical-rank sensitivity ms','Increase %','Admitted'],[
        [by[r['aid']]['label'],r['dep'],r['mean_rank_ms'],r['max_rank_ms'],100*r['latency_increase'],r['admit90']] for r in ranks if r['system']=='GB300' and r['cpr']==1])+'''<p>The critical-rank sensitivity samples 80 uniformly routed synthetic batches, counting maximum per-rank expert executions and unique expert weights. It is not measured router behavior. Ideal mean-rank routing remains the main baseline, and the sensitivity does not silently modify SLO boundaries.</p><p>routing_sensitivity.csv contrasts original broadcast collectives with routed mean occupancy, correlated routes and +25% expert execution load. precision_sensitivity.csv screens W4A8 MXFP4/FP8 compute, NVFP4 and FP8 experts, with FP8/BF16 KV and TP1/4. Changing stored weight precision is not the same as changing compute precision. Numerical precision support in a simulator does not establish equal model quality or a production kernel at that shape.</p>''')
    add('Practical architecture conclusions and remaining gaps','''<p><b>Use the scaling rule, not just “10T,” to interpret performance.</b> Fixed-active scaling increases expert capacity and weight residency with a smaller execution penalty; ratio-preserved scaling deliberately buys more active computation. Neither is known to preserve frontier quality. Wide/deep native controls expose extra non-MoE costs that a constant-width expert-count sweep omits.</p>
    <ul><li><b>Conservative candidates:</b> regular global MLA layers within KDA hybrids, or SWA with meaningful global layers; examine ratio-preserved active budgets alongside fixed-active envelopes.</li><li><b>Promising, kernel-sensitive candidates:</b> DSA with index reuse, compressed V4 attention, and SWA/sparse mixtures. Selected-cache traffic can be small while index scans and projections remain significant.</li><li><b>Aggressive performance bounds:</b> all-linear state and very sparse activation. Fast simulations are not evidence of equal reasoning or retrieval quality.</li></ul>
    <p>No universal “best 10T architecture” is justified without a quality constraint. Performance comparisons rank explicit hypotheses, not unknown OpenAI or Anthropic internals. The prior 5T calibration is inherited only as context; no new empirical 10T calibration is claimed.</p>
    <p>Still unresolved: real 10T kernel utilization and tails; native Rubin driver-visible capacity/profiles; true independent KV sharding/Helix; routing distributions and expert imbalance; sparse/stateful kernel execution and state rollback; model-specific draft acceptance; agent arrivals, queueing, think time and prefill/decode interference. AIC remains the practical primary tool; Vidur, LLMCompass and FlowSim remain complementary options described in the linked earlier report.</p>''')
    labels={k:d['label'] for k,d in by.items()}
    compact=[[r[k] for k in ('aid','system','dep','phase','cpr','latency_ms','tps_user','tps_gpu','memory_gib','admit90')] for r in data]
    add('Search the complete controlled grid','''<p>Main non-speculative results. Prefill TPS measures new input tokens, not generated tokens.</p><input id="search" placeholder="Architecture or scaling rule"><select id="hardware"><option value="">All hardware</option><option>GB300</option><option>HBM+fabric</option><option>VR200-dense</option></select><select id="phase"><option>decode</option><option>prefill</option></select><select id="dep"><option value="">All DEP</option><option>16</option><option>32</option><option>64</option></select><label><input id="fit" type="checkbox" checked>90%-admitted only</label><p id="count"></p><div id="results"></div><button id="prev">Previous</button><button id="next">Next</button>''')
    files=[]
    for p in sorted(P.glob('*.csv')):
        rows=util.read(p.name);files.append([p.name,len(rows),sum(r.get('status')=='error' for r in rows)])
    add('Downloads and reproducibility',table(['Dataset','Rows','Errors'],files)+'<ul>'+''.join(f'<li><a href="{p.name}">{p.name}</a></li>' for p in sorted(P.iterdir()) if p.suffix in ('.json','.jsonl','.csv'))+'''</ul><p>Scripts: study.py → diagnostics.py → native_boundaries.py → build_report.py, using the existing .venv-aic-kimi-k3 Python. No new environment or GPU installation was needed. The installed AIC checkout is unchanged. aic_cache.jsonl retains raw tasks, operation latencies/source tags and analytical resource ledgers. The manifest pins source/configuration hashes and AIC commit. Original 5T data are not overwritten.</p><p>Validation checks configuration scaling, finite/nonnegative metrics, memory/latency boundaries, expected row counts, GPU limits, HTML links and JavaScript syntax. These are consistency checks, not silicon validation. Cached results use the inherited adapter version and config/hardware/task signatures; change the version before modifying numerical formulas.</p>''')
    css='''body{font:16px/1.55 system-ui;background:#f2f5f6;color:#21333e;margin:0}main{max-width:1280px;margin:auto;padding:30px}h1{font-size:38px}h2{color:#12545a}section{background:white;border-radius:12px;margin:24px 0;padding:25px}.cards{display:flex;gap:16px}.cards>div{flex:1;background:#eaf3f1;padding:16px;border-radius:8px}nav{display:flex;flex-wrap:wrap;gap:10px}nav a{font-size:13px}a{color:#147383}.scroll{overflow:auto;max-height:620px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:8px;text-align:right;border-bottom:1px solid #dde5e7}th{background:#e4eff0;position:sticky;top:0}td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f7fafa}.plot svg{width:100%;height:auto}input,select,button{padding:8px;margin:4px}@media(max-width:700px){main{padding:10px}.cards{display:block}section{padding:14px}}'''
    js='''let page=0;const $=x=>document.getElementById(x);function show(){const needle=$('search').value.toLowerCase();let q=data.filter(r=>(r[0]+' '+labels[r[0]]).toLowerCase().includes(needle)&&(!$('hardware').value||r[1]===$('hardware').value)&&r[3]===$('phase').value&&(!$('dep').value||r[2]===Number($('dep').value))&&(!$('fit').checked||r[9]));page=Math.max(0,Math.min(page,Math.ceil(q.length/50)-1));$('count').textContent=q.length+' rows · page '+(page+1);$('results').innerHTML='<div class="scroll"><table><tr>'+['Architecture','Hardware','DEP','Phase','C/r','Latency ms','TPS/user','TPS/GPU','GiB/rank','Admitted'].map(x=>'<th>'+x+'</th>').join('')+'</tr>'+q.slice(page*50,page*50+50).map(r=>'<tr>'+r.map((x,i)=>'<td>'+(i===0?labels[x]:typeof x==='number'?x.toLocaleString(undefined,{maximumFractionDigits:2}):x)+'</td>').join('')+'</tr>').join('')+'</table></div>'}['search','hardware','phase','dep','fit'].forEach(id=>$(id).oninput=()=>{page=0;show()});$('prev').onclick=()=>{page--;show()};$('next').onclick=()=>{page++;show()};show();'''
    nav='<nav>'+''.join(f'<a href="#s{i}">{html.escape(t)}</a>' for i,(t,b) in enumerate(sections))+'</nav>'
    doc='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>10T frontier inference study</title><style>'+css+'</style><main><h1>10T frontier inference: architecture and hardware</h1><p>GB300 versus Rubin · decode and cached-prefix prefill · 8 September 2026</p>'+nav+''.join(f'<section id="s{i}"><h2>{html.escape(t)}</h2>{b}</section>' for i,(t,b) in enumerate(sections))+'</main><script>const data='+json.dumps(compact)+';const labels='+json.dumps(labels)+';'+js+'</script></html>'
    (P/'TEN_TRILLION_FRONTIER_STUDY.html').write_text(doc)
    validation=[]
    for name,count,errors in files:
        assert errors==0,(name,errors)
        validation.append(dict(dataset=name,rows=count,errors=errors))
    assert len(data)==982*3*3*2*6
    assert len(bounds)==982*2*3*6
    assert len(nb)==10*2*4*4*6 and all(r['world']==64 for r in nb)
    assert len(definitions)==len(by)
    for r in data:
        assert all(math.isfinite(r[k]) and r[k]>=0 for k in ('latency_ms','memory_gib','attention_ms','moe_ms','dispatch_ms'))
    for r in topology:
        assert r.get('world',0)<=72
    assert not any(r['capped'] for r in bounds)
    for d in definitions:
        if d.get('parent_aid','').startswith('factorial'):
            original=next(s for s in e.base.body_specs() if s.id==d['body'])
            assert d['experts']==2*original.experts
            assert d['topk']==original.topk*(1 if d['scaling']=='fixed_active' else 2)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.ids=set();self.links=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if 'id' in a:assert a['id'] not in self.ids;self.ids.add(a['id'])
            if tag=='a':self.links.append(a.get('href',''))
    p=Links();p.feed(doc)
    for link in p.links:
        if link.startswith('#'):assert link[1:] in p.ids
        elif '://' not in link:assert (P/link).exists(),link
    result=subprocess.run(['node','--check'],input='const data=[];const labels={};'+js,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert not re.search(r'\brefills?\b',doc,re.I)
    e.write_json(P/'validation.json',dict(passed=True,datasets=validation,total_csv_rows=sum(x[1] for x in files)))
    paths=list(st.HERE.glob('*.py'))+list(st.HERE.glob('models/**/config.json'))+list(P.glob('*.csv'))
    inherited=[e.HERE/'engine.py',e.HERE/'run.py',e.HERE/'report.py',e.base.AL_SOURCE,
       e.ROOT/'experiments/5t-architecture-design-study/run_study.py',e.ROOT/'experiments/5t-swa-architecture-study/run_swa_study.py',
       oldpath/'unified_results.csv',oldpath/'parallelism_screen.csv']
    modelpaths={__import__('pathlib').Path(d['model'])/'config.json' for d in definitions if d.get('model')}
    e.write_json(P/'manifest.json',dict(aic_commit=subprocess.check_output(['git','-C',str(e.base.AIC_REPO),'rev-parse','HEAD'],text=True).strip(),
        aic_status=subprocess.check_output(['git','-C',str(e.base.AIC_REPO),'status','--porcelain'],text=True),
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(paths+inherited+list(modelpaths)) if p.exists()}))
    print('Validated and built',P/'TEN_TRILLION_FRONTIER_STUDY.html')

if __name__=='__main__':main()
