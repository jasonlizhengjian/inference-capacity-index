"""Build an offline, self-contained HTML audit and results explorer."""
import collections
import csv
import hashlib
import html
import io
import json
import statistics
import subprocess
from datetime import datetime,timezone
from pathlib import Path
import engine as e
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=e.OUT
FIGURE_COUNT=0
def read(name):
    rows=list(csv.DictReader((P/name).open()))
    for r in rows:
        for k,v in r.items():
            if v in ('True','False'):r[k]=v=='True'
            elif v and k not in ('aid','label','family','kind','model','system','phase','status','error','source'):
                try:r[k]=float(v)
                except ValueError:pass
    return rows

def f(x):
    if x is None:return '—'
    if isinstance(x,float):return f'{x:,.2f}' if abs(x)<100 else f'{x:,.0f}'
    return html.escape(str(x))

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+f(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'

def svg(fig):
    global FIGURE_COUNT
    FIGURE_COUNT+=1
    if not (P/'curve_preview.png').exists():fig.savefig(P/'curve_preview.png',dpi=100,bbox_inches='tight')
    buf=io.StringIO();fig.savefig(buf,format='svg',bbox_inches='tight');plt.close(fig)
    text=buf.getvalue();prefix=f'figure{FIGURE_COUNT}_'
    text=text.replace('id="','id="'+prefix).replace('href="#','href="#'+prefix).replace('url(#','url(#'+prefix)
    return '<div class="plot">'+text[text.index('<svg'): ]+'</div>'

def main():
    data=read('unified_results.csv');bounds=read('capacity_slo_boundaries.csv');sat=read('saturation.csv')
    topo=read('topology_spec.csv');width=read('width_depth_results.csv');sens=read('assumption_sensitivity.csv')
    workloads=read('workload_sensitivity.csv');diag=read('resource_diagnostics.csv');cal=read('calibration_recheck.csv')
    defs=json.loads((P/'definitions.json').read_text());labels={d['aid']:d['label'] for d in defs}
    reps=['exact__k3_2p8t','exact__k3_5t_trend','exact__k3_10t_trend','exact__deepseek_v4_pro_5t_expert',
          'exact__glm52_ratio_preserved_5t','swa__full_gqa','swa__common_r5_w512','swa__swa5_dsa','swa__swa5_kda',
          'factorial__direct_L61_balanced_m3072_s1__kda75_dsa25']
    colors={'GB300':'#b34825','HBM+fabric':'#738296','VR200-dense':'#14776a'}
    sections=[]
    def add(title,body):sections.append((title,body))
    add('What this revision changes','''<p>This is the new review point for the frontier study. It preserves the old results as historical artifacts, and adds a corrected, explicitly labeled analytical extension to AIC. It is not a hardware benchmark and does not identify any closed model’s architecture.</p>
    <div class="cards"><div><b>506 existing designs</b><br>15 source/scaled configurations, 448 controlled MoE × attention combinations, 43 SWA/hybrid designs.</div><div><b>144 additional shapes</b><br>2.8/5/10T routed weights; residual widths 7,168/12,288/16,384; 64/96 layers; ~100/200B routed active parameters; GQA, SWA and DSA.</div><div><b>Two workloads</b><br>Decode at 256K cached tokens; cached-prefix prefill of 1K new tokens after 256K. Additional 8K–1M contexts and 1–4K prefill sizes.</div></div>
    <p><b>Main tables and curves are non-speculative SOL-style projections.</b> Speculation is a separate topology table, with borrowed DSpark acceptance lengths. Do not compare the new non-speculative table directly with an earlier speculative table.</p>
    <p>The most consequential corrections are (1) expert overlap under random routing, (2) routed traffic versus AIC’s broadcast collectives, (3) FP8 compute for W4A8, (4) consistent DSA residual dimensions and selection sizes, and (5) Rubin compute rates separated from a bandwidth-only upgrade.</p>''')
    hw=json.loads((P/'hardware.json').read_text())
    add('Hardware: bandwidth-only versus dense-spec Rubin',table(['Scenario','HBM TB/s','Capacity GiB','NVLink directional TB/s','BF16 PF/s','FP8 PF/s','FP4 PF/s'],[(name,x['gpu']['mem_bw']/1e12,x['gpu']['mem_capacity']/e.GIB,x['node']['inter_node_bw']/1e12,x['gpu']['bfloat16_tc_flops']/1e15,x['gpu']['fp8_tc_flops']/1e15,x['gpu']['fp4_tc_flops']/1e15) for name,x in hw.items()])+'''<p>VR200-dense is a <b>dense-spec proxy, not a native Rubin kernel simulator</b>: GB300 operation implementations/SM identity and unspecified parameters remain inherited. We use NVIDIA’s dense 35 PF/s FP4 rate, not the headline 50 PF/s inference rate. Source: <a href="https://www.nvidia.com/en-gb/data-center/vera-rubin-nvl72/">NVIDIA specifications</a>, accessed 2026-09-07; preliminary, subject to change.</p><p><b>Capacity caveat:</b> GB300 uses AIC’s 277.55 GiB driver-visible value; Rubin conservatively interprets advertised 288 GB as decimal (268.22 GiB). That is not evidence Rubin has less usable capacity. The bandwidth/fabric-only scenario retains identical GB300 capacity. Capacity differences of a few percent should not be treated as a hardware conclusion. Admission reserves 10% of capacity in addition to modeled runtime overhead.</p>''')
    cr=[]
    for r in cal:cr.append([r['concurrency_per_rank'],r['adapter_policy'],r['updated_step_ms'],r['measured_median_tpot_ms'],r['actual_throughput_fraction']])
    add('What “SOL” and “achievable” mean here','''<p>AIC sums per-operation roofline estimates for a particular execution strategy. Our routed-traffic extension uses the same style. This is an idealized <em>strategy-specific performance envelope</em>, not a proof of the minimum physically possible latency: kernel overlap/fusion can reduce a sum, while startup, poor utilization and imbalance can increase actual time.</p>
    <p>For the achievable scenarios, throughput = efficiency × SOL throughput and latency = SOL latency / efficiency. Thus 0.5× means twice the latency, not half. We retain 0.5 central and 0.7 optimistic as requested. They are not confidence intervals or universal measured efficiencies.</p>
    <p>The two historical Kimi TRT-LLM comparisons have been reproduced with BF16 non-MoE GEMMs and BF16 KV at 8K context, DEP16, no speculation:</p>'''+table(['Concurrency/rank','Routing model','Projection ms','Published median ms','Actual / projected throughput'],cr)+'''<p>The legacy estimates exactly reproduce the old calibration. The revised traffic/occupancy assumptions change the envelope and measured fraction, not always in the same direction. This is a valuable check, but two medians from one model and one context cannot validate 5T designs, sparse kernels, prefill, or Rubin. A serving median over 1K output is also not identical to a single static initial-context step. <a href="https://nvidia.github.io/TensorRT-LLM/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.html">Published TRT-LLM protocol</a>.</p>''')
    selected=[]
    for aid in reps:
        for hwname in ('GB300','VR200-dense'):
            b=[r for r in bounds if r['aid']==aid and r['system']==hwname and r['dep']==64 and r['efficiency']==1]
            if b:
                p=b[0];selected.append([labels[aid],hwname,p['max_cpr_memory90'],p['tps_at_memory_max'],next(x['max_cpr_target'] for x in b if x['target']==50),next(x['max_cpr_target'] for x in b if x['target']==100)])
    add('Memory-full concurrency versus useful concurrency', '''<p>DEP64, 256K cache, no speculation. Concurrency is <b>requests per attention-DP rank</b>; multiply by 64 for this group. “Memory maximum” means at most 90% modeled capacity. Integer boundaries and their immediate neighbors were evaluated, rather than inferred solely from a coarse grid. We assume identical independent requests; prefix sharing across users is not credited.</p>'''+table(['Architecture','Hardware','Max c/r by memory','TPS/user at max','Max c/r ≥50 TPS','Max c/r ≥100 TPS'],selected))
    ratios=[]
    for eff in (1.,.5,.7):
        for target in (50,100):
            grouped={}
            for r in bounds:
                if r['efficiency']==eff and r['target']==target:
                    z=grouped.setdefault(r['aid'],{});z[r['system']]=max(z.get(r['system'],0),r['max_cpr_target'])
            pairs=[(x['GB300'],x['VR200-dense']) for x in grouped.values()]
            vals=sorted(y/x for x,y in pairs if x and y)
            ratios.append([eff,target,len(vals),vals[int(.1*(len(vals)-1))],statistics.median(vals),vals[int(.9*(len(vals)-1))],sum(x==0 and y>0 for x,y in pairs),sum(x==0 and y==0 for x,y in pairs)])
    add('How much more concurrency does Rubin support?',table(['Efficiency','Target TPS','Both feasible designs','p10 ratio','Median ratio','p90 ratio','Newly feasible','Neither feasible'],ratios)+'''<p>Each design selects the better of DEP16/DEP64, normalized to 64 total GPUs (four DEP16 replicas versus one DEP64 group). Ratios compare the same architecture, context, precision and efficiency. These are <b>descriptive statistics of a deliberately constructed design grid</b>, not probabilities over real frontier models.</p><p>The answer is strongly SLO-dependent. A large bandwidth gain need not increase concurrency when both systems are already memory-capacity-limited. Conversely, a tight latency target can produce a very large concurrency ratio because Blackwell is close to the one-request threshold. “Newly feasible” cases have no finite ratio; they are never silently counted as infinity or discarded from the table. Some ratios below one arise from the conservative Rubin capacity convention.</p><p>Consequently there is no defensible architecture-independent statement that Rubin can always fill memory while meeting 50 or 100 TPS/user. Use the specific architecture row and efficiency scenario.</p>''')
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for ax,aid in zip(axes.flat,reps[:6]):
        for h,c in colors.items():
            q=sorted([r for r in data if r['aid']==aid and r['system']==h and r['dep']==64 and r['phase']=='decode'],key=lambda x:x['cpr'])
            ax.plot([r['cpr'] for r in q],[r['latency_ms'] for r in q],color=c,label=h)
            fit=[r for r in q if r['admit90']]
            ax.scatter([r['cpr'] for r in fit],[r['latency_ms'] for r in fit],color=c,s=16)
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_title(labels[aid],fontsize=10);ax.grid(alpha=.2);ax.set_xlabel('Concurrency / rank');ax.set_ylabel('TPOT ms')
    axes.flat[0].legend(fontsize=8);fig.tight_layout()
    add('Latency curves and why the gains change',svg(fig)+'''<p>Solid curves continue through memory-infeasible points to expose the resource shape; dots mark 90%-admissible points. Both axes are logarithmic. At low batch, weights/serial work often dominate. Higher batch amortizes expert weights until most experts are touched; thereafter active GEMM work and per-request cache traffic keep increasing. Individual roofline maxima create kinks. Neither latency nor per-user TPS has to plateau: aggregate throughput can approach a ceiling while per-user TPS keeps falling.</p><p>HBM+fabric versus GB300 isolates the earlier thought experiment. VR200-dense additionally changes tensor compute and the explicit capacity convention. These three curves prevent a compute gain from being attributed to memory bandwidth.</p>''')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,aid in zip(axes.flat,[reps[1],reps[3],reps[4],'swa__common_r5_w512']):
        for h in ('GB300','VR200-dense'):
            for dep,style in ((16,'--'),(64,'-')):
                q=sorted([r for r in sat if r['aid']==aid and r['system']==h and r['dep']==dep],key=lambda r:r['chunk'])
                ax.plot([r['chunk'] for r in q],[r['tps_gpu'] for r in q],style,color=colors[h],label=f'{h} DEP{dep}')
        ax.set_xscale('log',base=2);ax.set_title(labels[aid],fontsize=10);ax.set_xlabel('New prefill tokens / request (c/r=1)');ax.set_ylabel('Input tokens/s/GPU');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8);fig.tight_layout()
    satrows=[]
    for aid in reps:
        for h in ('GB300','VR200-dense'):
            for dep in (16,64):
                q=[r for r in sat if r['aid']==aid and r['system']==h and r['dep']==dep and r['admit90']]
                if q:
                    peak=max(x['tps_gpu'] for x in q);near=min((x for x in q if x['tps_gpu']>=.9*peak),key=lambda x:x['chunk'])
                    satrows.append([labels[aid],h,dep,near['chunk'],peak,max(x['chunk'] for x in q)])
    add('Cached-prefix prefill saturation and the DEP question',svg(fig)+'''<p>These are cached-prefix prefills, not cold 257K prefills. The curves vary new tokens while preserving 256K existing cache per request. Saturation below is defined as 90% of the best <b>memory-feasible sampled</b> input throughput through 4,096 tokens. It is not an established infinite-batch asymptote; a curve still rising at the last point remains unresolved.</p>'''+table(['Architecture','Hardware','DEP','First ≥90% sampled peak: new tokens','Peak input TPS/GPU','Largest feasible sampled chunk'],satrows)+'''<p>DEP64 is not intrinsically optimal for prefill, nor is a smaller group intrinsically optimal because of communication. Larger EP reduces per-rank expert compute/weights but increases the global batch and routes; attention is still per attention-DP rank. Compare both per-request completion time and throughput at a fixed total GPU budget. The corrected grid screens all designs at DEP8/16/32/64; native/source models additionally screen mixed attention TP1/2/4/8. That remains a screen, not an exhaustive globally optimal serving plan.</p>''')
    resource=[]
    for r in diag:
        if r['dep']==64 and r['cpr']==1 and r['phase']=='prefill':
            resource.append([labels[r['aid']],r['system'],r['step_ms'],r['attention_ms'],r['moe_ms'],r['dispatch_ms'],r['moe_compute_ms'],r['moe_memory_ms'],'OOM' if r['oom'] else 'fits'])
    add('Exactly which resource is saturated?',table(['Architecture','Hardware','Prefill ms','Attention ms','MoE ms','Routing ms','MoE compute roof ms','MoE memory roof ms','Memory'],resource)+'''<p>For each MoE operation, the model chooses max(compute FLOPs / tensor rate, bytes / HBM bandwidth), then sums across operations/layers. The diagnostic CSV preserves both candidate times. A compute-bound MoE does not mean every GPU resource is saturated for the whole request: attention, communication, and serial operations occupy different intervals.</p><p>Full attention grows with cached length × new query tokens. Sparse attention reduces selected value-cache work, but its indexer may still scan the full context. SWA reduces local cache reads and storage; global layers retain long-context work. Linear attention replaces growing KV reads with state updates, but prefill still needs chunked state processing. HBM bandwidth alone cannot remove those tensor-compute or serial bottlenecks.</p>''')
    responses=read('resource_counterfactuals.csv')
    response_table=[]
    for aid in (reps[1],reps[3],reps[4],'swa__common_r5_w512'):
        for h in ('GB300','VR200-dense'):
            q=[r for r in responses if r['aid']==aid and r['system']==h and r['dep']==64 and r['cpr']==1 and r['phase']=='prefill']
            if q:
                by={r['resource_doubled']:r for r in q}
                response_table.append([labels[aid],h]+[100*by[k]['latency_reduction'] for k in ('HBM','tensor','fabric')]+[100*by['HBM']['attention_reduction'],100*by['tensor']['attention_reduction']])
    add('Counterfactual resource tests',table(['Architecture','Hardware','2× HBM: latency reduction %','2× tensor: reduction %','2× fabric: reduction %','Attention reduction from HBM %','Attention reduction from tensor %'],response_table)+'''<p>256K+1K prefill, DEP64, c/r=1. Each test doubles only the named resource, leaving capacity unchanged. This directly separates attention bandwidth from attention compute: an “attention” timing category alone cannot establish which is limiting. Percentages are finite counterfactual improvements, not additive time shares; doubling a resource can move an operation across its roofline knee. The raw CSV also includes decode, c/r=4/16 and DEP16.</p>''')
    screen=data+read('parallelism_screen.csv')
    comparison=[]
    for aid in (reps[1],reps[3],reps[4],'swa__common_r5_w512'):
        for h in ('GB300','VR200-dense'):
            q={r['dep']:r for r in screen if r['aid']==aid and r['system']==h and r['phase']=='prefill' and r['cpr']==1}
            good=[r for r in q.values() if r['admit90']]
            comparison.append([labels[aid],h]+[q[d]['latency_ms'] if d in q and q[d]['admit90'] else 'not admitted' for d in (8,16,32,64)]+[min(good,key=lambda r:r['latency_ms'])['dep'] if good else None])
    add('Prefill: comparing all four DEP sizes',table(['Architecture','Hardware','DEP8 ms','DEP16 ms','DEP32 ms','DEP64 ms','Fastest admitted DEP'],comparison)+'''<p>Same 256K+1K request, c/r=1 and no speculation. A fixed 64-GPU deployment can host 8/4/2/1 independent replicas respectively. This keeps the total request count at 64 when every replica is populated. “Not admitted” applies the same 90% memory budget, not a claim that the model can never run with a different sharding.</p><p>This demonstrates why one universal prefill parallelism answer is inappropriate: weight amortization, attention work, expert top-k and routed traffic change the tradeoff by architecture. Mixed attention TP is separately available in topology_spec.csv, with its distinct communication assumptions.</p>''')
    changes=[]
    for aid in reps[:5]:
        q=[r for r in sens if r['aid']==aid and r['system']=='GB300' and r['dep']==64 and r['cpr']==1 and r['phase']=='prefill']
        if q:
            for r in q:changes.append([labels[aid],r['policy'],r['occupancy'],r['step_ms'],r['dispatch_ms'],r['moe_ms']])
    add('Routing, overlap and rank balance: explicit alternatives',table(['Architecture','Traffic policy','Expert occupancy','Prefill ms','Dispatch + combine ms','MoE ms'],changes)+'''<p>The pinned SGLang SOL path uses all-gather/reduce-scatter for this configuration. An observed DeepSeek-5T DEP64 prefill query used 469,762,048 bytes in each collective, accumulating ~125.37 ms of dispatch/combine across 61 layers. Calling this a measured optimized MoE all-to-all was incorrect. Its DeepEP SOL paths are unimplemented.</p>
    <p>The new pure-DEP extension packs tokens by destination rank. Expected unique experts for T independent tokens is E[1−(1−k/E)<sup>T</sup>]. Destination-rank probability is calculated without replacement, not k distinct ranks by assumption. At E=896, k=16, T=64, the expectation is ~613 experts, not 896. Compute remains Tk expert executions; only weight reuse changes. BF16 token payloads are used for both dispatch and combine; four-bit weights do not imply four-bit networking.</p>
    <p><b>This is a traffic ceiling, not validated DeepEP performance.</b> No startup, pack/unpack kernel cost, fabric contention, routing skew, or critical-path maximum-rank penalty is added. Fractional average expert occupancy assumes ideal balance and can be optimistic at tiny batches. Mixed TP keeps AIC’s native collectives, so comparing its winner against routed pure-DEP includes an implementation asymmetry. We preserve legacy/expected/correlated-routing/+25% expert-load sensitivity results rather than pretend the routing distribution is known. The correlated case shares routes in groups of four; it is not a measured distribution.</p>''')
    balance=read('rank_balance.csv')
    add('Small batches: the busiest rank is not the average',table(['Architecture','DEP','Mean-rank TPOT ms','Critical-rank sensitivity ms','Latency increase %'],[
        [labels[r['aid']],r['dep'],r['mean_rank_ms'],r['rank_tail_ms'],100*r['latency_increase_fraction']]
        for r in balance if r['system']=='GB300' and r['cpr']==1])+'''<p>Uniform random top-k routing, 80 synthetic batches per shape, fixed seed. Each batch counts expert executions and unique expert weights on every rank; the sensitivity uses the mean of the maximum-rank counts. Compute and weight maxima are taken separately, making this a conservative MoE-only approximation to the critical rank. It does not include skewed routers, networking tails, or empirical kernel utilization.</p><p>Even an unbiased router has finite-batch imbalance. The ideal mean-rank assumption is an intentional optimistic baseline, not a statement that all ranks are equally loaded. The main SLO boundaries have <b>not</b> been silently adjusted by this sensitivity; use it as an additional model uncertainty. Results also include concurrency/rank=4/16 and Rubin.</p>''')
    winners=[]
    for aid in reps[:5]:
        for h in ('GB300','VR200-dense'):
            q=[r for r in topo if r['aid']==aid and r['system']==h and r['status']=='ok' and r['admit90'] and r['phase']=='decode' and r['cpr']==1]
            for spec in (False,True):
                candidates=[r for r in q if bool(r['nextn'])==spec]
                if candidates:
                    r=min(candidates,key=lambda r:r['latency_ms']);winners.append([labels[aid],h,'spec' if spec else 'none',r['world'],r['dep'],r['tp'],r['nextn'],r['al'],1000/r['latency_ms'],r['memory_gib']])
    add('Speculative decode and mixed parallelism',table(['Architecture','Hardware','Speculation','GPUs','Attention DP','Attention TP','Draft tokens','AL','TPS/user, c/r=1','GiB/rank'],winners)+'''<p>This table minimizes latency among screened configurations at concurrency per attention-DP rank = 1; it is not equal global concurrency or equal GPU count. All rows use ≤64 GPUs, within the 72-GPU rack limit. Do not infer throughput-optimal deployment from a latency winner.</p><p>Verification uses n+1 tokens and existing cache, plus the AIC architecture-specific draft/MTP work; divide the step by borrowed AL. DSpark values: n=2 → 2.51, n=4 → 3.36, n=8 → 4.00. These acceptance lengths are assumptions for scaled models, not predictions of their draft quality. Kimi has explicit draft operations; other source models use their implemented MTP approximations. The extension preserves those differences. Draft routing correlation, rollback state, and runtime overlap need real implementation validation.</p><p>True decode-context parallelism is not silently modeled as TP. This CLI does not expose CP. <a href="https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog22_Helix_Parallelism_Scaling_Multi_Million_Token_Decoding_with_KV_Cache_Sharding.html">TRT-LLM Helix</a> shows why sharding KV independently of FFN weights is a relevant remaining option. Its communication/state behavior cannot be inferred just by dividing KV bytes by TP; no unvalidated Helix-plus-spec number is claimed here.</p>''')
    add('Architecture coverage and conservative choices','''<p>The existing controlled set varies depth, expert width/count/top-k, routed active parameters, direct versus latent expert interfaces, full MLA, KDA/MLA ratios, DSA top-512/1024/2048/4096 and index reuse, V4 compressed attention, and linear/sparse mixtures. The SWA set adds global GQA, windows 128–4096, local:global cadences 1:1 through 7:1, GPT-OSS/Inkling-style geometries, and SWA paired with MLA, V4, DSA or KDA.</p><p>The new 144 native-geometry shapes explicitly vary residual width independently of depth, active MoE budget and total parameter count. “5T” here means approximately the routed expert budget, not an exactly equal whole-model parameter count. Integer experts/top-k are rounded; actual routed counts are in expanded_definitions.json. Attention/dense/shared weights add on top. Architecture mixes are performance controls, not evidence of equal model quality.</p>
    <ul><li><b>Conservative starting points:</b> retain meaningful global-attention capacity (MLA/KDA hybrids or SWA with regular global layers), and examine both ~100B and ~160–200B routed active budgets. More activation may be necessary for quality; performance alone cannot decide.</li><li><b>Promising but implementation-sensitive:</b> DSA with shared index reuse; V4-style compression; SWA/global mixtures. Indexing, gather efficiency and compression kernels require measured validation.</li><li><b>Most inference-aggressive:</b> all-linear state, tiny selected KV sets, infrequent global layers and very sparse MoE. High projected throughput is not a recommendation that these can match a frontier model’s quality.</li></ul>
    <p>For closed frontier systems—including the OpenAI/Anthropic examples in the question—we can bound plausible mechanisms and sizes, not know which design is used. No finite Cartesian sweep covers every architecture. Important additional axes remain: recurrent depth/adaptive token compute, hierarchical routing, learned memory/retrieval with external traffic, separate attention/FFN workers, pipeline scheduling, and multimodal encoders/cross-attention. These are explicit residual gaps, not hidden claims of coverage.</p>''')
    controlled=[]
    for d in defs:
        if d.get('body')=='direct_L61_balanced_m3072_s1':
            aid=d['aid'];vals=[]
            for h in ('GB300','VR200-dense'):
                q={r['phase']:r for r in data if r['aid']==aid and r['system']==h and r['dep']==64 and r['cpr']==4}
                vals += [q['decode']['tps_user'],q['prefill']['latency_ms']]
            controlled.append([d['attention']]+vals)
    controlled.sort(key=lambda r:r[3],reverse=True)
    add('All else equal: attention mechanisms on one ~5T body',table(['Attention profile','GB300 decode TPS/user','GB300 prefill ms','VR200 decode TPS/user','VR200 prefill ms'],controlled)+'''<p>Same 61-layer direct-MoE body, ~4.9T routed parameters, ~100B routed active budget, residual width 7,168, one shared expert, DEP64, concurrency/rank=4. Decode has 256K cache; prefill adds 1K. Sorted by Rubin decode TPS, not by model quality. This is a more meaningful architecture-performance comparison than ranking unrelated native models by total parameters alone.</p><p>Composition replaces attention weights/cache and latency while retaining the body’s runtime-overhead estimate; it does not implement a trained fused hybrid kernel. DSA uses matched residual width and layer-index cadence. Other internal attention geometries intentionally remain architecture-specific. Numeric rankings close together should not outweigh that implementation uncertainty.</p>''')
    extended=json.loads((P/'expanded_definitions.json').read_text());extby={d['aid']:d for d in extended}
    wtable=[]
    for r in width:
        d=extby[r['aid']]
        if d['depth']==64 and 90<d['active_routed_b']<110 and 4.8<d['total_routed_t']<5.2 and r['dep']==64 and r['cpr']==4 and r['phase']=='decode':
            wtable.append([d['mechanism'],d['h'],d['experts'],d['topk'],d['active_routed_b'],r['system'],1000/r['latency_ms'],r['memory_gib'],r['admit90']])
    add('Residual width is a separate scaling axis',table(['Attention','Residual width','Experts/layer','Active/layer','Routed active B','Hardware','TPS/user','GiB/rank','Admit90'],wtable)+'''<p>64 layers, ~5T routed expert weights, ~100B active routed budget, DEP64 and c/r=4; expert intermediate width stays 3,072. Enlarging the residual width increases each expert’s parameters, so expert count/top-k change to retain the target budgets. It also changes dense projections and attention heads independently of MoE FLOPs. This is precisely why “same total and active parameters” is not sufficient to predict latency. All 144 geometries and both phases are in the width-depth CSV.</p>''')
    # Interactive view holds compact numeric rows; full operator ledgers remain downloadable.
    explorer=[]
    for r in data:
        explorer.append([r[k] for k in ('aid','system','dep','phase','cpr','latency_ms','tps_user','tps_gpu','memory_gib','admit90')])
    add('All 506 architectures: searchable results','''<p>Non-speculative main grid. Prefill TPS/user means <em>input</em> tokens per second, not output decode speed. At mixed TP, concurrency/rank elsewhere means attention-DP rank; this pure-DEP grid has one GPU per such rank.</p><input id="search" placeholder="Filter architecture name or ID"><select id="hw"><option value="">All hardware</option><option>GB300</option><option>HBM+fabric</option><option>VR200-dense</option></select><select id="phase"><option>decode</option><option>prefill</option></select><select id="dep"><option value="">DEP16 + DEP64</option><option>16</option><option>64</option></select><label><input type="checkbox" id="fits" checked> 90%-admissible only</label><p id="count"></p><div id="explorer"></div><button id="previous">Previous page</button> <button id="next">Next page</button>''')
    add('Alternatives: useful complements to AIC','''<p>AIC remains the primary engine here because it already represents the model families and supports CPU-only analytical studies. Replacing it would not automatically validate unknown 5T kernels.</p>'''+table(['Option','Best contribution','What would be needed'],[
        ['AIC native SILICON mode / operator profiles','Calibrate MoE, attention, communication and quantization at the exact shapes','Matched GB300 measurements; new Rubin profiles; clean support/source coverage'],
        ['Vidur','Continuous batching, arrivals, scheduling, throughput and latency distributions','Runtime/model profiles; port modern sparse/linear attention and MoE behavior'],
        ['LLMCompass','Independent hardware mapping, memory hierarchy and compute cross-checks','Implement modern architectures and their kernel mappings; validate against measurements'],
        ['FlowSim','Capture semantically rich workload traces and feed hardware simulators','An actual running workload; not an independent replacement for absent hardware data'],
        ['SGLang simulator + InferenceX benchmark client','Serving protocol and scheduler experiments on modeled operators','Fix and calibrate backend costs first; a realistic client does not make roofline kernels realistic']])+'''<p>Primary sources: <a href="https://github.com/microsoft/vidur">Vidur</a>, <a href="https://github.com/PrincetonUniversity/LLMCompass">LLMCompass</a>, <a href="https://github.com/microsoft/FlowSim">FlowSim</a>. No alternative has been installed or presented as a validated frontier-model oracle.</p>''')
    add('Remaining gaps, ranked by decision impact',table(['Priority','Gap','What the present study can say','Evidence needed to close it'],[
        ['1','Real operator efficiency and tail latency','Conditional SOL and 0.5/0.7 throughput envelopes','Matched GB300 shapes; separate prefill/decode; operator timing and p50/p95/p99'],
        ['1','Routed MoE critical path and tiny-batch imbalance','Ideal mean-traffic and load/correlation sensitivities','Per-rank expert histograms, dispatch/combine traces, actual DeepEP/WideEP kernels'],
        ['1','True Rubin implementation/capacity','Dense-spec sensitivity, not measured performance','Driver-visible capacity and Rubin kernels/topology measurements'],
        ['2','KV sharding / Helix / attention-FFN disaggregation','Known missing topology axis; no fabricated CP numbers','An explicit communication schedule and supported backend, then validation'],
        ['2','Service-level agent workload','Steady-state batch budgets only','Arrivals, tool-think time, variable cache lengths, cold misses, prefill interference, queueing'],
        ['2','Sparse/stateful kernel accuracy','Controlled AIC architecture comparisons','Index/gather, compression, KDA/SWA execution traces and cache allocation checks'],
        ['2','Speculation quality and state bookkeeping','Borrowed DSpark AL; implemented draft model only','Acceptance distributions by model/task/context, draft costs and state rollback'],
        ['3','Architecture quality and unknown frontier designs','Performance envelope over explicit hypotheses','Trained models and quality evaluation; cannot be solved with inference simulation']])+'''<p>Useful next physical experiment: reproduce the two Kimi GB300 anchors, add native DS/GLM at 32K/256K context, c/r=1/4/16, with 1/128/1024 prefill queries and no speculation, then add speculation separately. Measure operator traffic and median/tail step times. This distinguishes bad hardware assumptions from bad scheduling assumptions much more effectively than another arbitrary global efficiency multiplier.</p>''')
    files=[]
    for p in sorted(P.glob('*.csv')):
        rows=read(p.name);files.append([p.name,len(rows),sum(r.get('status')=='error' for r in rows)])
    downloads=''.join(f'<li><a href="{html.escape(p.name)}">{html.escape(p.name)}</a> ({p.stat().st_size/1048576:.1f} MiB)</li>' for p in sorted(P.iterdir()) if p.suffix in ('.csv','.json','.jsonl'))
    add('Data, reproducibility and checks',table(['Dataset','Rows','Errors'],files)+f'<ul>{downloads}</ul>'+'''<p>Every raw AIC estimate retains configuration, hardware, quantization, latency/source dictionaries and extension resource ledger in aic_cache.jsonl. Synthetic composition is in run.py; the scoped adapter is in engine.py. Old reports are not overwritten. Run run.py, extend.py, diagnostics.py, rank_balance.py, report.py and validate.py with the existing .venv-aic-kimi-k3 interpreter.</p><p>Checks include Monte Carlo destination-rank probability, exact single-token expert occupancy, resource monotonicity, restoration of temporary patches, nonnegative component accounting, memory/SLO neighbor verification, zero-error row audits, and exact reproduction of the two legacy calibration estimates. These are consistency checks, not silicon validation.</p>''')
    nav='<nav>'+''.join(f'<a href="#s{i}">{html.escape(t)}</a>' for i,(t,b) in enumerate(sections))+'</nav>'
    content=''.join(f'<section id="s{i}"><h2>{html.escape(t)}</h2>{b}</section>' for i,(t,b) in enumerate(sections))
    css='''body{font:16px/1.55 system-ui,sans-serif;color:#1d2935;background:#f4f6f7;margin:0}main{max-width:1250px;margin:auto;padding:32px}h1{font-size:38px;line-height:1.15}h2{font-size:25px;color:#114d50}section{background:white;padding:26px;margin:24px 0;border-radius:12px}nav{display:flex;flex-wrap:wrap;gap:10px}a{color:#116c78}nav a{font-size:13px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #dde5e7;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#e8f1f1;position:sticky;top:0}tr:nth-child(even){background:#f8fafb}.scroll{overflow:auto;max-height:600px}.plot svg{width:100%;height:auto}.cards{display:flex;gap:14px}.cards>div{flex:1;background:#eaf3f1;padding:16px;border-radius:8px}input,select,button{padding:8px;margin:4px}#search{width:35%}.subtitle{color:#516370}.warn{border-left:4px solid #bc6425;padding-left:15px}@media(max-width:700px){main{padding:12px}.cards{display:block}section{padding:14px}}'''
    js='''let page=0;const $=id=>document.getElementById(id);function render(){let needle=$('search').value.toLowerCase();let q=rows.filter(r=>(!$('hw').value||r[1]===$('hw').value)&&r[3]===$('phase').value&&(!$('dep').value||r[2]===Number($('dep').value))&&(!$('fits').checked||r[9])&&(r[0]+' '+labels[r[0]]).toLowerCase().includes(needle));page=Math.max(0,Math.min(page,Math.ceil(q.length/50)-1));$('count').textContent=q.length+' matching rows; page '+(page+1);let out='<div class="scroll"><table><tr>'+['Architecture','Hardware','DEP','Phase','C/r','Latency ms','TPS/user','TPS/GPU','GiB/rank','Admit90'].map(h=>'<th>'+h+'</th>').join('')+'</tr>';for(let r of q.slice(page*50,page*50+50)){out+='<tr>'+r.map((x,i)=>'<td>'+(i===0?labels[x]:typeof x==='number'?x.toLocaleString(undefined,{maximumFractionDigits:2}):x)+'</td>').join('')+'</tr>'} $('explorer').innerHTML=out+'</table></div>'}['search','hw','phase','dep','fits'].forEach(id=>$(id).addEventListener('input',()=>{page=0;render()}));$('previous').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};render();'''
    doc='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Frontier inference: corrected AIC study</title><style>'+css+'</style><main><h1>Frontier inference on GB300 and Rubin</h1><p class="subtitle">Corrected AIC envelopes · decode, cached-prefix prefill, architecture and hardware sensitivities · 7 September 2026</p>'+nav+content+'</main><script>const rows='+json.dumps(explorer)+';const labels='+json.dumps(labels)+';'+js+'</script></html>'
    (P/'FRONTIER_INFERENCE_STUDY.html').write_text(doc)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),version=e.VERSION,
        aic_commit=subprocess.check_output(['git','-C',str(e.base.AIC_REPO),'rev-parse','HEAD'],text=True).strip(),
        aic_status=subprocess.check_output(['git','-C',str(e.base.AIC_REPO),'status','--porcelain'],text=True),
        files={str(p.relative_to(e.HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(e.HERE.glob('*.py'))+list(P.glob('*.csv'))+list(e.HERE.glob('models/*/config.json'))+list(e.HERE.glob('systems/*.yaml'))},
        inherited_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
          [e.ROOT/'experiments/5t-architecture-design-study/run_study.py',e.ROOT/'experiments/5t-swa-architecture-study/run_swa_study.py',e.ROOT/'experiments/unified-prefill-study/run_unified_prefill.py',e.base.AL_SOURCE]},
        source_model_configs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
          (Path(d['model'])/'config.json' for d in defs if d.get('model')) if p.exists()})
    e.write_json(P/'manifest.json',manifest)
    print('Built',P/'FRONTIER_INFERENCE_STUDY.html')

if __name__=='__main__':main()
