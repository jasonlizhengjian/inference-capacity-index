"""Render attention dimensions and paired body/attention scale-up results."""
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
    out=list(csv.DictReader((OUT/name).open()))
    for r in out:
        for k,v in r.items():
            if v in ('True','False'):r[k]=v=='True'
            else:
                try:r[k]=float(v)
                except ValueError:pass
    return out
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ds=json.loads((HERE/'definitions.json').read_text());by={d['id']:d for d in ds};grid=read('grid.csv');slo=read('slo.csv');limits=read('memory_limits.csv')
    assert len(limits)==len(ds)*72 and {r['id'] for r in limits}==set(by)
    assert all(r['admit90'] for r in grid)
    fixed={(r['id'],r['system'],r['cpr'],r['nextn']):r for r in grid if r['world']==64 and r['tp']==1 and r['phase']=='decode'}
    def point(d,hw,c=4,nn=0):return fixed.get((d['id'],hw,c,nn))
    def rate(d,hw,c=4):
        r=point(d,hw,c);return f"{r['tps_user']:.1f}" if r else 'Not admitted'
    lm={(r['id'],r['system'],r['nextn']):r for r in limits if r['world']==64 and r['tp']==1 and r['phase']=='decode'}
    best=[];sg=collections.defaultdict(list)
    for r in slo:sg[r['id'],r['system'],r['efficiency'],r['target'],r['nextn']].append(r)
    for key,rows in sg.items():
        r=max(rows,key=lambda x:(x['requests_64gpu'],x['throughput_gpu']))
        best.append(dict(id=r['id'],system=r['system'],efficiency=r['efficiency'],target=r['target'],nextn=r['nextn'],world=r['world'],tp=r['tp'],cpr=r['cpr'],requests64=r['requests_64gpu'],tps_user=r['tps_user']*r['efficiency'],throughput_gpu=r['throughput_gpu']*r['efficiency'],memory_max_requests64=64*r['max_cpr']/r['tp'],memory_gib=r['memory_gib']))
    write('best_decode.csv',best)
    calibration=json.loads((OUT/'native_recalibration.json').read_text())
    parts=['''<h1>GLM-style attention: internal dimensions and joint scaling</h1><p>Extension of the model-body study · 58 variants at~5T and~10T · GB300 and Rubin dense-spec proxy · September9,2026</p>
<div class="callout"><b>Attention dimensions are no longer fixed.</b> We vary heads, Q/KV ranks, QK/value head dimensions and indexer dimensions independently, then compare coordinated attention/body widening against body-only widening. Sparse top-K2048 and the configured four-layer index reuse remain fixed. This is an inference-cost study, not an equal-quality model comparison.</div>
<h2>Important calibration correction discovered during the extension</h2><p>The pinned AIC decode SOL formula uses the same indexer work for full-index and reused-index layers. The earlier~0.7× native GLM calibration therefore included avoidable indexer work. This extension introduces a local analytical reuse correction, keeps the old files unchanged, and re-evaluates the same98 observed GLM strata (117,172 iterations). No new hardware measurement was made.</p>''']
    obs=calibration['observed_ms']
    parts.append(table(['Native GLM matched comparison','AIC weighted step ms','Measured GPU step ms','Achieved / modeled SOL rate'],[
        ['Prior formula, live batch',f"{calibration['old_sol_ms']:.2f}",f'{obs:.2f}',f"{calibration['old_sol_ms']/obs:.3f}×"],
        ['Index-reuse-aware, live batch',f"{calibration['reuse_sol_ms']:.2f}",f'{obs:.2f}',f"{calibration['reuse_sol_ms']/obs:.3f}×"],
        ['Index-reuse-aware, padding-aware hybrid',f"{calibration['reuse_hybrid_ms']:.2f}",f'{obs:.2f}',f"{calibration['reuse_hybrid_ms']/obs:.3f}×"],
        ['Index-reuse-aware, fully padded sensitivity',f"{calibration['reuse_padded_ms']:.2f}",f'{obs:.2f}',f"{calibration['reuse_padded_ms']/obs:.3f}×"]]))
    parts.append('''<p><b>The old0.7 factor is not validated for the new formula.</b> The new conditional native anchor is about0.45–0.52, not proof that every scaled model achieves that range. Missing all-rank state, exact MTP execution, heterogeneous contexts and kernel overlap remain. The same backend/shape/precision paired baseline bridge is available below; its old-formula synthetic5T step is10.90ms and reuse-aware step9.78ms at EP64,4 requests/rank,no speculation.</p>
<h2>Main architectural findings</h2><p>The following examples are~5T,EP64/TP1,four requests/rank,no speculation,GB300 SOL. They compare the same MoE body unless explicitly identified as a coordinated pair.</p><ul>
<li><b>KV rank is primarily a capacity knob at this load.</b> Rank512→256 reduces modeled256K cache from14.70→9.83GiB/request/rank, raises memory-max concurrency704→1088, and changes decode only102.3→102.8TPS/user. Rank1024 raises cache to24.45GiB and reduces memory-max concurrency to384, with101.2TPS/user at the fixed four-request shape. Sparse attention reads selected KV, not the whole resident cache.</li>
<li><b>Head count changes latency without multiplying latent-cache size.</b> Heads64→128 reduce decode102.3→80.7TPS/user. Cache/request stays14.70GiB, but larger weights also reduce memory headroom.</li>
<li><b>Body-only widening understates the cost of coordinated widening.</b> At L64/H12288, keeping native attention internals predicts84.9TPS/user; scaling heads/Q/KV/indexer heads predicts57.9. Memory-max concurrency drops768→320 at the same body parameters. Neither pair is quality-matched.</li>
<li><b>Compacting attention can buy both speed and concurrency.</b> The compact bundle gives135.5TPS/user and1344 memory-max requests versus102.3 and704 for baseline. This is an aggressive lower-dimensional hypothesis, not evidence it preserves frontier-model capability.</li>
<li><b>Prefill responds differently.</b> In the EP/TP throughput screen, the~5T GB300 baseline reaches10.4K new input TPS/GPU; compact reaches15.7K and expanded4.5K. These independently optimized rates include batching differences, unlike the fixed-layout decode examples.</li></ul>
<h2>Study design</h2>''')
    parts.append(table(['Axis','Baseline','Alternatives','Primary implication'],[
        ['Attention heads','64','32 / 128','More projection/attention work; latent KV remains shared across heads'],
        ['Q low-rank width','2048','1024 / 4096','Query projections; does not change persistent latent KV size'],
        ['KV low-rank width','512','256 / 1024','Persistent KV capacity and selected-cache read/attention work'],
        ['Non-RoPE QK dimension','192','96 / 384','Q expansion and absorbed attention transforms'],
        ['Value-head dimension','256','128 / 512','Value/output transformations'],
        ['Indexer heads','32','16 / 64','Index projection/logit compute, not one separate cache per head'],
        ['Indexer head dimension','128','64 / 256','Indexer key-cache size, traffic and compute'],
        ['Coordinated scale-up','Fixed internals at H8192/12288','Increase heads, Q/KV rank and indexer heads with residual width','Measures cost omitted by widening only the body']]))
    parts.append('''<p>One-variable sweeps keep the MoE body fixed:78 layers, residual width6144, expert width2048, one shared expert, E1728/K54 at~5T and E3392/K106 at~10T. Attention changes intentionally change total/active parameters; the routed body budget does not change. Joint pairs use the same L/H/M/E/K within each pair, so only attention dimensions differ. They cover L64/78/96 and H8192/12288. Expert M is rounded128 to hold approximate routed capacity; exact totals are in the manifest.</p>
<p>All synthetic layers are MoE. Quantization is NVFP4 routed experts, BF16 non-expert GEMMs and FP8 latent KV; the modeled RoPE portion is BF16. Context=262,144 cached tokens; prefill adds1,024 new tokens. Worker sizes16/32/64 and attention TP1/2/4/8 are searched within a64-GPU normalization. Decode includes no-spec and four-proposal/AL3.36 scenarios. This AL is borrowed, not measured for these architectures.</p>
<h2>One-variable performance controls</h2><p>EP64,attention TP1,four requests/rank,no speculation. Each axis has its own baseline; y=1 means unchanged throughput relative to the corrected baseline on the same hardware. Lines connect three analytical samples, not measurements.</p>''')
    axespec=[('heads','Attention heads'),('q_lora_rank','Q rank'),('kv_lora_rank','KV rank'),('qk_nope_head_dim','QK non-RoPE dimension'),('v_head_dim','Value dimension'),('index_n_heads','Indexer heads'),('index_head_dim','Indexer dimension')]
    for t in (5,10):
        base=next(d for d in ds if d['target_t']==t and d['profile']=='baseline')
        fig,axs=plt.subplots(2,4,figsize=(14,6.4));axs=axs.flatten()
        for ax,(field,label) in zip(axs,axespec):
            dd=sorted([base]+[d for d in ds if d['target_t']==t and d['family']=='one_axis' and d['profile'].startswith(field+'_')],key=lambda d:d[field])
            for hw,color in [('GB300','#c66b18'),('VR200-dense','#008774')]:
                ref=point(base,hw);q=[(d,point(d,hw)) for d in dd];q=[(d,r) for d,r in q if r and ref]
                ax.plot([d[field] for d,r in q],[r['tps_user']/ref['tps_user'] for d,r in q],'o-',label=hw,color=color)
            ax.axhline(1,color='gray',ls=':',lw=1);ax.set(title=label,xscale='log',ylabel='TPS/user ÷ baseline')
            ax.set_xticks([d[field] for d in dd],labels=[str(d[field]) for d in dd]);ax.set_ylim(.75,1.2);ax.minorticks_off();ax.grid(alpha=.2)
        axs[7].axis('off');handles,labels=axs[0].get_legend_handles_labels();axs[7].legend(handles,labels,loc='center')
        fig.suptitle(f'~{t}T: corrected SOL, EP64/TP1, 4 requests/rank, no speculation');fig.tight_layout();name=f'axes_{t}.svg';fig.savefig(OUT/name)
        if t==5:fig.savefig(OUT/'preview.png')
        plt.close(fig);parts.append(f'<img src="{name}" alt="{t}T one-variable attention controls">')
    controls=[]
    for d in ds:
        if d['family']=='joint':continue
        controls.append([d['target_t'],d['profile'],f"{d['approximate_active_b']:.1f}",f"{d['kv_bytes_per_token']*262144/2**30:.2f}",rate(d,'GB300'),rate(d,'VR200-dense'),int(lm[d['id'],'GB300',0]['max_requests64']),int(lm[d['id'],'VR200-dense',0]['max_requests64'])])
    parts.append(table(['Target T','Variant','Approx active B','256K cache GiB/request/rank','GB300 TPS/user','Rubin TPS/user','GB300 memory-max N/64 GPUs','Rubin memory-max N/64 GPUs'],controls))
    parts.append('''<h2>Coordinated attention/body widening</h2><p>Fixed-inner and scaled-inner rows share the exact same MoE body. Scaled-inner H8192 uses96 heads,Q2560,KV768,indexer48 heads; H12288 uses128 heads,Q4096,KV1024,indexer64 heads. QK/value/indexer head dimensions remain at baseline. Values below use EP64/TP1,no speculation. A memory rejection is shown rather than presenting its hypothetical rate as feasible.</p>''')
    pairs=[]
    for t in (5,10):
        for l in (64,78,96):
            for h in (8192,12288):
                a=next(d for d in ds if (d['target_t'],d['family'],d['profile'],d['layers'],d['hidden'])==(t,'joint','fixed_inner',l,h))
                b=next(d for d in ds if (d['target_t'],d['family'],d['profile'],d['layers'],d['hidden'])==(t,'joint','scaled_inner',l,h))
                for hw in ('GB300','VR200-dense'):
                    aa=point(a,hw);bb=point(b,hw)
                    pairs.append([t,l,h,hw,rate(a,hw,1)+' → '+rate(b,hw,1),rate(a,hw)+' → '+rate(b,hw),f"{bb['tps_user']/aa['tps_user']:.3f}×" if aa and bb else 'Not jointly admitted',f"{a['kv_bytes_per_token']*262144/2**30:.1f} → {b['kv_bytes_per_token']*262144/2**30:.1f}",f"{int(lm[a['id'],hw,0]['max_requests64'])} → {int(lm[b['id'],hw,0]['max_requests64'])}"])
    parts.append(table(['Target T','Layers','Residual H','Hardware','1 request/rank TPS fixed→scaled','4 requests/rank TPS fixed→scaled','Rate ratio at4/rank','Cache GiB/request fixed→scaled','Memory-max N fixed→scaled'],pairs))
    parts.append('''<h2>Optimized50/100 TPS-user operating points</h2><p>Each row selects the largest concurrency among searched configurations meeting the target. These are ideal load-balanced memory and SOL constraints, not measured admission limits. “0.7” is retained only as an explicitly hypothetical legacy scenario; the reuse-aware native calibration no longer supports it as a default.</p>''')
    parts.append('<details><summary>All optimized decode configurations</summary>'+table(['Model','Hardware','Efficiency scenario','Target','Draft tokens','EP/TP','Requests/group','Total N/64 GPUs','TPS/user','Output TPS/GPU','Memory-max N','GiB/rank'],[[r['id'],r['system'],r['efficiency'],r['target'],int(r['nextn']),f"{int(r['world'])}/{int(r['tp'])}",int(r['cpr']),int(r['requests64']),f"{r['tps_user']:.1f}",f"{r['throughput_gpu']:.1f}",int(r['memory_max_requests64']),f"{r['memory_gib']:.1f}"] for r in best])+'</details>')
    pf=[]
    for d in ds:
        for hw in ('GB300','VR200-dense'):
            q=[r for r in grid if r['id']==d['id'] and r['system']==hw and r['phase']=='prefill']
            if not q:continue
            peak=max(q,key=lambda r:r['throughput_gpu']);near=min((r for r in q if r['throughput_gpu']>=.95*peak['throughput_gpu']),key=lambda r:r['step_ms'])
            pf.append(dict(id=d['id'],system=hw,peak_input_tps_gpu=peak['throughput_gpu'],near_input_tps_gpu=near['throughput_gpu'],near_job_ms=near['step_ms'],world=near['world'],tp=near['tp'],cpr=near['cpr'],requests64=near['requests_64gpu']))
    write('best_prefill.csv',pf)
    parts.append('<h2>Prefill supplement</h2><p>Same256K prefix plus1K new input. PCP is not searched; this is an EP/TP throughput screen, not a claim of global prefill optimality.</p><details><summary>All prefill results</summary>'+table(['Model','Hardware','Peak input TPS/GPU','Near-peak input TPS/GPU','Near-peak job ms','EP/TP','Requests/group','N/64 GPUs'],[[r['id'],r['system'],f"{r['peak_input_tps_gpu']:.0f}",f"{r['near_input_tps_gpu']:.0f}",f"{r['near_job_ms']:.1f}",f"{int(r['world'])}/{int(r['tp'])}",int(r['cpr']),int(r['requests64'])] for r in pf])+'</details>')
    parts.append('''<h2>Checks, mechanisms and limitations</h2><ul>
<li>Every changed dimension is passed both into the model config and the analytical DSA dimension registry. The29 distinct5T configurations passed an explicit cache-memory check against L×tokens×(KV rank+2×RoPE dimension+indexer entry bytes). The10T counterparts have the same attention dimensions. This is simulator consistency, not a measured cache-allocation validation.</li>
<li>Heads and Q rank do not directly multiply stored latent KV. KV rank and indexer head dimension do. Indexer head count affects compute/projections but the key cache is shared across indexer heads.</li>
<li>The reuse correction removes index-only projection, logit and cache-read terms on reused-index layers, retaining Q/KV/output and sparse-attention work. It uses the existing absorbed-MLA SOL algebra, exact21/78 full-index fraction at depth78, and the SGLang graph for all layouts. It is a local analytical extension, not an upstream AIC fix or measured kernel.</li>
<li>The configured index_topk=2048, index_topk_freq=4 and index_skip_topk_offset=3 do not vary. Depth changes the exact full-layer fraction slightly. Model memory remains AIC's per-layer latent/indexer allocation; actual retained or shared caches could differ.</li>
<li>Shared-expert scaling correction from the body study remains. Ideal shared/routed overlap is still assumed. Expert balancing, hardware contention, exact MTP index reuse and padding are not fully modeled.</li>
<li>Some dimensions may need new or specialized kernels. SOL can evaluate the shapes without establishing backend support or efficiency. Compact attention is a lower-cost hypothesis, not evidence of equal capability. Expanded attention plausibly improves representational capacity but quality must be measured.</li>
<li>Rubin uses the inherited dense-spec hardware proxy and conservative decimal288GB capacity; GB300 uses driver-visible277.55GiB. Apparent lower Rubin memory-max concurrency is not evidence of a real capacity regression. No Rubin hardware calibration exists here.</li></ul>
<h2>Architecture manifest and raw results</h2>''')
    parts.append('<details><summary>All58 architecture definitions</summary>'+table(['ID','L/H/M','E/K','Heads','Q rank','KV rank','QK non-RoPE','Value dim','Indexer heads/dim','Routed T','Approx total T','Approx active B'],[[d['id'],f"{d['layers']}/{d['hidden']}/{d['expert_width']}",f"{d['experts']}/{d['active_experts']}",d['heads'],d['q_lora_rank'],d['kv_lora_rank'],d['qk_nope_head_dim'],d['v_head_dim'],f"{d['index_n_heads']}/{d['index_head_dim']}",f"{d['routed_total_t']:.3f}",f"{d['approximate_total_t']:.3f}",f"{d['approximate_active_b']:.1f}"] for d in ds])+'</details>')
    files=['grid.csv','slo.csv','memory_limits.csv','best_decode.csv','best_prefill.csv','dimension_checks.csv','baseline_bridge.csv','native_recalibration.csv','native_recalibration.json','manifest.json']
    parts.append('<ul>'+''.join(f'<li><a href="{f}">{f}</a></li>' for f in files)+'</ul><p><a href="../definitions.json">Definitions</a> · <a href="../prepare.py">Generator</a> · <a href="../study.py">Adapter and runner</a> · <a href="../validate_dimensions.py">Dimension checks</a> · <a href="../recalibrate.py">Native recalibration</a> · <a href="../../glm-body-design-study/output/GLM_BODY_DESIGN_STUDY.html">Earlier body study (old formula)</a>. Full AIC records are in aic_cache.jsonl.</p>')
    css='body{font:16px/1.55 system-ui;max-width:1500px;margin:32px auto;padding:0 24px;background:#f7fafb;color:#183043}h2{margin-top:38px}.callout{padding:20px;background:#e0f1ed;border-left:5px solid #008774}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:13px}td,th{padding:9px;border-bottom:1px solid #dae4ea;text-align:left}th{background:#e5eef2}img{max-width:100%;display:block;margin:24px auto}summary{cursor:pointer;font-weight:600}a{color:#007866}'
    (OUT/'GLM_ATTENTION_DIMENSIONS.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GLM attention dimensions</title><style>'+css+'</style><main>'+''.join(parts)+'</main></html>')
    sources=[HERE/f for f in ('prepare.py','study.py','report.py','validate_dimensions.py','recalibrate.py','definitions.json')]+[OUT/f for f in files if f!='manifest.json']+[Path(d['model'])/'config.json' for d in ds]+[HERE.parent/'glm-body-design-study/study.py',HERE.parent/'frontier-hardware-study/engine.py',HERE.parent/'frontier-hardware-study/output/hardware.json',HERE.parent/'inferencex-agentx-grounding/output/matched_decode_strata.csv',HERE.parent/'inferencex-agentx-grounding/models/glm/config.json',HERE.parent/'inferencex-agentx-grounding/models/glm/hf_quant_config.json']
    aic=HERE.parents[1]/'repos/aiconfigurator-kimi-k3-pr1435/aic-core/src/aiconfigurator_core/sdk'
    sources.extend(aic/p for p in ('operations/dsa.py','models/deepseek_v32.py','common.py'))
    (OUT/'manifest.json').write_text(json.dumps({'models':len(ds),'grid_rows':len(grid),'slo_rows':len(slo),'memory_layouts':len(limits),'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
    print(OUT/'GLM_ATTENTION_DIMENSIONS.html')
if __name__=='__main__':main()
