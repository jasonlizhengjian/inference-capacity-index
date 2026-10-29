"""Unfitted comparisons to a dated public InferenceX snapshot.

Standard-library curve analysis plus optional pinned AISim replay of logged GLM
strata and explicitly approximate Kimi configuration scenarios.
"""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import statistics as st
import sys

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
REF=ROOT/'reference/2026-10-15';OUT=ROOT/'docs/research/inferencex-validation'
RAW=ROOT/'results/inferencex-validation-2026-10-19/raw_logs'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'studies')]
from agent_index.index import capacity, area, envelope
from log400_release import combined_area

HW={'gb300':'GB300 NVL72','gb200':'GB200 NVL72','b300':'B300 · 800G','b200':'B200 · 800G',
    'mi355x':'MI355X · 400G','h200':'H200 SXM · FP8 · 400G','tpuv7':'TPU7x Ironwood · FP8','vr200':'VR200'}

def save(name, obj):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def csvout(name, rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def snapshots():
    return {model:json.loads((HERE/'evidence'/(model+'.json')).read_text()) for model in ('Kimi-K3','GLM-5.2')}

def scenario_summary(g, decode_efficiency, prefill_efficiency, low, high):
    """Shift native speed, not capacity; integrate exact unsmoothed frontiers."""
    shifted=[dict(p,speed=p['speed']*decode_efficiency) for p in g['sol_frontier']]
    prefill=g['prefill_rate']*prefill_efficiency;r=g['input_output_ratio']
    edges=sorted({low,high,*(p['speed'] for p in shifted if low<p['speed']<high)})
    ca=math.fsum(c*math.log1p((b-a)/(a*(1+r*c*b/prefill)))
                for a,b in zip(edges,edges[1:]) if (c:=capacity(shifted,(a+b)/2)))
    qfront=envelope([dict(p,capacity=p['capacity']*p['speed']/(1+r*p['capacity']*p['speed']/prefill)) for p in shifted])
    qa=area(qfront,low,high)
    measured_c=area(g['frontier'],low,high);measured_q=area(g['throughput_frontier'],low,high)
    return dict(model=g['model'],hardware=g['hardware'],decode_efficiency=decode_efficiency,prefill_efficiency=prefill_efficiency,
                low=low,high=high,measured_capacity_auc=measured_c,scenario_capacity_auc=ca,
                measured_capacity_ratio=measured_c/ca if ca else None,
                measured_output_auc=measured_q,scenario_output_auc=qa,measured_output_ratio=measured_q/qa if qa else None,
                scenario_max_speed=max(p['speed'] for p in shifted),
                observations_above_scenario_max=sum(p['speed']>max(x['speed'] for x in shifted) for p in g['observations']))

def curves():
    from rubin_release import report_inputs, REV
    revised=report_inputs()
    saved=json.loads((revised/'curves_and_inputs.json').read_text())
    rates={(r['hardware'],r['model']):r['tokens_chip_second'] for r in saved['prefill_rates']}
    groups=[];points=[];exclusions=[]
    for model, rows in snapshots().items():
        for hw, key in HW.items():
            chosen=[]
            for r in rows:
                if r['hardware']!=hw:continue
                m=r['metrics']
                if r['benchmark_type']!='agentic_traces' or not m.get('mean_tpot') or not m.get('output_tput_tps'):
                    exclusions.append(dict(id=r['id'],reason='Not AgentX with positive TPOT/output throughput'));continue
                # Aggregated metadata lists the same GPUs in both phase fields.
                n=r['num_decode_gpu']+r['num_prefill_gpu'] if r['disagg'] else max(r['num_decode_gpu'],r['num_prefill_gpu'])
                q=m['output_tput_tps']/n;speed=1/m['mean_tpot']
                assert math.isclose(q,m['output_tput_per_gpu'],rel_tol=2e-5,abs_tol=2e-5),(r['id'],q,m['output_tput_per_gpu'])
                point=dict(model=model,hardware=key,id=r['id'],date=r['date'],run_url=r['run_url'],framework=r['framework'],precision=r['precision'],spec_method=r['spec_method'],
                    chips=n,prefill_chips=r['num_prefill_gpu'] if r['disagg'] else 0,decode_or_shared_chips=r['num_decode_gpu'],disaggregated=r['disagg'],client_concurrency=r['conc'],
                    speed=speed,output_tps_chip=q,capacity=q/speed,mean_context=m.get('mean_input_tokens'),mean_output=m.get('mean_output_tokens_actual'),
                    decode_tp=r['decode_tp'],decode_ep=r['decode_ep'],decode_dp_attention=r['decode_dp_attention'],api_dcp=m.get('decode_dcp_size',m.get('dcp_size')),spec_tokens=None,
                    api_url='https://inferencex.semianalysis.com/api/v1/benchmarks?model='+model)
                chosen.append(point);points.append(point)
            if not chosen or model not in saved['curves'].get(key,{}):continue
            sol=saved['curves'][key][model];p=rates[key,model];ratio=saved['ratio']
            # Q/N/s is throughput-equivalent streaming density, NOT measured live concurrency.
            observed=envelope(chosen)
            throughput=envelope([dict(x,capacity=x['output_tps_chip']) for x in chosen])
            sol_q=envelope([dict(x,capacity=x['capacity']*x['speed']/(1+ratio*x['capacity']*x['speed']/p)) for x in sol])
            obs_area=area(observed);sol_area=combined_area(sol,p,ratio)
            obs_q_area=area(throughput);sol_q_area=area(sol_q)
            maxspeed=max(x['speed'] for x in chosen);shared_high=min(400,maxspeed)
            edges=sorted({50.,400.,*[x['speed'] for x in chosen+sol if 50<x['speed']<400]})
            def vals(s):
                c=capacity(sol,s);return dict(speed=s,measured_capacity=capacity(observed,s),sol_capacity=c/(1+ratio*s*c/p),measured_output=capacity(throughput,s),sol_output=capacity(sol_q,s))
            segments=[dict(low=a,high=b,**vals((a+b)/2)) for a,b in zip(edges,edges[1:])]
            # Numerical check split at every exact breakpoint; integration is not based on this plotting grid.
            numeric=0
            for a,b in zip(edges,edges[1:]):
                dx=math.log(b/a)/64
                numeric+=sum(vals(math.exp(math.log(a)+(i+.5)*dx))['sol_capacity']*dx for i in range(64))
            assert math.isclose(numeric,sol_area,rel_tol=1e-5)
            notes=[]
            if hw in ('h200','tpuv7') and model=='Kimi-K3':notes.append('Published FP4 weights versus index FP8 expert storage: not precision-matched.')
            if hw=='mi355x':notes.append('Mixed recipes and dates; index uses conservative node-local collectives/serial PP. Actual communication and precision vary.')
            if hw=='vr200':notes.append('20 October Rubin profile: 19.2 TB/s HBM, 1.5 TB/s outbound NVLink; measured-system specifications are not independently matched.')
            if len({r['precision'] for r in chosen})>1:notes.append('Observed envelope includes multiple precisions; see per-point evidence.')
            if hw=='tpuv7':notes.append('32 physical chips in the listed P/D deployments; TP32 is a logical-device count, not 32 physical chips per pool. Private serving branch; logs unavailable.')
            groups.append(dict(model=model,hardware=key,observations=chosen,frontier=observed,throughput_frontier=throughput,sol_frontier=sol,sol_throughput_frontier=sol_q,prefill_rate=p,input_output_ratio=ratio,
                measured_capacity_auc=obs_area,sol_capacity_auc=sol_area,capacity_auc_ratio=obs_area/sol_area,
                measured_throughput_auc=obs_q_area,sol_throughput_auc=sol_q_area,throughput_auc_ratio=obs_q_area/sol_q_area,
                measured_max_speed=maxspeed,observed_log_band_coverage=max(0,math.log(shared_high/50)/math.log(8)),
                observed_min_speed=min(x['speed'] for x in chosen),mean_context_range=[min(x['mean_context'] for x in chosen),max(x['mean_context'] for x in chosen)],
                dates=sorted({x['date'] for x in chosen}),notes=notes,plot_segments=segments,targets=[vals(s) for s in (50,100,200,400)]))
    scenarios=[]
    for g in groups:
        g['efficiency_scenarios']=[]
        for name,lo,hi in [('observed_range',g['observed_min_speed'],g['measured_max_speed']),('index_band',50.,400.)]:
            for d,p in [(1.,1.),(.5,1.),(.5,.5)]:
                s=dict(scenario_summary(g,d,p,lo,hi),band=name)
                g['efficiency_scenarios'].append(s);scenarios.append(s)
    save('curves.json',dict(snapshot='2026-10-19',modeled_revision='2026-10-20-rubin',refreshed='2026-10-29',band=[50,400],groups=groups,excluded=exclusions,manifest=json.loads((HERE/'evidence/manifest.json').read_text()),reference_sha256=hashlib.sha256((revised/'curves_and_inputs.json').read_bytes()).hexdigest(),revision_manifest_sha256=hashlib.sha256((REV/'manifest.json').read_bytes()).hexdigest()))
    csvout('efficiency_scenarios.csv',scenarios)
    csvout('observed_points.csv',points)
    csvout('curve_summary.csv',[{k:g[k] for k in ('model','hardware','measured_capacity_auc','sol_capacity_auc','capacity_auc_ratio','measured_throughput_auc','sol_throughput_auc','throughput_auc_ratio','measured_max_speed','observed_log_band_coverage')} for g in groups])
    print('Curve comparisons',[(g['model'],g['hardware'],round(g['capacity_auc_ratio'],3),round(g['throughput_auc_ratio'],3)) for g in groups],flush=True)

def profile_window(folder):
    text=(folder/'benchmark.out').read_text(errors='replace')
    found=re.findall(r'(\d\d:\d\d:\d\d).*?profiling phase sending',text)
    assert len(found)==2,(folder,len(found))
    return found

def native():
    from agent_index import simulation as sim
    from agent_index.current_native import doctor
    runtime=doctor(REF);logging.getLogger().setLevel(logging.ERROR)
    catalog=json.loads((REF/'catalog.json').read_text());perf,*_=sim._sdk()
    def engine(model,hw,tp,dp,dcp,ep,nn,backend,version,quant='index'):
        c=sim.make_config(catalog[hw],model,tp,dp,dcp,tp*dp//ep,ep,1,REF/'hardware')
        c.update(nextn=nn,backend=backend,backend_version=version)
        if quant=='checkpoint':c.pop('gemm_quant_mode');c.pop('moe_quant_mode')
        elif quant=='mxfp4_fp8':c.update(moe_quant_mode='w4a8_mxfp4_mxfp8')
        elif quant=='fp8_linear':c.update(gemm_quant_mode='fp8',moe_quant_mode='w4a8_mxfp4_mxfp8')
        return perf.best_available(c)
    def timing(e,b,ctx):return sum(o['latency_ms'] for o in e.static_phase_diagnostics(batch_size=b,context_length=int(ctx),prefill=False))
    allrows={int(r['id']):(m,r) for m,rs in snapshots().items() for r in rs}
    rx=re.compile(r'iter = (\d+).*?num_scheduled_requests = (\d+).*?prev_device_step_time = (N/A|[\d.]+ms), timestamp = ([\d\- :]+), states = (\{.*\})')
    strata=[];summaries=[];audits=[]
    for ident in (444604,444605,444607,444602):
        folder=RAW/str(ident);log=next(folder.glob('*decode_w0.out'));start,end=profile_window(folder)
        model,row=allrows[ident];world=row['decode_tp'];dp=world if row['decode_dp_attention'] else 1;tp=world//dp
        text=log.read_text(errors='replace');version=re.search(r'TensorRT LLM version: (\S+)',text)[1]
        eng=engine(model,'GB300 NVL72',tp,dp,1,row['decode_ep'],3,'trtllm',version,'checkpoint')
        stock=engine(model,'GB300 NVL72',tp,dp,1,row['decode_ep'],3,'trtllm',version,'index')
        buckets=defaultdict(list);counts=Counter();previous=None;sample_times=[]
        for lineno,line in enumerate(text.splitlines(),1):
            m=rx.search(line)
            if not m:continue
            it,b,device,t,state=m.groups();current=dict(it=int(it),b=int(b),device=float(device[:-2]) if device!='N/A' else None,t=t[-8:],line=lineno,**ast.literal_eval(state))
            if previous and start<=previous['t']<=end and start<=current['t']<=end:
                p=previous;counts['profile_pairs']+=1
                if current['it']!=p['it']+1:counts['nonconsecutive']+=1
                elif current['device'] is None:counts['missing_time']+=1
                elif p['num_ctx_requests'] or p['num_ctx_tokens']:counts['mixed_or_prefill']+=1
                elif not p['b'] or p['cached_kv_tokens']/p['b']<1024:counts['empty_or_tiny_context']+=1
                elif p['num_generation_tokens']%4 or p['num_generation_tokens']//4<p['b']:counts['unexpected_verify_shape']+=1
                else:
                    ctx=p['cached_kv_tokens']/p['b'];pad=p['num_generation_tokens']//4
                    buckets[p['b'],pad,int(ctx//32768)].append(dict(context=ctx,observed_ms=current['device'],line=p['line'],timing_line=lineno))
                    counts['eligible']+=1;sample_times.append(p['t'])
            previous=current
        selected=[]
        for (b,pad,bin),samples in sorted(buckets.items()):
            if len(samples)<20:continue
            ctx=round(st.median(x['context'] for x in samples));obs=st.median(x['observed_ms'] for x in samples)
            rep=min(samples,key=lambda x:abs(x['context']-ctx))
            r=dict(result_id=ident,model=model,tp=tp,dp=dp,ep=row['decode_ep'],dcp=1,nextn=3,batch=b,padded_batch=pad,context=ctx,count=len(samples),observed_device_ms=obs,
                checkpoint_live_ms=timing(eng,b,ctx),checkpoint_padded_ms=timing(eng,pad,ctx),index_precision_live_ms=timing(stock,b,ctx),representative_line=rep['line'],timing_line=rep['timing_line'])
            selected.append(r);strata.append(r)
        total=sum(r['count'] for r in selected)
        if total:
            avg=lambda key:sum(r[key]*r['count'] for r in selected)/total
            summaries.append(dict(result_id=ident,model=model,tp=tp,dp=dp,ep=row['decode_ep'],covered_iterations=total,eligible_iterations=counts['eligible'],strata=len(selected),
                observed_device_ms=avg('observed_device_ms'),sol_live_ms=avg('checkpoint_live_ms'),sol_padded_ms=avg('checkpoint_padded_ms'),live_fraction=avg('checkpoint_live_ms')/avg('observed_device_ms'),padded_fraction=avg('checkpoint_padded_ms')/avg('observed_device_ms'),index_precision_fraction=avg('index_precision_live_ms')/avg('observed_device_ms'),
                profile_window=[start,end],observed_sample_window=[min(sample_times),max(sample_times)],source_log_complete=json.loads((folder/(log.name+'.meta.json')).read_text())['complete'],backend_version=version))
        audits.append(dict(result_id=ident,counts=dict(counts),source=json.loads((folder/(log.name+'.meta.json')).read_text())))
    save('matched_glm.json',dict(runtime=runtime,summaries=summaries,audits=audits,strata=strata));csvout('matched_glm_strata.csv',strata)
    print('GLM matched',summaries,flush=True)
    # Kimi: expose the observational and backend limitations rather than calling
    # these interval-level proxies exact matched-step measurements.
    kimi=[];scenarios=[]
    for ident in (445218,445220,445227,445368,445370):
        folder=RAW/str(ident)
        if not folder.exists() or not (folder/'benchmark.out').exists():continue
        log=next(folder.glob('*w0.out'));text=log.read_text(errors='replace');model,row=allrows[ident];start,end=profile_window(folder)
        config=next(l for l in text.splitlines() if 'Initializing a V1 LLM engine' in l)
        tp=int(re.search(r'tensor_parallel_size=(\d+)',config)[1]);dcp=int(re.search(r'decode_context_parallel_size=(\d+)',config)[1]);nn=int(re.search(r'num_spec_tokens=(\d+)',config)[1])
        version=re.search(r'engine \(v([^)]*)\)',config)[1]
        intervals=[];accept=[];last_time=None
        for line in text.splitlines():
            mt=re.search(r'INFO \d\d-\d\d (\d\d:\d\d:\d\d)',line)
            if mt:last_time=mt[1]
            if not last_time or not start<=last_time<=end:continue
            m=re.search(r'Avg prompt throughput: ([\d.]+).*?Avg generation throughput: ([\d.]+).*?Running: (\d+)',line)
            if m:intervals.append(dict(prompt=float(m[1]),output=float(m[2]),running=int(m[3])))
            m=re.search(r'Mean acceptance length: ([\d.]+).*?Accepted: (\d+) tokens, Drafted: (\d+) tokens',line)
            if m and int(m[3]):accept.append((int(m[2]),int(m[3])))
        if not intervals:continue
        # In vLLM these counters exclude the bonus token. Drafted/gamma estimates
        # speculative steps; account for one target token in accepted length.
        al=1+nn*sum(x[0] for x in accept)/sum(x[1] for x in accept) if accept else None
        running=[x['running'] for x in intervals if x['running']>0];b=round(st.median(running));ctx=round(row['metrics']['mean_input_tokens']);hw=HW[row['hardware']]
        record=dict(result_id=ident,hardware=hw,tp=tp,dcp=dcp,api_dcp=row['metrics'].get('decode_dcp_size',row['metrics'].get('dcp_size')),ep=1,etp=tp,dp=1,nextn=nn,observed_al=al,backend_version=version,
            intervals=len(intervals),positive_running_median=b,positive_running_min=min(running),positive_running_max=max(running),mean_input_context=ctx,mean_tpot_ms=row['metrics']['mean_tpot']*1000,
            exact_match=False,reason='No per-step batch/context/device timing; sampled running requests are not scheduled kernel batches. MXFP4 and mixed precision differ from index assumptions. Logged AL is an execution statistic, not evidence of natural acceptance on this trace; recipes can impose synthetic acceptance.',
            source=json.loads((folder/(log.name+'.meta.json')).read_text()))
        if nn>5:
            record['native_backend_error']='Pinned AISim only accepts nextn 0..5; observed nextn='+str(nn)
            record['reason']+=' No timing replay: actual speculative width exceeds native support. Width is not clamped.'
            kimi.append(record);continue
        backend='vllm'
        try:e=engine(model,hw,tp,1,dcp,1,nn,backend,version)
        except ValueError as exc:
            record['native_backend_error']=str(exc);backend='sglang';e=engine(model,hw,tp,1,dcp,1,nn,backend,'0.5.17')
            record['reason']+=' Native Kimi/vLLM DCP unsupported: SGLang graph substituted, including different draft graph.'
        record['modeled_backend']=backend
        for batch in sorted({1,b,min(running),max(running)}):
            for context_label,context in [('mean',ctx),('p50',round(row['metrics']['median_input_tokens'])),('p90',round(row['metrics']['p90_input_tokens']))]:
                for quant in (['index','mxfp4_fp8','fp8_linear'] if hw.startswith('MI') else ['index','mxfp4_fp8']):
                    e=engine(model,hw,tp,1,dcp,1,nn,backend,version if backend=='vllm' else '0.5.17',quant)
                    ms=timing(e,batch,context)
                    scenarios.append(dict(result_id=ident,hardware=hw,batch=batch,context=context,context_basis=context_label,quantization=quant,nextn=nn,al=al,backend=backend,memory_admission_checked=False,raw_ms=ms,sol_tps=al*1000/ms if al else None,observed_tps=1/row['metrics']['mean_tpot'],observed_over_sol=ms/(al*1000*row['metrics']['mean_tpot']) if al else None))
        kimi.append(record)
    save('kimi_config_audit.json',dict(runtime=runtime,records=kimi,scenarios=scenarios));csvout('kimi_shape_scenarios.csv',scenarios)
    print('Kimi scenarios',len(scenarios),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--native',action='store_true');args=ap.parse_args();curves()
    if args.native:native()
