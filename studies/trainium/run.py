# MoE roofline shape derived from NVIDIA AISimulate; see THIRD_PARTY_NOTICES.md.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Isolated Trainium SOL scenarios; never changes published index inputs.

Run: .venv-current/bin/python studies/trainium/run.py --jobs 2
Uses the frozen four-model workload and exact log50-400 scoring. AWS-specific
MoE byte/compute correction is applied at leaves, inside native overlap trees.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'studies')]
from agent_index.current_native import Native, doctor, KEYS
from agent_index.interconnect import NativeLeaves
from agent_index.index import envelope
from agent_index.topology import events
from log400_release import combined_area, REFERENCE

REF = ROOT/'reference/2026-10-15'
OUT = ROOT/'results/trainium-2026-10-22-audit'
GIB = 2**30
MODELS = json.loads((REF/'curves_and_inputs.json').read_text())['models']
SPECS = {
    2: dict(capacity=96, bandwidth=2.9e12, bf16=632e12, fp8=1264e12,
            budget=64, quant='fp8'),
    3: dict(capacity=144, bandwidth=4.9e12, bf16=632e12, fp8=2517e12,
            budget=144, quant='w4a8_mxfp4_mxfp8'),
}
SOURCES = [
    'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html',
    'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html',
    'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html',
    'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium3.html',
    'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium3_arch.html',
]
# Not confidence intervals. Tuple = domain budget, local group, local / wider
# one-way collective bandwidth, startup us per modeled collective invocation.
# Trn2 ring: reported aggregate /2 directionality /degree(4 local,2 third axis).
# Trn3 detailed: 256/320 GB/s bidirectional local/rack; 128 cross-rack.
CASES = {
    2: {
        '8_chip_control': (8,16,128*GIB,64*GIB,0),
        '16_link_limited': (16,16,128*GIB,64*GIB,0),
        '64_link_limited': (64,16,128*GIB,64*GIB,0),
        '64_link_limited_4us': (64,16,128*GIB,64*GIB,4),
        '64_link_limited_oneway_tables': (64,16,256*GIB,128*GIB,0),
        '64_injection_ceiling': (64,16,1024*GIB,256*GIB,0),
    },
    3: {
        '8_chip_control': (8,4,128e9,160e9,0),
        '64_detailed_tiers': (64,4,128e9,160e9,0),
        '64_aggregate_half': (64,64,1024*GIB,1024*GIB,0),
        '144_detailed_slow_tier': (144,4,128e9,64e9,0),
        '144_detailed_slow_tier_4us': (144,4,128e9,64e9,4),
        '144_aggregate_half': (144,144,1024*GIB,1024*GIB,0),
        '144_injection_ceiling': (144,144,2.56e12,2.56e12,0),
    },
}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def save_csv(p, rows):
    with p.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


class Trainium(Native):
    def __init__(self, generation, model):
        self.gen=generation;self.spec=SPECS[generation]
        hw=dict(name=f'Trainium{generation}',profile=f'trainium{generation}',
                chip_budget=self.spec['budget'],restricted=False,
                quant=self.spec['quant'],hop_overlay=False)
        super().__init__(OUT/'inputs',hw,model)

    def memory(self, phase, layout, batch):
        m=super().memory(phase,layout,batch)
        if self.gen!=3 or 'memory_gib' not in m:return m
        _,g,_,_=self.graph(phase,layout)
        # Exact MXFP4 scale storage: 1 scale byte per 32 parameters, versus
        # native 0.5 byte/parameter => add 1/16 of the native expert payload.
        # Keep the pre-existing generic safety/metadata guard as a separate reserve.
        routed=sum(o.get_weights() for o in g.context_ops
                   if 'Moe' in json.loads(o._spec_json()))
        extra=routed/16/GIB
        m['mx_scale_gib']=extra;m['weights_gib']+=extra;m['memory_gib']+=extra
        m['fits']=m['fits'] and m['memory_gib']<=self.spec['capacity']*.9
        return m


def moe_time(v, x, spec, native=False):
    """Same activated-expert/floor model as native operators/moe.rs::moe_sol.

    Only precision mapping differs: MXFP4 weights plus scale bytes, MXFP8
    input/intermediate traffic, BF16 final output. Compute uses native FP8 peak.
    """
    h,i=v['hidden_size'],v['inter_size'];g=3 if v['is_gated'] else 2
    ep,tp=v['moe_ep_size'],v['moe_tp_size']
    tokens=x*v['attention_dp_size']*v['topk']
    flop=(tokens*h*i*g*2//ep)//tp
    t=tokens//ep
    activations=t*h*2+(t*i*g//tp)
    weights=(h*i*g//tp)*min(v['num_experts']//ep,t)
    if native:
        bw=1 if v['quant_mode']=='fp8' else .5
        memory=(weights+activations)*bw
    elif v['quant_mode']=='fp8':
        memory=weights+activations
    else:
        memory=weights*.53125+t*h*(1.03125+2)+(t*i*g//tp)*1.03125
    return max(flop/spec['fp8'],memory/spec['bandwidth'])*1000*v['scale_factor']


def run_one(task):
    generation,model,signature=task;logging.disable(logging.WARNING);doctor(REF)
    target=OUT/'runs'/f'trn{generation}-{model}.json'
    if target.exists():
        old=json.loads(target.read_text());assert old['signature']==signature
        return str(target)
    started=time.monotonic();engine=Trainium(generation,model)
    cases={k:dict(frontier=[],prefill=None,candidates_decode=0,candidates_prefill=0) for k in CASES[generation]}
    audits=[];max_parity=0.;max_moe_parity=0.
    print('START',generation,model,flush=True)
    for phase in ('decode','prefill'):
        for layout in engine.layouts(phase):
            try:cap=engine.batches(phase,layout)
            except (ValueError,AssertionError) as exc:
                # Record unsupported layouts, not arbitrary timing failures.
                audits.append(dict(phase=phase,**layout,error=str(exc)));engine.cache.clear();continue
            first=math.ceil(layout['ep']/(4*layout['dp']*7)) if phase=='decode' and model.startswith('MAI2') else 1
            audits.append(dict(phase=phase,**layout,first=first,last=cap))
            if cap<first:engine.cache.clear();continue
            c,g,native,_=engine.graph(phase,layout);leaves=NativeLeaves(c,g,phase,False)
            chips=layout['tp']*layout['dp'];spec=engine.spec
            active={k:v for k,v in CASES[generation].items() if chips<=v[0]}
            # Rebuild native engines against the actual tiered system profiles.
            # This also covers communication hidden inside attention modules.
            case_leaves={k:NativeLeaves(c|dict(system=f'trainium{generation}_{k}'),g,phase,False) for k in active}
            front={k:[] for k in active}
            for batch in range(first,cap+1):
                mem=engine.memory(phase,layout,batch);assert mem['fits']
                times=leaves.times(batch);x=batch*(1024 if phase=='prefill' else 4)
                if batch in (first,cap):
                    ms=sum(o['latency_ms'] for o in native.static_phase_diagnostics(batch_size=batch,
                        context_length=262144+(1024 if phase=='prefill' else 0),prefix=262144 if phase=='prefill' else 0,prefill=phase=='prefill'))
                    err=abs(leaves.tree.fold(times)-ms)/ms;max_parity=max(max_parity,err);assert err<1e-10
                adjusted={k:case_leaves[k].times(batch) for k in active}
                for j,s in enumerate(leaves.tree.leaves):
                    kind,v=next(iter(s.items()))
                    if kind=='Moe':
                        orig=moe_time(v,x,spec,True)
                        err=abs(orig-times[j])/max(times[j],1e-12);max_moe_parity=max(max_moe_parity,err);assert err<1e-10,(model,phase,'moe',err,batch,layout,v,orig,times[j])
                        for k in active:adjusted[k][j]=moe_time(v,x,spec)
                    elif leaves.tree.info[j]['communication']:
                        for k,case in active.items():
                            adjusted[k][j]+=leaves.tree.info[j]['calls']*case[4]/1000
                for k in active:
                    ms=leaves.tree.fold(adjusted[k]);assert ms>0
                    requests=batch*layout['dp']
                    row=dict(**layout,batch=batch,chips=chips,requests=requests,capacity=requests/chips,
                             raw_ms=ms,speed=3000/ms if phase=='decode' else 0.,memory=mem['memory_gib'],
                             input_tokens_chip_second=requests*1024*1000/chips/ms if phase=='prefill' else 0.,**mem)
                    cases[k]['candidates_'+phase]+=1
                    if phase=='decode':front[k].append(row)
                    elif cases[k]['prefill'] is None or row['input_tokens_chip_second']>cases[k]['prefill']['input_tokens_chip_second']:
                        cases[k]['prefill']=row
            for k in active:
                if phase=='decode':cases[k]['frontier']=envelope(cases[k]['frontier']+front[k])
            engine.cache.clear()
    save(target,dict(generation=generation,model=model,signature=signature,cases=cases,audits=audits,
        elapsed_seconds=time.monotonic()-started,parity=dict(tree=max_parity,moe=max_moe_parity)))
    print('DONE',generation,model,round(time.monotonic()-started,1),'s',flush=True)
    return str(target)


def summarize(paths):
    ref=json.loads((REF/'curves_and_inputs.json').read_text());ratio=ref['ratio']
    pref={r['model']:r['tokens_chip_second'] for r in ref['prefill_rates'] if r['hardware']==REFERENCE}
    scores=[];targets=[]
    for p in paths:
        r=json.loads(Path(p).read_text());m=r['model']
        for name,c in r['cases'].items():
            rate=c['prefill']['input_tokens_chip_second'] if c['prefill'] else 0.
            a=combined_area(c['frontier'],rate,ratio)
            refa=combined_area(ref['curves'][REFERENCE][m],pref[m],ratio)
            scores.append(dict(generation=r['generation'],scenario=name,model=m,score=a/refa,
                decode_score=combined_area(c['frontier'],rate,0)/combined_area(ref['curves'][REFERENCE][m],pref[m],0),
                prefill_score=rate/pref[m],prefill_tokens_chip_second=rate,**{k:c[k] for k in ('candidates_decode','candidates_prefill')}))
            for speed in (50,100,200,400):
                w=max((p for p in c['frontier'] if p['speed']>=speed),key=lambda p:p['capacity'],default=None)
                targets.append(dict(generation=r['generation'],scenario=name,model=m,target_tps=speed,
                    agents_chip=w['capacity'] if w else 0.,**{k:w[k] if w else None for k in (*KEYS,'batch','chips','requests','raw_ms','memory')}))
    save_csv(OUT/'per_model.csv',scores);save_csv(OUT/'targets.csv',targets)
    overall=[]
    for gen,cs in CASES.items():
        for name in cs:
            rows=[r for r in scores if r['generation']==gen and r['scenario']==name]
            if len(rows)!=len(MODELS):continue
            overall.append(dict(generation=gen,scenario=name,**{k:sum(r[k] for r in rows)/len(rows) for k in ('score','decode_score','prefill_score')}))
    if overall:save_csv(OUT/'overall.csv',overall)
    print(json.dumps(overall,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=1);p.add_argument('--generation',type=int,choices=(2,3),action='append');p.add_argument('--model',action='append');a=p.parse_args()
    runtime=doctor(REF);inputs=OUT/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(REF/'models.json',inputs/'models.json');shutil.copytree(REF/'models',inputs/'models',dirs_exist_ok=True)
    (inputs/'hardware').mkdir(exist_ok=True)
    for gen,s in SPECS.items():
        profile=dict(data_dir='data/estimate_only',gpu=dict(bfloat16_tc_flops=s['bf16'],fp8_tc_flops=s['fp8'],
            mem_bw=s['bandwidth'],mem_capacity=s['capacity']*GIB,mem_bw_empirical_scaling_factor=1,mem_empirical_constant_latency=0,sm_version=103),
            misc=dict(other_mem=4*GIB,nccl_mem={n:0 for n in (1,2,4,8,16,32,64,72,128,144,256)}),
            node=dict(intra_node_bw=1e12,inter_node_bw=1e12,inter_rack_bw=1e12,num_gpus_per_node=256,num_gpus_per_rack=256,p2p_latency=0))
        if gen==3:profile['gpu']['fp4_tc_flops']=s['fp8']
        (inputs/'hardware'/f'trainium{gen}.yaml').write_text(yaml.safe_dump(profile))
        for name,case in CASES[gen].items():
            _,local,near,far,_=case
            changed=profile|dict(node=dict(intra_node_bw=near,inter_node_bw=far,inter_rack_bw=far,
                num_gpus_per_node=local,num_gpus_per_rack=case[0],p2p_latency=0))
            (inputs/'hardware'/f'trainium{gen}_{name}.yaml').write_text(yaml.safe_dump(changed))
    signature={str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),*sorted((ROOT/'src/agent_index').glob('*.py')),REF/'curves_and_inputs.json',REF/'runtime.json',REF/'models.json',*sorted((REF/'models').glob('*/config.json'))]}
    save(OUT/'provenance.json',dict(runtime=runtime,signature=signature,specs=SPECS,cases=CASES,sources=SOURCES,
        axis='log',band=[50,400],nextn=3,accepted_length=3,context=262144,new_prefill_tokens=1024,
        input_output_ratio=json.loads((REF/'curves_and_inputs.json').read_text())['ratio'],
        status='Experimental topology scenarios, not promoted. No Neuron kernel or routing simulation.'))
    tasks=[(g,m,signature) for g in (a.generation or [2,3]) for m in (a.model or MODELS)]
    if a.jobs==1:paths=[run_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:paths=list(pool.map(run_one,tasks))
    summarize(paths)


if __name__=='__main__':main()
