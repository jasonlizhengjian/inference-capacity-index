"""Unified corrected AIC sweep, coherent width/attention extensions and topology probes."""
from __future__ import annotations
import copy
import itertools
import json
import math
import sys
from pathlib import Path
import engine as e

sys.path.insert(0,str(e.ROOT/'experiments/unified-prefill-study'))
import run_unified_prefill as old

CPRS=(1,2,4,8,16,32,64,128)
HW=('GB300','HBM+fabric','VR200-dense')


def dsa_weights(layers):
    # Compatible H=7168 interface; FP8 projections, native GLM internal dimensions.
    h,q,kv,nope,rope,v,heads,ih,idx=7168,2048,512,192,64,256,64,32,128
    return h*q+q*heads*(nope+rope)+h*(kv+rope)+kv*heads*(nope+v)+heads*v*h+(q*ih*idx+h*ih)*e.base.dsa_full_fraction(layers)


def prepare():
    specs=e.base.body_specs()
    refs={l:str(e.base.MODELS/f'attention_ref_kimi_L{l}') for l in e.base.DEPTHS}
    exact=old.exact_models(str(e.base.MODELS/'glm52_ratio_preserved_5t'))
    definitions=[]
    for x in exact:
        definitions.append(dict(x,aid='exact__'+x['id'],kind='exact'))
    for s in specs:
        for aid,p in e.base.ATTENTION_PROFILES.items():
            definitions.append(dict(aid=f'factorial__{s.id}__{aid}',label=f'{s.id} · {p["label"]}',
                                    family='Controlled 5T factorial',kind='factorial',body=s.id,
                                    attention=aid,depth=s.depth,experts=s.experts,topk=s.topk))
    for p in e.swa.PROFILES:
        definitions.append(dict(aid='swa__'+p['id'],label=p['label'],family='SWA and hybrid extensions',kind='swa',profile=p))
    dsa={}
    original=json.loads((e.base.MODELS/'glm52_fp8_dsa_validation/config.json').read_text())
    for layers in (48,61,78,93,116):
        for topk in (512,1024,2048,4096):
            for shared in ((True,False) if topk==2048 else (True,)):
                cfg=copy.deepcopy(original)
                cfg.update(hidden_size=7168,num_hidden_layers=layers,index_topk=topk,index_topk_freq=4 if shared else 1,
                           indexer_types=None,index_topk_pattern=None)
                path=e.HERE/'models'/f'dsa_L{layers}_k{topk}_{shared}'
                e.write_json(path/'config.json',cfg)
                dsa[(layers,topk,shared)]=str(path)
    return specs,refs,definitions,dsa


class Study:
    def __init__(self):
        self.hw=e.setup_systems()
        self.aic=e.Estimator(self.hw)
        self.specs,self.refs,self.defs,self.dsa=prepare()
        self.bybody={s.id:s for s in self.specs}
        self.balanced={l:next(s for s in self.specs if s.id==f'direct_L{l}_balanced_m{2944 if l==93 else 3072}_s1') for l in e.base.DEPTHS}

    def point(self,d,system,dep,cpr,phase='decode',context=262144,chunk=1,policy='routed',occupancy='expected'):
        args=dict(system=system,dep=dep,cpr=cpr,phase=phase,context=context,chunk=chunk,policy=policy,occupancy=occupancy)
        def get(model,**kw):
            r=self.aic.estimate(model,**args,**kw)
            if r['status']!='ok':
                raise RuntimeError(r['error'])
            return r
        if d['kind']=='exact':
            r=get(d['model'])
            return self.finish(d,args,r['step_ms'],r['attention_ms'],r['dispatch_ms'],r['moe_ms'],r['other_ms'],r['weight_gib'],r['kv_gib'],r['overhead_gib'])
        ctx=context+chunk if phase=='prefill' else context
        prefix='context_' if phase=='prefill' else 'generation_'
        if d['kind']=='factorial':
            spec=self.bybody[d['body']]
            body=get(spec.model_path)
            profile=e.base.ATTENTION_PROFILES[d['attention']]
            counts=e.base.allocate_counts(profile['parts'],spec.depth)
            native_w,_=e.base.body_native_weight_cache(spec,ctx)
            weight=body['weight_gib']-native_w/e.GIB
            parts={k:n for k,n in counts.items() if n}
            layers=spec.depth
        else:
            profile=d['profile']
            body=get(str(e.swa.MODELS/'common_global'))
            native_w=e.swa.LAYERS*e.swa.gqa_weight_bytes(e.swa.REFS['common_global'])
            weight=body['weight_gib']-native_w/e.GIB
            layers=61
            parts={}
            if profile['local']:
                parts[profile['local_ref']]=profile['local']
            k=profile['global_ref'] if profile['global_mechanism']=='gqa' else profile['global_mechanism']
            parts[k]=parts.get(k,0)+profile['global']
        body_time=body['step_ms']-body['attention_ms']
        attention=cache=0.
        for mechanism,n in parts.items():
            if mechanism in ('kda','mla'):
                ref=get(self.refs[layers])
                lat=ref['latency']
                if mechanism=='kda':
                    t=sum(v for k,v in lat.items() if k.startswith(prefix+'kda_'))/(layers-1)
                    w=e.base.kda_weights(); kv=e.base.kda_cache_per_layer()
                else:
                    t=sum(v for k,v in lat.items() if k.startswith(prefix+'mla_') or k.startswith(prefix+'bmm_') or k==prefix+'attention')
                    w=e.base.mla_weights(); kv=e.base.mla_cache_per_layer(ctx)
                t+=lat.get(prefix+'attn_res',0)/layers
            elif mechanism=='v4':
                ref=get(self.balanced[layers].model_path)
                t=ref['attention_ms']/layers
                w=e.base.v4_weights(layers)/layers; kv=e.base.v4_cache(layers,ctx)/layers
            elif mechanism=='dsa':
                topk=profile.get('dsa_topk',2048); shared=profile.get('dsa_shared_index',True)
                ref=get(self.dsa[(layers,topk,shared)],dsa_dims={'hidden_size':7168,'index_topk':topk})
                t=(ref['latency'].get(prefix+'attention',0)+ref['latency'].get(prefix+'add_norm_1',0))/layers
                w=dsa_weights(layers if shared else 1)
                kv=ref['kv_gib']*e.GIB/cpr/layers
            else:
                ref=get(str(e.swa.MODELS/mechanism))
                t=ref['attention_ms']/61
                w=e.swa.gqa_weight_bytes(e.swa.REFS[mechanism]); kv=e.swa.gqa_cache_bytes(e.swa.REFS[mechanism],ctx)
            attention+=n*t;weight+=n*w/e.GIB;cache+=n*kv/e.GIB*cpr
        other=body_time-body['dispatch_ms']-body['moe_ms']
        assert other>=-1e-6,(d['aid'],other)
        return self.finish(d,args,body_time+attention,attention,body['dispatch_ms'],body['moe_ms'],other,weight,cache,body['overhead_gib'])

    def finish(self,d,args,step,att,dispatch,moe,other,weight,kv,overhead):
        capacity=self.hw[args['system']]['gpu']['mem_capacity']/e.GIB
        memory=weight+kv+overhead
        cpr,dep=args['cpr'],args['dep']
        amount=args['chunk'] if args['phase']=='prefill' else 1
        return dict(aid=d['aid'],label=d['label'],family=d['family'],kind=d['kind'],**args,
                    latency_ms=step,tps_user=1000*amount/step,tps_gpu=1000*amount*cpr/step,
                    group_tps=1000*amount*cpr*dep/step,rack64_tps=1000*amount*cpr*64/step,
                    concurrency=dep*cpr,attention_ms=att,dispatch_ms=dispatch,moe_ms=moe,other_ms=other,
                    weight_gib=weight,kv_gib=kv,overhead_gib=overhead,memory_gib=memory,
                    capacity_gib=capacity,oom=memory>capacity,admit90=memory<=.9*capacity)


def expanded_models():
    """Coherent AIC-native GQA/SWA/DSA shapes at broad widths/depths/scales.

    Factorial controls pin total routed parameters and routed active FLOPs;
    each remaining geometry is valid but carries no claim of trained quality.
    """
    definitions=[]
    mimo=json.loads((e.swa.MODELS/'common_global/config.json').read_text())
    glm=json.loads((e.base.MODELS/'glm52_fp8_dsa_validation/config.json').read_text())
    for total,h,l,active in itertools.product((2.8,5.,10.),(7168,12288,16384),(64,96),(100,200)):
        inter=3072
        expert_params=3*h*inter*l
        experts=max(64,round(total*1e12/expert_params/64)*64)
        topk=max(1,round(active*1e9/expert_params))
        for attn in ('gqa','swa1','swa5','dsa2048'):
            cfg=copy.deepcopy(glm if attn=='dsa2048' else mimo)
            cfg.update(hidden_size=h,num_hidden_layers=l,n_routed_experts=experts,num_experts=experts,
                       num_experts_per_tok=topk,moe_intermediate_size=inter,vocab_size=129280,
                       n_shared_experts=1,num_attention_heads=h//128,intermediate_size=4*h)
            if attn=='dsa2048':
                cfg.update(indexer_types=None,index_topk_pattern=None,index_topk_freq=4)
            else:
                pattern=[1]*l if attn=='gqa' else [int(i%(2 if attn=='swa1' else 6)==0) for i in range(l)]
                cfg.update(hybrid_layer_pattern=pattern,moe_layer_freq=[1]*l,
                           num_key_value_heads=8,swa_num_attention_heads=h//128,swa_num_key_value_heads=8,
                           sliding_window=512,sliding_window_size=512)
            ident=f'width_T{total}_H{h}_L{l}_A{active}_{attn}'
            path=e.HERE/'models'/ident
            e.write_json(path/'config.json',cfg)
            definitions.append(dict(aid=ident,label=ident,family='Width-depth-attention extension',kind='exact',
                                    model=str(path),total_routed_t=expert_params*experts/1e12,
                                    active_routed_b=expert_params*topk/1e9,h=h,depth=l,experts=experts,topk=topk,
                                    mechanism=attn,dsa_dims={'hidden_size':h} if attn=='dsa2048' else None))
    return definitions


def main():
    s=Study()
    e.write_json(e.OUT/'definitions.json',s.defs)
    rows=[]
    # Whole old architecture breadth, both phases, fixed/memory-bounded service views.
    for system,dep,phase,cpr in itertools.product(HW,(16,64),('decode','prefill'),CPRS):
        for d in s.defs:
            rows.append(s.point(d,system,dep,cpr,phase=phase,chunk=1024 if phase=='prefill' else 1))
        print(f'Full grid {system} DEP{dep} {phase} c/r{cpr}',flush=True)
    e.write_csv(e.OUT/'unified_results.csv',rows)
    # Corrected saturation at both DEP sizes, using only memory-feasible points downstream.
    sat=[]
    for system,dep,chunk in itertools.product(('GB300','VR200-dense'),(16,64),(1,8,32,64,128,256,512,1024,2048,4096)):
        for d in s.defs:
            sat.append(s.point(d,system,dep,1,phase='prefill',chunk=chunk))
    e.write_csv(e.OUT/'saturation.csv',sat)
    # Native geometry topology + speculation, preserve original AIC mixed collectives.
    topology=[]
    native=[d for d in s.defs if d['kind']=='exact']
    for d,system,world,tp,phase in itertools.product(native,('GB300','VR200-dense'),(8,16,32,64),(1,2,4,8),('decode','prefill')):
        dep=world//tp
        for cpr in sorted({1,tp,4}):
            depths=(0,2,4,8) if phase=='decode' and cpr in (1,tp) else (0,)
            for n in depths:
                al={0:1.,2:2.51,4:3.36,8:4.0}[n]
                r=s.aic.estimate(d['model'],system,dep,cpr,phase=phase,chunk=1024 if phase=='prefill' else 1,tp=tp,nextn=n,al=al)
                topology.append(dict(aid=d['aid'],label=d['label'],**{k:v for k,v in r.items() if k not in ('latency','sources','ledger')}))
    e.write_csv(e.OUT/'topology_spec.csv',topology)
    # Controlled execution assumptions on key native and synthetic source bodies.
    sensitivities=[]
    selected=[d for d in native if d['id'] in ('k3_2p8t','k3_5t_trend','k3_10t_trend','deepseek_v4_pro_5t_expert','glm52_ratio_preserved_5t')]
    for d,system,dep,cpr,phase in itertools.product(selected,HW,(16,64),(1,4,16),('decode','prefill')):
        for policy,occupancy in (('legacy','legacy'),('legacy','expected'),('routed','expected'),('routed','correlated'),('routed','skew25')):
            r=s.aic.estimate(d['model'],system,dep,cpr,phase=phase,chunk=1024 if phase=='prefill' else 1,policy=policy,occupancy=occupancy)
            sensitivities.append(dict(aid=d['aid'],**{k:v for k,v in r.items() if k not in ('latency','sources','ledger')}))
    e.write_csv(e.OUT/'assumption_sensitivity.csv',sensitivities)
    # Coherent width and scale expansion: normalize internal DSA residual dimensions.
    expanded=expanded_models();e.write_json(e.OUT/'expanded_definitions.json',expanded)
    exrows=[]
    for d,system,dep,phase,cpr in itertools.product(expanded,('GB300','VR200-dense'),(16,64),('decode','prefill'),(1,4)):
        r=s.aic.estimate(d['model'],system,dep,cpr,phase=phase,chunk=1024 if phase=='prefill' else 1,dsa_dims=d['dsa_dims'])
        exrows.append(dict(aid=d['aid'],**{k:v for k,v in r.items() if k not in ('latency','sources','ledger')}))
    e.write_csv(e.OUT/'width_depth_results.csv',exrows)
    e.write_json(e.OUT/'metadata.json',dict(version=e.VERSION,architectures=len(s.defs),expanded=len(expanded),
                 grid_rows=len(rows),saturation_rows=len(sat),topology_rows=len(topology),
                 assumption_rows=len(sensitivities),width_rows=len(exrows),
                 status='AIC plus labeled analytical routing/occupancy extensions; not measured hardware'))


if __name__=='__main__':
    main()
