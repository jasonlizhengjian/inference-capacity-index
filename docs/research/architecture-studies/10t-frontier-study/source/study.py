"""10T extension of the audited frontier AIC study. Existing artifacts untouched."""
import copy
import dataclasses
import itertools
import json
import math
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'frontier-hardware-study'))
import engine as e
import run as previous
OUT=HERE/'output';OUT.mkdir(parents=True,exist_ok=True)
e.OUT=OUT
OLD_SWA=e.swa.MODELS
HARDWARE=('GB300','HBM+fabric','VR200-dense')

def scaled(source,destination,active_multiplier):
    cfg=json.loads((Path(source)/'config.json').read_text())
    body=cfg.get('text_config',cfg)
    for k in ('n_routed_experts','num_experts'):
        if body.get(k):body[k]*=2
    for k in ('num_experts_per_tok','num_experts_per_token'):
        if body.get(k):body[k]*=active_multiplier
    e.write_json(destination/'config.json',cfg)
    return str(destination)

class Study(previous.Study):
    def __init__(self,scaling):
        super().__init__();self.scaling=scaling
        mult=1 if scaling=='fixed_active' else 2
        self.specs=[dataclasses.replace(s,experts=s.experts*2,topk=s.topk*mult,
              routed_total=s.routed_total*2,routed_active=s.routed_active*mult,
              model_path=scaled(s.model_path,HERE/'models'/scaling/s.id,mult)) for s in self.specs]
        self.bybody={s.id:s for s in self.specs}
        self.balanced={l:self.bybody[s.id] for l,s in self.balanced.items()}
        self.swa_models=HERE/'models'/scaling/'swa'
        for p in OLD_SWA.glob('*/config.json'):scaled(p.parent,self.swa_models/p.parent.name,mult)
        self.defs=[dict(d,aid=scaling+'__'+d['aid'],parent_aid=d['aid'],scaling=scaling,
                       label=scaling+' · '+d['label'].replace('5T','10T'),family=d['family'].replace('5T','10T'))
                   for d in self.defs if d['kind']!='exact']
        for d in self.defs:
            if d['kind']=='factorial':
                s=self.bybody[d['body']]
                d.update(experts=s.experts,topk=s.topk,routed_total_t=s.routed_total/1e12,routed_active_b=s.routed_active/1e9,model=s.model_path)
            else:
                cfg=json.loads((self.swa_models/'common_global/config.json').read_text())
                h=cfg['hidden_size'];m=cfg['moe_intermediate_size'];l=cfg['num_hidden_layers']
                E=cfg.get('n_routed_experts',cfg.get('num_experts'));k=cfg['num_experts_per_tok']
                d.update(experts=E,topk=k,depth=l,routed_total_t=3*h*m*l*E/1e12,routed_active_b=3*h*m*l*k/1e9)

    def point(self,d,*args,**kwargs):
        old=e.swa.MODELS;e.swa.MODELS=self.swa_models
        try:r=super().point(d,*args,**kwargs)
        finally:e.swa.MODELS=old
        return dict(r,scaling=self.scaling,parent_aid=d['parent_aid'])

def native_models(original):
    result=[]
    for d in original.defs:
        if d['kind']=='exact' and d['id'] in ('k3_10t_expert','k3_10t_trend'):
            result.append(dict(d,scaling='native_10t'))
        if d['kind']=='exact' and d['id'] in ('deepseek_v4_pro_5t_expert','glm52_ratio_preserved_5t','deepseek_moe_normalized','kimi_moe_normalized'):
            for scaling,mult in (('fixed_active',1),('ratio_preserved',2)):
                result.append(dict(d,aid=scaling+'__'+d['aid'],label=scaling+' · '+d['label'].replace('5T','10T'),
                     model=scaled(d['model'],HERE/'models'/scaling/d['id'],mult),scaling=scaling))
    # Reuse coherent 10T width geometries, and extend active budget to ~400B.
    definitions=json.loads((HERE.parent/'frontier-hardware-study/output/expanded_definitions.json').read_text())
    for d in definitions:
        if not d['aid'].startswith('width_T10.0_'):continue
        result.append(dict(d,scaling='width_geometry'))
        if '_A200_' in d['aid']:
            cfg=json.loads((Path(d['model'])/'config.json').read_text())
            cfg['num_experts_per_tok']*=2
            ident=d['aid'].replace('_A200_','_A400_');path=HERE/'models'/ident
            e.write_json(path/'config.json',cfg)
            result.append(dict(d,aid=ident,label=ident,model=str(path),scaling='width_geometry',
                               topk=d['topk']*2,active_routed_b=d['active_routed_b']*2))
    return result

def compact(r):return {k:v for k,v in r.items() if k not in ('latency','sources','ledger')}

def main():
    original=previous.Study();studies=[Study(x) for x in ('fixed_active','ratio_preserved')]
    # Share the estimator to avoid duplicate reference evaluations and cache writers.
    for s in studies:s.aic=original.aic
    native=native_models(original)
    e.write_json(OUT/'definitions.json',[d for s in studies for d in s.defs]+native)
    rows=[]
    for s,hw,dep,phase,cpr in itertools.product(studies,HARDWARE,(16,32,64),('decode','prefill'),(1,2,4,8,16,32)):
        for d in s.defs:rows.append(s.point(d,hw,dep,cpr,phase=phase,chunk=1024 if phase=='prefill' else 1))
        print('Grid',s.scaling,hw,dep,phase,cpr,flush=True)
    e.write_csv(OUT/'architecture_grid.csv',rows)
    nrows=[]
    for d,hw,dep,phase,cpr in itertools.product(native,('GB300','VR200-dense'),(16,32,64),('decode','prefill'),(1,4,16)):
        r=original.aic.estimate(d['model'],hw,dep,cpr,phase=phase,chunk=1024 if phase=='prefill' else 1,dsa_dims=d.get('dsa_dims'))
        nrows.append(dict(aid=d['aid'],label=d['label'],scaling=d['scaling'],**compact(r)))
    e.write_csv(OUT/'native_geometry_grid.csv',nrows)
    saturation=[]
    for s,hw,dep,chunk in itertools.product(studies,('GB300','VR200-dense'),(16,64),(1,8,32,128,512,1024,2048,4096)):
        for d in s.defs:saturation.append(s.point(d,hw,dep,1,phase='prefill',chunk=chunk))
    e.write_csv(OUT/'prefill_saturation.csv',saturation)
    boundaries=[]
    for s,hw,dep in itertools.product(studies,('GB300','VR200-dense'),(16,32,64)):
        for i,d in enumerate(s.defs):
            a=s.point(d,hw,dep,1);b=s.point(d,hw,dep,2)
            slope=b['memory_gib']-a['memory_gib'];cap=.9*a['capacity_gib']
            maximum=max(0,math.floor((cap-a['memory_gib'])/slope)+1) if slope>0 else 0
            maximum=min(maximum,65536)
            while maximum and not s.point(d,hw,dep,maximum)['admit90']:maximum-=1
            while maximum<65536 and s.point(d,hw,dep,maximum+1)['admit90']:maximum+=1
            full=s.point(d,hw,dep,maximum) if maximum else None
            for efficiency,target in itertools.product((1.,.5,.7),(50,100)):
                lo,hi=0,maximum
                while lo<hi:
                    mid=(lo+hi+1)//2
                    if s.point(d,hw,dep,mid)['tps_user']*efficiency>=target:lo=mid
                    else:hi=mid-1
                if lo:assert s.point(d,hw,dep,lo)['admit90'] and s.point(d,hw,dep,lo)['tps_user']*efficiency>=target
                if lo<maximum:assert s.point(d,hw,dep,lo+1)['tps_user']*efficiency<target
                boundaries.append(dict(aid=d['aid'],label=d['label'],scaling=s.scaling,parent_aid=d['parent_aid'],
                    system=hw,dep=dep,efficiency=efficiency,target=target,max_cpr_memory90=maximum,
                    max_cpr_target=lo,concurrency64=64*lo,tps_at_memory_max=full['tps_user'] if full else None,
                    memory_at_max=full['memory_gib'] if full else None,capped=maximum==65536))
            if i%100==0:print('Boundaries',s.scaling,hw,dep,i,flush=True)
        e.write_csv(OUT/'capacity_slo.csv',boundaries)
    # Topology/spec screens use actual native AIC graphs, never arbitrary hybrid drafts.
    chosen=[d for d in native if d['scaling']!='width_geometry' or (d.get('h')==12288 and d.get('depth')==64)]
    for s in studies:
        for ident in ('direct_L61_balanced_m3072_s1','latent_L93_balanced_m3072_s1'):
            spec=s.bybody[ident];chosen.append(dict(aid=s.scaling+'__body__'+ident,label=s.scaling+' '+ident,model=spec.model_path))
    topology=[]
    als=e.base.load_dspark_al_curve()
    for d,hw,world,tp,phase,cpr in itertools.product(chosen,('GB300','VR200-dense'),(16,32,64),(1,2,4,8),('decode','prefill'),(1,4)):
        for n in ((0,2,4,8) if phase=='decode' else (0,)):
            r=original.aic.estimate(d['model'],hw,world//tp,cpr,tp=tp,phase=phase,chunk=1024 if phase=='prefill' else 1,
                      nextn=n,al=als[n],dsa_dims=d.get('dsa_dims'))
            topology.append(dict(aid=d['aid'],label=d['label'],**compact(r)))
    e.write_csv(OUT/'topology_spec.csv',topology)
    sensitivity=[]
    anchors=chosen[:10]
    for d,hw,dep,cpr,phase in itertools.product(anchors,('GB300','HBM-only','HBM+fabric','VR200-dense'),(16,64),(1,4),('decode','prefill')):
        for policy,occupancy in (('legacy','legacy'),('routed','expected'),('routed','correlated'),('routed','skew25')):
            r=original.aic.estimate(d['model'],hw,dep,cpr,phase=phase,chunk=1024 if phase=='prefill' else 1,policy=policy,occupancy=occupancy)
            sensitivity.append(dict(aid=d['aid'],**compact(r)))
    e.write_csv(OUT/'routing_sensitivity.csv',sensitivity)
    workloads=[]
    for d,hw,dep,cpr,context,phase in itertools.product(anchors,('GB300','VR200-dense'),(16,64),(1,4),(32768,1048576),('decode','prefill')):
        r=original.aic.estimate(d['model'],hw,dep,cpr,context=context,phase=phase,chunk=1024 if phase=='prefill' else 1)
        workloads.append(dict(aid=d['aid'],**compact(r)))
    e.write_csv(OUT/'context_sensitivity.csv',workloads)
    precision=[]
    for d,hw,tp,q,kv,phase in itertools.product(anchors,('GB300','VR200-dense'),(1,4),('w4a8_mxfp4_mxfp8_trtllm','nvfp4','fp8'),('fp8','bfloat16'),('decode','prefill')):
        r=original.aic.estimate(d['model'],hw,64//tp,1,tp=tp,quant=q,kv=kv,phase=phase,chunk=1024 if phase=='prefill' else 1)
        precision.append(dict(aid=d['aid'],**compact(r)))
    e.write_csv(OUT/'precision_sensitivity.csv',precision)
    e.write_json(OUT/'metadata.json',dict(controlled_architectures=sum(len(s.defs) for s in studies),native_geometries=len(native),
        architecture_rows=len(rows),boundary_rows=len(boundaries),topology_rows=len(topology),
        status='AIC plus analytical extensions; no measured 10T hardware validation'))
    print('10T sweeps complete',flush=True)

if __name__=='__main__':main()
