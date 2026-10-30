"""Fixed-DSA body study: memory-bounded decode and prefill with explicit assumptions."""
import itertools
import json
import math
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'output';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(HERE.parent/'frontier-hardware-study'))
import engine as e
e.OUT=OUT;e.VERSION+='-glm-body-v3-shared'
from aiconfigurator_core.sdk.operations.base import Operation
_cli=e.cli_estimate
def cli(*a,**kw):
    kw['backend_name']='sglang' if kw['tp_size']>1 and kw['attention_dp_size']>1 else 'trtllm'
    shared=json.loads((Path(a[0])/'config.json').read_text())['n_shared_experts']
    original=Operation.__init__
    def init(op,name,scale_factor,*args,**kwargs):
        return original(op,name,scale_factor*shared if '_shared_' in name else scale_factor,*args,**kwargs)
    Operation.__init__=init
    try:return _cli(*a,**kw)
    finally:Operation.__init__=original
e.cli_estimate=cli
DEFS=json.loads((HERE/'definitions.json').read_text())
HW=json.loads((HERE.parent/'frontier-hardware-study/output/hardware.json').read_text())
def compact(r):return {k:v for k,v in r.items() if k not in ('latency','sources','ledger')}

def main():
    est=e.Estimator(HW);rows=[];limits=[];slo=[]
    def point(d,h,w,t,b,phase,nn):
        al=3.36 if nn else 1
        r=est.estimate(d['model'],h,dep=w//t,cpr=b,tp=t,phase=phase,chunk=1024 if phase=='prefill' else 1,context=262144,
                      nextn=nn,al=al,quant='nvfp4',gemm='bfloat16',dsa_dims=d['dsa_dims'])
        if r['status']!='ok':raise RuntimeError(r)
        return dict(compact(r),id=d['id'],family=d['family'],target_t=d['target_t'],requests_64gpu=64*b/t,
                    throughput_gpu=b/t*(1024 if phase=='prefill' else al)*1000/r['step_ms'],tps_user=1000/r['latency_ms'] if phase=='decode' else None,
                    moe_compute_ledger_ms=sum(x['compute_ms'] for x in r['ledger'] if x['kind']=='moe'),moe_memory_ledger_ms=sum(x['memory_ms'] for x in r['ledger'] if x['kind']=='moe'))
    for i,d in enumerate(DEFS):
        for h,w,t,(phase,nn) in itertools.product(('GB300','VR200-dense'),(16,32,64),(1,2,4,8),(('decode',0),('decode',4),('prefill',0))):
            a=point(d,h,w,t,1,phase,nn);b=point(d,h,w,t,2,phase,nn)
            cap=.9*HW[h]['gpu']['mem_capacity']/e.GIB;slope=b['memory_gib']-a['memory_gib']
            maximum=max(0,math.floor((cap-a['memory_gib'])/slope)+1) if slope>0 else 0
            assert maximum<16384
            while maximum and not point(d,h,w,t,maximum,phase,nn)['admit90']:maximum-=1
            while point(d,h,w,t,maximum+1,phase,nn)['admit90']:maximum+=1
            limits.append(dict(id=d['id'],system=h,world=w,tp=t,phase=phase,nextn=nn,max_cpr=maximum,max_requests64=64*maximum/t,min_memory_gib=a['memory_gib']))
            if not maximum:continue
            batches=sorted({maximum,*[2**j for j in range(14) if 2**j<=maximum]})
            for c in batches:rows.append(dict(point(d,h,w,t,c,phase,nn),max_cpr=maximum))
            if phase=='decode':
                for efficiency,target in itertools.product((1,.7),(50,100)):
                    lo,hi=0,maximum
                    while lo<hi:
                        mid=(lo+hi+1)//2
                        if point(d,h,w,t,mid,phase,nn)['tps_user']*efficiency>=target:lo=mid
                        else:hi=mid-1
                    if lo:
                        rr=dict(point(d,h,w,t,lo,phase,nn),max_cpr=maximum,efficiency=efficiency,target=target)
                        assert rr['admit90'] and rr['tps_user']*efficiency>=target
                        if lo<maximum:assert point(d,h,w,t,lo+1,phase,nn)['tps_user']*efficiency<target
                        slo.append(rr)
        if i%5==0:
            print('Models',i+1,'/',len(DEFS),'grid rows',len(rows),'SLO rows',len(slo),flush=True)
            e.write_csv(OUT/'grid.csv',rows);e.write_csv(OUT/'slo.csv',slo);e.write_csv(OUT/'memory_limits.csv',limits)
    e.write_csv(OUT/'grid.csv',rows);e.write_csv(OUT/'slo.csv',slo);e.write_csv(OUT/'memory_limits.csv',limits)
    print('Study complete',flush=True)
if __name__=='__main__':main()
