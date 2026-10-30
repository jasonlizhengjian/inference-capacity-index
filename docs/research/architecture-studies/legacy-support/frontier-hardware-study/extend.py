"""Boundary verification, workload and precision sensitivities, independent checks."""
import itertools
import json
import math
import random
import sys
import run
import engine as e


def checks(s):
    tests=[]
    for E,k,ep in ((8,2,4),(16,4,8),(896,16,64)):
        assert abs(e.distinct_experts(E,k,1)-k)<1e-9
        assert e.distinct_experts(E,k,100000)<=E
        rng=random.Random(27); n=10000
        observed=sum(len({x%ep for x in rng.sample(range(E),k)}-{0}) for _ in range(n))/n
        predicted=e.destinations(E,k,ep)
        assert abs(observed-predicted)<.08,(observed,predicted)
        tests.append(dict(test='routing Monte Carlo',experts=E,topk=k,ep=ep,predicted=predicted,observed=observed))
    d=s.defs[0]
    for phase in ('decode','prefill'):
        rows=[s.aic.estimate(d['model'],h,64,4,phase=phase,chunk=1024 if phase=='prefill' else 1) for h in e.SYSTEMS]
        assert all(r['status']=='ok' for r in rows)
        assert rows[0]['step_ms']>=rows[1]['step_ms']>=rows[2]['step_ms']>=rows[3]['step_ms']
        tests.append(dict(test='resource monotonicity',phase=phase,latencies=[r['step_ms'] for r in rows]))
    assert e.MoE.query is e.ORIGINAL_MOE and e.MoEDispatch.query is e.ORIGINAL_DISPATCH
    tests.append(dict(test='scoped monkeypatch restored',passed=True))
    e.write_json(e.OUT/'validation.json',tests)


def main():
    s=run.Study();checks(s)
    bounds=[]
    for hw,dep in itertools.product(('GB300','VR200-dense'),(16,64)):
        for i,d in enumerate(s.defs):
            a=s.point(d,hw,dep,1);b=s.point(d,hw,dep,2)
            slope=b['memory_gib']-a['memory_gib']
            cap=.9*a['capacity_gib']
            maximum=max(0,math.floor((cap-a['memory_gib'])/slope)+1) if slope>0 else 0
            maximum=min(maximum,65536)
            # An affine prediction is merely a starting bracket; query both sides.
            while maximum and not s.point(d,hw,dep,maximum)['admit90']:
                maximum-=1
            while maximum<65536 and s.point(d,hw,dep,maximum+1)['admit90']:
                maximum+=1
            atmax=s.point(d,hw,dep,maximum) if maximum else None
            for efficiency,target in itertools.product((1.,.5,.7),(50,100)):
                low,high=0,maximum
                while low<high:
                    middle=(low+high+1)//2
                    p=s.point(d,hw,dep,middle)
                    if p['tps_user']*efficiency>=target: low=middle
                    else: high=middle-1
                current=s.point(d,hw,dep,low) if low else None
                following=s.point(d,hw,dep,low+1) if low<maximum else None
                assert not current or current['admit90'] and current['tps_user']*efficiency>=target
                assert not following or following['tps_user']*efficiency<target
                bounds.append(dict(aid=d['aid'],label=d['label'],family=d['family'],system=hw,dep=dep,
                    efficiency=efficiency,target=target,max_cpr_memory90=maximum,
                    max_cpr_target=low,concurrency_group=low*dep,concurrency_64gpu=low*64,
                    tps_at_memory_max=atmax['tps_user'] if atmax else None,
                    tps_at_target=current['tps_user']*efficiency if current else None,
                    memory_gib_at_max=atmax['memory_gib'] if atmax else None,
                    capped=maximum==65536))
            if i%100==0: print('Boundaries',hw,dep,i,flush=True)
        e.write_csv(e.OUT/'capacity_slo_boundaries.csv',bounds)
    natives=[d for d in s.defs if d['kind']=='exact']
    selected=[d for d in natives if d['id'] in ('k3_2p8t','k3_5t_trend','k3_10t_trend','deepseek_v4_pro_5t_expert','glm52_ratio_preserved_5t')]
    extra=[]
    for d,hw,dep,cpr,context,chunk in itertools.product(selected,('GB300','HBM-only','HBM+fabric','VR200-dense'),(16,64),(1,4),(8192,32768,262144,1048576),(1,128,1024,4096)):
        r=s.aic.estimate(d['model'],hw,dep,cpr,context=context,phase='decode' if chunk==1 else 'prefill',chunk=chunk)
        extra.append(dict(aid=d['aid'],**{k:v for k,v in r.items() if k not in ('latency','sources','ledger')}))
    e.write_csv(e.OUT/'workload_sensitivity.csv',extra)
    quant=[]
    for d,hw,dep,cpr,phase,q,kv in itertools.product(selected,('GB300','VR200-dense'),(16,64),(1,4),('decode','prefill'),('w4a8_mxfp4_mxfp8_trtllm','nvfp4','fp8'),('fp8','bfloat16')):
        r=s.aic.estimate(d['model'],hw,dep,cpr,phase=phase,chunk=1024 if phase=='prefill' else 1,quant=q,kv=kv)
        quant.append(dict(aid=d['aid'],**{k:v for k,v in r.items() if k not in ('latency','sources','ledger')}))
    e.write_csv(e.OUT/'precision_sensitivity.csv',quant)
    print('Supplemental complete',flush=True)


if __name__=='__main__':main()
