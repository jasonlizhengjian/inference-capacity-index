"""Fixed-64-GPU memory and service boundaries including attention TP and speculation."""
import itertools
import json
import math
import study as st
e=st.e

def main():
    est=e.Estimator(e.setup_systems());als=e.base.load_dspark_al_curve()
    defs=[d for d in json.loads((st.OUT/'definitions.json').read_text()) if d['kind']=='exact' and d['scaling']!='width_geometry']
    rows=[]
    for d,hw,tp,n in itertools.product(defs,('GB300','VR200-dense'),(1,2,4,8),(0,2,4,8)):
        dep=64//tp
        def get(cpr):
            r=est.estimate(d['model'],hw,dep,cpr,tp=tp,nextn=n,al=als[n])
            assert r['status']=='ok',r
            return r
        a,b=get(1),get(2);slope=b['memory_gib']-a['memory_gib']
        maximum=max(0,math.floor((.9*a['capacity_gib']-a['memory_gib'])/slope)+1)
        assert maximum<65536
        while maximum and not get(maximum)['admit90']:maximum-=1
        while get(maximum+1)['admit90']:maximum+=1
        full=get(maximum) if maximum else None
        for eff,target in itertools.product((1.,.5,.7),(50,100)):
            lo,hi=0,maximum
            while lo<hi:
                mid=(lo+hi+1)//2
                if 1000/get(mid)['latency_ms']*eff>=target:lo=mid
                else:hi=mid-1
            at=get(lo) if lo else None
            if lo:assert at['admit90'] and 1000/at['latency_ms']*eff>=target
            if lo<maximum:assert 1000/get(lo+1)['latency_ms']*eff<target
            rows.append(dict(aid=d['aid'],label=d['label'],system=hw,world=64,tp=tp,dep=dep,nextn=n,al=als[n],
                efficiency=eff,target=target,max_cpr_memory90=maximum,max_requests_memory90=maximum*dep,
                tps_user_at_memory_max=1000/full['latency_ms'] if full else None,
                max_cpr_target=lo,max_requests_target=lo*dep,tps_user_at_target=1000/at['latency_ms']*eff if at else None))
        print(d['aid'],hw,'TP',tp,'spec',n,flush=True)
    e.write_csv(st.OUT/'native_capacity_slo.csv',rows)

if __name__=='__main__':main()
