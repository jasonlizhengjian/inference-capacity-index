"""Expose dependence on AIC's ideal shared/routed overlap assumption."""
import json
from pathlib import Path
import study as s
from aiconfigurator_core.sdk.operations.overlap import OverlapOp
from aiconfigurator_core.sdk.performance_result import PerformanceResult
e=s.e
def main():
    base=e.Estimator(s.HW)
    ds=[d for d in s.DEFS if d['family']=='shared' or d['target_t'] in (5,10) and (d['family']=='capacity_fixed_ratio' or d['family']=='geometry' and (d['layers'],d['hidden']) in ((48,4096),(64,6144),(128,12288)))]
    cases=[]
    for d in ds:
        for hw in ('GB300','VR200-dense'):
            for c in (1,4,16):
                for nn in (0,4):
                    args=dict(model=d['model'],system=hw,dep=64,cpr=c,context=262144,nextn=nn,al=3.36 if nn else 1,quant='nvfp4',gemm='bfloat16',dsa_dims=d['dsa_dims'])
                    a=base.estimate(**args)
                    if a['status']=='ok' and a['admit90']:cases.append((d,args,a))
    original=OverlapOp.query
    def serial(op,database,**kwargs):
        if 'moe_overlap' not in op._name:return original(op,database,**kwargs)
        total=PerformanceResult(0,energy=0,source='sol')
        for inner in op._group_a+op._group_b:total+=inner.query(database,**kwargs)
        return total
    e.VERSION+='-serialize-shared';OverlapOp.query=serial
    rows=[]
    try:
        for d,args,a in cases:
            b=base.estimate(**args);assert b['status']=='ok',b
            assert b['step_ms']>=a['step_ms']-1e-8
            if d['shared_experts']==0:assert abs(b['step_ms']-a['step_ms'])<1e-8
            rows.append(dict(id=d['id'],family=d['family'],target_t=d['target_t'],shared_experts=d['shared_experts'],system=args['system'],cpr=args['cpr'],nextn=args['nextn'],
                             ideal_overlap_ms=a['step_ms'],serialized_ms=b['step_ms'],throughput_ratio=a['step_ms']/b['step_ms'],memory_gib=a['memory_gib']))
    finally:OverlapOp.query=original
    e.write_csv(s.OUT/'overlap_sensitivity.csv',rows)
    print('Shared-branch sensitivity rows',len(rows))
if __name__=='__main__':main()
