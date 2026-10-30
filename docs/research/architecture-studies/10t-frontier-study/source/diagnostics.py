"""Resource and critical-rank sensitivity for 10T native graphs."""
import copy
import itertools
import json
import yaml
import study as st
e=st.e

def main():
    hw=e.setup_systems();est=e.Estimator(hw)
    defs=json.loads((st.OUT/'definitions.json').read_text())
    chosen=[d for d in defs if d['kind']=='exact' and d['scaling']!='width_geometry']
    rows=[]
    for d,h,dep,cpr in itertools.product(chosen,('GB300','VR200-dense'),(16,64),(1,4)):
        a=est.estimate(d['model'],h,dep,cpr)
        b=est.estimate(d['model'],h,dep,cpr,occupancy='maxrank')
        assert a['status']==b['status']=='ok'
        rows.append(dict(aid=d['aid'],system=h,dep=dep,cpr=cpr,mean_rank_ms=a['step_ms'],max_rank_ms=b['step_ms'],
            latency_increase=b['step_ms']/a['step_ms']-1,admit90=a['admit90']))
    e.write_csv(st.OUT/'rank_balance.csv',rows)
    for h in ('GB300','VR200-dense'):
        for resource in ('HBM','tensor','fabric'):
            name=h+'-double-'+resource;cfg=copy.deepcopy(hw[h]);ident='10t_'+name.lower()
            if resource=='HBM':cfg['gpu']['mem_bw']*=2
            elif resource=='tensor':
                for k in ('fp4_tc_flops','fp8_tc_flops','bfloat16_tc_flops','int8_tc_flops'):cfg['gpu'][k]*=2
            else:
                for k in ('intra_node_bw','inter_node_bw','inter_rack_bw'):cfg['node'][k]*=2
            hw[name]=cfg;e.SYSTEMS[name]=ident
            (e.HERE/'systems'/f'{ident}.yaml').write_text(yaml.safe_dump(cfg))
    responses=[]
    for d,h,dep,cpr,phase in itertools.product(chosen,('GB300','VR200-dense'),(16,64),(1,4),('decode','prefill')):
        args=dict(dep=dep,cpr=cpr,phase=phase,chunk=1024 if phase=='prefill' else 1)
        a=est.estimate(d['model'],h,**args)
        for resource in ('HBM','tensor','fabric'):
            b=est.estimate(d['model'],h+'-double-'+resource,**args)
            assert a['status']==b['status']=='ok'
            moe=[x for x in a['ledger'] if x['kind']=='moe']
            responses.append(dict(aid=d['aid'],system=h,dep=dep,cpr=cpr,phase=phase,resource=resource,
                original_ms=a['step_ms'],counterfactual_ms=b['step_ms'],reduction=1-b['step_ms']/a['step_ms'],
                attention_ms=a['attention_ms'],moe_ms=a['moe_ms'],dispatch_ms=a['dispatch_ms'],
                moe_compute_ms=sum(x['compute_ms'] for x in moe),moe_memory_ms=sum(x['memory_ms'] for x in moe),
                admit90=a['admit90']))
    e.write_csv(st.OUT/'resource_counterfactuals.csv',responses)
    print('10T diagnostics complete')

if __name__=='__main__':main()
