"""Operator roofline and matched historical calibration diagnostics."""
import copy
import csv
import itertools
import run
import engine as e
import yaml

def main():
    s=run.Study();rows=[]
    natives=[d for d in s.defs if d['kind']=='exact' and d['id'] in ('k3_2p8t','k3_5t_trend','deepseek_v4_pro_5t_expert','glm52_ratio_preserved_5t')]
    for d,hw,dep,cpr,phase in itertools.product(natives,('GB300','VR200-dense'),(16,64),(1,4,16),('decode','prefill')):
        r=s.aic.estimate(d['model'],hw,dep,cpr,phase=phase,chunk=1024 if phase=='prefill' else 1)
        ledger=r.get('ledger',[])
        moe=[x for x in ledger if x['kind']=='moe']
        rows.append(dict(aid=d['aid'],system=hw,dep=dep,cpr=cpr,phase=phase,
            step_ms=r['step_ms'],attention_ms=r['attention_ms'],moe_ms=r['moe_ms'],dispatch_ms=r['dispatch_ms'],
            moe_compute_ms=sum(x['compute_ms'] for x in moe),moe_memory_ms=sum(x['memory_ms'] for x in moe),
            compute_bound_moe=sum(x['time_ms'] for x in moe if x['compute_ms']>=x['memory_ms']),
            memory_bound_moe=sum(x['time_ms'] for x in moe if x['memory_ms']>x['compute_ms']),
            non_analytical_ms=r['non_analytical_ms'],oom=r['oom']))
    e.write_csv(e.OUT/'resource_diagnostics.csv',rows)
    calibration=[]
    path=e.ROOT/'experiments/kimi-k3-gb300-decode/output/no_spec_external_calibration.csv'
    for source in csv.DictReader(path.open()):
        for policy,occupancy in (('legacy','legacy'),('routed','expected')):
            r=s.aic.estimate(natives[0]['model'],'GB300',16,int(source['concurrency_per_rank']),
                context=8192,policy=policy,occupancy=occupancy,gemm='bfloat16',kv='bfloat16')
            actual=float(source['measured_median_tpot_ms'])
            calibration.append(dict(**source,adapter_policy=policy,adapter_occupancy=occupancy,
                updated_step_ms=r['step_ms'],actual_throughput_fraction=r['step_ms']/actual,
                caveat='Historical published serving median; static start-context approximation, no holdout.'))
    e.write_csv(e.OUT/'calibration_recheck.csv',calibration)
    screen=[]
    for d,hw,dep,cpr,phase in itertools.product(s.defs,('GB300','VR200-dense'),(8,32),(1,4),('decode','prefill')):
        screen.append(s.point(d,hw,dep,cpr,phase=phase,chunk=1024 if phase=='prefill' else 1))
    e.write_csv(e.OUT/'parallelism_screen.csv',screen)
    # Finite counterfactuals distinguish attention compute from attention bytes.
    for hw in ('GB300','VR200-dense'):
        for resource in ('HBM','tensor','fabric'):
            name=f'{hw}-double-{resource}';ident=name.lower()
            cfg=copy.deepcopy(s.hw[hw])
            if resource=='HBM':cfg['gpu']['mem_bw']*=2
            elif resource=='tensor':
                for k in ('fp4_tc_flops','fp8_tc_flops','bfloat16_tc_flops','int8_tc_flops'):cfg['gpu'][k]*=2
            else:
                for k in ('inter_node_bw','intra_node_bw','inter_rack_bw'):cfg['node'][k]*=2
            s.hw[name]=cfg;e.SYSTEMS[name]=ident
            (e.HERE/'systems'/f'{ident}.yaml').write_text(yaml.safe_dump(cfg))
    responses=[]
    chosen=natives+[d for d in s.defs if d['aid'] in ('swa__full_gqa','swa__common_r5_w512','swa__swa5_kda','swa__swa5_dsa')]
    for d,hw,dep,cpr,phase in itertools.product(chosen,('GB300','VR200-dense'),(16,64),(1,4,16),('decode','prefill')):
        original=s.point(d,hw,dep,cpr,phase=phase,chunk=1024 if phase=='prefill' else 1)
        for resource in ('HBM','tensor','fabric'):
            revised=s.point(d,f'{hw}-double-{resource}',dep,cpr,phase=phase,chunk=1024 if phase=='prefill' else 1)
            responses.append(dict(aid=d['aid'],system=hw,dep=dep,cpr=cpr,phase=phase,resource_doubled=resource,
                original_ms=original['latency_ms'],revised_ms=revised['latency_ms'],
                latency_reduction=1-revised['latency_ms']/original['latency_ms'],
                attention_reduction=1-revised['attention_ms']/original['attention_ms'],
                admit90=original['admit90']))
    e.write_csv(e.OUT/'resource_counterfactuals.csv',responses)

if __name__=='__main__':main()
