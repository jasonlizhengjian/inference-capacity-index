"""Check actual AIC response to every dimension, plus native baseline bridge."""
import json
from pathlib import Path
import study as s
e=s.e
def main():
    est=e.Estimator(s.body.HW);rows=[]
    for d in s.body.DEFS:
        if d['target_t']!=5:continue
        r=est.estimate(d['model'],'GB300',dep=64,cpr=1,context=262144,quant='nvfp4',gemm='bfloat16',dsa_dims=d['dsa_dims'])
        assert r['status']=='ok',r
        expected=d['kv_bytes_per_token']*262145/(2**30)
        assert abs(r['kv_gib']/expected-1)<.0001,(d['id'],r['kv_gib'],expected)
        rows.append(dict(id=d['id'],profile=d['profile'],family=d['family'],step_ms=r['step_ms'],attention_ms=r['attention_ms'],moe_ms=r['moe_ms'],weight_gib=r['weight_gib'],kv_gib=r['kv_gib'],expected_kv_gib=expected))
    baseline=next(r for r in rows if r['profile']=='baseline')
    for r in rows:
        if r['family']!='one_axis' or r['profile']=='baseline':continue
        assert abs(r['attention_ms']-baseline['attention_ms'])>1e-10 or abs(r['weight_gib']-baseline['weight_gib'])>1e-10,('Ignored dimension',r)
        assert abs(r['moe_ms']-baseline['moe_ms'])<1e-8,('MoE body changed in attention-only control',r)
        if not r['profile'].startswith(('kv_lora_rank_','index_head_dim_')):assert abs(r['kv_gib']-baseline['kv_gib'])<1e-8
    for d in s.body.DEFS:
        if d['family']=='joint' and d['profile']=='scaled_inner':
            a=next(a for a in s.body.DEFS if a['family']=='joint' and a['profile']=='fixed_inner' and (a['target_t'],a['layers'],a['hidden'])==(d['target_t'],d['layers'],d['hidden']))
            for k in ('layers','hidden','expert_width','experts','active_experts','routed_total_t','routed_active_b'):assert a[k]==d[k]
    e.write_csv(s.OUT/'dimension_checks.csv',rows)
    # Paired corrected / original-formula query, same model, backend, precision and batch.
    bridge=[]
    for native in (False,True):
        d=next(d for d in s.body.DEFS if d['target_t']==5 and d['profile']=='baseline')
        model=str(s.HERE.parent/'inferencex-agentx-grounding/models/glm') if native else d['model']
        args=dict(model=model,system='GB300',dep=8 if native else 64,cpr=10 if native else 4,context=163388 if native else 262144,nextn=3 if native else 0,al=1,
                  quant=None if native else 'nvfp4',gemm=None if native else 'bfloat16',dsa_dims=None if native else d['dsa_dims'])
        corrected=est.estimate(**args)
        assert corrected['status']=='ok',corrected
        original=s.GenerationDSAModule._query_generation_dsa_module_table
        s.GenerationDSAModule._query_generation_dsa_module_table=s.original_table
        version=e.VERSION;e.VERSION+='-original-decode-indexer'
        try:legacy=est.estimate(**args)
        finally:s.GenerationDSAModule._query_generation_dsa_module_table=original;e.VERSION=version
        assert legacy['status']=='ok',legacy
        bridge.append(dict(native=native,original_formula_ms=legacy['step_ms'],reuse_corrected_ms=corrected['step_ms'],observed_device_ms=30.645503997802734 if native else None))
    e.write_csv(s.OUT/'baseline_bridge.csv',bridge)
    print('Validated',len(rows),'dimension configs; baseline bridge',bridge)
if __name__=='__main__':main()
