"""Reuse body sweep, with dimension-aware DSA and audited index-reuse SOL."""
import json
import importlib.util
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'output';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(HERE.parent/'glm-body-design-study'))
spec=importlib.util.spec_from_file_location('glm_body_base',HERE.parent/'glm-body-design-study/study.py')
body=importlib.util.module_from_spec(spec);spec.loader.exec_module(body)
from aiconfigurator_core.sdk.operations.dsa import GenerationDSAModule,DSA_MODEL_DIMS
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.performance_result import PerformanceResult
e=body.e
body.OUT=OUT;e.OUT=OUT;e.VERSION+='-attention-dims-v1-reuse'
body.DEFS=json.loads((HERE/'definitions.json').read_text())
original_cli=body._cli
def cli(*a,**kw):
    # SGLang graph carries the checkpoint's exact index-sharing fraction in both phases.
    kw['backend_name']='sglang'
    return original_cli(*a,**kw)
body._cli=cli
original_table=GenerationDSAModule._query_generation_dsa_module_table
def generation_table(cls,database,b,s,num_heads,kv_cache_dtype,gemm_quant_mode=common.GEMMQuantMode.bfloat16,database_mode=None,**kwargs):
    mode=database_mode or database._default_database_mode
    if not kwargs.get('skip_indexer') or mode!=common.DatabaseMode.SOL:
        return original_table(database,b,s,num_heads,kv_cache_dtype,gemm_quant_mode,database_mode,**kwargs)
    dims=DSA_MODEL_DIMS[kwargs.get('architecture') or 'GlmMoeDsaForCausalLM']
    h,q,kv,nope,rope,v=[dims[k] for k in ('hidden_size','q_lora_rank','kv_lora_rank','qk_nope_head_dim','qk_rope_head_dim','v_head_dim')]
    topk=kwargs.get('index_topk') or dims['index_topk'];effective=min(s,topk)
    # Same absorbed-MLA algebra as AIC's generation SOL, minus index-only work.
    gemm_ops=2*b*h*(q+kv+rope)+2*b*q*num_heads*(nope+rope)+2*b*num_heads*v*h+2*num_heads*b*nope*kv+2*num_heads*b*kv*v
    attn_ops=2*b*num_heads*(2*kv+rope)*effective
    weights=(h*(q+kv+rope)+q*num_heads*(nope+rope)+num_heads*v*h)*gemm_quant_mode.value.memory
    cache=b*effective*(kv+rope)*kv_cache_dtype.value.memory
    math_ms=1000*(gemm_ops/common.get_quant_tc_flops(database.system_spec,gemm_quant_mode)+attn_ops/common.get_quant_tc_flops(database.system_spec,common.FMHAQuantMode.bfloat16))
    mem_ms=1000*(weights+cache)/database.system_spec['gpu']['mem_bw']
    return PerformanceResult(max(math_ms,mem_ms),energy=0,source='analytical_extension')
GenerationDSAModule._query_generation_dsa_module_table=classmethod(generation_table)
if __name__=='__main__':body.main()
