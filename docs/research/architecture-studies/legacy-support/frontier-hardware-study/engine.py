"""AIC-based research adapter. No changes to the installed AIC checkout.

Extensions are scoped to each estimate and independently identified in outputs.
Routed communication is a traffic lower bound, not a measured DeepEP kernel.
"""
from __future__ import annotations
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'output'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/frontier-aic-mpl')
sys.path.insert(0, str(ROOT / 'experiments/5t-architecture-design-study'))
sys.path.insert(0, str(ROOT / 'experiments/5t-swa-architecture-study'))
import run_study as base
import run_swa_study as swa
import yaml
from aiconfigurator.cli.api import cli_estimate
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.moe import MoE, MoEDispatch
from aiconfigurator_core.sdk.operations.dsa import DSA_MODEL_DIMS
from aiconfigurator_core.sdk.performance_result import PerformanceResult

VERSION = 'frontier-adapter-v2'
GIB = 2**30
SYSTEMS = {
    'GB300': 'gb300',
    'HBM-only': 'hbm_only',
    'HBM+fabric': 'hbm_fabric',
    'VR200-dense': 'vr200_dense',
}
ORIGINAL_MOE = MoE.query
ORIGINAL_DISPATCH = MoEDispatch.query


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def setup_systems():
    original = yaml.safe_load((base.AIC_REPO / 'aic-core/src/aiconfigurator_core/systems/gb300.yaml').read_text())
    original['data_dir'] = str(base.AIC_REPO / 'aic-core/src/aiconfigurator_core/systems/data/gb300')
    records = {}
    for label, ident in SYSTEMS.items():
        cfg = copy.deepcopy(original)
        if label != 'GB300':
            cfg['gpu']['mem_bw'] = 22e12
        if label in ('HBM+fabric', 'VR200-dense'):
            for k in ('inter_node_bw', 'intra_node_bw'):
                cfg['node'][k] = 1.8e12  # directional; 3.6 TB/s bidirectional
            cfg['node']['inter_rack_bw'] = 200e9
        if label == 'VR200-dense':
            cfg['gpu'].update(fp8_tc_flops=17.5e15, bfloat16_tc_flops=4e15,
                              fp4_tc_flops=35e15, int8_tc_flops=250e12)
            # Published capacity interpreted conservatively as decimal GB.
            # Driver-visible capacity is not yet a validated input for this study.
            cfg['gpu']['mem_capacity'] = 288_000_000_000
        path = HERE / 'systems' / (ident + '.yaml')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        records[label] = cfg
    write_json(OUT / 'hardware.json', records)
    return records


@lru_cache(maxsize=None)
def destinations(experts, topk, ep):
    """Expected distinct remote ranks: uniform top-k without replacement.

    Average over source ranks; expert counts differ by at most one if E%EP!=0.
    """
    if ep == 1:
        return 0.0
    sizes = [experts // ep + (i < experts % ep) for i in range(ep)]
    hit = []
    for size in sizes:
        miss = 1.0
        for j in range(min(topk, experts)):
            miss *= max(0, experts - size - j) / (experts - j)
        hit.append(1-miss)
    return sum(hit)*(ep-1)/ep


def distinct_experts(experts, topk, tokens):
    if tokens <= 0:
        return 0.0
    if topk >= experts:
        return float(experts)
    return -experts * math.expm1(tokens * math.log1p(-topk/experts))


@lru_cache(maxsize=None)
def rank_tail(experts,topk,tokens,ep):
    """Uniform top-k routing sensitivity, mean maximum over 80 synthetic batches.

    Separately maximizes unique weight reads and expert executions. This is a
    conservative MoE-only critical-rank approximation, not measured routing.
    """
    rng=random.Random(20260907)
    max_unique=max_work=0
    for _ in range(80):
        touched=[set() for _ in range(ep)];counts=[0]*ep
        for _ in range(int(tokens)):
            for expert in rng.sample(range(experts),min(topk,experts)):
                rank=expert%ep;touched[rank].add(expert);counts[rank]+=1
        max_unique+=max(map(len,touched));max_work+=max(counts)
    return max_unique/80,max_work/80


@contextlib.contextmanager
def extensions(policy='routed', occupancy='expected', route_precision='bf16', dsa_dims=None):
    saved_dims = copy.deepcopy(DSA_MODEL_DIMS)
    ledger = []

    def moe_query(op, database, **kw):
        if occupancy == 'legacy':
            return ORIGINAL_MOE(op, database, **kw)
        tokens = kw['x'] * op._attention_dp_size
        e, k, ep, tp = op._num_experts, op._topk, op._moe_ep_size, op._moe_tp_size
        h, inter = op._hidden_size, op._inter_size
        q = kw.get('quant_mode') or op._quant_mode
        gemms = 3 if op._is_gated else 2
        counts = tokens*k/ep
        unique = distinct_experts(e, k, tokens)/ep
        if occupancy == 'maxrank':
            unique,counts=rank_tail(e,k,tokens,ep)
        if occupancy == 'correlated':
            # Explicit sensitivity: groups of four token routes perfectly correlate.
            unique = distinct_experts(e, k, max(1, tokens/4))/ep
        skew = 1.25 if occupancy == 'skew25' else 1.0
        flops = counts*h*inter*gemms*2/tp*skew
        weight_bytes = unique*h*inter*gemms/tp*q.value.memory
        # MXFP4 block-32 scale bytes omitted by the generic AIC mode (0.5 B).
        scale_bytes = unique*h*inter*gemms/tp/32 if 'mxfp4' in q.name else 0
        # Activation/intermediate traffic must not inherit four-bit weight width.
        act_bytes = 2.0 if q.value.compute_dtype == 'bfloat16' else 1.0
        io_bytes = counts * (2*h + gemms*inter/tp) * act_bytes * skew
        compute_ms = flops/common.get_quant_tc_flops(database.system_spec, q)*1000
        memory_ms = (weight_bytes+scale_bytes+io_bytes)/database.system_spec['gpu']['mem_bw']*1000
        result = max(compute_ms, memory_ms)*op._scale_factor
        ledger.append(dict(op=op._name if hasattr(op,'_name') else 'moe', kind='moe',
                           compute_ms=compute_ms*op._scale_factor,
                           memory_ms=memory_ms*op._scale_factor, time_ms=result,
                           extra_weight_bytes=(e/ep)*h*inter*gemms/tp/32*op._scale_factor if 'mxfp4' in q.name else 0))
        return PerformanceResult(result, energy=0, source='analytical_extension')

    def dispatch_query(op, database, **kw):
        # Only pure DEP is replaced. Mixed TP/CP retains AIC's native collectives.
        if policy == 'legacy' or op._attention_tp_size != 1 or op._moe_tp_size != 1 or op._attn_cp_size != 1:
            return ORIGINAL_DISPATCH(op, database, **kw)
        ep = op._moe_ep_size
        remotes = destinations(op._num_experts, op._topk, ep)
        tokens = kw['x']
        width = 1 if route_precision == 'fp8_dispatch' and op._pre_dispatch else 2
        # Pack each token once per remote rank; combine locally reduces that rank's experts.
        data_bytes = tokens*op._hidden_size*width*remotes
        if op._pre_dispatch:
            data_bytes += tokens*op._topk*4*(ep-1)/ep  # destination indices
        node = database.system_spec['node']
        bw = node['intra_node_bw'] if ep <= node['num_gpus_per_node'] else node['inter_node_bw']
        ms = data_bytes/bw*1000*op._scale_factor
        ledger.append(dict(kind='dispatch', bytes=data_bytes*op._scale_factor, time_ms=ms))
        return PerformanceResult(ms, energy=0, source='analytical_extension')

    try:
        MoE.query = moe_query
        MoEDispatch.query = dispatch_query
        if dsa_dims:
            DSA_MODEL_DIMS['GlmMoeDsaForCausalLM'].update(dsa_dims)
        yield ledger
    finally:
        MoE.query = ORIGINAL_MOE
        MoEDispatch.query = ORIGINAL_DISPATCH
        DSA_MODEL_DIMS.clear()
        DSA_MODEL_DIMS.update(saved_dims)


def signature(model):
    p = Path(model) / 'config.json'
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'builtin-pinned-checkout'


def identify_attention(lat):
    return sum(v for k,v in lat.items() if
               'attention' in k or '_kda_' in k or '_mla_' in k or '_bmm_' in k or
               k.endswith(('_attn_norm','_attn_res','_mhc_pre','_mhc_post','_qkv_gemm','_proj_gemm')))


class Estimator:
    def __init__(self, hardware):
        self.hardware = hardware
        self.cache = {}
        self.path = OUT / 'aic_cache.jsonl'
        if self.path.exists():
            for line in self.path.open():
                r = json.loads(line)
                self.cache[r['key']] = r
        self.calls = 0

    def estimate(self, model, system, dep=64, cpr=1, context=262144, chunk=1,
                 phase='decode', policy='routed', occupancy='expected', tp=1,
                 etp=1, cp=1, nextn=0, al=1.0, quant='w4a8_mxfp4_mxfp8_trtllm',
                 gemm='fp8_block', kv='fp8', route_precision='bf16', dsa_dims=None):
        task = dict(model=str(model),system=system,dep=dep,cpr=cpr,context=context,
                    chunk=chunk,phase=phase,policy=policy,occupancy=occupancy,tp=tp,etp=etp,
                    cp=cp,nextn=nextn,al=al,quant=quant,gemm=gemm,kv=kv,
                    route_precision=route_precision,dsa_dims=dsa_dims)
        key = hashlib.sha256(json.dumps([VERSION,task,signature(str(model)),self.hardware[system]],sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        try:
            assert cp == 1, 'This CLI does not expose CP; do not silently simulate it as TP/DP.'
            world = dep*tp*cp
            assert world % etp == 0 and world <= 72
            with extensions(policy,occupancy,route_precision,dsa_dims) as ledger, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                r = cli_estimate(str(model), SYSTEMS[system], mode='static_ctx' if phase=='prefill' else 'static_gen',
                    backend_name='sglang',backend_version='estimate',database_mode='SOL',
                    isl=context+chunk if phase=='prefill' else context,
                    prefix=context,osl=1 if phase=='prefill' else 2,batch_size=cpr,
                    tp_size=tp,pp_size=1,attention_dp_size=dep,moe_tp_size=etp,moe_ep_size=world//etp,
                    gemm_quant_mode=gemm,moe_quant_mode=quant,kvcache_quant_mode=kv,
                    fmha_quant_mode='bfloat16',nextn=nextn,nextn_accepted=al-1 if nextn else 0,
                    stride=1,engine_step_backend='python',systems_paths=f'{HERE / "systems"},default')
            lat = dict(r.summary.get_context_latency_dict() if phase=='prefill' else r.summary.get_generation_latency_dict())
            src = dict(r.summary.get_context_source_dict() if phase=='prefill' else r.summary.get_generation_source_dict())
            mem = r.summary.get_memory()
            extra = sum(x.get('extra_weight_bytes',0) for x in ledger)/GIB
            dispatch = sum(v for k,v in lat.items() if 'dispatch' in k)
            moe = sum(v for k,v in lat.items() if 'moe' in k and 'dispatch' not in k)
            attention = identify_attention(lat)
            step = sum(lat.values())
            row = dict(task,key=key,status='ok',world=world,latency=lat,sources=src,
                       step_ms=step,latency_ms=step/al if phase=='decode' else step,
                       attention_ms=attention,dispatch_ms=dispatch,moe_ms=moe,
                       other_ms=step-attention-dispatch-moe,ledger=ledger,
                       memory_gib=float(mem['total'])+extra,weight_gib=float(mem['weights'])+extra,
                       kv_gib=float(mem['kvcache']),overhead_gib=sum(float(mem[k]) for k in ('activations','nccl','others')),
                       capacity_gib=self.hardware[system]['gpu']['mem_capacity']/GIB,
                       non_analytical_ms=sum(v for k,v in lat.items() if v>0 and str(src.get(k)).lower() not in ('sol','analytical_extension')),
                       mapping='routed-pure-DEP' if policy=='routed' and tp==etp==cp==1 else 'AIC-native-collectives')
            row['oom'] = row['memory_gib'] > row['capacity_gib']
            row['admit90'] = row['memory_gib'] <= .9*row['capacity_gib']
        except Exception as e:
            row = dict(task,key=key,status='error',error=f'{type(e).__name__}: {e}')
        self.calls += 1
        if self.calls%500==0:
            print(f'AIC: {self.calls} new estimates',flush=True)
        self.cache[key]=row
        with self.path.open('a') as f:
            f.write(json.dumps(row,sort_keys=True)+'\n')
        return row
