"""Portable extraction of the current study's audited layout/memory/MAI rules.

Extracted without timing or policy changes from qwen_gqa/run_study.py,
prefill_throughput.py, mai2_swa.py and mai2_prefill.py (16 October 2026).
"""
from dataclasses import replace
import json
import math
from pathlib import Path
from . import simulation
CONTEXT = 262144
Q = 1024


def gqa_layouts(hardware, model):
    # Wider TP/DCP than the original pilot. DCP only partitions replicated KV
    # heads; do not grant DCP arbitrary cache sharding at TP <= KV heads.
    for stage in (1, 2, 4, 8, 16, 32, 64, 72, 128):
        if stage > hardware["chip_budget"] or (hardware["restricted"] and stage > 8):
            continue
        for tp in (1, 2, 4, 8, 16, 32, 64):
            if stage % tp or model["q_heads"] % tp or model["hidden"] % tp:
                continue
            kv = model["kv_heads"]
            if (tp <= kv and kv % tp) or (tp > kv and tp % kv):
                continue
            for dcp in (1, 2, 4, 8):
                if dcp > max(1, tp // kv) or tp % dcp:
                    continue
                for etp in (1, 2, 4, 8):
                    if stage % etp or model["experts"] % (stage // etp) or model["expert_width"] % etp:
                        continue
                    pps = (1, 2, 3, 4, 6, 8) if hardware["restricted"] and stage == 8 else (1,)
                    for pp in pps:
                        if stage * pp <= hardware["chip_budget"]:
                            yield dict(tp=tp, dp=stage // tp, dcp=dcp, etp=etp, ep=stage // etp, pp=pp)

def gqa_memory(c, g, batch, nn):
    m = simulation._sdk()[1](c["model"], c["system"], c["backend"], c["backend_version"],
        max_num_tokens=batch * (nn + 1), max_batch_size=batch,
        memory_fraction_kind="of_total", memory_fraction_value=.9,
        tp_size=c["tp"], pp_size=c["pp"], attention_dp_size=c["attention_dp"],
        moe_tp_size=c["moe_tp_size"], moe_ep_size=c["moe_ep_size"], cp_size=1,
        dcp_size=c["dcp"], nextn=nn, gemm_quant_mode=c["gemm_quant_mode"],
        moe_quant_mode=c["moe_quant_mode"], kvcache_quant_mode="fp8",
        fmha_quant_mode="bfloat16", comm_quant_mode="half",
        systems_path=c["systems_paths"][0], allow_naive_fallback=False, worker_type="decode")
    breakdown = m["memory_breakdown"]
    weights = breakdown["weights_bytes"]
    bound = simulation._weight_bound(g, c["pp"]) if c["pp"] > 1 else weights
    cache = batch * g.get_kvcache_rank_bytes_per_sequence(CONTEXT + nn + 1)
    stage_guard = max(0., bound - weights)
    boundary = 2 * batch * (nn + 1) * g._hidden_size * 2 if c["pp"] > 1 else 0.
    metadata = (bound + cache) / 32
    extra = cache + stage_guard + boundary + metadata
    return dict(fits=extra <= m["total_kv_size_bytes"],
        memory_gib=(sum(breakdown.values()) + extra) / 2**30,
        weights_gib=weights / 2**30, cache_gib=cache / 2**30,
        stage_weight_guard_gib=stage_guard / 2**30, metadata_guard_gib=metadata / 2**30,
        boundary_buffer_gib=boundary / 2**30)

def prefill_layouts(hw, model):
    experts = simulation.MODELS[model][3]
    for stage in (1, 2, 4, 8, 16, 32, 64, 72, 128):
        if stage > hw['chip_budget'] or (hw['restricted'] and stage > 8):
            continue
        for tp in (1, 2, 4, 8, 16, 32, 64):
            for etp in (1, 2, 4, 8):
                if stage % tp or stage % etp or experts % (stage // etp):
                    continue
                pps = (1, *simulation.PP_SIZES) if hw['restricted'] and stage == 8 else (1,)
                for pp in pps:
                    if stage * pp <= hw['chip_budget']:
                        yield dict(tp=tp, dp=stage // tp, dcp=1, etp=etp, ep=stage // etp, pp=pp)

def _prefill_memory(c, g, batch, q):
    m = simulation._sdk()[1](c['model'], c['system'], c['backend'], c['backend_version'],
        max_num_tokens=batch*q, max_batch_size=batch,
        memory_fraction_kind='of_total', memory_fraction_value=.9,
        tp_size=c['tp'], pp_size=c['pp'], attention_dp_size=c['attention_dp'],
        moe_tp_size=c['moe_tp_size'], moe_ep_size=c['moe_ep_size'], cp_size=1,
        dcp_size=1, nextn=0, gemm_quant_mode=c['gemm_quant_mode'],
        moe_quant_mode=c['moe_quant_mode'], kvcache_quant_mode=c['kvcache_quant_mode'],
        fmha_quant_mode=c['fmha_quant_mode'], comm_quant_mode=c['comm_quant_mode'],
        systems_path=c['systems_paths'][0], allow_naive_fallback=False, worker_type='prefill')
    bd = m['memory_breakdown']
    kv = batch*g.get_kvcache_rank_bytes_per_sequence(simulation.CONTEXT+q)
    bound = simulation._weight_bound(g, c['pp']) if c['pp'] > 1 else bd['weights_bytes']
    guard = max(0., bound-bd['weights_bytes'])
    boundary = 2*batch*q*g._hidden_size*2 if c['pp'] > 1 else 0.
    metadata = (bound+kv)/32
    extra = kv+guard+boundary+metadata
    return dict(fits=extra <= m['total_kv_size_bytes'],
        memory_GiB=(sum(bd.values())+extra)/2**30, cache_GiB=kv/2**30,
        weights_GiB=bd['weights_bytes']/2**30, stage_guard_GiB=guard/2**30,
        boundary_GiB=boundary/2**30, metadata_GiB=metadata/2**30)

def prefill_memory(c, g, batch, q):
    try:
        return _prefill_memory(c, g, batch, q)
    except ValueError as e:
        # An oversized workspace is an admission failure, not a reason to
        # discard smaller valid batches while bracketing the memory limit.
        if 'no KV budget:' in str(e):
            return dict(fits=False)
        raise

def install_adapter():
    from aisimulate_core.sdk.models.hybrid_moe import HybridMoEModel
    from aisimulate_core.sdk.models.base import _MODEL_REGISTRY
    from aisimulate_core.sdk import operations as ops

    class MaiProxy(HybridMoEModel):
        def set_hybrid_config(self, cfg):
            raw = json.loads((Path(self.model_path) / 'config.json').read_text())
            self.mai = raw['mai_proxy']
            cfg = replace(cfg, use_qk_norm=True)  # inherited from Base-1
            super().set_hybrid_config(cfg)
            # Hybrid family does not emit attention-output TP all-reduces.
            # Add them explicitly, plus shared-FFN TP output reduction.
            for phase, oplist, scale in [('context', self.context_ops, 1),
                                         ('generation', self.generation_ops, self._mtp_scale_factor)]:
                if self.config.tp_size > 1:
                    for part in ('attention', 'shared'):
                        oplist.append(ops.NCCL(f'{phase}_{part}_allreduce', self._num_layers * scale,
                            'all_reduce', num_elements_per_token=self._hidden_size,
                            num_gpus=self.config.tp_size, comm_quant_mode=self.config.comm_quant_mode))
                # Base-1 uses pre/post norms; account for output norms/residuals
                # not explicitly present in the inherited hybrid graph.
                oplist.append(ops.ElementWise(f'{phase}_postnorm_residual', self._num_layers * scale,
                    6 * self._hidden_size, 4 * self._hidden_size, 0.8))

        def _moe_ops(self, prefix, count, h, moe_q, wl_dist):
            latent = 4096
            block = super()._moe_ops(prefix, count, latent, moe_q, wl_dist)
            # Router sees original hidden states, not the compressed latent.
            block[0] = ops.GEMM(f'{prefix}_router_gemm', count, 512, h, self.config.gemm_quant_mode)
            block[1] = ops.MoEDispatch(f'{prefix}_moe_pre_dispatch', count, latent, 7, 512,
                self.config.moe_tp_size, self.config.moe_ep_size, self.config.attention_dp_size,
                True, quant_mode=None, backend='vllm', attn_ar_modeled=True)
            # Replicated compression/decompression is a valid conservative TP
            # implementation: no unaccounted projection all-gather is assumed.
            block.insert(1, ops.GEMM(f'{prefix}_latent_down', count, latent, h, self.config.gemm_quant_mode))
            block.append(ops.GEMM(f'{prefix}_latent_up', count, h, latent, self.config.gemm_quant_mode))
            block.extend(self._dense_ffn_ops(f'{prefix}_shared', count, h, self.config.tp_size,
                self.mai['shared_intermediate'] // self.config.tp_size, self.config.gemm_quant_mode))
            return block

        def get_kvcache_bytes_per_sequence(self, seq_len):
            tp = self.config.tp_size
            d = self._head_size
            w = self.mai['window']
            return 2 * d * self.config.kvcache_quant_mode.value.memory * (
                17 * math.ceil(2 / tp) * seq_len + 85 * math.ceil(16 / tp) * min(seq_len, w))

        def get_kvcache_elements_per_token(self):
            return int(self.get_kvcache_bytes_per_sequence(1) / self.config.kvcache_quant_mode.value.memory)

        def get_kvcache_max_tokens(self, budget):
            return self._binary_search_kvcache_max_tokens(budget)

    _MODEL_REGISTRY['HYBRIDMOE'] = MaiProxy

def mai_memory(c, g, batch, nn):
    if c['pp'] == 1:
        return gqa_memory(c, g, batch, nn)
    # Grouped global/local descriptors cannot use the old generic layer
    # guard. Count actual contiguous balanced stages, including a replicated
    # fixed embedding/logit reservation. Keep the existing conservative
    # whole-model KV reservation on every stage, not an untested KV discount.
    fixed, global_layer, local_layer = 0., 0., 0.
    for op in g.context_ops:
        weight = op.get_weights()
        if '_global_' in op._name:
            global_layer += weight / 17
        elif '_swa_' in op._name:
            local_layer += weight / 85
        else:
            fixed += weight
    assert math.isclose(fixed + 17*global_layer + 85*local_layer,
                        g.get_resident_weights_bytes(), rel_tol=1e-12)
    counts, offset = [], 0
    for stage in range(c['pp']):
        size = 102 // c['pp'] + (stage < 102 % c['pp'])
        ng = sum(i % 6 == 5 for i in range(offset, offset + size))
        counts.append(fixed + ng*global_layer + (size-ng)*local_layer)
        offset += size
    assert offset == 102
    bound = max(counts)
    m = simulation._sdk()[1](c['model'],c['system'],c['backend'],c['backend_version'],
        max_num_tokens=batch*(nn+1),max_batch_size=batch,
        memory_fraction_kind='of_total',memory_fraction_value=.9,
        tp_size=c['tp'],pp_size=c['pp'],attention_dp_size=c['attention_dp'],
        moe_tp_size=c['moe_tp_size'],moe_ep_size=c['moe_ep_size'],cp_size=1,dcp_size=1,nextn=nn,
        **{k:c[k] for k in ('gemm_quant_mode','moe_quant_mode','kvcache_quant_mode','fmha_quant_mode','comm_quant_mode')},
        systems_path=c['systems_paths'][0],allow_naive_fallback=False,worker_type='decode')
    bd=m['memory_breakdown'];weights=bd['weights_bytes']
    cache=batch*g.get_kvcache_rank_bytes_per_sequence(CONTEXT+nn+1)
    guard=max(0.,bound-weights);boundary=2*batch*(nn+1)*g._hidden_size*2
    metadata=(bound+cache)/32;extra=cache+guard+boundary+metadata
    return dict(fits=extra<=m['total_kv_size_bytes'],memory_gib=(sum(bd.values())+extra)/2**30,
        weights_gib=weights/2**30,cache_gib=cache/2**30,stage_weight_guard_gib=guard/2**30,
        metadata_guard_gib=metadata/2**30,boundary_buffer_gib=boundary/2**30)

def mai_prefill_memory(c, g, batch):
    try:
        m = simulation._sdk()[1](c['model'],c['system'],c['backend'],c['backend_version'],
            max_num_tokens=batch*Q,max_batch_size=batch,
            memory_fraction_kind='of_total',memory_fraction_value=.9,
            tp_size=c['tp'],pp_size=c['pp'],attention_dp_size=c['attention_dp'],
            moe_tp_size=c['moe_tp_size'],moe_ep_size=c['moe_ep_size'],cp_size=1,dcp_size=1,nextn=0,
            **{k:c[k] for k in ('gemm_quant_mode','moe_quant_mode','kvcache_quant_mode','fmha_quant_mode','comm_quant_mode')},
            systems_path=c['systems_paths'][0],allow_naive_fallback=False,worker_type='prefill')
    except ValueError as e:
        if 'no KV budget:' in str(e):return dict(fits=False)
        raise
    bd=m['memory_breakdown'];weights=bd['weights_bytes'];bound=weights
    if c['pp']>1:
        fixed, global_layer, local_layer=0.,0.,0.
        for op in g.context_ops:
            w=op.get_weights()
            if '_global_' in op._name:global_layer+=w/17
            elif '_swa_' in op._name:local_layer+=w/85
            else:fixed+=w
        assert math.isclose(fixed+17*global_layer+85*local_layer,
                            g.get_resident_weights_bytes(),rel_tol=1e-12)
        counts=[];offset=0
        for stage in range(c['pp']):
            size=102//c['pp']+(stage<102%c['pp'])
            ng=sum(i%6==5 for i in range(offset,offset+size))
            counts.append(fixed+ng*global_layer+(size-ng)*local_layer);offset+=size
        assert offset==102
        bound=max(counts)
    cache=batch*g.get_kvcache_rank_bytes_per_sequence(CONTEXT+Q)
    guard=max(0.,bound-weights)
    boundary=2*batch*Q*g._hidden_size*2 if c['pp']>1 else 0.
    metadata=(bound+cache)/32;extra=cache+guard+boundary+metadata
    return dict(fits=extra<=m['total_kv_size_bytes'],memory_gib=(sum(bd.values())+extra)/2**30,
        weights_gib=weights/2**30,cache_gib=cache/2**30,stage_weight_guard_gib=guard/2**30,
        metadata_guard_gib=metadata/2**30,boundary_buffer_gib=boundary/2**30)
