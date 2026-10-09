"""Self-contained adapter for the frozen AISim October 5 analytical engine.

Extracted from study.py, node_local_policy.py and the hardware sweep wrappers.
No serving engine, weights download, historical experiment import, or AISim
patch is required. Private descriptor introspection is deliberately version
pinned. Timing is native serial traversal, not an interleaved PP scheduler.
"""
from functools import lru_cache
import importlib.metadata
import json
import math
import os
from pathlib import Path

from .topology import penalty_ms

PINNED_AISIM = "0.13.0.dev202610050000000066"
CONTEXT = 262144
NEXTN = 3
ACCEPTED_LENGTH = 3
MODELS = {
    "GLM-5.2": ("nvidia/GLM-5.2-NVFP4", "vllm", "0.24.0", 256),
    "Kimi-K3": ("moonshotai/Kimi-K3", "sglang", "0.5.17", 896),
}
LAYOUTS = ((1, 1, 1), (2, 1, 1), (2, 2, 1), (4, 1, 1),
           (4, 4, 1), (4, 4, 4), (8, 1, 1), (8, 8, 1),
           (8, 8, 8), (2, 2, 2), (4, 4, 2), (8, 8, 2))
PP_SIZES = (2, 3, 4, 6, 8)


@lru_cache(maxsize=1)
def _sdk():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["AIC_ALLOW_UNLISTED_VERSIONS"] = "1"
    actual = importlib.metadata.version("aisimulate")
    if actual != PINNED_AISIM:
        raise RuntimeError(f"Reproduction requires AISim {PINNED_AISIM}; installed {actual}")
    from aisimulate_core.sdk import RustForwardPassPerfModel, estimate_kv_cache
    from aisimulate_core.sdk.config_builders import build_model_config, apply_nextn
    from aisimulate_core.sdk.models import get_model
    return RustForwardPassPerfModel, estimate_kv_cache, build_model_config, apply_nextn, get_model


def make_config(hardware, model, tp, dp, dcp, etp, ep, pp, systems_path=None):
    """Build the exact original workload/configuration without running it."""
    model_id, backend, version, experts = MODELS[model]
    values = (tp, dp, dcp, etp, ep, pp)
    if any(not isinstance(v, int) or v < 1 for v in values):
        raise ValueError("Parallelism dimensions must be positive integers")
    if tp * dp != etp * ep or tp % dcp or experts % ep:
        raise ValueError("Incompatible parallelism or nonintegral expert placement")
    if tp * dp * pp > hardware["chip_budget"]:
        raise ValueError("Instance exceeds the chip search budget")
    if hardware.get("restricted"):
        if tp * dp > 8 or (pp > 1 and (tp * dp != 8 or pp not in PP_SIZES)):
            raise ValueError("Restricted policy requires node-local collectives / 8-chip serial PP")
    elif pp != 1:
        raise ValueError("Published unrestricted-hardware search did not include PP")
    if systems_path is None:
        systems_path = Path(__file__).resolve().parents[2] / "configs" / "hardware"
    systems_path = Path(systems_path).resolve()
    profile = hardware["profile"]
    if not (systems_path / (profile + ".yaml")).is_file():
        raise FileNotFoundError(systems_path / (profile + ".yaml"))
    return dict(model=model_id, system=profile, backend=backend, backend_version=version,
        worker_type="decode", tp=tp, pp=pp, attention_dp=dp, dcp=dcp, cp_size=1,
        moe_tp_size=etp, moe_ep_size=ep, nextn=NEXTN, enable_shared_layer=False,
        gemm_quant_mode="bfloat16", moe_quant_mode=hardware["quant"],
        fmha_quant_mode="bfloat16", kvcache_quant_mode="fp8", comm_quant_mode="half",
        estimation_mode="op_level", database_mode="SOL", fallback_policy="deny",
        systems_paths=(str(systems_path),), estimator_config={"correction": {"enabled": False}})


@lru_cache(maxsize=256)
def _native(config_json):
    perf, _, build, apply, get_model = _sdk()
    c = json.loads(config_json)
    mc = build(tp_size=c["tp"], pp_size=c["pp"], attention_dp_size=c["attention_dp"],
        moe_tp_size=c["moe_tp_size"], moe_ep_size=c["moe_ep_size"], cp_size=1,
        dcp_size=c["dcp"], gemm_quant_mode=c["gemm_quant_mode"],
        moe_quant_mode=c["moe_quant_mode"], fmha_quant_mode=c["fmha_quant_mode"],
        kvcache_quant_mode=c["kvcache_quant_mode"], comm_quant_mode=c["comm_quant_mode"])
    apply(mc, NEXTN)
    g = get_model(c["model"], mc, c["backend"])
    return perf.best_available(c), g


def _weight_bound(g, pp):
    fixed, per_layer = 0., 0.
    for op in g.context_ops:
        weight = op.get_weights()
        if not weight:
            continue
        scale = next(iter(json.loads(op._spec_json()).values()))["scale_factor"]
        if scale < 1 or not float(scale).is_integer():
            raise ValueError("Unaudited fractional layer weight descriptor")
        if scale == 1:
            fixed += weight
        else:
            per_layer += weight / scale
    inventory = sum(op.get_weights() for op in g.context_ops)
    if not math.isclose(inventory, g.get_resident_weights_bytes(), rel_tol=1e-12):
        raise ValueError("Native weight inventory changed")
    return fixed + math.ceil(g._num_layers / pp) * per_layer


def _memory(c, g, batch):
    estimate = _sdk()[1]
    m = estimate(c["model"], c["system"], c["backend"], c["backend_version"],
        max_num_tokens=batch * 4, max_batch_size=batch,
        memory_fraction_kind="of_total", memory_fraction_value=.9,
        tp_size=c["tp"], pp_size=c["pp"], attention_dp_size=c["attention_dp"],
        moe_tp_size=c["moe_tp_size"], moe_ep_size=c["moe_ep_size"], cp_size=1,
        dcp_size=c["dcp"], nextn=NEXTN, gemm_quant_mode=c["gemm_quant_mode"],
        moe_quant_mode=c["moe_quant_mode"], kvcache_quant_mode=c["kvcache_quant_mode"],
        fmha_quant_mode=c["fmha_quant_mode"], comm_quant_mode=c["comm_quant_mode"],
        systems_path=c["systems_paths"][0], allow_naive_fallback=False, worker_type="decode")
    breakdown = m["memory_breakdown"]
    weights = breakdown["weights_bytes"]
    cache = batch * g.get_kvcache_rank_bytes_per_sequence(CONTEXT + 4)
    stage_guard = boundary = 0.
    bound = weights
    if c["pp"] > 1:
        if not math.isclose(weights, g.get_resident_weights_bytes() / c["pp"], rel_tol=1e-10):
            raise ValueError("Native stage weight accounting changed")
        bound = _weight_bound(g, c["pp"])
        stage_guard = max(0., bound - weights)
        boundary = 2 * batch * 4 * g._hidden_size * 2
    metadata = (bound + cache) / 32
    extra = cache + stage_guard + boundary + metadata
    return dict(fits=extra <= m["total_kv_size_bytes"],
        memory_gib=(sum(breakdown.values()) + extra) / 2**30,
        weights_gib=weights / 2**30, cache_gib=cache / 2**30,
        metadata_guard_gib=metadata / 2**30, stage_weight_bound_gib=bound / 2**30,
        stage_weight_guard_gib=stage_guard / 2**30, boundary_buffer_gib=boundary / 2**30)


def max_batch(hardware, model, tp, dp, dcp, etp, ep, pp=1, systems_path=None):
    c = make_config(hardware, model, tp, dp, dcp, etp, ep, pp, systems_path)
    _, g = _native(json.dumps(c, sort_keys=True))
    lo, hi = 0, 1
    while hi < 65536 and _memory(c, g, hi)["fits"]:
        lo, hi = hi, hi * 2
    if hi == 65536 and _memory(c, g, hi)["fits"]:
        raise ValueError("Capacity search limit reached")
    while lo + 1 < hi:
        middle = (lo + hi) // 2
        if _memory(c, g, middle)["fits"]:
            lo = middle
        else:
            hi = middle
    return lo


def evaluate_candidate(hardware, model, tp, dp, dcp, etp, ep, pp, batch,
                       systems_path=None, include_operations=False):
    """Evaluate one saved candidate; retain unfit status, never extrapolate it."""
    if not isinstance(batch, int) or batch < 1:
        raise ValueError("Batch must be a positive integer")
    c = make_config(hardware, model, tp, dp, dcp, etp, ep, pp, systems_path)
    m, g = _native(json.dumps(c, sort_keys=True))
    ops = m.static_phase_diagnostics(batch_size=batch, context_length=CONTEXT,
                                     prefix=0, prefill=False)
    native_ms = sum(op["latency_ms"] for op in ops)
    if native_ms <= 0 or any(op["latency_ms"] < 0 for op in ops):
        raise ValueError("Nonpositive/negative native operation timing")
    hop_ms = penalty_ms([json.loads(op._spec_json()) for op in g.generation_ops]) if hardware.get("hop_overlay") else 0.
    ms = native_ms + hop_ms
    requests, chips = batch * dp, tp * dp * pp
    result = dict(hardware=hardware["name"], model=model, profile=c["system"],
        status="ok", phase="decode", context=CONTEXT, new_tokens=1, nextn=NEXTN,
        al=ACCEPTED_LENGTH, quant=c["moe_quant_mode"], tp=tp, dp=dp, dcp=dcp,
        etp=etp, ep=ep, pp=pp, cp=1, world=chips, chip_budget=hardware["chip_budget"],
        batch_per_replica=batch, total_requests=requests, concurrency_per_rank=requests/chips,
        native_ms=native_ms, raw_ms=ms, hop_addition_ms=hop_ms, tpot_ms=ms/3,
        tps_user=3000/ms, tokens_s_accelerator=3000/ms*requests/chips,
        schedule="serial_single_wave" if pp > 1 else "native_pp1",
        boundary_ms=sum(op["latency_ms"] for op in ops if "p2p" in op["name"]),
        **_memory(c, g, batch))
    result["parallelism"] = f"TP{tp} DP{dp} DCP{dcp} ETP{etp} EP{ep} PP{pp}"
    if include_operations:
        result["operations"] = ops
    return result


def enumerate_layouts(hardware, model):
    """Original finite layout family, pruned to current headline policy/budget."""
    experts = MODELS[model][3]
    worlds = (4, 8) if hardware.get("restricted") else (4, 8, 16, 32, 64, 72, 128)
    for stage in worlds:
        if stage > hardware["chip_budget"]:
            continue
        for tp, dcp, etp in LAYOUTS:
            if max(tp, etp) > stage or stage % tp or stage % etp or experts % (stage // etp):
                continue
            pps = (1, *PP_SIZES) if hardware.get("restricted") and stage == 8 else (1,)
            for pp in pps:
                if stage * pp <= hardware["chip_budget"]:
                    yield dict(tp=tp, dp=stage//tp, dcp=dcp, etp=etp, ep=stage//etp, pp=pp)


def sweep(hardware, model, systems_path=None, audit=None):
    """Yield every memory-admitted integer batch for each published layout.

    No-budget layouts are omitted as in the source study; other errors abort.
    Caller owns output/checkpointing. No prior results or hypothetical models
    are loaded. The exhaustive sweep can take substantial CPU time.
    """
    for layout in enumerate_layouts(hardware, model):
        try:
            limit = max_batch(hardware, model, systems_path=systems_path, **layout)
        except ValueError as error:
            if "no KV budget:" in str(error):
                if audit:
                    audit(dict(layout=layout, status="excluded", reason=str(error)))
                continue
            raise
        if audit:
            audit(dict(layout=layout, status="admitted" if limit else "excluded",
                       max_batch_per_replica=limit,
                       reason="" if limit else "No admitted request"))
        previous = 0.
        for batch in range(1, limit + 1):
            row = evaluate_candidate(hardware, model, batch=batch,
                                     systems_path=systems_path, **layout)
            if not row["fits"] or row["raw_ms"] < previous - 1e-9:
                raise ValueError("Admission/monotonicity check failed")
            previous = row["raw_ms"]
            row["max_batch_per_replica"] = limit
            row["max_total_requests"] = limit * layout["dp"]
            yield row
