#!/usr/bin/env python3
"""Unified 256K-prefix cached-prefill sweep across every prior architecture family.

The request is fixed at 262,144 resident prefix tokens plus a variable extend
chunk.  AIC ``static_ctx`` SOL is evaluated at DEP64 on GB300 and the controlled
VR200 HBM+NVLink projection.  The full factorial and SWA studies are reconstructed
from the same prefix-aware AIC operation primitives used by their source studies.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_DIR = ROOT / "experiments/5t-architecture-design-study"
SWA_DIR = ROOT / "experiments/5t-swa-architecture-study"
OLD_PREFILL_DIR = ROOT / "experiments/5t-swa-prefill-study"
OUTPUT = HERE / "output"
CACHE = OUTPUT / "estimate_cache.jsonl"

sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aic-unified-prefill-mpl")

import run_study as base  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


swa = load_module("unified_swa", SWA_DIR / "run_swa_study.py")
from aiconfigurator.cli.api import cli_estimate  # noqa: E402


PREFIX = 262_144
CHUNKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096)
CPRS = (1, 2, 4, 8, 16, 32, 64)
DEP = 64
SYSTEMS = ("GB300", "VR200 HBM+NVLink")
CAPACITY_GIB = base.SYSTEMS["GB300"]["capacity"] / 2**30
GIB = 2**30


def signature(model: str) -> str:
    path = Path(model) / "config.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "builtin"


def cache_key(model: str, system: str, chunk: int, cpr: int) -> str:
    return json.dumps(
        [model, signature(model), system, PREFIX, chunk, DEP, cpr], separators=(",", ":")
    )


def load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE.exists():
        return {}
    return {
        row["cache_key"]: row
        for row in (
            json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    }


def classify_latency(latency: dict[str, float]) -> dict[str, float]:
    attention = dispatch = moe = 0.0
    for name, value in latency.items():
        lower = name.lower()
        if "pre_dispatch" in lower or "post_dispatch" in lower:
            dispatch += value
        elif (
            "attention" in lower or "_attn" in lower or "_kda_" in lower
            or "_mla_" in lower or "_mhc_" in lower or "_bmm_" in lower
        ):
            attention += value
        elif "moe" in lower:
            moe += value
    total = sum(latency.values())
    return {
        "attention_ms": attention,
        "dispatch_ms": dispatch,
        "moe_compute_ms": moe,
        "other_ms": max(0.0, total - attention - dispatch - moe),
    }


def estimate(model: str, system: str, chunk: int, cpr: int) -> dict[str, Any]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = cli_estimate(
            model,
            base.SYSTEMS[system]["id"],
            mode="static_ctx",
            backend_name="sglang",
            backend_version="estimate",
            database_mode="SOL",
            isl=PREFIX + chunk,
            osl=1,
            prefix=PREFIX,
            batch_size=cpr,
            tp_size=1,
            pp_size=1,
            attention_dp_size=DEP,
            moe_tp_size=1,
            moe_ep_size=DEP,
            gemm_quant_mode="fp8_block",
            moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
            kvcache_quant_mode="fp8",
            fmha_quant_mode="bfloat16",
            stride=1,
            engine_step_backend="python",
            systems_paths=base.SYSTEMS_PATHS,
        )
    latency = {str(k): float(v) for k, v in result.summary.get_context_latency_dict().items()}
    sources = {
        str(k): str(v or "unknown").lower()
        for k, v in result.summary.get_context_source_dict().items()
    }
    memory = result.summary.get_memory()
    return {
        "cache_key": cache_key(model, system, chunk, cpr),
        "model": model,
        "system": system,
        "chunk": chunk,
        "cpr": cpr,
        "latency_ms": sum(latency.values()),
        "latency": latency,
        "sources": sources,
        "non_sol_ms": sum(v for k, v in latency.items() if v > 0 and sources.get(k) != "sol"),
        "weights_gib": float(memory["weights"]),
        "kvcache_gib": float(memory["kvcache"]),
        "activations_gib": float(memory["activations"]),
        "nccl_gib": float(memory["nccl"]),
        "others_gib": float(memory["others"]),
        "memory_gib": float(memory["total"]),
        "oom": bool(result.summary.check_oom()),
        **classify_latency(latency),
    }


def run_estimates(models: list[str]) -> dict[str, dict[str, Any]]:
    cache = load_cache()
    models = list(dict.fromkeys(models))
    tasks = [(m, s, n, c) for m in models for s in SYSTEMS for n in CHUNKS for c in CPRS]
    pending = [task for task in tasks if cache_key(*task) not in cache]
    print(f"AIC references: {len(tasks)-len(pending)} cached, {len(pending)} pending", flush=True)
    if pending:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as stream:
            for index, task in enumerate(pending, 1):
                row = estimate(*task)
                cache[row["cache_key"]] = row
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                if index % 250 == 0 or index == len(pending):
                    print(f"  completed {index}/{len(pending)}", flush=True)
    return cache


def get(cache: dict[str, dict[str, Any]], model: str, system: str, chunk: int, cpr: int):
    return cache[cache_key(model, system, chunk, cpr)]


def config_dims(model: str) -> dict[str, Any]:
    path = Path(model) / "config.json"
    if not path.exists():
        return {}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg = cfg.get("text_config", cfg)
    return {
        "depth": cfg.get("num_hidden_layers"),
        "experts": cfg.get("num_experts", cfg.get("n_routed_experts")),
        "topk": cfg.get("num_experts_per_token", cfg.get("num_experts_per_tok")),
        "shared_experts": cfg.get("num_shared_experts", cfg.get("n_shared_experts")),
    }


def exact_models(glm_path: str) -> list[dict[str, Any]]:
    scale_manifest = json.loads(
        (ROOT / "experiments/kimi-k3-scale-up/models/manifest.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    for item in scale_manifest:
        rows.append({
            "id": item["id"], "label": item["label"], "family": "Kimi K3 scaling",
            "subcategory": item["family"], "model": item["model_path"],
            "total_t": item["text_parameters_trillion"],
            "active_b": item["active_parameters_billion"], "depth": item["num_layers"],
            "experts": item["num_experts"], "topk": item["topk"],
            "shared_experts": item["num_shared_experts"], "attention": "native KDA/MLA",
        })

    ds_meta = json.loads(
        (ROOT / "experiments/deepseek-v4-5t-sol/output/metadata.json").read_text(encoding="utf-8")
    )["variant"]
    rows.append({
        "id": "deepseek_v4_pro_5t_expert", "label": "DeepSeek V4 Pro ~5T expert-only",
        "family": "DeepSeek scaling", "subcategory": "expert_only",
        "model": str(ROOT / "experiments/deepseek-v4-5t-sol/models/deepseek_v4_pro_5t_expert"),
        "total_t": ds_meta["estimated_total_parameters_trillion"],
        "active_b": ds_meta["estimated_active_parameters_billion"],
        "depth": ds_meta["num_hidden_layers"], "experts": ds_meta["scaled_experts"],
        "topk": ds_meta["experts_per_token"], "shared_experts": 1,
        "attention": "native V4 compressed sparse",
    })

    controlled = json.loads(
        (ROOT / "experiments/kimi-vs-deepseek-5t-controlled/output/metadata.json").read_text(encoding="utf-8")
    )
    for label, item in controlled.items():
        dims = config_dims(item["path"])
        rows.append({
            "id": item["id"], "label": label, "family": "Controlled Kimi vs DeepSeek",
            "subcategory": item["family"].lower(), "model": item["path"],
            "total_t": item["total"] / 1e12, "active_b": item["active"] / 1e9,
            "attention": "native KDA/MLA" if item["family"] == "Kimi" else "native V4 compressed sparse",
            **dims,
        })

    rows.extend([
        {
            "id": "deepseek_v4_pro_native", "label": "DeepSeek V4 Pro native",
            "family": "Native sparse anchors", "subcategory": "deepseek",
            "model": "deepseek-ai/DeepSeek-V4-Pro", "total_t": 1.6, "active_b": 49.0,
            "depth": 61, "experts": 384, "topk": 6, "shared_experts": 1,
            "attention": "native V4 compressed sparse",
        },
        {
            "id": "glm52_native", "label": "GLM-5.2 NVFP4 native",
            "family": "Native sparse anchors", "subcategory": "glm",
            "model": "nvidia/GLM-5.2-NVFP4", "total_t": None, "active_b": None,
            "depth": 78, "experts": 256, "topk": 8, "shared_experts": 1,
            "attention": "native DSA top-2048",
        },
        {
            "id": "glm52_ratio_preserved_5t", "label": "GLM-5.2 ratio-preserved 5T",
            "family": "GLM scaling", "subcategory": "ratio_preserved",
            "model": glm_path, "total_t": 5.0, "active_b": 160.0,
            "depth": 78, "experts": 1664, "topk": 52, "shared_experts": 1,
            "attention": "native DSA top-2048",
        },
    ])
    return rows


def common_fields(
    architecture_id: str, label: str, family: str, kind: str, system: str,
    chunk: int, cpr: int, latency: float,
) -> dict[str, Any]:
    return {
        "architecture_id": architecture_id, "label": label, "family": family,
        "study_kind": kind, "system": system, "prefix_tokens": PREFIX,
        "new_tokens_per_request": chunk, "postfill_context_tokens": PREFIX + chunk,
        "dep": DEP, "cpr": cpr, "global_concurrency": DEP * cpr,
        "new_tokens_per_rank": chunk * cpr, "latency_ms": latency,
        "input_tps_user": chunk * 1000 / latency,
        "rack_input_tps": DEP * cpr * chunk * 1000 / latency,
    }


def exact_rows(models: list[dict[str, Any]], cache: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        for system in SYSTEMS:
            for chunk in CHUNKS:
                for cpr in CPRS:
                    run = get(cache, model["model"], system, chunk, cpr)
                    row = common_fields(
                        "exact__" + model["id"], model["label"], model["family"],
                        "direct AIC config", system, chunk, cpr, run["latency_ms"],
                    )
                    row.update({
                        "body_id": model["id"], "attention_id": "native",
                        "attention": model["attention"], "subcategory": model["subcategory"],
                        "total_t": model.get("total_t"), "active_b": model.get("active_b"),
                        "depth": model.get("depth"), "experts": model.get("experts"),
                        "topk": model.get("topk"), "shared_experts": model.get("shared_experts"),
                        "risk": "native/scaling hypothesis", "weight_gib": run["weights_gib"],
                        "kv_gib": run["kvcache_gib"], "memory_gib": run["memory_gib"],
                        "capacity_gib": CAPACITY_GIB, "oom": run["oom"],
                        "attention_ms": run["attention_ms"], "dispatch_ms": run["dispatch_ms"],
                        "moe_compute_ms": run["moe_compute_ms"], "body_other_ms": run["other_ms"],
                        "non_sol_ms": run["non_sol_ms"],
                    })
                    rows.append(row)
    return rows


def context_kimi_attention(latency: dict[str, float]) -> float:
    return sum(
        value for name, value in latency.items()
        if name.startswith("context_kda_") or name.startswith("context_mla_")
        or name.startswith("context_bmm_") or name in {"context_attention", "context_attn_res"}
    )


def context_v4_attention(latency: dict[str, float]) -> float:
    return sum(latency.get(name, 0.0) for name in (
        "context_attention", "context_mhc_pre", "context_mhc_post", "context_attn_norm"
    ))


def context_kimi_primitives(latency: dict[str, float], layers: int) -> tuple[float, float, float]:
    kda = sum(v for k, v in latency.items() if k.startswith("context_kda_")) / (layers - 1)
    mla = sum(
        v for k, v in latency.items()
        if k.startswith("context_mla_") or k.startswith("context_bmm_") or k == "context_attention"
    )
    attn_res = latency.get("context_attn_res", 0.0) / layers
    return kda, mla, attn_res


def make_dsa_models(source_model: str) -> dict[tuple[int, bool], str]:
    """Build native top-2048 shared/full refs; other top-k values use SOL ratios."""
    source = json.loads((Path(source_model) / "config.json").read_text(encoding="utf-8"))
    models: dict[tuple[int, bool], str] = {}
    for topk in (2048,):
        for shared in (True, False):
            cfg = json.loads(json.dumps(source))
            cfg["index_topk"] = topk
            if not shared:
                cfg["indexer_types"] = ["full"] * int(cfg["num_hidden_layers"])
                cfg["index_topk_freq"] = 1
            model_dir = HERE / "models" / f"dsa_top{topk}_{'shared' if shared else 'per_layer'}"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text(
                json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            models[(topk, shared)] = str(model_dir)
    return models


def dsa_context_formula(system: str, chunk: int, cpr: int, topk: int, skip: bool) -> float:
    """GLM-5.2 ContextDSAModule SOL formula, exposed for arbitrary top-k.

    AIC's model constructor currently fixes ContextDSAModule's top-k to the
    architecture default (2048), even when a synthetic config changes the JSON
    field.  We therefore retain AIC's direct top-2048 latency and apply the exact
    operation-level SOL ratio from ContextDSAModule.get_sol for other top-k values.
    """
    h, q, kv, nope, rope, v = 6144, 2048, 512, 192, 64, 256
    heads, index_heads, index_dim = 64, 32, 128
    full_s = PREFIX + chunk
    tokens = cpr * chunk
    proj_out = q + kv + rope + index_dim
    gemm_ops = (
        2 * tokens * h * proj_out
        + 2 * tokens * q * heads * (nope + rope)
        + 2 * tokens * q * index_heads * index_dim
        + 2 * tokens * h * index_heads
        + 2 * tokens * (heads * v) * h
        + 2 * heads * tokens * nope * kv
        + 2 * heads * tokens * kv * v
    )
    index_ops = 0 if skip else 2 * tokens * index_heads * index_dim * full_s
    pairs = tokens * min(full_s, topk)
    sparse_ops = 2 * heads * ((kv + rope) + kv) * pairs
    weight_bytes = h * proj_out + q * heads * (nope + rope) + q * index_heads * index_dim + h * index_heads + heads * v * h
    kv_bytes = cpr * heads * min(full_s, topk) * (kv + rope)
    index_bytes = 0 if skip else cpr * full_s * (index_dim + 4)
    q_io_bytes = tokens * heads * (nope + rope) * 2 * 2
    math_ms = (gemm_ops / 5e15 + index_ops / 5e15 + sparse_ops / 2.5e15) * 1000
    mem_ms = (weight_bytes + kv_bytes + index_bytes + q_io_bytes) / base.SYSTEMS[system]["mem_bw"] * 1000
    return max(math_ms, mem_ms)


def broad_primitives(
    spec: Any, ref_model: str, ds_ref_model: str, dsa_models: dict[tuple[int, bool], str],
    cache: dict[str, dict[str, Any]], system: str, chunk: int, cpr: int,
) -> dict[str, Any]:
    body_run = get(cache, spec.model_path, system, chunk, cpr)
    ref_run = get(cache, ref_model, system, chunk, cpr)
    ds_run = get(cache, ds_ref_model, system, chunk, cpr)
    dsa_runs = {key: get(cache, model, system, chunk, cpr) for key, model in dsa_models.items()}
    native_attn = (
        context_v4_attention(body_run["latency"])
        if spec.kind == "direct" else context_kimi_attention(body_run["latency"])
    )
    common_body = body_run["latency_ms"] - native_attn
    kda, mla, attn_res = context_kimi_primitives(ref_run["latency"], spec.depth)
    shared_run = dsa_runs[(2048, True)]
    full_run = dsa_runs[(2048, False)]
    full_fraction = 21 / 78
    norm = full_run["latency"].get("context_add_norm_1", 0.0) / 78
    full_2048 = full_run["latency"].get("context_attention", 0.0) / 78
    shared_2048 = shared_run["latency"].get("context_attention", 0.0) / 78
    skip_2048 = (shared_2048 - full_fraction * full_2048) / (1 - full_fraction)
    dsa: dict[tuple[int, bool], float] = {}
    for topk in (512, 1024, 2048, 4096):
        full_ratio = dsa_context_formula(system, chunk, cpr, topk, False) / dsa_context_formula(system, chunk, cpr, 2048, False)
        skip_ratio = dsa_context_formula(system, chunk, cpr, topk, True) / dsa_context_formula(system, chunk, cpr, 2048, True)
        full_value = full_2048 * full_ratio
        skip_value = skip_2048 * skip_ratio
        dsa[(topk, True)] = norm + full_fraction * full_value + (1 - full_fraction) * skip_value
        if topk == 2048:
            dsa[(topk, False)] = norm + full_value
    native_w, native_kv = base.body_native_weight_cache(spec, PREFIX + chunk)
    dispatch = body_run["dispatch_ms"]
    moe_compute = body_run["moe_compute_ms"]
    return {
        "body_ms": common_body, "dispatch_ms": dispatch, "moe_compute_ms": moe_compute,
        "body_other_ms": max(0.0, common_body - dispatch - moe_compute),
        "kda": kda + attn_res, "mla": mla + attn_res,
        "v4": context_v4_attention(ds_run["latency"]) / spec.depth,
        "dsa": dsa,
        "body_weight_bytes": body_run["weights_gib"] * GIB - native_w,
        "overhead_gib": body_run["activations_gib"] + body_run["nccl_gib"] + body_run["others_gib"],
        "non_sol_ms": body_run["non_sol_ms"] + ref_run["non_sol_ms"] + ds_run["non_sol_ms"]
        + sum(run["non_sol_ms"] for run in dsa_runs.values()),
    }


def broad_attention(profile_id: str, profile: dict[str, Any], prim: dict[str, Any], layers: int):
    counts = base.allocate_counts(profile["parts"], layers)
    value = counts.get("kda", 0) * prim["kda"] + counts.get("mla", 0) * prim["mla"]
    value += counts.get("v4", 0) * prim["v4"]
    if counts.get("dsa"):
        dsa_key = (
            int(profile.get("dsa_topk", 2048)),
            bool(profile.get("dsa_shared_index", True)),
        )
        value += counts["dsa"] * prim["dsa"][dsa_key]
    context = None
    return value, counts, context


def broad_weight_cache(profile: dict[str, Any], counts: dict[str, int], layers: int, context: int):
    weights = counts.get("kda", 0) * base.kda_weights() + counts.get("mla", 0) * base.mla_weights()
    cache = counts.get("kda", 0) * base.kda_cache_per_layer() + counts.get("mla", 0) * base.mla_cache_per_layer(context)
    if counts.get("v4"):
        weights += counts["v4"] * base.v4_weights(layers) / layers
        cache += counts["v4"] * base.v4_cache(layers, context) / layers
    if counts.get("dsa"):
        weights += counts["dsa"] * base.dsa_weights_per_layer(layers, bool(profile.get("dsa_shared_index", True)))
        cache += counts["dsa"] * base.dsa_cache_per_layer(context)
    return weights, cache


def broad_rows(
    specs: list[Any], refs: dict[int, str], dsa_models: dict[tuple[int, bool], str],
    cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    direct_balanced = {
        depth: min(
            (s for s in specs if s.kind == "direct" and s.active_tier == "balanced" and s.depth == depth),
            key=lambda s: abs(s.expert_intermediate - 3072),
        )
        for depth in base.DEPTHS
    }
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for system in SYSTEMS:
            for chunk in CHUNKS:
                for cpr in CPRS:
                    prim = broad_primitives(
                        spec, refs[spec.depth], direct_balanced[spec.depth].model_path,
                        dsa_models, cache, system, chunk, cpr,
                    )
                    for profile_id, profile in base.ATTENTION_PROFILES.items():
                        attention_ms, counts, _ = broad_attention(profile_id, profile, prim, spec.depth)
                        latency = prim["body_ms"] + attention_ms
                        attn_w, cache_per_request = broad_weight_cache(
                            profile, counts, spec.depth, PREFIX + chunk
                        )
                        weight_gib = (prim["body_weight_bytes"] + attn_w) / GIB
                        kv_gib = cpr * cache_per_request / GIB
                        memory_gib = weight_gib + kv_gib + prim["overhead_gib"]
                        total_p, active_p = base.estimated_params(spec, attn_w)
                        aid = f"factorial__{spec.id}__{profile_id}"
                        row = common_fields(
                            aid, f"{spec.id} · {profile['label']}", "Controlled 5T factorial",
                            "AIC component composite", system, chunk, cpr, latency,
                        )
                        row.update({
                            "body_id": spec.id, "attention_id": profile_id,
                            "attention": profile["label"], "subcategory": spec.kind,
                            "total_t": total_p / 1e12, "active_b": active_p / 1e9,
                            "depth": spec.depth, "experts": spec.experts, "topk": spec.topk,
                            "shared_experts": spec.shared_experts, "risk": (
                                "aggressive" if "aggressive" in (spec.active_tier, profile["risk"]) or spec.depth == 48
                                else "conservative" if spec.active_tier == "conservative" and profile["risk"] == "conservative" and spec.depth >= 78
                                else "balanced"
                            ),
                            "weight_gib": weight_gib, "kv_gib": kv_gib,
                            "memory_gib": memory_gib, "capacity_gib": CAPACITY_GIB,
                            "oom": memory_gib > CAPACITY_GIB, "attention_ms": attention_ms,
                            "dispatch_ms": prim["dispatch_ms"], "moe_compute_ms": prim["moe_compute_ms"],
                            "body_other_ms": prim["body_other_ms"], "non_sol_ms": prim["non_sol_ms"],
                        })
                        rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def swa_rows() -> list[dict[str, Any]]:
    raw_cache = {
        row["cache_key"]: row
        for row in (
            json.loads(line)
            for line in (OLD_PREFILL_DIR / "output/saturation_estimate_cache.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    main = read_csv(OLD_PREFILL_DIR / "output/all_results.csv")
    memory = {
        (r["architecture_id"], r["system"], int(r["cpr"])): r
        for r in main if int(r["new_tokens"]) == 1024
    }
    paths, v4_model, kimi_model = swa.make_models()

    def old_key(model: str, system: str, chunk: int, cpr: int) -> str:
        return json.dumps(
            [model, signature(model), system, PREFIX, chunk, DEP, cpr], separators=(",", ":")
        )

    def old_get(model: str, system: str, chunk: int, cpr: int) -> dict[str, Any]:
        return raw_cache[old_key(model, system, chunk, cpr)]

    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for chunk in CHUNKS:
            for cpr in CPRS:
                refs = {rid: old_get(path, system, chunk, cpr) for rid, path in paths.items()}
                common = refs["common_global"]
                common_attention = sum(
                    value for name, value in common["latency"].items()
                    if name == "context_attention" or name.endswith("_attn_norm")
                    or name.endswith("_qkv_gemm") or name.endswith("_proj_gemm")
                )
                body = sum(common["latency"].values()) - common_attention
                gqa = {
                    rid: sum(
                        value for name, value in run["latency"].items()
                        if name == "context_attention" or name.endswith("_attn_norm")
                        or name.endswith("_qkv_gemm") or name.endswith("_proj_gemm")
                    ) / swa.LAYERS
                    for rid, run in refs.items()
                }
                vr = old_get(v4_model, system, chunk, cpr)
                kr = old_get(kimi_model, system, chunk, cpr)
                dr = old_get(str(BASE_DIR / "models/glm52_fp8_dsa_validation"), system, chunk, cpr)
                kda, mla, attn_res = context_kimi_primitives(kr["latency"], swa.LAYERS)
                primitives = {
                    "gqa": gqa, "v4": context_v4_attention(vr["latency"]) / swa.LAYERS,
                    "kda": kda + attn_res, "mla": mla + attn_res,
                    "dsa": (
                        dr["latency"].get("context_attention", 0.0)
                        + dr["latency"].get("context_add_norm_1", 0.0)
                    ) / 78,
                }
                dispatch = common["latency"].get("context_global_moe_pre_dispatch", 0.0) + common["latency"].get("context_global_moe_post_dispatch", 0.0)
                moe_compute = common["latency"].get("context_global_moe", 0.0)
                body_other = max(0.0, body - dispatch - moe_compute)
                source_rows = list(refs.values()) + [vr, kr, dr]
                non_sol = sum(
                    sum(v for k, v in source["latency"].items() if v > 0 and source["sources"].get(k) != "sol")
                    for source in source_rows
                )
                for profile in swa.PROFILES:
                    attention = 0.0
                    if profile.get("local_ref"):
                        attention += profile["local"] * gqa[profile["local_ref"]]
                    mechanism = profile["global_mechanism"]
                    attention += profile["global"] * (
                        gqa[profile["global_ref"]] if mechanism == "gqa" else primitives[mechanism]
                    )
                    latency = body + attention
                    row = common_fields(
                        "swa__" + profile["id"], profile["label"], "SWA and hybrid extensions",
                        "AIC component composite", system, chunk, cpr, latency,
                    )
                    mem = memory.get((profile["id"], system, cpr)) if chunk == 1024 else None
                    row.update({
                        "body_id": "direct_L61_balanced_m3072_s1", "attention_id": profile["id"],
                        "attention": profile["label"], "subcategory": profile["category"],
                        "total_t": 5.0, "active_b": 125.9, "depth": 61,
                        "experts": 1216, "topk": 25, "shared_experts": 0, "risk": profile["risk"],
                        "weight_gib": float(mem["weight_gib"]) if mem else None,
                        "kv_gib": float(mem["kv_gib"]) if mem else None,
                        "memory_gib": float(mem["memory_gib"]) if mem else None,
                        "capacity_gib": CAPACITY_GIB,
                        "oom": (mem["oom"].lower() == "true") if mem else None,
                        "attention_ms": attention, "dispatch_ms": dispatch,
                        "moe_compute_ms": moe_compute, "body_other_ms": body_other,
                        "non_sol_ms": non_sol,
                    })
                    rows.append(row)
    return rows


def max_memory_cpr(points: list[dict[str, Any]]) -> int | None:
    measured = {int(r["cpr"]): r for r in points if r.get("memory_gib") is not None}
    if 1 not in measured or 2 not in measured:
        return None
    m1, m2 = float(measured[1]["memory_gib"]), float(measured[2]["memory_gib"])
    slope = m2 - m1
    fixed = m1 - slope
    if m1 > CAPACITY_GIB:
        return 0
    if slope <= 0:
        return None
    return max(0, math.floor((CAPACITY_GIB - fixed) / slope + 1e-9))


def dominant_resource(row: dict[str, Any]) -> str:
    choices = {
        "attention/cache": float(row["attention_ms"]),
        "MoE dispatch/fabric": float(row["dispatch_ms"]),
        "MoE compute/weights": float(row["moe_compute_ms"]),
        "other body": float(row["body_other_ms"]),
    }
    return max(choices, key=choices.get)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["architecture_id"], row["system"]), []).append(row)
    out: list[dict[str, Any]] = []
    for aid in sorted({r["architecture_id"] for r in rows}):
        sample = next(r for r in rows if r["architecture_id"] == aid)
        summary = {
            key: sample.get(key) for key in (
                "architecture_id", "label", "family", "study_kind", "subcategory", "body_id",
                "attention_id", "attention", "total_t", "active_b", "depth", "experts", "topk",
                "shared_experts", "risk",
            )
        }
        for system, prefix in (("GB300", "gb"), ("VR200 HBM+NVLink", "rubin")):
            points = groups[(aid, system)]
            c1 = sorted((r for r in points if r["cpr"] == 1), key=lambda r: r["new_tokens_per_request"])
            all_roof = max(r["rack_input_tps"] for r in points)
            for threshold in (0.90, 0.95, 0.99):
                hit = next((r for r in c1 if r["rack_input_tps"] >= threshold * all_roof), None)
                summary[f"{prefix}_chunk_c1_at_{int(threshold*100)}pct"] = hit["new_tokens_per_request"] if hit else None
                path = sorted(
                    (r for r in points if r["new_tokens_per_request"] == 1024), key=lambda r: r["cpr"]
                )
                hit = next((r for r in path if r["rack_input_tps"] >= threshold * all_roof), None)
                summary[f"{prefix}_cpr_1k_at_{int(threshold*100)}pct"] = hit["cpr"] if hit else None
            p1 = next(r for r in points if r["new_tokens_per_request"] == 1024 and r["cpr"] == 1)
            summary.update({
                f"{prefix}_1k_c1_latency_ms": p1["latency_ms"],
                f"{prefix}_1k_c1_input_tps_user": p1["input_tps_user"],
                f"{prefix}_1k_c1_rack_input_tps": p1["rack_input_tps"],
                f"{prefix}_1k_c1_dominant_resource": dominant_resource(p1),
                f"{prefix}_1k_c1_attention_share_pct": 100 * float(p1["attention_ms"]) / float(p1["latency_ms"]),
                f"{prefix}_1k_c1_dispatch_share_pct": 100 * float(p1["dispatch_ms"]) / float(p1["latency_ms"]),
                f"{prefix}_1k_c1_moe_compute_share_pct": 100 * float(p1["moe_compute_ms"]) / float(p1["latency_ms"]),
                f"{prefix}_memory_max_cpr": max_memory_cpr(
                    [r for r in points if r["new_tokens_per_request"] == 1024]
                ),
                f"{prefix}_observed_roof_rack_input_tps": all_roof,
            })
        summary["rubin_latency_gain_1k_c1"] = (
            summary["gb_1k_c1_latency_ms"] / summary["rubin_1k_c1_latency_ms"]
        )
        out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    specs = base.body_specs()
    refs = base.make_models(specs)
    anchors = base.make_native_anchors()
    dsa_model = base.make_dsa_validation_model()
    dsa_models = make_dsa_models(dsa_model)
    exact = exact_models(anchors["glm52_ratio"]["model_path"])

    direct_balanced = {
        depth: min(
            (s for s in specs if s.kind == "direct" and s.active_tier == "balanced" and s.depth == depth),
            key=lambda s: abs(s.expert_intermediate - 3072),
        ) for depth in base.DEPTHS
    }
    models = [m["model"] for m in exact] + [s.model_path for s in specs]
    models += list(refs.values()) + [s.model_path for s in direct_balanced.values()] + list(dsa_models.values())
    cache = run_estimates(models)

    rows = exact_rows(exact, cache)
    rows += broad_rows(specs, refs, dsa_models, cache)
    rows += swa_rows()
    summaries = summarize(rows)
    write_csv(OUTPUT / "all_results.csv", rows)
    write_csv(OUTPUT / "architecture_summary.csv", summaries)
    metadata = {
        "scope": "Unified cached-prefix prefill sweep over all prior architecture families",
        "workload": {"cached_prefix_tokens": PREFIX, "chunk_grid": CHUNKS, "postfill_context": "prefix+chunk"},
        "parallelism": "DEP64, TP1, PP1, attention-DP64, MoE-EP64",
        "systems": SYSTEMS, "capacity_gib": CAPACITY_GIB,
        "cpr_grid": CPRS, "database_mode": "SOL", "backend": "SGLang estimate",
        "counts": {
            "exact_configs": len(exact), "factorial_bodies": len(specs),
            "factorial_attention_profiles": len(base.ATTENTION_PROFILES),
            "factorial_architectures": len(specs) * len(base.ATTENTION_PROFILES),
            "swa_hybrids": len(swa.PROFILES), "total_architectures": len(summaries),
            "result_rows": len(rows),
        },
        "quantization": {"gemm": "fp8_block", "moe": "NVFP4 W4A8", "kv": "FP8", "fmha": "BF16"},
        "important_method_note": (
            "Native/scaled rows are direct AIC configs. Factorial and SWA rows are additive SOL component "
            "composites using prefix-aware static_ctx primitives; they are controlled architecture ablations."
        ),
        "saturation_definition": (
            "Observed roof is the maximum rack input-token throughput across chunk={1..4096 powers of two} "
            "and concurrency/rank={1..64 powers of two}; thresholds report first point on the named one-dimensional path."
        ),
        "excluded": ["cache lookup/transfer", "scheduler/queueing", "prefill-decode interference", "framework efficiency discount", "speculative decoding"],
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["counts"], indent=2))
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
