#!/usr/bin/env python3
"""Prefix-cache extend/prefill study for the controlled 5T SWA design space.

The workload is exact: each request begins with 262,144 cached tokens and
processes a 1,024-token context chunk in one AIC ``static_ctx`` forward pass.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DECODE_STUDY = ROOT / "experiments/5t-swa-architecture-study"
BASE_STUDY = ROOT / "experiments/5t-architecture-design-study"
sys.path.insert(0, str(DECODE_STUDY))
sys.path.insert(0, str(BASE_STUDY))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aic-5t-swa-prefill-mpl")

import run_study as base  # noqa: E402
import run_swa_study as decode  # noqa: E402
from aiconfigurator.cli.api import cli_estimate  # noqa: E402


OUTPUT = HERE / "output"
CACHE = OUTPUT / "prefill_estimate_cache.jsonl"
PREFIX_TOKENS = 262_144
NEW_TOKENS = 1_024
TOTAL_CONTEXT = PREFIX_TOKENS + NEW_TOKENS
DEP = 64
SYSTEMS = ("GB300", "VR200 HBM+NVLink")
MAX_CPR = 1_024
LATENCY_TARGETS_MS = (250, 500, 1_000, 2_000)
DSA_MODEL = BASE_STUDY / "models/glm52_fp8_dsa_validation"
DSA_LAYERS = 78


def config_signature(model: str) -> str:
    path = Path(model) / "config.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def cache_key(model: str, system: str, cpr: int) -> str:
    return json.dumps(
        [model, config_signature(model), system, PREFIX_TOKENS, NEW_TOKENS, DEP, cpr],
        separators=(",", ":"),
    )


def load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE.exists():
        return {}
    return {
        row["cache_key"]: row
        for row in (
            json.loads(line)
            for line in CACHE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def estimate(model: str, system: str, cpr: int) -> dict[str, Any]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = cli_estimate(
            model,
            base.SYSTEMS[system]["id"],
            mode="static_ctx",
            backend_name="sglang",
            backend_version="estimate",
            database_mode="SOL",
            isl=TOTAL_CONTEXT,
            osl=1,
            prefix=PREFIX_TOKENS,
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
    sources = {str(k): str(v or "unknown").lower() for k, v in result.summary.get_context_source_dict().items()}
    non_sol_ms = sum(v for k, v in latency.items() if v > 0 and sources.get(k) != "sol")
    memory = result.summary.get_memory()
    return {
        "cache_key": cache_key(model, system, cpr),
        "model": model,
        "system": system,
        "prefix_tokens": PREFIX_TOKENS,
        "new_tokens": NEW_TOKENS,
        "total_context": TOTAL_CONTEXT,
        "dep": DEP,
        "cpr": cpr,
        "context_latency_ms": sum(latency.values()),
        "latency": latency,
        "sources": sources,
        "non_sol_ms": non_sol_ms,
        "weights_gib": float(memory["weights"]),
        "kvcache_gib": float(memory["kvcache"]),
        "activations_gib": float(memory["activations"]),
        "nccl_gib": float(memory["nccl"]),
        "others_gib": float(memory["others"]),
        "memory_gib": float(memory["total"]),
        "oom": bool(result.summary.check_oom()),
        "modes": {k: result.raw.get(k) for k in ("gemm", "moe", "fmha", "kvcache")},
    }


def run_estimates(models: list[str], cprs: list[int]) -> dict[str, dict[str, Any]]:
    cache = load_cache()
    tasks = [(model, system, cpr) for model in models for system in SYSTEMS for cpr in cprs]
    pending = [task for task in tasks if cache_key(*task) not in cache]
    print(f"AIC static_ctx estimates: {len(tasks) - len(pending)} cached, {len(pending)} pending", flush=True)
    if pending:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as stream:
            for index, task in enumerate(pending, 1):
                row = estimate(*task)
                cache[row["cache_key"]] = row
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                if index % 200 == 0 or index == len(pending):
                    print(f"  completed {index}/{len(pending)}", flush=True)
    return cache


def get_run(cache: dict[str, dict[str, Any]], model: str, system: str, cpr: int) -> dict[str, Any]:
    return cache[cache_key(model, system, cpr)]


def gqa_attention(latency: dict[str, float]) -> float:
    return sum(
        value
        for name, value in latency.items()
        if name == "context_attention"
        or name.endswith("_attn_norm")
        or name.endswith("_qkv_gemm")
        or name.endswith("_proj_gemm")
    )


def v4_attention(latency: dict[str, float]) -> float:
    names = {"context_attention", "context_mhc_pre", "context_mhc_post", "context_attn_norm"}
    return sum(latency.get(name, 0.0) for name in names)


def kimi_attention_primitives(latency: dict[str, float]) -> tuple[float, float, float]:
    kda = sum(value for name, value in latency.items() if name.startswith("context_kda_")) / (decode.LAYERS - 1)
    mla = sum(
        value
        for name, value in latency.items()
        if name.startswith("context_mla_") or name == "context_attention"
    )
    attn_res = latency.get("context_attn_res", 0.0) / decode.LAYERS
    return kda, mla, attn_res


def primitives(
    cache: dict[str, dict[str, Any]],
    paths: dict[str, str],
    v4_model: str,
    kimi_model: str,
    system: str,
    cpr: int,
) -> dict[str, Any]:
    ref_runs = {rid: get_run(cache, path, system, cpr) for rid, path in paths.items()}
    common = ref_runs["common_global"]
    common_attention = gqa_attention(common["latency"])
    body = common["context_latency_ms"] - common_attention
    gqa = {rid: gqa_attention(run["latency"]) / decode.LAYERS for rid, run in ref_runs.items()}
    v4_run = get_run(cache, v4_model, system, cpr)
    kimi_run = get_run(cache, kimi_model, system, cpr)
    dsa_run = get_run(cache, str(DSA_MODEL), system, cpr)
    kda, mla, attn_res = kimi_attention_primitives(kimi_run["latency"])
    dsa = (
        dsa_run["latency"].get("context_attention", 0.0)
        + dsa_run["latency"].get("context_add_norm_1", 0.0)
    ) / DSA_LAYERS
    source_runs = list(ref_runs.values()) + [v4_run, kimi_run, dsa_run]
    return {
        "body": body,
        "gqa": gqa,
        "v4": v4_attention(v4_run["latency"]) / decode.LAYERS,
        "mla": mla + attn_res,
        "kda": kda + attn_res,
        "dsa": dsa,
        "overhead_gib": common["activations_gib"] + common["nccl_gib"] + common["others_gib"],
        "body_weight_gib": common["weights_gib"]
        - decode.LAYERS * decode.gqa_weight_bytes(decode.REFS["common_global"]) / 2**30,
        "source_non_sol_ms": sum(run["non_sol_ms"] for run in source_runs),
    }


def profile_attention(profile: dict[str, Any], prim: dict[str, Any]) -> float:
    value = 0.0
    if profile.get("local_ref"):
        value += profile["local"] * prim["gqa"][profile["local_ref"]]
    mechanism = profile["global_mechanism"]
    if mechanism == "gqa":
        value += profile["global"] * prim["gqa"][profile["global_ref"]]
    else:
        value += profile["global"] * prim[mechanism]
    return value


def compose_point(
    profile: dict[str, Any], prim: dict[str, Any], system: str, cpr: int
) -> dict[str, Any]:
    attention_ms = profile_attention(profile, prim)
    latency_ms = prim["body"] + attention_ms
    attention_weight_bytes, cache_per_request = decode.profile_weight_cache(profile, TOTAL_CONTEXT)
    weight_gib = prim["body_weight_gib"] + attention_weight_bytes / 2**30
    kv_gib = cpr * cache_per_request / 2**30
    memory_gib = weight_gib + kv_gib + prim["overhead_gib"]
    global_concurrency = DEP * cpr
    return {
        "architecture_id": profile["id"],
        "label": profile["label"],
        "category": profile["category"],
        "risk": profile["risk"],
        "system": system,
        "prefix_tokens": PREFIX_TOKENS,
        "new_tokens": NEW_TOKENS,
        "total_context": TOTAL_CONTEXT,
        "dep": DEP,
        "cpr": cpr,
        "global_concurrency": global_concurrency,
        "local_layers": profile["local"],
        "global_layers": profile["global"],
        "window": profile["window"],
        "global_mechanism": profile["global_mechanism"],
        "extend_latency_ms": latency_ms,
        "input_tps_user": NEW_TOKENS * 1000 / latency_ms,
        "input_tps_gpu": cpr * NEW_TOKENS * 1000 / latency_ms,
        "rack_input_tps": global_concurrency * NEW_TOKENS * 1000 / latency_ms,
        "rack_request_rate": global_concurrency * 1000 / latency_ms,
        "body_ms": prim["body"],
        "attention_ms": attention_ms,
        "attention_share_pct": 100 * attention_ms / latency_ms,
        "weight_gib": weight_gib,
        "kv_gib": kv_gib,
        "kv_gib_per_request": cache_per_request / 2**30,
        "overhead_gib": prim["overhead_gib"],
        "memory_gib": memory_gib,
        "capacity_gib": decode.CAPACITY_GIB,
        "oom": memory_gib > decode.CAPACITY_GIB,
        "source_non_sol_ms": prim["source_non_sol_ms"],
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for profile in decode.PROFILES:
        aid = profile["id"]
        summary: dict[str, Any] = {
            "architecture_id": aid,
            "label": profile["label"],
            "category": profile["category"],
            "risk": profile["risk"],
            "local_layers": profile["local"],
            "global_layers": profile["global"],
            "window": profile["window"],
            "global_mechanism": profile["global_mechanism"],
        }
        for system, prefix in (("GB300", "gb"), ("VR200 HBM+NVLink", "rubin")):
            system_rows = [r for r in rows if r["architecture_id"] == aid and r["system"] == system]
            feasible = [r for r in system_rows if not r["oom"]]
            for cpr in (1, 2, 4, 8, 16, 64):
                point = next(r for r in system_rows if r["cpr"] == cpr)
                summary[f"{prefix}_c{cpr}_latency_ms"] = point["extend_latency_ms"]
                summary[f"{prefix}_c{cpr}_input_tps_user"] = point["input_tps_user"]
                summary[f"{prefix}_c{cpr}_rack_input_tps"] = point["rack_input_tps"]
            summary[f"{prefix}_physical_max_cpr"] = max((r["cpr"] for r in feasible), default=0)
            for target in LATENCY_TARGETS_MS:
                passing = [r for r in feasible if r["extend_latency_ms"] <= target]
                summary[f"{prefix}_max_cpr_{target}ms"] = max((r["cpr"] for r in passing), default=0)
            if feasible:
                edge = max(feasible, key=lambda r: r["cpr"])
                best = max(feasible, key=lambda r: r["rack_input_tps"])
                summary[f"{prefix}_latency_at_physical_max_ms"] = edge["extend_latency_ms"]
                summary[f"{prefix}_rack_input_tps_at_physical_max"] = edge["rack_input_tps"]
                summary[f"{prefix}_best_throughput_cpr"] = best["cpr"]
                summary[f"{prefix}_best_rack_input_tps"] = best["rack_input_tps"]
        gb1 = next(r for r in rows if r["architecture_id"] == aid and r["system"] == "GB300" and r["cpr"] == 1)
        summary["kv_gib_per_request"] = gb1["kv_gib_per_request"]
        summary["weight_gib_per_rank"] = gb1["weight_gib"]
        summary["attention_share_gb_c1"] = gb1["attention_share_pct"]
        summary["rubin_latency_gain_c1"] = summary["gb_c1_latency_ms"] / summary["rubin_c1_latency_ms"]
        summary["rubin_latency_gain_c4"] = summary["gb_c4_latency_ms"] / summary["rubin_c4_latency_ms"]
        result.append(summary)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths, v4_model, kimi_model = decode.make_models()
    models = list(paths.values()) + [v4_model, kimi_model, str(DSA_MODEL)]
    cprs = list(range(1, MAX_CPR + 1))
    cache = run_estimates(models, cprs)
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for cpr in cprs:
            prim = primitives(cache, paths, v4_model, kimi_model, system, cpr)
            rows.extend(compose_point(profile, prim, system, cpr) for profile in decode.PROFILES)
    summaries = summarize(rows)
    write_csv(OUTPUT / "all_results.csv", rows)
    write_csv(OUTPUT / "architecture_summary.csv", summaries)
    metadata = {
        "scope": "Controlled 5T SWA/global cached-prefix extend-prefill study",
        "workload": {
            "prefix_tokens_already_cached": PREFIX_TOKENS,
            "new_input_tokens": NEW_TOKENS,
            "postfill_context_tokens": TOTAL_CONTEXT,
            "aic_mode": "static_ctx",
        },
        "layers": decode.LAYERS,
        "hidden_size": decode.HIDDEN,
        "experts": decode.EXPERTS,
        "topk": decode.TOPK,
        "shared_experts": 0,
        "moe_intermediate": decode.MOE_INTER,
        "dep": DEP,
        "systems": SYSTEMS,
        "capacity_gib": decode.CAPACITY_GIB,
        "profile_count": len(decode.PROFILES),
        "cpr_range": [1, MAX_CPR],
        "latency_targets_ms": LATENCY_TARGETS_MS,
        "aic_database_mode": "SOL",
        "quantization": {
            "gemm": "fp8_block",
            "moe": "w4a8_mxfp4_mxfp8_trtllm",
            "kv_cache": "fp8",
            "fmha": "bfloat16",
        },
        "throughput_definition": "new input tokens completed per synchronized static_ctx forward pass",
        "memory_method": "Explicit postfill per-layer FP8 KV/state plus AIC body weights and activation/other overhead",
        "hybrid_methods": {
            "gqa_swa": "AIC prefix-aware ContextAttention per-layer primitives",
            "mla_kda": "AIC Kimi-K3 prefix-aware context primitives",
            "v4": "AIC DeepSeek-V4 prefix-aware context primitive",
            "dsa": "AIC GLM-5.2 prefix-aware ContextDSAModule averaged across its shared-index cadence",
        },
        "excluded": [
            "prefix lookup and cache-hit probability",
            "KV transfer or remote cache fetch",
            "scheduler and queueing",
            "framework efficiency discount",
            "prefill/decode interference",
            "speculative decoding (decode-only technique)",
        ],
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Profiles: {len(decode.PROFILES)}")
    print(f"Rows: {len(rows)}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
