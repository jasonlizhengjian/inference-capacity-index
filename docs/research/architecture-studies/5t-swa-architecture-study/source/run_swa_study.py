#!/usr/bin/env python3
"""Controlled 5T sliding-window-attention study using AIC SOL primitives."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_STUDY = ROOT / "experiments/5t-architecture-design-study"
sys.path.insert(0, str(BASE_STUDY))
import run_study as base  # noqa: E402


MODELS = HERE / "models"
OUTPUT = HERE / "output"
CACHE = OUTPUT / "estimate_cache.jsonl"
LAYERS = 61
HIDDEN = 7168
VOCAB = 129_280
EXPERTS = 1216
TOPK = 25
MOE_INTER = 3072
CAPACITY_GIB = base.SYSTEMS["GB300"]["capacity"] / 2**30
SYSTEMS = ("GB300", "VR200 HBM+NVLink")
CONTEXTS = (32_768, 262_144, 1_048_576)
WINDOWS = (128, 512, 2048, 4096)
RATIOS = (1, 2, 3, 4, 5, 7)
SPEC_CPRS = (1, 4, 16, 64, 128)


@dataclass(frozen=True)
class RefSpec:
    id: str
    q_heads: int
    global_kv_heads: int
    local_kv_heads: int
    head_dim: int
    value_dim: int
    window: int
    local: bool


REFS = {
    "common_global": RefSpec("common_global", 56, 8, 8, 128, 128, 512, False),
    "common_local_w128": RefSpec("common_local_w128", 56, 8, 8, 128, 128, 128, True),
    "common_local_w512": RefSpec("common_local_w512", 56, 8, 8, 128, 128, 512, True),
    "common_local_w2048": RefSpec("common_local_w2048", 56, 8, 8, 128, 128, 2048, True),
    "common_local_w4096": RefSpec("common_local_w4096", 56, 8, 8, 128, 128, 4096, True),
    "gptoss_global": RefSpec("gptoss_global", 112, 8, 8, 64, 64, 128, False),
    "gptoss_local": RefSpec("gptoss_local", 112, 8, 8, 64, 64, 128, True),
    "inkling_global": RefSpec("inkling_global", 56, 8, 16, 128, 128, 512, False),
    "inkling_local": RefSpec("inkling_local", 56, 8, 16, 128, 128, 512, True),
}


def ratio_counts(local_per_global: int) -> tuple[int, int]:
    global_layers = max(1, round(LAYERS / (local_per_global + 1)))
    return LAYERS - global_layers, global_layers


def make_profiles() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = [{
        "id": "full_gqa", "label": "100% global GQA", "category": "control",
        "local": 0, "global": LAYERS, "local_ref": None, "global_ref": "common_global",
        "global_mechanism": "gqa", "window": 0, "risk": "conservative",
    }]
    for ratio in RATIOS:
        local, global_ = ratio_counts(ratio)
        for window in WINDOWS:
            profiles.append({
                "id": f"common_r{ratio}_w{window}",
                "label": f"{ratio}:1 SWA/global · W{window}", "category": "ratio-window grid",
                "local": local, "global": global_, "local_ref": f"common_local_w{window}",
                "global_ref": "common_global", "global_mechanism": "gqa", "window": window,
                "risk": "conservative" if ratio <= 2 else "balanced" if ratio <= 5 else "aggressive",
            })
    gpt_local, gpt_global = ratio_counts(1)
    profiles.append({
        "id": "gptoss_faithful", "label": "GPT-OSS-like 1:1 · W128", "category": "faithful-inspired",
        "local": gpt_local, "global": gpt_global, "local_ref": "gptoss_local",
        "global_ref": "gptoss_global", "global_mechanism": "gqa", "window": 128,
        "risk": "conservative",
    })
    ink_local, ink_global = ratio_counts(5)
    profiles.append({
        "id": "inkling_faithful", "label": "Inkling-like 5:1 · W512", "category": "faithful-inspired",
        "local": ink_local, "global": ink_global, "local_ref": "inkling_local",
        "global_ref": "inkling_global", "global_mechanism": "gqa", "window": 512,
        "risk": "balanced",
    })
    for ratio in (1, 3, 5):
        local, global_ = ratio_counts(ratio)
        for mechanism, label in (
            ("mla", "MLA"),
            ("v4", "V4 compressed"),
            ("dsa", "DSA top-2048"),
            ("kda", "KDA linear"),
        ):
            profiles.append({
                "id": f"swa{ratio}_{mechanism}",
                "label": f"{ratio}:1 SWA/{label} · W512", "category": "cross-mechanism hybrid",
                "local": local, "global": global_, "local_ref": "common_local_w512",
                "global_ref": None, "global_mechanism": mechanism, "window": 512,
                "risk": (
                    "conservative" if mechanism == "mla" and ratio <= 3
                    else "aggressive" if mechanism == "kda" or ratio > 5
                    else "balanced"
                ),
            })
    profiles.extend([
        {"id": "mla_all", "label": "100% MLA", "category": "mechanism control", "local": 0,
         "global": LAYERS, "local_ref": None, "global_ref": None, "global_mechanism": "mla",
         "window": 0, "risk": "conservative"},
        {"id": "v4_all", "label": "100% V4 compressed", "category": "mechanism control", "local": 0,
         "global": LAYERS, "local_ref": None, "global_ref": None, "global_mechanism": "v4",
         "window": 0, "risk": "conservative"},
        {"id": "kda_all", "label": "100% KDA", "category": "mechanism control", "local": 0,
         "global": LAYERS, "local_ref": None, "global_ref": None, "global_mechanism": "kda",
         "window": 0, "risk": "aggressive"},
        {"id": "dsa2048_all", "label": "100% DSA top-2048", "category": "mechanism control",
         "local": 0, "global": LAYERS, "local_ref": None, "global_ref": None,
         "global_mechanism": "dsa", "window": 2048, "risk": "balanced"},
    ])
    return profiles


PROFILES = make_profiles()


def model_path(ref: RefSpec) -> Path:
    return MODELS / ref.id


def make_models() -> tuple[dict[str, str], str, str]:
    MODELS.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for ref in REFS.values():
        config = {
            "architectures": ["MiMoV2FlashForCausalLM"], "model_type": "mimo_v2_flash",
            "num_hidden_layers": LAYERS, "hidden_size": HIDDEN,
            "num_attention_heads": ref.q_heads, "num_key_value_heads": ref.global_kv_heads,
            "head_dim": ref.head_dim, "v_head_dim": ref.value_dim,
            "intermediate_size": 4 * HIDDEN, "vocab_size": VOCAB,
            "max_position_embeddings": max(CONTEXTS), "sliding_window": ref.window,
            "sliding_window_size": ref.window, "attention_chunk_size": ref.window,
            "swa_num_attention_heads": ref.q_heads, "swa_num_key_value_heads": ref.local_kv_heads,
            "swa_head_dim": ref.head_dim, "swa_v_head_dim": ref.value_dim,
            "hybrid_layer_pattern": [0 if ref.local else 1] * LAYERS,
            "moe_layer_freq": [1] * LAYERS, "moe_intermediate_size": MOE_INTER,
            "n_routed_experts": EXPERTS, "n_shared_experts": None,
            "num_experts_per_tok": TOPK, "hidden_act": "silu", "tie_word_embeddings": False,
            "torch_dtype": "bfloat16", "use_cache": True,
            "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3",
                                    "quant_method": "fp8", "weight_block_size": [128, 128]},
        }
        path = model_path(ref)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        paths[ref.id] = str(path)

    specs = base.body_specs()
    attention_refs = base.make_models(specs)
    direct = next(s for s in specs if s.id == "direct_L61_balanced_m3072_s1")
    return paths, direct.model_path, attention_refs[LAYERS]


def load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE.exists():
        return {}
    return {row["cache_key"]: row for row in (
        json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines() if line.strip()
    )}


def run_estimates(
    models: list[str], points: list[tuple[str, int, int]], depths: dict[int, float],
) -> dict[str, dict[str, Any]]:
    """Run (system, context, c/r) at fixed DEP64, caching every AIC result."""
    cache = load_cache()
    tasks = [(m, system, context, 64, cpr, n, al)
             for m in models for system, context, cpr in points for n, al in depths.items()]
    pending = [task for task in tasks if base.cache_key(*task) not in cache]
    print(f"AIC estimates: {len(tasks)-len(pending)} cached, {len(pending)} pending", flush=True)
    if pending:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as stream:
            for index, task in enumerate(pending, 1):
                row = base.estimate(*task)
                cache[row["cache_key"]] = row
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                if index % 250 == 0 or index == len(pending):
                    print(f"  completed {index}/{len(pending)}", flush=True)
    return cache


def curve_cprs() -> list[int]:
    # Dense integer coverage makes the capacity and 50/100-TPS boundaries exact
    # within the studied c/r<=1024 range instead of sampled approximations.
    return list(range(1, 1025))


def hybrid_attention(latency: dict[str, float]) -> float:
    return sum(value for name, value in latency.items() if (
        name == "generation_attention" or name.endswith("_attn_norm")
        or name.endswith("_qkv_gemm") or name.endswith("_proj_gemm")
    ))


def gqa_weight_bytes(ref: RefSpec) -> float:
    kv = ref.local_kv_heads if ref.local else ref.global_kv_heads
    return float(HIDDEN * (ref.q_heads * ref.head_dim + kv * (ref.head_dim + ref.value_dim))
                 + HIDDEN * ref.q_heads * ref.value_dim)


def gqa_cache_bytes(ref: RefSpec, context: int) -> float:
    kv = ref.local_kv_heads if ref.local else ref.global_kv_heads
    tokens = min(context, ref.window) if ref.local else context
    return float(tokens * kv * (ref.head_dim + ref.value_dim))  # FP8 KV: one byte/element


def get_run(cache: dict[str, dict[str, Any]], model: str, system: str, context: int,
            cpr: int, nextn: int = 0, al: float = 1.0) -> dict[str, Any]:
    return cache[base.cache_key(model, system, context, 64, cpr, nextn, al)]


def primitives(cache: dict[str, dict[str, Any]], paths: dict[str, str], v4_model: str,
               kimi_model: str, system: str, context: int, cpr: int,
               nextn: int = 0, al: float = 1.0) -> dict[str, Any]:
    ref_runs = {rid: get_run(cache, path, system, context, cpr, nextn, al)
                for rid, path in paths.items()}
    common = ref_runs["common_global"]
    common_attn = hybrid_attention(common["latency"])
    body = common["verification_step_ms"] - common_attn
    gqa = {rid: hybrid_attention(run["latency"]) / LAYERS for rid, run in ref_runs.items()}
    v4_run = get_run(cache, v4_model, system, context, cpr, nextn, al)
    kimi_run = get_run(cache, kimi_model, system, context, cpr, nextn, al)
    kda, mla, attn_res = base.kimi_primitives(kimi_run["latency"], LAYERS)
    return {
        "body": body, "gqa": gqa,
        "v4": base.v4_attention_value(v4_run["latency"]) / LAYERS,
        "mla": mla + attn_res, "kda": kda + attn_res,
        "overhead_gib": common["activations_gib"] + common["nccl_gib"] + common["others_gib"],
        "body_weight_gib": common["weights_gib"] - LAYERS * gqa_weight_bytes(REFS["common_global"]) / 2**30,
        "source_non_sol_ms": sum(run["non_sol_ms"] for run in ref_runs.values())
                             + v4_run["non_sol_ms"] + kimi_run["non_sol_ms"],
    }


def profile_attention(profile: dict[str, Any], prim: dict[str, Any], system: str,
                      context: int, cpr: int, spec_scale: float = 1.0) -> float:
    local = profile["local"]
    global_ = profile["global"]
    mechanism = profile["global_mechanism"]
    value = 0.0
    if profile.get("local_ref"):
        value += local * prim["gqa"][profile["local_ref"]]
    if mechanism == "gqa":
        value += global_ * prim["gqa"][profile["global_ref"]]
    elif mechanism in {"mla", "v4", "kda"}:
        value += global_ * prim[mechanism]
    elif mechanism == "dsa":
        # The shared indexer cadence is averaged across the full stack, then charged
        # only to the periodic DSA global layers in a hybrid profile.
        value += (global_ / LAYERS) * spec_scale * base.dsa_latency(
            system, context, cpr, LAYERS, 2048, True
        )
    else:
        raise ValueError(mechanism)
    return value


def profile_weight_cache(profile: dict[str, Any], context: int) -> tuple[float, float]:
    local, global_ = profile["local"], profile["global"]
    weights = cache = 0.0
    if profile.get("local_ref"):
        ref = REFS[profile["local_ref"]]
        weights += local * gqa_weight_bytes(ref)
        cache += local * gqa_cache_bytes(ref, context)
    mechanism = profile["global_mechanism"]
    if mechanism == "gqa":
        ref = REFS[profile["global_ref"]]
        weights += global_ * gqa_weight_bytes(ref)
        cache += global_ * gqa_cache_bytes(ref, context)
    elif mechanism == "mla":
        weights += global_ * base.mla_weights()
        cache += global_ * base.mla_cache_per_layer(context)
    elif mechanism == "v4":
        weights += global_ * base.v4_weights(LAYERS) / LAYERS
        cache += global_ * base.v4_cache(LAYERS, context) / LAYERS
    elif mechanism == "kda":
        weights += global_ * base.kda_weights()
        cache += global_ * base.kda_cache_per_layer()
    elif mechanism == "dsa":
        weights += global_ * base.dsa_weights_per_layer(LAYERS, True)
        cache += global_ * base.dsa_cache_per_layer(context)
    return weights, cache


def avg_span(profile: dict[str, Any], context: int) -> float | None:
    if profile["global_mechanism"] != "gqa" or not profile.get("local_ref"):
        return None
    return (profile["local"] * min(context, profile["window"]) + profile["global"] * context) / LAYERS


def compose_point(profile: dict[str, Any], prim: dict[str, Any], system: str,
                  context: int, cpr: int, nextn: int = 0, al: float = 1.0) -> dict[str, Any]:
    attention = profile_attention(profile, prim, system, context, cpr)
    step_ms = prim["body"] + attention
    weight_bytes, cache_per_request = profile_weight_cache(profile, context)
    weight_gib = prim["body_weight_gib"] + weight_bytes / 2**30
    kv_gib = cpr * cache_per_request / 2**30
    memory_gib = weight_gib + kv_gib + prim["overhead_gib"]
    tpot = step_ms / al
    return {
        "architecture_id": profile["id"], "label": profile["label"],
        "category": profile["category"], "risk": profile["risk"],
        "system": system, "context": context, "dep": 64, "cpr": cpr,
        "global_concurrency": 64 * cpr, "local_layers": profile["local"],
        "global_layers": profile["global"], "window": profile["window"],
        "global_mechanism": profile["global_mechanism"],
        "average_attended_tokens": avg_span(profile, context),
        "nextn": nextn, "acceptance_length": al, "verification_step_ms": step_ms,
        "tpot_ms": tpot, "tps_user": 1000 / tpot,
        "aggregate_tps_dep64": 64 * cpr * 1000 / tpot,
        "body_ms": prim["body"], "attention_ms": attention,
        "attention_share_pct": 100 * attention / step_ms,
        "weight_gib": weight_gib, "kv_gib": kv_gib,
        "kv_gib_per_request": cache_per_request / 2**30,
        "overhead_gib": prim["overhead_gib"], "memory_gib": memory_gib,
        "capacity_gib": CAPACITY_GIB, "oom": memory_gib > CAPACITY_GIB,
        "source_non_sol_ms": prim["source_non_sol_ms"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    return {(r["architecture_id"], r["system"], r["context"], r["cpr"]): r for r in rows}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = row_index(rows)
    out = []
    for profile in PROFILES:
        aid = profile["id"]
        sample = index[(aid, "GB300", 262_144, 1)]
        result: dict[str, Any] = {
            "architecture_id": aid, "label": profile["label"], "category": profile["category"],
            "risk": profile["risk"], "local_layers": profile["local"],
            "global_layers": profile["global"], "window": profile["window"],
            "global_mechanism": profile["global_mechanism"],
            "average_attended_tokens_256k": sample["average_attended_tokens"],
            "kv_gib_per_request_256k": sample["kv_gib_per_request"],
            "weight_gib_per_rank": sample["weight_gib"],
        }
        for system, prefix in (("GB300", "gb"), ("VR200 HBM+NVLink", "rubin")):
            system_rows = [r for r in rows if r["architecture_id"] == aid and r["system"] == system
                           and r["context"] == 262_144]
            feasible = [r for r in system_rows if not r["oom"]]
            result[f"{prefix}_physical_max_cpr"] = max((r["cpr"] for r in feasible), default=0)
            for target in (100, 50):
                passing = [r for r in feasible if r["tps_user"] >= target]
                result[f"{prefix}_max_cpr_{target}tps"] = max((r["cpr"] for r in passing), default=0)
            for cpr in (1, 4, 16, 64, 128):
                key = (aid, system, 262_144, cpr)
                if key in index:
                    result[f"{prefix}_c{cpr}_tps"] = index[key]["tps_user"]
            physical = result[f"{prefix}_physical_max_cpr"]
            if physical:
                edge = index[(aid, system, 262_144, physical)]
                result[f"{prefix}_tps_at_physical_max"] = edge["tps_user"]
                result[f"{prefix}_rack_tps_at_physical_max"] = edge["aggregate_tps_dep64"]
        for system, prefix in (("GB300", "gb"), ("VR200 HBM+NVLink", "rubin")):
            t32 = index[(aid, system, 32_768, 1)]["tps_user"]
            t1m = index[(aid, system, 1_048_576, 1)]["tps_user"]
            result[f"{prefix}_context_retention_1m_over_32k"] = t1m / t32
        out.append(result)
    return out


def best_spec(spec_rows: list[dict[str, Any]], no_spec_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in spec_rows + [r for r in no_spec_rows if r["architecture_id"] in SPEC_PROFILE_IDS
                            and r["context"] == 262_144 and r["cpr"] in SPEC_CPRS]:
        key = (row["architecture_id"], row["system"], row["context"], row["cpr"])
        candidates.setdefault(key, []).append(row)
    out = []
    for rows in candidates.values():
        fitting = [r for r in rows if not r["oom"]]
        if not fitting:
            continue
        chosen = max(fitting, key=lambda r: r["tps_user"])
        base_row = next(r for r in rows if r["nextn"] == 0)
        chosen = dict(chosen)
        chosen["no_spec_tps_user"] = base_row["tps_user"]
        chosen["spec_gain"] = chosen["tps_user"] / base_row["tps_user"]
        out.append(chosen)
    return sorted(out, key=lambda r: (r["architecture_id"], r["system"], r["cpr"]))


SPEC_PROFILE_IDS = {
    "gptoss_faithful", "common_r2_w512", "common_r3_w512", "common_r4_w512",
    "inkling_faithful", "swa5_mla", "swa5_v4", "v4_all",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths, v4_model, kimi_model = make_models()
    reference_models = list(paths.values()) + [v4_model, kimi_model]
    cprs = curve_cprs()
    no_spec_points = [(system, 262_144, cpr) for system in SYSTEMS for cpr in cprs]
    no_spec_points += [(system, context, cpr) for system in SYSTEMS
                       for context in (32_768, 1_048_576) for cpr in (1, 4)]
    no_spec_points = sorted(set(no_spec_points))
    cache = run_estimates(reference_models, no_spec_points, {0: 1.0})

    no_spec_rows: list[dict[str, Any]] = []
    for system, context, cpr in no_spec_points:
        prim = primitives(cache, paths, v4_model, kimi_model, system, context, cpr)
        for profile in PROFILES:
            no_spec_rows.append(compose_point(profile, prim, system, context, cpr))
    summaries = summarize(no_spec_rows)

    al_curve = base.load_dspark_al_curve()
    spec_points = [(system, 262_144, cpr) for system in SYSTEMS for cpr in SPEC_CPRS]
    cache = run_estimates(reference_models, spec_points, {n: al for n, al in al_curve.items() if n})
    spec_rows: list[dict[str, Any]] = []
    selected_profiles = [p for p in PROFILES if p["id"] in SPEC_PROFILE_IDS]
    for system, context, cpr in spec_points:
        for nextn, al in al_curve.items():
            if nextn == 0:
                continue
            prim = primitives(cache, paths, v4_model, kimi_model, system, context, cpr, nextn, al)
            for profile in selected_profiles:
                spec_rows.append(compose_point(profile, prim, system, context, cpr, nextn, al))
    best = best_spec(spec_rows, no_spec_rows)

    write_csv(OUTPUT / "all_results.csv", no_spec_rows)
    write_csv(OUTPUT / "architecture_summary.csv", summaries)
    write_csv(OUTPUT / "spec_results.csv", spec_rows)
    write_csv(OUTPUT / "spec_best.csv", best)
    metadata = {
        "scope": "Controlled 5T SWA/global decode-only study",
        "layers": LAYERS, "hidden_size": HIDDEN, "experts": EXPERTS, "topk": TOPK,
        "shared_experts": 0, "routed_expert_hidden_size": HIDDEN,
        "moe_intermediate": MOE_INTER,
        "parameters_per_routed_expert": 3 * HIDDEN * MOE_INTER,
        "routed_parameters_per_layer": EXPERTS * 3 * HIDDEN * MOE_INTER,
        "active_routed_parameters_per_layer": TOPK * 3 * HIDDEN * MOE_INTER,
        "total_routed_parameters": LAYERS * EXPERTS * 3 * HIDDEN * MOE_INTER,
        "active_routed_parameters_per_token": LAYERS * TOPK * 3 * HIDDEN * MOE_INTER,
        "dep": 64, "contexts": CONTEXTS,
        "systems": SYSTEMS, "capacity_gib": CAPACITY_GIB,
        "profile_count": len(PROFILES), "aic_database_mode": "SOL",
        "memory_method": "Explicit per-layer FP8 KV; local layers capped at window",
        "hybrid_methods": {
            "mla": "AIC-derived MLA per-layer latency and latent-KV accounting on global layers",
            "v4": "AIC-derived mean V4 attention-layer latency and compressed-cache accounting",
            "dsa": "Published SOL extension averaged over shared-index cadence, charged to global layers",
            "kda": "AIC-derived KDA per-layer latency and recurrent-state accounting on global layers",
        },
        "speculative_acceptance": al_curve,
        "official_sources": {
            "gpt_oss": "https://deploymentsafety.openai.com/gpt-oss/evaluation",
            "inkling": "https://thinkingmachines.ai/news/introducing-inkling/",
            "inkling_config": "https://huggingface.co/thinkingmachines/Inkling/blob/main/config.json",
        },
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Profiles: {len(PROFILES)}")
    print(f"No-spec rows: {len(no_spec_rows)}; spec rows: {len(spec_rows)}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
