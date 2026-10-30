#!/usr/bin/env python3
"""Controlled ~5T architecture design-space study using AIC SOL components."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import geometric_mean
from typing import Any

import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aic-5t-design-mpl")

from aiconfigurator.cli.api import cli_estimate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AIC_REPO = ROOT / "repos/aiconfigurator-kimi-k3-pr1435"
MODELS = HERE / "models"
OUTPUT = HERE / "output"
SYSTEMS_DIR = HERE / "systems"
SYSTEMS_PATHS = f"{SYSTEMS_DIR},default"
CACHE = OUTPUT / "estimate_cache.jsonl"

KIMI_SOURCE = ROOT / "experiments/kimi-k3-scale-up/models/k3_5t_expert/config.json"
DS_SOURCE = ROOT / "experiments/deepseek-v4-5t-sol/models/deepseek_v4_pro_5t_expert/config.json"
GLM_SOURCE = AIC_REPO / "aic-core/src/aiconfigurator_core/model_configs/nvidia--GLM-5.2-NVFP4_config.json"
INFERENCEX_REPO = ROOT.parent / "inference-x"
AL_SOURCE = INFERENCEX_REPO / "golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml"

NATIVE_ANCHOR_SOURCES = {
    "kimi_ratio": ROOT / "experiments/kimi-vs-deepseek-5t-controlled/models/kimi_ratio_preserved",
    "kimi_trend": ROOT / "experiments/kimi-k3-scale-up/models/k3_5t_trend",
    "deepseek_ratio": ROOT / "experiments/kimi-vs-deepseek-5t-controlled/models/deepseek_ratio_preserved",
}

H = 7168
VOCAB = 129_280
TARGET_ROUTED = 4.90e12
TARGET_ACTIVE = {"aggressive": 50e9, "balanced": 100e9, "conservative": 160e9}
DEPTHS = (48, 61, 78, 93, 116)
SYSTEMS = {
    "GB300": {"id": "gb300", "mem_bw": 8.0e12, "fabric": 0.9e12, "capacity": 298_013_687_808},
    "VR200 HBM": {"id": "vr200_hbm", "mem_bw": 22.0e12, "fabric": 0.9e12, "capacity": 298_013_687_808},
    "VR200 HBM+NVLink": {"id": "vr200_hbm_nvlink", "mem_bw": 22.0e12, "fabric": 1.8e12, "capacity": 298_013_687_808},
}
CONTEXTS = (32_768, 262_144, 1_048_576)
DEPS = (16, 32, 64)


def load_dspark_al_curve() -> dict[int, float]:
    payload = yaml.safe_load(AL_SOURCE.read_text(encoding="utf-8"))
    curve = {0: 1.0, **{int(k): float(v) for k, v in payload["kimi-k3"]["thinking_on"].items()}}
    if set(curve) != set(range(9)):
        raise ValueError(f"Expected DSpark depths 0..8, got {sorted(curve)}")
    return curve

ATTENTION_PROFILES = {
    "mla100": {"label": "100% full MLA", "parts": {"mla": 1.0}, "risk": "conservative"},
    "kda75_mla25": {"label": "75% KDA + 25% MLA", "parts": {"kda": 0.75, "mla": 0.25}, "risk": "conservative"},
    "kda50_mla50": {"label": "50% KDA + 50% MLA", "parts": {"kda": 0.50, "mla": 0.50}, "risk": "conservative"},
    "kda90_mla10": {"label": "90% KDA + 10% MLA", "parts": {"kda": 0.90, "mla": 0.10}, "risk": "balanced"},
    "kda100": {"label": "100% KDA", "parts": {"kda": 1.0}, "risk": "aggressive"},
    "dsa512": {"label": "100% DSA top-512", "parts": {"dsa": 1.0}, "dsa_topk": 512, "risk": "aggressive"},
    "dsa1024": {"label": "100% DSA top-1024", "parts": {"dsa": 1.0}, "dsa_topk": 1024, "risk": "balanced"},
    "dsa2048": {"label": "100% DSA top-2048", "parts": {"dsa": 1.0}, "dsa_topk": 2048, "risk": "conservative"},
    "dsa2048_noreuse": {"label": "100% DSA top-2048, per-layer index", "parts": {"dsa": 1.0}, "dsa_topk": 2048, "dsa_shared_index": False, "risk": "conservative"},
    "dsa4096": {"label": "100% DSA top-4096", "parts": {"dsa": 1.0}, "dsa_topk": 4096, "risk": "conservative"},
    "v4_compressed": {"label": "100% DeepSeek-V4 compressed", "parts": {"v4": 1.0}, "risk": "conservative"},
    "kda75_dsa25": {"label": "75% KDA + 25% DSA-2048", "parts": {"kda": 0.75, "dsa": 0.25}, "dsa_topk": 2048, "risk": "balanced"},
    "kda50_dsa50": {"label": "50% KDA + 50% DSA-2048", "parts": {"kda": 0.50, "dsa": 0.50}, "dsa_topk": 2048, "risk": "balanced"},
    "kda25_dsa75": {"label": "25% KDA + 75% DSA-2048", "parts": {"kda": 0.25, "dsa": 0.75}, "dsa_topk": 2048, "risk": "balanced"},
    "kda75_v4_25": {"label": "75% KDA + 25% V4 compressed", "parts": {"kda": 0.75, "v4": 0.25}, "risk": "balanced"},
    "kda50_v4_50": {"label": "50% KDA + 50% V4 compressed", "parts": {"kda": 0.50, "v4": 0.50}, "risk": "balanced"},
}


@dataclass(frozen=True)
class BodySpec:
    id: str
    kind: str
    depth: int
    active_tier: str
    expert_intermediate: int
    routed_hidden: int
    experts: int
    topk: int
    shared_experts: int
    routed_total: int
    routed_active: int
    model_path: str


def _round64(value: float) -> int:
    return max(64, int(round(value / 64)) * 64)


def _body(kind: str, depth: int, tier: str, inter: int = 3072, shared: int = 1) -> BodySpec:
    # At L93, a 64-expert granularity step is too coarse with inter=3072:
    # E768 undershoots and E832 overshoots the 5T gate. A near-native 2944
    # intermediate width lets E832 land at 4.90T routed capacity.
    if kind == "direct" and depth == 93 and inter == 3072:
        inter = 2944
    routed_h = H if kind == "direct" else H // 2
    per_expert = depth * 3 * routed_h * inter
    experts = _round64(TARGET_ROUTED / per_expert)
    topk = max(1, int(round(TARGET_ACTIVE[tier] / per_expert)))
    model_id = f"{kind}_L{depth}_{tier}_m{inter}_s{shared}"
    return BodySpec(
        id=model_id, kind=kind, depth=depth, active_tier=tier,
        expert_intermediate=inter, routed_hidden=routed_h,
        experts=experts, topk=topk, shared_experts=shared,
        routed_total=experts * per_expert, routed_active=topk * per_expert,
        model_path=str(MODELS / model_id),
    )


def body_specs() -> list[BodySpec]:
    specs = [_body("direct", depth, tier) for depth in DEPTHS for tier in TARGET_ACTIVE]
    specs += [_body("latent", depth, tier) for depth in (61, 78, 93) for tier in TARGET_ACTIVE]
    specs += [_body("direct", 78, "balanced", inter) for inter in (2048, 4096)]
    specs += [_body("latent", 78, "balanced", 3072, shared) for shared in (0, 2)]
    unique = {spec.id: spec for spec in specs}
    return list(unique.values())


def v4_ratios(layers: int) -> list[int]:
    ratios = [128 if i < 2 or i % 2 else 4 for i in range(layers)]
    ratios[-1] = 0
    return ratios


def kimi_pattern(layers: int, kda_fraction: float = 0.75) -> tuple[list[int], list[int]]:
    # Round .5 upward to match allocate_counts' stable largest-remainder rule.
    kda_count = int(math.floor(layers * kda_fraction + 0.5))
    # Evenly distribute the non-KDA layers and keep the final layer non-KDA.
    mla_count = layers - kda_count
    mla = {layers}
    if mla_count > 1:
        for i in range(1, mla_count):
            mla.add(max(1, int(round(i * layers / mla_count))))
    while len(mla) > mla_count:
        mla.remove(min(mla))
    while len(mla) < mla_count:
        mla.add(next(i for i in range(1, layers + 1) if i not in mla))
    kda = [i for i in range(1, layers + 1) if i not in mla]
    return kda, sorted(mla)


def make_models(specs: list[BodySpec]) -> dict[int, str]:
    MODELS.mkdir(parents=True, exist_ok=True)
    ds_base = json.loads(DS_SOURCE.read_text(encoding="utf-8"))
    kimi_base = json.loads(KIMI_SOURCE.read_text(encoding="utf-8"))
    refs: dict[int, str] = {}
    for spec in specs:
        if spec.kind == "direct":
            cfg = json.loads(json.dumps(ds_base))
            cfg.update(
                num_hidden_layers=spec.depth, n_routed_experts=spec.experts,
                num_experts_per_tok=spec.topk, moe_intermediate_size=spec.expert_intermediate,
                n_shared_experts=1, hidden_size=H, vocab_size=VOCAB,
                compress_ratios=v4_ratios(spec.depth),
            )
        else:
            cfg = json.loads(json.dumps(kimi_base))
            text = cfg["text_config"]
            kda, mla = kimi_pattern(spec.depth)
            text.update(
                num_hidden_layers=spec.depth, num_experts=spec.experts,
                num_experts_per_token=spec.topk, moe_intermediate_size=spec.expert_intermediate,
                routed_expert_hidden_size=spec.routed_hidden, num_shared_experts=spec.shared_experts,
                first_k_dense_replace=0, hidden_size=H, vocab_size=VOCAB,
            )
            text["linear_attn_config"]["kda_layers"] = kda
            text["linear_attn_config"]["full_attn_layers"] = mla
        model_dir = Path(spec.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # One Kimi model per depth with all but one layer KDA gives clean per-layer KDA/MLA costs.
    for depth in DEPTHS:
        cfg = json.loads(json.dumps(kimi_base))
        text = cfg["text_config"]
        text.update(
            num_hidden_layers=depth, num_experts=specs[0].experts,
            num_experts_per_token=specs[0].topk, first_k_dense_replace=0,
            hidden_size=H, vocab_size=VOCAB,
        )
        text["linear_attn_config"]["kda_layers"] = list(range(1, depth))
        text["linear_attn_config"]["full_attn_layers"] = [depth]
        model_dir = MODELS / f"attention_ref_kimi_L{depth}"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        refs[depth] = str(model_dir)
    return refs


def make_native_anchors() -> dict[str, dict[str, Any]]:
    """Return native ~5T reference models, including a generated GLM-5.2 scaling."""
    anchors = {
        "kimi_ratio": {
            "label": "Kimi ratio-preserved 5T", "family": "Kimi K3",
            "model_path": str(NATIVE_ANCHOR_SOURCES["kimi_ratio"]),
            "total_t": 4.919, "active_b": 187.9, "depth": 93,
            "note": "Kimi K3 sparsity ratio preserved",
        },
        "kimi_trend": {
            "label": "Kimi-trend 5T", "family": "Kimi K3",
            "model_path": str(NATIVE_ANCHOR_SOURCES["kimi_trend"]),
            "total_t": 4.955, "active_b": 184.5, "depth": 116,
            "note": "K2.5-to-K3 depth/expert trend extrapolated",
        },
        "deepseek_ratio": {
            "label": "DeepSeek ratio-preserved 5T", "family": "DeepSeek V4",
            "model_path": str(NATIVE_ANCHOR_SOURCES["deepseek_ratio"]),
            "total_t": 4.953, "active_b": 150.1, "depth": 61,
            "note": "DeepSeek V4 sparsity ratio preserved",
        },
    }

    # AIC's GLM path models all 78 blocks as MoE. E=1664 places that modeled
    # routed capacity at 4.904T; top-52 preserves GLM's native 8/256 sparsity.
    cfg = json.loads(GLM_SOURCE.read_text(encoding="utf-8"))
    cfg.update(n_routed_experts=1664, num_experts=1664, num_experts_per_tok=52)
    glm_dir = MODELS / "glm52_ratio_preserved_5t"
    glm_dir.mkdir(parents=True, exist_ok=True)
    (glm_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    anchors["glm52_ratio"] = {
        "label": "GLM-5.2 ratio-preserved 5T", "family": "GLM-5.2",
        "model_path": str(glm_dir), "total_t": 5.0, "active_b": 160.0, "depth": 78,
        "note": "Native DSA; AIC-modeled MoE capacity, 52/1664 routing",
    }
    return anchors


def make_dsa_validation_model() -> str:
    """Generate a native GLM DSA stack with uniformly FP8 attention projections."""
    cfg = json.loads(GLM_SOURCE.read_text(encoding="utf-8"))
    cfg["quantization_config"]["ignore"] = ["lm_head", "model.embed_tokens"]
    model_dir = MODELS / "glm52_fp8_dsa_validation"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(model_dir)


def points() -> list[tuple[str, int, int, int]]:
    out = []
    for system in SYSTEMS:
        for context in CONTEXTS:
            for dep in (16, 64):
                for cpr in ((1, 2, 4, 8) if context == 262_144 else (1, 2, 4)):
                    out.append((system, context, dep, cpr))
        for cpr in (1, 2, 4, 8):
            out.append((system, 262_144, 32, cpr))
    return out


def cache_key(
    model: str, system: str, context: int, dep: int, cpr: int,
    nextn: int = 0, acceptance_length: float = 1.0,
) -> str:
    config_path = Path(model) / "config.json"
    signature = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16] if config_path.exists() else "builtin"
    key: list[Any] = [model, signature, system, context, dep, cpr]
    # Retain the original no-spec cache identity so existing evidence is reused.
    if nextn:
        key += [nextn, acceptance_length]
    return json.dumps(key, separators=(",", ":"))


def load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["cache_key"]] = row
    return out


def estimate(
    model: str, system: str, context: int, dep: int, cpr: int,
    nextn: int = 0, acceptance_length: float = 1.0,
) -> dict[str, Any]:
    stdout, stderr = io.StringIO(), io.StringIO()
    kwargs: dict[str, Any] = {}
    if nextn:
        # AIC expects accepted *draft* tokens; InferenceX AL includes the
        # guaranteed target token.
        kwargs["nextn_accepted"] = acceptance_length - 1.0
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = cli_estimate(
            model, SYSTEMS[system]["id"], mode="static_gen", backend_name="sglang",
            backend_version="estimate", database_mode="SOL", isl=context, osl=2,
            prefix=context, batch_size=cpr, tp_size=1, pp_size=1,
            attention_dp_size=dep, moe_tp_size=1, moe_ep_size=dep,
            gemm_quant_mode="fp8_block", moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
            kvcache_quant_mode="fp8", fmha_quant_mode="bfloat16",
            nextn=nextn, stride=1, engine_step_backend="python", systems_paths=SYSTEMS_PATHS,
            **kwargs,
        )
    latency = {str(k): float(v) for k, v in result.summary.get_generation_latency_dict().items()}
    sources = {str(k): str(v or "unknown").lower() for k, v in result.summary.get_generation_source_dict().items()}
    non_sol = sum(v for k, v in latency.items() if v > 0 and sources.get(k) != "sol")
    memory = result.summary.get_memory()
    return {
        "cache_key": cache_key(model, system, context, dep, cpr, nextn, acceptance_length),
        "model": model, "system": system, "context": context, "dep": dep, "cpr": cpr,
        "nextn": nextn, "acceptance_length": acceptance_length,
        "verification_step_ms": sum(latency.values()),
        "tpot_ms": sum(latency.values()) / acceptance_length, "latency": latency, "sources": sources,
        "non_sol_ms": non_sol, "weights_gib": float(memory["weights"]),
        "kvcache_gib": float(memory["kvcache"]), "activations_gib": float(memory["activations"]),
        "nccl_gib": float(memory["nccl"]), "others_gib": float(memory["others"]),
        "memory_gib": float(memory["total"]), "oom": bool(result.summary.check_oom()),
        "modes": {k: result.raw.get(k) for k in ("gemm", "moe", "fmha", "kvcache")},
    }


def run_estimates(
    models: list[str], all_points: list[tuple[str, int, int, int]],
    al_curve: dict[int, float] | None = None,
) -> dict[str, dict[str, Any]]:
    cache = load_cache()
    curve = al_curve or {0: 1.0}
    tasks = [(m, *p, nextn, al) for m in models for p in all_points for nextn, al in curve.items()]
    pending = [task for task in tasks if cache_key(*task) not in cache]
    print(f"AIC estimates: {len(tasks) - len(pending)} cached, {len(pending)} pending", flush=True)
    if pending:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as stream:
            for index, task in enumerate(pending, 1):
                row = estimate(*task)
                cache[row["cache_key"]] = row
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                if index % 100 == 0 or index == len(pending):
                    print(f"completed {index}/{len(pending)}", flush=True)
    return cache


def v4_attention_names() -> set[str]:
    return {"generation_attention", "generation_mhc_pre", "generation_mhc_post", "generation_attn_norm"}


def kimi_attention_value(latency: dict[str, float]) -> float:
    return sum(
        value for name, value in latency.items()
        if name.startswith("generation_kda_") or name.startswith("generation_mla_")
        or name.startswith("generation_bmm_") or name in {"generation_attention", "generation_attn_res"}
    )


def v4_attention_value(latency: dict[str, float]) -> float:
    return sum(latency.get(name, 0.0) for name in v4_attention_names())


def kimi_primitives(latency: dict[str, float], layers: int) -> tuple[float, float, float]:
    kda = sum(v for k, v in latency.items() if k.startswith("generation_kda_")) / (layers - 1)
    mla = sum(
        v for k, v in latency.items()
        if k.startswith("generation_mla_") or k.startswith("generation_bmm_") or k == "generation_attention"
    )
    attn_res = latency.get("generation_attn_res", 0.0) / layers
    return kda, mla, attn_res


def allocate_counts(parts: dict[str, float], layers: int) -> dict[str, int]:
    raw = {k: v * layers for k, v in parts.items()}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remaining = layers - sum(counts.values())
    for key in sorted(parts, key=lambda k: raw[k] - counts[k], reverse=True)[:remaining]:
        counts[key] += 1
    return counts


def kda_weights() -> float:
    heads, hd, p = 96, 128, 96 * 128
    return float(4 * p * H + (heads + hd) * H + p * hd + H * p)


def mla_weights() -> float:
    heads, q, kv, nope, rope, v = 96, 1536, 512, 128, 64, 128
    return float((q + kv + rope) * H + heads * (nope + rope) * q + heads * (nope + v) * kv + 2 * heads * v * H)


def kda_cache_per_layer() -> float:
    heads, hd, kernel = 96, 128, 4
    return float((heads * hd * hd * 4 + (kernel - 1) * 3 * heads * hd * 2) * 5)


def mla_cache_per_layer(context: int) -> float:
    return float(context * (512 + 64))


def dsa_full_fraction(layers: int) -> float:
    def skips(lid: int) -> bool:
        return max(lid - 3 + 1, 0) % 4 != 0
    return sum(not skips(lid) for lid in range(layers)) / layers


def dsa_layer_sol(system: str, context: int, cpr: int, topk: int, skip_indexer: bool) -> float:
    # GLM-5.2 native DSA dimensions; explicit extension of AIC's published SOL formula.
    h, q, kv, nope, rope, v = 6144, 2048, 512, 192, 64, 256
    heads, idx_heads, idx_dim = 64, 32, 128
    effective = min(context, topk)
    proj_out = q + kv + rope + (0 if skip_indexer else idx_dim)
    gemm_ops = (
        2 * cpr * h * proj_out + 2 * cpr * q * heads * (nope + rope)
        + (0 if skip_indexer else 2 * cpr * q * idx_heads * idx_dim + 2 * cpr * h * idx_heads)
        + 2 * cpr * heads * v * h + 2 * heads * cpr * nope * kv + 2 * heads * cpr * kv * v
    )
    index_ops = 0 if skip_indexer else 2 * cpr * idx_heads * idx_dim * context
    sparse_ops = 2 * cpr * heads * ((kv + rope) + kv) * effective
    weight_bytes = (
        h * proj_out + q * heads * (nope + rope)
        + (0 if skip_indexer else q * idx_heads * idx_dim + h * idx_heads)
        + heads * v * h
    )
    cache_bytes = (0 if skip_indexer else cpr * context * (idx_dim + 4)) + cpr * effective * (kv + rope)
    math_ms = (gemm_ops / 5e15 + index_ops / 5e15 + sparse_ops / 2.5e15) * 1000
    mem_ms = (weight_bytes + cache_bytes) / SYSTEMS[system]["mem_bw"] * 1000
    return max(math_ms, mem_ms)


def dsa_latency(system: str, context: int, cpr: int, layers: int, topk: int, shared_index: bool = True) -> float:
    full = dsa_layer_sol(system, context, cpr, topk, False)
    if not shared_index:
        return layers * full
    skip = dsa_layer_sol(system, context, cpr, topk, True)
    fraction = dsa_full_fraction(layers)
    return layers * (fraction * full + (1 - fraction) * skip)


def dsa_weights_per_layer(layers: int, shared_index: bool = True) -> float:
    h, q, kv, nope, rope, v = 6144, 2048, 512, 192, 64, 256
    heads, idx_heads, idx_dim = 64, 32, 128
    base = h * q + q * heads * (nope + rope) + h * (kv + rope) + kv * heads * (nope + v) + heads * v * h
    index = q * idx_heads * idx_dim + h * idx_heads
    return base + (dsa_full_fraction(layers) if shared_index else 1.0) * index


def dsa_cache_per_layer(context: int) -> float:
    # AIC stores full latent KV and indexer cache even though decode reads only top-k entries.
    return float(context * (512 + 64 * 2 + 128 + 4))


def v4_weights(layers: int) -> float:
    total = 0.0
    for ratio in v4_ratios(layers):
        heads, h, q, o, hd, idx_heads, idx_dim, groups = 128, H, 1536, 1024, 512, 64, 128, 16
        gemm = h * q + q * heads * hd + h * hd + groups * o * h
        bf16 = heads * hd * o
        fp32 = heads
        if ratio:
            mult = 2 if ratio == 4 else 1
            gemm += 2 * h * mult * hd
            fp32 += ratio * mult * hd
        if ratio == 4:
            gemm += q * idx_heads * idx_dim + 2 * h * 2 * idx_dim
            bf16 += h * idx_heads
            fp32 += ratio * 2 * idx_dim
        total += gemm + 2 * bf16 + 4 * fp32
    return total


def v4_cache(layers: int, context: int) -> float:
    total = 0.0
    hd, window, idx_dim = 512, 128, 128
    for ratio in v4_ratios(layers):
        total += min(context, window) * hd
        if ratio:
            entries = context // ratio
            total += entries * hd
            mult = 2 if ratio == 4 else 1
            total += 2 * ratio * mult * hd * 4
            if ratio == 4:
                total += entries * idx_dim * 0.5 + 2 * ratio * 2 * idx_dim * 4
    return total


def native_kimi_counts(layers: int) -> tuple[int, int]:
    kda, mla = kimi_pattern(layers)
    return len(kda), len(mla)


def attention_metrics(
    profile_id: str, system: str, context: int, cpr: int, layers: int,
    ref_latency: dict[str, float], v4_latency: float, dsa_spec_scale: float = 1.0,
) -> tuple[float, float, float, str]:
    profile = ATTENTION_PROFILES[profile_id]
    counts = allocate_counts(profile["parts"], layers)
    kda, mla, attnres = kimi_primitives(ref_latency, layers)
    latency = counts.get("kda", 0) * (kda + attnres) + counts.get("mla", 0) * (mla + attnres)
    if counts.get("dsa"):
        latency += dsa_spec_scale * dsa_latency(
            system, context, cpr, counts["dsa"], int(profile.get("dsa_topk", 2048)),
            bool(profile.get("dsa_shared_index", True)),
        )
    if counts.get("v4"):
        latency += counts["v4"] * v4_latency / layers
    weights = counts.get("kda", 0) * kda_weights() + counts.get("mla", 0) * mla_weights()
    cache = counts.get("kda", 0) * kda_cache_per_layer() + counts.get("mla", 0) * mla_cache_per_layer(context)
    if counts.get("dsa"):
        weights += counts["dsa"] * dsa_weights_per_layer(
            counts["dsa"], bool(profile.get("dsa_shared_index", True))
        )
        cache += counts["dsa"] * dsa_cache_per_layer(context)
    if counts.get("v4"):
        weights += v4_weights(layers) * counts["v4"] / layers
        cache += v4_cache(layers, context) * counts["v4"] / layers
    provenance = "AIC SOL composite"
    if counts.get("dsa"):
        provenance += "+DSA top-k/index-reuse SOL extension"
    return latency, weights, cache, provenance


def body_native_weight_cache(spec: BodySpec, context: int) -> tuple[float, float]:
    if spec.kind == "direct":
        return v4_weights(spec.depth), v4_cache(spec.depth, context)
    kda, mla = native_kimi_counts(spec.depth)
    return (
        kda * kda_weights() + mla * mla_weights(),
        kda * kda_cache_per_layer() + mla * mla_cache_per_layer(context),
    )


def estimated_params(spec: BodySpec, attention_weights: float) -> tuple[float, float]:
    embed = 2 * VOCAB * H
    router = spec.depth * H * spec.experts
    shared = spec.depth * 3 * H * spec.expert_intermediate * spec.shared_experts
    latent_proj = 0 if spec.kind == "direct" else spec.depth * 2 * H * spec.routed_hidden
    total = spec.routed_total + embed + router + shared + latent_proj + attention_weights
    active = spec.routed_active + embed + router + shared + latent_proj + attention_weights
    return total, active


def compose_rows(specs: list[BodySpec], refs: dict[int, str], cache: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    direct_balanced = {
        depth: min(
            (s for s in specs if s.kind == "direct" and s.active_tier == "balanced" and s.depth == depth),
            key=lambda s: abs(s.expert_intermediate - 3072),
        )
        for depth in DEPTHS
    }
    for spec in specs:
        for system, context, dep, cpr in points():
            body_run = cache[cache_key(spec.model_path, system, context, dep, cpr)]
            ref_run = cache[cache_key(refs[spec.depth], system, context, dep, cpr)]
            ds_ref = cache[cache_key(direct_balanced[spec.depth].model_path, system, context, dep, cpr)]
            native_attn = v4_attention_value(body_run["latency"]) if spec.kind == "direct" else kimi_attention_value(body_run["latency"])
            common_body = body_run["tpot_ms"] - native_attn
            native_w, native_kv = body_native_weight_cache(spec, context)
            base_weight_bytes = body_run["weights_gib"] * 2**30 - native_w
            non_weight_cache = body_run["memory_gib"] - body_run["weights_gib"] - body_run["kvcache_gib"]
            v4_attn = v4_attention_value(ds_ref["latency"])
            for profile_id in ATTENTION_PROFILES:
                attn, attn_w, attn_kv, provenance = attention_metrics(
                    profile_id, system, context, cpr, spec.depth, ref_run["latency"], v4_attn
                )
                tpot = common_body + attn
                weight_gib = (base_weight_bytes + attn_w) / 2**30
                kv_gib = cpr * attn_kv / 2**30
                memory = non_weight_cache + weight_gib + kv_gib
                total_p, active_p = estimated_params(spec, attn_w)
                risk = "aggressive" if "aggressive" in (spec.active_tier, ATTENTION_PROFILES[profile_id]["risk"]) or spec.depth == 48 else (
                    "conservative" if spec.active_tier == "conservative" and ATTENTION_PROFILES[profile_id]["risk"] == "conservative" and spec.depth >= 78 else "balanced"
                )
                rows.append({
                    "architecture_id": f"{spec.id}__{profile_id}", "body_id": spec.id,
                    "attention_id": profile_id, "attention": ATTENTION_PROFILES[profile_id]["label"],
                    "body_kind": spec.kind, "depth": spec.depth, "active_tier": spec.active_tier,
                    "plausibility_tier": risk, "expert_intermediate": spec.expert_intermediate,
                    "routed_hidden": spec.routed_hidden, "experts": spec.experts, "topk": spec.topk,
                    "shared_experts": spec.shared_experts, "routed_total_t": spec.routed_total / 1e12,
                    "routed_active_b": spec.routed_active / 1e9, "estimated_total_t": total_p / 1e12,
                    "estimated_active_b": active_p / 1e9, "active_ratio_pct": active_p / total_p * 100,
                    "system": system, "context": context, "dep": dep, "cpr": cpr,
                    "global_concurrency": dep * cpr, "tpot_ms": tpot, "tps_user": 1000 / tpot,
                    "tps_gpu": cpr * 1000 / tpot, "common_body_ms": common_body,
                    "attention_ms": attn, "attention_share_pct": attn / tpot * 100,
                    "weight_gib": weight_gib, "kvcache_gib": kv_gib, "memory_gib": memory,
                    "capacity_gib": SYSTEMS[system]["capacity"] / 2**30,
                    "oom": memory > SYSTEMS[system]["capacity"] / 2**30,
                    "provenance": provenance, "body_non_sol_ms": body_run["non_sol_ms"],
                })
    return rows


def compose_spec_rows(
    specs: list[BodySpec], refs: dict[int, str], dsa_validation_model: str,
    cache: dict[str, dict[str, Any]], al_curve: dict[int, float], no_spec_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compose arbitrary-attention speculative verification steps.

    Each native body supplies its AIC target-verification geometry. Native Kimi
    bodies use AIC's K3/DSpark path; direct bodies use AIC's generic MTP target
    verifier. The same measured Kimi-K3 DSpark AL curve discounts every step.
    """
    rows: list[dict[str, Any]] = []
    baseline = lookup(no_spec_rows)
    direct_balanced = {
        depth: min(
            (s for s in specs if s.kind == "direct" and s.active_tier == "balanced" and s.depth == depth),
            key=lambda s: abs(s.expert_intermediate - 3072),
        )
        for depth in DEPTHS
    }
    for spec in specs:
        for system, context, dep, cpr in points():
            native_w, native_kv = body_native_weight_cache(spec, context)
            for nextn, al in al_curve.items():
                if nextn == 0:
                    continue
                body_run = cache[cache_key(spec.model_path, system, context, dep, cpr, nextn, al)]
                ref_run = cache[cache_key(refs[spec.depth], system, context, dep, cpr, nextn, al)]
                ds_ref = cache[cache_key(direct_balanced[spec.depth].model_path, system, context, dep, cpr, nextn, al)]
                dsa_ref = cache[cache_key(dsa_validation_model, system, context, dep, cpr, nextn, al)]
                native_attn = (
                    v4_attention_value(body_run["latency"])
                    if spec.kind == "direct" else kimi_attention_value(body_run["latency"])
                )
                common_body_step = body_run["verification_step_ms"] - native_attn
                base_weight_bytes = body_run["weights_gib"] * 2**30 - native_w
                non_weight_cache = body_run["memory_gib"] - body_run["weights_gib"] - body_run["kvcache_gib"]
                # Kimi's DSpark path allocates extra verification/cache state
                # beyond the resident target attention state. Preserve it when
                # replacing the native attention architecture.
                spec_kv_overhead_gib = body_run["kvcache_gib"] - cpr * native_kv / 2**30
                v4_attn = v4_attention_value(ds_ref["latency"])
                dsa_base = dsa_latency(system, context, cpr, 78, 2048, shared_index=False)
                dsa_spec_scale = dsa_ref["latency"]["generation_attention"] / dsa_base
                for profile_id in ATTENTION_PROFILES:
                    attn, attn_w, attn_kv, provenance = attention_metrics(
                        profile_id, system, context, cpr, spec.depth, ref_run["latency"],
                        v4_attn, dsa_spec_scale,
                    )
                    step_ms = common_body_step + attn
                    tpot = step_ms / al
                    weight_gib = (base_weight_bytes + attn_w) / 2**30
                    kv_gib = cpr * attn_kv / 2**30 + spec_kv_overhead_gib
                    memory = non_weight_cache + weight_gib + kv_gib
                    total_p, active_p = estimated_params(spec, attn_w)
                    risk = "aggressive" if "aggressive" in (spec.active_tier, ATTENTION_PROFILES[profile_id]["risk"]) or spec.depth == 48 else (
                        "conservative" if spec.active_tier == "conservative" and ATTENTION_PROFILES[profile_id]["risk"] == "conservative" and spec.depth >= 78 else "balanced"
                    )
                    base = baseline[(f"{spec.id}__{profile_id}", system, context, dep, cpr)]
                    rows.append({
                        "architecture_id": f"{spec.id}__{profile_id}", "body_id": spec.id,
                        "attention_id": profile_id, "attention": ATTENTION_PROFILES[profile_id]["label"],
                        "body_kind": spec.kind, "depth": spec.depth, "active_tier": spec.active_tier,
                        "plausibility_tier": risk, "expert_intermediate": spec.expert_intermediate,
                        "routed_hidden": spec.routed_hidden, "experts": spec.experts, "topk": spec.topk,
                        "shared_experts": spec.shared_experts, "estimated_total_t": total_p / 1e12,
                        "estimated_active_b": active_p / 1e9, "system": system, "context": context,
                        "dep": dep, "cpr": cpr, "global_concurrency": dep * cpr,
                        "nextn": nextn, "acceptance_length": al, "aic_nextn_accepted": al - 1.0,
                        "verification_step_ms": step_ms, "tpot_ms": tpot, "tps_user": 1000 / tpot,
                        "tps_gpu": cpr * 1000 / tpot, "no_spec_tps_user": base["tps_user"],
                        "gain": (1000 / tpot) / base["tps_user"], "common_body_step_ms": common_body_step,
                        "attention_step_ms": attn, "weight_gib": weight_gib, "kvcache_gib": kv_gib,
                        "spec_kv_overhead_gib": spec_kv_overhead_gib,
                        "memory_gib": memory, "capacity_gib": SYSTEMS[system]["capacity"] / 2**30,
                        "oom": memory > SYSTEMS[system]["capacity"] / 2**30,
                        "verifier_path": "Kimi-K3 DSpark" if spec.kind == "latent" else "AIC generic MTP target verifier",
                        "provenance": provenance + "+DSpark AL sensitivity",
                        "body_non_sol_ms": body_run["non_sol_ms"],
                    })
    return rows


def best_spec_rows(no_spec_rows: list[dict[str, Any]], spec_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_spec: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in spec_rows:
        key = (row["architecture_id"], row["system"], row["context"], row["dep"], row["cpr"])
        by_spec.setdefault(key, []).append(row)
    out = []
    for base in no_spec_rows:
        key = (base["architecture_id"], base["system"], base["context"], base["dep"], base["cpr"])
        candidates = [row for row in by_spec[key] if not row["oom"]]
        base_candidate = {
            "nextn": 0, "acceptance_length": 1.0, "verification_step_ms": base["tpot_ms"],
            "tpot_ms": base["tpot_ms"], "tps_user": base["tps_user"], "tps_gpu": base["tps_gpu"],
            "memory_gib": base["memory_gib"], "oom": base["oom"], "verifier_path": "none",
        }
        if not base["oom"]:
            candidates.append(base_candidate)
        best = max(candidates or [base_candidate], key=lambda row: row["tps_user"])
        out.append({
            "architecture_id": base["architecture_id"], "body_id": base["body_id"],
            "attention_id": base["attention_id"], "attention": base["attention"],
            "body_kind": base["body_kind"], "depth": base["depth"],
            "plausibility_tier": base["plausibility_tier"], "estimated_total_t": base["estimated_total_t"],
            "estimated_active_b": base["estimated_active_b"], "system": base["system"],
            "context": base["context"], "dep": base["dep"], "cpr": base["cpr"],
            "global_concurrency": base["global_concurrency"], "no_spec_tps_user": base["tps_user"],
            "best_nextn": best["nextn"], "acceptance_length": best["acceptance_length"],
            "verification_step_ms": best["verification_step_ms"], "effective_tpot_ms": best["tpot_ms"],
            "spec_tps_user": best["tps_user"], "spec_tps_gpu": best["tps_gpu"],
            "gain": best["tps_user"] / base["tps_user"], "memory_gib": best["memory_gib"],
            "oom": best["oom"], "verifier_path": best["verifier_path"],
        })
    return out


def spec_architecture_summary(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (r["architecture_id"], r["system"], r["context"], r["dep"], r["cpr"]): r
        for r in best_rows
    }
    out = []
    for architecture_id in sorted({r["architecture_id"] for r in best_rows}):
        sample = index[(architecture_id, "GB300", 262_144, 64, 1)]
        row = {
            "architecture_id": architecture_id, "body_id": sample["body_id"],
            "attention_id": sample["attention_id"], "attention": sample["attention"],
            "body_kind": sample["body_kind"], "depth": sample["depth"],
            "plausibility_tier": sample["plausibility_tier"],
            "estimated_total_t": sample["estimated_total_t"], "estimated_active_b": sample["estimated_active_b"],
        }
        for cpr in (1, 2, 4, 8):
            point = index[(architecture_id, "GB300", 262_144, 64, cpr)]
            prefix = f"gb_dep64_c{cpr}"
            row.update({
                f"{prefix}_no_spec_tps": point["no_spec_tps_user"],
                f"{prefix}_best_nextn": point["best_nextn"],
                f"{prefix}_acceptance_length": point["acceptance_length"],
                f"{prefix}_spec_tps": point["spec_tps_user"],
                f"{prefix}_gain": point["gain"], f"{prefix}_memory_gib": point["memory_gib"],
            })
        out.append(row)
    return out


def native_anchor_rows(anchors: dict[str, dict[str, Any]], cache: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for anchor_id, anchor in anchors.items():
        for system, context, dep, cpr in points():
            run = cache[cache_key(anchor["model_path"], system, context, dep, cpr)]
            rows.append({
                "anchor_id": anchor_id, "label": anchor["label"], "family": anchor["family"],
                "total_t": anchor["total_t"], "active_b": anchor["active_b"], "depth": anchor["depth"],
                "note": anchor["note"], "system": system, "context": context, "dep": dep, "cpr": cpr,
                "global_concurrency": dep * cpr, "tpot_ms": run["tpot_ms"],
                "tps_user": 1000 / run["tpot_ms"], "tps_gpu": cpr * 1000 / run["tpot_ms"],
                "memory_gib": run["memory_gib"], "weight_gib": run["weights_gib"],
                "kvcache_gib": run["kvcache_gib"], "oom": run["oom"],
                "non_sol_ms": run["non_sol_ms"], "provenance": "direct AIC SOL",
            })
    return rows


def native_anchor_spec_rows(
    anchors: dict[str, dict[str, Any]], cache: dict[str, dict[str, Any]], al_curve: dict[int, float],
) -> list[dict[str, Any]]:
    rows = []
    for anchor_id, anchor in anchors.items():
        for system, context, dep, cpr in points():
            base = cache[cache_key(anchor["model_path"], system, context, dep, cpr)]
            for nextn, al in al_curve.items():
                if nextn == 0:
                    continue
                run = cache[cache_key(anchor["model_path"], system, context, dep, cpr, nextn, al)]
                rows.append({
                    "anchor_id": anchor_id, "label": anchor["label"], "family": anchor["family"],
                    "total_t": anchor["total_t"], "active_b": anchor["active_b"], "depth": anchor["depth"],
                    "system": system, "context": context, "dep": dep, "cpr": cpr,
                    "global_concurrency": dep * cpr, "nextn": nextn, "acceptance_length": al,
                    "aic_nextn_accepted": al - 1.0, "verification_step_ms": run["verification_step_ms"],
                    "effective_tpot_ms": run["tpot_ms"], "tps_user": 1000 / run["tpot_ms"],
                    "tps_gpu": cpr * 1000 / run["tpot_ms"], "no_spec_tps_user": 1000 / base["tpot_ms"],
                    "gain": base["tpot_ms"] / run["tpot_ms"], "memory_gib": run["memory_gib"],
                    "oom": run["oom"], "non_sol_ms": run["non_sol_ms"],
                    "verifier_path": "Kimi-K3 DSpark" if anchor["family"] == "Kimi K3" else "AIC family-native MTP target verifier",
                    "provenance": "direct AIC SOL+Kimi DSpark AL sensitivity",
                })
    return rows


def best_native_anchor_spec_rows(
    no_spec_rows: list[dict[str, Any]], spec_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in spec_rows:
        key = (row["anchor_id"], row["system"], row["context"], row["dep"], row["cpr"])
        groups.setdefault(key, []).append(row)
    out = []
    for base in no_spec_rows:
        key = (base["anchor_id"], base["system"], base["context"], base["dep"], base["cpr"])
        candidates = [row for row in groups[key] if not row["oom"]]
        baseline = {
            "nextn": 0, "acceptance_length": 1.0, "verification_step_ms": base["tpot_ms"],
            "effective_tpot_ms": base["tpot_ms"], "tps_user": base["tps_user"],
            "tps_gpu": base["tps_gpu"], "memory_gib": base["memory_gib"], "oom": base["oom"],
            "verifier_path": "none",
        }
        if not base["oom"]:
            candidates.append(baseline)
        best = max(candidates or [baseline], key=lambda row: row["tps_user"])
        out.append({
            "anchor_id": base["anchor_id"], "label": base["label"], "family": base["family"],
            "total_t": base["total_t"], "active_b": base["active_b"], "depth": base["depth"],
            "system": base["system"], "context": base["context"], "dep": base["dep"], "cpr": base["cpr"],
            "global_concurrency": base["global_concurrency"], "no_spec_tps_user": base["tps_user"],
            "best_nextn": best["nextn"], "acceptance_length": best["acceptance_length"],
            "verification_step_ms": best["verification_step_ms"], "effective_tpot_ms": best["effective_tpot_ms"],
            "spec_tps_user": best["tps_user"], "spec_tps_gpu": best["tps_gpu"],
            "gain": best["tps_user"] / base["tps_user"], "memory_gib": best["memory_gib"],
            "oom": best["oom"], "verifier_path": best["verifier_path"],
        })
    return out


def validate_reconstruction(
    specs: list[BodySpec], refs: dict[int, str], cache: dict[str, dict[str, Any]], dsa_validation_model: str,
    al_curve: dict[int, float],
) -> dict[str, float]:
    """Prove that native attention decomposes/recomposes at representative points."""
    point = ("GB300", 262_144, 64, 1)
    direct = next(s for s in specs if s.kind == "direct" and s.depth == 78 and s.active_tier == "balanced" and s.expert_intermediate == 3072)
    latent = next(s for s in specs if s.kind == "latent" and s.depth == 78 and s.active_tier == "balanced" and s.shared_experts == 1)
    direct_run = cache[cache_key(direct.model_path, *point)]
    latent_run = cache[cache_key(latent.model_path, *point)]
    ref_run = cache[cache_key(refs[78], *point)]

    direct_rebuilt, direct_w, direct_kv, _ = attention_metrics(
        "v4_compressed", *point[:2], point[3], direct.depth, ref_run["latency"],
        v4_attention_value(direct_run["latency"]),
    )
    latent_rebuilt, latent_w, latent_kv, _ = attention_metrics(
        "kda75_mla25", *point[:2], point[3], latent.depth, ref_run["latency"],
        v4_attention_value(direct_run["latency"]),
    )
    native_direct_w, native_direct_kv = body_native_weight_cache(direct, point[1])
    native_latent_w, native_latent_kv = body_native_weight_cache(latent, point[1])
    dsa_run = cache[cache_key(dsa_validation_model, *point)]
    spec_n, spec_al = 3, al_curve[3]
    direct_spec_run = cache[cache_key(direct.model_path, *point, spec_n, spec_al)]
    latent_spec_run = cache[cache_key(latent.model_path, *point, spec_n, spec_al)]
    ref_spec_run = cache[cache_key(refs[78], *point, spec_n, spec_al)]
    direct_spec_rebuilt, _, _, _ = attention_metrics(
        "v4_compressed", *point[:2], point[3], direct.depth, ref_spec_run["latency"],
        v4_attention_value(direct_spec_run["latency"]),
    )
    latent_spec_rebuilt, _, _, _ = attention_metrics(
        "kda75_mla25", *point[:2], point[3], latent.depth, ref_spec_run["latency"],
        v4_attention_value(direct_spec_run["latency"]),
    )
    errors = {
        "direct_latency_ms": abs(direct_rebuilt - v4_attention_value(direct_run["latency"])),
        "direct_weight_bytes": abs(direct_w - native_direct_w),
        "direct_cache_bytes": abs(direct_kv - native_direct_kv),
        "latent_latency_ms": abs(latent_rebuilt - kimi_attention_value(latent_run["latency"])),
        "latent_weight_bytes": abs(latent_w - native_latent_w),
        "latent_cache_bytes": abs(latent_kv - native_latent_kv),
        "direct_spec_latency_ms": abs(direct_spec_rebuilt - v4_attention_value(direct_spec_run["latency"])),
        "latent_spec_latency_ms": abs(latent_spec_rebuilt - kimi_attention_value(latent_spec_run["latency"])),
        # AIC's current SOL skip path charges the full indexer, so validate the
        # custom equation against its per-layer-index (no reuse) native path.
        "dsa_aic_vs_formula_ms": abs(
            dsa_run["latency"]["generation_attention"]
            - dsa_latency("GB300", point[1], point[3], 78, 2048, shared_index=False)
        ),
    }
    if any(errors[k] > 1e-9 for k in ("direct_latency_ms", "latent_latency_ms", "direct_spec_latency_ms", "latent_spec_latency_ms")) or errors["dsa_aic_vs_formula_ms"] > 1e-5:
        raise RuntimeError(f"native latency reconstruction failed: {errors}")
    if any(errors[k] > 1e-3 for k in errors if k.endswith("_bytes")):
        raise RuntimeError(f"native memory reconstruction failed: {errors}")
    return errors


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def lookup(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    return {(r["architecture_id"], r["system"], r["context"], r["dep"], r["cpr"]): r for r in rows}


def architecture_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = lookup(rows)
    archs = sorted({r["architecture_id"] for r in rows})
    summaries = []
    for arch in archs:
        sample = next(r for r in rows if r["architecture_id"] == arch)
        def one(system: str, context: int, dep: int, cpr: int) -> dict[str, Any]:
            return by[(arch, system, context, dep, cpr)]
        gb16_1, gb16_4 = one("GB300", 262_144, 16, 1), one("GB300", 262_144, 16, 4)
        gb64_1, gb64_4 = one("GB300", 262_144, 64, 1), one("GB300", 262_144, 64, 4)
        vr64_1, vr64_4 = one("VR200 HBM", 262_144, 64, 1), one("VR200 HBM", 262_144, 64, 4)
        vrf64_1, vrf64_4 = one("VR200 HBM+NVLink", 262_144, 64, 1), one("VR200 HBM+NVLink", 262_144, 64, 4)
        short, long = one("GB300", 32_768, 64, 1), one("GB300", 1_048_576, 64, 1)
        # Linear capacity estimate at 256K; include 90% operational and raw boundaries.
        m1, m2 = one("GB300", 262_144, 64, 1)["memory_gib"], one("GB300", 262_144, 64, 2)["memory_gib"]
        slope, fixed = m2 - m1, 2 * m1 - m2
        cap = SYSTEMS["GB300"]["capacity"] / 2**30
        max90 = max(0, math.floor((0.9 * cap - fixed) / slope)) if slope > 0 else 0
        maxraw = max(0, math.floor((cap - fixed) / slope)) if slope > 0 else 0
        summaries.append({
            "architecture_id": arch, "plausibility_tier": sample["plausibility_tier"],
            "body_kind": sample["body_kind"], "depth": sample["depth"], "active_tier": sample["active_tier"],
            "attention_id": sample["attention_id"], "attention": sample["attention"],
            "experts": sample["experts"], "topk": sample["topk"], "expert_intermediate": sample["expert_intermediate"],
            "routed_hidden": sample["routed_hidden"], "shared_experts": sample["shared_experts"],
            "estimated_total_t": sample["estimated_total_t"], "estimated_active_b": sample["estimated_active_b"],
            "active_ratio_pct": sample["active_ratio_pct"], "gb_dep16_c1_tps": gb16_1["tps_user"],
            "gb_dep16_c4_tps": gb16_4["tps_user"], "gb_dep16_c4_tps_gpu": gb16_4["tps_gpu"],
            "gb_dep64_c1_tps": gb64_1["tps_user"], "gb_dep64_c4_tps": gb64_4["tps_user"],
            "gb_dep64_c4_tps_gpu": gb64_4["tps_gpu"],
            "concurrency_retention_dep64": gb64_4["tps_user"] / gb64_1["tps_user"],
            "context_retention_1m_over_32k": long["tps_user"] / short["tps_user"],
            "vr_hbm_dep64_c1_tps": vr64_1["tps_user"], "vr_hbm_gain": vr64_1["tps_user"] / gb64_1["tps_user"],
            "vr_hbm_dep64_c4_tps": vr64_4["tps_user"],
            "vr_hbm_fabric_dep64_c1_tps": vrf64_1["tps_user"],
            "vr_fabric_increment": vrf64_1["tps_user"] / vr64_1["tps_user"],
            "vr_hbm_fabric_dep64_c4_tps": vrf64_4["tps_user"],
            "vr_fabric_increment_c4": vrf64_4["tps_user"] / vr64_4["tps_user"],
            "memory_gib_dep64_c1": gb64_1["memory_gib"], "max_cpr_90pct_dep64": max90,
            "max_cpr_raw_dep64": maxraw, "attention_share_gb_dep64_c1": gb64_1["attention_share_pct"],
        })
    return summaries


def pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("gb_dep64_c1_tps", "gb_dep64_c4_tps_gpu", "context_retention_1m_over_32k", "max_cpr_90pct_dep64")
    front = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if all(float(other[m]) >= float(row[m]) for m in metrics) and any(float(other[m]) > float(row[m]) for m in metrics):
                dominated = True; break
        if not dominated:
            front.append(row)
    return front


def pick_recommendations(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    valid = [r for r in summaries if 4.85 <= r["estimated_total_t"] <= 5.10]
    conservative = [r for r in valid if r["plausibility_tier"] == "conservative"]
    balanced = [r for r in valid if r["plausibility_tier"] == "balanced"]
    def best(seq, key): return max(seq, key=key)
    def balanced_score(r):
        return geometric_mean([
            r["gb_dep64_c1_tps"], r["gb_dep64_c4_tps_gpu"],
            100 * r["context_retention_1m_over_32k"], max(1, r["max_cpr_90pct_dep64"]),
        ])
    hybrids = [r for r in valid if r["plausibility_tier"] != "aggressive" and r["attention_id"].startswith("kda") and ("dsa" in r["attention_id"] or "v4" in r["attention_id"])]
    latent = [r for r in valid if r["body_kind"] == "latent" and r["plausibility_tier"] != "aggressive"]
    full_context = [r for r in conservative if "mla" in r["attention_id"]]
    return {
        "Aggressive latency floor": best(valid, lambda r: geometric_mean([r["gb_dep16_c1_tps"], r["gb_dep64_c1_tps"]])),
        "Balanced recommendation": best(balanced, balanced_score),
        "Conservative recommendation": best(conservative, balanced_score),
        "KDA+sparse hybrid": best(hybrids, balanced_score),
        "Latent-MoE option": best(latent, balanced_score),
        "Full-attention fallback": best(full_context, balanced_score),
    }


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if i else "---" for i in range(len(headers))) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def write_report(
    specs: list[BodySpec], rows: list[dict[str, Any]], summaries: list[dict[str, Any]],
    recs: dict[str, dict[str, Any]], anchors: list[dict[str, Any]], validation: dict[str, float],
    spec_best: list[dict[str, Any]], native_spec_best: list[dict[str, Any]], al_curve: dict[int, float],
) -> None:
    lines = [
        "# Controlled 5T architecture design-space study", "",
        "## Scope and interpretation", "",
        "This is an all-else-equal AIC SOL study for resident-cache decode. The primary point is 256K; 32K and 1M are context sensitivities. "
        "There is no prefill, scheduler, queueing, or framework overhead. No-spec results are the baseline; a separate speculative sensitivity applies the same measured Kimi-K3 DSpark acceptance curve to every design. Synthetic attention mixtures are operation-level "
        "composites under AIC's additive SOL model; they are not runnable checkpoints and AIC does not predict model quality.", "",
        "All bodies are approximately 5T, hidden-size 7168, FP4 routed experts, FP8 non-expert weights, FP8 KV, TP1/PP1, and attention-DP=MoE-EP. "
        "The hardware cases are GB300; GB300 compute/fabric with 22-TB/s VR200 HBM; and the same with directional NVLink doubled to 1.8 TB/s.", "",
    ]
    balanced_rec = recs["Balanced recommendation"]
    conservative_rec = recs["Conservative recommendation"]
    aggressive_rec = recs["Aggressive latency floor"]
    def spec_one(architecture_id: str, system: str, context: int, dep: int, cpr: int):
        return next(r for r in spec_best if r["architecture_id"] == architecture_id and r["system"] == system and r["context"] == context and r["dep"] == dep and r["cpr"] == cpr)
    balanced_spec_c4 = spec_one(balanced_rec["architecture_id"], "GB300", 262_144, 64, 4)
    balanced_frontier_c1 = max(
        (r for r in spec_best if r["system"] == "GB300" and r["context"] == 262_144 and r["dep"] == 64 and r["cpr"] == 1 and r["plausibility_tier"] == "balanced" and not r["oom"]),
        key=lambda r: r["spec_tps_user"],
    )
    lines += [
        "## Executive findings", "",
        f"- The best balanced point is a 61-layer direct-MoE body with DeepSeek-V4 compressed attention: {balanced_rec['gb_dep64_c1_tps']:.1f} TPS/user on GB300 and {balanced_rec['vr_hbm_dep64_c1_tps']:.1f} on the VR200-HBM substitution at DEP64, c/r=1, 256K.",
        f"- A more conservative 78-layer/high-active design reaches {conservative_rec['gb_dep64_c1_tps']:.1f} TPS/user. The {aggressive_rec['gb_dep64_c1_tps']:.1f}-TPS/user DSA result is an optimistic quality and implementation frontier, not the central recommendation.",
        "- Under controlled size and active-work budgets, depth is the strongest architectural penalty. Native Kimi-trend scaling is slow primarily because it grows to 116 sequential layers; sparse attention alone does not erase that cost.",
        "- V4 compressed attention gives the strongest combined latency/capacity result. DSA gives the lowest modeled attention latency, while KDA gives the flattest context scaling; both involve larger quality or implementation uncertainty when used for most layers.",
        f"- The 22/8 HBM-bandwidth substitution yields about {balanced_rec['vr_hbm_gain']:.2f}x on the balanced design. Doubling directional fabric adds only {balanced_rec['vr_fabric_increment']:.2f}x at c/r=1 but {balanced_rec['vr_fabric_increment_c4']:.2f}x at c/r=4.",
        f"- With the common DSpark AL curve, the spec-aware balanced frontier switches to {balanced_frontier_c1['attention']} at c/r=1 ({balanced_frontier_c1['spec_tps_user']:.1f} TPS/user, nextn={balanced_frontier_c1['best_nextn']}); V4 compressed attention remains best at c/r=4 ({balanced_spec_c4['spec_tps_user']:.1f}, nextn={balanced_spec_c4['best_nextn']}).",
        "", "## Recommended architectures", "",
    ]
    rec_rows = []
    for category, r in recs.items():
        rec_rows.append([
            category, r["plausibility_tier"], str(r["depth"]), r["body_kind"], r["active_tier"],
            r["attention"], f"{r['estimated_active_b']:.0f}B", f"{r['gb_dep64_c1_tps']:.1f}",
            f"{r['gb_dep64_c4_tps']:.1f}", f"{r['vr_hbm_dep64_c1_tps']:.1f}",
            f"{100*r['context_retention_1m_over_32k']:.0f}%", f"{r['memory_gib_dep64_c1']:.1f}",
            str(r["max_cpr_90pct_dep64"]),
        ])
    lines += md_table(
        ["Category", "Tier", "L", "MoE", "Active tier", "Attention", "Active", "GB c1", "GB c4", "VR-HBM c1", "1M retention", "GiB", "90% max c/r"], rec_rows
    )
    lines += ["", "TPS columns are TPS/user at DEP64 and 256K. The aggressive row is a hardware floor, not a quality recommendation. The 90% maximum is an extrapolated capacity boundary, not a latency optimum.", ""]

    lines += ["## Native scaled-architecture anchors", ""]
    anchor_rows = []
    anchor_ids = sorted({r["anchor_id"] for r in anchors})
    def anchor_one(anchor_id: str, system: str, context: int, dep: int, cpr: int):
        return next(r for r in anchors if r["anchor_id"] == anchor_id and r["system"] == system and r["context"] == context and r["dep"] == dep and r["cpr"] == cpr)
    for anchor_id in anchor_ids:
        gb1 = anchor_one(anchor_id, "GB300", 262_144, 64, 1)
        gb4 = anchor_one(anchor_id, "GB300", 262_144, 64, 4)
        vr1 = anchor_one(anchor_id, "VR200 HBM", 262_144, 64, 1)
        long = anchor_one(anchor_id, "GB300", 1_048_576, 64, 1)
        short = anchor_one(anchor_id, "GB300", 32_768, 64, 1)
        anchor_rows.append([
            gb1["label"], gb1["family"], str(gb1["depth"]), f"{gb1['total_t']:.3f}T",
            f"{gb1['active_b']:.0f}B", f"{gb1['tps_user']:.1f}", f"{gb4['tps_user']:.1f}",
            f"{vr1['tps_user']:.1f}", f"{100*long['tps_user']/short['tps_user']:.0f}%", f"{gb1['memory_gib']:.1f}",
        ])
    lines += md_table(["Anchor", "Family", "L", "Total", "Active", "GB c1", "GB c4", "VR c1", "1M retention", "GiB"], anchor_rows)
    lines += [
        "", "These are direct AIC executions of structurally scaled native families under the same quantization and workload assumptions. They are comparison anchors, not members of the all-else-equal synthetic search; family-specific hidden sizes, MoE layouts, and attention stacks remain intact.", "",
    ]

    lines += ["## Speculative decoding with the common DSpark AL curve", ""]
    depths = list(range(9))
    lines += md_table(
        ["nextn", *[str(n) for n in depths]],
        [["InferenceX golden AL", *[f"{al_curve[n]:.2f}" for n in depths]]],
    )
    lines += [
        "", "AL includes the guaranteed target token, so AIC receives `nextn_accepted = AL - 1`. For every architecture/workload point, the selected row is the fastest memory-fitting depth among no speculation and nextn 1–8.", "",
        "### Recommended designs on GB300, DEP64, 256K", "",
    ]
    spec_table = []
    for category, rec in recs.items():
        for cpr in (1, 2, 4, 8):
            s = spec_one(rec["architecture_id"], "GB300", 262_144, 64, cpr)
            spec_table.append([
                category, str(cpr), f"{s['no_spec_tps_user']:.1f}", str(s["best_nextn"]),
                f"{s['acceptance_length']:.2f}", f"{s['spec_tps_user']:.1f}",
                f"{s['gain']:.2f}x", f"{s['memory_gib']:.1f}",
            ])
    lines += md_table(["Design", "c/r", "No spec", "Best nextn", "AL", "Spec TPS/user", "Gain", "GiB/rank"], spec_table)

    lines += ["", "### Speculative performance frontiers", ""]
    frontier_rows = []
    for cpr in (1, 2, 4, 8):
        candidates = [
            r for r in spec_best
            if r["system"] == "GB300" and r["context"] == 262_144 and r["dep"] == 64
            and r["cpr"] == cpr and not r["oom"]
        ]
        for tier_label, tier in (("Overall", None), ("Balanced", "balanced"), ("Conservative", "conservative")):
            eligible = candidates if tier is None else [r for r in candidates if r["plausibility_tier"] == tier]
            best = max(eligible, key=lambda r: r["spec_tps_user"])
            frontier_rows.append([
                tier_label, str(cpr), str(best["depth"]), best["body_kind"], best["attention"],
                f"{best['estimated_active_b']:.0f}B", str(best["best_nextn"]),
                f"{best['acceptance_length']:.2f}", f"{best['spec_tps_user']:.1f}", f"{best['gain']:.2f}x",
            ])
    lines += md_table(["Tier", "c/r", "L", "MoE", "Attention", "Active", "nextn", "AL", "Spec TPS/user", "Gain"], frontier_rows)
    lines += [
        "", "Speculation changes the architecture ordering: shared-index DSA is the modeled low-c/r frontier, while V4 compressed attention wins at c/r=4–8 because wider verification penalizes DSA's indexer and sparse-attention work more strongly. The DSA frontier retains the implementation and quality caveats of the non-spec study.",
    ]

    lines += ["", "### Hardware sensitivity for the balanced and conservative designs", ""]
    spec_hw_rows = []
    for category in ("Balanced recommendation", "Conservative recommendation"):
        rec = recs[category]
        for system in SYSTEMS:
            for cpr in (1, 4):
                s = spec_one(rec["architecture_id"], system, 262_144, 64, cpr)
                spec_hw_rows.append([
                    category.replace(" recommendation", ""), system, str(cpr), f"{s['no_spec_tps_user']:.1f}",
                    str(s["best_nextn"]), f"{s['acceptance_length']:.2f}", f"{s['spec_tps_user']:.1f}", f"{s['gain']:.2f}x",
                ])
    lines += md_table(["Design", "System", "c/r", "No spec", "nextn", "AL", "Spec TPS/user", "Gain"], spec_hw_rows)

    lines += ["", "### Native-family anchors with the same AL assumption", ""]
    native_spec_rows = []
    for anchor_id in anchor_ids:
        for cpr in (1, 4):
            s = next(r for r in native_spec_best if r["anchor_id"] == anchor_id and r["system"] == "GB300" and r["context"] == 262_144 and r["dep"] == 64 and r["cpr"] == cpr)
            native_spec_rows.append([
                s["label"], str(cpr), f"{s['no_spec_tps_user']:.1f}", str(s["best_nextn"]),
                f"{s['acceptance_length']:.2f}", f"{s['spec_tps_user']:.1f}", f"{s['gain']:.2f}x",
            ])
    lines += md_table(["Anchor", "c/r", "No spec", "nextn", "AL", "Spec TPS/user", "Gain"], native_spec_rows)
    lines += [
        "", "The AL curve is measured Kimi-K3 SPEED-Bench coding data from InferenceX and is deliberately held constant here. It is not evidence that another architecture or a scaled 5T checkpoint would achieve the same acceptance. Native Kimi bodies use AIC's K3/DSpark verification path; direct DeepSeek-style bodies use AIC's generic target-verification/MTP geometry while borrowing the DSpark AL. Therefore relative speculative gains outside Kimi-like bodies are sensitivity bounds, not implementation forecasts.", "",
    ]

    lines += ["## What controls performance", ""]
    # Representative body ablation: fixed attention V4, depth 78, balanced unless varied.
    def get(**kwargs):
        candidates = summaries
        for k, v in kwargs.items(): candidates = [r for r in candidates if r[k] == v]
        return max(candidates, key=lambda r: r["gb_dep64_c1_tps"]) if candidates else None
    mechanism_rows = []
    for tier in TARGET_ACTIVE:
        r = get(depth=78, body_kind="direct", active_tier=tier, attention_id="v4_compressed", expert_intermediate=3072)
        if r: mechanism_rows.append([f"Direct active={tier}", f"{r['estimated_active_b']:.0f}B", f"{r['gb_dep64_c1_tps']:.1f}", f"{r['gb_dep64_c4_tps']:.1f}", f"{r['vr_hbm_gain']:.2f}x"])
    for kind in ("direct", "latent"):
        r = get(depth=78, body_kind=kind, active_tier="balanced", attention_id="v4_compressed", expert_intermediate=3072, shared_experts=1)
        if r: mechanism_rows.append([f"{kind.capitalize()} MoE", f"{r['estimated_active_b']:.0f}B", f"{r['gb_dep64_c1_tps']:.1f}", f"{r['gb_dep64_c4_tps']:.1f}", f"{r['vr_hbm_gain']:.2f}x"])
    lines += md_table(["Ablation", "Active", "GB c1", "GB c4", "VR HBM gain"], mechanism_rows)
    lines += [
        "", "Lower routed-active weight budgets reduce single-token weight traffic but increase quality risk. More depth adds sequential attention, routing, normalization, and projection work even when total and active expert weights are held approximately fixed. "
        "Latent MoE halves routed activation/dispatch width but adds replicated hidden↔latent projections and a larger router/expert pool.", "",
    ]

    lines += ["### Expert granularity and shared experts", ""]
    granularity_rows = []
    for inter in (2048, 3072, 4096):
        r = get(depth=78, body_kind="direct", active_tier="balanced", attention_id="v4_compressed", expert_intermediate=inter)
        if r:
            granularity_rows.append([str(inter), str(r["experts"]), str(r["topk"]), f"{r['estimated_active_b']:.0f}B", f"{r['gb_dep64_c1_tps']:.1f}", f"{r['gb_dep64_c4_tps']:.1f}"])
    lines += md_table(["Expert width", "Experts", "top-k", "Active", "GB c1", "GB c4"], granularity_rows)
    shared_rows = []
    for shared in (0, 1, 2):
        r = get(depth=78, body_kind="latent", active_tier="balanced", attention_id="v4_compressed", expert_intermediate=3072, shared_experts=shared)
        if r:
            shared_rows.append([str(shared), str(r["experts"]), str(r["topk"]), f"{r['estimated_active_b']:.0f}B", f"{r['gb_dep64_c1_tps']:.1f}", f"{r['gb_dep64_c4_tps']:.1f}"])
    lines += [""] + md_table(["Latent shared experts", "Routed experts", "top-k", "Active", "GB c1", "GB c4"], shared_rows)
    lines += [
        "", "Narrower experts require more expert selections for the same active budget; wider experts reduce routing fan-out but make each selected expert coarser. Shared experts add dense per-token work on every layer, so their quality/stability benefit has a direct decode cost.", "",
    ]

    lines += ["## Attention tradeoffs on the common 78-layer balanced direct body", ""]
    attn_rows = []
    for aid, profile in ATTENTION_PROFILES.items():
        r = get(depth=78, body_kind="direct", active_tier="balanced", attention_id=aid, expert_intermediate=3072)
        if r:
            attn_rows.append([
                profile["label"], profile["risk"], f"{r['gb_dep64_c1_tps']:.1f}", f"{r['gb_dep64_c4_tps']:.1f}",
                f"{100*r['context_retention_1m_over_32k']:.0f}%", f"{r['vr_hbm_gain']:.2f}x",
                f"{r['memory_gib_dep64_c1']:.1f}", f"{r['attention_share_gb_dep64_c1']:.0f}%",
            ])
    attn_rows.sort(key=lambda x: float(x[2]), reverse=True)
    lines += md_table(["Attention", "Risk", "GB c1", "GB c4", "1M retention", "VR gain", "GiB", "Attn share"], attn_rows)
    lines += [
        "", "KDA has context-independent recurrent state but pays dense projection cost every layer. MLA stores a compact latent KV cache but scans the full context. DSA reads only selected KV entries during attention, yet stores full latent KV plus its indexer cache; smaller DSA top-k therefore improves latency more than capacity. "
        "The shared-index DSA rows assume GLM-style index reuse every four layers. The per-layer-index row removes that reuse and is the conservative control. DeepSeek-V4 compressed attention reduces both reads and stored cache, but relies on a more novel compressor/indexer stack. Mixed KDA+sparse designs trade a bounded recurrent path against explicit token retrieval.", "",
    ]

    lines += ["### Attention architecture decision matrix", ""]
    lines += md_table(
        ["Mechanism", "Decode context work", "Stored state", "Performance upside", "Principal tradeoff"],
        [
            ["MLA", "Full latent-cache scan", "Compact KV/token", "Mature full retrieval", "TPS falls with context"],
            ["KDA", "Fixed recurrent update", "Fixed recurrent state", "Flat long-context cost", "Dense projections; retrieval/quality risk"],
            ["DSA", "Indexer scan + top-k KV", "Full latent KV + index keys", "Fastest modeled sparse read", "Indexer/reuse kernels and selection quality"],
            ["V4 compressed", "Multi-rate compressed scan", "Window + compressed histories", "Best latency/capacity balance", "Novel compressor/indexer stack"],
            ["KDA+sparse", "Bounded path + sparse retrieval", "Recurrent + selected cache", "Robust hybrid failure modes", "Two mechanisms to train and optimize"],
        ],
    )
    lines += [
        "", "The study holds model quality out of scope. In particular, an O(1) or top-k attention path is only a performance win if training preserves the useful long-range information that dense attention would have retrieved.", "",
    ]

    lines += ["## Depth scaling at balanced active work and V4 attention", ""]
    depth_rows = []
    for depth in DEPTHS:
        candidates = [x for x in summaries if x["depth"] == depth and x["body_kind"] == "direct" and x["active_tier"] == "balanced" and x["attention_id"] == "v4_compressed"]
        r = min(candidates, key=lambda x: abs(x["expert_intermediate"] - 3072)) if candidates else None
        if r: depth_rows.append([str(depth), str(r["experts"]), str(r["topk"]), f"{r['estimated_total_t']:.3f}T", f"{r['estimated_active_b']:.0f}B", f"{r['gb_dep64_c1_tps']:.1f}", f"{r['gb_dep64_c4_tps']:.1f}", f"{r['vr_hbm_dep64_c1_tps']:.1f}"])
    lines += md_table(["Layers", "Experts", "top-k", "Total", "Active", "GB c1", "GB c4", "VR c1"], depth_rows)
    lines += ["", "Shallower designs win performance because less non-expert work is sequential. The tradeoff is architectural capacity: increasing experts can preserve total and active parameter budgets, but it does not replace additional transformation depth.", ""]

    lines += ["## Context-length sensitivity", ""]
    context_rows = []
    for label in ("Aggressive latency floor", "Balanced recommendation", "Conservative recommendation", "Full-attention fallback"):
        r = recs[label]
        values = []
        for context in CONTEXTS:
            for cpr in (1, 4):
                point = next(x for x in rows if x["architecture_id"] == r["architecture_id"] and x["system"] == "GB300" and x["context"] == context and x["dep"] == 64 and x["cpr"] == cpr)
                values.append(f"{point['tps_user']:.1f}")
        context_rows.append([label, *values])
    lines += md_table(["Design", "32K c1", "32K c4", "256K c1", "256K c4", "1M c1", "1M c4"], context_rows)
    lines += [
        "", "MLA's cache scan becomes conspicuous at 1M and higher c/r. DSA still scans index keys but limits expensive KV attention to top-k tokens; V4 compresses older history; KDA avoids token-cache reads entirely. Context robustness and memory capacity are separate: native DSA stores a full latent cache even when it reads only top-k values.", "",
    ]

    lines += ["## Parallelism sensitivity", ""]
    parallel_rows = []
    for label in ("Balanced recommendation", "Conservative recommendation"):
        r = recs[label]
        for dep in DEPS:
            for cpr in (1, 4):
                point = next(x for x in rows if x["architecture_id"] == r["architecture_id"] and x["system"] == "GB300" and x["context"] == 262_144 and x["dep"] == dep and x["cpr"] == cpr)
                parallel_rows.append([label.replace(" recommendation", ""), str(dep), str(cpr), str(dep*cpr), f"{point['tps_user']:.1f}", f"{point['tps_gpu']:.1f}", f"{point['memory_gib']:.1f}"])
    lines += md_table(["Design", "DEP", "c/r", "Global conc.", "TPS/user", "TPS/GPU", "GiB/rank"], parallel_rows)
    lines += [
        "", "Increasing DEP shards routed experts and their weights over more ranks, but it does not remove sequential attention, projections, routing, normalization, or communication. Raising c/r improves weight reuse and aggregate TPS/GPU until KV/state traffic, activation work, or communication dominates; TPS/user usually falls.", "",
    ]

    lines += ["## Rubin sensitivity", ""]
    rubin_rows = []
    for category, r in recs.items():
        rubin_rows.append([
            category, f"{r['gb_dep64_c1_tps']:.1f}", f"{r['vr_hbm_dep64_c1_tps']:.1f}", f"{r['vr_hbm_gain']:.2f}x",
            f"{r['vr_hbm_fabric_dep64_c1_tps']:.1f}", f"{r['vr_fabric_increment']:.2f}x",
            f"{r['vr_hbm_dep64_c4_tps']:.1f}", f"{r['vr_hbm_fabric_dep64_c4_tps']:.1f}", f"{r['vr_fabric_increment_c4']:.2f}x",
        ])
    lines += md_table(["Architecture", "GB c1", "VR-HBM c1", "HBM gain", "+fabric c1", "Fabric gain", "VR-HBM c4", "+fabric c4", "Fabric gain c4"], rubin_rows)
    lines += ["", "HBM-heavy expert and projection work approaches the 22/8 bandwidth ratio. Architectures with larger attention-compute or recurrent-kernel shares gain less. Fabric matters more as c/r raises dispatch volume; these are still isolated specification substitutions, not a complete Rubin implementation model.", ""]

    lines += ["## Source provenance", ""]
    lines += [
        "- AIC model implementation: the local `repos/aiconfigurator-kimi-k3-pr1435` checkout; its exact commit is recorded in `metadata.json`.",
        "- SGLang simulator feature context: [sglang PR #33824](https://github.com/sgl-project/sglang/pull/33824).",
        "- GB300 hardware inputs: AIC's checked-in `aiconfigurator_core/systems/gb300.yaml`.",
        "- VR200 hardware inputs: [NVIDIA Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) and [NVIDIA's Rubin architecture article](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/).",
        "- Native-family anchor configs: the existing Kimi/DeepSeek scaling experiments plus AIC's checked-in NVIDIA GLM-5.2-NVFP4 config.",
        "- Speculative acceptance: InferenceX `golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml`; exact repository commit is recorded in `metadata.json`.",
        "",
    ]

    lines += ["## Method and limitations", ""]
    lines += [
        "- All nonzero direct body components must report SOL provenance; zero-time TP1 collectives may retain an empirical label.",
        "- DSA top-k and shared-index variants extend AIC's published DSA SOL equations because the model parser fixes native GLM dimensions/top-2048. These rows are labeled in the raw provenance column.",
        f"- Native reconstruction validation passed: maximum no-spec Kimi/V4 latency error {max(validation['direct_latency_ms'], validation['latent_latency_ms']):.3g} ms, maximum speculative Kimi/V4 error {max(validation['direct_spec_latency_ms'], validation['latent_spec_latency_ms']):.3g} ms, DSA no-reuse error {validation['dsa_aic_vs_formula_ms']:.3g} ms, and maximum memory-component error {max(v for k, v in validation.items() if k.endswith('_bytes')):.3g} bytes.",
        "- Composite memory replaces native attention weights and KV using AIC's architecture formulas. Maximum c/r is a linear capacity estimate; use the 90% boundary operationally.",
        "- Speculative Kimi-like composites preserve the additional DSpark verification/cache allocation reported by AIC; native-profile timing and memory reconstruct the direct AIC run exactly.",
        "- VR200 cases retain GB300 tensor compute and kernel efficiencies. They isolate HBM and NVLink specifications and are not full Rubin performance predictions.",
        "- Low active ratios, shallow depth, all-linear attention, and narrow sparse retrieval may reduce quality. No performance model can quantify that loss.",
        "- SOL is a lower-bound hardware model. Achievable serving throughput will be lower and architecture-dependent.",
        "", "## Artifacts", "",
        "- `all_results.csv`: every architecture/workload/hardware result.",
        "- `architecture_summary.csv`: one-row primary metrics per architecture.",
        "- `pareto_front.csv`: non-dominated architectures.",
        "- `body_manifest.csv`: exact generated MoE bodies.",
        "- `native_anchors.csv`: direct native-family scaled-model comparisons.",
        "- `spec_all_results.csv`: every depth 1–8 for every controlled architecture/workload point.",
        "- `spec_best.csv`: fastest memory-fitting depth, including no-spec fallback.",
        "- `spec_architecture_summary.csv`: primary GB300 speculative metrics per architecture.",
        "- `native_anchor_spec_results.csv` and `native_anchor_spec_best.csv`: native-family speculative sensitivity.",
        "- `estimate_cache.jsonl`: direct AIC evidence and operation breakdowns.",
        "",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    al_curve = load_dspark_al_curve()
    specs = body_specs()
    refs = make_models(specs)
    anchor_specs = make_native_anchors()
    dsa_validation_model = make_dsa_validation_model()
    models = [s.model_path for s in specs] + list(refs.values()) + [a["model_path"] for a in anchor_specs.values()] + [dsa_validation_model]
    cache = run_estimates(models, points(), al_curve)
    bad = [r for r in cache.values() if r["model"] in models and r["non_sol_ms"] > 1e-12]
    if bad:
        raise RuntimeError(f"{len(bad)} direct estimates contain non-SOL latency")
    validation = validate_reconstruction(specs, refs, cache, dsa_validation_model, al_curve)
    rows = compose_rows(specs, refs, cache)
    invalid = [r for r in rows if not (4.85 <= r["estimated_total_t"] <= 5.10)]
    if invalid:
        raise RuntimeError(f"{len(invalid)} composed rows fall outside the 4.85T--5.10T gate")
    summaries = architecture_summary(rows)
    front = pareto(summaries)
    recs = pick_recommendations(summaries)
    anchors = native_anchor_rows(anchor_specs, cache)
    spec_rows = compose_spec_rows(specs, refs, dsa_validation_model, cache, al_curve, rows)
    spec_best = best_spec_rows(rows, spec_rows)
    spec_summaries = spec_architecture_summary(spec_best)
    anchor_spec_rows = native_anchor_spec_rows(anchor_specs, cache, al_curve)
    anchor_spec_best = best_native_anchor_spec_rows(anchors, anchor_spec_rows)
    write_csv(OUTPUT / "all_results.csv", rows)
    write_csv(OUTPUT / "architecture_summary.csv", summaries)
    write_csv(OUTPUT / "pareto_front.csv", front)
    write_csv(OUTPUT / "body_manifest.csv", [asdict(s) for s in specs])
    write_csv(OUTPUT / "native_anchors.csv", anchors)
    write_csv(OUTPUT / "spec_all_results.csv", spec_rows)
    write_csv(OUTPUT / "spec_best.csv", spec_best)
    write_csv(OUTPUT / "spec_architecture_summary.csv", spec_summaries)
    write_csv(OUTPUT / "native_anchor_spec_results.csv", anchor_spec_rows)
    write_csv(OUTPUT / "native_anchor_spec_best.csv", anchor_spec_best)
    write_report(specs, rows, summaries, recs, anchors, validation, spec_best, anchor_spec_best, al_curve)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=AIC_REPO, text=True, capture_output=True, check=True).stdout.strip()
    inferencex_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=INFERENCEX_REPO, text=True, capture_output=True, check=True).stdout.strip()
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "aic_commit": commit,
        "body_count": len(specs), "attention_profile_count": len(ATTENTION_PROFILES),
        "architecture_count": len(summaries), "result_count": len(rows), "pareto_count": len(front),
        "native_anchor_count": len(anchor_specs), "native_anchor_result_count": len(anchors),
        "spec_result_count": len(spec_rows), "spec_best_count": len(spec_best),
        "spec_architecture_summary_count": len(spec_summaries),
        "native_anchor_spec_result_count": len(anchor_spec_rows),
        "native_anchor_spec_best_count": len(anchor_spec_best),
        "dspark_acceptance_curve": al_curve, "inferencex_al_source": str(AL_SOURCE),
        "inferencex_commit": inferencex_commit,
        "reconstruction_validation": validation,
        "systems": SYSTEMS, "contexts": CONTEXTS, "deps": DEPS,
        "recommendations": {k: v["architecture_id"] for k, v in recs.items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Bodies: {len(specs)}; architectures: {len(summaries)}; no-spec rows: {len(rows)}; spec rows: {len(spec_rows)}; Pareto: {len(front)}")
    print(f"Report: {OUTPUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
