#!/usr/bin/env python3
"""Generate controlled 5T/10T Kimi-K3-family model configurations."""

from __future__ import annotations

import copy
import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
SIMULATOR_DIR = EXPERIMENT_DIR.parents[1]
AIC_REPO = SIMULATOR_DIR / "repos" / "aiconfigurator-kimi-k3-pr1435"
BASE_CONFIG = (
    AIC_REPO
    / "aic-core"
    / "src"
    / "aiconfigurator_core"
    / "model_configs"
    / "moonshotai--Kimi-K3_config.json"
)
MODELS_DIR = EXPERIMENT_DIR / "models"


VARIANTS = (
    {
        "id": "k3_2p8t",
        "label": "K3 2.8T baseline",
        "family": "baseline",
        "description": "Unmodified Kimi-K3 text geometry.",
    },
    {
        "id": "k3_5t_expert",
        "label": "~5T expert-only",
        "family": "expert_only",
        "num_experts": 1600,
        "description": "Adds routed experts only; top-16 and all active geometry remain K3-sized.",
    },
    {
        "id": "k3_10t_expert",
        "label": "~10T expert-only",
        "family": "expert_only",
        "num_experts": 3200,
        "description": "Adds routed experts only; top-16 and all active geometry remain K3-sized.",
    },
    {
        "id": "k3_5t_trend",
        "label": "~5T Kimi-trend",
        "family": "kimi_trend",
        "num_layers": 116,
        "num_experts": 1280,
        "topk": 24,
        "num_heads": 128,
        "num_shared_experts": 3,
        "dense_intermediate_size": 45056,
        "description": "Continues K2.5→K3-style compound growth in depth, experts, active top-k, heads, and shared experts.",
    },
    {
        "id": "k3_10t_trend",
        "label": "~10T Kimi-trend",
        "family": "kimi_trend",
        "num_layers": 144,
        "num_experts": 2048,
        "topk": 32,
        "num_heads": 192,
        "num_shared_experts": 4,
        "dense_intermediate_size": 57344,
        "description": "Continues K2.5→K3-style compound growth; top-k is conservatively capped at 32 rather than extrapolated to about 40.",
    },
    {
        "id": "k3_5t_depth",
        "label": "~5T depth-only",
        "family": "depth_only",
        "num_layers": 166,
        "description": "Increases layer count only while preserving K3 layer geometry and expert count.",
    },
    {
        "id": "k3_10t_depth",
        "label": "~10T depth-only",
        "family": "depth_only",
        "num_layers": 332,
        "description": "Increases layer count only; an active-compute-heavy sensitivity bound, not the expected design.",
    },
)


def _set_layer_pattern(text: dict, layers: int) -> None:
    # K3 repeats three KDA layers then one MLA layer. Preserve that 3:1 motif,
    # and make the final layer MLA when it would otherwise end on KDA.
    full = [layer for layer in range(1, layers + 1) if layer % 4 == 0]
    if layers not in full:
        full.append(layers)
    full_set = set(full)
    kda = [layer for layer in range(1, layers + 1) if layer not in full_set]
    text["num_hidden_layers"] = layers
    text["linear_attn_config"]["full_attn_layers"] = full
    text["linear_attn_config"]["kda_layers"] = kda


def _parameter_accounting(text: dict) -> dict[str, float | int]:
    layers = int(text["num_hidden_layers"])
    h = int(text["hidden_size"])
    vocab = int(text["vocab_size"])
    kda_cfg = text["linear_attn_config"]
    kda_layers = len(kda_cfg["kda_layers"])
    mla_layers = layers - kda_layers
    heads = int(text["num_attention_heads"])
    kda_heads = int(kda_cfg["num_heads"])
    kda_hd = int(kda_cfg["head_dim"])
    p = kda_heads * kda_hd
    q_rank = int(text["q_lora_rank"])
    kv_rank = int(text["kv_lora_rank"])
    qk_nope = int(text["qk_nope_head_dim"])
    qk_rope = int(text["qk_rope_head_dim"])
    v_dim = int(text["v_head_dim"])
    dense_layers = int(text["first_k_dense_replace"])
    moe_layers = layers - dense_layers
    experts = int(text["num_experts"])
    topk = int(text["num_experts_per_token"])
    moe_inter = int(text["moe_intermediate_size"])
    latent = int(text["routed_expert_hidden_size"])
    shared = int(text["num_shared_experts"])

    embedding_and_logits = 2 * vocab * h
    kda_per_layer = (
        4 * p * h
        + (kda_heads + kda_hd) * h
        + p * kda_hd
        + h * p
    )
    mla_per_layer = (
        (q_rank + kv_rank + qk_rope) * h
        + heads * (qk_nope + qk_rope) * q_rank
        + heads * (qk_nope + v_dim) * kv_rank
        + 2 * heads * v_dim * h
    )
    dense = dense_layers * 3 * h * int(text["intermediate_size"])
    router = moe_layers * h * experts
    latent_projections = moe_layers * 2 * h * latent
    routed_all = moe_layers * experts * 3 * latent * moe_inter
    routed_active = moe_layers * topk * 3 * latent * moe_inter
    shared_params = moe_layers * 3 * h * shared * moe_inter
    total = (
        embedding_and_logits
        + kda_layers * kda_per_layer
        + mla_layers * mla_per_layer
        + dense
        + router
        + latent_projections
        + routed_all
        + shared_params
    )
    active = total - routed_all + routed_active
    # Matches the study's AIC quantization assumption: routed experts at
    # MXFP4 payload width, attention/dense/shared/router weights at BF16.
    packed_bytes = routed_all * 0.5 + (total - routed_all) * 2.0
    return {
        "text_parameters": total,
        "text_parameters_trillion": total / 1e12,
        "active_parameters_per_token": active,
        "active_parameters_billion": active / 1e9,
        "routed_expert_parameters": routed_all,
        "routed_expert_fraction": routed_all / total,
        "approx_packed_weight_bytes": packed_bytes,
        "approx_packed_weight_gib": packed_bytes / (1 << 30),
        "num_layers": layers,
        "kda_layers": kda_layers,
        "mla_layers": mla_layers,
        "num_experts": experts,
        "topk": topk,
        "num_heads": heads,
        "num_shared_experts": shared,
    }


def main() -> None:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for variant in VARIANTS:
        config = copy.deepcopy(base)
        text = config["text_config"]
        if "num_layers" in variant:
            _set_layer_pattern(text, int(variant["num_layers"]))
        if "num_experts" in variant:
            text["num_experts"] = int(variant["num_experts"])
        if "topk" in variant:
            text["num_experts_per_token"] = int(variant["topk"])
        if "num_heads" in variant:
            heads = int(variant["num_heads"])
            text["num_attention_heads"] = heads
            text["num_key_value_heads"] = heads
            text["linear_attn_config"]["num_heads"] = heads
        if "num_shared_experts" in variant:
            text["num_shared_experts"] = int(variant["num_shared_experts"])
        if "dense_intermediate_size" in variant:
            text["intermediate_size"] = int(variant["dense_intermediate_size"])

        model_dir = MODELS_DIR / str(variant["id"])
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metadata = {
            **variant,
            **_parameter_accounting(text),
            "model_path": str(model_dir),
            "base_config": str(BASE_CONFIG),
        }
        (model_dir / "variant_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest.append(metadata)
        print(
            f"{metadata['label']}: {metadata['text_parameters_trillion']:.3f}T total, "
            f"{metadata['active_parameters_billion']:.1f}B active, "
            f"{metadata['approx_packed_weight_gib']:.1f} GiB packed"
        )

    (MODELS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
