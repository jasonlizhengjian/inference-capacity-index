#!/usr/bin/env python3
"""AIC SOL attention-only ablation: Kimi KDA/MLA versus DeepSeek-V4 sparse attention."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aic-attention-only-mpl")

from aiconfigurator.cli.api import cli_estimate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KIMI_SOURCE = ROOT / "experiments/kimi-vs-deepseek-5t-controlled/models/kimi_moe_normalized/config.json"
COMMON_BODY = ROOT / "experiments/kimi-vs-deepseek-5t-controlled/models/deepseek_moe_normalized"
KIMI_ATTENTION_MODEL = HERE / "models/kimi_attention_61_layers"
OUTPUT = HERE / "output"
CSV_PATH = OUTPUT / "attention_only_sol.csv"
REPORT_PATH = OUTPUT / "REPORT.md"

LAYERS = 61
CONTEXT = 262_144
SYSTEMS = {"GB300": "gb300", "B300": "b300_sxm"}
DEPS = (16, 64)
CPRS = (1, 2, 3, 4)


def make_kimi_attention_model() -> None:
    config = json.loads(KIMI_SOURCE.read_text(encoding="utf-8"))
    text = config["text_config"]
    text["num_hidden_layers"] = LAYERS
    full = [layer for layer in range(1, LAYERS + 1) if layer % 4 == 0]
    if LAYERS not in full:
        full.append(LAYERS)
    full_set = set(full)
    text["linear_attn_config"]["full_attn_layers"] = full
    text["linear_attn_config"]["kda_layers"] = [layer for layer in range(1, LAYERS + 1) if layer not in full_set]
    KIMI_ATTENTION_MODEL.mkdir(parents=True, exist_ok=True)
    (KIMI_ATTENTION_MODEL / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(model: Path, system: str, dep: int, cpr: int):
    return cli_estimate(
        str(model), SYSTEMS[system], mode="static_gen", backend_name="sglang",
        backend_version="estimate", database_mode="SOL", isl=CONTEXT, osl=2,
        prefix=CONTEXT, batch_size=cpr, tp_size=1, pp_size=1,
        attention_dp_size=dep, moe_tp_size=1, moe_ep_size=dep,
        gemm_quant_mode="fp8_block", moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
        kvcache_quant_mode="fp8", fmha_quant_mode="bfloat16",
        nextn=0, stride=1, engine_step_backend="python",
    )


def latency(result) -> dict[str, float]:
    return {str(k): float(v) for k, v in result.summary.get_generation_latency_dict().items()}


def kimi_attention_ops(values: dict[str, float]) -> dict[str, float]:
    return {
        name: value for name, value in values.items()
        if name.startswith("generation_kda_")
        or name.startswith("generation_mla_")
        or name.startswith("generation_bmm_")
        or name in {"generation_attention", "generation_attn_res"}
    }


def deepseek_attention_ops(values: dict[str, float]) -> dict[str, float]:
    return {
        name: value for name, value in values.items()
        if name in {
            "generation_attention", "generation_mhc_pre", "generation_mhc_post",
            "generation_attn_norm",
        }
    }


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if i else "---" for i in range(len(headers))) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def main() -> None:
    make_kimi_attention_model()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for system in SYSTEMS:
        for dep in DEPS:
            for cpr in CPRS:
                ds_result = run(COMMON_BODY, system, dep, cpr)
                kimi_result = run(KIMI_ATTENTION_MODEL, system, dep, cpr)
                ds_latency = latency(ds_result)
                kimi_latency = latency(kimi_result)
                ds_attn_parts = deepseek_attention_ops(ds_latency)
                kimi_attn_parts = kimi_attention_ops(kimi_latency)
                ds_attn = sum(ds_attn_parts.values())
                kimi_attn = sum(kimi_attn_parts.values())
                common = float(ds_result.tpot) - ds_attn
                ds_total = common + ds_attn
                kimi_total = common + kimi_attn
                sources = {
                    str(k): str(v or "unknown").lower()
                    for k, v in ds_result.summary.get_generation_source_dict().items()
                }
                kimi_sources = {
                    str(k): str(v or "unknown").lower()
                    for k, v in kimi_result.summary.get_generation_source_dict().items()
                }
                if any(value > 0 and sources.get(k) != "sol" for k, value in ds_attn_parts.items()):
                    raise RuntimeError("Non-SOL DeepSeek attention component")
                if any(value > 0 and kimi_sources.get(k) != "sol" for k, value in kimi_attn_parts.items()):
                    raise RuntimeError("Non-SOL Kimi attention component")
                rows.append({
                    "system": system, "dep": dep, "concurrency_per_rank": cpr,
                    "global_concurrency": dep * cpr, "layers": LAYERS,
                    "context_tokens": CONTEXT, "common_body_ms": common,
                    "kimi_attention_ms": kimi_attn, "deepseek_attention_ms": ds_attn,
                    "kimi_attention_tpot_ms": kimi_total, "deepseek_attention_tpot_ms": ds_total,
                    "kimi_attention_tps_user": 1000 / kimi_total,
                    "deepseek_attention_tps_user": 1000 / ds_total,
                    "deepseek_over_kimi_tps": kimi_total / ds_total,
                    "kimi_attention_parts": json.dumps(kimi_attn_parts, sort_keys=True),
                    "deepseek_attention_parts": json.dumps(ds_attn_parts, sort_keys=True),
                })
    with CSV_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    values = []
    for row in rows:
        values.append([
            row["system"], str(row["dep"]), str(row["concurrency_per_rank"]),
            f"{row['common_body_ms']:.2f}", f"{row['kimi_attention_ms']:.2f}",
            f"{row['deepseek_attention_ms']:.2f}", f"{row['kimi_attention_tps_user']:.1f}",
            f"{row['deepseek_attention_tps_user']:.1f}", f"{row['deepseek_over_kimi_tps']:.2f}x",
        ])
    lines = [
        "# Attention-architecture-only AIC SOL ablation", "",
        "Synthetic 61-layer models with the same DeepSeek-V4 5T common body: hidden size 7168, "
        "1216 routed experts, top-25, one shared expert, intermediate size 3072, FP8 trunk, "
        "FP4 routed experts, and identical embedding/logits/MoE/norm work. The only substituted "
        "operation set is attention: 45 Kimi KDA + 16 Kimi MLA layers versus 61 DeepSeek "
        "SWA/CSA/HCA compressed-attention layers (including its mHC residual operations).", "",
        "AIC SOL, SGLang, 256K resident-cache decode-only, FP8 KV, BF16 attention kernels, no speculative decoding, "
        "TP1/PP1/attention-DP=DEP/MoE-EP=DEP.", "",
        "AIC couples attention implementations to model classes. This report therefore uses an operation-level "
        "composite: common body latency = full DeepSeek latency minus its attention operation set; each attention "
        "operation set is then added to that same body. This is exact under AIC's additive SOL latency model, "
        "but it is a synthetic architectural ablation rather than a runnable checkpoint configuration.", "",
    ]
    lines += table(
        ["System", "DEP", "c/r", "Common body ms", "Kimi attn ms", "DS attn ms", "Kimi TPS/u", "DS TPS/u", "DS/Kimi"],
        values,
    )
    lines += ["", "Memory-full boundaries are intentionally omitted: AIC does not expose per-operation weight/cache memory, so the composite's memory cannot be assembled with the same rigor as its latency.", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(rows)} comparisons to {CSV_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
