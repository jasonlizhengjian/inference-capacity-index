#!/usr/bin/env python3
"""Hypothetical 288-GPU rack-local DEP sensitivity for the 10T Kimi trend."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from aiconfigurator.cli.api import cli_estimate


HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "k3_10t_trend"
OUTPUT = HERE / "output"
CSV_PATH = OUTPUT / "kimi_k3_10t_nvl288_dep_sensitivity.csv"
REPORT_PATH = OUTPUT / "KIMI_K3_10T_NVL288_DEP_SENSITIVITY.md"
METADATA_PATH = OUTPUT / "kimi_k3_10t_nvl288_dep_metadata.json"
SYSTEM_PATHS = f"{HERE / 'systems'},{HERE.parent / 'kimi-k3-gb300-decode' / 'systems'},default"
SYSTEMS = (("GB300 72-GPU rack", "gb300"), ("Hypothetical 288-GPU NVLink domain", "gb300_nvl288"))
DEP_SIZES = (64, 128, 256)
LOADS = (1, 2, 3, 4)
CONTEXT = 262_144


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    variant = json.loads((MODEL / "variant_metadata.json").read_text(encoding="utf-8"))
    routed_gib = float(variant["routed_expert_parameters"]) * 0.5 / (1 << 30)
    non_routed_bf16_gib = float(variant["approx_packed_weight_gib"]) - routed_gib
    rows = []
    for system_label, system_name in SYSTEMS:
        for dep in DEP_SIZES:
            for load in LOADS:
                result = cli_estimate(
                    str(MODEL),
                    system_name,
                    mode="static_gen",
                    backend_name="trtllm",
                    backend_version="estimate",
                    database_mode="SOL",
                    isl=CONTEXT,
                    osl=2,
                    batch_size=load,
                    tp_size=1,
                    pp_size=1,
                    attention_dp_size=dep,
                    moe_tp_size=1,
                    moe_ep_size=dep,
                    kvcache_quant_mode="bfloat16",
                    nextn=0,
                    stride=1,
                    engine_step_backend="python",
                    systems_paths=SYSTEM_PATHS,
                )
                memory = result.summary.get_memory()
                latency = result.summary.get_generation_latency_dict()
                tps_user = 1000.0 / float(result.tpot)
                expert_shard_gib = routed_gib / dep
                fp8_non_routed_weights_gib = non_routed_bf16_gib / 2.0 + expert_shard_gib
                fp8_total_gib = (
                    fp8_non_routed_weights_gib
                    + float(memory["kvcache"])
                    + float(memory["activations"])
                    + float(memory["nccl"])
                    + float(memory["others"])
                )
                rows.append(
                    {
                        "system": system_label,
                        "system_name": system_name,
                        "dep_size": dep,
                        "concurrency_per_rank": load,
                        "global_concurrency": dep * load,
                        "tpot_ms": float(result.tpot),
                        "tokens_s_user": tps_user,
                        "tokens_s_total": dep * load * tps_user,
                        "tokens_s_gpu": load * tps_user,
                        "moe_ms": float(latency.get("generation_moe", 0.0)),
                        "attention_ms": float(latency.get("generation_attention", 0.0)),
                        "non_moe_ms": float(result.tpot) - float(latency.get("generation_moe", 0.0)),
                        "memory_gib_per_rank": float(memory["total"]),
                        "weights_gib_per_rank": float(memory["weights"]),
                        "cache_gib_per_rank": float(memory["kvcache"]),
                        "oom": bool(result.summary.check_oom()),
                        "estimated_fp8_non_routed_weights_gib": fp8_non_routed_weights_gib,
                        "estimated_fp8_total_gib": fp8_total_gib,
                        "estimated_fp8_fits": fp8_total_gib <= 277.6,
                    }
                )

    with CSV_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    custom = [row for row in rows if row["system_name"] == "gb300_nvl288"]
    lines = [
        "# Hypothetical 288-GPU NVLink-domain DEP sensitivity for 10T Kimi-trend",
        "",
        "This keeps GB300 compute, 8-TB/s HBM, 277.6-GiB usable capacity, and 900-GB/s-per-GPU "
        "rack-local fabric unchanged while extending the rack-local domain from 72 to 288 GPUs. It is "
        "formula-only AIC SOL, not a proposed or real NVIDIA topology.",
        "",
        "| DEP | c/r | Global | TPS/user | TPS/GPU | Aggregate TPS | TPOT | MoE | Current GiB/rank | "
        "FP8 non-routed estimate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in custom:
        lines.append(
            f"| {row['dep_size']} | {row['concurrency_per_rank']} | {row['global_concurrency']} | "
            f"{row['tokens_s_user']:.1f} | {row['tokens_s_gpu']:.1f} | {row['tokens_s_total']:.0f} | "
            f"{row['tpot_ms']:.2f} ms | {row['moe_ms']:.2f} ms | {row['memory_gib_per_rank']:.1f} "
            f"({'OOM' if row['oom'] else 'fits'}) | {row['estimated_fp8_total_gib']:.1f} "
            f"({'fits' if row['estimated_fp8_fits'] else 'OOM'}) |"
        )
    dep64 = next(row for row in custom if row["dep_size"] == 64 and row["concurrency_per_rank"] == 1)
    dep128 = next(row for row in custom if row["dep_size"] == 128 and row["concurrency_per_rank"] == 1)
    dep256 = next(row for row in custom if row["dep_size"] == 256 and row["concurrency_per_rank"] == 1)
    floor_ms = min(row["non_moe_ms"] for row in custom if row["concurrency_per_rank"] == 1)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"At c/r=1, DEP128 improves TPS/user by {(dep128['tokens_s_user'] / dep64['tokens_s_user'] - 1) * 100:.1f}% "
            f"over DEP64; DEP256 improves it by {(dep256['tokens_s_user'] / dep64['tokens_s_user'] - 1) * 100:.1f}%. "
            "Aggregate throughput grows much more because the cluster admits proportionally more requests.",
            "",
            f"The non-MoE floor is about {floor_ms:.1f} ms/token, so even free routed-MoE compute would cap this "
            f"TP1 DEP design near {1000 / floor_ms:.1f} TPS/user. Wider EP cannot shard the active KDA/MLA, "
            "latent, shared-expert, dense, router, or output work.",
            "",
            "All native-precision rows remain OOM because approximately 303 GiB of non-routed BF16 weights "
            "are replicated under TP1. The FP8 column is arithmetic only: it halves that component without "
            "changing timing. It is not a verified FP8-kernel simulation.",
            "",
            f"Raw data: [`{CSV_PATH.name}`]({CSV_PATH.name}); metadata: "
            f"[`{METADATA_PATH.name}`]({METADATA_PATH.name}).",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    METADATA_PATH.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": str(MODEL),
                "context": CONTEXT,
                "dep_sizes": DEP_SIZES,
                "loads": LOADS,
                "systems": SYSTEMS,
                "rows": len(rows),
                "database_mode": "SOL",
                "backend": "trtllm",
                "non_routed_bf16_gib": non_routed_bf16_gib,
                "routed_mxfp4_gib_total": routed_gib,
                "fp8_memory_is_arithmetic_only": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {CSV_PATH}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
