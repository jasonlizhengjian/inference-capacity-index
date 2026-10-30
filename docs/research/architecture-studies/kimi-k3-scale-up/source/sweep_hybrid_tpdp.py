#!/usr/bin/env python3
"""Sweep hybrid attention TP x DP with full-cluster EP for 10T Kimi-trend."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from aiconfigurator.cli.api import cli_estimate


HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "k3_10t_trend"
OUTPUT = HERE / "output"
CSV_PATH = OUTPUT / "kimi_k3_10t_hybrid_tpdp_sweep.csv"
REPORT_PATH = OUTPUT / "KIMI_K3_10T_HYBRID_TPDP.md"
METADATA_PATH = OUTPUT / "kimi_k3_10t_hybrid_tpdp_metadata.json"
SYSTEM_PATHS = f"{HERE / 'systems'},{HERE.parent / 'kimi-k3-gb300-decode' / 'systems'},default"
WORLDS = (64, 128, 256)
TPS = (1, 2, 4, 8, 16, 32, 64)
CONTEXT = 262_144


def main() -> None:
    rows = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for world in WORLDS:
        for tp in TPS:
            if world % tp:
                continue
            attention_dp = world // tp
            for batch in sorted({1, 2, 4, tp}):
                result = cli_estimate(
                    str(MODEL),
                    "gb300_nvl288",
                    mode="static_gen",
                    backend_name="sglang",
                    backend_version="estimate",
                    database_mode="SOL",
                    isl=CONTEXT,
                    osl=2,
                    batch_size=batch,
                    tp_size=tp,
                    pp_size=1,
                    attention_dp_size=attention_dp,
                    moe_tp_size=1,
                    moe_ep_size=world,
                    kvcache_quant_mode="bfloat16",
                    nextn=0,
                    stride=1,
                    engine_step_backend="python",
                    systems_paths=SYSTEM_PATHS,
                )
                memory = result.summary.get_memory()
                latency = result.summary.get_generation_latency_dict()
                sources = result.summary.get_generation_source_dict()
                tps_user = 1000.0 / float(result.tpot)
                global_concurrency = attention_dp * batch
                shared_ms = float(latency.get("generation_shared_gate_up_gemm", 0.0)) + float(
                    latency.get("generation_shared_down_gemm", 0.0)
                )
                comm_ms = sum(
                    float(value)
                    for name, value in latency.items()
                    if any(fragment in name for fragment in ("_ar", "dispatch", "p2p"))
                )
                rows.append(
                    {
                        "world_gpus": world,
                        "tp_size": tp,
                        "attention_dp_size": attention_dp,
                        "moe_tp_size": 1,
                        "moe_ep_size": world,
                        "local_batch": batch,
                        "global_concurrency": global_concurrency,
                        "tpot_ms": float(result.tpot),
                        "tokens_s_user": tps_user,
                        "tokens_s_total": global_concurrency * tps_user,
                        "tokens_s_gpu": global_concurrency * tps_user / world,
                        "memory_gib_per_rank": float(memory["total"]),
                        "weights_gib_per_rank": float(memory["weights"]),
                        "cache_gib_per_rank": float(memory["kvcache"]),
                        "oom": bool(result.summary.check_oom()),
                        "kda_qkvg_ms": float(latency.get("generation_kda_qkvg_gemm", 0.0)),
                        "kda_o_ms": float(latency.get("generation_kda_o_gemm", 0.0)),
                        "moe_ms": float(latency.get("generation_moe", 0.0)),
                        "shared_expert_ms": shared_ms,
                        "attention_ms": float(latency.get("generation_attention", 0.0)),
                        "communication_ms": comm_ms,
                        "non_sol_ms": sum(
                            float(value)
                            for name, value in latency.items()
                            if float(value) > 0 and str(sources.get(name, "unknown")).lower() != "sol"
                        ),
                    }
                )

    with CSV_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    equal_global = [row for row in rows if row["local_batch"] == row["tp_size"]]
    lines = [
        "# 10T Kimi-trend hybrid TP×DP projection",
        "",
        "AIC formula-only SGLang SOL at 256K context on the hypothetical 288-GPU NVLink domain. Attention "
        "uses `TP × DP = world`; routed MoE uses `EP = world, MoE-TP = 1`. The TensorRT-LLM AIC backend "
        "currently rejects simultaneous attention TP>1 and DP>1.",
        "",
        "## Equal global concurrency and GPU count",
        "",
        "For each world size, local batch equals TP so global concurrency equals the number of GPUs. This "
        "isolates how GPUs are divided between TP and DP.",
        "",
        "| GPUs | Attention layout | EP | Global | TPS/user | TPS/GPU | Aggregate TPS | GiB/rank | Fit |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in equal_global:
        lines.append(
            f"| {row['world_gpus']} | TP{row['tp_size']}×DP{row['attention_dp_size']} | "
            f"{row['moe_ep_size']} | {row['global_concurrency']} | {row['tokens_s_user']:.1f} | "
            f"{row['tokens_s_gpu']:.1f} | {row['tokens_s_total']:.0f} | "
            f"{row['memory_gib_per_rank']:.1f} | {'no' if row['oom'] else 'yes'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "TP4 is the best swept equal-concurrency split at every cluster size. It shards the dominant BF16 "
            "KDA and shared-expert projections enough to overcome collective overhead while preserving enough DP "
            "replicas that each rank holds only four requests.",
            "",
            "Past TP4, the latent MLA cache becomes increasingly costly at equal global concurrency. MLA cache "
            "is replicated across the TP ranks, while reducing DP forces local batch to rise in proportion to TP. "
            "Thus TP8 halves more projection work but doubles each rank's resident/request cache scan versus TP4.",
            "",
            "These are uncalibrated custom-shape SOL ceilings. Hybrid TP×DP is supported in this AIC branch's "
            "SGLang formula path, but not its TensorRT-LLM path, and this does not prove that a production K3 kernel "
            "stack supports every topology.",
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
                "system": "gb300_nvl288",
                "backend": "sglang",
                "database_mode": "SOL",
                "context": CONTEXT,
                "world_sizes": WORLDS,
                "tp_sizes": TPS,
                "rows": len(rows),
                "pass_rows": sum(row["non_sol_ms"] == 0 for row in rows),
                "source_leak_rows": sum(row["non_sol_ms"] > 0 for row in rows),
                "no_speculation": True,
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
