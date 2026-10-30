#!/usr/bin/env python3
"""DSpark sensitivity for hybrid TP x DP 10T Kimi-trend projections."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from aiconfigurator.cli.api import cli_estimate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = HERE / "models" / "k3_10t_trend"
AL_PATH = ROOT / "inference-x" / "golden_al_distribution" / "kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml"
OUTPUT = HERE / "output"
CSV_PATH = OUTPUT / "kimi_k3_10t_hybrid_tpdp_spec_sweep.csv"
REPORT_PATH = OUTPUT / "KIMI_K3_10T_HYBRID_TPDP_SPEC.md"
METADATA_PATH = OUTPUT / "kimi_k3_10t_hybrid_tpdp_spec_metadata.json"
SYSTEM_PATHS = f"{HERE / 'systems'},{HERE.parent / 'kimi-k3-gb300-decode' / 'systems'},default"
WORLDS = (64, 128, 256)
TPS = (2, 4, 8)
CONTEXT = 262_144


def main() -> None:
    payload = yaml.safe_load(AL_PATH.read_text(encoding="utf-8"))
    al_curve = {0: 1.0, **{int(k): float(v) for k, v in payload["kimi-k3"]["thinking_on"].items()}}
    rows = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for world in WORLDS:
        for tp in TPS:
            dp = world // tp
            batch = tp  # holds global concurrency equal to world size
            for nextn, golden_al in sorted(al_curve.items()):
                kwargs = {
                    "mode": "static_gen",
                    "backend_name": "sglang",
                    "backend_version": "estimate",
                    "database_mode": "SOL",
                    "isl": CONTEXT,
                    "osl": 2,
                    "batch_size": batch,
                    "tp_size": tp,
                    "pp_size": 1,
                    "attention_dp_size": dp,
                    "moe_tp_size": 1,
                    "moe_ep_size": world,
                    "kvcache_quant_mode": "bfloat16",
                    "nextn": nextn,
                    "stride": 1,
                    "engine_step_backend": "python",
                    "systems_paths": SYSTEM_PATHS,
                }
                if nextn:
                    kwargs["nextn_accepted"] = golden_al - 1.0
                result = cli_estimate(str(MODEL), "gb300_nvl288", **kwargs)
                memory = result.summary.get_memory()
                latency = result.summary.get_generation_latency_dict()
                sources = result.summary.get_generation_source_dict()
                effective_tpot = float(result.tpot)
                global_concurrency = dp * batch
                tps_user = 1000.0 / effective_tpot
                rows.append(
                    {
                        "world_gpus": world,
                        "tp_size": tp,
                        "attention_dp_size": dp,
                        "moe_ep_size": world,
                        "local_batch": batch,
                        "global_concurrency": global_concurrency,
                        "nextn": nextn,
                        "borrowed_k3_golden_al": golden_al,
                        "accepted_draft_tokens": golden_al - 1.0,
                        "verification_step_ms": effective_tpot * golden_al,
                        "effective_tpot_ms": effective_tpot,
                        "tokens_s_user": tps_user,
                        "tokens_s_total": global_concurrency * tps_user,
                        "tokens_s_gpu": global_concurrency * tps_user / world,
                        "memory_gib_per_rank": float(memory["total"]),
                        "oom": bool(result.summary.check_oom()),
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

    lines = [
        "# 10T Kimi-trend hybrid TP×DP speculative-decoding sensitivity",
        "",
        "AIC formula-only SGLang SOL at 256K on the hypothetical 288-GPU NVLink domain. Global concurrency "
        "equals GPU count. DSpark acceptance lengths are borrowed from InferenceX's measured K3 SPEED-Bench "
        "thinking-on curve; they are not measurements of a hypothetical 10T target.",
        "",
        "| GPUs | Layout | No-spec TPS/user | Best nextn | Borrowed AL | Spec TPS/user | Speedup | TPS/GPU | GiB/rank |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for world in WORLDS:
        for tp in TPS:
            candidates = [
                row
                for row in rows
                if row["world_gpus"] == world and row["tp_size"] == tp and not row["oom"]
            ]
            if not candidates:
                continue
            baseline = next(row for row in candidates if row["nextn"] == 0)
            best = max(candidates, key=lambda row: row["tokens_s_user"])
            lines.append(
                f"| {world} | TP{tp}×DP{world // tp}, EP{world} | {baseline['tokens_s_user']:.1f} | "
                f"{best['nextn']} | {best['borrowed_k3_golden_al']:.2f} | {best['tokens_s_user']:.1f} | "
                f"{best['tokens_s_user'] / baseline['tokens_s_user']:.2f}× | {best['tokens_s_gpu']:.1f} | "
                f"{best['memory_gib_per_rank']:.1f} |"
            )
    lines.extend(
        [
            "",
            "Speculation raises effective progress per target verification step, but the target must verify a "
            "wider token block and run the five-layer DSpark draft. The optimum is where the borrowed AL curve's "
            "flattening no longer pays for deeper verification.",
            "",
            "The acceptance curve is the dominant uncertainty. The 10T target preserves K3 hidden size in this "
            "study, so the same DSpark geometry can be priced, but its prediction agreement with a future target "
            "could be materially different. Treat speedups as conditional sensitivities, not forecasts.",
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
                "oom_rows": sum(row["oom"] for row in rows),
                "acceptance_source": str(AL_PATH),
                "acceptance_curve": al_curve,
                "acceptance_is_borrowed_from_k3": True,
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
