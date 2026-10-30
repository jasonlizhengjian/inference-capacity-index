#!/usr/bin/env python3
"""Optimize hybrid TP x DP plus DSpark across non-depth Kimi scale variants."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from aiconfigurator.cli.api import cli_estimate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODELS_DIR = HERE / "models"
AL_PATH = ROOT / "inference-x" / "golden_al_distribution" / "kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml"
OUTPUT = HERE / "output"
CSV_PATH = OUTPUT / "all_models_hybrid_tpdp_spec_sweep.csv"
REPORT_PATH = OUTPUT / "ALL_MODELS_HYBRID_TPDP_SPEC_BEST.md"
METADATA_PATH = OUTPUT / "all_models_hybrid_tpdp_spec_metadata.json"
SYSTEM_PATHS = f"{HERE / 'systems'},{HERE.parent / 'kimi-k3-gb300-decode' / 'systems'},default"
VARIANT_IDS = ("k3_2p8t", "k3_5t_expert", "k3_5t_trend", "k3_10t_expert", "k3_10t_trend")
WORLDS = (64, 128, 256)
TP_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)
CONTEXT = 262_144


def main() -> None:
    manifest = {
        item["id"]: item
        for item in json.loads((MODELS_DIR / "manifest.json").read_text(encoding="utf-8"))
    }
    variants = [manifest[variant_id] for variant_id in VARIANT_IDS]
    payload = yaml.safe_load(AL_PATH.read_text(encoding="utf-8"))
    al_curve = {0: 1.0, **{int(k): float(v) for k, v in payload["kimi-k3"]["thinking_on"].items()}}
    rows = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        for world in WORLDS:
            for tp in TP_CANDIDATES:
                if world % tp or int(variant["num_heads"]) % tp:
                    continue
                attention_dp = world // tp
                batch = tp  # global concurrency = attention_dp * batch = world
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
                        "attention_dp_size": attention_dp,
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
                    result = cli_estimate(variant["model_path"], "gb300_nvl288", **kwargs)
                    memory = result.summary.get_memory()
                    latency = result.summary.get_generation_latency_dict()
                    sources = result.summary.get_generation_source_dict()
                    tpot = float(result.tpot)
                    tps_user = 1000.0 / tpot
                    rows.append(
                        {
                            "variant_id": variant["id"],
                            "variant_label": variant["label"],
                            "variant_family": variant["family"],
                            "text_parameters_trillion": variant["text_parameters_trillion"],
                            "active_parameters_billion": variant["active_parameters_billion"],
                            "num_layers": variant["num_layers"],
                            "num_experts": variant["num_experts"],
                            "topk": variant["topk"],
                            "world_gpus": world,
                            "tp_size": tp,
                            "attention_dp_size": attention_dp,
                            "moe_ep_size": world,
                            "local_batch": batch,
                            "global_concurrency": world,
                            "nextn": nextn,
                            "assumed_golden_al": golden_al,
                            "effective_tpot_ms": tpot,
                            "tokens_s_user": tps_user,
                            "tokens_s_total": world * tps_user,
                            "tokens_s_gpu": tps_user,
                            "memory_gib_per_rank": float(memory["total"]),
                            "weights_gib_per_rank": float(memory["weights"]),
                            "cache_gib_per_rank": float(memory["kvcache"]),
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
        "# Best hybrid TP×DP + DSpark configurations across scaled Kimi models",
        "",
        "AIC formula-only SGLang SOL at 256K resident context on the hypothetical 288-GPU NVLink domain. "
        "Global request concurrency equals GPU count, BF16 KV is resident, and every model is assumed to retain "
        "K3's measured InferenceX SPEED-Bench DSpark acceptance curve. Depth-only variants are excluded.",
        "",
        "## Model definitions",
        "",
        "| Model | Scaling direction | Params | Active/token | Layers | Experts/layer | Top-k |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    family_labels = {
        "baseline": "K3 baseline",
        "expert_only": "More experts only",
        "kimi_trend": "Deeper + more/larger active experts",
    }
    for variant in variants:
        lines.append(
            f"| {variant['label']} | {family_labels[variant['family']]} | "
            f"{variant['text_parameters_trillion']:.2f}T | {variant['active_parameters_billion']:.0f}B | "
            f"{variant['num_layers']} | {variant['num_experts']} | {variant['topk']} |"
        )
    lines.extend(
        [
            "",
            "## Jointly optimized configurations",
            "",
            "| Model | GPUs / MoE EP | Global concurrency | Concurrency/rank | Best attention layout | "
            "Best nextn | AL | Best no-spec TPS/user (layout) | Spec TPS/user | Gain | TPS/GPU | "
            "Aggregate TPS | GiB/rank |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    winners = []
    for variant in variants:
        for world in WORLDS:
            fit = [
                row
                for row in rows
                if row["variant_id"] == variant["id"] and row["world_gpus"] == world and not row["oom"]
            ]
            no_spec = max((row for row in fit if row["nextn"] == 0), key=lambda row: row["tokens_s_user"])
            best = max(fit, key=lambda row: row["tokens_s_user"])
            winners.append(best)
            lines.append(
                f"| {variant['label']} | {world} | {world} | {best['local_batch']} | "
                f"TP{best['tp_size']}×DP{best['attention_dp_size']} | {best['nextn']} | "
                f"{best['assumed_golden_al']:.2f} | {no_spec['tokens_s_user']:.1f} "
                f"(TP{no_spec['tp_size']}×DP{no_spec['attention_dp_size']}) | "
                f"{best['tokens_s_user']:.1f} | {best['tokens_s_user'] / no_spec['tokens_s_user']:.2f}× | "
                f"{best['tokens_s_gpu']:.1f} | {best['tokens_s_total']:.0f} | "
                f"{best['memory_gib_per_rank']:.1f} |"
            )
    lines.extend(
        [
            "",
            "`TPS/GPU` equals `TPS/user` in this comparison because global concurrency equals GPU count. The "
            "winning TP/DP split and speculative depth are selected jointly after excluding OOM configurations.",
            "",
            "The fixed AL assumption is intentional for this scenario, but every result remains an uncalibrated "
            "custom-model SOL ceiling. It excludes prefill, KV transfer, queueing, scheduler overhead, and runtime "
            "efficiency loss.",
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
                "system": "gb300_nvl288",
                "backend": "sglang",
                "database_mode": "SOL",
                "context": CONTEXT,
                "world_sizes": WORLDS,
                "tp_candidates": TP_CANDIDATES,
                "variants": VARIANT_IDS,
                "rows": len(rows),
                "pass_rows": sum(row["non_sol_ms"] == 0 for row in rows),
                "source_leak_rows": sum(row["non_sol_ms"] > 0 for row in rows),
                "oom_rows": sum(row["oom"] for row in rows),
                "acceptance_source": str(AL_PATH),
                "acceptance_curve": al_curve,
                "acceptance_assumed_unchanged": True,
                "selection": "max TPS/user among memory-fitting TP/DP and nextn; global concurrency = GPUs",
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
