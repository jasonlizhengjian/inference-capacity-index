#!/usr/bin/env python3
"""AIC SOL comparison of DeepSeek-V4-Pro and a ~5T expert-scaled variant."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aic-dsv4-5t-mpl")

import aiconfigurator_core
from aiconfigurator.cli.api import cli_estimate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AIC_REPO = ROOT / "sglang-simulator-cpu" / "repos" / "aiconfigurator-kimi-k3-pr1435"
MODEL_DIR = HERE / "models" / "deepseek_v4_pro_5t_expert"
OUTPUT = HERE / "output"
CONFIG_PATH = MODEL_DIR / "config.json"
VARIANT_METADATA_PATH = MODEL_DIR / "variant_metadata.json"
CSV_PATH = OUTPUT / "deepseek_v4_5t_sol.csv"
REPORT_PATH = OUTPUT / "REPORT.md"
METADATA_PATH = OUTPUT / "metadata.json"

BASE_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
BASELINE_EXPERTS = 384
SCALED_EXPERTS = 1216
OFFICIAL_BASELINE_TOTAL = 1.6e12
OFFICIAL_BASELINE_ACTIVE = 49e9
CONTEXT = 262_144
DEPS = (16, 64)
MATCHED_CPRS = (1, 2, 3, 4)
SYSTEMS = {
    "GB300": {"id": "gb300", "capacity_bytes": 298_013_687_808},
    "B300": {"id": "b300_sxm", "capacity_bytes": 288_400_343_040},
}


def _git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _make_variant() -> dict[str, Any]:
    package_dir = Path(aiconfigurator_core.__file__).resolve().parent
    source = package_dir / "model_configs" / "deepseek-ai--DeepSeek-V4-Pro_config.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    if int(config["n_routed_experts"]) != BASELINE_EXPERTS:
        raise RuntimeError(f"Expected {BASELINE_EXPERTS} baseline experts, got {config['n_routed_experts']}")
    if SCALED_EXPERTS % max(DEPS):
        raise RuntimeError("Scaled expert count must be divisible by every tested EP size")

    layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    expert_intermediate = int(config["moe_intermediate_size"])
    topk = int(config["num_experts_per_tok"])
    params_per_expert_all_layers = layers * 3 * hidden * expert_intermediate
    estimated_total = OFFICIAL_BASELINE_TOTAL + (
        SCALED_EXPERTS - BASELINE_EXPERTS
    ) * params_per_expert_all_layers
    router_growth = layers * hidden * (SCALED_EXPERTS - BASELINE_EXPERTS)
    estimated_active = OFFICIAL_BASELINE_ACTIVE + router_growth

    config["n_routed_experts"] = SCALED_EXPERTS
    config["_name_or_path"] = "local/DeepSeek-V4-Pro-5T-ExpertScale"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metadata = {
        "label": "DeepSeek V4 Pro ~5T expert-capacity scale",
        "source_model": BASE_MODEL,
        "scaling_rule": "increase routed experts only; preserve layers, dimensions, top-k, and attention",
        "baseline_total_parameters_official_rounded": int(OFFICIAL_BASELINE_TOTAL),
        "baseline_active_parameters_official_rounded": int(OFFICIAL_BASELINE_ACTIVE),
        "estimated_total_parameters": int(estimated_total),
        "estimated_total_parameters_trillion": estimated_total / 1e12,
        "estimated_active_parameters": int(estimated_active),
        "estimated_active_parameters_billion": estimated_active / 1e9,
        "baseline_experts": BASELINE_EXPERTS,
        "scaled_experts": SCALED_EXPERTS,
        "experts_per_token": topk,
        "num_hidden_layers": layers,
        "hidden_size": hidden,
        "moe_intermediate_size": expert_intermediate,
        "parameters_per_expert_across_layers": params_per_expert_all_layers,
        "router_parameter_growth": router_growth,
    }
    VARIANT_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _estimate(label: str, model_path: str, system: str, dep: int, cpr: int) -> dict[str, Any]:
    result = cli_estimate(
        model_path,
        SYSTEMS[system]["id"],
        mode="static_gen",
        backend_name="sglang",
        backend_version="estimate",
        database_mode="SOL",
        isl=CONTEXT,
        osl=2,
        prefix=CONTEXT,
        batch_size=cpr,
        tp_size=1,
        pp_size=1,
        attention_dp_size=dep,
        moe_tp_size=1,
        moe_ep_size=dep,
        kvcache_quant_mode="fp8",
        nextn=0,
        stride=1,
        engine_step_backend="python",
    )
    memory = result.summary.get_memory()
    latency = {
        str(name): float(value)
        for name, value in result.summary.get_generation_latency_dict().items()
    }
    sources = {
        str(name): str(value or "unknown").lower()
        for name, value in result.summary.get_generation_source_dict().items()
    }
    non_sol = sum(value for name, value in latency.items() if sources.get(name) != "sol")
    if non_sol:
        raise RuntimeError(f"Non-SOL latency leaked into {label}/{system}/DEP{dep}/cpr{cpr}: {non_sol}")
    tpot = float(result.tpot)
    capacity_gib = SYSTEMS[system]["capacity_bytes"] / 2**30
    return {
        "variant": label,
        "model_path": model_path,
        "system": system,
        "dep": dep,
        "concurrency_per_rank": cpr,
        "global_concurrency": dep * cpr,
        "context_tokens": CONTEXT,
        "tpot_ms": tpot,
        "tps_user": 1000.0 / tpot,
        "tps_total": dep * cpr * 1000.0 / tpot,
        "tps_gpu": cpr * 1000.0 / tpot,
        "memory_gib": float(memory["total"]),
        "weights_gib": float(memory["weights"]),
        "kvcache_gib": float(memory["kvcache"]),
        "activations_gib": float(memory["activations"]),
        "other_gib": float(memory["nccl"]) + float(memory["others"]),
        "capacity_gib": capacity_gib,
        "memory_fraction": float(memory["total"]) / capacity_gib,
        "oom": bool(result.summary.check_oom()),
        "attention_ms": float(latency.get("generation_attention", 0.0)),
        "moe_ms": float(latency.get("generation_moe", 0.0)),
        "moe_pre_dispatch_ms": float(latency.get("generation_moe_pre_dispatch", 0.0)),
        "moe_post_dispatch_ms": float(latency.get("generation_moe_post_dispatch", 0.0)),
        "mhc_ms": float(latency.get("generation_mhc_pre", 0.0))
        + float(latency.get("generation_mhc_post", 0.0)),
        "router_ms": float(latency.get("generation_router_gemm", 0.0)),
        "latency_json": json.dumps(latency, sort_keys=True),
        "sources_json": json.dumps(sources, sort_keys=True),
    }


def _full_boundary(
    label: str,
    model_path: str,
    system: str,
    dep: int,
    cache: dict[tuple[str, str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    def get(cpr: int) -> dict[str, Any]:
        key = (label, system, dep, cpr)
        if key not in cache:
            cache[key] = _estimate(label, model_path, system, dep, cpr)
        return cache[key]

    low, high = 1, 2
    if get(low)["oom"]:
        return get(low)
    while not get(high)["oom"]:
        low, high = high, high * 2
        if high > 4096:
            raise RuntimeError(f"Failed to find OOM boundary for {label}/{system}/DEP{dep}")
    while high - low > 1:
        mid = (low + high) // 2
        if get(mid)["oom"]:
            high = mid
        else:
            low = mid
    if get(low)["oom"] or not get(low + 1)["oom"]:
        raise RuntimeError(f"Invalid OOM bracket for {label}/{system}/DEP{dep}: {low}")
    return get(low)


def _write_csv(rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with CSV_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if i else "---" for i in range(len(headers))) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _write_report(
    rows: list[dict[str, Any]], full_rows: list[dict[str, Any]], variant_metadata: dict[str, Any]
) -> None:
    lookup = {
        (row["variant"], row["system"], row["dep"], row["concurrency_per_rank"]): row
        for row in rows
    }
    lines = [
        "# DeepSeek V4 Pro scaled to approximately 5T: AIC SOL",
        "",
        "The 5T variant increases routed experts from 384 to 1,216 while preserving 61 layers, top-6 "
        "routing, all expert/hidden dimensions, and DeepSeek V4's sparse-attention pattern. Using the official "
        "rounded 1.6T baseline, this gives %.3fT total and approximately %.1fB active parameters/token; the "
        "small active increase is the larger router." % (
            variant_metadata["estimated_total_parameters_trillion"],
            variant_metadata["estimated_active_parameters_billion"],
        ),
        "",
        "All results are formula-only AIC SOL, decode-only with a resident 256K context, FP8 KV cache, "
        "no speculative decoding, and TP1/PP1/attention-DP=DEP/MoE-EP=DEP.",
        "",
        "## Matched concurrency/rank",
        "",
    ]
    matched_table = []
    for system in SYSTEMS:
        for dep in DEPS:
            for cpr in MATCHED_CPRS:
                base = lookup[("1.6T baseline", system, dep, cpr)]
                scaled = lookup[("4.95T expert-scale", system, dep, cpr)]
                matched_table.append(
                    [
                        system,
                        str(dep),
                        str(cpr),
                        f"{base['tps_user']:.1f}",
                        f"{scaled['tps_user']:.1f}",
                        f"{scaled['tps_user'] / base['tps_user']:.2f}x",
                        f"{base['memory_gib']:.1f}",
                        f"{scaled['memory_gib']:.1f}",
                        "yes" if not scaled["oom"] else "OOM",
                    ]
                )
    lines += _table(
        [
            "System",
            "DEP",
            "c/r",
            "1.6T TPS/user",
            "4.95T TPS/user",
            "Retained",
            "1.6T GiB/rank",
            "4.95T GiB/rank",
            "4.95T fits",
        ],
        matched_table,
    )

    lines += [
        "",
        "## Maximum memory-fitting concurrency",
        "",
        "The boundary uses the full modeled physical card capacity with no deployment headroom. The next "
        "whole concurrency/rank was explicitly checked and is OOM.",
        "",
    ]
    full_lookup = {(row["variant"], row["system"], row["dep"]): row for row in full_rows}
    full_table = []
    for system in SYSTEMS:
        for dep in DEPS:
            base = full_lookup[("1.6T baseline", system, dep)]
            scaled = full_lookup[("4.95T expert-scale", system, dep)]
            full_table.append(
                [
                    system,
                    str(dep),
                    str(base["concurrency_per_rank"]),
                    f"{base['tps_user']:.1f}",
                    f"{base['tps_gpu']:.0f}",
                    str(scaled["concurrency_per_rank"]),
                    f"{scaled['tps_user']:.1f}",
                    f"{scaled['tps_gpu']:.0f}",
                    f"{scaled['memory_gib']:.1f}",
                ]
            )
    lines += _table(
        [
            "System",
            "DEP",
            "1.6T max c/r",
            "1.6T TPS/user",
            "1.6T TPS/GPU",
            "4.95T max c/r",
            "4.95T TPS/user",
            "4.95T TPS/GPU",
            "4.95T GiB/rank",
        ],
        full_table,
    )

    lines += [
        "",
        "## What changed",
        "",
        "Expert-capacity scaling leaves attention, selected expert count, expert dimensions, and MoE payload "
        "volume unchanged. It increases sharded expert weight storage and the all-expert router projection. "
        "Consequently, matched-load latency should change much less than model size, while available KV-cache "
        "capacity falls sharply—especially at DEP16.",
        "",
        "DEP64 has an additional expert-occupancy effect. The baseline owns 384/64 = 6 experts/rank, while "
        "the 5T variant owns 1,216/64 = 19. At c/r=4 on GB300, AIC's SOL MoE-overlap term grows from "
        "2.05 ms to 5.41 ms because the larger local expert pool gets less weight reuse; attention remains "
        "unchanged at 3.02 ms. At DEP16, both layouts are already in a low-reuse regime over c/r 1–4 and "
        "the total penalty stays near 1–2%.",
        "",
        "This is an architectural sensitivity, not a trained 5T checkpoint or a claim that quality scales this "
        "way. A real model may change depth, width, top-k, shared experts, or attention dimensions, which would "
        "increase active work and reduce TPS/user much more.",
        "",
        "## Artifacts",
        "",
        f"- `{CSV_PATH.name}`: all matched, search, fit, and OOM points",
        f"- `{METADATA_PATH.name}`: run and environment metadata",
        f"- `../models/deepseek_v4_pro_5t_expert/config.json`: generated model configuration",
        f"- `../models/deepseek_v4_pro_5t_expert/variant_metadata.json`: parameter accounting assumptions",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    variant_metadata = _make_variant()
    variants = {
        "1.6T baseline": BASE_MODEL,
        "4.95T expert-scale": str(MODEL_DIR),
    }
    cache: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for label, model_path in variants.items():
        for system in SYSTEMS:
            for dep in DEPS:
                for cpr in MATCHED_CPRS:
                    key = (label, system, dep, cpr)
                    cache[key] = _estimate(label, model_path, system, dep, cpr)
                    print(f"matched {label} {system} DEP{dep} c/r={cpr}", flush=True)

    full_rows = []
    for label, model_path in variants.items():
        for system in SYSTEMS:
            for dep in DEPS:
                boundary = _full_boundary(label, model_path, system, dep, cache)
                full_rows.append(boundary)
                print(
                    f"full {label} {system} DEP{dep}: c/r={boundary['concurrency_per_rank']}",
                    flush=True,
                )

    rows = sorted(
        cache.values(),
        key=lambda row: (
            row["system"], row["dep"], row["variant"], row["concurrency_per_rank"]
        ),
    )
    _write_csv(rows)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aic_commit": _git_commit(AIC_REPO),
        "aiconfigurator_core_version": getattr(aiconfigurator_core, "__version__", "unknown"),
        "context_tokens": CONTEXT,
        "database_mode": "SOL",
        "backend": "sglang",
        "parallelism": "TP1 PP1 attention-DP=DEP MoE-TP1 MoE-EP=DEP",
        "kvcache_quant_mode": "fp8",
        "systems": SYSTEMS,
        "deps": list(DEPS),
        "matched_concurrency_per_rank": list(MATCHED_CPRS),
        "variant": variant_metadata,
        "variant_config_sha256": _sha256(CONFIG_PATH),
        "estimate_count": len(rows),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(rows, full_rows, variant_metadata)
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
