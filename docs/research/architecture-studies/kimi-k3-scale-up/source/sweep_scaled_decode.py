#!/usr/bin/env python3
"""Sweep scaled Kimi-K3-family decode-only SOL performance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiconfigurator
import aiconfigurator_core
from aiconfigurator.cli.api import cli_estimate


EXPERIMENT_DIR = Path(__file__).resolve().parent
SIMULATOR_DIR = EXPERIMENT_DIR.parents[1]
AIC_REPO = SIMULATOR_DIR / "repos" / "aiconfigurator-kimi-k3-pr1435"
BASE_EXPERIMENT = SIMULATOR_DIR / "experiments" / "kimi-k3-gb300-decode"
SYSTEMS_PATHS = f"{BASE_EXPERIMENT / 'systems'},default"
MODELS_DIR = EXPERIMENT_DIR / "models"
OUTPUT_DIR = EXPERIMENT_DIR / "output"
CSV_PATH = OUTPUT_DIR / "scaled_kimi_k3_256k_1m_sol_sweep.csv"
OPERATIONS_PATH = OUTPUT_DIR / "scaled_kimi_k3_256k_operation_breakdown.csv"
REPORT_PATH = OUTPUT_DIR / "SCALED_KIMI_K3_SOL_REPORT.md"
METADATA_PATH = OUTPUT_DIR / "scaled_kimi_k3_metadata.json"

CONTEXTS = (262_144, 1_048_576)
LOADS = (1, 2, 3, 4)
DEP_SIZES = (16, 32, 64, 128, 256)
TEP_CANDIDATES = (16, 24, 32, 48, 64, 96, 128, 192)
SYSTEMS = (
    ("GB300", "gb300", 8.0),
    ("GB300 + Rubin HBM", "gb300_rubin_memory", 22.0),
)

FIELDS = [
    "variant_id",
    "variant_label",
    "variant_family",
    "text_parameters_trillion",
    "active_parameters_billion",
    "num_layers",
    "kda_layers",
    "mla_layers",
    "num_experts",
    "topk",
    "num_heads",
    "system_label",
    "system_name",
    "hbm_bandwidth_tb_s",
    "context_tokens_per_request",
    "layout",
    "world_gpus",
    "tp_size",
    "attention_dp_size",
    "moe_tp_size",
    "moe_ep_size",
    "local_batch",
    "global_concurrency",
    "tpot_ms",
    "tokens_s_user",
    "tokens_s_total",
    "tokens_s_gpu",
    "memory_gib_per_rank",
    "weights_gib_per_rank",
    "kvcache_gib_per_rank",
    "activations_gib_per_rank",
    "other_gib_per_rank",
    "attention_ms",
    "attention_share_pct",
    "moe_ms",
    "moe_share_pct",
    "non_sol_ms",
    "oom",
    "status",
    "error",
]

OP_FIELDS = [
    "variant_id",
    "system_label",
    "context_tokens_per_request",
    "layout",
    "world_gpus",
    "local_batch",
    "global_concurrency",
    "tpot_ms",
    "operation",
    "operation_ms",
    "operation_share_pct",
    "source",
]


def _git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _estimate(
    variant: dict[str, Any],
    system_label: str,
    system_name: str,
    bandwidth: float,
    context: int,
    layout: str,
    world: int,
    local_batch: int,
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    if layout == "DEP":
        tp, attention_dp, moe_tp, moe_ep = 1, world, 1, world
        global_concurrency = world * local_batch
    else:
        tp, attention_dp, moe_tp, moe_ep = world, 1, 1, world
        global_concurrency = local_batch
    row: dict[str, Any] = {
        "variant_id": variant["id"],
        "variant_label": variant["label"],
        "variant_family": variant["family"],
        "text_parameters_trillion": variant["text_parameters_trillion"],
        "active_parameters_billion": variant["active_parameters_billion"],
        "num_layers": variant["num_layers"],
        "kda_layers": variant["kda_layers"],
        "mla_layers": variant["mla_layers"],
        "num_experts": variant["num_experts"],
        "topk": variant["topk"],
        "num_heads": variant["num_heads"],
        "system_label": system_label,
        "system_name": system_name,
        "hbm_bandwidth_tb_s": bandwidth,
        "context_tokens_per_request": context,
        "layout": layout,
        "world_gpus": world,
        "tp_size": tp,
        "attention_dp_size": attention_dp,
        "moe_tp_size": moe_tp,
        "moe_ep_size": moe_ep,
        "local_batch": local_batch,
        "global_concurrency": global_concurrency,
    }
    try:
        result = cli_estimate(
            variant["model_path"],
            system_name,
            mode="static_gen",
            backend_name="trtllm",
            backend_version="estimate",
            database_mode="SOL",
            isl=context,
            osl=2,
            batch_size=local_batch,
            tp_size=tp,
            pp_size=1,
            attention_dp_size=attention_dp,
            moe_tp_size=moe_tp,
            moe_ep_size=moe_ep,
            kvcache_quant_mode="bfloat16",
            nextn=0,
            stride=1,
            engine_step_backend="python",
            systems_paths=SYSTEMS_PATHS,
        )
        latency = {
            str(name): float(value)
            for name, value in result.summary.get_generation_latency_dict().items()
        }
        sources = {
            str(name): str(value or "unknown").lower()
            for name, value in result.summary.get_generation_source_dict().items()
        }
        memory = result.summary.get_memory()
        non_sol = sum(
            value
            for name, value in latency.items()
            if value > 0 and sources.get(name, "unknown") != "sol"
        )
        tpot = float(result.tpot)
        tps_user = 1000.0 / tpot
        attention = float(latency.get("generation_attention", 0.0))
        moe = float(latency.get("generation_moe", 0.0))
        row.update(
            {
                "tpot_ms": tpot,
                "tokens_s_user": tps_user,
                "tokens_s_total": global_concurrency * tps_user,
                "tokens_s_gpu": global_concurrency * tps_user / world,
                "memory_gib_per_rank": float(memory["total"]),
                "weights_gib_per_rank": float(memory["weights"]),
                "kvcache_gib_per_rank": float(memory["kvcache"]),
                "activations_gib_per_rank": float(memory["activations"]),
                "other_gib_per_rank": float(memory["nccl"]) + float(memory["others"]),
                "attention_ms": attention,
                "attention_share_pct": attention / tpot * 100.0,
                "moe_ms": moe,
                "moe_share_pct": moe / tpot * 100.0,
                "non_sol_ms": non_sol,
                "oom": bool(result.summary.check_oom()),
                "status": "PASS" if non_sol == 0 else "SOURCE_LEAK",
                "error": "",
            }
        )
        return row, latency, sources
    except Exception as error:
        row.update({"status": "FAIL", "error": f"{type(error).__name__}: {error}"})
        return row, {}, {}


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    integer_fields = {
        "num_layers",
        "kda_layers",
        "mla_layers",
        "num_experts",
        "topk",
        "num_heads",
        "context_tokens_per_request",
        "world_gpus",
        "tp_size",
        "attention_dp_size",
        "moe_tp_size",
        "moe_ep_size",
        "local_batch",
        "global_concurrency",
    }
    float_fields = {
        "text_parameters_trillion",
        "active_parameters_billion",
        "hbm_bandwidth_tb_s",
        "tpot_ms",
        "tokens_s_user",
        "tokens_s_total",
        "tokens_s_gpu",
        "memory_gib_per_rank",
        "weights_gib_per_rank",
        "kvcache_gib_per_rank",
        "activations_gib_per_rank",
        "other_gib_per_rank",
        "attention_ms",
        "attention_share_pct",
        "moe_ms",
        "moe_share_pct",
        "non_sol_ms",
    }
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in integer_fields:
            if row.get(field):
                row[field] = int(row[field])
        for field in float_fields:
            if row.get(field):
                row[field] = float(row[field])
        row["oom"] = row.get("oom", "").lower() == "true"
    return rows


def _one(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any]:
    selected = [row for row in rows if all(row.get(key) == value for key, value in filters.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected[0]


def _fit(rows: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("status") == "PASS"
        and not row.get("oom")
        and all(row.get(key) == value for key, value in filters.items())
    ]


def _table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> list[str]:
    aligns = aligns or ["---"] * len(headers)
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _report(rows: list[dict[str, Any]], variants: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    lines = [
        "# Scaling Kimi-K3 to approximately 5T and 10T: decode-only SOL study",
        "",
        "These are hypothetical architecture projections, not announced Moonshot models. Every point is a "
        "formula-only AIC TensorRT-LLM speed-of-light decode step with BF16 KV already resident. There is no "
        "prefill, KV transfer, speculation, scheduler, queueing, or production-runtime loss.",
        "",
        "## Scaling assumptions",
        "",
        "Kimi K2.5→K3 was a compound scale-up, not merely more experts: layers 61→93, routed experts "
        "384→896, top-k 8→16, expert intermediate width 2048→3072, shared experts 1→2, and heads 64→96, "
        "while hidden size stayed 7168. K3 also replaced all-MLA with 69 KDA + 24 MLA layers and introduced "
        "a 3584-dimensional LatentMoE path.",
        "",
    ]
    lines.extend(
        _table(
            ["Dimension", "Kimi K2.5", "Kimi K3", "Observed change"],
            [
                ["Nominal scale class", "~1T", "~2.8T text / ~3T", "~2.8–3×"],
                ["Hidden size", "7168", "7168", "unchanged"],
                ["Layers", "61", "93", "1.52×"],
                ["Routed experts", "384", "896", "2.33×"],
                ["Experts/token", "8", "16", "2.00×"],
                ["Expert intermediate", "2048", "3072", "1.50×"],
                ["Shared experts", "1", "2", "2.00×"],
                ["Attention heads", "64", "96", "1.50×"],
                ["Dense-stem intermediate", "18432", "33792", "1.83×"],
                ["Attention pattern", "61 MLA", "69 KDA + 24 MLA", "hybridized"],
                ["Maximum context", "256K", "1M", "4.00×"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "The table is taken from the two AIC model configurations used by this workspace. K3's LatentMoE "
            "changes the expert input geometry, so no single K2.5→K3 dimension ratio can be extrapolated safely.",
            "",
            "The study therefore brackets three directions:",
            "",
            "- **Expert-only:** increase total routed experts to 1600/3200 while retaining K3's 93 layers, top-16, "
            "and per-expert geometry. This is the constant-active-compute lower bound.",
            "- **Kimi-trend:** jointly increase depth, experts, heads, top-k, shared experts, and the dense stem, while "
            "preserving K3's hidden size, LatentMoE expert geometry, and approximately 3:1 KDA:MLA pattern. This is "
            "the central scenario informed by K2.5→K3.",
            "- **Depth-only:** hold every K3 dimension except layer count. This makes active work scale nearly with "
            "total parameters and serves as a high-penalty bound, not the expected product design.",
            "",
            "Parameter counts below are text-model accounting from the operation geometry. They omit the small vision "
            "tower and quantization metadata overhead.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Variant", "Family", "Params", "Active/token", "Layers KDA/MLA", "Experts", "Top-k", "Heads", "Packed weights"],
            [
                [
                    variant["label"],
                    variant["family"],
                    f"{variant['text_parameters_trillion']:.2f}T",
                    f"{variant['active_parameters_billion']:.0f}B",
                    f"{variant['num_layers']} ({variant['kda_layers']}/{variant['mla_layers']})",
                    str(variant["num_experts"]),
                    str(variant["topk"]),
                    str(variant["num_heads"]),
                    f"{variant['approx_packed_weight_gib']:.0f} GiB",
                ]
                for variant in variants
            ],
            ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        )
    )
    lines.extend(
        [
            "",
            "## Minimum memory-fit topology in the sweep",
            "",
            "`DEP` means TP1 with attention-DP=MoE-EP=world. `TEP` means attention TP=world and MoE-EP=world. "
            "Only TP widths that evenly divide the variant's head count were evaluated. DEP128/256 and most TEP "
            "widths are theoretical K3 layouts, not documented serving recipes.",
            "",
        ]
    )
    fit_table: list[list[str]] = []
    for variant in variants:
        for system_label, _, _ in SYSTEMS:
            for context in CONTEXTS:
                for layout in ("DEP", "TEP"):
                    candidates = _fit(
                        rows,
                        variant_id=variant["id"],
                        system_label=system_label,
                        context_tokens_per_request=context,
                        layout=layout,
                        local_batch=1,
                    )
                    if candidates:
                        minimum = min(candidates, key=lambda row: int(row["world_gpus"]))
                        fit_table.append(
                            [
                                variant["label"],
                                system_label,
                                f"{context // 1024}K" if context < 1_048_576 else "1M",
                                layout,
                                str(minimum["world_gpus"]),
                                f"{minimum['tokens_s_user']:.1f}",
                                f"{minimum['memory_gib_per_rank']:.1f}",
                                f"{minimum['weights_gib_per_rank']:.1f}",
                                f"{minimum['kvcache_gib_per_rank']:.1f}",
                            ]
                        )
                    else:
                        fit_table.append(
                            [variant["label"], system_label, f"{context // 1024}K" if context < 1_048_576 else "1M", layout, "none", "—", "—", "—", "—"]
                        )
    lines.extend(
        _table(
            ["Variant", "System", "Context", "Layout", "Min GPUs", "TPS/user", "GiB/rank", "Weights", "Cache/state"],
            fit_table,
            ["---", "---", "---:", "---", "---:", "---:", "---:", "---:", "---:"],
        )
    )

    lines.extend(
        [
            "",
            "## Comparable 256K DEP64 directional slice",
            "",
            "This holds the topology at DEP64, local batch at 1, and global concurrency at 64. It is the "
            "cleanest apples-to-apples measure of the architectural penalty. Rows marked `no` exceed 277.6 "
            "GiB/rank; their TPS/user is a timing counterfactual, not a runnable configuration.",
            "",
        ]
    )
    directional_table: list[list[str]] = []
    gb_baseline = _one(
        rows,
        variant_id="k3_2p8t",
        system_label="GB300",
        context_tokens_per_request=262_144,
        layout="DEP",
        world_gpus=64,
        local_batch=1,
    )
    hbm_baseline = _one(
        rows,
        variant_id="k3_2p8t",
        system_label="GB300 + Rubin HBM",
        context_tokens_per_request=262_144,
        layout="DEP",
        world_gpus=64,
        local_batch=1,
    )
    for variant in variants:
        gb_row = _one(
            rows,
            variant_id=variant["id"],
            system_label="GB300",
            context_tokens_per_request=262_144,
            layout="DEP",
            world_gpus=64,
            local_batch=1,
        )
        hbm_row = _one(
            rows,
            variant_id=variant["id"],
            system_label="GB300 + Rubin HBM",
            context_tokens_per_request=262_144,
            layout="DEP",
            world_gpus=64,
            local_batch=1,
        )
        directional_table.append(
            [
                variant["label"],
                f"{variant['active_parameters_billion']:.0f}B",
                f"{gb_row['tokens_s_user']:.1f}",
                f"{gb_row['tokens_s_user'] / gb_baseline['tokens_s_user']:.2f}×",
                f"{hbm_row['tokens_s_user']:.1f}",
                f"{hbm_row['tokens_s_user'] / hbm_baseline['tokens_s_user']:.2f}×",
                f"{gb_row['memory_gib_per_rank']:.1f}",
                "no" if gb_row["oom"] else "yes",
            ]
        )
    lines.extend(
        _table(
            ["Variant", "Active/token", "GB300 TPS/user", "vs K3", "22 TB/s TPS/user", "vs K3", "GiB/rank", "Fits"],
            directional_table,
            ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
        )
    )

    lines.extend(
        [
            "",
            "## Why the 10T Kimi-trend model does not fit DEP—and how quantization or TEP changes that",
            "",
            "In this experiment, routed experts use MXFP4 while attention, dense, shared-expert, router, latent, "
            "and output weights remain BF16. `DEP64, c/r=1` means one request per rank and therefore 64 requests "
            "globally. Reducing the global workload to one request would remove cache/state from idle ranks, but "
            "would not solve the capacity problem: weights alone exceed one GB300's 277.6 GiB.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["10T Kimi-trend layout", "Weights", "Cache/state", "Other", "Total", "Fits GB300"],
            [
                ["DEP64, c/r=1", "373.4 GiB", "16.7 GiB", "3.6 GiB", "393.7 GiB", "no"],
                ["DEP256, c/r=1", "320.7 GiB", "16.7 GiB", "3.6 GiB", "340.9 GiB", "no"],
                ["TEP24, local batch=1", "218.6 GiB", "10.4 GiB", "3.9 GiB", "232.9 GiB", "yes"],
                ["DEP64 + estimated FP8 non-routed weights, c/r=1", "~221.9 GiB", "16.7 GiB", "3.6 GiB", "~242.2 GiB", "yes"],
            ],
            ["---", "---:", "---:", "---:", "---:", "---"],
        )
    )
    lines.extend(
        [
            "",
            "DEP already shards the routed expert payload through EP, but leaves roughly 303 GiB of non-routed "
            "BF16 weights replicated on every rank. TEP tensor-shards much of that non-routed component while "
            "retaining expert parallelism. This helps even with MLA: MLA reduces the token-dependent KV cache, "
            "but its projection weights remain shardable. Moreover, K3 is hybrid rather than all-MLA—baseline K3 "
            "has 69 KDA + 24 MLA layers, and the hypothetical 10T model has 108 KDA + 36 MLA layers.",
            "",
            "The FP8 row is capacity arithmetic, not a rerun with verified FP8 kernels: halving the approximately "
            "303 GiB replicated BF16 component while leaving the roughly 70.4 GiB MXFP4 expert shard unchanged "
            "gives about 221.9 GiB of weights. That would fit at 256K through approximately c/r=3 (~275.5 GiB "
            "total) but not c/r=4 (~292.2 GiB). Accuracy, kernel availability, and realized performance require "
            "separate validation.",
            "",
        ]
    )

    lines.extend(["", "## 256K DEP64 comparison", ""])
    dep64_table: list[list[str]] = []
    for variant in variants:
        for system_label, _, _ in SYSTEMS:
            for load in LOADS:
                row = _one(
                    rows,
                    variant_id=variant["id"],
                    system_label=system_label,
                    context_tokens_per_request=262_144,
                    layout="DEP",
                    world_gpus=64,
                    local_batch=load,
                )
                dep64_table.append(
                    [
                        variant["label"],
                        system_label,
                        str(load),
                        str(64 * load),
                        f"{row['tokens_s_user']:.1f}" if row["status"] == "PASS" else "FAIL",
                        f"{row['tokens_s_gpu']:.1f}" if row["status"] == "PASS" else "—",
                        f"{row['memory_gib_per_rank']:.1f}" if row["status"] == "PASS" else "—",
                        "yes" if row.get("oom") else "no",
                    ]
                )
    lines.extend(
        _table(
            ["Variant", "System", "c/r", "Global", "TPS/user", "TPS/GPU", "GiB/rank", "OOM"],
            dep64_table,
            ["---", "---", "---:", "---:", "---:", "---:", "---:", "---"],
        )
    )

    lines.extend(
        [
            "",
            "## Theoretical best memory-fit 256K latency at each load",
            "",
            "This section minimizes per-user latency across the whole sweep. It can assign 96–192 GPUs to one "
            "request and has very poor TPS/GPU, so it is a mathematical latency frontier—not a deployment "
            "recommendation.",
            "",
        ]
    )
    best_table: list[list[str]] = []
    for variant in variants:
        for system_label, _, _ in SYSTEMS:
            for load in LOADS:
                candidates = _fit(
                    rows,
                    variant_id=variant["id"],
                    system_label=system_label,
                    context_tokens_per_request=262_144,
                    local_batch=load,
                )
                if not candidates:
                    continue
                best = max(candidates, key=lambda row: float(row["tokens_s_user"]))
                best_table.append(
                    [
                        variant["label"],
                        system_label,
                        str(load),
                        f"{best['layout']}{best['world_gpus']}",
                        str(best["global_concurrency"]),
                        f"{best['tokens_s_user']:.1f}",
                        f"{best['tokens_s_gpu']:.1f}",
                        f"{best['memory_gib_per_rank']:.1f}",
                    ]
                )
    lines.extend(
        _table(
            ["Variant", "System", "Local batch", "Best layout", "Global", "TPS/user", "TPS/GPU", "GiB/rank"],
            best_table,
            ["---", "---", "---:", "---", "---:", "---:", "---:", "---:"],
        )
    )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Total parameters are not enough to predict decode speed for an MoE. Expert-only 5T/10T keeps "
            "routed active work fixed; the penalties are larger weight capacity, a wider replicated router, and "
            "more difficult expert-parallel communication/load balance. With enough EP ranks, TPS/user can remain "
            "surprisingly close to K3.",
            "- The Kimi-trend variants are much slower because K2.5→K3 also increased active structure. More layers "
            "multiply KDA, MLA, shared-expert, and communication work; larger top-k raises routed MoE work directly; "
            "more heads enlarge KDA and MLA projections.",
            "- Depth-only shows why a dense-like interpretation of 5T/10T is pessimistic for MoE but useful as a "
            "bound: both active weights and per-request KDA/MLA state grow nearly linearly with model size.",
            "- 1M context can become a memory constraint even when weights are well sharded. More MLA layers grow "
            "token-linear KV, while more KDA layers grow fixed recurrent state per admitted request. TEP shards KDA "
            "state and dense projections but does not eliminate MLA KV reads.",
            "- The 22-TB/s substitution remains a sensitivity, not a Rubin benchmark. As the active model grows, "
            "HBM bandwidth remains valuable, but attention compute and wider/more frequent communication take a "
            "larger non-scaling share.",
            "- On the normalized 256K DEP64 slice, expert-only scaling retains 97%/96% of baseline single-user "
            "GB300 speed at 5T/10T. The Kimi-trend path retains 60%/34%; depth-only retains 57%/29%. The 10T "
            "Kimi-trend and depth-only DEP64 rows are over capacity.",
            "- The 10T Kimi-trend design cannot fit any swept DEP topology because DEP leaves its roughly 321 GiB "
            "of non-routed TP1 weights replicated per rank even at DEP256. It first fits as TEP24 (233 GiB/rank "
            "at 256K), which illustrates why architecture and parallelism have to be designed together.",
            "- At 256K on GB300 DEP64, the Kimi-trend slowdown is mostly active model work, not cache reading. "
            "The explicit MLA cache-attention term is 5.1% of baseline TPOT, 3.8% at 5T, and 3.0% at 10T; "
            "KDA projections, routed MoE, and shared experts dominate instead.",
            "- At fixed GB300 DEP64, moving from 256K to 1M lowers TPS/user by 13% for baseline K3, 10% for the "
            "5T Kimi-trend model, and 8% for the over-capacity 10T timing counterfactual. At the minimum-fit TEP "
            "topologies, the same context increase costs about 41–45%, because a single sharded request exposes "
            "cache-attention latency rather than amortizing model work across 64 DEP requests.",
            "",
            "## Confidence and caveats",
            "",
            "- The baseline K3 formula path has limited calibration; custom 5T/10T shapes have none. Treat absolute "
            "TPS/user as low-confidence ceilings and model-family comparisons as directional.",
            "- AIC formula mode can price arbitrary dimensions, but it does not prove that kernels, load balancing, "
            "routing quality, training stability, or these TP/EP layouts would exist in a real implementation.",
            "- DEP64+ and TEP widths beyond documented K3 recipes idealize multi-node collectives. The 10T models "
            "would likely require new parallelism, DCP, pipeline parallelism, weight offload, or more HBM capacity.",
            "- No speculative decoding is modeled. A future 5T/10T draft model would need its own measured acceptance "
            "curve and verification-cost geometry; reusing K3's DSpark values would not be justified.",
            "",
            f"Raw sweep: [`{CSV_PATH.name}`]({CSV_PATH.name}); operation breakdown: "
            f"[`{OPERATIONS_PATH.name}`]({OPERATIONS_PATH.name}); metadata: "
            f"[`{METADATA_PATH.name}`]({METADATA_PATH.name}).",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate the Markdown report from the existing sweep CSV.",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = json.loads((MODELS_DIR / "manifest.json").read_text(encoding="utf-8"))
    if args.report_only:
        rows = _read_csv(CSV_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        metadata["input_hashes"] = {
            "models_manifest": _sha256(MODELS_DIR / "manifest.json"),
            "generator": _sha256(EXPERIMENT_DIR / "generate_scaled_configs.py"),
            "sweep": _sha256(Path(__file__)),
        }
        METADATA_PATH.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _report(rows, variants, metadata)
        print(f"regenerated report from {len(rows)} existing rows")
        print(f"report={REPORT_PATH}")
        return
    rows: list[dict[str, Any]] = []
    operation_rows: list[dict[str, Any]] = []
    for variant in variants:
        valid_tep = [size for size in TEP_CANDIDATES if int(variant["num_heads"]) % size == 0]
        layouts = [("DEP", size) for size in DEP_SIZES] + [("TEP", size) for size in valid_tep]
        for system_label, system_name, bandwidth in SYSTEMS:
            for context in CONTEXTS:
                for layout, world in layouts:
                    for load in LOADS:
                        print(
                            f"{variant['id']} {system_label} ctx={context} {layout}{world} load={load}",
                            flush=True,
                        )
                        row, latency, sources = _estimate(
                            variant,
                            system_label,
                            system_name,
                            bandwidth,
                            context,
                            layout,
                            world,
                            load,
                        )
                        rows.append(row)
                        if context == 262_144 and load in (1, 4) and world in (64, 96, 128):
                            for operation, operation_ms in latency.items():
                                if operation_ms <= 0:
                                    continue
                                operation_rows.append(
                                    {
                                        "variant_id": variant["id"],
                                        "system_label": system_label,
                                        "context_tokens_per_request": context,
                                        "layout": layout,
                                        "world_gpus": world,
                                        "local_batch": load,
                                        "global_concurrency": row["global_concurrency"],
                                        "tpot_ms": row.get("tpot_ms", ""),
                                        "operation": operation,
                                        "operation_ms": operation_ms,
                                        "operation_share_pct": operation_ms / float(row["tpot_ms"]) * 100.0,
                                        "source": sources.get(operation, "unknown"),
                                    }
                                )

    _write_csv(CSV_PATH, FIELDS, rows)
    _write_csv(OPERATIONS_PATH, OP_FIELDS, operation_rows)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "aic_commit": _git_commit(AIC_REPO),
        "aiconfigurator_version": aiconfigurator.__version__,
        "aiconfigurator_core_path": str(Path(aiconfigurator_core.__file__).resolve()),
        "python": platform.python_version(),
        "backend": "trtllm",
        "backend_version": "estimate",
        "database_mode": "SOL",
        "contexts": CONTEXTS,
        "loads": LOADS,
        "dep_sizes": DEP_SIZES,
        "tep_candidates": TEP_CANDIDATES,
        "systems": SYSTEMS,
        "no_speculation": True,
        "kvcache_quant_mode": "bfloat16",
        "variants": variants,
        "rows": len(rows),
        "operation_rows": len(operation_rows),
        "pass_rows": sum(row["status"] == "PASS" for row in rows),
        "failed_rows": sum(row["status"] == "FAIL" for row in rows),
        "source_leak_rows": sum(row["status"] == "SOURCE_LEAK" for row in rows),
        "oom_rows": sum(bool(row.get("oom")) for row in rows),
        "input_hashes": {
            "models_manifest": _sha256(MODELS_DIR / "manifest.json"),
            "generator": _sha256(EXPERIMENT_DIR / "generate_scaled_configs.py"),
            "sweep": _sha256(Path(__file__)),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _report(rows, variants, metadata)
    print(
        f"completed rows={len(rows)} pass={metadata['pass_rows']} failures={metadata['failed_rows']} "
        f"source_leaks={metadata['source_leak_rows']} oom={metadata['oom_rows']}"
    )
    print(f"csv={CSV_PATH}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
