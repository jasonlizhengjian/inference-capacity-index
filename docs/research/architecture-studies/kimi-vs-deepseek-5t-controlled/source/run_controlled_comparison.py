#!/usr/bin/env python3
"""Controlled 5T Kimi-K3 versus DeepSeek-V4-Pro AIC SOL comparisons."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aic-5t-controlled-mpl")

from aiconfigurator.cli.api import cli_estimate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KIMI_SOURCE = ROOT / "experiments/kimi-k3-scale-up/models/k3_5t_expert/config.json"
DS_SOURCE = ROOT / "experiments/deepseek-v4-5t-sol/models/deepseek_v4_pro_5t_expert/config.json"
MODELS = HERE / "models"
OUTPUT = HERE / "output"
CSV_PATH = OUTPUT / "controlled_5t_sol.csv"
REPORT_PATH = OUTPUT / "REPORT.md"
CONTEXT = 262_144
DEPS = (16, 64)
MATCHED_CPRS = (1, 2, 3, 4)
SYSTEMS = {
    "GB300": {"id": "gb300", "capacity_bytes": 298_013_687_808},
    "B300": {"id": "b300_sxm", "capacity_bytes": 288_400_343_040},
}

# Exact accounting used in the earlier Kimi scale-up study and rounded official
# accounting used by the earlier DeepSeek-V4-Pro study.
KIMI_BASE_TOTAL = 2_779_468_955_648
KIMI_BASE_ACTIVE = 105_348_497_408
DS_BASE_TOTAL = 1_600_000_000_000
DS_BASE_ACTIVE = 49_000_000_000
KIMI_5T_TOTAL = 4_919_229_579_264
DS_5T_TOTAL = 4_952_691_736_576

VARIANTS = {
    # Nearest integral top-k values to the original active/total ratios.
    "Kimi ratio-preserved": {
        "family": "Kimi",
        "source": KIMI_SOURCE,
        "topk": 43,
        "total": KIMI_5T_TOTAL,
        "base_ratio": KIMI_BASE_ACTIVE / KIMI_BASE_TOTAL,
        "id": "kimi_ratio_preserved",
    },
    "DeepSeek ratio-preserved": {
        "family": "DeepSeek",
        "source": DS_SOURCE,
        "topk": 31,
        "total": DS_5T_TOTAL,
        "base_ratio": DS_BASE_ACTIVE / DS_BASE_TOTAL,
        "id": "deepseek_ratio_preserved",
    },
    # Almost identical aggregate routed-MoE capacity and selection fraction:
    # Kimi 33/1600=2.0625%, DeepSeek 25/1216=2.056%. This also makes selected
    # routed-expert weights nearly equal (~100B) despite native expert shapes.
    "Kimi MoE-normalized": {
        "family": "Kimi",
        "source": KIMI_SOURCE,
        "topk": 33,
        "total": KIMI_5T_TOTAL,
        "base_ratio": KIMI_BASE_ACTIVE / KIMI_BASE_TOTAL,
        "id": "kimi_moe_normalized",
    },
    "DeepSeek MoE-normalized": {
        "family": "DeepSeek",
        "source": DS_SOURCE,
        "topk": 25,
        "total": DS_5T_TOTAL,
        "base_ratio": DS_BASE_ACTIVE / DS_BASE_TOTAL,
        "id": "deepseek_moe_normalized",
    },
}


def make_models() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    for label, variant in VARIANTS.items():
        config = json.loads(Path(variant["source"]).read_text(encoding="utf-8"))
        text = config.get("text_config", config)
        if variant["family"] == "Kimi":
            text["num_experts_per_token"] = variant["topk"]
            layers = int(text["num_hidden_layers"]) - int(text["first_k_dense_replace"])
            per_selected = layers * 3 * int(text["routed_expert_hidden_size"]) * int(text["moe_intermediate_size"])
            base_active = 105_812_754_432
            active = base_active + (variant["topk"] - 16) * per_selected
            experts = int(text["num_experts"])
            routed_total = layers * experts * 3 * int(text["routed_expert_hidden_size"]) * int(text["moe_intermediate_size"])
        else:
            text["num_experts_per_tok"] = variant["topk"]
            layers = int(text["num_hidden_layers"])
            per_selected = layers * 3 * int(text["hidden_size"]) * int(text["moe_intermediate_size"])
            base_active = 49_363_790_336
            active = base_active + (variant["topk"] - 6) * per_selected
            experts = int(text["n_routed_experts"])
            routed_total = layers * experts * 3 * int(text["hidden_size"]) * int(text["moe_intermediate_size"])
        model_dir = MODELS / variant["id"]
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        variant.update(
            path=model_dir,
            active=active,
            active_ratio=active / variant["total"],
            experts=experts,
            selection_fraction=variant["topk"] / experts,
            routed_total=routed_total,
            routed_active=per_selected * variant["topk"],
        )


def estimate(label: str, system: str, dep: int, cpr: int) -> dict[str, Any]:
    variant = VARIANTS[label]
    result = cli_estimate(
        str(variant["path"]), SYSTEMS[system]["id"],
        mode="static_gen", backend_name="sglang", backend_version="estimate",
        database_mode="SOL", isl=CONTEXT, osl=2, prefix=CONTEXT,
        batch_size=cpr, tp_size=1, pp_size=1, attention_dp_size=dep,
        moe_tp_size=1, moe_ep_size=dep, kvcache_quant_mode="fp8",
        nextn=0, stride=1, engine_step_backend="python",
    )
    latency = {str(k): float(v) for k, v in result.summary.get_generation_latency_dict().items()}
    sources = {str(k): str(v or "unknown").lower() for k, v in result.summary.get_generation_source_dict().items()}
    non_sol = sum(v for k, v in latency.items() if sources.get(k) != "sol")
    if non_sol:
        raise RuntimeError(f"Non-SOL latency in {label}/{system}/DEP{dep}/cpr{cpr}: {non_sol}")
    memory = result.summary.get_memory()
    tpot = float(result.tpot)
    capacity = SYSTEMS[system]["capacity_bytes"] / 2**30
    return {
        "variant": label, "family": variant["family"], "system": system,
        "dep": dep, "concurrency_per_rank": cpr, "global_concurrency": dep * cpr,
        "tpot_ms": tpot, "tps_user": 1000 / tpot, "tps_gpu": cpr * 1000 / tpot,
        "memory_gib": float(memory["total"]), "weights_gib": float(memory["weights"]),
        "kvcache_gib": float(memory["kvcache"]), "capacity_gib": capacity,
        "oom": bool(result.summary.check_oom()),
        "attention_ms": float(latency.get("generation_attention", 0)),
        "moe_ms": float(latency.get("generation_moe", 0)) + float(latency.get("generation_moe_overlap", 0)),
        "moe_dispatch_ms": float(latency.get("generation_moe_pre_dispatch", 0)) + float(latency.get("generation_moe_post_dispatch", 0)),
        "latency_json": json.dumps(latency, sort_keys=True),
    }


def full_boundary(label: str, system: str, dep: int, cache: dict[tuple, dict]) -> dict:
    def get(cpr: int) -> dict:
        key = (label, system, dep, cpr)
        if key not in cache:
            cache[key] = estimate(label, system, dep, cpr)
        return cache[key]
    if get(1)["oom"]:
        return get(1)
    low, high = 1, 2
    while not get(high)["oom"]:
        low, high = high, high * 2
        if high > 4096:
            raise RuntimeError("No OOM boundary found")
    while high - low > 1:
        mid = (low + high) // 2
        if get(mid)["oom"]:
            high = mid
        else:
            low = mid
    return get(low)


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if i else "---" for i in range(len(headers))) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def write_report(rows: list[dict], full: list[dict]) -> None:
    lookup = {(r["variant"], r["system"], r["dep"], r["concurrency_per_rank"]): r for r in rows}
    pair_specs = [
        ("Constant original active/total ratio", "Kimi ratio-preserved", "DeepSeek ratio-preserved"),
        ("MoE-normalized", "Kimi MoE-normalized", "DeepSeek MoE-normalized"),
    ]
    lines = [
        "# Controlled 5T Kimi K3 versus DeepSeek V4 Pro SOL comparison", "",
        "AIC SOL, SGLang, 256K resident-cache decode-only, FP8 KV, no speculative decoding, "
        "TP1/PP1/attention-DP=DEP/MoE-EP=DEP.", "",
        "## Variant definitions", "",
    ]
    definitions = []
    for label, v in VARIANTS.items():
        definitions.append([
            label, str(v["experts"]), str(v["topk"]), f"{100*v['selection_fraction']:.3f}%",
            f"{v['total']/1e12:.3f}T", f"{v['routed_total']/1e12:.3f}T",
            f"{v['routed_active']/1e9:.1f}B", f"{v['active']/1e9:.1f}B", f"{100*v['active_ratio']:.3f}%",
        ])
    lines += table(["Variant", "Experts", "top-k", "k/E", "Total", "Routed pool", "Routed active", "All active", "Active/total"], definitions)
    lines += [
        "", "The ratio-preserved pair uses the nearest integer top-k to each baseline active/total ratio: "
        f"Kimi {100*KIMI_BASE_ACTIVE/KIMI_BASE_TOTAL:.3f}% and DeepSeek {100*DS_BASE_ACTIVE/DS_BASE_TOTAL:.3f}%. "
        "The MoE-normalized pair preserves native layer counts and expert tensor shapes but matches aggregate routed capacity, "
        "selection fraction, and selected routed weights. This is an aggregate MoE control, not literally identical per-layer tensors.", "",
        "If sparsity instead means preserving each baseline's top-k/expert-count ratio, the nearest 5T choices are "
        "Kimi top-29/1600 (145.3B all-active) and DeepSeek top-19/1216 (101.7B all-active). "
        "That does not preserve active/total ratios because the fixed non-routed parts of each model were not scaled.", "",
    ]
    for title, kimi_label, ds_label in pair_specs:
        lines += [f"## {title}: matched concurrency/rank", ""]
        values = []
        for system in SYSTEMS:
            for dep in DEPS:
                for cpr in MATCHED_CPRS:
                    k = lookup[(kimi_label, system, dep, cpr)]
                    d = lookup[(ds_label, system, dep, cpr)]
                    values.append([
                        system, str(dep), str(cpr),
                        "OOM" if k["oom"] else f"{k['tps_user']:.1f}",
                        "OOM" if d["oom"] else f"{d['tps_user']:.1f}",
                        "—" if k["oom"] or d["oom"] else f"{d['tps_user']/k['tps_user']:.2f}x",
                        f"{k['memory_gib']:.1f}", f"{d['memory_gib']:.1f}",
                    ])
        lines += table(["System", "DEP", "c/r", "Kimi TPS/user", "DeepSeek TPS/user", "DS/Kimi", "Kimi GiB", "DS GiB"], values)
        lines += [""]
    lines += ["## GB300 DEP16 c/r=1 latency decomposition", ""]
    decomposition = []
    for title, kl, dl in pair_specs:
        for label in (kl, dl):
            r = lookup[(label, "GB300", 16, 1)]
            decomposition.append([
                title, r["family"], f"{r['tpot_ms']:.2f}", f"{r['moe_ms']:.2f}",
                f"{r['tpot_ms']-r['moe_ms']:.2f}", f"{r['attention_ms']:.2f}",
            ])
    lines += table(["Control", "Model", "TPOT ms", "Routed MoE ms", "Everything else ms", "Attention bucket ms"], decomposition)
    lines += [
        "", "For Kimi, AIC reports the KDA/MLA projection GEMMs separately from the attention bucket; "
        "therefore the attention-bucket column must not be treated as the whole attention-stack cost. "
        "In the MoE-normalized control, routed MoE time is nearly equal (6.27 versus 6.46 ms), "
        "while the remaining stack is 14.60 versus 3.23 ms.", "",
    ]
    lines += ["## Maximum memory-fitting concurrency", ""]
    flookup = {(r["variant"], r["system"], r["dep"]): r for r in full}
    values = []
    for title, kl, dl in pair_specs:
        for system in SYSTEMS:
            for dep in DEPS:
                k, d = flookup[(kl, system, dep)], flookup[(dl, system, dep)]
                values.append([
                    title, system, str(dep), str(k["concurrency_per_rank"]), f"{k['tps_user']:.1f}", f"{k['tps_gpu']:.0f}",
                    str(d["concurrency_per_rank"]), f"{d['tps_user']:.1f}", f"{d['tps_gpu']:.0f}", f"{d['tps_gpu']/k['tps_gpu']:.2f}x",
                ])
    lines += table(["Control", "System", "DEP", "K c/r", "K TPS/u", "K TPS/GPU", "DS c/r", "DS TPS/u", "DS TPS/GPU", "DS/K throughput"], values)
    lines += ["", "SOL is a hardware/model lower-bound projection, not achieved serving performance. OOM boundaries use raw physical capacity with no operational headroom.", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    make_models()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple, dict] = {}
    for label in VARIANTS:
        for system in SYSTEMS:
            for dep in DEPS:
                for cpr in MATCHED_CPRS:
                    cache[(label, system, dep, cpr)] = estimate(label, system, dep, cpr)
    full = [full_boundary(label, system, dep, cache) for label in VARIANTS for system in SYSTEMS for dep in DEPS]
    rows = sorted(cache.values(), key=lambda r: (r["variant"], r["system"], r["dep"], r["concurrency_per_rank"]))
    with CSV_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_report(rows, full)
    metadata = {label: {k: (str(v) if isinstance(v, Path) else v) for k, v in data.items() if k != "source"} for label, data in VARIANTS.items()}
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} estimates to {CSV_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
