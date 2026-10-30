#!/usr/bin/env python3
"""Sweep cached-prefix prefill chunk size and concurrency to locate saturation."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DECODE_STUDY = ROOT / "experiments/5t-swa-architecture-study"
BASE_STUDY = ROOT / "experiments/5t-architecture-design-study"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DECODE_STUDY))
sys.path.insert(0, str(BASE_STUDY))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aic-5t-swa-prefill-saturation-mpl")

import run_prefill_study as prefill  # noqa: E402
import run_study as base  # noqa: E402
import run_swa_study as decode  # noqa: E402
from aiconfigurator.cli.api import cli_estimate  # noqa: E402


OUTPUT = HERE / "output"
CACHE = OUTPUT / "saturation_estimate_cache.jsonl"
PREFIX = 262_144
CHUNKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096)
CPRS = (1, 2, 4, 8, 16, 32, 64)
SYSTEMS = prefill.SYSTEMS
PROFILE_IDS = {
    "full_gqa",
    "gptoss_faithful",
    "inkling_faithful",
    "common_r3_w512",
    "swa5_mla",
    "swa5_dsa",
    "swa5_kda",
    "swa5_v4",
    "mla_all",
    "dsa2048_all",
    "kda_all",
    "v4_all",
}


def signature(model: str) -> str:
    return hashlib.sha256((Path(model) / "config.json").read_bytes()).hexdigest()[:16]


def key(model: str, system: str, chunk: int, cpr: int) -> str:
    return json.dumps([model, signature(model), system, PREFIX, chunk, prefill.DEP, cpr], separators=(",", ":"))


def load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE.exists():
        return {}
    return {
        row["cache_key"]: row
        for row in (json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def estimate(model: str, system: str, chunk: int, cpr: int) -> dict[str, Any]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = cli_estimate(
            model,
            base.SYSTEMS[system]["id"],
            mode="static_ctx",
            backend_name="sglang",
            backend_version="estimate",
            database_mode="SOL",
            isl=PREFIX + chunk,
            osl=1,
            prefix=PREFIX,
            batch_size=cpr,
            tp_size=1,
            pp_size=1,
            attention_dp_size=prefill.DEP,
            moe_tp_size=1,
            moe_ep_size=prefill.DEP,
            gemm_quant_mode="fp8_block",
            moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
            kvcache_quant_mode="fp8",
            fmha_quant_mode="bfloat16",
            stride=1,
            engine_step_backend="python",
            systems_paths=base.SYSTEMS_PATHS,
        )
    latency = {str(k): float(v) for k, v in result.summary.get_context_latency_dict().items()}
    sources = {str(k): str(v or "unknown").lower() for k, v in result.summary.get_context_source_dict().items()}
    return {
        "cache_key": key(model, system, chunk, cpr),
        "model": model,
        "system": system,
        "chunk": chunk,
        "cpr": cpr,
        "latency": latency,
        "sources": sources,
        "non_sol_ms": sum(value for name, value in latency.items() if value > 0 and sources.get(name) != "sol"),
    }


def run(models: list[str]) -> dict[str, dict[str, Any]]:
    cache = load_cache()
    tasks = [(model, system, chunk, cpr) for model in models for system in SYSTEMS for chunk in CHUNKS for cpr in CPRS]
    pending = [task for task in tasks if key(*task) not in cache]
    print(f"Saturation references: {len(tasks)-len(pending)} cached, {len(pending)} pending", flush=True)
    if pending:
        with CACHE.open("a", encoding="utf-8") as stream:
            for index, task in enumerate(pending, 1):
                row = estimate(*task)
                cache[row["cache_key"]] = row
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                if index % 200 == 0 or index == len(pending):
                    print(f"  completed {index}/{len(pending)}", flush=True)
    return cache


def get(cache: dict[str, dict[str, Any]], model: str, system: str, chunk: int, cpr: int) -> dict[str, Any]:
    return cache[key(model, system, chunk, cpr)]


def primitives(
    cache: dict[str, dict[str, Any]], paths: dict[str, str], v4_model: str, kimi_model: str,
    system: str, chunk: int, cpr: int,
) -> dict[str, Any]:
    refs = {rid: get(cache, path, system, chunk, cpr) for rid, path in paths.items()}
    common = refs["common_global"]
    common_total = sum(common["latency"].values())
    common_attention = prefill.gqa_attention(common["latency"])
    body = common_total - common_attention
    gqa = {rid: prefill.gqa_attention(row["latency"]) / decode.LAYERS for rid, row in refs.items()}
    v4_run = get(cache, v4_model, system, chunk, cpr)
    kimi_run = get(cache, kimi_model, system, chunk, cpr)
    dsa_run = get(cache, str(prefill.DSA_MODEL), system, chunk, cpr)
    kda, mla, attn_res = prefill.kimi_attention_primitives(kimi_run["latency"])
    dsa = (
        dsa_run["latency"].get("context_attention", 0.0)
        + dsa_run["latency"].get("context_add_norm_1", 0.0)
    ) / prefill.DSA_LAYERS
    latency = common["latency"]
    dispatch = latency.get("context_global_moe_pre_dispatch", 0.0) + latency.get(
        "context_global_moe_post_dispatch", 0.0
    )
    moe_compute = latency.get("context_global_moe", 0.0)
    body_other = body - dispatch - moe_compute
    source_rows = list(refs.values()) + [v4_run, kimi_run, dsa_run]
    return {
        "body": body,
        "dispatch": dispatch,
        "moe_compute": moe_compute,
        "body_other": body_other,
        "gqa": gqa,
        "v4": prefill.v4_attention(v4_run["latency"]) / decode.LAYERS,
        "mla": mla + attn_res,
        "kda": kda + attn_res,
        "dsa": dsa,
        "non_sol_ms": sum(row["non_sol_ms"] for row in source_rows),
    }


def compose(profile: dict[str, Any], prim: dict[str, Any], system: str, chunk: int, cpr: int) -> dict[str, Any]:
    attention = prefill.profile_attention(profile, prim)
    latency = prim["body"] + attention
    new_tokens_per_rank = chunk * cpr
    rack_tps = prefill.DEP * new_tokens_per_rank * 1000 / latency
    return {
        "architecture_id": profile["id"],
        "label": profile["label"],
        "system": system,
        "prefix_tokens": PREFIX,
        "new_tokens_per_request": chunk,
        "cpr": cpr,
        "new_tokens_per_rank": new_tokens_per_rank,
        "global_concurrency": prefill.DEP * cpr,
        "latency_ms": latency,
        "input_tps_user": chunk * 1000 / latency,
        "rack_input_tps": rack_tps,
        "body_ms": prim["body"],
        "dispatch_ms": prim["dispatch"],
        "moe_compute_ms": prim["moe_compute"],
        "body_other_ms": prim["body_other"],
        "attention_ms": attention,
        "dispatch_share_pct": 100 * prim["dispatch"] / latency,
        "moe_compute_share_pct": 100 * prim["moe_compute"] / latency,
        "attention_share_pct": 100 * attention / latency,
        "non_sol_ms": prim["non_sol_ms"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def saturation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for aid in sorted(PROFILE_IDS):
        label = next(row["label"] for row in rows if row["architecture_id"] == aid)
        out: dict[str, Any] = {"architecture_id": aid, "label": label}
        for system, short in (("GB300", "gb"), ("VR200 HBM+NVLink", "rubin")):
            selected = [row for row in rows if row["architecture_id"] == aid and row["system"] == system]
            asymptote = max(row["rack_input_tps"] for row in selected)
            out[f"{short}_observed_asymptotic_rack_tps"] = asymptote
            for threshold in (0.90, 0.95, 0.99):
                # Chunk-only path: c/r=1, so this directly answers how many new
                # tokens/request are needed without extra request concurrency.
                chunk_path = sorted((row for row in selected if row["cpr"] == 1), key=lambda row: row["new_tokens_per_request"])
                hit = next((row for row in chunk_path if row["rack_input_tps"] >= threshold * asymptote), None)
                out[f"{short}_chunk_at_{int(threshold*100)}pct"] = hit["new_tokens_per_request"] if hit else None
                # Concurrency-only path for the requested 1K chunk.
                concurrency_path = sorted(
                    (row for row in selected if row["new_tokens_per_request"] == 1_024), key=lambda row: row["cpr"]
                )
                hit = next((row for row in concurrency_path if row["rack_input_tps"] >= threshold * asymptote), None)
                out[f"{short}_cpr_1k_at_{int(threshold*100)}pct"] = hit["cpr"] if hit else None
            point = next(row for row in selected if row["new_tokens_per_request"] == 1_024 and row["cpr"] == 1)
            for key_name in (
                "latency_ms", "rack_input_tps", "dispatch_share_pct", "moe_compute_share_pct", "attention_share_pct"
            ):
                out[f"{short}_1k_c1_{key_name}"] = point[key_name]
        summaries.append(out)
    return summaries


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths, v4_model, kimi_model = decode.make_models()
    models = list(paths.values()) + [v4_model, kimi_model, str(prefill.DSA_MODEL)]
    cache = run(models)
    profiles = [profile for profile in decode.PROFILES if profile["id"] in PROFILE_IDS]
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for chunk in CHUNKS:
            for cpr in CPRS:
                prim = primitives(cache, paths, v4_model, kimi_model, system, chunk, cpr)
                rows.extend(compose(profile, prim, system, chunk, cpr) for profile in profiles)
    summary = saturation_summary(rows)
    write_csv(OUTPUT / "saturation_results.csv", rows)
    write_csv(OUTPUT / "saturation_summary.csv", summary)
    assert all(row["non_sol_ms"] == 0 for row in rows)
    print(f"Rows: {len(rows)}; summaries: {len(summary)}; non-SOL latency: 0 ms")


if __name__ == "__main__":
    main()
