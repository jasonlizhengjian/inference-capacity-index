#!/usr/bin/env python3
"""Sweep exact 256K/DEP64 concurrency limits for fixed per-user TPS SLOs."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
STUDY_PATH = HERE / "run_study.py"
CONTEXT = 262_144
DEP = 64
MAX_CPR = 280
TARGETS = (50, 100)


def load_study():
    spec = importlib.util.spec_from_file_location("architecture_study", STUDY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {STUDY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_controlled(rows: list[dict[str, Any]], systems: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["architecture_id"], row["system"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (architecture_id, system), group in sorted(groups.items()):
        group.sort(key=lambda row: row["cpr"])
        fitting = [row for row in group if not row["oom"]]
        if not fitting:
            raise RuntimeError(f"{architecture_id} does not fit at c/r=1 on {system}")
        physical = fitting[-1]
        if physical["cpr"] == MAX_CPR:
            raise RuntimeError(f"MAX_CPR={MAX_CPR} did not bracket {architecture_id} on {system}")
        capacity = systems[system]["capacity"] / 2**30
        operational = [row for row in fitting if row["memory_gib"] <= 0.9 * capacity]
        operational_edge = operational[-1] if operational else fitting[0]
        sample = group[0]
        summary: dict[str, Any] = {
            "architecture_id": architecture_id,
            "body_id": sample["body_id"],
            "attention_id": sample["attention_id"],
            "attention": sample["attention"],
            "body_kind": sample["body_kind"],
            "depth": sample["depth"],
            "active_tier": sample["active_tier"],
            "plausibility_tier": sample["plausibility_tier"],
            "estimated_total_t": sample["estimated_total_t"],
            "estimated_active_b": sample["estimated_active_b"],
            "system": system,
            "physical_max_cpr": physical["cpr"],
            "physical_max_global_concurrency": DEP * physical["cpr"],
            "physical_edge_tps_user": physical["tps_user"],
            "physical_edge_memory_gib": physical["memory_gib"],
            "operational_90pct_max_cpr": operational_edge["cpr"],
            "operational_90pct_global_concurrency": DEP * operational_edge["cpr"],
            "operational_90pct_edge_tps_user": operational_edge["tps_user"],
            "operational_90pct_edge_memory_gib": operational_edge["memory_gib"],
        }
        for target in TARGETS:
            passing = [row for row in fitting if row["tps_user"] >= target]
            edge = passing[-1] if passing else None
            cpr = int(edge["cpr"]) if edge else 0
            summary.update({
                f"max_cpr_at_{target}_tps": cpr,
                f"max_global_concurrency_at_{target}_tps": DEP * cpr,
                f"edge_tps_at_{target}_target": edge["tps_user"] if edge else None,
                f"edge_memory_gib_at_{target}_target": edge["memory_gib"] if edge else None,
                f"fraction_physical_concurrency_at_{target}_target": cpr / physical["cpr"],
                f"physical_edge_clears_{target}_tps": physical["tps_user"] >= target,
                f"operational_90pct_edge_clears_{target}_tps": operational_edge["tps_user"] >= target,
            })
        summaries.append(summary)
    return summaries


def summarize_native(rows: list[dict[str, Any]], systems: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["anchor_id"], row["system"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (anchor_id, system), group in sorted(groups.items()):
        group.sort(key=lambda row: row["cpr"])
        fitting = [row for row in group if not row["oom"]]
        physical = fitting[-1]
        if physical["cpr"] == MAX_CPR:
            raise RuntimeError(f"MAX_CPR={MAX_CPR} did not bracket native anchor {anchor_id}")
        capacity = systems[system]["capacity"] / 2**30
        operational = [row for row in fitting if row["memory_gib"] <= 0.9 * capacity]
        operational_edge = operational[-1] if operational else fitting[0]
        sample = group[0]
        summary: dict[str, Any] = {
            "anchor_id": anchor_id,
            "label": sample["label"],
            "family": sample["family"],
            "depth": sample["depth"],
            "total_t": sample["total_t"],
            "active_b": sample["active_b"],
            "system": system,
            "physical_max_cpr": physical["cpr"],
            "physical_edge_tps_user": physical["tps_user"],
            "operational_90pct_max_cpr": operational_edge["cpr"],
            "operational_90pct_edge_tps_user": operational_edge["tps_user"],
        }
        for target in TARGETS:
            passing = [row for row in fitting if row["tps_user"] >= target]
            edge = passing[-1] if passing else None
            cpr = int(edge["cpr"]) if edge else 0
            summary.update({
                f"max_cpr_at_{target}_tps": cpr,
                f"max_global_concurrency_at_{target}_tps": DEP * cpr,
                f"fraction_physical_concurrency_at_{target}_target": cpr / physical["cpr"],
            })
        summaries.append(summary)
    return summaries


def pivot_systems(
    rows: list[dict[str, Any]], id_field: str, metadata_fields: tuple[str, ...], systems: tuple[str, ...]
) -> list[dict[str, Any]]:
    index = {(row[id_field], row["system"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for item_id in sorted({row[id_field] for row in rows}):
        first = index[(item_id, systems[0])]
        row: dict[str, Any] = {field: first[field] for field in metadata_fields}
        for system in systems:
            point = index[(item_id, system)]
            prefix = {"GB300": "gb300", "VR200 HBM": "vr200_hbm", "VR200 HBM+NVLink": "vr200_full"}[system]
            for field in ("physical_max_cpr", "physical_edge_tps_user", "operational_90pct_max_cpr", "operational_90pct_edge_tps_user"):
                row[f"{prefix}_{field}"] = point[field]
            for target in TARGETS:
                row[f"{prefix}_max_cpr_at_{target}_tps"] = point[f"max_cpr_at_{target}_tps"]
                row[f"{prefix}_max_global_concurrency_at_{target}_tps"] = point[f"max_global_concurrency_at_{target}_tps"]
                row[f"{prefix}_fraction_physical_at_{target}_tps"] = point[f"fraction_physical_concurrency_at_{target}_target"]
        for target in TARGETS:
            gb = row[f"gb300_max_cpr_at_{target}_tps"]
            for prefix in ("vr200_hbm", "vr200_full"):
                upgraded = row[f"{prefix}_max_cpr_at_{target}_tps"]
                row[f"{prefix}_concurrency_lift_at_{target}_tps"] = upgraded / gb if gb else None
                row[f"{prefix}_newly_feasible_at_{target}_tps"] = bool(upgraded and not gb)
        output.append(row)
    return output


def fmt(value: float | int | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def report(
    comparison: list[dict[str, Any]], native: list[dict[str, Any]], systems: tuple[str, ...], recommendations: dict[str, str]
) -> str:
    lines = [
        "# 50/100 TPS-user concurrency scaling at 256K",
        "",
        "## Scope",
        "",
        f"This is an exact integer sweep of c/r=1..{MAX_CPR} for all {len(comparison)} controlled 5T architectures at DEP64 and 256K resident context. It reports the largest concurrency/rank that still clears 50 or 100 TPS/user. Global concurrency is c/r × 64, so hardware concurrency multipliers are identical in either unit.",
        "",
        "The primary results are no-spec AIC SOL. GB300 and both Rubin substitutions use the same 277.5-GiB effective capacity; `VR200 HBM` changes only 8→22 TB/s HBM, while `VR200 HBM+NVLink` also changes directional fabric from 0.9→1.8 TB/s. A value of c/r=0 means the design misses the target even with one request/rank. Such a GB300→Rubin transition is called newly feasible, not an infinite multiplier.",
        "",
        "## Whole design-space result",
        "",
        "| Target | Upgrade | GB-feasible pairs | Newly feasible | Median lift | P10–P90 | Max lift | Median usable fraction of physical c/r | Architectures clearing target at physical max |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    prefix_labels = (("vr200_hbm", "22-TB/s HBM only"), ("vr200_full", "HBM + 2× fabric"))
    for target in TARGETS:
        for prefix, label in prefix_labels:
            paired = [row for row in comparison if row[f"gb300_max_cpr_at_{target}_tps"] > 0]
            lifts = [row[f"{prefix}_concurrency_lift_at_{target}_tps"] for row in paired]
            newly = sum(bool(row[f"{prefix}_newly_feasible_at_{target}_tps"]) for row in comparison)
            fractions = [row[f"{prefix}_fraction_physical_at_{target}_tps"] for row in comparison]
            at_full = sum(
                row[f"{prefix}_max_cpr_at_{target}_tps"] == row[f"{prefix}_physical_max_cpr"]
                for row in comparison
            )
            lines.append(
                f"| {target} | {label} | {len(paired)}/{len(comparison)} | {newly} | "
                f"{fmt(statistics.median(lifts), 2)}x | {fmt(percentile(lifts, .1), 2)}–{fmt(percentile(lifts, .9), 2)}x | "
                f"{fmt(max(lifts), 2)}x | {fmt(100 * statistics.median(fractions), 0)}% | {at_full}/{len(comparison)} |"
            )

    lines += [
        "",
        "The paired lift statistics include only architectures already meeting the target on GB300. `Newly feasible` counts designs whose GB300 c/r is zero but whose Rubin c/r is positive. The physical-max column is the direct test of whether HBM can be filled while retaining the target rate.",
        "",
        "## Attention-profile medians across all MoE bodies",
        "",
    ]
    for target in TARGETS:
        lines += [
            f"### {target} TPS/user",
            "",
            "| Attention | GB c/r | HBM c/r | HBM lift | HBM+fabric c/r | Full lift |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for attention_id in sorted({row["attention_id"] for row in comparison}):
            subset = [row for row in comparison if row["attention_id"] == attention_id]
            paired = [row for row in subset if row[f"gb300_max_cpr_at_{target}_tps"] > 0]
            label = subset[0]["attention"]
            gb = statistics.median(row[f"gb300_max_cpr_at_{target}_tps"] for row in subset)
            hbm = statistics.median(row[f"vr200_hbm_max_cpr_at_{target}_tps"] for row in subset)
            full = statistics.median(row[f"vr200_full_max_cpr_at_{target}_tps"] for row in subset)
            hbm_lifts = [row[f"vr200_hbm_concurrency_lift_at_{target}_tps"] for row in paired]
            full_lifts = [row[f"vr200_full_concurrency_lift_at_{target}_tps"] for row in paired]
            lines.append(
                f"| {label} | {fmt(gb, 1)} | {fmt(hbm, 1)} | {fmt(statistics.median(hbm_lifts), 2)}x | "
                f"{fmt(full, 1)} | {fmt(statistics.median(full_lifts), 2)}x |"
            )
        lines.append("")

    lines += [
        "## Recommended controlled designs",
        "",
        "| Design | Target | GB c/r | HBM c/r | HBM lift | HBM+fabric c/r | Full lift | Physical max c/r |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rec_labels = {
        "aggressive": "Aggressive latency floor",
        "balanced": "Balanced recommendation",
        "conservative": "Conservative recommendation",
        "hybrid": "KDA+sparse hybrid",
        "latent": "Latent-MoE option",
        "fallback": "Full-attention fallback",
    }
    index = {row["architecture_id"]: row for row in comparison}
    for rec_id, architecture_id in recommendations.items():
        if architecture_id not in index:
            continue
        row = index[architecture_id]
        for target in TARGETS:
            gb = row[f"gb300_max_cpr_at_{target}_tps"]
            hbm = row[f"vr200_hbm_max_cpr_at_{target}_tps"]
            full = row[f"vr200_full_max_cpr_at_{target}_tps"]
            lines.append(
                f"| {rec_labels.get(rec_id, rec_id)} | {target} | {gb} | {hbm} | "
                f"{fmt(hbm / gb if gb else None, 2)}x | {full} | {fmt(full / gb if gb else None, 2)}x | "
                f"{row['gb300_physical_max_cpr']} |"
            )

    lines += [
        "",
        "## Native-family scaling anchors",
        "",
        "| Anchor | Target | GB c/r | HBM c/r | HBM lift | HBM+fabric c/r | Full lift | Physical max c/r |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in native:
        for target in TARGETS:
            gb = row[f"gb300_max_cpr_at_{target}_tps"]
            hbm = row[f"vr200_hbm_max_cpr_at_{target}_tps"]
            full = row[f"vr200_full_max_cpr_at_{target}_tps"]
            lines.append(
                f"| {row['label']} | {target} | {gb} | {hbm} | {fmt(hbm / gb if gb else None, 2)}x | "
                f"{full} | {fmt(full / gb if gb else None, 2)}x | {row['gb300_physical_max_cpr']} |"
            )

    lines += [
        "",
        "## Interpretation limits",
        "",
        "- These are hardware speed-of-light boundaries, not achieved serving concurrency. Scheduler, runtime, imbalance, queueing, and kernel inefficiency reduce the practical count.",
        "- The VR200 rows are controlled substitutions into a GB300 compute model, not a complete Rubin prediction. HBM-only therefore deliberately exposes the non-HBM ceiling; the full row tests one fabric upgrade but still retains GB300 compute.",
        "- Every request has a resident 256K cache and there is no prefill. Capacity is held equal to isolate bandwidth/fabric effects.",
        "- Medians describe coverage of this constructed 448-design grid, not probabilities that a future architecture will take each form.",
        "- The detailed per-architecture values, including zero/newly-feasible cases and fraction of physical capacity used at each target, are in `slo_concurrency_comparison.csv`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    study = load_study()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    specs = study.body_specs()
    refs = study.make_models(specs)
    anchors = study.make_native_anchors()
    models = [body.model_path for body in specs] + list(refs.values()) + [anchor["model_path"] for anchor in anchors.values()]
    sweep_points = [
        (system, CONTEXT, DEP, cpr)
        for system in study.SYSTEMS
        for cpr in range(1, MAX_CPR + 1)
    ]
    cache = study.run_estimates(models, sweep_points, {0: 1.0})

    original_points = study.points
    study.points = lambda: sweep_points
    try:
        controlled_rows = study.compose_rows(specs, refs, cache)
        native_rows = study.native_anchor_rows(anchors, cache)
    finally:
        study.points = original_points

    controlled_summary = summarize_controlled(controlled_rows, study.SYSTEMS)
    native_summary = summarize_native(native_rows, study.SYSTEMS)
    system_names = tuple(study.SYSTEMS)
    controlled_comparison = pivot_systems(
        controlled_summary,
        "architecture_id",
        ("architecture_id", "body_id", "attention_id", "attention", "body_kind", "depth", "active_tier", "plausibility_tier", "estimated_total_t", "estimated_active_b"),
        system_names,
    )
    native_comparison = pivot_systems(
        native_summary,
        "anchor_id",
        ("anchor_id", "label", "family", "depth", "total_t", "active_b"),
        system_names,
    )

    metadata = json.loads((OUTPUT / "metadata.json").read_text(encoding="utf-8"))
    write_csv(OUTPUT / "slo_concurrency_by_system.csv", controlled_summary)
    write_csv(OUTPUT / "slo_concurrency_comparison.csv", controlled_comparison)
    write_csv(OUTPUT / "slo_native_anchor_comparison.csv", native_comparison)
    (OUTPUT / "SLO_CONCURRENCY_REPORT.md").write_text(
        report(controlled_comparison, native_comparison, system_names, metadata["recommendations"]),
        encoding="utf-8",
    )
    run_metadata = {
        "context": CONTEXT,
        "dep": DEP,
        "max_cpr_swept": MAX_CPR,
        "targets_tps_user": TARGETS,
        "controlled_architectures": len(controlled_comparison),
        "native_anchors": len(native_comparison),
        "systems": study.SYSTEMS,
        "speculative_decoding": False,
        "source_study_metadata": str(OUTPUT / "metadata.json"),
    }
    (OUTPUT / "slo_concurrency_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Controlled architectures: {len(controlled_comparison)}")
    print(f"Native anchors: {len(native_comparison)}")
    print(f"Report: {OUTPUT / 'SLO_CONCURRENCY_REPORT.md'}")


if __name__ == "__main__":
    main()
