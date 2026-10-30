#!/usr/bin/env python3
"""Map calibrated achievable-TPS targets to best-spec SOL concurrency limits."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from sweep_slo_concurrency import CONTEXT, DEP, fmt, load_study, percentile, write_csv
from sweep_slo_spec_concurrency import AdaptiveEvaluator, adaptive_boundaries, adaptive_native_boundary


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
CALIBRATIONS = (
    {"name": "conservative", "efficiency": 0.4},
    {"name": "central", "efficiency": 0.5},
    {"name": "optimistic", "efficiency": 0.7},
)
ACHIEVED_TARGETS = (50, 100)
SCENARIOS = tuple(
    {
        "calibration": calibration["name"],
        "efficiency": calibration["efficiency"],
        "achieved_target_tps_user": achieved,
        "required_sol_tps_user": achieved / calibration["efficiency"],
    }
    for achieved in ACHIEVED_TARGETS
    for calibration in CALIBRATIONS
)
SOL_TARGETS = tuple(sorted({scenario["required_sol_tps_user"] for scenario in SCENARIOS}))
SYSTEM_PREFIXES = {
    "GB300": "gb300",
    "VR200 HBM": "vr200_hbm",
    "VR200 HBM+NVLink": "vr200_full",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def scenario_for_sol(target: float) -> dict[str, Any]:
    return min(SCENARIOS, key=lambda scenario: abs(scenario["required_sol_tps_user"] - target))


def expand_controlled(
    sol_rows: list[dict[str, Any]], no_spec: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    output = []
    for row in sol_rows:
        scenario = scenario_for_sol(float(row["target_tps_user"]))
        metadata = no_spec[row["architecture_id"]]
        output.append({
            "architecture_id": row["architecture_id"],
            "body_id": row["body_id"],
            "attention_id": row["attention_id"],
            "attention": row["attention"],
            "body_kind": row["body_kind"],
            "depth": row["depth"],
            "active_tier": row["active_tier"],
            "plausibility_tier": metadata["plausibility_tier"],
            "system": row["system"],
            **scenario,
            "max_cpr": row["max_cpr"],
            "max_global_concurrency": row["max_global_concurrency"],
            "edge_sol_tps_user": row["edge_tps_user"],
            "edge_projected_achieved_tps_user": (
                row["edge_tps_user"] * scenario["efficiency"] if row["edge_tps_user"] is not None else None
            ),
            "edge_nextn": row["edge_nextn"],
            "edge_acceptance_length": row["edge_acceptance_length"],
            "edge_memory_gib": row["edge_memory_gib"],
            "physical_max_cpr": row["physical_max_cpr_no_spec_memory"],
            "fraction_physical_concurrency": row["max_cpr"] / row["physical_max_cpr_no_spec_memory"],
            "clears_target_at_physical_max": row["clears_target_at_physical_max"],
        })
    return output


def expand_native(sol_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in sol_rows:
        scenario = scenario_for_sol(float(row["target_tps_user"]))
        output.append({
            "anchor_id": row["anchor_id"],
            "label": row["label"],
            "family": row["family"],
            "depth": row["depth"],
            "system": row["system"],
            **scenario,
            "max_cpr": row["max_cpr"],
            "max_global_concurrency": row["max_global_concurrency"],
            "edge_sol_tps_user": row["edge_tps_user"],
            "edge_projected_achieved_tps_user": (
                row["edge_tps_user"] * scenario["efficiency"] if row["edge_tps_user"] is not None else None
            ),
            "edge_nextn": row["edge_nextn"],
            "edge_acceptance_length": row["edge_acceptance_length"],
            "physical_max_cpr": row["physical_max_cpr_no_spec_memory"],
            "fraction_physical_concurrency": row["max_cpr"] / row["physical_max_cpr_no_spec_memory"],
            "clears_target_at_physical_max": row["clears_target_at_physical_max"],
        })
    return output


def pivot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["architecture_id"], row["calibration"], row["achieved_target_tps_user"])
        groups[key][row["system"]] = row
    output = []
    for (architecture_id, calibration, achieved), group in sorted(groups.items()):
        first = next(iter(group.values()))
        row: dict[str, Any] = {
            field: first[field]
            for field in (
                "architecture_id", "body_id", "attention_id", "attention", "body_kind", "depth",
                "active_tier", "plausibility_tier", "calibration", "efficiency",
                "achieved_target_tps_user", "required_sol_tps_user",
            )
        }
        for system, prefix in SYSTEM_PREFIXES.items():
            point = group[system]
            for field in (
                "max_cpr", "max_global_concurrency", "edge_sol_tps_user",
                "edge_projected_achieved_tps_user", "edge_nextn", "edge_acceptance_length",
                "physical_max_cpr", "fraction_physical_concurrency", "clears_target_at_physical_max",
            ):
                row[f"{prefix}_{field}"] = point[field]
        gb = row["gb300_max_cpr"]
        for prefix in ("vr200_hbm", "vr200_full"):
            upgraded = row[f"{prefix}_max_cpr"]
            row[f"{prefix}_concurrency_lift_vs_gb300"] = upgraded / gb if gb else None
            row[f"{prefix}_newly_feasible_vs_gb300"] = bool(upgraded and not gb)
        output.append(row)
    return output


def make_report(
    comparison: list[dict[str, Any]], native: list[dict[str, Any]], recommendations: dict[str, str],
) -> str:
    lines = [
        "# Calibrated achievable-concurrency sweep",
        "",
        "## Scope",
        "",
        "This maps achieved 50/100 TPS-user targets onto the best-spec AIC SOL envelope at DEP64 and 256K resident context. The efficiency factors apply to throughput at fixed concurrency: 0.4× conservative, 0.5× central, and 0.7× optimistic. Equivalently, actual TPOT is modeled as 2.5×, 2.0×, or 1.43× SOL TPOT.",
        "",
        "| Achieved target | Calibration | Required SOL target |",
        "| ---: | --- | ---: |",
    ]
    for scenario in sorted(SCENARIOS, key=lambda row: (row["achieved_target_tps_user"], row["efficiency"])):
        lines.append(
            f"| {scenario['achieved_target_tps_user']} | {scenario['calibration']} ({scenario['efficiency']:.1f}×) | "
            f"{scenario['required_sol_tps_user']:.1f} |"
        )

    lines += [
        "",
        "## Whole 448-architecture grid",
        "",
        "| Achieved target | Efficiency | Hardware | Feasible | Median c/r | P10–P90 c/r | Max c/r | Median usable physical c/r | Clears target at physical max | Median lift vs GB300 |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for achieved in ACHIEVED_TARGETS:
        for calibration in CALIBRATIONS:
            subset = [
                row for row in comparison
                if row["achieved_target_tps_user"] == achieved and row["calibration"] == calibration["name"]
            ]
            for prefix, label in (("vr200_hbm", "Rubin HBM"), ("vr200_full", "Rubin HBM+fabric")):
                cprs = [row[f"{prefix}_max_cpr"] for row in subset]
                feasible = [value for value in cprs if value > 0]
                fractions = [row[f"{prefix}_fraction_physical_concurrency"] for row in subset]
                full = sum(row[f"{prefix}_clears_target_at_physical_max"] for row in subset)
                paired = [row for row in subset if row["gb300_max_cpr"] > 0]
                lifts = [row[f"{prefix}_concurrency_lift_vs_gb300"] for row in paired]
                lines.append(
                    f"| {achieved} | {calibration['efficiency']:.1f}× | {label} | {len(feasible)}/{len(subset)} | "
                    f"{fmt(statistics.median(cprs), 1)} | {fmt(percentile(cprs, .1), 1)}–{fmt(percentile(cprs, .9), 1)} | "
                    f"{max(cprs)} | {fmt(100 * statistics.median(fractions), 0)}% | {full}/{len(subset)} | "
                    f"{fmt(statistics.median(lifts), 2)}x |"
                )

    lines += [
        "",
        "## Recommended designs on Rubin HBM+fabric",
        "",
        "Each cell is c/r (global concurrency).",
        "",
        "| Design | Achieved target | 0.4× conservative | 0.5× central | 0.7× optimistic | Physical max c/r |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    index = {
        (row["architecture_id"], row["achieved_target_tps_user"], row["calibration"]): row
        for row in comparison
    }
    for label, architecture_id in recommendations.items():
        for achieved in ACHIEVED_TARGETS:
            cells = []
            for calibration in CALIBRATIONS:
                row = index[(architecture_id, achieved, calibration["name"])]
                cpr = row["vr200_full_max_cpr"]
                cells.append(f"{cpr} ({cpr * DEP:,})")
            physical = index[(architecture_id, achieved, "central")]["vr200_full_physical_max_cpr"]
            lines.append(f"| {label} | {achieved} | {cells[0]} | {cells[1]} | {cells[2]} | {physical} |")

    lines += [
        "",
        "## Native-family anchors on Rubin HBM+fabric",
        "",
        "| Anchor | Achieved target | 0.4× conservative | 0.5× central | 0.7× optimistic | Physical max c/r |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    native_index = {
        (row["anchor_id"], row["achieved_target_tps_user"], row["calibration"]): row
        for row in native if row["system"] == "VR200 HBM+NVLink"
    }
    for anchor_id in sorted({row["anchor_id"] for row in native}):
        label = native_index[(anchor_id, 50, "central")]["label"]
        for achieved in ACHIEVED_TARGETS:
            cells = []
            for calibration in CALIBRATIONS:
                row = native_index[(anchor_id, achieved, calibration["name"])]
                cells.append(f"{row['max_cpr']} ({row['max_global_concurrency']:,})")
            physical = native_index[(anchor_id, achieved, "central")]["physical_max_cpr"]
            lines.append(f"| {label} | {achieved} | {cells[0]} | {cells[1]} | {cells[2]} | {physical} |")

    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Calibration is a global sensitivity multiplier, not an operator-by-operator achievable model. Real efficiency can change with c/r, architecture, kernel maturity, and hardware.",
        "- The 0.7× case is an optimized upside, especially uncertain for synthetic attention mechanisms and future Rubin kernels.",
        "- The common DSpark acceptance curve remains a borrowed sensitivity assumption outside Kimi-like bodies; no-spec is always available as a fallback.",
        "- Results remain decode-only SOL-derived projections with resident 256K cache and no prefill, queueing, or scheduler effects.",
        "- `achievable_concurrency_comparison.csv` contains every architecture/scenario and both c/r and global concurrency.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    study = load_study()
    specs = study.body_specs()
    refs = study.make_models(specs)
    anchors = study.make_native_anchors()
    dsa_model = study.make_dsa_validation_model()
    al_curve = study.load_dspark_al_curve()
    evaluator = AdaptiveEvaluator(study, specs, refs, dsa_model, al_curve)
    evaluator.anchors = anchors

    no_spec_rows = read_csv(OUTPUT / "slo_concurrency_comparison.csv")
    no_spec = {row["architecture_id"]: row for row in no_spec_rows}
    sol_results: list[dict[str, Any]] = []
    for body_index, body in enumerate(specs, 1):
        for system, prefix in SYSTEM_PREFIXES.items():
            physical = {
                profile_id: int(no_spec[f"{body.id}__{profile_id}"][f"{prefix}_physical_max_cpr"])
                for profile_id in study.ATTENTION_PROFILES
            }
            sol_results.extend(adaptive_boundaries(evaluator, body, system, physical, SOL_TARGETS))
            print(
                f"controlled {body_index}/{len(specs)} {body.id} {system}; "
                f"new AIC estimates={evaluator.new_estimates}", flush=True,
            )

    native_no_spec = {
        row["anchor_id"]: row for row in read_csv(OUTPUT / "slo_native_anchor_comparison.csv")
    }
    native_sol_results = []
    for anchor_id, anchor in anchors.items():
        for system, prefix in SYSTEM_PREFIXES.items():
            physical = int(native_no_spec[anchor_id][f"{prefix}_physical_max_cpr"])
            for target in SOL_TARGETS:
                native_sol_results.append(
                    adaptive_native_boundary(evaluator, anchor, system, target, physical)
                )

    controlled = expand_controlled(sol_results, no_spec)
    native = expand_native(native_sol_results)
    comparison = pivot(controlled)
    metadata = json.loads((OUTPUT / "metadata.json").read_text(encoding="utf-8"))
    write_csv(OUTPUT / "achievable_concurrency_by_system.csv", controlled)
    write_csv(OUTPUT / "achievable_concurrency_comparison.csv", comparison)
    write_csv(OUTPUT / "achievable_native_anchor_by_system.csv", native)
    (OUTPUT / "ACHIEVABLE_CONCURRENCY_REPORT.md").write_text(
        make_report(comparison, native, metadata["recommendations"]), encoding="utf-8"
    )
    (OUTPUT / "achievable_concurrency_metadata.json").write_text(json.dumps({
        "context": CONTEXT,
        "dep": DEP,
        "calibrations": CALIBRATIONS,
        "achieved_targets_tps_user": ACHIEVED_TARGETS,
        "required_sol_targets_tps_user": SOL_TARGETS,
        "controlled_architectures": len(no_spec),
        "native_anchors": len(anchors),
        "aic_estimates_added": evaluator.new_estimates,
        "acceptance_curve": al_curve,
        "search": "adaptive integer binary search with selected edge+1 validation",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Added AIC estimates: {evaluator.new_estimates}")
    print(f"Report: {OUTPUT / 'ACHIEVABLE_CONCURRENCY_REPORT.md'}")


if __name__ == "__main__":
    main()
