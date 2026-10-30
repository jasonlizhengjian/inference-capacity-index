#!/usr/bin/env python3
"""Extend one SOL decode curve beyond its physical memory-admission limit."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aic-unlimited-memory-curve-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sweep_slo_concurrency import CONTEXT, DEP, load_study, write_csv
from sweep_slo_spec_concurrency import no_spec_point


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
SYSTEMS = ("GB300", "VR200 HBM+NVLink")
SYSTEM_PREFIXES = {"GB300": "gb300", "VR200 HBM+NVLink": "vr200_full"}
MAX_CPR = 2048


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sample_cprs() -> list[int]:
    # Dense around the real memory boundary, progressively coarser once the
    # asymptotic regime is well established.
    return sorted(set(
        list(range(1, 257))
        + list(range(264, 513, 8))
        + list(range(528, 1025, 16))
        + list(range(1056, MAX_CPR + 1, 32))
        + [MAX_CPR]
    ))


def main() -> None:
    study = load_study()
    specs = study.body_specs()
    refs = study.make_models(specs)
    metadata = json.loads((OUTPUT / "metadata.json").read_text(encoding="utf-8"))
    architecture_id = metadata["recommendations"]["Balanced recommendation"]
    body_id, attention_id = architecture_id.split("__", 1)
    body = next(spec for spec in specs if spec.id == body_id)
    direct_balanced = {
        depth: min(
            (spec for spec in specs if spec.kind == "direct" and spec.active_tier == "balanced" and spec.depth == depth),
            key=lambda spec: abs(spec.expert_intermediate - 3072),
        )
        for depth in study.DEPTHS
    }
    capacity_row = next(
        row for row in read_csv(OUTPUT / "slo_concurrency_comparison.csv")
        if row["architecture_id"] == architecture_id
    )
    physical_max_by_system = {
        system: int(capacity_row[f"{prefix}_physical_max_cpr"])
        for system, prefix in SYSTEM_PREFIXES.items()
    }
    existing = {
        (row["system"], int(row["cpr"])): row
        for row in read_csv(OUTPUT / "latency_concurrency_curves.csv")
        if row["config_id"] == architecture_id and row["system"] in SYSTEMS
    }

    cprs = sample_cprs()
    model_paths = sorted({body.model_path, refs[body.depth], direct_balanced[body.depth].model_path})
    points = [(system, CONTEXT, DEP, cpr) for system in SYSTEMS for cpr in cprs]
    cache = study.run_estimates(model_paths, points, {0: 1.0})

    rows = []
    for system in SYSTEMS:
        physical_max = physical_max_by_system[system]
        for cpr in cprs:
            if cpr <= physical_max:
                source = existing[(system, cpr)]
                tpot = float(source["sol_tpot_ms"])
                tps = float(source["sol_tps_user"])
                nextn = int(source["best_nextn"])
                acceptance = float(source["acceptance_length"])
                memory_gib = float(source["memory_gib"])
                mode = "best-spec memory-fitting"
            else:
                point = no_spec_point(
                    study, body, attention_id, system, cpr, refs, direct_balanced, cache
                )
                tpot = point["tpot_ms"]
                tps = point["tps_user"]
                nextn = 0
                acceptance = 1.0
                memory_gib = point["memory_gib"]
                mode = "no-spec unlimited-memory counterfactual"
            rows.append({
                "architecture_id": architecture_id,
                "system": system,
                "context": CONTEXT,
                "dep": DEP,
                "cpr": cpr,
                "global_concurrency": DEP * cpr,
                "physical_max_cpr": physical_max,
                "beyond_physical_capacity": cpr > physical_max,
                "sol_tpot_ms": tpot,
                "sol_tps_user": tps,
                "aggregate_tps_rank": cpr * tps,
                "aggregate_tps_dep64": DEP * cpr * tps,
                "best_nextn": nextn,
                "acceptance_length": acceptance,
                "mode": mode,
                "modeled_memory_gib_if_available": memory_gib,
            })

    # Fit the high-c/r SOL latency slope. If L(c)=a+b*c, aggregate rack
    # throughput asymptotically approaches DEP*1000/b.
    fits = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        tail = system_rows[-20:]
        xs = [row["cpr"] for row in tail]
        ys = [row["sol_tpot_ms"] for row in tail]
        xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
        slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum((x - xbar) ** 2 for x in xs)
        fits[system] = {
            "intercept_ms": ybar - slope * xbar,
            "slope_ms_per_cpr": slope,
            "asymptotic_dep64_tps": DEP * 1000 / slope,
        }

    fig, latency_ax = plt.subplots(figsize=(12.5, 7.2))
    throughput_ax = latency_ax.twinx()
    shared_physical_max = physical_max_by_system["GB300"]
    latency_ax.axvspan(shared_physical_max, MAX_CPR, color="#f3ead8", alpha=0.55, zorder=0)
    latency_ax.axvline(shared_physical_max, color="#8b5e34", linestyle=(0, (5, 4)), linewidth=1.7)
    styles = {
        "GB300": {"latency": "#5a6472", "throughput": "#d47728", "label": "GB300 (Blackwell)"},
        "VR200 HBM+NVLink": {"latency": "#276fbf", "throughput": "#159a77", "label": "Rubin HBM + fabric"},
    }
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        style = styles[system]
        latency_ax.plot(
            [row["cpr"] for row in system_rows], [row["sol_tpot_ms"] for row in system_rows],
            color=style["latency"], linewidth=2.5, label=f"{style['label']} SOL TPOT",
        )
        throughput_ax.plot(
            [row["cpr"] for row in system_rows], [row["aggregate_tps_dep64"] for row in system_rows],
            color=style["throughput"], linewidth=2.5, label=f"{style['label']} aggregate TPS",
        )
        asymptote = fits[system]["asymptotic_dep64_tps"]
        throughput_ax.axhline(
            asymptote, color=style["throughput"], linestyle=(0, (3, 3)), linewidth=1.2, alpha=0.75,
            label=f"{style['label']} asymptote ≈ {asymptote / 1000:.0f}K",
        )
    latency_ax.text(
        shared_physical_max + 18, max(row["sol_tpot_ms"] for row in rows) * 0.90,
        f"Modeled memory limit for both\nc/r={shared_physical_max} ({shared_physical_max * DEP:,} requests)",
        color="#6f451f", fontsize=9, va="top",
    )
    latency_ax.text(
        shared_physical_max + (MAX_CPR - shared_physical_max) * 0.52,
        max(row["sol_tpot_ms"] for row in rows) * 0.96,
        "Counterfactual unlimited-memory region", color="#8b5e34", fontsize=10, ha="center",
    )
    latency_ax.set_xlim(0, MAX_CPR)
    latency_ax.set_ylim(bottom=0)
    throughput_ax.set_ylim(bottom=0)
    latency_ax.set_xlabel("Concurrency/rank (global concurrency = 64×)")
    latency_ax.set_ylabel("AIC SOL TPOT (ms/token)")
    throughput_ax.set_ylabel("Aggregate DEP64 throughput (tokens/s)")
    latency_ax.grid(True, color="#dfe4ea", linewidth=0.7)
    latency_ax.set_axisbelow(True)
    handles1, labels1 = latency_ax.get_legend_handles_labels()
    handles2, labels2 = throughput_ax.get_legend_handles_labels()
    latency_ax.legend(handles1 + handles2, labels1 + labels2, loc="lower right", frameon=False)
    fig.suptitle("Balanced 5T V4: Blackwell vs Rubin with unlimited memory", fontsize=16, fontweight="bold")
    latency_ax.set_title(
        "DEP64 · 256K resident context · raw AIC SOL · best spec through real capacity, no-spec thereafter",
        fontsize=10, color="#4d5661", pad=12,
    )
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"balanced_v4_unlimited_memory_curve.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)

    write_csv(OUTPUT / "balanced_v4_unlimited_memory_curve.csv", rows)
    (OUTPUT / "balanced_v4_unlimited_memory_metadata.json").write_text(json.dumps({
        "architecture_id": architecture_id,
        "systems": SYSTEMS,
        "context": CONTEXT,
        "dep": DEP,
        "physical_max_cpr": physical_max_by_system,
        "counterfactual_max_cpr": MAX_CPR,
        "high_cpr_fits": fits,
        "interpretation": "Memory capacity ignored beyond physical_max_cpr; hardware bandwidth, compute, and fabric unchanged.",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Points: {len(rows)}")
    print(f"Physical max c/r: {physical_max_by_system}")
    for system in SYSTEMS:
        final = next(row for row in reversed(rows) if row["system"] == system)
        print(
            f"{system}: final c/r={final['cpr']}; TPOT={final['sol_tpot_ms']:.2f} ms; "
            f"aggregate={final['aggregate_tps_dep64']:.0f} TPS; "
            f"asymptote={fits[system]['asymptotic_dep64_tps']:.0f} TPS"
        )
    print(f"Plot: {OUTPUT / 'balanced_v4_unlimited_memory_curve.png'}")


if __name__ == "__main__":
    main()
