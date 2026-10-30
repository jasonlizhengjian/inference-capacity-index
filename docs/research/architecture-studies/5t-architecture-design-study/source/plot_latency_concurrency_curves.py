#!/usr/bin/env python3
"""Generate calibrated latency-versus-concurrency curves for key 5T designs."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aic-latency-curves-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sweep_slo_concurrency import CONTEXT, DEP, load_study, write_csv
from sweep_slo_spec_concurrency import AdaptiveEvaluator


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
EFFICIENCIES = (0.4, 0.5, 0.7)
SYSTEM_PREFIXES = {
    "GB300": "gb300",
    "VR200 HBM": "vr200_hbm",
    "VR200 HBM+NVLink": "vr200_full",
}
SYSTEM_LABELS = {
    "GB300": "GB300",
    "VR200 HBM": "Rubin HBM only",
    "VR200 HBM+NVLink": "Rubin HBM + fabric",
}
SYSTEM_COLORS = {
    "GB300": "#5a6472",
    "VR200 HBM": "#e68632",
    "VR200 HBM+NVLink": "#159a77",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def selected_configs(metadata: dict[str, Any], anchors: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    recs = metadata["recommendations"]
    return [
        {"id": recs["Balanced recommendation"], "label": "Balanced V4", "kind": "controlled"},
        {"id": recs["Conservative recommendation"], "label": "Conservative V4", "kind": "controlled"},
        {"id": recs["KDA+sparse hybrid"], "label": "KDA + V4 hybrid", "kind": "controlled"},
        {"id": "deepseek_ratio", "label": anchors["deepseek_ratio"]["label"], "kind": "native"},
        {"id": "kimi_ratio", "label": anchors["kimi_ratio"]["label"], "kind": "native"},
        {"id": "glm52_ratio", "label": anchors["glm52_ratio"]["label"], "kind": "native"},
    ]


def collect_rows(
    study: Any, evaluator: AdaptiveEvaluator, specs: list[Any], anchors: dict[str, dict[str, Any]],
    configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    controlled_capacity = {
        row["architecture_id"]: row
        for row in read_csv(OUTPUT / "slo_concurrency_comparison.csv")
    }
    native_capacity = {
        row["anchor_id"]: row
        for row in read_csv(OUTPUT / "slo_native_anchor_comparison.csv")
    }
    bodies = {spec.id: spec for spec in specs}
    rows: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs, 1):
        for system, prefix in SYSTEM_PREFIXES.items():
            if config["kind"] == "controlled":
                capacity = controlled_capacity[config["id"]]
                physical = int(capacity[f"{prefix}_physical_max_cpr"])
                body_id, attention_id = config["id"].split("__", 1)
                body = bodies[body_id]
                for cpr in range(1, physical + 1):
                    point = evaluator.controlled(body, system, cpr)[attention_id]
                    row = {
                        "config_id": config["id"], "config_label": config["label"],
                        "config_kind": config["kind"], "body_id": body_id,
                        "attention_id": attention_id, "system": system,
                        "cpr": cpr, "global_concurrency": cpr * DEP,
                        "physical_max_cpr": physical,
                        "sol_tpot_ms": point["tpot_ms"], "sol_tps_user": point["tps_user"],
                        "best_nextn": point["nextn"], "acceptance_length": point["acceptance_length"],
                        "memory_gib": point["memory_gib"],
                    }
                    for efficiency in EFFICIENCIES:
                        suffix = str(efficiency).replace(".", "p")
                        row[f"achieved_tpot_ms_at_{suffix}x"] = point["tpot_ms"] / efficiency
                        row[f"achieved_tps_user_at_{suffix}x"] = point["tps_user"] * efficiency
                    rows.append(row)
            else:
                capacity = native_capacity[config["id"]]
                physical = int(capacity[f"{prefix}_physical_max_cpr"])
                anchor = anchors[config["id"]]
                for cpr in range(1, physical + 1):
                    point = evaluator.native(anchor["model_path"], system, cpr)
                    row = {
                        "config_id": config["id"], "config_label": config["label"],
                        "config_kind": config["kind"], "body_id": config["id"],
                        "attention_id": anchor["family"], "system": system,
                        "cpr": cpr, "global_concurrency": cpr * DEP,
                        "physical_max_cpr": physical,
                        "sol_tpot_ms": point["tpot_ms"], "sol_tps_user": point["tps_user"],
                        "best_nextn": point["nextn"], "acceptance_length": point["acceptance_length"],
                        "memory_gib": point["memory_gib"],
                    }
                    for efficiency in EFFICIENCIES:
                        suffix = str(efficiency).replace(".", "p")
                        row[f"achieved_tpot_ms_at_{suffix}x"] = point["tpot_ms"] / efficiency
                        row[f"achieved_tps_user_at_{suffix}x"] = point["tps_user"] * efficiency
                    rows.append(row)
            print(
                f"curves {config_index}/{len(configs)} {config['label']} {system}; "
                f"new AIC estimates={evaluator.new_estimates}", flush=True,
            )
    return rows


def setup_panel(ax: Any, title: str, max_cpr: int) -> None:
    ax.axhline(10, color="#9b59b6", linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.8)
    ax.axhline(20, color="#b55246", linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.8)
    ax.text(max_cpr * 0.985, 10.6, "100 TPS", color="#7d3c98", fontsize=8, ha="right")
    ax.text(max_cpr * 0.985, 20.6, "50 TPS", color="#943126", fontsize=8, ha="right")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, max_cpr * 1.02)
    ax.grid(True, color="#dfe4ea", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def plot_hardware(rows: list[dict[str, Any]], configs: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["config_id"], row["system"])].append(row)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), sharey=True)
    overall_max = max(row["sol_tpot_ms"] for row in rows)
    for ax, config in zip(axes.flat, configs):
        max_cpr = max(row["physical_max_cpr"] for system in SYSTEM_PREFIXES for row in grouped[(config["id"], system)])
        setup_panel(ax, config["label"], max_cpr)
        for system in SYSTEM_PREFIXES:
            points = sorted(grouped[(config["id"], system)], key=lambda row: row["cpr"])
            ax.plot(
                [row["cpr"] for row in points],
                [row["sol_tpot_ms"] for row in points],
                label=SYSTEM_LABELS[system], color=SYSTEM_COLORS[system], linewidth=2.0,
            )
            ax.scatter(
                [points[-1]["cpr"]], [points[-1]["sol_tpot_ms"]],
                color=SYSTEM_COLORS[system], s=20, zorder=3,
            )
        ax.set_xlabel("Concurrency/rank (global = 64×)")
        ax.set_ylim(0, overall_max * 1.08)
    for ax in axes[:, 0]:
        ax.set_ylabel("AIC SOL TPOT (ms/token)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("AIC SOL latency vs concurrency", fontsize=16, fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.965,
        "DEP64 · 256K resident context · best speculative depth at each point · endpoint marker = physical memory limit",
        ha="center", va="top", fontsize=9, color="#4d5661",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.925))
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"latency_vs_concurrency_sol.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(rows: list[dict[str, Any]], configs: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["system"] == "VR200 HBM+NVLink"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[row["config_id"]].append(row)
    colors = {0.4: "#b55246", 0.5: "#276fbf", 0.7: "#159a77"}
    labels = {0.4: "0.4× conservative", 0.5: "0.5× central", 0.7: "0.7× optimistic"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), sharey=True)
    overall_max = max(row["achieved_tpot_ms_at_0p4x"] for row in selected)
    for ax, config in zip(axes.flat, configs):
        points = sorted(grouped[config["id"]], key=lambda row: row["cpr"])
        setup_panel(ax, config["label"], points[-1]["physical_max_cpr"])
        ax.set_title("")
        ax.text(
            0.02, 0.96, config["label"], transform=ax.transAxes,
            ha="left", va="top", fontsize=10, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
        )
        for efficiency in EFFICIENCIES:
            suffix = str(efficiency).replace(".", "p")
            ax.plot(
                [row["cpr"] for row in points],
                [row[f"achieved_tpot_ms_at_{suffix}x"] for row in points],
                label=labels[efficiency], color=colors[efficiency], linewidth=2.0,
            )
        ax.set_xlabel("Concurrency/rank (global = 64×)")
        ax.set_ylim(0, overall_max * 1.08)
    for ax in axes[:, 0]:
        ax.set_ylabel("Projected achieved TPOT (ms/token)")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Rubin HBM + fabric: calibration sensitivity", fontsize=16, fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.965,
        "DEP64 · 256K resident context · best speculative depth at each point",
        ha="center", va="top", fontsize=9, color="#4d5661",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.925))
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"latency_vs_concurrency_rubin_calibration.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_report(configs: list[dict[str, Any]], spec_slo: list[dict[str, str]]) -> None:
    lookup = {
        (row["architecture_id"], row["target_tps_user"]): row
        for row in spec_slo
    }
    lines = [
        "# Latency-versus-concurrency curves",
        "",
        "All curves are DEP64 decode-only projections with a resident 256K context and no prefill. Every integer c/r through the physical memory boundary is directly represented. The selected point at each c/r is the fastest memory-fitting speculative depth under the common DSpark AL assumption, including no-spec fallback.",
        "",
        "These figures contain raw AIC SOL TPOT only. No achievable-performance efficiency multiplier is applied.",
        "",
        "## SOL hardware comparison",
        "",
        "![SOL hardware latency curves](latency_vs_concurrency_sol.png)",
        "",
        "## Rubin HBM+fabric threshold summary",
        "",
        "Each cell is maximum c/r; global concurrency is c/r × 64.",
        "",
        "| Configuration | Target | GB300 | Rubin HBM | Rubin HBM+fabric | Physical max on Rubin |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config in configs:
        if config["kind"] != "controlled":
            continue
        for target in (50, 100):
            row = lookup[(config["id"], str(target))]
            values = [row[f"{prefix}_spec_max_cpr"] for prefix in ("gb300", "vr200_hbm", "vr200_full")]
            lines.append(
                f"| {config['label']} | {target} | {values[0]} | {values[1]} | {values[2]} | "
                f"{row['vr200_full_physical_max_cpr']} |"
            )
    lines += [
        "",
        "## Reading the curves",
        "",
        "- The 10-ms line is 100 TPS/user; the 20-ms line is 50 TPS/user.",
        "- A line crossing a threshold gives the maximum c/r at that achieved rate.",
        "- Endpoint markers are hard modeled memory boundaries, not recommended production allocations.",
        "- Kinks can occur when AIC changes the fastest speculative depth or a kernel regime changes.",
        "- These are SOL latency curves, not achieved serving latency forecasts.",
        "",
        "Raw points are in `latency_concurrency_curves.csv`; SVG versions are available for lossless inspection.",
        "",
    ]
    (OUTPUT / "LATENCY_CONCURRENCY_CURVES.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    study = load_study()
    specs = study.body_specs()
    refs = study.make_models(specs)
    anchors = study.make_native_anchors()
    dsa_model = study.make_dsa_validation_model()
    al_curve = study.load_dspark_al_curve()
    evaluator = AdaptiveEvaluator(study, specs, refs, dsa_model, al_curve)
    evaluator.anchors = anchors
    metadata = json.loads((OUTPUT / "metadata.json").read_text(encoding="utf-8"))
    configs = selected_configs(metadata, anchors)
    rows = collect_rows(study, evaluator, specs, anchors, configs)
    write_csv(OUTPUT / "latency_concurrency_curves.csv", rows)
    plot_hardware(rows, configs)
    spec_slo = read_csv(OUTPUT / "slo_spec_concurrency_comparison.csv")
    write_report(configs, spec_slo)
    (OUTPUT / "latency_concurrency_curves_metadata.json").write_text(json.dumps({
        "context": CONTEXT,
        "dep": DEP,
        "graph_metric": "raw AIC SOL TPOT",
        "configs": configs,
        "systems": tuple(SYSTEM_PREFIXES),
        "aic_estimates_added": evaluator.new_estimates,
        "speculative_decoding": "best nextn 0..8 at each point",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Curve rows: {len(rows)}")
    print(f"Added AIC estimates: {evaluator.new_estimates}")
    print(f"Report: {OUTPUT / 'LATENCY_CONCURRENCY_CURVES.md'}")


if __name__ == "__main__":
    main()
