#!/usr/bin/env python3
"""Find spec-aware 50/100 TPS-user concurrency limits by adaptive exact search."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from sweep_slo_concurrency import CONTEXT, DEP, TARGETS, fmt, load_study, percentile, write_csv


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def no_spec_point(
    study: Any, spec: Any, profile_id: str, system: str, cpr: int,
    refs: dict[int, str], direct_balanced: dict[int, Any], cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    body_run = cache[study.cache_key(spec.model_path, system, CONTEXT, DEP, cpr)]
    ref_run = cache[study.cache_key(refs[spec.depth], system, CONTEXT, DEP, cpr)]
    ds_ref = cache[study.cache_key(direct_balanced[spec.depth].model_path, system, CONTEXT, DEP, cpr)]
    native_attn = (
        study.v4_attention_value(body_run["latency"])
        if spec.kind == "direct" else study.kimi_attention_value(body_run["latency"])
    )
    common_body = body_run["tpot_ms"] - native_attn
    native_w, _ = study.body_native_weight_cache(spec, CONTEXT)
    base_weight_bytes = body_run["weights_gib"] * 2**30 - native_w
    non_weight_cache = body_run["memory_gib"] - body_run["weights_gib"] - body_run["kvcache_gib"]
    v4_attn = study.v4_attention_value(ds_ref["latency"])
    attn, attn_w, attn_kv, _ = study.attention_metrics(
        profile_id, system, CONTEXT, cpr, spec.depth, ref_run["latency"], v4_attn
    )
    tpot = common_body + attn
    memory = non_weight_cache + (base_weight_bytes + attn_w) / 2**30 + cpr * attn_kv / 2**30
    return {
        "nextn": 0,
        "acceptance_length": 1.0,
        "tps_user": 1000 / tpot,
        "tpot_ms": tpot,
        "memory_gib": memory,
        "oom": memory > study.SYSTEMS[system]["capacity"] / 2**30,
    }


def best_spec_point(
    study: Any, spec: Any, profile_id: str, system: str, cpr: int,
    refs: dict[int, str], direct_balanced: dict[int, Any], dsa_model: str,
    al_curve: dict[int, float], cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base = no_spec_point(study, spec, profile_id, system, cpr, refs, direct_balanced, cache)
    candidates = [base] if not base["oom"] else []
    native_w, native_kv = study.body_native_weight_cache(spec, CONTEXT)
    capacity = study.SYSTEMS[system]["capacity"] / 2**30
    for nextn, al in al_curve.items():
        if nextn == 0:
            continue
        body_run = cache[study.cache_key(spec.model_path, system, CONTEXT, DEP, cpr, nextn, al)]
        ref_run = cache[study.cache_key(refs[spec.depth], system, CONTEXT, DEP, cpr, nextn, al)]
        ds_ref = cache[study.cache_key(direct_balanced[spec.depth].model_path, system, CONTEXT, DEP, cpr, nextn, al)]
        dsa_ref = cache[study.cache_key(dsa_model, system, CONTEXT, DEP, cpr, nextn, al)]
        native_attn = (
            study.v4_attention_value(body_run["latency"])
            if spec.kind == "direct" else study.kimi_attention_value(body_run["latency"])
        )
        common_body_step = body_run["verification_step_ms"] - native_attn
        base_weight_bytes = body_run["weights_gib"] * 2**30 - native_w
        non_weight_cache = body_run["memory_gib"] - body_run["weights_gib"] - body_run["kvcache_gib"]
        spec_kv_overhead = body_run["kvcache_gib"] - cpr * native_kv / 2**30
        v4_attn = study.v4_attention_value(ds_ref["latency"])
        dsa_base = study.dsa_latency(system, CONTEXT, cpr, 78, 2048, shared_index=False)
        dsa_scale = dsa_ref["latency"]["generation_attention"] / dsa_base
        attn, attn_w, attn_kv, _ = study.attention_metrics(
            profile_id, system, CONTEXT, cpr, spec.depth, ref_run["latency"], v4_attn, dsa_scale
        )
        step_ms = common_body_step + attn
        memory = (
            non_weight_cache + (base_weight_bytes + attn_w) / 2**30
            + cpr * attn_kv / 2**30 + spec_kv_overhead
        )
        candidate = {
            "nextn": nextn,
            "acceptance_length": al,
            "tps_user": 1000 * al / step_ms,
            "tpot_ms": step_ms / al,
            "memory_gib": memory,
            "oom": memory > capacity,
        }
        if not candidate["oom"]:
            candidates.append(candidate)
    if not candidates:
        return base
    return max(candidates, key=lambda row: row["tps_user"])


class AdaptiveEvaluator:
    def __init__(self, study: Any, specs: list[Any], refs: dict[int, str], dsa_model: str, al_curve: dict[int, float]):
        self.study = study
        self.refs = refs
        self.dsa_model = dsa_model
        self.al_curve = al_curve
        self.cache = study.load_cache()
        self.direct_balanced = {
            depth: min(
                (body for body in specs if body.kind == "direct" and body.active_tier == "balanced" and body.depth == depth),
                key=lambda body: abs(body.expert_intermediate - 3072),
            )
            for depth in study.DEPTHS
        }
        self.new_estimates = 0

    def ensure(self, body: Any, system: str, cpr: int) -> None:
        models = {
            body.model_path,
            self.refs[body.depth],
            self.direct_balanced[body.depth].model_path,
            self.dsa_model,
        }
        pending = []
        for model in models:
            for nextn, al in self.al_curve.items():
                if nextn == 0:
                    continue
                key = self.study.cache_key(model, system, CONTEXT, DEP, cpr, nextn, al)
                if key not in self.cache:
                    pending.append((model, system, CONTEXT, DEP, cpr, nextn, al))
        if not pending:
            return
        with self.study.CACHE.open("a", encoding="utf-8") as stream:
            for task in pending:
                row = self.study.estimate(*task)
                self.cache[row["cache_key"]] = row
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                self.new_estimates += 1

    def controlled(self, body: Any, system: str, cpr: int) -> dict[str, dict[str, Any]]:
        self.ensure(body, system, cpr)
        return {
            profile_id: best_spec_point(
                self.study, body, profile_id, system, cpr, self.refs, self.direct_balanced,
                self.dsa_model, self.al_curve, self.cache,
            )
            for profile_id in self.study.ATTENTION_PROFILES
        }

    def ensure_native(self, model: str, system: str, cpr: int) -> None:
        pending = []
        for nextn, al in self.al_curve.items():
            if nextn == 0:
                continue
            key = self.study.cache_key(model, system, CONTEXT, DEP, cpr, nextn, al)
            if key not in self.cache:
                pending.append((model, system, CONTEXT, DEP, cpr, nextn, al))
        if not pending:
            return
        with self.study.CACHE.open("a", encoding="utf-8") as stream:
            for task in pending:
                row = self.study.estimate(*task)
                self.cache[row["cache_key"]] = row
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                self.new_estimates += 1

    def native(self, model: str, system: str, cpr: int) -> dict[str, Any]:
        self.ensure_native(model, system, cpr)
        base = self.cache[self.study.cache_key(model, system, CONTEXT, DEP, cpr)]
        candidates = [{
            "nextn": 0, "acceptance_length": 1.0, "tps_user": 1000 / base["tpot_ms"],
            "tpot_ms": base["tpot_ms"], "memory_gib": base["memory_gib"], "oom": base["oom"],
        }]
        for nextn, al in self.al_curve.items():
            if nextn == 0:
                continue
            run = self.cache[self.study.cache_key(model, system, CONTEXT, DEP, cpr, nextn, al)]
            if not run["oom"]:
                candidates.append({
                    "nextn": nextn, "acceptance_length": al, "tps_user": 1000 / run["tpot_ms"],
                    "tpot_ms": run["tpot_ms"], "memory_gib": run["memory_gib"], "oom": run["oom"],
                })
        return max(candidates, key=lambda row: row["tps_user"])


def adaptive_boundaries(
    evaluator: AdaptiveEvaluator, body: Any, system: str, physical_by_profile: dict[str, int],
    targets: tuple[float, ...] = TARGETS,
) -> list[dict[str, Any]]:
    states = {
        (profile_id, target): {"lo": 0, "hi": physical_by_profile[profile_id] + 1}
        for profile_id in evaluator.study.ATTENTION_PROFILES
        for target in targets
    }
    evaluated: dict[int, dict[str, dict[str, Any]]] = {}
    while any(state["hi"] - state["lo"] > 1 for state in states.values()):
        mids = {
            (state["lo"] + state["hi"]) // 2
            for state in states.values() if state["hi"] - state["lo"] > 1
        }
        for cpr in sorted(mids):
            if cpr not in evaluated:
                evaluated[cpr] = evaluator.controlled(body, system, cpr)
        for (profile_id, target), state in states.items():
            if state["hi"] - state["lo"] <= 1:
                continue
            mid = (state["lo"] + state["hi"]) // 2
            point = evaluated[mid][profile_id]
            if not point["oom"] and point["tps_user"] >= target:
                state["lo"] = mid
            else:
                state["hi"] = mid

    output = []
    for (profile_id, target), state in states.items():
        cpr = state["lo"]
        point = evaluated[cpr][profile_id] if cpr else None
        next_point = None
        if cpr < physical_by_profile[profile_id]:
            if cpr + 1 not in evaluated:
                evaluated[cpr + 1] = evaluator.controlled(body, system, cpr + 1)
            next_point = evaluated[cpr + 1][profile_id]
            if not next_point["oom"] and next_point["tps_user"] >= target:
                raise RuntimeError(f"Boundary validation failed for {body.id}/{profile_id}/{system}/{target}")
        output.append({
            "architecture_id": f"{body.id}__{profile_id}",
            "body_id": body.id,
            "attention_id": profile_id,
            "attention": evaluator.study.ATTENTION_PROFILES[profile_id]["label"],
            "body_kind": body.kind,
            "depth": body.depth,
            "active_tier": body.active_tier,
            "system": system,
            "target_tps_user": target,
            "max_cpr": cpr,
            "max_global_concurrency": DEP * cpr,
            "edge_tps_user": point["tps_user"] if point else None,
            "edge_nextn": point["nextn"] if point else None,
            "edge_acceptance_length": point["acceptance_length"] if point else None,
            "edge_memory_gib": point["memory_gib"] if point else None,
            "physical_max_cpr_no_spec_memory": physical_by_profile[profile_id],
            "clears_target_at_physical_max": cpr == physical_by_profile[profile_id],
            "next_cpr_tps_user": next_point["tps_user"] if next_point else None,
            "next_cpr_oom": next_point["oom"] if next_point else None,
        })
    return output


def adaptive_native_boundary(
    evaluator: AdaptiveEvaluator, anchor: dict[str, Any], system: str, target: int, physical: int,
) -> dict[str, Any]:
    lo, hi = 0, physical + 1
    evaluated: dict[int, dict[str, Any]] = {}
    while hi - lo > 1:
        mid = (lo + hi) // 2
        evaluated[mid] = evaluator.native(anchor["model_path"], system, mid)
        point = evaluated[mid]
        if not point["oom"] and point["tps_user"] >= target:
            lo = mid
        else:
            hi = mid
    point = evaluated.get(lo) if lo else None
    if lo and point is None:
        point = evaluator.native(anchor["model_path"], system, lo)
    if lo < physical:
        next_point = evaluator.native(anchor["model_path"], system, lo + 1)
        if not next_point["oom"] and next_point["tps_user"] >= target:
            raise RuntimeError(f"Native boundary validation failed for {anchor['label']}/{system}/{target}")
    return {
        "anchor_id": next(key for key, value in evaluator.anchors.items() if value is anchor),
        "label": anchor["label"], "family": anchor["family"], "depth": anchor["depth"],
        "system": system, "target_tps_user": target, "max_cpr": lo,
        "max_global_concurrency": DEP * lo,
        "edge_tps_user": point["tps_user"] if point else None,
        "edge_nextn": point["nextn"] if point else None,
        "edge_acceptance_length": point["acceptance_length"] if point else None,
        "physical_max_cpr_no_spec_memory": physical,
        "clears_target_at_physical_max": lo == physical,
    }


def pivot(rows: list[dict[str, Any]], no_spec: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[(row["architecture_id"], row["target_tps_user"])][row["system"]] = row
    output = []
    prefixes = {"GB300": "gb300", "VR200 HBM": "vr200_hbm", "VR200 HBM+NVLink": "vr200_full"}
    for (architecture_id, target), group in sorted(groups.items()):
        first = next(iter(group.values()))
        row = {field: first[field] for field in ("architecture_id", "body_id", "attention_id", "attention", "body_kind", "depth", "active_tier")}
        row["target_tps_user"] = target
        old = no_spec[architecture_id]
        for system, prefix in prefixes.items():
            point = group[system]
            spec_cpr = point["max_cpr"]
            base_cpr = int(old[f"{prefix}_max_cpr_at_{target}_tps"])
            row[f"{prefix}_no_spec_max_cpr"] = base_cpr
            row[f"{prefix}_spec_max_cpr"] = spec_cpr
            row[f"{prefix}_spec_vs_no_spec_lift"] = spec_cpr / base_cpr if base_cpr else None
            row[f"{prefix}_newly_feasible_from_spec"] = bool(spec_cpr and not base_cpr)
            row[f"{prefix}_edge_tps_user"] = point["edge_tps_user"]
            row[f"{prefix}_edge_nextn"] = point["edge_nextn"]
            row[f"{prefix}_edge_acceptance_length"] = point["edge_acceptance_length"]
            row[f"{prefix}_clears_at_physical_max"] = point["clears_target_at_physical_max"]
            row[f"{prefix}_physical_max_cpr"] = point["physical_max_cpr_no_spec_memory"]
        gb = row["gb300_spec_max_cpr"]
        for prefix in ("vr200_hbm", "vr200_full"):
            upgraded = row[f"{prefix}_spec_max_cpr"]
            row[f"{prefix}_spec_hardware_lift_vs_gb300"] = upgraded / gb if gb else None
            row[f"{prefix}_spec_newly_feasible_vs_gb300"] = bool(upgraded and not gb)
        output.append(row)
    return output


def make_report(
    comparison: list[dict[str, Any]], native_rows: list[dict[str, Any]], recommendations: dict[str, str], al_curve: dict[int, float]
) -> str:
    lines = [
        "# Spec-aware 50/100 TPS-user concurrency scaling at 256K",
        "",
        "## Scope",
        "",
        "This extends the exact no-spec SLO sweep with the common Kimi-K3 DSpark acceptance-length curve. At each adaptively searched c/r, it evaluates nextn 1–8 and selects the fastest memory-fitting result, with no-spec as a fallback. Each reported boundary and the immediately following failing c/r were directly evaluated.",
        "",
        "| nextn | " + " | ".join(str(value) for value in al_curve) + " |",
        "| ---: | " + " | ".join("---:" for _ in al_curve) + " |",
        "| AL | " + " | ".join(f"{al_curve[value]:.2f}" for value in al_curve) + " |",
        "",
        "## Spec-aware hardware concurrency lift across all 448 designs",
        "",
        "| Target | Upgrade | GB-feasible | Newly feasible | Median lift | P10–P90 | Max lift | Clears target at physical max |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target in TARGETS:
        subset = [row for row in comparison if row["target_tps_user"] == target]
        for prefix, label in (("vr200_hbm", "22-TB/s HBM only"), ("vr200_full", "HBM + 2× fabric")):
            paired = [row for row in subset if row["gb300_spec_max_cpr"] > 0]
            lifts = [row[f"{prefix}_spec_hardware_lift_vs_gb300"] for row in paired]
            newly = sum(row[f"{prefix}_spec_newly_feasible_vs_gb300"] for row in subset)
            at_full = sum(row[f"{prefix}_clears_at_physical_max"] for row in subset)
            lines.append(
                f"| {target} | {label} | {len(paired)}/{len(subset)} | {newly} | {fmt(statistics.median(lifts), 2)}x | "
                f"{fmt(percentile(lifts, .1), 2)}–{fmt(percentile(lifts, .9), 2)}x | {fmt(max(lifts), 2)}x | {at_full}/{len(subset)} |"
            )

    lines += [
        "",
        "## How much speculation itself raises the SLO concurrency boundary",
        "",
        "| Target | Hardware | No-spec-feasible | Newly feasible with spec | Median spec/no-spec c/r | P10–P90 | Boundary falls back to no-spec |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target in TARGETS:
        subset = [row for row in comparison if row["target_tps_user"] == target]
        for prefix, label in (("gb300", "GB300"), ("vr200_hbm", "VR200 HBM"), ("vr200_full", "VR200 HBM+fabric")):
            paired = [row for row in subset if row[f"{prefix}_no_spec_max_cpr"] > 0]
            lifts = [row[f"{prefix}_spec_vs_no_spec_lift"] for row in paired]
            newly = sum(row[f"{prefix}_newly_feasible_from_spec"] for row in subset)
            fallback = sum(row[f"{prefix}_edge_nextn"] == 0 for row in subset if row[f"{prefix}_spec_max_cpr"] > 0)
            lines.append(
                f"| {target} | {label} | {len(paired)}/{len(subset)} | {newly} | {fmt(statistics.median(lifts), 2)}x | "
                f"{fmt(percentile(lifts, .1), 2)}–{fmt(percentile(lifts, .9), 2)}x | {fallback}/{len(subset)} |"
            )

    lines += [
        "",
        "## Recommended controlled designs: no-spec → best-spec c/r",
        "",
        "| Design | Target | GB300 | HBM | HBM+fabric | Spec-aware HBM lift vs GB | Spec-aware full lift vs GB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rec_labels = {
        "aggressive": "Aggressive latency floor", "balanced": "Balanced recommendation",
        "conservative": "Conservative recommendation", "hybrid": "KDA+sparse hybrid",
        "latent": "Latent-MoE option", "fallback": "Full-attention fallback",
    }
    index = {(row["architecture_id"], row["target_tps_user"]): row for row in comparison}
    for rec_id, architecture_id in recommendations.items():
        for target in TARGETS:
            row = index[(architecture_id, target)]
            lines.append(
                f"| {rec_labels.get(rec_id, rec_id)} | {target} | {row['gb300_no_spec_max_cpr']}→{row['gb300_spec_max_cpr']} | "
                f"{row['vr200_hbm_no_spec_max_cpr']}→{row['vr200_hbm_spec_max_cpr']} | "
                f"{row['vr200_full_no_spec_max_cpr']}→{row['vr200_full_spec_max_cpr']} | "
                f"{fmt(row['vr200_hbm_spec_hardware_lift_vs_gb300'], 2)}x | {fmt(row['vr200_full_spec_hardware_lift_vs_gb300'], 2)}x |"
            )

    lines += [
        "",
        "## Native-family anchors: best-spec c/r",
        "",
        "| Anchor | Target | GB c/r | HBM c/r | HBM lift | HBM+fabric c/r | Full lift |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    native_groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in native_rows:
        native_groups[(row["anchor_id"], row["target_tps_user"])][row["system"]] = row
    for (_, target), group in sorted(native_groups.items()):
        gb = group["GB300"]["max_cpr"]
        hbm = group["VR200 HBM"]["max_cpr"]
        full = group["VR200 HBM+NVLink"]["max_cpr"]
        lines.append(
            f"| {group['GB300']['label']} | {target} | {gb} | {hbm} | {fmt(hbm / gb if gb else None, 2)}x | "
            f"{full} | {fmt(full / gb if gb else None, 2)}x |"
        )

    lines += [
        "",
        "## Interpretation limits",
        "",
        "- This is still SOL: no scheduler, framework overhead, queueing, prefill, or imbalance penalty.",
        "- The AL curve is measured Kimi-K3 SPEED-Bench coding behavior and is deliberately held fixed. It is not evidence that a synthetic 5T architecture, DeepSeek-style body, or GLM body would achieve the same acceptance.",
        "- Kimi-like bodies use AIC's K3/DSpark verifier; direct bodies use its generic MTP verifier. Cross-body speculative comparisons are sensitivity bounds.",
        "- The adaptive search assumes the best-depth TPS envelope does not recover above a failed c/r. Every selected edge and edge+1 are directly checked, but this is not a dense evaluation of every speculative c/r.",
        "- See `slo_spec_concurrency_comparison.csv` for every architecture/target and `SLO_CONCURRENCY_REPORT.md` for the dense no-spec control.",
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
    results: list[dict[str, Any]] = []
    for body_index, body in enumerate(specs, 1):
        for system in study.SYSTEMS:
            prefix = {"GB300": "gb300", "VR200 HBM": "vr200_hbm", "VR200 HBM+NVLink": "vr200_full"}[system]
            physical = {
                profile_id: int(no_spec[f"{body.id}__{profile_id}"][f"{prefix}_physical_max_cpr"])
                for profile_id in study.ATTENTION_PROFILES
            }
            results.extend(adaptive_boundaries(evaluator, body, system, physical))
            print(
                f"controlled {body_index}/{len(specs)} {body.id} {system}; "
                f"new AIC estimates={evaluator.new_estimates}", flush=True,
            )

    native_no_spec = {row["anchor_id"]: row for row in read_csv(OUTPUT / "slo_native_anchor_comparison.csv")}
    native_results = []
    for anchor_id, anchor in anchors.items():
        for system in study.SYSTEMS:
            prefix = {"GB300": "gb300", "VR200 HBM": "vr200_hbm", "VR200 HBM+NVLink": "vr200_full"}[system]
            physical = int(native_no_spec[anchor_id][f"{prefix}_physical_max_cpr"])
            for target in TARGETS:
                native_results.append(adaptive_native_boundary(evaluator, anchor, system, target, physical))

    comparison = pivot(results, no_spec)
    metadata = json.loads((OUTPUT / "metadata.json").read_text(encoding="utf-8"))
    write_csv(OUTPUT / "slo_spec_concurrency_by_system.csv", results)
    write_csv(OUTPUT / "slo_spec_concurrency_comparison.csv", comparison)
    write_csv(OUTPUT / "slo_spec_native_anchor_by_system.csv", native_results)
    (OUTPUT / "SLO_SPEC_CONCURRENCY_REPORT.md").write_text(
        make_report(comparison, native_results, metadata["recommendations"], al_curve), encoding="utf-8"
    )
    (OUTPUT / "slo_spec_concurrency_metadata.json").write_text(json.dumps({
        "context": CONTEXT, "dep": DEP, "targets_tps_user": TARGETS,
        "controlled_architectures": len(no_spec), "native_anchors": len(anchors),
        "aic_estimates_added": evaluator.new_estimates, "acceptance_curve": al_curve,
        "search": "adaptive integer binary search with selected edge+1 validation",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Added AIC estimates: {evaluator.new_estimates}")
    print(f"Report: {OUTPUT / 'SLO_SPEC_CONCURRENCY_REPORT.md'}")


if __name__ == "__main__":
    main()
