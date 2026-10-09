"""Exact capacity-envelope scoring, independent of AISim and plotting libraries.

Ported from the author's cross-hardware study agent_index_data.py. Saved rows
are observations, not samples to interpolate. No hardware timing occurs here.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

MODELS = ("GLM-5.2", "Kimi-K3")
REFERENCE = "GB300 NVL72"
RELEASE = "2026-10-09"
SCENARIOS = {
    "log400": (50, 400, True), "linear400": (50, 400, False),
    "log300": (50, 300, True), "linear200": (50, 200, False),
}


def root_path():
    return Path(__file__).resolve().parents[2]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_catalog(root):
    return json.loads((root / "configs/catalog.json").read_text())


def verify_inputs(root):
    manifest = json.loads((root / f"reference/{RELEASE}/manifest.json").read_text())
    mismatches = []
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Manifest path escapes repository: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            mismatches.append(name)
    if mismatches:
        raise ValueError(f"Frozen input integrity failed: {mismatches}")
    return len(manifest["files"])


def saved_candidates(root):
    folder = root / "reference" / RELEASE
    catalog = load_catalog(root)
    points = []
    for filename, china in (("main_frontiers.csv", False), ("china_frontiers.csv", True)):
        for r in read_csv(folder / filename):
            if r["model"] not in MODELS:
                continue
            if china and (r["case"] != "base" or int(float(r["chip_budget"])) != 128):
                continue
            hardware = r["hardware"] if china else r["Hardware"]
            def integer(key):
                value = float(r[key])
                if not value.is_integer():
                    raise ValueError(f"Noninteger {key}: {r[key]}")
                return int(value)
            if r["status"] != "ok" or r["fits"].lower() != "true":
                raise ValueError("Non-admitted reference row")
            if (integer("context"), integer("nextn"), integer("al")) != (262144, 3, 3):
                raise ValueError("Reference workload differs from the frozen release")
            world, tp, dp, pp = (integer(k) for k in ("world", "tp", "dp", "pp"))
            requests = integer("total_requests")
            density = float(r["concurrency_per_rank"])
            speed = float(r["tps_user"])
            if world != tp * dp * pp or world > catalog[hardware]["chip_budget"]:
                raise ValueError("Inconsistent chip accounting")
            if not math.isclose(density, requests / world, rel_tol=1e-12):
                raise ValueError("Inconsistent request density")
            if not math.isfinite(speed) or speed <= 0 or density <= 0:
                raise ValueError("Invalid speed or density")
            points.append({
                "hardware": hardware, "model": r["model"], "speed": speed,
                "capacity": density, "chips": world, "requests": requests,
                "tp": tp, "dp": dp, "pp": pp, "dcp": integer("dcp"),
                "etp": integer("etp"), "ep": integer("ep"),
                "batch": integer("batch_per_replica"), "memory": float(r["memory_gib"]),
                "raw_ms": float(r["raw_ms"]), "layout": r["parallelism"],
                "source": filename,
            })
    return points


def envelope(points):
    kept, best = [], -1.0
    for point in sorted(points, key=lambda p: (-p["speed"], -p["capacity"])):
        if point["capacity"] > best:
            kept.append(point)
            best = point["capacity"]
    return sorted(kept, key=lambda p: p["speed"])


def capacity(points, speed):
    return max((p["capacity"] for p in points if p["speed"] >= speed), default=0.0)


def area(points, low=50, high=400, logarithmic=True):
    if not 0 < low < high:
        raise ValueError("Require 0 < low < high")
    edges = sorted({low, high, *(p["speed"] for p in points if low < p["speed"] < high)})
    return math.fsum(capacity(points, (a + b) / 2) *
                     (math.log(b / a) if logarithmic else b - a)
                     for a, b in zip(edges, edges[1:]))


def build(root, candidates=None):
    catalog = load_catalog(root)
    curves = {h: {m: [] for m in MODELS} for h in catalog}
    for point in saved_candidates(root) if candidates is None else candidates:
        curves[point["hardware"]][point["model"]].append(point)
    curves = {h: {m: envelope(p) for m, p in models.items()} for h, models in curves.items()}
    results = {}
    for hardware, models in curves.items():
        item = {"models": {}, "combined": {}}
        for model, points in models.items():
            item["models"][model] = {}
            for scenario, (low, high, log) in SCENARIOS.items():
                own = area(points, low, high, log)
                ref = area(curves[REFERENCE][model], low, high, log)
                if ref <= 0:
                    raise ValueError(f"No reference area for {model}")
                item["models"][model][scenario] = {"area": own, "reference_area": ref, "score": own / ref}
        for scenario in SCENARIOS:
            item["combined"][scenario] = math.fsum(
                item["models"][m][scenario]["score"] for m in MODELS) / len(MODELS)
        results[hardware] = item
    return {"release": RELEASE, "band": [50, 400], "weighting": "logarithmic",
            "models": list(MODELS), "reference": REFERENCE, "curves": curves, "results": results}


def validate_reference(root, data):
    catalog = load_catalog(root)
    names = {v["name"]: k for k, v in catalog.items()}
    expected = read_csv(root / f"reference/{RELEASE}/expected_scores.csv")
    if len(expected) != len(data["results"]) or len(expected) != 17:
        raise ValueError("Expected 17 accelerators")
    for row in expected:
        r = data["results"][names[row["accelerator"]]]
        pairs = [(r["combined"]["log400"], row["agent_index_GB300e"])]
        pairs.extend((r["models"][m]["log400"]["score"], row[c])
                     for m, c in zip(MODELS, ("GLM_5_2_GB300e", "Kimi_K3_GB300e")))
        if not all(math.isclose(a, float(b), rel_tol=1e-12, abs_tol=1e-14) for a, b in pairs):
            raise ValueError(f"Reference score mismatch: {row['accelerator']}")
    for row in read_csv(root / f"reference/{RELEASE}/main_ensemble.csv"):
        for model, column in zip(MODELS, ("GLM", "Kimi")):
            actual = data["results"][row["Hardware"]]["models"][model]["log300"]["score"]
            if not math.isclose(actual, float(row[column]), rel_tol=1e-11):
                raise ValueError("Historical log50–300 score mismatch")
    return {"hardware_count": 17, "model_count": 2, "score_checks": len(expected) * 3,
            "reference_parity": True, "simulation_rerun": False}
