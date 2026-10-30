#!/usr/bin/env python3
"""Consistency checks for the unified prefill evidence and HTML artifact."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def rows(name: str):
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    all_rows = rows("all_results.csv")
    summary = rows("architecture_summary.csv")
    dep_rows = rows("dep_sweep_results.csv")
    dep_summary = rows("dep_sweep_summary.csv")
    metadata = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
    report = (OUT / "UNIFIED_PREFIX_CACHE_PREFILL_STUDY.html").read_text(encoding="utf-8")

    assert len(summary) == 506
    assert len(all_rows) == 506 * 2 * 13 * 7 == 92_092
    assert len({r["architecture_id"] for r in summary}) == 506
    counts = Counter(r["architecture_id"] for r in all_rows)
    assert set(counts.values()) == {182}
    assert metadata["counts"]["total_architectures"] == 506
    assert metadata["counts"]["result_rows"] == 92_092
    assert all(float(r["non_sol_ms"]) == 0 for r in all_rows)
    assert all(r["gb_cpr_1k_at_95pct"] == "1" for r in summary)
    assert all(r["rubin_cpr_1k_at_95pct"] == "1" for r in summary)
    assert all(float(r["rubin_1k_c1_latency_ms"]) <= float(r["gb_1k_c1_latency_ms"]) for r in summary)
    assert len(dep_rows) == 506 * 2 * 4 * 4 == 16_192
    assert len(dep_summary) == 506
    assert set(Counter(r["architecture_id"] for r in dep_rows).values()) == {32}
    assert Counter(r["gb_best_dep_cpr1"] for r in dep_summary) == {"8": 3, "16": 499, "32": 2, "0": 2}
    assert Counter(r["gb_best_dep_global64"] for r in dep_summary) == {"64": 504, "0": 2}
    assert all(float(r["non_sol_ms"]) == 0 for r in dep_rows)

    dsa = {}
    for row in all_rows:
        aid = row["architecture_id"]
        if (
            aid.startswith("factorial__latent_L61_aggressive_m3072_s1__dsa")
            and row["system"] == "GB300" and row["new_tokens_per_request"] == "1024"
            and row["cpr"] == "1" and "noreuse" not in aid
        ):
            dsa[aid.rsplit("__", 1)[1]] = float(row["attention_ms"])
    assert dsa["dsa512"] < dsa["dsa1024"] < dsa["dsa2048"] < dsa["dsa4096"]

    assert report.count('data-family="') == 521  # 15 direct table + 506 full table
    for text in ("506 configurations", "92,092", "Kimi K3 scaling", "DeepSeek scaling", "SWA and hybrid extensions"):
        assert text in report
    for target in ("all_results.csv", "architecture_summary.csv", "dep_sweep_results.csv", "dep_sweep_summary.csv", "metadata.json"):
        assert (OUT / target).exists() and f'href="{target}"' in report
    print("PASS: 506 architectures, 92,092 main + 16,192 DEP SOL rows, DSA ordering, report links and tables")


if __name__ == "__main__":
    main()
