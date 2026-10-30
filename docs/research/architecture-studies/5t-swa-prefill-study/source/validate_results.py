#!/usr/bin/env python3
"""Integrity checks for the cached-prefix prefill study outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
EXPECTED_PROFILES = 43
EXPECTED_SYSTEMS = 2
EXPECTED_CPRS = 1_024


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    all_rows = rows(OUTPUT / "all_results.csv")
    summaries = rows(OUTPUT / "architecture_summary.csv")
    metadata = json.loads((OUTPUT / "metadata.json").read_text(encoding="utf-8"))
    expected = EXPECTED_PROFILES * EXPECTED_SYSTEMS * EXPECTED_CPRS
    assert len(all_rows) == expected, (len(all_rows), expected)
    assert len(summaries) == EXPECTED_PROFILES
    assert metadata["workload"] == {
        "prefix_tokens_already_cached": 262_144,
        "new_input_tokens": 1_024,
        "postfill_context_tokens": 263_168,
        "aic_mode": "static_ctx",
    }
    assert all(float(row["source_non_sol_ms"]) == 0 for row in all_rows)
    assert all(float(row["extend_latency_ms"]) > 0 for row in all_rows)

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        groups[(row["architecture_id"], row["system"])].append(row)
        assert int(row["prefix_tokens"]) == 262_144
        assert int(row["new_tokens"]) == 1_024
        assert int(row["total_context"]) == 263_168
        assert int(row["global_concurrency"]) == 64 * int(row["cpr"])

    assert len(groups) == EXPECTED_PROFILES * EXPECTED_SYSTEMS
    for group in groups.values():
        group.sort(key=lambda row: int(row["cpr"]))
        assert [int(row["cpr"]) for row in group] == list(range(1, EXPECTED_CPRS + 1))
        latency = [float(row["extend_latency_ms"]) for row in group]
        memory = [float(row["memory_gib"]) for row in group]
        assert all(b >= a for a, b in zip(latency, latency[1:]))
        assert all(b >= a for a, b in zip(memory, memory[1:]))

    for summary in summaries:
        cap = int(summary["gb_physical_max_cpr"])
        assert 0 < cap < EXPECTED_CPRS, f"capacity boundary truncated: {summary['architecture_id']}={cap}"
        gb = groups[(summary["architecture_id"], "GB300")]
        assert gb[cap - 1]["oom"] == "False"
        assert gb[cap]["oom"] == "True"
        assert int(summary["rubin_physical_max_cpr"]) == cap

    html = (OUTPUT / "PREFIX_CACHE_PREFILL_STUDY.html").read_text(encoding="utf-8")
    assert "__SUMMARY__" not in html and "__CURVES__" not in html
    assert "256K resident prefix" in html and "All 43 profiles" in html
    print(
        json.dumps(
            {
                "status": "PASS",
                "profiles": len(summaries),
                "composed_points": len(all_rows),
                "direct_aic_references": sum(1 for _ in (OUTPUT / "prefill_estimate_cache.jsonl").open()),
                "non_sol_latency_ms": 0,
                "cpr_range": [1, EXPECTED_CPRS],
            }
        )
    )


if __name__ == "__main__":
    main()
