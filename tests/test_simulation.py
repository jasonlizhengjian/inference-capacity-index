"""Pure policy tests; optional pinned-runtime tests use saved numerical fixtures.

Set AGENT_INDEX_TEST_AISIM=1 to execute five small analytical evaluations.
These are not GPU or serving-engine benchmarks.
"""
import json
import math
import os
from pathlib import Path
import unittest

from agent_index.simulation import enumerate_layouts, evaluate_candidate, make_config
from agent_index.topology import events, phase_hops

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "configs/catalog.json").read_text())


class PolicyTests(unittest.TestCase):
    catalog = CATALOG

    def test_integral_experts_eliminate_72(self):
        for model in ("GLM-5.2", "Kimi-K3"):
            layouts = list(enumerate_layouts(self.catalog["GB300 NVL72"], model))
            self.assertTrue(layouts)
            self.assertTrue(all(r["tp"] * r["dp"] != 72 for r in layouts))
            self.assertTrue(all(r["pp"] == 1 for r in layouts))

    def test_local_policy(self):
        layouts = list(enumerate_layouts(self.catalog["B300 · 800G"], "Kimi-K3"))
        self.assertEqual({r["pp"] for r in layouts}, {1, 2, 3, 4, 6, 8})
        for row in layouts:
            self.assertLessEqual(row["tp"] * row["dp"], 8)
            self.assertLessEqual(row["tp"] * row["dp"] * row["pp"], 64)
            if row["pp"] > 1:
                self.assertEqual(row["tp"] * row["dp"], 8)

    def test_reject_invalid_layout(self):
        hw = self.catalog["B300 · 800G"]
        with self.assertRaises(ValueError):
            make_config(hw, "GLM-5.2", 1, 16, 1, 1, 16, 1)
        with self.assertRaises(ValueError):
            make_config(hw, "GLM-5.2", 2, 4, 4, 1, 8, 1)

    def test_overlay(self):
        self.assertEqual(phase_hops(4, "all_gather"), 2)
        self.assertEqual(phase_hops(8, "all_gather"), 5)
        self.assertEqual(phase_hops(8, "all_reduce"), 10)
        with self.assertRaises(ValueError):
            phase_hops(72, "all_reduce")
        with self.assertRaises(ValueError):
            events({"UnknownComm": {"name": "comm"}})


@unittest.skipUnless(os.environ.get("AGENT_INDEX_TEST_AISIM") == "1",
                     "Enable AGENT_INDEX_TEST_AISIM=1 with the pinned wheel")
class NativeFixtureTests(unittest.TestCase):
    catalog = CATALOG

    def test_saved_candidate_parity(self):
        # Extracted from frozen complete_results/frontier_nodes.csv, October 9.
        # hw, model, TP, DP, DCP, ETP, EP, PP, batch, raw ms, memory GiB.
        fixtures = [
            ("GB300 NVL72", "GLM-5.2", 1, 32, 1, 1, 32, 1, 6,
             15.621996935076922, 143.31644384004176),
            ("GB300 NVL72", "Kimi-K3", 4, 16, 4, 1, 64, 1, 54,
             57.855627072, 148.29074470885098),
            ("B300 · 800G", "Kimi-K3", 4, 2, 4, 1, 8, 2, 68,
             88.93624794196415, 231.71462778933343),
            ("TPU8i · corrected · hops", "GLM-5.2", 1, 32, 1, 1, 32, 1, 10,
             20.923931776426716, 203.95970315299928),
            ("H100 SXM · FP8 · 400G", "GLM-5.2", 8, 1, 8, 1, 8, 2, 3,
             20.86028090452354, 63.24726387357805),
        ]
        for hw, model, tp, dp, dcp, etp, ep, pp, batch, ms, memory in fixtures:
            with self.subTest(hardware=hw, model=model):
                result = evaluate_candidate(self.catalog[hw], model, tp, dp, dcp,
                                            etp, ep, pp, batch)
                self.assertTrue(result["fits"])
                self.assertTrue(math.isclose(result["raw_ms"], ms, rel_tol=1e-12, abs_tol=1e-9))
                self.assertTrue(math.isclose(result["memory_gib"], memory, rel_tol=1e-12, abs_tol=1e-9))
                self.assertAlmostEqual(result["tps_user"], 3000 / ms)
                self.assertEqual(result["world"], tp * dp * pp)
                self.assertEqual(result["total_requests"], batch * dp)


if __name__ == "__main__":
    unittest.main()
