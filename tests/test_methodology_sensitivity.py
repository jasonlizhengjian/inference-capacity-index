"""Focused checks for the new analytical integrations, not simulator benchmarks."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"studies"))
from methodology_sensitivity import area, combined_capacity, evaluate, scenarios


class SensitivityTests(unittest.TestCase):
    def test_no_prefill_closed_forms(self):
        p = [{"speed": 150., "capacity": 4.}]
        self.assertAlmostEqual(area(p, 0, 0, 50, 200, "linear"), 400)
        self.assertAlmostEqual(area(p, 0, 0, 50, 200, "log"), 4*math.log(3))

    def test_prefill_integrals_against_quadrature(self):
        p = [{"speed": 150., "capacity": 4.}, {"speed": 300., "capacity": 2.}]
        for axis in ("linear", "log"):
            numeric = 0.
            for a, b in [(50, 150), (150, 300), (300, 400)]:
                n = 4096
                width = (b-a)/n
                numeric += width*math.fsum(
                    combined_capacity(p, x := a+(i+.5)*width, 500., 1.5)/(x if axis == "log" else 1)
                    for i in range(n))
            self.assertAlmostEqual(area(p, 500, 1.5, 50, 400, axis)/numeric, 1, places=7)

    def test_edges_and_missing_capacity(self):
        p = [{"speed": 100., "capacity": 3.}]
        self.assertEqual(combined_capacity(p, 100, 1000, 0), 3)
        self.assertEqual(combined_capacity(p, 100.00001, 1000, 0), 0)
        self.assertEqual(combined_capacity(p, 50, 0, 1), 0)
        self.assertEqual(area(p, 0, 1, 50, 200, "linear"), 0)

    def test_zero_reference_is_not_reweighted(self):
        saved = {"curves": {"GB300 NVL72": {"m":[]}, "other":{"m":[{"speed":100,"capacity":1}]}},
                 "prefill_rates":[{"hardware":h, "model":"m", "tokens_chip_second":1} for h in ("GB300 NVL72","other")]}
        # Use a real model label so the diagnostic's model-name mapping is exercised.
        for v in saved["curves"].values():
            v["GLM-5.2"] = v.pop("m")
        for v in saved["prefill_rates"]:
            v["model"] = "GLM-5.2"
        case = dict(id="log_50_400", family="axis_band", weights={"GLM-5.2":1},
                    axis="log", low=50, high=400, target=None, ratio=0)
        rows, _ = evaluate(saved, [case])
        self.assertTrue(all(r["score_GB300e"] is None for r in rows))

    def test_scenario_controls(self):
        models = ["GLM-5.2", "Kimi-K3", "Qwen-480B", "MAI2-informed-h128-w512-sh4352"]
        cases = scenarios(models, 1.5)
        self.assertEqual(len(cases), 28)
        self.assertEqual(len({c["id"] for c in cases}), len(cases))
        for c in cases:
            self.assertAlmostEqual(sum(c["weights"].values()), 1)
        base = next(c for c in cases if c["id"] == "log_50_400")
        control = next(c for c in cases if c["id"] == "proxy_25")
        self.assertEqual(base["weights"], control["weights"])


if __name__ == "__main__":
    unittest.main()
