"""Small offline regression check for the dated Rubin-only overlay."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'studies'))
import rubin_release


class RubinRevisionTest(unittest.TestCase):
    def test_rebuild_and_visual_winners(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            rubin_release.main(out)
            with (out / 'index.csv').open() as stream:
                rows = {r['hardware']: r for r in csv.DictReader(stream)}
            self.assertAlmostEqual(float(rows['VR200']['agent_index_GB300e']), 2.177366570368923)
            self.assertEqual(float(rows['GB300 NVL72']['agent_index_GB300e']), 1.)
            visual = json.loads((out / 'visual_inputs.json').read_text())
            winners = [r for r in visual['prefill_winners'] if r['hardware'] == 'VR200']
            self.assertEqual(len(winners), 4)
            for row in winners:
                self.assertAlmostEqual(row['input_tokens_per_chip_second'],
                    row['requests'] * 1024 * 1000 / row['chips'] / row['forward_ms'])
            self.assertFalse(json.loads((out / 'provenance.json').read_text())['simulation_rerun'])

    def test_frozen_output_rejected(self):
        for path in (rubin_release.BASE, rubin_release.REV, rubin_release.REV / 'nested'):
            with self.assertRaises(ValueError):
                rubin_release.main(path)


if __name__ == '__main__':
    unittest.main()
