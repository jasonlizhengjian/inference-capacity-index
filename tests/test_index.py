import math
import tempfile
import unittest
from pathlib import Path

from agent_index.index import area, capacity, envelope, build, root_path, validate_reference, verify_inputs
from agent_index.publication import publish


class IndexTests(unittest.TestCase):
    def test_exact_steps(self):
        p=[dict(speed=100.,capacity=4.),dict(speed=200.,capacity=2.)]
        self.assertEqual(capacity(p,100),4)
        self.assertEqual(capacity(p,math.nextafter(100,math.inf)),2)
        self.assertEqual(capacity(p,201),0)
        self.assertAlmostEqual(area(p),6*math.log(2))
        self.assertEqual(area(p,logarithmic=False),400)
        self.assertEqual(area([]),0)

    def test_envelope(self):
        p=[dict(speed=100.,capacity=4.),dict(speed=200.,capacity=2.),dict(speed=99.,capacity=1.)]
        self.assertEqual(len(envelope(p+p)),2)
        self.assertEqual(area(p),area(envelope(p)))

    def test_bad_band(self):
        with self.assertRaises(ValueError):area([],0,1)

    def test_frozen_release(self):
        root=root_path()
        self.assertGreater(verify_inputs(root),20)
        result=validate_reference(root,build(root))
        self.assertEqual(result['score_checks'],51)

    def test_publication(self):
        root=root_path();data=build(root)
        with tempfile.TemporaryDirectory() as folder:
            output=publish(root,data,Path(folder),validate_reference(root,data))
            self.assertTrue((output/'REPORT.html').is_file())
            self.assertNotIn('/home/',(output/'agent_index.csv').read_text())
            # Export columns, floats, ordering, encoding and newline policy match exactly.
            self.assertEqual((output/'agent_index.csv').read_bytes(),(root/'reference/2026-10-09/expected_scores.csv').read_bytes())


if __name__=='__main__':unittest.main()
