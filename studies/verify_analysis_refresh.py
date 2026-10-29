"""Check the hardware-only supporting-analysis refresh against its dated archive."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SENS = ROOT/'docs/research/methodology-sensitivity'
VALID = ROOT/'docs/research/inferencex-validation'
ARCHIVE = ROOT/'docs/research/archives/2026-10-19-pre-rubin-refresh'


def main():
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(ARCHIVE/'supporting-analyses.zip') as z:
        def old(path): return z.read(str(path.relative_to(ROOT)))
        manifest = json.loads((ARCHIVE/'manifest.json').read_text())
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected, name
        before = list(csv.DictReader(io.StringIO(old(SENS/'scenario_results.csv').decode())))
        after = list(csv.DictReader((SENS/'scenario_results.csv').open()))
        prior = {(r['hardware'], r['scenario']): r for r in before}
        unchanged = 0
        for r in after:
            if r['hardware'] != 'VR200':
                assert r['score_GB300e'] == prior[r['hardware'],r['scenario']]['score_GB300e']
                unchanged += 1
        assert len(after) == 644 and unchanged == 616
        h200 = {r['scenario']:float(r['score_GB300e']) for r in after if r['hardware'].startswith('H200')}
        baseline = {r['hardware']:float(r['score_GB300e']) for r in after if r['scenario']=='log_50_400'}
        assert baseline['GB300 NVL72'] == 1
        assert math.isclose(baseline['VR200'],2.177366570368923,rel_tol=1e-12)
        assert old(SENS/'scenarios.json') == (SENS/'scenarios.json').read_bytes()
        old_layouts = json.loads(old(VALID/'layout_restricted.json'))
        layouts = json.loads((VALID/'layout_restricted.json').read_text())
        non_rubin = lambda rows: [r for r in rows if r['hardware']!='VR200']
        assert non_rubin(old_layouts['recipes']) == non_rubin(layouts['recipes'])
        assert old_layouts['excluded'] == layouts['excluded']
        retained = ['observed_points.csv','matched_glm.json','matched_glm_strata.csv',
                    'kimi_config_audit.json','kimi_shape_scenarios.csv']
        for name in retained:
            assert old(VALID/name) == (VALID/name).read_bytes(), name
        for name, key in [('curves.json','groups'),('three_curve_comparison.json','groups'),
                          ('point_agreement.json','summaries')]:
            a=json.loads(old(VALID/name));b=json.loads((VALID/name).read_text())
            assert non_rubin(a[key]) == non_rubin(b[key]), name
        agreement=json.loads((VALID/'point_agreement.json').read_text())
        case=next(r for r in agreement['summaries'] if r['model']=='Kimi-K3' and
            r['hardware']=='GB300 NVL72' and r['metric']=='output' and r['band']=='index_band' and
            r['decode_efficiency']==r['prefill_efficiency']==.5)['cohorts']['frontier']['restricted']
        assert case['count']==5 and math.isclose(case['mean_absolute_error'],.11555208855816657,rel_tol=1e-12)
        files = [p for p in SENS.iterdir() if p.is_file()] + [p for p in VALID.iterdir()
                 if p.is_file() and p.name!='refresh_verification.json']
        files += [ROOT/'reference/2026-10-15/manifest.json',ROOT/'reference/2026-10-20-rubin/manifest.json',
                  ARCHIVE/'manifest.json', ARCHIVE/'supporting-analyses.zip', Path(__file__),
                  ROOT/'studies/methodology_sensitivity.py',ROOT/'studies/rubin_release.py',
                  *sorted((ROOT/'studies/inferencex_validation').glob('*.py'))]
        result=dict(status='passed',refreshed='2026-10-29',observations='2026-10-19',
            modeled_revision='2026-10-15 + 2026-10-20-rubin',scenarios=28,score_cells=644,
            unchanged_non_rubin_score_cells=unchanged,unchanged_non_rubin_recipes=len(non_rubin(layouts['recipes'])),
            retimed_rubin_recipes=layouts['hardware_refresh']['retimed_recipes'],
            unchanged_observation_and_probe_files=retained,unchanged_non_rubin_comparisons=True,
            VR200_ICI=baseline['VR200'],GB300_H200_headline_ratio=1/h200['log_50_400'],
            GB300_H200_Kimi_ratio=1/h200['only_1'],Kimi_GB300_five_point=case,
            sha256={str(p.relative_to(ROOT)):digest(p) for p in files})
        (VALID/'refresh_verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='sha256'},indent=2))


if __name__=='__main__': main()
