"""Rebuild the October 20 Rubin revision from saved native sweeps (offline).

Only Rubin's four decode frontiers and prefill rates replace the October 15
inputs. The original release and its manifests are never modified.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import log400_release as calculation

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reference/2026-10-15'
REV = ROOT / 'reference/2026-10-20-rubin'


def report_inputs():
    """Verified report inputs for supporting analyses; frozen references stay intact."""
    for name, digest in json.loads((BASE / 'manifest.json').read_text()).items():
        if hashlib.sha256((BASE / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Base reference changed: ' + name)
    out = ROOT / 'results/analysis-refresh-inputs'
    main(out)
    return out


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(out):
    out = Path(out)
    if out.resolve() == (ROOT / 'reference').resolve() or (ROOT / 'reference').resolve() in out.resolve().parents:
        raise ValueError('Output must not overwrite frozen references')
    manifest = json.loads((REV / 'manifest.json').read_text())
    for name, digest in manifest.items():
        if hashlib.sha256((REV / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Rubin reference changed: ' + name)
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((BASE / 'curves_and_inputs.json').read_text())
    visual = json.loads((BASE / 'visual_inputs.json').read_text())
    expected = {}
    winners = []
    counts = {}
    for model in data['models']:
        part = json.loads((REV / 'october_specs' / (model + '.json')).read_text())
        expected[model] = part['metrics']
        counts[model] = part['candidates']
        points = part['frontier']
        for row in points:
            row['hardware'] = 'VR200'
            row['other_memory_gib'] = row['memory'] - row['weights_gib'] - row['cache_gib']
        data['curves']['VR200'][model] = points
        visual['enriched_curves']['VR200'][model] = points
        winner = dict(part['best_prefill'], hardware='VR200')
        winners.append(winner)
        for rate in data['prefill_rates']:
            if rate['hardware'] == 'VR200' and rate['model'] == model:
                rate['tokens_chip_second'] = winner['input_tokens_chip_second']
        label = 'MAI-2 2.8T' if model.startswith('MAI2') else model
        # Replace entire rows: retain no old timing breakdown or winning layout.
        visual['prefill_winners'] = [r for r in visual['prefill_winners']
            if not (r['hardware'] == 'VR200' and r['model'] == label)]
        visual['prefill_winners'].append(dict(winner, model=label,
            input_tokens_per_chip_second=winner['input_tokens_chip_second'],
            forward_ms=winner['raw_ms'],
            layout=' '.join(f'{k.upper()}{winner[k]}' for k in ('tp','dp','dcp','etp','ep','pp'))))
    revision = dict(date='2026-10-20', hardware='VR200',
        hbm_bytes_second=19_200_000_000_000,
        nvlink_outbound_bytes_second=1_500_000_000_000,
        prior_hbm_bytes_second=22_000_000_000_000,
        prior_nvlink_outbound_bytes_second=1_800_000_000_000,
        scope='Rubin only; other 22 frozen profiles unchanged. Not the later live-sheet hardware roster.',
        native_sweep_candidates=counts,
        native_runtime=json.loads((REV / 'provenance.json').read_text())['signature']['runtime'],
        reference_manifest_sha256=hashlib.sha256((REV / 'manifest.json').read_bytes()).hexdigest())
    save(out / 'curves_and_inputs.json', data)
    save(out / 'visual_inputs.json', visual)
    save(out / 'revision.json', revision)
    save(out / 'rubin_prefill_winners.json', winners)
    shutil.copyfile(BASE / 'workload.json', out / 'workload.json')
    # Source directory differs from the score destination: this is reintegration,
    # not a new timing run. Keep that distinction in calculation provenance.
    inputs = out / 'inputs'
    inputs.mkdir(exist_ok=True)
    for name in ('curves_and_inputs.json', 'workload.json'):
        shutil.copyfile(out / name, inputs / name)
    save(inputs / 'provenance.json', revision)
    calculation.main(inputs, out)
    with (out / 'index.csv').open() as stream:
        actual = {r['hardware']: r for r in csv.DictReader(stream)}
    with (BASE / 'expected_index.csv').open() as stream:
        old = {r['hardware']: r for r in csv.DictReader(stream)}
    fields = ('agent_index_GB300e', 'decode_only_GB300e', 'prefill_only_GB300e', *calculation.COMPONENTS)
    for hw in old:
        for key in fields:
            if hw != 'VR200':
                assert math.isclose(float(actual[hw][key]), float(old[hw][key]), rel_tol=1e-12, abs_tol=1e-15), (hw, key)
    for key, metric in (('agent_index_GB300e', 'combined_GB300e'),
                        ('decode_only_GB300e', 'decode_GB300e'),
                        ('prefill_only_GB300e', 'prefill_GB300e')):
        assert math.isclose(float(actual['VR200'][key]), sum(p[metric] for p in expected.values()) / 4, rel_tol=1e-12)
    for model, key in zip(data['models'], calculation.COMPONENTS):
        assert math.isclose(float(actual['VR200'][key]), expected[model]['combined_GB300e'], rel_tol=1e-12)
    assert math.isclose(float(actual['VR200']['agent_index_GB300e']), 2.177366570368923, rel_tol=1e-12)
    save(out / 'reproduction.json', dict(status='passed', unchanged_hardware=22,
        score_cells_checked=23 * len(fields), revised_hardware='VR200',
        native_timings_rerun=False, revision=revision))
    print('Rubin revision verified; other 22 hardware scores unchanged.')
