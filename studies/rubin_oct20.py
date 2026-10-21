"""Re-evaluate the frozen and October-spec Rubin profiles; preserve releases.

Run: .venv-current/bin/python studies/rubin_oct20.py
Reuses the existing full finite-layout/integer-batch sensitivity evaluator.
"""
import copy
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'studies/gb300_sensitivity'))
import curvature as search
from run import REF, doctor, save
import yaml

OUT = ROOT / 'results/rubin-spec-update-2026-10-20'
CASES = {'frozen_baseline': {}, 'october_specs': {
    'hbm_bandwidth': 19.2 / 22, 'interconnect_bandwidth': 1.5 / 1.8}}


def execute(job):
    search.OUT = OUT
    search.REFERENCE = 'VR200'  # Search/anchor hardware, not GB300 denominator.
    search.previous_case = lambda changes: None
    return search.execute(job)


def main():
    files = [Path(__file__), Path(search.__file__),
             ROOT / 'studies/gb300_sensitivity/run.py',
             ROOT / 'studies/log400_release.py',
             *sorted((ROOT / 'src/agent_index').glob('*.py')),
             REF / 'manifest.json', REF / 'curves_and_inputs.json',
             REF / 'prefill_winners.json', REF / 'catalog.json',
             REF / 'hardware/vr200_deployment.yaml']
    signature = dict(runtime=doctor(REF), files={str(p.relative_to(ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    base = yaml.safe_load((REF / 'hardware/vr200_deployment.yaml').read_text())
    provenance = dict(signature=signature, changes=CASES,
        source_urls=['https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/',
                     'https://www.nvidia.com/en-us/data-center/nvlink/'],
        source_checked='2026-10-20', baseline_profile=base,
        notes=['Only HBM bandwidth and NVLink outbound bandwidth change.',
               'Literal revised values: 19.2e12 and 1.5e12 bytes/s.',
               '72-chip domain/budget, capacity, precision, nextn3/AL3 and memory admission unchanged.',
               'Full saved finite-layout family and every admitted integer batch re-searched in both phases.',
               'Four models; 256Ki context; 1Ki new cached-prefix prefill; log 50-400.',
               'Frozen GB300 denominator and input/output ratio retained.',
               'No timing reuse: frozen Rubin baseline is also fully rerun to validate parity.',
               'Frozen release files and production profiles remain unchanged.'])
    if (OUT / 'provenance.json').exists():
        assert json.loads((OUT / 'provenance.json').read_text()) == provenance
    inputs = OUT / 'inputs'
    (inputs / 'hardware').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REF / 'models.json', inputs / 'models.json')
    shutil.copytree(REF / 'models', inputs / 'models', dirs_exist_ok=True)
    for case in CASES:
        profile = copy.deepcopy(base)
        if case == 'october_specs':
            profile['gpu']['mem_bw'] = 19_200_000_000_000
            for key in ('intra_node_bw', 'inter_node_bw', 'inter_rack_bw'):
                profile['node'][key] = 1_500_000_000_000
        (inputs / 'hardware' / ('curvature_' + case + '.yaml')).write_text(yaml.safe_dump(profile))
    save(OUT / 'provenance.json', provenance)
    jobs = [(case, changes, signature) for case, changes in CASES.items()]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(execute, jobs):
            print('COMPLETE', json.dumps(result), flush=True)
    frozen = json.loads((REF / 'curves_and_inputs.json').read_text())
    metrics = ('combined_GB300e', 'decode_GB300e', 'prefill_GB300e')
    rows, parity, counts = [], [], {}
    for model in frozen['models']:
        before = json.loads((OUT / 'frozen_baseline' / (model + '.json')).read_text())
        after = json.loads((OUT / 'october_specs' / (model + '.json')).read_text())
        original_prefill = next(p for p in frozen['prefill_rates']
            if p['hardware'] == 'VR200' and p['model'] == model)['tokens_chip_second']
        expected = search.metrics(frozen['curves']['VR200'][model],
            {'input_tokens_chip_second': original_prefill}, frozen, model)
        errors = {k: abs(before['metrics'][k] / expected[k] - 1) for k in metrics}
        assert max(errors.values()) < 1e-9, (model, errors)
        assert before['candidates'] == after['candidates'], model
        parity.append(dict(model=model, relative_errors=errors))
        counts[model] = after['candidates']
        row = dict(model=model)
        for k in metrics:
            b, a = before['metrics'][k], after['metrics'][k]
            row.update({k+'_before': b, k+'_after': a, k+'_change_pct': 100*(a/b-1)})
        rows.append(row)
    overall = dict(model='Four-model average')
    for k in metrics:
        b = sum(r[k+'_before'] for r in rows)/4
        a = sum(r[k+'_after'] for r in rows)/4
        overall.update({k+'_before': b, k+'_after': a, k+'_change_pct': 100*(a/b-1)})
    with (OUT / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(overall))
        writer.writeheader(); writer.writerows([overall, *rows])
    save(OUT / 'validation.json', dict(status='complete', baseline_parity=parity,
        revised_candidate_counts=counts, profiles_differ_only_in_bandwidth=True,
        overall=overall))
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == '__main__':
    main()
