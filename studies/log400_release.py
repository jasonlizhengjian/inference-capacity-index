"""Build the current log-speed index from frozen phase results; no simulations."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agent_index.index import capacity, area as decode_area

SOURCE = ROOT / 'reference/2026-10-15'
OUT = ROOT / 'results/workload-capacity-oct15-log400'
LOW, HIGH = 50., 400.
REFERENCE = 'GB300 NVL72'
COMPONENTS = ('GLM_GB300e', 'Kimi_GB300e', 'Qwen3_GB300e', 'MAI_proxy_GB300e')


def segments(points):
    edges = sorted({LOW, HIGH, *(p['speed'] for p in points if LOW < p['speed'] < HIGH)})
    return [(a, b, capacity(points, (a+b)/2)) for a, b in zip(edges, edges[1:])]


def combined_area(points, prefill, ratio):
    """Integral of C/(1+r*s*C/P) d ln(s), exact on each decode segment."""
    if ratio == 0:
        return decode_area(points, LOW, HIGH, True)
    if prefill <= 0:
        return 0.
    return math.fsum(c*math.log1p((b-a)/(a*(1+ratio*c*b/prefill)))
                     for a, b, c in segments(points) if c)


def save_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main(source=None, output=None):
    global SOURCE, OUT
    if source is not None: SOURCE = Path(source)
    if output is not None: OUT = Path(output)
    saved = json.loads((SOURCE/'curves_and_inputs.json').read_text())
    curves, models, ratio = saved['curves'], saved['models'], saved['ratio']
    pref = {(p['hardware'], p['model']):p['tokens_chip_second'] for p in saved['prefill_rates']}
    assert len(curves) == 23 and len(models) == 4
    per_model, scenarios, targets = [], [], []
    max_error = 0.
    for label, r in [('decode_only', 0.), ('half_input', ratio/2), ('observed_ratio', ratio), ('double_input', ratio*2)]:
        refs = {m:combined_area(curves[REFERENCE][m], pref[REFERENCE,m], r) for m in models}
        assert all(v > 0 for v in refs.values())
        for h, by_model in curves.items():
            vals = []
            for m in models:
                points, p = by_model[m], pref[h,m]
                a = combined_area(points, p, r)
                vals.append(a/refs[m])
                per_model.append(dict(hardware=h, model=m, scenario=label, input_output_ratio=r,
                                      combined_auc=a, GB300_auc=refs[m], model_GB300e=a/refs[m],
                                      prefill_tokens_chip_second=p))
                if label != 'observed_ratio':
                    continue
                # Independent midpoint quadrature in log speed, separately at every breakpoint.
                numeric = 0.
                for lo, hi, c in segments(points):
                    step = math.log(hi/lo)/128
                    if p > 0:
                        numeric += step*math.fsum(c/(1+r*c*math.exp(math.log(lo)+(i+.5)*step)/p) for i in range(128))
                if a:
                    max_error = max(max_error, abs(numeric/a-1))
                assert a <= combined_area(points, p, 0.)+1e-10
                for speed in (50, 100, 200, 300, 400):
                    c = capacity(points, speed)
                    b = c/(1+r*speed*c/p) if c and p > 0 else 0.
                    targets.append(dict(hardware=h, model=m, target_TPS=speed, decode_agents_chip=c,
                                        combined_agents_chip=b, prefill_chip_fraction=1-b/c if c else None))
            scenarios.append(dict(hardware=h, scenario=label, agent_index_GB300e=sum(vals)/4,
                                  **dict(zip(COMPONENTS, vals))))
    assert max_error < 1e-6
    lookup = {(v['hardware'], v['scenario']):v for v in scenarios}
    table = []
    for h in curves:
        row = dict(lookup[h,'observed_ratio']); row.pop('scenario')
        row['decode_only_GB300e'] = lookup[h,'decode_only']['agent_index_GB300e']
        row['prefill_only_GB300e'] = sum(pref[h,m]/pref[REFERENCE,m] for m in models)/4
        for label in ('half_input','double_input'):
            row[label+'_GB300e'] = lookup[h,label]['agent_index_GB300e']
        shares = [t['prefill_chip_fraction'] for t in targets if t['hardware']==h and t['target_TPS']==100 and t['prefill_chip_fraction'] is not None]
        row['prefill_fraction_at100_min'], row['prefill_fraction_at100_max'] = min(shares), max(shares)
        row['notes'] = 'Log-speed capacity AUC, 50–400 SOL TPS/user; equal GLM, Kimi, Qwen and MAI-2 2.8T weights; separately provisioned prefill/decode pools.'
        table.append(row)
    table.sort(key=lambda v:(v['hardware'] != REFERENCE, -v['agent_index_GB300e']))
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('curves_and_inputs.json', 'workload.json'):
        if (SOURCE/name).resolve() != (OUT/name).resolve():
            shutil.copyfile(SOURCE/name, OUT/name)
    for name, rows in [('index.csv',table), ('per_model_scores.csv',per_model), ('ratio_sensitivity.csv',scenarios), ('capacity_at_targets.csv',targets)]:
        save_csv(name, rows)
    native = SOURCE.resolve() == OUT.resolve()
    parent_provenance = SOURCE/('native_provenance.json' if native else 'provenance.json')
    inputs = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (SOURCE/'curves_and_inputs.json', SOURCE/'workload.json', parent_provenance)}
    provenance = dict(band=[LOW,HIGH], axis='log', models=models, weights=[.25]*4,
                      inputs=inputs, source_provenance=str(parent_provenance.relative_to(ROOT)) if parent_provenance.is_relative_to(ROOT) else parent_provenance.name,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      integration='Integral C_combined(s) ds/s; exact at all saved speed breakpoints; infeasible capacity is zero.',
                      simulation_rerun=native, upstream_commit='2f0f4cd8606d580567b253d02f3d699fe36db865')
    (OUT/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    checks = dict(hardware=23, models=4, complete_hardware_model_cells=92,
                  log_integral_max_relative_numeric_disagreement=max_error,
                  saved_curves_and_prefill_inputs_byte_identical=not native, simulation_rerun=native)
    (OUT/'validation.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks))
    print('VR200:', lookup['VR200','observed_ratio']['agent_index_GB300e'])


if __name__ == '__main__':
    main()
