"""One-factor GB300 sensitivity; reruns native timings, admission and layout search.

Run: .venv-current/bin/python studies/gb300_sensitivity/run.py
Outputs are separate from immutable release inputs. No native AISim modifications.
"""
import argparse
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'studies')]
import yaml
from agent_index.current_native import Native, KEYS, doctor
from agent_index.index import envelope, capacity
from log400_release import combined_area, REFERENCE

REF = ROOT/'reference/2026-10-15'
OUT = ROOT/'results/gb300-sensitivity-2026-10-18'
FIELDS = {
    'hbm_bandwidth': [('gpu', 'mem_bw')],
    'hbm_capacity': [('gpu', 'mem_capacity')],
    'all_compute': [('gpu', k) for k in ('bfloat16_tc_flops', 'fp8_tc_flops', 'fp4_tc_flops')],
    'bf16_compute': [('gpu', 'bfloat16_tc_flops')],
    'fp8_compute': [('gpu', 'fp8_tc_flops')],
    'fp4_compute': [('gpu', 'fp4_tc_flops')],
    'interconnect_bandwidth': [('node', k) for k in ('intra_node_bw', 'inter_node_bw', 'inter_rack_bw')],
}
SCENARIOS = {'baseline': ('baseline', 1.)}
SCENARIOS.update({f'{name}_{suffix}': (name, factor) for name in FIELDS
                 for suffix, factor in [('half', .5), ('double', 2.)]})


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def identity(row):
    return tuple(row[k] for k in KEYS)+ (row['batch'],)


def metrics(frontier, prefill, frozen, model):
    reference_p = next(r['tokens_chip_second'] for r in frozen['prefill_rates']
                       if r['hardware']==REFERENCE and r['model']==model)
    reference_f = frozen['curves'][REFERENCE][model]
    p = prefill['input_tokens_chip_second'] if prefill else 0.
    a = combined_area(frontier, p, frozen['ratio'])
    ref_a = combined_area(reference_f, reference_p, frozen['ratio'])
    return dict(combined_GB300e=a/ref_a, combined_auc=a, reference_auc=ref_a,
                decode_GB300e=combined_area(frontier,p,0)/combined_area(reference_f,reference_p,0),
                prefill_GB300e=p/reference_p, prefill_tokens_chip_second=p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=list(SCENARIOS), action='append')
    parser.add_argument('--model', action='append')
    args = parser.parse_args()
    runtime = doctor(REF)
    logging.disable(logging.WARNING)
    frozen = json.loads((REF/'curves_and_inputs.json').read_text())
    hw0 = json.loads((REF/'catalog.json').read_text())[REFERENCE]
    pref0 = {r['model']:r for r in json.loads((REF/'prefill_winners.json').read_text()) if r['hardware']==REFERENCE}
    profile0 = yaml.safe_load((REF/'hardware/gb300_deployment.yaml').read_text())
    paths = [Path(__file__), *sorted((ROOT/'src/agent_index').glob('*.py')),
             ROOT/'studies/log400_release.py', REF/'manifest.json', REF/'curves_and_inputs.json',
             REF/'prefill_winners.json', REF/'hardware/gb300_deployment.yaml']
    signature = dict(runtime=runtime, files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    inputs = OUT/'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REF/'models.json', inputs/'models.json')
    shutil.copytree(REF/'models', inputs/'models', dirs_exist_ok=True)
    (inputs/'hardware').mkdir(exist_ok=True)
    save(OUT/'provenance.json', dict(signature=signature, baseline_profile=profile0,
        scenarios=SCENARIOS, fields=FIELDS, ratio=frozen['ratio'], band=[50,400], axis='log',
        notes=['One-factor changes; no scale-up/domain-size change; 72-chip budget.',
               'Native nextn3/AL3, 256Ki cache; throughput-optimized cached-prefix prefill with 1Ki new tokens.',
               'Frozen precision/storage formats and admission policy, 90% memory, 4GiB reservation, metadata guards.',
               'p2p_latency=0 in baseline: multiplicative latency sensitivity is identically zero and omitted.',
               'All three interconnect tiers scaled together, preserving topology; layouts never exceed 72-chip local domain.',
               'Exhaustive integer batches within existing finite layout search, not exhaustive over all possible implementations.',
               'Frozen-candidate replay reoptimizes only among previously winning decode candidates; prefill winner fixed. Memory-infeasible candidates dropped.',
               'Original frozen GB300 normalization retained. Baseline parity reported, never assumed.']))
    for scenario in args.scenario or SCENARIOS:
        dimension, factor = SCENARIOS[scenario]
        profile = copy.deepcopy(profile0)
        for section,key in FIELDS.get(dimension,[]):
            profile[section][key] *= factor
            if key=='mem_capacity': profile[section][key] = int(profile[section][key])
        profile_name = 'gb300_sensitivity_'+scenario
        (inputs/'hardware'/(profile_name+'.yaml')).write_text(yaml.safe_dump(profile))
        hw = dict(hw0, name=scenario, profile=profile_name)
        for model in args.model or frozen['models']:
            path = OUT/scenario/(model+'.json')
            if path.exists():
                if json.loads(path.read_text())['signature'] != signature:
                    raise RuntimeError('Existing results have different source signature; choose a fresh output directory')
                print('cached', scenario, model, flush=True)
                continue
            started = time.monotonic()
            engine = Native(inputs, hw, model)
            audit, phases = [], {}
            locked_decode_ids = {identity(r) for r in frozen['curves'][REFERENCE][model]}
            locked_pref_id = identity(pref0[model])
            locked_decode, locked_pref, anchors = [], None, []
            anchor_rows = {}
            for speed in (50,100,200,400):
                winner = max((r for r in frozen['curves'][REFERENCE][model] if r['speed']>=speed),
                             key=lambda r:r['capacity'], default=None)
                if winner: anchor_rows[speed] = winner
            for phase in ('decode','prefill'):
                frontier, best, count = [], None, 0
                layouts = list(engine.layouts(phase))
                for i,p in enumerate(layouts):
                    try:
                        cap = engine.batches(phase,p)
                    except (ValueError,AssertionError) as exc:
                        audit.append(dict(phase=phase,**p,status='unsupported',reason=str(exc)))
                        continue
                    first = math.ceil(p['ep']/(4*p['dp']*7)) if phase=='decode' and model.startswith('MAI2') else 1
                    audit.append(dict(phase=phase,**p,min_batch=first,max_batch=cap))
                    rows = []
                    for batch in range(first,cap+1):
                        row = engine.evaluate(phase,p,batch)
                        count += 1
                        if phase=='decode':
                            rows.append(row)
                            if identity(row) in locked_decode_ids: locked_decode.append(row)
                            for speed,old in anchor_rows.items():
                                if identity(old)==identity(row):
                                    anchors.append(dict(target=speed,baseline=old,variant=row,
                                                        fixed_configuration_speedup=row['speed']/old['speed']))
                        else:
                            if best is None or row['input_tokens_chip_second']>best['input_tokens_chip_second']: best=row
                            if identity(row)==locked_pref_id: locked_pref=row
                    if rows: frontier=envelope(frontier+rows)
                    if (i+1)%100==0:
                        print(scenario,model,phase,i+1,'/',len(layouts),'layouts',count,'candidates',flush=True)
                phases[phase] = dict(frontier=frontier,best=best,count=count)
                print(scenario,model,phase,count,'candidates',round(time.monotonic()-started,1),'seconds',flush=True)
            f,p = phases['decode']['frontier'],phases['prefill']['best']
            targets=[]
            for speed in (50,100,200,400):
                winner=max((r for r in f if r['speed']>=speed),key=lambda r:r['capacity'],default=None)
                c=winner['capacity'] if winner else 0.
                rate=p['input_tokens_chip_second'] if p else 0.
                targets.append(dict(speed=speed,decode_capacity=c,combined_capacity=c/(1+frozen['ratio']*speed*c/rate) if c and rate else 0.,winner=winner))
            part=dict(signature=signature,scenario=scenario,dimension=dimension,factor=factor,model=model,
                metrics=metrics(f,p,frozen,model), frontier=f,best_prefill=p,
                locked_metrics=metrics(locked_decode,locked_pref,frozen,model),
                locked_frontier=envelope(locked_decode),locked_prefill=locked_pref,fixed_configuration_anchors=anchors,
                targets=targets,audit=audit,candidates={k:v['count'] for k,v in phases.items()},
                elapsed_seconds=time.monotonic()-started)
            save(path,part)
            print('SCORE',scenario,model,part['metrics'],flush=True)


if __name__=='__main__': main()
