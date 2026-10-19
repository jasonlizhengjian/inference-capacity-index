#!/usr/bin/env python3
"""Inference Capacity Index v1: rebuild, native replay/search, or portable bundle."""
import argparse
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
REF = ROOT/'reference/2026-10-15'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'studies')]
from agent_index.index import envelope
import log400_release as calculation


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify():
    manifest=json.loads((REF/'manifest.json').read_text())
    for name,digest in manifest.items():
        if hashlib.sha256((REF/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen input changed: '+name)
    return len(manifest)


def parity(out):
    expected={r['hardware']:r for r in csv.DictReader((REF/'expected_index.csv').open())}
    actual=list(csv.DictReader((out/'index.csv').open()))
    columns=['agent_index_GB300e','decode_only_GB300e','prefill_only_GB300e',*calculation.COMPONENTS]
    if {r['hardware'] for r in actual}!=set(expected):raise RuntimeError('Incomplete hardware set')
    worst=0.
    for r in actual:
        for key in columns:
            a,b=float(r[key]),float(expected[r['hardware']][key])
            worst=max(worst,abs(a-b)/max(abs(b),1e-15))
    if worst>1e-8:raise RuntimeError(f'Score mismatch: maximum relative error {worst}')
    result=dict(hardware=23,models=4,score_cells=23*len(columns),max_relative_error=worst)
    save(out/'reproduction.json',result);print(json.dumps(result))


def render(out):
    import methodology_visual as method
    import performance_knees_visual as knees
    import accelerator_comparison_visual as comparison
    method.STUDY=out;method.INPUT=out/'curves_and_inputs.json';method.OUT=out
    knees.OUT=out;comparison.STUDY=out;comparison.OUT=out
    method.main();knees.main();comparison.main()


def native(args, out):
    from agent_index.current_native import Native, doctor, KEYS
    runtime=doctor(REF);logging.getLogger().setLevel(logging.ERROR)
    frozen=json.loads((REF/'curves_and_inputs.json').read_text())
    catalog=json.loads((REF/'catalog.json').read_text())
    pref={(r['hardware'],r['model']):r for r in json.loads((REF/'prefill_winners.json').read_text())}
    aliases={'MAI-2 2.8T':'MAI2-informed-h128-w512-sh4352','Qwen3-Coder 480B':'Qwen-480B'}
    selected=[aliases.get(m,m) for m in args.model] if args.model else frozen['models']
    if set(selected)-set(frozen['models']):raise ValueError('Unknown model filter')
    if args.hardware and set(args.hardware)-set(catalog):raise ValueError('Unknown hardware filter')
    signature=dict(runtime=runtime,mode=args.command,limit=args.limit,
        inputs=hashlib.sha256((REF/'manifest.json').read_bytes()).hexdigest(),
        code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (Path(__file__),ROOT/'src/agent_index/current_native.py',ROOT/'src/agent_index/current_models.py',
               ROOT/'src/agent_index/simulation.py',ROOT/'src/agent_index/topology.py')})
    curves={};rates=[];count=0;max_timing_error=0.
    for number,(h,hw) in enumerate(catalog.items()):
        if args.hardware and h not in args.hardware:continue
        curves[h]={}
        for model in selected:
            path=out/'parts'/f'{number:02d}-{model}.json'
            if path.exists():
                part=json.loads(path.read_text())
                if part['signature']!=signature:raise RuntimeError('Run signature changed; choose a fresh output folder')
                print('Resume',h,model,flush=True)
            else:
                engine=Native(REF,hw|{'name':h},model);observations={};audit=[];errors=[]
                for phase in ('decode','prefill'):
                    rows=[]
                    if args.command=='replay':
                        saved=frozen['curves'][h][model] if phase=='decode' else [pref[h,model]]
                        if args.limit:saved=saved[:args.limit]
                        for old in saved:
                            p={k:int(old[k]) for k in KEYS};row=engine.evaluate(phase,p,int(old['batch']))
                            err=abs(row['raw_ms']/old['raw_ms']-1);errors.append(err)
                            memory=old.get('memory',old.get('memory_gib'))
                            if err>1e-9 or not math.isclose(row['memory'],memory,rel_tol=1e-9):
                                raise RuntimeError(f'Timing/memory mismatch: {h}, {model}, {phase}, {p}, B={old["batch"]}: {row["raw_ms"]} vs {old["raw_ms"]}')
                            rows.append(row)
                    else:
                        for p in engine.layouts(phase):
                            try:cap=engine.batches(phase,p)
                            except (ValueError, AssertionError) as exc:
                                # Native prefill graphs reject some head/TP divisions.
                                # Keep the exclusion visible, never invent a timing.
                                if phase!='prefill':raise
                                audit.append(dict(phase=phase,**p,status='unsupported',reason=str(exc)));continue
                            # Original MAI study excludes native floored zero
                            # expert-read costs at fewer assignments than EP ranks.
                            first=math.ceil(p['ep']/(4*p['dp']*7)) if phase=='decode' and model.startswith('MAI2') else 1
                            audit.append(dict(phase=phase,**p,min_batch=first,max_batch=cap))
                            for b in range(first,cap+1):rows.append(engine.evaluate(phase,p,b))
                    observations[phase]=rows
                part=dict(signature=signature,hardware=h,model=model,
                    frontier=envelope(observations['decode']),best_prefill=max(observations['prefill'],key=lambda r:r['input_tokens_chip_second']) if observations['prefill'] else None,
                    decode_candidates=len(observations['decode']),prefill_candidates=len(observations['prefill']),
                    max_timing_error=max(errors,default=0.),audit=audit)
                save(path,part)
                print('Completed',h,model,part['decode_candidates'],'decode,',part['prefill_candidates'],'prefill',flush=True)
            curves[h][model]=part['frontier'];winner=part['best_prefill']
            rates.append(dict(hardware=h,model=model,tokens_chip_second=winner['input_tokens_chip_second'] if winner else 0.))
            count+=part['decode_candidates']+part['prefill_candidates']
            max_timing_error=max(max_timing_error,part['max_timing_error'])
    complete=not args.limit and set(curves)==set(catalog) and set(selected)==set(frozen['models'])
    save(out/'native_validation.json',dict(complete_index=complete,evaluated_candidates=count,max_timing_relative_error=max_timing_error,mode=args.command,runtime=runtime))
    if not complete:
        print('Selected cases complete; not a full index. No partial-basket score published.');return
    save(out/'curves_and_inputs.json',dict(models=frozen['models'],curves=curves,prefill_rates=rates,ratio=frozen['ratio']))
    shutil.copyfile(REF/'workload.json',out/'workload.json')
    save(out/'native_provenance.json',signature)
    calculation.main(out,out);parity(out)


def bundle(destination):
    # Explicit allowlist: no workspace logs, private reference documents, credentials,
    # environments, caches, or unrelated research scripts.
    runtime=json.loads((REF/'runtime.json').read_text())
    wheel=ROOT/'vendor'/runtime['wheel']
    if not wheel.exists() or hashlib.sha256(wheel.read_bytes()).hexdigest()!=runtime['wheel_sha256']:
        raise RuntimeError('Complete bundle requires the verified frozen wheel in vendor/')
    names=['README.md','reproduce.py','run.py','pyproject.toml','.gitignore','.gitattributes',
           'THIRD_PARTY_NOTICES.md','LICENSES','src','configs','reference','environment','tests',
           'docs','vendor','studies/log400_release.py','studies/methodology_visual.py',
           'studies/performance_knees_visual.py','studies/accelerator_comparison_visual.py',
           'studies/inferencex_validation']
    destination.parent.mkdir(parents=True,exist_ok=True)
    def filter_file(info):
        return None if '__pycache__' in Path(info.name).parts or info.name.endswith(('.pyc','.pyo')) else info
    with tarfile.open(destination,'w:gz') as archive:
        for name in names:archive.add(ROOT/name,arcname='inference-capacity-index/'+name,filter=filter_file)
    print(destination,hashlib.sha256(destination.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['rebuild','doctor','replay','sweep','bundle'],nargs='?',default='rebuild')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--hardware',action='append',help='Exact hardware key from reference/2026-10-15/catalog.json')
    parser.add_argument('--model',action='append',help='GLM-5.2, Kimi-K3, Qwen-480B or "MAI-2 2.8T"')
    parser.add_argument('--limit',type=int,help='Replay: limit decode candidates per hardware/model; still check each prefill winner')
    args=parser.parse_args()
    if args.limit is not None and (args.limit<1 or args.command!='replay'):parser.error('--limit is a positive replay-only option')
    verify()
    out=(args.output or Path('results/current' if args.command=='rebuild' else 'results/current-'+args.command))
    if not out.is_absolute():out=ROOT/out
    if args.command=='bundle':bundle(args.output or ROOT/'dist/inference-capacity-index-v1.0.0.tar.gz')
    elif args.command=='doctor':
        from agent_index.current_native import doctor
        print(json.dumps(doctor(REF),indent=2))
    elif args.command=='rebuild':
        calculation.main(REF,out);parity(out);render(out)
    else:native(args,out)


if __name__=='__main__':
    main()
