"""Extended one-factor curves and capacity/bandwidth interaction grid.

Uses the identical Native evaluator and admission/search rules as run.py.
Frozen baseline/half/double results are reused only after source-hash checks.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import time
import yaml
from run import ROOT, REF, FIELDS, Native, KEYS, doctor, identity, metrics, envelope, save, REFERENCE

OUT=ROOT/'results/gb300-curvature-2026-10-19'
PREVIOUS=ROOT/'results/gb300-sensitivity-2026-10-18'
FACTORS=[.25,.375,.5,.625,.75,.875,1.,1.125,1.25,1.5,2.,3.,4.]
DIMS=['hbm_bandwidth','hbm_capacity','all_compute','interconnect_bandwidth']
GRID=[.5,.75,1.,1.5,2.]


def case_id(changes):
    parts=[f'{k}_{v:g}'.replace('.','p') for k,v in sorted(changes.items()) if v!=1]
    return '__'.join(parts) or 'baseline'


def cases():
    result={}
    for d in DIMS:
        for f in FACTORS:
            c={d:f} if f!=1 else {}
            result[case_id(c)]=c
    for c in GRID:
        for b in GRID:
            change={k:v for k,v in [('hbm_capacity',c),('hbm_bandwidth',b)] if v!=1}
            result[case_id(change)]=change
    return result


def previous_case(changes):
    if not changes:return 'baseline'
    if len(changes)==1:
        d,f=next(iter(changes.items()))
        if f in (.5,2):return d+('_half' if f==.5 else '_double')
    return None


def execute(job):
    case,changes,signature=job
    logging.disable(logging.WARNING)
    doctor(REF)
    frozen=json.loads((REF/'curves_and_inputs.json').read_text())
    hw0=json.loads((REF/'catalog.json').read_text())[REFERENCE]
    hw=dict(hw0,name=case,profile='curvature_'+case)
    pref0={r['model']:r for r in json.loads((REF/'prefill_winners.json').read_text()) if r['hardware']==REFERENCE}
    started=time.monotonic()
    results=[]
    for model in frozen['models']:
        path=OUT/case/(model+'.json')
        if path.exists():
            old=json.loads(path.read_text())
            assert old['study_signature']==signature,'Resume inputs changed'
            results.append(old['metrics']['combined_GB300e']);continue
        reuse=previous_case(changes)
        oldpath=PREVIOUS/reuse/(model+'.json') if reuse else None
        if oldpath and oldpath.exists():
            part=json.loads(oldpath.read_text())
            for name,digest in part['signature']['files'].items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Cannot reuse changed inputs: '+name
            part.update(study_signature=signature,case=case,changes=changes,
                        reused_from=str(oldpath.relative_to(ROOT)),reused_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest())
            save(path,part);results.append(part['metrics']['combined_GB300e']);continue
        engine=Native(OUT/'inputs',hw,model)
        audit,phases=[],{}
        locked_ids={identity(r) for r in frozen['curves'][REFERENCE][model]}
        locked_pref_id=identity(pref0[model])
        locked_decode,locked_pref,anchors=[],None,[]
        anchor_rows={}
        for speed in (50,100,200,400):
            winner=max((r for r in frozen['curves'][REFERENCE][model] if r['speed']>=speed),key=lambda r:r['capacity'],default=None)
            if winner:anchor_rows[speed]=winner
        for phase in ('decode','prefill'):
            frontier,best,count=[],None,0
            for p in engine.layouts(phase):
                try:cap=engine.batches(phase,p)
                except (ValueError,AssertionError) as exc:
                    assert model=='Kimi-K3' and phase=='prefill' and p['tp']==64 and str(exc)=='num_heads 96 should be divisible by tp_size 64 ',str(exc)
                    audit.append(dict(phase=phase,**p,status='invalid_heads',reason=str(exc)));continue
                first=math.ceil(p['ep']/(4*p['dp']*7)) if phase=='decode' and model.startswith('MAI2') else 1
                audit.append(dict(phase=phase,**p,min_batch=first,max_batch=cap))
                rows=[]
                for batch in range(first,cap+1):
                    row=engine.evaluate(phase,p,batch);count+=1
                    if phase=='decode':
                        rows.append(row)
                        if identity(row) in locked_ids:locked_decode.append(row)
                        for speed,old in anchor_rows.items():
                            if identity(old)==identity(row):anchors.append(dict(target=speed,baseline=old,variant=row,fixed_configuration_speedup=row['speed']/old['speed']))
                    else:
                        if best is None or row['input_tokens_chip_second']>best['input_tokens_chip_second']:best=row
                        if identity(row)==locked_pref_id:locked_pref=row
                if rows:frontier=envelope(frontier+rows)
            phases[phase]=dict(frontier=frontier,best=best,count=count)
        f,p=phases['decode']['frontier'],phases['prefill']['best']
        targets=[]
        for speed in (50,100,200,400):
            winner=max((r for r in f if r['speed']>=speed),key=lambda r:r['capacity'],default=None)
            c=winner['capacity'] if winner else 0.
            rate=p['input_tokens_chip_second'] if p else 0.
            targets.append(dict(speed=speed,decode_capacity=c,combined_capacity=c/(1+frozen['ratio']*speed*c/rate) if c and rate else 0.,winner=winner))
        part=dict(study_signature=signature,case=case,changes=changes,model=model,
            metrics=metrics(f,p,frozen,model),frontier=f,best_prefill=p,
            locked_metrics=metrics(locked_decode,locked_pref,frozen,model),
            fixed_configuration_anchors=anchors,targets=targets,audit=audit,
            candidates={k:v['count'] for k,v in phases.items()},elapsed_seconds=time.monotonic()-started,reused_from=None)
        save(path,part);results.append(part['metrics']['combined_GB300e'])
        print('MODEL',case,model,round(results[-1],5),flush=True)
    return dict(case=case,score=sum(results)/4,seconds=round(time.monotonic()-started,1))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();runtime=doctor(REF)
    paths=[Path(__file__),Path(__file__).with_name('run.py'),*sorted((ROOT/'src/agent_index').glob('*.py')),
           ROOT/'studies/log400_release.py',REF/'manifest.json',REF/'curves_and_inputs.json',REF/'prefill_winners.json']
    signature=dict(runtime=runtime,files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    inp=OUT/'inputs';inp.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(REF/'models.json',inp/'models.json');shutil.copytree(REF/'models',inp/'models',dirs_exist_ok=True)
    (inp/'hardware').mkdir(exist_ok=True)
    base=yaml.safe_load((REF/'hardware/gb300_deployment.yaml').read_text())
    jobs=[]
    for case,changes in cases().items():
        profile=copy.deepcopy(base)
        for dim,factor in changes.items():
            for section,key in FIELDS[dim]:
                profile[section][key]*=factor
                if key=='mem_capacity':profile[section][key]=int(profile[section][key])
        (inp/'hardware'/('curvature_'+case+'.yaml')).write_text(yaml.safe_dump(profile))
        jobs.append((case,changes,signature))
    save(OUT/'provenance.json',dict(signature=signature,factors=FACTORS,dimensions=DIMS,grid=GRID,cases=cases(),
        baseline_profile=base,band=[50,400],axis='log',notes=[
        'Same evaluator, four-model basket, nextn3/AL3, 256Ki context, 1Ki prefill, precision and memory policies as prior sensitivity study.',
        'Scale-up domain and chip budget fixed at 72. Four one-factor curves and a 5x5 HBM capacity/bandwidth grid.',
        'Baseline and 0.5x/2x one-factor cases reused with per-file source identity checks; each reuse has source path/hash.',
        'Every newly evaluated case reoptimizes every allowed layout over all admitted integer batches.',
        'Connecting resource samples in plots is illustrative; no score inferred by interpolation. Index integration remains exact at frontier breakpoints.',
        'Latency and SRAM effects remain outside this analytical sensitivity.']))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute,j) for j in jobs]
        for i,future in enumerate(as_completed(futures),1):
            print('COMPLETE',i,'/',len(jobs),json.dumps(future.result()),flush=True)
    save(OUT/'complete.json',dict(cases=len(jobs),models=4,status='complete'))


if __name__=='__main__':main()
