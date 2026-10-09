"""Opt-in native reevaluation/search. Never overwrites immutable references."""
import csv
import base64
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
from pathlib import Path

from .index import MODELS, build, load_catalog, saved_candidates, verify_inputs


def doctor(root):
    expected=json.loads((root/'environment/runtime.json').read_text())
    try:
        dist=importlib.metadata.distribution('aisimulate')
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError('AISim is not installed. Frozen rebuild/validate do not require it; see environment/README.md.') from error
    actual={"aisimulate_version":dist.version,"python_version":platform.python_version(),"hashes":{}}
    files={expected['runtime_path']:expected['runtime_sha256'],**expected['model_config_sha256']}
    for name,digest in files.items():
        path=Path(dist.locate_file(name))
        found=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        actual['hashes'][name]=found
        if found!=digest:
            raise RuntimeError(f'Pinned runtime/model identity mismatch: {name}; expected {digest}, found {found}')
    if actual['aisimulate_version']!=expected['aisimulate_version']:
        raise RuntimeError('AISim version mismatch')
    checked=0
    for file in dist.files or []:
        if not file.hash:
            continue
        path=Path(dist.locate_file(file))
        if not path.is_file():
            raise RuntimeError(f'Installed AISim file missing: {file}')
        digest=base64.urlsafe_b64encode(hashlib.new(file.hash.mode,path.read_bytes()).digest()).rstrip(b'=').decode()
        if digest!=file.hash.value:
            raise RuntimeError(f'Installed AISim file differs from package RECORD: {file}')
        checked+=1
    if not checked:
        raise RuntimeError('No installed package hashes; use the pinned noneditable AISim wheel')
    actual['installed_package_file_hashes_checked']=checked
    actual['pinned_identity_matches']=True
    return actual


def to_point(row, key):
    return dict(hardware=key,model=row['model'],speed=row['tps_user'],capacity=row['concurrency_per_rank'],
                chips=row['world'],requests=row['total_requests'],tp=row['tp'],dp=row['dp'],pp=row['pp'],
                dcp=row['dcp'],etp=row['etp'],ep=row['ep'],batch=row['batch_per_replica'],
                memory=row['memory_gib'],raw_ms=row['raw_ms'],layout=row['parallelism'],source='native_rerun')


def run(args,root):
    verify_inputs(root)
    identity=doctor(root)
    if args.command=='doctor':
        print(json.dumps(identity,indent=2));return
    from .simulation import evaluate_candidate, sweep
    class OncePerMessage(logging.Filter):
        def __init__(self):
            super().__init__();self.seen=set();self.suppressed=0
        def filter(self,record):
            message=record.getMessage()
            if message in self.seen:
                self.suppressed+=1;return False
            self.seen.add(message);return True
    warnings=OncePerMessage()
    logging.getLogger('aisimulate_core.sdk.models.base').addFilter(warnings)
    catalog=load_catalog(root)
    hardware=args.hardware or list(catalog)
    models=args.model or list(MODELS)
    if any(h not in catalog for h in hardware):
        raise ValueError('Unknown hardware. Exact keys: '+', '.join(catalog))
    if getattr(args,'limit',None) is not None and args.limit<1:
        raise ValueError('--limit must be positive')
    destination=args.output.resolve()
    if destination==root or any(destination.is_relative_to((root/p).resolve()) for p in ('reference','configs','src','docs','environment')):
        raise ValueError('Use a separate results directory')
    destination.mkdir(parents=True,exist_ok=True)
    raw_path=destination/'native_candidates.csv'
    if raw_path.exists():
        raise FileExistsError(f'{raw_path} exists; choose a fresh --output to preserve prior runs')
    systems=root/'configs/hardware'
    points=[];checks=[];admission=[];n=0
    if args.command=='rerun':
        candidates=[p for p in saved_candidates(root) if p['hardware'] in hardware and p['model'] in models]
        if args.limit is not None:candidates=candidates[:args.limit]
        def generate():
            for p in candidates:
                dims={k:p[k] for k in ('tp','dp','dcp','etp','ep','pp','batch')}
                yield p['hardware'],evaluate_candidate(catalog[p['hardware']],p['model'],systems_path=systems,**dims),p
    else:
        def generate():
            for h in hardware:
                for model in models:
                    print(f'Searching {h} / {model}',flush=True)
                    for row in sweep(catalog[h],model,systems_path=systems,
                                     audit=lambda event:admission.append(dict(hardware=h,model=model,**event))):
                        yield h,row,None
    source_paths=sorted((root/'src').rglob('*.py'))+[root/'environment/requirements-aisim.lock']
    state=dict(command=args.command,status='running',identity=identity,hardware=hardware,models=models,
               code_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (destination/'run.json').write_text(json.dumps(state,indent=2)+'\n')
    with raw_path.open('w',newline='',encoding='utf-8') as stream:
        writer=None
        for h,row,expected in generate():
            row['hardware']=h
            if writer is None:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
            writer.writerow(row)
            if not row['fits']:
                raise ValueError(f'Replayed candidate no longer fits: {h}/{row["model"]}')
            point=to_point(row,h);points.append(point);n+=1
            if expected is not None:
                errors={k:abs(point[k]-expected[k])/max(abs(expected[k]),1e-20) for k in ('raw_ms','speed','memory','capacity')}
                checks.append(dict(hardware=h,model=row['model'],batch=point['batch'],layout=point['layout'],relative_errors=errors))
                if any(v>1e-9 for v in errors.values()):
                    (destination/'parity_failure.json').write_text(json.dumps(checks[-1],indent=2)+'\n')
                    raise ValueError(f'Native parity failure: {checks[-1]}')
            if n%1000==0:
                stream.flush();print(f'{n} native candidates evaluated',flush=True)
    coverage={(p['hardware'],p['model']) for p in points}
    complete=coverage=={(h,m) for h in catalog for m in MODELS}
    # A limited replay may cover the labels but not all frontier points.
    if args.command=='rerun' and args.limit is not None:complete=False
    state.update(status='simulation_complete',candidate_count=n,complete_portfolio=complete,
                 candidate_csv_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                 maximum_relative_errors={k:max((c['relative_errors'][k] for c in checks),default=0) for k in ('raw_ms','speed','memory','capacity')})
    (destination/'run.json').write_text(json.dumps(state,indent=2)+'\n')
    if checks:(destination/'parity.json').write_text(json.dumps(checks,indent=2)+'\n')
    if admission:(destination/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
    if complete:
        from .publication import publish
        from .index import validate_reference
        data=build(root,points)
        validation=validate_reference(root,data)
        validation.update(simulation_rerun=True,search_kind=args.command,candidate_count=n)
        publish(root,data,destination/'index',validation)
        state['index_reference_parity']=True
    state['status']='complete'
    state['repeated_native_log_messages_suppressed']=warnings.suppressed
    state['native_warning_policy']='Each unique native model message printed once; repeats counted, formulas unchanged.'
    (destination/'run.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state,indent=2))
