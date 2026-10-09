"""CLI. Simulation dependencies are loaded only by simulation commands."""
import argparse
import json
from pathlib import Path

from .index import build, root_path, verify_inputs, validate_reference


def main(argv=None):
    parser=argparse.ArgumentParser(description="Reproduce the frozen agent capacity index")
    parser.add_argument('--root',type=Path,default=root_path(),help='Repository with configs/ and reference/')
    sub=parser.add_subparsers(dest='command',required=True)
    rebuild=sub.add_parser('rebuild',help='Re-integrate frozen frontiers and write CSV/HTML/JSON; no AISim')
    rebuild.add_argument('--output',type=Path,default=Path('results/frozen'))
    sub.add_parser('validate',help='Check input integrity and parity with all frozen scores; no simulations')
    sub.add_parser('doctor',help='Inspect optional AISim version/runtime/model hashes; no simulation')
    rerun=sub.add_parser('rerun',help='Re-evaluate saved frontier candidates using the pinned AISim runtime')
    rerun.add_argument('--hardware',action='append',help='Exact catalog key; repeat to select several')
    rerun.add_argument('--model',action='append',choices=['GLM-5.2','Kimi-K3'])
    rerun.add_argument('--limit',type=int,help='Maximum candidates; omitted means all selected saved frontier points')
    rerun.add_argument('--output',type=Path,default=Path('results/rerun'))
    sweep=sub.add_parser('sweep',help='Fresh layout/batch search; does not overwrite frozen data')
    sweep.add_argument('--hardware',action='append',help='Exact catalog key; omitted means all hardware')
    sweep.add_argument('--model',action='append',choices=['GLM-5.2','Kimi-K3'])
    sweep.add_argument('--output',type=Path,default=Path('results/sweep'))
    args=parser.parse_args(argv)
    root=args.root.resolve()
    if args.command in ('doctor','rerun','sweep'):
        from .runner import run
        return run(args,root)
    checked=verify_inputs(root)
    data=build(root)
    validation=validate_reference(root,data)
    validation['input_hash_checks']=checked
    if args.command=='rebuild':
        from .publication import publish
        print(f"Rebuilt {publish(root,data,args.output,validation) / 'REPORT.html'}")
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    main()
