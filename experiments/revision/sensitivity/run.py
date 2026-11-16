"""Phase 4 CLI: check -> smoke -> run. self-test is synthetic development only."""
import argparse
import copy
import os
from pathlib import Path
import torch
from experiments.revision import common
from experiments.revision.core import exclusive_run
from . import verification, strength, gates, random_size, readout, reporting
from .design import ALPHAS, independent_output, read_json


def run_formal(args):
    verification.verify_smoke(args)
    if not torch.cuda.is_available():
        raise ValueError('Formal Phase 4 requires the validated CUDA environment')
    verification.validate_upstream(args)
    done=args.output_root/'PHASE4_COMPLETE.json'
    if done.exists():
        completed=read_json(done)
        for name,digest in completed['artifact_hashes'].items():
            if common.hash_file(args.output_root/name)!=digest:
                raise ValueError(f'Completed Phase 4 artifact changed: {name}')
        print('PHASE 4 COMPLETE (verified existing artifacts)',flush=True)
        return
    print('PHASE 4: fixed-selection gate sensitivity',flush=True)
    gates.run(args)
    for alpha in ALPHAS:
        sub=copy.copy(args);sub.output_root=args.output_root/'strength'/f'alpha{alpha:g}'
        sub.policy=sub.output_root/'policy.json'
        common.write_immutable_json(sub.policy,read_json(common.ROOT/'protocols/revision/h3.json'))
        for target in common.TARGETS:
            for seed in common.SEEDS:
                strength.configure(alpha,target,seed,args.phase2)
                sub.target,sub.seed=target,seed
                print(f'PHASE 4: alpha={alpha} {target} seed={seed}',flush=True)
                strength.run_pair(sub)
        strength.analyze(sub)
    print('PHASE 4: shared random-deletion fits',flush=True)
    random_size.run(args)
    print('PHASE 4: target-supervised post-hoc readout',flush=True)
    readout.run(args)
    gates.run(args,include_strength=True)
    # Preserve the reference snapshot even if an upstream file changes mid-run.
    common.write_immutable_json(args.output_root/'UPSTREAM_REFERENCE_LOCK.json',verification.upstream_snapshot(args))
    reporting.finish(args)
    print(f'PHASE 4 COMPLETE: {args.output_root}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','smoke','run','self-test'])
    parser.add_argument('--phase2',type=Path,default=common.ROOT/'results/generated/revision/core')
    parser.add_argument('--phase3',type=Path,default=common.ROOT/'results/generated/revision/comparisons')
    parser.add_argument('--historical',type=Path,default=common.ROOT/'results/generated/h2_h3')
    parser.add_argument('--output-root',type=Path,default=common.ROOT/'results/generated/revision/sensitivity')
    parser.add_argument('--execute', action='store_true')
    args=parser.parse_args()
    if not args.execute:
        parser.error('Stage execution requires --execute; use run_revision.py plan for a safe preview')
    os.chdir(common.ROOT)
    for key in ('phase2','phase3','historical'):
        setattr(args,key,getattr(args,key).resolve())
    args.output_root=independent_output(args.output_root,[args.phase2,args.phase3,args.historical,common.PREP])
    args.execute,args.resume,args.cpu,args.chunk_size=True,True,False,250000
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    with exclusive_run(common.ROOT/'results/generated/.revision.lock'):
        if args.command=='self-test':
            verification.lock_local(args)
        else:
            verification.check(args)
        if args.command=='check':
            return
        if args.command in ('smoke','self-test'):
            from .smoke import run
            run(args,formal_check_passed=args.command=='smoke')
        else:
            run_formal(args)


if __name__=='__main__':
    main()
