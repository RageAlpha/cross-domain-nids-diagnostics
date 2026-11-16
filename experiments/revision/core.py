"""Run corrected H2/H3 experiments from the fixed main-experiment inputs."""
from pathlib import Path
import argparse
import copy
import json
import os
import sys
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

@contextmanager
def exclusive_run(path):
    """OS lock is released after crashes; the file itself may safely remain."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)

def main():
    commands = ['check', 'smoke', 'run', 'h2-shift', 'h2-candidates', 'h2-select', 'h2-random', 'h2-analyze', 'h3-pair', 'h3-analyze']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=commands, nargs='?', default='check')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/generated/revision/core')
    parser.add_argument('--chunk-size', type=int, default=250000)
    parser.add_argument('--target', choices=['CICIDS2018', 'BoT-IoT', 'ToN-IoT'])
    parser.add_argument('--seed', type=int, choices=[42, 123, 2024])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Stage execution requires --execute; use run_revision.py plan')
    os.chdir(ROOT)
    from .common import generated_output
    args.output_root = generated_output(args.output_root)
    if not args.output_root.is_relative_to(ROOT / 'results') or args.output_root == ROOT / 'results' or any(
        part.startswith(('protocol_', 'revision_phase1_')) for part in args.output_root.relative_to(ROOT / 'results').parts):
        parser.error('Use a new result directory under project/results; historical output is protected.')
    if args.chunk_size < 1:
        parser.error('--chunk-size must be positive')
    if args.command == 'h3-pair' and (args.target is None or args.seed is None):
        parser.error('h3-pair requires --target and --seed')
    import torch
    from experiments.revision import common, h2, h2_analysis, h3
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.execute, args.resume, args.cpu, args.corr_sample_rows = True, True, False, 300000
    policies = ROOT / 'protocols/revision'
    args.policy = policies / 'h2.json'
    args.h3_policy = policies / 'h3.json'
    with exclusive_run(ROOT / 'results/generated/.revision.lock'):
        common.verify_and_lock(args.output_root)
        h3.load_policy(args.h3_policy)
        policy = json.loads(args.policy.read_text())
        if policy['protocol_id'] != common.PROTOCOL_ID or policy['random_replicates_per_seed'] != 10:
            raise ValueError('H2 policy differs from the approved corrected protocol')
        if args.command == 'check':
            print('CHECK PASS: corrected partitions, 10 masks, Phase 2 only. No formal training started.', flush=True)
            return
        if args.command == 'smoke':
            from experiments.revision.smoke import run
            run(args)
            return
        smoke_path = args.output_root / 'smoke/SMOKE_PASS.json'
        if not smoke_path.exists():
            raise ValueError('First run: python run_revision.py core smoke --execute')
        passed = json.loads(smoke_path.read_text())
        if passed['input_lock_hash'] != common.hash_file(args.output_root / 'INPUT_LOCK.json'):
            raise ValueError('Smoke check does not match this input lock')
        if passed['environment_hash'] != common.hash_file(args.output_root / 'environment.json'):
            raise ValueError('Environment changed since smoke; rerun the smoke command first.')
        if not torch.cuda.is_available():
            raise RuntimeError('Formal MLP/H3 execution requires the configured CUDA environment.')
        dispatch = {'h2-shift': h2.shift, 'h2-candidates': h2.candidates, 'h2-select': h2_analysis.select_and_lock,
                    'h2-random': h2.random_controls, 'h2-analyze': h2_analysis.analyze}
        if args.command == 'run':
            for name, action in dispatch.items():
                print(f'=== {name} ===', flush=True)
                action(args)
            args.policy = args.h3_policy
            for target in common.TARGETS:
                for seed in common.SEEDS:
                    args.target, args.seed = target, seed
                    print(f'=== h3-pair {target} seed={seed} ===', flush=True)
                    h3.run_pair(args)
            h3.analyze(args)
            from experiments.revision.reporting import report
            report(args.output_root)
            print(f'PHASE 2 COMPLETE: {args.output_root}', flush=True)
        elif args.command in dispatch:
            dispatch[args.command](args)
        else:
            args.policy = args.h3_policy
            (h3.run_pair if args.command == 'h3-pair' else h3.analyze)(args)

if __name__ == '__main__':
    main()
