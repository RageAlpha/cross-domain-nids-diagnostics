"""Run corrected H2/H3 experiments or inspect their fixed inputs."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
for path in (ROOT / 'src', ROOT):
    sys.path.insert(0, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'verify', 'core'], nargs='?', default='plan')
    parser.add_argument('command', nargs='?', default='check', help='Core action: check, smoke, run')
    parser.add_argument('--execute', action='store_true', help='Permit experiment execution')
    args, forwarded = parser.parse_known_args()
    if args.stage in ('plan', 'verify') and forwarded:
        parser.error('Plan and verify do not accept extra stage arguments')
    if args.stage == 'plan':
        print(json.dumps({
            'protocol': 'KBS-R1-CORRECTED-20261111-v1',
            'scope': 'H1 and corrected H2/H3 only',
            'h1_entry_point': 'run_h1.py',
            'execution_order': ['core check', 'core smoke', 'core run'],
            'fixed_h2_fits': 108, 'fixed_h3_fits': 18,
            'training_seeds': [42, 123, 2024], 'random_masks_per_seed': 10,
            'frozen_inputs': 'data/splits/revision',
            'outputs': 'results/generated/revision/core',
            'requirements': 'Provider CSVs and CUDA for formal training'
        }, indent=2))
        return
    if args.stage == 'verify':
        from experiments.revision.common import verify
        verify(raw=False)
        print('PASS corrected H2/H3 frozen inputs; no training started')
        return
    if not args.execute:
        parser.error('Use --execute to run a stage; plan and verify are safe without it')
    sys.argv = [str(ROOT / 'run_revision.py'), args.command, '--execute', *forwarded]
    from experiments.revision.core import main as run_core
    run_core()


if __name__ == '__main__':
    main()
