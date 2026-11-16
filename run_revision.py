"""Current manuscript experiments: safe plan/verify or explicit staged execution."""
from pathlib import Path
import argparse
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parent
for path in (ROOT / 'src', ROOT):
    sys.path.insert(0, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'verify', 'core', 'comparisons', 'sensitivity',
                                         'partitions', 'calibration', 'matrix'], nargs='?', default='plan')
    parser.add_argument('command', nargs='?', default='check',
                        help='Stage action: check, smoke, run; sensitivity also supports self-test')
    parser.add_argument('--execute', action='store_true', help='Permit data preparation or stage execution')
    args, forwarded = parser.parse_known_args()
    if args.stage == 'plan':
        if forwarded:
            parser.error('Plan does not accept stage arguments')
        print(json.dumps({'protocol': 'KBS-R1-CORRECTED-20261111-v1',
            'execution_order': ['core check/smoke/run', 'comparisons check/smoke/run', 'sensitivity check/smoke/run'],
            'fixed_nominal_cells': 312, 'training_seeds': [42, 123, 2024], 'random_masks_per_seed': 10,
            'frozen_inputs': 'data/splits/revision', 'outputs': 'results/generated/revision',
            'requirements': 'Provider CSVs and CUDA for formal training; no training is started by plan or verify',
            'historical_gate_inputs': 'run_h3.py and run_h2.py regenerate the separate historical protocol'}, indent=2))
        return
    if args.stage == 'verify':
        if forwarded:
            parser.error('Verify does not accept stage arguments')
        from experiments.revision.common import verify
        verify(raw=False)
        print('PASS corrected revision frozen inputs; no training started')
        return
    if not args.execute:
        parser.error('Use --execute to run a stage; plan and verify are safe without it')
    modules = {'core': 'experiments.revision.core',
               'comparisons': 'experiments.revision.comparisons.run',
               'sensitivity': 'experiments.revision.sensitivity.run',
               'partitions': 'experiments.revision.partitions',
               'calibration': 'experiments.revision.calibration',
               'matrix': 'experiments.revision.matrix'}
    if args.stage in ('partitions', 'calibration', 'matrix'):
        sys.argv = [str(ROOT / 'run_revision.py'), *forwarded]
    else:
        sys.argv = [str(ROOT / 'run_revision.py'), args.command, '--execute', *forwarded]
    importlib.import_module(modules[args.stage]).main()


if __name__ == '__main__':
    main()
