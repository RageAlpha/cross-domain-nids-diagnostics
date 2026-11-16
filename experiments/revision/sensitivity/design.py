"""Immutable Phase 4 scope, identities and shared validators."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.revision import common

ALPHAS = (.1, .5)
SIZES = (3, 5, 10, 15)
TOLERANCES = (0., .01, .03, .05)
RULES = ('positive_mean', 'mean_and_two', 'all_three')
POLICY = {
    'protocol_id': common.PROTOCOL_ID, 'seeds': common.SEEDS,
    'alphas_new': list(ALPHAS), 'alphas_reused': [0., 1.],
    'random_sizes': list(SIZES), 'random_model': 'MLP', 'random_target': 'ToN-IoT',
    'random_replicates': 10, 'fnr_tolerances': list(TOLERANCES), 'seed_rules': list(RULES),
    'selection_replayed': False, 'representation_gate_retained': True,
    'readout': {'sample_seed': 314159, 'split_seed': 271828, 'sample_rows': 20000,
                'train_fraction': .7, 'scaler_fit': 'readout_train_only', 'C': 1.,
                'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 1000,
                'scope': 'ToN-IoT Phase2 C0/C1, three training seeds',
                'label_use': 'target-supervised post-hoc diagnostic; never model selection'},
    'multi_source': False,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def matrix():
    table = pd.read_csv(common.PREP / 'formal_matrix.csv', keep_default_na=False)
    table = table[table.phase == 4].copy()
    expected = {f'strength/DANN/alpha{alpha:g}/{target}/{seed}'
                for alpha in ALPHAS for target in common.TARGETS for seed in common.SEEDS}
    expected |= {f'random_size/MLP/k{k}_rep{rep:02}/ToN-IoT/{seed}'
                 for k in SIZES for rep in range(10) for seed in common.SEEDS}
    if len(table) != 138 or set(table.run_id) != expected or table.run_id.duplicated().any():
        raise ValueError('Phase 4 must have exactly the frozen 18 + 120 run IDs')
    for row in table.itertuples():
        seed = int(row.training_seed)
        if row.package == 'strength':
            alpha = float(row.strength_max)
            valid = (row.model == 'DANN' and row.condition == f'alpha{alpha:g}'
                     and row.evaluate_targets in common.TARGETS and row.source_split_seed == 42
                     and row.zero_control_id == f'H3/DANN/zero/{row.evaluate_targets}/{seed}'
                     and alpha in ALPHAS)
        else:
            valid = (row.model == 'MLP' and row.evaluate_targets == 'ToN-IoT'
                     and row.source_split_seed == seed and row.zero_control_id == '')
        if not valid:
            raise ValueError(f'Changed Phase 4 matrix row: {row.run_id}')
    return table


def three_seeds(frame):
    if len(frame) != 3 or set(frame.seed) != set(common.SEEDS) or frame.seed.duplicated().any():
        raise ValueError('Exactly three unique frozen seeds are required')


def finite(frame, columns):
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f'Nonfinite or missing metrics: {columns}')


def independent_output(output, inputs):
    output = common.generated_output(output)
    results = (common.ROOT / 'results').resolve()
    if output == results or not output.is_relative_to(results):
        raise ValueError('Use an independent output directory under results')
    for source in inputs:
        source = Path(source).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f'Output overlaps protected input: {source}')
    return output
