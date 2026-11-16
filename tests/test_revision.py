"""Revision-specific numerical, isolation and portability regressions."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.revision import common, estimators, partitions

ROOT = Path(__file__).resolve().parents[1]


def test_alignment_gradients_and_independent_references():
    assert all(estimators.numerical_tests().values())


def test_mrmr_uses_complementary_information_and_places_constants_last():
    a = np.tile([0, 0, 1, 1], 128)
    b = np.tile([0, 1, 0, 1], 128)
    ranking, _, constants = estimators.rank_mrmr(
        np.column_stack([a, a, b, np.zeros(len(a))]), 2 * a + b,
        ['a', 'duplicate', 'b', 'constant'])
    assert ranking[:2] == [0, 2]
    assert ranking[-1] == 3 and constants == [3]


def test_group_partition_keeps_groups_indivisible():
    sizes = np.array([3, 11, 4, 19, 7, 8, 23, 6, 9, 12])
    attacks = sizes * (np.arange(len(sizes)) % 2)
    first = partitions.group_partition(sizes, attacks, 42)
    np.testing.assert_array_equal(first, partitions.group_partition(sizes, attacks, 42))
    assert set(first) == {0, 1, 2}
    assert len(first) == len(sizes)


def test_frozen_masks_and_input_integrity():
    from experiments.revision.matrix import mask
    common.verify(raw=False)
    bank = json.loads((common.PREP / 'random_masks.json').read_text())
    assert len(bank['masks']) == 120
    for entry in bank['masks']:
        assert mask(bank['features'], entry['seed'], entry['k'], entry['replicate']) == entry['removed']


def test_corrected_loto_isolation_and_missing_evidence(tmp_path):
    from experiments.revision.verification import verify_logic
    assert all(verify_logic(tmp_path).values())


def test_core_estimators_streamed_evaluation_and_pairing(tmp_path):
    import pandas as pd
    from experiments.revision import h2, integration_check, recovery_check
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    data = {'feature_cols': common.FEATURES,
            'membership_hashes': {'train': 'synthetic_train', 'val': 'synthetic_val', 'test': 'synthetic_test'}}
    for name, count in [('train', 512), ('val', 64), ('test', 512)]:
        frame = pd.DataFrame(rng.normal(size=(count, 49)), columns=common.FEATURES)
        frame['PROTOCOL'] = rng.integers(0, 3, count)
        frame['L7_PROTO'] = rng.integers(0, 4, count)
        data['X_' + name] = frame.astype(np.float64)
        data['y_' + name] = np.arange(count) % 2
        data[name + '_row_ids'] = np.arange(count)
    for model in ('RandomForest', 'MLP'):
        fitted, _, metadata = h2.fit(data, 42, model, [], tmp_path / model, smoke=True)
        assert len(fitted.predict(data['X_test'])) == len(data['y_test'])
        if model == 'MLP':
            assert metadata['validation_source'] == 'explicit_validation_data'
    pre = fitted.named_steps['preprocessor']
    fixture = data['X_test'].copy()
    fixture['Label'] = data['y_test']
    fixture.to_csv(tmp_path / 'evaluation_fixture.csv', index=False)
    source = dict(data, preprocessor=pre)
    for name in ('train', 'val', 'test'):
        source['X_' + name] = pre.transform(data['X_' + name]).astype(np.float32)
    assert all(integration_check.check(tmp_path, source).values())
    assert all(recovery_check.check(tmp_path / 'recovery').values())


def test_feature_selection_and_shared_fit_fixture(tmp_path):
    from experiments.revision.comparisons.test_features import run
    run(tmp_path)


def test_revision_evidence_arithmetic():
    from experiments.revision.evidence import verify
    assert verify()['checks'] >= 1000


@pytest.mark.parametrize('stage', ['core', 'comparisons', 'sensitivity', 'partitions'])
def test_execution_requires_explicit_flag(stage):
    run = subprocess.run([sys.executable, str(ROOT / 'run_revision.py'), stage, 'run'],
                         capture_output=True, text=True)
    assert run.returncode != 0 and '--execute' in run.stderr


def test_revision_cli_from_unrelated_directory(tmp_path):
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    for args in (['plan'], ['verify'], ['--help']):
        subprocess.run([sys.executable, str(ROOT / 'run_revision.py'), *args], cwd=tmp_path,
                       env=env, check=True, capture_output=True, text=True)


@pytest.mark.parametrize('path', ['results/reference/revision', 'data/splits/revision', '.'])
def test_revision_rejects_protected_destinations(path):
    with pytest.raises(ValueError):
        common.generated_output(ROOT / path)


def test_ft_optimizer_has_stable_parameter_order():
    from experiments.revision.comparisons.paired import FTDomainModel
    torch.manual_seed(42)
    model = FTDomainModel()
    order = {id(p): i for i, p in enumerate(model.parameters())}
    optimizer = model.optimizer()
    for group in optimizer.param_groups:
        indices = [order[id(p)] for p in group['params']]
        assert indices == sorted(indices)
