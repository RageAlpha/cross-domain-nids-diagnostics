"""Check interrupted-epoch recovery against uninterrupted short training."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import h3, estimators as mlp

class SimulatedInterruption(RuntimeError):
    pass

def compare_states(left, right):
    assert set(left) == set(right)
    for name in left:
        torch.testing.assert_close(left[name].cpu(), right[name].cpu(), rtol=0, atol=0)

def check(root):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(783)
    x, v, target = (rng.normal(size=(n, 49)).astype(np.float32) for n in (256, 64, 256))
    y, vy = (np.arange(n) % 2 for n in (256, 64))
    data = dict(X_train=x, y_train=y, X_val=v, y_val=vy, membership_hashes={'fixture': 'recovery-only'})
    h3.set_all_seeds(42)
    initial = {k: value.clone() for k, value in h3.DANN(49, dropout=0, grl_alpha=1).state_dict().items()}
    settings = h3.H3Settings()
    for variant in ('H3-C0', 'H3-C1'):
        baseline, _, logs, meta = h3.train_variant(data, target, initial, settings, variant, 42, root / variant / 'baseline',
                                                  cpu=False, resume=True, required_epochs=2)
        original_save = h3.save_training_checkpoint
        def interrupt(path, **payload):
            original_save(path, **payload)
            raise SimulatedInterruption('After the first committed epoch')
        interrupted_path = root / variant / 'resumed'
        if not (interrupted_path / 'TRAINED.json').exists():
            with patch.object(h3, 'save_training_checkpoint', interrupt):
                try:
                    h3.train_variant(data, target, initial, settings, variant, 42, interrupted_path,
                                     cpu=False, resume=True, required_epochs=2)
                except SimulatedInterruption:
                    pass
                else:
                    raise AssertionError('Recovery test did not interrupt')
        resumed, _, resumed_logs, resumed_meta = h3.train_variant(data, target, initial, settings, variant, 42, interrupted_path,
                                                               cpu=False, resume=True, required_epochs=2)
        compare_states(baseline.state_dict(), resumed.state_dict())
        assert logs == resumed_logs and meta == resumed_meta
        compare_states(torch.load(root / variant / 'baseline/final.pt', weights_only=True),
                       torch.load(interrupted_path / 'final.pt', weights_only=True))
    def estimator(folder):
        folder.mkdir(parents=True, exist_ok=True)
        model = mlp.RevisionMLP(max_epochs=2, batch_size=128, random_state=42, require_gpu=False)
        model.checkpoint_dir = str(folder)
        return model
    baseline = estimator(root / 'MLP/baseline').fit(x, y, validation_data=(v, vy))
    resumed = estimator(root / 'MLP/resumed')
    original_save = mlp.save_training_checkpoint
    def interrupt_mlp(path, **payload):
        original_save(path, **payload)
        raise SimulatedInterruption('After the first committed MLP epoch')
    if not (Path(resumed.checkpoint_dir) / 'resume.pt').exists():
        with patch.object(mlp, 'save_training_checkpoint', interrupt_mlp):
            try:
                resumed.fit(x, y, validation_data=(v, vy))
            except SimulatedInterruption:
                pass
            else:
                raise AssertionError('MLP recovery test did not interrupt')
    resumed.fit(x, y, validation_data=(v, vy))
    compare_states(baseline.model_.state_dict(), resumed.model_.state_dict())
    assert baseline.history_ == resumed.history_
    return {'h3_epoch_recovery_matches_uninterrupted': True, 'mlp_epoch_recovery_matches_uninterrupted': True}
