"""Short real-data checks, isolated from formal model/evaluation outputs."""
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from . import common, h2, h3

def run(args):
    import torch
    started = time.perf_counter()
    # A failed smoke check never leaves a stale PASS marker.
    out = args.output_root / 'smoke'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'SMOKE_PASS.json').unlink(missing_ok=True)
    data = common.source_data(42, smoke=True)
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not np.intersect1d(data[f'{a}_row_ids'], data[f'{b}_row_ids']).size
    checks = {'disjoint_frozen_source_membership': True}
    for model in ('RandomForest', 'MLP'):
        unit = out / model
        pipeline, kept, meta = h2.fit(data, 42, model, [], unit, smoke=True)
        prediction = pipeline.predict(data['X_test'][kept])
        loaded = joblib.load(unit / 'model.joblib')
        np.testing.assert_array_equal(prediction, loaded.predict(data['X_test'][kept]))
        assert set(np.unique(prediction)).issubset({0, 1})
        if model == 'MLP':
            assert meta['validation_source'] == 'explicit_validation_data'
            assert pipeline.named_steps['model'].n_epochs_ == 2
        checks[f'{model}_fit_and_reload'] = True
        del pipeline, loaded
    raw_test = data['X_test'].copy()
    raw_test['Label'] = data['y_test']
    raw_test.to_csv(out / 'evaluation_fixture.csv', index=False)
    del data
    gc.collect()
    source = common.h3_source(smoke=True)
    target, target_ids, metadata = common.load_target('BoT-IoT', 42, source['preprocessor'], smoke=True)
    assert metadata['target_detection_labels_read'] is False
    assert len(target_ids) == 8192
    checks['frozen_label_free_adaptation_sample'] = True
    h3.set_all_seeds(42)
    initial = h3.DANN(49, dropout=0.0, grl_alpha=1.0).state_dict()
    initial = {name: tensor.detach().cpu().clone() for name, tensor in initial.items()}
    settings = h3.H3Settings()
    execution, reservoirs = {}, {}
    for variant in ('H3-C0', 'H3-C1'):
        model, device, logs, meta = h3.train_variant(source, target, initial, settings, variant, 42,
                                                    out / variant, cpu=False, resume=True, required_epochs=2)
        assert meta['epochs_completed'] == 2
        assert all(np.isfinite(row['source_val_loss']) for row in logs)
        assert all(row['protocol_id'] == common.PROTOCOL_ID for row in logs)
        assert logs[1]['lambda'] == (0 if variant == 'H3-C0' else 1 / 39)
        (out / 'predictions').mkdir(exist_ok=True)
        metrics, embedding, labels, ids, reservoir = h3.evaluate_full_target_reservoir(
            torch, model, device, source['preprocessor'], source['feature_cols'], 'SMOKE_FIXTURE',
            out / 'evaluation_fixture.csv', 42, variant, out, 257, 256)
        h3.validate_confusion_counts(metrics, len(raw_test))
        assert reservoir['stream_rows_seen'] == len(raw_test)
        assert len(embedding) == 256
        reservoirs[variant] = ids
        execution[variant] = meta
        clone = h3.DANN(49, dropout=0.0, grl_alpha=1.0).to(device)
        clone.load_state_dict(torch.load(out / variant / 'best.pt', map_location=device, weights_only=True))
        clone.eval()
        np.testing.assert_allclose(h3.predict_logits_numpy(torch, model, source['X_test'], device)[0],
                                   h3.predict_logits_numpy(torch, clone, source['X_test'], device)[0], rtol=0, atol=0)
    for key in ('source_actual_loader_order_hash', 'target_actual_loader_order_hash', 'epochs_completed'):
        assert execution['H3-C0'][key] == execution['H3-C1'][key]
    np.testing.assert_array_equal(reservoirs['H3-C0'], reservoirs['H3-C1'])
    checks.update(h3_matched_training=True, h3_schedule=True, full_stream_evaluation_fixture=True,
                  h3_checkpoint_reload=True, paired_reservoir_ids=True)
    bank = json.loads((common.PREP / 'random_masks.json').read_text())
    for entry in bank['masks']:
        assert h2.random_mask(bank['features'], entry['seed'], entry['k'], entry['replicate']) == entry['removed']
    checks['all_120_frozen_masks_match'] = True
    from .verification import verify_logic
    checks.update(verify_logic(out / 'logic'))
    from .recovery_check import check
    checks.update(check(out / 'recovery'))
    from .integration_check import check as integration_check
    checks.update(integration_check(out, source))
    common.atomic_write_json(out / 'SMOKE_PASS.json', {'status': 'PASS', 'smoke_only': True,
                             'input_lock_hash': common.hash_file(args.output_root / 'INPUT_LOCK.json'),
                             'environment_hash': common.hash_file(args.output_root / 'environment.json'),
                             'checks': checks, 'seconds': time.perf_counter() - started,
                             'formal_performance_claim': False})
    print(f'SMOKE PASS: {len(checks)} checks. No formal experimental conclusion. {out}', flush=True)
