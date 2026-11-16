"""End-to-end checks on an explicitly marked small fixture, never formal evidence."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from . import common, h2, h3

def check(out, source):
    fixture = out / 'evaluation_fixture.csv'
    raw = pd.read_csv(fixture)
    pipeline = joblib.load(out / 'RandomForest/model.joblib')
    kept = list(pipeline.named_steps['preprocessor'].feature_names_in_)
    with patch.object(h2, 'data_path', return_value=fixture), patch.dict(h2.TOTALS, {'fixture': len(raw)}):
        metrics = h2.evaluate(pipeline, kept, 'fixture', 257)
    predicted = pipeline.predict(common.canonicalize_chunk(raw)[kept].astype(np.float64))
    expected = h2.gap1.binary_metrics(raw['Label'].astype(int).to_numpy(), predicted)
    for key in ('tp', 'tn', 'fp', 'fn', 'mcc', 'binary_f1'):
        assert metrics[key] == expected[key]
    features = h2.shift_metrics.feature_shift_feature_columns()
    args = SimpleNamespace(quantile_sample_rows=1024, reservoir_seed=42, n_bins=50, bin_strategy='quantile', chunk_size=257)
    clean, audit = h2.shift_metrics.minimal_clean_features(raw[features], features)
    profile = h2.shift_metrics.SourceProfile(42, clean, h2.shift_metrics.compute_source_stats(clean, features),
                                  h2.shift_metrics.categorical_counts_from_frame(clean, h2.shift_metrics.categorical_jsd_features()), audit)
    target = h2.shift_metrics.load_target_profile('fixture', fixture, features, args)
    rows = h2.shift_metrics.build_metric_rows(profile, target, features, h2.shift_metrics.numeric_jsd_features(features), args)
    assert len(rows) == 49 and all(np.isfinite(r['jsd']) for r in rows)
    assert all(abs(r['jsd']) < 1e-12 for r in rows)  # identical distributions
    settings = h3.H3Settings(min_epochs=2, patience=0, embedding_sample_size=512)
    args = SimpleNamespace(output_root=out / 'integration', target='BoT-IoT', seed=42, execute=True,
                           policy=common.ROOT / 'protocols/revision/h3.json', chunk_size=257, resume=True, cpu=False)
    adapted = source['X_test']
    sample = (adapted, np.arange(len(adapted)), {'target_total_samples': len(raw), 'fixture_only': True})
    with patch.object(h3, 'h3_source', return_value=source), patch.object(h3, 'load_target', return_value=sample), \
         patch.object(h3, 'H3Settings', return_value=settings), patch.dict(h3.TARGET_DATASETS, {'BoT-IoT': str(fixture)}):
        h3.run_pair(args)
        # Repeat exercises the completed-unit hash validation path.
        h3.run_pair(args)
    return {'h2_chunked_counts_match_direct': True, 'jsd_full_stream_fixture': True, 'h3_complete_pair_and_reuse': True}
