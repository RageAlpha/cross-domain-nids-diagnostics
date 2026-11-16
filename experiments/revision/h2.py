"""Corrected H2 fits, full-target evaluation and shared ten-mask controls."""
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from experiments.h1 import run_multiseed_training as gap1
from experiments.h2 import compute_feature_shift as shift_metrics
from experiments.h2.run_loto_filtering import (
    STRATEGIES, UNIVERSAL, TARGET_SPECIFIC, masks_for_seed, ranking_path,
    normalize_metrics, candidate_target_path, validate_development_cartesian,
)
from cross_domain_nids.protocol_utils import deterministic_seed, hash_file, hash_json, atomic_write_json, write_immutable_json
from experiments.revision.matrix import mask as random_mask
from .common import ROOT, PREP, PROTOCOL_ID, SEEDS, TARGETS, TOTALS, FEATURES, source_data, data_path, canonicalize_chunk, atomic_csv
from . import h2_analysis

CONDITIONS = dict(zip(STRATEGIES, ['full', 'shift3', 'shift5', 'shift10', 'shift15', 'dup_columns', 'corr098', 'shift5_corr098']))

def verify_receipt(path):
    receipt = json.loads(path.read_text())
    for name, digest in receipt['files'].items():
        if hash_file(path.parent / name) != digest:
            raise ValueError(f'Completed artifact changed: {path.parent / name}')
    return receipt

def receipt_for(path, files, **extra):
    atomic_write_json(path, {'protocol_id': PROTOCOL_ID, 'files': {p.name: hash_file(p) for p in files}, **extra})

def fit(data, seed, model, removed, unit, smoke=False):
    unit.mkdir(parents=True, exist_ok=True)
    kept = [name for name in data['feature_cols'] if name not in removed]
    if len(set(removed)) != len(removed) or not set(removed).issubset(data['feature_cols']) or not kept:
        raise ValueError('Invalid deletion mask')
    contract = {'protocol_id': PROTOCOL_ID, 'seed': seed, 'model': model, 'removed': removed,
                'kept': kept, 'source_membership': data['membership_hashes'], 'smoke_only': smoke}
    write_immutable_json(unit / 'fit_contract.json', contract)
    complete, artifact = unit / 'TRAINED.json', unit / 'model.joblib'
    if complete.exists():
        meta = verify_receipt(complete)
        if meta['contract_hash'] != hash_file(unit / 'fit_contract.json'):
            raise ValueError('Fit contract mismatch')
        return joblib.load(artifact), kept, meta
    from cross_domain_nids.model_factory import construct_estimator
    estimator = construct_estimator('E2', model, seed)
    if model == 'MLP':
        from .estimators import RevisionMLP
        estimator = RevisionMLP(**estimator.get_params(deep=False))
        estimator.checkpoint_dir = str(unit)
    if smoke:
        estimator.set_params(**({'n_estimators': 10, 'n_jobs': 2} if model == 'RandomForest' else {'max_epochs': 2, 'require_gpu': False}))
    started = time.perf_counter()
    pre = gap1.build_preprocessor_for_features(kept)
    train = pre.fit_transform(data['X_train'][kept].astype(np.float64))
    if model == 'MLP':
        val = pre.transform(data['X_val'][kept].astype(np.float64))
        estimator.fit(train, data['y_train'], validation_data=(val, data['y_val']))
        if estimator.validation_source_ != 'explicit_validation_data':
            raise ValueError('MLP did not use the external corrected validation split')
    else:
        estimator.fit(train, data['y_train'])
    pipeline = Pipeline([('preprocessor', pre), ('model', estimator)])
    tmp = artifact.with_suffix('.tmp')
    joblib.dump(pipeline, tmp)
    tmp.replace(artifact)
    if model == 'MLP':
        atomic_csv(pd.DataFrame(estimator.history_), unit / 'training_log.csv')
    meta = {'contract_hash': hash_file(unit / 'fit_contract.json'), 'train_time_s': time.perf_counter() - started,
            'epochs': getattr(estimator, 'n_epochs_', None), 'parameters': estimator.get_params(deep=False),
            'validation_source': getattr(estimator, 'validation_source_', 'not_used'), 'smoke_only': smoke}
    receipt_for(complete, [artifact, unit / 'fit_contract.json'] + ([unit / 'training_log.csv'] if model == 'MLP' else []), **meta)
    (unit / 'resume.pt').unlink(missing_ok=True)
    print(f'TRAINED {unit} ({meta["train_time_s"]:.1f}s)', flush=True)
    return pipeline, kept, meta

def evaluate(pipeline, kept, target, chunk_size):
    counts = dict(tp=0, tn=0, fp=0, fn=0)
    started = time.perf_counter()
    columns = kept + ['Label']
    for chunk in pd.read_csv(data_path(target), usecols=columns, dtype={c: np.float64 for c in columns}, chunksize=chunk_size):
        chunk = canonicalize_chunk(chunk)
        y = chunk['Label'].to_numpy(dtype=np.int64)
        if not np.isin(y, [0, 1]).all():
            raise ValueError('Invalid evaluation labels')
        prediction = pipeline.predict(chunk[kept].astype(np.float64))
        gap1.update_counts(counts, y, prediction)
    if sum(counts.values()) != TOTALS[target]:
        raise ValueError(f'Incomplete full-target evaluation: {target}')
    return {**gap1.binary_metrics_from_counts(counts), 'evaluation_time_s': time.perf_counter() - started,
            'evaluation_rows': sum(counts.values()), 'evaluation_scope': 'full_target'}

def evaluate_cached(args, data, seed, model, strategy, removed, targets, unit, run_id):
    rows, pending = [], []
    for target in targets:
        complete = unit / f'{target}.COMPLETE.json'
        if complete.exists():
            verify_receipt(complete)
            verify_receipt(unit / 'TRAINED.json')
            rows.append(json.loads((unit / f'{target}.json').read_text()))
        else:
            pending.append(target)
    if pending:
        pipeline, kept, meta = fit(data, seed, model, removed, unit)
        for target in pending:
            print(f'EVALUATING {run_id} -> {target}', flush=True)
            metrics = evaluate(pipeline, kept, target, args.chunk_size)
            row = gap1.make_result_row('H2_corrected', seed, model, target, strategy, metrics)
            row.update(protocol_id=PROTOCOL_ID, run_id=run_id, removed_features=json.dumps(removed),
                       mask_hash=hash_json(removed), n_removed_features=len(removed), train_time_s=meta['train_time_s'],
                       source_membership_hash=hash_json(data['membership_hashes']), artifact=str(unit / 'model.joblib'))
            path = unit / f'{target}.json'
            atomic_write_json(path, row)
            receipt_for(unit / f'{target}.COMPLETE.json', [path, unit / 'TRAINED.json', unit / 'model.joblib'])
            rows.append(row)
        del pipeline
        gc.collect()
    return normalize_metrics(pd.DataFrame(rows)).to_dict('records')

def shift(args):
    # Preserve the frozen feature-shift estimator and feature-order tie break.
    features = shift_metrics.feature_shift_feature_columns()
    args.quantile_sample_rows, args.reservoir_seed = 500000, 42
    args.n_bins, args.bin_strategy = 50, 'quantile'
    for target in TARGETS:
        pending = []
        for seed in SEEDS:
            path = ranking_path(args, target, seed)
            stamp = path.with_suffix('.COMPLETE.json')
            if stamp.exists():
                verify_receipt(stamp)
            else:
                pending.append(seed)
        if not pending:
            continue
        profile = shift_metrics.load_target_profile(target, data_path(target), features, args)
        for seed in pending:
            data = source_data(seed)
            clean, audit = shift_metrics.minimal_clean_features(data['X_train'], features)
            source = shift_metrics.SourceProfile(seed, clean, shift_metrics.compute_source_stats(clean, features),
                                       shift_metrics.categorical_counts_from_frame(clean, shift_metrics.categorical_jsd_features()), audit)
            frame = pd.DataFrame(shift_metrics.build_metric_rows(source, profile, features, shift_metrics.numeric_jsd_features(features), args))
            if len(frame) != 49 or not np.isfinite(frame['jsd']).all():
                raise ValueError('Invalid JSD ranking')
            frame['_order'] = frame['feature'].map({f: i for i, f in enumerate(features)})
            frame = frame.sort_values(['jsd', '_order'], ascending=[False, True]).drop(columns='_order')
            frame['rank'] = np.arange(1, 50)
            path = ranking_path(args, target, seed)
            atomic_csv(frame, path)
            receipt_for(path.with_suffix('.COMPLETE.json'), [path])
            del data, source, clean
            gc.collect()
        del profile
    merged = pd.concat([pd.read_csv(ranking_path(args, t, s)) for t in TARGETS for s in SEEDS], ignore_index=True)
    atomic_csv(merged, args.output_root / 'h2/shift/jsd_long.csv')
    atomic_csv(merged.groupby(['target_dataset', 'feature'])['jsd'].agg(['mean', 'std']).reset_index(),
               args.output_root / 'h2/shift/jsd_mean_std.csv')

def candidates(args):
    rows = []
    for target in TARGETS:
        for seed in SEEDS:
            verify_receipt(ranking_path(args, target, seed).with_suffix('.COMPLETE.json'))
    for seed in SEEDS:
        data = source_data(seed)
        masks = masks_for_seed(args, data, seed)
        mask_path = args.output_root / 'h2/masks' / f'seed{seed}.json'
        write_immutable_json(mask_path, masks)
        for model in ('RandomForest', 'MLP'):
            for strategy in STRATEGIES:
                scopes = [('all_targets', TARGETS)] if strategy in UNIVERSAL else [(t, [t]) for t in TARGETS]
                for scope, targets in scopes:
                    run_id = f'H2/{model}/{CONDITIONS[strategy]}/{scope}/{seed}'
                    unit = args.output_root / 'fits' / run_id
                    rows.extend(evaluate_cached(args, data, seed, model, strategy, masks[targets[0]][strategy], targets, unit, run_id))
        del data
        gc.collect()
    frame = pd.DataFrame(rows)
    for target in TARGETS:
        group = frame[frame['target_dataset'] == target]
        for model in ('RandomForest', 'MLP'):
            validate_development_cartesian(group[group['model'] == model], [target], model)
        atomic_csv(group, candidate_target_path(args, target))
    atomic_csv(frame, args.output_root / 'h2/analysis/all_candidates_long.csv')

def random_controls(args):
    policy = json.loads(args.policy.read_text())
    policy['_path'] = str(args.policy)
    requests = {}
    for target in TARGETS:
        for model in ('RandomForest', 'MLP'):
            path = args.output_root / 'h2/selection_locks' / f'{target}_{model}.json'
            lock = json.loads(path.read_text())
            h2_analysis.validate_selection_lock(lock, path, policy, target, model)
            if lock['selected_strategy'] == 'none':
                continue
            for seed in SEEDS:
                k = len(lock['held_out_removed_features_by_seed'][str(seed)])
                requests.setdefault((seed, model, k), []).append((target, hash_file(path)))
    registry = []
    for seed in SEEDS:
        groups = [(model, k, targets) for (s, model, k), targets in requests.items() if s == seed]
        if not groups:
            continue
        data = source_data(seed)
        for model, k, target_locks in groups:
            for replicate in range(10):
                removed = random_mask(FEATURES, seed, k, replicate)
                fixed = model == 'MLP' and k in (3, 5, 10, 15)
                run_id = (f'random_size/MLP/k{k}_rep{replicate:02}/ToN-IoT/{seed}' if fixed else
                          f'random_conditional/{model}/k{k}_rep{replicate:02}/all_targets/{seed}')
                targets = [t for t, _ in target_locks]
                # A predeclared Phase 4 cell must also retain its ToN evaluation.
                if fixed and 'ToN-IoT' not in targets:
                    targets.append('ToN-IoT')
                rows = evaluate_cached(args, data, seed, model, 'matched_random', removed, targets,
                                       args.output_root / 'fits' / run_id, run_id)
                for target, lock_hash in target_locks:
                    row = next(dict(r) for r in rows if r['target_dataset'] == target)
                    row.update(replicate=replicate, random_seed=deterministic_seed(PROTOCOL_ID, 'random_mask', seed, k, replicate),
                               selected_mask_size=k, lock_hash=lock_hash)
                    path = args.output_root / 'h2/random/batches' / target / model / f'seed{seed}' / f'rep{replicate:02}' / 'random_metrics.csv'
                    atomic_csv(pd.DataFrame([row]), path)
                registry.append({'run_id': run_id, 'model': model, 'seed': seed, 'k': k, 'replicate': replicate,
                                 'evaluate_targets': '|'.join(targets), 'predeclared_phase4_reuse': fixed})
        del data
        gc.collect()
    atomic_write_json(args.output_root / 'h2/random/run_registry.json', {'protocol_id': PROTOCOL_ID, 'fits': registry,
                      'unique_fits': len(registry), 'additional_conditional_fits': sum(not r['predeclared_phase4_reuse'] for r in registry)})
