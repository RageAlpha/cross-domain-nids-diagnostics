"""Data access for the approved corrected partitions. Never regenerate membership."""
from pathlib import Path
import gc
import json
import os
import platform
import sys
import joblib
import numpy as np
import pandas as pd
from cross_domain_nids.protocol_utils import deterministic_seed, hash_file, hash_array, hash_json, atomic_write_json, write_immutable_json
from experiments.h1.run_multiseed_training import feature_columns_from_frame, build_preprocessor_for_features

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / 'data/splits/revision'
PROTOCOL_ID = 'KBS-R1-CORRECTED-20261111-v1'
SEEDS = [42, 123, 2024]
TARGETS = ['CICIDS2018', 'BoT-IoT', 'ToN-IoT']
TOTALS = {'CICIDS2018': 20115529, 'BoT-IoT': 16933808, 'ToN-IoT': 27520260}
FEATURES = json.loads((PREP / 'random_masks.json').read_text())['features']

STRING_COLUMNS = {'IPV4_SRC_ADDR', 'IPV4_DST_ADDR'}

def canonicalize_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply cross-file deterministic cleaning without using target labels."""
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if "L7_PROTO" in frame.columns:
        frame["L7_PROTO"] = frame["L7_PROTO"].fillna(0.0).round()
    for column in STRING_COLUMNS.intersection(frame.columns):
        frame[column] = frame[column].astype("string").fillna("<NA>").astype(str)
    return frame

def random_mask(features, seed, k, replicate):
    generator = np.random.default_rng(deterministic_seed(PROTOCOL_ID, "random_mask", seed, k, replicate))
    return sorted(generator.choice(features, k, replace=False).tolist())


def data_path(name):
    return ROOT / 'data/raw' / f'NF-{name}-v3.csv'

def read_rows(path, ids, columns, chunk_size=250000):
    """Read selected original record IDs, preserving their specified order."""
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) != len(np.unique(ids)) or np.any(ids < 0):
        raise ValueError('Duplicate or negative record IDs')
    ordered = np.sort(ids)
    frames, offset = [], 0
    for chunk in pd.read_csv(path, usecols=columns, dtype={c: np.float64 for c in columns}, chunksize=chunk_size):
        end = offset + len(chunk)
        lo, hi = np.searchsorted(ordered, [offset, end])
        if hi > lo:
            frame = chunk.iloc[ordered[lo:hi] - offset].copy()
            frame.index = ordered[lo:hi]
            frames.append(canonicalize_chunk(frame))
        offset = end
        if len(ordered) and offset > ordered[-1]:
            break
    if not frames or sum(map(len, frames)) != len(ids):
        raise ValueError(f'Missing requested records: {path}')
    return pd.concat(frames).loc[ids, columns].astype(np.float64)

def source_data(seed, smoke=False):
    with np.load(PREP / f'source_seed{seed}.npz') as item:
        ids, split = item['row_ids'], item['split']
    selected = {}
    for code, name in enumerate(('train', 'val', 'test')):
        members = ids[split == code]
        if smoke:
            members = np.sort(np.random.default_rng(9000 + code).choice(members, [8192, 2048, 2048][code], replace=False))
        selected[name] = members
    all_ids = np.concatenate(list(selected.values()))
    raw = read_rows(data_path('UNSW-NB15'), all_ids, FEATURES + ['Label'])
    features = feature_columns_from_frame(raw)
    data = {'feature_cols': features, 'source_seed': seed, 'smoke_only': smoke}
    for name, members in selected.items():
        data[f'X_{name}'] = raw.loc[members, features].copy()
        data[f'y_{name}'] = raw.loc[members, 'Label'].astype(np.int64).to_numpy()
        data[f'{name}_row_ids'] = members
        if not np.isin(data[f'y_{name}'], [0, 1]).all():
            raise ValueError('Invalid source labels')
    data['membership_hashes'] = {name: hash_array(members) for name, members in selected.items()}
    return data

def h3_source(smoke=False):
    data = source_data(42, smoke)
    # All learned preprocessing quantities come from the frozen source training pool.
    # Reconstructing it avoids distributing version-sensitive serialized objects.
    preprocessor = build_preprocessor_for_features(FEATURES).fit(data['X_train'][FEATURES])
    for name in ('train', 'val', 'test'):
        data[f'X_{name}'] = preprocessor.transform(data[f'X_{name}'][FEATURES]).astype(np.float32)
    data['feature_cols'] = FEATURES
    data['preprocessor'] = preprocessor
    return data

def load_target(target, seed, preprocessor, smoke=False):
    with np.load(PREP / f'{target}_adaptation_seed{seed}.npz') as item:
        ids = item['row_ids']
    if len(ids) != int(TOTALS[target] * .05):
        raise ValueError('Adaptation membership size differs from frozen design')
    if smoke:
        ids = np.sort(np.random.default_rng(8000).choice(ids, 8192, replace=False))
    # Label and Attack are deliberately absent from usecols.
    raw = read_rows(data_path(target), ids, FEATURES)
    transformed = preprocessor.transform(raw[FEATURES]).astype(np.float32)
    return transformed, ids, {'target_total_samples': TOTALS[target], 'adaptation_rows': len(ids),
                             'target_detection_labels_read': False, 'adaptation_row_ids_hash': hash_array(ids),
                             'smoke_only': smoke}

def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)

def generated_output(path):
    from cross_domain_nids.release_paths import require_generated_output
    return require_generated_output(path, 'results')


def code_fingerprint():
    paths = [ROOT / 'run_revision.py', ROOT / 'data/model_hyperparameters.json']
    for folder in ('experiments', 'src'):
        paths.extend(sorted((ROOT / folder).rglob('*.py')))
    paths.extend(sorted((ROOT / 'protocols').rglob('*.json')))
    return {p.relative_to(ROOT).as_posix(): hash_file(p) for p in paths}


def verify(root=ROOT, raw=True):
    """Verify the curated immutable inputs; raw-data checks are execution-only."""
    prep = root / 'data/splits/revision'
    manifest = json.loads((prep / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol_id'] != PROTOCOL_ID:
        raise ValueError('Unexpected corrected protocol')
    for name, digest in manifest['artifacts'].items():
        path = (prep / name).resolve()
        if not path.is_relative_to(prep.resolve()) or hash_file(path) != digest:
            raise ValueError(f'Prepared artifact changed: {name}')
    if raw:
        for item in manifest['data_files']:
            path = root / item['path']
            if not path.is_file():
                raise FileNotFoundError(f'Download the provider dataset first: {path}')
            if hash_file(path) != item['sha256']:
                raise ValueError(f'Raw dataset identity differs: {path.name}')
    for name, column, count in [('source_partition_overlap.csv', 'shared_keys', 27),
                               ('source_target_overlap.csv', 'overlap_rows', 36)]:
        table = pd.read_csv(prep / name)
        if len(table) != count or table[column].ne(0).any():
            raise ValueError(f'Partition audit differs: {name}')
    matrix = pd.read_csv(prep / 'formal_matrix.csv')
    if len(matrix) != 126 or matrix.run_id.nunique() != 126 or matrix.package.value_counts().to_dict() != {'H2': 108, 'H3': 18}:
        raise ValueError('Frozen main-experiment matrix requires 108 H2 fits and 18 H3 fits')


def verify_and_lock(out):
    """Verify raw/frozen inputs and freeze the public code identity for this run."""
    out = generated_output(out)
    verify(ROOT)
    lock = {'protocol_id': PROTOCOL_ID, 'phase1_manifest_hash': hash_file(PREP / 'manifest.json'),
            'code_and_templates': code_fingerprint()}
    out.mkdir(parents=True, exist_ok=True)
    write_immutable_json(out / 'INPUT_LOCK.json', lock)
    import torch
    import sklearn
    env = {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
           'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'torch': torch.__version__,
           'cuda_available': torch.cuda.is_available(),
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    atomic_write_json(out / 'environment.json', env)
    return lock
