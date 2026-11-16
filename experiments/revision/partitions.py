"""Prepare corrected source membership and audit it against full target files.

This data-only entry point does not train models or read target detection labels.
It reuses the project's canonicalization, double fingerprints and lookup code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataclasses import dataclass
from typing import Iterable
from pandas.util import hash_pandas_object
import hashlib
import numpy as np
import pandas as pd
FLOW_COLUMNS = ['FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR',
                'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'PROTOCOL']
STRING_COLUMNS = {'IPV4_SRC_ADDR', 'IPV4_DST_ADDR'}
KEY_DTYPE = np.dtype([('h1', '<u8'), ('h2', '<u8')])
KEY_DTYPE = np.dtype([("h1", "<u8"), ("h2", "<u8")])
@dataclass
class SourceIndex:
    unique_keys: np.ndarray
    inverse: np.ndarray
    plan_membership: np.ndarray


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def hash_indices(indices: np.ndarray) -> str:
    values = np.sort(np.asarray(indices, dtype="<i8"))
    return sha256_bytes(values.tobytes(order="C"))


def structured_keys(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    if len(h1) != len(h2):
        raise ValueError("Fingerprint words have different lengths")
    keys = np.empty(len(h1), dtype=KEY_DTYPE)
    keys["h1"] = np.asarray(h1, dtype=np.uint64)
    keys["h2"] = np.asarray(h2, dtype=np.uint64)
    return keys


def double_hash(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return two order-sensitive 64-bit row fingerprints."""
    h1 = hash_pandas_object(
        frame, index=False, categorize=True, hash_key="0123456789abcdef"
    ).to_numpy(dtype=np.uint64, copy=False)
    reversed_frame = frame.loc[:, list(reversed(frame.columns))]
    h2 = hash_pandas_object(
        reversed_frame, index=False, categorize=True, hash_key="fedcba9876543210"
    ).to_numpy(dtype=np.uint64, copy=False)
    return h1, h2


def canonicalize_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply cross-file deterministic cleaning without using target labels."""
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if "L7_PROTO" in frame.columns:
        frame["L7_PROTO"] = frame["L7_PROTO"].fillna(0.0).round()
    for column in STRING_COLUMNS.intersection(frame.columns):
        frame[column] = frame[column].astype("string").fillna("<NA>").astype(str)
    return frame


def frame_dtypes(columns: Iterable[str]) -> dict[str, object]:
    """Force equivalent CSV numbers to the same dtype across all datasets."""
    return {
        column: (str if column in STRING_COLUMNS else np.float64)
        for column in columns
    }


def lookup_plan_masks(index: SourceIndex, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row plan membership and matched unique-source positions."""
    positions = np.searchsorted(index.unique_keys, keys)
    in_bounds = positions < len(index.unique_keys)
    candidate_rows = np.flatnonzero(in_bounds)
    if len(candidate_rows):
        candidate_positions = positions[candidate_rows]
        exact = index.unique_keys[candidate_positions] == keys[candidate_rows]
        matched_rows = candidate_rows[exact]
        matched_positions = candidate_positions[exact]
    else:
        matched_rows = np.empty(0, dtype=np.int64)
        matched_positions = np.empty(0, dtype=np.int64)
    masks = np.zeros(len(keys), dtype=np.uint8)
    masks[matched_rows] = index.plan_membership[matched_positions]
    return masks, matched_positions

SEEDS = (42, 123, 2024)
TARGET_ROWS = {'CICIDS2018': 20115529, 'BoT-IoT': 16933808, 'ToN-IoT': 27520260}
CRITERIA = ('exact_parsed_nonlabel_row', 'normalized_nonlabel_row', 'timestamped_session')
PROTOCOL = 'KBS-R1-CORRECTED-20261111-v1'


def fingerprints(frame):
    return structured_keys(*double_hash(frame))


def group_partition(group_rows, group_attacks, seed):
    """Shuffle groups within pure-benign/pure-attack/mixed strata once.

    Select group boundaries closest to 70% and 80% cumulative row counts.
    Each group is indivisible. Ties choose the earlier boundary.
    """
    strata = np.where(group_attacks == 0, 0, np.where(group_attacks == group_rows, 1, 2))
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(group_rows), dtype=np.uint8)
    for stratum in (0, 1, 2):
        ids = rng.permutation(np.flatnonzero(strata == stratum))
        cumulative = np.r_[0, np.cumsum(group_rows[ids])]
        boundaries = [int(np.argmin(np.abs(cumulative - ratio * cumulative[-1]))) for ratio in (0.7, 0.8)]
        assignment[ids[:boundaries[0]]] = 0
        assignment[ids[boundaries[0]:boundaries[1]]] = 1
        assignment[ids[boundaries[1]:]] = 2
    return assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/generated/revision_preparation')
    parser.add_argument('--chunk-size', type=int, default=250000)
    args = parser.parse_args()
    from cross_domain_nids.release_paths import require_generated_output
    out = require_generated_output(args.output_dir, 'results')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('Completed output exists. Use a new output directory for another version.')
    started = time.perf_counter()
    input_manifest = __import__('csv').DictReader((ROOT / 'data/dataset_checksums.csv').open(encoding='utf-8-sig'))
    verified = []
    for item in input_manifest:
        path = ROOT / 'data/raw' / Path(item['filename']).name
        actual = sha256_file(path)
        if actual != item['sha256']:
            raise RuntimeError(f'Input hash differs from audited input: {path}')
        verified.append({'path': str(path.relative_to(ROOT)), 'sha256': actual, 'bytes': path.stat().st_size})

    source_path = ROOT / 'data/raw/NF-UNSW-NB15-v3.csv'
    columns = pd.read_csv(source_path, nrows=0).columns.tolist()
    nonlabel = [name for name in columns if name not in ('Label', 'Attack')]
    dtypes = frame_dtypes(columns)
    dtypes['Attack'] = str
    source_size = 2365424
    keys = {name: np.empty(source_size, dtype=KEY_DTYPE) for name in CRITERIA}
    complete_keys = np.empty(source_size, dtype=KEY_DTYPE)
    labels = np.empty(source_size, dtype=np.int8)
    offset = 0
    missing_groups = 0
    for chunk in pd.read_csv(source_path, dtype=dtypes, chunksize=args.chunk_size):
        end = offset + len(chunk)
        missing_groups += int(chunk[FLOW_COLUMNS].isna().any(axis=1).sum())
        if not chunk['Label'].isin([0, 1]).all():
            raise ValueError('Invalid source binary labels')
        keys[CRITERIA[0]][offset:end] = fingerprints(chunk[nonlabel])
        clean = canonicalize_chunk(chunk)
        keys[CRITERIA[1]][offset:end] = fingerprints(clean[nonlabel])
        keys[CRITERIA[2]][offset:end] = fingerprints(clean[FLOW_COLUMNS])
        complete_keys[offset:end] = fingerprints(clean[columns])
        labels[offset:end] = clean['Label'].to_numpy(dtype=np.int8)
        offset = end
    if offset != source_size or missing_groups:
        raise ValueError(f'Source row count/group-field issue: rows={offset}, missing={missing_groups}')
    _, first, inverse_raw = np.unique(complete_keys, return_index=True, return_inverse=True)
    representatives = first[inverse_raw]
    retained = np.sort(first)
    retained_labels = labels[retained]
    group_keys, group_inverse = np.unique(keys[CRITERIA[2]][retained], return_inverse=True)
    group_rows = np.bincount(group_inverse)
    group_attacks = np.bincount(group_inverse, weights=retained_labels).astype(np.int64)
    np.savez_compressed(out / 'source_deduplication.npz', retained_row_ids=retained,
                        representative_row_ids=representatives, group_keys=group_keys, group_index=group_inverse)
    summaries, overlaps, source_splits = [], [], {}
    for seed in SEEDS:
        split = group_partition(group_rows, group_attacks, seed)[group_inverse]
        source_splits[seed] = split
        np.savez_compressed(out / f'source_seed{seed}.npz', row_ids=retained, split=split)
        for split_id, name in enumerate(('train', 'validation', 'test')):
            selected = split == split_id
            counts = np.bincount(retained_labels[selected], minlength=2)
            summaries.append({'seed': seed, 'split': name, 'rows': int(selected.sum()),
                              'benign': int(counts[0]), 'attack': int(counts[1]),
                              'attack_fraction': float(counts[1] / selected.sum()),
                              'member_index_sha256': hash_indices(retained[selected])})
        for criterion in CRITERIA:
            unique, inverse = np.unique(keys[criterion][retained], return_inverse=True)
            membership = np.zeros(len(unique), dtype=np.uint8)
            np.bitwise_or.at(membership, inverse, np.left_shift(np.uint8(1), split))
            for a, b in ((0, 1), (0, 2), (1, 2)):
                shared = (membership & (1 << a) != 0) & (membership & (1 << b) != 0)
                overlaps.append({'seed': seed, 'criterion': criterion, 'split_a': a, 'split_b': b,
                                 'shared_keys': int(shared.sum()),
                                 'rows_a': int(np.sum(shared[inverse] & (split == a))),
                                 'rows_b': int(np.sum(shared[inverse] & (split == b)))})
    pd.DataFrame(summaries).to_csv(out / 'source_split_summary.csv', index=False)
    pd.DataFrame(overlaps).to_csv(out / 'source_partition_overlap.csv', index=False)
    if any(row['shared_keys'] for row in overlaps):
        raise RuntimeError('Corrected source partitions still overlap under an audited direct key.')
    print(f'Source ready: retained={len(retained)}, removed={source_size-len(retained)}, groups={len(group_keys)}', flush=True)

    indices = {}
    for criterion in CRITERIA:
        unique, inverse = np.unique(keys[criterion][retained], return_inverse=True)
        membership = np.zeros(len(unique), dtype=np.uint8)
        for i, seed in enumerate(SEEDS):
            np.bitwise_or.at(membership, inverse[source_splits[seed] == 0], np.uint8(1 << i))
        indices[criterion] = SourceIndex(unique, inverse, membership)
        positive_ids = np.flatnonzero(source_splits[42] == 0)[:1000]
        masks, _ = lookup_plan_masks(indices[criterion], keys[criterion][retained[positive_ids]])
        if not np.all(masks & 1):
            raise RuntimeError('Positive-control source self-lookup failed.')

    target_overlaps, target_metadata = [], []
    for target, total in TARGET_ROWS.items():
        target_start = time.perf_counter()
        path = ROOT / 'data/raw' / f'NF-{target}-v3.csv'
        if pd.read_csv(path, nrows=0).columns.tolist() != columns:
            raise RuntimeError(f'Target schema mismatch: {target}')
        for seed in SEEDS:
            sample = np.sort(np.random.default_rng(seed).choice(total, int(total * .05), replace=False))
            np.savez_compressed(out / f'{target}_adaptation_seed{seed}.npz', row_ids=sample)
        counts = {name: np.zeros(3, dtype=np.int64) for name in (*CRITERIA, 'union_direct')}
        offset = 0
        missing_groups = 0
        for chunk in pd.read_csv(path, usecols=nonlabel, dtype=frame_dtypes(nonlabel), chunksize=args.chunk_size):
            missing_groups += int(chunk[FLOW_COLUMNS].isna().any(axis=1).sum())
            exact = fingerprints(chunk[nonlabel])
            clean = canonicalize_chunk(chunk)
            chunk_keys = {CRITERIA[0]: exact, CRITERIA[1]: fingerprints(clean[nonlabel]),
                          CRITERIA[2]: fingerprints(clean[FLOW_COLUMNS])}
            union = np.zeros(len(chunk), dtype=np.uint8)
            for criterion, values in chunk_keys.items():
                membership, _ = lookup_plan_masks(indices[criterion], values)
                union |= membership
                for i in range(3):
                    counts[criterion][i] += np.count_nonzero(membership & (1 << i))
            for i in range(3):
                counts['union_direct'][i] += np.count_nonzero(union & (1 << i))
            offset += len(chunk)
            if offset % 5000000 == 0:
                print(f'{target}: {offset}/{total} rows audited', flush=True)
        if offset != total:
            raise RuntimeError(f'Target row count mismatch: {target}')
        for criterion, values in counts.items():
            for i, seed in enumerate(SEEDS):
                target_overlaps.append({'target': target, 'source_seed': seed, 'criterion': criterion,
                                        'evaluation_rows': total, 'overlap_rows': int(values[i]),
                                        'overlap_fraction': float(values[i] / total)})
        target_metadata.append({'target': target, 'evaluation_rows': total, 'adaptation_rows_per_seed': int(total*.05),
                                'adaptation_in_evaluation': True, 'missing_session_fields': missing_groups,
                                'target_detection_labels_read': False, 'audit_seconds': time.perf_counter()-target_start})
        pd.DataFrame(target_overlaps).to_csv(out / 'source_target_overlap.csv', index=False)
        print(f'{target} complete, union overlap={counts["union_direct"].tolist()}', flush=True)

    manifest = {'protocol_id': PROTOCOL, 'source_origin': 'independent corrected reconstruction; historical H1 membership unavailable',
                'source_original_rows': source_size, 'source_retained_rows': len(retained),
                'source_removed_duplicates': source_size-len(retained), 'session_groups': len(group_keys),
                'mixed_label_session_groups': int(np.sum((group_attacks > 0) & (group_attacks < group_rows))),
                'source_group_fields_missing': 0, 'session_columns': FLOW_COLUMNS,
                'split_codes': {'train': 0, 'validation': 1, 'test': 2}, 'seeds': list(SEEDS),
                'deduplication': 'normalized complete source row, all 55 columns, first original row retained',
                'group_allocation': 'PCG64 seed shuffle within benign/attack/mixed strata; nearest cumulative-row 70% and 80% boundaries',
                'identity_method': 'paired uint64 pandas fingerprints, shared canonical numeric dtypes',
                'exact_row_definition': '53 parsed nonlabel fields before cleaning; numeric spelling differences are parsed to the same value',
                'targets': target_metadata, 'data_files': verified,
                'source_code_sha256': sha256_file(Path(__file__)),
                'elapsed_seconds': time.perf_counter()-started,
                'status': 'complete',
                'artifacts': {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'complete', 'seconds': manifest['elapsed_seconds'], 'output': str(out)}), flush=True)


if __name__ == '__main__':
    main()
