"""Audit archived target-only oracle partitions for raw/session dependence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
PROTOCOL_ID = "NIDS-TARGET-ORACLE-PARTITION-AUDIT-20260717-v1"
SEEDS = (42, 123, 2024)
SPLITS = ("train", "val", "test")
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
SESSION_COLUMNS = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "PROTOCOL",
]
TARGETS = {
    "CICIDS2018": {
        "dataset_file": "NF-CICIDS2018-v3.csv",
        "map_stem": "CICIDS2018",
        "file_slug": "cicids2018",
    },
    "BoT-IoT": {
        "dataset_file": "NF-BoT-IoT-v3.csv",
        "map_stem": "BoT_IoT",
        "file_slug": "bot_iot",
    },
    "ToN-IoT": {
        "dataset_file": "NF-ToN-IoT-v3.csv",
        "map_stem": "ToN_IoT",
        "file_slug": "ton_iot",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"{utc_now()}\t{message}", flush=True)


def sha256_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes(order="C")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def normalize_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if "L7_PROTO" in frame.columns:
        frame["L7_PROTO"] = frame["L7_PROTO"].fillna(0).round().astype(np.int64)
    return frame


def double_hash(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    first = hash_pandas_object(
        frame, index=False, categorize=True, hash_key="0123456789abcdef"
    ).to_numpy(dtype=np.uint64, copy=False)
    reverse = frame.loc[:, list(reversed(frame.columns))]
    second = hash_pandas_object(
        reverse, index=False, categorize=True, hash_key="fedcba9876543210"
    ).to_numpy(dtype=np.uint64, copy=False)
    return first, second


def expected_paths(args: argparse.Namespace) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for target, spec in TARGETS.items():
        dataset = args.project_root / "data" / "raw" / spec["dataset_file"]
        maps: dict[str, Any] = {}
        for seed in SEEDS:
            csv_path = args.oracle_root / "split_maps" / f"oracle_split_map_{spec['map_stem']}_seed{seed}.csv"
            maps[str(seed)] = {"csv": csv_path, "metadata": csv_path.with_suffix(".json")}
        targets[target] = {"dataset": dataset, "maps": maps}
    return {
        "targets": targets,
        "runner": args.project_root / "src" / "cross_domain_nids" / "oracle_engine.py",
        "results": args.oracle_root / "oracle_all_seeds_long.csv",
        "validation_json": args.oracle_root / "oracle_validation_report.json",
        "validation_text": args.oracle_root / "oracle_validation_report.txt",
    }


def file_record(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {role}: {path}")
    stat = path.stat()
    log(f"hashing role={role} bytes={stat.st_size} path={path}")
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def lock_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = expected_paths(args)
    records: list[dict[str, Any]] = []
    for target, item in paths["targets"].items():
        header = pd.read_csv(item["dataset"], nrows=0).columns.tolist()
        missing = sorted(set(SESSION_COLUMNS + ["L7_PROTO", "Label", "Attack"]) - set(header))
        if missing:
            raise RuntimeError(f"{target} missing audit columns: {missing}")
        records.append(file_record(item["dataset"], f"target_csv:{target}"))
        for seed, map_paths in item["maps"].items():
            records.append(file_record(map_paths["csv"], f"split_map:{target}:seed{seed}"))
            records.append(file_record(map_paths["metadata"], f"split_metadata:{target}:seed{seed}"))
    for role in ("runner", "results", "validation_json", "validation_text"):
        records.append(file_record(paths[role], role))
    for role, path in (
        ("frozen_policy", args.policy),
        ("audit_source", Path(__file__).resolve()),
    ):
        records.append(file_record(path, role))

    results = pd.read_csv(paths["results"])
    required = {"target_dataset", "seed", "model", "train_size", "val_size", "test_size", "full_target_rows"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise RuntimeError(f"Historical oracle result table missing columns: {missing}")
    if len(results) != 27:
        raise RuntimeError(f"Expected 27 historical oracle rows, found {len(results)}")
    units = results[["target_dataset", "seed"]].drop_duplicates()
    expected_units = {(target, seed) for target in TARGETS for seed in SEEDS}
    actual_units = {(str(r.target_dataset), int(r.seed)) for r in units.itertuples(index=False)}
    if actual_units != expected_units:
        raise RuntimeError(f"Historical oracle units mismatch: {sorted(actual_units)}")

    lock = {
        "protocol_id": PROTOCOL_ID,
        "locked_utc": utc_now(),
        "records": records,
        "result_rows": int(len(results)),
        "split_units": int(len(units)),
        "training_calls": 0,
    }
    atomic_json(args.output_dir / "input_lock.json", lock)
    log(f"input lock complete files={len(records)} sha256={sha256_file(args.output_dir / 'input_lock.json')}")
    return lock


def load_lock(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_dir / "input_lock.json"
    if not path.is_file():
        raise RuntimeError("Missing input_lock.json; run lock --execute first")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Input lock protocol mismatch")
    for record in lock["records"]:
        source = Path(record["path"])
        if not source.is_file():
            raise RuntimeError(f"Locked input missing: {source}")
        stat = source.stat()
        if int(stat.st_size) != int(record["bytes"]) or int(stat.st_mtime_ns) != int(record["mtime_ns"]):
            raise RuntimeError(f"Locked input metadata changed: {source}")
    return lock


def expected_rows_for_target(map_items: dict[str, Any]) -> int:
    totals = set()
    for item in map_items.values():
        metadata = json.loads(Path(item["metadata"]).read_text(encoding="utf-8"))
        counts = metadata["split_counts"]
        totals.add(int(counts["train"]) + int(counts["val"]) + int(counts["test"]))
    if len(totals) != 1:
        raise RuntimeError(f"Split metadata row totals disagree: {sorted(totals)}")
    return totals.pop()


def open_array(path: Path, dtype: Any, rows: int, mode: str = "w+") -> np.memmap:
    return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=(rows,))


def scan_target(target: str, dataset: Path, rows: int, temp: Path, chunksize: int) -> tuple[dict[str, Path], dict[str, Any]]:
    temp.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_h1": temp / "raw_h1.npy",
        "raw_h2": temp / "raw_h2.npy",
        "session_h1": temp / "session_h1.npy",
        "session_h2": temp / "session_h2.npy",
        "labels": temp / "labels.npy",
    }
    arrays = {
        name: open_array(path, np.uint8 if name == "labels" else np.uint64, rows)
        for name, path in paths.items()
    }
    header = pd.read_csv(dataset, nrows=0).columns.tolist()
    dtype_signatures: dict[str, set[str]] = {column: set() for column in header}
    offset = 0
    for chunk_number, chunk in enumerate(pd.read_csv(dataset, chunksize=chunksize, low_memory=False), start=1):
        end = offset + len(chunk)
        if end > rows:
            raise RuntimeError(f"{target} target CSV exceeds metadata rows")
        chunk = normalize_chunk(chunk)
        for column, dtype in chunk.dtypes.items():
            dtype_signatures[column].add(str(dtype))
        labels = pd.to_numeric(chunk["Label"], errors="coerce")
        if labels.isna().any() or not set(labels.unique()).issubset({0, 1, 0.0, 1.0}):
            raise RuntimeError(f"{target} has non-binary Label values")
        arrays["labels"][offset:end] = labels.to_numpy(dtype=np.uint8, copy=False)
        for kind, frame in (("raw", chunk[header]), ("session", chunk[SESSION_COLUMNS])):
            first, second = double_hash(frame)
            arrays[f"{kind}_h1"][offset:end] = first
            arrays[f"{kind}_h2"][offset:end] = second
        offset = end
        if chunk_number == 1 or chunk_number % 5 == 0 or offset == rows:
            log(f"fingerprinted target={target} rows={offset}/{rows}")
        del chunk
        gc.collect()
    if offset != rows:
        raise RuntimeError(f"{target} target rows {offset} != metadata rows {rows}")
    for array in arrays.values():
        array.flush()
    inconsistent = {column: sorted(values) for column, values in dtype_signatures.items() if len(values) != 1}
    if inconsistent:
        raise RuntimeError(f"{target} chunk dtype drift prevents reliable cross-chunk hashing: {inconsistent}")
    metadata = {
        "target": target,
        "rows": rows,
        "columns": header,
        "column_count": len(header),
        "dtype_signature": {column: next(iter(values)) for column, values in dtype_signatures.items()},
        "label_sha256": hash_array(arrays["labels"]),
    }
    del arrays
    gc.collect()
    return paths, metadata


def load_and_validate_map(
    target: str,
    seed: int,
    csv_path: Path,
    metadata_path: Path,
    target_labels_path: Path,
    rows: int,
    code_path: Path,
    chunksize: int,
) -> dict[str, Any]:
    target_labels = open_array(target_labels_path, np.uint8, rows, mode="r")
    codes = open_array(code_path, np.uint8, rows)
    expected = 0
    label_mismatches = 0
    unknown_splits = 0
    for chunk in pd.read_csv(csv_path, usecols=["row_id", "binary_label", "split"], chunksize=chunksize):
        n = len(chunk)
        end = expected + n
        row_ids = chunk["row_id"].to_numpy(dtype=np.int64, copy=False)
        if n and (row_ids[0] != expected or row_ids[-1] != end - 1 or not np.array_equal(row_ids, np.arange(expected, end))):
            raise RuntimeError(f"{target} seed={seed} split-map row IDs are not contiguous")
        mapped = chunk["split"].map(SPLIT_CODE)
        unknown_splits += int(mapped.isna().sum())
        if unknown_splits:
            raise RuntimeError(f"{target} seed={seed} split map has unknown split labels")
        map_labels = pd.to_numeric(chunk["binary_label"], errors="coerce").to_numpy()
        if np.isnan(map_labels).any():
            raise RuntimeError(f"{target} seed={seed} split map has invalid labels")
        map_labels = map_labels.astype(np.uint8, copy=False)
        label_mismatches += int(np.count_nonzero(map_labels != target_labels[expected:end]))
        codes[expected:end] = mapped.to_numpy(dtype=np.uint8, copy=False)
        expected = end
        del chunk
    codes.flush()
    if expected != rows:
        raise RuntimeError(f"{target} seed={seed} split map rows {expected} != {rows}")
    counts = {name: int(np.count_nonzero(codes == code)) for name, code in SPLIT_CODE.items()}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_counts = {name: int(metadata["split_counts"]["validation" if name == "val" and "validation" in metadata["split_counts"] else name]) for name in SPLITS}
    row = {
        "target": target,
        "seed": seed,
        "rows": rows,
        "row_ids_contiguous_unique_complete": expected == rows,
        "label_mismatches": label_mismatches,
        "labels_match_target": label_mismatches == 0,
        "train_rows": counts["train"],
        "val_rows": counts["val"],
        "test_rows": counts["test"],
        "partition_sum_matches_rows": sum(counts.values()) == rows,
        "counts_match_metadata": counts == metadata_counts,
        "assignment_sha256": hash_array(codes),
        "target_label_sha256": hash_array(target_labels),
    }
    del codes, target_labels
    gc.collect()
    return row


def group_layout(h1_path: Path, h2_path: Path, rows: int, kind: str, target: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    h1 = open_array(h1_path, np.uint64, rows, mode="r")
    h2 = open_array(h2_path, np.uint64, rows, mode="r")
    log(f"sorting fingerprints target={target} key={kind} rows={rows}")
    order = np.lexsort((h2, h1))
    sorted_h1 = h1[order]
    sorted_h2 = h2[order]
    change = np.empty(rows, dtype=bool)
    change[0] = True
    change[1:] = (sorted_h1[1:] != sorted_h1[:-1]) | (sorted_h2[1:] != sorted_h2[:-1])
    starts = np.flatnonzero(change).astype(np.int64, copy=False)
    sizes = np.diff(starts, append=rows).astype(np.int64, copy=False)
    structure = {
        "target": target,
        "key_kind": kind,
        "rows": rows,
        "unique_groups": int(len(starts)),
        "multirow_groups": int(np.count_nonzero(sizes > 1)),
        "rows_in_multirow_groups": int(sizes[sizes > 1].sum()),
        "duplicate_or_repeated_rows_beyond_first": int(rows - len(starts)),
        "maximum_group_size": int(sizes.max()),
    }
    del h1, h2, sorted_h1, sorted_h2, change
    gc.collect()
    log(f"grouped target={target} key={kind} groups={len(starts)} multirow={structure['multirow_groups']}")
    return order, starts, structure


def summarize_assignment(order: np.ndarray, starts: np.ndarray, code_path: Path, rows: int, target: str, seed: int, kind: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    codes = open_array(code_path, np.uint8, rows, mode="r")
    sorted_codes = codes[order]
    masks = np.bitwise_or.reduceat(np.left_shift(np.uint8(1), sorted_codes), starts)
    counts = [
        np.add.reduceat((sorted_codes == code).astype(np.uint8), starts, dtype=np.int64)
        for code in range(3)
    ]
    sizes = counts[0] + counts[1] + counts[2]
    crossing = ~np.isin(masks, np.array([1, 2, 4], dtype=np.uint8))
    test_any = ((masks & 4) != 0) & ((masks & 3) != 0)
    val_train = ((masks & 2) != 0) & ((masks & 1) != 0)
    split_totals = [int(np.count_nonzero(codes == code)) for code in range(3)]
    summary = {
        "target": target,
        "seed": seed,
        "key_kind": kind,
        "rows": rows,
        "groups": int(len(starts)),
        "cross_partition_groups": int(np.count_nonzero(crossing)),
        "rows_in_cross_partition_groups": int(sizes[crossing].sum()),
        "fraction_rows_in_cross_partition_groups": float(sizes[crossing].sum() / rows),
        "test_rows_with_train_or_val_identity": int(counts[2][test_any].sum()),
        "test_fraction_with_train_or_val_identity": float(counts[2][test_any].sum() / max(1, split_totals[2])),
        "val_rows_with_train_identity": int(counts[1][val_train].sum()),
        "val_fraction_with_train_identity": float(counts[1][val_train].sum() / max(1, split_totals[1])),
        "test_independent": bool(not test_any.any()),
        "all_partitions_independent": bool(not crossing.any()),
    }
    pair_rows: list[dict[str, Any]] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        shared = ((masks & (1 << left)) != 0) & ((masks & (1 << right)) != 0)
        pair_rows.append({
            "target": target,
            "seed": seed,
            "key_kind": kind,
            "left_partition": SPLITS[left],
            "right_partition": SPLITS[right],
            "shared_groups": int(np.count_nonzero(shared)),
            "left_rows_in_shared_groups": int(counts[left][shared].sum()),
            "right_rows_in_shared_groups": int(counts[right][shared].sum()),
            "pair_disjoint": bool(not shared.any()),
        })
    del codes, sorted_codes, masks, counts, sizes, crossing, test_any, val_train
    gc.collect()
    return summary, pair_rows


def audit_one_target(args: argparse.Namespace, target: str, target_paths: dict[str, Any], lock_sha: str) -> dict[str, Any]:
    unit_path = args.output_dir / "units" / f"{target}.json"
    if args.resume and unit_path.is_file():
        existing = json.loads(unit_path.read_text(encoding="utf-8"))
        if existing.get("input_lock_sha256") == lock_sha:
            log(f"resume target={target} from {unit_path}")
            return existing
    temp = args.output_dir / "temp" / target
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    rows = expected_rows_for_target(target_paths["maps"])
    hash_paths, scan_metadata = scan_target(target, target_paths["dataset"], rows, temp, args.chunksize)
    map_rows = []
    code_paths: dict[int, Path] = {}
    for seed in SEEDS:
        map_paths = target_paths["maps"][str(seed)]
        code_path = temp / f"split_codes_seed{seed}.npy"
        code_paths[seed] = code_path
        log(f"validating split map target={target} seed={seed}")
        map_rows.append(load_and_validate_map(
            target, seed, map_paths["csv"], map_paths["metadata"], hash_paths["labels"], rows, code_path, args.chunksize
        ))
    structures = []
    summaries = []
    pairwise = []
    for kind in ("normalized_raw", "timestamped_session"):
        prefix = "raw" if kind == "normalized_raw" else "session"
        order, starts, structure = group_layout(hash_paths[f"{prefix}_h1"], hash_paths[f"{prefix}_h2"], rows, kind, target)
        structures.append(structure)
        for seed in SEEDS:
            log(f"counting overlap target={target} seed={seed} key={kind}")
            summary, pairs = summarize_assignment(order, starts, code_paths[seed], rows, target, seed, kind)
            summaries.append(summary)
            pairwise.extend(pairs)
        del order, starts
        gc.collect()
    result = {
        "protocol_id": PROTOCOL_ID,
        "input_lock_sha256": lock_sha,
        "target": target,
        "scan_metadata": scan_metadata,
        "split_map_validation": map_rows,
        "dataset_structure": structures,
        "overlap_by_seed": summaries,
        "pairwise_overlap": pairwise,
    }
    atomic_json(unit_path, result)
    if not args.keep_temp:
        shutil.rmtree(temp)
    log(f"completed target={target}")
    return result


def run_audit(args: argparse.Namespace) -> None:
    load_lock(args)
    lock_path = args.output_dir / "input_lock.json"
    lock_sha = sha256_file(lock_path)
    paths = expected_paths(args)
    for target, target_paths in paths["targets"].items():
        audit_one_target(args, target, target_paths, lock_sha)
    atomic_json(args.output_dir / "AUDIT_COMPLETED.json", {
        "protocol_id": PROTOCOL_ID,
        "completed_utc": utc_now(),
        "split_units": 9,
        "training_calls": 0,
        "input_lock_sha256": lock_sha,
    })


def check_row(checks: list[dict[str, Any]], category: str, check: str, passed: bool, detail: Any) -> None:
    checks.append({"category": category, "check": check, "status": "Pass" if passed else "Fail", "detail": detail})


def make_report(final: dict[str, Any], overlap: pd.DataFrame, pairwise: pd.DataFrame, structure: pd.DataFrame, validation: pd.DataFrame) -> str:
    target_summary = (
        overlap.groupby(["target", "key_kind"], as_index=False)
        .agg(
            maximum_cross_partition_groups=("cross_partition_groups", "max"),
            maximum_test_rows_with_train_or_val_identity=("test_rows_with_train_or_val_identity", "max"),
            maximum_test_fraction_with_train_or_val_identity=("test_fraction_with_train_or_val_identity", "max"),
            all_seed_test_independent=("test_independent", "all"),
        )
    )
    lines = [
        "# Target-only oracle internal partition integrity audit",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        f"**Decision:** `{final['decision']}`",
        "",
        f"Training runs: **0**. Split units: **{final['split_units']}/9**. Historical oracle rows linked: **{final['historical_result_rows']}/27**.",
        "",
        "## Target-level overlap",
        "",
        target_summary.to_markdown(index=False),
        "",
        "## Dataset identity structure",
        "",
        structure.to_markdown(index=False),
        "",
        "## Pairwise partition overlap",
        "",
        pairwise.to_markdown(index=False),
        "",
        "## Mechanical validation",
        "",
        f"Checks: {len(validation)}; failures: {(validation['status'] != 'Pass').sum()}.",
        "",
        "## Interpretation",
        "",
    ]
    if final["decision"] == "Blocker":
        lines.extend([
            "The historical row-stratified target-only oracle is not a duplicate/session-independent ceiling under the frozen criteria. Its numerical results may be optimistic, but this audit does not estimate the magnitude of inflation. A duplicate-free, timestamped-session-disjoint oracle split must be constructed and the oracle models retrained before those values are described as a clean target-domain ceiling.",
            "",
        ])
    elif final["decision"] == "Risk":
        lines.extend([
            "The historical test partition is independent under the frozen keys, but train-validation dependence remains. This is a model-selection risk for validation-using models and should be repaired before the oracle is described as fully partition-independent.",
            "",
        ])
    else:
        lines.extend([
            "All archived oracle partitions are row-, normalized-raw-, and timestamped-session-disjoint under the frozen definitions.",
            "",
        ])
    lines.extend([
        "The audit uses all archived rows and exact row-index maps. It performs no training, prediction, threshold tuning, checkpoint selection, or counterfactual metric correction.",
        "",
    ])
    return "\n".join(lines)


def make_disposition(final: dict[str, Any]) -> str:
    decision = final["decision"]
    if decision == "Blocker":
        action = "Do not present the historical oracle numbers as a clean ceiling. Construct one common duplicate-free, timestamped-session-disjoint 70/10/20 split per target and seed, then retrain RF, XGB, and MLP on those locked maps."
        wording = "An audit of the historical target-only oracle found that its row-stratified partitions were not independent at the normalized-record and/or timestamped-session level; we therefore treat the archived values as uncorrected reference results rather than a clean oracle ceiling."
    elif decision == "Risk":
        action = "Retain test results only with an explicit train-validation dependence caveat, and rebuild the split before relying on validation-based model selection."
        wording = "The oracle test sets were disjoint under the audited identities, although train-validation dependence remained for validation-using models."
    else:
        action = "Retain the oracle ceiling and add the partition-integrity audit to the protocol and Supplement."
        wording = "The target-only oracle train, validation, and test partitions were disjoint at row, normalized-record, and timestamped-session levels."
    return "\n".join([
        "# Target-only oracle manuscript revision disposition",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        f"Decision: **{decision}**",
        "",
        "## Required action",
        "",
        action,
        "",
        "## Reviewer-facing wording",
        "",
        f"> {wording}",
        "",
        "## Claim boundary",
        "",
        "Overlap establishes a credible leakage route, not the numerical amount of inflation. Historical metrics cannot be repaired analytically; a corrected ceiling requires retraining. Random-row disjointness must not be described as session independence.",
        "",
    ])


def build_release_manifest(args: argparse.Namespace, lock: dict[str, Any]) -> None:
    manifest = args.output_dir / "release_manifest_sha256.csv"
    rows = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path != manifest and "temp" not in path.parts:
            rows.append({
                "role": "generated",
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(args.output_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    for item in lock["records"]:
        rows.append({
            "role": "locked_input:" + item["role"],
            "path": item["path"],
            "relative_path": "",
            "sha256": item["sha256"],
            "bytes": item["bytes"],
        })
    pd.DataFrame(rows).drop_duplicates(subset=["role", "path"]).to_csv(manifest, index=False)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_lock(args)
    completion = args.output_dir / "AUDIT_COMPLETED.json"
    if not completion.is_file():
        raise RuntimeError("Audit is incomplete; run audit --execute first")
    units = []
    for target in TARGETS:
        path = args.output_dir / "units" / f"{target}.json"
        if not path.is_file():
            raise RuntimeError(f"Missing target audit unit: {path}")
        units.append(json.loads(path.read_text(encoding="utf-8")))
    map_validation = pd.DataFrame([row for unit in units for row in unit["split_map_validation"]])
    structure = pd.DataFrame([row for unit in units for row in unit["dataset_structure"]])
    overlap = pd.DataFrame([row for unit in units for row in unit["overlap_by_seed"]])
    pairwise = pd.DataFrame([row for unit in units for row in unit["pairwise_overlap"]])
    map_validation.to_csv(args.output_dir / "split_map_validation.csv", index=False)
    structure.to_csv(args.output_dir / "dataset_identity_structure.csv", index=False)
    overlap.to_csv(args.output_dir / "overlap_by_seed.csv", index=False)
    pairwise.to_csv(args.output_dir / "pairwise_overlap.csv", index=False)

    results_path = Path(next(item["path"] for item in lock["records"] if item["role"] == "results"))
    historical = pd.read_csv(results_path)
    linked = historical.merge(
        map_validation,
        left_on=["target_dataset", "seed"],
        right_on=["target", "seed"],
        how="left",
        validate="many_to_one",
    )
    linked["result_counts_match_map"] = (
        (linked["train_size"].astype(int) == linked["train_rows"].astype(int))
        & (linked["val_size"].astype(int) == linked["val_rows"].astype(int))
        & (linked["test_size"].astype(int) == linked["test_rows"].astype(int))
        & (linked["full_target_rows"].astype(int) == linked["rows"].astype(int))
    )
    linkage_columns = [
        "target_dataset", "seed", "model", "train_size", "val_size", "test_size",
        "full_target_rows", "assignment_sha256", "result_counts_match_map",
    ]
    linked[linkage_columns].to_csv(args.output_dir / "oracle_result_linkage.csv", index=False)

    checks: list[dict[str, Any]] = []
    check_row(checks, "coverage", "nine split units complete", len(map_validation) == 9, len(map_validation))
    check_row(checks, "coverage", "eighteen identity-seed units complete", len(overlap) == 18, len(overlap))
    check_row(checks, "coverage", "fifty-four pairwise units complete", len(pairwise) == 54, len(pairwise))
    check_row(checks, "coverage", "twenty-seven historical result rows linked", len(linked) == 27, len(linked))
    for row in map_validation.itertuples(index=False):
        prefix = f"{row.target} seed={row.seed}"
        check_row(checks, "split_map", prefix + " row IDs complete", bool(row.row_ids_contiguous_unique_complete), row.rows)
        check_row(checks, "split_map", prefix + " labels match target", bool(row.labels_match_target), row.label_mismatches)
        check_row(checks, "split_map", prefix + " partition sum", bool(row.partition_sum_matches_rows), row.rows)
        check_row(checks, "split_map", prefix + " counts match metadata", bool(row.counts_match_metadata), f"{row.train_rows}/{row.val_rows}/{row.test_rows}")
    for row in linked.itertuples(index=False):
        check_row(checks, "result_linkage", f"{row.target_dataset} seed={row.seed} model={row.model} counts", bool(row.result_counts_match_map), row.assignment_sha256)

    mechanical_failures = [row for row in checks if row["status"] != "Pass"]
    if mechanical_failures:
        decision = "Unknown"
    else:
        any_test = bool((overlap["test_rows_with_train_or_val_identity"] > 0).any())
        any_cross = bool((overlap["cross_partition_groups"] > 0).any())
        decision = "Blocker" if any_test else ("Risk" if any_cross else "Pass")
    validation = pd.DataFrame(checks)
    validation.to_csv(args.output_dir / "validation_checks.csv", index=False)
    final = {
        "protocol_id": PROTOCOL_ID,
        "generated_utc": utc_now(),
        "training_runs": 0,
        "split_units": int(len(map_validation)),
        "identity_seed_units": int(len(overlap)),
        "pairwise_units": int(len(pairwise)),
        "historical_result_rows": int(len(linked)),
        "mechanical_checks": int(len(validation)),
        "mechanical_failures": int((validation["status"] != "Pass").sum()),
        "normalized_raw_test_overlap_present": bool((overlap.loc[overlap.key_kind == "normalized_raw", "test_rows_with_train_or_val_identity"] > 0).any()),
        "timestamped_session_test_overlap_present": bool((overlap.loc[overlap.key_kind == "timestamped_session", "test_rows_with_train_or_val_identity"] > 0).any()),
        "decision": decision,
        "historical_oracle_ceiling_clean": decision == "Pass",
        "requires_corrected_split_retraining": decision == "Blocker",
    }
    atomic_json(args.output_dir / "final_verdict.json", final)
    report = make_report(final, overlap, pairwise, structure, validation)
    (args.output_dir / "TARGET_ORACLE_PARTITION_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    (args.output_dir / "TARGET_ORACLE_MANUSCRIPT_REVISION_DISPOSITION.md").write_text(make_disposition(final), encoding="utf-8")
    build_release_manifest(args, lock)
    print(json.dumps(final, indent=2), flush=True)
    return final


def environment_record(args: argparse.Namespace) -> None:
    atomic_json(args.output_dir / "environment_manifest.json", {
        "protocol_id": PROTOCOL_ID,
        "generated_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "command": sys.argv,
        "working_directory": str(Path.cwd()),
    })


def plan(args: argparse.Namespace) -> None:
    paths = expected_paths(args)
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "mode": "audit_only_no_training",
        "targets": list(TARGETS),
        "seeds": list(SEEDS),
        "split_units": 9,
        "identity_seed_units": 18,
        "pairwise_units": 54,
        "training_calls": 0,
        "output_dir": str(args.output_dir),
        "all_required_paths_exist": all(
            path.is_file()
            for item in paths["targets"].values()
            for path in [item["dataset"]] + [p for pair in item["maps"].values() for p in pair.values()]
        ),
    }, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="plan", choices=("plan", "lock", "audit", "analyze", "all"))
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=REPOSITORY_ROOT / "external" / "historical_oracle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "generated" / "oracle_audit",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=REPOSITORY_ROOT / "protocols" / "oracle" / "audit_policy.json",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        plan(args)
        return
    if not args.execute:
        raise SystemExit("Write/scan commands require --execute; use plan for the safe default")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment_record(args)
    log("mode=audit-only; training/prediction/checkpoint-selection/metric-correction prohibited")
    if args.command in ("lock", "all"):
        lock_inputs(args)
    if args.command in ("audit", "all"):
        run_audit(args)
    if args.command in ("analyze", "all"):
        analyze(args)


if __name__ == "__main__":
    main()
