"""Generate minimally repaired duplicate-free/session-disjoint oracle maps."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from cross_domain_nids import oracle_split_audit as audit  # noqa: E402


PROTOCOL_ID = "NIDS-TARGET-ORACLE-CORRECTED-CEILING-20260717-v1"
EXPECTED_PARENT_LOCK_SHA256 = "7e6a9970ba72ea851afec31b4590349631a0802e41826d75691ec1a4c7c57918"
TARGET_FRACTIONS = np.array([0.7, 0.1, 0.2], dtype=float)
SPLIT_NAMES = np.array(["train", "val", "test", "drop"], dtype=object)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"{utc_now()}\t{message}", flush=True)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def hash_file(path: Path) -> str:
    return audit.sha256_file(path)


def hash_array(values: np.ndarray, dtype: Any | None = None) -> str:
    array = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def parent_lock_path(args: argparse.Namespace) -> Path:
    return args.parent_audit_dir / "input_lock.json"


def verify_parent_lock(args: argparse.Namespace, full_hash: bool = False) -> dict[str, Any]:
    path = parent_lock_path(args)
    if hash_file(path) != EXPECTED_PARENT_LOCK_SHA256:
        raise RuntimeError("Parent audit input lock SHA-256 mismatch")
    lock = json.loads(path.read_text(encoding="utf-8"))
    for record in lock["records"]:
        source = Path(record["path"])
        if not source.is_file():
            raise RuntimeError(f"Parent-locked input missing: {source}")
        stat = source.stat()
        if int(stat.st_size) != int(record["bytes"]) or int(stat.st_mtime_ns) != int(record["mtime_ns"]):
            raise RuntimeError(f"Parent-locked input metadata changed: {source}")
        if full_hash and hash_file(source) != record["sha256"]:
            raise RuntimeError(f"Parent-locked input SHA-256 changed: {source}")
    return lock


def lock_execution(args: argparse.Namespace) -> dict[str, Any]:
    parent = verify_parent_lock(args, full_hash=True)
    sources = [
        args.policy.resolve(),
        Path(__file__).resolve(),
        (HERE / "run_target_oracle.py").resolve(),
        (HERE / "finalize_results.py").resolve(),
        (REPOSITORY_ROOT / "src" / "cross_domain_nids" / "oracle_split_audit.py").resolve(),
    ]
    records = []
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": str(path), "sha256": hash_file(path), "bytes": path.stat().st_size})
    value = {
        "protocol_id": PROTOCOL_ID,
        "locked_utc": utc_now(),
        "parent_input_lock_path": str(parent_lock_path(args).resolve()),
        "parent_input_lock_sha256": EXPECTED_PARENT_LOCK_SHA256,
        "parent_locked_records": len(parent["records"]),
        "frozen_sources": records,
        "training_runs_at_lock": 0,
    }
    atomic_json(args.output_dir / "execution_lock.json", value)
    log(f"execution lock complete sha256={hash_file(args.output_dir / 'execution_lock.json')}")
    return value


def verify_execution_lock(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_dir / "execution_lock.json"
    if not path.is_file():
        raise RuntimeError("Missing execution_lock.json; run lock --execute after design review")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("Execution lock protocol mismatch")
    if value["parent_input_lock_sha256"] != EXPECTED_PARENT_LOCK_SHA256:
        raise RuntimeError("Execution lock parent mismatch")
    verify_parent_lock(args, full_hash=False)
    for record in value["frozen_sources"]:
        path = Path(record["path"])
        if hash_file(path) != record["sha256"]:
            raise RuntimeError(f"Frozen source changed after execution lock: {path}")
    return value


def paths_from_parent(args: argparse.Namespace) -> dict[str, Any]:
    proxy = argparse.Namespace(project_root=args.project_root, oracle_root=args.historical_oracle_root)
    return audit.expected_paths(proxy)


def raw_representatives(raw_h1: Path, raw_h2: Path, rows: int, target: str) -> tuple[np.ndarray, dict[str, Any]]:
    order, starts, structure = audit.group_layout(raw_h1, raw_h2, rows, "normalized_raw", target)
    retained = np.sort(np.minimum.reduceat(order, starts).astype(np.int64, copy=False))
    if len(retained) != structure["unique_groups"] or np.unique(retained).size != len(retained):
        raise RuntimeError(f"{target} raw representative construction failed")
    structure["retained_rows"] = int(len(retained))
    structure["dropped_duplicate_rows"] = int(rows - len(retained))
    structure["dropped_fraction"] = float((rows - len(retained)) / rows)
    structure["retained_row_ids_sha256"] = hash_array(retained, dtype="<i8")
    del order, starts
    gc.collect()
    return retained, structure


def retained_session_layout(session_h1_path: Path, session_h2_path: Path, retained: np.ndarray, rows: int, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    h1 = audit.open_array(session_h1_path, np.uint64, rows, mode="r")
    h2 = audit.open_array(session_h2_path, np.uint64, rows, mode="r")
    retained_h1 = h1[retained]
    retained_h2 = h2[retained]
    log(f"sorting retained sessions target={target} retained_rows={len(retained)}")
    order_position = np.lexsort((retained_h2, retained_h1))
    sorted_ids = retained[order_position]
    sorted_h1 = retained_h1[order_position]
    sorted_h2 = retained_h2[order_position]
    change = np.empty(len(retained), dtype=bool)
    change[0] = True
    change[1:] = (sorted_h1[1:] != sorted_h1[:-1]) | (sorted_h2[1:] != sorted_h2[:-1])
    starts = np.flatnonzero(change).astype(np.int64, copy=False)
    sizes = np.diff(starts, append=len(retained)).astype(np.int64, copy=False)
    lowest_ids = np.minimum.reduceat(sorted_ids, starts).astype(np.int64, copy=False)
    structure = {
        "target": target,
        "retained_rows": int(len(retained)),
        "session_groups": int(len(starts)),
        "multirow_session_groups": int(np.count_nonzero(sizes > 1)),
        "rows_in_multirow_session_groups": int(sizes[sizes > 1].sum()),
        "maximum_session_group_size": int(sizes.max()),
    }
    del h1, h2, retained_h1, retained_h2, order_position, sorted_h1, sorted_h2, change
    gc.collect()
    return sorted_ids, starts, lowest_ids, structure


def plurality_repair(
    historical: np.ndarray,
    sorted_retained_ids: np.ndarray,
    starts: np.ndarray,
    lowest_ids: np.ndarray,
    rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    historical_sorted = historical[sorted_retained_ids]
    counts = [
        np.add.reduceat((historical_sorted == code).astype(np.uint8), starts, dtype=np.int64)
        for code in range(3)
    ]
    matrix = np.column_stack(counts)
    assignment = np.argmax(matrix, axis=1).astype(np.uint8)
    maximum = matrix.max(axis=1, keepdims=True)
    tied = (matrix == maximum).sum(axis=1) > 1
    assignment[tied] = historical[lowest_ids[tied]]
    sizes = matrix.sum(axis=1).astype(np.int64, copy=False)
    corrected = np.full(rows, 3, dtype=np.uint8)
    corrected[sorted_retained_ids] = np.repeat(assignment, sizes)
    masks = np.bitwise_or.reduceat(np.left_shift(np.uint8(1), corrected[sorted_retained_ids]), starts)
    crossing = int(np.count_nonzero(~np.isin(masks, np.array([1, 2, 4], dtype=np.uint8))))
    diagnostics = {
        "session_groups": int(len(starts)),
        "tied_session_groups": int(np.count_nonzero(tied)),
        "session_groups_crossing_corrected_partitions": crossing,
    }
    del historical_sorted, counts, matrix, assignment, maximum, tied, sizes, masks
    gc.collect()
    return corrected, diagnostics


def write_map(path: Path, split_codes: np.ndarray, binary_labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        split_codes=np.asarray(split_codes, dtype=np.uint8),
        binary_labels=np.asarray(binary_labels, dtype=np.uint8),
    )
    os.replace(temporary, path)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def validate_corrected(
    corrected: np.ndarray,
    historical: np.ndarray,
    labels: np.ndarray,
    retained: np.ndarray,
    session_crossing: int,
) -> dict[str, Any]:
    retained_mask = corrected != 3
    if int(retained_mask.sum()) != len(retained) or not np.array_equal(np.flatnonzero(retained_mask), retained):
        raise RuntimeError("Corrected retained membership mismatch")
    counts = np.array([np.count_nonzero(corrected == code) for code in range(3)], dtype=np.int64)
    fractions = counts / len(retained)
    overall_rate = float(labels[retained].mean())
    rates = np.array([labels[corrected == code].mean() for code in range(3)], dtype=float)
    class_presence = [set(np.unique(labels[corrected == code]).tolist()) == {0, 1} for code in range(3)]
    moved = int(np.count_nonzero(corrected[retained] != historical[retained]))
    return {
        "train_rows": int(counts[0]),
        "val_rows": int(counts[1]),
        "test_rows": int(counts[2]),
        "drop_rows": int(np.count_nonzero(corrected == 3)),
        "retained_rows": int(len(retained)),
        "train_fraction": float(fractions[0]),
        "val_fraction": float(fractions[1]),
        "test_fraction": float(fractions[2]),
        "maximum_absolute_partition_fraction_deviation": float(np.max(np.abs(fractions - TARGET_FRACTIONS))),
        "retained_attack_rate": overall_rate,
        "train_attack_rate": float(rates[0]),
        "val_attack_rate": float(rates[1]),
        "test_attack_rate": float(rates[2]),
        "maximum_absolute_attack_prevalence_deviation": float(np.max(np.abs(rates - overall_rate))),
        "both_classes_in_all_partitions": bool(all(class_presence)),
        "session_groups_crossing_corrected_partitions": int(session_crossing),
        "retained_normalized_duplicate_groups": 0,
        "moved_retained_rows": moved,
        "moved_retained_fraction": float(moved / len(retained)),
        "train_membership_jaccard": jaccard(corrected == 0, historical == 0),
        "val_membership_jaccard": jaccard(corrected == 1, historical == 1),
        "test_membership_jaccard": jaccard(corrected == 2, historical == 2),
        "assignment_sha256": hash_array(corrected, dtype=np.uint8),
    }


def target_complete(args: argparse.Namespace, target: str, execution_lock_sha: str) -> bool:
    unit = args.output_dir / "units" / f"{target}.json"
    if not args.resume or not unit.is_file():
        return False
    value = json.loads(unit.read_text(encoding="utf-8"))
    if value.get("execution_lock_sha256") != execution_lock_sha:
        return False
    for row in value.get("split_maps", []):
        path = Path(row["split_map_path"])
        if not path.is_file() or hash_file(path) != row["split_map_sha256"]:
            return False
    return True


def generate_target(args: argparse.Namespace, target: str, target_paths: dict[str, Any], execution_lock_sha: str) -> dict[str, Any]:
    unit_path = args.output_dir / "units" / f"{target}.json"
    if target_complete(args, target, execution_lock_sha):
        log(f"resume corrected maps target={target}")
        return json.loads(unit_path.read_text(encoding="utf-8"))
    temp = args.output_dir / "temp" / target
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    rows = audit.expected_rows_for_target(target_paths["maps"])
    hash_paths, scan_metadata = audit.scan_target(target, target_paths["dataset"], rows, temp, args.chunksize)
    retained, raw_structure = raw_representatives(hash_paths["raw_h1"], hash_paths["raw_h2"], rows, target)
    sorted_ids, session_starts, lowest_ids, session_structure = retained_session_layout(
        hash_paths["session_h1"], hash_paths["session_h2"], retained, rows, target
    )
    labels = audit.open_array(hash_paths["labels"], np.uint8, rows, mode="r")
    split_rows = []
    for seed in audit.SEEDS:
        historical_path = temp / f"historical_codes_seed{seed}.npy"
        map_paths = target_paths["maps"][str(seed)]
        historical_validation = audit.load_and_validate_map(
            target, seed, map_paths["csv"], map_paths["metadata"], hash_paths["labels"], rows,
            historical_path, args.chunksize
        )
        if not historical_validation["labels_match_target"] or not historical_validation["counts_match_metadata"]:
            raise RuntimeError(f"Historical split validation failed target={target} seed={seed}")
        historical = audit.open_array(historical_path, np.uint8, rows, mode="r")
        corrected, repair = plurality_repair(historical, sorted_ids, session_starts, lowest_ids, rows)
        validation = validate_corrected(corrected, historical, labels, retained, repair["session_groups_crossing_corrected_partitions"])
        if validation["retained_normalized_duplicate_groups"] != 0 or validation["session_groups_crossing_corrected_partitions"] != 0:
            raise RuntimeError(f"Integrity gate failed target={target} seed={seed}")
        if validation["maximum_absolute_partition_fraction_deviation"] > 0.01:
            raise RuntimeError(f"Partition fraction gate failed target={target} seed={seed}")
        if validation["maximum_absolute_attack_prevalence_deviation"] > 0.01:
            raise RuntimeError(f"Attack prevalence gate failed target={target} seed={seed}")
        if not validation["both_classes_in_all_partitions"]:
            raise RuntimeError(f"Class-presence gate failed target={target} seed={seed}")
        slug = audit.TARGETS[target]["file_slug"]
        map_path = args.output_dir / "split_maps" / f"oracle__{slug}__seed_{seed}.npz"
        write_map(map_path, corrected, labels)
        row = {
            "protocol_id": PROTOCOL_ID,
            "target": target,
            "seed": seed,
            "target_csv_path": f"data/raw/{target_paths['dataset'].name}",
            "historical_split_map_path": (
                f"external/historical_oracle/split_maps/{map_paths['csv'].name}"
            ),
            "split_map_path": f"split_maps/{map_path.name}",
            "split_map_sha256": hash_file(map_path),
            "target_label_sha256": scan_metadata["label_sha256"],
            "retained_row_ids_sha256": raw_structure["retained_row_ids_sha256"],
            **repair,
            **validation,
        }
        metadata_path = map_path.with_suffix(".json")
        atomic_json(metadata_path, row)
        row["metadata_path"] = f"split_maps/{metadata_path.name}"
        row["metadata_sha256"] = hash_file(metadata_path)
        split_rows.append(row)
        log(
            f"corrected map target={target} seed={seed} retained={validation['retained_rows']} "
            f"drop={validation['drop_rows']} moved={validation['moved_retained_fraction']:.6f}"
        )
        del historical, corrected
        gc.collect()
    value = {
        "protocol_id": PROTOCOL_ID,
        "execution_lock_sha256": execution_lock_sha,
        "target": target,
        "scan_metadata": scan_metadata,
        "raw_structure": raw_structure,
        "session_structure": session_structure,
        "split_maps": split_rows,
    }
    atomic_json(unit_path, value)
    del labels, retained, sorted_ids, session_starts, lowest_ids
    gc.collect()
    if not args.keep_temp:
        shutil.rmtree(temp)
    return value


def consolidate(args: argparse.Namespace, units: list[dict[str, Any]]) -> None:
    rows = pd.DataFrame([row for unit in units for row in unit["split_maps"]])
    rows.to_csv(args.output_dir / "split_manifest.csv", index=False)
    checks = []
    for row in rows.itertuples(index=False):
        prefix = f"{row.target} seed={row.seed}"
        values = {
            "normalized duplicate groups zero": row.retained_normalized_duplicate_groups == 0,
            "session crossing groups zero": row.session_groups_crossing_corrected_partitions == 0,
            "partition fraction gate": row.maximum_absolute_partition_fraction_deviation <= 0.01,
            "attack prevalence gate": row.maximum_absolute_attack_prevalence_deviation <= 0.01,
            "both classes present": bool(row.both_classes_in_all_partitions),
            "map hash matches": (
                hash_file(args.output_dir / Path(row.split_map_path))
                == row.split_map_sha256
            ),
        }
        for name, passed in values.items():
            checks.append({"target": row.target, "seed": row.seed, "check": prefix + " " + name, "status": "Pass" if passed else "Fail"})
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(args.output_dir / "split_validation_checks.csv", index=False)
    if len(rows) != 9 or (check_frame["status"] != "Pass").any():
        raise RuntimeError("Corrected split consolidation failed")
    summary = rows[[
        "target", "seed", "train_rows", "val_rows", "test_rows", "drop_rows",
        "moved_retained_fraction", "maximum_absolute_partition_fraction_deviation",
        "maximum_absolute_attack_prevalence_deviation", "split_map_sha256",
    ]]
    report = "\n".join([
        "# Corrected target-only oracle split report",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        "All nine maps pass duplicate-free, timestamped-session-disjoint, partition-fraction, prevalence, and class-presence gates.",
        "",
        summary.to_markdown(index=False),
        "",
        "These maps are frozen common inputs for RF, XGB, and MLP. No model outcome was used in their construction.",
        "",
    ])
    (args.output_dir / "CORRECTED_ORACLE_SPLIT_REPORT.md").write_text(report, encoding="utf-8")
    atomic_json(args.output_dir / "SPLITS_COMPLETED.json", {
        "protocol_id": PROTOCOL_ID,
        "completed_utc": utc_now(),
        "split_maps": 9,
        "training_runs": 0,
        "validation_checks": len(check_frame),
        "validation_failures": int((check_frame["status"] != "Pass").sum()),
    })


def generate(args: argparse.Namespace) -> None:
    verify_execution_lock(args)
    execution_lock_sha = hash_file(args.output_dir / "execution_lock.json")
    paths = paths_from_parent(args)
    units = []
    for target in audit.TARGETS:
        units.append(generate_target(args, target, paths["targets"][target], execution_lock_sha))
    consolidate(args, units)


def plan(args: argparse.Namespace) -> None:
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "mode": "split_generation_only_no_training",
        "targets": list(audit.TARGETS),
        "seeds": list(audit.SEEDS),
        "maps": 9,
        "training_calls": 0,
        "parent_input_lock_sha256": EXPECTED_PARENT_LOCK_SHA256,
        "output_dir": str(args.output_dir),
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="plan", choices=("plan", "lock", "generate", "all"))
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--historical-oracle-root",
        type=Path,
        default=REPOSITORY_ROOT / "external" / "historical_oracle",
    )
    parser.add_argument(
        "--parent-audit-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "generated" / "oracle_audit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "generated" / "oracle_splits",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=REPOSITORY_ROOT / "protocols" / "oracle" / "split_policy.json",
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
        raise SystemExit("lock/generate require --execute")
    from cross_domain_nids.release_paths import require_generated_output

    args.output_dir = require_generated_output(args.output_dir, "results")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log("training_calls=0; split generation only")
    if args.command in ("lock", "all"):
        lock_execution(args)
    if args.command in ("generate", "all"):
        generate(args)


if __name__ == "__main__":
    main()
