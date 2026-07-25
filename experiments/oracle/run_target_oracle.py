"""Run the 27 corrected target-only oracle ceiling experiments."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
PROTOCOL_ID = "NIDS-TARGET-ORACLE-CORRECTED-CEILING-20260717-v1"
TARGETS = ("CICIDS2018", "BoT-IoT", "ToN-IoT")
SEEDS = (42, 123, 2024)
MODELS = ("RandomForest", "XGBoost", "TorchMLP")
TARGET_FILES = {
    "CICIDS2018": "NF-CICIDS2018-v3.csv",
    "BoT-IoT": "NF-BoT-IoT-v3.csv",
    "ToN-IoT": "NF-ToN-IoT-v3.csv",
}
MAP_STEMS = {
    "CICIDS2018": "cicids2018",
    "BoT-IoT": "bot_iot",
    "ToN-IoT": "ton_iot",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"{utc_now()}\t{message}", flush=True)


def hash_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def load_parent(project_root: Path):
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from cross_domain_nids import oracle_engine as module
    module.FIXED_PROTOCOL_VALUES = {
        "experiment": "CorrectedOracle",
        "filtering_strategy": "none",
        "oracle_protocol": "duplicate_free_session_disjoint_target_only_supervised_ceiling",
        "preprocessing_fit_scope": "corrected_target_train_only",
        "label_protocol": "binary_Benign_vs_Attack",
        "hyperparameter_tuning": "none_reused_E2_binary_params",
        "training_data_scope": "corrected_target_train_only",
        "evaluation_scope": "corrected_target_heldout_test_only",
    }
    return module


def split_map_path(split_root: Path, target: str, seed: int) -> Path:
    return split_root / f"oracle__{MAP_STEMS[target]}__seed_{seed}.npz"


def load_split_manifest(split_root: Path) -> pd.DataFrame:
    path = split_root / "manifest.csv"
    if not path.is_file():
        path = split_root.parent / "split_manifest.csv"
    if not path.is_file():
        raise RuntimeError(f"Missing corrected split manifest: {path}")
    frame = pd.read_csv(path)
    if len(frame) != 9:
        raise RuntimeError(f"Corrected split manifest has {len(frame)} rows, expected 9")
    return frame


def load_corrected_map(split_root: Path, manifest: pd.DataFrame, target: str, seed: int) -> dict[str, Any]:
    row = manifest[(manifest.target == target) & (manifest.seed.astype(int) == seed)]
    if len(row) != 1:
        raise RuntimeError(f"Corrected map manifest unit missing/duplicated: {target} seed={seed}")
    row = row.iloc[0]
    path = split_map_path(split_root, target, seed)
    if path.name != Path(row.split_map_path).name:
        raise RuntimeError(f"Corrected map path mismatch: {target} seed={seed}")
    actual_sha = hash_file(path)
    if actual_sha != row.split_map_sha256:
        raise RuntimeError(f"Corrected map hash mismatch: {target} seed={seed}")
    with np.load(path, allow_pickle=False) as archive:
        split_codes = archive["split_codes"].astype(np.uint8, copy=False)
        binary_labels = archive["binary_labels"].astype(np.uint8, copy=False)
    if len(split_codes) != len(binary_labels):
        raise RuntimeError("Split map code/label length mismatch")
    allowed = set(np.unique(split_codes).tolist())
    if not allowed.issubset({0, 1, 2, 3}):
        raise RuntimeError(f"Corrected map contains unknown codes: {allowed}")
    sizes = {name: int(np.count_nonzero(split_codes == code)) for name, code in (("train", 0), ("val", 1), ("test", 2), ("drop", 3))}
    rates = {name: float(binary_labels[split_codes == code].mean()) for name, code in (("train", 0), ("val", 1), ("test", 2))}
    expected_sizes = {"train": int(row.train_rows), "val": int(row.val_rows), "test": int(row.test_rows), "drop": int(row.drop_rows)}
    if sizes != expected_sizes:
        raise RuntimeError(f"Corrected map size mismatch: {target} seed={seed} {sizes} != {expected_sizes}")
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "binary_labels": binary_labels,
        "split_codes": split_codes,
        "sizes": sizes,
        "attack_rates": rates,
        "full_target_rows": int(len(split_codes)),
        "retained_target_rows": int(len(split_codes) - sizes["drop"]),
        "seed": int(seed),
        "label_source": "locked_corrected_split_binary_labels",
    }


def load_train_val(parent, dataset_name: str, dataset_path: Path, split_map: dict[str, Any], chunk_size: int) -> dict[str, Any]:
    log(f"streaming corrected train/val target={dataset_name}")
    train_parts = []
    val_parts = []
    y_train_parts = []
    y_val_parts = []
    feature_cols = None
    row_offset = 0
    for chunk_idx, chunk in enumerate(
        pd.read_csv(dataset_path, usecols=parent.target_feature_usecols, chunksize=chunk_size, low_memory=False), start=1
    ):
        n_rows = len(chunk)
        next_offset = row_offset + n_rows
        chunk_splits = split_map["split_codes"][row_offset:next_offset]
        chunk_labels = split_map["binary_labels"][row_offset:next_offset]
        if len(chunk_splits) != n_rows:
            raise RuntimeError(f"Corrected map shorter than target CSV: {dataset_name}")
        keep_mask = (chunk_splits == 0) | (chunk_splits == 1)
        if keep_mask.any():
            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk = parent.clean_l7_proto(chunk)
            x_base = chunk.drop(columns=[col for col in [parent.LABEL_COL, parent.ATTACK_COL] if col in chunk.columns])
            if feature_cols is None:
                feature_cols = parent.feature_columns_from_frame(x_base)
                if len(feature_cols) != 49:
                    raise RuntimeError(f"Expected 49 oracle features for {dataset_name}, found {len(feature_cols)}")
            x_chunk = parent.align_feature_frame(x_base, feature_cols)
            train_mask = chunk_splits == 0
            val_mask = chunk_splits == 1
            if train_mask.any():
                train_parts.append(x_chunk.iloc[train_mask].copy())
                y_train_parts.append(pd.Series(chunk_labels[train_mask], dtype=int))
            if val_mask.any():
                val_parts.append(x_chunk.iloc[val_mask].copy())
                y_val_parts.append(pd.Series(chunk_labels[val_mask], dtype=int))
            del x_base, x_chunk
        row_offset = next_offset
        if chunk_idx == 1 or chunk_idx % 5 == 0 or row_offset == len(split_map["split_codes"]):
            log(f"streamed train/val target={dataset_name} rows={row_offset}/{len(split_map['split_codes'])}")
        del chunk
        gc.collect()
    if row_offset != len(split_map["split_codes"]):
        raise RuntimeError(f"Corrected map/target row mismatch: {dataset_name}")
    x_train = pd.concat(train_parts, ignore_index=True)
    x_val = pd.concat(val_parts, ignore_index=True)
    y_train = pd.concat(y_train_parts, ignore_index=True).astype(int)
    y_val = pd.concat(y_val_parts, ignore_index=True).astype(int)
    if len(x_train) != split_map["sizes"]["train"] or len(x_val) != split_map["sizes"]["val"]:
        raise RuntimeError(f"Corrected train/val counts mismatch: {dataset_name}")
    return {
        "X_train": x_train,
        "X_val": x_val,
        "y_train": y_train,
        "y_val": y_val,
        "feature_cols": feature_cols,
        "label_source": split_map["label_source"],
        "split_map": split_map,
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(split_map["sizes"]["test"]),
    }


def artifact_path(output_dir: Path, target: str, seed: int, model: str) -> Path:
    return output_dir / "artifacts" / target / f"seed{seed}" / f"{model}.joblib"


def marker_path(output_dir: Path, target: str, seed: int, model: str) -> Path:
    return output_dir / "completions" / target / f"seed{seed}" / f"{model}.json"


def valid_marker(output_dir: Path, target: str, seed: int, model: str, map_sha: str) -> dict[str, Any] | None:
    path = marker_path(output_dir, target, seed, model)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    artifact = Path(value["result"]["artifact"])
    if (
        value.get("protocol_id") != PROTOCOL_ID
        or value.get("split_map_sha256") != map_sha
        or not artifact.is_file()
        or hash_file(artifact) != value.get("artifact_sha256")
    ):
        return None
    return value


def all_markers(output_dir: Path) -> list[dict[str, Any]]:
    values = []
    for path in output_dir.glob("completions/*/seed*/*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        artifact = Path(value["result"]["artifact"])
        if artifact.is_file() and hash_file(artifact) == value["artifact_sha256"]:
            values.append(value)
    return values


def write_partial_results(output_dir: Path) -> None:
    markers = all_markers(output_dir)
    rows = [value["result"] for value in markers]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        order_target = {name: idx for idx, name in enumerate(TARGETS)}
        order_model = {name: idx for idx, name in enumerate(MODELS)}
        frame["_target_order"] = frame.target_dataset.map(order_target)
        frame["_model_order"] = frame.model.map(order_model)
        frame = frame.sort_values(["_target_order", "seed", "_model_order"]).drop(columns=["_target_order", "_model_order"])
    path = output_dir / "oracle_seed_metrics.csv"
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def result_row(parent, target: str, seed: int, model_name: str, metrics: dict[str, Any], data: dict[str, Any], train_time: float, inference_time: float, artifact: Path, artifact_sha: str, chunk_size: int) -> dict[str, Any]:
    split_map = data["split_map"]
    return {
        **parent.FIXED_PROTOCOL_VALUES,
        "protocol_id": PROTOCOL_ID,
        "seed": int(seed),
        "model": model_name,
        "target_dataset": target,
        **metrics,
        "train_size": int(data["train_size"]),
        "val_size": int(data["val_size"]),
        "test_size": int(data["test_size"]),
        "drop_size": int(split_map["sizes"]["drop"]),
        "retained_target_rows": int(split_map["retained_target_rows"]),
        "full_target_rows": int(split_map["full_target_rows"]),
        "label_source": data["label_source"],
        "feature_count": int(len(data["feature_cols"])),
        "split_map_path": split_map["path"],
        "split_map_sha256": split_map["sha256"],
        "split_source": "minimal_repair_duplicate_free_timestamped_session_disjoint",
        "test_evaluation_mode": "chunk_streaming_corrected_test_only",
        "chunk_size": int(chunk_size),
        "train_attack_rate": split_map["attack_rates"]["train"],
        "val_attack_rate": split_map["attack_rates"]["val"],
        "test_attack_rate": split_map["attack_rates"]["test"],
        "preprocessing_validation_scope": "corrected_target_validation_transform_only",
        "early_stopping_scope": "explicit_corrected_target_validation_split" if model_name == "TorchMLP" else "not_applicable",
        "train_time_s": float(train_time),
        "inference_time_s": float(inference_time),
        "artifact": str(artifact.resolve()),
        "artifact_sha256": artifact_sha,
    }


def save_artifact(parent, path: Path, preprocessor: Any, model: Any, data: dict[str, Any], seed: int, target: str, model_name: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "preprocessor": preprocessor,
        "model": model,
        "feature_cols": data["feature_cols"],
        "metadata": {
            "protocol_id": PROTOCOL_ID,
            "experiment": "CorrectedOracle",
            "dataset": target,
            "seed": seed,
            "model": model_name,
            "protocol": parent.FIXED_PROTOCOL_VALUES,
            "split_map": data["split_map"]["path"],
            "split_map_sha256": data["split_map"]["sha256"],
        },
    }, path)
    return hash_file(path)


def run(args: argparse.Namespace) -> None:
    parent = load_parent(args.project_root)
    manifest = load_split_manifest(args.split_root)
    selected_targets = (args.only_target,) if args.only_target else TARGETS
    selected_seeds = (args.only_seed,) if args.only_seed is not None else SEEDS
    selected_models = (args.only_model,) if args.only_model else MODELS
    for target in selected_targets:
        dataset_path = args.data_dir / TARGET_FILES[target]
        for seed in selected_seeds:
            split_map = load_corrected_map(args.split_root, manifest, target, seed)
            pending = [model for model in selected_models if valid_marker(args.output_dir, target, seed, model, split_map["sha256"]) is None]
            if not pending:
                log(f"resume skip complete group target={target} seed={seed}")
                continue
            log(f"preparing group target={target} seed={seed} pending={pending}")
            parent.set_all_seeds(seed)
            data = load_train_val(parent, target, dataset_path, split_map, args.chunk_size)
            preprocessor, x_train_t, x_val_t = parent.preprocess_splits(data)
            data["X_train"] = None
            data["X_val"] = None
            gc.collect()
            for model_name in pending:
                parent.set_all_seeds(seed)
                model = parent.build_oracle_model(model_name, seed)
                log(f"training target={target} seed={seed} model={model_name}")
                start = time.time()
                model = parent.fit_oracle_model(model_name, model, x_train_t, data["y_train"], x_val_t, data["y_val"])
                train_time = time.time() - start
                log(f"evaluating target={target} seed={seed} model={model_name}")
                start = time.time()
                metrics = parent.evaluate_oracle_test_stream(
                    model=model,
                    preprocessor=preprocessor,
                    dataset_name=target,
                    dataset_path=dataset_path,
                    split_map=split_map,
                    feature_cols=data["feature_cols"],
                    chunk_size=args.chunk_size,
                )
                inference_time = time.time() - start
                artifact = artifact_path(args.output_dir, target, seed, model_name)
                artifact_sha = save_artifact(parent, artifact, preprocessor, model, data, seed, target, model_name)
                row = result_row(parent, target, seed, model_name, metrics, data, train_time, inference_time, artifact, artifact_sha, args.chunk_size)
                marker = {
                    "protocol_id": PROTOCOL_ID,
                    "completed_utc": utc_now(),
                    "target": target,
                    "seed": seed,
                    "model": model_name,
                    "split_map_sha256": split_map["sha256"],
                    "artifact_sha256": artifact_sha,
                    "result": row,
                }
                atomic_json(marker_path(args.output_dir, target, seed, model_name), marker)
                write_partial_results(args.output_dir)
                log(
                    f"completed target={target} seed={seed} model={model_name} "
                    f"MCC={metrics['mcc']:.6f} Binary-F1={metrics['binary_f1']:.6f}"
                )
                del model
                gc.collect()
            del data, preprocessor, x_train_t, x_val_t, split_map
            gc.collect()
    markers = all_markers(args.output_dir)
    write_partial_results(args.output_dir)
    log(f"training progress complete_units={len(markers)}/27")


def dry_run(args: argparse.Namespace) -> None:
    manifest = load_split_manifest(args.split_root)
    units = []
    for target in TARGETS:
        for seed in SEEDS:
            split_map = load_corrected_map(args.split_root, manifest, target, seed)
            for model in MODELS:
                units.append({
                    "target": target,
                    "seed": seed,
                    "model": model,
                    "split_map_sha256": split_map["sha256"],
                    "status": "complete" if valid_marker(args.output_dir, target, seed, model, split_map["sha256"]) else "pending",
                })
    print(json.dumps({"protocol_id": PROTOCOL_ID, "training_units": len(units), "units": units}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="dry-run", choices=("dry-run", "run"))
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "generated" / "oracle",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "splits" / "oracle",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--only-target", choices=TARGETS)
    parser.add_argument("--only-seed", type=int, choices=SEEDS)
    parser.add_argument("--only-model", choices=MODELS)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "dry-run":
        dry_run(args)
        return
    if not args.execute:
        raise SystemExit("run requires --execute")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
