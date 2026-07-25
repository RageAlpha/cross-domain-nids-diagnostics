"""
Target-only oracle baselines using full target datasets.

This runner trains supervised reference-ceiling models separately inside each
target dataset. It never reads UNSW, never runs adaptation, and reports metrics
only on each target held-out test split.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from cross_domain_nids.config import (
    ATTACK_COL,
    BOTIOT_PATH,
    CATEGORICAL_FEATURES,
    CICIDS_PATH,
    LABEL_COL,
    NUMERIC_FEATURES,
    PROTOCOL_FEATURES,
    STRESS_B_CHUNK_SIZE,
    TONIOT_PATH,
)
from cross_domain_nids.models.pipeline_models import TorchMLPClassifier
from cross_domain_nids.preprocessing.netflow import clean_l7_proto


SEEDS = [42, 123, 2024]
TARGET_DATASETS = {
    "CICIDS2018": CICIDS_PATH,
    "BoT-IoT": BOTIOT_PATH,
    "ToN-IoT": TONIOT_PATH,
}
MODELS = ["RandomForest", "XGBoost", "TorchMLP"]
METRIC_COLUMNS = ["binary_f1", "mcc", "fpr", "fnr", "weighted_f1"]
SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}
CODE_TO_SPLIT = np.array(["train", "val", "test"], dtype=object)

FIXED_PROTOCOL_VALUES = {
    "experiment": "Oracle",
    "filtering_strategy": "none",
    "oracle_protocol": "target_only_supervised_reference_ceiling",
    "preprocessing_fit_scope": "target_train_only",
    "label_protocol": "binary_Benign_vs_Attack",
    "hyperparameter_tuning": "none_reused_E2_binary_params",
    "training_data_scope": "target_train_only",
    "evaluation_scope": "target_heldout_test_stream",
}


def print_flush(message):
    print(message, flush=True)


def set_all_seeds(seed):
    from cross_domain_nids.determinism import configure_determinism

    configure_determinism(seed)


def feature_columns_from_frame(df):
    configured = NUMERIC_FEATURES + PROTOCOL_FEATURES + CATEGORICAL_FEATURES
    return [col for col in configured if col in df.columns]


def build_preprocessor_for_features(feature_cols):
    numeric = [col for col in NUMERIC_FEATURES if col in feature_cols]
    protocol = [col for col in PROTOCOL_FEATURES if col in feature_cols]
    categorical = [col for col in CATEGORICAL_FEATURES if col in feature_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    protocol_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        (
            "encoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric),
        ("protocol", protocol_pipeline, protocol),
        ("cat", categorical_pipeline, categorical),
    ], remainder="drop")


def load_target_frame(dataset_path):
    required_columns = set(
        NUMERIC_FEATURES
        + PROTOCOL_FEATURES
        + CATEGORICAL_FEATURES
        + [LABEL_COL, ATTACK_COL]
    )
    df = pd.read_csv(
        dataset_path,
        usecols=lambda col: col in required_columns,
        low_memory=False,
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    return clean_l7_proto(df)


def derive_binary_labels(df):
    if LABEL_COL in df.columns:
        numeric = pd.to_numeric(df[LABEL_COL], errors="coerce")
        values = set(numeric.dropna().unique().tolist())
        if numeric.notna().all() and values.issubset({0, 1, 0.0, 1.0}):
            return numeric.astype(int), "Label_binary"

    if ATTACK_COL not in df.columns:
        raise ValueError(
            f"Cannot derive binary labels: missing both usable {LABEL_COL} "
            f"and {ATTACK_COL} columns."
        )
    return (df[ATTACK_COL].astype(str) != "Benign").astype(int), "Attack_to_binary"


def safe_dataset_name(name):
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")


def split_map_path(dataset_name, seed, args):
    return Path(args.split_map_dir) / (
        f"oracle_split_map_{safe_dataset_name(dataset_name)}_seed{seed}.csv"
    )


def split_map_metadata_path(path):
    return Path(path).with_suffix(".json")


def label_usecols(col):
    return col in {LABEL_COL, ATTACK_COL}


def read_target_labels_stream(dataset_name, dataset_path, chunk_size):
    labels = []
    label_source = None
    total = 0
    for chunk in pd.read_csv(
        dataset_path,
        usecols=label_usecols,
        chunksize=chunk_size,
        low_memory=False,
    ):
        y_chunk, chunk_label_source = derive_binary_labels(chunk)
        if label_source is None:
            label_source = chunk_label_source
        elif label_source != chunk_label_source:
            raise ValueError(
                f"Inconsistent label source while reading {dataset_name}: "
                f"{label_source} then {chunk_label_source}."
            )
        labels.append(y_chunk.to_numpy(dtype=np.uint8, copy=False))
        total += len(chunk)
        if len(labels) % 5 == 0:
            print_flush(f"{dataset_name}: scanned {total:,} label rows")
        del chunk, y_chunk
        gc.collect()

    if not labels:
        raise ValueError(f"No rows found while reading labels for {dataset_name}.")
    return np.concatenate(labels), label_source


def make_stratified_split_codes(labels, seed):
    """Create deterministic stratified 70/10/20 split codes without feature data."""
    row_ids = np.arange(len(labels), dtype=np.int64)
    split_codes = np.empty(len(labels), dtype=np.uint8)
    train_val_ids, test_ids, train_val_y, _ = train_test_split(
        row_ids,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=0.1 / 0.8,
        stratify=train_val_y,
        random_state=seed,
    )
    split_codes[train_ids] = SPLIT_TO_CODE["train"]
    split_codes[val_ids] = SPLIT_TO_CODE["val"]
    split_codes[test_ids] = SPLIT_TO_CODE["test"]
    return split_codes


def write_split_map_csv(path, labels, split_codes, chunk_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for start in range(0, len(labels), chunk_size):
        end = min(start + chunk_size, len(labels))
        frame = pd.DataFrame({
            "row_id": np.arange(start, end, dtype=np.int64),
            "binary_label": labels[start:end].astype(np.uint8, copy=False),
            "split": CODE_TO_SPLIT[split_codes[start:end]],
        })
        frame.to_csv(
            path,
            mode="w" if start == 0 else "a",
            header=start == 0,
            index=False,
        )
        del frame
    return path


def load_split_map_csv(path):
    labels = []
    split_codes = []
    expected_start = 0
    for chunk in pd.read_csv(path, chunksize=STRESS_B_CHUNK_SIZE):
        row_ids = chunk["row_id"].to_numpy(dtype=np.int64, copy=False)
        if len(row_ids) and (row_ids[0] != expected_start or row_ids[-1] != expected_start + len(row_ids) - 1):
            raise ValueError(f"Split map row_id values are not contiguous in {path}.")
        expected_start += len(row_ids)
        labels.append(chunk["binary_label"].to_numpy(dtype=np.uint8, copy=False))
        split_codes.append(
            chunk["split"]
            .map(SPLIT_TO_CODE)
            .to_numpy(dtype=np.uint8, copy=False)
        )
        del chunk
    if not labels:
        raise ValueError(f"Split map is empty: {path}")
    return np.concatenate(labels), np.concatenate(split_codes)


def split_sizes_from_codes(split_codes):
    return {
        "train": int((split_codes == SPLIT_TO_CODE["train"]).sum()),
        "val": int((split_codes == SPLIT_TO_CODE["val"]).sum()),
        "test": int((split_codes == SPLIT_TO_CODE["test"]).sum()),
    }


def split_attack_rates(labels, split_codes):
    rates = {}
    for split_name, split_code in SPLIT_TO_CODE.items():
        mask = split_codes == split_code
        rates[split_name] = float(labels[mask].mean()) if mask.any() else np.nan
    return rates


def build_or_load_split_map(dataset_name, dataset_path, seed, args):
    """
    Read only label columns from the full target dataset, create a stratified
    70/10/20 train/val/test split, and save row_id -> split mapping.
    """
    path = split_map_path(dataset_name, seed, args)
    metadata_path = split_map_metadata_path(path)
    if path.exists() and not args.rebuild_split_maps:
        print_flush(f"Loading existing oracle split map: {path}")
        labels, split_codes = load_split_map_csv(path)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            label_source = metadata.get("label_source", "split_map_binary_label")
        else:
            label_source = "split_map_binary_label"
    else:
        print_flush(f"Building oracle split map: target={dataset_name}, seed={seed}")
        labels, label_source = read_target_labels_stream(
            dataset_name,
            dataset_path,
            args.chunk_size,
        )
        split_codes = make_stratified_split_codes(labels, seed)
        write_split_map_csv(path, labels, split_codes, args.chunk_size)
        metadata_path.write_text(
            json.dumps(
                {
                    "dataset_name": dataset_name,
                    "dataset_path": str(dataset_path),
                    "seed": int(seed),
                    "label_source": label_source,
                    "split_counts": split_sizes_from_codes(split_codes),
                    "protocol": "stratified_70_10_20_target_oracle_split_map",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print_flush(f"Saved oracle split map: {path}")

    sizes = split_sizes_from_codes(split_codes)
    attack_rates = split_attack_rates(labels, split_codes)
    print_flush(
        f"{dataset_name} seed={seed} split map: "
        f"train={sizes['train']:,}, val={sizes['val']:,}, test={sizes['test']:,}"
    )
    return {
        "path": str(path),
        "binary_labels": labels,
        "split_codes": split_codes,
        "label_source": label_source,
        "sizes": sizes,
        "attack_rates": attack_rates,
        "full_target_rows": int(len(split_codes)),
        "seed": int(seed),
    }


def target_feature_usecols(col):
    return col in set(
        NUMERIC_FEATURES
        + PROTOCOL_FEATURES
        + CATEGORICAL_FEATURES
        + [LABEL_COL, ATTACK_COL]
    )


def align_feature_frame(frame, feature_cols):
    present = [col for col in feature_cols if col in frame.columns]
    X = frame[present].copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    return X[feature_cols]


def load_train_val_from_split_map(dataset_name, dataset_path, split_map, args):
    print_flush(f"Streaming target train/val rows: {dataset_name} ({dataset_path})")

    train_parts = []
    val_parts = []
    y_train_parts = []
    y_val_parts = []
    feature_cols = None
    row_offset = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            dataset_path,
            usecols=target_feature_usecols,
            chunksize=args.chunk_size,
            low_memory=False,
        ),
        start=1,
    ):
        n_rows = len(chunk)
        next_offset = row_offset + n_rows
        chunk_splits = split_map["split_codes"][row_offset:next_offset]
        chunk_labels = split_map["binary_labels"][row_offset:next_offset]
        if len(chunk_splits) != n_rows:
            raise ValueError(
                f"Split map for {dataset_name} has fewer rows than target CSV."
            )

        keep_mask = chunk_splits != SPLIT_TO_CODE["test"]
        if keep_mask.any():
            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk = clean_l7_proto(chunk)
            X_base = chunk.drop(
                columns=[col for col in [LABEL_COL, ATTACK_COL] if col in chunk.columns]
            )
            if feature_cols is None:
                feature_cols = feature_columns_from_frame(X_base)
                if not feature_cols:
                    raise ValueError(
                        f"No configured feature columns are present in {dataset_name}."
                    )
            X_chunk = align_feature_frame(X_base, feature_cols)

            train_mask = chunk_splits == SPLIT_TO_CODE["train"]
            val_mask = chunk_splits == SPLIT_TO_CODE["val"]
            if train_mask.any():
                train_parts.append(X_chunk.iloc[train_mask].copy())
                y_train_parts.append(pd.Series(chunk_labels[train_mask], dtype=int))
            if val_mask.any():
                val_parts.append(X_chunk.iloc[val_mask].copy())
                y_val_parts.append(pd.Series(chunk_labels[val_mask], dtype=int))
            del X_base, X_chunk

        if chunk_idx % 5 == 0:
            print_flush(f"{dataset_name}: streamed {next_offset:,} rows for train/val")
        row_offset = next_offset
        del chunk
        gc.collect()

    if row_offset != len(split_map["split_codes"]):
        raise ValueError(
            f"Split map row count {len(split_map['split_codes']):,} does not match "
            f"target CSV row count {row_offset:,} for {dataset_name}."
        )
    if feature_cols is None:
        raise ValueError(f"No train/val feature rows loaded for {dataset_name}.")

    X_train = pd.concat(train_parts, ignore_index=True)
    X_val = pd.concat(val_parts, ignore_index=True)
    y_train = pd.concat(y_train_parts, ignore_index=True)
    y_val = pd.concat(y_val_parts, ignore_index=True)
    print_flush(
        f"{dataset_name} seed={split_map['seed']} split: "
        f"train={len(X_train):,}, val={len(X_val):,}, "
        f"test={split_map['sizes']['test']:,}, features={len(feature_cols)}, "
        f"label_source={split_map['label_source']}"
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train.astype(int),
        "y_val": y_val.astype(int),
        "feature_cols": feature_cols,
        "label_source": split_map["label_source"],
        "split_map": split_map,
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(split_map["sizes"]["test"]),
    }


def load_target_splits(dataset_name, dataset_path, seed, args):
    split_map = build_or_load_split_map(dataset_name, dataset_path, seed, args)
    return load_train_val_from_split_map(dataset_name, dataset_path, split_map, args)


def as_float32(X):
    if hasattr(X, "astype"):
        return X.astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def preprocess_splits(data):
    preprocessor = build_preprocessor_for_features(data["feature_cols"])
    print_flush("Fitting preprocessor on target training split only.")
    X_train_t = as_float32(preprocessor.fit_transform(data["X_train"]))
    X_val_t = as_float32(preprocessor.transform(data["X_val"]))
    return preprocessor, X_train_t, X_val_t


def build_oracle_model(model_name, seed):
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda",
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "TorchMLP":
        return TorchMLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=0.001,
            learning_rate=0.001,
            batch_size=4096,
            max_epochs=30,
            patience=5,
            validation_fraction=0.1,
            require_gpu=True,
            random_state=seed,
            verbose=False,
        )
    raise ValueError(f"Unknown oracle model: {model_name}")


def fit_oracle_model(model_name, model, X_train, y_train, X_val, y_val):
    if model_name == "TorchMLP":
        model.fit(X_train, y_train, validation_data=(X_val, y_val))
    else:
        model.fit(X_train, y_train)
    return model


def binary_metrics_from_counts(counts):
    tp = int(counts["tp"])
    tn = int(counts["tn"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    n = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    binary_f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / (denominator ** 0.5) if denominator else 0.0

    benign_precision = tn / (tn + fn) if (tn + fn) else 0.0
    benign_recall = tn / (tn + fp) if (tn + fp) else 0.0
    benign_f1 = (
        2 * benign_precision * benign_recall / (benign_precision + benign_recall)
        if (benign_precision + benign_recall)
        else 0.0
    )
    weighted_f1 = (((tn + fp) * benign_f1) + ((tp + fn) * binary_f1)) / n if n else 0.0
    return {
        "binary_f1": binary_f1,
        "mcc": mcc,
        "fpr": fpr,
        "fnr": fnr,
        "weighted_f1": weighted_f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def update_counts(counts, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    counts["tp"] += int(((y_true == 1) & (y_pred == 1)).sum())
    counts["tn"] += int(((y_true == 0) & (y_pred == 0)).sum())
    counts["fp"] += int(((y_true == 0) & (y_pred == 1)).sum())
    counts["fn"] += int(((y_true == 1) & (y_pred == 0)).sum())


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "binary_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "fpr": fpr,
        "fnr": fnr,
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=[0, 1],
            average="weighted",
            zero_division=0,
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_oracle_test_stream(
    model,
    preprocessor,
    dataset_name,
    dataset_path,
    split_map,
    feature_cols,
    chunk_size,
):
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    row_offset = 0
    test_seen = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            dataset_path,
            usecols=target_feature_usecols,
            chunksize=chunk_size,
            low_memory=False,
        ),
        start=1,
    ):
        n_rows = len(chunk)
        next_offset = row_offset + n_rows
        chunk_splits = split_map["split_codes"][row_offset:next_offset]
        chunk_labels = split_map["binary_labels"][row_offset:next_offset]
        if len(chunk_splits) != n_rows:
            raise ValueError(
                f"Split map for {dataset_name} has fewer rows than target CSV."
            )

        test_mask = chunk_splits == SPLIT_TO_CODE["test"]
        if test_mask.any():
            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk = clean_l7_proto(chunk)
            X_base = chunk.drop(
                columns=[col for col in [LABEL_COL, ATTACK_COL] if col in chunk.columns]
            )
            X_chunk = align_feature_frame(X_base, feature_cols)
            X_test = X_chunk.iloc[test_mask]
            X_test_t = as_float32(preprocessor.transform(X_test))
            y_pred = model.predict(X_test_t)
            y_true = chunk_labels[test_mask]
            update_counts(counts, y_true, y_pred)
            test_seen += int(test_mask.sum())
            del X_base, X_chunk, X_test, X_test_t, y_pred, y_true

        if chunk_idx % 5 == 0:
            metrics_so_far = binary_metrics_from_counts(counts)
            print_flush(
                f"{dataset_name}: streamed {next_offset:,} rows, "
                f"test_seen={test_seen:,}, "
                f"Binary-F1={metrics_so_far['binary_f1']:.4f}"
            )
        row_offset = next_offset
        del chunk
        gc.collect()

    if row_offset != len(split_map["split_codes"]):
        raise ValueError(
            f"Split map row count {len(split_map['split_codes']):,} does not match "
            f"target CSV row count {row_offset:,} for {dataset_name}."
        )
    expected_test = split_map["sizes"]["test"]
    if test_seen != expected_test:
        raise ValueError(
            f"Oracle test stream saw {test_seen:,} test rows for {dataset_name}, "
            f"expected {expected_test:,}."
        )
    return binary_metrics_from_counts(counts)


def make_result_row(dataset_name, seed, model_name, metrics, data, timings, artifact_path):
    split_map = data["split_map"]
    row = {
        **FIXED_PROTOCOL_VALUES,
        "seed": int(seed),
        "model": model_name,
        "target_dataset": dataset_name,
        "train_size": int(data["train_size"]),
        "val_size": int(data["val_size"]),
        "test_size": int(data["test_size"]),
        "full_target_rows": int(split_map["full_target_rows"]),
        "label_source": data["label_source"],
        "feature_count": int(len(data["feature_cols"])),
        "split_map_path": split_map["path"],
        "split_source": "full_target_dataset_stratified_70_10_20",
        "test_evaluation_mode": "chunk_streaming_accumulated_counts",
        "chunk_size": timings["chunk_size"],
        "train_attack_rate": split_map["attack_rates"]["train"],
        "val_attack_rate": split_map["attack_rates"]["val"],
        "test_attack_rate": split_map["attack_rates"]["test"],
        "split_map_train_size": int(split_map["sizes"]["train"]),
        "split_map_val_size": int(split_map["sizes"]["val"]),
        "split_map_test_size": int(split_map["sizes"]["test"]),
        "preprocessing_validation_scope": "target_validation_transform_only",
        "early_stopping_scope": (
            "explicit_target_validation_split"
            if model_name == "TorchMLP"
            else "not_applicable"
        ),
        "train_time_s": timings["train_time_s"],
        "inference_time_s": timings["inference_time_s"],
        "artifact": str(artifact_path) if artifact_path else "",
    }
    row.update(metrics)
    return row


def save_artifact(path, preprocessor, model, data, seed, dataset_name, model_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
            "feature_cols": data["feature_cols"],
            "metadata": {
                "experiment": "Oracle",
                "dataset": dataset_name,
                "seed": seed,
                "model": model_name,
                "protocol": FIXED_PROTOCOL_VALUES,
                "split_map": data["split_map"]["path"],
            },
        },
        path,
    )
    return path


def run_one(dataset_name, dataset_path, seed, model_name, args):
    set_all_seeds(seed)
    data = load_target_splits(dataset_name, dataset_path, seed, args)
    preprocessor, X_train_t, X_val_t = preprocess_splits(data)

    model = build_oracle_model(model_name, seed)
    print_flush(f"Training Oracle {model_name}: target={dataset_name}, seed={seed}")
    t0 = time.time()
    model = fit_oracle_model(
        model_name,
        model,
        X_train_t,
        data["y_train"],
        X_val_t,
        data["y_val"],
    )
    train_time = time.time() - t0

    print_flush(
        f"Evaluating target held-out test split by streaming: "
        f"{dataset_name}, seed={seed}"
    )
    t0 = time.time()
    metrics = evaluate_oracle_test_stream(
        model=model,
        preprocessor=preprocessor,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        split_map=data["split_map"],
        feature_cols=data["feature_cols"],
        chunk_size=args.chunk_size,
    )
    inference_time = time.time() - t0

    artifact_path = ""
    if args.save_artifacts:
        artifact_path = (
            Path(args.artifact_dir)
            / dataset_name
            / f"seed{seed}"
            / f"{model_name}.joblib"
        )
        save_artifact(artifact_path, preprocessor, model, data, seed, dataset_name, model_name)

    row = make_result_row(
        dataset_name,
        seed,
        model_name,
        metrics,
        data,
        {
            "train_time_s": train_time,
            "inference_time_s": inference_time,
            "chunk_size": args.chunk_size,
        },
        artifact_path,
    )

    del data, preprocessor, X_train_t, X_val_t, model
    gc.collect()
    return row


def write_outputs(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "oracle_all_seeds_long.csv"
    summary_path = output_dir / "oracle_mean_std_summary.csv"

    long_df = pd.DataFrame(rows)
    leading = [
        "experiment",
        "seed",
        "model",
        "target_dataset",
        "filtering_strategy",
        "binary_f1",
        "mcc",
        "fpr",
        "fnr",
        "weighted_f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "train_size",
        "val_size",
        "test_size",
        "full_target_rows",
        "oracle_protocol",
        "preprocessing_fit_scope",
        "label_protocol",
        "hyperparameter_tuning",
        "training_data_scope",
        "evaluation_scope",
        "split_map_path",
        "split_source",
        "test_evaluation_mode",
        "chunk_size",
        "train_attack_rate",
        "val_attack_rate",
        "test_attack_rate",
    ]
    for col in leading:
        if col not in long_df.columns:
            long_df[col] = np.nan
    trailing = [col for col in long_df.columns if col not in leading]
    long_df = long_df[leading + trailing]
    long_df.to_csv(long_path, index=False)

    summary = (
        long_df
        .groupby(["experiment", "model", "target_dataset", "filtering_strategy"])[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(summary_path, index=False)
    return long_df, long_path, summary_path


def validate_outputs(long_df, allow_incomplete=False):
    errors = []
    expected_seeds = set(SEEDS)
    expected_targets = set(TARGET_DATASETS)
    expected_models = set(MODELS)

    actual_targets = set(long_df["target_dataset"].dropna().unique())
    actual_models = set(long_df["model"].dropna().unique())
    actual_seeds = set(long_df["seed"].dropna().astype(int).unique())

    if not allow_incomplete:
        if actual_targets != expected_targets:
            errors.append(f"Targets are {sorted(actual_targets)}, expected {sorted(expected_targets)}.")
        if actual_models != expected_models:
            errors.append(f"Models are {sorted(actual_models)}, expected {sorted(expected_models)}.")
        if actual_seeds != expected_seeds:
            errors.append(f"Seeds are {sorted(actual_seeds)}, expected {sorted(expected_seeds)}.")

    for (target, model), group in long_df.groupby(["target_dataset", "model"]):
        seeds = set(group["seed"].astype(int).tolist())
        if not allow_incomplete and seeds != expected_seeds:
            errors.append(f"{target} x {model} has seeds {sorted(seeds)}, expected {sorted(expected_seeds)}.")

    ranges = {
        "binary_f1": (0, 1),
        "weighted_f1": (0, 1),
        "fpr": (0, 1),
        "fnr": (0, 1),
        "mcc": (-1, 1),
    }
    for metric, (low, high) in ranges.items():
        values = pd.to_numeric(long_df[metric], errors="coerce")
        bad = values.isna() | (values < low) | (values > high)
        if bad.any():
            errors.append(f"{metric} has {int(bad.sum())} missing/out-of-range values.")

    count_sum = long_df[["tp", "tn", "fp", "fn"]].sum(axis=1)
    mismatched = count_sum.astype(int) != long_df["test_size"].astype(int)
    if mismatched.any():
        errors.append(f"{int(mismatched.sum())} rows have confusion counts not equal to test_size.")

    size_sum = long_df[["train_size", "val_size", "test_size"]].sum(axis=1)
    full_size_mismatched = size_sum.astype(int) != long_df["full_target_rows"].astype(int)
    if full_size_mismatched.any():
        errors.append(
            f"{int(full_size_mismatched.sum())} rows have split sizes not equal to full_target_rows."
        )

    split_map_sum = long_df[[
        "split_map_train_size",
        "split_map_val_size",
        "split_map_test_size",
    ]].sum(axis=1)
    split_map_mismatched = split_map_sum.astype(int) != long_df["full_target_rows"].astype(int)
    if split_map_mismatched.any():
        errors.append(
            f"{int(split_map_mismatched.sum())} rows have split-map sizes not equal to full_target_rows."
        )

    for col in ["train_attack_rate", "val_attack_rate", "test_attack_rate"]:
        values = pd.to_numeric(long_df[col], errors="coerce")
        bad = values.isna() | (values < 0) | (values > 1)
        if bad.any():
            errors.append(f"{col} has {int(bad.sum())} missing/out-of-range values.")

    expected_metadata = {
        "split_source": "full_target_dataset_stratified_70_10_20",
        "test_evaluation_mode": "chunk_streaming_accumulated_counts",
    }
    for col, expected in expected_metadata.items():
        values = set(long_df[col].dropna().unique())
        if values != {expected}:
            errors.append(f"{col} values are {sorted(values)}, expected only {expected}.")

    torch_mlp_scope = set(
        long_df.loc[long_df["model"].eq("TorchMLP"), "early_stopping_scope"]
        .dropna()
        .unique()
    )
    if torch_mlp_scope and torch_mlp_scope != {"explicit_target_validation_split"}:
        errors.append(
            "TorchMLP early_stopping_scope must be explicit_target_validation_split; "
            f"found {sorted(torch_mlp_scope)}."
        )

    for col, expected in FIXED_PROTOCOL_VALUES.items():
        values = set(long_df[col].dropna().unique())
        if values != {expected}:
            errors.append(f"{col} values are {sorted(values)}, expected only {expected}.")

    if "UNSW" in actual_targets:
        errors.append("UNSW appeared in oracle outputs; target-only oracle must exclude UNSW.")

    if errors:
        raise ValueError("Oracle validation failed:\n- " + "\n- ".join(errors))

    return {
        "seeds": [int(seed) for seed in sorted(actual_seeds)],
        "target_datasets": sorted(actual_targets),
        "models": sorted(actual_models),
        "rows": int(len(long_df)),
        "metric_range_checks": "passed",
        "confusion_count_checks": "passed",
        "protocol_checks": "passed",
        "leakage_checks": {
            "unsw_used": False,
            "source_trained_models_used": False,
            "dann_used": False,
            "pseudo_labeling_used": False,
            "target_test_used_for_preprocessing_fit": False,
            "target_test_used_for_early_stopping": False,
            "full_target_evaluation_after_training": False,
            "target_test_streaming_only": True,
            "grid_search_or_hyperparameter_tuning": False,
        },
    }


def write_report(long_df, validation_report, output_dir, long_path, summary_path):
    split_sizes = (
        long_df[
            ["target_dataset", "seed", "train_size", "val_size", "test_size"]
        ]
        .drop_duplicates()
        .sort_values(["target_dataset", "seed"])
    )
    report = [
        "Target-only oracle baselines were trained separately within each target dataset using supervised target labels.",
        "For each target dataset, label-only split maps were built first and saved as row_id/binary_label/split CSV files.",
        "The split maps define stratified 70/10/20 train/validation/test splits under seeds 42, 123, and 2024.",
        "Model hyperparameters were reused from the verified E2 binary-source protocol.",
        "Preprocessing was fit only on the target-domain training split and applied to validation and held-out test splits.",
        "Final metrics were computed by streaming the full target CSV and accumulating counts only for rows whose split_map entry is test.",
        "Oracle results are reported only as supervised reference ceilings for recoverable target-domain performance and are not used in any source-only, feature-filtering, or DANN adaptation experiment.",
        "",
        f"Datasets used: {', '.join(sorted(TARGET_DATASETS))}",
        f"Models used: {', '.join(MODELS)}",
        f"Seeds used: {', '.join(map(str, SEEDS))}",
        "",
        "Split sizes:",
        split_sizes.to_string(index=False),
        "",
        f"Long-format output: {long_path}",
        f"Mean/std summary output: {summary_path}",
        "",
        "Validation results:",
        json.dumps(validation_report, indent=2, sort_keys=True),
    ]
    report_path = Path(output_dir) / "oracle_validation_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run target-only oracle baselines on full target datasets."
    )
    parser.add_argument("--output-dir", default="results/generated/oracle")
    parser.add_argument("--artifact-dir", default="artifacts/oracle")
    parser.add_argument(
        "--split-map-dir",
        default="results/generated/oracle/split_maps",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--only-dataset", choices=list(TARGET_DATASETS.keys()))
    parser.add_argument("--only-seed", type=int, choices=SEEDS)
    parser.add_argument("--only-model", choices=MODELS)
    parser.add_argument("--rebuild-split-maps", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def requested_runs(args):
    datasets = (
        {args.only_dataset: TARGET_DATASETS[args.only_dataset]}
        if args.only_dataset
        else TARGET_DATASETS
    )
    seeds = [args.only_seed] if args.only_seed is not None else SEEDS
    models = [args.only_model] if args.only_model else MODELS
    for dataset_name, dataset_path in datasets.items():
        for seed in seeds:
            for model_name in models:
                yield dataset_name, dataset_path, seed, model_name


def load_existing_rows_for_resume(output_dir):
    path = Path(output_dir) / "oracle_all_seeds_long.csv"
    if not path.exists():
        return [], set()
    existing = pd.read_csv(path)
    completed = set()
    required = {"target_dataset", "seed", "model"}
    resume_schema = required | {
        "evaluation_scope",
        "split_map_path",
        "test_evaluation_mode",
        "full_target_rows",
    }
    missing_schema = sorted(resume_schema - set(existing.columns))
    if missing_schema:
        raise ValueError(
            f"Cannot --resume from {path}; missing new Oracle columns: {missing_schema}. "
            "Rerun without --resume or archive the old output first."
        )
    if required.issubset(existing.columns):
        completed = {
            (str(row.target_dataset), int(row.seed), str(row.model))
            for row in existing[list(required)].itertuples(index=False)
        }
    return existing.to_dict("records"), completed


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.split_map_dir).mkdir(parents=True, exist_ok=True)
    rows, completed = load_existing_rows_for_resume(output_dir) if args.resume else ([], set())
    runs = list(requested_runs(args))

    if args.dry_run:
        print_flush("Oracle dry run. Planned runs:")
        for dataset_name, _, seed, model_name in runs:
            status = "skip-resume" if (dataset_name, seed, model_name) in completed else "run"
            print_flush(
                f"- {dataset_name} seed={seed} model={model_name} "
                f"chunk_size={args.chunk_size} status={status}"
            )
        return

    try:
        for dataset_name, dataset_path, seed, model_name in runs:
            key = (dataset_name, seed, model_name)
            if args.resume and key in completed:
                print_flush(
                    f"Skipping completed Oracle run due to --resume: "
                    f"target={dataset_name}, seed={seed}, model={model_name}"
                )
                continue
            print_flush("\n" + "=" * 80)
            print_flush(
                f"Oracle run: target={dataset_name}, seed={seed}, model={model_name}"
            )
            print_flush("=" * 80)
            rows.append(run_one(dataset_name, dataset_path, seed, model_name, args))
            completed.add(key)
            long_df, long_path, summary_path = write_outputs(rows, output_dir)
            print_flush(f"Saved partial long results: {long_path}")

        if not rows:
            print_flush("No Oracle rows to write.")
            return

        long_df, long_path, summary_path = write_outputs(rows, output_dir)
        validation_report = validate_outputs(long_df, allow_incomplete=args.allow_incomplete)
        validation_json_path = output_dir / "oracle_validation_report.json"
        validation_json_path.write_text(
            json.dumps(validation_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        report_path = write_report(
            long_df,
            validation_report,
            output_dir,
            long_path,
            summary_path,
        )
        print_flush(f"Saved long-format output: {long_path}")
        print_flush(f"Saved mean/std summary: {summary_path}")
        print_flush(f"Saved validation JSON: {validation_json_path}")
        print_flush(f"Saved final report: {report_path}")
    except MemoryError as exc:
        raise RuntimeError(
            "Full target-only oracle run exhausted memory. No subset oracle was "
            "produced; rerun on a larger machine or implement disk-backed arrays "
            "for this environment."
        ) from exc


if __name__ == "__main__":
    main()
