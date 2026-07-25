"""
H1 multi-seed runner for E1-E4.

Protocol implemented here:
  - E1/E2: train and evaluate only seeds 123 and 2024, then merge existing
    verified seed 42 rows.
  - E3/E4: rerun seeds 42, 123, and 2024 under target-specific Scheme A.

The old combined-target E3/E4 seed 42 output is intentionally not read by this
script. All E3/E4 rows produced here use the same target-specific filtering
protocol across all three seeds.
"""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

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
    UNSW_PATH,
)
from cross_domain_nids.preprocessing.netflow import (
    build_preprocessor,
    clean_l7_proto,
    load_and_clean_data,
    prepare_cicids_chunk,
)
from cross_domain_nids.model_factory import DEFAULT_PARAMETERS_PATH, construct_estimator


E12_SEEDS_TO_RUN = [123, 2024]
E34_SEEDS_TO_RUN = [42, 123, 2024]
FINAL_SEEDS = [42, 123, 2024]

E1_E2_MODELS = ["RandomForest", "XGBoost", "MLP"]
E3_E4_MODELS = ["RandomForest", "MLP"]
SEED42_E1_UNSW_REPAIR_MODELS = ["RandomForest", "XGBoost", "MLP"]
TARGET_DATASETS = {
    "CICIDS2018": CICIDS_PATH,
    "BoT-IoT": BOTIOT_PATH,
    "ToN-IoT": TONIOT_PATH,
}
CANONICAL_DATASET_NAMES = {
    "UNSW": "UNSW",
    "NF-UNSW-NB15-v3": "UNSW",
    "CICIDS2018": "CICIDS2018",
    "CICIDS": "CICIDS2018",
    "NF-CSE-CIC-IDS2018-v3": "CICIDS2018",
    "BoT-IoT": "BoT-IoT",
    "Bot-IoT": "BoT-IoT",
    "BOT-IOT": "BoT-IoT",
    "NF-BoT-IoT-v3": "BoT-IoT",
    "ToN-IoT": "ToN-IoT",
    "TON-IOT": "ToN-IoT",
    "NF-ToN-IoT-v3": "ToN-IoT",
}
EXPERIMENT_NAME_MAP = {
    "E1": "E1",
    "E1_pure_transfer": "E1",
    "E2": "E2",
    "E2_binary_source_training": "E2",
    "E3": "E3",
    "E4": "E4",
}
SHIFT_TOP_K = [3, 5, 10, 15]
CORR_THRESHOLD = 0.98
SAMPLED_ROBUST_SHIFT_SCORE_COLUMN = "sampled_robust_shift_score"
FORMAL_FEATURE_SHIFT_COLUMNS = [
    "robust_shift_score",
    "standard_smd",
    "abs_standard_smd",
    "jsd",
]

REQUIRED_LONG_COLUMNS = [
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
]
REPAIR_METRIC_COLUMNS = [
    "binary_f1",
    "mcc",
    "fpr",
    "fnr",
    "weighted_f1",
    "tp",
    "tn",
    "fp",
    "fn",
    "n_samples",
]


def print_flush(message):
    print(message, flush=True)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric),
        ("protocol", protocol_pipeline, protocol),
        ("cat", categorical_pipeline, categorical),
    ], remainder="drop")


def load_unsw_splits(seed, task):
    """Return leakage-safe UNSW splits for either multiclass or binary source training."""
    print_flush(f"Loading UNSW source data for {task}, seed={seed}: {UNSW_PATH}")
    df = load_and_clean_data(UNSW_PATH)
    df = clean_l7_proto(df)

    if task == "multiclass":
        y = df[ATTACK_COL].astype(str)
        stratify = y
    elif task == "binary":
        y = df[LABEL_COL].astype(int)
        stratify = y
    else:
        raise ValueError(f"Unknown task: {task}")

    X = df.drop(columns=[col for col in [ATTACK_COL, LABEL_COL] if col in df.columns])
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=stratify,
        random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1 / 0.8,
        stratify=y_train_val,
        random_state=seed,
    )

    feature_cols = feature_columns_from_frame(X_train)
    print_flush(
        f"Split seed={seed}: train={len(X_train):,}, "
        f"val={len(X_val):,}, test={len(X_test):,}, features={len(feature_cols)}"
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }


def set_estimator_seed(estimator, seed):
    params = estimator.get_params(deep=False)
    updates = {}
    if "random_state" in params:
        updates["random_state"] = seed
    if "verbose" in params:
        updates["verbose"] = False
    if updates:
        estimator.set_params(**updates)
    return estimator


def construct_released_estimator(experiment, model_name, seed, parameters_path=None):
    """Construct an estimator from the released auditable parameter JSON."""
    path = Path(parameters_path) if parameters_path else DEFAULT_PARAMETERS_PATH
    return construct_estimator(experiment, model_name, seed, path)


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
    weighted_f1 = (((tp + fn) * binary_f1) + ((tn + fp) * benign_f1)) / n if n else 0.0

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
        "n_samples": n,
    }


def binary_metrics(y_true_bin, y_pred_bin):
    y_true_bin = np.asarray(y_true_bin, dtype=int)
    y_pred_bin = np.asarray(y_pred_bin, dtype=int)
    counts = {
        "tp": int(((y_true_bin == 1) & (y_pred_bin == 1)).sum()),
        "tn": int(((y_true_bin == 0) & (y_pred_bin == 0)).sum()),
        "fp": int(((y_true_bin == 0) & (y_pred_bin == 1)).sum()),
        "fn": int(((y_true_bin == 1) & (y_pred_bin == 0)).sum()),
    }
    return binary_metrics_from_counts(counts)


def update_counts(counts, y_true_bin, y_pred_bin):
    y_true_bin = np.asarray(y_true_bin, dtype=int)
    y_pred_bin = np.asarray(y_pred_bin, dtype=int)
    counts["tp"] += int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    counts["tn"] += int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    counts["fp"] += int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    counts["fn"] += int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())


def canonical_dataset_name(name):
    if pd.isna(name):
        return name
    return CANONICAL_DATASET_NAMES.get(str(name), str(name))


def canonical_experiment_name(name):
    if pd.isna(name):
        return name
    return EXPERIMENT_NAME_MAP.get(str(name), str(name))


def make_result_row(experiment, seed, model, target_dataset, filtering_strategy, metrics, **extra):
    row = {
        "experiment": canonical_experiment_name(experiment),
        "seed": int(seed),
        "model": model,
        "target_dataset": canonical_dataset_name(target_dataset),
        "filtering_strategy": filtering_strategy if filtering_strategy else "none",
    }
    row.update(metrics)
    row.update(extra)
    return row


def fit_pipeline(estimator, data, kept_features):
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor_for_features(kept_features)),
        ("model", estimator),
    ])
    t0 = time.time()
    pipeline.fit(data["X_train"][kept_features], data["y_train"])
    return pipeline, time.time() - t0


def fit_source_pipeline(estimator, data):
    """Fit E1/E2 with the same preprocessing construction used by seed 42."""
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])
    t0 = time.time()
    pipeline.fit(data["X_train"], data["y_train"])
    return pipeline, time.time() - t0, list(data["X_train"].columns)


def align_target_features(chunk, feature_cols):
    present = [col for col in feature_cols if col in chunk.columns]
    X_chunk = chunk[present].copy()
    for col in feature_cols:
        if col not in X_chunk.columns:
            X_chunk[col] = 0
    return X_chunk[feature_cols]


def evaluate_unsw_binary(pipeline, data, seed, experiment, model, filtering_strategy, task, kept_features):
    y_pred = pipeline.predict(data["X_test"][kept_features])
    if task == "multiclass":
        y_true_bin = (data["y_test"].astype(str) != "Benign").astype(int)
        y_pred_bin = (pd.Series(y_pred).astype(str) != "Benign").astype(int)
        prediction_mapping = "Benign->0; all_non_Benign_attack_classes->1"
    else:
        y_true_bin = data["y_test"].astype(int)
        y_pred_bin = pd.Series(y_pred).astype(int)
        prediction_mapping = "binary_label_0_Benign_1_Attack"
    metrics = binary_metrics(y_true_bin, y_pred_bin)
    return make_result_row(
        experiment,
        seed,
        model,
        "UNSW",
        filtering_strategy,
        metrics,
        evaluation_scope="heldout_test",
        metric_protocol="binary_detection_metrics",
        prediction_mapping=prediction_mapping,
    )


def evaluate_target_binary(
    pipeline,
    dataset_name,
    dataset_path,
    seed,
    experiment,
    model,
    filtering_strategy,
    task,
    kept_features,
    chunk_size,
):
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size, low_memory=False):
        chunk = prepare_cicids_chunk(chunk)
        X_chunk = align_target_features(chunk, kept_features)
        y_pred = pipeline.predict(X_chunk)
        if task == "multiclass":
            y_true_bin = (chunk[ATTACK_COL].astype(str) != "Benign").astype(int)
            y_pred_bin = (pd.Series(y_pred).astype(str) != "Benign").astype(int)
            prediction_mapping = "Benign->0; all_non_Benign_attack_classes->1"
        else:
            y_true_bin = chunk[LABEL_COL].astype(int)
            y_pred_bin = pd.Series(y_pred).astype(int)
            prediction_mapping = "binary_label_0_Benign_1_Attack"
        update_counts(counts, y_true_bin, y_pred_bin)
        del chunk, X_chunk, y_pred, y_true_bin, y_pred_bin
        gc.collect()

    metrics = binary_metrics_from_counts(counts)
    return make_result_row(
        experiment,
        seed,
        model,
        dataset_name,
        filtering_strategy,
        metrics,
        evaluation_scope="full_target",
        metric_protocol="binary_detection_metrics",
        prediction_mapping=prediction_mapping,
    )


def run_e1_e2_new_seed(seed, args):
    """Run only E1/E2 for a new seed. Seed 42 is loaded separately."""
    set_all_seeds(seed)
    rows = []

    e_specs = [
        ("E1", "multiclass", "pure_transfer_multiclass"),
        ("E2", "binary", "binary_source_training"),
    ]
    for experiment, task, training_protocol in e_specs:
        data = load_unsw_splits(seed, task)
        for model_name in E1_E2_MODELS:
            print_flush(f"Training {experiment} {model_name}, seed={seed}")
            estimator = construct_released_estimator(
                experiment,
                model_name,
                seed,
                getattr(args, "model_parameters", None),
            )
            pipeline, train_time, raw_feature_cols = fit_source_pipeline(estimator, data)

            artifact_out = (
                Path(args.output_dir)
                / "artifacts"
                / experiment.lower()
                / f"seed{seed}"
                / f"{model_name}.joblib"
            )
            artifact_out.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, artifact_out)

            rows.append(
                evaluate_unsw_binary(
                    pipeline,
                    data,
                    seed,
                    experiment,
                    model_name,
                    "none",
                    task,
                    raw_feature_cols,
                )
            )
            rows[-1].update({
                "run_status": "new_run_same_protocol_as_seed42",
                "training_protocol": training_protocol,
                "target_label_usage": "final_metric_computation_only",
                "hyperparameter_tuning": "none_constructed_from_released_parameters",
                "train_time_s": train_time,
                "artifact": str(artifact_out),
            })

            for target_name, target_path in TARGET_DATASETS.items():
                row = evaluate_target_binary(
                    pipeline,
                    target_name,
                    target_path,
                    seed,
                    experiment,
                    model_name,
                    "none",
                    task,
                    raw_feature_cols,
                    args.chunk_size,
                )
                row.update({
                    "run_status": "new_run_same_protocol_as_seed42",
                    "training_protocol": training_protocol,
                    "target_label_usage": "final_metric_computation_only",
                    "hyperparameter_tuning": "none_constructed_from_released_parameters",
                    "train_time_s": train_time,
                    "artifact": str(artifact_out),
                })
                rows.append(row)

            del pipeline
            gc.collect()

    return pd.DataFrame(rows)


def sample_source_features(X_train, feature_cols, max_rows, seed):
    X = X_train[feature_cols]
    if len(X) <= max_rows:
        return X.copy()
    return X.sample(n=max_rows, random_state=seed).copy()


def sample_target_features(dataset_path, feature_cols, max_rows, chunk_size, seed):
    pieces = []
    remaining = max_rows
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size, low_memory=False):
        chunk = prepare_cicids_chunk(chunk)
        usable = align_target_features(chunk, feature_cols)
        if len(usable) > remaining:
            usable = usable.sample(n=remaining, random_state=seed)
        pieces.append(usable)
        remaining -= len(usable)
        del chunk
        gc.collect()
        if remaining <= 0:
            break
    if not pieces:
        return pd.DataFrame(columns=feature_cols)
    return pd.concat(pieces, ignore_index=True)


def numeric_view(df, feature_cols):
    view = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return view.replace([np.inf, -np.inf], np.nan)


def compute_shift_scores(source_sample, target_sample, feature_cols):
    source = numeric_view(source_sample, feature_cols)
    target = numeric_view(target_sample, feature_cols)
    rows = []
    for feature in feature_cols:
        s = source[feature]
        t = target[feature]
        s_median = float(s.median())
        t_median = float(t.median())
        s_iqr = float(s.quantile(0.75) - s.quantile(0.25))
        t_iqr = float(t.quantile(0.75) - t.quantile(0.25))
        denom = s_iqr + t_iqr
        if not np.isfinite(denom) or denom == 0:
            denom = float(s.std(skipna=True) + t.std(skipna=True))
        if not np.isfinite(denom) or denom == 0:
            denom = 1.0
        rows.append({
            "feature": feature,
            "source_median": s_median,
            "target_median": t_median,
            "source_iqr": s_iqr,
            "target_iqr": t_iqr,
            SAMPLED_ROBUST_SHIFT_SCORE_COLUMN: abs(s_median - t_median) / denom,
        })
    return pd.DataFrame(rows).sort_values(SAMPLED_ROBUST_SHIFT_SCORE_COLUMN, ascending=False)


def compute_target_specific_shift_scores(data, seed, args):
    feature_cols = data["feature_cols"]
    source_sample = sample_source_features(
        data["X_train"],
        feature_cols,
        args.source_sample_rows,
        seed,
    )
    scores_by_target = {}
    score_dir = Path(args.output_dir) / "feature_filtering" / f"seed{seed}"
    score_dir.mkdir(parents=True, exist_ok=True)

    for target_name, target_path in TARGET_DATASETS.items():
        print_flush(f"Computing target-specific shift scores: seed={seed}, target={target_name}")
        target_sample = sample_target_features(
            target_path,
            feature_cols,
            args.target_sample_rows,
            args.chunk_size,
            seed,
        )
        scores = compute_shift_scores(source_sample, target_sample, feature_cols)
        scores["seed"] = seed
        scores["target_dataset"] = target_name
        scores["source_dataset"] = "NF-UNSW-NB15-v3"
        scores["source_split"] = "train_only"
        scores["source_distribution_mode"] = "fixed_source_train_sample"
        scores["target_distribution_mode"] = "fixed_unlabeled_target_sample"
        scores["source_sample_size"] = args.source_sample_rows
        scores["target_sample_size"] = args.target_sample_rows
        scores["target_label_usage"] = "not_used"
        scores["metric_name"] = SAMPLED_ROBUST_SHIFT_SCORE_COLUMN
        scores["metric_description"] = "robust median-IQR based feature shift score"
        scores["formal_feature_shift_metric_columns_reserved"] = ",".join(FORMAL_FEATURE_SHIFT_COLUMNS)
        scores.to_csv(score_dir / f"feature_sampled_robust_shift_scores_{target_name}.csv", index=False)
        scores_by_target[target_name] = scores
    return scores_by_target


def compute_redundant_features(data, seed, args):
    feature_cols = data["feature_cols"]
    X = data["X_train"][feature_cols]

    duplicate_removed = []
    seen = {}
    for col in feature_cols:
        signature = pd.util.hash_pandas_object(X[col], index=False).sum()
        if signature in seen and X[col].equals(X[seen[signature]]):
            duplicate_removed.append(col)
        else:
            seen[signature] = col

    corr_sample = sample_source_features(X, feature_cols, args.corr_sample_rows, seed)
    corr = numeric_view(corr_sample, feature_cols).corr().abs()
    corr_removed = []
    corr_pairs = []
    for i, left in enumerate(feature_cols):
        if left in corr_removed:
            continue
        for right in feature_cols[i + 1:]:
            if right in corr_removed:
                continue
            value = corr.loc[left, right]
            if pd.notna(value) and value > args.corr_threshold:
                corr_removed.append(right)
                corr_pairs.append({
                    "kept_feature": left,
                    "removed_feature": right,
                    "abs_corr": float(value),
                })

    redundant_dir = Path(args.output_dir) / "feature_filtering" / f"seed{seed}"
    redundant_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"removed_feature": duplicate_removed}).to_csv(
        redundant_dir / "exact_duplicate_features.csv",
        index=False,
    )
    pd.DataFrame(corr_pairs).to_csv(
        redundant_dir / f"high_corr_pairs_gt_{args.corr_threshold}.csv",
        index=False,
    )
    return duplicate_removed, corr_removed


def removed_feature_row(seed, target_dataset, strategy, removed_features, selection_scope):
    return {
        "seed": int(seed),
        "target_dataset": target_dataset,
        "filtering_strategy": strategy,
        "removed_features": json.dumps(list(removed_features)),
        "number_of_removed_features": len(removed_features),
        "selection_scope": selection_scope,
        "feature_filter_protocol": "target_specific_scheme_a",
        "target_label_usage": "not_used_for_feature_selection",
    }


def train_and_eval_filtered_model(
    data,
    seed,
    model_name,
    experiment,
    filtering_strategy,
    removed_features,
    eval_targets,
    args,
):
    base_features = data["feature_cols"]
    removed = [feature for feature in removed_features if feature in base_features]
    kept = [feature for feature in base_features if feature not in set(removed)]
    estimator = construct_released_estimator(
        "E2",
        model_name,
        seed,
        getattr(args, "model_parameters", None),
    )
    pipeline, train_time = fit_pipeline(estimator, data, kept)
    artifact_scope = (
        eval_targets[0]
        if len(eval_targets) == 1 and eval_targets[0] != "UNSW"
        else "universal"
    )

    artifact_out = (
        Path(args.output_dir)
        / "artifacts"
        / experiment.lower()
        / f"seed{seed}"
        / artifact_scope
        / filtering_strategy
        / f"{model_name}.joblib"
    )
    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_out)

    rows = []
    for target_name in eval_targets:
        if target_name == "UNSW":
            row = evaluate_unsw_binary(
                pipeline,
                data,
                seed,
                experiment,
                model_name,
                filtering_strategy,
                "binary",
                kept,
            )
        else:
            row = evaluate_target_binary(
                pipeline,
                target_name,
                TARGET_DATASETS[target_name],
                seed,
                experiment,
                model_name,
                filtering_strategy,
                "binary",
                kept,
                args.chunk_size,
            )
        row.update({
            "run_status": "new_run_target_specific_scheme_a",
            "feature_filter_protocol": "target_specific_scheme_a",
            "filtering_target_dataset": artifact_scope,
            "target_label_usage": "final_metric_computation_only",
            "hyperparameter_tuning": "none_constructed_from_released_parameters",
            "n_removed_features": len(removed),
            "removed_features": json.dumps(removed),
            "train_time_s": train_time,
            "artifact": str(artifact_out),
        })
        rows.append(row)

    del pipeline
    gc.collect()
    return rows


def run_e3_e4_target_specific_seed(seed, args):
    """Run E3/E4 for one seed under target-specific Scheme A."""
    set_all_seeds(seed)
    data = load_unsw_splits(seed, "binary")
    scores_by_target = compute_target_specific_shift_scores(data, seed, args)
    duplicate_removed, corr_removed = compute_redundant_features(data, seed, args)

    result_rows = []
    removed_rows = []

    for target_name, scores in scores_by_target.items():
        for k in SHIFT_TOP_K:
            strategy = f"remove_shift_top{k}"
            removed = scores.head(k)["feature"].tolist()
            removed_rows.append(
                removed_feature_row(
                    seed,
                    target_name,
                    strategy,
                    removed,
                    "UNSW train features + unlabeled target features",
                )
            )
            for model_name in E3_E4_MODELS:
                result_rows.extend(
                    train_and_eval_filtered_model(
                        data,
                        seed,
                        model_name,
                        "E3",
                        strategy,
                        removed,
                        [target_name],
                        args,
                    )
                )

        combined_strategy = f"remove_shift_top5_plus_corr_gt_{args.corr_threshold}"
        combined_removed = sorted(set(scores.head(5)["feature"].tolist() + corr_removed))
        removed_rows.append(
            removed_feature_row(
                seed,
                target_name,
                combined_strategy,
                combined_removed,
                "Target-specific shift top 5 + UNSW-train high-correlation redundancy",
            )
        )
        for model_name in E3_E4_MODELS:
            result_rows.extend(
                train_and_eval_filtered_model(
                    data,
                    seed,
                    model_name,
                    "E4",
                    combined_strategy,
                    combined_removed,
                    [target_name],
                    args,
                )
            )

    universal_e4_sets = [
        (
            "remove_exact_duplicates",
            duplicate_removed,
            "UNSW training split only",
        ),
        (
            f"remove_corr_gt_{args.corr_threshold}",
            corr_removed,
            "UNSW training split only",
        ),
    ]
    universal_eval_targets = ["UNSW", *TARGET_DATASETS.keys()]
    for strategy, removed, scope in universal_e4_sets:
        for target_name in TARGET_DATASETS:
            removed_rows.append(removed_feature_row(seed, target_name, strategy, removed, scope))
        for model_name in E3_E4_MODELS:
            result_rows.extend(
                train_and_eval_filtered_model(
                    data,
                    seed,
                    model_name,
                    "E4",
                    strategy,
                    removed,
                    universal_eval_targets,
                    args,
                )
            )

    return pd.DataFrame(result_rows), pd.DataFrame(removed_rows)


def is_present(value):
    return not pd.isna(value) and value != ""


def legacy_metric_value(row, *names, default=np.nan):
    for name in names:
        value = row.get(name, np.nan)
        if is_present(value):
            return value
    return default


def legacy_metrics_from_row(row, prefix=""):
    tp_name = f"{prefix}tp" if prefix else "tp"
    tn_name = f"{prefix}tn" if prefix else "tn"
    fp_name = f"{prefix}fp" if prefix else "fp"
    fn_name = f"{prefix}fn" if prefix else "fn"
    tp = legacy_metric_value(row, tp_name)
    tn = legacy_metric_value(row, tn_name)
    fp = legacy_metric_value(row, fp_name)
    fn = legacy_metric_value(row, fn_name)
    if all(is_present(value) for value in [tp, tn, fp, fn]):
        metrics = binary_metrics_from_counts({
            "tp": int(float(tp)),
            "tn": int(float(tn)),
            "fp": int(float(fp)),
            "fn": int(float(fn)),
        })
    else:
        metrics = {
            "binary_f1": np.nan,
            "mcc": np.nan,
            "fpr": np.nan,
            "fnr": np.nan,
            "weighted_f1": np.nan,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n_samples": legacy_metric_value(row, "n_samples"),
        }

    metrics["binary_f1"] = legacy_metric_value(
        row,
        f"{prefix}f1",
        "binary_f1",
        default=metrics["binary_f1"],
    )
    metrics["mcc"] = legacy_metric_value(row, "mcc", default=metrics["mcc"])
    metrics["fpr"] = legacy_metric_value(row, f"{prefix}fpr", "fpr", default=metrics["fpr"])
    metrics["fnr"] = legacy_metric_value(row, f"{prefix}fnr", "fnr", default=metrics["fnr"])
    metrics["weighted_f1"] = legacy_metric_value(
        row,
        f"{prefix}weighted_f1",
        "weighted_f1",
        default=metrics["weighted_f1"],
    )
    metrics["tp"] = tp
    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["n_samples"] = legacy_metric_value(row, "n_samples", default=metrics["n_samples"])
    return metrics


def normalize_existing_seed42_e1(path):
    df = pd.read_csv(path)
    rows = []
    model_filter = df["model"].isin(E1_E2_MODELS)
    for _, row in df[model_filter].iterrows():
        target_dataset = canonical_dataset_name(row.get("dataset", row.get("target_dataset")))
        if pd.isna(target_dataset):
            continue
        metrics = legacy_metrics_from_row(row, prefix="binary_")
        rows.append(make_result_row(
            "E1",
            42,
            row["model"],
            target_dataset,
            "none",
            metrics,
            evaluation_scope=row.get("evaluation_scope", ""),
            run_status="loaded_verified_result",
            metric_protocol="binary_detection_metrics",
            prediction_mapping="Benign->0; all_non_Benign_attack_classes->1",
            target_label_usage="final_metric_computation_only",
            hyperparameter_tuning="none_loaded_verified_seed42_result",
        ))
    return pd.DataFrame(rows)


def normalize_existing_seed42_e2(path):
    df = pd.read_csv(path)
    rows = []
    model_filter = df["model"].isin(E1_E2_MODELS)
    for _, row in df[model_filter].iterrows():
        target_dataset = canonical_dataset_name(row.get("dataset", row.get("target_dataset")))
        if pd.isna(target_dataset):
            continue
        metrics = legacy_metrics_from_row(row)
        rows.append(make_result_row(
            "E2",
            42,
            row["model"],
            target_dataset,
            "none",
            metrics,
            evaluation_scope=row.get("evaluation_scope", ""),
            run_status="loaded_verified_result",
            metric_protocol="binary_detection_metrics",
            prediction_mapping="binary_label_0_Benign_1_Attack",
            target_label_usage="final_metric_computation_only",
            hyperparameter_tuning="none_loaded_verified_seed42_result",
        ))
    return pd.DataFrame(rows)


def load_existing_seed42_e1_e2(args):
    if args.existing_seed42_e1_e2_long:
        df = pd.read_csv(args.existing_seed42_e1_e2_long)
        if "target_dataset" not in df.columns and "dataset" in df.columns:
            df["target_dataset"] = df["dataset"]
        if "filtering_strategy" not in df.columns and "feature_set" in df.columns:
            df["filtering_strategy"] = df["feature_set"]
        df["experiment"] = df["experiment"].map(canonical_experiment_name)
        df["target_dataset"] = df["target_dataset"].map(canonical_dataset_name)
        df = df[df["seed"].astype(int).eq(42) & df["experiment"].isin(["E1", "E2"])].copy()
        df["filtering_strategy"] = df["filtering_strategy"].replace("", np.nan).fillna("none")
        return df

    e1 = normalize_existing_seed42_e1(args.existing_seed42_e1)
    e2 = normalize_existing_seed42_e2(args.existing_seed42_e2)
    return pd.concat([e1, e2], ignore_index=True)


def seed42_repair_mask(df):
    return (
        df["experiment"].eq("E1")
        & df["seed"].astype(int).eq(42)
        & df["target_dataset"].eq("UNSW")
        & df["model"].isin(SEED42_E1_UNSW_REPAIR_MODELS)
    )


def repair_seed42_e1_unsw_metrics(args):
    """Repair missing E1 seed-42 UNSW binary metrics by inference only.

    This function deliberately does not call fit(), GridSearchCV, or any model
    construction path. It loads the verified seed-42 pipelines and calls
    predict() on the recreated original UNSW held-out test split.
    """
    original = standardize_long_columns(load_existing_seed42_e1_e2(args))
    repaired = original.copy(deep=True)

    repair_mask = seed42_repair_mask(repaired)
    expected_count = len(SEED42_E1_UNSW_REPAIR_MODELS)
    if int(repair_mask.sum()) != expected_count:
        raise ValueError(
            f"Expected {expected_count} E1 seed42 UNSW rows to repair, "
            f"found {int(repair_mask.sum())}."
        )

    original_columns = list(original.columns)
    untouched_before = original.loc[~repair_mask, original_columns].reset_index(drop=True)
    print_flush("Recreating original E1 seed42 UNSW held-out split for inference-only repair.")
    data = load_unsw_splits(42, "multiclass")
    X_test = data["X_test"]
    y_true_bin = (data["y_test"].astype(str) != "Benign").astype(int)

    for model_name in SEED42_E1_UNSW_REPAIR_MODELS:
        pipeline_path = Path(args.e1_artifact_dir) / f"{model_name}_pipeline.joblib"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Missing verified E1 pipeline: {pipeline_path}")

        print_flush(f"Loading verified E1 seed42 pipeline for inference only: {pipeline_path}")
        pipeline = joblib.load(pipeline_path)
        y_pred = pipeline.predict(X_test)
        y_pred_bin = (pd.Series(y_pred).astype(str) != "Benign").astype(int)
        metrics = binary_metrics(y_true_bin, y_pred_bin)

        row_mask = repair_mask & repaired["model"].eq(model_name)
        for col in REPAIR_METRIC_COLUMNS:
            repaired.loc[row_mask, col] = metrics[col]
        repaired.loc[row_mask, "run_status"] = "loaded_verified_result"
        repaired.loc[row_mask, "metric_repair"] = "seed42_e1_unsw_inference_only"
        repaired.loc[row_mask, "training_performed"] = "false"
        repaired.loc[row_mask, "metric_protocol"] = "binary_detection_metrics"
        repaired.loc[row_mask, "prediction_mapping"] = "Benign->0; all_non_Benign_attack_classes->1"
        repaired.loc[row_mask, "target_label_usage"] = "final_metric_computation_only"
        repaired.loc[row_mask, "hyperparameter_tuning"] = "none_loaded_verified_seed42_result"
        repaired.loc[row_mask, "artifact"] = str(pipeline_path)

        del pipeline, y_pred, y_pred_bin
        gc.collect()

    untouched_after = repaired.loc[~repair_mask, original_columns].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(untouched_before, untouched_after, check_dtype=False)
        unchanged_non_repaired = True
    except AssertionError as exc:
        unchanged_non_repaired = False
        raise AssertionError(
            "Rows outside E1 seed42 UNSW changed during repair; aborting save."
        ) from exc

    output_path = Path(args.seed42_repaired_output)
    if not output_path.is_absolute():
        output_path = Path(args.output_dir) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired = standardize_long_columns(repaired)
    repaired.to_csv(output_path, index=False)

    sanity = sanity_check_seed42_e1_e2(repaired)
    sanity["unchanged_non_repaired_rows"] = unchanged_non_repaired
    sanity["output_path"] = str(output_path)
    print_flush(json.dumps(sanity, indent=2, sort_keys=True))
    return output_path, repaired, sanity


def sanity_check_seed42_e1_e2(df):
    df = standardize_long_columns(df.copy())
    required_metrics = ["binary_f1", "mcc", "fpr", "fnr", "weighted_f1"]
    required_targets = {"UNSW", *TARGET_DATASETS.keys()}
    required_models = set(E1_E2_MODELS)

    report = {
        "experiments": sorted(df["experiment"].dropna().unique().tolist()),
        "seeds": sorted(df["seed"].dropna().astype(int).unique().tolist()),
        "filtering_strategy": sorted(df["filtering_strategy"].dropna().unique().tolist()),
        "checks": {},
    }
    for experiment in ["E1", "E2"]:
        exp_df = df[df["experiment"].eq(experiment)].copy()
        metric_missing = {
            metric: int(pd.to_numeric(exp_df[metric], errors="coerce").isna().sum())
            for metric in required_metrics
        }
        combos = exp_df.groupby(["model", "target_dataset"]).size().reset_index(name="n")
        bad_combos = combos[combos["n"].ne(1)]
        report["checks"][experiment] = {
            "rows": int(len(exp_df)),
            "models": sorted(exp_df["model"].dropna().unique().tolist()),
            "targets": sorted(exp_df["target_dataset"].dropna().unique().tolist()),
            "missing_models": sorted(required_models - set(exp_df["model"].dropna().unique())),
            "missing_targets": sorted(required_targets - set(exp_df["target_dataset"].dropna().unique())),
            "duplicate_or_missing_combo_count": int(len(bad_combos)),
            "metric_missing": metric_missing,
        }

    repair_rows = df[seed42_repair_mask(df)].copy()
    report["checks"]["E1_seed42_UNSW_repair_rows"] = {
        "rows": int(len(repair_rows)),
        "metric_missing": {
            metric: int(pd.to_numeric(repair_rows[metric], errors="coerce").isna().sum())
            for metric in required_metrics
        },
        "metric_repair": sorted(repair_rows.get("metric_repair", pd.Series(dtype=object)).dropna().unique().tolist()),
        "training_performed": sorted(repair_rows.get("training_performed", pd.Series(dtype=object)).dropna().unique().tolist()),
    }
    return report


def standardize_long_columns(df):
    if "target_dataset" not in df.columns and "dataset" in df.columns:
        df["target_dataset"] = df["dataset"]
    if "filtering_strategy" not in df.columns and "feature_set" in df.columns:
        df["filtering_strategy"] = df["feature_set"]
    for col in REQUIRED_LONG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["experiment"] = df["experiment"].map(canonical_experiment_name)
    df["target_dataset"] = df["target_dataset"].map(canonical_dataset_name)
    df["filtering_strategy"] = df["filtering_strategy"].replace("", np.nan).fillna("none")
    e12_mask = df["experiment"].isin(["E1", "E2"])
    df.loc[e12_mask, "filtering_strategy"] = "none"
    leading = REQUIRED_LONG_COLUMNS
    trailing = [col for col in df.columns if col not in leading]
    return df[leading + trailing]


def write_mean_std_summary(all_long, output_dir):
    metrics = ["binary_f1", "mcc", "fpr", "fnr", "weighted_f1"]
    group_cols = ["experiment", "model", "target_dataset", "filtering_strategy"]
    summary = (
        all_long
        .groupby(group_cols, dropna=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    path = Path(output_dir) / "e1_e4_mean_std_summary.csv"
    summary.to_csv(path, index=False)
    return path


def e3_filtering_strategies():
    return [f"remove_shift_top{k}" for k in SHIFT_TOP_K]


def e4_filtering_strategies(corr_threshold):
    return [
        "remove_exact_duplicates",
        f"remove_corr_gt_{corr_threshold}",
        f"remove_shift_top5_plus_corr_gt_{corr_threshold}",
    ]


def validate_e34_internal_protocol(all_long, removed_features, errors, corr_threshold):
    """Validate the new E3/E4 protocol internally across seeds 42/123/2024."""
    e34_rows = all_long[all_long["experiment"].isin(["E3", "E4"])].copy()
    if e34_rows.empty:
        errors.append("No E3/E4 rows were produced.")
        return

    e34_protocol_values = set(e34_rows["feature_filter_protocol"].dropna().unique())
    if e34_protocol_values != {"target_specific_scheme_a"}:
        errors.append(
            "E3/E4 result rows must all use target_specific_scheme_a; "
            f"found {sorted(e34_protocol_values)}."
        )
    e34_run_status_values = set(e34_rows["run_status"].dropna().unique())
    if e34_run_status_values != {"new_run_target_specific_scheme_a"}:
        errors.append(
            "E3/E4 rows must all be new_run_target_specific_scheme_a; "
            f"found {sorted(e34_run_status_values)}."
        )

    actual_seeds = sorted(e34_rows["seed"].dropna().astype(int).unique().tolist())
    if actual_seeds != E34_SEEDS_TO_RUN:
        errors.append(f"E3/E4 result seeds are {actual_seeds}, expected {E34_SEEDS_TO_RUN}.")

    identity_cols = [
        "experiment",
        "model",
        "target_dataset",
        "filtering_strategy",
        "evaluation_scope",
        "filtering_target_dataset",
    ]
    for col in identity_cols:
        if col not in e34_rows.columns:
            errors.append(f"E3/E4 result rows are missing protocol column: {col}.")
            return

    per_seed_keys = {
        seed: set(map(tuple, group[identity_cols].drop_duplicates().to_numpy()))
        for seed, group in e34_rows.groupby(e34_rows["seed"].astype(int))
    }
    reference_seed = E34_SEEDS_TO_RUN[0]
    reference_keys = per_seed_keys.get(reference_seed, set())
    for seed in E34_SEEDS_TO_RUN:
        keys = per_seed_keys.get(seed, set())
        if keys != reference_keys:
            errors.append(
                f"E3/E4 protocol keys for seed {seed} differ from seed {reference_seed}: "
                f"missing={sorted(reference_keys - keys)}, extra={sorted(keys - reference_keys)}."
            )

    target_names = set(TARGET_DATASETS)
    for seed, group in e34_rows.groupby(e34_rows["seed"].astype(int)):
        e3 = group[group["experiment"].eq("E3")]
        for strategy in e3_filtering_strategies():
            strategy_rows = e3[e3["filtering_strategy"].eq(strategy)]
            targets = set(strategy_rows["target_dataset"].dropna().unique())
            filtering_targets = set(strategy_rows["filtering_target_dataset"].dropna().unique())
            models = set(strategy_rows["model"].dropna().unique())
            if targets != target_names or filtering_targets != target_names:
                errors.append(
                    f"E3 seed {seed} strategy {strategy} is not target-specific: "
                    f"target_dataset={sorted(targets)}, "
                    f"filtering_target_dataset={sorted(filtering_targets)}."
                )
            if models != set(E3_E4_MODELS):
                errors.append(
                    f"E3 seed {seed} strategy {strategy} models are {sorted(models)}, "
                    f"expected {E3_E4_MODELS}."
                )

        e4 = group[group["experiment"].eq("E4")]
        combined = e4[e4["filtering_strategy"].str.startswith("remove_shift_top5_plus_corr_gt_", na=False)]
        for target_name in TARGET_DATASETS:
            rows = combined[
                combined["target_dataset"].eq(target_name)
                & combined["filtering_target_dataset"].eq(target_name)
            ]
            models = set(rows["model"].dropna().unique())
            if models != set(E3_E4_MODELS):
                errors.append(
                    f"E4 combined strategy seed {seed} target {target_name} models are "
                    f"{sorted(models)}, expected {E3_E4_MODELS}."
                )

        universal = e4[e4["filtering_strategy"].isin([
            "remove_exact_duplicates",
            f"remove_corr_gt_{corr_threshold}",
        ])]
        if not universal.empty:
            scopes = set(universal["filtering_target_dataset"].dropna().unique())
            if scopes != {"universal"}:
                errors.append(
                    f"E4 redundancy-only seed {seed} should use universal scope; "
                    f"found {sorted(scopes)}."
                )

    if removed_features.empty:
        errors.append("E3/E4 removed-feature tracking is empty.")
        return

    required_removed_cols = [
        "seed",
        "target_dataset",
        "filtering_strategy",
        "removed_features",
        "number_of_removed_features",
        "selection_scope",
        "feature_filter_protocol",
        "target_label_usage",
    ]
    missing_removed_cols = [col for col in required_removed_cols if col not in removed_features.columns]
    if missing_removed_cols:
        errors.append(f"Removed-feature tracking is missing columns: {missing_removed_cols}.")
        return

    e34_removed_seeds = sorted(removed_features["seed"].dropna().astype(int).unique().tolist())
    if e34_removed_seeds != E34_SEEDS_TO_RUN:
        errors.append(
            f"E3/E4 removed-feature seeds are {e34_removed_seeds}, expected {E34_SEEDS_TO_RUN}."
        )
    protocol_values = set(removed_features["feature_filter_protocol"].dropna().unique())
    if protocol_values != {"target_specific_scheme_a"}:
        errors.append(f"Unexpected E3/E4 feature-filter protocols: {sorted(protocol_values)}.")
    label_usage_values = set(removed_features["target_label_usage"].dropna().unique())
    if label_usage_values != {"not_used_for_feature_selection"}:
        errors.append(f"Unexpected target label usage in removed-feature tracking: {sorted(label_usage_values)}.")

    expected_removed_strategies = set(e3_filtering_strategies() + e4_filtering_strategies(corr_threshold))
    expected_removed_keys = {
        (target_name, strategy)
        for target_name in TARGET_DATASETS
        for strategy in expected_removed_strategies
    }
    removed_identity_cols = ["target_dataset", "filtering_strategy"]
    per_seed_removed_keys = {
        seed: set(map(tuple, group[removed_identity_cols].drop_duplicates().to_numpy()))
        for seed, group in removed_features.groupby(removed_features["seed"].astype(int))
    }
    for seed in E34_SEEDS_TO_RUN:
        keys = per_seed_removed_keys.get(seed, set())
        if keys != expected_removed_keys:
            errors.append(
                f"Removed-feature tracking keys for seed {seed} are incomplete: "
                f"missing={sorted(expected_removed_keys - keys)}, extra={sorted(keys - expected_removed_keys)}."
            )

    duplicate_removed = removed_features.duplicated(
        subset=["seed", "target_dataset", "filtering_strategy"],
        keep=False,
    )
    if duplicate_removed.any():
        errors.append(
            "Removed-feature tracking has duplicate seed/target/strategy rows: "
            f"{int(duplicate_removed.sum())} duplicate rows."
        )


def validate_outputs(all_long, removed_features, corr_threshold=CORR_THRESHOLD):
    errors = []
    seeds = sorted(all_long["seed"].dropna().astype(int).unique().tolist())
    if seeds != FINAL_SEEDS:
        errors.append(f"Final seeds are {seeds}, expected {FINAL_SEEDS}.")

    required_metadata_cols = [
        "run_status",
        "metric_protocol",
        "prediction_mapping",
        "target_label_usage",
        "hyperparameter_tuning",
    ]
    missing_metadata_cols = [col for col in required_metadata_cols if col not in all_long.columns]
    if missing_metadata_cols:
        errors.append(f"Missing required protocol metadata columns: {missing_metadata_cols}.")
    else:
        e12 = all_long[all_long["experiment"].isin(["E1", "E2"])].copy()
        e12_seed42_status = set(e12[e12["seed"].astype(int).eq(42)]["run_status"].dropna().unique())
        if e12_seed42_status != {"loaded_verified_result"}:
            errors.append(f"E1/E2 seed 42 run_status is {sorted(e12_seed42_status)}, expected loaded_verified_result.")
        e12_new_status = set(e12[e12["seed"].astype(int).isin([123, 2024])]["run_status"].dropna().unique())
        if e12_new_status != {"new_run_same_protocol_as_seed42"}:
            errors.append(
                f"E1/E2 seed 123/2024 run_status is {sorted(e12_new_status)}, "
                "expected new_run_same_protocol_as_seed42."
            )
        metric_protocols = set(all_long["metric_protocol"].dropna().unique())
        if metric_protocols != {"binary_detection_metrics"}:
            errors.append(f"Unexpected metric protocols: {sorted(metric_protocols)}.")
        target_label_usage_values = set(all_long["target_label_usage"].dropna().unique())
        if target_label_usage_values != {"final_metric_computation_only"}:
            errors.append(f"Unexpected target label usage values: {sorted(target_label_usage_values)}.")
        tuning_values = set(all_long["hyperparameter_tuning"].dropna().unique())
        bad_tuning_values = [value for value in tuning_values if not str(value).startswith("none_")]
        if bad_tuning_values:
            errors.append(f"Unexpected hyperparameter tuning values: {sorted(bad_tuning_values)}.")
        e1_mapping = set(
            all_long.loc[all_long["experiment"].eq("E1"), "prediction_mapping"]
            .dropna()
            .unique()
        )
        if e1_mapping != {"Benign->0; all_non_Benign_attack_classes->1"}:
            errors.append(f"E1 prediction mapping is {sorted(e1_mapping)}, expected multiclass-to-binary mapping.")
        e2_e34_mapping = set(
            all_long.loc[all_long["experiment"].isin(["E2", "E3", "E4"]), "prediction_mapping"]
            .dropna()
            .unique()
        )
        if e2_e34_mapping != {"binary_label_0_Benign_1_Attack"}:
            errors.append(f"E2/E3/E4 prediction mapping is {sorted(e2_e34_mapping)}, expected binary label mapping.")

    coverage = {
        "E1": E1_E2_MODELS,
        "E2": E1_E2_MODELS,
        "E3": E3_E4_MODELS,
        "E4": E3_E4_MODELS,
    }
    for experiment, expected_models in coverage.items():
        actual = sorted(all_long.loc[all_long["experiment"].eq(experiment), "model"].dropna().unique())
        missing = sorted(set(expected_models) - set(actual))
        if missing:
            errors.append(f"{experiment} missing models: {missing}.")

    metric_ranges = {
        "binary_f1": (0, 1),
        "weighted_f1": (0, 1),
        "fpr": (0, 1),
        "fnr": (0, 1),
        "mcc": (-1, 1),
    }
    for metric, (low, high) in metric_ranges.items():
        values = pd.to_numeric(all_long[metric], errors="coerce").dropna()
        bad = values[(values < low) | (values > high)]
        if not bad.empty:
            errors.append(f"{metric} has {len(bad)} values outside [{low}, {high}].")

    group_cols = ["experiment", "model", "target_dataset", "filtering_strategy"]
    for keys, group in all_long.groupby(group_cols, dropna=False):
        group_label = dict(zip(group_cols, keys))
        for metric in metric_ranges:
            metric_values = pd.to_numeric(group[metric], errors="coerce")
            metric_seeds = sorted(group.loc[metric_values.notna(), "seed"].astype(int).unique().tolist())
            if metric_seeds != FINAL_SEEDS:
                errors.append(
                    f"{group_label} has {metric} for seeds {metric_seeds}, "
                    f"expected {FINAL_SEEDS}."
                )

    count_cols = ["tp", "tn", "fp", "fn"]
    count_df = all_long[count_cols].apply(pd.to_numeric, errors="coerce")
    negative_counts = (count_df < 0).any(axis=1)
    if negative_counts.any():
        errors.append(f"{int(negative_counts.sum())} rows have negative confusion counts.")
    if "n_samples" in all_long.columns:
        n_samples = pd.to_numeric(all_long["n_samples"], errors="coerce")
        count_sums = count_df.sum(axis=1)
        comparable = n_samples.notna() & count_df.notna().all(axis=1)
        mismatched = comparable & (count_sums.astype(float) != n_samples.astype(float))
        if mismatched.any():
            errors.append(f"{int(mismatched.sum())} rows have counts that do not sum to n_samples.")

    validate_e34_internal_protocol(all_long, removed_features, errors, corr_threshold)

    if errors:
        raise ValueError("Validation failed:\n- " + "\n- ".join(errors))


def run_all(args):
    from cross_domain_nids.release_paths import require_generated_output

    output_dir = require_generated_output(args.output_dir, "results")
    args.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    e12_new_frames = []
    for seed in E12_SEEDS_TO_RUN:
        seed_df = run_e1_e2_new_seed(seed, args)
        e12_path = output_dir / f"e1_e2_seed{seed}_results_long.csv"
        standardize_long_columns(seed_df).to_csv(e12_path, index=False)
        e12_new_frames.append(seed_df)
        print_flush(f"Saved E1/E2 seed {seed}: {e12_path}")

    e34_frames = []
    removed_frames = []
    for seed in E34_SEEDS_TO_RUN:
        result_df, removed_df = run_e3_e4_target_specific_seed(seed, args)
        result_path = output_dir / f"e3_e4_seed{seed}_target_specific_results_long.csv"
        standardize_long_columns(result_df).to_csv(result_path, index=False)
        e34_frames.append(result_df)
        removed_frames.append(removed_df)
        print_flush(f"Saved target-specific E3/E4 seed {seed}: {result_path}")

    removed_features = pd.concat(removed_frames, ignore_index=True)
    removed_path = output_dir / "e3_e4_removed_features_by_seed.csv"
    removed_features.to_csv(removed_path, index=False)

    seed42_e12 = load_existing_seed42_e1_e2(args)
    all_long = pd.concat([seed42_e12, *e12_new_frames, *e34_frames], ignore_index=True)
    all_long = standardize_long_columns(all_long)

    for seed in [123, 2024]:
        seed_path = output_dir / f"e1_e4_seed{seed}_results_long.csv"
        all_long[all_long["seed"].astype(int).eq(seed)].to_csv(seed_path, index=False)
        print_flush(f"Saved per-seed E1-E4 file: {seed_path}")

    all_path = output_dir / "e1_e4_all_seeds_long.csv"
    all_long.to_csv(all_path, index=False)
    summary_path = write_mean_std_summary(all_long, output_dir)
    validate_outputs(all_long, removed_features, args.corr_threshold)

    print_flush(f"Saved removed-feature tracking: {removed_path}")
    print_flush(f"Saved merged long-format results: {all_path}")
    print_flush(f"Saved mean/std summary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Gap 1 E1-E4 multi-seed experiments without mixing E3/E4 protocols."
    )
    parser.add_argument(
        "--output-dir",
        default=REPOSITORY_ROOT / "results" / "generated" / "h1" / "multiseed",
    )
    parser.add_argument("--chunk-size", type=int, default=STRESS_B_CHUNK_SIZE)
    parser.add_argument("--source-sample-rows", type=int, default=300_000)
    parser.add_argument("--target-sample-rows", type=int, default=200_000)
    parser.add_argument("--corr-sample-rows", type=int, default=300_000)
    parser.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD)
    parser.add_argument("--e1-artifact-dir", default="artifacts/pipelines")
    parser.add_argument("--e2-artifact-dir", default="artifacts/stress_b_v2/binary")
    parser.add_argument(
        "--model-parameters",
        type=Path,
        default=DEFAULT_PARAMETERS_PATH,
        help="Released JSON used to construct E1/E2 estimators without templates.",
    )
    parser.add_argument(
        "--repair-seed42-e1-unsw-only",
        action="store_true",
        help=(
            "Do not run experiments. Repair only missing E1 seed42 UNSW "
            "binary metrics by loading verified pipelines and running "
            "inference on the original held-out split."
        ),
    )
    parser.add_argument(
        "--seed42-repaired-output",
        default="seed42_e1_e2_long_repaired.csv",
        help=(
            "Output CSV for --repair-seed42-e1-unsw-only. Relative paths are "
            "written under --output-dir."
        ),
    )
    parser.add_argument(
        "--existing-seed42-e1",
        default="external/historical_h1/fair_pipeline_cross_dataset_results.csv",
        help="Verified existing E1 seed-42 cross-dataset result table.",
    )
    parser.add_argument(
        "--existing-seed42-e2",
        default="external/historical_h1/binary_training_results.csv",
        help="Verified existing E2 seed-42 result table.",
    )
    parser.add_argument(
        "--existing-seed42-e1-e2-long",
        default="",
        help=(
            "Optional verified long-format seed-42 E1/E2 table. If supplied, "
            "it takes precedence over the legacy E1/E2 tables."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.repair_seed42_e1_unsw_only:
        repair_seed42_e1_unsw_metrics(parsed_args)
    else:
        run_all(parsed_args)
