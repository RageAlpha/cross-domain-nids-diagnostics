"""Frozen-BN H3 DANN representation-adaptation experiments.

This script re-runs the H3 DANN variants with target-forward BatchNorm
running-stat updates disabled. The original H3 DANN results are retained for
reproducibility, but are considered confounded by target-induced BatchNorm
running-stat contamination.

It is intentionally target-label safe:
  - target labels are never used for adaptation sampling, training, early
    stopping, hyperparameter tuning, threshold selection, or model selection
  - target labels are read only for final full-target evaluation and optional
    post-hoc label-colored visualization
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from cross_domain_nids.config import (
    ATTACK_COL,
    BOTIOT_PATH,
    CATEGORICAL_FEATURES,
    CICIDS_PATH,
    LABEL_COL,
    NUMERIC_FEATURES,
    PROTOCOL_FEATURES,
    SEED,
    STRESS_B_CHUNK_SIZE,
    TONIOT_PATH,
    UNSW_PATH,
)
from cross_domain_nids.models.dann import BinaryMLP, DANN
from cross_domain_nids.preprocessing.netflow import (
    clean_l7_proto,
    load_and_clean_data,
    prepare_cicids_chunk,
)


def build_preprocessor_for_features(feature_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    numeric = [c for c in NUMERIC_FEATURES if c in feature_cols]
    protocol = [c for c in PROTOCOL_FEATURES if c in feature_cols]
    categorical = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

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


TARGET_DATASETS = {
    "CICIDS2018": CICIDS_PATH,
    "BoT-IoT": BOTIOT_PATH,
    "ToN-IoT": TONIOT_PATH,
}

SOURCE_SPLIT_SEED = SEED
SEEDS = [42, 123, 2024]
CORRECTED_EXPERIMENT_NAME = "h3_dann_corrected_frozen_bn"
CORRECTED_METADATA = {
    "bn_control_mode": "freeze_target_forward",
    "batchnorm_policy": "source_forward_updates_bn_target_forward_frozen",
    "source_forward_updates_bn_running_stats": "Yes",
    "target_forward_updates_bn_running_stats": "No",
    "corrected_from": "original_h3_dann",
    "reason_for_correction": "target-induced BatchNorm running-stat contamination",
    "original_results_status": "confounded_not_deleted",
}
DANN_VARIANTS = ["h3_dann", "h3_filtered_dann"]
VARIANTS = {
    "h3_mlp": {
        "model": "MLP",
        "method": "standard binary MLP",
        "feature_set": "all features",
        "use_dann": False,
        "use_filtered_features": False,
        "unlabeled_target_data_used": "No",
    },
    "h3_dann": {
        "model": "MLP-DANN",
        "method": "binary MLP-DANN",
        "feature_set": "all features",
        "use_dann": True,
        "use_filtered_features": False,
        "unlabeled_target_data_used": "Yes",
    },
    "h3_filtered_mlp": {
        "model": "MLP",
        "method": "filtered binary MLP",
        "feature_set": "filtered features",
        "use_dann": False,
        "use_filtered_features": True,
        "unlabeled_target_data_used": "No",
    },
    "h3_filtered_dann": {
        "model": "MLP-DANN",
        "method": "filtered binary MLP-DANN",
        "feature_set": "filtered features",
        "use_dann": True,
        "use_filtered_features": True,
        "unlabeled_target_data_used": "Yes",
    },
}


@dataclass
class H3Config:
    experiment_name: str
    bn_control_mode: str
    batchnorm_policy: str
    source_forward_updates_bn_running_stats: str
    target_forward_updates_bn_running_stats: str
    corrected_from: str
    reason_for_correction: str
    original_results_status: str
    source_dataset: str
    target_dataset: str
    variant: str
    experiment_mode: str
    seed: int
    source_split_seed: int
    run_seed: int
    feature_list: list[str]
    removed_feature_list: list[str]
    preprocessor_path: str
    target_total_samples: int
    target_adaptation_sample_ratio: float
    target_adaptation_sample_ratio_requested: float
    target_adaptation_sample_size: int
    target_adaptation_effective_ratio: float
    target_adaptation_sample_cap: int | None
    target_sampling_method: str
    target_sampling_method_code: str
    chunk_selection_rule: str
    target_labels_used_for_sampling: str
    lambda_schedule: str
    min_epochs_before_early_stop: int
    early_stopping_mode: str
    min_epochs: int
    patience: int
    final_epoch: int
    best_epoch: int
    final_lambda_value: float
    best_source_val_loss: float
    target_labels_used_for_model_selection: str
    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping_rule: str
    target_labels_used_in_training: str
    evaluation_protocol: str
    grl_alpha: float
    loss_weighting_style: str


def print_flush(message):
    print(message, flush=True)


@contextmanager
def freeze_batchnorm_running_stats(model):
    """Freeze BatchNorm running stats for a target-domain forward pass."""
    import torch.nn as nn

    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    old_modes = [m.training for m in bn_layers]
    try:
        for layer in bn_layers:
            layer.eval()
        yield
    finally:
        for layer, was_training in zip(bn_layers, old_modes):
            layer.train(was_training)


def parse_optional_sample_cap(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null", "0"}:
        return None
    cap = int(text)
    if cap < 0:
        raise argparse.ArgumentTypeError("sample cap must be non-negative, 0, or None")
    return cap


def set_all_seeds(seed):
    from cross_domain_nids.determinism import configure_determinism

    configure_determinism(seed)


def binary_metrics_from_counts(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / math.sqrt(denominator) if denominator else 0.0

    benign_precision = tn / (tn + fn) if (tn + fn) else 0.0
    benign_recall = tn / (tn + fp) if (tn + fp) else 0.0
    benign_f1 = (
        2 * benign_precision * benign_recall / (benign_precision + benign_recall)
        if (benign_precision + benign_recall)
        else 0.0
    )
    binary_weighted_f1 = ((tn + fp) * benign_f1 + (tp + fn) * f1) / n if n else 0.0

    return {
        "Attack-class Binary-F1": f1,
        "MCC": mcc,
        "FPR": fpr,
        "FNR": fnr,
        "Binary weighted-F1 secondary": binary_weighted_f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def prepare_source_splits(split_seed=SOURCE_SPLIT_SEED, feature_cols=None):
    """Fit source-only preprocessor and return transformed UNSW splits."""
    source = load_and_clean_data(UNSW_PATH)
    source = clean_l7_proto(source)
    y = source[LABEL_COL].astype(int)
    X = source.drop(columns=[LABEL_COL, ATTACK_COL])
    if feature_cols is not None:
        X = X[feature_cols]
    else:
        feature_cols = list(X.columns)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=split_seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1 / 0.8,
        stratify=y_train_val,
        random_state=split_seed,
    )

    preprocessor = build_preprocessor_for_features(feature_cols)
    preprocessor.fit(X_train)
    return {
        "feature_cols": feature_cols,
        "preprocessor": preprocessor,
        "X_train": preprocessor.transform(X_train).astype(np.float32),
        "X_val": preprocessor.transform(X_val).astype(np.float32),
        "X_test": preprocessor.transform(X_test).astype(np.float32),
        "y_train": y_train.to_numpy(dtype=np.int64),
        "y_val": y_val.to_numpy(dtype=np.int64),
        "y_test": y_test.to_numpy(dtype=np.int64),
    }


def load_removed_features(target_dataset, feature_set_name, model="MLP"):
    """Load H2 filtered features without using target labels.

    The feature list is read from the completed H2 feature-ablation output. It
    was generated by the H2 feature-shift/redundancy pipeline and is reused
    here without target-label access.
    """
    path = Path("results/stress_b_v2/feature_ablation/feature_ablation_results.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    match = df[
        df["dataset"].eq(target_dataset)
        & df["model"].eq(model)
        & df["feature_set"].eq(feature_set_name)
    ]
    if match.empty:
        match = df[df["feature_set"].eq(feature_set_name)]
    if match.empty:
        return []
    raw = match["removed_features"].iloc[0]
    try:
        return list(json.loads(raw))
    except (TypeError, json.JSONDecodeError):
        return []


def label_free_target_sample(
    dataset_name,
    dataset_path,
    feature_cols,
    preprocessor,
    seed,
    sample_ratio,
    max_samples,
    chunk_size,
    target_sampling_method="exact_random_5pct",
):
    """Randomly sample target rows using only row indices and seed."""
    rng = np.random.default_rng(seed)
    if target_sampling_method not in {"exact_random_5pct", "chunk_exact_5pct"}:
        raise ValueError(f"Unknown target sampling method: {target_sampling_method}")

    def transform_sample(sample):
        drop_cols = [c for c in [LABEL_COL, ATTACK_COL] if c in sample.columns]
        X = sample.drop(columns=drop_cols)
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_cols]
        return preprocessor.transform(X).astype(np.float32)

    total = 0
    chunk_sizes = []
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size, low_memory=False):
        chunk = prepare_cicids_chunk(chunk)
        chunk_sizes.append(len(chunk))
        total += len(chunk)
        del chunk

    requested_sample_size = int(total * sample_ratio)
    sample_size = requested_sample_size
    sampling_method = f"label-free exact-size random {sample_ratio:.0%}"
    chunk_selection_rule = ""
    if target_sampling_method == "chunk_exact_5pct":
        sampling_method = f"label-free chunk-based exact-size random {sample_ratio:.0%}"
        chunk_selection_rule = (
            "random chunk permutation; take full chunks until requested size; "
            "sample rows from final selected chunk if needed"
        )
    if max_samples is not None:
        sample_size = min(requested_sample_size, max_samples)
        sampling_method = f"{sampling_method} capped at {max_samples}"

    if target_sampling_method == "exact_random_5pct" and sample_size > 0:
        selected_indices = np.sort(rng.choice(total, size=sample_size, replace=False))
    else:
        selected_indices = np.empty((0,), dtype=np.int64)
    chunk_take_counts = {}
    if target_sampling_method == "chunk_exact_5pct" and sample_size > 0:
        running = 0
        for chunk_idx in rng.permutation(len(chunk_sizes)):
            remaining = sample_size - running
            take = min(remaining, chunk_sizes[chunk_idx])
            chunk_take_counts[int(chunk_idx)] = int(take)
            running += take
            if running >= sample_size:
                break

    parts = []
    selected = 0
    sample_cursor = 0
    row_offset = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(dataset_path, chunksize=chunk_size, low_memory=False)):
        chunk = prepare_cicids_chunk(chunk)
        chunk_start = row_offset
        chunk_end = row_offset + len(chunk)
        row_offset = chunk_end

        if target_sampling_method == "exact_random_5pct":
            if sample_cursor >= len(selected_indices):
                del chunk
                continue
            next_cursor = np.searchsorted(selected_indices, chunk_end, side="left")
            chunk_indices = selected_indices[sample_cursor:next_cursor]
            sample_cursor = next_cursor
            if len(chunk_indices) == 0:
                del chunk
                continue
            local_indices = chunk_indices - chunk_start
            sample = chunk.iloc[local_indices].copy()
        else:
            if chunk_idx not in chunk_take_counts or selected >= sample_size:
                del chunk
                continue
            take = chunk_take_counts[chunk_idx]
            if take == len(chunk):
                sample = chunk.copy()
            else:
                local_indices = np.sort(rng.choice(len(chunk), size=take, replace=False))
                sample = chunk.iloc[local_indices].copy()

        Xt = transform_sample(sample)
        parts.append(Xt)
        selected += len(Xt)
        del chunk, sample
        gc.collect()
    if parts:
        X_target = np.vstack(parts)
    else:
        X_target = np.empty((0, len(feature_cols)), dtype=np.float32)
    effective_ratio = (len(X_target) / total) if total else 0.0
    metadata = {
        "target_dataset": dataset_name,
        "target_total_samples": total,
        "target_adaptation_sample_ratio": sample_ratio,
        "target_adaptation_sample_ratio_requested": sample_ratio,
        "target_adaptation_sample_size": int(len(X_target)),
        "target_adaptation_effective_ratio": effective_ratio,
        "target_adaptation_sample_cap": max_samples,
        "sampling_method": sampling_method,
        "target_sampling_method_code": target_sampling_method,
        "chunk_selection_rule": chunk_selection_rule,
        "sampling_seed": seed,
        "target_labels_used_for_sampling": "No",
    }
    return X_target, metadata


def lambda_value_for_epoch(schedule, epoch, epochs):
    if schedule == "constant_0":
        return 0.0
    if schedule == "linear_0_0.1":
        return 0.1 * epoch / max(epochs - 1, 1)
    if schedule in {"linear_0_1", "linear_0_1_min_epochs"}:
        return epoch / max(epochs - 1, 1)
    if schedule == "constant_1":
        return 1.0
    raise ValueError(f"Unknown lambda schedule: {schedule}")


def can_apply_early_stop(epoch, args):
    return (epoch + 1) >= args.min_epochs_before_early_stop


def binary_rates_from_metrics(metrics):
    tp = int(metrics["TP"])
    tn = int(metrics["TN"])
    fp = int(metrics["FP"])
    fn = int(metrics["FN"])
    n = tp + tn + fp + fn
    return {
        "true_attack_rate": ((tp + fn) / n) if n else 0.0,
        "pred_attack_rate": ((tp + fp) / n) if n else 0.0,
    }


def is_inside_or_equal(path, base):
    path = Path(path).resolve()
    base = Path(base).resolve()
    return path == base or base in path.parents


def guard_output_dir(output_dir):
    out_dir = Path(output_dir).resolve()
    allowed_root = REPOSITORY_ROOT / "results" / "generated" / "h3"
    if out_dir != allowed_root and allowed_root not in out_dir.parents:
        raise ValueError(f"H3 output must remain under {allowed_root}; got {out_dir}")


def make_loader(torch, X, y=None, batch_size=4096, seed=SEED, shuffle=True):
    tensor_x = torch.as_tensor(X, dtype=torch.float32)
    if y is None:
        dataset = torch.utils.data.TensorDataset(tensor_x)
    else:
        tensor_y = torch.as_tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


def source_validation_metrics(torch, model, X_val, y_val, device):
    model.eval()
    logits_all = []
    with torch.no_grad():
        for start in range(0, len(X_val), 8192):
            xb = torch.as_tensor(X_val[start:start + 8192], dtype=torch.float32, device=device)
            if hasattr(model, "forward_label"):
                logits = model.forward_label(xb)
            else:
                logits = model(xb)
            logits_all.append(logits.detach().cpu().numpy())
    logits = np.vstack(logits_all)
    pred = logits.argmax(axis=1)
    tp = int(((y_val == 1) & (pred == 1)).sum())
    tn = int(((y_val == 0) & (pred == 0)).sum())
    fp = int(((y_val == 0) & (pred == 1)).sum())
    fn = int(((y_val == 1) & (pred == 0)).sum())
    return binary_metrics_from_counts(tp, tn, fp, fn)


def train_source_mlp(data, input_dim, args, seed, variant, target_dataset, log_rows):
    import torch
    from torch import nn

    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = BinaryMLP(input_dim=input_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(torch, data["X_train"], data["y_train"], args.batch_size, seed)

    best_state = None
    best_val_loss = float("inf")
    best_val_metrics = None
    best_epoch = -1
    final_epoch = -1
    final_lambda_value = 0.0
    stale = 0
    for epoch in range(args.epochs):
        final_epoch = epoch
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(xb)

        model.eval()
        with torch.no_grad():
            val_x = torch.as_tensor(data["X_val"], dtype=torch.float32, device=device)
            val_y = torch.as_tensor(data["y_val"], dtype=torch.long, device=device)
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y).detach().cpu())
        val_metrics = source_validation_metrics(torch, model, data["X_val"], data["y_val"], device)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        should_stop = stale >= args.patience and can_apply_early_stop(epoch, args)
        val_rates = binary_rates_from_metrics(val_metrics)
        log_rows.append({
            "epoch": epoch,
            "seed": seed,
            "target_dataset": target_dataset,
            "variant": variant,
            "lambda_schedule": "constant_0",
            "min_epochs_before_early_stop": args.min_epochs_before_early_stop,
            "lambda_value": 0.0,
            "source_train_label_loss": total_loss / len(data["X_train"]),
            "source_val_label_loss": val_loss,
            "source_val_Binary-F1": val_metrics["Attack-class Binary-F1"],
            "source_val_MCC": val_metrics["MCC"],
            "source_val_FPR": val_metrics["FPR"],
            "source_val_FNR": val_metrics["FNR"],
            "source_val_true_attack_rate": val_rates["true_attack_rate"],
            "source_val_pred_attack_rate": val_rates["pred_attack_rate"],
            "domain_loss": np.nan,
            "domain_accuracy_train": np.nan,
            "learning_rate": args.learning_rate,
            "early_stop_flag": should_stop,
        })
        if should_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_val_metrics = best_val_metrics or {}
    best_val_rates = binary_rates_from_metrics(best_val_metrics) if best_val_metrics else {}
    training_metadata = {
        "early_stopping_mode": "source_val_loss",
        "min_epochs": args.min_epochs_before_early_stop,
        "patience": args.patience,
        "epochs": args.epochs,
        "final_epoch": final_epoch,
        "best_epoch": best_epoch,
        "final_lambda_value": final_lambda_value,
        "best_source_val_loss": best_val_loss,
        "best_source_val_Binary-F1": best_val_metrics.get("Attack-class Binary-F1", np.nan),
        "best_source_val_MCC": best_val_metrics.get("MCC", np.nan),
        "best_source_val_FPR": best_val_metrics.get("FPR", np.nan),
        "best_source_val_FNR": best_val_metrics.get("FNR", np.nan),
        "best_source_val_true_attack_rate": best_val_rates.get("true_attack_rate", np.nan),
        "best_source_val_pred_attack_rate": best_val_rates.get("pred_attack_rate", np.nan),
        "target_labels_used_for_model_selection": "No",
    }
    return model, device, training_metadata


def train_dann(data, X_target_adapt, input_dim, args, seed, variant, target_dataset, log_rows):
    import torch
    from torch import nn

    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = DANN(input_dim=input_dim, dropout=args.dropout, grl_alpha=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    label_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    source_loader = make_loader(torch, data["X_train"], data["y_train"], args.batch_size, seed)
    target_loader = make_loader(torch, X_target_adapt, None, args.batch_size, seed + 17)

    best_state = None
    best_val_loss = float("inf")
    best_val_metrics = None
    best_epoch = -1
    final_epoch = -1
    final_lambda_value = 0.0
    stale = 0
    for epoch in range(args.epochs):
        final_epoch = epoch
        model.train()
        lambda_value = lambda_value_for_epoch(args.lambda_schedule, epoch, args.epochs)
        final_lambda_value = lambda_value
        total_label_loss = 0.0
        total_domain_loss = 0.0
        domain_correct = 0
        domain_total = 0
        target_iter = iter(target_loader)
        for xs, ys in source_loader:
            try:
                (xt,) = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                (xt,) = next(target_iter)

            xs = xs.to(device)
            ys = ys.to(device)
            xt = xt.to(device)
            optimizer.zero_grad(set_to_none=True)

            label_logits, features_s = model.forward_label(xs, return_features=True)
            label_loss = label_criterion(label_logits, ys)
            with freeze_batchnorm_running_stats(model):
                _, features_t = model.forward_label(xt, return_features=True)
            features_all = torch.cat([features_s, features_t], dim=0)
            domain_labels = torch.cat([
                torch.zeros(len(features_s), dtype=torch.long, device=device),
                torch.ones(len(features_t), dtype=torch.long, device=device),
            ])
            domain_logits = model.forward_domain(features_all)
            domain_loss = domain_criterion(domain_logits, domain_labels)
            loss = label_loss + lambda_value * domain_loss
            loss.backward()
            optimizer.step()

            total_label_loss += float(label_loss.detach().cpu()) * len(xs)
            total_domain_loss += float(domain_loss.detach().cpu()) * len(features_all)
            domain_correct += int((domain_logits.argmax(dim=1) == domain_labels).sum().detach().cpu())
            domain_total += len(domain_labels)

        model.eval()
        with torch.no_grad():
            val_x = torch.as_tensor(data["X_val"], dtype=torch.float32, device=device)
            val_y = torch.as_tensor(data["y_val"], dtype=torch.long, device=device)
            val_logits = model.forward_label(val_x)
            val_loss = float(label_criterion(val_logits, val_y).detach().cpu())
        val_metrics = source_validation_metrics(torch, model, data["X_val"], data["y_val"], device)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        should_stop = stale >= args.patience and can_apply_early_stop(epoch, args)
        val_rates = binary_rates_from_metrics(val_metrics)
        log_rows.append({
            "epoch": epoch,
            "seed": seed,
            "target_dataset": target_dataset,
            "variant": variant,
            "experiment_name": CORRECTED_EXPERIMENT_NAME,
            **CORRECTED_METADATA,
            "lambda_schedule": args.lambda_schedule,
            "min_epochs_before_early_stop": args.min_epochs_before_early_stop,
            "lambda_value": lambda_value,
            "source_train_label_loss": total_label_loss / len(data["X_train"]),
            "source_val_label_loss": val_loss,
            "source_val_Binary-F1": val_metrics["Attack-class Binary-F1"],
            "source_val_MCC": val_metrics["MCC"],
            "source_val_FPR": val_metrics["FPR"],
            "source_val_FNR": val_metrics["FNR"],
            "source_val_true_attack_rate": val_rates["true_attack_rate"],
            "source_val_pred_attack_rate": val_rates["pred_attack_rate"],
            "domain_loss": total_domain_loss / max(domain_total, 1),
            "domain_accuracy_train": domain_correct / max(domain_total, 1),
            "learning_rate": args.learning_rate,
            "early_stop_flag": should_stop,
        })
        if should_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_val_metrics = best_val_metrics or {}
    best_val_rates = binary_rates_from_metrics(best_val_metrics) if best_val_metrics else {}
    training_metadata = {
        "early_stopping_mode": "source_val_loss",
        "min_epochs": args.min_epochs_before_early_stop,
        "patience": args.patience,
        "epochs": args.epochs,
        "final_epoch": final_epoch,
        "best_epoch": best_epoch,
        "final_lambda_value": final_lambda_value,
        "best_source_val_loss": best_val_loss,
        "best_source_val_Binary-F1": best_val_metrics.get("Attack-class Binary-F1", np.nan),
        "best_source_val_MCC": best_val_metrics.get("MCC", np.nan),
        "best_source_val_FPR": best_val_metrics.get("FPR", np.nan),
        "best_source_val_FNR": best_val_metrics.get("FNR", np.nan),
        "best_source_val_true_attack_rate": best_val_rates.get("true_attack_rate", np.nan),
        "best_source_val_pred_attack_rate": best_val_rates.get("pred_attack_rate", np.nan),
        "target_labels_used_for_model_selection": "No",
    }
    return model, device, training_metadata


def predict_logits_numpy(torch, model, X, device):
    model.eval()
    logits_all = []
    features_all = []
    with torch.no_grad():
        for start in range(0, len(X), 8192):
            xb = torch.as_tensor(X[start:start + 8192], dtype=torch.float32, device=device)
            if hasattr(model, "forward_label"):
                logits, features = model.forward_label(xb, return_features=True)
            else:
                logits, features = model(xb, return_features=True)
            logits_all.append(logits.detach().cpu().numpy())
            features_all.append(features.detach().cpu().numpy())
    return np.vstack(logits_all), np.vstack(features_all)


def evaluate_full_target(
    torch,
    model,
    device,
    preprocessor,
    feature_cols,
    dataset_name,
    dataset_path,
    seed,
    variant,
    output_dir,
    chunk_size,
    embedding_sample_size,
):
    embedding_parts = []
    label_parts = []
    sample_id_parts = []
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    sample_offset = 0
    rng = np.random.default_rng(seed)
    prediction_path = (
        output_dir
        / "predictions"
        / f"{variant}_UNSW_to_{dataset_name}_seed{seed}_predictions.csv"
    )
    if prediction_path.exists():
        prediction_path.unlink()
    wrote_header = False
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size, low_memory=False):
        chunk = prepare_cicids_chunk(chunk)
        y_true = chunk[LABEL_COL].astype(int).to_numpy()
        X = chunk.drop(columns=[c for c in [LABEL_COL, ATTACK_COL] if c in chunk.columns])
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_cols]
        Xt = preprocessor.transform(X).astype(np.float32)
        logits, features = predict_logits_numpy(torch, model, Xt, device)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        y_pred = (probs[:, 1] >= 0.5).astype(int)
        counts["tp"] += int(((y_true == 1) & (y_pred == 1)).sum())
        counts["tn"] += int(((y_true == 0) & (y_pred == 0)).sum())
        counts["fp"] += int(((y_true == 0) & (y_pred == 1)).sum())
        counts["fn"] += int(((y_true == 1) & (y_pred == 0)).sum())
        pd.DataFrame({
            "sample_id": np.arange(sample_offset, sample_offset + len(y_true)),
            "source_dataset": "UNSW",
            "target_dataset": dataset_name,
            "seed": seed,
            "variant": variant,
            "y_true_binary": y_true,
            "y_pred_binary": y_pred,
            "p_benign": probs[:, 0],
            "p_attack": probs[:, 1],
        }).to_csv(prediction_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True

        current_embedding_n = sum(len(part) for part in embedding_parts)
        remaining = max(embedding_sample_size - current_embedding_n, 0)
        if remaining > 0:
            take = min(remaining, len(features))
            idx = rng.choice(len(features), size=take, replace=False)
            embedding_parts.append(features[idx])
            label_parts.append(y_true[idx])
            sample_id_parts.append(sample_offset + idx)
        sample_offset += len(y_true)
        del chunk, X, Xt, logits, probs
        gc.collect()

    metrics = binary_metrics_from_counts(counts["tp"], counts["tn"], counts["fp"], counts["fn"])
    target_embeddings = (
        np.vstack(embedding_parts)
        if embedding_parts
        else np.empty((0, 64), dtype=np.float32)
    )
    target_labels = (
        np.concatenate(label_parts)
        if label_parts
        else np.empty((0,), dtype=np.int64)
    )
    target_sample_ids = (
        np.concatenate(sample_id_parts)
        if sample_id_parts
        else np.empty((0,), dtype=np.int64)
    )
    return metrics, target_embeddings, target_labels, target_sample_ids


def source_embeddings(torch, model, device, data):
    _, features = predict_logits_numpy(torch, model, data["X_test"], device)
    sample_ids = np.arange(len(data["y_test"]), dtype=np.int64)
    return features, data["y_test"], sample_ids


def save_embedding_sample_npz(
    output_dir,
    target_dataset,
    variant,
    run_seed,
    feature_set,
    lambda_schedule,
    source_emb,
    source_labels,
    source_sample_ids,
    target_emb,
    target_labels,
    target_sample_ids,
):
    path = (
        output_dir
        / "embeddings"
        / f"embeddings_{variant}_UNSW_to_{target_dataset}_seed{run_seed}.npz"
    )
    np.savez_compressed(
        path,
        source_embeddings=source_emb,
        source_binary_labels=source_labels,
        source_sample_ids=source_sample_ids,
        target_embeddings=target_emb,
        target_binary_labels=target_labels,
        target_sample_ids=target_sample_ids,
        source_dataset="NF-UNSW-NB15-v3",
        target_dataset=target_dataset,
        variant=variant,
        seed=run_seed,
        run_seed=run_seed,
        feature_set=feature_set,
        lambda_schedule=lambda_schedule,
        experiment_name=CORRECTED_EXPERIMENT_NAME,
        bn_control_mode=CORRECTED_METADATA["bn_control_mode"],
        batchnorm_policy=CORRECTED_METADATA["batchnorm_policy"],
        source_forward_updates_bn_running_stats=(
            CORRECTED_METADATA["source_forward_updates_bn_running_stats"]
        ),
        target_forward_updates_bn_running_stats=(
            CORRECTED_METADATA["target_forward_updates_bn_running_stats"]
        ),
        corrected_from=CORRECTED_METADATA["corrected_from"],
        reason_for_correction=CORRECTED_METADATA["reason_for_correction"],
        original_results_status=CORRECTED_METADATA["original_results_status"],
        target_labels_used_only_posthoc="Yes",
    )


def domain_discriminability(source_emb, target_emb, seed):
    n = min(len(source_emb), len(target_emb), 100_000)
    rng = np.random.default_rng(seed)
    source_idx = rng.choice(len(source_emb), size=n, replace=False)
    target_idx = rng.choice(len(target_emb), size=n, replace=False)
    X = np.vstack([source_emb[source_idx], target_emb[target_idx]])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=seed,
    )
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auroc = roc_auc_score(y_test, proba)
    return {
        "domain_classifier_type": "LogisticRegression",
        "domain_AUROC": auroc,
        "domain_accuracy": accuracy_score(y_test, pred),
        "domain_balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "source_embedding_n": n,
        "target_embedding_n": n,
        "heldout_split_ratio": 0.3,
        "interpretation": (
            "closer_to_0.5_less_domain_discriminable"
            if auroc < 0.75
            else "source_target_embeddings_remain_discriminable"
        ),
    }




def write_config(output_dir, config):
    path = output_dir / "configs" / (
        f"h3_dann_config_UNSW_to_{config.target_dataset}_seed{config.seed}_{config.variant}.json"
    )
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def run_variant(target_dataset, dataset_path, variant, variant_spec, args, run_seed, output_dir):
    import torch

    removed = []
    all_source = load_and_clean_data(UNSW_PATH)
    all_source = clean_l7_proto(all_source)
    base_feature_cols = list(all_source.drop(columns=[LABEL_COL, ATTACK_COL]).columns)
    del all_source

    if variant_spec["use_filtered_features"]:
        removed = load_removed_features(
            target_dataset,
            args.filtered_feature_set,
            model="MLP",
        )
    feature_cols = [c for c in base_feature_cols if c not in set(removed)]
    feature_set_label = (
        args.filtered_feature_set
        if variant_spec["use_filtered_features"]
        else variant_spec["feature_set"]
    )
    data = prepare_source_splits(split_seed=SOURCE_SPLIT_SEED, feature_cols=feature_cols)
    preprocessor_path = (
        output_dir
        / "configs"
        / f"source_preprocessor_UNSW_to_{target_dataset}_seed{run_seed}_{variant}.joblib"
    )
    joblib.dump(data["preprocessor"], preprocessor_path)

    X_target_adapt = np.empty((0, data["X_train"].shape[1]), dtype=np.float32)
    sample_meta = {
        "target_dataset": target_dataset,
        "target_total_samples": 0,
        "target_adaptation_sample_ratio": 0.0,
        "target_adaptation_sample_ratio_requested": 0.0,
        "target_adaptation_sample_size": 0,
        "target_adaptation_effective_ratio": 0.0,
        "target_adaptation_sample_cap": None,
        "sampling_method": "not_used_source_only",
        "target_sampling_method_code": "not_used_source_only",
        "chunk_selection_rule": "",
        "sampling_seed": run_seed,
        "target_labels_used_for_sampling": "No",
    }
    if variant_spec["use_dann"]:
        X_target_adapt, sample_meta = label_free_target_sample(
            target_dataset,
            dataset_path,
            feature_cols,
            data["preprocessor"],
            run_seed,
            args.target_sample_ratio,
            args.max_target_adaptation_samples,
            args.chunk_size,
            args.target_sampling_method,
        )

    logs = []
    input_dim = data["X_train"].shape[1]
    if variant_spec["use_dann"]:
        if len(X_target_adapt) == 0:
            raise RuntimeError(
                f"No label-free target adaptation samples collected for "
                f"{target_dataset} seed={run_seed}; DANN training cannot proceed."
            )
        model, device, training_metadata = train_dann(
            data,
            X_target_adapt,
            input_dim,
            args,
            run_seed,
            variant,
            target_dataset,
            logs,
        )
    else:
        model, device, training_metadata = train_source_mlp(
            data,
            input_dim,
            args,
            run_seed,
            variant,
            target_dataset,
            logs,
        )

    metrics, target_emb, target_labels, target_sample_ids = evaluate_full_target(
        torch,
        model,
        device,
        data["preprocessor"],
        feature_cols,
        target_dataset,
        dataset_path,
        run_seed,
        variant,
        output_dir,
        args.chunk_size,
        args.embedding_sample_size,
    )
    source_emb, source_labels, source_sample_ids = source_embeddings(torch, model, device, data)
    domain_metrics = domain_discriminability(source_emb, target_emb, run_seed)

    embedding_source = {
        "h3_mlp": "standard_binary_MLP_embedding",
        "h3_dann": "binary_DANN_embedding",
        "h3_filtered_mlp": "filtered_binary_MLP_embedding",
        "h3_filtered_dann": "filtered_binary_DANN_embedding",
    }[variant]
    lambda_schedule = args.lambda_schedule if variant_spec["use_dann"] else "constant_0"

    if args.save_embedding_samples:
        save_embedding_sample_npz(
            output_dir,
            target_dataset,
            variant,
            run_seed,
            feature_set_label,
            lambda_schedule,
            source_emb,
            source_labels,
            source_sample_ids,
            target_emb,
            target_labels,
            target_sample_ids,
        )

    config = H3Config(
        experiment_name=CORRECTED_EXPERIMENT_NAME,
        **CORRECTED_METADATA,
        source_dataset="NF-UNSW-NB15-v3",
        target_dataset=target_dataset,
        variant=variant,
        experiment_mode="debug_smoke_test" if args.debug else "formal",
        seed=run_seed,
        source_split_seed=SOURCE_SPLIT_SEED,
        run_seed=run_seed,
        feature_list=feature_cols,
        removed_feature_list=removed,
        preprocessor_path=str(preprocessor_path),
        target_total_samples=int(sample_meta.get("target_total_samples", 0)),
        target_adaptation_sample_ratio=(
            args.target_sample_ratio if variant_spec["use_dann"] else 0.0
        ),
        target_adaptation_sample_ratio_requested=sample_meta.get(
            "target_adaptation_sample_ratio_requested",
            args.target_sample_ratio if variant_spec["use_dann"] else 0.0,
        ),
        target_adaptation_sample_size=int(sample_meta["target_adaptation_sample_size"]),
        target_adaptation_effective_ratio=float(
            sample_meta.get("target_adaptation_effective_ratio", 0.0)
        ),
        target_adaptation_sample_cap=sample_meta.get("target_adaptation_sample_cap"),
        target_sampling_method=sample_meta["sampling_method"],
        target_sampling_method_code=sample_meta.get("target_sampling_method_code", ""),
        chunk_selection_rule=sample_meta.get("chunk_selection_rule", ""),
        target_labels_used_for_sampling="No",
        lambda_schedule=lambda_schedule,
        min_epochs_before_early_stop=args.min_epochs_before_early_stop,
        early_stopping_mode=training_metadata["early_stopping_mode"],
        min_epochs=training_metadata["min_epochs"],
        patience=training_metadata["patience"],
        final_epoch=training_metadata["final_epoch"],
        best_epoch=training_metadata["best_epoch"],
        final_lambda_value=training_metadata["final_lambda_value"],
        best_source_val_loss=training_metadata["best_source_val_loss"],
        target_labels_used_for_model_selection=(
            training_metadata["target_labels_used_for_model_selection"]
        ),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_rule="source_validation_loss_only",
        target_labels_used_in_training="No",
        evaluation_protocol="full target dataset",
        grl_alpha=1.0,
        loss_weighting_style="label_loss + lambda * domain_loss",
    )
    write_config(output_dir, config)

    result_row = {
        "experiment_name": CORRECTED_EXPERIMENT_NAME,
        "hypothesis_tested": "H3 representation misalignment",
        **CORRECTED_METADATA,
        "source_dataset": "NF-UNSW-NB15-v3",
        "target_dataset": target_dataset,
        "model": variant_spec["model"],
        "method": variant_spec["method"],
        "variant": variant,
        "experiment_mode": "debug_smoke_test" if args.debug else "formal",
        "feature_set": feature_set_label,
        "training_data_used": (
            f"UNSW labeled + {target_dataset} unlabeled"
            if variant_spec["use_dann"]
            else "UNSW labeled only"
        ),
        "unlabeled_target_data_used": variant_spec["unlabeled_target_data_used"],
        "target_labels_used_in_training": "No",
        "target_labels_used_for_sampling": "No",
        "evaluation_protocol": "full target",
        "source_split_seed": SOURCE_SPLIT_SEED,
        "seed": run_seed,
        "run_seed": run_seed,
        "lambda_schedule": lambda_schedule,
        "min_epochs_before_early_stop": args.min_epochs_before_early_stop,
        **training_metadata,
        **metrics,
        "notes": "Target labels used only after training for final evaluation.",
    }
    domain_row = {
        "experiment_name": CORRECTED_EXPERIMENT_NAME,
        **CORRECTED_METADATA,
        "source_dataset": "NF-UNSW-NB15-v3",
        "target_dataset": target_dataset,
        "source_split_seed": SOURCE_SPLIT_SEED,
        "seed": run_seed,
        "run_seed": run_seed,
        "feature_set": feature_set_label,
        "lambda_schedule": lambda_schedule,
        "min_epochs_before_early_stop": args.min_epochs_before_early_stop,
        "experiment_mode": "debug_smoke_test" if args.debug else "formal",
        "variant": variant,
        "embedding_source": embedding_source,
        **domain_metrics,
    }
    sampling_row = {
        "experiment_name": CORRECTED_EXPERIMENT_NAME,
        **CORRECTED_METADATA,
        "target_dataset": target_dataset,
        "variant": variant,
        "experiment_mode": "debug_smoke_test" if args.debug else "formal",
        "source_split_seed": SOURCE_SPLIT_SEED,
        "seed": run_seed,
        "run_seed": run_seed,
        "lambda_schedule": lambda_schedule,
        "min_epochs_before_early_stop": args.min_epochs_before_early_stop,
        "target_total_samples": int(sample_meta.get("target_total_samples", 0)),
        "target_adaptation_sample_ratio": sample_meta.get(
            "target_adaptation_sample_ratio",
            args.target_sample_ratio if variant_spec["use_dann"] else 0.0,
        ),
        "target_adaptation_sample_ratio_requested": sample_meta.get(
            "target_adaptation_sample_ratio_requested",
            args.target_sample_ratio if variant_spec["use_dann"] else 0.0,
        ),
        "target_adaptation_sample_size": int(sample_meta["target_adaptation_sample_size"]),
        "target_adaptation_effective_ratio": float(
            sample_meta.get("target_adaptation_effective_ratio", 0.0)
        ),
        "target_adaptation_sample_cap": sample_meta.get("target_adaptation_sample_cap"),
        "sampling_method": sample_meta["sampling_method"],
        "target_sampling_method_code": sample_meta.get("target_sampling_method_code", ""),
        "chunk_selection_rule": sample_meta.get("chunk_selection_rule", ""),
        "sampling_seed": run_seed,
        "target_labels_used_for_sampling": "No",
    }
    training_summary_row = {
        "experiment_name": CORRECTED_EXPERIMENT_NAME,
        **CORRECTED_METADATA,
        "target_dataset": target_dataset,
        "variant": variant,
        "experiment_mode": "debug_smoke_test" if args.debug else "formal",
        "source_split_seed": SOURCE_SPLIT_SEED,
        "seed": run_seed,
        "run_seed": run_seed,
        "feature_set": feature_set_label,
        "lambda_schedule": lambda_schedule,
        **training_metadata,
    }
    return result_row, domain_row, logs, sampling_row, training_summary_row


def write_mean_std(results, output_dir):
    df = pd.DataFrame(results)
    metrics = [
        "Attack-class Binary-F1",
        "MCC",
        "FPR",
        "FNR",
        "Binary weighted-F1 secondary",
        "TP",
        "TN",
        "FP",
        "FN",
    ]
    grouped = df.groupby([
        "target_dataset",
        "variant",
        "feature_set",
        "lambda_schedule",
        "min_epochs_before_early_stop",
    ], as_index=False)
    rows = []
    for keys, group in grouped:
        target_dataset, variant, feature_set, lambda_schedule, min_epochs = keys
        row = {
            "experiment_name": CORRECTED_EXPERIMENT_NAME,
            **CORRECTED_METADATA,
            "target_dataset": target_dataset,
            "variant": variant,
            "feature_set": feature_set,
            "lambda_schedule": lambda_schedule,
            "min_epochs_before_early_stop": min_epochs,
            "notes": "mean and std over seeds 42, 123, 2024",
        }
        for metric in metrics:
            label = metric.replace("Attack-class ", "").replace(" secondary", "")
            row[f"{label} mean"] = group[metric].mean()
            row[f"{label} std"] = group[metric].std(ddof=1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "h3_dann_results_mean_std.csv", index=False)


def write_domain_mean_std(domain_rows, output_dir):
    df = pd.DataFrame(domain_rows)
    grouped = df.groupby([
        "target_dataset",
        "variant",
        "feature_set",
        "lambda_schedule",
        "min_epochs_before_early_stop",
    ], as_index=False)
    rows = []
    for keys, group in grouped:
        target_dataset, variant, feature_set, lambda_schedule, min_epochs = keys
        domain_auroc_mean = group["domain_AUROC"].mean()
        rows.append({
            "experiment_name": CORRECTED_EXPERIMENT_NAME,
            **CORRECTED_METADATA,
            "target_dataset": target_dataset,
            "variant": variant,
            "feature_set": feature_set,
            "lambda_schedule": lambda_schedule,
            "min_epochs_before_early_stop": min_epochs,
            "domain_AUROC_mean": domain_auroc_mean,
            "domain_AUROC_std": group["domain_AUROC"].std(ddof=1),
            "domain_accuracy_mean": group["domain_accuracy"].mean(),
            "domain_accuracy_std": group["domain_accuracy"].std(ddof=1),
            "domain_balanced_accuracy_mean": group["domain_balanced_accuracy"].mean(),
            "domain_balanced_accuracy_std": group["domain_balanced_accuracy"].std(ddof=1),
            "source_embedding_n_mean": group["source_embedding_n"].mean(),
            "target_embedding_n_mean": group["target_embedding_n"].mean(),
            "interpretation": (
                "closer_to_0.5_less_domain_discriminable"
                if domain_auroc_mean < 0.75
                else "source_target_embeddings_remain_discriminable"
            ),
        })
    pd.DataFrame(rows).to_csv(
        output_dir / "h3_domain_discriminability_mean_std.csv",
        index=False,
    )


def write_summary_with_domain(output_dir):
    results_path = output_dir / "h3_dann_results_mean_std.csv"
    domain_path = output_dir / "h3_domain_discriminability_mean_std.csv"
    if not results_path.exists() or not domain_path.exists():
        return
    results = pd.read_csv(results_path)
    domain = pd.read_csv(domain_path)
    merged = results.merge(
        domain,
        on=[
            "experiment_name",
            "bn_control_mode",
            "batchnorm_policy",
            "source_forward_updates_bn_running_stats",
            "target_forward_updates_bn_running_stats",
            "corrected_from",
            "reason_for_correction",
            "original_results_status",
            "target_dataset",
            "variant",
            "feature_set",
            "lambda_schedule",
            "min_epochs_before_early_stop",
        ],
        how="left",
    )
    merged.to_csv(output_dir / "h3_dann_summary_with_domain_mean_std.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=STRESS_B_CHUNK_SIZE)
    parser.add_argument(
        "--target-sample-ratio",
        "--target-adaptation-sample-ratio",
        dest="target_sample_ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max-target-adaptation-samples",
        type=parse_optional_sample_cap,
        default=None,
        help="Optional cap for smoke tests only. Formal H3 should use None or 0.",
    )
    parser.add_argument("--embedding-sample-size", type=int, default=100_000)
    parser.add_argument(
        "--target-sampling-method",
        choices=["exact_random_5pct", "chunk_exact_5pct"],
        default="exact_random_5pct",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--lambda-schedule",
        choices=["linear_0_0.1", "linear_0_1", "constant_1", "linear_0_1_min_epochs"],
        default="linear_0_1",
    )
    parser.add_argument(
        "--min-epochs-before-early-stop",
        "--min-epochs",
        dest="min_epochs_before_early_stop",
        type=int,
        default=0,
    )
    parser.add_argument("--filtered-feature-set", default="remove_shift_top5_plus_corr_gt_0.98")
    parser.add_argument("--targets", "--target", nargs="+", default=list(TARGET_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--variants", nargs="+", default=DANN_VARIANTS)
    parser.add_argument("--save-embedding-samples", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    unknown_variants = sorted(set(args.variants) - set(VARIANTS))
    if unknown_variants:
        raise ValueError(f"Unknown H3 variants: {unknown_variants}")
    non_dann_variants = [
        variant for variant in args.variants if not VARIANTS[variant]["use_dann"]
    ]
    if non_dann_variants:
        raise ValueError(
            "Corrected Frozen-BN H3 only re-runs DANN variants. "
            f"Remove non-DANN variants: {non_dann_variants}"
        )

    if not args.debug and args.max_target_adaptation_samples is not None:
        raise ValueError(
            "Capped target adaptation sampling is only allowed with --debug. "
            "Formal H3 must use uncapped label-free random 5% sampling."
        )

    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else (
            REPOSITORY_ROOT / "results" / "generated" / "h3" / "domain_loss_models_smoke"
            if args.debug
            else REPOSITORY_ROOT / "results" / "generated" / "h3" / "domain_loss_models"
        )
    )
    guard_output_dir(output_dir)
    for subdir in ["predictions", "logs", "configs", "embeddings"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    all_results = []
    domain_rows = []
    training_logs = []
    sampling_rows = []
    training_summary_rows = []
    for run_seed in args.seeds:
        for target in args.targets:
            dataset_path = TARGET_DATASETS[target]
            for variant in args.variants:
                result, domain_row, logs, sampling_row, training_summary_row = run_variant(
                    target,
                    dataset_path,
                    variant,
                    VARIANTS[variant],
                    args,
                    run_seed,
                    output_dir,
                )
                all_results.append(result)
                domain_rows.append(domain_row)
                training_logs.extend(logs)
                sampling_rows.append(sampling_row)
                training_summary_rows.append(training_summary_row)
                pd.DataFrame(all_results).to_csv(output_dir / "h3_dann_results_long.csv", index=False)
                pd.DataFrame(domain_rows).to_csv(output_dir / "h3_domain_discriminability_auroc.csv", index=False)
                write_domain_mean_std(domain_rows, output_dir)
                pd.DataFrame(sampling_rows).to_csv(
                    output_dir / "target_adaptation_sampling_metadata.csv",
                    index=False,
                )
                pd.DataFrame(training_logs).to_csv(
                    output_dir / "logs" / "h3_dann_training_log.csv",
                    index=False,
                )
                pd.DataFrame(training_summary_rows).to_csv(
                    output_dir / "logs" / "h3_dann_training_summary.csv",
                    index=False,
                )

    write_mean_std(all_results, output_dir)
    write_domain_mean_std(domain_rows, output_dir)
    write_summary_with_domain(output_dir)
    print_flush(f"Saved H3 DANN outputs under {output_dir}")


if __name__ == "__main__":
    main()
