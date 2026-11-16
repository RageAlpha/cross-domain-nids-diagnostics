"""
Full-target feature-space shift quantification for H2.

This script is intentionally independent from the training runners. It does
not train models; it computes post-hoc feature distribution shift on the 49
config-ordered NetFlow modeling features.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np
import pandas as pd

from cross_domain_nids.config import (
    ATTACK_COL,
    BOTIOT_PATH,
    CATEGORICAL_FEATURES,
    CICIDS_PATH,
    DROP_COLUMNS,
    LABEL_COL,
    NUMERIC_FEATURES,
    PROTOCOL_FEATURES,
    TONIOT_PATH,
    UNSW_PATH,
)


SOURCE_DATASET = "NF-UNSW-NB15-v3"
SOURCE_SPLIT = "train_only"
SOURCE_DISTRIBUTION_MODE = "UNSW_train_split_full"
TARGET_DISTRIBUTION_MODE = "full_target_feature_distribution"
TARGET_LABEL_USAGE = "not_used"
EXPECTED_NUM_FEATURES = 49
EPS = 1e-12
NAN_CATEGORY = "__FEATURE_SHIFT_NAN__"
DEFAULT_TARGETS = ["CICIDS2018", "BoT-IoT", "ToN-IoT"]
TARGET_FULL_NAMES = {
    "CICIDS2018": "NF-CSE-CIC-IDS2018-v3",
    "BoT-IoT": "NF-BoT-IoT-v3",
    "ToN-IoT": "NF-ToN-IoT-v3",
}
IMPORTANCE_SHIFT_JOIN_COLUMNS = [
    "seed",
    "target_dataset",
    "feature",
    "feature_family",
    "rf_importance",
    "importance_rank",
    "robust_shift_score",
    "standard_smd",
    "abs_standard_smd",
    "jsd",
    "importance_x_jsd",
    "importance_x_abs_smd",
]
IMPORTANCE_SHIFT_CORRELATION_COLUMNS = [
    "target_dataset",
    "metric",
    "spearman_corr_importance_vs_shift",
    "p_value",
]


@dataclass
class CleaningAudit:
    num_inf_before_cleaning: int
    num_inf_after_cleaning: int
    num_l7_proto_non_integer_before_cleaning: int
    num_l7_proto_non_integer_after_cleaning: int
    missing_rate_mean: float
    missing_rate_max: float
    protocol_missing_rate: float
    l7_proto_missing_rate: float


@dataclass
class SourceProfile:
    seed: int
    frame: pd.DataFrame
    stats: pd.DataFrame
    categorical_counts: Dict[str, Counter]
    audit: CleaningAudit


@dataclass
class TargetProfile:
    target_dataset: str
    path: Path
    stats: pd.DataFrame
    quantile_sample: pd.DataFrame
    categorical_counts: Dict[str, Counter]
    total_rows: int
    target_quantile_method: str
    target_quantile_sample_rows: int
    quantile_estimation: str


class StreamingFeatureStats:
    def __init__(self, feature_columns: Sequence[str]):
        self.feature_columns = list(feature_columns)
        self.total_rows = 0
        self.count = pd.Series(0, index=self.feature_columns, dtype="int64")
        self.missing = pd.Series(0, index=self.feature_columns, dtype="int64")
        self.mean = pd.Series(0.0, index=self.feature_columns, dtype="float64")
        self.m2 = pd.Series(0.0, index=self.feature_columns, dtype="float64")
        self.min = pd.Series(np.nan, index=self.feature_columns, dtype="float64")
        self.max = pd.Series(np.nan, index=self.feature_columns, dtype="float64")

    def update(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return

        frame = frame[self.feature_columns]
        self.total_rows += len(frame)
        is_missing = frame.isna()
        self.missing += is_missing.sum()

        chunk_count = frame.notna().sum().astype("int64")
        chunk_mean = frame.mean(skipna=True)
        chunk_var = frame.var(skipna=True)
        chunk_m2 = (chunk_var.fillna(0.0) * (chunk_count - 1).clip(lower=0)).astype(float)

        old_count = self.count.copy()
        old_mean = self.mean.copy()
        new_count = old_count + chunk_count
        has_values = chunk_count > 0
        already_has_values = old_count > 0
        combine_mask = has_values & already_has_values
        initialize_mask = has_values & ~already_has_values

        delta = chunk_mean.fillna(0.0) - old_mean
        self.mean.loc[combine_mask] = (
            old_mean.loc[combine_mask]
            + delta.loc[combine_mask]
            * chunk_count.loc[combine_mask]
            / new_count.loc[combine_mask]
        )
        self.m2.loc[combine_mask] = (
            self.m2.loc[combine_mask]
            + chunk_m2.loc[combine_mask]
            + (delta.loc[combine_mask] ** 2)
            * old_count.loc[combine_mask]
            * chunk_count.loc[combine_mask]
            / new_count.loc[combine_mask]
        )
        self.mean.loc[initialize_mask] = chunk_mean.loc[initialize_mask]
        self.m2.loc[initialize_mask] = chunk_m2.loc[initialize_mask]
        self.count = new_count

        chunk_min = frame.min(skipna=True)
        chunk_max = frame.max(skipna=True)
        self.min = pd.concat([self.min, chunk_min], axis=1).min(axis=1, skipna=True)
        self.max = pd.concat([self.max, chunk_max], axis=1).max(axis=1, skipna=True)

    def to_frame(self, prefix: str, quantile_stats: pd.DataFrame) -> pd.DataFrame:
        n = self.count.astype(float)
        mean = self.mean.where(self.count > 0, np.nan)
        var = self.m2 / (n - 1).replace(0, np.nan)
        var = var.clip(lower=0)
        std = np.sqrt(var)

        out = pd.DataFrame(
            {
                "feature": self.feature_columns,
                f"{prefix}_n": self.total_rows,
                f"{prefix}_non_missing_n": self.count.values,
                f"{prefix}_missing_rate": (
                    self.missing / max(self.total_rows, 1)
                ).values,
                f"{prefix}_mean": mean.values,
                f"{prefix}_std": std.values,
                f"{prefix}_var": var.values,
                f"{prefix}_min": self.min.values,
                f"{prefix}_max": self.max.values,
            }
        )
        return out.merge(quantile_stats, on="feature", how="left")


class ReservoirSampler:
    """Keep the rows with the smallest deterministic random keys."""

    def __init__(self, max_rows: int, feature_columns: Sequence[str], seed: int):
        self.max_rows = int(max_rows)
        self.feature_columns = list(feature_columns)
        self.rng = np.random.default_rng(seed)
        self.sample = pd.DataFrame(columns=["_feature_shift_key", *self.feature_columns])

    def update(self, frame: pd.DataFrame) -> None:
        if self.max_rows <= 0 or frame.empty:
            return
        keyed = frame[self.feature_columns].copy()
        keyed.insert(0, "_feature_shift_key", self.rng.random(len(keyed)))
        if self.sample.empty:
            combined = keyed
        else:
            combined = pd.concat([self.sample, keyed], ignore_index=True)
        if len(combined) > self.max_rows:
            combined = combined.nsmallest(self.max_rows, "_feature_shift_key")
        self.sample = combined.reset_index(drop=True)

    def to_frame(self) -> pd.DataFrame:
        if self.sample.empty:
            return pd.DataFrame(columns=self.feature_columns)
        return self.sample.drop(columns=["_feature_shift_key"]).reset_index(drop=True)


def print_flush(message: str) -> None:
    print(message, flush=True)


def feature_shift_feature_columns() -> List[str]:
    feature_columns = list(NUMERIC_FEATURES) + list(PROTOCOL_FEATURES) + list(CATEGORICAL_FEATURES)
    duplicate_features = sorted(
        feature for feature in set(feature_columns) if feature_columns.count(feature) > 1
    )
    if duplicate_features:
        raise ValueError(f"Duplicate Feature shift feature columns: {duplicate_features}")
    if len(feature_columns) != EXPECTED_NUM_FEATURES:
        raise ValueError(
            f"Feature shift must use {EXPECTED_NUM_FEATURES} features; found {len(feature_columns)}."
        )

    forbidden = forbidden_columns_in_features(feature_columns)
    if forbidden:
        raise ValueError(f"Forbidden non-modeling columns in Feature shift feature list: {forbidden}")
    return feature_columns


def forbidden_columns_in_features(feature_columns: Sequence[str]) -> List[str]:
    forbidden_exact = set(DROP_COLUMNS + [LABEL_COL, ATTACK_COL])
    forbidden_patterns = [
        re.compile(r"(^|_)IPV4_(SRC|DST)_ADDR$", re.IGNORECASE),
        re.compile(r"(^|_)ADDR$", re.IGNORECASE),
        re.compile("timestamp", re.IGNORECASE),
        re.compile("flow_start", re.IGNORECASE),
        re.compile("flow_end", re.IGNORECASE),
    ]
    found = []
    for feature in feature_columns:
        if feature in forbidden_exact or any(pattern.search(feature) for pattern in forbidden_patterns):
            found.append(feature)
    return found


def has_ip_identifier_columns(columns: Sequence[str]) -> bool:
    return any(col in {"IPV4_SRC_ADDR", "IPV4_DST_ADDR"} or col.upper().endswith("_ADDR") for col in columns)


def has_timestamp_columns(columns: Sequence[str]) -> bool:
    return any("TIME" in col.upper() or col in {"FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"} for col in columns)


def target_paths(args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "CICIDS2018": Path(args.cicids_path),
        "BoT-IoT": Path(args.bot_iot_path),
        "ToN-IoT": Path(args.ton_iot_path),
    }


def ensure_feature_columns_exist(path: Path, dataset: str, feature_columns: Sequence[str]) -> pd.Index:
    if not path.exists():
        raise FileNotFoundError(f"{dataset} path does not exist: {path}")
    header = pd.read_csv(path, nrows=0).columns
    missing = [feature for feature in feature_columns if feature not in header]
    if missing:
        raise ValueError(f"{dataset} is missing Feature shift features: {missing}")
    return header


def count_inf_values(frame: pd.DataFrame) -> int:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float, copy=False)
    return int(np.isinf(values).sum())


def count_l7_non_integer(frame: pd.DataFrame) -> int:
    if "L7_PROTO" not in frame.columns:
        return 0
    values = pd.to_numeric(frame["L7_PROTO"], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return 0
    return int(((finite % 1) != 0).sum())


def minimal_clean_features(frame: pd.DataFrame, feature_columns: Sequence[str]) -> tuple[pd.DataFrame, CleaningAudit]:
    selected = frame.loc[:, list(feature_columns)].copy()
    inf_before = count_inf_values(selected)
    l7_non_integer_before = count_l7_non_integer(selected)

    cleaned = selected.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    if "L7_PROTO" in cleaned.columns:
        cleaned["L7_PROTO"] = cleaned["L7_PROTO"].round()

    inf_after = count_inf_values(cleaned)
    l7_non_integer_after = count_l7_non_integer(cleaned)
    missing_rates = cleaned.isna().mean()
    audit = CleaningAudit(
        num_inf_before_cleaning=inf_before,
        num_inf_after_cleaning=inf_after,
        num_l7_proto_non_integer_before_cleaning=l7_non_integer_before,
        num_l7_proto_non_integer_after_cleaning=l7_non_integer_after,
        missing_rate_mean=float(missing_rates.mean()),
        missing_rate_max=float(missing_rates.max()),
        protocol_missing_rate=(
            float(missing_rates["PROTOCOL"]) if "PROTOCOL" in missing_rates.index else np.nan
        ),
        l7_proto_missing_rate=(
            float(missing_rates["L7_PROTO"]) if "L7_PROTO" in missing_rates.index else np.nan
        ),
    )
    return cleaned, audit


def categorical_jsd_features() -> List[str]:
    return list(PROTOCOL_FEATURES) + list(CATEGORICAL_FEATURES)


def numeric_jsd_features(feature_columns: Sequence[str]) -> List[str]:
    categorical = set(categorical_jsd_features())
    return [feature for feature in feature_columns if feature not in categorical]


def update_categorical_counts(
    counts: MutableMapping[str, Counter],
    frame: pd.DataFrame,
    categorical_features: Sequence[str],
) -> None:
    for feature in categorical_features:
        values = frame[feature]
        keys = values.where(values.notna(), NAN_CATEGORY).astype(str)
        counts[feature].update(keys.value_counts().to_dict())


def categorical_counts_from_frame(
    frame: pd.DataFrame,
    categorical_features: Sequence[str],
) -> Dict[str, Counter]:
    counts = {feature: Counter() for feature in categorical_features}
    update_categorical_counts(counts, frame, categorical_features)
    return counts


def compute_quantile_stats(frame: pd.DataFrame, feature_columns: Sequence[str], prefix: str) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        series = frame[feature]
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        rows.append(
            {
                "feature": feature,
                f"{prefix}_median": float(series.median()) if series.notna().any() else np.nan,
                f"{prefix}_iqr": float(q75 - q25) if pd.notna(q25) and pd.notna(q75) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_source_stats(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    streamer = StreamingFeatureStats(feature_columns)
    streamer.update(frame)
    quantiles = compute_quantile_stats(frame, feature_columns, "source")
    return streamer.to_frame("source", quantiles)


def load_source_train_profile(seed: int, unsw_path: Path, feature_columns: Sequence[str]) -> SourceProfile:
    header = ensure_feature_columns_exist(unsw_path, SOURCE_DATASET, feature_columns)
    for required in [LABEL_COL, ATTACK_COL]:
        if required not in header:
            raise ValueError(f"{SOURCE_DATASET} is missing required split label column: {required}")

    print_flush(f"[FeatureShift] Loading UNSW source train split via H1: seed={seed}")
    from experiments.h1 import run_multiseed_training as h1_runner

    original_unsw_path = h1_runner.UNSW_PATH
    h1_runner.UNSW_PATH = str(unsw_path)
    try:
        data = h1_runner.load_unsw_splits(seed, "binary")
    finally:
        h1_runner.UNSW_PATH = original_unsw_path

    if list(data["feature_cols"]) != list(feature_columns):
        raise ValueError(
            "Feature-shift order differs from the H1 source split feature order: "
            f"feature_shift={list(feature_columns)}, source={list(data['feature_cols'])}"
        )

    clean_train, audit = minimal_clean_features(data["X_train"], feature_columns)
    stats = compute_source_stats(clean_train, feature_columns)
    counts = categorical_counts_from_frame(clean_train, categorical_jsd_features())
    return SourceProfile(
        seed=seed,
        frame=clean_train,
        stats=stats,
        categorical_counts=counts,
        audit=audit,
    )


def precheck_source_row(
    seed: int,
    unsw_path: Path,
    source_profile: SourceProfile,
    feature_columns: Sequence[str],
) -> dict:
    header = pd.read_csv(unsw_path, nrows=0).columns if unsw_path.exists() else pd.Index([])
    notes = [
        "source split is reused from the H1 binary training split",
        "statistics use X_train only from that existing split function",
        "features are raw_or_minimally_cleaned, not transformed",
    ]
    selected_features_has_label_column = LABEL_COL in feature_columns
    selected_features_has_attack_column = ATTACK_COL in feature_columns
    selected_features_has_ip_columns = has_ip_identifier_columns(feature_columns)
    selected_features_has_timestamp_columns = has_timestamp_columns(feature_columns)
    usable = (
        unsw_path.exists()
        and len(feature_columns) == EXPECTED_NUM_FEATURES
        and not forbidden_columns_in_features(feature_columns)
        and not selected_features_has_label_column
        and not selected_features_has_attack_column
        and not selected_features_has_ip_columns
        and not selected_features_has_timestamp_columns
        and source_profile.audit.num_inf_after_cleaning == 0
        and source_profile.audit.num_l7_proto_non_integer_after_cleaning == 0
    )
    return {
        "dataset": SOURCE_DATASET,
        "role": "source",
        "seed": seed,
        "path_exists": bool(unsw_path.exists()),
        "num_rows_checked": len(source_profile.frame),
        "num_features": len(feature_columns),
        "feature_order_matches_config": True,
        "raw_header_has_label_column": LABEL_COL in header,
        "raw_header_has_attack_column": ATTACK_COL in header,
        "selected_features_has_label_column": selected_features_has_label_column,
        "selected_features_has_attack_column": selected_features_has_attack_column,
        "raw_header_has_ip_columns": has_ip_identifier_columns(header),
        "raw_header_has_timestamp_columns": has_timestamp_columns(header),
        "selected_features_has_ip_columns": selected_features_has_ip_columns,
        "selected_features_has_timestamp_columns": selected_features_has_timestamp_columns,
        "num_inf_before_cleaning": source_profile.audit.num_inf_before_cleaning,
        "num_inf_after_cleaning": source_profile.audit.num_inf_after_cleaning,
        "num_l7_proto_non_integer_before_cleaning": (
            source_profile.audit.num_l7_proto_non_integer_before_cleaning
        ),
        "num_l7_proto_non_integer_after_cleaning": (
            source_profile.audit.num_l7_proto_non_integer_after_cleaning
        ),
        "missing_rate_mean": source_profile.audit.missing_rate_mean,
        "missing_rate_max": source_profile.audit.missing_rate_max,
        "protocol_missing_rate": source_profile.audit.protocol_missing_rate,
        "l7_proto_missing_rate": source_profile.audit.l7_proto_missing_rate,
        "data_stage_judgment": "raw_or_minimally_cleaned",
        "feature_shift_usable": "yes" if usable else "no",
        "notes": "; ".join(notes),
    }


def precheck_target_row(
    dataset: str,
    path: Path,
    feature_columns: Sequence[str],
    precheck_rows: int,
) -> dict:
    path_exists = path.exists()
    header = pd.read_csv(path, nrows=0).columns if path_exists else pd.Index([])
    missing_features = [feature for feature in feature_columns if feature not in header]

    if path_exists and not missing_features:
        sample = pd.read_csv(path, usecols=feature_columns, nrows=precheck_rows, low_memory=False)
        cleaned, audit = minimal_clean_features(sample, feature_columns)
        num_rows_checked = len(cleaned)
    else:
        audit = CleaningAudit(0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan)
        num_rows_checked = 0

    notes = ["target labels are intentionally not read by Feature shift usecols"]
    if missing_features:
        notes.append(f"missing features: {missing_features}")
    selected_features_has_label_column = LABEL_COL in feature_columns
    selected_features_has_attack_column = ATTACK_COL in feature_columns
    selected_features_has_ip_columns = has_ip_identifier_columns(feature_columns)
    selected_features_has_timestamp_columns = has_timestamp_columns(feature_columns)
    usable = (
        path_exists
        and not missing_features
        and len(feature_columns) == EXPECTED_NUM_FEATURES
        and not forbidden_columns_in_features(feature_columns)
        and not selected_features_has_label_column
        and not selected_features_has_attack_column
        and not selected_features_has_ip_columns
        and not selected_features_has_timestamp_columns
        and audit.num_inf_after_cleaning == 0
        and audit.num_l7_proto_non_integer_after_cleaning == 0
    )
    return {
        "dataset": dataset,
        "role": "target",
        "seed": "all",
        "path_exists": bool(path_exists),
        "num_rows_checked": num_rows_checked,
        "num_features": len(feature_columns) - len(missing_features),
        "feature_order_matches_config": not missing_features,
        "raw_header_has_label_column": LABEL_COL in header,
        "raw_header_has_attack_column": ATTACK_COL in header,
        "selected_features_has_label_column": selected_features_has_label_column,
        "selected_features_has_attack_column": selected_features_has_attack_column,
        "raw_header_has_ip_columns": has_ip_identifier_columns(header),
        "raw_header_has_timestamp_columns": has_timestamp_columns(header),
        "selected_features_has_ip_columns": selected_features_has_ip_columns,
        "selected_features_has_timestamp_columns": selected_features_has_timestamp_columns,
        "num_inf_before_cleaning": audit.num_inf_before_cleaning,
        "num_inf_after_cleaning": audit.num_inf_after_cleaning,
        "num_l7_proto_non_integer_before_cleaning": audit.num_l7_proto_non_integer_before_cleaning,
        "num_l7_proto_non_integer_after_cleaning": audit.num_l7_proto_non_integer_after_cleaning,
        "missing_rate_mean": audit.missing_rate_mean,
        "missing_rate_max": audit.missing_rate_max,
        "protocol_missing_rate": audit.protocol_missing_rate,
        "l7_proto_missing_rate": audit.l7_proto_missing_rate,
        "data_stage_judgment": "raw_or_minimally_cleaned",
        "feature_shift_usable": "yes" if usable else "no",
        "notes": "; ".join(notes),
    }


def load_target_profile(
    target_dataset: str,
    path: Path,
    feature_columns: Sequence[str],
    args: argparse.Namespace,
) -> TargetProfile:
    ensure_feature_columns_exist(path, target_dataset, feature_columns)
    print_flush(f"[FeatureShift] Streaming target stats: target={target_dataset}, path={path}")

    stats = StreamingFeatureStats(feature_columns)
    sampler = ReservoirSampler(
        max_rows=args.quantile_sample_rows,
        feature_columns=feature_columns,
        seed=args.reservoir_seed,
    )
    cat_counts = {feature: Counter() for feature in categorical_jsd_features()}

    for chunk in pd.read_csv(
        path,
        usecols=feature_columns,
        chunksize=args.chunk_size,
        low_memory=False,
    ):
        cleaned, _ = minimal_clean_features(chunk, feature_columns)
        stats.update(cleaned)
        sampler.update(cleaned)
        update_categorical_counts(cat_counts, cleaned, categorical_jsd_features())

    sample = sampler.to_frame()
    quantiles = compute_quantile_stats(sample, feature_columns, "target")
    target_stats = stats.to_frame("target", quantiles)
    return TargetProfile(
        target_dataset=target_dataset,
        path=path,
        stats=target_stats,
        quantile_sample=sample,
        categorical_counts=cat_counts,
        total_rows=stats.total_rows,
        target_quantile_method="deterministic_reservoir_sample",
        target_quantile_sample_rows=len(sample),
        quantile_estimation=(
            f"deterministic_full_target_reservoir_sample_rows={len(sample)}"
        ),
    )


def make_bin_edges(
    source_values: pd.Series,
    target_sample_values: pd.Series,
    n_bins: int,
    strategy: str,
) -> np.ndarray:
    combined = pd.concat([source_values, target_sample_values], ignore_index=True)
    finite = pd.to_numeric(combined, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if finite.empty:
        return np.array([-np.inf, np.inf], dtype=float)

    values = finite.to_numpy(dtype=float, copy=False)
    if np.nanmin(values) == np.nanmax(values):
        center = float(values[0])
        delta = 0.5 if center == 0 else abs(center) * 0.01
        return np.array([-np.inf, center - delta, center + delta, np.inf], dtype=float)

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        finite_edges = np.unique(np.quantile(values, quantiles))
    elif strategy == "uniform":
        finite_edges = np.linspace(np.nanmin(values), np.nanmax(values), n_bins + 1)
    else:
        raise ValueError(f"Unknown bin strategy: {strategy}")

    finite_edges = finite_edges[np.isfinite(finite_edges)]
    finite_edges = np.unique(finite_edges)
    if len(finite_edges) < 2:
        center = float(finite_edges[0])
        finite_edges = np.array([center - 0.5, center + 0.5], dtype=float)
    interior = finite_edges[1:-1]
    return np.concatenate([np.array([-np.inf]), interior, np.array([np.inf])])


def histogram_counts(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    nan_count = int(numeric.isna().sum())
    finite = numeric.dropna().to_numpy(dtype=float, copy=False)
    hist, _ = np.histogram(finite, bins=edges)
    return np.concatenate([hist.astype(float), np.array([float(nan_count)])])


def source_numeric_histograms(
    source_frame: pd.DataFrame,
    numeric_features: Sequence[str],
    edges_by_feature: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    return {
        feature: histogram_counts(source_frame[feature], edges_by_feature[feature])
        for feature in numeric_features
    }


def target_numeric_histograms(
    path: Path,
    feature_columns: Sequence[str],
    numeric_features: Sequence[str],
    edges_by_feature: Mapping[str, np.ndarray],
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    counts = {
        feature: np.zeros(len(edges_by_feature[feature]), dtype=float)
        for feature in numeric_features
    }
    for chunk in pd.read_csv(
        path,
        usecols=feature_columns,
        chunksize=chunk_size,
        low_memory=False,
    ):
        cleaned, _ = minimal_clean_features(chunk, feature_columns)
        for feature in numeric_features:
            counts[feature] += histogram_counts(cleaned[feature], edges_by_feature[feature])
    return counts


def jensen_shannon_divergence(left_counts: np.ndarray, right_counts: np.ndarray) -> float:
    left_total = float(left_counts.sum())
    right_total = float(right_counts.sum())
    if left_total == 0 and right_total == 0:
        return 0.0
    if left_total == 0 or right_total == 0:
        return 1.0

    p = left_counts.astype(float) / left_total
    q = right_counts.astype(float) / right_total
    m = 0.5 * (p + q)

    def kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return float(np.clip(jsd, 0.0, 1.0))


def jsd_from_counters(left: Counter, right: Counter) -> float:
    support = sorted(set(left) | set(right))
    left_counts = np.array([left.get(key, 0.0) for key in support], dtype=float)
    right_counts = np.array([right.get(key, 0.0) for key in support], dtype=float)
    return jensen_shannon_divergence(left_counts, right_counts)


def feature_family(feature: str) -> str:
    name = feature.upper()
    if "ICMP" in name:
        return "ICMP"
    if "PORT" in name:
        return "Port"
    if "TCP_FLAGS" in name or name in {"CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"}:
        return "TCP flag"
    if name in {"PROTOCOL", "L7_PROTO"} or name.startswith("DNS_") or name.startswith("FTP_"):
        return "Protocol / L7"
    if "BYTES" in name or "THROUGHPUT" in name:
        return "Byte volume"
    if "PKT" in name or "PKTS" in name:
        return "Packet count"
    if "DURATION" in name or "_IAT_" in name:
        return "Duration / timing"
    if "SRC_TO_DST" in name or "DST_TO_SRC" in name:
        return "Directional ratio"
    return "Other"


def build_metric_rows(
    source_profile: SourceProfile,
    target_profile: TargetProfile,
    feature_columns: Sequence[str],
    numeric_features: Sequence[str],
    args: argparse.Namespace,
) -> List[dict]:
    edges_by_feature = {
        feature: make_bin_edges(
            source_profile.frame[feature],
            target_profile.quantile_sample[feature],
            args.n_bins,
            args.bin_strategy,
        )
        for feature in numeric_features
    }
    source_hist = source_numeric_histograms(source_profile.frame, numeric_features, edges_by_feature)
    target_hist = target_numeric_histograms(
        target_profile.path,
        feature_columns,
        numeric_features,
        edges_by_feature,
        args.chunk_size,
    )
    numeric_jsd = {
        feature: jensen_shannon_divergence(source_hist[feature], target_hist[feature])
        for feature in numeric_features
    }
    categorical_jsd = {
        feature: jsd_from_counters(
            source_profile.categorical_counts[feature],
            target_profile.categorical_counts[feature],
        )
        for feature in categorical_jsd_features()
    }

    source_stats = source_profile.stats.set_index("feature")
    target_stats = target_profile.stats.set_index("feature")
    rows = []
    for feature in feature_columns:
        source = source_stats.loc[feature]
        target = target_stats.loc[feature]
        robust_shift_score = abs(source["source_median"] - target["target_median"]) / (
            source["source_iqr"] + target["target_iqr"] + EPS
        )
        pooled_var = (source["source_var"] + target["target_var"]) / 2.0
        standard_smd = (target["target_mean"] - source["source_mean"]) / math.sqrt(
            pooled_var + EPS
        )
        jsd = numeric_jsd.get(feature, categorical_jsd.get(feature, np.nan))
        rows.append(
            {
                "seed": source_profile.seed,
                "source_dataset": SOURCE_DATASET,
                "source_split": SOURCE_SPLIT,
                "source_distribution_mode": SOURCE_DISTRIBUTION_MODE,
                "target_dataset": target_profile.target_dataset,
                "target_distribution_mode": TARGET_DISTRIBUTION_MODE,
                "target_label_usage": TARGET_LABEL_USAGE,
                "feature": feature,
                "feature_family": feature_family(feature),
                "source_n": int(source["source_n"]),
                "target_n": int(target["target_n"]),
                "source_missing_rate": source["source_missing_rate"],
                "target_missing_rate": target["target_missing_rate"],
                "missing_rate_diff": target["target_missing_rate"] - source["source_missing_rate"],
                "source_mean": source["source_mean"],
                "target_mean": target["target_mean"],
                "source_std": source["source_std"],
                "target_std": target["target_std"],
                "source_median": source["source_median"],
                "target_median": target["target_median"],
                "source_iqr": source["source_iqr"],
                "target_iqr": target["target_iqr"],
                "robust_shift_score": robust_shift_score,
                "standard_smd": standard_smd,
                "abs_standard_smd": abs(standard_smd),
                "jsd": jsd,
                "target_quantile_method": target_profile.target_quantile_method,
                "target_quantile_sample_rows": target_profile.target_quantile_sample_rows,
                "target_quantile_estimation": target_profile.quantile_estimation,
            }
        )
    return rows


def write_mean_std_summary(long_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    metrics = ["robust_shift_score", "standard_smd", "abs_standard_smd", "jsd"]
    summary = (
        long_df.groupby(["target_dataset", "feature", "feature_family"], dropna=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    rank_specs = {
        "rank_by_jsd": "jsd_mean",
        "rank_by_abs_smd": "abs_standard_smd_mean",
        "rank_by_robust_shift": "robust_shift_score_mean",
    }
    for rank_col, metric_col in rank_specs.items():
        summary[rank_col] = (
            summary.groupby("target_dataset")[metric_col]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

    path = output_dir / "feature_shift_feature_shift_mean_std_summary.csv"
    summary.to_csv(path, index=False)
    return summary


def write_top_shift_features(summary: pd.DataFrame, output_dir: Path, top_k: int) -> pd.DataFrame:
    metric_map = {
        "jsd_mean": "jsd_mean",
        "abs_standard_smd_mean": "abs_standard_smd_mean",
        "robust_shift_score_mean": "robust_shift_score_mean",
    }
    rows = []
    for target, group in summary.groupby("target_dataset"):
        for rank_metric, metric_col in metric_map.items():
            ranked = group.sort_values(metric_col, ascending=False).head(top_k)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "target_dataset": target,
                        "rank_metric": rank_metric,
                        "rank": rank,
                        "feature": row["feature"],
                        "feature_family": row["feature_family"],
                        "jsd_mean": row["jsd_mean"],
                        "abs_standard_smd_mean": row["abs_standard_smd_mean"],
                        "robust_shift_score_mean": row["robust_shift_score_mean"],
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "feature_shift_top_shift_features_by_target.csv", index=False)
    return out


def write_feature_family_summary(summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for (target, family), group in summary.groupby(["target_dataset", "feature_family"]):
        top_features = (
            group.sort_values("jsd_mean", ascending=False)["feature"].head(5).tolist()
        )
        rows.append(
            {
                "target_dataset": target,
                "feature_family": family,
                "num_features": int(group["feature"].nunique()),
                "mean_jsd": group["jsd_mean"].mean(),
                "mean_abs_standard_smd": group["abs_standard_smd_mean"].mean(),
                "mean_robust_shift_score": group["robust_shift_score_mean"].mean(),
                "top_features_by_jsd": json.dumps(top_features),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "feature_shift_feature_family_shift_summary.csv", index=False)
    return out


def strip_transformer_prefix(name: str) -> str:
    return name.split("__", 1)[1] if "__" in name else name


def candidate_rf_artifact_paths(seed: int) -> List[Path]:
    paths = [
        REPOSITORY_ROOT
        / "results"
        / "generated"
        / "h1"
        / "artifacts"
        / "e2"
        / f"seed{seed}"
        / "RandomForest.joblib",
    ]
    return paths


def write_rf_importance_availability(seeds: Sequence[int], output_dir: Path) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        candidates = candidate_rf_artifact_paths(seed)
        selected = next((path for path in candidates if path.exists()), None)
        rows.append(
            {
                "seed": seed,
                "rf_artifact_found": selected is not None,
                "selected_artifact": str(selected) if selected else "",
                "candidate_artifacts": json.dumps([str(path) for path in candidates]),
            }
        )
    availability = pd.DataFrame(rows)
    availability.to_csv(output_dir / "feature_shift_rf_importance_availability.csv", index=False)
    return availability


def extract_rf_importance_for_seed(seed: int, feature_columns: Sequence[str]) -> pd.DataFrame:
    try:
        import joblib
    except ImportError:
        print_flush("[FeatureShift] joblib is unavailable; skipping RF importance export.")
        return pd.DataFrame()

    artifact_path = next((path for path in candidate_rf_artifact_paths(seed) if path.exists()), None)
    if artifact_path is None:
        print_flush(f"[FeatureShift] No RF artifact found for seed={seed}; skipping this seed.")
        return pd.DataFrame()

    pipeline = joblib.load(artifact_path)
    model = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else pipeline
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print_flush(f"[FeatureShift] RF artifact lacks feature_importances_: {artifact_path}")
        return pd.DataFrame()

    feature_names = list(feature_columns)
    if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocessor"]
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = [strip_transformer_prefix(name) for name in preprocessor.get_feature_names_out()]

    if len(feature_names) != len(importances):
        if len(feature_columns) == len(importances):
            feature_names = list(feature_columns)
        else:
            raise ValueError(
                f"Cannot align RF importances for seed={seed}: "
                f"{len(importances)} importances vs {len(feature_names)} names."
            )

    out = pd.DataFrame(
        {
            "seed": seed,
            "feature": feature_names,
            "rf_importance": importances,
        }
    )
    out = out[out["feature"].isin(feature_columns)].copy()
    out["importance_rank"] = out["rf_importance"].rank(method="dense", ascending=False).astype(int)
    return out.sort_values(["seed", "importance_rank", "feature"])


def load_or_export_rf_importance(
    seeds: Sequence[int],
    feature_columns: Sequence[str],
    output_dir: Path,
    explicit_path: str,
) -> pd.DataFrame:
    if explicit_path:
        path = Path(explicit_path)
    else:
        path = output_dir / "rf_feature_importance_by_seed.csv"

    if path.exists():
        return pd.read_csv(path)

    frames = [extract_rf_importance_for_seed(seed, feature_columns) for seed in seeds]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["seed", "feature", "rf_importance", "importance_rank"])

    importance = pd.concat(frames, ignore_index=True)
    importance.to_csv(output_dir / "rf_feature_importance_by_seed.csv", index=False)
    return importance


def spearman_corr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan, np.nan
    try:
        from scipy.stats import spearmanr

        corr, p_value = spearmanr(x[valid], y[valid])
        return float(corr), float(p_value)
    except Exception:
        corr = x[valid].rank().corr(y[valid].rank(), method="pearson")
        return float(corr), np.nan


def write_importance_shift_outputs(
    long_df: pd.DataFrame,
    importance: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if importance.empty:
        empty_join = pd.DataFrame(columns=IMPORTANCE_SHIFT_JOIN_COLUMNS)
        empty_corr = pd.DataFrame(columns=IMPORTANCE_SHIFT_CORRELATION_COLUMNS)
        empty_join.to_csv(output_dir / "feature_shift_importance_shift_join.csv", index=False)
        empty_corr.to_csv(output_dir / "feature_shift_importance_shift_correlation.csv", index=False)
        return empty_join, empty_corr

    joined = long_df.merge(importance, on=["seed", "feature"], how="inner")
    joined["importance_x_jsd"] = joined["rf_importance"] * joined["jsd"]
    joined["importance_x_abs_smd"] = joined["rf_importance"] * joined["abs_standard_smd"]

    joined[IMPORTANCE_SHIFT_JOIN_COLUMNS].to_csv(
        output_dir / "feature_shift_importance_shift_join.csv",
        index=False,
    )

    rows = []
    for target, group in joined.groupby("target_dataset"):
        for metric in ["jsd", "abs_standard_smd", "robust_shift_score"]:
            corr, p_value = spearman_corr(group["rf_importance"], group[metric])
            rows.append(
                {
                    "target_dataset": target,
                    "metric": metric,
                    "spearman_corr_importance_vs_shift": corr,
                    "p_value": p_value,
                }
            )
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(output_dir / "feature_shift_importance_shift_correlation.csv", index=False)
    return joined[IMPORTANCE_SHIFT_JOIN_COLUMNS], corr_df


def parse_removed_features(value: object) -> List[str]:
    if isinstance(value, list):
        return value
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    try:
        import ast

        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def write_removed_feature_overlap(
    long_df: pd.DataFrame,
    removed_features_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    if not removed_features_path.exists():
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "feature_shift_removed_feature_shift_overlap.csv", index=False)
        return empty

    removed_df = pd.read_csv(removed_features_path)
    rows = []
    for _, removed_row in removed_df.iterrows():
        seed = int(removed_row["seed"])
        target = removed_row["target_dataset"]
        removed = parse_removed_features(removed_row["removed_features"])
        metrics = long_df[
            long_df["seed"].eq(seed) & long_df["target_dataset"].eq(target)
        ].copy()
        removed_metrics = metrics[metrics["feature"].isin(removed)]
        top_by_jsd = metrics.sort_values("jsd", ascending=False)

        def percent_in_top(k: int) -> float:
            if not removed:
                return 0.0
            top_features = set(top_by_jsd.head(k)["feature"])
            return 100.0 * len(set(removed) & top_features) / len(set(removed))

        rows.append(
            {
                "seed": seed,
                "target_dataset": target,
                "filtering_strategy": removed_row["filtering_strategy"],
                "num_removed": len(removed),
                "removed_features": json.dumps(removed),
                "mean_removed_jsd": removed_metrics["jsd"].mean(),
                "mean_all_features_jsd": metrics["jsd"].mean(),
                "mean_removed_abs_standard_smd": removed_metrics["abs_standard_smd"].mean(),
                "mean_all_features_abs_standard_smd": metrics["abs_standard_smd"].mean(),
                "mean_removed_robust_shift_score": removed_metrics["robust_shift_score"].mean(),
                "mean_all_features_robust_shift_score": metrics["robust_shift_score"].mean(),
                "percent_removed_in_top5_jsd": percent_in_top(5),
                "percent_removed_in_top10_jsd": percent_in_top(10),
                "percent_removed_in_top15_jsd": percent_in_top(15),
                "target_label_usage": TARGET_LABEL_USAGE,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "feature_shift_removed_feature_shift_overlap.csv", index=False)
    return out


def load_sampled_shift_file(sampled_dir: Path, seed: int, target: str) -> pd.DataFrame:
    candidates = [
        sampled_dir / f"seed{seed}" / f"feature_robust_shift_scores_{target}.csv",
        sampled_dir / f"seed{seed}" / f"feature_sampled_robust_shift_scores_{target}.csv",
        Path("results/generated/h1/multiseed/feature_filtering")
        / f"seed{seed}"
        / f"feature_shift_scores_{target}.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    rename_map = {}
    if "robust_shift_score" in df.columns:
        rename_map["robust_shift_score"] = "sampled_robust_shift_score"
    if "shift_score" in df.columns:
        rename_map["shift_score"] = "sampled_robust_shift_score"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df[["feature", "sampled_robust_shift_score"]]


def write_sample_vs_full_stability(
    long_df: pd.DataFrame,
    sampled_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for (seed, target), full in long_df.groupby(["seed", "target_dataset"]):
        sampled = load_sampled_shift_file(sampled_dir, int(seed), target)
        if sampled.empty:
            rows.append(
                {
                    "seed": seed,
                    "target_dataset": target,
                    "spearman_corr_sampled_vs_full": np.nan,
                    "top5_overlap": np.nan,
                    "top10_overlap": np.nan,
                    "top15_overlap": np.nan,
                }
            )
            continue
        joined = sampled.merge(
            full[["feature", "robust_shift_score"]].rename(
                columns={"robust_shift_score": "full_robust_shift_score"}
            ),
            on="feature",
            how="inner",
        )
        corr, _ = spearman_corr(
            joined["sampled_robust_shift_score"],
            joined["full_robust_shift_score"],
        )

        sampled_ranked = joined.sort_values("sampled_robust_shift_score", ascending=False)
        full_ranked = joined.sort_values("full_robust_shift_score", ascending=False)

        def overlap(k: int) -> float:
            left = set(sampled_ranked.head(k)["feature"])
            right = set(full_ranked.head(k)["feature"])
            return len(left & right) / k

        rows.append(
            {
                "seed": seed,
                "target_dataset": target,
                "spearman_corr_sampled_vs_full": corr,
                "top5_overlap": overlap(5),
                "top10_overlap": overlap(10),
                "top15_overlap": overlap(15),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "feature_shift_sample_vs_full_stability.csv", index=False)
    return out


def write_metadata(args: argparse.Namespace, feature_columns: Sequence[str], output_dir: Path) -> None:
    metadata = {
        "experiment": "Feature shift full-target feature-space shift quantification",
        "source_dataset": SOURCE_DATASET,
        "source_split": SOURCE_SPLIT,
        "source_distribution_mode": SOURCE_DISTRIBUTION_MODE,
        "target_datasets": [
            TARGET_FULL_NAMES[target] for target in args.targets
        ],
        "target_distribution_mode": TARGET_DISTRIBUTION_MODE,
        "target_label_usage": TARGET_LABEL_USAGE,
        "num_features": len(feature_columns),
        "feature_order": "NUMERIC_FEATURES + PROTOCOL_FEATURES + CATEGORICAL_FEATURES",
        "seeds": args.seeds,
        "metrics": [
            "robust_shift_score",
            "standard_smd",
            "abs_standard_smd",
            "jsd",
        ],
        "robust_shift_score_formula": (
            "abs(source_median - target_median) / (source_iqr + target_iqr + eps)"
        ),
        "robust_shift_score_quantile_method": (
            "source median/IQR are exact on the H1-compatible full UNSW train split; "
            "target median/IQR are deterministic reservoir-sample approximations"
        ),
        "robust_shift_score_quantile_note": (
            "robust_shift_score is retained as the formal Feature shift robust shift metric, "
            "but target-side quantiles are approximate unless a streaming quantile sketch "
            "or exact full-target quantile mode is added later."
        ),
        "standard_smd_formula": (
            "(target_mean - source_mean) / sqrt((source_var + target_var) / 2 + eps)"
        ),
        "jsd_range": "[0, 1]",
        "jsd_base": 2,
        "chunk_size": args.chunk_size,
        "n_bins": args.n_bins,
        "bin_strategy": args.bin_strategy,
        "target_quantile_estimation": (
            "deterministic full-target reservoir sample for target median/IQR; "
            "full target streaming for target mean/std/JSD"
        ),
        "quantile_sample_rows": args.quantile_sample_rows,
        "old_shift_score_renamed_to": "sampled_robust_shift_score",
        "notes": (
            "Formal Feature shift uses full UNSW train split and full target feature distributions. "
            "Previous sampled shift scores are used only for sampled-vs-full stability comparison."
        ),
    }
    with open(output_dir / "feature_shift_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def write_sanity_report(
    long_df: pd.DataFrame,
    feature_columns: Sequence[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    expected_rows = len(args.seeds) * len(args.targets) * len(feature_columns)
    checks = {
        "expected_rows": expected_rows,
        "actual_rows": int(len(long_df)),
        "each_seed_target_has_49_rows": bool(
            long_df.groupby(["seed", "target_dataset"]).size().eq(EXPECTED_NUM_FEATURES).all()
        ),
        "seeds_present": sorted(long_df["seed"].dropna().astype(int).unique().tolist()),
        "targets_present": sorted(long_df["target_dataset"].dropna().unique().tolist()),
        "no_forbidden_feature_columns": not forbidden_columns_in_features(feature_columns),
        "target_label_usage_always_not_used": bool(long_df["target_label_usage"].eq(TARGET_LABEL_USAGE).all()),
        "source_distribution_mode_ok": bool(
            long_df["source_distribution_mode"].eq(SOURCE_DISTRIBUTION_MODE).all()
        ),
        "target_distribution_mode_ok": bool(
            long_df["target_distribution_mode"].eq(TARGET_DISTRIBUTION_MODE).all()
        ),
        "jsd_within_0_1": bool(long_df["jsd"].between(0, 1).all()),
        "standard_smd_no_inf": bool(np.isfinite(long_df["standard_smd"].dropna()).all()),
        "robust_shift_score_no_inf": bool(np.isfinite(long_df["robust_shift_score"].dropna()).all()),
        "missing_rates_within_0_1": bool(
            long_df["source_missing_rate"].between(0, 1).all()
            and long_df["target_missing_rate"].between(0, 1).all()
        ),
        "old_shift_score_not_formal_column": "shift_score" not in long_df.columns,
        "formal_smd_column_names_ok": (
            "standard_smd" in long_df.columns and "abs_standard_smd" in long_df.columns
        ),
    }
    checks["feature_shift_ready_for_interpretation"] = all(
        value for key, value in checks.items() if isinstance(value, bool)
    ) and checks["actual_rows"] == checks["expected_rows"]

    with open(output_dir / "feature_shift_sanity_check_report.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Feature shift full-target feature-space shift quantification."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--targets", nargs="+", choices=DEFAULT_TARGETS, default=DEFAULT_TARGETS)
    parser.add_argument("--target-mode", choices=["full"], default="full")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--bin-strategy", choices=["quantile", "uniform"], default="quantile")
    parser.add_argument("--output-dir", default="results/generated/h2/feature_shift")
    parser.add_argument("--unsw-path", default=UNSW_PATH)
    parser.add_argument("--cicids-path", default=CICIDS_PATH)
    parser.add_argument("--bot-iot-path", default=BOTIOT_PATH)
    parser.add_argument("--ton-iot-path", default=TONIOT_PATH)
    parser.add_argument("--precheck-rows", type=int, default=100_000)
    parser.add_argument("--quantile-sample-rows", type=int, default=500_000)
    parser.add_argument("--reservoir-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument(
        "--removed-features-path",
        default="results/generated/h1/multiseed/removed_features_by_seed.csv",
    )
    parser.add_argument(
        "--sampled-shift-dir",
        default="results/generated/h1/multiseed/feature_filtering",
    )
    parser.add_argument("--rf-importance-path", default="")
    parser.add_argument(
        "--only-precheck",
        action="store_true",
        help="Write feature_shift_precheck_audit.csv only, without computing shift metrics.",
    )
    return parser.parse_args()


def run_feature_shift(args: argparse.Namespace) -> None:
    if args.target_mode != "full":
        raise ValueError("Formal Feature shift currently supports only --target-mode full.")

    from cross_domain_nids.release_paths import require_generated_output

    output_dir = require_generated_output(args.output_dir, "results")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = feature_shift_feature_columns()
    target_path_map = target_paths(args)
    selected_target_paths = {target: target_path_map[target] for target in args.targets}

    for target, path in selected_target_paths.items():
        ensure_feature_columns_exist(path, target, feature_columns)

    source_profiles = {}
    audit_rows = []
    for seed in args.seeds:
        profile = load_source_train_profile(seed, Path(args.unsw_path), feature_columns)
        source_profiles[seed] = profile
        audit_rows.append(precheck_source_row(seed, Path(args.unsw_path), profile, feature_columns))

    for target, path in selected_target_paths.items():
        audit_rows.append(precheck_target_row(target, path, feature_columns, args.precheck_rows))

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "feature_shift_precheck_audit.csv", index=False)
    if audit["feature_shift_usable"].ne("yes").any():
        raise ValueError(
            "Feature shift precheck failed. Inspect feature_shift_precheck_audit.csv before running metrics."
        )
    if args.only_precheck:
        write_metadata(args, feature_columns, output_dir)
        print_flush(f"[FeatureShift] Precheck only complete: {output_dir / 'feature_shift_precheck_audit.csv'}")
        return

    target_profiles = {
        target: load_target_profile(target, path, feature_columns, args)
        for target, path in selected_target_paths.items()
    }

    long_rows = []
    numeric_features = numeric_jsd_features(feature_columns)
    for seed, source_profile in source_profiles.items():
        for target, target_profile in target_profiles.items():
            print_flush(f"[FeatureShift] Computing metrics: seed={seed}, target={target}")
            long_rows.extend(
                build_metric_rows(
                    source_profile,
                    target_profile,
                    feature_columns,
                    numeric_features,
                    args,
                )
            )

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(output_dir / "feature_shift_feature_shift_scores_long.csv", index=False)

    summary = write_mean_std_summary(long_df, output_dir)
    write_top_shift_features(summary, output_dir, args.top_k)
    family_summary = write_feature_family_summary(summary, output_dir)

    importance = load_or_export_rf_importance(
        args.seeds,
        feature_columns,
        output_dir,
        args.rf_importance_path,
    )
    write_rf_importance_availability(args.seeds, output_dir)
    importance_shift, _ = write_importance_shift_outputs(long_df, importance, output_dir)
    removed_overlap = write_removed_feature_overlap(
        long_df,
        Path(args.removed_features_path),
        output_dir,
    )
    write_sample_vs_full_stability(long_df, Path(args.sampled_shift_dir), output_dir)
    write_metadata(args, feature_columns, output_dir)
    write_sanity_report(long_df, feature_columns, args, output_dir)


    print_flush(f"[FeatureShift] Complete. Outputs saved under: {output_dir}")


if __name__ == "__main__":
    run_feature_shift(parse_args())
