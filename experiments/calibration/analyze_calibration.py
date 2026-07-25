"""Step 5 E1-E4 calibration and ECE analysis.

This script is inference-only. It loads saved Pipeline artifacts, uses their
fitted preprocessors and classifiers as-is, and never calls fit on either the
model or preprocessing steps.

Default mode is a one-combination sanity check:
  E1 RandomForest CICIDS2018 seed42 none

Use --mode full only after the sanity output has been inspected.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import joblib
import numpy as np
import pandas as pd

from cross_domain_nids.config import (
    ATTACK_COL,
    BOTIOT_PATH,
    CICIDS_PATH,
    LABEL_COL,
    STRESS_B_CHUNK_SIZE,
    TONIOT_PATH,
    UNSW_PATH,
)
from experiments.h1.run_multiseed_training import (
    load_unsw_splits as load_h1_unsw_splits,
)
from cross_domain_nids.preprocessing.netflow import prepare_cicids_chunk


GROUP_COLS = ["experiment", "model", "target_dataset", "filtering_strategy"]
COMBO_COLS = ["experiment", "model", "target_dataset", "filtering_strategy", "seed"]
TARGET_DATASETS = {
    "CICIDS2018": CICIDS_PATH,
    "BoT-IoT": BOTIOT_PATH,
    "ToN-IoT": TONIOT_PATH,
}
EXPERIMENT_TASK = {
    "E1": "multiclass",
    "E2": "binary",
    "E3": "binary",
    "E4": "binary",
}
REFERENCE_METRICS = [
    "n_samples",
    "tp",
    "tn",
    "fp",
    "fn",
    "binary_f1",
    "mcc",
    "fpr",
    "fnr",
]


def print_flush(message: str) -> None:
    print(message, flush=True)


def value_is_present(value) -> bool:
    return value is not None and not pd.isna(value) and str(value) != ""


def as_repo_path(path_value, repo_root: Path) -> Path | None:
    if not value_is_present(path_value):
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = repo_root / path
    return path


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def combo_slug(combo: "Combo") -> str:
    parts = [
        combo.experiment,
        combo.model,
        combo.target_dataset,
        combo.filtering_strategy,
        f"seed{combo.seed}",
    ]
    return "__".join(slugify(part) for part in parts)


def safe_float(value, default=np.nan) -> float:
    if not value_is_present(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Combo:
    experiment: str
    model: str
    target_dataset: str
    filtering_strategy: str
    seed: int
    artifact_hint: str | None
    filtering_target_dataset: str | None
    reference_metrics: dict

    @property
    def task(self) -> str:
        return EXPERIMENT_TASK[self.experiment]

    def meta(self) -> dict:
        return {
            "experiment": self.experiment,
            "model": self.model,
            "target_dataset": self.target_dataset,
            "filtering_strategy": self.filtering_strategy,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ProbabilitySpec:
    task: str
    class_labels: list[str]
    class_source: str
    probability_mapping: str
    attack_class_index: int | None = None
    benign_class_index: int | None = None


class CalibrationAccumulator:
    def __init__(self, n_bins: int):
        self.n_bins = n_bins
        self.bin_n = np.zeros(n_bins, dtype=np.int64)
        self.bin_p_sum = np.zeros(n_bins, dtype=np.float64)
        self.bin_y_sum = np.zeros(n_bins, dtype=np.float64)
        self.n = 0
        self.p_sum = 0.0
        self.y_sum = 0.0
        self.pred_sum = 0.0
        self.p05_pred_sum = 0.0
        self.brier_sum = 0.0
        self.log_loss_sum = 0.0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp_p05 = 0
        self.tn_p05 = 0
        self.fp_p05 = 0
        self.fn_p05 = 0

    def update(self, y_true_binary, y_pred_binary, p_attack) -> None:
        y_true = np.asarray(y_true_binary, dtype=np.int8)
        y_pred = np.asarray(y_pred_binary, dtype=np.int8)
        p = np.asarray(p_attack, dtype=np.float64)
        p = np.clip(p, 0.0, 1.0)
        y_pred_p05 = (p >= 0.5).astype(np.int8)

        if len(y_true) != len(y_pred) or len(y_true) != len(p):
            raise ValueError(
                "Mismatched lengths: "
                f"y_true={len(y_true)}, y_pred={len(y_pred)}, p_attack={len(p)}"
            )

        bins = np.floor(p * self.n_bins).astype(int)
        bins = np.clip(bins, 0, self.n_bins - 1)
        self.bin_n += np.bincount(bins, minlength=self.n_bins)
        self.bin_p_sum += np.bincount(bins, weights=p, minlength=self.n_bins)
        self.bin_y_sum += np.bincount(bins, weights=y_true, minlength=self.n_bins)

        eps = 1e-15
        p_safe = np.clip(p, eps, 1.0 - eps)
        self.n += int(len(y_true))
        self.p_sum += float(p.sum())
        self.y_sum += float(y_true.sum())
        self.pred_sum += float(y_pred.sum())
        self.p05_pred_sum += float(y_pred_p05.sum())
        self.brier_sum += float(np.square(p - y_true).sum())
        self.log_loss_sum += float(
            -(y_true * np.log(p_safe) + (1 - y_true) * np.log(1.0 - p_safe)).sum()
        )
        self.tp += int(((y_true == 1) & (y_pred == 1)).sum())
        self.tn += int(((y_true == 0) & (y_pred == 0)).sum())
        self.fp += int(((y_true == 0) & (y_pred == 1)).sum())
        self.fn += int(((y_true == 1) & (y_pred == 0)).sum())
        self.tp_p05 += int(((y_true == 1) & (y_pred_p05 == 1)).sum())
        self.tn_p05 += int(((y_true == 0) & (y_pred_p05 == 0)).sum())
        self.fp_p05 += int(((y_true == 0) & (y_pred_p05 == 1)).sum())
        self.fn_p05 += int(((y_true == 1) & (y_pred_p05 == 0)).sum())

    @staticmethod
    def binary_metrics_from_counts(
        tp: int,
        tn: int,
        fp: int,
        fn: int,
        n: int,
        suffix: str = "",
    ) -> dict:
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
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt(denominator) if denominator else 0.0

        benign_precision = tn / (tn + fn) if (tn + fn) else 0.0
        benign_recall = tn / (tn + fp) if (tn + fp) else 0.0
        benign_f1 = (
            2 * benign_precision * benign_recall / (benign_precision + benign_recall)
            if (benign_precision + benign_recall)
            else 0.0
        )
        weighted_f1 = (((tp + fn) * binary_f1) + ((tn + fp) * benign_f1)) / n if n else 0.0
        return {
            f"tp{suffix}": tp,
            f"tn{suffix}": tn,
            f"fp{suffix}": fp,
            f"fn{suffix}": fn,
            f"binary_f1{suffix}": binary_f1,
            f"weighted_f1{suffix}": weighted_f1,
            f"mcc{suffix}": mcc,
            f"fpr{suffix}": fpr,
            f"fnr{suffix}": fnr,
        }

    def binary_metrics(self) -> dict:
        return self.binary_metrics_from_counts(self.tp, self.tn, self.fp, self.fn, self.n)

    def binary_metrics_p05(self) -> dict:
        return self.binary_metrics_from_counts(
            self.tp_p05,
            self.tn_p05,
            self.fp_p05,
            self.fn_p05,
            self.n,
            suffix="_p05",
        )

    def bin_rows(self, meta: dict) -> list[dict]:
        rows = []
        total = max(self.n, 1)
        for bin_id in range(self.n_bins):
            n_bin = int(self.bin_n[bin_id])
            if n_bin:
                mean_p = float(self.bin_p_sum[bin_id] / n_bin)
                empirical = float(self.bin_y_sum[bin_id] / n_bin)
                abs_gap = abs(mean_p - empirical)
                weight = n_bin / total
                contribution = weight * abs_gap
            else:
                mean_p = np.nan
                empirical = np.nan
                abs_gap = np.nan
                weight = 0.0
                contribution = 0.0

            rows.append({
                **meta,
                "bin_id": bin_id,
                "bin_lower": bin_id / self.n_bins,
                "bin_upper": (bin_id + 1) / self.n_bins,
                "n_samples": n_bin,
                "bin_weight": weight,
                "mean_p_attack": mean_p,
                "empirical_attack_rate": empirical,
                "abs_gap": abs_gap,
                "ece_contribution": contribution,
            })
        return rows

    def summary_row(self, meta: dict) -> dict:
        total = max(self.n, 1)
        bins = self.bin_rows(meta)
        metrics = self.binary_metrics()
        p05_metrics = self.binary_metrics_p05()
        model_pred_attack_rate = float(self.pred_sum / total)
        return {
            **meta,
            "n_samples": self.n,
            "ece": float(sum(row["ece_contribution"] for row in bins)),
            "brier_score": float(self.brier_sum / total),
            "log_loss": float(self.log_loss_sum / total),
            "nll": float(self.log_loss_sum / total),
            "mean_p_attack": float(self.p_sum / total),
            "true_attack_rate": float(self.y_sum / total),
            "pred_attack_rate": model_pred_attack_rate,
            "model_pred_attack_rate": model_pred_attack_rate,
            "p05_pred_attack_rate": float(self.p05_pred_sum / total),
            **metrics,
            **p05_metrics,
        }


def load_combos(args, repo_root: Path) -> list[Combo]:
    path = as_repo_path(args.input_long_csv, repo_root)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing E1-E4 long result CSV: {args.input_long_csv}")

    df = pd.read_csv(path)
    missing = [col for col in COMBO_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    df["experiment"] = df["experiment"].astype(str)
    df["model"] = df["model"].astype(str)
    df["target_dataset"] = df["target_dataset"].astype(str)
    df["filtering_strategy"] = df["filtering_strategy"].fillna("none").astype(str)
    df["seed"] = df["seed"].astype(int)
    df = df.drop_duplicates(subset=COMBO_COLS)

    if args.mode == "sanity":
        df = df[
            df["experiment"].eq("E1")
            & df["model"].eq("RandomForest")
            & df["target_dataset"].eq("CICIDS2018")
            & df["filtering_strategy"].eq("none")
            & df["seed"].eq(42)
        ]
    else:
        df = apply_optional_filters(df, args)

    if args.skip_unsw:
        df = df[~df["target_dataset"].eq("UNSW")]

    combos = []
    for _, row in df.sort_values(COMBO_COLS).iterrows():
        reference = {
            metric: safe_float(row.get(metric))
            for metric in REFERENCE_METRICS
            if metric in row.index
        }
        combos.append(Combo(
            experiment=str(row["experiment"]),
            model=str(row["model"]),
            target_dataset=str(row["target_dataset"]),
            filtering_strategy=str(row["filtering_strategy"]),
            seed=int(row["seed"]),
            artifact_hint=str(row["artifact"]) if value_is_present(row.get("artifact")) else None,
            filtering_target_dataset=(
                str(row["filtering_target_dataset"])
                if value_is_present(row.get("filtering_target_dataset"))
                else None
            ),
            reference_metrics=reference,
        ))
    return combos


def apply_optional_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    filtered = df
    for col, values in [
        ("experiment", args.experiments),
        ("model", args.models),
        ("target_dataset", args.target_datasets),
        ("filtering_strategy", args.filtering_strategies),
    ]:
        if values:
            filtered = filtered[filtered[col].isin(values)]
    if args.seeds:
        filtered = filtered[filtered["seed"].isin(args.seeds)]
    return filtered


def infer_artifact_path(combo: Combo, repo_root: Path, args) -> Path:
    hint = as_repo_path(combo.artifact_hint, repo_root)
    if hint is not None and hint.exists():
        return hint

    output_root = as_repo_path(args.multiseed_result_dir, repo_root)
    if output_root is None:
        output_root = repo_root / "results" / "generated" / "h1"

    if combo.experiment == "E1":
        return output_root / "artifacts" / "e1" / f"seed{combo.seed}" / f"{combo.model}.joblib"

    if combo.experiment == "E2":
        return output_root / "artifacts" / "e2" / f"seed{combo.seed}" / f"{combo.model}.joblib"

    if combo.experiment == "E3":
        return (
            output_root
            / "artifacts"
            / "e3"
            / f"seed{combo.seed}"
            / combo.target_dataset
            / combo.filtering_strategy
            / f"{combo.model}.joblib"
        )

    if combo.experiment == "E4":
        artifact_scope = combo.filtering_target_dataset
        if not artifact_scope:
            artifact_scope = (
                "universal"
                if combo.filtering_strategy in {"remove_exact_duplicates", "remove_corr_gt_0.98"}
                else combo.target_dataset
            )
        return (
            output_root
            / "artifacts"
            / "e4"
            / f"seed{combo.seed}"
            / artifact_scope
            / combo.filtering_strategy
            / f"{combo.model}.joblib"
        )

    raise ValueError(f"Unknown experiment: {combo.experiment}")


def dataset_path_for(combo: Combo, repo_root: Path) -> Path:
    if combo.target_dataset == "UNSW":
        return repo_root / UNSW_PATH
    if combo.target_dataset not in TARGET_DATASETS:
        raise ValueError(f"Unknown target dataset: {combo.target_dataset}")
    return repo_root / TARGET_DATASETS[combo.target_dataset]


def get_inner_model(pipeline):
    if hasattr(pipeline, "named_steps"):
        return pipeline.named_steps.get("model")
    return None


def class_labels_from_pipeline(pipeline) -> list[str]:
    inner = get_inner_model(pipeline)
    classes = getattr(inner, "classes_", None)
    if classes is None:
        classes = getattr(pipeline, "classes_", None)
    if classes is None:
        return []
    return [str(value) for value in list(classes)]


def is_torch_mlp(pipeline) -> bool:
    inner = get_inner_model(pipeline)
    return inner is not None and inner.__class__.__name__ == "TorchMLPClassifier"


def build_probability_spec(pipeline, task: str) -> ProbabilitySpec:
    labels = class_labels_from_pipeline(pipeline)
    lower_labels = [label.strip().lower() for label in labels]
    class_source = "pipeline.named_steps['model'].classes_"

    if task == "multiclass":
        benign_candidates = {"benign", "normal"}
        benign_indices = [
            idx for idx, label in enumerate(lower_labels) if label in benign_candidates
        ]
        if not benign_indices:
            raise ValueError(f"Could not find Benign/Normal class in classes_: {labels}")
        benign_index = benign_indices[0]
        return ProbabilitySpec(
            task=task,
            class_labels=labels,
            class_source=class_source,
            probability_mapping="p_attack = 1 - p(Benign)",
            benign_class_index=benign_index,
        )

    if is_torch_mlp(pipeline):
        return ProbabilitySpec(
            task=task,
            class_labels=labels,
            class_source="TorchMLPClassifier probability index convention",
            probability_mapping="p_attack = predict_proba[:, 1]",
            attack_class_index=1,
        )

    attack_candidates = {"1", "attack", "malicious", "true"}
    attack_indices = [
        idx for idx, label in enumerate(lower_labels) if label in attack_candidates
    ]
    if not attack_indices:
        raise ValueError(f"Could not find attack class 1 in classes_: {labels}")
    attack_index = attack_indices[0]
    return ProbabilitySpec(
        task=task,
        class_labels=labels,
        class_source=class_source,
        probability_mapping="p_attack = predict_proba[:, class_index_for_1]",
        attack_class_index=attack_index,
    )


def p_attack_from_proba(proba, spec: ProbabilitySpec):
    proba = np.asarray(proba)
    if proba.ndim != 2:
        raise ValueError(f"predict_proba returned non-2D array with shape {proba.shape}")
    if spec.task == "multiclass":
        if spec.benign_class_index is None or spec.benign_class_index >= proba.shape[1]:
            raise ValueError("Invalid Benign class index for multiclass p_attack extraction.")
        return 1.0 - proba[:, spec.benign_class_index]

    if spec.attack_class_index is None or spec.attack_class_index >= proba.shape[1]:
        raise ValueError("Invalid Attack class index for binary p_attack extraction.")
    return proba[:, spec.attack_class_index]


def force_single_thread_inference(pipeline) -> None:
    candidates = [pipeline]
    if hasattr(pipeline, "named_steps"):
        candidates.extend(step for step in pipeline.named_steps.values() if step is not None)

    for estimator in candidates:
        try:
            if hasattr(estimator, "get_params") and "n_jobs" in estimator.get_params(deep=False):
                estimator.set_params(n_jobs=1)
            elif hasattr(estimator, "n_jobs"):
                estimator.n_jobs = 1
        except (TypeError, ValueError):
            continue


def binary_from_attack_column(series: pd.Series) -> np.ndarray:
    attack_type = series.where(series.notna(), "Unknown").astype(str).str.strip()
    return (attack_type.str.lower() != "benign").astype(np.int8).to_numpy()


def binary_from_label_column(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return (numeric.to_numpy(dtype=float) > 0).astype(np.int8)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0,
        "benign": 0,
        "normal": 0,
        "false": 0,
        "1": 1,
        "attack": 1,
        "malicious": 1,
        "true": 1,
    }
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        bad_values = sorted(normalized[mapped.isna()].dropna().unique().tolist())[:10]
        raise ValueError(f"Unrecognized Label values for binary y_true: {bad_values}")
    return mapped.to_numpy(dtype=np.int8)


def binary_from_predictions(values) -> np.ndarray:
    arr = np.asarray(values)
    numeric = pd.to_numeric(pd.Series(arr), errors="coerce")
    if numeric.notna().all():
        return (numeric.to_numpy(dtype=float) > 0).astype(np.int8)

    normalized = pd.Series(arr).astype(str).str.strip().str.lower()
    benign_values = {"0", "benign", "normal", "false"}
    return (~normalized.isin(benign_values)).to_numpy(dtype=np.int8)


def expected_feature_columns(pipeline) -> list[str]:
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)
    if hasattr(pipeline, "named_steps"):
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
            return list(preprocessor.feature_names_in_)
    raise ValueError("Pipeline does not expose fitted raw feature_names_in_.")


def align_features(frame: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=expected_cols, fill_value=0)


def load_unsw_heldout(seed: int, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    print_flush(
        "Loading UNSW held-out split via the H1 training splitter: "
        f"seed={seed}, task={task}, path={UNSW_PATH}"
    )
    data = load_h1_unsw_splits(seed, task)
    X_test = data["X_test"]

    if task == "multiclass":
        y_test_binary = binary_from_attack_column(data["y_test"])
    elif task == "binary":
        y_test_binary = data["y_test"].astype(np.int8)
    else:
        raise ValueError(f"Unknown task: {task}")

    gc.collect()
    return X_test, np.asarray(y_test_binary, dtype=np.int8)


def evaluate_frame(pipeline, X_raw: pd.DataFrame, y_true_binary, spec: ProbabilitySpec) -> tuple:
    y_pred_raw = pipeline.predict(X_raw)
    y_pred_binary = binary_from_predictions(y_pred_raw)
    proba = pipeline.predict_proba(X_raw)
    p_attack = p_attack_from_proba(proba, spec)
    return np.asarray(y_true_binary, dtype=np.int8), y_pred_binary, p_attack


def evaluate_combo(combo: Combo, args, repo_root: Path) -> tuple[dict, pd.DataFrame, dict]:
    started = time.time()
    artifact_path = infer_artifact_path(combo, repo_root, args)
    dataset_path = dataset_path_for(combo, repo_root)
    log_row = {
        **combo.meta(),
        "artifact_path": str(artifact_path),
        "dataset_path": str(dataset_path),
        "status": "started",
        "message": "",
        "n_samples": 0,
        "elapsed_s": np.nan,
    }

    if not artifact_path.exists():
        log_row.update({
            "status": "skipped",
            "message": f"Missing artifact: {artifact_path}",
            "elapsed_s": time.time() - started,
        })
        return {}, pd.DataFrame(), log_row
    if not dataset_path.exists():
        log_row.update({
            "status": "skipped",
            "message": f"Missing dataset: {dataset_path}",
            "elapsed_s": time.time() - started,
        })
        return {}, pd.DataFrame(), log_row

    print_flush("\n" + "=" * 80)
    print_flush(
        "Evaluating "
        f"{combo.experiment} {combo.model} {combo.target_dataset} "
        f"{combo.filtering_strategy} seed={combo.seed}"
    )
    print_flush(f"Artifact: {artifact_path}")
    print_flush(f"Dataset:  {dataset_path}")

    pipeline = joblib.load(artifact_path)
    force_single_thread_inference(pipeline)
    expected_cols = expected_feature_columns(pipeline)
    spec = build_probability_spec(pipeline, combo.task)
    accumulator = CalibrationAccumulator(args.n_bins)

    print_flush(f"Raw feature columns expected by artifact: {len(expected_cols)}")
    print_flush(f"Probability mapping: {spec.probability_mapping}")
    print_flush(f"Class labels: {spec.class_labels}")

    if combo.target_dataset == "UNSW":
        X_test, y_true = load_unsw_heldout(combo.seed, combo.task)
        X_eval = align_features(X_test, expected_cols)
        y_true, y_pred, p_attack = evaluate_frame(pipeline, X_eval, y_true, spec)
        accumulator.update(y_true, y_pred, p_attack)
        del X_test, X_eval, y_true, y_pred, p_attack
        gc.collect()
    else:
        reader = pd.read_csv(dataset_path, chunksize=args.chunk_size, low_memory=False)
        for chunk_id, chunk in enumerate(reader, start=1):
            print_flush(
                f"{combo.target_dataset} chunk {chunk_id}: "
                f"{len(chunk):,} rows before cleaning"
            )
            chunk = prepare_cicids_chunk(chunk)
            if combo.task == "multiclass":
                y_true = binary_from_attack_column(chunk[ATTACK_COL])
            else:
                y_true = binary_from_label_column(chunk[LABEL_COL])
            drop_cols = [col for col in [ATTACK_COL, LABEL_COL] if col in chunk.columns]
            X_base = chunk.drop(columns=drop_cols)
            X_eval = align_features(X_base, expected_cols)
            y_true, y_pred, p_attack = evaluate_frame(pipeline, X_eval, y_true, spec)
            accumulator.update(y_true, y_pred, p_attack)
            print_flush(f"Samples accumulated: {accumulator.n:,}")
            del chunk, X_base, X_eval, y_true, y_pred, p_attack
            gc.collect()

    meta = {
        **combo.meta(),
        "artifact_path": str(artifact_path),
        "dataset_path": str(dataset_path),
        "evaluation_scope": "heldout_test" if combo.target_dataset == "UNSW" else "full_target_stream",
        "chunk_size": "" if combo.target_dataset == "UNSW" else args.chunk_size,
        "n_bins": args.n_bins,
        "probability_mapping": spec.probability_mapping,
        "class_source": spec.class_source,
        "class_labels": json.dumps(spec.class_labels),
        "attack_class_index": spec.attack_class_index,
        "benign_class_index": spec.benign_class_index,
        "target_labels_used_only_posthoc": "Yes",
        "training_performed": False,
        "preprocessing_fit_performed": False,
    }
    metric_row = accumulator.summary_row(meta)
    metric_row.update(reference_comparison_columns(combo, metric_row))
    metric_row = add_delta_warnings(metric_row)
    bins = pd.DataFrame(accumulator.bin_rows(meta))

    log_row.update({
        "status": "completed",
        "message": "ok",
        "n_samples": accumulator.n,
        "elapsed_s": time.time() - started,
    })
    print_flush(
        f"Completed {combo_slug(combo)}: n={accumulator.n:,}, "
        f"ECE={metric_row['ece']:.6f}, Brier={metric_row['brier_score']:.6f}"
    )
    del pipeline
    gc.collect()
    return metric_row, bins, log_row


def reference_comparison_columns(combo: Combo, metric_row: dict) -> dict:
    out = {}
    for metric, reference_value in combo.reference_metrics.items():
        if not np.isnan(reference_value):
            out[f"reference_{metric}"] = reference_value
            current = safe_float(metric_row.get(metric))
            out[f"delta_{metric}"] = current - reference_value
    return out


def add_delta_warnings(metric_row: dict, tolerance: float = 1e-6) -> dict:
    warning_metrics = []
    for metric in ["n_samples", "tp", "tn", "fp", "fn", "binary_f1", "mcc", "fpr", "fnr"]:
        delta_key = f"delta_{metric}"
        if delta_key in metric_row:
            delta = safe_float(metric_row[delta_key])
            if not np.isnan(delta) and abs(delta) > tolerance:
                warning_metrics.append(f"{metric}: delta={delta}")

    metric_row["reference_match_warning"] = "; ".join(warning_metrics)
    metric_row["reference_match_ok"] = len(warning_metrics) == 0
    return metric_row


def write_reliability_plot(bins: pd.DataFrame, combo: Combo, plot_dir: Path) -> Path | None:
    valid = bins[bins["n_samples"] > 0].copy()
    if valid.empty:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print_flush("matplotlib is not available; skipping reliability plot.")
        return None

    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"{combo_slug(combo)}__reliability.png"

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0, label="perfect")
    ax.plot(
        valid["mean_p_attack"],
        valid["empirical_attack_rate"],
        marker="o",
        linewidth=1.8,
        label="observed",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted p_attack")
    ax.set_ylabel("Empirical attack rate")
    ax.set_title(
        f"{combo.experiment} {combo.model} {combo.target_dataset}\n"
        f"{combo.filtering_strategy}, seed={combo.seed}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def merge_existing_output(
    new_df: pd.DataFrame,
    path: Path,
    identity_cols: list[str],
    overwrite: bool,
) -> pd.DataFrame:
    if overwrite or not path.exists():
        return new_df
    old_df = pd.read_csv(path)
    combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    present_identity_cols = [col for col in identity_cols if col in combined.columns]
    if present_identity_cols:
        combined = combined.drop_duplicates(subset=present_identity_cols, keep="last")
    return combined


def write_mean_std_summary(all_metrics: pd.DataFrame, output_dir: Path) -> Path:
    metric_cols = [
        "n_samples",
        "ece",
        "brier_score",
        "log_loss",
        "nll",
        "mean_p_attack",
        "true_attack_rate",
        "pred_attack_rate",
        "model_pred_attack_rate",
        "p05_pred_attack_rate",
        "binary_f1",
        "weighted_f1",
        "mcc",
        "fpr",
        "fnr",
        "tp",
        "tn",
        "fp",
        "fn",
        "binary_f1_p05",
        "weighted_f1_p05",
        "mcc_p05",
        "fpr_p05",
        "fnr_p05",
        "tp_p05",
        "tn_p05",
        "fp_p05",
        "fn_p05",
    ]
    available = [col for col in metric_cols if col in all_metrics.columns]
    grouped = all_metrics.groupby(GROUP_COLS, dropna=False)
    summary = grouped[available].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([part for part in col if part])
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    seed_counts = grouped["seed"].nunique().reset_index(name="n_seeds")
    row_counts = grouped.size().reset_index(name="n_completed_rows")
    summary = summary.merge(seed_counts, on=GROUP_COLS, how="left")
    summary = summary.merge(row_counts, on=GROUP_COLS, how="left")
    path = output_dir / "calibration_summary.csv"
    summary.to_csv(path, index=False)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference-only Step 5 E1-E4 calibration/ECE analysis."
    )
    parser.add_argument(
        "--mode",
        choices=["sanity", "full"],
        default="sanity",
        help="sanity runs only E1 RandomForest CICIDS2018 seed42 none.",
    )
    parser.add_argument(
        "--input-long-csv",
        default="results/generated/h1/all_results_long.csv",
    )
    parser.add_argument(
        "--multiseed-result-dir",
        default="results/generated/h1",
    )
    parser.add_argument(
        "--output-dir",
        default="results/generated/calibration",
    )
    parser.add_argument("--chunk-size", type=int, default=STRESS_B_CHUNK_SIZE)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--skip-unsw",
        action="store_true",
        help=(
            "Exclude UNSW held-out rows from the selected combinations. "
            "Useful when focusing Step 5 on cross-domain target calibration."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace output CSVs instead of merging with existing files in "
            "--output-dir. By default staged runs are merged by combination key."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="In full mode, raise if the merged run log contains errors or skipped combinations.",
    )
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--target-datasets", nargs="*", default=None)
    parser.add_argument("--filtering-strategies", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = as_repo_path(args.output_dir, repo_root)
    from cross_domain_nids.release_paths import require_generated_output

    output_dir = require_generated_output(
        output_dir, "results", repository_root=repo_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = output_dir / "bin_level"
    plot_dir = output_dir / "reliability_plots"

    combos = load_combos(args, repo_root)
    if not combos:
        raise ValueError("No E1-E4 combinations selected for calibration.")

    print_flush(f"Mode: {args.mode}")
    print_flush(f"Selected combinations: {len(combos)}")
    print_flush(f"Output directory: {output_dir}")
    print_flush("This is inference-only: no model fitting and no preprocessing fitting.")
    if args.overwrite:
        print_flush("Output mode: overwrite existing aggregate CSVs.")
    else:
        print_flush("Output mode: merge with existing aggregate CSVs by combination key.")

    metric_rows = []
    bin_frames = []
    log_rows = []
    for combo in combos:
        try:
            metric_row, bins, log_row = evaluate_combo(combo, args, repo_root)
        except Exception as exc:  # noqa: BLE001 - continue full-run after one failed combo.
            log_row = {
                **combo.meta(),
                "artifact_path": str(infer_artifact_path(combo, repo_root, args)),
                "dataset_path": str(dataset_path_for(combo, repo_root)),
                "status": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "n_samples": 0,
                "elapsed_s": np.nan,
            }
            metric_row = {}
            bins = pd.DataFrame()
            print_flush(f"ERROR {combo_slug(combo)}: {log_row['message']}")

        log_rows.append(log_row)
        if metric_row:
            metric_rows.append(metric_row)
        if not bins.empty:
            per_combo_path = bin_dir / f"{combo_slug(combo)}__bins.csv"
            per_combo_path.parent.mkdir(parents=True, exist_ok=True)
            bins.to_csv(per_combo_path, index=False)
            if not args.no_plots:
                plot_path = write_reliability_plot(bins, combo, plot_dir)
                if plot_path is not None and metric_row:
                    metric_row["reliability_plot"] = str(plot_path)
            bin_frames.append(bins)

    log_path = output_dir / "calibration_run_log.csv"
    new_log = pd.DataFrame(log_rows)
    all_log = merge_existing_output(new_log, log_path, COMBO_COLS, args.overwrite)
    all_log.to_csv(log_path, index=False)
    error_count = int(all_log["status"].eq("error").sum()) if "status" in all_log.columns else 0
    skipped_count = int(all_log["status"].eq("skipped").sum()) if "status" in all_log.columns else 0

    if not metric_rows:
        print_flush(f"Saved run log: {log_path}")
        raise RuntimeError("No combinations completed successfully.")

    all_path = output_dir / "calibration_seed_metrics.csv"
    new_metrics = pd.DataFrame(metric_rows)
    all_metrics = merge_existing_output(new_metrics, all_path, COMBO_COLS, args.overwrite)
    all_metrics.to_csv(all_path, index=False)

    bins_all_path = output_dir / "calibration_bins.csv"
    new_bins = pd.concat(bin_frames, ignore_index=True)
    all_bins = merge_existing_output(
        new_bins,
        bins_all_path,
        [*COMBO_COLS, "bin_id"],
        args.overwrite,
    )
    all_bins.to_csv(bins_all_path, index=False)
    summary_path = write_mean_std_summary(all_metrics, output_dir)

    print_flush(f"Saved long calibration metrics: {all_path}")
    print_flush(f"Saved bin-level calibration table: {bins_all_path}")
    print_flush(f"Saved mean/std summary: {summary_path}")
    print_flush(f"Saved run log: {log_path}")

    if args.mode == "full" and (error_count > 0 or skipped_count > 0):
        message = (
            "WARNING: calibration completed with "
            f"{error_count} errors and {skipped_count} skipped combinations. "
            f"Check {log_path}."
        )
        print_flush(message)
        if args.strict:
            raise RuntimeError(message)


if __name__ == "__main__":
    main()
