"""Formal H1 corrected-split E1/E2 retraining runner.

The default invocation is a no-fit dry run. Model fitting requires both
``--execute`` and a passing frozen design review. One seed is executed per
invocation so completed seed bundles are independently auditable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
POLICY_PATH = REPOSITORY_ROOT / "protocols" / "h1" / "training_policy.json"
MODEL_PARAMETERS_PATH = REPOSITORY_ROOT / "data" / "model_hyperparameters.json"
H1_SPLIT_ROOT = REPOSITORY_ROOT / "data" / "splits" / "h1"
EXPECTED_POLICY_SHA = "2e269b92b37f765ae45840c569da0642162898522d22dde77238ba5527d81f48"
EXPECTED_PROTOCOL_ID = "NIDS-H1-CORRECTED-SPLIT-SENS-20260717-v1.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values, dtype) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def load_effective_policy() -> tuple[dict, dict]:
    policy_sha = sha256_file(POLICY_PATH)
    if policy_sha != EXPECTED_POLICY_SHA:
        raise RuntimeError(f"Policy hash mismatch: {policy_sha}")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy["protocol_id"] != EXPECTED_PROTOCOL_ID:
        raise RuntimeError(f"Unexpected effective protocol: {policy['protocol_id']}")
    return policy, {"policy_sha256": policy_sha}


def load_split(policy: dict, seed: int):
    import numpy as np

    spec = policy["corrected_splits"]["files"][str(seed)]
    path = (H1_SPLIT_ROOT / Path(spec["path"]).name).resolve()
    if sha256_file(path) != spec["file_sha256"]:
        raise RuntimeError(f"Corrected split file hash mismatch for seed {seed}")
    with np.load(path, allow_pickle=False) as archive:
        retained = archive["retained_original_index"].astype(np.int64, copy=False)
        assignment = archive["assignment"].astype(np.uint8, copy=False)
        embedded_seed = int(archive["seed"][0])
        embedded_protocol = str(archive["protocol_id"][0])
    if embedded_seed != seed or embedded_protocol != policy["corrected_splits"]["protocol_id"]:
        raise RuntimeError(f"Corrected split metadata mismatch for seed {seed}")
    if sha256_array(assignment, np.uint8) != spec["assignment_sha256"]:
        raise RuntimeError(f"Corrected split assignment hash mismatch for seed {seed}")
    if sha256_array(retained, np.int64) != policy["corrected_splits"]["retained_original_index_sha256"]:
        raise RuntimeError(f"Retained-population hash mismatch for seed {seed}")
    if len(retained) != len(assignment) or set(np.unique(assignment).tolist()) != {0, 1, 2}:
        raise RuntimeError(f"Malformed corrected split for seed {seed}")
    return retained, assignment, path


def set_all_seeds(seed: int) -> None:
    from cross_domain_nids.determinism import configure_determinism

    configure_determinism(seed)


def construct_released_estimator(
    experiment: str,
    model: str,
    seed: int,
    parameters_path: Path = MODEL_PARAMETERS_PATH,
):
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from cross_domain_nids.model_factory import construct_estimator, load_parameter_record

    record = load_parameter_record(experiment, model, parameters_path)
    estimator = construct_estimator(experiment, model, seed, parameters_path)
    return estimator, parameters_path.resolve(), record


def align_features(frame, feature_cols):
    present = [column for column in feature_cols if column in frame.columns]
    aligned = frame[present].copy()
    for column in feature_cols:
        if column not in aligned.columns:
            aligned[column] = 0
    return aligned[feature_cols]


def empty_counts() -> dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


def update_counts(counts: dict[str, int], y_true, y_pred) -> None:
    import numpy as np

    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    counts["tp"] += int(((true == 1) & (pred == 1)).sum())
    counts["tn"] += int(((true == 0) & (pred == 0)).sum())
    counts["fp"] += int(((true == 0) & (pred == 1)).sum())
    counts["fn"] += int(((true == 1) & (pred == 0)).sum())


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, tn, fp, fn = (int(counts[k]) for k in ("tp", "tn", "fp", "fn"))
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    binary_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / (denominator**0.5) if denominator else 0.0
    benign_precision = tn / (tn + fn) if tn + fn else 0.0
    benign_recall = tn / (tn + fp) if tn + fp else 0.0
    benign_f1 = (
        2 * benign_precision * benign_recall / (benign_precision + benign_recall)
        if benign_precision + benign_recall
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


def binary_prediction(task: str, values):
    import numpy as np

    array = np.asarray(values)
    return (array.astype(str) != "Benign").astype(int) if task == "multiclass" else array.astype(int)


def result_row(experiment: str, seed: int, model: str, dataset: str, scope: str, metrics: dict, **extra) -> dict:
    row = {
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "experiment": experiment,
        "seed": seed,
        "model": model,
        "target_dataset": dataset,
        "evaluation_scope": scope,
        "filtering_strategy": "none",
        "metric_protocol": "binary_detection_metrics",
        **metrics,
        **extra,
    }
    return row


def emit(log_path: Path, event: str, **fields) -> None:
    payload = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    try:
        print(json.dumps(payload, sort_keys=True), flush=True)
    except OSError:
        # A detached Windows process can lose its inherited stdout handle.
        # The append-only JSONL file remains the authoritative execution log.
        pass
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def preflight(
    seed: int | None = None,
    parameters_path: Path = MODEL_PARAMETERS_PATH,
) -> dict:
    import numpy as np
    import pandas as pd
    import sklearn
    import torch
    import xgboost

    policy, freeze = load_effective_policy()
    for relative, expected_sha in policy["implementation_dependencies"].items():
        if relative == "project_root":
            continue
        dependency = REPOSITORY_ROOT / relative
        if not dependency.is_file() or sha256_file(dependency) != expected_sha:
            raise RuntimeError(f"Frozen implementation dependency mismatch: {relative}")
    seeds = [seed] if seed is not None else policy["seeds"]
    retained_reference = None
    split_summary = {}
    for value in seeds:
        retained, assignment, path = load_split(policy, value)
        if retained_reference is None:
            retained_reference = retained
        elif not np.array_equal(retained, retained_reference):
            raise RuntimeError("Corrected splits do not share a retained source population")
        split_summary[str(value)] = {
            "path": str(path),
            "retained": int(len(retained)),
            "train": int((assignment == 0).sum()),
            "validation": int((assignment == 1).sum()),
            "test": int((assignment == 2).sum()),
        }
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    for experiment in ("E1", "E2"):
        for model in policy["models"]:
            estimator, _, record = construct_released_estimator(
                experiment, model, seeds[0], parameters_path
            )
            actual = estimator.get_params(deep=False)
            if actual.get("random_state") != seeds[0]:
                raise RuntimeError(f"Constructor seed mismatch: {experiment}/{model}")
            if record["task"] != policy["experiments"][experiment]["source_label"].replace(
                "Attack", "multiclass"
            ).replace("Label", "binary"):
                raise RuntimeError(f"Constructor task mismatch: {experiment}/{model}")
    cuda_available = torch.cuda.is_available()
    return {
        "status": "Pass",
        "mode": "dry_run_no_fit",
        "protocol_id": policy["protocol_id"],
        **freeze,
        "model_parameters": str(parameters_path.resolve()),
        "model_construction": "released_parameter_json",
        "splits": split_summary,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "xgboost": xgboost.__version__,
            "cuda_available": cuda_available,
            "cuda_device": (
                torch.cuda.get_device_name(0) if cuda_available else "not_available"
            ),
        },
    }


def execute_seed(
    seed: int,
    output_dir: Path,
    chunk_size: int,
    data_dir: Path,
    parameters_path: Path,
) -> None:
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.pipeline import Pipeline

    dry = preflight(seed, parameters_path)
    policy, freeze = load_effective_policy()
    if not torch.cuda.is_available():
        raise RuntimeError("Formal H1 execution requires CUDA for the frozen MLP configuration")
    if seed not in policy["seeds"]:
        raise ValueError(f"Seed {seed} is not frozen in the protocol")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"execution_seed{seed}.jsonl"
    completion_path = output_dir / f"seed{seed}_complete.json"
    if completion_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed seed bundle: {completion_path}")
    if log_path.exists():
        raise FileExistsError(f"Refusing to append to a prior incomplete run: {log_path}")
    emit(
        log_path,
        "execution_start",
        seed=seed,
        protocol_id=policy["protocol_id"],
        freeze=freeze,
        model_construction="released_parameter_json",
        dry_run=dry,
    )

    sys.path.insert(0, str(SRC_ROOT))
    from cross_domain_nids.config import ATTACK_COL, LABEL_COL
    from cross_domain_nids.preprocessing.netflow import (
        build_preprocessor,
        clean_l7_proto,
        load_and_clean_data,
        prepare_cicids_chunk,
    )

    set_all_seeds(seed)
    retained, assignment, split_path = load_split(policy, seed)
    train_idx = retained[assignment == 0]
    val_idx = retained[assignment == 1]
    test_idx = retained[assignment == 2]
    source_path = data_dir / Path(policy["inputs"]["source"]["path"]).name
    emit(log_path, "source_load_start", path=str(source_path))
    frame = clean_l7_proto(load_and_clean_data(source_path))
    if len(frame) <= int(retained.max()):
        raise RuntimeError("Corrected split indices exceed source row count")
    feature_cols = [column for column in frame.columns if column not in (ATTACK_COL, LABEL_COL)]
    X = frame[feature_cols]
    y_by_task = {
        "multiclass": frame[ATTACK_COL].astype(str),
        "binary": frame[LABEL_COL].astype(int),
    }
    emit(
        log_path,
        "source_load_complete",
        rows=len(frame),
        retained=len(retained),
        train=len(train_idx),
        validation=len(val_idx),
        test=len(test_idx),
        features=len(feature_cols),
        split_path=str(split_path),
    )

    experiment_specs = {
        "E1": {"task": "multiclass", "training_protocol": "pure_transfer_multiclass", "mapping": "Benign->0; all_non_Benign_attack_classes->1"},
        "E2": {"task": "binary", "training_protocol": "binary_source_training", "mapping": "binary_label_0_Benign_1_Attack"},
    }
    pipelines = {}
    rows = []
    validation_rows = []
    manifests = []
    artifact_root = output_dir / "artifacts"

    for experiment, experiment_spec in experiment_specs.items():
        task = experiment_spec["task"]
        y = y_by_task[task]
        for model in policy["models"]:
            set_all_seeds(seed)
            estimator, parameter_path, parameter_record = construct_released_estimator(
                experiment, model, seed, parameters_path
            )
            pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
            emit(log_path, "fit_start", experiment=experiment, model=model, seed=seed)
            fit_start = time.perf_counter()
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            train_time = time.perf_counter() - fit_start
            artifact_path = artifact_root / experiment.lower() / f"seed{seed}" / f"{model}.joblib"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, artifact_path)
            artifact_sha = sha256_file(artifact_path)
            fitted_model = pipeline.named_steps["model"]
            manifest = {
                "protocol_id": policy["protocol_id"],
                "seed": seed,
                "experiment": experiment,
                "task": task,
                "model": model,
                "model_construction": "released_parameter_json",
                "parameter_path": str(parameter_path),
                "source_template_sha256": parameter_record["source_template_sha256"],
                "template_parameters_json": json.dumps(
                    parameter_record["parameters"], sort_keys=True, default=str
                ),
                "effective_random_state": seed,
                "train_rows": int(len(train_idx)),
                "validation_rows_external_diagnostic_only": int(len(val_idx)),
                "test_rows": int(len(test_idx)),
                "train_time_s": train_time,
                "artifact": str(artifact_path),
                "artifact_sha256": artifact_sha,
                "mlp_internal_validation_source": getattr(fitted_model, "validation_source_", None),
                "mlp_epochs": getattr(fitted_model, "n_epochs_", None),
                "mlp_best_val_loss": getattr(fitted_model, "best_val_loss_", None),
            }
            manifests.append(manifest)
            common = {
                "training_protocol": experiment_spec["training_protocol"],
                "prediction_mapping": experiment_spec["mapping"],
                "target_label_usage": "final_metric_computation_only",
                "hyperparameter_tuning": "none_constructed_from_released_parameters",
                "train_time_s": train_time,
                "artifact": str(artifact_path),
                "artifact_sha256": artifact_sha,
            }
            for idx, scope, collector in ((test_idx, "corrected_source_test", rows), (val_idx, "corrected_source_validation_diagnostic_only", validation_rows)):
                predicted = pipeline.predict(X.iloc[idx])
                true_binary = binary_prediction(task, y.iloc[idx])
                pred_binary = binary_prediction(task, predicted)
                counts = empty_counts()
                update_counts(counts, true_binary, pred_binary)
                collector.append(result_row(experiment, seed, model, "UNSW", scope, metrics_from_counts(counts), **common))
            pipelines[(experiment, model)] = {"pipeline": pipeline, "task": task, "common": common}
            emit(log_path, "fit_complete", experiment=experiment, model=model, seed=seed, train_time_s=train_time, artifact_sha256=artifact_sha)

    del frame, X, y_by_task
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for dataset_name, target_spec in policy["inputs"]["targets"].items():
        target_path = data_dir / Path(target_spec["path"]).name
        counts_by_model = {key: empty_counts() for key in pipelines}
        chunk_count = 0
        target_rows = 0
        emit(log_path, "target_start", dataset=dataset_name, path=str(target_path))
        for chunk in pd.read_csv(target_path, chunksize=chunk_size, low_memory=False):
            chunk_count += 1
            chunk = prepare_cicids_chunk(chunk)
            target_rows += len(chunk)
            features = align_features(chunk, feature_cols)
            true_by_task = {
                "multiclass": (chunk[ATTACK_COL].astype(str).to_numpy() != "Benign").astype(int),
                "binary": chunk[LABEL_COL].astype(int).to_numpy(),
            }
            for key, bundle in pipelines.items():
                predicted = bundle["pipeline"].predict(features)
                pred_binary = binary_prediction(bundle["task"], predicted)
                update_counts(counts_by_model[key], true_by_task[bundle["task"]], pred_binary)
                del predicted, pred_binary
            emit(log_path, "target_chunk_complete", dataset=dataset_name, chunk=chunk_count, rows=len(chunk), cumulative_rows=target_rows)
            del chunk, features, true_by_task
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for (experiment, model), counts in counts_by_model.items():
            bundle = pipelines[(experiment, model)]
            rows.append(
                result_row(
                    experiment,
                    seed,
                    model,
                    dataset_name,
                    "full_target",
                    metrics_from_counts(counts),
                    **bundle["common"],
                )
            )
        emit(log_path, "target_complete", dataset=dataset_name, chunks=chunk_count, rows=target_rows)

    results_path = output_dir / f"results_seed{seed}.csv"
    validation_path = output_dir / f"source_validation_diagnostic_seed{seed}.csv"
    manifest_path = output_dir / f"training_manifest_seed{seed}.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
    pd.DataFrame(manifests).to_csv(manifest_path, index=False)
    expected_rows = 2 * len(policy["models"]) * (1 + len(policy["inputs"]["targets"]))
    if len(rows) != expected_rows or len(validation_rows) != 2 * len(policy["models"]) or len(manifests) != 2 * len(policy["models"]):
        raise RuntimeError("Seed output coverage mismatch")
    completion = {
        "protocol_id": policy["protocol_id"],
        "seed": seed,
        **freeze,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "result_rows": len(rows),
        "validation_diagnostic_rows": len(validation_rows),
        "training_runs": len(manifests),
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
        "validation_path": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "training_manifest_path": str(manifest_path),
        "training_manifest_sha256": sha256_file(manifest_path),
        "execution_log_path": str(log_path),
    }
    completion_path.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    emit(log_path, "execution_complete", seed=seed, completion_path=str(completion_path), result_rows=len(rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Authorize fitting after all frozen checks pass")
    parser.add_argument("--seed", type=int, choices=[42, 123, 2024])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "generated" / "h1",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "raw",
        help="Directory containing the four provider CSV files.",
    )
    parser.add_argument(
        "--model-parameters",
        type=Path,
        default=MODEL_PARAMETERS_PATH,
        help="Released JSON used to construct estimators without joblib templates.",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.execute:
        print(
            json.dumps(
                preflight(args.seed, args.model_parameters),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    from cross_domain_nids.release_paths import require_generated_output

    args.output_dir = require_generated_output(args.output_dir, "results")
    if args.seed is None:
        raise SystemExit("--execute requires exactly one frozen --seed")
    if args.chunk_size != 1_000_000:
        raise SystemExit("Formal execution requires the frozen chunk size 1000000")
    execute_seed(
        args.seed,
        args.output_dir.resolve(),
        args.chunk_size,
        args.data_dir.resolve(),
        args.model_parameters.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
