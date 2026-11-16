"""Shared training and statistical helpers for corrected H2/H3 experiments."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from cross_domain_nids.config import STRESS_B_CHUNK_SIZE, UNSW_PATH
from cross_domain_nids.protocol_utils import (
    PROTOCOL_ID, PROTOCOL_RESULTS_ROOT, ProtocolError, atomic_write_json,
    audited_run, completed_marker, deterministic_seed, hash_file, hash_json,
    paired_bootstrap, require_execute_flag, validate_feature_order,
    validate_protocol_lock_checklist, write_immutable_json,
)
from experiments.h1.run_multiseed_training import (
    CORR_THRESHOLD, E3_E4_MODELS, TARGET_DATASETS, compute_redundant_features,
    load_unsw_splits, standardize_long_columns, train_and_eval_filtered_model,
)


SEEDS = [42, 123, 2024]
TARGETS = ["CICIDS2018", "BoT-IoT", "ToN-IoT"]
MODELS = ["RandomForest", "MLP"]
STRATEGIES = [
    "none", "remove_shift_top3", "remove_shift_top5", "remove_shift_top10",
    "remove_shift_top15", "remove_exact_duplicates", "remove_corr_gt_0.98",
    "remove_shift_top5_plus_corr_gt_0.98",
]
TARGET_SPECIFIC = {
    "remove_shift_top3", "remove_shift_top5", "remove_shift_top10",
    "remove_shift_top15", "remove_shift_top5_plus_corr_gt_0.98",
}
UNIVERSAL = {"none", "remove_exact_duplicates", "remove_corr_gt_0.98"}
METRICS = ["mcc", "binary_f1", "fnr", "fpr"]
TARGET_SLUGS = {
    "CICIDS2018": "cicids2018",
    "BoT-IoT": "bot_iot",
    "ToN-IoT": "ton_iot",
}
MODEL_SLUGS = {"RandomForest": "random_forest", "MLP": "mlp"}


def h2_root(args: argparse.Namespace) -> Path:
    return args.output_root / "h2"


def selection_lock_path(args: argparse.Namespace, target: str, model: str) -> Path:
    name = f"{TARGET_SLUGS[target]}__{MODEL_SLUGS[model]}.json"
    return h2_root(args) / "selection_locks" / name


def protocol_data_paths() -> list[str | Path]:
    return [UNSW_PATH, *TARGET_DATASETS.values()]


def protocol_code_paths() -> list[Path]:
    return [ROOT / "experiments", ROOT / "src" / "cross_domain_nids"]


def validate_runtime_lock(args: argparse.Namespace) -> None:
    validate_protocol_lock_checklist(
        args.output_root, policy_paths={"h2": args.policy, "h3": args.h3_policy},
        seeds=SEEDS, targets=TARGETS, data_paths=protocol_data_paths(),
        code_paths=protocol_code_paths(),
    )


def candidate_target_path(args: argparse.Namespace, target: str) -> Path:
    return h2_root(args) / "candidates" / "by_target" / f"{target}.csv"


def ranking_path(args: argparse.Namespace, target: str, seed: int) -> Path:
    return h2_root(args) / "shift" / "rankings" / f"{target}_seed{seed}.csv"


def normalize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = standardize_long_columns(frame.copy())
    aliases = {
        "MCC": "mcc", "Attack-class Binary-F1": "binary_f1",
        "FNR": "fnr", "FPR": "fpr", "TP": "tp", "TN": "tn", "FP": "fp", "FN": "fn",
    }
    for source, target in aliases.items():
        if target not in frame and source in frame:
            frame[target] = frame[source]
    missing = [name for name in ["seed", "model", "target_dataset", "filtering_strategy", *METRICS] if name not in frame]
    if missing:
        raise ProtocolError(f"Candidate results missing columns: {missing}")
    return frame


def run_feature_shift(args: argparse.Namespace) -> None:
    require_execute_flag(args.execute, "H2 full-target JSD")
    validate_runtime_lock(args)
    shift_dir = h2_root(args) / "shift"
    command = [
        sys.executable, str(ROOT / "experiments" / "h2" / "compute_feature_shift.py"),
        "--seeds", *map(str, SEEDS), "--targets", *TARGETS,
        "--n-bins", "50", "--bin-strategy", "quantile", "--top-k", "15",
        "--chunk-size", str(args.chunk_size), "--output-dir", str(shift_dir / "raw_feature_shift"),
    ]
    if args.print_command_only:
        print(subprocess.list2cmdline(command))
        return
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise ProtocolError(f"Feature-shift subprocess failed with exit code {result.returncode}")
    candidates = list((shift_dir / "raw_feature_shift").glob("*feature_shift*long*.csv"))
    if len(candidates) != 1:
        raise ProtocolError(f"Expected one feature-shift JSD file; found {candidates}")
    long = pd.read_csv(candidates[0])
    feature_col = "feature"
    jsd_col = "jsd" if "jsd" in long else "JSD"
    required = {"seed", "target_dataset", feature_col, jsd_col}
    if not required.issubset(long.columns):
        raise ProtocolError(
            f"Feature-shift JSD output lacks {sorted(required - set(long.columns))}"
        )
    out_rows = []
    rank_dir = shift_dir / "rankings"; rank_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for target in TARGETS:
            group = long[(long["seed"] == seed) & (long["target_dataset"] == target)].copy()
            if len(group) != 49:
                raise ProtocolError(f"JSD ranking {target}/seed{seed} has {len(group)} rows, expected 49")
            group["_feature_order"] = pd.Categorical(group[feature_col], categories=group[feature_col].tolist(), ordered=True)
            group = group.sort_values([jsd_col, "_feature_order"], ascending=[False, True]).drop(columns="_feature_order")
            group["rank"] = np.arange(1, 50)
            group.to_csv(ranking_path(args, target, seed), index=False)
            out_rows.append(group)
    merged = pd.concat(out_rows, ignore_index=True)
    merged.to_csv(shift_dir / "jsd_long.csv", index=False)
    merged.groupby(["target_dataset", feature_col], as_index=False)[jsd_col].agg(["mean", "std"]).reset_index().to_csv(
        shift_dir / "jsd_mean_std.csv", index=False,
    )


def shift_features(args: argparse.Namespace, target: str, seed: int, k: int) -> list[str]:
    path = ranking_path(args, target, seed)
    if not path.exists():
        raise ProtocolError(f"Missing frozen JSD ranking: {path}")
    frame = pd.read_csv(path)
    if len(frame) != 49 or "feature" not in frame:
        raise ProtocolError(f"Invalid ranking file: {path}")
    return frame.sort_values("rank")["feature"].head(k).tolist()


def masks_for_seed(args: argparse.Namespace, data: dict[str, Any], seed: int) -> dict[str, dict[str, list[str]]]:
    helper_args = SimpleNamespace(
        output_dir=str(h2_root(args) / "mask_inputs"), corr_sample_rows=args.corr_sample_rows,
        corr_threshold=CORR_THRESHOLD,
    )
    duplicate, corr = compute_redundant_features(data, seed, helper_args)
    masks: dict[str, dict[str, list[str]]] = {target: {} for target in TARGETS}
    for target in TARGETS:
        masks[target] = {
            "none": [], "remove_exact_duplicates": duplicate,
            "remove_corr_gt_0.98": corr,
            "remove_shift_top3": shift_features(args, target, seed, 3),
            "remove_shift_top5": shift_features(args, target, seed, 5),
            "remove_shift_top10": shift_features(args, target, seed, 10),
            "remove_shift_top15": shift_features(args, target, seed, 15),
        }
        masks[target]["remove_shift_top5_plus_corr_gt_0.98"] = list(dict.fromkeys(
            masks[target]["remove_shift_top5"] + corr
        ))
    return masks


def training_args(args: argparse.Namespace, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(output_dir),
        model_parameters=args.model_parameters,
        chunk_size=args.chunk_size,
    )


def run_candidates(args: argparse.Namespace) -> None:
    require_execute_flag(args.execute, "H2 candidate training")
    validate_runtime_lock(args)
    candidates_root = h2_root(args) / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    (candidates_root / "by_target").mkdir(parents=True, exist_ok=True)
    (h2_root(args) / "analysis").mkdir(parents=True, exist_ok=True)
    target_rows: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}
    for target in TARGETS:
        path = candidate_target_path(args, target)
        if path.exists():
            target_rows[target] = pd.read_csv(path).to_dict("records")
    existing = {
        (int(row["seed"]), row["model"], row["target_dataset"], row["filtering_strategy"])
        for rows in target_rows.values() for row in rows
    }
    for seed in args.seeds:
        data = load_unsw_splits(seed, "binary")
        validate_feature_order(data["feature_cols"])
        masks = masks_for_seed(args, data, seed)
        for model in args.models:
            for strategy in UNIVERSAL:
                missing_targets = [t for t in TARGETS if (seed, model, t, strategy) not in existing]
                if missing_targets:
                    unit = candidates_root / "units" / f"seed{seed}" / model / strategy
                    with audited_run(unit, {"stage": "H2-candidate", "seed": seed, "model": model, "strategy": strategy}):
                        rows = train_and_eval_filtered_model(
                            data, seed, model, "H2_frozen_candidates", strategy,
                            masks[TARGETS[0]][strategy], missing_targets,
                            training_args(args, unit),
                        )
                        for row in normalize_metrics(pd.DataFrame(rows)).to_dict("records"):
                            row.update({"protocol_id": PROTOCOL_ID, "mask_hash": hash_json(json.loads(row["removed_features"]))})
                            target_rows[row["target_dataset"]].append(row)
                        completed_marker(unit, {"status": "complete"})
            for target in TARGETS:
                for strategy in TARGET_SPECIFIC:
                    if (seed, model, target, strategy) in existing:
                        continue
                    unit = candidates_root / "units" / f"seed{seed}" / target / model / strategy
                    with audited_run(unit, {"stage": "H2-candidate", "seed": seed, "target": target,
                                                    "model": model, "strategy": strategy}):
                        rows = train_and_eval_filtered_model(
                            data, seed, model, "H2_frozen_candidates", strategy,
                            masks[target][strategy], [target], training_args(args, unit),
                        )
                        row = normalize_metrics(pd.DataFrame(rows)).iloc[0].to_dict()
                        row.update({"protocol_id": PROTOCOL_ID, "mask_hash": hash_json(json.loads(row["removed_features"]))})
                        target_rows[target].append(row)
                        completed_marker(unit, {"status": "complete"})
                pd.DataFrame(target_rows[target]).to_csv(candidate_target_path(args, target), index=False)
        del data; gc.collect()
    all_rows = pd.concat([pd.DataFrame(rows) for rows in target_rows.values()], ignore_index=True)
    all_rows.to_csv(h2_root(args) / "analysis" / "all_candidates_long.csv", index=False)


def read_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ProtocolError(
            f"Frozen H2 Section 3.7 policy is required: {path}. "
            "Selection cannot be frozen from an unspecified eligibility/tie-break rule."
        )
    policy = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_id", "seeds", "targets", "models",
        "min_mean_mcc_delta", "min_mean_binary_f1_delta", "max_mean_fnr_delta",
        "development_required_positive_seed_fraction", "random_replicates_per_seed",
        "final_required_positive_seeds", "final_required_seed_count", "tie_break",
        "model_parameters_sha256", "model_factory_sha256",
        "pipeline_models_sha256", "determinism_sha256",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ProtocolError(f"H2 policy missing keys: {missing}")
    allowed_ties = ["mean_mcc_delta", "mean_binary_f1_delta", "mean_fnr_delta", "strategy_order"]
    if any(key not in allowed_ties for key in policy["tie_break"]):
        raise ProtocolError(f"Unsupported H2 tie-break: {policy['tie_break']}")
    expected = {
        "protocol_id": PROTOCOL_ID, "seeds": SEEDS, "targets": TARGETS,
        "models": MODELS, "random_replicates_per_seed": 20,
        "final_required_positive_seeds": len(SEEDS), "final_required_seed_count": len(SEEDS),
    }
    mismatches = {key: (policy.get(key), value) for key, value in expected.items() if policy.get(key) != value}
    if mismatches:
        raise ProtocolError(f"H2 policy is not frozen for this protocol: {mismatches}")
    return policy


def validate_released_model_inputs(args: argparse.Namespace) -> None:
    policy = read_policy(args.policy)
    if hash_file(args.model_parameters) != policy["model_parameters_sha256"]:
        raise ProtocolError("Released model parameter JSON hash mismatch")
    factory = ROOT / "src" / "cross_domain_nids" / "model_factory.py"
    if hash_file(factory) != policy["model_factory_sha256"]:
        raise ProtocolError("Released model factory hash mismatch")
    pipeline_models = (
        ROOT / "src" / "cross_domain_nids" / "models" / "pipeline_models.py"
    )
    if hash_file(pipeline_models) != policy["pipeline_models_sha256"]:
        raise ProtocolError("Released pipeline-model implementation hash mismatch")
    determinism = ROOT / "src" / "cross_domain_nids" / "determinism.py"
    if hash_file(determinism) != policy["determinism_sha256"]:
        raise ProtocolError("Released determinism implementation hash mismatch")


def validate_development_cartesian(
    frame: pd.DataFrame, development: list[str], model: str,
) -> None:
    key_columns = ["filtering_strategy", "target_dataset", "seed"]
    missing_columns = sorted(set(["model", *key_columns, *METRICS]) - set(frame.columns))
    if missing_columns:
        raise ProtocolError(f"Development results lack columns for {model}: {missing_columns}")
    if set(frame["model"]) != {model}:
        raise ProtocolError(f"Development frame contains unexpected models for {model}")
    duplicate_mask = frame.duplicated(subset=key_columns, keep=False)
    if duplicate_mask.any():
        duplicate_keys = frame.loc[duplicate_mask, key_columns].to_dict("records")
        raise ProtocolError(f"Duplicate development candidate keys for {model}: {duplicate_keys}")
    expected = {
        (strategy, target, seed)
        for strategy in STRATEGIES for target in development for seed in SEEDS
    }
    actual = {
        (str(row.filtering_strategy), str(row.target_dataset), int(row.seed))
        for row in frame[key_columns].itertuples(index=False)
    }
    if actual != expected or len(frame) != len(expected):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ProtocolError(
            f"Development Cartesian coverage mismatch for {model}: "
            f"expected_rows={len(expected)}, actual_rows={len(frame)}, "
            f"missing={missing}, extra={extra}"
        )
    metric_values = frame[METRICS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(metric_values).all():
        raise ProtocolError(f"Development metrics contain missing/non-finite values for {model}")


def validate_chosen_masks(
    chosen: pd.DataFrame, lock: dict[str, Any], target: str, model: str,
) -> None:
    required = {"seed", "mask_hash", "removed_features"}
    missing = sorted(required - set(chosen.columns))
    if missing:
        raise ProtocolError(f"Chosen candidate lacks mask audit columns for {target}/{model}: {missing}")
    locked_masks = lock["held_out_removed_features_by_seed"]
    locked_hashes = lock["mask_hashes_by_seed"]
    for _, row in chosen.iterrows():
        seed = str(int(row["seed"]))
        raw_removed = row["removed_features"]
        removed = json.loads(raw_removed) if isinstance(raw_removed, str) else list(raw_removed)
        recomputed_hash = hash_json(removed)
        if removed != locked_masks[seed]:
            raise ProtocolError(f"Chosen candidate mask differs from selection lock: {target}/{model}/seed{seed}")
        if str(row["mask_hash"]) != recomputed_hash:
            raise ProtocolError(f"Chosen candidate mask_hash is invalid: {target}/{model}/seed{seed}")
        if recomputed_hash != locked_hashes[seed]:
            raise ProtocolError(f"Chosen candidate mask_hash differs from lock: {target}/{model}/seed{seed}")


def select_one(args: argparse.Namespace, held_out: str, model: str, policy: dict[str, Any]) -> dict[str, Any]:
    development = [target for target in TARGETS if target != held_out]
    # Crucial isolation: no held-out candidate file is opened in this function.
    frame = pd.concat([normalize_metrics(pd.read_csv(candidate_target_path(args, target))) for target in development])
    frame = frame[frame["model"] == model].copy()
    validate_development_cartesian(frame, development, model)
    baseline = frame[frame["filtering_strategy"] == "none"][["target_dataset", "seed", *METRICS]].rename(
        columns={metric: f"baseline_{metric}" for metric in METRICS}
    )
    joined = frame.merge(baseline, on=["target_dataset", "seed"], validate="many_to_one")
    for metric in METRICS:
        joined[f"delta_{metric}"] = joined[metric] - joined[f"baseline_{metric}"]
    seed_direction = joined.groupby(["filtering_strategy", "seed"], as_index=False).agg(
        seed_mcc_delta=("delta_mcc", "mean"),
        seed_binary_f1_delta=("delta_binary_f1", "mean"),
        seed_fnr_delta=("delta_fnr", "mean"),
    )
    summary = joined.groupby("filtering_strategy", as_index=False).agg(
        mean_mcc_delta=("delta_mcc", "mean"),
        mean_binary_f1_delta=("delta_binary_f1", "mean"),
        mean_fnr_delta=("delta_fnr", "mean"),
    )
    direction_summary = seed_direction.groupby("filtering_strategy", as_index=False).agg(
        positive_mcc_seeds=("seed_mcc_delta", lambda values: int((values > 0).sum())),
        positive_binary_f1_seeds=("seed_binary_f1_delta", lambda values: int((values > 0).sum())),
        seed_count=("seed", "nunique"),
    )
    summary = summary.merge(direction_summary, on="filtering_strategy", validate="one_to_one")
    required_development_positive = int(math.ceil(len(SEEDS) * policy["development_required_positive_seed_fraction"]))
    summary["eligible"] = (
        (summary["filtering_strategy"] != "none")
        & (summary["mean_mcc_delta"] >= policy["min_mean_mcc_delta"])
        & (summary["mean_binary_f1_delta"] >= policy["min_mean_binary_f1_delta"])
        & (summary["mean_fnr_delta"] <= policy["max_mean_fnr_delta"])
        & (summary["positive_mcc_seeds"] >= required_development_positive)
        & (summary["positive_binary_f1_seeds"] >= required_development_positive)
        & (summary["seed_count"] == len(SEEDS))
    )
    eligible = summary[summary["eligible"]].copy()
    selected = "none"
    if not eligible.empty:
        eligible["strategy_order"] = eligible["filtering_strategy"].map({name: i for i, name in enumerate(STRATEGIES)})
        ascending = []
        for key in policy["tie_break"]:
            ascending.append(key in {"mean_fnr_delta", "strategy_order"})
        selected = eligible.sort_values(policy["tie_break"], ascending=ascending).iloc[0]["filtering_strategy"]
    held_masks: dict[str, list[str]] = {}
    for seed in SEEDS:
        if selected == "none":
            held_masks[str(seed)] = []
        elif selected.startswith("remove_shift_top"):
            k = 5 if "plus_corr" in selected else int(selected.split("top", 1)[1])
            held_masks[str(seed)] = shift_features(args, held_out, seed, k)
            if "plus_corr" in selected:
                data = load_unsw_splits(seed, "binary")
                held_masks[str(seed)] = masks_for_seed(args, data, seed)[held_out][selected]
                del data
        else:
            data = load_unsw_splits(seed, "binary")
            held_masks[str(seed)] = masks_for_seed(args, data, seed)[held_out][selected]
            del data
    return {
        "protocol_id": PROTOCOL_ID, "held_out_target": held_out, "model": model,
        "development_targets": development, "policy": policy, "policy_hash": hash_file(args.policy),
        "development_deltas": summary.to_dict("records"), "selected_strategy": selected,
        "held_out_removed_features_by_seed": held_masks,
        "mask_hashes_by_seed": {seed: hash_json(mask) for seed, mask in held_masks.items()},
        "development_input_hashes": {target: hash_file(candidate_target_path(args, target)) for target in development},
        "held_out_metrics_loaded": False,
    }


def select_and_lock(args: argparse.Namespace) -> None:
    require_execute_flag(args.execute, "H2 LOTO selection lock")
    policy = read_policy(args.policy)
    validate_runtime_lock(args)
    lock_dir = h2_root(args) / "selection_locks"; lock_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        for model in MODELS:
            lock = select_one(args, target, model, policy)
            write_immutable_json(selection_lock_path(args, target, model), lock)


def random_mask(features: list[str], k: int, target: str, model: str, seed: int, replicate: int) -> tuple[list[str], int]:
    random_seed = deterministic_seed(PROTOCOL_ID, "H2-random", target, model, seed, replicate)
    removed = np.random.default_rng(random_seed).choice(features, size=k, replace=False).tolist()
    return removed, random_seed


def run_random_batch(args: argparse.Namespace) -> None:
    require_execute_flag(args.execute, "H2 matched random controls")
    policy = read_policy(args.policy)
    validate_runtime_lock(args)
    lock_path = selection_lock_path(args, args.target, args.model)
    if not lock_path.exists():
        raise ProtocolError(f"Selection lock must exist before random controls: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    policy_with_path = {**policy, "_path": str(args.policy)}
    validate_selection_lock(lock, lock_path, policy_with_path, args.target, args.model)
    strategy = lock["selected_strategy"]
    batch_dir = h2_root(args) / "random" / "batches" / args.target / args.model / f"seed{args.seed}" / f"rep{args.rep_start:02d}_{args.rep_start + args.rep_count - 1:02d}"
    if strategy == "none":
        completed_marker(batch_dir, {"status": "not_applicable_selected_none", "lock_hash": hash_file(lock_path)})
        return
    data = load_unsw_splits(args.seed, "binary"); validate_feature_order(data["feature_cols"])
    k = len(lock["held_out_removed_features_by_seed"][str(args.seed)])
    rows = []
    with audited_run(batch_dir, {"stage": "H2-random", "target": args.target, "model": args.model,
                                         "seed": args.seed, "rep_start": args.rep_start, "rep_count": args.rep_count}):
        for replicate in range(args.rep_start, args.rep_start + args.rep_count):
            removed, random_seed = random_mask(data["feature_cols"], k, args.target, args.model, args.seed, replicate)
            result = train_and_eval_filtered_model(
                data, args.seed, args.model, "H2_frozen_random", f"random_rep{replicate:02d}",
                removed, [args.target], training_args(args, batch_dir / f"rep{replicate:02d}"),
            )
            row = normalize_metrics(pd.DataFrame(result)).iloc[0].to_dict()
            row.update({"protocol_id": PROTOCOL_ID, "replicate": replicate, "random_seed": random_seed,
                        "mask_hash": hash_json(removed), "selected_mask_size": k, "lock_hash": hash_file(lock_path)})
            rows.append(row)
        pd.DataFrame(rows).to_csv(batch_dir / "random_metrics.csv", index=False)
        completed_marker(batch_dir, {"status": "complete", "rows": len(rows), "lock_hash": hash_file(lock_path)})


def validate_selection_lock(
    lock: dict[str, Any], lock_path: Path, policy: dict[str, Any],
    target: str, model: str,
) -> None:
    expected_masks = {str(seed) for seed in SEEDS}
    if lock.get("protocol_id") != PROTOCOL_ID or lock.get("held_out_target") != target or lock.get("model") != model:
        raise ProtocolError(f"Selection-lock identity mismatch: {lock_path}")
    if lock.get("policy_hash") != hash_file(Path(policy["_path"])):
        raise ProtocolError(f"Selection-lock policy hash mismatch: {lock_path}")
    expected_development = [candidate for candidate in TARGETS if candidate != target]
    locked_development = lock.get("development_targets")
    if (
        not isinstance(locked_development, list)
        or len(locked_development) != len(expected_development)
        or set(locked_development) != set(expected_development)
    ):
        raise ProtocolError(f"Selection-lock development targets mismatch: {lock_path}")
    locked_inputs = lock.get("development_input_hashes")
    if not isinstance(locked_inputs, dict) or set(locked_inputs) != set(expected_development):
        raise ProtocolError(f"Selection-lock development input hash set mismatch: {lock_path}")
    output_root = lock_path.parents[2]
    for development_target in expected_development:
        development_path = output_root / "h2" / "candidates" / "by_target" / f"{development_target}.csv"
        if not development_path.is_file():
            raise ProtocolError(f"Selection-lock development input is missing: {development_path}")
        actual_hash = hash_file(development_path)
        if locked_inputs[development_target] != actual_hash:
            raise ProtocolError(
                f"Selection-lock development input changed: {development_target}: "
                f"locked={locked_inputs[development_target]}, actual={actual_hash}"
            )
    masks = lock.get("held_out_removed_features_by_seed", {})
    hashes = lock.get("mask_hashes_by_seed", {})
    if set(masks) != expected_masks or set(hashes) != expected_masks:
        raise ProtocolError(f"Selection-lock seed coverage mismatch: {lock_path}")
    for seed in expected_masks:
        if hashes[seed] != hash_json(masks[seed]):
            raise ProtocolError(f"Selection-lock mask hash mismatch: {lock_path}, seed={seed}")
    if lock.get("held_out_metrics_loaded") is not False:
        raise ProtocolError(f"Selection lock was not created under held-out isolation: {lock_path}")


def load_complete_random_controls(
    args: argparse.Namespace, target: str, model: str,
    lock: dict[str, Any], lock_path: Path, policy: dict[str, Any],
) -> pd.DataFrame:
    paths = list((h2_root(args) / "random" / "batches" / target / model).glob("seed*/rep*/random_metrics.csv"))
    if lock["selected_strategy"] == "none":
        if paths:
            raise ProtocolError(f"Random metrics exist although selected strategy is none: {target}/{model}")
        return pd.DataFrame()
    if not paths:
        raise ProtocolError(f"No random controls found for selected non-empty mask: {target}/{model}")
    random = normalize_metrics(pd.concat([pd.read_csv(path) for path in paths], ignore_index=True))
    required_columns = {
        "protocol_id", "replicate", "random_seed", "mask_hash", "removed_features",
        "selected_mask_size", "lock_hash", "seed", "model", "target_dataset",
    }
    missing = sorted(required_columns - set(random.columns))
    if missing:
        raise ProtocolError(f"Random controls lack audit columns for {target}/{model}: {missing}")
    expected_lock_hash = hash_file(lock_path)
    expected_replicates = set(range(policy["random_replicates_per_seed"]))
    all_keys: list[tuple[int, int]] = []
    for seed in SEEDS:
        group = random[random["seed"].astype(int) == seed].copy()
        actual_replicates = set(group["replicate"].astype(int))
        if len(group) != len(expected_replicates) or actual_replicates != expected_replicates:
            raise ProtocolError(
                f"Random replicate coverage mismatch for {target}/{model}/seed{seed}: "
                f"rows={len(group)}, replicates={sorted(actual_replicates)}"
            )
        if group["replicate"].astype(int).duplicated().any():
            raise ProtocolError(f"Duplicate/overlapping random replicate for {target}/{model}/seed{seed}")
        expected_k = len(lock["held_out_removed_features_by_seed"][str(seed)])
        for _, row in group.iterrows():
            removed = json.loads(row["removed_features"])
            if len(removed) != expected_k or int(row["selected_mask_size"]) != expected_k:
                raise ProtocolError(f"Random mask size mismatch for {target}/{model}/seed{seed}")
            if row["mask_hash"] != hash_json(removed):
                raise ProtocolError(f"Random mask hash mismatch for {target}/{model}/seed{seed}")
        if set(group["lock_hash"]) != {expected_lock_hash}:
            raise ProtocolError(f"Random lock hash mismatch for {target}/{model}/seed{seed}")
        if set(group["protocol_id"]) != {PROTOCOL_ID} or set(group["model"]) != {model} or set(group["target_dataset"]) != {target}:
            raise ProtocolError(f"Random-control identity mismatch for {target}/{model}/seed{seed}")
        if group["random_seed"].astype(int).nunique() != len(expected_replicates):
            raise ProtocolError(f"Random seeds are not unique for {target}/{model}/seed{seed}")
        all_keys.extend((seed, replicate) for replicate in actual_replicates)
    if len(all_keys) != len(set(all_keys)):
        raise ProtocolError(f"Overlapping random batches detected for {target}/{model}")
    return random


def h2_comparator_pass(
    group: pd.DataFrame, suffix: str, policy: dict[str, Any],
) -> bool:
    return bool(
        len(group) == policy["final_required_seed_count"]
        and group[f"delta_mcc_{suffix}"].mean() >= policy["min_mean_mcc_delta"]
        and group[f"delta_binary_f1_{suffix}"].mean() >= policy["min_mean_binary_f1_delta"]
        and group[f"delta_fnr_{suffix}"].mean() <= policy["max_mean_fnr_delta"]
        and int((group[f"delta_mcc_{suffix}"] > 0).sum()) >= policy["final_required_positive_seeds"]
        and int((group[f"delta_binary_f1_{suffix}"] > 0).sum()) >= policy["final_required_positive_seeds"]
    )


def analyze(args: argparse.Namespace) -> None:
    require_execute_flag(args.execute, "H2 analysis writing")
    policy = read_policy(args.policy)
    validate_runtime_lock(args)
    analysis_dir = h2_root(args) / "analysis"; analysis_dir.mkdir(parents=True, exist_ok=True)
    selected_rows, random_rows = [], []
    policy = dict(policy)
    policy["_path"] = str(args.policy)
    for target in TARGETS:
        held = normalize_metrics(pd.read_csv(candidate_target_path(args, target)))
        for model in MODELS:
            lock_path = selection_lock_path(args, target, model)
            if not lock_path.exists():
                raise ProtocolError(f"Missing selection lock: {lock_path}")
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            validate_selection_lock(lock, lock_path, policy, target, model)
            strategy = lock["selected_strategy"]
            chosen = held[(held["model"] == model) & (held["filtering_strategy"] == strategy)].copy()
            baseline = held[(held["model"] == model) & (held["filtering_strategy"] == "none")].copy()
            if set(chosen["seed"].astype(int)) != set(SEEDS) or len(chosen) != len(SEEDS):
                raise ProtocolError(f"Selected-candidate coverage mismatch: {target}/{model}/{strategy}")
            validate_chosen_masks(chosen, lock, target, model)
            if set(baseline["seed"].astype(int)) != set(SEEDS) or len(baseline) != len(SEEDS):
                raise ProtocolError(f"E2 coverage mismatch: {target}/{model}")
            joined = chosen.merge(baseline[["seed", *METRICS]], on="seed", suffixes=("", "_e2"), validate="one_to_one")
            joined["selected_strategy"] = strategy; joined["lock_hash"] = hash_file(lock_path)
            for metric in METRICS:
                joined[f"delta_{metric}_vs_e2"] = joined[metric] - joined[f"{metric}_e2"]
            selected_rows.extend(joined.to_dict("records"))
            random = load_complete_random_controls(args, target, model, lock, lock_path, policy)
            if not random.empty:
                medians = random.groupby("seed", as_index=False)[METRICS].median()
                compare = joined.merge(medians, on="seed", suffixes=("_selected", "_random_median"), validate="one_to_one")
                for metric in METRICS:
                    compare[f"delta_{metric}_vs_random_median"] = compare[f"{metric}_selected"] - compare[f"{metric}_random_median"]
                random_rows.extend(compare.to_dict("records"))
    selected_df = pd.DataFrame(selected_rows); random_df = pd.DataFrame(random_rows)
    selected_df.to_csv(analysis_dir / "loto_selected_vs_e2_seedwise.csv", index=False)
    random_df.to_csv(analysis_dir / "loto_selected_vs_random_seedwise.csv", index=False)
    bootstrap_rows = []
    for label, frame, suffix in [
        ("e2", selected_df, "vs_e2"), ("random_median", random_df, "vs_random_median")
    ]:
        if frame.empty:
            continue
        for (target, model), group in frame.groupby(["target_dataset", "model"]):
            for metric in METRICS:
                column = f"delta_{metric}_{suffix}"
                bootstrap_rows.append({"target": target, "model": model, "comparator": label, "metric": metric,
                                       **paired_bootstrap(group[column].to_numpy(), seed=deterministic_seed(target, model, label, metric))})
    pd.DataFrame(bootstrap_rows).to_csv(analysis_dir / "bootstrap_10000.csv", index=False)
    # Final mechanical verdict intentionally requires the same frozen policy used by selection.
    verdicts = []
    for (target, model), group in selected_df.groupby(["target_dataset", "model"]):
        random_group = (
            random_df[(random_df["target_dataset"] == target) & (random_df["model"] == model)]
            if not random_df.empty else pd.DataFrame()
        )
        strategy = str(group["selected_strategy"].iloc[0])
        passed_e2 = strategy != "none" and h2_comparator_pass(group, "vs_e2", policy)
        passed_random = strategy != "none" and not random_group.empty and h2_comparator_pass(
            random_group, "vs_random_median", policy,
        )
        verdicts.append({"target": target, "model": model,
                         "verdict": "shift_guided_repair" if passed_e2 and passed_random else "regularization_or_feature_sensitivity" if passed_e2 else "not_supported",
                         "beats_e2": passed_e2, "beats_random_median": passed_random, "policy_hash": hash_file(args.policy)})
    pd.DataFrame(verdicts).to_csv(analysis_dir / "verdict_by_target_model.csv", index=False)
