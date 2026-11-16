"""Frozen LOTO selection and verdicts applied to corrected-split candidates."""
from experiments.h2.run_loto_filtering import *
from .common import PROTOCOL_ID, source_data, FEATURES
from experiments.revision.matrix import mask as frozen_random_mask

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
    held_masks = {}
    input_hashes = {}
    for seed in SEEDS:
        mask_path = h2_root(args) / 'masks' / f'seed{seed}.json'
        held_masks[str(seed)] = json.loads(mask_path.read_text())[held_out][selected]
        input_hashes[str(mask_path.relative_to(args.output_root))] = hash_file(mask_path)
        rank_path = ranking_path(args, held_out, seed)
        input_hashes[str(rank_path.relative_to(args.output_root))] = hash_file(rank_path)
    return {
        "protocol_id": PROTOCOL_ID, "held_out_target": held_out, "model": model,
        "development_targets": development, "policy": policy, "policy_hash": hash_file(args.policy),
        "development_deltas": summary.to_dict("records"), "selected_strategy": selected,
        "held_out_removed_features_by_seed": held_masks,
        "mask_hashes_by_seed": {seed: hash_json(mask) for seed, mask in held_masks.items()},
        "development_input_hashes": {target: hash_file(candidate_target_path(args, target)) for target in development},
        "held_out_metrics_loaded": False, "selection_feature_input_hashes": input_hashes,
    }

def select_and_lock(args: argparse.Namespace) -> None:
    require_execute_flag(args.execute, "H2 LOTO selection lock")
    policy = json.loads(args.policy.read_text())
    # The CLI verifies data and locks inputs before dispatch.
    lock_dir = h2_root(args) / "selection_locks"; lock_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        for model in MODELS:
            lock = select_one(args, target, model, policy)
            write_immutable_json(lock_dir / f"{target}_{model}.json", lock)

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
    inputs = lock.get('selection_feature_input_hashes', {})
    if len(inputs) != 6:
        raise ProtocolError('Missing frozen ranking/mask hashes')
    for name, expected in inputs.items():
        if hash_file(output_root / name) != expected:
            raise ProtocolError(f'Selection feature input changed: {name}')
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
    if set(random['seed'].astype(int)) != set(SEEDS):
        raise ProtocolError('Unexpected random-control seed')
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
            replicate = int(row['replicate'])
            if removed != frozen_random_mask(FEATURES, seed, expected_k, replicate):
                raise ProtocolError('Random mask differs from the frozen shared mask generator')
            if int(row['random_seed']) != deterministic_seed(PROTOCOL_ID, 'random_mask', seed, expected_k, replicate):
                raise ProtocolError('Random mask seed differs from the frozen design')
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

def analyze(args: argparse.Namespace) -> None:
    require_execute_flag(args.execute, "H2 analysis writing")
    policy = json.loads(args.policy.read_text())
    # The CLI verifies data and locks inputs before dispatch.
    analysis_dir = h2_root(args) / "analysis"; analysis_dir.mkdir(parents=True, exist_ok=True)
    selected_rows, random_rows = [], []
    policy = dict(policy)
    policy["_path"] = str(args.policy)
    for target in TARGETS:
        held = normalize_metrics(pd.read_csv(candidate_target_path(args, target)))
        for model in MODELS:
            lock_path = h2_root(args) / "selection_locks" / f"{target}_{model}.json"
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
