"""Verify the minimal release inventory and principal manuscript values."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def table(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def one(records: list[dict[str, str]], **keys: object) -> dict[str, str]:
    selected = [
        row
        for row in records
        if all(str(row.get(key)) == str(value) for key, value in keys.items())
    ]
    if len(selected) != 1:
        raise AssertionError(f"Expected one row for {keys}, found {len(selected)}")
    return selected[0]


def close(actual: object, expected: float, tolerance: float = 5e-5) -> None:
    value = float(actual)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{value} != {expected} within {tolerance}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_inventory() -> None:
    required = [
        "run_h1.py",
        "run_h2.py",
        "run_h3.py",
        "run_oracle.py",
        "run_calibration.py",
        "data/dataset_checksums.csv",
        "data/feature_schema.csv",
        "data/model_hyperparameters.json",
        "results/reference/h1/delta_summary.csv",
        "results/reference/h2/verdict.csv",
        "results/reference/h3/verdict.csv",
        "results/reference/oracle/summary.csv",
        "results/reference/calibration/summary.csv",
        "figures/scripts/plot_h2_loto_results.py",
        "figures/scripts/plot_h3_paired_domain_loss.py",
        "figures/reference/figure_2.pdf",
        "figures/reference/figure_3.pdf",
        "src/cross_domain_nids/model_factory.py",
        "experiments/h1/run_training.py",
        "experiments/h2/run_loto_filtering.py",
        "experiments/h3/run_paired_domain_loss.py",
        "experiments/oracle/run_target_oracle.py",
        "protocols/h1/training_policy.json",
        "protocols/h2/selection_policy.json",
        "protocols/h3/verdict_policy.json",
        "protocols/oracle/split_policy.json",
        "data/reference_inputs/h1_historical_results.csv",
        "data/reference_inputs/oracle_historical_results.csv",
        ".github/workflows/verify.yml",
        "release_manifest_sha256.csv",
    ]
    missing = [item for item in required if not (ROOT / item).is_file()]
    if missing:
        raise AssertionError(f"Missing required files: {missing}")

    h1_maps = list(
        (ROOT / "data/splits/h1").glob("split_seed_*.npz")
    )
    oracle_maps = list((ROOT / "data/splits/oracle").glob("*.npz"))
    locks = list((ROOT / "protocols/h2/selection_locks").glob("*.json"))
    if (len(h1_maps), len(oracle_maps), len(locks)) != (3, 9, 6):
        raise AssertionError(
            "Expected 3 H1 maps, 9 Oracle maps, and 6 H2 locks; "
            f"found {len(h1_maps)}, {len(oracle_maps)}, {len(locks)}"
        )


def check_split_hashes() -> None:
    h1 = {
        "split_seed_42.npz": "eb36fd67cc1d482dde69191ea44d56fbbc618b4695d39f452bab3bfdfe84041c",
        "split_seed_123.npz": "20e8adb47f209fafdbfe3f64cb52ef05acf29d19fb4dc9d047e4249323350738",
        "split_seed_2024.npz": "28e1c2a3c7fe603b305e339eedc9b4fc5f68b7d96750bd139821370bc67c2154",
    }
    root = ROOT / "data/splits/h1"
    for name, expected in h1.items():
        actual = sha256(root / name)
        if actual != expected:
            raise AssertionError(f"H1 split hash mismatch: {name}")

    manifest = table("data/splits/oracle/manifest.csv")
    for row in manifest:
        manifest_path = Path(row["split_map_path"])
        metadata_path = Path(row["metadata_path"])
        if manifest_path.is_absolute() or metadata_path.is_absolute():
            raise AssertionError("Oracle manifest contains an absolute path")
        path = ROOT / manifest_path
        if sha256(path) != row["split_map_sha256"]:
            raise AssertionError(f"Oracle split hash mismatch: {path.name}")
        if sha256(ROOT / metadata_path) != row["metadata_sha256"]:
            raise AssertionError(f"Oracle metadata hash mismatch: {metadata_path.name}")


def check_model_construction() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from cross_domain_nids.model_factory import (
        constructor_signature,
        load_parameter_record,
    )

    expected_classes = {
        "RandomForest": "sklearn.ensemble._forest.RandomForestClassifier",
        "XGBoost": "cross_domain_nids.models.pipeline_models.LabelEncodedXGBClassifier",
        "MLP": "cross_domain_nids.models.pipeline_models.TorchMLPClassifier",
    }
    for experiment in ("E1", "E2"):
        for model, expected_class in expected_classes.items():
            signature = constructor_signature(experiment, model, 123)
            if signature["class"] != expected_class:
                raise AssertionError(
                    f"Constructor class mismatch for {experiment}/{model}: "
                    f"{signature['class']}"
                )
            if signature["parameters"]["random_state"] != 123:
                raise AssertionError(f"Constructor seed mismatch: {experiment}/{model}")
            if "verbose" in signature["parameters"] and signature["parameters"]["verbose"]:
                raise AssertionError(f"Constructor verbosity mismatch: {experiment}/{model}")
            expected = dict(load_parameter_record(experiment, model)["parameters"])
            expected["random_state"] = 123
            if "verbose" in expected:
                expected["verbose"] = False
            for key, value in expected.items():
                if signature["parameters"].get(key) != value:
                    raise AssertionError(
                        f"Constructor parameter mismatch for "
                        f"{experiment}/{model}/{key}"
                    )


def check_h2_selection_locks() -> None:
    policy_path = ROOT / "protocols" / "h2" / "selection_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_hash = sha256(policy_path)
    target_slugs = {
        "CICIDS2018": "cicids2018",
        "BoT-IoT": "bot_iot",
        "ToN-IoT": "ton_iot",
    }
    model_slugs = {"MLP": "mlp", "RandomForest": "random_forest"}
    strategy_order = [
        "none",
        "remove_shift_top3",
        "remove_shift_top5",
        "remove_shift_top10",
        "remove_shift_top15",
        "remove_exact_duplicates",
        "remove_corr_gt_0.98",
        "remove_shift_top5_plus_corr_gt_0.98",
    ]
    strategies = set(strategy_order)
    paths = sorted((policy_path.parent / "selection_locks").glob("*.json"))
    if len(paths) != 6:
        raise AssertionError(f"Expected six H2 selection locks, found {len(paths)}")
    expected_combinations = {
        (target, model)
        for target in policy["targets"]
        for model in policy["models"]
    }
    actual_combinations: set[tuple[str, str]] = set()
    for path in paths:
        lock = json.loads(path.read_text(encoding="utf-8"))
        combination = (lock.get("held_out_target"), lock.get("model"))
        if combination not in expected_combinations or combination in actual_combinations:
            raise AssertionError(f"Duplicate or unknown H2 lock identity: {path.name}")
        actual_combinations.add(combination)
        expected_name = (
            f"{target_slugs[lock['held_out_target']]}__"
            f"{model_slugs[lock['model']]}.json"
        )
        if path.name != expected_name:
            raise AssertionError(f"Unexpected H2 lock name: {path.name}")
        if lock.get("policy_hash") != policy_hash or lock.get("policy") != policy:
            raise AssertionError(f"H2 lock policy mismatch: {path.name}")
        development_targets = lock.get("development_targets")
        expected_development = [
            target for target in policy["targets"]
            if target != lock["held_out_target"]
        ]
        if development_targets != expected_development:
            raise AssertionError(f"H2 development-target mismatch: {path.name}")
        if (
            lock["held_out_target"] in development_targets
            or set(lock.get("development_input_hashes", {})) != set(development_targets)
            or lock.get("held_out_metrics_loaded") is not False
        ):
            raise AssertionError(f"H2 held-out leakage invariant failed: {path.name}")
        deltas = lock.get("development_deltas")
        if not isinstance(deltas, list) or len(deltas) != len(strategies):
            raise AssertionError(f"H2 candidate coverage mismatch: {path.name}")
        candidate_names = [row.get("filtering_strategy") for row in deltas]
        if set(candidate_names) != strategies or len(candidate_names) != len(set(candidate_names)):
            raise AssertionError(f"H2 duplicate or missing candidate: {path.name}")
        required_positive = math.ceil(
            len(policy["seeds"])
            * policy["development_required_positive_seed_fraction"]
        )
        eligible_rows = []
        for row in deltas:
            expected_eligible = (
                row["filtering_strategy"] != "none"
                and row["mean_mcc_delta"] >= policy["min_mean_mcc_delta"]
                and row["mean_binary_f1_delta"]
                >= policy["min_mean_binary_f1_delta"]
                and row["mean_fnr_delta"] <= policy["max_mean_fnr_delta"]
                and row["positive_mcc_seeds"] >= required_positive
                and row["positive_binary_f1_seeds"] >= required_positive
                and row["seed_count"] == len(policy["seeds"])
            )
            if row.get("eligible") is not expected_eligible:
                raise AssertionError(f"H2 candidate eligibility mismatch: {path.name}")
            if expected_eligible:
                eligible_rows.append(row)
        selected = lock.get("selected_strategy")
        if not eligible_rows:
            if selected != "none":
                raise AssertionError(f"H2 abstention mismatch: {path.name}")
        else:
            order = {name: index for index, name in enumerate(strategy_order)}
            for row in eligible_rows:
                row["strategy_order"] = order[row["filtering_strategy"]]
            ascending = [
                key in {"mean_fnr_delta", "strategy_order"}
                for key in policy["tie_break"]
            ]
            ranked = sorted(
                eligible_rows,
                key=lambda row: tuple(
                    row[key] if asc else -row[key]
                    for key, asc in zip(policy["tie_break"], ascending)
                ),
            )
            if selected != ranked[0]["filtering_strategy"]:
                raise AssertionError(f"H2 selected candidate mismatch: {path.name}")
        seed_keys = {str(seed) for seed in policy["seeds"]}
        masks = lock.get("held_out_removed_features_by_seed", {})
        mask_hashes = lock.get("mask_hashes_by_seed", {})
        if set(masks) != seed_keys or set(mask_hashes) != seed_keys:
            raise AssertionError(f"H2 seed-mask coverage mismatch: {path.name}")
        sys.path.insert(0, str(ROOT / "src"))
        from cross_domain_nids.protocol_utils import hash_json

        for seed in seed_keys:
            if mask_hashes[seed] != hash_json(masks[seed]):
                raise AssertionError(f"H2 mask hash mismatch: {path.name}/seed{seed}")
            if selected == "none" and masks[seed] != []:
                raise AssertionError(f"H2 abstention mask is non-empty: {path.name}")
    if actual_combinations != expected_combinations:
        raise AssertionError("H2 target-model lock coverage is incomplete")
    if policy["random_replicates_per_seed"] != 20:
        raise AssertionError("H2 random-control replicate count is not frozen at 20")
    if sha256(ROOT / "data" / "model_hyperparameters.json") != policy[
        "model_parameters_sha256"
    ]:
        raise AssertionError("H2 model-parameter hash mismatch")
    if sha256(ROOT / "src" / "cross_domain_nids" / "model_factory.py") != policy[
        "model_factory_sha256"
    ]:
        raise AssertionError("H2 model-factory hash mismatch")
    if sha256(
        ROOT / "src" / "cross_domain_nids" / "models" / "pipeline_models.py"
    ) != policy["pipeline_models_sha256"]:
        raise AssertionError("H2 pipeline-model hash mismatch")
    if sha256(ROOT / "src" / "cross_domain_nids" / "determinism.py") != policy[
        "determinism_sha256"
    ]:
        raise AssertionError("H2 determinism hash mismatch")


def check_release_manifest() -> None:
    manifest = table("release_manifest_sha256.csv")
    recorded_paths = [row["path"] for row in manifest]
    if len(recorded_paths) != len(set(recorded_paths)):
        raise AssertionError("Release manifest contains duplicate paths")
    for row in manifest:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise AssertionError(f"Unsafe release manifest path: {row['path']}")
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"Release manifest file is missing: {row['path']}")
        if int(row["bytes"]) != path.stat().st_size or row["sha256"] != sha256(path):
            raise AssertionError(f"Release manifest hash mismatch: {row['path']}")


def check_git_inventory() -> None:
    """CI-only check that every tracked release file is represented."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    tracked = {
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    }
    manifest_paths = {row["path"] for row in table("release_manifest_sha256.csv")}
    expected = tracked - {"release_manifest_sha256.csv"}
    if manifest_paths != expected:
        missing = sorted(expected - manifest_paths)
        extra = sorted(manifest_paths - expected)
        raise AssertionError(
            f"Git/release manifest mismatch; missing={missing}, extra={extra}"
        )


def check_h1() -> None:
    frame = table("results/reference/h1/delta_summary.csv")
    for (model, target), expected in {
        ("XGBoost", "BoT-IoT"): 0.7127,
        ("XGBoost", "ToN-IoT"): 0.5149,
        ("RandomForest", "ToN-IoT"): 0.0144,
    }.items():
        close(one(frame, model=model, target_dataset=target)["corrected_mean_delta_binary_f1"], expected)


def check_h2() -> None:
    frame = table("results/reference/h2/verdict.csv")
    if len(frame) != 6 or any(row["verdict"] != "not_supported" for row in frame):
        raise AssertionError("H2 verdict table does not match the six frozen strata")
    if one(frame, target="ToN-IoT", model="RandomForest")["beats_random_median"].lower() != "true":
        raise AssertionError("Expected ToN-IoT/RandomForest to beat random median only")


def check_h3() -> None:
    frame = table("results/reference/h3/verdict.csv")
    for target, values in {
        "BoT-IoT": (-0.0049, -0.0375, 0.0199),
        "CICIDS2018": (-0.0586, -0.0957, 0.2570),
        "ToN-IoT": (-0.4422, -0.5336, 0.5602),
    }.items():
        row = one(frame, target=target)
        if row["verdict"] != "Fail":
            raise AssertionError(f"H3 {target} verdict is not Fail")
        for column, expected in zip(
            ("delta_MCC", "delta_Attack-class Binary-F1", "delta_FNR"), values
        ):
            close(row[column], expected)


def check_oracle() -> None:
    frame = table("results/reference/oracle/summary.csv")
    for target, values in {
        "CICIDS2018": (0.9802, 0.9780),
        "BoT-IoT": (0.9992, 0.7017),
        "ToN-IoT": (0.9349, 0.8914),
    }.items():
        row = one(frame, model="XGBoost", target_dataset=target)
        close(row["binary_f1_mean"], values[0])
        close(row["mcc_mean"], values[1])


def check_calibration() -> None:
    row = one(
        table("results/reference/calibration/summary.csv"),
        experiment="E2",
        model="XGBoost",
        target_dataset="CICIDS2018",
        filtering_strategy="none",
    )
    close(row["ece_mean"], 0.4394)
    close(row["brier_score_mean"], 0.4007)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-git-inventory",
        action="store_true",
        help="CI-only: compare the manifest inventory with git ls-files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = {
        "inventory": check_inventory,
        "split hashes": check_split_hashes,
        "model construction": check_model_construction,
        "H2 selection locks": check_h2_selection_locks,
        "release manifest": check_release_manifest,
        "H1": check_h1,
        "historical H2": check_h2,
        "historical H3": check_h3,
        "Oracle": check_oracle,
        "Calibration": check_calibration,
    }
    for name, check in checks.items():
        check()
        print(f"PASS {name}")
    from experiments.revision.common import verify as verify_inputs
    from experiments.revision.evidence import verify as verify_revision
    verify_inputs(raw=False)
    revision = verify_revision()
    print(f"PASS current revision: {revision['checks']} numerical checks")
    if args.check_git_inventory:
        check_git_inventory()
        print("PASS Git tracked-file inventory")
    print("PASS minimal reproducibility package")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
