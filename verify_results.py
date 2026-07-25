"""Verify the minimal release inventory and principal manuscript values."""

from __future__ import annotations

import csv
import hashlib
import json
import math
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
    paths = sorted((policy_path.parent / "selection_locks").glob("*.json"))
    if len(paths) != 6:
        raise AssertionError(f"Expected six H2 selection locks, found {len(paths)}")
    for path in paths:
        lock = json.loads(path.read_text(encoding="utf-8"))
        expected_name = (
            f"{target_slugs[lock['held_out_target']]}__"
            f"{model_slugs[lock['model']]}.json"
        )
        if path.name != expected_name:
            raise AssertionError(f"Unexpected H2 lock name: {path.name}")
        if lock.get("policy_hash") != policy_hash or lock.get("policy") != policy:
            raise AssertionError(f"H2 lock policy mismatch: {path.name}")


def check_release_manifest() -> None:
    manifest = table("release_manifest_sha256.csv")
    expected_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.name != "release_manifest_sha256.csv"
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
    }
    recorded_paths = {row["path"] for row in manifest}
    if recorded_paths != expected_paths:
        missing = sorted(expected_paths - recorded_paths)
        extra = sorted(recorded_paths - expected_paths)
        raise AssertionError(
            f"Release manifest inventory mismatch; missing={missing}, extra={extra}"
        )
    for row in manifest:
        path = ROOT / row["path"]
        if int(row["bytes"]) != path.stat().st_size or row["sha256"] != sha256(path):
            raise AssertionError(f"Release manifest hash mismatch: {row['path']}")


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


def main() -> int:
    checks = {
        "inventory": check_inventory,
        "split hashes": check_split_hashes,
        "model construction": check_model_construction,
        "H2 selection locks": check_h2_selection_locks,
        "release manifest": check_release_manifest,
        "H1": check_h1,
        "H2": check_h2,
        "H3": check_h3,
        "Oracle": check_oracle,
        "Calibration": check_calibration,
    }
    for name, check in checks.items():
        check()
        print(f"PASS {name}")
    print("PASS minimal reproducibility package")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
