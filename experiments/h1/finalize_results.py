"""Validate, summarize, and mechanically adjudicate H1 corrected-split results."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
RUN = REPOSITORY_ROOT / "results" / "generated" / "h1"
PROTOCOL_ID = "NIDS-H1-CORRECTED-SPLIT-SENS-20260717-v1.1"
POLICY_SHA = "9d73ea94237dd093b491e8313ee6dec6d8ca877ed9525eb71d611324f52fb055"
SEEDS = [42, 123, 2024]
MODELS = ["RandomForest", "XGBoost", "MLP"]
TARGETS = ["CICIDS2018", "BoT-IoT", "ToN-IoT"]
TARGET_ROWS = {"CICIDS2018": 20_115_529, "BoT-IoT": 16_933_808, "ToN-IoT": 27_520_260}
HISTORICAL_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "reference_inputs"
    / "h1_historical_results.csv"
)
HISTORICAL_SHA = "d69852b5eee9971360c9949ded76a5aec0b28c5e6308dd3974d496a8a6768c93"
METRICS = ["binary_f1", "mcc", "fpr", "fnr", "weighted_f1"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def independent_metrics(row: pd.Series) -> dict[str, float]:
    tp, tn, fp, fn = (int(row[key]) for key in ("tp", "tn", "fp", "fn"))
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    attack_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / (denom**0.5) if denom else 0.0
    benign_precision = tn / (tn + fn) if tn + fn else 0.0
    benign_recall = tn / (tn + fp) if tn + fp else 0.0
    benign_f1 = 2 * benign_precision * benign_recall / (benign_precision + benign_recall) if benign_precision + benign_recall else 0.0
    weighted = (((tp + fn) * attack_f1) + ((tn + fp) * benign_f1)) / n if n else 0.0
    return {"binary_f1": attack_f1, "mcc": mcc, "fpr": fpr, "fnr": fnr, "weighted_f1": weighted, "n_samples": n}


def paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    targets = frame[frame["target_dataset"].isin(TARGETS)].copy()
    index = ["seed", "model", "target_dataset"]
    pieces = []
    for metric in METRICS:
        pivot = targets.pivot(index=index, columns="experiment", values=metric)
        if set(pivot.columns) != {"E1", "E2"}:
            raise RuntimeError(f"Missing E1/E2 pair for {metric}")
        piece = pivot.rename(columns={"E1": f"E1_{metric}", "E2": f"E2_{metric}"})
        piece[f"delta_{metric}"] = piece[f"E2_{metric}"] - piece[f"E1_{metric}"]
        pieces.append(piece)
    out = pd.concat(pieces, axis=1).reset_index()
    return out.loc[:, ~out.columns.duplicated()].sort_values(index).reset_index(drop=True)


def delta_summary(deltas: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    for (model, target), group in deltas.groupby(["model", "target_dataset"], sort=False):
        row: dict[str, object] = {"model": model, "target_dataset": target, "n_seeds": len(group)}
        for metric in METRICS:
            values = group[f"delta_{metric}"].astype(float)
            row[f"{prefix}_mean_delta_{metric}"] = values.mean()
            row[f"{prefix}_sd_delta_{metric}"] = values.std(ddof=1)
            row[f"{prefix}_min_delta_{metric}"] = values.min()
            row[f"{prefix}_max_delta_{metric}"] = values.max()
            row[f"{prefix}_positive_seeds_delta_{metric}"] = int((values > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "target_dataset"]).reset_index(drop=True)


def add_check(checks: list[dict[str, object]], category: str, check: str, passed: bool, detail: str) -> None:
    checks.append({"category": category, "check": check, "status": "Pass" if passed else "Blocker", "detail": detail})


def main() -> int:
    checks: list[dict[str, object]] = []
    result_frames = []
    validation_frames = []
    manifest_frames = []

    for seed in SEEDS:
        completion_path = RUN / f"seed{seed}_complete.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        add_check(checks, "completion", f"seed{seed}_protocol", completion["protocol_id"] == PROTOCOL_ID, completion["protocol_id"])
        add_check(
            checks,
            "completion",
            f"seed{seed}_freeze",
            completion["policy_sha256"] == POLICY_SHA,
            completion["policy_sha256"],
        )
        for key, expected_rows in (("results", 24), ("validation", 6), ("training_manifest", 6)):
            path = Path(completion[f"{key}_path"])
            actual_sha = sha256_file(path)
            add_check(checks, "completion", f"seed{seed}_{key}_sha", actual_sha == completion[f"{key}_sha256"], actual_sha)
            frame = pd.read_csv(path)
            add_check(checks, "coverage", f"seed{seed}_{key}_rows", len(frame) == expected_rows, str(len(frame)))
            if key == "results":
                result_frames.append(frame)
            elif key == "validation":
                validation_frames.append(frame)
            else:
                manifest_frames.append(frame)
        log_lines = (RUN / f"execution_seed{seed}.jsonl").read_text(encoding="utf-8").strip().splitlines()
        final_event = json.loads(log_lines[-1])
        add_check(checks, "completion", f"seed{seed}_final_event", final_event.get("event") == "execution_complete", final_event.get("event", "missing"))

    results = pd.concat(result_frames, ignore_index=True)
    validation = pd.concat(validation_frames, ignore_index=True)
    manifests = pd.concat(manifest_frames, ignore_index=True)
    results.to_csv(RUN / "all_results_long.csv", index=False)
    validation.to_csv(RUN / "all_source_validation_diagnostic.csv", index=False)
    manifests.to_csv(RUN / "all_training_manifest.csv", index=False)

    add_check(checks, "coverage", "total_result_rows", len(results) == 72, str(len(results)))
    add_check(checks, "coverage", "full_target_rows", int(results["evaluation_scope"].eq("full_target").sum()) == 54, str(int(results["evaluation_scope"].eq("full_target").sum())))
    add_check(checks, "coverage", "source_test_rows", int(results["evaluation_scope"].eq("corrected_source_test").sum()) == 18, str(int(results["evaluation_scope"].eq("corrected_source_test").sum())))
    add_check(checks, "coverage", "unique_result_keys", not results.duplicated(["experiment", "seed", "model", "target_dataset"]).any(), str(int(results.duplicated(["experiment", "seed", "model", "target_dataset"]).sum())))
    expected_keys = pd.MultiIndex.from_product([["E1", "E2"], SEEDS, MODELS, ["UNSW", *TARGETS]], names=["experiment", "seed", "model", "target_dataset"])
    actual_keys = pd.MultiIndex.from_frame(results[["experiment", "seed", "model", "target_dataset"]])
    add_check(checks, "coverage", "exact_result_grid", set(actual_keys) == set(expected_keys), f"actual={len(set(actual_keys))}; expected={len(set(expected_keys))}")
    add_check(checks, "coverage", "validation_rows", len(validation) == 18, str(len(validation)))
    add_check(checks, "coverage", "training_runs", len(manifests) == 18 and not manifests.duplicated(["experiment", "seed", "model"]).any(), str(len(manifests)))

    for target, n_rows in TARGET_ROWS.items():
        subset = results[results["target_dataset"].eq(target)]
        add_check(checks, "coverage", f"{target}_rows_per_result", len(subset) == 18 and subset["n_samples"].eq(n_rows).all(), f"result_rows={len(subset)}; n_samples={sorted(subset['n_samples'].unique().tolist())}")
    metric_failures = []
    for index, row in results.iterrows():
        reference = independent_metrics(row)
        for metric in [*METRICS, "n_samples"]:
            if not np.isclose(float(row[metric]), float(reference[metric]), rtol=0, atol=1e-12):
                metric_failures.append(f"row{index}:{metric}")
    add_check(checks, "metrics", "independent_confusion_count_recalculation", not metric_failures, ";".join(metric_failures[:10]) if metric_failures else "72 rows x 6 quantities")

    artifact_failures = []
    for _, row in manifests.iterrows():
        artifact = Path(row["artifact"])
        actual_sha = sha256_file(artifact)
        if actual_sha != row["artifact_sha256"]:
            artifact_failures.append(str(artifact))
        if int(row["effective_random_state"]) != int(row["seed"]):
            artifact_failures.append(f"seed:{row['experiment']}/{row['seed']}/{row['model']}")
    add_check(checks, "artifacts", "all_18_artifact_hashes_and_seeds", not artifact_failures, ";".join(artifact_failures[:10]) if artifact_failures else "18/18")
    add_check(
        checks,
        "artifacts",
        "portable_runner_present",
        (ROOT / "train.py").is_file(),
        sha256_file(ROOT / "train.py"),
    )
    add_check(
        checks,
        "artifacts",
        "released_parameter_construction",
        manifests["model_construction"].eq("released_parameter_json").all(),
        f"{int(manifests['model_construction'].eq('released_parameter_json').sum())}/18",
    )

    corrected = paired_deltas(results)
    corrected.to_csv(RUN / "paired_deltas.csv", index=False)
    corrected_summary = delta_summary(corrected, "corrected")
    corrected_summary.to_csv(RUN / "delta_summary.csv", index=False)

    add_check(checks, "historical", "historical_results_hash", sha256_file(HISTORICAL_PATH) == HISTORICAL_SHA, sha256_file(HISTORICAL_PATH))
    historical = pd.read_csv(HISTORICAL_PATH)
    historical = historical[
        historical["experiment"].isin(["E1", "E2"])
        & historical["seed"].isin(SEEDS)
        & historical["model"].isin(MODELS)
        & historical["target_dataset"].isin(TARGETS)
    ].copy()
    historical_deltas = paired_deltas(historical)
    historical_deltas.to_csv(RUN / "historical_paired_deltas.csv", index=False)
    historical_summary = delta_summary(historical_deltas, "historical")
    comparison = corrected_summary.merge(historical_summary, on=["model", "target_dataset", "n_seeds"], validate="one_to_one")
    comparison["historical_sign_delta_binary_f1"] = np.sign(comparison["historical_mean_delta_binary_f1"])
    comparison["corrected_sign_delta_binary_f1"] = np.sign(comparison["corrected_mean_delta_binary_f1"])
    comparison["major_sign_reversal"] = (
        comparison["historical_sign_delta_binary_f1"].ne(comparison["corrected_sign_delta_binary_f1"])
        & comparison["historical_sign_delta_binary_f1"].ne(0)
        & comparison["corrected_sign_delta_binary_f1"].ne(0)
        & comparison["historical_mean_delta_binary_f1"].abs().ge(0.10)
        & comparison["corrected_mean_delta_binary_f1"].abs().ge(0.10)
    )
    comparison.to_csv(RUN / "historical_comparison.csv", index=False)

    gates: list[dict[str, object]] = []
    for target in ["BoT-IoT", "ToN-IoT"]:
        row = comparison[(comparison["model"].eq("XGBoost")) & (comparison["target_dataset"].eq(target))].iloc[0]
        conditions = {
            "mean_delta_binary_f1_gt_0": bool(row["corrected_mean_delta_binary_f1"] > 0),
            "positive_seed_count_ge_2": bool(row["corrected_positive_seeds_delta_binary_f1"] >= 2),
            "mean_delta_mcc_gt_0": bool(row["corrected_mean_delta_mcc"] > 0),
            "mean_delta_fnr_lt_0": bool(row["corrected_mean_delta_fnr"] < 0),
        }
        gates.append({"gate": f"XGBoost_positive::{target}", "status": "Pass" if all(conditions.values()) else "Fail", "detail": json.dumps({**conditions, "mean_delta_binary_f1": float(row["corrected_mean_delta_binary_f1"]), "positive_seeds": int(row["corrected_positive_seeds_delta_binary_f1"]), "mean_delta_mcc": float(row["corrected_mean_delta_mcc"]), "mean_delta_fnr": float(row["corrected_mean_delta_fnr"])}, sort_keys=True)})
    rf_conditions = {}
    for target in TARGETS:
        value = float(comparison[(comparison["model"].eq("RandomForest")) & (comparison["target_dataset"].eq(target))]["corrected_mean_delta_binary_f1"].iloc[0])
        rf_conditions[target] = {"mean_delta_binary_f1": value, "lt_0": value < 0}
    gates.append({"gate": "RandomForest_reverse::all_targets", "status": "Pass" if all(item["lt_0"] for item in rf_conditions.values()) else "Fail", "detail": json.dumps(rf_conditions, sort_keys=True)})
    corrected_means = comparison["corrected_mean_delta_binary_f1"]
    heterogeneity = bool((corrected_means > 0).any() and (corrected_means < 0).any())
    gates.append({"gate": "target_model_heterogeneity", "status": "Pass" if heterogeneity else "Fail", "detail": f"positive_strata={int((corrected_means > 0).sum())}; negative_strata={int((corrected_means < 0).sum())}"})
    no_major_reversal = not bool(comparison["major_sign_reversal"].any())
    gates.append({"gate": "no_major_historical_sign_reversal", "status": "Pass" if no_major_reversal else "Fail", "detail": f"major_reversals={int(comparison['major_sign_reversal'].sum())}"})
    gates_df = pd.DataFrame(gates)
    gates_df.to_csv(RUN / "verdict_gate_checks.csv", index=False)
    qualitative_unchanged = bool(gates_df["status"].eq("Pass").all())

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(RUN / "artifact_validation_checks.csv", index=False)
    blockers = checks_df[checks_df["status"].eq("Blocker")]
    artifact_validation_passed = blockers.empty
    verdict = {
        "protocol_id": PROTOCOL_ID,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_validation_passed": artifact_validation_passed,
        "validation_checks": len(checks_df),
        "validation_passes": int(checks_df["status"].eq("Pass").sum()),
        "validation_blockers": int(checks_df["status"].eq("Blocker").sum()),
        "training_runs": len(manifests),
        "full_target_result_rows": int(results["evaluation_scope"].eq("full_target").sum()),
        "source_test_result_rows": int(results["evaluation_scope"].eq("corrected_source_test").sum()),
        "qualitative_h1_verdict_unchanged": qualitative_unchanged if artifact_validation_passed else None,
        "disposition": (
            "corrected-split sensitivity supports the qualitative H1 verdict"
            if artifact_validation_passed and qualitative_unchanged
            else "update H1; do not select the more favorable historical or corrected run"
            if artifact_validation_passed
            else "artifact validation failed; do not interpret H1"
        ),
        "gate_results": {row["gate"]: row["status"] for row in gates},
        "major_sign_reversal_strata": comparison.loc[comparison["major_sign_reversal"], ["model", "target_dataset"]].to_dict("records"),
    }
    (RUN / "final_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    display_cols = ["model", "target_dataset", "corrected_mean_delta_binary_f1", "corrected_positive_seeds_delta_binary_f1", "corrected_mean_delta_mcc", "corrected_mean_delta_fnr"]
    display = comparison[display_cols].copy()
    numeric_cols = [column for column in display.columns if column.startswith("corrected_mean")]
    display[numeric_cols] = display[numeric_cols].round(6)
    hist_display = comparison[["model", "target_dataset", "historical_mean_delta_binary_f1", "corrected_mean_delta_binary_f1", "major_sign_reversal"]].copy()
    hist_display[["historical_mean_delta_binary_f1", "corrected_mean_delta_binary_f1"]] = hist_display[["historical_mean_delta_binary_f1", "corrected_mean_delta_binary_f1"]].round(6)
    lines = [
        "# H1 Corrected-Split Sensitivity — Final Report",
        "",
        f"**Protocol:** `{PROTOCOL_ID}`  ",
        f"**Artifact validation:** {'PASS' if artifact_validation_passed else 'BLOCKER'} ({verdict['validation_passes']}/{verdict['validation_checks']})  ",
        f"**Frozen H1 verdict:** {'UNCHANGED' if qualitative_unchanged else 'CHANGED'}  ",
        f"**Disposition:** {verdict['disposition']}.",
        "",
        "## Coverage",
        "",
        "E1 and E2 were retrained on identical corrected memberships for seeds 42, 123, and 2024. Coverage is 18 model fits, 54 full-target result rows, 18 corrected source-test rows, and 18 source-validation diagnostic rows. No target labels entered fitting or model selection.",
        "",
        "## Corrected E2 − E1 target deltas",
        "",
        display.to_markdown(index=False),
        "",
        "## Frozen gate results",
        "",
        gates_df.to_markdown(index=False),
        "",
        "## Historical versus corrected Binary-F1 direction",
        "",
        hist_display.to_markdown(index=False),
        "",
        "## Interpretation boundary",
        "",
        "This sensitivity test closes the source-membership mismatch for the reported H1 protocol. It does not uniquely identify label semantics because the E1 and E2 arms retain their frozen arm-specific estimator parameters. Three-seed variation is descriptive, and CUDA artifacts are not claimed bitwise portable across hardware/software stacks.",
        "",
        "## Validation blockers",
        "",
        "None." if blockers.empty else blockers.to_markdown(index=False),
        "",
    ]
    (RUN / "H1_CORRECTED_SPLIT_SENSITIVITY_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    manifest_rows = []
    manifest_roots = [
        REPOSITORY_ROOT / "protocols" / "h1" / "training_policy.json",
        ROOT / "experiment_plan.md",
        ROOT / "protocol_amendment.md",
        ROOT / "train.py",
        Path(__file__).resolve(),
    ]
    manifest_roots.extend(path for path in RUN.rglob("*") if path.is_file() and path.name != "release_manifest_sha256.csv")
    for path in sorted(set(manifest_roots), key=lambda item: str(item).lower()):
        manifest_rows.append({"relative_path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(manifest_rows).to_csv(RUN / "release_manifest_sha256.csv", index=False)
    print(json.dumps(verdict, sort_keys=True))
    return 0 if artifact_validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
