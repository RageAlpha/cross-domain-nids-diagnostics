"""Validate, compare, and release the corrected target-only oracle ceiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
PROTOCOL_ID = "NIDS-TARGET-ORACLE-CORRECTED-CEILING-20260717-v1"
TARGETS = ("CICIDS2018", "BoT-IoT", "ToN-IoT")
SEEDS = (42, 123, 2024)
MODELS = ("RandomForest", "XGBoost", "TorchMLP")
METRICS = ("binary_f1", "mcc", "fpr", "fnr", "weighted_f1")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def add_check(checks: list[dict[str, Any]], category: str, check: str, passed: bool, detail: Any) -> None:
    checks.append({"category": category, "check": check, "status": "Pass" if passed else "Fail", "detail": detail})


def load_markers(output_dir: Path) -> list[dict[str, Any]]:
    markers = []
    for path in sorted(output_dir.glob("completions/*/seed*/*.json")):
        markers.append(json.loads(path.read_text(encoding="utf-8")))
    return markers


def build_release_manifest(args: argparse.Namespace) -> None:
    path = args.output_dir / "release_manifest_sha256.csv"
    rows = []
    for source in sorted(args.output_dir.rglob("*")):
        if source.is_file() and source != path and "temp" not in source.parts:
            rows.append({
                "role": "generated",
                "path": str(source.resolve()),
                "relative_path": str(source.relative_to(args.output_dir)),
                "sha256": hash_file(source),
                "bytes": source.stat().st_size,
            })
    for source in (
        REPOSITORY_ROOT / "protocols" / "oracle" / "split_policy.json",
        HERE / "create_splits.py",
        HERE / "run_target_oracle.py",
        HERE / "finalize_results.py",
        REPOSITORY_ROOT / "src" / "cross_domain_nids" / "oracle_split_audit.py",
        REPOSITORY_ROOT / "tests" / "test_oracle_splits.py",
    ):
        rows.append({
            "role": "frozen_source",
            "path": str(source.resolve()),
            "relative_path": "",
            "sha256": hash_file(source),
            "bytes": source.stat().st_size,
        })
    pd.DataFrame(rows).drop_duplicates(subset=["role", "path"]).to_csv(path, index=False)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    split_manifest_path = REPOSITORY_ROOT / "data" / "splits" / "oracle" / "manifest.csv"
    split_checks_path = (
        REPOSITORY_ROOT / "data" / "splits" / "oracle" / "validation_checks.csv"
    )
    if not split_manifest_path.is_file() or not split_checks_path.is_file():
        raise RuntimeError("Released Oracle split inputs are incomplete")
    markers = load_markers(args.output_dir)
    rows = pd.DataFrame([marker["result"] for marker in markers])
    split_manifest = pd.read_csv(split_manifest_path)
    split_checks = pd.read_csv(split_checks_path)
    checks: list[dict[str, Any]] = []
    add_check(checks, "coverage", "nine corrected split maps", len(split_manifest) == 9, len(split_manifest))
    add_check(checks, "coverage", "all corrected split checks pass", bool((split_checks.status == "Pass").all()), int((split_checks.status != "Pass").sum()))
    add_check(checks, "coverage", "twenty-seven completion markers", len(markers) == 27, len(markers))
    expected_units = {(target, seed, model) for target in TARGETS for seed in SEEDS for model in MODELS}
    actual_units = {(str(row.target_dataset), int(row.seed), str(row.model)) for row in rows.itertuples(index=False)} if not rows.empty else set()
    add_check(checks, "coverage", "exact target-seed-model Cartesian product", actual_units == expected_units, len(actual_units))

    if not rows.empty:
        for row in rows.itertuples(index=False):
            prefix = f"{row.target_dataset} seed={row.seed} model={row.model}"
            artifact = Path(row.artifact)
            add_check(checks, "artifact", prefix + " artifact present", artifact.is_file(), artifact)
            add_check(checks, "artifact", prefix + " artifact hash", artifact.is_file() and hash_file(artifact) == row.artifact_sha256, row.artifact_sha256)
            map_path = Path(row.split_map_path)
            add_check(checks, "split", prefix + " split hash", map_path.is_file() and hash_file(map_path) == row.split_map_sha256, row.split_map_sha256)
            add_check(checks, "counts", prefix + " confusion sum", int(row.tp + row.tn + row.fp + row.fn) == int(row.test_size), row.test_size)
            add_check(checks, "counts", prefix + " retained split sum", int(row.train_size + row.val_size + row.test_size) == int(row.retained_target_rows), row.retained_target_rows)
            add_check(checks, "counts", prefix + " full split sum", int(row.retained_target_rows + row.drop_size) == int(row.full_target_rows), row.full_target_rows)
            add_check(checks, "schema", prefix + " 49 features", int(row.feature_count) == 49, row.feature_count)
            add_check(checks, "protocol", prefix + " corrected protocol", row.protocol_id == PROTOCOL_ID, row.protocol_id)
            for metric in METRICS:
                value = float(getattr(row, metric))
                low = -1.0 if metric == "mcc" else 0.0
                add_check(checks, "metric", prefix + " " + metric + " range", np.isfinite(value) and low <= value <= 1.0, value)

    validation = pd.DataFrame(checks)
    validation.to_csv(args.output_dir / "final_validation_checks.csv", index=False)
    failures = int((validation.status != "Pass").sum())

    if len(rows) == 27:
        order_target = {name: idx for idx, name in enumerate(TARGETS)}
        order_model = {name: idx for idx, name in enumerate(MODELS)}
        rows["_target_order"] = rows.target_dataset.map(order_target)
        rows["_model_order"] = rows.model.map(order_model)
        rows = rows.sort_values(["_target_order", "seed", "_model_order"]).drop(columns=["_target_order", "_model_order"])
    rows.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    if not rows.empty:
        summary = rows.groupby(["model", "target_dataset"])[list(METRICS)].agg(["mean", "std"]).reset_index()
        summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    else:
        summary = pd.DataFrame()
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    historical = pd.read_csv(args.historical_results)
    merge_keys = ["target_dataset", "seed", "model"]
    paired = rows.merge(historical[merge_keys + list(METRICS)], on=merge_keys, suffixes=("_corrected", "_historical"), validate="one_to_one") if len(rows) == 27 else pd.DataFrame()
    if not paired.empty:
        for metric in METRICS:
            paired[f"delta_{metric}"] = paired[f"{metric}_corrected"] - paired[f"{metric}_historical"]
    paired.to_csv(args.output_dir / "historical_comparison.csv", index=False)

    clean = failures == 0 and len(rows) == 27
    final = {
        "protocol_id": PROTOCOL_ID,
        "generated_utc": utc_now(),
        "split_maps": int(len(split_manifest)),
        "training_runs": int(len(rows)),
        "validation_checks": int(len(validation)),
        "validation_failures": failures,
        "clean_target_domain_ceiling": clean,
        "decision": "clean_target_domain_ceiling_established" if clean else "corrected_oracle_incomplete_or_invalid",
        "historical_values_replaced": bool(clean),
    }
    atomic_json(args.output_dir / "final_verdict.json", final)

    summary_md = summary.to_markdown(index=False) if not summary.empty else "No complete summary."
    delta_cols = ["target_dataset", "seed", "model"] + [f"delta_{metric}" for metric in METRICS]
    paired_md = paired[delta_cols].to_markdown(index=False) if not paired.empty else "No complete paired comparison."
    report = "\n".join([
        "# Corrected target-only oracle ceiling final report",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        f"**Decision:** `{final['decision']}`",
        "",
        f"Corrected splits: {len(split_manifest)}/9. Training runs: {len(rows)}/27. Validation failures: {failures}.",
        "",
        "## Corrected mean and standard deviation",
        "",
        summary_md,
        "",
        "## Corrected minus historical paired differences",
        "",
        paired_md,
        "",
        "Corrected values replace the historical row-stratified oracle values without favorable-result selection.",
        "",
    ])
    (args.output_dir / "final_report.md").write_text(report, encoding="utf-8")
    disposition = "\n".join([
        "# Corrected target-only oracle manuscript disposition",
        "",
        f"Decision: **{'Pass' if clean else 'Blocker'}**",
        "",
        "> We estimate the recoverable target-domain ceiling using supervised target-only models trained on duplicate-free, timestamped-session-disjoint 70/10/20 partitions. Preprocessing is fit only on target training rows, MLP early stopping uses only the target validation partition, and all reported metrics are computed on the held-out target test partition.",
        "",
        "The corrected table must replace the historical row-stratified oracle table. Historical values may appear only in the partition-sensitivity comparison, not as an alternative ceiling.",
        "",
    ])
    (args.output_dir / "manuscript_disposition.md").write_text(disposition, encoding="utf-8")

    build_release_manifest(args)
    print(json.dumps(final, indent=2), flush=True)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "generated" / "oracle",
    )
    parser.add_argument(
        "--historical-results",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "data"
            / "reference_inputs"
            / "oracle_historical_results.csv"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.execute:
        raise SystemExit("finalization requires --execute")
    from cross_domain_nids.release_paths import require_generated_output

    args.output_dir = require_generated_output(args.output_dir, "results")
    finalize(args)


if __name__ == "__main__":
    main()
