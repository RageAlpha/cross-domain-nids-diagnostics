"""Expand the predeclared fixed cells and shared random-mask bank."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cross_domain_nids.protocol_utils import deterministic_seed

OUT = ROOT / "results/generated/revision_preparation"
TARGETS = ("CICIDS2018", "BoT-IoT", "ToN-IoT")
SEEDS = (42, 123, 2024)
ALL = "|".join(TARGETS)
PROTOCOL = "KBS-R1-CORRECTED-20261111-v1"
PHASE_PACKAGES = {
    2: ("H2", "H3"),
    3: ("alignment", "FT", "BN", "weight", "mRMR"),
    4: ("strength", "random_size"),
}


def mask(features, seed, k, replicate):
    generator = np.random.default_rng(deterministic_seed(PROTOCOL, "random_mask", seed, k, replicate))
    return sorted(generator.choice(features, k, replace=False).tolist())


def main():
    with (OUT / "alignment_coefficients.csv").open(encoding="utf-8") as stream:
        coefficients = {(r["target"], int(r["seed"]), r["method"]): r for r in csv.DictReader(stream)}
    with (ROOT / "data/raw/NF-UNSW-NB15-v3.csv").open(encoding="utf-8-sig") as stream:
        features = [c for c in next(csv.reader(stream)) if c not in
                    {"Label", "Attack", "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"}]
    rows = []
    bank = []

    def add(package, model, condition, target, seed, source_seed, reference="", strength="", bandwidth=""):
        scope = "all_targets" if target == ALL else target
        run_id = f"{package}/{model}/{condition}/{scope}/{seed}"
        rows.append((run_id, package, model, condition, target, seed, source_seed, reference, strength, bandwidth))
        return run_id

    for seed in SEEDS:
        for model in ("RandomForest", "MLP"):
            for condition in ("full", "dup_columns", "corr098"):
                add("H2", model, condition, ALL, seed, seed)
            for target in TARGETS:
                for condition in ("shift3", "shift5", "shift10", "shift15", "shift5_corr098"):
                    add("H2", model, condition, target, seed, seed)
            for k in (5, 10):
                add("mRMR", model, f"remove{k}", ALL, seed, seed)
        for k in (3, 5, 10, 15):
            for replicate in range(10):
                bank.append({"seed": seed, "k": k, "replicate": replicate, "removed": mask(features, seed, k, replicate)})
                add("random_size", "MLP", f"k{k}_rep{replicate:02}", "ToN-IoT", seed, seed)
        for target in TARGETS:
            zero = add("H3", "DANN", "zero", target, seed, 42, strength=0)
            add("H3", "DANN", "original", target, seed, 42, zero, 1)
            for alpha in (.1, .5):
                add("strength", "DANN", f"alpha{alpha}", target, seed, 42, zero, alpha)
            for method in ("coral", "mmd"):
                coeff = coefficients[target, seed, method]
                add("alignment", method, "nonzero", target, seed, 42, zero,
                    coeff["nonzero_coefficient_max"], coeff["mmd_bandwidth_squared"])
        for package, model in (("FT", "FT"), ("BN", "DANN_BN_updated"), ("weight", "DANN_weighted")):
            zero = add(package, model, "zero", "ToN-IoT", seed, 42, strength=0)
            add(package, model, "original", "ToN-IoT", seed, 42, zero, 1)
    with (OUT / "formal_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run_id","package","model","condition","evaluate_targets","training_seed",
                         "source_split_seed","zero_control_id","strength_max","mmd_bandwidth_squared","phase"))
        for phase, packages in PHASE_PACKAGES.items():
            for package in packages:
                writer.writerows((*row, phase) for row in rows if row[1] == package)
    (OUT / "random_masks.json").write_text(json.dumps({"features": features, "masks": bank}, indent=2), encoding="utf-8")
    print(f"{len(rows)} fixed design rows, {len(bank)} random masks")


if __name__ == "__main__":
    main()
