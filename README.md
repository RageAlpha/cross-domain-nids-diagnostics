# Cross-domain NIDS diagnostics

This repository contains the finalized code, frozen protocols, fixed data
partitions, compact reference results, and figure inputs required to reproduce
or verify the experiments reported in the current manuscript.

Historical drafts, duplicate run trees, checkpoints, predictions, internal
review records, and raw provider datasets are intentionally excluded.

## Quick verification

Install the dependencies and verify the released evidence:

```powershell
python -m pip install -r requirements.txt
python verify_results.py
```

The stable root-level entry points are:

| Manuscript block | Safe command | Immutable reference results |
|---|---|---|
| H1 formulation sensitivity | `python run_h1.py` | `results/reference/h1/` |
| H2 LOTO filtering | `python run_h2.py plan` | `results/reference/h2/` |
| H3 paired domain loss | `python run_h3.py plan` | `results/reference/h3/` |
| Target-oracle ceiling | `python run_oracle.py dry-run` | `results/reference/oracle/` |
| Calibration analysis | `python run_calibration.py --help` | `results/reference/calibration/` |

The default H1 invocation, H2/H3 `plan`, and Oracle `dry-run` perform no model
fitting. New executions write only below `results/generated/`; they do not
overwrite the manuscript reference tables.

## Full reproduction

Run all commands from the repository root. Create the verified environment
with either:

```powershell
conda env create -f environment.yml
conda activate cross-domain-nids
```

or:

```powershell
python -m pip install -r requirements.txt
```

Place the four provider CSV files under `data/raw/` and verify their SHA-256
values against `data/dataset_checksums.csv`.

### H1

```powershell
python run_h1.py
python run_h1.py --execute --seed 42
python run_h1.py --execute --seed 123
python run_h1.py --execute --seed 2024
python run_h1.py finalize
```

### H3 paired domain loss

H3 is completed before H2 because both use the same frozen protocol root:

```powershell
python run_h3.py init --execute
$targets = @("CICIDS2018", "BoT-IoT", "ToN-IoT")
$seeds = @(42, 123, 2024)
foreach ($target in $targets) {
  foreach ($seed in $seeds) {
    python run_h3.py pair --target $target --seed $seed --execute
  }
}
foreach ($seed in $seeds) {
  python run_h3.py s0 --seed $seed --execute
}
python run_h3.py analyze --execute
```

### H2 LOTO filtering

```powershell
python run_h2.py shift --execute
python run_h2.py candidates --execute
python run_h2.py select --execute
$targets = @("CICIDS2018", "BoT-IoT", "ToN-IoT")
$models = @("RandomForest", "MLP")
$seeds = @(42, 123, 2024)
$starts = @(0, 5, 10, 15)
foreach ($target in $targets) {
  foreach ($model in $models) {
    foreach ($seed in $seeds) {
      foreach ($start in $starts) {
        python run_h2.py random-batch --target $target --model $model `
          --seed $seed --rep-start $start --rep-count 5 --execute
      }
    }
  }
}
python run_h2.py analyze --execute
```

### Target-oracle ceiling

```powershell
python run_oracle.py dry-run
python run_oracle.py run --execute
python run_oracle.py finalize --execute
```

### Calibration

Calibration consumes the fitted H1 pipelines:

```powershell
python run_calibration.py --mode full --strict
```

Formal H1 execution requires a CUDA-capable GPU because the released H1 model
set includes the Torch MLP. H3 supports `--cpu`, although a CUDA-capable GPU is
strongly recommended for full training. Dry-runs, plans, tests, split
verification, and manuscript-number verification remain available without
CUDA. Full execution reads tens of millions of rows and may require many hours
and tens of gigabytes of working storage. Run stages sequentially unless
output isolation and available resources have been checked.

## Repository layout

```text
src/cross_domain_nids/   Reusable models, preprocessing, and protocol utilities
experiments/             H1, H2, H3, Oracle, and calibration experiment runners
protocols/               Frozen policies and H2 selection locks
data/splits/             Released H1 and Oracle partition maps
data/reference_inputs/   Compact metric inputs required by finalizers
results/reference/       Immutable compact results reported in the manuscript
results/generated/       Ignored output location for new executions
figures/scripts/         Semantic figure-generation entry points
figures/data/            Compact figure inputs and validation notes
figures/reference/       Reference PDFs used to check the manuscript figures
tests/                   Protocol, split, and release-inventory tests
docs/                    Reproducibility boundaries
```

All public file and directory names are stable and date-free. Historical
protocol identifiers remain inside metadata values to preserve the scientific
identity of archived results.

## Data and model construction

Raw NetFlow CSVs are not redistributed. Obtain the four files identified in
`data/dataset_checksums.csv`, place them under `data/raw/`, and verify their
SHA-256 values before training.

H1 and H2 instantiate estimators directly from
`data/model_hyperparameters.json`. Historical `.joblib` templates are not
required. Newly fitted models may still be serialized under
`results/generated/`.

The compact historical metric tables required by the H1 and Oracle finalizers
are included under `data/reference_inputs/`; they contain no model binaries or
row-level predictions.

Dataset labels use manuscript spelling (`CICIDS2018`, `BoT-IoT`, `ToN-IoT`);
file names use stable lowercase slugs (`cicids2018`, `bot_iot`, `ton_iot`).

## Figures

```powershell
python figures/scripts/plot_h2_loto_results.py
python figures/scripts/plot_h3_paired_domain_loss.py
```

Generated figures are written to `figures/generated/`. The immutable manuscript
references are `figures/reference/figure_2.pdf` and
`figures/reference/figure_3.pdf`.

## Scope

This is a minimal reproducibility repository, not a bitwise archive of the
author's workstation. See `docs/REPRODUCIBILITY_LIMITS.md` before claiming
one-command full retraining.
