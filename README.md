# Cross-domain NIDS experiments

Experiment code for cross-domain network intrusion detection: H1 formulation
sensitivity, corrected H2 feature selection and H3 domain alignment, comparison
methods, sensitivity analysis, target oracle and calibration. SHAP is excluded.
Runs save models, metrics and completion records under `results/generated/`.

## Setup and data

Use Python 3.12.7. Install the direct dependencies:

```bash
python -m pip install -r requirements.txt
python run_revision.py plan
python run_revision.py verify
```

`plan` shows the experiment schedule; `verify` checks the frozen revision inputs.
Neither command trains models. Formal revision training requires CUDA and the
provider CSVs. Full execution reads tens of millions of rows and needs substantial
RAM, storage and runtime. Dependency pins do not lock GPU drivers or all transitive
packages; results can vary across hardware.

Download the datasets from the [University of Queensland NIDS dataset page](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
and put these files under `data/raw/`:

| Provider dataset | Required filename |
| --- | --- |
| NF-UNSW-NB15-v3 | `NF-UNSW-NB15-v3.csv` |
| NF-CSE-CIC-IDS2018-v3 | `NF-CICIDS2018-v3.csv` |
| NF-BoT-IoT-v3 | `NF-BoT-IoT-v3.csv` |
| NF-ToN-IoT-v3 | `NF-ToN-IoT-v3.csv` |

Check their hashes against `data/dataset_checksums.csv`. Provider terms apply.
`data/model_hyperparameters.json` specifies model parameters. `data/splits/`
contains the fixed split maps, masks, coefficients and input checks used by the
runners. H1, oracle and revision maps are separate and cannot be substituted.
The two tables in `data/reference_inputs/` are consumed by H1/oracle finalization.

## Corrected H2/H3 and comparisons

Run the stages sequentially. Each stage checks its inputs and runs a bounded
training check before formal execution:

```bash
python run_revision.py core check --execute
python run_revision.py core smoke --execute
python run_revision.py core run --execute
python run_revision.py comparisons check --execute
python run_revision.py comparisons smoke --execute
python run_revision.py comparisons run --execute
```

The corrected protocol uses three seeds (42, 123, 2024) and ten random masks per
seed. H3 adapts to a fixed 5% target sample and evaluates the entire target file,
including adaptation rows. Comparisons include CORAL, MMD, FT-Transformer,
BatchNorm, source weighting, mRMR and equal-budget feature shift.

Outputs use `results/generated/revision/core/` and `comparisons/`. Repeat the same
command after interruption: complete units are checked and reused, and supported
training resumes at complete epoch boundaries. Code and environment identities
are locked for each run; start a new output directory after changing the code.

## H1, oracle and calibration

```bash
python run_h1.py --execute --seed 42
python run_h1.py --execute --seed 123
python run_h1.py --execute --seed 2024
python run_h1.py finalize --execute
python run_oracle.py run --execute
python run_oracle.py finalize --execute
python run_calibration.py --mode full --strict
```

H1 includes a Torch MLP and requires CUDA. Calibration uses the fitted H1
pipelines and raw target CSVs to compute calibration metrics and bin tables.

## Historical inputs for sensitivity analysis

Sensitivity analysis compares the corrected gates with the historical 20-mask
protocol. After H1, regenerate the historical H3 and H2 outputs below, in this
order. These commands use `results/generated/h2_h3/`; corrected primary results
come from `run_revision.py core`.

The loops below use PowerShell:

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
python run_h2.py shift --execute
python run_h2.py candidates --execute
python run_h2.py select --execute
$models = @("RandomForest", "MLP")
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

Then run sensitivity experiments:

```bash
python run_revision.py sensitivity check --execute
python run_revision.py sensitivity smoke --execute
python run_revision.py sensitivity run --execute
```

This stage covers gate sensitivity, DANN strength, random deletion size and a
post-hoc target-supervised readout. Outputs use
`results/generated/revision/sensitivity/`. The `smoke` modules and their helper
checks are part of the execution prerequisites.

## Rebuild preparation inputs

The fixed inputs are included. To independently regenerate them from provider CSVs:

```bash
python run_revision.py partitions --execute
python run_revision.py calibration --execute
python run_revision.py matrix --execute
```

Outputs go to `results/generated/revision_preparation/`. Compare memberships and
numerical values with `data/splits/revision/`; container bytes can vary.

## Layout and licensing

- `run_*.py`: experiment entry points.
- `experiments/`: training, evaluation, preparation and execution checks.
- `src/cross_domain_nids/`: shared models, preprocessing and protocol utilities.
- `protocols/`: experiment settings and historical selection locks.
- `data/`: model settings, fixed partitions and required finalization inputs.

See `LICENSE`, `THIRD_PARTY_NOTICES.md` and the retained FT-Transformer vendor
license for licensing. This cleanup preserves experiment calculations; validation
uses synthetic runs and input checks, without a full provider-data retraining.
