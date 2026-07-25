# Cross-domain NIDS diagnostics

This repository contains the finalized code, frozen protocols, fixed data
partitions, compact reference results, and figure inputs associated with the
current manuscript. It is a minimal reproducibility package, not a copy of the
authors' workstation.

Raw NetFlow datasets must be downloaded by the user from the University of
Queensland NIDS dataset page:
https://staff.itee.uq.edu.au/marius/NIDS_datasets/

## Quick verification

The commands in this section verify the committed reference tables, split
maps, protocol locks, model-construction parameters, and release hashes. They
do **not** retrain the models or rerun the full experiments.

From a fresh clone:

```powershell
python -m pip install -r requirements.txt
python verify_results.py
pytest -q
python run_h1.py
python run_h1.py finalize --help
python run_h2.py plan
python run_h3.py plan
python run_oracle.py dry-run
python run_calibration.py --help
python experiments/h2/compute_feature_shift.py --help
```

The safe entry points require neither raw data nor a GPU and perform no model
fitting. They do not depend on `PYTHONPATH`, an IDE working directory,
uncommitted configuration files, Conda-specific environment variables, or
directories on the authors' computers.

The root entry points are also relocatable. For example, from the parent of a
fresh clone:

```powershell
python cross-domain-nids-diagnostics/run_h1.py
```

`verify_results.py` is evidence verification only. A successful verification
must not be described as a complete experimental rerun.
Untracked raw data, caches, and files below `results/generated/` or
`figures/generated/` are ignored by this runtime verification. CI separately
checks that every Git-tracked release file is represented in the release
manifest.

## Full experiment reproduction

The commands in this section read the raw provider CSVs and regenerate fitted
models and experiment outputs under `results/generated/`. They never overwrite
the committed `results/reference/` tree.

Create an environment with the pinned direct dependencies using either:

```powershell
conda env create -f environment.yml
conda activate cross-domain-nids
```

or:

```powershell
python -m pip install -r requirements.txt
```

Download the four datasets from
https://staff.itee.uq.edu.au/marius/NIDS_datasets/, place the expected CSV
files under `data/raw/`, and verify their SHA-256 values against
`data/dataset_checksums.csv`. The exact name mapping is documented in
`data/README.md`.

### H1 formulation sensitivity

The three `--execute` commands retrain E1 and E2. `finalize` validates their
outputs and generates compact tables; it does not fit another model.

```powershell
python run_h1.py
python run_h1.py --execute --seed 42
python run_h1.py --execute --seed 123
python run_h1.py --execute --seed 2024
python run_h1.py finalize --execute
```

### H3 paired domain loss

`init` creates frozen execution state. `pair` and `s0` train models. `analyze`
generates the H3 statistical summaries from the completed runs.

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

Pass `--cpu` to H3 execution commands only when GPU execution is unavailable;
CPU training is supported but substantially slower.

### H2 LOTO filtering

`shift`, `candidates`, and each `random-batch` command compute features or train
models. `select` creates the immutable held-out-target selection locks from
development-target results. `analyze` generates the final H2 tables.

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

H3 is completed before H2 because both use the same frozen H2/H3 protocol
root.

### Target-oracle ceiling

`run` retrains 27 target-only units. `finalize` verifies their coverage and
generates compact result tables.

```powershell
python run_oracle.py dry-run
python run_oracle.py run --execute
python run_oracle.py finalize --execute
```

### Calibration

Calibration is inference-only: it consumes the fitted H1 pipelines and raw
target CSVs, then writes calibration tables and plots. It does not fit models
or preprocessing objects.

```powershell
python run_calibration.py --mode full --strict
```

## Figure and table regeneration

The following commands use only the committed compact figure inputs. They do
not require raw datasets or a GPU and write only to `figures/generated/`:

```powershell
python figures/scripts/plot_h2_loto_results.py
python figures/scripts/plot_h3_paired_domain_loss.py
```

The immutable comparison PDFs remain under `figures/reference/`. Experiment
tables are regenerated by the H1/Oracle finalizers and the H2/H3 `analyze`
commands after their prerequisite runs have completed. Calibration writes its
tables during the inference-only calibration command.

## Hardware requirements

- Quick verification, tests, plans, dry-runs, and figure regeneration are
  CPU-safe.
- Formal H1 execution requires a CUDA-capable GPU because the released H1
  model set includes the Torch MLP.
- H3 supports `--cpu`; a CUDA-capable GPU is strongly recommended.
- H2 and Oracle contain CPU estimators and Torch models and should be run on a
  machine with sufficient RAM, storage, and GPU capacity for the selected
  units.
- Full execution reads tens of millions of rows and may require many hours and
  tens of gigabytes of working storage.

Random seeds and deterministic execution settings are fixed where supported.
Cross-hardware and cross-environment bitwise identity is not claimed.

`requirements.txt` and `environment.yml` pin the direct dependencies used by
the project. They are not complete transitive lockfiles and do not lock a GPU
driver, CUDA runtime, or platform-specific PyTorch build. Use a generated
`requirements-lock.txt` or `conda-lock` file if deployment requires a fully
resolved platform-specific environment.

## Data and model construction

Raw datasets are not redistributed. The expected inputs are
`NF-UNSW-NB15-v3.csv`, `NF-CICIDS2018-v3.csv`, `NF-BoT-IoT-v3.csv`, and
`NF-ToN-IoT-v3.csv`. Dataset labels use manuscript spelling
(`CICIDS2018`, `BoT-IoT`, `ToN-IoT`), while lock and split filenames use
lowercase slugs (`cicids2018`, `bot_iot`, `ton_iot`).

H1 and H2 instantiate estimators directly from
`data/model_hyperparameters.json`. Historical `.joblib` templates are not
required. The compact historical metric tables required by the H1 and Oracle
finalizers are included under `data/reference_inputs/`.

## Repository layout

```text
src/cross_domain_nids/   Models, preprocessing, output guards, protocol tools
experiments/             H1, H2, H3, Oracle, and calibration runners
protocols/               Frozen policies and six H2 selection locks
data/splits/             Released H1 and Oracle partition maps
data/reference_inputs/   Compact metric inputs required by finalizers
results/reference/       Immutable manuscript reference results
results/generated/       Ignored destination for new experiment executions
figures/scripts/         Figure-generation entry points
figures/data/            Compact figure inputs and validation notes
figures/reference/       Immutable manuscript comparison PDFs
figures/generated/       Ignored destination for regenerated figures
tests/                   Protocol, CLI, portability, and hygiene tests
docs/                    Reproducibility and release guidance
```

All public filenames are stable and date-free. Historical protocol identifiers
remain inside metadata values to preserve the identity of archived results.
Every execution-capable public path rejects output destinations inside
`results/reference/` or `figures/reference/`.

## Release and anonymity

The fixed local release identifier is `v1.0.0-manuscript`. Obtain its commit
with:

```powershell
git rev-parse v1.0.0-manuscript^{commit}
```

For strict double-anonymous review, do not publish this repository from an
account that can be linked to the authors. Use an anonymous review service or
an independent anonymous account. Add a repository URL to `CITATION.cff` and
the manuscript only after that anonymous URL has been chosen and verified.
See `docs/CODE_AVAILABILITY_TEMPLATE.md`.

See `docs/REPRODUCIBILITY_LIMITS.md` and `THIRD_PARTY_NOTICES.md` before making
claims about complete retraining, redistribution, or third-party licensing.

## Clean source archive

Distribute a Git archive rather than zipping the working directory:

```powershell
git archive --format=zip `
  --output cross-domain-nids-diagnostics-v1.0.0-manuscript.zip `
  v1.0.0-manuscript
```

This includes only tracked files from the fixed tag and excludes `.git/`,
caches, untracked raw data, generated outputs, and obsolete empty directories.
