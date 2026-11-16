# Cross-domain NIDS diagnostics

Reproducible code, frozen inputs and compact evidence for the revised manuscript.
The current H2/H3 experiments use corrected group-disjoint source partitions,
three training seeds and ten random masks per seed. Additional experiments cover
CORAL, MMD-loss alignment, FT-Transformer, BatchNorm, source weighting, mRMR-MID,
equal-budget shift, sensitivity analyses and target-supervised readout.

The SHAP extension is excluded. XGBoost remains a detector in H1, oracle and
calibration. The existing finalized H1, oracle and calibration paths are retained.

## Start here

Use Python 3.12 and the pinned direct dependencies:

```bash
python -m pip install -r requirements.txt
python verify_results.py
python run_revision.py plan
python run_revision.py verify
python -m pytest -q
```

These commands require no provider data and perform no formal training. Tests use
bounded synthetic data and can use CUDA if available. Verification recomputes
metrics, seed means, paired deltas and gates from released compact tables; it is
not a complete experimental rerun.

For new experiments, use the single root entry point `run_revision.py`.
Implementation modules are grouped in `experiments/revision/`.

## Data and frozen inputs

Download the four provider datasets listed in [data/README.md](data/README.md).
Put the CSVs in `data/raw/` and check [data/dataset_checksums.csv](data/dataset_checksums.csv).
Raw datasets are not redistributed.

`data/splits/revision/` contains corrected source maps, target adaptation and
calibration indices, alignment coefficients, the fixed matrix and random masks.
H1 and oracle use separate maps under `data/splits/h1/` and `data/splits/oracle/`.
Do not substitute their memberships for the revision maps. Source preprocessing
is fitted on the frozen source training rows. Models are constructed from
`data/model_hyperparameters.json` without old serialized templates.

## Run the current revision

These commands need provider CSVs. Formal training requires CUDA. Complete each
stage's check and smoke before running it. Run one stage at a time; the stages
share a process lock and validate upstream identities.

```bash
python run_revision.py core check --execute
python run_revision.py core smoke --execute
python run_revision.py core run --execute
python run_revision.py comparisons check --execute
python run_revision.py comparisons smoke --execute
python run_revision.py comparisons run --execute
```

The sensitivity stage includes a historical gate comparison. Before running it,
regenerate the historical H2/H3 execution products with the commands in
[docs/HISTORICAL_WORKFLOW.md](docs/HISTORICAL_WORKFLOW.md). Those historical outputs
use `results/generated/h2_h3/`, separately from corrected revision runs.

```bash
python run_revision.py sensitivity check --execute
python run_revision.py sensitivity smoke --execute
python run_revision.py sensitivity run --execute
```

New outputs use `results/generated/revision/{core,comparisons,sensitivity}`.
Repeat the same command after interruption. Complete units are checked and reused;
supported training resumes at complete epoch boundaries. Fresh execution creates
its own selection locks and provenance instead of importing old completion markers.

The fixed matrix has 108 H2 fits, 18 DANN fits, 48 comparison fits, 18 additional
strength fits and 120 nominal deletion-size cells. Shared controls are reused;
these counts are not independent new fits. The reviewed inventories record 342
unique completed fits across stages. Full execution processes tens of millions
of rows and requires substantial time and storage.

Core substeps are `h2-shift`, `h2-candidates`, `h2-select`, `h2-random`,
`h2-analyze`, `h3-pair` and `h3-analyze`; the pair command also requires `--target`
and `--seed`. Existing input and smoke checks still apply. Use default output
paths for the documented workflow.

## H1, oracle and calibration

These retained experiments keep their own established protocols:

```bash
python run_h1.py --execute --seed 42
python run_h1.py --execute --seed 123
python run_h1.py --execute --seed 2024
python run_h1.py finalize --execute
python run_oracle.py run --execute
python run_oracle.py finalize --execute
python run_calibration.py --mode full --strict
```

Calibration consumes fitted H1 pipelines. Root commands `run_h2.py` and `run_h3.py`
retain the historical protocol for provenance and sensitivity inputs. They do not
produce the corrected primary results; use `run_revision.py core` for those.

## Figures and reference evidence

```bash
python figures/scripts/plot_h2_loto_results.py
python figures/scripts/plot_revision_figures.py
```

The first command regenerates the archived feature-family JSD context figure.
The second regenerates current H1, matched H3 and random-deletion figures from
released per-seed evidence, plus historical class-confusion, feature-shift and
calibration summary plots from compact tables. No raw data or retraining is required. Outputs go only
to `figures/generated/`. Current comparison PDFs are in `figures/reference/revision/`.
The older H3 figure remains historical. Rendering may vary by platform; numerical
inputs are checked.

Current tables are in `results/reference/revision/`. The existing
`results/reference/h2/` and `results/reference/h3/` trees are historical.
See [docs/RELEASE_CHANGES.md](docs/RELEASE_CHANGES.md) for the paper-to-code mapping.

## Optional preparation audit

Frozen inputs are already included. To independently regenerate preparation
products from the provider CSVs:

```bash
python run_revision.py partitions --execute
python run_revision.py calibration --execute
python run_revision.py matrix --execute
```

Outputs go to `results/generated/revision_preparation/`, without replacing frozen
release inputs. Compare array memberships and numerical values; container bytes
and floating-point calibration can differ between environments.

## Release scope

Excluded: raw data, fitted models, per-row predictions, caches, internal review
documents and optional experiments absent from the manuscript. Full formal result
trees are not bundled. Compact reviewed evidence is included; fresh runs regenerate
checkpoints and selection locks.

Base commit: `bfa5be4931c442390435ca9edefdbc0037f9539c`. The revised source has new
hashes in `release_manifest_sha256.csv`; it is not the unchanged baseline.
See [docs/REPRODUCIBILITY_LIMITS.md](docs/REPRODUCIBILITY_LIMITS.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
