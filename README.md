# Cross-domain NIDS: H1, H2 and H3

Code for three main experiments:

- **H1:** compare multiclass and binary source-label formulations.
- **H2:** evaluate feature filtering with leave-one-target-out selection and
  matched random-deletion controls.
- **H3:** compare matched DANN models with and without domain loss.

H2/H3 use the corrected source partitions. Runs save fitted models, metrics and
completion records under `results/generated/`.

## Setup

Use Python 3.12.7 and install the pinned direct dependencies:

```bash
python -m pip install -r requirements.txt
python run_revision.py plan
python run_revision.py verify
```

`plan` shows the H2/H3 schedule. `verify` checks the frozen H2/H3 inputs without
training. Formal execution requires CUDA and the raw datasets. Full runs process
tens of millions of rows and need substantial RAM, storage and runtime.

## Data

Download the datasets from the [University of Queensland NIDS dataset page](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
and place them under `data/raw/`:

| Provider dataset | Required filename |
| --- | --- |
| NF-UNSW-NB15-v3 | `NF-UNSW-NB15-v3.csv` |
| NF-CSE-CIC-IDS2018-v3 | `NF-CICIDS2018-v3.csv` |
| NF-BoT-IoT-v3 | `NF-BoT-IoT-v3.csv` |
| NF-ToN-IoT-v3 | `NF-ToN-IoT-v3.csv` |

Execution checks the input hashes recorded in `protocols/h1/training_policy.json`
and `data/splits/revision/manifest.json`. Provider terms apply to the raw data.
H1 and H2/H3 use separate fixed partition maps; keep their memberships unchanged.
Model settings are in `data/model_hyperparameters.json`. H1 finalization also uses
`data/reference_inputs/h1_historical_results.csv`.

## H1

```bash
python run_h1.py --execute --seed 42
python run_h1.py --execute --seed 123
python run_h1.py --execute --seed 2024
python run_h1.py finalize --execute
```

H1 fits Random Forest, XGBoost and MLP for both source-label formulations on the
same corrected memberships. Results are written to `results/generated/h1/`.

## Corrected H2 and H3

Run the input check and bounded training check before formal execution:

```bash
python run_revision.py core check --execute
python run_revision.py core smoke --execute
python run_revision.py core run --execute
```

The fixed matrix contains 108 H2 candidate fits and 18 matched H3 fits across
seeds 42, 123 and 2024. H2 also runs ten matched random masks per seed for each
selected target/model combination. The random controls are part of H2.
H3 adapts to a fixed 5% target sample and evaluates the full target file,
including the adaptation rows.

Outputs use `results/generated/revision/core/`. Repeating a command verifies and
reuses completed units; supported training resumes at complete epoch boundaries.
Code and environment identities are locked for each run. Use a new output
folder after changing the code.

Individual core actions are `h2-shift`, `h2-candidates`, `h2-select`, `h2-random`,
`h2-analyze`, `h3-pair` and `h3-analyze`. `h3-pair` also needs `--target` and
`--seed`. These actions use the same input and smoke prerequisites.

## Files

- `run_h1.py`, `run_revision.py`: the two experiment entry points.
- `experiments/h1/`: H1 execution and shared preprocessing/training helpers.
- `experiments/h2/`, `experiments/h3/`: shared feature-shift, training and statistical helpers used by corrected H2/H3.
- `experiments/revision/`: corrected H2/H3 execution and required runtime checks.
- `src/cross_domain_nids/`: models, preprocessing and run utilities.
- `protocols/`, `data/`: fixed settings and inputs needed by these experiments.

The experiment settings and retained partition memberships match the preceding
release. Validation uses bounded synthetic runs and input checks; full provider-data
training was not rerun for this cleanup. Direct dependency pins do not lock every
transitive package, GPU driver or hardware-specific behavior. Project code uses
`LICENSE`; installed dependencies retain their own licenses.
