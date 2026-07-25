# Reproducibility limits and intentionally excluded files

## Included

- Final H1, H2, H3, target-oracle, and calibration code paths.
- Frozen policies, H2 selection locks, and all 12 fixed partition maps.
- Compact per-seed/summary result tables under `results/reference/`.
- Final Figure 2 and Figure 3 plotting code, inputs, and reference PDFs.
- Dataset checksums, 49-feature schema, and model parameter dictionaries.

## Intentionally excluded

- Raw datasets: redistribution is governed by the data providers.
- Historical H1/E2 estimator templates: about 360 MB of joblib/pickle objects.
  They are not required: H1 and H2 now construct the same estimator classes
  directly from the released parameter dictionaries.
- H3 checkpoints and per-row predictions: the historical tree is about
  23.7 GB. They are unnecessary for checking manuscript values but are needed
  to rerun the archived complement analysis without retraining.
- Oracle fitted models: about 142 MB; the code and fixed split maps are enough
  to retrain them from raw data.
- Calibration per-row predictions: compact ECE/Brier outputs are included;
  exact row-level recalculation requires regenerated or separately deposited
  predictions.
- Historical manifests, command logs, completion markers, repeated
  `formal_run_final` trees, review reports, old figure variants, and unrelated
  experiment scripts.

## Known portability boundaries

Repository inputs, Oracle manifests, and public runner defaults now use
repository-relative paths. H1/H2 training no longer loads historical estimator
templates. Raw provider data are still required for retraining. CUDA is
required for formal H1 execution and strongly recommended for H3, which also
supports the slower `--cpu` mode. Full H3 complement recalculation still needs
retrained or separately deposited predictions.

The package supports manuscript-number verification immediately. Full
retraining additionally requires provider data and suitable hardware.

The dependency files pin direct dependencies only. They do not constitute a
complete transitive, CUDA-runtime, driver, or platform lock.
