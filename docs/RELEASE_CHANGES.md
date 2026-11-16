# Revised release and manuscript mapping

Base: public commit `bfa5be4931c442390435ca9edefdbc0037f9539c`.

| Evidence | Implementation | Reference tables |
| --- | --- | --- |
| H1 formulation sensitivity | `run_h1.py`, `experiments/h1/` | `results/reference/h1/` |
| Corrected H2 and dual controls | `experiments/revision/h2.py`, `h2_analysis.py` | `results/reference/revision/core/` |
| Corrected matched H3 | `experiments/revision/h3.py` | `results/reference/revision/core/` |
| CORAL/MMD, FT, BN, source weighting | `experiments/revision/comparisons/paired.py`, `estimators.py` | `results/reference/revision/comparisons/` |
| mRMR and equal-budget shift | `experiments/revision/comparisons/features.py`, `selection.py` | `results/reference/revision/comparisons/feature_*` |
| Gate sensitivity | `experiments/revision/sensitivity/gates.py` | `results/reference/revision/sensitivity/gate_*` |
| DANN strength | `experiments/revision/sensitivity/strength.py` | `results/reference/revision/sensitivity/strength_*` |
| Random deletion | `experiments/revision/sensitivity/random_size.py` | `results/reference/revision/sensitivity/random_*` |
| Supervised target readout | `experiments/revision/sensitivity/readout.py` | `results/reference/revision/sensitivity/readout_*` |
| Oracle and calibration | `run_oracle.py`, `run_calibration.py` | `results/reference/oracle/`, `calibration/` |
| Figure 2: archived feature-family shift | `figures/scripts/plot_h2_loto_results.py` | `figures/data/h2_loto_figure_data.csv` |
| Current Figures 3, 4 and 5 | `figures/scripts/plot_revision_figures.py` | Current H3, H1 and random-mask tables |

New experiments reuse public model, metric, preprocessing, feature-shift and
protocol code. One `estimators.py` replaces four mixed-purpose helper modules.
Tests remain separate from runtime logic. Original exploratory commands are omitted.

The SHAP package, routines, outputs and settings are excluded. Only the mRMR
functions were extracted from the mixed feature-selection helper. XGBoost baseline
detectors remain. RF/MLP constructor parameters were compared with historical
templates and matched after the original seed/verbosity changes. Runtime loading
of those templates is removed. FT source hashes and its original license remain.

Reference tables retain numerical values and run identities. Absolute model paths
were removed from the random-mask table. Review-check tables retain names/statuses,
with private evidence paths omitted. Status fields do not establish that absent
checkpoints were rechecked during release preparation.

Validation includes compact evidence arithmetic, frozen masks, model construction,
synthetic training, paired controls, checkpoint reuse, interrupted MLP/FT/DANN
recovery, two-budget selection, 120-cell random scheduling, readout partitioning,
output guards and CLI relocation. No full dataset training was run for packaging.
