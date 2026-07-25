# H3 figure data validation

The public Figure 3 regeneration path uses only files present in this release:

- `figures/data/h3_domain_loss_figure_data.csv`
- `results/reference/h3/seed_deltas.csv`
- `results/reference/h3/verdict.csv`
- `figures/reference/figure_3.pdf`

The figure-data table contains 18 rows: three targets, three training seeds,
and two variants. MCC, Binary-F1, MMD, and domain discriminability are stored
for every target/seed/variant cell. Domain discriminability is the mean over
the three probe splits of `abs(domain_AUROC - 0.5)`.

The seed-level and target-level cross-check errors are at floating-point
round-off:

- maximum seed-level performance-delta error: `2.220e-16`
- maximum target-mean error: `1.110e-16`

## Released hashes

- `h3_domain_loss_figure_data.csv`:
  `082FE81B28AFC4B1649ACFA59F4E97860B7C10FA1FD23C68F724144A18A8AD05`
- `seed_deltas.csv`:
  `F492DEA0F43F3BF635309EEC499674E3CCC45BE9EBD69EA3B1A56E184F9C23B3`
- `verdict.csv`:
  `164F74263E0EDCA7F67BA8C6C9C2AF2022F10363C14358EC66486DA7531B9BF1`
- `figure_3.pdf`:
  `3A615386F00A9375C4109E531ABB300001E3033B837A86B45C888CF39A636D75`

Internal raw checkpoints, per-row predictions, and evidence-package provenance
material are not included in this public release. They are not required to
regenerate Figure 3 from the released compact data.

## Figure readiness

**Pass.** The three domain-probe splits use the mean over probe splits within each training-seed/variant cell and are not treated as independent training seeds.
