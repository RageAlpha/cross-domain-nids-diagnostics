# H3 dual-panel data audit

- Protocol: `NIDS-H2H3-CONFIRM-20260713-v2`
- Archived source root in this repository: `results/generated/h2_h3`
- Final manuscript baseline: `latex/main.pdf`, page 10 (Section 5.4), page 11 (Figure 3), and page 12 (Table 4)
- Performance rows: 18 = 3 targets x 3 seeds x 2 variants
- MMD rows: 18 = 3 targets x 3 seeds x 2 variants
- Domain-probe rows: 54 = 3 targets x 3 seeds x 2 variants x 3 probe splits
- Domain discriminability definition: mean over probe splits of `abs(domain_AUROC - 0.5)` for each target/seed/variant
- Maximum performance-delta error versus frozen `paired_per_seed.csv`: 2.220e-16
- Maximum target-mean error versus frozen `verdict_by_target.csv`: 1.110e-16
- Final PDF Table 4 rounded-value checks: Pass for all three targets

## Frozen analysis hashes

- `paired_per_seed.csv`: `F492DEA0F43F3BF635309EEC499674E3CCC45BE9EBD69EA3B1A56E184F9C23B3`
- `verdict_by_target.csv`: `164F74263E0EDCA7F67BA8C6C9C2AF2022F10363C14358EC66486DA7531B9BF1`

## Paper Evidence Package cross-check

- `04_tables_final/04_h3_dann/confirmatory_v2/H3__v2_matched_c1_c0__paired_seedwise.csv`: `F492DEA0F43F3BF635309EEC499674E3CCC45BE9EBD69EA3B1A56E184F9C23B3` (matches the original frozen analysis file)
- `04_tables_final/04_h3_dann/confirmatory_v2/H3__v2_matched_c1_c0_verdict__target.csv`: `164F74263E0EDCA7F67BA8C6C9C2AF2022F10363C14358EC66486DA7531B9BF1` (matches the original frozen analysis file)
- Raw `paired_metrics.csv` files found in the evidence package: 0
- Raw `mmd.csv` files found in the evidence package: 0
- Raw `domain_probe.csv` files found in the evidence package: 0

## Figure readiness

**Pass.** `h3_dual_panel_data.csv` contains the original C0 and C1 values for MCC, Binary-F1, MMD, and domain discriminability for every target and training seed. No C0/C1 value was reconstructed from a delta.

The three domain-probe split seeds are averaged within each training-seed/variant cell and are not treated as independent training seeds.
