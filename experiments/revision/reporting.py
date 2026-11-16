"""Collect Phase 2 evidence without automatically claiming support for a paper."""
import pandas as pd
from .common import atomic_csv, atomic_write_json, PROTOCOL_ID, PREP


def report(root):
    fixed = pd.read_csv(PREP / 'formal_matrix.csv')
    candidates = pd.read_csv(root / 'h2/analysis/all_candidates_long.csv')
    paired_metrics = pd.concat([pd.read_csv(p) for p in (root / 'h3/runs').glob('*/seed*/paired_metrics.csv')], ignore_index=True)
    if set(candidates['run_id']) != set(fixed.loc[fixed['package'] == 'H2', 'run_id']):
        raise ValueError('H2 fits do not cover the approved 108 fixed cells')
    if set(paired_metrics['run_id']) != set(fixed.loc[fixed['package'] == 'H3', 'run_id']):
        raise ValueError('H3 fits do not cover the approved 18 fixed cells')
    tables = root / 'tables'
    tables.mkdir(exist_ok=True)
    selected = pd.read_csv(root / 'h2/analysis/loto_selected_vs_e2_seedwise.csv')
    atomic_csv(selected.groupby(['target_dataset', 'model'])[['mcc', 'binary_f1', 'fnr', 'fpr']].agg(['mean', 'std']).reset_index(),
               tables / 'h2_selected_mean_std.csv')
    atomic_csv(paired_metrics.groupby(['target', 'variant'])[['MCC', 'Attack-class Binary-F1', 'FNR', 'FPR']].agg(['mean', 'std']).reset_index(),
               tables / 'h3_mean_std.csv')
    h2 = pd.read_csv(root / 'h2/analysis/verdict_by_target_model.csv')
    h3 = pd.read_csv(root / 'h3/analysis/verdict_by_target.csv')
    rows = []
    for _, row in h2.iterrows():
        rows.append({'experiment': 'H2', 'target': row['target'], 'model': row['model'], 'gate': row['verdict']})
    for _, row in h3.iterrows():
        rows.append({'experiment': 'H3', 'target': row['target'], 'model': 'DANN', 'gate': row['verdict']})
    for row in rows:
        row.update(original_claim='', conclusion_impact='Assess against the manuscript: retained/strengthened/weakened/insufficient evidence',
                   supporting_table=f'{row["experiment"].lower()}/analysis', manuscript_change='')
    atomic_csv(pd.DataFrame(rows), root / 'conclusion_review.csv')
    atomic_write_json(root / 'PHASE2_COMPLETE.json', {'protocol_id': PROTOCOL_ID, 'status': 'complete',
                      'scope': 'Phase 2 H2/H3 and required matched random controls',
                      'fixed_h2_fits': candidates['run_id'].nunique(), 'fixed_h3_fits': paired_metrics['run_id'].nunique(),
                      'next_stage': 'Inspect the H2/H3 metric tables and verdicts.'})
