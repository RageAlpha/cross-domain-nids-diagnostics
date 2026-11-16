"""Collect Phase 2 evidence without automatically claiming support for a paper."""
import pandas as pd
from .common import atomic_csv, atomic_write_json, PROTOCOL_ID, PREP

def paired_figure(frame, group_columns, metrics, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = list(frame.groupby(group_columns, sort=True))
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.8))
    for ax, (column, label) in zip(axes, metrics):
        for index, (group_name, group) in enumerate(groups):
            values = group[column].to_numpy()
            ax.scatter([index] * len(values), values, s=22, alpha=.65, color='#3274a1')
            ax.scatter(index, values.mean(), marker='_', s=180, color='black')
        labels = [' / '.join(map(str, name)) if isinstance(name, tuple) else str(name) for name, _ in groups]
        ax.set_xticks(range(len(groups)), labels, rotation=30, ha='right')
        ax.axhline(0, color='gray', linestyle='--', linewidth=.8)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(title + ' (dots: seeds; black: mean)')
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=200)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)

def report(root):
    fixed = pd.read_csv(PREP / 'formal_matrix.csv')
    candidates = pd.read_csv(root / 'h2/analysis/all_candidates_long.csv')
    paired_metrics = pd.concat([pd.read_csv(p) for p in (root / 'h3/runs').glob('*/seed*/paired_metrics.csv')], ignore_index=True)
    if set(candidates['run_id']) != set(fixed.loc[fixed['package'] == 'H2', 'run_id']):
        raise ValueError('H2 fits do not cover the approved 108 fixed cells')
    if set(paired_metrics['run_id']) != set(fixed.loc[fixed['package'] == 'H3', 'run_id']):
        raise ValueError('H3 fits do not cover the approved 18 fixed cells')
    tables = root / 'tables_figures'
    tables.mkdir(exist_ok=True)
    selected = pd.read_csv(root / 'h2/analysis/loto_selected_vs_e2_seedwise.csv')
    atomic_csv(selected.groupby(['target_dataset', 'model'])[['mcc', 'binary_f1', 'fnr', 'fpr']].agg(['mean', 'std']).reset_index(),
               tables / 'h2_selected_mean_std.csv')
    atomic_csv(paired_metrics.groupby(['target', 'variant'])[['MCC', 'Attack-class Binary-F1', 'FNR', 'FPR']].agg(['mean', 'std']).reset_index(),
               tables / 'h3_mean_std.csv')
    paired_figure(selected, ['target_dataset', 'model'], [(f'delta_{m}_vs_e2', label) for m, label in
                  [('mcc', 'MCC delta'), ('binary_f1', 'Binary F1 delta'), ('fnr', 'FNR delta')]],
                  tables / 'h2_selected_vs_full', 'H2 selected - full features')
    paired_figure(pd.read_csv(root / 'h3/analysis/paired_per_seed.csv'), ['target'],
                  [('delta_MCC', 'MCC delta'), ('delta_Attack-class Binary-F1', 'Binary F1 delta'), ('delta_FNR', 'FNR delta')],
                  tables / 'h3_c1_minus_c0', 'H3 C1 - C0')
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
                      'next_stage': 'Review conclusion impact, then implement/run Phase 3 under the approved protocol.'})
