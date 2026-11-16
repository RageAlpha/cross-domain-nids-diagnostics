"""Recompute published revision metrics and gates from compact evidence."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

def verify():
    ROOT = Path(__file__).resolve().parents[2]
    P2 = ROOT / "results/reference/revision/core"
    P3 = ROOT / "results/reference/revision/comparisons"
    P4 = ROOT / "results/reference/revision/sensitivity"
    inputs, checks = {}, []
    
    
    def read(folder, name):
        path = folder / (name + '.csv')
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)
    
    
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    
    
    def equal(a, b):
        return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-7, atol=1e-9)
    
    
    def compare(a, b, keys, cols, label):
        a, b = a.set_index(keys).sort_index(), b.set_index(keys).sort_index()
        check(label, a.index.equals(b.index) and equal(a[cols], b[cols]))
    
    
    def counts(frame, mapping):
        tp, tn, fp, fn = (frame[mapping[k]].astype(float) for k in ['TP', 'TN', 'FP', 'FN'])
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return {'MCC': np.divide(tp * tn - fp * fn, denom, out=np.zeros(len(tp)), where=denom != 0),
                'Binary_F1': 2 * tp / (2 * tp + fp + fn), 'FNR': fn / (tp + fn), 'FPR': fp / (tn + fp)}
    
    
    def metric_check(frame, mapping, colmap, label):
        for key, value in counts(frame, mapping).items():
            check(label + ' ' + key, equal(frame[colmap[key]], value))
    
    
    for folder, expected in [(P2, 16), (P3, 39), (P4, 31)]:
        d = read(folder, 'validation_checks')
        check(folder.name + ' review checks', len(d) == expected and d.status.str.lower().isin(['pass', 'passed']).all())
    
    r2, r3, r4 = [read(p, 'run_inventory') for p in [P2, P3, P4]]
    sets = [set(r.run_id) for r in [r2, r3, r4]]
    check('unique fits and cross-phase reuse', [len(s) for s in sets] == [186, 48, 138]
          and len(sets[0] & sets[2]) == 30 and not (sets[0] & sets[1]) and not (sets[1] & sets[2])
          and len(set.union(*sets)) == 342)
    reg3 = read(P3, 'random_registry')
    check('Phase 3 random references are reused', len(reg3) == 60 and reg3.run_id.nunique() == 30
          and reg3.reused_phase2_fit.all() and set(reg3.run_id).issubset(sets[0]))
    mapping = read(P3, 'feature_mapping')
    check('four shared 49-feature schemas', len(mapping) == 196 and mapping.present.all()
          and mapping.groupby('dataset').mapped_feature.nunique().eq(49).all())
    
    m = ['MCC', 'Binary_F1', 'FNR', 'FPR']
    identity = {k: k for k in ['TP', 'TN', 'FP', 'FN']}
    p3 = read(P3, 'paired_metrics')
    metric_check(p3, identity, {k: k for k in m}, 'Phase 3 confusion counts')
    delta3 = read(P3, 'paired_deltas')
    cols = m + ['MMD', 'domain_discriminability']
    paired = p3.pivot(index=['method', 'target', 'seed'], columns='variant', values=cols)
    recalc = pd.DataFrame({k: paired[k]['H3-C1'] - paired[k]['H3-C0'] for k in cols}).reset_index()
    compare(delta3, recalc, ['method', 'target', 'seed'], cols, '27 paired Phase 3 differences')
    means3 = read(P3, 'paired_means')
    compare(means3, p3.groupby(['method', 'target', 'variant'], as_index=False)[cols].mean(),
            ['method', 'target', 'variant'], cols, 'Phase 3 absolute means')
    g3 = read(P3, 'recomputed_h3_gates')
    expected = delta3.groupby(['method', 'target'], as_index=False)[cols].mean()
    compare(g3, expected.rename(columns={k: 'mean_delta_' + k for k in cols}), ['method', 'target'],
            ['mean_delta_' + k for k in cols], 'Phase 3 gate mean inputs')
    for _, row in g3.iterrows():
        group = delta3[(delta3.method == row.method) & (delta3.target == row.target)]
        perf = (group[['MCC', 'Binary_F1']] > 0).all().all() and group.FNR.mean() <= .01
        rep = (group[['MMD', 'domain_discriminability']].mean() <= 0).all()
        check('Phase 3 gates ' + row.method + '/' + row.target,
              perf == row.performance_gate and rep == row.representation_gate and (perf and rep) == row.joint_gate)
    
    feature = read(P3, 'feature_metrics')
    metric_check(feature, identity, {k: k for k in m}, 'Phase 3 feature confusion counts')
    fd = read(P3, 'feature_deltas')
    fg = read(P3, 'recomputed_h2_gates')
    check('mRMR and shift abstention coverage', len(fg) == 12 and fg.selection.eq('none').sum() == 10)
    
    p2 = read(P2, 'phase2_key_metrics')
    h3 = p2[p2.experiment.eq('H3') & p2.analysis_version.str.startswith('corrected')]
    h3seeds = h3[h3.training_seed.ne('mean')]
    compare(h3[h3.training_seed.eq('mean')], h3seeds.groupby(['target', 'condition'], as_index=False)[cols].mean(),
            ['target', 'condition'], cols, 'Phase 2 H3 means')
    h2g = read(P2, 'recomputed_h2_gates')
    check('Phase 2 H2 four abstentions', len(h2g) == 6 and h2g.selection.eq('none').sum() == 4)
    
    strength = read(P4, 'strength_per_seed')
    sm = read(P4, 'strength_seed_means')
    check('36 strength cells', len(strength) == 36 and not strength.duplicated(['target', 'seed', 'alpha']).any())
    for _, group in strength.groupby(['target', 'seed']):
        zero = group[group.alpha.eq(0)].iloc[0]
        for metric in ['MCC', 'Attack-class Binary-F1', 'FNR']:
            check('strength matched delta ' + str((zero.target, zero.seed, metric)),
                  equal(group['delta_' + metric], group[metric] - zero[metric]))
    compare(sm, strength.groupby(['target', 'alpha'], as_index=False)[list(sm.columns[2:])].mean(),
            ['target', 'alpha'], list(sm.columns[2:]), 'strength means')
    check('strength zero/original detection agrees with Phase 2', all(
        equal(strength[strength.alpha.eq(alpha)].sort_values(['target', 'seed'])[['MCC', 'Attack-class Binary-F1', 'FNR']],
              h3seeds[h3seeds.condition.eq(condition)].assign(seed=lambda d: d.training_seed.astype(int)).sort_values(['target', 'seed'])[['MCC', 'Binary_F1', 'FNR']])
        for alpha, condition in [(0, 'H3-C0'), (1, 'H3-C1')]))
    
    random = read(P4, 'random_per_mask')
    metric_check(random, {k: k.lower() for k in identity},
                 {'MCC': 'mcc', 'Binary_F1': 'binary_f1', 'FNR': 'fnr', 'FPR': 'fpr'}, '120 random counts')
    check('random origins', random.origin.value_counts().to_dict() == {'phase4_new': 90, 'reused': 30})
    dist_rows = []
    for (seed, k), group in random.groupby(['seed', 'k']):
        check('random ten masks ' + str((seed, k)), len(group) == 10 and set(group.replicate) == set(range(10)))
        for metric in ['mcc', 'binary_f1', 'fnr']:
            values, changes = group[metric], group['delta_' + metric]
            check('random delta ' + str((seed, k, metric)), equal(changes, values - group[metric + '_full']))
            dist_rows.append(dict(seed=seed, k=k, metric=metric, mean=values.mean(), median=values.median(),
                                  min=values.min(), max=values.max(), q25=values.quantile(.25), q75=values.quantile(.75),
                                  delta_mean=changes.mean(), delta_median=changes.median()))
    dist = pd.DataFrame(dist_rows)
    compare(read(P4, 'random_distributions'), dist, ['seed', 'k', 'metric'], list(dist.columns[3:]), 'random distributions')
    rm = read(P4, 'random_seed_means')
    compare(rm, dist.groupby(['k', 'metric'], as_index=False)[['mean', 'median', 'delta_mean', 'delta_median']].mean(),
            ['k', 'metric'], ['mean', 'median', 'delta_mean', 'delta_median'], 'nested random means')
    
    ro = read(P4, 'readout_metrics')
    metric_check(ro, identity, {k: k for k in m}, 'readout counts')
    check('readout coverage and convergence', len(ro) == 12 and ro.fit_converged.all()
          and ro[list(identity)].sum(axis=1).eq(6000).all())
    ov = read(P4, 'readout_overlap')
    check('readout group independence', len(ov) == 9 and ov[['shared_groups', 'train_rows_in_overlap', 'test_rows_in_overlap']].eq(0).all().all())
    rom = read(P4, 'readout_seed_means')
    compare(rom.rename(columns={'Attack-class Binary-F1': 'Binary_F1'}), ro.groupby(['variant', 'evaluation'], as_index=False)[m].mean(),
            ['variant', 'evaluation'], m[:3], 'readout means')
    
    grid, gd = read(P4, 'recomputed_gate_grid'), read(P4, 'gate_deltas')
    keys = ['analysis_version', 'method', 'target', 'model']
    check('gate coverage', len(grid) == 540 and len(grid[keys].drop_duplicates()) == 45)
    for key, group in grid.groupby(keys):
        d = gd.copy()
        for col, val in zip(keys, key):
            d = d[d[col].eq(val)]
        for _, cell in group.iterrows():
            if cell.status == 'not_applicable_abstention':
                check('abstention ' + str(key), d.empty and pd.isna(cell.joint_pass))
                continue
            passed = True
            for _, dg in d.groupby('comparator'):
                check('three seed gate inputs ' + str(key), len(dg) == 3 and set(dg.seed) == {42, 123, 2024})
                mean_ok = (dg[['MCC', 'Binary_F1']].mean() > 0).all()
                n = (dg[['MCC', 'Binary_F1']] > 0).sum().min()
                score = mean_ok if cell.seed_rule == 'positive_mean' else (mean_ok and n >= 2) if cell.seed_rule == 'mean_and_two' else n == 3
                passed = passed and score and dg.FNR.mean() <= cell.fnr_tolerance
            rep = (d[['MMD', 'domain_discriminability']].mean() <= 0).all() if key[3] == 'H3' else True
            check('gate cell ' + str((*key, cell.fnr_tolerance, cell.seed_rule)),
                  bool(passed) == cell.performance_pass and bool(passed and rep) == cell.joint_pass)
    check('gate totals', grid.joint_pass.eq(True).sum() == 12 and grid.joint_pass.eq(False).sum() == 324 and grid.joint_pass.isna().sum() == 204)
    return {"checks": len(checks), "source_tables": len(inputs)}
