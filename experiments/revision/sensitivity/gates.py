"""Retrospective gates. Fixed selections, no training and no reselection."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from experiments.revision import common, h2, h2_analysis, h3
from experiments.h2 import run_loto_filtering as historical_h2
from experiments.h3 import run_paired_domain_loss as historical_h3
from experiments.revision.comparisons import selection, paired
from .design import TOLERANCES, RULES, read_json, three_seeds, finite


def performance(frame, tolerance, rule):
    three_seeds(frame)
    finite(frame, ['MCC', 'Binary_F1', 'FNR'])
    m, f = frame.MCC, frame.Binary_F1
    mean = bool(m.mean() > 0 and f.mean() > 0)
    if rule == 'positive_mean':
        passed = mean
    elif rule == 'mean_and_two':
        passed = mean and (m > 0).sum() >= 2 and (f > 0).sum() >= 2
    elif rule == 'all_three':
        passed = bool((m > 0).all() and (f > 0).all())
    else:
        raise ValueError(f'Unknown frozen seed rule: {rule}')
    return bool(passed and frame.FNR.mean() <= tolerance)


def grid(identity, comparisons, representation=None, abstained=False):
    rows = []
    for tolerance in TOLERANCES:
        for rule in RULES:
            row = {**identity, 'fnr_tolerance': tolerance, 'seed_rule': rule,
                   'primary': tolerance == .01 and rule == 'all_three'}
            if abstained:
                row.update(status='not_applicable_abstention', performance_pass=None,
                           representation_pass=None, joint_pass=None)
            else:
                if not comparisons:
                    raise ValueError('Missing comparisons cannot be treated as a failed gate')
                passes = {name: performance(frame, tolerance, rule) for name, frame in comparisons.items()}
                row.update({f'{name}_pass': value for name, value in passes.items()})
                row.update(status='complete', performance_pass=all(passes.values()),
                           representation_pass=representation,
                           joint_pass=all(passes.values()) and (representation is None or representation))
            rows.append(row)
    return rows


def h2_rows(root, policy_path, module, version, mask_count, method):
    args = SimpleNamespace(output_root=root)
    policy = read_json(policy_path)
    policy['_path'] = str(policy_path)
    if policy['random_replicates_per_seed'] != mask_count:
        raise ValueError('Historical and corrected random-mask budgets must stay separate')
    rows, deltas = [], []
    for target in common.TARGETS:
        candidates = h2.normalize_metrics(pd.read_csv(root/'h2/candidates/by_target'/f'{target}.csv'))
        finite(candidates, historical_h2.METRICS)
        for model in ('RandomForest', 'MLP'):
            frame = candidates[candidates.model == model]
            module.validate_development_cartesian(frame, [target], model)
            lock_path = root/'h2/selection_locks'/f'{target}_{model}.json'
            lock = read_json(lock_path)
            module.validate_selection_lock(lock, lock_path, policy, target, model)
            strategy = lock['selected_strategy']
            chosen = frame[frame.filtering_strategy == strategy].copy()
            baseline = frame[frame.filtering_strategy == 'none'].copy()
            three_seeds(chosen); three_seeds(baseline)
            module.validate_chosen_masks(chosen, lock, target, model)
            identity = dict(analysis_version=version, method=method, target=target, model=model,
                            mask_count=mask_count, selection=strategy, lock_hash=common.hash_file(lock_path))
            comparisons = {}
            random = module.load_complete_random_controls(args, target, model, lock, lock_path, policy)
            if strategy != 'none':
                if random.empty:
                    raise ValueError('Nonempty selection lacks matched random controls')
                finite(random, historical_h2.METRICS)
                controls = {'full': baseline, 'random_median': random.groupby('seed', as_index=False)[historical_h2.METRICS].median()}
                for label, control in controls.items():
                    joined = chosen.merge(control[['seed', *historical_h2.METRICS]], on='seed', suffixes=('', '_control'), validate='one_to_one')
                    delta = pd.DataFrame({'seed': joined.seed, **{out: joined[key]-joined[key+'_control']
                        for out, key in [('MCC', 'mcc'), ('Binary_F1', 'binary_f1'), ('FNR', 'fnr')]}})
                    comparisons[label] = delta
                    deltas.extend({**identity, 'comparator': label, **r} for r in delta.to_dict('records'))
            rows.extend(grid(identity, comparisons, abstained=strategy == 'none'))
    return rows, deltas


def h3_rows(root, policy_path, module, version, method):
    policy = module.load_policy(policy_path)
    coverage = module.validate_h3_coverage(root, policy)
    metrics, mmd, probes = (coverage[k].copy() for k in ('metrics', 'mmd', 'probes'))
    finite(metrics, ['MCC', 'Attack-class Binary-F1', 'FNR'])
    finite(mmd, ['mmd_rbf_unbiased']); finite(probes, ['domain_AUROC'])
    if not probes.domain_AUROC.between(0, 1).all():
        raise ValueError('Invalid domain-probe AUROC')
    probes['discriminability'] = (probes.domain_AUROC-.5).abs()
    rows, deltas = [], []
    for target in policy['targets']:
        frame = metrics[metrics.target == target]
        c0, c1 = [frame[frame.variant == v].set_index('seed').sort_index() for v in ('H3-C0','H3-C1')]
        delta = pd.DataFrame({'seed': c0.index, **{out: (c1[key]-c0[key]).to_numpy()
            for out, key in [('MCC','MCC'), ('Binary_F1','Attack-class Binary-F1'), ('FNR','FNR')]}})
        mp = mmd[mmd.target == target].pivot(index='seed', columns='variant', values='mmd_rbf_unbiased')
        dp = probes[probes.target == target].groupby(['seed','variant']).discriminability.mean().unstack()
        delta['MMD'] = (mp['H3-C1']-mp['H3-C0']).reindex(delta.seed).to_numpy()
        delta['domain_discriminability'] = (dp['H3-C1']-dp['H3-C0']).reindex(delta.seed).to_numpy()
        finite(delta, ['MMD','domain_discriminability'])
        representation = bool(delta.MMD.mean() <= 0 and delta.domain_discriminability.mean() <= 0)
        identity = dict(analysis_version=version, method=method, target=target, model='H3', mask_count=0,
                        mean_mmd_delta=delta.MMD.mean(), mean_domain_discriminability_delta=delta.domain_discriminability.mean())
        rows.extend(grid(identity, {'zero': delta}, representation))
        deltas.extend({**identity, 'comparator':'zero', **r} for r in delta.to_dict('records'))
    return rows, deltas


def run(args, include_strength=False):
    all_rows, all_deltas = [], []
    h2_sources = [
        (args.historical, common.ROOT/'protocols/h2/selection_policy.json', historical_h2, 'historical_20_masks', 20, 'H2'),
        (args.phase2, common.ROOT/'protocols/revision/h2.json', h2_analysis, 'corrected_10_masks', 10, 'H2'),
        *[(args.phase3/m, args.phase3/'h2_policy.json', selection, 'corrected_10_masks', 10, m) for m in ('mRMR','shift_budget')],
    ]
    for source in h2_sources:
        rows, deltas = h2_rows(*source); all_rows.extend(rows); all_deltas.extend(deltas)
    h3_sources = [
        (args.historical, common.ROOT/'protocols/h3/verdict_policy.json', historical_h3, 'historical', 'DANN'),
        (args.phase2, common.ROOT/'protocols/revision/h3.json', h3, 'corrected', 'DANN'),
    ]
    for source in h3_sources:
        rows, deltas = h3_rows(*source); all_rows.extend(rows); all_deltas.extend(deltas)
    for method in ('coral','mmd','FT','BN','weight'):
        paired.configure(method)
        rows, deltas = h3_rows(args.phase3/method, args.phase3/method/'policy.json', paired, 'corrected', method)
        all_rows.extend(rows); all_deltas.extend(deltas)
    paired.configure('coral')
    if include_strength:
        for alpha in (.1, .5):
            root = args.output_root/'strength'/f'alpha{alpha:g}'
            rows, deltas = h3_rows(root, root/'policy.json', h3, 'corrected', f'DANN_alpha{alpha:g}')
            all_rows.extend(rows); all_deltas.extend(deltas)
    frame = pd.DataFrame(all_rows)
    common.atomic_csv(frame, args.output_root/'gates/grid.csv')
    common.atomic_csv(pd.DataFrame(all_deltas), args.output_root/'gates/paired_deltas.csv')
    keys = ['analysis_version','method','target','model']
    stability = []
    for key, group in frame.groupby(keys):
        main = group[group.primary].iloc[0]
        applicable = main.status == 'complete'
        stability.append({**dict(zip(keys,key)), 'status': main.status, 'primary_pass': main.joint_pass,
                          'passing_cells': int(group.joint_pass.sum()) if applicable else None,
                          'total_cells': 12, 'stable_across_grid': group.joint_pass.nunique() == 1 if applicable else None})
    common.atomic_csv(pd.DataFrame(stability), args.output_root/'gates/stability.csv')
    return frame
