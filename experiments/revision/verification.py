"""Small artificial fixtures test selection isolation and fail-closed analysis."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
from . import common, h2, h2_analysis, h3
from experiments.h2.run_loto_filtering import STRATEGIES, validate_development_cartesian

def expect_rejection(action):
    try:
        action()
    except (ValueError, RuntimeError, FileNotFoundError):
        return
    raise AssertionError('Invalid/incomplete evidence was accepted')

def verify_logic(root):
    root.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(output_root=root, policy=common.ROOT / 'protocols/revision/h2.json', execute=True)
    policy = json.loads(args.policy.read_text())
    frames = {}
    for target in common.TARGETS:
        rows = []
        for model in ('RandomForest', 'MLP'):
            for seed in common.SEEDS:
                masks = {t: {strategy: ([] if strategy == 'none' else common.FEATURES[:3]) for strategy in STRATEGIES} for t in common.TARGETS}
                common.atomic_write_json(root / 'h2/masks' / f'seed{seed}.json', masks)
                common.atomic_csv(pd.DataFrame({'feature': common.FEATURES, 'rank': np.arange(1, 50)}),
                                  root / 'h2/shift/rankings' / f'{target}_seed{seed}.csv')
                for strategy in STRATEGIES:
                    winner = strategy == 'remove_shift_top3'
                    removed = masks[target][strategy]
                    rows.append(dict(protocol_id=common.PROTOCOL_ID, seed=seed, model=model, target_dataset=target,
                                     filtering_strategy=strategy, mcc=.4 + .1 * winner, binary_f1=.5 + .1 * winner,
                                     fnr=.2 - .01 * winner, fpr=.1, removed_features=json.dumps(removed),
                                     mask_hash=common.hash_json(removed)))
        frame = pd.DataFrame(rows)
        common.atomic_csv(frame, h2.candidate_target_path(args, target))
        frames[target] = frame
    original = h2_analysis.select_one(args, 'ToN-IoT', 'MLP', policy)
    assert original['selected_strategy'] == 'remove_shift_top3'
    held_path = h2.candidate_target_path(args, 'ToN-IoT')
    original_text = held_path.read_text()
    held_path.write_text('THIS HELD-OUT FILE MUST NOT BE OPENED', encoding='utf-8')
    repeated = h2_analysis.select_one(args, 'ToN-IoT', 'MLP', policy)
    assert original == repeated
    held_path.write_text(original_text, encoding='utf-8')
    h2_analysis.select_and_lock(args)
    lock_path = root / 'h2/selection_locks/ToN-IoT_MLP.json'
    lock = json.loads(lock_path.read_text())
    policy['_path'] = str(args.policy)
    h2_analysis.validate_selection_lock(lock, lock_path, policy, 'ToN-IoT', 'MLP')
    for seed in common.SEEDS:
        for replicate in range(10):
            removed = h2.random_mask(common.FEATURES, seed, 3, replicate)
            row = dict(protocol_id=common.PROTOCOL_ID, model='MLP', target_dataset='ToN-IoT', seed=seed,
                       filtering_strategy='matched_random', replicate=replicate,
                       random_seed=h2.deterministic_seed(common.PROTOCOL_ID, 'random_mask', seed, 3, replicate),
                       removed_features=json.dumps(removed), mask_hash=common.hash_json(removed), selected_mask_size=3,
                       lock_hash=common.hash_file(lock_path), mcc=.4, binary_f1=.5, fnr=.2, fpr=.1)
            common.atomic_csv(pd.DataFrame([row]), root / f'h2/random/batches/ToN-IoT/MLP/seed{seed}/rep{replicate:02}/random_metrics.csv')
    complete = h2_analysis.load_complete_random_controls(args, 'ToN-IoT', 'MLP', lock, lock_path, policy)
    assert len(complete) == 30
    path = root / 'h2/random/batches/ToN-IoT/MLP/seed42/rep09/random_metrics.csv'
    content = path.read_text()
    path.unlink()
    expect_rejection(lambda: h2_analysis.load_complete_random_controls(args, 'ToN-IoT', 'MLP', lock, lock_path, policy))
    path.write_text(content, encoding='utf-8')
    frame = frames['CICIDS2018']
    group = frame[frame['model'] == 'MLP']
    expect_rejection(lambda: validate_development_cartesian(group.iloc[:-1], ['CICIDS2018'], 'MLP'))
    expect_rejection(lambda: validate_development_cartesian(pd.concat([group, group.iloc[:1]]), ['CICIDS2018'], 'MLP'))
    expect_rejection(lambda: h3.validate_h3_coverage(root, h3.load_policy(common.ROOT / 'protocols/revision/h3.json')))
    return {'loto_held_out_metrics_isolated': True, 'ten_masks_per_seed_required': True,
            'missing_or_duplicate_candidates_rejected': True, 'incomplete_h3_rejected': True}
