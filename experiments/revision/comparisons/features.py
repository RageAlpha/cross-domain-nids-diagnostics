"""Source-only mRMR and equal-budget shift comparison with frozen LOTO gates."""
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from experiments.revision import common, h2
from experiments.revision.estimators import rank_mrmr
from . import selection
from . import cache

def prepare(args):
    for method in ('mRMR', 'shift_budget'):
        root = args.output_root / method
        rows = []
        for seed in common.SEEDS:
            data = common.source_data(seed) if method == 'mRMR' else None
            masks = {t: {'none': []} for t in common.TARGETS}
            if method == 'mRMR':
                ids = np.sort(np.random.default_rng(seed + 365).choice(data['train_row_ids'], 8192, replace=False))
                pre = h2.gap1.build_preprocessor_for_features(data['feature_cols'])
                pre.fit(data['X_train'])
                names = [n.split('__',1)[1] for n in pre.get_feature_names_out()]
                transformed = pre.transform(data['X_train'].loc[ids])
                x = pd.DataFrame(transformed,columns=names)[common.FEATURES].to_numpy()
                positions = pd.Index(data['train_row_ids']).get_indexer(ids)
                ranking, relevance, constants = rank_mrmr(x, data['y_train'][positions], common.FEATURES)
                ordered = [common.FEATURES[i] for i in ranking]
                audit = {'seed': seed, 'source_row_ids': ids.tolist(), 'source_member_hash': data['membership_hashes']['train'],
                         'ranking': ordered, 'constants': [common.FEATURES[i] for i in constants], 'target_labels_used': False}
                common.write_immutable_json(root / 'rankings' / f'seed{seed}.json', audit)
                for target in common.TARGETS:
                    for k in (5,10):
                        masks[target][f'remove{k}'] = ordered[-k:]
                    common.atomic_csv(pd.DataFrame({'feature': ordered, 'rank': range(1,50)}),root/'h2/shift/rankings'/f'{target}_seed{seed}.csv')
            else:
                previous = json.loads((args.phase2/'h2/masks'/f'seed{seed}.json').read_text())
                for target in common.TARGETS:
                    for k in (5,10):
                        masks[target][f'remove{k}'] = previous[target][f'remove_shift_top{k}']
                    dest = root/'h2/shift/rankings'/f'{target}_seed{seed}.csv'
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(args.phase2/'h2/shift/rankings'/dest.name, dest)
            common.write_immutable_json(root/'h2/masks'/f'seed{seed}.json', masks)
            for model in ('RandomForest','MLP'):
                if method == 'mRMR':
                    for k in (5,10):
                        strategy = f'remove{k}'
                        run_id = f'mRMR/{model}/remove{k}/all_targets/{seed}'
                        rows.extend(cache.evaluate_cached(args,data,seed,model,strategy,masks[common.TARGETS[0]][strategy],
                                                       common.TARGETS,args.output_root/'fits'/run_id,run_id))
                for target in common.TARGETS:
                    old = pd.read_csv(args.phase2/'h2/candidates/by_target'/f'{target}.csv')
                    allowed = ['none'] if method == 'mRMR' else ['none','remove_shift_top5','remove_shift_top10']
                    rows.extend(old[(old.seed==seed)&(old.model==model)&old.filtering_strategy.isin(allowed)].to_dict('records'))
            del data
        frame = pd.DataFrame(rows)
        frame['filtering_strategy'] = frame['filtering_strategy'].replace({'remove_shift_top5':'remove5','remove_shift_top10':'remove10'})
        frame['comparison_method'] = method
        for target in common.TARGETS:
            group=frame[frame.target_dataset==target]
            for model in ('RandomForest','MLP'):
                selection.validate_development_cartesian(group[group.model==model], [target], model)
            common.atomic_csv(group,root/'h2/candidates/by_target'/f'{target}.csv')
        common.atomic_csv(frame,root/'h2/analysis/all_candidates_long.csv')

def controls(args, method_args):
    requests = {}
    policy = json.loads(method_args.policy.read_text())
    policy['_path'] = str(method_args.policy)
    for target in common.TARGETS:
        for model in ('RandomForest','MLP'):
            path=method_args.output_root/'h2/selection_locks'/f'{target}_{model}.json'
            lock=json.loads(path.read_text())
            selection.validate_selection_lock(lock,path,policy,target,model)
            if lock['selected_strategy']=='none':
                continue
            for seed in common.SEEDS:
                k=len(lock['held_out_removed_features_by_seed'][str(seed)])
                requests.setdefault((seed,model,k),[]).append((target,common.hash_file(path)))
    registry=[]
    for seed in common.SEEDS:
        groups=[(model,k,targets) for (s,model,k),targets in requests.items() if s==seed]
        if not groups:
            continue
        data=common.source_data(seed)
        for model,k,targets in groups:
            for rep in range(10):
                removed=h2.random_mask(common.FEATURES,seed,k,rep)
                run_id=(f'random_size/MLP/k{k}_rep{rep:02}/ToN-IoT/{seed}' if model=='MLP' else
                        f'random_conditional/{model}/k{k}_rep{rep:02}/all_targets/{seed}')
                unit=args.output_root/'shared_random'/run_id
                old=args.phase2/'fits'/run_id
                reused=False
                if (old/'TRAINED.json').exists():
                    cache.validate_contract(old,data,seed,model,removed)
                    reused=True
                    cache.import_fit(old,unit)
                names=[t for t,_ in targets]
                if model=='MLP' and 'ToN-IoT' not in names:
                    names.append('ToN-IoT')
                rows=cache.evaluate_cached(args,data,seed,model,'matched_random',removed,names,unit,run_id)
                for target,lock_hash in targets:
                    row=next(dict(r) for r in rows if r['target_dataset']==target)
                    row.update(replicate=rep,random_seed=h2.deterministic_seed(common.PROTOCOL_ID,'random_mask',seed,k,rep),
                               selected_mask_size=k,lock_hash=lock_hash)
                    common.atomic_csv(pd.DataFrame([row]),method_args.output_root/'h2/random/batches'/target/model/f'seed{seed}'/f'rep{rep:02}'/'random_metrics.csv')
                registry.append({'run_id':run_id,'reused_phase2_fit':reused,'targets':names})
        del data
    common.atomic_write_json(method_args.output_root/'h2/random/run_registry.json',{'fits':registry})

def run(args):
    prepare(args)
    for method in ('mRMR','shift_budget'):
        sub=copy.copy(args)
        sub.output_root=args.output_root/method
        sub.policy=args.output_root/'h2_policy.json'
        selection.select_and_lock(sub)
        controls(args,sub)
        selection.analyze(sub)

def mapping(args):
    rows=[]
    for dataset in ['UNSW-NB15',*common.TARGETS]:
        columns=set(pd.read_csv(common.data_path(dataset),nrows=0).columns)
        for feature in common.FEATURES:
            rows.append({'dataset':dataset,'source_feature':feature,'mapped_feature':feature,
                         'present':feature in columns,'mapping':'same named NF-v3 shared feature',
                         'feature_family':h2.shift_metrics.feature_family(feature),
                         'semantic_limit':'Header availability is verified; equivalent acquisition conditions and heterogeneous feature transfer are not established.'})
    if not all(r['present'] for r in rows):
        raise ValueError('Missing required shared feature')
    common.atomic_csv(pd.DataFrame(rows),args.output_root/'feature_mapping_49.csv')
