"""Bounded integration fixture for source-only selection and shared controls."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.revision import common,h2
from . import features,selection

def run(root):
    args=SimpleNamespace(output_root=root/'out',phase2=root/'previous',chunk_size=128,execute=True)
    rng=np.random.default_rng(617)
    data={}
    for seed in common.SEEDS:
        x=pd.DataFrame(rng.normal(size=(8192,49)),columns=common.FEATURES)
        x['PROTOCOL']=rng.integers(0,3,len(x));x['L7_PROTO']=rng.integers(0,4,len(x))
        data[seed]={'X_train':x,'y_train':(x.iloc[:,0]>0).to_numpy(dtype=np.int64),
                    'feature_cols':common.FEATURES,'train_row_ids':np.arange(len(x)),
                    'membership_hashes':{'train':f'fixture{seed}'}}
        masks={t:{'none':[],'remove_shift_top5':common.FEATURES[:5],'remove_shift_top10':common.FEATURES[:10]} for t in common.TARGETS}
        common.atomic_write_json(args.phase2/'h2/masks'/f'seed{seed}.json',masks)
        for target in common.TARGETS:
            common.atomic_csv(pd.DataFrame({'feature':common.FEATURES,'rank':range(1,50)}),args.phase2/'h2/shift/rankings'/f'{target}_seed{seed}.csv')
    for target in common.TARGETS:
        rows=[]
        for seed in common.SEEDS:
            for model in ('RandomForest','MLP'):
                for strategy,k in [('none',0),('remove_shift_top5',5),('remove_shift_top10',10)]:
                    rows.append(dict(seed=seed,model=model,target_dataset=target,filtering_strategy=strategy,
                                     mcc=.5+.05*(k==5),binary_f1=.6+.05*(k==5),fnr=.2,fpr=.1,
                                     removed_features=json.dumps(common.FEATURES[:k]),mask_hash=common.hash_json(common.FEATURES[:k]),run_id=f'fixture/{target}/{model}/{seed}/{k}'))
        common.atomic_csv(pd.DataFrame(rows),args.phase2/'h2/candidates/by_target'/f'{target}.csv')
    policy=json.loads((common.ROOT/'protocols/revision/h2.json').read_text())
    common.atomic_write_json(args.output_root/'h2_policy.json',policy)
    calls=[]
    def evaluate(args,data,seed,model,strategy,removed,targets,unit,run_id):
        calls.append((run_id,tuple(targets)))
        value=.7 if strategy=='remove5' else .5
        return [dict(seed=seed,model=model,target_dataset=t,filtering_strategy=strategy,mcc=value,binary_f1=value,
                     fnr=.2,fpr=.1,protocol_id=common.PROTOCOL_ID,run_id=run_id,
                     removed_features=json.dumps(removed),mask_hash=common.hash_json(removed)) for t in targets]
    with patch.object(common,'source_data',side_effect=lambda s:data[s]),patch.object(h2,'evaluate_cached',side_effect=evaluate):
        features.run(args)
    for method in ('mRMR','shift_budget'):
        table=pd.read_csv(args.output_root/method/'h2/analysis/verdict_by_target_model.csv')
        assert len(table)==6
        for target in common.TARGETS:
            for model in ('RandomForest','MLP'):
                lock=json.loads((args.output_root/method/'h2/selection_locks'/f'{target}_{model}.json').read_text())
                assert lock['selected_strategy']=='remove5'
                assert lock['held_out_metrics_loaded'] is False
    fits={run_id for run_id,_ in calls if run_id.startswith('mRMR/')}
    assert len(fits)==12
    assert set(selection.STRATEGIES)=={'none','remove5','remove10'}
    print('FEATURE INTEGRATION PASS: 12 mRMR cells, separate 2-budget selectors, 10-mask controls and verdict outputs.')

if __name__=='__main__':
    run(common.ROOT/'tmp/phase3_feature_fixture')
