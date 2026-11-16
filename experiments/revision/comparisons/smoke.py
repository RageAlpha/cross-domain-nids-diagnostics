"""Synthetic engineering checks; not formal transfer results."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from experiments.revision import common, h3 as base
from experiments.revision.estimators import numerical_tests
from experiments.revision.estimators import rank_mrmr
from . import paired, selection

class Identity:
    def transform(self, frame):
        return frame.to_numpy(dtype=np.float32)

def run(args):
    out=args.output_root/'smoke'
    out.mkdir(parents=True,exist_ok=True)
    (out/'PASS.json').unlink(missing_ok=True)
    checks=numerical_tests()
    rng=np.random.default_rng(42)
    data={'feature_cols':common.FEATURES,'preprocessor':Identity(),'membership_hashes':{'train':'synthetic','val':'synthetic-val','test':'synthetic-test'}}
    for name,n in [('train',512),('val',64),('test',512)]:
        data['X_'+name]=rng.normal(size=(n,49)).astype(np.float32)
        data['y_'+name]=(np.arange(n)%8==0).astype(np.int64) if name=='train' else np.arange(n,dtype=np.int64)%2
    target=rng.normal(size=(300,49)).astype(np.float32)
    frame=pd.DataFrame(data['X_test'],columns=common.FEATURES)
    frame['Label']=data['y_test']
    fixture=out/'target_fixture.csv';frame.to_csv(fixture,index=False)
    settings=base.H3Settings(batch_size=128,min_epochs=2,patience=0,embedding_sample_size=128,chunk_size=128)
    policy=json.loads((common.ROOT/'protocols/revision/h3.json').read_text())
    sub=SimpleNamespace(execute=True,chunk_size=128,target='ToN-IoT',seed=42,cpu=False,resume=True,
                        output_root=out/'reference',policy=out/'reference_policy.json')
    common.atomic_write_json(sub.policy,policy)
    sample=(target,np.arange(len(target)),{'target_total_samples':len(frame),'synthetic_only':True})
    with patch.object(base,'h3_source',return_value=data),patch.object(base,'load_target',return_value=sample),patch.object(base,'H3Settings',return_value=settings),patch.dict(base.TARGET_DATASETS,{'ToN-IoT':str(fixture)}):
        base.run_pair(sub)
    reference=base.unit_dir(sub.output_root,'ToN-IoT',42)
    for method in ('coral','mmd','FT','BN','weight'):
        paired.configure(method,'ToN-IoT',42,sub.output_root)
        policy['targets']=list(paired.TARGET_DATASETS)
        current=copy.copy(sub);current.output_root=out/method;current.policy=out/f'{method}_policy.json'
        common.atomic_write_json(current.policy,policy)
        config=replace(settings,learning_rate=1e-4 if method=='FT' else .001)
        with patch.object(paired,'h3_source',return_value=data),patch.object(paired,'load_target',return_value=sample),patch.object(paired,'H3Settings',return_value=config),patch.dict(paired.TARGET_DATASETS,{'ToN-IoT':str(fixture)}):
            paired.run_pair(current)
            paired.run_pair(current)
        dest=paired.unit_dir(current.output_root,'ToN-IoT',42)
        logs=pd.read_csv(dest/'training_log.csv')
        assert np.isfinite(logs.source_val_loss).all()
        for variant in ('H3-C0','H3-C1'):
            with np.load(dest/variant/'reservoir_embeddings.npz') as arrays:
                assert arrays['target_embeddings'].shape[1]==(128 if method=='FT' else 64)
        if method in ('coral','mmd'):
            assert common.hash_file(dest/'H3-C0/best.pt')==common.hash_file(reference/'H3-C0/best.pt')
            checks[f'{method}_reuses_exact_phase2_C0']=True
        checks[f'{method}_complete_pair_and_reload']=True
    paired.configure('coral')
    paired.SCALE=1.
    paired.set_all_seeds(42)
    initial=paired.DANN(49,dropout=0,grl_alpha=1).state_dict()
    # Compare the complete zero-weight optimization path with Phase 2.
    for module,name in ((base,'old'),(paired,'new')):
        folder=out/f'zero_equivalence_{name}';folder.mkdir(exist_ok=True)
        module.set_all_seeds(42)
        module.train_variant_raw(data,target,initial,settings,'H3-C0',42,folder,cpu=False,resume=False,required_epochs=2)
    a=torch.load(out/'zero_equivalence_old/final.pt',weights_only=True)
    b=torch.load(out/'zero_equivalence_new/final.pt',weights_only=True)
    for key in a:
        torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    checks['two_epoch_zero_path_exact_state_equality']=True
    # Resume tests include FT dropout RNG and target-updated BN buffers.
    for method in ('mmd','FT','BN','weight'):
        paired.configure(method)
        paired.set_all_seeds(42)
        initial=paired.DANN(49,dropout=0,grl_alpha=1).state_dict()
        config=replace(settings,learning_rate=1e-4 if method=='FT' else .001)
        normal=out/f'recovery_{method}/normal';resumed=out/f'recovery_{method}/resumed'
        normal.mkdir(parents=True,exist_ok=True);resumed.mkdir(parents=True,exist_ok=True)
        paired.set_all_seeds(42)
        paired.train_variant_raw(data,target,initial,config,'H3-C1',42,normal,cpu=False,resume=False,required_epochs=2)
        save=paired.save_training_checkpoint
        class Interrupted(Exception): pass
        def stop(path,**payload):
            save(path,**payload)
            raise Interrupted()
        paired.set_all_seeds(42)
        with patch.object(paired,'save_training_checkpoint',stop):
            try:
                paired.train_variant_raw(data,target,initial,config,'H3-C1',42,resumed,cpu=False,resume=False,required_epochs=2)
            except Interrupted:
                pass
        paired.train_variant_raw(data,target,initial,config,'H3-C1',42,resumed,cpu=False,resume=True,required_epochs=2)
        a=torch.load(normal/'final.pt',weights_only=True);b=torch.load(resumed/'final.pt',weights_only=True)
        for key in a:
            torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
        checks[f'{method}_resume_exact_state']=True
    a=np.tile([0,0,1,1],128);b=np.tile([0,1,0,1],128)
    ranking,_,_=rank_mrmr(np.column_stack([a,a,b,np.zeros(len(a))]),2*a+b,['a','duplicate','b','constant'])
    assert ranking[:2]==[0,2] and ranking[-1]==3
    checks['mrmr_complement_and_constant_rules']=True
    # Equal-budget selector must not open its held-out metric file.
    sargs=SimpleNamespace(output_root=out/'selection',policy=args.output_root/'h2_policy.json',execute=True)
    for target_name in common.TARGETS:
        rows=[]
        for seed in common.SEEDS:
            masks={t:{'none':[],'remove5':common.FEATURES[:5],'remove10':common.FEATURES[:10]} for t in common.TARGETS}
            common.atomic_write_json(sargs.output_root/'h2/masks'/f'seed{seed}.json',masks)
            common.atomic_csv(pd.DataFrame({'feature':common.FEATURES,'rank':range(1,50)}),sargs.output_root/'h2/shift/rankings'/f'{target_name}_seed{seed}.csv')
            for strategy in selection.STRATEGIES:
                rows.append(dict(seed=seed,model='MLP',target_dataset=target_name,filtering_strategy=strategy,
                                 mcc=.6 if strategy=='remove5' else .5,binary_f1=.7 if strategy=='remove5' else .6,fnr=.1,fpr=.1))
        common.atomic_csv(pd.DataFrame(rows),selection.candidate_target_path(sargs,target_name))
    policy=json.loads(sargs.policy.read_text())
    first=selection.select_one(sargs,'ToN-IoT','MLP',policy)
    selection.candidate_target_path(sargs,'ToN-IoT').write_text('Must not be read')
    assert first==selection.select_one(sargs,'ToN-IoT','MLP',policy)
    assert first['selected_strategy']=='remove5'
    checks['two_budget_loto_heldout_isolation']=True
    import unittest
    from .test_performance import PerformanceTests
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(PerformanceTests))
    if not result.wasSuccessful():
        raise ValueError('Optimized loader, BN audit or loss logging check failed')
    checks['optimized_loader_rng_bn_audit_and_loss_logging']=True
    common.atomic_write_json(out/'PASS.json',{'checks':checks,'synthetic_only':True,
        **{name:common.hash_file(args.output_root/name) for name in ('INPUT_LOCK.json','environment.json')}})
    print(f'PHASE 3 SMOKE PASS: {len(checks)} checks; no formal experimental evidence.',flush=True)
