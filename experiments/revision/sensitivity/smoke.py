"""Bounded synthetic training, real diagnostic paths and interrupted recovery."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from experiments.revision import common, h3 as base
from experiments.revision.recovery_check import compare_states
from experiments.revision.comparisons.smoke import Identity
from experiments.revision.comparisons.test_review import CacheReview
from experiments.revision.comparisons.test_performance import PerformanceTests
from . import strength
from .design import read_json
from .test_logic import Phase4Tests


def training(root):
    rng=np.random.default_rng(42)
    data={'feature_cols':common.FEATURES,'preprocessor':Identity(),
          'membership_hashes':{'train':'synthetic','val':'synthetic-val','test':'synthetic-test'}}
    for name,n in [('train',514),('val',64),('test',512)]:
        data['X_'+name]=rng.normal(size=(n,49)).astype(np.float32)
        data['y_'+name]=np.arange(n,dtype=np.int64)%2
    target=rng.normal(size=(258,49)).astype(np.float32)
    frame=pd.DataFrame(data['X_test'],columns=common.FEATURES);frame['Label']=data['y_test']
    fixture=root/'target.csv';frame.to_csv(fixture,index=False)
    config=base.H3Settings(batch_size=128,min_epochs=2,patience=0,embedding_sample_size=128,chunk_size=128)
    policy=read_json(common.ROOT/'protocols/revision/h3.json')
    sub=SimpleNamespace(execute=True,chunk_size=128,target='ToN-IoT',seed=42,cpu=False,resume=True,
                        output_root=root/'reference',policy=root/'policy.json')
    common.atomic_write_json(sub.policy,policy)
    sample=(target,np.arange(len(target)),{'target_total_samples':len(frame),'synthetic_only':True})
    with patch.object(base,'h3_source',return_value=data),patch.object(base,'load_target',return_value=sample),patch.object(base,'H3Settings',return_value=config),patch.dict(base.TARGET_DATASETS,{'ToN-IoT':str(fixture)}):
        base.run_pair(sub)
    reference=base.unit_dir(sub.output_root,'ToN-IoT',42)
    checks={}
    for alpha in (.1,.5):
        strength.configure(alpha,'ToN-IoT',42,sub.output_root)
        current=copy.copy(sub);current.output_root=root/f'alpha{alpha:g}'
        with patch.object(strength,'h3_source',return_value=data),patch.object(strength,'load_target',return_value=sample),patch.object(strength,'H3Settings',return_value=config),patch.dict(strength.TARGET_DATASETS,{'ToN-IoT':str(fixture)}):
            strength.run_pair(current)
            # A completed unit must be reusable without entering training/data loading.
            with patch.object(strength,'h3_source',side_effect=AssertionError('Completed pair retrained')):
                strength.run_pair(current)
        dest=strength.unit_dir(current.output_root,'ToN-IoT',42)
        assert common.hash_file(dest/'H3-C0/best.pt')==common.hash_file(reference/'H3-C0/best.pt')
        logs=pd.read_csv(dest/'training_log.csv');active=logs[logs.variant=='H3-C1']
        np.testing.assert_allclose(active['lambda'],alpha*active.epoch/39,atol=1e-16,rtol=0)
        assert set(pd.read_csv(dest/'paired_metrics.csv').run_id)=={
            'H3/DANN/zero/ToN-IoT/42',f'strength/DANN/alpha{alpha:g}/ToN-IoT/42'}
        assert all(read_json(dest/'pair_execution_manifest.json')['checks'].values())
        checks[f'alpha{alpha:g}_complete_pair_exact_zero_reuse_and_reload']=True
        initial=torch.load(reference/'initial.pt',weights_only=True)
        normal=root/f'recovery_{alpha}/normal';resumed=root/f'recovery_{alpha}/resumed'
        normal.mkdir(parents=True);resumed.mkdir(parents=True)
        strength.set_all_seeds(42)
        _,_,normal_logs,normal_meta=strength.train_variant_raw(data,target,initial,config,'H3-C1',42,normal,cpu=False,resume=False,required_epochs=2)
        save=strength.save_training_checkpoint
        class Interrupted(Exception):pass
        def stop(path,**payload):
            save(path,**payload);raise Interrupted()
        strength.set_all_seeds(42)
        with patch.object(strength,'save_training_checkpoint',side_effect=stop):
            try:
                strength.train_variant_raw(data,target,initial,config,'H3-C1',42,resumed,cpu=False,resume=False,required_epochs=2)
            except Interrupted:pass
            else:raise AssertionError('Recovery fixture did not interrupt')
        _,_,resumed_logs,resumed_meta=strength.train_variant_raw(data,target,initial,config,'H3-C1',42,resumed,cpu=False,resume=True,required_epochs=2)
        compare_states(torch.load(normal/'final.pt',weights_only=True),torch.load(resumed/'final.pt',weights_only=True))
        assert normal_logs==resumed_logs and normal_meta==resumed_meta
        checks[f'alpha{alpha:g}_epoch_recovery_exact']=True
    # Ensure incorrect member identity is rejected before importing a C0.
    wrong=copy.deepcopy(data);wrong['membership_hashes']['train']='changed'
    dest=root/'wrong_zero';dest.mkdir()
    try:strength.reuse_zero(wrong,target,initial,config,42,dest,False)
    except (ValueError,RuntimeError):pass
    else:raise AssertionError('Mismatched C0 accepted')
    checks['wrong_zero_source_membership_rejected']=True
    return checks


def run(args,formal_check_passed=False):
    out=args.output_root/'smoke';out.mkdir(parents=True,exist_ok=True)
    (out/'PASS.json').unlink(missing_ok=True)
    attempt=Path(tempfile.mkdtemp(prefix='attempt_',dir=out))
    suite=unittest.TestSuite()
    for cls in (Phase4Tests,CacheReview,PerformanceTests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise ValueError('Phase 4 rule/caching regression checks failed')
    checks=training(attempt)
    from .test_integration import run as integration
    checks.update(integration(attempt/'integration'))
    common.atomic_write_json(out/'PASS.json',{'synthetic_only':True,
        'formal_check_passed':formal_check_passed,'unittest_count':result.testsRun,'checks':checks,
        'attempt':str(attempt),**{name:common.hash_file(args.output_root/name)
                                 for name in ('INPUT_LOCK.json','environment.json','policy.json')}})
    print(f'PHASE 4 {"SMOKE" if formal_check_passed else "SELF-TEST"} PASS: {result.testsRun} regression tests, {len(checks)} integration checks; synthetic only.',flush=True)
