"""Scientific-rule edge cases and corruption/reuse regression checks."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.revision import common, h2
from experiments.revision.comparisons import cache
from . import gates, readout, random_size, verification
from .design import matrix, independent_output, POLICY


class Phase4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='phase4_test_',dir=common.ROOT/'results/generated')
        self.root=Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def delta(self,m=(.1,.1,.1),f=(.1,.1,.1),n=(.01,.01,.01)):
        return pd.DataFrame({'seed':common.SEEDS,'MCC':m,'Binary_F1':f,'FNR':n})

    def test_grid_boundary_and_strict_improvement(self):
        rows=gates.grid({}, {'full':self.delta(),'random':self.delta()})
        self.assertEqual(len(rows),12)
        self.assertEqual(sum(r['primary'] for r in rows),1)
        self.assertTrue(next(r for r in rows if r['primary'])['joint_pass'])
        self.assertFalse(gates.performance(self.delta(m=(0,.1,.1)),.01,'all_three'))
        self.assertFalse(gates.performance(self.delta(n=(.01001,)*3),.01,'all_three'))

    def test_majority_metrics_need_not_share_seed_pair(self):
        frame=self.delta(m=(.2,.2,-.1),f=(-.1,.2,.2))
        self.assertTrue(gates.performance(frame,.01,'mean_and_two'))
        self.assertFalse(gates.performance(frame,.01,'all_three'))
        self.assertFalse(gates.performance(self.delta(m=(.1,.1,-.5)),.01,'mean_and_two'))

    def test_representation_gate_cannot_be_bypassed(self):
        rows=gates.grid({}, {'zero':self.delta()},representation=False)
        self.assertFalse(any(r['joint_pass'] for r in rows))

    def test_abstention_is_na(self):
        rows=gates.grid({}, {},abstained=True)
        self.assertTrue(all(r['joint_pass'] is None and r['status']=='not_applicable_abstention' for r in rows))
        with self.assertRaises(ValueError):gates.grid({}, {})

    def test_missing_duplicate_nonfinite_seeds_fail(self):
        for frame in (self.delta().iloc[:2],self.delta().assign(seed=[42,42,2024]),self.delta(m=(np.nan,.1,.1))):
            with self.assertRaises(ValueError):gates.performance(frame,.01,'all_three')

    def test_frozen_matrix_and_no_optional_work(self):
        frame=matrix()
        self.assertEqual(len(frame),138)
        self.assertEqual((frame.package=='random_size').sum(),120)
        self.assertFalse(POLICY['multi_source'])

    def test_protected_output_rejected(self):
        base=common.ROOT/'results/protocol_test'
        for dest in (base,base/'child',base.parent):
            with self.assertRaises(ValueError):independent_output(dest,[base])

    def test_session_split_and_three_overlap_audits(self):
        raw=pd.DataFrame({name:np.repeat(np.arange(100),2) for name in readout.FLOW_COLUMNS})
        raw['value']=np.arange(200)
        train,test,audit=readout.split_and_audit(raw)
        self.assertTrue(np.array_equal(train[::2],train[1::2]))
        self.assertEqual(train.sum(),140);self.assertEqual(len(audit),3)
        self.assertTrue(all(r['shared_groups']==0 for r in audit))
        t2,_,_=readout.split_and_audit(raw)
        np.testing.assert_array_equal(train,t2)

    def test_readout_scaler_only_fits_training_data(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(100,4));y=np.arange(100)%2
        train=np.arange(100)<70;test=~train;x[test]+=100
        model,pred,converged=readout.fit_readout(x,y,train,test)
        np.testing.assert_allclose(model[0].mean_,x[train].mean(axis=0))
        self.assertEqual(len(pred),30);self.assertTrue(converged)
        with self.assertRaises(ValueError):readout.fit_readout(x,np.zeros(100),train,test)

    def test_raw_row_reader_preserves_order_and_excludes_labels(self):
        frame=pd.DataFrame({name:np.arange(12) for name in readout.FLOW_COLUMNS})
        frame['Label']='must_not_parse';frame['Attack']='must_not_parse';frame['L7_PROTO']=1.2
        path=self.root/'raw.csv';frame.to_csv(path,index=False)
        out=readout.raw_rows(path,np.array([10,2,7]),chunk_size=4)
        self.assertEqual(out.index.tolist(),[10,2,7]);self.assertNotIn('Label',out)
        self.assertTrue((out.L7_PROTO==1.2).all())

    def create_fit(self,root,content=b'model'):
        root.mkdir(parents=True)
        data={'feature_cols':['a','b'],'membership_hashes':{'train':'train','val':'val','test':'test'}}
        common.atomic_write_json(root/'fit_contract.json',{'protocol_id':common.PROTOCOL_ID,'seed':42,'model':'MLP',
            'removed':['a'],'kept':['b'],'source_membership':data['membership_hashes'],'smoke_only':False})
        (root/'model.joblib').write_bytes(content)
        h2.receipt_for(root/'TRAINED.json',[root/'fit_contract.json',root/'model.joblib'],contract_hash=common.hash_file(root/'fit_contract.json'))
        return data

    def test_cross_phase_reuse_and_conflict_detection(self):
        args=SimpleNamespace(phase2=self.root/'p2',phase3=self.root/'p3')
        data=self.create_fit(args.phase2/'fits/cell')
        self.create_fit(args.phase3/'shared_random/cell')
        found=random_size.compatible_candidates(args,'cell',data,42,['a'])
        self.assertEqual(len(found),2)
        other=found[1];(other/'model.joblib').write_bytes(b'other')
        h2.receipt_for(other/'TRAINED.json',[other/'fit_contract.json',other/'model.joblib'],contract_hash=common.hash_file(other/'fit_contract.json'))
        with self.assertRaises(ValueError):random_size.compatible_candidates(args,'cell',data,42,['a'])

    def test_incomplete_upstream_fit_cannot_trigger_retraining(self):
        args=SimpleNamespace(phase2=self.root/'p2',phase3=self.root/'p3')
        (args.phase2/'fits/cell').mkdir(parents=True)
        data={'feature_cols':['a','b'],'membership_hashes':{'train':'fixture'}}
        with self.assertRaises(FileNotFoundError):random_size.compatible_candidates(args,'cell',data,42,['a'])

    def test_transient_copy_lock_retries_and_persistent_error_surfaces(self):
        source=self.root/'source';data=self.create_fit(source)
        error=PermissionError('synthetic Windows file lock');error.winerror=32
        original=cache.import_fit;calls=[]
        def transient(a,b):
            calls.append(1)
            if len(calls)==1:raise error
            return original(a,b)
        with patch.object(cache,'import_fit',side_effect=transient),patch.object(random_size.time,'sleep'):
            random_size.import_fit(source,self.root/'destination')
        self.assertEqual(len(calls),2)
        cache.validate_contract(self.root/'destination',data,42,'MLP',['a'])
        with patch.object(cache,'import_fit',side_effect=error) as mocked,patch.object(random_size.time,'sleep'):
            with self.assertRaises(PermissionError):random_size.import_fit(source,self.root/'blocked')
            self.assertEqual(mocked.call_count,5)

    def test_missing_stage_completion_blocks_formal_run(self):
        args=SimpleNamespace(phase2=self.root/'p2',phase3=self.root/'p3')
        common.atomic_write_json(args.phase2/'PHASE2_COMPLETE.json',{'protocol_id':common.PROTOCOL_ID,'status':'complete','fixed_h2_fits':108,'fixed_h3_fits':18})
        common.atomic_write_json(args.phase3/'PHASE3_COMPLETE.json',{'protocol_id':common.PROTOCOL_ID,'status':'running','fixed_fits':48})
        with self.assertRaises(ValueError):verification.validate_stage_markers(args)

    def test_self_test_cannot_authorize_formal_run(self):
        args=SimpleNamespace(output_root=self.root)
        for name in ('INPUT_LOCK.json','environment.json','policy.json'):
            common.atomic_write_json(self.root/name,{'fixture':name})
        record={name:common.hash_file(self.root/name) for name in ('INPUT_LOCK.json','environment.json','policy.json')}
        common.atomic_write_json(self.root/'smoke/PASS.json',{**record,'formal_check_passed':False})
        with self.assertRaises(ValueError):verification.verify_smoke(args)
        common.atomic_write_json(self.root/'smoke/PASS.json',{**record,'formal_check_passed':True})
        verification.verify_smoke(args)
        common.atomic_write_json(self.root/'environment.json',{'fixture':'changed'})
        with self.assertRaises(ValueError):verification.verify_smoke(args)


if __name__=='__main__':
    unittest.main()
