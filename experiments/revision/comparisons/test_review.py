"""Regression checks for imported fit identity and interrupted cache copies."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.revision import common,h2
from . import cache

class CacheReview(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='phase3_review_',dir=common.ROOT/'results/generated')
        self.root=Path(self.temp.name)
        self.source=self.root/'source';self.source.mkdir()
        self.data={'feature_cols':['a','b','c'],'membership_hashes':{'train':'train-42','val':'val-42','test':'test-42'}}
        self.contract={'protocol_id':common.PROTOCOL_ID,'seed':42,'model':'MLP','removed':['a'],'kept':['b','c'],
                       'source_membership':self.data['membership_hashes'],'smoke_only':False}
        common.atomic_write_json(self.source/'fit_contract.json',self.contract)
        (self.source/'model.joblib').write_bytes(b'fixture-model')
        h2.receipt_for(self.source/'TRAINED.json',[self.source/'model.joblib',self.source/'fit_contract.json'],
                       contract_hash=common.hash_file(self.source/'fit_contract.json'))

    def tearDown(self):
        self.temp.cleanup()

    def test_correct_contract_accepted(self):
        cache.validate_contract(self.source,self.data,42,'MLP',['a'])

    def test_wrong_mask_model_seed_membership_rejected(self):
        changed=copy.deepcopy(self.data);changed['membership_hashes']['train']='other'
        for data,seed,model,removed in [(self.data,42,'MLP',['b']),(self.data,123,'MLP',['a']),
                                         (self.data,42,'RandomForest',['a']),(changed,42,'MLP',['a'])]:
            with self.subTest(seed=seed,model=model,removed=removed,data=data):
                with self.assertRaises(ValueError):cache.validate_contract(self.source,data,seed,model,removed)

    def test_interrupted_copy_retries_without_partial_destination(self):
        dest=self.root/'destination'
        def interrupted(source,stage,**kwargs):
            stage.mkdir(parents=True,exist_ok=True)
            (stage/'model.joblib').write_bytes(b'partial')
            raise OSError('simulated interrupted copy')
        with patch.object(cache.shutil,'copytree',side_effect=interrupted):
            with self.assertRaises(OSError):cache.import_fit(self.source,dest)
        self.assertFalse(dest.exists())
        cache.import_fit(self.source,dest)
        cache.validate_contract(dest,self.data,42,'MLP',['a'])
        self.assertEqual((dest/'model.joblib').read_bytes(),b'fixture-model')

    def test_corrupted_model_rejected(self):
        (self.source/'model.joblib').write_bytes(b'changed')
        with self.assertRaises(ValueError):cache.validate_contract(self.source,self.data,42,'MLP',['a'])

    def test_wrong_cached_result_identity_rejected(self):
        row={'run_id':'expected','seed':42,'model':'RandomForest','target_dataset':'ToN-IoT',
             'removed_features':'["a"]','mask_hash':common.hash_json(['a'])}
        with patch.object(h2,'evaluate_cached',return_value=[row]):
            with self.assertRaises(ValueError):
                cache.evaluate_cached(None,self.data,42,'MLP','matched_random',['a'],['ToN-IoT'],self.source,'expected')

if __name__=='__main__':
    unittest.main()
