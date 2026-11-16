"""Synthetic orchestration fixtures; all counts/metrics here are artificial."""
import json
from types import SimpleNamespace
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from experiments.revision import common,h2,estimators as mlp
from experiments.revision.recovery_check import compare_states,SimulatedInterruption
from experiments.revision.comparisons.test_features import run as feature_fixture
from experiments.revision.comparisons import selection
from . import gates,random_size,readout,reporting
from .design import matrix,read_json


def run(root):
    root.mkdir(parents=True,exist_ok=True)
    checks={}
    # Real two-epoch MLP recovery, including optimizer/RNG and explicit validation.
    rng=np.random.default_rng(819)
    x=rng.normal(size=(258,49)).astype(np.float32);v=rng.normal(size=(64,49)).astype(np.float32)
    y=np.arange(len(x))%2;vy=np.arange(len(v))%2
    def estimator(name):
        folder=root/name;folder.mkdir()
        model=mlp.RevisionMLP(max_epochs=2,batch_size=128,random_state=42,require_gpu=False)
        model.checkpoint_dir=str(folder);return model
    normal=estimator('mlp_normal').fit(x,y,validation_data=(v,vy))
    resumed=estimator('mlp_resumed');save=mlp.save_training_checkpoint
    def stop(path,**payload):
        save(path,**payload);raise SimulatedInterruption()
    with patch.object(mlp,'save_training_checkpoint',side_effect=stop):
        try:resumed.fit(x,y,validation_data=(v,vy))
        except SimulatedInterruption:pass
        else:raise AssertionError('MLP did not interrupt')
    resumed.fit(x,y,validation_data=(v,vy))
    compare_states(normal.model_.state_dict(),resumed.model_.state_dict())
    assert normal.history_==resumed.history_
    checks['shared_random_mlp_epoch_recovery_exact']=True

    feature_fixture(root/'features')
    for method in ('mRMR','shift_budget'):
        rows,deltas=gates.h2_rows(root/'features/out'/method,root/'features/out/h2_policy.json',
                                 selection,'synthetic_corrected_10_masks',10,method)
        assert len(rows)==72 and len(deltas)==36
    checks['fixed_two_budget_h2_gates_read_existing_locks']=True

    args=SimpleNamespace(output_root=root/'output',phase2=root/'phase2',phase3=root/'phase3',chunk_size=128)
    data={s:{'feature_cols':common.FEATURES,'membership_hashes':{'train':f'train{s}','val':f'val{s}','test':f'test{s}'}} for s in common.SEEDS}
    fits=[];evaluations=[]
    def fit(data,seed,model,removed,unit,smoke=False):
        unit.mkdir(parents=True,exist_ok=True)
        kept=[c for c in common.FEATURES if c not in removed]
        contract={'protocol_id':common.PROTOCOL_ID,'seed':seed,'model':model,'removed':removed,'kept':kept,
                  'source_membership':data['membership_hashes'],'smoke_only':False}
        common.write_immutable_json(unit/'fit_contract.json',contract)
        if (unit/'TRAINED.json').exists():
            return joblib.load(unit/'model.joblib'),kept,h2.verify_receipt(unit/'TRAINED.json')
        fits.append(str(unit));joblib.dump({'fixture_only':True},unit/'model.joblib')
        h2.receipt_for(unit/'TRAINED.json',[unit/'fit_contract.json',unit/'model.joblib'],
                       contract_hash=common.hash_file(unit/'fit_contract.json'),train_time_s=0.)
        return {'fixture_only':True},kept,read_json(unit/'TRAINED.json')
    def evaluate(pipeline,kept,target,chunk_size):
        evaluations.append(target)
        return {'mcc':.4,'binary_f1':.5,'fnr':.2,'fpr':.1,'weighted_f1':.6,
                'tp':40,'tn':40,'fp':10,'fn':10,'n_samples':common.TOTALS[target],
                'evaluation_rows':common.TOTALS[target],'evaluation_scope':'full_target'}
    table=matrix()
    for row in table[table.package=='random_size'].itertuples():
        seed=int(row.training_seed);k=int(row.condition.split('_')[0][1:]);rep=int(row.condition.split('rep')[1])
        if k in (5,10):
            unit=(args.phase2/'fits' if k==10 else args.phase3/'shared_random')/row.run_id
            fit(data[seed],seed,'MLP',h2.random_mask(common.FEATURES,seed,k,rep),unit)
    baseline=pd.DataFrame([{'seed':s,'model':'MLP','target_dataset':'ToN-IoT','filtering_strategy':'none',
                           'mcc':.3,'binary_f1':.4,'fnr':.3,'fpr':.1} for s in common.SEEDS])
    common.atomic_csv(baseline,args.phase2/'h2/candidates/by_target/ToN-IoT.csv')
    fits.clear()
    with patch.object(common,'source_data',side_effect=lambda s:data[s]),patch.object(h2,'fit',side_effect=fit),patch.object(h2,'evaluate',side_effect=evaluate):
        frame=random_size.run(args)
        assert len(frame)==120 and len(fits)==60 and len(evaluations)==120
        random_size.run(args)
        assert len(fits)==60 and len(evaluations)==120
    registry=read_json(args.output_root/'random_size/run_registry.json')
    assert registry['reused_fits']==60 and registry['new_fits']==60
    checks['120_random_cells_cross_phase_reuse_and_resume_no_refit']=True
    checks.update(readout_and_reporting(args,root))
    return checks


def readout_and_reporting(args,root):
    from cross_domain_nids.models.dann import DANN
    from experiments.revision import h3
    from .design import POLICY
    import torch
    rng=np.random.default_rng(311)
    n=512
    raw=pd.DataFrame({name:np.repeat(np.arange(n//2),2) for name in readout.FLOW_COLUMNS})
    raw['L7_PROTO']=rng.normal(size=n)
    fixture=root/'readout_target.csv';raw.to_csv(fixture,index=False)
    z=rng.normal(size=(n,64)).astype(np.float32);labels=(z[:,0]>0).astype(np.int64)
    for seed in common.SEEDS:
        unit=h3.unit_dir(args.phase2,'ToN-IoT',seed)
        common.atomic_write_json(unit/'COMPLETED.json',{'fixture_only':True})
        for variant in ('H3-C0','H3-C1'):
            directory=unit/variant;directory.mkdir()
            np.savez_compressed(directory/'reservoir_embeddings.npz',target_ids=np.arange(n),
                                target_labels=labels,target_embeddings=z)
            torch.manual_seed(seed);torch.save(DANN(49).state_dict(),directory/'best.pt')
    policy={**POLICY['readout'],'sample_rows':200}
    with patch.dict(POLICY,{'readout':policy}),patch.object(common,'data_path',return_value=fixture),patch.object(h3,'validate_completed_artifact_hashes'):
        result=readout.run(args)
    assert len(result)==12 and result.fit_converged.all()
    audit=pd.read_csv(args.output_root/'readout/overlap_audit.csv')
    assert len(audit)==9 and (audit.shared_groups==0).all()
    # Build explicit artificial summaries to exercise the full report/plot path.
    for alpha,base in [(1,args.phase2),(.1,args.output_root/'strength/alpha0.1'),(.5,args.output_root/'strength/alpha0.5')]:
        for target in common.TARGETS:
            for seed in common.SEEDS:
                unit=h3.unit_dir(base,target,seed)
                metrics=[];mmd=[];probes=[]
                for variant in ('H3-C0','H3-C1'):
                    active=variant=='H3-C1'
                    run_id=(f'H3/DANN/zero/{target}/{seed}' if not active else
                            f'H3/DANN/original/{target}/{seed}' if alpha==1 else f'strength/DANN/alpha{alpha:g}/{target}/{seed}')
                    metrics.append({'run_id':run_id,'target':target,'seed':seed,'variant':variant,
                                    'MCC':.6-.2*alpha*active,'Attack-class Binary-F1':.7-.1*alpha*active,'FNR':.1+.1*alpha*active})
                    mmd.append({'variant':variant,'mmd_rbf_unbiased':.1-.01*alpha*active})
                    for probe_seed in common.SEEDS:
                        probes.append({'variant':variant,'probe_seed':probe_seed,'domain_AUROC':.9-.1*alpha*active})
                common.atomic_csv(pd.DataFrame(metrics),unit/'paired_metrics.csv')
                common.atomic_csv(pd.DataFrame(mmd),unit/'mmd.csv')
                common.atomic_csv(pd.DataFrame(probes),unit/'domain_probe.csv')
    entries=[]
    for model,abstained in [('MLP',False),('RandomForest',True)]:
        delta=pd.DataFrame({'seed':common.SEEDS,'MCC':[.1]*3,'Binary_F1':[.1]*3,'FNR':[0]*3})
        entries.extend(gates.grid({'analysis_version':'synthetic','method':'H2','target':'ToN-IoT','model':model},
                                  {'full':delta},abstained=abstained))
    common.atomic_csv(pd.DataFrame(entries),args.output_root/'gates/grid.csv')
    common.atomic_csv(pd.DataFrame([{'analysis_version':'synthetic','method':'H2','target':'ToN-IoT','model':'MLP'}]),args.output_root/'gates/stability.csv')
    reporting.finish(args)
    complete=read_json(args.output_root/'PHASE4_COMPLETE.json')
    assert complete['fixed_matrix_cells']==138
    assert complete['random_reused_fits']==60 and complete['random_new_fits']==60
    assert len(list((args.output_root/'figures').glob('*.png')))==5
    assert 'seed' not in pd.read_csv(args.output_root/'strength/seed_means.csv').columns
    return {'posthoc_readout_end_to_end_group_audit':True,'138_cell_summary_figures_and_completion':True}
