"""Phase 3 CLI. Run from project root: python run_revision.py comparisons check|smoke|run --execute."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import importlib.metadata
import pandas as pd
import torch
from experiments.revision import common, h3, h2
from experiments.revision.core import exclusive_run
from . import paired, features

METHODS = ('coral','mmd','FT','BN','weight')

def check(args):
    from experiments.revision.common import verify
    verify(common.ROOT)
    import sklearn
    if sys.version_info[:2] != (3,12) or sklearn.__version__ != '1.5.1':
        raise RuntimeError('Use the validated Python 3.12 / scikit-learn 1.5.1 environment.')
    manifest=json.loads((common.PREP/'manifest.json').read_text())
    for name,digest in manifest['artifacts'].items():
        if common.hash_file(common.PREP/name)!=digest:
            raise ValueError(f'Prepared input changed: {name}')
    previous=json.loads((args.phase2/'INPUT_LOCK.json').read_text())
    for name,digest in previous['code_and_templates'].items():
        if common.hash_file(common.ROOT/name)!=digest:
            raise ValueError(f'Phase 2 dependency changed: {name}')
    code=common.code_fingerprint()
    lock={'protocol_id':common.PROTOCOL_ID,'phase3_code':code,'phase2_input_lock':common.hash_file(args.phase2/'INPUT_LOCK.json')}
    common.write_immutable_json(args.output_root/'INPUT_LOCK.json',lock)
    common.atomic_write_json(args.output_root/'environment.json',{'python':sys.version,'torch':torch.__version__,
        'packages':{name:importlib.metadata.version(name) for name in ('numpy','pandas','scipy','joblib','xgboost')},
        'sklearn':sklearn.__version__,'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    policy=json.loads((common.ROOT/'protocols/revision/h2.json').read_text())
    common.write_immutable_json(args.output_root/'h2_policy.json',policy)
    print('PHASE 3 CHECK PASS',flush=True)

def freeze_phase2(args):
    completed=json.loads((args.phase2/'PHASE2_COMPLETE.json').read_text())
    if completed.get('status')!='complete' or completed.get('fixed_h2_fits')!=108 or completed.get('fixed_h3_fits')!=18:
        raise ValueError('Phase 2 must finish and pass its matrix coverage check first')
    h3.validate_h3_coverage(args.phase2,h3.load_policy(common.ROOT/'protocols/revision/h3.json'))
    validate_phase2_selection(args.phase2)
    hashes={}
    for p in sorted(args.phase2.rglob('*')):
        relative=p.relative_to(args.phase2)
        if p.is_file() and not {'smoke','predictions'}.intersection(relative.parts) and p.suffix in ('.json','.csv','.joblib','.pt'):
            hashes[relative.as_posix()]=common.hash_file(p)
    common.write_immutable_json(args.output_root/'PHASE2_REFERENCE_LOCK.json',hashes)

def validate_phase2_selection(root):
    from experiments.revision import h2_analysis
    from experiments.h2.run_loto_filtering import validate_development_cartesian, validate_chosen_masks
    policy_path=common.ROOT/'protocols/revision/h2.json'
    policy=json.loads(policy_path.read_text())
    policy['_path']=str(policy_path)
    sub=argparse.Namespace(output_root=root)
    for target in common.TARGETS:
        frame=h2.normalize_metrics(pd.read_csv(root/'h2/candidates/by_target'/f'{target}.csv'))
        for model in ('RandomForest','MLP'):
            group=frame[frame.model==model]
            validate_development_cartesian(group,[target],model)
            path=root/'h2/selection_locks'/f'{target}_{model}.json'
            lock=json.loads(path.read_text())
            h2_analysis.validate_selection_lock(lock,path,policy,target,model)
            validate_chosen_masks(group[group.filtering_strategy==lock['selected_strategy']],lock,target,model)
            h2_analysis.load_complete_random_controls(sub,target,model,lock,path,policy)

def summary(args):
    rows=[]
    all_metrics=[]
    for method in METHODS:
        root=args.output_root/method
        metrics=pd.concat([pd.read_csv(p) for p in (root/'h3/runs').glob('*/seed*/paired_metrics.csv')],ignore_index=True)
        metrics['method']=method
        all_metrics.append(metrics)
        verdict=pd.read_csv(root/'h3/analysis/verdict_by_target.csv')
        rows.extend({'method':method,**r} for r in verdict.to_dict('records'))
    deep=pd.concat(all_metrics,ignore_index=True)
    for method in ('mRMR','shift_budget'):
        verdict=pd.read_csv(args.output_root/method/'h2/analysis/verdict_by_target_model.csv')
        rows.extend({'method':method,**r} for r in verdict.to_dict('records'))
    mrmr=pd.read_csv(args.output_root/'mRMR/h2/analysis/all_candidates_long.csv')
    actual=set(deep.run_id)|set(mrmr[mrmr.run_id.str.startswith('mRMR/')].run_id)
    actual={r for r in actual if not r.startswith('H3/')}
    matrix=pd.read_csv(common.PREP/'formal_matrix.csv')
    expected=set(matrix[matrix.phase==3].run_id)
    if actual!=expected:
        raise ValueError(f'Phase 3 fixed coverage mismatch: missing={expected-actual}, extra={actual-expected}')
    common.atomic_csv(deep,args.output_root/'paired_metrics_all.csv')
    baseline=pd.concat([pd.read_csv(args.phase2/'h3/runs/ToN-IoT'/f'seed{seed}'/'paired_metrics.csv')
                        for seed in common.SEEDS],ignore_index=True)
    baseline['method']='reference_frozen_bn_unweighted'
    mechanism=pd.concat([baseline,deep[deep.method.isin(['BN','weight'])]],ignore_index=True)
    common.atomic_csv(mechanism,args.output_root/'mechanism_absolute_metrics.csv')
    deltas=[]
    for (method,seed),group in mechanism.groupby(['method','seed']):
        indexed=group.set_index('variant')
        deltas.append({'method':method,'seed':seed,**{f'delta_{m}':float(indexed.loc['H3-C1',m]-indexed.loc['H3-C0',m])
                       for m in ('MCC','Attack-class Binary-F1','FNR','FPR')}})
    common.atomic_csv(pd.DataFrame(deltas),args.output_root/'mechanism_paired_deltas.csv')
    review=pd.DataFrame(rows)
    review['conclusion_impact']='Assess against the manuscript: retained/strengthened/weakened/insufficient evidence'
    common.atomic_csv(review,args.output_root/'conclusion_review.csv')
    common.atomic_write_json(args.output_root/'PHASE3_COMPLETE.json',{'protocol_id':common.PROTOCOL_ID,'status':'complete','fixed_fits':48,
        'conclusion_review_required':True,'random_fit_registry_locations':['mRMR/h2/random/run_registry.json','shift_budget/h2/random/run_registry.json']})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','smoke','run'])
    parser.add_argument('--phase2',type=Path,default=common.ROOT/'results/generated/revision/core')
    parser.add_argument('--output-root',type=Path,default=common.ROOT/'results/generated/revision/comparisons')
    parser.add_argument('--execute', action='store_true')
    args=parser.parse_args()
    if not args.execute:
        parser.error('Stage execution requires --execute; use run_revision.py plan for a safe preview')
    os.chdir(common.ROOT)
    args.phase2=args.phase2.resolve();args.output_root=common.generated_output(args.output_root)
    if not args.output_root.is_relative_to(common.ROOT/'results') or args.output_root==common.ROOT/'results' or args.output_root.is_relative_to(args.phase2) or args.phase2.is_relative_to(args.output_root) or args.output_root.is_relative_to(common.PREP):
        parser.error('Use an independent Phase 3 output directory under results')
    args.execute,args.resume,args.cpu,args.chunk_size=True,True,False,250000
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    with exclusive_run(common.ROOT/'results/generated/.revision.lock'):
        check(args)
        if args.command=='check':
            return
        if args.command=='smoke':
            from .smoke import run
            run(args)
            return
        passed=json.loads((args.output_root/'smoke/PASS.json').read_text())
        for name in ('INPUT_LOCK.json','environment.json'):
            if passed[name]!=common.hash_file(args.output_root/name):
                raise ValueError('Code/environment changed since Phase 3 smoke')
        if not torch.cuda.is_available():
            raise ValueError('Formal execution requires CUDA')
        freeze_phase2(args)
        features.mapping(args)
        for method in METHODS:
            paired.configure(method)
            sub=copy.copy(args);sub.output_root=args.output_root/method
            policy=json.loads((common.ROOT/'protocols/revision/h3.json').read_text())
            policy['targets']=list(paired.TARGET_DATASETS)
            sub.policy=sub.output_root/'policy.json'
            common.write_immutable_json(sub.policy,policy)
            for target in policy['targets']:
                for seed in common.SEEDS:
                    paired.configure(method,target,seed,args.phase2)
                    sub.target,sub.seed=target,seed
                    print(f'PHASE 3 {method} {target} seed={seed}',flush=True)
                    paired.run_pair(sub)
            paired.analyze(sub)
        features.run(args)
        summary(args)
        print(f'PHASE 3 COMPLETE: {args.output_root}',flush=True)

if __name__=='__main__':
    main()
