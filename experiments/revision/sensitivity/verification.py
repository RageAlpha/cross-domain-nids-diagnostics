"""Read-only validation of upstream inputs and immutable Phase 4 identity."""
import importlib.metadata
import platform
import sys
import numpy as np
import pandas as pd
import torch
from experiments.revision import common, h3, h2
from experiments.revision.comparisons.run import validate_phase2_selection
from experiments.revision.comparisons import paired, selection
from .design import POLICY, matrix, read_json, finite


def code_fingerprint():
    return common.code_fingerprint()


def environment():
    return {'python':sys.version,'platform':platform.platform(),
            'packages':{name:importlib.metadata.version(name) for name in
                        ('torch','numpy','pandas','scipy','scikit-learn','joblib','matplotlib','xgboost')},
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'tf32':torch.backends.cuda.matmul.allow_tf32,'threads':torch.get_num_threads()}


def lock_local(args):
    if sys.flags.optimize:
        raise ValueError('Run without -O/-OO: smoke assertions must remain enabled')
    common.write_immutable_json(args.output_root/'policy.json',POLICY)
    matrix()
    common.write_immutable_json(args.output_root/'INPUT_LOCK.json',{
        'protocol_id':common.PROTOCOL_ID,'code':code_fingerprint(),
        'phase1_manifest':common.hash_file(common.PREP/'manifest.json'),
        'matrix':common.hash_file(common.PREP/'formal_matrix.csv'),
        'mask_bank':common.hash_file(common.PREP/'random_masks.json'),
        'phase2_path':str(args.phase2),'phase3_path':str(args.phase3),'historical_path':str(args.historical)})
    common.atomic_write_json(args.output_root/'environment.json',environment())


def check(args):
    from experiments.revision.common import verify
    import sklearn
    if sys.version_info[:2]!=(3,12) or sklearn.__version__!='1.5.1':
        raise ValueError('Use the validated Python 3.12 / sklearn 1.5.1 environment')
    verify(common.ROOT)
    manifest=read_json(common.PREP/'manifest.json')
    for name,digest in manifest['artifacts'].items():
        if common.hash_file(common.PREP/name)!=digest:
            raise ValueError(f'Prepared input changed: {name}')
    old=read_json(args.phase2/'INPUT_LOCK.json')
    for name,digest in old['code_and_templates'].items():
        if common.hash_file(common.ROOT/name)!=digest:
            raise ValueError(f'Phase 2 dependency changed: {name}')
    previous=read_json(args.phase3/'INPUT_LOCK.json')
    if previous['phase2_input_lock']!=common.hash_file(args.phase2/'INPUT_LOCK.json'):
        raise ValueError('Phase 3 references a different Phase 2 input lock')
    for name,digest in previous['phase3_code'].items():
        if common.hash_file(common.ROOT/name)!=digest:
            raise ValueError(f'Phase 3 code changed: {name}')
    lock_local(args)
    print('PHASE 4 CHECK PASS',flush=True)


def validate_stage_markers(args):
    p2=read_json(args.phase2/'PHASE2_COMPLETE.json')
    p3=read_json(args.phase3/'PHASE3_COMPLETE.json')
    if (p2.get('status')!='complete' or p2.get('fixed_h2_fits')!=108 or p2.get('fixed_h3_fits')!=18
            or p3.get('status')!='complete' or p3.get('fixed_fits')!=48
            or p2.get('protocol_id')!=common.PROTOCOL_ID or p3.get('protocol_id')!=common.PROTOCOL_ID):
        raise ValueError('Phase 2 and Phase 3 must finish before Phase 4 formal execution')


def upstream_snapshot(args):
    hashes={}
    for label,root in [('phase2',args.phase2),('phase3',args.phase3),('historical',args.historical)]:
        if not root.is_dir():
            raise ValueError(f'Missing upstream results: {root}')
        files={}
        for path in sorted(root.rglob('*')):
            rel=path.relative_to(root)
            if ({'smoke','predictions','checkpoint_sensitivity_v1'}.intersection(rel.parts)
                    or not path.is_file() or path.suffix not in ('.json','.csv','.npz','.pt','.joblib')):
                continue
            files[rel.as_posix()]=common.hash_file(path)
        hashes[label]=files
    return hashes


def validate_upstream(args):
    validate_stage_markers(args)
    # Verify the Phase 3 frozen references without writing into either stage.
    for name,digest in read_json(args.phase3/'PHASE2_REFERENCE_LOCK.json').items():
        if common.hash_file(args.phase2/name)!=digest:
            raise ValueError(f'Phase 2 differs from the Phase 3 reference: {name}')
    validate_phase2_selection(args.phase2)
    h3.validate_h3_coverage(args.phase2,h3.load_policy(common.ROOT/'protocols/revision/h3.json'))
    actual=set()
    for method in ('coral','mmd','FT','BN','weight'):
        paired.configure(method)
        coverage=paired.validate_h3_coverage(args.phase3/method,paired.load_policy(args.phase3/method/'policy.json'))
        actual.update(r for r in coverage['metrics'].run_id if not r.startswith('H3/'))
    paired.configure('coral')
    mrmr=pd.read_csv(args.phase3/'mRMR/h2/analysis/all_candidates_long.csv')
    actual.update(r for r in mrmr.run_id if r.startswith('mRMR/'))
    full=pd.read_csv(common.PREP/'formal_matrix.csv')
    if actual!=set(full[full.phase==3].run_id):
        raise ValueError('Phase 3 fixed matrix coverage differs')
    for method in ('mRMR','shift_budget'):
        policy=read_json(args.phase3/'h2_policy.json');policy['_path']=str(args.phase3/'h2_policy.json')
        from types import SimpleNamespace
        sub=SimpleNamespace(output_root=args.phase3/method)
        for target in common.TARGETS:
            frame=h2.normalize_metrics(pd.read_csv(sub.output_root/'h2/candidates/by_target'/f'{target}.csv'))
            for model in ('MLP','RandomForest'):
                selection.validate_development_cartesian(frame[frame.model==model],[target],model)
                path=sub.output_root/'h2/selection_locks'/f'{target}_{model}.json';lock=read_json(path)
                selection.validate_selection_lock(lock,path,policy,target,model)
                selection.load_complete_random_controls(sub,target,model,lock,path,policy)
    # All trained fits and evaluation receipts in the corrected stages must hash-match.
    for root in (args.phase2,args.phase3):
        for name in ('TRAINED.json','*.COMPLETE.json'):
            for path in root.rglob(name):
                if 'smoke' in path.relative_to(root).parts:
                    continue
                item=read_json(path)
                if 'files' in item:
                    h2.verify_receipt(path)
    common.write_immutable_json(args.output_root/'UPSTREAM_REFERENCE_LOCK.json',upstream_snapshot(args))


def verify_smoke(args):
    passed=read_json(args.output_root/'smoke/PASS.json')
    for name in ('INPUT_LOCK.json','environment.json','policy.json'):
        if passed[name]!=common.hash_file(args.output_root/name):
            raise ValueError('Code, policy or environment changed since smoke')
    if passed.get('formal_check_passed') is not True:
        raise ValueError('Local self-test does not replace receiver check and smoke')
