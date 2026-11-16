"""Aggregate sensitivity metrics and validate completed experiment coverage."""
import pandas as pd
from experiments.revision import common
from .design import read_json, matrix




def strength_summary(args):
    rows=[]
    for alpha,root in [(0,args.phase2),(1,args.phase2),(.1,args.output_root/'strength/alpha0.1'),(.5,args.output_root/'strength/alpha0.5')]:
        for target in common.TARGETS:
            for seed in common.SEEDS:
                unit=root/'h3/runs'/target/f'seed{seed}'
                frame=pd.read_csv(unit/'paired_metrics.csv')
                variant='H3-C0' if alpha==0 else 'H3-C1'
                current=frame[frame.variant==variant]
                if len(current)!=1:
                    raise ValueError('Missing strength metric')
                row=current.iloc[0].to_dict()
                zero=frame[frame.variant=='H3-C0'].iloc[0]
                row.update(alpha=alpha,reference='matched Phase 2 C0')
                for key in ('MCC','Attack-class Binary-F1','FNR'):
                    row['delta_'+key]=row[key]-zero[key]
                # MMD shares a bandwidth within each pair; do not treat raw MMD
                # at different alphas as a common-kernel curve.
                mmd=pd.read_csv(unit/'mmd.csv').set_index('variant')
                probe=pd.read_csv(unit/'domain_probe.csv')
                disc=probe.assign(d=(probe.domain_AUROC-.5).abs()).groupby('variant').d.mean()
                row.update(MMD=mmd.loc[variant,'mmd_rbf_unbiased'],
                    delta_MMD=mmd.loc[variant,'mmd_rbf_unbiased']-mmd.loc['H3-C0','mmd_rbf_unbiased'],
                    domain_discriminability=disc[variant],
                    delta_domain_discriminability=disc[variant]-disc['H3-C0'],
                    mmd_scope='pair-specific shared bandwidth; raw values across alpha are not a common-kernel comparison')
                rows.append(row)
    frame=pd.DataFrame(rows)
    common.atomic_csv(frame,args.output_root/'strength/per_seed_all_alphas.csv')
    metrics=['MCC','Attack-class Binary-F1','FNR','delta_MCC','delta_Attack-class Binary-F1','delta_FNR','delta_MMD','delta_domain_discriminability']
    mean=frame.groupby(['target','alpha'],as_index=False)[metrics].mean()
    common.atomic_csv(mean,args.output_root/'strength/seed_means.csv')
    return frame


def finish(args):
    strength=strength_summary(args)
    random=pd.read_csv(args.output_root/'random_size/per_mask.csv')
    expected=set(matrix().run_id)
    actual=set(strength[strength.alpha.isin([.1,.5])].run_id)|set(random.run_id)
    if len(strength)!=36 or len(random)!=120 or actual!=expected:
        raise ValueError('Final Phase 4 coverage does not match the fixed matrix')
    readout=pd.read_csv(args.output_root/'readout/per_seed_metrics.csv')
    if len(readout)!=12:
        raise ValueError('Readout requires 3 seeds x 2 variants x 2 evaluation heads')
    review=pd.read_csv(args.output_root/'gates/stability.csv')
    review['conclusion_impact']='Assess against the manuscript: retained/strengthened/weakened/insufficient evidence'
    review['manuscript_change']='Record the corresponding statement, table or figure'
    common.atomic_csv(review,args.output_root/'conclusion_review.csv')
    registry=read_json(args.output_root/'random_size/run_registry.json')
    (args.output_root/'Phase4_report.md').write_text(
        '# Sensitivity results\n\n'
        'See gates/, strength/, random_size/ and readout/ for complete results.\n'
        f'Random fits reused: {registry["reused_fits"]}; new fits: {registry["new_fits"]}.\n'
        'Selections and controls remain frozen. Random masks are nested within seeds.\n'
        'The target-supervised readout is a post-hoc diagnostic.\n', encoding='utf-8')
    artifacts={p.relative_to(args.output_root).as_posix():common.hash_file(p) for p in args.output_root.rglob('*')
               if p.is_file() and p.suffix in ('.csv','.json','.npz','.pt','.joblib','.pdf','.png','.md')
               and 'smoke' not in p.relative_to(args.output_root).parts and 'predictions' not in p.relative_to(args.output_root).parts
               and p.name!='PHASE4_COMPLETE.json'}
    common.atomic_write_json(args.output_root/'PHASE4_COMPLETE.json',{
        'protocol_id':common.PROTOCOL_ID,'status':'complete','fixed_matrix_cells':138,'strength_new_fits':18,
        'random_fixed_cells':120,'random_reused_fits':registry['reused_fits'],'random_new_fits':registry['new_fits'],
        'conclusion_review_required':True,'artifact_hashes':artifacts})
