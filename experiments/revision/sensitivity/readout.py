"""Target-supervised post-hoc readout. It cannot choose a transfer model."""
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from experiments.revision import common, h3
from experiments.revision.partitions import FLOW_COLUMNS, frame_dtypes, canonicalize_chunk
from experiments.revision.partitions import fingerprints
from cross_domain_nids.models.dann import DANN
from .design import POLICY, read_json


def raw_rows(path, ids, chunk_size=250000):
    columns=list(pd.read_csv(path,nrows=0).columns)
    columns=[c for c in columns if c not in ('Label','Attack')]
    if not set(FLOW_COLUMNS).issubset(columns):
        raise ValueError('Missing session identifiers for readout audit')
    ids=np.asarray(ids,dtype=np.int64)
    if not len(ids) or len(ids)!=len(np.unique(ids)) or (ids<0).any():
        raise ValueError('Invalid readout row IDs')
    ordered=np.sort(ids);frames=[];offset=0
    with pd.read_csv(path,usecols=columns,dtype=frame_dtypes(columns),chunksize=chunk_size) as reader:
        for chunk in reader:
            lo,hi=np.searchsorted(ordered,[offset,offset+len(chunk)])
            if hi>lo:
                selected=chunk.iloc[ordered[lo:hi]-offset].copy();selected.index=ordered[lo:hi];frames.append(selected)
            offset+=len(chunk)
            if offset>ordered[-1]:
                break
    if sum(map(len,frames))!=len(ids):
        raise ValueError('Readout sample rows are missing')
    return pd.concat(frames).loc[ids,columns]


def split_and_audit(raw):
    if raw[FLOW_COLUMNS].isna().any().any():
        raise ValueError('Missing session field in readout sample')
    clean=canonicalize_chunk(raw.copy())
    if clean[FLOW_COLUMNS].isna().any().any():
        raise ValueError('Nonfinite session field in readout sample')
    keys={
        'exact_parsed_nonlabel_row':fingerprints(raw),
        'normalized_nonlabel_row':fingerprints(clean),
        'timestamped_session':fingerprints(clean[FLOW_COLUMNS]),
    }
    groups,inverse=np.unique(keys['timestamped_session'],return_inverse=True)
    if len(groups)<2:
        raise ValueError('Readout requires at least two session groups')
    order=np.random.default_rng(271828).permutation(len(groups))
    counts=np.bincount(inverse)
    boundaries=np.cumsum(counts[order])[:-1]
    cut=int(np.argmin(np.abs(boundaries-.7*len(raw))))+1
    train=np.isin(inverse,order[:cut]);test=~train
    audit=[]
    for name,values in keys.items():
        shared=np.intersect1d(values[train],values[test])
        audit.append({'criterion':name,'shared_groups':len(shared),
                      'train_rows_in_overlap':int(np.isin(values[train],shared).sum()),
                      'test_rows_in_overlap':int(np.isin(values[test],shared).sum()),
                      'train_rows':int(train.sum()),'test_rows':int(test.sum())})
    # Full nonlabel duplicates necessarily share the seven session fields.
    if any(row['shared_groups'] for row in audit):
        raise ValueError('Readout cross-partition dependence detected')
    return train,test,audit


def fit_readout(embedding, labels, train, test):
    if not np.isfinite(embedding).all() or not np.isin(labels,[0,1]).all():
        raise ValueError('Invalid readout embeddings or labels')
    if set(labels[train])!={0,1} or set(labels[test])!={0,1}:
        raise ValueError('Readout partitions must each contain both classes; do not resample based on labels')
    estimator=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',
        solver='lbfgs',max_iter=1000,random_state=271828))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        estimator.fit(embedding[train],labels[train])
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    prediction=(estimator.predict_proba(embedding[test])[:,1]>=.5).astype(int)
    return estimator,prediction,converged


def metrics(labels,prediction):
    return h3.binary_metrics_from_counts(int(((labels==1)&(prediction==1)).sum()),
        int(((labels==0)&(prediction==0)).sum()),int(((labels==0)&(prediction==1)).sum()),
        int(((labels==1)&(prediction==0)).sum()))


def run(args):
    rows,audits=[] ,[]
    for seed in common.SEEDS:
        source=h3.unit_dir(args.phase2,'ToN-IoT',seed)
        h3.validate_completed_artifact_hashes(source,read_json(source/'COMPLETED.json'))
        reservoirs={}
        for variant in ('H3-C0','H3-C1'):
            with np.load(source/variant/'reservoir_embeddings.npz') as z:
                reservoirs[variant]={key:z[key].copy() for key in ('target_ids','target_labels','target_embeddings')}
        a,b=reservoirs.values()
        if not np.array_equal(a['target_ids'],b['target_ids']) or not np.array_equal(a['target_labels'],b['target_labels']):
            raise ValueError('Readout requires identical paired target IDs and labels')
        if len(np.unique(a['target_ids']))!=len(a['target_ids']):
            raise ValueError('Duplicate reservoir row IDs')
        size=POLICY['readout']['sample_rows']
        if len(a['target_ids'])<size:
            raise ValueError('Formal reservoir has fewer than 20,000 rows')
        # Row-ID sorting makes sampling independent of serialization order.
        sorted_positions=np.argsort(a['target_ids'])
        positions=sorted_positions[np.random.default_rng(314159).choice(len(sorted_positions),size,replace=False)]
        positions=positions[np.argsort(a['target_ids'][positions])]
        ids=a['target_ids'][positions];labels=a['target_labels'][positions]
        raw=raw_rows(common.data_path('ToN-IoT'),ids,args.chunk_size)
        train,test,audit=split_and_audit(raw)
        root=args.output_root/'readout'/f'seed{seed}'
        common.write_immutable_json(root/'selection_lock.json',{
            'protocol_id':common.PROTOCOL_ID,'training_seed':seed,'policy':POLICY['readout'],
            'source_completion_hash':common.hash_file(source/'COMPLETED.json'),
            'row_ids':ids.tolist(),'train_ids':ids[train].tolist(),'test_ids':ids[test].tolist(),
            'sampling_and_split_use_labels':False,'interpretation':'conditional post-hoc target-supervised evidence'})
        common.atomic_csv(pd.DataFrame({'row_id':ids,'split':np.where(train,'readout_train','readout_test'),'label':labels}),root/'members.csv')
        audits.extend({'training_seed':seed,**r} for r in audit)
        for variant, arrays in reservoirs.items():
            embedding=arrays['target_embeddings'][positions]
            estimator,prediction,converged=fit_readout(embedding,labels,train,test)
            joblib.dump(estimator,root/f'{variant}_linear_readout.joblib')
            model=DANN(49,dropout=0.,grl_alpha=1.)
            model.load_state_dict(torch.load(source/variant/'best.pt',map_location='cpu',weights_only=True));model.eval()
            with torch.no_grad():
                probabilities=model.label_predictor(torch.as_tensor(embedding[test],dtype=torch.float32)).softmax(dim=1)[:,1].numpy()
            head=(probabilities>=.5).astype(int)
            common.atomic_csv(pd.DataFrame({'row_id':ids[test],'label':labels[test],
                'readout_prediction':prediction,'transferred_head_prediction':head,'transferred_head_p_attack':probabilities}),root/f'{variant}_test_predictions.csv')
            for name,pred in [('target_supervised_linear_readout',prediction),('transferred_source_head',head)]:
                rows.append({'training_seed':seed,'variant':variant,'evaluation':name,
                    'target_supervised':name=='target_supervised_linear_readout',
                    'fit_converged':converged if name=='target_supervised_linear_readout' else True,
                    'train_rows':int(train.sum()),'test_rows':int(test.sum()),
                    'train_attack_rows':int(labels[train].sum()),'test_attack_rows':int(labels[test].sum()),
                    'scope':'same group-disjoint target diagnostic test subset; not full-target transfer score',
                    **metrics(labels[test],pred)})
    frame=pd.DataFrame(rows)
    common.atomic_csv(frame,args.output_root/'readout/per_seed_metrics.csv')
    common.atomic_csv(pd.DataFrame(audits),args.output_root/'readout/overlap_audit.csv')
    means=frame.groupby(['variant','evaluation'],as_index=False)[['MCC','Attack-class Binary-F1','FNR']].mean()
    means['aggregation']='equal-weight mean over three training seeds'
    common.atomic_csv(means,args.output_root/'readout/seed_means.csv')
    return frame
