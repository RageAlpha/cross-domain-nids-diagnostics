"""120 predeclared ToN/MLP random cells, with verified cross-phase reuse."""
import gc
import json
import time
import numpy as np
import pandas as pd
from experiments.revision import common, h2
from experiments.revision.comparisons import cache
from .design import matrix, read_json, finite, three_seeds


def import_fit(source, destination):
    """Bounded retry for transient Windows scanner locks during publication.

    Each retry still copies into the validated staging directory. Persistent
    access errors are propagated; no partial destination is accepted.
    """
    for attempt in range(5):
        try:
            cache.import_fit(source,destination)
            return
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32,33) or attempt==4:
                raise
            time.sleep(.2*(attempt+1))


def compatible_candidates(args, run_id, data, seed, removed):
    candidates = [args.phase2/'fits'/run_id, args.phase3/'shared_random'/run_id]
    valid = []
    for unit in candidates:
        if not unit.exists():
            continue
        # A present but incomplete/corrupt upstream fit is an input error, not a
        # reason to silently train a replacement or pick a more favorable copy.
        cache.validate_contract(unit, data, seed, 'MLP', removed)
        for path in unit.glob('*.COMPLETE.json'):
            h2.verify_receipt(path)
        valid.append(unit)
    if len(valid) == 2:
        for name in ('model.joblib', 'fit_contract.json'):
            if common.hash_file(valid[0]/name) != common.hash_file(valid[1]/name):
                raise ValueError(f'Conflicting duplicate random model: {run_id}')
    return valid


def run(args):
    table = matrix()
    table = table[table.package == 'random_size']
    bank = read_json(common.PREP/'random_masks.json')
    masks = {(r['seed'],r['k'],r['replicate']):r['removed'] for r in bank['masks']}
    rows, registry = [], []
    for seed in common.SEEDS:
        data = common.source_data(seed)
        for row in table[table.training_seed == seed].itertuples():
            k, rep = [int(s) for s in row.condition.replace('k','').replace('rep','').split('_')]
            removed = h2.random_mask(common.FEATURES,seed,k,rep)
            if masks[seed,k,rep] != removed:
                raise ValueError('Random mask differs from the frozen mask bank')
            unit = args.output_root/'random_size/fits'/row.run_id
            candidates = compatible_candidates(args,row.run_id,data,seed,removed)
            source = candidates[0] if candidates else None
            provenance = {'run_id':row.run_id, 'seed':seed,'k':k,'replicate':rep,
                          'origin':'reused' if source else 'phase4_new',
                          'source':str(source) if source else None,
                          'source_receipt_hash':common.hash_file(source/'TRAINED.json') if source else None}
            common.write_immutable_json(args.output_root/'random_size/provenance'/f'{seed}_{k}_{rep}.json',provenance)
            if source:
                import_fit(source,unit)
            result = cache.evaluate_cached(args,data,seed,'MLP','matched_random',removed,
                                            ['ToN-IoT'],unit,row.run_id)
            value = result[0]
            if value['protocol_id'] != common.PROTOCOL_ID or value['source_membership_hash'] != common.hash_json(data['membership_hashes']):
                raise ValueError('Random evaluation source/protocol mismatch')
            if int(value.get('n_samples', value.get('evaluation_rows', -1))) != common.TOTALS['ToN-IoT']:
                raise ValueError('Random evaluation does not cover the full target')
            rows.append({**value,'k':k,'replicate':rep,'origin':provenance['origin']})
            registry.append(provenance)
        del data
        gc.collect()
    frame = pd.DataFrame(rows)
    if len(frame) != 120 or set(frame.run_id) != set(table.run_id) or frame.run_id.duplicated().any():
        raise ValueError('Incomplete random-size matrix')
    finite(frame,['mcc','binary_f1','fnr'])
    baseline = h2.normalize_metrics(pd.read_csv(args.phase2/'h2/candidates/by_target/ToN-IoT.csv'))
    baseline = baseline[(baseline.model=='MLP') & (baseline.filtering_strategy=='none')]
    three_seeds(baseline)
    frame = frame.merge(baseline[['seed','mcc','binary_f1','fnr']],on='seed',suffixes=('','_full'),validate='many_to_one')
    for metric in ('mcc','binary_f1','fnr'):
        frame['delta_'+metric] = frame[metric]-frame[metric+'_full']
    common.atomic_csv(frame,args.output_root/'random_size/per_mask.csv')
    summary = []
    for (seed,k), group in frame.groupby(['seed','k']):
        if len(group)!=10 or set(group.replicate)!=set(range(10)):
            raise ValueError('Each seed/size requires exactly ten masks')
        for metric in ('mcc','binary_f1','fnr'):
            values=group[metric];delta=group['delta_'+metric]
            summary.append({'seed':seed,'k':k,'metric':metric,'mask_count':10,
                            'mean':values.mean(),'median':values.median(),'min':values.min(),'max':values.max(),
                            'delta_mean':delta.mean(),'delta_median':delta.median(),
                            'q25':values.quantile(.25),'q75':values.quantile(.75)})
    summary=pd.DataFrame(summary)
    common.atomic_csv(summary,args.output_root/'random_size/per_seed_distribution.csv')
    means=summary.groupby(['k','metric'],as_index=False)[['mean','median','delta_mean','delta_median']].mean()
    means['aggregation']='equal-weight mean over three training seeds; masks are nested repetitions'
    common.atomic_csv(means,args.output_root/'random_size/seed_means.csv')
    common.atomic_write_json(args.output_root/'random_size/run_registry.json',{'fits':registry,
        'fixed_cells':120,'reused_fits':sum(r['origin']=='reused' for r in registry),
        'new_fits':sum(r['origin']=='phase4_new' for r in registry)})
    return frame
