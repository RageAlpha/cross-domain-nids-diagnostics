"""Source-only preprocessing and label-free CORAL/MMD scale calibration."""

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.revision.partitions import canonicalize_chunk, sha256_file
from experiments.revision.estimators import coral, mmd
from experiments.h3.train_domain_loss_models import freeze_batchnorm_running_stats
from experiments.h1.run_multiseed_training import build_preprocessor_for_features
from cross_domain_nids.models.dann import DANN

OUT = ROOT / 'results/generated/revision_preparation'
SEEDS = (42, 123, 2024)
TOTALS = {'CICIDS2018': 20115529, 'BoT-IoT': 16933808, 'ToN-IoT': 27520260}
BATCH = 4096
BATCHES = 16


def paired_forward(model, source, target):
    logits, source_z = model.forward_label(source, return_features=True)
    with freeze_batchnorm_running_stats(model):
        _, target_z = model.forward_label(target, return_features=True)
    return logits, source_z, target_z


def gradient_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True)
    return float(torch.sqrt(sum(g.square().sum() for g in grads)).detach())


def new_model(seed, feature_count, device):
    torch.manual_seed(seed)
    model = DANN(feature_count, dropout=0.0, grl_alpha=1.0).to(device)
    return model


def main():
    if (OUT / 'alignment_calibration.json').exists():
        raise RuntimeError('Calibration exists; use a new protocol version for recalibration.')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_map = np.load(OUT / 'source_seed42.npz')
    training_ids = source_map['row_ids'][source_map['split'] == 0]
    columns = pd.read_csv(ROOT / 'data/raw/NF-UNSW-NB15-v3.csv', nrows=0).columns.tolist()
    features = [x for x in columns if x not in {'Label', 'Attack', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR'}]
    raw = pd.read_csv(ROOT / 'data/raw/NF-UNSW-NB15-v3.csv', usecols=features+['Label'],
                      dtype={name: np.float64 for name in features+['Label']})
    source = canonicalize_chunk(raw.iloc[training_ids].copy())
    preprocessor = build_preprocessor_for_features(features)
    preprocessor.fit(source[features])
    joblib.dump(preprocessor, OUT / 'source_seed42_preprocessor.joblib')
    calibration_ids, source_arrays, source_labels = {}, {}, {}
    for seed in SEEDS:
        positions = np.random.default_rng(seed + 11001).choice(len(training_ids), BATCH * BATCHES, replace=False)
        calibration_ids[seed] = training_ids[positions]
        source_arrays[seed] = preprocessor.transform(source.iloc[positions][features]).astype(np.float32)
        source_labels[seed] = source.iloc[positions]['Label'].to_numpy(dtype=np.int64)
    del source, raw
    details, summaries, zero_checks = [], [], []
    target_sample_map = {}
    for target, total in TOTALS.items():
        target_ids = {}
        for seed in SEEDS:
            adaptation = np.sort(np.random.default_rng(seed).choice(total, int(total*.05), replace=False))
            target_ids[seed] = np.sort(np.random.default_rng(seed+22001).choice(adaptation, BATCH * BATCHES, replace=False))
        arrays = {seed: [] for seed in SEEDS}
        offset = 0
        for chunk in pd.read_csv(ROOT / 'data/raw' / f'NF-{target}-v3.csv', usecols=features,
                                 dtype={name: np.float64 for name in features}, chunksize=250000):
            end = offset + len(chunk)
            for seed in SEEDS:
                lo, hi = np.searchsorted(target_ids[seed], [offset, end])
                if hi > lo:
                    selected = canonicalize_chunk(chunk.iloc[target_ids[seed][lo:hi]-offset].copy())
                    arrays[seed].append(preprocessor.transform(selected[features]).astype(np.float32))
            offset = end
        if offset != total:
            raise ValueError('Unexpected target row count')
        target_sample_map[target] = {}
        for seed in SEEDS:
            xt = np.concatenate(arrays[seed])
            # Random batch order is fixed independently of target detection labels.
            order = np.random.default_rng(seed+33001).permutation(len(xt))
            xt = xt[order]
            ordered_target_ids = target_ids[seed][order]
            np.savez_compressed(OUT / f'{target}_calibration_seed{seed}.npz',
                                source_row_ids=calibration_ids[seed], target_row_ids=ordered_target_ids)
            target_sample_map[target][str(seed)] = hashlib.sha256(ordered_target_ids.astype('<i8').tobytes()).hexdigest()
            model = new_model(seed, len(features), device)
            initial = copy.deepcopy(model.state_dict())
            model.train()
            sx = torch.tensor(source_arrays[seed], device=device)
            sy = torch.tensor(source_labels[seed], device=device)
            tx = torch.tensor(xt, device=device)
            with torch.no_grad():
                _, source_z, target_z = paired_forward(model, sx[:512], tx[:512])
                joined = torch.cat([source_z, target_z]).double().cpu().numpy()
                distances = ((joined[:, None] - joined[None, :])**2).sum(2)
                upper = distances[np.triu_indices(len(joined), k=1)]
                bandwidth = float(np.median(upper[upper > 0]))
            if not np.isfinite(bandwidth) or bandwidth <= 0:
                raise ValueError('Invalid MMD bandwidth')
            for method in ('coral', 'mmd'):
                ratios, timings = [], []
                for batch in range(BATCHES):
                    model.load_state_dict(initial)
                    model.train()
                    lo, hi = batch*BATCH, (batch+1)*BATCH
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    tick = time.perf_counter()
                    logits, zs, zt = paired_forward(model, sx[lo:hi], tx[lo:hi])
                    task_loss = nn.functional.cross_entropy(logits, sy[lo:hi])
                    alignment_loss = coral(zs, zt) if method == 'coral' else mmd(zs, zt, bandwidth)
                    params = list(model.feature_extractor.parameters())
                    task_norm = gradient_norm(task_loss, params)
                    align_norm = gradient_norm(alignment_loss, params)
                    if not np.isfinite([task_norm, align_norm]).all() or min(task_norm, align_norm) <= 0:
                        raise ValueError(f'Nonpositive/nonfinite calibration gradient: {target}/{seed}/{method}')
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter()-tick
                    ratio = task_norm/align_norm
                    ratios.append(ratio)
                    timings.append(elapsed)
                    details.append({'target':target,'seed':seed,'method':method,'batch':batch,
                                    'task_gradient_norm':task_norm,'alignment_gradient_norm':align_norm,
                                    'gradient_ratio':ratio,'calibration_seconds':elapsed})
                reference = float(np.median(ratios))
                coefficient = .1 * reference
                # Compare a complete Adam step at zero coefficient against DANN zero.
                states = []
                for kind in ('dann', method):
                    model.load_state_dict(initial)
                    model.train()
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                    optimizer.zero_grad(set_to_none=True)
                    logits, zs, zt = paired_forward(model, sx[:BATCH], tx[:BATCH])
                    if kind == 'dann':
                        domain_logits = model.forward_domain(torch.cat([zs, zt]))
                        domain_y = torch.cat([torch.zeros(BATCH, dtype=torch.long, device=device), torch.ones(BATCH, dtype=torch.long, device=device)])
                        extra = nn.functional.cross_entropy(domain_logits, domain_y)
                    else:
                        extra = coral(zs, zt) if method == 'coral' else mmd(zs, zt, bandwidth)
                    (nn.functional.cross_entropy(logits, sy[:BATCH]) + 0 * extra).backward()
                    optimizer.step()
                    states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
                exact = all(torch.equal(states[0][k], states[1][k]) for k in states[0])
                zero_checks.append({'target':target,'seed':seed,'method':method,'one_step_state_equals_dann_zero':exact})
                summaries.append({'target':target,'seed':seed,'method':method,'mmd_bandwidth_squared':bandwidth if method=='mmd' else None,
                                  'gradient_ratio_median':reference,'gradient_ratio_min':min(ratios),'gradient_ratio_max':max(ratios),
                                  'multiplier':.1,'nonzero_coefficient_max':coefficient,
                                  'median_calibration_batch_seconds':float(np.median(timings))})
                print(target, seed, method, 'coefficient', coefficient, 'zero_equal',exact,flush=True)
            del model, sx, sy, tx
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    pd.DataFrame(details).to_csv(OUT / 'alignment_calibration_batches.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT / 'alignment_coefficients.csv',index=False)
    result={'status':'complete','device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
            'source_split_seed':42,'source_train_membership_hash':hashlib.sha256(training_ids.astype('<i8').tobytes()).hexdigest(),
            'preprocessor_sha256':sha256_file(OUT/'source_seed42_preprocessor.joblib'),
            'batch_size':BATCH,'batch_count':BATCHES,'coefficient_rule':'0.1 * median(task backbone gradient norm / alignment backbone gradient norm)',
            'coefficient_schedule':'coefficient_max * epoch_index / 39',
            'mmd_kernel':'mean of three RBF kernels; sigma_squared factors 0.5, 1, 2',
            'mmd_bandwidth_rule':'median positive pairwise squared distance on 512 source + 512 target initial representations',
            'target_detection_labels_read':False,'zero_checks':zero_checks,'target_sample_ids_hashes':target_sample_map,
            'source_code_sha256':sha256_file(Path(__file__))}
    (OUT/'alignment_calibration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
