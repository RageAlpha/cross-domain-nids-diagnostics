"""Validate cached fit identity and publish imported fits only after full copying."""
import json
import shutil
from experiments.revision import common, h2

def validate_contract(unit, data, seed, model, removed):
    expected = {'protocol_id': common.PROTOCOL_ID, 'seed': seed, 'model': model,
                'removed': removed, 'kept': [f for f in data['feature_cols'] if f not in removed],
                'source_membership': data['membership_hashes'], 'smoke_only': False}
    path = unit / 'fit_contract.json'
    if json.loads(path.read_text(encoding='utf-8')) != expected:
        raise ValueError(f'Cached fit does not match requested model, mask or source partitions: {unit}')
    receipt = h2.verify_receipt(unit / 'TRAINED.json')
    if receipt['contract_hash'] != common.hash_file(path):
        raise ValueError(f'Cached fit contract hash differs: {unit}')

def import_fit(source, destination):
    """A crash leaves a retryable staging folder, never a published partial fit."""
    h2.verify_receipt(source / 'TRAINED.json')
    if destination.exists():
        return
    stage = destination.with_name(destination.name + '.importing')
    stage.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, stage, dirs_exist_ok=True)
    h2.verify_receipt(stage / 'TRAINED.json')
    for receipt in stage.glob('*.COMPLETE.json'):
        h2.verify_receipt(receipt)
    stage.replace(destination)

def evaluate_cached(args, data, seed, model, strategy, removed, targets, unit, run_id):
    if (unit / 'TRAINED.json').exists():
        validate_contract(unit, data, seed, model, removed)
    rows = h2.evaluate_cached(args, data, seed, model, strategy, removed, targets, unit, run_id)
    for row in rows:
        if (row['run_id'] != run_id or int(row['seed']) != seed or row['model'] != model
                or row['target_dataset'] not in targets or json.loads(row['removed_features']) != removed
                or row['mask_hash'] != common.hash_json(removed)):
            raise ValueError(f'Cached evaluation identity differs from the requested experiment: {unit}')
    return rows
