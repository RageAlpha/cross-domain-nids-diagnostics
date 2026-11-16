"""Frozen-BN DANN strengths; isolated derivation of the validated Phase 3 loop.

No Phase 2/3 code is modified. C0 is imported only after exact identity checks.
The sole intervention is alpha * epoch / 39 for alpha in (0.1, 0.5).
"""
import shutil
from contextlib import nullcontext
from experiments.h3.run_paired_domain_loss import *
from experiments.h3.run_paired_domain_loss import _update_order_digest, _optimizer_to_device
from experiments.revision.common import PROTOCOL_ID, h3_source, load_target
from experiments.revision import h3 as phase2_h3
from experiments.revision.comparisons.performance import make_loader, FrozenBNAudit, weighted_scalar_sum
from experiments.revision.comparisons.paired import PriorityReservoir, evaluate_full_target_reservoir
from experiments.revision.h3 import load_policy, validate_h3_coverage, analyze as _base_analyze

METHOD, SCALE, BANDWIDTH, REFERENCE = 'strength', .1, 1., None

def analyze(args):
    _base_analyze(args)
    path = args.output_root / 'h3/analysis/verdict_by_target.csv'
    # A seed is an identifier, not a quantity to average.
    table = pd.read_csv(path).drop(columns=['seed'], errors='ignore')
    table.to_csv(path, index=False)

def configure(alpha, target=None, seed=None, phase2=None):
    global SCALE, REFERENCE, VARIANTS
    if alpha not in (.1, .5):
        raise ValueError('Only the frozen alpha=0.1 and 0.5 are permitted')
    SCALE = float(alpha)
    VARIANTS = {'H3-C0': 'constant_0', 'H3-C1': f'{SCALE:g} * epoch / 39'}
    REFERENCE = unit_dir(phase2, target, seed) if phase2 is not None else None

def run_id(condition, target, seed):
    return (f'H3/DANN/zero/{target}/{seed}' if condition == 'zero' else
            f'strength/DANN/alpha{SCALE:g}/{target}/{seed}')

def run_pair(args: argparse.Namespace) -> None:
    require_execute_flag(args.execute, "H3 paired training")
    load_policy(args.policy)
    import joblib
    import torch

    settings = H3Settings(chunk_size=args.chunk_size, learning_rate=.001)
    out = unit_dir(args.output_root, args.target, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    if REFERENCE is None:
        raise ProtocolError('A validated Phase 2 reference is required')
    write_immutable_json(out / 'strength_identity.json', {
        'protocol_id': PROTOCOL_ID, 'alpha': SCALE, 'target': args.target, 'seed': args.seed,
        'reference_completion_hash': hash_file(REFERENCE / 'COMPLETED.json'),
        'reference_contract_hash': hash_file(REFERENCE / 'H3-C0/training_contract.json'),
    })
    if (out / "COMPLETED.json").exists():
        validate_completed_artifact_hashes(out, json.loads((out / "COMPLETED.json").read_text()))
        print(f"Already complete: {out}")
        return
    with audited_run(out, {"protocol_id": PROTOCOL_ID, "stage": "H3", "target": args.target, "seed": args.seed}):
        set_all_seeds(args.seed)
        data = h3_source()
        validate_feature_order(data["feature_cols"])
        target_x, indices, sample_meta = load_target(args.target, args.seed, data["preprocessor"])
        preprocessor_path = out / "preprocessor.joblib"
        joblib.dump(data["preprocessor"], preprocessor_path)
        set_all_seeds(args.seed)
        initial_model = DANN(data["X_train"].shape[1], dropout=settings.dropout, grl_alpha=1.0)
        initial_state = {key: value.detach().cpu().clone() for key, value in initial_model.state_dict().items()}
        torch.save(initial_state, out / "initial.pt")
        common = {
            "protocol_id": PROTOCOL_ID, "target": args.target, "seed": args.seed,
            "feature_order": data["feature_cols"], "feature_order_hash": hash_json(data["feature_cols"]),
            "source_member_hashes": data["membership_hashes"],
            "source_class_counts": np.bincount(data["y_train"], minlength=2).tolist(),
            "target_bn_updates": METHOD == "BN",
            "class_weight_rule": "N/(2*n_class)" if METHOD == "weight" else "unweighted",
            "source_split_hash": hash_json({
                "X_train": hash_array(data["X_train"]), "y_train": hash_array(data["y_train"]),
                "X_val": hash_array(data["X_val"]), "y_val": hash_array(data["y_val"]),
                "X_test": hash_array(data["X_test"]), "y_test": hash_array(data["y_test"]),
            }),
            "preprocessor_hash": hash_file(preprocessor_path),
            "target_index_hash": hash_array(indices), "target_sample_hash": hash_array(target_x),
            "initial_state_hash": hash_state_dict(initial_state),
            "source_loader_seed": args.seed,
            "target_loader_seed": args.seed + 17,
            **asdict(settings), "target_sample_metadata": sample_meta,
        }
        pair_manifest = {
            **common, "variants": {
                name: {**{key: common[key] for key in PAIR_KEYS}, "lambda_schedule": schedule}
                for name, schedule in VARIANTS.items()
            }, "only_intended_difference": "alignment_loss_coefficient", "phase4_method": METHOD, "maximum_coefficient": SCALE, "training_bandwidth_squared": BANDWIDTH,
        }
        write_immutable_json(out / "pair_manifest.json", pair_manifest)
        rows, domain_rows, mmd_rows, class_rows, all_logs = [], [], [], [], []
        embedding_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        execution_metadata: dict[str, dict[str, Any]] = {}
        required_epochs: int | None = None
        for variant in VARIANTS:
            variant_dir = out / variant
            variant_dir.mkdir(exist_ok=True)
            model, device, logs, train_meta = train_variant(
                data, target_x, initial_state, settings, variant, args.seed,
                variant_dir, cpu=args.cpu, resume=args.resume,
                required_epochs=required_epochs,
            )
            if variant == "H3-C0":
                required_epochs = int(train_meta["epochs_completed"])
            execution_metadata[variant] = train_meta
            if METHOD == 'strength' and variant == 'H3-C0' and REFERENCE is not None:
                metrics, target_emb, target_labels, target_ids, reservoir_meta, source_emb, source_labels, source_ids = reuse_evaluation()
                validate_confusion_counts(metrics, sample_meta['target_total_samples'])
            else:
                metrics, target_emb, target_labels, target_ids, reservoir_meta = evaluate_full_target_reservoir(
                    torch, model, device, data["preprocessor"], data["feature_cols"], args.target,
                    TARGET_DATASETS[args.target], args.seed, variant, out,
                    settings.chunk_size, settings.embedding_sample_size,
                )
                validate_confusion_counts(metrics, sample_meta["target_total_samples"])
                source_emb, source_labels, source_ids = source_reservoir_embeddings(
                    torch, model, device, data, args.seed, settings.embedding_sample_size,
                )
            np.savez_compressed(
                variant_dir / "reservoir_embeddings.npz", source_embeddings=source_emb,
                source_labels=source_labels, source_ids=source_ids, target_embeddings=target_emb,
                target_labels=target_labels, target_ids=target_ids,
                reservoir_method=reservoir_meta["method"],
                target_stream_rows_seen=reservoir_meta["stream_rows_seen"],
                target_reservoir_ids_hash=reservoir_meta["sample_ids_hash"],
            )
            condition = 'zero' if variant == 'H3-C0' else 'original'
            rows.append({"run_id": run_id(condition, args.target, args.seed), "protocol_id": PROTOCOL_ID, "target": args.target, "seed": args.seed,
                         "variant": variant, **train_meta, **reservoir_meta, **metrics})
            for probe_seed in (42, 123, 2024):
                domain_rows.append({"target": args.target, "seed": args.seed, "variant": variant,
                                    "probe_seed": probe_seed, **domain_discriminability(source_emb, target_emb, probe_seed)})
            embedding_cache[variant] = (source_emb, target_emb)
            class_rows.extend(class_conditional_rows(
                source_emb, source_labels, target_emb, target_labels,
                target=args.target, seed=args.seed, variant=variant,
            ))
            all_logs.extend(logs)
        c0_execution, c1_execution = execution_metadata["H3-C0"], execution_metadata["H3-C1"]
        execution_checks = {
            "epochs_equal": c0_execution["epochs_completed"] == c1_execution["epochs_completed"],
            "source_actual_loader_order_equal": c0_execution["source_actual_loader_order_hash"] == c1_execution["source_actual_loader_order_hash"],
            "target_actual_loader_order_equal": c0_execution["target_actual_loader_order_hash"] == c1_execution["target_actual_loader_order_hash"],
        }
        c0_npz = np.load(out / "H3-C0" / "reservoir_embeddings.npz")
        c1_npz = np.load(out / "H3-C1" / "reservoir_embeddings.npz")
        execution_checks["source_reservoir_ids_equal"] = np.array_equal(c0_npz["source_ids"], c1_npz["source_ids"])
        execution_checks["target_reservoir_ids_equal"] = np.array_equal(c0_npz["target_ids"], c1_npz["target_ids"])
        if not all(execution_checks.values()):
            raise ProtocolError(f"C0/C1 execution invariant failed: {execution_checks}")
        write_immutable_json(out / "pair_execution_manifest.json", {
            "protocol_id": PROTOCOL_ID, "target": args.target, "seed": args.seed,
            "variants": execution_metadata, "checks": execution_checks,
            "status": "passed",
        })
        bandwidth = shared_rbf_bandwidth(
            [array for variant in VARIANTS for array in embedding_cache[variant]],
            deterministic_seed(args.target, args.seed, "shared_mmd_bandwidth"),
        )
        for variant, (source_emb, target_emb) in embedding_cache.items():
            mmd_rows.append({
                "target": args.target, "seed": args.seed, "variant": variant,
                "shared_bandwidth_hash": hash_json(bandwidth),
                **rbf_mmd(source_emb, target_emb, deterministic_seed(args.target, args.seed, "mmd"), bandwidth),
            })
        pd.DataFrame(rows).to_csv(out / "paired_metrics.csv", index=False)
        pd.DataFrame(domain_rows).to_csv(out / "domain_probe.csv", index=False)
        pd.DataFrame(mmd_rows).to_csv(out / "mmd.csv", index=False)
        pd.DataFrame(class_rows).to_csv(out / "class_conditional_probe.csv", index=False)
        pd.DataFrame(all_logs).to_csv(out / "training_log.csv", index=False)
        completed_marker(out, {"protocol_id": PROTOCOL_ID,
            "pair_manifest_hash": hash_file(out / "pair_manifest.json"),
            "artifact_hashes": build_artifact_hashes(out),
            "status": "complete",
        })


def train_variant_raw(
    data: dict[str, Any], target_x: np.ndarray, initial_state: dict[str, Any],
    settings: H3Settings, variant: str, seed: int, output_dir: Path,
    *, cpu: bool, resume: bool, required_epochs: int | None = None,
) -> tuple[Any, Any, list[dict[str, Any]], dict[str, Any]]:
    import torch
    from torch import nn

    schedule = VARIANTS[variant]
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model = DANN(data["X_train"].shape[1], dropout=settings.dropout, grl_alpha=1.0).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    weights = None
    label_loss_fn = nn.CrossEntropyLoss(weight=weights)
    domain_loss_fn = nn.CrossEntropyLoss()
    source_generator = torch.Generator().manual_seed(seed)
    target_generator = torch.Generator().manual_seed(seed + 17)
    source_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(data["X_train"], dtype=torch.float32),
        torch.as_tensor(data["y_train"], dtype=torch.long),
        torch.arange(len(data["X_train"]), dtype=torch.int64),
    )
    target_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(target_x, dtype=torch.float32),
        torch.arange(len(target_x), dtype=torch.int64),
    )
    checkpoint_path = output_dir / "resume.pt"
    start_epoch, stale, best_loss, best_epoch = 0, 0, float("inf"), -1
    best_state, logs = None, []
    source_epoch_order_hashes: list[str] = []
    target_epoch_order_hashes: list[str] = []
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        _optimizer_to_device(optimizer, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        stale, best_loss, best_epoch = checkpoint["stale"], checkpoint["best_loss"], checkpoint["best_epoch"]
        best_state, logs = checkpoint["best_state"], checkpoint["logs"]
        restore_rng_states(checkpoint, source_generator, target_generator, device)
        source_epoch_order_hashes = list(checkpoint["source_epoch_order_hashes"])
        target_epoch_order_hashes = list(checkpoint["target_epoch_order_hashes"])

    for epoch in range(start_epoch, settings.max_epochs):
        if required_epochs is not None and epoch >= required_epochs:
            break
        if required_epochs is None and epoch >= settings.min_epochs and stale >= settings.patience:
            break
        source_loader = make_loader(
            source_dataset, batch_size=settings.batch_size,
            generator=source_generator, cuda=device.type == 'cuda',
        )
        target_loader = make_loader(
            target_dataset, batch_size=settings.batch_size,
            generator=target_generator, cuda=device.type == 'cuda',
        )
        model.train()
        coefficient = 0.0 if schedule == "constant_0" else SCALE * epoch / 39
        target_iter = iter(target_loader)
        source_order_digest = hashlib.sha256()
        target_order_digest = hashlib.sha256()
        totals = {"label": 0.0, "domain": 0.0, "source_n": 0, "domain_n": 0}
        label_values, domain_values = [], []
        bn_audit = FrozenBNAudit(model, device)
        for source_x, source_y, source_indices in source_loader:
            try:
                target_batch, target_indices = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch, target_indices = next(target_iter)
            _update_order_digest(source_order_digest, source_indices)
            _update_order_digest(target_order_digest, target_indices)
            source_x, source_y, target_batch = (tensor.to(device, non_blocking=True)
                                               for tensor in (source_x, source_y, target_batch))
            optimizer.zero_grad(set_to_none=True)
            label_logits, source_features = model.forward_label(source_x, return_features=True)
            label_loss = label_loss_fn(label_logits, source_y)
            before = bn_audit.snapshot() if bn_audit is not None else None
            with freeze_batchnorm_running_stats(model):
                _, target_features = model.forward_label(target_batch, return_features=True)
            if bn_audit is not None:
                bn_audit.observe(before)
            domain_features = torch.cat([source_features, target_features])
            domain_y = torch.cat([
                torch.zeros(len(source_features), dtype=torch.long, device=device),
                torch.ones(len(target_features), dtype=torch.long, device=device),
            ])
            domain_logits = model.forward_domain(domain_features)
            # The zero path intentionally computes the same domain head as Phase 2.
            # This preserves zero gradients and Adam state for the shared C0.
            domain_loss = domain_loss_fn(domain_logits, domain_y)
            (label_loss + coefficient * domain_loss).backward()
            optimizer.step()
            label_values.append((label_loss.detach(), len(source_x)))
            domain_values.append((domain_loss.detach(), len(domain_y)))
            totals["source_n"] += len(source_x)
            totals["domain_n"] += len(domain_y)

        if bn_audit is not None:
            bn_audit.verify()
        totals['label'] = weighted_scalar_sum(label_values)
        totals['domain'] = weighted_scalar_sum(domain_values)
        model.eval()
        # Bounded validation is essential for the FT attention activations.
        numerator = denominator = 0.
        with torch.no_grad():
            for lo in range(0, len(data['X_val']), settings.batch_size):
                vx = torch.as_tensor(data['X_val'][lo:lo+settings.batch_size], dtype=torch.float32, device=device)
                vy = torch.as_tensor(data['y_val'][lo:lo+settings.batch_size], dtype=torch.long, device=device)
                logits = model.forward_label(vx)
                numerator += float(nn.functional.cross_entropy(logits, vy, weight=weights, reduction='sum').cpu())
                denominator += len(vy) if weights is None else float(weights[vy].sum().cpu())
        val_loss = numerator / denominator
        if not np.isfinite(val_loss):
            raise ProtocolError('Nonfinite source validation loss')
        val_metrics = source_validation_metrics(torch, model, data["X_val"], data["y_val"], device)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        logs.append({
            "protocol_id": PROTOCOL_ID, "variant": variant, "seed": seed,
            "epoch": epoch, "lambda": coefficient,
            "source_train_label_loss": totals["label"] / max(totals["source_n"], 1),
            "domain_loss": totals["domain"] / max(totals["domain_n"], 1),
            "source_val_loss": val_loss,
            "source_val_mcc": val_metrics["MCC"],
            "source_val_binary_f1": val_metrics["Attack-class Binary-F1"],
        })
        print(f'{variant} seed={seed} epoch={epoch + 1} val_loss={val_loss:.6f}', flush=True)
        source_epoch_order_hashes.append(source_order_digest.hexdigest())
        target_epoch_order_hashes.append(target_order_digest.hexdigest())
        save_training_checkpoint(
            checkpoint_path, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
            stale=stale, best_loss=best_loss, best_epoch=best_epoch, best_state=best_state,
            logs=logs, source_generator_state=source_generator.get_state(),
            target_generator_state=target_generator.get_state(), python_rng_state=random.getstate(),
            numpy_rng_state=np.random.get_state(), torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            source_epoch_order_hashes=source_epoch_order_hashes,
            target_epoch_order_hashes=target_epoch_order_hashes,
        )
        if required_epochs is not None and (epoch + 1) >= required_epochs:
            break
        if required_epochs is None and (epoch + 1) >= settings.min_epochs and stale >= settings.patience:
            break

    if best_state is None:
        raise ProtocolError(f"No best checkpoint produced for {variant}")
    torch.save(model.state_dict(), output_dir / "final.pt")
    torch.save(best_state, output_dir / "best.pt")
    model.load_state_dict(best_state)
    # Keep the atomic epoch checkpoint until the complete training receipt exists.
    metadata = {
        "best_epoch": best_epoch, "best_source_val_loss": best_loss,
        "epochs_completed": len(logs),
        "source_actual_loader_order_hash": hash_json(source_epoch_order_hashes),
        "target_actual_loader_order_hash": hash_json(target_epoch_order_hashes),
        "source_epoch_order_hashes": source_epoch_order_hashes,
        "target_epoch_order_hashes": target_epoch_order_hashes,
    }
    return model, device, logs, metadata


def train_variant(data, target_x, initial_state, settings, variant, seed, output_dir, *, cpu, resume, required_epochs=None):
    import torch
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {'protocol_id': PROTOCOL_ID, 'source_membership': data['membership_hashes'],
                'target_hash': hash_array(target_x), 'initial_state': hash_state_dict(initial_state),
                'settings': asdict(settings), 'variant': variant, 'seed': seed,
                'required_epochs': required_epochs, 'cpu': cpu, 'method': METHOD, 'scale': SCALE, 'bandwidth': BANDWIDTH}
    write_immutable_json(output_dir / 'training_contract.json', contract)
    if METHOD == 'strength' and variant == 'H3-C0' and REFERENCE is not None:
        return reuse_zero(data, target_x, initial_state, settings, seed, output_dir, cpu)
    done = output_dir / 'TRAINED.json' 
    if done.exists():
        result = json.loads(done.read_text())
        for name, digest in result['files'].items():
            if hash_file(output_dir / name) != digest:
                raise ProtocolError(f'Training artifact changed: {output_dir / name}')
        device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
        model = DANN(data['X_train'].shape[1], dropout=settings.dropout, grl_alpha=1.0).to(device)
        model.load_state_dict(torch.load(output_dir / 'best.pt', map_location=device, weights_only=True))
        model.eval()
        return model, device, result['logs'], result['metadata']
    set_all_seeds(seed)
    model, device, logs, metadata = train_variant_raw(data, target_x, initial_state, settings, variant, seed,
                                                     output_dir, cpu=cpu, resume=resume, required_epochs=required_epochs)
    atomic_write_json(done, {'protocol_id': PROTOCOL_ID, 'logs': logs, 'metadata': metadata,
                            'files': {name: hash_file(output_dir / name) for name in ('best.pt', 'final.pt', 'training_contract.json')}})
    (output_dir / 'resume.pt').unlink(missing_ok=True)
    return model, device, logs, metadata


def reuse_zero(data, target_x, initial_state, settings, seed, output_dir, cpu):
    import torch
    source = REFERENCE / 'H3-C0'
    phase2_h3.validate_completed_artifact_hashes(REFERENCE, json.loads((REFERENCE/'COMPLETED.json').read_text()))
    receipt = json.loads((source/'TRAINED.json').read_text())
    for name, digest in receipt['files'].items():
        if hash_file(source/name) != digest:
            raise ProtocolError(f'Phase 2 zero artifact changed: {source/name}')
    old = json.loads((source/'training_contract.json').read_text())
    expected = {'source_membership': data['membership_hashes'], 'target_hash': hash_array(target_x),
                'initial_state': hash_state_dict(initial_state), 'settings': asdict(settings), 'seed': seed, 'variant': 'H3-C0', 'required_epochs': None, 'cpu': cpu}
    # Evaluation chunk size does not affect training.
    old['settings']['chunk_size'] = settings.chunk_size
    for key,value in expected.items():
        if old[key] != value:
            raise ProtocolError(f'Cannot share Phase 2 C0: {key} differs')
    for name in ('best.pt','final.pt'):
        shutil.copy2(source/name, output_dir/name)
    write_immutable_json(output_dir/'REUSED.json', {'source': str(source), 'receipt_hash': hash_file(source/'TRAINED.json'),
                         'source_completed_hash': hash_file(REFERENCE/'COMPLETED.json')})
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    model = DANN(49,dropout=0.,grl_alpha=1.).to(device)
    model.load_state_dict(torch.load(source/'best.pt', map_location=device, weights_only=True))
    model.eval()
    return model, device, receipt['logs'], receipt['metadata']


def reuse_evaluation():
    table = pd.read_csv(REFERENCE/'paired_metrics.csv')
    metrics = table[table.variant == 'H3-C0'].iloc[0].to_dict()
    reservoir = {key: metrics[key] for key in ('method','capacity','stream_rows_seen','sample_size','sample_ids_hash','priority_hash')}
    with np.load(REFERENCE/'H3-C0/reservoir_embeddings.npz') as arrays:
        return (metrics, arrays['target_embeddings'], arrays['target_labels'], arrays['target_ids'], reservoir,
                arrays['source_embeddings'], arrays['source_labels'], arrays['source_ids'])

