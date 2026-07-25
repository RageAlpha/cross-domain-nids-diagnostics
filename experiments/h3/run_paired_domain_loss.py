"""Frozen H3 matched DANN-vs-lambda-zero confirmation runner.

The safe default is ``plan``.  Commands that train, evaluate, or write protocol
state require ``--execute``.  C0 and C1 share source data, target sample,
initial state, loader-order seeds, optimizer settings, and stopping rule; the
only intended difference is the domain-loss coefficient schedule.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from cross_domain_nids.config import ATTACK_COL, LABEL_COL, STRESS_B_CHUNK_SIZE, UNSW_PATH
from cross_domain_nids.models.dann import DANN
from cross_domain_nids.preprocessing.netflow import prepare_cicids_chunk
from cross_domain_nids.protocol_utils import (
    PROTOCOL_ID, PROTOCOL_RESULTS_ROOT, ProtocolError, atomic_write_json,
    audited_run, completed_marker, deterministic_seed, environment_manifest,
    file_fingerprint, git_manifest, hash_array, hash_file, hash_json,
    hash_state_dict, paired_bootstrap, require_execute_flag,
    source_tree_manifest, utc_now, validate_confusion_counts,
    validate_feature_order, validate_protocol_lock_checklist,
    write_immutable_json, write_protocol_lock_checklist,
)
from experiments.h3.train_domain_loss_models import (
    TARGET_DATASETS, binary_metrics_from_counts, domain_discriminability,
    freeze_batchnorm_running_stats,
    label_free_target_sample, predict_logits_numpy, prepare_source_splits,
    set_all_seeds, source_validation_metrics,
    train_source_mlp,
)


SEEDS = [42, 123, 2024]
VARIANTS = {"H3-C0": "constant_0", "H3-C1": "linear_0_1"}
H3_COMPLETED_ARTIFACTS = (
    "paired_metrics.csv", "domain_probe.csv", "mmd.csv",
    "class_conditional_probe.csv", "training_log.csv",
    "H3-C0/reservoir_embeddings.npz", "H3-C1/reservoir_embeddings.npz",
    "H3-C0/best.pt", "H3-C1/best.pt",
    "H3-C0/final.pt", "H3-C1/final.pt",
)
PAIR_KEYS = [
    "source_split_hash", "feature_order_hash", "preprocessor_hash",
    "target_index_hash", "target_sample_hash", "initial_state_hash",
    "source_loader_seed", "target_loader_seed", "batch_size",
    "learning_rate", "dropout", "max_epochs", "patience", "min_epochs",
]


@dataclass(frozen=True)
class H3Settings:
    batch_size: int = 4096
    learning_rate: float = 1e-3
    dropout: float = 0.0
    max_epochs: int = 40
    patience: int = 5
    min_epochs: int = 20
    target_ratio: float = 0.05
    source_split_seed: int = 42
    threshold: float = 0.5
    embedding_sample_size: int = 100_000
    chunk_size: int = STRESS_B_CHUNK_SIZE


def unit_dir(root: Path, target: str, seed: int) -> Path:
    return root / "h3" / "runs" / target / f"seed{seed}"


def build_artifact_hashes(run_dir: Path) -> dict[str, str]:
    missing = [relative for relative in H3_COMPLETED_ARTIFACTS if not (run_dir / relative).is_file()]
    if missing:
        raise ProtocolError(f"Cannot complete H3 unit; missing artifacts: {missing}")
    return {relative: hash_file(run_dir / relative) for relative in H3_COMPLETED_ARTIFACTS}


def validate_completed_artifact_hashes(run_dir: Path, completed: dict[str, Any]) -> None:
    locked = completed.get("artifact_hashes")
    if not isinstance(locked, dict) or set(locked) != set(H3_COMPLETED_ARTIFACTS):
        raise ProtocolError(f"H3 completed artifact hash set is incomplete: {run_dir}")
    actual = build_artifact_hashes(run_dir)
    mismatches = {
        relative: (locked.get(relative), actual[relative])
        for relative in H3_COMPLETED_ARTIFACTS
        if locked.get(relative) != actual[relative]
    }
    if mismatches:
        raise ProtocolError(f"H3 completed artifacts changed after completion: {run_dir}: {mismatches}")


def state_hash_for_object(value: Any) -> str:
    """Stable hash for a fitted sklearn preprocessor."""
    import joblib
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
        path = Path(handle.name)
    try:
        joblib.dump(value, path)
        return hash_file(path)
    finally:
        path.unlink(missing_ok=True)


def exact_target_indices(total: int, ratio: float, seed: int) -> np.ndarray:
    size = int(total * ratio)
    return np.sort(np.random.default_rng(seed).choice(total, size=size, replace=False))


def loader_order_hash(length: int, batch_size: int, seed: int, epochs: int) -> str:
    """Hash indices actually yielded by DataLoader, including iterator RNG use."""
    import torch
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(torch.arange(length, dtype=torch.int64))
    epoch_hashes = []
    for _ in range(epochs):
        digest = hashlib.sha256()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=False,
        )
        for (indices,) in loader:
            digest.update(np.asarray(indices, dtype=np.int64).tobytes())
            digest.update(len(indices).to_bytes(8, "little"))
        epoch_hashes.append(digest.hexdigest())
    return hash_json(epoch_hashes)


def _update_order_digest(digest: Any, indices: Any) -> None:
    values = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    digest.update(values.tobytes())
    digest.update(len(values).to_bytes(8, "little"))


def _optimizer_to_device(optimizer: Any, device: Any) -> None:
    import torch
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_rng_states(
    checkpoint: dict[str, Any], source_generator: Any,
    target_generator: Any, device: Any,
) -> None:
    import torch
    source_generator.set_state(checkpoint["source_generator_state"].cpu())
    target_generator.set_state(checkpoint["target_generator_state"].cpu())
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])


def snapshot_bn(model: Any) -> dict[str, np.ndarray]:
    import torch.nn as nn
    result = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm1d):
            result[f"{name}.running_mean"] = layer.running_mean.detach().cpu().numpy().copy()
            result[f"{name}.running_var"] = layer.running_var.detach().cpu().numpy().copy()
            result[f"{name}.num_batches_tracked"] = layer.num_batches_tracked.detach().cpu().numpy().copy()
    return result


def assert_bn_unchanged(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> None:
    changed = [key for key in before if not np.array_equal(before[key], after[key])]
    if changed:
        raise ProtocolError(f"Target forward changed frozen BatchNorm buffers: {changed}")


def save_training_checkpoint(path: Path, **payload: Any) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train_variant(
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
    label_loss_fn = nn.CrossEntropyLoss()
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
        source_loader = torch.utils.data.DataLoader(
            source_dataset, batch_size=settings.batch_size, shuffle=True,
            generator=source_generator, drop_last=False,
        )
        target_loader = torch.utils.data.DataLoader(
            target_dataset, batch_size=settings.batch_size, shuffle=True,
            generator=target_generator, drop_last=False,
        )
        model.train()
        coefficient = 0.0 if schedule == "constant_0" else epoch / max(settings.max_epochs - 1, 1)
        target_iter = iter(target_loader)
        source_order_digest = hashlib.sha256()
        target_order_digest = hashlib.sha256()
        totals = {"label": 0.0, "domain": 0.0, "source_n": 0, "domain_n": 0}
        for source_x, source_y, source_indices in source_loader:
            try:
                target_batch, target_indices = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch, target_indices = next(target_iter)
            _update_order_digest(source_order_digest, source_indices)
            _update_order_digest(target_order_digest, target_indices)
            source_x, source_y, target_batch = source_x.to(device), source_y.to(device), target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            label_logits, source_features = model.forward_label(source_x, return_features=True)
            label_loss = label_loss_fn(label_logits, source_y)
            before = snapshot_bn(model)
            with freeze_batchnorm_running_stats(model):
                _, target_features = model.forward_label(target_batch, return_features=True)
            assert_bn_unchanged(before, snapshot_bn(model))
            domain_features = torch.cat([source_features, target_features])
            domain_y = torch.cat([
                torch.zeros(len(source_features), dtype=torch.long, device=device),
                torch.ones(len(target_features), dtype=torch.long, device=device),
            ])
            domain_logits = model.forward_domain(domain_features)
            domain_loss = domain_loss_fn(domain_logits, domain_y)
            (label_loss + coefficient * domain_loss).backward()
            optimizer.step()
            totals["label"] += float(label_loss.detach().cpu()) * len(source_x)
            totals["domain"] += float(domain_loss.detach().cpu()) * len(domain_y)
            totals["source_n"] += len(source_x)
            totals["domain_n"] += len(domain_y)

        model.eval()
        with torch.no_grad():
            val_x = torch.as_tensor(data["X_val"], dtype=torch.float32, device=device)
            val_y = torch.as_tensor(data["y_val"], dtype=torch.long, device=device)
            val_loss = float(label_loss_fn(model.forward_label(val_x), val_y).detach().cpu())
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
    checkpoint_path.unlink(missing_ok=True)
    metadata = {
        "best_epoch": best_epoch, "best_source_val_loss": best_loss,
        "epochs_completed": len(logs),
        "source_actual_loader_order_hash": hash_json(source_epoch_order_hashes),
        "target_actual_loader_order_hash": hash_json(target_epoch_order_hashes),
        "source_epoch_order_hashes": source_epoch_order_hashes,
        "target_epoch_order_hashes": target_epoch_order_hashes,
    }
    return model, device, logs, metadata


def shared_rbf_bandwidth(arrays: list[np.ndarray], seed: int, max_n: int = 1000) -> float:
    rng = np.random.default_rng(seed)
    sampled = []
    per_array = max(2, max_n // len(arrays))
    for array in arrays:
        ids = np.sort(rng.choice(len(array), min(len(array), per_array), replace=False))
        sampled.append(array[ids].astype(np.float64))
    joined = np.vstack(sampled)
    distances = ((joined[:, None, :] - joined[None, :, :]) ** 2).sum(axis=2)
    positive = distances[np.triu_indices_from(distances, k=1)]
    bandwidth_sq = float(np.median(positive[positive > 0]))
    if not np.isfinite(bandwidth_sq) or bandwidth_sq <= 0:
        raise ProtocolError("Unable to determine positive shared RBF bandwidth")
    return bandwidth_sq


def rbf_mmd(
    source: np.ndarray, target: np.ndarray, seed: int, bandwidth_sq: float,
    max_n: int = 2000,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = min(len(source), len(target), max_n)
    if n < 2:
        raise ProtocolError("MMD requires at least two source and target embeddings")
    source_ids = np.sort(rng.choice(len(source), n, replace=False))
    target_ids = np.sort(rng.choice(len(target), n, replace=False))
    x, y = source[source_ids].astype(np.float64), target[target_ids].astype(np.float64)
    kernel = lambda a, b: np.exp(-((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2) / (2 * bandwidth_sq))
    kxx, kyy, kxy = kernel(x, x), kernel(y, y), kernel(x, y)
    np.fill_diagonal(kxx, 0.0); np.fill_diagonal(kyy, 0.0)
    value = kxx.sum() / (n * (n - 1)) + kyy.sum() / (n * (n - 1)) - 2 * kxy.mean()
    return {
        "mmd_rbf_unbiased": float(value), "bandwidth_sq": bandwidth_sq, "n_per_domain": n,
        "source_ids_hash": hash_array(source_ids), "target_ids_hash": hash_array(target_ids),
    }


def class_conditional_rows(
    source_emb: np.ndarray, source_labels: np.ndarray, target_emb: np.ndarray,
    target_labels: np.ndarray, *, target: str, seed: int, variant: str,
) -> list[dict[str, Any]]:
    rows = []
    for label, name in ((0, "benign"), (1, "attack")):
        source_class = source_emb[source_labels == label]
        target_class = target_emb[target_labels == label]
        if len(source_class) == 0 or len(target_class) == 0:
            rows.append({"target": target, "seed": seed, "variant": variant,
                         "class_label": label, "class_name": name, "status": "missing_class"})
            continue
        centroid = float(np.linalg.norm(source_class.mean(axis=0) - target_class.mean(axis=0)))
        for probe_seed in (42, 123, 2024):
            rows.append({
                "target": target, "seed": seed, "variant": variant,
                "class_label": label, "class_name": name, "status": "complete",
                "centroid_distance": centroid, "probe_seed": probe_seed,
                **domain_discriminability(source_class, target_class, probe_seed),
            })
    return rows


class PriorityReservoir:
    """Uniform full-stream reservoir via deterministic bottom-k random priorities."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.features = np.empty((0, 64), dtype=np.float32)
        self.labels = np.empty((0,), dtype=np.int64)
        self.sample_ids = np.empty((0,), dtype=np.int64)
        self.priorities = np.empty((0,), dtype=np.float64)
        self.seen = 0

    def update(self, features: np.ndarray, labels: np.ndarray, sample_ids: np.ndarray) -> None:
        if not (len(features) == len(labels) == len(sample_ids)):
            raise ProtocolError("Reservoir update arrays have inconsistent lengths")
        priorities = self.rng.random(len(features))
        all_features = np.vstack([self.features, features.astype(np.float32, copy=False)])
        all_labels = np.concatenate([self.labels, labels.astype(np.int64, copy=False)])
        all_ids = np.concatenate([self.sample_ids, sample_ids.astype(np.int64, copy=False)])
        all_priorities = np.concatenate([self.priorities, priorities])
        keep_n = min(self.capacity, len(all_priorities))
        keep = np.argpartition(all_priorities, keep_n - 1)[:keep_n] if keep_n else np.empty(0, dtype=int)
        order = np.argsort(all_ids[keep], kind="stable")
        keep = keep[order]
        self.features = all_features[keep]
        self.labels = all_labels[keep]
        self.sample_ids = all_ids[keep]
        self.priorities = all_priorities[keep]
        self.seen += len(features)

    def metadata(self) -> dict[str, Any]:
        return {
            "method": "full_stream_bottom_k_random_priority_reservoir",
            "capacity": self.capacity, "stream_rows_seen": self.seen,
            "sample_size": len(self.sample_ids),
            "sample_ids_hash": hash_array(self.sample_ids),
            "priority_hash": hash_array(self.priorities),
        }


def evaluate_full_target_reservoir(
    torch: Any, model: Any, device: Any, preprocessor: Any,
    feature_cols: list[str], dataset_name: str, dataset_path: str | Path,
    seed: int, variant: str, output_dir: Path, chunk_size: int,
    reservoir_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate every target row and sample embeddings across the entire stream."""
    reservoir = PriorityReservoir(reservoir_size, seed)
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    sample_offset = 0
    prediction_path = output_dir / "predictions" / f"{variant}_UNSW_to_{dataset_name}_seed{seed}_predictions.csv"
    prediction_path.unlink(missing_ok=True)
    wrote_header = False
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size, low_memory=False):
        chunk = prepare_cicids_chunk(chunk)
        y_true = chunk[LABEL_COL].astype(int).to_numpy()
        x_frame = chunk.drop(columns=[col for col in (LABEL_COL, ATTACK_COL) if col in chunk.columns])
        for feature in feature_cols:
            if feature not in x_frame.columns:
                x_frame[feature] = 0
        transformed = preprocessor.transform(x_frame[feature_cols]).astype(np.float32)
        logits, features = predict_logits_numpy(torch, model, transformed, device)
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        predictions = (probabilities[:, 1] >= 0.5).astype(int)
        counts["tp"] += int(((y_true == 1) & (predictions == 1)).sum())
        counts["tn"] += int(((y_true == 0) & (predictions == 0)).sum())
        counts["fp"] += int(((y_true == 0) & (predictions == 1)).sum())
        counts["fn"] += int(((y_true == 1) & (predictions == 0)).sum())
        ids = np.arange(sample_offset, sample_offset + len(y_true), dtype=np.int64)
        reservoir.update(features, y_true, ids)
        pd.DataFrame({
            "sample_id": ids, "source_dataset": "UNSW", "target_dataset": dataset_name,
            "seed": seed, "variant": variant, "y_true_binary": y_true,
            "y_pred_binary": predictions, "p_benign": probabilities[:, 0],
            "p_attack": probabilities[:, 1],
        }).to_csv(prediction_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        sample_offset += len(y_true)
        del chunk, x_frame, transformed, logits, features, probabilities, predictions
        gc.collect()
    metrics = binary_metrics_from_counts(counts["tp"], counts["tn"], counts["fp"], counts["fn"])
    metadata = reservoir.metadata()
    if metadata["stream_rows_seen"] != sample_offset:
        raise ProtocolError("Reservoir did not observe every evaluated target row")
    return metrics, reservoir.features, reservoir.labels, reservoir.sample_ids, metadata


def source_reservoir_embeddings(
    torch: Any, model: Any, device: Any, data: dict[str, Any], seed: int,
    reservoir_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(data["X_test"]), reservoir_size)
    ids = np.sort(np.random.default_rng(seed).choice(len(data["X_test"]), size=n, replace=False))
    _, embeddings = predict_logits_numpy(torch, model, data["X_test"][ids], device)
    return embeddings, data["y_test"][ids], ids.astype(np.int64)


def run_pair(args: argparse.Namespace) -> None:
    require_execute_flag(args.execute, "H3 paired training")
    load_policy(args.policy)
    validate_protocol_lock_checklist(
        args.output_root, policy_paths={"h2": args.h2_policy, "h3": args.policy},
        seeds=SEEDS, targets=list(TARGET_DATASETS), data_paths=protocol_data_paths(),
        code_paths=protocol_code_paths(),
    )
    import joblib
    import torch

    settings = H3Settings(chunk_size=args.chunk_size)
    out = unit_dir(args.output_root, args.target, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    if (out / "COMPLETED.json").exists():
        print(f"Already complete: {out}")
        return
    with audited_run(out, {"stage": "H3", "target": args.target, "seed": args.seed}):
        set_all_seeds(args.seed)
        data = prepare_source_splits(split_seed=settings.source_split_seed)
        validate_feature_order(data["feature_cols"])
        target_x, sample_meta = label_free_target_sample(
            args.target, TARGET_DATASETS[args.target], data["feature_cols"], data["preprocessor"],
            args.seed, settings.target_ratio, None, settings.chunk_size, "exact_random_5pct",
        )
        indices = exact_target_indices(sample_meta["target_total_samples"], settings.target_ratio, args.seed)
        if len(indices) != len(target_x):
            raise ProtocolError("Reconstructed target index count differs from transformed target sample")
        preprocessor_path = out / "preprocessor.joblib"
        joblib.dump(data["preprocessor"], preprocessor_path)
        set_all_seeds(args.seed)
        initial_model = DANN(data["X_train"].shape[1], dropout=settings.dropout, grl_alpha=1.0)
        initial_state = {key: value.detach().cpu().clone() for key, value in initial_model.state_dict().items()}
        torch.save(initial_state, out / "initial.pt")
        common = {
            "protocol_id": PROTOCOL_ID, "target": args.target, "seed": args.seed,
            "feature_order": data["feature_cols"], "feature_order_hash": hash_json(data["feature_cols"]),
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
            }, "only_intended_difference": "lambda_schedule/domain_loss_coefficient",
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
            rows.append({"protocol_id": PROTOCOL_ID, "target": args.target, "seed": args.seed,
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
        completed_marker(out, {
            "pair_manifest_hash": hash_file(out / "pair_manifest.json"),
            "artifact_hashes": build_artifact_hashes(out),
            "status": "complete",
        })


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ProtocolError(
            f"Frozen H3 Section 4.8 policy is required for verdict generation: {path}. "
            "Metric aggregation may proceed, but verdicts must not be invented."
        )
    policy = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_id", "seeds", "targets", "variants", "probe_seeds",
        "mcc_min_delta", "binary_f1_min_delta", "fnr_max_increase",
        "required_positive_seeds", "required_seed_count", "representation_required",
        "domain_discriminability_abs_max_delta", "mmd_max_delta",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ProtocolError(f"H3 verdict policy missing keys: {missing}")
    expected = {
        "protocol_id": PROTOCOL_ID, "seeds": SEEDS, "targets": list(TARGET_DATASETS),
        "variants": list(VARIANTS), "probe_seeds": SEEDS,
        "required_positive_seeds": len(SEEDS), "required_seed_count": len(SEEDS),
    }
    mismatches = {key: (policy.get(key), value) for key, value in expected.items() if policy.get(key) != value}
    if mismatches:
        raise ProtocolError(f"H3 policy is not frozen for this protocol: {mismatches}")
    return policy


def domain_discriminability_score(auroc: float) -> float:
    return abs(float(auroc) - 0.5)


def h3_performance_pass(group: pd.DataFrame, policy: dict[str, Any]) -> bool:
    return bool(
        len(group) == policy["required_seed_count"]
        and group["delta_MCC"].mean() >= policy["mcc_min_delta"]
        and group["delta_Attack-class Binary-F1"].mean() >= policy["binary_f1_min_delta"]
        and group["delta_FNR"].mean() <= policy["fnr_max_increase"]
        and int((group["delta_MCC"] > 0).sum()) >= policy["required_positive_seeds"]
        and int((group["delta_Attack-class Binary-F1"] > 0).sum()) >= policy["required_positive_seeds"]
    )


def protocol_data_paths() -> list[str | Path]:
    return [UNSW_PATH, *TARGET_DATASETS.values()]


def protocol_code_paths() -> list[Path]:
    return [ROOT / "experiments", ROOT / "models"]


def validate_h3_coverage(root: Path, policy: dict[str, Any]) -> dict[str, pd.DataFrame]:
    expected_units = {(target, seed) for target in policy["targets"] for seed in policy["seeds"]}
    discovered_paths = set((root / "h3" / "runs").glob("*/seed*/paired_metrics.csv"))
    expected_paths = {unit_dir(root, target, seed) / "paired_metrics.csv" for target, seed in expected_units}
    if discovered_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - discovered_paths)
        extra = sorted(str(path) for path in discovered_paths - expected_paths)
        raise ProtocolError(f"H3 coverage mismatch; missing={missing}, unexpected={extra}")
    metric_frames, mmd_frames, probe_frames, class_frames = [], [], [], []
    expected_variants = set(policy["variants"])
    expected_probes = set(policy["probe_seeds"])
    for target, seed in sorted(expected_units):
        run_dir = unit_dir(root, target, seed)
        for required_file in ("COMPLETED.json", "pair_manifest.json", "pair_execution_manifest.json"):
            if not (run_dir / required_file).exists():
                raise ProtocolError(f"Missing H3 audit artifact: {run_dir / required_file}")
        completed = json.loads((run_dir / "COMPLETED.json").read_text(encoding="utf-8"))
        execution = json.loads((run_dir / "pair_execution_manifest.json").read_text(encoding="utf-8"))
        if completed.get("protocol_id") != PROTOCOL_ID or execution.get("protocol_id") != PROTOCOL_ID:
            raise ProtocolError(f"Protocol ID mismatch in {run_dir}")
        if execution.get("status") != "passed" or not all(execution.get("checks", {}).values()):
            raise ProtocolError(f"Pair execution invariants did not pass: {run_dir}")
        if completed.get("pair_manifest_hash") != hash_file(run_dir / "pair_manifest.json"):
            raise ProtocolError(f"Completed marker does not match pair manifest: {run_dir}")
        validate_completed_artifact_hashes(run_dir, completed)
        metrics = pd.read_csv(run_dir / "paired_metrics.csv")
        if len(metrics) != 2 or set(metrics["variant"]) != expected_variants or metrics["variant"].duplicated().any():
            raise ProtocolError(f"Incomplete/duplicate C0-C1 metrics: {run_dir}")
        if set(metrics["protocol_id"]) != {PROTOCOL_ID} or set(metrics["seed"].astype(int)) != {seed} or set(metrics["target"]) != {target}:
            raise ProtocolError(f"H3 metric identity mismatch: {run_dir}")
        mmd = pd.read_csv(run_dir / "mmd.csv")
        if len(mmd) != 2 or set(mmd["variant"]) != expected_variants or mmd["variant"].duplicated().any():
            raise ProtocolError(f"Incomplete/duplicate MMD coverage: {run_dir}")
        if set(mmd["target"]) != {target} or set(mmd["seed"].astype(int)) != {seed}:
            raise ProtocolError(f"MMD identity mismatch: {run_dir}")
        if mmd["shared_bandwidth_hash"].nunique() != 1:
            raise ProtocolError(f"C0/C1 do not share MMD bandwidth: {run_dir}")
        probes = pd.read_csv(run_dir / "domain_probe.csv")
        expected_probe_keys = {(variant, probe_seed) for variant in expected_variants for probe_seed in expected_probes}
        actual_probe_keys = set(zip(probes["variant"], probes["probe_seed"].astype(int)))
        if len(probes) != len(expected_probe_keys) or actual_probe_keys != expected_probe_keys:
            raise ProtocolError(f"Domain-probe coverage mismatch: {run_dir}")
        if set(probes["target"]) != {target} or set(probes["seed"].astype(int)) != {seed}:
            raise ProtocolError(f"Domain-probe identity mismatch: {run_dir}")
        classes = pd.read_csv(run_dir / "class_conditional_probe.csv")
        expected_class_keys = {
            (variant, label, probe_seed)
            for variant in expected_variants for label in (0, 1) for probe_seed in expected_probes
        }
        actual_class_keys = set(zip(classes["variant"], classes["class_label"].astype(int), classes["probe_seed"].astype(int)))
        if len(classes) != len(expected_class_keys) or actual_class_keys != expected_class_keys or set(classes["status"]) != {"complete"}:
            raise ProtocolError(f"Class-conditional coverage mismatch: {run_dir}")
        if set(classes["target"]) != {target} or set(classes["seed"].astype(int)) != {seed}:
            raise ProtocolError(f"Class-conditional identity mismatch: {run_dir}")
        for variant in expected_variants:
            reservoir_path = run_dir / variant / "reservoir_embeddings.npz"
            if not reservoir_path.exists():
                raise ProtocolError(f"Missing reservoir artifact: {reservoir_path}")
            with np.load(reservoir_path) as reservoir:
                method = str(reservoir["reservoir_method"])
                rows_seen = int(reservoir["target_stream_rows_seen"])
                if method != "full_stream_bottom_k_random_priority_reservoir" or rows_seen <= len(reservoir["target_ids"]):
                    raise ProtocolError(f"Artifact is not a full-target reservoir: {reservoir_path}")
        metric_frames.append(metrics); mmd_frames.append(mmd)
        probe_frames.append(probes); class_frames.append(classes)
    return {
        "metrics": pd.concat(metric_frames, ignore_index=True),
        "mmd": pd.concat(mmd_frames, ignore_index=True),
        "probes": pd.concat(probe_frames, ignore_index=True),
        "classes": pd.concat(class_frames, ignore_index=True),
    }


def analyze(args: argparse.Namespace) -> None:
    require_execute_flag(args.execute, "H3 analysis writing")
    policy = load_policy(args.policy)
    validate_protocol_lock_checklist(
        args.output_root, policy_paths={"h2": args.h2_policy, "h3": args.policy},
        seeds=SEEDS, targets=list(TARGET_DATASETS), data_paths=protocol_data_paths(),
        code_paths=protocol_code_paths(),
    )
    coverage = validate_h3_coverage(args.output_root, policy)
    long = coverage["metrics"]
    paired_rows, bootstrap_rows = [], []
    for (target, seed), group in long.groupby(["target", "seed"]):
        pivot = group.set_index("variant")
        row = {"target": target, "seed": int(seed)}
        for metric in ("MCC", "Attack-class Binary-F1", "FNR", "FPR"):
            row[f"delta_{metric}"] = float(pivot.loc["H3-C1", metric] - pivot.loc["H3-C0", metric])
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    expected_pairs = len(policy["targets"]) * len(policy["seeds"])
    if len(paired) != expected_pairs:
        raise ProtocolError(f"Expected {expected_pairs} H3 pairs; got {len(paired)}")
    analysis_dir = args.output_root / "h3" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(analysis_dir / "paired_per_seed.csv", index=False)
    for target, group in paired.groupby("target"):
        for metric in ("MCC", "Attack-class Binary-F1", "FNR", "FPR"):
            bootstrap_rows.append({"target": target, "metric": metric,
                                   **paired_bootstrap(group[f"delta_{metric}"].to_numpy(), seed=deterministic_seed(target, metric))})
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(analysis_dir / "bootstrap_10000.csv", index=False)
    mmd = coverage["mmd"].pivot(index=["target", "seed"], columns="variant", values="mmd_rbf_unbiased")
    probe_long = coverage["probes"].copy()
    probe_long["domain_discriminability"] = probe_long["domain_AUROC"].map(domain_discriminability_score)
    probe = probe_long.groupby(["target", "seed", "variant"], as_index=False)["domain_discriminability"].mean().pivot(
        index=["target", "seed"], columns="variant", values="domain_discriminability"
    )
    mechanism = {}
    for target in policy["targets"]:
        mechanism[target] = {
            "mean_mmd_delta": float((mmd.loc[target, "H3-C1"] - mmd.loc[target, "H3-C0"]).mean()),
            "mean_domain_discriminability_delta": float((probe.loc[target, "H3-C1"] - probe.loc[target, "H3-C0"]).mean()),
        }
    verdicts = []
    for target, group in paired.groupby("target"):
        means = group.mean(numeric_only=True)
        passed = h3_performance_pass(group, policy)
        mechanism_values = mechanism.get(target, {})
        mechanism_passed = (
            not policy["representation_required"]
            or (
                mechanism_values.get("mean_mmd_delta", math.inf) <= policy["mmd_max_delta"]
                and mechanism_values.get("mean_domain_discriminability_delta", math.inf)
                <= policy["domain_discriminability_abs_max_delta"]
            )
        )
        verdicts.append({"target": target, "verdict": "Pass" if passed and mechanism_passed else "Fail", **means.to_dict(),
                         **mechanism_values, "mechanism_passed": mechanism_passed,
                         "policy_hash": hash_file(args.policy)})
    pd.DataFrame(verdicts).to_csv(analysis_dir / "verdict_by_target.csv", index=False)


def run_source_only(args: argparse.Namespace) -> None:
    """Train S0 once for a seed and reuse it for all three target evaluations."""
    require_execute_flag(args.execute, "H3 source-only supporting boundary")
    load_policy(args.policy)
    validate_protocol_lock_checklist(
        args.output_root, policy_paths={"h2": args.h2_policy, "h3": args.policy},
        seeds=SEEDS, targets=list(TARGET_DATASETS), data_paths=protocol_data_paths(),
        code_paths=protocol_code_paths(),
    )
    import torch
    settings = H3Settings(chunk_size=args.chunk_size)
    out = args.output_root / "h3" / "source_only" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True); (out / "predictions").mkdir(exist_ok=True)
    if (out / "COMPLETED.json").exists():
        return
    with audited_run(out, {"stage": "H3-S0", "seed": args.seed}):
        data = prepare_source_splits(split_seed=settings.source_split_seed)
        train_args = SimpleNamespace(
            cpu=args.cpu, dropout=settings.dropout, learning_rate=settings.learning_rate,
            batch_size=settings.batch_size, epochs=settings.max_epochs, patience=settings.patience,
            min_epochs_before_early_stop=settings.min_epochs,
        )
        model, device, metadata = train_source_mlp(
            data, data["X_train"].shape[1], train_args, args.seed,
            "H3-S0", "all_targets_shared_training", [],
        )
        torch.save(model.state_dict(), out / "best.pt")
        rows = []
        for target, path in TARGET_DATASETS.items():
            metrics, _, _, _, _ = evaluate_full_target_reservoir(
                torch, model, device, data["preprocessor"], data["feature_cols"],
                target, path, args.seed, "H3-S0", out, settings.chunk_size,
                settings.embedding_sample_size,
            )
            rows.append({"protocol_id": PROTOCOL_ID, "seed": args.seed, "target": target,
                         "variant": "H3-S0", **metadata, **metrics})
        pd.DataFrame(rows).to_csv(out / "source_only_metrics.csv", index=False)
        completed_marker(out, {"status": "complete", "training_reused_for_targets": list(TARGET_DATASETS)})


def initialise(args: argparse.Namespace) -> None:
    require_execute_flag(args.execute, "H3 protocol initialization")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    load_policy(args.policy)
    h2_policy = json.loads(args.h2_policy.read_text(encoding="utf-8"))
    if h2_policy.get("protocol_id") != PROTOCOL_ID or h2_policy.get("seeds") != SEEDS:
        raise ProtocolError("H2 policy does not match the v2 three-seed protocol")
    policy_hashes = {"h2": hash_file(args.h2_policy), "h3": hash_file(args.policy)}
    protocol = {
        "protocol_id": PROTOCOL_ID, "created_utc": utc_now(), "stage": "H3",
        "settings": asdict(H3Settings(chunk_size=args.chunk_size)),
        "targets": list(TARGET_DATASETS), "seeds": SEEDS, "variants": VARIANTS,
        "policy_files": {"h2": str(args.h2_policy), "h3": str(args.policy)},
        "policy_hashes": policy_hashes,
    }
    write_immutable_json(root / "protocol.json", protocol)
    atomic_write_json(root / "environment_manifest.json", environment_manifest())
    atomic_write_json(root / "git_manifest.json", git_manifest(ROOT))
    atomic_write_json(root / "code_manifest.json", source_tree_manifest([ROOT / "experiments", ROOT / "models"]))
    data_paths = protocol_data_paths()
    atomic_write_json(root / "data_manifest.json", [file_fingerprint(path) for path in data_paths])
    write_protocol_lock_checklist(
        root, policy_paths={"h2": args.h2_policy, "h3": args.policy},
        seeds=SEEDS, targets=list(TARGET_DATASETS), data_paths=data_paths,
        code_paths=protocol_code_paths(),
    )


def print_plan(args: argparse.Namespace) -> None:
    plan = {
        "protocol_id": PROTOCOL_ID, "execution_order": ["H3", "H2"],
        "output_root": str(args.output_root), "settings": asdict(H3Settings(chunk_size=args.chunk_size)),
        "primary_pairs": len(TARGET_DATASETS) * len(SEEDS), "primary_runs": len(TARGET_DATASETS) * len(SEEDS) * 2,
        "source_only_runs": len(SEEDS), "targets": list(TARGET_DATASETS), "seeds": SEEDS,
        "safety": "No execution occurs without --execute",
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "init", "pair", "s0", "analyze"], nargs="?", default="plan")
    parser.add_argument("--output-root", type=Path, default=PROTOCOL_RESULTS_ROOT)
    parser.add_argument("--target", choices=list(TARGET_DATASETS), default="CICIDS2018")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--policy", type=Path, default=ROOT / "protocols" / "h3" / "verdict_policy.json")
    parser.add_argument("--h2-policy", type=Path, default=ROOT / "protocols" / "h2" / "selection_policy.json")
    parser.add_argument("--chunk-size", type=int, default=STRESS_B_CHUNK_SIZE)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    expected_root = PROTOCOL_RESULTS_ROOT.resolve()
    if args.output_root != expected_root and expected_root not in args.output_root.parents:
        raise ProtocolError(f"H3 output must remain under {expected_root}; got {args.output_root}")
    {"plan": print_plan, "init": initialise, "pair": run_pair, "s0": run_source_only,
     "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
