"""Audit primitives for the frozen H2/H3 confirmation protocol.

This module deliberately contains no experiment-specific training logic.  It
provides canonical hashing, immutable JSON writes, run manifests, completion
markers, paired bootstrap helpers, and invariant validation for the thin H2/H3
orchestrators.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np


PROTOCOL_ID = "NIDS-H2H3-CONFIRM-20260713-v2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_RESULTS_ROOT = REPOSITORY_ROOT / "results" / "generated" / "h2_h3"
FORBIDDEN_FEATURE_TOKENS = (
    "label", "attack", "timestamp", "flow_id", "src_ip", "dst_ip",
    "source_ip", "destination_ip",
)


class ProtocolError(RuntimeError):
    """Raised when a frozen-protocol invariant is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(cast(Any, value)))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def hash_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def hash_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = canonical_json_bytes({"dtype": str(value.dtype), "shape": value.shape})
    return sha256_bytes(header + memoryview(value).cast("B"))


def hash_state_dict(state: Mapping[str, Any]) -> str:
    """Hash a PyTorch-style state dict without importing torch."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().contiguous().numpy()
        else:
            tensor = np.asarray(tensor)
        digest.update(name.encode("utf-8"))
        digest.update(canonical_json_bytes({"dtype": str(tensor.dtype), "shape": tensor.shape}))
        digest.update(memoryview(np.ascontiguousarray(tensor)).cast("B"))
    return digest.hexdigest()


def file_fingerprint(path: str | Path, *, content_hash: bool = True) -> dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    item = {
        "path": str(path), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        item["sha256"] = hash_file(path)
    return item


def source_tree_manifest(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return [file_fingerprint(path) for path in sorted(set(files)) if path.exists()]


def environment_manifest() -> dict[str, Any]:
    env_keys = [
        "CONDA_DEFAULT_ENV", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
        "MKL_NUM_THREADS", "PYTHONHASHSEED",
    ]
    packages = {}
    for name in ("numpy", "pandas", "sklearn", "torch", "joblib"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # manifest creation must survive optional packages
            packages[name] = f"unavailable:{type(exc).__name__}"
    return {
        "captured_utc": utc_now(), "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": sys.version,
        "python_executable": sys.executable,
        "argv": list(sys.argv), "cwd": str(Path.cwd().resolve()),
        "environment": {key: os.environ.get(key) for key in env_keys},
        "packages": packages,
    }


def git_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    def capture(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, check=True, capture_output=True,
                text=True, timeout=30,
            )
            return result.stdout.strip()
        except Exception as exc:
            return f"unavailable:{type(exc).__name__}"
    return {
        "commit": capture("rev-parse", "HEAD"),
        "status_porcelain": capture("status", "--porcelain=v1"),
    }


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ProtocolError(f"Immutable artifact already exists: {path}")
    payload = json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_immutable_json(path: str | Path, value: Any) -> str:
    path = Path(path)
    payload_hash = hash_json(value)
    if path.exists():
        current = read_json(path)
        if hash_json(current) != payload_hash:
            raise ProtocolError(f"Refusing to alter frozen artifact: {path}")
        return payload_hash
    atomic_write_json(path, value, overwrite=False)
    return payload_hash


def write_protocol_lock_checklist(
    root: str | Path, *, policy_paths: Mapping[str, str | Path],
    seeds: Sequence[int], targets: Sequence[str], data_paths: Sequence[str | Path],
    code_paths: Sequence[str | Path],
) -> dict[str, Any]:
    policies = {
        name: {"path": str(Path(path).resolve()), "sha256": hash_file(path)}
        for name, path in policy_paths.items()
    }
    checklist = {
        "protocol_id": PROTOCOL_ID, "locked_utc": utc_now(),
        "seeds": list(seeds), "targets": list(targets), "policies": policies,
        "data": [file_fingerprint(path) for path in data_paths],
        "code": source_tree_manifest(code_paths),
        "status": "frozen",
    }
    write_immutable_json(Path(root) / "protocol_lock_checklist.json", checklist)
    return checklist


def validate_protocol_lock_checklist(
    root: str | Path, *, policy_paths: Mapping[str, str | Path],
    seeds: Sequence[int], targets: Sequence[str], data_paths: Sequence[str | Path],
    code_paths: Sequence[str | Path],
) -> dict[str, Any]:
    path = Path(root) / "protocol_lock_checklist.json"
    if not path.exists():
        raise ProtocolError(f"Protocol lock checklist is missing: {path}")
    checklist = read_json(path)
    if checklist.get("protocol_id") != PROTOCOL_ID or checklist.get("status") != "frozen":
        raise ProtocolError(f"Invalid protocol lock checklist identity/status: {path}")
    if checklist.get("seeds") != list(seeds) or checklist.get("targets") != list(targets):
        raise ProtocolError("Protocol lock checklist seeds/targets do not match the runner")
    for name, policy_path in policy_paths.items():
        locked = checklist.get("policies", {}).get(name, {})
        actual_hash = hash_file(policy_path)
        if locked.get("sha256") != actual_hash:
            raise ProtocolError(f"Policy hash differs from protocol lock: {name}")
    def stable(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            [
                {
                    "path": str(Path(item["path"]).resolve()),
                    "size_bytes": int(item["size_bytes"]),
                    "sha256": str(item["sha256"]),
                }
                for item in items
            ],
            key=lambda item: item["path"],
        )
    locked_data = checklist.get("data")
    locked_code = checklist.get("code")
    if not isinstance(locked_data, list) or not isinstance(locked_code, list):
        raise ProtocolError("Protocol lock lacks frozen data/code fingerprints")
    actual_data = [file_fingerprint(data_path) for data_path in data_paths]
    actual_code = source_tree_manifest(code_paths)
    if stable(locked_data) != stable(actual_data):
        raise ProtocolError("Data path/size/SHA256 differs from protocol lock")
    if stable(locked_code) != stable(actual_code):
        raise ProtocolError("Code path/size/SHA256 differs from protocol lock")
    return checklist


def require_fresh_protocol_root(root: str | Path, *, allow_existing_empty: bool = True) -> Path:
    root = Path(root).resolve()
    expected = PROTOCOL_RESULTS_ROOT.resolve()
    if root != expected and expected not in root.parents:
        raise ProtocolError(f"Output must remain under {expected}; got {root}")
    if root.exists():
        entries = list(root.iterdir())
        if entries or not allow_existing_empty:
            raise ProtocolError(f"Fresh output required; directory is not empty: {root}")
    return root


def require_execute_flag(execute: bool, operation: str) -> None:
    if not execute:
        raise ProtocolError(
            f"{operation} is execution-capable and requires explicit --execute. "
            "Use the plan command for a no-run inspection."
        )


def validate_feature_order(features: Sequence[str], expected_count: int = 49) -> None:
    if len(features) != expected_count:
        raise ProtocolError(f"Expected {expected_count} modelling features; got {len(features)}")
    if len(set(features)) != len(features):
        raise ProtocolError("Modelling feature order contains duplicates")
    forbidden = [
        name for name in features
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ProtocolError(f"Forbidden label/identifier/time features: {forbidden}")


def validate_confusion_counts(metrics: Mapping[str, Any], expected_n: int) -> None:
    aliases = [("TP", "tp"), ("TN", "tn"), ("FP", "fp"), ("FN", "fn")]
    counts = []
    for upper, lower in aliases:
        value = metrics.get(upper, metrics.get(lower))
        if value is None or int(value) < 0:
            raise ProtocolError(f"Invalid confusion count {upper}: {value}")
        counts.append(int(value))
    if sum(counts) != int(expected_n):
        raise ProtocolError(f"Confusion counts sum to {sum(counts)}, expected {expected_n}")


def compare_invariants(left: Mapping[str, Any], right: Mapping[str, Any], keys: Sequence[str]) -> None:
    mismatches = {key: (left.get(key), right.get(key)) for key in keys if left.get(key) != right.get(key)}
    if mismatches:
        raise ProtocolError(f"Paired invariant mismatch: {mismatches}")


def deterministic_seed(*parts: Any) -> int:
    return int(hash_json(parts)[:16], 16) % (2**32 - 1)


def paired_bootstrap(
    deltas: Sequence[float], *, repetitions: int = 10_000,
    seed: int = 42, confidence: float = 0.95,
) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ProtocolError("Paired bootstrap requires a non-empty finite 1-D delta vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "n_pairs": len(values), "repetitions": repetitions, "seed": seed,
        "mean": float(values.mean()), "std_ddof1": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
    }


def capture_rng_state(torch_module: Any | None = None) -> dict[str, Any]:
    import random
    state: dict[str, Any] = {
        "python": repr(random.getstate()),
        "numpy": _jsonable(np.random.get_state()),
    }
    if torch_module is not None:
        state["torch_cpu"] = torch_module.get_rng_state()
        if torch_module.cuda.is_available():
            state["torch_cuda"] = torch_module.cuda.get_rng_state_all()
    return state


def completed_marker(unit_dir: str | Path, payload: Mapping[str, Any]) -> None:
    data = {"protocol_id": PROTOCOL_ID, "completed_utc": utc_now(), **dict(payload)}
    write_immutable_json(Path(unit_dir) / "COMPLETED.json", data)


@contextmanager
def audited_run(unit_dir: str | Path, metadata: Mapping[str, Any]):
    """Write RUNNING/FAILED audit state; caller writes COMPLETED after validation."""
    unit_dir = Path(unit_dir)
    unit_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    running = {
        "protocol_id": PROTOCOL_ID, "status": "running",
        "started_utc": utc_now(), **dict(metadata),
    }
    atomic_write_json(unit_dir / "RUNNING.json", running)
    try:
        yield
    except BaseException as exc:
        atomic_write_json(unit_dir / "FAILED.json", {
            **running, "status": "failed", "failed_utc": utc_now(),
            "wall_clock_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        (unit_dir / "RUNNING.json").unlink(missing_ok=True)
