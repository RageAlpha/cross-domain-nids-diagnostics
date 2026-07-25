"""Shared random-seed and deterministic-execution configuration."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def configure_determinism(
    seed: int,
    *,
    torch_module: Any | None = None,
) -> None:
    """Configure supported Python, NumPy, Torch, cuDNN, and cuBLAS controls."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.deterministic = True
        torch_module.backends.cudnn.benchmark = False
