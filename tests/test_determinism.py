"""All Torch training paths share the same supported determinism controls."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from cross_domain_nids.determinism import configure_determinism


def test_configure_determinism_repeats_rng_state_and_torch_flags() -> None:
    configure_determinism(123, torch_module=torch)
    first = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1).item()),
    )
    configure_determinism(123, torch_module=torch)
    second = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1).item()),
    )
    assert first == second
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
