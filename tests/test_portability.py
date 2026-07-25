"""Portability and frozen-input checks not covered by numeric verification."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from cross_domain_nids import config
from cross_domain_nids.protocol_utils import PROTOCOL_RESULTS_ROOT


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_data_paths_use_released_raw_directory() -> None:
    expected = ROOT / "data" / "raw"
    assert config.DATA_DIR == expected
    for path in (
        config.UNSW_PATH,
        config.CICIDS_PATH,
        config.BOTIOT_PATH,
        config.TONIOT_PATH,
    ):
        assert path.parent == expected


def test_protocol_output_is_repository_local() -> None:
    assert PROTOCOL_RESULTS_ROOT == ROOT / "results" / "generated" / "h2_h3"


def test_compact_finalizer_inputs_are_frozen() -> None:
    expected = {
        "h1_historical_results.csv": (
            "d69852b5eee9971360c9949ded76a5aec0b28c5e6308dd3974d496a8a6768c93"
        ),
        "oracle_historical_results.csv": (
            "a413d2a52963ba6ed2ffc4d60ee33d0c9692cd5a5b097ebbb50ad61a8cb15c2f"
        ),
    }
    for name, digest in expected.items():
        assert sha256(ROOT / "data" / "reference_inputs" / name) == digest


def test_h2_plan_from_foreign_working_directory(tmp_path: Path) -> None:
    command = [sys.executable, str(ROOT / "run_h2.py"), "plan"]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert Path(plan["output_root"]) == ROOT / "results" / "generated" / "h2_h3"


def test_h3_plan_from_foreign_working_directory(tmp_path: Path) -> None:
    command = [sys.executable, str(ROOT / "run_h3.py"), "plan"]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert Path(plan["output_root"]) == ROOT / "results" / "generated" / "h2_h3"
