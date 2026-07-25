"""Safe CLI entry points must work from an unrelated working directory."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_safe_entry_points_from_foreign_directory(tmp_path: Path) -> None:
    reference_before = {
        **tree_hashes(ROOT / "results" / "reference"),
        **tree_hashes(ROOT / "figures" / "reference"),
    }
    generated_before = {
        path.relative_to(ROOT).as_posix()
        for base in (ROOT / "results" / "generated", ROOT / "figures" / "generated")
        for path in base.rglob("*")
        if path.is_file()
    }
    commands = [
        [sys.executable, str(ROOT / "verify_results.py")],
        [sys.executable, str(ROOT / "run_h1.py")],
        [sys.executable, str(ROOT / "run_h2.py"), "plan"],
        [sys.executable, str(ROOT / "run_h3.py"), "plan"],
        [sys.executable, str(ROOT / "run_oracle.py"), "dry-run"],
        [sys.executable, str(ROOT / "run_calibration.py"), "--help"],
        [
            sys.executable,
            str(ROOT / "experiments" / "h2" / "compute_feature_shift.py"),
            "--help",
        ],
    ]
    clean_env = os.environ.copy()
    for name in (
        "PYTHONPATH",
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "CONDA_PYTHON_EXE",
    ):
        clean_env.pop(name, None)
    for command in commands:
        subprocess.run(
            command,
            cwd=tmp_path,
            env=clean_env,
            check=True,
            capture_output=True,
            text=True,
        )
    reference_after = {
        **tree_hashes(ROOT / "results" / "reference"),
        **tree_hashes(ROOT / "figures" / "reference"),
    }
    generated_after = {
        path.relative_to(ROOT).as_posix()
        for base in (ROOT / "results" / "generated", ROOT / "figures" / "generated")
        for path in base.rglob("*")
        if path.is_file()
    }
    assert reference_after == reference_before
    assert generated_after == generated_before
