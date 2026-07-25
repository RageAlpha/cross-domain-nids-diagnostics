"""Generated outputs must never be redirected into immutable references."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from cross_domain_nids.release_paths import require_generated_output


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["results", "figures"])
def test_reference_paths_are_rejected(kind: str) -> None:
    reference = ROOT / kind / "reference"
    with pytest.raises(ValueError, match="read-only"):
        require_generated_output(reference, kind)
    with pytest.raises(ValueError, match="read-only"):
        require_generated_output(reference / "nested", kind)


@pytest.mark.parametrize(
    "command",
    [
        [
            "run_h1.py", "--execute", "--seed", "42", "--output-dir",
            "results/reference/h1",
        ],
        [
            "run_h2.py", "plan", "--output-root", "results/reference/h2",
        ],
        [
            "run_h3.py", "plan", "--output-root", "results/reference/h3",
        ],
        [
            "run_oracle.py", "run", "--execute", "--output-dir",
            "results/reference/oracle",
        ],
        [
            "run_calibration.py", "--output-dir",
            "results/reference/calibration",
        ],
    ],
)
def test_result_entry_points_reject_reference_output(command: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / command[0]), *command[1:]],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "reference" in (result.stdout + result.stderr).lower()


def test_figure_defaults_write_only_to_generated() -> None:
    generated = ROOT / "figures" / "generated"
    with tempfile.TemporaryDirectory(prefix="output_guard_", dir=generated) as temp:
        temp_path = Path(temp)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "figures" / "scripts" / "plot_h2_loto_results.py"),
                "--output",
                str(temp_path / "h2_loto_results"),
            ],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "figures" / "scripts" / "plot_h3_paired_domain_loss.py"),
                "--output-dir",
                str(temp_path),
            ],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        assert (temp_path / "h2_loto_results.pdf").is_file()
        assert (temp_path / "h3_paired_domain_loss.pdf").is_file()
