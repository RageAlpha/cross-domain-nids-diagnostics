"""H1 finalizer exposes a safe CLI and only references released inputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from experiments.h1.finalize_results import (
    REPOSITORY_ROOT,
    finalizer_release_sources,
)


ROOT = Path(__file__).resolve().parents[1]


def test_h1_finalizer_help_is_side_effect_free(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "run_h1.py"), "finalize", "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output-dir" in result.stdout
    assert "--historical-results" in result.stdout
    assert "--execute" in result.stdout


def test_h1_finalizer_release_sources_exist_and_are_repository_relative() -> None:
    for path in finalizer_release_sources():
        assert path.is_file()
        assert path.relative_to(REPOSITORY_ROOT).as_posix()
