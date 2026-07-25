"""Untracked local data and generated outputs must not invalidate a release."""

from __future__ import annotations

from pathlib import Path

from verify_results import check_release_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_release_manifest_ignores_local_outputs() -> None:
    local_files = [
        ROOT / "data" / "raw" / "local_test.csv",
        ROOT / "results" / "generated" / "local_test.csv",
        ROOT / "figures" / "generated" / "local_test.png",
        ROOT / "artifacts" / "local_test.joblib",
        ROOT / "checkpoints" / "local_test.pt",
        ROOT / "predictions" / "local_test.csv",
        ROOT / "external" / "local_test.json",
        ROOT / ".pytest_cache" / "local_manifest_test.txt",
        ROOT / "__pycache__" / "local_manifest_test.pyc",
    ]
    try:
        for path in local_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"local untracked material\n")
        check_release_manifest()
    finally:
        for path in local_files:
            path.unlink(missing_ok=True)
        for directory in sorted(
            {path.parent for path in local_files},
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
