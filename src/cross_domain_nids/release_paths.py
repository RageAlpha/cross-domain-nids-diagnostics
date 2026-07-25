"""Repository output guards for immutable manuscript reference artifacts."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def require_generated_output(
    path: str | Path,
    artifact_kind: str,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> Path:
    """Return a resolved output path only when it is below a generated tree."""
    if artifact_kind not in {"results", "figures"}:
        raise ValueError(f"Unsupported artifact kind: {artifact_kind}")
    root = Path(repository_root).resolve()
    resolved = Path(path).resolve()
    generated = (root / artifact_kind / "generated").resolve()
    reference = (root / artifact_kind / "reference").resolve()
    if resolved == reference or reference in resolved.parents:
        raise ValueError(f"Reference artifacts are read-only: {resolved}")
    if resolved != generated and generated not in resolved.parents:
        raise ValueError(
            f"Output must remain under {generated}; got {resolved}"
        )
    return resolved
