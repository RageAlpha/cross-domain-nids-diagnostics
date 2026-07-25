"""Release inventory and text must not expose workstation-specific material."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def released_paths() -> list[str]:
    with (ROOT / "release_manifest_sha256.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        return [row["path"] for row in csv.DictReader(handle)]


def test_no_forbidden_release_paths() -> None:
    paths = released_paths()
    forbidden_suffixes = {
        ".joblib", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".onnx"
    }
    for relative in paths:
        path = Path(relative)
        lowered = relative.lower()
        assert path.suffix.lower() not in forbidden_suffixes
        assert "__pycache__" not in path.parts
        assert ".pytest_cache" not in path.parts
        assert ("gap" + "3") not in lowered
        assert not re.search(r"(^|[._-])" + "e" + r"5([._-]|$)", lowered)
        assert not re.search(r"20\d{2}[-_]?\d{2}", path.name)


def test_no_absolute_workstation_paths_or_old_stage_names() -> None:
    forbidden = (
        re.compile(r"C:\\Users\\", re.IGNORECASE),
        re.compile(r"E:\\Paper\\", re.IGNORECASE),
        re.compile(r"\b" + "gap" + r"3\b", re.IGNORECASE),
        re.compile(r"\b" + "e" + r"5\b", re.IGNORECASE),
    )
    text_suffixes = {
        ".cff", ".csv", ".json", ".md", ".py", ".txt", ".yml", ".yaml"
    }
    for relative in released_paths():
        path = ROOT / relative
        if path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="strict")
        for pattern in forbidden:
            assert pattern.search(text) is None, f"{pattern.pattern} in {relative}"


def test_case_sensitive_public_paths_exist() -> None:
    expected = (
        "data/raw/NF-UNSW-NB15-v3.csv",
        "data/raw/NF-CICIDS2018-v3.csv",
        "data/raw/NF-BoT-IoT-v3.csv",
        "data/raw/NF-ToN-IoT-v3.csv",
        "figures/data/h2_loto_figure_data.csv",
        "figures/data/h3_domain_loss_figure_data.csv",
        "protocols/h2/selection_locks/cicids2018__random_forest.json",
        "protocols/h2/selection_locks/bot_iot__mlp.json",
        "protocols/h2/selection_locks/ton_iot__random_forest.json",
    )
    released = set(released_paths())
    for relative in expected:
        if relative.startswith("data/raw/"):
            assert relative not in released
        else:
            assert relative in released
