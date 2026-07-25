"""Stable entry point for the E1/E2 calibration analysis."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
SCRIPT = ROOT / "experiments" / "calibration" / "analyze_calibration.py"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))
sys.argv[0] = str(SCRIPT)
runpy.run_path(str(SCRIPT), run_name="__main__")
