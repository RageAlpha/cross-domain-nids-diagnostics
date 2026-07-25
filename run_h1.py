"""Stable entry point for the frozen H1 corrected-split runner."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if len(sys.argv) > 1 and sys.argv[1] == "finalize":
    SCRIPT = ROOT / "experiments" / "h1" / "finalize_results.py"
    del sys.argv[1]
else:
    SCRIPT = ROOT / "experiments" / "h1" / "run_training.py"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))
sys.argv[0] = str(SCRIPT)
runpy.run_path(str(SCRIPT), run_name="__main__")
