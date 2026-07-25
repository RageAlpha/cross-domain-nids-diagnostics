"""Stable entry point for the frozen H2 LOTO protocol."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
SCRIPT = ROOT / "experiments" / "h2" / "run_loto_filtering.py"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))
if "--policy" not in sys.argv:
    sys.argv.extend(["--policy", str(ROOT / "protocols" / "h2" / "selection_policy.json")])
if "--h3-policy" not in sys.argv:
    sys.argv.extend(["--h3-policy", str(ROOT / "protocols" / "h3" / "verdict_policy.json")])
sys.argv[0] = str(SCRIPT)
runpy.run_path(str(SCRIPT), run_name="__main__")
