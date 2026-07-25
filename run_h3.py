"""Stable entry point for the frozen H3 paired domain-loss protocol."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
SCRIPT = ROOT / "experiments" / "h3" / "run_paired_domain_loss.py"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))
if "--policy" not in sys.argv:
    sys.argv.extend(["--policy", str(ROOT / "protocols" / "h3" / "verdict_policy.json")])
if "--h2-policy" not in sys.argv:
    sys.argv.extend(["--h2-policy", str(ROOT / "protocols" / "h2" / "selection_policy.json")])
sys.argv[0] = str(SCRIPT)
runpy.run_path(str(SCRIPT), run_name="__main__")
