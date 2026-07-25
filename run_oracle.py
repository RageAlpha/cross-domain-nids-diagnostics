"""Stable entry point for the corrected target-oracle ceiling."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if len(sys.argv) > 1 and sys.argv[1] == "finalize":
    SCRIPT = ROOT / "experiments" / "oracle" / "finalize_results.py"
    del sys.argv[1]
else:
    SCRIPT = ROOT / "experiments" / "oracle" / "run_target_oracle.py"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))
if SCRIPT.name == "run_target_oracle.py" and "--project-root" not in sys.argv:
    sys.argv.extend(["--project-root", str(ROOT)])
if SCRIPT.name == "run_target_oracle.py" and "--data-dir" not in sys.argv:
    sys.argv.extend(["--data-dir", str(ROOT / "data" / "raw")])
if SCRIPT.name == "run_target_oracle.py" and "--split-root" not in sys.argv:
    sys.argv.extend(["--split-root", str(ROOT / "data" / "splits" / "oracle")])
if "--output-dir" not in sys.argv:
    sys.argv.extend(["--output-dir", str(ROOT / "results" / "generated" / "oracle")])
sys.argv[0] = str(SCRIPT)
runpy.run_path(str(SCRIPT), run_name="__main__")
