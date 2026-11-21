#!/usr/bin/env python3
"""Step 02: Curate all raw datasets into standardized per-source and master TSVs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abrank.curate.cli import main

if __name__ == "__main__":
    main()
