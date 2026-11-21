#!/usr/bin/env python3
"""Step 03: Assign canonical IDs, export FASTAs, and generate all zero-leakage splits."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abrank.splits import main

if __name__ == "__main__":
    main()
