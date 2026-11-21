#!/usr/bin/env python3
"""Step 04: Generate comprehensive JSON and Markdown summary statistics for AbRank v2."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abrank.stats import main

if __name__ == "__main__":
    main()
