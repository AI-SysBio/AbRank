#!/usr/bin/env python3
"""Download the pre-built AbRank v2 benchmark release from Kaggle and link/copy it into `data/`."""

from __future__ import annotations

import argparse
from pathlib import Path

from abrank.kaggle_io import DEFAULT_KAGGLE_DATASET, materialize_kaggle_into_repo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch pre-curated AbRank benchmark tables & WT structures from Kaggle."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_KAGGLE_DATASET,
        help=f"Kaggle dataset slug (default: {DEFAULT_KAGGLE_DATASET})",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Whether to symlink (fast, saves disk) or copy files into `data/`.",
    )
    parser.add_argument(
        "--no-structures",
        action="store_true",
        help="Skip linking/copying wild-type 3D scaffolds even if present in the Kaggle bundle.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional AbRank repository root (defaults to auto-detected repo root).",
    )
    args = parser.parse_args()

    out = materialize_kaggle_into_repo(
        dataset_slug=args.dataset,
        repo_root=args.repo_root,
        mode=args.mode,
        include_structures=not args.no_structures,
    )
    print(f"AbRank Kaggle dataset `{args.dataset}` materialized cleanly:")
    for k, p in out.items():
        print(f"  - {k:12s}: {p}")


if __name__ == "__main__":
    main()
