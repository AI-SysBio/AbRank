#!/usr/bin/env python3
"""Stage a clean Kaggle dataset upload directory for AbRank v2.

Packages:
  1. `benchmark/`     — Full curated `data/benchmark/` release (4 `assays_*.tsv`, all 5 zero-leakage
                        3-state split suites, deduplicated FASTAs, and summary JSON/MD; ~802 MB)
  2. `metadata/`      — Complete `metadata/` directory (Croissant 1.0, `sources_manifest.yaml`,
                        `cofolding_manifest.tsv`, antigen catalog, PDB mappings, constructs; ~0.6 MB)
  3. `structures_wt/` — Optional wild-type 3D scaffolds (`rcsb/`, `igfold/`, `scaffolds/`, `abagym_pdbs/`;
                        ~400 MB) for Track 1B WT-3D baselines (excluding multi-GB derived `.pt` caches
                        and 11 GB co-folded ensembles).
  4. `dataset-metadata.json` — Ready for `kaggle datasets create -p <out-dir> --dir-mode zip`.

Usage:
  python scripts/00_prepare_kaggle_bundle.py \\
      --source-data /path/to/Abrank/data \\
      --out-dir /tmp/abrank-kaggle-bundle \\
      --slug aurlienplissier/abrank \\
      --include-wt-structures
  kaggle datasets version -p /tmp/abrank-kaggle-bundle --dir-mode zip -t -m "AbRank v2 release"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


WT_STRUCTURE_SUBDIRS = ("rcsb", "igfold", "scaffolds", "abagym_pdbs")


def _dir_size_bytes(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _human_size(n_bytes: int) -> str:
    val = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024.0:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} TB"


def build_dataset_metadata(slug: str, title: str) -> dict:
    return {
        "title": title,
        "id": slug,
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
        "subtitle": "Sequence- and Lineage-Disjoint 3-State Antibody-Antigen Affinity Ranking Benchmark (v2)",
        "description": (
            "AbRank v2 is a standardized, sequence- and lineage-disjoint (<50% antigen identity, "
            "<80% antibody clonal-lineage identity) benchmark for antibody-antigen and single-domain "
            "nanobody (VHH) affinity ranking across 4 Core Community Suites: AbRank-Maturation "
            "(22-cluster LOACO local L1-L3 edits), AbRank-Discovery (25 unfiltered continuous panels + VHH), "
            "AbRank-Generative (1,354 wet-lab generative AI designs), and AbRank-Transfer (Viral IC50, "
            "RBD Escape, ELISA, DDG, and AbAgym DMS). See https://github.com/AI-SysBio/AbRank for full "
            "code, evaluation CLI (`python -m abrank.evaluate`), and curation scripts, and "
            "https://github.com/AI-SysBio/logica/tree/main/antibodies-abrank for the AbLang2 + ESM-2 LogiCA baseline."
        ),
        "keywords": [
            "biology",
            "chemistry",
            "computer science",
            "deep learning",
            "healthcare",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Kaggle upload bundle for AbRank v2.")
    parser.add_argument(
        "--source-data",
        type=Path,
        default=Path("data"),
        help="Path to the populated `data/` directory containing `benchmark/` (and optionally `structures/`).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("metadata"),
        help="Path to AbRank `metadata/` directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination staging folder for `kaggle datasets create -p <out-dir> --dir-mode zip`.",
    )
    parser.add_argument(
        "--slug",
        type=str,
        default="aurlienplissier/abrank",
        help="Kaggle dataset slug `<owner>/<dataset-name>` (default: aurlienplissier/abrank).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="AbRank: Antibody Affinity Ranking",
        help="Kaggle dataset title.",
    )
    parser.add_argument(
        "--include-wt-structures",
        action="store_true",
        help="Also bundle wild-type 3D scaffolds (`rcsb/`, `igfold/`, `scaffolds/`, `abagym_pdbs/`, ~400 MB).",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="Copy files (default, safe across filesystems) or hardlink.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned bundle contents and sizes without writing files.",
    )
    args = parser.parse_args()

    src_data = args.source_data.expanduser().resolve()
    bench_src = src_data / "benchmark" if (src_data / "benchmark").is_dir() else src_data
    if not (bench_src / "assays_kd.tsv").exists():
        raise FileNotFoundError(f"Expected `{bench_src / 'assays_kd.tsv'}` not found.")

    plan: list[tuple[Path, Path]] = [(bench_src, args.out_dir / "benchmark")]
    meta_src = args.metadata_dir.expanduser().resolve()
    if meta_src.is_dir():
        plan.append((meta_src, args.out_dir / "metadata"))

    if args.include_wt_structures:
        st_root = src_data / "structures"
        for sub in WT_STRUCTURE_SUBDIRS:
            sub_p = st_root / sub
            if sub_p.is_dir():
                plan.append((sub_p, args.out_dir / "structures_wt" / sub))

    total_bytes = 0
    print(f"=== AbRank Kaggle Bundle Plan ({args.slug}) ===")
    for s, d in plan:
        sz = _dir_size_bytes(s)
        total_bytes += sz
        print(f"  + {str(d.relative_to(args.out_dir)):28s}  ({_human_size(sz):>9s})  <- {s}")
    print(f"Total uncompressed bundle size: {_human_size(total_bytes)}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for s, d in plan:
        if d.exists():
            shutil.rmtree(d)
        d.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "hardlink":
            shutil.copytree(s, d, copy_function=os.link)
        else:
            shutil.copytree(s, d)

    meta_json = build_dataset_metadata(args.slug, args.title)
    (args.out_dir / "dataset-metadata.json").write_text(
        json.dumps(meta_json, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nBundle written to: {args.out_dir}")
    print("Next step — upload a new version to Kaggle (preserves previous versions, keeps .tsv intact):")
    print(f"  kaggle datasets version -p {args.out_dir} --dir-mode zip -t -m 'AbRank v2 release'")


if __name__ == "__main__":
    import os

    main()
