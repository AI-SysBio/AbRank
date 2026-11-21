#!/usr/bin/env python3
"""Download & extract precomputed 34,040 co-folded 3D poses from Kaggle (`aurlienplissier/abrank-cofolded-3d`).

Skip GPU co-folding entirely and populate the exact directory layout expected by
`scripts/cofolding/03_ingest_and_evaluate_cofolded.py`:

    data/structures/
      ├── cofolded_structures/<method>/<scaffold_id>/pose_00..09.cif
      ├── cofolded_structures/MANIFEST.tsv
      ├── cofolded_confidence/<method>/<scaffold_id>/pose_00..09.{json,npz}
      ├── cofolded_confidence/MANIFEST_confidence.tsv
      └── cofolded_standardized/<method>/<scaffold_id>__p00..09.pdb

Examples:
    # Download Boltz-2 and Chai-1 poses + standardized LOACO PDBs (open-weight models only):
    python scripts/cofolding/00_download_cofolded_from_kaggle.py --methods boltz2 chai1

    # Download all 4 methods (AF3, Boltz-2, Chai-1, Protenix; 34,040 poses, 3.1 GB download):
    python scripts/cofolding/00_download_cofolded_from_kaggle.py --methods all

    # Download only the 3,760 standardized LOACO-Local PDBs (~145 MB archive):
    python scripts/cofolding/00_download_cofolded_from_kaggle.py --methods none --include-standardized
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ALL_METHODS = ("af3", "boltz2", "chai1", "protenix")
DEFAULT_DATASET = "aurlienplissier/abrank-cofolded-3d"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _download_dataset_file(dataset: str, filename: str, use_cli: bool, cache_dir: Path | None) -> Path:
    """Download a single file from the Kaggle dataset (avoids downloading unneeded method archives)."""
    if not use_cli:
        try:
            import kagglehub

            downloaded = kagglehub.dataset_download(dataset, path=filename)
            return Path(downloaded)
        except Exception as exc:
            print(f"[warn] kagglehub single-file download failed ({exc}); falling back to `kaggle` CLI...")

    target_dir = cache_dir or (Path.home() / ".cache" / "abrank_cofolded_3d")
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-f",
        filename,
        "-p",
        str(target_dir),
    ]
    subprocess.run(cmd, check=True)
    candidate = target_dir / filename
    if candidate.exists():
        return candidate
    zipped = target_dir / f"{filename}.zip"
    if zipped.exists():
        shutil.unpack_archive(str(zipped), str(target_dir))
        zipped.unlink(missing_ok=True)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Expected downloaded file not found: {candidate}")


def _safe_extract_tgz(archive_path: Path, dest_dir: Path) -> int:
    """Safely extract a gzip-compressed tar archive (including `.tgz.bin`) into `dest_dir`."""
    dest_resolved = dest_dir.resolve()
    dest_resolved.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in members:
            target = (dest_resolved / member.name).resolve()
            if not str(target).startswith(str(dest_resolved) + "/") and target != dest_resolved:
                raise RuntimeError(f"Unsafe tar member path rejected: {member.name}")
        if sys.version_info >= (3, 12):
            tf.extractall(path=dest_resolved, members=members, filter="data")
        else:
            tf.extractall(path=dest_resolved, members=members)
        count = sum(1 for m in members if m.isfile())
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download & extract precomputed AbRank co-folded 3D poses from Kaggle."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Kaggle dataset slug (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=[*ALL_METHODS, "all", "none"],
        help="Co-folding methods to download & extract (`all`, `none`, or subset of af3 boltz2 chai1 protenix).",
    )
    parser.add_argument(
        "--include-standardized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also download & extract `cofolded_standardized_loaco.tgz.bin` (3,760 LOACO-Local PDBs; default: True).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination `data/structures` directory (default: <repo>/data/structures).",
    )
    parser.add_argument(
        "--use-cli",
        action="store_true",
        help="Use `kaggle` CLI directly instead of `kagglehub`.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional local cache directory when using `--use-cli`.",
    )
    args = parser.parse_args()

    dest_dir = (args.dest or (_repo_root() / "data" / "structures")).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    if "none" in args.methods:
        methods: list[str] = []
    elif "all" in args.methods:
        methods = list(ALL_METHODS)
    else:
        methods = list(dict.fromkeys(args.methods))

    print(f"Target structures directory : {dest_dir}")
    print(f"Kaggle companion dataset    : {args.dataset}")
    print(f"Selected methods            : {methods or ['(none — manifests/standardized only)']}")
    print(f"Include standardized LOACO  : {args.include_standardized}")

    if "af3" in methods:
        print(
            "[note] AlphaFold 3 predicted structures (`af3_10poses.tgz.bin`) are governed by "
            "Google DeepMind's non-commercial academic AlphaFold 3 Output Terms."
        )

    # 1. Fetch manifests
    st_manifest = _download_dataset_file(args.dataset, "MANIFEST_structures.tsv", args.use_cli, args.cache_dir)
    cf_manifest = _download_dataset_file(args.dataset, "MANIFEST_confidence.tsv", args.use_cli, args.cache_dir)
    (dest_dir / "cofolded_structures").mkdir(parents=True, exist_ok=True)
    (dest_dir / "cofolded_confidence").mkdir(parents=True, exist_ok=True)
    shutil.copy2(st_manifest, dest_dir / "cofolded_structures" / "MANIFEST.tsv")
    shutil.copy2(cf_manifest, dest_dir / "cofolded_confidence" / "MANIFEST_confidence.tsv")
    print("  [OK] Copied MANIFEST.tsv and MANIFEST_confidence.tsv")

    # 2. Fetch and extract selected method archives
    total_files = 0
    for method in methods:
        archive_name = f"{method}_10poses.tgz.bin"
        print(f"  [downloading] {archive_name} ...")
        archive_path = _download_dataset_file(args.dataset, archive_name, args.use_cli, args.cache_dir)
        print(f"  [extracting]  {archive_path.name} -> {dest_dir} ...")
        n = _safe_extract_tgz(archive_path, dest_dir)
        total_files += n
        print(f"  [OK] {method}: extracted {n:,} structure & confidence files")

    # 3. Fetch and extract standardized LOACO-Local PDBs
    if args.include_standardized:
        std_archive = "cofolded_standardized_loaco.tgz.bin"
        print(f"  [downloading] {std_archive} ...")
        std_path = _download_dataset_file(args.dataset, std_archive, args.use_cli, args.cache_dir)
        print(f"  [extracting]  {std_path.name} -> {dest_dir} ...")
        n = _safe_extract_tgz(std_path, dest_dir)
        total_files += n
        print(f"  [OK] cofolded_standardized: extracted {n:,} standardized LOACO PDBs")

    print(f"\nDone! Extracted {total_files:,} files into {dest_dir}.")
    if methods:
        method_arg = "all" if len(methods) == len(ALL_METHODS) else methods[0]
        print(
            "Next step:\n"
            f"  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\\n"
            f"    PYTHONPATH=src python3 scripts/cofolding/03_ingest_and_evaluate_cofolded.py "
            f"--method {method_arg} --max-poses 10"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
