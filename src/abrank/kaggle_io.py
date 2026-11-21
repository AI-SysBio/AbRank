"""Kaggle & local dataset resolution helpers for AbRank.

Supports loading AbRank benchmark tables seamlessly from:
  1. An explicit ``--data-dir`` / ``data_dir`` path,
  2. The ``ABRANK_DATA_DIR`` environment variable,
  3. A local checkout's ``data/benchmark/`` directory, or
  4. The public Kaggle dataset (default ``aisysbio/abrank``) via ``kagglehub`` or the ``kaggle`` CLI.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

DEFAULT_KAGGLE_DATASET = "aisysbio/abrank"
SENTINEL_FILES = (
    "assays_kd.tsv",
    "splits/unseen_ag/test.tsv",
)


def _find_benchmark_root(base: Path) -> Path | None:
    """Locate the directory containing ``assays_kd.tsv`` under ``base``."""
    base = Path(base).expanduser().resolve()
    candidates = [
        base,
        base / "benchmark",
        base / "data" / "benchmark",
        base / "abrank" / "benchmark",
        base / "abrank" / "data" / "benchmark",
    ]
    for cand in candidates:
        if all((cand / rel).exists() for rel in SENTINEL_FILES):
            return cand
    return None


def download_from_kaggle(
    dataset_slug: str = DEFAULT_KAGGLE_DATASET,
    cache_dir: Path | None = None,
) -> Path:
    """Download an AbRank dataset from Kaggle via ``kagglehub`` (preferred) or ``kaggle`` CLI.

    Returns the path to the downloaded dataset directory on disk.
    """
    slug = dataset_slug or os.environ.get("ABRANK_KAGGLE_DATASET", DEFAULT_KAGGLE_DATASET)

    # 1. Preferred: kagglehub (no credentials needed for public datasets in recent versions,
    #    handles caching & versioning automatically).
    try:
        import kagglehub  # type: ignore[import-not-found]

        downloaded = Path(kagglehub.dataset_download(slug))
        return downloaded
    except ImportError:
        pass
    except Exception as exc:
        # Fall through to kaggle CLI if installed, otherwise re-raise with helpful guidance.
        kaggle_bin = shutil.which("kaggle")
        if not kaggle_bin:
            raise RuntimeError(
                f"kagglehub failed to download '{slug}': {exc}\n"
                "Verify the Kaggle dataset slug (`--kaggle-dataset`) and credentials, "
                "or pass `--data-dir /path/to/abrank/data/benchmark`."
            ) from exc

    # 2. Fallback: official `kaggle` CLI
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin:
        out_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else Path.home() / ".cache" / "abrank" / "kaggle" / slug.replace("/", "__")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = _find_benchmark_root(out_dir)
        if existing is not None:
            return out_dir
        cmd = [kaggle_bin, "datasets", "download", "-d", slug, "-p", str(out_dir), "--unzip"]
        subprocess.run(cmd, check=True)
        return out_dir

    raise RuntimeError(
        "Neither `kagglehub` nor the `kaggle` CLI is installed.\n"
        "Install one of them to fetch AbRank automatically from Kaggle:\n"
        "    pip install kagglehub\n"
        "Or download the dataset manually / run `python scripts/01_download_raw.py && "
        "python scripts/02_curate_all.py && python scripts/03_build_benchmark.py`, "
        "then pass `--data-dir /path/to/data/benchmark` or set `ABRANK_DATA_DIR`."
    )


def resolve_benchmark_dir(
    data_dir: Path | str | None = None,
    *,
    kaggle_dataset: str | None = None,
    repo_root: Path | None = None,
    auto_download: bool = True,
) -> Path:
    """Resolve the path to AbRank's ``data/benchmark`` directory.

    Search order:
      1. Explicit ``data_dir`` argument (if provided),
      2. ``ABRANK_DATA_DIR`` environment variable (if set),
      3. Local ``<repo_root>/data/benchmark`` (if populated),
      4. Kaggle download of ``kaggle_dataset`` (or ``ABRANK_KAGGLE_DATASET`` / ``aisysbio/abrank``).
    """
    if data_dir is not None:
        found = _find_benchmark_root(Path(data_dir))
        if found is None:
            raise FileNotFoundError(
                f"Could not find AbRank benchmark files (`assays_kd.tsv`, `splits/unseen_ag/test.tsv`) "
                f"under `{data_dir}`."
            )
        return found

    env_dir = os.environ.get("ABRANK_DATA_DIR")
    if env_dir:
        found = _find_benchmark_root(Path(env_dir))
        if found is not None:
            return found

    from abrank.config import get_repo_root

    root = Path(repo_root) if repo_root is not None else get_repo_root()
    local_found = _find_benchmark_root(root / "data" / "benchmark")
    if local_found is not None:
        return local_found

    if not auto_download:
        raise FileNotFoundError(
            "AbRank benchmark tables were not found locally under `data/benchmark/` and "
            "`auto_download=False`. Download from Kaggle (`python scripts/00_download_from_kaggle.py`), "
            "pass `--data-dir`, or run the curation pipeline (`scripts/01`–`03`)."
        )

    slug = kaggle_dataset or os.environ.get("ABRANK_KAGGLE_DATASET", DEFAULT_KAGGLE_DATASET)
    dl_root = download_from_kaggle(slug)
    found = _find_benchmark_root(dl_root)
    if found is None:
        raise FileNotFoundError(
            f"Downloaded Kaggle dataset `{slug}` to `{dl_root}`, but expected files "
            f"{SENTINEL_FILES} were not found inside."
        )
    return found


def materialize_kaggle_into_repo(
    dataset_slug: str = DEFAULT_KAGGLE_DATASET,
    repo_root: Path | None = None,
    *,
    mode: str = "symlink",
    include_structures: bool = True,
) -> dict[str, Path]:
    """Download AbRank from Kaggle and link/copy ``benchmark/`` (and optional WT structures)
    into ``<repo_root>/data/``.
    """
    from abrank.config import get_repo_root

    root = Path(repo_root) if repo_root is not None else get_repo_root()
    dl_root = download_from_kaggle(dataset_slug)
    bench_src = _find_benchmark_root(dl_root)
    if bench_src is None:
        raise FileNotFoundError(
            f"Kaggle dataset `{dataset_slug}` at `{dl_root}` does not contain `assays_kd.tsv`."
        )

    data_root = root / "data"
    bench_dst = data_root / "benchmark"
    bench_dst.mkdir(parents=True, exist_ok=True)

    def _place(src: Path, dst: Path) -> None:
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)
        if mode == "symlink":
            dst.symlink_to(src)
        elif mode == "copy":
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unsupported mode: {mode!r} (expected 'symlink' or 'copy').")

    for child in sorted(bench_src.iterdir()):
        _place(child, bench_dst / child.name)

    materialized = {"benchmark": bench_dst}

    if include_structures:
        struct_candidates = [
            dl_root / "structures_wt",
            dl_root / "structures",
            dl_root / "data" / "structures",
        ]
        for sc in struct_candidates:
            if sc.is_dir():
                st_dst = data_root / "structures"
                st_dst.mkdir(parents=True, exist_ok=True)
                for sub in sorted(sc.iterdir()):
                    _place(sub, st_dst / sub.name)
                materialized["structures"] = st_dst
                break

    return materialized
