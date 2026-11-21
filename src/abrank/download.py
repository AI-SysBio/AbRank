"""Reproducible raw dataset download and integrity verification against `sources_manifest.yaml`."""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
import yaml

from abrank.config import RepoPaths


def _download_url(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (AbRank-Benchmark-Downloader/2.0)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def _build_sabdab_vhh_rcsb_seqs(aff_summary_path: Path, out_tsv: Path) -> None:
    """Fetch RCSB FASTA sequences for all single-domain VHH complexes in `SabDab_aff_summary.tsv`."""
    ssum = pd.read_csv(aff_summary_path, sep="\t")
    vhh_sab = ssum[ssum["Lchain"].isna() & ssum["Hchain"].notna() & ssum["antigen_chain"].notna()].copy()
    pdbs = sorted(vhh_sab["pdb"].astype(str).str.upper().unique())

    def _fetch(pdb_id: str) -> tuple[str, str]:
        req = urllib.request.Request(
            f"https://www.rcsb.org/fasta/entry/{pdb_id}",
            headers={"User-Agent": "Mozilla/5.0 (AbRank-Benchmark-Downloader/2.0)"},
        )
        return pdb_id, urllib.request.urlopen(req, timeout=30).read().decode("utf-8")

    fasta_map: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        for pdb_id, txt in ex.map(_fetch, pdbs):
            fasta_map[pdb_id] = txt

    def _parse_chains(fasta_text: str) -> dict[str, str]:
        c2s: dict[str, str] = {}
        hdr, buf = None, []

        def _commit(h: str | None, s: str) -> None:
            if not h:
                return
            parts = h.split("|")
            if len(parts) < 2:
                return
            ch_part = re.sub(r"^Chains?\s+", "", parts[1])
            for tok in ch_part.split(","):
                tok = tok.strip()
                m_auth = re.match(r"^([A-Za-z0-9]+)\s*\[auth\s+([A-Za-z0-9]+)\]$", tok)
                if m_auth:
                    c2s[m_auth.group(2)] = s
                    c2s.setdefault(m_auth.group(1), s)
                elif tok:
                    c2s[tok] = s

        for ln in fasta_text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(">"):
                _commit(hdr, "".join(buf))
                hdr, buf = ln[1:], []
            else:
                buf.append(ln)
        _commit(hdr, "".join(buf))
        return c2s

    rows = []
    for _, r in vhh_sab.iterrows():
        pdb_id = str(r["pdb"]).strip().upper()
        hch = str(r["Hchain"]).strip()
        ag_chs = [c.strip() for c in str(r["antigen_chain"]).split("|") if c.strip() and c.strip() != "NA"]
        cmap = _parse_chains(fasta_map.get(pdb_id, ""))
        hseq = cmap.get(hch) or cmap.get(hch.upper())
        ag_seqs = [s for s in (cmap.get(ac) or cmap.get(ac.upper()) for ac in ag_chs) if s]
        if hseq and ag_seqs:
            rows.append(
                {
                    "pdb": pdb_id,
                    "Hchain": hch,
                    "antigen_chain": str(r["antigen_chain"]).strip(),
                    "antigen_name": str(r.get("antigen_name", "")),
                    "antigen_species": str(r.get("antigen_species", "")),
                    "affinity": r.get("affinity"),
                    "affinity_method": str(r.get("affinity_method", "")),
                    "vhh_seq": hseq,
                    "ag_seq": "".join(dict.fromkeys(ag_seqs)),
                }
            )
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)


def _download_boltzgen_zip(zip_url: str, local_dir: Path) -> None:
    """Download BoltzGen `adaptyv_data.zip` from HuggingFace and extract `*_summary.csv`."""
    summaries_dir = local_dir / "adaptyv_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "adaptyv_data.zip"
        print(f"  [BoltzGen] Downloading {zip_url} ...")
        _download_url(zip_url, zpath, timeout=300)
        with zipfile.ZipFile(zpath, "r") as zf:
            for member in zf.namelist():
                if member.endswith("_summary.csv"):
                    out_file = summaries_dir / Path(member).name
                    with zf.open(member) as src, open(out_file, "wb") as dst:
                        shutil.copyfileobj(src, dst)


def verify_or_download_raw(
    paths: RepoPaths | None = None,
    *,
    download_missing: bool = True,
    sources_filter: list[str] | None = None,
) -> dict[str, list[str]]:
    """Check all required raw files in `metadata/sources_manifest.yaml` and optionally download."""
    paths = paths or RepoPaths.from_root()
    manifest_path = paths.metadata / "sources_manifest.yaml"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    gdrive_url = manifest.get("gdrive_archive_url", "")
    sources: dict = manifest.get("sources", {})

    present: list[str] = []
    downloaded: list[str] = []
    missing: list[str] = []

    for src_name, spec in sources.items():
        if sources_filter and src_name not in sources_filter:
            continue
        local_dir = paths.root / spec["local_dir"]
        req_files = spec.get("required_files", [])

        # Special archive extractor for BoltzGen zip if any summary CSV is missing
        if (
            download_missing
            and "archive_zip_url" in spec
            and any(not (local_dir / rf["path"]).exists() for rf in req_files)
        ):
            try:
                _download_boltzgen_zip(spec["archive_zip_url"], local_dir)
            except Exception as exc:
                print(f"  [{src_name}] Archive download failed: {exc}")

        for rf in req_files:
            rel = rf["path"]
            fpath = local_dir / rel
            label = f"{src_name}/{rel}"
            if fpath.exists() and fpath.stat().st_size > 0:
                present.append(label)
                continue

            url = rf.get("url")
            if download_missing and rel == "SabDab_vhh_rcsb_seqs.tsv" and (local_dir / "SabDab_aff_summary.tsv").exists():
                print(f"  Building {label} from RCSB FASTA API ...")
                try:
                    _build_sabdab_vhh_rcsb_seqs(local_dir / "SabDab_aff_summary.tsv", fpath)
                    downloaded.append(label)
                    continue
                except Exception as exc:
                    print(f"  [WARN] Failed to build {label}: {exc}")

            if download_missing and url:
                print(f"  Downloading {label} from {url} ...")
                try:
                    _download_url(url, fpath)
                    downloaded.append(label)
                    continue
                except Exception as exc:
                    print(f"  [WARN] Failed to download {label}: {exc}")

            missing.append(label)

    print(
        f"Raw data check complete: {len(present)} present, "
        f"{len(downloaded)} downloaded, {len(missing)} missing."
    )
    if missing:
        print(
            "NOTE: The following files were not found locally and require fetching from "
            f"the AbRank Google Drive archive ({gdrive_url}) or upstream portal:"
        )
        for m in missing:
            print(f"  - {m}")

    return {"present": present, "downloaded": downloaded, "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and/or verify raw datasets for the AbRank v2 benchmark."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing local raw files without making network requests.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional subset of source names from sources_manifest.yaml.",
    )
    args = parser.parse_args()

    res = verify_or_download_raw(
        download_missing=not args.verify_only,
        sources_filter=args.sources,
    )
    if res["missing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
