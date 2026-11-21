"""Master curation orchestrator (`abrank-curate` / `scripts/02_curate_all.py`).

Runs all source-specific curations, writes clean per-source tables to `data/processed/`,
deduplicates across sources in log-affinity space (for exact `(VH, VL, Ag)` overlaps across
curated databases such as `SKEMPIv2`, `SabDab`, `HER2`, and `AbBibench`), and writes the four
master benchmark tables into `data/benchmark/`:
  - `assays_kd.tsv`          (paired `VH + VL` equilibrium / kinetic $K_d$)
  - `assays_cross_modal.tsv` (paired `VH + VL` `Enrichment`, `ELISA`, `EC50`, `DMS`)
  - `assays_transfer.tsv`    (paired `VH + VL` neutralization `IC50` + viral `Escape`)
  - `assays_vhh.tsv`         (single-domain `VHH` / nanobody $K_d$)
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from abrank.config import RepoPaths
from abrank.curate.common import ASSAY_COLUMNS
from abrank.curate.cross_modal import (
    curate_abagym_dms,
    curate_abbibench_enrichment,
    curate_abdesign_ddg,
    curate_abdesign_elisa,
    curate_kothiwal_ec50,
    curate_nbbench_galectin3,
)
from abrank.curate.kd_2026 import (
    curate_abbate_gfral,
    curate_aintibody,
    curate_germinal_fab,
    curate_kirby,
    curate_kothiwal_kd,
    curate_peterson,
    curate_porebski_her2,
    curate_shanker_kd,
)
from abrank.curate.kd_core import (
    curate_abbibench_kd,
    curate_abcov_kd,
    curate_her2,
    curate_igdesign,
    curate_origin1,
    curate_ova,
    curate_sabdab,
    curate_skempiv2,
)
from abrank.curate.transfer import (
    curate_abcov_ic50,
    curate_catnap_ic50,
    curate_rbd_escape_bloom,
    curate_shanker_ic50,
)
from abrank.curate.vhh import (
    curate_abcov_vhh,
    curate_adaptyv_egfr_vhh,
    curate_bennett_vhh,
    curate_boltzgen_vhh,
    curate_germinal_vhh,
    curate_porebski_vhh,
    curate_sabdab_vhh,
    curate_skempiv2_vhh,
)


def _dedup_master_kd(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse exact cross-dataset `(VH, VL, Ag)` duplicates in `pKd` space."""
    grp = ["Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"]
    if not df.duplicated(subset=grp).any():
        return df.reset_index(drop=True)
    before = len(df)
    agg = (
        df.groupby(grp, as_index=False, sort=False)
        .agg(
            Ab_name=("Ab_name", "first"),
            WT_Abname=("WT_Abname", "first"),
            WT_PDB=("WT_PDB", lambda s: next((x for x in s if x != "NA"), "NA")),
            Ag_name=("Ag_name", "first"),
            Ag_group=("Ag_group", "first"),
            Ag_type=("Ag_type", "first"),
            Assay_modality=("Assay_modality", "first"),
            raw_unit=("raw_unit", "first"),
            binding_score=("binding_score", "mean"),
            Ag_epitope_restrictions=("Ag_epitope_restrictions", "first"),
            N_replicates=("N_replicates", "sum"),
            Dataset=("Dataset", "first"),
        )
    )
    agg["raw_value"] = 10.0 ** (9.0 - agg["binding_score"])
    print(f"[Master KD] Cross-dataset deduplication: {before:,} -> {len(agg):,} unique (VH, VL, Ag) complexes.")
    return agg[ASSAY_COLUMNS].reset_index(drop=True)


def curate_all_datasets(paths: RepoPaths | None = None) -> dict[str, int]:
    """Execute all curation modules and export `data/processed/` + `data/benchmark/assays_*.tsv`."""
    paths = paths or RepoPaths.from_root()
    paths.processed.mkdir(parents=True, exist_ok=True)
    paths.benchmark.mkdir(parents=True, exist_ok=True)

    # 1. Core paired-chain KD sources (ordered with gold-standard biophysics first for cross-source dedup)
    kd_builders = [
        ("SKEMPIv2", curate_skempiv2),
        ("SabDab", curate_sabdab),
        ("Origin1", curate_origin1),
        ("IgDesign", curate_igdesign),
        ("HER2", curate_her2),
        ("Porebski", curate_porebski_her2),
        ("OVA", curate_ova),
        ("AIntibody", curate_aintibody),
        ("Kothiwal", curate_kothiwal_kd),
        ("Abbate", curate_abbate_gfral),
        ("Germinal", curate_germinal_fab),
        ("Shanker", curate_shanker_kd),
        ("Peterson", curate_peterson),
        ("Kirby", curate_kirby),
        ("AbCoV", curate_abcov_kd),
        ("AbBibench", curate_abbibench_kd),
    ]
    kd_frames = []
    for name, fn in kd_builders:
        print(f"Curating KD source: {name} ...")
        t = fn(paths)
        t.to_csv(paths.processed / f"kd_{name.lower()}.tsv", sep="\t", index=False)
        print(f"  -> {name}: {len(t):,} unique complexes across {t['Ag_group'].nunique()} Ag_group(s)")
        kd_frames.append(t)

    kd_master = _dedup_master_kd(pd.concat(kd_frames, ignore_index=True))
    kd_master.to_csv(paths.benchmark / "assays_kd.tsv", sep="\t", index=False)

    # 2. Cross-modal sources (Enrichment, ELISA, DDG, EC50, DMS; including NbBench Galectin-3 VHHs)
    cm_builders = [
        ("abbibench_enrichment", curate_abbibench_enrichment),
        ("nbbench_enrichment", curate_nbbench_galectin3),
        ("abdesign_elisa", curate_abdesign_elisa),
        ("abdesign_ddg", curate_abdesign_ddg),
        ("kothiwal_ec50", curate_kothiwal_ec50),
        ("abagym_dms", curate_abagym_dms),
    ]
    cm_frames = []
    for name, fn in cm_builders:
        print(f"Curating Cross-Modal source: {name} ...")
        t = fn(paths)
        t.to_csv(paths.processed / f"cm_{name}.tsv", sep="\t", index=False)
        print(f"  -> {name}: {len(t):,} unique complexes across {t['Ag_group'].nunique()} Ag_group(s)")
        cm_frames.append(t)

    cm_master = pd.concat(cm_frames, ignore_index=True)
    cm_master = cm_master.drop_duplicates(
        subset=["Assay_modality", "Dataset", "Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"]
    ).reset_index(drop=True)
    cm_master.to_csv(paths.benchmark / "assays_cross_modal.tsv", sep="\t", index=False)

    # 3. Transfer-learning paired-chain sources (Neutralization IC50 + RBD Escape)
    tr_builders = [
        ("catnap_ic50", curate_catnap_ic50),
        ("abcov_ic50", curate_abcov_ic50),
        ("shanker_ic50", curate_shanker_ic50),
        ("rbd_escape_bloom", curate_rbd_escape_bloom),
    ]
    tr_frames = []
    for name, fn in tr_builders:
        print(f"Curating Transfer source: {name} ...")
        t = fn(paths)
        t.to_csv(paths.processed / f"tr_{name}.tsv", sep="\t", index=False)
        print(f"  -> {name}: {len(t):,} unique complexes")
        tr_frames.append(t)

    tr_master = pd.concat(tr_frames, ignore_index=True)
    tr_master = tr_master.drop_duplicates(
        subset=["Assay_modality", "Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"]
    ).reset_index(drop=True)
    tr_master.to_csv(paths.benchmark / "assays_transfer.tsv", sep="\t", index=False)

    # 4. Dedicated Single-Domain VHH / Nanobody KD sources (biophysical structural sources first for dedup)
    vhh_builders = [
        ("skempiv2_vhh", curate_skempiv2_vhh),
        ("sabdab_vhh", curate_sabdab_vhh),
        ("abcov_vhh", curate_abcov_vhh),
        ("porebski_vhh", curate_porebski_vhh),
        ("germinal_vhh", curate_germinal_vhh),
        ("boltzgen_vhh", curate_boltzgen_vhh),
        ("adaptyv_egfr_vhh", curate_adaptyv_egfr_vhh),
        ("bennett_vhh", curate_bennett_vhh),
    ]
    vhh_frames = []
    for name, fn in vhh_builders:
        print(f"Curating VHH source: {name} ...")
        t = fn(paths)
        t.to_csv(paths.processed / f"vhh_{name}.tsv", sep="\t", index=False)
        print(f"  -> {name}: {len(t):,} unique VHH complexes across {t['Ag_group'].nunique()} Ag_group(s)")
        vhh_frames.append(t)

    vhh_master = _dedup_master_kd(pd.concat(vhh_frames, ignore_index=True))
    vhh_master.to_csv(paths.benchmark / "assays_vhh.tsv", sep="\t", index=False)

    counts = {
        "assays_kd": len(kd_master),
        "assays_cross_modal": len(cm_master),
        "assays_transfer": len(tr_master),
        "assays_vhh": len(vhh_master),
    }
    print("\nMaster Curation Summary:", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate all AbRank v2 datasets from data/raw/.")
    parser.parse_args()
    curate_all_datasets()


if __name__ == "__main__":
    main()
