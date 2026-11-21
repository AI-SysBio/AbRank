"""2024–2026 paired-chain KD curation modules:
AIntibody (2026 final challenge), Kothiwal (10 non-viral cell-surface SPR panels),
MAGMA-seq Peterson (H1 HA), MAGMA-seq Kirby (SARS-CoV-2 S1), Abbate (anti-GFRAL SPR),
Germinal (de novo paired Fabs), FLAb-Shanker (Omicron BQ.1.1 / XBB.1.5 BLI), and Porebski (HER2 Fabs).
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

from abrank.config import RepoPaths
from abrank.curate.common import (
    finalize_source_table,
    load_antigen_catalog,
    read_fasta_records,
)
from abrank.schema import clean_seq, kd_nm_to_pkd, trim_antigen_tags, trim_fab_to_fv

MAGMASEQ_TOP_CONCENTRATION_NM = 2000.0


def curate_aintibody(paths: RepoPaths) -> pd.DataFrame:
    """Curate AIntibody 2026 final challenge HT-SPR results (`MOESM4_ESM.xlsx`).

    Combines `Dataset 3` (`0 < KD_SPR <= 1e-6 M`, dropping the 25 fake `1000 nM` non-binders
    from Challenge 2/3 CSV and 5 extrapolated `>1 uM` fits) plus the 1 characterized binder
    from `Dataset 2` absent from `Dataset 3`, collapsing replicate `(VH, VL)` in log10 space.
    """
    catalog = load_antigen_catalog(paths.metadata)
    _, rbd_seq = catalog["SARSCoV2_RBD"]
    xl_path = paths.raw / "AIntibody" / "41587_2026_3238_MOESM4_ESM.xlsx"
    xl = pd.ExcelFile(xl_path)

    d3 = xl.parse("Dataset 3", header=1)
    kd_spr = pd.to_numeric(d3["KD_SPR"], errors="coerce")
    d3_valid = d3[kd_spr.notna() & (kd_spr > 0) & (kd_spr <= 1e-6)].copy()
    d3_valid["_kd_nm"] = pd.to_numeric(d3_valid["KD_SPR"], errors="coerce") * 1e9

    rows = []
    seen_fv: set[tuple[str, str]] = set()
    for _, r in d3_valid.iterrows():
        h, l = clean_seq(r["sequence_aa_heavy"]), clean_seq(r["sequence_aa_light"])
        seen_fv.add((h, l))
        pid = str(r.get("PAPER_ID", r.get("ID", "AIntibody"))).strip()
        grp = str(r.get("group", "AIntibody")).strip()
        rows.append(
            {
                "Ab_name": f"AIntibody-{pid}",
                "WT_Abname": f"AIntibody_{grp}",
                "WT_PDB": "NA",
                "Ag_name": "SARSCoV2_RBD",
                "Ag_type": "Viral",
                "raw_value": float(r["_kd_nm"]),
                "binding_score": float(kd_nm_to_pkd(float(r["_kd_nm"]))),
                "Ab_heavy_chain_seq": h,
                "Ab_light_chain_seq": l,
                "Ag_seq": rbd_seq,
                "Ag_epitope_restrictions": "RBD",
            }
        )

    # Check Dataset 2 for any characterized strict `< 1e-6 M` binder omitted from Dataset 3
    d2 = xl.parse("Dataset 2", header=1)
    d2_char = d2[d2["characterized"] == True].copy()  # noqa: E712
    aff2 = pd.to_numeric(d2_char["affinity"], errors="coerce")
    d2_char = d2_char[aff2.notna() & (aff2 > 0) & (aff2 < 1e-6)].copy()
    for idx, r in d2_char.iterrows():
        h, l = clean_seq(r["sequence_aa_heavy"]), clean_seq(r["sequence_aa_light"])
        if (h, l) not in seen_fv:
            kd_nm = float(r["affinity"]) * 1e9
            rows.append(
                {
                    "Ab_name": f"AIntibody-D2-{idx}",
                    "WT_Abname": "AIntibody_Dataset2",
                    "WT_PDB": "NA",
                    "Ag_name": "SARSCoV2_RBD",
                    "Ag_type": "Viral",
                    "raw_value": kd_nm,
                    "binding_score": float(kd_nm_to_pkd(kd_nm)),
                    "Ab_heavy_chain_seq": h,
                    "Ab_light_chain_seq": l,
                    "Ag_seq": rbd_seq,
                    "Ag_epitope_restrictions": "RBD",
                }
            )

    return finalize_source_table(
        pd.DataFrame(rows), dataset_name="AIntibody", modality="KD", raw_unit="nM"
    )


def curate_kothiwal_kd(paths: RepoPaths) -> pd.DataFrame:
    """Curate Kothiwal et al. 10-target non-viral cell-surface SPR panels (with M->nM unit fix)."""
    panels = sorted((paths.raw / "FLAb-Kothiwal").glob("kothiwal2025htp_*_spr.csv"))
    df = pd.concat([pd.read_csv(f) for f in panels], ignore_index=True)

    ag_name = (
        df["Antigen"]
        .astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
        .replace({"DKK 1.00": "DKK1", "hROBO2N": "ROBO2N"})
    )
    kd = pd.to_numeric(df["SPR kinetics - KD (nM)"], errors="coerce")
    valid = kd.notna() & (kd > 0)
    df = df.loc[valid].copy().reset_index(drop=True)
    ag_name = ag_name.loc[valid].reset_index(drop=True)
    kd_arr = kd.loc[valid].to_numpy(dtype=float)

    # 29 rows in PDL2 & ROBO2N were recorded in Molar (<1e-4) despite column header "(nM)"
    kd_nm = np.where(kd_arr < 1e-4, kd_arr * 1e9, kd_arr)

    out = pd.DataFrame(
        {
            "Ab_name": df["Antibody_Name"].astype(str).values,
            "WT_Abname": (ag_name + "_Kothiwal").values,
            "WT_PDB": "NA",
            "Ag_name": ag_name.values,
            "Ag_type": "Non-Viral",
            "raw_value": kd_nm,
            "binding_score": kd_nm_to_pkd(kd_nm),
            "Ab_heavy_chain_seq": df["heavy"].values,
            "Ab_light_chain_seq": df["light"].values,
            "Ag_seq": [trim_antigen_tags(s) for s in df["antigen_seq"]],
            "Ag_epitope_restrictions": "None",
        }
    )
    return finalize_source_table(out, dataset_name="Kothiwal", modality="KD", raw_unit="nM")


def _assign_magmaseq_parents(
    ab_df: pd.DataFrame, heavy_col: str, kd_col: str, mle_tables: list[pd.DataFrame]
) -> pd.Series:
    mle = pd.concat(mle_tables, ignore_index=True)
    mle["ab"] = (
        mle["Variant"].astype(str).str.extract(r"^([^>]+)>")[0].str.replace("_UCA", "", regex=False)
    )
    mle["kd_round"] = pd.to_numeric(mle["Kd"], errors="coerce").round(2)

    per_kd = mle.groupby("kd_round")["ab"].nunique()
    unambiguous = set(per_kd[per_kd == 1].index)
    lookup = (
        mle[mle["kd_round"].isin(unambiguous)].drop_duplicates("kd_round").set_index("kd_round")["ab"]
    )

    hint = pd.to_numeric(ab_df[kd_col], errors="coerce").round(2).map(lookup)
    cluster = ab_df[heavy_col].astype(str).str[:14] + "|" + ab_df[heavy_col].astype(str).str.len().astype(str)

    labels = pd.Series([""] * len(ab_df), index=ab_df.index, dtype=object)
    for clus, idx in cluster.groupby(cluster).groups.items():
        votes = hint.loc[idx].value_counts()
        if votes.empty:
            raise ValueError(f"MAGMA-seq cluster {clus} has no identifiable parent antibody")
        labels.loc[idx] = str(votes.index[0])
    return labels


def curate_peterson(paths: RepoPaths) -> pd.DataFrame:
    """Curate Petersen et al. 2024 MAGMA-seq anti-H1 HA anchor-antibody libraries."""
    df = pd.read_csv(paths.raw / "MAGMAseq-Peterson" / "peterson2024integrated_ab_H1HA_kd.csv")
    ha_records = read_fasta_records(
        paths.metadata / "constructs" / "Influenza_AH1_Brisbane_02_2018_ectodomain.fasta"
    )
    ha_seq = next(iter(ha_records.values()))

    xl = pd.ExcelFile(
        paths.raw / "MAGMAseq-Peterson" / "Petersen2024_SupplementaryData2_MLE_estimates.xlsx"
    )
    mle_tables = [xl.parse(s)[["Variant", "Kd"]] for s in ("all_barcodes_222-1C06", "all_barcodes_319-345")]
    parents = _assign_magmaseq_parents(df, "heavy", "KD", mle_tables)

    kd_nm = pd.to_numeric(df["KD"], errors="coerce")
    keep = kd_nm.notna() & (kd_nm > 0) & (kd_nm < MAGMASEQ_TOP_CONCENTRATION_NM)
    df = df.loc[keep].copy().reset_index(drop=True)
    parents = parents.loc[keep].reset_index(drop=True)
    kd_arr = kd_nm.loc[keep].to_numpy(dtype=float)

    wt_names = "Peterson_" + parents.astype(str)
    out = pd.DataFrame(
        {
            "Ab_name": [f"{wt}-{i}" for i, wt in enumerate(wt_names)],
            "WT_Abname": wt_names.values,
            "WT_PDB": "NA",
            "Ag_name": "Influenza_AH1",
            "Ag_type": "Viral",
            "raw_value": kd_arr,
            "binding_score": kd_nm_to_pkd(kd_arr),
            "Ab_heavy_chain_seq": df["heavy"].values,
            "Ab_light_chain_seq": df["light"].values,
            "Ag_seq": ha_seq,
            "Ag_epitope_restrictions": "HA_anchor",
        }
    )
    return finalize_source_table(out, dataset_name="Peterson", modality="KD", raw_unit="nM")


def curate_kirby(paths: RepoPaths) -> pd.DataFrame:
    """Curate Kirby et al. 2025 MAGMA-seq anti-SARS-CoV-2 S1 antibody development libraries."""
    df = pd.read_csv(paths.raw / "MAGMAseq-Kirby" / "kirby2024retrospective_ab-SARSCoV2_kd.csv")
    s1_records = read_fasta_records(paths.metadata / "constructs" / "SARSCoV2_WuhanHu1_S1.fasta")
    s1_seq = next(iter(s1_records.values()))

    xl = pd.ExcelFile(paths.raw / "MAGMAseq-Kirby" / "pnas.2412787122.sd01.xlsx")
    mle_tables = [xl.parse(s)[["Variant", "Kd"]] for s in ("YL009 MLE filter", "YL011 MLE filter")]
    parents = _assign_magmaseq_parents(df, "heavy", "Kd [M]", mle_tables)

    kd_nm = pd.to_numeric(df["Kd [M]"], errors="coerce")
    keep = kd_nm.notna() & (kd_nm > 0) & (kd_nm < MAGMASEQ_TOP_CONCENTRATION_NM)
    df = df.loc[keep].copy().reset_index(drop=True)
    parents = parents.loc[keep].reset_index(drop=True)
    kd_arr = kd_nm.loc[keep].to_numpy(dtype=float)

    wt_names = "Kirby_" + parents.astype(str).str.replace("Ab_", "", regex=False)
    out = pd.DataFrame(
        {
            "Ab_name": [f"{wt}-{i}" for i, wt in enumerate(wt_names)],
            "WT_Abname": wt_names.values,
            "WT_PDB": "NA",
            "Ag_name": "SARSCoV2_S1",
            "Ag_type": "Viral",
            "raw_value": kd_arr,
            "binding_score": kd_nm_to_pkd(kd_arr),
            "Ab_heavy_chain_seq": df["heavy"].values,
            "Ab_light_chain_seq": df["light"].values,
            "Ag_seq": s1_seq,
            "Ag_epitope_restrictions": "RBD",
        }
    )
    return finalize_source_table(out, dataset_name="Kirby", modality="KD", raw_unit="nM")


def curate_abbate_gfral(paths: RepoPaths) -> pd.DataFrame:
    """Curate Abbate et al. 2026 anti-GFRAL Carterra LSA SPR panel.

    Strictly cross-references `Positive.tsv` (sorted against DTA-free `cynoGFRAL-His`) so only
    verified pure `GFRAL` ECD binders (`FIT QUALITY == 'GOOD FIT'`) enter the core `KD` benchmark,
    eliminating any potential DTA-tag binding confounders.
    """
    gfral_records = read_fasta_records(
        paths.metadata / "constructs" / "hGFRAL_ECD_NP_997293.2_19_351.fasta"
    )
    gfral_ecd_seq = gfral_records["NP_997293.2_19_351"]

    seq_df = pd.read_csv(paths.raw / "Abbate-GFRAL" / "df_GFRAL.tsv", sep="\t")
    spr_df = pd.read_excel(paths.raw / "Abbate-GFRAL" / "GFRAL_tag.xlsx", sheet_name="REPORT MG")
    pos_df = pd.read_csv(paths.raw / "Abbate-GFRAL" / "Positive.tsv", sep="\t")

    dta_free_vh = set(pos_df["aaSeqHeavy"].dropna().map(clean_seq))
    merged = seq_df.merge(spr_df, left_on="Name", right_on="GDB Clone Name", how="inner")
    kd_m = pd.to_numeric(merged["KD (M)"], errors="coerce")
    good = (
        merged["FIT QUALITY"].astype(str).str.strip().str.upper().eq("GOOD FIT")
        & kd_m.notna()
        & (kd_m > 0)
        & merged["aaSeqHeavy"].map(clean_seq).isin(dta_free_vh)
    )
    sub = merged.loc[good].copy().reset_index(drop=True)
    kd_nm = (pd.to_numeric(sub["KD (M)"], errors="coerce") * 1e9).to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "Ab_name": ("Abbate-" + sub["Clone ID"].astype(str)).values,
            "WT_Abname": ("Abbate-" + sub["Clone ID"].astype(str)).values,
            "WT_PDB": "NA",
            "Ag_name": "GFRAL",
            "Ag_type": "Non-Viral",
            "raw_value": kd_nm,
            "binding_score": kd_nm_to_pkd(kd_nm),
            "Ab_heavy_chain_seq": sub["aaSeqHeavy"].values,
            "Ab_light_chain_seq": sub["aaSeqLight"].values,
            "Ag_seq": gfral_ecd_seq,
            "Ag_epitope_restrictions": "GFRAL_ECD_verified_DTA_free",
        }
    )
    return finalize_source_table(out, dataset_name="Abbate", modality="KD", raw_unit="nM")


def _parse_germinal_kd_nm(val: object) -> float:
    """Parse Germinal `Kd (nM)` strings such as `'240 (SPR); 3800 (BLI)'` (preferring SPR)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float, np.floating, np.integer)):
        return float(val) if float(val) > 0 else np.nan
    s = str(val).strip()
    m_spr = re.search(r"(\d+(?:\.\d+)?)\s*\(\s*SPR\s*\)", s, flags=re.IGNORECASE)
    if m_spr:
        return float(m_spr.group(1))
    m_bli = re.search(r"(\d+(?:\.\d+)?)\s*\(\s*BLI\s*\)", s, flags=re.IGNORECASE)
    if m_bli:
        return float(m_bli.group(1))
    try:
        v = float(s)
        return v if v > 0 else np.nan
    except ValueError:
        return np.nan


def curate_germinal_fab(paths: RepoPaths) -> pd.DataFrame:
    """Curate Germinal 2026 de novo paired-chain Fab designs (`PD-L1 scFv` and `IL3 scFv` sheets)."""
    ag_records = read_fasta_records(paths.metadata / "constructs" / "Germinal_antigens.fasta")
    target_map = {
        "PD-L1 scFv": ("PDL1", ag_records["PDL1_WT_ectodomain"]),
        "IL3 scFv": ("IL3", ag_records["IL3_WT_mature"]),
    }
    xl = pd.ExcelFile(paths.raw / "Germinal" / "41587_2026_3187_MOESM5_ESM.xlsx")

    rows = []
    for sheet, (ag_name, ag_seq) in target_map.items():
        df = xl.parse(sheet)
        for _, r in df.iterrows():
            kd_nm = _parse_germinal_kd_nm(r.get("Kd (nM)"))
            if not np.isfinite(kd_nm):
                continue
            prot = str(r["Protein_Sequence"]).strip()
            if ":" not in prot:
                continue
            raw_h, raw_l = prot.split(":", 1)
            vh, vl = trim_fab_to_fv(raw_h, raw_l)
            nm = str(r["Name"]).strip()
            rows.append(
                {
                    "Ab_name": f"Germinal-{nm}",
                    "WT_Abname": f"Germinal_{ag_name}",
                    "WT_PDB": "35TL" if ag_name == "PDL1" else "NA",
                    "Ag_name": ag_name,
                    "Ag_type": "Non-Viral",
                    "raw_value": float(kd_nm),
                    "binding_score": float(kd_nm_to_pkd(kd_nm)),
                    "Ab_heavy_chain_seq": vh,
                    "Ab_light_chain_seq": vl,
                    "Ag_seq": ag_seq,
                    "Ag_epitope_restrictions": "None",
                }
            )
    return finalize_source_table(
        pd.DataFrame(rows), dataset_name="Germinal", modality="KD", raw_unit="nM"
    )


def curate_shanker_kd(paths: RepoPaths) -> pd.DataFrame:
    """Curate Shanker et al. 2024 paired-chain BLI KD variants against Omicron BQ.1.1 & XBB.1.5 RBD."""
    ag_records = read_fasta_records(
        paths.metadata / "constructs" / "Shanker_SARS_CoV_2_antigens.fasta"
    )
    files = [
        (
            "shanker2024unsupervised_Ly1404-BQ.1.1_Kd.csv",
            "LY-CoV1404",
            "7MMO",
            "SARSCoV2_RBD_BQ.1.1",
            ag_records["SARSCoV2_Omicron_BQ.1.1_RBD_319-541"],
        ),
        (
            "shanker2024unsupervised_SA58-BQ.1.1_Kd.csv",
            "SA58",
            "7Y0W",
            "SARSCoV2_RBD_BQ.1.1",
            ag_records["SARSCoV2_Omicron_BQ.1.1_RBD_319-541"],
        ),
        (
            "shanker2024unsupervised_SA58-XBB.1.5_Kd.csv",
            "SA58",
            "7Y0W",
            "SARSCoV2_RBD_XBB.1.5",
            ag_records["SARSCoV2_Omicron_XBB.1.5_RBD_319-541"],
        ),
    ]

    rows = []
    for fname, wt_ab, pdb_id, ag_name, ag_seq in files:
        df = pd.read_csv(paths.raw / "FLAb-Shanker" / fname)
        kd_m = pd.to_numeric(df["Kd avg (M)"], errors="coerce")
        # Filter out 0.0 sentinel and >1 mM unconverged non-binder fits
        valid = kd_m.notna() & (kd_m > 1e-13) & (kd_m < 1e-3)
        sub = df.loc[valid].copy().reset_index(drop=True)
        for _, r in sub.iterrows():
            kd_nm = float(r["Kd avg (M)"]) * 1e9
            des = str(r["Design"]).strip()
            rows.append(
                {
                    "Ab_name": f"Shanker-{wt_ab}-{des}",
                    "WT_Abname": wt_ab,
                    "WT_PDB": pdb_id,
                    "Ag_name": ag_name,
                    "Ag_type": "Viral",
                    "raw_value": kd_nm,
                    "binding_score": float(kd_nm_to_pkd(kd_nm)),
                    "Ab_heavy_chain_seq": r["heavy"],
                    "Ab_light_chain_seq": r["light"],
                    "Ag_seq": ag_seq,
                    "Ag_epitope_restrictions": "RBD",
                }
            )

    return finalize_source_table(
        pd.DataFrame(rows), dataset_name="Shanker", modality="KD", raw_unit="nM"
    )


def curate_porebski_her2(paths: RepoPaths) -> pd.DataFrame:
    """Curate Porebski et al. 2024 Nature Biomed. Eng. paired anti-HER2 (`G98A`/`C6.5`) SPR variants."""
    catalog = load_antigen_catalog(paths.metadata)
    _, her2_seq = catalog["HER2_ECD"]

    df = pd.read_excel(paths.raw / "Porebski" / "Affinity_data_Porebski_curated.xlsx", sheet_name="Affinity")
    kd_m = pd.to_numeric(df["KD (M)"], errors="coerce")
    sub = df[(df["Antigen"] == "HER2") & (df["Nb/Ab"] == "Ab") & kd_m.notna() & (kd_m > 0)].copy()

    # Exact C6.5 / G98A framework (stripping N-terminal Met and C-terminal ribosome-display tether)
    vh_prefix = "QVQLVQSGAEVKKPGESLKISCKGSGYSFTSYWIAWVRQMPGKGLEYMGLIYPGDSDTKYSPSFQGQVTISVDKSVSTAYLQWSSLKPSDSAVYFCAR"
    vh_suffix = "WGQGTLVTVSS"
    vl_wt = "QSVLTQPPSVSAAPGQKVTISCSGSSSNIGNNYVSWYQQLPGTAPKLLIYGHTNRPAGVPDRFSGSKSGTSASLAISGFRSEDEADYYCASWDYTLSGWVFGGGTKLTVL"
    vl_ml39 = vl_wt.replace("LLIYGHTNRPA", "LLIYDHTNRPA")  # G50D on ML3-9 control (HER20002)

    rows = []
    for _, r in sub.iterrows():
        cid = str(r["Clone ID"]).strip()
        hcdr3 = str(r["VHCDR3"]).strip()
        if hcdr3 == "WT":
            hcdr3 = "HDVAYCSSSNCAKWPEYFQH"
        vh = vh_prefix + hcdr3 + vh_suffix
        vl = vl_ml39 if cid == "HER20002" else vl_wt
        kd_nm = float(r["KD (M)"]) * 1e9
        rows.append(
            {
                "Ab_name": f"Porebski-{cid}",
                "WT_Abname": "G98A_C6.5",
                "WT_PDB": "NA",
                "Ag_name": "HER2_ECD",
                "Ag_type": "Non-Viral",
                "raw_value": kd_nm,
                "binding_score": float(kd_nm_to_pkd(kd_nm)),
                "Ab_heavy_chain_seq": vh,
                "Ab_light_chain_seq": vl,
                "Ag_seq": her2_seq,
                "Ag_epitope_restrictions": "None",
            }
        )

    return finalize_source_table(
        pd.DataFrame(rows), dataset_name="Porebski", modality="KD", raw_unit="nM"
    )
