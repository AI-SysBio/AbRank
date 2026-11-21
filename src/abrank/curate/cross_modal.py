"""Cross-modal assay curation (`Enrichment`, `ELISA`, `DDG`, `EC50`, and `DMS`), strictly
unified so `binding_score` ALWAYS has HIGHER = STRONGER BINDING.
"""

from __future__ import annotations

import zipfile
import numpy as np
import pandas as pd

from abrank.config import RepoPaths
from abrank.curate.common import (
    finalize_source_table,
    load_antigen_catalog,
    load_pdb_mapping,
)
from abrank.schema import RT_LN10, kd_nm_to_pkd, trim_antigen_tags, trim_fab_to_fv

# 7 AbBiBench enrichment panels: (WT_PDB, Ag_name, Ag_type, WT_Abname, is_depletion_sign)
# CRITICAL: `1mlc` and `1mlc_LC` (Warszawski 2019 D44.1-HEWL) report ln(unsorted/sorted) depletion
# (+8.8 = knockout, -2.9 = affinity-enhancing) and MUST be negated & converted from ln to log10!
# All other 5 panels report standard log-enrichment (higher = stronger binding).
ABBIBENCH_ENRICHMENT_PANELS: dict[str, tuple[str, str, str, str, bool]] = {
    "1mlc": ("1MLC", "HEWL", "Non-Viral", "D44.1", True),
    "1mlc_LC": ("1MLC", "HEWL", "Non-Viral", "D44.1", True),
    "2fjg": ("2FJG", "VEGF", "Non-Viral", "G6.31", False),
    "2fjg_LC": ("2FJG", "VEGF", "Non-Viral", "G6.31", False),
    "4d5_her2": ("1N8Z", "HER2_ECD", "Non-Viral", "Trastuzumab", False),
    "5a12_ang2": ("4ZFG", "ANG2", "Non-Viral", "5A12", False),
    "5a12_vegf": ("4ZFF", "VEGF", "Non-Viral", "5A12", False),
}

# 23 AbAgym datasets excluded due to exact upstream duplication with RBD_Escape_Bloom (19),
# internal duplicate COV2-2196 vs AZD8895 (1), and AbBiBench Enrichment (3)
ABAGYM_EXCLUDED_DATASETS: frozenset[str] = frozenset({
    "COVID-19_2021a_AZD1061",
    "COVID-19_2021c_C105",
    "COVID-19_2021c_C110",
    "COVID-19_2021c_C144",
    "COVID-19_2021c_LY-CoV016",
    "COVID-19_2021c_LY-CoV555",
    "COVID-19_2021c_REGN10987",
    "COVID-19_2021d_S2D106",
    "COVID-19_2021d_S2H13",
    "COVID-19_2021d_S2H14",
    "COVID-19_2021d_S2H97",
    "COVID-19_2021d_S2X259",
    "COVID-19_2021d_S2X35",
    "COVID-19_2021d_S309",
    "COVID-19_2022_C119",
    "COVID-19_2022_C121",
    "COVID-19_2022_COV2-2130",
    "COVID-19_2022_LY-CoV1404",
    "COVID-19_2022_WIBP-2B11",
    "COVID-19_2022_COV2-2196",  # Identical mAb/experiment to COVID-19_2021a_AZD8895
    "lysozyme_2019_D441",       # Duplicates AbBiBench 1mlc / 1mlc_LC
    "VEGF_2017b_G6",            # Duplicates AbBiBench 2fjg / 2fjg_LC
    "Ang2_2017_G6",             # Duplicates AbBiBench 5a12_ang2 / G6
})

_AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def curate_abbibench_enrichment(paths: RepoPaths) -> pd.DataFrame:
    """Curate the 7 AbBiBench phage/yeast-display enrichment panels.

    Applies `trim_fab_to_fv` to strip constant `CH1`/`CL` domains (rescuing all 29,981
    `5a12_vegf` variants whose `XXXXXXXX` mask sits inside `CH1`), and fixes the inverted
    depletion sign on `1mlc` and `1mlc_LC`.
    """
    catalog = load_antigen_catalog(paths.metadata)
    raw_dir = paths.raw / "AbBibench" / "Raw"
    frames = []

    for cid, (pdb_id, ag_name, ag_type, wt_ab, negate_ln) in ABBIBENCH_ENRICHMENT_PANELS.items():
        df = pd.read_csv(raw_dir / f"{cid}_benchmarking_data.csv")
        bs = pd.to_numeric(df["binding_score"], errors="coerce")
        df = df[bs.notna() & np.isfinite(bs)].copy().reset_index(drop=True)
        raw_bs = pd.to_numeric(df["binding_score"], errors="coerce").to_numpy(dtype=float)

        if negate_ln:
            score = -raw_bs / np.log(10.0)
        else:
            score = raw_bs

        _, ag_seq = catalog[ag_name]
        trimmed = [trim_fab_to_fv(h, l) for h, l in zip(df["heavy_chain_seq"], df["light_chain_seq"])]
        hs = [t[0] for t in trimmed]
        ls = [t[1] for t in trimmed]

        frames.append(
            pd.DataFrame(
                {
                    "Ab_name": [f"{wt_ab}_{cid}_{i}" for i in range(len(df))],
                    "WT_Abname": wt_ab,
                    "WT_PDB": pdb_id,
                    "Ag_name": ag_name,
                    "Ag_type": ag_type,
                    "raw_value": raw_bs,
                    "binding_score": score,
                    "Ab_heavy_chain_seq": hs,
                    "Ab_light_chain_seq": ls,
                    "Ag_seq": ag_seq,
                    "Ag_epitope_restrictions": "None",
                }
            )
        )

    merged = pd.concat(frames, ignore_index=True)
    return finalize_source_table(
        merged,
        dataset_name="AbBibench",
        modality="Enrichment",
        raw_unit="log_enrichment",
    )


def _curate_abdesign_by_type(
    paths: RepoPaths,
    *,
    target_affinity_type: str,
    modality: str,
    raw_unit: str,
    wt_raw_val: float,
) -> pd.DataFrame:
    """Curate a modality-pure subset (`ELISA` or `DDG`) of AbDesign (`datasets_mut.csv` + `datasets_wt.csv`)."""
    ag_map, ty_map, ab_map = load_pdb_mapping(paths.metadata)
    mut_df = pd.read_csv(paths.raw / "AbDesign" / "datasets_mut.csv", index_col=0)
    wt_df = pd.read_csv(paths.raw / "AbDesign" / "datasets_wt.csv", index_col=0)

    mut_df["pdb_up"] = mut_df["pdb_name"].astype(str).str.strip().str.upper()
    aff = pd.to_numeric(mut_df["affinity"], errors="coerce")
    atype = mut_df["affinity_type"].astype(str).str.strip().str.lower()

    if target_affinity_type == "elisa_mut_to_wt_ratio":
        valid = (atype == "elisa_mut_to_wt_ratio") & aff.notna() & (aff > 0) & mut_df["pdb_up"].isin(ag_map)
        sub = mut_df.loc[valid].copy().reset_index(drop=True)
        raw_vals = pd.to_numeric(sub["affinity"], errors="coerce").to_numpy(dtype=float)
        score = np.log10(raw_vals)
    else:
        # Exclude artificial non-binder placeholder ddG == +8.0 kcal/mol so censored rows never pair as neutral
        valid = (atype == "ddg") & aff.notna() & (aff < 7.99) & mut_df["pdb_up"].isin(ag_map)
        sub = mut_df.loc[valid].copy().reset_index(drop=True)
        raw_vals = pd.to_numeric(sub["affinity"], errors="coerce").to_numpy(dtype=float)
        score = -raw_vals / RT_LN10

    mut_part = pd.DataFrame(
        {
            "Ab_name": (sub["dataset"].astype(str) + "-" + sub["index"].astype(int).astype(str).str.zfill(3)).values,
            "WT_Abname": sub["pdb_up"].map(ab_map).values,
            "WT_PDB": sub["pdb_up"].values,
            "Ag_name": sub["pdb_up"].map(ag_map).values,
            "Ag_type": sub["pdb_up"].map(ty_map).values,
            "raw_value": raw_vals,
            "binding_score": score,
            "Ab_heavy_chain_seq": sub["heavy_sequence_variable"].values,
            "Ab_light_chain_seq": sub["light_sequence_variable"].values,
            "Ag_seq": [trim_antigen_tags(s) for s in sub["antigen_sequence"]],
            "Ag_epitope_restrictions": "None",
        }
    )

    wt_df["pdb_up"] = wt_df["pdb_name"].astype(str).str.strip().str.upper()
    wt_sub = wt_df[wt_df["pdb_up"].isin(set(sub["pdb_up"].unique()))].copy().reset_index(drop=True)
    wt_part = pd.DataFrame(
        {
            "Ab_name": wt_sub["pdb_up"].map(ab_map).values,
            "WT_Abname": wt_sub["pdb_up"].map(ab_map).values,
            "WT_PDB": wt_sub["pdb_up"].values,
            "Ag_name": wt_sub["pdb_up"].map(ag_map).values,
            "Ag_type": wt_sub["pdb_up"].map(ty_map).values,
            "raw_value": wt_raw_val,
            "binding_score": 0.0,
            "Ab_heavy_chain_seq": wt_sub["heavy_sequence_variable"].values,
            "Ab_light_chain_seq": wt_sub["light_sequence_variable"].values,
            "Ag_seq": [trim_antigen_tags(s) for s in wt_sub["antigen_sequence"]],
            "Ag_epitope_restrictions": "None",
        }
    )

    combined = pd.concat([wt_part, mut_part], ignore_index=True)
    return finalize_source_table(
        combined,
        dataset_name="AbDesign",
        modality=modality,
        raw_unit=raw_unit,
    )


def curate_abdesign_elisa(paths: RepoPaths) -> pd.DataFrame:
    """Curate AbDesign pure ELISA ratio panel (`binding_score = +log10(OD_mut / OD_wt)`)."""
    return _curate_abdesign_by_type(
        paths,
        target_affinity_type="elisa_mut_to_wt_ratio",
        modality="ELISA",
        raw_unit="elisa_wt_ratio",
        wt_raw_val=1.0,
    )


def curate_abdesign_ddg(paths: RepoPaths) -> pd.DataFrame:
    """Curate AbDesign pure ΔΔG panel (`binding_score = -ddG / (R * T * ln(10))`)."""
    return _curate_abdesign_by_type(
        paths,
        target_affinity_type="ddg",
        modality="DDG",
        raw_unit="kcal_per_mol",
        wt_raw_val=0.0,
    )


def curate_kothiwal_ec50(paths: RepoPaths) -> pd.DataFrame:
    """Curate Kothiwal et al. 10-target cell-display EC50 panels (`pEC50 = 9 - log10(EC50_nM)`)."""
    panels = sorted((paths.raw / "FLAb-Kothiwal").glob("kothiwal2025htp_*_ec50.csv"))
    df = pd.concat([pd.read_csv(f) for f in panels], ignore_index=True)

    ag_name = (
        df["Antigen"]
        .astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
        .replace({"DKK 1.00": "DKK1", "hROBO2N": "ROBO2N"})
    )
    ec50 = pd.to_numeric(df["Cell Display  - EC50 (nM)"], errors="coerce")
    valid = ec50.notna() & (ec50 > 0)
    df = df.loc[valid].copy().reset_index(drop=True)
    ag_name = ag_name.loc[valid].reset_index(drop=True)
    ec_nm = ec50.loc[valid].to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "Ab_name": df["Antibody_Name"].astype(str).values,
            "WT_Abname": (ag_name + "_Kothiwal").values,
            "WT_PDB": "NA",
            "Ag_name": ag_name.values,
            "Ag_type": "Non-Viral",
            "raw_value": ec_nm,
            "binding_score": kd_nm_to_pkd(ec_nm),
            "Ab_heavy_chain_seq": df["heavy"].values,
            "Ab_light_chain_seq": df["light"].values,
            "Ag_seq": [trim_antigen_tags(s) for s in df["antigen_seq"]],
            "Ag_epitope_restrictions": "None",
        }
    )
    return finalize_source_table(out, dataset_name="Kothiwal", modality="EC50", raw_unit="nM")


def _parse_pdb_chains_from_zip(
    zf: zipfile.ZipFile, pdb_stem: str
) -> tuple[dict[str, list[str]], dict[tuple[str, str], tuple[str, int]]]:
    """Extract ordered per-chain amino-acid lists and `(chain, res_id) -> (chain, pos_idx)` from CA atoms."""
    target_member = None
    for name in zf.namelist():
        if name.endswith(f"/{pdb_stem}.pdb") or name == f"{pdb_stem}.pdb":
            target_member = name
            break
    if target_member is None:
        raise FileNotFoundError(f"PDB {pdb_stem}.pdb not found in AbAgym PDB_files.zip")

    chain_seqs: dict[str, list[str]] = {}
    res_index: dict[tuple[str, str], tuple[str, int]] = {}
    seen_res: set[tuple[str, str]] = set()

    for line in zf.read(target_member).decode("utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM  "):
            continue
        if line[12:16].strip() != "CA":
            continue
        altloc = line[16]
        if altloc not in (" ", "A", "1"):
            continue
        resname = line[17:20].strip()
        aa = _AA3_TO_1.get(resname)
        if aa is None:
            continue
        chain = line[21]
        resid = f"{line[22:26].strip()}{line[26].strip()}"
        if (chain, resid) in seen_res:
            continue
        seen_res.add((chain, resid))
        lst = chain_seqs.setdefault(chain, [])
        res_index[(chain, resid)] = (chain, len(lst))
        lst.append(aa)

    return chain_seqs, res_index


def curate_abagym_dms(paths: RepoPaths) -> pd.DataFrame:
    """Curate the 45 net-new paired-chain DMS datasets from AbAgym (`AbAgym_data_non-redundant_interface.csv`).

    Reconstructs exact wild-type and mutant `VH`, `VL`, and `Ag` sequences from `PDB_files.zip`,
    trims Fab constant domains (`CH1`/`CL`), deduplicates against `RBD_Escape_Bloom` and `AbBiBench`,
    and sets `binding_score = -DMS_score` (since AbAgym orients all 68 datasets as `+ddG` / escape,
    where higher `DMS_score` = weaker binding).
    """
    raw_dir = paths.raw / "AbAgym"
    meta = pd.read_csv(raw_dir / "AbAgym_metadata.csv")
    dms = pd.read_csv(raw_dir / "AbAgym_data_non-redundant_interface.csv")

    meta = meta[~meta["DMS_name"].isin(ABAGYM_EXCLUDED_DATASETS)].copy().reset_index(drop=True)
    dms = dms[dms["DMS_name"].isin(set(meta["DMS_name"].unique()))].copy()

    rows = []
    with zipfile.ZipFile(raw_dir / "PDB_files.zip", "r") as zf:
        for _, mrow in meta.iterrows():
            dms_name = str(mrow["DMS_name"]).strip()
            pdb_file = str(mrow["PDB_file"]).strip()
            pdb_id = str(mrow["template_PDB_ID"]).strip().upper()
            raw_ag = str(mrow["antigen_name"]).strip()
            ag_chains_spec = str(mrow["Antigen Chains"]).strip()

            # Distinguish H1 vs H3 Influenza HA in AbAgym
            if raw_ag.lower() == "influenza":
                ag_name = "Influenza_AH3" if "H3" in dms_name or pdb_id == "3ZTJ" else "Influenza_AH1"
            else:
                ag_name = raw_ag

            chain_seqs, res_idx = _parse_pdb_chains_from_zip(zf, pdb_file)
            if "H" not in chain_seqs or "L" not in chain_seqs:
                continue

            is_heterodimer = ag_chains_spec == "[ACE][BDF]"
            score_range_spec = str(mrow["DMS_score_range"]).strip()
            is_unit_escape = score_range_spec.startswith("[0")
            is_pct_scale = "100" in score_range_spec
            sub = dms[dms["DMS_name"] == dms_name]
            if sub.empty:
                continue

            for r_idx, r in enumerate(sub.itertuples(index=False)):
                score_raw = float(r.DMS_score)
                if not np.isfinite(score_raw):
                    continue
                if is_unit_escape:
                    # Escape / survival fraction in [0, 1] -> log10(max(1 - f, 0.01))
                    if score_raw >= 0.99:
                        continue
                    b_score = float(np.log10(np.clip(1.0 - score_raw, 1e-2, 1.0)))
                elif is_pct_scale:
                    b_score = -score_raw / 10.0
                else:
                    b_score = -score_raw

                mut_chains = {c: lst.copy() for c, lst in chain_seqs.items()}
                mut_tokens = str(r.mut_names).strip().split("&")
                ok = True
                mutated_chain_ids: set[str] = set()
                for tok in mut_tokens:
                    tok = tok.strip()
                    if len(tok) < 4:
                        ok = False
                        break
                    wt_aa, ch, site, mut_aa = tok[0], tok[1], tok[2:-1], tok[-1]
                    loc = res_idx.get((ch, site))
                    if loc is None or mut_chains[ch][loc[1]] != wt_aa:
                        ok = False
                        break
                    mut_chains[ch][loc[1]] = mut_aa
                    mutated_chain_ids.add(ch)
                if not ok:
                    continue

                vh, vl = trim_fab_to_fv("".join(mut_chains["H"]), "".join(mut_chains["L"]))
                if is_heterodimer and "A" in mut_chains and "B" in mut_chains:
                    ag_seq = "".join(mut_chains["A"]) + "".join(mut_chains["B"])
                else:
                    # Monomer or symmetric homo-oligomer: take chain A (unless only B/C was mutated)
                    chosen_ch = "A"
                    if "A" not in mutated_chain_ids:
                        for alt_ch in ("B", "C"):
                            if alt_ch in mutated_chain_ids and alt_ch in mut_chains:
                                chosen_ch = alt_ch
                                break
                    ag_seq = "".join(mut_chains[chosen_ch])

                rows.append(
                    {
                        "Ab_name": f"AbAgym-{dms_name}-{r_idx + 1:04d}",
                        "WT_Abname": f"AbAgym_{dms_name}",
                        "WT_PDB": pdb_id,
                        "Ag_name": ag_name,
                        "Ag_type": "Viral" if ag_name in {"COVID-19", "HIV", "Influenza_AH1", "Influenza_AH3", "Lassa", "Nipah", "zika"} else "Non-Viral",
                        "raw_value": score_raw,
                        "binding_score": b_score,
                        "Ab_heavy_chain_seq": vh,
                        "Ab_light_chain_seq": vl,
                        "Ag_seq": ag_seq,
                        "Ag_epitope_restrictions": str(mrow["DMS_on"]),
                    }
                )

    return finalize_source_table(
        pd.DataFrame(rows),
        dataset_name="AbAgym",
        modality="DMS",
        raw_unit="neg_DMS_score",
    )


def curate_nbbench_galectin3(paths: RepoPaths) -> pd.DataFrame:
    """Curate NbBench (`ZYMScott/vhh_affinity-seq`, 12,735 unique `cAbBCII10`-scaffold VHHs)
    deep-sequencing affinity enrichment against human Galectin-3 (`LGALS3`, UniProt `P17931`).

    Transforms raw positive fold-enrichment ratio `score > 0` into `log10(score)` (`higher = stronger`).
    """
    catalog = load_antigen_catalog(paths.metadata)
    _, gal3_seq = catalog["Galectin3"]
    nb_dir = paths.raw / "NbBench"
    df = pd.concat(
        [pd.read_csv(nb_dir / f"{sp}.csv") for sp in ("train", "val", "test")],
        ignore_index=True,
    )
    sc = pd.to_numeric(df["score"], errors="coerce")
    df = df[sc.notna() & (sc > 0)].copy().reset_index(drop=True)
    raw_sc = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "Ab_name": [f"NbBench_Gal3-{int(i)}" for i in df["ID"]],
            "WT_Abname": "cAbBCII10",
            "WT_PDB": "NA",
            "Ag_name": "Galectin3",
            "Ag_type": "Non-Viral",
            "raw_value": raw_sc,
            "binding_score": np.log10(raw_sc),
            "Ab_heavy_chain_seq": df["seq"].astype(str).values,
            "Ab_light_chain_seq": "",
            "Ag_seq": gal3_seq,
            "Ag_epitope_restrictions": "None",
        }
    )
    return finalize_source_table(
        out,
        dataset_name="NbBench",
        modality="Enrichment",
        raw_unit="log10_enrichment",
        require_light_chain=False,
    )
