"""Transfer-learning track curation:
- Neutralization `IC50` (`CATNAP` HIV-1, `AbCoV` Coronavirus, `FLAb-Shanker` SARS-CoV-2)
- Viral escape DMS (`RBD_Escape_Bloom` SARS-CoV-2 RBD single-mutant escape maps)

Unified sign convention (`higher binding_score = stronger neutralization / stronger binding`):
- `IC50 [ug/mL]` -> `binding_score = -log10(IC50_ug_mL)`
- `Escape f in [0, 1]` -> `binding_score = +log10(max(1.0 - f, 0.01))`
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

from abrank.config import RepoPaths
from abrank.curate.common import finalize_source_table, read_fasta_records
from abrank.curate.kd_core import _ABCOV_SPIKE_SEQS
from abrank.schema import parse_numeric_measurement, trim_fab_to_fv


def _extract_catnap_id(seq_id: str) -> str:
    if any(w in seq_id.lower() for w in ("chain", "heavy", "light")):
        return seq_id.split("__")[0] if "__" in seq_id else seq_id.split("_")[0]
    return seq_id.split(".")[-2]


def _read_catnap_fasta(filepath: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    seq_id: str | None = None
    buf: list[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    cid = _extract_catnap_id(seq_id)
                    out[cid] = "".join(buf).replace("-", "").replace("*", "").replace("#", "").replace("X", "")
                seq_id = line[1:]
                buf = []
            else:
                buf.append(line)
        if seq_id is not None:
            cid = _extract_catnap_id(seq_id)
            out[cid] = "".join(buf).replace("-", "").replace("*", "").replace("#", "").replace("X", "")
    return out


def curate_catnap_ic50(paths: RepoPaths) -> pd.DataFrame:
    """Curate exact uncensored HIV-1 pseudovirus neutralization IC50 [ug/mL] from LANL CATNAP."""
    raw_dir = paths.raw / "CATNAP"
    ab_meta = pd.read_csv(raw_dir / "abs.txt", sep="\t")
    ab_epi = {
        str(r["Name"]).replace(" ", ""): str(r["Binding Type"])
        for _, r in ab_meta.iterrows()
    }

    assay = pd.read_csv(raw_dir / "assay.txt", sep="\t", low_memory=False)
    ag_dict = _read_catnap_fasta(raw_dir / "virseqs_aa.fasta")
    h_dict = _read_catnap_fasta(raw_dir / "heavy_seqs_aa.fasta")
    l_dict = _read_catnap_fasta(raw_dir / "light_seqs_aa.fasta")

    rows = []
    for _, r in assay.iterrows():
        ab_raw = str(r["Antibody"]).replace(" ", "")
        if "+" in ab_raw:
            # Exclude antibody cocktails so every measurement maps to a single mAb
            continue
        virus = str(r["Virus"]).replace(" ", "")
        if ab_raw not in h_dict or ab_raw not in l_dict or virus not in ag_dict:
            continue
        fval, op = parse_numeric_measurement(r["IC50"])
        if op != "=" or not np.isfinite(fval):
            continue

        rows.append(
            {
                "Ab_name": f"abHIV-{ab_raw}",
                "WT_Abname": f"abHIV-{ab_raw}",
                "WT_PDB": "NA",
                "Ag_name": f"HIV-{virus}",
                "Ag_type": "Viral",
                "raw_value": float(fval),
                "binding_score": float(-np.log10(fval)),
                "Ab_heavy_chain_seq": h_dict[ab_raw],
                "Ab_light_chain_seq": l_dict[ab_raw],
                "Ag_seq": ag_dict[virus],
                "Ag_epitope_restrictions": ab_epi.get(ab_raw, "None"),
            }
        )

    return finalize_source_table(
        pd.DataFrame(rows), dataset_name="CATNAP", modality="IC50", raw_unit="ug/mL"
    )


def curate_abcov_ic50(paths: RepoPaths) -> pd.DataFrame:
    """Curate exact uncensored coronavirus neutralization IC50 [ug/mL] from AbCoV."""
    abcov_df = pd.read_csv(paths.raw / "AbCoV" / "AbCoV_AbAg_affinity.csv")
    details = pd.read_csv(paths.raw / "AbCoV" / "AbCoV_antibody_details.csv")

    h_map, l_map, pdb_map = {}, {}, {}
    for _, r in details.iterrows():
        nm = str(r["Name"]).replace(" ", "")
        h_map[nm] = r["VH or VHH"]
        l_map[nm] = r["VL"]
        pdb_map[nm] = str(r["Structures"]).replace("https://www.rcsb.org/structure/", "").strip()

    rows = []
    for _, r in abcov_df.iterrows():
        ab_nm = str(r["Antibody name"]).replace(" ", "")
        if ab_nm not in h_map:
            continue
        raw_ic = str(r.get("IC50", "")).strip()
        if not raw_ic or raw_ic.upper() in {"NA", "NAN", "ND"} or raw_ic.startswith(("<", ">", "≤", "≥")):
            continue

        ic_ug_ml = np.nan
        low = raw_ic.lower()
        m_num = re.search(r"^\d+(?:\.\d+)?(?:e[-+]?\d+)?", low)
        if m_num:
            val = float(m_num.group(0))
            if "ug" in low or "µg" in low:
                ic_ug_ml = val
            elif "ng" in low:
                ic_ug_ml = val / 1000.0
            elif "nm" in low:
                ic_ug_ml = (val * 150.0) / 1000.0
        if not (np.isfinite(ic_ug_ml) and ic_ug_ml > 0):
            continue

        neut = str(r["Neutralizes"])
        ag_prefix, ag_seq = None, None
        for key, (pref, seq) in _ABCOV_SPIKE_SEQS.items():
            if key in neut:
                ag_prefix, ag_seq = pref, seq
                break
        if ag_prefix is None:
            continue

        epi_raw = str(r["Viral protein:epitope"])
        epi = "None"
        for tag in ("non-RBD", "RBD", "S2", "NTD", "S1"):
            if tag in epi_raw:
                epi = tag.replace("-", "")
                break

        hs, ls = trim_fab_to_fv(h_map[ab_nm], l_map[ab_nm])
        rows.append(
            {
                "Ab_name": f"AbCoV-{ab_nm}",
                "WT_Abname": f"AbCoV-{ab_nm}",
                "WT_PDB": pdb_map.get(ab_nm, "NA"),
                "Ag_name": f"{ag_prefix}_{epi}",
                "Ag_type": "Viral",
                "raw_value": float(ic_ug_ml),
                "binding_score": float(-np.log10(ic_ug_ml)),
                "Ab_heavy_chain_seq": hs,
                "Ab_light_chain_seq": ls,
                "Ag_seq": ag_seq,
                "Ag_epitope_restrictions": epi,
            }
        )

    return finalize_source_table(
        pd.DataFrame(rows), dataset_name="AbCoV", modality="IC50", raw_unit="ug/mL"
    )


def curate_shanker_ic50(paths: RepoPaths) -> pd.DataFrame:
    """Curate Shanker et al. 2024 SARS-CoV-2 pseudovirus IC50 panels (fixing FLAb 1,000x unit error).

    In FLAb's CSVs, all 4 files label the column `'Avg Neutralization IC50 (ng/µL)'` (`= ug/mL`),
    but `Ly1404_Wuhan_IC50.csv` and `SA58-BA.1_IC50.csv` were recorded in **`ng/mL`** (`WT = 2.66`),
    whereas `Ly1404-BQ.1.1_IC50.csv` and `SA58-BQ.1.1_IC50.csv` were recorded in **`ug/mL`**!
    """
    ag_records = read_fasta_records(
        paths.metadata / "constructs" / "Shanker_SARS_CoV_2_antigens.fasta"
    )
    panels = [
        (
            "shanker2024unsupervised_Ly1404_Wuhan_IC50.csv",
            "LY-CoV1404",
            "SARSCoV2_Wuhan",
            ag_records["SARSCoV2_Wuhan-Hu-1_Spike_delta21"],
            1e-3,  # ng/mL -> ug/mL
        ),
        (
            "shanker2024unsupervised_SA58-BA.1_IC50.csv",
            "SA58",
            "SARSCoV2_BA.1",
            ag_records["SARSCoV2_Omicron_BA.1_Spike_delta21"],
            1e-3,  # ng/mL -> ug/mL
        ),
        (
            "shanker2024unsupervised_Ly1404-BQ.1.1_IC50.csv",
            "LY-CoV1404",
            "SARSCoV2_BQ.1.1",
            ag_records["SARSCoV2_Omicron_BQ.1.1_Spike_delta21"],
            1.0,   # already ug/mL
        ),
        (
            "shanker2024unsupervised_SA58-BQ.1.1_IC50.csv",
            "SA58",
            "SARSCoV2_BQ.1.1",
            ag_records["SARSCoV2_Omicron_BQ.1.1_Spike_delta21"],
            1.0,   # already ug/mL
        ),
    ]

    rows = []
    for fname, wt_ab, ag_name, ag_seq, scale_to_ug_ml in panels:
        df = pd.read_csv(paths.raw / "FLAb-Shanker" / fname)
        ic_col = "Avg Neutralization IC50 (ng/µL)"
        vals = pd.to_numeric(df[ic_col], errors="coerce")
        sub = df[vals.notna() & (vals > 0)].copy().reset_index(drop=True)
        for _, r in sub.iterrows():
            ic_ug_ml = float(r[ic_col]) * scale_to_ug_ml
            des = str(r["design"]).strip()
            rows.append(
                {
                    "Ab_name": f"Shanker-{wt_ab}-{des}",
                    "WT_Abname": wt_ab,
                    "WT_PDB": "NA",
                    "Ag_name": ag_name,
                    "Ag_type": "Viral",
                    "raw_value": ic_ug_ml,
                    "binding_score": float(-np.log10(ic_ug_ml)),
                    "Ab_heavy_chain_seq": r["heavy"],
                    "Ab_light_chain_seq": r["light"],
                    "Ag_seq": ag_seq,
                    "Ag_epitope_restrictions": "RBD",
                }
            )

    return finalize_source_table(
        pd.DataFrame(rows), dataset_name="Shanker", modality="IC50", raw_unit="ug/mL"
    )


def curate_rbd_escape_bloom(paths: RepoPaths) -> pd.DataFrame:
    """Curate Bloom Lab SARS-CoV-2 RBD deep mutational scanning escape dataset (`full_dataset.csv`).

    Uses the exact 201-aa mutated RBD sequence directly from the `rbd` column and computes
    `binding_score = +log10(max(1.0 - mut_escape, 0.01))` (`higher = stronger binding / less escape`).
    """
    df = pd.read_csv(paths.raw / "RBD_Escape_Bloom" / "full_dataset.csv")
    esc = pd.to_numeric(df["mut_escape"], errors="coerce")
    df = df[esc.notna() & (esc >= 0.0)].copy().reset_index(drop=True)
    esc_arr = pd.to_numeric(df["mut_escape"], errors="coerce").to_numpy(dtype=float)

    binding_score = np.log10(np.clip(1.0 - esc_arr, 0.01, 1.0))
    h_seqs = df["heavy_variable"].astype(str).str.replace("-", "", regex=False).values
    l_seqs = df["light_variable"].astype(str).str.replace("-", "", regex=False).values
    ag_seqs = df["rbd"].astype(str).str.replace("-", "", regex=False).values
    ab_names = ("AbBloom-" + df["antibody"].astype(str).str.strip()).values
    ag_names = (
        "SARSCoV2_RBD_" + df["site"].astype(str) + df["mutation"].astype(str)
    ).values
    epis = ("RBD_m" + df["subtype"].astype(str)).values

    out = pd.DataFrame(
        {
            "Ab_name": ab_names,
            "WT_Abname": ab_names,
            "WT_PDB": "NA",
            "Ag_name": ag_names,
            "Ag_type": "Viral",
            "raw_value": esc_arr,
            "binding_score": binding_score,
            "Ab_heavy_chain_seq": h_seqs,
            "Ab_light_chain_seq": l_seqs,
            "Ag_seq": ag_seqs,
            "Ag_epitope_restrictions": epis,
        }
    )
    return finalize_source_table(
        out,
        dataset_name="RBD_Escape_Bloom",
        modality="Escape",
        raw_unit="escape_fraction",
    )
