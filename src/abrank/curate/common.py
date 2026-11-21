"""Shared helpers and unified output columns for AbRank v2 dataset curation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from abrank.schema import validate_assay_dataframe
from abrank.taxonomy import canonical_ag_group, canonical_ag_type

ASSAY_COLUMNS = [
    "Ab_name",
    "WT_Abname",
    "WT_PDB",
    "Ag_name",
    "Ag_group",
    "Ag_type",
    "Assay_modality",
    "raw_value",
    "raw_unit",
    "binding_score",
    "Ab_heavy_chain_seq",
    "Ab_light_chain_seq",
    "Ag_seq",
    "Ag_epitope_restrictions",
    "N_replicates",
    "Dataset",
]


def read_fasta_records(path: Path) -> dict[str, str]:
    """Read a FASTA file into `{header_first_token: sequence}` and `{full_header: sequence}`."""
    records: dict[str, str] = {}
    header: str | None = None
    buf: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(buf)
                    records[header] = seq
                    records[header.split()[0].split("|")[0]] = seq
                header = line[1:].strip()
                buf = []
            else:
                buf.append(line)
        if header is not None:
            seq = "".join(buf)
            records[header] = seq
            records[header.split()[0].split("|")[0]] = seq
    return records


def load_pdb_mapping(metadata_dir: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Load `pdb_mapping.tsv` into `(pdb_to_agname, pdb_to_agtype, pdb_to_abname)`."""
    df = pd.read_csv(metadata_dir / "pdb_mapping.tsv", sep="\t")
    df["PDB_ID"] = df["PDB_ID"].astype(str).str.strip()
    ag_map = df.dropna(subset=["Antigen_name"]).set_index("PDB_ID")["Antigen_name"].to_dict()
    ty_map = df.dropna(subset=["Ag_type"]).set_index("PDB_ID")["Ag_type"].to_dict()
    ab_map = df.dropna(subset=["Antibody_name"]).set_index("PDB_ID")["Antibody_name"].to_dict()
    return ag_map, ty_map, ab_map


def load_antigen_catalog(metadata_dir: Path) -> dict[str, tuple[str, str]]:
    """Load `antigen_catalog.tsv` into `{Antigen_name: (Ag_type, Ag_seq)}`."""
    df = pd.read_csv(metadata_dir / "antigen_catalog.tsv", sep="\t")
    return {
        str(r.Antigen_name).strip(): (str(r.Ag_type).strip(), str(r.Ag_seq).strip())
        for r in df.itertuples(index=False)
    }


def finalize_source_table(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    modality: str,
    raw_unit: str,
    require_light_chain: bool = True,
    min_ag_len: int = 10,
    collapse_replicates: bool = True,
) -> pd.DataFrame:
    """Standardize columns, canonicalize `Ag_group` / `Ag_type`, collapse replicates, and validate."""
    out = df.copy()
    out["Dataset"] = dataset_name
    out["Assay_modality"] = modality
    out["raw_unit"] = raw_unit

    if "WT_Abname" not in out.columns:
        out["WT_Abname"] = out["Ab_name"]
    out["WT_Abname"] = out["WT_Abname"].fillna(out["Ab_name"]).astype(str)

    if "WT_PDB" not in out.columns:
        out["WT_PDB"] = "NA"
    out["WT_PDB"] = out["WT_PDB"].fillna("NA").astype(str)

    if "Ag_epitope_restrictions" not in out.columns:
        out["Ag_epitope_restrictions"] = "None"
    out["Ag_epitope_restrictions"] = out["Ag_epitope_restrictions"].fillna("None").astype(str)

    if "N_replicates" not in out.columns:
        out["N_replicates"] = 1
    out["N_replicates"] = pd.to_numeric(out["N_replicates"], errors="coerce").fillna(1).astype(int)

    # Canonicalize Ag_group and Ag_type
    species_series = out["antigen_species"] if "antigen_species" in out.columns else pd.Series([None] * len(out), index=out.index)
    hint_series = out["Ag_type"] if "Ag_type" in out.columns else pd.Series([None] * len(out), index=out.index)
    out["Ag_group"] = [canonical_ag_group(n) for n in out["Ag_name"]]
    out["Ag_type"] = [
        canonical_ag_type(grp, hint=h, species=sp)
        for grp, h, sp in zip(out["Ag_group"], hint_series, species_series)
    ]

    out = validate_assay_dataframe(
        out,
        name=f"{dataset_name}:{modality}",
        require_light_chain=require_light_chain,
        min_ag_len=min_ag_len,
    )

    if collapse_replicates and not out.empty:
        grp_cols = ["Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"]
        if out.duplicated(subset=grp_cols).any():
            before = len(out)
            agg = (
                out.groupby(grp_cols, as_index=False, sort=False)
                .agg(
                    Ab_name=("Ab_name", "first"),
                    WT_Abname=("WT_Abname", "first"),
                    WT_PDB=("WT_PDB", "first"),
                    Ag_name=("Ag_name", "first"),
                    Ag_group=("Ag_group", "first"),
                    Ag_type=("Ag_type", "first"),
                    Assay_modality=("Assay_modality", "first"),
                    raw_unit=("raw_unit", "first"),
                    binding_score=("binding_score", "mean"),
                    raw_value=("raw_value", lambda s: float(np.exp(np.mean(np.log(np.clip(pd.to_numeric(s, errors="coerce"), 1e-15, None))))) if modality in {"KD", "EC50", "IC50"} else float(np.mean(pd.to_numeric(s, errors="coerce")))),
                    Ag_epitope_restrictions=("Ag_epitope_restrictions", "first"),
                    N_replicates=("N_replicates", "sum"),
                    Dataset=("Dataset", "first"),
                )
            )
            # Re-derive exact raw_value for KD/EC50 from mean binding_score (pKd = 9 - log10(Kd_nM))
            if modality in {"KD", "EC50"}:
                agg["raw_value"] = 10.0 ** (9.0 - agg["binding_score"])
            elif modality == "IC50":
                agg["raw_value"] = 10.0 ** (-agg["binding_score"])
            out = agg
            print(f"[{dataset_name}:{modality}] Collapsed {before:,} -> {len(out):,} unique (VH, VL, Ag) complexes in log space.")

    return out[ASSAY_COLUMNS].reset_index(drop=True)
