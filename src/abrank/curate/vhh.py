"""Single-domain VHH / nanobody KD curation across 8 experimental sources:
`Germinal_VHH`, `BoltzGen_VHH`, `SAbDab_VHH`, `AbCoV_VHH`, `Porebski_VHH`,
`SKEMPIv2_VHH`, `Adaptyv_EGFR_VHH`, and `Bennett_VHH`.

Kept strictly separate from the paired-chain `VH + VL` core benchmark and evaluated in a
dedicated `vhh` 3-state ranking and cross-architecture transfer track (`data/benchmark/assays_vhh.tsv`).
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
from abrank.curate.kd_2026 import _parse_germinal_kd_nm
from abrank.curate.kd_core import _ABCOV_SPIKE_SEQS, _SABDAB_TERM_MAP
from abrank.schema import (
    clean_seq,
    kd_nm_to_pkd,
    parse_numeric_measurement,
    trim_antigen_tags,
)
from abrank.taxonomy import SABDAB_PDB_OVERRIDES

_BOLTZGEN_FILE_TO_TARGET: dict[str, tuple[str, str]] = {
    "AMBP Binders_summary.csv": ("AMBP", "AMBP"),
    "GM2A Binders_summary.csv": ("GM2A", "GM2A"),
    "HNMT binders_summary.csv": ("HNMT", "HNMT"),
    "IDI2 binders_summary.csv": ("IDI2", "IDI2"),
    "MZB1 Binders_summary.csv": ("MZB1", "MZB1"),
    "ORM2 Binders_summary.csv": ("ORM2", "ORM2"),
    "PHYH_summary.csv": ("PHYH", "PHYH"),
    "PMVK Binders_summary.csv": ("PMVK", "PMVK"),
    "RFK Binders_summary.csv": ("RFK", "RFK"),
    "IL-7RA binders_summary.csv": ("IL7RA", "IL-7RA"),
    "Insulin binders_summary.csv": ("InsulinR", "InsulinR"),
    "PDGFR binders_summary.csv": ("PDGFRB", "PDGFRB"),
    "PDL1 binders_summary.csv": ("PDL1", "PD-L1"),
    "TNFa binders_summary.csv": ("TNFa", "TNFa_nano_designs_15cifs"),
}


def _trim_vhh_tags(seq: str) -> str:
    """Trim N-terminal leader residues (`MSG`, `MA`) and C-terminal linker/His6 tags on recombinant VHHs."""
    s = clean_seq(seq)
    if s.startswith("MSG") and s[3:7] in {"QVQL", "EVQL", "DVQL", "QVKL"}:
        s = s[3:]
    # Strip C-terminal affinity tags (e.g. SGSGSHHWGSTHHHHHH, ALEHHHHHH, LEHHHHHH, KKKHHHHHH, LE)
    s = re.sub(r"(?:SGSGSHHWGST|ENLYFQGS|ALE|LE|KKK|AAALE|G{1,4}S)?H{5,10}$", "", s)
    if s.endswith("TVSSLE"):
        s = s[:-2]
    return s


def curate_germinal_vhh(paths: RepoPaths) -> pd.DataFrame:
    """Curate Germinal 2026 de novo VHH BLI binders (`PDL1`, `IL3`, `IL20`, `EBV_BHRF1`)."""
    ag_records = read_fasta_records(paths.metadata / "constructs" / "Germinal_antigens.fasta")
    sheets = [
        ("PDL1 nanobody", "PDL1", "Non-Viral", ag_records["PDL1_WT_ectodomain"]),
        ("IL3 nanobody", "IL3", "Non-Viral", ag_records["IL3_WT_mature"]),
        ("IL20 nanobody", "IL20", "Non-Viral", ag_records["IL20_WT_mature"]),
        ("BHRF1 nanobody", "EBV_BHRF1", "Viral", ag_records["BHRF1_WT_soluble_core"]),
    ]
    xl = pd.ExcelFile(paths.raw / "Germinal" / "41587_2026_3187_MOESM5_ESM.xlsx")

    rows = []
    for sheet, ag_name, ag_type, ag_seq in sheets:
        df = xl.parse(sheet)
        seq_col = "Protein_Sequence" if "Protein_Sequence" in df.columns else "Protein_sequence"
        for _, r in df.iterrows():
            kd_nm = _parse_germinal_kd_nm(r.get("Kd (nM)"))
            if not np.isfinite(kd_nm):
                continue
            nm = str(r["Name"]).strip()
            rows.append(
                {
                    "Ab_name": f"GerminalVHH-{nm}",
                    "WT_Abname": f"GerminalVHH_{ag_name}",
                    "WT_PDB": "NA",
                    "Ag_name": ag_name,
                    "Ag_type": ag_type,
                    "raw_value": float(kd_nm),
                    "binding_score": float(kd_nm_to_pkd(kd_nm)),
                    "Ab_heavy_chain_seq": _trim_vhh_tags(str(r[seq_col])),
                    "Ab_light_chain_seq": "",
                    "Ag_seq": ag_seq,
                    "Ag_epitope_restrictions": "None",
                }
            )

    return finalize_source_table(
        pd.DataFrame(rows),
        dataset_name="Germinal_VHH",
        modality="KD",
        raw_unit="nM",
        require_light_chain=False,
    )


def curate_boltzgen_vhh(paths: RepoPaths) -> pd.DataFrame:
    """Curate BoltzGen 2025/2026 Adaptyv Bio SPR/BLI VHH binders across 14 targets.

    Strictly filters to VHH / nanobody designs (`'nano'` in `name` or `QVQL`/`EVQL` framework,
    excluding de novo non-immunoglobulin mini-protein binders) and averages replicate sensorgrams
    in `log10(Kd [nM])` space.
    """
    ag_records = read_fasta_records(
        paths.metadata / "constructs" / "BoltzGen_target_sequences.fasta"
    )
    sum_dir = paths.raw / "BoltzGen" / "adaptyv_summaries"

    rows = []
    for fname, (ag_name, fasta_key) in sorted(_BOLTZGEN_FILE_TO_TARGET.items()):
        fpath = sum_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        kd_m = pd.to_numeric(df["kd"], errors="coerce")
        is_vhh = df["name"].astype(str).str.lower().str.contains("nano") | df["sequence"].astype(
            str
        ).str.startswith(("QVQL", "EVQL", "DVQL"))
        sub = df[is_vhh & kd_m.notna() & (kd_m > 0)].copy().reset_index(drop=True)
        if sub.empty:
            continue

        ag_seq = ag_records[fasta_key]
        for _, r in sub.iterrows():
            kd_nm = float(r["kd"]) * 1e9
            nm = str(r["name"]).strip()
            rows.append(
                {
                    "Ab_name": f"BoltzGen-{nm}",
                    "WT_Abname": f"BoltzGen_{ag_name}",
                    "WT_PDB": "NA",
                    "Ag_name": ag_name,
                    "Ag_type": "Non-Viral",
                    "raw_value": kd_nm,
                    "binding_score": float(kd_nm_to_pkd(kd_nm)),
                    "Ab_heavy_chain_seq": _trim_vhh_tags(str(r["sequence"])),
                    "Ab_light_chain_seq": "",
                    "Ag_seq": ag_seq,
                    "Ag_epitope_restrictions": "None",
                }
            )

    return finalize_source_table(
        pd.DataFrame(rows),
        dataset_name="BoltzGen_VHH",
        modality="KD",
        raw_unit="nM",
        require_light_chain=False,
    )


def curate_sabdab_vhh(paths: RepoPaths) -> pd.DataFrame:
    """Curate structurally resolved single-domain VHH–protein complexes with exact KD from SAbDab-nano.

    Excludes non-protein haptens (`5VM0`, `3QXT`, `3QXV`), misannotated paired-chain tandem DFabs
    (`len(vhh) > 185`, e.g. `4HJJ`, `5FHX`), and non-standard peptide residues (`X`).
    """
    fpath = paths.raw / "SabDab" / "SabDab_vhh_rcsb_seqs.tsv"
    df = pd.read_csv(fpath, sep="\t")

    hapten_pdbs = {"5VM0", "3QXT", "3QXV"}
    rows = []
    for _, r in df.iterrows():
        pdb_id = str(r["pdb"]).strip().upper()
        if pdb_id in hapten_pdbs:
            continue
        fval, op = parse_numeric_measurement(r["affinity"])
        if op != "=" or not np.isfinite(fval) or fval <= 0:
            continue
        kd_nm = float(fval) * 1e9  # SAbDab_aff_summary stores affinity in Molar (M)

        vhh = _trim_vhh_tags(str(r["vhh_seq"]))
        if not (90 <= len(vhh) <= 185):
            continue
        ag_seq = trim_antigen_tags(r["ag_seq"])
        if len(ag_seq) < 15:
            continue

        sp = str(r.get("antigen_species", ""))
        raw_det = str(r.get("antigen_name", "")).split("|")[0].split(",")[0].strip()
        if pdb_id in SABDAB_PDB_OVERRIDES:
            ag_name, ag_type = SABDAB_PDB_OVERRIDES[pdb_id]
        else:
            det_low = raw_det.lower()
            ag_name = _SABDAB_TERM_MAP.get(det_low, raw_det.lower().replace(" ", "_"))
            ag_type = "Non-Viral"

        rows.append(
            {
                "Ab_name": f"SabDabVHH-{pdb_id}_{r['Hchain']}",
                "WT_Abname": f"SabDabVHH_{pdb_id}",
                "WT_PDB": pdb_id,
                "Ag_name": ag_name,
                "Ag_type": ag_type,
                "antigen_species": sp,
                "raw_value": kd_nm,
                "binding_score": float(kd_nm_to_pkd(kd_nm)),
                "Ab_heavy_chain_seq": vhh,
                "Ab_light_chain_seq": "",
                "Ag_seq": ag_seq,
                "Ag_epitope_restrictions": "None",
            }
        )

    return finalize_source_table(
        pd.DataFrame(rows),
        dataset_name="SabDab_VHH",
        modality="KD",
        raw_unit="nM",
        require_light_chain=False,
    )


def curate_abcov_vhh(paths: RepoPaths) -> pd.DataFrame:
    """Curate exact uncensored coronavirus nanobody (`Nb`) KD measurements from AbCoV."""
    abcov_df = pd.read_csv(paths.raw / "AbCoV" / "AbCoV_AbAg_affinity.csv")
    details = pd.read_csv(paths.raw / "AbCoV" / "AbCoV_antibody_details.csv")

    vhh_map: dict[str, str] = {}
    pdb_map: dict[str, str] = {}
    for _, r in details.iterrows():
        is_nb = str(r.get("Ab or Nb", "")).strip().lower() == "nb"
        has_vhh_only = pd.isna(r.get("VL")) and pd.notna(r.get("VH or VHH"))
        if not (is_nb and has_vhh_only):
            continue
        nm = str(r["Name"]).replace(" ", "")
        vhh_map[nm] = _trim_vhh_tags(str(r["VH or VHH"]))
        pdb_raw = str(r.get("Structures", "")).replace("https://www.rcsb.org/structure/", "").strip()
        pdb_map[nm] = pdb_raw if pdb_raw and pdb_raw.lower() != "nan" else "NA"

    rows = []
    for _, r in abcov_df.iterrows():
        ab_nm = str(r["Antibody name"]).replace(" ", "")
        if ab_nm not in vhh_map:
            continue
        fval, op = parse_numeric_measurement(r["Binding Affinity (KD)"])
        if op != "=" or not np.isfinite(fval) or fval <= 0:
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

        rows.append(
            {
                "Ab_name": f"AbCoVVHH-{ab_nm}",
                "WT_Abname": f"AbCoVVHH-{ab_nm}",
                "WT_PDB": pdb_map.get(ab_nm, "NA"),
                "Ag_name": f"{ag_prefix}_{epi}",
                "Ag_type": "Viral",
                "raw_value": float(fval),
                "binding_score": float(kd_nm_to_pkd(fval)),
                "Ab_heavy_chain_seq": vhh_map[ab_nm],
                "Ab_light_chain_seq": "",
                "Ag_seq": ag_seq,
                "Ag_epitope_restrictions": epi,
            }
        )

    return finalize_source_table(
        pd.DataFrame(rows),
        dataset_name="AbCoV_VHH",
        modality="KD",
        raw_unit="nM",
        require_light_chain=False,
    )


def curate_porebski_vhh(paths: RepoPaths) -> pd.DataFrame:
    """Curate Porebski et al. (Nat. Biomed. Eng. 2024) anti-HEL Octet BLI synthetic camelid VHHs.

    Reconstructs exact full-length Kruse synthetic VHH sequences (`Kerafast EF0014-FP`, McMahon et al.,
    Nat. Struct. Mol. Biol. 2018; Supplementary Table `DNA oligos for Kruse nanobody library assembly`)
    and filters out clamped non-binder ceiling fits (`KD == 10 uM` or unphysical negative `Kd (1/s) <= 0`)
    from `41551_2023_1093_MOESM5_ESM.xlsx` (`Nb _HEL` sheet).
    """
    catalog = load_antigen_catalog(paths.metadata)
    _, hewl_seq = catalog["HEWL"]
    df = pd.read_excel(
        paths.raw / "Porebski" / "41551_2023_1093_MOESM5_ESM.xlsx",
        sheet_name="Nb _HEL",
        header=1,
    )

    fr1 = "QVQLQESGGGLVQAGGSLRLSCAAS"
    fr2_core = "GWYRQAPGKER"
    fr3 = "ADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYC"
    fr4 = "WGQGTQVTVSS"

    rows = []
    for _, r in df.iterrows():
        kd_m = pd.to_numeric(r.get("KD (M)"), errors="coerce")
        koff = pd.to_numeric(r.get("Kd (1/s)"), errors="coerce")
        # Exclude clamped 10 uM Octet non-binder ceiling and unphysical negative dissociation fits
        if not (np.isfinite(kd_m) and 0 < kd_m < 9.99e-6 and np.isfinite(koff) and koff > 0):
            continue

        c1 = clean_seq(r["CDR1"])
        c2 = clean_seq(r["CDR2"])
        c3 = clean_seq(r["CDR3"])
        # Strip any partial J-region suffix already included in CDR3 column before appending canonical FR4
        c3_core = re.sub(r"WGQG(?:TQVT(?:VSS)?)?$", "", c3)
        # Note: clone F9 carries a 14-AA CDR2 ending in 'YYA' (overlapping the first 'A' of FR3 'ADSVK...')
        fr3_use = fr3[1:] if c2.endswith("YYA") else fr3
        vhh_seq = fr1 + c1 + fr2_core + c2 + fr3_use + c3_core + fr4
        kd_nm = float(kd_m) * 1e9
        cid = str(r["Clone ID"]).strip()

        rows.append(
            {
                "Ab_name": f"PorebskiVHH-{cid}",
                "WT_Abname": "Kruse_Nb_HEL",
                "WT_PDB": "NA",
                "Ag_name": "HEWL",
                "Ag_type": "Non-Viral",
                "raw_value": kd_nm,
                "binding_score": float(kd_nm_to_pkd(kd_nm)),
                "Ab_heavy_chain_seq": vhh_seq,
                "Ab_light_chain_seq": "",
                "Ag_seq": hewl_seq,
                "Ag_epitope_restrictions": "None",
            }
        )

    return finalize_source_table(
        pd.DataFrame(rows),
        dataset_name="Porebski_VHH",
        modality="KD",
        raw_unit="nM",
        require_light_chain=False,
    )


def curate_skempiv2_vhh(paths: RepoPaths) -> pd.DataFrame:
    """Curate SKEMPI 2.0 single-domain EGFR nanobodies (`7D12` `4KRL`, `EgA1` `4KRO`, `9G8` `4KRP`)
    wild-type and exact uncensored SPR point/double mutants.
    """
    sk_dir = paths.raw / "SKEMPIv2"
    sk = pd.read_csv(sk_dir / "skempi_v2_original.csv", sep="\t")
    ab_wt = read_fasta_records(sk_dir / "skempiv2_ab_seqres.fasta")
    ag_wt = read_fasta_records(sk_dir / "skempiv2_ag_seqres.fasta")

    nb_meta = {
        "4KRL": ("7D12", "4krl_B", "4krl_A"),
        "4KRO": ("EgA1", "4kro_B", "4kro_A"),
        "4KRP": ("9G8", "4krp_B", "4krp_A"),
    }
    sub = sk[sk["#Pdb"].astype(str).str[:4].str.upper().isin(nb_meta)].copy()

    rows = []
    seen_wt: set[str] = set()
    for _, r in sub.iterrows():
        pdb4 = str(r["#Pdb"])[:4].upper()
        nb_name, ab_key, ag_key = nb_meta[pdb4]
        wt_vhh = _trim_vhh_tags(ab_wt[ab_key])
        wt_ag = trim_antigen_tags(ag_wt[ag_key])

        if pdb4 not in seen_wt:
            wt_val, wt_op = parse_numeric_measurement(r["Affinity_wt (M)"])
            if wt_op == "=" and np.isfinite(wt_val) and wt_val > 0:
                seen_wt.add(pdb4)
                wt_nm = float(wt_val) * 1e9
                rows.append(
                    {
                        "Ab_name": f"SKEMPIVHH-{nb_name}-WT",
                        "WT_Abname": nb_name,
                        "WT_PDB": pdb4,
                        "Ag_name": "EGFR",
                        "Ag_type": "Non-Viral",
                        "raw_value": wt_nm,
                        "binding_score": float(kd_nm_to_pkd(wt_nm)),
                        "Ab_heavy_chain_seq": wt_vhh,
                        "Ab_light_chain_seq": "",
                        "Ag_seq": wt_ag,
                        "Ag_epitope_restrictions": "None",
                    }
                )

        mut_val, mut_op = parse_numeric_measurement(r["Affinity_mut (M)"])
        if mut_op != "=" or not np.isfinite(mut_val) or mut_val <= 0:
            continue

        mut_clean = str(r["Mutation(s)_cleaned"]).strip()
        mut_slug = mut_clean.replace(",", "-")
        ab_mut_file = sk_dir / "mutant-seqres" / "ab" / f"{pdb4.lower()}-{mut_slug}.fasta"
        ag_mut_file = sk_dir / "mutant-seqres" / "ag" / f"{pdb4.lower()}-{mut_slug}.fasta"

        vhh_seq = wt_vhh
        ag_seq = wt_ag
        if ab_mut_file.exists():
            recs = read_fasta_records(ab_mut_file)
            vhh_seq = _trim_vhh_tags(next(iter(recs.values())))
        if ag_mut_file.exists():
            recs = read_fasta_records(ag_mut_file)
            ag_seq = trim_antigen_tags(next(iter(recs.values())))

        kd_nm = float(mut_val) * 1e9
        rows.append(
            {
                "Ab_name": f"SKEMPIVHH-{nb_name}-{mut_slug}",
                "WT_Abname": nb_name,
                "WT_PDB": pdb4,
                "Ag_name": "EGFR",
                "Ag_type": "Non-Viral",
                "raw_value": kd_nm,
                "binding_score": float(kd_nm_to_pkd(kd_nm)),
                "Ab_heavy_chain_seq": vhh_seq,
                "Ab_light_chain_seq": "",
                "Ag_seq": ag_seq,
                "Ag_epitope_restrictions": "None",
            }
        )

    return finalize_source_table(
        pd.DataFrame(rows),
        dataset_name="SKEMPIv2_VHH",
        modality="KD",
        raw_unit="nM",
        require_light_chain=False,
    )


def curate_adaptyv_egfr_vhh(paths: RepoPaths) -> pd.DataFrame:
    """Curate Adaptyv Bio EGFR Round 1 & Round 2 de novo competition single-domain VHH binders.

    Strictly selects true single-domain immunoglobulin VHHs (`95 <= len <= 165`, no `G4S` scFv linker,
    canonical `C...W[VFIY]RQ...Y[YF]C...WG[QKR]GT` VHH motif) and excludes clamped non-binder ceiling
    entries (`kd >= 9.99e-6 M`).
    """
    catalog = load_antigen_catalog(paths.metadata)
    _, egfr_seq = catalog["EGFR"]
    egfr_dir = paths.raw / "Adaptyv-EGFR"

    rows = []
    for rnd, fname in (
        ("R1", "egfr_round1_result_summary.csv"),
        ("R2", "egfr_round2_result_summary.csv"),
    ):
        df = pd.read_csv(egfr_dir / fname)
        kd_m = pd.to_numeric(df["kd"], errors="coerce")
        sub = df[kd_m.notna() & (kd_m > 0) & (kd_m < 9.99e-6)].copy()
        for _, r in sub.iterrows():
            raw_seq = clean_seq(r["sequence"])
            if "GGGGSGGGGS" in raw_seq or not (95 <= len(raw_seq) <= 165):
                continue
            if not re.search(r"C.{10,25}W[VFIY]RQ.{30,60}Y[YF]C.{3,30}WG[QKR]GT", raw_seq):
                continue
            vhh = _trim_vhh_tags(raw_seq)
            kd_nm = float(r["kd"]) * 1e9
            nm = str(r["name"]).strip()
            rows.append(
                {
                    "Ab_name": f"AdaptyvEGFR_{rnd}-{nm}",
                    "WT_Abname": f"AdaptyvEGFR_{rnd}",
                    "WT_PDB": "NA",
                    "Ag_name": "EGFR",
                    "Ag_type": "Non-Viral",
                    "raw_value": kd_nm,
                    "binding_score": float(kd_nm_to_pkd(kd_nm)),
                    "Ab_heavy_chain_seq": vhh,
                    "Ab_light_chain_seq": "",
                    "Ag_seq": egfr_seq,
                    "Ag_epitope_restrictions": "None",
                }
            )

    return finalize_source_table(
        pd.DataFrame(rows),
        dataset_name="Adaptyv_EGFR_VHH",
        modality="KD",
        raw_unit="nM",
        require_light_chain=False,
    )


def curate_bennett_vhh(paths: RepoPaths) -> pd.DataFrame:
    """Curate Bennett et al. (Nature 2026) cryo-EM & SPR validated RFdiffusion `VHH_flu_01` (`PDB: 9NH7`).

    `VHH_flu_01` binds pandemic Influenza A/USA:Iowa/1943 (H1N1) HA stem at `Kd = 78 nM` (SPR kinetic fit).
    """
    recs = read_fasta_records(paths.raw / "Bennett-RFantibody" / "9NH7.fasta")
    vhh_seq = _trim_vhh_tags(recs["9NH7_1"])
    ha1_seq = trim_antigen_tags(recs["9NH7_2"])
    ha2_seq = trim_antigen_tags(recs["9NH7_3"])
    # Trim C-terminal T4 foldon + Avi-tag fusion (`GSGYIPEAPRDGQAYVRKDGEWVLLSTFL...`) on recombinant HA2
    foldon_idx = ha2_seq.find("GSGYIPEAPRDGQAYVRKDGEWVLLSTFL")
    if foldon_idx > 50:
        ha2_seq = ha2_seq[:foldon_idx]
    ag_seq = ha1_seq + ha2_seq

    rows = [
        {
            "Ab_name": "BennettVHH-VHH_flu_01",
            "WT_Abname": "NbBCII10_RFdiffusion",
            "WT_PDB": "9NH7",
            "Ag_name": "Influenza_AH1",
            "Ag_type": "Viral",
            "raw_value": 78.0,
            "binding_score": float(kd_nm_to_pkd(78.0)),
            "Ab_heavy_chain_seq": vhh_seq,
            "Ab_light_chain_seq": "",
            "Ag_seq": ag_seq,
            "Ag_epitope_restrictions": "HA_stem",
        }
    ]
    return finalize_source_table(
        pd.DataFrame(rows),
        dataset_name="Bennett_VHH",
        modality="KD",
        raw_unit="nM",
        require_light_chain=False,
    )
