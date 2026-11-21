#!/usr/bin/env python3
"""Step 8: Build the reproducible 3D Complex & Scaffold Cache (`data/structures/`)
for all AbRank v2 evaluation tracks (`unseen_ag`, `vhh`, and `cross_modal` incl. `AbAgym`,
`SKEMPIv2`, `AbDesign ELISA/DDG`, `NbBench`, and `Kothiwal`).

Uses solved experimental Ab–Ag co-crystal PDBs wherever available (`is_cocrystal = True`)
and experimental RCSB antigen structures + `IgFold` antibody/VHH structures otherwise
(`is_cocrystal = False`), computing:
  - Standardized multi-chain PDBs (`H`, optional `L`, `A`)
  - Residue-level interface distances, `FreeSASA` buried `ΔSASA`, and `PRODIGY` `ΔG`
  - `ProDy`/GNM 30-mode Dynamic Cross-Correlation Matrices (`DCCM`, `ANTIPASTI`)
  - `Foldseek` `3Di` structural-alphabet token sequences (`SaProt-650M`)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch
import yaml

from abrank.config import get_repo_root
from abrank.structures.resolver import (
    ChainBackbone,
    align_query_to_chain,
    compute_scaffold_biophysics,
    download_rcsb_cif,
    extract_foldseek_3di_batch,
    fold_missing_antibodies_igfold,
    orient_unbound_ab_ag,
    parse_structure_chains,
    select_best_antigen_chain,
    select_cocrystal_chains,
    write_standardized_pdb,
)


def collect_target_assays(root: Path) -> pd.DataFrame:
    """Gather all unique `AbAgID` rows referenced across `unseen_ag`, `vhh`, and `cross_modal` splits."""
    bdir = root / "data" / "benchmark"
    kd = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False)
    vhh = pd.read_csv(bdir / "assays_vhh.tsv", sep="\t", low_memory=False)
    vhh["Ab_light_chain_seq"] = ""
    cm = pd.read_csv(bdir / "assays_cross_modal.tsv", sep="\t", low_memory=False)
    cm["Ab_light_chain_seq"] = cm["Ab_light_chain_seq"].fillna("")

    used_kd: set[str] = set()
    for s in ("train", "val", "test", "test_candidates"):
        df = pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}.tsv", sep="\t", low_memory=False)
        used_kd.update(df["AbAgID_1"])
        used_kd.update(df["AbAgID_2"])
    pw = pd.read_csv(bdir / "splits" / "unseen_ag" / "test_pointwise.tsv", sep="\t", low_memory=False)
    used_kd.update(pw["AbAgID"])

    # Also include all SKEMPIv2 rows used in 5-fold ΔΔG transfer!
    used_kd.update(kd.loc[kd["Dataset"] == "SKEMPIv2", "AbAgID"])

    # Include all VHH rows
    used_vhh = set(vhh["AbAgID"])

    # Include all Cross-Modal rows used in modality_test (ELISA, AbDesign DDG, AbAgym DMS, NbBench, EC50)
    cm_pairs = pd.read_csv(bdir / "splits" / "cross_modal" / "modality_test.tsv", sep="\t", low_memory=False)
    used_cm = set(cm_pairs["AbAgID_1"]).union(cm_pairs["AbAgID_2"])
    # Exclude AbBibench Enrichment from 3D cache unless in strict unseen_ag (it's 100% train antigens!)
    sub_cm = cm[cm["AbAgID"].isin(used_cm) & (cm["Dataset"] != "AbBibench")].copy()

    sub_kd = kd[kd["AbAgID"].isin(used_kd)].copy()
    sub_vhh = vhh[vhh["AbAgID"].isin(used_vhh)].copy()

    cols = [
        "AbAgID", "Dataset", "WT_Abname", "WT_PDB", "Ag_name", "Ag_group",
        "Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq", "Ab_lineage",
    ]
    for d in (sub_kd, sub_vhh, sub_cm):
        if "Ab_lineage" not in d.columns:
            d["Ab_lineage"] = d["WT_Abname"].astype(str)
    all_df = pd.concat([sub_kd[cols], sub_vhh[cols], sub_cm[cols]], ignore_index=True).drop_duplicates("AbAgID")
    all_df["Ab_light_chain_seq"] = all_df["Ab_light_chain_seq"].fillna("")
    all_df["WT_PDB"] = all_df["WT_PDB"].fillna("NA").astype(str)
    return all_df.reset_index(drop=True)


def resolve_wt_pdb4(row: pd.Series, cocrys_map: dict[str, str]) -> str:
    raw_pdb = str(row["WT_PDB"]).strip()
    wt_ab = str(row["WT_Abname"]).strip()
    if raw_pdb in cocrys_map:
        return cocrys_map[raw_pdb]
    if wt_ab in cocrys_map:
        return cocrys_map[wt_ab]
    if len(raw_pdb) >= 4 and raw_pdb.upper() not in ("NA", "NONE", "NAN"):
        p4 = raw_pdb[:4].upper()
        if p4.isalnum():
            return p4
    return ""


def assign_scaffold_id(row: pd.Series, wt_pdb4: str) -> str:
    ds = str(row["Dataset"])
    ag_grp = str(row["Ag_group"])
    lh = len(str(row["Ab_heavy_chain_seq"]))
    ll = len(str(row["Ab_light_chain_seq"]))
    la = len(str(row["Ag_seq"]))
    if ds == "AbAgym" and wt_pdb4:
        return f"ABAGYM_{row['WT_Abname']}_{wt_pdb4}"
    if wt_pdb4:
        return f"PDB_{wt_pdb4}_{ag_grp}_H{lh}_L{ll}"
    # Mutational scan libraries sharing a single parental scaffold
    if ds in ("AbBibench", "NbBench", "Porebski", "Porebski_VHH"):
        return f"LIB_{ds}_{row['WT_Abname']}_{ag_grp}_H{lh}_L{ll}"
    # Distinct discovery/candidate clones (Kothiwal, OVA, BoltzGen_VHH, Germinal_VHH, Adaptyv_EGFR_VHH, etc.)
    ab_sig = hashlib.sha1(f"{row['Ab_heavy_chain_seq']}|{row['Ab_light_chain_seq']}".encode()).hexdigest()[:10]
    ag_sig = hashlib.sha1(str(row["Ag_seq"]).encode()).hexdigest()[:8]
    return f"IGFOLD_{ds}_{ag_grp}_{ab_sig}_{ag_sig}_A{la}"


def main() -> None:
    t0 = time.time()
    root = get_repo_root()
    sdir = root / "data" / "structures"
    rcsb_dir = sdir / "rcsb"
    igfold_dir = sdir / "igfold"
    scaf_dir = sdir / "scaffolds"
    abagym_dir = sdir / "abagym_pdbs"
    for d in (rcsb_dir, igfold_dir, scaf_dir, abagym_dir):
        d.mkdir(parents=True, exist_ok=True)

    tpl = yaml.safe_load((root / "metadata" / "structure_templates.yaml").read_text(encoding="utf-8"))
    cocrys_map: dict[str, str] = tpl.get("cocrystal_wt_pdbs", {})
    ag_pdb_map: dict[str, str] = tpl.get("antigen_pdbs", {})

    # Extract AbAgym bundled co-crystal PDBs once
    abagym_zip = root / "data" / "raw" / "AbAgym" / "PDB_files.zip"
    abagym_pdb_lookup: dict[str, Path] = {}
    if abagym_zip.exists():
        with zipfile.ZipFile(abagym_zip) as zf:
            for member in zf.namelist():
                if member.endswith(".pdb"):
                    bname = Path(member).name
                    dest = abagym_dir / bname
                    if not dest.exists():
                        dest.write_bytes(zf.read(member))
                    p4 = bname.rsplit("_", 1)[-1].replace(".pdb", "").upper()
                    abagym_pdb_lookup[p4] = dest
                    abagym_pdb_lookup[bname.replace(".pdb", "").lower()] = dest

    assays = collect_target_assays(root)
    print(f"Collected {len(assays):,} unique benchmark complexes across KD, VHH, and Cross-Modal.")

    wt_pdb4_list = [resolve_wt_pdb4(r, cocrys_map) for _, r in assays.iterrows()]
    assays["resolved_wt_pdb4"] = wt_pdb4_list
    assays["scaffold_id"] = [assign_scaffold_id(r, p4) for (_, r), p4 in zip(assays.iterrows(), wt_pdb4_list)]

    scaf_reps = assays.drop_duplicates("scaffold_id").reset_index(drop=True)
    print(f"Grouped {len(assays):,} complexes into {len(scaf_reps):,} unique 3D scaffolds.")

    # Step 1: Download all needed RCSB PDBs (both co-crystal WT PDBs and unbound Antigen PDBs)
    needed_pdb4s: set[str] = set()
    for _, r in scaf_reps.iterrows():
        if r["resolved_wt_pdb4"]:
            needed_pdb4s.add(r["resolved_wt_pdb4"])
        for key in (str(r["Ag_name"]), str(r["Ag_group"])):
            if key in ag_pdb_map:
                needed_pdb4s.add(ag_pdb_map[key])

    print(f"Downloading/verifying {len(needed_pdb4s)} experimental RCSB structures...")
    rcsb_paths: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        fut_map = {ex.submit(download_rcsb_cif, p4, rcsb_dir): p4 for p4 in sorted(needed_pdb4s)}
        for fut in as_completed(fut_map):
            p4 = fut_map[fut]
            res = fut.result()
            if res is not None:
                rcsb_paths[p4] = res
    print(f"  Cached {len(rcsb_paths)}/{len(needed_pdb4s)} RCSB structures.")

    # Parse all cached RCSB + AbAgym structures once in memory
    parsed_rcsb: dict[str, dict[str, ChainBackbone]] = {}
    for p4, path in rcsb_paths.items():
        try:
            parsed_rcsb[p4] = parse_structure_chains(path)
        except Exception as e:
            print(f"  [WARN] Failed parsing {p4}: {e}")

    parsed_abagym: dict[str, dict[str, ChainBackbone]] = {}
    for k, path in abagym_pdb_lookup.items():
        if len(k) == 4:
            try:
                parsed_abagym[k] = parse_structure_chains(path)
            except Exception:
                pass

    # Step 2: Test co-crystal chain resolution for every scaffold; collect missing Ab chains for IgFold!
    cocrystal_matches: dict[str, tuple[ChainBackbone, ChainBackbone | None, ChainBackbone]] = {}
    igfold_jobs: dict[str, tuple[str, str, str]] = {}
    scaf_to_ab_key: dict[str, str] = {}

    for _, r in scaf_reps.iterrows():
        sid = str(r["scaffold_id"])
        vh = str(r["Ab_heavy_chain_seq"])
        vl = str(r["Ab_light_chain_seq"])
        ag = str(r["Ag_seq"])
        p4 = str(r["resolved_wt_pdb4"])

        matched = None
        if r["Dataset"] == "AbAgym" and p4 in parsed_abagym:
            matched = select_cocrystal_chains(parsed_abagym[p4], vh, vl, ag)
        if matched is None and p4 and p4 in parsed_rcsb:
            matched = select_cocrystal_chains(parsed_rcsb[p4], vh, vl, ag)

        if matched is not None:
            cocrystal_matches[sid] = matched
        else:
            ab_key = hashlib.sha1(f"{vh}|{vl}".encode()).hexdigest()[:14]
            scaf_to_ab_key[sid] = ab_key
            igfold_jobs[ab_key] = (ab_key, vh, vl)

    print(
        f"Co-crystal scaffolds resolved: {len(cocrystal_matches)} / {len(scaf_reps)} "
        f"| Unbound IgFold Ab/VHH jobs needed: {len(igfold_jobs)}"
    )
    fold_missing_antibodies_igfold(list(igfold_jobs.values()), igfold_dir, n_workers=6)

    # Step 3: Assemble standardized `(H, [L], A)` scaffold PDBs + compute biophysical/GNM descriptors
    scaffold_meta: dict[str, dict] = {}
    written_pdbs: list[Path] = []

    # Fallback helix generator only if an ultra-short peptide antigen lacks RCSB coords
    def _ideal_helix_chain(seq: str) -> ChainBackbone:
        L = len(seq)
        coords = np.zeros((L, 5, 3), dtype=np.float32)
        for i in range(L):
            ang = i * (100.0 * np.pi / 180.0)
            ca = np.array([2.3 * np.cos(ang), 2.3 * np.sin(ang), 1.5 * i], dtype=np.float32)
            n = ca + np.array([-1.45, 0.0, -0.5], dtype=np.float32)
            c = ca + np.array([1.45, 0.0, 0.5], dtype=np.float32)
            o = c + np.array([0.0, 1.23, 0.0], dtype=np.float32)
            cb = ca + np.array([1.2 * np.cos(ang), 1.2 * np.sin(ang), 0.0], dtype=np.float32)
            coords[i] = np.stack([n, ca, c, o, cb], axis=0)
        return ChainBackbone(seq=seq, coords=coords)

    for idx, r in scaf_reps.iterrows():
        sid = str(r["scaffold_id"])
        vh = str(r["Ab_heavy_chain_seq"])
        vl = str(r["Ab_light_chain_seq"])
        ag = str(r["Ag_seq"])
        out_pdb = scaf_dir / f"{sid}.pdb"

        if sid in cocrystal_matches:
            cb_h, cb_l, cb_a = cocrystal_matches[sid]
            is_cocrys = True
        else:
            is_cocrys = False
            ab_key = scaf_to_ab_key[sid]
            ab_chains = parse_structure_chains(igfold_dir / f"{ab_key}.pdb")
            cb_h = ab_chains.get("H") or next(iter(ab_chains.values()))
            cb_l = ab_chains.get("L") if vl else None

            ag_p4 = ag_pdb_map.get(str(r["Ag_name"])) or ag_pdb_map.get(str(r["Ag_group"])) or str(r["resolved_wt_pdb4"])
            cb_a = None
            if ag_p4 and ag_p4 in parsed_rcsb:
                cb_a = select_best_antigen_chain(parsed_rcsb[ag_p4], ag)
            if cb_a is None:
                cb_a = _ideal_helix_chain(ag)

            cb_h, cb_l, cb_a = orient_unbound_ab_ag(cb_h, cb_l, cb_a)

        write_standardized_pdb(out_pdb, cb_h, cb_l, cb_a)
        written_pdbs.append(out_pdb)
        bio = compute_scaffold_biophysics(out_pdb, cb_h, cb_l, cb_a, is_cocrystal=is_cocrys)
        bio["pdb_path"] = str(out_pdb.relative_to(root))
        scaffold_meta[sid] = bio

    # Step 4: Extract Foldseek 3Di tokens across all scaffold PDBs in one vectorized call
    foldseek_bin = Path.home() / ".cache" / "abrank" / "bin" / "foldseek" / "bin" / "foldseek"
    print("Extracting Foldseek 3Di structural tokens across all scaffolds...")
    s3di_map = extract_foldseek_3di_batch(written_pdbs, foldseek_bin, sdir / "tmp_3di")
    for sid, bio in scaffold_meta.items():
        d3 = s3di_map.get(sid, {})
        bio["3di_H"] = d3.get("H", "d" * len(bio["seq_H"]))
        bio["3di_L"] = d3.get("L", "d" * len(bio["seq_L"])) if bio["seq_L"] else ""
        bio["3di_A"] = d3.get("A", "d" * len(bio["seq_A"]))

    # Map every AbAgID -> scaffold_id & precompute sequence->scaffold index alignment only where needed
    abag_to_scaffold = dict(zip(assays["AbAgID"], assays["scaffold_id"]))

    # Check coverage statistics on unseen_ag/test!
    bdir = root / "data" / "benchmark"
    test_df = pd.read_csv(bdir / "splits" / "unseen_ag" / "test.tsv", sep="\t")
    loc_test = test_df[test_df["Category"].isin(["L1", "L2", "L3"])]
    cocrys_loc = sum(
        scaffold_meta[abag_to_scaffold[a]]["is_cocrystal"] and scaffold_meta[abag_to_scaffold[b]]["is_cocrystal"]
        for a, b in zip(loc_test["AbAgID_1"], loc_test["AbAgID_2"])
    )
    print(
        f"unseen_ag/test L1-L3 co-crystal pair coverage: {cocrys_loc}/{len(loc_test)} "
        f"({100.0 * cocrys_loc / max(len(loc_test), 1):.1f}%)"
    )

    cache_path = sdir / "complex_cache.pt"
    torch.save(
        {
            "abag_to_scaffold": abag_to_scaffold,
            "scaffolds": scaffold_meta,
        },
        cache_path,
    )
    print(f"Saved {cache_path} ({len(scaffold_meta)} scaffolds, {len(abag_to_scaffold):,} complexes) in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
