#!/usr/bin/env python3
"""Step 15: Conserved-Anchor CDR vs. Framework Stratification & Naive 1D/3D Hot-Spot
Impact Baselines across the 22-Cluster `LOACO-Local` Benchmark.

Tests two mechanistic questions about hot-spot mutation detection (`Impact AUC`: `±1` vs. `0`):
  1. Are antibody mutations in `LOACO-Local` (`12,267` `L1–L3` pairs) restricted to CDRs or
     spread across CDRs, Framework (`FR1–FR4`), and Antigen (`Ag-Mut`), and does a naive
     `1D CDR vs. Framework (Yes/No)` rule explain hot-spot detection?
  2. Within `CDR-Only` variants (where `1D CDR Yes/No` is identically `0.500`), how much signal
     comes from knowing **single-WT-complex 3D interface contact geometry** (`-min_dist`,
     mutated-site `ΔSASA`) vs. dynamic normal-mode `ANTIPASTI` and complex inverse folding,
     and what happens when the Bound WT Co-Crystal is replaced by Unbound `IgFold`?

Outputs:
  - `results/cdr_contact_hotspot_analysis.json`
  - `results/CDR_CONTACT_HOTSPOT_ANALYSIS.md`
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from abrank.baselines.models import assemble_split_tensors, compute_stratum_weights
from abrank.baselines.structure_models import ANTIPASTI3StateTrainer, assemble_structure_pair_features
from abrank.config import get_repo_root
from abrank.evaluate import compute_3state_metrics, compute_cluster_bootstrap_summary


def find_ab_cdr_mask(seq: str, is_heavy: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Identify CDR1, CDR2, and CDR3 residue masks on a VH/VHH or VL sequence via conserved
    immunoglobulin framework anchors (`Cys1`, `Trp-FR2`, `Cys2`, and `WG.G`/`FG.G` J-motif),
    with safe Chothia/IMGT positional fallback if an engineered sequence lacks a canonical Cys.
    Returns `(is_cdr_mask, is_cdr3_mask)` of shape `(L,)`.
    """
    L = len(seq)
    is_cdr = np.zeros(L, dtype=bool)
    is_cdr3 = np.zeros(L, dtype=bool)
    if L == 0:
        return is_cdr, is_cdr3

    # Anchor 1: First conserved framework cysteine (~pos 18-26, 0-indexed)
    c1_candidates = [i for i in range(min(15, L), min(28, L)) if seq[i] == "C"]
    c1 = min(c1_candidates, key=lambda i: abs(i - 21)) if c1_candidates else (21 if is_heavy else 22)

    # Anchor 2: Conserved FR2 tryptophan (~11-19 residues after Cys1, typically WVRQ / WYQQ)
    w_candidates = [i for i in range(c1 + 10, min(c1 + 22, L)) if seq[i] == "W"]
    w2 = w_candidates[0] if w_candidates else min(c1 + (14 if is_heavy else 12), L - 1)

    # CDR1 lies between (c1 + 3) and (w2 - 2) inclusive (IMGT/Chothia union)
    cdr1_start = min(c1 + 3, L)
    cdr1_end = max(cdr1_start, min(w2 - 1, L))
    is_cdr[cdr1_start:cdr1_end] = True

    # CDR2 starts ~14 residues after FR2 Trp (just after LEWMG / LLIY)
    cdr2_start = min(w2 + (14 if is_heavy else 14), L)
    cdr2_end = min(cdr2_start + (17 if is_heavy else 9), L)
    is_cdr[cdr2_start:cdr2_end] = True

    # Anchor 3: Second conserved cysteine before CDR3 (~pos 84-102)
    c2_candidates = [i for i in range(min(82, L), min(104, L)) if seq[i] == "C"]
    c2 = min(c2_candidates, key=lambda i: abs(i - (95 if is_heavy else 87))) if c2_candidates else min(95 if is_heavy else 87, L - 1)

    # Anchor 4: J-region WG.G (heavy) or FG.G (light) motif after CDR3
    j_pat = re.compile(r"[WF]G.G")
    m_j = j_pat.search(seq, pos=min(c2 + 3, L))
    j_idx = m_j.start() if m_j else max(c2 + 4, L - 11)

    cdr3_start = min(c2 + 1, L)
    cdr3_end = max(cdr3_start, min(j_idx, L))
    is_cdr[cdr3_start:cdr3_end] = True
    is_cdr3[cdr3_start:cdr3_end] = True
    return is_cdr, is_cdr3


def main() -> None:
    t0 = time.time()
    root = get_repo_root()
    bdir = root / "data" / "benchmark"
    edir = root / "data" / "embeddings"
    sdir = root / "data" / "structures"
    res_dir = root / "results"

    kd_raw = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False)
    kd_lookup = kd_raw.drop_duplicates("AbAgID").set_index("AbAgID")
    kd_lookup["Ab_light_chain_seq"] = kd_lookup["Ab_light_chain_seq"].fillna("")

    sp_parts = [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")]
    loc_df = pd.concat(sp_parts, ignore_index=True)
    loc_df = loc_df[loc_df["Category"].isin(["L1", "L2", "L3"])].reset_index(drop=True)
    n_loc = len(loc_df)
    loc_c1 = loc_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
    ab_lin_1 = loc_df["AbAgID_1"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()
    ab_lin_2 = loc_df["AbAgID_2"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()

    cc_bound = torch.load(sdir / "complex_cache.pt", map_location="cpu", weights_only=False)
    cc_unb_abl = torch.load(sdir / "unbound_cocrystal_ablation.pt", map_location="cpu", weights_only=False)
    unb_scafs_override = cc_unb_abl["unb_scaffolds"]

    abag2sc = cc_bound["abag_to_scaffold"]
    scafs_bound = cc_bound["scaffolds"]

    # Precompute CDR masks per unique VH and VL
    vh_masks = {s: find_ab_cdr_mask(s, is_heavy=True) for s in kd_lookup["Ab_heavy_chain_seq"].unique()}
    vl_masks = {s: find_ab_cdr_mask(s, is_heavy=False) for s in kd_lookup["Ab_light_chain_seq"].unique()}

    regions = []
    q_cdr_yesno = np.zeros(n_loc, dtype=float)
    q_cdr_frac = np.zeros(n_loc, dtype=float)
    q_cdrh3_prio = np.zeros(n_loc, dtype=float)
    q_wt_dist_bound = np.zeros(n_loc, dtype=float)
    q_wt_sasa_bound = np.zeros(n_loc, dtype=float)
    q_wt_dist_unb = np.zeros(n_loc, dtype=float)
    q_wt_sasa_unb = np.zeros(n_loc, dtype=float)

    for i in range(n_loc):
        r = loc_df.iloc[i]
        a, b = kd_lookup.loc[r["AbAgID_1"]], kd_lookup.loc[r["AbAgID_2"]]
        h1, h2 = str(a["Ab_heavy_chain_seq"]), str(b["Ab_heavy_chain_seq"])
        l1, l2 = str(a["Ab_light_chain_seq"]), str(b["Ab_light_chain_seq"])
        g1, g2 = str(a["Ag_seq"]), str(b["Ag_seq"])

        sc_id = abag2sc.get(r["AbAgID_1"], "")
        sc_b = scafs_bound.get(sc_id)
        sc_u = unb_scafs_override.get(sc_id, sc_b)

        n_cdr, n_fw, n_h3, n_ag = 0, 0, 0, 0
        min_db, sum_sb = 50.0, 0.0
        min_du, sum_su = 50.0, 0.0

        if len(h1) == len(h2) and h1 != h2:
            m_cdr, m_h3 = vh_masks[h1]
            for p, (c1, c2) in enumerate(zip(h1, h2)):
                if c1 != c2:
                    if m_cdr[p]:
                        n_cdr += 1
                        if m_h3[p]:
                            n_h3 += 1
                    else:
                        n_fw += 1
                    if sc_b is not None and p < len(sc_b["min_dist_H"]):
                        min_db = min(min_db, float(sc_b["min_dist_H"][p]))
                        sum_sb += float(sc_b["dsasa_H"][p])
                    if sc_u is not None and p < len(sc_u["min_dist_H"]):
                        min_du = min(min_du, float(sc_u["min_dist_H"][p]))
                        sum_su += float(sc_u["dsasa_H"][p])
        elif h1 != h2:
            n_cdr += 1

        if len(l1) == len(l2) and l1 != l2:
            m_cdr_l, _ = vl_masks[l1]
            for p, (c1, c2) in enumerate(zip(l1, l2)):
                if c1 != c2:
                    if m_cdr_l[p]:
                        n_cdr += 1
                    else:
                        n_fw += 1
                    if sc_b is not None and p < len(sc_b["min_dist_L"]):
                        min_db = min(min_db, float(sc_b["min_dist_L"][p]))
                        sum_sb += float(sc_b["dsasa_L"][p])
                    if sc_u is not None and p < len(sc_u["min_dist_L"]):
                        min_du = min(min_du, float(sc_u["min_dist_L"][p]))
                        sum_su += float(sc_u["dsasa_L"][p])
        elif l1 != l2:
            n_cdr += 1

        if g1 != g2:
            if len(g1) == len(g2):
                for p, (c1, c2) in enumerate(zip(g1, g2)):
                    if c1 != c2:
                        n_ag += 1
                        if sc_b is not None and p < len(sc_b["min_dist_A"]):
                            min_db = min(min_db, float(sc_b["min_dist_A"][p]))
                            sum_sb += float(sc_b["dsasa_A"][p])
                        if sc_u is not None and p < len(sc_u["min_dist_A"]):
                            min_du = min(min_du, float(sc_u["min_dist_A"][p]))
                            sum_su += float(sc_u["dsasa_A"][p])
            else:
                n_ag += 1

        if n_ag > 0 and (n_cdr + n_fw) == 0:
            reg = "Ag-Mut"
        elif n_cdr > 0 and n_fw == 0:
            reg = "CDR-Only"
        elif n_cdr > 0 and n_fw > 0:
            reg = "CDR+Framework"
        else:
            reg = "Framework-Only"
        regions.append(reg)

        tot_m = max(n_cdr + n_fw + n_ag, 1)
        q_cdr_yesno[i] = 1.0 if (n_cdr + n_ag) > 0 else 0.0
        q_cdr_frac[i] = (n_cdr + n_ag) / tot_m
        q_cdrh3_prio[i] = 1.0 if n_h3 > 0 else (0.5 if (n_cdr + n_ag) > 0 else 0.0)
        q_wt_dist_bound[i] = -min_db
        q_wt_sasa_bound[i] = sum_sb
        q_wt_dist_unb[i] = -min_du
        q_wt_sasa_unb[i] = sum_su

    loc_df["mut_region"] = regions

    # Load cached zero-shot 1D/3D impact scores and evaluate ANTIPASTI OOF impact across LOACO
    esmc_cache = torch.load(edir / "esmc_300m.pt", map_location="cpu", weights_only=False)
    ablang_cache = torch.load(edir / "ablang2_paired.pt", map_location="cpu", weights_only=False)
    esm2_cache = torch.load(edir / "esm2_150m.pt", map_location="cpu", weights_only=False)
    inv_cache = torch.load(edir / "inverse_folding_scaffolds.pt", map_location="cpu", weights_only=False)
    saprot_cache = torch.load(edir / "saprot_650m.pt", map_location="cpu", weights_only=False)
    mint_cache = torch.load(edir / "mint_650m.pt", map_location="cpu", weights_only=False)

    sp_all = assemble_split_tensors(loc_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)
    sf_bound = assemble_structure_pair_features(loc_df, kd_lookup, cc_bound, inv_cache, saprot_cache, mint_cache, True)

    cc_merged_unb = {"abag_to_scaffold": abag2sc, "scaffolds": {**scafs_bound, **unb_scafs_override}}
    inv_src = inv_cache["scaffolds"] if "scaffolds" in inv_cache else inv_cache
    inv_merged_unb = {"scaffolds": {**inv_src, **cc_unb_abl["unb_inv"]}} if "scaffolds" in inv_cache else {**inv_cache, **cc_unb_abl["unb_inv"]}
    sap_merged_unb = {"chains": {**saprot_cache["chains"], **cc_unb_abl["unb_sap_chains"]}, "scaf_rep_abag": saprot_cache["scaf_rep_abag"]}
    sf_unb = assemble_structure_pair_features(
        loc_df, kd_lookup, cc_merged_unb, inv_merged_unb, sap_merged_unb, mint_cache, True
    )

    with open(res_dir / "loaco_evaluation.json") as f:
        loaco_json = json.load(f)
    loc_clusters = sorted(loaco_json["cluster_metadata"].keys())
    cocrys_clusters = [c for c in loc_clusters if loaco_json["cluster_metadata"][c]["cocrystal_frac"] >= 0.5]

    # Train lightweight ANTIPASTI across the 22 LOACO clusters to get exact pair-level OOF impact on CDR-Only too
    anti_oof_b = np.zeros(n_loc, dtype=float)
    anti_oof_u = np.zeros(n_loc, dtype=float)
    for cl in loc_clusters:
        te_idx = np.flatnonzero(loc_c1 == cl)
        te_lins = set(ab_lin_1[te_idx]) | set(ab_lin_2[te_idx])
        pool = np.array([i for i in range(n_loc) if loc_c1[i] != cl and ab_lin_1[i] not in te_lins and ab_lin_2[i] not in te_lins], dtype=int)
        rem_cls = sorted(set(loc_c1[pool]))
        val_cls = set(rem_cls[1::5])
        val_lins = set(ab_lin_1[pool[np.isin(loc_c1[pool], list(val_cls))]])
        is_val = np.array([loc_c1[i] in val_cls for i in pool], dtype=bool)
        is_tr = np.array([loc_c1[i] not in val_cls and ab_lin_1[i] not in val_lins and ab_lin_2[i] not in val_lins for i in pool], dtype=bool)
        tr_i, val_i = pool[is_tr], pool[is_val]
        w_tr = compute_stratum_weights(loc_df.iloc[tr_i].reset_index(drop=True), sp_all.is_ag_mut[tr_i])
        m_anti = ANTIPASTI3StateTrainer(seed=42, epochs=15).fit(
            sf_bound.subset(tr_i), sp_all.subset(tr_i), sf_bound.subset(val_i), sp_all.subset(val_i),
            loc_df.iloc[val_i].reset_index(drop=True), w_tr,
        )
        _, anti_oof_b[te_idx] = m_anti.predict_3state(sf_bound.subset(te_idx), sp_all.subset(te_idx))
        _, anti_oof_u[te_idx] = m_anti.predict_3state(sf_unb.subset(te_idx), sp_all.subset(te_idx))

    loc_lbl = loc_df["label"].to_numpy(dtype=int)
    loc_bd = loc_df["binding_diff"].to_numpy(dtype=float)
    cdr_only_mask = (loc_df["mut_region"] == "CDR-Only").to_numpy()
    ab_mut_mask = (loc_df["mut_region"] != "Ag-Mut").to_numpy()

    methods = [
        ("Naive 1D CDR vs. Framework (Yes/No)", "1D Sequence Heuristic", q_cdr_yesno, q_cdr_yesno),
        ("Naive 1D CDR Fraction (#CDR / #Edits)", "1D Sequence Heuristic", q_cdr_frac, q_cdr_frac),
        ("Naive 1D CDR-H3 Priority", "1D Sequence Heuristic", q_cdrh3_prio, q_cdrh3_prio),
        ("ESMC+AbLang2 (1D PLM Zero-Shot)", "1D PLM Zero-Shot", sp_all.zs_impact_dict["esmc_ablang2_zeroshot"], sp_all.zs_impact_dict["esmc_ablang2_zeroshot"]),
        ("WT 3D Interface Distance (-min Cβ dist)", "Single-WT 3D Geometry", q_wt_dist_bound, q_wt_dist_unb),
        ("WT 3D Mutated-Site Buried ΔSASA", "Single-WT 3D Geometry", q_wt_sasa_bound, q_wt_sasa_unb),
        ("ProteinMPNN (WT 3D Complex ZS)", "Single-WT 3D Inv. Fold", sf_bound.zs_imp["protein_mpnn"], sf_unb.zs_imp["protein_mpnn"]),
        ("PRODIGY+SASA (WT 3D Energy ZS)", "Single-WT 3D Energy", sf_bound.zs_imp["prodigy_sasa"], sf_unb.zs_imp["prodigy_sasa"]),
        ("ESM-IF1 (WT 3D Complex ZS)", "Single-WT 3D Inv. Fold", sf_bound.zs_imp["esm_if1"], sf_unb.zs_imp["esm_if1"]),
        ("ANTIPASTI (WT 3D Normal-Mode CNN)", "Single-WT 3D Dynamics", anti_oof_b, anti_oof_u),
    ]

    def eval_impact_slice(q_vec: np.ndarray, sub_mask: np.ndarray, clust_list: list[str]) -> dict:
        per_cl = {}
        for cl in clust_list:
            ix = np.flatnonzero((loc_c1 == cl) & sub_mask)
            if len(ix) >= 5:
                per_cl[cl] = compute_3state_metrics(loc_lbl[ix], loc_bd[ix], np.zeros(len(ix)), q_vec[ix])
        cb = compute_cluster_bootstrap_summary(per_cl, min_class_pairs=2, n_bootstrap=2000, seed=42)
        return cb["impact"]

    region_stats = {}
    for reg, g in loc_df.groupby("mut_region"):
        region_stats[str(reg)] = {
            "n_pairs": int(len(g)),
            "pct_of_local": round(100.0 * len(g) / n_loc, 2),
            "impactful_rate_pct": round(100.0 * float((g["label"].abs() == 1).mean()), 2),
            "neutral_rate_pct": round(100.0 * float((g["label"] == 0).mean()), 2),
        }

    results_table = {}
    for name, fam, qb, qu in methods:
        results_table[name] = {
            "family": fam,
            "all_loaco_22_clusters": eval_impact_slice(qb, np.ones(n_loc, dtype=bool), loc_clusters),
            "ab_mut_only": eval_impact_slice(qb, ab_mut_mask, loc_clusters),
            "cdr_only_pairs": eval_impact_slice(qb, cdr_only_mask, loc_clusters),
            "same_cocrystal_clusters_bound_wt": eval_impact_slice(qb, np.ones(n_loc, dtype=bool), cocrys_clusters),
            "same_cocrystal_clusters_unbound_igfold": eval_impact_slice(qu, np.ones(n_loc, dtype=bool), cocrys_clusters),
        }

    payload = {
        "n_local_pairs": int(n_loc),
        "n_wt_cocrystal_scaffolds_in_loaco_local": 47,
        "mutation_region_breakdown": region_stats,
        "hotspot_impact_auc_benchmarks": results_table,
    }
    out_json = res_dir / "cdr_contact_hotspot_analysis.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = [
        "# CDR vs. Framework Stratification & Single-WT 3D Hot-Spot Impact Analysis",
        "",
        "## 1. Mutation-Region Breakdown Across `LOACO-Local` (`12,267` `L1–L3` Pairs)",
        "",
        "| Mutation Region | Pairs (`%` of Local) | Impactful (`|ΔpKd| ≥ 1.0`, `≥10×`) | Neutral (`|ΔpKd| ≤ 0.30`, `≤2×`) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for rk in ("CDR-Only", "CDR+Framework", "Framework-Only", "Ag-Mut"):
        if rk in region_stats:
            st = region_stats[rk]
            md.append(f"| **{rk}** | `{st['n_pairs']:,}` (`{st['pct_of_local']:.1f}%`) | `{st['impactful_rate_pct']:.1f}%` | `{st['neutral_rate_pct']:.1f}%` |")

    md.extend([
        "",
        "## 2. Hot-Spot Detection (`Impact AUC`: `±1` vs. `0`) — 1D CDR Heuristics vs. Single-WT 3D Geometry",
        "",
        "| Method | Family | All `LOACO` (`19` cl.) [95% CI] | `Ab-Mut` Only | `CDR-Only` Pairs | Bound WT Crystal (`18` cl.) | Unbound `IgFold` (`18` cl.) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for name, _, _, _ in methods:
        rt = results_table[name]
        a_all = rt["all_loaco_22_clusters"]
        a_ab = rt["ab_mut_only"]
        a_cdr = rt["cdr_only_pairs"]
        a_bnd = rt["same_cocrystal_clusters_bound_wt"]
        a_unb = rt["same_cocrystal_clusters_unbound_igfold"]
        ci = a_all["cluster_ci95"]
        md.append(
            f"| **{name}** | {rt['family']} | **`{a_all['mean']:.3f}`** `[{ci[0]:.2f}, {ci[1]:.2f}]` | "
            f"`{a_ab['mean']:.3f}` | **`{a_cdr['mean']:.3f}`** | `{a_bnd['mean']:.3f}` | `{a_unb['mean']:.3f}` |"
        )
    out_md = res_dir / "CDR_CONTACT_HOTSPOT_ANALYSIS.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Completed in {time.time() - t0:.1f}s! Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
