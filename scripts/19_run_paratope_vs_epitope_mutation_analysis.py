#!/usr/bin/env python3
"""Step 19: Paratope (`Ab-Mut`) vs. Epitope (`Ag-Mut`) Mutation Stratification &
Two-Sided Generalization Benchmark across AbRank v2.

Addresses three core mechanistic questions on antibody-side vs. antigen-side mutations:
  1. Inventory: Across all four AbRank tracks (`Core KD`, `Cross-Modal`, `Viral Transfer`,
     and `Single-Domain VHH`), how many pairs and unique variants mutate the antibody
     paratope/framework (`Ab-Mut`: ΔAb > 0, ΔAg = 0), the antigen epitope (`Ag-Mut`:
     ΔAb = 0, ΔAg > 0), or both (`Global` cross-complex pairs)?
  2. Architectural Scope & Unseen-Antigen Generalization: How do antibody-specialized
     models (`AbLang2-Paired`, `AntiFold`) vs. symmetric two-sided complex models
     (`ESMC-300M`, `PRODIGY+SASA`, `ProteinMPNN`, `ESM-IF1`, `ANTIPASTI`, `GearBind`,
     `WALLE-Affinity`, `LogiCA`, and `1D/3D L2-SP`) perform on `Ab-Mut` vs.
     `Ag-Mut` across `LOACO-Local` (`12,267` pairs), `SKEMPI 2.0` (`2,419` pairs), and
     `AbAgym Interface DMS` (`16,754` pairs), including a matched 5-cluster co-crystal
     control holding target identity and bound structure availability fixed?
  3. Evolutionary Escape Inversion & Adaptation: Why do unsupervised PLM log-likelihoods
     correlate positively with binding on physiological `Ag-Mut` (`SKEMPI 2.0`) yet invert
     below chance (`< 0.50` Sign AUC) on viral `Ag-Mut` (`HIV-1 IC50`, `SARS-CoV-2 RBD Escape`),
     and how does `KD -> Target L2-SP` sample efficiency differ between `Ab-Mut` and `Ag-Mut`?

Outputs:
  - `results/paratope_vs_epitope_mutations.json`
  - `results/PARATOPE_VS_EPITOPE_MUTATIONS.md`
  - Enriches `results/structure_dms_ddg_benchmarks.json` (`slices["Ab-Mut"]` / `slices["Ag-Mut"]`)
"""

from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import gc
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import Levenshtein
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
torch.set_num_threads(1)

from abrank.baselines.embeddings import seq_hash
from abrank.baselines.features import build_complex_handcrafted_table
from abrank.baselines.logica_models import (
    LogiCA3StateTrainer,
)
from abrank.baselines.models import (
    AntisymmetricLinearRanker,
    NeuralPairwiseTrainer,
    OrdinalHurdleRanker,
    SplitPairTensors,
    SymmetricImpactClassifier,
    TwoHeadRegimeRanker,
    assemble_split_tensors,
    build_odd_linear_matrix,
    compute_stratum_weights,
)
from abrank.baselines.structure_models import (
    ANTIPASTI3StateTrainer,
    GearBind3StateTrainer,
    StructureSplitFeatures,
    assemble_structure_pair_features,
    fit_3state_foundation_hurdle,
    fit_walle_affinity_3state,
)
from abrank.baselines.transfer_finetune import (
    KDResidualAdapterRanker,
    KDSourceRanker,
    L2SPFineTunedRanker,
    TransferPairTensors,
    assemble_transfer_tensors,
    build_transfer_odd_matrix,
    fit_kd_source_ranker,
)
from abrank.config import get_repo_root
from abrank.evaluate import (
    compute_3state_metrics,
    compute_cluster_bootstrap_summary,
    compute_per_antigen_macro_metrics,
)
from abrank.pairs import build_within_antigen_pairs
from abrank.splits import assign_antibody_lineages


LOACO_MODEL_SPECS = [
    ("Random Baseline", "Naive"),
    ("WT 3D Contact (-min d_Cb)", "Single-WT 3D Geometry"),
    ("Biophys+AAC (Linear)", "Classical 1D"),
    ("k-mer Dipeptide (Ridge)", "Classical 1D"),
    ("AbLang2-Paired (1D ZS)", "Zero-Shot 1D (Ab-Only)"),
    ("ESMC-300M (1D ZS)", "Zero-Shot 1D (Complex)"),
    ("ESMC+AbLang2 (1D ZS)", "Zero-Shot 1D (Ensemble)"),
    ("PRODIGY+SASA (3D Energy ZS)", "Zero-Shot 3D (Complex)"),
    ("ProteinMPNN (3D Complex ZS)", "Zero-Shot 3D (Complex)"),
    ("AntiFold (3D Ab-Ag ZS)", "Zero-Shot 3D (Ab-Focused)"),
    ("ESM-IF1 (3D Complex ZS)", "Zero-Shot 3D (Complex)"),
    ("SaProt-650M (3D-Token ZS)", "Zero-Shot 3D (Complex)"),
    ("MINT-650M (Multi-Chain ZS)", "Zero-Shot Multi-Chain"),
    ("ESMC-300M (Multi-Scale Binary)", "Impact-Blind 1D"),
    ("ESMC-300M (3-State Hurdle)", "3-State 1D PLM"),
    ("ESMC+AbLang2+LLR (Bilinear)", "3-State 1D PLM"),
    ("LogiCA (1D Logit Cross-Attn)", "1D Logit Cross-Attention"),
    ("SaProt-650M (3-State Hurdle)", "3-State 3D / PPI"),
    ("MINT-650M (3-State Hurdle)", "3-State 3D / PPI"),
    ("ANTIPASTI (Normal-Mode 3D CNN)", "3-State 3D / PPI"),
    ("GearBind (3D Geometric GNN)", "3-State 3D / PPI"),
    ("WALLE-Affinity (3-State 3D Graph)", "3-State 3D / PPI"),
]


_G: dict[str, Any] = {}


def _slice_gb(gb_t: tuple, ix: np.ndarray) -> tuple:
    (a1, e1, m1), (a2, e2, m2) = gb_t
    return ((a1[ix], e1[ix], m1[ix]), (a2[ix], e2[ix], m2[ix]))


def augment_tp_with_3d(tp: TransferPairTensors, sf: StructureSplitFeatures) -> TransferPairTensors:
    return TransferPairTensors(
        y=tp.y,
        labels=tp.labels,
        is_impactful=tp.is_impactful,
        bdiff=tp.bdiff,
        is_ab_mut=tp.is_ab_mut,
        is_ag_mut=tp.is_ag_mut,
        is_same_ag=tp.is_same_ag,
        esmc_ab_1=np.concatenate([tp.esmc_ab_1, sf.mpnn_emb_1], axis=1),
        esmc_ab_2=np.concatenate([tp.esmc_ab_2, sf.mpnn_emb_2], axis=1),
        esmc_ag_1=tp.esmc_ag_1,
        esmc_ag_2=tp.esmc_ag_2,
        ablang_1=tp.ablang_1,
        ablang_2=tp.ablang_2,
        zs_features=np.concatenate([tp.zs_features, sf.struct_odd], axis=1),
        biophys_odd=tp.biophys_odd,
        impact_even=np.concatenate([tp.impact_even, sf.struct_even], axis=1),
        zs_dict=tp.zs_dict,
        zs_impact_dict=tp.zs_impact_dict,
    )


def compute_mutation_side_flags(df: pd.DataFrame, lookup: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return `(is_ab_mut, is_ag_mut)` boolean arrays for `df`."""
    r1 = lookup.loc[df["AbAgID_1"]]
    r2 = lookup.loc[df["AbAgID_2"]]
    h1 = r1["Ab_heavy_chain_seq"].fillna("").astype(str).to_numpy()
    h2 = r2["Ab_heavy_chain_seq"].fillna("").astype(str).to_numpy()
    l1 = r1["Ab_light_chain_seq"].fillna("").astype(str).to_numpy()
    l2 = r2["Ab_light_chain_seq"].fillna("").astype(str).to_numpy()
    g1 = r1["Ag_seq"].fillna("").astype(str).to_numpy()
    g2 = r2["Ag_seq"].fillna("").astype(str).to_numpy()
    is_ab = (h1 != h2) | (l1 != l2)
    is_ag = g1 != g2
    return is_ab, is_ag


def summarize_inventory_entry(name: str, track: str, df: pd.DataFrame, lookup: pd.DataFrame) -> dict[str, Any]:
    is_ab, is_ag = compute_mutation_side_flags(df, lookup)
    ab_only = is_ab & (~is_ag)
    ag_only = is_ag & (~is_ab)
    both = is_ab & is_ag
    lbl = df["label"].to_numpy(dtype=int)

    r1 = lookup.loc[df["AbAgID_1"]]
    r2 = lookup.loc[df["AbAgID_2"]]
    ab_seqs = set(
        zip(r1["Ab_heavy_chain_seq"].fillna(""), r1["Ab_light_chain_seq"].fillna(""))
    ) | set(
        zip(r2["Ab_heavy_chain_seq"].fillna(""), r2["Ab_light_chain_seq"].fillna(""))
    )
    ag_seqs = set(r1["Ag_seq"].fillna("")) | set(r2["Ag_seq"].fillna(""))
    ag_groups = set(df["Ag_group_1"].astype(str)) | set(df["Ag_group_2"].astype(str))
    n = len(df)
    return {
        "name": name,
        "track": track,
        "n_pairs": int(n),
        "n_ab_only": int(ab_only.sum()),
        "pct_ab_only": round(100.0 * float(ab_only.mean()), 2) if n else 0.0,
        "n_ag_only": int(ag_only.sum()),
        "pct_ag_only": round(100.0 * float(ag_only.mean()), 2) if n else 0.0,
        "n_both_global": int(both.sum()),
        "pct_both_global": round(100.0 * float(both.mean()), 2) if n else 0.0,
        "n_impactful": int((lbl != 0).sum()),
        "n_neutral": int((lbl == 0).sum()),
        "ab_only_impactful_pct": round(100.0 * float((lbl[ab_only] != 0).mean()), 2) if ab_only.any() else None,
        "ag_only_impactful_pct": round(100.0 * float((lbl[ag_only] != 0).mean()), 2) if ag_only.any() else None,
        "n_unique_ab_seqs": int(len(ab_seqs)),
        "n_unique_ag_seqs": int(len(ag_seqs)),
        "n_ag_groups": int(len(ag_groups)),
    }


def compute_wt_contact_impact(loc_df: pd.DataFrame, kd_lookup: pd.DataFrame, cc_bound: dict) -> np.ndarray:
    """Compute single-WT 3D `-min d_Cb` mutated-site contact score on `loc_df`."""
    abag2sc = cc_bound["abag_to_scaffold"]
    scafs = cc_bound["scaffolds"]
    n = len(loc_df)
    q_dist = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r = loc_df.iloc[i]
        a, b = kd_lookup.loc[r["AbAgID_1"]], kd_lookup.loc[r["AbAgID_2"]]
        h1, h2 = str(a["Ab_heavy_chain_seq"]), str(b["Ab_heavy_chain_seq"])
        l1, l2 = str(a["Ab_light_chain_seq"]), str(b["Ab_light_chain_seq"])
        g1, g2 = str(a["Ag_seq"]), str(b["Ag_seq"])
        sc = scafs.get(abag2sc.get(r["AbAgID_1"], ""))
        min_d = 50.0
        if sc is not None:
            for s1, s2, dk in ((h1, h2, "min_dist_H"), (l1, l2, "min_dist_L"), (g1, g2, "min_dist_A")):
                if s1 != s2 and len(s1) == len(s2):
                    md = sc[dk]
                    for p, (c1, c2) in enumerate(zip(s1, s2)):
                        if c1 != c2 and p < len(md):
                            min_d = min(min_d, float(md[p]))
        q_dist[i] = -min_d
    return q_dist


def _loaco_cluster_worker(job: tuple[int, str]) -> tuple[str, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], float]:
    """Fit all supervised models on leak-free LOACO fold `cl` and predict on held-out `LOACO-Local`."""
    ci, cl = job
    torch.set_num_threads(1)
    t_c = time.time()

    loc_c1 = _G["loc_c1"]
    cand_c1 = _G["cand_c1"]
    pnl_c1 = _G["pnl_c1"]
    loc_df = _G["loc_df"]
    cand_df = _G["cand_df"]
    pnl_ids_by_cl = _G["pnl_ids_by_cl"]
    kd_lookup = _G["kd_lookup"]
    pool_c1 = _G["pool_c1"]
    pool_c2 = _G["pool_c2"]
    pool_l1 = _G["pool_l1"]
    pool_l2 = _G["pool_l2"]
    is_loc_pool = _G["is_loc_pool"]
    pool_df = _G["pool_df"]
    pool_sp = _G["pool_sp"]
    pool_sf = _G["pool_sf"]
    xb_pool = _G["xb_pool"]
    xk_pool = _G["xk_pool"]
    gb_pool = _G["gb_pool"]
    q_wt_contact = _G["q_wt_contact"]

    loc_ix = np.flatnonzero(loc_c1 == cl)
    cand_ix = np.flatnonzero(cand_c1 == cl)

    held_ids: set[str] = set()
    if len(loc_ix):
        held_ids.update(loc_df.iloc[loc_ix]["AbAgID_1"])
        held_ids.update(loc_df.iloc[loc_ix]["AbAgID_2"])
    if len(cand_ix):
        held_ids.update(cand_df.iloc[cand_ix]["AbAgID_1"])
        held_ids.update(cand_df.iloc[cand_ix]["AbAgID_2"])
    if cl in pnl_ids_by_cl:
        held_ids.update(pnl_ids_by_cl[cl])
    te_lins = {str(kd_lookup.loc[a, "Ab_lineage"]) for a in held_ids}

    keep_m = (pool_c1 != cl) & (pool_c2 != cl) & (~np.isin(pool_l1, list(te_lins))) & (~np.isin(pool_l2, list(te_lins)))
    kp_idx = np.flatnonzero(keep_m)
    rem_loc_cls = sorted(set(pool_c1[kp_idx[is_loc_pool[kp_idx]]]))
    val_cls = set(rem_loc_cls[1::5]) if len(rem_loc_cls) >= 5 else set(rem_loc_cls[:1])
    is_v = np.isin(pool_c1[kp_idx], list(val_cls)) | np.isin(pool_c2[kp_idx], list(val_cls))
    val_lins = set(pool_l1[kp_idx[is_v]]) | set(pool_l2[kp_idx[is_v]])
    is_tr = (~is_v) & (~np.isin(pool_l1[kp_idx], list(val_lins))) & (~np.isin(pool_l2[kp_idx], list(val_lins)))

    tr_idx, va_idx = kp_idx[is_tr], kp_idx[is_v]
    tr_df = pool_df.iloc[tr_idx].reset_index(drop=True)
    va_df = pool_df.iloc[va_idx].reset_index(drop=True)
    tr_sp = pool_sp.subset(tr_idx)
    va_sp = pool_sp.subset(va_idx)
    tr_sf = pool_sf.subset(tr_idx)
    va_sf = pool_sf.subset(va_idx)
    w_tr = compute_stratum_weights(tr_df, tr_sp.is_ag_mut)

    pool_loc_indices = np.flatnonzero(is_loc_pool)[loc_ix]
    esp = pool_sp.subset(pool_loc_indices)
    esf = pool_sf.subset(pool_loc_indices)
    xb_e = xb_pool[pool_loc_indices]
    xk_e = xk_pool[pool_loc_indices]
    gb_tr = _slice_gb(gb_pool, tr_idx)
    gb_va = _slice_gb(gb_pool, va_idx)
    gb_e = _slice_gb(gb_pool, pool_loc_indices)
    seed = 42 + ci

    n_e = len(esp.y)
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Random Baseline": (np.zeros(n_e), np.zeros(n_e)),
        "WT 3D Contact (-min d_Cb)": (np.zeros(n_e), q_wt_contact[loc_ix]),
        "AbLang2-Paired (1D ZS)": (esp.zs_dict["ablang2_zeroshot"], esp.zs_impact_dict["ablang2_zeroshot"]),
        "ESMC-300M (1D ZS)": (esp.zs_dict["esmc_300m_zeroshot"], esp.zs_impact_dict["esmc_300m_zeroshot"]),
        "ESMC+AbLang2 (1D ZS)": (esp.zs_dict["esmc_ablang2_zeroshot"], esp.zs_impact_dict["esmc_ablang2_zeroshot"]),
        "PRODIGY+SASA (3D Energy ZS)": (esf.zs_dir["prodigy_sasa"], esf.zs_imp["prodigy_sasa"]),
        "ProteinMPNN (3D Complex ZS)": (esf.zs_dir["protein_mpnn"], esf.zs_imp["protein_mpnn"]),
        "AntiFold (3D Ab-Ag ZS)": (esf.zs_dir["antifold"], esf.zs_imp["antifold"]),
        "ESM-IF1 (3D Complex ZS)": (esf.zs_dir["esm_if1"], esf.zs_imp["esm_if1"]),
        "SaProt-650M (3D-Token ZS)": (esf.zs_dir["saprot_zs"], esf.zs_imp["saprot_zs"]),
        "MINT-650M (Multi-Chain ZS)": (esf.zs_dir["mint_zs"], esf.zs_imp["mint_zs"]),
    }

    m_bio = AntisymmetricLinearRanker(use_neutral_nullspace=True).fit(
        xb_pool[tr_idx], tr_sp.y, w_tr, xb_pool[va_idx], va_df, va_sp.is_ag_mut, labels_tr=tr_sp.labels
    )
    m_km = AntisymmetricLinearRanker(use_neutral_nullspace=True).fit(
        xk_pool[tr_idx], tr_sp.y, w_tr, xk_pool[va_idx], va_df, va_sp.is_ag_mut, labels_tr=tr_sp.labels
    )
    db, dk = m_bio.predict(xb_e), m_km.predict(xk_e)
    preds["Biophys+AAC (Linear)"] = (db, np.abs(db))
    preds["k-mer Dipeptide (Ridge)"] = (dk, np.abs(dk))

    xm_tr = build_odd_linear_matrix([tr_sp.esmc_ab_1, tr_sp.esmc_ag_1], [tr_sp.esmc_ab_2, tr_sp.esmc_ag_2], include_unit_direction=True)
    xm_va = build_odd_linear_matrix([va_sp.esmc_ab_1, va_sp.esmc_ag_1], [va_sp.esmc_ab_2, va_sp.esmc_ag_2], include_unit_direction=True)
    xr_tr = build_odd_linear_matrix([tr_sp.esmc_ab_1, tr_sp.esmc_ag_1], [tr_sp.esmc_ab_2, tr_sp.esmc_ag_2], include_unit_direction=False)
    xr_va = build_odd_linear_matrix([va_sp.esmc_ab_1, va_sp.esmc_ag_1], [va_sp.esmc_ab_2, va_sp.esmc_ag_2], include_unit_direction=False)

    m_ms = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(xm_tr, tr_sp.y, w_tr, xm_va, va_df, va_sp.is_ag_mut, labels_tr=tr_sp.labels)
    m_rw = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(xr_tr, tr_sp.y, None, xr_va, va_df, va_sp.is_ag_mut, labels_tr=tr_sp.labels)
    e2h = TwoHeadRegimeRanker(local_head=m_ms, global_head=m_rw)
    dv_tr, dv_va = e2h.predict(tr_sp, xm_tr, xr_tr), e2h.predict(va_sp, xm_va, xr_va)
    imp_e = SymmetricImpactClassifier(seed=seed).fit(tr_sp, va_sp, dv_tr, dv_va, val_df=va_df)
    hurd_e = OrdinalHurdleRanker(e2h, imp_e).calibrate(va_df, va_sp, dv_va)
    m_bil = NeuralPairwiseTrainer(arch="bilinear", seed=seed, epochs=22).fit(tr_sp, va_sp, va_df, w_tr)
    m_log = LogiCA3StateTrainer(seed=42, epochs=15).fit(tr_sp, va_sp, va_df, w_tr)

    xm_e = build_odd_linear_matrix([esp.esmc_ab_1, esp.esmc_ag_1], [esp.esmc_ab_2, esp.esmc_ag_2], include_unit_direction=True)
    xr_e = build_odd_linear_matrix([esp.esmc_ab_1, esp.esmc_ag_1], [esp.esmc_ab_2, esp.esmc_ag_2], include_unit_direction=False)
    dms = m_ms.predict(xm_e)
    preds["ESMC-300M (Multi-Scale Binary)"] = (dms, np.abs(dms))
    preds["ESMC-300M (3-State Hurdle)"] = hurd_e.predict_3state(esp, e2h.predict(esp, xm_e, xr_e))
    preds["ESMC+AbLang2+LLR (Bilinear)"] = m_bil.predict_3state(esp)
    preds["LogiCA (1D Logit Cross-Attn)"] = m_log.predict_3state(esp)

    pred_sap = fit_3state_foundation_hurdle(
        tr_sf.saprot_ab_1, tr_sf.saprot_ab_2, tr_sf.saprot_ag_1, tr_sf.saprot_ag_2,
        np.concatenate([tr_sf.zs_dir["saprot_zs"][:, None], tr_sp.biophys_odd], axis=1), tr_sf.struct_even[:, :9],
        va_sf.saprot_ab_1, va_sf.saprot_ab_2, va_sf.saprot_ag_1, va_sf.saprot_ag_2,
        np.concatenate([va_sf.zs_dir["saprot_zs"][:, None], va_sp.biophys_odd], axis=1), va_sf.struct_even[:, :9],
        tr_sp, va_sp, va_df, w_tr, seed=seed,
    )
    pred_mint = fit_3state_foundation_hurdle(
        tr_sf.mint_ab_1, tr_sf.mint_ab_2, tr_sf.mint_ag_1, tr_sf.mint_ag_2,
        np.concatenate([tr_sf.zs_dir["mint_zs"][:, None], tr_sp.biophys_odd], axis=1), np.zeros((len(tr_df), 1), dtype=np.float32),
        va_sf.mint_ab_1, va_sf.mint_ab_2, va_sf.mint_ag_1, va_sf.mint_ag_2,
        np.concatenate([va_sf.zs_dir["mint_zs"][:, None], va_sp.biophys_odd], axis=1), np.zeros((len(va_df), 1), dtype=np.float32),
        tr_sp, va_sp, va_df, w_tr, seed=seed,
    )
    m_anti = ANTIPASTI3StateTrainer(seed=seed, epochs=15).fit(tr_sf, tr_sp, va_sf, va_sp, va_df, w_tr)
    m_gb = GearBind3StateTrainer(seed=seed, epochs=15).fit(
        tr_sf, tr_sp, va_sf, va_sp, va_df, w_tr, precomp_tr_g=gb_tr, precomp_val_g=gb_va
    )
    pred_walle = fit_walle_affinity_3state(tr_sf, tr_sp, va_sf, va_sp, va_df, w_tr, seed=seed)

    preds["SaProt-650M (3-State Hurdle)"] = pred_sap(
        esp, esf.saprot_ab_1, esf.saprot_ab_2, esf.saprot_ag_1, esf.saprot_ag_2,
        np.concatenate([esf.zs_dir["saprot_zs"][:, None], esp.biophys_odd], axis=1), esf.struct_even[:, :9],
    )
    preds["MINT-650M (3-State Hurdle)"] = pred_mint(
        esp, esf.mint_ab_1, esf.mint_ab_2, esf.mint_ag_1, esf.mint_ag_2,
        np.concatenate([esf.zs_dir["mint_zs"][:, None], esp.biophys_odd], axis=1), np.zeros((n_e, 1), dtype=np.float32),
    )
    preds["ANTIPASTI (Normal-Mode 3D CNN)"] = m_anti.predict_3state(esf, esp)
    preds["GearBind (3D Geometric GNN)"] = m_gb.predict_3state(esf, _precomp_g=gb_e)
    preds["WALLE-Affinity (3-State 3D Graph)"] = pred_walle(esf, esp)

    return cl, loc_ix, preds, time.time() - t_c


def _eval_loaco_slice(
    labels: np.ndarray,
    bdiff: np.ndarray,
    loc_c1: np.ndarray,
    pred_d: np.ndarray,
    pred_i: np.ndarray,
    mask: np.ndarray,
    clusters: list[str],
    precomp_per_cl: dict[str, dict] | None = None,
    pooled_override: dict[str, Any] | None = None,
    min_pairs: int = 5,
    min_class_pairs: int = 2,
) -> dict[str, Any]:
    per_cl: dict[str, dict] = {}
    for cl in sorted(clusters):
        ix = np.flatnonzero((loc_c1 == cl) & mask)
        if len(ix) >= min_pairs:
            if precomp_per_cl is not None and cl in precomp_per_cl:
                per_cl[cl] = precomp_per_cl[cl]
            else:
                per_cl[cl] = compute_3state_metrics(labels[ix], bdiff[ix], pred_d[ix], pred_i[ix])
    cb = compute_cluster_bootstrap_summary(per_cl, min_class_pairs=min_class_pairs, n_bootstrap=2000, seed=42)
    ix_all = np.flatnonzero(mask)
    if pooled_override is not None:
        pooled = pooled_override
    else:
        pooled = compute_3state_metrics(labels[ix_all], bdiff[ix_all], pred_d[ix_all], pred_i[ix_all]) if len(ix_all) >= 5 else {}
    return {
        "n_pairs": int(len(ix_all)),
        "n_clusters_total": int(len(per_cl)),
        "n_directional_clusters": int(cb["sign"]["n_clusters"]),
        "n_impact_clusters": int(cb["impact"]["n_clusters"]),
        "auc_dir": cb["auc_dir"],
        "dir_ci95": cb["sign"]["cluster_ci95"],
        "auc_impact": cb["auc_impact"],
        "imp_ci95": cb["impact"]["cluster_ci95"],
        "auc_ordinal": cb["auc_ordinal"],
        "ord_ci95": cb["ordinal"]["cluster_ci95"],
        "pooled_auc_dir": pooled.get("auc_dir"),
        "pooled_auc_impact": pooled.get("auc_impact"),
        "pooled_auc_ordinal": pooled.get("auc_ordinal"),
    }


def _transfer_3d_fold_worker(job: tuple[int, frozenset[str], bool]) -> tuple[int, Any, Any, float]:
    """Fit leak-free KD source models (`ANTIPASTI`, `GearBind`) for one fold key."""
    kid, excl_ags, excl_skempi = job
    torch.set_num_threads(1)
    t0 = time.time()

    kd_pool_df = _G["kd_pool_df"]
    kd_pool_tp = _G["kd_pool_tp"]
    kd_pool_sf = _G["kd_pool_sf"]
    gb_kd_pool = _G["gb_kd_pool"]

    m = (~kd_pool_df["Ag_group_1"].isin(excl_ags)) & (~kd_pool_df["Ag_group_2"].isin(excl_ags))
    if excl_skempi:
        m &= (kd_pool_df["Dataset_1"] != "SKEMPIv2") & (kd_pool_df["Dataset_2"] != "SKEMPIv2")
    keep = np.flatnonzero(m.to_numpy(dtype=bool))
    sub_df = kd_pool_df.iloc[keep].reset_index(drop=True)
    loc_ags = sorted(sub_df.loc[sub_df["Category"] != "Global", "Ag_group_1"].unique())
    val_ags = set(loc_ags[1::5]) if len(loc_ags) >= 5 else set(loc_ags[:1])
    is_val = (sub_df["Ag_group_1"].isin(val_ags) | sub_df["Ag_group_2"].isin(val_ags)).to_numpy()

    tr_k = keep[~is_val]
    va_k = keep[is_val]
    tr_df = kd_pool_df.iloc[tr_k].reset_index(drop=True)
    va_df = kd_pool_df.iloc[va_k].reset_index(drop=True)
    tr_tp = kd_pool_tp.subset(tr_k)
    va_tp = kd_pool_tp.subset(va_k)
    tr_sf = kd_pool_sf.subset(tr_k)
    va_sf = kd_pool_sf.subset(va_k)
    w_tr = compute_stratum_weights(tr_df, tr_tp.is_ag_mut)
    gb_tr = _slice_gb(gb_kd_pool, tr_k)
    gb_va = _slice_gb(gb_kd_pool, va_k)

    m_anti = ANTIPASTI3StateTrainer(seed=42, epochs=15).fit(tr_sf, tr_tp, va_sf, va_tp, va_df, w_tr)
    m_gb = GearBind3StateTrainer(seed=42, epochs=15).fit(
        tr_sf, tr_tp, va_sf, va_tp, va_df, w_tr, precomp_tr_g=gb_tr, precomp_val_g=gb_va
    )
    return kid, m_anti, m_gb, time.time() - t0


def _linear_kd_src_worker(job: tuple[int, frozenset[str], frozenset[str], bool]) -> tuple[int, KDSourceRanker, float]:
    """Fit leak-free 1D `KDSourceRanker` for one unique `(eff_ags, eff_lins, excl_sk)` key in parallel."""
    kid, eff_ags, eff_lins, excl_sk = job
    torch.set_num_threads(1)
    t0 = time.time()
    kd_pool_df = _G["kd_pool_df"]
    kd_pool_tp = _G["kd_pool_tp"]
    X_pool = _G["X_pool"]
    kd_lookup_tf = _G["kd_lookup_tf"]

    m = (~kd_pool_df["Ag_group_1"].isin(eff_ags)) & (~kd_pool_df["Ag_group_2"].isin(eff_ags))
    if excl_sk:
        m &= (kd_pool_df["Dataset_1"] != "SKEMPIv2") & (kd_pool_df["Dataset_2"] != "SKEMPIv2")
    if eff_lins:
        l1 = kd_lookup_tf.loc[kd_pool_df["AbAgID_1"], "Ab_lineage"].to_numpy()
        l2 = kd_lookup_tf.loc[kd_pool_df["AbAgID_2"], "Ab_lineage"].to_numpy()
        m &= (~np.isin(l1, list(eff_lins))) & (~np.isin(l2, list(eff_lins)))
    keep = np.flatnonzero(m.to_numpy(dtype=bool))
    sub_df = kd_pool_df.iloc[keep].reset_index(drop=True)
    sub_tp = kd_pool_tp.subset(keep)
    w_all = compute_stratum_weights(sub_df, sub_tp.is_ag_mut)
    loc_ags = sorted(sub_df.loc[sub_df["Category"] != "Global", "Ag_group_1"].unique())
    val_ags = set(loc_ags[1::5]) if len(loc_ags) >= 5 else set(loc_ags[:1])
    is_val = (sub_df["Ag_group_1"].isin(val_ags) | sub_df["Ag_group_2"].isin(val_ags)).to_numpy()
    tr_i, va_i = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    src = fit_kd_source_ranker(
        X_pool[keep[tr_i]], sub_tp.y[tr_i], w_all[tr_i],
        X_pool[keep[va_i]], sub_tp.y[va_i],
        tp_tr=sub_tp.subset(tr_i), tp_val=sub_tp.subset(va_i),
    )
    return kid, src, time.time() - t0


def _fmt_macro(m: dict[str, Any]) -> str:
    if not m or m.get("auc_ordinal") is None:
        return "—"
    d_s = f"{m['auc_dir']:.3f}" if m.get("auc_dir") is not None else "—"
    i_s = f"{m['auc_impact']:.3f}" if m.get("auc_impact") is not None else "—"
    o_s = f"{m['auc_ordinal']:.3f}"
    return f"`{d_s}` / `{i_s}` / **`{o_s}`**"


def main() -> None:
    t_start = time.time()
    root = get_repo_root()
    bdir = root / "data" / "benchmark"
    edir = root / "data" / "embeddings"
    sdir = root / "data" / "structures"
    res_dir = root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading master assay tables and computing complete benchmark mutation-side inventory...", flush=True)
    kd_raw = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False)
    kd_lookup = kd_raw.drop_duplicates("AbAgID").set_index("AbAgID")
    kd_lookup["Ab_light_chain_seq"] = kd_lookup["Ab_light_chain_seq"].fillna("")

    cma_raw = pd.read_csv(bdir / "assays_cross_modal.tsv", sep="\t", low_memory=False)
    cma_lookup = cma_raw.drop_duplicates("AbAgID").set_index("AbAgID")
    cma_lookup["Ab_light_chain_seq"] = cma_lookup["Ab_light_chain_seq"].fillna("")

    tfa_raw = pd.read_csv(bdir / "assays_transfer.tsv", sep="\t", low_memory=False)
    tfa_lookup = tfa_raw.drop_duplicates("AbAgID").set_index("AbAgID")
    tfa_lookup["Ab_light_chain_seq"] = tfa_lookup["Ab_light_chain_seq"].fillna("")

    vhh_raw = pd.read_csv(bdir / "assays_vhh.tsv", sep="\t", low_memory=False)
    vhh_raw["Ab_light_chain_seq"] = ""
    vhh_lookup = vhh_raw.drop_duplicates("AbAgID").set_index("AbAgID")

    pool_df = pd.concat(
        [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")],
        ignore_index=True,
    )
    is_loc_pool = pool_df["Category"].isin(["L1", "L2", "L3"]).to_numpy()
    loc_df = pool_df.iloc[np.flatnonzero(is_loc_pool)].reset_index(drop=True)
    glob_df = pool_df.iloc[np.flatnonzero(~is_loc_pool)].reset_index(drop=True)

    cand_all_df = pd.concat(
        [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}_candidates.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")],
        ignore_index=True,
    )

    skempi_assays = kd_raw[kd_raw["Dataset"] == "SKEMPIv2"].copy()
    skempi_df, _ = build_within_antigen_pairs(
        skempi_assays, m=1.0, m_neutral=0.30, include_neutral=True, neutral_ratio=1.0,
        modality="DDG_SKEMPIv2", ab_cap=3, per_ag_cap=500, seed=42, prefix="SK10",
    )
    skempi_df = skempi_df.reset_index(drop=True)

    cm_all = pd.read_csv(bdir / "splits" / "cross_modal" / "modality_test.tsv", sep="\t", low_memory=False)
    dms_df = cm_all[cm_all["Assay_modality"] == "DMS"].reset_index(drop=True)
    elisa_df = cm_all[cm_all["Assay_modality"] == "ELISA"].reset_index(drop=True)
    abddg_df = cm_all[cm_all["Assay_modality"] == "DDG"].reset_index(drop=True)
    enrich_fv_df = cm_all[(cm_all["Assay_modality"] == "Enrichment") & (cm_all["Dataset_1"] == "AbBibench")].reset_index(drop=True)
    enrich_vhh_df = cm_all[cm_all["Dataset_1"] == "NbBench"].reset_index(drop=True)
    ec50_df = cm_all[cm_all["Assay_modality"] == "EC50"].reset_index(drop=True)
    gen_df = pd.read_csv(bdir / "splits" / "generative" / "generative_pairs.tsv", sep="\t", low_memory=False).reset_index(drop=True)
    gen_lookup = pd.concat([kd_lookup, vhh_lookup, cma_lookup])
    gen_lookup = gen_lookup[~gen_lookup.index.duplicated(keep="first")]

    ic50_df = pd.read_csv(bdir / "splits" / "transfer" / "ic50_test.tsv", sep="\t", low_memory=False).reset_index(drop=True)
    esc_df = pd.read_csv(bdir / "splits" / "transfer" / "escape_test.tsv", sep="\t", low_memory=False).reset_index(drop=True)
    vhh_pairs_all = pd.read_csv(bdir / "splits" / "vhh" / "vhh_ranking_test.tsv", sep="\t", low_memory=False)
    vhh_within_df = vhh_pairs_all[vhh_pairs_all["Category"] != "Global"].reset_index(drop=True)

    inventory = [
        summarize_inventory_entry("LOACO-Local (L1–L3 KD)", "Suite 1: Maturation", loc_df, kd_lookup),
        summarize_inventory_entry("Candidate Ranking (d_Lev > 3 KD)", "Suite 2: Discovery", cand_all_df, kd_lookup),
        summarize_inventory_entry("Cross-Antigen Global KD", "Suite 2: Discovery", glob_df, kd_lookup),
        summarize_inventory_entry("Generative Redesign (1,354 Designs)", "Suite 4: Generative", gen_df, gen_lookup),
        summarize_inventory_entry("SKEMPI 2.0 Thermodynamic ΔΔG", "Suite 1/3: Structural ΔΔG", skempi_df, kd_lookup),
        summarize_inventory_entry("AbAgym Interface DMS", "Suite 3: Cross-Modal", dms_df, cma_lookup),
        summarize_inventory_entry("AbBiBench Display Enrichment", "Suite 3: Cross-Modal", enrich_fv_df, cma_lookup),
        summarize_inventory_entry("ELISA (AbDesign)", "Suite 3: Cross-Modal", elisa_df, cma_lookup),
        summarize_inventory_entry("NbBench Galectin-3 VHH", "Suite 3: Cross-Modal", enrich_vhh_df, cma_lookup),
        summarize_inventory_entry("AbDesign Mutational ΔΔG", "Suite 3: Cross-Modal", abddg_df, cma_lookup),
        summarize_inventory_entry("Cell-Surface EC50 (Kothiwal)", "Suite 3: Cross-Modal", ec50_df, cma_lookup),
        summarize_inventory_entry("Viral Neutralization IC50", "Suite 3: Viral Transfer", ic50_df, tfa_lookup),
        summarize_inventory_entry("SARS-CoV-2 RBD Escape (Bloom)", "Suite 3: Viral Transfer", esc_df, tfa_lookup),
        summarize_inventory_entry("Single-Domain VHH KD (Within-Ag)", "Suite 3/4: Nanobody VHH", vhh_within_df, vhh_lookup),
    ]
    for row in inventory:
        print(
            f"  [{row['name']:34s}] n={row['n_pairs']:6,d} | Ab-Only={row['n_ab_only']:6,d} ({row['pct_ab_only']:5.1f}%) | "
            f"Ag-Only={row['n_ag_only']:6,d} ({row['pct_ag_only']:5.1f}%) | Global={row['n_both_global']:6,d}",
            flush=True,
        )

    ckpt_path = Path(os.environ["ABRANK_PVE_CKPT"]) if os.environ.get("ABRANK_PVE_CKPT") else None
    ckpt_obj: dict[str, Any] = torch.load(ckpt_path, weights_only=False) if (ckpt_path and ckpt_path.exists()) else {}
    have_step3_ckpt = "oof_loaco_d" in ckpt_obj and "oof_loaco_i" in ckpt_obj

    print("\n2. Loading PLM, 3D scaffold, inverse-folding, SaProt, and MINT caches...", flush=True)
    esmc_cache = torch.load(edir / "esmc_300m.pt", map_location="cpu", weights_only=False)
    ablang_cache = torch.load(edir / "ablang2_paired.pt", map_location="cpu", weights_only=False)
    esm2_cache = torch.load(edir / "esm2_150m.pt", map_location="cpu", weights_only=False)
    complex_cache = torch.load(sdir / "complex_cache.pt", map_location="cpu", weights_only=False)
    inv_cache = torch.load(edir / "inverse_folding_scaffolds.pt", map_location="cpu", weights_only=False)

    loc_c1 = loc_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
    loc_clusters = sorted(set(loc_c1))
    n_loc = len(loc_df)
    loc_is_ab, loc_is_ag = compute_mutation_side_flags(loc_df, kd_lookup)
    loc_ab_only = loc_is_ab & (~loc_is_ag)
    loc_ag_only = loc_is_ag & (~loc_is_ab)
    loc_labels = loc_df["label"].to_numpy(dtype=int)
    loc_bdiff = loc_df["binding_diff"].to_numpy(dtype=np.float64)

    zs_loaco_names = {
        "Random Baseline", "WT 3D Contact (-min d_Cb)", "AbLang2-Paired (1D ZS)", "ESMC-300M (1D ZS)",
        "ESMC+AbLang2 (1D ZS)", "PRODIGY+SASA (3D Energy ZS)", "ProteinMPNN (3D Complex ZS)",
        "AntiFold (3D Ab-Ag ZS)", "ESM-IF1 (3D Complex ZS)", "SaProt-650M (3D-Token ZS)", "MINT-650M (Multi-Chain ZS)",
    }
    loaco_eval_ref = json.loads((res_dir / "loaco_evaluation.json").read_text(encoding="utf-8"))["models"]
    pure_ab_clusters = {
        cl for cl in loc_clusters
        if ((loc_c1 == cl) & loc_ab_only).sum() >= 5 and ((loc_c1 == cl) & loc_ag_only).sum() == 0
    }
    pure_ag_clusters = {
        cl for cl in loc_clusters
        if ((loc_c1 == cl) & loc_ag_only).sum() >= 5 and ((loc_c1 == cl) & loc_ab_only).sum() == 0
    }
    mixed_clusters = sorted(
        cl for cl in loc_clusters
        if ((loc_c1 == cl) & loc_ab_only).sum() > 0 and ((loc_c1 == cl) & loc_ag_only).sum() > 0 and ((loc_c1 == cl).sum() >= 5)
    )

    print("\n3. Running 22-Cluster LOACO-Local Paratope (`Ab-Mut`) vs. Epitope (`Ag-Mut`) Evaluation...", flush=True)
    if have_step3_ckpt:
        print("  Restored Step 3 mixed LOACO fold predictions from stage checkpoint!", flush=True)
        oof_loaco_d = ckpt_obj["oof_loaco_d"]
        oof_loaco_i = ckpt_obj["oof_loaco_i"]
    else:
        saprot_cache = torch.load(edir / "saprot_650m.pt", map_location="cpu", weights_only=False)
        mint_cache = torch.load(edir / "mint_650m.pt", map_location="cpu", weights_only=False)
        abag_3d = set(complex_cache["abag_to_scaffold"].keys())
        esmc_keys = set(esmc_cache["plls"].keys())
        esm2_keys = set(esm2_cache["plls"].keys())
        ablang_keys = set(ablang_cache["plls"].keys())

        def _is_fully_cached(aid: str) -> bool:
            if aid not in abag_3d:
                return False
            r = kd_lookup.loc[aid]
            vh_s, vl_s, ag_s = str(r["Ab_heavy_chain_seq"]), str(r["Ab_light_chain_seq"]), str(r["Ag_seq"])
            kh, ka = seq_hash(vh_s), seq_hash(ag_s)
            kl = seq_hash(vl_s) if vl_s else kh
            kp = hashlib.sha1(f"{vh_s}|{vl_s}".encode("ascii")).hexdigest()[:16] if vl_s else ""
            return (
                kh in esmc_keys and kl in esmc_keys and ka in esmc_keys
                and kh in esm2_keys and kl in esm2_keys and ka in esm2_keys
                and (kp in ablang_keys if vl_s else True)
            )

        cand_ok = [_is_fully_cached(a) and _is_fully_cached(b) for a, b in zip(cand_all_df["AbAgID_1"], cand_all_df["AbAgID_2"])]
        cand_df = cand_all_df[cand_ok].reset_index(drop=True)

        pw_raw = pd.concat(
            [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}_pointwise.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")],
            ignore_index=True,
        )
        pw_cached = pw_raw[[_is_fully_cached(a) for a in pw_raw["AbAgID"]]].copy()
        pnl_ids_by_cl: dict[str, set[str]] = {}
        for (_, _), g in pw_cached.groupby(["Ag_group", "Dataset"], sort=True):
            if 6 <= len(g) <= 200 and g["binding_score"].nunique() >= 3:
                cl_p = str(kd_lookup.loc[g["AbAgID"].iloc[0], "Ag_cluster"])
                pnl_ids_by_cl.setdefault(cl_p, set()).update(g["AbAgID"].tolist())

        pool_sp = assemble_split_tensors(pool_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)
        pool_sf = assemble_structure_pair_features(pool_df, kd_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)
        gb_pool = (
            GearBind3StateTrainer._precompute_graphs(pool_sf.patch_ca_1, pool_sf.patch_cb_1, pool_sf.patch_chain_1, pool_sf.patch_seqpos_1),
            GearBind3StateTrainer._precompute_graphs(pool_sf.patch_ca_2, pool_sf.patch_cb_2, pool_sf.patch_chain_2, pool_sf.patch_seqpos_2),
        )

        used_pool_ids = sorted(set(pool_df["AbAgID_1"]).union(pool_df["AbAgID_2"]))
        sub_assays = kd_lookup.loc[used_pool_ids].reset_index()
        bio_map = build_complex_handcrafted_table(sub_assays, include_dipeptides=False)
        kmer_map = build_complex_handcrafted_table(sub_assays, include_dipeptides=True)
        xb_pool = build_odd_linear_matrix(
            [np.stack([bio_map[k] for k in pool_df["AbAgID_1"]])],
            [np.stack([bio_map[k] for k in pool_df["AbAgID_2"]])],
            include_unit_direction=True,
        )
        xk_pool = build_odd_linear_matrix(
            [np.stack([kmer_map[k] for k in pool_df["AbAgID_1"]])],
            [np.stack([kmer_map[k] for k in pool_df["AbAgID_2"]])],
            include_unit_direction=True,
        )
        q_wt_contact = compute_wt_contact_impact(loc_df, kd_lookup, complex_cache)

        cand_c1 = cand_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
        pool_c1 = pool_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
        pool_c2 = pool_df["AbAgID_2"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
        pool_l1 = pool_df["AbAgID_1"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()
        pool_l2 = pool_df["AbAgID_2"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()

        all_loaco_clusters = sorted(set(loc_c1) | set(cand_c1) | set(pnl_ids_by_cl.keys()))
        ci_by_cl = {cl: idx for idx, cl in enumerate(all_loaco_clusters)}

        _G.clear()
        _G.update(
            loc_c1=loc_c1, cand_c1=cand_c1, pnl_c1=None, loc_df=loc_df, cand_df=cand_df,
            pnl_ids_by_cl=pnl_ids_by_cl, kd_lookup=kd_lookup, pool_c1=pool_c1, pool_c2=pool_c2,
            pool_l1=pool_l1, pool_l2=pool_l2, is_loc_pool=is_loc_pool, pool_df=pool_df,
            pool_sp=pool_sp, pool_sf=pool_sf, xb_pool=xb_pool, xk_pool=xk_pool,
            gb_pool=gb_pool, q_wt_contact=q_wt_contact,
        )

        loc_pool_ix_all = np.flatnonzero(is_loc_pool)
        loc_sp_all = pool_sp.subset(loc_pool_ix_all)
        loc_sf_all = pool_sf.subset(loc_pool_ix_all)
        zs_loaco_map = {
            "Random Baseline": (np.zeros(n_loc), np.zeros(n_loc)),
            "WT 3D Contact (-min d_Cb)": (np.zeros(n_loc), q_wt_contact),
            "AbLang2-Paired (1D ZS)": (loc_sp_all.zs_dict["ablang2_zeroshot"], loc_sp_all.zs_impact_dict["ablang2_zeroshot"]),
            "ESMC-300M (1D ZS)": (loc_sp_all.zs_dict["esmc_300m_zeroshot"], loc_sp_all.zs_impact_dict["esmc_300m_zeroshot"]),
            "ESMC+AbLang2 (1D ZS)": (loc_sp_all.zs_dict["esmc_ablang2_zeroshot"], loc_sp_all.zs_impact_dict["esmc_ablang2_zeroshot"]),
            "PRODIGY+SASA (3D Energy ZS)": (loc_sf_all.zs_dir["prodigy_sasa"], loc_sf_all.zs_imp["prodigy_sasa"]),
            "ProteinMPNN (3D Complex ZS)": (loc_sf_all.zs_dir["protein_mpnn"], loc_sf_all.zs_imp["protein_mpnn"]),
            "AntiFold (3D Ab-Ag ZS)": (loc_sf_all.zs_dir["antifold"], loc_sf_all.zs_imp["antifold"]),
            "ESM-IF1 (3D Complex ZS)": (loc_sf_all.zs_dir["esm_if1"], loc_sf_all.zs_imp["esm_if1"]),
            "SaProt-650M (3D-Token ZS)": (loc_sf_all.zs_dir["saprot_zs"], loc_sf_all.zs_imp["saprot_zs"]),
            "MINT-650M (Multi-Chain ZS)": (loc_sf_all.zs_dir["mint_zs"], loc_sf_all.zs_imp["mint_zs"]),
        }

        oof_loaco_d = {m: np.zeros(n_loc, dtype=np.float64) for m, _ in LOACO_MODEL_SPECS}
        oof_loaco_i = {m: np.zeros(n_loc, dtype=np.float64) for m, _ in LOACO_MODEL_SPECS}
        for m_z, (zd, zi) in zs_loaco_map.items():
            oof_loaco_d[m_z] = np.asarray(zd, dtype=np.float64).copy()
            oof_loaco_i[m_z] = np.asarray(zi, dtype=np.float64).copy()

        loaco_jobs = [(ci_by_cl[cl], cl) for cl in mixed_clusters]
        print(
            f"  Reusing exact per-cluster metrics on {len(pure_ab_clusters)} pure Ab-Mut + {len(pure_ag_clusters)} pure Ag-Mut clusters, "
            f"and fitting {len(loaco_jobs)} mixed Ab+Ag LOACO folds in 1 parallel wave...",
            flush=True,
        )
        with mp.get_context("fork").Pool(min(10, len(loaco_jobs))) as pool:
            for cl, loc_ix, preds_cl, dt_cl in pool.imap_unordered(_loaco_cluster_worker, loaco_jobs):
                for m_name, (d_v, i_v) in preds_cl.items():
                    oof_loaco_d[m_name][loc_ix] = d_v
                    oof_loaco_i[m_name][loc_ix] = i_v
                print(f"    [LOACO {cl}] n={len(loc_ix):4d} done in {dt_cl:.1f}s", flush=True)
        _G.clear()
        del pool_sp, pool_sf, gb_pool, saprot_cache, mint_cache, xb_pool, xk_pool
        gc.collect()
        if ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_obj["oof_loaco_d"] = oof_loaco_d
            ckpt_obj["oof_loaco_i"] = oof_loaco_i
            torch.save(ckpt_obj, ckpt_path)

    # Matched shared clusters: clusters containing >=5 Ab-Mut pairs AND >=5 Ag-Mut pairs (100% co-crystals)
    shared_ab_ag_clusters = [
        cl for cl in loc_clusters
        if ((loc_c1 == cl) & loc_ab_only).sum() >= 5 and ((loc_c1 == cl) & loc_ag_only).sum() >= 5
    ]
    print(f"  Shared Ab-Mut & Ag-Mut co-crystal clusters (>=5 each): {len(shared_ab_ag_clusters)} -> {shared_ab_ag_clusters}", flush=True)

    loaco_summary: dict[str, Any] = {}
    for m_name, fam in LOACO_MODEL_SPECS:
        pd_v, pi_v = oof_loaco_d[m_name], oof_loaco_i[m_name]
        ref_loc = loaco_eval_ref[m_name]["loaco_local_22_clusters"] if m_name in loaco_eval_ref else None
        ref_per_cl = ref_loc["per_cluster"] if ref_loc is not None else None
        ref_pure_ab = {cl: ref_per_cl[cl] for cl in pure_ab_clusters if ref_per_cl and cl in ref_per_cl} if ref_per_cl else None
        ref_pure_ag = {cl: ref_per_cl[cl] for cl in pure_ag_clusters if ref_per_cl and cl in ref_per_cl} if ref_per_cl else None
        ref_pooled_all = (
            {
                "auc_dir": ref_loc["pooled_auc_dir"],
                "auc_impact": ref_loc["pooled_auc_impact"],
                "auc_ordinal": ref_loc["pooled_auc_ordinal"],
            }
            if ref_loc is not None
            else None
        )
        is_zs = m_name in zs_loaco_names
        loaco_summary[m_name] = {
            "family": fam,
            "all_22_clusters": _eval_loaco_slice(
                loc_labels, loc_bdiff, loc_c1, pd_v, pi_v, np.ones(n_loc, dtype=bool), loc_clusters,
                precomp_per_cl=ref_per_cl, pooled_override=ref_pooled_all,
            ),
            "ab_mut_22_clusters": _eval_loaco_slice(
                loc_labels, loc_bdiff, loc_c1, pd_v, pi_v, loc_ab_only, loc_clusters,
                precomp_per_cl=ref_pure_ab, pooled_override=None if is_zs else {},
            ),
            "ag_mut_clusters": _eval_loaco_slice(
                loc_labels, loc_bdiff, loc_c1, pd_v, pi_v, loc_ag_only, loc_clusters,
                precomp_per_cl=ref_pure_ag, pooled_override=None if is_zs else {},
            ),
            "matched_shared_clusters_ab_mut": _eval_loaco_slice(
                loc_labels, loc_bdiff, loc_c1, pd_v, pi_v, loc_ab_only & np.isin(loc_c1, shared_ab_ag_clusters), shared_ab_ag_clusters
            ),
            "matched_shared_clusters_ag_mut": _eval_loaco_slice(
                loc_labels, loc_bdiff, loc_c1, pd_v, pi_v, loc_ag_only & np.isin(loc_c1, shared_ab_ag_clusters), shared_ab_ag_clusters
            ),
        }

    print("\n4. Running 5-Fold Unseen-Antigen GroupCV on SKEMPI 2.0 & AbAgym Interface DMS (`Ab-Mut` vs. `Ag-Mut`)...", flush=True)
    # Assign joint <80% clonal lineages across KD + Transfer tables (matching Step 07 & Step 17)
    combined_ab = pd.concat(
        [
            kd_raw[["AbID", "Ab_heavy_chain_seq", "Ab_light_chain_seq", "WT_Abname", "Dataset"]],
            tfa_raw[["AbID", "Ab_heavy_chain_seq", "Ab_light_chain_seq", "WT_Abname", "Dataset"]],
        ],
        ignore_index=True,
    ).drop_duplicates("AbID")
    combined_ab["Ab_lineage_joint"] = assign_antibody_lineages(combined_ab, identity_threshold=0.80)
    abid_to_joint_lin = combined_ab.set_index("AbID")["Ab_lineage_joint"].to_dict()
    kd_lookup_tf = kd_lookup.copy()
    kd_lookup_tf["Ab_lineage"] = [abid_to_joint_lin.get(i, l) for i, l in zip(kd_lookup_tf["AbID"], kd_lookup_tf["Ab_lineage"])]

    clust_to_groups: dict[str, set[str]] = {}
    for grp, clust in kd_raw[["Ag_group", "Ag_cluster"]].drop_duplicates().itertuples(index=False):
        clust_to_groups.setdefault(str(clust), set()).add(str(grp))
    ag_to_homolog_groups = {
        str(grp): set(clust_to_groups[str(clust)])
        for grp, clust in kd_raw[["Ag_group", "Ag_cluster"]].drop_duplicates().itertuples(index=False)
    }

    kd_pool_parts = [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test", "test_candidates")]
    kd_pool_df = pd.concat(kd_pool_parts, ignore_index=True).drop_duplicates(["AbAgID_1", "AbAgID_2"]).reset_index(drop=True)
    kd_pool_tp_1d = assemble_transfer_tensors(kd_pool_df, kd_lookup_tf, esmc_cache, ablang_cache)
    X_kd_1d = build_transfer_odd_matrix(kd_pool_tp_1d, include_biophys=True)
    kd_ags_present = set(kd_pool_df["Ag_group_1"]).union(kd_pool_df["Ag_group_2"])
    kd_lins_present = set(kd_lookup_tf.loc[kd_pool_df["AbAgID_1"], "Ab_lineage"]).union(
        set(kd_lookup_tf.loc[kd_pool_df["AbAgID_2"], "Ab_lineage"])
    )

    struct_track_specs = [
        ("DDG_SKEMPIv2", "SKEMPI 2.0 Thermodynamic ΔΔG", skempi_df, kd_lookup_tf, True),
        ("DMS_AbAgym", "AbAgym Interface DMS", dms_df, cma_lookup, False),
    ]
    viral_specs = [
        ("ic50", "Viral Neutralization IC50 (HIV-1 Env + CoV Spike)", ic50_df),
        ("escape_bloom", "SARS-CoV-2 RBD Escape DMS (Bloom)", esc_df),
    ]

    # Collect unique (eff_ags, excl_skempi) fold keys across both 3D tracks
    fold_key_list: list[tuple[frozenset[str], bool]] = []
    track_folds_map: dict[str, list[tuple[np.ndarray, np.ndarray, tuple[frozenset[str], bool]]]] = {}
    lin_src_keys: list[tuple[frozenset[str], frozenset[str], bool]] = []
    for t_key, _, df_t, _, excl_sk in struct_track_specs:
        grps = df_t["Ag_group_1"].astype(str).to_numpy()
        gkf = GroupKFold(n_splits=5)
        flist = []
        for tr_i, te_i in gkf.split(df_t, df_t["label"], grps):
            held_ags: set[str] = set()
            for g in set(df_t.iloc[te_i]["Ag_group_1"].astype(str)) | set(df_t.iloc[te_i]["Ag_group_2"].astype(str)):
                held_ags.update(ag_to_homolog_groups.get(g, {g}))
            fk = (frozenset(held_ags & kd_ags_present), bool(excl_sk))
            if fk not in fold_key_list:
                fold_key_list.append(fk)
            lk = (fk[0], frozenset(), fk[1])
            if lk not in lin_src_keys:
                lin_src_keys.append(lk)
            flist.append((tr_i, te_i, fk))
        track_folds_map[t_key] = flist

    viral_folds_map: dict[str, list[tuple[np.ndarray, np.ndarray, frozenset[str]]]] = {}
    for v_key, _, df_v in viral_specs:
        grps_v = np.array([abid_to_joint_lin[tfa_lookup.loc[a, "AbID"]] for a in df_v["AbAgID_1"]], dtype=object)
        gkf_v = GroupKFold(n_splits=5)
        vlist = []
        for tr_i, te_i in gkf_v.split(df_v, df_v["label"], grps_v):
            eff_lins = frozenset(set(grps_v[te_i].tolist()) & kd_lins_present)
            lk = (frozenset(), eff_lins, False)
            if lk not in lin_src_keys:
                lin_src_keys.append(lk)
            vlist.append((tr_i, te_i, eff_lins))
        viral_folds_map[v_key] = vlist

    neural_src_cache: dict[tuple[frozenset[str], bool], tuple[Any, Any]] = {}
    if "neural_src_cache" in ckpt_obj:
        print("  Restored Step 4 ANTIPASTI/GearBind source models from stage checkpoint!", flush=True)
        neural_src_cache = ckpt_obj["neural_src_cache"]
    else:
        kd_pool_sf = assemble_structure_pair_features(kd_pool_df, kd_lookup_tf, complex_cache, inv_cache, None, None, build_patches_and_dccm=True)
        gb_kd_pool = (
            GearBind3StateTrainer._precompute_graphs(kd_pool_sf.patch_ca_1, kd_pool_sf.patch_cb_1, kd_pool_sf.patch_chain_1, kd_pool_sf.patch_seqpos_1),
            GearBind3StateTrainer._precompute_graphs(kd_pool_sf.patch_ca_2, kd_pool_sf.patch_cb_2, kd_pool_sf.patch_chain_2, kd_pool_sf.patch_seqpos_2),
        )
        _G.clear()
        _G.update(kd_pool_df=kd_pool_df, kd_pool_tp=kd_pool_tp_1d, kd_pool_sf=kd_pool_sf, gb_kd_pool=gb_kd_pool)
        tf_jobs = [(kid, fk[0], fk[1]) for kid, fk in enumerate(fold_key_list)]
        print(f"  Fitting {len(tf_jobs)} unique leak-free 3D ANTIPASTI & GearBind KD source models in parallel...", flush=True)
        with mp.get_context("fork").Pool(min(10, len(tf_jobs))) as pool:
            for kid, m_anti, m_gb, dt_k in pool.imap_unordered(_transfer_3d_fold_worker, tf_jobs):
                neural_src_cache[fold_key_list[kid]] = (m_anti, m_gb)
                print(f"    [3D KD Source {kid + 1}/{len(tf_jobs)}] done in {dt_k:.1f}s", flush=True)
        _G.clear()
        del kd_pool_sf, gb_kd_pool
        gc.collect()
        if ckpt_path is not None:
            ckpt_obj["neural_src_cache"] = neural_src_cache
            torch.save(ckpt_obj, ckpt_path)

    # Fit all unique 1D linear KD source rankers across SKEMPI, AbAgym, IC50, and Escape in 1 parallel wave
    lin_src_cache: dict[tuple[frozenset[str], frozenset[str], bool], KDSourceRanker] = {}
    _G.clear()
    _G.update(kd_pool_df=kd_pool_df, kd_pool_tp=kd_pool_tp_1d, X_pool=X_kd_1d, kd_lookup_tf=kd_lookup_tf)
    lin_jobs = [(kid, lk[0], lk[1], lk[2]) for kid, lk in enumerate(lin_src_keys)]
    print(f"  Fitting {len(lin_jobs)} unique leak-free 1D KDSourceRanker models in 1 parallel wave...", flush=True)
    with mp.get_context("fork").Pool(min(16, len(lin_jobs))) as pool:
        for kid, src_obj, dt_l in pool.imap_unordered(_linear_kd_src_worker, lin_jobs):
            lin_src_cache[lin_src_keys[kid]] = src_obj
            print(f"    [1D KD Source {kid + 1}/{len(lin_jobs)}] done in {dt_l:.1f}s", flush=True)
    _G.clear()

    s10_path = res_dir / "structure_dms_ddg_benchmarks.json"
    s10_data = json.loads(s10_path.read_text(encoding="utf-8"))

    structure_tracks_out: dict[str, Any] = {}
    fewshot_stratified_out: dict[str, Any] = {}

    for t_key, t_title, df_t, lkp, excl_sk in struct_track_specs:
        print(f"\n  Evaluating full 1D & 3D suite on {t_title} (n={len(df_t)})...", flush=True)
        grps = df_t["Ag_group_1"].astype(str).to_numpy()
        tp_1d = assemble_transfer_tensors(df_t, lkp, esmc_cache, ablang_cache, esm2_cache)
        sf_t = assemble_structure_pair_features(df_t, lkp, complex_cache, inv_cache, None, None, build_patches_and_dccm=True)
        gb_t = (
            GearBind3StateTrainer._precompute_graphs(sf_t.patch_ca_1, sf_t.patch_cb_1, sf_t.patch_chain_1, sf_t.patch_seqpos_1),
            GearBind3StateTrainer._precompute_graphs(sf_t.patch_ca_2, sf_t.patch_cb_2, sf_t.patch_chain_2, sf_t.patch_seqpos_2),
        )
        X_1d = build_transfer_odd_matrix(tp_1d, include_biophys=True)
        n = len(df_t)

        oof: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "AbLang2-Paired (1D ZS, Ab-Only)": (tp_1d.zs_dict["ablang2_zeroshot"], tp_1d.zs_impact_dict["ablang2_zeroshot"]),
            "ESMC-300M (1D ZS, Complex)": (tp_1d.zs_dict["esmc_300m_zeroshot"], tp_1d.zs_impact_dict["esmc_300m_zeroshot"]),
            "ESMC+AbLang2 (1D ZS)": (tp_1d.zs_dict["esmc_ablang2_zeroshot"], tp_1d.zs_impact_dict["esmc_ablang2_zeroshot"]),
            "PRODIGY+SASA (3D Energy ZS)": (sf_t.zs_dir["prodigy_sasa"], sf_t.zs_imp["prodigy_sasa"]),
            "ProteinMPNN (3D Complex ZS)": (sf_t.zs_dir["protein_mpnn"], sf_t.zs_imp["protein_mpnn"]),
            "AntiFold (3D Ab-Ag ZS)": (sf_t.zs_dir["antifold"], sf_t.zs_imp["antifold"]),
            "ESM-IF1 (3D Complex ZS)": (sf_t.zs_dir["esm_if1"], sf_t.zs_imp["esm_if1"]),
        }
        sup_names = [
            "Direct 1D KD Transfer (ESMC Hurdle)",
            "Direct 3D KD Transfer (ANTIPASTI)",
            "Direct 3D KD Transfer (GearBind)",
            "Target From Scratch (1D Hurdle)",
            "1D KD -> Target Fine-Tuned (L2-SP)",
            "1D KD -> Target Fine-Tuned (Residual Adapter)",
        ]
        for sn in sup_names:
            oof[sn] = (np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64))

        fs_fracs = (0.10, 0.25, 0.50, 1.00)
        fs_scr_d = {fr: np.zeros(n, dtype=np.float64) for fr in fs_fracs}
        fs_scr_i = {fr: np.zeros(n, dtype=np.float64) for fr in fs_fracs}
        fs_l2_d = {fr: np.zeros(n, dtype=np.float64) for fr in fs_fracs}
        fs_l2_i = {fr: np.zeros(n, dtype=np.float64) for fr in fs_fracs}

        for fold_i, (tr_i, te_i, fk) in enumerate(track_folds_map[t_key]):
            kd_1d_src = lin_src_cache[(fk[0], frozenset(), fk[1])]
            m_anti, m_gb = neural_src_cache[fk]

            te_tp_1d = tp_1d.subset(te_i)
            te_sf = sf_t.subset(te_i)
            te_gb = _slice_gb(gb_t, te_i)

            d, q = kd_1d_src.predict_3state(te_tp_1d, X_1d[te_i])
            oof["Direct 1D KD Transfer (ESMC Hurdle)"][0][te_i], oof["Direct 1D KD Transfer (ESMC Hurdle)"][1][te_i] = d, q

            d, q = m_anti.predict_3state(te_sf, te_tp_1d)
            oof["Direct 3D KD Transfer (ANTIPASTI)"][0][te_i], oof["Direct 3D KD Transfer (ANTIPASTI)"][1][te_i] = d, q

            d, q = m_gb.predict_3state(te_sf, _precomp_g=te_gb)
            oof["Direct 3D KD Transfer (GearBind)"][0][te_i], oof["Direct 3D KD Transfer (GearBind)"][1][te_i] = d, q

            tr_tp_1d = tp_1d.subset(tr_i)
            m_scr = L2SPFineTunedRanker(kd_source=None).fit(X_1d[tr_i], tr_tp_1d.y, grps[tr_i], tp_tr=tr_tp_1d)
            d_sc, q_sc = m_scr.predict_3state(te_tp_1d, X_1d[te_i])
            oof["Target From Scratch (1D Hurdle)"][0][te_i], oof["Target From Scratch (1D Hurdle)"][1][te_i] = d_sc, q_sc

            m_ft1 = L2SPFineTunedRanker(kd_source=kd_1d_src).fit(X_1d[tr_i], tr_tp_1d.y, grps[tr_i], tp_tr=tr_tp_1d)
            d_ft1, q_ft1 = m_ft1.predict_3state(te_tp_1d, X_1d[te_i])
            oof["1D KD -> Target Fine-Tuned (L2-SP)"][0][te_i], oof["1D KD -> Target Fine-Tuned (L2-SP)"][1][te_i] = d_ft1, q_ft1

            m_adp = KDResidualAdapterRanker(kd_source=kd_1d_src, seed=42 + fold_i).fit(tr_tp_1d, X_1d[tr_i], grps[tr_i])
            d_ad, q_ad = m_adp.predict_3state(te_tp_1d, X_1d[te_i])
            oof["1D KD -> Target Fine-Tuned (Residual Adapter)"][0][te_i], oof["1D KD -> Target Fine-Tuned (Residual Adapter)"][1][te_i] = d_ad, q_ad

            if t_key == "DDG_SKEMPIv2":
                rng = np.random.default_rng(1000 + fold_i)
                for fr in fs_fracs:
                    if fr >= 0.999:
                        fs_scr_d[fr][te_i], fs_scr_i[fr][te_i] = d_sc, q_sc
                        fs_l2_d[fr][te_i], fs_l2_i[fr][te_i] = d_ft1, q_ft1
                    else:
                        nb = max(32, int(round(fr * len(tr_i))))
                        sub_tr = tr_i[rng.permutation(len(tr_i))[:nb]]
                        tp_sub = tp_1d.subset(sub_tr)
                        s_f = L2SPFineTunedRanker(kd_source=None).fit(X_1d[sub_tr], tp_sub.y, grps[sub_tr], tp_tr=tp_sub)
                        l_f = L2SPFineTunedRanker(kd_source=kd_1d_src).fit(X_1d[sub_tr], tp_sub.y, grps[sub_tr], tp_tr=tp_sub)
                        fs_scr_d[fr][te_i], fs_scr_i[fr][te_i] = s_f.predict_3state(te_tp_1d, X_1d[te_i])
                        fs_l2_d[fr][te_i], fs_l2_i[fr][te_i] = l_f.predict_3state(te_tp_1d, X_1d[te_i])

        ab_m = (~tp_1d.is_ag_mut)
        ag_m = tp_1d.is_ag_mut
        sub_ab_df = df_t.iloc[np.flatnonzero(ab_m)]
        sub_ag_df = df_t.iloc[np.flatnonzero(ag_m)]

        raw_methods: dict[str, Any] = {}
        for m_name, (dv, qv) in oof.items():
            all_r = compute_per_antigen_macro_metrics(df_t, dv, qv, min_pairs=5, n_bootstrap=500)
            ab_r = compute_per_antigen_macro_metrics(sub_ab_df, dv[ab_m], qv[ab_m], min_pairs=5, n_bootstrap=500)
            ag_r = compute_per_antigen_macro_metrics(sub_ag_df, dv[ag_m], qv[ag_m], min_pairs=5, n_bootstrap=500)
            raw_methods[m_name] = {
                "all": all_r,
                "ab_mut": ab_r,
                "ag_mut": ag_r,
                "delta_ord_ag_minus_ab": round(float(ag_r["auc_ordinal"] - ab_r["auc_ordinal"]), 4),
                "delta_sign_ag_minus_ab": round(float(ag_r["auc_dir"] - ab_r["auc_dir"]), 4),
            }

        # Include verified WALLE & LogiCA slices from Step 10 (`structure_dms_ddg_benchmarks.json`)
        for ref_k, disp_name in (
            ("walle_3d_direct", "Direct 3D KD Transfer (WALLE-Affinity)"),
            ("walle_3d_l2sp", "3D KD -> Target Fine-Tuned (WALLE L2-SP)"),
            ("logica_1d_direct", "Direct 1D KD Transfer (LogiCA)"),
        ):
            if ref_k not in s10_data["tracks"][t_key]["methods"]:
                continue
            all_r = s10_data["tracks"][t_key]["methods"][ref_k]
            ab_r = s10_data["tracks"][t_key]["slices"]["Ab-Mut"][ref_k]
            ag_r = s10_data["tracks"][t_key]["slices"]["Ag-Mut"][ref_k]
            raw_methods[disp_name] = {
                "all": all_r,
                "ab_mut": ab_r,
                "ag_mut": ag_r,
                "delta_ord_ag_minus_ab": round(float(ag_r["auc_ordinal"] - ab_r["auc_ordinal"]), 4),
                "delta_sign_ag_minus_ab": round(float(ag_r["auc_dir"] - ab_r["auc_dir"]), 4),
            }

        ordered_struct_names = [
            "AbLang2-Paired (1D ZS, Ab-Only)",
            "ESMC-300M (1D ZS, Complex)",
            "ESMC+AbLang2 (1D ZS)",
            "PRODIGY+SASA (3D Energy ZS)",
            "ProteinMPNN (3D Complex ZS)",
            "AntiFold (3D Ab-Ag ZS)",
            "ESM-IF1 (3D Complex ZS)",
            "Direct 1D KD Transfer (ESMC Hurdle)",
            "Direct 1D KD Transfer (LogiCA)",
            "Direct 3D KD Transfer (ANTIPASTI)",
            "Direct 3D KD Transfer (GearBind)",
            "Direct 3D KD Transfer (WALLE-Affinity)",
            "Target From Scratch (1D Hurdle)",
            "1D KD -> Target Fine-Tuned (L2-SP)",
            "1D KD -> Target Fine-Tuned (Residual Adapter)",
            "3D KD -> Target Fine-Tuned (WALLE L2-SP)",
        ]
        methods_dict = {k: raw_methods[k] for k in ordered_struct_names}
        for m_name, mv in methods_dict.items():
            ab_r, ag_r = mv["ab_mut"], mv["ag_mut"]
            print(
                f"    {m_name:44s}: Ab(S/I/O)={ab_r['auc_dir']:.3f}/{ab_r['auc_impact']:.3f}/{ab_r['auc_ordinal']:.3f} | "
                f"Ag(S/I/O)={ag_r['auc_dir']:.3f}/{ag_r['auc_impact']:.3f}/{ag_r['auc_ordinal']:.3f}",
                flush=True,
            )

        structure_tracks_out[t_key] = {
            "title": t_title,
            "n_pairs": int(n),
            "n_ab_mut": int(ab_m.sum()),
            "n_ag_mut": int(ag_m.sum()),
            "n_ab_antigens_eval": int(methods_dict["ProteinMPNN (3D Complex ZS)"]["ab_mut"]["n_antigens_evaluated"]),
            "n_ag_antigens_eval": int(methods_dict["ProteinMPNN (3D Complex ZS)"]["ag_mut"]["n_antigens_evaluated"]),
            "methods": methods_dict,
        }

        # Enrich `structure_dms_ddg_benchmarks.json` `slices` with missing 1D/3D baselines
        extra_slice_map = {
            "ablang2_1d_zs": "AbLang2-Paired (1D ZS, Ab-Only)",
            "esmc_1d_zs": "ESMC-300M (1D ZS, Complex)",
            "prodigy_sasa": "PRODIGY+SASA (3D Energy ZS)",
            "antifold": "AntiFold (3D Ab-Ag ZS)",
            "antipasti_3d_direct": "Direct 3D KD Transfer (ANTIPASTI)",
            "gearbind_3d_direct": "Direct 3D KD Transfer (GearBind)",
        }
        for s_lbl, sub_k in (("Ab-Mut", "ab_mut"), ("Ag-Mut", "ag_mut")):
            if s_lbl in s10_data["tracks"][t_key].get("slices", {}):
                for sk_short, full_n in extra_slice_map.items():
                    s10_data["tracks"][t_key]["slices"][s_lbl][sk_short] = methods_dict[full_n][sub_k]

        if t_key == "DDG_SKEMPIv2":
            dkd_ab = methods_dict["Direct 1D KD Transfer (ESMC Hurdle)"]["ab_mut"]
            dkd_ag = methods_dict["Direct 1D KD Transfer (ESMC Hurdle)"]["ag_mut"]
            fs_curve = {
                "0.00": {
                    "ab_scratch_ord": 0.5000, "ab_l2sp_ord": dkd_ab["auc_ordinal"],
                    "ag_scratch_ord": 0.5000, "ag_l2sp_ord": dkd_ag["auc_ordinal"],
                }
            }
            for fr in fs_fracs:
                fs_curve[f"{fr:.2f}"] = {
                    "ab_scratch_ord": compute_per_antigen_macro_metrics(sub_ab_df, fs_scr_d[fr][ab_m], fs_scr_i[fr][ab_m], min_pairs=5)["auc_ordinal"],
                    "ab_l2sp_ord": compute_per_antigen_macro_metrics(sub_ab_df, fs_l2_d[fr][ab_m], fs_l2_i[fr][ab_m], min_pairs=5)["auc_ordinal"],
                    "ag_scratch_ord": compute_per_antigen_macro_metrics(sub_ag_df, fs_scr_d[fr][ag_m], fs_scr_i[fr][ag_m], min_pairs=5)["auc_ordinal"],
                    "ag_l2sp_ord": compute_per_antigen_macro_metrics(sub_ag_df, fs_l2_d[fr][ag_m], fs_l2_i[fr][ag_m], min_pairs=5)["auc_ordinal"],
                }
            fewshot_stratified_out["DDG_SKEMPIv2"] = fs_curve

    s10_path.write_text(json.dumps(s10_data, indent=2) + "\n", encoding="utf-8")

    print("\n5. Running 5-Fold Unseen-Lineage GroupCV on Viral IC50 & SARS-CoV-2 RBD Escape...", flush=True)
    mt_ref_data = json.loads((res_dir / "modality_transfer_and_finetuning.json").read_text(encoding="utf-8"))
    mt_ref_by_key = {t["track_key"]: t for t in mt_ref_data["tracks"]}
    viral_tracks_out: dict[str, Any] = {}

    for v_key, v_title, df_v in viral_specs:
        grps_v = np.array([abid_to_joint_lin[tfa_lookup.loc[a, "AbID"]] for a in df_v["AbAgID_1"]], dtype=object)
        tp_v = assemble_transfer_tensors(df_v, tfa_lookup, esmc_cache, ablang_cache, esm2_cache)
        X_v = build_transfer_odd_matrix(tp_v, include_biophys=True)
        nv = len(df_v)

        oof_v: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "AbLang2-Paired (1D ZS, Ab-Only)": (tp_v.zs_dict["ablang2_zeroshot"], tp_v.zs_impact_dict["ablang2_zeroshot"]),
            "ESMC-300M (1D ZS, Complex)": (tp_v.zs_dict["esmc_300m_zeroshot"], tp_v.zs_impact_dict["esmc_300m_zeroshot"]),
            "ESMC+AbLang2 (1D ZS)": (tp_v.zs_dict["esmc_ablang2_zeroshot"], tp_v.zs_impact_dict["esmc_ablang2_zeroshot"]),
        }
        v_sup = [
            "Direct 1D KD Transfer (ESMC Hurdle)",
            "Target From Scratch (1D Hurdle)",
            "1D KD -> Target Fine-Tuned (L2-SP)",
            "1D KD -> Target Fine-Tuned (Residual Adapter)",
        ]
        for sn in v_sup:
            oof_v[sn] = (np.zeros(nv, dtype=np.float64), np.zeros(nv, dtype=np.float64))

        fs_fracs = (0.10, 0.25, 0.50, 1.00)
        fs_scr_d = {fr: np.zeros(nv, dtype=np.float64) for fr in fs_fracs}
        fs_scr_i = {fr: np.zeros(nv, dtype=np.float64) for fr in fs_fracs}
        fs_l2_d = {fr: np.zeros(nv, dtype=np.float64) for fr in fs_fracs}
        fs_l2_i = {fr: np.zeros(nv, dtype=np.float64) for fr in fs_fracs}

        for fold_i, (tr_i, te_i, eff_lins) in enumerate(viral_folds_map[v_key]):
            kd_src = lin_src_cache[(frozenset(), eff_lins, False)]

            te_tp = tp_v.subset(te_i)
            tr_tp = tp_v.subset(tr_i)

            d, q = kd_src.predict_3state(te_tp, X_v[te_i])
            oof_v["Direct 1D KD Transfer (ESMC Hurdle)"][0][te_i], oof_v["Direct 1D KD Transfer (ESMC Hurdle)"][1][te_i] = d, q

            m_scr = L2SPFineTunedRanker(kd_source=None).fit(X_v[tr_i], tr_tp.y, grps_v[tr_i], tp_tr=tr_tp)
            d_sc, q_sc = m_scr.predict_3state(te_tp, X_v[te_i])
            oof_v["Target From Scratch (1D Hurdle)"][0][te_i], oof_v["Target From Scratch (1D Hurdle)"][1][te_i] = d_sc, q_sc

            m_ft = L2SPFineTunedRanker(kd_source=kd_src).fit(X_v[tr_i], tr_tp.y, grps_v[tr_i], tp_tr=tr_tp)
            d_ft, q_ft = m_ft.predict_3state(te_tp, X_v[te_i])
            oof_v["1D KD -> Target Fine-Tuned (L2-SP)"][0][te_i], oof_v["1D KD -> Target Fine-Tuned (L2-SP)"][1][te_i] = d_ft, q_ft

            m_adp = KDResidualAdapterRanker(kd_source=kd_src, seed=42 + fold_i).fit(tr_tp, X_v[tr_i], grps_v[tr_i])
            d_ad, q_ad = m_adp.predict_3state(te_tp, X_v[te_i])
            oof_v["1D KD -> Target Fine-Tuned (Residual Adapter)"][0][te_i], oof_v["1D KD -> Target Fine-Tuned (Residual Adapter)"][1][te_i] = d_ad, q_ad

            if v_key == "ic50":
                rng = np.random.default_rng(1000 + fold_i)
                for fr in fs_fracs:
                    if fr >= 0.999:
                        fs_scr_d[fr][te_i], fs_scr_i[fr][te_i] = d_sc, q_sc
                        fs_l2_d[fr][te_i], fs_l2_i[fr][te_i] = d_ft, q_ft
                    else:
                        nb = max(32, int(round(fr * len(tr_i))))
                        sub_tr = tr_i[rng.permutation(len(tr_i))[:nb]]
                        tp_sub = tp_v.subset(sub_tr)
                        s_f = L2SPFineTunedRanker(kd_source=None).fit(X_v[sub_tr], tp_sub.y, grps_v[sub_tr], tp_tr=tp_sub)
                        l_f = L2SPFineTunedRanker(kd_source=kd_src).fit(X_v[sub_tr], tp_sub.y, grps_v[sub_tr], tp_tr=tp_sub)
                        fs_scr_d[fr][te_i], fs_scr_i[fr][te_i] = s_f.predict_3state(te_tp, X_v[te_i])
                        fs_l2_d[fr][te_i], fs_l2_i[fr][te_i] = l_f.predict_3state(te_tp, X_v[te_i])

        ab_m = tp_v.is_ab_mut & (~tp_v.is_ag_mut)
        ag_m = tp_v.is_ag_mut & (~tp_v.is_ab_mut)
        sub_ab_df = df_v.iloc[np.flatnonzero(ab_m)] if ab_m.sum() >= 6 else None
        sub_ag_df = df_v.iloc[np.flatnonzero(ag_m)] if ag_m.sum() >= 6 else None

        raw_v_methods: dict[str, Any] = {}
        for m_name, (dv, qv) in oof_v.items():
            all_r = compute_per_antigen_macro_metrics(df_v, dv, qv, min_pairs=5, n_bootstrap=500)
            ab_r = compute_per_antigen_macro_metrics(sub_ab_df, dv[ab_m], qv[ab_m], min_pairs=5, n_bootstrap=500) if sub_ab_df is not None else None
            ag_r = compute_per_antigen_macro_metrics(sub_ag_df, dv[ag_m], qv[ag_m], min_pairs=5, n_bootstrap=500) if sub_ag_df is not None else None
            raw_v_methods[m_name] = {
                "all": all_r,
                "ab_mut": ab_r,
                "ag_mut": ag_r,
            }

        mt_log = mt_ref_by_key[v_key]["methods"]["Direct KD Transfer (LogiCA 1D)"]
        raw_v_methods["Direct 1D KD Transfer (LogiCA)"] = {
            "all": {
                "auc_dir": mt_log["auc_dir"],
                "auc_impact": mt_log["auc_impact"],
                "auc_ordinal": mt_log["auc_ordinal"],
                "pooled_auc_dir": mt_log.get("pooled_auc_dir"),
                "pooled_auc_impact": mt_log.get("pooled_auc_impact"),
                "pooled_auc_ordinal": mt_log.get("pooled_auc_ordinal"),
            },
            "ab_mut": (
                {"auc_dir": mt_log["ab_mut_dir"], "auc_impact": mt_log["ab_mut_imp"], "auc_ordinal": mt_log["ab_mut_ord"]}
                if mt_log.get("ab_mut_ord") is not None
                else None
            ),
            "ag_mut": (
                {"auc_dir": mt_log["ag_mut_dir"], "auc_impact": mt_log["ag_mut_imp"], "auc_ordinal": mt_log["ag_mut_ord"]}
                if mt_log.get("ag_mut_ord") is not None
                else None
            ),
        }

        ordered_v_names = [
            "AbLang2-Paired (1D ZS, Ab-Only)",
            "ESMC-300M (1D ZS, Complex)",
            "ESMC+AbLang2 (1D ZS)",
            "Direct 1D KD Transfer (ESMC Hurdle)",
            "Direct 1D KD Transfer (LogiCA)",
            "Target From Scratch (1D Hurdle)",
            "1D KD -> Target Fine-Tuned (L2-SP)",
            "1D KD -> Target Fine-Tuned (Residual Adapter)",
        ]
        v_methods = {k: raw_v_methods[k] for k in ordered_v_names}
        for m_name, mv in v_methods.items():
            ab_r, ag_r = mv["ab_mut"], mv["ag_mut"]
            ab_str = f"{ab_r['auc_dir']:.3f}/{ab_r['auc_impact']:.3f}/{ab_r['auc_ordinal']:.3f}" if ab_r else "—"
            ag_str = f"{ag_r['auc_dir']:.3f}/{ag_r['auc_impact']:.3f}/{ag_r['auc_ordinal']:.3f}" if ag_r else "—"
            print(f"    [{v_key:12s}] {m_name:44s}: Ab={ab_str:17s} | Ag={ag_str}", flush=True)

        viral_tracks_out[v_key] = {
            "title": v_title,
            "n_pairs": int(nv),
            "n_ab_mut": int(ab_m.sum()),
            "n_ag_mut": int(ag_m.sum()),
            "methods": v_methods,
        }

        if v_key == "ic50" and sub_ab_df is not None and sub_ag_df is not None:
            dkd_ab = v_methods["Direct 1D KD Transfer (ESMC Hurdle)"]["ab_mut"]
            dkd_ag = v_methods["Direct 1D KD Transfer (ESMC Hurdle)"]["ag_mut"]
            fs_curve = {
                "0.00": {
                    "ab_scratch_ord": 0.5000, "ab_l2sp_ord": dkd_ab["auc_ordinal"],
                    "ag_scratch_ord": 0.5000, "ag_l2sp_ord": dkd_ag["auc_ordinal"],
                }
            }
            for fr in fs_fracs:
                fs_curve[f"{fr:.2f}"] = {
                    "ab_scratch_ord": compute_per_antigen_macro_metrics(sub_ab_df, fs_scr_d[fr][ab_m], fs_scr_i[fr][ab_m], min_pairs=5)["auc_ordinal"],
                    "ab_l2sp_ord": compute_per_antigen_macro_metrics(sub_ab_df, fs_l2_d[fr][ab_m], fs_l2_i[fr][ab_m], min_pairs=5)["auc_ordinal"],
                    "ag_scratch_ord": compute_per_antigen_macro_metrics(sub_ag_df, fs_scr_d[fr][ag_m], fs_scr_i[fr][ag_m], min_pairs=5)["auc_ordinal"],
                    "ag_l2sp_ord": compute_per_antigen_macro_metrics(sub_ag_df, fs_l2_d[fr][ag_m], fs_l2_i[fr][ag_m], min_pairs=5)["auc_ordinal"],
                }
            fewshot_stratified_out["ic50"] = fs_curve

    payload = {
        "benchmark": "AbRank v2.0.0",
        "experiment": "Paratope (Ab-Mut) vs. Epitope (Ag-Mut) Mutation Stratification & Two-Sided Generalization",
        "inventory": inventory,
        "shared_loaco_cocrystal_clusters": shared_ab_ag_clusters,
        "loaco_local": loaco_summary,
        "structure_tracks": structure_tracks_out,
        "viral_tracks": viral_tracks_out,
        "fewshot_stratified": fewshot_stratified_out,
    }

    out_json = res_dir / "paratope_vs_epitope_mutations.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Build Markdown summary report
    md = [
        "# AbRank v2 — Paratope (`Ab-Mut`) vs. Epitope (`Ag-Mut`) Mutation Stratification",
        "",
        "Disaggregates antibody-side paratope/framework edits (`Ab-Mut`: `ΔAb > 0, ΔAg = 0`) from antigen-side epitope/escape edits (`Ag-Mut`: `ΔAb = 0, ΔAg > 0`) across all four AbRank tracks under `<50%` antigen-cluster and `<80%` antibody clonal-lineage holdouts.",
        "",
        "## 1. Benchmark-Wide Mutation-Side Inventory",
        "",
        "| Suite / Modality | Track | Total Pairs | `Ab-Mut` Only (%) | `Ag-Mut` Only (%) | `Both` / Global (%) | Unique Abs / Ags |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in inventory:
        md.append(
            f"| **{r['name']}** | {r['track']} | `{r['n_pairs']:,}` | "
            f"`{r['n_ab_only']:,}` (`{r['pct_ab_only']:.1f}%`) | "
            f"`{r['n_ag_only']:,}` (`{r['pct_ag_only']:.1f}%`) | "
            f"`{r['n_both_global']:,}` (`{r['pct_both_global']:.1f}%`) | "
            f"`{r['n_unique_ab_seqs']:,}` / `{r['n_unique_ag_seqs']:,}` |"
        )

    n_ab_cl = loaco_summary["ProteinMPNN (3D Complex ZS)"]["ab_mut_22_clusters"]["n_clusters_total"]
    n_ag_cl = loaco_summary["ProteinMPNN (3D Complex ZS)"]["ag_mut_clusters"]["n_clusters_total"]
    n_sh_cl = len(shared_ab_ag_clusters)
    md.extend([
        "",
        f"## 2. Core `KD` 22-Cluster `LOACO-Local` (`11,533 Ab-Mut` vs. `734 Ag-Mut` + Matched {n_sh_cl}-Cluster Co-Crystal Control)",
        "",
        "Cells report **Cluster-Macro `Sign AUC` / `Impact AUC` / `3-Class Ordinal AUC`** (`≥5` pairs per cluster):",
        "",
        f"| Model | Family | All LOACO (`22` cl.) | `Ab-Mut` (`{n_ab_cl}` cl.) | `Ag-Mut` (`{n_ag_cl}` cl.) | Matched `Ab-Mut` (`{n_sh_cl}` cl.) | Matched `Ag-Mut` (`{n_sh_cl}` cl.) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for m_name, fam in LOACO_MODEL_SPECS:
        ms = loaco_summary[m_name]
        md.append(
            f"| **{m_name}** | {fam} | {_fmt_macro(ms['all_22_clusters'])} | "
            f"{_fmt_macro(ms['ab_mut_22_clusters'])} | {_fmt_macro(ms['ag_mut_clusters'])} | "
            f"{_fmt_macro(ms['matched_shared_clusters_ab_mut'])} | {_fmt_macro(ms['matched_shared_clusters_ag_mut'])} |"
        )

    for t_key, sv in structure_tracks_out.items():
        md.extend([
            "",
            f"## 3.{1 if t_key == 'DDG_SKEMPIv2' else 2} {sv['title']} (`{sv['n_ab_mut']:,} Ab-Mut` vs. `{sv['n_ag_mut']:,} Ag-Mut`; 5-Fold Unseen-Antigen GroupCV)",
            "",
            "| Model | All Pairs (`Sign / Imp / Ord`) | `Ab-Mut` (`Sign / Imp / Ord`) | `Ag-Mut` (`Sign / Imp / Ord`) | `ΔOrd (Ag - Ab)` |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for m_name, mv in sv["methods"].items():
            md.append(
                f"| **{m_name}** | {_fmt_macro(mv['all'])} | {_fmt_macro(mv['ab_mut'])} | "
                f"{_fmt_macro(mv['ag_mut'])} | `{mv['delta_ord_ag_minus_ab']:+.3f}` |"
            )

    md.extend([
        "",
        "## 4. Viral Functional Transfer: `IC50` (`886 Ab-Mut` vs. `2,081 Ag-Mut`) & `RBD Escape` (`8,336 Ag-Mut`; 5-Fold Unseen-Lineage GroupCV)",
        "",
        "| Model | `IC50 Ab-Mut` (`Sign / Imp / Ord`) | `IC50 Ag-Mut` (`Sign / Imp / Ord`) | `RBD Escape Ag-Mut` (`Sign / Imp / Ord`) |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for m_name in viral_tracks_out["ic50"]["methods"]:
        ic_m = viral_tracks_out["ic50"]["methods"][m_name]
        es_m = viral_tracks_out["escape_bloom"]["methods"][m_name]
        md.append(
            f"| **{m_name}** | {_fmt_macro(ic_m['ab_mut'])} | {_fmt_macro(ic_m['ag_mut'])} | {_fmt_macro(es_m['ag_mut'])} |"
        )

    out_md = res_dir / "PARATOPE_VS_EPITOPE_MUTATIONS.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nSaved {out_json} and {out_md} in {time.time() - t_start:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
