#!/usr/bin/env python3
"""Step 13: Empirical Validation of the 3-State Noise-Buffered Formulation, Margin &
Holdout Sensitivity, and Practical Wet-Lab Hit Selection vs. Scalar Regression / Binary Probes.

Addresses three core methodological questions:
  1. Part A — Empirical Replicate & Inter-Platform Uncertainty Calibration:
       Quantifies |ΔpKd| dispersion across technical SPR replicates (`AIntibody` HT-SPR,
       `SKEMPI 2.0` duplicates) and cross-platform / multi-crystal measurements (`AIntibody`
       paired `SPR` vs. `KinExA`, `SAbDab` independent co-crystals, cross-study complexes),
       validating `m_neu = 0.30` (≤2x replicate equivalence) and `m = 1.0` (≥10x signal).
  2. Part B — Margin `(m_neu, m)` & Identity-Cutoff Sensitivity Analysis:
       Evaluates model rankings across four `(m_neu, m)` threshold grids and three
       sequence-identity holdout stringencies (`Loose 70%/90%`, `Canonical 50%/80%`,
       `Ultra-Strict 35%/70%`), quantifying leaderboard Spearman rank stability.
  3. Part C — Practical Wet-Lab Hit Selection (`Precision@Top-20%`, `Precision@Top-3`,
       `Neutral/Deleterious FDR`, & Mean `ΔpKd` Gain) on Unfiltered Variant Panels:
       Compares Continuous Pointwise `pKd` Regression (Ridge & MLP), Impact-Blind Binary
       Ranking, 1D 3-State Hurdle, and 3D Structure Models under strict Leave-One-Antigen-
       Cluster-Out (`LOACO`) across all 22 cached unfiltered variant panels (`1,242` variants).

Outputs:
  - `results/three_state_validation.json`
  - `results/THREE_STATE_VALIDATION.md`
"""

from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

from itertools import combinations
import gc
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import Levenshtein
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch
torch.set_num_threads(1)

from abrank.baselines.embeddings import pair_hash, seq_hash
from abrank.baselines.models import (
    AntisymmetricLinearRanker,
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
    assemble_structure_pair_features,
    fit_3state_foundation_hurdle,
)
from abrank.config import get_repo_root
from abrank.evaluate import compute_per_antigen_macro_metrics, evaluate_pointwise_spearman
from abrank.schema import PAIR_COLUMNS, kd_nm_to_pkd, parse_numeric_measurement, trim_antigen_tags, trim_fab_to_fv


def compute_empirical_replicate_uncertainty(root: Path) -> dict:
    """Measure empirical |ΔpKd| across within-platform technical replicates and cross-platform assays."""
    raw = root / "data" / "raw"
    pdir = root / "data" / "processed"

    # 1. AIntibody HT-SPR technical replicates + reported per-clone SPR SD + paired SPR vs KinExA!
    xl = pd.ExcelFile(raw / "AIntibody" / "41587_2026_3238_MOESM4_ESM.xlsx")
    d3 = xl.parse("Dataset 3", header=1)
    d3["kd_spr"] = pd.to_numeric(d3["KD_SPR"], errors="coerce")
    d3["sd_spr"] = pd.to_numeric(d3["STDEV_SPR"], errors="coerce")
    d3["kd_kin"] = pd.to_numeric(d3["KD_KinExA"], errors="coerce")

    ai_tech_pairs = []
    d3_spr = d3[d3["kd_spr"].notna() & (d3["kd_spr"] > 0) & (d3["kd_spr"] <= 1e-6)].copy()
    d3_spr["pkd_spr"] = -np.log10(d3_spr["kd_spr"].to_numpy())
    for _, g in d3_spr.groupby(["sequence_aa_heavy", "sequence_aa_light"]):
        if len(g) >= 2:
            v = g["pkd_spr"].to_numpy()
            for i, j in combinations(range(len(v)), 2):
                ai_tech_pairs.append(float(abs(v[i] - v[j])))

    # Expected pairwise |ΔpKd| from per-clone multi-replicate SD (`E|X1 - X2| = 2/sqrt(pi) * sigma_log10`)
    valid_cv = d3_spr[d3_spr["sd_spr"].notna() & (d3_spr["sd_spr"] > 0)]
    sigma_log10 = np.log10(1.0 + (valid_cv["sd_spr"] / valid_cv["kd_spr"]).clip(upper=2.0).to_numpy())
    ai_expected_rep_diff = (np.sqrt(2.0) * 0.79788456 * sigma_log10).tolist()

    # Cross-platform: exact same antibody measured by BOTH HT-SPR and solution-phase KinExA!
    both_plat = d3_spr[d3_spr["kd_kin"].notna() & (d3_spr["kd_kin"] > 0)].copy()
    spr_vs_kinexa = np.abs(-np.log10(both_plat["kd_spr"].to_numpy()) - (-np.log10(both_plat["kd_kin"].to_numpy()))).tolist()

    # 2. SKEMPI 2.0 identical-sequence entries
    sk = pd.read_csv(raw / "SKEMPIv2" / "SKEMPI_preprocessed.csv")
    sk["kd"] = pd.to_numeric(sk["Affinity_Kd [nM]"], errors="coerce")
    sk = sk[sk["kd"].notna() & (sk["kd"] > 0)].copy()
    sk["pkd"] = kd_nm_to_pkd(sk["kd"].to_numpy())
    sk_rep = []
    for _, g in sk.groupby(["Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"]):
        if len(g) >= 2:
            v = g["pkd"].to_numpy()
            for i, j in combinations(range(len(v)), 2):
                sk_rep.append(float(abs(v[i] - v[j])))

    # 3. SAbDab independent co-crystal entries + cross-source identical (VH, VL, Ag) complexes
    sb = pd.read_csv(raw / "SabDab" / "SabDab_preprocessed.csv", sep="\t")
    parsed = [parse_numeric_measurement(v) for v in sb["Affinity_Kd [nM]"]]
    sb["kd"] = [p[0] if p[1] == "=" and np.isfinite(p[0]) and p[0] > 0 else np.nan for p in parsed]
    sb = sb.dropna(subset=["kd"]).copy()
    sb["pkd"] = kd_nm_to_pkd(sb["kd"].to_numpy())
    sb["vh"] = [trim_fab_to_fv(h, l)[0] for h, l in zip(sb["Ab_heavy_chain_seq"], sb["Ab_light_chain_seq"])]
    sb["vl"] = [trim_fab_to_fv(h, l)[1] for h, l in zip(sb["Ab_heavy_chain_seq"], sb["Ab_light_chain_seq"])]
    sb["ag"] = [trim_antigen_tags(s) for s in sb["Ag_seq"]]
    sb_rep = []
    for _, g in sb.groupby(["vh", "vl", "ag"]):
        if len(g) >= 2:
            v = g["pkd"].to_numpy()
            for i, j in combinations(range(len(v)), 2):
                sb_rep.append(float(abs(v[i] - v[j])))

    kd_frames = [pd.read_csv(f, sep="\t", low_memory=False)[["Dataset", "Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq", "binding_score"]] for f in sorted(pdir.glob("kd_*.tsv"))]
    all_proc = pd.concat(kd_frames, ignore_index=True)
    cross_src = []
    for _, g in all_proc.groupby(["Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"]):
        if len(g) >= 2:
            v = g["binding_score"].to_numpy(dtype=float)
            for i, j in combinations(range(len(v)), 2):
                cross_src.append(float(abs(v[i] - v[j])))

    def _stats(arr: list[float]) -> dict:
        a = np.asarray(arr, dtype=float)
        return {
            "n": int(len(a)),
            "median": round(float(np.median(a)), 4),
            "iqr": [round(float(np.percentile(a, 25)), 4), round(float(np.percentile(a, 75)), 4)],
            "p90": round(float(np.percentile(a, 90)), 4),
            "pct_le_0_30": round(100.0 * float((a <= 0.30).mean()), 2),
            "pct_lt_1_00": round(100.0 * float((a < 1.00).mean()), 2),
        }

    tech_all = ai_tech_pairs + sk_rep + ai_expected_rep_diff
    inter_plat_all = spr_vs_kinexa + sb_rep + cross_src
    return {
        "within_platform_technical_replicates": _stats(tech_all),
        "aintibody_ht_spr_duplicate_clones": _stats(ai_tech_pairs),
        "aintibody_multi_replicate_spr_sd": _stats(ai_expected_rep_diff),
        "skempiv2_identical_seq_replicates": _stats(sk_rep),
        "cross_platform_spr_vs_kinexa_and_multicrystal": _stats(inter_plat_all),
        "aintibody_paired_spr_vs_kinexa": _stats(spr_vs_kinexa),
    }


def build_panel_pairs_custom_margins(pw_df: pd.DataFrame, m_neu: float, m_imp: float) -> pd.DataFrame:
    """Build within-panel 3-state pairs at arbitrary `(m_neu, m_imp)` thresholds (`0 < m_neu < m_imp`)."""
    rows = []
    for (ag_grp, ds), g in pw_df.groupby(["Ag_group", "Dataset"], sort=True):
        if not (6 <= len(g) <= 200) or g["binding_score"].nunique() < 3:
            continue
        ids = g["AbAgID"].to_numpy()
        scores = g["binding_score"].to_numpy(dtype=float)
        ag_t = str(g["Ag_type"].iloc[0])
        for i, j in combinations(range(len(g)), 2):
            diff = float(scores[i] - scores[j])
            ad = abs(diff)
            if ad >= m_imp:
                lbl = 1 if diff > 0 else -1
            elif ad <= m_neu:
                lbl = 0
            else:
                continue
            # Deterministic 50/50 orientation balance
            if (len(rows) % 2 == 1 and lbl == 1) or (len(rows) % 2 == 0 and lbl == -1):
                i_use, j_use, d_use, l_use = j, i, -diff, -lbl
            else:
                i_use, j_use, d_use, l_use = i, j, diff, lbl
            rows.append(
                {
                    "pair_id": f"SENS_{len(rows) + 1:06d}",
                    "AbAgID_1": ids[i_use],
                    "AbAgID_2": ids[j_use],
                    "binding_diff": d_use,
                    "logKD_diff": -d_use,
                    "label": l_use,
                    "Category": "Candidate",
                    "Lev_distance": 3,
                    "AbRank_group": f"{ag_grp}::{ds}",
                    "Ag_group_1": ag_grp,
                    "Ag_group_2": ag_grp,
                    "Ag_type": ag_t,
                    "Assay_modality": "KD",
                    "Dataset_1": ds,
                    "Dataset_2": ds,
                }
            )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def build_all_tournament_pairs(pw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ag_grp, ds), g in pw_df.groupby(["Ag_group", "Dataset"], sort=True):
        if not (6 <= len(g) <= 200) or g["binding_score"].nunique() < 3:
            continue
        ids = g["AbAgID"].to_numpy()
        scores = g["binding_score"].to_numpy(dtype=float)
        ag_t = str(g["Ag_type"].iloc[0])
        for i, j in combinations(range(len(g)), 2):
            diff = float(scores[i] - scores[j])
            rows.append(
                {
                    "pair_id": f"TPNL_{len(rows) + 1:06d}",
                    "AbAgID_1": ids[i],
                    "AbAgID_2": ids[j],
                    "binding_diff": diff,
                    "logKD_diff": -diff,
                    "label": 1 if diff >= 1.0 else (-1 if diff <= -1.0 else 0),
                    "Category": "Candidate",
                    "Lev_distance": 3,
                    "AbRank_group": f"{ag_grp}::{ds}",
                    "Ag_group_1": ag_grp,
                    "Ag_group_2": ag_grp,
                    "Ag_type": ag_t,
                    "Assay_modality": "KD",
                    "Dataset_1": ds,
                    "Dataset_2": ds,
                }
            )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def tournament_scores(pnl_pairs_df: pd.DataFrame, pred_diff: np.ndarray) -> dict[str, float]:
    sums: dict[str, float] = {}
    cnts: dict[str, int] = {}
    for a, b, p in zip(pnl_pairs_df["AbAgID_1"], pnl_pairs_df["AbAgID_2"], pred_diff):
        fp = float(p)
        sums[a] = sums.get(a, 0.0) + fp
        cnts[a] = cnts.get(a, 0) + 1
        sums[b] = sums.get(b, 0.0) - fp
        cnts[b] = cnts.get(b, 0) + 1
    return {k: sums[k] / max(cnts[k], 1) for k in sums}


def evaluate_wetlab_hit_selection(pw_df: pd.DataFrame, pred_scores: dict[str, float]) -> dict:
    """Evaluate practical wet-lab binder selection across unfiltered variant panels:
      - `Precision@Top-20%`: fraction of top-20% ranked candidates that are true tight-binder hits
        (`y_true >= max(Q75, median + 0.30)`, i.e. >2x above panel median and top quartile)
      - `Precision@Top-3`: fraction of top-3 selected candidates that are true tight-binder hits
      - `Neutral/Deleterious FDR@Top-20%`: fraction of top-20% picks with `y_true <= median + 0.30`
      - `Mean ΔpKd Shift@Top-20%`: mean `(y_selected - median(y_panel))` in pKd units (`+0.30` = 2x tighter)
    """
    df = pw_df[pw_df["AbAgID"].isin(pred_scores)].copy()
    df["_pred"] = df["AbAgID"].map(pred_scores)
    sp_res = evaluate_pointwise_spearman(df, pred_scores, min_group_size=6)

    p_top20, base_rate, p_top3, fdr_top20, dpkd_top20 = [], [], [], [], []
    for (ag_grp, ds), g in df.groupby(["Ag_group", "Dataset"], sort=True):
        if not (6 <= len(g) <= 200) or g["binding_score"].nunique() < 3:
            continue
        y_true = g["binding_score"].to_numpy(dtype=float)
        y_pred = g["_pred"].to_numpy(dtype=float)
        med = float(np.median(y_true))
        q75 = float(np.percentile(y_true, 75.0))
        thresh = max(q75, med + 0.30)
        is_hit = y_true >= thresh
        if is_hit.sum() == 0 or is_hit.sum() == len(y_true):
            continue

        order = np.argsort(-y_pred, kind="mergesort")
        k20 = max(1, int(round(0.20 * len(y_true))))
        k3 = min(3, max(1, len(y_true) // 3))

        sel20 = order[:k20]
        sel3 = order[:k3]
        p_top20.append(float(is_hit[sel20].mean()))
        base_rate.append(float(is_hit.mean()))
        p_top3.append(float(is_hit[sel3].mean()))
        fdr_top20.append(float((y_true[sel20] <= med + 0.30).mean()))
        dpkd_top20.append(float(np.mean(y_true[sel20] - med)))

    mean_p20 = float(np.mean(p_top20))
    mean_base = float(np.mean(base_rate))
    return {
        "n_panels": len(p_top20),
        "mean_spearman_rho": sp_res["mean_spearman_rho"],
        "mean_top_binder_auc": sp_res["mean_top_binder_auc"],
        "random_hit_rate": round(mean_base, 4),
        "precision_at_top20_pct": round(mean_p20, 4),
        "enrichment_factor_top20": round(mean_p20 / max(mean_base, 1e-6), 3),
        "precision_at_top3": round(float(np.mean(p_top3)), 4),
        "neutral_or_worse_fdr_top20": round(float(np.mean(fdr_top20)), 4),
        "mean_dpkd_over_median_top20": round(float(np.mean(dpkd_top20)), 4),
    }


def build_pointwise_complex_features(df_pw: pd.DataFrame, esmc_cache: dict, ablang_cache: dict) -> np.ndarray:
    """Build complex feature vector `[VH_mean, VL_mean, VH_cdr3, AbLang_paired, Ag_mean, Ag_std]` for scalar pKd regression."""
    em, ec, es = esmc_cache["means"], esmc_cache["cdr3s"], esmc_cache["stds"]
    am, ac = ablang_cache["paired_means"], ablang_cache["vh_cdr3s"]
    feats = []
    for h, l, a in zip(df_pw["Ab_heavy_chain_seq"], df_pw["Ab_light_chain_seq"].fillna(""), df_pw["Ag_seq"]):
        kh, ka = seq_hash(str(h)), seq_hash(str(a))
        kl = seq_hash(str(l)) if l else kh
        kp = pair_hash(str(h), str(l)) if l else ""
        ab_vec = np.concatenate([em[kh], em[kl], ec[kh]]).astype(np.float32)
        abl_vec = np.concatenate([am[kp], ac[kp]]).astype(np.float32) if (kp and kp in am) else np.zeros(960, dtype=np.float32)
        ag_vec = np.concatenate([em[ka], es[ka]]).astype(np.float32)
        feats.append(np.concatenate([ab_vec, abl_vec, ag_vec]))
    return np.stack(feats, axis=0)


_G13: dict[str, Any] = {}


def _worker_panel_cluster(cl: str) -> tuple[str, np.ndarray, dict[str, float], dict[str, float], dict[str, tuple[np.ndarray, np.ndarray]], float]:
    torch.set_num_threads(1)
    t_c = time.time()
    pnl_clust = _G13["pnl_clust"]
    pw_panels_df = _G13["pw_panels_df"]
    reg_clust = _G13["reg_clust"]
    reg_lin = _G13["reg_lin"]
    X_reg_pool = _G13["X_reg_pool"]
    y_reg_pool = _G13["y_reg_pool"]
    X_reg_eval = _G13["X_reg_eval"]
    pool_c1 = _G13["pool_c1"]
    pool_c2 = _G13["pool_c2"]
    pool_l1 = _G13["pool_l1"]
    pool_l2 = _G13["pool_l2"]
    pair_pool_df = _G13["pair_pool_df"]
    pool_sp = _G13["pool_sp"]
    pool_sf = _G13["pool_sf"]
    pnl_sp = _G13["pnl_sp"]
    pnl_sf = _G13["pnl_sf"]

    te_p_mask = pnl_clust == cl
    te_p_idx = np.flatnonzero(te_p_mask)
    pw_cl = pw_panels_df[pw_panels_df["Ag_cluster"].astype(str) == cl]
    te_lins = set(pw_cl["Ab_lineage"].astype(str))

    reg_keep = (reg_clust != cl) & (~np.isin(reg_lin, list(te_lins)))
    sc_reg = StandardScaler()
    Xr_tr = sc_reg.fit_transform(X_reg_pool[reg_keep])
    yr_tr = y_reg_pool[reg_keep]
    Xr_te = sc_reg.transform(X_reg_eval[(pw_panels_df["Ag_cluster"].astype(str) == cl).to_numpy()])

    ridge = RidgeCV(alphas=[1.0, 10.0, 100.0, 1000.0, 5000.0]).fit(Xr_tr, yr_tr)
    mlp_r = MLPRegressor(hidden_layer_sizes=(128, 32), alpha=10.0, max_iter=60, early_stopping=True, random_state=42).fit(Xr_tr, yr_tr)
    rg_sc = {str(aid): float(p_rg) for aid, p_rg in zip(pw_cl["AbAgID"], ridge.predict(Xr_te))}
    ml_sc = {str(aid): float(p_ml) for aid, p_ml in zip(pw_cl["AbAgID"], mlp_r.predict(Xr_te))}

    is_loc_pool = _G13["is_loc_pool"]
    keep_pair = (pool_c1 != cl) & (pool_c2 != cl) & (~np.isin(pool_l1, list(te_lins))) & (~np.isin(pool_l2, list(te_lins)))
    kp_idx = np.flatnonzero(keep_pair)
    rem_loc_cls = sorted(set(pool_c1[kp_idx[is_loc_pool[kp_idx]]]))
    val_cls = set(rem_loc_cls[1::5]) if len(rem_loc_cls) >= 5 else set(rem_loc_cls[:1])
    is_v = np.isin(pool_c1[kp_idx], list(val_cls)) | np.isin(pool_c2[kp_idx], list(val_cls))
    val_lins = set(pool_l1[kp_idx[is_v]]) | set(pool_l2[kp_idx[is_v]])
    is_tr = (~is_v) & (~np.isin(pool_l1[kp_idx], list(val_lins))) & (~np.isin(pool_l2[kp_idx], list(val_lins)))

    tr_i = kp_idx[is_tr]
    va_i = kp_idx[is_v]
    tr_df, va_df = pair_pool_df.iloc[tr_i].reset_index(drop=True), pair_pool_df.iloc[va_i].reset_index(drop=True)
    tr_sp, va_sp, te_sp = pool_sp.subset(tr_i), pool_sp.subset(va_i), pnl_sp.subset(te_p_idx)
    tr_sf, va_sf, te_sf = pool_sf.subset(tr_i), pool_sf.subset(va_i), pnl_sf.subset(te_p_idx)
    w_tr = compute_stratum_weights(tr_df, tr_sp.is_ag_mut)

    xm_tr = build_odd_linear_matrix([tr_sp.esmc_ab_1, tr_sp.esmc_ag_1], [tr_sp.esmc_ab_2, tr_sp.esmc_ag_2], include_unit_direction=True)
    xm_va = build_odd_linear_matrix([va_sp.esmc_ab_1, va_sp.esmc_ag_1], [va_sp.esmc_ab_2, va_sp.esmc_ag_2], include_unit_direction=True)
    xm_te = build_odd_linear_matrix([te_sp.esmc_ab_1, te_sp.esmc_ag_1], [te_sp.esmc_ab_2, te_sp.esmc_ag_2], include_unit_direction=True)
    xr_tr = build_odd_linear_matrix([tr_sp.esmc_ab_1, tr_sp.esmc_ag_1], [tr_sp.esmc_ab_2, tr_sp.esmc_ag_2], include_unit_direction=False)
    xr_va = build_odd_linear_matrix([va_sp.esmc_ab_1, va_sp.esmc_ag_1], [va_sp.esmc_ab_2, va_sp.esmc_ag_2], include_unit_direction=False)
    xr_te = build_odd_linear_matrix([te_sp.esmc_ab_1, te_sp.esmc_ag_1], [te_sp.esmc_ab_2, te_sp.esmc_ag_2], include_unit_direction=False)

    m_bin = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(xm_tr, tr_sp.y, w_tr, xm_va, va_df, va_sp.is_ag_mut, labels_tr=tr_sp.labels)
    m_raw = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(xr_tr, tr_sp.y, None, xr_va, va_df, va_sp.is_ag_mut, labels_tr=tr_sp.labels)
    db = m_bin.predict(xm_te)

    e2h = TwoHeadRegimeRanker(local_head=m_bin, global_head=m_raw)
    dv_tr = e2h.predict(tr_sp, xm_tr, xr_tr)
    dv_va = e2h.predict(va_sp, xm_va, xr_va)
    imp_c = SymmetricImpactClassifier(seed=42).fit(tr_sp, va_sp, dv_tr, dv_va, val_df=va_df)
    hurd = OrdinalHurdleRanker(e2h, imp_c).calibrate(va_df, va_sp, dv_va)
    dh, ih = hurd.predict_3state(te_sp, e2h.predict(te_sp, xm_te, xr_te))

    pred_sap = fit_3state_foundation_hurdle(
        tr_sf.saprot_ab_1, tr_sf.saprot_ab_2, tr_sf.saprot_ag_1, tr_sf.saprot_ag_2,
        np.concatenate([tr_sf.zs_dir["saprot_zs"][:, None], tr_sp.biophys_odd], axis=1),
        tr_sf.struct_even[:, :9],
        va_sf.saprot_ab_1, va_sf.saprot_ab_2, va_sf.saprot_ag_1, va_sf.saprot_ag_2,
        np.concatenate([va_sf.zs_dir["saprot_zs"][:, None], va_sp.biophys_odd], axis=1),
        va_sf.struct_even[:, :9],
        tr_sp, va_sp, va_df, w_tr, seed=42,
    )
    ds, is_ = pred_sap(
        te_sp, te_sf.saprot_ab_1, te_sf.saprot_ab_2, te_sf.saprot_ag_1, te_sf.saprot_ag_2,
        np.concatenate([te_sf.zs_dir["saprot_zs"][:, None], te_sp.biophys_odd], axis=1),
        te_sf.struct_even[:, :9],
    )

    m_anti = ANTIPASTI3StateTrainer(seed=42, epochs=14).fit(tr_sf, tr_sp, va_sf, va_sp, va_df, w_tr)
    da, ia = m_anti.predict_3state(te_sf, te_sp)

    pdict = {
        "ESMC Multi-Scale (Binary Probe)": (db, np.abs(db)),
        "ESMC-300M (3-State Hurdle)": (dh, ih),
        "SaProt-650M (3-State Hurdle)": (ds, is_),
        "ANTIPASTI (Normal-Mode 3D CNN)": (da, ia),
    }
    return cl, te_p_idx, rg_sc, ml_sc, pdict, time.time() - t_c


def _seq_sim_str(s1: str, s2: str) -> float:
    sh_s, lg_s = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    if len(sh_s) >= 25 and sh_s in lg_s:
        return 1.0
    return 1.0 - Levenshtein.distance(s1, s2) / max(len(s1), len(s2), 1)


def _worker_cutoff_job(job: tuple[str, float, float, str]) -> tuple[str, str, dict[str, float | None]]:
    label_cut, ag_cut, ab_cut, cl = job
    torch.set_num_threads(1)
    lc1 = _G13["lc1"]
    lg1 = _G13["lg1"]
    ll1 = _G13["ll1"]
    ll2 = _G13["ll2"]
    ag_seq_by_cl = _G13["ag_seq_by_cl"]
    ag_seq_by_grp = _G13["ag_seq_by_grp"]
    ab_seq_by_lin = _G13["ab_seq_by_lin"]
    loc_pool_only = _G13["loc_pool_only"]
    loc_only_sp = _G13["loc_only_sp"]
    loc_only_sf = _G13["loc_only_sf"]

    if cl == "AgClust-015":
        te_ix = np.flatnonzero((lc1 == cl) & (lg1 == "EGFR"))
    else:
        te_ix = np.flatnonzero(lc1 == cl)
    te_grps = set(lg1[te_ix])
    te_lset = set(ll1[te_ix]) | set(ll2[te_ix])

    if ag_cut >= 0.65:
        excl_grps = {
            g2 for g2, s2 in ag_seq_by_grp.items()
            if any(_seq_sim_str(str(ag_seq_by_grp[tg]), str(s2)) >= ag_cut for tg in te_grps)
        }
        kp = ~np.isin(lg1, list(excl_grps))
    elif ag_cut >= 0.45:
        kp = (lc1 != cl) & (~np.isin(ll1, list(te_lset))) & (~np.isin(ll2, list(te_lset)))
    else:
        s_ag_te = str(ag_seq_by_cl[cl])
        excl_cls = {c2 for c2, s2 in ag_seq_by_cl.items() if _seq_sim_str(s_ag_te, str(s2)) >= ag_cut}
        excl_lins = {
            l2 for tl in te_lset for l2, s2 in ab_seq_by_lin.items()
            if _seq_sim_str(str(ab_seq_by_lin[tl]), str(s2)) >= ab_cut
        }
        kp = (~np.isin(lc1, list(excl_cls))) & (~np.isin(ll1, list(excl_lins))) & (~np.isin(ll2, list(excl_lins)))

    kp_ix = np.flatnonzero(kp)
    r_cls = sorted(set(lc1[kp_ix]))
    v_cls = set(r_cls[1::5]) if len(r_cls) >= 5 else set(r_cls[:1])
    iv = np.isin(lc1[kp_ix], list(v_cls))
    v_lins = set(ll1[kp_ix[iv]]) | set(ll2[kp_ix[iv]])
    itr = (~iv) & (~np.isin(ll1[kp_ix], list(v_lins))) & (~np.isin(ll2[kp_ix], list(v_lins)))
    tr_i, va_i = kp_ix[itr], kp_ix[iv]
    tr_d, va_d, te_d = loc_pool_only.iloc[tr_i].reset_index(drop=True), loc_pool_only.iloc[va_i].reset_index(drop=True), loc_pool_only.iloc[te_ix].reset_index(drop=True)
    tr_s, va_s, te_s = loc_only_sp.subset(tr_i), loc_only_sp.subset(va_i), loc_only_sp.subset(te_ix)
    tr_f, va_f, te_f = loc_only_sf.subset(tr_i), loc_only_sf.subset(va_i), loc_only_sf.subset(te_ix)
    wt = compute_stratum_weights(tr_d, tr_s.is_ag_mut)

    xm_tr = build_odd_linear_matrix([tr_s.esmc_ab_1, tr_s.esmc_ag_1], [tr_s.esmc_ab_2, tr_s.esmc_ag_2], include_unit_direction=True)
    xm_va = build_odd_linear_matrix([va_s.esmc_ab_1, va_s.esmc_ag_1], [va_s.esmc_ab_2, va_s.esmc_ag_2], include_unit_direction=True)
    xm_te = build_odd_linear_matrix([te_s.esmc_ab_1, te_s.esmc_ag_1], [te_s.esmc_ab_2, te_s.esmc_ag_2], include_unit_direction=True)
    mb = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(xm_tr, tr_s.y, wt, xm_va, va_d, va_s.is_ag_mut, labels_tr=tr_s.labels)
    ic = SymmetricImpactClassifier(seed=42).fit(tr_s, va_s, mb.predict(xm_tr), mb.predict(xm_va), val_df=va_d)
    hd = OrdinalHurdleRanker(mb, ic).calibrate(va_d, va_s, mb.predict(xm_va))
    d_eh, i_eh = hd.predict_3state(te_s, mb.predict(xm_te))

    psap = fit_3state_foundation_hurdle(
        tr_f.saprot_ab_1, tr_f.saprot_ab_2, tr_f.saprot_ag_1, tr_f.saprot_ag_2,
        np.concatenate([tr_f.zs_dir["saprot_zs"][:, None], tr_s.biophys_odd], axis=1), tr_f.struct_even[:, :9],
        va_f.saprot_ab_1, va_f.saprot_ab_2, va_f.saprot_ag_1, va_f.saprot_ag_2,
        np.concatenate([va_f.zs_dir["saprot_zs"][:, None], va_s.biophys_odd], axis=1), va_f.struct_even[:, :9],
        tr_s, va_s, va_d, wt, seed=42,
    )
    d_sp, i_sp = psap(
        te_s, te_f.saprot_ab_1, te_f.saprot_ab_2, te_f.saprot_ag_1, te_f.saprot_ag_2,
        np.concatenate([te_f.zs_dir["saprot_zs"][:, None], te_s.biophys_odd], axis=1), te_f.struct_even[:, :9],
    )

    out_m: dict[str, float | None] = {}
    for mk, (dv, iv_) in (
        ("ESMC+AbLang2 (1D ZS)", (te_s.zs_dict["esmc_ablang2_zeroshot"], te_s.zs_impact_dict["esmc_ablang2_zeroshot"])),
        ("PRODIGY+SASA (3D ZS)", (te_f.zs_dir["prodigy_sasa"], te_f.zs_imp["prodigy_sasa"])),
        ("ProteinMPNN (3D ZS)", (te_f.zs_dir["protein_mpnn"], te_f.zs_imp["protein_mpnn"])),
        ("ESMC-300M (3-State Hurdle)", (d_eh, i_eh)),
        ("SaProt-650M (3-State Hurdle)", (d_sp, i_sp)),
    ):
        m_cl = compute_per_antigen_macro_metrics(te_d, dv, iv_, min_pairs=5)
        out_m[mk] = float(m_cl["auc_ordinal"]) if m_cl["auc_ordinal"] is not None else None
    return label_cut, cl, out_m


def main() -> None:
    t0 = time.time()
    root = get_repo_root()
    bdir = root / "data" / "benchmark"
    edir = root / "data" / "embeddings"
    sdir = root / "data" / "structures"
    res_dir = root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Part A: Empirical Replicate & Cross-Platform Uncertainty Calibration
    # ------------------------------------------------------------------
    print("=== Part A: Computing Empirical Replicate & Cross-Platform Uncertainty ===")
    rep_unc = compute_empirical_replicate_uncertainty(root)
    print("  Technical within-platform |ΔpKd|:", rep_unc["within_platform_technical_replicates"])
    print("  Cross-platform SPR vs KinExA / Multi-Crystal |ΔpKd|:", rep_unc["cross_platform_spr_vs_kinexa_and_multicrystal"])

    # Load caches and tables for Parts B & C
    kd_raw = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False)
    kd_lookup = kd_raw.drop_duplicates("AbAgID").set_index("AbAgID")
    kd_lookup["Ab_light_chain_seq"] = kd_lookup["Ab_light_chain_seq"].fillna("")

    esmc_cache = torch.load(edir / "esmc_300m.pt", map_location="cpu", weights_only=False)
    ablang_cache = torch.load(edir / "ablang2_paired.pt", map_location="cpu", weights_only=False)
    esm2_cache = torch.load(edir / "esm2_150m.pt", map_location="cpu", weights_only=False)
    complex_cache = torch.load(sdir / "complex_cache.pt", map_location="cpu", weights_only=False)
    inv_cache = torch.load(edir / "inverse_folding_scaffolds.pt", map_location="cpu", weights_only=False)
    saprot_cache = torch.load(edir / "saprot_650m.pt", map_location="cpu", weights_only=False)
    mint_cache = torch.load(edir / "mint_650m.pt", map_location="cpu", weights_only=False)

    abag_in_3d = set(complex_cache["abag_to_scaffold"].keys())
    esmc_keys = set(esmc_cache["plls"].keys())
    esm2_keys = set(esm2_cache["plls"].keys())
    ablang_keys = set(ablang_cache["plls"].keys())

    # Collect all cached unfiltered panels (`6 <= len <= 200`) across train + val + test pointwise splits!
    pw_all = pd.concat(
        [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}_pointwise.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")],
        ignore_index=True,
    )
    pw_all["Ab_light_chain_seq"] = pw_all["Ab_light_chain_seq"].fillna("")
    pw_cached_mask = [
        (aid in abag_in_3d)
        and (seq_hash(str(h)) in esmc_keys)
        and (seq_hash(str(l)) in esmc_keys if l else True)
        and (seq_hash(str(a)) in esmc_keys)
        and (seq_hash(str(h)) in esm2_keys)
        and (seq_hash(str(l)) in esm2_keys if l else True)
        and (seq_hash(str(a)) in esm2_keys)
        and (pair_hash(str(h), str(l)) in ablang_keys if l else True)
        for aid, h, l, a in zip(pw_all["AbAgID"], pw_all["Ab_heavy_chain_seq"], pw_all["Ab_light_chain_seq"], pw_all["Ag_seq"])
    ]
    pw_sub = pw_all[pw_cached_mask].copy()
    valid_panels = []
    for (ag_g, ds), g in pw_sub.groupby(["Ag_group", "Dataset"], sort=True):
        if 6 <= len(g) <= 200 and g["binding_score"].nunique() >= 3:
            valid_panels.append(g)
    pw_panels_df = pd.concat(valid_panels, ignore_index=True)
    print(f"Assembled {len(valid_panels)} complete cached unfiltered variant panels ({len(pw_panels_df):,} variants).")

    # Assemble LOACO training pool from `unseen_ag/{train,val,test}.tsv`
    pair_pool_df = pd.concat(
        [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")],
        ignore_index=True,
    )
    pool_sp = assemble_split_tensors(pair_pool_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)
    pool_sf = assemble_structure_pair_features(pair_pool_df, kd_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)

    pool_c1 = pair_pool_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
    pool_c2 = pair_pool_df["AbAgID_2"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
    pool_l1 = pair_pool_df["AbAgID_1"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()
    pool_l2 = pair_pool_df["AbAgID_2"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()

    # Build all tournament pairs across the 22 panels & run strict LOACO predictions once!
    pnl_tour_df = build_all_tournament_pairs(pw_panels_df)
    pnl_sp = assemble_split_tensors(pnl_tour_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)
    pnl_sf = assemble_structure_pair_features(pnl_tour_df, kd_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)
    pnl_clust = pnl_tour_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()

    # Also build pointwise training pool (`non-AbBiBench` + stratified `AbBiBench` cached rows) for Scalar pKd Regression!
    pw_train_cand = pw_sub.drop_duplicates("AbAgID").copy()
    rng_pw = np.random.default_rng(42)
    abbib_rows = pw_train_cand[pw_train_cand["Dataset"] == "AbBibench"]
    non_abbib_rows = pw_train_cand[pw_train_cand["Dataset"] != "AbBibench"]
    if len(abbib_rows) > 1500:
        abbib_rows = abbib_rows.iloc[rng_pw.choice(len(abbib_rows), size=1500, replace=False)]
    pw_reg_pool = pd.concat([non_abbib_rows, abbib_rows], ignore_index=True)
    X_reg_pool = build_pointwise_complex_features(pw_reg_pool, esmc_cache, ablang_cache)
    y_reg_pool = pw_reg_pool["binding_score"].to_numpy(dtype=np.float32)
    reg_clust = pw_reg_pool["Ag_cluster"].astype(str).to_numpy()
    reg_lin = pw_reg_pool["Ab_lineage"].astype(str).to_numpy()
    X_reg_eval = build_pointwise_complex_features(pw_panels_df, esmc_cache, ablang_cache)
    del esmc_cache, ablang_cache, esm2_cache, complex_cache, inv_cache, saprot_cache, mint_cache
    gc.collect()

    print("\n=== Running LOACO across all 22 Unfiltered Panels (Scalar pKd Regression vs. Binary vs. 3-State vs. 3D) ===")
    n_tp = len(pnl_tour_df)
    tour_preds: dict[str, np.ndarray] = {
        "ESMC+AbLang2 (1D ZS)": pnl_sp.zs_dict["esmc_ablang2_zeroshot"].copy(),
        "PRODIGY+SASA (3D Energy ZS)": pnl_sf.zs_dir["prodigy_sasa"].copy(),
        "ProteinMPNN (3D Complex ZS)": pnl_sf.zs_dir["protein_mpnn"].copy(),
        "ESMC Multi-Scale (Binary Probe)": np.zeros(n_tp, dtype=np.float64),
        "ESMC-300M (3-State Hurdle)": np.zeros(n_tp, dtype=np.float64),
        "SaProt-650M (3-State Hurdle)": np.zeros(n_tp, dtype=np.float64),
        "ANTIPASTI (Normal-Mode 3D CNN)": np.zeros(n_tp, dtype=np.float64),
    }
    tour_imps: dict[str, np.ndarray] = {
        "ESMC+AbLang2 (1D ZS)": pnl_sp.zs_impact_dict["esmc_ablang2_zeroshot"].copy(),
        "PRODIGY+SASA (3D Energy ZS)": pnl_sf.zs_imp["prodigy_sasa"].copy(),
        "ProteinMPNN (3D Complex ZS)": pnl_sf.zs_imp["protein_mpnn"].copy(),
        "ESMC Multi-Scale (Binary Probe)": np.zeros(n_tp, dtype=np.float64),
        "ESMC-300M (3-State Hurdle)": np.zeros(n_tp, dtype=np.float64),
        "SaProt-650M (3-State Hurdle)": np.zeros(n_tp, dtype=np.float64),
        "ANTIPASTI (Normal-Mode 3D CNN)": np.zeros(n_tp, dtype=np.float64),
    }
    reg_ridge_scores: dict[str, float] = {}
    reg_mlp_scores: dict[str, float] = {}

    is_loc_pool = pair_pool_df["Category"].isin(["L1", "L2", "L3"]).to_numpy()
    panel_clusters = sorted(set(pnl_clust))
    _G13.update(
        is_loc_pool=is_loc_pool,
        pnl_clust=pnl_clust, pw_panels_df=pw_panels_df,
        reg_clust=reg_clust, reg_lin=reg_lin, X_reg_pool=X_reg_pool, y_reg_pool=y_reg_pool, X_reg_eval=X_reg_eval,
        pool_c1=pool_c1, pool_c2=pool_c2, pool_l1=pool_l1, pool_l2=pool_l2,
        pair_pool_df=pair_pool_df, pool_sp=pool_sp, pool_sf=pool_sf, pnl_sp=pnl_sp, pnl_sf=pnl_sf,
    )
    n_w13 = min(8, len(panel_clusters))
    done_p = 0
    with mp.get_context("fork").Pool(n_w13) as pool:
        for cl, te_p_idx, rg_sc, ml_sc, pdict, dt_c in pool.imap_unordered(_worker_panel_cluster, panel_clusters):
            done_p += 1
            print(f"  [{done_p:2d}/{len(panel_clusters)}] Panel Cluster {cl} (pairs={len(te_p_idx)}) done in {dt_c:.1f}s", flush=True)
            reg_ridge_scores.update(rg_sc)
            reg_mlp_scores.update(ml_sc)
            for mk, (dv, iv_) in pdict.items():
                tour_preds[mk][te_p_idx] = dv
                tour_imps[mk][te_p_idx] = iv_

    # Part C: Practical Wet-Lab Hit Selection Comparison
    hit_selection = {
        "Scalar pKd Ridge Regression": evaluate_wetlab_hit_selection(pw_panels_df, reg_ridge_scores),
        "Scalar pKd MLP Regression": evaluate_wetlab_hit_selection(pw_panels_df, reg_mlp_scores),
    }
    for m_name, p_arr in tour_preds.items():
        hit_selection[m_name] = evaluate_wetlab_hit_selection(pw_panels_df, tournament_scores(pnl_tour_df, p_arr))

    # Part B.1: Margin `(m_neu, m_imp)` Sensitivity Grid across all 22 Unfiltered Panels
    margin_grids = [(0.15, 0.50), (0.20, 0.75), (0.30, 1.00), (0.40, 1.25)]
    pair_key_to_idx = {(a, b): idx for idx, (a, b) in enumerate(zip(pnl_tour_df["AbAgID_1"], pnl_tour_df["AbAgID_2"]))}
    margin_sensitivity = {}
    for m_neu, m_imp in margin_grids:
        g_df = build_panel_pairs_custom_margins(pw_panels_df, m_neu=m_neu, m_imp=m_imp)
        # Map predictions with orientation sign!
        g_res = {}
        for m_name in tour_preds:
            pd_g = np.zeros(len(g_df), dtype=np.float64)
            pi_g = np.zeros(len(g_df), dtype=np.float64)
            for ii, (a, b) in enumerate(zip(g_df["AbAgID_1"], g_df["AbAgID_2"])):
                if (a, b) in pair_key_to_idx:
                    k = pair_key_to_idx[(a, b)]
                    pd_g[ii] = tour_preds[m_name][k]
                    pi_g[ii] = tour_imps[m_name][k]
                else:
                    k = pair_key_to_idx[(b, a)]
                    pd_g[ii] = -tour_preds[m_name][k]
                    pi_g[ii] = tour_imps[m_name][k]
            mm = compute_per_antigen_macro_metrics(g_df, pd_g, pi_g, min_pairs=5)
            g_res[m_name] = {
                "auc_dir": mm["auc_dir"],
                "auc_impact": mm["auc_impact"],
                "auc_ordinal": mm["auc_ordinal"],
                "n_pairs": mm["n_pairs"],
            }
        margin_sensitivity[f"neu_{m_neu:.2f}_imp_{m_imp:.2f}"] = g_res

    ref_ords = [margin_sensitivity["neu_0.30_imp_1.00"][m]["auc_ordinal"] for m in tour_preds]
    margin_rank_corrs = {
        gk: round(float(spearmanr(ref_ords, [gv[m]["auc_ordinal"] for m in tour_preds])[0]), 4)
        for gk, gv in margin_sensitivity.items()
    }

    # Part B.2: Sequence-Identity Holdout Threshold Sensitivity (`35%/70%`, `50%/80%`, `70%/90%`) on Test Panels
    ag_seq_by_cl = kd_raw.groupby("Ag_cluster")["Ag_seq"].first().to_dict()
    ag_seq_by_grp = kd_raw.groupby("Ag_group")["Ag_seq"].first().to_dict()
    ab_seq_by_lin = kd_raw.assign(_fv=kd_raw["Ab_heavy_chain_seq"] + kd_raw["Ab_light_chain_seq"].fillna("")).groupby("Ab_lineage")["_fv"].first().to_dict()

    cutoff_specs = [
        ("Loose (<70% Ag, <90% Ab)", 0.70, 0.90),
        ("Canonical (<50% Ag, <80% Ab)", 0.50, 0.80),
        ("Ultra-Strict (<35% Ag, <70% Ab)", 0.35, 0.70),
    ]
    loc_ix_pool = np.flatnonzero(is_loc_pool)
    loc_pool_only = pair_pool_df.iloc[loc_ix_pool].reset_index(drop=True)
    loc_only_sp = pool_sp.subset(loc_ix_pool)
    loc_only_sf = pool_sf.subset(loc_ix_pool)
    lc1 = loc_pool_only["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
    lg1 = loc_pool_only["Ag_group_1"].astype(str).to_numpy()
    ll1 = loc_pool_only["AbAgID_1"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()
    ll2 = loc_pool_only["AbAgID_2"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()

    _G13.update(
        lc1=lc1, lg1=lg1, ll1=ll1, ll2=ll2, ag_seq_by_cl=ag_seq_by_cl, ag_seq_by_grp=ag_seq_by_grp, ab_seq_by_lin=ab_seq_by_lin,
        loc_pool_only=loc_pool_only, loc_only_sp=loc_only_sp, loc_only_sf=loc_only_sf,
    )
    eval_cls_sample = ["AgClust-064", "AgClust-027", "AgClust-015", "AgClust-037", "AgClust-020", "AgClust-054", "AgClust-016", "AgClust-002"]
    cut_jobs = [(lbl, ag_c, ab_c, cl) for lbl, ag_c, ab_c in cutoff_specs for cl in eval_cls_sample]
    per_cut_cl: dict[str, dict[str, list[float]]] = {
        lbl: {
            "ESMC+AbLang2 (1D ZS)": [],
            "PRODIGY+SASA (3D ZS)": [],
            "ProteinMPNN (3D ZS)": [],
            "ESMC-300M (3-State Hurdle)": [],
            "SaProt-650M (3-State Hurdle)": [],
        }
        for lbl, _, _ in cutoff_specs
    }
    with mp.get_context("fork").Pool(min(8, len(cut_jobs))) as pool:
        for lbl_c, _, out_m in pool.imap_unordered(_worker_cutoff_job, cut_jobs):
            for mk, val_o in out_m.items():
                if val_o is not None:
                    per_cut_cl[lbl_c][mk].append(val_o)
    cutoff_sensitivity = {lbl: {mk: round(float(np.mean(vs)), 4) for mk, vs in per_cut_cl[lbl].items()} for lbl, _, _ in cutoff_specs}

    payload = {
        "benchmark": "AbRank v2.0.0",
        "part_a_replicate_uncertainty": rep_unc,
        "part_b_margin_sensitivity": margin_sensitivity,
        "part_b_margin_leaderboard_spearman_vs_canonical": margin_rank_corrs,
        "part_b_holdout_cutoff_sensitivity_ordinal_auc": cutoff_sensitivity,
        "part_c_wetlab_hit_selection_22_panels": hit_selection,
    }
    out_json = res_dir / "three_state_validation.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = [
        "# AbRank v2 — 3-State Calibration, Margin Sensitivity, & Practical Wet-Lab Hit Selection",
        "",
        "## 1. Empirical Replicate & Cross-Platform Uncertainty Calibration",
        "",
        "| Experimental Replicate Source | Pairs / Clones | Median `|ΔpKd|` | IQR `[Q25, Q75]` | 90th Pct | `% ≤ 0.30` (`≤2×`) | `% < 1.00` (`<10×`) |",
        "| :--- | ---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k, title in (
        ("within_platform_technical_replicates", "**All Within-Platform Technical Replicates**"),
        ("aintibody_ht_spr_duplicate_clones", "`AIntibody` HT-SPR Duplicate Clones"),
        ("aintibody_multi_replicate_spr_sd", "`AIntibody` Multi-Replicate SPR SD (`n_spr` ≥ 2)"),
        ("skempiv2_identical_seq_replicates", "`SKEMPI 2.0` Identical-Sequence Entries"),
        ("cross_platform_spr_vs_kinexa_and_multicrystal", "**Cross-Platform (`SPR` vs `KinExA`) & Multi-Crystal**"),
        ("aintibody_paired_spr_vs_kinexa", "`AIntibody` Paired `SPR` vs. `KinExA`"),
    ):
        s = rep_unc[k]
        md.append(
            f"| {title} | {s['n']} | `{s['median']:.3f}` | `[{s['iqr'][0]:.3f}, {s['iqr'][1]:.3f}]` | "
            f"`{s['p90']:.3f}` | **`{s['pct_le_0_30']:.1f}%`** | **`{s['pct_lt_1_00']:.1f}%`** |"
        )

    md.extend([
        "",
        "## 2. Practical Wet-Lab Hit Prioritization on Unfiltered Variant Panels (`25` LOACO Panels, `1,237` Variants)",
        "",
        "| Method | Spearman `ρ` | Top-Binder AUC | Precision@Top-20% (`Base = " + f"{list(hit_selection.values())[0]['random_hit_rate']:.3f}" + "`) | Enrichment (`EF@20%`) | Precision@Top-3 | Neutral/Worse `FDR@20%` | Mean `ΔpKd` Over Median |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for m_name, hs in hit_selection.items():
        md.append(
            f"| **{m_name}** | `{hs['mean_spearman_rho']:+.3f}` | `{hs['mean_top_binder_auc']:.3f}` | "
            f"**`{hs['precision_at_top20_pct']:.3f}`** | `{hs['enrichment_factor_top20']:.2f}×` | "
            f"`{hs['precision_at_top3']:.3f}` | `{hs['neutral_or_worse_fdr_top20']:.3f}` | `{hs['mean_dpkd_over_median_top20']:+.3f}` |"
        )

    md.extend([
        "",
        "## 3. Sensitivity to `(m_neu, m_imp)` Thresholds (`Ordinal AUC` & Leaderboard Stability)",
        "",
        f"Leaderboard Spearman rank correlation vs. canonical `(0.30, 1.00)`: `{margin_rank_corrs}`",
        "",
        "| Model | `(0.15, 0.50)` | `(0.20, 0.75)` | **Canonical `(0.30, 1.00)`** | `(0.40, 1.25)` |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for m_name in tour_preds:
        cvals = [f"`{margin_sensitivity[gk][m_name]['auc_ordinal']:.3f}`" for gk in margin_sensitivity]
        md.append(f"| **{m_name}** | {' | '.join(cvals)} |")

    md.extend([
        "",
        "## 4. Sensitivity to Sequence-Identity Holdout Thresholds (`3-Class Ordinal AUC`)",
        "",
        "| Model | Loose (`<70%` Ag, `<90%` Ab) | **Canonical (`<50%` Ag, `<80%` Ab)** | Ultra-Strict (`<35%` Ag, `<70%` Ab) |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for mk in next(iter(cutoff_sensitivity.values())):
        md.append(
            f"| **{mk}** | `{cutoff_sensitivity['Loose (<70% Ag, <90% Ab)'][mk]:.3f}` | "
            f"**`{cutoff_sensitivity['Canonical (<50% Ag, <80% Ab)'][mk]:.3f}`** | "
            f"`{cutoff_sensitivity['Ultra-Strict (<35% Ag, <70% Ab)'][mk]:.3f}` |"
        )

    out_md = res_dir / "THREE_STATE_VALIDATION.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nCompleted 3-State Validation & Hit Selection in {time.time() - t0:.1f}s! Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
