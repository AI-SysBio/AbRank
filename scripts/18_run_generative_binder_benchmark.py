#!/usr/bin/env python3
"""Step 18: AbRank-Generative (`--suite generative`) — Post-Generator Best-of-N Reranking,
Conditional Generative Log-Likelihoods, & 3-State DPO Verifier Benchmark across 1,354 Wet-Lab
Generative AI Antibody & Nanobody Designs (`IgDesign`, `Origin1`, `AIntibody`, `BoltzGen_VHH`,
`Germinal`, `Germinal_VHH`, `Adaptyv_EGFR_VHH`, `Shanker`).

Outputs:
  - `data/benchmark/splits/generative/generative_panels.tsv` (`1,354` designs)
  - `data/benchmark/splits/generative/generative_pairs.tsv` (`3-state` preference pairs)
  - `results/generative_binder_benchmark.json`
  - `results/GENERATIVE_BINDER_BENCHMARK.md`
"""

from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

from itertools import combinations
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
import torch

torch.set_num_threads(1)

from abrank.baselines.embeddings import (
    extract_ablang2_cache,
    extract_hf_plm_cache,
    pair_hash,
    seq_hash,
)
from abrank.baselines.features import build_complex_handcrafted_table
from abrank.baselines.logica_models import (
    LogiCA3StateTrainer,
)
from abrank.baselines.models import (
    AntisymmetricLinearRanker,
    NeuralPairwiseTrainer,
    OrdinalHurdleRanker,
    SymmetricImpactClassifier,
    TwoHeadRegimeRanker,
    assemble_split_tensors,
    build_odd_linear_matrix,
    compute_stratum_weights,
)
from abrank.baselines.structure_models import (
    ANTIPASTI3StateTrainer,
    GearBind3StateTrainer,
    assemble_structure_pair_features,
    fit_3state_foundation_hurdle,
    fit_walle_affinity_3state,
)
from abrank.config import get_repo_root
from abrank.evaluate import (
    compute_cluster_bootstrap_summary,
    compute_per_antigen_macro_metrics,
    evaluate_pointwise_spearman,
)
from abrank.pairs import build_within_antigen_pairs
from abrank.schema import PAIR_COLUMNS


GEN_KD_DATASETS = ["IgDesign", "Origin1", "AIntibody", "Shanker", "Germinal"]
GEN_VHH_DATASETS = ["BoltzGen_VHH", "Germinal_VHH", "Adaptyv_EGFR_VHH"]
ALL_GEN_DATASETS = GEN_KD_DATASETS + GEN_VHH_DATASETS


def _merge_1d_cache(base: dict, extra: dict) -> dict:
    out = {}
    for k in base:
        if isinstance(base[k], dict):
            out[k] = {**base[k], **extra.get(k, {})}
        else:
            out[k] = base[k]
    return out


def _build_all_panel_pairs(pw_df: pd.DataFrame, min_size: int = 5) -> pd.DataFrame:
    rows = []
    for (ag_grp, ds), g in pw_df.groupby(["Ag_group", "Dataset"], sort=True):
        if len(g) < min_size or g["binding_score"].nunique() < 2:
            continue
        # Subsample AIntibody (n=550 -> 151,000 pairs) deterministically to max 160 variants for tournament pairs
        if len(g) > 160:
            g = g.sort_values("AbAgID").iloc[np.linspace(0, len(g) - 1, 160, dtype=int)]
        ids = g["AbAgID"].to_numpy()
        scores = g["binding_score"].to_numpy(dtype=float)
        ag_t = str(g["Ag_type"].iloc[0]) if "Ag_type" in g.columns else "Protein"
        for i, j in combinations(range(len(g)), 2):
            diff = float(scores[i] - scores[j])
            rows.append(
                {
                    "pair_id": f"GPNL_{len(rows) + 1:06d}",
                    "AbAgID_1": ids[i],
                    "AbAgID_2": ids[j],
                    "binding_diff": diff,
                    "logKD_diff": -diff,
                    "label": 1 if diff >= 1.0 else (-1 if diff <= -1.0 else 0),
                    "Category": "Candidate",
                    "Lev_distance": 4,
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


def _tournament_scores(pairs_df: pd.DataFrame, pred_diff: np.ndarray) -> dict[str, float]:
    sums: dict[str, float] = {}
    cnts: dict[str, int] = {}
    for a, b, p in zip(pairs_df["AbAgID_1"], pairs_df["AbAgID_2"], pred_diff):
        fp = float(p)
        sums[a] = sums.get(a, 0.0) + fp
        cnts[a] = cnts.get(a, 0) + 1
        sums[b] = sums.get(b, 0.0) - fp
        cnts[b] = cnts.get(b, 0) + 1
    return {k: sums[k] / max(cnts[k], 1) for k in sums}


def _evaluate_best_of_n_funnel(pw_df: pd.DataFrame, score_dict: dict[str, float], min_size: int = 5) -> dict:
    sub = pw_df[pw_df["AbAgID"].isin(score_dict)].copy()
    sp = evaluate_pointwise_spearman(sub, score_dict, min_group_size=min_size)
    prec20, base20, prec3, neudel20, regret20, gain20 = [], [], [], [], [], []
    per_campaign: dict[str, list[float]] = {}

    for (ag_grp, ds), g in sub.groupby(["Ag_group", "Dataset"], sort=True):
        if len(g) < min_size or g["binding_score"].nunique() < 2:
            continue
        yt = g["binding_score"].to_numpy(dtype=float)
        yp = g["AbAgID"].map(score_dict).to_numpy(dtype=float)
        k20 = max(1, int(np.ceil(0.20 * len(yt))))
        k3 = min(3, len(yt))
        order = np.argsort(-yp)
        top20 = order[:k20]
        top3 = order[:k3]
        med = float(np.median(yt))
        thr = max(float(np.percentile(yt, 75.0)), med + 0.30)
        p20 = float((yt[top20] >= thr).mean())
        prec20.append(p20)
        base20.append(float((yt >= thr).mean()))
        prec3.append(float((yt[top3] >= thr).mean()))
        neudel20.append(float((yt[top20] <= med + 0.30).mean()))
        reg = float(np.max(yt) - np.max(yt[top20]))
        regret20.append(reg)
        gain20.append(float(np.mean(yt[top20]) - med))
        fam_key = "DeNovo_VHH" if ds in GEN_VHH_DATASETS else str(ds)
        per_campaign.setdefault(fam_key, []).append(p20)

    mp20 = float(np.mean(prec20)) if prec20 else 0.0
    mb20 = float(np.mean(base20)) if base20 else 0.21
    return {
        "n_panels": int(len(prec20)),
        "mean_spearman_rho": sp["mean_spearman_rho"],
        "mean_top_binder_auc": sp["mean_top_binder_auc"],
        "base_hit_rate": round(mb20, 4),
        "precision_at_top20_pct": round(mp20, 4),
        "enrichment_factor_top20": round(mp20 / max(mb20, 1e-8), 4),
        "precision_at_top3": round(float(np.mean(prec3)), 4) if prec3 else None,
        "neu_del_at_top20_pct": round(float(np.mean(neudel20)), 4) if neudel20 else None,
        "mean_top20_regret_pkd": round(float(np.mean(regret20)), 4) if regret20 else None,
        "mean_dpkd_over_median_top20": round(float(np.mean(gain20)), 4) if gain20 else None,
        "by_campaign_prec20": {k: round(float(np.mean(v)), 4) for k, v in per_campaign.items()},
    }


def _audit_dpo_preference_noise(pw_df: pd.DataFrame) -> dict:
    """Quantify experimental-noise tie contamination inside naive binary ΔpKd > 0 DPO pairs."""
    per_ds = {}
    all_diffs = []
    for ds, g_ds in pw_df.groupby("Dataset", sort=True):
        ds_diffs = []
        for _, g in g_ds.groupby("Ag_group", sort=True):
            if len(g) < 2:
                continue
            vals = g["binding_score"].to_numpy(dtype=float)
            d = np.abs(vals[:, None] - vals[None, :])[np.triu_indices(len(vals), 1)]
            ds_diffs.extend(d.tolist())
        if ds_diffs:
            arr = np.asarray(ds_diffs, dtype=float)
            all_diffs.extend(ds_diffs)
            per_ds[str(ds)] = {
                "n_all_pairs": int(len(arr)),
                "pct_noise_ties_le_0p30": round(100.0 * float((arr <= 0.30).mean()), 2),
                "pct_moderate_0p30_to_1p00": round(100.0 * float(((arr > 0.30) & (arr < 1.00)).mean()), 2),
                "pct_decisive_ge_1p00": round(100.0 * float((arr >= 1.00).mean()), 2),
            }
    arr_all = np.asarray(all_diffs, dtype=float)
    return {
        "n_total_within_target_pairs": int(len(arr_all)),
        "pct_noise_ties_le_0p30": round(100.0 * float((arr_all <= 0.30).mean()), 2),
        "pct_moderate_0p30_to_1p00": round(100.0 * float(((arr_all > 0.30) & (arr_all < 1.00)).mean()), 2),
        "pct_decisive_ge_1p00": round(100.0 * float((arr_all >= 1.00).mean()), 2),
        "by_dataset": per_ds,
    }


def _slice_gb(gb_t: tuple, ix: np.ndarray) -> tuple:
    if len(gb_t) == 3:
        (a1, e1, m1), (a2, e2, m2), (bo, be) = gb_t
        return ((a1[ix], e1[ix], m1[ix]), (a2[ix], e2[ix], m2[ix]), (bo[ix], be[ix]))
    (a1, e1, m1), (a2, e2, m2) = gb_t
    return ((a1[ix], e1[ix], m1[ix]), (a2[ix], e2[ix], m2[ix]))


def _fmt(val: float | None, ndigits: int = 3) -> str:
    return f"{val:.{ndigits}f}" if val is not None and np.isfinite(val) else "N/A"


def _concat_dc(parts: list[Any]) -> Any:
    import dataclasses
    cls = type(parts[0])
    kw = {}
    for f in dataclasses.fields(cls):
        vals = [getattr(p, f.name) for p in parts]
        if isinstance(vals[0], np.ndarray):
            kw[f.name] = np.concatenate(vals, axis=0)
        elif isinstance(vals[0], torch.Tensor):
            kw[f.name] = torch.cat(vals, dim=0)
        elif isinstance(vals[0], dict):
            kw[f.name] = {
                k: (
                    np.concatenate([v[k] for v in vals], axis=0)
                    if isinstance(vals[0][k], np.ndarray)
                    else torch.cat([v[k] for v in vals], dim=0)
                )
                for k in vals[0]
            }
        else:
            kw[f.name] = vals[0]
    return cls(**kw)


def _slim_tensors(sp: Any | None, sf: Any | None) -> None:
    if sp is not None:
        n = len(sp.y)
        z0 = np.empty((n, 0), dtype=np.float32)
        for attr in ("esm2_ab_1", "esm2_ab_2", "esm2_ag_1", "esm2_ag_2"):
            if hasattr(sp, attr):
                setattr(sp, attr, z0)
    if sf is not None:
        n = len(sf.is_cocrystal)
        z0 = np.empty((n, 0), dtype=np.float32)
        for attr in (
            "dccm_1", "dccm_2", "saprot_ab_1", "saprot_ab_2", "saprot_ag_1", "saprot_ag_2",
            "mint_ab_1", "mint_ab_2", "mint_ag_1", "mint_ag_2",
            "if1_emb_1", "if1_emb_2", "mpnn_emb_1", "mpnn_emb_2",
            "patch_edge_1", "patch_edge_2", "patch_adj_1", "patch_adj_2", "patch_mask_1", "patch_mask_2",
        ):
            if hasattr(sf, attr):
                setattr(sf, attr, z0)


def _shard_3d_worker(job: tuple[str, int, int, int]) -> tuple[str, int, Any, Any, tuple]:
    key, sid, lo, hi = job
    torch.set_num_threads(1)
    df = _G18[key].iloc[lo:hi].reset_index(drop=True)
    sp = assemble_split_tensors(df, _G18["all_assays"], _G18["esmc_cache"], _G18["ablang_cache"], _G18["esm2_cache"]) if key != "pool_df" else None
    sf = assemble_structure_pair_features(df, _G18["all_assays"], _G18["cc_bound"], _G18["inv_cache"], _G18["sap_cache"], _G18["mint_cache"], True)
    if _G18.get("only_logica", False):
        _slim_tensors(sp, sf)
    gb = (
        GearBind3StateTrainer._precompute_graphs(sf.patch_ca_1, sf.patch_cb_1, sf.patch_chain_1, sf.patch_seqpos_1),
        GearBind3StateTrainer._precompute_graphs(sf.patch_ca_2, sf.patch_cb_2, sf.patch_chain_2, sf.patch_seqpos_2),
    )
    return key, sid, sp, sf, gb


_G18: dict[str, Any] = {}


def _cluster_1d_worker(cl: str) -> dict[str, Any]:
    torch.set_num_threads(1)
    only_logica = _G18.get("only_logica", False)
    pool_df, pool_sp = _G18["pool_df"], _G18["pool_sp"]
    xb_pool, xk_pool = _G18["xb_pool"], _G18["xk_pool"]
    gp_sp, xb_gp, xk_gp = _G18["gp_sp"], _G18["xb_gp"], _G18["xk_gp"]
    gt_sp, xb_gt, xk_gt = _G18["gt_sp"], _G18["xb_gt"], _G18["xk_gt"]
    pw_esmc_ab, pw_esmc_ag, pw_y_tr = _G18["pw_esmc_ab"], _G18["pw_esmc_ag"], _G18["pw_y_tr"]
    gen_pw_x = _G18["gen_pw_x"]

    pool_c1, pool_c2 = _G18["pool_c1"], _G18["pool_c2"]
    pool_l1, pool_l2 = _G18["pool_l1"], _G18["pool_l2"]
    is_loc_pool = _G18["is_loc_pool"]
    gp_c1, gt_c1, gpw_c = _G18["gp_c1"], _G18["gt_c1"], _G18["gpw_c"]
    gp_l1, gp_l2 = _G18["gp_l1"], _G18["gp_l2"]
    gt_l1, gt_l2 = _G18["gt_l1"], _G18["gt_l2"]
    pw_tr_c, pw_tr_l = _G18["pw_tr_c"], _G18["pw_tr_l"]

    te_gp = np.flatnonzero(gp_c1 == cl)
    te_gt = np.flatnonzero(gt_c1 == cl)
    te_gpw = np.flatnonzero(gpw_c == cl)

    te_lins = set(gp_l1[te_gp]) | set(gp_l2[te_gp]) | set(gt_l1[te_gt]) | set(gt_l2[te_gt])
    keep_m = (pool_c1 != cl) & (pool_c2 != cl) & (~np.isin(pool_l1, list(te_lins))) & (~np.isin(pool_l2, list(te_lins)))
    kp_idx = np.flatnonzero(keep_m)
    rem_cls = sorted(set(pool_c1[kp_idx[is_loc_pool[kp_idx]]]))
    val_cls = set(rem_cls[1::5]) if len(rem_cls) >= 5 else set(rem_cls[:1])
    is_v = np.isin(pool_c1[kp_idx], list(val_cls)) & np.isin(pool_c2[kp_idx], list(val_cls))
    val_lins = set(pool_l1[kp_idx[is_v]]) | set(pool_l2[kp_idx[is_v]])
    is_tr = (
        (~np.isin(pool_c1[kp_idx], list(val_cls)))
        & (~np.isin(pool_c2[kp_idx], list(val_cls)))
        & (~np.isin(pool_l1[kp_idx], list(val_lins)))
        & (~np.isin(pool_l2[kp_idx], list(val_lins)))
    )

    tr_idx, va_idx = kp_idx[is_tr], kp_idx[is_v]
    tr_df, va_df = pool_df.iloc[tr_idx].reset_index(drop=True), pool_df.iloc[va_idx].reset_index(drop=True)
    tr_sp, va_sp = pool_sp.subset(tr_idx), pool_sp.subset(va_idx)
    w_tr = compute_stratum_weights(tr_df, tr_sp.is_ag_mut)

    m_log = LogiCA3StateTrainer(seed=42, epochs=16).fit(tr_sp, va_sp, va_df, w_tr)
    if only_logica:
        def _pred_log_only(ix: np.ndarray, sp_all: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            if len(ix) == 0:
                return {}
            esp = sp_all.subset(ix)
            return {"LogiCA (1D Logit Verifier)": m_log.predict_3state(esp)}

        return {
            "cl": cl,
            "te_gp": te_gp,
            "te_gt": te_gt,
            "te_gpw": te_gpw,
            "gp_preds": _pred_log_only(te_gp, gp_sp),
            "gt_preds": _pred_log_only(te_gt, gt_sp),
            "sc_ridge_gpw": None,
        }

    # Fit 1D supervised verifiers
    m_bio = AntisymmetricLinearRanker(use_neutral_nullspace=True).fit(xb_pool[tr_idx], tr_sp.y, w_tr, xb_pool[va_idx], va_df, va_sp.is_ag_mut, labels_tr=tr_sp.labels)
    m_km = AntisymmetricLinearRanker(use_neutral_nullspace=True).fit(xk_pool[tr_idx], tr_sp.y, w_tr, xk_pool[va_idx], va_df, va_sp.is_ag_mut, labels_tr=tr_sp.labels)

    xm_tr = build_odd_linear_matrix([tr_sp.esmc_ab_1, tr_sp.esmc_ag_1], [tr_sp.esmc_ab_2, tr_sp.esmc_ag_2], include_unit_direction=True)
    xm_va = build_odd_linear_matrix([va_sp.esmc_ab_1, va_sp.esmc_ag_1], [va_sp.esmc_ab_2, va_sp.esmc_ag_2], include_unit_direction=True)
    xr_tr = build_odd_linear_matrix([tr_sp.esmc_ab_1, tr_sp.esmc_ag_1], [tr_sp.esmc_ab_2, tr_sp.esmc_ag_2], include_unit_direction=False)
    xr_va = build_odd_linear_matrix([va_sp.esmc_ab_1, va_sp.esmc_ag_1], [va_sp.esmc_ab_2, va_sp.esmc_ag_2], include_unit_direction=False)
    m_ms = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(xm_tr, tr_sp.y, w_tr, xm_va, va_df, va_sp.is_ag_mut, labels_tr=tr_sp.labels)
    m_rw = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(xr_tr, tr_sp.y, None, xr_va, va_df, va_sp.is_ag_mut, labels_tr=tr_sp.labels)
    e2h = TwoHeadRegimeRanker(local_head=m_ms, global_head=m_rw)
    dv_tr, dv_va = e2h.predict(tr_sp, xm_tr, xr_tr), e2h.predict(va_sp, xm_va, xr_va)
    imp_e = SymmetricImpactClassifier(seed=42).fit(tr_sp, va_sp, dv_tr, dv_va, val_df=va_df)
    hurd_e = OrdinalHurdleRanker(e2h, imp_e).calibrate(va_df, va_sp, dv_va)
    m_bil = NeuralPairwiseTrainer(arch="bilinear", seed=42, epochs=18).fit(tr_sp, va_sp, va_df, w_tr)

    # Scalar pKd Ridge on disjoint pointwise pool
    kp_pw = (pw_tr_c != cl) & (~np.isin(pw_tr_l, list(te_lins)))
    x_pw_tr = np.concatenate([pw_esmc_ab[kp_pw], pw_esmc_ag[kp_pw]], axis=1)
    reg_ridge = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(x_pw_tr, pw_y_tr[kp_pw])
    sc_ridge_gpw = reg_ridge.predict(gen_pw_x[te_gpw]) if len(te_gpw) > 0 else np.zeros(0)

    def _pred_bundle(ix: np.ndarray, sp_all: Any, xb_all: np.ndarray, xk_all: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if len(ix) == 0:
            return {}
        esp = sp_all.subset(ix)
        out_b: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        out_b["ESMC-300M (1D Gen. Log-Lik)"] = (esp.zs_dict["esmc_300m_zeroshot"], esp.zs_impact_dict["esmc_300m_zeroshot"])
        out_b["AbLang2-Paired (1D Gen. Log-Lik)"] = (esp.zs_dict["ablang2_zeroshot"], esp.zs_impact_dict["ablang2_zeroshot"])
        out_b["ESMC+AbLang2 (1D Gen. Log-Lik)"] = (esp.zs_dict["esmc_ablang2_zeroshot"], esp.zs_impact_dict["esmc_ablang2_zeroshot"])
        db = m_bio.predict(xb_all[ix])
        dk = m_km.predict(xk_all[ix])
        out_b["Biophys+AAC (Linear)"] = (db, np.abs(db))
        out_b["k-mer Dipeptide (Ridge)"] = (dk, np.abs(dk))
        xm_e = build_odd_linear_matrix([esp.esmc_ab_1, esp.esmc_ag_1], [esp.esmc_ab_2, esp.esmc_ag_2], include_unit_direction=True)
        xr_e = build_odd_linear_matrix([esp.esmc_ab_1, esp.esmc_ag_1], [esp.esmc_ab_2, esp.esmc_ag_2], include_unit_direction=False)
        dms = m_ms.predict(xm_e)
        out_b["ESMC Multi-Scale (Binary DPO)"] = (dms, np.abs(dms))
        out_b["ESMC-300M (3-State Verifier)"] = hurd_e.predict_3state(esp, e2h.predict(esp, xm_e, xr_e))
        out_b["ESMC+AbLang2 (Bilinear Verifier)"] = m_bil.predict_3state(esp)
        out_b["LogiCA (1D Logit Verifier)"] = m_log.predict_3state(esp)
        return out_b

    return {
        "cl": cl,
        "te_gp": te_gp,
        "te_gt": te_gt,
        "te_gpw": te_gpw,
        "gp_preds": _pred_bundle(te_gp, gp_sp, xb_gp, xk_gp),
        "gt_preds": _pred_bundle(te_gt, gt_sp, xb_gt, xk_gt),
        "sc_ridge_gpw": sc_ridge_gpw,
    }


def _cluster_3d_worker(cl: str) -> dict[str, Any]:
    torch.set_num_threads(1)
    only_logica = _G18.get("only_logica", False)
    pool_df, pool_sp, pool_sf, gb_pool = _G18["pool_df"], _G18["pool_sp"], _G18["pool_sf"], _G18["gb_pool"]
    gp3_sp, gp3_sf, gb_gp3 = _G18["gp3_sp"], _G18["gp3_sf"], _G18["gb_gp3"]
    gt3_sp, gt3_sf, gb_gt3 = _G18["gt3_sp"], _G18["gt3_sf"], _G18["gb_gt3"]

    pool_c1, pool_c2 = _G18["pool_c1"], _G18["pool_c2"]
    pool_l1, pool_l2 = _G18["pool_l1"], _G18["pool_l2"]
    is_loc_pool = _G18["is_loc_pool"]
    gp3_c1, gt3_c1 = _G18["gp3_c1"], _G18["gt3_c1"]
    gp3_l1, gp3_l2 = _G18["gp3_l1"], _G18["gp3_l2"]
    gt3_l1, gt3_l2 = _G18["gt3_l1"], _G18["gt3_l2"]

    te_gp3 = np.flatnonzero(gp3_c1 == cl)
    te_gt3 = np.flatnonzero(gt3_c1 == cl)
    te_lins = set(gp3_l1[te_gp3]) | set(gp3_l2[te_gp3]) | set(gt3_l1[te_gt3]) | set(gt3_l2[te_gt3])

    keep_m = (pool_c1 != cl) & (pool_c2 != cl) & (~np.isin(pool_l1, list(te_lins))) & (~np.isin(pool_l2, list(te_lins)))
    kp_idx = np.flatnonzero(keep_m)
    rem_cls = sorted(set(pool_c1[kp_idx[is_loc_pool[kp_idx]]]))
    val_cls = set(rem_cls[1::5]) if len(rem_cls) >= 5 else set(rem_cls[:1])
    is_v = np.isin(pool_c1[kp_idx], list(val_cls)) & np.isin(pool_c2[kp_idx], list(val_cls))
    val_lins = set(pool_l1[kp_idx[is_v]]) | set(pool_l2[kp_idx[is_v]])
    is_tr = (
        (~np.isin(pool_c1[kp_idx], list(val_cls)))
        & (~np.isin(pool_c2[kp_idx], list(val_cls)))
        & (~np.isin(pool_l1[kp_idx], list(val_lins)))
        & (~np.isin(pool_l2[kp_idx], list(val_lins)))
    )

    tr_idx, va_idx = kp_idx[is_tr], kp_idx[is_v]
    tr_df, va_df = pool_df.iloc[tr_idx].reset_index(drop=True), pool_df.iloc[va_idx].reset_index(drop=True)
    tr_sp, va_sp = pool_sp.subset(tr_idx), pool_sp.subset(va_idx)
    tr_sf, va_sf = pool_sf.subset(tr_idx), pool_sf.subset(va_idx)
    gb_tr, gb_va = _slice_gb(gb_pool, tr_idx), _slice_gb(gb_pool, va_idx)
    w_tr = compute_stratum_weights(tr_df, tr_sp.is_ag_mut)

    m_log = LogiCA3StateTrainer(seed=42, epochs=15).fit(tr_sp, va_sp, va_df, w_tr)

    if only_logica:
        def _pred_3d_log_only(ix: np.ndarray, sp_all: Any, sf_all: Any, gb_all: tuple) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            if len(ix) == 0:
                return {}
            esp = sp_all.subset(ix)
            return {
                "LogiCA (1D Logit Verifier)": m_log.predict_3state(esp),
            }

        return {
            "cl": cl,
            "te_gp3": te_gp3,
            "te_gt3": te_gt3,
            "gp3_preds": _pred_3d_log_only(te_gp3, gp3_sp, gp3_sf, gb_gp3),
            "gt3_preds": _pred_3d_log_only(te_gt3, gt3_sp, gt3_sf, gb_gt3),
        }

    pred_sap = fit_3state_foundation_hurdle(
        tr_sf.saprot_ab_1, tr_sf.saprot_ab_2, tr_sf.saprot_ag_1, tr_sf.saprot_ag_2,
        np.concatenate([tr_sf.zs_dir["saprot_zs"][:, None], tr_sp.biophys_odd], axis=1), tr_sf.struct_even[:, :9],
        va_sf.saprot_ab_1, va_sf.saprot_ab_2, va_sf.saprot_ag_1, va_sf.saprot_ag_2,
        np.concatenate([va_sf.zs_dir["saprot_zs"][:, None], va_sp.biophys_odd], axis=1), va_sf.struct_even[:, :9],
        tr_sp, va_sp, va_df, w_tr, seed=42,
    )
    m_anti = ANTIPASTI3StateTrainer(seed=42, epochs=14).fit(tr_sf, tr_sp, va_sf, va_sp, va_df, w_tr)
    m_gb = GearBind3StateTrainer(seed=42, epochs=14).fit(tr_sf, tr_sp, va_sf, va_sp, va_df, w_tr, precomp_tr_g=gb_tr, precomp_val_g=gb_va)
    pred_walle = fit_walle_affinity_3state(tr_sf, tr_sp, va_sf, va_sp, va_df, w_tr, seed=42)

    def _pred_3d_bundle(ix: np.ndarray, sp_all: Any, sf_all: Any, gb_all: tuple) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if len(ix) == 0:
            return {}
        esp = sp_all.subset(ix)
        esf = sf_all.subset(ix)
        egb = _slice_gb(gb_all, ix)
        out_b: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        out_b["ESMC+AbLang2 (1D Gen. Log-Lik)"] = (esp.zs_dict["esmc_ablang2_zeroshot"], esp.zs_impact_dict["esmc_ablang2_zeroshot"])
        out_b["PRODIGY+SASA (3D Energy ZS)"] = (esf.zs_dir["prodigy_sasa"], esf.zs_imp["prodigy_sasa"])
        out_b["ProteinMPNN (3D Gen. Log-Lik)"] = (esf.zs_dir["protein_mpnn"], esf.zs_imp["protein_mpnn"])
        out_b["AntiFold (3D Ab-Ag Gen. Log-Lik)"] = (esf.zs_dir["antifold"], esf.zs_imp["antifold"])
        out_b["ESM-IF1 (3D Gen. Log-Lik)"] = (esf.zs_dir["esm_if1"], esf.zs_imp["esm_if1"])
        out_b["SaProt-650M (3D-Token Gen. Log-Lik)"] = (esf.zs_dir["saprot_zs"], esf.zs_imp["saprot_zs"])
        out_b["MINT-650M (Multi-Chain Gen. Log-Lik)"] = (esf.zs_dir["mint_zs"], esf.zs_imp["mint_zs"])
        out_b["LogiCA (1D Logit Verifier)"] = m_log.predict_3state(esp)
        out_b["SaProt-650M (3-State 3D Verifier)"] = pred_sap(
            esp, esf.saprot_ab_1, esf.saprot_ab_2, esf.saprot_ag_1, esf.saprot_ag_2,
            np.concatenate([esf.zs_dir["saprot_zs"][:, None], esp.biophys_odd], axis=1), esf.struct_even[:, :9],
        )
        out_b["ANTIPASTI (Normal-Mode 3D Verifier)"] = m_anti.predict_3state(esf, esp)
        out_b["GearBind (3D Geometric GNN Verifier)"] = m_gb.predict_3state(esf, _precomp_g=egb)
        out_b["WALLE-Affinity (3D Graph Verifier)"] = pred_walle(esf, esp)
        return out_b

    return {
        "cl": cl,
        "te_gp3": te_gp3,
        "te_gt3": te_gt3,
        "gp3_preds": _pred_3d_bundle(te_gp3, gp3_sp, gp3_sf, gb_gp3),
        "gt3_preds": _pred_3d_bundle(te_gt3, gt3_sp, gt3_sf, gb_gt3),
    }


def main() -> None:
    t0 = time.time()
    root = get_repo_root()
    bdir = root / "data" / "benchmark"
    edir = root / "data" / "embeddings"
    sdir = root / "data" / "structures"
    res_dir = root / "results"

    print("1. Assembling canonical AbRank-Generative splits (1,354 wet-lab generative AI designs)...", flush=True)
    kd_raw = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False)
    vhh_raw = pd.read_csv(bdir / "assays_vhh.tsv", sep="\t", low_memory=False)
    kd_raw["Ab_light_chain_seq"] = kd_raw["Ab_light_chain_seq"].fillna("").astype(str)
    vhh_raw["Ab_light_chain_seq"] = ""

    gen_kd = kd_raw[kd_raw["Dataset"].isin(GEN_KD_DATASETS)].copy()
    gen_vhh = vhh_raw[vhh_raw["Dataset"].isin(GEN_VHH_DATASETS)].copy()
    gen_pw = pd.concat([gen_kd, gen_vhh], ignore_index=True)

    gen_loc, gen_cand = build_within_antigen_pairs(
        gen_pw, m=1.0, m_neutral=0.30, include_neutral=True, neutral_ratio=1.0,
        modality="KD_Generative", ab_cap=4, per_ag_cap=600, include_candidates=True, seed=42, prefix="GEN",
    )
    gen_pairs = pd.concat([gen_loc, gen_cand], ignore_index=True).reset_index(drop=True)

    gen_split_dir = bdir / "splits" / "generative"
    gen_split_dir.mkdir(parents=True, exist_ok=True)
    gen_pw.to_csv(gen_split_dir / "generative_panels.tsv", sep="\t", index=False)
    gen_pairs.to_csv(gen_split_dir / "generative_pairs.tsv", sep="\t", index=False)

    dpo_audit = _audit_dpo_preference_noise(gen_pw)
    print(
        f"   Exported {len(gen_pw)} generative designs & {len(gen_pairs)} 3-state preference pairs! "
        f"DPO Noise-Tie Contamination: {dpo_audit['pct_noise_ties_le_0p30']:.1f}%",
        flush=True,
    )

    # Load base 1D PLM caches + extract incremental generative cache if needed
    esmc_base = torch.load(edir / "esmc_300m.pt", map_location="cpu", weights_only=False)
    ablang_base = torch.load(edir / "ablang2_paired.pt", map_location="cpu", weights_only=False)
    esm2_base = torch.load(edir / "esm2_150m.pt", map_location="cpu", weights_only=False)

    need_seqs = sorted(
        {
            s for s in set(gen_pw["Ab_heavy_chain_seq"]).union(gen_pw["Ab_light_chain_seq"]).union(gen_pw["Ag_seq"])
            if s and seq_hash(str(s)) not in esmc_base["means"]
        }
    )
    need_pairs = sorted(
        {
            (str(h), str(l))
            for h, l in zip(gen_pw["Ab_heavy_chain_seq"], gen_pw["Ab_light_chain_seq"])
            if h and l and pair_hash(str(h), str(l)) not in ablang_base["paired_means"]
        }
    )

    extra_esmc_p = edir / "generative_extra_esmc_300m.pt"
    extra_esm2_p = edir / "generative_extra_esm2_150m.pt"
    extra_ablang_p = edir / "generative_extra_ablang2.pt"

    extra_esmc = extract_hf_plm_cache(need_seqs, extra_esmc_p, model_name="Synthyra/ESMplusplus_small", trust_remote_code=True, n_workers=16) if (need_seqs or extra_esmc_p.exists()) else {}
    extra_esm2 = extract_hf_plm_cache(need_seqs, extra_esm2_p, model_name="facebook/esm2_t30_150M_UR50D", trust_remote_code=False, n_workers=16) if (need_seqs or extra_esm2_p.exists()) else {}
    extra_ablang = extract_ablang2_cache(need_pairs, extra_ablang_p, n_workers=16) if (need_pairs or extra_ablang_p.exists()) else {}

    esmc_cache = _merge_1d_cache(esmc_base, extra_esmc)
    esm2_cache = _merge_1d_cache(esm2_base, extra_esm2)
    ablang_cache = _merge_1d_cache(ablang_base, extra_ablang)
    del esmc_base, esm2_base, ablang_base, extra_esmc, extra_esm2, extra_ablang
    torch.set_num_threads(1)

    all_assays = pd.concat([kd_raw, vhh_raw], ignore_index=True).drop_duplicates("AbAgID").set_index("AbAgID")
    all_assays["Ab_light_chain_seq"] = all_assays["Ab_light_chain_seq"].fillna("").astype(str)

    split_dir = bdir / "splits" / "unseen_ag"
    pool_df = pd.concat([pd.read_csv(split_dir / f"{s}.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")], ignore_index=True)
    is_loc_pool = pool_df["Category"].isin(["L1", "L2", "L3"]).to_numpy()

    gen_tourn_1d = _build_all_panel_pairs(gen_pw, min_size=5)
    print(f"2. Assembling 1D features across pool ({len(pool_df)}), gen_pairs ({len(gen_pairs)}), & tournament ({len(gen_tourn_1d)})...", flush=True)

    used_1d_ids = sorted(
        set(pool_df["AbAgID_1"]).union(pool_df["AbAgID_2"])
        .union(gen_pairs["AbAgID_1"]).union(gen_pairs["AbAgID_2"])
        .union(gen_tourn_1d["AbAgID_1"]).union(gen_tourn_1d["AbAgID_2"])
        .union(gen_pw["AbAgID"])
    )
    sub_1d_df = all_assays.loc[used_1d_ids].reset_index()
    bio_map = build_complex_handcrafted_table(sub_1d_df, include_dipeptides=False)
    km_map = build_complex_handcrafted_table(sub_1d_df, include_dipeptides=True)
    id2r = {aid: i for i, aid in enumerate(used_1d_ids)}
    bio_arr = np.asarray([bio_map[aid] for aid in used_1d_ids], dtype=np.float32)
    km_arr = np.asarray([km_map[aid] for aid in used_1d_ids], dtype=np.float32)

    def _pair_hc(df_p: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        i1 = np.array([id2r[a] for a in df_p["AbAgID_1"]], dtype=int)
        i2 = np.array([id2r[b] for b in df_p["AbAgID_2"]], dtype=int)
        return bio_arr[i1] - bio_arr[i2], km_arr[i1] - km_arr[i2]

    ckpt_path = res_dir / "checkpoints" / "generative_oof_ckpt.pt"
    only_logica = ckpt_path.exists()
    prev_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False) if only_logica else None

    pool_sp = assemble_split_tensors(pool_df, all_assays, esmc_cache, ablang_cache, esm2_cache)
    gp_sp = assemble_split_tensors(gen_pairs, all_assays, esmc_cache, ablang_cache, esm2_cache)
    gt_sp = assemble_split_tensors(gen_tourn_1d, all_assays, esmc_cache, ablang_cache, esm2_cache)
    if only_logica:
        _slim_tensors(pool_sp, None)
        _slim_tensors(gp_sp, None)
        _slim_tensors(gt_sp, None)
    xb_pool, xk_pool = _pair_hc(pool_df)
    xb_gp, xk_gp = _pair_hc(gen_pairs)
    xb_gt, xk_gt = _pair_hc(gen_tourn_1d)

    # Pointwise ESMC features for Scalar pKd Ridge
    pw_tr_all = pd.concat([pd.read_csv(split_dir / f"{s}_pointwise.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")], ignore_index=True)
    pw_tr_all["Ab_light_chain_seq"] = pw_tr_all["Ab_light_chain_seq"].fillna("").astype(str)
    pw_tr_all = pw_tr_all[[seq_hash(str(h)) in esmc_cache["means"] and (not l or seq_hash(str(l)) in esmc_cache["means"]) and seq_hash(str(a)) in esmc_cache["means"] for h, l, a in zip(pw_tr_all["Ab_heavy_chain_seq"], pw_tr_all["Ab_light_chain_seq"], pw_tr_all["Ag_seq"])]].reset_index(drop=True)

    def _pw_esmc_mat(df_w: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ab_m, ag_m = [], []
        for h, l, a in zip(df_w["Ab_heavy_chain_seq"], df_w["Ab_light_chain_seq"], df_w["Ag_seq"]):
            mh = esmc_cache["means"][seq_hash(str(h))]
            ml = esmc_cache["means"][seq_hash(str(l))] if l else mh
            ma = esmc_cache["means"][seq_hash(str(a))]
            ab_m.append(0.5 * (mh + ml))
            ag_m.append(ma)
        return np.asarray(ab_m, dtype=np.float32), np.asarray(ag_m, dtype=np.float32)

    pw_esmc_ab, pw_esmc_ag = _pw_esmc_mat(pw_tr_all)
    gpw_ab, gpw_ag = _pw_esmc_mat(gen_pw)
    gen_pw_x = np.concatenate([gpw_ab, gpw_ag], axis=1)

    pool_c1 = pool_df["AbAgID_1"].map(all_assays["Ag_cluster"]).astype(str).to_numpy()
    pool_c2 = pool_df["AbAgID_2"].map(all_assays["Ag_cluster"]).astype(str).to_numpy()
    pool_l1 = pool_df["AbAgID_1"].map(all_assays["Ab_lineage"]).astype(str).to_numpy()
    pool_l2 = pool_df["AbAgID_2"].map(all_assays["Ab_lineage"]).astype(str).to_numpy()
    gp_c1 = gen_pairs["AbAgID_1"].map(all_assays["Ag_cluster"]).astype(str).to_numpy()
    gt_c1 = gen_tourn_1d["AbAgID_1"].map(all_assays["Ag_cluster"]).astype(str).to_numpy()
    gpw_c = gen_pw["Ag_cluster"].astype(str).to_numpy()
    gp_l1 = gen_pairs["AbAgID_1"].map(all_assays["Ab_lineage"]).astype(str).to_numpy()
    gp_l2 = gen_pairs["AbAgID_2"].map(all_assays["Ab_lineage"]).astype(str).to_numpy()
    gt_l1 = gen_tourn_1d["AbAgID_1"].map(all_assays["Ab_lineage"]).astype(str).to_numpy()
    gt_l2 = gen_tourn_1d["AbAgID_2"].map(all_assays["Ab_lineage"]).astype(str).to_numpy()
    pw_tr_c = pw_tr_all["Ag_cluster"].astype(str).to_numpy()
    pw_tr_l = pw_tr_all["Ab_lineage"].astype(str).to_numpy()

    _G18.update(
        only_logica=only_logica,
        pool_df=pool_df, pool_sp=pool_sp, xb_pool=xb_pool, xk_pool=xk_pool,
        gp_sp=gp_sp, xb_gp=xb_gp, xk_gp=xk_gp,
        gt_sp=gt_sp, xb_gt=xb_gt, xk_gt=xk_gt,
        pw_esmc_ab=pw_esmc_ab, pw_esmc_ag=pw_esmc_ag, pw_y_tr=pw_tr_all["binding_score"].to_numpy(dtype=np.float32),
        gen_pw_x=gen_pw_x, is_loc_pool=is_loc_pool,
        pool_c1=pool_c1, pool_c2=pool_c2, pool_l1=pool_l1, pool_l2=pool_l2,
        gp_c1=gp_c1, gt_c1=gt_c1, gpw_c=gpw_c,
        gp_l1=gp_l1, gp_l2=gp_l2, gt_l1=gt_l1, gt_l2=gt_l2,
        pw_tr_c=pw_tr_c, pw_tr_l=pw_tr_l,
    )

    gen_clusters_1d = sorted(set(gp_c1) | set(gt_c1) | set(gpw_c))
    print(f"3. Running Track A (All 1,354 Generative Designs) across {len(gen_clusters_1d)} LOACO clusters (only_logica={only_logica})...", flush=True)
    m1d_names = [
        "ESMC-300M (1D Gen. Log-Lik)",
        "AbLang2-Paired (1D Gen. Log-Lik)",
        "ESMC+AbLang2 (1D Gen. Log-Lik)",
        "Biophys+AAC (Linear)",
        "k-mer Dipeptide (Ridge)",
        "ESMC Multi-Scale (Binary DPO)",
        "ESMC-300M (3-State Verifier)",
        "ESMC+AbLang2 (Bilinear Verifier)",
        "LogiCA (1D Logit Verifier)",
    ]
    if prev_ckpt is not None:
        oof_gp_1d = prev_ckpt["oof_gp_1d"]
        oof_gt_1d = prev_ckpt["oof_gt_1d"]
        oof_ridge_pw = prev_ckpt["oof_ridge_pw"]
    else:
        oof_gp_1d = {m: (np.zeros(len(gen_pairs)), np.zeros(len(gen_pairs))) for m in m1d_names}
        oof_gt_1d = {m: np.zeros(len(gen_tourn_1d)) for m in m1d_names}
        oof_ridge_pw = np.zeros(len(gen_pw))

    with mp.get_context("fork").Pool(min(len(gen_clusters_1d), 10)) as pool:
        for out in pool.imap_unordered(_cluster_1d_worker, gen_clusters_1d):
            igp, igt, igpw = out["te_gp"], out["te_gt"], out["te_gpw"]
            if len(igpw) > 0 and out["sc_ridge_gpw"] is not None:
                oof_ridge_pw[igpw] = out["sc_ridge_gpw"]
            for m, (dv, iv) in out["gp_preds"].items():
                oof_gp_1d[m][0][igp] = dv
                oof_gp_1d[m][1][igp] = iv
            for m, (dv, _) in out["gt_preds"].items():
                oof_gt_1d[m][igt] = dv

    track_a_results: dict[str, dict] = {}
    ridge_smap = dict(zip(gen_pw["AbAgID"], oof_ridge_pw))
    rd_diff = (gen_pairs["AbAgID_1"].map(ridge_smap) - gen_pairs["AbAgID_2"].map(ridge_smap)).to_numpy(dtype=float)
    rm_pair = compute_per_antigen_macro_metrics(gen_pairs, rd_diff, np.abs(rd_diff), min_pairs=3, n_bootstrap=1000, seed=42)
    track_a_results["Scalar pKd Ridge (Reward Probe)"] = {
        "pairwise_3state": {
            "auc_dir": rm_pair["auc_dir"],
            "auc_impact": rm_pair["auc_impact"],
            "auc_ordinal": rm_pair["auc_ordinal"],
            "ord_ci95": rm_pair["ord_ci95"],
        },
        "best_of_n_funnel": _evaluate_best_of_n_funnel(gen_pw, ridge_smap, min_size=5),
    }

    for m in m1d_names:
        mp_m = compute_per_antigen_macro_metrics(gen_pairs, oof_gp_1d[m][0], oof_gp_1d[m][1], min_pairs=3, n_bootstrap=1000, seed=42)
        t_smap = _tournament_scores(gen_tourn_1d, oof_gt_1d[m])
        track_a_results[m] = {
            "pairwise_3state": {
                "auc_dir": mp_m["auc_dir"],
                "auc_impact": mp_m["auc_impact"],
                "auc_ordinal": mp_m["auc_ordinal"],
                "ord_ci95": mp_m["ord_ci95"],
            },
            "best_of_n_funnel": _evaluate_best_of_n_funnel(gen_pw, t_smap, min_size=5),
        }
        print(
            f"  [Track A 1,354] {m:34s} Ord={_fmt(mp_m['auc_ordinal'])} | "
            f"TopBind={_fmt(track_a_results[m]['best_of_n_funnel']['mean_top_binder_auc'])} | "
            f"Prec@20%={_fmt(track_a_results[m]['best_of_n_funnel']['precision_at_top20_pct'])} | "
            f"Regret={_fmt(track_a_results[m]['best_of_n_funnel']['mean_top20_regret_pkd'])}",
            flush=True,
        )

    # =========================================================================
    # Track B: 3D-Resolved Generative Campaigns (`351` designs across IgDesign, BoltzGen_VHH, Germinal, Adaptyv, Shanker)
    # =========================================================================
    cc_bound = torch.load(sdir / "complex_cache.pt", map_location="cpu", weights_only=False)
    inv_cache = torch.load(edir / "inverse_folding_scaffolds.pt", map_location="cpu", weights_only=False)
    sap_cache = None if only_logica else torch.load(edir / "saprot_650m.pt", map_location="cpu", weights_only=False)
    mint_cache = None if only_logica else torch.load(edir / "mint_650m.pt", map_location="cpu", weights_only=False)

    in_3d_set = set(cc_bound["abag_to_scaffold"].keys())
    gen_pw_3d = gen_pw[gen_pw["AbAgID"].isin(in_3d_set)].reset_index(drop=True)
    gen_pairs_3d = gen_pairs[gen_pairs["AbAgID_1"].isin(in_3d_set) & gen_pairs["AbAgID_2"].isin(in_3d_set)].reset_index(drop=True)
    gen_tourn_3d = _build_all_panel_pairs(gen_pw_3d, min_size=5)

    print(f"\n4. Sharding 3D feature assembly for Track B ({len(gen_pw_3d)} 3D-resolved generative designs, {len(gen_pairs_3d)} pairs)...", flush=True)
    _G18.update(
        gen_pairs_3d=gen_pairs_3d, gen_tourn_3d=gen_tourn_3d,
        all_assays=all_assays, esmc_cache=esmc_cache, ablang_cache=ablang_cache, esm2_cache=esm2_cache,
        cc_bound=cc_bound, inv_cache=inv_cache, sap_cache=sap_cache, mint_cache=mint_cache,
    )
    shard_jobs = []
    for k_t, df_t, n_sh in (("pool_df", pool_df, 12), ("gen_pairs_3d", gen_pairs_3d, 2), ("gen_tourn_3d", gen_tourn_3d, 6)):
        edges = np.linspace(0, len(df_t), n_sh + 1, dtype=int)
        for sid in range(n_sh):
            shard_jobs.append((k_t, sid, int(edges[sid]), int(edges[sid + 1])))
    sh_3d: dict[str, dict[int, tuple]] = {"pool_df": {}, "gen_pairs_3d": {}, "gen_tourn_3d": {}}
    with mp.get_context("fork").Pool(12) as pool:
        for k_o, sid_o, sp_o, sf_o, gb_o in pool.imap_unordered(_shard_3d_worker, shard_jobs):
            sh_3d[k_o][sid_o] = (sp_o, sf_o, gb_o)

    def _merge_gb(parts: list[tuple]) -> tuple:
        if len(parts[0]) == 3:
            return (
                tuple(np.concatenate([p[0][j] for p in parts], axis=0) for j in range(3)),
                tuple(np.concatenate([p[1][j] for p in parts], axis=0) for j in range(3)),
                tuple(np.concatenate([p[2][j] for p in parts], axis=0) for j in range(2)),
            )
        return (
            tuple(np.concatenate([p[0][j] for p in parts], axis=0) for j in range(3)),
            tuple(np.concatenate([p[1][j] for p in parts], axis=0) for j in range(3)),
        )

    p_ord = [sh_3d["pool_df"][i] for i in sorted(sh_3d["pool_df"])]
    pool_sf = _concat_dc([x[1] for x in p_ord])
    gb_pool = _merge_gb([x[2] for x in p_ord])

    gp_ord = [sh_3d["gen_pairs_3d"][i] for i in sorted(sh_3d["gen_pairs_3d"])]
    gp3_sp = _concat_dc([x[0] for x in gp_ord])
    gp3_sf = _concat_dc([x[1] for x in gp_ord])
    gb_gp3 = _merge_gb([x[2] for x in gp_ord])

    gt_ord = [sh_3d["gen_tourn_3d"][i] for i in sorted(sh_3d["gen_tourn_3d"])]
    gt3_sp = _concat_dc([x[0] for x in gt_ord])
    gt3_sf = _concat_dc([x[1] for x in gt_ord])
    gb_gt3 = _merge_gb([x[2] for x in gt_ord])
    del sh_3d
    for k_drop in ("esmc_cache", "ablang_cache", "esm2_cache", "cc_bound", "inv_cache", "sap_cache", "mint_cache"):
        _G18.pop(k_drop, None)
    del esmc_cache, ablang_cache, esm2_cache, cc_bound, inv_cache, sap_cache, mint_cache

    gp3_c1 = gen_pairs_3d["AbAgID_1"].map(all_assays["Ag_cluster"]).astype(str).to_numpy()
    gt3_c1 = gen_tourn_3d["AbAgID_1"].map(all_assays["Ag_cluster"]).astype(str).to_numpy()
    gp3_l1 = gen_pairs_3d["AbAgID_1"].map(all_assays["Ab_lineage"]).astype(str).to_numpy()
    gp3_l2 = gen_pairs_3d["AbAgID_2"].map(all_assays["Ab_lineage"]).astype(str).to_numpy()
    gt3_l1 = gen_tourn_3d["AbAgID_1"].map(all_assays["Ab_lineage"]).astype(str).to_numpy()
    gt3_l2 = gen_tourn_3d["AbAgID_2"].map(all_assays["Ab_lineage"]).astype(str).to_numpy()

    _G18.update(
        pool_sf=pool_sf, gb_pool=gb_pool,
        gp3_sp=gp3_sp, gp3_sf=gp3_sf, gb_gp3=gb_gp3,
        gt3_sp=gt3_sp, gt3_sf=gt3_sf, gb_gt3=gb_gt3,
        gp3_c1=gp3_c1, gt3_c1=gt3_c1,
        gp3_l1=gp3_l1, gp3_l2=gp3_l2, gt3_l1=gt3_l1, gt3_l2=gt3_l2,
    )

    gen_clusters_3d = sorted(set(gp3_c1) | set(gt3_c1))
    m3d_names = [
        "ESMC+AbLang2 (1D Gen. Log-Lik)",
        "PRODIGY+SASA (3D Energy ZS)",
        "ProteinMPNN (3D Gen. Log-Lik)",
        "AntiFold (3D Ab-Ag Gen. Log-Lik)",
        "ESM-IF1 (3D Gen. Log-Lik)",
        "SaProt-650M (3D-Token Gen. Log-Lik)",
        "MINT-650M (Multi-Chain Gen. Log-Lik)",
        "LogiCA (1D Logit Verifier)",
        "SaProt-650M (3-State 3D Verifier)",
        "ANTIPASTI (Normal-Mode 3D Verifier)",
        "GearBind (3D Geometric GNN Verifier)",
        "WALLE-Affinity (3D Graph Verifier)",
    ]
    if prev_ckpt is not None:
        oof_gp_3d = prev_ckpt["oof_gp_3d"]
        oof_gt_3d = prev_ckpt["oof_gt_3d"]
    else:
        oof_gp_3d = {m: (np.zeros(len(gen_pairs_3d)), np.zeros(len(gen_pairs_3d))) for m in m3d_names}
        oof_gt_3d = {m: np.zeros(len(gen_tourn_3d)) for m in m3d_names}

    print(f"5. Running Track B (351 3D-Resolved Generative Designs) across {len(gen_clusters_3d)} LOACO clusters (only_logica={only_logica})...", flush=True)
    with mp.get_context("fork").Pool(min(len(gen_clusters_3d), 10)) as pool:
        for out in pool.imap_unordered(_cluster_3d_worker, gen_clusters_3d):
            igp3, igt3 = out["te_gp3"], out["te_gt3"]
            for m, (dv, iv) in out["gp3_preds"].items():
                oof_gp_3d[m][0][igp3] = dv
                oof_gp_3d[m][1][igp3] = iv
            for m, (dv, _) in out["gt3_preds"].items():
                oof_gt_3d[m][igt3] = dv

    ckpt_dir = res_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "oof_gp_1d": oof_gp_1d, "oof_gt_1d": oof_gt_1d, "oof_ridge_pw": oof_ridge_pw,
            "oof_gp_3d": oof_gp_3d, "oof_gt_3d": oof_gt_3d,
        },
        ckpt_dir / "generative_oof_ckpt.pt",
    )

    track_b_results: dict[str, dict] = {}
    is_vhh_p3 = gen_pairs_3d["Dataset_1"].isin(GEN_VHH_DATASETS).to_numpy()
    is_igdes_p3 = (gen_pairs_3d["Dataset_1"] == "IgDesign").to_numpy()

    for m in m3d_names:
        mp_m = compute_per_antigen_macro_metrics(gen_pairs_3d, oof_gp_3d[m][0], oof_gp_3d[m][1], min_pairs=3, n_bootstrap=1000, seed=42)
        mp_vhh = compute_per_antigen_macro_metrics(gen_pairs_3d[is_vhh_p3].reset_index(drop=True), oof_gp_3d[m][0][is_vhh_p3], oof_gp_3d[m][1][is_vhh_p3], min_pairs=2)
        mp_igd = compute_per_antigen_macro_metrics(gen_pairs_3d[is_igdes_p3].reset_index(drop=True), oof_gp_3d[m][0][is_igdes_p3], oof_gp_3d[m][1][is_igdes_p3], min_pairs=3)
        t_smap = _tournament_scores(gen_tourn_3d, oof_gt_3d[m])
        track_b_results[m] = {
            "pairwise_3state": {
                "auc_dir": mp_m["auc_dir"],
                "auc_impact": mp_m["auc_impact"],
                "auc_ordinal": mp_m["auc_ordinal"],
                "ord_ci95": mp_m["ord_ci95"],
                "igdesign_ordinal_auc": mp_igd["auc_ordinal"],
                "denovo_vhh_boltzgen_germinal_ordinal_auc": mp_vhh["auc_ordinal"],
            },
            "best_of_n_funnel": _evaluate_best_of_n_funnel(gen_pw_3d, t_smap, min_size=5),
        }
        print(
            f"  [Track B 3D-351] {m:36s} Ord={_fmt(mp_m['auc_ordinal'])} (IgDes={_fmt(mp_igd['auc_ordinal'])}, DeNovoVHH={_fmt(mp_vhh['auc_ordinal'])}) | "
            f"TopBind={_fmt(track_b_results[m]['best_of_n_funnel']['mean_top_binder_auc'])} | "
            f"Prec@20%={_fmt(track_b_results[m]['best_of_n_funnel']['precision_at_top20_pct'])} | "
            f"Regret={_fmt(track_b_results[m]['best_of_n_funnel']['mean_top20_regret_pkd'])}",
            flush=True,
        )

    summary_payload = {
        "benchmark": "AbRank-Generative (--suite generative)",
        "n_total_generative_designs": int(len(gen_pw)),
        "n_total_antigens": int(gen_pw["Ag_group"].nunique()),
        "n_generative_pairs_all": int(len(gen_pairs)),
        "n_3d_resolved_generative_designs": int(len(gen_pw_3d)),
        "n_generative_pairs_3d": int(len(gen_pairs_3d)),
        "datasets": gen_pw["Dataset"].value_counts().to_dict(),
        "dpo_preference_noise_audit": dpo_audit,
        "track_a_all_1354_generative_designs": track_a_results,
        "track_b_351_3d_resolved_generative_designs": track_b_results,
    }
    out_json = res_dir / "generative_binder_benchmark.json"
    out_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    # Write clean Markdown summary report
    md_lines = [
        "# AbRank-Generative (`--suite generative`) — Generative Binder Reranking, Conditional Log-Likelihood, & DPO Verifier Benchmark",
        "",
        f"Evaluates generative sequence/inverse-folding log-likelihoods and 3-state reward/DPO verifiers directly on **`{len(gen_pw):,}` wet-lab SPR/BLI-assayed generative AI designs across `{gen_pw['Ag_group'].nunique()}` unique antigen targets** (`IgDesign`, `Origin1`, `AIntibody`, `BoltzGen_VHH`, `Germinal`, `Germinal_VHH`, `Adaptyv_EGFR_VHH`, `Shanker`) under strict Leave-One-Antigen-Cluster-Out (`<50%` Ag + `<80%` Ab) holdouts—avoiding circular *in silico* oracle bias.",
        "",
        "## 1. Noise-Buffered DPO Preference Audit Inside Real Generative Design Cohorts",
        "",
        f"- Across all **`{dpo_audit['n_total_within_target_pairs']:,}`** within-target candidate pairs in synthesized generative campaigns, **`{dpo_audit['pct_noise_ties_le_0p30']:.1f}%` have `|ΔpKd| ≤ 0.30` (`≤2×` experimental replicate noise)**, `{dpo_audit['pct_moderate_0p30_to_1p00']:.1f}%` lie in `0.30 < |ΔpKd| < 1.00`, and only **`{dpo_audit['pct_decisive_ge_1p00']:.1f}%` represent decisive `≥10×` (`|ΔpKd| ≥ 1.0`) improvements/depletions**.",
        "",
        "## 2. Track A — All `1,354` Wet-Lab Generative Designs (`20` Panels across `IgDesign`, `Origin1`, `AIntibody`, `BoltzGen_VHH`, `Germinal`, `Shanker`)",
        "",
        "| Model / Verifier | 3-State DPO (`Sign` / `Imp` / **`Ord`**) | `Top-Binder AUC` | `Prec@Top-20%` (`EF`) | `Top-20% Regret` (`ΔpKd` ↓) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for m, d in track_a_results.items():
        p3 = d["pairwise_3state"]
        fn = d["best_of_n_funnel"]
        md_lines.append(
            f"| **{m}** | `{_fmt(p3['auc_dir'])}` / `{_fmt(p3['auc_impact'])}` / **`{_fmt(p3['auc_ordinal'])}`** | "
            f"`{_fmt(fn['mean_top_binder_auc'])}` | **`{_fmt(fn['precision_at_top20_pct'])}`** (`{_fmt(fn['enrichment_factor_top20'], 2)}×`) | "
            f"`{_fmt(fn['mean_top20_regret_pkd'])}` |"
        )

    md_lines.extend([
        "",
        "## 3. Track B — `351` 3D-Complex-Resolved Generative Designs (`IgDesign`, `BoltzGen_VHH`, `Germinal`, `Adaptyv_EGFR_VHH`, `Shanker`)",
        "",
        "| Generative Log-Likelihood / 3D Verifier | All 3D Gen (`Sign` / `Imp` / **`Ord`**) | `IgDesign` Ord | *De Novo* `VHH` (`BoltzGen`+`Germinal`) Ord | `Top-Binder AUC` | `Prec@Top-20%` (`EF`) | `Regret` (`ΔpKd` ↓) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for m, d in track_b_results.items():
        p3 = d["pairwise_3state"]
        fn = d["best_of_n_funnel"]
        md_lines.append(
            f"| **{m}** | `{_fmt(p3['auc_dir'])}` / `{_fmt(p3['auc_impact'])}` / **`{_fmt(p3['auc_ordinal'])}`** | "
            f"`{_fmt(p3['igdesign_ordinal_auc'])}` | `{_fmt(p3['denovo_vhh_boltzgen_germinal_ordinal_auc'])}` | "
            f"`{_fmt(fn['mean_top_binder_auc'])}` | **`{_fmt(fn['precision_at_top20_pct'])}`** (`{_fmt(fn['enrichment_factor_top20'], 2)}×`) | "
            f"`{_fmt(fn['mean_top20_regret_pkd'])}` |"
        )

    (res_dir / "GENERATIVE_BINDER_BENCHMARK.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nSaved generative_binder_benchmark.json & GENERATIVE_BINDER_BENCHMARK.md in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
