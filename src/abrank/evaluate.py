"""Official AbRank v2 evaluation suite:
  1. Directional Sign ROC-AUC (`auc_dir`, on impactful pairs `label in {-1, +1}`)
  2. Impact / Hot-Spot Detection ROC-AUC (`auc_impact`, `|label| == 1` vs `label == 0`)
  3. 3-Class Ordinal Concordance (`auc_ordinal`, `-1 < 0 < +1`) & Selective Hit ROC-AUC (`+1` vs `{0, -1}`)
  4. Unfiltered Full-Panel Spearman rank correlation & Top-Binder Enrichment (`evaluate_pointwise_spearman`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def compute_3state_metrics(
    labels: np.ndarray,
    binding_diff: np.ndarray,
    pred_diff: np.ndarray,
    pred_impact: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Compute exact orientation-invariant 3-state pairwise metrics in O(N log N).

    Satisfies:
      - Invariance under flipping any subset `(1, 2) <-> (2, 1)` (`pred_diff -> -pred_diff`,
        `pred_impact -> +pred_impact`, `label -> -label`, `binding_diff -> -binding_diff`).
      - Random / zero predictor = `0.5000` on `auc_dir`, `auc_impact`, `auc_ordinal`, `auc_selective_hit`.
      - Oracle `pred_diff = binding_diff` = `1.0000` on all metrics.
    """
    lbl = np.asarray(labels, dtype=int)
    bdiff = np.asarray(binding_diff, dtype=np.float64)
    pdiff = np.asarray(pred_diff, dtype=np.float64)
    pimp = np.abs(pdiff) if pred_impact is None else np.asarray(pred_impact, dtype=np.float64)

    imp_mask = lbl != 0
    neu_mask = lbl == 0
    n_imp = int(imp_mask.sum())
    n_neu = int(neu_mask.sum())

    # 1. Directional Sign AUC (on impactful pairs)
    if n_imp > 0:
        y_imp = (bdiff[imp_mask] > 0).astype(int)
        p_imp = pdiff[imp_mask]
        acc_dir = float(np.mean((p_imp > 0) == (y_imp == 1)))
        auc_dir = float(
            roc_auc_score(
                np.concatenate([y_imp, 1 - y_imp]),
                np.concatenate([p_imp, -p_imp]),
            )
        )
    else:
        acc_dir = float("nan")
        auc_dir = float("nan")

    # 2. Impact / Hot-Spot Detection AUC (|label| == 1 vs label == 0)
    if n_imp > 0 and n_neu > 0:
        y_det = imp_mask.astype(int)
        auc_impact = float(roc_auc_score(y_det, pimp))
    else:
        auc_impact = float("nan")

    # 3. Symmetrized 3-Class Ordinal Concordance (-1 < 0 < +1) & Selective Hit AUC (+1 vs {0, -1})
    if n_imp > 0 and n_neu > 0:
        s_plus = np.where(bdiff[imp_mask] > 0, pdiff[imp_mask], -pdiff[imp_mask])
        s_minus = -s_plus
        s_zero = np.concatenate([pdiff[neu_mask], -pdiff[neu_mask]])

        # Concordance on (+1 vs 0), which equals concordance on (0 vs -1) by exact symmetry
        y_0p = np.concatenate([np.ones(n_imp, dtype=int), np.zeros(2 * n_neu, dtype=int)])
        p_0p = np.concatenate([s_plus, s_zero])
        auc_0p = float(roc_auc_score(y_0p, p_0p))

        w_pm = float(n_imp) * float(n_imp)
        w_0p = 2.0 * float(n_imp) * float(n_neu)
        auc_ordinal = (w_pm * auc_dir + 2.0 * w_0p * auc_0p) / (w_pm + 2.0 * w_0p)

        y_hit = np.concatenate([np.ones(n_imp, dtype=int), np.zeros(n_imp + 2 * n_neu, dtype=int)])
        p_hit = np.concatenate([s_plus, s_minus, s_zero])
        auc_hit = float(roc_auc_score(y_hit, p_hit))
    elif n_imp > 0:
        auc_ordinal = auc_dir
        auc_hit = auc_dir
    else:
        auc_ordinal = float("nan")
        auc_hit = float("nan")

    def _r(v: float) -> float | None:
        return round(float(v), 4) if np.isfinite(v) else None

    return {
        "n_pairs": int(len(lbl)),
        "n_impactful": n_imp,
        "n_neutral": n_neu,
        "accuracy": _r(acc_dir),
        "auc_sym": _r(auc_dir),
        "auc_dir": _r(auc_dir),
        "auc_impact": _r(auc_impact),
        "auc_ordinal": _r(auc_ordinal),
        "auc_selective_hit": _r(auc_hit),
    }


def compute_per_antigen_macro_metrics(
    pairs_df: pd.DataFrame,
    pred_diff: np.ndarray | pd.Series,
    pred_impact: np.ndarray | pd.Series | None = None,
    *,
    min_pairs: int = 5,
    min_class_pairs: int = 2,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> dict:
    """Compute Per-Antigen Macro-Averaged 3-state metrics over antigens with `>= min_pairs` pairs,
    while also preserving exact Pooled (Micro-Averaged) metrics (`pooled_auc_*`).

    Each eligible antigen group (or `(Ag_group_1, Assay_modality)` stratum when `pairs_df` spans
    multiple assay modalities) is scored independently with `compute_3state_metrics` and averaged
    with equal weight across antigens so mega-panels cannot dominate and `<min_pairs` singletons
    cannot introduce 0/1 coin-flip variance.
    """
    df = pairs_df.reset_index(drop=True)
    pdiff = np.asarray(pred_diff, dtype=np.float64)
    if len(pdiff) != len(df):
        raise ValueError(f"Expected {len(df)} predictions, got {len(pdiff)}.")
    pimp = np.abs(pdiff) if pred_impact is None else np.asarray(pred_impact, dtype=np.float64)

    lbl = (
        df["label"].to_numpy(dtype=int)
        if "label" in df.columns
        else np.where(df["binding_diff"].to_numpy(dtype=float) > 0, 1, -1)
    )
    bdiff = df["binding_diff"].to_numpy(dtype=np.float64)
    pooled = compute_3state_metrics(lbl, bdiff, pdiff, pimp)

    same_ag = np.ones(len(df), dtype=bool)
    if "Category" in df.columns:
        same_ag &= (df["Category"] != "Global").to_numpy()
    if "Ag_group_1" in df.columns and "Ag_group_2" in df.columns:
        same_ag &= (df["Ag_group_1"].astype(str) == df["Ag_group_2"].astype(str)).to_numpy()

    if int(same_ag.sum()) == 0 or "Ag_group_1" not in df.columns:
        out = {
            **pooled,
            "n_antigens_evaluated": 0,
            "pooled_auc_dir": pooled["auc_dir"],
            "pooled_auc_impact": pooled["auc_impact"],
            "pooled_auc_ordinal": pooled["auc_ordinal"],
            "pooled_auc_selective_hit": pooled["auc_selective_hit"],
            "per_antigen": {},
        }
        if n_bootstrap > 0 and len(lbl) >= 8:
            rng = np.random.default_rng(seed)
            n = len(lbl)
            b_d, b_i, b_o, b_h = [], [], [], []
            for _ in range(n_bootstrap):
                bi = rng.integers(0, n, size=n)
                mb = compute_3state_metrics(lbl[bi], bdiff[bi], pdiff[bi], pimp[bi])
                if mb["auc_dir"] is not None: b_d.append(mb["auc_dir"])
                if mb["auc_impact"] is not None: b_i.append(mb["auc_impact"])
                if mb["auc_ordinal"] is not None: b_o.append(mb["auc_ordinal"])
                if mb["auc_selective_hit"] is not None: b_h.append(mb["auc_selective_hit"])
            for k1, k2, arr in (
                ("auc_dir_ci95", "dir_ci95", b_d),
                ("auc_impact_ci95", "imp_ci95", b_i),
                ("auc_ordinal_ci95", "ord_ci95", b_o),
                ("auc_selective_hit_ci95", "hit_ci95", b_h),
            ):
                ci = [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)] if arr else [0.5, 0.5]
                out[k1] = ci
                out[k2] = ci
        return out

    sub_df = df.iloc[np.flatnonzero(same_ag)]
    multi_mod = "Assay_modality" in sub_df.columns and sub_df["Assay_modality"].nunique() > 1
    grp_indices: list[tuple[str, np.ndarray]] = []
    if multi_mod:
        for (ag, mod), g in sub_df.groupby(["Ag_group_1", "Assay_modality"], sort=True):
            if len(g) >= min_pairs:
                grp_indices.append((f"{ag}::{mod}", g.index.to_numpy()))
    else:
        for ag, g in sub_df.groupby("Ag_group_1", sort=True):
            if len(g) >= min_pairs:
                grp_indices.append((str(ag), g.index.to_numpy()))

    def _agg_from_group_metrics(g_metrics: list[dict], fb: dict) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        d_list = [m["auc_dir"] for m in g_metrics if m["n_impactful"] >= min_class_pairs and m["auc_dir"] is not None]
        acc_list = [m["accuracy"] for m in g_metrics if m["n_impactful"] >= min_class_pairs and m["accuracy"] is not None]
        i_list = [
            m["auc_impact"]
            for m in g_metrics
            if m["n_impactful"] >= min_class_pairs and m["n_neutral"] >= min_class_pairs and m["auc_impact"] is not None
        ]
        o_list = [m["auc_ordinal"] for m in g_metrics if m["n_impactful"] >= min_class_pairs and m["auc_ordinal"] is not None]
        h_list = [m["auc_selective_hit"] for m in g_metrics if m["n_impactful"] >= min_class_pairs and m["auc_selective_hit"] is not None]

        # Graceful fallback for sparse single-edit strata if needed
        if not d_list:
            d_list = [m["auc_dir"] for m in g_metrics if m["auc_dir"] is not None]
        if not acc_list:
            acc_list = [m["accuracy"] for m in g_metrics if m["accuracy"] is not None]
        if not i_list:
            i_list = [m["auc_impact"] for m in g_metrics if m["auc_impact"] is not None]
        if not o_list:
            o_list = [m["auc_ordinal"] for m in g_metrics if m["auc_ordinal"] is not None]
        if not h_list:
            h_list = [m["auc_selective_hit"] for m in g_metrics if m["auc_selective_hit"] is not None]

        r_dir = round(float(np.mean(d_list)), 4) if d_list else fb["auc_dir"]
        r_acc = round(float(np.mean(acc_list)), 4) if acc_list else fb["accuracy"]
        r_imp = round(float(np.mean(i_list)), 4) if i_list else fb["auc_impact"]
        r_ord = round(float(np.mean(o_list)), 4) if o_list else fb["auc_ordinal"]
        r_hit = round(float(np.mean(h_list)), 4) if h_list else fb["auc_selective_hit"]
        return r_acc, r_dir, r_imp, r_ord, r_hit

    per_ag: dict[str, dict] = {}
    g_list: list[dict] = []
    for gname, gidx in grp_indices:
        mg = compute_3state_metrics(lbl[gidx], bdiff[gidx], pdiff[gidx], pimp[gidx])
        per_ag[gname] = mg
        g_list.append(mg)

    r_acc, r_dir, r_imp, r_ord, r_hit = _agg_from_group_metrics(g_list, pooled)
    out = {
        "n_pairs": pooled["n_pairs"],
        "n_impactful": pooled["n_impactful"],
        "n_neutral": pooled["n_neutral"],
        "n_antigens_evaluated": len(grp_indices),
        "accuracy": r_acc,
        "auc_sym": r_dir,
        "auc_dir": r_dir,
        "auc_impact": r_imp,
        "auc_ordinal": r_ord,
        "auc_selective_hit": r_hit,
        "pooled_auc_dir": pooled["auc_dir"],
        "pooled_auc_impact": pooled["auc_impact"],
        "pooled_auc_ordinal": pooled["auc_ordinal"],
        "pooled_auc_selective_hit": pooled["auc_selective_hit"],
        "per_antigen": per_ag,
    }

    if n_bootstrap > 0 and grp_indices:
        rng = np.random.default_rng(seed)
        b_d, b_i, b_o, b_h = [], [], [], []
        for _ in range(n_bootstrap):
            bg_list = []
            for _, gidx in grp_indices:
                b_sel = gidx[rng.integers(0, len(gidx), size=len(gidx))]
                bg_list.append(compute_3state_metrics(lbl[b_sel], bdiff[b_sel], pdiff[b_sel], pimp[b_sel]))
            _, bd, bi, bo, bh = _agg_from_group_metrics(bg_list, pooled)
            if bd is not None: b_d.append(bd)
            if bi is not None: b_i.append(bi)
            if bo is not None: b_o.append(bo)
            if bh is not None: b_h.append(bh)

        for k1, k2, arr in (
            ("auc_dir_ci95", "dir_ci95", b_d),
            ("auc_impact_ci95", "imp_ci95", b_i),
            ("auc_ordinal_ci95", "ord_ci95", b_o),
            ("auc_selective_hit_ci95", "hit_ci95", b_h),
        ):
            ci = [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)] if arr else [0.5, 0.5]
            out[k1] = ci
            out[k2] = ci

    return out


def evaluate_pairwise_predictions(
    pairs_df: pd.DataFrame,
    pred_diff: np.ndarray | pd.Series,
    pred_impact: np.ndarray | pd.Series | None = None,
    *,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> dict:
    """Evaluate predicted pairwise binding differences (`pred_diff = s_hat_1 - s_hat_2`)
    and optional impact scores (`pred_impact`, defaulting to `|pred_diff|`).
    """
    df = pairs_df.reset_index(drop=True)
    preds = np.asarray(pred_diff, dtype=np.float64)
    if len(preds) != len(df):
        raise ValueError(f"Expected {len(df)} predictions, got {len(preds)}.")
    pimp = np.abs(preds) if pred_impact is None else np.asarray(pred_impact, dtype=np.float64)

    lbl = df["label"].to_numpy(dtype=int) if "label" in df.columns else np.where(df["binding_diff"].to_numpy(dtype=float) > 0, 1, -1)
    bdiff = df["binding_diff"].to_numpy(dtype=np.float64)

    overall = compute_3state_metrics(lbl, bdiff, preds, pimp)
    macro_res = compute_per_antigen_macro_metrics(
        df, preds, pimp, min_pairs=5, n_bootstrap=n_bootstrap, seed=seed
    )

    by_category = {}
    for cat, g in df.groupby("Category", sort=True):
        idx = g.index.to_numpy()
        by_category[str(cat)] = compute_per_antigen_macro_metrics(
            g, preds[idx], pimp[idx], min_pairs=5
        )

    by_group = {}
    for grp, g in df.groupby("AbRank_group", sort=True):
        idx = g.index.to_numpy()
        by_group[str(grp)] = compute_3state_metrics(lbl[idx], bdiff[idx], preds[idx], pimp[idx])

    res = {
        **macro_res,
        "macro_group_auc_sym": macro_res["auc_dir"],
        "macro_group_auc_impact": macro_res["auc_impact"],
        "macro_group_auc_ordinal": macro_res["auc_ordinal"],
        "pooled_overall": overall,
        "by_category": by_category,
        "by_group": by_group,
    }
    return res


def evaluate_pointwise_spearman(
    assays_df: pd.DataFrame,
    pred_scores: dict[str, float] | pd.Series,
    min_group_size: int = 6,
) -> dict:
    """Evaluate unfiltered full-panel continuous ranking (`Spearman rho` & `Top-Binder AUC`).

    Evaluates scalar complex predictions `f(AbAgID)` across ALL tested variants in each
    `(Ag_group, Dataset)` panel (including neutral `<2x`, moderate `2x-10x`, and `>=10x` variants).
    """
    df = assays_df[assays_df["AbAgID"].isin(pred_scores)].copy()
    df["_pred"] = df["AbAgID"].map(pred_scores)
    per_ag = {}
    rhos = []
    hit_aucs = []
    weighted_num = 0.0
    weighted_den = 0

    for (ag_grp, ds), g in df.groupby(["Ag_group", "Dataset"], sort=True):
        if len(g) < min_group_size or g["binding_score"].nunique() < 3:
            continue
        y_true = g["binding_score"].to_numpy(dtype=np.float64)
        y_pred = g["_pred"].to_numpy(dtype=np.float64)
        if np.std(y_pred) < 1e-12:
            rho = 0.0
            pval = 1.0
        else:
            rho_val, pval = spearmanr(y_true, y_pred)
            rho = float(rho_val) if np.isfinite(rho_val) else 0.0

        # Top-Binder Enrichment AUC: top quartile (and > median + 0.30 log10) vs rest of panel
        q75 = float(np.percentile(y_true, 75.0))
        med = float(np.median(y_true))
        thresh = max(q75, med + 0.30)
        y_top = (y_true >= thresh).astype(int)
        if 1 <= y_top.sum() < len(y_top):
            top_auc = float(roc_auc_score(y_top, y_pred))
            hit_aucs.append(top_auc)
        else:
            top_auc = None

        rhos.append(rho)
        weighted_num += rho * len(g)
        weighted_den += len(g)
        per_ag[f"{ag_grp}::{ds}"] = {
            "n": int(len(g)),
            "spearman_rho": round(rho, 4),
            "top_binder_auc": round(top_auc, 4) if top_auc is not None else None,
            "p_value": float(pval),
        }

    mean_rho = float(np.mean(rhos)) if rhos else float("nan")
    w_rho = float(weighted_num / weighted_den) if weighted_den > 0 else float("nan")
    mean_top_auc = float(np.mean(hit_aucs)) if hit_aucs else float("nan")
    return {
        "n_panels": len(per_ag),
        "n_variants_evaluated": int(weighted_den),
        "mean_spearman_rho": round(mean_rho, 4) if np.isfinite(mean_rho) else None,
        "weighted_spearman_rho": round(w_rho, 4) if np.isfinite(w_rho) else None,
        "mean_top_binder_auc": round(mean_top_auc, 4) if np.isfinite(mean_top_auc) else None,
        "groups": per_ag,
    }


def compute_cluster_bootstrap_summary(
    per_cluster_metrics: dict[str, dict],
    *,
    min_class_pairs: int = 2,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Aggregate independent Leave-One-Antigen-Cluster-Out (`LOACO`) cluster metrics with
    target-level cluster-resampled bootstrap 95% CIs, SD, Median, IQR, and explicit coverage.
    """
    n_total_clusters = len(per_cluster_metrics)

    def _summarize_field(vals: list[float], rng_seed: int) -> dict:
        if not vals:
            return {
                "n_clusters": 0,
                "coverage": f"0/{n_total_clusters}",
                "mean": None,
                "sd": None,
                "median": None,
                "iqr": [None, None],
                "cluster_ci95": [None, None],
            }
        arr = np.asarray(vals, dtype=np.float64)
        k = len(arr)
        rng = np.random.default_rng(rng_seed)
        boots = rng.choice(arr, size=(n_bootstrap, k), replace=True).mean(axis=1)
        return {
            "n_clusters": int(k),
            "coverage": f"{k}/{n_total_clusters}",
            "mean": round(float(np.mean(arr)), 4),
            "sd": round(float(np.std(arr, ddof=1 if k > 1 else 0)), 4),
            "median": round(float(np.median(arr)), 4),
            "iqr": [round(float(np.percentile(arr, 25.0)), 4), round(float(np.percentile(arr, 75.0)), 4)],
            "cluster_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
        }

    dir_vals = [
        float(m["auc_dir"])
        for m in per_cluster_metrics.values()
        if m.get("n_impactful", 0) >= min_class_pairs and m.get("auc_dir") is not None
    ]
    imp_vals = [
        float(m["auc_impact"])
        for m in per_cluster_metrics.values()
        if m.get("n_impactful", 0) >= min_class_pairs
        and m.get("n_neutral", 0) >= min_class_pairs
        and m.get("auc_impact") is not None
    ]
    ord_vals = [
        float(m["auc_ordinal"])
        for m in per_cluster_metrics.values()
        if m.get("n_impactful", 0) >= min_class_pairs and m.get("auc_ordinal") is not None
    ]
    hit_vals = [
        float(m["auc_selective_hit"])
        for m in per_cluster_metrics.values()
        if m.get("n_impactful", 0) >= min_class_pairs and m.get("auc_selective_hit") is not None
    ]

    s_dir = _summarize_field(dir_vals, seed)
    s_imp = _summarize_field(imp_vals, seed + 1)
    s_ord = _summarize_field(ord_vals, seed + 2)
    s_hit = _summarize_field(hit_vals, seed + 3)

    return {
        "n_eligible_clusters": int(n_total_clusters),
        "auc_dir": s_dir["mean"],
        "auc_impact": s_imp["mean"],
        "auc_ordinal": s_ord["mean"],
        "auc_selective_hit": s_hit["mean"],
        "sign": s_dir,
        "impact": s_imp,
        "ordinal": s_ord,
        "hit": s_hit,
    }


def _resolve_pair_preds(pairs_df: pd.DataFrame, pr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Align a predictions DataFrame to `pairs_df` either via `pair_id` (`pred_diff`) or
    automatically via pointwise `AbAgID` (`pred_score` -> `s_1 - s_2`)."""
    if "pair_id" in pr.columns:
        pmap = pr.drop_duplicates("pair_id").set_index("pair_id")
        col = "pred_diff" if "pred_diff" in pmap.columns else pmap.columns[-1]
        sub = pmap.reindex(pairs_df["pair_id"])
        if sub[col].isna().any():
            missing = int(sub[col].isna().sum())
            raise ValueError(f"Predictions file is missing {missing}/{len(pairs_df)} required `pair_id` rows.")
        pdiff = sub[col].to_numpy(dtype=np.float64)
        pimp = sub["pred_impact"].to_numpy(dtype=np.float64) if "pred_impact" in pmap.columns else None
        return pdiff, pimp

    if "AbAgID" in pr.columns:
        smap = pr.drop_duplicates("AbAgID").set_index("AbAgID")
        scol = "pred_score" if "pred_score" in smap.columns else smap.columns[-1]
        s1 = pairs_df["AbAgID_1"].map(smap[scol])
        s2 = pairs_df["AbAgID_2"].map(smap[scol])
        if s1.isna().any() or s2.isna().any():
            raise ValueError("Pointwise `AbAgID` predictions do not cover all pairs in target split.")
        pdiff = (s1 - s2).to_numpy(dtype=np.float64)
        return pdiff, np.abs(pdiff)

    if len(pr) == len(pairs_df):
        col = "pred_diff" if "pred_diff" in pr.columns else pr.columns[-1]
        pimp = pr["pred_impact"].to_numpy(dtype=np.float64) if "pred_impact" in pr.columns else None
        return pr[col].to_numpy(dtype=np.float64), pimp

    raise ValueError("Predictions file must contain `pair_id`, pointwise `AbAgID`, or match split row count.")


def evaluate_flagship_suite(
    suite: str,
    preds_df: pd.DataFrame,
    *,
    repo_root: Path | None = None,
    data_dir: Path | str | None = None,
    kaggle_dataset: str | None = None,
    track: str = "prospective",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Evaluate predictions on one of the four canonical AbRank community flagship suites:
      - `maturation`: 22-Cluster `LOACO` Local Mutation 3-State Ranking (`12,267` L1-L3 pairs;
                      Track `prospective` [Seq/Predicted-3D] or `wt-3d` [Single WT Co-Crystal])
      - `discovery` : 25 Unfiltered `LOACO` Discovery Panels (`Top-Binder AUC` & `Precision@Top-20%`)
                      + optional `VHH` Nanobody Kd Ranking (`11` evaluable targets)
      - `transfer`  : Cross-Assay & Functional Viral Adaptation (`IC50`, `RBD Escape`, `AbAgym DMS`, `ELISA`, `DDG`)
      - `generative`: Post-Generator Best-of-N Reranking, Conditional Log-Likelihood, & 3-State DPO Verifier
                      Benchmark across `1,354` wet-lab generative AI designs (`IgDesign`, `Origin1`,
                      `BoltzGen_VHH`, `Germinal`, `AIntibody`, `Shanker`, `Adaptyv_EGFR_VHH`)
    """
    from abrank.kaggle_io import resolve_benchmark_dir

    bdir = resolve_benchmark_dir(
        data_dir=data_dir,
        kaggle_dataset=kaggle_dataset,
        repo_root=repo_root,
        auto_download=kaggle_dataset is not None,
    )

    if suite == "maturation":
        kd = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", usecols=["AbAgID", "Ag_cluster"], low_memory=False).drop_duplicates("AbAgID").set_index("AbAgID")
        parts = [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")]
        loc_df = pd.concat(parts, ignore_index=True)
        loc_df = loc_df[loc_df["Category"].isin(["L1", "L2", "L3"])].reset_index(drop=True)
        pdiff, pimp = _resolve_pair_preds(loc_df, preds_df)
        pimp_use = np.abs(pdiff) if pimp is None else pimp

        c1 = loc_df["AbAgID_1"].map(kd["Ag_cluster"]).astype(str).to_numpy()
        clusters = sorted(c for c, cnt in pd.Series(c1).value_counts().items() if cnt >= 5)
        lbls = loc_df["label"].to_numpy(dtype=int)
        bds = loc_df["binding_diff"].to_numpy(dtype=float)
        per_cl = {cl: compute_3state_metrics(lbls[c1 == cl], bds[c1 == cl], pdiff[c1 == cl], pimp_use[c1 == cl]) for cl in clusters}
        cb = compute_cluster_bootstrap_summary(per_cl, min_class_pairs=2, n_bootstrap=n_bootstrap, seed=seed)
        return {
            "suite": "AbRank-Maturation",
            "track": track,
            "n_pairs": int(len(loc_df)),
            "headline_metric": "loaco_ordinal_auc",
            "loaco_ordinal_auc": cb["auc_ordinal"],
            "loaco_sign_auc": cb["auc_dir"],
            "loaco_impact_auc": cb["auc_impact"],
            "cluster_bootstrap": cb,
        }

    if suite == "discovery":
        if "AbAgID" not in preds_df.columns:
            raise ValueError("`--suite discovery` expects pointwise predictions with columns `AbAgID` and `pred_score`.")
        smap = preds_df.drop_duplicates("AbAgID").set_index("AbAgID")
        scol = "pred_score" if "pred_score" in smap.columns else smap.columns[-1]
        score_dict = smap[scol].astype(float).to_dict()

        pw_all = pd.concat([pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}_pointwise.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")], ignore_index=True)
        panels = [g for (_, _), g in pw_all.groupby(["Ag_group", "Dataset"], sort=True) if 6 <= len(g) <= 200 and g["binding_score"].nunique() >= 3]
        pw_df = pd.concat(panels, ignore_index=True)
        sp_res = evaluate_pointwise_spearman(pw_df, score_dict, min_group_size=6)

        prec20, neudel20 = [], []
        for (_, _), g in pw_df[pw_df["AbAgID"].isin(score_dict)].groupby(["Ag_group", "Dataset"], sort=True):
            if len(g) < 6 or g["binding_score"].nunique() < 3:
                continue
            yt = g["binding_score"].to_numpy(dtype=float)
            yp = g["AbAgID"].map(score_dict).to_numpy(dtype=float)
            k20 = max(1, int(np.ceil(0.20 * len(yt))))
            top_ix = np.argsort(-yp)[:k20]
            med = float(np.median(yt))
            thr = max(float(np.percentile(yt, 75.0)), med + 0.30)
            prec20.append(float((yt[top_ix] >= thr).mean()))
            neudel20.append(float((yt[top_ix] <= med + 0.30).mean()))

        out_disc: dict = {
            "suite": "AbRank-Discovery",
            "n_panels": sp_res["n_panels"],
            "n_variants_evaluated": sp_res["n_variants_evaluated"],
            "headline_metrics": ["mean_top_binder_auc", "precision_at_top20_pct"],
            "mean_top_binder_auc": sp_res["mean_top_binder_auc"],
            "precision_at_top20_pct": round(float(np.mean(prec20)), 4) if prec20 else None,
            "neu_del_at_top20_pct": round(float(np.mean(neudel20)), 4) if neudel20 else None,
            "mean_spearman_rho": sp_res["mean_spearman_rho"],
        }
        vhh_sp = pd.read_csv(bdir / "splits" / "vhh" / "vhh_ranking_test.tsv", sep="\t", low_memory=False)
        if set(vhh_sp["AbAgID_1"]).union(vhh_sp["AbAgID_2"]) <= set(score_dict.keys()):
            vd, vi = _resolve_pair_preds(vhh_sp, preds_df)
            vm = compute_per_antigen_macro_metrics(vhh_sp, vd, vi, min_pairs=5, n_bootstrap=n_bootstrap, seed=seed)
            out_disc["vhh_nanobody_ordinal_auc"] = vm["auc_ordinal"]
            out_disc["vhh_nanobody_sign_auc"] = vm["auc_dir"]
        return out_disc

    if suite == "transfer":
        split_specs = [
            ("viral_ic50", bdir / "splits" / "transfer" / "ic50_test.tsv"),
            ("rbd_escape", bdir / "splits" / "transfer" / "escape_test.tsv"),
            ("cross_modal", bdir / "splits" / "cross_modal" / "modality_test.tsv"),
        ]
        tr_out: dict = {"suite": "AbRank-Transfer", "tracks": {}}
        for tname, sp_path in split_specs:
            gt_t = pd.read_csv(sp_path, sep="\t", low_memory=False)
            if "pair_id" in preds_df.columns:
                sub_gt = gt_t[gt_t["pair_id"].isin(set(preds_df["pair_id"]))].reset_index(drop=True)
            elif "AbAgID" in preds_df.columns:
                avail = set(preds_df["AbAgID"])
                sub_gt = gt_t[gt_t["AbAgID_1"].isin(avail) & gt_t["AbAgID_2"].isin(avail)].reset_index(drop=True)
            else:
                continue
            if len(sub_gt) == 0:
                continue
            pd_t, pi_t = _resolve_pair_preds(sub_gt, preds_df)
            tm = compute_per_antigen_macro_metrics(sub_gt, pd_t, pi_t, min_pairs=5, n_bootstrap=n_bootstrap, seed=seed)
            tr_out["tracks"][tname] = {
                "n_pairs": tm["n_pairs"],
                "n_antigens_evaluated": tm["n_antigens_evaluated"],
                "auc_ordinal": tm["auc_ordinal"],
                "auc_dir": tm["auc_dir"],
                "auc_impact": tm["auc_impact"],
            }
        return tr_out

    if suite == "generative":
        gen_dir = bdir / "splits" / "generative"
        pairs_path = gen_dir / "generative_pairs.tsv"
        panels_path = gen_dir / "generative_panels.tsv"

        if panels_path.exists():
            gen_pw = pd.read_csv(panels_path, sep="\t", low_memory=False)
        else:
            kd_g = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False)
            vhh_g = pd.read_csv(bdir / "assays_vhh.tsv", sep="\t", low_memory=False)
            kd_g = kd_g[kd_g["Dataset"].isin(["IgDesign", "Origin1", "AIntibody", "Shanker", "Germinal"])]
            vhh_g = vhh_g[vhh_g["Dataset"].isin(["BoltzGen_VHH", "Germinal_VHH", "Adaptyv_EGFR_VHH"])]
            gen_pw = pd.concat([kd_g, vhh_g], ignore_index=True)

        if pairs_path.exists():
            gen_pairs = pd.read_csv(pairs_path, sep="\t", low_memory=False)
        else:
            from abrank.splits import build_within_antigen_pairs

            gen_loc, gen_cand = build_within_antigen_pairs(
                gen_pw, m=1.0, m_neutral=0.30, include_neutral=True, neutral_ratio=1.0,
                modality="KD_Generative", ab_cap=4, per_ag_cap=600, include_candidates=True, seed=42, prefix="GEN",
            )
            gen_pairs = pd.concat([gen_loc, gen_cand], ignore_index=True).reset_index(drop=True)

        gen_out: dict = {
            "suite": "AbRank-Generative",
            "n_total_generative_designs": int(len(gen_pw)),
            "headline_metrics": ["generative_ordinal_auc", "mean_top_binder_auc", "precision_at_top20_pct", "mean_top20_regret_pkd"],
        }

        # 1. 3-State DPO / Verifier Pairwise Ranking on wet-lab generative cohorts
        sub_gp = gen_pairs
        if "pair_id" in preds_df.columns:
            sub_gp = gen_pairs[gen_pairs["pair_id"].isin(set(preds_df["pair_id"]))].reset_index(drop=True)
        elif "AbAgID" in preds_df.columns:
            avail_ids = set(preds_df["AbAgID"])
            sub_gp = gen_pairs[gen_pairs["AbAgID_1"].isin(avail_ids) & gen_pairs["AbAgID_2"].isin(avail_ids)].reset_index(drop=True)

        if len(sub_gp) > 0:
            pd_g, pi_g = _resolve_pair_preds(sub_gp, preds_df)
            gm = compute_per_antigen_macro_metrics(sub_gp, pd_g, pi_g, min_pairs=3, n_bootstrap=n_bootstrap, seed=seed)
            gen_out["n_pairs_evaluated"] = gm["n_pairs"]
            gen_out["n_antigens_evaluated"] = gm["n_antigens_evaluated"]
            gen_out["generative_ordinal_auc"] = gm["auc_ordinal"]
            gen_out["generative_sign_auc"] = gm["auc_dir"]
            gen_out["generative_impact_auc"] = gm["auc_impact"]
            if "ord_ci95" in gm:
                gen_out["generative_ordinal_ci95"] = gm["ord_ci95"]

        # 2. Post-Generator Best-of-N Funnel Metrics across wet-lab generative panels (`n >= 5`)
        if "AbAgID" in preds_df.columns:
            smap = preds_df.drop_duplicates("AbAgID").set_index("AbAgID")
            scol = "pred_score" if "pred_score" in smap.columns else smap.columns[-1]
            score_dict = smap[scol].astype(float).to_dict()
            sp_res = evaluate_pointwise_spearman(gen_pw, score_dict, min_group_size=5)

            prec20, base20, neudel20, regret20 = [], [], [], []
            for (_, _), g in gen_pw[gen_pw["AbAgID"].isin(score_dict)].groupby(["Ag_group", "Dataset"], sort=True):
                if len(g) < 5 or g["binding_score"].nunique() < 2:
                    continue
                yt = g["binding_score"].to_numpy(dtype=float)
                yp = g["AbAgID"].map(score_dict).to_numpy(dtype=float)
                k20 = max(1, int(np.ceil(0.20 * len(yt))))
                top_ix = np.argsort(-yp)[:k20]
                med = float(np.median(yt))
                thr = max(float(np.percentile(yt, 75.0)), med + 0.30)
                prec20.append(float((yt[top_ix] >= thr).mean()))
                base20.append(float((yt >= thr).mean()))
                neudel20.append(float((yt[top_ix] <= med + 0.30).mean()))
                regret20.append(float(np.max(yt) - np.max(yt[top_ix])))

            mean_p20 = float(np.mean(prec20)) if prec20 else None
            mean_b20 = float(np.mean(base20)) if base20 else None
            gen_out["n_panels_evaluated"] = int(len(prec20))
            gen_out["mean_top_binder_auc"] = sp_res["mean_top_binder_auc"]
            gen_out["precision_at_top20_pct"] = round(mean_p20, 4) if mean_p20 is not None else None
            gen_out["enrichment_factor_top20"] = round(mean_p20 / max(mean_b20, 1e-8), 4) if (mean_p20 is not None and mean_b20) else None
            gen_out["neu_del_at_top20_pct"] = round(float(np.mean(neudel20)), 4) if neudel20 else None
            gen_out["mean_top20_regret_pkd"] = round(float(np.mean(regret20)), 4) if regret20 else None
            gen_out["mean_spearman_rho"] = sp_res["mean_spearman_rho"]

        return gen_out

    raise ValueError(f"Unknown flagship suite: {suite}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions on AbRank v2 flagship suites or custom split TSVs.")
    parser.add_argument("--suite", choices=["maturation", "discovery", "transfer", "generative"], default=None, help="Canonical AbRank flagship task suite")
    parser.add_argument("--track", choices=["prospective", "wt-3d"], default="prospective", help="Structural regime tag for `--suite maturation`")
    parser.add_argument("--split", type=Path, default=None, help="Ground-truth pairwise split TSV (when not using `--suite`)")
    parser.add_argument("--preds", type=Path, required=True, help="Predictions TSV/CSV (`pair_id` + `pred_diff` or pointwise `AbAgID` + `pred_score`)")
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to AbRank benchmark data directory (or set ABRANK_DATA_DIR)")
    parser.add_argument("--kaggle-dataset", type=str, default=None, help="Optional Kaggle dataset slug (e.g. `aurlienplissier/abrank`) to auto-fetch if local data is absent")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap replicates for 95%% CI")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    sep = "\t" if str(args.preds).endswith(".tsv") else ","
    pr = pd.read_csv(args.preds, sep=sep)

    if args.suite is not None:
        metrics = evaluate_flagship_suite(
            args.suite,
            pr,
            data_dir=args.data_dir,
            kaggle_dataset=args.kaggle_dataset,
            track=args.track,
            n_bootstrap=args.bootstrap,
        )
    else:
        if args.split is None:
            parser.error("Provide either `--suite {maturation,discovery,transfer,generative}` or `--split <path>`.")
        gt = pd.read_csv(args.split, sep="\t")
        pdiff, pimp = _resolve_pair_preds(gt, pr)
        metrics = evaluate_pairwise_predictions(gt, pdiff, pred_impact=pimp, n_bootstrap=args.bootstrap)

    print(json.dumps(metrics, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
