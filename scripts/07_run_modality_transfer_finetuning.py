#!/usr/bin/env python3
"""Step 7: Cross-modality 3-state zero-shot transfer vs. supervised fine-tuning
from core `KD` (`SPR`/`BLI`/`KinExA`) to:
  1. `ELISA` (`AbDesign` pure ELISA ratio)
  2. `IC50` Neutralization (`CATNAP` + `Shanker` + `AbCoV`)
  3. `ΔΔG (SKEMPIv2)` — Structural Interface Thermodynamic `ΔΔG`
  4. `ΔΔG (AbDesign)` — Structural Mutational `ΔΔG`
  5. `ΔΔG / DMS (AbAgym)` — Interface Deep Mutational Scanning
  6. `Escape DMS (Bloom)` — Viral SARS-CoV-2 RBD Escape DMS

Strictly enforces zero data leakage across 5-fold `GroupKFold`:
  - Holds out entire antigen groups (`ELISA`, `SKEMPIv2 ΔΔG`, `AbDesign ΔΔG`, `AbAgym DMS`)
    or entire 80%-Fv antibody clonal lineages (`IC50`, `Bloom Escape`).
  - Refits the `KD` source ranker per fold strictly excluding held-out homologous `<50%`
    antigen clusters, held-out `<80%` antibody lineages, and `SKEMPIv2` (when evaluating `SKEMPIv2`).
  - Evaluates all 3-state metrics: `Directional Sign AUC` (`±1`), `Impact Detection AUC` (`±1` vs `0`),
    `3-Class Ordinal AUC` (`-1 < 0 < +1`), and `Selective Hit AUC` (`+1` vs `{0, -1}`).

Outputs:
  - `results/modality_transfer_and_finetuning.json`
  - `results/figures/modality_transfer_and_finetuning.png`
  - `results/MODALITY_TRANSFER_AND_FINETUNING.md`
"""

from __future__ import annotations

import os
for _env_k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_k, "8")

import argparse
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
torch.set_num_threads(8)

from abrank.config import get_repo_root
from abrank.evaluate import compute_3state_metrics, compute_per_antigen_macro_metrics
from abrank.pairs import build_within_antigen_pairs
from abrank.splits import assign_antibody_lineages
from abrank.baselines.models import compute_stratum_weights
from abrank.baselines.transfer_finetune import (
    KDResidualAdapterRanker,
    KDSourceRanker,
    L2SPFineTunedRanker,
    TransferPairTensors,
    assemble_transfer_tensors,
    build_transfer_odd_matrix,
    fit_kd_source_ranker,
)


METHODS = [
    "Random Baseline",
    "Zero-Shot PLM (ESMC+AbLang2)",
    "Direct KD Transfer (0% Target Data)",
    "Target From Scratch (3-State Hurdle)",
    "KD -> Target Fine-Tuned (L2-SP Hurdle)",
    "KD -> Target Fine-Tuned (Residual Adapter)",
]


def bootstrap_3state(
    labels: np.ndarray,
    bdiff: np.ndarray,
    pred_diff: np.ndarray,
    pred_impact: np.ndarray,
    n_boot: int = 600,
    seed: int = 42,
) -> dict:
    base = compute_3state_metrics(labels, bdiff, pred_diff, pred_impact)
    if len(labels) < 8 or n_boot <= 0:
        return base
    rng = np.random.default_rng(seed)
    n = len(labels)
    b_dir, b_imp, b_ord, b_hit = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_3state_metrics(labels[idx], bdiff[idx], pred_diff[idx], pred_impact[idx])
        if m["auc_dir"] is not None:
            b_dir.append(m["auc_dir"])
        if m["auc_impact"] is not None:
            b_imp.append(m["auc_impact"])
        if m["auc_ordinal"] is not None:
            b_ord.append(m["auc_ordinal"])
        if m["auc_selective_hit"] is not None:
            b_hit.append(m["auc_selective_hit"])
    out = dict(base)
    for k, arr in (("dir_ci95", b_dir), ("imp_ci95", b_imp), ("ord_ci95", b_ord), ("hit_ci95", b_hit)):
        out[k] = [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)] if arr else [0.5, 0.5]
    return out


def build_leak_free_kd_source(
    kd_pool_df: pd.DataFrame,
    kd_pool_tp: TransferPairTensors,
    X_kd_pool: np.ndarray,
    kd_lookup: pd.DataFrame,
    *,
    exclude_ag_groups: set[str],
    exclude_ab_lineages: set[str] | None = None,
    exclude_skempi: bool = False,
) -> KDSourceRanker:
    """Fit a 3-state `KD` source ranker strictly disjoint from the held-out target test fold."""
    m = (~kd_pool_df["Ag_group_1"].isin(exclude_ag_groups)) & (~kd_pool_df["Ag_group_2"].isin(exclude_ag_groups))
    if exclude_skempi:
        m &= (kd_pool_df["Dataset_1"] != "SKEMPIv2") & (kd_pool_df["Dataset_2"] != "SKEMPIv2")
    if exclude_ab_lineages:
        l1 = kd_lookup.loc[kd_pool_df["AbAgID_1"], "Ab_lineage"].to_numpy()
        l2 = kd_lookup.loc[kd_pool_df["AbAgID_2"], "Ab_lineage"].to_numpy()
        m &= (~np.isin(l1, list(exclude_ab_lineages))) & (~np.isin(l2, list(exclude_ab_lineages)))

    keep_idx = np.flatnonzero(m.to_numpy(dtype=bool))
    sub_df = kd_pool_df.iloc[keep_idx].reset_index(drop=True)
    sub_tp = kd_pool_tp.subset(keep_idx)
    w_all = compute_stratum_weights(sub_df, sub_tp.is_ag_mut)

    # Carve out leak-free inner validation antigens for C selection
    loc_ags = sorted(sub_df.loc[sub_df["Category"] != "Global", "Ag_group_1"].unique())
    val_ags = set(loc_ags[1::5]) if len(loc_ags) >= 5 else set(loc_ags[:1])
    is_val = (sub_df["Ag_group_1"].isin(val_ags) | sub_df["Ag_group_2"].isin(val_ags)).to_numpy()
    if (is_val & sub_tp.is_impactful).sum() < 25 or ((~is_val) & sub_tp.is_impactful).sum() < 100:
        rng = np.random.default_rng(42)
        is_val = rng.random(len(sub_df)) < 0.15

    tr_i = np.flatnonzero(~is_val)
    va_i = np.flatnonzero(is_val)
    return fit_kd_source_ranker(
        X_kd_pool[keep_idx[tr_i]],
        sub_tp.y[tr_i],
        w_all[tr_i],
        X_kd_pool[keep_idx[va_i]],
        sub_tp.y[va_i],
        tp_tr=sub_tp.subset(tr_i),
        tp_val=sub_tp.subset(va_i),
    )


def evaluate_target_track(
    track_key: str,
    track_title: str,
    df: pd.DataFrame,
    tp: TransferPairTensors,
    groups: np.ndarray,
    group_mode: str,
    *,
    exclude_skempi_from_kd: bool,
    kd_pool_df: pd.DataFrame,
    kd_pool_tp: TransferPairTensors,
    X_kd_pool: np.ndarray,
    kd_lookup: pd.DataFrame,
    ag_to_homolog_groups: dict[str, set[str]],
    kd_cache: dict[tuple, KDSourceRanker],
    n_folds: int = 5,
    few_shot_fracs: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
) -> dict:
    """Run leak-free 5-fold GroupKFold 3-state transfer + fine-tuning + few-shot curves."""
    n = len(df)
    X_all = build_transfer_odd_matrix(tp, include_biophys=True)
    gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))

    oof_diff: dict[str, np.ndarray] = {m: np.zeros(n, dtype=np.float64) for m in METHODS}
    oof_imp: dict[str, np.ndarray] = {m: np.zeros(n, dtype=np.float64) for m in METHODS}
    oof_diff["Zero-Shot PLM (ESMC+AbLang2)"] = tp.zs_dict["esmc_ablang2_zeroshot"].astype(np.float64)
    oof_imp["Zero-Shot PLM (ESMC+AbLang2)"] = tp.zs_impact_dict["esmc_ablang2_zeroshot"].astype(np.float64)

    kd_lins_present = set(kd_lookup.loc[kd_pool_df["AbAgID_1"], "Ab_lineage"]).union(
        set(kd_lookup.loc[kd_pool_df["AbAgID_2"], "Ab_lineage"])
    )
    kd_ags_present = set(kd_pool_df["Ag_group_1"]).union(kd_pool_df["Ag_group_2"])

    def _get_cached_kd(excl_ags: set[str], excl_lins: set[str] | None) -> KDSourceRanker:
        eff_lins = frozenset((excl_lins or set()) & kd_lins_present)
        eff_ags = frozenset(excl_ags & kd_ags_present)
        key = (eff_ags, eff_lins, bool(exclude_skempi_from_kd))
        if key not in kd_cache:
            kd_cache[key] = build_leak_free_kd_source(
                kd_pool_df, kd_pool_tp, X_kd_pool, kd_lookup,
                exclude_ag_groups=set(eff_ags),
                exclude_ab_lineages=set(eff_lins) if eff_lins else None,
                exclude_skempi=exclude_skempi_from_kd,
            )
        return kd_cache[key]

    fs_l2sp_d = {fr: np.zeros(n, dtype=np.float64) for fr in few_shot_fracs}
    fs_l2sp_i = {fr: np.zeros(n, dtype=np.float64) for fr in few_shot_fracs}
    fs_scr_d = {fr: np.zeros(n, dtype=np.float64) for fr in few_shot_fracs}
    fs_scr_i = {fr: np.zeros(n, dtype=np.float64) for fr in few_shot_fracs}
    fs_adp_d = {fr: np.zeros(n, dtype=np.float64) for fr in few_shot_fracs}
    fs_adp_i = {fr: np.zeros(n, dtype=np.float64) for fr in few_shot_fracs}
    fold_params: list[dict] = []

    actual_folds = min(n_folds, len(np.unique(groups)))
    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X_all, tp.y, groups)):
        t_f0 = time.time()
        te_df = df.iloc[te_idx]
        if group_mode == "antigen":
            held_ags: set[str] = set()
            for g in set(te_df["Ag_group_1"]).union(te_df["Ag_group_2"]):
                held_ags.update(ag_to_homolog_groups.get(str(g), {str(g)}))
            kd_src = _get_cached_kd(held_ags, None)
        else:
            held_lins = set(groups[te_idx].tolist())
            kd_src = _get_cached_kd(set(), held_lins)

        tp_tr = tp.subset(tr_idx)
        tp_te = tp.subset(te_idx)

        # 1. Direct Zero-Shot KD Transfer
        d_kd, i_kd = kd_src.predict_3state(tp_te, X_all[te_idx])
        oof_diff["Direct KD Transfer (0% Target Data)"][te_idx] = d_kd
        oof_imp["Direct KD Transfer (0% Target Data)"][te_idx] = i_kd

        # 2. Target From Scratch (3-State Hurdle)
        m_scr = L2SPFineTunedRanker(kd_source=None).fit(X_all[tr_idx], tp_tr.y, groups[tr_idx], tp_tr=tp_tr)
        d_sc, i_sc = m_scr.predict_3state(tp_te, X_all[te_idx])
        oof_diff["Target From Scratch (3-State Hurdle)"][te_idx] = d_sc
        oof_imp["Target From Scratch (3-State Hurdle)"][te_idx] = i_sc

        # 3. KD -> Target Fine-Tuned (L2-SP 3-State Hurdle)
        m_ft = L2SPFineTunedRanker(kd_source=kd_src).fit(X_all[tr_idx], tp_tr.y, groups[tr_idx], tp_tr=tp_tr)
        d_ft, i_ft = m_ft.predict_3state(tp_te, X_all[te_idx])
        oof_diff["KD -> Target Fine-Tuned (L2-SP Hurdle)"][te_idx] = d_ft
        oof_imp["KD -> Target Fine-Tuned (L2-SP Hurdle)"][te_idx] = i_ft
        fold_params.append(m_ft.best_params)

        # 4. KD -> Target Fine-Tuned (Residual Adapter)
        m_adp = KDResidualAdapterRanker(kd_source=kd_src, seed=42 + fold_i).fit(tp_tr, X_all[tr_idx], groups[tr_idx])
        d_ad, i_ad = m_adp.predict_3state(tp_te, X_all[te_idx])
        oof_diff["KD -> Target Fine-Tuned (Residual Adapter)"][te_idx] = d_ad
        oof_imp["KD -> Target Fine-Tuned (Residual Adapter)"][te_idx] = i_ad

        # 5. Few-Shot Learning Curves on the exact same outer test fold
        rng = np.random.default_rng(1000 + fold_i)
        for fr in few_shot_fracs:
            if fr >= 0.999:
                fs_scr_d[fr][te_idx], fs_scr_i[fr][te_idx] = d_sc, i_sc
                fs_l2sp_d[fr][te_idx], fs_l2sp_i[fr][te_idx] = d_ft, i_ft
                fs_adp_d[fr][te_idx], fs_adp_i[fr][te_idx] = d_ad, i_ad
                continue

            n_budget = max(32, int(round(fr * len(tr_idx))))
            perm = rng.permutation(len(tr_idx))
            sub_tr = tr_idx[perm[:n_budget]]
            tp_sub = tp.subset(sub_tr)

            scr_f = L2SPFineTunedRanker(kd_source=None).fit(X_all[sub_tr], tp_sub.y, groups[sub_tr], tp_tr=tp_sub)
            ft_f = L2SPFineTunedRanker(kd_source=kd_src).fit(X_all[sub_tr], tp_sub.y, groups[sub_tr], tp_tr=tp_sub)
            adp_f = KDResidualAdapterRanker(kd_source=kd_src, epochs=14, n_inner_models=1, seed=200 + fold_i).fit(
                tp_sub, X_all[sub_tr], groups[sub_tr]
            )
            fs_scr_d[fr][te_idx], fs_scr_i[fr][te_idx] = scr_f.predict_3state(tp_te, X_all[te_idx])
            fs_l2sp_d[fr][te_idx], fs_l2sp_i[fr][te_idx] = ft_f.predict_3state(tp_te, X_all[te_idx])
            fs_adp_d[fr][te_idx], fs_adp_i[fr][te_idx] = adp_f.predict_3state(tp_te, X_all[te_idx])

        print(f"    [Fold {fold_i + 1}/{actual_folds}] done in {time.time() - t_f0:.1f}s (L2-SP={m_ft.best_params})", flush=True)

    method_summaries = {}
    ab_only = tp.is_ab_mut & (~tp.is_ag_mut)
    ag_only = tp.is_ag_mut & (~tp.is_ab_mut)

    for m_name in METHODS:
        pd_v = oof_diff[m_name]
        pi_v = oof_imp[m_name]
        boot = compute_per_antigen_macro_metrics(df, pd_v, pi_v, min_pairs=5, n_bootstrap=500)

        ab_m = (
            compute_per_antigen_macro_metrics(df.iloc[np.flatnonzero(ab_only)], pd_v[ab_only], pi_v[ab_only], min_pairs=5)
            if ab_only.sum() >= 6
            else {}
        )
        ag_m = (
            compute_per_antigen_macro_metrics(df.iloc[np.flatnonzero(ag_only)], pd_v[ag_only], pi_v[ag_only], min_pairs=5)
            if ag_only.sum() >= 6
            else {}
        )

        method_summaries[m_name] = {
            **boot,
            "ab_mut_dir": ab_m.get("auc_dir"),
            "ab_mut_imp": ab_m.get("auc_impact"),
            "ab_mut_ord": ab_m.get("auc_ordinal"),
            "ag_mut_dir": ag_m.get("auc_dir"),
            "ag_mut_imp": ag_m.get("auc_impact"),
            "ag_mut_ord": ag_m.get("auc_ordinal"),
        }

    dkd_s = method_summaries["Direct KD Transfer (0% Target Data)"]
    few_shot_summary = {
        "0.00": {
            "n_train_avg": 0,
            "direct_kd_ord": dkd_s["auc_ordinal"],
            "direct_kd_dir": dkd_s["auc_dir"],
            "scratch_ord": 0.5000,
            "kd_l2sp_ord": dkd_s["auc_ordinal"],
            "kd_adapter_ord": dkd_s["auc_ordinal"],
            "scratch_dir": 0.5000,
            "kd_l2sp_dir": dkd_s["auc_dir"],
            "kd_adapter_dir": dkd_s["auc_dir"],
        }
    }
    avg_fold_train = int(round(n * (actual_folds - 1) / actual_folds))
    for fr in few_shot_fracs:
        m_sc = compute_per_antigen_macro_metrics(df, fs_scr_d[fr], fs_scr_i[fr], min_pairs=5)
        m_l2 = compute_per_antigen_macro_metrics(df, fs_l2sp_d[fr], fs_l2sp_i[fr], min_pairs=5)
        m_ad = compute_per_antigen_macro_metrics(df, fs_adp_d[fr], fs_adp_i[fr], min_pairs=5)
        few_shot_summary[f"{fr:.2f}"] = {
            "n_train_avg": max(32, int(round(fr * avg_fold_train))) if fr < 0.999 else avg_fold_train,
            "scratch_ord": m_sc["auc_ordinal"],
            "kd_l2sp_ord": m_l2["auc_ordinal"],
            "kd_adapter_ord": m_ad["auc_ordinal"],
            "scratch_dir": m_sc["auc_dir"],
            "kd_l2sp_dir": m_l2["auc_dir"],
            "kd_adapter_dir": m_ad["auc_dir"],
        }

    return {
        "track_key": track_key,
        "track_title": track_title,
        "n_pairs": int(n),
        "n_impactful": int(tp.is_impactful.sum()),
        "n_neutral": int((~tp.is_impactful).sum()),
        "n_groups": int(len(np.unique(groups))),
        "group_mode": group_mode,
        "n_ab_mut": int(ab_only.sum()),
        "n_ag_mut": int(ag_only.sum()),
        "methods": method_summaries,
        "few_shot_curve": few_shot_summary,
        "l2sp_fold_params": fold_params,
    }


def make_transfer_figure(track_results: list[dict], out_path: Path) -> None:
    """Render publication-quality 3-row figure:
      Row 1: 3-Class Ordinal AUC (`-1 < 0 < +1`) across all 6 target assays.
      Row 2: Impact Detection AUC (`±1` vs `0`) vs. Directional Sign AUC (`+1` vs `-1`).
      Row 3: Few-Shot Fine-Tuning learning curves (`0%` -> `100%` target pairs).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_tr = len(track_results)
    fig = plt.figure(figsize=(19.5, 12.2))
    gs = fig.add_gridspec(3, n_tr, height_ratios=[1.15, 1.05, 0.95], hspace=0.40, wspace=0.26)

    bar_methods = [
        ("Zero-Shot PLM (ESMC+AbLang2)", "Zero-Shot PLM", "#3498db"),
        ("Direct KD Transfer (0% Target Data)", "Direct KD (0% Target)", "#e67e22"),
        ("Target From Scratch (3-State Hurdle)", "Target Scratch (100%)", "#7f8c8d"),
        ("KD -> Target Fine-Tuned (L2-SP Hurdle)", "KD -> L2-SP Hurdle", "#2ecc71"),
        ("KD -> Target Fine-Tuned (Residual Adapter)", "KD -> Residual Adapter", "#1b7943"),
    ]

    # Row 1: 3-Class Ordinal AUC
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(n_tr)
    width = 0.155
    ax1.axhline(0.50, color="#c0392b", linestyle="--", linewidth=1.25, zorder=1)
    for m_i, (m_key, m_lbl, col) in enumerate(bar_methods):
        off = (m_i - 2.0) * width
        vals = [tr["methods"][m_key]["auc_ordinal"] for tr in track_results]
        cis = [tr["methods"][m_key]["ord_ci95"] for tr in track_results]
        elo = [max(0.0, v - ci[0]) for v, ci in zip(vals, cis)]
        ehi = [max(0.0, ci[1] - v) for v, ci in zip(vals, cis)]
        bars = ax1.bar(x + off, vals, width=width * 0.92, yerr=[elo, ehi], capsize=2.2, color=col, alpha=0.90, edgecolor="#1f2d3d", linewidth=0.6, label=m_lbl, zorder=3)
        for bx, v, hi in zip(bars, vals, ehi):
            ax1.text(bx.get_x() + bx.get_width() / 2.0, v + hi + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#1a252f")

    x_labels = [f"{tr['track_title']}\n(n={tr['n_pairs']:,}: {tr['n_impactful']} ±1, {tr['n_neutral']} 0)" for tr in track_results]
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=9.3, fontweight="bold")
    ax1.set_ylim(0.38, 0.96)
    ax1.set_ylabel("3-Class Ordinal AUC\n(-1 < 0 < +1)", fontsize=10.0, fontweight="bold")
    ax1.set_title("A. Joint 3-State Ranking Concordance (Depleted -1 < Neutral 0 < Improved +1, 5-Fold Zero-Leakage GroupCV)", fontsize=11.5, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.55)
    ax1.legend(loc="upper left", ncol=5, frameon=True, fontsize=8.8)

    # Row 2: Left half = Directional Sign AUC, Right half = Impact Detection AUC
    mid = n_tr // 2
    ax2_l = fig.add_subplot(gs[1, :mid])
    ax2_r = fig.add_subplot(gs[1, mid:])
    for ax_sub, m_field, ci_field, sub_title, ylab in (
        (ax2_l, "auc_dir", "dir_ci95", "B. Directional Sign AUC (+1 vs -1 on Impactful Pairs)", "Directional Sign AUC"),
        (ax2_r, "auc_impact", "imp_ci95", "C. Mutation Impact / Hot-Spot AUC (±1 vs 0 Neutral)", "Impact Detection AUC"),
    ):
        ax_sub.axhline(0.50, color="#c0392b", linestyle="--", linewidth=1.2, zorder=1)
        for m_i, (m_key, m_lbl, col) in enumerate(bar_methods):
            off = (m_i - 2.0) * width
            vals = [tr["methods"][m_key][m_field] for tr in track_results]
            ax_sub.bar(x + off, vals, width=width * 0.90, color=col, alpha=0.88, edgecolor="#1f2d3d", linewidth=0.55, zorder=3)
        ax_sub.set_xticks(x)
        ax_sub.set_xticklabels([tr["track_title"] for tr in track_results], fontsize=8.6, fontweight="bold", rotation=10)
        ax_sub.set_ylim(0.36, 0.96)
        ax_sub.set_ylabel(ylab, fontsize=9.5, fontweight="bold")
        ax_sub.set_title(sub_title, fontsize=10.6, fontweight="bold")
        ax_sub.grid(axis="y", linestyle=":", alpha=0.55)

    # Row 3: Few-shot learning curves (3-Class Ordinal AUC)
    pct_keys = ["0.00", "0.10", "0.25", "0.50", "1.00"]
    pct_x = [0, 10, 25, 50, 100]
    for idx, tr in enumerate(track_results):
        ax = fig.add_subplot(gs[2, idx])
        fs = tr["few_shot_curve"]
        kd_z = fs["0.00"]["direct_kd_ord"]
        y_scr = [fs[k]["scratch_ord"] for k in pct_keys[1:]]
        y_l2 = [fs[k]["kd_l2sp_ord"] for k in pct_keys]
        y_ad = [fs[k]["kd_adapter_ord"] for k in pct_keys]

        ax.axhline(0.50, color="#c0392b", linestyle=":", linewidth=1.0)
        ax.axhline(kd_z, color="#e67e22", linestyle="--", linewidth=1.2, alpha=0.85, label="Direct KD (0%)")
        ax.plot(pct_x[1:], y_scr, marker="s", color="#7f8c8d", linewidth=1.6, markersize=4.4, label="Target Scratch")
        ax.plot(pct_x, y_l2, marker="o", color="#2ecc71", linewidth=1.9, markersize=4.6, label="KD -> L2-SP")
        ax.plot(pct_x, y_ad, marker="^", color="#1b7943", linewidth=1.9, markersize=4.8, label="KD -> Adapter")

        ax.set_title(tr["track_title"], fontsize=9.2, fontweight="bold")
        ax.set_xlabel("% Target Training Pairs", fontsize=8.3)
        if idx == 0:
            ax.set_ylabel("3-Class Ordinal AUC", fontsize=9.0, fontweight="bold")
            ax.legend(loc="lower right", fontsize=7.2, frameon=True)
        ax.set_xticks([0, 25, 50, 100])
        ax.set_ylim(0.42, 0.92)
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(
        "AbRank v2 — 3-State Cross-Assay Transfer & Fine-Tuning (KD -> ELISA, IC50, ΔΔG, DMS, Escape)",
        fontsize=13.5,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_transfer_markdown(
    track_results: list[dict],
    ic50_cross_virus: dict,
    out_md: Path,
) -> None:
    """Write standalone `results/MODALITY_TRANSFER_AND_FINETUNING.md` report."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    by_key = {r["track_key"]: r for r in track_results}

    def _best_ft(r: dict) -> tuple[str, dict]:
        m1 = r["methods"]["KD -> Target Fine-Tuned (L2-SP Hurdle)"]
        m2 = r["methods"]["KD -> Target Fine-Tuned (Residual Adapter)"]
        return ("KD -> Residual Adapter", m2) if m2["auc_ordinal"] >= m1["auc_ordinal"] else ("KD -> L2-SP Hurdle", m1)

    el = by_key["elisa"]
    ic = by_key["ic50"]
    sk = by_key["ddg_skempiv2"]
    ab_ddg = by_key["ddg_abdesign"]
    dms = by_key["ddg_abagym_dms"]
    esc = by_key["escape_bloom"]
    vhh_tr = by_key.get("vhh_kd")

    el_n, el_ft = _best_ft(el)
    ic_n, ic_ft = _best_ft(ic)
    sk_n, sk_ft = _best_ft(sk)
    abd_n, abd_ft = _best_ft(ab_ddg)
    dm_n, dm_ft = _best_ft(dms)
    es_n, es_ft = _best_ft(esc)
    vhh_bullet = []
    if vhh_tr is not None:
        vhh_n, vhh_ft = _best_ft(vhh_tr)
        vhh_bullet = [
            f"4. **Single-Domain Nanobody `VHH KD` (`Paired Fv -> Single-Domain VHH`, `n = {vhh_tr['n_pairs']:,}` pairs across {vhh_tr['n_groups']} unseen antigen clusters)**:",
            f"   - Direct `Paired-KD` Transfer reaches `{vhh_tr['methods']['Direct KD Transfer (0% Target Data)']['auc_ordinal']:.3f}` Ordinal AUC (`Sign`: `{vhh_tr['methods']['Direct KD Transfer (0% Target Data)']['auc_dir']:.3f}`, `Impact`: `{vhh_tr['methods']['Direct KD Transfer (0% Target Data)']['auc_impact']:.3f}`), and **`{vhh_n}`** lifts unseen-antigen VHH ranking to **`{vhh_ft['auc_ordinal']:.3f}` Ordinal AUC** (`Sign`: `{vhh_ft['auc_dir']:.3f}`, `Impact`: `{vhh_ft['auc_impact']:.3f}`).",
        ]

    lines = [
        "# AbRank v2 — 3-State Cross-Modality Transfer & Fine-Tuning (`KD` $\\to$ `ELISA`, `IC50`, `ΔΔG`, `Escape`)",
        "",
        "## Executive Summary",
        "",
        "Using AbRank v2's **3-state noise-buffered evaluation** (`Depleted -1`, `Neutral 0`, `Improved +1`), we test when a `KD`-pretrained model (`SPR` / `BLI` / `KinExA`) transfers directly versus when supervised target-assay fine-tuning is required—measuring **both** **Directional Sign AUC** (`+1` vs `-1`) and **Mutation Impact / Hot-Spot Detection AUC** (`±1` vs `0`):",
        "",
        f"1. **`ELISA` (`AbDesign`, `n = {el['n_pairs']:,}` pairs across {el['n_groups']} antigens) transfers cleanly without target labels**:",
        f"   - Zero-shot PLM (`ESMC+AbLang2`) and Direct `KD` Transfer already reach **`{el['methods']['Zero-Shot PLM (ESMC+AbLang2)']['auc_dir']:.3f}` / `{el['methods']['Direct KD Transfer (0% Target Data)']['auc_dir']:.3f}` Sign AUC** and **`{el['methods']['Direct KD Transfer (0% Target Data)']['auc_ordinal']:.3f}` 3-Class Ordinal AUC**, and **`{el_n}`** refines held-out antigen performance to **`{el_ft['auc_ordinal']:.3f}` Ordinal AUC** (`Sign`: `{el_ft['auc_dir']:.3f}`, `Impact`: `{el_ft['auc_impact']:.3f}`).",
        f"2. **`IC50` Neutralization (`CATNAP + Shanker + AbCoV`, `n = {ic['n_pairs']:,}` pairs across {ic['n_groups']} clonal lineages) correlates weakly with monomeric `KD` zero-shot (`{ic['methods']['Direct KD Transfer (0% Target Data)']['auc_ordinal']:.3f}` Ordinal AUC) and jumps dramatically under fine-tuning (`{ic_ft['auc_ordinal']:.3f}` Ordinal AUC, `{ic_ft['auc_dir']:.3f}` Sign AUC)**:",
        f"   - Because viral neutralization is dominated by trimeric envelope avidity, conformational masking, and N-linked glycan shields on `HIV1_Env` and `SARSCoV2_Spike`, supervised adaptation (`{ic_n}`) yields a **`+{ic_ft['auc_ordinal'] - ic['methods']['Direct KD Transfer (0% Target Data)']['auc_ordinal']:.3f}` Ordinal AUC gain** (`Sign`: `{ic['methods']['Direct KD Transfer (0% Target Data)']['auc_dir']:.3f}` $\\to$ **`{ic_ft['auc_dir']:.3f}`**, `Impact`: `{ic['methods']['Direct KD Transfer (0% Target Data)']['auc_impact']:.3f}` $\\to$ **`{ic_ft['auc_impact']:.3f}`**).",
        "3. **`ΔΔG` (`SKEMPIv2`, `AbDesign ΔΔG`, `AbAgym DMS`) & Viral `Escape` (`Bloom`) require fine-tuning to learn hot-spot impact (`±1` vs `0`) and alanine-scan / viral-escape rules**:",
        f"   - **`SKEMPIv2` Thermodynamic `ΔΔG` (`n = {sk['n_pairs']:,}`, `SKEMPIv2` strictly excluded from `KD`)**: Direct `KD` gets `{sk['methods']['Direct KD Transfer (0% Target Data)']['auc_ordinal']:.3f}` Ordinal AUC (`Impact`: `{sk['methods']['Direct KD Transfer (0% Target Data)']['auc_impact']:.3f}`), whereas **`{sk_n}`** reaches **`{sk_ft['auc_ordinal']:.3f}` Ordinal AUC** (`Sign`: **`{sk_ft['auc_dir']:.3f}`**, `Impact`: **`{sk_ft['auc_impact']:.3f}`**).",
        f"   - **`AbDesign` Mutational `ΔΔG` (`n = {ab_ddg['n_pairs']:,}`)**: Direct `KD` gets `{ab_ddg['methods']['Direct KD Transfer (0% Target Data)']['auc_ordinal']:.3f}` Ordinal AUC, rising to **`{abd_ft['auc_ordinal']:.3f}`** (`Sign`: `{abd_ft['auc_dir']:.3f}`, `Impact`: `{abd_ft['auc_impact']:.3f}`) with `{abd_n}`.",
        f"   - **`AbAgym` Interface DMS (`n = {dms['n_pairs']:,}`)**: Direct `KD` gets `{dms['methods']['Direct KD Transfer (0% Target Data)']['auc_ordinal']:.3f}` Ordinal AUC, climbing to **`{dm_ft['auc_ordinal']:.3f}`** (`Sign`: `{dm_ft['auc_dir']:.3f}`, `Impact`: `{dm_ft['auc_impact']:.3f}`) on held-out antigens.",
        f"   - **`RBD_Escape_Bloom` Viral Escape (`n = {esc['n_pairs']:,}`)**: Direct `KD` gets `{esc['methods']['Direct KD Transfer (0% Target Data)']['auc_ordinal']:.3f}` Ordinal AUC, jumping to **`{es_ft['auc_ordinal']:.3f}`** (`Sign`: **`{es_ft['auc_dir']:.3f}`**, `Impact`: **`{es_ft['auc_impact']:.3f}`**) on unseen antibody lineages.",
        *vhh_bullet,
        "",
        "![3-State Cross-Modality Transfer and Fine-Tuning](figures/modality_transfer_and_finetuning.png)",
        "",
        "---",
        "",
        "## 1. Main Per-Antigen Macro 3-State Transfer & Fine-Tuning Table (5-Fold Zero-Leakage GroupCV)",
        "",
        "Each cell reports **Per-Antigen Macro-Averaged `Sign AUC` / `Impact AUC` / `3-Class Ordinal AUC`** across antigens with `≥5` pairs (exact Pooled Micro-AUCs are also preserved in `results/modality_transfer_and_finetuning.json` and Section 4 below):",
        "",
        "| Target Assay Track | Pairs (`±1` / `0`) | Zero-Shot PLM | Direct `KD` (`0%` Target) | Target Scratch (`3-State`) | `KD -> L2-SP Hurdle` | `KD -> Residual Adapter` | Ordinal Gain (`FT - KD`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for tr in track_results:
        m = tr["methods"]
        zs = m["Zero-Shot PLM (ESMC+AbLang2)"]
        dk = m["Direct KD Transfer (0% Target Data)"]
        sc = m["Target From Scratch (3-State Hurdle)"]
        l2 = m["KD -> Target Fine-Tuned (L2-SP Hurdle)"]
        ad = m["KD -> Target Fine-Tuned (Residual Adapter)"]
        best_ord = max(l2["auc_ordinal"], ad["auc_ordinal"])
        lines.append(
            f"| **{tr['track_title']}** | `{tr['n_pairs']:,}` (`{tr['n_impactful']}` / `{tr['n_neutral']}`) | "
            f"`{zs['auc_dir']:.3f}` / `{zs['auc_impact']:.3f}` / `{zs['auc_ordinal']:.3f}` | "
            f"`{dk['auc_dir']:.3f}` / `{dk['auc_impact']:.3f}` / `{dk['auc_ordinal']:.3f}` | "
            f"`{sc['auc_dir']:.3f}` / `{sc['auc_impact']:.3f}` / `{sc['auc_ordinal']:.3f}` | "
            f"`{l2['auc_dir']:.3f}` / `{l2['auc_impact']:.3f}` / **`{l2['auc_ordinal']:.3f}`** | "
            f"`{ad['auc_dir']:.3f}` / `{ad['auc_impact']:.3f}` / **`{ad['auc_ordinal']:.3f}`** | "
            f"**`{best_ord - dk['auc_ordinal']:+.3f}`** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. `Ab-Mut` vs. `Ag-Mut` Domain-Shift Breakdown (`Sign` / `Impact` / `Ordinal`)",
        "",
        "| Track | Slice | Pairs | Zero-Shot PLM | Direct `KD` | `KD -> L2-SP Hurdle` | `KD -> Residual Adapter` |",
        "| :--- | :--- | ---: | :---: | :---: | :---: | :---: |",
    ])

    for tr in (ic, sk, dms):
        m = tr["methods"]
        for sl_lbl, pfx, cnt in (
            ("Antibody-Mutated (`Ab-Mut`)", "ab_mut", tr["n_ab_mut"]),
            ("Antigen-Mutated (`Ag-Mut`)", "ag_mut", tr["n_ag_mut"]),
        ):
            if cnt >= 10:
                def _fmt(md: dict) -> str:
                    d_v = md.get(f"{pfx}_dir")
                    i_v = md.get(f"{pfx}_imp")
                    o_v = md.get(f"{pfx}_ord")
                    ds = f"{d_v:.3f}" if d_v is not None else "—"
                    is_ = f"{i_v:.3f}" if i_v is not None else "—"
                    os_ = f"{o_v:.3f}" if o_v is not None else "—"
                    return f"`{ds}` / `{is_}` / `{os_}`"

                lines.append(
                    f"| **{tr['track_title']}** | {sl_lbl} | {cnt:,} | "
                    f"{_fmt(m['Zero-Shot PLM (ESMC+AbLang2)'])} | "
                    f"{_fmt(m['Direct KD Transfer (0% Target Data)'])} | "
                    f"{_fmt(m['KD -> Target Fine-Tuned (L2-SP Hurdle)'])} | "
                    f"**{_fmt(m['KD -> Target Fine-Tuned (Residual Adapter)'])}** |"
                )

    lines.extend([
        "",
        "### Cross-Virus `IC50` Transfer Check (`HIV1_Env` $\\leftrightarrow$ `Coronavirus Spike`, 3-Class Ordinal AUC)",
        "",
        f"- **Train on `HIV-1 Env` (`n={ic50_cross_virus['n_hiv']}`) $\\to$ Test Zero-Shot on `CoV Spike` (`n={ic50_cross_virus['n_cov']}`)**:",
        f"  - `Direct KD`: `{ic50_cross_virus['hiv_to_cov_direct_kd_ord']:.3f}` $\\to$ `KD + HIV-1 L2-SP`: **`{ic50_cross_virus['hiv_to_cov_l2sp_ord']:.3f}`** (`HIV-only Scratch`: `{ic50_cross_virus['hiv_to_cov_scratch_ord']:.3f}`).",
        f"- **Train on `CoV Spike` (`n={ic50_cross_virus['n_cov']}`) $\\to$ Test Zero-Shot on `HIV-1 Env` (`n={ic50_cross_virus['n_hiv']}`)**:",
        f"  - `Direct KD`: `{ic50_cross_virus['cov_to_hiv_direct_kd_ord']:.3f}` $\\to$ `KD + CoV L2-SP`: **`{ic50_cross_virus['cov_to_hiv_l2sp_ord']:.3f}`** (`CoV-only Scratch`: `{ic50_cross_virus['cov_to_hiv_scratch_ord']:.3f}`).",
        "",
        "---",
        "",
        "## 3. Few-Shot Fine-Tuning Sample Efficiency (`3-Class Ordinal AUC` across `0%` $\\to$ `100%` Target Data)",
        "",
        "| Target Assay Track | Method | `0%` (`Direct KD`) | `10%` Target | `25%` Target | `50%` Target | `100%` Target |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for tr in track_results:
        fs = tr["few_shot_curve"]
        lines.append(
            f"| **{tr['track_title']}** | `Target Scratch` | `0.500` | "
            f"`{fs['0.10']['scratch_ord']:.3f}` (n={fs['0.10']['n_train_avg']}) | "
            f"`{fs['0.25']['scratch_ord']:.3f}` | `{fs['0.50']['scratch_ord']:.3f}` | "
            f"`{fs['1.00']['scratch_ord']:.3f}` (n={fs['1.00']['n_train_avg']}) |"
        )
        lines.append(
            f"| **{tr['track_title']}** | **`KD -> L2-SP Hurdle`** | **`{fs['0.00']['direct_kd_ord']:.3f}`** | "
            f"**`{fs['0.10']['kd_l2sp_ord']:.3f}`** | **`{fs['0.25']['kd_l2sp_ord']:.3f}`** | "
            f"**`{fs['0.50']['kd_l2sp_ord']:.3f}`** | **`{fs['1.00']['kd_l2sp_ord']:.3f}`** |"
        )
        lines.append(
            f"| **{tr['track_title']}** | **`KD -> Adapter`** | **`{fs['0.00']['direct_kd_ord']:.3f}`** | "
            f"`{fs['0.10']['kd_adapter_ord']:.3f}` | `{fs['0.25']['kd_adapter_ord']:.3f}` | "
            f"`{fs['0.50']['kd_adapter_ord']:.3f}` | **`{fs['1.00']['kd_adapter_ord']:.3f}`** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Per-Antigen Macro-AUC (`≥5` Pairs) vs. Pooled Micro-AUC Comparison",
        "",
        "| Target Assay Track | Eval Ags (`≥5`) | Direct `KD` Macro / Pooled Ord | `KD -> L2-SP` Macro / Pooled Ord | `KD -> Adapter` Macro / Pooled Ord |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for tr in track_results:
        m = tr["methods"]
        dk = m["Direct KD Transfer (0% Target Data)"]
        l2 = m["KD -> Target Fine-Tuned (L2-SP Hurdle)"]
        ad = m["KD -> Target Fine-Tuned (Residual Adapter)"]
        lines.append(
            f"| **{tr['track_title']}** | `{dk['n_antigens_evaluated']}` | "
            f"**`{dk['auc_ordinal']:.3f}`** / `{dk['pooled_auc_ordinal']:.3f}` | "
            f"**`{l2['auc_ordinal']:.3f}`** / `{l2['pooled_auc_ordinal']:.3f}` | "
            f"**`{ad['auc_ordinal']:.3f}`** / `{ad['pooled_auc_ordinal']:.3f}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python scripts/05_extract_embeddings.py --workers 16",
        "python scripts/07_run_modality_transfer_finetuning.py",
        "```",
        "",
    ])

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3-state KD -> ELISA, IC50, DDG & Escape transfer & fine-tuning.")
    parser.add_argument("--report-only", action="store_true", help="Regenerate plot & Markdown from saved JSON")
    args = parser.parse_args()

    t0 = time.time()
    root = get_repo_root()
    bdir = root / "data" / "benchmark"
    emb_dir = root / "data" / "embeddings"
    res_dir = root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_json = res_dir / "modality_transfer_and_finetuning.json"

    if args.report_only and out_json.exists():
        saved = json.loads(out_json.read_text(encoding="utf-8"))
        make_transfer_figure(saved["tracks"], res_dir / "figures" / "modality_transfer_and_finetuning.png")
        write_transfer_markdown(saved["tracks"], saved["ic50_cross_virus"], res_dir / "MODALITY_TRANSFER_AND_FINETUNING.md")
        print("Regenerated figure and MODALITY_TRANSFER_AND_FINETUNING.md from JSON.")
        return

    print("Loading master assay tables and splits...")
    kd_raw = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False)
    kd_lookup = kd_raw.drop_duplicates("AbAgID").set_index("AbAgID")
    cma_lookup = pd.read_csv(bdir / "assays_cross_modal.tsv", sep="\t", low_memory=False).drop_duplicates("AbAgID").set_index("AbAgID")
    tfa_raw = pd.read_csv(bdir / "assays_transfer.tsv", sep="\t", low_memory=False)
    tfa_lookup = tfa_raw.drop_duplicates("AbAgID").set_index("AbAgID")

    clust_to_groups: dict[str, set[str]] = {}
    for grp, clust in kd_raw[["Ag_group", "Ag_cluster"]].drop_duplicates().itertuples(index=False):
        clust_to_groups.setdefault(str(clust), set()).add(str(grp))
    ag_to_homolog_groups = {str(grp): set(clust_to_groups[str(clust)]) for grp, clust in kd_raw[["Ag_group", "Ag_cluster"]].drop_duplicates().itertuples(index=False)}

    combined_ab = pd.concat(
        [
            kd_raw[["AbID", "Ab_heavy_chain_seq", "Ab_light_chain_seq", "WT_Abname", "Dataset"]],
            tfa_raw[["AbID", "Ab_heavy_chain_seq", "Ab_light_chain_seq", "WT_Abname", "Dataset"]],
        ],
        ignore_index=True,
    ).drop_duplicates("AbID")
    combined_ab["Ab_lineage_joint"] = assign_antibody_lineages(combined_ab, identity_threshold=0.80)
    abid_to_joint_lin = combined_ab.set_index("AbID")["Ab_lineage_joint"].to_dict()
    kd_lookup["Ab_lineage"] = [abid_to_joint_lin.get(i, l) for i, l in zip(kd_lookup["AbID"], kd_lookup["Ab_lineage"])]

    print("Loading cached PLM embeddings (`esmc_300m.pt`, `ablang2_paired.pt`)...")
    esmc_cache = torch.load(emb_dir / "esmc_300m.pt", map_location="cpu", weights_only=False)
    ablang_cache = torch.load(emb_dir / "ablang2_paired.pt", map_location="cpu", weights_only=False)

    kd_pool_parts = [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test", "test_candidates")]
    kd_pool_df = pd.concat(kd_pool_parts, ignore_index=True).drop_duplicates(["AbAgID_1", "AbAgID_2"]).reset_index(drop=True)
    kd_pool_tp = assemble_transfer_tensors(kd_pool_df, kd_lookup, esmc_cache, ablang_cache)
    X_kd_pool = build_transfer_odd_matrix(kd_pool_tp, include_biophys=True)

    cm_all = pd.read_csv(bdir / "splits" / "cross_modal" / "modality_test.tsv", sep="\t", low_memory=False)
    elisa_df = cm_all[cm_all["Assay_modality"] == "ELISA"].reset_index(drop=True)
    abdesign_ddg_df = cm_all[cm_all["Assay_modality"] == "DDG"].reset_index(drop=True)
    dms_df = cm_all[cm_all["Assay_modality"] == "DMS"].reset_index(drop=True)

    ic50_df = pd.read_csv(bdir / "splits" / "transfer" / "ic50_test.tsv", sep="\t", low_memory=False).reset_index(drop=True)
    esc_df = pd.read_csv(bdir / "splits" / "transfer" / "escape_test.tsv", sep="\t", low_memory=False).reset_index(drop=True)

    vhh_raw = pd.read_csv(bdir / "assays_vhh.tsv", sep="\t", low_memory=False)
    vhh_raw["Ab_light_chain_seq"] = ""
    vhh_lookup = vhh_raw.drop_duplicates("AbAgID").set_index("AbAgID")
    vhh_pairs_all = pd.read_csv(bdir / "splits" / "vhh" / "vhh_ranking_test.tsv", sep="\t", low_memory=False)
    vhh_df = vhh_pairs_all[vhh_pairs_all["Category"] != "Global"].reset_index(drop=True)

    # Also map VHH Ag_groups into ag_to_homolog_groups via <50% sequence homology to Core KD antigens
    kd_ag_seqs_by_grp = kd_raw.groupby("Ag_group")["Ag_seq"].first().to_dict()
    import Levenshtein as _Lev
    for v_grp, v_seq in vhh_raw.groupby("Ag_group")["Ag_seq"].first().items():
        homologs = set(ag_to_homolog_groups.get(str(v_grp), {str(v_grp)}))
        for k_grp, k_seq in kd_ag_seqs_by_grp.items():
            s_short, s_long = (str(v_seq), str(k_seq)) if len(str(v_seq)) <= len(str(k_seq)) else (str(k_seq), str(v_seq))
            if (len(s_short) >= 25 and s_short in s_long) or _Lev.ratio(str(v_seq), str(k_seq), score_cutoff=0.50) >= 0.50:
                homologs.update(ag_to_homolog_groups.get(str(k_grp), {str(k_grp)}))
        ag_to_homolog_groups[str(v_grp)] = homologs

    skempi_assays = kd_raw[kd_raw["Dataset"] == "SKEMPIv2"].copy()
    skempi_df, _ = build_within_antigen_pairs(
        skempi_assays,
        m=1.0,
        m_neutral=0.30,
        include_neutral=True,
        neutral_ratio=1.0,
        modality="DDG_SKEMPIv2",
        ab_cap=3,
        per_ag_cap=500,
        seed=42,
        prefix="SK10",
    )
    skempi_df = skempi_df.reset_index(drop=True)

    print("Assembling target track 3-state pair tensors...")
    elisa_tp = assemble_transfer_tensors(elisa_df, cma_lookup, esmc_cache, ablang_cache)
    ic50_tp = assemble_transfer_tensors(ic50_df, tfa_lookup, esmc_cache, ablang_cache)
    skempi_tp = assemble_transfer_tensors(skempi_df, kd_lookup, esmc_cache, ablang_cache)
    abddg_tp = assemble_transfer_tensors(abdesign_ddg_df, cma_lookup, esmc_cache, ablang_cache)
    dms_tp = assemble_transfer_tensors(dms_df, cma_lookup, esmc_cache, ablang_cache)
    esc_tp = assemble_transfer_tensors(esc_df, tfa_lookup, esmc_cache, ablang_cache)
    vhh_tp = assemble_transfer_tensors(vhh_df, vhh_lookup, esmc_cache, ablang_cache)

    elisa_groups = elisa_df["Ag_group_1"].astype(str).to_numpy()
    skempi_groups = skempi_df["Ag_group_1"].astype(str).to_numpy()
    abddg_groups = abdesign_ddg_df["Ag_group_1"].astype(str).to_numpy()
    dms_groups = dms_df["Ag_group_1"].astype(str).to_numpy()
    vhh_groups = np.array([vhh_lookup.loc[a, "Ag_cluster"] for a in vhh_df["AbAgID_1"]], dtype=object)
    ic50_groups = np.array([abid_to_joint_lin[tfa_lookup.loc[a, "AbID"]] for a in ic50_df["AbAgID_1"]], dtype=object)
    esc_groups = np.array([abid_to_joint_lin[tfa_lookup.loc[a, "AbID"]] for a in esc_df["AbAgID_1"]], dtype=object)

    shared_kd_cache: dict[tuple, KDSourceRanker] = {}
    common_kw = dict(
        kd_pool_df=kd_pool_df,
        kd_pool_tp=kd_pool_tp,
        X_kd_pool=X_kd_pool,
        kd_lookup=kd_lookup,
        ag_to_homolog_groups=ag_to_homolog_groups,
        kd_cache=shared_kd_cache,
    )

    track_specs = [
        ("elisa", "ELISA (AbDesign)", elisa_df, elisa_tp, elisa_groups, "antigen", False),
        ("ic50", "IC50 Neutralization", ic50_df, ic50_tp, ic50_groups, "lineage", False),
        ("ddg_skempiv2", "ΔΔG (SKEMPIv2)", skempi_df, skempi_tp, skempi_groups, "antigen", True),
        ("ddg_abdesign", "ΔΔG (AbDesign)", abdesign_ddg_df, abddg_tp, abddg_groups, "antigen", True),
        ("ddg_abagym_dms", "ΔΔG / DMS (AbAgym)", dms_df, dms_tp, dms_groups, "antigen", False),
        ("escape_bloom", "Escape DMS (Bloom)", esc_df, esc_tp, esc_groups, "lineage", False),
        ("vhh_kd", "VHH Nanobody KD", vhh_df, vhh_tp, vhh_groups, "antigen", True),
    ]

    track_results: list[dict] = []
    for t_key, t_title, t_df, t_tp, t_grp, g_mode, excl_sk in track_specs:
        print(f"\n=== Evaluating {t_title} (n={len(t_df):,}, groups={len(np.unique(t_grp))}, mode={g_mode}) ===")
        res = evaluate_target_track(
            t_key, t_title, t_df, t_tp, t_grp, g_mode,
            exclude_skempi_from_kd=excl_sk,
            **common_kw,
        )
        track_results.append(res)
        for m_name in METHODS[1:]:
            ms = res["methods"][m_name]
            print(
                f"  {m_name:42s} | Sign={ms['auc_dir']:.3f} | Impact={ms['auc_impact']:.3f} | Ordinal={ms['auc_ordinal']:.3f}"
            )

    print("\n=== Running Cross-Virus IC50 Transfer Check (HIV-1 <-> CoV) ===")
    X_ic50 = build_transfer_odd_matrix(ic50_tp, include_biophys=True)
    is_hiv = (ic50_df["Ag_group_1"] == "HIV1_Env").to_numpy()
    is_cov = ~is_hiv
    kd_no_viral_ic = build_leak_free_kd_source(
        kd_pool_df, kd_pool_tp, X_kd_pool, kd_lookup,
        exclude_ag_groups=set(),
        exclude_ab_lineages=set(ic50_groups.tolist()),
        exclude_skempi=False,
    )
    tp_hiv = ic50_tp.subset(np.flatnonzero(is_hiv))
    tp_cov = ic50_tp.subset(np.flatnonzero(is_cov))

    hiv_ft = L2SPFineTunedRanker(kd_source=kd_no_viral_ic).fit(X_ic50[is_hiv], tp_hiv.y, ic50_groups[is_hiv], tp_tr=tp_hiv)
    hiv_scr = L2SPFineTunedRanker(kd_source=None).fit(X_ic50[is_hiv], tp_hiv.y, ic50_groups[is_hiv], tp_tr=tp_hiv)
    cov_ft = L2SPFineTunedRanker(kd_source=kd_no_viral_ic).fit(X_ic50[is_cov], tp_cov.y, ic50_groups[is_cov], tp_tr=tp_cov)
    cov_scr = L2SPFineTunedRanker(kd_source=None).fit(X_ic50[is_cov], tp_cov.y, ic50_groups[is_cov], tp_tr=tp_cov)

    hiv_df = ic50_df.iloc[np.flatnonzero(is_hiv)].reset_index(drop=True)
    cov_df = ic50_df.iloc[np.flatnonzero(is_cov)].reset_index(drop=True)

    def _ord_on(sub_df: pd.DataFrame, d_p: np.ndarray, i_p: np.ndarray) -> float:
        return float(compute_per_antigen_macro_metrics(sub_df, d_p, i_p, min_pairs=5)["auc_ordinal"])

    ic50_cross_virus = {
        "n_hiv": int(is_hiv.sum()),
        "n_cov": int(is_cov.sum()),
        "hiv_to_cov_direct_kd_ord": round(_ord_on(cov_df, *kd_no_viral_ic.predict_3state(tp_cov, X_ic50[is_cov])), 4),
        "hiv_to_cov_scratch_ord": round(_ord_on(cov_df, *hiv_scr.predict_3state(tp_cov, X_ic50[is_cov])), 4),
        "hiv_to_cov_l2sp_ord": round(_ord_on(cov_df, *hiv_ft.predict_3state(tp_cov, X_ic50[is_cov])), 4),
        "cov_to_hiv_direct_kd_ord": round(_ord_on(hiv_df, *kd_no_viral_ic.predict_3state(tp_hiv, X_ic50[is_hiv])), 4),
        "cov_to_hiv_scratch_ord": round(_ord_on(hiv_df, *cov_scr.predict_3state(tp_hiv, X_ic50[is_hiv])), 4),
        "cov_to_hiv_l2sp_ord": round(_ord_on(hiv_df, *cov_ft.predict_3state(tp_hiv, X_ic50[is_hiv])), 4),
    }
    print("Cross-virus IC50 transfer (3-Class Ordinal AUC):", ic50_cross_virus)

    payload = {
        "benchmark": "AbRank v2.0.0",
        "study": "3-State KD -> ELISA / IC50 / DDG / Escape Modality Transfer & Fine-Tuning",
        "tracks": track_results,
        "ic50_cross_virus": ic50_cross_virus,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fig_path = res_dir / "figures" / "modality_transfer_and_finetuning.png"
    make_transfer_figure(track_results, fig_path)

    md_path = res_dir / "MODALITY_TRANSFER_AND_FINETUNING.md"
    write_transfer_markdown(track_results, ic50_cross_virus, md_path)

    print(f"\nCompleted in {time.time() - t0:.1f}s! Wrote:\n  - {out_json}\n  - {fig_path}\n  - {md_path}")


if __name__ == "__main__":
    main()
