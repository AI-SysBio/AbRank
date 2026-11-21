#!/usr/bin/env python3
"""Step 6: Train and evaluate zero-shot, classical, binary-probe, and 3-state hurdle/neural
baselines on AbRank v2's `unseen_ag` generalization benchmark (`{-1, 0, +1}` pairs + unfiltered
variant panels).

Outputs:
  - `results/unseen_ag_baselines.json`
  - `results/figures/unseen_ag_baselines_auc.png`
  - `results/UNSEEN_ANTIGEN_BASELINES.md`
"""

from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

import argparse
from itertools import combinations
import json
import time
from pathlib import Path
import Levenshtein
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(8)

from abrank.config import get_repo_root
from abrank.evaluate import (
    compute_3state_metrics,
    compute_per_antigen_macro_metrics,
    evaluate_pairwise_predictions,
    evaluate_pointwise_spearman,
)
from abrank.schema import PAIR_COLUMNS
from abrank.baselines.features import build_complex_handcrafted_table
from abrank.baselines.masked_marginal import (
    compute_ablang2_masked_marginals,
    compute_esmc_masked_marginals,
)
from abrank.baselines.models import (
    AntisymmetricLinearRanker,
    AntisymmetricXGBoostRanker,
    NeuralPairwiseTrainer,
    OrdinalHurdleRanker,
    SplitPairTensors,
    SymmetricImpactClassifier,
    TwoHeadRegimeRanker,
    assemble_split_tensors,
    build_odd_linear_matrix,
    compute_stratum_weights,
    sym_auc,
)
from abrank.baselines.logica_models import (
    LogiCA3StateTrainer,
)
from abrank.baselines.structure_models import (
    ANTIPASTI3StateTrainer,
    GearBind3StateTrainer,
    assemble_structure_pair_features,
    fit_3state_foundation_hurdle,
    fit_walle_affinity_3state,
)


def bootstrap_3state_slice(
    labels: np.ndarray,
    bdiff: np.ndarray,
    pred_diff: np.ndarray,
    pred_impact: np.ndarray | None = None,
    n_boot: int = 800,
    seed: int = 42,
) -> dict:
    """Compute 3-state slice metrics (`auc_dir`, `auc_impact`, `auc_ordinal`, `auc_selective_hit`) + 95% CIs."""
    base = compute_3state_metrics(labels, bdiff, pred_diff, pred_impact)
    if len(labels) < 8 or n_boot <= 0:
        return base

    pimp = np.abs(pred_diff) if pred_impact is None else pred_impact
    rng = np.random.default_rng(seed)
    n = len(labels)
    b_dir, b_imp, b_ord, b_hit = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_3state_metrics(labels[idx], bdiff[idx], pred_diff[idx], pimp[idx])
        if m["auc_dir"] is not None:
            b_dir.append(m["auc_dir"])
        if m["auc_impact"] is not None:
            b_imp.append(m["auc_impact"])
        if m["auc_ordinal"] is not None:
            b_ord.append(m["auc_ordinal"])
        if m["auc_selective_hit"] is not None:
            b_hit.append(m["auc_selective_hit"])

    out = dict(base)
    for k, arr in (("auc_dir_ci95", b_dir), ("auc_impact_ci95", b_imp), ("auc_ordinal_ci95", b_ord), ("auc_selective_hit_ci95", b_hit)):
        out[k] = [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)] if arr else None
    return out


def seq_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b))


def _max_seq_id(q: str, targets: list[str], cutoff: float) -> bool:
    """Return True iff `q` has `< cutoff` global alignment/edit identity (`1 - d_Lev / max(L1, L2)`)
    and zero substring containment against `targets` (using C `score_cutoff` early exit!).
    """
    lq = len(q)
    for t in targets:
        lt = len(t)
        short_s, long_s = (q, t) if lq <= lt else (t, q)
        if len(short_s) >= 25 and short_s in long_s:
            return False
        max_len = lq if lq >= lt else lt
        # Need 1 - d / max_len >= cutoff  <=>  d <= floor((1 - cutoff) * max_len)
        max_d = int((1.0 - cutoff) * max_len)
        if abs(lq - lt) > max_d:
            continue
        if Levenshtein.distance(q, t, score_cutoff=max_d) <= max_d:
            return False
    return True


def build_strict_unseen_cross_modal(root: Path, tr_df: pd.DataFrame, kd_lookup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter `cross_modal/modality_test.tsv` (`Enrichment`, `ELISA`, `DDG`, `EC50`) strictly to
    antigens with `<50%` sequence identity AND antibodies with `<80%` paired Fv / VHH identity
    relative to `unseen_ag/train`.
    """
    bdir = root / "data" / "benchmark"
    cm_pairs = pd.read_csv(bdir / "splits" / "cross_modal" / "modality_test.tsv", sep="\t", low_memory=False)
    cma = pd.read_csv(bdir / "assays_cross_modal.tsv", sep="\t", low_memory=False).drop_duplicates("AbAgID").set_index("AbAgID")

    tr_ids = list(set(tr_df["AbAgID_1"]).union(tr_df["AbAgID_2"]))
    tr_sub = kd_lookup.loc[tr_ids]
    tr_ag_groups = set(tr_df["Ag_group_1"]).union(tr_df["Ag_group_2"])
    tr_ag_seqs = sorted(set(tr_sub["Ag_seq"]))
    tr_ab_seqs = sorted(set(tr_sub["Ab_seq"]))
    tr_vh_seqs = sorted(set(tr_sub["Ab_heavy_chain_seq"]))

    cand = cm_pairs[
        (~cm_pairs["Ag_group_1"].isin(tr_ag_groups))
        & (~cm_pairs["Ag_group_2"].isin(tr_ag_groups))
        & (cm_pairs["Assay_modality"].isin(["Enrichment", "ELISA", "DDG", "EC50"]))
    ].copy()

    used_ids = list(set(cand["AbAgID_1"]).union(cand["AbAgID_2"]))
    sub_cma = cma.loc[used_ids]

    valid_ags = {ag_s for ag_s in sub_cma["Ag_seq"].unique() if _max_seq_id(str(ag_s), tr_ag_seqs, 0.50)}
    valid_abs: set[str] = set()
    max_tr_vh_len = max((len(s) for s in tr_vh_seqs), default=0)
    # Pre-check representative sequence per (Dataset, WT_Abname) using exact Levenshtein triangle inequality
    rep_min_d: dict[tuple[str, str], tuple[str, int, bool]] = {}
    for ds_n, wt_n, ab_s, l_s in sub_cma[["Dataset", "WT_Abname", "Ab_seq", "Ab_light_chain_seq"]].drop_duplicates("Ab_seq").itertuples(index=False):
        s_ab = str(ab_s)
        has_l = pd.notna(l_s) and len(str(l_s)) > 0
        if not has_l:
            gkey = (str(ds_n), str(wt_n))
            if gkey in rep_min_d:
                rep_seq, min_d_rep, rep_ok = rep_min_d[gkey]
                if rep_ok and len(s_ab) == len(rep_seq) and len(s_ab) >= 25:
                    d_mut = Levenshtein.distance(s_ab, rep_seq, score_cutoff=6)
                    if d_mut <= 5:
                        req_d = int(0.20 * max(len(s_ab), max_tr_vh_len))
                        if min_d_rep - d_mut > req_d:
                            valid_abs.add(s_ab)
                            continue
            else:
                ok0 = _max_seq_id(s_ab, tr_vh_seqs, 0.80)
                min_d0 = min((Levenshtein.distance(s_ab, t) for t in tr_vh_seqs), default=999) if ok0 else 0
                rep_min_d[gkey] = (s_ab, min_d0, ok0)
                if ok0:
                    valid_abs.add(s_ab)
                continue
        ref_pool = tr_ab_seqs if has_l else tr_vh_seqs
        if _max_seq_id(s_ab, ref_pool, 0.80):
            valid_abs.add(s_ab)

    valid_abag = set(sub_cma.index[sub_cma["Ag_seq"].isin(valid_ags) & sub_cma["Ab_seq"].isin(valid_abs)])
    strict_cm = cand[cand["AbAgID_1"].isin(valid_abag) & cand["AbAgID_2"].isin(valid_abag)].reset_index(drop=True)
    return strict_cm, cma


def build_strict_unseen_vhh(root: Path, tr_df: pd.DataFrame, kd_lookup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter `vhh/vhh_ranking_test.tsv` strictly to antigens with `<50%` sequence identity
    AND VHHs with `<80%` identity relative to `unseen_ag/train` so zero-shot Paired Fv -> VHH
    transfer has 0% antigen or antibody overlap with training!
    """
    bdir = root / "data" / "benchmark"
    vhh_pairs = pd.read_csv(bdir / "splits" / "vhh" / "vhh_ranking_test.tsv", sep="\t", low_memory=False)
    vhha = pd.read_csv(bdir / "assays_vhh.tsv", sep="\t", low_memory=False).drop_duplicates("AbAgID").set_index("AbAgID")
    vhha["Ab_light_chain_seq"] = ""

    tr_ids = list(set(tr_df["AbAgID_1"]).union(tr_df["AbAgID_2"]))
    tr_sub = kd_lookup.loc[tr_ids]
    tr_ag_groups = set(tr_df["Ag_group_1"]).union(tr_df["Ag_group_2"])
    tr_ag_seqs = sorted(set(tr_sub["Ag_seq"]))
    tr_vh_seqs = sorted(set(tr_sub["Ab_heavy_chain_seq"]))

    cand = vhh_pairs[
        (~vhh_pairs["Ag_group_1"].isin(tr_ag_groups))
        & (~vhh_pairs["Ag_group_2"].isin(tr_ag_groups))
    ].copy()
    used_ids = list(set(cand["AbAgID_1"]).union(cand["AbAgID_2"]))
    sub_vhha = vhha.loc[used_ids]

    valid_ags = {ag_s for ag_s in sub_vhha["Ag_seq"].unique() if _max_seq_id(str(ag_s), tr_ag_seqs, 0.50)}
    valid_vhhs = {vh_s for vh_s in sub_vhha["Ab_heavy_chain_seq"].unique() if _max_seq_id(str(vh_s), tr_vh_seqs, 0.80)}

    valid_abag = set(sub_vhha.index[sub_vhha["Ag_seq"].isin(valid_ags) & sub_vhha["Ab_heavy_chain_seq"].isin(valid_vhhs)])
    strict_vhh = cand[cand["AbAgID_1"].isin(valid_abag) & cand["AbAgID_2"].isin(valid_abag)].reset_index(drop=True)
    return strict_vhh, vhha


def build_panel_tournament_pairs(pw_df: pd.DataFrame, min_group_size: int = 6) -> pd.DataFrame:
    """Build all-pairs tournament table within each unfiltered `(Ag_group, Dataset)` panel."""
    rows = []
    for (ag_grp, ds), g in pw_df.groupby(["Ag_group", "Dataset"], sort=True):
        if len(g) < min_group_size or g["binding_score"].nunique() < 3:
            continue
        ids = g["AbAgID"].to_numpy()
        scores = g["binding_score"].to_numpy(dtype=float)
        ag_t = str(g["Ag_type"].iloc[0])
        for i, j in combinations(range(len(g)), 2):
            diff = float(scores[i] - scores[j])
            rows.append(
                {
                    "pair_id": f"PNL_{len(rows) + 1:06d}",
                    "AbAgID_1": ids[i],
                    "AbAgID_2": ids[j],
                    "binding_diff": diff,
                    "logKD_diff": -diff,
                    "label": 1 if diff >= 1.0 else (-1 if diff <= -1.0 else 0),
                    "Category": "Candidate",
                    "Lev_distance": 5,
                    "AbRank_group": f"{ag_grp}::{ds}",
                    "Ag_group_1": ag_grp,
                    "Ag_group_2": ag_grp,
                    "Ag_type": ag_t,
                    "Assay_modality": "KD",
                    "Dataset_1": ds,
                    "Dataset_2": ds,
                }
            )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS) if rows else pd.DataFrame(columns=PAIR_COLUMNS)


def tournament_scores_from_pairs(panel_pairs_df: pd.DataFrame, pred_diff: np.ndarray) -> dict[str, float]:
    """Aggregate antisymmetric pairwise scores `f(i, j)` into Copeland/Borda complex scores."""
    sums: dict[str, float] = {}
    cnts: dict[str, int] = {}
    id1 = panel_pairs_df["AbAgID_1"].to_numpy()
    id2 = panel_pairs_df["AbAgID_2"].to_numpy()
    for a, b, p in zip(id1, id2, pred_diff):
        fp = float(p)
        sums[a] = sums.get(a, 0.0) + fp
        cnts[a] = cnts.get(a, 0) + 1
        sums[b] = sums.get(b, 0.0) - fp
        cnts[b] = cnts.get(b, 0) + 1
    return {k: sums[k] / max(cnts[k], 1) for k in sums}


def detailed_eval(
    name: str,
    family: str,
    pv_d: np.ndarray,
    pv_i: np.ndarray | None,
    pt_d: np.ndarray,
    pt_i: np.ndarray | None,
    pc_d: np.ndarray,
    pc_i: np.ndarray | None,
    pcm_d: np.ndarray,
    pcm_i: np.ndarray | None,
    ppnl_d: np.ndarray,
    pvhh_d: np.ndarray,
    pvhh_i: np.ndarray | None,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    cm_df: pd.DataFrame,
    vhh_df: pd.DataFrame,
    pw_test_df: pd.DataFrame,
    pnl_pairs_df: pd.DataFrame,
    val_sp: SplitPairTensors,
    test_sp: SplitPairTensors,
    cand_sp: SplitPairTensors,
    cm_sp: SplitPairTensors,
    vhh_sp: SplitPairTensors,
    cocrys_test_mask: np.ndarray | None = None,
) -> dict:
    """Compute comprehensive Per-Antigen Macro-Averaged (`min_pairs=5`) + Pooled 3-state and full-panel metrics."""
    test_full = evaluate_pairwise_predictions(test_df, pt_d, pt_i, n_bootstrap=600)
    cand_full = evaluate_pairwise_predictions(cand_df, pc_d, pc_i, n_bootstrap=600)

    t_bdiff = test_df["binding_diff"].to_numpy(dtype=np.float64)

    t_loc = test_df["Category"].isin(["L1", "L2", "L3"]).to_numpy()
    t_glb = (test_df["Category"] == "Global").to_numpy()
    t_abm = t_loc & (~test_sp.is_ag_mut)
    t_agm = t_loc & test_sp.is_ag_mut

    loc_res = compute_per_antigen_macro_metrics(
        test_df.iloc[np.flatnonzero(t_loc)],
        pt_d[t_loc],
        pt_i[t_loc] if pt_i is not None else None,
        min_pairs=5,
        n_bootstrap=600,
    )
    cocrys_loc_res = None
    if cocrys_test_mask is not None:
        t_cc = t_loc & cocrys_test_mask
        if t_cc.sum() >= 10:
            cocrys_loc_res = compute_per_antigen_macro_metrics(
                test_df.iloc[np.flatnonzero(t_cc)],
                pt_d[t_cc],
                pt_i[t_cc] if pt_i is not None else None,
                min_pairs=5,
                n_bootstrap=600,
            )
    abm_res = compute_per_antigen_macro_metrics(
        test_df.iloc[np.flatnonzero(t_abm)],
        pt_d[t_abm],
        pt_i[t_abm] if pt_i is not None else None,
        min_pairs=5,
    )
    agm_res = compute_per_antigen_macro_metrics(
        test_df.iloc[np.flatnonzero(t_agm)],
        pt_d[t_agm],
        pt_i[t_agm] if pt_i is not None else None,
        min_pairs=5,
    )
    glb_res = bootstrap_3state_slice(
        test_sp.labels[t_glb], t_bdiff[t_glb], pt_d[t_glb], pt_i[t_glb] if pt_i is not None else None, n_boot=600
    )

    v_loc = val_df["Category"].isin(["L1", "L2", "L3"]).to_numpy()
    v_loc_res = compute_per_antigen_macro_metrics(
        val_df.iloc[np.flatnonzero(v_loc)],
        pv_d[v_loc],
        pv_i[v_loc] if pv_i is not None else None,
        min_pairs=5,
        n_bootstrap=400,
    )

    cm_res = compute_per_antigen_macro_metrics(
        cm_df, pcm_d, pcm_i, min_pairs=5, n_bootstrap=500
    )
    cm_by_mod = {}
    for mod, g in cm_df.groupby("Assay_modality", sort=True):
        idx = g.index.to_numpy()
        cm_by_mod[str(mod)] = compute_per_antigen_macro_metrics(
            g, pcm_d[idx], pcm_i[idx] if pcm_i is not None else None, min_pairs=5
        )

    vhh_bdiff = vhh_df["binding_diff"].to_numpy(dtype=np.float64)
    vhh_same_ag = (vhh_df["Category"] != "Global").to_numpy()
    vhh_within_res = compute_per_antigen_macro_metrics(
        vhh_df.iloc[np.flatnonzero(vhh_same_ag)],
        pvhh_d[vhh_same_ag],
        pvhh_i[vhh_same_ag] if pvhh_i is not None else None,
        min_pairs=5,
        n_bootstrap=600,
    )
    vhh_glb_mask = ~vhh_same_ag
    vhh_glb_res = (
        compute_3state_metrics(
            vhh_sp.labels[vhh_glb_mask],
            vhh_bdiff[vhh_glb_mask],
            pvhh_d[vhh_glb_mask],
            pvhh_i[vhh_glb_mask] if pvhh_i is not None else None,
        )
        if vhh_glb_mask.any()
        else None
    )
    vhh_res = {
        **vhh_within_res,
        "within_ag": vhh_within_res,
        "global": vhh_glb_res,
        "pooled_all_including_global": compute_3state_metrics(vhh_sp.labels, vhh_bdiff, pvhh_d, pvhh_i),
    }

    # Unfiltered full-panel evaluation on test_pointwise.tsv
    complex_scores = tournament_scores_from_pairs(pnl_pairs_df, ppnl_d)
    panel_res = evaluate_pointwise_spearman(pw_test_df, complex_scores, min_group_size=6)

    macro_3state_local = float(
        np.mean([loc_res["auc_dir"], loc_res["auc_impact"], loc_res["auc_ordinal"]])
    )
    macro_3track_dir = float(
        np.mean([loc_res["auc_dir"], cand_full["auc_dir"], glb_res["auc_dir"]])
    )

    return {
        "name": name,
        "family": family,
        "test_local_l1_l3": {
            **loc_res,
            "ab_mut_auc_dir": abm_res["auc_dir"],
            "ab_mut_auc_impact": abm_res["auc_impact"],
            "ab_mut_auc_ordinal": abm_res["auc_ordinal"],
            "ab_mut_pooled_auc_dir": abm_res["pooled_auc_dir"],
            "ab_mut_pooled_auc_impact": abm_res["pooled_auc_impact"],
            "ab_mut_pooled_auc_ordinal": abm_res["pooled_auc_ordinal"],
            "ag_mut_auc_dir": agm_res["auc_dir"],
            "ag_mut_auc_impact": agm_res["auc_impact"],
            "ag_mut_auc_ordinal": agm_res["auc_ordinal"],
            "macro_3state_auc": round(macro_3state_local, 4),
        },
        "test_candidates": {
            "n_pairs": cand_full["n_pairs"],
            "n_impactful": cand_full["n_impactful"],
            "n_neutral": cand_full["n_neutral"],
            "n_antigens_evaluated": cand_full["n_antigens_evaluated"],
            "auc_dir": cand_full["auc_dir"],
            "auc_dir_ci95": cand_full.get("auc_dir_ci95"),
            "auc_impact": cand_full["auc_impact"],
            "auc_impact_ci95": cand_full.get("auc_impact_ci95"),
            "auc_ordinal": cand_full["auc_ordinal"],
            "auc_ordinal_ci95": cand_full.get("auc_ordinal_ci95"),
            "pooled_auc_dir": cand_full["pooled_auc_dir"],
            "pooled_auc_impact": cand_full["pooled_auc_impact"],
            "pooled_auc_ordinal": cand_full["pooled_auc_ordinal"],
            "per_antigen": cand_full["per_antigen"],
            "by_group": cand_full["by_group"],
        },
        "test_global": glb_res,
        "test_panels_unfiltered": panel_res,
        "test_overall": test_full,
        "test_macro_3track_dir_auc": round(macro_3track_dir, 4),
        "val_local_l1_l3": v_loc_res,
        "cross_modal_unseen_ag": {
            **cm_res,
            "by_modality": cm_by_mod,
        },
        "vhh_zeroshot_transfer": vhh_res,
        "test_cocrystal_l1_l3": cocrys_loc_res,
    }


def make_comparison_figure(results: list[dict], out_path: Path) -> None:
    """Render publication-quality 4-panel horizontal bar chart across 3-state + panel metrics."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [r["name"] for r in results]
    n = len(names)
    y_pos = np.arange(n)[::-1]

    family_colors = {
        "Naive / Classical": "#7f8c8d",
        "Zero-Shot PLM": "#2980b9",
        "Zero-Shot 3D / Multi-Chain": "#16a085",
        "Impact-Blind Binary Probe": "#e67e22",
        "3-State Hurdle / Neural": "#27ae60",
        "3D Structure / Multi-Chain 3-State": "#8e44ad",
    }
    colors = [family_colors.get(r["family"], "#34495e") for r in results]

    fig, axes = plt.subplots(1, 4, figsize=(21.0, max(9.0, 0.42 * n)), sharey=True)
    panels = [
        ("A. Local L1–L3 Sign AUC\n(+1 vs -1, Impactful)", "dir", (0.22, 1.05), 0.50),
        ("B. Local L1–L3 Impact AUC\n(±1 vs 0, Effect Detection)", "imp", (0.22, 1.05), 0.50),
        ("C. Local L1–L3 3-Class Ordinal\n(-1 < 0 < +1 Concordance)", "ord", (0.22, 1.05), 0.50),
        ("D. Unfiltered Variant Panels\n(Top-Binder AUC, All Variants)", "top", (0.22, 1.05), 0.50),
    ]

    for ax, (title, mtype, xlim, ref_line) in zip(axes, panels):
        if mtype == "dir":
            vals = [r["test_local_l1_l3"]["auc_dir"] for r in results]
            cis = [r["test_local_l1_l3"].get("auc_dir_ci95") for r in results]
        elif mtype == "imp":
            vals = [r["test_local_l1_l3"]["auc_impact"] for r in results]
            cis = [r["test_local_l1_l3"].get("auc_impact_ci95") for r in results]
        elif mtype == "ord":
            vals = [r["test_local_l1_l3"]["auc_ordinal"] for r in results]
            cis = [r["test_local_l1_l3"].get("auc_ordinal_ci95") for r in results]
        else:
            vals = [r["test_panels_unfiltered"]["mean_top_binder_auc"] or 0.5 for r in results]
            cis = [None for _ in results]

        err_lo = [max(0.0, v - ci[0]) if ci else 0.0 for v, ci in zip(vals, cis)]
        err_hi = [max(0.0, ci[1] - v) if ci else 0.0 for v, ci in zip(vals, cis)]

        ax.axvline(ref_line, color="#c0392b", linestyle="--", linewidth=1.25, zorder=1)
        ax.barh(
            y_pos,
            vals,
            xerr=[err_lo, err_hi] if mtype != "top" else None,
            color=colors,
            alpha=0.88,
            height=0.68,
            capsize=2.2,
            edgecolor="#2c3e50",
            linewidth=0.6,
            zorder=2,
        )
        for yp, v, hi in zip(y_pos, vals, err_hi):
            x_txt = max(v + hi + 0.012, ref_line + 0.012)
            ax.text(x_txt, yp, f"{v:.3f}", va="center", ha="left", fontsize=8.2, fontweight="bold", color="#1a252f")
        ax.set_xlim(*xlim)
        ax.set_xlabel("Symmetrized AUC (95% CI)" if mtype != "top" else "Mean Panel Top-Binder ROC-AUC", fontsize=9.8)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names, fontsize=9.2)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, edgecolor="#2c3e50", label=fam) for fam, c in family_colors.items()]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=True, fontsize=9.5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "AbRank v2 — Unseen-Antigen (unseen_ag) 3-State Mutation Effect & Full-Panel Ranking Baselines",
        fontsize=13.2,
        fontweight="bold",
        y=1.06,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    results: list[dict],
    masked_check: dict,
    test_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    cm_df: pd.DataFrame,
    out_md: Path,
) -> None:
    """Write standalone `results/UNSEEN_ANTIGEN_BASELINES.md` report."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    by_dir = sorted(results, key=lambda r: r["test_local_l1_l3"]["auc_dir"], reverse=True)
    by_ord = sorted(results, key=lambda r: r["test_local_l1_l3"]["auc_ordinal"], reverse=True)
    by_imp = sorted(results, key=lambda r: r["test_local_l1_l3"]["auc_impact"], reverse=True)
    by_top = sorted(results, key=lambda r: r["test_panels_unfiltered"]["mean_top_binder_auc"] or 0.0, reverse=True)
    by_rho = sorted(results, key=lambda r: r["test_panels_unfiltered"]["mean_spearman_rho"] or -1.0, reverse=True)
    by_cnd = sorted(results, key=lambda r: r["test_candidates"]["auc_ordinal"], reverse=True)
    best_ord, sec_ord = by_ord[0], by_ord[1]
    best_imp, sec_imp = by_imp[0], by_imp[1]
    best_dir, sec_dir = by_dir[0], by_dir[1]
    best_top, sec_top = by_top[0], by_top[1]
    best_rho = by_rho[0]
    best_cnd = by_cnd[0]
    best_cm = max(results, key=lambda r: r["cross_modal_unseen_ag"]["auc_ordinal"])
    best_vhh = max(results, key=lambda r: r["vhh_zeroshot_transfer"]["auc_ordinal"])

    n_loc_tot = results[0]["test_local_l1_l3"]["n_pairs"]
    n_loc_imp = results[0]["test_local_l1_l3"]["n_impactful"]
    n_loc_neu = results[0]["test_local_l1_l3"]["n_neutral"]
    n_loc_ag = results[0]["test_local_l1_l3"]["n_antigens_evaluated"]
    n_cnd_ag = results[0]["test_candidates"]["n_antigens_evaluated"]
    n_cm_ag = results[0]["cross_modal_unseen_ag"]["n_antigens_evaluated"]
    n_vhh_ag = results[0]["vhh_zeroshot_transfer"]["n_antigens_evaluated"]
    n_glb = results[0]["test_global"]["n_pairs"]
    n_vhh = results[0]["vhh_zeroshot_transfer"]["n_pairs"]

    lines = [
        "# AbRank v2 — Unseen-Antigen (`unseen_ag`) 3-State & Full-Panel Baseline Results (29 Models)",
        "",
        "## Executive Summary",
        "",
        "AbRank v2 evaluates **29 sequence, multi-chain, 3D inverse-folding, biophysical energy, normal-mode, and 3D geometric GNN baselines** on unseen homologous antigen clusters (`<50%` sequence identity + zero subdomain overlap) and unseen antibody lineages (`<80%` paired Fv / VHH identity). **All within-antigen pairwise AUCs (`Sign`, `Impact`, `Ordinal`, `Hit`) and continuous panel metrics (`Spearman ρ`, `Top-Binder AUC`) are Per-Antigen Macro-Averaged** (each held-out antigen or `(antigen, modality)` stratum with `≥5` pairs is scored independently and averaged with equal weight across antigens so mega-panels cannot dominate; exact Pooled Micro-AUCs are also preserved in JSON and Section 5 below):",
        "",
        f"- **Best Per-Antigen Macro 3-Class Ordinal Local Mutation Ranker (`Depleted -1 < Neutral 0 < Improved +1`, `{n_loc_ag}` held-out antigens with `≥5` pairs, `n = {n_loc_tot}` total pairs)**: **`{best_ord['name']}`** leads overall (**`{best_ord['test_local_l1_l3']['auc_ordinal']:.3f}` Macro Ordinal AUC**, **`{best_ord['test_local_l1_l3']['auc_dir']:.3f}` Sign AUC**, **`{best_ord['test_local_l1_l3']['auc_impact']:.3f}` Impact AUC**; Pooled Micro Ordinal: `{best_ord['test_local_l1_l3']['pooled_auc_ordinal']:.3f}`), followed by **`{sec_ord['name']}`** (**`{sec_ord['test_local_l1_l3']['auc_ordinal']:.3f}` Macro Ordinal**, `{sec_ord['test_local_l1_l3']['auc_dir']:.3f}` Sign, `{sec_ord['test_local_l1_l3']['auc_impact']:.3f}` Impact).",
        f"- **Best Per-Antigen Mutation Impact / Hot-Spot Detector (`|label| = 1` vs `label = 0`)**: **`{best_imp['name']}`** reaches **`{best_imp['test_local_l1_l3']['auc_impact']:.3f}` Macro Impact AUC** (`95% CI: [{best_imp['test_local_l1_l3']['auc_impact_ci95'][0]:.3f}, {best_imp['test_local_l1_l3']['auc_impact_ci95'][1]:.3f}]`), ahead of **`{sec_imp['name']}`** (`{sec_imp['test_local_l1_l3']['auc_impact']:.3f}`).",
        f"- **Best Per-Antigen Directional Sign Rankers (`+1` vs `-1` on $\\ge 10\\times$ mutations)**: **`{best_dir['name']}`** (`{best_dir['test_local_l1_l3']['auc_dir']:.3f}`) and **`{sec_dir['name']}`** (`{sec_dir['test_local_l1_l3']['auc_dir']:.3f}`).",
        f"- **Unfiltered Continuous Variant Panels (`test_pointwise.tsv`, all `320` panel variants across `{best_top['test_panels_unfiltered']['n_panels']}` held-out panels)**: **`{best_top['name']}`** (`{best_top['test_panels_unfiltered']['mean_top_binder_auc']:.3f}` mean Top-Binder AUC) and **`{sec_top['name']}`** (`{sec_top['test_panels_unfiltered']['mean_top_binder_auc']:.3f}`), with **`{best_rho['name']}`** (`{best_rho['test_panels_unfiltered']['mean_spearman_rho']:+.3f}`) leading continuous Spearman $\\rho$.",
        f"- **Same-Antigen `Candidate` Ranking (`d_Lev > 3`, `{n_cnd_ag}` antigens, `n = {len(cand_df)}`) & Zero-Shot `VHH` Nanobody Transfer (`{n_vhh_ag}` within-Ag clusters, `n = {n_vhh}`)**: **`{best_cnd['name']}`** reaches **`{best_cnd['test_candidates']['auc_ordinal']:.3f}` Macro Ordinal AUC** on `Candidate` pairs, and **`{best_vhh['name']}`** reaches **`{best_vhh['vhh_zeroshot_transfer']['auc_ordinal']:.3f}` Macro Ordinal AUC** (`Sign`: `{best_vhh['vhh_zeroshot_transfer']['auc_dir']:.3f}`, `Impact`: `{best_vhh['vhh_zeroshot_transfer']['auc_impact']:.3f}`) zero-shot on single-domain VHH nanobodies.",
        f"- **Zero-Shot Transfer to Strict Unseen-Antigen Cross-Modal Assays (`{n_cm_ag}` `(Ag, Modality)` strata with `≥5` pairs, `n = {len(cm_df):,}` pairs)**: **`{best_cm['name']}`** achieves **`{best_cm['cross_modal_unseen_ag']['auc_ordinal']:.3f}` Macro Ordinal AUC** (`Sign`: `{best_cm['cross_modal_unseen_ag']['auc_dir']:.3f}`, `Impact`: `{best_cm['cross_modal_unseen_ag']['auc_impact']:.3f}`).",
        "",
        "![Unseen Antigen 3-State Baseline Comparison](figures/unseen_ag_baselines_auc.png)",
        "",
        "---",
        "",
        "## 1. Per-Antigen Macro-Averaged 3-State Evaluation Protocol",
        "",
        "All supervised models are trained **exclusively** on `data/benchmark/splits/unseen_ag/train.tsv` (`21,836` pairs: `16,448` impactful `$\\pm 1$` with $|\\Delta\\text{p}K_d| \\ge 1.0$ and `5,388` neutral `0` with $|\\Delta\\text{p}K_d| \\le 0.30`), with feature scalers fit strictly on `train` and hyperparameters selected strictly on `unseen_ag/val.tsv` via **equal-weight Per-Antigen Macro-AUC** across all 6 validation antigens. At test time, every within-antigen track scores each held-out antigen (or `(Ag_group, Assay_modality)` stratum) with `≥5` evaluable pairs independently and averages with equal weight across antigens:",
        "",
        f"1. **3-State Local Mutations (`L1–L3`, `{n_loc_ag}` antigens with `≥5` pairs; `{n_loc_tot}` total pairs = `{n_loc_imp}` impactful `$\\pm 1$` + `{n_loc_neu}` neutral `0`)**: Evaluates `Sign AUC` (`+1` vs `-1`), `Impact AUC` (`±1` vs `0`), `3-Class Ordinal AUC` (`-1 < 0 < +1`), and `Selective Hit AUC` (`+1` vs `{0, -1}`).",
        f"2. **3-State Same-Antigen Candidates (`Lev > 3`, `{n_cnd_ag}` antigens, `{len(cand_df)}` pairs)**: Distinct antibody candidates competing for the same held-out antigen (`TSLP`, `LOX1`, `PDL1`, `IL23R`, `DKK1`).",
        "3. **Unfiltered Variant Panels (`test_pointwise.tsv`)**: Per-panel continuous **Spearman $\\rho$** and **Top-Binder ROC-AUC** macro-averaged across held-out panels.",
        f"4. **Cross-Antigen `Global` Pairs (`{n_glb}` test pairs)**: Cross-target $K_d$ scale ranking across distinct unseen antigens (`g_1 != g_2`).",
        f"5. **Strict Unseen-Antigen `Cross-Modal` (`{n_cm_ag}` strata, `{len(cm_df):,}` pairs) & Within-Antigen `VHH` (`{n_vhh_ag}` antigens, `{n_vhh}` pairs) Transfer**: Filtered strictly to `<50%` antigen identity and `<80%` antibody/VHH identity relative to `unseen_ag/train`.",
        "",
        "---",
        "",
        "## 2. Main Per-Antigen Macro 3-State Local Mutation & Unfiltered Panel Results",
        "",
        f"| Baseline Model | Family | `L1–L3` Macro Sign AUC (`{n_loc_ag}` Ags) | `L1–L3` Macro Impact AUC | `L1–L3` Macro 3-Class Ordinal | `L1–L3` Macro Hit AUC | Unfiltered Panel Spearman `$\\rho$` | Panel Top-Binder AUC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        loc = r["test_local_l1_l3"]
        pnl = r["test_panels_unfiltered"]
        lines.append(
            f"| **{r['name']}** | {r['family']} | "
            f"`{loc['auc_dir']:.3f}` [{loc['auc_dir_ci95'][0]:.2f}, {loc['auc_dir_ci95'][1]:.2f}] | "
            f"`{loc['auc_impact']:.3f}` [{loc['auc_impact_ci95'][0]:.2f}, {loc['auc_impact_ci95'][1]:.2f}] | "
            f"**`{loc['auc_ordinal']:.3f}`** [{loc['auc_ordinal_ci95'][0]:.2f}, {loc['auc_ordinal_ci95'][1]:.2f}] | "
            f"`{loc['auc_selective_hit']:.3f}` | "
            f"`{pnl['mean_spearman_rho']:.3f}` | "
            f"`{pnl['mean_top_binder_auc']:.3f}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Multi-Regime Generalization (`Candidate`, `Global`, `Cross-Modal`, and `Unseen-Ag VHH` — Per-Antigen Macro)",
        "",
        "| Baseline Model | `Candidate` Macro Sign / Imp / Ord (`" + f"{n_cnd_ag}" + "` Ags) | `Global` Cross-Ag AUC (`n=" + f"{n_glb}" + "`) | `Cross-Modal` Macro Sign / Imp / Ord (`" + f"{n_cm_ag}" + "` Strata) | `Unseen-Ag VHH` Macro Sign / Imp / Ord (`" + f"{n_vhh_ag}" + "` Ags) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for r in results:
        cnd = r["test_candidates"]
        glb = r["test_global"]
        cm = r["cross_modal_unseen_ag"]
        vhh = r["vhh_zeroshot_transfer"]
        lines.append(
            f"| **{r['name']}** | "
            f"`{cnd['auc_dir']:.3f}` / `{cnd['auc_impact']:.3f}` / **`{cnd['auc_ordinal']:.3f}`** | "
            f"`{glb['auc_dir']:.3f}` | "
            f"`{cm['auc_dir']:.3f}` / `{cm['auc_impact']:.3f}` / **`{cm['auc_ordinal']:.3f}`** | "
            f"`{vhh['auc_dir']:.3f}` / `{vhh['auc_impact']:.3f}` / **`{vhh['auc_ordinal']:.3f}`** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Antibody-Mutated (`Ab-Mut`) vs. Antigen-Mutated (`Ag-Mut`) Per-Antigen Macro Slice Analysis",
        "",
        "| Model | `Ab-Mut` Macro Sign / Impact / Ordinal | `Ag-Mut` Macro Sign / Impact / Ordinal | Exact `<mask>`-Marginal Macro Sign / Impact |",
        "| :--- | :---: | :---: | :---: |",
    ])

    for r in results:
        loc = r["test_local_l1_l3"]
        mm_str = "—"
        if r["name"] in masked_check:
            mm_d, mm_i = masked_check[r["name"]]
            mm_str = f"`{mm_d:.3f}` / `{mm_i:.3f}`"
        ag_imp_str = f"{loc['ag_mut_auc_impact']:.3f}" if loc["ag_mut_auc_impact"] is not None else "—"
        lines.append(
            f"| **{r['name']}** | "
            f"`{loc['ab_mut_auc_dir']:.3f}` / `{loc['ab_mut_auc_impact']:.3f}` / `{loc['ab_mut_auc_ordinal']:.3f}` | "
            f"`{loc['ag_mut_auc_dir']:.3f}` / `{ag_imp_str}` / `{loc['ag_mut_auc_ordinal']:.3f}` | "
            f"{mm_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Per-Antigen Macro-AUC (`≥5` Pairs) vs. Pooled Micro-AUC Comparison",
        "",
        "To ensure mega-panels (`TSLP`: `82.9%` of `unseen_ag/test L1–L3`; `NbBench Galectin-3`: `48.3%` of `Cross-Modal`; `SARSCoV2_Spike`: `78.1%` of `VHH`) do not dominate benchmark rankings, primary metrics use **Per-Antigen Macro-AUC**, while exact **Pooled Micro-AUC** is reported side-by-side below:",
        "",
        "| Model | `L1–L3` Macro Ord (`Sign`/`Imp`) | `L1–L3` Pooled Ord (`Sign`/`Imp`) | `Candidate` Macro / Pooled Ord | `Cross-Modal` Macro / Pooled Ord | `VHH` Macro / Pooled Ord |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in results:
        loc = r["test_local_l1_l3"]
        cnd = r["test_candidates"]
        cm = r["cross_modal_unseen_ag"]
        vhh = r["vhh_zeroshot_transfer"]
        lines.append(
            f"| **{r['name']}** | "
            f"**`{loc['auc_ordinal']:.3f}`** (`{loc['auc_dir']:.3f}`/`{loc['auc_impact']:.3f}`) | "
            f"`{loc['pooled_auc_ordinal']:.3f}` (`{loc['pooled_auc_dir']:.3f}`/`{loc['pooled_auc_impact']:.3f}`) | "
            f"**`{cnd['auc_ordinal']:.3f}`** / `{cnd['pooled_auc_ordinal']:.3f}` | "
            f"**`{cm['auc_ordinal']:.3f}`** / `{cm['pooled_auc_ordinal']:.3f}` | "
            f"**`{vhh['auc_ordinal']:.3f}`** / `{vhh['pooled_auc_ordinal']:.3f}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "# 1. Extract cached 1D PLM embeddings (`ESMC-300M`, `AbLang2-Paired`, `ESM2-150M`)",
        "python scripts/05_extract_embeddings.py --workers 16",
        "",
        "# 2. Resolve 851 3D scaffolds (RCSB co-crystals + IgFold Fv/VHH) and extract 3D / multi-chain features",
        "python scripts/08_prepare_3d_structures.py",
        "python scripts/09_extract_structure_and_multichain_features.py",
        "",
        "# 3. Train & evaluate all 29 unseen_ag baselines and 5-track 3D DMS/ΔΔG benchmarks",
        "python scripts/06_run_unseen_ag_baselines.py",
        "python scripts/10_run_3d_dms_ddg_benchmarks.py",
        "```",
        "",
    ])

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate AbRank v2 3-state unseen_ag baselines.")
    parser.add_argument("--report-only", action="store_true", help="Regenerate plot & Markdown from saved JSON")
    args = parser.parse_args()

    t0 = time.time()
    root = get_repo_root()
    bdir = root / "data" / "benchmark"
    emb_dir = root / "data" / "embeddings"
    res_dir = root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_json = res_dir / "unseen_ag_baselines.json"

    test_df = pd.read_csv(bdir / "splits" / "unseen_ag" / "test.tsv", sep="\t", low_memory=False)
    cand_df = pd.read_csv(bdir / "splits" / "unseen_ag" / "test_candidates.tsv", sep="\t", low_memory=False)

    if args.report_only and out_json.exists():
        saved = json.loads(out_json.read_text(encoding="utf-8"))
        results = saved["models"]
        masked_check = saved["exact_masked_marginal_test_l1_l3"]
        n_cm = results[0]["cross_modal_unseen_ag"]["n_pairs"]
        make_comparison_figure(results, res_dir / "figures" / "unseen_ag_baselines_auc.png")
        write_markdown_report(results, masked_check, test_df, cand_df, pd.DataFrame(index=range(n_cm)), res_dir / "UNSEEN_ANTIGEN_BASELINES.md")
        print("Regenerated figure and UNSEEN_ANTIGEN_BASELINES.md from JSON.")
        return

    print("Loading benchmark splits and master assay tables...")
    kd_lookup = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False).drop_duplicates("AbAgID").set_index("AbAgID")
    tr_df = pd.read_csv(bdir / "splits" / "unseen_ag" / "train.tsv", sep="\t", low_memory=False)
    val_df = pd.read_csv(bdir / "splits" / "unseen_ag" / "val.tsv", sep="\t", low_memory=False)
    pw_test_df = pd.read_csv(bdir / "splits" / "unseen_ag" / "test_pointwise.tsv", sep="\t", low_memory=False)
    pnl_pairs_df = build_panel_tournament_pairs(pw_test_df, min_group_size=6)

    cm_df, cma_lookup = build_strict_unseen_cross_modal(root, tr_df, kd_lookup)
    vhh_df, vhha_lookup = build_strict_unseen_vhh(root, tr_df, kd_lookup)
    print(
        f"Loaded 3-state splits: train={len(tr_df)}, val={len(val_df)}, test={len(test_df)}, "
        f"cand={len(cand_df)}, panel_pairs={len(pnl_pairs_df)}, cross_modal_unseen={len(cm_df)}, vhh_unseen={len(vhh_df)}"
    )

    print("Loading cached PLM embeddings (`esmc_300m.pt`, `ablang2_paired.pt`, `esm2_150m.pt`)...")
    esmc_cache = torch.load(emb_dir / "esmc_300m.pt", map_location="cpu", weights_only=False)
    ablang_cache = torch.load(emb_dir / "ablang2_paired.pt", map_location="cpu", weights_only=False)
    esm2_cache = torch.load(emb_dir / "esm2_150m.pt", map_location="cpu", weights_only=False)

    print("Assembling 3-state pair tensors...")
    tr_sp = assemble_split_tensors(tr_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)
    val_sp = assemble_split_tensors(val_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)
    test_sp = assemble_split_tensors(test_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)
    cand_sp = assemble_split_tensors(cand_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)
    cm_sp = assemble_split_tensors(cm_df, cma_lookup, esmc_cache, ablang_cache, esm2_cache)
    vhh_sp = assemble_split_tensors(vhh_df, vhha_lookup, esmc_cache, ablang_cache, esm2_cache)
    pnl_sp = assemble_split_tensors(pnl_pairs_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)

    w_tr = compute_stratum_weights(tr_df, tr_sp.is_ag_mut)

    print("Computing exact <mask>-marginal zero-shot check on unseen_ag/test L1-L3...")
    test_local_df = test_df[test_df["Category"].isin(["L1", "L2", "L3"])].reset_index(drop=True)
    mm_cache_path = emb_dir / "masked_marginal_test_l1_l3.json"
    import hashlib
    split_sig = hashlib.sha1("|".join(test_local_df["pair_id"].astype(str)).encode()).hexdigest()
    masked_check = None
    if mm_cache_path.exists():
        cached_mm = json.loads(mm_cache_path.read_text(encoding="utf-8"))
        if cached_mm.get("sig") == split_sig and cached_mm.get("agg") == "per_antigen_macro":
            masked_check = cached_mm["metrics"]

    if masked_check is None:
        mm_esmc_ab, mm_esmc_ag = compute_esmc_masked_marginals(test_local_df, kd_lookup)
        mm_ablang = compute_ablang2_masked_marginals(test_local_df, kd_lookup)
        masked_check = {}
        for mk_name, mk_vec in (
            ("AbLang2-Paired (Zero-Shot)", mm_ablang),
            ("ESMC-300M (Zero-Shot)", mm_esmc_ab + mm_esmc_ag),
            ("ESMC+AbLang2 (Zero-Shot)", 0.5 * mm_esmc_ab + 0.5 * mm_ablang + mm_esmc_ag),
        ):
            m3 = compute_per_antigen_macro_metrics(test_local_df, mk_vec, np.abs(mk_vec), min_pairs=5)
            masked_check[mk_name] = [m3["auc_dir"], m3["auc_impact"]]
        mm_cache_path.write_text(
            json.dumps({"sig": split_sig, "agg": "per_antigen_macro", "metrics": masked_check}, indent=2),
            encoding="utf-8",
        )
    print("Exact masked-marginal check on test L1-L3 (Sign, Impact):", masked_check)

    results: list[dict] = []
    struct_cache_file = root / "data" / "structures" / "complex_cache.pt"
    inv_cache_file = emb_dir / "inverse_folding_scaffolds.pt"
    saprot_cache_file = emb_dir / "saprot_650m.pt"
    mint_cache_file = emb_dir / "mint_650m.pt"

    cocrys_test_mask: np.ndarray | None = None
    if struct_cache_file.exists():
        _cc_tmp = torch.load(struct_cache_file, map_location="cpu", weights_only=False)
        _a2s = _cc_tmp["abag_to_scaffold"]
        _scafs = _cc_tmp["scaffolds"]
        cocrys_test_mask = np.array(
            [
                bool(_scafs[_a2s[a]]["is_cocrystal"] and _scafs[_a2s[b]]["is_cocrystal"])
                for a, b in zip(test_df["AbAgID_1"], test_df["AbAgID_2"])
            ],
            dtype=bool,
        )

    def record(
        name: str,
        fam: str,
        pv_d: np.ndarray,
        pt_d: np.ndarray,
        pc_d: np.ndarray,
        pcm_d: np.ndarray,
        ppnl_d: np.ndarray,
        pvhh_d: np.ndarray,
        *,
        pv_i: np.ndarray | None = None,
        pt_i: np.ndarray | None = None,
        pc_i: np.ndarray | None = None,
        pcm_i: np.ndarray | None = None,
        pvhh_i: np.ndarray | None = None,
    ) -> None:
        r = detailed_eval(
            name, fam,
            pv_d, pv_i, pt_d, pt_i, pc_d, pc_i, pcm_d, pcm_i, ppnl_d, pvhh_d, pvhh_i,
            val_df, test_df, cand_df, cm_df, vhh_df, pw_test_df, pnl_pairs_df,
            val_sp, test_sp, cand_sp, cm_sp, vhh_sp,
            cocrys_test_mask=cocrys_test_mask,
        )
        results.append(r)
        loc = r["test_local_l1_l3"]
        pnl = r["test_panels_unfiltered"]
        print(
            f"  {name:36s} | L1-3 Sign={loc['auc_dir']:.3f} Imp={loc['auc_impact']:.3f} Ord={loc['auc_ordinal']:.3f} "
            f"| Panel rho={pnl['mean_spearman_rho']:.3f} | CM Ord={r['cross_modal_unseen_ag']['auc_ordinal']:.3f} "
            f"| VHH Ord={r['vhh_zeroshot_transfer']['auc_ordinal']:.3f}"
        )

    # ------------------------------------------------------------------
    # Family A: Naive & Handcrafted Sequence Baselines
    # ------------------------------------------------------------------
    print("\n--- Running Family A: Naive & Classical Sequence Baselines ---")
    record(
        "Random Baseline",
        "Naive / Classical",
        np.zeros(len(val_df)),
        np.zeros(len(test_df)),
        np.zeros(len(cand_df)),
        np.zeros(len(cm_df)),
        np.zeros(len(pnl_pairs_df)),
        np.zeros(len(vhh_df)),
    )

    used_kd_ids = list(
        set(tr_df["AbAgID_1"]).union(tr_df["AbAgID_2"])
        .union(val_df["AbAgID_1"]).union(val_df["AbAgID_2"])
        .union(test_df["AbAgID_1"]).union(test_df["AbAgID_2"])
        .union(cand_df["AbAgID_1"]).union(cand_df["AbAgID_2"])
        .union(pnl_pairs_df["AbAgID_1"]).union(pnl_pairs_df["AbAgID_2"])
    )
    used_cm_ids = list(set(cm_df["AbAgID_1"]).union(cm_df["AbAgID_2"]))
    used_vhh_ids = list(set(vhh_df["AbAgID_1"]).union(vhh_df["AbAgID_2"]))
    combined_assays = pd.concat(
        [
            kd_lookup.loc[used_kd_ids].reset_index(),
            cma_lookup.loc[used_cm_ids].reset_index(),
            vhha_lookup.loc[used_vhh_ids].reset_index(),
        ],
        ignore_index=True,
    ).drop_duplicates("AbAgID")

    for label, use_dipep in (("Biophys+AAC (Linear)", False), ("k-mer Dipeptide (Ridge)", True)):
        feat_map = build_complex_handcrafted_table(combined_assays, include_dipeptides=use_dipep)

        def _mat(df: pd.DataFrame) -> np.ndarray:
            x1 = np.stack([feat_map[k] for k in df["AbAgID_1"]], axis=0)
            x2 = np.stack([feat_map[k] for k in df["AbAgID_2"]], axis=0)
            return build_odd_linear_matrix([x1], [x2], include_unit_direction=True)

        mdl = AntisymmetricLinearRanker(use_neutral_nullspace=True).fit(
            _mat(tr_df), tr_sp.y, w_tr, _mat(val_df), val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels
        )
        record(
            label, "Naive / Classical",
            mdl.predict(_mat(val_df)), mdl.predict(_mat(test_df)),
            mdl.predict(_mat(cand_df)), mdl.predict(_mat(cm_df)), mdl.predict(_mat(pnl_pairs_df)),
            mdl.predict(_mat(vhh_df)),
        )

    # ------------------------------------------------------------------
    # Family B: Zero-Shot Pretrained Language Models (0 Supervised Params)
    # ------------------------------------------------------------------
    print("\n--- Running Family B: Zero-Shot PLM Baselines ---")
    for zs_key, zs_label in (
        ("esmc_cosine", "ESMC-300M Cosine (Zero-Shot)"),
        ("esm2_150m_zeroshot", "ESM2-150M (Zero-Shot)"),
        ("ablang2_zeroshot", "AbLang2-Paired (Zero-Shot)"),
        ("esmc_300m_zeroshot", "ESMC-300M (Zero-Shot)"),
        ("esmc_ablang2_zeroshot", "ESMC+AbLang2 (Zero-Shot)"),
    ):
        record(
            zs_label,
            "Zero-Shot PLM",
            val_sp.zs_dict[zs_key],
            test_sp.zs_dict[zs_key],
            cand_sp.zs_dict[zs_key],
            cm_sp.zs_dict[zs_key],
            pnl_sp.zs_dict[zs_key],
            vhh_sp.zs_dict[zs_key],
            pv_i=val_sp.zs_impact_dict[zs_key],
            pt_i=test_sp.zs_impact_dict[zs_key],
            pc_i=cand_sp.zs_impact_dict[zs_key],
            pcm_i=cm_sp.zs_impact_dict[zs_key],
            pvhh_i=vhh_sp.zs_impact_dict[zs_key],
        )

    # ------------------------------------------------------------------
    # Family C: Impact-Blind Binary Probes (Trained on ±1 Only)
    # ------------------------------------------------------------------
    print("\n--- Running Family C: Impact-Blind Binary Probes (±1 Only) ---")
    def _esm2_mat(sp: SplitPairTensors) -> np.ndarray:
        return build_odd_linear_matrix([sp.esm2_ab_1, sp.esm2_ag_1], [sp.esm2_ab_2, sp.esm2_ag_2], include_unit_direction=True)

    m_esm2 = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(
        _esm2_mat(tr_sp), tr_sp.y, w_tr, _esm2_mat(val_sp), val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels
    )
    record("ESM2-150M (Binary Probe)", "Impact-Blind Binary Probe", m_esm2.predict(_esm2_mat(val_sp)), m_esm2.predict(_esm2_mat(test_sp)), m_esm2.predict(_esm2_mat(cand_sp)), m_esm2.predict(_esm2_mat(cm_sp)), m_esm2.predict(_esm2_mat(pnl_sp)), m_esm2.predict(_esm2_mat(vhh_sp)))

    def _abl_mat(sp: SplitPairTensors) -> np.ndarray:
        return build_odd_linear_matrix([sp.ablang_1], [sp.ablang_2], include_unit_direction=True)

    m_abl = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(
        _abl_mat(tr_sp), tr_sp.y, w_tr, _abl_mat(val_sp), val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels
    )
    record("AbLang2-Paired (Binary Probe)", "Impact-Blind Binary Probe", m_abl.predict(_abl_mat(val_sp)), m_abl.predict(_abl_mat(test_sp)), m_abl.predict(_abl_mat(cand_sp)), m_abl.predict(_abl_mat(cm_sp)), m_abl.predict(_abl_mat(pnl_sp)), m_abl.predict(_abl_mat(vhh_sp)))

    def _esmc_raw(sp: SplitPairTensors) -> np.ndarray:
        return build_odd_linear_matrix([sp.esmc_ab_1, sp.esmc_ag_1], [sp.esmc_ab_2, sp.esmc_ag_2], include_unit_direction=False)

    m_esmc_raw = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(
        _esmc_raw(tr_sp), tr_sp.y, None, _esmc_raw(val_sp), val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels
    )
    record("ESMC-300M (Raw-Diff Binary Probe)", "Impact-Blind Binary Probe", m_esmc_raw.predict(_esmc_raw(val_sp)), m_esmc_raw.predict(_esmc_raw(test_sp)), m_esmc_raw.predict(_esmc_raw(cand_sp)), m_esmc_raw.predict(_esmc_raw(cm_sp)), m_esmc_raw.predict(_esmc_raw(pnl_sp)), m_esmc_raw.predict(_esmc_raw(vhh_sp)))

    def _esmc_ms(sp: SplitPairTensors) -> np.ndarray:
        return build_odd_linear_matrix([sp.esmc_ab_1, sp.esmc_ag_1], [sp.esmc_ab_2, sp.esmc_ag_2], include_unit_direction=True)

    m_esmc_bin = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(
        _esmc_ms(tr_sp), tr_sp.y, w_tr, _esmc_ms(val_sp), val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels
    )
    record("ESMC-300M (Multi-Scale Binary)", "Impact-Blind Binary Probe", m_esmc_bin.predict(_esmc_ms(val_sp)), m_esmc_bin.predict(_esmc_ms(test_sp)), m_esmc_bin.predict(_esmc_ms(cand_sp)), m_esmc_bin.predict(_esmc_ms(cm_sp)), m_esmc_bin.predict(_esmc_ms(pnl_sp)), m_esmc_bin.predict(_esmc_ms(vhh_sp)))

    def _hybrid_lin(sp: SplitPairTensors) -> np.ndarray:
        return build_odd_linear_matrix(
            [sp.esmc_ab_1, sp.ablang_1, sp.esmc_ag_1],
            [sp.esmc_ab_2, sp.ablang_2, sp.esmc_ag_2],
            extra_odd=np.concatenate([sp.zs_features, sp.biophys_odd], axis=1),
            include_unit_direction=True,
        )

    def _hybrid_glb(sp: SplitPairTensors) -> np.ndarray:
        return build_odd_linear_matrix(
            [sp.esmc_ab_1, sp.ablang_1, sp.esmc_ag_1],
            [sp.esmc_ab_2, sp.ablang_2, sp.esmc_ag_2],
            extra_odd=sp.zs_features,
            include_unit_direction=False,
        )

    m_hbin_loc = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(
        _hybrid_lin(tr_sp), tr_sp.y, w_tr, _hybrid_lin(val_sp), val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels
    )
    glb_tr_mask = ~(tr_sp.is_same_ag | (~tr_sp.is_ab_mut))
    glb_val_mask = (val_df["Category"] == "Global").to_numpy()
    m_hglb = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(
        _hybrid_glb(tr_sp)[glb_tr_mask],
        tr_sp.y[glb_tr_mask],
        None,
        _hybrid_glb(val_sp)[glb_val_mask],
        val_df.loc[glb_val_mask].reset_index(drop=True),
        val_sp.is_ag_mut[glb_val_mask],
    )
    m_hyb_bin2h = TwoHeadRegimeRanker(local_head=m_hbin_loc, global_head=m_hglb)
    record(
        "ESMC+AbLang2+LLR (Two-Head Binary)",
        "Impact-Blind Binary Probe",
        m_hyb_bin2h.predict(val_sp, _hybrid_lin(val_sp), _hybrid_glb(val_sp)),
        m_hyb_bin2h.predict(test_sp, _hybrid_lin(test_sp), _hybrid_glb(test_sp)),
        m_hyb_bin2h.predict(cand_sp, _hybrid_lin(cand_sp), _hybrid_glb(cand_sp)),
        m_hyb_bin2h.predict(cm_sp, _hybrid_lin(cm_sp), _hybrid_glb(cm_sp)),
        m_hyb_bin2h.predict(pnl_sp, _hybrid_lin(pnl_sp), _hybrid_glb(pnl_sp)),
        m_hyb_bin2h.predict(vhh_sp, _hybrid_lin(vhh_sp), _hybrid_glb(vhh_sp)),
    )

    # ------------------------------------------------------------------
    # Family D: 3-State Hurdle & Neural Models ({-1, 0, +1})
    # ------------------------------------------------------------------
    print("\n--- Running Family D: 3-State Hurdle & Neural Models ---")
    # 1. ESMC-300M 3-State Hurdle: exact conditional factorization P(|Y|=1|X) * tanh(gamma * z_dir(X))
    esmc_2h_cond = TwoHeadRegimeRanker(local_head=m_esmc_bin, global_head=m_esmc_raw)
    dv_esmc_tr = esmc_2h_cond.predict(tr_sp, _esmc_ms(tr_sp), _esmc_raw(tr_sp))
    dv_esmc_val = esmc_2h_cond.predict(val_sp, _esmc_ms(val_sp), _esmc_raw(val_sp))
    imp_esmc = SymmetricImpactClassifier(seed=42).fit(tr_sp, val_sp, dv_esmc_tr, dv_esmc_val, val_df=val_df)
    hurdle_esmc = OrdinalHurdleRanker(esmc_2h_cond, imp_esmc).calibrate(val_df, val_sp, dv_esmc_val)

    ev_d, ev_i = hurdle_esmc.predict_3state(val_sp, dv_esmc_val)
    et_d, et_i = hurdle_esmc.predict_3state(test_sp, esmc_2h_cond.predict(test_sp, _esmc_ms(test_sp), _esmc_raw(test_sp)))
    ec_d, ec_i = hurdle_esmc.predict_3state(cand_sp, esmc_2h_cond.predict(cand_sp, _esmc_ms(cand_sp), _esmc_raw(cand_sp)))
    ecm_d, ecm_i = hurdle_esmc.predict_3state(cm_sp, esmc_2h_cond.predict(cm_sp, _esmc_ms(cm_sp), _esmc_raw(cm_sp)))
    epnl_d, _ = hurdle_esmc.predict_3state(pnl_sp, esmc_2h_cond.predict(pnl_sp, _esmc_ms(pnl_sp), _esmc_raw(pnl_sp)))
    evhh_d, evhh_i = hurdle_esmc.predict_3state(vhh_sp, esmc_2h_cond.predict(vhh_sp, _esmc_ms(vhh_sp), _esmc_raw(vhh_sp)))
    record(
        "ESMC-300M (3-State Hurdle)",
        "3-State Hurdle / Neural",
        ev_d, et_d, ec_d, ecm_d, epnl_d, evhh_d,
        pv_i=ev_i, pt_i=et_i, pc_i=ec_i, pcm_i=ecm_i, pvhh_i=evhh_i,
    )

    # 2. ESMC + AbLang2 + LLR 3-State Hurdle
    dv_hyb_tr = m_hyb_bin2h.predict(tr_sp, _hybrid_lin(tr_sp), _hybrid_glb(tr_sp))
    dv_hyb_val = m_hyb_bin2h.predict(val_sp, _hybrid_lin(val_sp), _hybrid_glb(val_sp))
    imp_hyb = SymmetricImpactClassifier(seed=42).fit(tr_sp, val_sp, dv_hyb_tr, dv_hyb_val, val_df=val_df)
    hurdle_hyb = OrdinalHurdleRanker(m_hyb_bin2h, imp_hyb).calibrate(val_df, val_sp, dv_hyb_val)

    hv_d, hv_i = hurdle_hyb.predict_3state(val_sp, dv_hyb_val)
    ht_d, ht_i = hurdle_hyb.predict_3state(test_sp, m_hyb_bin2h.predict(test_sp, _hybrid_lin(test_sp), _hybrid_glb(test_sp)))
    hc_d, hc_i = hurdle_hyb.predict_3state(cand_sp, m_hyb_bin2h.predict(cand_sp, _hybrid_lin(cand_sp), _hybrid_glb(cand_sp)))
    hcm_d, hcm_i = hurdle_hyb.predict_3state(cm_sp, m_hyb_bin2h.predict(cm_sp, _hybrid_lin(cm_sp), _hybrid_glb(cm_sp)))
    hpnl_d, _ = hurdle_hyb.predict_3state(pnl_sp, m_hyb_bin2h.predict(pnl_sp, _hybrid_lin(pnl_sp), _hybrid_glb(pnl_sp)))
    hvhh_d, hvhh_i = hurdle_hyb.predict_3state(vhh_sp, m_hyb_bin2h.predict(vhh_sp, _hybrid_lin(vhh_sp), _hybrid_glb(vhh_sp)))
    record(
        "ESMC+AbLang2+LLR (3-State Hurdle)",
        "3-State Hurdle / Neural",
        hv_d, ht_d, hc_d, hcm_d, hpnl_d, hvhh_d,
        pv_i=hv_i, pt_i=ht_i, pc_i=hc_i, pcm_i=hcm_i, pvhh_i=hvhh_i,
    )

    # 3. 3-Class Antisymmetrized XGBoost
    m_xgb = AntisymmetricXGBoostRanker(seed=42).fit(tr_sp, val_sp, val_df, w_tr)
    xv_d, xv_i = m_xgb.predict_3state(val_sp)
    xt_d, xt_i = m_xgb.predict_3state(test_sp)
    xc_d, xc_i = m_xgb.predict_3state(cand_sp)
    xcm_d, xcm_i = m_xgb.predict_3state(cm_sp)
    xpnl_d, _ = m_xgb.predict_3state(pnl_sp)
    xvhh_d, xvhh_i = m_xgb.predict_3state(vhh_sp)
    record(
        "ESMC+AbLang2+LLR (3-State XGBoost)",
        "3-State Hurdle / Neural",
        xv_d, xt_d, xc_d, xcm_d, xpnl_d, xvhh_d,
        pv_i=xv_i, pt_i=xt_i, pc_i=xc_i, pcm_i=xcm_i, pvhh_i=xvhh_i,
    )

    # 4. 3-State Bilinear Net & 5. 3-State Siamese RankNet (3-seed ensembles)
    for arch_key, arch_title in (
        ("bilinear", "ESMC+AbLang2+LLR (3-State Bilinear)"),
        ("siamese", "ESMC+AbLang2+LLR (3-State Siamese)"),
    ):
        vd_l, vi_l, td_l, ti_l, cd_l, ci_l, cmd_l, cmi_l, pd_l, vhhd_l, vhhi_l = [], [], [], [], [], [], [], [], [], [], []
        for sd in (42, 43, 44):
            nn_m = NeuralPairwiseTrainer(arch=arch_key, seed=sd).fit(tr_sp, val_sp, val_df, w_tr)
            a, b = nn_m.predict_3state(val_sp); vd_l.append(a); vi_l.append(b)
            a, b = nn_m.predict_3state(test_sp); td_l.append(a); ti_l.append(b)
            a, b = nn_m.predict_3state(cand_sp); cd_l.append(a); ci_l.append(b)
            a, b = nn_m.predict_3state(cm_sp); cmd_l.append(a); cmi_l.append(b)
            a, _ = nn_m.predict_3state(pnl_sp); pd_l.append(a)
            a, b = nn_m.predict_3state(vhh_sp); vhhd_l.append(a); vhhi_l.append(b)
        record(
            arch_title,
            "3-State Hurdle / Neural",
            np.mean(vd_l, axis=0),
            np.mean(td_l, axis=0),
            np.mean(cd_l, axis=0),
            np.mean(cmd_l, axis=0),
            np.mean(pd_l, axis=0),
            np.mean(vhhd_l, axis=0),
            pv_i=np.mean(vi_l, axis=0),
            pt_i=np.mean(ti_l, axis=0),
            pc_i=np.mean(ci_l, axis=0),
            pcm_i=np.mean(cmi_l, axis=0),
            pvhh_i=np.mean(vhhi_l, axis=0),
        )

    # 6. LogiCA (1D Logit-Space Cross-Attention, 3-seed ensemble)
    lg_vd, lg_vi, lg_td, lg_ti, lg_cd, lg_ci, lg_cmd, lg_cmi, lg_pd, lg_vhhd, lg_vhhi = [], [], [], [], [], [], [], [], [], [], []
    for sd in (42, 43, 44):
        m_log = LogiCA3StateTrainer(seed=sd).fit(tr_sp, val_sp, val_df, w_tr)
        a, b = m_log.predict_3state(val_sp); lg_vd.append(a); lg_vi.append(b)
        a, b = m_log.predict_3state(test_sp); lg_td.append(a); lg_ti.append(b)
        a, b = m_log.predict_3state(cand_sp); lg_cd.append(a); lg_ci.append(b)
        a, b = m_log.predict_3state(cm_sp); lg_cmd.append(a); lg_cmi.append(b)
        a, _ = m_log.predict_3state(pnl_sp); lg_pd.append(a)
        a, b = m_log.predict_3state(vhh_sp); lg_vhhd.append(a); lg_vhhi.append(b)
    record(
        "LogiCA (1D Logit Cross-Attn)",
        "3-State Hurdle / Neural",
        np.mean(lg_vd, axis=0),
        np.mean(lg_td, axis=0),
        np.mean(lg_cd, axis=0),
        np.mean(lg_cmd, axis=0),
        np.mean(lg_pd, axis=0),
        np.mean(lg_vhhd, axis=0),
        pv_i=np.mean(lg_vi, axis=0),
        pt_i=np.mean(lg_ti, axis=0),
        pc_i=np.mean(lg_ci, axis=0),
        pcm_i=np.mean(lg_cmi, axis=0),
        pvhh_i=np.mean(lg_vhhi, axis=0),
    )

    # ------------------------------------------------------------------
    # Families E & F: 3D Structure-Conditioned, Structure-Token (`SaProt`),
    # Multi-Chain (`MINT`), Normal-Mode (`ANTIPASTI`), & Geometric GNN (`GearBind`, `WALLE-Affinity`)
    # ------------------------------------------------------------------
    if struct_cache_file.exists() and inv_cache_file.exists():
        print("\n--- Loading 3D Structure, Inverse-Folding, SaProt-650M, and MINT-650M Caches ---")
        complex_cache = torch.load(struct_cache_file, map_location="cpu", weights_only=False)
        inv_cache = torch.load(inv_cache_file, map_location="cpu", weights_only=False)
        saprot_cache = torch.load(saprot_cache_file, map_location="cpu", weights_only=False) if saprot_cache_file.exists() else None
        mint_cache = torch.load(mint_cache_file, map_location="cpu", weights_only=False) if mint_cache_file.exists() else None

        tr_sf = assemble_structure_pair_features(tr_df, kd_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)
        val_sf = assemble_structure_pair_features(val_df, kd_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)
        test_sf = assemble_structure_pair_features(test_df, kd_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)
        cand_sf = assemble_structure_pair_features(cand_df, kd_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)
        cm_sf = assemble_structure_pair_features(cm_df, cma_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)
        pnl_sf = assemble_structure_pair_features(pnl_pairs_df, kd_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)
        vhh_sf = assemble_structure_pair_features(vhh_df, vhha_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)

        print("\n--- Running Family E: Zero-Shot 3D Inverse Folding, Energetic & Multi-Chain Models ---")
        zs_3d_specs = [
            ("prodigy_sasa", "PRODIGY+SASA (3D Energy ZS)"),
            ("protein_mpnn", "ProteinMPNN (3D Complex ZS)"),
            ("antifold", "AntiFold (3D Ab-Ag ZS)"),
            ("esm_if1", "ESM-IF1 (3D Complex ZS)"),
        ]
        if saprot_cache is not None:
            zs_3d_specs.append(("saprot_zs", "SaProt-650M (3D-Token ZS)"))
        if mint_cache is not None:
            zs_3d_specs.append(("mint_zs", "MINT-650M (Multi-Chain ZS)"))

        for zk, zlabel in zs_3d_specs:
            record(
                zlabel,
                "Zero-Shot 3D / Multi-Chain",
                val_sf.zs_dir[zk], test_sf.zs_dir[zk], cand_sf.zs_dir[zk],
                cm_sf.zs_dir[zk], pnl_sf.zs_dir[zk], vhh_sf.zs_dir[zk],
                pv_i=val_sf.zs_imp[zk], pt_i=test_sf.zs_imp[zk], pc_i=cand_sf.zs_imp[zk],
                pcm_i=cm_sf.zs_imp[zk], pvhh_i=vhh_sf.zs_imp[zk],
            )

        print("\n--- Running Family F: Supervised 3D Structure & Multi-Chain 3-State Models ---")
        # 1. SaProt-650M (3-State Hurdle)
        if saprot_cache is not None:
            pred_sap = fit_3state_foundation_hurdle(
                tr_sf.saprot_ab_1, tr_sf.saprot_ab_2, tr_sf.saprot_ag_1, tr_sf.saprot_ag_2,
                np.concatenate([tr_sf.zs_dir["saprot_zs"][:, None], tr_sp.biophys_odd], axis=1),
                tr_sf.struct_even[:, :9],
                val_sf.saprot_ab_1, val_sf.saprot_ab_2, val_sf.saprot_ag_1, val_sf.saprot_ag_2,
                np.concatenate([val_sf.zs_dir["saprot_zs"][:, None], val_sp.biophys_odd], axis=1),
                val_sf.struct_even[:, :9],
                tr_sp, val_sp, val_df, w_tr,
            )
            sv_d, sv_i = pred_sap(val_sp, val_sf.saprot_ab_1, val_sf.saprot_ab_2, val_sf.saprot_ag_1, val_sf.saprot_ag_2, np.concatenate([val_sf.zs_dir["saprot_zs"][:, None], val_sp.biophys_odd], axis=1), val_sf.struct_even[:, :9])
            st_d, st_i = pred_sap(test_sp, test_sf.saprot_ab_1, test_sf.saprot_ab_2, test_sf.saprot_ag_1, test_sf.saprot_ag_2, np.concatenate([test_sf.zs_dir["saprot_zs"][:, None], test_sp.biophys_odd], axis=1), test_sf.struct_even[:, :9])
            sc_d, sc_i = pred_sap(cand_sp, cand_sf.saprot_ab_1, cand_sf.saprot_ab_2, cand_sf.saprot_ag_1, cand_sf.saprot_ag_2, np.concatenate([cand_sf.zs_dir["saprot_zs"][:, None], cand_sp.biophys_odd], axis=1), cand_sf.struct_even[:, :9])
            scm_d, scm_i = pred_sap(cm_sp, cm_sf.saprot_ab_1, cm_sf.saprot_ab_2, cm_sf.saprot_ag_1, cm_sf.saprot_ag_2, np.concatenate([cm_sf.zs_dir["saprot_zs"][:, None], cm_sp.biophys_odd], axis=1), cm_sf.struct_even[:, :9])
            spnl_d, _ = pred_sap(pnl_sp, pnl_sf.saprot_ab_1, pnl_sf.saprot_ab_2, pnl_sf.saprot_ag_1, pnl_sf.saprot_ag_2, np.concatenate([pnl_sf.zs_dir["saprot_zs"][:, None], pnl_sp.biophys_odd], axis=1), pnl_sf.struct_even[:, :9])
            svhh_d, svhh_i = pred_sap(vhh_sp, vhh_sf.saprot_ab_1, vhh_sf.saprot_ab_2, vhh_sf.saprot_ag_1, vhh_sf.saprot_ag_2, np.concatenate([vhh_sf.zs_dir["saprot_zs"][:, None], vhh_sp.biophys_odd], axis=1), vhh_sf.struct_even[:, :9])
            record(
                "SaProt-650M (3-State Hurdle)",
                "3D Structure / Multi-Chain 3-State",
                sv_d, st_d, sc_d, scm_d, spnl_d, svhh_d,
                pv_i=sv_i, pt_i=st_i, pc_i=sc_i, pcm_i=scm_i, pvhh_i=svhh_i,
            )

        # 2. MINT-650M (3-State Hurdle)
        if mint_cache is not None:
            pred_mint = fit_3state_foundation_hurdle(
                tr_sf.mint_ab_1, tr_sf.mint_ab_2, tr_sf.mint_ag_1, tr_sf.mint_ag_2,
                np.concatenate([tr_sf.zs_dir["mint_zs"][:, None], tr_sp.biophys_odd], axis=1),
                np.zeros((len(tr_df), 1), dtype=np.float32),
                val_sf.mint_ab_1, val_sf.mint_ab_2, val_sf.mint_ag_1, val_sf.mint_ag_2,
                np.concatenate([val_sf.zs_dir["mint_zs"][:, None], val_sp.biophys_odd], axis=1),
                np.zeros((len(val_df), 1), dtype=np.float32),
                tr_sp, val_sp, val_df, w_tr,
            )
            mv_d, mv_i = pred_mint(val_sp, val_sf.mint_ab_1, val_sf.mint_ab_2, val_sf.mint_ag_1, val_sf.mint_ag_2, np.concatenate([val_sf.zs_dir["mint_zs"][:, None], val_sp.biophys_odd], axis=1), np.zeros((len(val_df), 1), dtype=np.float32))
            mt_d, mt_i = pred_mint(test_sp, test_sf.mint_ab_1, test_sf.mint_ab_2, test_sf.mint_ag_1, test_sf.mint_ag_2, np.concatenate([test_sf.zs_dir["mint_zs"][:, None], test_sp.biophys_odd], axis=1), np.zeros((len(test_df), 1), dtype=np.float32))
            mc_d, mc_i = pred_mint(cand_sp, cand_sf.mint_ab_1, cand_sf.mint_ab_2, cand_sf.mint_ag_1, cand_sf.mint_ag_2, np.concatenate([cand_sf.zs_dir["mint_zs"][:, None], cand_sp.biophys_odd], axis=1), np.zeros((len(cand_df), 1), dtype=np.float32))
            mcm_d, mcm_i = pred_mint(cm_sp, cm_sf.mint_ab_1, cm_sf.mint_ab_2, cm_sf.mint_ag_1, cm_sf.mint_ag_2, np.concatenate([cm_sf.zs_dir["mint_zs"][:, None], cm_sp.biophys_odd], axis=1), np.zeros((len(cm_df), 1), dtype=np.float32))
            mpnl_d, _ = pred_mint(pnl_sp, pnl_sf.mint_ab_1, pnl_sf.mint_ab_2, pnl_sf.mint_ag_1, pnl_sf.mint_ag_2, np.concatenate([pnl_sf.zs_dir["mint_zs"][:, None], pnl_sp.biophys_odd], axis=1), np.zeros((len(pnl_pairs_df), 1), dtype=np.float32))
            mvhh_d, mvhh_i = pred_mint(vhh_sp, vhh_sf.mint_ab_1, vhh_sf.mint_ab_2, vhh_sf.mint_ag_1, vhh_sf.mint_ag_2, np.concatenate([vhh_sf.zs_dir["mint_zs"][:, None], vhh_sp.biophys_odd], axis=1), np.zeros((len(vhh_df), 1), dtype=np.float32))
            record(
                "MINT-650M (3-State Hurdle)",
                "3D Structure / Multi-Chain 3-State",
                mv_d, mt_d, mc_d, mcm_d, mpnl_d, mvhh_d,
                pv_i=mv_i, pt_i=mt_i, pc_i=mc_i, pcm_i=mcm_i, pvhh_i=mvhh_i,
            )

        # 3. ANTIPASTI (Normal-Mode 3D CNN, 3-seed ensemble)
        av_d, av_i, at_d, at_i, ac_d, ac_i, acm_d, acm_i, apnl_d, avhh_d, avhh_i = [], [], [], [], [], [], [], [], [], [], []
        for sd in (42, 43, 44):
            m_anti = ANTIPASTI3StateTrainer(seed=sd).fit(tr_sf, tr_sp, val_sf, val_sp, val_df, w_tr)
            a, b = m_anti.predict_3state(val_sf, val_sp); av_d.append(a); av_i.append(b)
            a, b = m_anti.predict_3state(test_sf, test_sp); at_d.append(a); at_i.append(b)
            a, b = m_anti.predict_3state(cand_sf, cand_sp); ac_d.append(a); ac_i.append(b)
            a, b = m_anti.predict_3state(cm_sf, cm_sp); acm_d.append(a); acm_i.append(b)
            a, _ = m_anti.predict_3state(pnl_sf, pnl_sp); apnl_d.append(a)
            a, b = m_anti.predict_3state(vhh_sf, vhh_sp); avhh_d.append(a); avhh_i.append(b)
        record(
            "ANTIPASTI (Normal-Mode 3D CNN)",
            "3D Structure / Multi-Chain 3-State",
            np.mean(av_d, axis=0), np.mean(at_d, axis=0), np.mean(ac_d, axis=0),
            np.mean(acm_d, axis=0), np.mean(apnl_d, axis=0), np.mean(avhh_d, axis=0),
            pv_i=np.mean(av_i, axis=0), pt_i=np.mean(at_i, axis=0), pc_i=np.mean(ac_i, axis=0),
            pcm_i=np.mean(acm_i, axis=0), pvhh_i=np.mean(avhh_i, axis=0),
        )

        # 4. GearBind (3D Geometric Interface GNN, 3-seed ensemble)
        gv_d, gv_i, gt_d, gt_i, gc_d, gc_i, gcm_d, gcm_i, gpnl_d, gvhh_d, gvhh_i = [], [], [], [], [], [], [], [], [], [], []
        for sd in (42, 43, 44):
            m_gb = GearBind3StateTrainer(seed=sd).fit(tr_sf, tr_sp, val_sf, val_sp, val_df, w_tr)
            a, b = m_gb.predict_3state(val_sf); gv_d.append(a); gv_i.append(b)
            a, b = m_gb.predict_3state(test_sf); gt_d.append(a); gt_i.append(b)
            a, b = m_gb.predict_3state(cand_sf); gc_d.append(a); gc_i.append(b)
            a, b = m_gb.predict_3state(cm_sf); gcm_d.append(a); gcm_i.append(b)
            a, _ = m_gb.predict_3state(pnl_sf); gpnl_d.append(a)
            a, b = m_gb.predict_3state(vhh_sf); gvhh_d.append(a); gvhh_i.append(b)
        record(
            "GearBind (3D Geometric GNN)",
            "3D Structure / Multi-Chain 3-State",
            np.mean(gv_d, axis=0), np.mean(gt_d, axis=0), np.mean(gc_d, axis=0),
            np.mean(gcm_d, axis=0), np.mean(gpnl_d, axis=0), np.mean(gvhh_d, axis=0),
            pv_i=np.mean(gv_i, axis=0), pt_i=np.mean(gt_i, axis=0), pc_i=np.mean(gc_i, axis=0),
            pcm_i=np.mean(gcm_i, axis=0), pvhh_i=np.mean(gvhh_i, axis=0),
        )

        # 5. WALLE-Affinity (3-State 3D Complex Graph + PLM Hurdle)
        pred_walle = fit_walle_affinity_3state(tr_sf, tr_sp, val_sf, val_sp, val_df, w_tr, seed=42)
        wv_d, wv_i = pred_walle(val_sf, val_sp)
        wt_d, wt_i = pred_walle(test_sf, test_sp)
        wc_d, wc_i = pred_walle(cand_sf, cand_sp)
        wcm_d, wcm_i = pred_walle(cm_sf, cm_sp)
        wpnl_d, _ = pred_walle(pnl_sf, pnl_sp)
        wvhh_d, wvhh_i = pred_walle(vhh_sf, vhh_sp)
        record(
            "WALLE-Affinity (3-State 3D Graph)",
            "3D Structure / Multi-Chain 3-State",
            wv_d, wt_d, wc_d, wcm_d, wpnl_d, wvhh_d,
            pv_i=wv_i, pt_i=wt_i, pc_i=wc_i, pcm_i=wcm_i, pvhh_i=wvhh_i,
        )

    payload = {
        "benchmark": "AbRank v2.0.0",
        "tier": "unseen_ag",
        "exact_masked_marginal_test_l1_l3": {k: [round(v[0], 4), round(v[1], 4)] for k, v in masked_check.items()},
        "models": results,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fig_path = res_dir / "figures" / "unseen_ag_baselines_auc.png"
    make_comparison_figure(results, fig_path)

    md_path = res_dir / "UNSEEN_ANTIGEN_BASELINES.md"
    write_markdown_report(results, masked_check, test_df, cand_df, cm_df, md_path)

    print(f"\nDone in {time.time() - t0:.1f}s! Saved:\n  - {out_json}\n  - {fig_path}\n  - {md_path}")


if __name__ == "__main__":
    main()
