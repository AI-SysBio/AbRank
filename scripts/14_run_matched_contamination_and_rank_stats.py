#!/usr/bin/env python3
"""Step 14: Matched Fixed-Test Contamination Experiment & Paired Cluster-Bootstrap
Rank-Reversal Inference.

Resolves two critical statistical questions on the Leakage Ladder:
  1. Isolates Training-Set Contamination from Test-Set Distribution Shift by holding the
     EXACT SAME complex-disjoint evaluation set (`T_matched`: `3,032` L1-L3 pairs across
     `21` harmonized antigens) strictly fixed while nesting four training regimes:
       - `C0_disjoint_ag_ab`: `<50%` Ag + `<80%` Ab disjoint (`0%` Ag, lineage, or complex overlap)
       - `C1_disjoint_ag_only`: `<50%` Ag holdout allowing cross-reactive lineages (`0%` Ag/complex overlap)
       - `C2_seen_ag_ab_complex_disjoint`: Seen-Ag & Seen-Ab, `0%` complex overlap with `T_matched`
       - `C3_transductive_shared_complex`: Adds cross-fold pairs sharing complexes with `T_matched`
         while holding out 100% of `T_matched` test pairs (and reverse orientations).
  2. Computes 5,000-replicate paired cluster-level bootstrap CIs and two-sided hypothesis tests
     for (a) per-model contamination inflation (`C3 - C0`, `C3 - C2`, `C2 - C0`), (b) per-model
     leaderboard rank shifts (`ΔRank`), (c) global rank decoupling (`Spearman ρ` & `Kendall τ`
     across all 15 models and 11 supervised models), and (d) head-to-head Difference-in-Differences
     (`DiD`) rank reversals between over-parameterized vs. physics/3D-regularized models.

Outputs:
  - `results/matched_contamination_stats.json`
  - `results/MATCHED_CONTAMINATION_STATS.md`
"""

from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import importlib.util
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon
from sklearn.model_selection import GroupKFold
import torch
torch.set_num_threads(1)

from abrank.baselines.features import build_complex_handcrafted_table
from abrank.baselines.models import assemble_split_tensors, build_odd_linear_matrix
from abrank.baselines.structure_models import GearBind3StateTrainer, assemble_structure_pair_features
from abrank.config import get_repo_root
from abrank.evaluate import compute_cluster_bootstrap_summary, compute_per_antigen_macro_metrics


def _load_step11():
    p = Path(__file__).resolve().parent / "11_run_leakage_ladder.py"
    spec = importlib.util.spec_from_file_location("step11", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_G: dict[str, Any] = {}


def _worker_job(job: tuple[str, int, np.ndarray, np.ndarray, np.ndarray, int]):
    cond, jid, tr_idx, val_idx, te_idx, seed = job
    torch.set_num_threads(1)
    t0 = time.time()
    preds = _G["run_fold_models"](
        tr_idx, val_idx, te_idx,
        _G["loc_df"], _G["loc_sp"], _G["loc_sf"], _G["x_bio"], _G["x_kmer"], _G["gb_graphs"],
        seed=seed,
    )
    return cond, jid, te_idx, preds, time.time() - t0


def paired_cluster_bootstrap_inference(
    per_ag_by_cond: dict[str, dict[str, dict[str, dict]]],
    models: list[str],
    sup_models: list[str],
    cond_hi: str,
    cond_lo: str,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """Paired cluster-resampled bootstrap across antigens for rank shifts, correlations, and DiD."""
    ags = sorted(
        ag for ag, st in per_ag_by_cond[cond_lo][models[0]].items()
        if st.get("auc_ordinal") is not None and np.isfinite(st["auc_ordinal"])
    )
    g = len(ags)
    m_cnt = len(models)
    hi_mat = np.array([[per_ag_by_cond[cond_hi][m][ag]["auc_ordinal"] for ag in ags] for m in models], dtype=float)
    lo_mat = np.array([[per_ag_by_cond[cond_lo][m][ag]["auc_ordinal"] for ag in ags] for m in models], dtype=float)

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, g, size=(n_boot, g))
    hi_b = hi_mat[:, boot_idx].mean(axis=2)  # (M, B)
    lo_b = lo_mat[:, boot_idx].mean(axis=2)  # (M, B)
    infl_b = hi_b - lo_b

    # 1-indexed leaderboard ranks (1 = best / highest Ordinal AUC)
    rk_hi_obs = m_cnt - np.argsort(np.argsort(hi_mat.mean(axis=1)))
    rk_lo_obs = m_cnt - np.argsort(np.argsort(lo_mat.mean(axis=1)))
    rk_hi_b = m_cnt - np.argsort(np.argsort(hi_b, axis=0), axis=0)
    rk_lo_b = m_cnt - np.argsort(np.argsort(lo_b, axis=0), axis=0)
    drk_b = rk_hi_b - rk_lo_b  # positive = improved rank under clean C0 vs leaky C3!

    sup_mask = np.array([m in sup_models for m in models], dtype=bool)
    rho_all_b = np.array([spearmanr(hi_b[:, b], lo_b[:, b])[0] for b in range(n_boot)])
    tau_all_b = np.array([kendalltau(hi_b[:, b], lo_b[:, b])[0] for b in range(n_boot)])
    rho_sup_b = np.array([spearmanr(hi_b[sup_mask, b], lo_b[sup_mask, b])[0] for b in range(n_boot)])

    # Pairwise model discordance fraction across all C(15, 2) = 105 pairs
    disc_b = []
    for b in range(n_boot):
        d_hi = hi_b[:, b][:, None] - hi_b[:, b][None, :]
        d_lo = lo_b[:, b][:, None] - lo_b[:, b][None, :]
        iu = np.triu_indices(m_cnt, k=1)
        disc_b.append(float((d_hi[iu] * d_lo[iu] < 0).mean()))
    iu = np.triu_indices(m_cnt, k=1)
    d_hi_obs = hi_mat.mean(1)[:, None] - hi_mat.mean(1)[None, :]
    d_lo_obs = lo_mat.mean(1)[:, None] - lo_mat.mean(1)[None, :]
    disc_obs = float((d_hi_obs[iu] * d_lo_obs[iu] < 0).mean())

    per_model = {}
    for idx, m in enumerate(models):
        diff_g = hi_mat[idx] - lo_mat[idx]
        p_boot = 2.0 * min(float((infl_b[idx] <= 0).mean()), float((infl_b[idx] >= 0).mean()))
        try:
            _, p_wil = wilcoxon(diff_g)
        except ValueError:
            p_wil = 1.0
        per_model[m] = {
            "hi_mean": round(float(hi_mat[idx].mean()), 4),
            "lo_mean": round(float(lo_mat[idx].mean()), 4),
            "inflation": round(float(diff_g.mean()), 4),
            "inflation_ci95": [round(float(np.percentile(infl_b[idx], 2.5)), 4), round(float(np.percentile(infl_b[idx], 97.5)), 4)],
            "inflation_boot_p": round(float(max(p_boot, 1.0 / n_boot)), 5),
            "inflation_wilcoxon_p": round(float(p_wil), 5),
            "rank_leaky": int(rk_hi_obs[idx]),
            "rank_clean": int(rk_lo_obs[idx]),
            "rank_gain_under_clean": int(rk_hi_obs[idx] - rk_lo_obs[idx]),
            "rank_clean_ci95": [int(np.percentile(rk_lo_b[idx], 2.5)), int(np.percentile(rk_lo_b[idx], 97.5))],
            "rank_gain_ci95": [int(np.percentile(drk_b[idx], 2.5)), int(np.percentile(drk_b[idx], 97.5))],
        }

    # Head-to-head Difference-in-Differences (`DiD`) reversals
    h2h_pairs = [
        ("WALLE-Affinity (3-State 3D Graph)", "ProteinMPNN (3D Complex ZS)"),
        ("WALLE-Affinity (3-State 3D Graph)", "ANTIPASTI (Normal-Mode 3D CNN)"),
        ("SaProt-650M (3-State Hurdle)", "ProteinMPNN (3D Complex ZS)"),
        ("SaProt-650M (3-State Hurdle)", "ANTIPASTI (Normal-Mode 3D CNN)"),
        ("GearBind (3D Geometric GNN)", "ProteinMPNN (3D Complex ZS)"),
        ("ESMC-300M (3-State Hurdle)", "ProteinMPNN (3D Complex ZS)"),
        ("k-mer Dipeptide (Ridge)", "ProteinMPNN (3D Complex ZS)"),
    ]
    did_out = []
    m_to_i = {m: i for i, m in enumerate(models)}
    for ma, mb in h2h_pairs:
        ia, ib = m_to_i[ma], m_to_i[mb]
        gap_hi_g = hi_mat[ia] - hi_mat[ib]
        gap_lo_g = lo_mat[ia] - lo_mat[ib]
        did_g = gap_hi_g - gap_lo_g
        did_b = (hi_b[ia] - hi_b[ib]) - (lo_b[ia] - lo_b[ib])
        p_did = 2.0 * min(float((did_b <= 0).mean()), float((did_b >= 0).mean()))
        _, p_wil = wilcoxon(did_g)
        rev_prob = float(((hi_b[ia] > hi_b[ib]) & (lo_b[ia] < lo_b[ib])).mean())
        did_out.append({
            "model_a": ma,
            "model_b": mb,
            "gap_leaky_a_minus_b": round(float(gap_hi_g.mean()), 4),
            "gap_clean_a_minus_b": round(float(gap_lo_g.mean()), 4),
            "did_estimate": round(float(did_g.mean()), 4),
            "did_ci95": [round(float(np.percentile(did_b, 2.5)), 4), round(float(np.percentile(did_b, 97.5)), 4)],
            "did_boot_p": round(float(max(p_did, 1.0 / n_boot)), 5),
            "did_wilcoxon_p": round(float(p_wil), 5),
            "bootstrap_reversal_prob": round(rev_prob, 4),
        })

    rho_all_obs, p_all_obs = spearmanr(hi_mat.mean(1), lo_mat.mean(1))
    tau_all_obs, pt_all_obs = kendalltau(hi_mat.mean(1), lo_mat.mean(1))
    rho_sup_obs, p_sup_obs = spearmanr(hi_mat[sup_mask].mean(1), lo_mat[sup_mask].mean(1))
    return {
        "n_antigens": g,
        "spearman_all15": {"obs": round(float(rho_all_obs), 4), "perm_p_vs_zero": round(float(p_all_obs), 4), "cluster_ci95": [round(float(np.nanpercentile(rho_all_b, 2.5)), 4), round(float(np.nanpercentile(rho_all_b, 97.5)), 4)]},
        "kendall_all15": {"obs": round(float(tau_all_obs), 4), "perm_p_vs_zero": round(float(pt_all_obs), 4), "cluster_ci95": [round(float(np.nanpercentile(tau_all_b, 2.5)), 4), round(float(np.nanpercentile(tau_all_b, 97.5)), 4)]},
        "spearman_sup11": {"obs": round(float(rho_sup_obs), 4), "perm_p_vs_zero": round(float(p_sup_obs), 4), "cluster_ci95": [round(float(np.nanpercentile(rho_sup_b, 2.5)), 4), round(float(np.nanpercentile(rho_sup_b, 97.5)), 4)]},
        "pairwise_discordance_all15": {"obs": round(disc_obs, 4), "cluster_ci95": [round(float(np.percentile(disc_b, 2.5)), 4), round(float(np.percentile(disc_b, 97.5)), 4)]},
        "per_model": per_model,
        "head_to_head_did": did_out,
    }


def main() -> None:
    t0 = time.time()
    s11 = _load_step11()
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

    ag_clust_1 = loc_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
    ab_lin_1 = loc_df["AbAgID_1"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()
    ab_lin_2 = loc_df["AbAgID_2"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()

    print("Loading caches & assembling pair features...")
    esmc_cache = torch.load(edir / "esmc_300m.pt", map_location="cpu", weights_only=False)
    ablang_cache = torch.load(edir / "ablang2_paired.pt", map_location="cpu", weights_only=False)
    esm2_cache = torch.load(edir / "esm2_150m.pt", map_location="cpu", weights_only=False)
    complex_cache = torch.load(sdir / "complex_cache.pt", map_location="cpu", weights_only=False)
    inv_cache = torch.load(edir / "inverse_folding_scaffolds.pt", map_location="cpu", weights_only=False)
    saprot_cache = torch.load(edir / "saprot_650m.pt", map_location="cpu", weights_only=False)
    mint_cache = torch.load(edir / "mint_650m.pt", map_location="cpu", weights_only=False)

    loc_sp = assemble_split_tensors(loc_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)
    loc_sf = assemble_structure_pair_features(loc_df, kd_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)

    used_ids = sorted(set(loc_df["AbAgID_1"]).union(loc_df["AbAgID_2"]))
    sub_assays = kd_lookup.loc[used_ids].reset_index()
    bio_map = build_complex_handcrafted_table(sub_assays, include_dipeptides=False)
    kmer_map = build_complex_handcrafted_table(sub_assays, include_dipeptides=True)
    x_bio = build_odd_linear_matrix([np.stack([bio_map[k] for k in loc_df["AbAgID_1"]])], [np.stack([bio_map[k] for k in loc_df["AbAgID_2"]])], include_unit_direction=True)
    x_kmer = build_odd_linear_matrix([np.stack([kmer_map[k] for k in loc_df["AbAgID_1"]])], [np.stack([kmer_map[k] for k in loc_df["AbAgID_2"]])], include_unit_direction=True)
    gb_graphs = (
        GearBind3StateTrainer._precompute_graphs(loc_sf.patch_ca_1, loc_sf.patch_cb_1, loc_sf.patch_chain_1, loc_sf.patch_seqpos_1),
        GearBind3StateTrainer._precompute_graphs(loc_sf.patch_ca_2, loc_sf.patch_cb_2, loc_sf.patch_chain_2, loc_sf.patch_seqpos_2),
    )
    _G.update(run_fold_models=s11.run_fold_models, loc_df=loc_df, loc_sp=loc_sp, loc_sf=loc_sf, x_bio=x_bio, x_kmer=x_kmer, gb_graphs=gb_graphs)

    jobs: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray, int]] = []

    # Full 12,267-pair R1 (Random Pair), R4 (Ag-Only), R5 (Joint Ag+Ab)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fi, (tr_v_idx, te_idx) in enumerate(kf.split(loc_df)):
        perm = np.random.default_rng(100 + fi).permutation(len(tr_v_idx))
        n_v = max(200, int(0.15 * len(tr_v_idx)))
        jobs.append(("R1", fi, tr_v_idx[perm[n_v:]], tr_v_idx[perm[:n_v]], te_idx, 42 + fi))

    gkf_ag = GroupKFold(n_splits=5)
    for fi, (tr_v_idx, te_idx) in enumerate(gkf_ag.split(loc_df, loc_sp.y, ag_clust_1)):
        rem_cls = sorted(set(ag_clust_1[tr_v_idx]))
        val_cls = set(rem_cls[1::5])
        is_val4 = np.array([ag_clust_1[i] in val_cls for i in tr_v_idx], dtype=bool)
        jobs.append(("R4_C1", fi, tr_v_idx[~is_val4], tr_v_idx[is_val4], te_idx, 72 + fi))

        te_lins = set(ab_lin_1[te_idx]) | set(ab_lin_2[te_idx])
        clean_tr_v = np.array([i for i in tr_v_idx if ab_lin_1[i] not in te_lins and ab_lin_2[i] not in te_lins], dtype=int)
        rem_cls5 = sorted(set(ag_clust_1[clean_tr_v]))
        val_cls5 = set(rem_cls5[1::5])
        val_lins5 = set(ab_lin_1[clean_tr_v[np.isin(ag_clust_1[clean_tr_v], list(val_cls5))]])
        is_val5 = np.array([ag_clust_1[i] in val_cls5 for i in clean_tr_v], dtype=bool)
        is_tr5 = np.array([ag_clust_1[i] not in val_cls5 and ab_lin_1[i] not in val_lins5 and ab_lin_2[i] not in val_lins5 for i in clean_tr_v], dtype=bool)
        jobs.append(("R5_C0", fi, clean_tr_v[is_tr5], clean_tr_v[is_val5], te_idx, 82 + fi))

    # Matched Complex-Disjoint Evaluation Subset (`T_matched`: 3,032 pairs across 21 antigens)
    rng2 = np.random.default_rng(42)
    complex_fold: dict[str, int] = {}
    for _, g in sub_assays.groupby(["Ag_cluster", "Ab_lineage"]):
        ids_g = g["AbAgID"].to_numpy().copy()
        rng2.shuffle(ids_g)
        for ii, cid in enumerate(ids_g):
            complex_fold[str(cid)] = ii % 4
    c1_f = np.array([complex_fold[a] for a in loc_df["AbAgID_1"]], dtype=int)
    c2_f = np.array([complex_fold[b] for b in loc_df["AbAgID_2"]], dtype=int)
    unord_pairs = [frozenset((a, b)) for a, b in zip(loc_df["AbAgID_1"], loc_df["AbAgID_2"])]

    for fk in range(4):
        te_mask = (c1_f == fk) & (c2_f == fk)
        te_idx = np.flatnonzero(te_mask)
        val_f = (fk + 1) % 4
        # C2: Seen-Ag & Seen-Ab, 0% complex overlap with `te_idx`
        val_mask2 = (c1_f == val_f) & (c2_f == val_f)
        tr_mask2 = (c1_f != fk) & (c2_f != fk) & (c1_f != val_f) & (c2_f != val_f)
        jobs.append(("C2", fk, np.flatnonzero(tr_mask2), np.flatnonzero(val_mask2), te_idx, 52 + fk))

        # C3: Transductive Shared-Complex Contamination on EXACT SAME `te_idx` (0% unordered pair overlap!)
        te_unord = {unord_pairs[i] for i in te_idx}
        pool3 = np.array([i for i in range(n_loc) if unord_pairs[i] not in te_unord], dtype=int)
        perm3 = np.random.default_rng(200 + fk).permutation(len(pool3))
        nv3 = max(200, int(0.15 * len(pool3)))
        jobs.append(("C3", fk, pool3[perm3[nv3:]], pool3[perm3[:nv3]], te_idx, 92 + fk))

    model_names = [m for m, _ in s11.MODEL_SPECS if m != "Random Baseline"]
    sup_models = [m for m, fam in s11.MODEL_SPECS if fam not in ("Naive", "Zero-Shot 1D PLM", "Zero-Shot 3D / PPI")]

    oof_full = {c: {m: (np.zeros(n_loc), np.zeros(n_loc)) for m, _ in s11.MODEL_SPECS} for c in ("R1", "R4_C1", "R5_C0")}
    folds_matched: dict[str, dict[int, tuple[np.ndarray, dict]]] = {"C2": {}, "C3": {}}

    print(f"Dispatching {len(jobs)} jobs across 8 workers...", flush=True)
    with mp.get_context("fork").Pool(8) as pool:
        for cond, jid, te_idx, preds, dt in pool.imap_unordered(_worker_job, jobs):
            print(f"  [{cond} fold {jid}] n_te={len(te_idx)} done in {dt:.1f}s", flush=True)
            if cond in oof_full:
                for m, (pd_v, pi_v) in preds.items():
                    oof_full[cond][m][0][te_idx] = pd_v
                    oof_full[cond][m][1][te_idx] = pi_v
            else:
                folds_matched[cond][jid] = (te_idx, preds)

    # Assemble Matched Fixed-Test (`T_matched`: exact same 3,032 pairs in exact same order!)
    t_idx_cat = np.concatenate([folds_matched["C2"][fk][0] for fk in range(4)])
    t_matched_df = loc_df.iloc[t_idx_cat].reset_index(drop=True)

    matched_preds: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        "C0_disjoint_ag_ab": {m: (oof_full["R5_C0"][m][0][t_idx_cat], oof_full["R5_C0"][m][1][t_idx_cat]) for m, _ in s11.MODEL_SPECS},
        "C1_disjoint_ag_only": {m: (oof_full["R4_C1"][m][0][t_idx_cat], oof_full["R4_C1"][m][1][t_idx_cat]) for m, _ in s11.MODEL_SPECS},
        "C2_seen_ag_ab_complex_disjoint": {
            m: (
                np.concatenate([folds_matched["C2"][fk][1][m][0] for fk in range(4)]),
                np.concatenate([folds_matched["C2"][fk][1][m][1] for fk in range(4)]),
            )
            for m, _ in s11.MODEL_SPECS
        },
        "C3_transductive_shared_complex": {
            m: (
                np.concatenate([folds_matched["C3"][fk][1][m][0] for fk in range(4)]),
                np.concatenate([folds_matched["C3"][fk][1][m][1] for fk in range(4)]),
            )
            for m, _ in s11.MODEL_SPECS
        },
    }

    matched_summary = {}
    matched_per_ag: dict[str, dict[str, dict]] = {}
    for ck, mp_dict in matched_preds.items():
        matched_summary[ck] = {}
        matched_per_ag[ck] = {}
        for m, fam in s11.MODEL_SPECS:
            mac = compute_per_antigen_macro_metrics(t_matched_df, mp_dict[m][0], mp_dict[m][1], min_pairs=5, n_bootstrap=0)
            cb = compute_cluster_bootstrap_summary(mac["per_antigen"], min_class_pairs=2, n_bootstrap=2000, seed=42)
            matched_per_ag[ck][m] = mac["per_antigen"]
            matched_summary[ck][m] = {
                "family": fam,
                "auc_dir": mac["auc_dir"],
                "auc_impact": mac["auc_impact"],
                "auc_ordinal": mac["auc_ordinal"],
                "n_antigens": mac["n_antigens_evaluated"],
                "cluster_bootstrap": cb,
            }

    full_per_ag: dict[str, dict[str, dict]] = {}
    for ck in ("R1", "R4_C1", "R5_C0"):
        full_per_ag[ck] = {}
        for m, _ in s11.MODEL_SPECS:
            mac = compute_per_antigen_macro_metrics(loc_df, oof_full[ck][m][0], oof_full[ck][m][1], min_pairs=5, n_bootstrap=0)
            full_per_ag[ck][m] = mac["per_antigen"]

    inf_full_r1_vs_r5 = paired_cluster_bootstrap_inference(full_per_ag, model_names, sup_models, "R1", "R5_C0", n_boot=5000, seed=42)
    inf_matched_c3_vs_c0 = paired_cluster_bootstrap_inference(matched_per_ag, model_names, sup_models, "C3_transductive_shared_complex", "C0_disjoint_ag_ab", n_boot=5000, seed=42)
    inf_matched_c3_vs_c2 = paired_cluster_bootstrap_inference(matched_per_ag, model_names, sup_models, "C3_transductive_shared_complex", "C2_seen_ag_ab_complex_disjoint", n_boot=5000, seed=42)
    inf_matched_c2_vs_c0 = paired_cluster_bootstrap_inference(matched_per_ag, model_names, sup_models, "C2_seen_ag_ab_complex_disjoint", "C0_disjoint_ag_ab", n_boot=5000, seed=42)

    out_payload = {
        "n_matched_test_pairs": int(len(t_matched_df)),
        "n_matched_antigens": int(matched_summary["C0_disjoint_ag_ab"]["Random Baseline"]["n_antigens"]),
        "matched_fixed_test_regimes": matched_summary,
        "paired_inference_full_12267_R1_vs_R5": inf_full_r1_vs_r5,
        "paired_inference_matched_3032_C3_vs_C0_total": inf_matched_c3_vs_c0,
        "paired_inference_matched_3032_C3_vs_C2_pure_transductive": inf_matched_c3_vs_c2,
        "paired_inference_matched_3032_C2_vs_C0_scaffold_familiarity": inf_matched_c2_vs_c0,
    }
    out_json = res_dir / "matched_contamination_stats.json"
    out_json.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Matched Fixed-Test Contamination & Paired Cluster-Bootstrap Rank-Reversal Inference",
        "",
        f"Evaluated on the **exact same `T_matched` test set (`{len(t_matched_df)}` pairs across `{out_payload['n_matched_antigens']}` antigens)** with `0%` unordered pair overlap in all four regimes:",
        "",
        "| Model | `C0` (`<50%` Ag + `<80%` Ab) | `C1` (`<50%` Ag Only) | `C2` (Seen-Ag/Ab, `0%` Complex) | `C3` (Transductive Shared-Complex) | Scaffold Shift (`C2-C0`) | Pure Transductive (`C3-C2`) | Total (`C3-C0`) [95% CI] | Rank (`C3` -> `C0`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m in model_names:
        c0 = matched_summary["C0_disjoint_ag_ab"][m]["auc_ordinal"]
        c1 = matched_summary["C1_disjoint_ag_only"][m]["auc_ordinal"]
        c2 = matched_summary["C2_seen_ag_ab_complex_disjoint"][m]["auc_ordinal"]
        c3 = matched_summary["C3_transductive_shared_complex"][m]["auc_ordinal"]
        pm = inf_matched_c3_vs_c0["per_model"][m]
        ci = pm["inflation_ci95"]
        md.append(
            f"| **{m}** | `{c0:.3f}` | `{c1:.3f}` | `{c2:.3f}` | `{c3:.3f}` | `{c2 - c0:+.3f}` | `{c3 - c2:+.3f}` | **`{c3 - c0:+.3f}`** `[+{ci[0]:.2f}, +{ci[1]:.2f}]` | `#{pm['rank_leaky']} -> #{pm['rank_clean']}` (`{pm['rank_gain_under_clean']:+d}`) |"
        )
    out_md = res_dir / "MATCHED_CONTAMINATION_STATS.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Done in {time.time() - t0:.1f}s! Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
