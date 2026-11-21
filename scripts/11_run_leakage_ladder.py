#!/usr/bin/env python3
"""Step 11: The 5-Rung Evaluation Leakage Ladder — Quantifying Evaluation-Induced
Performance Inflation and Model Rank Reversals in Antibody–Antigen Affinity Prediction.

Evaluates 16 representative models across all 6 methodological families under five
progressively stricter split protocols on the exact same core `KD` local mutational
dataset (`12,267` `L1–L3` pairs across `22` `<50%`-identity antigen clusters):

  1. Rung 1 — Random Pair Split (`Transductive`):
       Standard random 5-fold split over pairs (`AbAgID` complexes shared across train/test).
  2. Rung 2 — Complex-Disjoint Split (`Seen-Ab, Seen-Ag`):
       Strict 0% complex (`AbAgID`) overlap between train, val, and test, while sharing
       parental antibody clonal lineages and target antigens.
  3. Rung 3 — Antibody-Lineage Holdout (`<80% Fv`, `Seen-Ag`):
       5-fold `GroupKFold` holding out entire `<80%` antibody clonal lineages on multi-lineage
       antigens so target antigens are seen in train while antibody families are unseen.
  4. Rung 4 — Antigen-Cluster Holdout (`<50% Ag`, `Cross-Reactive Lineage Allowed`):
       5-fold `GroupKFold` holding out entire `<50%` homologous antigen clusters without
       filtering cross-reactive antibody lineages (e.g., pan-HA `CR9114`/`CR6261`).
  5. Rung 5 — Joint Antigen-Cluster + Clonal-Lineage Holdout (`<50% Ag + <80% Ab`):
       Canonical zero-leakage out-of-distribution holdout (`0%` antigen homology overlap
       AND `0%` antibody lineage overlap).

Every rung uses the identical orientation-symmetrized, Per-Antigen Macro-Averaged (`>=5` pairs)
evaluator so performance shifts isolate molecular data leakage purely.

Outputs:
  - `results/leakage_ladder.json`
  - `results/LEAKAGE_LADDER.md`
"""

from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, KFold
import torch
torch.set_num_threads(1)

from abrank.baselines.features import build_complex_handcrafted_table
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
)
from abrank.baselines.logica_models import (
    LogiCA3StateTrainer,
)
from abrank.baselines.structure_models import (
    ANTIPASTI3StateTrainer,
    GearBind3StateTrainer,
    StructureSplitFeatures,
    assemble_structure_pair_features,
    fit_3state_foundation_hurdle,
    fit_walle_affinity_3state,
)
from abrank.config import get_repo_root
from abrank.evaluate import compute_cluster_bootstrap_summary, compute_per_antigen_macro_metrics


MODEL_SPECS = [
    ("Random Baseline", "Naive"),
    ("Biophys+AAC (Linear)", "Classical"),
    ("k-mer Dipeptide (Ridge)", "Classical"),
    ("ESMC+AbLang2 (1D ZS)", "Zero-Shot 1D PLM"),
    ("PRODIGY+SASA (3D Energy ZS)", "Zero-Shot 3D / PPI"),
    ("ProteinMPNN (3D Complex ZS)", "Zero-Shot 3D / PPI"),
    ("ESM-IF1 (3D Complex ZS)", "Zero-Shot 3D / PPI"),
    ("ESMC-300M (Raw-Diff Binary)", "Impact-Blind"),
    ("ESMC-300M (Multi-Scale Binary)", "Impact-Blind"),
    ("ESMC-300M (3-State Hurdle)", "3-State 1D PLM"),
    ("ESMC+AbLang2+LLR (XGBoost)", "3-State 1D PLM"),
    ("ESMC+AbLang2+LLR (Bilinear)", "3-State 1D PLM"),
    ("LogiCA (1D Logit Cross-Attn)", "1D Logit Cross-Attention"),
    ("SaProt-650M (3-State Hurdle)", "3-State 3D / PPI"),
    ("ANTIPASTI (Normal-Mode 3D CNN)", "3-State 3D / PPI"),
    ("GearBind (3D Geometric GNN)", "3-State 3D / PPI"),
    ("WALLE-Affinity (3-State 3D Graph)", "3-State 3D / PPI"),
]


def _esmc_raw(sp: SplitPairTensors) -> np.ndarray:
    return build_odd_linear_matrix([sp.esmc_ab_1, sp.esmc_ag_1], [sp.esmc_ab_2, sp.esmc_ag_2], include_unit_direction=False)


def _esmc_ms(sp: SplitPairTensors) -> np.ndarray:
    return build_odd_linear_matrix([sp.esmc_ab_1, sp.esmc_ag_1], [sp.esmc_ab_2, sp.esmc_ag_2], include_unit_direction=True)


def run_fold_models(
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    te_idx: np.ndarray,
    df_all: pd.DataFrame,
    sp_all: SplitPairTensors,
    sf_all: StructureSplitFeatures,
    x_bio_all: np.ndarray,
    x_kmer_all: np.ndarray,
    gb_graphs_all: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Train all supervised models strictly on `tr_idx` (validated on `val_idx`) and predict on `te_idx`."""
    tr_df = df_all.iloc[tr_idx].reset_index(drop=True)
    val_df = df_all.iloc[val_idx].reset_index(drop=True)
    tr_sp = sp_all.subset(tr_idx)
    val_sp = sp_all.subset(val_idx)
    te_sp = sp_all.subset(te_idx)
    tr_sf = sf_all.subset(tr_idx)
    val_sf = sf_all.subset(val_idx)
    te_sf = sf_all.subset(te_idx)
    w_tr = compute_stratum_weights(tr_df, tr_sp.is_ag_mut)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_te = len(te_idx)
    out["Random Baseline"] = (np.zeros(n_te, dtype=np.float64), np.zeros(n_te, dtype=np.float64))

    # Zero-shot anchors
    out["ESMC+AbLang2 (1D ZS)"] = (te_sp.zs_dict["esmc_ablang2_zeroshot"], te_sp.zs_impact_dict["esmc_ablang2_zeroshot"])
    out["PRODIGY+SASA (3D Energy ZS)"] = (te_sf.zs_dir["prodigy_sasa"], te_sf.zs_imp["prodigy_sasa"])
    out["ProteinMPNN (3D Complex ZS)"] = (te_sf.zs_dir["protein_mpnn"], te_sf.zs_imp["protein_mpnn"])
    out["ESM-IF1 (3D Complex ZS)"] = (te_sf.zs_dir["esm_if1"], te_sf.zs_imp["esm_if1"])

    # Classical handcrafted
    m_bio = AntisymmetricLinearRanker(use_neutral_nullspace=True).fit(
        x_bio_all[tr_idx], tr_sp.y, w_tr, x_bio_all[val_idx], val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels
    )
    d_bio = m_bio.predict(x_bio_all[te_idx])
    out["Biophys+AAC (Linear)"] = (d_bio, np.abs(d_bio))

    m_km = AntisymmetricLinearRanker(use_neutral_nullspace=True).fit(
        x_kmer_all[tr_idx], tr_sp.y, w_tr, x_kmer_all[val_idx], val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels
    )
    d_km = m_km.predict(x_kmer_all[te_idx])
    out["k-mer Dipeptide (Ridge)"] = (d_km, np.abs(d_km))

    # Impact-blind binary probes
    xr_tr, xr_val, xr_te = _esmc_raw(tr_sp), _esmc_raw(val_sp), _esmc_raw(te_sp)
    m_raw = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(
        xr_tr, tr_sp.y, None, xr_val, val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels
    )
    d_raw = m_raw.predict(xr_te)
    out["ESMC-300M (Raw-Diff Binary)"] = (d_raw, np.abs(d_raw))

    xm_tr, xm_val, xm_te = _esmc_ms(tr_sp), _esmc_ms(val_sp), _esmc_ms(te_sp)
    m_ms = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(
        xm_tr, tr_sp.y, w_tr, xm_val, val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels
    )
    d_ms = m_ms.predict(xm_te)
    out["ESMC-300M (Multi-Scale Binary)"] = (d_ms, np.abs(d_ms))

    # 3-State 1D PLM models
    esmc_2h = TwoHeadRegimeRanker(local_head=m_ms, global_head=m_raw)
    dv_tr = esmc_2h.predict(tr_sp, xm_tr, xr_tr)
    dv_val = esmc_2h.predict(val_sp, xm_val, xr_val)
    imp_esmc = SymmetricImpactClassifier(seed=seed).fit(tr_sp, val_sp, dv_tr, dv_val, val_df=val_df)
    hurdle_esmc = OrdinalHurdleRanker(esmc_2h, imp_esmc).calibrate(val_df, val_sp, dv_val)
    out["ESMC-300M (3-State Hurdle)"] = hurdle_esmc.predict_3state(te_sp, esmc_2h.predict(te_sp, xm_te, xr_te))

    m_xgb = AntisymmetricXGBoostRanker(seed=seed).fit(tr_sp, val_sp, val_df, w_tr)
    out["ESMC+AbLang2+LLR (XGBoost)"] = m_xgb.predict_3state(te_sp)

    m_bil = NeuralPairwiseTrainer(arch="bilinear", seed=seed, epochs=22).fit(tr_sp, val_sp, val_df, w_tr)
    out["ESMC+AbLang2+LLR (Bilinear)"] = m_bil.predict_3state(te_sp)

    m_log = LogiCA3StateTrainer(seed=seed, epochs=20).fit(tr_sp, val_sp, val_df, w_tr)
    out["LogiCA (1D Logit Cross-Attn)"] = m_log.predict_3state(te_sp)

    # 3-State 3D models
    pred_sap = fit_3state_foundation_hurdle(
        tr_sf.saprot_ab_1, tr_sf.saprot_ab_2, tr_sf.saprot_ag_1, tr_sf.saprot_ag_2,
        np.concatenate([tr_sf.zs_dir["saprot_zs"][:, None], tr_sp.biophys_odd], axis=1),
        tr_sf.struct_even[:, :9],
        val_sf.saprot_ab_1, val_sf.saprot_ab_2, val_sf.saprot_ag_1, val_sf.saprot_ag_2,
        np.concatenate([val_sf.zs_dir["saprot_zs"][:, None], val_sp.biophys_odd], axis=1),
        val_sf.struct_even[:, :9],
        tr_sp, val_sp, val_df, w_tr, seed=seed,
    )
    out["SaProt-650M (3-State Hurdle)"] = pred_sap(
        te_sp, te_sf.saprot_ab_1, te_sf.saprot_ab_2, te_sf.saprot_ag_1, te_sf.saprot_ag_2,
        np.concatenate([te_sf.zs_dir["saprot_zs"][:, None], te_sp.biophys_odd], axis=1),
        te_sf.struct_even[:, :9],
    )

    m_anti = ANTIPASTI3StateTrainer(seed=seed, epochs=15).fit(tr_sf, tr_sp, val_sf, val_sp, val_df, w_tr)
    out["ANTIPASTI (Normal-Mode 3D CNN)"] = m_anti.predict_3state(te_sf, te_sp)

    (a1_all, e1_all, m1_all), (a2_all, e2_all, m2_all) = gb_graphs_all
    g_tr = ((a1_all[tr_idx], e1_all[tr_idx], m1_all[tr_idx]), (a2_all[tr_idx], e2_all[tr_idx], m2_all[tr_idx]))
    g_val = ((a1_all[val_idx], e1_all[val_idx], m1_all[val_idx]), (a2_all[val_idx], e2_all[val_idx], m2_all[val_idx]))
    g_te = ((a1_all[te_idx], e1_all[te_idx], m1_all[te_idx]), (a2_all[te_idx], e2_all[te_idx], m2_all[te_idx]))
    m_gb = GearBind3StateTrainer(seed=seed, epochs=15).fit(
        tr_sf, tr_sp, val_sf, val_sp, val_df, w_tr, precomp_tr_g=g_tr, precomp_val_g=g_val
    )
    out["GearBind (3D Geometric GNN)"] = m_gb.predict_3state(te_sf, _precomp_g=g_te)

    pred_walle = fit_walle_affinity_3state(tr_sf, tr_sp, val_sf, val_sp, val_df, w_tr, seed=seed)
    out["WALLE-Affinity (3-State 3D Graph)"] = pred_walle(te_sf, te_sp)

    return out


_G: dict[str, Any] = {}


def _worker_fold(job: tuple[int, int, np.ndarray, np.ndarray, np.ndarray, int]) -> tuple[int, int, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], float]:
    rung_id, fi, tr_idx, val_idx, te_idx, seed = job
    torch.set_num_threads(1)
    t_f = time.time()
    preds = run_fold_models(
        tr_idx, val_idx, te_idx,
        _G["loc_df"], _G["loc_sp"], _G["loc_sf"], _G["x_bio"], _G["x_kmer"], _G["gb_graphs"],
        seed=seed,
    )
    return rung_id, fi, te_idx, preds, time.time() - t_f


def summarize_rung_predictions(
    rung_key: str,
    rung_title: str,
    eval_df: pd.DataFrame,
    oof_d: dict[str, np.ndarray],
    oof_i: dict[str, np.ndarray],
) -> dict:
    """Compute Per-Antigen Macro-AUC + Cluster-Resampled Bootstrap 95% CIs across evaluated antigens."""
    models_out = {}
    for m_name, m_fam in MODEL_SPECS:
        macro = compute_per_antigen_macro_metrics(eval_df, oof_d[m_name], oof_i[m_name], min_pairs=5, n_bootstrap=0)
        clust_boot = compute_cluster_bootstrap_summary(macro["per_antigen"], min_class_pairs=2, n_bootstrap=2000, seed=42)
        models_out[m_name] = {
            "family": m_fam,
            "auc_dir": macro["auc_dir"],
            "auc_impact": macro["auc_impact"],
            "auc_ordinal": macro["auc_ordinal"],
            "auc_selective_hit": macro["auc_selective_hit"],
            "pooled_auc_dir": macro["pooled_auc_dir"],
            "pooled_auc_impact": macro["pooled_auc_impact"],
            "pooled_auc_ordinal": macro["pooled_auc_ordinal"],
            "n_antigens_evaluated": macro["n_antigens_evaluated"],
            "cluster_bootstrap": clust_boot,
        }
    return {
        "rung_key": rung_key,
        "rung_title": rung_title,
        "n_eval_pairs": int(len(eval_df)),
        "n_eval_antigens": int(models_out["Random Baseline"]["n_antigens_evaluated"]),
        "models": models_out,
    }


def main() -> None:
    t0 = time.time()
    root = get_repo_root()
    bdir = root / "data" / "benchmark"
    edir = root / "data" / "embeddings"
    sdir = root / "data" / "structures"
    res_dir = root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    print("Loading master KD table and assembling all 12,267 L1–L3 local mutation pairs...")
    kd_raw = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False)
    kd_lookup = kd_raw.drop_duplicates("AbAgID").set_index("AbAgID")
    kd_lookup["Ab_light_chain_seq"] = kd_lookup["Ab_light_chain_seq"].fillna("")

    sp_parts = [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")]
    all_df = pd.concat(sp_parts, ignore_index=True)
    loc_df = all_df[all_df["Category"].isin(["L1", "L2", "L3"])].reset_index(drop=True)
    n_loc = len(loc_df)

    ag_clust_1 = loc_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
    ab_lin_1 = loc_df["AbAgID_1"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()
    ab_lin_2 = loc_df["AbAgID_2"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()

    print("Loading cached 1D PLM & 3D structure representations...")
    esmc_cache = torch.load(edir / "esmc_300m.pt", map_location="cpu", weights_only=False)
    ablang_cache = torch.load(edir / "ablang2_paired.pt", map_location="cpu", weights_only=False)
    esm2_cache = torch.load(edir / "esm2_150m.pt", map_location="cpu", weights_only=False)
    complex_cache = torch.load(sdir / "complex_cache.pt", map_location="cpu", weights_only=False)
    inv_cache = torch.load(edir / "inverse_folding_scaffolds.pt", map_location="cpu", weights_only=False)
    saprot_cache = torch.load(edir / "saprot_650m.pt", map_location="cpu", weights_only=False)
    mint_cache = torch.load(edir / "mint_650m.pt", map_location="cpu", weights_only=False)

    print("Assembling pair tensors, handcrafted matrices, and precomputed GearBind 3D graphs...")
    loc_sp = assemble_split_tensors(loc_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)
    loc_sf = assemble_structure_pair_features(loc_df, kd_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)

    used_ids = sorted(set(loc_df["AbAgID_1"]).union(loc_df["AbAgID_2"]))
    sub_assays = kd_lookup.loc[used_ids].reset_index()
    bio_map = build_complex_handcrafted_table(sub_assays, include_dipeptides=False)
    kmer_map = build_complex_handcrafted_table(sub_assays, include_dipeptides=True)
    x_bio = build_odd_linear_matrix(
        [np.stack([bio_map[k] for k in loc_df["AbAgID_1"]])],
        [np.stack([bio_map[k] for k in loc_df["AbAgID_2"]])],
        include_unit_direction=True,
    )
    x_kmer = build_odd_linear_matrix(
        [np.stack([kmer_map[k] for k in loc_df["AbAgID_1"]])],
        [np.stack([kmer_map[k] for k in loc_df["AbAgID_2"]])],
        include_unit_direction=True,
    )
    gb_g1 = GearBind3StateTrainer._precompute_graphs(
        loc_sf.patch_ca_1, loc_sf.patch_cb_1, loc_sf.patch_chain_1, loc_sf.patch_seqpos_1
    )
    gb_g2 = GearBind3StateTrainer._precompute_graphs(
        loc_sf.patch_ca_2, loc_sf.patch_cb_2, loc_sf.patch_chain_2, loc_sf.patch_seqpos_2
    )
    gb_graphs = (gb_g1, gb_g2)
    _G.update(loc_df=loc_df, loc_sp=loc_sp, loc_sf=loc_sf, x_bio=x_bio, x_kmer=x_kmer, gb_graphs=gb_graphs)

    jobs: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, int]] = []

    # 1. Rung 1: Random Pair Split (Transductive, 5-fold KFold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fi, (tr_v_idx, te_idx) in enumerate(kf.split(loc_df)):
        rng = np.random.default_rng(100 + fi)
        perm = rng.permutation(len(tr_v_idx))
        n_v = max(200, int(0.15 * len(tr_v_idx)))
        val_idx = tr_v_idx[perm[:n_v]]
        tr_idx = tr_v_idx[perm[n_v:]]
        jobs.append((1, fi, tr_idx, val_idx, te_idx, 42 + fi))

    # 2. Rung 2: Complex-Disjoint Split (0% Complex Overlap, Seen-Ab & Seen-Ag)
    rng2 = np.random.default_rng(42)
    complex_fold: dict[str, int] = {}
    for _, g in sub_assays.groupby(["Ag_cluster", "Ab_lineage"]):
        ids_g = g["AbAgID"].to_numpy().copy()
        rng2.shuffle(ids_g)
        for ii, cid in enumerate(ids_g):
            complex_fold[str(cid)] = ii % 4
    c1_f = np.array([complex_fold[a] for a in loc_df["AbAgID_1"]], dtype=int)
    c2_f = np.array([complex_fold[b] for b in loc_df["AbAgID_2"]], dtype=int)
    for fk in range(4):
        te_mask = (c1_f == fk) & (c2_f == fk)
        val_f = (fk + 1) % 4
        val_mask = (c1_f == val_f) & (c2_f == val_f)
        tr_mask = (c1_f != fk) & (c2_f != fk) & (c1_f != val_f) & (c2_f != val_f)
        jobs.append((2, fk, np.flatnonzero(tr_mask), np.flatnonzero(val_mask), np.flatnonzero(te_mask), 52 + fk))

    # 3. Rung 3: Antibody-Lineage Holdout (<80% Fv, Seen-Ag)
    gkf_lin = GroupKFold(n_splits=5)
    for fi, (tr_v_idx, te_idx) in enumerate(gkf_lin.split(loc_df, loc_sp.y, ab_lin_1)):
        te_lins = set(ab_lin_1[te_idx]) | set(ab_lin_2[te_idx])
        clean_tr_v = np.array([i for i in tr_v_idx if ab_lin_1[i] not in te_lins and ab_lin_2[i] not in te_lins], dtype=int)
        rem_lins = sorted(set(ab_lin_1[clean_tr_v]))
        val_lins = set(rem_lins[1::5])
        is_val = np.array([ab_lin_1[i] in val_lins or ab_lin_2[i] in val_lins for i in clean_tr_v], dtype=bool)
        jobs.append((3, fi, clean_tr_v[~is_val], clean_tr_v[is_val], te_idx, 62 + fi))

    # 4. Rung 4: Antigen-Cluster Holdout Only (<50% Ag, Cross-Reactive Lineages Allowed)
    gkf_ag = GroupKFold(n_splits=5)
    for fi, (tr_v_idx, te_idx) in enumerate(gkf_ag.split(loc_df, loc_sp.y, ag_clust_1)):
        rem_cls = sorted(set(ag_clust_1[tr_v_idx]))
        val_cls = set(rem_cls[1::5])
        is_val = np.array([ag_clust_1[i] in val_cls for i in tr_v_idx], dtype=bool)
        jobs.append((4, fi, tr_v_idx[~is_val], tr_v_idx[is_val], te_idx, 72 + fi))

    # 5. Rung 5: Joint Antigen-Cluster + Clonal-Lineage Holdout (<50% Ag + <80% Ab)
    for fi, (tr_v_idx, te_idx) in enumerate(gkf_ag.split(loc_df, loc_sp.y, ag_clust_1)):
        te_lins = set(ab_lin_1[te_idx]) | set(ab_lin_2[te_idx])
        clean_tr_v = np.array([i for i in tr_v_idx if ab_lin_1[i] not in te_lins and ab_lin_2[i] not in te_lins], dtype=int)
        rem_cls = sorted(set(ag_clust_1[clean_tr_v]))
        val_cls = set(rem_cls[1::5])
        val_lins = set(ab_lin_1[clean_tr_v[np.isin(ag_clust_1[clean_tr_v], list(val_cls))]])
        is_val = np.array([ag_clust_1[i] in val_cls for i in clean_tr_v], dtype=bool)
        is_tr = np.array(
            [ag_clust_1[i] not in val_cls and ab_lin_1[i] not in val_lins and ab_lin_2[i] not in val_lins for i in clean_tr_v],
            dtype=bool,
        )
        jobs.append((5, fi, clean_tr_v[is_tr], clean_tr_v[is_val], te_idx, 82 + fi))

    n_workers = min(8, len(jobs))
    print(f"\n=== Dispatching all {len(jobs)} Rung×Fold jobs across {n_workers} fork workers ===", flush=True)
    oof_d = {m: np.zeros(n_loc, dtype=np.float64) for m, _ in MODEL_SPECS}
    oof_i = {m: np.zeros(n_loc, dtype=np.float64) for m, _ in MODEL_SPECS}
    oof3_d = {m: np.zeros(n_loc, dtype=np.float64) for m, _ in MODEL_SPECS}
    oof3_i = {m: np.zeros(n_loc, dtype=np.float64) for m, _ in MODEL_SPECS}
    oof4_d = {m: np.zeros(n_loc, dtype=np.float64) for m, _ in MODEL_SPECS}
    oof4_i = {m: np.zeros(n_loc, dtype=np.float64) for m, _ in MODEL_SPECS}
    oof5_d = {m: np.zeros(n_loc, dtype=np.float64) for m, _ in MODEL_SPECS}
    oof5_i = {m: np.zeros(n_loc, dtype=np.float64) for m, _ in MODEL_SPECS}
    r2_folds: dict[int, tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}

    done_cnt = 0
    with mp.get_context("fork").Pool(n_workers) as pool:
        for rung_id, fi, te_idx, preds, dt_f in pool.imap_unordered(_worker_fold, jobs):
            done_cnt += 1
            print(f"  [{done_cnt:2d}/{len(jobs)}] Rung {rung_id} Fold {fi + 1} (te={len(te_idx)}) done in {dt_f:.1f}s", flush=True)
            if rung_id == 1:
                for m, (pd_v, pi_v) in preds.items():
                    oof_d[m][te_idx] = pd_v
                    oof_i[m][te_idx] = pi_v
            elif rung_id == 2:
                r2_folds[fi] = (te_idx, preds)
            elif rung_id == 3:
                for m, (pd_v, pi_v) in preds.items():
                    oof3_d[m][te_idx] = pd_v
                    oof3_i[m][te_idx] = pi_v
            elif rung_id == 4:
                for m, (pd_v, pi_v) in preds.items():
                    oof4_d[m][te_idx] = pd_v
                    oof4_i[m][te_idx] = pi_v
            elif rung_id == 5:
                for m, (pd_v, pi_v) in preds.items():
                    oof5_d[m][te_idx] = pd_v
                    oof5_i[m][te_idx] = pi_v

    rungs_out: list[dict] = []
    rungs_out.append(summarize_rung_predictions("rung1_random_pair", "1. Random Pair Split (Transductive)", loc_df, oof_d, oof_i))

    r2_eval_idx_list = [r2_folds[fk][0] for fk in range(4)]
    r2_eval_df = loc_df.iloc[np.concatenate(r2_eval_idx_list)].reset_index(drop=True)
    r2_d_cat = {m: np.concatenate([r2_folds[fk][1][m][0] for fk in range(4)]) for m, _ in MODEL_SPECS}
    r2_i_cat = {m: np.concatenate([r2_folds[fk][1][m][1] for fk in range(4)]) for m, _ in MODEL_SPECS}
    rungs_out.append(summarize_rung_predictions("rung2_complex_disjoint", "2. Complex-Disjoint Split (Seen-Ab, Seen-Ag)", r2_eval_df, r2_d_cat, r2_i_cat))

    lin_per_ag = loc_df.assign(_lin=ab_lin_1).groupby("Ag_group_1")["_lin"].nunique()
    multi_lin_ags = set(lin_per_ag.index[lin_per_ag >= 2])
    r3_mask = loc_df["Ag_group_1"].isin(multi_lin_ags).to_numpy()
    rungs_out.append(
        summarize_rung_predictions(
            "rung3_lineage_holdout",
            "3. Antibody-Lineage Holdout (<80% Fv, Seen-Ag)",
            loc_df.iloc[np.flatnonzero(r3_mask)].reset_index(drop=True),
            {m: v[r3_mask] for m, v in oof3_d.items()},
            {m: v[r3_mask] for m, v in oof3_i.items()},
        )
    )
    rungs_out.append(summarize_rung_predictions("rung4_antigen_only", "4. Antigen-Cluster Holdout (<50% Ag Only)", loc_df, oof4_d, oof4_i))
    rungs_out.append(summarize_rung_predictions("rung5_joint_ag_ab", "5. Joint Ag-Cluster + Ab-Lineage Holdout (Zero-Leakage)", loc_df, oof5_d, oof5_i))

    # Also measure specific cross-reactive HA leakage effect between Rung 4 & Rung 5 (`Influenza_H1_HA` + `Influenza_H3_HA` sharing pan-HA `CR9114`/`CR6261`!)
    ha_mask = loc_df["Ag_group_1"].isin(["Influenza_H1_HA", "Influenza_H3_HA"]).to_numpy()
    ha_sub = loc_df.iloc[np.flatnonzero(ha_mask)].reset_index(drop=True)
    cross_reactive_ha = {}
    for m_name, _ in MODEL_SPECS:
        if m_name == "Random Baseline":
            continue
        m4 = compute_per_antigen_macro_metrics(ha_sub, oof4_d[m_name][ha_mask], oof4_i[m_name][ha_mask], min_pairs=5)
        m5 = compute_per_antigen_macro_metrics(ha_sub, oof5_d[m_name][ha_mask], oof5_i[m_name][ha_mask], min_pairs=5)
        cross_reactive_ha[m_name] = {
            "ag_only_ord": m4["auc_ordinal"],
            "joint_holdout_ord": m5["auc_ordinal"],
            "inflation_ord": round(float(m4["auc_ordinal"] - m5["auc_ordinal"]), 4),
            "ag_only_sign": m4["auc_dir"],
            "joint_holdout_sign": m5["auc_dir"],
        }

    # Leaderboard Spearman rank correlations across rungs (supervised models)
    sup_models = [m for m, fam in MODEL_SPECS if fam not in ("Naive", "Zero-Shot 1D PLM", "Zero-Shot 3D / PPI")]
    r1_ords = [rungs_out[0]["models"][m]["auc_ordinal"] for m in sup_models]
    rank_corrs = {}
    for rg in rungs_out:
        rg_ords = [rg["models"][m]["auc_ordinal"] for m in sup_models]
        rho, _ = spearmanr(r1_ords, rg_ords)
        rank_corrs[rg["rung_key"]] = round(float(rho), 4)

    payload = {
        "benchmark": "AbRank v2.0.0",
        "experiment": "5-Rung Evaluation Leakage Ladder",
        "rungs": rungs_out,
        "supervised_leaderboard_spearman_vs_rung1": rank_corrs,
        "cross_reactive_pan_ha_leakage_rung4_vs_rung5": cross_reactive_ha,
    }
    out_json = res_dir / "leakage_ladder.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Write clean standalone Markdown report
    md = [
        "# AbRank v2 — The 5-Rung Evaluation Leakage Ladder",
        "",
        "## Quantifying Evaluation-Induced Performance Inflation & Rank Reversals",
        "",
        "Each cell reports **Per-Antigen Macro-Averaged `Sign AUC` / `Impact AUC` / `3-Class Ordinal AUC`** across all eligible antigens (`≥5` pairs):",
        "",
        "| Model | Family | Rung 1: Random Pair | Rung 2: Complex-Disjoint | Rung 3: Ab-Lineage Holdout | Rung 4: Ag-Cluster Only | Rung 5: Joint Ag+Ab (Zero-Leakage) | Inflation (`R1 − R5` Ord) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name, m_fam in MODEL_SPECS:
        cells = []
        for rg in rungs_out:
            mv = rg["models"][m_name]
            cells.append(f"`{mv['auc_dir']:.3f}`/`{mv['auc_impact']:.3f}`/**`{mv['auc_ordinal']:.3f}`**")
        infl = rungs_out[0]["models"][m_name]["auc_ordinal"] - rungs_out[4]["models"][m_name]["auc_ordinal"]
        md.append(f"| **{m_name}** | {m_fam} | {' | '.join(cells)} | **`{infl:+.3f}`** |")

    md.extend([
        "",
        f"**Supervised Leaderboard Rank Correlation vs. Rung 1 (`Spearman ρ`)**: `{rank_corrs}`",
        "",
        "## Cross-Reactive Clonal Lineage Leakage (`CR9114`/`CR6261` across `<50%` Influenza H1 & H3 HA Clusters: Rung 4 vs. Rung 5)",
        "",
        "| Model | Rung 4 (`<50%` Ag Holdout Only) Ord (`Sign`) | Rung 5 (`<50%` Ag + `<80%` Ab Joint Holdout) Ord (`Sign`) | Cross-Reactive Lineage Inflation |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for m_name, cr in cross_reactive_ha.items():
        md.append(
            f"| **{m_name}** | `{cr['ag_only_ord']:.3f}` (`{cr['ag_only_sign']:.3f}`) | "
            f"`{cr['joint_holdout_ord']:.3f}` (`{cr['joint_holdout_sign']:.3f}`) | **`{cr['inflation_ord']:+.3f}`** |"
        )

    out_md = res_dir / "LEAKAGE_LADDER.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nCompleted Leakage Ladder in {time.time() - t0:.1f}s! Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
