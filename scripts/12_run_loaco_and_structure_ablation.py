#!/usr/bin/env python3
"""Step 12: Leave-One-Antigen-Cluster-Out (`LOACO`) Evaluation & Controlled Bound Co-Crystal
vs. Unbound `IgFold` Structure Ablation across the Full AbRank `KD` Benchmark.

Implements two foundational scientific experiments:
  1. Part A — Leave-One-Antigen-Cluster-Out (`LOACO`) across all independent `<50%`-identity
     antigen clusters:
       - `LOACO-Local` (`L1–L3`): 22 independent `<50%` antigen clusters (`12,267` pairs,
         `5,426` impactful `±1` pairs — 65x more impactful test pairs than the single fixed split).
       - `LOACO-Candidate` (`d_Lev > 3`): 19 independent `<50%` antigen clusters (`3,629` 3D-cached
         candidate pairs).
       - `LOACO-Panels`: 22 complete unfiltered `(Ag_group, Dataset)` variant panels (`1,242` variants).
     Each fold $g$ strictly excludes all complexes in cluster $g$ AND all `<80%` antibody clonal
     lineages present in $g$, using nested inner antigen-cluster validation (`0%` test peeking).
     Reports Cluster-Macro Mean ± SD, Median [IQR], 2,000-replicate Cluster-Resampled Bootstrap
     95% CIs, explicit cluster coverage, and Target Difficulty Correlates.

  2. Part B — Controlled Bound Co-Crystal vs. Unbound `IgFold` Paired Structure Ablation:
     On the exact same 16 solved co-crystal `LOACO-Local` clusters (`47` co-crystal scaffolds),
     replaces experimental bound antibody Fv chains with unbound `IgFold`-predicted Fv models
     oriented against the RCSB antigen (`is_cocrystal = False`), isolating how much of the 3D
     advantage comes from privileged bound complex geometry vs. unbound paratope structure.

Outputs:
  - `results/loaco_evaluation.json`
  - `results/LOACO_EVALUATION.md`
"""

from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import gc
import hashlib
import importlib.util
from itertools import combinations
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import Levenshtein
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
torch.set_num_threads(1)

from abrank.baselines.embeddings import seq_hash
from abrank.baselines.features import build_complex_handcrafted_table
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
from abrank.evaluate import (
    compute_3state_metrics,
    compute_cluster_bootstrap_summary,
    evaluate_pointwise_spearman,
)
from abrank.schema import PAIR_COLUMNS
from abrank.structures.resolver import (
    compute_scaffold_biophysics,
    extract_foldseek_3di_batch,
    fold_missing_antibodies_igfold,
    orient_unbound_ab_ag,
    parse_structure_chains,
    write_standardized_pdb,
)


LOACO_MODELS = [
    ("Random Baseline", "Naive"),
    ("Biophys+AAC (Linear)", "Classical"),
    ("k-mer Dipeptide (Ridge)", "Classical"),
    ("AbLang2-Paired (1D ZS)", "Zero-Shot 1D PLM"),
    ("ESMC-300M (1D ZS)", "Zero-Shot 1D PLM"),
    ("ESMC+AbLang2 (1D ZS)", "Zero-Shot 1D PLM"),
    ("PRODIGY+SASA (3D Energy ZS)", "Zero-Shot 3D / PPI"),
    ("ProteinMPNN (3D Complex ZS)", "Zero-Shot 3D / PPI"),
    ("AntiFold (3D Ab-Ag ZS)", "Zero-Shot 3D / PPI"),
    ("ESM-IF1 (3D Complex ZS)", "Zero-Shot 3D / PPI"),
    ("SaProt-650M (3D-Token ZS)", "Zero-Shot 3D / PPI"),
    ("MINT-650M (Multi-Chain ZS)", "Zero-Shot 3D / PPI"),
    ("ESMC-300M (Multi-Scale Binary)", "Impact-Blind"),
    ("ESMC-300M (3-State Hurdle)", "3-State 1D PLM"),
    ("ESMC+AbLang2+LLR (Bilinear)", "3-State 1D PLM"),
    ("LogiCA (1D Logit Cross-Attn)", "1D Logit Cross-Attention"),
    ("SaProt-650M (3-State Hurdle)", "3-State 3D / PPI"),
    ("MINT-650M (3-State Hurdle)", "3-State 3D / PPI"),
    ("ANTIPASTI (Normal-Mode 3D CNN)", "3-State 3D / PPI"),
    ("GearBind (3D Geometric GNN)", "3-State 3D / PPI"),
    ("WALLE-Affinity (3-State 3D Graph)", "3-State 3D / PPI"),
]


def _load_step09_workers(root: Path):
    spec = importlib.util.spec_from_file_location("step09", root / "scripts" / "09_extract_structure_and_multichain_features.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._if1_antifold_worker, mod._protein_mpnn_worker, mod._saprot_worker


def build_or_load_unbound_ablation_cache(
    root: Path,
    cocrys_sids: list[str],
    complex_cache: dict,
    inv_cache: dict,
    saprot_cache: dict,
    loc_df: pd.DataFrame,
    kd_lookup: pd.DataFrame,
) -> tuple[dict, dict, dict]:
    """Build or load unbound `IgFold` shadow scaffolds for the 47 solved co-crystal scaffolds."""
    sdir = root / "data" / "structures"
    ablation_cache_path = sdir / "unbound_cocrystal_ablation.pt"
    if ablation_cache_path.exists():
        saved = torch.load(ablation_cache_path, map_location="cpu", weights_only=False)
        if set(cocrys_sids) <= set(saved["unb_scaffolds"].keys()):
            print(f"Loaded cached unbound IgFold ablation scaffolds ({len(saved['unb_scaffolds'])}) from {ablation_cache_path}")
            return _merge_ablation_caches(cocrys_sids, complex_cache, inv_cache, saprot_cache, saved)

    print(f"Building unbound IgFold shadow scaffolds for {len(cocrys_sids)} solved co-crystal scaffolds...")
    igfold_dir = sdir / "igfold"
    unb_pdb_dir = sdir / "unbound_ablation_scaffolds"
    unb_pdb_dir.mkdir(parents=True, exist_ok=True)
    scafs_orig = complex_cache["scaffolds"]

    igfold_jobs: dict[str, tuple[str, str, str]] = {}
    sid_to_ab_key: dict[str, str] = {}
    for sid in cocrys_sids:
        bio = scafs_orig[sid]
        vh, vl = str(bio["seq_H"]), str(bio["seq_L"])
        ab_key = hashlib.sha1(f"{vh}|{vl}".encode()).hexdigest()[:14]
        sid_to_ab_key[sid] = ab_key
        igfold_jobs[ab_key] = (ab_key, vh, vl)

    fold_missing_antibodies_igfold(list(igfold_jobs.values()), igfold_dir, n_workers=4)

    unb_scaffolds: dict[str, dict] = {}
    written_pdbs: list[Path] = []
    shard_jobs: list[tuple[str, str, bool]] = []
    for sid in cocrys_sids:
        orig_pdb = root / scafs_orig[sid]["pdb_path"]
        orig_chains = parse_structure_chains(orig_pdb)
        cb_a = orig_chains["A"]
        ab_chains = parse_structure_chains(igfold_dir / f"{sid_to_ab_key[sid]}.pdb")
        cb_h = ab_chains.get("H") or next(iter(ab_chains.values()))
        vl_s = str(scafs_orig[sid]["seq_L"])
        cb_l = ab_chains.get("L") if vl_s else None

        cb_h_o, cb_l_o, cb_a_o = orient_unbound_ab_ag(cb_h, cb_l, cb_a)
        unb_sid = f"UNB_{sid}"
        out_pdb = unb_pdb_dir / f"{unb_sid}.pdb"
        write_standardized_pdb(out_pdb, cb_h_o, cb_l_o, cb_a_o)
        written_pdbs.append(out_pdb)

        bio_u = compute_scaffold_biophysics(out_pdb, cb_h_o, cb_l_o, cb_a_o, is_cocrystal=False)
        bio_u["pdb_path"] = str(out_pdb.relative_to(root))
        unb_scaffolds[sid] = bio_u
        shard_jobs.append((unb_sid, str(out_pdb), bool(vl_s)))

    foldseek_bin = Path.home() / ".cache" / "abrank" / "bin" / "foldseek" / "bin" / "foldseek"
    s3di_map = extract_foldseek_3di_batch(written_pdbs, foldseek_bin, sdir / "tmp_3di_unb")
    for sid in cocrys_sids:
        unb_sid = f"UNB_{sid}"
        d3 = s3di_map.get(unb_sid, {})
        bio_u = unb_scaffolds[sid]
        bio_u["3di_H"] = d3.get("H", "d" * len(bio_u["seq_H"]))
        bio_u["3di_L"] = d3.get("L", "d" * len(bio_u["seq_L"])) if bio_u["seq_L"] else ""
        bio_u["3di_A"] = d3.get("A", "d" * len(bio_u["seq_A"]))

    if1_worker, mpnn_worker, sap_worker = _load_step09_workers(root)
    mpnn_repo = str(Path.home() / ".cache" / "abrank" / "repos" / "ProteinMPNN")
    print("  Running ESM-IF1, AntiFold, and ProteinMPNN on 47 unbound IgFold scaffolds...")
    if1_af_raw = if1_worker((shard_jobs, str(unb_pdb_dir), ""))
    mpnn_raw = mpnn_worker((shard_jobs, mpnn_repo))

    unb_inv: dict[str, dict] = {}
    for sid in cocrys_sids:
        unb_sid = f"UNB_{sid}"
        unb_inv[sid] = {**if1_af_raw[unb_sid], **mpnn_raw[unb_sid]}

    # Extract SaProt chains for the unbound 3Di tokens (scaffold reps + non-AbBiBench variants)
    a2s = complex_cache["abag_to_scaffold"]
    used_ids = sorted(set(loc_df["AbAgID_1"]).union(loc_df["AbAgID_2"]))
    sap_existing = saprot_cache["chains"]
    sap_jobs: dict[str, tuple[str, str, str]] = {}
    cocrys_set = set(cocrys_sids)
    for aid in used_ids:
        sid = a2s[aid]
        if sid not in cocrys_set:
            continue
        r = kd_lookup.loc[aid]
        if str(r["Dataset"]) == "AbBibench" and aid != saprot_cache["scaf_rep_abag"].get(sid):
            continue
        bio_u = unb_scaffolds[sid]
        for seq_col, s3_k in (("Ab_heavy_chain_seq", "3di_H"), ("Ab_light_chain_seq", "3di_L"), ("Ag_seq", "3di_A")):
            s_aa = str(r[seq_col])
            if not s_aa:
                continue
            s3 = str(bio_u[s3_k])
            ckey = hashlib.sha1(f"{s_aa}|{s3[:len(s_aa)]}".encode()).hexdigest()[:16]
            if ckey not in sap_existing and ckey not in sap_jobs:
                sap_jobs[ckey] = (ckey, s_aa, s3)

    unb_sap_chains: dict[str, dict] = {}
    if sap_jobs:
        print(f"  Running SaProt-650M on {len(sap_jobs)} unbound `(seq, 3Di)` chains...")
        unb_sap_chains = sap_worker((list(sap_jobs.values()), 8))

    saved_payload = {
        "unb_scaffolds": unb_scaffolds,
        "unb_inv": unb_inv,
        "unb_sap_chains": unb_sap_chains,
    }
    torch.save(saved_payload, ablation_cache_path)
    print(f"  Saved {ablation_cache_path}!")
    return _merge_ablation_caches(cocrys_sids, complex_cache, inv_cache, saprot_cache, saved_payload)


def _merge_ablation_caches(
    cocrys_sids: list[str],
    complex_cache: dict,
    inv_cache: dict,
    saprot_cache: dict,
    saved: dict,
) -> tuple[dict, dict, dict]:
    cc_unb = {
        "abag_to_scaffold": complex_cache["abag_to_scaffold"],
        "scaffolds": {**complex_cache["scaffolds"], **saved["unb_scaffolds"]},
    }
    inv_src = inv_cache["scaffolds"] if "scaffolds" in inv_cache else inv_cache
    inv_unb = {"scaffolds": {**inv_src, **saved["unb_inv"]}} if "scaffolds" in inv_cache else {**inv_cache, **saved["unb_inv"]}
    sap_unb = {
        "chains": {**saprot_cache["chains"], **saved["unb_sap_chains"]},
        "scaf_rep_abag": saprot_cache["scaf_rep_abag"],
    }
    return cc_unb, inv_unb, sap_unb


def build_panel_tournament_pairs(pw_df: pd.DataFrame) -> pd.DataFrame:
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
                    "pair_id": f"LPNL_{len(rows) + 1:06d}",
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


def tournament_scores_from_pairs(pnl_pairs_df: pd.DataFrame, pred_diff: np.ndarray) -> dict[str, float]:
    sums: dict[str, float] = {}
    cnts: dict[str, int] = {}
    for a, b, p in zip(pnl_pairs_df["AbAgID_1"], pnl_pairs_df["AbAgID_2"], pred_diff):
        fp = float(p)
        sums[a] = sums.get(a, 0.0) + fp
        cnts[a] = cnts.get(a, 0) + 1
        sums[b] = sums.get(b, 0.0) - fp
        cnts[b] = cnts.get(b, 0) + 1
    return {k: sums[k] / max(cnts[k], 1) for k in sums}


def fit_and_predict_cluster(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    tr_sp: SplitPairTensors,
    va_sp: SplitPairTensors,
    tr_sf: StructureSplitFeatures,
    va_sf: StructureSplitFeatures,
    xb_tr: np.ndarray,
    xb_va: np.ndarray,
    xk_tr: np.ndarray,
    xk_va: np.ndarray,
    gb_tr: tuple,
    gb_va: tuple,
    eval_bundles: dict[str, tuple[SplitPairTensors, StructureSplitFeatures, np.ndarray, np.ndarray, tuple | None]],
    seed: int = 42,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Fit all supervised models once for a held-out LOACO cluster and predict on all its eval bundles."""
    w_tr = compute_stratum_weights(tr_df, tr_sp.is_ag_mut)
    preds_by_bundle: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {bk: {} for bk in eval_bundles}

    # 1. Naive + Zero-Shot 1D & 3D
    for bk, (esp, esf, _, _, _) in eval_bundles.items():
        n_e = len(esp.y)
        preds_by_bundle[bk]["Random Baseline"] = (np.zeros(n_e), np.zeros(n_e))
        preds_by_bundle[bk]["AbLang2-Paired (1D ZS)"] = (esp.zs_dict["ablang2_zeroshot"], esp.zs_impact_dict["ablang2_zeroshot"])
        preds_by_bundle[bk]["ESMC-300M (1D ZS)"] = (esp.zs_dict["esmc_300m_zeroshot"], esp.zs_impact_dict["esmc_300m_zeroshot"])
        preds_by_bundle[bk]["ESMC+AbLang2 (1D ZS)"] = (esp.zs_dict["esmc_ablang2_zeroshot"], esp.zs_impact_dict["esmc_ablang2_zeroshot"])
        preds_by_bundle[bk]["PRODIGY+SASA (3D Energy ZS)"] = (esf.zs_dir["prodigy_sasa"], esf.zs_imp["prodigy_sasa"])
        preds_by_bundle[bk]["ProteinMPNN (3D Complex ZS)"] = (esf.zs_dir["protein_mpnn"], esf.zs_imp["protein_mpnn"])
        preds_by_bundle[bk]["AntiFold (3D Ab-Ag ZS)"] = (esf.zs_dir["antifold"], esf.zs_imp["antifold"])
        preds_by_bundle[bk]["ESM-IF1 (3D Complex ZS)"] = (esf.zs_dir["esm_if1"], esf.zs_imp["esm_if1"])
        preds_by_bundle[bk]["SaProt-650M (3D-Token ZS)"] = (esf.zs_dir["saprot_zs"], esf.zs_imp["saprot_zs"])
        preds_by_bundle[bk]["MINT-650M (Multi-Chain ZS)"] = (esf.zs_dir["mint_zs"], esf.zs_imp["mint_zs"])

    # 2. Classical Handcrafted
    m_bio = AntisymmetricLinearRanker(use_neutral_nullspace=True).fit(xb_tr, tr_sp.y, w_tr, xb_va, va_df, va_sp.is_ag_mut, labels_tr=tr_sp.labels)
    m_km = AntisymmetricLinearRanker(use_neutral_nullspace=True).fit(xk_tr, tr_sp.y, w_tr, xk_va, va_df, va_sp.is_ag_mut, labels_tr=tr_sp.labels)
    for bk, (_, _, xb_e, xk_e, _) in eval_bundles.items():
        db = m_bio.predict(xb_e)
        dk = m_km.predict(xk_e)
        preds_by_bundle[bk]["Biophys+AAC (Linear)"] = (db, np.abs(db))
        preds_by_bundle[bk]["k-mer Dipeptide (Ridge)"] = (dk, np.abs(dk))

    # 3. Binary Probe & 1D 3-State Hurdle / Bilinear
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
    m_log = LogiCA3StateTrainer(seed=seed, epochs=22).fit(tr_sp, va_sp, va_df, w_tr)

    for bk, (esp, _, _, _, _) in eval_bundles.items():
        xm_e = build_odd_linear_matrix([esp.esmc_ab_1, esp.esmc_ag_1], [esp.esmc_ab_2, esp.esmc_ag_2], include_unit_direction=True)
        xr_e = build_odd_linear_matrix([esp.esmc_ab_1, esp.esmc_ag_1], [esp.esmc_ab_2, esp.esmc_ag_2], include_unit_direction=False)
        dms = m_ms.predict(xm_e)
        preds_by_bundle[bk]["ESMC-300M (Multi-Scale Binary)"] = (dms, np.abs(dms))
        preds_by_bundle[bk]["ESMC-300M (3-State Hurdle)"] = hurd_e.predict_3state(esp, e2h.predict(esp, xm_e, xr_e))
        preds_by_bundle[bk]["ESMC+AbLang2+LLR (Bilinear)"] = m_bil.predict_3state(esp)
        preds_by_bundle[bk]["LogiCA (1D Logit Cross-Attn)"] = m_log.predict_3state(esp)

    # 4. Supervised 3D & Multi-Chain Models (`SaProt`, `MINT`, `ANTIPASTI`, `GearBind`, `WALLE-Affinity`)
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

    for bk, (esp, esf, _, _, gb_e) in eval_bundles.items():
        preds_by_bundle[bk]["SaProt-650M (3-State Hurdle)"] = pred_sap(
            esp, esf.saprot_ab_1, esf.saprot_ab_2, esf.saprot_ag_1, esf.saprot_ag_2,
            np.concatenate([esf.zs_dir["saprot_zs"][:, None], esp.biophys_odd], axis=1), esf.struct_even[:, :9],
        )
        preds_by_bundle[bk]["MINT-650M (3-State Hurdle)"] = pred_mint(
            esp, esf.mint_ab_1, esf.mint_ab_2, esf.mint_ag_1, esf.mint_ag_2,
            np.concatenate([esf.zs_dir["mint_zs"][:, None], esp.biophys_odd], axis=1), np.zeros((len(esp.y), 1), dtype=np.float32),
        )
        preds_by_bundle[bk]["ANTIPASTI (Normal-Mode 3D CNN)"] = m_anti.predict_3state(esf, esp)
        preds_by_bundle[bk]["GearBind (3D Geometric GNN)"] = m_gb.predict_3state(esf, _precomp_g=gb_e)
        preds_by_bundle[bk]["WALLE-Affinity (3-State 3D Graph)"] = pred_walle(esf, esp)

    return preds_by_bundle


_G12: dict[str, Any] = {}


def _slice_gb_tup(gb_t: tuple, ix: np.ndarray) -> tuple:
    (a1, e1, m1), (a2, e2, m2) = gb_t
    return ((a1[ix], e1[ix], m1[ix]), (a2[ix], e2[ix], m2[ix]))


def _worker_loaco_cluster(job: tuple[int, str]) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, dict, dict | None, float]:
    ci, cl = job
    torch.set_num_threads(1)
    t_c = time.time()
    loc_c1 = _G12["loc_c1"]
    cand_c1 = _G12["cand_c1"]
    pnl_c1 = _G12["pnl_c1"]
    loc_df = _G12["loc_df"]
    cand_df = _G12["cand_df"]
    pnl_pairs_df = _G12["pnl_pairs_df"]
    kd_lookup = _G12["kd_lookup"]
    pool_c1 = _G12["pool_c1"]
    pool_c2 = _G12["pool_c2"]
    pool_l1 = _G12["pool_l1"]
    pool_l2 = _G12["pool_l2"]
    is_loc_pool = _G12["is_loc_pool"]
    pool_df = _G12["pool_df"]
    pool_sp = _G12["pool_sp"]
    pool_sf = _G12["pool_sf"]
    xb_pool = _G12["xb_pool"]
    xk_pool = _G12["xk_pool"]
    gb_pool = _G12["gb_pool"]
    loc_sp_only = _G12["loc_sp_only"]
    loc_sf_unb = _G12["loc_sf_unb"]
    xb_loc = _G12["xb_loc"]
    xk_loc = _G12["xk_loc"]
    gb_loc_unb = _G12["gb_loc_unb"]
    cand_sp = _G12["cand_sp"]
    cand_sf = _G12["cand_sf"]
    xb_cand = _G12["xb_cand"]
    xk_cand = _G12["xk_cand"]
    gb_cand = _G12["gb_cand"]
    pnl_sp = _G12["pnl_sp"]
    pnl_sf = _G12["pnl_sf"]
    xb_pnl = _G12["xb_pnl"]
    xk_pnl = _G12["xk_pnl"]
    gb_pnl = _G12["gb_pnl"]
    ag_seq_by_cl = _G12["ag_seq_by_cl"]
    a2s = _G12["a2s"]
    scafs_orig = _G12["scafs_orig"]

    loc_ix = np.flatnonzero(loc_c1 == cl)
    cand_ix = np.flatnonzero(cand_c1 == cl)
    pnl_ix = np.flatnonzero(pnl_c1 == cl)

    held_ids = set()
    if len(loc_ix):
        held_ids.update(loc_df.iloc[loc_ix]["AbAgID_1"])
        held_ids.update(loc_df.iloc[loc_ix]["AbAgID_2"])
    if len(cand_ix):
        held_ids.update(cand_df.iloc[cand_ix]["AbAgID_1"])
        held_ids.update(cand_df.iloc[cand_ix]["AbAgID_2"])
    if len(pnl_ix):
        held_ids.update(pnl_pairs_df.iloc[pnl_ix]["AbAgID_1"])
        held_ids.update(pnl_pairs_df.iloc[pnl_ix]["AbAgID_2"])
    te_lins = {str(kd_lookup.loc[a, "Ab_lineage"]) for a in held_ids}

    keep_m = (pool_c1 != cl) & (pool_c2 != cl) & (~np.isin(pool_l1, list(te_lins))) & (~np.isin(pool_l2, list(te_lins)))
    kp_idx = np.flatnonzero(keep_m)
    rem_loc_cls = sorted(set(pool_c1[kp_idx[is_loc_pool[kp_idx]]]))
    val_cls = set(rem_loc_cls[1::5]) if len(rem_loc_cls) >= 5 else set(rem_loc_cls[:1])
    is_v = np.isin(pool_c1[kp_idx], list(val_cls)) | np.isin(pool_c2[kp_idx], list(val_cls))
    val_lins = set(pool_l1[kp_idx[is_v]]) | set(pool_l2[kp_idx[is_v]])
    is_tr = (~is_v) & (~np.isin(pool_l1[kp_idx], list(val_lins))) & (~np.isin(pool_l2[kp_idx], list(val_lins)))

    tr_idx, va_idx = kp_idx[is_tr], kp_idx[is_v]

    bundles: dict[str, tuple] = {}
    if len(loc_ix):
        pool_loc_indices = np.flatnonzero(is_loc_pool)[loc_ix]
        bundles["loc"] = (
            pool_sp.subset(pool_loc_indices),
            pool_sf.subset(pool_loc_indices),
            xb_pool[pool_loc_indices],
            xk_pool[pool_loc_indices],
            _slice_gb_tup(gb_pool, pool_loc_indices),
        )
        bundles["loc_unb"] = (
            loc_sp_only.subset(loc_ix),
            loc_sf_unb.subset(loc_ix),
            xb_loc[loc_ix],
            xk_loc[loc_ix],
            _slice_gb_tup(gb_loc_unb, loc_ix),
        )
    if len(cand_ix):
        bundles["cand"] = (
            cand_sp.subset(cand_ix),
            cand_sf.subset(cand_ix),
            xb_cand[cand_ix],
            xk_cand[cand_ix],
            _slice_gb_tup(gb_cand, cand_ix),
        )
    if len(pnl_ix):
        bundles["pnl"] = (
            pnl_sp.subset(pnl_ix),
            pnl_sf.subset(pnl_ix),
            xb_pnl[pnl_ix],
            xk_pnl[pnl_ix],
            _slice_gb_tup(gb_pnl, pnl_ix),
        )

    res_b = fit_and_predict_cluster(
        pool_df.iloc[tr_idx].reset_index(drop=True),
        pool_df.iloc[va_idx].reset_index(drop=True),
        pool_sp.subset(tr_idx),
        pool_sp.subset(va_idx),
        pool_sf.subset(tr_idx),
        pool_sf.subset(va_idx),
        xb_pool[tr_idx],
        xb_pool[va_idx],
        xk_pool[tr_idx],
        xk_pool[va_idx],
        _slice_gb_tup(gb_pool, tr_idx),
        _slice_gb_tup(gb_pool, va_idx),
        bundles,
        seed=42 + ci,
    )

    meta_cl = None
    if len(loc_ix):
        sub_l = loc_df.iloc[loc_ix]
        grps_l = sorted(sub_l["Ag_group_1"].unique())
        cc_frac = float(pool_sf.is_cocrystal[np.flatnonzero(is_loc_pool)[loc_ix]].mean())
        s_te = str(ag_seq_by_cl[cl])
        tr_cl_set = set(pool_c1[tr_idx])
        max_tr_id = max(
            (1.0 - Levenshtein.distance(s_te, str(ag_seq_by_cl[c2])) / max(len(s_te), len(str(ag_seq_by_cl[c2])), 1) for c2 in tr_cl_set),
            default=0.0,
        )
        sids_cl = {a2s[a] for a in sub_l["AbAgID_1"]}
        mean_dsasa = float(np.mean([scafs_orig[s].get("buried_sasa", 0.0) for s in sids_cl]))
        meta_cl = {
            "ag_groups": grps_l,
            "ag_type": str(sub_l["Ag_type"].iloc[0]),
            "n_pairs": int(len(sub_l)),
            "n_impactful": int((sub_l["label"] != 0).sum()),
            "n_neutral": int((sub_l["label"] == 0).sum()),
            "cocrystal_frac": round(cc_frac, 4),
            "max_train_ag_identity": round(float(max_tr_id), 4),
            "mean_buried_sasa": round(mean_dsasa, 1),
        }
    return cl, loc_ix, cand_ix, pnl_ix, res_b, meta_cl, time.time() - t_c


def main() -> None:
    t0 = time.time()
    root = get_repo_root()
    bdir = root / "data" / "benchmark"
    edir = root / "data" / "embeddings"
    sdir = root / "data" / "structures"
    res_dir = root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    print("Loading master KD table, splits, and PLM / 3D caches...")
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

    # 1. Full `L1–L3` + `Global` pool (`24,752` pairs; `12,267` L1–L3 across 22 `<50%` clusters)
    pool_df = pd.concat(
        [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")],
        ignore_index=True,
    )
    is_loc_pool = pool_df["Category"].isin(["L1", "L2", "L3"]).to_numpy()
    loc_df = pool_df.iloc[np.flatnonzero(is_loc_pool)].reset_index(drop=True)

    # 2. All 3D+PLM cached `Candidate` pairs (`3,629` pairs across `19` `<50%` clusters)
    cand_raw = pd.concat(
        [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}_candidates.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")],
        ignore_index=True,
    )
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

    cand_ok = [_is_fully_cached(a) and _is_fully_cached(b) for a, b in zip(cand_raw["AbAgID_1"], cand_raw["AbAgID_2"])]
    cand_df = cand_raw[cand_ok].reset_index(drop=True)

    # 3. All cached Unfiltered Pointwise Variant Panels (`6 <= len <= 200`, `22` panels)
    pw_raw = pd.concat(
        [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}_pointwise.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")],
        ignore_index=True,
    )
    pw_cached = pw_raw[[_is_fully_cached(a) for a in pw_raw["AbAgID"]]].copy()
    pw_panels = [g for (_, _), g in pw_cached.groupby(["Ag_group", "Dataset"], sort=True) if 6 <= len(g) <= 200 and g["binding_score"].nunique() >= 3]
    pw_df = pd.concat(pw_panels, ignore_index=True)
    pnl_pairs_df = build_panel_tournament_pairs(pw_df)

    # 4. Build unbound IgFold shadow cache for all 47 solved co-crystal scaffolds in `L1–L3`!
    a2s = complex_cache["abag_to_scaffold"]
    scafs_orig = complex_cache["scaffolds"]
    cocrys_sids = sorted({a2s[a] for a in set(loc_df["AbAgID_1"]).union(loc_df["AbAgID_2"]) if scafs_orig[a2s[a]]["is_cocrystal"]})
    cc_unb, inv_unb, sap_unb = build_or_load_unbound_ablation_cache(
        root, cocrys_sids, complex_cache, inv_cache, saprot_cache, loc_df, kd_lookup
    )

    print("Assembling pair tensors, 3D features, and precomputed GearBind graphs...")
    pool_sp = assemble_split_tensors(pool_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)
    pool_sf = assemble_structure_pair_features(pool_df, kd_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)

    cand_sp = assemble_split_tensors(cand_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)
    cand_sf = assemble_structure_pair_features(cand_df, kd_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)

    pnl_sp = assemble_split_tensors(pnl_pairs_df, kd_lookup, esmc_cache, ablang_cache, esm2_cache)
    pnl_sf = assemble_structure_pair_features(pnl_pairs_df, kd_lookup, complex_cache, inv_cache, saprot_cache, mint_cache, True)

    # Assemble unbound-IgFold shadow features on `loc_df` for Controlled Paired Ablation!
    loc_sp_only = pool_sp.subset(np.flatnonzero(is_loc_pool))
    loc_sf_unb = assemble_structure_pair_features(loc_df, kd_lookup, cc_unb, inv_unb, sap_unb, mint_cache, True)
    del esmc_cache, ablang_cache, esm2_cache, inv_cache, saprot_cache, mint_cache, cc_unb, inv_unb, sap_unb
    gc.collect()

    all_used_ids = sorted(
        set(pool_df["AbAgID_1"]).union(pool_df["AbAgID_2"])
        .union(cand_df["AbAgID_1"]).union(cand_df["AbAgID_2"])
        .union(pnl_pairs_df["AbAgID_1"]).union(pnl_pairs_df["AbAgID_2"])
    )
    sub_assays = kd_lookup.loc[all_used_ids].reset_index()
    bio_map = build_complex_handcrafted_table(sub_assays, include_dipeptides=False)
    kmer_map = build_complex_handcrafted_table(sub_assays, include_dipeptides=True)

    def _hc_mat(d: pd.DataFrame, fmap: dict) -> np.ndarray:
        return build_odd_linear_matrix(
            [np.stack([fmap[k] for k in d["AbAgID_1"]])],
            [np.stack([fmap[k] for k in d["AbAgID_2"]])],
            include_unit_direction=True,
        )

    xb_pool, xk_pool = _hc_mat(pool_df, bio_map), _hc_mat(pool_df, kmer_map)
    xb_cand, xk_cand = _hc_mat(cand_df, bio_map), _hc_mat(cand_df, kmer_map)
    xb_pnl, xk_pnl = _hc_mat(pnl_pairs_df, bio_map), _hc_mat(pnl_pairs_df, kmer_map)
    xb_loc, xk_loc = xb_pool[is_loc_pool], xk_pool[is_loc_pool]

    gb_pool = (
        GearBind3StateTrainer._precompute_graphs(pool_sf.patch_ca_1, pool_sf.patch_cb_1, pool_sf.patch_chain_1, pool_sf.patch_seqpos_1),
        GearBind3StateTrainer._precompute_graphs(pool_sf.patch_ca_2, pool_sf.patch_cb_2, pool_sf.patch_chain_2, pool_sf.patch_seqpos_2),
    )
    gb_cand = (
        GearBind3StateTrainer._precompute_graphs(cand_sf.patch_ca_1, cand_sf.patch_cb_1, cand_sf.patch_chain_1, cand_sf.patch_seqpos_1),
        GearBind3StateTrainer._precompute_graphs(cand_sf.patch_ca_2, cand_sf.patch_cb_2, cand_sf.patch_chain_2, cand_sf.patch_seqpos_2),
    )
    gb_pnl = (
        GearBind3StateTrainer._precompute_graphs(pnl_sf.patch_ca_1, pnl_sf.patch_cb_1, pnl_sf.patch_chain_1, pnl_sf.patch_seqpos_1),
        GearBind3StateTrainer._precompute_graphs(pnl_sf.patch_ca_2, pnl_sf.patch_cb_2, pnl_sf.patch_chain_2, pnl_sf.patch_seqpos_2),
    )
    gb_loc_unb = (
        GearBind3StateTrainer._precompute_graphs(loc_sf_unb.patch_ca_1, loc_sf_unb.patch_cb_1, loc_sf_unb.patch_chain_1, loc_sf_unb.patch_seqpos_1),
        GearBind3StateTrainer._precompute_graphs(loc_sf_unb.patch_ca_2, loc_sf_unb.patch_cb_2, loc_sf_unb.patch_chain_2, loc_sf_unb.patch_seqpos_2),
    )

    def _slice_gb(gb_t: tuple, ix: np.ndarray) -> tuple:
        (a1, e1, m1), (a2, e2, m2) = gb_t
        return ((a1[ix], e1[ix], m1[ix]), (a2[ix], e2[ix], m2[ix]))

    pool_c1 = pool_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
    pool_c2 = pool_df["AbAgID_2"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
    pool_l1 = pool_df["AbAgID_1"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()
    pool_l2 = pool_df["AbAgID_2"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()

    loc_c1 = loc_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
    cand_c1 = cand_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
    pnl_c1 = pnl_pairs_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()

    loc_clusters = sorted(c for c, cnt in pd.Series(loc_c1).value_counts().items() if cnt >= 5)
    cand_clusters = sorted(c for c, cnt in pd.Series(cand_c1).value_counts().items() if cnt >= 5)
    pnl_clusters = sorted(set(pnl_c1))
    all_eval_clusters = sorted(set(loc_clusters) | set(cand_clusters) | set(pnl_clusters))

    print(
        f"\n=== Running Full Zero-Leakage LOACO across {len(all_eval_clusters)} `<50%` Antigen Clusters "
        f"(Local={len(loc_clusters)}, Candidate={len(cand_clusters)}, Panels={len(pnl_panels := pw_panels)}) ==="
    )

    oof_loc_d = {m: np.zeros(len(loc_df)) for m, _ in LOACO_MODELS}
    oof_loc_i = {m: np.zeros(len(loc_df)) for m, _ in LOACO_MODELS}
    oof_loc_unb_d = {m: np.zeros(len(loc_df)) for m, _ in LOACO_MODELS}
    oof_loc_unb_i = {m: np.zeros(len(loc_df)) for m, _ in LOACO_MODELS}
    oof_cand_d = {m: np.zeros(len(cand_df)) for m, _ in LOACO_MODELS}
    oof_cand_i = {m: np.zeros(len(cand_df)) for m, _ in LOACO_MODELS}
    oof_pnl_d = {m: np.zeros(len(pnl_pairs_df)) for m, _ in LOACO_MODELS}

    cluster_meta: dict[str, dict] = {}
    ag_seq_by_cl = kd_raw.groupby("Ag_cluster")["Ag_seq"].first().to_dict()

    _G12.update(
        loc_c1=loc_c1, cand_c1=cand_c1, pnl_c1=pnl_c1,
        loc_df=loc_df, cand_df=cand_df, pnl_pairs_df=pnl_pairs_df, kd_lookup=kd_lookup,
        pool_c1=pool_c1, pool_c2=pool_c2, pool_l1=pool_l1, pool_l2=pool_l2, is_loc_pool=is_loc_pool,
        pool_df=pool_df, pool_sp=pool_sp, pool_sf=pool_sf, xb_pool=xb_pool, xk_pool=xk_pool, gb_pool=gb_pool,
        loc_sp_only=loc_sp_only, loc_sf_unb=loc_sf_unb, xb_loc=xb_loc, xk_loc=xk_loc, gb_loc_unb=gb_loc_unb,
        cand_sp=cand_sp, cand_sf=cand_sf, xb_cand=xb_cand, xk_cand=xk_cand, gb_cand=gb_cand,
        pnl_sp=pnl_sp, pnl_sf=pnl_sf, xb_pnl=xb_pnl, xk_pnl=xk_pnl, gb_pnl=gb_pnl,
        ag_seq_by_cl=ag_seq_by_cl, a2s=a2s, scafs_orig=scafs_orig,
    )
    n_w12 = min(13, len(all_eval_clusters))
    done_c = 0
    with mp.get_context("fork").Pool(n_w12, maxtasksperchild=2) as pool:
        for cl, loc_ix, cand_ix, pnl_ix, res_b, meta_cl, dt_c in pool.imap_unordered(_worker_loaco_cluster, list(enumerate(all_eval_clusters))):
            done_c += 1
            for m, _ in LOACO_MODELS:
                if "loc" in res_b:
                    oof_loc_d[m][loc_ix], oof_loc_i[m][loc_ix] = res_b["loc"][m]
                    oof_loc_unb_d[m][loc_ix], oof_loc_unb_i[m][loc_ix] = res_b["loc_unb"][m]
                if "cand" in res_b:
                    oof_cand_d[m][cand_ix], oof_cand_i[m][cand_ix] = res_b["cand"][m]
                if "pnl" in res_b:
                    oof_pnl_d[m][pnl_ix] = res_b["pnl"][m][0]
            if meta_cl is not None:
                cluster_meta[cl] = meta_cl
            print(
                f"  [{done_c:2d}/{len(all_eval_clusters)}] {cl} (loc={len(loc_ix)}, cand={len(cand_ix)}, pnl={len(pnl_ix)}) "
                f"done in {dt_c:.1f}s",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Aggregate Part A: LOACO-Local (22 clusters), LOACO-Candidate (19 clusters), LOACO-Panels (22 panels)
    # ------------------------------------------------------------------
    loc_lbl = loc_df["label"].to_numpy(dtype=int)
    loc_bd = loc_df["binding_diff"].to_numpy(dtype=float)
    cand_lbl = cand_df["label"].to_numpy(dtype=int)
    cand_bd = cand_df["binding_diff"].to_numpy(dtype=float)

    loaco_summary: dict[str, dict] = {}
    cocrys_clusters = [c for c in loc_clusters if cluster_meta[c]["cocrystal_frac"] >= 0.5]
    igfold_clusters = [c for c in loc_clusters if cluster_meta[c]["cocrystal_frac"] < 0.5]

    for m_name, m_fam in LOACO_MODELS:
        per_cl_loc = {}
        per_cl_loc_unb = {}
        for cl in loc_clusters:
            ix = np.flatnonzero(loc_c1 == cl)
            per_cl_loc[cl] = compute_3state_metrics(loc_lbl[ix], loc_bd[ix], oof_loc_d[m_name][ix], oof_loc_i[m_name][ix])
            per_cl_loc_unb[cl] = compute_3state_metrics(loc_lbl[ix], loc_bd[ix], oof_loc_unb_d[m_name][ix], oof_loc_unb_i[m_name][ix])

        per_cl_cand = {}
        for cl in cand_clusters:
            ix = np.flatnonzero(cand_c1 == cl)
            per_cl_cand[cl] = compute_3state_metrics(cand_lbl[ix], cand_bd[ix], oof_cand_d[m_name][ix], oof_cand_i[m_name][ix])

        loc_boot = compute_cluster_bootstrap_summary(per_cl_loc, min_class_pairs=2, n_bootstrap=2000, seed=42)
        cand_boot = compute_cluster_bootstrap_summary(per_cl_cand, min_class_pairs=2, n_bootstrap=2000, seed=42)
        cocrys_boot = compute_cluster_bootstrap_summary({c: per_cl_loc[c] for c in cocrys_clusters}, min_class_pairs=2, n_bootstrap=2000, seed=42)
        igfold_nat_boot = compute_cluster_bootstrap_summary({c: per_cl_loc[c] for c in igfold_clusters}, min_class_pairs=2, n_bootstrap=2000, seed=42)
        cocrys_to_unb_boot = compute_cluster_bootstrap_summary({c: per_cl_loc_unb[c] for c in cocrys_clusters}, min_class_pairs=2, n_bootstrap=2000, seed=42)

        t_scores = tournament_scores_from_pairs(pnl_pairs_df, oof_pnl_d[m_name])
        pnl_eval = evaluate_pointwise_spearman(pw_df, t_scores, min_group_size=6)

        pooled_loc = compute_3state_metrics(loc_lbl, loc_bd, oof_loc_d[m_name], oof_loc_i[m_name])
        loaco_summary[m_name] = {
            "family": m_fam,
            "loaco_local_22_clusters": {
                **loc_boot,
                "pooled_auc_dir": pooled_loc["auc_dir"],
                "pooled_auc_impact": pooled_loc["auc_impact"],
                "pooled_auc_ordinal": pooled_loc["auc_ordinal"],
                "per_cluster": per_cl_loc,
            },
            "loaco_candidate_19_clusters": {
                **cand_boot,
                "per_cluster": per_cl_cand,
            },
            "loaco_panels_22": pnl_eval,
            "regime1_bound_cocrystal_16_clusters": cocrys_boot,
            "regime2_natural_unbound_igfold_6_clusters": igfold_nat_boot,
            "controlled_ablation_same_16_clusters_unbound_igfold": cocrys_to_unb_boot,
        }

    # Target difficulty correlates across the 20 directional LOACO-Local clusters
    correlates = {}
    for m_key in ("PRODIGY+SASA (3D Energy ZS)", "ESMC-300M (3-State Hurdle)", "SaProt-650M (3-State Hurdle)", "ANTIPASTI (Normal-Mode 3D CNN)", "GearBind (3D Geometric GNN)"):
        pc = loaco_summary[m_key]["loaco_local_22_clusters"]["per_cluster"]
        elig = [c for c in loc_clusters if pc[c]["n_impactful"] >= 2 and pc[c]["auc_ordinal"] is not None]
        ords = [pc[c]["auc_ordinal"] for c in elig]
        cc_v = [cluster_meta[c]["cocrystal_frac"] for c in elig]
        id_v = [cluster_meta[c]["max_train_ag_identity"] for c in elig]
        sa_v = [cluster_meta[c]["mean_buried_sasa"] for c in elig]
        correlates[m_key] = {
            "rho_vs_cocrystal": round(float(spearmanr(ords, cc_v)[0]), 3),
            "rho_vs_nearest_train_ag_id": round(float(spearmanr(ords, id_v)[0]), 3),
            "rho_vs_buried_sasa": round(float(spearmanr(ords, sa_v)[0]), 3),
        }

    payload = {
        "benchmark": "AbRank v2.0.0",
        "protocol": "Leave-One-Antigen-Cluster-Out (LOACO: <50% Ag + <80% Ab + Nested Val)",
        "n_local_clusters": len(loc_clusters),
        "n_local_pairs": int(len(loc_df)),
        "n_local_impactful": int((loc_lbl != 0).sum()),
        "n_candidate_clusters": len(cand_clusters),
        "n_candidate_pairs": int(len(cand_df)),
        "n_panels": int(len(pnl_panels)),
        "n_cocrystal_clusters": len(cocrys_clusters),
        "n_unbound_igfold_clusters": len(igfold_clusters),
        "cluster_metadata": cluster_meta,
        "target_difficulty_correlates": correlates,
        "models": loaco_summary,
    }
    out_json = res_dir / "loaco_evaluation.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Write standalone Markdown summary
    md = [
        "# AbRank v2 — Leave-One-Antigen-Cluster-Out (`LOACO`) & Controlled Structure Ablation",
        "",
        f"Evaluates **20 models across all 6 families** over **`{len(loc_clusters)}` independent `<50%`-identity local antigen clusters (`{len(loc_df):,}` `L1–L3` pairs; `{(loc_lbl != 0).sum():,}` impactful `±1`)**, **`{len(cand_clusters)}` candidate clusters (`{len(cand_df):,}` pairs)**, and **`{len(pnl_panels)}` unfiltered variant panels (`{len(pw_df):,}` variants)** with **2,000-replicate Cluster-Resampled Bootstrap 95% CIs**:",
        "",
        "## 1. Main `LOACO` Results (`Cluster-Macro Mean` [Cluster-Bootstrap 95% CI])",
        "",
        "| Model | Family | `LOACO-Local` Sign (`20/22`) | `LOACO-Local` Impact (`19/22`) | `LOACO-Local` Ordinal ± SD (`[CI95]`) | `LOACO-Cand` Ord (`17/19`) | `LOACO-Panels` Spearman `ρ` / Top-Bind (`25`) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name, m_fam in LOACO_MODELS:
        ms = loaco_summary[m_name]
        lc = ms["loaco_local_22_clusters"]
        cd = ms["loaco_candidate_19_clusters"]
        pn = ms["loaco_panels_22"]
        ci_o = lc["ordinal"]["cluster_ci95"]
        md.append(
            f"| **{m_name}** | {m_fam} | "
            f"`{lc['auc_dir']:.3f}` | `{lc['auc_impact']:.3f}` | "
            f"**`{lc['auc_ordinal']:.3f}`** ± `{lc['ordinal']['sd']:.3f}` [`{ci_o[0]:.2f}`, `{ci_o[1]:.2f}`] | "
            f"`{cd['auc_ordinal']:.3f}` | `{pn['mean_spearman_rho']:+.3f}` / `{pn['mean_top_binder_auc']:.3f}` |"
        )

    md.extend([
        "",
        "## 2. Controlled Paired Structure Ablation on the Exact Same 19 Co-Crystal Clusters (`18` Directional; `Bound Co-Crystal` vs. `Unbound IgFold Fv + RCSB Ag`)",
        "",
        "| Model | Bound Co-Crystal (`Sign` / `Imp` / `Ord`) | Controlled Unbound `IgFold` (`Sign` / `Imp` / `Ord`) | Privileged Bound-Interface ΔOrd | Natural Unbound Discovery Clusters (`3` Clusters) Ord |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for m_name, _ in LOACO_MODELS:
        if m_name == "Random Baseline":
            continue
        ms = loaco_summary[m_name]
        b_cc = ms["regime1_bound_cocrystal_16_clusters"]
        b_un = ms["controlled_ablation_same_16_clusters_unbound_igfold"]
        b_nat = ms["regime2_natural_unbound_igfold_6_clusters"]
        d_ord = b_cc["auc_ordinal"] - b_un["auc_ordinal"]
        md.append(
            f"| **{m_name}** | `{b_cc['auc_dir']:.3f}` / `{b_cc['auc_impact']:.3f}` / **`{b_cc['auc_ordinal']:.3f}`** | "
            f"`{b_un['auc_dir']:.3f}` / `{b_un['auc_impact']:.3f}` / **`{b_un['auc_ordinal']:.3f}`** | "
            f"`{d_ord:+.3f}` | `{b_nat['auc_ordinal']:.3f}` |"
        )

    out_md = res_dir / "LOACO_EVALUATION.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nCompleted Full LOACO & Controlled Structure Ablation in {time.time() - t0:.1f}s! Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
