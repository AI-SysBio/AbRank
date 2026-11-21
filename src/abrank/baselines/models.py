"""Baseline, 3-state hurdle/ordinal, and neural ranking models for AbRank v2.

All supervised models adhere strictly to leak-free ML practices:
  - Feature scalers, PCA projections, and model weights are fit **exclusively** on `train`.
  - Hyperparameters and early-stopping checkpoints are selected **exclusively** on `val`.
  - Every directional scorer satisfies exact antisymmetry $f(1, 2) = -f(2, 1)$ and every
    impact/hot-spot scorer satisfies exact symmetry $q(1, 2) = +q(2, 1)$.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.nn.functional as F

from abrank.baselines.embeddings import pair_hash, seq_hash
from abrank.baselines.features import (
    AA_TO_IDX,
    extract_mutation_biophys_odd,
    extract_mutation_impact_even,
)
from abrank.evaluate import compute_3state_metrics, compute_per_antigen_macro_metrics


def sym_auc(y_bin: np.ndarray, preds: np.ndarray) -> float:
    """Exact orientation-symmetrized ROC-AUC."""
    if len(y_bin) == 0:
        return float("nan")
    ys = np.concatenate([y_bin, 1 - y_bin])
    ps = np.concatenate([preds, -preds])
    return float(roc_auc_score(ys, ps))


def val_balanced_objective(pairs_df: pd.DataFrame, preds: np.ndarray, is_ag_mut: np.ndarray) -> float:
    """Balanced validation AUC across Global and Per-Antigen Macro Local slices."""
    imp = (pairs_df["label"].to_numpy(dtype=int) != 0) if "label" in pairs_df.columns else np.ones(len(pairs_df), dtype=bool)
    y = (pairs_df["binding_diff"].to_numpy(dtype=float) > 0).astype(int)
    is_global = (pairs_df["Category"] == "Global").to_numpy() & imp
    is_local = (~(pairs_df["Category"] == "Global").to_numpy())

    parts = []
    if is_global.sum() >= 5:
        parts.append(sym_auc(y[is_global], preds[is_global]))
    if is_local.sum() >= 5 and "Ag_group_1" in pairs_df.columns:
        m_loc = compute_per_antigen_macro_metrics(pairs_df.iloc[np.flatnonzero(is_local)], preds[is_local], min_pairs=5)
        if m_loc["auc_dir"] is not None:
            parts.append(float(m_loc["auc_dir"]))
    else:
        for mask in (is_local & (~is_ag_mut) & imp, is_local & is_ag_mut & imp):
            if mask.sum() >= 5:
                parts.append(sym_auc(y[mask], preds[mask]))
    if parts:
        return float(np.mean(parts))
    return sym_auc(y[imp], preds[imp]) if imp.any() else 0.5


def compute_stratum_weights(pairs_df: pd.DataFrame, is_ag_mut: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights balancing Global, Local Ab-mut, Local Ag-mut, and Neutral."""
    lbl = pairs_df["label"].to_numpy(dtype=int) if "label" in pairs_df.columns else np.ones(len(pairs_df), dtype=int)
    imp = lbl != 0
    neu = lbl == 0
    is_global = (pairs_df["Category"] == "Global").to_numpy() & imp
    is_local_ab = (~(pairs_df["Category"] == "Global").to_numpy()) & (~is_ag_mut) & imp
    is_local_ag = (~(pairs_df["Category"] == "Global").to_numpy()) & is_ag_mut & imp

    w = np.ones(len(pairs_df), dtype=np.float32)
    if neu.any():
        shares = ((is_global, 0.32), (is_local_ab, 0.32), (is_local_ag, 0.16), (neu, 0.20))
    else:
        shares = ((is_global, 0.40), (is_local_ab, 0.40), (is_local_ag, 0.20))
    for mask, target_share in shares:
        cnt = int(mask.sum())
        if cnt > 0:
            w[mask] = (target_share * len(pairs_df)) / cnt
    return w / max(float(w.mean()), 1e-8)


def _sym_site_llr(s1: str, s2: str, lp1: np.ndarray, lp2: np.ndarray, max_muts: int = 20) -> tuple[float, bool]:
    """Symmetrized residue-site log-likelihood ratio at differing positions."""
    if len(s1) != len(s2) or s1 == s2:
        return 0.0, (s1 == s2)
    L = min(len(s1), lp1.shape[0], lp2.shape[0])
    diffs = [i for i in range(L) if s1[i] != s2[i]]
    if not diffs or len(diffs) > max_muts:
        return 0.0, False
    total = 0.0
    for i in diffs:
        a1 = AA_TO_IDX[s1[i]]
        a2 = AA_TO_IDX[s2[i]]
        d1 = float(lp1[i, a1]) - float(lp1[i, a2])
        d2 = float(lp2[i, a1]) - float(lp2[i, a2])
        total += 0.5 * (d1 + d2)
    return total, True


def _sym_site_abs_llr(s1: str, s2: str, lp1: np.ndarray, lp2: np.ndarray, max_muts: int = 20) -> float:
    """Non-canceling sum of per-site `|LLR|` magnitudes (`f(1,2) == +f(2,1)`)."""
    if len(s1) != len(s2) or s1 == s2:
        return 0.0
    L = min(len(s1), lp1.shape[0], lp2.shape[0])
    diffs = [i for i in range(L) if s1[i] != s2[i]]
    if not diffs or len(diffs) > max_muts:
        return 0.0
    total = 0.0
    for i in diffs:
        a1 = AA_TO_IDX[s1[i]]
        a2 = AA_TO_IDX[s2[i]]
        d1 = float(lp1[i, a1]) - float(lp1[i, a2])
        d2 = float(lp2[i, a1]) - float(lp2[i, a2])
        total += abs(0.5 * (d1 + d2))
    return total


def _unit_norm_rows(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


@dataclass
class SplitPairTensors:
    """Pre-assembled antisymmetric, symmetric-impact, and complex-level arrays for a 3-state split."""

    y: np.ndarray                  # [N] in {0, 1} (`binding_diff > 0`)
    labels: np.ndarray             # [N] in {-1, 0, +1}
    is_impactful: np.ndarray       # [N] bool (`labels != 0`)
    is_ab_mut: np.ndarray          # [N] bool
    is_ag_mut: np.ndarray          # [N] bool
    is_same_ag: np.ndarray         # [N] bool
    esmc_ab_1: np.ndarray          # [N, 2880] (VH mean, VL mean, VH cdr3)
    esmc_ab_2: np.ndarray
    esmc_ag_1: np.ndarray          # [N, 1920] (Ag mean, Ag std)
    esmc_ag_2: np.ndarray
    ablang_1: np.ndarray           # [N, 960]  (paired_mean, vh_cdr3)
    ablang_2: np.ndarray
    esm2_ab_1: np.ndarray          # [N, 1920] (VH mean, VL mean, VH cdr3; D=640)
    esm2_ab_2: np.ndarray
    esm2_ag_1: np.ndarray          # [N, 1280] (Ag mean, Ag std; D=640)
    esm2_ag_2: np.ndarray
    zs_features: np.ndarray        # [N, 10] strictly antisymmetric LLR / PLL features
    biophys_odd: np.ndarray        # [N, 18] strictly antisymmetric mutation biophysics
    impact_even: np.ndarray        # [N, 38] strictly even-symmetric (`f(1,2) == +f(2,1)`) impact features
    zs_dict: dict[str, np.ndarray] # Named zero-shot directional score vectors
    zs_impact_dict: dict[str, np.ndarray]  # Named zero-shot impact magnitude vectors

    def subset(self, idx: np.ndarray | list[int]) -> "SplitPairTensors":
        ix = np.asarray(idx)
        return SplitPairTensors(
            y=self.y[ix],
            labels=self.labels[ix],
            is_impactful=self.is_impactful[ix],
            is_ab_mut=self.is_ab_mut[ix],
            is_ag_mut=self.is_ag_mut[ix],
            is_same_ag=self.is_same_ag[ix],
            esmc_ab_1=self.esmc_ab_1[ix],
            esmc_ab_2=self.esmc_ab_2[ix],
            esmc_ag_1=self.esmc_ag_1[ix],
            esmc_ag_2=self.esmc_ag_2[ix],
            ablang_1=self.ablang_1[ix],
            ablang_2=self.ablang_2[ix],
            esm2_ab_1=self.esm2_ab_1[ix],
            esm2_ab_2=self.esm2_ab_2[ix],
            esm2_ag_1=self.esm2_ag_1[ix],
            esm2_ag_2=self.esm2_ag_2[ix],
            zs_features=self.zs_features[ix],
            biophys_odd=self.biophys_odd[ix],
            impact_even=self.impact_even[ix],
            zs_dict={k: v[ix] for k, v in self.zs_dict.items()},
            zs_impact_dict={k: v[ix] for k, v in self.zs_impact_dict.items()},
        )


def assemble_split_tensors(
    pairs_df: pd.DataFrame,
    assays_lookup: pd.DataFrame,
    esmc_cache: dict,
    ablang_cache: dict,
    esm2_cache: dict,
) -> SplitPairTensors:
    """Assemble complex embeddings, antisymmetric directional features, and symmetric impact features."""
    n = len(pairs_df)
    r1 = assays_lookup.loc[pairs_df["AbAgID_1"]]
    r2 = assays_lookup.loc[pairs_df["AbAgID_2"]]

    vh1_list = [str(s) for s in r1["Ab_heavy_chain_seq"].tolist()]
    vl1_list = ["" if pd.isna(s) else str(s) for s in r1["Ab_light_chain_seq"].tolist()]
    ag1_list = [str(s) for s in r1["Ag_seq"].tolist()]
    vh2_list = [str(s) for s in r2["Ab_heavy_chain_seq"].tolist()]
    vl2_list = ["" if pd.isna(s) else str(s) for s in r2["Ab_light_chain_seq"].tolist()]
    ag2_list = [str(s) for s in r2["Ag_seq"].tolist()]

    y = (pairs_df["binding_diff"].to_numpy(dtype=float) > 0).astype(int)
    labels = (
        pairs_df["label"].to_numpy(dtype=int)
        if "label" in pairs_df.columns
        else np.where(pairs_df["binding_diff"].to_numpy(dtype=float) > 0, 1, -1)
    )
    is_impactful = labels != 0
    is_ab_mut = np.array([(h1 != h2) or (l1 != l2) for h1, l1, h2, l2 in zip(vh1_list, vl1_list, vh2_list, vl2_list)], dtype=bool)
    is_ag_mut = np.array([a1 != a2 for a1, a2 in zip(ag1_list, ag2_list)], dtype=bool)
    is_same_ag = ~is_ag_mut

    def _gather_single(cache: dict, vh_l: list[str], vl_l: list[str], ag_l: list[str]):
        means, stds, cdr3s = cache["means"], cache["stds"], cache["cdr3s"]
        d_emb = next(iter(means.values())).shape[0]
        zero_m = np.zeros(d_emb, dtype=np.float32)
        ab_arr = np.stack(
            [
                np.concatenate([
                    means[seq_hash(h)].astype(np.float32),
                    means[seq_hash(l)].astype(np.float32) if l else zero_m,
                    cdr3s[seq_hash(h)].astype(np.float32),
                ])
                for h, l in zip(vh_l, vl_l)
            ],
            axis=0,
        )
        ag_arr = np.stack(
            [np.concatenate([means[seq_hash(a)], stds[seq_hash(a)]]).astype(np.float32) for a in ag_l],
            axis=0,
        )
        return ab_arr, ag_arr

    esmc_ab_1, esmc_ag_1 = _gather_single(esmc_cache, vh1_list, vl1_list, ag1_list)
    esmc_ab_2, esmc_ag_2 = _gather_single(esmc_cache, vh2_list, vl2_list, ag2_list)
    esm2_ab_1, esm2_ag_1 = _gather_single(esm2_cache, vh1_list, vl1_list, ag1_list)
    esm2_ab_2, esm2_ag_2 = _gather_single(esm2_cache, vh2_list, vl2_list, ag2_list)

    zero_abl = np.zeros(960, dtype=np.float32)
    ablang_1 = np.stack(
        [
            np.concatenate([ablang_cache["paired_means"][pair_hash(h, l)], ablang_cache["vh_cdr3s"][pair_hash(h, l)]]).astype(np.float32)
            if l else zero_abl
            for h, l in zip(vh1_list, vl1_list)
        ],
        axis=0,
    )
    ablang_2 = np.stack(
        [
            np.concatenate([ablang_cache["paired_means"][pair_hash(h, l)], ablang_cache["vh_cdr3s"][pair_hash(h, l)]]).astype(np.float32)
            if l else zero_abl
            for h, l in zip(vh2_list, vl2_list)
        ],
        axis=0,
    )

    zs_feats = np.zeros((n, 10), dtype=np.float32)
    zs_abs_llr = np.zeros((n, 3), dtype=np.float32)
    zs_esmc_ab = np.zeros(n, dtype=np.float32)
    zs_esmc_ag = np.zeros(n, dtype=np.float32)
    zs_ablang = np.zeros(n, dtype=np.float32)
    zs_esm2_total = np.zeros(n, dtype=np.float32)
    zs_esmc_cos = np.zeros(n, dtype=np.float32)
    zs_esm2_imp = np.zeros(n, dtype=np.float32)

    e_plls, e_lps = esmc_cache["plls"], esmc_cache["aa_lps"]
    m2_plls, m2_lps = esm2_cache["plls"], esm2_cache["aa_lps"]
    ab_plls, ab_vh_lps, ab_vl_lps = ablang_cache["plls"], ablang_cache["vh_lps"], ablang_cache["vl_lps"]
    empty_lp = np.zeros((0, 20), dtype=np.float16)

    for i in range(n):
        vh1, vl1, ag1 = vh1_list[i], vl1_list[i], ag1_list[i]
        vh2, vl2, ag2 = vh2_list[i], vl2_list[i], ag2_list[i]
        has_l1, has_l2 = bool(vl1), bool(vl2)
        kh1, ka1 = seq_hash(vh1), seq_hash(ag1)
        kh2, ka2 = seq_hash(vh2), seq_hash(ag2)
        kl1 = seq_hash(vl1) if has_l1 else ""
        kl2 = seq_hash(vl2) if has_l2 else ""
        kp1 = pair_hash(vh1, vl1) if has_l1 else ""
        kp2 = pair_hash(vh2, vl2) if has_l2 else ""

        pll_l1 = e_plls[kl1] if has_l1 else e_plls[kh1]
        pll_l2 = e_plls[kl2] if has_l2 else e_plls[kh2]
        d_esmc_pll_ab = 0.5 * ((e_plls[kh1] + pll_l1) - (e_plls[kh2] + pll_l2))
        d_esmc_pll_ag = e_plls[ka1] - e_plls[ka2]
        d_esmc_sum_ab = (len(vh1) * e_plls[kh1] + (len(vl1) * e_plls[kl1] if has_l1 else 0.0)) - (
            len(vh2) * e_plls[kh2] + (len(vl2) * e_plls[kl2] if has_l2 else 0.0)
        )
        d_esmc_sum_ag = len(ag1) * e_plls[ka1] - len(ag2) * e_plls[ka2]

        site_vh, ok_vh = _sym_site_llr(vh1, vh2, e_lps[kh1], e_lps[kh2])
        site_vl, ok_vl = _sym_site_llr(vl1, vl2, e_lps[kl1] if has_l1 else empty_lp, e_lps[kl2] if has_l2 else empty_lp)
        site_ag, ok_ag = _sym_site_llr(ag1, ag2, e_lps[ka1], e_lps[ka2])
        site_ab_esmc = site_vh + site_vl

        abs_site_esmc_ab = _sym_site_abs_llr(vh1, vh2, e_lps[kh1], e_lps[kh2]) + (
            _sym_site_abs_llr(vl1, vl2, e_lps[kl1], e_lps[kl2]) if (has_l1 and has_l2) else 0.0
        )
        abs_site_esmc_ag = _sym_site_abs_llr(ag1, ag2, e_lps[ka1], e_lps[ka2])

        if has_l1 and has_l2:
            d_ablang_pll = ab_plls[kp1] - ab_plls[kp2]
            d_ablang_sum = (len(vh1) + len(vl1)) * ab_plls[kp1] - (len(vh2) + len(vl2)) * ab_plls[kp2]
            ab_site_h, ok_abh = _sym_site_llr(vh1, vh2, ab_vh_lps[kp1], ab_vh_lps[kp2])
            ab_site_l, ok_abl = _sym_site_llr(vl1, vl2, ab_vl_lps[kp1], ab_vl_lps[kp2])
            site_ab_ablang = ab_site_h + ab_site_l
            abs_site_ablang = _sym_site_abs_llr(vh1, vh2, ab_vh_lps[kp1], ab_vh_lps[kp2]) + _sym_site_abs_llr(vl1, vl2, ab_vl_lps[kp1], ab_vl_lps[kp2])
        else:
            d_ablang_pll = 0.0
            d_ablang_sum = 0.0
            ok_abh, ok_abl = False, False
            site_ab_ablang = 0.0
            abs_site_ablang = 0.0

        m2_vh, m2_ok_vh = _sym_site_llr(vh1, vh2, m2_lps[kh1], m2_lps[kh2])
        m2_vl, m2_ok_vl = _sym_site_llr(vl1, vl2, m2_lps[kl1] if has_l1 else empty_lp, m2_lps[kl2] if has_l2 else empty_lp)
        m2_ag, m2_ok_ag = _sym_site_llr(ag1, ag2, m2_lps[ka1], m2_lps[ka2])
        m2_pll_l1 = m2_plls[kl1] if has_l1 else m2_plls[kh1]
        m2_pll_l2 = m2_plls[kl2] if has_l2 else m2_plls[kh2]
        m2_ab = (m2_vh + m2_vl) if (m2_ok_vh and m2_ok_vl) else 0.5 * ((m2_plls[kh1] + m2_pll_l1) - (m2_plls[kh2] + m2_pll_l2))
        m2_ag_s = m2_ag if m2_ok_ag else (m2_plls[ka1] - m2_plls[ka2])
        abs_m2 = (
            _sym_site_abs_llr(vh1, vh2, m2_lps[kh1], m2_lps[kh2])
            + (_sym_site_abs_llr(vl1, vl2, m2_lps[kl1], m2_lps[kl2]) if (has_l1 and has_l2) else 0.0)
            + _sym_site_abs_llr(ag1, ag2, m2_lps[ka1], m2_lps[ka2])
        )

        zs_esmc_ab[i] = site_ab_esmc if (ok_vh and ok_vl) else d_esmc_pll_ab
        zs_esmc_ag[i] = site_ag if ok_ag else d_esmc_pll_ag
        zs_ablang[i] = (site_ab_ablang if (ok_abh and ok_abl) else d_ablang_pll) if (has_l1 and has_l2) else 0.0
        zs_esm2_total[i] = m2_ab + m2_ag_s
        zs_esm2_imp[i] = abs_m2 if abs_m2 > 0 else abs(zs_esm2_total[i])

        v_ab1 = 0.5 * (esmc_ab_1[i, :960] + esmc_ab_1[i, 960:1920]) if has_l1 else esmc_ab_1[i, :960]
        v_ab2 = 0.5 * (esmc_ab_2[i, :960] + esmc_ab_2[i, 960:1920]) if has_l2 else esmc_ab_2[i, :960]
        v_ag1 = esmc_ag_1[i, :960]
        v_ag2 = esmc_ag_2[i, :960]
        c1 = float(np.dot(v_ab1, v_ag1) / (np.linalg.norm(v_ab1) * np.linalg.norm(v_ag1) + 1e-8))
        c2 = float(np.dot(v_ab2, v_ag2) / (np.linalg.norm(v_ab2) * np.linalg.norm(v_ag2) + 1e-8))
        zs_esmc_cos[i] = c1 - c2

        zs_feats[i] = [
            d_esmc_pll_ab,
            d_esmc_pll_ag,
            d_esmc_sum_ab / 10.0,
            d_esmc_sum_ag / 10.0,
            site_ab_esmc,
            site_ag,
            d_ablang_pll,
            d_ablang_sum / 10.0,
            site_ab_ablang,
            zs_esmc_cos[i],
        ]
        zs_abs_llr[i] = [
            abs_site_esmc_ab if abs_site_esmc_ab > 0 else abs(d_esmc_pll_ab),
            abs_site_esmc_ag if abs_site_esmc_ag > 0 else abs(d_esmc_pll_ag),
            abs_site_ablang if abs_site_ablang > 0 else abs(d_ablang_pll),
        ]

    biophys_odd = extract_mutation_biophys_odd(vh1_list, vl1_list, ag1_list, vh2_list, vl2_list, ag2_list)
    mut_even = extract_mutation_impact_even(vh1_list, vl1_list, ag1_list, vh2_list, vl2_list, ag2_list)

    # Embedding perturbation norms (strictly even: ||b1 - b2|| == ||b2 - b1||)
    emb_norms = np.stack(
        [
            np.log1p(np.linalg.norm(esmc_ab_1 - esmc_ab_2, axis=1)),
            np.log1p(np.linalg.norm(esmc_ab_1[:, 1920:] - esmc_ab_2[:, 1920:], axis=1)),
            np.log1p(np.linalg.norm(ablang_1 - ablang_2, axis=1)),
            np.log1p(np.linalg.norm(esmc_ag_1 - esmc_ag_2, axis=1)),
        ],
        axis=1,
    ).astype(np.float32)

    impact_even = np.concatenate([mut_even, np.abs(zs_feats), zs_abs_llr, emb_norms], axis=1).astype(np.float32)

    has_both_l = np.array([bool(l1) and bool(l2) for l1, l2 in zip(vl1_list, vl2_list)], dtype=bool)
    zs_dict = {
        "esmc_cosine": zs_esmc_cos,
        "esm2_150m_zeroshot": zs_esm2_total,
        "ablang2_zeroshot": zs_ablang,
        "esmc_300m_zeroshot": zs_esmc_ab + zs_esmc_ag,
        "esmc_ablang2_zeroshot": np.where(has_both_l, 0.5 * zs_esmc_ab + 0.5 * zs_ablang, zs_esmc_ab) + zs_esmc_ag,
    }
    zs_impact_dict = {
        "esmc_cosine": np.abs(zs_esmc_cos),
        "esm2_150m_zeroshot": zs_esm2_imp,
        "ablang2_zeroshot": zs_abs_llr[:, 2],
        "esmc_300m_zeroshot": zs_abs_llr[:, 0] + zs_abs_llr[:, 1],
        "esmc_ablang2_zeroshot": np.where(has_both_l, 0.5 * zs_abs_llr[:, 0] + 0.5 * zs_abs_llr[:, 2], zs_abs_llr[:, 0]) + zs_abs_llr[:, 1],
    }

    return SplitPairTensors(
        y=y,
        labels=labels,
        is_impactful=is_impactful,
        is_ab_mut=is_ab_mut,
        is_ag_mut=is_ag_mut,
        is_same_ag=is_same_ag,
        esmc_ab_1=esmc_ab_1,
        esmc_ab_2=esmc_ab_2,
        esmc_ag_1=esmc_ag_1,
        esmc_ag_2=esmc_ag_2,
        ablang_1=ablang_1,
        ablang_2=ablang_2,
        esm2_ab_1=esm2_ab_1,
        esm2_ab_2=esm2_ab_2,
        esm2_ag_1=esm2_ag_1,
        esm2_ag_2=esm2_ag_2,
        zs_features=zs_feats,
        biophys_odd=biophys_odd,
        impact_even=impact_even,
        zs_dict=zs_dict,
        zs_impact_dict=zs_impact_dict,
    )


def build_odd_linear_matrix(
    blocks_1: list[np.ndarray],
    blocks_2: list[np.ndarray],
    extra_odd: np.ndarray | None = None,
    *,
    include_unit_direction: bool = True,
) -> np.ndarray:
    r"""Construct strictly antisymmetric feature matrix $\psi(1, 2) = -\psi(2, 1)$."""
    parts = []
    for b1, b2 in zip(blocks_1, blocks_2):
        diff = b1 - b2
        parts.append(diff)
        if include_unit_direction:
            parts.append(_unit_norm_rows(diff))
    if extra_odd is not None:
        parts.append(extra_odd)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _fit_soft_target_linear(
    X_s: np.ndarray,
    t_soft: np.ndarray,
    weights: np.ndarray,
    c_reg: float,
) -> np.ndarray:
    r"""Minimize $\sum_i w_i [\log(1+e^{z_i}) - t_i z_i] + \frac{1}{2C}\|w\|_2^2$ for $t_i \in \{0, 0.5, 1\}$.

    Setting $t_i = 0.5$ on `Neutral (label == 0)` pairs yields $\log(2\cosh(z_i/2))$, actively
    projecting empirically neutral mutation directions toward zero (`z_i -> 0`).
    """
    X32 = np.ascontiguousarray(X_s, dtype=np.float32)
    t64 = t_soft.astype(np.float64, copy=False)
    w64 = weights.astype(np.float64, copy=False)
    w_sum = max(float(w64.sum()), 1.0)
    lam = 1.0 / max(c_reg * w_sum, 1e-8)

    def obj_grad(coef: np.ndarray) -> tuple[float, np.ndarray]:
        c32 = coef.astype(np.float32)
        z = (X32 @ c32).astype(np.float64)
        loss = np.sum(w64 * (np.logaddexp(0.0, z) - t64 * z)) / w_sum + 0.5 * lam * float(np.dot(coef, coef))
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
        r32 = (w64 * (p - t64) / w_sum).astype(np.float32)
        grad = (X32.T @ r32).astype(np.float64) + lam * coef
        return float(loss), grad

    res = minimize(obj_grad, np.zeros(X_s.shape[1], dtype=np.float64), method="L-BFGS-B", jac=True, options={"maxiter": 80, "gtol": 1e-4})
    return res.x.astype(np.float64)


class AntisymmetricLinearRanker:
    r"""Zero-intercept L2-regularized pairwise ranker ($f(1,2) = -f(2,1)$).

    Supports both standard impactful-only binary training (`use_neutral_nullspace=False`)
    and 3-state neutral-nullspace training (`use_neutral_nullspace=True`, setting $t=0.5$
    on `label == 0` pairs so neutral mutation directions shrink to zero).
    """

    def __init__(
        self,
        c_grid: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
        *,
        use_neutral_nullspace: bool = False,
    ):
        self.c_grid = c_grid
        self.use_neutral_nullspace = use_neutral_nullspace
        self.scaler = StandardScaler(with_mean=False)
        self.coef_: np.ndarray | None = None
        self.best_c: float = 1e-3

    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        w_tr: np.ndarray | None,
        X_val: np.ndarray,
        val_df: pd.DataFrame,
        val_is_ag_mut: np.ndarray,
        *,
        labels_tr: np.ndarray | None = None,
    ) -> "AntisymmetricLinearRanker":
        if labels_tr is not None and not self.use_neutral_nullspace:
            imp = labels_tr != 0
            X_fit = X_tr[imp]
            y_fit = y_tr[imp]
            w_fit = w_tr[imp] if w_tr is not None else None
        else:
            X_fit = X_tr
            y_fit = y_tr
            w_fit = w_tr

        X_tr_s = self.scaler.fit_transform(X_fit)
        X_val_s = self.scaler.transform(X_val)

        best_score = -1.0
        best_coef = np.zeros(X_tr.shape[1], dtype=np.float64)

        if self.use_neutral_nullspace and labels_tr is not None and (labels_tr == 0).any():
            t_soft = np.where(labels_tr > 0, 1.0, np.where(labels_tr < 0, 0.0, 0.5))
            weights = np.ones(len(labels_tr), dtype=np.float64) if w_fit is None else w_fit.astype(np.float64)
            val_lbl = val_df["label"].to_numpy(dtype=int) if "label" in val_df.columns else np.where(val_df["binding_diff"] > 0, 1, -1)
            val_bd = val_df["binding_diff"].to_numpy(dtype=np.float64)
            for c in self.c_grid:
                coef = _fit_soft_target_linear(X_tr_s, t_soft, weights, c)
                val_preds = X_val_s @ coef
                sc_dir = val_balanced_objective(val_df, val_preds, val_is_ag_mut)
                sc_ord = compute_per_antigen_macro_metrics(val_df, val_preds, min_pairs=5)["auc_ordinal"] or sc_dir
                score = 0.65 * sc_dir + 0.35 * float(sc_ord)
                if score > best_score:
                    best_score = score
                    best_coef = coef
                    self.best_c = c
        else:
            for c in self.c_grid:
                clf = LogisticRegression(C=c, fit_intercept=False, solver="lbfgs", max_iter=1200)
                clf.fit(X_tr_s, y_fit, sample_weight=w_fit)
                val_preds = clf.decision_function(X_val_s)
                score = val_balanced_objective(val_df, val_preds, val_is_ag_mut)
                if score > best_score:
                    best_score = score
                    best_coef = clf.coef_.ravel().astype(np.float64).copy()
                    self.best_c = c

        self.coef_ = best_coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None
        return (self.scaler.transform(X) @ self.coef_).astype(np.float64)


class TwoHeadRegimeRanker:
    """Regime-decoupled antisymmetric ranker routing same-partner vs. cross-antigen comparisons."""

    def __init__(self, local_head: AntisymmetricLinearRanker, global_head: AntisymmetricLinearRanker):
        self.local_head = local_head
        self.global_head = global_head

    def predict(self, sp: SplitPairTensors, X_local: np.ndarray, X_global: np.ndarray) -> np.ndarray:
        same_partner = sp.is_same_ag | (~sp.is_ab_mut)
        out = np.zeros(len(sp.y), dtype=np.float64)
        if same_partner.any():
            out[same_partner] = self.local_head.predict(X_local[same_partner])
        if (~same_partner).any():
            out[~same_partner] = self.global_head.predict(X_global[~same_partner])
        return out


class SymmetricImpactClassifier:
    r"""Even-symmetric (`q(1,2) == +q(2,1)`) hot-spot / mutation-impact classifier.

    Predicts $P(|\text{label}| = 1 \mid x_1, x_2)$ from strictly symmetric structural/evolutionary
    perturbation descriptors (`impact_even`), PCA-compressed elementwise embedding change
    magnitudes `|e_1 - e_2|`, and mean complex context `0.5 * (e_1 + e_2)`.
    """

    def __init__(self, n_pca_diff: int = 32, n_pca_ctx: int = 24, seed: int = 42):
        self.n_pca_diff = n_pca_diff
        self.n_pca_ctx = n_pca_ctx
        self.seed = seed
        self.ab_scaler = StandardScaler()
        self.ag_scaler = StandardScaler()
        self.pca_ab_diff = PCA(n_components=n_pca_diff, random_state=seed)
        self.pca_ag_diff = PCA(n_components=max(8, n_pca_diff // 2), random_state=seed)
        self.pca_ctx = PCA(n_components=n_pca_ctx, random_state=seed)
        self.feat_scaler = StandardScaler()
        self.clf: LogisticRegression | None = None

    def _build_even_matrix(
        self,
        sp: SplitPairTensors,
        dir_logits: np.ndarray | None = None,
        *,
        fit_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        ab1_raw = np.concatenate([sp.esmc_ab_1, sp.ablang_1], axis=1)
        ab2_raw = np.concatenate([sp.esmc_ab_2, sp.ablang_2], axis=1)
        if fit_mask is not None:
            self.ab_scaler.fit(np.vstack([ab1_raw[fit_mask], ab2_raw[fit_mask]]))
            self.ag_scaler.fit(np.vstack([sp.esmc_ag_1[fit_mask], sp.esmc_ag_2[fit_mask]]))

        ab1 = self.ab_scaler.transform(ab1_raw)
        ab2 = self.ab_scaler.transform(ab2_raw)
        ag1 = self.ag_scaler.transform(sp.esmc_ag_1)
        ag2 = self.ag_scaler.transform(sp.esmc_ag_2)

        abs_dab = np.abs(ab1 - ab2)
        abs_dag = np.abs(ag1 - ag2)
        ctx_mid = np.concatenate([0.5 * (ab1 + ab2), 0.5 * (ag1 + ag2)], axis=1)

        if fit_mask is not None:
            self.pca_ab_diff.fit(abs_dab[fit_mask])
            self.pca_ag_diff.fit(abs_dag[fit_mask])
            self.pca_ctx.fit(ctx_mid[fit_mask])

        z_dab = self.pca_ab_diff.transform(abs_dab)
        z_dag = self.pca_ag_diff.transform(abs_dag)
        z_ctx = self.pca_ctx.transform(ctx_mid)

        parts = [sp.impact_even, z_dab, z_dag, z_ctx]
        if dir_logits is not None:
            parts.append(np.abs(dir_logits).reshape(-1, 1))
        X_even = np.concatenate(parts, axis=1).astype(np.float32)
        if fit_mask is not None:
            self.feat_scaler.fit(X_even[fit_mask])
        return self.feat_scaler.transform(X_even)

    def fit(
        self,
        tr_sp: SplitPairTensors,
        val_sp: SplitPairTensors,
        dir_tr: np.ndarray | None = None,
        dir_val: np.ndarray | None = None,
        val_df: pd.DataFrame | None = None,
    ) -> "SymmetricImpactClassifier":
        # Fit scalers, PCA, and classifier strictly on same-partner pairs (L1-L3 + Candidate)
        loc_tr = tr_sp.is_same_ag | (~tr_sp.is_ab_mut)
        loc_val = val_sp.is_same_ag | (~val_sp.is_ab_mut)

        X_tr_all = self._build_even_matrix(tr_sp, dir_tr, fit_mask=loc_tr)
        X_val_all = self._build_even_matrix(val_sp, dir_val, fit_mask=None)

        X_tr = X_tr_all[loc_tr]
        y_tr = tr_sp.is_impactful[loc_tr].astype(int)
        X_v = X_val_all[loc_val]
        y_v = val_sp.is_impactful[loc_val].astype(int)
        val_sub = val_df.iloc[np.flatnonzero(loc_val)].reset_index(drop=True) if val_df is not None else None
        z_zero = np.zeros(len(X_v), dtype=np.float64)

        best_auc = -1.0
        for c in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3):
            m = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs", max_iter=800)
            m.fit(X_tr, y_tr)
            p_v = m.predict_proba(X_v)[:, 1]
            if val_sub is not None and "Ag_group_1" in val_sub.columns:
                sc = float(compute_per_antigen_macro_metrics(val_sub, z_zero, p_v, min_pairs=5)["auc_impact"] or 0.5)
            elif len(np.unique(y_v)) >= 2:
                sc = float(roc_auc_score(y_v, p_v))
            else:
                sc = 0.5
            if sc > best_auc:
                best_auc = sc
                self.clf = m
        return self

    def predict_proba(self, sp: SplitPairTensors, dir_logits: np.ndarray | None = None) -> np.ndarray:
        assert self.clf is not None
        X = self._build_even_matrix(sp, dir_logits, fit_mask=None)
        p = self.clf.predict_proba(X)[:, 1].astype(np.float64)
        same_partner = sp.is_same_ag | (~sp.is_ab_mut)
        p[~same_partner] = 1.0
        return p


class OrdinalHurdleRanker:
    r"""3-State Hurdle Ranker combining an even-symmetric Impact Head with a Neutral-Nullspace Directional Head.

    Outputs:
      - `pred_diff(1, 2)`   = $\hat{p}_{\text{imp}}(1, 2)^\alpha \cdot \tanh(\gamma \hat{z}_{\text{dir}}(1, 2))$ (`-f(2, 1)`)
      - `pred_impact(1, 2)` = $\hat{p}_{\text{imp}}(1, 2)$ (`+q(2, 1)`)
    """

    def __init__(self, dir_ranker: TwoHeadRegimeRanker | AntisymmetricLinearRanker, impact_clf: SymmetricImpactClassifier):
        self.dir_ranker = dir_ranker
        self.impact_clf = impact_clf
        self.gamma: float = 0.5
        self.alpha: float = 1.0

    def calibrate(self, val_df: pd.DataFrame, val_sp: SplitPairTensors, dir_val: np.ndarray) -> "OrdinalHurdleRanker":
        p_imp = self.impact_clf.predict_proba(val_sp, dir_val)
        lbl = val_sp.labels
        bd = val_df["binding_diff"].to_numpy(dtype=np.float64)
        best_ord = -1.0
        for gamma in (0.25, 0.5, 1.0, 2.0):
            for alpha in (0.5, 1.0, 1.5):
                pred_d = (p_imp ** alpha) * np.tanh(gamma * dir_val)
                m = compute_per_antigen_macro_metrics(val_df, pred_d, p_imp, min_pairs=5)
                sc = (m["auc_ordinal"] or 0.5) + 0.5 * (m["auc_selective_hit"] or 0.5)
                if sc > best_ord:
                    best_ord = sc
                    self.gamma = gamma
                    self.alpha = alpha
        return self

    def predict_3state(self, sp: SplitPairTensors, dir_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p_imp = self.impact_clf.predict_proba(sp, dir_logits)
        pred_diff = (p_imp ** self.alpha) * np.tanh(self.gamma * dir_logits)
        return pred_diff.astype(np.float64), p_imp.astype(np.float64)


class AntisymmetricXGBoostRanker:
    r"""PCA-compressed + antisymmetrized 3-state XGBoost ranker (`{-1, 0, +1}`)."""

    def __init__(self, n_pca_ab: int = 48, n_pca_ag: int = 32, seed: int = 42):
        self.n_pca_ab = n_pca_ab
        self.n_pca_ag = n_pca_ag
        self.seed = seed
        self.ab_scaler = StandardScaler()
        self.ag_scaler = StandardScaler()
        self.pca_ab = PCA(n_components=n_pca_ab, random_state=seed)
        self.pca_ag = PCA(n_components=n_pca_ag, random_state=seed)
        self.model = None

    def _pair_features(self, ab1: np.ndarray, ab2: np.ndarray, ag1: np.ndarray, ag2: np.ndarray, zs_odd: np.ndarray, imp_even: np.ndarray) -> np.ndarray:
        z_ab1 = self.pca_ab.transform(self.ab_scaler.transform(ab1))
        z_ab2 = self.pca_ab.transform(self.ab_scaler.transform(ab2))
        z_ag1 = self.pca_ag.transform(self.ag_scaler.transform(ag1))
        z_ag2 = self.pca_ag.transform(self.ag_scaler.transform(ag2))

        d_ab = z_ab1 - z_ab2
        u_ab = _unit_norm_rows(d_ab)
        d_ag = z_ag1 - z_ag2
        u_ag = _unit_norm_rows(d_ag)
        ag_ctx = 0.5 * (z_ag1 + z_ag2)
        return np.concatenate([d_ab, u_ab, d_ag, u_ag, ag_ctx, zs_odd, imp_even], axis=1).astype(np.float32)

    def fit(self, tr: SplitPairTensors, val: SplitPairTensors, val_df: pd.DataFrame, w_tr: np.ndarray) -> "AntisymmetricXGBoostRanker":
        import xgboost as xgb

        ab_tr_all = np.concatenate([tr.esmc_ab_1, tr.ablang_1], axis=1)
        ab_tr_all2 = np.concatenate([tr.esmc_ab_2, tr.ablang_2], axis=1)
        self.ab_scaler.fit(np.vstack([ab_tr_all, ab_tr_all2]))
        self.pca_ab.fit(self.ab_scaler.transform(np.vstack([ab_tr_all, ab_tr_all2])))

        self.ag_scaler.fit(np.vstack([tr.esmc_ag_1, tr.esmc_ag_2]))
        self.pca_ag.fit(self.ag_scaler.transform(np.vstack([tr.esmc_ag_1, tr.esmc_ag_2])))

        X12 = self._pair_features(ab_tr_all, ab_tr_all2, tr.esmc_ag_1, tr.esmc_ag_2, tr.zs_features, tr.impact_even)
        X21 = self._pair_features(ab_tr_all2, ab_tr_all, tr.esmc_ag_2, tr.esmc_ag_1, -tr.zs_features, tr.impact_even)
        # 3-class symmetrized target: 0 -> Depleted (-1), 1 -> Neutral (0), 2 -> Improved (+1)
        y12 = tr.labels + 1
        y21 = (-tr.labels) + 1
        X_aug = np.vstack([X12, X21])
        y_aug = np.concatenate([y12, y21])
        w_aug = np.concatenate([w_tr, w_tr])

        ab_v1 = np.concatenate([val.esmc_ab_1, val.ablang_1], axis=1)
        ab_v2 = np.concatenate([val.esmc_ab_2, val.ablang_2], axis=1)
        Xv12 = self._pair_features(ab_v1, ab_v2, val.esmc_ag_1, val.esmc_ag_2, val.zs_features, val.impact_even)
        Xv21 = self._pair_features(ab_v2, ab_v1, val.esmc_ag_2, val.esmc_ag_1, -val.zs_features, val.impact_even)

        best_score = -1.0
        for max_depth, lr, n_est in ((3, 0.04, 250), (4, 0.03, 300)):
            clf = xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=max_depth,
                learning_rate=lr,
                subsample=0.8,
                colsample_bytree=0.6,
                reg_lambda=5.0,
                reg_alpha=1.0,
                n_jobs=max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))),
                random_state=self.seed,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
            )
            clf.fit(X_aug, y_aug, sample_weight=w_aug)
            p12 = clf.predict_proba(Xv12)
            p21 = clf.predict_proba(Xv21)
            preds = 0.5 * ((p12[:, 2] - p12[:, 0]) - (p21[:, 2] - p21[:, 0]))
            sc = val_balanced_objective(val_df, preds, val.is_ag_mut)
            if sc > best_score:
                best_score = sc
                self.model = clf
        return self

    def predict_3state(self, sp: SplitPairTensors) -> tuple[np.ndarray, np.ndarray]:
        assert self.model is not None
        ab1 = np.concatenate([sp.esmc_ab_1, sp.ablang_1], axis=1)
        ab2 = np.concatenate([sp.esmc_ab_2, sp.ablang_2], axis=1)
        X12 = self._pair_features(ab1, ab2, sp.esmc_ag_1, sp.esmc_ag_2, sp.zs_features, sp.impact_even)
        X21 = self._pair_features(ab2, ab1, sp.esmc_ag_2, sp.esmc_ag_1, -sp.zs_features, sp.impact_even)
        p12 = self.model.predict_proba(X12)
        p21 = self.model.predict_proba(X21)
        pred_diff = 0.5 * ((p12[:, 2] - p12[:, 0]) - (p21[:, 2] - p21[:, 0]))
        pred_imp = 1.0 - 0.5 * (p12[:, 1] + p21[:, 1])
        return pred_diff.astype(np.float64), pred_imp.astype(np.float64)


class _BilinearModule(nn.Module):
    """Low-rank Ab x Ag bilinear cross-interaction + symmetric impact head."""

    def __init__(self, ab_dim: int, ag_dim: int, zs_dim: int, even_dim: int, latent_dim: int = 128, dropout: float = 0.35):
        super().__init__()
        self.ab_proj = nn.Sequential(nn.Linear(ab_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(dropout))
        self.ag_proj = nn.Sequential(nn.Linear(ag_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(dropout))
        self.ab_linear = nn.Linear(ab_dim, 1, bias=False)
        self.ag_linear = nn.Linear(ag_dim, 1, bias=False)
        self.zs_linear = nn.Linear(zs_dim, 1, bias=False)
        self.imp_head = nn.Sequential(
            nn.Linear(even_dim + latent_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.scale = float(latent_dim) ** -0.5

    def forward(self, ab1: torch.Tensor, ab2: torch.Tensor, ag1: torch.Tensor, ag2: torch.Tensor, zs_odd: torch.Tensor, imp_even: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_ab1 = self.ab_proj(ab1)
        z_ab2 = self.ab_proj(ab2)
        z_ag1 = self.ag_proj(ag1)
        z_ag2 = self.ag_proj(ag2)
        inter_1 = (z_ab1 * z_ag1).sum(dim=-1, keepdim=True) * self.scale
        inter_2 = (z_ab2 * z_ag2).sum(dim=-1, keepdim=True) * self.scale
        dir_logit = (inter_1 - inter_2 + self.ab_linear(ab1 - ab2) + self.ag_linear(ag1 - ag2) + self.zs_linear(zs_odd)).squeeze(-1)

        sym_lat = torch.abs(z_ab1 - z_ab2) + torch.abs(z_ag1 - z_ag2)
        imp_logit = self.imp_head(torch.cat([imp_even, sym_lat], dim=-1)).squeeze(-1)
        return dir_logit, imp_logit


class _SiameseHybridModule(nn.Module):
    r"""Antisymmetric Mutation-Aware 3-State Siamese RankNet: $(g(u_{12}) - g(u_{21}),\; h(u_{12}) + h(u_{21}))$."""

    def __init__(self, ab_dim: int, ag_dim: int, zs_dim: int, even_dim: int, hidden_dim: int = 192, dropout: float = 0.40):
        super().__init__()
        self.ab_ctx = nn.Sequential(nn.Linear(ab_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.ag_ctx = nn.Sequential(nn.Linear(ag_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.ab_dir = nn.Sequential(nn.Linear(ab_dim, hidden_dim, bias=False), nn.Tanh())
        self.ag_dir = nn.Sequential(nn.Linear(ag_dim, hidden_dim, bias=False), nn.Tanh())

        in_dim = hidden_dim * 4 + zs_dim
        self.trunk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
        )
        self.dir_head = nn.Linear(64, 1)
        self.imp_head = nn.Linear(64 + even_dim, 1)
        self.skip_zs = nn.Linear(zs_dim, 1, bias=False)

    def _branch(self, ab_a: torch.Tensor, ag_a: torch.Tensor, ag_b: torch.Tensor, u_ab: torch.Tensor, u_ag: torch.Tensor, zs_ab: torch.Tensor) -> torch.Tensor:
        za = self.ab_ctx(ab_a)
        ga = self.ag_ctx(ag_a)
        gb = self.ag_ctx(ag_b)
        g_mid = 0.5 * (ga + gb)
        d_ab = self.ab_dir(u_ab)
        d_ag = self.ag_dir(u_ag)
        return self.trunk(torch.cat([za * ga, d_ab * g_mid, d_ab, d_ag, zs_ab], dim=-1))

    def forward(self, ab1: torch.Tensor, ab2: torch.Tensor, ag1: torch.Tensor, ag2: torch.Tensor, u_ab: torch.Tensor, u_ag: torch.Tensor, zs_odd: torch.Tensor, imp_even: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h12 = self._branch(ab1, ag1, ag2, u_ab, u_ag, zs_odd)
        h21 = self._branch(ab2, ag2, ag1, -u_ab, -u_ag, -zs_odd)
        dir_logit = (self.dir_head(h12) - self.dir_head(h21) + self.skip_zs(zs_odd)).squeeze(-1)
        h_sym = 0.5 * (h12 + h21)
        imp_logit = self.imp_head(torch.cat([h_sym, imp_even], dim=-1)).squeeze(-1)
        return dir_logit, imp_logit


class NeuralPairwiseTrainer:
    """Leak-free 3-state trainer for Bilinear and Mutation-Aware Siamese neural rankers."""

    def __init__(
        self,
        arch: str = "siamese",
        *,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 8e-4,
        weight_decay: float = 2e-2,
        seed: int = 42,
    ):
        self.arch = arch
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.ab_scaler = StandardScaler()
        self.ag_scaler = StandardScaler()
        self.zs_scaler = StandardScaler(with_mean=False)
        self.even_scaler = StandardScaler()
        self.net: nn.Module | None = None

    def _prep(self, sp: SplitPairTensors, fit: bool = False):
        ab1_raw = np.concatenate([sp.esmc_ab_1, sp.ablang_1], axis=1)
        ab2_raw = np.concatenate([sp.esmc_ab_2, sp.ablang_2], axis=1)
        zs_comb = np.concatenate([sp.zs_features, sp.biophys_odd], axis=1)
        if fit:
            self.ab_scaler.fit(np.vstack([ab1_raw, ab2_raw]))
            self.ag_scaler.fit(np.vstack([sp.esmc_ag_1, sp.esmc_ag_2]))
            self.zs_scaler.fit(zs_comb)
            self.even_scaler.fit(sp.impact_even)

        ab1 = self.ab_scaler.transform(ab1_raw).astype(np.float32)
        ab2 = self.ab_scaler.transform(ab2_raw).astype(np.float32)
        ag1 = self.ag_scaler.transform(sp.esmc_ag_1).astype(np.float32)
        ag2 = self.ag_scaler.transform(sp.esmc_ag_2).astype(np.float32)
        u_ab = _unit_norm_rows(ab1 - ab2).astype(np.float32)
        u_ag = _unit_norm_rows(ag1 - ag2).astype(np.float32)
        zs = self.zs_scaler.transform(zs_comb).astype(np.float32)
        ev = self.even_scaler.transform(sp.impact_even).astype(np.float32)
        return ab1, ab2, ag1, ag2, u_ab, u_ag, zs, ev

    def fit(self, tr: SplitPairTensors, val: SplitPairTensors, val_df: pd.DataFrame, w_tr: np.ndarray) -> "NeuralPairwiseTrainer":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ab1, ab2, ag1, ag2, u_ab, u_ag, zs, ev = self._prep(tr, fit=True)
        v_ab1, v_ab2, v_ag1, v_ag2, vu_ab, vu_ag, vzs, vev = self._prep(val, fit=False)

        ab_dim, ag_dim, zs_dim, ev_dim = ab1.shape[1], ag1.shape[1], zs.shape[1], ev.shape[1]
        if self.arch == "bilinear":
            net: nn.Module = _BilinearModule(ab_dim, ag_dim, zs_dim, ev_dim).to(device)
        else:
            net = _SiameseHybridModule(ab_dim, ag_dim, zs_dim, ev_dim).to(device)

        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        n = len(tr.y)
        t_soft = np.where(tr.labels > 0, 1.0, np.where(tr.labels < 0, 0.0, 0.5)).astype(np.float32)
        y_imp = tr.is_impactful.astype(np.float32)
        loc_mask = (tr.is_same_ag | (~tr.is_ab_mut)).astype(np.float32)

        t_y = torch.from_numpy(t_soft).to(device)
        t_imp = torch.from_numpy(y_imp).to(device)
        t_loc = torch.from_numpy(loc_mask).to(device)
        w_t = torch.from_numpy(w_tr.astype(np.float32)).to(device)

        t_ab1 = torch.from_numpy(ab1).to(device)
        t_ab2 = torch.from_numpy(ab2).to(device)
        t_ag1 = torch.from_numpy(ag1).to(device)
        t_ag2 = torch.from_numpy(ag2).to(device)
        tu_ab = torch.from_numpy(u_ab).to(device)
        tu_ag = torch.from_numpy(u_ag).to(device)
        t_zs = torch.from_numpy(zs).to(device)
        t_ev = torch.from_numpy(ev).to(device)

        best_val = -1.0
        best_state = None
        rng = np.random.default_rng(self.seed)
        val_bd = val_df["binding_diff"].to_numpy(dtype=np.float64)

        for _epoch in range(self.epochs):
            net.train()
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s : s + self.batch_size]
                opt.zero_grad()
                if self.arch == "bilinear":
                    d_log, i_log = net(t_ab1[idx], t_ab2[idx], t_ag1[idx], t_ag2[idx], t_zs[idx], t_ev[idx])
                else:
                    d_log, i_log = net(t_ab1[idx], t_ab2[idx], t_ag1[idx], t_ag2[idx], tu_ab[idx], tu_ag[idx], t_zs[idx], t_ev[idx])
                loss_dir = (F.binary_cross_entropy_with_logits(d_log, t_y[idx], reduction="none") * w_t[idx]).mean()
                loss_imp = (F.binary_cross_entropy_with_logits(i_log, t_imp[idx], reduction="none") * t_loc[idx]).mean()
                loss = loss_dir + 0.5 * loss_imp
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            sched.step()

            net.eval()
            with torch.no_grad():
                va1 = torch.from_numpy(v_ab1).to(device)
                va2 = torch.from_numpy(v_ab2).to(device)
                vg1 = torch.from_numpy(v_ag1).to(device)
                vg2 = torch.from_numpy(v_ag2).to(device)
                vua = torch.from_numpy(vu_ab).to(device)
                vug = torch.from_numpy(vu_ag).to(device)
                vz = torch.from_numpy(vzs).to(device)
                ve = torch.from_numpy(vev).to(device)
                if self.arch == "bilinear":
                    vd, vi = net(va1, va2, vg1, vg2, vz, ve)
                else:
                    vd, vi = net(va1, va2, vg1, vg2, vua, vug, vz, ve)
                vp_imp = torch.sigmoid(vi).cpu().numpy()
                vp_dir = (vp_imp * torch.tanh(0.5 * vd).cpu().numpy())
            sc_d = val_balanced_objective(val_df, vp_dir, val.is_ag_mut)
            m3 = compute_per_antigen_macro_metrics(val_df, vp_dir, vp_imp, min_pairs=5)
            sc = 0.65 * sc_d + 0.35 * float(m3["auc_ordinal"] or sc_d)
            if sc > best_val:
                best_val = sc
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        assert best_state is not None
        net.load_state_dict(best_state)
        net.to("cpu").eval()
        self.net = net
        return self

    def predict_3state(self, sp: SplitPairTensors) -> tuple[np.ndarray, np.ndarray]:
        assert self.net is not None
        ab1, ab2, ag1, ag2, u_ab, u_ag, zs, ev = self._prep(sp, fit=False)
        with torch.no_grad():
            t_ab1 = torch.from_numpy(ab1)
            t_ab2 = torch.from_numpy(ab2)
            t_ag1 = torch.from_numpy(ag1)
            t_ag2 = torch.from_numpy(ag2)
            tu_ab = torch.from_numpy(u_ab)
            tu_ag = torch.from_numpy(u_ag)
            t_zs = torch.from_numpy(zs)
            t_ev = torch.from_numpy(ev)
            if self.arch == "bilinear":
                d_log, i_log = self.net(t_ab1, t_ab2, t_ag1, t_ag2, t_zs, t_ev)
            else:
                d_log, i_log = self.net(t_ab1, t_ab2, t_ag1, t_ag2, tu_ab, tu_ag, t_zs, t_ev)
            p_imp = torch.sigmoid(i_log).numpy().astype(np.float64)
            same_partner = sp.is_same_ag | (~sp.is_ab_mut)
            p_imp[~same_partner] = 1.0
            pred_diff = p_imp * np.tanh(0.5 * d_log.numpy().astype(np.float64))
            return pred_diff, p_imp
