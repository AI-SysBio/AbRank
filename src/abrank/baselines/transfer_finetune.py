"""Cross-modality 3-state transfer & fine-tuning models (`KD -> ELISA / IC50 / ΔΔG / Escape`).

Implements strictly leak-free 3-state (`{-1: Depleted, 0: Neutral, +1: Improved}`) transfer methods:
  1. `Zero-Shot PLM` (`ESMC-300M` + `AbLang2-Paired` site-marginal LLRs & `|LLR|` impact scores)
  2. `Direct KD Transfer` (0 target-assay labels; pretrained strictly on non-overlapping `KD` pairs)
  3. `Target-Assay From Scratch` (3-state hurdle ranker fit only on target training folds)
  4. `KD -> Target L2-SP Fine-Tuning` (3-state hurdle ranker anchored to `KD` weights via L2-SP)
  5. `KD -> Target Residual Adapter` (antisymmetric Siamese residual adapter with symmetric impact head)

Every directional prediction satisfies exact antisymmetry $f(1, 2) = -f(2, 1)$, every impact
prediction satisfies exact symmetry $q(1, 2) = +q(2, 1)$, and all scalers & hyperparameters are
fit strictly inside training splits.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.nn.functional as F

from abrank.baselines.embeddings import pair_hash, seq_hash
from abrank.baselines.features import (
    extract_mutation_biophys_odd,
    extract_mutation_impact_even,
)
from abrank.baselines.models import (
    _sym_site_abs_llr,
    _sym_site_llr,
    _unit_norm_rows,
    sym_auc,
)
from abrank.evaluate import compute_3state_metrics


@dataclass
class TransferPairTensors:
    """Pre-assembled 3-state antisymmetric, symmetric-impact, and complex arrays."""

    y: np.ndarray               # [N] in {0, 1} (`binding_diff > 0`)
    labels: np.ndarray          # [N] in {-1, 0, +1}
    is_impactful: np.ndarray    # [N] bool (`labels != 0`)
    bdiff: np.ndarray           # [N] float64 (`binding_diff`)
    is_ab_mut: np.ndarray       # [N] bool
    is_ag_mut: np.ndarray       # [N] bool
    is_same_ag: np.ndarray      # [N] bool
    esmc_ab_1: np.ndarray       # [N, 2880]
    esmc_ab_2: np.ndarray
    esmc_ag_1: np.ndarray       # [N, 1920]
    esmc_ag_2: np.ndarray
    ablang_1: np.ndarray        # [N, 960]
    ablang_2: np.ndarray
    zs_features: np.ndarray     # [N, 10] strictly odd PLM LLR / PLL / cosine features
    biophys_odd: np.ndarray     # [N, 18] strictly odd site-mutation biophysical features
    impact_even: np.ndarray     # [N, 38] strictly even (`q(1,2) == +q(2,1)`) impact features
    zs_dict: dict[str, np.ndarray]
    zs_impact_dict: dict[str, np.ndarray]

    def subset(self, idx: np.ndarray | list[int]) -> "TransferPairTensors":
        arr = np.asarray(idx)
        return TransferPairTensors(
            y=self.y[arr],
            labels=self.labels[arr],
            is_impactful=self.is_impactful[arr],
            bdiff=self.bdiff[arr],
            is_ab_mut=self.is_ab_mut[arr],
            is_ag_mut=self.is_ag_mut[arr],
            is_same_ag=self.is_same_ag[arr],
            esmc_ab_1=self.esmc_ab_1[arr],
            esmc_ab_2=self.esmc_ab_2[arr],
            esmc_ag_1=self.esmc_ag_1[arr],
            esmc_ag_2=self.esmc_ag_2[arr],
            ablang_1=self.ablang_1[arr],
            ablang_2=self.ablang_2[arr],
            zs_features=self.zs_features[arr],
            biophys_odd=self.biophys_odd[arr],
            impact_even=self.impact_even[arr],
            zs_dict={k: v[arr] for k, v in self.zs_dict.items()},
            zs_impact_dict={k: v[arr] for k, v in self.zs_impact_dict.items()},
        )


def assemble_transfer_tensors(
    pairs_df: pd.DataFrame,
    assays_lookup: pd.DataFrame,
    esmc_cache: dict,
    ablang_cache: dict,
    esm2_cache: dict | None = None,
) -> TransferPairTensors:
    """Assemble `ESMC-300M` + `AbLang2-Paired` + 3-state odd/even mutation features."""
    n = len(pairs_df)
    r1 = assays_lookup.loc[pairs_df["AbAgID_1"]]
    r2 = assays_lookup.loc[pairs_df["AbAgID_2"]]

    vh1_l = [str(s) for s in r1["Ab_heavy_chain_seq"].tolist()]
    vl1_l = ["" if pd.isna(s) else str(s) for s in r1["Ab_light_chain_seq"].tolist()]
    ag1_l = [str(s) for s in r1["Ag_seq"].tolist()]
    vh2_l = [str(s) for s in r2["Ab_heavy_chain_seq"].tolist()]
    vl2_l = ["" if pd.isna(s) else str(s) for s in r2["Ab_light_chain_seq"].tolist()]
    ag2_l = [str(s) for s in r2["Ag_seq"].tolist()]

    bdiff = pairs_df["binding_diff"].to_numpy(dtype=np.float64)
    y = (bdiff > 0).astype(int)
    labels = pairs_df["label"].to_numpy(dtype=int) if "label" in pairs_df.columns else np.where(bdiff > 0, 1, -1)
    is_impactful = labels != 0
    is_ab_mut = np.array([(h1 != h2) or (l1 != l2) for h1, l1, h2, l2 in zip(vh1_l, vl1_l, vh2_l, vl2_l)], dtype=bool)
    is_ag_mut = np.array([a1 != a2 for a1, a2 in zip(ag1_l, ag2_l)], dtype=bool)
    is_same_ag = ~is_ag_mut

    means, stds, cdr3s = esmc_cache["means"], esmc_cache["stds"], esmc_cache["cdr3s"]
    zero_m960 = np.zeros(960, dtype=np.float32)

    def _esmc_gather(vh_list: list[str], vl_list: list[str], ag_list: list[str]):
        ab_arr = np.stack(
            [
                np.concatenate([
                    means[seq_hash(h)].astype(np.float32),
                    means[seq_hash(l)].astype(np.float32) if l else zero_m960,
                    cdr3s[seq_hash(h)].astype(np.float32),
                ])
                for h, l in zip(vh_list, vl_list)
            ],
            axis=0,
        )
        ag_arr = np.stack(
            [np.concatenate([means[seq_hash(a)], stds[seq_hash(a)]]).astype(np.float32) for a in ag_list],
            axis=0,
        )
        return ab_arr, ag_arr

    esmc_ab_1, esmc_ag_1 = _esmc_gather(vh1_l, vl1_l, ag1_l)
    esmc_ab_2, esmc_ag_2 = _esmc_gather(vh2_l, vl2_l, ag2_l)

    zero_abl = np.zeros(960, dtype=np.float32)
    ablang_1 = np.stack(
        [
            np.concatenate([ablang_cache["paired_means"][pair_hash(h, l)], ablang_cache["vh_cdr3s"][pair_hash(h, l)]]).astype(np.float32)
            if l else zero_abl
            for h, l in zip(vh1_l, vl1_l)
        ],
        axis=0,
    )
    ablang_2 = np.stack(
        [
            np.concatenate([ablang_cache["paired_means"][pair_hash(h, l)], ablang_cache["vh_cdr3s"][pair_hash(h, l)]]).astype(np.float32)
            if l else zero_abl
            for h, l in zip(vh2_l, vl2_l)
        ],
        axis=0,
    )

    zs_feats = np.zeros((n, 10), dtype=np.float32)
    zs_abs_llr = np.zeros((n, 3), dtype=np.float32)
    zs_esmc_ab = np.zeros(n, dtype=np.float32)
    zs_esmc_ag = np.zeros(n, dtype=np.float32)
    zs_ablang = np.zeros(n, dtype=np.float32)
    zs_esmc_cos = np.zeros(n, dtype=np.float32)
    has_both_l = np.ones(n, dtype=bool)

    e_plls, e_lps = esmc_cache["plls"], esmc_cache["aa_lps"]
    ab_plls, ab_vh_lps, ab_vl_lps = ablang_cache["plls"], ablang_cache["vh_lps"], ablang_cache["vl_lps"]
    empty_lp = np.zeros((0, 20), dtype=np.float16)

    for i in range(n):
        vh1, vl1, ag1 = vh1_l[i], vl1_l[i], ag1_l[i]
        vh2, vl2, ag2 = vh2_l[i], vl2_l[i], ag2_l[i]
        has_l1, has_l2 = bool(vl1), bool(vl2)
        has_both_l[i] = has_l1 and has_l2
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

        zs_esmc_ab[i] = site_ab_esmc if (ok_vh and ok_vl) else d_esmc_pll_ab
        zs_esmc_ag[i] = site_ag if ok_ag else d_esmc_pll_ag
        zs_ablang[i] = (site_ab_ablang if (ok_abh and ok_abl) else d_ablang_pll) if (has_l1 and has_l2) else 0.0

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

    biophys_odd = extract_mutation_biophys_odd(vh1_l, vl1_l, ag1_l, vh2_l, vl2_l, ag2_l)
    mut_even = extract_mutation_impact_even(vh1_l, vl1_l, ag1_l, vh2_l, vl2_l, ag2_l)
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

    zs_dict = {
        "esmc_300m_zeroshot": zs_esmc_ab + zs_esmc_ag,
        "ablang2_zeroshot": zs_ablang,
        "esmc_ablang2_zeroshot": np.where(has_both_l, 0.5 * zs_esmc_ab + 0.5 * zs_ablang, zs_esmc_ab) + zs_esmc_ag,
        "esmc_cosine": zs_esmc_cos,
    }
    if esm2_cache is not None and "plls" in esm2_cache and "aa_lps" in esm2_cache:
        m2_plls, m2_lps = esm2_cache["plls"], esm2_cache["aa_lps"]
        zs_esm2 = np.zeros(n, dtype=np.float32)
        for i in range(n):
            vh1, vl1, ag1 = vh1_l[i], vl1_l[i], ag1_l[i]
            vh2, vl2, ag2 = vh2_l[i], vl2_l[i], ag2_l[i]
            has_l1, has_l2 = bool(vl1), bool(vl2)
            kh1, ka1 = seq_hash(vh1), seq_hash(ag1)
            kh2, ka2 = seq_hash(vh2), seq_hash(ag2)
            kl1 = seq_hash(vl1) if has_l1 else ""
            kl2 = seq_hash(vl2) if has_l2 else ""
            if kh1 in m2_plls and kh2 in m2_plls and ka1 in m2_plls and ka2 in m2_plls:
                m2_vh, m2_ok_vh = _sym_site_llr(vh1, vh2, m2_lps[kh1], m2_lps[kh2])
                m2_vl, m2_ok_vl = _sym_site_llr(vl1, vl2, m2_lps[kl1] if (has_l1 and kl1 in m2_lps) else empty_lp, m2_lps[kl2] if (has_l2 and kl2 in m2_lps) else empty_lp)
                m2_ag, m2_ok_ag = _sym_site_llr(ag1, ag2, m2_lps[ka1], m2_lps[ka2])
                m2_pll_l1 = m2_plls[kl1] if (has_l1 and kl1 in m2_plls) else m2_plls[kh1]
                m2_pll_l2 = m2_plls[kl2] if (has_l2 and kl2 in m2_plls) else m2_plls[kh2]
                m2_ab = (m2_vh + m2_vl) if (m2_ok_vh and m2_ok_vl) else 0.5 * ((m2_plls[kh1] + m2_pll_l1) - (m2_plls[kh2] + m2_pll_l2))
                m2_ag_s = m2_ag if m2_ok_ag else (m2_plls[ka1] - m2_plls[ka2])
                zs_esm2[i] = m2_ab + m2_ag_s
        zs_dict["esm2_150m_zeroshot"] = zs_esm2
    zs_impact_dict = {
        "esmc_300m_zeroshot": zs_abs_llr[:, 0] + zs_abs_llr[:, 1],
        "ablang2_zeroshot": zs_abs_llr[:, 2],
        "esmc_ablang2_zeroshot": np.where(has_both_l, 0.5 * zs_abs_llr[:, 0] + 0.5 * zs_abs_llr[:, 2], zs_abs_llr[:, 0]) + zs_abs_llr[:, 1],
    }

    return TransferPairTensors(
        y=y,
        labels=labels,
        is_impactful=is_impactful,
        bdiff=bdiff,
        is_ab_mut=is_ab_mut,
        is_ag_mut=is_ag_mut,
        is_same_ag=is_same_ag,
        esmc_ab_1=esmc_ab_1,
        esmc_ab_2=esmc_ab_2,
        esmc_ag_1=esmc_ag_1,
        esmc_ag_2=esmc_ag_2,
        ablang_1=ablang_1,
        ablang_2=ablang_2,
        zs_features=zs_feats,
        biophys_odd=biophys_odd,
        impact_even=impact_even,
        zs_dict=zs_dict,
        zs_impact_dict=zs_impact_dict,
    )


def build_transfer_odd_matrix(tp: TransferPairTensors, include_biophys: bool = True) -> np.ndarray:
    """Build strictly antisymmetric multi-scale matrix `[d_ab, u_ab, d_abl, u_abl, d_ag, u_ag, zs, biophys]`."""
    blocks_1 = [tp.esmc_ab_1, tp.ablang_1, tp.esmc_ag_1]
    blocks_2 = [tp.esmc_ab_2, tp.ablang_2, tp.esmc_ag_2]
    parts = []
    for b1, b2 in zip(blocks_1, blocks_2):
        diff = b1 - b2
        parts.append(diff)
        parts.append(_unit_norm_rows(diff))
    parts.append(tp.zs_features)
    if include_biophys:
        parts.append(tp.biophys_odd)
    return np.concatenate(parts, axis=1).astype(np.float32)


@dataclass
class KDSourceRanker:
    """Pretrained 3-state `KD` source ranker (antisymmetric directional head + symmetric impact head)."""

    scaler: StandardScaler
    w_kd: np.ndarray            # [D] standardized directional weight vector
    best_c: float
    imp_scaler: StandardScaler | None = None
    imp_clf: LogisticRegression | None = None

    def predict_raw_matrix(self, X: np.ndarray) -> np.ndarray:
        return (self.scaler.transform(X) @ self.w_kd).astype(np.float64)

    def predict_3state(self, tp: TransferPairTensors, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z_dir = self.predict_raw_matrix(X)
        if self.imp_clf is not None and self.imp_scaler is not None:
            even_in = np.concatenate([tp.impact_even, np.abs(z_dir).reshape(-1, 1)], axis=1)
            p_imp = self.imp_clf.predict_proba(self.imp_scaler.transform(even_in))[:, 1].astype(np.float64)
            pred_diff = p_imp * np.tanh(0.5 * z_dir)
            return pred_diff, p_imp
        return z_dir, np.abs(z_dir)


def fit_kd_source_ranker(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray | None,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    tp_tr: TransferPairTensors | None = None,
    tp_val: TransferPairTensors | None = None,
    c_grid: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
) -> KDSourceRanker:
    """Fit zero-intercept L2-regularized `KD` directional + impact heads strictly on leak-free `KD` splits."""
    imp_tr = tp_tr.is_impactful if tp_tr is not None else np.ones(len(y_tr), dtype=bool)
    imp_val = tp_val.is_impactful if tp_val is not None else np.ones(len(y_val), dtype=bool)

    scaler = StandardScaler(with_mean=False)
    X_tr_s = scaler.fit_transform(X_tr[imp_tr])
    X_val_s = scaler.transform(X_val[imp_val])
    w_fit = w_tr[imp_tr] if w_tr is not None else None

    best_auc = -1.0
    best_w = np.zeros(X_tr.shape[1], dtype=np.float64)
    best_c = 1e-3
    for c in c_grid:
        clf = LogisticRegression(C=c, fit_intercept=False, solver="lbfgs", max_iter=400, tol=1e-3)
        clf.fit(X_tr_s, y_tr[imp_tr], sample_weight=w_fit)
        preds = clf.decision_function(X_val_s)
        sc = sym_auc(y_val[imp_val], preds)
        if sc > best_auc:
            best_auc = sc
            best_w = clf.coef_.ravel().astype(np.float64).copy()
            best_c = c

    imp_scaler = None
    imp_clf = None
    if tp_tr is not None and (tp_tr.labels == 0).any() and (tp_tr.labels != 0).any():
        loc_tr = tp_tr.is_same_ag | (~tp_tr.is_ab_mut)
        if loc_tr.sum() >= 30 and len(np.unique(tp_tr.is_impactful[loc_tr])) >= 2:
            z_tr = (scaler.transform(X_tr[loc_tr]) @ best_w).reshape(-1, 1)
            even_tr = np.concatenate([tp_tr.impact_even[loc_tr], np.abs(z_tr)], axis=1)
            imp_scaler = StandardScaler()
            E_tr_s = imp_scaler.fit_transform(even_tr)
            imp_clf = LogisticRegression(C=1e-2, class_weight="balanced", solver="lbfgs", max_iter=500)
            imp_clf.fit(E_tr_s, tp_tr.is_impactful[loc_tr].astype(int))

    return KDSourceRanker(scaler=scaler, w_kd=best_w, best_c=best_c, imp_scaler=imp_scaler, imp_clf=imp_clf)


def _solve_l2sp_dual_or_primal(
    X_s: np.ndarray,
    t_soft: np.ndarray,
    w_anchor: np.ndarray,
    lam_sp: float,
    lam_l2: float,
    *,
    K_mat: np.ndarray | None = None,
    z0_unit: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Solve soft-target L2-SP logistic regression (`t_soft in {0, 0.5, 1}`).

    Uses exact Representer-Theorem dual Newton for `N <= 2000` (`<15ms`/solve) and
    `float32` primal L-BFGS for larger tracks (`AbAgym`, `Bloom Escape`).
    """
    n, d = X_s.shape
    lam_tot = max(lam_sp + lam_l2, 1e-8)
    c_sp = lam_sp / lam_tot
    y64 = t_soft.astype(np.float64, copy=False)

    if K_mat is not None:
        K64 = K_mat
        z0 = (c_sp * z0_unit) if (z0_unit is not None and lam_sp > 0.0) else (
            (X_s @ (c_sp * w_anchor)).astype(np.float64) if lam_sp > 0.0 else np.zeros(n, dtype=np.float64)
        )
        reg = n * lam_tot
        u = np.zeros(n, dtype=np.float64)
        Ku = np.zeros(n, dtype=np.float64)
        for _ in range(10):
            z = z0 + Ku
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
            res_kkt = (p - y64) + reg * u
            if float(np.max(np.abs(res_kkt))) < 1e-6:
                break
            wt = np.maximum(p * (1.0 - p), 1e-9)
            s_wt = np.sqrt(wt)
            A = (s_wt[:, None] * K64) * s_wt[None, :]
            A[np.diag_indices_from(A)] += reg
            rhs = -(res_kkt / s_wt)
            dv = np.linalg.solve(A, rhs)
            du = s_wt * dv
            Kdu = K64 @ du
            cur_obj = float(np.mean(np.logaddexp(0.0, z) - y64 * z) + 0.5 * lam_tot * np.dot(u, Ku))
            step = 1.0
            while step > 1e-4:
                u_try = u + step * du
                Ku_try = Ku + step * Kdu
                z_try = z0 + Ku_try
                try_obj = float(np.mean(np.logaddexp(0.0, z_try) - y64 * z_try) + 0.5 * lam_tot * np.dot(u_try, Ku_try))
                if try_obj <= cur_obj + 1e-10:
                    u, Ku = u_try, Ku_try
                    break
                step *= 0.5
            else:
                u = u + step * du
                Ku = Ku + step * Kdu

        return u, np.empty(0, dtype=np.float64)

    X32 = np.ascontiguousarray(X_s, dtype=np.float32)
    y32 = t_soft.astype(np.float32, copy=False)
    w_anc32 = w_anchor.astype(np.float32, copy=False)
    w_base = (c_sp * w_anc32).astype(np.float64)
    n_inv = 1.0 / max(n, 1)

    def obj_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        w32 = w.astype(np.float32)
        z = (X32 @ w32).astype(np.float64)
        loss_vec = np.logaddexp(0.0, z) - y64 * z
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
        diff_sp = w - w_anc32
        val = float(loss_vec.mean() + 0.5 * lam_sp * np.dot(diff_sp, diff_sp) + 0.5 * lam_l2 * np.dot(w, w))
        r32 = (p - y32).astype(np.float32)
        grad = (X32.T @ r32).astype(np.float64) * n_inv + lam_sp * diff_sp + lam_l2 * w
        return val, grad

    res = minimize(obj_and_grad, w_base.copy(), method="L-BFGS-B", jac=True, options={"maxiter": 45, "gtol": 2e-4, "ftol": 1e-5})
    return None, res.x.astype(np.float64)


class L2SPFineTunedRanker:
    r"""3-State Hurdle Target Ranker anchored to `KD` weights via L2-Starting-Point (`L2-SP`).

    Trains:
      1. Soft-target L2-SP antisymmetric directional head ($t \in \{0, 0.5, 1\}$),
      2. Even-symmetric impact/hot-spot head (`impact_even` + `|z_dir|` + optional `KD` impact prior).
    """

    def __init__(self, kd_source: KDSourceRanker | None = None):
        self.kd_source = kd_source
        self.scaler = StandardScaler(with_mean=False)
        self.w: np.ndarray | None = None
        self.imp_scaler = StandardScaler()
        self.imp_clf: LogisticRegression | None = None
        self.best_params: dict = {}

    def _build_imp_in(self, tp: TransferPairTensors, X: np.ndarray, z_dir: np.ndarray) -> np.ndarray:
        parts = [tp.impact_even, np.abs(z_dir).reshape(-1, 1)]
        if self.kd_source is not None:
            _, kd_p_imp = self.kd_source.predict_3state(tp, X)
            parts.append(kd_p_imp.reshape(-1, 1))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        groups_tr: np.ndarray,
        *,
        tp_tr: TransferPairTensors | None = None,
    ) -> "L2SPFineTunedRanker":
        self.scaler.fit(X_tr)
        if self.kd_source is not None:
            kd_scale = self.kd_source.scaler.scale_.copy()
            tgt_scale = self.scaler.scale_.copy()
            blended_scale = np.where(kd_scale > 1e-5, kd_scale, tgt_scale)
            blended_scale = np.where(blended_scale > 1e-8, blended_scale, 1.0)
            self.scaler.scale_ = blended_scale
            w_kd_base = self.kd_source.w_kd * (blended_scale / np.where(kd_scale > 1e-8, kd_scale, 1.0))
        else:
            w_kd_base = np.zeros(X_tr.shape[1], dtype=np.float64)

        X_s = self.scaler.transform(X_tr)
        labels_tr = tp_tr.labels if tp_tr is not None else np.where(y_tr > 0, 1, -1)
        bdiff_tr = tp_tr.bdiff if tp_tr is not None else np.where(y_tr > 0, 1.5, -1.5)
        t_soft = np.where(labels_tr > 0, 1.0, np.where(labels_tr < 0, 0.0, 0.5))

        uniq_g = np.unique(groups_tr)
        n_inner = min(3, len(uniq_g))
        if n_inner >= 2:
            gkf = GroupKFold(n_splits=n_inner)
            inner_splits = list(gkf.split(X_s, y_tr, groups_tr))
        else:
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(y_tr))
            mid = max(1, len(y_tr) // 2)
            inner_splits = [(perm[mid:], perm[:mid]), (perm[:mid], perm[mid:])]

        if self.kd_source is None:
            candidates = [(0.0, lam2, 0.0) for lam2 in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)]
        else:
            imp_mask = labels_tr != 0
            if imp_mask.sum() >= 6 and len(np.unique(y_tr[imp_mask])) >= 2:
                kd_logits = (X_s[imp_mask] @ w_kd_base).astype(np.float64)
                cal = LogisticRegression(C=1.0, fit_intercept=False, solver="lbfgs")
                cal.fit(kd_logits.reshape(-1, 1), y_tr[imp_mask])
                alpha_pos = float(np.clip(cal.coef_[0, 0], 0.05, 2.5))
            else:
                alpha_pos = 1.0
            candidates = [
                (40.0, 0.05, 1.0),
                (12.0, 0.1, 1.0),
                (3.0, 0.4, 1.0),
                (12.0, 0.1, alpha_pos),
                (2.5, 0.4, alpha_pos),
                (0.8, 1.0, alpha_pos),
                (0.25, 3.0, alpha_pos),
                (0.0, 1.0, 0.0),
                (0.0, 10.0, 0.0),
            ]

        use_dual = (X_s.shape[0] < X_s.shape[1]) and (X_s.shape[0] <= 2000)
        K_full = (X_s @ X_s.T).astype(np.float64) if use_dual else None
        kd_logits_all = (X_s @ w_kd_base).astype(np.float64) if self.kd_source is not None else np.zeros(len(y_tr), dtype=np.float64)

        best_cv = -1.0
        best_cfg = candidates[0]
        for lam_sp, lam2, a_val in candidates:
            lam_tot = max(lam_sp + lam2, 1e-8)
            c_sp = lam_sp / lam_tot
            fold_aucs = []
            for itr, ival in inner_splits:
                imp_v = labels_tr[ival] != 0
                if imp_v.sum() < 2 or len(itr) < 4:
                    continue
                if use_dual and K_full is not None:
                    K_itr = K_full[np.ix_(itr, itr)]
                    u_fold, _ = _solve_l2sp_dual_or_primal(
                        X_s[itr], t_soft[itr], a_val * w_kd_base, lam_sp, lam2,
                        K_mat=K_itr, z0_unit=a_val * kd_logits_all[itr],
                    )
                    assert u_fold is not None
                    val_preds = (c_sp * a_val) * kd_logits_all[ival] + K_full[np.ix_(ival, itr)] @ u_fold
                else:
                    _, w_fold = _solve_l2sp_dual_or_primal(X_s[itr], t_soft[itr], a_val * w_kd_base, lam_sp, lam2)
                    val_preds = X_s[ival] @ w_fold
                m3 = compute_3state_metrics(labels_tr[ival], bdiff_tr[ival], val_preds)
                sc_d = m3["auc_dir"] if m3["auc_dir"] is not None else 0.5
                sc_o = m3["auc_ordinal"] if m3["auc_ordinal"] is not None else sc_d
                fold_aucs.append(0.65 * sc_d + 0.35 * sc_o)
            mean_cv = float(np.mean(fold_aucs)) if fold_aucs else 0.5
            if mean_cv > best_cv:
                best_cv = mean_cv
                best_cfg = (lam_sp, lam2, a_val)

        lam_sp_star, lam2_star, a_star = best_cfg
        self.best_params = {"lam_sp": lam_sp_star, "lam_l2": lam2_star, "alpha": round(a_star, 4), "inner_cv": round(best_cv, 4)}
        u_final, w_final = _solve_l2sp_dual_or_primal(
            X_s, t_soft, a_star * w_kd_base, lam_sp_star, lam2_star,
            K_mat=K_full, z0_unit=a_star * kd_logits_all,
        )
        if u_final is not None:
            c_sp_star = lam_sp_star / max(lam_sp_star + lam2_star, 1e-8)
            self.w = (c_sp_star * a_star * w_kd_base + (X_s.T @ u_final)).astype(np.float64)
        else:
            self.w = w_final

        # Fit symmetric impact classifier if 3-state labels present
        if tp_tr is not None and (labels_tr == 0).any() and (labels_tr != 0).any():
            z_tr = (X_s @ self.w).astype(np.float64)
            imp_in = self._build_imp_in(tp_tr, X_tr, z_tr)
            E_s = self.imp_scaler.fit_transform(imp_in)
            y_imp = (labels_tr != 0).astype(int)
            best_ic = 1e-2
            best_iauc = -1.0
            for ic in (1e-3, 1e-2, 1e-1, 0.5):
                iaucs = []
                for itr, ival in inner_splits:
                    if len(np.unique(y_imp[itr])) < 2 or len(np.unique(y_imp[ival])) < 2:
                        continue
                    mc = LogisticRegression(C=ic, class_weight="balanced", solver="lbfgs", max_iter=400)
                    mc.fit(E_s[itr], y_imp[itr])
                    iaucs.append(float(roc_auc_score(y_imp[ival], mc.predict_proba(E_s[ival])[:, 1])))
                m_ia = float(np.mean(iaucs)) if iaucs else 0.5
                if m_ia > best_iauc:
                    best_iauc = m_ia
                    best_ic = ic
            self.imp_clf = LogisticRegression(C=best_ic, class_weight="balanced", solver="lbfgs", max_iter=500)
            self.imp_clf.fit(E_s, y_imp)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        return (self.scaler.transform(X) @ self.w).astype(np.float64)

    def predict_3state(self, tp: TransferPairTensors, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z_dir = self.predict(X)
        if self.imp_clf is not None:
            imp_in = self._build_imp_in(tp, X, z_dir)
            p_imp = self.imp_clf.predict_proba(self.imp_scaler.transform(imp_in))[:, 1].astype(np.float64)
            return (p_imp * np.tanh(0.5 * z_dir)).astype(np.float64), p_imp
        return z_dir, np.abs(z_dir)


class _ResidualAdapterNet(nn.Module):
    r"""Antisymmetric 3-State KD-Anchored Residual Adapter Network."""

    def __init__(self, ab_dim: int, ag_dim: int, meta_dim: int, even_dim: int, hidden_dim: int = 96, dropout: float = 0.35):
        super().__init__()
        self.cal_linear = nn.Linear(meta_dim, 1, bias=False)
        self.ab_ctx = nn.Sequential(nn.Linear(ab_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.ag_ctx = nn.Sequential(nn.Linear(ag_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.ab_dir = nn.Sequential(nn.Linear(ab_dim, hidden_dim, bias=False), nn.Tanh())
        self.ag_dir = nn.Sequential(nn.Linear(ag_dim, hidden_dim, bias=False), nn.Tanh())

        in_dim = hidden_dim * 4 + meta_dim
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
        )
        self.head = nn.Linear(32, 1, bias=False)
        nn.init.zeros_(self.head.weight)
        self.imp_head = nn.Sequential(
            nn.Linear(32 + even_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def _branch(self, ab_a: torch.Tensor, ag_a: torch.Tensor, ag_b: torch.Tensor, u_ab: torch.Tensor, u_ag: torch.Tensor, meta_odd: torch.Tensor) -> torch.Tensor:
        za = self.ab_ctx(ab_a)
        ga = self.ag_ctx(ag_a)
        gb = self.ag_ctx(ag_b)
        g_mid = 0.5 * (ga + gb)
        dab = self.ab_dir(u_ab)
        dag = self.ag_dir(u_ag)
        return self.mlp(torch.cat([za * ga, dab * g_mid, dab, dag, meta_odd], dim=-1))

    def forward(
        self,
        ab1: torch.Tensor,
        ab2: torch.Tensor,
        ag1: torch.Tensor,
        ag2: torch.Tensor,
        u_ab: torch.Tensor,
        u_ag: torch.Tensor,
        meta_odd: torch.Tensor,
        even_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h12 = self._branch(ab1, ag1, ag2, u_ab, u_ag, meta_odd)
        h21 = self._branch(ab2, ag2, ag1, -u_ab, -u_ag, -meta_odd)
        dir_logit = (self.cal_linear(meta_odd) + self.head(h12) - self.head(h21)).squeeze(-1)
        h_sym = 0.5 * (h12 + h21)
        imp_logit = self.imp_head(torch.cat([h_sym, even_in], dim=-1)).squeeze(-1)
        return dir_logit, imp_logit


class KDResidualAdapterRanker:
    """Leak-free 3-state trainer for `_ResidualAdapterNet` with inner GroupKFold early stopping."""

    def __init__(
        self,
        kd_source: KDSourceRanker,
        *,
        epochs: int = 20,
        batch_size: int = 224,
        lr: float = 1.2e-3,
        weight_decay: float = 3e-2,
        seed: int = 42,
        n_inner_models: int = 2,
    ):
        self.kd_source = kd_source
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.n_inner_models = n_inner_models
        self.ab_scaler = StandardScaler()
        self.ag_scaler = StandardScaler()
        self.meta_scaler = StandardScaler(with_mean=False)
        self.even_scaler = StandardScaler()
        self.models: list[nn.Module] = []

    def _build_meta_and_even(self, tp: TransferPairTensors, X_odd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kd_raw = self.kd_source.predict_raw_matrix(X_odd).reshape(-1, 1).astype(np.float32)
        _, kd_p_imp = self.kd_source.predict_3state(tp, X_odd)
        ab_gate = (kd_raw[:, 0] * tp.is_ab_mut.astype(np.float32)).reshape(-1, 1)
        ag_gate = (kd_raw[:, 0] * tp.is_ag_mut.astype(np.float32)).reshape(-1, 1)
        meta = np.concatenate([kd_raw, ab_gate, ag_gate, tp.zs_features, tp.biophys_odd], axis=1).astype(np.float32)
        even = np.concatenate([tp.impact_even, np.abs(kd_raw), kd_p_imp.reshape(-1, 1).astype(np.float32)], axis=1).astype(np.float32)
        return meta, even

    def _prep(self, tp: TransferPairTensors, X_odd: np.ndarray, fit: bool = False):
        ab1_raw = np.concatenate([tp.esmc_ab_1, tp.ablang_1], axis=1)
        ab2_raw = np.concatenate([tp.esmc_ab_2, tp.ablang_2], axis=1)
        meta_raw, even_raw = self._build_meta_and_even(tp, X_odd)
        if fit:
            self.ab_scaler.fit(np.vstack([ab1_raw, ab2_raw]))
            self.ag_scaler.fit(np.vstack([tp.esmc_ag_1, tp.esmc_ag_2]))
            self.meta_scaler.fit(meta_raw)
            self.even_scaler.fit(even_raw)

        ab1 = self.ab_scaler.transform(ab1_raw).astype(np.float32)
        ab2 = self.ab_scaler.transform(ab2_raw).astype(np.float32)
        ag1 = self.ag_scaler.transform(tp.esmc_ag_1).astype(np.float32)
        ag2 = self.ag_scaler.transform(tp.esmc_ag_2).astype(np.float32)
        u_ab = _unit_norm_rows(ab1 - ab2).astype(np.float32)
        u_ag = _unit_norm_rows(ag1 - ag2).astype(np.float32)
        meta = self.meta_scaler.transform(meta_raw).astype(np.float32)
        even = self.even_scaler.transform(even_raw).astype(np.float32)
        return ab1, ab2, ag1, ag2, u_ab, u_ag, meta, even

    def fit(self, tp_tr: TransferPairTensors, X_tr: np.ndarray, groups_tr: np.ndarray) -> "KDResidualAdapterRanker":
        ab1, ab2, ag1, ag2, u_ab, u_ag, meta, even = self._prep(tp_tr, X_tr, fit=True)
        uniq_g = np.unique(groups_tr)
        n_inner = min(3, len(uniq_g))
        if n_inner >= 2:
            gkf = GroupKFold(n_splits=n_inner)
            splits = list(gkf.split(ab1, tp_tr.y, groups_tr))
        else:
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(len(tp_tr.y))
            mid = max(1, int(0.8 * len(tp_tr.y)))
            splits = [(perm[:mid], perm[mid:])]

        t_soft_all = np.where(tp_tr.labels > 0, 1.0, np.where(tp_tr.labels < 0, 0.0, 0.5)).astype(np.float32)
        y_imp_all = tp_tr.is_impactful.astype(np.float32)
        has_neu = bool((tp_tr.labels == 0).any() and (tp_tr.labels != 0).any())

        self.models = []
        for f_i, (itr, ival) in enumerate(splits[: self.n_inner_models]):
            torch.manual_seed(self.seed + f_i)
            np.random.seed(self.seed + f_i)
            net = _ResidualAdapterNet(ab1.shape[1], ag1.shape[1], meta.shape[1], even.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

            t_ab1 = torch.from_numpy(ab1[itr])
            t_ab2 = torch.from_numpy(ab2[itr])
            t_ag1 = torch.from_numpy(ag1[itr])
            t_ag2 = torch.from_numpy(ag2[itr])
            tu_ab = torch.from_numpy(u_ab[itr])
            tu_ag = torch.from_numpy(u_ag[itr])
            t_meta = torch.from_numpy(meta[itr])
            t_even = torch.from_numpy(even[itr])
            t_y = torch.from_numpy(t_soft_all[itr])
            t_imp = torch.from_numpy(y_imp_all[itr])

            v_ab1 = torch.from_numpy(ab1[ival])
            v_ab2 = torch.from_numpy(ab2[ival])
            v_ag1 = torch.from_numpy(ag1[ival])
            v_ag2 = torch.from_numpy(ag2[ival])
            vu_ab = torch.from_numpy(u_ab[ival])
            vu_ag = torch.from_numpy(u_ag[ival])
            v_meta = torch.from_numpy(meta[ival])
            v_even = torch.from_numpy(even[ival])

            best_sc = -1.0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            rng = np.random.default_rng(self.seed + 100 + f_i)
            n_sub = len(itr)

            for _ep in range(self.epochs):
                net.train()
                perm = rng.permutation(n_sub)
                for s in range(0, n_sub, self.batch_size):
                    b = perm[s : s + self.batch_size]
                    opt.zero_grad()
                    d_log, i_log = net(t_ab1[b], t_ab2[b], t_ag1[b], t_ag2[b], tu_ab[b], tu_ag[b], t_meta[b], t_even[b])
                    loss = F.binary_cross_entropy_with_logits(d_log, t_y[b])
                    if has_neu:
                        loss = loss + 0.5 * F.binary_cross_entropy_with_logits(i_log, t_imp[b])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                    opt.step()
                sched.step()

                net.eval()
                with torch.no_grad():
                    vd, vi = net(v_ab1, v_ab2, v_ag1, v_ag2, vu_ab, vu_ag, v_meta, v_even)
                    vp_imp = torch.sigmoid(vi).numpy() if has_neu else np.abs(vd.numpy())
                    vp_d = (vp_imp * np.tanh(0.5 * vd.numpy())) if has_neu else vd.numpy()
                m3 = compute_3state_metrics(tp_tr.labels[ival], tp_tr.bdiff[ival], vp_d, vp_imp)
                sc_d = m3["auc_dir"] if m3["auc_dir"] is not None else 0.5
                sc_o = m3["auc_ordinal"] if m3["auc_ordinal"] is not None else sc_d
                sc = 0.65 * sc_d + 0.35 * sc_o
                if sc >= best_sc:
                    best_sc = sc
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

            net.load_state_dict(best_state)
            net.eval()
            self.models.append(net)

        return self

    def predict_3state(self, tp: TransferPairTensors, X_odd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.models
        ab1, ab2, ag1, ag2, u_ab, u_ag, meta, even = self._prep(tp, X_odd, fit=False)
        t_ab1 = torch.from_numpy(ab1)
        t_ab2 = torch.from_numpy(ab2)
        t_ag1 = torch.from_numpy(ag1)
        t_ag2 = torch.from_numpy(ag2)
        tu_ab = torch.from_numpy(u_ab)
        tu_ag = torch.from_numpy(u_ag)
        t_meta = torch.from_numpy(meta)
        t_even = torch.from_numpy(even)
        d_list, i_list = [], []
        with torch.no_grad():
            for m in self.models:
                dl, il = m(t_ab1, t_ab2, t_ag1, t_ag2, tu_ab, tu_ag, t_meta, t_even)
                d_list.append(dl.numpy())
                i_list.append(torch.sigmoid(il).numpy())
        d_mean = np.mean(d_list, axis=0).astype(np.float64)
        p_imp = np.mean(i_list, axis=0).astype(np.float64)
        return (p_imp * np.tanh(0.5 * d_mean)).astype(np.float64), p_imp

    def predict(self, tp: TransferPairTensors, X_odd: np.ndarray) -> np.ndarray:
        return self.predict_3state(tp, X_odd)[0]
