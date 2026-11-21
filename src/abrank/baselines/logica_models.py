"""1D `LogiCA` (Dual-PLM Native-Logit Cross-Attention) baseline for 3-state (`{-1, 0, +1}`)
antibody-antigen affinity ranking, hot-spot identification, and generative binder verification.

For the full end-to-end `AbLang2-Paired` + `ESM-2` token-level LogiCA implementation
(including Stage-1 InfoNCE pretraining, Stage-2 3-state LoRA fine-tuning, 2D paratope
dependency maps, and antigen-conditioned CDR Coordinate Gibbs sampling), see the companion
LogiCA repository:
  https://github.com/AI-SysBio/LogiCA/tree/main/applications/antibodies-abrank

Key Components Implemented Here (Cached-Feature Fast 1D Baseline):
  1. Stage-1 Self-Supervised Interface Pretraining (`_DualLMLogiCANet.stage1_infonce_loss`):
     Pretrains bidirectional cross-attention on unique bound `(Ab, Ag)` complexes from the training
     split using Contextual-Gain InfoNCE (`tau = 0.12`) with homologous-antigen false-negative
     masking (`cos(e_Ag,i, e_Ag,j) > 0.85`).
  2. Within-Scaffold Grouped Contrastive Batches + 3-State Bradley-Terry + Neutral Deadzone Hinge +
     2x2 Holo-vs-Apo Double-Difference Tetrads (`build_scaffold_grouped_batches`,
     `logica_contrastive_3state_loss`).
  3. Scaffold-Invariant Hot-Spot Impact Scorer (`ScaffoldInvariantImpactScorer`):
     Scores hot-spot impact `q(1, 2) = +q(2, 1)` without memorizing parental scaffold identities.

Strict Anti-Leakage & Symmetry Guarantees:
  - Every directional score satisfies exact odd antisymmetry `f(1, 2) = -f(2, 1)`.
  - Every hot-spot impact score satisfies exact even symmetry `q(1, 2) = +q(2, 1)`.
  - At step 0, all cross-attention gates are zero-initialized (`tanh(0) = 0`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.nn.functional as F

from abrank.baselines.models import (
    AntisymmetricLinearRanker,
    SplitPairTensors,
    build_odd_linear_matrix,
    val_balanced_objective,
)
from abrank.evaluate import compute_per_antigen_macro_metrics


# ==============================================================================
# 1. Fast Warm-Started L-BFGS Ranker & Scaffold-Grouped 3-State Contrastive Loss
# ==============================================================================

class _FastSoftLinearRanker:
    """Fast float32 warm-started L-BFGS linear ranker with soft neutral deadzone regularization."""

    def __init__(
        self,
        c_grid: tuple[float, ...] = (3e-4, 1e-3, 3e-3),
        neutral_weight: float = 0.25,
        maxiter: int = 28,
    ) -> None:
        self.c_grid = c_grid
        self.neutral_weight = neutral_weight
        self.maxiter = maxiter
        self.scaler = StandardScaler(with_mean=False)
        self.w: np.ndarray | None = None
        self.best_c: float = 1e-3

    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        w_tr: np.ndarray | None,
        X_val: np.ndarray,
        val_df: pd.DataFrame,
        val_is_ag_mut: np.ndarray,
        labels_tr: np.ndarray | None = None,
    ) -> "_FastSoftLinearRanker":
        Xs = self.scaler.fit_transform(X_tr).astype(np.float32, copy=False)
        Xv = self.scaler.transform(X_val).astype(np.float32, copy=False)

        if labels_tr is not None:
            imp_m = labels_tr != 0
            neu_m = labels_tr == 0
        else:
            imp_m = np.ones(len(y_tr), dtype=bool)
            neu_m = np.zeros(len(y_tr), dtype=bool)

        X_imp = Xs[imp_m]
        y_sign = np.where(y_tr[imp_m] > 0, 1.0, -1.0).astype(np.float32)
        sw = (
            np.ones(len(X_imp), dtype=np.float32)
            if w_tr is None
            else w_tr[imp_m].astype(np.float32).copy()
        )
        sw /= max(float(sw.mean()), 1e-8)
        X_neu = Xs[neu_m] if neu_m.any() else None

        n_imp = max(len(X_imp), 1)
        n_neu = max(int(neu_m.sum()), 1)
        d = Xs.shape[1]
        w_cur = np.zeros(d, dtype=np.float64)

        best_sc = -1e9
        best_w = w_cur.copy()

        for c in sorted(self.c_grid):
            alpha = 1.0 / (float(c) * n_imp)
            nw = float(self.neutral_weight) / n_neu

            def _obj(w_vec: np.ndarray) -> tuple[float, np.ndarray]:
                wf = w_vec.astype(np.float32)
                z = X_imp @ wf
                yz = y_sign * z
                loss_i = float(np.sum(sw * np.logaddexp(0.0, -yz)) / n_imp)
                sig = 1.0 / (1.0 + np.exp(np.clip(yz, -30.0, 30.0)))
                grad_i = (X_imp.T @ (sw * (-y_sign * sig))) / float(n_imp)

                loss = loss_i + 0.5 * alpha * float(np.dot(w_vec, w_vec))
                grad = grad_i.astype(np.float64) + alpha * w_vec

                if X_neu is not None and self.neutral_weight > 0:
                    zn = X_neu @ wf
                    loss += 0.5 * nw * float(np.dot(zn, zn))
                    grad += nw * (X_neu.T @ zn).astype(np.float64)
                return loss, grad

            res = minimize(
                _obj,
                w_cur,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": self.maxiter, "ftol": 1e-5},
            )
            w_cur = res.x
            preds_v = Xv @ w_cur.astype(np.float32)
            sc = _val_local_macro_score(val_df, None, preds_v, val_is_ag_mut=val_is_ag_mut)
            if sc > best_sc:
                best_sc = sc
                best_w = w_cur.copy()
                self.best_c = c

        self.w = best_w.astype(np.float32)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        Xs = self.scaler.transform(X).astype(np.float32, copy=False)
        return (Xs @ self.w).astype(np.float64)


def select_stratified_local_indices(
    ag_emb: np.ndarray,
    is_local: np.ndarray,
    labels: np.ndarray,
    max_pairs: int = 3200,
    seed: int = 42,
    ab_emb: np.ndarray | None = None,
    per_scaf_max: int | None = None,
) -> np.ndarray:
    """Select a stratified local training subset balanced across parental antigen scaffolds."""
    loc_idx = np.flatnonzero(is_local)
    if len(loc_idx) <= max_pairs:
        return loc_idx
    rng = np.random.default_rng(seed)
    sig = np.round(ag_emb[loc_idx, :3] * 100.0).astype(np.int32)
    if ab_emb is not None and len(ab_emb) == len(ag_emb):
        sig_ab = np.round(ab_emb[loc_idx, :2] * 100.0).astype(np.int32)
        sig = np.concatenate([sig, sig_ab], axis=1)
    _, inv = np.unique(sig, axis=0, return_inverse=True)
    n_scaf = int(inv.max()) + 1 if len(inv) > 0 else 1
    per_scaf_cap = max(40, int(np.ceil(max_pairs / max(n_scaf, 1))) * 2)
    if per_scaf_max is not None:
        per_scaf_cap = min(per_scaf_max, per_scaf_cap)

    kept: list[np.ndarray] = []
    overflow: list[np.ndarray] = []
    for g in range(n_scaf):
        g_ix = loc_idx[inv == g]
        if len(g_ix) <= per_scaf_cap:
            kept.append(g_ix)
        else:
            lbl_g = labels[g_ix]
            pos = rng.permutation(g_ix[lbl_g > 0])
            neg = rng.permutation(g_ix[lbl_g < 0])
            neu = rng.permutation(g_ix[lbl_g == 0])
            k3 = max(1, per_scaf_cap // 3)
            sel = np.concatenate([pos[:k3], neg[:k3], neu[: max(1, per_scaf_cap - 2 * k3)]])
            kept.append(sel)
            rem = np.setdiff1d(g_ix, sel, assume_unique=True)
            if len(rem) > 0:
                overflow.append(rem)

    out = np.concatenate(kept) if kept else loc_idx
    if len(out) > max_pairs:
        out = rng.choice(out, size=max_pairs, replace=False)
    elif len(out) < max_pairs and overflow and per_scaf_max is None:
        rem_all = np.concatenate(overflow)
        take = min(max_pairs - len(out), len(rem_all))
        out = np.concatenate([out, rng.choice(rem_all, size=take, replace=False)])
    return np.sort(out)


def build_scaffold_grouped_batches(
    ag_emb: np.ndarray,
    valid_indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    ab_emb: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Group training pairs by parental antigen scaffold so each mini-batch contrasts
    mutations on the same parental complex (`Within-Scaffold Contrastive Batching`).
    """
    if len(valid_indices) == 0:
        return []
    sig = np.round(ag_emb[valid_indices, :3] * 100.0).astype(np.int32)
    if ab_emb is not None and len(ab_emb) == len(ag_emb):
        sig_ab = np.round(ab_emb[valid_indices, :2] * 100.0).astype(np.int32)
        sig = np.concatenate([sig, sig_ab], axis=1)
    _, inv = np.unique(sig, axis=0, return_inverse=True)
    n_groups = int(inv.max()) + 1 if len(inv) > 0 else 1
    group_order = rng.permutation(n_groups)

    ordered: list[np.ndarray] = []
    for g in group_order:
        g_idx = valid_indices[inv == g]
        ordered.append(rng.permutation(g_idx))
    flat = np.concatenate(ordered)
    return [flat[s : s + batch_size] for s in range(0, len(flat), batch_size)]


def logica_contrastive_3state_loss(
    dir_logit: torch.Tensor,
    apo_logit: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    tau_bt: float = 0.45,
    eps_neu: float = 0.20,
    margin_ord: float = 0.35,
    lambda_neu: float = 0.35,
    lambda_tetrad: float = 0.25,
) -> torch.Tensor:
    """Unified 3-State Contrastive + Neutral Deadzone + 2x2 Double-Difference Tetrad Loss."""
    imp_mask = (labels != 0).float()
    neu_mask = (labels == 0).float()
    y_sign = labels.float()

    bt_per = F.softplus(-y_sign * dir_logit / tau_bt)
    loss_bt = (bt_per * imp_mask * weights).sum() / ((imp_mask * weights).sum().clamp(min=1.0))

    neu_hinge = F.relu(torch.abs(dir_logit) - eps_neu).pow(2)
    loss_neu = (neu_hinge * neu_mask * weights).sum() / ((neu_mask * weights).sum().clamp(min=1.0))

    pos_idx = torch.nonzero(labels > 0, as_tuple=False).squeeze(-1)
    neu_idx = torch.nonzero(labels == 0, as_tuple=False).squeeze(-1)
    neg_idx = torch.nonzero(labels < 0, as_tuple=False).squeeze(-1)
    loss_ord = torch.tensor(0.0, device=dir_logit.device)
    if len(pos_idx) > 0 and len(neu_idx) > 0:
        d_pn = dir_logit[pos_idx[:32], None] - dir_logit[None, neu_idx[:32]]
        loss_ord = loss_ord + F.relu(margin_ord - d_pn).mean()
    if len(neu_idx) > 0 and len(neg_idx) > 0:
        d_nn = dir_logit[neu_idx[:32], None] - dir_logit[None, neg_idx[:32]]
        loss_ord = loss_ord + F.relu(margin_ord - d_nn).mean()

    dd_bind = dir_logit - apo_logit.detach()
    loss_tetrad = (F.softplus(-y_sign * dd_bind / tau_bt) * imp_mask * weights).sum() / (
        (imp_mask * weights).sum().clamp(min=1.0)
    )

    return loss_bt + lambda_neu * loss_neu + 0.20 * loss_ord + lambda_tetrad * loss_tetrad


# ==============================================================================
# 2. Scaffold-Invariant Hot-Spot Impact Scorer
# ==============================================================================

class ScaffoldInvariantImpactScorer:
    """Even-symmetric (`q(1, 2) == +q(2, 1)`) Hot-Spot & Mutation Impact Scorer
    that avoids memorizing parental scaffold identity embeddings on unseen `<50%` antigens.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.scaler = StandardScaler()
        self.clf: LogisticRegression | None = None
        self.prior_w: float = 0.50
        self.prior_med: float = 0.0
        self.prior_std: float = 1.0
        self.kl_std: float = 1.0
        self.dir_std: float = 1.0

    def _prior_score(
        self,
        sp: SplitPairTensors,
        dir_logits: np.ndarray,
        kl_imp: np.ndarray | None = None,
    ) -> np.ndarray:
        ie = sp.impact_even
        n = len(sp.y)
        if ie.shape[1] >= 20:
            phys_1d = (
                0.35 * (ie[:, 9] + ie[:, 15])
                + 0.35 * (ie[:, 11] + ie[:, 17] + ie[:, 19])
                + 0.25 * (ie[:, 7] + ie[:, 13] + ie[:, 10] + ie[:, 16])
                + 0.25 * (ie[:, 5] + ie[:, 6] + 0.5 * (ie[:, 0] + ie[:, 2]) - 0.5 * ie[:, 4])
            )
        else:
            phys_1d = np.mean(np.abs(ie), axis=1)

        fb_imp = ie[:, 31] + ie[:, 32] if ie.shape[1] >= 33 else np.mean(np.abs(ie), axis=1)
        esmc_abl_imp = sp.zs_impact_dict.get("esmc_ablang2_zeroshot", fb_imp)
        esmc_imp = sp.zs_impact_dict.get("esmc_300m_zeroshot", fb_imp)
        esm2_imp = sp.zs_impact_dict.get("esm2_150m_zeroshot", esmc_imp)
        site_1d = 0.40 * esmc_abl_imp + 0.35 * esm2_imp + 0.25 * esmc_imp

        abs_d = np.abs(dir_logits) / max(self.dir_std, 1e-6)
        kl_n = (kl_imp / max(self.kl_std, 1e-6)) if kl_imp is not None else np.zeros(n, dtype=np.float64)
        return 0.50 * phys_1d + 0.20 * site_1d + 0.20 * abs_d + 0.10 * kl_n

    def _build_features(
        self,
        sp: SplitPairTensors,
        dir_logits: np.ndarray,
        kl_imp: np.ndarray | None = None,
    ) -> np.ndarray:
        if sp.impact_even.shape[1] >= 34:
            ie_clean = np.concatenate([sp.impact_even[:, :21], sp.impact_even[:, 31:]], axis=1)
        else:
            ie_clean = sp.impact_even
        parts = [ie_clean, np.abs(dir_logits)[:, None]]
        if kl_imp is not None:
            parts.append(kl_imp[:, None])
        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit(
        self,
        tr_sp: SplitPairTensors,
        val_sp: SplitPairTensors,
        dir_tr: np.ndarray,
        dir_val: np.ndarray,
        val_df: pd.DataFrame,
        kl_tr: np.ndarray | None = None,
        kl_val: np.ndarray | None = None,
    ) -> "ScaffoldInvariantImpactScorer":
        loc_tr = tr_sp.is_same_ag | (~tr_sp.is_ab_mut)
        loc_val = val_sp.is_same_ag | (~val_sp.is_ab_mut)
        if loc_tr.sum() < 10:
            loc_tr = np.ones(len(tr_sp.y), dtype=bool)
        if loc_val.sum() < 5:
            loc_val = np.ones(len(val_sp.y), dtype=bool)

        self.dir_std = max(float(np.std(np.abs(dir_tr[loc_tr]))), 1e-6)
        if kl_tr is not None:
            self.kl_std = max(float(np.std(kl_tr[loc_tr])), 1e-6)

        X_tr_all = self._build_features(tr_sp, dir_tr, kl_tr)
        X_val_all = self._build_features(val_sp, dir_val, kl_val)

        self.scaler.fit(X_tr_all[loc_tr])
        X_tr = self.scaler.transform(X_tr_all[loc_tr])
        X_v = self.scaler.transform(X_val_all[loc_val])
        y_tr = tr_sp.is_impactful[loc_tr].astype(int)
        y_v = val_sp.is_impactful[loc_val].astype(int)

        prior_tr_raw = self._prior_score(tr_sp, dir_tr, kl_tr)[loc_tr]
        self.prior_med = float(np.median(prior_tr_raw))
        self.prior_std = max(float(np.std(prior_tr_raw)), 1e-6)

        prior_v_raw = self._prior_score(val_sp, dir_val, kl_val)[loc_val]
        prior_v = 1.0 / (1.0 + np.exp(-np.clip((prior_v_raw - self.prior_med) / self.prior_std, -20, 20)))

        val_sub = val_df.iloc[np.flatnonzero(loc_val)].reset_index(drop=True)
        z_zero = np.zeros(len(X_v), dtype=np.float64)

        best_sc = -1.0
        for c in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
            m = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs", max_iter=500)
            if len(np.unique(y_tr)) < 2:
                break
            m.fit(X_tr, y_tr)
            p_clf = m.predict_proba(X_v)[:, 1]
            for pw in (0.35, 0.55, 0.75, 0.90):
                p_blend = (1.0 - pw) * p_clf + pw * prior_v
                if "Ag_group_1" in val_sub.columns:
                    sc = float(compute_per_antigen_macro_metrics(val_sub, z_zero, p_blend, min_pairs=5)["auc_impact"] or 0.5)
                elif len(np.unique(y_v)) >= 2:
                    sc = float(roc_auc_score(y_v, p_blend))
                else:
                    sc = 0.5
                if sc > best_sc:
                    best_sc = sc
                    self.clf = m
                    self.prior_w = pw
        return self

    def predict_proba(
        self,
        sp: SplitPairTensors,
        dir_logits: np.ndarray,
        kl_imp: np.ndarray | None = None,
    ) -> np.ndarray:
        prior_raw = self._prior_score(sp, dir_logits, kl_imp)
        prior_p = 1.0 / (1.0 + np.exp(-np.clip((prior_raw - self.prior_med) / self.prior_std, -20, 20)))
        if self.clf is None:
            p = prior_p
        else:
            X = self.scaler.transform(self._build_features(sp, dir_logits, kl_imp))
            p_clf = self.clf.predict_proba(X)[:, 1]
            p = (1.0 - self.prior_w) * p_clf + self.prior_w * prior_p
        same_partner = sp.is_same_ag | (~sp.is_ab_mut)
        p = p.astype(np.float64)
        p[~same_partner] = 1.0
        return p


# ==============================================================================
# 3. Neural Network: `_DualLMLogiCANet` (1D Logit Cross-Attention)
# ==============================================================================

class _LogiCACrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention block with orthogonal projections and `tanh(0)` gates."""

    def __init__(self, width: int = 96, heads: int = 4, dropout: float = 0.15) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(width)
        self.norm_kv = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q_n = self.norm_q(query)
        kv_n = self.norm_kv(context)
        attn_out, _ = self.attn(q_n, kv_n, kv_n, need_weights=False)
        h = query + self.dropout(attn_out)
        return h + self.dropout(self.ff(self.norm_ff(h)))


class _DualLMLogiCANet(nn.Module):
    """1D `LogiCA` native-logit cross-attention network with:
      - Biological token slots for paired Antibody (`AbLang2` + `ESMC`) and Antigen (`ESMC`),
      - Stage-1 self-supervised partner-matching InfoNCE pretraining (`stage1_infonce_loss`),
      - Zero-initialized `tanh(0)` gates preserving exact step-0 base PLM equivalence,
      - Thermodynamic `apo` + `holo - apo` double-difference scorer, and
      - Partner-induced cross-attention shift for hot-spot impact scoring.
    """

    def __init__(
        self,
        ab_dim: int,
        ag_dim: int,
        odd_dim: int,
        even_dim: int = 0,
        width: int = 96,
        heads: int = 4,
        n_ab_tokens: int = 6,
        n_ag_tokens: int = 3,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.width = width
        self.n_ab_tokens = n_ab_tokens
        self.n_ag_tokens = n_ag_tokens
        self.even_dim = even_dim

        self.proj_ab_in = nn.Linear(ab_dim, width * n_ab_tokens, bias=False)
        self.proj_ag_in = nn.Linear(ag_dim, width * n_ag_tokens, bias=False)
        nn.init.orthogonal_(self.proj_ab_in.weight)
        nn.init.orthogonal_(self.proj_ag_in.weight)

        self.ab_slot_emb = nn.Parameter(torch.randn(1, n_ab_tokens, width) * 0.02)
        self.ag_slot_emb = nn.Parameter(torch.randn(1, n_ag_tokens, width) * 0.02)

        self.ab_to_ag = _LogiCACrossAttentionBlock(width=width, heads=heads, dropout=dropout)
        self.ag_to_ab = _LogiCACrossAttentionBlock(width=width, heads=heads, dropout=dropout)

        self.gate_ab = nn.Parameter(torch.zeros(1))
        self.gate_ag = nn.Parameter(torch.zeros(1))
        self.gate_logit = nn.Parameter(torch.zeros(1))

        self.base_llr_proj = nn.Linear(odd_dim, 1, bias=False)
        self.ctx_logit_head = nn.Sequential(
            nn.LayerNorm(width * 4 + odd_dim),
            nn.Linear(width * 4 + odd_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1, bias=False),
        )
        nn.init.normal_(self.ctx_logit_head[-1].weight, mean=0.0, std=0.02)

        self.partner_proj_ab = nn.Linear(width, width, bias=False)
        self.partner_proj_ag = nn.Linear(width, width, bias=False)
        nn.init.orthogonal_(self.partner_proj_ab.weight)
        nn.init.orthogonal_(self.partner_proj_ag.weight)

    def _tokens(self, ab: torch.Tensor, ag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = ab.shape[0]
        t_ab = self.proj_ab_in(ab).view(bsz, self.n_ab_tokens, self.width) + self.ab_slot_emb
        t_ag = self.proj_ag_in(ag).view(bsz, self.n_ag_tokens, self.width) + self.ag_slot_emb
        return t_ab, t_ag

    def _encode_complex(self, ab: torch.Tensor, ag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t_ab, t_ag = self._tokens(ab, ag)
        ab_ctx = self.ag_to_ab(t_ab, t_ag)
        ag_ctx = self.ab_to_ag(t_ag, t_ab)

        h_ab_apo = t_ab.mean(dim=1)
        h_ag_apo = t_ag.mean(dim=1)
        h_ab_holo = (t_ab + torch.tanh(self.gate_ab) * ab_ctx).mean(dim=1)
        h_ag_holo = (t_ag + torch.tanh(self.gate_ag) * ag_ctx).mean(dim=1)
        return torch.cat([h_ab_holo, h_ag_holo], dim=-1), torch.cat([h_ab_apo, h_ag_apo], dim=-1)

    def stage1_infonce_loss(self, ab: torch.Tensor, ag: torch.Tensor, tau: float = 0.12) -> torch.Tensor:
        t_ab, t_ag = self._tokens(ab, ag)
        ab_ctx = self.ag_to_ab(t_ab, t_ag).mean(dim=1)
        ag_ctx = self.ab_to_ag(t_ag, t_ab).mean(dim=1)
        z_ab = F.normalize(self.partner_proj_ab(t_ab.mean(dim=1) + ab_ctx), dim=-1)
        z_ag = F.normalize(self.partner_proj_ag(t_ag.mean(dim=1) + ag_ctx), dim=-1)
        logits = torch.matmul(z_ab, z_ag.T) / tau

        with torch.no_grad():
            ag_norm = F.normalize(ag, dim=-1)
            sim_ag = torch.matmul(ag_norm, ag_norm.T)
            eye = torch.eye(len(ab), dtype=torch.bool, device=ab.device)
            fn_mask = (sim_ag > 0.85) & (~eye)
        logits = logits.masked_fill(fn_mask, -1e9)
        targets = torch.arange(len(ab), device=ab.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))

    def forward(
        self,
        ab1: torch.Tensor,
        ab2: torch.Tensor,
        ag1: torch.Tensor,
        ag2: torch.Tensor,
        x_odd: torch.Tensor,
        x_even: torch.Tensor | None = None,
        *,
        return_apo: bool = False,
    ):
        c1_holo, c1_apo = self._encode_complex(ab1, ag1)
        c2_holo, c2_apo = self._encode_complex(ab2, ag2)

        apo_llr = self.base_llr_proj(x_odd).squeeze(-1)
        dc = F.normalize(c1_holo - c2_holo, dim=-1, eps=1e-6)
        c_mid = F.normalize(0.5 * (c1_holo + c2_holo), dim=-1, eps=1e-6)
        delta_12 = self.ctx_logit_head(torch.cat([dc, dc * c_mid, x_odd], dim=-1)).squeeze(-1)
        delta_21 = self.ctx_logit_head(torch.cat([-dc, -dc * c_mid, -x_odd], dim=-1)).squeeze(-1)
        dir_logit = apo_llr + torch.tanh(self.gate_logit) * 0.5 * (delta_12 - delta_21)

        ctx_shift = torch.norm((c1_holo - c1_apo) - (c2_holo - c2_apo), dim=-1)
        if x_even is not None:
            ctx_shift = ctx_shift + 0.05 * torch.norm(x_even, dim=-1)
        if return_apo:
            return dir_logit, apo_llr, ctx_shift
        return dir_logit, ctx_shift


# ==============================================================================
# 4. Trainer: `LogiCA3StateTrainer` (1D Logit Cross-Attention)
# ==============================================================================

def _val_local_macro_score(
    val_df: pd.DataFrame,
    val_sp: SplitPairTensors | None,
    dir_preds: np.ndarray,
    imp_preds: np.ndarray | None = None,
    *,
    val_is_ag_mut: np.ndarray | None = None,
) -> float:
    """Validation objective focused on within-antigen Local (`L1-L3` + `Candidate`) macro ranking."""
    if val_sp is not None:
        loc_m = val_sp.is_same_ag | (~val_sp.is_ab_mut)
        ag_mut = val_sp.is_ag_mut
    elif "Category" in val_df.columns:
        loc_m = val_df["Category"].isin(["L1", "L2", "L3", "Candidate"]).to_numpy(dtype=bool)
        ag_mut = val_is_ag_mut if val_is_ag_mut is not None else np.zeros(len(val_df), dtype=bool)
    else:
        loc_m = np.ones(len(val_df), dtype=bool)
        ag_mut = val_is_ag_mut if val_is_ag_mut is not None else np.zeros(len(val_df), dtype=bool)

    if loc_m.sum() >= 5 and "Ag_group_1" in val_df.columns:
        sub_df = val_df.iloc[np.flatnonzero(loc_m)].reset_index(drop=True)
        d_sub = dir_preds[loc_m]
        i_sub = imp_preds[loc_m] if imp_preds is not None else np.abs(d_sub)
        m3 = compute_per_antigen_macro_metrics(sub_df, d_sub, i_sub, min_pairs=5)
        sc_dir = float(m3["auc_dir"] or 0.5)
        sc_ord = float(m3["auc_ordinal"] or sc_dir)
        return 0.55 * sc_dir + 0.45 * sc_ord
    return val_balanced_objective(val_df, dir_preds, ag_mut)


def _is_unseen_antigen_split(tr_sp: SplitPairTensors, val_sp: SplitPairTensors) -> bool:
    """Return True when `val_sp` evaluates on held-out `<50%` antigen clusters."""
    if len(tr_sp.y) == 0 or len(val_sp.y) == 0:
        return True
    tr_ag = np.round(tr_sp.esmc_ag_1[:, :3] * 100.0).astype(np.int32)
    val_ag = np.round(val_sp.esmc_ag_1[:, :3] * 100.0).astype(np.int32)
    tr_set = {tuple(r) for r in np.unique(tr_ag, axis=0)}
    ov = float(np.mean([tuple(r) in tr_set for r in val_ag]))
    return ov < 0.25


def _compute_scaffold_balanced_weights(
    tr_sp: SplitPairTensors,
    w_tr: np.ndarray,
    power: float = 0.85,
) -> np.ndarray:
    """Reweight local training pairs by inverse antigen-scaffold frequency (`1 / n_scaf^power`)."""
    w_out = w_tr.astype(np.float32).copy()
    is_loc = tr_sp.is_same_ag | (~tr_sp.is_ab_mut)
    loc_idx = np.flatnonzero(is_loc)
    if len(loc_idx) < 20:
        return w_out
    ag_sig = np.round(tr_sp.esmc_ag_1[loc_idx, :3] * 100.0).astype(np.int32)
    _, inv, counts = np.unique(ag_sig, axis=0, return_inverse=True, return_counts=True)
    scaf_w = 1.0 / (np.maximum(counts[inv].astype(np.float32), 25.0) ** power)
    scaf_w /= max(float(scaf_w.mean()), 1e-8)
    w_out[loc_idx] *= scaf_w
    w_out /= max(float(w_out.mean()), 1e-8)
    return w_out


class LogiCA3StateTrainer:
    """Leak-free 3-state trainer for `LogiCA (1D Logit Cross-Attn)`."""

    def __init__(
        self,
        seed: int = 42,
        epochs: int = 16,
        pretrain_epochs: int = 6,
        batch_size: int = 256,
        lr: float = 9e-4,
        weight_decay: float = 2e-2,
    ) -> None:
        self.seed = seed
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.ab_scaler = StandardScaler()
        self.ag_scaler = StandardScaler()
        self.odd_scaler = StandardScaler(with_mean=False)
        self.net: _DualLMLogiCANet | None = None
        self.thermo_loc: _FastSoftLinearRanker | None = None
        self.emb_loc: _FastSoftLinearRanker | None = None
        self.glb_head: AntisymmetricLinearRanker | None = None
        self.imp_scorer: ScaffoldInvariantImpactScorer | None = None
        self.weights_loc: tuple[float, float, float, float] = (0.20, 0.20, 0.25, 0.35)
        self.comp_stds: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
        self.gamma: float = 0.5
        self.alpha: float = 1.0

    @staticmethod
    def _ab_ag_blocks(sp: SplitPairTensors) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ab1 = np.concatenate([sp.esmc_ab_1, sp.ablang_1], axis=1)
        ab2 = np.concatenate([sp.esmc_ab_2, sp.ablang_2], axis=1)
        ag1 = sp.esmc_ag_1
        ag2 = sp.esmc_ag_2
        return ab1, ab2, ag1, ag2

    @staticmethod
    def _is_cand_pair_mask(sp: SplitPairTensors) -> np.ndarray:
        same_p = sp.is_same_ag | (~sp.is_ab_mut)
        if sp.impact_even.shape[1] >= 7:
            edit_log = sp.impact_even[:, 5] + sp.impact_even[:, 6]
            return same_p & (edit_log > 1.45)
        return np.zeros(len(sp.y), dtype=bool)

    @staticmethod
    def _is_vhh_mask(sp: SplitPairTensors) -> np.ndarray:
        return (np.linalg.norm(sp.ablang_1, axis=1) < 1e-6) & (np.linalg.norm(sp.ablang_2, axis=1) < 1e-6)

    @staticmethod
    def _clean_zs_and_biophys(sp: SplitPairTensors) -> tuple[np.ndarray, np.ndarray]:
        zf = sp.zs_features.astype(np.float32).copy()
        bo = sp.biophys_odd.astype(np.float32).copy()
        is_cand = LogiCA3StateTrainer._is_cand_pair_mask(sp)
        len_diff = np.abs(bo[:, 17]) > 1e-5 if bo.shape[1] >= 18 else is_cand
        mask_len = is_cand | len_diff
        if mask_len.any():
            if zf.shape[1] >= 8:
                zf[mask_len, 2] = 0.0
                zf[mask_len, 7] = 0.0
            if bo.shape[1] >= 18:
                bo[mask_len, 5] = 0.0
                bo[mask_len, 17] = 0.0
        return zf, bo

    @staticmethod
    def _1d_zs_prior(sp: SplitPairTensors) -> np.ndarray:
        same_p = sp.is_same_ag | (~sp.is_ab_mut)
        is_cand = LogiCA3StateTrainer._is_cand_pair_mask(sp)
        is_vhh = LogiCA3StateTrainer._is_vhh_mask(sp)
        zf = sp.zs_features
        fb_esmc = (zf[:, 4] + zf[:, 5]) if zf.shape[1] >= 6 else zf[:, 0]
        fb_abl = (fb_esmc + 0.5 * zf[:, 8]) if zf.shape[1] >= 9 else fb_esmc
        fb_cos = zf[:, 9] if zf.shape[1] >= 10 else zf[:, 0]

        esmc_abl = sp.zs_dict.get("esmc_ablang2_zeroshot", fb_abl)
        esmc_300 = sp.zs_dict.get("esmc_300m_zeroshot", fb_esmc)
        esm2_150 = sp.zs_dict.get("esm2_150m_zeroshot", esmc_300)
        esmc_cos = sp.zs_dict.get("esmc_cosine", fb_cos)

        loc_zs = 0.40 * esmc_abl + 0.35 * esmc_300 + 0.25 * esm2_150
        if is_cand.any() and sp.biophys_odd.shape[1] >= 15:
            bo = sp.biophys_odd
            bio_cand = 0.30 * bo[:, 0] + 0.35 * bo[:, 4] + 0.35 * bo[:, 6] + 0.30 * bo[:, 14]
            bio_vhh = 0.35 * bo[:, 1] + 0.35 * bo[:, 2] + 0.30 * bo[:, 4] + 0.30 * bo[:, 14]
            cand_zs = np.where(
                is_vhh,
                0.35 * esmc_300 + 0.25 * esm2_150 + 0.20 * (esmc_cos * 5.0) + 0.20 * bio_vhh,
                0.35 * esmc_300 + 0.30 * (esmc_cos * 5.0) + 0.35 * bio_cand,
            )
            loc_zs = np.where(is_cand, cand_zs, loc_zs)
        glb_zs = 0.45 * esm2_150 + 0.35 * esmc_300 + 0.20 * esmc_cos
        return np.where(same_p, loc_zs, glb_zs).astype(np.float64)

    @staticmethod
    def _thermo_mat(sp: SplitPairTensors) -> np.ndarray:
        zf, bo = LogiCA3StateTrainer._clean_zs_and_biophys(sp)
        return np.concatenate([zf, bo], axis=1).astype(np.float32)

    @staticmethod
    def _emb_loc_mat(sp: SplitPairTensors) -> np.ndarray:
        zf, bo = LogiCA3StateTrainer._clean_zs_and_biophys(sp)
        return build_odd_linear_matrix(
            [sp.esmc_ab_1, sp.ablang_1, sp.esmc_ag_1],
            [sp.esmc_ab_2, sp.ablang_2, sp.esmc_ag_2],
            extra_odd=np.concatenate([zf, bo], axis=1),
            include_unit_direction=True,
        )

    def _prep_tensors(self, sp: SplitPairTensors, fit: bool = False):
        ab1_r, ab2_r, ag1_r, ag2_r = self._ab_ag_blocks(sp)
        odd_r = self._thermo_mat(sp)
        if fit:
            self.ab_scaler.fit(np.vstack([ab1_r, ab2_r]))
            self.ag_scaler.fit(np.vstack([ag1_r, ag2_r]))
            self.odd_scaler.fit(odd_r)
        return (
            self.ab_scaler.transform(ab1_r).astype(np.float32),
            self.ab_scaler.transform(ab2_r).astype(np.float32),
            self.ag_scaler.transform(ag1_r).astype(np.float32),
            self.ag_scaler.transform(ag2_r).astype(np.float32),
            self.odd_scaler.transform(odd_r).astype(np.float32),
        )

    def _neural_raw(self, sp: SplitPairTensors) -> tuple[np.ndarray, np.ndarray]:
        assert self.net is not None
        ab1, ab2, ag1, ag2, xo = self._prep_tensors(sp, fit=False)
        n = len(xo)
        d_out = np.zeros(n, dtype=np.float64)
        kl_out = np.zeros(n, dtype=np.float64)
        with torch.no_grad():
            for s in range(0, n, 512):
                e = min(s + 512, n)
                dl, kl_s = self.net(
                    torch.from_numpy(ab1[s:e]),
                    torch.from_numpy(ab2[s:e]),
                    torch.from_numpy(ag1[s:e]),
                    torch.from_numpy(ag2[s:e]),
                    torch.from_numpy(xo[s:e]),
                )
                d_out[s:e] = dl.numpy().astype(np.float64)
                kl_out[s:e] = kl_s.numpy().astype(np.float64)
        return d_out, kl_out

    def fit(
        self,
        tr_sp: SplitPairTensors,
        val_sp: SplitPairTensors,
        val_df: pd.DataFrame,
        w_tr: np.ndarray,
    ) -> "LogiCA3StateTrainer":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        ab1, ab2, ag1, ag2, xo = self._prep_tensors(tr_sp, fit=True)
        v_ab1, v_ab2, v_ag1, v_ag2, v_xo = self._prep_tensors(val_sp, fit=False)

        net = _DualLMLogiCANet(
            ab_dim=ab1.shape[1],
            ag_dim=ag1.shape[1],
            odd_dim=xo.shape[1],
        )
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        sig_c = np.round(ab1[:, :3] * 100.0).astype(np.int32)
        _, uniq_idx = np.unique(sig_c, axis=0, return_index=True)
        uniq_idx = np.sort(uniq_idx)
        for _ in range(self.pretrain_epochs):
            net.train()
            p_u = rng.permutation(uniq_idx)
            for s in range(0, len(p_u), 128):
                u_b = p_u[s : s + 128]
                if len(u_b) < 4:
                    continue
                opt.zero_grad()
                loss_s1 = net.stage1_infonce_loss(
                    torch.from_numpy(ab1[u_b]),
                    torch.from_numpy(ag1[u_b]),
                )
                loss_s1.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()

        is_unseen_ag = _is_unseen_antigen_split(tr_sp, val_sp)
        ab_grp = ab1 if is_unseen_ag else None
        is_loc_tr = tr_sp.is_same_ag | (~tr_sp.is_ab_mut)
        sub_loc_idx = select_stratified_local_indices(
            ag1,
            is_loc_tr,
            tr_sp.labels,
            max_pairs=3200,
            seed=self.seed,
            ab_emb=ab_grp,
            per_scaf_max=220 if is_unseen_ag else None,
        )
        if len(sub_loc_idx) < 10:
            sub_loc_idx = np.arange(min(len(tr_sp.y), 3200), dtype=np.int64)
        w_clean = np.ones(len(tr_sp.y), dtype=np.float32)
        best_val = -1.0
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        for _ in range(min(self.epochs, 10)):
            net.train()
            batches = build_scaffold_grouped_batches(ag1, sub_loc_idx, self.batch_size, rng, ab_emb=ab_grp)
            for idx in batches:
                if len(idx) < 2:
                    continue
                flip = rng.random(len(idx)) < 0.5
                b_ab1 = torch.from_numpy(np.where(flip[:, None], ab2[idx], ab1[idx]))
                b_ab2 = torch.from_numpy(np.where(flip[:, None], ab1[idx], ab2[idx]))
                b_ag1 = torch.from_numpy(np.where(flip[:, None], ag2[idx], ag1[idx]))
                b_ag2 = torch.from_numpy(np.where(flip[:, None], ag1[idx], ag2[idx]))
                b_xo = torch.from_numpy(np.where(flip[:, None], -xo[idx], xo[idx]))
                b_lbl = torch.from_numpy(np.where(flip, -tr_sp.labels[idx], tr_sp.labels[idx]).astype(np.int64))
                b_w = torch.from_numpy(w_clean[idx])

                opt.zero_grad()
                dl, apo_l, _ = net(b_ab1, b_ab2, b_ag1, b_ag2, b_xo, return_apo=True)
                loss = logica_contrastive_3state_loss(dl, apo_l, b_lbl, b_w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()

            net.eval()
            with torch.no_grad():
                vdl, vkl = net(
                    torch.from_numpy(v_ab1),
                    torch.from_numpy(v_ab2),
                    torch.from_numpy(v_ag1),
                    torch.from_numpy(v_ag2),
                    torch.from_numpy(v_xo),
                )
                vd_np = vdl.numpy().astype(np.float64)
                vkl_np = vkl.numpy().astype(np.float64)
            sc = _val_local_macro_score(val_df, val_sp, vd_np, vkl_np)
            if sc > best_val:
                best_val = sc
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        net.load_state_dict(best_state)
        net.eval()
        self.net = net

        w_lin = _compute_scaffold_balanced_weights(tr_sp, w_tr, power=0.85) if is_unseen_ag else w_tr
        X_th_tr = self._thermo_mat(tr_sp)
        X_th_val = self._thermo_mat(val_sp)
        self.thermo_loc = _FastSoftLinearRanker(
            c_grid=(3e-4, 1e-3, 3e-3, 1e-2), maxiter=32
        ).fit(
            X_th_tr, tr_sp.y, w_lin,
            X_th_val, val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels,
        )

        X_emb_tr = self._emb_loc_mat(tr_sp)
        X_emb_val = self._emb_loc_mat(val_sp)
        self.emb_loc = _FastSoftLinearRanker(
            c_grid=(1e-4, 3e-4) if is_unseen_ag else (3e-4, 1e-3), maxiter=25
        ).fit(
            X_emb_tr, tr_sp.y, w_lin,
            X_emb_val, val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels,
        )

        glb_tr_m = ~is_loc_tr
        if glb_tr_m.sum() >= 20:
            self.glb_head = AntisymmetricLinearRanker(
                c_grid=(1e-4, 1e-3, 1e-2), use_neutral_nullspace=False
            ).fit(
                tr_sp.zs_features[glb_tr_m], tr_sp.y[glb_tr_m], None,
                val_sp.zs_features, val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels[glb_tr_m],
            )

        tr_sub_sp = tr_sp.subset(sub_loc_idx)
        zs_tr = self._1d_zs_prior(tr_sub_sp)
        zs_val = self._1d_zs_prior(val_sp)
        th_tr = self.thermo_loc.predict(X_th_tr[sub_loc_idx])
        th_val = self.thermo_loc.predict(X_th_val)
        nd_tr, kl_tr = self._neural_raw(tr_sub_sp)
        nd_val, kl_val = self._neural_raw(val_sp)
        em_tr = self.emb_loc.predict(X_emb_tr[sub_loc_idx])
        em_val = self.emb_loc.predict(X_emb_val)

        def _norm(v_tr: np.ndarray, v_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            sd = max(float(np.std(v_tr)), 1e-6)
            return v_tr / sd, v_eval / sd, sd

        zs_tr_n, zs_val_n, sd_zs = _norm(zs_tr, zs_val)
        th_tr_n, th_val_n, sd_th = _norm(th_tr, th_val)
        nd_tr_n, nd_val_n, sd_nd = _norm(nd_tr, nd_val)
        em_tr_n, em_val_n, sd_em = _norm(em_tr, em_val)
        self.comp_stds = (sd_zs, sd_th, sd_nd, sd_em)

        best_w = (0.15, 0.05, 0.50, 0.30) if is_unseen_ag else (0.15, 0.15, 0.30, 0.40)
        best_w_sc = -1.0
        candidates_w = (
            [
                (0.20, 0.10, 0.45, 0.25),
                (0.15, 0.05, 0.50, 0.30),
                (0.10, 0.05, 0.50, 0.35),
            ]
            if is_unseen_ag
            else [
                (0.20, 0.15, 0.30, 0.35),
                (0.15, 0.15, 0.30, 0.40),
                (0.10, 0.10, 0.30, 0.50),
            ]
        )
        for w_cand in candidates_w:
            w0, w1, w2, w3 = w_cand
            cand_v = w0 * zs_val_n + w1 * th_val_n + w2 * nd_val_n + w3 * em_val_n
            sc_v = _val_local_macro_score(val_df, val_sp, cand_v)
            if sc_v > best_w_sc:
                best_w_sc = sc_v
                best_w = w_cand
        self.weights_loc = best_w

        w0, w1, w2, w3 = self.weights_loc
        dv_tr = w0 * zs_tr_n + w1 * th_tr_n + w2 * nd_tr_n + w3 * em_tr_n
        dv_val = w0 * zs_val_n + w1 * th_val_n + w2 * nd_val_n + w3 * em_val_n

        self.imp_scorer = ScaffoldInvariantImpactScorer(seed=self.seed).fit(
            tr_sub_sp, val_sp, dv_tr, dv_val, val_df, kl_tr=kl_tr, kl_val=kl_val
        )
        p_imp_val = self.imp_scorer.predict_proba(val_sp, dv_val, kl_imp=kl_val)

        best_ord = -1.0
        for gamma in (0.35, 0.60, 1.0, 1.5):
            for alpha in (0.5, 0.85, 1.2):
                pred_d = (p_imp_val ** alpha) * np.tanh(gamma * dv_val)
                if "Ag_group_1" in val_df.columns:
                    m = compute_per_antigen_macro_metrics(val_df, pred_d, p_imp_val, min_pairs=5)
                    sc = (m["auc_ordinal"] or 0.5) + 0.5 * (m["auc_dir"] or 0.5)
                else:
                    sc = val_balanced_objective(val_df, pred_d, val_sp.is_ag_mut)
                if sc > best_ord:
                    best_ord = sc
                    self.gamma = gamma
                    self.alpha = alpha
        return self

    def predict_3state(self, sp: SplitPairTensors) -> tuple[np.ndarray, np.ndarray]:
        assert self.thermo_loc is not None and self.emb_loc is not None and self.imp_scorer is not None
        zs_p = self._1d_zs_prior(sp)
        th_p = self.thermo_loc.predict(self._thermo_mat(sp))
        nd_p, kl_p = self._neural_raw(sp)
        em_p = self.emb_loc.predict(self._emb_loc_mat(sp))

        sd_zs, sd_th, sd_nd, sd_em = getattr(self, "comp_stds", (1.0, 1.0, 1.0, 1.0))

        def _np_s(v: np.ndarray, s: float) -> np.ndarray:
            return v / max(s, float(np.std(v)) if len(v) > 1 else s, 1e-6)

        zs_n = _np_s(zs_p, sd_zs)
        th_n = _np_s(th_p, sd_th)
        nd_n = _np_s(nd_p, sd_nd)
        em_n = _np_s(em_p, sd_em)
        w0, w1, w2, w3 = self.weights_loc
        dv_loc = w0 * zs_n + w1 * th_n + w2 * nd_n + w3 * em_n
        is_cand = self._is_cand_pair_mask(sp)
        if is_cand.any():
            dv_cand = 0.70 * zs_n + 0.20 * nd_n + 0.10 * th_n
            dv_loc = np.where(is_cand, dv_cand, dv_loc)

        same_p = sp.is_same_ag | (~sp.is_ab_mut)
        if (~same_p).any():
            glb_zs = zs_p / sd_zs
            if self.glb_head is not None:
                glb_lin = self.glb_head.predict(sp.zs_features)
                dv_glb = 0.65 * glb_zs + 0.35 * glb_lin
            else:
                dv_glb = glb_zs
            dv = np.where(same_p, dv_loc, dv_glb)
        else:
            dv = dv_loc

        p_imp = self.imp_scorer.predict_proba(sp, dv, kl_imp=kl_p)
        pred_diff = (p_imp ** self.alpha) * np.tanh(self.gamma * dv)
        return pred_diff.astype(np.float64), p_imp.astype(np.float64)
