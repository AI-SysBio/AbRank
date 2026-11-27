"""3D Structure-Conditioned Inverse Folding, Multi-Chain (`MINT`), Structure-Token (`SaProt`),
Normal-Mode CNN (`ANTIPASTI`), Geometric Interface GNN (`GearBind`), and Complex Graph
(`WALLE-Affinity`) 3-State Ranking Models for AbRank v2.

Every model strictly enforces:
  1. Zero data leakage: all scalers, PCA projections, linear probes, CNNs, and GNNs are fit
     exclusively on `train` and selected/calibrated exclusively on `val`.
  2. Exact antisymmetry $f(1, 2) = -f(2, 1)$ for directional ranking scores.
  3. Exact even symmetry $q(1, 2) = +q(2, 1)$ for mutation impact / hot-spot detection scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.nn.functional as F

from abrank.baselines.features import AA_TO_IDX
from abrank.baselines.models import (
    AntisymmetricLinearRanker,
    OrdinalHurdleRanker,
    SplitPairTensors,
    SymmetricImpactClassifier,
    TwoHeadRegimeRanker,
    build_odd_linear_matrix,
    sym_auc,
    val_balanced_objective,
)
from abrank.evaluate import compute_3state_metrics, compute_per_antigen_macro_metrics
from abrank.structures.resolver import (
    CHARGE_PH74,
    HYDROPHOBICITY,
    VOLUME_NORM,
    _cdr_residue_indices,
    align_query_to_chain,
)


@dataclass
class StructureSplitFeatures:
    """Pre-assembled 3D inverse-folding, biophysical, SaProt, MINT, DCCM, and 3D patch tensors for a split."""

    is_cocrystal: np.ndarray            # [N] bool (`both complexes resolved to experimental Ab-Ag co-crystals`)
    zs_dir: dict[str, np.ndarray]       # `esm_if1`, `protein_mpnn`, `antifold`, `prodigy_sasa`, `saprot_zs`, `mint_zs`
    zs_imp: dict[str, np.ndarray]       # Symmetric hot-spot impact scores `q(1,2) == +q(2,1)`
    struct_odd: np.ndarray              # [N, 22] strictly antisymmetric 3D interface + ΔΔlog p features
    struct_even: np.ndarray             # [N, 24] strictly symmetric 3D interface distance / ΔSASA / |ΔΔlog p| features
    if1_emb_1: np.ndarray               # [N, 2048] 3D GVP-Transformer complex pooled features (H_int, H_mean, L_mean, A_int)
    if1_emb_2: np.ndarray
    mpnn_emb_1: np.ndarray              # [N, 512] ProteinMPNN 3D encoder pooled features
    mpnn_emb_2: np.ndarray
    saprot_ab_1: np.ndarray             # [N, 3840] (VH mean, VL mean, VH cdr3; D=1280)
    saprot_ab_2: np.ndarray
    saprot_ag_1: np.ndarray             # [N, 2560] (Ag mean, Ag std; D=1280)
    saprot_ag_2: np.ndarray
    mint_ab_1: np.ndarray               # [N, 3840] (VH mean, VL mean, VH cdr3; D=1280 co-attended)
    mint_ab_2: np.ndarray
    mint_ag_1: np.ndarray               # [N, 2560] (Ag mean, Ag std; D=1280 co-attended)
    mint_ag_2: np.ndarray
    dccm_1: np.ndarray                  # [N, 2, 64, 64] float16 Normal-Mode + stiffness maps for ANTIPASTI
    dccm_2: np.ndarray
    patch_node_1: np.ndarray            # [N, K, D_node] float16 3D interface+mutation patch nodes for GearBind
    patch_node_2: np.ndarray
    patch_ca_1: np.ndarray              # [N, K, 3] float32 C-alpha coords of 3D patch
    patch_ca_2: np.ndarray
    patch_cb_1: np.ndarray              # [N, K, 3] float32 C-beta coords of 3D patch
    patch_cb_2: np.ndarray
    patch_chain_1: np.ndarray           # [N, K] int8 (0=H, 1=L, 2=A, -1=pad)
    patch_chain_2: np.ndarray
    patch_seqpos_1: np.ndarray          # [N, K] int16 residue index along chain
    patch_seqpos_2: np.ndarray

    def subset(self, idx: np.ndarray | list[int]) -> "StructureSplitFeatures":
        ix = np.asarray(idx)
        return StructureSplitFeatures(
            is_cocrystal=self.is_cocrystal[ix],
            zs_dir={k: v[ix] for k, v in self.zs_dir.items()},
            zs_imp={k: v[ix] for k, v in self.zs_imp.items()},
            struct_odd=self.struct_odd[ix],
            struct_even=self.struct_even[ix],
            if1_emb_1=self.if1_emb_1[ix],
            if1_emb_2=self.if1_emb_2[ix],
            mpnn_emb_1=self.mpnn_emb_1[ix],
            mpnn_emb_2=self.mpnn_emb_2[ix],
            saprot_ab_1=self.saprot_ab_1[ix],
            saprot_ab_2=self.saprot_ab_2[ix],
            saprot_ag_1=self.saprot_ag_1[ix],
            saprot_ag_2=self.saprot_ag_2[ix],
            mint_ab_1=self.mint_ab_1[ix],
            mint_ab_2=self.mint_ab_2[ix],
            mint_ag_1=self.mint_ag_1[ix],
            mint_ag_2=self.mint_ag_2[ix],
            dccm_1=self.dccm_1[ix] if len(self.dccm_1) else self.dccm_1,
            dccm_2=self.dccm_2[ix] if len(self.dccm_2) else self.dccm_2,
            patch_node_1=self.patch_node_1[ix] if len(self.patch_node_1) else self.patch_node_1,
            patch_node_2=self.patch_node_2[ix] if len(self.patch_node_2) else self.patch_node_2,
            patch_ca_1=self.patch_ca_1[ix] if len(self.patch_ca_1) else self.patch_ca_1,
            patch_ca_2=self.patch_ca_2[ix] if len(self.patch_ca_2) else self.patch_ca_2,
            patch_cb_1=self.patch_cb_1[ix] if len(self.patch_cb_1) else self.patch_cb_1,
            patch_cb_2=self.patch_cb_2[ix] if len(self.patch_cb_2) else self.patch_cb_2,
            patch_chain_1=self.patch_chain_1[ix] if len(self.patch_chain_1) else self.patch_chain_1,
            patch_chain_2=self.patch_chain_2[ix] if len(self.patch_chain_2) else self.patch_chain_2,
            patch_seqpos_1=self.patch_seqpos_1[ix] if len(self.patch_seqpos_1) else self.patch_seqpos_1,
            patch_seqpos_2=self.patch_seqpos_2[ix] if len(self.patch_seqpos_2) else self.patch_seqpos_2,
        )


def _get_align_map(
    q_seq: str,
    s_seq: str,
    cache: dict[tuple[str, str], np.ndarray],
) -> np.ndarray:
    if len(q_seq) == len(s_seq):
        return np.arange(len(q_seq), dtype=np.int32)
    key = (q_seq, s_seq)
    if key not in cache:
        _, _, q2s = align_query_to_chain(q_seq, s_seq)
        cache[key] = q2s
    return cache[key]


def _score_chain_mutations_3d(
    s1: str,
    s2: str,
    scaf_seq: str,
    lp_cplx: np.ndarray,
    lp_unb: np.ndarray,
    min_dist: np.ndarray,
    dsasa: np.ndarray,
    is_cocrys: bool,
    align_cache: dict[tuple[str, str], np.ndarray],
) -> tuple[float, float, float, float, list[tuple[int, str, str]]]:
    """Return `(dd_lp_bind, abs_dd_lp_weighted, d_lp_cplx, d_lp_unb, mut_sites_in_scaf)` for one chain."""
    if s1 == s2 or not s1 or not s2:
        return 0.0, 0.0, 0.0, 0.0, []

    if len(s1) == len(s2):
        diffs = [i for i in range(len(s1)) if s1[i] != s2[i]]
        if len(diffs) <= 25:
            q2s = _get_align_map(s1, scaf_seq, align_cache)
            dd_bind = 0.0
            abs_w = 0.0
            dc_tot = 0.0
            du_tot = 0.0
            mut_scaf: list[tuple[int, str, str]] = []
            L_lp = min(len(lp_cplx), len(lp_unb), len(min_dist))
            for pos in diffs:
                si = int(q2s[pos]) if pos < len(q2s) else -1
                if si < 0 or si >= L_lp:
                    continue
                a1 = AA_TO_IDX.get(s1[pos], -1)
                a2 = AA_TO_IDX.get(s2[pos], -1)
                if a1 < 0 or a2 < 0:
                    continue
                dc = float(lp_cplx[si, a1]) - float(lp_cplx[si, a2])
                du = float(lp_unb[si, a1]) - float(lp_unb[si, a2])
                if is_cocrys:
                    # Thermodynamic binding cycle ΔΔlog p_bind + complex stability prior
                    dd = 0.65 * (dc - du) + 0.35 * dc
                    d_ang = float(min_dist[si])
                    w_int = 1.0 + 1.5 * float(np.exp(-max(d_ang - 3.5, 0.0) / 5.0)) + 0.02 * float(dsasa[si])
                else:
                    dd = du
                    w_int = 1.0
                dd_bind += dd
                abs_w += abs(dd) * w_int
                dc_tot += dc
                du_tot += du
                mut_scaf.append((si, s1[pos], s2[pos]))
            return dd_bind, abs_w, dc_tot, du_tot, mut_scaf

    # Candidate / Global fallback (`len(s1) != len(s2)` or `>25` mutations): mean sequence log-prob on scaffold
    def _mean_lp(seq_q: str, lp_arr: np.ndarray) -> float:
        ckey = ("__mlp__", seq_q, id(lp_arr))
        cached_v = align_cache.get(ckey)  # type: ignore[arg-type]
        if cached_v is not None:
            return float(cached_v)  # type: ignore[arg-type]
        L_lp = len(lp_arr)
        ikey = ("__midx__", seq_q, scaf_seq, L_lp)
        idx_pair = align_cache.get(ikey)  # type: ignore[arg-type]
        if idx_pair is None:
            q2s = _get_align_map(seq_q, scaf_seq, align_cache)
            si_list = []
            ai_list = []
            for pos, ch in enumerate(seq_q):
                if pos < len(q2s):
                    si = int(q2s[pos])
                    ai = AA_TO_IDX.get(ch, -1)
                    if 0 <= si < L_lp and ai >= 0:
                        si_list.append(si)
                        ai_list.append(ai)
            idx_pair = (np.asarray(si_list, dtype=np.int32), np.asarray(ai_list, dtype=np.int32))
            align_cache[ikey] = idx_pair  # type: ignore[index]
        si_v, ai_v = idx_pair  # type: ignore[misc]
        res_v = float(lp_arr[si_v, ai_v].mean()) if len(si_v) > 0 else 0.0
        align_cache[ckey] = res_v  # type: ignore[index]
        return res_v

    dc = _mean_lp(s1, lp_cplx) - _mean_lp(s2, lp_cplx)
    du = _mean_lp(s1, lp_unb) - _mean_lp(s2, lp_unb)
    dd = (0.65 * (dc - du) + 0.35 * dc) if is_cocrys else du
    return dd, abs(dd), dc, du, []


def _pool_3d_complex_emb(
    emb_dict: dict[str, np.ndarray],
    bio: dict,
    dim: int,
) -> np.ndarray:
    """Pool `[H_interface, H_mean, L_mean, A_interface]` (`4 * dim`) from per-chain 3D embeddings."""
    eh = emb_dict.get("H", np.zeros((1, dim), dtype=np.float16)).astype(np.float32)
    el = emb_dict.get("L", np.zeros((0, dim), dtype=np.float16)).astype(np.float32)
    ea = emb_dict.get("A", np.zeros((1, dim), dtype=np.float16)).astype(np.float32)

    dh = bio["min_dist_H"].astype(np.float32)[: len(eh)]
    da = bio["min_dist_A"].astype(np.float32)[: len(ea)]
    wh = np.exp(-np.maximum(dh - 4.0, 0.0) / 5.0)
    wa = np.exp(-np.maximum(da - 4.0, 0.0) / 5.0)
    h_int = (eh * wh[:, None]).sum(axis=0) / (wh.sum() + 1e-6)
    a_int = (ea * wa[:, None]).sum(axis=0) / (wa.sum() + 1e-6)
    h_mean = eh.mean(axis=0)
    l_mean = el.mean(axis=0) if len(el) > 0 else np.zeros(dim, dtype=np.float32)
    return np.concatenate([h_int, h_mean, l_mean, a_int], axis=0)


def _build_3d_patch(
    vh: str,
    vl: str,
    ag: str,
    bio: dict,
    inv: dict,
    mut_sites_by_chain: dict[str, set[int]],
    align_cache: dict[tuple[str, str], np.ndarray],
    k_nodes: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct fixed-size `K=40` 3D Interface + Mutation Microenvironment Patch for GearBind:
    Node feature dim = `20 (one-hot AA) + 64 (compressed MPNN 3D state) + 6 (3D geometry/SASA/LP) = 90`.
    """
    bkey = ("__resbank__", vh, vl, ag, id(bio), id(inv), k_nodes)
    bank = align_cache.get(bkey)  # type: ignore[arg-type]
    if bank is None:
        base_cands: list[tuple[float, int, int, str]] = []
        s2aa_map: dict[int, dict[int, str]] = {0: {}, 1: {}, 2: {}}
        chain_specs = [
            ("H", 0, vh, bio["seq_H"], bio["min_dist_H"]),
            ("L", 1, vl, bio["seq_L"], bio["min_dist_L"]),
            ("A", 2, ag, bio["seq_A"], bio["min_dist_A"]),
        ]
        for _, ccode, qseq, sseq, mdist in chain_specs:
            if not qseq or not sseq or len(mdist) == 0:
                continue
            q2s = _get_align_map(qseq, sseq, align_cache)
            s2aa: dict[int, str] = {}
            for qi, ch in enumerate(qseq):
                if qi < len(q2s) and q2s[qi] >= 0:
                    s2aa[int(q2s[qi])] = ch
            s2aa_map[ccode] = s2aa
            L_s = len(sseq)
            for si in range(len(mdist)):
                aa = s2aa.get(si, sseq[si] if si < L_s else "A")
                base_cands.append((float(mdist[si]), ccode, si, aa))
        base_cands.sort(key=lambda x: x[0])
        top_base = base_cands[:k_nodes]
        res_feat_cache: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
        bank = (top_base, s2aa_map, res_feat_cache)
        align_cache[bkey] = bank  # type: ignore[index]
    else:
        top_base, s2aa_map, res_feat_cache = bank  # type: ignore[misc]

    m_h = mut_sites_by_chain.get("H", set())
    m_l = mut_sites_by_chain.get("L", set())
    m_a = mut_sites_by_chain.get("A", set())
    if not m_h and not m_l and not m_a:
        chosen = list(top_base)
    else:
        m_by_c = {0: m_h, 1: m_l, 2: m_a}
        seen_cs: set[tuple[int, int]] = set()
        merged: list[tuple[float, int, int, str]] = []
        for prio, ccode, si, aa in top_base:
            seen_cs.add((ccode, si))
            if si in m_by_c[ccode]:
                merged.append((-100.0 + prio * 0.01, ccode, si, aa))
            else:
                merged.append((prio, ccode, si, aa))
        for cname, ccode in (("H", 0), ("L", 1), ("A", 2)):
            mset = m_by_c[ccode]
            if not mset:
                continue
            mdist = bio[f"min_dist_{cname}"]
            sseq = bio[f"seq_{cname}"]
            if not sseq or len(mdist) == 0:
                continue
            s2aa = s2aa_map[ccode]
            L_s = len(sseq)
            for si in sorted(mset):
                if 0 <= si < len(mdist) and (ccode, si) not in seen_cs:
                    aa = s2aa.get(si, sseq[si] if si < L_s else "A")
                    merged.append((-100.0 + float(mdist[si]) * 0.01, ccode, si, aa))
        merged.sort(key=lambda x: x[0])
        chosen = merged[:k_nodes]

    # Sort deterministically by `(chain_code, scaf_idx)` so node ordering is canonical!
    chosen.sort(key=lambda x: (x[1], x[2]))

    d_node = 90
    nodes = np.zeros((k_nodes, d_node), dtype=np.float16)
    ca = np.zeros((k_nodes, 3), dtype=np.float32)
    cb = np.zeros((k_nodes, 3), dtype=np.float32)
    cids = np.full(k_nodes, -1, dtype=np.int8)
    spos = np.zeros(k_nodes, dtype=np.int16)

    cname_by_code = {0: "H", 1: "L", 2: "A"}
    for idx, (_, ccode, si, aa) in enumerate(chosen):
        cn = cname_by_code[ccode]
        rkey = (ccode, si, aa)
        if rkey not in res_feat_cache:
            ca_arr = bio[f"ca_{cn}"]
            if si >= len(ca_arr):
                res_feat_cache[rkey] = None
            else:
                cb_arr = bio[f"cb_{cn}"]
                md_arr = bio[f"min_dist_{cn}"]
                ds_arr = bio[f"dsasa_{cn}"]
                mp_emb = inv["mpnn_emb_cplx"].get(cn)
                if_lp = inv["if1_lp_cplx"].get(cn)
                mp_lp = inv["mpnn_lp_cplx"].get(cn)

                vec = np.zeros(d_node, dtype=np.float32)
                ai = AA_TO_IDX.get(aa, 0)
                vec[ai] = 1.0
                if mp_emb is not None and si < len(mp_emb):
                    me = mp_emb[si].astype(np.float32)
                    vec[20:84] = 0.5 * (me[:64] + me[64:128])
                d_val = float(md_arr[si]) if si < len(md_arr) else 20.0
                sasa_val = float(ds_arr[si]) if si < len(ds_arr) else 0.0
                lp_if_v = float(if_lp[si, ai]) if (if_lp is not None and si < len(if_lp)) else -3.0
                lp_mp_v = float(mp_lp[si, ai]) if (mp_lp is not None and si < len(mp_lp)) else -3.0
                vec[84] = np.exp(-max(d_val - 3.5, 0.0) / 5.0)
                vec[85] = sasa_val / 100.0
                vec[86] = 1.0 if bio["is_cocrystal"] else 0.0
                vec[87] = lp_if_v / 5.0
                vec[88] = lp_mp_v / 5.0
                res_feat_cache[rkey] = (
                    vec.astype(np.float16),
                    ca_arr[si].astype(np.float32),
                    cb_arr[si].astype(np.float32),
                )
        rf = res_feat_cache[rkey]
        if rf is None:
            continue
        vec_f16, ca_v, cb_v = rf
        ca[idx] = ca_v
        cb[idx] = cb_v
        cids[idx] = ccode
        spos[idx] = si
        nodes[idx] = vec_f16
        if si in mut_sites_by_chain.get(cn, set()):
            nodes[idx, 89] = 1.0

    return nodes, ca, cb, cids, spos


def assemble_structure_pair_features(
    pairs_df: pd.DataFrame,
    assays_lookup: pd.DataFrame,
    complex_cache: dict,
    inv_cache: dict,
    saprot_cache: dict | None = None,
    mint_cache: dict | None = None,
    build_patches_and_dccm: bool = True,
) -> StructureSplitFeatures:
    """Assemble all 3D structure-conditioned scores, multimodal embeddings, DCCM maps, and 3D patches."""
    n = len(pairs_df)
    abag_to_scaf: dict[str, str] = complex_cache["abag_to_scaffold"]
    scaffolds: dict[str, dict] = complex_cache["scaffolds"]

    id1_arr = pairs_df["AbAgID_1"].to_numpy()
    id2_arr = pairs_df["AbAgID_2"].to_numpy()

    is_cocrys = np.zeros(n, dtype=bool)
    zs_keys = ("esm_if1", "protein_mpnn", "antifold", "prodigy_sasa", "saprot_zs", "mint_zs")
    zs_dir = {k: np.zeros(n, dtype=np.float32) for k in zs_keys}
    zs_imp = {k: np.zeros(n, dtype=np.float32) for k in zs_keys}

    struct_odd = np.zeros((n, 22), dtype=np.float32)
    struct_even = np.zeros((n, 24), dtype=np.float32)

    if1_emb_1 = np.zeros((n, 2048), dtype=np.float32)
    if1_emb_2 = np.zeros((n, 2048), dtype=np.float32)
    mpnn_emb_1 = np.zeros((n, 512), dtype=np.float32)
    mpnn_emb_2 = np.zeros((n, 512), dtype=np.float32)

    saprot_ab_1 = np.zeros((n, 3840), dtype=np.float32)
    saprot_ab_2 = np.zeros((n, 3840), dtype=np.float32)
    saprot_ag_1 = np.zeros((n, 2560), dtype=np.float32)
    saprot_ag_2 = np.zeros((n, 2560), dtype=np.float32)

    mint_ab_1 = np.zeros((n, 3840), dtype=np.float32)
    mint_ab_2 = np.zeros((n, 3840), dtype=np.float32)
    mint_ag_1 = np.zeros((n, 2560), dtype=np.float32)
    mint_ag_2 = np.zeros((n, 2560), dtype=np.float32)

    k_nodes = 40
    dccm_1 = np.zeros((n, 2, 64, 64), dtype=np.float16) if build_patches_and_dccm else np.zeros((0, 2, 64, 64), dtype=np.float16)
    dccm_2 = np.zeros((n, 2, 64, 64), dtype=np.float16) if build_patches_and_dccm else np.zeros((0, 2, 64, 64), dtype=np.float16)
    patch_node_1 = np.zeros((n, k_nodes, 90), dtype=np.float16) if build_patches_and_dccm else np.zeros((0, k_nodes, 90), dtype=np.float16)
    patch_node_2 = np.zeros((n, k_nodes, 90), dtype=np.float16) if build_patches_and_dccm else np.zeros((0, k_nodes, 90), dtype=np.float16)
    patch_ca_1 = np.zeros((n, k_nodes, 3), dtype=np.float32) if build_patches_and_dccm else np.zeros((0, k_nodes, 3), dtype=np.float32)
    patch_ca_2 = np.zeros((n, k_nodes, 3), dtype=np.float32) if build_patches_and_dccm else np.zeros((0, k_nodes, 3), dtype=np.float32)
    patch_cb_1 = np.zeros((n, k_nodes, 3), dtype=np.float32) if build_patches_and_dccm else np.zeros((0, k_nodes, 3), dtype=np.float32)
    patch_cb_2 = np.zeros((n, k_nodes, 3), dtype=np.float32) if build_patches_and_dccm else np.zeros((0, k_nodes, 3), dtype=np.float32)
    patch_chain_1 = np.zeros((n, k_nodes), dtype=np.int8) if build_patches_and_dccm else np.zeros((0, k_nodes), dtype=np.int8)
    patch_chain_2 = np.zeros((n, k_nodes), dtype=np.int8) if build_patches_and_dccm else np.zeros((0, k_nodes), dtype=np.int8)
    patch_seqpos_1 = np.zeros((n, k_nodes), dtype=np.int16) if build_patches_and_dccm else np.zeros((0, k_nodes), dtype=np.int16)
    patch_seqpos_2 = np.zeros((n, k_nodes), dtype=np.int16) if build_patches_and_dccm else np.zeros((0, k_nodes), dtype=np.int16)

    # Pre-pool scaffold 3D embeddings once for used scaffolds
    scaf_if1_pool: dict[str, np.ndarray] = {}
    scaf_mpnn_pool: dict[str, np.ndarray] = {}
    used_sids = {abag_to_scaf[a] for a in set(id1_arr).union(id2_arr) if a in abag_to_scaf}
    for sid in used_sids:
        if sid in scaffolds and sid in inv_cache:
            bio = scaffolds[sid]
            scaf_if1_pool[sid] = _pool_3d_complex_emb(inv_cache[sid]["if1_emb_cplx"], bio, 512)
            scaf_mpnn_pool[sid] = _pool_3d_complex_emb(inv_cache[sid]["mpnn_emb_cplx"], bio, 128)

    align_cache: dict[tuple[str, str], np.ndarray] = {}
    sap_chains = saprot_cache["chains"] if saprot_cache is not None else {}
    sap_rep_map = saprot_cache["scaf_rep_abag"] if saprot_cache is not None else {}
    mint_cplx = mint_cache["complexes"] if mint_cache is not None else {}
    mint_rep_map = mint_cache["scaf_rep_abag"] if mint_cache is not None else {}

    def _get_saprot_chain(aa_s: str, s3_s: str, fb_aa: str) -> dict | None:
        if not aa_s or not sap_chains:
            return None
        ckey = hashlib.sha1(f"{aa_s}|{s3_s[:len(aa_s)]}".encode()).hexdigest()[:16]
        if ckey in sap_chains:
            return sap_chains[ckey]
        fb_key = hashlib.sha1(f"{fb_aa}|{s3_s[:len(fb_aa)]}".encode()).hexdigest()[:16]
        return sap_chains.get(fb_key)

    def _get_mint_entry(abag_id: str, sid: str) -> tuple[dict | None, bool]:
        if not mint_cplx:
            return None, False
        if abag_id in mint_cplx:
            return mint_cplx[abag_id], True
        rep_id = mint_rep_map.get(sid, "")
        return mint_cplx.get(rep_id), False

    # Per-AbAgID caches so tournament panels & repeated complexes compute unary 3D features once!
    unary_3d_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    dccm_cache: dict[str, np.ndarray] = {}
    patch_cache: dict[tuple[str, tuple], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def _get_unary_3d(aid: str, vh_q: str, vl_q: str, ag_q: str, sid_q: str, bio_q: dict, inv_q: dict) -> tuple[np.ndarray, np.ndarray]:
        if aid in unary_3d_cache:
            return unary_3d_cache[aid]
        p_if = scaf_if1_pool[sid_q].copy()
        p_mp = scaf_mpnn_pool[sid_q].copy()
        for cn_q, s_q in (("H", vh_q), ("L", vl_q), ("A", ag_q)):
            sc_s = bio_q[f"seq_{cn_q}"]
            if s_q and sc_s and s_q != sc_s and cn_q in inv_q["if1_emb_cplx"]:
                q2s_q = _get_align_map(s_q, sc_s, align_cache)
                e_if_q = inv_q["if1_emb_cplx"][cn_q].astype(np.float32)
                e_mp_q = inv_q["mpnn_emb_cplx"][cn_q].astype(np.float32)
                lp_if_q = inv_q["if1_lp_cplx"][cn_q].astype(np.float32)
                slot_q = 0 if cn_q == "H" else (2 if cn_q == "L" else 3)
                for pos_q, ch_q in enumerate(s_q):
                    si_q = int(q2s_q[pos_q]) if pos_q < len(q2s_q) else -1
                    if 0 <= si_q < min(len(sc_s), len(e_if_q)) and ch_q != sc_s[si_q]:
                        aq_i = AA_TO_IDX.get(ch_q, -1)
                        aw_i = AA_TO_IDX.get(sc_s[si_q], -1)
                        if aq_i >= 0 and aw_i >= 0:
                            dq_w = (float(lp_if_q[si_q, aq_i]) - float(lp_if_q[si_q, aw_i])) * 0.1 + (
                                HYDROPHOBICITY.get(ch_q, 0.0) - HYDROPHOBICITY.get(sc_s[si_q], 0.0)
                            ) * 0.05
                            p_if[slot_q * 512 : (slot_q + 1) * 512] += dq_w * e_if_q[si_q]
                            if si_q < len(e_mp_q):
                                p_mp[slot_q * 128 : (slot_q + 1) * 128] += dq_w * e_mp_q[si_q]
        unary_3d_cache[aid] = (p_if, p_mp)
        return p_if, p_mp

    def _get_dccm2(aid: str, vh_q: str, vl_q: str, ag_q: str, b_ref: dict) -> np.ndarray:
        if aid in dccm_cache:
            return dccm_cache[aid]
        m0 = b_ref["dccm_64"].astype(np.float32)
        prof = np.zeros(64, dtype=np.float32)
        cdr_h = _cdr_residue_indices(len(vh_q), True)
        if len(vh_q) > 0:
            sel_h = cdr_h[np.linspace(0, len(cdr_h) - 1, 26).round().astype(int)] if len(cdr_h) >= 26 else np.linspace(0, len(vh_q) - 1, 26).round().astype(int)
            prof[:26] = [0.4 * VOLUME_NORM.get(vh_q[int(k)], 0.5) + 0.1 * HYDROPHOBICITY.get(vh_q[int(k)], 0.0) for k in sel_h]
        if len(vl_q) > 0:
            cdr_l = _cdr_residue_indices(len(vl_q), False)
            sel_l = cdr_l[np.linspace(0, len(cdr_l) - 1, 18).round().astype(int)] if len(cdr_l) >= 18 else np.linspace(0, len(vl_q) - 1, 18).round().astype(int)
            prof[26:44] = [0.4 * VOLUME_NORM.get(vl_q[int(k)], 0.5) + 0.1 * HYDROPHOBICITY.get(vl_q[int(k)], 0.0) for k in sel_l]
        if len(ag_q) > 0 and len(b_ref["min_dist_A"]) > 0:
            md_a = b_ref["min_dist_A"]
            ag_k = np.sort(np.argsort(md_a)[: min(20, len(md_a))])
            prof[44 : 44 + len(ag_k)] = [
                0.4 * VOLUME_NORM.get(ag_q[min(int(k), len(ag_q) - 1)], 0.5) + 0.1 * HYDROPHOBICITY.get(ag_q[min(int(k), len(ag_q) - 1)], 0.0)
                for k in ag_k
            ]
        m1_mod = m0 * (0.5 * (prof[:, None] + prof[None, :]))
        out_m = np.stack([m0, m1_mod], axis=0).astype(np.float16)
        dccm_cache[aid] = out_m
        return out_m

    # Fast column extraction from assays_lookup
    vh_s = assays_lookup["Ab_heavy_chain_seq"].fillna("").astype(str)
    vl_s = assays_lookup["Ab_light_chain_seq"].fillna("").astype(str)
    ag_s = assays_lookup["Ag_seq"].fillna("").astype(str)

    for i in range(n):
        a1 = id1_arr[i]
        a2 = id2_arr[i]
        sid1 = abag_to_scaf[a1]
        sid2 = abag_to_scaf[a2]
        bio1 = scaffolds[sid1]
        bio2 = scaffolds[sid2]
        inv1 = inv_cache[sid1]
        inv2 = inv_cache[sid2]

        cc_both = bool(bio1["is_cocrystal"] and bio2["is_cocrystal"])
        is_cocrys[i] = cc_both

        vh1, vl1, ag1 = vh_s.loc[a1], vl_s.loc[a1], ag_s.loc[a1]
        vh2, vl2, ag2 = vh_s.loc[a2], vl_s.loc[a2], ag_s.loc[a2]

        # Symmetrized evaluation across scaffold 1 and scaffold 2 so f(1,2) == -f(2,1) ALWAYS holds!
        if1_dd, if1_imp, if1_dc, if1_du = 0.0, 0.0, 0.0, 0.0
        mp_dd, mp_imp, mp_dc, mp_du = 0.0, 0.0, 0.0, 0.0
        af_dd, af_imp, af_dc, af_du = 0.0, 0.0, 0.0, 0.0

        mut_sites_1: dict[str, set[int]] = {"H": set(), "L": set(), "A": set()}
        mut_sites_2: dict[str, set[int]] = {"H": set(), "L": set(), "A": set()}
        mut_dists: list[float] = []
        mut_dsasas: list[float] = []
        ddg_stat_local = 0.0
        ddg_stat_abs = 0.0

        for cname, s_one, s_two in (("H", vh1, vh2), ("L", vl1, vl2), ("A", ag1, ag2)):
            if not s_one and not s_two:
                continue
            # Evaluate on scaffold 1 and scaffold 2 and average (exact antisymmetry even when sid1 != sid2!)
            for w_sc, b_ref, iv_ref, mdict in ((0.5, bio1, inv1, mut_sites_1), (0.5, bio2, inv2, mut_sites_2)):
                sc_seq = b_ref[f"seq_{cname}"]
                md_arr = b_ref[f"min_dist_{cname}"]
                ds_arr = b_ref[f"dsasa_{cname}"]
                if not sc_seq or cname not in iv_ref["if1_lp_cplx"]:
                    continue

                d1, im1, dc1, du1, ms1 = _score_chain_mutations_3d(
                    s_one, s_two, sc_seq,
                    iv_ref["if1_lp_cplx"][cname], iv_ref["if1_lp_unb"][cname],
                    md_arr, ds_arr, cc_both, align_cache,
                )
                if1_dd += w_sc * d1; if1_imp += w_sc * im1; if1_dc += w_sc * dc1; if1_du += w_sc * du1

                d2, im2, dc2, du2, _ = _score_chain_mutations_3d(
                    s_one, s_two, sc_seq,
                    iv_ref["mpnn_lp_cplx"][cname], iv_ref["mpnn_lp_unb"][cname],
                    md_arr, ds_arr, cc_both, align_cache,
                )
                mp_dd += w_sc * d2; mp_imp += w_sc * im2; mp_dc += w_sc * dc2; mp_du += w_sc * du2

                d3, im3, dc3, du3, _ = _score_chain_mutations_3d(
                    s_one, s_two, sc_seq,
                    iv_ref["af_lp_cplx"][cname], iv_ref["af_lp_unb"][cname],
                    md_arr, ds_arr, cc_both, align_cache,
                )
                af_dd += w_sc * d3; af_imp += w_sc * im3; af_dc += w_sc * dc3; af_du += w_sc * du3

                for si, aa1, aa2 in ms1:
                    mdict[cname].add(si)
                    d_ang = float(md_arr[si]) if si < len(md_arr) else 20.0
                    sasa_b = float(ds_arr[si]) if si < len(ds_arr) else 0.0
                    mut_dists.append(d_ang)
                    mut_dsasas.append(sasa_b)

                    # Statistical 3D interfacial ΔΔG contribution (`PRODIGY+SASA`)
                    f_int = float(np.exp(-max(d_ang - 3.5, 0.0) / 4.0)) + 0.015 * sasa_b
                    d_hyd = HYDROPHOBICITY.get(aa1, 0.0) - HYDROPHOBICITY.get(aa2, 0.0)
                    d_arom = (1.0 if aa1 in "YWF" else 0.0) - (1.0 if aa2 in "YWF" else 0.0)
                    d_abs_chg = abs(CHARGE_PH74.get(aa1, 0.0)) - abs(CHARGE_PH74.get(aa2, 0.0))
                    d_vol = VOLUME_NORM.get(aa1, 0.5) - VOLUME_NORM.get(aa2, 0.5)
                    # Interfacial Tyr/Trp/hydrophobic burial favors binding; burying uncompensated charge or Gly/Pro loss penalizes
                    contrib = f_int * (0.18 * d_hyd + 0.65 * d_arom - 0.25 * (sasa_b / 50.0) * d_abs_chg + 0.20 * d_vol)
                    ddg_stat_local += w_sc * contrib
                    ddg_stat_abs += w_sc * abs(contrib)

        prod_global = float(bio1["prodigy_neg_dg"] - bio2["prodigy_neg_dg"])
        prod_score = ddg_stat_local + 0.25 * prod_global
        prod_imp = ddg_stat_abs + 0.25 * abs(prod_global) + (
            float(np.sum([np.exp(-max(d - 4.0, 0.0) / 4.0) for d in mut_dists])) * 0.5 if mut_dists else 0.0
        )

        zs_dir["esm_if1"][i] = if1_dd
        zs_imp["esm_if1"][i] = if1_imp
        zs_dir["protein_mpnn"][i] = mp_dd
        zs_imp["protein_mpnn"][i] = mp_imp
        zs_dir["antifold"][i] = af_dd
        zs_imp["antifold"][i] = af_imp
        zs_dir["prodigy_sasa"][i] = prod_score
        zs_imp["prodigy_sasa"][i] = prod_imp

        p_if1_1, p_mp_1 = _get_unary_3d(a1, vh1, vl1, ag1, sid1, bio1, inv1)
        p_if1_2, p_mp_2 = _get_unary_3d(a2, vh2, vl2, ag2, sid2, bio2, inv2)

        if1_emb_1[i] = p_if1_1
        if1_emb_2[i] = p_if1_2
        mpnn_emb_1[i] = p_mp_1
        mpnn_emb_2[i] = p_mp_2

        # SaProt-650M features & zero-shot score
        if sap_chains:
            rep1_id = sap_rep_map.get(sid1, a1)
            rep2_id = sap_rep_map.get(sid2, a2)
            r1_fb = assays_lookup.loc[rep1_id] if rep1_id in assays_lookup.index else assays_lookup.loc[a1]
            r2_fb = assays_lookup.loc[rep2_id] if rep2_id in assays_lookup.index else assays_lookup.loc[a2]

            sh1 = _get_saprot_chain(vh1, bio1["3di_H"], str(r1_fb["Ab_heavy_chain_seq"]))
            sh2 = _get_saprot_chain(vh2, bio2["3di_H"], str(r2_fb["Ab_heavy_chain_seq"]))
            sl1 = _get_saprot_chain(vl1, bio1["3di_L"], str(r1_fb["Ab_light_chain_seq"])) if vl1 else None
            sl2 = _get_saprot_chain(vl2, bio2["3di_L"], str(r2_fb["Ab_light_chain_seq"])) if vl2 else None
            sa1 = _get_saprot_chain(ag1, bio1["3di_A"], str(r1_fb["Ag_seq"]))
            sa2 = _get_saprot_chain(ag2, bio2["3di_A"], str(r2_fb["Ag_seq"]))

            z1280 = np.zeros(1280, dtype=np.float32)
            saprot_ab_1[i] = np.concatenate([
                sh1["mean"].astype(np.float32) if sh1 else z1280,
                sl1["mean"].astype(np.float32) if sl1 else z1280,
                sh1["cdr3"].astype(np.float32) if sh1 else z1280,
            ])
            saprot_ab_2[i] = np.concatenate([
                sh2["mean"].astype(np.float32) if sh2 else z1280,
                sl2["mean"].astype(np.float32) if sl2 else z1280,
                sh2["cdr3"].astype(np.float32) if sh2 else z1280,
            ])
            saprot_ag_1[i] = np.concatenate([
                sa1["mean"].astype(np.float32) if sa1 else z1280,
                sa1["std"].astype(np.float32) if sa1 else z1280,
            ])
            saprot_ag_2[i] = np.concatenate([
                sa2["mean"].astype(np.float32) if sa2 else z1280,
                sa2["std"].astype(np.float32) if sa2 else z1280,
            ])

            sap_llr, sap_abs = 0.0, 0.0
            any_loc = False
            for s_one, s_two, d_one, d_two in ((vh1, vh2, sh1, sh2), (vl1, vl2, sl1, sl2), (ag1, ag2, sa1, sa2)):
                if s_one and s_two and d_one and d_two and len(s_one) == len(s_two) and s_one != s_two:
                    lp_a = d_one["lp"].astype(np.float32)
                    lp_b = d_two["lp"].astype(np.float32)
                    L_m = min(len(s_one), len(lp_a), len(lp_b))
                    diffs = [p for p in range(L_m) if s_one[p] != s_two[p]]
                    if 0 < len(diffs) <= 20:
                        any_loc = True
                        for p in diffs:
                            c1, c2 = AA_TO_IDX.get(s_one[p], -1), AA_TO_IDX.get(s_two[p], -1)
                            if c1 >= 0 and c2 >= 0:
                                v = 0.5 * ((lp_a[p, c1] - lp_a[p, c2]) + (lp_b[p, c1] - lp_b[p, c2]))
                                sap_llr += float(v)
                                sap_abs += abs(float(v))
                                if d_one is d_two:
                                    m_vec = d_one["mean"].astype(np.float32)
                                    if s_one is vh1:
                                        saprot_ab_1[i, :1280] += 0.01 * float(v) * m_vec
                                        saprot_ab_2[i, :1280] -= 0.01 * float(v) * m_vec
                                    elif s_one is vl1:
                                        saprot_ab_1[i, 1280:2560] += 0.01 * float(v) * m_vec
                                        saprot_ab_2[i, 1280:2560] -= 0.01 * float(v) * m_vec
            if not any_loc:
                pll1 = (sh1["pll"] if sh1 else 0.0) + (sl1["pll"] if sl1 else 0.0) + (sa1["pll"] if sa1 else 0.0)
                pll2 = (sh2["pll"] if sh2 else 0.0) + (sl2["pll"] if sl2 else 0.0) + (sa2["pll"] if sa2 else 0.0)
                sap_llr = float(pll1 - pll2)
                sap_abs = abs(sap_llr)
            zs_dir["saprot_zs"][i] = sap_llr
            zs_imp["saprot_zs"][i] = sap_abs

        # MINT-650M multi-chain features & zero-shot score
        if mint_cplx:
            m1, exact1 = _get_mint_entry(a1, sid1)
            m2, exact2 = _get_mint_entry(a2, sid2)
            if m1 is not None and m2 is not None:
                mint_ab_1[i] = np.concatenate([m1["h_mean"], m1["l_mean"], m1["h_cdr3"]]).astype(np.float32)
                mint_ab_2[i] = np.concatenate([m2["h_mean"], m2["l_mean"], m2["h_cdr3"]]).astype(np.float32)
                mint_ag_1[i] = np.concatenate([m1["a_mean"], m1["a_std"]]).astype(np.float32)
                mint_ag_2[i] = np.concatenate([m2["a_mean"], m2["a_std"]]).astype(np.float32)

                mt_llr, mt_abs = 0.0, 0.0
                any_loc_m = False
                for cname, s_one, s_two in (("H", vh1, vh2), ("L", vl1, vl2), ("A", ag1, ag2)):
                    if s_one and s_two and len(s_one) == len(s_two) and s_one != s_two:
                        lp_a = m1[f"lp_{cname}"].astype(np.float32)
                        lp_b = m2[f"lp_{cname}"].astype(np.float32)
                        L_m = min(len(s_one), len(lp_a), len(lp_b))
                        diffs = [p for p in range(L_m) if s_one[p] != s_two[p]]
                        if 0 < len(diffs) <= 20:
                            any_loc_m = True
                            for p in diffs:
                                c1, c2 = AA_TO_IDX.get(s_one[p], -1), AA_TO_IDX.get(s_two[p], -1)
                                if c1 >= 0 and c2 >= 0:
                                    v = 0.5 * ((lp_a[p, c1] - lp_a[p, c2]) + (lp_b[p, c1] - lp_b[p, c2]))
                                    mt_llr += float(v)
                                    mt_abs += abs(float(v))
                                    if (not exact1 or not exact2) and f"res_{cname}" in m1 and f"res_{cname}" in m2:
                                        if p < min(len(m1[f"res_{cname}"]), len(m2[f"res_{cname}"])):
                                            r_vec = 0.5 * (
                                                m1[f"res_{cname}"][p].astype(np.float32)
                                                + m2[f"res_{cname}"][p].astype(np.float32)
                                            )
                                            if cname == "H":
                                                mint_ab_1[i, 2560:3840] += 0.025 * float(v) * r_vec
                                                mint_ab_2[i, 2560:3840] -= 0.025 * float(v) * r_vec
                                            elif cname == "L":
                                                mint_ab_1[i, 1280:2560] += 0.025 * float(v) * r_vec
                                                mint_ab_2[i, 1280:2560] -= 0.025 * float(v) * r_vec
                                            else:
                                                mint_ag_1[i, :1280] += 0.025 * float(v) * r_vec
                                                mint_ag_2[i, :1280] -= 0.025 * float(v) * r_vec
                if not any_loc_m:
                    mt_llr = float((m1["pll_ab"] + m1["pll_ag"]) - (m2["pll_ab"] + m2["pll_ag"]))
                    mt_abs = abs(mt_llr)
                zs_dir["mint_zs"][i] = mt_llr
                zs_imp["mint_zs"][i] = mt_abs

        # Assemble strictly antisymmetric `struct_odd` [22] and strictly symmetric `struct_even` [24]
        struct_odd[i] = np.array([
            if1_dd, if1_dc, if1_du, if1_dc - if1_du,
            mp_dd, mp_dc, mp_du, mp_dc - mp_du,
            af_dd, af_dc, af_du, af_dc - af_du,
            ddg_stat_local, prod_global,
            float(bio1["buried_sasa"] - bio2["buried_sasa"]) / 500.0,
            zs_dir["saprot_zs"][i],
            zs_dir["mint_zs"][i],
            if1_dd * (1.0 if cc_both else 0.0),
            mp_dd * (1.0 if cc_both else 0.0),
            af_dd * (1.0 if cc_both else 0.0),
            (if1_dd + mp_dd + af_dd) / 3.0,
            ddg_stat_local * (1.0 if cc_both else 0.0),
        ], dtype=np.float32)

        d_min_mut = float(np.min(mut_dists)) if mut_dists else 15.0
        d_mean_mut = float(np.mean(mut_dists)) if mut_dists else 15.0
        sasa_max_mut = float(np.max(mut_dsasas)) if mut_dsasas else 0.0
        sasa_sum_mut = float(np.sum(mut_dsasas)) if mut_dsasas else 0.0
        n_core_5a = float(sum(1 for d in mut_dists if d <= 5.5))
        n_rim_8a = float(sum(1 for d in mut_dists if 5.5 < d <= 8.5))
        n_distal = float(sum(1 for d in mut_dists if d > 10.0))

        struct_even[i] = np.array([
            1.0 if cc_both else 0.0,
            np.exp(-max(d_min_mut - 3.5, 0.0) / 4.0),
            np.exp(-max(d_mean_mut - 3.5, 0.0) / 5.0),
            min(d_min_mut, 30.0) / 30.0,
            sasa_max_mut / 100.0,
            sasa_sum_mut / 150.0,
            n_core_5a,
            n_rim_8a,
            n_distal,
            if1_imp,
            abs(if1_dc - if1_du),
            mp_imp,
            abs(mp_dc - mp_du),
            af_imp,
            abs(af_dc - af_du),
            prod_imp,
            zs_imp["saprot_zs"][i],
            zs_imp["mint_zs"][i],
            float(np.linalg.norm(p_if1_1 - p_if1_2)) / 10.0,
            float(np.linalg.norm(p_mp_1 - p_mp_2)) / 5.0,
            0.5 * float(bio1["buried_sasa"] + bio2["buried_sasa"]) / 1000.0,
            abs(float(bio1["prodigy_neg_dg"] - bio2["prodigy_neg_dg"])),
            (if1_imp + mp_imp + af_imp) / 3.0,
            n_core_5a * ((if1_imp + mp_imp) * 0.5),
        ], dtype=np.float32)

        if build_patches_and_dccm:
            dccm_1[i] = _get_dccm2(a1, vh1, vl1, ag1, bio1)
            dccm_2[i] = _get_dccm2(a2, vh2, vl2, ag2, bio2)

            # Union of mutated positions so patch1 and patch2 inspect identical structural sites on local pairs!
            union_mut = {c: mut_sites_1[c] | mut_sites_2[c] for c in ("H", "L", "A")}
            mkey = (tuple(sorted(union_mut["H"])), tuple(sorted(union_mut["L"])), tuple(sorted(union_mut["A"])))
            pk1 = (a1, mkey)
            if pk1 not in patch_cache:
                patch_cache[pk1] = _build_3d_patch(vh1, vl1, ag1, bio1, inv1, union_mut, align_cache, k_nodes)
            pk2 = (a2, mkey)
            if pk2 not in patch_cache:
                patch_cache[pk2] = _build_3d_patch(vh2, vl2, ag2, bio2, inv2, union_mut, align_cache, k_nodes)
            pn1, ca1, cb1, cc1, sp1 = patch_cache[pk1]
            pn2, ca2, cb2, cc2, sp2 = patch_cache[pk2]
            patch_node_1[i], patch_ca_1[i], patch_cb_1[i], patch_chain_1[i], patch_seqpos_1[i] = pn1, ca1, cb1, cc1, sp1
            patch_node_2[i], patch_ca_2[i], patch_cb_2[i], patch_chain_2[i], patch_seqpos_2[i] = pn2, ca2, cb2, cc2, sp2

    return StructureSplitFeatures(
        is_cocrystal=is_cocrys,
        zs_dir=zs_dir,
        zs_imp=zs_imp,
        struct_odd=struct_odd,
        struct_even=struct_even,
        if1_emb_1=if1_emb_1,
        if1_emb_2=if1_emb_2,
        mpnn_emb_1=mpnn_emb_1,
        mpnn_emb_2=mpnn_emb_2,
        saprot_ab_1=saprot_ab_1,
        saprot_ab_2=saprot_ab_2,
        saprot_ag_1=saprot_ag_1,
        saprot_ag_2=saprot_ag_2,
        mint_ab_1=mint_ab_1,
        mint_ab_2=mint_ab_2,
        mint_ag_1=mint_ag_1,
        mint_ag_2=mint_ag_2,
        dccm_1=dccm_1,
        dccm_2=dccm_2,
        patch_node_1=patch_node_1,
        patch_node_2=patch_node_2,
        patch_ca_1=patch_ca_1,
        patch_ca_2=patch_ca_2,
        patch_cb_1=patch_cb_1,
        patch_cb_2=patch_cb_2,
        patch_chain_1=patch_chain_1,
        patch_chain_2=patch_chain_2,
        patch_seqpos_1=patch_seqpos_1,
        patch_seqpos_2=patch_seqpos_2,
    )


def fit_3state_foundation_hurdle(
    tr_ab_1: np.ndarray,
    tr_ab_2: np.ndarray,
    tr_ag_1: np.ndarray,
    tr_ag_2: np.ndarray,
    tr_extra_odd: np.ndarray,
    tr_extra_even: np.ndarray,
    val_ab_1: np.ndarray,
    val_ab_2: np.ndarray,
    val_ag_1: np.ndarray,
    val_ag_2: np.ndarray,
    val_extra_odd: np.ndarray,
    val_extra_even: np.ndarray,
    tr_sp: SplitPairTensors,
    val_sp: SplitPairTensors,
    val_df: pd.DataFrame,
    w_tr: np.ndarray,
    seed: int = 42,
):
    """Fit a leak-free Two-Head Antisymmetric Ranker + 3D-augmented Symmetric Impact Hurdle."""
    def _loc(ab1, ab2, ag1, ag2, eo):
        return build_odd_linear_matrix([ab1, ag1], [ab2, ag2], extra_odd=eo, include_unit_direction=True)

    def _glb(ab1, ab2, ag1, ag2, eo):
        return build_odd_linear_matrix([ab1, ag1], [ab2, ag2], extra_odd=eo, include_unit_direction=False)

    X_loc_tr = _loc(tr_ab_1, tr_ab_2, tr_ag_1, tr_ag_2, tr_extra_odd)
    X_loc_val = _loc(val_ab_1, val_ab_2, val_ag_1, val_ag_2, val_extra_odd)
    X_glb_tr = _glb(tr_ab_1, tr_ab_2, tr_ag_1, tr_ag_2, tr_extra_odd)
    X_glb_val = _glb(val_ab_1, val_ab_2, val_ag_1, val_ag_2, val_extra_odd)

    m_loc = AntisymmetricLinearRanker(use_neutral_nullspace=True).fit(
        X_loc_tr, tr_sp.y, w_tr, X_loc_val, val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels
    )
    glb_tr_m = ~(tr_sp.is_same_ag | (~tr_sp.is_ab_mut))
    if glb_tr_m.sum() < 4:
        glb_tr_m = np.ones(len(tr_sp.y), dtype=bool)
    glb_val_m = (val_df["Category"] == "Global").to_numpy()
    if glb_val_m.sum() < 4:
        glb_val_m = np.ones(len(val_df), dtype=bool)
    m_glb = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(
        X_glb_tr[glb_tr_m], tr_sp.y[glb_tr_m], None,
        X_glb_val[glb_val_m], val_df.loc[glb_val_m].reset_index(drop=True), val_sp.is_ag_mut[glb_val_m],
    )
    two_head = TwoHeadRegimeRanker(local_head=m_loc, global_head=m_glb)
    dv_tr = two_head.predict(tr_sp, X_loc_tr, X_glb_tr)
    dv_val = two_head.predict(val_sp, X_loc_val, X_glb_val)

    # Augment `impact_even` with 3D structural symmetry features (`d_partner`, `ΔSASA`, `|ΔΔlog p|`)
    tr_sp_aug = SplitPairTensors(
        **{**tr_sp.__dict__, "impact_even": np.concatenate([tr_sp.impact_even, tr_extra_even], axis=1)}
    )
    val_sp_aug = SplitPairTensors(
        **{**val_sp.__dict__, "impact_even": np.concatenate([val_sp.impact_even, val_extra_even], axis=1)}
    )
    imp_clf = SymmetricImpactClassifier(seed=seed).fit(tr_sp_aug, val_sp_aug, dv_tr, dv_val, val_df=val_df)
    hurdle = OrdinalHurdleRanker(two_head, imp_clf).calibrate(val_df, val_sp_aug, dv_val)

    def _predict_fn(sp: SplitPairTensors, ab1, ab2, ag1, ag2, eo, ee) -> tuple[np.ndarray, np.ndarray]:
        xl = _loc(ab1, ab2, ag1, ag2, eo)
        xg = _glb(ab1, ab2, ag1, ag2, eo)
        dv = two_head.predict(sp, xl, xg)
        sp_a = SplitPairTensors(**{**sp.__dict__, "impact_even": np.concatenate([sp.impact_even, ee], axis=1)})
        return hurdle.predict_3state(sp_a, dv)

    return _predict_fn


# ---------------------------------------------------------------------------
# Supervised 3D Architecture 1: ANTIPASTI (Normal-Mode DCCM 2D CNN, 3-State)
# ---------------------------------------------------------------------------

class _ANTIPASTICNN(nn.Module):
    """Symmetric 2D Convolution over Normal-Mode Correlation Maps (Michalewicz et al., Structure 2024)
    coupled to a 3-state antisymmetric directional + symmetric impact head.
    """

    def __init__(self, n_filters: int = 8, filter_size: int = 4, pool_size: int = 2, odd_dim: int = 22, even_dim: int = 62):
        super().__init__()
        self.conv1 = nn.Conv2d(2, n_filters, kernel_size=filter_size)
        self.pool = nn.MaxPool2d((pool_size, pool_size))
        grid_out = (64 - filter_size + 1) // pool_size
        flat_dim = n_filters * grid_out * grid_out
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.dir_head = nn.Linear(64 + odd_dim, 1, bias=False)
        self.imp_head = nn.Sequential(
            nn.Linear(64 + even_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def encode_map(self, x: torch.Tensor) -> torch.Tensor:
        # Exact Michalewicz et al. symmetric 2D filter: conv(x) + conv(x)^T
        c = self.conv1(x)
        c_sym = 0.5 * (c + c.transpose(2, 3))
        h = self.pool(F.relu(c_sym))
        return self.proj(h.flatten(1))

    def forward(self, m1: torch.Tensor, m2: torch.Tensor, x_odd: torch.Tensor, x_even: torch.Tensor) -> torch.Tensor:
        z1 = self.encode_map(m1)
        z2 = self.encode_map(m2)
        z_dir = self.dir_head(torch.cat([z1 - z2, x_odd], dim=-1)).squeeze(-1)
        z_imp = self.imp_head(torch.cat([torch.abs(z1 - z2), x_even], dim=-1)).squeeze(-1)
        # 3-state logits `[-1, 0, +1]` preserving exact swap symmetry!
        return torch.stack([-z_dir, -z_imp, +z_dir], dim=-1)


class ANTIPASTI3StateTrainer:
    """Leak-free trainer for `ANTIPASTI (Normal-Mode 3D CNN)` on `unseen_ag`."""

    def __init__(self, seed: int = 42, epochs: int = 24, batch_size: int = 256, lr: float = 8e-4):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.odd_scaler = StandardScaler(with_mean=False)
        self.even_scaler = StandardScaler()
        self.net: _ANTIPASTICNN | None = None

    def fit(
        self,
        tr_sf: StructureSplitFeatures,
        tr_sp: SplitPairTensors,
        val_sf: StructureSplitFeatures,
        val_sp: SplitPairTensors,
        val_df: pd.DataFrame,
        w_tr: np.ndarray,
    ) -> "ANTIPASTI3StateTrainer":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        odd_tr = self.odd_scaler.fit_transform(np.concatenate([tr_sf.struct_odd, tr_sp.biophys_odd], axis=1)).astype(np.float32)
        even_tr = self.even_scaler.fit_transform(np.concatenate([tr_sf.struct_even, tr_sp.impact_even], axis=1)).astype(np.float32)
        odd_val = self.odd_scaler.transform(np.concatenate([val_sf.struct_odd, val_sp.biophys_odd], axis=1)).astype(np.float32)
        even_val = self.even_scaler.transform(np.concatenate([val_sf.struct_even, val_sp.impact_even], axis=1)).astype(np.float32)

        net = _ANTIPASTICNN(odd_dim=odd_tr.shape[1], even_dim=even_tr.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=2e-2)

        y_cls_tr = (tr_sp.labels + 1).astype(np.int64)  # {-1,0,+1} -> {0,1,2}
        n = len(y_cls_tr)
        rng = np.random.default_rng(self.seed)
        best_score = -1.0
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        val_loc_m = val_df["Category"].isin(["L1", "L2", "L3"]).to_numpy()
        val_loc_df = val_df.iloc[np.flatnonzero(val_loc_m)].reset_index(drop=True)

        for _ in range(self.epochs):
            net.train()
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s : s + self.batch_size]
                # Random 50% pair orientation flip during training
                flip = rng.random(len(idx)) < 0.5
                m1 = tr_sf.dccm_1[idx].astype(np.float32)
                m2 = tr_sf.dccm_2[idx].astype(np.float32)
                m1_f = np.where(flip[:, None, None, None], m2, m1)
                m2_f = np.where(flip[:, None, None, None], m1, m2)
                xo = np.where(flip[:, None], -odd_tr[idx], odd_tr[idx])
                xe = even_tr[idx]
                yc = np.where(flip, 2 - y_cls_tr[idx], y_cls_tr[idx])
                wb = torch.from_numpy(w_tr[idx])

                logits = net(torch.from_numpy(m1_f), torch.from_numpy(m2_f), torch.from_numpy(xo), torch.from_numpy(xe))
                loss = (F.cross_entropy(logits, torch.from_numpy(yc), reduction="none") * wb).mean()
                l1_reg = 1e-4 * sum(p.abs().sum() for p in net.conv1.parameters())
                opt.zero_grad()
                (loss + l1_reg).backward()
                opt.step()

            net.eval()
            self.net = net
            pd_v, pi_v = self.predict_3state(val_sf, val_sp)
            m3 = compute_per_antigen_macro_metrics(val_loc_df, pd_v[val_loc_m], pi_v[val_loc_m], min_pairs=5)
            sc = 0.5 * (m3["auc_ordinal"] or 0.5) + 0.5 * (m3["auc_impact"] or 0.5)
            if sc > best_score:
                best_score = sc
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        net.load_state_dict(best_state)
        net.eval()
        self.net = net
        return self

    def predict_3state(self, sf: StructureSplitFeatures, sp: SplitPairTensors) -> tuple[np.ndarray, np.ndarray]:
        assert self.net is not None
        self.net.eval()
        xo = self.odd_scaler.transform(np.concatenate([sf.struct_odd, sp.biophys_odd], axis=1)).astype(np.float32)
        xe = self.even_scaler.transform(np.concatenate([sf.struct_even, sp.impact_even], axis=1)).astype(np.float32)
        n = len(xo)
        p_diff = np.zeros(n, dtype=np.float32)
        p_imp = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            for s in range(0, n, 512):
                m1 = torch.from_numpy(sf.dccm_1[s : s + 512].astype(np.float32))
                m2 = torch.from_numpy(sf.dccm_2[s : s + 512].astype(np.float32))
                b_o = torch.from_numpy(xo[s : s + 512])
                b_e = torch.from_numpy(xe[s : s + 512])
                lg = self.net(m1, m2, b_o, b_e)
                probs = F.softmax(lg, dim=-1).numpy()
                p_diff[s : s + 512] = probs[:, 2] - probs[:, 0]
                p_imp[s : s + 512] = probs[:, 2] + probs[:, 0]
        return p_diff, p_imp


# ---------------------------------------------------------------------------
# Supervised 3D Architecture 2: GearBind (Multi-Relational 3D Geometric GNN)
# ---------------------------------------------------------------------------

class _GearBindRelationalLayer(nn.Module):
    """Dense batched multi-relational geometric graph convolution over `K=40` 3D interface patches:
      Relation 0: Sequential (`|pos_i - pos_j| <= 2` on same chain)
      Relation 1: Spatial intra-chain (`d_CB < 8.0 Å` on same chain)
      Relation 2: Cross-interface Ab–Ag (`d_CB < 10.5 Å` between Ab `{H,L}` and Ag `{A}`)
    """

    def __init__(self, dim: int, edge_dim: int = 20):
        super().__init__()
        self.lin_rel = nn.Linear(dim * 3 + edge_dim * 3, dim)
        self.edge_gate = nn.Sequential(nn.Linear(edge_dim, 3), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, adj_rel: torch.Tensor, edge_feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: [B, K, D], adj_rel: [B, 3, K, K], edge_feat: [B, K, K, E]
        bsz, k_len, _, e_dim = edge_feat.shape
        gates = self.edge_gate(edge_feat).permute(0, 3, 1, 2)  # [B, 3, K, K]
        w_gated = adj_rel * gates                              # [B, 3, K, K]
        ef_flat = edge_feat.reshape(bsz * k_len, k_len, e_dim)
        msgs = []
        for r in range(3):
            w = w_gated[:, r, :, :]                            # [B, K, K]
            deg = adj_rel[:, r, :, :].sum(dim=-1, keepdim=True).clamp(min=1.0)
            agg_h = torch.bmm(w, h) / deg                      # [B, K, D]
            agg_e = torch.bmm(w.reshape(bsz * k_len, 1, k_len), ef_flat).reshape(bsz, k_len, e_dim) / deg
            msgs.extend([agg_h, agg_e])
        out = self.lin_rel(torch.cat(msgs, dim=-1))
        out = self.norm(h + F.gelu(out))
        return out * mask.unsqueeze(-1)


class _GearBind3StateNet(nn.Module):
    """GearBind multi-relational geometric encoder + antisymmetric `BindModel` & 3D impact heads."""

    def __init__(self, in_dim: int = 90, hid_dim: int = 96, odd_dim: int = 22, even_dim: int = 24):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.LayerNorm(hid_dim), nn.GELU())
        self.gnn1 = _GearBindRelationalLayer(hid_dim, edge_dim=20)
        self.gnn2 = _GearBindRelationalLayer(hid_dim, edge_dim=20)
        self.gnn3 = _GearBindRelationalLayer(hid_dim, edge_dim=20)
        self.attn_pool = nn.Linear(hid_dim, 1, bias=False)

        # Exact GearBind BindModel antisymmetric MLP([g1, g2]) - MLP([g2, g1]) + 3D ΔΔlog p residual
        self.bind_mlp = nn.Sequential(
            nn.Linear(hid_dim * 4, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hid_dim, 1, bias=False),
        )
        self.odd_skip = nn.Linear(odd_dim, 1, bias=False)
        self.imp_mlp = nn.Sequential(
            nn.Linear(hid_dim * 2 + even_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hid_dim, 1),
        )

    @staticmethod
    def build_geom_graph(
        ca: torch.Tensor,
        cb: torch.Tensor,
        chain_id: torch.Tensor,
        seqpos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = (chain_id >= 0).float()  # [B, K]
        pair_valid = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, K, K]

        d_ca = torch.cdist(ca, ca)
        d_cb = torch.cdist(cb, cb)
        same_c = (chain_id.unsqueeze(2) == chain_id.unsqueeze(1)).float() * pair_valid
        is_ab = ((chain_id == 0) | (chain_id == 1)).float()
        is_ag = (chain_id == 2).float()
        cross_int = (is_ab.unsqueeze(2) * is_ag.unsqueeze(1) + is_ag.unsqueeze(2) * is_ab.unsqueeze(1)) * pair_valid

        seq_sep = torch.abs(seqpos.unsqueeze(2) - seqpos.unsqueeze(1)).float()
        eye = torch.eye(ca.shape[1], device=ca.device).unsqueeze(0)
        r0 = same_c * (seq_sep <= 2.0).float() * (1.0 - eye)
        r1 = same_c * (d_cb <= 8.0).float() * (1.0 - eye)
        r2 = cross_int * (d_cb <= 10.5).float()
        adj_rel = torch.stack([r0, r1, r2], dim=1)

        # 16 Gaussian RBFs on C-beta distance (2..14 Å) + C-alpha distance + orientational cosine
        centers = torch.linspace(2.0, 14.0, 16, device=ca.device).view(1, 1, 1, 16)
        rbf = torch.exp(-0.5 * ((d_cb.unsqueeze(-1) - centers) / 0.85) ** 2)
        u_vec = F.normalize(cb - ca, dim=-1, eps=1e-6)
        ori_dot = torch.einsum("bik,bjk->bij", u_vec, u_vec).unsqueeze(-1)
        ef = torch.cat(
            [
                rbf,
                (d_ca.clamp(max=20.0) / 20.0).unsqueeze(-1),
                same_c.unsqueeze(-1),
                cross_int.unsqueeze(-1),
                ori_dot,
            ],
            dim=-1,
        )
        return adj_rel, ef, mask

    def encode_patch(
        self,
        nodes: torch.Tensor,
        ca: torch.Tensor | None = None,
        cb: torch.Tensor | None = None,
        chain_id: torch.Tensor | None = None,
        seqpos: torch.Tensor | None = None,
        *,
        precomp: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if precomp is not None:
            adj_rel, ef, mask = precomp
        else:
            assert ca is not None and cb is not None and chain_id is not None and seqpos is not None
            adj_rel, ef, mask = self.build_geom_graph(ca, cb, chain_id, seqpos)
        h = self.in_proj(nodes) * mask.unsqueeze(-1)
        h = self.gnn1(h, adj_rel, ef, mask)
        h = self.gnn2(h, adj_rel, ef, mask)
        h = self.gnn3(h, adj_rel, ef, mask)

        # Attention readout + mutated-site readout (`nodes[:, :, 89]` == is_mutated indicator!)
        att_logits = self.attn_pool(h).squeeze(-1).masked_fill(mask < 0.5, -1e9)
        att_w = F.softmax(att_logits, dim=-1)
        g_att = torch.einsum("bk,bkd->bd", att_w, h)

        mut_m = nodes[:, :, 89] * mask
        mut_sum = mut_m.sum(dim=-1, keepdim=True)
        g_mut = torch.where(
            mut_sum > 0.5,
            torch.einsum("bk,bkd->bd", mut_m, h) / mut_sum.clamp(min=1.0),
            g_att,
        )
        return torch.cat([g_att, g_mut], dim=-1)  # [B, 2*hid_dim]

    def forward(
        self,
        n1: torch.Tensor, ca1: torch.Tensor | None, cb1: torch.Tensor | None, c1: torch.Tensor | None, s1: torch.Tensor | None,
        n2: torch.Tensor, ca2: torch.Tensor | None, cb2: torch.Tensor | None, c2: torch.Tensor | None, s2: torch.Tensor | None,
        x_odd: torch.Tensor,
        x_even: torch.Tensor,
        *,
        precomp1: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        precomp2: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        g1 = self.encode_patch(n1, ca1, cb1, c1, s1, precomp=precomp1)
        g2 = self.encode_patch(n2, ca2, cb2, c2, s2, precomp=precomp2)
        z_dir = (
            self.bind_mlp(torch.cat([g1, g2], dim=-1)).squeeze(-1)
            - self.bind_mlp(torch.cat([g2, g1], dim=-1)).squeeze(-1)
            + self.odd_skip(x_odd).squeeze(-1)
        )
        z_imp = self.imp_mlp(torch.cat([torch.abs(g1 - g2), x_even], dim=-1)).squeeze(-1)
        return torch.stack([-z_dir, -z_imp, +z_dir], dim=-1)


class GearBind3StateTrainer:
    """Leak-free trainer for `GearBind (3D Geometric Interface GNN)` on `unseen_ag`."""

    def __init__(self, seed: int = 42, epochs: int = 18, batch_size: int = 256, lr: float = 9e-4):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.odd_scaler = StandardScaler(with_mean=False)
        self.even_scaler = StandardScaler()
        self.net: _GearBind3StateNet | None = None

    @staticmethod
    def _precompute_graphs(
        ca: np.ndarray,
        cb: np.ndarray,
        ch: np.ndarray,
        sp_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, k, _ = ca.shape
        adj_all = np.zeros((n, 3, k, k), dtype=np.uint8)
        ef_all = np.zeros((n, k, k, 20), dtype=np.float16)
        mask_all = np.zeros((n, k), dtype=np.float16)
        with torch.no_grad():
            for s in range(0, n, 512):
                e = min(s + 512, n)
                adj, ef, msk = _GearBind3StateNet.build_geom_graph(
                    torch.from_numpy(ca[s:e].astype(np.float32)),
                    torch.from_numpy(cb[s:e].astype(np.float32)),
                    torch.from_numpy(ch[s:e].astype(np.int64)),
                    torch.from_numpy(sp_pos[s:e].astype(np.int64)),
                )
                adj_all[s:e] = adj.numpy().astype(np.uint8)
                ef_all[s:e] = ef.numpy().astype(np.float16)
                mask_all[s:e] = msk.numpy().astype(np.float16)
        return adj_all, ef_all, mask_all

    def fit(
        self,
        tr_sf: StructureSplitFeatures,
        tr_sp: SplitPairTensors,
        val_sf: StructureSplitFeatures,
        val_sp: SplitPairTensors,
        val_df: pd.DataFrame,
        w_tr: np.ndarray,
        *,
        precomp_tr_g: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
        precomp_val_g: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    ) -> "GearBind3StateTrainer":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        odd_tr = self.odd_scaler.fit_transform(tr_sf.struct_odd).astype(np.float32)
        even_tr = self.even_scaler.fit_transform(tr_sf.struct_even).astype(np.float32)

        net = _GearBind3StateNet(in_dim=90, hid_dim=80, odd_dim=odd_tr.shape[1], even_dim=even_tr.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=2e-2)

        y_cls_tr = (tr_sp.labels + 1).astype(np.int64)
        n = len(y_cls_tr)
        rng = np.random.default_rng(self.seed)
        best_score = -1.0
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        val_loc_m = val_df["Category"].isin(["L1", "L2", "L3"]).to_numpy()
        val_loc_df = val_df.iloc[np.flatnonzero(val_loc_m)].reset_index(drop=True)

        if precomp_tr_g is not None:
            (adj1_tr, ef1_tr, msk1_tr), (adj2_tr, ef2_tr, msk2_tr) = precomp_tr_g
        else:
            adj1_tr, ef1_tr, msk1_tr = self._precompute_graphs(
                tr_sf.patch_ca_1, tr_sf.patch_cb_1, tr_sf.patch_chain_1, tr_sf.patch_seqpos_1
            )
            adj2_tr, ef2_tr, msk2_tr = self._precompute_graphs(
                tr_sf.patch_ca_2, tr_sf.patch_cb_2, tr_sf.patch_chain_2, tr_sf.patch_seqpos_2
            )
        if precomp_val_g is not None:
            val_g1, val_g2 = precomp_val_g
        else:
            val_g1 = self._precompute_graphs(
                val_sf.patch_ca_1, val_sf.patch_cb_1, val_sf.patch_chain_1, val_sf.patch_seqpos_1
            )
            val_g2 = self._precompute_graphs(
                val_sf.patch_ca_2, val_sf.patch_cb_2, val_sf.patch_chain_2, val_sf.patch_seqpos_2
            )

        for _ in range(self.epochs):
            net.train()
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s : s + self.batch_size]
                flip = rng.random(len(idx)) < 0.5

                def _sel(a, b, dt):
                    arr_a = a[idx].astype(dt)
                    arr_b = b[idx].astype(dt)
                    sh = (len(idx),) + (1,) * (arr_a.ndim - 1)
                    fl = flip.reshape(sh)
                    return torch.from_numpy(np.where(fl, arr_b, arr_a)), torch.from_numpy(np.where(fl, arr_a, arr_b))

                n1, n2 = _sel(tr_sf.patch_node_1, tr_sf.patch_node_2, np.float32)
                a1, a2 = _sel(adj1_tr, adj2_tr, np.float32)
                e1, e2 = _sel(ef1_tr, ef2_tr, np.float32)
                m1, m2 = _sel(msk1_tr, msk2_tr, np.float32)

                xo = torch.from_numpy(np.where(flip[:, None], -odd_tr[idx], odd_tr[idx]))
                xe = torch.from_numpy(even_tr[idx])
                yc = torch.from_numpy(np.where(flip, 2 - y_cls_tr[idx], y_cls_tr[idx]))
                wb = torch.from_numpy(w_tr[idx])

                lg = net(
                    n1, None, None, None, None,
                    n2, None, None, None, None,
                    xo, xe,
                    precomp1=(a1, e1, m1),
                    precomp2=(a2, e2, m2),
                )
                loss = (F.cross_entropy(lg, yc, reduction="none") * wb).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            net.eval()
            self.net = net
            pd_v, pi_v = self.predict_3state(val_sf, _precomp_g=(val_g1, val_g2))
            m3 = compute_per_antigen_macro_metrics(val_loc_df, pd_v[val_loc_m], pi_v[val_loc_m], min_pairs=5)
            sc = 0.55 * (m3["auc_ordinal"] or 0.5) + 0.45 * (m3["auc_impact"] or 0.5)
            if sc > best_score:
                best_score = sc
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        net.load_state_dict(best_state)
        net.eval()
        self.net = net
        return self

    def predict_3state(
        self,
        sf: StructureSplitFeatures,
        *,
        _precomp_g: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.net is not None
        self.net.eval()
        xo = self.odd_scaler.transform(sf.struct_odd).astype(np.float32)
        xe = self.even_scaler.transform(sf.struct_even).astype(np.float32)
        n = len(xo)
        p_diff = np.zeros(n, dtype=np.float32)
        p_imp = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            for s in range(0, n, 256):
                e = min(s + 256, n)
                if _precomp_g is not None:
                    (a1, ef1, m1), (a2, ef2, m2) = _precomp_g
                    lg = self.net(
                        torch.from_numpy(sf.patch_node_1[s:e].astype(np.float32)),
                        None, None, None, None,
                        torch.from_numpy(sf.patch_node_2[s:e].astype(np.float32)),
                        None, None, None, None,
                        torch.from_numpy(xo[s:e]),
                        torch.from_numpy(xe[s:e]),
                        precomp1=(
                            torch.from_numpy(a1[s:e].astype(np.float32)),
                            torch.from_numpy(ef1[s:e].astype(np.float32)),
                            torch.from_numpy(m1[s:e].astype(np.float32)),
                        ),
                        precomp2=(
                            torch.from_numpy(a2[s:e].astype(np.float32)),
                            torch.from_numpy(ef2[s:e].astype(np.float32)),
                            torch.from_numpy(m2[s:e].astype(np.float32)),
                        ),
                    )
                else:
                    lg = self.net(
                        torch.from_numpy(sf.patch_node_1[s:e].astype(np.float32)),
                        torch.from_numpy(sf.patch_ca_1[s:e]),
                        torch.from_numpy(sf.patch_cb_1[s:e]),
                        torch.from_numpy(sf.patch_chain_1[s:e].astype(np.int64)),
                        torch.from_numpy(sf.patch_seqpos_1[s:e].astype(np.int64)),
                        torch.from_numpy(sf.patch_node_2[s:e].astype(np.float32)),
                        torch.from_numpy(sf.patch_ca_2[s:e]),
                        torch.from_numpy(sf.patch_cb_2[s:e]),
                        torch.from_numpy(sf.patch_chain_2[s:e].astype(np.int64)),
                        torch.from_numpy(sf.patch_seqpos_2[s:e].astype(np.int64)),
                        torch.from_numpy(xo[s:e]),
                        torch.from_numpy(xe[s:e]),
                    )
                probs = F.softmax(lg, dim=-1).numpy()
                p_diff[s:e] = probs[:, 2] - probs[:, 0]
                p_imp[s:e] = probs[:, 2] + probs[:, 0]
        return p_diff, p_imp


# ---------------------------------------------------------------------------
# Supervised 3D Architecture 3: WALLE-Affinity (3-State 3D Complex Graph + PLM)
# ---------------------------------------------------------------------------

def fit_walle_affinity_3state(
    tr_sf: StructureSplitFeatures,
    tr_sp: SplitPairTensors,
    val_sf: StructureSplitFeatures,
    val_sp: SplitPairTensors,
    val_df: pd.DataFrame,
    w_tr: np.ndarray,
    seed: int = 42,
):
    """Fit `WALLE-Affinity (3-State 3D Graph)` strictly leak-free on `unseen_ag/train`:
    Combines 10 Å contact-pooled 3D GVP/MPNN graph embeddings (`if1_emb`, `mpnn_emb`) with
    PLM node embeddings (`ESMC-300M` + `AbLang2`) and 3D interface thermodynamic/SASA features.
    """
    def _loc_mat(sf: StructureSplitFeatures, sp: SplitPairTensors) -> np.ndarray:
        return build_odd_linear_matrix(
            [sp.esmc_ab_1, sp.ablang_1, sp.esmc_ag_1, sf.if1_emb_1, sf.mpnn_emb_1],
            [sp.esmc_ab_2, sp.ablang_2, sp.esmc_ag_2, sf.if1_emb_2, sf.mpnn_emb_2],
            extra_odd=np.concatenate([sp.zs_features, sp.biophys_odd, sf.struct_odd], axis=1),
            include_unit_direction=True,
        )

    def _glb_mat(sf: StructureSplitFeatures, sp: SplitPairTensors) -> np.ndarray:
        return build_odd_linear_matrix(
            [sp.esmc_ab_1, sp.ablang_1, sp.esmc_ag_1, sf.if1_emb_1, sf.mpnn_emb_1],
            [sp.esmc_ab_2, sp.ablang_2, sp.esmc_ag_2, sf.if1_emb_2, sf.mpnn_emb_2],
            extra_odd=np.concatenate([sp.zs_features, sf.struct_odd], axis=1),
            include_unit_direction=False,
        )

    X_loc_tr = _loc_mat(tr_sf, tr_sp)
    X_loc_val = _loc_mat(val_sf, val_sp)
    X_glb_tr = _glb_mat(tr_sf, tr_sp)
    X_glb_val = _glb_mat(val_sf, val_sp)

    m_loc = AntisymmetricLinearRanker(use_neutral_nullspace=True).fit(
        X_loc_tr, tr_sp.y, w_tr, X_loc_val, val_df, val_sp.is_ag_mut, labels_tr=tr_sp.labels
    )
    glb_tr_m = ~(tr_sp.is_same_ag | (~tr_sp.is_ab_mut))
    if glb_tr_m.sum() < 4:
        glb_tr_m = np.ones(len(tr_sp.y), dtype=bool)
    glb_val_m = (val_df["Category"] == "Global").to_numpy()
    if glb_val_m.sum() < 4:
        glb_val_m = np.ones(len(val_df), dtype=bool)
    m_glb = AntisymmetricLinearRanker(use_neutral_nullspace=False).fit(
        X_glb_tr[glb_tr_m], tr_sp.y[glb_tr_m], None,
        X_glb_val[glb_val_m], val_df.loc[glb_val_m].reset_index(drop=True), val_sp.is_ag_mut[glb_val_m],
    )
    two_head = TwoHeadRegimeRanker(local_head=m_loc, global_head=m_glb)
    dv_tr = two_head.predict(tr_sp, X_loc_tr, X_glb_tr)
    dv_val = two_head.predict(val_sp, X_loc_val, X_glb_val)

    tr_sp_aug = SplitPairTensors(
        **{**tr_sp.__dict__, "impact_even": np.concatenate([tr_sp.impact_even, tr_sf.struct_even], axis=1)}
    )
    val_sp_aug = SplitPairTensors(
        **{**val_sp.__dict__, "impact_even": np.concatenate([val_sp.impact_even, val_sf.struct_even], axis=1)}
    )
    imp_clf = SymmetricImpactClassifier(seed=seed).fit(tr_sp_aug, val_sp_aug, dv_tr, dv_val, val_df=val_df)
    hurdle = OrdinalHurdleRanker(two_head, imp_clf).calibrate(val_df, val_sp_aug, dv_val)

    def _predict_fn(sf: StructureSplitFeatures, sp: SplitPairTensors) -> tuple[np.ndarray, np.ndarray]:
        dv = two_head.predict(sp, _loc_mat(sf, sp), _glb_mat(sf, sp))
        sp_a = SplitPairTensors(**{**sp.__dict__, "impact_even": np.concatenate([sp.impact_even, sf.struct_even], axis=1)})
        return hurdle.predict_3state(sp_a, dv)

    return _predict_fn
