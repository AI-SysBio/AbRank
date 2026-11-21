#!/usr/bin/env python3
"""Step C3: Ingest Multi-Pose Co-Folded Structures (`af3`, `boltz2`, `chai1`, `protenix`),
Compute `DockQ` vs. All 340 Native Experimental WT Co-Crystals, and Evaluate Multi-Pose
Strategies + All 8 3D Models Across the 18-Cluster Co-Crystal Ablation and 22-Cluster
Prospective Crystal-Free `LOACO-Local` Benchmark.

Supports both directory layouts:
  1. Extracted archive layout (`data/structures/cofolded_structures/<method>/<scaffold_id>/pose_XX.cif`
     + `data/structures/cofolded_confidence/<method>/<scaffold_id>/pose_XX.{json,npz}`)
  2. Raw worker layout (`data/structures/cofolded/<method>/<scaffold_id>/`)

Key capabilities:
  1. Computes exact Kabsch `DockQ`, `fnat`, `iRMSD`, and `LRMSD` across all 340 solved WT
     co-crystals (`3,400` poses/method = `13,600` poses) and extracts global `ipTM`,
     cross-interface `Ab-Ag Pair ipTM` (removing `VH-VL` dimer confounding on 3-chain Fabs),
     `pTM`, and `ranking_score` across all 851 WT scaffolds (`34,040` poses).
  2. Accounts for the `pose_00` ordering difference across tools (`Boltz-2` emits `model_0`
     pre-sorted by `confidence_score`, whereas `AF3`, `Chai-1`, and `Protenix` emit
     unranked `seed-101_sample-0` first) by reporting both `pose0_single_seed` and
     `expected_single_pose` (mean across individual poses `k=0..9`).
  3. Standardizes all 10 poses for all 94 `LOACO-Local` scaffolds (`940` poses/method =
     `3,760` poses) and extracts 3D biophysical features (`FreeSASA`, `PRODIGY`, `GNM DCCM`,
     `Cα/Cβ` contact distances), `ProteinMPNN`, `ESM-IF1`, and `AntiFold` log-likelihoods,
     plus `ANTIPASTI` and `GearBind` 3-state predictions.
  4. Evaluates 8 multi-pose selection & ensembling strategies across both the 18-cluster
     (`19` Impact) Controlled Co-Crystal Ablation (`47` solved WT co-crystals) and the full
     22-cluster Crystal-Free `LOACO-Local` benchmark (`94` co-folded scaffolds), plus a
     40-pose Multi-Method Consensus (`AF3 + Boltz-2 + Chai-1 + Protenix`).

Outputs:
  - `data/structures/cofolded_dockq_and_confidence_cache.pt`
  - `data/structures/cofolded_cache_<method>.pt`
  - `results/cofolding_<method>_evaluation.json` & `results/COFOLDING_<METHOD>_EVALUATION.md`
  - `results/cofolding_all_methods_evaluation.json` & `results/COFOLDING_BENCHMARK_EVALUATION.md`
"""

from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import torch
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")

from abrank.baselines.logica_models import (
    LogiCA3StateTrainer,
    select_stratified_local_indices,
)
from abrank.baselines.models import assemble_split_tensors, compute_stratum_weights
from abrank.baselines.structure_models import (
    ANTIPASTI3StateTrainer,
    GearBind3StateTrainer,
    assemble_structure_pair_features,
)
from abrank.config import get_repo_root
from abrank.evaluate import compute_3state_metrics, compute_cluster_bootstrap_summary
from abrank.schema import trim_fab_to_fv
from abrank.structures.resolver import (
    compute_scaffold_biophysics,
    parse_structure_chains,
    write_standardized_pdb,
)

warnings.filterwarnings("ignore")

METHODS = ["af3", "boltz2", "chai1", "protenix"]
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Superpose `P` onto `Q` (`(N, 3)`) via Kabsch SVD; return `(R, t, rmsd)`."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    P0, Q0 = P - cp, Q - cq
    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, -1.0 if d < 0 else 1.0])
    R = Vt.T @ S @ U.T
    t = cq - cp @ R.T
    P_fit = P @ R.T + t
    rmsd = float(np.sqrt(np.mean(np.sum((P_fit - Q) ** 2, axis=1))))
    return R, t, rmsd


def compute_dockq_ca(
    ca_ab_pred: np.ndarray,
    ca_ag_pred: np.ndarray,
    ca_ab_nat: np.ndarray,
    ca_ag_nat: np.ndarray,
) -> dict[str, float]:
    """Compute standard Basu & Wallner (2016) `DockQ` from native vs. predicted Ab/Ag Cα traces."""
    n_ab = min(len(ca_ab_pred), len(ca_ab_nat))
    n_ag = min(len(ca_ag_pred), len(ca_ag_nat))
    ab_p = np.asarray(ca_ab_pred[:n_ab], dtype=np.float64)
    ab_n = np.asarray(ca_ab_nat[:n_ab], dtype=np.float64)
    ag_p = np.asarray(ca_ag_pred[:n_ag], dtype=np.float64)
    ag_n = np.asarray(ca_ag_nat[:n_ag], dtype=np.float64)

    # 1. fnat: native inter-chain Cα contacts within 8.5 Å
    d_nat = np.linalg.norm(ab_n[:, None, :] - ag_n[None, :, :], axis=-1)
    d_prd = np.linalg.norm(ab_p[:, None, :] - ag_p[None, :, :], axis=-1)
    nat_contacts = d_nat <= 8.5
    n_nat_c = int(nat_contacts.sum())
    fnat = float((nat_contacts & (d_prd <= 8.5)).sum() / max(n_nat_c, 1))

    # 2. LRMSD: superpose receptor (Antigen) and measure RMSD on ligand (Antibody Fv/VHH)
    R_ag, t_ag, _ = kabsch_rmsd(ag_p, ag_n)
    ab_after_ag = ab_p @ R_ag.T + t_ag
    lrmsd = float(np.sqrt(np.mean(np.sum((ab_after_ag - ab_n) ** 2, axis=1))))

    # 3. iRMSD: superpose native interface residues (Cα <= 10.0 Å across Ab-Ag)
    ab_if = np.any(d_nat <= 10.0, axis=1)
    ag_if = np.any(d_nat <= 10.0, axis=0)
    if int(ab_if.sum()) + int(ag_if.sum()) >= 4:
        P_if = np.concatenate([ab_p[ab_if], ag_p[ag_if]], axis=0)
        Q_if = np.concatenate([ab_n[ab_if], ag_n[ag_if]], axis=0)
        _, _, irmsd = kabsch_rmsd(P_if, Q_if)
    else:
        irmsd = lrmsd

    dockq = float((fnat + 1.0 / (1.0 + (irmsd / 1.5) ** 2) + 1.0 / (1.0 + (lrmsd / 8.5) ** 2)) / 3.0)
    return {"DockQ": round(dockq, 4), "fnat": round(fnat, 4), "iRMSD": round(irmsd, 3), "LRMSD": round(lrmsd, 3)}


def read_confidence_file(method: str, cpath: Path, has_l: bool) -> dict[str, float]:
    """Extract `iptm`, cross-interface `ab_ag_iptm`, `ptm`, and `ranking_score` from JSON/NPZ."""
    if method == "chai1" or cpath.suffix.lower() == ".npz":
        nd = np.load(cpath)
        iptm = float(np.asarray(nd["iptm"]).reshape(-1)[0])
        ptm = float(np.asarray(nd.get("ptm", nd["iptm"])).reshape(-1)[0])
        rscore = float(np.asarray(nd.get("aggregate_score", nd["iptm"])).reshape(-1)[0])
        if "per_chain_pair_iptm" in nd:
            cp = np.asarray(nd["per_chain_pair_iptm"])[0]
            if has_l and cp.shape[0] >= 3:
                ab_ag_iptm = float(0.25 * (cp[0, 2] + cp[2, 0] + cp[1, 2] + cp[2, 1]))
            else:
                ab_ag_iptm = float(0.5 * (cp[0, 1] + cp[1, 0]))
        else:
            ab_ag_iptm = iptm
        return {"iptm": iptm, "ab_ag_iptm": ab_ag_iptm, "ptm": ptm, "ranking_score": rscore}

    jd = json.loads(cpath.read_text())
    iptm = float(jd["iptm"])
    ptm = float(jd.get("ptm", iptm))
    rscore = float(jd.get("ranking_score", jd.get("confidence_score", iptm)))
    ab_ag_iptm = iptm
    if method == "af3" and "chain_pair_iptm" in jd:
        cp = jd["chain_pair_iptm"]
        # AF3 sorts chains alphabetically: A (0), H (1), [L (2)]
        ab_ag_iptm = float(0.5 * (cp[0][1] + cp[0][2])) if (has_l and len(cp) >= 3) else float(cp[0][1])
    elif method == "boltz2" and "pair_chains_iptm" in jd:
        cp = jd["pair_chains_iptm"]
        ab_ag_iptm = (
            float(0.25 * (cp["0"]["2"] + cp["2"]["0"] + cp["1"]["2"] + cp["2"]["1"]))
            if (has_l and "2" in cp)
            else float(0.5 * (cp["0"]["1"] + cp["1"]["0"]))
        )
    elif "chain_pair_iptm" in jd:
        cp = jd["chain_pair_iptm"]
        ab_ag_iptm = float(0.5 * (cp[0][2] + cp[1][2])) if (has_l and len(cp) >= 3) else float(cp[0][1])
    return {"iptm": iptm, "ab_ag_iptm": ab_ag_iptm, "ptm": ptm, "ranking_score": rscore}


def discover_scaffold_poses(sdir: Path, method: str, sid: str, has_l: bool, max_poses: int = 10) -> list[dict]:
    """Discover up to `max_poses` structure + confidence pairs for `(method, sid)`."""
    st_dir = sdir / "cofolded_structures" / method / sid
    cf_dir = sdir / "cofolded_confidence" / method / sid
    if st_dir.exists() and cf_dir.exists():
        poses = []
        for k in range(max_poses):
            pfile = st_dir / f"pose_{k:02d}.cif"
            cfile = cf_dir / (f"pose_{k:02d}.npz" if method == "chai1" else f"pose_{k:02d}.json")
            if not pfile.exists():
                break
            conf = read_confidence_file(method, cfile, has_l) if cfile.exists() else {"iptm": 0.5, "ab_ag_iptm": 0.5, "ptm": 0.5, "ranking_score": 0.5}
            poses.append({"pose_idx": k, "struct_path": pfile, **conf})
        return poses

    # Fallback: legacy single directory `data/structures/cofolded/<method>/<sid>/`
    legacy_dir = sdir / "cofolded" / method / sid
    if not legacy_dir.exists():
        return []
    struct_files = sorted(
        p for p in legacy_dir.rglob("*")
        if p.suffix.lower() in (".cif", ".mmcif", ".pdb") and "ranking" not in p.name.lower()
    )
    sample_files = [p for p in struct_files if ("sample" in p.as_posix().lower() or "model_" in p.name.lower() or "pred." in p.name.lower() or "pose_" in p.name.lower())]
    use_files = (sample_files if sample_files else struct_files)[:max_poses]
    poses = []
    for k, sp in enumerate(use_files):
        conf = {"iptm": 0.5, "ab_ag_iptm": 0.5, "ptm": 0.5, "ranking_score": 0.5}
        cand = list(sp.parent.glob(f"*{sp.stem.replace('_model', '')}*summary*.json")) + list(sp.parent.glob("confidence*.json")) + list(sp.parent.glob("scores*.npz"))
        if cand:
            try:
                conf = read_confidence_file(method, cand[0], has_l)
            except Exception:
                pass
        poses.append({"pose_idx": k, "struct_path": sp, **conf})
    return poses


def _biophysics_pose_worker(args: tuple) -> tuple[str, int, dict]:
    """Standardize `(H, [L], A)` PDB and compute scaffold biophysics (`FreeSASA`, `PRODIGY`, `DCCM`) for one pose."""
    sid, k_idx, pfile_str, std_pdb_str, seq_h, seq_l, seq_a, nat_3di_h, nat_3di_l, nat_3di_a = args
    pfile = Path(pfile_str)
    std_pdb = Path(std_pdb_str)
    by_id = parse_structure_chains(pfile)
    ordered = list(by_id.values())
    ch_h = by_id.get("H", by_id.get("A", ordered[0]))
    if seq_l:
        ch_l = by_id.get("L", by_id.get("B", ordered[min(1, len(ordered) - 1)]))
        ch_a = by_id.get("A" if "H" in by_id else "C", ordered[-1])
    else:
        ch_l = None
        ch_a = by_id.get("A" if "H" in by_id else "B", ordered[-1])

    if not std_pdb.exists():
        write_standardized_pdb(std_pdb, ch_h, ch_l, ch_a)
    bio = compute_scaffold_biophysics(std_pdb, ch_h, ch_l, ch_a, is_cocrystal=True)
    bio["pdb_path"] = str(std_pdb)
    bio["3di_H"] = nat_3di_h
    bio["3di_L"] = nat_3di_l
    bio["3di_A"] = nat_3di_a
    return sid, k_idx, bio


def _fast_protein_mpnn_worker(args: tuple[list[tuple[str, str, bool, bool]], str]) -> dict[str, dict]:
    """Worker running official ProteinMPNN (`v_48_020` + `soluble_v_48_020` ensemble) single-threaded per worker."""
    shard, mpnn_repo = args
    import torch
    import torch.nn.functional as F
    if mpnn_repo not in sys.path:
        sys.path.insert(0, mpnn_repo)
    from protein_mpnn_utils import ProteinMPNN, cat_neighbors_nodes, gather_nodes, parse_PDB, tied_featurize

    mpnn_alpha = "ACDEFGHIKLMNPQRSTVWYX"
    reorder = [mpnn_alpha.index(a) for a in AA_LIST]

    models = []
    for sub_dir in ("vanilla_model_weights", "soluble_model_weights"):
        ckpt_p = Path(mpnn_repo) / sub_dir / "v_48_020.pt"
        ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        mdl = ProteinMPNN(
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=0.0,
            k_neighbors=ckpt["num_edges"],
        )
        mdl.load_state_dict(ckpt["model_state_dict"])
        mdl.eval()
        models.append(mdl)

    f0 = models[0].features

    def _fast_rbf(A: torch.Tensor, B: torch.Tensor, E_idx: torch.Tensor) -> torch.Tensor:
        B_n = gather_nodes(B, E_idx)
        d = torch.sqrt(torch.sum((A[:, :, None, :] - B_n) ** 2, -1) + 1e-6)
        return f0._rbf(d)

    def _score_pdb_chains(pdb_path: str, chains: list[str], want_emb: bool = False):
        parsed = parse_PDB(pdb_path, input_chain_list=chains)
        X, S, mask, lengths, chain_M, chain_encoding_all, *_rest = tied_featurize(
            [parsed[0]], torch.device("cpu"), None, None, None, None, None, None, ca_only=False
        )
        residue_idx = _rest[6]
        lp_sum = None
        emb_sum = None
        with torch.no_grad():
            g = torch.Generator()
            g.manual_seed(42)
            randn = torch.randn(chain_M.shape, generator=g)

            b = X[:, :, 1, :] - X[:, :, 0, :]
            c = X[:, :, 2, :] - X[:, :, 1, :]
            a = torch.cross(b, c, dim=-1)
            Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
            Ca, N, C, O = X[:, :, 1, :], X[:, :, 0, :], X[:, :, 2, :], X[:, :, 3, :]
            D_neighbors, E_idx = f0._dist(Ca, mask)
            RBF_all = torch.cat([
                f0._rbf(D_neighbors),
                _fast_rbf(N, N, E_idx), _fast_rbf(C, C, E_idx), _fast_rbf(O, O, E_idx), _fast_rbf(Cb, Cb, E_idx),
                _fast_rbf(Ca, N, E_idx), _fast_rbf(Ca, C, E_idx), _fast_rbf(Ca, O, E_idx), _fast_rbf(Ca, Cb, E_idx),
                _fast_rbf(N, C, E_idx), _fast_rbf(N, O, E_idx), _fast_rbf(N, Cb, E_idx),
                _fast_rbf(Cb, C, E_idx), _fast_rbf(Cb, O, E_idx), _fast_rbf(O, C, E_idx),
                _fast_rbf(N, Ca, E_idx), _fast_rbf(C, Ca, E_idx), _fast_rbf(O, Ca, E_idx), _fast_rbf(Cb, Ca, E_idx),
                _fast_rbf(C, N, E_idx), _fast_rbf(O, N, E_idx), _fast_rbf(Cb, N, E_idx),
                _fast_rbf(C, Cb, E_idx), _fast_rbf(O, Cb, E_idx), _fast_rbf(C, O, E_idx),
            ], dim=-1)
            L_seq = E_idx.shape[1]
            offset = (residue_idx[:, :, None] - torch.gather(residue_idx.unsqueeze(1).expand(-1, L_seq, -1), 2, E_idx)).long()
            E_chains = (chain_encoding_all[:, :, None] == torch.gather(chain_encoding_all.unsqueeze(1).expand(-1, L_seq, -1), 2, E_idx)).long()
            mask_attend_enc = mask.unsqueeze(-1) * gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])

            chain_M_m = chain_M * mask
            decoding_order = torch.argsort((chain_M_m + 0.0001) * torch.abs(randn))
            rank = torch.argsort(decoding_order, dim=-1)
            rank_neighbors = torch.gather(rank.unsqueeze(1).expand(-1, L_seq, -1), 2, E_idx)
            mask_attend_dec = (rank.unsqueeze(-1) > rank_neighbors).float().unsqueeze(-1)
            mask_bw = mask_1D * mask_attend_dec
            mask_fw = mask_1D * (1.0 - mask_attend_dec)

            for mdl in models:
                E_pos = mdl.features.embeddings(offset, E_chains)
                E = mdl.features.norm_edges(mdl.features.edge_embedding(torch.cat((E_pos, RBF_all), -1)))
                h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]))
                h_E = mdl.W_e(E)
                for layer in mdl.encoder_layers:
                    h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend_enc)
                if want_emb:
                    hv = h_V[0].detach().cpu()
                    emb_sum = hv if emb_sum is None else (emb_sum + hv)

                h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
                h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

                h_EXV_u = mask_1D * h_EXV_encoder
                h_V_u = h_V
                for layer in mdl.decoder_layers:
                    h_V_u = layer(h_V_u, h_EXV_u, mask)
                lp_u = F.log_softmax(mdl.W_out(h_V_u), dim=-1)[0, :, reorder]

                h_S = mdl.W_s(S)
                h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
                h_EXV_c_fw = mask_fw * h_EXV_encoder
                h_V_c = h_V
                for layer in mdl.decoder_layers:
                    h_ESV = cat_neighbors_nodes(h_V_c, h_ES, E_idx)
                    h_ESV = mask_bw * h_ESV + h_EXV_c_fw
                    h_V_c = layer(h_V_c, h_ESV, mask)
                lp_c = F.log_softmax(mdl.W_out(h_V_c), dim=-1)[0, :, reorder]

                lp = 0.5 * (lp_u + lp_c)
                lp_sum = lp if lp_sum is None else (lp_sum + lp)
        lp_avg = (lp_sum / len(models)).numpy()
        ce = chain_encoding_all[0].numpy().astype(int)
        emb_avg = (emb_sum / len(models)).numpy() if want_emb else None
        return lp_avg, ce, emb_avg

    out: dict[str, dict] = {}
    for pk, pdb_path, has_l, need_ag in shard:
        cplx_chains = ["H", "L", "A"] if has_l else ["H", "A"]
        ab_chains = ["H", "L"] if has_l else ["H"]
        lp_c, ce_c, emb_c = _score_pdb_chains(pdb_path, cplx_chains, want_emb=True)
        lp_ab, ce_ab, _ = _score_pdb_chains(pdb_path, ab_chains, want_emb=False)

        cplx_lp_map: dict[str, np.ndarray] = {}
        cplx_emb_map: dict[str, np.ndarray] = {}
        unb_lp_map: dict[str, np.ndarray] = {}
        for idx1, cid in enumerate(sorted(cplx_chains), start=1):
            m = ce_c == idx1
            cplx_lp_map[cid] = lp_c[m].astype(np.float16)
            cplx_emb_map[cid] = emb_c[m].astype(np.float16)
        for idx1, cid in enumerate(sorted(ab_chains), start=1):
            m = ce_ab == idx1
            unb_lp_map[cid] = lp_ab[m].astype(np.float16)
        if need_ag:
            lp_ag, _, _ = _score_pdb_chains(pdb_path, ["A"], want_emb=False)
            unb_lp_map["A"] = lp_ag.astype(np.float16)
        else:
            unb_lp_map["A"] = cplx_lp_map["A"]

        out[pk] = {
            "mpnn_lp_cplx": cplx_lp_map,
            "mpnn_lp_unb": unb_lp_map,
            "mpnn_emb_cplx": cplx_emb_map,
        }
    return out


def _fast_if1_antifold_worker(args: tuple[list[tuple[str, str, bool, bool]], str]) -> dict[str, dict]:
    """Worker running `ESM-IF1` and `AntiFold` on a shard of `(pose_key, pdb_path, has_l, need_ag)` poses."""
    shard, scaf_dir_str = args
    import gc
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_k] = "2"
    import torch
    torch.set_num_threads(2)
    from antifold.antiscripts import (
        dataset_dataloader_to_predictions_list,
        df_logits_to_logprobs,
        get_dataset_dataloader,
        load_model,
        predictions_list_to_df_logits_list,
    )

    m_if1 = load_model("ESM-IF1")
    m_af = load_model()

    def _run_both(df_in: pd.DataFrame):
        ds, dl = get_dataset_dataloader(df_in, scaf_dir_str, batch_size=1, custom_chain_mode=True, num_threads=0)
        sp_if, _ = dataset_dataloader_to_predictions_list(m_if1, ds, dl, batch_size=1, extract_embeddings=False)
        dfl_if = predictions_list_to_df_logits_list(sp_if, ds, dl)
        sp_af, _ = dataset_dataloader_to_predictions_list(m_af, ds, dl, batch_size=1, extract_embeddings=False)
        dfl_af = predictions_list_to_df_logits_list(sp_af, ds, dl)
        return dfl_if, dfl_af

    out: dict[str, dict] = {}
    for pk, _, has_l, need_ag in shard:
        if has_l:
            row_cplx = {"pdb": pk, "Hchain": "H", "Lchain": "L", "chain3": "A"}
            row_ab = {"pdb": pk, "Hchain": "H", "Lchain": "L"}
        else:
            row_cplx = {"pdb": pk, "Hchain": "H", "Lchain": "A"}
            row_ab = {"pdb": pk, "Hchain": "H", "Lchain": np.nan}

        dfl_if_c, dfl_af_c = _run_both(pd.DataFrame([row_cplx]))
        lp_if_c = df_logits_to_logprobs(dfl_if_c[0])[AA_LIST].to_numpy(dtype=np.float32)
        lp_af_c = df_logits_to_logprobs(dfl_af_c[0])[AA_LIST].to_numpy(dtype=np.float32)
        chains_c = dfl_if_c[0]["pdb_chain"].to_numpy()

        dfl_if_ab, dfl_af_ab = _run_both(pd.DataFrame([row_ab]))
        lp_if_ab = df_logits_to_logprobs(dfl_if_ab[0])[AA_LIST].to_numpy(dtype=np.float32)
        lp_af_ab = df_logits_to_logprobs(dfl_af_ab[0])[AA_LIST].to_numpy(dtype=np.float32)

        n_ag = int((chains_c == "A").sum())
        if need_ag:
            row_ag = {"pdb": pk, "Hchain": "A", "Lchain": np.nan}
            dfl_if_ag, dfl_af_ag = _run_both(pd.DataFrame([row_ag]))
            lp_if_ag = df_logits_to_logprobs(dfl_if_ag[0])[AA_LIST].to_numpy(dtype=np.float32)
            lp_af_ag = df_logits_to_logprobs(dfl_af_ag[0])[AA_LIST].to_numpy(dtype=np.float32)
        else:
            lp_if_ag = lp_if_c[chains_c == "A"] if n_ag > 0 else np.zeros((0, 20), dtype=np.float32)
            lp_af_ag = lp_af_c[chains_c == "A"] if n_ag > 0 else np.zeros((0, 20), dtype=np.float32)

        lp_if_unb = np.concatenate([lp_if_ab, lp_if_ag], axis=0)
        lp_af_unb = np.concatenate([lp_af_ab, lp_af_ag], axis=0)

        def _split_by_chain(arr: np.ndarray) -> dict[str, np.ndarray]:
            res: dict[str, np.ndarray] = {}
            for cid in ("H", "L", "A"):
                m = chains_c == cid
                if m.any():
                    res[cid] = arr[m].astype(np.float16)
            return res

        out[pk] = {
            "if1_lp_cplx": _split_by_chain(lp_if_c),
            "if1_lp_unb": _split_by_chain(lp_if_unb),
            "if1_emb_cplx": {},
            "af_lp_cplx": _split_by_chain(lp_af_c),
            "af_lp_unb": _split_by_chain(lp_af_unb),
        }
        gc.collect()
    return out


def _contact_impact_scores(loc_df: pd.DataFrame, kd_lookup: pd.DataFrame, abag2sc: dict, scafs: dict) -> np.ndarray:
    n = len(loc_df)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        r = loc_df.iloc[i]
        a, b = kd_lookup.loc[r["AbAgID_1"]], kd_lookup.loc[r["AbAgID_2"]]
        sc = scafs.get(abag2sc.get(r["AbAgID_1"], ""))
        if sc is None:
            continue
        min_d = 50.0
        for col, dk in (("Ab_heavy_chain_seq", "min_dist_H"), ("Ab_light_chain_seq", "min_dist_L"), ("Ag_seq", "min_dist_A")):
            s1, s2 = str(a[col]), str(b[col])
            if len(s1) == len(s2) and s1 != s2:
                arr = sc[dk]
                for p, (c1, c2) in enumerate(zip(s1, s2)):
                    if c1 != c2 and p < len(arr):
                        min_d = min(min_d, float(arr[p]))
        out[i] = -min_d
    return out


def build_or_load_method_loaco_cache(
    method: str,
    root: Path,
    loaco_sids: list[str],
    ag_mut_sids: set[str],
    scafs_nat: dict,
    man_df: pd.DataFrame,
    dockq_cache: dict,
    mod09: Any,
    max_poses: int = 10,
) -> dict:
    """Build or load `data/structures/cofolded_cache_<method>.pt` containing 10-pose biophysics,
    10-pose `ProteinMPNN`, and key-pose `ESM-IF1` + `AntiFold` features across all 94 `LOACO-Local` scaffolds.
    """
    sdir = root / "data" / "structures"
    cache_out = sdir / f"cofolded_cache_{method}.pt"
    if cache_out.exists():
        saved = torch.load(cache_out, map_location="cpu", weights_only=False)
        if set(loaco_sids) <= set(saved.get("pose_records", {}).keys()) and "if1_done_v2" in saved:
            print(f"[{method}] Loaded cached 10-pose LOACO features ({len(saved['pose_records'])} scaffolds) from {cache_out}")
            return saved

    std_out = sdir / "cofolded_standardized" / method
    std_out.mkdir(parents=True, exist_ok=True)
    dq_recs = dockq_cache["records"]

    bio_tasks = []
    pose_meta: dict[str, list[dict]] = {}
    for sid in loaco_sids:
        r = man_df.loc[sid]
        seq_h, seq_l, seq_a = str(r["seq_H"]), str(r["seq_L"]), str(r["seq_A"])
        nat = scafs_nat[sid]
        n3h = str(nat.get("3di_H", "d" * len(seq_h)))
        n3l = str(nat.get("3di_L", "d" * len(seq_l))) if seq_l else ""
        n3a = str(nat.get("3di_A", "d" * len(seq_a)))
        d_poses = dq_recs[sid]["methods"][method][:max_poses]
        pose_meta[sid] = []
        for p in d_poses:
            k_idx = int(p["k"])
            pfile = sdir / "cofolded_structures" / method / sid / f"pose_{k_idx:02d}.cif"
            std_pdb = std_out / f"{sid}__p{k_idx:02d}.pdb"
            bio_tasks.append((sid, k_idx, str(pfile), str(std_pdb), seq_h, seq_l, seq_a, n3h, n3l, n3a))
            pose_meta[sid].append({
                "pose_idx": k_idx,
                "pdb_path": str(std_pdb),
                "iptm": float(p["iptm"]),
                "ab_ag_iptm": float(p["ab_ag_iptm"]),
                "ptm": float(p["ptm"]),
                "ranking_score": float(p["ranking_score"]),
                "dockq": {k: p[k] for k in ("DockQ", "fnat", "iRMSD", "LRMSD")} if "DockQ" in p else None,
            })

    t_b0 = time.time()
    print(f"[{method}] Standardizing & computing scaffold biophysics across {len(bio_tasks)} poses (24 workers)...", flush=True)
    with ProcessPoolExecutor(max_workers=24) as ex:
        bio_res = list(ex.map(_biophysics_pose_worker, bio_tasks))

    pose_records: dict[str, list[dict]] = {sid: pose_meta[sid] for sid in loaco_sids}
    for sid, k_idx, bio in bio_res:
        pose_records[sid][k_idx]["bio"] = bio

    print(f"[{method}] Biophysics complete in {time.time() - t_b0:.1f}s! Running ProteinMPNN on {len(bio_tasks)} poses...", flush=True)
    t_m0 = time.time()
    mpnn_jobs = []
    for sid in loaco_sids:
        has_l = bool(man_df.loc[sid, "seq_L"])
        need_ag = sid in ag_mut_sids
        for pr in pose_records[sid]:
            pk = f"{sid}__p{pr['pose_idx']:02d}"
            mpnn_jobs.append((pk, pr["pdb_path"], has_l, need_ag))

    n_w = 48
    mpnn_jobs.sort(key=lambda x: os.path.getsize(x[1]), reverse=True)
    mpnn_shards = [mpnn_jobs[w::n_w] for w in range(n_w) if mpnn_jobs[w::n_w]]
    mpnn_repo = str(Path.home() / ".cache" / "abrank" / "repos" / "ProteinMPNN")
    mpnn_merged: dict[str, dict] = {}
    with mp.get_context("fork").Pool(len(mpnn_shards)) as pool:
        for d_part in pool.map(_fast_protein_mpnn_worker, [(sh, mpnn_repo) for sh in mpnn_shards]):
            mpnn_merged.update(d_part)
    print(f"[{method}] ProteinMPNN complete in {time.time() - t_m0:.1f}s!", flush=True)

    # Identify key poses per scaffold (`pose_00`, `top1_iptm`, `top1_abag_iptm`, `best_dockq`) for ESM-IF1 & AntiFold
    if1_jobs = []
    key_pose_map: dict[str, int] = {}
    for sid in loaco_sids:
        plist = pose_records[sid]
        has_l = bool(man_df.loc[sid, "seq_L"])
        need_ag = sid in ag_mut_sids
        k0 = 0
        ki = int(np.argmax([p["iptm"] for p in plist]))
        ka = int(np.argmax([p["ab_ag_iptm"] for p in plist]))
        has_dq = plist[0]["dockq"] is not None
        kb = int(np.argmax([p["dockq"]["DockQ"] for p in plist])) if has_dq else ki
        key_pose_map[sid] = ki
        use_ks = sorted({ki, kb}) if has_dq else [ki]
        for k_u in use_ks:
            pk = f"{sid}__p{k_u:02d}"
            if1_jobs.append((pk, plist[k_u]["pdb_path"], has_l, need_ag))

    t_i0 = time.time()
    n_if1_w = 24
    if1_jobs.sort(key=lambda x: os.path.getsize(x[1]), reverse=True)
    print(f"[{method}] Running ESM-IF1 & AntiFold on {len(if1_jobs)} unique key poses across {n_if1_w}x2 workers...", flush=True)
    if1_shards = [if1_jobs[w::n_if1_w] for w in range(n_if1_w) if if1_jobs[w::n_if1_w]]
    if1_merged: dict[str, dict] = {}
    with mp.get_context("spawn").Pool(len(if1_shards)) as pool:
        for d_part in pool.map(_fast_if1_antifold_worker, [(sh, str(std_out)) for sh in if1_shards]):
            if1_merged.update(d_part)
    print(f"[{method}] ESM-IF1 & AntiFold complete in {time.time() - t_i0:.1f}s!", flush=True)

    inv_all: dict[str, dict] = {}
    for sid in loaco_sids:
        fb_pk = f"{sid}__p{key_pose_map[sid]:02d}"
        fb_if1 = if1_merged[fb_pk]
        for pr in pose_records[sid]:
            pk = f"{sid}__p{pr['pose_idx']:02d}"
            if_entry = if1_merged.get(pk, fb_if1)
            inv_all[pk] = {**if_entry, **mpnn_merged[pk]}

    payload = {
        "method": method,
        "if1_done_v2": True,
        "pose_records": pose_records,
        "inv_all": inv_all,
        "dockq_summary_all_340": dockq_cache["all_851_summary"]["methods"][method],
        "dockq_summary_loaco_47": dockq_cache["tier1_loaco_summary"]["methods"][method],
    }
    torch.save(payload, cache_out)
    print(f"[{method}] Saved cache -> {cache_out}", flush=True)
    return payload


_EVAL_CTX: dict[str, Any] = {}


def _slice_gb(gb_t: tuple, ix: np.ndarray) -> tuple:
    (a1, e1, m1), (a2, e2, m2) = gb_t
    return ((a1[ix], e1[ix], m1[ix]), (a2[ix], e2[ix], m2[ix]))


def _slim_tensors(sp: Any | None, sf: Any | None) -> None:
    if sf is not None:
        z_sf = np.empty((len(sf.is_cocrystal), 0), dtype=np.float32)
        for attr in (
            "saprot_ab_1", "saprot_ab_2", "saprot_ag_1", "saprot_ag_2",
            "mint_ab_1", "mint_ab_2", "mint_ag_1", "mint_ag_2",
            "if1_emb_1", "if1_emb_2", "mpnn_emb_1", "mpnn_emb_2",
        ):
            if hasattr(sf, attr):
                setattr(sf, attr, z_sf)
    if sp is not None:
        z_sp = np.empty((len(sp.y), 0), dtype=np.float32)
        for attr in ("esm2_ab_1", "esm2_ab_2", "esm2_ag_1", "esm2_ag_2"):
            if hasattr(sp, attr):
                setattr(sp, attr, z_sp)


def _train_loaco_cl(cl: str) -> tuple:
    t_c0 = time.time()
    torch.set_num_threads(1)
    loc_df = _EVAL_CTX["loc_df"]
    loc_c1 = _EVAL_CTX["loc_c1"]
    ab_lin_1 = _EVAL_CTX["ab_lin_1"]
    ab_lin_2 = _EVAL_CTX["ab_lin_2"]
    n_loc = _EVAL_CTX["n_loc"]
    sp_loc = _EVAL_CTX["sp_loc"]
    sf_nat = _EVAL_CTX["sf_nat"]
    gb_nat = _EVAL_CTX["gb_nat"]

    te_idx = np.flatnonzero(loc_c1 == cl)
    te_lins = set(ab_lin_1[te_idx]) | set(ab_lin_2[te_idx])
    pool = np.array([i for i in range(n_loc) if loc_c1[i] != cl and ab_lin_1[i] not in te_lins and ab_lin_2[i] not in te_lins], dtype=int)
    rem_cls = sorted(set(loc_c1[pool]))
    val_cls = set(rem_cls[1::5])
    val_lins = set(ab_lin_1[pool[np.isin(loc_c1[pool], list(val_cls))]])
    is_val = np.array([loc_c1[i] in val_cls for i in pool], dtype=bool)
    is_tr = np.array([loc_c1[i] not in val_cls and ab_lin_1[i] not in val_lins and ab_lin_2[i] not in val_lins for i in pool], dtype=bool)
    tr_i, val_i = pool[is_tr], pool[is_val]
    val_df_sub = loc_df.iloc[val_i].reset_index(drop=True)
    w_tr = compute_stratum_weights(loc_df.iloc[tr_i].reset_index(drop=True), sp_loc.is_ag_mut[tr_i])

    tr_sf = sf_nat.subset(tr_i)
    val_sf = sf_nat.subset(val_i)
    tr_sp = sp_loc.subset(tr_i)
    val_sp = sp_loc.subset(val_i)
    tr_gb = _slice_gb(gb_nat, tr_i)
    val_gb = _slice_gb(gb_nat, val_i)

    sub_idx = select_stratified_local_indices(
        tr_sp.esmc_ag_1, np.ones(len(tr_sp.y), dtype=bool), tr_sp.labels, max_pairs=3200, seed=42
    )
    tr_sf_sub = tr_sf.subset(sub_idx)
    tr_sp_sub = tr_sp.subset(sub_idx)
    tr_gb_sub = _slice_gb(tr_gb, sub_idx)
    w_tr_sub = w_tr[sub_idx]

    m_anti = ANTIPASTI3StateTrainer(seed=42, epochs=9).fit(
        tr_sf_sub, tr_sp_sub, val_sf, val_sp, val_df_sub, w_tr_sub
    )
    m_gb = GearBind3StateTrainer(seed=42, epochs=9).fit(
        tr_sf_sub, tr_sp_sub, val_sf, val_sp, val_df_sub, w_tr_sub,
        precomp_tr_g=tr_gb_sub, precomp_val_g=val_gb,
    )
    if cl == "AgClust_001":
        print(f"  [AgClust_001] ANTIPASTI + GearBind done in {time.time() - t_c0:.1f}s.", flush=True)
    m_l1d = LogiCA3StateTrainer(seed=42, epochs=12).fit(
        tr_sp, val_sp, val_df_sub, w_tr
    )
    d1d, i1d = m_l1d.predict_3state(sp_loc.subset(te_idx))
    del tr_sf, val_sf, tr_sp, val_sp, tr_gb, val_gb, tr_sf_sub, tr_sp_sub, tr_gb_sub
    gc.collect()
    return cl, te_idx, m_anti, m_gb, d1d, i1d, time.time() - t_c0


def _score_single_pose(k_i: int) -> tuple:
    torch.set_num_threads(1)
    loc_df = _EVAL_CTX["loc_df"]
    kd_lookup = _EVAL_CTX["kd_lookup"]
    abag2sc = _EVAL_CTX["abag2sc"]
    scafs_nat = _EVAL_CTX["scafs_nat"]
    inv_base = _EVAL_CTX["inv_base"]
    n_loc = _EVAL_CTX["n_loc"]
    sp_loc = _EVAL_CTX["sp_loc"]
    trained_by_cl = _EVAL_CTX["trained_by_cl"]
    pose_records = _EVAL_CTX["pose_records"]
    inv_all = _EVAL_CTX["inv_all"]

    sc_k = dict(scafs_nat)
    inv_k = dict(inv_base)
    for sid, plist in pose_records.items():
        use_p = plist[min(k_i, len(plist) - 1)]
        pk = f"{sid}__p{use_p['pose_idx']:02d}"
        sc_k[sid] = use_p["bio"]
        inv_k[sid] = inv_all[pk]
    sf_k = assemble_structure_pair_features(
        loc_df, kd_lookup, {"abag_to_scaffold": abag2sc, "scaffolds": sc_k}, inv_k, None, None, True
    )
    _slim_tensors(None, sf_k)
    gb_k = (
        GearBind3StateTrainer._precompute_graphs(sf_k.patch_ca_1, sf_k.patch_cb_1, sf_k.patch_chain_1, sf_k.patch_seqpos_1),
        GearBind3StateTrainer._precompute_graphs(sf_k.patch_ca_2, sf_k.patch_cb_2, sf_k.patch_chain_2, sf_k.patch_seqpos_2),
    )
    loc_dir_k = {mk: np.zeros(n_loc) for mk in ("prodigy_sasa", "protein_mpnn", "esm_if1", "antifold", "antipasti", "gearbind")}
    loc_imp_k = {mk: np.zeros(n_loc) for mk in ("wt_contact", "prodigy_sasa", "protein_mpnn", "esm_if1", "antifold", "antipasti", "gearbind")}
    loc_imp_k["wt_contact"] = _contact_impact_scores(loc_df, kd_lookup, abag2sc, sc_k)
    for zk in ("prodigy_sasa", "protein_mpnn", "esm_if1", "antifold"):
        loc_dir_k[zk] = sf_k.zs_dir[zk]
        loc_imp_k[zk] = sf_k.zs_imp[zk]

    for _cl, (te_idx, m_anti, m_gb) in trained_by_cl.items():
        sf_te = sf_k.subset(te_idx)
        sp_te = sp_loc.subset(te_idx)
        gb_te = _slice_gb(gb_k, te_idx)
        da, ia = m_anti.predict_3state(sf_te, sp_te)
        loc_dir_k["antipasti"][te_idx], loc_imp_k["antipasti"][te_idx] = da, ia
        dg, ig = m_gb.predict_3state(sf_te, _precomp_g=gb_te)
        loc_dir_k["gearbind"][te_idx], loc_imp_k["gearbind"][te_idx] = dg, ig

    iptm_k = np.full(n_loc, 0.5)
    abag_k = np.full(n_loc, 0.5)
    rscore_k = np.full(n_loc, 0.5)
    dockq_k = np.full(n_loc, 0.0)
    for i_row, a1 in enumerate(loc_df["AbAgID_1"]):
        sid1 = abag2sc.get(a1, "")
        if sid1 in pose_records:
            pr = pose_records[sid1][min(k_i, len(pose_records[sid1]) - 1)]
            iptm_k[i_row] = pr["iptm"]
            abag_k[i_row] = pr["ab_ag_iptm"]
            rscore_k[i_row] = pr["ranking_score"]
            dockq_k[i_row] = pr["dockq"]["DockQ"] if pr["dockq"] is not None else pr["iptm"]
    del sf_k, gb_k
    gc.collect()
    return k_i, loc_dir_k, loc_imp_k, iptm_k, abag_k, rscore_k, dockq_k


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest multi-pose co-folded complexes and evaluate DockQ + LOACO AUCs.")
    ap.add_argument("--method", default="all", choices=["af3", "boltz2", "chai1", "protenix", "all"])
    ap.add_argument("--iptm-gate", type=float, default=0.75, help="ipTM confidence gate threshold (default: 0.75).")
    ap.add_argument("--max-poses", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    root = get_repo_root()
    sdir = root / "data" / "structures"
    edir = root / "data" / "embeddings"
    bdir = root / "data" / "benchmark"
    res_dir = root / "results"

    cc_nat = torch.load(sdir / "complex_cache.pt", map_location="cpu", weights_only=False)
    scafs_nat = cc_nat["scaffolds"]
    abag2sc = cc_nat["abag_to_scaffold"]
    man_df = pd.read_csv(root / "metadata" / "cofolding_manifest.tsv", sep="\t", keep_default_na=False).set_index("scaffold_id")

    # Load or compute the 34,040-pose DockQ & Confidence cache
    dq_cache_path = sdir / "cofolded_dockq_and_confidence_cache.pt"
    if not dq_cache_path.exists():
        raise FileNotFoundError(f"Missing {dq_cache_path}. Run eval_all_dockq_and_conf.py first.")
    dockq_cache = torch.load(dq_cache_path, map_location="cpu", weights_only=False)

    run_methods = METHODS if args.method == "all" else [args.method]
    mod09 = None

    kd_lookup = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False).drop_duplicates("AbAgID").set_index("AbAgID")
    kd_lookup["Ab_light_chain_seq"] = kd_lookup["Ab_light_chain_seq"].fillna("")
    loc_df = pd.concat([pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}.tsv", sep="\t") for s in ("train", "val", "test")], ignore_index=True)
    loc_df = loc_df[loc_df["Category"].isin(["L1", "L2", "L3"])].reset_index(drop=True)
    n_loc = len(loc_df)
    loc_c1 = loc_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
    ab_lin_1 = loc_df["AbAgID_1"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()
    ab_lin_2 = loc_df["AbAgID_2"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()

    with open(res_dir / "loaco_evaluation.json") as f:
        loaco_json = json.load(f)
    loc_clusters_22 = [c for c in sorted(loaco_json["cluster_metadata"].keys()) if int((loc_c1 == c).sum()) >= 5]
    cocrys_clusters_19 = [c for c in loc_clusters_22 if loaco_json["cluster_metadata"][c]["cocrystal_frac"] >= 0.5]

    loaco_sids = sorted({abag2sc[a] for a in set(loc_df["AbAgID_1"]).union(loc_df["AbAgID_2"]) if a in abag2sc})
    ag_mut_sids = set()
    for i in range(n_loc):
        r = loc_df.iloc[i]
        a1, a2 = kd_lookup.loc[r["AbAgID_1"]], kd_lookup.loc[r["AbAgID_2"]]
        if str(a1["Ag_seq"]) != str(a2["Ag_seq"]):
            ag_mut_sids.add(abag2sc[r["AbAgID_1"]])
            ag_mut_sids.add(abag2sc[r["AbAgID_2"]])

    # Load or build base sequence & native 3D embeddings cache (without running PyTorch ops in parent before fork!)
    base_cache_path = sdir / "cofolded_base_loaco_tensors.pt"
    if base_cache_path.exists():
        print(f"Loading cached base LOACO tensors from {base_cache_path}...", flush=True)
        b_cache = torch.load(base_cache_path, map_location="cpu", weights_only=False)
        inv_base = b_cache["inv_base"]
        sp_loc = b_cache["sp_loc"]
        sf_nat = b_cache["sf_nat"]
        gb_nat = b_cache["gb_nat"]
    else:
        inv_nat = torch.load(edir / "inverse_folding_scaffolds.pt", map_location="cpu", weights_only=False)
        inv_base = inv_nat["scaffolds"] if "scaffolds" in inv_nat else inv_nat
        mint_nat = torch.load(edir / "mint_650m.pt", map_location="cpu", weights_only=False)
        esmc_nat = torch.load(edir / "esmc_300m.pt", map_location="cpu", weights_only=False)
        abl_nat = torch.load(edir / "ablang2_paired.pt", map_location="cpu", weights_only=False)
        esm2_nat = torch.load(edir / "esm2_150m.pt", map_location="cpu", weights_only=False)

        sp_loc = assemble_split_tensors(loc_df, kd_lookup, esmc_nat, abl_nat, esm2_nat)
        sf_nat = assemble_structure_pair_features(loc_df, kd_lookup, cc_nat, inv_base, None, mint_nat, True)
        _slim_tensors(sp_loc, sf_nat)
        del mint_nat, esmc_nat, abl_nat, esm2_nat
        gc.collect()
        gb_nat = (
            GearBind3StateTrainer._precompute_graphs(sf_nat.patch_ca_1, sf_nat.patch_cb_1, sf_nat.patch_chain_1, sf_nat.patch_seqpos_1),
            GearBind3StateTrainer._precompute_graphs(sf_nat.patch_ca_2, sf_nat.patch_cb_2, sf_nat.patch_chain_2, sf_nat.patch_seqpos_2),
        )
        torch.save({
            "loc_df": loc_df, "kd_lookup": kd_lookup, "abag2sc": abag2sc, "scafs_nat": scafs_nat, "inv_base": inv_base,
            "loc_c1": loc_c1, "ab_lin_1": ab_lin_1, "ab_lin_2": ab_lin_2, "n_loc": n_loc, "sp_loc": sp_loc, "sf_nat": sf_nat, "gb_nat": gb_nat,
        }, base_cache_path)

    _EVAL_CTX.update({
        "loc_df": loc_df,
        "kd_lookup": kd_lookup,
        "abag2sc": abag2sc,
        "scafs_nat": scafs_nat,
        "inv_base": inv_base,
        "loc_c1": loc_c1,
        "ab_lin_1": ab_lin_1,
        "ab_lin_2": ab_lin_2,
        "n_loc": n_loc,
        "sp_loc": sp_loc,
        "sf_nat": sf_nat,
        "gb_nat": gb_nat,
    })

    # Train supervised models once per LOACO cluster on clean disjoint train folds (plus 1D LogiCA for confidence-gated fallback)
    torch.set_num_threads(1)
    sup_cache_path = sdir / "cofolded_loaco_supervised_models.pt"
    if sup_cache_path.exists():
        print(f"Loading cached 24-cluster supervised LOACO models from {sup_cache_path}...", flush=True)
        sup_c = torch.load(sup_cache_path, map_location="cpu", weights_only=False)
        trained_by_cl = sup_c["trained_by_cl"]
        logica_1d_dir = sup_c["logica_1d_dir"]
        logica_1d_imp = sup_c["logica_1d_imp"]
    else:
        print(f"Training supervised 3D & 1D models across {len(loc_clusters_22)} LOACO clusters (24 workers)...", flush=True)
        t_cl0 = time.time()
        trained_by_cl: dict[str, tuple] = {}
        logica_1d_dir = np.zeros(n_loc)
        logica_1d_imp = np.zeros(n_loc)
        done_cl = 0
        with mp.get_context("fork").Pool(processes=min(22, len(loc_clusters_22))) as pool:
            for cl, te_idx, m_anti, m_gb, d1d, i1d, dt_c in pool.imap_unordered(_train_loaco_cl, loc_clusters_22):
                done_cl += 1
                logica_1d_dir[te_idx] = d1d
                logica_1d_imp[te_idx] = i1d
                trained_by_cl[cl] = (te_idx, m_anti, m_gb)
                print(f"  [{done_cl:2d}/{len(loc_clusters_22)}] {cl} trained in {dt_c:.1f}s", flush=True)
        torch.save({
            "trained_by_cl": trained_by_cl,
            "logica_1d_dir": logica_1d_dir,
            "logica_1d_imp": logica_1d_imp,
        }, sup_cache_path)
        print(f"Supervised 22-cluster training complete in {time.time() - t_cl0:.1f}s! Saved -> {sup_cache_path}", flush=True)
    _EVAL_CTX["trained_by_cl"] = trained_by_cl

    lbls = loc_df["label"].to_numpy(dtype=int)
    bds = loc_df["binding_diff"].to_numpy(dtype=float)
    cols = np.arange(n_loc)

    def _eval_clusters(dv: np.ndarray, iv: np.ndarray, cl_list: list[str], n_boot: int = 500) -> dict:
        per_cl = {cl: compute_3state_metrics(lbls[loc_c1 == cl], bds[loc_c1 == cl], dv[loc_c1 == cl], iv[loc_c1 == cl]) for cl in cl_list}
        return compute_cluster_bootstrap_summary(per_cl, min_class_pairs=2, n_bootstrap=n_boot, seed=42)

    all_methods_out: dict[str, dict] = {}
    pose_dir_by_m: dict[str, dict[str, np.ndarray]] = {}
    pose_imp_by_m: dict[str, dict[str, np.ndarray]] = {}
    pose_iptm_by_m: dict[str, np.ndarray] = {}
    pose_abag_by_m: dict[str, np.ndarray] = {}
    pose_dockq_by_m: dict[str, np.ndarray] = {}

    for m in run_methods:
        if mod09 is None and not (sdir / f"cofolded_cache_{m}.pt").exists():
            spec09 = importlib.util.spec_from_file_location("step09", root / "scripts" / "09_extract_structure_and_multichain_features.py")
            assert spec09 is not None and spec09.loader is not None
            mod09 = importlib.util.module_from_spec(spec09)
            spec09.loader.exec_module(mod09)
        mc = build_or_load_method_loaco_cache(
            m, root, loaco_sids, ag_mut_sids, scafs_nat, man_df, dockq_cache, mod09, max_poses=args.max_poses
        )
        print(f"[{m}] Assembling 10-pose structure features & scoring 8 3D models (10 workers)...", flush=True)
        t_m0 = time.time()
        pose_records = mc["pose_records"]
        inv_all = mc["inv_all"]
        max_k = args.max_poses
        _EVAL_CTX["pose_records"] = pose_records
        _EVAL_CTX["inv_all"] = inv_all

        p_dir = {mk: np.zeros((max_k, n_loc)) for mk in ("prodigy_sasa", "protein_mpnn", "esm_if1", "antifold", "antipasti", "gearbind")}
        p_imp = {mk: np.zeros((max_k, n_loc)) for mk in ("wt_contact", "prodigy_sasa", "protein_mpnn", "esm_if1", "antifold", "antipasti", "gearbind")}
        p_iptm = np.full((max_k, n_loc), 0.5)
        p_abag = np.full((max_k, n_loc), 0.5)
        p_rscore = np.full((max_k, n_loc), 0.5)
        p_dockq = np.full((max_k, n_loc), 0.0)

        with mp.get_context("fork").Pool(processes=max_k) as pool:
            for k_i, loc_dir_k, loc_imp_k, iptm_k, abag_k, rscore_k, dockq_k in pool.imap_unordered(_score_single_pose, list(range(max_k))):
                for mk, arr in loc_dir_k.items():
                    p_dir[mk][k_i] = arr
                for mk, arr in loc_imp_k.items():
                    p_imp[mk][k_i] = arr
                p_iptm[k_i] = iptm_k
                p_abag[k_i] = abag_k
                p_rscore[k_i] = rscore_k
                p_dockq[k_i] = dockq_k
        print(f"[{m}] 10-pose scoring complete in {time.time() - t_m0:.1f}s!", flush=True)

        pose_dir_by_m[m] = p_dir
        pose_imp_by_m[m] = p_imp
        pose_iptm_by_m[m] = p_iptm
        pose_abag_by_m[m] = p_abag
        pose_dockq_by_m[m] = p_dockq

        idx_iptm = np.argmax(p_iptm, axis=0)
        idx_abag = np.argmax(p_abag, axis=0)
        idx_rscore = np.argmax(p_rscore, axis=0)
        idx_oracle = np.argmax(p_dockq, axis=0)

        w_soft = np.exp(5.0 * np.maximum(p_abag, 0.0))
        w_soft /= w_soft.sum(axis=0, keepdims=True)
        hi_conf = p_abag.max(axis=0) >= args.iptm_gate

        def _build_strats(d_mat: np.ndarray, i_mat: np.ndarray, fb_d: np.ndarray, fb_i: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            d_s = (w_soft * d_mat).sum(axis=0)
            i_s = (w_soft * i_mat).sum(axis=0)
            return {
                "pose0_single_seed": (d_mat[0], i_mat[0]),
                "top1_iptm_pose": (d_mat[idx_iptm, cols], i_mat[idx_iptm, cols]),
                "top1_abag_iptm_pose": (d_mat[idx_abag, cols], i_mat[idx_abag, cols]),
                "top1_rscore_pose": (d_mat[idx_rscore, cols], i_mat[idx_rscore, cols]),
                "uniform_mean_ensemble": (d_mat.mean(axis=0), i_mat.mean(axis=0)),
                "iptm_softmax_ensemble": (d_s, i_s),
                "iptm_confidence_gated": (np.where(hi_conf, d_s, fb_d), np.where(hi_conf, i_s, fb_i)),
                "best_dockq_oracle_pose": (d_mat[idx_oracle, cols], i_mat[idx_oracle, cols]),
            }

        mpnn_strats = _build_strats(p_dir["protein_mpnn"], p_imp["protein_mpnn"], sf_nat.zs_dir["mint_zs"], sf_nat.zs_imp["mint_zs"])
        gb_strats = _build_strats(p_dir["gearbind"], p_imp["gearbind"], logica_1d_dir, logica_1d_imp)

        strat_22_mpnn = {sn: _eval_clusters(dv, iv, loc_clusters_22) for sn, (dv, iv) in mpnn_strats.items()}
        strat_18_mpnn = {sn: _eval_clusters(dv, iv, cocrys_clusters_19) for sn, (dv, iv) in mpnn_strats.items()}
        strat_22_gb = {sn: _eval_clusters(dv, iv, loc_clusters_22) for sn, (dv, iv) in gb_strats.items()}
        strat_18_gb = {sn: _eval_clusters(dv, iv, cocrys_clusters_19) for sn, (dv, iv) in gb_strats.items()}

        # Also compute Expected Single Pose (average of individual pose k=0..9 AUCs)
        for tgt_d, d_mat, i_mat, cl_l in [
            (strat_22_mpnn, p_dir["protein_mpnn"], p_imp["protein_mpnn"], loc_clusters_22),
            (strat_18_mpnn, p_dir["protein_mpnn"], p_imp["protein_mpnn"], cocrys_clusters_19),
            (strat_22_gb, p_dir["gearbind"], p_imp["gearbind"], loc_clusters_22),
            (strat_18_gb, p_dir["gearbind"], p_imp["gearbind"], cocrys_clusters_19),
        ]:
            k_evals = [_eval_clusters(d_mat[k], i_mat[k], cl_l, n_boot=10) for k in range(max_k)]
            tgt_d["expected_single_pose"] = {
                metric: {
                    "mean": round(float(np.mean([ke[metric]["mean"] for ke in k_evals])), 4),
                    "cluster_ci95": tgt_d["pose0_single_seed"][metric]["cluster_ci95"],
                }
                for metric in ("sign", "impact", "ordinal")
            }

        # Evaluate all 7 3D models at `top1_iptm_pose`, `iptm_softmax_ensemble`, and `best_dockq_oracle_pose`
        # on both 18-cluster Co-Crystal Ablation and 22-cluster LOACO-Local!
        model_names_map = [
            ("WT 3D Contact (-min Cβ)", None, "wt_contact"),
            ("PRODIGY+SASA (3D ZS)", "prodigy_sasa", "prodigy_sasa"),
            ("ProteinMPNN (3D ZS)", "protein_mpnn", "protein_mpnn"),
            ("ESM-IF1 (3D ZS)", "esm_if1", "esm_if1"),
            ("AntiFold (3D Ab-Ag ZS)", "antifold", "antifold"),
            ("ANTIPASTI (3D CNN)", "antipasti", "antipasti"),
            ("GearBind (3D GNN)", "gearbind", "gearbind"),
        ]
        models_18_top1, models_18_ens, models_18_oracle = {}, {}, {}
        models_22_top1, models_22_ens = {}, {}
        for mlabel, dk, ik in model_names_map:
            d_m = np.zeros((max_k, n_loc)) if dk is None else p_dir[dk]
            i_m = p_imp[ik]
            d_top1, i_top1 = d_m[idx_iptm, cols], i_m[idx_iptm, cols]
            d_ens, i_ens = (w_soft * d_m).sum(axis=0), (w_soft * i_m).sum(axis=0)
            d_ora, i_ora = d_m[idx_oracle, cols], i_m[idx_oracle, cols]
            models_18_top1[mlabel] = _eval_clusters(d_top1, i_top1, cocrys_clusters_19)
            models_18_ens[mlabel] = _eval_clusters(d_ens, i_ens, cocrys_clusters_19)
            models_18_oracle[mlabel] = _eval_clusters(d_ora, i_ora, cocrys_clusters_19)
            models_22_top1[mlabel] = _eval_clusters(d_top1, i_top1, loc_clusters_22)
            models_22_ens[mlabel] = _eval_clusters(d_ens, i_ens, loc_clusters_22)

        m_payload = {
            "method": m,
            "dockq_vs_native_all_340": dockq_cache["all_851_summary"]["methods"][m],
            "dockq_vs_native_loaco_47": dockq_cache["tier1_loaco_summary"]["methods"][m],
            "loaco_22cl_proteinmpnn_strategies": strat_22_mpnn,
            "loaco_18cl_proteinmpnn_strategies": strat_18_mpnn,
            "loaco_22cl_gearbind_strategies": strat_22_gb,
            "loaco_18cl_gearbind_strategies": strat_18_gb,
            "models_18cl_cocrystal_top1_iptm": models_18_top1,
            "models_18cl_cocrystal_softmax_ensemble": models_18_ens,
            "models_18cl_cocrystal_best10_oracle": models_18_oracle,
            "models_22cl_loaco_top1_iptm": models_22_top1,
            "models_22cl_loaco_softmax_ensemble": models_22_ens,
        }
        all_methods_out[m] = m_payload
        out_json = res_dir / f"cofolding_{m}_evaluation.json"
        out_json.write_text(json.dumps(m_payload, indent=2) + "\n")

        dq340 = m_payload["dockq_vs_native_all_340"]
        dq47 = m_payload["dockq_vs_native_loaco_47"]
        md = [
            f"# Multi-Pose Co-Folding Benchmark Evaluation (`{m.upper()}`)",
            "",
            f"- **Ingested Scaffolds**: `851` WT scaffolds (`10` poses/scaffold = `8,510` poses)",
            f"- **All Solved WT Co-Crystals (`n=340`)**: Pose-0 = `{dq340['pose0_mean_DockQ']:.3f}`, Expected 1-Pose = `{dq340['expected_single_pose_DockQ']:.3f}`, Top-`ipTM` = **`{dq340['top1_iptm_mean_DockQ']:.3f}`** (`{dq340['top1_acceptable_pct_ge_023']}% >= 0.23`, `{dq340['top1_medium_pct_ge_049']}% >= 0.49`, `{dq340['top1_high_pct_ge_080']}% >= 0.80`), Best-of-10 Oracle = **`{dq340['best_of_10_mean_DockQ']:.3f}`** (`{dq340['best10_acceptable_pct_ge_023']}% >= 0.23`), $\\rho(\\text{{AbAg-ipTM}}, \\text{{DockQ}}) = {dq340['spearman_abag_iptm_vs_dockq']:+.3f}$",
            f"- **LOACO-Local Solved WT Co-Crystals (`n=47`)**: Pose-0 = `{dq47['pose0_mean_DockQ']:.3f}`, Expected 1-Pose = `{dq47['expected_single_pose_DockQ']:.3f}`, Top-`ipTM` = **`{dq47['top1_iptm_mean_DockQ']:.3f}`** (`{dq47['top1_acceptable_pct_ge_023']}% >= 0.23`, `{dq47['top1_medium_pct_ge_049']}% >= 0.49`), Best-of-10 Oracle = **`{dq47['best_of_10_mean_DockQ']:.3f}`**",
            "",
            "## 1. Multi-Pose Strategy Comparison (`18`-Cluster Co-Crystal Ablation vs. `22`-Cluster Full `LOACO-Local`)",
            "",
            "| Multi-Pose Strategy | `ProteinMPNN` 18-Cl (Sign / Imp / **Ord**) | `GearBind` 18-Cl (Sign / Imp / **Ord**) | `ProteinMPNN` 22-Cl (**Ord**) | `GearBind` 22-Cl (**Ord**) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for sname in ("pose0_single_seed", "expected_single_pose", "top1_iptm_pose", "top1_abag_iptm_pose", "uniform_mean_ensemble", "iptm_softmax_ensemble", "iptm_confidence_gated", "best_dockq_oracle_pose"):
            mp18 = strat_18_mpnn[sname]
            gb18 = strat_18_gb[sname]
            mp22 = strat_22_mpnn[sname]
            gb22 = strat_22_gb[sname]
            md.append(
                f"| **{sname}** | `{mp18['sign']['mean']:.3f}` / `{mp18['impact']['mean']:.3f}` / **`{mp18['ordinal']['mean']:.3f}`** | "
                f"`{gb18['sign']['mean']:.3f}` / `{gb18['impact']['mean']:.3f}` / **`{gb18['ordinal']['mean']:.3f}`** | "
                f"**`{mp22['ordinal']['mean']:.3f}`** | **`{gb22['ordinal']['mean']:.3f}`** |"
            )
        md.extend([
            "",
            "## 2. All 7 3D Models on `18`-Cluster Controlled Co-Crystal Ablation (`47` WT Scaffolds)",
            "",
            "| 3D Baseline Model | Top-1 `ipTM` Pose (Sign / Imp / **Ord**) | 10-Pose `ipTM`-Softmax Ensemble (Sign / Imp / **Ord**) | Best-of-10 `DockQ` Oracle (Sign / Imp / **Ord**) |",
            "| :--- | :---: | :---: | :---: |",
        ])
        for mlabel, _, _ in model_names_map:
            t1m = models_18_top1[mlabel]
            enm = models_18_ens[mlabel]
            orm = models_18_oracle[mlabel]
            md.append(
                f"| **{mlabel}** | `{t1m['sign']['mean']:.3f}` / `{t1m['impact']['mean']:.3f}` / **`{t1m['ordinal']['mean']:.3f}`** | "
                f"`{enm['sign']['mean']:.3f}` / `{enm['impact']['mean']:.3f}` / **`{enm['ordinal']['mean']:.3f}`** | "
                f"`{orm['sign']['mean']:.3f}` / `{orm['impact']['mean']:.3f}` / **`{orm['ordinal']['mean']:.3f}`** |"
            )
        out_md = res_dir / f"COFOLDING_{m.upper()}_EVALUATION.md"
        out_md.write_text("\n".join(md) + "\n")
        print(f"[{m}] Wrote {out_json} and {out_md}", flush=True)

    if len(run_methods) == 4:
        # Evaluate 40-pose Multi-Method Pool (`af3 + boltz2 + chai1 + protenix`)
        all40_mpnn_d = np.concatenate([pose_dir_by_m[m]["protein_mpnn"] for m in METHODS], axis=0)
        all40_mpnn_i = np.concatenate([pose_imp_by_m[m]["protein_mpnn"] for m in METHODS], axis=0)
        all40_gb_d = np.concatenate([pose_dir_by_m[m]["gearbind"] for m in METHODS], axis=0)
        all40_gb_i = np.concatenate([pose_imp_by_m[m]["gearbind"] for m in METHODS], axis=0)
        all40_iptm = np.concatenate([pose_iptm_by_m[m] for m in METHODS], axis=0)
        all40_abag = np.concatenate([pose_abag_by_m[m] for m in METHODS], axis=0)
        all40_dockq = np.concatenate([pose_dockq_by_m[m] for m in METHODS], axis=0)

        idx40_iptm = np.argmax(all40_iptm, axis=0)
        idx40_abag = np.argmax(all40_abag, axis=0)
        idx40_oracle = np.argmax(all40_dockq, axis=0)
        w40 = np.exp(5.0 * np.maximum(all40_abag, 0.0))
        w40 /= w40.sum(axis=0, keepdims=True)
        hi40 = all40_abag.max(axis=0) >= args.iptm_gate

        mm_models_18_top1, mm_models_18_ens, mm_models_18_oracle = {}, {}, {}
        mm_models_22_top1, mm_models_22_ens = {}, {}
        for mlabel, dk, ik in model_names_map:
            d40 = np.zeros((40, n_loc)) if dk is None else np.concatenate([pose_dir_by_m[m][dk] for m in METHODS], axis=0)
            i40 = np.concatenate([pose_imp_by_m[m][ik] for m in METHODS], axis=0)
            dt1, it1 = d40[idx40_iptm, cols], i40[idx40_iptm, cols]
            den, ien = (w40 * d40).sum(0), (w40 * i40).sum(0)
            dor, ior = d40[idx40_oracle, cols], i40[idx40_oracle, cols]
            mm_models_18_top1[mlabel] = _eval_clusters(dt1, it1, cocrys_clusters_19)
            mm_models_18_ens[mlabel] = _eval_clusters(den, ien, cocrys_clusters_19)
            mm_models_18_oracle[mlabel] = _eval_clusters(dor, ior, cocrys_clusters_19)
            mm_models_22_top1[mlabel] = _eval_clusters(dt1, it1, loc_clusters_22)
            mm_models_22_ens[mlabel] = _eval_clusters(den, ien, loc_clusters_22)

        mm_summary = {
            "dockq_vs_native_all_340": dockq_cache["all_851_summary"]["methods"]["multi_method_40pose"],
            "dockq_vs_native_loaco_47": dockq_cache["tier1_loaco_summary"]["methods"]["multi_method_40pose"],
            "proteinmpnn_18cl_top1_iptm": _eval_clusters(all40_mpnn_d[idx40_iptm, cols], all40_mpnn_i[idx40_iptm, cols], cocrys_clusters_19),
            "proteinmpnn_18cl_top1_abag_iptm": _eval_clusters(all40_mpnn_d[idx40_abag, cols], all40_mpnn_i[idx40_abag, cols], cocrys_clusters_19),
            "proteinmpnn_18cl_softmax40": _eval_clusters((w40 * all40_mpnn_d).sum(0), (w40 * all40_mpnn_i).sum(0), cocrys_clusters_19),
            "proteinmpnn_18cl_gated40": _eval_clusters(
                np.where(hi40, (w40 * all40_mpnn_d).sum(0), sf_nat.zs_dir["mint_zs"]),
                np.where(hi40, (w40 * all40_mpnn_i).sum(0), sf_nat.zs_imp["mint_zs"]),
                cocrys_clusters_19,
            ),
            "proteinmpnn_18cl_best40_oracle": _eval_clusters(all40_mpnn_d[idx40_oracle, cols], all40_mpnn_i[idx40_oracle, cols], cocrys_clusters_19),
            "gearbind_18cl_top1_iptm": _eval_clusters(all40_gb_d[idx40_iptm, cols], all40_gb_i[idx40_iptm, cols], cocrys_clusters_19),
            "gearbind_18cl_top1_abag_iptm": _eval_clusters(all40_gb_d[idx40_abag, cols], all40_gb_i[idx40_abag, cols], cocrys_clusters_19),
            "gearbind_18cl_softmax40": _eval_clusters((w40 * all40_gb_d).sum(0), (w40 * all40_gb_i).sum(0), cocrys_clusters_19),
            "gearbind_18cl_gated40": _eval_clusters(
                np.where(hi40, (w40 * all40_gb_d).sum(0), logica_1d_dir),
                np.where(hi40, (w40 * all40_gb_i).sum(0), logica_1d_imp),
                cocrys_clusters_19,
            ),
            "gearbind_18cl_best40_oracle": _eval_clusters(all40_gb_d[idx40_oracle, cols], all40_gb_i[idx40_oracle, cols], cocrys_clusters_19),
            "proteinmpnn_22cl_top1_iptm": _eval_clusters(all40_mpnn_d[idx40_iptm, cols], all40_mpnn_i[idx40_iptm, cols], loc_clusters_22),
            "proteinmpnn_22cl_softmax40": _eval_clusters((w40 * all40_mpnn_d).sum(0), (w40 * all40_mpnn_i).sum(0), loc_clusters_22),
            "proteinmpnn_22cl_gated40": _eval_clusters(
                np.where(hi40, (w40 * all40_mpnn_d).sum(0), sf_nat.zs_dir["mint_zs"]),
                np.where(hi40, (w40 * all40_mpnn_i).sum(0), sf_nat.zs_imp["mint_zs"]),
                loc_clusters_22,
            ),
            "gearbind_22cl_top1_iptm": _eval_clusters(all40_gb_d[idx40_iptm, cols], all40_gb_i[idx40_iptm, cols], loc_clusters_22),
            "gearbind_22cl_softmax40": _eval_clusters((w40 * all40_gb_d).sum(0), (w40 * all40_gb_i).sum(0), loc_clusters_22),
            "gearbind_22cl_gated40": _eval_clusters(
                np.where(hi40, (w40 * all40_gb_d).sum(0), logica_1d_dir),
                np.where(hi40, (w40 * all40_gb_i).sum(0), logica_1d_imp),
                loc_clusters_22,
            ),
            "models_18cl_cocrystal_top1_iptm": mm_models_18_top1,
            "models_18cl_cocrystal_softmax_ensemble": mm_models_18_ens,
            "models_18cl_cocrystal_best40_oracle": mm_models_18_oracle,
            "models_22cl_loaco_top1_iptm": mm_models_22_top1,
            "models_22cl_loaco_softmax_ensemble": mm_models_22_ens,
        }
        master_json = {
            "dockq_summaries": {
                "all_340_cocrystals": dockq_cache["all_851_summary"],
                "loaco_47_cocrystals": dockq_cache["tier1_loaco_summary"],
                "paired_fv_249_cocrystals": dockq_cache["paired_fv_summary"],
                "vhh_91_cocrystals": dockq_cache["vhh_summary"],
            },
            "methods": all_methods_out,
            "multi_method_40pose": mm_summary,
        }
        master_json_path = res_dir / "cofolding_all_methods_evaluation.json"
        master_json_path.write_text(json.dumps(master_json, indent=2) + "\n")

        # Also write master markdown summary `results/COFOLDING_BENCHMARK_EVALUATION.md`
        b_md = [
            "# Multi-Method, 10-Pose Antibody–Antigen Co-Folding Benchmark (`AF3`, `Boltz-2`, `Chai-1`, `Protenix`)",
            "",
            "## 1. Structural Accuracy (`DockQ` vs. `340` Native WT Co-Crystals & `47` `LOACO-Local` Co-Crystals)",
            "",
            "| Co-Folding Method | All 340 Pose-0 `DockQ` | All 340 Expected 1-Pose | All 340 Top-`ipTM` `DockQ` (`%>=0.23` / `0.49` / `0.80`) | All 340 Best-of-10 `DockQ` | $\\rho(\\text{AbAg-ipTM},\\text{DockQ})$ | `LOACO-Local` (`n=47`) Top-`ipTM` / Best-of-10 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for mk_label, mk_key in [("AlphaFold 3 (`templates=[]`)", "af3"), ("Boltz-2", "boltz2"), ("Chai-1", "chai1"), ("Protenix", "protenix"), ("40-Pose Multi-Method Pool", "multi_method_40pose")]:
            d340 = dockq_cache["all_851_summary"]["methods"][mk_key]
            d47 = dockq_cache["tier1_loaco_summary"]["methods"][mk_key]
            p0_s = f"`{d340['pose0_mean_DockQ']:.3f}`" if "pose0_mean_DockQ" in d340 else "—"
            ex_s = f"`{d340['expected_single_pose_DockQ']:.3f}`" if "expected_single_pose_DockQ" in d340 else "—"
            b_k = "best_of_40_mean_DockQ" if "best_of_40_mean_DockQ" in d340 else "best_of_10_mean_DockQ"
            b_k47 = "best_of_40_mean_DockQ" if "best_of_40_mean_DockQ" in d47 else "best_of_10_mean_DockQ"
            rho_s = f"`{d340['spearman_abag_iptm_vs_dockq']:+.3f}`" if "spearman_abag_iptm_vs_dockq" in d340 else "—"
            b_md.append(
                f"| **{mk_label}** | {p0_s} | {ex_s} | **`{d340['top1_iptm_mean_DockQ']:.3f}`** (`{d340['top1_acceptable_pct_ge_023']:.1f}%` / `{d340['top1_medium_pct_ge_049']:.1f}%` / `{d340['top1_high_pct_ge_080']:.1f}%`) | **`{d340[b_k]:.3f}`** | {rho_s} | **`{d47['top1_iptm_mean_DockQ']:.3f}`** / **`{d47[b_k47]:.3f}`** |"
            )
        b_md.extend([
            "",
            "## 2. All 8 3D Models Across Co-Folding Methods (`18`-Cluster Co-Crystal Ablation, 10-Pose `ipTM`-Softmax Ensemble)",
            "",
            "| 3D Baseline Model | `AF3` (Sign / Imp / **Ord**) | `Boltz-2` (Sign / Imp / **Ord**) | `Chai-1` (Sign / Imp / **Ord**) | `Protenix` (Sign / Imp / **Ord**) | `40-Pose Consensus` (Sign / Imp / **Ord**) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for mlabel, _, _ in model_names_map:
            row_cells = [f"**{mlabel}**"]
            for mk_key in METHODS:
                mc_e = all_methods_out[mk_key]["models_18cl_cocrystal_softmax_ensemble"][mlabel]
                row_cells.append(f"`{mc_e['sign']['mean']:.3f}` / `{mc_e['impact']['mean']:.3f}` / **`{mc_e['ordinal']['mean']:.3f}`**")
            mm_e = mm_models_18_ens[mlabel]
            row_cells.append(f"`{mm_e['sign']['mean']:.3f}` / `{mm_e['impact']['mean']:.3f}` / **`{mm_e['ordinal']['mean']:.3f}`**")
            b_md.append("| " + " | ".join(row_cells) + " |")

        b_md.extend([
            "",
            "## 3. All 8 3D Models Across Co-Folding Methods (`22`-Cluster Full Crystal-Free `LOACO-Local`, 10-Pose `ipTM`-Softmax Ensemble)",
            "",
            "| 3D Baseline Model | `AF3` (Sign / Imp / **Ord**) | `Boltz-2` (Sign / Imp / **Ord**) | `Chai-1` (Sign / Imp / **Ord**) | `Protenix` (Sign / Imp / **Ord**) | `40-Pose Consensus` (Sign / Imp / **Ord**) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for mlabel, _, _ in model_names_map:
            row_cells = [f"**{mlabel}**"]
            for mk_key in METHODS:
                mc_e = all_methods_out[mk_key]["models_22cl_loaco_softmax_ensemble"][mlabel]
                row_cells.append(f"`{mc_e['sign']['mean']:.3f}` / `{mc_e['impact']['mean']:.3f}` / **`{mc_e['ordinal']['mean']:.3f}`**")
            mm_e = mm_models_22_ens[mlabel]
            row_cells.append(f"`{mm_e['sign']['mean']:.3f}` / `{mm_e['impact']['mean']:.3f}` / **`{mm_e['ordinal']['mean']:.3f}`**")
            b_md.append("| " + " | ".join(row_cells) + " |")

        bench_md_path = res_dir / "COFOLDING_BENCHMARK_EVALUATION.md"
        bench_md_path.write_text("\n".join(b_md) + "\n")
        print(f"Wrote master JSON -> {master_json_path} and {bench_md_path} in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
