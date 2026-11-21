#!/usr/bin/env python3
"""Step 16: Structural Noise, Misplaced Contacts, & Epitope-Localization Sensitivity Analysis
Across All 47 Solved WT Co-Crystal Scaffolds (`19` Co-Crystal Clusters / `18` Directional
Clusters in `LOACO-Local`).

Directly answers a core mechanistic question:
  "How sensitive are 3D antibody-antigen models to coordinate noise / misplaced contacts,
   versus simply knowing WHERE the antibody binds on the antigen (epitope localization)?"

Evaluates 10 controlled structural regimes on the exact same 19 co-crystal clusters (`47` WT scaffolds):
  Axis A — Local Contact / Coordinate Noise AT THE TRUE EPITOPE (intra-residue N-CA-C-O-CB rigid):
    0. `native_cocrystal`          : Exact Bound WT Co-Crystal (`0.0 Å`, True Epitope)
    1. `local_noise_0p5A`          : `σ = 0.5 Å` 3D RMSD + `3°` frame jitter (Cryo-EM / B-factor noise)
    2. `local_noise_1p0A`          : `σ = 1.0 Å` 3D RMSD + `7°` frame jitter (Few misplaced contacts)
    3. `local_noise_2p0A`          : `σ = 2.0 Å` 3D RMSD + `14°` frame jitter (Scrambled H-bonds, intact epitope)
    4. `local_noise_3p5A`          : `σ = 3.5 Å` 3D RMSD + `22°` frame jitter (Severe local distortion)
  Axis B — Rigid-Body Docking Shift, Unbound CDR Conformation, & Wrong Epitope:
    5. `dock_jitter_2A_10deg`      : Intact native monomers + `2 Å / 10°` pose jitter (True Epitope)
    6. `dock_shift_5A_25deg`       : Intact native monomers + `5 Å / 25°` tangential shift (Edge of Epitope)
    7. `igfold_at_true_epitope`    : Unbound `IgFold` Fv Kabsch-aligned to True WT Pose (True Epitope, unbound CDRs)
    8. `igfold_unguided_patch`     : Unbound `IgFold` Fv at heuristic exposed patch (Unknown Epitope)
    9. `native_fv_wrong_epitope`   : Intact native X-ray Fv docked at antipodal surface (Wrong Epitope!)

Outputs:
  - `data/structures/structure_noise_sensitivity_cache.pt`
  - `results/structure_noise_sensitivity.json`
  - `results/STRUCTURE_NOISE_SENSITIVITY.md`
"""

from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

from abrank.baselines.models import assemble_split_tensors, compute_stratum_weights
from abrank.baselines.structure_models import (
    ANTIPASTI3StateTrainer,
    GearBind3StateTrainer,
    assemble_structure_pair_features,
)
from abrank.config import get_repo_root
from abrank.evaluate import compute_3state_metrics, compute_cluster_bootstrap_summary
from abrank.structures.resolver import (
    ChainBackbone,
    align_query_to_chain,
    compute_scaffold_biophysics,
    parse_structure_chains,
    write_standardized_pdb,
)


def _rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = axis_angle / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def perturb_chain_residue_frames(cb: ChainBackbone, sigma_3d: float, sigma_deg: float, rng: np.random.Generator) -> ChainBackbone:
    """Perturb each residue's rigid `[N, CA, C, O, CB]` frame so intra-residue bond lengths stay exact
    while 3D Cα RMSD equals `sigma_3d` Å."""
    coords = cb.coords.copy()
    L = len(cb.seq)
    s_coord = sigma_3d / np.sqrt(3.0)
    s_rad = (sigma_deg * np.pi / 180.0) / np.sqrt(3.0)
    for i in range(L):
        ca = coords[i, 1, :].copy()
        R = _rodrigues(rng.normal(scale=s_rad, size=3).astype(np.float32))
        dt = rng.normal(scale=s_coord, size=3).astype(np.float32)
        coords[i] = (coords[i] - ca[None, :]) @ R.T + ca[None, :] + dt[None, :]
    return ChainBackbone(seq=cb.seq, coords=coords)


def _relieve_clash(ab_coords_list: list[np.ndarray], ag_ca: np.ndarray, push_dir: np.ndarray, min_ca: float = 3.8) -> list[np.ndarray]:
    ab_ca = np.concatenate([c[:, 1, :] for c in ab_coords_list], axis=0)
    d_min = float(np.min(np.linalg.norm(ab_ca[:, None, :] - ag_ca[None, :, :], axis=-1)))
    if d_min < min_ca:
        shift = (min_ca - d_min) * push_dir[None, None, :]
        return [c + shift for c in ab_coords_list]
    return ab_coords_list


def rigid_dock_perturb(
    cb_h: ChainBackbone, cb_l: ChainBackbone | None, cb_a: ChainBackbone, shift_a: float, rot_deg: float
) -> tuple[ChainBackbone, ChainBackbone | None, ChainBackbone]:
    """Rigidly rotate & tangentially shift intact native Ab relative to intact native Ag."""
    ab_list = [cb_h.coords.copy()] + ([cb_l.coords.copy()] if cb_l is not None else [])
    ab_ca = np.concatenate([c[:, 1, :] for c in ab_list], axis=0)
    ag_ca = cb_a.coords[:, 1, :]

    dmat = np.linalg.norm(ab_ca[:, None, :] - ag_ca[None, :, :], axis=-1)
    if_mask = np.any(dmat <= 10.0, axis=1)
    pivot = ab_ca[if_mask].mean(axis=0) if np.any(if_mask) else ab_ca.mean(axis=0)

    n_vec = ab_ca.mean(axis=0) - ag_ca.mean(axis=0)
    n_hat = (n_vec / (np.linalg.norm(n_vec) + 1e-8)).astype(np.float32)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(n_hat[0])) < 0.8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    t1 = np.cross(n_hat, ref); t1 = (t1 / (np.linalg.norm(t1) + 1e-8)).astype(np.float32)
    t2 = np.cross(n_hat, t1); t2 = (t2 / (np.linalg.norm(t2) + 1e-8)).astype(np.float32)

    R = _rodrigues(t1 * float(rot_deg * np.pi / 180.0))
    trans = shift_a * t2
    moved = [((c - pivot[None, None, :]) @ R.T) + pivot[None, None, :] + trans[None, None, :] for c in ab_list]
    moved = _relieve_clash(moved, ag_ca, n_hat, min_ca=3.8)
    new_h = ChainBackbone(cb_h.seq, moved[0])
    new_l = ChainBackbone(cb_l.seq, moved[1]) if cb_l is not None else None
    return new_h, new_l, cb_a


def superpose_igfold_onto_true_epitope(
    cb_h_ig: ChainBackbone, cb_l_ig: ChainBackbone | None,
    cb_h_nat: ChainBackbone, cb_l_nat: ChainBackbone | None, cb_a_nat: ChainBackbone,
) -> tuple[ChainBackbone, ChainBackbone | None, ChainBackbone]:
    """Kabsch-align unbound `IgFold` Fv onto the true native WT Fv pose so global epitope is exact."""
    _, _, idx_h = align_query_to_chain(cb_h_ig.seq, cb_h_nat.seq)
    ok_h = idx_h >= 0
    P_pts = [cb_h_ig.coords[ok_h, 1, :]]
    Q_pts = [cb_h_nat.coords[idx_h[ok_h], 1, :]]
    if cb_l_ig is not None and cb_l_nat is not None:
        _, _, idx_l = align_query_to_chain(cb_l_ig.seq, cb_l_nat.seq)
        ok_l = idx_l >= 0
        if np.any(ok_l):
            P_pts.append(cb_l_ig.coords[ok_l, 1, :])
            Q_pts.append(cb_l_nat.coords[idx_l[ok_l], 1, :])
    P = np.concatenate(P_pts, axis=0)
    Q = np.concatenate(Q_pts, axis=0)
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    S = np.diag([1.0, 1.0, -1.0 if np.linalg.det(Vt.T @ U.T) < 0 else 1.0]).astype(np.float32)
    R = (Vt.T @ S @ U.T).astype(np.float32)
    t = (cq - cp @ R.T).astype(np.float32)

    ab_list = [cb_h_ig.coords @ R.T + t[None, None, :]]
    if cb_l_ig is not None:
        ab_list.append(cb_l_ig.coords @ R.T + t[None, None, :])
    ag_ca = cb_a_nat.coords[:, 1, :]
    n_vec = np.concatenate([c[:, 1, :] for c in ab_list], 0).mean(0) - ag_ca.mean(0)
    n_hat = (n_vec / (np.linalg.norm(n_vec) + 1e-8)).astype(np.float32)
    ab_list = _relieve_clash(ab_list, ag_ca, n_hat, min_ca=3.6)
    return ChainBackbone(cb_h_ig.seq, ab_list[0]), (ChainBackbone(cb_l_ig.seq, ab_list[1]) if cb_l_ig is not None else None), cb_a_nat


def place_native_fv_at_wrong_epitope(
    cb_h_nat: ChainBackbone, cb_l_nat: ChainBackbone | None, cb_a_nat: ChainBackbone,
) -> tuple[ChainBackbone, ChainBackbone | None, ChainBackbone]:
    """Place intact native X-ray Fv against the antipodal non-epitope surface of the antigen."""
    ab_list = [cb_h_nat.coords.copy()] + ([cb_l_nat.coords.copy()] if cb_l_nat is not None else [])
    ab_ca = np.concatenate([c[:, 1, :] for c in ab_list], axis=0)
    ag_ca = cb_a_nat.coords[:, 1, :]
    dmat = np.linalg.norm(ab_ca[:, None, :] - ag_ca[None, :, :], axis=-1)
    ep_mask = np.any(dmat <= 8.5, axis=0)
    true_ep_center = ag_ca[ep_mask].mean(0) if np.any(ep_mask) else ag_ca[np.argmin(dmat.min(0))]
    ag_center = ag_ca.mean(0)

    # Pick surface residue on Ag farthest from true epitope!
    anti_idx = int(np.argmax(np.linalg.norm(ag_ca - true_ep_center[None, :], axis=-1)))
    anti_pt = ag_ca[anti_idx]
    out_dir = anti_pt - ag_center
    out_hat = (out_dir / (np.linalg.norm(out_dir) + 1e-8)).astype(np.float32)

    # Rotate native Fv by 180 deg so paratope (which pointed towards -n_orig) points towards -out_hat!
    if_ab_mask = np.any(dmat <= 8.5, axis=1)
    para_center = ab_ca[if_ab_mask].mean(0) if np.any(if_ab_mask) else ab_ca.mean(0)
    v_para = para_center - ab_ca.mean(0)
    v_para = (v_para / (np.linalg.norm(v_para) + 1e-8)).astype(np.float32)
    target_v = -out_hat
    ax = np.cross(v_para, target_v)
    n_ax = float(np.linalg.norm(ax))
    dot_v = float(np.clip(np.dot(v_para, target_v), -1.0, 1.0))
    if n_ax < 1e-6:
        R = np.eye(3, dtype=np.float32) if dot_v > 0 else np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    else:
        R = _rodrigues((ax / n_ax) * float(np.arccos(dot_v)))

    target_para_pos = anti_pt + 6.0 * out_hat
    moved = [((c - para_center[None, None, :]) @ R.T) + target_para_pos[None, None, :] for c in ab_list]
    moved = _relieve_clash(moved, ag_ca, out_hat, min_ca=5.5)
    return ChainBackbone(cb_h_nat.seq, moved[0]), (ChainBackbone(cb_l_nat.seq, moved[1]) if cb_l_nat is not None else None), cb_a_nat


def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def main() -> None:
    t0 = time.time()
    root = get_repo_root()
    bdir = root / "data" / "benchmark"
    edir = root / "data" / "embeddings"
    sdir = root / "data" / "structures"
    res_dir = root / "results"

    mod09 = _load_mod(root / "scripts" / "09_extract_structure_and_multichain_features.py", "step09")
    cf03 = _load_mod(root / "scripts" / "cofolding" / "03_ingest_and_evaluate_cofolded.py", "cf03")

    kd_lookup = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False).drop_duplicates("AbAgID").set_index("AbAgID")
    kd_lookup["Ab_light_chain_seq"] = kd_lookup["Ab_light_chain_seq"].fillna("")
    loc_df = pd.concat([pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}.tsv", sep="\t") for s in ("train", "val", "test")], ignore_index=True)
    loc_df = loc_df[loc_df["Category"].isin(["L1", "L2", "L3"])].reset_index(drop=True)
    n_loc = len(loc_df)
    loc_c1 = loc_df["AbAgID_1"].map(kd_lookup["Ag_cluster"]).astype(str).to_numpy()
    ab_lin_1 = loc_df["AbAgID_1"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()
    ab_lin_2 = loc_df["AbAgID_2"].map(kd_lookup["Ab_lineage"]).astype(str).to_numpy()

    cc_nat = torch.load(sdir / "complex_cache.pt", map_location="cpu", weights_only=False)
    unb_abl = torch.load(sdir / "unbound_cocrystal_ablation.pt", map_location="cpu", weights_only=False)
    abag2sc = cc_nat["abag_to_scaffold"]
    scafs_nat = cc_nat["scaffolds"]

    with open(res_dir / "loaco_evaluation.json") as f:
        loaco_json = json.load(f)
    loc_clusters = sorted(loaco_json["cluster_metadata"].keys())
    cocrys_clusters = [c for c in loc_clusters if loaco_json["cluster_metadata"][c]["cocrystal_frac"] >= 0.5]
    cocrys_sids = sorted({abag2sc[a] for a in set(loc_df["AbAgID_1"]).union(loc_df["AbAgID_2"]) if scafs_nat[abag2sc[a]]["is_cocrystal"]})

    new_regimes = [
        ("local_noise_0p5A", "Local Contact Jitter (0.5 Å, 3°)"),
        ("local_noise_1p0A", "Local Contact Jitter (1.0 Å, 7°)"),
        ("local_noise_2p0A", "Local Contact Jitter (2.0 Å, 14°)"),
        ("local_noise_3p5A", "Local Contact Jitter (3.5 Å, 22°)"),
        ("dock_jitter_2A_10deg", "Rigid Dock Jitter (2 Å + 10°, True Epitope)"),
        ("dock_shift_5A_25deg", "Rigid Dock Shift (5 Å + 25°, Shifted Pose)"),
        ("igfold_at_true_epitope", "Unbound IgFold @ True Epitope (Kabsch Fv)"),
        ("native_fv_wrong_epitope", "Native Fv @ Wrong Antipodal Epitope"),
    ]

    cache_path = sdir / "structure_noise_sensitivity_cache.pt"
    if cache_path.exists():
        print(f"Loading cached perturbed scaffolds from {cache_path}...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        reg_scafs = cached["reg_scafs"]
        reg_inv = cached["reg_inv"]
        reg_geom_stats = cached["reg_geom_stats"]
    else:
        pdb_dir = sdir / "perturbed_noise_scaffolds"
        pdb_dir.mkdir(parents=True, exist_ok=True)
        igfold_dir = sdir / "igfold"

        reg_scafs = {rk: {} for rk, _ in new_regimes}
        reg_geom_rows = {rk: [] for rk, _ in new_regimes}
        all_shard_jobs: list[tuple[str, str, bool]] = []

        print(f"Generating 8 perturbation regimes x {len(cocrys_sids)} WT co-crystal scaffolds ({8 * len(cocrys_sids)} PDBs)...")
        for s_idx, sid in enumerate(cocrys_sids):
            nat_bio = scafs_nat[sid]
            orig_chains = parse_structure_chains(root / nat_bio["pdb_path"])
            cb_h_n = orig_chains["H"]
            cb_l_n = orig_chains.get("L") if str(nat_bio["seq_L"]) else None
            cb_a_n = orig_chains["A"]
            has_l = cb_l_n is not None

            ab_key = hashlib.sha1(f"{nat_bio['seq_H']}|{nat_bio['seq_L']}".encode()).hexdigest()[:14]
            ig_chains = parse_structure_chains(igfold_dir / f"{ab_key}.pdb")
            cb_h_ig = ig_chains.get("H") or next(iter(ig_chains.values()))
            cb_l_ig = ig_chains.get("L") if has_l else None

            ca_ab_nat = np.concatenate([cb_h_n.coords[:, 1, :], cb_l_n.coords[:, 1, :]], 0) if has_l else cb_h_n.coords[:, 1, :]
            ca_ag_nat = cb_a_n.coords[:, 1, :]

            rng = np.random.default_rng(1000 + s_idx)
            pert_map = {
                "local_noise_0p5A": (
                    perturb_chain_residue_frames(cb_h_n, 0.5, 3.0, rng),
                    perturb_chain_residue_frames(cb_l_n, 0.5, 3.0, rng) if has_l else None,
                    perturb_chain_residue_frames(cb_a_n, 0.5, 3.0, rng),
                    True,
                ),
                "local_noise_1p0A": (
                    perturb_chain_residue_frames(cb_h_n, 1.0, 7.0, rng),
                    perturb_chain_residue_frames(cb_l_n, 1.0, 7.0, rng) if has_l else None,
                    perturb_chain_residue_frames(cb_a_n, 1.0, 7.0, rng),
                    True,
                ),
                "local_noise_2p0A": (
                    perturb_chain_residue_frames(cb_h_n, 2.0, 14.0, rng),
                    perturb_chain_residue_frames(cb_l_n, 2.0, 14.0, rng) if has_l else None,
                    perturb_chain_residue_frames(cb_a_n, 2.0, 14.0, rng),
                    True,
                ),
                "local_noise_3p5A": (
                    perturb_chain_residue_frames(cb_h_n, 3.5, 22.0, rng),
                    perturb_chain_residue_frames(cb_l_n, 3.5, 22.0, rng) if has_l else None,
                    perturb_chain_residue_frames(cb_a_n, 3.5, 22.0, rng),
                    True,
                ),
                "dock_jitter_2A_10deg": (*rigid_dock_perturb(cb_h_n, cb_l_n, cb_a_n, 2.0, 10.0), True),
                "dock_shift_5A_25deg": (*rigid_dock_perturb(cb_h_n, cb_l_n, cb_a_n, 5.0, 25.0), True),
                "igfold_at_true_epitope": (*superpose_igfold_onto_true_epitope(cb_h_ig, cb_l_ig, cb_h_n, cb_l_n, cb_a_n), False),
                "native_fv_wrong_epitope": (*place_native_fv_at_wrong_epitope(cb_h_n, cb_l_n, cb_a_n), False),
            }

            for rk, (ph, pl, pa, is_cc_flag) in pert_map.items():
                psid = f"{rk}__{sid}"
                out_pdb = pdb_dir / f"{psid}.pdb"
                write_standardized_pdb(out_pdb, ph, pl, pa)
                bio_p = compute_scaffold_biophysics(out_pdb, ph, pl, pa, is_cocrystal=is_cc_flag)
                bio_p["pdb_path"] = str(out_pdb.relative_to(root))
                bio_p["3di_H"] = nat_bio["3di_H"]
                bio_p["3di_L"] = nat_bio["3di_L"]
                bio_p["3di_A"] = nat_bio["3di_A"]
                reg_scafs[rk][sid] = bio_p
                all_shard_jobs.append((psid, str(out_pdb), has_l))

                ca_ab_p = np.concatenate([ph.coords[:, 1, :], pl.coords[:, 1, :]], 0) if pl is not None else ph.coords[:, 1, :]
                dq = cf03.compute_dockq_ca(ca_ab_p, pa.coords[:, 1, :], ca_ab_nat, ca_ag_nat)
                reg_geom_rows[rk].append(dq)

        # Also compute DockQ for cached `igfold_unguided_patch`!
        unb_dq_rows = []
        for sid in cocrys_sids:
            nat_bio = scafs_nat[sid]
            unb_bio = unb_abl["unb_scaffolds"][sid]
            ca_ab_n = np.concatenate([nat_bio["ca_H"], nat_bio["ca_L"]], 0) if len(nat_bio["ca_L"]) > 0 else nat_bio["ca_H"]
            ca_ab_u = np.concatenate([unb_bio["ca_H"], unb_bio["ca_L"]], 0) if len(unb_bio["ca_L"]) > 0 else unb_bio["ca_H"]
            unb_dq_rows.append(cf03.compute_dockq_ca(ca_ab_u, unb_bio["ca_A"], ca_ab_n, nat_bio["ca_A"]))

        def _agg_dq(rows: list[dict]) -> dict:
            return {
                "DockQ": round(float(np.mean([r["DockQ"] for r in rows])), 3),
                "fnat": round(float(np.mean([r["fnat"] for r in rows])), 3),
                "iRMSD": round(float(np.mean([r["iRMSD"] for r in rows])), 2),
                "LRMSD": round(float(np.mean([r["LRMSD"] for r in rows])), 2),
            }

        reg_geom_stats = {
            "native_cocrystal": {"DockQ": 1.000, "fnat": 1.000, "iRMSD": 0.00, "LRMSD": 0.00},
            **{rk: _agg_dq(reg_geom_rows[rk]) for rk, _ in new_regimes},
            "igfold_unguided_patch": _agg_dq(unb_dq_rows),
        }

        n_w = 8
        shards = [all_shard_jobs[w::n_w] for w in range(n_w)]
        mpnn_repo = str(Path.home() / ".cache" / "abrank" / "repos" / "ProteinMPNN")
        print(f"Extracting ESM-IF1, AntiFold, and ProteinMPNN across {len(all_shard_jobs)} perturbed PDBs ({n_w} workers)...", flush=True)
        if1_merged: dict[str, dict] = {}
        mpnn_merged: dict[str, dict] = {}
        with mp.get_context("fork").Pool(n_w) as pool:
            for d_part in pool.map(mod09._if1_antifold_worker, [(sh, str(pdb_dir), "") for sh in shards]):
                if1_merged.update(d_part)
            for d_part in pool.map(mod09._protein_mpnn_worker, [(sh, mpnn_repo) for sh in shards]):
                mpnn_merged.update(d_part)

        reg_inv = {rk: {} for rk, _ in new_regimes}
        for rk, _ in new_regimes:
            for sid in cocrys_sids:
                psid = f"{rk}__{sid}"
                reg_inv[rk][sid] = {**if1_merged[psid], **mpnn_merged[psid]}

        torch.save({"reg_scafs": reg_scafs, "reg_inv": reg_inv, "reg_geom_stats": reg_geom_stats}, cache_path)
        print(f"Saved cache -> {cache_path}")

    # Load base embeddings and assemble structure features for all 10 regimes
    esmc_c = torch.load(edir / "esmc_300m.pt", map_location="cpu", weights_only=False)
    abl_c = torch.load(edir / "ablang2_paired.pt", map_location="cpu", weights_only=False)
    esm2_c = torch.load(edir / "esm2_150m.pt", map_location="cpu", weights_only=False)
    inv_nat = torch.load(edir / "inverse_folding_scaffolds.pt", map_location="cpu", weights_only=False)
    sap_nat = torch.load(edir / "saprot_650m.pt", map_location="cpu", weights_only=False)
    mint_nat = torch.load(edir / "mint_650m.pt", map_location="cpu", weights_only=False)

    sp_all = assemble_split_tensors(loc_df, kd_lookup, esmc_c, abl_c, esm2_c)
    inv_base = inv_nat["scaffolds"] if "scaffolds" in inv_nat else inv_nat

    all_10_regimes = [
        ("native_cocrystal", "Native WT Co-Crystal (0.0 Å, True Epitope)"),
        *new_regimes[:7],
        ("igfold_unguided_patch", "Unbound IgFold @ Unguided Patch (Unknown Epitope)"),
        new_regimes[7],
    ]

    sf_by_reg: dict[str, Any] = {}
    contact_by_reg: dict[str, np.ndarray] = {}
    gb_by_reg: dict[str, tuple] = {}
    for rk, _ in all_10_regimes:
        if rk == "native_cocrystal":
            sc_map = scafs_nat
            iv_map = inv_base
        elif rk == "igfold_unguided_patch":
            sc_map = {**scafs_nat, **unb_abl["unb_scaffolds"]}
            iv_map = {**inv_base, **unb_abl["unb_inv"]}
        else:
            sc_map = {**scafs_nat, **reg_scafs[rk]}
            iv_map = {**inv_base, **reg_inv[rk]}
        sf_r = assemble_structure_pair_features(loc_df, kd_lookup, {"abag_to_scaffold": abag2sc, "scaffolds": sc_map}, iv_map, sap_nat, mint_nat, True)
        sf_by_reg[rk] = sf_r
        contact_by_reg[rk] = _contact_impact_scores(loc_df, kd_lookup, abag2sc, sc_map)
        gb_by_reg[rk] = (
            GearBind3StateTrainer._precompute_graphs(sf_r.patch_ca_1, sf_r.patch_cb_1, sf_r.patch_chain_1, sf_r.patch_seqpos_1),
            GearBind3StateTrainer._precompute_graphs(sf_r.patch_ca_2, sf_r.patch_cb_2, sf_r.patch_chain_2, sf_r.patch_seqpos_2),
        )

    def _slice_gb(gb_t: tuple, ix: np.ndarray) -> tuple:
        (a1, e1, m1), (a2, e2, m2) = gb_t
        return ((a1[ix], e1[ix], m1[ix]), (a2[ix], e2[ix], m2[ix]))

    # Fit supervised ANTIPASTI & GearBind once per LOACO co-crystal cluster on clean disjoint train folds,
    # and evaluate test-time robustness across all 10 structural regimes!
    torch.set_num_threads(min(16, os.cpu_count() or 8))
    print(f"Evaluating supervised ANTIPASTI & GearBind + zero-shot 3D models across {len(cocrys_clusters)} co-crystal clusters x 10 regimes...", flush=True)
    sf_clean = sf_by_reg["native_cocrystal"]
    gb_clean = gb_by_reg["native_cocrystal"]
    anti_preds = {rk: (np.zeros(n_loc), np.zeros(n_loc)) for rk, _ in all_10_regimes}
    gb_preds = {rk: (np.zeros(n_loc), np.zeros(n_loc)) for rk, _ in all_10_regimes}

    for cl in cocrys_clusters:
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
        w_tr = compute_stratum_weights(loc_df.iloc[tr_i].reset_index(drop=True), sp_all.is_ag_mut[tr_i])

        m_anti = ANTIPASTI3StateTrainer(seed=42, epochs=15).fit(
            sf_clean.subset(tr_i), sp_all.subset(tr_i), sf_clean.subset(val_i), sp_all.subset(val_i), val_df_sub, w_tr
        )
        m_gb = GearBind3StateTrainer(seed=42, epochs=15).fit(
            sf_clean.subset(tr_i), sp_all.subset(tr_i), sf_clean.subset(val_i), sp_all.subset(val_i), val_df_sub, w_tr,
            precomp_tr_g=_slice_gb(gb_clean, tr_i), precomp_val_g=_slice_gb(gb_clean, val_i),
        )
        for rk, _ in all_10_regimes:
            da, ia = m_anti.predict_3state(sf_by_reg[rk].subset(te_idx), sp_all.subset(te_idx))
            anti_preds[rk][0][te_idx], anti_preds[rk][1][te_idx] = da, ia
            dg, ig = m_gb.predict_3state(sf_by_reg[rk].subset(te_idx), _precomp_g=_slice_gb(gb_by_reg[rk], te_idx))
            gb_preds[rk][0][te_idx], gb_preds[rk][1][te_idx] = dg, ig

    lbls = loc_df["label"].to_numpy(dtype=int)
    bds = loc_df["binding_diff"].to_numpy(dtype=float)

    model_keys = [
        "WT 3D Contact (-min Cβ)",
        "PRODIGY+SASA (3D ZS)",
        "ProteinMPNN (3D ZS)",
        "ESM-IF1 (3D ZS)",
        "AntiFold (3D Ab-Ag ZS)",
        "ANTIPASTI (3D CNN)",
        "GearBind (3D GNN)",
    ]
    results_out: dict[str, dict] = {}
    for rk, rtitle in all_10_regimes:
        sf_r = sf_by_reg[rk]
        m_vecs = {
            "WT 3D Contact (-min Cβ)": (np.zeros(n_loc), contact_by_reg[rk]),
            "PRODIGY+SASA (3D ZS)": (sf_r.zs_dir["prodigy_sasa"], sf_r.zs_imp["prodigy_sasa"]),
            "ProteinMPNN (3D ZS)": (sf_r.zs_dir["protein_mpnn"], sf_r.zs_imp["protein_mpnn"]),
            "ESM-IF1 (3D ZS)": (sf_r.zs_dir["esm_if1"], sf_r.zs_imp["esm_if1"]),
            "AntiFold (3D Ab-Ag ZS)": (sf_r.zs_dir["antifold"], sf_r.zs_imp["antifold"]),
            "ANTIPASTI (3D CNN)": anti_preds[rk],
            "GearBind (3D GNN)": gb_preds[rk],
        }
        m_res = {}
        for mk in model_keys:
            dv, iv = m_vecs[mk]
            per_cl = {cl: compute_3state_metrics(lbls[loc_c1 == cl], bds[loc_c1 == cl], dv[loc_c1 == cl], iv[loc_c1 == cl]) for cl in cocrys_clusters}
            m_res[mk] = compute_cluster_bootstrap_summary(per_cl, min_class_pairs=2, n_bootstrap=2000, seed=42)
        results_out[rk] = {
            "title": rtitle,
            "geometry_vs_native": reg_geom_stats[rk],
            "models": m_res,
        }

    out_json = res_dir / "structure_noise_sensitivity.json"
    out_json.write_text(json.dumps({"n_cocrystal_clusters": len(cocrys_clusters), "n_scaffolds": len(cocrys_sids), "regimes": results_out}, indent=2) + "\n")

    md = [
        "# Structural Noise, Misplaced Contacts, & Epitope-Localization Sensitivity (`18` Directional Co-Crystal Clusters, `47` WT Scaffolds)",
        "",
        "Each cell reports **`Sign AUC` / `Impact AUC` / `Ordinal AUC`** across the exact same `19` solved WT co-crystal clusters (`18` directional):",
        "",
        "| Structural Regime | `DockQ` (`fnat`, `iRMSD`) | `WT Contact` (Imp) | `PRODIGY+SASA` | `ProteinMPNN` | `ESM-IF1` | `ANTIPASTI` | `GearBind` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for rk, rtitle in all_10_regimes:
        ro = results_out[rk]
        gm = ro["geometry_vs_native"]
        ms = ro["models"]
        c_imp = ms["WT 3D Contact (-min Cβ)"]["impact"]["mean"]
        cells = []
        for mk in ("PRODIGY+SASA (3D ZS)", "ProteinMPNN (3D ZS)", "ESM-IF1 (3D ZS)", "ANTIPASTI (3D CNN)", "GearBind (3D GNN)"):
            mm = ms[mk]
            cells.append(f"`{mm['sign']['mean']:.3f}`/`{mm['impact']['mean']:.3f}`/**`{mm['ordinal']['mean']:.3f}`**")
        md.append(
            f"| **{rtitle}** | `{gm['DockQ']:.2f}` (`{gm['fnat']:.2f}`, `{gm['iRMSD']:.1f}Å`) | **`{c_imp:.3f}`** | {' | '.join(cells)} |"
        )
    out_md = res_dir / "STRUCTURE_NOISE_SENSITIVITY.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"Completed in {time.time() - t0:.1f}s! Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
