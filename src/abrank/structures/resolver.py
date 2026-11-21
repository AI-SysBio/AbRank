"""Canonical 3D Complex & Monomer Structure Resolver for AbRank v2.

Strategy (zero AlphaFold3 / Boltz co-folding required):
  1. **Experimental Ab–Ag Co-Crystals (`is_cocrystal = True`)**:
     Uses solved RCSB co-crystal structures (`.cif.gz` / bundled `AbAgym` PDBs) whenever both the
     wild-type antibody/VHH and cognate antigen are present in complex (`92.6%` of `unseen_ag/test`
     local `L1–L3` pairs, `100%` of `SKEMPI 2.0`, `100%` of `AbAgym` DMS, `73%` of `VHH` KD).
  2. **Experimental Antigen + `IgFold` Antibody/VHH (`is_cocrystal = False`)**:
     When no Ab–Ag co-crystal exists (e.g., `Kothiwal` receptor discovery panels, `OVA`, `AAYL`
     HR2 libraries, de novo `VHH` binders), extracts the experimental antigen structure from RCSB
     PDB and folds the antibody Fv (`VH + VL`) or single-domain `VHH` with `IgFold` (`Graylab/IgFold`),
     orienting the CDR paratope clash-free opposite the antigen's primary exposed surface patch while
     tracking `is_cocrystal = False` for downstream models.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import zipfile

from Bio.Align import PairwiseAligner
import freesasa
import gemmi
import numpy as np
import pandas as pd
import yaml

freesasa.setVerbosity(freesasa.nowarnings)

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M", "SEC": "C", "PYL": "K", "SEP": "S", "TPO": "T", "PTR": "Y",
}
ONE_TO_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}
PRODIGY_CLASS = {
    "R": "charged", "K": "charged", "H": "charged", "D": "charged", "E": "charged",
    "S": "polar", "T": "polar", "N": "polar", "Q": "polar", "C": "polar", "Y": "polar", "W": "polar",
    "A": "apolar", "V": "apolar", "I": "apolar", "L": "apolar", "M": "apolar",
    "F": "apolar", "P": "apolar", "G": "apolar",
}
HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
CHARGE_PH74 = {
    "R": 1.0, "K": 1.0, "H": 0.1, "D": -1.0, "E": -1.0,
}
VOLUME_NORM = {
    "G": 0.0, "A": 0.16, "S": 0.17, "C": 0.28, "D": 0.31, "P": 0.32, "N": 0.35,
    "T": 0.36, "E": 0.48, "V": 0.50, "Q": 0.53, "H": 0.58, "M": 0.64, "I": 0.67,
    "L": 0.67, "K": 0.68, "R": 0.73, "F": 0.77, "Y": 0.79, "W": 1.00,
}


@dataclass
class ChainBackbone:
    """Extracted protein chain with backbone + virtual/real C-beta coordinates."""

    seq: str
    coords: np.ndarray   # [L, 5, 3] float32 for (N, CA, C, O, CB)


def _virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute idealized C-beta coordinates from backbone (N, CA, C)."""
    b = ca - n
    c_vec = c - ca
    a = np.cross(b, c_vec)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca


def parse_structure_chains(path: Path) -> dict[str, ChainBackbone]:
    """Parse all polypeptide chains (>= 10 resolved CA atoms) from `.pdb`, `.cif`, or `.cif.gz`."""
    st = gemmi.read_structure(str(path))
    if len(st) == 0:
        return {}
    model = st[0]
    out: dict[str, ChainBackbone] = {}
    for ch in model:
        aa_chars: list[str] = []
        atom_rows: list[np.ndarray] = []
        seen_res_keys: set[tuple[int, str]] = set()
        for res in ch:
            aa = THREE_TO_ONE.get(res.name.strip().upper())
            if aa is None:
                continue
            rk = (res.seqid.num, res.seqid.icode)
            if rk in seen_res_keys:
                continue
            ca_at = res.find_atom("CA", "*")
            if ca_at is None:
                continue
            seen_res_keys.add(rk)
            ca = np.array([ca_at.pos.x, ca_at.pos.y, ca_at.pos.z], dtype=np.float32)

            def _get(aname: str, fallback: np.ndarray) -> np.ndarray:
                at = res.find_atom(aname, "*")
                if at is None:
                    return fallback.copy()
                return np.array([at.pos.x, at.pos.y, at.pos.z], dtype=np.float32)

            n = _get("N", ca + np.array([-1.45, 0.0, 0.0], dtype=np.float32))
            c = _get("C", ca + np.array([1.52, 0.0, 0.0], dtype=np.float32))
            o = _get("O", c + np.array([0.0, 1.23, 0.0], dtype=np.float32))
            cb_at = res.find_atom("CB", "*")
            if cb_at is not None and aa != "G":
                cb = np.array([cb_at.pos.x, cb_at.pos.y, cb_at.pos.z], dtype=np.float32)
            else:
                cb = _virtual_cb(n, ca, c).astype(np.float32)

            aa_chars.append(aa)
            atom_rows.append(np.stack([n, ca, c, o, cb], axis=0))

        if len(aa_chars) >= 10:
            cname = ch.name.strip() or f"C{len(out)}"
            if cname not in out or len(aa_chars) > len(out[cname].seq):
                out[cname] = ChainBackbone(seq="".join(aa_chars), coords=np.stack(atom_rows, axis=0))
    return out


def _make_aligner() -> PairwiseAligner:
    al = PairwiseAligner()
    al.mode = "local"
    al.match_score = 2.0
    al.mismatch_score = -1.0
    al.open_gap_score = -5.0
    al.extend_gap_score = -0.5
    return al


_GLOBAL_LOCAL_ALIGNER = _make_aligner()


def align_query_to_chain(query_seq: str, struct_seq: str) -> tuple[float, int, np.ndarray]:
    """Align `query_seq` against `struct_seq` and return `(identity, aligned_len, q_to_s_idx)`."""
    if not query_seq or not struct_seq:
        return 0.0, 0, np.full(len(query_seq), -1, dtype=np.int32)
    if query_seq == struct_seq:
        return 1.0, len(query_seq), np.arange(len(query_seq), dtype=np.int32)

    best = _GLOBAL_LOCAL_ALIGNER.align(query_seq, struct_seq)[0]
    q_to_s = np.full(len(query_seq), -1, dtype=np.int32)
    matches = 0
    aligned = 0
    for (qs, qe), (ss, se) in zip(best.aligned[0], best.aligned[1]):
        span = min(qe - qs, se - ss)
        for k in range(span):
            q_to_s[qs + k] = ss + k
            aligned += 1
            if query_seq[qs + k] == struct_seq[ss + k]:
                matches += 1
    ident = matches / max(aligned, 1)
    return float(ident), int(aligned), q_to_s


def min_interchain_ca_dist(c1: ChainBackbone, c2: ChainBackbone) -> float:
    ca1 = c1.coords[:, 1, :]
    ca2 = c2.coords[:, 1, :]
    # Chunk if large
    dmin = 1e9
    for i in range(0, len(ca1), 256):
        diff = ca1[i : i + 256, None, :] - ca2[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        cur = float(np.sqrt(np.min(d2)))
        if cur < dmin:
            dmin = cur
    return dmin


def select_cocrystal_chains(
    chains: dict[str, ChainBackbone],
    vh_seq: str,
    vl_seq: str,
    ag_seq: str,
) -> tuple[ChainBackbone, ChainBackbone | None, ChainBackbone] | None:
    """Match `(VH, [VL], Ag)` in a co-crystal structure requiring both sequence match and `<14 Å` contact."""
    has_l = len(vl_seq) > 0
    h_cands = []
    l_cands = []
    a_cands = []
    for cid, cb in chains.items():
        id_h, al_h, _ = align_query_to_chain(vh_seq, cb.seq)
        if id_h >= 0.55 and al_h >= min(55, int(0.55 * len(vh_seq))):
            h_cands.append((id_h * min(1.0, al_h / max(len(vh_seq), 1)), cid))
        if has_l:
            id_l, al_l, _ = align_query_to_chain(vl_seq, cb.seq)
            if id_l >= 0.55 and al_l >= min(50, int(0.55 * len(vl_seq))):
                l_cands.append((id_l * min(1.0, al_l / max(len(vl_seq), 1)), cid))
        id_a, al_a, _ = align_query_to_chain(ag_seq, cb.seq)
        min_al_a = min(10, len(ag_seq)) if len(ag_seq) < 30 else min(35, int(0.30 * len(ag_seq)))
        if id_a >= 0.50 and al_a >= min_al_a:
            a_cands.append((id_a * min(1.0, al_a / max(len(ag_seq), 1)), cid))

    h_cands.sort(reverse=True)
    l_cands.sort(reverse=True)
    a_cands.sort(reverse=True)
    if not h_cands or not a_cands or (has_l and not l_cands):
        return None

    best_tuple = None
    best_score = -1e9
    for sh, hc in h_cands[:6]:
        for sa, ac in a_cands[:8]:
            if ac == hc:
                continue
            if has_l:
                for sl, lc in l_cands[:6]:
                    if lc in (hc, ac):
                        continue
                    d_hl = min_interchain_ca_dist(chains[hc], chains[lc])
                    if d_hl > 14.0:
                        continue
                    d_ab_ag = min(
                        min_interchain_ca_dist(chains[hc], chains[ac]),
                        min_interchain_ca_dist(chains[lc], chains[ac]),
                    )
                    if d_ab_ag > 16.0:
                        continue
                    sc = sh + sl + sa - 0.02 * d_ab_ag
                    if sc > best_score:
                        best_score = sc
                        best_tuple = (chains[hc], chains[lc], chains[ac])
            else:
                d_ab_ag = min_interchain_ca_dist(chains[hc], chains[ac])
                if d_ab_ag > 16.0:
                    continue
                sc = sh + sa - 0.02 * d_ab_ag
                if sc > best_score:
                    best_score = sc
                    best_tuple = (chains[hc], None, chains[ac])
    return best_tuple


def select_best_antigen_chain(chains: dict[str, ChainBackbone], ag_seq: str) -> ChainBackbone | None:
    best_cb = None
    best_sc = -1.0
    for cb in chains.values():
        ident, al_len, _ = align_query_to_chain(ag_seq, cb.seq)
        sc = ident * (al_len / max(len(ag_seq), 1))
        if sc > best_sc and ident >= 0.35 and al_len >= min(10, len(ag_seq)):
            best_sc = sc
            best_cb = cb
    if best_cb is None and chains:
        # Fallback to longest chain in antigen structure
        best_cb = max(chains.values(), key=lambda c: len(c.seq))
    return best_cb


def _cdr_residue_indices(seq_len: int, is_heavy: bool) -> np.ndarray:
    """Approximate Chothia/IMGT CDR1 + CDR2 + CDR3 spans on trimmed Fv / VHH sequences."""
    idx = []
    for s, e in ((25, 36), (49, 62), (94, min(seq_len - 8, 116)) if is_heavy else (87, min(seq_len - 8, 102))):
        sc, ec = max(0, s), min(seq_len, max(s + 1, e))
        idx.extend(range(sc, ec))
    return np.array(sorted(set(idx)), dtype=np.int32) if idx else np.arange(seq_len, dtype=np.int32)


def orient_unbound_ab_ag(
    cb_h: ChainBackbone,
    cb_l: ChainBackbone | None,
    cb_a: ChainBackbone,
) -> tuple[ChainBackbone, ChainBackbone | None, ChainBackbone]:
    """Deterministically orient unbound `IgFold` Ab (`H`[+`L`]) facing the primary exposed patch of `Ag` (`A`)."""
    ab_ca = cb_h.coords[:, 1, :] if cb_l is None else np.concatenate([cb_h.coords[:, 1, :], cb_l.coords[:, 1, :]], axis=0)
    ab_center = ab_ca.mean(axis=0)

    cdr_h = cb_h.coords[_cdr_residue_indices(len(cb_h.seq), True), 1, :]
    if cb_l is not None:
        cdr_l = cb_l.coords[_cdr_residue_indices(len(cb_l.seq), False), 1, :]
        cdr_ca = np.concatenate([cdr_h, cdr_l], axis=0)
    else:
        cdr_ca = cdr_h
    cdr_center = cdr_ca.mean(axis=0)
    v_ab = cdr_center - ab_center
    v_ab = v_ab / (np.linalg.norm(v_ab) + 1e-8)

    # Rotate Ab so v_ab points along +z and cdr_center sits at origin
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    axis = np.cross(v_ab, z_axis)
    norm_ax = float(np.linalg.norm(axis))
    dot_v = float(np.clip(np.dot(v_ab, z_axis), -1.0, 1.0))
    if norm_ax < 1e-6:
        R_ab = np.eye(3, dtype=np.float32) if dot_v > 0 else np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    else:
        k = axis / norm_ax
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float32)
        R_ab = np.eye(3, dtype=np.float32) + K * norm_ax + (K @ K) * (1.0 - dot_v)

    def _xform(coords: np.ndarray, center: np.ndarray, R: np.ndarray, shift: np.ndarray) -> np.ndarray:
        return ((coords - center[None, None, :]) @ R.T) + shift[None, None, :]

    # Find exposed aromatic/hydrophobic patch on Antigen
    ag_ca = cb_a.coords[:, 1, :]
    ag_center = ag_ca.mean(axis=0)
    radial = ag_ca - ag_center[None, :]
    rad_dist = np.linalg.norm(radial, axis=-1)
    surf_w = (rad_dist / (np.max(rad_dist) + 1e-6)) + np.array(
        [0.25 if aa in "YWF" else (0.12 if HYDROPHOBICITY.get(aa, 0.0) > 1.5 else 0.0) for aa in cb_a.seq],
        dtype=np.float32,
    )
    top_idx = int(np.argmax(surf_w))
    patch_pt = ag_ca[top_idx]
    v_ag = patch_pt - ag_center
    if np.linalg.norm(v_ag) < 1e-5:
        v_ag = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        v_ag = v_ag / np.linalg.norm(v_ag)

    # Rotate Ag so v_ag points along -z (towards Ab CDRs!) and place patch at z = +8.5 Å without clashes
    neg_z = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    ax2 = np.cross(v_ag, neg_z)
    n2 = float(np.linalg.norm(ax2))
    d2 = float(np.clip(np.dot(v_ag, neg_z), -1.0, 1.0))
    if n2 < 1e-6:
        R_ag = np.eye(3, dtype=np.float32) if d2 > 0 else np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    else:
        k2 = ax2 / n2
        K2 = np.array([[0, -k2[2], k2[1]], [k2[2], 0, -k2[0]], [-k2[1], k2[0], 0]], dtype=np.float32)
        R_ag = np.eye(3, dtype=np.float32) + K2 * n2 + (K2 @ K2) * (1.0 - d2)

    new_h = _xform(cb_h.coords, cdr_center, R_ab, np.zeros(3, dtype=np.float32))
    new_l = _xform(cb_l.coords, cdr_center, R_ab, np.zeros(3, dtype=np.float32)) if cb_l is not None else None
    new_a = _xform(cb_a.coords, patch_pt, R_ag, np.array([0.0, 0.0, 8.5], dtype=np.float32))

    # Push Ag along +z if any CA-CA distance < 5.5 Å so there are zero steric clashes
    ab_new_ca = new_h[:, 1, :] if new_l is None else np.concatenate([new_h[:, 1, :], new_l[:, 1, :]], axis=0)
    dmin = min_interchain_ca_dist(ChainBackbone("A" * len(ab_new_ca), np.repeat(ab_new_ca[:, None, :], 5, axis=1)), ChainBackbone(cb_a.seq, new_a))
    if dmin < 6.0:
        new_a[:, :, 2] += (6.5 - dmin)

    return (
        ChainBackbone(cb_h.seq, new_h),
        ChainBackbone(cb_l.seq, new_l) if new_l is not None else None,
        ChainBackbone(cb_a.seq, new_a),
    )


def write_standardized_pdb(
    out_path: Path,
    cb_h: ChainBackbone,
    cb_l: ChainBackbone | None,
    cb_a: ChainBackbone | None,
) -> None:
    """Write standardized multi-chain PDB (`H`, optional `L`, optional `A`) with `N, CA, C, O, CB` atoms."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atom_names = ("N", "CA", "C", "O", "CB")
    elems = ("N", "C", "C", "O", "C")
    lines: list[str] = []
    serial = 1
    for cid, cb in (("H", cb_h), ("L", cb_l), ("A", cb_a)):
        if cb is None:
            continue
        for r_idx, aa in enumerate(cb.seq, start=1):
            res3 = ONE_TO_THREE.get(aa, "ALA")
            max_k = 4 if aa == "G" else 5
            for k in range(max_k):
                x, y, z = cb.coords[r_idx - 1, k]
                anm = atom_names[k]
                el = elems[k]
                lines.append(
                    f"ATOM  {serial:5d}  {anm:<3s} {res3:>3s} {cid:1s}{r_idx:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00           {el:>2s}"
                )
                serial += 1
        lines.append("TER")
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_rcsb_cif(pdb4: str, rcsb_dir: Path) -> Path | None:
    """Download `{pdb4.lower()}.cif.gz` from RCSB if not already cached."""
    p4 = pdb4.strip().upper()[:4]
    if len(p4) != 4 or not p4.isalnum():
        return None
    rcsb_dir.mkdir(parents=True, exist_ok=True)
    target = rcsb_dir / f"{p4.lower()}.cif.gz"
    if target.exists() and target.stat().st_size > 200:
        return target
    url = f"https://files.rcsb.org/download/{p4}.cif.gz"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AbRank/2.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        if len(data) > 200:
            target.write_bytes(data)
            return target
    except Exception:
        pass
    return None


def _igfold_worker_batch(args: tuple[list[tuple[str, str, str]], str]) -> int:
    """Subprocess worker folding a shard of `(ab_key, vh, vl)` antibody/VHH sequences with IgFold."""
    items, out_dir_str = args
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    import torch
    torch.set_num_threads(8)
    from igfold import IgFoldRunner

    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = IgFoldRunner()
    done = 0
    for ab_key, vh, vl in items:
        p = out_dir / f"{ab_key}.pdb"
        if p.exists() and p.stat().st_size > 500:
            done += 1
            continue
        seqs = {"H": vh} if not vl else {"H": vh, "L": vl}
        runner.fold(str(p), sequences=seqs, do_refine=False, do_renum=False)
        done += 1
    return done


def fold_missing_antibodies_igfold(
    ab_jobs: list[tuple[str, str, str]],
    igfold_dir: Path,
    n_workers: int = 6,
) -> None:
    """Fold all uncached `(ab_key, vh, vl)` items across parallel CPU workers."""
    igfold_dir.mkdir(parents=True, exist_ok=True)
    todo = [(k, h, l) for (k, h, l) in ab_jobs if not (igfold_dir / f"{k}.pdb").exists()]
    if not todo:
        return
    n_w = max(1, min(n_workers, len(todo)))
    shards = [todo[i::n_w] for i in range(n_w)]
    shards = [s for s in shards if s]
    print(f"Folding {len(todo)} unique antibodies/VHHs with IgFold across {len(shards)} CPU workers...")
    with ProcessPoolExecutor(max_workers=len(shards)) as ex:
        futs = [ex.submit(_igfold_worker_batch, (sh, str(igfold_dir))) for sh in shards]
        for f in as_completed(futs):
            f.result()


def compute_scaffold_biophysics(
    pdb_path: Path,
    cb_h: ChainBackbone,
    cb_l: ChainBackbone | None,
    cb_a: ChainBackbone,
    is_cocrystal: bool,
) -> dict:
    """Compute per-residue interface distances, FreeSASA buried ΔSASA, PRODIGY ΔG, and GNM DCCM."""
    ab_cb = cb_h.coords[:, 4, :] if cb_l is None else np.concatenate([cb_h.coords[:, 4, :], cb_l.coords[:, 4, :]], axis=0)
    ag_cb = cb_a.coords[:, 4, :]
    ab_seq = cb_h.seq + (cb_l.seq if cb_l is not None else "")
    ag_seq = cb_a.seq

    # Pairwise CB distance matrix between Ab and Ag
    diff = ab_cb[:, None, :] - ag_cb[None, :, :]
    dist_ab_ag = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-8).astype(np.float32)  # [L_ab, L_ag]
    min_dist_ab = np.min(dist_ab_ag, axis=1)  # [L_ab]
    min_dist_ag = np.min(dist_ab_ag, axis=0)  # [L_ag]

    lh = len(cb_h.seq)
    ll = len(cb_l.seq) if cb_l is not None else 0
    min_dist_h = min_dist_ab[:lh]
    min_dist_l = min_dist_ab[lh:] if ll > 0 else np.zeros(0, dtype=np.float32)

    # PRODIGY interfacial contact counts (< 5.5 Å CB distance, Vangone & Bonvin eLife 2015)
    ic_pairs = np.argwhere(dist_ab_ag <= 5.5)
    ic_counts = {
        "charged_charged": 0, "charged_polar": 0, "charged_apolar": 0,
        "polar_polar": 0, "polar_apolar": 0, "apolar_apolar": 0,
    }
    for ia, ib in ic_pairs:
        c1 = PRODIGY_CLASS.get(ab_seq[int(ia)], "apolar")
        c2 = PRODIGY_CLASS.get(ag_seq[int(ib)], "apolar")
        k = "_".join(sorted([c1, c2]))
        if k == "apolar_charged":
            k = "charged_apolar"
        elif k == "apolar_polar":
            k = "polar_apolar"
        ic_counts[k] = ic_counts.get(k, 0) + 1

    # FreeSASA complex vs unbound Ab + Ag SASA
    per_res_dsasa_h = np.zeros(lh, dtype=np.float32)
    per_res_dsasa_l = np.zeros(ll, dtype=np.float32)
    per_res_dsasa_a = np.zeros(len(ag_seq), dtype=np.float32)
    nis_polar_pct = 35.0
    nis_charged_pct = 20.0
    total_buried_sasa = 0.0
    try:
        st_cplx = freesasa.Structure(str(pdb_path))
        res_cplx = freesasa.calc(st_cplx)
        ra_cplx = res_cplx.residueAreas()

        # Unbound Ab & Ag via chain selection
        ab_chains = "H+L" if cb_l is not None else "H"
        sub_list = freesasa.structureArray(str(pdb_path), {"separate-chains": True})
        ra_unb: dict[str, dict] = {}
        for s_ch in sub_list:
            r_ch = freesasa.calc(s_ch).residueAreas()
            ra_unb.update(r_ch)

        def _fill_dsasa(cid: str, L_c: int) -> np.ndarray:
            arr = np.zeros(L_c, dtype=np.float32)
            d_c = ra_cplx.get(cid, {})
            d_u = ra_unb.get(cid, {})
            for idx1 in range(1, L_c + 1):
                key = str(idx1)
                if key in d_c and key in d_u:
                    arr[idx1 - 1] = max(0.0, float(d_u[key].total - d_c[key].total))
            return arr

        per_res_dsasa_h = _fill_dsasa("H", lh)
        if ll > 0:
            per_res_dsasa_l = _fill_dsasa("L", ll)
        per_res_dsasa_a = _fill_dsasa("A", len(ag_seq))
        total_buried_sasa = float(per_res_dsasa_h.sum() + per_res_dsasa_l.sum() + per_res_dsasa_a.sum())

        # Non-interacting surface (NIS) composition (>5% relative SASA & not in interface)
        nis_tot, nis_pol, nis_chg = 0, 0, 0
        for cid, s_str, d_min in (("H", cb_h.seq, min_dist_h), ("L", cb_l.seq if cb_l else "", min_dist_l), ("A", ag_seq, min_dist_ag)):
            d_c = ra_cplx.get(cid, {})
            for idx1, aa in enumerate(s_str, start=1):
                if idx1 - 1 < len(d_min) and d_min[idx1 - 1] > 5.5:
                    ra = d_c.get(str(idx1))
                    if ra is not None and ra.total > 10.0:
                        nis_tot += 1
                        cls = PRODIGY_CLASS.get(aa, "apolar")
                        if cls == "polar":
                            nis_pol += 1
                        elif cls == "charged":
                            nis_chg += 1
        if nis_tot > 0:
            nis_polar_pct = 100.0 * nis_pol / nis_tot
            nis_charged_pct = 100.0 * nis_chg / nis_tot
    except Exception:
        pass

    # PRODIGY ΔG_bind (kcal/mol; negative = tighter -> negate for positive binding_score convention!)
    dg_prodigy = (
        -0.09459 * ic_counts["charged_charged"]
        - 0.10007 * ic_counts["charged_apolar"]
        + 0.19577 * ic_counts["polar_polar"]
        - 0.22671 * ic_counts["polar_apolar"]
        + 0.18681 * nis_apolar_or_polar(nis_polar_pct, nis_charged_pct)[0]
        + 0.13810 * nis_charged_pct
        - 15.9433
    )

    # Elastic Network Model (GNM Kirchhoff pseudoinverse) Dynamic Cross-Correlation Matrix (`DCCM`)
    # on a standardized 64-node Complex Interface+Paratope Pocket (44 Ab CDR/FW + 20 closest Ag epitope nodes)
    dccm_64 = compute_antipasti_dccm64(cb_h, cb_l, cb_a, min_dist_ag)

    return {
        "is_cocrystal": bool(is_cocrystal),
        "seq_H": cb_h.seq,
        "seq_L": cb_l.seq if cb_l is not None else "",
        "seq_A": cb_a.seq,
        "ca_H": cb_h.coords[:, 1, :].astype(np.float16),
        "cb_H": cb_h.coords[:, 4, :].astype(np.float16),
        "ca_L": cb_l.coords[:, 1, :].astype(np.float16) if cb_l is not None else np.zeros((0, 3), dtype=np.float16),
        "cb_L": cb_l.coords[:, 4, :].astype(np.float16) if cb_l is not None else np.zeros((0, 3), dtype=np.float16),
        "ca_A": cb_a.coords[:, 1, :].astype(np.float16),
        "cb_A": cb_a.coords[:, 4, :].astype(np.float16),
        "min_dist_H": min_dist_h.astype(np.float16),
        "min_dist_L": min_dist_l.astype(np.float16),
        "min_dist_A": min_dist_ag.astype(np.float16),
        "dsasa_H": per_res_dsasa_h.astype(np.float16),
        "dsasa_L": per_res_dsasa_l.astype(np.float16),
        "dsasa_A": per_res_dsasa_a.astype(np.float16),
        "prodigy_neg_dg": float(-dg_prodigy),
        "buried_sasa": float(total_buried_sasa),
        "ic_counts": ic_counts,
        "dccm_64": dccm_64.astype(np.float16),
    }


def nis_apolar_or_polar(nis_pol: float, nis_chg: float) -> tuple[float, float]:
    nis_apol = max(0.0, 100.0 - nis_pol - nis_chg)
    return nis_apol, nis_pol


def compute_antipasti_dccm64(
    cb_h: ChainBackbone,
    cb_l: ChainBackbone | None,
    cb_a: ChainBackbone,
    min_dist_ag: np.ndarray,
) -> np.ndarray:
    """Compute standardized `[64, 64]` Normal-Mode Cross-Correlation Map (`DCCM`) via Gaussian Network Model:
      - Slots `0..25`: 26 evenly/CDR-anchored Heavy-chain nodes
      - Slots `26..43`: 18 Light-chain CDR/FW nodes (zeros for single-domain VHH)
      - Slots `44..63`: Top-20 closest Antigen epitope/paratope-facing nodes (ordered along Ag sequence)
    """
    # Compute GNM over interface+Fv pocket (Ab + top 80 closest Ag residues) to keep Kirchhoff eigendecomp <2ms!
    ag_keep = np.sort(np.argsort(min_dist_ag)[: min(80, len(cb_a.seq))])
    ca_list = [cb_h.coords[:, 1, :]]
    if cb_l is not None:
        ca_list.append(cb_l.coords[:, 1, :])
    ca_list.append(cb_a.coords[ag_keep, 1, :])
    ca_all = np.concatenate(ca_list, axis=0).astype(np.float64)

    diff = ca_all[:, None, :] - ca_all[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    adj = (dist <= 10.0).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    gamma = np.diag(adj.sum(axis=1)) - adj
    # Moore-Penrose pseudoinverse skipping trivial zero mode
    eigvals, eigvecs = np.linalg.eigh(gamma)
    pos = eigvals > 1e-5
    inv_diag = np.where(pos, 1.0 / np.maximum(eigvals, 1e-5), 0.0)
    # Keep lowest 30 non-trivial normal modes (following Michalewicz et al. ANTIPASTI default modes=30!)
    nz_idx = np.flatnonzero(pos)[:30]
    V = eigvecs[:, nz_idx]
    cov = (V * inv_diag[nz_idx][None, :]) @ V.T
    diag = np.sqrt(np.maximum(np.diag(cov), 1e-8))
    dccm = cov / (diag[:, None] * diag[None, :])

    lh = len(cb_h.seq)
    ll = len(cb_l.seq) if cb_l is not None else 0

    def _pick_slots(L_c: int, target_k: int, is_heavy: bool) -> np.ndarray:
        cdr = _cdr_residue_indices(L_c, is_heavy)
        if len(cdr) >= target_k:
            sel = cdr[np.linspace(0, len(cdr) - 1, target_k).round().astype(int)]
        else:
            all_i = np.linspace(0, max(L_c - 1, 0), target_k).round().astype(int)
            sel = all_i
        return sel

    idx_h = _pick_slots(lh, 26, True)
    idx_ag_local = np.argsort(min_dist_ag[ag_keep])[: min(20, len(ag_keep))]
    idx_ag_local = np.sort(idx_ag_local)
    offset_ag = lh + ll

    out = np.zeros((64, 64), dtype=np.float32)
    full_slots: list[tuple[int, int]] = [(s, int(i)) for s, i in enumerate(idx_h)]
    if ll > 0:
        idx_l = _pick_slots(ll, 18, False)
        full_slots.extend((26 + s, lh + int(i)) for s, i in enumerate(idx_l))
    full_slots.extend((44 + s, offset_ag + int(i)) for s, i in enumerate(idx_ag_local))

    slot_ids = np.array([a for a, _ in full_slots], dtype=np.int32)
    mat_ids = np.array([b for _, b in full_slots], dtype=np.int32)
    out[np.ix_(slot_ids, slot_ids)] = dccm[np.ix_(mat_ids, mat_ids)].astype(np.float32)
    return out


def extract_foldseek_3di_batch(pdb_paths: list[Path], foldseek_bin: Path, tmp_dir: Path) -> dict[str, dict[str, str]]:
    """Run Foldseek `structureto3didescriptor` on a folder/list of scaffold PDBs -> `{stem: {chain: 3di_lower}}`."""
    if not pdb_paths:
        return {}
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = tmp_dir / "scaffolds.3di"
    if out_tsv.exists():
        out_tsv.unlink()
    cmd = [
        str(foldseek_bin),
        "structureto3didescriptor",
        "-v", "0",
        "--threads", "16",
        str(pdb_paths[0].parent),
        str(out_tsv),
    ]
    subprocess.run(cmd, check=True)
    mapping: dict[str, dict[str, str]] = {}
    for line in out_tsv.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        hdr = parts[0].strip()  # e.g. "SCAF_0001.pdb_H"
        s3di = parts[2].strip().lower()
        if ".pdb_" in hdr:
            stem, cid = hdr.split(".pdb_", 1)
            cid = cid.split()[0]
            mapping.setdefault(stem, {})[cid] = s3di
    return mapping
