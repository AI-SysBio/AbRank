"""Parallel multi-worker PLM embedding & zero-shot log-likelihood extraction.

Supports:
  1. `ESMC-300M` (`Synthyra/ESMplusplus_small`) — single-chain VH, VL, and Antigen
     embeddings (`mean`, `std`, `cdr3_win`), unmasked sequence log-likelihood (`pll`),
     per-residue 20-AA log-probability matrices (`[L, 20]` float16), and exact
     masked-marginal mutation scoring.
  2. `AbLang2-Paired` (`ablang2-paired`) — paired `(VH, VL)` antibody representations
     (`paired_mean`, `vh_mean`, `vl_mean`, `vh_cdr3`), paired sequence log-likelihood,
     per-residue 20-AA log-probabilities (`vh_lp`, `vl_lp`), and exact paired
     masked-marginal mutation scoring.
  3. `ESM2-150M` (`facebook/esm2_t30_150M_UR50D`) — classic ESM-2 comparison baseline.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from abrank.baselines.features import AA20, AA_TO_IDX


def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode("ascii")).hexdigest()[:16]


def pair_hash(vh: str, vl: str) -> str:
    return hashlib.sha1(f"{vh}|{vl}".encode("ascii")).hexdigest()[:16]


def _hf_plm_worker(args: tuple) -> dict:
    """Worker process for HuggingFace MaskedLM models (ESMC-300M or ESM2)."""
    rank, model_name, trust_remote, items, max_len, threads_per_worker, device_str = args
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(threads_per_worker)

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = torch.device(device_str)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=trust_remote
    ).to(device)
    model.eval()
    tok = getattr(model, "tokenizer", None) or AutoTokenizer.from_pretrained(model_name)
    aa_token_ids = torch.tensor([tok.convert_tokens_to_ids(a) for a in AA20], dtype=torch.long, device=device)

    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    cdr3s: dict[str, np.ndarray] = {}
    plls: dict[str, float] = {}
    aa_lps: dict[str, np.ndarray] = {}

    # Items are already sorted by length; dynamic batch size based on sequence length
    idx = 0
    n_total = len(items)
    t_start = time.time()
    while idx < n_total:
        cur_len = min(len(items[idx][1]), max_len)
        if cur_len <= 160:
            bsz = 32
        elif cur_len <= 400:
            bsz = 12
        elif cur_len <= 800:
            bsz = 4
        else:
            bsz = 2

        chunk = items[idx : idx + bsz]
        idx += bsz
        keys = [k for k, _ in chunk]
        seqs = [s[:max_len] for _, s in chunk]

        enc = tok(seqs, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            hidden = out.last_hidden_state if getattr(out, "last_hidden_state", None) is not None else out.hidden_states[-1]
            logits = out.logits

        # Log-softmax strictly over the 20 standard amino acids for clean biological calibration
        logits_20 = logits.index_select(dim=-1, index=aa_token_ids)
        logp_20 = torch.log_softmax(logits_20.float(), dim=-1).cpu().numpy()
        hidden_np = hidden.float().cpu().numpy()

        for b_i, (k, s) in enumerate(zip(keys, seqs)):
            L = len(s)
            h_res = hidden_np[b_i, 1 : 1 + L]  # [L, D]
            lp_res = logp_20[b_i, 1 : 1 + L]   # [L, 20]

            means[k] = h_res.mean(axis=0).astype(np.float16)
            stds[k] = h_res.std(axis=0).astype(np.float16)
            w_start = max(0, L - 34)
            w_end = max(w_start + 1, L - 10) if L >= 25 else L
            cdr3s[k] = h_res[w_start:w_end].mean(axis=0).astype(np.float16)

            # Unmasked sequence log-likelihood over true residues
            res_aa_idx = [AA_TO_IDX[ch] for ch in s]
            pos_lp = lp_res[np.arange(L), res_aa_idx]
            plls[k] = float(pos_lp.mean())
            aa_lps[k] = lp_res.astype(np.float16)

        if rank == 0 and (idx % 320 < bsz or idx >= n_total):
            elapsed = time.time() - t_start
            print(f"  [{model_name} worker 0] {idx}/{n_total} ({elapsed:.1f}s)", flush=True)

    return {
        "means": means,
        "stds": stds,
        "cdr3s": cdr3s,
        "plls": plls,
        "aa_lps": aa_lps,
    }


def extract_hf_plm_cache(
    sequences: Sequence[str],
    cache_path: Path,
    *,
    model_name: str = "Synthyra/ESMplusplus_small",
    trust_remote_code: bool = True,
    max_len: int = 1536,
    n_workers: int = 16,
) -> dict:
    """Extract and cache single-chain PLM embeddings + per-position 20-AA log-probs."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if cache_path.exists():
        existing = torch.load(cache_path, map_location="cpu", weights_only=False)

    means = existing.get("means", {})
    stds = existing.get("stds", {})
    cdr3s = existing.get("cdr3s", {})
    plls = existing.get("plls", {})
    aa_lps = existing.get("aa_lps", {})

    uniq_seqs = sorted(set(sequences), key=len)
    missing = [(seq_hash(s), s) for s in uniq_seqs if seq_hash(s) not in means]
    if not missing:
        print(f"[Cache Hit] {model_name}: all {len(uniq_seqs)} sequences loaded from {cache_path.name}")
        return {"means": means, "stds": stds, "cdr3s": cdr3s, "plls": plls, "aa_lps": aa_lps}

    print(f"[Extracting] {model_name}: {len(missing)} new unique sequences (cached={len(means)})...")
    total_cpus = os.cpu_count() or 16
    if torch.cuda.is_available():
        worker_out = _hf_plm_worker((0, model_name, trust_remote_code, missing, max_len, 4, "cuda"))
        results = [worker_out]
    else:
        actual_workers = min(n_workers, max(1, len(missing) // 8), max(1, total_cpus // 2))
        threads_per = max(1, total_cpus // actual_workers)
        shards = [missing[r::actual_workers] for r in range(actual_workers)]
        worker_args = [
            (r, model_name, trust_remote_code, shards[r], max_len, threads_per, "cpu")
            for r in range(actual_workers)
        ]
        if actual_workers == 1:
            results = [_hf_plm_worker(worker_args[0])]
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=actual_workers) as pool:
                results = pool.map(_hf_plm_worker, worker_args)

    for res in results:
        means.update(res["means"])
        stds.update(res["stds"])
        cdr3s.update(res["cdr3s"])
        plls.update(res["plls"])
        aa_lps.update(res["aa_lps"])

    payload = {"means": means, "stds": stds, "cdr3s": cdr3s, "plls": plls, "aa_lps": aa_lps}
    torch.save(payload, cache_path)
    print(f"[Saved] {cache_path} ({len(means)} sequences)")
    return payload


def _ablang2_worker(args: tuple) -> dict:
    """Worker process for paired AbLang2 (`ablang2-paired`)."""
    rank, items, threads_per_worker, device_str = args
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    torch.set_num_threads(threads_per_worker)

    import ablang2

    device = torch.device(device_str)
    ab = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, device=device_str)
    ab.freeze()
    tok = ab.tokenizer
    aa_token_ids = torch.tensor([tok.aa_to_token[a] for a in AA20], dtype=torch.long, device=device)

    paired_means: dict[str, np.ndarray] = {}
    vh_means: dict[str, np.ndarray] = {}
    vl_means: dict[str, np.ndarray] = {}
    vh_cdr3s: dict[str, np.ndarray] = {}
    plls: dict[str, float] = {}
    vh_lps: dict[str, np.ndarray] = {}
    vl_lps: dict[str, np.ndarray] = {}

    bsz = 48
    n_total = len(items)
    t_start = time.time()
    for start in range(0, n_total, bsz):
        chunk = items[start : start + bsz]
        keys = [k for k, _, _ in chunk]
        formatted = [f"<{vh}>|<{vl}>" for _, vh, vl in chunk]
        tokens = tok(formatted, pad=True, w_extra_tkns=False, device=device)

        with torch.no_grad():
            rep = ab.AbLang.AbRep(tokens).last_hidden_states
            logits = ab.AbLang.AbHead(rep)

        logits_20 = logits.index_select(dim=-1, index=aa_token_ids)
        logp_20 = torch.log_softmax(logits_20.float(), dim=-1).cpu().numpy()
        rep_np = rep.float().cpu().numpy()

        for b_i, (k, vh, vl) in enumerate(chunk):
            lh = len(vh)
            ll = len(vl)
            # Token layout for `<{vh}>|<{vl}>`:
            # 0: `<`, 1..lh: vh, lh+1: `>`, lh+2: `|`, lh+3: `<`, lh+4..lh+3+ll: vl, lh+4+ll: `>`
            h_vh = rep_np[b_i, 1 : 1 + lh]
            h_vl = rep_np[b_i, 4 + lh : 4 + lh + ll]
            lp_vh = logp_20[b_i, 1 : 1 + lh]
            lp_vl = logp_20[b_i, 4 + lh : 4 + lh + ll]

            both = np.concatenate([h_vh, h_vl], axis=0)
            paired_means[k] = both.mean(axis=0).astype(np.float16)
            vh_means[k] = h_vh.mean(axis=0).astype(np.float16)
            vl_means[k] = h_vl.mean(axis=0).astype(np.float16)
            w_start = max(0, lh - 34)
            w_end = max(w_start + 1, lh - 10) if lh >= 25 else lh
            vh_cdr3s[k] = h_vh[w_start:w_end].mean(axis=0).astype(np.float16)

            vh_idx = [AA_TO_IDX[ch] for ch in vh]
            vl_idx = [AA_TO_IDX[ch] for ch in vl]
            pll_val = 0.5 * (
                float(lp_vh[np.arange(lh), vh_idx].mean())
                + float(lp_vl[np.arange(ll), vl_idx].mean())
            )
            plls[k] = pll_val
            vh_lps[k] = lp_vh.astype(np.float16)
            vl_lps[k] = lp_vl.astype(np.float16)

        done = min(start + bsz, n_total)
        if rank == 0 and (done % 480 < bsz or done >= n_total):
            elapsed = time.time() - t_start
            print(f"  [ablang2-paired worker 0] {done}/{n_total} ({elapsed:.1f}s)", flush=True)

    return {
        "paired_means": paired_means,
        "vh_means": vh_means,
        "vl_means": vl_means,
        "vh_cdr3s": vh_cdr3s,
        "plls": plls,
        "vh_lps": vh_lps,
        "vl_lps": vl_lps,
    }


def extract_ablang2_cache(
    antibody_pairs: Sequence[tuple[str, str]],
    cache_path: Path,
    *,
    n_workers: int = 16,
) -> dict:
    """Extract and cache paired AbLang2 representations & per-residue 20-AA log-probs."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if cache_path.exists():
        existing = torch.load(cache_path, map_location="cpu", weights_only=False)

    paired_means = existing.get("paired_means", {})
    vh_means = existing.get("vh_means", {})
    vl_means = existing.get("vl_means", {})
    vh_cdr3s = existing.get("vh_cdr3s", {})
    plls = existing.get("plls", {})
    vh_lps = existing.get("vh_lps", {})
    vl_lps = existing.get("vl_lps", {})

    uniq = sorted(set(antibody_pairs), key=lambda p: len(p[0]) + len(p[1]))
    missing = [(pair_hash(vh, vl), vh, vl) for vh, vl in uniq if pair_hash(vh, vl) not in paired_means]
    if not missing:
        print(f"[Cache Hit] ablang2-paired: all {len(uniq)} paired antibodies loaded from {cache_path.name}")
        return {
            "paired_means": paired_means,
            "vh_means": vh_means,
            "vl_means": vl_means,
            "vh_cdr3s": vh_cdr3s,
            "plls": plls,
            "vh_lps": vh_lps,
            "vl_lps": vl_lps,
        }

    # Ensure weights are downloaded once in main process before spawning workers
    import ablang2
    _ = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, device="cpu")

    print(f"[Extracting] ablang2-paired: {len(missing)} new paired antibodies (cached={len(paired_means)})...")
    total_cpus = os.cpu_count() or 16
    if torch.cuda.is_available():
        results = [_ablang2_worker((0, missing, 4, "cuda"))]
    else:
        actual_workers = min(n_workers, max(1, len(missing) // 16), max(1, total_cpus // 2))
        threads_per = max(1, total_cpus // actual_workers)
        shards = [missing[r::actual_workers] for r in range(actual_workers)]
        worker_args = [(r, shards[r], threads_per, "cpu") for r in range(actual_workers)]
        if actual_workers == 1:
            results = [_ablang2_worker(worker_args[0])]
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=actual_workers) as pool:
                results = pool.map(_ablang2_worker, worker_args)

    for res in results:
        paired_means.update(res["paired_means"])
        vh_means.update(res["vh_means"])
        vl_means.update(res["vl_means"])
        vh_cdr3s.update(res["vh_cdr3s"])
        plls.update(res["plls"])
        vh_lps.update(res["vh_lps"])
        vl_lps.update(res["vl_lps"])

    payload = {
        "paired_means": paired_means,
        "vh_means": vh_means,
        "vl_means": vl_means,
        "vh_cdr3s": vh_cdr3s,
        "plls": plls,
        "vh_lps": vh_lps,
        "vl_lps": vl_lps,
    }
    torch.save(payload, cache_path)
    print(f"[Saved] {cache_path} ({len(paired_means)} paired antibodies)")
    return payload
