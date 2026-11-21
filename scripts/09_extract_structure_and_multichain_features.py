#!/usr/bin/env python3
"""Step 9: Extract and cache 3D structure-conditioned and multi-chain representations &
zero-shot log-likelihood tables across AbRank v2:

  1. `data/embeddings/inverse_folding_scaffolds.pt`:
     - `ESM-IF1` (`esm_if1_gvp4_t16_142M_UR50`, Hsu et al., ICML 2022): complex & unbound
       `[L, 20]` log-probabilities + `[L, 512]` GVP-Transformer 3D structural embeddings.
     - `AntiFold` (`oxpig/AntiFold`, Hummer et al., 2024): complex & unbound `[L, 20]`
       antibody/nanobody inverse-folding log-probabilities + `[L, 512]` embeddings.
     - `ProteinMPNN` (`v_48_020` + `soluble_v_48_020`, Dauparas et al., Science 2022):
       complex & unbound `[L, 20]` conditional/unconditional log-probabilities + `[L, 128]`
       3D encoder embeddings.
  2. `data/embeddings/saprot_650m.pt`:
     - `SaProt-650M` (`westlake-repl/SaProt_650M_PDB`, Su et al., ICLR 2024): Foldseek `3Di`
       structure-token + amino-acid embeddings (`D = 1280`) and 3Di-conditioned AA log-probs.
  3. `data/embeddings/mint_650m.pt`:
     - `MINT-650M` (`varunullanat2012/mint`, Ullanat et al., 2025): joint multi-chain
       `(VH, VL, Ag)` / `(VHH, Ag)` cross-chain self-attention embeddings (`D = 1280` per chain)
       and antigen-conditioned residue log-probabilities.
"""

from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from abrank.baselines.embeddings import seq_hash
from abrank.baselines.features import AA20 as AA_ORDER
from abrank.config import get_repo_root

warnings.filterwarnings("ignore")

AA_LIST = list(AA_ORDER)  # "ACDEFGHIKLMNPQRSTVWY"


def _if1_antifold_worker(args: tuple[list[tuple[str, str, bool]], str, str]) -> dict[str, dict]:
    """Worker running both `ESM-IF1` and `AntiFold` on a shard of `(sid, pdb_path, has_L)` scaffolds."""
    shard, scaf_dir_str, _ = args
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    import torch
    torch.set_num_threads(8)
    import antifold.esm
    from antifold.antiscripts import (
        dataset_dataloader_to_predictions_list,
        df_logits_to_logprobs,
        get_dataset_dataloader,
        load_model,
        predictions_list_to_df_logits_list,
    )

    m_if1 = load_model("ESM-IF1")
    m_af = load_model()

    def _run_df(model, df_in: pd.DataFrame, want_emb: bool) -> tuple[list[pd.DataFrame], list[np.ndarray]]:
        ds, dl = get_dataset_dataloader(df_in, scaf_dir_str, batch_size=1, custom_chain_mode=True, num_threads=0)
        sp_list, emb_list = dataset_dataloader_to_predictions_list(
            model, ds, dl, batch_size=1, extract_embeddings=want_emb
        )
        df_list = predictions_list_to_df_logits_list(sp_list, ds, dl)
        return df_list, emb_list

    out: dict[str, dict] = {}
    for sid, _, has_l in shard:
        # 1) Complex (H, [L], A)
        if has_l:
            row_cplx = {"pdb": sid, "Hchain": "H", "Lchain": "L", "chain3": "A"}
            row_ab = {"pdb": sid, "Hchain": "H", "Lchain": "L"}
        else:
            row_cplx = {"pdb": sid, "Hchain": "H", "Lchain": "A"}
            row_ab = {"pdb": sid, "Hchain": "H", "Lchain": np.nan}
        row_ag = {"pdb": sid, "Hchain": "A", "Lchain": np.nan}

        df_c = pd.DataFrame([row_cplx])
        df_ab = pd.DataFrame([row_ab])
        df_ag = pd.DataFrame([row_ag])

        # ESM-IF1
        dfl_if_c, emb_if_c = _run_df(m_if1, df_c, True)
        lp_if_c = df_logits_to_logprobs(dfl_if_c[0])[AA_LIST].to_numpy(dtype=np.float32)
        chains_c = dfl_if_c[0]["pdb_chain"].to_numpy()
        emb_c = emb_if_c[0].astype(np.float16)

        dfl_if_ab, _ = _run_df(m_if1, df_ab, False)
        lp_if_ab = df_logits_to_logprobs(dfl_if_ab[0])[AA_LIST].to_numpy(dtype=np.float32)
        dfl_if_ag, _ = _run_df(m_if1, df_ag, False)
        lp_if_ag = df_logits_to_logprobs(dfl_if_ag[0])[AA_LIST].to_numpy(dtype=np.float32)
        lp_if_unb = np.concatenate([lp_if_ab, lp_if_ag], axis=0)

        # AntiFold
        dfl_af_c, emb_af_c = _run_df(m_af, df_c, True)
        lp_af_c = df_logits_to_logprobs(dfl_af_c[0])[AA_LIST].to_numpy(dtype=np.float32)
        dfl_af_ab, _ = _run_df(m_af, df_ab, False)
        lp_af_ab = df_logits_to_logprobs(dfl_af_ab[0])[AA_LIST].to_numpy(dtype=np.float32)
        dfl_af_ag, _ = _run_df(m_af, df_ag, False)
        lp_af_ag = df_logits_to_logprobs(dfl_af_ag[0])[AA_LIST].to_numpy(dtype=np.float32)
        lp_af_unb = np.concatenate([lp_af_ab, lp_af_ag], axis=0)

        def _split_by_chain(arr: np.ndarray) -> dict[str, np.ndarray]:
            res: dict[str, np.ndarray] = {}
            for cid in ("H", "L", "A"):
                m = chains_c == cid
                if m.any():
                    res[cid] = arr[m].astype(np.float16)
            return res

        out[sid] = {
            "if1_lp_cplx": _split_by_chain(lp_if_c),
            "if1_lp_unb": _split_by_chain(lp_if_unb),
            "if1_emb_cplx": _split_by_chain(emb_c),
            "af_lp_cplx": _split_by_chain(lp_af_c),
            "af_lp_unb": _split_by_chain(lp_af_unb),
            "af_emb_cplx": _split_by_chain(emb_af_c[0]),
        }
    return out


def _protein_mpnn_worker(args: tuple[list[tuple[str, str, bool]], str]) -> dict[str, dict]:
    """Worker running official ProteinMPNN (`v_48_020` + `soluble_v_48_020` ensemble) on scaffolds."""
    shard, mpnn_repo = args
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    import torch
    torch.set_num_threads(8)
    sys.path.insert(0, mpnn_repo)
    from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

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
            for mdl in models:
                # Average unconditional + autoregressive teacher-forced log-probs for low-variance site LLR
                lp_u = mdl.unconditional_probs(X, mask, residue_idx, chain_encoding_all)[0, :, reorder]
                lp_c = mdl(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)[0, :, reorder]
                lp = 0.5 * (lp_u + lp_c)
                lp_sum = lp if lp_sum is None else (lp_sum + lp)
                if want_emb:
                    E, E_idx = mdl.features(X, mask, residue_idx, chain_encoding_all)
                    h_V = torch.zeros((E.shape[0], E.shape[1], 128))
                    h_E = mdl.W_e(E)
                    mask_attend = (mask.unsqueeze(-1) * torch.gather(mask.unsqueeze(1).expand(-1, E_idx.shape[1], -1), 2, E_idx))
                    for layer in mdl.encoder_layers:
                        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
                    hv = h_V[0].detach().cpu()
                    emb_sum = hv if emb_sum is None else (emb_sum + hv)
        lp_avg = (lp_sum / len(models)).numpy()
        ce = chain_encoding_all[0].numpy().astype(int)
        emb_avg = (emb_sum / len(models)).numpy() if want_emb else None
        return lp_avg, ce, emb_avg

    out: dict[str, dict] = {}
    for sid, pdb_path, has_l in shard:
        cplx_chains = ["H", "L", "A"] if has_l else ["H", "A"]
        ab_chains = ["H", "L"] if has_l else ["H"]
        lp_c, ce_c, emb_c = _score_pdb_chains(pdb_path, cplx_chains, want_emb=True)
        lp_ab, ce_ab, _ = _score_pdb_chains(pdb_path, ab_chains, want_emb=False)
        lp_ag, _, _ = _score_pdb_chains(pdb_path, ["A"], want_emb=False)

        # parse_PDB + tied_featurize sorts chains ALPHABETICALLY (`['A', 'H', 'L']`) with chain_encoding 1, 2, [3]!
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
        unb_lp_map["A"] = lp_ag.astype(np.float16)

        out[sid] = {
            "mpnn_lp_cplx": cplx_lp_map,
            "mpnn_lp_unb": unb_lp_map,
            "mpnn_emb_cplx": cplx_emb_map,
        }
    return out


def _saprot_worker(args: tuple[list[tuple[str, str, str]], int]) -> dict[str, dict]:
    """Worker running `SaProt_650M_PDB` (FP32 AVX2) on `(key, aa_seq, s3di_lower)` items."""
    items, n_threads = args
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    import torch
    torch.set_num_threads(n_threads)
    from transformers import EsmForMaskedLM, EsmTokenizer

    tok = EsmTokenizer.from_pretrained("westlake-repl/SaProt_650M_PDB")
    model = EsmForMaskedLM.from_pretrained("westlake-repl/SaProt_650M_PDB", torch_dtype=torch.float32).eval()
    vocab = tok.get_vocab()

    # Prebuild token ID lookup `(aa, 3di_char) -> vocab_id` so we marginalize/extract exact AA log-probs!
    s3di_alphabet = "acdefghiklmnpqrstvwy#"
    aa_3di_tok_ids: dict[str, dict[str, int]] = {}
    for s3 in s3di_alphabet:
        d = {}
        for aa in AA_LIST:
            tid = vocab.get(aa + s3, vocab.get(aa + "#"))
            if tid is not None:
                d[aa] = tid
        aa_3di_tok_ids[s3] = d

    out: dict[str, dict] = {}
    # Sort by length for minimal padding in mini-batches of 8
    items_sorted = sorted(items, key=lambda x: len(x[1]))
    bs = 8
    for i in range(0, len(items_sorted), bs):
        batch = items_sorted[i : i + bs]
        joint_strs = []
        norm_3dis = []
        for _, aa_s, s3_s in batch:
            L = len(aa_s)
            if len(s3_s) < L:
                s3_n = (s3_s + "#" * L)[:L]
            else:
                s3_n = s3_s[:L]
            s3_n = "".join(ch if ch in aa_3di_tok_ids else "#" for ch in s3_n.lower())
            norm_3dis.append(s3_n)
            joint_strs.append("".join(a + b for a, b in zip(aa_s, s3_n)))

        enc = tok(joint_strs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        with torch.no_grad():
            res = model(**enc, output_hidden_states=True)
            hid = res.hidden_states[-1]  # [B, T, 1280]
            logits = res.logits          # [B, T, V]

        for b_idx, (key, aa_s, _) in enumerate(batch):
            L = min(len(aa_s), hid.shape[1] - 2)
            h = hid[b_idx, 1 : 1 + L].numpy()  # [L, 1280]
            lg = logits[b_idx, 1 : 1 + L]      # [L, V]
            s3_n = norm_3dis[b_idx][:L]

            # Marginal log-prob across the 20 amino acids conditioned on each site's 3Di state!
            lp_20 = np.zeros((L, 20), dtype=np.float32)
            for pos in range(L):
                tmap = aa_3di_tok_ids[s3_n[pos]]
                tids = [tmap[a] for a in AA_LIST]
                sub_lg = lg[pos, tids]
                lp_20[pos] = F.log_softmax(sub_lg, dim=-1).numpy()

            cdr_s = min(max(L - 30, 0), 94)
            cdr_e = min(L, max(cdr_s + 4, 114))
            wt_idx = np.array([AA_ORDER.index(ch) for ch in aa_s[:L]], dtype=np.int64)
            pll = float(lp_20[np.arange(L), wt_idx].mean()) if L > 0 else 0.0

            out[key] = {
                "mean": h.mean(axis=0).astype(np.float16),
                "std": h.std(axis=0).astype(np.float16),
                "cdr3": h[cdr_s:cdr_e].mean(axis=0).astype(np.float16),
                "lp": lp_20.astype(np.float16),
                "pll": pll,
            }
    return out


def _mint_worker(args: tuple[list[tuple[str, str, str, str, bool]], str, str, int]) -> dict[str, dict]:
    """Worker running `MINT-650M` (FP32 AVX2) on `(abag_id, vh, vl, ag, save_per_res)` complexes."""
    items, mint_repo, mint_ckpt, n_threads = args
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    import torch
    torch.set_num_threads(n_threads)
    sys.path.insert(0, mint_repo)
    from mint.helpers.extract import CollateFn, MINTWrapper, load_config

    cfg = load_config(str(Path(mint_repo) / "data" / "esm2_t33_650M_UR50D.json"))
    wrapper = MINTWrapper(cfg, mint_ckpt, sep_chains=True, device="cpu")
    wrapper.model.float().eval()
    collate = CollateFn()
    alphabet = collate.alphabet
    aa_tok_ids = [alphabet.get_idx(a) for a in AA_LIST]

    def _pll(seq_s: str, lp_arr: np.ndarray) -> float:
        L_c = min(len(seq_s), len(lp_arr))
        if L_c == 0:
            return 0.0
        idx = np.array([AA_ORDER.index(ch) for ch in seq_s[:L_c]], dtype=np.int64)
        return float(lp_arr[:L_c][np.arange(L_c), idx].mean())

    out: dict[str, dict] = {}
    # Group by `bool(vl)` (3-chain paired Fv+Ag vs 2-chain VHH+Ag) and sort by total sequence length
    # so mini-batches of 8 share identical or near-identical chain lengths!
    groups_by_arity = [
        sorted([it for it in items if bool(it[2]) == has_vl], key=lambda x: len(x[1]) + len(x[2]) + min(len(x[3]), 650))
        for has_vl in (True, False)
    ]
    bs = 8
    for grp in groups_by_arity:
        for i in range(0, len(grp), bs):
            batch = grp[i : i + bs]
            has_vl = bool(batch[0][2])
            chain_list = [
                (vh, vl, ag[:650]) if has_vl else (vh, ag[:650])
                for _, vh, vl, ag, _ in batch
            ]
            chains, chain_ids = collate(chain_list)
            with torch.no_grad():
                res = wrapper.model(chains, chain_ids, repr_layers=[33])
                reps = res["representations"][33]  # [B, T, 1280]
                lgs = res["logits"]                # [B, T, 33]

            for b_idx, (abag_id, vh, vl, ag, save_per_res) in enumerate(batch):
                ag_t = ag[:650]
                rep = reps[b_idx]
                lg = lgs[b_idx]
                cids = chain_ids[b_idx].numpy() if chain_ids.ndim == 2 else chain_ids.numpy()
                toks = chains[b_idx].numpy()
                valid = (
                    (toks != wrapper.model.cls_idx)
                    & (toks != wrapper.model.eos_idx)
                    & (toks != wrapper.model.padding_idx)
                )

                m_h = (cids == 0) & valid
                if has_vl:
                    m_l = (cids == 1) & valid
                    m_a = (cids == 2) & valid
                else:
                    m_l = np.zeros_like(valid)
                    m_a = (cids == 1) & valid

                rep_h = rep[m_h].numpy()
                rep_l = rep[m_l].numpy() if has_vl else np.zeros((0, 1280), dtype=np.float32)
                rep_a = rep[m_a].numpy()

                lp_h = F.log_softmax(lg[m_h][:, aa_tok_ids], dim=-1).numpy()
                lp_l = F.log_softmax(lg[m_l][:, aa_tok_ids], dim=-1).numpy() if has_vl else np.zeros((0, 20), dtype=np.float32)
                lp_a = F.log_softmax(lg[m_a][:, aa_tok_ids], dim=-1).numpy()

                lh = len(rep_h)
                cdr_s = min(max(lh - 30, 0), 94)
                cdr_e = min(lh, max(cdr_s + 4, 114))

                entry = {
                    "h_mean": rep_h.mean(axis=0).astype(np.float16),
                    "h_cdr3": rep_h[cdr_s:cdr_e].mean(axis=0).astype(np.float16),
                    "l_mean": rep_l.mean(axis=0).astype(np.float16) if has_vl and len(rep_l) else np.zeros(1280, dtype=np.float16),
                    "a_mean": rep_a.mean(axis=0).astype(np.float16),
                    "a_std": rep_a.std(axis=0).astype(np.float16),
                    "pll_ab": _pll(vh, lp_h) + (_pll(vl, lp_l) if has_vl else 0.0),
                    "pll_ag": _pll(ag_t, lp_a),
                    "lp_H": lp_h.astype(np.float16),
                    "lp_L": lp_l.astype(np.float16),
                    "lp_A": lp_a[: min(len(lp_a), 650)].astype(np.float16),
                }
                if save_per_res:
                    entry["res_H"] = rep_h.astype(np.float16)
                    entry["res_L"] = rep_l.astype(np.float16)
                    entry["res_A"] = rep_a.astype(np.float16)
                out[abag_id] = entry
    return out


def main() -> None:
    t0 = time.time()
    root = get_repo_root()
    sdir = root / "data" / "structures"
    edir = root / "data" / "embeddings"
    edir.mkdir(parents=True, exist_ok=True)

    cc = torch.load(sdir / "complex_cache.pt", map_location="cpu", weights_only=False)
    scaffolds: dict[str, dict] = cc["scaffolds"]
    abag_to_scaf: dict[str, str] = cc["abag_to_scaffold"]

    # ------------------------------------------------------------------
    # 1. Run ESM-IF1, AntiFold, and ProteinMPNN across all 851 3D scaffolds
    # ------------------------------------------------------------------
    inv_path = edir / "inverse_folding_scaffolds.pt"
    if not inv_path.exists():
        print(f"Extracting ESM-IF1, AntiFold, and ProteinMPNN across {len(scaffolds)} scaffolds...")
        scaf_items = [
            (sid, str(root / bio["pdb_path"]), len(bio["seq_L"]) > 0)
            for sid, bio in sorted(scaffolds.items())
        ]
        n_w = 8
        shards = [scaf_items[i::n_w] for i in range(n_w)]

        if1_af_all: dict[str, dict] = {}
        with ProcessPoolExecutor(max_workers=n_w) as ex:
            futs = [ex.submit(_if1_antifold_worker, (sh, str(sdir / "scaffolds"), "")) for sh in shards if sh]
            for f in as_completed(futs):
                if1_af_all.update(f.result())
        print(f"  Completed ESM-IF1 & AntiFold on {len(if1_af_all)} scaffolds ({time.time() - t0:.1f}s).")

        mpnn_repo = str(Path.home() / ".cache" / "abrank" / "repos" / "ProteinMPNN")
        mpnn_all: dict[str, dict] = {}
        with ProcessPoolExecutor(max_workers=n_w) as ex:
            futs = [ex.submit(_protein_mpnn_worker, (sh, mpnn_repo)) for sh in shards if sh]
            for f in as_completed(futs):
                mpnn_all.update(f.result())
        print(f"  Completed ProteinMPNN on {len(mpnn_all)} scaffolds ({time.time() - t0:.1f}s).")

        merged_inv = {sid: {**if1_af_all[sid], **mpnn_all[sid]} for sid in scaffolds}
        torch.save(merged_inv, inv_path)
        print(f"  Saved {inv_path}")
    else:
        print(f"Using cached {inv_path}")

    # ------------------------------------------------------------------
    # Gather target complexes for SaProt-650M and MINT-650M
    # ------------------------------------------------------------------
    bdir = root / "data" / "benchmark"
    kd = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False).set_index("AbAgID")
    vhh = pd.read_csv(bdir / "assays_vhh.tsv", sep="\t", low_memory=False).set_index("AbAgID")
    vhh["Ab_light_chain_seq"] = ""
    cm = pd.read_csv(bdir / "assays_cross_modal.tsv", sep="\t", low_memory=False).set_index("AbAgID")
    cm["Ab_light_chain_seq"] = cm["Ab_light_chain_seq"].fillna("")

    tr_df = pd.read_csv(bdir / "splits" / "unseen_ag" / "train.tsv", sep="\t", low_memory=False)
    val_df = pd.read_csv(bdir / "splits" / "unseen_ag" / "val.tsv", sep="\t", low_memory=False)
    test_df = pd.read_csv(bdir / "splits" / "unseen_ag" / "test.tsv", sep="\t", low_memory=False)
    cand_df = pd.read_csv(bdir / "splits" / "unseen_ag" / "test_candidates.tsv", sep="\t", low_memory=False)
    pw_df = pd.read_csv(bdir / "splits" / "unseen_ag" / "test_pointwise.tsv", sep="\t", low_memory=False)
    vhh_df = pd.read_csv(bdir / "splits" / "vhh" / "vhh_ranking_test.tsv", sep="\t", low_memory=False)
    cm_df = pd.read_csv(bdir / "splits" / "cross_modal" / "modality_test.tsv", sep="\t", low_memory=False)

    # All evaluation complexes (100% exact!) + all non-AbBiBench train complexes + stratified AbBiBench + scaffold reps
    eval_kd_ids = (
        set(val_df["AbAgID_1"]) | set(val_df["AbAgID_2"])
        | set(test_df["AbAgID_1"]) | set(test_df["AbAgID_2"])
        | set(cand_df["AbAgID_1"]) | set(cand_df["AbAgID_2"])
        | set(pw_df["AbAgID"])
    )
    tr_ids_all = sorted(set(tr_df["AbAgID_1"]) | set(tr_df["AbAgID_2"]))
    tr_sub = kd.loc[tr_ids_all]
    tr_non_abbib = set(tr_sub.index[tr_sub["Dataset"] != "AbBibench"])
    # Stratified sample of AbBiBench complexes (150 per WT_Abname = ~1,200) + scaffold wild-type representatives
    rng = np.random.default_rng(42)
    tr_abbib_sample: set[str] = set()
    for _, g in tr_sub[tr_sub["Dataset"] == "AbBibench"].groupby("WT_Abname"):
        take = min(len(g), 150)
        tr_abbib_sample.update(rng.choice(g.index.to_numpy(), size=take, replace=False).tolist())

    exact_kd_ids = eval_kd_ids | tr_non_abbib | tr_abbib_sample
    exact_vhh_ids = set(vhh_df["AbAgID_1"]) | set(vhh_df["AbAgID_2"])
    # Strict unseen-Ag cross-modal (ELISA, DDG, EC50; NbBench uses scaffold rep + sampled variants)
    tr_ag_groups = set(tr_df["Ag_group_1"]) | set(tr_df["Ag_group_2"])
    cm_unseen = cm_df[
        (~cm_df["Ag_group_1"].isin(tr_ag_groups))
        & (~cm_df["Ag_group_2"].isin(tr_ag_groups))
        & (cm_df["Assay_modality"].isin(["ELISA", "DDG", "EC50", "Enrichment"]))
        & (cm_df["Dataset_1"] != "AbBibench")
        & (cm_df["Dataset_2"] != "AbBibench")
    ]
    cm_ids_used = set(cm_unseen["AbAgID_1"]) | set(cm_unseen["AbAgID_2"])
    cm_sub = cm.loc[cm.index.intersection(cm_ids_used)]
    exact_cm_ids = set(cm_sub.index[cm_sub["Dataset"] != "NbBench"])
    # Add 250 NbBench variants + scaffold rep
    nbb_ids = cm_sub.index[cm_sub["Dataset"] == "NbBench"].to_numpy()
    if len(nbb_ids) > 0:
        exact_cm_ids.update(rng.choice(nbb_ids, size=min(250, len(nbb_ids)), replace=False).tolist())

    combined_lookup = pd.concat(
        [
            kd.loc[kd.index.intersection(set(tr_ids_all) | eval_kd_ids)],
            vhh.loc[vhh.index.intersection(exact_vhh_ids)],
            cm.loc[cm.index.intersection(cm_ids_used)],
        ]
    )
    combined_lookup = combined_lookup[~combined_lookup.index.duplicated(keep="first")]

    # Identify one representative AbAgID per scaffold so scaffold-fallback is 100% populated!
    scaf_rep_abag: dict[str, str] = {}
    for abag_id in combined_lookup.index:
        sid = abag_to_scaf.get(abag_id)
        if sid and sid not in scaf_rep_abag:
            scaf_rep_abag[sid] = abag_id

    raw_all_ids = exact_kd_ids | exact_vhh_ids | exact_cm_ids | set(scaf_rep_abag.values())
    exact_all_ids = {i for i in raw_all_ids if i in abag_to_scaf and i in combined_lookup.index}
    scaf_rep_set = set(scaf_rep_abag.values())

    # ------------------------------------------------------------------
    # 2. Extract SaProt-650M (`data/embeddings/saprot_650m.pt`)
    # ------------------------------------------------------------------
    saprot_path = edir / "saprot_650m.pt"
    if not saprot_path.exists():
        t_sp = time.time()
        saprot_jobs: dict[str, tuple[str, str, str]] = {}
        for abag_id in exact_all_ids:
            if abag_id not in combined_lookup.index:
                continue
            r = combined_lookup.loc[abag_id]
            sid = abag_to_scaf[abag_id]
            bio = scaffolds[sid]
            for cid, seq_col, s3_key in (
                ("H", "Ab_heavy_chain_seq", "3di_H"),
                ("L", "Ab_light_chain_seq", "3di_L"),
                ("A", "Ag_seq", "3di_A"),
            ):
                s_aa = str(r[seq_col])
                if not s_aa:
                    continue
                s3 = str(bio[s3_key])
                ckey = hashlib.sha1(f"{s_aa}|{s3[:len(s_aa)]}".encode()).hexdigest()[:16]
                saprot_jobs[ckey] = (ckey, s_aa, s3)

        job_list = list(saprot_jobs.values())
        print(f"Extracting SaProt-650M across {len(job_list):,} unique `(seq, 3Di)` chains...")
        n_w = 8
        shards = [job_list[i::n_w] for i in range(n_w)]
        saprot_out: dict[str, dict] = {}
        with ProcessPoolExecutor(max_workers=n_w) as ex:
            futs = [ex.submit(_saprot_worker, (sh, 8)) for sh in shards if sh]
            for f in as_completed(futs):
                saprot_out.update(f.result())
        torch.save({"chains": saprot_out, "scaf_rep_abag": scaf_rep_abag}, saprot_path)
        print(f"  Saved {saprot_path} ({len(saprot_out):,} chains in {time.time() - t_sp:.1f}s).")
    else:
        print(f"Using cached {saprot_path}")

    # ------------------------------------------------------------------
    # 3. Extract MINT-650M (`data/embeddings/mint_650m.pt`)
    # ------------------------------------------------------------------
    mint_path = edir / "mint_650m.pt"
    if not mint_path.exists():
        t_mt = time.time()
        from huggingface_hub import hf_hub_download
        mint_ckpt = hf_hub_download(repo_id="varunullanat2012/mint", filename="mint.ckpt")
        mint_repo = str(Path.home() / ".cache" / "abrank" / "repos" / "mint")

        mint_jobs: list[tuple[str, str, str, str, bool]] = []
        for abag_id in sorted(exact_all_ids):
            if abag_id not in combined_lookup.index:
                continue
            r = combined_lookup.loc[abag_id]
            mint_jobs.append(
                (
                    abag_id,
                    str(r["Ab_heavy_chain_seq"]),
                    str(r["Ab_light_chain_seq"]),
                    str(r["Ag_seq"]),
                    abag_id in scaf_rep_set,
                )
            )
        print(f"Extracting MINT-650M multi-chain representations across {len(mint_jobs):,} complexes...")
        n_w = 8
        shards = [mint_jobs[i::n_w] for i in range(n_w)]
        mint_out: dict[str, dict] = {}
        with ProcessPoolExecutor(max_workers=n_w) as ex:
            futs = [ex.submit(_mint_worker, (sh, mint_repo, mint_ckpt, 8)) for sh in shards if sh]
            for f in as_completed(futs):
                mint_out.update(f.result())
        torch.save({"complexes": mint_out, "scaf_rep_abag": scaf_rep_abag}, mint_path)
        print(f"  Saved {mint_path} ({len(mint_out):,} complexes in {time.time() - t_mt:.1f}s).")
    else:
        print(f"Using cached {mint_path}")

    print(f"All 3D & multi-chain feature caches ready in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
