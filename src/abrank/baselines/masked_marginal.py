r"""Exact masked-marginal zero-shot mutation scoring for ESMC-300M and AbLang2-Paired.

For any pair `(Complex 1, Complex 2)` where mutated chain(s) share equal length and
differ at $1 \le k \le 15$ positions $\mathcal{M}$, masks all $i \in \mathcal{M}$
simultaneously so neither mutant allele leaks into self-attention, and evaluates:
    $\sum_{i \in \mathcal{M}} [\log P(s^{(1)}_i \mid s_{-\mathcal{M}}) - \log P(s^{(2)}_i \mid s_{-\mathcal{M}})]$.
Strictly antisymmetric in `(1, 2)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from abrank.baselines.features import AA20, AA_TO_IDX


def _diff_positions(s1: str, s2: str, max_muts: int = 15) -> list[int]:
    if len(s1) != len(s2) or s1 == s2:
        return []
    pos = [i for i, (a, b) in enumerate(zip(s1, s2)) if a != b]
    return pos if 1 <= len(pos) <= max_muts else []


def compute_esmc_masked_marginals(
    pairs_df: pd.DataFrame,
    assays_lookup: pd.DataFrame,
    *,
    model_name: str = "Synthyra/ESMplusplus_small",
    max_len: int = 1536,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(ab_masked_llr, ag_masked_llr)` arrays aligned with `pairs_df`."""
    from transformers import AutoModelForMaskedLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    tok = model.tokenizer
    mask_id = tok.mask_token_id
    aa_ids = torch.tensor([tok.convert_tokens_to_ids(a) for a in AA20], dtype=torch.long, device=device)

    # Deduplicate masked forward passes by `(masked_token_key)`
    ab_scores = np.zeros(len(pairs_df), dtype=np.float32)
    ag_scores = np.zeros(len(pairs_df), dtype=np.float32)

    memo_lp: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}

    def score_chain_pair(s1: str, s2: str) -> float:
        s1_t, s2_t = s1[:max_len], s2[:max_len]
        pos = _diff_positions(s1_t, s2_t)
        if not pos:
            return 0.0
        # Build canonical masked sequence (identical whether derived from s1 or s2!)
        key_chars = list(s1_t)
        for p in pos:
            key_chars[p] = "*"
        mkey = ("".join(key_chars), tuple(pos))
        if mkey not in memo_lp:
            enc = tok([s1_t], return_tensors="pt")
            ids = enc["input_ids"].to(device)
            for p in pos:
                ids[0, 1 + p] = mask_id
            with torch.no_grad():
                logits = model(input_ids=ids).logits[0]
            logits_20 = logits.index_select(dim=-1, index=aa_ids)
            lp_20 = torch.log_softmax(logits_20.float(), dim=-1).cpu().numpy()
            memo_lp[mkey] = lp_20[[1 + p for p in pos]]  # [k, 20]

        lp_sub = memo_lp[mkey]
        total = 0.0
        for k_i, p in enumerate(pos):
            total += float(lp_sub[k_i, AA_TO_IDX[s1_t[p]]] - lp_sub[k_i, AA_TO_IDX[s2_t[p]]])
        return total

    r1 = assays_lookup.loc[pairs_df["AbAgID_1"]]
    r2 = assays_lookup.loc[pairs_df["AbAgID_2"]]

    for idx, (vh1, vl1, ag1, vh2, vl2, ag2) in enumerate(
        zip(
            r1["Ab_heavy_chain_seq"],
            r1["Ab_light_chain_seq"],
            r1["Ag_seq"],
            r2["Ab_heavy_chain_seq"],
            r2["Ab_light_chain_seq"],
            r2["Ag_seq"],
        )
    ):
        ab_scores[idx] = score_chain_pair(vh1, vh2) + score_chain_pair(vl1, vl2)
        ag_scores[idx] = score_chain_pair(ag1, ag2)

    return ab_scores, ag_scores


def compute_ablang2_masked_marginals(
    pairs_df: pd.DataFrame,
    assays_lookup: pd.DataFrame,
) -> np.ndarray:
    """Return paired AbLang2 masked-marginal LLR array aligned with `pairs_df`."""
    import ablang2

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    ab = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, device=device_str)
    ab.freeze()
    tok = ab.tokenizer
    aa_ids = torch.tensor([tok.aa_to_token[a] for a in AA20], dtype=torch.long, device=device)

    scores = np.zeros(len(pairs_df), dtype=np.float32)
    memo_lp: dict[str, np.ndarray] = {}

    r1 = assays_lookup.loc[pairs_df["AbAgID_1"]]
    r2 = assays_lookup.loc[pairs_df["AbAgID_2"]]

    for idx, (vh1, vl1, vh2, vl2) in enumerate(
        zip(
            r1["Ab_heavy_chain_seq"],
            r1["Ab_light_chain_seq"],
            r2["Ab_heavy_chain_seq"],
            r2["Ab_light_chain_seq"],
        )
    ):
        if len(vh1) != len(vh2) or len(vl1) != len(vl2):
            continue
        pos_h = _diff_positions(vh1, vh2)
        pos_l = _diff_positions(vl1, vl2)
        if not pos_h and not pos_l:
            continue
        if len(pos_h) + len(pos_l) > 15:
            continue

        vh_m = list(vh1)
        for p in pos_h:
            vh_m[p] = "*"
        vl_m = list(vl1)
        for p in pos_l:
            vl_m[p] = "*"
        masked_str = f"<{''.join(vh_m)}>|<{''.join(vl_m)}>"

        if masked_str not in memo_lp:
            tokens = tok([masked_str], pad=True, w_extra_tkns=False, device=device)
            with torch.no_grad():
                rep = ab.AbLang.AbRep(tokens).last_hidden_states
                logits = ab.AbLang.AbHead(rep)[0]
            logits_20 = logits.index_select(dim=-1, index=aa_ids)
            memo_lp[masked_str] = torch.log_softmax(logits_20.float(), dim=-1).cpu().numpy()

        lp_full = memo_lp[masked_str]
        lh = len(vh1)
        total = 0.0
        for p in pos_h:
            row_lp = lp_full[1 + p]
            total += float(row_lp[AA_TO_IDX[vh1[p]]] - row_lp[AA_TO_IDX[vh2[p]]])
        for p in pos_l:
            row_lp = lp_full[4 + lh + p]
            total += float(row_lp[AA_TO_IDX[vl1[p]]] - row_lp[AA_TO_IDX[vl2[p]]])
        scores[idx] = total

    return scores
