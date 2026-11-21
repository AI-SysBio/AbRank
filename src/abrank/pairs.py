"""Deterministic 3-state noise-buffered pair construction (`Global`, `L1`-`L3` Local,
`Candidate`, and Cross-Modal / Transfer tracks) with guaranteed within-group orientation balance.

Pairwise label specification (`label` in `{-1, 0, +1}`):
  - `+1` (Improved) : `binding_diff >= +m`
  - `0`  (Neutral)  : `|binding_diff| <= m_neutral`
  - `-1` (Depleted) : `binding_diff <= -m`
"""

from __future__ import annotations

from itertools import combinations
import Levenshtein
import numpy as np
import pandas as pd

from abrank.schema import PAIR_COLUMNS


def balance_pair_orientation(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Deterministically orient pairs 50/50 within each `(AbRank_group, Category)`.

    Guarantees:
      - Among impactful rows (`label != 0`), `+1` and `-1` are balanced 50/50 overall and
        every `(AbRank_group, Category)` with >= 2 impactful rows has both `+1` and `-1`.
      - Among neutral rows (`label == 0`), `(AbAgID_1, AbAgID_2)` ordering is randomized
        50/50 within each `(AbRank_group, Category)` while preserving `label == 0`.
    """
    if df.empty:
        return df.copy()

    out = df.reset_index(drop=True).copy()
    out["label"] = out["label"].astype(int)
    rng = np.random.default_rng(seed)

    # 1. Canonicalize impactful rows so Complex 1 is initially the stronger binder (+1),
    #    and canonicalize neutral rows lexicographically by (AbAgID_1, AbAgID_2).
    imp_neg = (out["label"] != 0) & (out["binding_diff"] < 0)
    if imp_neg.any():
        out = _swap_rows(out, imp_neg)
    out.loc[out["label"] != 0, "label"] = 1

    neu_swap_init = (out["label"] == 0) & (out["AbAgID_1"].astype(str) > out["AbAgID_2"].astype(str))
    if neu_swap_init.any():
        out = _swap_rows(out, neu_swap_init)

    # 2. Flip exact half of impactful rows AND exact half of neutral rows per (AbRank_group, Category)
    flip_mask = np.zeros(len(out), dtype=bool)
    grp_cols = ["AbRank_group", "Category"]

    for is_neu_flag in (False, True):
        sub_mask = (out["label"] == 0) if is_neu_flag else (out["label"] != 0)
        if not sub_mask.any():
            continue
        for _, idx in out.loc[sub_mask].groupby(grp_cols, sort=True).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            n = len(idx_arr)
            if n == 1:
                if rng.random() < 0.5:
                    flip_mask[idx_arr[0]] = True
            else:
                perm = rng.permutation(idx_arr)
                n_flip = n // 2 if (rng.random() < 0.5) else (n - n // 2)
                flip_mask[perm[:n_flip]] = True

    if flip_mask.any():
        out = _swap_rows(out, pd.Series(flip_mask, index=out.index))

    out["binding_diff"] = out["binding_diff"].round(4)
    out["logKD_diff"] = (-out["binding_diff"]).round(4)
    out["label"] = out["label"].astype(int)
    return out


def _swap_rows(df: pd.DataFrame, mask: pd.Series | np.ndarray) -> pd.DataFrame:
    out = df.copy()
    m = np.asarray(mask, dtype=bool)
    for col1, col2 in (
        ("AbAgID_1", "AbAgID_2"),
        ("Ag_group_1", "Ag_group_2"),
        ("Dataset_1", "Dataset_2"),
    ):
        if col1 in out.columns and col2 in out.columns:
            tmp = out.loc[m, col1].values.copy()
            out.loc[m, col1] = out.loc[m, col2].values
            out.loc[m, col2] = tmp
    out.loc[m, "binding_diff"] = -out.loc[m, "binding_diff"].values
    out.loc[m, "logKD_diff"] = -out.loc[m, "logKD_diff"].values
    out.loc[m, "label"] = -out.loc[m, "label"].astype(int).values
    return out


def cap_abagid_occurrences(df: pd.DataFrame, cap: int = 3, seed: int = 42) -> pd.DataFrame:
    """Ensure each `AbAgID` appears at most `cap` times across `AbAgID_1` and `AbAgID_2`."""
    if df.empty or cap <= 0:
        return df.copy()
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy().copy()
    rng.shuffle(idx)

    counts: dict[str, int] = {}
    keep: list[int] = []
    id1_arr = df["AbAgID_1"]
    id2_arr = df["AbAgID_2"]

    for i in idx:
        a = id1_arr.at[i]
        b = id2_arr.at[i]
        ca = counts.get(a, 0)
        cb = counts.get(b, 0)
        if ca < cap and cb < cap:
            keep.append(i)
            counts[a] = ca + 1
            counts[b] = cb + 1

    return df.loc[sorted(keep)].reset_index(drop=True)


def subsample_cap_per_group(
    df: pd.DataFrame,
    cap: int = 500,
    seed: int = 42,
    group_col: str = "AbRank_group",
) -> pd.DataFrame:
    """Subsample at most `cap` pairs per `group_col` deterministically."""
    if df.empty or cap <= 0:
        return df.copy()
    parts = []
    for i, (_, g) in enumerate(df.groupby(group_col, sort=True)):
        if len(g) > cap:
            parts.append(g.sample(n=cap, random_state=seed + i))
        else:
            parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def build_global_pairs(
    assays_df: pd.DataFrame,
    *,
    m: float = 1.0,
    n_pairs: int = 25000,
    exclude_ag_groups: list[str] | tuple[str, ...] = ("SARSCoV2_HR2",),
    ab_cap: int = 3,
    seed: int = 42,
    prefix: str = "GLB",
) -> pd.DataFrame:
    """Sample cross-antigen (`Ag_group_1 != Ag_group_2`) affinity-scale pairs within `assays_df`.

    Excludes short linear peptide antigens (`SARSCoV2_HR2`) so yeast-mating peptide apparent Kd
    values are never compared against soluble folded-protein SPR/BLI Kd.
    """
    sub = assays_df[~assays_df["Ag_group"].isin(set(exclude_ag_groups))].copy()
    sub = sub[np.isfinite(sub["binding_score"])].copy()
    if sub["Ag_group"].nunique() < 2 or n_pairs <= 0:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    rng = np.random.default_rng(seed)

    rep = (
        sub.sort_values("AbAgID", kind="mergesort")
        .drop_duplicates(subset=["Ag_group", "AbID"], keep="first")
        .reset_index(drop=True)
    )

    group_to_rows: dict[str, np.ndarray] = {}
    for grp, gdf in rep.groupby("Ag_group", sort=True):
        group_to_rows[grp] = gdf[
            ["AbAgID", "binding_score", "Ag_group", "Ag_type", "Dataset", "AbAg_seq"]
        ].to_numpy()

    groups = np.array(sorted(group_to_rows.keys()))
    if len(groups) < 2:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    weights = np.array([np.log(len(group_to_rows[g]) + 3.0) for g in groups], dtype=float)
    probs = weights / weights.sum()

    rows = []
    seen: set[tuple[str, str]] = set()
    id_counts: dict[str, int] = {}
    max_tries = max(n_pairs * 80, 50_000)
    tries = 0

    while len(rows) < n_pairs and tries < max_tries:
        tries += 1
        g1 = rng.choice(groups, p=probs)
        g2 = rng.choice(groups, p=probs)
        if g1 == g2:
            continue

        arr1 = group_to_rows[g1]
        arr2 = group_to_rows[g2]
        r1 = arr1[rng.integers(len(arr1))]
        r2 = arr2[rng.integers(len(arr2))]

        id1, s1, ag1, t1, ds1, seq1 = r1
        id2, s2, ag2, t2, ds2, seq2 = r2
        s1 = float(s1)
        s2 = float(s2)

        if not (np.isfinite(s1) and np.isfinite(s2)):
            continue
        diff = s1 - s2
        if abs(diff) < m:
            continue

        if ab_cap > 0 and (id_counts.get(id1, 0) >= ab_cap or id_counts.get(id2, 0) >= ab_cap):
            continue

        key = (id1, id2) if id1 < id2 else (id2, id1)
        if key in seen:
            continue
        seen.add(key)
        id_counts[id1] = id_counts.get(id1, 0) + 1
        id_counts[id2] = id_counts.get(id2, 0) + 1

        ag_type = t1 if t1 == t2 else "Mixed"
        rows.append(
            {
                "pair_id": f"{prefix}_{len(rows) + 1:06d}",
                "AbAgID_1": id1,
                "AbAgID_2": id2,
                "binding_diff": diff,
                "logKD_diff": -diff,
                "label": 1 if diff > 0 else -1,
                "Category": "Global",
                "Lev_distance": max(abs(len(str(seq1)) - len(str(seq2))), 50),
                "AbRank_group": "Global",
                "Ag_group_1": ag1,
                "Ag_group_2": ag2,
                "Ag_type": ag_type,
                "Assay_modality": "KD",
                "Dataset_1": ds1,
                "Dataset_2": ds2,
            }
        )

    out = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    out = balance_pair_orientation(out, seed=seed)
    out["pair_id"] = [f"{prefix}_{i + 1:06d}" for i in range(len(out))]
    return out


def build_within_antigen_pairs(
    assays_df: pd.DataFrame,
    *,
    m: float = 1.0,
    m_neutral: float = 0.30,
    include_neutral: bool = True,
    neutral_ratio: float = 1.0,
    modality: str = "KD",
    ab_cap: int = 3,
    per_ag_cap: int = 500,
    include_candidates: bool = False,
    exhaustive_limit: int = 2200,
    sampling_budget: int = 1_500_000,
    max_l1_per_study: int = 3_000,
    seed: int = 42,
    prefix: str = "LOC",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build within-antigen 3-state (`-1`, `0`, `+1`) local (`L1`-`L3`) and `Candidate` (`Lev > 3`) pairs.

    Pairs are restricted to the SAME `(Ag_group, Dataset)` so experimental calibration cancels out:
      - Impactful pairs (`|diff| >= m`): `label = +1` (`diff > 0`) or `-1` (`diff < 0`)
      - Neutral pairs (`|diff| <= m_neutral`): `label = 0`
    """
    df = assays_df[np.isfinite(assays_df["binding_score"])].copy()
    if df.empty:
        empty = pd.DataFrame(columns=PAIR_COLUMNS)
        return empty, empty

    rng = np.random.default_rng(seed)
    local_rows: list[dict] = []
    cand_rows: list[dict] = []
    want_neu = bool(include_neutral and m_neutral >= 0.0)

    for (ag_group, dataset), sub in df.groupby(["Ag_group", "Dataset"], sort=True):
        sub_u = (
            sub.sort_values("AbAgID", kind="mergesort")
            .drop_duplicates(subset=["AbAg_seq"], keep="first")
            .reset_index(drop=True)
        )
        n = len(sub_u)
        if n < 2:
            continue

        ids = sub_u["AbAgID"].to_numpy()
        scores = sub_u["binding_score"].to_numpy(dtype=float)
        ab_seqs = sub_u["Ab_seq"].astype(str).to_numpy()
        ag_seqs = sub_u["Ag_seq"].astype(str).to_numpy()
        ag_type = str(sub_u["Ag_type"].iloc[0])
        ab_lens = np.array([len(s) for s in ab_seqs], dtype=int)
        single_ag = int(sub_u["Ag_seq"].nunique()) == 1

        if n <= exhaustive_limit:
            for i, j in combinations(range(n), 2):
                diff = scores[i] - scores[j]
                adiff = abs(diff)
                is_imp = adiff >= m
                is_neu = want_neu and (adiff <= m_neutral)
                if not (is_imp or is_neu):
                    continue
                lbl = (1 if diff > 0 else -1) if is_imp else 0

                if single_ag or ag_seqs[i] == ag_seqs[j]:
                    len_diff = abs(int(ab_lens[i]) - int(ab_lens[j]))
                    d = Levenshtein.distance(ab_seqs[i], ab_seqs[j], score_cutoff=3) if len_diff <= 3 else 4
                    if d in (1, 2, 3):
                        local_rows.append(
                            _make_pair_row(
                                ids[i], ids[j], diff, lbl, f"L{d}", d, ag_group, ag_type, modality, dataset
                            )
                        )
                    elif include_candidates:
                        full_d = Levenshtein.distance(ab_seqs[i], ab_seqs[j])
                        cand_rows.append(
                            _make_pair_row(
                                ids[i], ids[j], diff, lbl, "Candidate", full_d, ag_group, ag_type, modality, dataset
                            )
                        )
                elif ab_seqs[i] == ab_seqs[j]:
                    len_diff_ag = abs(len(ag_seqs[i]) - len(ag_seqs[j]))
                    if len_diff_ag <= 3:
                        d_ag = Levenshtein.distance(ag_seqs[i], ag_seqs[j], score_cutoff=3)
                        if d_ag in (1, 2, 3):
                            local_rows.append(
                                _make_pair_row(
                                    ids[i], ids[j], diff, lbl, f"L{d_ag}", d_ag, ag_group, ag_type, modality, dataset
                                )
                            )
        else:
            # Large mutational libraries: sub-group by exact Ag_seq and index 35-AA anchor slices
            for _, ag_idx_series in sub_u.groupby("Ag_seq", sort=False).groups.items():
                ag_idx = np.asarray(list(ag_idx_series), dtype=int)
                if len(ag_idx) < 2:
                    continue

                buckets: dict[tuple[int, str], list[int]] = {}
                for idx_pos in ag_idx:
                    s = ab_seqs[idx_pos]
                    for slot, frag in (
                        (0, s[:35]),
                        (1, s[45:80]),
                        (2, s[90:125]),
                        (3, s[140:175]),
                        (4, s[-35:]),
                    ):
                        buckets.setdefault((slot, frag), []).append(int(idx_pos))

                seen_pairs: set[int] = set()
                imp_cnt = {1: 0, 2: 0, 3: 0}
                neu_cnt = {1: 0, 2: 0, 3: 0}
                bucket_keys = sorted(buckets.keys())
                rng.shuffle(bucket_keys)
                comparisons = 0

                for bkey in bucket_keys:
                    members = buckets[bkey]
                    m_len = len(members)
                    if m_len < 2:
                        continue
                    if m_len > 160:
                        members = list(rng.choice(members, size=160, replace=False))
                        m_len = 160
                    for a_i in range(m_len):
                        i = members[a_i]
                        si = ab_seqs[i]
                        li = int(ab_lens[i])
                        sci = scores[i]
                        for a_j in range(a_i + 1, m_len):
                            j = members[a_j]
                            diff = sci - scores[j]
                            adiff = abs(diff)
                            is_imp = adiff >= m
                            is_neu = want_neu and (adiff <= m_neutral)
                            if not (is_imp or is_neu):
                                continue
                            if abs(li - int(ab_lens[j])) > 3:
                                continue
                            ii, jj = (i, j) if i < j else (j, i)
                            pair_code = ii * n + jj
                            if pair_code in seen_pairs:
                                continue
                            seen_pairs.add(pair_code)
                            comparisons += 1
                            d = Levenshtein.distance(si, ab_seqs[j], score_cutoff=3)
                            if d in (1, 2, 3):
                                if is_imp and imp_cnt[d] < max_l1_per_study:
                                    imp_cnt[d] += 1
                                    local_rows.append(
                                        _make_pair_row(
                                            ids[i], ids[j], diff, 1 if diff > 0 else -1, f"L{d}", d, ag_group, ag_type, modality, dataset
                                        )
                                    )
                                elif is_neu and neu_cnt[d] < max_l1_per_study:
                                    neu_cnt[d] += 1
                                    local_rows.append(
                                        _make_pair_row(
                                            ids[i], ids[j], diff, 0, f"L{d}", d, ag_group, ag_type, modality, dataset
                                        )
                                    )
                        done_imp = all(imp_cnt[k] >= max_l1_per_study for k in (1, 2, 3))
                        done_neu = (not want_neu) or all(neu_cnt[k] >= max_l1_per_study for k in (1, 2, 3))
                        if done_imp and done_neu:
                            break
                    done_imp = all(imp_cnt[k] >= max_l1_per_study for k in (1, 2, 3))
                    done_neu = (not want_neu) or all(neu_cnt[k] >= max_l1_per_study for k in (1, 2, 3))
                    if comparisons >= sampling_budget or (done_imp and done_neu):
                        break

                if include_candidates and len(ag_idx) >= 2:
                    c_imp, c_neu, tries = 0, 0, 0
                    while (c_imp < 1500 or (want_neu and c_neu < 1500)) and tries < 20000:
                        tries += 1
                        i, j = rng.choice(ag_idx, size=2, replace=False)
                        diff = scores[i] - scores[j]
                        adiff = abs(diff)
                        is_imp = adiff >= m
                        is_neu = want_neu and (adiff <= m_neutral)
                        if not (is_imp or is_neu):
                            continue
                        if is_imp and c_imp >= 1500:
                            continue
                        if is_neu and c_neu >= 1500:
                            continue
                        full_d = Levenshtein.distance(ab_seqs[i], ab_seqs[j])
                        if full_d > 3:
                            if is_imp:
                                c_imp += 1
                                lbl = 1 if diff > 0 else -1
                            else:
                                c_neu += 1
                                lbl = 0
                            cand_rows.append(
                                _make_pair_row(
                                    ids[i], ids[j], diff, lbl, "Candidate", full_d, ag_group, ag_type, modality, dataset
                                )
                            )

            # Same-antibody antigen-side DMS when n > exhaustive_limit (e.g. AbAgym)
            if not single_ag:
                for _, ab_idx_series in sub_u.groupby("Ab_seq", sort=False).groups.items():
                    ab_idx = np.asarray(list(ab_idx_series), dtype=int)
                    if len(ab_idx) < 2:
                        continue
                    if len(ab_idx) > 400:
                        ab_idx = rng.choice(ab_idx, size=400, replace=False)
                    for a_i in range(len(ab_idx)):
                        i = int(ab_idx[a_i])
                        for a_j in range(a_i + 1, len(ab_idx)):
                            j = int(ab_idx[a_j])
                            diff = scores[i] - scores[j]
                            adiff = abs(diff)
                            is_imp = adiff >= m
                            is_neu = want_neu and (adiff <= m_neutral)
                            if not (is_imp or is_neu):
                                continue
                            if abs(len(ag_seqs[i]) - len(ag_seqs[j])) > 3:
                                continue
                            d_ag = Levenshtein.distance(ag_seqs[i], ag_seqs[j], score_cutoff=3)
                            if d_ag in (1, 2, 3):
                                lbl = (1 if diff > 0 else -1) if is_imp else 0
                                local_rows.append(
                                    _make_pair_row(
                                        ids[i], ids[j], diff, lbl, f"L{d_ag}", d_ag, ag_group, ag_type, modality, dataset
                                    )
                                )

    neu_cap = max(1, int(round(per_ag_cap * neutral_ratio))) if per_ag_cap > 0 else 0

    def _cap_3state(raw_df: pd.DataFrame, categories: list[str], seed_base: int) -> pd.DataFrame:
        if raw_df.empty:
            return pd.DataFrame(columns=PAIR_COLUMNS)
        parts: list[pd.DataFrame] = []
        for ci, cat in enumerate(categories):
            sub_cat = raw_df[raw_df["Category"] == cat]
            if sub_cat.empty:
                continue
            for si, (mask, g_cap) in enumerate(
                ((sub_cat["label"] != 0, per_ag_cap), (sub_cat["label"] == 0, neu_cap))
            ):
                s_df = sub_cat[mask]
                if s_df.empty:
                    continue
                if ab_cap > 0:
                    s_df = cap_abagid_occurrences(s_df, cap=ab_cap, seed=seed_base + ci * 10 + si)
                if g_cap > 0:
                    s_df = subsample_cap_per_group(s_df, cap=g_cap, seed=seed_base + 50 + ci * 10 + si)
                parts.append(s_df)
        if not parts:
            return pd.DataFrame(columns=PAIR_COLUMNS)
        merged = pd.concat(parts, ignore_index=True)
        return balance_pair_orientation(merged, seed=seed_base + 99)

    local_df = _cap_3state(
        pd.DataFrame(local_rows, columns=PAIR_COLUMNS) if local_rows else pd.DataFrame(columns=PAIR_COLUMNS),
        ["L1", "L2", "L3"],
        seed,
    )
    if not local_df.empty:
        local_df["pair_id"] = [f"{prefix}_{i + 1:06d}" for i in range(len(local_df))]

    cand_df = _cap_3state(
        pd.DataFrame(cand_rows, columns=PAIR_COLUMNS) if cand_rows else pd.DataFrame(columns=PAIR_COLUMNS),
        ["Candidate"],
        seed + 200,
    )
    if not cand_df.empty:
        cand_df["pair_id"] = [f"CND_{i + 1:06d}" for i in range(len(cand_df))]

    return local_df, cand_df


def _make_pair_row(
    id1: str,
    id2: str,
    diff: float,
    label: int,
    category: str,
    lev_dist: int,
    ag_group: str,
    ag_type: str,
    modality: str,
    dataset: str,
) -> dict:
    return {
        "pair_id": "",
        "AbAgID_1": id1,
        "AbAgID_2": id2,
        "binding_diff": float(diff),
        "logKD_diff": float(-diff),
        "label": int(label),
        "Category": category,
        "Lev_distance": int(lev_dist),
        "AbRank_group": ag_group,
        "Ag_group_1": ag_group,
        "Ag_group_2": ag_group,
        "Ag_type": ag_type,
        "Assay_modality": modality,
        "Dataset_1": dataset,
        "Dataset_2": dataset,
    }
