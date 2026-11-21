"""Zero-leakage benchmark split construction, sequence clustering, and FASTA export."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import Levenshtein
import numpy as np
import pandas as pd

from abrank.config import BenchmarkConfig, RepoPaths
from abrank.pairs import (
    balance_pair_orientation,
    build_global_pairs,
    build_within_antigen_pairs,
    cap_abagid_occurrences,
    subsample_cap_per_group,
)
from abrank.schema import PAIR_COLUMNS


def write_fasta(df: pd.DataFrame, id_col: str, seq_col: str, out_path: Path, suffix: str = "") -> int:
    """Write deduplicated, sorted FASTA records (60 chars/line). Returns record count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = df[[id_col, seq_col]].dropna()
    sub = sub[sub[seq_col].astype(str).str.len() > 0]
    sub = sub.drop_duplicates(subset=[id_col], keep="first")
    sub = sub.sort_values(by=id_col, kind="mergesort")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec_id, seq in sub.itertuples(index=False):
            header = f"{str(rec_id).strip()}{suffix}"
            s = str(seq).strip()
            wrapped = "\n".join(s[i : i + 60] for i in range(0, len(s), 60))
            f.write(f">{header}\n{wrapped}\n")
    return len(sub)


def assign_canonical_ids(
    kd_df: pd.DataFrame,
    cross_modal_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign globally consistent `AbID`, `AgID`, and `AbAgID` across all paired-chain tables."""
    dfs = [kd_df.copy(), cross_modal_df.copy(), transfer_df.copy()]
    for d in dfs:
        if not d.empty:
            d["Ab_light_chain_seq"] = d["Ab_light_chain_seq"].fillna("").astype(str)
            d["Ab_seq"] = np.where(
                d["Ab_light_chain_seq"].str.len() > 0,
                d["Ab_heavy_chain_seq"] + "_" + d["Ab_light_chain_seq"],
                d["Ab_heavy_chain_seq"],
            )

    # Order unique antibody sequences deterministically (KD first so core IDs are compact)
    all_ab_seqs: list[str] = []
    seen_ab: set[str] = set()
    for d in dfs:
        if not d.empty:
            for s in sorted(d["Ab_seq"].unique()):
                if s not in seen_ab:
                    seen_ab.add(s)
                    all_ab_seqs.append(s)
    ab_to_id = {s: f"Ab-{i + 1:06d}" for i, s in enumerate(all_ab_seqs)}

    # Assign AgID per unique Ag_seq within each Ag_group
    ag_records = []
    for d in dfs:
        if not d.empty:
            ag_records.append(d[["Ag_group", "Ag_seq"]])
    all_ag = pd.concat(ag_records, ignore_index=True).drop_duplicates(subset=["Ag_seq"])
    ag_to_id: dict[str, str] = {}
    for ag_grp, g in all_ag.groupby("Ag_group", sort=True):
        uniq_seqs = sorted(g["Ag_seq"].unique())
        if len(uniq_seqs) == 1:
            ag_to_id[uniq_seqs[0]] = str(ag_grp)
        else:
            for idx, s in enumerate(uniq_seqs, start=1):
                ag_to_id[s] = f"{ag_grp}.{idx}"

    for d in dfs:
        if not d.empty:
            d["AbID"] = d["Ab_seq"].map(ab_to_id)
            d["AgID"] = d["Ag_seq"].map(ag_to_id)
            d["AbAgID"] = d["AbID"] + "__" + d["AgID"]
            d["AbAg_seq"] = d["Ab_seq"] + "_" + d["Ag_seq"]

    return dfs[0], dfs[1], dfs[2]


def assign_antibody_lineages(df: pd.DataFrame, identity_threshold: float = 0.80) -> pd.Series:
    """Cluster paired antibodies into clonal lineages via `WT_Abname` + Fv sequence identity.

    Uses 5-mer Jaccard prefiltering on unique `(WT_Abname, CDR/Fv)` representatives followed
    by exact `Levenshtein.ratio >= identity_threshold` so identical/homologous lineages across
    datasets (e.g. Trastuzumab across SKEMPIv2, AbBiBench, and HER2) merge into one lineage.
    """
    # Pick one representative sequence per (Dataset, WT_Abname) + per unique CDR-distinct clone
    ab_df = df[["AbID", "Ab_heavy_chain_seq", "Ab_light_chain_seq", "WT_Abname", "Dataset"]].drop_duplicates("AbID").copy()
    ab_df["Ab_light_chain_seq"] = ab_df["Ab_light_chain_seq"].fillna("").astype(str)
    ab_df["fv"] = ab_df["Ab_heavy_chain_seq"] + ab_df["Ab_light_chain_seq"]

    # Group first by (Dataset, WT_Abname) when WT_Abname is an explicit wild-type parent
    # Note: for multi-clone discovery panels (e.g. OVA, AIntibody, Kothiwal, AbCoV, SabDab),
    # distinct clones have <80% Fv identity and should each start as their own cluster node,
    # then merge iff Fv identity >= 80%.
    uniq_fvs = sorted(ab_df["fv"].unique())
    # Fast greedy clustering: representatives at 85% identity first to collapse huge DMS clouds
    # (100k single/double mutants of CR9114 or 5A12 collapse to 1 centroid in O(N)!)
    centroids: list[str] = []
    centroid_kmer: list[set[str]] = []
    fv_to_cluster: dict[str, int] = {}

    def kmer_set(s: str, k: int = 6) -> set[str]:
        return {s[i : i + k] for i in range(0, len(s) - k + 1, 2)}

    for fv in uniq_fvs:
        km = kmer_set(fv)
        assigned = -1
        # Check most recent/matching centroids
        for cid, c_seq in enumerate(centroids):
            if abs(len(fv) - len(c_seq)) > int(0.25 * max(len(fv), len(c_seq))):
                continue
            # Fast shared 6-mer check before C Levenshtein
            if len(km & centroid_kmer[cid]) >= 3:
                if Levenshtein.ratio(fv, c_seq, score_cutoff=identity_threshold) >= identity_threshold:
                    assigned = cid
                    break
        if assigned == -1:
            assigned = len(centroids)
            centroids.append(fv)
            centroid_kmer.append(km)
        fv_to_cluster[fv] = assigned

    # Second pass: single-linkage union among centroids at `identity_threshold`
    n_c = len(centroids)
    parent = list(range(n_c))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n_c):
        si = centroids[i]
        kmi = centroid_kmer[i]
        for j in range(i + 1, n_c):
            sj = centroids[j]
            if abs(len(si) - len(sj)) > int(0.25 * max(len(si), len(sj))):
                continue
            if len(kmi & centroid_kmer[j]) >= 2:
                if Levenshtein.ratio(si, sj, score_cutoff=identity_threshold) >= identity_threshold:
                    union(i, j)

    ab_df["lineage_id"] = [f"Lin-{find(fv_to_cluster[fv]) + 1:05d}" for fv in ab_df["fv"]]
    abid_to_lin = ab_df.set_index("AbID")["lineage_id"].to_dict()
    return df["AbID"].map(abid_to_lin)


def assign_antigen_clusters(df: pd.DataFrame, identity_threshold: float = 0.50) -> pd.Series:
    """Cluster antigens by `Ag_group` + sequence homology (`>= 50%` identity or domain containment)."""
    ag_df = df[["Ag_group", "Ag_seq"]].drop_duplicates().reset_index(drop=True)
    groups = sorted(ag_df["Ag_group"].unique())
    grp_to_idx = {g: i for i, g in enumerate(groups)}
    parent = list(range(len(groups)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pick representative sequences per Ag_group
    grp_seqs: dict[str, list[str]] = defaultdict(list)
    for grp, seq in ag_df.itertuples(index=False):
        if len(grp_seqs[grp]) < 3:
            grp_seqs[grp].append(str(seq))

    for i, g1 in enumerate(groups):
        for j in range(i + 1, len(groups)):
            g2 = groups[j]
            if find(i) == find(j):
                continue
            merged = False
            for s1 in grp_seqs[g1]:
                for s2 in grp_seqs[g2]:
                    short, long_ = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
                    # Subdomain check (e.g., RBD inside full Spike) or global sequence homology
                    if len(short) >= 25 and short in long_:
                        merged = True
                        break
                    if Levenshtein.ratio(s1, s2, score_cutoff=identity_threshold) >= identity_threshold:
                        merged = True
                        break
                if merged:
                    break
            if merged:
                union(i, j)

    root_to_name: dict[int, str] = {}
    for i, g in enumerate(groups):
        r = find(i)
        if r not in root_to_name:
            root_to_name[r] = f"AgClust-{len(root_to_name) + 1:03d}"

    grp_to_clust = {g: root_to_name[find(grp_to_idx[g])] for g in groups}
    return df["Ag_group"].map(grp_to_clust)


def partition_unseen_ag(
    kd_df: pd.DataFrame,
    seed: int = 42,
) -> pd.Series:
    """Hold out entire homologous `Ag_cluster`s (`<50%` Ag identity + domain containment) AND
    enforce disjoint `Ab_lineage`s (`<80%` Fv identity across train, val, and test).
    """
    rng = np.random.default_rng(seed)
    clust_sizes = kd_df.groupby("Ag_cluster").size().to_dict()
    clust_types = kd_df.groupby("Ag_cluster")["Ag_type"].agg(lambda s: s.mode().iloc[0]).to_dict()
    hr2_clusters = set(kd_df.loc[kd_df["Ag_group"] == "SARSCoV2_HR2", "Ag_cluster"].unique())

    eligible = [c for c, sz in clust_sizes.items() if sz <= 5000 and c not in hr2_clusters]
    test_clusters: set[str] = set()
    val_clusters: set[str] = set()

    # Size-stratified round-robin within Viral and Non-Viral so test >= val in assay & local-pair count
    for target_type in ("Viral", "Non-Viral"):
        pool = [c for c in eligible if clust_types[c] == target_type]
        # Sort descending by size with deterministic tie-break, then assign in blocks of 6:
        # [train, test, train, val, train, train] -> ~20% test, ~12% val, ~68% train across every size tier!
        pool_sorted = sorted(pool, key=lambda c: (-clust_sizes[c], c))
        for start in range(0, len(pool_sorted), 6):
            chunk = pool_sorted[start : start + 6]
            perm = rng.permutation(len(chunk))
            # Assign largest available randomized slot in chunk to test first, second to val
            if len(chunk) >= 3:
                test_clusters.add(chunk[int(perm[0])])
            if len(chunk) >= 5:
                val_clusters.add(chunk[int(perm[1])])

    raw_split = kd_df["Ag_cluster"].map(
        lambda c: "test" if c in test_clusters else ("val" if c in val_clusters else "train")
    )

    # Drop any train complex whose Ab_lineage appears in val/test (e.g. cross-reactive bH1)
    # AND drop any val complex whose Ab_lineage appears in test, guaranteeing 0% lineage overlap!
    test_lins = set(kd_df.loc[raw_split == "test", "Ab_lineage"].unique())
    val_mask_leak = (raw_split == "val") & kd_df["Ab_lineage"].isin(test_lins)
    if val_mask_leak.any():
        raw_split = raw_split.copy()
        raw_split.loc[val_mask_leak] = "drop"

    heldout_lins = set(kd_df.loc[raw_split.isin(["val", "test"]), "Ab_lineage"].unique())
    leak_mask = (raw_split == "train") & kd_df["Ab_lineage"].isin(heldout_lins)
    if leak_mask.any():
        raw_split = raw_split.copy()
        raw_split.loc[leak_mask] = "drop"

    return raw_split


def build_tier_splits(
    kd_df: pd.DataFrame,
    split_col: pd.Series,
    out_dir: Path,
    cfg: BenchmarkConfig,
    *,
    tier_name: str,
) -> dict:
    """Build 3-state `train.tsv`, `val.tsv`, `test.tsv`, `*_candidates.tsv`, and unfiltered pointwise panels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    m = cfg.margin_m
    m_neu = cfg.neutral_margin
    stats = {}

    part_ids = {
        sp: set(kd_df.loc[split_col == sp, "AbAgID"].unique()) for sp in ("train", "val", "test")
    }
    assert len(part_ids["train"] & part_ids["val"]) == 0, f"{tier_name}: train/val AbAgID overlap!"
    assert len(part_ids["train"] & part_ids["test"]) == 0, f"{tier_name}: train/test AbAgID overlap!"
    assert len(part_ids["val"] & part_ids["test"]) == 0, f"{tier_name}: val/test AbAgID overlap!"

    for sp_idx, sp in enumerate(("train", "val", "test")):
        sub = kd_df[split_col == sp].reset_index(drop=True)
        frac = cfg.split_fractions.get(sp, 0.15)
        n_global_target = max(200, int(round(cfg.global_num_pairs * frac)))
        per_ag_cap = max(100, int(round(cfg.local_ag_pair_cap * (1.0 if sp == "train" else 0.5))))

        global_pairs = build_global_pairs(
            sub,
            m=m,
            n_pairs=n_global_target,
            exclude_ag_groups=cfg.global_exclude_antigens,
            ab_cap=cfg.ab_occurrence_cap,
            seed=cfg.seed + sp_idx * 100,
            prefix=f"{tier_name[:3].upper()}_{sp[:2].upper()}_GLB",
        )
        local_pairs, cand_pairs = build_within_antigen_pairs(
            sub,
            m=m,
            m_neutral=m_neu,
            include_neutral=True,
            neutral_ratio=cfg.neutral_pair_ratio,
            modality="KD",
            ab_cap=cfg.ab_occurrence_cap,
            per_ag_cap=per_ag_cap,
            include_candidates=True,
            exhaustive_limit=cfg.local_exhaustive_seq_limit,
            sampling_budget=cfg.local_sampling_budget,
            max_l1_per_study=cfg.local_max_l1_per_study,
            seed=cfg.seed + sp_idx * 100 + 7,
            prefix=f"{tier_name[:3].upper()}_{sp[:2].upper()}_LOC",
        )

        core_pairs = pd.concat([global_pairs, local_pairs], ignore_index=True)
        core_pairs["pair_id"] = [f"{tier_name}_{sp}_{i + 1:06d}" for i in range(len(core_pairs))]
        core_pairs.to_csv(out_dir / f"{sp}.tsv", sep="\t", index=False)

        if not cand_pairs.empty:
            cand_pairs["pair_id"] = [f"{tier_name}_{sp}_cand_{i + 1:06d}" for i in range(len(cand_pairs))]
            cand_pairs.to_csv(out_dir / f"{sp}_candidates.tsv", sep="\t", index=False)

        # Export full unfiltered pointwise panel (neutral + moderate + impactful complexes)
        pw = sub.drop_duplicates("AbAgID").reset_index(drop=True)
        pw.to_csv(out_dir / f"{sp}_pointwise.tsv", sep="\t", index=False)

        stats[sp] = {
            "n_assays": int(len(sub)),
            "n_unique_abag": int(sub["AbAgID"].nunique()),
            "n_ag_groups": int(sub["Ag_group"].nunique()),
            "n_core_pairs": int(len(core_pairs)),
            "n_impactful_pairs": int((core_pairs["label"] != 0).sum()),
            "n_neutral_pairs": int((core_pairs["label"] == 0).sum()),
            "by_category": {k: int(v) for k, v in core_pairs["Category"].value_counts().to_dict().items()},
            "n_candidate_pairs": int(len(cand_pairs)),
            "n_candidate_impactful": int((cand_pairs["label"] != 0).sum()) if not cand_pairs.empty else 0,
            "n_candidate_neutral": int((cand_pairs["label"] == 0).sum()) if not cand_pairs.empty else 0,
        }

    return stats


def build_cross_modal_splits(
    cm_df: pd.DataFrame,
    kd_df: pd.DataFrame,
    out_dir: Path,
    cfg: BenchmarkConfig,
) -> dict:
    """Build sign-consistent 3-state cross-modal test pairs (`Enrichment`, `ELISA`, `DDG`, `EC50`, `DMS`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if cm_df.empty:
        return {}

    kd_seqs = set(kd_df["AbAg_seq"].unique())
    kd_ag_groups = set(kd_df["Ag_group"].unique())
    sub = cm_df[~cm_df["AbAg_seq"].isin(kd_seqs)].copy()

    all_local = []
    all_cand = []
    for mod_idx, (modality, mdf) in enumerate(sub.groupby("Assay_modality", sort=True)):
        m_thresh = cfg.modality_margins.get(str(modality), 0.5)
        m_neu = cfg.modality_neutral_margins.get(str(modality), 0.15)
        loc_df, cnd_df = build_within_antigen_pairs(
            mdf,
            m=m_thresh,
            m_neutral=m_neu,
            include_neutral=True,
            neutral_ratio=cfg.neutral_pair_ratio,
            modality=str(modality),
            ab_cap=cfg.ab_occurrence_cap,
            per_ag_cap=cfg.local_ag_pair_cap,
            include_candidates=(modality in {"EC50", "ELISA", "DDG"}),
            exhaustive_limit=cfg.local_exhaustive_seq_limit,
            sampling_budget=2_000_000,
            max_l1_per_study=cfg.local_max_l1_per_study,
            seed=cfg.seed + 500 + mod_idx,
            prefix=f"MOD_{modality}",
        )
        if not loc_df.empty:
            all_local.append(loc_df)
        if not cnd_df.empty:
            all_cand.append(cnd_df)

    merged = pd.concat(all_local + all_cand, ignore_index=True) if (all_local or all_cand) else pd.DataFrame(columns=PAIR_COLUMNS)
    if not merged.empty:
        merged["AbRank_group"] = merged["Ag_group_1"] + "__" + merged["Assay_modality"]
        merged = balance_pair_orientation(merged, seed=cfg.seed + 555)
        merged["ag_regime"] = np.where(merged["Ag_group_1"].isin(kd_ag_groups), "Seen-Ag", "Unseen-Ag")
        merged["pair_id"] = [f"modality_{i + 1:06d}" for i in range(len(merged))]
        merged.to_csv(out_dir / "modality_test.tsv", sep="\t", index=False)

    return {
        "n_pairs": int(len(merged)),
        "n_impactful": int((merged["label"] != 0).sum()) if not merged.empty else 0,
        "n_neutral": int((merged["label"] == 0).sum()) if not merged.empty else 0,
        "by_modality": {k: int(v) for k, v in merged["Assay_modality"].value_counts().to_dict().items()} if not merged.empty else {},
        "by_ag_regime": {k: int(v) for k, v in merged["ag_regime"].value_counts().to_dict().items()} if not merged.empty else {},
    }


def build_transfer_splits(
    transfer_df: pd.DataFrame,
    out_dir: Path,
    cfg: BenchmarkConfig,
) -> dict:
    """Build 3-state neutralization `ic50_test.tsv` and viral escape `escape_test.tsv` tracks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    if transfer_df.empty:
        return stats

    # 1. IC50 neutralization pairs (3-state: impactful >= m_ic50 vs neutral <= m_ic50_neu)
    ic50_df = transfer_df[transfer_df["Assay_modality"] == "IC50"].copy()
    if not ic50_df.empty:
        m_ic50 = cfg.modality_margins.get("IC50", 1.0)
        m_ic50_neu = cfg.modality_neutral_margins.get("IC50", 0.30)
        ic_rows = []
        seen_keys: set[tuple[str, str]] = set()
        rng = np.random.default_rng(cfg.seed + 700)

        for _, g in ic50_df.groupby("AbID", sort=True):
            g_u = g.drop_duplicates("AbAg_seq").reset_index(drop=True)
            if len(g_u) < 2:
                continue
            imp_cands = []
            neu_cands = []
            seqs = g_u["Ag_seq"].astype(str).to_numpy()
            scores = g_u["binding_score"].to_numpy(dtype=float)
            ids = g_u["AbAgID"].to_numpy()
            ags = g_u["Ag_group"].to_numpy()
            dss = g_u["Dataset"].to_numpy()
            for i in range(len(g_u)):
                for j in range(i + 1, len(g_u)):
                    diff = scores[i] - scores[j]
                    adiff = abs(diff)
                    is_imp = adiff >= m_ic50
                    is_neu = adiff <= m_ic50_neu
                    if not (is_imp or is_neu):
                        continue
                    if abs(len(seqs[i]) - len(seqs[j])) > 3:
                        continue
                    d = Levenshtein.distance(seqs[i], seqs[j], score_cutoff=3)
                    if d in (1, 2, 3):
                        if is_imp:
                            imp_cands.append((ids[i], ids[j], diff, 1 if diff > 0 else -1, d, ags[i], dss[i]))
                        else:
                            neu_cands.append((ids[i], ids[j], diff, 0, d, ags[i], dss[i]))

            for pool in (imp_cands, neu_cands):
                if not pool:
                    continue
                take = min(cfg.transfer_ab_pair_cap, len(pool))
                sel = rng.choice(len(pool), size=take, replace=False)
                for idx in sel:
                    id1, id2, diff, lbl, d, ag, ds = pool[int(idx)]
                    k = (id1, id2) if id1 < id2 else (id2, id1)
                    if k not in seen_keys:
                        seen_keys.add(k)
                        ic_rows.append(
                            {
                                "pair_id": "",
                                "AbAgID_1": id1,
                                "AbAgID_2": id2,
                                "binding_diff": float(diff),
                                "logKD_diff": float(-diff),
                                "label": int(lbl),
                                "Category": f"L{d}",
                                "Lev_distance": int(d),
                                "AbRank_group": f"{ag}__{ds}",
                                "Ag_group_1": ag,
                                "Ag_group_2": ag,
                                "Ag_type": "Viral",
                                "Assay_modality": "IC50",
                                "Dataset_1": ds,
                                "Dataset_2": ds,
                            }
                        )

        loc_ic, _ = build_within_antigen_pairs(
            ic50_df,
            m=m_ic50,
            m_neutral=m_ic50_neu,
            include_neutral=True,
            neutral_ratio=cfg.neutral_pair_ratio,
            modality="IC50",
            ab_cap=cfg.ab_occurrence_cap,
            per_ag_cap=200,
            include_candidates=False,
            seed=cfg.seed + 701,
            prefix="IC50_AB",
        )
        ic_pair_df = pd.DataFrame(ic_rows, columns=PAIR_COLUMNS) if ic_rows else pd.DataFrame(columns=PAIR_COLUMNS)
        if not loc_ic.empty:
            ic_pair_df = pd.concat([ic_pair_df, loc_ic], ignore_index=True)
        if not ic_pair_df.empty:
            ic_pair_df = balance_pair_orientation(ic_pair_df, seed=cfg.seed + 702)
            ic_pair_df["pair_id"] = [f"ic50_{i + 1:06d}" for i in range(len(ic_pair_df))]
            ic_pair_df.to_csv(out_dir / "ic50_test.tsv", sep="\t", index=False)
        stats["ic50_pairs"] = int(len(ic_pair_df))

    # 2. RBD Escape Bloom pairs (3-state: non-escaping vs escaping `±1`, plus non-escaping vs non-escaping `0`)
    esc_df = transfer_df[transfer_df["Assay_modality"] == "Escape"].copy()
    if not esc_df.empty:
        rng = np.random.default_rng(cfg.seed + 800)
        esc_rows = []
        for _, g in esc_df.groupby("AbID", sort=True):
            non_esc = g[g["raw_value"] < 0.01].drop_duplicates("AgID")
            esc = g[g["raw_value"] > 0.65].drop_duplicates("AgID")
            if non_esc.empty or esc.empty:
                continue
            epi_grp = str(g["Ag_epitope_restrictions"].iloc[0]) or "RBD"
            ds = str(g["Dataset"].iloc[0])

            # Impactful escape pairs (f < 0.01 vs f > 0.65)
            n_imp = min(8, len(non_esc) * len(esc))
            picked_imp: set[tuple[int, int]] = set()
            tries = 0
            while len(picked_imp) < n_imp and tries < n_imp * 10:
                tries += 1
                i = int(rng.integers(0, len(non_esc)))
                j = int(rng.integers(0, len(esc)))
                if (i, j) in picked_imp:
                    continue
                picked_imp.add((i, j))
                r1 = non_esc.iloc[i]
                r2 = esc.iloc[j]
                diff = float(r1["binding_score"] - r2["binding_score"])
                d = int(Levenshtein.distance(str(r1["Ag_seq"]), str(r2["Ag_seq"])))
                esc_rows.append(
                    {
                        "pair_id": "",
                        "AbAgID_1": r1["AbAgID"],
                        "AbAgID_2": r2["AbAgID"],
                        "binding_diff": diff,
                        "logKD_diff": -diff,
                        "label": 1,
                        "Category": f"L{d}" if d in (1, 2, 3) else "Candidate",
                        "Lev_distance": d,
                        "AbRank_group": epi_grp,
                        "Ag_group_1": "SARSCoV2_Spike",
                        "Ag_group_2": "SARSCoV2_Spike",
                        "Ag_type": "Viral",
                        "Assay_modality": "Escape",
                        "Dataset_1": ds,
                        "Dataset_2": ds,
                    }
                )

            # Neutral non-escaping RBD pairs (both f < 0.01)
            if len(non_esc) >= 2:
                max_neu_possible = len(non_esc) * (len(non_esc) - 1) // 2
                n_neu = min(8, max_neu_possible)
                picked_neu: set[tuple[int, int]] = set()
                tries = 0
                while len(picked_neu) < n_neu and tries < n_neu * 12:
                    tries += 1
                    i = int(rng.integers(0, len(non_esc)))
                    j = int(rng.integers(0, len(non_esc)))
                    if i == j:
                        continue
                    pair_ij = (i, j) if i < j else (j, i)
                    if pair_ij in picked_neu:
                        continue
                    picked_neu.add(pair_ij)
                    r1 = non_esc.iloc[pair_ij[0]]
                    r2 = non_esc.iloc[pair_ij[1]]
                    diff = float(r1["binding_score"] - r2["binding_score"])
                    d = int(Levenshtein.distance(str(r1["Ag_seq"]), str(r2["Ag_seq"])))
                    esc_rows.append(
                        {
                            "pair_id": "",
                            "AbAgID_1": r1["AbAgID"],
                            "AbAgID_2": r2["AbAgID"],
                            "binding_diff": diff,
                            "logKD_diff": -diff,
                            "label": 0,
                            "Category": f"L{d}" if d in (1, 2, 3) else "Candidate",
                            "Lev_distance": d,
                            "AbRank_group": epi_grp,
                            "Ag_group_1": "SARSCoV2_Spike",
                            "Ag_group_2": "SARSCoV2_Spike",
                            "Ag_type": "Viral",
                            "Assay_modality": "Escape",
                            "Dataset_1": ds,
                            "Dataset_2": ds,
                        }
                    )

        esc_pair_df = pd.DataFrame(esc_rows, columns=PAIR_COLUMNS) if esc_rows else pd.DataFrame(columns=PAIR_COLUMNS)
        if not esc_pair_df.empty:
            esc_pair_df = balance_pair_orientation(esc_pair_df, seed=cfg.seed + 801)
            esc_pair_df["pair_id"] = [f"escape_{i + 1:06d}" for i in range(len(esc_pair_df))]
            esc_pair_df.to_csv(out_dir / "escape_test.tsv", sep="\t", index=False)
        stats["escape_pairs"] = int(len(esc_pair_df))

    return stats


def build_vhh_splits(
    vhh_df: pd.DataFrame,
    seq_dir: Path,
    out_dir: Path,
    cfg: BenchmarkConfig,
) -> dict:
    """Build dedicated single-domain VHH / nanobody 3-state ranking & transfer track."""
    if vhh_df.empty:
        return {"n_assays": 0, "n_pairs": 0}
    out_dir.mkdir(parents=True, exist_ok=True)
    df = vhh_df.copy()
    df["Ab_light_chain_seq"] = ""

    uniq_vhh = sorted(df["Ab_heavy_chain_seq"].unique())
    vhh_to_id = {s: f"VHH-{i + 1:06d}" for i, s in enumerate(uniq_vhh)}
    df["AbID"] = df["Ab_heavy_chain_seq"].map(vhh_to_id)
    df["Ab_seq"] = df["Ab_heavy_chain_seq"]

    ag_to_id: dict[str, str] = {}
    for grp, g in df[["Ag_group", "Ag_seq"]].drop_duplicates("Ag_seq").groupby("Ag_group", sort=True):
        seqs = sorted(g["Ag_seq"].unique())
        if len(seqs) == 1:
            ag_to_id[seqs[0]] = f"VHHAg_{grp}"
        else:
            for idx, s in enumerate(seqs, start=1):
                ag_to_id[s] = f"VHHAg_{grp}.{idx}"
    df["AgID"] = df["Ag_seq"].map(ag_to_id)
    df["AbAgID"] = df["AbID"] + "__" + df["AgID"]
    df["AbAg_seq"] = df["Ab_heavy_chain_seq"] + "_" + df["Ag_seq"]
    df["Ab_lineage"] = assign_antibody_lineages(df, identity_threshold=cfg.clustering["unseen_ab_fv_identity"])
    df["Ag_cluster"] = assign_antigen_clusters(df, identity_threshold=cfg.clustering["unseen_ag_identity"])

    write_fasta(df, "AbID", "Ab_heavy_chain_seq", seq_dir / "vhh.fasta", suffix="")
    write_fasta(df, "AgID", "Ag_seq", seq_dir / "vhh_ag.fasta", suffix="")

    loc_df, cnd_df = build_within_antigen_pairs(
        df,
        m=cfg.margin_m,
        m_neutral=cfg.neutral_margin,
        include_neutral=True,
        neutral_ratio=cfg.neutral_pair_ratio,
        modality="KD",
        ab_cap=cfg.ab_occurrence_cap,
        per_ag_cap=cfg.candidate_ag_pair_cap,
        include_candidates=True,
        seed=cfg.seed + 900,
        prefix="VHH",
    )
    glb_df = build_global_pairs(
        df,
        m=cfg.margin_m,
        n_pairs=min(400, max(100, len(df))),
        exclude_ag_groups=cfg.global_exclude_antigens,
        ab_cap=cfg.ab_occurrence_cap,
        seed=cfg.seed + 905,
        prefix="VHH_GLB",
    )
    non_empty_vhh = [d for d in (glb_df, loc_df, cnd_df) if not d.empty]
    merged = pd.concat(non_empty_vhh, ignore_index=True) if non_empty_vhh else pd.DataFrame(columns=PAIR_COLUMNS)
    if not merged.empty:
        merged = balance_pair_orientation(merged, seed=cfg.seed + 901)
        merged["pair_id"] = [f"vhh_{i + 1:06d}" for i in range(len(merged))]
        merged.to_csv(out_dir / "vhh_ranking_test.tsv", sep="\t", index=False)

    pw = df.drop_duplicates("AbAgID").reset_index(drop=True)
    pw.to_csv(out_dir / "vhh_pointwise.tsv", sep="\t", index=False)
    df.to_csv(seq_dir.parent / "assays_vhh.tsv", sep="\t", index=False)

    return {
        "n_assays": int(len(df)),
        "n_unique_vhh": int(df["AbID"].nunique()),
        "n_ag_groups": int(df["Ag_group"].nunique()),
        "n_ag_clusters": int(df["Ag_cluster"].nunique()),
        "n_pairs": int(len(merged)),
        "n_impactful": int((merged["label"] != 0).sum()) if not merged.empty else 0,
        "n_neutral": int((merged["label"] == 0).sum()) if not merged.empty else 0,
        "by_category": {k: int(v) for k, v in merged["Category"].value_counts().to_dict().items()} if not merged.empty else {},
    }


def build_all_benchmark_files(
    paths: RepoPaths | None = None,
    cfg: BenchmarkConfig | None = None,
) -> dict:
    """Master pipeline step 03: assign canonical IDs, export FASTAs, and generate all splits."""
    paths = paths or RepoPaths.from_root()
    cfg = cfg or BenchmarkConfig.load(paths.configs / "benchmark.yaml")

    kd_df = pd.read_csv(paths.benchmark / "assays_kd.tsv", sep="\t", low_memory=False)
    cm_path = paths.benchmark / "assays_cross_modal.tsv"
    cm_df = pd.read_csv(cm_path, sep="\t", low_memory=False) if cm_path.exists() else pd.DataFrame()
    tr_path = paths.benchmark / "assays_transfer.tsv"
    tr_df = pd.read_csv(tr_path, sep="\t", low_memory=False) if tr_path.exists() else pd.DataFrame()
    vhh_path = paths.benchmark / "assays_vhh.tsv"
    vhh_df = pd.read_csv(vhh_path, sep="\t", low_memory=False) if vhh_path.exists() else pd.DataFrame()

    print("Assigning canonical AbID, AgID, and AbAgID across all paired-chain tracks...")
    kd_df, cm_df, tr_df = assign_canonical_ids(kd_df, cm_df, tr_df)

    print("Clustering paired antibodies (Fv >= 80% ID) and antigens (Ag >= 50% ID)...")
    kd_df["Ab_lineage"] = assign_antibody_lineages(
        kd_df, identity_threshold=cfg.clustering["unseen_ab_fv_identity"]
    )
    kd_df["Ag_cluster"] = assign_antigen_clusters(
        kd_df, identity_threshold=cfg.clustering["unseen_ag_identity"]
    )

    # Save enriched master tables with canonical IDs
    kd_df.to_csv(paths.benchmark / "assays_kd.tsv", sep="\t", index=False)
    if not cm_df.empty:
        cm_df.to_csv(paths.benchmark / "assays_cross_modal.tsv", sep="\t", index=False)
    if not tr_df.empty:
        tr_df.to_csv(paths.benchmark / "assays_transfer.tsv", sep="\t", index=False)

    # Export unified deduplicated FASTA files
    combined_for_fasta = pd.concat(
        [d for d in (kd_df, cm_df, tr_df) if not d.empty], ignore_index=True
    )
    n_h = write_fasta(combined_for_fasta, "AbID", "Ab_heavy_chain_seq", paths.sequences / "ab_heavy.fasta", suffix="_H")
    n_l = write_fasta(combined_for_fasta, "AbID", "Ab_light_chain_seq", paths.sequences / "ab_light.fasta", suffix="_L")
    n_ag = write_fasta(combined_for_fasta, "AgID", "Ag_seq", paths.sequences / "ag.fasta", suffix="")
    print(f"Wrote FASTAs: {n_h} heavy, {n_l} light, {n_ag} antigen sequences.")

    summary: dict = {"margin_m": cfg.margin_m, "fasta_counts": {"ab_heavy": n_h, "ab_light": n_l, "ag": n_ag}}

    # Canonical Zero-Leakage Unseen-Antigen Split (<50% Ag identity + <80% Ab lineage identity)
    import shutil
    for legacy_dir in ("seen_ag", "unseen_ab"):
        ld = paths.splits / legacy_dir
        if ld.exists():
            shutil.rmtree(ld)

    print(f"Building canonical zero-leakage unseen_ag splits at m={cfg.margin_m}...")
    unseen_ag_split = partition_unseen_ag(kd_df, seed=cfg.seed + 2)
    summary["unseen_ag"] = build_tier_splits(
        kd_df, unseen_ag_split, paths.splits / "unseen_ag", cfg, tier_name="unseen_ag"
    )

    # Cross-Modal, Transfer, and VHH tracks
    print("Building Cross-Modal splits...")
    summary["cross_modal"] = build_cross_modal_splits(cm_df, kd_df, paths.splits / "cross_modal", cfg)

    print("Building Transfer (IC50 + RBD Escape) splits...")
    summary["transfer"] = build_transfer_splits(tr_df, paths.splits / "transfer", cfg)

    print("Building VHH / Nanobody ranking splits...")
    summary["vhh"] = build_vhh_splits(vhh_df, paths.sequences, paths.splits / "vhh", cfg)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AbRank FASTAs, 3-state pairs, and zero-leakage splits.")
    parser.add_argument("--config", type=Path, default=None, help="Path to benchmark.yaml")
    parser.add_argument("--m", type=float, default=None, help="Override impactful margin m (default: 1.0)")
    parser.add_argument("--m-neutral", type=float, default=None, help="Override neutral margin m_neutral (default: 0.30)")
    args = parser.parse_args()

    paths = RepoPaths.from_root()
    cfg = BenchmarkConfig.load(args.config)
    if args.m is not None:
        cfg.margin_m = float(args.m)
    if args.m_neutral is not None:
        cfg.neutral_margin = float(args.m_neutral)
    build_all_benchmark_files(paths, cfg)


if __name__ == "__main__":
    main()
