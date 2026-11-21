#!/usr/bin/env python3
"""Step 5: Extract & cache PLM embeddings + zero-shot residue log-probabilities
(`ESMC-300M`, `AbLang2-Paired`, and `ESM2-150M`) for the `unseen_ag` benchmark
and zero-shot cross-modal unseen-antigen transfer pairs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from abrank.config import get_repo_root
from abrank.baselines.embeddings import extract_ablang2_cache, extract_hf_plm_cache


def collect_target_sequences(
    root: Path,
    include_cross_modal: bool = True,
    include_all_transfer: bool = True,
    include_vhh: bool = True,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Collect all unique single chains (VH/VHH, VL, Ag) and paired (VH, VL) antibodies."""
    bdir = root / "data" / "benchmark"
    kd = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False).drop_duplicates("AbAgID").set_index("AbAgID")
    kd["Ab_light_chain_seq"] = kd["Ab_light_chain_seq"].fillna("").astype(str)

    abag_ids: set[str] = set()
    tr_ags: set[str] = set()
    for split in ("train", "val", "test", "test_candidates"):
        df = pd.read_csv(bdir / "splits" / "unseen_ag" / f"{split}.tsv", sep="\t", low_memory=False)
        abag_ids.update(df["AbAgID_1"])
        abag_ids.update(df["AbAgID_2"])
        if split == "train":
            tr_ags.update(df["Ag_group_1"])
            tr_ags.update(df["Ag_group_2"])

    for pw_split in ("val_pointwise", "test_pointwise"):
        pw_path = bdir / "splits" / "unseen_ag" / f"{pw_split}.tsv"
        if pw_path.exists():
            pw_df = pd.read_csv(pw_path, sep="\t", low_memory=False)
            abag_ids.update(pw_df["AbAgID"])

    if include_all_transfer:
        skempi_ids = set(kd.index[kd["Dataset"] == "SKEMPIv2"])
        abag_ids.update(skempi_ids)

    sub_kd = kd.loc[list(abag_ids)]
    single_chains = set(sub_kd["Ab_heavy_chain_seq"]).union(sub_kd["Ab_light_chain_seq"]).union(sub_kd["Ag_seq"])
    ab_pairs = {(h, l) for h, l in zip(sub_kd["Ab_heavy_chain_seq"], sub_kd["Ab_light_chain_seq"]) if h and l}

    if include_cross_modal and (bdir / "splits" / "cross_modal" / "modality_test.tsv").exists():
        cm_pairs = pd.read_csv(bdir / "splits" / "cross_modal" / "modality_test.tsv", sep="\t", low_memory=False)
        if include_all_transfer:
            cm_sub = cm_pairs
        else:
            cm_sub = cm_pairs[
                (~cm_pairs["Ag_group_1"].isin(tr_ags))
                & (~cm_pairs["Ag_group_2"].isin(tr_ags))
                & (cm_pairs["Assay_modality"].isin(["Enrichment", "ELISA", "DDG", "EC50"]))
            ]
        cm_ids = list(set(cm_sub["AbAgID_1"]).union(cm_sub["AbAgID_2"]))
        cma = pd.read_csv(bdir / "assays_cross_modal.tsv", sep="\t", low_memory=False).drop_duplicates("AbAgID").set_index("AbAgID")
        cma["Ab_light_chain_seq"] = cma["Ab_light_chain_seq"].fillna("").astype(str)
        sub_cm = cma.loc[cm_ids]
        single_chains.update(sub_cm["Ab_heavy_chain_seq"])
        single_chains.update(sub_cm["Ab_light_chain_seq"])
        single_chains.update(sub_cm["Ag_seq"])
        ab_pairs.update((h, l) for h, l in zip(sub_cm["Ab_heavy_chain_seq"], sub_cm["Ab_light_chain_seq"]) if h and l)

    if include_all_transfer and (bdir / "assays_transfer.tsv").exists():
        tfa = pd.read_csv(bdir / "assays_transfer.tsv", sep="\t", low_memory=False).drop_duplicates("AbAgID").set_index("AbAgID")
        tfa["Ab_light_chain_seq"] = tfa["Ab_light_chain_seq"].fillna("").astype(str)
        tf_ids: set[str] = set()
        for fname in ("ic50_test.tsv", "escape_test.tsv"):
            fpath = bdir / "splits" / "transfer" / fname
            if fpath.exists():
                tdf = pd.read_csv(fpath, sep="\t", low_memory=False)
                tf_ids.update(tdf["AbAgID_1"])
                tf_ids.update(tdf["AbAgID_2"])
        if tf_ids:
            sub_tf = tfa.loc[list(tf_ids)]
            single_chains.update(sub_tf["Ab_heavy_chain_seq"])
            single_chains.update(sub_tf["Ab_light_chain_seq"])
            single_chains.update(sub_tf["Ag_seq"])
            ab_pairs.update((h, l) for h, l in zip(sub_tf["Ab_heavy_chain_seq"], sub_tf["Ab_light_chain_seq"]) if h and l)

    if include_vhh and (bdir / "assays_vhh.tsv").exists():
        vhha = pd.read_csv(bdir / "assays_vhh.tsv", sep="\t", low_memory=False)
        single_chains.update(vhha["Ab_heavy_chain_seq"].dropna().astype(str))
        single_chains.update(vhha["Ag_seq"].dropna().astype(str))

    single_chains.discard("")
    return sorted(single_chains), sorted(ab_pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cached PLM embeddings for AbRank baselines.")
    parser.add_argument("--workers", type=int, default=16, help="Number of CPU worker processes")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ablang2", "esmc", "esm2"],
        choices=["ablang2", "esmc", "esm2"],
        help="PLMs to extract",
    )
    parser.add_argument(
        "--unseen-ag-only",
        action="store_true",
        help="Extract only unseen_ag + unseen-antigen cross-modal sequences",
    )
    args = parser.parse_args()

    root = get_repo_root()
    emb_dir = root / "data" / "embeddings"
    single_chains, ab_pairs = collect_target_sequences(
        root,
        include_cross_modal=True,
        include_all_transfer=not args.unseen_ag_only,
    )
    print(f"Target inventory: {len(single_chains):,} unique single chains, {len(ab_pairs):,} unique paired antibodies.")

    if "ablang2" in args.models:
        extract_ablang2_cache(ab_pairs, emb_dir / "ablang2_paired.pt", n_workers=args.workers)

    if "esmc" in args.models:
        extract_hf_plm_cache(
            single_chains,
            emb_dir / "esmc_300m.pt",
            model_name="Synthyra/ESMplusplus_small",
            trust_remote_code=True,
            n_workers=args.workers,
        )

    if "esm2" in args.models:
        esm2_chains, _ = collect_target_sequences(
            root,
            include_cross_modal=True,
            include_all_transfer=False,
        )
        extract_hf_plm_cache(
            esm2_chains,
            emb_dir / "esm2_150m.pt",
            model_name="facebook/esm2_t30_150M_UR50D",
            trust_remote_code=False,
            n_workers=args.workers,
        )


if __name__ == "__main__":
    main()
