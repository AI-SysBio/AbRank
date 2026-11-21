#!/usr/bin/env python3
"""Step C1: Generate Standardized Multi-Method, Multi-Pose Co-Folding Inputs for All 851
Wild-Type (`WT`) Parental Scaffolds in AbRank (Including Both Experimentally Solved
Co-Crystals and Unresolved Complexes).

Exports:
  1. `metadata/cofolding_manifest.tsv` (committed to Git so inputs can be generated on any
     remote GPU cluster without needing local `.pt` caches).
  2. Method-specific input directories under `data/structures/cofolding_inputs/`:
       - `af3/`      : AlphaFold 3 JSON (`dialect="alphafold3", version=2`, `templates=[]` by default)
       - `boltz2/`   : Boltz-1 / Boltz-2 YAML (`version=1`, protein chains `H`, `[L]`, `A`)
       - `chai1/`    : Chai-1 multi-chain FASTA (`>protein|name=H`, `[>protein|name=L]`, `>protein|name=A`)
       - `protenix/` : Protenix (open-source AF3) JSON (`sequences` + seed list)

All 851 scaffolds—both the 340 experimentally resolved WT co-crystals and the 511 unresolved
WT scaffolds—are generated identically (with `templates=[]` by default) so predicted poses can
be benchmarked head-to-head against native co-crystals (`DockQ` + downstream `LOACO` AUCs) and
deployed uniformly across unresolved targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from abrank.config import get_repo_root
from abrank.schema import trim_fab_to_fv


def build_or_load_manifest(root: Path, force_rebuild: bool = False) -> pd.DataFrame:
    manifest_path = root / "metadata" / "cofolding_manifest.tsv"
    cc_path = root / "data" / "structures" / "complex_cache.pt"

    if manifest_path.exists() and not (force_rebuild and cc_path.exists()):
        df = pd.read_csv(manifest_path, sep="\t", keep_default_na=False)
        df["is_experimental_cocrystal"] = df["is_experimental_cocrystal"].astype(str).str.lower() == "true"
        return df

    if not cc_path.exists():
        raise FileNotFoundError(
            f"Neither {manifest_path} nor {cc_path} found. Run from a clone containing metadata/cofolding_manifest.tsv."
        )

    cc = torch.load(cc_path, map_location="cpu", weights_only=False)
    scafs = cc["scaffolds"]
    abag2sc = cc["abag_to_scaffold"]

    bdir = root / "data" / "benchmark"
    loc_ids, cand_ids, vhh_ids, cm_ids = set(), set(), set(), set()
    for s in ("train", "val", "test"):
        d = pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}.tsv", sep="\t")
        loc_sub = d[d["Category"].isin(["L1", "L2", "L3"])]
        loc_ids.update(loc_sub["AbAgID_1"]); loc_ids.update(loc_sub["AbAgID_2"])
    dc = pd.read_csv(bdir / "splits" / "unseen_ag" / "test_candidates.tsv", sep="\t")
    cand_ids.update(dc["AbAgID_1"]); cand_ids.update(dc["AbAgID_2"])
    dv = pd.read_csv(bdir / "splits" / "vhh" / "vhh_ranking_test.tsv", sep="\t")
    vhh_ids.update(dv["AbAgID_1"]); vhh_ids.update(dv["AbAgID_2"])
    dcm = pd.read_csv(bdir / "splits" / "cross_modal" / "modality_test.tsv", sep="\t")
    cm_ids.update(dcm["AbAgID_1"]); cm_ids.update(dcm["AbAgID_2"])

    loc_sc = {abag2sc[x] for x in loc_ids if x in abag2sc}
    cand_sc = {abag2sc[x] for x in cand_ids if x in abag2sc}
    vhh_sc = {abag2sc[x] for x in vhh_ids if x in abag2sc}
    cm_sc = {abag2sc[x] for x in cm_ids if x in abag2sc}

    # Map each scaffold to exact Fv lengths from standardized benchmark tables (handles both murine & human Fabs)
    scaf_fv_lens: dict[str, tuple[int, int]] = {}
    for fn in ("assays_kd.tsv", "assays_vhh.tsv", "assays_cross_modal.tsv"):
        adf = pd.read_csv(bdir / fn, sep="\t", usecols=["AbAgID", "Ab_heavy_chain_seq", "Ab_light_chain_seq"], low_memory=False)
        adf["Ab_light_chain_seq"] = adf["Ab_light_chain_seq"].fillna("")
        for aid, sh, sl in zip(adf["AbAgID"], adf["Ab_heavy_chain_seq"], adf["Ab_light_chain_seq"]):
            s_id = abag2sc.get(aid)
            if s_id and s_id not in scaf_fv_lens:
                scaf_fv_lens[s_id] = (len(str(sh)), len(str(sl)))

    rows = []
    for sid in sorted(scafs.keys()):
        v = scafs[sid]
        raw_h = str(v["seq_H"]).strip()
        raw_l = str(v["seq_L"]).strip()
        # Enforce uniform Fv/VHH trimming (stripping trailing Fab CH1/CL constant domains
        # from X-ray Fab scaffolds so all 851 solved + unresolved complexes fold pure Fv/VHH + Ag
        # without CH1-CL dimerization inflating multi-chain ipTM).
        if raw_l:
            seq_h, seq_l = trim_fab_to_fv(raw_h, raw_l)
            if sid in scaf_fv_lens:
                lh_ref, ll_ref = scaf_fv_lens[sid]
                if 90 <= lh_ref < len(seq_h):
                    seq_h = seq_h[:lh_ref]
                if 80 <= ll_ref < len(seq_l):
                    seq_l = seq_l[:ll_ref]
        else:
            seq_h, seq_l = raw_h, ""
            if sid in scaf_fv_lens and 90 <= scaf_fv_lens[sid][0] < len(seq_h):
                seq_h = seq_h[: scaf_fv_lens[sid][0]]
        seq_a = str(v["seq_A"]).strip()
        is_cc = bool(v["is_cocrystal"])

        wt_pdb = "NONE"
        if sid.startswith("PDB_") and len(sid.split("_")[1]) == 4:
            wt_pdb = sid.split("_")[1]
        elif sid.startswith("ABAGYM_"):
            wt_pdb = sid.split("_")[-1]

        tr_tags = []
        if sid in loc_sc:
            tr_tags.append("loaco_local")
        if sid in cand_sc:
            tr_tags.append("candidate")
        if sid in vhh_sc:
            tr_tags.append("vhh")
        if sid in cm_sc:
            tr_tags.append("cross_modal")
        if not tr_tags:
            tr_tags.append("global_or_panel")

        if sid in loc_sc:
            tier = "1_loaco_local"
        elif sid in cand_sc or sid in vhh_sc:
            tier = "2_candidate_and_vhh"
        else:
            tier = "3_cross_modal_and_global"

        rows.append(
            {
                "scaffold_id": sid,
                "priority_tier": tier,
                "tracks": ",".join(tr_tags),
                "is_experimental_cocrystal": is_cc,
                "wt_pdb_id": wt_pdb,
                "ref_scaffold_pdb": str(v.get("pdb_path", "")),
                "n_chains": 3 if seq_l else 2,
                "total_aa_len": len(seq_h) + len(seq_l) + len(seq_a),
                "len_H": len(seq_h),
                "len_L": len(seq_l),
                "len_A": len(seq_a),
                "seq_H": seq_h,
                "seq_L": seq_l,
                "seq_A": seq_a,
            }
        )

    df = pd.DataFrame(rows).sort_values(["priority_tier", "scaffold_id"]).reset_index(drop=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, sep="\t", index=False)
    print(f"Wrote {len(df)} WT scaffolds to {manifest_path}")
    return df


def export_method_inputs(
    df: pd.DataFrame,
    out_root: Path,
    methods: list[str],
    num_poses: int = 10,
    allow_templates: bool = False,
) -> None:
    # AF3 & Protenix generate 5 diffusion samples per seed by default: e.g. 2 seeds x 5 = 10 poses
    n_seeds_5x = max(1, (num_poses + 4) // 5)
    seeds_list = [101 + k for k in range(n_seeds_5x)]

    for m in methods:
        (out_root / m).mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        sid = str(row["scaffold_id"])
        chains = [("H", str(row["seq_H"]))]
        if str(row["seq_L"]):
            chains.append(("L", str(row["seq_L"])))
        chains.append(("A", str(row["seq_A"])))

        if "af3" in methods:
            af3_seqs = []
            for cid, seq in chains:
                p_obj: dict = {"id": cid, "sequence": seq}
                if not allow_templates:
                    # Setting templates=[] instructs AlphaFold 3 data pipeline NOT to inject PDB templates,
                    # preventing experimental WT co-crystals from leaking into co-folding benchmark runs.
                    p_obj["templates"] = []
                af3_seqs.append({"protein": p_obj})
            af3_payload = {
                "name": sid,
                "modelSeeds": seeds_list,
                "sequences": af3_seqs,
                "dialect": "alphafold3",
                "version": 2,
            }
            (out_root / "af3" / f"{sid}.json").write_text(json.dumps(af3_payload, indent=2) + "\n", encoding="utf-8")

        if "boltz2" in methods:
            lines = ["version: 1", "sequences:"]
            for cid, seq in chains:
                lines.extend([
                    "  - protein:",
                    f"      id: {cid}",
                    f"      sequence: {seq}",
                ])
            (out_root / "boltz2" / f"{sid}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

        if "chai1" in methods:
            fasta_lines = []
            for cid, seq in chains:
                fasta_lines.append(f">protein|name={cid}")
                fasta_lines.append(seq)
            (out_root / "chai1" / f"{sid}.fasta").write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")

        if "protenix" in methods:
            ptx_seqs = [{"proteinChain": {"sequence": seq, "count": 1}} for _, seq in chains]
            ptx_payload = [{"name": sid, "sequences": ptx_seqs}]
            (out_root / "protenix" / f"{sid}.json").write_text(json.dumps(ptx_payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate multi-method WT complex co-folding inputs for AbRank.")
    ap.add_argument("--methods", nargs="+", default=["af3", "boltz2", "chai1", "protenix"], choices=["af3", "boltz2", "chai1", "protenix"])
    ap.add_argument("--num-poses", type=int, default=10, help="Target poses per complex per method (default: 10).")
    ap.add_argument("--tier", choices=["all", "1", "2", "3"], default="all", help="Priority tier subset (default: all 851 scaffolds).")
    ap.add_argument("--allow-templates", action="store_true", help="Allow PDB template search in AF3 (default: disabled for leak-free evaluation).")
    ap.add_argument("--rebuild-manifest", action="store_true", help="Rebuild metadata/cofolding_manifest.tsv from complex_cache.pt.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: data/structures/cofolding_inputs).")
    args = ap.parse_args()

    root = get_repo_root()
    df = build_or_load_manifest(root, force_rebuild=args.rebuild_manifest)
    if args.tier != "all":
        df = df[df["priority_tier"].str.startswith(f"{args.tier}_")].reset_index(drop=True)

    out_root = args.out_dir or (root / "data" / "structures" / "cofolding_inputs")
    export_method_inputs(df, out_root, args.methods, num_poses=args.num_poses, allow_templates=args.allow_templates)
    n_cc = int(df["is_experimental_cocrystal"].sum())
    print(
        f"Exported {len(df)} WT scaffolds ({n_cc} solved co-crystals + {len(df) - n_cc} unresolved) "
        f"for {args.methods} (target={args.num_poses} poses/method, allow_templates={args.allow_templates}) -> {out_root}"
    )


if __name__ == "__main__":
    main()
