#!/usr/bin/env python3
"""Step 10: Benchmark 3D Structure-Conditioned Zero-Shot & Transfer Models
(`PRODIGY+SASA`, `ProteinMPNN`, `AntiFold`, `ESM-IF1`, `Direct 3D WALLE-Affinity`, and
`3D L2-SP Fine-Tuned`) across the five structure-resolved cross-modality & VHH tracks:
  1. `ELISA (AbDesign, n=2,517)`
  2. `ΔΔG (SKEMPI 2.0, n=2,419, 100% co-crystals)`
  3. `ΔΔG (AbDesign, n=938)`
  4. `Interface DMS (AbAgym, n=16,754, 100% co-crystals, 85.6% Ag-Mut)`
  5. `VHH Nanobody KD (n=520 across 18 unseen Ag clusters)`

Uses the exact same 5-fold zero-leakage `GroupKFold` protocol and antigen/lineage exclusions
as `07_run_modality_transfer_finetuning.py`.

Outputs:
  - `results/structure_dms_ddg_benchmarks.json`
  - `results/STRUCTURE_DMS_DDG_BENCHMARKS.md`
"""

from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

import json
import time
from pathlib import Path
import Levenshtein
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
torch.set_num_threads(8)

from abrank.baselines.models import compute_stratum_weights
from abrank.baselines.structure_models import assemble_structure_pair_features
from abrank.baselines.transfer_finetune import (
    L2SPFineTunedRanker,
    TransferPairTensors,
    assemble_transfer_tensors,
    build_transfer_odd_matrix,
    fit_kd_source_ranker,
)
from abrank.config import get_repo_root
from abrank.evaluate import compute_3state_metrics, compute_per_antigen_macro_metrics
from abrank.pairs import build_within_antigen_pairs


def bootstrap_3state(
    labels: np.ndarray,
    bdiff: np.ndarray,
    pred_diff: np.ndarray,
    pred_impact: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    base = compute_3state_metrics(labels, bdiff, pred_diff, pred_impact)
    rng = np.random.default_rng(seed)
    n = len(labels)
    b_dir, b_imp, b_ord = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_3state_metrics(labels[idx], bdiff[idx], pred_diff[idx], pred_impact[idx])
        if m["auc_dir"] is not None:
            b_dir.append(m["auc_dir"])
        if m["auc_impact"] is not None:
            b_imp.append(m["auc_impact"])
        if m["auc_ordinal"] is not None:
            b_ord.append(m["auc_ordinal"])
    out = dict(base)
    out["dir_ci95"] = [round(float(np.percentile(b_dir, 2.5)), 4), round(float(np.percentile(b_dir, 97.5)), 4)] if b_dir else [0.5, 0.5]
    out["imp_ci95"] = [round(float(np.percentile(b_imp, 2.5)), 4), round(float(np.percentile(b_imp, 97.5)), 4)] if b_imp else [0.5, 0.5]
    out["ord_ci95"] = [round(float(np.percentile(b_ord, 2.5)), 4), round(float(np.percentile(b_ord, 97.5)), 4)] if b_ord else [0.5, 0.5]
    return out


def augment_tp_with_3d(tp: TransferPairTensors, sf) -> TransferPairTensors:
    """Append 3D structural odd (`struct_odd`) and even (`struct_even`) features to `TransferPairTensors`."""
    return TransferPairTensors(
        y=tp.y,
        labels=tp.labels,
        is_impactful=tp.is_impactful,
        bdiff=tp.bdiff,
        is_ab_mut=tp.is_ab_mut,
        is_ag_mut=tp.is_ag_mut,
        is_same_ag=tp.is_same_ag,
        esmc_ab_1=np.concatenate([tp.esmc_ab_1, sf.mpnn_emb_1], axis=1),
        esmc_ab_2=np.concatenate([tp.esmc_ab_2, sf.mpnn_emb_2], axis=1),
        esmc_ag_1=tp.esmc_ag_1,
        esmc_ag_2=tp.esmc_ag_2,
        ablang_1=tp.ablang_1,
        ablang_2=tp.ablang_2,
        zs_features=np.concatenate([tp.zs_features, sf.struct_odd], axis=1),
        biophys_odd=tp.biophys_odd,
        impact_even=np.concatenate([tp.impact_even, sf.struct_even], axis=1),
        zs_dict=tp.zs_dict,
        zs_impact_dict=tp.zs_impact_dict,
    )


def main() -> None:
    t0 = time.time()
    root = get_repo_root()
    bdir = root / "data" / "benchmark"
    edir = root / "data" / "embeddings"
    sdir = root / "data" / "structures"
    res_dir = root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    kd_raw = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False)
    kd_lookup = kd_raw.drop_duplicates("AbAgID").set_index("AbAgID")
    cma_lookup = pd.read_csv(bdir / "assays_cross_modal.tsv", sep="\t", low_memory=False).drop_duplicates("AbAgID").set_index("AbAgID")
    cma_lookup["Ab_light_chain_seq"] = cma_lookup["Ab_light_chain_seq"].fillna("")
    vhh_raw = pd.read_csv(bdir / "assays_vhh.tsv", sep="\t", low_memory=False)
    vhh_raw["Ab_light_chain_seq"] = ""
    vhh_lookup = vhh_raw.drop_duplicates("AbAgID").set_index("AbAgID")

    clust_to_groups: dict[str, set[str]] = {}
    for grp, clust in kd_raw[["Ag_group", "Ag_cluster"]].drop_duplicates().itertuples(index=False):
        clust_to_groups.setdefault(str(clust), set()).add(str(grp))
    ag_to_homolog_groups = {
        str(grp): set(clust_to_groups[str(clust)])
        for grp, clust in kd_raw[["Ag_group", "Ag_cluster"]].drop_duplicates().itertuples(index=False)
    }
    kd_ag_seqs = kd_raw.groupby("Ag_group")["Ag_seq"].first().to_dict()
    for v_grp, v_seq in vhh_raw.groupby("Ag_group")["Ag_seq"].first().items():
        hset = set(ag_to_homolog_groups.get(str(v_grp), {str(v_grp)}))
        for k_grp, k_seq in kd_ag_seqs.items():
            s_sh, s_lg = (str(v_seq), str(k_seq)) if len(str(v_seq)) <= len(str(k_seq)) else (str(k_seq), str(v_seq))
            max_l = max(len(s_sh), len(s_lg))
            max_d = int(0.50 * max_l)
            if (len(s_sh) >= 25 and s_sh in s_lg) or (
                abs(len(s_sh) - len(s_lg)) <= max_d and Levenshtein.distance(s_sh, s_lg, score_cutoff=max_d) <= max_d
            ):
                hset.update(ag_to_homolog_groups.get(str(k_grp), {str(k_grp)}))
        ag_to_homolog_groups[str(v_grp)] = hset

    print("Loading PLM, 3D scaffold, and inverse-folding caches...")
    esmc_cache = torch.load(edir / "esmc_300m.pt", map_location="cpu", weights_only=False)
    ablang_cache = torch.load(edir / "ablang2_paired.pt", map_location="cpu", weights_only=False)
    complex_cache = torch.load(sdir / "complex_cache.pt", map_location="cpu", weights_only=False)
    inv_cache = torch.load(edir / "inverse_folding_scaffolds.pt", map_location="cpu", weights_only=False)

    kd_pool_parts = [pd.read_csv(bdir / "splits" / "unseen_ag" / f"{s}.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test", "test_candidates")]
    kd_pool_df = pd.concat(kd_pool_parts, ignore_index=True).drop_duplicates(["AbAgID_1", "AbAgID_2"]).reset_index(drop=True)
    kd_pool_tp_1d = assemble_transfer_tensors(kd_pool_df, kd_lookup, esmc_cache, ablang_cache)
    kd_pool_sf = assemble_structure_pair_features(kd_pool_df, kd_lookup, complex_cache, inv_cache, None, None, build_patches_and_dccm=False)
    kd_pool_tp_3d = augment_tp_with_3d(kd_pool_tp_1d, kd_pool_sf)
    X_kd_3d = build_transfer_odd_matrix(kd_pool_tp_3d, include_biophys=True)

    cm_all = pd.read_csv(bdir / "splits" / "cross_modal" / "modality_test.tsv", sep="\t", low_memory=False)
    elisa_df = cm_all[cm_all["Assay_modality"] == "ELISA"].reset_index(drop=True)
    abddg_df = cm_all[cm_all["Assay_modality"] == "DDG"].reset_index(drop=True)
    dms_df = cm_all[cm_all["Assay_modality"] == "DMS"].reset_index(drop=True)

    vhh_pairs_all = pd.read_csv(bdir / "splits" / "vhh" / "vhh_ranking_test.tsv", sep="\t", low_memory=False)
    vhh_df = vhh_pairs_all[vhh_pairs_all["Category"] != "Global"].reset_index(drop=True)

    skempi_assays = kd_raw[kd_raw["Dataset"] == "SKEMPIv2"].copy()
    skempi_df, _ = build_within_antigen_pairs(
        skempi_assays, m=1.0, m_neutral=0.30, include_neutral=True, neutral_ratio=1.0,
        modality="DDG_SKEMPIv2", ab_cap=3, per_ag_cap=500, seed=42, prefix="SK10",
    )
    skempi_df = skempi_df.reset_index(drop=True)

    tracks = [
        ("ELISA", "ELISA (AbDesign)", elisa_df, cma_lookup, elisa_df["Ag_group_1"].astype(str).to_numpy(), False),
        ("DDG_SKEMPIv2", "ΔΔG (SKEMPI 2.0)", skempi_df, kd_lookup, skempi_df["Ag_group_1"].astype(str).to_numpy(), True),
        ("DDG_AbDesign", "ΔΔG (AbDesign)", abddg_df, cma_lookup, abddg_df["Ag_group_1"].astype(str).to_numpy(), False),
        ("DMS_AbAgym", "Interface DMS (AbAgym)", dms_df, cma_lookup, dms_df["Ag_group_1"].astype(str).to_numpy(), False),
        ("VHH_KD", "VHH Nanobody KD", vhh_df, vhh_lookup, np.array([vhh_lookup.loc[a, "Ag_cluster"] for a in vhh_df["AbAgID_1"]], dtype=object), False),
    ]

    summary_out: dict[str, dict] = {}
    md_lines = [
        "# AbRank v2 — 3D Structure-Conditioned Benchmarks across ΔΔG, Interface DMS (`AbAgym`), ELISA, & VHH",
        "",
        "Each cell reports **Per-Antigen Macro-Averaged `Sign AUC` / `Impact AUC` / `3-Class Ordinal AUC`** across antigens with `≥5` pairs under 5-fold zero-leakage `GroupKFold` (exact Pooled Micro-AUCs are also preserved in `results/structure_dms_ddg_benchmarks.json` and Section 2 below):",
        "",
        "| Track | Eval Ags (`≥5`) / Pairs (`±1` / `0`) | 1D PLM (ZS) | PRODIGY+SASA (3D) | ProteinMPNN (3D) | AntiFold (3D) | ESM-IF1 (3D) | Direct 3D WALLE (0%) | 3D L2-SP Adapted |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for t_key, t_title, df_t, lkp, groups, excl_skempi in tracks:
        print(f"\n=== Evaluating 3D Models on {t_title} (n={len(df_t)}) ===")
        tp_1d = assemble_transfer_tensors(df_t, lkp, esmc_cache, ablang_cache)
        sf_t = assemble_structure_pair_features(df_t, lkp, complex_cache, inv_cache, None, None, build_patches_and_dccm=False)
        tp_3d = augment_tp_with_3d(tp_1d, sf_t)
        X_t_3d = build_transfer_odd_matrix(tp_3d, include_biophys=True)

        n = len(df_t)
        oof_walle_d = np.zeros(n, dtype=np.float32)
        oof_walle_i = np.zeros(n, dtype=np.float32)
        oof_l2sp_d = np.zeros(n, dtype=np.float32)
        oof_l2sp_i = np.zeros(n, dtype=np.float32)

        gkf = GroupKFold(n_splits=5)
        for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(df_t, tp_3d.y, groups)):
            te_ags = set(df_t.iloc[te_idx]["Ag_group_1"].astype(str)) | set(df_t.iloc[te_idx]["Ag_group_2"].astype(str))
            excl_ags: set[str] = set()
            for g in te_ags:
                excl_ags.update(ag_to_homolog_groups.get(g, {g}))

            m_kd = (~kd_pool_df["Ag_group_1"].isin(excl_ags)) & (~kd_pool_df["Ag_group_2"].isin(excl_ags))
            if excl_skempi:
                m_kd &= (kd_pool_df["Dataset_1"] != "SKEMPIv2") & (kd_pool_df["Dataset_2"] != "SKEMPIv2")
            keep = np.flatnonzero(m_kd.to_numpy(dtype=bool))
            sub_kd_df = kd_pool_df.iloc[keep].reset_index(drop=True)
            sub_kd_tp = kd_pool_tp_3d.subset(keep)
            sub_X_kd = X_kd_3d[keep]
            w_all = compute_stratum_weights(sub_kd_df, sub_kd_tp.is_ag_mut)

            loc_ags = sorted(sub_kd_df.loc[sub_kd_df["Category"] != "Global", "Ag_group_1"].unique())
            val_ags = set(loc_ags[1::5]) if len(loc_ags) >= 5 else set(loc_ags[:1])
            is_val = (sub_kd_df["Ag_group_1"].isin(val_ags) | sub_kd_df["Ag_group_2"].isin(val_ags)).to_numpy()

            kd_3d_src = fit_kd_source_ranker(
                sub_X_kd[~is_val], sub_kd_tp.y[~is_val], w_all[~is_val],
                sub_X_kd[is_val], sub_kd_tp.y[is_val],
                tp_tr=sub_kd_tp.subset(np.flatnonzero(~is_val)),
                tp_val=sub_kd_tp.subset(np.flatnonzero(is_val)),
            )

            te_tp = tp_3d.subset(te_idx)
            d0, i0 = kd_3d_src.predict_3state(te_tp, X_t_3d[te_idx])
            oof_walle_d[te_idx] = d0
            oof_walle_i[te_idx] = i0

            ft_3d = L2SPFineTunedRanker(kd_source=kd_3d_src).fit(
                X_t_3d[tr_idx], tp_3d.y[tr_idx], groups[tr_idx], tp_tr=tp_3d.subset(tr_idx)
            )
            d1, i1 = ft_3d.predict_3state(te_tp, X_t_3d[te_idx])
            oof_l2sp_d[te_idx] = d1
            oof_l2sp_i[te_idx] = i1

        lbl = tp_3d.labels
        zs1d_d = tp_1d.zs_dict["esmc_ablang2_zeroshot"]
        zs1d_i = tp_1d.zs_impact_dict["esmc_ablang2_zeroshot"]
        methods_res = {
            "plm_1d_zs": compute_per_antigen_macro_metrics(df_t, zs1d_d, zs1d_i, min_pairs=5, n_bootstrap=500),
            "prodigy_sasa": compute_per_antigen_macro_metrics(df_t, sf_t.zs_dir["prodigy_sasa"], sf_t.zs_imp["prodigy_sasa"], min_pairs=5, n_bootstrap=500),
            "protein_mpnn": compute_per_antigen_macro_metrics(df_t, sf_t.zs_dir["protein_mpnn"], sf_t.zs_imp["protein_mpnn"], min_pairs=5, n_bootstrap=500),
            "antifold": compute_per_antigen_macro_metrics(df_t, sf_t.zs_dir["antifold"], sf_t.zs_imp["antifold"], min_pairs=5, n_bootstrap=500),
            "esm_if1": compute_per_antigen_macro_metrics(df_t, sf_t.zs_dir["esm_if1"], sf_t.zs_imp["esm_if1"], min_pairs=5, n_bootstrap=500),
            "walle_3d_direct": compute_per_antigen_macro_metrics(df_t, oof_walle_d, oof_walle_i, min_pairs=5, n_bootstrap=500),
            "walle_3d_l2sp": compute_per_antigen_macro_metrics(df_t, oof_l2sp_d, oof_l2sp_i, min_pairs=5, n_bootstrap=500),
        }
        # Also disaggregate Ab-Mut vs Ag-Mut for DMS_AbAgym & SKEMPIv2!
        slices = {}
        for sname, smask in (("Ab-Mut", ~tp_3d.is_ag_mut), ("Ag-Mut", tp_3d.is_ag_mut)):
            if smask.sum() >= 15:
                sub_s_df = df_t.iloc[np.flatnonzero(smask)]
                slices[sname] = {
                    mk: compute_per_antigen_macro_metrics(sub_s_df, md_arr[smask], mi_arr[smask], min_pairs=5)
                    for mk, (md_arr, mi_arr) in (
                        ("plm_1d_zs", (zs1d_d, zs1d_i)),
                        ("protein_mpnn", (sf_t.zs_dir["protein_mpnn"], sf_t.zs_imp["protein_mpnn"])),
                        ("esm_if1", (sf_t.zs_dir["esm_if1"], sf_t.zs_imp["esm_if1"])),
                        ("walle_3d_direct", (oof_walle_d, oof_walle_i)),
                        ("walle_3d_l2sp", (oof_l2sp_d, oof_l2sp_i)),
                    )
                }

        n_ags_eval = methods_res["plm_1d_zs"]["n_antigens_evaluated"]
        summary_out[t_key] = {
            "title": t_title,
            "n_pairs": int(n),
            "n_impactful": int((lbl != 0).sum()),
            "n_neutral": int((lbl == 0).sum()),
            "n_antigens_evaluated": int(n_ags_eval),
            "cocrystal_pct": round(100.0 * float(sf_t.is_cocrystal.mean()), 2),
            "methods": methods_res,
            "slices": slices,
        }

        def _fmt(m: dict) -> str:
            return f"`{m['auc_dir']:.3f}` / `{m['auc_impact']:.3f}` / **`{m['auc_ordinal']:.3f}`**"

        md_lines.append(
            f"| **{t_title}** | `{n_ags_eval}` Ags / `{n:,}` (`{(lbl!=0).sum()}`/`{(lbl==0).sum()}`) | "
            f"{_fmt(methods_res['plm_1d_zs'])} | {_fmt(methods_res['prodigy_sasa'])} | "
            f"{_fmt(methods_res['protein_mpnn'])} | {_fmt(methods_res['antifold'])} | "
            f"{_fmt(methods_res['esm_if1'])} | {_fmt(methods_res['walle_3d_direct'])} | "
            f"{_fmt(methods_res['walle_3d_l2sp'])} |"
        )
        for mk, mv in methods_res.items():
            print(f"  {mk:20s}: Sign={mv['auc_dir']:.3f} | Imp={mv['auc_impact']:.3f} | Ord={mv['auc_ordinal']:.3f} (Pooled Ord={mv['pooled_auc_ordinal']:.3f})")

    md_lines.extend([
        "",
        "## 2. Per-Antigen Macro-AUC (`≥5` Pairs) vs. Pooled Micro-AUC Comparison",
        "",
        "| Track | ESM-IF1 Macro / Pooled Ord | Direct 3D WALLE Macro / Pooled Ord | 3D L2-SP Macro / Pooled Ord |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for t_key, sv in summary_out.items():
        mr = sv["methods"]
        md_lines.append(
            f"| **{sv['title']}** | "
            f"**`{mr['esm_if1']['auc_ordinal']:.3f}`** / `{mr['esm_if1']['pooled_auc_ordinal']:.3f}` | "
            f"**`{mr['walle_3d_direct']['auc_ordinal']:.3f}`** / `{mr['walle_3d_direct']['pooled_auc_ordinal']:.3f}` | "
            f"**`{mr['walle_3d_l2sp']['auc_ordinal']:.3f}`** / `{mr['walle_3d_l2sp']['pooled_auc_ordinal']:.3f}` |"
        )

    out_json = res_dir / "structure_dms_ddg_benchmarks.json"
    out_md = res_dir / "STRUCTURE_DMS_DDG_BENCHMARKS.md"
    out_json.write_text(json.dumps({"benchmark": "AbRank v2.0.0", "tracks": summary_out}, indent=2) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nSaved {out_json} and {out_md} in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
