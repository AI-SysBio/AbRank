"""Automated integrity, schema, 3-state noise-buffer, and zero-leakage tests for AbRank v2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abrank.config import BenchmarkConfig, RepoPaths
from abrank.evaluate import (
    compute_cluster_bootstrap_summary,
    compute_per_antigen_macro_metrics,
    evaluate_pairwise_predictions,
    evaluate_pointwise_spearman,
)
from abrank.schema import (
    STANDARD_AA,
    kd_nm_to_pkd,
    parse_numeric_measurement,
    trim_antigen_tags,
    trim_fab_to_fv,
)


def test_unit_helpers() -> None:
    assert parse_numeric_measurement("12.5 nM") == (12.5, "=")
    assert parse_numeric_measurement(">1000") == (1000.0, ">")
    assert parse_numeric_measurement("10-200")[1] == "range"
    assert abs(float(kd_nm_to_pkd(1.0)) - 9.0) < 1e-9
    assert abs(float(kd_nm_to_pkd(100.0)) - 7.0) < 1e-9

    ag = "MGSSHHHHHHSSGLVPRGSHMYDFTNCDFEKIKAAYLSTISKD"
    assert trim_antigen_tags(ag) == "YDFTNCDFEKIKAAYLSTISKD"

    vh_ch1 = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSSASTKGPSVFPLAPXXXXXXXX"
    vl_cl = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTAS"
    vh, vl = trim_fab_to_fv(vh_ch1, vl_cl)
    assert "X" not in vh
    assert vh.endswith("WGQGTLVTVSS")
    assert vl.endswith("FGQGTKVEIK")


def test_master_tables_hygiene() -> None:
    from abrank.kaggle_io import resolve_benchmark_dir

    paths = RepoPaths.from_root()
    try:
        bdir = resolve_benchmark_dir(repo_root=paths.root, auto_download=False)
    except FileNotFoundError:
        pytest.skip("Benchmark tables not yet built.")

    kd = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False)
    assert len(kd) > 100_000
    assert kd["binding_score"].isna().sum() == 0
    assert set(kd["Ag_type"].unique()) == {"Viral", "Non-Viral"}

    for col in ("Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"):
        bad = kd[col].astype(str).map(lambda s: not (set(s) <= STANDARD_AA and len(s) >= 10))
        assert int(bad.sum()) == 0, f"Found {int(bad.sum())} non-standard sequences in {col}"

    hr2 = kd[kd["Ag_group"] == "SARSCoV2_HR2"]
    assert len(hr2) > 30_000
    assert float(hr2["binding_score"].min()) < 6.0
    assert float(hr2["binding_score"].max()) < 10.5

    cm = pd.read_csv(bdir / "assays_cross_modal.tsv", sep="\t", low_memory=False)
    assert {"Enrichment", "ELISA", "DDG", "EC50", "DMS"} <= set(cm["Assay_modality"].unique())


def test_zero_leakage_and_3state_invariants() -> None:
    from abrank.kaggle_io import resolve_benchmark_dir

    paths = RepoPaths.from_root()
    cfg = BenchmarkConfig.load(paths.configs / "benchmark.yaml")
    try:
        bdir = resolve_benchmark_dir(repo_root=paths.root, auto_download=False)
    except FileNotFoundError:
        pytest.skip("Splits not yet built.")
    splits_dir = bdir / "splits"

    # Legacy multi-tier directories must not exist (single canonical unseen_ag protocol)
    assert not (splits_dir / "seen_ag").exists()
    assert not (splits_dir / "unseen_ab").exists()

    kd = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", low_memory=False)
    id_to_lin = kd.set_index("AbAgID")["Ab_lineage"].to_dict()
    id_to_agclust = kd.set_index("AbAgID")["Ag_cluster"].to_dict()

    for tier in ("unseen_ag",):
        part_ids: dict[str, set[str]] = {}
        for sp in ("train", "val", "test"):
            df = pd.read_csv(splits_dir / tier / f"{sp}.tsv", sep="\t")
            assert len(df) > 0, f"{tier}/{sp}.tsv is empty!"
            assert set(df["label"].unique()) == {-1, 0, 1}, f"{tier}/{sp} missing 3-state labels!"

            imp = df[df["label"] != 0]
            neu = df[df["label"] == 0]
            assert len(imp) > 0 and len(neu) > 0

            # 3-State margin invariants
            assert (imp["binding_diff"].abs() >= cfg.margin_m - 1e-4).all()
            assert (neu["binding_diff"].abs() <= cfg.neutral_margin + 1e-4).all()

            # Sign consistency on impactful rows
            assert ((imp["binding_diff"] > 0) == (imp["label"] == 1)).all()
            assert ((imp["logKD_diff"] < 0) == (imp["label"] == 1)).all()
            assert np.allclose(df["logKD_diff"], -df["binding_diff"], atol=1e-4)

            # Short linear peptide isolation from cross-antigen Global pairs
            glb = df[df["Category"] == "Global"]
            assert (glb["label"] != 0).all()
            assert "SARSCoV2_HR2" not in set(glb["Ag_group_1"]).union(set(glb["Ag_group_2"]))

            # Balanced orientation among impactful rows (~50/50)
            pos_frac = float((imp["label"] == 1).mean())
            assert 0.45 <= pos_frac <= 0.55, f"{tier}/{sp} impactful label balance skewed: {pos_frac}"

            # Unfiltered pointwise panel exists and is disjoint across splits
            pw_df = pd.read_csv(splits_dir / tier / f"{sp}_pointwise.tsv", sep="\t", low_memory=False)
            assert len(pw_df) > 0
            part_ids[sp] = set(pw_df["AbAgID"]).union(df["AbAgID_1"]).union(df["AbAgID_2"])

        assert len(part_ids["train"] & part_ids["val"]) == 0
        assert len(part_ids["train"] & part_ids["test"]) == 0
        assert len(part_ids["val"] & part_ids["test"]) == 0

        train_lins = {id_to_lin[i] for i in part_ids["train"]}
        val_lins = {id_to_lin[i] for i in part_ids["val"]}
        test_lins = {id_to_lin[i] for i in part_ids["test"]}
        assert len(train_lins & test_lins) == 0, "unseen_ag: antibody lineage leaked between train & test!"
        assert len(train_lins & val_lins) == 0, "unseen_ag: antibody lineage leaked between train & val!"
        assert len(val_lins & test_lins) == 0, "unseen_ag: antibody lineage leaked between val & test!"

        train_ag = {id_to_agclust[i] for i in part_ids["train"]}
        val_ag = {id_to_agclust[i] for i in part_ids["val"]}
        test_ag = {id_to_agclust[i] for i in part_ids["test"]}
        assert len(train_ag & test_ag) == 0, "unseen_ag: homologous antigen cluster leaked!"
        assert len(train_ag & val_ag) == 0, "unseen_ag: homologous antigen cluster leaked!"
        assert len(val_ag & test_ag) == 0, "unseen_ag: homologous antigen cluster leaked!"

    # Verify expanded VHH KD track and NbBench Galectin-3 Cross-Modal Enrichment
    vhh_assays = pd.read_csv(bdir / "assays_vhh.tsv", sep="\t", low_memory=False)
    assert len(vhh_assays) >= 290
    assert vhh_assays["Dataset"].nunique() == 8
    vhh_pairs = pd.read_csv(splits_dir / "vhh" / "vhh_ranking_test.tsv", sep="\t")
    assert len(vhh_pairs) >= 750
    assert set(vhh_pairs["label"].unique()) == {-1, 0, 1}

    cm = pd.read_csv(bdir / "assays_cross_modal.tsv", sep="\t", low_memory=False)
    assert "NbBench" in set(cm["Dataset"].unique())


def test_3state_evaluator_and_panel_spearman() -> None:
    from abrank.kaggle_io import resolve_benchmark_dir

    paths = RepoPaths.from_root()
    try:
        bdir = resolve_benchmark_dir(repo_root=paths.root, auto_download=False)
    except FileNotFoundError:
        pytest.skip("Splits not yet built.")
    test_path = bdir / "splits" / "unseen_ag" / "test.tsv"

    df = pd.read_csv(test_path, sep="\t")
    # Oracle predictions = binding_diff -> all 3-state AUCs == 1.0, random == 0.5
    m_oracle = evaluate_pairwise_predictions(df, df["binding_diff"])
    assert m_oracle["auc_dir"] == 1.0
    assert m_oracle["auc_impact"] == 1.0
    assert m_oracle["auc_ordinal"] == 1.0
    assert m_oracle["auc_selective_hit"] == 1.0

    m_zero = evaluate_pairwise_predictions(df, np.zeros(len(df)))
    assert m_zero["auc_dir"] == 0.5
    assert m_zero["auc_impact"] == 0.5
    assert m_zero["auc_ordinal"] == 0.5
    assert m_zero["auc_selective_hit"] == 0.5

    # Exact orientation invariance under flipping any subset of rows
    flipped = df.copy()
    flipped.iloc[::3, flipped.columns.get_loc("binding_diff")] *= -1
    flipped.iloc[::3, flipped.columns.get_loc("label")] *= -1
    preds = df["binding_diff"].to_numpy().copy() * 0.7 + 0.05
    preds_flip = preds.copy()
    preds_flip[::3] *= -1
    m_a = evaluate_pairwise_predictions(df, preds)
    m_b = evaluate_pairwise_predictions(flipped, preds_flip)
    assert m_a["auc_dir"] == m_b["auc_dir"]
    assert m_a["auc_impact"] == m_b["auc_impact"]
    assert m_a["auc_ordinal"] == m_b["auc_ordinal"]

    pw_test = pd.read_csv(bdir / "splits" / "unseen_ag" / "test_pointwise.tsv", sep="\t", low_memory=False)
    oracle_map = dict(zip(pw_test["AbAgID"], pw_test["binding_score"]))
    sp_res = evaluate_pointwise_spearman(pw_test, oracle_map, min_group_size=6)
    assert sp_res["mean_spearman_rho"] == 1.0


def test_antisymmetric_and_symmetric_feature_invariance() -> None:
    from abrank.baselines.features import (
        chain_biophys_features,
        extract_mutation_biophys_odd,
        extract_mutation_impact_even,
    )
    from abrank.baselines.models import AntisymmetricLinearRanker, build_odd_linear_matrix

    vh1 = ["EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKSYGAFDYWGQGTLVTVSS"] * 30
    vh2 = ["EVQLVESGGGLVQPGGSLRLSCAASGFTFSSWAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKSAGAFDYWGQGTLVTVSS"] * 30
    vl1 = ["DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"] * 30
    vl2 = vl1
    ag1 = ["NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKC"] * 30
    ag2 = ["NITNLCPFGEVFAATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKC"] * 30

    f1 = chain_biophys_features(vh1[0], is_antibody_var=True)
    f2 = chain_biophys_features(vh2[0], is_antibody_var=True)
    assert f1.shape == f2.shape and not np.allclose(f1, f2)

    odd12 = extract_mutation_biophys_odd(vh1, vl1, ag1, vh2, vl2, ag2)
    odd21 = extract_mutation_biophys_odd(vh2, vl2, ag2, vh1, vl1, ag1)
    assert np.allclose(odd12, -odd21, atol=1e-6)

    even12 = extract_mutation_impact_even(vh1, vl1, ag1, vh2, vl2, ag2)
    even21 = extract_mutation_impact_even(vh2, vl2, ag2, vh1, vl1, ag1)
    assert np.allclose(even12, even21, atol=1e-6)

    rng = np.random.default_rng(0)
    x1 = rng.normal(size=(60, 24)).astype(np.float32)
    x2 = rng.normal(size=(60, 24)).astype(np.float32)
    m12 = build_odd_linear_matrix([x1], [x2], include_unit_direction=True)
    m21 = build_odd_linear_matrix([x2], [x1], include_unit_direction=True)
    assert np.allclose(m12, -m21, atol=1e-6)

    labels = np.array([1, -1, 0] * 20, dtype=int)
    y = (labels > 0).astype(int)
    val_df = pd.DataFrame({
        "binding_diff": np.where(labels[40:] > 0, 1.5, np.where(labels[40:] < 0, -1.5, 0.05)),
        "label": labels[40:],
        "Category": ["L1"] * 10 + ["Global"] * 10,
    })
    ranker = AntisymmetricLinearRanker(use_neutral_nullspace=True).fit(
        m12[:40], y[:40], None, m12[40:], val_df, np.zeros(20, dtype=bool), labels_tr=labels[:40]
    )
    assert np.allclose(ranker.predict(m12[40:]), -ranker.predict(m21[40:]), atol=1e-6)


def test_transfer_3state_antisymmetry_and_impact_symmetry() -> None:
    from abrank.baselines.features import extract_mutation_biophys_odd, extract_mutation_impact_even
    from abrank.baselines.transfer_finetune import (
        KDResidualAdapterRanker,
        L2SPFineTunedRanker,
        TransferPairTensors,
        build_transfer_odd_matrix,
        fit_kd_source_ranker,
    )

    n = 36
    vh1 = ["EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS"] * n
    vh2 = ["EVQLVESGGGLVQPGGSLRLSCAASGFTFSSWAMSWVRQAPGKGLEWVS"] * n
    vl1 = ["DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIY"] * n
    vl2 = vl1
    ag1 = ["NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKC"] * n
    ag2 = ["NITNLCPFGEVFAATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKC"] * n

    b12 = extract_mutation_biophys_odd(vh1, vl1, ag1, vh2, vl2, ag2)
    b21 = extract_mutation_biophys_odd(vh2, vl2, ag2, vh1, vl1, ag1)
    ev = np.hstack([extract_mutation_impact_even(vh1, vl1, ag1, vh2, vl2, ag2), np.ones((n, 17), dtype=np.float32)])

    rng = np.random.default_rng(7)
    labels = np.array([1, -1, 0] * (n // 3), dtype=int)
    bdiff = np.where(labels > 0, 1.4, np.where(labels < 0, -1.4, 0.05))
    y = (bdiff > 0).astype(int)
    zs = rng.normal(size=(n, 10)).astype(np.float32)

    tp12 = TransferPairTensors(
        y=y,
        labels=labels,
        is_impactful=labels != 0,
        bdiff=bdiff,
        is_ab_mut=np.ones(n, dtype=bool),
        is_ag_mut=np.ones(n, dtype=bool),
        is_same_ag=np.zeros(n, dtype=bool),
        esmc_ab_1=rng.normal(size=(n, 32)).astype(np.float32),
        esmc_ab_2=rng.normal(size=(n, 32)).astype(np.float32),
        esmc_ag_1=rng.normal(size=(n, 24)).astype(np.float32),
        esmc_ag_2=rng.normal(size=(n, 24)).astype(np.float32),
        ablang_1=rng.normal(size=(n, 16)).astype(np.float32),
        ablang_2=rng.normal(size=(n, 16)).astype(np.float32),
        zs_features=zs,
        biophys_odd=b12,
        impact_even=ev,
        zs_dict={"esmc_ablang2_zeroshot": zs[:, 0]},
        zs_impact_dict={"esmc_ablang2_zeroshot": np.abs(zs[:, 0])},
    )
    tp21 = TransferPairTensors(
        y=1 - y,
        labels=-labels,
        is_impactful=tp12.is_impactful,
        bdiff=-bdiff,
        is_ab_mut=tp12.is_ab_mut,
        is_ag_mut=tp12.is_ag_mut,
        is_same_ag=tp12.is_same_ag,
        esmc_ab_1=tp12.esmc_ab_2,
        esmc_ab_2=tp12.esmc_ab_1,
        esmc_ag_1=tp12.esmc_ag_2,
        esmc_ag_2=tp12.esmc_ag_1,
        ablang_1=tp12.ablang_2,
        ablang_2=tp12.ablang_1,
        zs_features=-tp12.zs_features,
        biophys_odd=b21,
        impact_even=ev,
        zs_dict={"esmc_ablang2_zeroshot": -zs[:, 0]},
        zs_impact_dict={"esmc_ablang2_zeroshot": np.abs(zs[:, 0])},
    )

    X12 = build_transfer_odd_matrix(tp12)
    X21 = build_transfer_odd_matrix(tp21)
    assert np.allclose(X12, -X21, atol=1e-5)

    kd_src = fit_kd_source_ranker(X12[:24], y[:24], None, X12[24:], y[24:], tp_tr=tp12.subset(range(24)), tp_val=tp12.subset(range(24, 36)))
    groups = np.array([f"G{i % 4}" for i in range(24)])
    ft = L2SPFineTunedRanker(kd_source=kd_src).fit(X12[:24], y[:24], groups, tp_tr=tp12.subset(range(24)))
    d12, i12 = ft.predict_3state(tp12.subset(range(24, 36)), X12[24:])
    d21, i21 = ft.predict_3state(tp21.subset(range(24, 36)), X21[24:])
    assert np.allclose(d12, -d21, atol=1e-5)
    assert np.allclose(i12, i21, atol=1e-5)

    adp = KDResidualAdapterRanker(kd_source=kd_src, epochs=4, batch_size=12).fit(tp12.subset(range(24)), X12[:24], groups)
    ad12, ai12 = adp.predict_3state(tp12.subset(range(24, 36)), X12[24:])
    ad21, ai21 = adp.predict_3state(tp21.subset(range(24, 36)), X21[24:])
    assert np.allclose(ad12, -ad21, atol=1e-5)
    assert np.allclose(ai12, ai21, atol=1e-5)


def test_structure_3state_models_permutation_symmetry() -> None:
    """Verify exact antisymmetry `f(1,2) == -f(2,1)` and impact symmetry `q(1,2) == +q(2,1)`
    on `ANTIPASTI` (Normal-Mode 3D CNN), `GearBind` (3D Geometric GNN), and `WALLE-Affinity`.
    """
    from abrank.baselines.models import SplitPairTensors
    from abrank.baselines.structure_models import (
        ANTIPASTI3StateTrainer,
        GearBind3StateTrainer,
        StructureSplitFeatures,
        fit_walle_affinity_3state,
    )

    rng = np.random.default_rng(123)
    n = 54
    labels = np.array([1, -1, 0] * 18, dtype=int)
    bdiff = np.where(labels > 0, 1.5, np.where(labels < 0, -1.5, 0.05))
    y = (bdiff > 0).astype(int)

    def _rand_sym64(cnt: int) -> np.ndarray:
        a = rng.normal(size=(cnt, 2, 64, 64)).astype(np.float32)
        return (0.5 * (a + np.swapaxes(a, 2, 3))).astype(np.float16)

    d1, d2 = _rand_sym64(n), _rand_sym64(n)
    k = 40
    pn1 = rng.normal(size=(n, k, 90)).astype(np.float16)
    pn2 = rng.normal(size=(n, k, 90)).astype(np.float16)
    ca1 = rng.normal(scale=6.0, size=(n, k, 3)).astype(np.float32)
    ca2 = rng.normal(scale=6.0, size=(n, k, 3)).astype(np.float32)
    cb1 = ca1 + 0.8
    cb2 = ca2 + 0.8
    cc1 = np.tile(np.array([0] * 15 + [1] * 10 + [2] * 15, dtype=np.int8), (n, 1))
    sp_idx = np.tile(np.arange(k, dtype=np.int16), (n, 1))

    s_odd = rng.normal(size=(n, 22)).astype(np.float32)
    s_even = np.abs(rng.normal(size=(n, 24))).astype(np.float32)
    if1_1 = rng.normal(size=(n, 64)).astype(np.float32)
    if1_2 = rng.normal(size=(n, 64)).astype(np.float32)
    mp_1 = rng.normal(size=(n, 48)).astype(np.float32)
    mp_2 = rng.normal(size=(n, 48)).astype(np.float32)

    sf12 = StructureSplitFeatures(
        is_cocrystal=np.ones(n, dtype=bool),
        zs_dir={"esm_if1": s_odd[:, 0]},
        zs_imp={"esm_if1": s_even[:, 0]},
        struct_odd=s_odd,
        struct_even=s_even,
        if1_emb_1=if1_1, if1_emb_2=if1_2,
        mpnn_emb_1=mp_1, mpnn_emb_2=mp_2,
        saprot_ab_1=mp_1, saprot_ab_2=mp_2, saprot_ag_1=mp_1, saprot_ag_2=mp_2,
        mint_ab_1=mp_1, mint_ab_2=mp_2, mint_ag_1=mp_1, mint_ag_2=mp_2,
        dccm_1=d1, dccm_2=d2,
        patch_node_1=pn1, patch_node_2=pn2,
        patch_ca_1=ca1, patch_ca_2=ca2,
        patch_cb_1=cb1, patch_cb_2=cb2,
        patch_chain_1=cc1, patch_chain_2=cc1,
        patch_seqpos_1=sp_idx, patch_seqpos_2=sp_idx,
    )
    sf21 = StructureSplitFeatures(
        is_cocrystal=sf12.is_cocrystal,
        zs_dir={"esm_if1": -s_odd[:, 0]},
        zs_imp={"esm_if1": s_even[:, 0]},
        struct_odd=-s_odd,
        struct_even=s_even,
        if1_emb_1=if1_2, if1_emb_2=if1_1,
        mpnn_emb_1=mp_2, mpnn_emb_2=mp_1,
        saprot_ab_1=mp_2, saprot_ab_2=mp_1, saprot_ag_1=mp_2, saprot_ag_2=mp_1,
        mint_ab_1=mp_2, mint_ab_2=mp_1, mint_ag_1=mp_2, mint_ag_2=mp_1,
        dccm_1=d2, dccm_2=d1,
        patch_node_1=pn2, patch_node_2=pn1,
        patch_ca_1=ca2, patch_ca_2=ca1,
        patch_cb_1=cb2, patch_cb_2=cb1,
        patch_chain_1=cc1, patch_chain_2=cc1,
        patch_seqpos_1=sp_idx, patch_seqpos_2=sp_idx,
    )

    b_odd = rng.normal(size=(n, 18)).astype(np.float32)
    i_even = np.abs(rng.normal(size=(n, 38))).astype(np.float32)
    zs10 = rng.normal(size=(n, 10)).astype(np.float32)
    same_ag = np.array([True] * 36 + [False] * 18, dtype=bool)
    sp12 = SplitPairTensors(
        y=y, labels=labels, is_impactful=(labels != 0),
        is_ab_mut=np.ones(n, dtype=bool), is_ag_mut=~same_ag, is_same_ag=same_ag,
        esmc_ab_1=mp_1, esmc_ab_2=mp_2, esmc_ag_1=mp_1, esmc_ag_2=mp_2,
        ablang_1=mp_1, ablang_2=mp_2, esm2_ab_1=mp_1, esm2_ab_2=mp_2, esm2_ag_1=mp_1, esm2_ag_2=mp_2,
        zs_features=zs10, biophys_odd=b_odd, impact_even=i_even, zs_dict={}, zs_impact_dict={},
    )
    sp21 = SplitPairTensors(
        y=1 - y, labels=-labels, is_impactful=(labels != 0),
        is_ab_mut=sp12.is_ab_mut, is_ag_mut=sp12.is_ag_mut, is_same_ag=sp12.is_same_ag,
        esmc_ab_1=mp_2, esmc_ab_2=mp_1, esmc_ag_1=mp_2, esmc_ag_2=mp_1,
        ablang_1=mp_2, ablang_2=mp_1, esm2_ab_1=mp_2, esm2_ab_2=mp_1, esm2_ag_1=mp_2, esm2_ag_2=mp_1,
        zs_features=-zs10, biophys_odd=-b_odd, impact_even=i_even, zs_dict={}, zs_impact_dict={},
    )
    val_df = pd.DataFrame({"binding_diff": bdiff, "label": labels, "Category": ["L1"] * 36 + ["Global"] * 18})
    w_tr = np.ones(n, dtype=np.float32)

    m_anti = ANTIPASTI3StateTrainer(seed=42, epochs=3, batch_size=15).fit(sf12, sp12, sf12, sp12, val_df, w_tr)
    ad12, ai12 = m_anti.predict_3state(sf12, sp12)
    ad21, ai21 = m_anti.predict_3state(sf21, sp21)
    assert np.allclose(ad12, -ad21, atol=1e-5)
    assert np.allclose(ai12, ai21, atol=1e-5)

    m_gb = GearBind3StateTrainer(seed=42, epochs=3, batch_size=15).fit(sf12, sp12, sf12, sp12, val_df, w_tr)
    gd12, gi12 = m_gb.predict_3state(sf12)
    gd21, gi21 = m_gb.predict_3state(sf21)
    assert np.allclose(gd12, -gd21, atol=1e-5)
    assert np.allclose(gi12, gi21, atol=1e-5)

    pred_walle = fit_walle_affinity_3state(sf12, sp12, sf12, sp12, val_df, w_tr, seed=42)
    wd12, wi12 = pred_walle(sf12, sp12)
    wd21, wi21 = pred_walle(sf21, sp21)
    assert np.allclose(wd12, -wd21, atol=1e-5)
    assert np.allclose(wi12, wi21, atol=1e-5)


def test_per_antigen_macro_averaging() -> None:
    """Verify equal-weight per-antigen macro-averaging (`min_pairs=5`), singleton exclusion,
    pooled micro-AUC preservation, and orientation-flip invariance.
    """
    # Antigen A: mega-panel (200 pairs) where predictions are 100% perfect -> AUC_A = 1.0
    lbl_a = np.array([1, -1, 0, 0] * 50, dtype=int)
    bd_a = np.where(lbl_a > 0, 1.5, np.where(lbl_a < 0, -1.5, 0.05))
    pd_a = bd_a.copy()
    pi_a = np.where(lbl_a != 0, 2.0, 0.1)

    # Antigen B: smaller panel (12 pairs >= 5) where predictions are all zero -> AUC_B = 0.5
    lbl_b = np.array([1, -1, 0, 0] * 3, dtype=int)
    bd_b = np.where(lbl_b > 0, 1.5, np.where(lbl_b < 0, -1.5, 0.05))
    pd_b = np.zeros(len(lbl_b), dtype=float)
    pi_b = np.zeros(len(lbl_b), dtype=float)

    # Antigen C: singleton (< 5 pairs: 2 pairs) where predictions are inverted (AUC_C = 0.0)
    # Must be excluded from macro-average (`min_pairs=5`) so singletons cannot swing benchmark AUC!
    lbl_c = np.array([1, 0], dtype=int)
    bd_c = np.array([1.5, 0.05], dtype=float)
    pd_c = np.array([-3.0, 5.0], dtype=float)
    pi_c = np.array([0.0, 5.0], dtype=float)

    df = pd.DataFrame(
        {
            "Ag_group_1": ["AgA"] * 200 + ["AgB"] * 12 + ["AgC"] * 2,
            "Ag_group_2": ["AgA"] * 200 + ["AgB"] * 12 + ["AgC"] * 2,
            "Category": ["L1"] * 214,
            "label": np.concatenate([lbl_a, lbl_b, lbl_c]),
            "binding_diff": np.concatenate([bd_a, bd_b, bd_c]),
        }
    )
    preds = np.concatenate([pd_a, pd_b, pd_c])
    pimps = np.concatenate([pi_a, pi_b, pi_c])

    res = compute_per_antigen_macro_metrics(df, preds, pimps, min_pairs=5, n_bootstrap=100, seed=7)
    assert res["n_antigens_evaluated"] == 2
    assert set(res["per_antigen"].keys()) == {"AgA", "AgB"}
    # Equal 50/50 weight between AgA (1.0) and AgB (0.5) -> 0.7500, NOT ~0.96 pooled!
    assert res["auc_dir"] == pytest.approx(0.75, abs=1e-4)
    assert res["auc_impact"] == pytest.approx(0.75, abs=1e-4)
    assert res["auc_ordinal"] == pytest.approx(0.75, abs=1e-4)
    assert res["auc_selective_hit"] == pytest.approx(0.75, abs=1e-4)
    assert res["pooled_auc_dir"] > 0.90
    assert res["pooled_auc_ordinal"] > 0.90

    # Orientation-flip invariance on macro-averaged metrics
    flipped = df.copy()
    flipped.iloc[::2, flipped.columns.get_loc("label")] *= -1
    flipped.iloc[::2, flipped.columns.get_loc("binding_diff")] *= -1
    preds_f = preds.copy()
    preds_f[::2] *= -1
    res_f = compute_per_antigen_macro_metrics(flipped, preds_f, pimps, min_pairs=5)
    assert res_f["auc_dir"] == res["auc_dir"]
    assert res_f["auc_impact"] == res["auc_impact"]
    assert res_f["auc_ordinal"] == res["auc_ordinal"]
    assert res_f["auc_selective_hit"] == res["auc_selective_hit"]


def test_cluster_bootstrap_and_croissant_metadata() -> None:
    """Verify target-level LOACO cluster-bootstrap aggregation and Croissant RAI metadata."""
    import json

    per_cl = {
        "C1": {"n_impactful": 20, "n_neutral": 20, "auc_dir": 0.90, "auc_impact": 0.80, "auc_ordinal": 0.85, "auc_selective_hit": 0.82},
        "C2": {"n_impactful": 10, "n_neutral": 10, "auc_dir": 0.70, "auc_impact": 0.60, "auc_ordinal": 0.65, "auc_selective_hit": 0.62},
        "C3": {"n_impactful": 0, "n_neutral": 15, "auc_dir": None, "auc_impact": None, "auc_ordinal": None, "auc_selective_hit": None},
    }
    s = compute_cluster_bootstrap_summary(per_cl, min_class_pairs=2, n_bootstrap=500, seed=42)
    assert s["n_eligible_clusters"] == 3
    assert s["ordinal"]["coverage"] == "2/3"
    assert s["auc_dir"] == pytest.approx(0.80, abs=1e-4)
    assert s["auc_impact"] == pytest.approx(0.70, abs=1e-4)
    assert s["auc_ordinal"] == pytest.approx(0.75, abs=1e-4)
    assert s["ordinal"]["cluster_ci95"] == [0.65, 0.85]

    paths = RepoPaths.from_root()
    croissant_path = paths.metadata / "croissant.json"
    assert croissant_path.exists()
    cr = json.loads(croissant_path.read_text(encoding="utf-8"))
    assert cr["@type"] == "sc:Dataset"
    assert "http://mlcommons.org/croissant/1.0" in cr["conformsTo"]
    assert "rai:dataBiases" in cr and "rai:personalSensitiveInformation" in cr
    from abrank.kaggle_io import resolve_benchmark_dir

    try:
        bdir = resolve_benchmark_dir(repo_root=paths.root, auto_download=False)
    except FileNotFoundError:
        bdir = None
    if bdir is not None:
        for fobj in cr["distribution"]:
            if fobj["@type"] == "cr:FileObject":
                fname = Path(fobj["contentUrl"]).name
                assert (bdir / fname).exists()


def test_cofolding_manifest_and_dockq_invariance(tmp_path) -> None:
    """Verify the 851-scaffold co-folding manifest, standalone input exporter, and SE(3)-invariant DockQ."""
    import importlib.util
    import json

    paths = RepoPaths.from_root()
    man_path = paths.metadata / "cofolding_manifest.tsv"
    assert man_path.exists()
    df = pd.read_csv(man_path, sep="\t", keep_default_na=False)
    assert len(df) == 851
    n_cc = int((df["is_experimental_cocrystal"].astype(str).str.lower() == "true").sum())
    assert n_cc == 340

    spec1 = importlib.util.spec_from_file_location("cf01", paths.root / "scripts" / "cofolding" / "01_generate_cofolding_inputs.py")
    assert spec1 is not None and spec1.loader is not None
    cf01 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(cf01)

    cf01.export_method_inputs(df.head(3), tmp_path, ["af3", "boltz2", "chai1", "protenix"], num_poses=10, allow_templates=False)
    af3_files = list((tmp_path / "af3").glob("*.json"))
    assert len(af3_files) == 3
    sample_af3 = json.loads(af3_files[0].read_text())
    assert sample_af3["modelSeeds"] == [101, 102]
    assert all(s["protein"]["templates"] == [] for s in sample_af3["sequences"])

    spec3 = importlib.util.spec_from_file_location("cf03", paths.root / "scripts" / "cofolding" / "03_ingest_and_evaluate_cofolded.py")
    assert spec3 is not None and spec3.loader is not None
    cf03 = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(cf03)

    rng = np.random.default_rng(11)
    ca_ab = rng.normal(scale=4.0, size=(30, 3))
    ca_ag = rng.normal(scale=4.0, size=(40, 3)) + np.array([5.0, 0.0, 0.0])
    # Rigid-body rotated + translated copy must yield DockQ == 1.0
    Q_rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q_rot) < 0:
        Q_rot[:, 0] *= -1
    t_vec = np.array([12.5, -7.2, 3.3])
    dq_ident = cf03.compute_dockq_ca(ca_ab @ Q_rot.T + t_vec, ca_ag @ Q_rot.T + t_vec, ca_ab, ca_ag)
    assert dq_ident["DockQ"] == pytest.approx(1.0, abs=1e-3)
    # Displaced antibody decoy must yield low DockQ
    dq_bad = cf03.compute_dockq_ca(ca_ab + np.array([25.0, 0.0, 0.0]), ca_ag, ca_ab, ca_ag)
    assert dq_bad["DockQ"] < 0.15


def test_structure_perturbation_preserve_bond_lengths_and_wrong_epitope() -> None:
    """Verify rigid-residue coordinate perturbation preserves covalent bond geometry and antipodal decoy clears fnat."""
    import importlib.util
    from abrank.structures.resolver import ChainBackbone

    paths = RepoPaths.from_root()
    spec16 = importlib.util.spec_from_file_location("s16", paths.root / "scripts" / "16_run_structure_noise_and_epitope_sensitivity.py")
    assert spec16 is not None and spec16.loader is not None
    s16 = importlib.util.module_from_spec(spec16)
    spec16.loader.exec_module(s16)

    rng = np.random.default_rng(99)
    L = 25
    ca = np.cumsum(rng.normal(scale=2.2, size=(L, 3)), axis=0).astype(np.float32)
    base = np.stack([
        ca + np.array([-1.46, 0.0, 0.0], dtype=np.float32),
        ca,
        ca + np.array([1.52, 0.0, 0.0], dtype=np.float32),
        ca + np.array([1.52, 1.23, 0.0], dtype=np.float32),
        ca + np.array([0.0, -1.53, 0.0], dtype=np.float32),
    ], axis=1)
    cb_orig = ChainBackbone(seq="A" * L, coords=base)
    cb_pert = s16.perturb_chain_residue_frames(cb_orig, sigma_3d=2.0, sigma_deg=14.0, rng=rng)

    # Intra-residue N-CA, CA-C, and CA-CB bond lengths must remain invariant under SE(3) frame jitter
    d_nca_0 = np.linalg.norm(cb_orig.coords[:, 0] - cb_orig.coords[:, 1], axis=-1)
    d_nca_1 = np.linalg.norm(cb_pert.coords[:, 0] - cb_pert.coords[:, 1], axis=-1)
    assert np.allclose(d_nca_0, d_nca_1, atol=1e-4)

    # Wrong antipodal epitope placement on an extended antigen must eliminate native contacts
    ag_x = np.linspace(0.0, 60.0, 30, dtype=np.float32)
    ag_ca = np.stack([ag_x, np.zeros(30, dtype=np.float32), np.zeros(30, dtype=np.float32)], axis=-1)
    cb_ag = ChainBackbone(seq="G" * 30, coords=np.repeat(ag_ca[:, None, :], 5, axis=1))
    ab_ca = np.stack([np.linspace(-2.0, 4.0, 15, dtype=np.float32), np.full(15, 6.0, dtype=np.float32), np.zeros(15, dtype=np.float32)], axis=-1)
    cb_ab = ChainBackbone(seq="Y" * 15, coords=np.repeat(ab_ca[:, None, :], 5, axis=1))
    moved_h, _, _ = s16.place_native_fv_at_wrong_epitope(cb_ab, None, cb_ag)
    # Originally near x=0 (ag_ca[0]); antipodal placement must move near x=60 (ag_ca[-1])
    assert float(np.linalg.norm(moved_h.coords[:, 1].mean(0) - ag_ca[-1])) < float(np.linalg.norm(moved_h.coords[:, 1].mean(0) - ag_ca[0]))


def test_flagship_suites_maturation_discovery_transfer() -> None:
    """Verify `--suite maturation | discovery | transfer` evaluator with both pair_id and pointwise AbAgID inputs."""
    from abrank.evaluate import evaluate_flagship_suite
    from abrank.kaggle_io import resolve_benchmark_dir

    paths = RepoPaths.from_root()
    try:
        bdir = resolve_benchmark_dir(repo_root=paths.root, auto_download=False)
    except FileNotFoundError:
        pytest.skip("Benchmark tables not downloaded locally (set ABRANK_DATA_DIR or run scripts/00_download_from_kaggle.py).")

    # 1. Maturation suite from pointwise oracle `AbAgID -> binding_score` must hit 1.0000 Ordinal & Sign AUC
    kd = pd.read_csv(bdir / "assays_kd.tsv", sep="\t", usecols=["AbAgID", "binding_score"], low_memory=False).drop_duplicates("AbAgID")
    pw_oracle = pd.DataFrame({"AbAgID": kd["AbAgID"], "pred_score": kd["binding_score"]})
    res_mat = evaluate_flagship_suite("maturation", pw_oracle, data_dir=bdir, track="wt-3d", n_bootstrap=50)
    assert res_mat["suite"] == "AbRank-Maturation"
    assert res_mat["n_pairs"] == 12267
    assert res_mat["loaco_ordinal_auc"] == pytest.approx(1.0, abs=1e-4)
    assert res_mat["loaco_sign_auc"] == pytest.approx(1.0, abs=1e-4)

    # 2. Discovery suite from zero/constant scores must yield 0.5000 Top-Binder AUC across all panels
    pw_zero = pd.DataFrame({"AbAgID": kd["AbAgID"], "pred_score": np.zeros(len(kd))})
    res_disc = evaluate_flagship_suite("discovery", pw_zero, data_dir=bdir, n_bootstrap=20)
    assert res_disc["suite"] == "AbRank-Discovery"
    assert res_disc["n_panels"] >= 22
    assert res_disc["mean_top_binder_auc"] == pytest.approx(0.50, abs=1e-4)

    # 3. Transfer suite on viral IC50 oracle
    ic50 = pd.read_csv(bdir / "splits" / "transfer" / "ic50_test.tsv", sep="\t")
    tr_oracle = pd.DataFrame({"pair_id": ic50["pair_id"], "pred_diff": ic50["binding_diff"], "pred_impact": np.abs(ic50["binding_diff"])})
    res_tr = evaluate_flagship_suite("transfer", tr_oracle, data_dir=bdir, n_bootstrap=20)
    assert res_tr["tracks"]["viral_ic50"]["auc_ordinal"] == pytest.approx(1.0, abs=1e-4)


def test_logica_antisymmetry_and_zero_init() -> None:
    """Verify exact odd antisymmetry (`d(A,B) = -d(B,A)`), even symmetry (`q(A,B) = q(B,A)`),
    and zero-initialized `tanh(0)` cross-attention gates in 1D `LogiCA`."""
    import torch
    from abrank.baselines.logica_models import _DualLMLogiCANet

    rng = np.random.default_rng(7)
    n = 12
    ab1 = torch.from_numpy(rng.normal(size=(n, 64)).astype(np.float32))
    ab2 = torch.from_numpy(rng.normal(size=(n, 64)).astype(np.float32))
    ag1 = torch.from_numpy(rng.normal(size=(n, 48)).astype(np.float32))
    ag2 = torch.from_numpy(rng.normal(size=(n, 48)).astype(np.float32))
    xo = torch.from_numpy(rng.normal(size=(n, 16)).astype(np.float32))
    xe = torch.from_numpy(np.abs(rng.normal(size=(n, 20))).astype(np.float32))

    net_1d = _DualLMLogiCANet(ab_dim=64, ag_dim=48, odd_dim=16, even_dim=20)
    net_1d.eval()
    assert float(torch.tanh(net_1d.gate_ab)) == 0.0
    assert float(torch.tanh(net_1d.gate_ag)) == 0.0
    assert float(torch.tanh(net_1d.gate_logit)) == 0.0

    with torch.no_grad():
        d_f, q_f = net_1d(ab1, ab2, ag1, ag2, xo, xe)
        d_r, q_r = net_1d(ab2, ab1, ag2, ag1, -xo, xe)
    assert float(torch.max(torch.abs(d_f + d_r))) == pytest.approx(0.0, abs=1e-5)
    assert float(torch.max(torch.abs(q_f - q_r))) == pytest.approx(0.0, abs=1e-5)


def test_kaggle_data_dir_resolution(tmp_path: Path) -> None:
    """Verify `resolve_benchmark_dir` supports flat, `benchmark/`, and `data/benchmark/` layouts."""
    from abrank.kaggle_io import resolve_benchmark_dir

    bench = tmp_path / "kaggle_dl" / "benchmark"
    (bench / "splits" / "unseen_ag").mkdir(parents=True)
    (bench / "assays_kd.tsv").write_text("AbAgID\n1\n", encoding="utf-8")
    (bench / "splits" / "unseen_ag" / "test.tsv").write_text("pair_id\n1\n", encoding="utf-8")

    assert resolve_benchmark_dir(data_dir=tmp_path / "kaggle_dl", auto_download=False) == bench.resolve()
    assert resolve_benchmark_dir(data_dir=bench, auto_download=False) == bench.resolve()


def test_generative_flagship_suite_evaluator() -> None:
    """Verify `--suite generative` (`evaluate_flagship_suite`) scores both pointwise
    generative log-likelihoods (`AbAgID,pred_score`) and pairwise 3-state DPO verifiers."""
    from abrank.evaluate import evaluate_flagship_suite
    from abrank.kaggle_io import resolve_benchmark_dir

    try:
        bdir = resolve_benchmark_dir(auto_download=False)
    except FileNotFoundError:
        pytest.skip("Benchmark tables not downloaded locally (set ABRANK_DATA_DIR or run scripts/00_download_from_kaggle.py).")
    panels_path = bdir / "splits" / "generative" / "generative_panels.tsv"
    pairs_path = bdir / "splits" / "generative" / "generative_pairs.tsv"
    assert panels_path.exists() and pairs_path.exists()

    gen_pw = pd.read_csv(panels_path, sep="\t", low_memory=False)
    gen_pairs = pd.read_csv(pairs_path, sep="\t", low_memory=False)
    assert len(gen_pw) >= 1300
    assert len(gen_pairs) >= 2000

    # Oracle pointwise test: perfect `pred_score = binding_score` achieves 0 regret and ~1.0 Top-Binder AUC
    pw_preds = pd.DataFrame({"AbAgID": gen_pw["AbAgID"], "pred_score": gen_pw["binding_score"]})
    res_pw = evaluate_flagship_suite("generative", pw_preds, data_dir=bdir, n_bootstrap=20)
    assert res_pw["suite"] == "AbRank-Generative"
    assert res_pw["mean_top20_regret_pkd"] == pytest.approx(0.0, abs=1e-6)
    assert res_pw["mean_top_binder_auc"] > 0.99
    assert res_pw["generative_ordinal_auc"] > 0.99

    # Pairwise 3-state verifier test (`pair_id`, `pred_diff`, `pred_impact`)
    pair_preds = pd.DataFrame({
        "pair_id": gen_pairs["pair_id"],
        "pred_diff": gen_pairs["binding_diff"],
        "pred_impact": np.abs(gen_pairs["binding_diff"]),
    })
    res_pair = evaluate_flagship_suite("generative", pair_preds, data_dir=bdir, n_bootstrap=20)
    assert res_pair["generative_ordinal_auc"] > 0.99
    assert res_pair["generative_sign_auc"] > 0.99
