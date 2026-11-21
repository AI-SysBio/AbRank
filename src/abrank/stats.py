"""Comprehensive dataset & 3-state split statistics generator for AbRank v2."""

from __future__ import annotations

import json
import pandas as pd

from abrank.config import BenchmarkConfig, RepoPaths


def _summarize_split_df(df: pd.DataFrame) -> dict:
    n_imp = int((df["label"] != 0).sum()) if len(df) else 0
    n_neu = int((df["label"] == 0).sum()) if len(df) else 0
    pos_imp = round(float((df.loc[df["label"] != 0, "label"] == 1).mean()), 4) if n_imp > 0 else 0.0
    entry = {
        "n_pairs": int(len(df)),
        "n_impactful": n_imp,
        "n_neutral": n_neu,
        "impactful_pos_frac": pos_imp,
        "by_category": {k: int(v) for k, v in df["Category"].value_counts().to_dict().items()},
        "n_groups": int(df["AbRank_group"].nunique()) if len(df) else 0,
    }
    if "Assay_modality" in df.columns:
        entry["by_modality"] = {k: int(v) for k, v in df["Assay_modality"].value_counts().to_dict().items()}
    if "ag_regime" in df.columns:
        entry["by_ag_regime"] = {k: int(v) for k, v in df["ag_regime"].value_counts().to_dict().items()}
    return entry


def generate_benchmark_report(paths: RepoPaths | None = None) -> tuple[dict, str]:
    """Inspect generated benchmark tables and splits and return `(json_dict, markdown_str)`."""
    paths = paths or RepoPaths.from_root()
    cfg = BenchmarkConfig.load(paths.configs / "benchmark.yaml")

    kd = pd.read_csv(paths.benchmark / "assays_kd.tsv", sep="\t", low_memory=False)
    cm_path = paths.benchmark / "assays_cross_modal.tsv"
    cm = pd.read_csv(cm_path, sep="\t", low_memory=False) if cm_path.exists() else pd.DataFrame()
    tr_path = paths.benchmark / "assays_transfer.tsv"
    tr = pd.read_csv(tr_path, sep="\t", low_memory=False) if tr_path.exists() else pd.DataFrame()
    vhh_path = paths.benchmark / "assays_vhh.tsv"
    vhh = pd.read_csv(vhh_path, sep="\t", low_memory=False) if vhh_path.exists() else pd.DataFrame()

    report: dict = {
        "benchmark_version": cfg.benchmark_version,
        "margin_m": cfg.margin_m,
        "neutral_margin": cfg.neutral_margin,
        "kd_core": {
            "total_assays": int(len(kd)),
            "unique_antibodies": int(kd["AbID"].nunique()),
            "unique_antigens_seq": int(kd["AgID"].nunique()),
            "unique_ag_groups": int(kd["Ag_group"].nunique()),
            "unique_ab_lineages": int(kd["Ab_lineage"].nunique()) if "Ab_lineage" in kd.columns else 0,
            "viral_ag_groups": int(kd[kd["Ag_type"] == "Viral"]["Ag_group"].nunique()),
            "nonviral_ag_groups": int(kd[kd["Ag_type"] == "Non-Viral"]["Ag_group"].nunique()),
            "by_dataset": {},
        },
        "cross_modal": {},
        "transfer": {},
        "vhh": {},
        "splits": {},
    }

    for ds, g in kd.groupby("Dataset", sort=True):
        report["kd_core"]["by_dataset"][str(ds)] = {
            "assays": int(len(g)),
            "unique_abs": int(g["AbID"].nunique()),
            "ag_groups": int(g["Ag_group"].nunique()),
            "ag_types": sorted(g["Ag_type"].unique().tolist()),
        }

    if not cm.empty:
        for (ds, mod), g in cm.groupby(["Dataset", "Assay_modality"], sort=True):
            report["cross_modal"][f"{ds} ({mod})"] = {
                "assays": int(len(g)),
                "ag_groups": int(g["Ag_group"].nunique()),
            }

    if not tr.empty:
        for (ds, mod), g in tr.groupby(["Dataset", "Assay_modality"], sort=True):
            report["transfer"][f"{ds} ({mod})"] = {
                "assays": int(len(g)),
                "unique_abs": int(g["AbID"].nunique()),
                "unique_ags": int(g["AgID"].nunique()),
            }

    if not vhh.empty:
        for ds, g in vhh.groupby("Dataset", sort=True):
            report["vhh"][str(ds)] = {
                "assays": int(len(g)),
                "ag_groups": int(g["Ag_group"].nunique()),
            }

    for tier in ("unseen_ag",):
        tdir = paths.splits / tier
        if not tdir.exists():
            continue
        report["splits"][tier] = {}
        for sp in ("train", "val", "test", "test_candidates"):
            p = tdir / f"{sp}.tsv"
            if p.exists():
                df = pd.read_csv(p, sep="\t")
                entry = _summarize_split_df(df)
                pw_path = tdir / f"{sp}_pointwise.tsv"
                if pw_path.exists():
                    pw_df = pd.read_csv(pw_path, sep="\t", low_memory=False)
                    entry["n_pointwise_variants"] = int(len(pw_df))
                report["splits"][tier][sp] = entry

    for extra_name, rel in (
        ("cross_modal", "cross_modal/modality_test.tsv"),
        ("transfer_ic50", "transfer/ic50_test.tsv"),
        ("transfer_escape", "transfer/escape_test.tsv"),
        ("vhh_ranking", "vhh/vhh_ranking_test.tsv"),
    ):
        p = paths.splits / rel
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            report["splits"][extra_name] = _summarize_split_df(df)

    md = _format_markdown_report(report)
    out_json = paths.benchmark / "benchmark_summary.json"
    out_md = paths.benchmark / "BENCHMARK_SUMMARY.md"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(md, encoding="utf-8")
    return report, md


def _format_markdown_report(r: dict) -> str:
    kd = r["kd_core"]
    lines = [
        f"# AbRank v{r['benchmark_version']} — Curated Dataset & 3-State Split Statistics (`m = {r['margin_m']}`, `m_neutral = {r['neutral_margin']}`)",
        "",
        "## 1. Core Paired-Chain `KD` Assay Inventory",
        f"- **Total deduplicated `(VH, VL, Ag)` $K_d$ assays**: **{kd['total_assays']:,}**",
        f"- **Unique paired antibodies (`AbID`)**: **{kd['unique_antibodies']:,}** across **{kd['unique_ab_lineages']:,}** clonal lineages (`<80%` Fv identity)",
        f"- **Unique antigen constructs (`AgID`)**: **{kd['unique_antigens_seq']:,}** across **{kd['unique_ag_groups']:,}** canonical antigen groups (**{kd['viral_ag_groups']}** Viral, **{kd['nonviral_ag_groups']}** Non-Viral)",
        "",
        "| Source Dataset | Unique Assays | Unique Antibodies | Canonical Antigen Groups | Target Types |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ]
    for ds, info in sorted(kd["by_dataset"].items(), key=lambda kv: -kv[1]["assays"]):
        lines.append(
            f"| `{ds}` | {info['assays']:,} | {info['unique_abs']:,} | {info['ag_groups']} | {', '.join(info['ag_types'])} |"
        )

    if r["cross_modal"]:
        lines += [
            "",
            "## 2. Cross-Modal, Transfer & VHH Assay Tracks",
            "",
            "| Track | Dataset (Modality) | Validated Assays | Antigens / Groups |",
            "| :--- | :--- | ---: | ---: |",
        ]
        for k, v in r["cross_modal"].items():
            lines.append(f"| Cross-Modal | `{k}` | {v['assays']:,} | {v['ag_groups']:,} |")
        for k, v in r["transfer"].items():
            lines.append(f"| Transfer | `{k}` | {v['assays']:,} | {v['unique_ags']:,} |")
        for k, v in r["vhh"].items():
            lines.append(f"| VHH (Single-Domain) | `{k} (KD)` | {v['assays']:,} | {v['ag_groups']:,} |")

    lines += [
        "",
        "## 3. Zero-Leakage 3-State Pairwise Splits (`{-1: Depleted, 0: Neutral, +1: Improved}`) & Unfiltered Panels",
        "",
        "| Split Tier / Track | Partition | Total Pairs | Impactful (`±1`) | Neutral (`0`) | `Global` | `L1` | `L2` | `L3` | `Candidate` | Groups | `+1` Frac (`±1`) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier in ("unseen_ag",):
        if tier not in r["splits"]:
            continue
        for sp, info in r["splits"][tier].items():
            cats = info["by_category"]
            lines.append(
                f"| `{tier}` | `{sp}` | {info['n_pairs']:,} | {info['n_impactful']:,} | {info['n_neutral']:,} | "
                f"{cats.get('Global', 0):,} | {cats.get('L1', 0):,} | {cats.get('L2', 0):,} | "
                f"{cats.get('L3', 0):,} | {cats.get('Candidate', 0):,} | {info['n_groups']} | {info['impactful_pos_frac']:.3f} |"
            )

    for extra in ("cross_modal", "transfer_ic50", "transfer_escape", "vhh_ranking"):
        if extra in r["splits"]:
            info = r["splits"][extra]
            cats = info["by_category"]
            lines.append(
                f"| `{extra}` | `test` | {info['n_pairs']:,} | {info['n_impactful']:,} | {info['n_neutral']:,} | "
                f"{cats.get('Global', 0):,} | {cats.get('L1', 0):,} | {cats.get('L2', 0):,} | "
                f"{cats.get('L3', 0):,} | {cats.get('Candidate', 0):,} | {info['n_groups']} | {info['impactful_pos_frac']:.3f} |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    _, md = generate_benchmark_report()
    print(md)


if __name__ == "__main__":
    main()
