# AbRank v2 — Unseen-Antigen (`unseen_ag`) 3-State & Full-Panel Baseline Results (29 Models)

## Executive Summary

AbRank v2 evaluates **29 sequence, multi-chain, 3D inverse-folding, biophysical energy, normal-mode, and 3D geometric GNN baselines** on unseen homologous antigen clusters (`<50%` sequence identity + zero subdomain overlap) and unseen antibody lineages (`<80%` paired Fv / VHH identity). **All within-antigen pairwise AUCs (`Sign`, `Impact`, `Ordinal`, `Hit`) and continuous panel metrics (`Spearman ρ`, `Top-Binder AUC`) are Per-Antigen Macro-Averaged** (each held-out antigen or `(antigen, modality)` stratum with `≥5` pairs is scored independently and averaged with equal weight across antigens so mega-panels cannot dominate; exact Pooled Micro-AUCs are also preserved in JSON and Section 5 below):

- **Best Per-Antigen Macro 3-Class Ordinal Local Mutation Ranker (`Depleted -1 < Neutral 0 < Improved +1`, `5` held-out antigens with `≥5` pairs, `n = 556` total pairs)**: **`SaProt-650M (3-State Hurdle)`** leads overall (**`0.754` Macro Ordinal AUC**, **`0.835` Sign AUC**, **`0.648` Impact AUC**; Pooled Micro Ordinal: `0.641`), followed by **`GearBind (3D Geometric GNN)`** (**`0.723` Macro Ordinal**, `0.693` Sign, `0.634` Impact).
- **Best Per-Antigen Mutation Impact / Hot-Spot Detector (`|label| = 1` vs `label = 0`)**: **`ANTIPASTI (Normal-Mode 3D CNN)`** reaches **`0.783` Macro Impact AUC** (`95% CI: [0.673, 0.845]`), ahead of **`PRODIGY+SASA (3D Energy ZS)`** (`0.670`).
- **Best Per-Antigen Directional Sign Rankers (`+1` vs `-1` on $\ge 10\times$ mutations)**: **`SaProt-650M (3-State Hurdle)`** (`0.835`) and **`ESMC-300M (Multi-Scale Binary)`** (`0.798`).
- **Unfiltered Continuous Variant Panels (`test_pointwise.tsv`, all `320` panel variants across `6` held-out panels)**: **`ESMC-300M (Multi-Scale Binary)`** (`0.611` mean Top-Binder AUC) and **`ESMC-300M (3-State Hurdle)`** (`0.601`), with **`Biophys+AAC (Linear)`** (`+0.189`) leading continuous Spearman $\rho$.
- **Same-Antigen `Candidate` Ranking (`d_Lev > 3`, `5` antigens, `n = 583`) & Zero-Shot `VHH` Nanobody Transfer (`9` within-Ag clusters, `n = 480`)**: **`** reaches **`0.590` Macro Ordinal AUC** on `Candidate` pairs, and **`ESM2-150M (Binary Probe)`** reaches **`0.649` Macro Ordinal AUC** (`Sign`: `0.721`, `Impact`: `0.493`) zero-shot on single-domain VHH nanobodies.
- **Zero-Shot Transfer to Strict Unseen-Antigen Cross-Modal Assays (`10` `(Ag, Modality)` strata with `≥5` pairs, `n = 2,111` pairs)**: **`LogiCA (1D Logit Cross-Attn)`** achieves **`0.685` Macro Ordinal AUC** (`Sign`: `0.753`, `Impact`: `0.583`).

![Unseen Antigen 3-State Baseline Comparison](figures/unseen_ag_baselines_auc.png)

---

## 1. Per-Antigen Macro-Averaged 3-State Evaluation Protocol

All supervised models are trained **exclusively** on `data/benchmark/splits/unseen_ag/train.tsv` (`21,836` pairs: `16,448` impactful `$\pm 1$` with $|\Delta\text{p}K_d| \ge 1.0$ and `5,388` neutral `0` with $|\Delta\text{p}K_d| \le 0.30`), with feature scalers fit strictly on `train` and hyperparameters selected strictly on `unseen_ag/val.tsv` via **equal-weight Per-Antigen Macro-AUC** across all 6 validation antigens. At test time, every within-antigen track scores each held-out antigen (or `(Ag_group, Assay_modality)` stratum) with `≥5` evaluable pairs independently and averages with equal weight across antigens:

1. **3-State Local Mutations (`L1–L3`, `5` antigens with `≥5` pairs; `556` total pairs = `84` impactful `$\pm 1$` + `472` neutral `0`)**: Evaluates `Sign AUC` (`+1` vs `-1`), `Impact AUC` (`±1` vs `0`), `3-Class Ordinal AUC` (`-1 < 0 < +1`), and `Selective Hit AUC` (`+1` vs `(0, -1)`).
2. **3-State Same-Antigen Candidates (`Lev > 3`, `5` antigens, `583` pairs)**: Distinct antibody candidates competing for the same held-out antigen (`TSLP`, `LOX1`, `PDL1`, `IL23R`, `DKK1`).
3. **Unfiltered Variant Panels (`test_pointwise.tsv`)**: Per-panel continuous **Spearman $\rho$** and **Top-Binder ROC-AUC** macro-averaged across held-out panels.
4. **Cross-Antigen `Global` Pairs (`454` test pairs)**: Cross-target $K_d$ scale ranking across distinct unseen antigens (`g_1 != g_2`).
5. **Strict Unseen-Antigen `Cross-Modal` (`10` strata, `2,111` pairs) & Within-Antigen `VHH` (`9` antigens, `480` pairs) Transfer**: Filtered strictly to `<50%` antigen identity and `<80%` antibody/VHH identity relative to `unseen_ag/train`.

---

## 2. Main Per-Antigen Macro 3-State Local Mutation & Unfiltered Panel Results

| Baseline Model | Family | `L1–L3` Macro Sign AUC (`5` Ags) | `L1–L3` Macro Impact AUC | `L1–L3` Macro 3-Class Ordinal | `L1–L3` Macro Hit AUC | Unfiltered Panel Spearman `$\rho$` | Panel Top-Binder AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** | Naive / Classical | `0.500` [0.50, 0.50] | `0.500` [0.50, 0.50] | **`0.500`** [0.50, 0.50] | `0.500` | `0.000` | `0.500` |
| **Biophys+AAC (Linear)** | Naive / Classical | `0.589` [0.51, 0.74] | `0.511` [0.35, 0.68] | **`0.605`** [0.53, 0.71] | `0.608` | `0.189` | `0.507` |
| **k-mer Dipeptide (Ridge)** | Naive / Classical | `0.467` [0.30, 0.65] | `0.593` [0.42, 0.72] | **`0.503`** [0.36, 0.65] | `0.502` | `0.081` | `0.559` |
| **ESMC-300M Cosine (Zero-Shot)** | Zero-Shot PLM | `0.276` [0.16, 0.44] | `0.371` [0.27, 0.51] | **`0.396`** [0.29, 0.54] | `0.384` | `0.085` | `0.543` |
| **ESM2-150M (Zero-Shot)** | Zero-Shot PLM | `0.433` [0.28, 0.66] | `0.475` [0.33, 0.62] | **`0.459`** [0.33, 0.61] | `0.457` | `-0.088` | `0.455` |
| **AbLang2-Paired (Zero-Shot)** | Zero-Shot PLM | `0.504` [0.35, 0.65] | `0.472` [0.34, 0.64] | **`0.486`** [0.41, 0.57] | `0.486` | `0.022` | `0.519` |
| **ESMC-300M (Zero-Shot)** | Zero-Shot PLM | `0.384` [0.25, 0.58] | `0.488` [0.34, 0.64] | **`0.468`** [0.34, 0.60] | `0.457` | `-0.028` | `0.472` |
| **ESMC+AbLang2 (Zero-Shot)** | Zero-Shot PLM | `0.476` [0.26, 0.65] | `0.481` [0.32, 0.64] | **`0.493`** [0.35, 0.61] | `0.489` | `-0.021` | `0.476` |
| **ESM2-150M (Binary Probe)** | Impact-Blind Binary Probe | `0.541` [0.38, 0.70] | `0.414` [0.26, 0.60] | **`0.574`** [0.46, 0.67] | `0.570` | `0.017` | `0.520` |
| **AbLang2-Paired (Binary Probe)** | Impact-Blind Binary Probe | `0.406` [0.25, 0.52] | `0.518` [0.39, 0.64] | **`0.446`** [0.36, 0.52] | `0.443` | `-0.179` | `0.408` |
| **ESMC-300M (Raw-Diff Binary Probe)** | Impact-Blind Binary Probe | `0.532` [0.38, 0.76] | `0.407` [0.29, 0.54] | **`0.547`** [0.43, 0.67] | `0.543` | `-0.018` | `0.578` |
| **ESMC-300M (Multi-Scale Binary)** | Impact-Blind Binary Probe | `0.798` [0.66, 0.94] | `0.596` [0.44, 0.72] | **`0.710`** [0.61, 0.83] | `0.721` | `0.084` | `0.611` |
| **ESMC+AbLang2+LLR (Two-Head Binary)** | Impact-Blind Binary Probe | `0.721` [0.57, 0.87] | `0.480` [0.35, 0.63] | **`0.682`** [0.59, 0.79] | `0.687` | `-0.025` | `0.535` |
| **ESMC-300M (3-State Hurdle)** | 3-State Hurdle / Neural | `0.784` [0.64, 0.93] | `0.577` [0.41, 0.71] | **`0.711`** [0.62, 0.82] | `0.719` | `0.063` | `0.601` |
| **ESMC+AbLang2+LLR (3-State Hurdle)** | 3-State Hurdle / Neural | `0.722` [0.57, 0.87] | `0.492` [0.35, 0.66] | **`0.678`** [0.59, 0.78] | `0.683` | `-0.066` | `0.525` |
| **ESMC+AbLang2+LLR (3-State XGBoost)** | 3-State Hurdle / Neural | `0.700` [0.55, 0.83] | `0.503` [0.37, 0.64] | **`0.667`** [0.57, 0.75] | `0.669` | `-0.171` | `0.491` |
| **ESMC+AbLang2+LLR (3-State Bilinear)** | 3-State Hurdle / Neural | `0.661` [0.44, 0.79] | `0.527` [0.40, 0.67] | **`0.598`** [0.47, 0.70] | `0.603` | `-0.059` | `0.542` |
| **ESMC+AbLang2+LLR (3-State Siamese)** | 3-State Hurdle / Neural | `0.525` [0.39, 0.69] | `0.441` [0.30, 0.59] | **`0.553`** [0.45, 0.65] | `0.551` | `0.013` | `0.566` |
| **PRODIGY+SASA (3D Energy ZS)** | Zero-Shot 3D / Multi-Chain | `0.693` [0.53, 0.88] | `0.670` [0.53, 0.79] | **`0.633`** [0.49, 0.77] | `0.639` | `0.148` | `0.561` |
| **ProteinMPNN (3D Complex ZS)** | Zero-Shot 3D / Multi-Chain | `0.594` [0.46, 0.79] | `0.530` [0.39, 0.69] | **`0.572`** [0.44, 0.70] | `0.575` | `-0.013` | `0.480` |
| **AntiFold (3D Ab-Ag ZS)** | Zero-Shot 3D / Multi-Chain | `0.532` [0.33, 0.72] | `0.396` [0.27, 0.54] | **`0.497`** [0.36, 0.64] | `0.497` | `0.011` | `0.472` |
| **ESM-IF1 (3D Complex ZS)** | Zero-Shot 3D / Multi-Chain | `0.496` [0.40, 0.68] | `0.631` [0.49, 0.76] | **`0.461`** [0.36, 0.60] | `0.464` | `0.114` | `0.541` |
| **SaProt-650M (3D-Token ZS)** | Zero-Shot 3D / Multi-Chain | `0.461` [0.27, 0.65] | `0.485` [0.34, 0.64] | **`0.498`** [0.36, 0.62] | `0.492` | `0.026` | `0.430` |
| **MINT-650M (Multi-Chain ZS)** | Zero-Shot 3D / Multi-Chain | `0.532` [0.36, 0.75] | `0.537` [0.39, 0.67] | **`0.555`** [0.40, 0.71] | `0.550` | `-0.053` | `0.475` |
| **SaProt-650M (3-State Hurdle)** | 3D Structure / Multi-Chain 3-State | `0.835` [0.68, 0.89] | `0.648` [0.49, 0.77] | **`0.754`** [0.66, 0.82] | `0.765` | `0.102` | `0.561` |
| **MINT-650M (3-State Hurdle)** | 3D Structure / Multi-Chain 3-State | `0.404` [0.30, 0.56] | `0.424` [0.30, 0.58] | **`0.438`** [0.34, 0.57] | `0.431` | `0.094` | `0.549` |
| **ANTIPASTI (Normal-Mode 3D CNN)** | 3D Structure / Multi-Chain 3-State | `0.690` [0.61, 0.92] | `0.783` [0.67, 0.85] | **`0.667`** [0.58, 0.80] | `0.668` | `0.132` | `0.486` |
| **GearBind (3D Geometric GNN)** | 3D Structure / Multi-Chain 3-State | `0.693` [0.61, 0.92] | `0.634` [0.46, 0.81] | **`0.723`** [0.63, 0.82] | `0.717` | `0.093` | `0.540` |
| **WALLE-Affinity (3-State 3D Graph)** | 3D Structure / Multi-Chain 3-State | `0.542` [0.41, 0.74] | `0.599` [0.44, 0.74] | **`0.523`** [0.39, 0.66] | `0.522` | `-0.121` | `0.511` |
| **LogiCA (1D Logit Cross-Attn)** | 3-State Hurdle / Neural | `0.634` [0.45, 0.76] | `0.549` [0.39, 0.71] | **`0.622`** [0.49, 0.71] | `0.619` | `-0.019` | `0.500` |

---

## 3. Multi-Regime Generalization (`Candidate`, `Global`, `Cross-Modal`, and `Unseen-Ag VHH` — Per-Antigen Macro)

| Baseline Model | `Candidate` Macro Sign / Imp / Ord (`5` Ags) | `Global` Cross-Ag AUC (`n=454`) | `Cross-Modal` Macro Sign / Imp / Ord (`10` Strata) | `Unseen-Ag VHH` Macro Sign / Imp / Ord (`9` Ags) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Baseline** | `0.500` / `0.500` / **`0.500`** | `0.500` | `0.500` / `0.500` / **`0.500`** | `0.500` / `0.500` / **`0.500`** |
| **Biophys+AAC (Linear)** | `0.573` / `0.607` / **`0.537`** | `0.150` | `0.597` / `0.508` / **`0.543`** | `0.530` / `0.459` / **`0.503`** |
| **k-mer Dipeptide (Ridge)** | `0.533` / `0.559` / **`0.520`** | `0.231` | `0.526` / `0.586` / **`0.528`** | `0.453` / `0.551` / **`0.428`** |
| **ESMC-300M Cosine (Zero-Shot)** | `0.553` / `0.481` / **`0.531`** | `0.736` | `0.523` / `0.538` / **`0.515`** | `0.518` / `0.527` / **`0.525`** |
| **ESM2-150M (Zero-Shot)** | `0.508` / `0.491` / **`0.501`** | `0.792` | `0.490` / `0.469` / **`0.499`** | `0.561` / `0.471` / **`0.527`** |
| **AbLang2-Paired (Zero-Shot)** | `0.494` / `0.493` / **`0.485`** | `0.680` | `0.555` / `0.447` / **`0.517`** | `0.500` / `0.500` / **`0.500`** |
| **ESMC-300M (Zero-Shot)** | `0.537` / `0.454` / **`0.515`** | `0.768` | `0.525` / `0.477` / **`0.531`** | `0.593` / `0.418` / **`0.558`** |
| **ESMC+AbLang2 (Zero-Shot)** | `0.534` / `0.480` / **`0.512`** | `0.769` | `0.540` / `0.445` / **`0.538`** | `0.593` / `0.418` / **`0.558`** |
| **ESM2-150M (Binary Probe)** | `0.470` / `0.502` / **`0.479`** | `0.268` | `0.582` / `0.502` / **`0.548`** | `0.721` / `0.493` / **`0.649`** |
| **AbLang2-Paired (Binary Probe)** | `0.311` / `0.477` / **`0.394`** | `0.214` | `0.597` / `0.468` / **`0.559`** | `0.500` / `0.500` / **`0.500`** |
| **ESMC-300M (Raw-Diff Binary Probe)** | `0.446` / `0.519` / **`0.464`** | `0.219` | `0.455` / `0.597` / **`0.460`** | `0.443` / `0.354` / **`0.464`** |
| **ESMC-300M (Multi-Scale Binary)** | `0.448` / `0.484` / **`0.475`** | `0.202` | `0.467` / `0.532` / **`0.480`** | `0.321` / `0.406` / **`0.414`** |
| **ESMC+AbLang2+LLR (Two-Head Binary)** | `0.350` / `0.502` / **`0.410`** | `0.423` | `0.511` / `0.555` / **`0.505`** | `0.302` / `0.483` / **`0.393`** |
| **ESMC-300M (3-State Hurdle)** | `0.495` / `0.505` / **`0.495`** | `0.219` | `0.467` / `0.503` / **`0.479`** | `0.359` / `0.398` / **`0.442`** |
| **ESMC+AbLang2+LLR (3-State Hurdle)** | `0.380` / `0.507` / **`0.426`** | `0.423` | `0.517` / `0.591` / **`0.481`** | `0.357` / `0.390` / **`0.435`** |
| **ESMC+AbLang2+LLR (3-State XGBoost)** | `0.405` / `0.459` / **`0.433`** | `0.549` | `0.481` / `0.521` / **`0.458`** | `0.470` / `0.534` / **`0.453`** |
| **ESMC+AbLang2+LLR (3-State Bilinear)** | `0.446` / `0.501` / **`0.463`** | `0.185` | `0.547` / `0.548` / **`0.498`** | `0.482` / `0.386` / **`0.502`** |
| **ESMC+AbLang2+LLR (3-State Siamese)** | `0.504` / `0.508` / **`0.503`** | `0.608` | `0.521` / `0.571` / **`0.501`** | `0.557` / `0.366` / **`0.539`** |
| **PRODIGY+SASA (3D Energy ZS)** | `0.624` / `0.501` / **`0.568`** | `0.880` | `0.524` / `0.567` / **`0.513`** | `0.378` / `0.481` / **`0.438`** |
| **ProteinMPNN (3D Complex ZS)** | `0.438` / `0.487` / **`0.464`** | `0.461` | `0.583` / `0.546` / **`0.540`** | `0.472` / `0.445` / **`0.485`** |
| **AntiFold (3D Ab-Ag ZS)** | `0.596` / `0.476` / **`0.551`** | `0.769` | `0.442` / `0.538` / **`0.465`** | `0.610` / `0.553` / **`0.540`** |
| **ESM-IF1 (3D Complex ZS)** | `0.536` / `0.504` / **`0.523`** | `0.221` | `0.523` / `0.543` / **`0.536`** | `0.394` / `0.599` / **`0.437`** |
| **SaProt-650M (3D-Token ZS)** | `0.614` / `0.442` / **`0.565`** | `0.777` | `0.605` / `0.516` / **`0.541`** | `0.670` / `0.573` / **`0.541`** |
| **MINT-650M (Multi-Chain ZS)** | `0.561` / `0.430` / **`0.544`** | `0.689` | `0.535` / `0.490` / **`0.512`** | `0.584` / `0.461` / **`0.577`** |
| **SaProt-650M (3-State Hurdle)** | `0.509` / `0.499` / **`0.499`** | `0.085` | `0.436` / `0.479` / **`0.476`** | `0.385` / `0.341` / **`0.395`** |
| **MINT-650M (3-State Hurdle)** | `0.591` / `0.484` / **`0.559`** | `0.109` | `0.452` / `0.406` / **`0.468`** | `0.586` / `0.363` / **`0.536`** |
| **ANTIPASTI (Normal-Mode 3D CNN)** | `0.590` / `0.529` / **`0.549`** | `0.170` | `0.630` / `0.537` / **`0.571`** | `0.491` / `0.619` / **`0.527`** |
| **GearBind (3D Geometric GNN)** | `0.472` / `0.547` / **`0.486`** | `0.751` | `0.689` / `0.491` / **`0.609`** | `0.677` / `0.536` / **`0.608`** |
| **WALLE-Affinity (3-State 3D Graph)** | `0.409` / `0.483` / **`0.446`** | `0.400` | `0.523` / `0.573` / **`0.495`** | `0.463` / `0.479` / **`0.527`** |
| **LogiCA (1D Logit Cross-Attn)** | `0.400` / `0.510` / **`0.452`** | `0.823` | `0.753` / `0.583` / **`0.685`** | `0.622` / `0.447` / **`0.571`** |

---

## 4. Antibody-Mutated (`Ab-Mut`) vs. Antigen-Mutated (`Ag-Mut`) Per-Antigen Macro Slice Analysis

| Model | `Ab-Mut` Macro Sign / Impact / Ordinal | `Ag-Mut` Macro Sign / Impact / Ordinal | Exact `<mask>`-Marginal Macro Sign / Impact |
| :--- | :---: | :---: | :---: |
| **Random Baseline** | `0.500` / `0.500` / `0.500` | `0.500` / `0.500` / `0.500` | — |
| **Biophys+AAC (Linear)** | `0.471` / `0.499` / `0.474` | `0.973` / `0.630` / `0.907` | — |
| **k-mer Dipeptide (Ridge)** | `0.490` / `0.533` / `0.491` | `0.580` / `0.678` / `0.586` | — |
| **ESMC-300M Cosine (Zero-Shot)** | `0.090` / `0.345` / `0.337` | `0.647` / `0.404` / `0.558` | — |
| **ESM2-150M (Zero-Shot)** | `0.465` / `0.491` / `0.464` | `0.434` / `0.452` / `0.472` | — |
| **AbLang2-Paired (Zero-Shot)** | `0.496` / `0.425` / `0.474` | `0.500` / `0.500` / `0.500` | `0.549` / `0.458` |
| **ESMC-300M (Zero-Shot)** | `0.330` / `0.437` / `0.445` | `0.467` / `0.510` / `0.502` | `0.461` / `0.371` |
| **ESMC+AbLang2 (Zero-Shot)** | `0.433` / `0.429` / `0.470` | `0.467` / `0.510` / `0.502` | `0.505` / `0.428` |
| **ESM2-150M (Binary Probe)** | `0.564` / `0.421` / `0.536` | `0.624` / `0.312` / `0.656` | — |
| **AbLang2-Paired (Binary Probe)** | `0.300` / `0.623` / `0.346` | `0.500` / `0.500` / `0.500` | — |
| **ESMC-300M (Raw-Diff Binary Probe)** | `0.639` / `0.537` / `0.658` | `0.381` / `0.457` / `0.447` | — |
| **ESMC-300M (Multi-Scale Binary)** | `0.735` / `0.649` / `0.610` | `0.973` / `0.380` / `0.871` | — |
| **ESMC+AbLang2+LLR (Two-Head Binary)** | `0.515` / `0.466` / `0.512` | `0.961` / `0.375` / `0.866` | — |
| **ESMC-300M (3-State Hurdle)** | `0.674` / `0.612` / `0.605` | `0.977` / `0.486` / `0.880` | — |
| **ESMC+AbLang2+LLR (3-State Hurdle)** | `0.493` / `0.485` / `0.498` | `0.961` / `0.476` / `0.872` | — |
| **ESMC+AbLang2+LLR (3-State XGBoost)** | `0.645` / `0.471` / `0.592` | `0.692` / `0.356` / `0.702` | — |
| **ESMC+AbLang2+LLR (3-State Bilinear)** | `0.718` / `0.476` / `0.591` | `0.532` / `0.654` / `0.585` | — |
| **ESMC+AbLang2+LLR (3-State Siamese)** | `0.401` / `0.427` / `0.435` | `0.699` / `0.327` / `0.699` | — |
| **PRODIGY+SASA (3D Energy ZS)** | `0.721` / `0.676` / `0.639` | `0.737` / `0.577` / `0.669` | — |
| **ProteinMPNN (3D Complex ZS)** | `0.518` / `0.501` / `0.490` | `0.819` / `0.731` / `0.763` | — |
| **AntiFold (3D Ab-Ag ZS)** | `0.495` / `0.325` / `0.491` | `0.517` / `0.471` / `0.494` | — |
| **ESM-IF1 (3D Complex ZS)** | `0.630` / `0.625` / `0.619` | `0.537` / `0.764` / `0.492` | — |
| **SaProt-650M (3D-Token ZS)** | `0.420` / `0.520` / `0.455` | `0.427` / `0.356` / `0.500` | — |
| **MINT-650M (Multi-Chain ZS)** | `0.493` / `0.498` / `0.546` | `0.518` / `0.452` / `0.524` | — |
| **SaProt-650M (3-State Hurdle)** | `0.724` / `0.653` / `0.627` | `0.965` / `0.683` / `0.910` | — |
| **MINT-650M (3-State Hurdle)** | `0.537` / `0.423` / `0.535` | `0.354` / `0.495` / `0.368` | — |
| **ANTIPASTI (Normal-Mode 3D CNN)** | `0.607` / `0.743` / `0.574` | `0.903` / `0.894` / `0.851` | — |
| **GearBind (3D Geometric GNN)** | `0.629` / `0.579` / `0.662` | `0.937` / `0.827` / `0.894` | — |
| **WALLE-Affinity (3-State 3D Graph)** | `0.353` / `0.645` / `0.383` | `0.745` / `0.399` / `0.682` | — |
| **LogiCA (1D Logit Cross-Attn)** | `0.729` / `0.472` / `0.642` | `0.475` / `0.635` / `0.524` | — |

---

## 5. Per-Antigen Macro-AUC (`≥5` Pairs) vs. Pooled Micro-AUC Comparison

To ensure mega-panels (`TSLP`: `82.9%` of `unseen_ag/test L1–L3`; `NbBench Galectin-3`: `48.3%` of `Cross-Modal`; `SARSCoV2_Spike`: `78.1%` of `VHH`) do not dominate benchmark rankings, primary metrics use **Per-Antigen Macro-AUC**, while exact **Pooled Micro-AUC** is reported side-by-side below:

| Model | `L1–L3` Macro Ord (`Sign`/`Imp`) | `L1–L3` Pooled Ord (`Sign`/`Imp`) | `Candidate` Macro / Pooled Ord | `Cross-Modal` Macro / Pooled Ord | `VHH` Macro / Pooled Ord |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** | **`0.500`** (`0.500`/`0.500`) | `0.500` (`0.500`/`0.500`) | **`0.500`** / `0.500` | **`0.500`** / `0.500` | **`0.500`** / `0.500` |
| **Biophys+AAC (Linear)** | **`0.605`** (`0.589`/`0.511`) | `0.529` (`0.553`/`0.538`) | **`0.537`** / `0.551` | **`0.543`** / `0.485` | **`0.503`** / `0.514` |
| **k-mer Dipeptide (Ridge)** | **`0.503`** (`0.467`/`0.593`) | `0.476` (`0.453`/`0.541`) | **`0.520`** / `0.531` | **`0.528`** / `0.478` | **`0.428`** / `0.550` |
| **ESMC-300M Cosine (Zero-Shot)** | **`0.396`** (`0.276`/`0.371`) | `0.413` (`0.312`/`0.377`) | **`0.531`** / `0.524` | **`0.515`** / `0.502` | **`0.525`** / `0.519` |
| **ESM2-150M (Zero-Shot)** | **`0.459`** (`0.433`/`0.475`) | `0.503` (`0.501`/`0.492`) | **`0.501`** / `0.480` | **`0.499`** / `0.483` | **`0.527`** / `0.470` |
| **AbLang2-Paired (Zero-Shot)** | **`0.486`** (`0.504`/`0.472`) | `0.523` (`0.564`/`0.449`) | **`0.485`** / `0.517` | **`0.517`** / `0.533` | **`0.500`** / `0.500` |
| **ESMC-300M (Zero-Shot)** | **`0.468`** (`0.384`/`0.488`) | `0.495` (`0.493`/`0.451`) | **`0.515`** / `0.529` | **`0.531`** / `0.520` | **`0.558`** / `0.464` |
| **ESMC+AbLang2 (Zero-Shot)** | **`0.493`** (`0.476`/`0.481`) | `0.503` (`0.507`/`0.490`) | **`0.512`** / `0.529` | **`0.538`** / `0.518` | **`0.558`** / `0.464` |
| **ESM2-150M (Binary Probe)** | **`0.574`** (`0.541`/`0.414`) | `0.505` (`0.506`/`0.492`) | **`0.479`** / `0.506` | **`0.548`** / `0.497` | **`0.649`** / `0.536` |
| **AbLang2-Paired (Binary Probe)** | **`0.446`** (`0.406`/`0.518`) | `0.370` (`0.245`/`0.406`) | **`0.394`** / `0.372` | **`0.559`** / `0.545` | **`0.500`** / `0.500` |
| **ESMC-300M (Raw-Diff Binary Probe)** | **`0.547`** (`0.532`/`0.407`) | `0.569` (`0.636`/`0.395`) | **`0.464`** / `0.455` | **`0.460`** / `0.476` | **`0.464`** / `0.562` |
| **ESMC-300M (Multi-Scale Binary)** | **`0.710`** (`0.798`/`0.596`) | `0.701` (`0.834`/`0.601`) | **`0.475`** / `0.493` | **`0.480`** / `0.483` | **`0.414`** / `0.544` |
| **ESMC+AbLang2+LLR (Two-Head Binary)** | **`0.682`** (`0.721`/`0.480`) | `0.611` (`0.712`/`0.493`) | **`0.410`** / `0.436` | **`0.505`** / `0.503` | **`0.393`** / `0.535` |
| **ESMC-300M (3-State Hurdle)** | **`0.711`** (`0.784`/`0.577`) | `0.733` (`0.823`/`0.742`) | **`0.495`** / `0.481` | **`0.479`** / `0.469` | **`0.442`** / `0.541` |
| **ESMC+AbLang2+LLR (3-State Hurdle)** | **`0.678`** (`0.722`/`0.492`) | `0.630` (`0.719`/`0.682`) | **`0.426`** / `0.430` | **`0.481`** / `0.491` | **`0.435`** / `0.539` |
| **ESMC+AbLang2+LLR (3-State XGBoost)** | **`0.667`** (`0.700`/`0.503`) | `0.587` (`0.642`/`0.695`) | **`0.433`** / `0.422` | **`0.458`** / `0.493` | **`0.453`** / `0.548` |
| **ESMC+AbLang2+LLR (3-State Bilinear)** | **`0.598`** (`0.661`/`0.527`) | `0.566` (`0.609`/`0.722`) | **`0.463`** / `0.458` | **`0.498`** / `0.500` | **`0.502`** / `0.577` |
| **ESMC+AbLang2+LLR (3-State Siamese)** | **`0.553`** (`0.525`/`0.441`) | `0.489` (`0.483`/`0.590`) | **`0.503`** / `0.504` | **`0.501`** / `0.507` | **`0.539`** / `0.587` |
| **PRODIGY+SASA (3D Energy ZS)** | **`0.633`** (`0.693`/`0.670`) | `0.760` (`0.886`/`0.741`) | **`0.568`** / `0.601` | **`0.513`** / `0.506` | **`0.438`** / `0.503` |
| **ProteinMPNN (3D Complex ZS)** | **`0.572`** (`0.594`/`0.530`) | `0.582` (`0.654`/`0.560`) | **`0.464`** / `0.441` | **`0.540`** / `0.521` | **`0.485`** / `0.452` |
| **AntiFold (3D Ab-Ag ZS)** | **`0.497`** (`0.532`/`0.396`) | `0.555` (`0.595`/`0.585`) | **`0.551`** / `0.572` | **`0.465`** / `0.526` | **`0.540`** / `0.432` |
| **ESM-IF1 (3D Complex ZS)** | **`0.461`** (`0.496`/`0.631`) | `0.605` (`0.697`/`0.542`) | **`0.523`** / `0.548` | **`0.536`** / `0.543` | **`0.437`** / `0.423` |
| **SaProt-650M (3D-Token ZS)** | **`0.498`** (`0.461`/`0.485`) | `0.466` (`0.422`/`0.428`) | **`0.565`** / `0.544` | **`0.541`** / `0.505` | **`0.541`** / `0.480` |
| **MINT-650M (Multi-Chain ZS)** | **`0.555`** (`0.532`/`0.537`) | `0.530` (`0.557`/`0.550`) | **`0.544`** / `0.556` | **`0.512`** / `0.497` | **`0.577`** / `0.429` |
| **SaProt-650M (3-State Hurdle)** | **`0.754`** (`0.835`/`0.648`) | `0.641` (`0.735`/`0.744`) | **`0.499`** / `0.503` | **`0.476`** / `0.512` | **`0.395`** / `0.505` |
| **MINT-650M (3-State Hurdle)** | **`0.438`** (`0.404`/`0.424`) | `0.576` (`0.604`/`0.653`) | **`0.559`** / `0.579` | **`0.468`** / `0.523` | **`0.536`** / `0.513` |
| **ANTIPASTI (Normal-Mode 3D CNN)** | **`0.667`** (`0.690`/`0.783`) | `0.714` (`0.828`/`0.749`) | **`0.549`** / `0.564` | **`0.571`** / `0.524` | **`0.527`** / `0.547` |
| **GearBind (3D Geometric GNN)** | **`0.723`** (`0.693`/`0.634`) | `0.728` (`0.870`/`0.686`) | **`0.486`** / `0.543` | **`0.609`** / `0.515` | **`0.608`** / `0.579` |
| **WALLE-Affinity (3-State 3D Graph)** | **`0.523`** (`0.542`/`0.599`) | `0.610` (`0.641`/`0.724`) | **`0.446`** / `0.461` | **`0.495`** / `0.501` | **`0.527`** / `0.540` |
| **LogiCA (1D Logit Cross-Attn)** | **`0.622`** (`0.634`/`0.549`) | `0.553` (`0.589`/`0.628`) | **`0.452`** / `0.507` | **`0.685`** / `0.506` | **`0.571`** / `0.537` |

---

## Reproducibility

```bash
# 1. Extract cached 1D PLM embeddings (`ESMC-300M`, `AbLang2-Paired`, `ESM2-150M`)
python scripts/05_extract_embeddings.py --workers 16

# 2. Resolve 851 3D scaffolds (RCSB co-crystals + IgFold Fv/VHH) and extract 3D / multi-chain features
python scripts/08_prepare_3d_structures.py
python scripts/09_extract_structure_and_multichain_features.py

# 3. Train & evaluate all 29 unseen_ag baselines and 5-track 3D DMS/ΔΔG benchmarks
python scripts/06_run_unseen_ag_baselines.py
python scripts/10_run_3d_dms_ddg_benchmarks.py
```
