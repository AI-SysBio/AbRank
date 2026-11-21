# AbRank-Generative (`--suite generative`) — Generative Binder Reranking, Conditional Log-Likelihood, & DPO Verifier Benchmark

Evaluates generative sequence/inverse-folding log-likelihoods and 3-state reward/DPO verifiers directly on **`1,354` wet-lab SPR/BLI-assayed generative AI designs across `26` unique antigen targets** (`IgDesign`, `Origin1`, `AIntibody`, `BoltzGen_VHH`, `Germinal`, `Germinal_VHH`, `Adaptyv_EGFR_VHH`, `Shanker`) under strict Leave-One-Antigen-Cluster-Out (`<50%` Ag + `<80%` Ab) holdouts—avoiding circular *in silico* oracle bias.

## 1. Noise-Buffered DPO Preference Audit Inside Real Generative Design Cohorts

- Across all **`183,041`** within-target candidate pairs in synthesized generative campaigns, **`20.6%` have `|ΔpKd| ≤ 0.30` (`≤2×` experimental replicate noise)**, `32.4%` lie in `0.30 < |ΔpKd| < 1.00`, and only **`47.0%` represent decisive `≥10×` (`|ΔpKd| ≥ 1.0`) improvements/depletions**.

## 2. Track A — All `1,354` Wet-Lab Generative Designs (`20` Panels across `IgDesign`, `Origin1`, `AIntibody`, `BoltzGen_VHH`, `Germinal`, `Shanker`)

| Model / Verifier | 3-State DPO (`Sign` / `Imp` / **`Ord`**) | `Top-Binder AUC` | `Prec@Top-20%` (`EF`) | `Top-20% Regret` (`ΔpKd` ↓) |
| :--- | :---: | :---: | :---: | :---: |
| **Scalar pKd Ridge (Reward Probe)** | `0.518` / `0.528` / **`0.526`** | `0.497` | **`0.197`** (`1.02×`) | `0.629` |
| **ESMC-300M (1D Gen. Log-Lik)** | `0.415` / `0.604` / **`0.428`** | `0.428` | **`0.182`** (`0.94×`) | `0.687` |
| **AbLang2-Paired (1D Gen. Log-Lik)** | `0.525` / `0.569` / **`0.527`** | `0.499` | **`0.177`** (`0.92×`) | `0.562` |
| **ESMC+AbLang2 (1D Gen. Log-Lik)** | `0.497` / `0.611` / **`0.489`** | `0.470` | **`0.199`** (`1.03×`) | `0.548` |
| **Biophys+AAC (Linear)** | `0.571` / `0.605` / **`0.563`** | `0.547` | **`0.176`** (`0.91×`) | `0.606` |
| **k-mer Dipeptide (Ridge)** | `0.512` / `0.546` / **`0.541`** | `0.475` | **`0.230`** (`1.19×`) | `0.456` |
| **ESMC Multi-Scale (Binary DPO)** | `0.470` / `0.508` / **`0.486`** | `0.453` | **`0.169`** (`0.87×`) | `0.524` |
| **ESMC-300M (3-State Verifier)** | `0.493` / `0.428` / **`0.496`** | `0.468` | **`0.161`** (`0.83×`) | `0.665` |
| **ESMC+AbLang2 (Bilinear Verifier)** | `0.357` / `0.526` / **`0.424`** | `0.455` | **`0.156`** (`0.81×`) | `0.950` |
| **LogiCA (1D Logit Verifier)** | `0.457` / `0.547` / **`0.491`** | `0.462` | **`0.142`** (`0.73×`) | `0.636` |

## 3. Track B — `351` 3D-Complex-Resolved Generative Designs (`IgDesign`, `BoltzGen_VHH`, `Germinal`, `Adaptyv_EGFR_VHH`, `Shanker`)

| Generative Log-Likelihood / 3D Verifier | All 3D Gen (`Sign` / `Imp` / **`Ord`**) | `IgDesign` Ord | *De Novo* `VHH` (`BoltzGen`+`Germinal`) Ord | `Top-Binder AUC` | `Prec@Top-20%` (`EF`) | `Regret` (`ΔpKd` ↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ESMC+AbLang2 (1D Gen. Log-Lik)** | `0.450` / `0.582` / **`0.453`** | `0.532` | `0.433` | `0.424` | **`0.207`** (`1.00×`) | `0.467` |
| **PRODIGY+SASA (3D Energy ZS)** | `0.543` / `0.518` / **`0.546`** | `0.713` | `0.428` | `0.573` | **`0.216`** (`1.04×`) | `0.463` |
| **ProteinMPNN (3D Gen. Log-Lik)** | `0.528` / `0.574` / **`0.527`** | `0.591` | `0.467` | `0.457` | **`0.209`** (`1.01×`) | `0.609` |
| **AntiFold (3D Ab-Ag Gen. Log-Lik)** | `0.652` / `0.517` / **`0.589`** | `0.669` | `0.599` | `0.467` | **`0.257`** (`1.24×`) | `0.252` |
| **ESM-IF1 (3D Gen. Log-Lik)** | `0.538` / `0.567` / **`0.492`** | `0.587` | `0.414` | `0.497` | **`0.308`** (`1.49×`) | `0.445` |
| **SaProt-650M (3D-Token Gen. Log-Lik)** | `0.551` / `0.451` / **`0.532`** | `0.464` | `0.587` | `0.448` | **`0.211`** (`1.02×`) | `0.565` |
| **MINT-650M (Multi-Chain Gen. Log-Lik)** | `0.471` / `0.517` / **`0.490`** | `0.455` | `0.550` | `0.484` | **`0.257`** (`1.24×`) | `0.376` |
| **LogiCA (1D Logit Verifier)** | `0.445` / `0.465` / **`0.488`** | `0.540` | `0.471` | `0.442` | **`0.138`** (`0.67×`) | `0.570` |
| **SaProt-650M (3-State 3D Verifier)** | `0.619` / `0.538` / **`0.575`** | `0.438` | `0.638` | `0.611` | **`0.251`** (`1.21×`) | `0.660` |
| **ANTIPASTI (Normal-Mode 3D Verifier)** | `0.628` / `0.614` / **`0.582`** | `0.509` | `0.640` | `0.512` | **`0.269`** (`1.30×`) | `0.612` |
| **GearBind (3D Geometric GNN Verifier)** | `0.736` / `0.627` / **`0.656`** | `0.596` | `0.671` | `0.479` | **`0.240`** (`1.16×`) | `0.528` |
| **WALLE-Affinity (3D Graph Verifier)** | `0.457` / `0.493` / **`0.468`** | `0.580` | `0.404` | `0.412` | **`0.152`** (`0.74×`) | `0.668` |
