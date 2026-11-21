# AbRank v2 — 3-State Calibration, Margin Sensitivity, & Practical Wet-Lab Hit Selection

## 1. Empirical Replicate & Cross-Platform Uncertainty Calibration

| Experimental Replicate Source | Pairs / Clones | Median `|ΔpKd|` | IQR `[Q25, Q75]` | 90th Pct | `% ≤ 0.30` (`≤2×`) | `% < 1.00` (`<10×`) |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: |
| **All Within-Platform Technical Replicates** | 483 | `0.067` | `[0.037, 0.122]` | `0.221` | **`95.2%`** | **`99.8%`** |
| `AIntibody` HT-SPR Duplicate Clones | 32 | `0.039` | `[0.010, 0.074]` | `0.092` | **`93.8%`** | **`96.9%`** |
| `AIntibody` Multi-Replicate SPR SD (`n_spr` ≥ 2) | 408 | `0.077` | `[0.044, 0.130]` | `0.209` | **`96.8%`** | **`100.0%`** |
| `SKEMPI 2.0` Identical-Sequence Entries | 43 | `0.000` | `[0.000, 0.000]` | `0.646` | **`81.4%`** | **`100.0%`** |
| **Cross-Platform (`SPR` vs `KinExA`) & Multi-Crystal** | 109 | `0.398` | `[0.160, 0.555]` | `0.646` | **`34.9%`** | **`99.1%`** |
| `AIntibody` Paired `SPR` vs. `KinExA` | 99 | `0.405` | `[0.205, 0.560]` | `0.634` | **`32.3%`** | **`100.0%`** |

## 2. Practical Wet-Lab Hit Prioritization on Unfiltered Variant Panels (`25` LOACO Panels, `1,237` Variants)

| Method | Spearman `ρ` | Top-Binder AUC | Precision@Top-20% (`Base = 0.214`) | Enrichment (`EF@20%`) | Precision@Top-3 | Neutral/Worse `FDR@20%` | Mean `ΔpKd` Over Median |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scalar pKd Ridge Regression** | `-0.001` | `0.526` | **`0.316`** | `1.48×` | `0.327` | `0.632` | `-0.040` |
| **Scalar pKd MLP Regression** | `+0.015` | `0.525` | **`0.218`** | `1.02×` | `0.193` | `0.718` | `-0.110` |
| **ESMC+AbLang2 (1D ZS)** | `-0.042` | `0.458` | **`0.188`** | `0.88×` | `0.173` | `0.753` | `-0.175` |
| **PRODIGY+SASA (3D Energy ZS)** | `+0.092` | `0.538` | **`0.252`** | `1.18×` | `0.220` | `0.706` | `-0.134` |
| **ProteinMPNN (3D Complex ZS)** | `+0.125` | `0.550` | **`0.212`** | `0.99×` | `0.200` | `0.690` | `+0.009` |
| **ESMC Multi-Scale (Binary Probe)** | `-0.043` | `0.481` | **`0.236`** | `1.10×` | `0.260` | `0.729` | `-0.163` |
| **ESMC-300M (3-State Hurdle)** | `-0.100` | `0.457` | **`0.229`** | `1.07×` | `0.220` | `0.756` | `-0.165` |
| **SaProt-650M (3-State Hurdle)** | `-0.069` | `0.482` | **`0.252`** | `1.18×` | `0.220` | `0.725` | `-0.181` |
| **ANTIPASTI (Normal-Mode 3D CNN)** | `+0.113` | `0.545` | **`0.276`** | `1.29×` | `0.260` | `0.622` | `+0.001` |
| **LogiCA (1D Logit Cross-Attn)** | `+0.055` | `0.509` | **`0.200`** | `0.93×` | `0.173` | `0.755` | `-0.133` |

## 3. Sensitivity to `(m_neu, m_imp)` Thresholds (`Ordinal AUC` & Leaderboard Stability)

Leaderboard Spearman rank correlation vs. canonical `(0.30, 1.00)`: `{'neu_0.15_imp_0.50': 1.0, 'neu_0.20_imp_0.75': 0.9286, 'neu_0.30_imp_1.00': 1.0, 'neu_0.40_imp_1.25': 0.7143}`

| Model | `(0.15, 0.50)` | `(0.20, 0.75)` | **Canonical `(0.30, 1.00)`** | `(0.40, 1.25)` |
| :--- | :---: | :---: | :---: | :---: |
| **ESMC+AbLang2 (1D ZS)** | `0.485` | `0.488` | `0.476` | `0.472` |
| **PRODIGY+SASA (3D Energy ZS)** | `0.550` | `0.554` | `0.547` | `0.549` |
| **ProteinMPNN (3D Complex ZS)** | `0.553` | `0.553` | `0.557` | `0.552` |
| **ESMC Multi-Scale (Binary Probe)** | `0.487` | `0.486` | `0.483` | `0.478` |
| **ESMC-300M (3-State Hurdle)** | `0.468` | `0.472` | `0.474` | `0.488` |
| **SaProt-650M (3-State Hurdle)** | `0.467` | `0.469` | `0.468` | `0.480` |
| **ANTIPASTI (Normal-Mode 3D CNN)** | `0.545` | `0.539` | `0.536` | `0.531` |
| **LogiCA (1D Logit Cross-Attn)** | `0.523` | `0.517` | `0.508` | `0.507` |

## 4. Sensitivity to Sequence-Identity Holdout Thresholds (`3-Class Ordinal AUC`)

| Model | Loose (`<70%` Ag, `<90%` Ab) | **Canonical (`<50%` Ag, `<80%` Ab)** | Ultra-Strict (`<35%` Ag, `<70%` Ab) |
| :--- | :---: | :---: | :---: |
| **ESMC+AbLang2 (1D ZS)** | `0.512` | **`0.512`** | `0.512` |
| **PRODIGY+SASA (3D ZS)** | `0.684` | **`0.684`** | `0.684` |
| **ProteinMPNN (3D ZS)** | `0.583` | **`0.583`** | `0.583` |
| **ESMC-300M (3-State Hurdle)** | `0.532` | **`0.579`** | `0.561` |
| **SaProt-650M (3-State Hurdle)** | `0.476` | **`0.463`** | `0.485` |
| **LogiCA (1D Logit Cross-Attn)** | `0.535` | **`0.567`** | `0.557` |
