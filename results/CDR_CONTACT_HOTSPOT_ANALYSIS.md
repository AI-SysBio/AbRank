# CDR vs. Framework Stratification & Single-WT 3D Hot-Spot Impact Analysis

## 1. Mutation-Region Breakdown Across `LOACO-Local` (`12,267` `L1–L3` Pairs)

| Mutation Region | Pairs (`%` of Local) | Impactful (`|ΔpKd| ≥ 1.0`, `≥10×`) | Neutral (`|ΔpKd| ≤ 0.30`, `≤2×`) |
| :--- | :---: | :---: | :---: |
| **CDR-Only** | `7,078` (`57.7%`) | `44.0%` | `56.0%` |
| **CDR+Framework** | `3,175` (`25.9%`) | `52.0%` | `48.0%` |
| **Framework-Only** | `1,280` (`10.4%`) | `24.2%` | `75.8%` |
| **Ag-Mut** | `734` (`6.0%`) | `47.7%` | `52.3%` |

## 2. Hot-Spot Detection (`Impact AUC`: `±1` vs. `0`) — 1D CDR Heuristics vs. Single-WT 3D Geometry

| Method | Family | All `LOACO` (`19` cl.) [95% CI] | `Ab-Mut` Only | `CDR-Only` Pairs | Bound WT Crystal (`18` cl.) | Unbound `IgFold` (`18` cl.) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive 1D CDR vs. Framework (Yes/No)** | 1D Sequence Heuristic | **`0.511`** `[0.50, 0.53]` | `0.512` | **`0.500`** | `0.513` | `0.513` |
| **Naive 1D CDR Fraction (#CDR / #Edits)** | 1D Sequence Heuristic | **`0.506`** `[0.48, 0.54]` | `0.504` | **`0.500`** | `0.508` | `0.508` |
| **Naive 1D CDR-H3 Priority** | 1D Sequence Heuristic | **`0.500`** `[0.48, 0.52]` | `0.505` | **`0.493`** | `0.505` | `0.505` |
| **ESMC+AbLang2 (1D PLM Zero-Shot)** | 1D PLM Zero-Shot | **`0.531`** `[0.48, 0.58]` | `0.528` | **`0.518`** | `0.535` | `0.535` |
| **WT 3D Interface Distance (-min Cβ dist)** | Single-WT 3D Geometry | **`0.607`** `[0.55, 0.67]` | `0.622` | **`0.608`** | `0.602` | `0.537` |
| **WT 3D Mutated-Site Buried ΔSASA** | Single-WT 3D Geometry | **`0.562`** `[0.52, 0.61]` | `0.577` | **`0.569`** | `0.556` | `0.508` |
| **ProteinMPNN (WT 3D Complex ZS)** | Single-WT 3D Inv. Fold | **`0.606`** `[0.56, 0.65]` | `0.565` | **`0.551`** | `0.621` | `0.588` |
| **PRODIGY+SASA (WT 3D Energy ZS)** | Single-WT 3D Energy | **`0.615`** `[0.58, 0.66]` | `0.603` | **`0.585`** | `0.624` | `0.545` |
| **ESM-IF1 (WT 3D Complex ZS)** | Single-WT 3D Inv. Fold | **`0.619`** `[0.58, 0.66]` | `0.603` | **`0.597`** | `0.617` | `0.558` |
| **ANTIPASTI (WT 3D Normal-Mode CNN)** | Single-WT 3D Dynamics | **`0.612`** `[0.56, 0.66]` | `0.594` | **`0.575`** | `0.633` | `0.585` |
| **LogiCA (1D Logit Cross-Attn)** | 3-State 1D Logit | **`0.555`** `[0.51, 0.60]` | `0.564` | **`0.548`** | `0.546` | `0.546` |
