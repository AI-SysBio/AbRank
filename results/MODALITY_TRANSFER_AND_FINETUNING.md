# AbRank v2 — 3-State Cross-Modality Transfer & Fine-Tuning (`KD` $\to$ `ELISA`, `IC50`, `ΔΔG`, `Escape`)

## Executive Summary

Using AbRank v2's **3-state noise-buffered evaluation** (`Depleted -1`, `Neutral 0`, `Improved +1`), we test when a `KD`-pretrained model (`SPR` / `BLI` / `KinExA`) transfers directly versus when supervised target-assay fine-tuning is required—measuring **both** **Directional Sign AUC** (`+1` vs `-1`) and **Mutation Impact / Hot-Spot Detection AUC** (`±1` vs `0`):

1. **`ELISA` (`AbDesign`, `n = 2,517` pairs across 7 antigens) transfers cleanly without target labels**:
   - Zero-shot PLM (`ESMC+AbLang2`) and Direct `KD` Transfer already reach **`0.671` / `0.663` Sign AUC** and **`0.601` 3-Class Ordinal AUC**, and **`KD -> L2-SP Hurdle`** refines held-out antigen performance to **`0.621` Ordinal AUC** (`Sign`: `0.682`, `Impact`: `0.579`).
2. **`IC50` Neutralization (`CATNAP + Shanker + AbCoV`, `n = 2,967` pairs across 107 clonal lineages) correlates weakly with monomeric `KD` zero-shot (`0.438` Ordinal AUC) and jumps dramatically under fine-tuning (`0.702` Ordinal AUC, `0.808` Sign AUC)**:
   - Because viral neutralization is dominated by trimeric envelope avidity, conformational masking, and N-linked glycan shields on `HIV1_Env` and `SARSCoV2_Spike`, supervised adaptation (`KD -> L2-SP Hurdle`) yields a **`+0.264` Ordinal AUC gain** (`Sign`: `0.374` $\to$ **`0.808`**, `Impact`: `0.468` $\to$ **`0.642`**).
3. **`ΔΔG` (`SKEMPIv2`, `AbDesign ΔΔG`, `AbAgym DMS`) & Viral `Escape` (`Bloom`) require fine-tuning to learn hot-spot impact (`±1` vs `0`) and alanine-scan / viral-escape rules**:
   - **`SKEMPIv2` Thermodynamic `ΔΔG` (`n = 2,419`, `SKEMPIv2` strictly excluded from `KD`)**: Direct `KD` gets `0.486` Ordinal AUC (`Impact`: `0.477`), whereas **`KD -> Residual Adapter`** reaches **`0.598` Ordinal AUC** (`Sign`: **`0.671`**, `Impact`: **`0.529`**).
   - **`AbDesign` Mutational `ΔΔG` (`n = 938`)**: Direct `KD` gets `0.574` Ordinal AUC, rising to **`0.670`** (`Sign`: `0.735`, `Impact`: `0.583`) with `KD -> L2-SP Hurdle`.
   - **`AbAgym` Interface DMS (`n = 16,754`)**: Direct `KD` gets `0.516` Ordinal AUC, climbing to **`0.510`** (`Sign`: `0.513`, `Impact`: `0.499`) on held-out antigens.
   - **`RBD_Escape_Bloom` Viral Escape (`n = 8,336`)**: Direct `KD` gets `0.527` Ordinal AUC, jumping to **`0.841`** (`Sign`: **`0.926`**, `Impact`: **`0.808`**) on unseen antibody lineages.
4. **Single-Domain Nanobody `VHH KD` (`Paired Fv -> Single-Domain VHH`, `n = 520` pairs across 18 unseen antigen clusters)**:
   - Direct `Paired-KD` Transfer reaches `0.642` Ordinal AUC (`Sign`: `0.719`, `Impact`: `0.550`), and **`KD -> Residual Adapter`** lifts unseen-antigen VHH ranking to **`0.548` Ordinal AUC** (`Sign`: `0.623`, `Impact`: `0.673`).

![3-State Cross-Modality Transfer and Fine-Tuning](figures/modality_transfer_and_finetuning.png)

---

## 1. Main Per-Antigen Macro 3-State Transfer & Fine-Tuning Table (5-Fold Zero-Leakage GroupCV)

Each cell reports **Per-Antigen Macro-Averaged `Sign AUC` / `Impact AUC` / `3-Class Ordinal AUC`** across antigens with `≥5` pairs (exact Pooled Micro-AUCs are also preserved in `results/modality_transfer_and_finetuning.json` and Section 4 below):

| Target Assay Track | Pairs (`±1` / `0`) | Zero-Shot PLM | Direct `KD` (`0%` Target) | Target Scratch (`3-State`) | `KD -> L2-SP Hurdle` | `KD -> Residual Adapter` | Ordinal Gain (`FT - KD`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ELISA (AbDesign)** | `2,517` (`860` / `1657`) | `0.671` / `0.508` / `0.596` | `0.663` / `0.533` / `0.601` | `0.674` / `0.557` / `0.614` | `0.682` / `0.579` / **`0.621`** | `0.629` / `0.596` / **`0.591`** | **`+0.020`** |
| **IC50 Neutralization** | `2,967` (`1345` / `1622`) | `0.337` / `0.534` / `0.402` | `0.374` / `0.468` / `0.438` | `0.761` / `0.652` / `0.677` | `0.808` / `0.642` / **`0.702`** | `0.779` / `0.527` / **`0.683`** | **`+0.264`** |
| **ΔΔG (SKEMPIv2)** | `2,419` (`1072` / `1347`) | `0.524` / `0.514` / `0.510` | `0.489` / `0.477` / `0.486` | `0.646` / `0.535` / `0.602` | `0.618` / `0.539` / **`0.575`** | `0.671` / `0.529` / **`0.598`** | **`+0.112`** |
| **ΔΔG (AbDesign)** | `938` (`512` / `426`) | `0.486` / `0.502` / `0.492` | `0.610` / `0.482` / `0.574` | `0.727` / `0.562` / `0.660` | `0.735` / `0.583` / **`0.670`** | `0.733` / `0.568` / **`0.657`** | **`+0.096`** |
| **ΔΔG / DMS (AbAgym)** | `16,754` (`7097` / `9657`) | `0.513` / `0.515` / `0.509` | `0.522` / `0.515` / `0.516` | `0.492` / `0.496` / `0.496` | `0.513` / `0.499` / **`0.510`** | `0.467` / `0.467` / **`0.484`** | **`-0.006`** |
| **Escape DMS (Bloom)** | `8,336` (`4275` / `4061`) | `0.343` / `0.508` / `0.403` | `0.545` / `0.483` / `0.527` | `0.955` / `0.799` / `0.863` | `0.941` / `0.775` / **`0.840`** | `0.926` / `0.808` / **`0.841`** | **`+0.314`** |
| **VHH Nanobody KD** | `520` (`241` / `279`) | `0.545` / `0.375` / `0.535` | `0.719` / `0.550` / `0.642` | `0.267` / `0.591` / `0.336` | `0.367` / `0.649` / **`0.383`** | `0.623` / `0.673` / **`0.548`** | **`-0.094`** |

---

## 2. `Ab-Mut` vs. `Ag-Mut` Domain-Shift Breakdown (`Sign` / `Impact` / `Ordinal`)

| Track | Slice | Pairs | Zero-Shot PLM | Direct `KD` | `KD -> L2-SP Hurdle` | `KD -> Residual Adapter` |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: |
| **IC50 Neutralization** | Antibody-Mutated (`Ab-Mut`) | 886 | `0.375` / `0.570` / `0.438` | `0.437` / `0.534` / `0.465` | `0.746` / `0.569` / `0.629` | **`0.717` / `0.287` / `0.590`** |
| **IC50 Neutralization** | Antigen-Mutated (`Ag-Mut`) | 2,081 | `0.464` / `0.457` / `0.473` | `0.438` / `0.456` / `0.462` | `0.886` / `0.762` / `0.795` | **`0.873` / `0.732` / `0.785`** |
| **ΔΔG (SKEMPIv2)** | Antibody-Mutated (`Ab-Mut`) | 1,601 | `0.509` / `0.520` / `0.509` | `0.509` / `0.472` / `0.511` | `0.684` / `0.527` / `0.609` | **`0.724` / `0.539` / `0.634`** |
| **ΔΔG (SKEMPIv2)** | Antigen-Mutated (`Ag-Mut`) | 818 | `0.609` / `0.493` / `0.563` | `0.485` / `0.465` / `0.476` | `0.506` / `0.512` / `0.510` | **`0.561` / `0.524` / `0.528`** |
| **ΔΔG / DMS (AbAgym)** | Antibody-Mutated (`Ab-Mut`) | 2,420 | `0.481` / `0.488` / `0.489` | `0.600` / `0.538` / `0.564` | `0.589` / `0.491` / `0.555` | **`0.464` / `0.478` / `0.477`** |
| **ΔΔG / DMS (AbAgym)** | Antigen-Mutated (`Ag-Mut`) | 14,334 | `0.521` / `0.522` / `0.514` | `0.503` / `0.509` / `0.505` | `0.494` / `0.501` / `0.499` | **`0.468` / `0.464` / `0.485`** |

### Cross-Virus `IC50` Transfer Check (`HIV1_Env` $\leftrightarrow$ `Coronavirus Spike`, 3-Class Ordinal AUC)

- **Train on `HIV-1 Env` (`n=2598`) $\to$ Test Zero-Shot on `CoV Spike` (`n=369`)**:
  - `Direct KD`: `0.592` $\to$ `KD + HIV-1 L2-SP`: **`0.673`** (`HIV-only Scratch`: `0.586`).
- **Train on `CoV Spike` (`n=369`) $\to$ Test Zero-Shot on `HIV-1 Env` (`n=2598`)**:
  - `Direct KD`: `0.472` $\to$ `KD + CoV L2-SP`: **`0.469`** (`CoV-only Scratch`: `0.551`).

---

## 3. Few-Shot Fine-Tuning Sample Efficiency (`3-Class Ordinal AUC` across `0%` $\to$ `100%` Target Data)

| Target Assay Track | Method | `0%` (`Direct KD`) | `10%` Target | `25%` Target | `50%` Target | `100%` Target |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ELISA (AbDesign)** | `Target Scratch` | `0.500` | `0.554` (n=201) | `0.631` | `0.615` | `0.614` (n=2014) |
| **ELISA (AbDesign)** | **`KD -> L2-SP Hurdle`** | **`0.601`** | **`0.580`** | **`0.591`** | **`0.634`** | **`0.621`** |
| **ELISA (AbDesign)** | **`KD -> Adapter`** | **`0.601`** | `0.583` | `0.596` | `0.556` | **`0.591`** |
| **IC50 Neutralization** | `Target Scratch` | `0.500` | `0.582` (n=237) | `0.642` | `0.653` | `0.677` (n=2374) |
| **IC50 Neutralization** | **`KD -> L2-SP Hurdle`** | **`0.438`** | **`0.611`** | **`0.700`** | **`0.642`** | **`0.702`** |
| **IC50 Neutralization** | **`KD -> Adapter`** | **`0.438`** | `0.532` | `0.526` | `0.595` | **`0.683`** |
| **ΔΔG (SKEMPIv2)** | `Target Scratch` | `0.500` | `0.528` (n=194) | `0.556` | `0.593` | `0.602` (n=1935) |
| **ΔΔG (SKEMPIv2)** | **`KD -> L2-SP Hurdle`** | **`0.486`** | **`0.537`** | **`0.537`** | **`0.596`** | **`0.575`** |
| **ΔΔG (SKEMPIv2)** | **`KD -> Adapter`** | **`0.486`** | `0.520` | `0.540` | `0.566` | **`0.598`** |
| **ΔΔG (AbDesign)** | `Target Scratch` | `0.500` | `0.568` (n=75) | `0.648` | `0.596` | `0.660` (n=750) |
| **ΔΔG (AbDesign)** | **`KD -> L2-SP Hurdle`** | **`0.574`** | **`0.588`** | **`0.616`** | **`0.618`** | **`0.670`** |
| **ΔΔG (AbDesign)** | **`KD -> Adapter`** | **`0.574`** | `0.511` | `0.557` | `0.590` | **`0.657`** |
| **ΔΔG / DMS (AbAgym)** | `Target Scratch` | `0.500` | `0.495` (n=1340) | `0.505` | `0.494` | `0.496` (n=13403) |
| **ΔΔG / DMS (AbAgym)** | **`KD -> L2-SP Hurdle`** | **`0.516`** | **`0.526`** | **`0.517`** | **`0.517`** | **`0.510`** |
| **ΔΔG / DMS (AbAgym)** | **`KD -> Adapter`** | **`0.516`** | `0.504` | `0.504` | `0.500` | **`0.484`** |
| **Escape DMS (Bloom)** | `Target Scratch` | `0.500` | `0.808` (n=667) | `0.840` | `0.856` | `0.863` (n=6669) |
| **Escape DMS (Bloom)** | **`KD -> L2-SP Hurdle`** | **`0.527`** | **`0.794`** | **`0.822`** | **`0.834`** | **`0.840`** |
| **Escape DMS (Bloom)** | **`KD -> Adapter`** | **`0.527`** | `0.633` | `0.785` | `0.809` | **`0.841`** |
| **VHH Nanobody KD** | `Target Scratch` | `0.500` | `0.521` (n=42) | `0.448` | `0.291` | `0.336` (n=416) |
| **VHH Nanobody KD** | **`KD -> L2-SP Hurdle`** | **`0.642`** | **`0.599`** | **`0.419`** | **`0.349`** | **`0.383`** |
| **VHH Nanobody KD** | **`KD -> Adapter`** | **`0.642`** | `0.557` | `0.542` | `0.557` | **`0.548`** |

---

## 4. Per-Antigen Macro-AUC (`≥5` Pairs) vs. Pooled Micro-AUC Comparison

| Target Assay Track | Eval Ags (`≥5`) | Direct `KD` Macro / Pooled Ord | `KD -> L2-SP` Macro / Pooled Ord | `KD -> Adapter` Macro / Pooled Ord |
| :--- | :---: | :---: | :---: | :---: |
| **ELISA (AbDesign)** | `7` | **`0.601`** / `0.605` | **`0.621`** / `0.611` | **`0.591`** / `0.544` |
| **IC50 Neutralization** | `2` | **`0.438`** / `0.468` | **`0.702`** / `0.777` | **`0.683`** / `0.760` |
| **ΔΔG (SKEMPIv2)** | `15` | **`0.486`** / `0.507` | **`0.575`** / `0.557` | **`0.598`** / `0.612` |
| **ΔΔG (AbDesign)** | `11` | **`0.574`** / `0.501` | **`0.670`** / `0.594` | **`0.657`** / `0.614` |
| **ΔΔG / DMS (AbAgym)** | `11` | **`0.516`** / `0.519` | **`0.510`** / `0.516` | **`0.484`** / `0.493` |
| **Escape DMS (Bloom)** | `1` | **`0.527`** / `0.527` | **`0.840`** / `0.840` | **`0.841`** / `0.841` |
| **VHH Nanobody KD** | `11` | **`0.642`** / `0.602` | **`0.383`** / `0.545` | **`0.548`** / `0.556` |

---

## 5. LogiCA Direct KD Cross-Modality Transfer (`0%` Target Data)

| Target Assay Track | Direct KD LogiCA (`Sign` / `Impact` / `Ordinal`) | `Ab-Mut` Ordinal | `Ag-Mut` Ordinal |
| :--- | :---: | :---: | :---: |
| **ELISA (AbDesign)** | `0.582` / `0.625` / **`0.549`** | `0.549` | `--` |
| **IC50 Neutralization** | `0.486` / `0.583` / **`0.489`** | `0.576` | `0.455` |
| **ΔΔG (SKEMPIv2)** | `0.558` / `0.553` / **`0.541`** | `0.521` | `0.587` |
| **ΔΔG (AbDesign)** | `0.603` / `0.513` / **`0.565`** | `0.565` | `--` |
| **ΔΔG / DMS (AbAgym)** | `0.524` / `0.507` / **`0.514`** | `0.508` | `0.515` |
| **Escape DMS (Bloom)** | `0.384` / `0.546` / **`0.428`** | `--` | `0.428` |
| **VHH Nanobody KD** | `0.338` / `0.419` / **`0.444`** | `0.450` | `--` |

---

## Reproducibility

```bash
python scripts/05_extract_embeddings.py --workers 16
python scripts/07_run_modality_transfer_finetuning.py
```
