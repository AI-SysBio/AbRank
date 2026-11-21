# Multi-Pose Co-Folding Benchmark Evaluation (`PROTENIX`)

- **Ingested Scaffolds**: `851` WT scaffolds (`10` poses/scaffold = `8,510` poses)
- **All Solved WT Co-Crystals (`n=340`)**: Pose-0 = `0.358`, Expected 1-Pose = `0.353`, Top-`ipTM` = **`0.380`** (`47.35% >= 0.23`, `43.82% >= 0.49`, `24.71% >= 0.80`), Best-of-10 Oracle = **`0.456`** (`57.06% >= 0.23`), $\rho(\text{AbAg-ipTM}, \text{DockQ}) = +0.731$
- **LOACO-Local Solved WT Co-Crystals (`n=47`)**: Pose-0 = `0.370`, Expected 1-Pose = `0.366`, Top-`ipTM` = **`0.407`** (`51.06% >= 0.23`, `46.81% >= 0.49`), Best-of-10 Oracle = **`0.477`**

## 1. Multi-Pose Strategy Comparison (`18`-Cluster Co-Crystal Ablation vs. `22`-Cluster Full `LOACO-Local`)

| Multi-Pose Strategy | `ProteinMPNN` 18-Cl (Sign / Imp / **Ord**) | `ProteinMPNN` 22-Cl (**Ord**) |
| :--- | :---: | :---: |
| **pose0_single_seed** | `0.588` / `0.568` / **`0.559`** | **`0.550`** |
| **expected_single_pose** | `0.586` / `0.564` / **`0.557`** | **`0.551`** |
| **top1_iptm_pose** | `0.588` / `0.573` / **`0.561`** | **`0.556`** |
| **top1_abag_iptm_pose** | `0.593` / `0.574` / **`0.563`** | **`0.557`** |
| **uniform_mean_ensemble** | `0.620` / `0.574` / **`0.576`** | **`0.567`** |
| **iptm_softmax_ensemble** | `0.615` / `0.575` / **`0.573`** | **`0.564`** |
| **iptm_confidence_gated** | `0.547` / `0.577` / **`0.536`** | **`0.547`** |
| **best_dockq_oracle_pose** | `0.617` / `0.572` / **`0.577`** | **`0.571`** |

## 2. All 8 3D Models on `18`-Cluster Controlled Co-Crystal Ablation (`47` WT Scaffolds)

| 3D Baseline Model | Top-1 `ipTM` Pose (Sign / Imp / **Ord**) | 10-Pose `ipTM`-Softmax Ensemble (Sign / Imp / **Ord**) | Best-of-10 `DockQ` Oracle (Sign / Imp / **Ord**) |
| :--- | :---: | :---: | :---: |
| **WT 3D Contact (-min Cβ)** | `0.500` / `0.579` / **`0.500`** | `0.500` / `0.589` / **`0.500`** | `0.500` / `0.584` / **`0.500`** |
| **PRODIGY+SASA (3D ZS)** | `0.600` / `0.575` / **`0.563`** | `0.616` / `0.583` / **`0.572`** | `0.619` / `0.598` / **`0.575`** |
| **ProteinMPNN (3D ZS)** | `0.588` / `0.573` / **`0.561`** | `0.615` / `0.575` / **`0.573`** | `0.617` / `0.572` / **`0.577`** |
| **ESM-IF1 (3D ZS)** | `0.564` / `0.568` / **`0.547`** | `0.569` / `0.564` / **`0.550`** | `0.591` / `0.571` / **`0.564`** |
| **AntiFold (3D Ab-Ag ZS)** | `0.566` / `0.591` / **`0.542`** | `0.566` / `0.590` / **`0.542`** | `0.561` / `0.569` / **`0.538`** |
| **ANTIPASTI (3D CNN)** | `0.595` / `0.619` / **`0.575`** | `0.616` / `0.620` / **`0.590`** | `0.638` / `0.617` / **`0.601`** |
| **GearBind (3D GNN)** | `0.639` / `0.595` / **`0.592`** | `0.651` / `0.599` / **`0.602`** | `0.650` / `0.615` / **`0.605`** |
