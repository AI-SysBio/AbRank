# Multi-Pose Co-Folding Benchmark Evaluation (`BOLTZ2`)

- **Ingested Scaffolds**: `851` WT scaffolds (`10` poses/scaffold = `8,510` poses)
- **All Solved WT Co-Crystals (`n=340`)**: Pose-0 = `0.561`, Expected 1-Pose = `0.516`, Top-`ipTM` = **`0.550`** (`65.88% >= 0.23`, `63.24% >= 0.49`, `43.24% >= 0.80`), Best-of-10 Oracle = **`0.598`** (`69.71% >= 0.23`), $\rho(\text{AbAg-ipTM}, \text{DockQ}) = +0.731$
- **LOACO-Local Solved WT Co-Crystals (`n=47`)**: Pose-0 = `0.638`, Expected 1-Pose = `0.606`, Top-`ipTM` = **`0.628`** (`76.6% >= 0.23`, `72.34% >= 0.49`), Best-of-10 Oracle = **`0.656`**

## 1. Multi-Pose Strategy Comparison (`18`-Cluster Co-Crystal Ablation vs. `22`-Cluster Full `LOACO-Local`)

| Multi-Pose Strategy | `ProteinMPNN` 18-Cl (Sign / Imp / **Ord**) | `ProteinMPNN` 22-Cl (**Ord**) |
| :--- | :---: | :---: |
| **pose0_single_seed** | `0.667` / `0.615` / **`0.600`** | **`0.577`** |
| **expected_single_pose** | `0.657` / `0.604` / **`0.594`** | **`0.581`** |
| **top1_iptm_pose** | `0.661` / `0.608` / **`0.595`** | **`0.575`** |
| **top1_abag_iptm_pose** | `0.660` / `0.603` / **`0.595`** | **`0.575`** |
| **uniform_mean_ensemble** | `0.658` / `0.613` / **`0.593`** | **`0.576`** |
| **iptm_softmax_ensemble** | `0.655` / `0.615` / **`0.590`** | **`0.574`** |
| **iptm_confidence_gated** | `0.616` / `0.590` / **`0.569`** | **`0.555`** |
| **best_dockq_oracle_pose** | `0.627` / `0.612` / **`0.571`** | **`0.553`** |

## 2. All 8 3D Models on `18`-Cluster Controlled Co-Crystal Ablation (`47` WT Scaffolds)

| 3D Baseline Model | Top-1 `ipTM` Pose (Sign / Imp / **Ord**) | 10-Pose `ipTM`-Softmax Ensemble (Sign / Imp / **Ord**) | Best-of-10 `DockQ` Oracle (Sign / Imp / **Ord**) |
| :--- | :---: | :---: | :---: |
| **WT 3D Contact (-min Cβ)** | `0.500` / `0.620` / **`0.500`** | `0.500` / `0.617` / **`0.500`** | `0.500` / `0.611` / **`0.500`** |
| **PRODIGY+SASA (3D ZS)** | `0.613` / `0.628` / **`0.570`** | `0.625` / `0.627` / **`0.575`** | `0.618` / `0.624` / **`0.574`** |
| **ProteinMPNN (3D ZS)** | `0.661` / `0.608` / **`0.595`** | `0.655` / `0.615` / **`0.590`** | `0.627` / `0.612` / **`0.571`** |
| **ESM-IF1 (3D ZS)** | `0.612` / `0.600` / **`0.576`** | `0.612` / `0.602` / **`0.576`** | `0.593` / `0.619` / **`0.559`** |
| **AntiFold (3D Ab-Ag ZS)** | `0.520` / `0.593` / **`0.504`** | `0.526` / `0.594` / **`0.508`** | `0.526` / `0.591` / **`0.506`** |
| **ANTIPASTI (3D CNN)** | `0.650` / `0.635` / **`0.611`** | `0.654` / `0.633` / **`0.614`** | `0.635` / `0.633` / **`0.601`** |
| **GearBind (3D GNN)** | `0.642` / `0.626` / **`0.593`** | `0.646` / `0.625` / **`0.594`** | `0.639` / `0.625` / **`0.589`** |
