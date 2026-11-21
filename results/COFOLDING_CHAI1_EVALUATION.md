# Multi-Pose Co-Folding Benchmark Evaluation (`CHAI1`)

- **Ingested Scaffolds**: `851` WT scaffolds (`10` poses/scaffold = `8,510` poses)
- **All Solved WT Co-Crystals (`n=340`)**: Pose-0 = `0.391`, Expected 1-Pose = `0.392`, Top-`ipTM` = **`0.395`** (`50.29% >= 0.23`, `45.59% >= 0.49`, `27.94% >= 0.80`), Best-of-10 Oracle = **`0.440`** (`53.24% >= 0.23`), $\rho(\text{AbAg-ipTM}, \text{DockQ}) = +0.771$
- **LOACO-Local Solved WT Co-Crystals (`n=47`)**: Pose-0 = `0.407`, Expected 1-Pose = `0.413`, Top-`ipTM` = **`0.421`** (`53.19% >= 0.23`, `53.19% >= 0.49`), Best-of-10 Oracle = **`0.445`**

## 1. Multi-Pose Strategy Comparison (`18`-Cluster Co-Crystal Ablation vs. `22`-Cluster Full `LOACO-Local`)

| Multi-Pose Strategy | `ProteinMPNN` 18-Cl (Sign / Imp / **Ord**) | `ProteinMPNN` 22-Cl (**Ord**) |
| :--- | :---: | :---: |
| **pose0_single_seed** | `0.679` / `0.618` / **`0.626`** | **`0.604`** |
| **expected_single_pose** | `0.674` / `0.616` / **`0.619`** | **`0.600`** |
| **top1_iptm_pose** | `0.707` / `0.618` / **`0.640`** | **`0.611`** |
| **top1_abag_iptm_pose** | `0.669` / `0.626` / **`0.617`** | **`0.593`** |
| **uniform_mean_ensemble** | `0.680` / `0.625` / **`0.624`** | **`0.600`** |
| **iptm_softmax_ensemble** | `0.680` / `0.624` / **`0.623`** | **`0.600`** |
| **iptm_confidence_gated** | `0.514` / `0.570` / **`0.515`** | **`0.527`** |
| **best_dockq_oracle_pose** | `0.689` / `0.608` / **`0.629`** | **`0.601`** |

## 2. All 8 3D Models on `18`-Cluster Controlled Co-Crystal Ablation (`47` WT Scaffolds)

| 3D Baseline Model | Top-1 `ipTM` Pose (Sign / Imp / **Ord**) | 10-Pose `ipTM`-Softmax Ensemble (Sign / Imp / **Ord**) | Best-of-10 `DockQ` Oracle (Sign / Imp / **Ord**) |
| :--- | :---: | :---: | :---: |
| **WT 3D Contact (-min Cβ)** | `0.500` / `0.591` / **`0.500`** | `0.500` / `0.595` / **`0.500`** | `0.500` / `0.579` / **`0.500`** |
| **PRODIGY+SASA (3D ZS)** | `0.632` / `0.617` / **`0.587`** | `0.643` / `0.614` / **`0.592`** | `0.646` / `0.605` / **`0.591`** |
| **ProteinMPNN (3D ZS)** | `0.707` / `0.618` / **`0.640`** | `0.680` / `0.624` / **`0.623`** | `0.689` / `0.608` / **`0.629`** |
| **ESM-IF1 (3D ZS)** | `0.630` / `0.591` / **`0.573`** | `0.627` / `0.589` / **`0.572`** | `0.590` / `0.583` / **`0.552`** |
| **AntiFold (3D Ab-Ag ZS)** | `0.627` / `0.591` / **`0.592`** | `0.629` / `0.590` / **`0.593`** | `0.642` / `0.587` / **`0.606`** |
| **ANTIPASTI (3D CNN)** | `0.663` / `0.622` / **`0.624`** | `0.652` / `0.639` / **`0.614`** | `0.669` / `0.619` / **`0.623`** |
| **GearBind (3D GNN)** | `0.675` / `0.634` / **`0.618`** | `0.665` / `0.631` / **`0.612`** | `0.665` / `0.618` / **`0.613`** |
