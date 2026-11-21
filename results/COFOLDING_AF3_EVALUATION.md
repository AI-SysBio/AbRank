# Multi-Pose Co-Folding Benchmark Evaluation (`AF3`)

- **Ingested Scaffolds**: `851` WT scaffolds (`10` poses/scaffold = `8,510` poses)
- **All Solved WT Co-Crystals (`n=340`)**: Pose-0 = `0.296`, Expected 1-Pose = `0.300`, Top-`ipTM` = **`0.340`** (`43.82% >= 0.23`, `38.24% >= 0.49`, `20.59% >= 0.80`), Best-of-10 Oracle = **`0.415`** (`53.53% >= 0.23`), $\rho(\text{AbAg-ipTM}, \text{DockQ}) = +0.684$
- **LOACO-Local Solved WT Co-Crystals (`n=47`)**: Pose-0 = `0.352`, Expected 1-Pose = `0.342`, Top-`ipTM` = **`0.365`** (`44.68% >= 0.23`, `42.55% >= 0.49`), Best-of-10 Oracle = **`0.453`**

## 1. Multi-Pose Strategy Comparison (`18`-Cluster Co-Crystal Ablation vs. `22`-Cluster Full `LOACO-Local`)

| Multi-Pose Strategy | `ProteinMPNN` 18-Cl (Sign / Imp / **Ord**) | `ProteinMPNN` 22-Cl (**Ord**) |
| :--- | :---: | :---: |
| **pose0_single_seed** | `0.578` / `0.587` / **`0.554`** | **`0.538`** |
| **expected_single_pose** | `0.562` / `0.578` / **`0.548`** | **`0.540`** |
| **top1_iptm_pose** | `0.584` / `0.579` / **`0.557`** | **`0.546`** |
| **top1_abag_iptm_pose** | `0.568` / `0.575` / **`0.545`** | **`0.539`** |
| **uniform_mean_ensemble** | `0.568` / `0.589` / **`0.548`** | **`0.539`** |
| **iptm_softmax_ensemble** | `0.562` / `0.590` / **`0.542`** | **`0.530`** |
| **iptm_confidence_gated** | `0.518` / `0.580` / **`0.517`** | **`0.529`** |
| **best_dockq_oracle_pose** | `0.576` / `0.603` / **`0.557`** | **`0.546`** |

## 2. All 8 3D Models on `18`-Cluster Controlled Co-Crystal Ablation (`47` WT Scaffolds)

| 3D Baseline Model | Top-1 `ipTM` Pose (Sign / Imp / **Ord**) | 10-Pose `ipTM`-Softmax Ensemble (Sign / Imp / **Ord**) | Best-of-10 `DockQ` Oracle (Sign / Imp / **Ord**) |
| :--- | :---: | :---: | :---: |
| **WT 3D Contact (-min Cβ)** | `0.500` / `0.577` / **`0.500`** | `0.500` / `0.583` / **`0.500`** | `0.500` / `0.603` / **`0.500`** |
| **PRODIGY+SASA (3D ZS)** | `0.625` / `0.576` / **`0.575`** | `0.605` / `0.574` / **`0.562`** | `0.614` / `0.612` / **`0.573`** |
| **ProteinMPNN (3D ZS)** | `0.584` / `0.579` / **`0.557`** | `0.562` / `0.590` / **`0.542`** | `0.576` / `0.603` / **`0.557`** |
| **ESM-IF1 (3D ZS)** | `0.570` / `0.560` / **`0.555`** | `0.574` / `0.564` / **`0.556`** | `0.564` / `0.568` / **`0.552`** |
| **AntiFold (3D Ab-Ag ZS)** | `0.583` / `0.582` / **`0.546`** | `0.577` / `0.573` / **`0.543`** | `0.538` / `0.545` / **`0.516`** |
| **ANTIPASTI (3D CNN)** | `0.593` / `0.592` / **`0.569`** | `0.581` / `0.600` / **`0.561`** | `0.592` / `0.625` / **`0.573`** |
| **GearBind (3D GNN)** | `0.634` / `0.590` / **`0.586`** | `0.625` / `0.589` / **`0.583`** | `0.602` / `0.615` / **`0.575`** |
