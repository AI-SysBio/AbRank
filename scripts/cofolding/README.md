# AbRank Multi-Method, Multi-Pose WT Complex Co-Folding Guide

This directory contains the turnkey pipeline for generating **multi-pose all-atom complex structures (`K = 10` poses per complex per method)** across **all `851` Wild-Type (`WT`) parental antibody–antigen and nanobody–antigen scaffolds** in AbRank.

## Key Experimental Design Rules

1. **WT Parental Complexes Only (`851` Scaffolds Total, Not Per-Mutant Folds)**:
   All 3D local mutation and candidate baselines in AbRank condition on the **wild-type (`WT`) parental scaffold** (`seq_H`, optional `seq_L` for paired IgGs/Fabs or empty for single-domain VHHs, and `seq_A`).
2. **Fold *Every* Scaffold—Including Experimentally Solved Co-Crystals (`340` Solved + `511` Unresolved = `851`)**:
   Do **not** skip complexes that already have an experimental PDB co-crystal! Generating poses on the `340` experimentally resolved WT complexes (with **PDB templates disabled**, `templates=[]`) allows us to:
   - Measure structural pose accuracy (`DockQ`, `fnat`, `iRMSD`, `LRMSD`) directly against native co-crystals,
   - Calibrate `ipTM` / `ranking_score` confidence gates, and
   - Run a 100% consistent co-folding pipeline across solved and unresolved targets alike.
3. **Multiple Methods & Multiple Poses (`10` Poses per Method)**:
   Run at least two complementary co-folding architectures (recommended: **`af3` [AlphaFold 3]** and **`boltz2` [Boltz-2]**, plus **`chai1` [Chai-1]** and/or **`protenix`** if GPU quota permits) with **`--num-poses 10`** (`2 seeds × 5 diffusion samples = 10 poses` per complex).

---

## Priority Tiers in `metadata/cofolding_manifest.tsv`

Every scaffold in [`metadata/cofolding_manifest.tsv`](../../metadata/cofolding_manifest.tsv) is tagged with a `priority_tier`:
- **`1_loaco_local` (`94` WT scaffolds: `47` solved co-crystals + `47` unresolved)**: Covers **100% of the primary `12,267`-pair `LOACO-Local` benchmark and the `18`-cluster Controlled Structure Ablation** (`53` core + `41` expanded `LOACO-Local` scaffolds). Ideal to run first and verify outputs!
- **`2_candidate_and_vhh` (`416` WT scaffolds: `90` solved + `326` unresolved; `283` VHH)**: Covers all `Candidate` (`d_Lev > 3`) and single-domain `VHH` nanobody $K_d$ scaffolds.
- **`3_cross_modal_and_global` (`341` WT scaffolds: `203` solved + `138` unresolved)**: Covers `AbAgym` (45 DMS complexes), `AbDesign` ELISA/$\Delta\Delta G$, `NbBench`, and cross-antigen `Global` scaffolds (`94 + 416 + 341 = 851` total).

---

## Option 0 (Fastest — Skip GPU Co-Folding): Download All 34,040 Precomputed Poses from Kaggle

All **`34,040` template-free poses** (`851` WT scaffolds × `4` methods × `10` poses), **`34,040` confidence files**, and **`3,760` standardized `LOACO-Local` PDBs** are hosted in the companion Kaggle dataset [**`aurlienplissier/abrank-cofolded-3d`**](https://www.kaggle.com/datasets/aurlienplissier/abrank-cofolded-3d) (`3.1 GB` across 8 files; you can also pull individual methods):

```bash
pip install -e ".[kaggle]"

# Download open-weight Boltz-2 + Chai-1 poses and 3,760 standardized LOACO PDBs:
python scripts/cofolding/00_download_cofolded_from_kaggle.py --methods boltz2 chai1 --include-standardized

# Or download all 4 methods (AF3, Boltz-2, Chai-1, Protenix — 34,040 poses total):
python scripts/cofolding/00_download_cofolded_from_kaggle.py --methods all

# Immediately run DockQ + multi-pose LOACO evaluation (Step 4 below):
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src python3 scripts/cofolding/03_ingest_and_evaluate_cofolded.py --method all --max-poses 10
```

*(Note: `af3_10poses.tgz.bin` contains AlphaFold 3 predicted outputs governed by Google DeepMind's non-commercial academic Output Terms; `boltz2`, `chai1`, and `protenix` are generated from open-weight MIT/Apache-2.0 models.)*

---

## Step-by-Step Instructions for GPU Cluster Execution

### Step 1: Clone Repository & Generate All Co-Folding Inputs (CPU, ~3 seconds)

`metadata/cofolding_manifest.tsv` is already tracked in Git, so you do **not** need any large `.pt` caches on the GPU cluster to generate the input files:

```bash
git clone https://github.com/AI-SysBio/AbRank.git
cd AbRank

# Generate 10-pose template-free inputs for all 851 WT scaffolds across AF3, Boltz-2, Chai-1 & Protenix:
PYTHONPATH=src python3 scripts/cofolding/01_generate_cofolding_inputs.py \
  --methods af3 boltz2 chai1 protenix \
  --num-poses 10 \
  --tier all
```

*(Note: By default, `templates: []` is set on every AF3 protein chain and `--use_templates_server=False` on Chai-1 / Boltz-2 so solved PDB co-crystals cannot leak via template lookup. If you also want an explicit template-enabled ablation later, pass `--allow-templates --out-dir data/structures/cofolding_inputs_with_templates`.)*

---

### Step 2: Install Co-Folding Method Environments

Use separate conda/venv/containers if dependencies conflict:

- **Boltz-2 (`boltz2`, MIT open-weight)**:
  ```bash
  pip install "boltz[cuda]" -U
  ```
- **Chai-1 (`chai1`, Apache-2.0 open-weight)**:
  ```bash
  pip install chai_lab==0.6.1
  ```
- **Protenix (`protenix`, Apache-2.0 open-source AF3 reproduction)**:
  ```bash
  pip install protenix
  ```
- **AlphaFold 3 (`af3`, Google DeepMind non-commercial academic weights)**:
  Follow official Docker/Singularity setup from `https://github.com/google-deepmind/alphafold3` and set `AF3_MODEL_DIR` and `AF3_DB_DIR`.

---

### Step 3: Run Co-Folding (Single GPU or SLURM GPU Array)

`scripts/cofolding/02_run_cofolding_worker.sh` supports automatic sharding and checkpoint resume (`_COMPLETED_10POSES` flags, so preempted jobs safely resume where they left off).

#### Option A — Quick Priority-Tier-1 Run (`53` `LOACO-Local` Scaffolds First)
```bash
bash scripts/cofolding/02_run_cofolding_worker.sh --method boltz2 --num-poses 10 --tier 1
bash scripts/cofolding/02_run_cofolding_worker.sh --method chai1  --num-poses 10 --tier 1
AF3_MODEL_DIR=/path/to/af3_weights AF3_DB_DIR=/path/to/af3_dbs \
  bash scripts/cofolding/02_run_cofolding_worker.sh --method af3 --num-poses 10 --tier 1
```

#### Option B — Full `851`-Scaffold SLURM 8-GPU Array (`~1–2 hours` wall-clock on 8× A100/H100)
```bash
# Boltz-2 (10 poses x 851 scaffolds)
sbatch --job-name=abrank_boltz2 --array=0-7 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  scripts/cofolding/02_run_cofolding_worker.sh --method boltz2 --num-poses 10 --tier all --num-shards 8

# Chai-1 (10 poses x 851 scaffolds)
sbatch --job-name=abrank_chai1 --array=0-7 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  scripts/cofolding/02_run_cofolding_worker.sh --method chai1 --num-poses 10 --tier all --num-shards 8

# AlphaFold 3 (2 seeds x 5 samples = 10 poses x 851 scaffolds, templates=[])
sbatch --job-name=abrank_af3 --array=0-7 --gres=gpu:1 --cpus-per-task=12 --mem=80G --time=18:00:00 \
  --export=ALL,AF3_MODEL_DIR=/path/to/af3_weights,AF3_DB_DIR=/path/to/af3_dbs \
  scripts/cofolding/02_run_cofolding_worker.sh --method af3 --num-poses 10 --tier all --num-shards 8
```

---

### Step 4: Package Predicted Structures & Run `DockQ` + Multi-Pose `LOACO` Evaluation

All predicted structures and confidence JSON/NPZ files are written to `data/structures/cofolded/<method>/<scaffold_id>/`. To transfer them back or evaluate directly:

```bash
# Optional: archive predicted poses + confidence files to transfer back
tar -czf abrank_cofolded_poses.tar.gz data/structures/cofolded/

# Ingest all K poses across all 4 methods, compute DockQ vs. the 340 native WT co-crystals, and run 8 multi-pose LOACO strategies:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src python3 scripts/cofolding/03_ingest_and_evaluate_cofolded.py --method all --max-poses 10
```

This generates `results/COFOLDING_<METHOD>_EVALUATION.md`, `results/cofolding_<method>_evaluation.json`, `results/COFOLDING_BENCHMARK_EVALUATION.md`, and `results/cofolding_all_methods_evaluation.json` comparing:
1. `pose0_single_seed` (`pose_00`; note that `Boltz-2` pre-sorts `model_0` by confidence score whereas `AF3`, `Chai-1`, and `Protenix` emit `seed-101_sample-0` first),
2. `expected_single_pose` (unbiased expectation across individual poses `k = 0..9`),
3. `top1_iptm_pose` (best global `ipTM` out of 10),
4. `top1_abag_iptm_pose` (best cross-interface `Ab–Ag Pair ipTM` out of 10, removing `VH–VL` dimer inflation on 3-chain Fabs),
5. `uniform_mean_ensemble` (10-pose uniform mean),
6. `iptm_softmax_ensemble` (`Ab–Ag ipTM`-softmax weighted 10-pose ensemble),
7. `iptm_confidence_gated` (3D ensemble when `max(Ab–Ag ipTM) >= 0.75`, sequence fallback otherwise), and
8. `best_dockq_oracle_pose` (retrospective best-`DockQ` pose out of 10 against the native co-crystal).

---

## Summary of Completed `34,040`-Pose Co-Folding Benchmark Results (`851` WT Scaffolds × `4` Methods × `10` Poses)

### 1. Structural Pose Accuracy (`DockQ` vs. All `340` Native WT Co-Crystals & `47` `LOACO-Local` Co-Crystals)

| Co-Folding Method (`templates=[]`) | All 340 Pose-0 `DockQ` | All 340 Expected 1-Pose | All 340 Top-`ipTM` `DockQ` (`%>=0.23` / `0.49` / `0.80`) | All 340 Best-of-10 `DockQ` (`%>=0.23`) | $\rho(\text{AbAg-ipTM},\text{DockQ})$ | `LOACO-Local` (`n=47`) Top-`ipTM` / Best-of-10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AlphaFold 3 (`af3`)** | `0.296` | `0.300` | **`0.340`** (`43.8%` / `38.2%` / `20.6%`) | **`0.415`** (`53.5%`) | `+0.684` | **`0.365`** (`44.7% >= 0.23`) / **`0.453`** |
| **Protenix (`protenix`)** | `0.358` | `0.353` | **`0.380`** (`47.4%` / `43.8%` / `24.7%`) | **`0.456`** (`57.1%`) | `+0.731` | **`0.407`** (`51.1% >= 0.23`) / **`0.477`** |
| **Chai-1 (`chai1`)** | `0.391` | `0.392` | **`0.395`** (`50.3%` / `45.6%` / `27.9%`) | **`0.440`** (`53.2%`) | **`+0.771`** | **`0.421`** (`53.2% >= 0.23`) / **`0.445`** |
| **Boltz-2 (`boltz2`)** | **`0.561`** | **`0.516`** | **`0.550`** (`65.9%` / `63.2%` / `43.2%`) | **`0.598`** (`69.7%`) | `+0.731` | **`0.628`** (`76.6% >= 0.23`) / **`0.656`** |
| **40-Pose Multi-Method Pool** | — | — | **`0.577`** (`68.8%` / `65.9%` / `45.9%`) | **`0.677`** (`77.9%`) | — | **`0.631`** (`76.6% >= 0.23`) / **`0.686`** |

### 2. Recovery of 3D Affinity Ranking on the `18`-Cluster Controlled Co-Crystal Ablation (`47` WT Scaffolds)

| 3D Baseline Model | Native Bound Co-Crystal (Sign / Imp / **Ord**) | Unbound `IgFold` + Rigid Orient (Sign / Imp / **Ord**) | `AF3` Top-1 `ipTM` (**Ord**) | `Protenix` 10-Pose Ens (**Ord**) | `Boltz-2` Top-1 `ipTM` (Sign / Imp / **Ord**) | `Chai-1` Top-1 `ipTM` (Sign / Imp / **Ord**) | `40-Pose Consensus` (Sign / Imp / **Ord**) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WT 3D Contact ($-\min d_{C\beta}$)** | `0.500` / `0.627` / **`0.500`** | `0.500` / `0.524` / **`0.500`** | `0.577` (Imp) | `0.589` (Imp) | `0.500` / **`0.620`** / `0.500` | `0.500` / `0.591` / `0.500` | `0.500` / `0.615` / `0.500` |
| **PRODIGY+SASA (3D ZS)** | `0.630` / `0.615` / **`0.581`** | `0.566` / `0.554` / **`0.531`** | **`0.575`** | **`0.572`** | `0.613` / `0.628` / **`0.570`** | `0.632` / `0.617` / **`0.587`** | `0.626` / `0.626` / **`0.578`** |
| **ProteinMPNN (3D ZS)** | `0.701` / `0.625` / **`0.621`** | `0.515` / `0.562` / **`0.517`** | **`0.557`** | **`0.573`** | `0.661` / `0.608` / **`0.595`** | **`0.707`** / `0.618` / **`0.640`** | `0.678` / `0.623` / **`0.610`** |
| **ESM-IF1 (3D ZS)** | `0.641` / `0.579` / **`0.575`** | `0.483` / `0.543` / **`0.491`** | **`0.555`** | **`0.550`** | `0.612` / `0.600` / **`0.576`** | `0.630` / `0.591` / **`0.573`** | **`0.634`** / **`0.608`** / **`0.591`** |
| **AntiFold (3D Ab-Ag ZS)** | `0.613` / `0.578` / **`0.570`** | `0.611` / `0.578` / **`0.568`** | **`0.546`** | **`0.542`** | `0.520` / `0.593` / **`0.504`** | **`0.627`** / `0.591` / **`0.592`** | `0.554` / `0.599` / **`0.523`** |
| **ANTIPASTI (3D CNN)** | `0.650` / `0.608` / **`0.602`** | `0.548` / `0.533` / **`0.525`** | **`0.569`** | **`0.590`** | `0.650` / `0.635` / **`0.611`** | **`0.663`** / `0.622` / **`0.624`** | `0.650` / **`0.642`** / **`0.611`** |
| **GearBind (3D GNN)** | `0.643` / `0.594` / **`0.590`** | `0.566` / `0.567` / **`0.543`** | **`0.586`** | **`0.602`** | `0.642` / `0.626` / **`0.593`** | **`0.675`** / **`0.634`** / **`0.618`** | `0.656` / `0.627` / **`0.603`** |
