#!/usr/bin/env bash
# ==============================================================================
# Step C2: Unified Multi-Method, Multi-Pose Co-Folding GPU Runner (Single-Node or SLURM Array)
#
# Runs AlphaFold 3 (`af3`), Boltz-2 (`boltz2`), Chai-1 (`chai1`), or Protenix (`protenix`)
# across all 851 AbRank WT parental complexes (including BOTH the 340 experimentally solved
# co-crystals and the 511 unresolved complexes, with PDB templates disabled by default).
#
# Usage (Single GPU):
#   bash scripts/cofolding/02_run_cofolding_worker.sh --method boltz2 --num-poses 10
#
# Usage (SLURM 8-GPU Array):
#   sbatch --array=0-7 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
#     scripts/cofolding/02_run_cofolding_worker.sh --method af3 --num-poses 10 --num-shards 8
# ==============================================================================
set -euo pipefail

METHOD="boltz2"
NUM_POSES=10
TIER="all"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
AF3_MODEL_DIR="${AF3_MODEL_DIR:-/path/to/af3_weights}"
AF3_DB_DIR="${AF3_DB_DIR:-/path/to/af3_databases}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)      METHOD="$2"; shift 2 ;;
    --num-poses)   NUM_POSES="$2"; shift 2 ;;
    --tier)        TIER="$2"; shift 2 ;;
    --shard-id)    SHARD_ID="$2"; shift 2 ;;
    --num-shards)  NUM_SHARDS="$2"; shift 2 ;;
    --af3-models)  AF3_MODEL_DIR="$2"; shift 2 ;;
    --af3-dbs)     AF3_DB_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT_ROOT="${REPO_ROOT}/data/structures/cofolding_inputs/${METHOD}"
OUT_ROOT="${REPO_ROOT}/data/structures/cofolded/${METHOD}"
mkdir -p "${OUT_ROOT}"

# Ensure standardized inputs exist for the requested method and pose count
if [[ ! -d "${INPUT_ROOT}" ]] || [[ -z "$(ls -A "${INPUT_ROOT}" 2>/dev/null)" ]]; then
  echo "[Init] Generating co-folding inputs for method=${METHOD}, num_poses=${NUM_POSES}, tier=${TIER}..."
  PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/cofolding/01_generate_cofolding_inputs.py" \
    --methods "${METHOD}" --num-poses "${NUM_POSES}" --tier "${TIER}"
fi

mapfile -t ALL_FILES < <(find "${INPUT_ROOT}" -maxdepth 1 -type f | sort)
TOTAL="${#ALL_FILES[@]}"
echo "[Worker] Method=${METHOD} | Poses=${NUM_POSES} | Total=${TOTAL} | Shard=${SHARD_ID}/${NUM_SHARDS}"

N_SEEDS_5X=$(( (NUM_POSES + 4) / 5 ))

for (( idx=0; idx<TOTAL; idx++ )); do
  if (( idx % NUM_SHARDS != SHARD_ID )); then
    continue
  fi

  IN_FILE="${ALL_FILES[$idx]}"
  FNAME="$(basename "${IN_FILE}")"
  SID="${FNAME%.*}"
  TARGET_OUT="${OUT_ROOT}/${SID}"
  DONE_FLAG="${TARGET_OUT}/_COMPLETED_${NUM_POSES}POSES"

  if [[ -f "${DONE_FLAG}" ]]; then
    echo "  [Skip ${idx}/${TOTAL}] ${SID} already completed."
    continue
  fi
  mkdir -p "${TARGET_OUT}"
  echo "  [Run  ${idx}/${TOTAL}] Folding ${SID} with ${METHOD} (${NUM_POSES} poses)..."

  case "${METHOD}" in
    af3)
      # Official Google DeepMind AlphaFold 3 CLI (JSON already encodes modelSeeds & templates=[])
      "${PYTHON_BIN}" run_alphafold.py \
        --json_path="${IN_FILE}" \
        --model_dir="${AF3_MODEL_DIR}" \
        --db_dir="${AF3_DB_DIR}" \
        --output_dir="${TARGET_OUT}"
      ;;

    boltz2)
      # Official Boltz-1 / Boltz-2 CLI (template-free unless YAML includes templates)
      boltz predict "${IN_FILE}" \
        --out_dir "${TARGET_OUT}" \
        --diffusion_samples "${NUM_POSES}" \
        --recycling_steps 10 \
        --sampling_steps 200 \
        --use_msa_server \
        --output_format mmcif
      ;;

    chai1)
      # Official Chai-1 Python API across multiple seeds x 5 diffusion samples = NUM_POSES poses
      "${PYTHON_BIN}" - "${IN_FILE}" "${TARGET_OUT}" "${N_SEEDS_5X}" <<'PYEOF'
import sys
from pathlib import Path
from chai_lab.chai1 import run_inference

fasta_path = Path(sys.argv[1])
out_base = Path(sys.argv[2])
n_seeds = int(sys.argv[3])

for s_idx in range(n_seeds):
    seed_val = 101 + s_idx
    seed_dir = out_base / f"seed_{seed_val}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    run_inference(
        fasta_file=fasta_path,
        output_dir=seed_dir,
        num_trunk_recycles=10,
        num_diffn_timesteps=200,
        num_diffn_samples=5,
        seed=seed_val,
        use_esm_embeddings=True,
        use_msa_server=True,
        use_templates_server=False,
    )
PYEOF
      ;;

    protenix)
      SEED_CSV="$(seq -s, 101 $((100 + N_SEEDS_5X)))"
      protenix predict \
        --input "${IN_FILE}" \
        --out_dir "${TARGET_OUT}" \
        --seeds "${SEED_CSV}" \
        --N_sample 5 \
        --use_msa_server
      ;;

    *)
      echo "Unsupported method: ${METHOD}" >&2
      exit 1
      ;;
  esac

  touch "${DONE_FLAG}"
done

echo "[Done] Shard ${SHARD_ID}/${NUM_SHARDS} finished for method=${METHOD}."
