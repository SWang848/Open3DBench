#!/bin/bash
#SBATCH --job-name=dopp
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch DOPP.sh --design <design_name> [options]

Required:
  --design <name>          Design base name (e.g., bp_multi, swerv_wrapper)

Optional:
  --fitness-csv <path>     Fitness CSV path (default: auto-detect in HMSA_solution_eval)
  --threshold <float>      D-opt threshold for non-zero weight (default: 1e-6)
  -h, --help               Show help

This script runs:
  1) HMSA (HierarchyMultiObjectiveSA.py) in Place-3D apptainer
  2) Feature construction (FeatureConstructionByManual.py)
  3) D-opt design (D-opt.py)
  4) Extract selected indices from d_optimal_results.npy
  5) Submit array jobs for:
     - Place-3D/dp_hmsa_cc.slurm
     - OpenROAD-3D/flow/autoflow_hmsa_cc.slurm (after placement succeeds)
EOF
}

mkdir -p logs

DESIGN_NAME=""
FITNESS_CSV=""
THRESHOLD="1e-6"
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --design)
      DESIGN_NAME="$2"
      shift 2
      ;;
    --fitness-csv)
      FITNESS_CSV="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${DESIGN_NAME}" ]]; then
  echo "Error: --design is required." >&2
  usage
  exit 1
fi


REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
WORK="${SCRATCH}/${USER}/dopp_results/${DESIGN_NAME}/${SEED}"
mkdir -p "${WORK}"

DESIGN_3D="${DESIGN_NAME}_3D"
HMSA_OUT_DIR="${WORK}/hmsa_results/"
REGRESSION_OUT_DIR="${WORK}/regression_results/"

module load apptainer
mkdir -p "${HMSA_OUT_DIR}" "${REGRESSION_OUT_DIR}"

echo "[1/5] Running HMSA pipeline in apptainer..."
apptainer exec \
  --bind "${REPO_ROOT}/Place-3D:/workspace" \
  --bind "${SCRATCH}:/scratch" \
  "${REPO_ROOT}/Place-3D/dreamplace.sif" \
  bash -lc "
    set -euo pipefail
    cd /workspace/install
    python dreamplace/HierarchyMultiObjectiveSA.py \
      test/or_3D/${DESIGN_3D}.json \
      --output ${HMSA_OUT_DIR} \
      --seed ${SEED}

    echo "[2/5] Running feature construction in apptainer..."
      
    python dreamplace/FeatureConstructionByManual.py \
      test/or_3D/${DESIGN_3D}.json \
      ${HMSA_OUT_DIR}/hmsa_results.json \
      --output ${REGRESSION_OUT_DIR}

    echo "[3/5] Running D-opt in apptainer..."
    python dreamplace/D-opt.py \
      ${REGRESSION_OUT_DIR}/manual_features.npy \
      --threshold ${THRESHOLD} \
      --output ${REGRESSION_OUT_DIR} 
  "



apptainer exec \
  --bind "${PLACE_DIR}:/workspace" \
  --bind "${REPO_ROOT}/HMSA_solution_eval:/workspace_eval" \
  --bind "${SCRATCH}:/scratch" \
  "${PLACE_DIR}/dreamplace.sif" \
  bash -lc "
    set -euo pipefail
    cd /workspace/install
    python dreamplace/D-opt.py \
      dreamplace/regression_results/${DESIGN_3D}/manual_features.npy \
      --fitness-csv /workspace_eval/$(basename "${FITNESS_CSV}") \
      --method ${DOPT_METHOD} \
      --threshold ${THRESHOLD} \
      --output dreamplace/regression_results/${DESIGN_3D}/d_optimal_results.npy \
      ${TOP_K:+--top-k ${TOP_K}}
  "

echo "[4/5] Extracting selected indices..."
python - <<PY
import numpy as np
from pathlib import Path

dopt = Path("${DOPT_FILE}")
if not dopt.exists():
    raise FileNotFoundError(f"D-opt result not found: {dopt}")

data = np.load(dopt, allow_pickle=True).item()
idx = data.get("selected_indices", None)
if idx is None:
    raise KeyError("selected_indices missing in d_optimal_results.npy")

idx = sorted({int(x) for x in idx})
if not idx:
    raise ValueError("selected_indices is empty")

out = Path("${SELECTED_TXT}")
out.write_text(",".join(map(str, idx)) + "\n", encoding="utf-8")
print(f"selected_count={len(idx)}")
print(f"array_spec={','.join(map(str, idx))}")
PY

ARRAY_SPEC="$(tr -d '\n' < "${SELECTED_TXT}")"
if [[ -z "${ARRAY_SPEC}" ]]; then
  echo "Error: failed to build array spec from ${SELECTED_TXT}" >&2
  exit 1
fi
echo "Selected array indices: ${ARRAY_SPEC}"

echo "[5/5] Preparing and submitting dependent array jobs..."
DP_TEMPLATE="${PLACE_DIR}/dp_hmsa_cc.slurm"
OR_TEMPLATE="${FLOW_DIR}/autoflow_hmsa_cc.slurm"
DP_JOB="${TMP_DIR}/dp_hmsa_cc.${DESIGN_NAME}.slurm"
OR_JOB="${TMP_DIR}/autoflow_hmsa_cc.${DESIGN_NAME}.slurm"

if [[ ! -f "${DP_TEMPLATE}" || ! -f "${OR_TEMPLATE}" ]]; then
  echo "Error: template slurm scripts not found." >&2
  exit 1
fi

sed \
  -e "s|^DESIGN_NAME=.*|DESIGN_NAME=\"${DESIGN_3D}\"|g" \
  -e "s|^CASE_NAME=.*|CASE_NAME=\"${DESIGN_NAME}\"|g" \
  "${DP_TEMPLATE}" > "${DP_JOB}"

sed \
  -e "s|^CASE_NAME=.*|CASE_NAME=\"${DESIGN_NAME}\"|g" \
  -e "s|^DESIGN_NAME=.*|DESIGN_NAME=\"${DESIGN_NAME}\"|g" \
  "${OR_TEMPLATE}" > "${OR_JOB}"

chmod +x "${DP_JOB}" "${OR_JOB}"

DP_SUBMIT_MSG="$(
  cd "${PLACE_DIR}" && \
  sbatch --array="${ARRAY_SPEC}" "${DP_JOB}"
)"
DP_JOB_ID="$(echo "${DP_SUBMIT_MSG}" | awk '{print $4}')"
if [[ -z "${DP_JOB_ID}" ]]; then
  echo "Error: failed to parse placement job id from: ${DP_SUBMIT_MSG}" >&2
  exit 1
fi

OR_SUBMIT_MSG="$(
  cd "${FLOW_DIR}" && \
  sbatch --dependency=afterok:"${DP_JOB_ID}" --array="${ARRAY_SPEC}" "${OR_JOB}"
)"
OR_JOB_ID="$(echo "${OR_SUBMIT_MSG}" | awk '{print $4}')"
if [[ -z "${OR_JOB_ID}" ]]; then
  echo "Error: failed to parse OpenROAD job id from: ${OR_SUBMIT_MSG}" >&2
  exit 1
fi

echo
echo "Submitted placement array job : ${DP_JOB_ID}"
echo "Submitted OpenROAD array job  : ${OR_JOB_ID} (afterok:${DP_JOB_ID})"
echo "Generated scripts:"
echo "  ${DP_JOB}"
echo "  ${OR_JOB}"
echo "Selected indices file:"
echo "  ${SELECTED_TXT}"
