#!/bin/bash
#SBATCH --job-name=dopp
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/%u/dopp_logs/%x-%j.out
#SBATCH --error=/scratch/%u/dopp_logs/%x-%j.err

set -euo pipefail
mkdir -p "${SCRATCH}/dopp_logs"

usage() {
  cat <<'EOF'
Usage:
  sbatch DOPP.sh --design <design_name> [options]

Required:
  --design <name>          Design base name (e.g., bp_multi, swerv_wrapper)

Optional:
  --fitness-csv <path>     Fitness CSV path (currently unused in this script)
  --threshold <float>      D-opt threshold for non-zero weight (default: 1e-6)
  --seed <int>             Random seed for HMSA (default: 42)
  -h, --help               Show help

This script runs:
  1) HMSA candidate generation (HierarchyMultiObjectiveSA.py) in Place-3D apptainer
  2) Feature construction (FeatureConstructionByManual.py)
  3) D-opt design (D-opt.py)
  4) Extract selected indices from d_optimal_results.npy
  5) Submit dependent array jobs for:
     - Place-3D/dp_hmsa_cc.slurm
     - OpenROAD-3D/flow/autoflow_hmsa_cc.slurm (after placement succeeds)
EOF
}

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
PLACE_DIR="${REPO_ROOT}/Place-3D"
FLOW_DIR="${REPO_ROOT}/OpenROAD-3D/flow"
WORK="${SCRATCH}/dopp_results/${DESIGN_NAME}/${SEED}"
mkdir -p "${WORK}"

DESIGN_3D="${DESIGN_NAME}_3D"
HMSA_OUT_DIR="${WORK}/hmsa_results/"
REGRESSION_OUT_DIR="${WORK}/regression_results/"
HMSA_SOLUTION_EVAL_DIR="${SCRATCH}/HMSA_solution_eval/${DESIGN_NAME}_3D"

module load python
module load apptainer
mkdir -p "${HMSA_OUT_DIR}" "${REGRESSION_OUT_DIR}" "${HMSA_SOLUTION_EVAL_DIR}"

apptainer exec \
  --bind "${PLACE_DIR}:/workspace" \
  --bind "${SCRATCH}:${SCRATCH}" \
  "${PLACE_DIR}/dreamplace.sif" \
  bash <<EOF
set -euo pipefail
cd "/workspace/install"

echo '[1/5] Running HMSA candidate generation in apptainer...'
python dreamplace/HierarchyMultiObjectiveSA.py \
  "test/or_3D/${DESIGN_3D}.json" \
  --output "${HMSA_OUT_DIR}" \
  --seed "${SEED}"

echo '[2/5] Running feature construction in apptainer...'
python dreamplace/FeatureConstructionByManual.py \
  "test/or_3D/${DESIGN_3D}.json" \
  "${HMSA_OUT_DIR}/hmsa_results.json" \
  --output "${REGRESSION_OUT_DIR}"

echo '[3/5] Running D-opt in apptainer...'
python dreamplace/D-opt.py \
  "${REGRESSION_OUT_DIR}/manual_features.npy" \
  --threshold "${THRESHOLD}" \
  --output "${REGRESSION_OUT_DIR}"
EOF

echo "[4/5] Extracting selected indices..."
pip install numpy
pip install scipy
ARRAY_SPEC="$(
python - <<PY
import numpy as np
from pathlib import Path

dopt = Path("${REGRESSION_OUT_DIR}") / "d_optimal_results.npy"
if not dopt.exists():
    raise FileNotFoundError(f"D-opt result not found: {dopt}")

data = np.load(dopt, allow_pickle=True).item()
idx = data.get("selected_indices")
if idx is None:
    raise KeyError("selected_indices missing in d_optimal_results.npy")

idx = sorted({int(x) for x in idx})
if not idx:
    raise ValueError("selected_indices is empty")

print(",".join(map(str, idx)))
PY
)"

if [[ -z "${ARRAY_SPEC}" ]]; then
  echo "Error: selected indices are empty" >&2
  exit 1
fi

echo "Selected array indices: ${ARRAY_SPEC}"

echo '[4/5] Preparing and submitting dependent array jobs...'
DP_TEMPLATE="${PLACE_DIR}/dp_hmsa_cc.slurm"
OR_TEMPLATE="${FLOW_DIR}/autoflow_hmsa_cc.slurm"


DP_SUBMIT_MSG="$(
  cd "${PLACE_DIR}" && \
  sbatch \
    --array="${ARRAY_SPEC}" \
    --export=ALL,CASE_NAME="${DESIGN_NAME}",DESIGN_NAME="${DESIGN_3D}",HMSA_RESULTS_DIR="${HMSA_OUT_DIR}", HMSA_SOLUTION_EVAL_DIR="${HMSA_SOLUTION_EVAL_DIR}"\
    "${DP_TEMPLATE}"
)"
DP_JOB_ID="$(echo "${DP_SUBMIT_MSG}" | awk '{print $4}')"
if [[ -z "${DP_JOB_ID}" ]]; then
  echo "Error: failed to parse placement job id from: ${DP_SUBMIT_MSG}" >&2
  exit 1
fi

OR_SUBMIT_MSG="$(
  cd "${FLOW_DIR}" && \
  sbatch \
    --dependency=afterok:"${DP_JOB_ID}" \
    --array="${ARRAY_SPEC}" \
    --export=ALL,CASE_NAME="${DESIGN_NAME}",DESIGN_NAME="${DESIGN_NAME}",HMSA_SOLUTION_EVAL_DIR="${HMSA_SOLUTION_EVAL_DIR}" \
    "${OR_TEMPLATE}"
)"
OR_JOB_ID="$(echo "${OR_SUBMIT_MSG}" | awk '{print $4}')"
if [[ -z "${OR_JOB_ID}" ]]; then
  echo "Error: failed to parse OpenROAD job id from: ${OR_SUBMIT_MSG}" >&2
  exit 1
fi
echo "Submitted placement array job : ${DP_JOB_ID}"
echo "Submitted OpenROAD array job  : ${OR_JOB_ID} (afterok:${DP_JOB_ID})"

# apptainer exec \
#   --bind "${PLACE_DIR}:/workspace" \
#   --bind "${SCRATCH}:/scratch" \
#   "${PLACE_DIR}/dreamplace.sif" \
#   bash -lc "
#     set -euo pipefail
#     cd /workspace/install
#     echo '[5/5] Running weighted regression in apptainer...'
#     python dreamplace/Regression.py \
#       ${REGRESSION_OUT_DIR}/manual_features.npy \
#       ${FITNESS_CSV} \
#       --d-opt-results ${REGRESSION_OUT_DIR}/d_optimal_results.npy \
#       --output ${REGRESSION_OUT_DIR}
#   "


