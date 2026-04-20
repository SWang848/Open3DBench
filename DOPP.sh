#!/bin/bash
#SBATCH --job-name=dopp
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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
HMSA_SOLUTION_EVAL_DIR="${WORK}/HMSA_solution_eval/${DESIGN_NAME}_3D"

module load python
module load apptainer
mkdir -p "${HMSA_OUT_DIR}" "${REGRESSION_OUT_DIR}" "${HMSA_SOLUTION_EVAL_DIR}"

apptainer exec \
  --bind "${REPO_ROOT}:/repo" \
  --bind "${SCRATCH}:${SCRATCH}" \
  "${PLACE_DIR}/dreamplace.sif" \
  bash <<EOF
set -euo pipefail
export PYTHONPATH="/repo:${PYTHONPATH:-}"
cd "/repo"

echo '[1/6] Running HMSA candidate generation in apptainer...'
python -m algorithms.dopp.hierarchy_multi_objective_sa \
  "Place-3D/install/test/or_3D/${DESIGN_3D}.json" \
  --output "${HMSA_OUT_DIR}" \
  --seed "${SEED}"

echo '[2/6] Running feature construction in apptainer...'
python -m algorithms.dopp.feature_construction_manual \
  "Place-3D/install/test/or_3D/${DESIGN_3D}.json" \
  "${HMSA_OUT_DIR}/hmsa_results.json" \
  --output "${REGRESSION_OUT_DIR}"

echo '[3/6] Running D-opt in apptainer...'
python -m algorithms.dopp.d_opt \
  "${REGRESSION_OUT_DIR}/manual_features.npy" \
  --threshold "${THRESHOLD}" \
  --output "${REGRESSION_OUT_DIR}"
EOF

echo "[4/6] Extracting selected indices..."
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
    --array="${ARRAY_SPEC}" --cpus-per-task=4 --mem=8G --time=1:00:00\
    --export=ALL,CASE_NAME="${DESIGN_NAME}",DESIGN_NAME="${DESIGN_3D}",HMSA_RESULTS_DIR="${HMSA_OUT_DIR}",HMSA_SOLUTION_EVAL_DIR="${HMSA_SOLUTION_EVAL_DIR}"\
    "${DP_TEMPLATE}"
)"
DP_JOB_ID="$(echo "${DP_SUBMIT_MSG}" | awk '{print $4}' | cut -d'_' -f1)"
if [[ -z "${DP_JOB_ID}" ]]; then
  echo "Error: failed to parse placement job id from: ${DP_SUBMIT_MSG}" >&2
  exit 1
fi

echo "Submitted placement array job : ${DP_JOB_ID}"

# Submit one OpenROAD "single-element array" per selected index, depending on the matching placement element
# Because there is a bug for using aftercorr: in OpenROAD array jobs
OR_JOB_IDS=()
for i in ${ARRAY_SPEC//,/ }; do
  OR_JOB_ID="$(
    cd "${FLOW_DIR}" && \
    sbatch --parsable \
      --dependency=afterok:${DP_JOB_ID}_${i} \
      --array="${i}" --cpus-per-task=8 --mem=16G --time=2:00:00 \
      --export=ALL,CASE_NAME="${DESIGN_NAME}",DESIGN_NAME="${DESIGN_NAME}",HMSA_SOLUTION_EVAL_DIR="${HMSA_SOLUTION_EVAL_DIR}" \
      "${OR_TEMPLATE}"
  )"
  OR_JOB_ID="${OR_JOB_ID%%;*}"

  if [[ -z "${OR_JOB_ID}" ]]; then
    echo "Error: failed to get OpenROAD job id for index ${i}" >&2
    exit 1
  fi

  OR_JOB_IDS+=("${OR_JOB_ID}")
  echo "Submitted OpenROAD job : ${OR_JOB_ID} (afterok:${DP_JOB_ID}_${i}, array=${i})"
done

echo "All OpenROAD jobs: ${OR_JOB_IDS[*]}"

# Wait until all OpenROAD jobs are finished, or the only remaining ones are
# stuck in PENDING with Reason=DependencyNeverSatisfied.
echo "Waiting for OpenROAD jobs to finish (or become DependencyNeverSatisfied only)..."
while true; do
  any_active=0
  any_non_dns_active=0

  for jobid in "${OR_JOB_IDS[@]}"; do
    # squeue prints nothing if the job is no longer pending/running
    line="$(squeue -j "${jobid}" -h -o "%i %t %r" 2>/dev/null | head -n 1 || true)"
    if [[ -z "${line}" ]]; then
      # Job has finished (COMPLETED/FAILED/CANCELLED/etc.)
      continue
    fi

    any_active=1
    state="$(awk '{print $2}' <<< "${line}")"
    # Reason may contain spaces; join fields 3..NF
    reason="$(awk '{for (i=3; i<=NF; i++) printf (i==3 ? $i : " " $i)}' <<< "${line}")"

    if [[ "${state}" == "PD" && "${reason}" == "DependencyNeverSatisfied" ]]; then
      # This job is pending but can never run; do not count as non-DNS-active
      continue
    fi

    # Any other pending/running state means we should keep waiting
    any_non_dns_active=1
  done

  # Stop when:
  #  - no jobs are active at all (all finished), OR
  #  - the only remaining active jobs are DependencyNeverSatisfied.
  if [[ "${any_active}" -eq 0 || "${any_non_dns_active}" -eq 0 ]]; then
    echo "OpenROAD jobs reached terminal state (all finished or only DependencyNeverSatisfied remain)."
    break
  fi

  echo "Still waiting on OpenROAD jobs (some pending/running not DependencyNeverSatisfied)..."
  sleep 60
done

echo '[5/6] Running HMSA solution evaluation in apptainer...'
cd "${WORK}"

# unzip the openroad_logs.zip
bash << EOF
for dir in {0..389}; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    cd "$dir" || continue
    
    # Process openroad_logs.zip if it exists
    if [ -f "openroad_logs.zip" ]; then
        echo "Extracting $dir/openroad_logs.zip..."
        mkdir -p openroad_logs && unzip -o -q ./openroad_logs.zip -d ./openroad_logs && rm -f ./openroad_logs.zip || echo "Failed: $dir/openroad_logs.zip"
    fi
    
    # Process openroad_results.zip if it exists
    if [ -f "openroad_results.zip" ]; then
        echo "Extracting $dir/openroad_results.zip..."
        mkdir -p openroad_results && unzip -o -q ./openroad_results.zip -d ./openroad_results && rm -f ./openroad_results.zip || echo "Failed: $dir/openroad_results.zip"
    fi

    cd ..
done
EOF
cp ${HMSA_OUT_DIR}/hmsa_results.json ${HMSA_SOLUTION_EVAL_DIR}

python ${REPO_ROOT}/HMSA_solution_eval/get_metrics.py \
  --dataset_name "${DESIGN_3D}" \
  --dir_path "${WORK}/${DESIGN_3D}" \
  --metrics_path "${HMSA_SOLUTION_EVAL_DIR}" \

apptainer exec \
  --bind "${REPO_ROOT}:/repo" \
  --bind "${SCRATCH}:/scratch" \
  "${PLACE_DIR}/dreamplace.sif" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH="/repo:${PYTHONPATH:-}"
    cd /repo
    echo '[6/6] Running weighted regression in apptainer...'
    python -m algorithms.dopp.regression \
      ${REGRESSION_OUT_DIR}/manual_features.npy \
      ${HMSA_SOLUTION_EVAL_DIR}/metrics.csv \
      --d-opt-results ${REGRESSION_OUT_DIR}/d_optimal_results.npy \
      --output ${REGRESSION_OUT_DIR}
  "


