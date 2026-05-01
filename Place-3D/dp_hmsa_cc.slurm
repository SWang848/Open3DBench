#!/bin/bash
#SBATCH --job-name=hmsa3d_cpu
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-999%50
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.err

set -euo pipefail
mkdir -p "${SCRATCH}/dopp_logs"

DESIGN_NAME="${DESIGN_NAME}"
CASE_NAME="${CASE_NAME}"
HMSA_RESULTS_DIR="${HMSA_RESULTS_DIR}"
TOTAL_SOLUTIONS="${TOTAL_SOLUTIONS:-10126}"
START_IDX="${START_IDX:-0}"
BATCH_SIZE="${BATCH_SIZE:-10}"
IDX=$((START_IDX + SLURM_ARRAY_TASK_ID * BATCH_SIZE))
END_IDX=$((IDX + BATCH_SIZE))
HMSA_SOLUTION_EVAL_DIR="${HMSA_SOLUTION_EVAL_DIR:-${SCRATCH}/HMSA_solution_eval/${DESIGN_NAME}}"

if [ "${IDX}" -ge "${TOTAL_SOLUTIONS}" ]; then
    echo "Skipping array task ${SLURM_ARRAY_TASK_ID}; start=${IDX}, TOTAL_SOLUTIONS=${TOTAL_SOLUTIONS}"
    exit 0
fi

if [ "${END_IDX}" -gt "${TOTAL_SOLUTIONS}" ]; then
    END_IDX="${TOTAL_SOLUTIONS}"
fi

module load apptainer

# workspace is the `Place-3D` directory
apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:/workspace" \
    --bind "${SCRATCH}:${SCRATCH}" \
    --bind "${SLURM_TMPDIR}:/tmp_slurm" \
    dreamplace.sif \
    bash -lc "
        REPO=/workspace/install
        DESIGN_NAME=\"${DESIGN_NAME}\"
        CASE_NAME=\"${CASE_NAME}\"
        IDX=\"${IDX}\"
        END_IDX=\"${END_IDX}\"
        HMSA_SOLUTION_EVAL_DIR='${HMSA_SOLUTION_EVAL_DIR}'

        if [ ! -f \"\${HMSA_RESULTS_DIR}/candidates.json\" ]; then
            echo \"ERROR: candidates.json not found at \${HMSA_RESULTS_DIR}/candidates.json\"
            ls -la \"\${HMSA_RESULTS_DIR}\" || true
            exit 1
        fi

        for ((; IDX<END_IDX; IDX++)); do
            mkdir -p \"\${HMSA_SOLUTION_EVAL_DIR}/\${IDX}\"

            # per-solution isolated working directory on tmp_slurm (fast local I/O)
            WORK=/tmp_slurm/hmsa_dp_results/\${DESIGN_NAME}_\${IDX}
            mkdir -p \"\${WORK}\"
            cd \"\${WORK}\"

            # Create intermediate_result directory in WORK directory
            mkdir -p \"\${WORK}/benchmarks/or_3D/intermediate_result\"

            # Symlink input directories from REPO (read-only, safe for parallel execution)
            ln -sf \"\${REPO}/benchmarks/or_3D/\${CASE_NAME}\" \"\${WORK}/benchmarks/or_3D/\${CASE_NAME}\" || true
            ln -sf \"\${REPO}/benchmarks/nangate45\" \"\${WORK}/benchmarks/nangate45\" || true
            ln -sf \"\${REPO}/benchmarks/nangate45_3D\" \"\${WORK}/benchmarks/nangate45_3D\" || true
            ln -sf \"\${REPO}/thirdparty\" \"\${WORK}/thirdparty\" || true

            START_TS=\$(date +%s)
            echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Starting placement for \${DESIGN_NAME} array_task=${SLURM_ARRAY_TASK_ID} global_idx=\${IDX}\"
            python \"\${REPO}/dreamplace/Placer_3D_hmsa.py\" \
                \"\${REPO}/test/or_3D/\${DESIGN_NAME}.json\" \
                \"\${IDX}\" \
                --hmsa_results_dir \"\${HMSA_RESULTS_DIR}\"
            END_TS=\$(date +%s)
            ELAPSED_SEC=\$((END_TS - START_TS))
            echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Finished placement for \${DESIGN_NAME} global_idx=\${IDX} in \${ELAPSED_SEC}s\"

            for result_path in \"\${WORK}/results/\${DESIGN_NAME}\"/*; do
                [ -e \"\${result_path}\" ] || continue
                result_name=\$(basename \"\${result_path}\")
                [[ \"\${result_name}\" == *.gp.def ]] && continue
                cp -r \"\${result_path}\" \"\${HMSA_SOLUTION_EVAL_DIR}/\${IDX}/\" || true
            done
            echo \"done. Outputs in: \${HMSA_SOLUTION_EVAL_DIR}/\${IDX}\"
        done
    "