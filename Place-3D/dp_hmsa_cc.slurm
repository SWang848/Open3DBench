#!/bin/bash
#SBATCH --job-name=hmsa3d_cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-389
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.err

set -euo pipefail
mkdir -p "${SCRATCH}/dopp_logs"

DESIGN_NAME="${DESIGN_NAME}"
CASE_NAME="${CASE_NAME}"
HMSA_RESULTS_DIR="${HMSA_RESULTS_DIR}"
IDX="${SLURM_ARRAY_TASK_ID}"
HMSA_SOLUTION_EVAL_DIR="${HMSA_SOLUTION_EVAL_DIR:-${SCRATCH}/HMSA_solution_eval/${DESIGN_NAME}/${IDX}}"

module load apptainer

# workspace is \Place-3D directory
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
        HMSA_SOLUTION_EVAL_DIR='${HMSA_SOLUTION_EVAL_DIR}'

        mkdir -p \"\${HMSA_SOLUTION_EVAL_DIR}\"

        # per-task isolated working directory on tmp_slurm (fast local I/O)
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
        
        if [ ! -f \"\${HMSA_RESULTS_DIR}/hmsa_results.json\" ]; then
            echo \"ERROR: hmsa_results.json not found at \${HMSA_RESULTS_DIR}/hmsa_results.json\"
            ls -la \"\${HMSA_RESULTS_DIR}\" || true
            exit 1
        fi
        
        # Run with GPU enabled (JSON already has gpu: 1)
        START_TS=\$(date +%s)
        echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Starting placement for \${DESIGN_NAME} idx=\${IDX}\"
        python \"\${REPO}/dreamplace/Placer_3D_hmsa.py\" \
            \"\${REPO}/test/or_3D/\${DESIGN_NAME}.json\" \
            \"\${IDX}\" \
            --hmsa_results_dir \"\${HMSA_RESULTS_DIR}\"
        END_TS=\$(date +%s)
        ELAPSED_SEC=\$((END_TS - START_TS))
        echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Finished placement for \${DESIGN_NAME} idx=\${IDX} in \${ELAPSED_SEC}s\"

        cp -r \"\${WORK}/results/\${DESIGN_NAME}\"/* \"\${HMSA_SOLUTION_EVAL_DIR}/\" || true
        echo \"done. Outputs in: \${HMSA_SOLUTION_EVAL_DIR}\"
    "