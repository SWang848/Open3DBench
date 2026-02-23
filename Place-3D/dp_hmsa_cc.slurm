#!/bin/bash
#SBATCH --job-name=hmsa3d_cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-389
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs

DESIGN_NAME="swerv_wrapper_3D"
CASE_NAME="swerv_wrapper"
IDX="${SLURM_ARRAY_TASK_ID}"

module load apptainer

# workspace is \Place-3D directory
apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:/workspace" \
    --bind "${SCRATCH}:/scratch" \
    dreamplace.sif \
    bash -lc "
        REPO=/workspace/install
        DESIGN_NAME=\"${DESIGN_NAME}\"
        CASE_NAME=\"${CASE_NAME}\"
        IDX=\"${IDX}\"

        OUT_BASE=/scratch/\${USER}/HMSA_solution_eval/\${DESIGN_NAME}/\${IDX}
        mkdir -p \"\${OUT_BASE}\"

        # per-task isolated working directory to avoid ./results conflicts
        WORK=/scratch/\${USER}/hmsa_dp_results/\${DESIGN_NAME}_\${IDX}
        rm -rf \"\${WORK}\"
        mkdir -p \"\${WORK}\"
        cd \"\${WORK}\"

        # Create intermediate_result directory in WORK directory
        mkdir -p \"\${WORK}/benchmarks/or_3D/intermediate_result\"
        
        # Symlink input directories from REPO (read-only, safe for parallel execution)
        ln -sf \"\${REPO}/benchmarks/or_3D/\${CASE_NAME}\" \"\${WORK}/benchmarks/or_3D/\${CASE_NAME}\" || true
        ln -sf \"\${REPO}/benchmarks/nangate45\" \"\${WORK}/benchmarks/nangate45\" || true
        ln -sf \"\${REPO}/benchmarks/nangate45_3D\" \"\${WORK}/benchmarks/nangate45_3D\" || true
        ln -sf \"\${REPO}/thirdparty\" \"\${WORK}/thirdparty\" || true
        
        # Copy hmsa_results.json to WORK directory
        mkdir -p \"\${WORK}/hmsa_results/\${DESIGN_NAME}\"
        if [ ! -f \"\${REPO}/hmsa_results/\${DESIGN_NAME}/hmsa_results.json\" ]; then
            echo \"ERROR: hmsa_results.json not found at \${REPO}/hmsa_results/\${DESIGN_NAME}/hmsa_results.json\"
            ls -la \"\${REPO}/hmsa_results/\" || true
            exit 1
        fi
        cp \"\${REPO}/hmsa_results/\${DESIGN_NAME}_2/hmsa_results.json\" \"\${WORK}/hmsa_results/\${DESIGN_NAME}/\"
        
        # Run with GPU enabled (JSON already has gpu: 1)
        START_TS=\$(date +%s)
        echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Starting placement for \${DESIGN_NAME} idx=\${IDX}\"
        python \"\${REPO}/dreamplace/Placer_3D_hmsa.py\" \
            \"\${REPO}/test/or_3D/\${DESIGN_NAME}.json\" \
            \"\${IDX}\"
        END_TS=\$(date +%s)
        ELAPSED_SEC=\$((END_TS - START_TS))
        echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Finished placement for \${DESIGN_NAME} idx=\${IDX} in \${ELAPSED_SEC}s\"

        cp -r \"\${WORK}/results/\${DESIGN_NAME}\"/* \"\${OUT_BASE}/\" || true
        echo \"done. Outputs in: \${OUT_BASE}\"
    "