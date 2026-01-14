#!/bin/bash
#SBATCH --job-name=openroad_hmsa_eval
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-57
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs

# Configuration - adjust these as needed
CASE_NAME="bp_quad"
DESIGN_NAME="bp_quad"
DESIGN_DIMENSION="3D"
DEF_VERSION=${CASE_NAME}

IDX="${SLURM_ARRAY_TASK_ID}"

module load apptainer

# workspace is OpenROAD-3D/flow directory
apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:/workspace" \
    --bind "${SCRATCH}:/scratch" \
    --bind "${SLURM_TMPDIR}:/tmp_slurm" \
    openroad.sif \
    bash -lc "
        set -euo pipefail

        FLOW_DIR=/workspace
        CASE_NAME='${CASE_NAME}'
        export DESIGN_NAME='${DESIGN_NAME}'
        export DESIGN_DIMENSION='${DESIGN_DIMENSION}'
        export DEF_VERSION='${DEF_VERSION}'
        IDX='${IDX}'
        # SLURM array job variables (for log file naming)
        # %A in output format = SLURM_ARRAY_JOB_ID (master job ID)
        # %a in output format = SLURM_ARRAY_TASK_ID (task index)
        SLURM_ARRAY_JOB_ID='${SLURM_ARRAY_JOB_ID}'
        SLURM_ARRAY_TASK_ID='${SLURM_ARRAY_TASK_ID}'

        HMSA_BASE_DIR=/scratch/\${USER}/HMSA_solution_eval/\${CASE_NAME}_3D
        SOLUTION_DIR=\${HMSA_BASE_DIR}/\${IDX}
        DEF_FILE=\${SOLUTION_DIR}/\${CASE_NAME}_3D.gp.def
        
        # Function to copy logs on exit (success or failure)
        # Defined early so trap can catch errors from subsequent operations
        copy_logs_on_exit() {
            local exit_code=\$?
            echo \"Script exiting with code: \${exit_code}\"
            
            cp \"\${FLOW_DIR}/logs/openroad_hmsa_eval-\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.out\" \"\${SOLUTION_DIR}/\"
            cp \"\${FLOW_DIR}/logs/openroad_hmsa_eval-\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.err\" \"\${SOLUTION_DIR}/\"
            
            exit \${exit_code}
        }
        
        # Set trap to copy logs on exit (EXIT fires on both success and failure)
        # Must be set AFTER function definition but BEFORE any operations that might fail
        trap copy_logs_on_exit EXIT

        # Set up OSS CAD Suite environment (provides python/openroad tools)
        source /oss-cad-suite/environment

        # Set up isolated working directory in tmp_slurm for faster I/O
        WORK_DIR=/tmp_slurm/hmsa_or_results/\${CASE_NAME}_3D_\${IDX}
        mkdir -p \"\${WORK_DIR}\"
        cd \"\${WORK_DIR}\"

        # -----------------------------
        # Bring in read-only flow assets
        # -----------------------------
        echo \"Creating symlinks for read-only directories...\"
        ln -sfn \"\${FLOW_DIR}/platforms\"  \"\${WORK_DIR}/platforms\"
        ln -sfn \"\${FLOW_DIR}/scripts\"    \"\${WORK_DIR}/scripts\"
        ln -sfn \"\${FLOW_DIR}/scripts_3D\"  \"\${WORK_DIR}/scripts_3D\"
        ln -sfn \"\${FLOW_DIR}/util\"       \"\${WORK_DIR}/util\"
        ln -sfn \"\${FLOW_DIR}/tutorials\"  \"\${WORK_DIR}/tutorials\"
        
        # Copy HotSpot instead of symlinking to avoid write conflicts
        # HotSpot writes to scripts/output/ and examples/*/outputs/ during execution
        # Each parallel job needs its own isolated copy (no rebuild needed - binaries are pre-built)
        # HotSpot is copied to WORK_DIR/HotSpot
        echo \"Copying HotSpot directory (pre-built, no rebuild needed)...\"
        # Use rsync for more robust copying (handles I/O errors better than cp)
        # Retry up to 3 times if copy fails
        MAX_RETRIES=3
        RETRY_COUNT=0
        while [ \$RETRY_COUNT -lt \$MAX_RETRIES ]; do
          if rsync -a --no-g \"\${FLOW_DIR}/HotSpot/\" \"\${WORK_DIR}/HotSpot/\" 2>/dev/null; then
            break
          fi
          RETRY_COUNT=\$((RETRY_COUNT + 1))
          echo \"HotSpot copy attempt \$RETRY_COUNT failed, retrying...\"
          sleep 2
        done
        if [ \$RETRY_COUNT -eq \$MAX_RETRIES ]; then
          echo \"ERROR: Failed to copy HotSpot after \$MAX_RETRIES attempts\"
          exit 3
        fi

        # -----------------------------
        # Symlink + copy only mutable design dir
        # -----------------------------
        mkdir -p \"\${WORK_DIR}/designs\"
        ln -sfn \"\${FLOW_DIR}/designs/nangate45\" \"\${WORK_DIR}/designs/nangate45\"

        mkdir -p \"\${WORK_DIR}/designs/nangate45_3D\"
        cp -r \"\${FLOW_DIR}/designs/nangate45_3D/\${DESIGN_NAME}\" \
              \"\${WORK_DIR}/designs/nangate45_3D/\${DESIGN_NAME}\"

        # Copy Makefiles locally so we can patch them safely
        echo \"Copying essential files...\"
        cp \"\${FLOW_DIR}/Makefile\" \"\${WORK_DIR}/\"
        cp \"\${FLOW_DIR}/Makefile.py\" \"\${WORK_DIR}/\"

        # -----------------------------
        # FIX: genElapsedTime.py permission denied
        # Patch Makefile to call it via python3 instead of executing it
        # -----------------------------
        if grep -q \"genElapsedTime.py\" \"\${WORK_DIR}/Makefile\"; then
          sed -i 's|\\\$(UTILS_DIR)/genElapsedTime.py|python3 \\\$(UTILS_DIR)/genElapsedTime.py|g' \"\${WORK_DIR}/Makefile\"
        fi

        # Create output directories
        mkdir -p \"\${WORK_DIR}/logs\" \"\${WORK_DIR}/results\" \"\${WORK_DIR}/reports\" \"\${WORK_DIR}/objects\"

        # Copy DEF file from solution directory into the design directory
        echo \"Copying DEF file for solution index: \${IDX}\"
        cp \"\${DEF_FILE}\" \"\${WORK_DIR}/designs/nangate45_3D/\${DESIGN_NAME}/\"

        # Set up environment variables
        export OPENROAD_EXE=\$(command -v openroad || echo openroad)
        export YOSYS_EXE=\$(command -v yosys || echo yosys)
        export WORK_HOME=\${WORK_DIR}

        # -----------------------------
        # Clean previous build artifacts to ensure fresh run
        # -----------------------------
        echo \"Cleaning previous build artifacts...\"
        make DESIGN_CONFIG=designs/nangate45_3D/\${DESIGN_NAME}/config_upper_shrink.mk clean_all || true
        make DESIGN_CONFIG=designs/nangate45_3D/\${DESIGN_NAME}/config.mk clean_all || true

        # Makefile not include synthesis, so copy 1_synth.sdc from flow folder to work folder
        RESULTS_DIR=\${WORK_DIR}/results/nangate45_3D/\${CASE_NAME}/3D
        mkdir -p \"\${RESULTS_DIR}\"
        FLOW_RESULTS_DIR=\${FLOW_DIR}/results/nangate45_3D/\${CASE_NAME}/3D
        cp \"\${FLOW_RESULTS_DIR}/1_synth.sdc\" \"\${RESULTS_DIR}/1_synth.sdc\"
        echo \"Copied 1_synth.sdc from \${FLOW_RESULTS_DIR} to \${RESULTS_DIR}\"

        # -----------------------------
        # Run make commands for this solution
        # -----------------------------
        echo \"Running autoflow for solution index: \${IDX}\"
        make DESIGN_CONFIG=designs/nangate45_3D/\${DESIGN_NAME}/config_upper_shrink.mk do-autoflow

        echo \"Running CTS evaluation for solution index: \${IDX}\"
        make DESIGN_CONFIG=designs/nangate45_3D/\${DESIGN_NAME}/config.mk do-cts_eval

        echo \"Running hotspot analysis for solution index: \${IDX}\"
        # HotSpot is now fully isolated per job (copied, not symlinked)
        # No need to override HOTSPOT_OUTPUT or HOTSPOT_EXAMPLES_DIR - defaults work correctly
        make DESIGN_CONFIG=designs/nangate45_3D/\${DESIGN_NAME}/config.mk do-hotspot

        # Zip and move results (trap will also handle this on failure)
        echo \"Zipping results...\"
        (cd \"./logs/nangate45_3D/\${CASE_NAME}/3D\" && zip -r -0 \"\${WORK_DIR}/openroad_logs.zip\" *) || true
        (cd \"./results/nangate45_3D/\${CASE_NAME}/3D\" && zip -r -0 \"\${WORK_DIR}/openroad_results.zip\" *) || true
        mv \"./openroad_logs.zip\" \"\${SOLUTION_DIR}/\"
        mv \"./openroad_results.zip\" \"\${SOLUTION_DIR}/\"

        echo \"Completed processing solution index: \${IDX}\"
        echo \"Results saved to: \${SOLUTION_DIR}\"
    "
