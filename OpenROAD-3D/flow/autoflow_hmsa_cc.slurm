#!/bin/bash
#SBATCH --job-name=openroad_hmsa_eval
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=0-72%16
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs

# Configuration - adjust these as needed
CASE_NAME="bp_fe"
DESIGN_NAME="bp_fe_top"
DESIGN_DIMENSION="3D"
DEF_VERSION=${CASE_NAME}

IDX="${SLURM_ARRAY_TASK_ID}"

module load apptainer

# workspace is OpenROAD-3D/flow directory
apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:/workspace" \
    --bind "${SCRATCH}:/scratch" \
    openroad.sif \
    bash -lc "
        set -euo pipefail

        FLOW_DIR=/workspace
        CASE_NAME='${CASE_NAME}'
        export DESIGN_NAME='${DESIGN_NAME}'
        export DESIGN_DIMENSION='${DESIGN_DIMENSION}'
        export DEF_VERSION='${DEF_VERSION}'
        IDX='${IDX}'

        HMSA_BASE_DIR=/scratch/\${USER}/HMSA_solution_eval/\${CASE_NAME}_3D
        SOLUTION_DIR=\${HMSA_BASE_DIR}/\${IDX}
        DEF_FILE=\${SOLUTION_DIR}/\${CASE_NAME}_3D.gp.def

        # Make sure required inputs exist early
        if [ ! -f \"\${DEF_FILE}\" ]; then
          echo \"ERROR: DEF file not found: \${DEF_FILE}\"
          ls -la \"\${SOLUTION_DIR}\" || true
          exit 2
        fi

        # OSS CAD Suite env (provides python/openroad tools in your image)
        source /oss-cad-suite/environment

        # Set up isolated working directory in scratch to avoid conflicts
        WORK_DIR=/scratch/\${USER}/hmsa_or_results/\${CASE_NAME}_3D_\${IDX}
        rm -rf \"\${WORK_DIR}\"
        mkdir -p \"\${WORK_DIR}\"
        cd \"\${WORK_DIR}\"

        # -----------------------------
        # Bring in read-only flow assets
        # -----------------------------
        echo \"Creating symlinks for read-only directories...\"
        ln -sfn \"\${FLOW_DIR}/HotSpot\"    \"\${WORK_DIR}/HotSpot\"
        ln -sfn \"\${FLOW_DIR}/platforms\"  \"\${WORK_DIR}/platforms\"
        ln -sfn \"\${FLOW_DIR}/scripts\"    \"\${WORK_DIR}/scripts\"
        ln -sfn \"\${FLOW_DIR}/scripts_3D\"  \"\${WORK_DIR}/scripts_3D\"
        ln -sfn \"\${FLOW_DIR}/util\"       \"\${WORK_DIR}/util\"
        ln -sfn \"\${FLOW_DIR}/tutorials\"  \"\${WORK_DIR}/tutorials\"

        # -----------------------------
        # Designs tree: symlink + copy only mutable design dir
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

        # Create output directories (fresh for each task)
        mkdir -p \"\${WORK_DIR}/logs\" \"\${WORK_DIR}/results\" \"\${WORK_DIR}/reports\" \"\${WORK_DIR}/objects\"

        # Copy DEF file from solution directory into the design directory
        echo \"Copying DEF file for solution index: \${IDX}\"
        cp \"\${DEF_FILE}\" \"\${WORK_DIR}/designs/nangate45_3D/\${DESIGN_NAME}/\"

        # Set up environment variables
        export OPENROAD_EXE=\$(command -v openroad || echo openroad)
        export YOSYS_EXE=\$(command -v yosys || echo yosys)
        export WORK_HOME=\${WORK_DIR}

        # -----------------------------
        # FIX 2: HotSpot scripts path robustness
        # Some flows expect HOTSPOT_SCRIPTS_DIR; ensure it points to the real scripts dir.
        # -----------------------------
        HOTSPOT_DIR=\"\${WORK_DIR}/HotSpot\"

        # Prefer HotSpot/scripts, but auto-detect if missing
        if [ -f \"\${HOTSPOT_DIR}/scripts/divide_def.py\" ]; then
          export HOTSPOT_SCRIPTS_DIR=\"\${HOTSPOT_DIR}\"
        fi
        
        # Create a unique output directory per job to avoid conflicts
        # Since HotSpot is symlinked, all jobs would write to the same output directory
        # Create a per-job output directory in WORK_DIR instead
        HOTSPOT_OUTPUT_DIR=\"\${WORK_DIR}/hotspot_output\"
        mkdir -p \"\${HOTSPOT_OUTPUT_DIR}\"
        export HOTSPOT_OUTPUT=\"\${HOTSPOT_OUTPUT_DIR}\"

        # -----------------------------
        # Clean previous build artifacts to ensure fresh run
        # -----------------------------
        echo \"Cleaning previous build artifacts...\"
        make DESIGN_CONFIG=designs/nangate45_3D/\${DESIGN_NAME}/config_upper_shrink.mk clean_all || true
        make DESIGN_CONFIG=designs/nangate45_3D/\${DESIGN_NAME}/config.mk clean_all || true

        RESULTS_DIR=\${WORK_DIR}/results/nangate45_3D/\${CASE_NAME}/3D
        mkdir -p \"\${RESULTS_DIR}\"

        # Copy 1_synth.sdc from flow folder to work folder
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
        # Pass HOTSPOT_OUTPUT as make variable to override Makefile default
        # This ensures each parallel job writes to its own isolated output directory
        make HOTSPOT_OUTPUT=\"\${HOTSPOT_OUTPUT_DIR}\" DESIGN_CONFIG=designs/nangate45_3D/\${DESIGN_NAME}/config.mk do-hotspot

        # Create directories for storing results in solution folder
        mkdir -p \"\${SOLUTION_DIR}/openroad_logs/\"
        mkdir -p \"\${SOLUTION_DIR}/openroad_results/\"

        # Copy results to solution folder
        echo \"Copying results to solution directory: \${SOLUTION_DIR}\"
        cp -r \"./logs/nangate45_3D/\${CASE_NAME}/3D\"/* \"\${SOLUTION_DIR}/openroad_logs/\" || true
        cp -r \"./results/nangate45_3D/\${CASE_NAME}/3D\"/* \"\${SOLUTION_DIR}/openroad_results/\" || true

        echo \"Completed processing solution index: \${IDX}\"
        echo \"Results saved to: \${SOLUTION_DIR}\"
    "
