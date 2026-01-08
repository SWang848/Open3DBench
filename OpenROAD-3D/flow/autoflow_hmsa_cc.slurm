#!/bin/bash
#SBATCH --job-name=openroad_hmsa_eval
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs

# Configuration - adjust these as needed
CASE_NAME="bp_be"
DESIGN_NAME="bp_be_top"
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
        CASE_NAME=\"${CASE_NAME}\"
        DESIGN_NAME=\"${DESIGN_NAME}\"
        IDX=\"${IDX}\"
        
        HMSA_BASE_DIR=\${FLOW_DIR}/../../HMSA_solution_eval/\${CASE_NAME}_3D
        SOLUTION_DIR=\${HMSA_BASE_DIR}/\${IDX}
        DEF_FILE=\${SOLUTION_DIR}/\${CASE_NAME}_3D.gp.def
        
        source /oss-cad-suite/environment
        # Set up isolated working directory in scratch to avoid conflicts
        WORK_DIR=/scratch/\${USER}/hmsa_or_results/\${CASE_NAME}_3D_\${IDX}
        rm -rf \"\${WORK_DIR}\"
        mkdir -p \"\${WORK_DIR}\"
        
        # Change to work directory
        cd \"\${WORK_DIR}\"
        
        # Symlink read-only directories (safe for parallel execution, saves copy time)
        echo \"Creating symlinks for read-only directories...\"
        ln -sf \"\${FLOW_DIR}/HotSpot\" \"\${WORK_DIR}/HotSpot\" 
        ln -sf \"\${FLOW_DIR}/platforms\" \"\${WORK_DIR}/platforms\"
        ln -sf \"\${FLOW_DIR}/scripts\" \"\${WORK_DIR}/scripts\"
        ln -sf \"\${FLOW_DIR}/scripts_3D\" \"\${WORK_DIR}/scripts_3D\"
        ln -sf \"\${FLOW_DIR}/util\" \"\${WORK_DIR}/util\"
        ln -sf \"\${FLOW_DIR}/tutorials\" \"\${WORK_DIR}/tutorials\"
        
        # Create designs directory structure and symlink read-only parts
        mkdir -p \"\${WORK_DIR}/designs\"
        ln -sf \"\${FLOW_DIR}/designs/nangate45\" \"\${WORK_DIR}/designs/nangate45\"
        # Create nangate45_3D directory and symlink all designs except the one we need to modify
        mkdir -p \"\${WORK_DIR}/designs/nangate45_3D\"
        # Copy only the specific design directory we need to modify
        cp -r \"\${FLOW_DIR}/designs/nangate45_3D/\${DESIGN_NAME}\" \"\${WORK_DIR}/designs/nangate45_3D/\${DESIGN_NAME}\"
        
        # Copy essential files that might be modified
        echo \"Copying essential files...\"
        cp \"\${FLOW_DIR}/Makefile\" \"\${WORK_DIR}/\"
        cp \"\${FLOW_DIR}/Makefile.py\" \"\${WORK_DIR}/\"
        
        # Create output directories (fresh for each task)
        mkdir -p \"\${WORK_DIR}/logs\" \"\${WORK_DIR}/results\" \"\${WORK_DIR}/reports\" \"\${WORK_DIR}/objects\"
        
        # Copy DEF file from solution directory
        echo \"Copying DEF file for solution index: \${IDX}\"
        cp \"\${DEF_FILE}\" \"\${WORK_DIR}/designs/nangate45_3D/\${DESIGN_NAME}/\"
        
        # Set up environment variables (same as autoflow.sh)
        export OPENROAD_EXE=\$(command -v openroad || echo openroad)
        export YOSYS_EXE=\$(command -v yosys || echo yosys)
        
        # Clean previous build artifacts to ensure fresh build for this solution
        echo \"Cleaning previous build artifacts...\"
        make DESIGN_CONFIG=designs/nangate45_3D/\${DESIGN_NAME}/config_upper_shrink.mk clean_all || true
        make DESIGN_CONFIG=designs/nangate45_3D/\${DESIGN_NAME}/config.mk clean_all || true
        
        # Run make commands for this solution
        echo \"Running autoflow for solution index: \${IDX}\"
        make DESIGN_CONFIG=designs/nangate45_3D/\${DESIGN_NAME}/config_upper_shrink.mk do-autoflow
        
        echo \"Running CTS evaluation for solution index: \${IDX}\"
        make DESIGN_CONFIG=designs/nangate45_3D/\${DESIGN_NAME}/config.mk do-cts_eval
        
        echo \"Running hotspot analysis for solution index: \${IDX}\"
        make DESIGN_CONFIG=designs/nangate45_3D/\${DESIGN_NAME}/config.mk do-hotspot
        
        # Generate metrics for this specific solution
        echo \"Generating metrics for solution index: \${IDX}\"
        python3 scripts_3D/get_metrics.py
        
        # Create directories for storing results in solution folder
        mkdir -p \"\${SOLUTION_DIR}/openroad_logs/\"
        mkdir -p \"\${SOLUTION_DIR}/openroad_results/\"
        
        # Copy results to solution folder
        echo \"Copying results to solution directory: \${SOLUTION_DIR}\"
        cp -r \"./logs/nangate45_3D/\${CASE_NAME}/3D\"/* \"\${SOLUTION_DIR}/openroad_logs/\" 
        cp \"./final.csv\" \"\${SOLUTION_DIR}/\"
        cp -r \"./results/nangate45_3D/\${CASE_NAME}/3D\"/* \"\${SOLUTION_DIR}/openroad_results/\" 
        
        echo \"Completed processing solution index: \${IDX}\"
        echo \"Results saved to: \${SOLUTION_DIR}\"
    "

