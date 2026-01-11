#!/bin/bash
set -euo pipefail
export DESIGN_DIMENSION=$1
export DEF_VERSION=$2
export DESIGN_NAME=$3
export HMSA_EVAL_FLAG=$4
echo $DESIGN_DIMENSION, $DEF_VERSION, $DESIGN_NAME, $HMSA_EVAL_FLAG
export OPENROAD_EXE=$(command -v openroad)
export YOSYS_EXE=$(command -v yosys)
if [ "$DESIGN_DIMENSION" = "3D" ]
then
    if [ "$HMSA_EVAL_FLAG" = "true" ]
    then
        # Iterate through all solution index subfolders
        for solution_dir in ../../HMSA_solution_eval/${DEF_VERSION}_3D/*/
        do
            # Extract solution index from path
            solution_idx=$(basename "$solution_dir")
            echo "Processing HMSA solution index: $solution_idx"

            # Copy def file from the solution subfolder
            cp ${solution_dir}${DEF_VERSION}_3D.gp.def designs/nangate45_3D/${DESIGN_NAME}/
            
            # Clean previous build artifacts to ensure fresh build for this solution
            make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config_upper_shrink.mk clean_all
            make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config.mk clean_all
            
            # Run make commands for this solution
            make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config_upper_shrink.mk do-autoflow 
            make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config.mk do-cts_eval 
            make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config.mk do-hotspot
            
            # Generate metrics for this specific solution
            python3 scripts_3D/get_metrics.py
            
            # Create directories for storing results
            mkdir -p ../../HMSA_solution_eval/${DEF_VERSION}_3D/${solution_idx}/openroad_logs/
            mkdir -p ../../HMSA_solution_eval/${DEF_VERSION}_3D/${solution_idx}/openroad_results/
            
            # Copy results to solution folder
            cp -r ./logs/nangate45_3D/${DEF_VERSION}/3D/* ../../HMSA_solution_eval/${DEF_VERSION}_3D/${solution_idx}/openroad_logs/
            cp ./final.csv ../../HMSA_solution_eval/${DEF_VERSION}_3D/${solution_idx}/
            cp -r ./results/nangate45_3D/${DEF_VERSION}/3D/* ../../HMSA_solution_eval/${DEF_VERSION}_3D/${solution_idx}/openroad_results/
            echo "Completed processing solution index: $solution_idx"
        done 
    else
        cp ../../Place-3D/install/results/${DEF_VERSION}_${DESIGN_DIMENSION}/${DEF_VERSION}_${DESIGN_DIMENSION}.gp.def designs/nangate45_3D/${DESIGN_NAME}/
        make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config_upper_shrink.mk do-autoflow 
        make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config.mk do-cts_eval 
        make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config.mk do-hotspot
    fi
    

elif [ "$DESIGN_DIMENSION" = "3D_tiling" ]
then
    cp ../../Place-3D/install/results/${DEF_VERSION}_${DESIGN_DIMENSION}/${DEF_VERSION}_${DESIGN_DIMENSION}.gp.def designs/nangate45_3D/${DESIGN_NAME}/
    make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config_tiling_upper_shrink.mk do-autoflow 
    make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config_tiling.mk do-cts_eval 
    make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config_tiling.mk do-hotspot 

elif [ "$DESIGN_DIMENSION" = "true_3D" ]
then
    cp ../../Place-3D/install/results/${DEF_VERSION}_${DESIGN_DIMENSION}/${DEF_VERSION}_${DESIGN_DIMENSION}.gp.def designs/nangate45_3D/${DESIGN_NAME}/
    make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config_true_3D_upper_shrink.mk do-autoflow 
    make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config_true_3D.mk do-cts_eval 
    make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config_true_3D.mk do-hotspot 

elif [ "$DESIGN_DIMENSION" = "2D" ] 
then
    cp ../../Place-3D/install/results/${DEF_VERSION}_${DESIGN_DIMENSION}/${DEF_VERSION}_${DESIGN_DIMENSION}.gp.def designs/nangate45/${DESIGN_NAME}/
    make DESIGN_CONFIG=designs/nangate45/${DESIGN_NAME}/config_2d_dmp.mk do-def_eval
    make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config.mk do-hotspot_2D

elif [ "$DESIGN_DIMENSION" = "2D_mp" ]
then
    cp ../../Place-3D/install/results/${DEF_VERSION}_${DESIGN_DIMENSION}/${DEF_VERSION}_${DESIGN_DIMENSION}.gp.def designs/nangate45/${DESIGN_NAME}/
    make DESIGN_CONFIG=designs/nangate45/${DESIGN_NAME}/config_mp.mk do-def_eval
    make DESIGN_CONFIG=designs/nangate45_3D/${DESIGN_NAME}/config.mk do-hotspot_2D

else
    echo "unknown running mode, exit"
fi
python3 scripts_3D/get_metrics.py
echo "job done, exit"