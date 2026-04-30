#!/bin/bash

for start in $(seq 0 1000 10000); do
  sbatch \
    --export=ALL,START_IDX="${start}",TOTAL_SOLUTIONS=10126,DESIGN_NAME=bp_multi_3D,CASE_NAME=bp_multi,HMSA_RESULTS_DIR="${SCRATCH}/bp_multi_4" \
    Place-3D/dp_hmsa_cc.slurm
done