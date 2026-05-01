#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

TOTAL_SOLUTIONS=10126
DESIGN_NAME=bp_multi_3D
CASE_NAME=bp_multi
HMSA_RESULTS_DIR="${SCRATCH}/bp_multi_4"


sbatch \
  --export=ALL,TOTAL_SOLUTIONS="${TOTAL_SOLUTIONS}",DESIGN_NAME="${DESIGN_NAME}",CASE_NAME="${CASE_NAME}",HMSA_RESULTS_DIR="${HMSA_RESULTS_DIR}" \
  dp_hmsa_cc.slurm