#!/bin/bash
set -euo pipefail

design_name=$1
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
place_dir="${repo_root}/Place-3D"

cd "${place_dir}/build"
cmake ..
make -j 8
make -j 8 install
cd "${place_dir}/install"

export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
cp "${repo_root}/OpenROAD-3D/flow/reports/nangate45/${design_name}/2D_dmp/post_place_timing_setup.rpt" "./results/${design_name}_2D"

output_file="timer_results_3D_tpgnn.csv"
if [ ! -f "$output_file" ]; then
    echo "Design_name, Elapsed_time" > "$output_file"
fi

start_seconds=$(date +%s)
python -m dreamplace.Placer_3D "test/or_3D/${design_name}_3D.json" --tpgnn true
end_seconds=$(date +%s)
elapsed_time=$((end_seconds - start_seconds))
echo "$design_name,$elapsed_time" >> "$output_file"
