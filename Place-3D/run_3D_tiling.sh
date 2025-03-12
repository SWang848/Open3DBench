design_name=$1

cd build
cmake ..
make -j 8
make -j 8 install
cd ../install
# output_file="timer_results_3D_tiling.csv"
# if [ ! -f "$output_file" ]; then
#     echo "Design_name, Elapsed_time" > "$output_file"
# fi
start_seconds=$(date +%s)
python dreamplace/Placer_3D_heuristic.py test/or_3D_tiling/${design_name}_3D.json
end_seconds=$(date +%s)
elapsed_time=$((end_seconds - start_seconds))
