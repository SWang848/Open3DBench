design_name=$1

cd build
cmake ..
make -j 8
make -j 8 install
cd ../install
# cd install
cp ../../OpenROAD-3D/flow/reports/nangate45/${design_name}/2D_dmp/post_place_timing_setup.rpt ./results/${design_name}_2D
output_file="timer_results_3D_tpgnn.csv"
if [ ! -f "$output_file" ]; then
    echo "Design_name, Elapsed_time" > "$output_file"
fi
start_seconds=$(date +%s)
python dreamplace/Placer_3D.py test/or_3D/${design_name}_3D.json True
end_seconds=$(date +%s)
elapsed_time=$((end_seconds - start_seconds))
echo "$design_name,$elapsed_time" >> "$output_file"