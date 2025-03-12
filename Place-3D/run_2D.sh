design_dimension=$1
design_name=$2

cd build
cmake ..
make -j
make -j install
cd ../install
output_file="timer_results_$design_dimension.csv"
if [ ! -f "$output_file" ]; then
    echo "Design_name, Elapsed_time" > "$output_file"
fi
start_seconds=$(date +%s)
python dreamplace/Placer.py test/or_${design_dimension}/${design_name}_${design_dimension}.json


if [ "${design_dimension}" = "2D" ]; then
    file=results/${design_name}_${design_dimension}/${design_name}_${design_dimension}.gp.def
    sed -i 's|/|_|g' "$file"

    newfile=${file}_1
    awk '/fakeram/ {print; getline; sub(/PLACED/, "FIXED"); print; next} 1' "$file" > "$newfile"
    mv $newfile $file

    chmod 777 -R ../install

    python dreamplace/fix_and_tune.py --in_def $file --out_def benchmarks/or_${design_dimension}/${design_name}/${design_name}_dmp.gp.def

    python dreamplace/Placer.py test/or_${design_dimension}_stage2/${design_name}_${design_dimension}.json
    end_seconds=$(date +%s)
    elapsed_time=$((end_seconds - start_seconds))
    echo "$design_name,$elapsed_time" >> "$output_file"
fi

# bash run.sh 2D bp_be
# bash run.sh 2D_mp bp_be