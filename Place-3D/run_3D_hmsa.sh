design_name=$1
num_solutions=$2

# Create HMSA_solution_eval folder if it doesn't exist
mkdir -p ../HMSA_solution_eval/${design_name}_3D

cd build
cmake ..
make -j 8
make -j 8 install
cd ../install

for i in $(seq 0 $((num_solutions - 1)))
do
    mkdir -p ../../HMSA_solution_eval/${design_name}_3D/${i}
    echo "Running placement for solution index: $i"
    python dreamplace/Placer_3D_hmsa.py test/or_3D/${design_name}_3D.json $i
    cp -r ./results/${design_name}_3D/* ../../HMSA_solution_eval/${design_name}_3D/${i}/
done