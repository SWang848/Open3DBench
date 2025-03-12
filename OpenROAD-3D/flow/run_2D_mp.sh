case_name=$1
case_nickname=$2

cp ../../DREAMPlace/install/results/${case_nickname}_2D_mp/${case_nickname}_2D_mp.gp.def designs/nangate45/${case_name}/${case_nickname}_2D_mp.gp.def
bash scripts_3D/autoflow.sh 2D_mp ${case_nickname}_2D_mp ${case_name}

python3 scripts_3D/get_metrics.py

# bash run_2D_mp.sh black_parrot bp