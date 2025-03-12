case_name=$1
case_nickname=$2

cp ../../DREAMPlace/install/results/${case_nickname}_2D/${case_nickname}_2D.gp.def designs/nangate45/${case_name}/${case_nickname}_2D.gp.def
bash scripts_3D/autoflow.sh 2D ${case_nickname}_2D ${case_name}

python3 scripts_3D/get_metrics.py

# bash run_2D_dmp.sh bp_be_top bp_be