#!/bin/bash

bash scripts_3D/autoflow.sh 3D bp_be bp_be_top
bash scripts_3D/autoflow.sh 3D bp_fe bp_fe_top
bash scripts_3D/autoflow.sh 3D ariane133 ariane133
bash scripts_3D/autoflow.sh 3D ariane136 ariane136
bash scripts_3D/autoflow.sh 3D bp black_parrot
bash scripts_3D/autoflow.sh 3D bp_multi bp_multi_top
bash scripts_3D/autoflow.sh 3D swerv_wrapper swerv_wrapper
bash scripts_3D/autoflow.sh 3D bp_quad bp_quad

bash scripts_3D/autoflow.sh 3D_tiling bp_be bp_be_top
bash scripts_3D/autoflow.sh 3D_tiling bp_fe bp_fe_top
bash scripts_3D/autoflow.sh 3D_tiling ariane133 ariane133
bash scripts_3D/autoflow.sh 3D_tiling ariane136 ariane136
bash scripts_3D/autoflow.sh 3D_tiling bp black_parrot
bash scripts_3D/autoflow.sh 3D_tiling bp_multi bp_multi_top
bash scripts_3D/autoflow.sh 3D_tiling swerv_wrapper swerv_wrapper
bash scripts_3D/autoflow.sh 3D_tiling bp_quad bp_quad