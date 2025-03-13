# Open3DBench

Official implementation of the paper "Open3DBench: Open-Source Benchmark for 3D-IC Backend Implementation and PPA Evaluation". We aim to build a standardized, replicable, and user-friendly framework for 3D-IC PPA evaluation, specifically providing a benchmark for 3D placement techniques.

## 0. Overview

We provide 8 synthesized netlists in this repository, which can be partitioned and placed in Place-3D (including the two macro placement strategies mentioned in the paper) to generate DEF results.

The DEF results will be processed in OpenROAD-3D for Place-Opt, CTS, Legalization, Routing, RC Extraction, and HotSpot thermal simulation, leading to comprehensive PPA evaluation results.

The above two steps can be done sequentially or separately. Note that due to the small randomness in DREAMPlace across different machines (which slightly affects layout and thus PPA results), if you obtain layout results using Place-3D, perfect reproduction of PPA results is not guaranteed. However, our tests show that the impact is minimal and does not affect the conclusions of the paper. To reproduce the PPA results, we provide 16 DEF layouts corresponding to the two 3D placement strategies in Table 2 of the paper for 8 designs. By deploying the corresponding version of OpenROAD as instructed below, you can perfectly reproduce the PPA results in the table.

## 1. How to build

### 1.1 Build Place-3D

If you want to perform partitioning and 3D placement, you need to install the standard environment for DREAMPlace. We strongly recommend using the official Docker environment provided by DREAMPlace as it is very convenient and easy to use.

After configuring the environment, you need to first download the [dataset](https://drive.google.com/file/d/15D2ge4FJsn4HP4o4AVzoQms6Xx-3ugZ0/view?usp=sharing) and place it in the `Place-3D/` directory.

Next, execute the following commands:

```bash
cd Open3DBench/Place-3D
mkdir build
cd build
cmake ..
make
make install
cd ..
```

At this point, Place-3D is successfully configured.

### 1.2 Build OpenROAD-3D

If you want to perform 3D PPA evaluation, you need to install the corresponding version of OpenROAD. The relevant commit hash is specified in the paper as [fbca14c](https://github.com/The-OpenROAD-Project/OpenROAD/commit/fbca14c). From our tests, we found that a pre-built [binary](https://github.com/Precision-Innovations/OpenROAD/releases/tag/2.0-17198-g8396d0866) version can also perfectly reproduce the PPA results. We strongly recommend directly installing this binary as it is very convenient and easy to use. Installation details can be found [here](https://openroad-flow-scripts.readthedocs.io/en/latest/user/BuildWithPrebuilt.html).

## 2. How to use

### 2.1 Run Place-3D

Place-3D includes 3D tier partitioning and 3D placement. The initial netlists are already included in the benchmark downloaded earlier. After following the build process above, navigate to the Place-3D directory: `cd Open3DBench/Place-3D`, where multiple run scripts are provided.

The script `run_2D.sh` includes commands for running `Hier-RTLMP-2D` and `DREAMPlace-2D` (with pre-computed macro placement results for Hier-RTLMP; only global placement is executed here). The script `run_3D.sh` includes commands for running `Open3D-DMP`, and `run_3D_tiling.sh` includes commands for running `Open3D-Tiling`. For specific commands, refer to `run_2D_all.sh` and `run_3D_all.sh`, which can be executed directly to run all designs, or you can copy a single line to execute an individual case.

If you successfully execute the steps above, the corresponding DEF output will be located in the path `Open3DBench/Place-3D/install/results/`. You can proceed to the next step for PPA evaluation.

If you cannot run Place-3D but want to directly reproduce the 3D PPA evaluation results, you can use the DEF files we provide. These correspond to the results of `Open3D-Tiling` and `Open3D-DMP` methods on all 8 cases in Table 3 of the paper. They can be used directly for the next evaluation step. Download the `results.zip` file from this [link](https://drive.google.com/file/d/18RCp2zEz23TpSA8kvkA9XBfv-ZvzUub3/view?usp=sharing) and extract it to the path `Open3DBench/Place-3D/install/`. Note that if you have not built Place-3D, you will need to manually create the `install` folder in the Place-3D directory.

### 2.2 Run OpenROAD-3D

OpenROAD-3D includes the complete 3D PPA evaluation flow. Navigate to the directory `cd Open3DBench/OpenROAD-3D/flow`, where two scripts, `run_2D.sh` and `run_3D.sh`, are provided for running the two 2D methods and two 3D methods, respectively. You can copy any line from these scripts and run it directly in the current directory to execute the corresponding placement strategy and test design. Here, `2D` refers to `DREAMPlace-2D`, `2D_mp` refers to `Hier-RTLMP-2D`, `3D` refers to `Open3D-DMP`, and `3D_tiling` refers to `Open3D-Tiling`. Note that the corresponding placement steps in Section 2.1 must be completed beforehand, as the scripts will look for the layout DEF results in the path `Open3DBench/Place-3D/install/results/` as input. The results will be stored in the directories `Open3DBench/OpenROAD-3D/logs` and `Open3DBench/OpenROAD-3D/results`.

## 3. Citation

```

```