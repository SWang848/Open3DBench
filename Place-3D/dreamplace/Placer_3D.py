##
# @file   Placer.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  Main file to run the entire placement flow.
#

import matplotlib
matplotlib.use('Agg')
import os
import sys
import time
import numpy as np
import logging
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import dreamplace.configure as configure
import Params
import PlaceDB
import Timer
import NonLinearPlace
import pdb
import re
import torch
import random

from Partitioner import partition
from TPGNN import TPGNN

def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def place(params, partition_result=None, choice=None, upper_die_names=None):
    """
    @brief Top API to run the entire placement flow.
    @param params parameters
    """

    assert (not params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
            "CANNOT enable GPU without CUDA compiled"

    # read database
    tt = time.time()
    placedb = PlaceDB.PlaceDB()
    if choice != '2D-init':
        placedb(params, partition_result, upper_die_names)
    else:
        placedb(params)
    logging.info("reading database takes %.2f seconds" % (time.time() - tt))
    
    # Read timing constraints provided in the benchmarks into out timing analysis
    # engine and then pass the timer into the placement core.
    timer = None
    if params.timing_opt_flag:
        tt = time.time()
        timer = Timer.Timer()
        timer(params, placedb)
        # This must be done to explicitly execute the parser builders.
        # The parsers in OpenTimer are all in lazy mode.
        timer.update_timing()
        logging.info("reading timer takes %.2f seconds" % (time.time() - tt))

        # Dump example here. Some dump functions are defined.
        # Check instance methods defined in Timer.py for debugging.
        # timer.dump_pin_cap("pin_caps.txt")
        # timer.dump_graph("timing_graph.txt")

    # solve placement
    tt = time.time()
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    logging.info("non-linear placement initialization takes %.2f seconds" %
                 (time.time() - tt))
    metrics = placer(params, placedb)
    logging.info("non-linear placement takes %.2f seconds" %
                 (time.time() - tt))

    # write placement solution
    path = "%s/%s" % (params.result_dir, params.design_name())
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    gp_out_file = os.path.join(
        path,
        "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
    placedb.write(params, gp_out_file)
    """
    Post-process def files, including fixing macros, renaming components (i.e., add "_upper" and "_bottom"), and tuning the coordinates.
    """
    # for 2D placement
    if choice == '2D':
        new_def = ''
        with open(gp_out_file, 'r', encoding='utf-8') as def_file:
            indicator = False
            pre_line = ''
            for line in def_file:
                if 'COMPONENTS' in line:
                    indicator = True
                    if 'END' in line:
                        indicator = False
                
                if indicator:
                    if 'fakeram' in pre_line:
                        line = line.replace('PLACED', 'FIXED')
                        line = line.replace('FS', 'N')  # only "N" is allowed
                        # adjust y coordinate for pin alignment
                        y = line.split()[4]
                        num_row = int((float(y) - 70) / 280)
                        y_ = str(280 * num_row + 70)
                        line = line.replace(y, y_)
                new_def = new_def + pre_line    
                pre_line = line
            new_def = new_def + line
        with open(gp_out_file, 'w', encoding='utf-8') as def_file:
            def_file.write(new_def)
    
    # for initialization of 3D placement
    if choice == '2D-init':
        if partition_result is not None:
            name_map = {}
            for name in placedb.node_names:
                name = name.decode('utf-8')
                if placedb.node_name2id_map[name] in partition_result:
                    name_map[name] = '_bottom'
                else:
                    name_map[name] = '_upper'
                    
        new_def = ''
        with open(gp_out_file, 'r', encoding='utf-8') as def_file:
            indicator = False
            pre_line = ''
            for line in def_file:
                if 'COMPONENTS' in line:
                    indicator = True
                    if 'END' in line:
                        indicator = False
                
                if indicator:
                    if ('PLACED' in line) or ('FIXED' in line):
                        name = pre_line.split()[-2]
                        class_name = pre_line.split()[-1]
                        pure_name = name.split('/')[-1]
                        if 'bottom' in name_map[name]:
                            line = line.replace('PLACED', 'FIXED')
                        
                new_def = new_def + pre_line    
                pre_line = line
            new_def = new_def + line
        with open("benchmarks/or_3D/intermediate_result/upper.def", 'w', encoding='utf-8') as def_file:
            def_file.write(new_def)
    
    # for memory placement on the upper die
    if choice == 'mem-upper':
        if partition_result is not None:
            name_map = {}
            for name in placedb.node_names:
                name = name.decode('utf-8')
                if placedb.node_name2id_map[name] in partition_result:
                    name_map[name] = '_bottom'
                else:
                    name_map[name] = '_upper'      
        new_def = ''
        with open(gp_out_file, 'r', encoding='utf-8') as def_file:
            indicator = False
            pre_line = ''
            for line in def_file:
                if 'COMPONENTS' in line:
                    indicator = True
                    if 'END' in line:
                        indicator = False
                
                if indicator:
                    if ('PLACED' in line) or ('FIXED' in line):
                        name = pre_line.split()[-2]
                        class_name = pre_line.split()[-1]
                        if 'upper' in name_map[name]:
                            line = line.replace('PLACED', 'FIXED')
                            line = line.replace('FS', 'N')  # only "N" is allowed
                            # adjust y coordinate for pin alignment
                            y = line.split()[4]
                            num_row = int((float(y) - 70) / 280)
                            y_ = str(280 * num_row + 70)
                            line = line.replace(y, y_)
                            
                new_def = new_def + pre_line    
                pre_line = line
            new_def = new_def + line
        with open("benchmarks/or_3D/intermediate_result/mem_upper.def", 'w', encoding='utf-8') as def_file:
            def_file.write(new_def)
    
    # for memory placement on the bottom die
    if choice == 'mem-bottom':
        new_def = ''
        with open(gp_out_file, 'r', encoding='utf-8') as def_file:
            indicator = False
            pre_line = ''
            for line in def_file:
                if 'COMPONENTS' in line:
                    indicator = True
                    if 'END' in line:
                        indicator = False
                
                if indicator:
                    if 'fakeram' in pre_line:
                        line = line.replace('PLACED', 'FIXED')
                        # only "N" is allowed
                        line = line.replace('FS', 'N')  
                        # adjust y coordinate for pin alignment
                        y = line.split()[4]
                        num_row = int((float(y) - 70) / 280)
                        y_ = str(280 * num_row + 70)
                        line = line.replace(y, y_)

                new_def = new_def + pre_line    
                pre_line = line
            new_def = new_def + line
        with open("benchmarks/or_3D/intermediate_result/mem_bot.def", 'w', encoding='utf-8') as def_file:
            def_file.write(new_def)

    # for mixed placement on the upper die
    if choice == 'upper_die_placement':
        new_def = ''
        with open(gp_out_file, 'r', encoding='utf-8') as def_file:
            indicator = False
            for line in def_file:
                if 'COMPONENTS' in line:
                    indicator = True
                    if 'END' in line:
                        indicator = False
                
                if indicator:
                    if 'PLACED' in line:
                        line = line.replace('PLACED', 'FIXED')
                    elif 'FIXED' in line:
                        line = line.replace('FIXED', 'PLACED')
                new_def = new_def + line
        with open("benchmarks/or_3D/intermediate_result/bottom.def", 'w', encoding='utf-8') as def_file:
            def_file.write(new_def)
    
    # for mixed / cell placement on the bottom die
    if choice == 'bot_die_placement':
        new_def = ''
        with open(gp_out_file, 'r', encoding='utf-8') as def_file:
            indicator = False
            pre_line = ''
            for line in def_file:
                if 'COMPONENTS' in line:
                    indicator = True
                    if 'END' in line:
                        indicator = False
                
                if indicator:
                    if ('PLACED' in line) or ('FIXED' in line):
                        class_name = pre_line.split()[-1]
                        name = pre_line.split()[-2]
                        if name in upper_die_names:
                            pre_line = pre_line.replace(class_name, class_name + '_upper')
                        else:
                            pre_line = pre_line.replace(class_name, class_name + '_bottom')
                new_def = new_def + pre_line
                pre_line = line
            new_def = new_def + line
        with open(gp_out_file, 'w', encoding='utf-8') as def_file:
            def_file.write(new_def)
                        
    # call external detailed placement
    # TODO: support more external placers, currently only support
    # 1. NTUplace3/NTUplace4h with Bookshelf format
    # 2. NTUplace_4dr with LEF/DEF format
    if params.detailed_place_engine and os.path.exists(
            params.detailed_place_engine):
        logging.info("Use external detailed placement engine %s" %
                     (params.detailed_place_engine))
        if params.solution_file_suffix() == "pl" and any(
                dp_engine in params.detailed_place_engine
                for dp_engine in ['ntuplace3', 'ntuplace4h']):
            dp_out_file = gp_out_file.replace(".gp.pl", "")
            # add target density constraint if provided
            target_density_cmd = ""
            if params.target_density < 1.0 and not params.routability_opt_flag:
                target_density_cmd = " -util %f" % (params.target_density)
            cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal %s" % (
                params.detailed_place_engine, params.aux_input, gp_out_file,
                target_density_cmd, dp_out_file, params.detailed_place_command)
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            logging.info("External detailed placement takes %.2f seconds" %
                         (time.time() - tt))

            if params.plot_flag:
                # read solution and evaluate
                placedb.read_pl(params, dp_out_file + ".ntup.pl")
                iteration = len(metrics)
                pos = placer.init_pos
                pos[0:placedb.num_physical_nodes] = placedb.node_x
                pos[placedb.num_nodes:placedb.num_nodes +
                    placedb.num_physical_nodes] = placedb.node_y
                hpwl, density_overflow, max_density = placer.validate(
                    placedb, pos, iteration)
                logging.info(
                    "iteration %4d, HPWL %.3E, overflow %.3E, max density %.3E"
                    % (iteration, hpwl, density_overflow, max_density))
                placer.plot(params, placedb, iteration, pos)
        elif 'ntuplace_4dr' in params.detailed_place_engine:
            dp_out_file = gp_out_file.replace(".gp.def", "")
            cmd = "%s" % (params.detailed_place_engine)
            for lef in params.lef_input:
                if "tech.lef" in lef:
                    cmd += " -tech_lef %s" % (lef)
                else:
                    cmd += " -cell_lef %s" % (lef)
                benchmark_dir = os.path.dirname(lef)
            cmd += " -floorplan_def %s" % (gp_out_file)
            if(params.verilog_input):
                cmd += " -verilog %s" % (params.verilog_input)
            cmd += " -out ntuplace_4dr_out"
            cmd += " -placement_constraints %s/placement.constraints" % (
                # os.path.dirname(params.verilog_input))
                benchmark_dir)
            cmd += " -noglobal %s ; " % (params.detailed_place_command)
            # cmd += " %s ; " % (params.detailed_place_command) ## test whole flow
            cmd += "mv ntuplace_4dr_out.fence.plt %s.fence.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out.init.plt %s.init.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out %s.ntup.def ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.overflow.plt %s.ntup.overflow.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.plt %s.ntup.plt ; " % (
                dp_out_file)
            if os.path.exists("%s/dat" % (os.path.dirname(dp_out_file))):
                cmd += "rm -r %s/dat ; " % (os.path.dirname(dp_out_file))
            cmd += "mv dat %s/ ; " % (os.path.dirname(dp_out_file))
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            logging.info("External detailed placement takes %.2f seconds" %
                         (time.time() - tt))
        else:
            logging.warning(
                "External detailed placement only supports NTUplace3/NTUplace4dr API"
            )
    elif params.detailed_place_engine:
        logging.warning(
            "External detailed placement engine %s or aux file NOT found" %
            (params.detailed_place_engine))

    return metrics


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow.
    """
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        # stream=sys.stdout)
                        filename='placement_3D_ariane133.log',
                        filemode='w')
    
    TPGNN_flag = True
    params = Params.Params()
    seed_everything(params.random_seed)
    params.printWelcome()
    if len(sys.argv) == 1 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        params.printHelp()
        exit()
    elif len(sys.argv) != 2:
        logging.error("One input parameters in json format in required")
        params.printHelp()
        exit()

    # load parameters
    params.load(sys.argv[1])
    logging.info("parameters = %s" % (params))
    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    '''
    params.partition_params["type"]:
        0: No partition, i.e., 2D placement.
        1: All macros on the upper die, i.e., pure memory-on-logic placement.
        2: Some macros are on the bottom die, supporting both max cut and min cut.
        3: Mixed placement on both upper and bottom dies.
    params.shrink["type"]:
        1: Sizes of all components are divided by 2.
        2: Shrink the bottom-die components to a small size, for upper-die marco placement. However, the shrunk size can not be too small, which may lead to adding too many fillers. Empirically set to 1000. 
        3: Shrink all the fixed components to a small size of 0.001.
        4: Shrink the upper-die components to a small size of 0.001.
    '''
    
    if params.partition_params["type"] == 0:   # no partition
        partition_result = None
        # run 2D placement
        tt = time.time()
        place(params, partition_result, choice='2D')
        
        logging.info("2D placement takes %.3f seconds" % (time.time() - tt))
    
    if params.partition_params["type"] == 1:
        # run partition
        partition_result, upper_die_names = partition(params)
        # run 2D placement for memory placement
        tt = time.time()
        place(params, partition_result, choice="mem-upper")
        
        logging.info("mem-upper placement takes %.3f seconds" % (time.time() - tt))
        
        # run bottom die placement
        tt = time.time()
        params.shrink['type'] = 4
        params.def_input = "benchmarks/or_3D/intermediate_result/mem_upper.def"
        place(params, partition_result, choice="bot_die_placement", upper_die_names=upper_die_names)
        
        logging.info("Bottom die placement takes %.3f seconds" % (time.time() - tt))
        
    if params.partition_params["type"] == 2:
        # run partition
        if TPGNN_flag:
            # Initialize placement database
            placedb = PlaceDB.PlaceDB()
            placedb(params)
            
            # Initialize TPGNN with placement database
            tpgnn = TPGNN(placedb)
            
            # Load timing features - construct path relative to current working directory
            timing_report_path = os.path.join(os.getcwd(), params.timing_report_input)
            tpgnn.timing_features = tpgnn.parse_timing_report(timing_report_path)
            
            # Build clique graph
            G = tpgnn.clique_graph_construction()
            
            # Apply hierarchy-aware edge contraction
            G_contracted = tpgnn.hierarchy_aware_graph_construction(G)
            
            # # reset the def_input to the original def file
            # reorder_nodes(params)
            
            
            # Generate embeddings using TP-GNN
            tpgnn_model, tpgnn_results = tpgnn.generate_embeddings(G_contracted, 
                                           output_dir="./tpgnn_results", 
                                           epochs=1)
            
            # Run partitioning using GNN embeddings
            partition_result, upper_die_names = tpgnn.partition(
                G,
                G_contracted,
                tpgnn_results['final_embeddings'],
                output_dir="./tpgnn_results"
            )
            
            # update partition result based on the clear def file
            placedb = PlaceDB.PlaceDB()
            params.placed_def_input = ""
            placedb(params)
            partition_result = []
            for name in placedb.node_names:
                name = name.decode('utf-8')
                node = placedb.node_name2id_map[name]
                if node < (placedb.num_physical_nodes - placedb.num_terminal_NIs):    
                    if name in upper_die_names:
                        continue
                    else:
                        partition_result.append(node)
            
        else:
            partition_result, upper_die_names = partition(params)
        # run 2D placement for memory placement
                    # --- START: ADDED FOR DEBUGGING ---
        # Log the area distribution after partitioning
        placedb = PlaceDB.PlaceDB()
        placedb(params)
        top_die_macro_area = 0
        bottom_die_macro_area = 0
        top_die_macro_count = 0
        bottom_die_macro_count = 0

        # Re-check is_macro for the new placedb instance
        is_macro_after_partition = np.zeros(placedb.num_nodes, dtype=bool)
        for i in range(placedb.num_physical_nodes):
                # A common heuristic: if a cell is much taller than a standard cell row, it's a macro
            if placedb.node_size_y[i] > placedb.row_height * 2:
                is_macro_after_partition[i] = True

        for i in range(placedb.num_physical_nodes):
            if is_macro_after_partition[i]:
                node_id = i
                node_name = placedb.node_names[i].decode('utf-8')
                area = placedb.node_size_x[i] * placedb.node_size_y[i]
                if node_name in upper_die_names:
                    top_die_macro_area += area
                    top_die_macro_count += 1
                else:
                    bottom_die_macro_area += area
                    bottom_die_macro_count += 1
        
        logging.info("--- Partition Analysis ---")
        logging.info(f"Top Die Macro Count: {top_die_macro_count}")
        logging.info(f"Top Die Macro Area: {top_die_macro_area:.2f}")
        logging.info(f"Bottom Die Macro Count: {bottom_die_macro_count}")
        logging.info(f"Bottom Die Macro Area: {bottom_die_macro_area:.2f}")
        logging.info("--------------------------")
        # --- END: ADDED FOR DEBUGGING ---
        
        tt = time.time()
        params.plot_flag = 0
        place(params, partition_result, choice="mem-upper")
        breakpoint()
        logging.info("mem-upper placement takes %.3f seconds" % (time.time() - tt))
        
        # run 2D placement for memory placement
        tt = time.time()
        params.shrink['type'] = 3
        # params.enable_fillers = 1
        # params.target_density = 0.4
        params.def_input = "benchmarks/or_3D/intermediate_result/mem_upper.def"
        place(params, partition_result, choice="mem-bottom")
        
        logging.info("mem-bottom placement takes %.3f seconds" % (time.time() - tt))
        
        # run bottom die placement
        tt = time.time()
        params.shrink['type'] = 4
        params.def_input = "benchmarks/or_3D/intermediate_result/mem_bot.def"
        params.plot_flag = 1
        place(params, partition_result, choice="bot_die_placement", upper_die_names=upper_die_names)
        
        logging.info("Bottom die placement takes %.3f seconds" % (time.time() - tt))
        
    if params.partition_params["type"] == 3:
        # run partition
        partition_result = partition(params)
        # run 2D placement
        tt = time.time()
        place(params, partition_result, choice="2D-init")
        
        logging.info("2D placement takes %.3f seconds" % (time.time() - tt))
        
        # run upper die placement
        tt = time.time()
        params.target_density = 1.0
        params.legalize_flag = 0
        params.def_input = "benchmarks/or_3D/intermediate_result/upper.def"
        place(params, partition_result, choice="upper_die_placement")

        logging.info("Upper die placement takes %.3f seconds" % (time.time() - tt))

        # run bottom die placement
        tt = time.time()
        params.target_density = 0.8
        params.legalize_flag = 0
        params.def_input = "benchmarks/or_3D/intermediate_result/bottom.def"
        place(params, partition_result, choice="bot_die_placement")
        
        logging.info("Bottom die placement takes %.3f seconds" % (time.time() - tt))
    
