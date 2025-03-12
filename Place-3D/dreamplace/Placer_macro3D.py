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

def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
def describe_macros(db):
    # compute area
    mean_node_area = 0.
    num = 0
    all_nodes = []
    for node_name in db.node_names:
        node = db.node_name2id_map[node_name.decode('utf-8')]
        all_nodes.append(node)
        if node < (db.num_physical_nodes - db.num_terminal_NIs):  # exclude IO ports
            node_area = db.node_size_x[node] * db.node_size_y[node]
            mean_node_area += node_area
            num += 1
    
    mean_node_area = mean_node_area / num
    
    # collect macros
    macros = []
    for node_name in db.node_names:
        node = db.node_name2id_map[node_name.decode('utf-8')]
        if node < (db.num_physical_nodes - db.num_terminal_NIs):  # exclude IO ports
            node_area = db.node_size_x[node] * db.node_size_y[node]
            if (node_area > (mean_node_area * 10)) and (db.node_size_y[node] > (db.row_height * 2)):
                macros.append({'name': node_name.decode('utf-8'),
                               'node': node,
                               'size_x': db.node_size_x[node],
                               'size_y': db.node_size_y[node]})
    
    return macros, all_nodes

def place_macros(macros, db, all_nodes):
    # internals between macros 
    internal_w = 0.5 * float((db.xh - db.xl) / len(macros))
    internal_h = 0.5 * float((db.yh - db.yl) / len(macros))
    
    def greedy(macros):
        # sort all the macros by width (first) and height (second)
        sorted_macros = sorted(macros, key=lambda x: (-x['size_y'], -x['size_x']))
        
        placements = []
        move_to_bot_idx = []
        skyline = [(db.xl, db.xh - db.xl, db.yl)]
        placements = []

        for j, macro in enumerate(sorted_macros):
            aw, ah = macro['size_x'] + 2 * internal_w, macro['size_y'] + 2 * internal_h
            
            best_height = float('inf')
            best_pos = None
            best_segment_idx = -1

            # best position for placement
            for i, (x_s, x_e, s_h) in enumerate(skyline):
                if (x_e - x_s >= aw) and (s_h + ah < db.yh):
                    current_height = s_h + ah
                    # lowest and leftest
                    if (current_height < best_height or 
                        (current_height == best_height and x_s < best_pos[0])):
                        best_height = current_height
                        best_pos = (x_s, s_h)
                        best_segment_idx = i
            if best_pos is None:
                move_to_bot_idx.append(j)
                continue
                
            # update skyline
            x_place, y_place = best_pos
            placements.append((x_place + internal_w, y_place + internal_h))
            seg_x_s, seg_x_e, seg_h = skyline[best_segment_idx]

            # insert new skyline
            new_seg = (x_place, x_place + aw, seg_h + ah)
            remaining_right = (x_place + aw, seg_x_e, seg_h) if (x_place + aw < seg_x_e) else None

            del skyline[best_segment_idx]
            skyline.insert(best_segment_idx, new_seg)
            if remaining_right:
                skyline.insert(best_segment_idx + 1, remaining_right)

            # merge neighbor lines
            if best_segment_idx > 0:
                prev = skyline[best_segment_idx - 1]
                current = skyline[best_segment_idx]
                if prev[1] == current[0] and prev[2] == current[2]:
                    merged = (prev[0], current[1], current[2])
                    skyline[best_segment_idx - 1: best_segment_idx + 1] = [merged]
                    best_segment_idx -= 1
        
        if len(move_to_bot_idx) > 0:
            # move left macros to the bottom die
            bottom_macros = [sorted_macros[idx] for idx in move_to_bot_idx]
            # partition results
            upper_die_id = []
            upper_die_names = []
            bot_die_id = []
            for i in range(len(sorted_macros)):
                if i not in move_to_bot_idx:
                    upper_die_id.append(sorted_macros[i]['node'])
                    upper_die_names.append(sorted_macros[i]['name'])
                else:
                    bot_die_id.append(sorted_macros[i]['node'])
        else:
            upper_die_id = [macro['node'] for macro in sorted_macros]
            upper_die_names = [macro['name'] for macro in sorted_macros]
            bottom_macros, bot_die_id = None, None

        return placements, bottom_macros, upper_die_id, bot_die_id, upper_die_names


    # greedy placement for upper_die macros
    upper_die_placements, bottom_macros, upper_die_id, bot_die_id, upper_die_names = greedy(macros)
    # greedy placement for bot_die macros
    if bot_die_id is not None:
        bot_die_placements, _, _, _, _ = greedy(bottom_macros)

    # write placedb
    for i, (x, y) in enumerate(upper_die_placements):
        db.node_x[upper_die_id[i]] = x
        db.node_y[upper_die_id[i]] = y
    if bot_die_id is not None:
        for i, (x, y) in enumerate(bot_die_placements):
            db.node_x[bot_die_id[i]] = x
            db.node_y[bot_die_id[i]] = y
    
    return db, upper_die_names

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
    placedb(params, partition_result, upper_die_names)
    
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
    if choice == 'mem':
        macros, all_nodes = describe_macros(placedb)
        placedb, output_upper_die_names = place_macros(macros, placedb, all_nodes)
        metrics = None
    else: 
        tt = time.time()
        placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
        logging.info("non-linear placement initialization takes %.2f seconds" %
                    (time.time() - tt))
        metrics = placer(params, placedb)
        logging.info("non-linear placement takes %.2f seconds" %
                    (time.time() - tt))
        
        output_upper_die_names = None

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
    
    # for memory placement on the upper die
    if choice == 'mem':
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
                        line = line.replace('UNPLACED', 'FIXED')
                        # only "N" is allowed
                        line = line.replace('UNKNOWN', 'N')  
                        # adjust y coordinate for pin alignment
                        x = line.split()[3]
                        y = line.split()[4]
                        num_row = int((float(y) - 70) / 280)
                        y_ = str(280 * num_row + 70)
                        x_ = str(int((float(x)) / 10.0) * 10)
                        line = line.replace(y, y_)
                        line = line.replace(x, x_)
                    
                    if 'UNKNOWN' in line:
                        line = line.replace('UNKNOWN', 'N') 
                new_def = new_def + pre_line    
                pre_line = line
            new_def = new_def + line
        with open("benchmarks/or_3D/intermediate_result/mem_place.def", 'w', encoding='utf-8') as def_file:
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

    return metrics, output_upper_die_names


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow.
    """
    
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
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
    
    # run partition
    # partition_result, upper_die_names = partition(params)
    
    # memory placement by hand-crafted rules
    tt = time.time()
    params.plot_flag = 0
    metrics, upper_die_names = place(params, choice="mem")

    logging.info("mem placement takes %.3f seconds" % (time.time() - tt))
    
    # run global placement
    tt = time.time()
    params.shrink['type'] = 4
    params.def_input = "benchmarks/or_3D/intermediate_result/mem_place.def"
    params.plot_flag = 1
    place(params, choice="bot_die_placement", upper_die_names=upper_die_names)
    
    logging.info("Bottom die placement takes %.3f seconds" % (time.time() - tt))