import os
import re
import json
import pdb
import logging
from typing import Dict, Tuple, List, Optional, Sequence, Mapping, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_pareto_front(
    final_df: pd.DataFrame,
    x_col: str = "Cut_size",
    y_col: str = "Area_imbalance",
    fitness_col: str = "Fitness",
    save_path: str = "./pareto_front.png",
    max_points: Optional[int] = None,
) -> None:
    """
    Plot the Pareto front using columns already available in `final_df`.

    Args:
        final_df: DataFrame containing the columns used for axes and fitness scoring.
        x_col: Column name to use on the x-axis.
        y_col: Column name to use on the y-axis.
        fitness_col: Column name representing fitness scores.
        save_path: Destination path for the generated plot.
        max_points: Optionally limit the number of solutions plotted (sorted by ascending fitness).
    """

    if final_df.empty:
        logging.warning("final_df is empty. Skipping Pareto plot generation.")
        return

    required_cols = {"Idx", x_col, y_col, fitness_col}
    if not required_cols.issubset(final_df.columns):
        missing = required_cols - set(final_df.columns)
        logging.warning(
            "Missing required columns %s for Pareto plot; expected at least %s.",
            list(missing),
            list(required_cols),
        )
        return

    data = final_df[list(required_cols)].copy()
    data["Idx"] = pd.to_numeric(data["Idx"], errors="coerce")
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data[fitness_col] = pd.to_numeric(data[fitness_col], errors="coerce")

    plot_data = data.dropna(subset=["Idx", x_col, y_col])
    if plot_data.empty:
        logging.warning("No valid data points to plot in Pareto front.")
        return

    plot_data = plot_data.sort_values(fitness_col, ascending=True, na_position='last').reset_index(drop=True)
    if max_points is not None and max_points > 0:
        plot_data = plot_data.head(max_points)

    x_values = plot_data[x_col].to_numpy(dtype=float)
    y_values = plot_data[y_col].to_numpy(dtype=float)
    fitness = plot_data[fitness_col].to_numpy(dtype=float)

    valid_mask = ~pd.isna(fitness)
    x_valid = x_values[valid_mask]
    y_valid = y_values[valid_mask]
    fitness_valid = fitness[valid_mask]

    plt.figure(figsize=(10, 6))
    
    vmin = float(fitness_valid.min())
    vmax = float(fitness_valid.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")
    
    scatter = plt.scatter(
        x_valid,
        y_valid,
        c=fitness_valid,
        cmap=cmap,
        norm=norm,
        s=110,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.8,
    )
    
    # Plot NaN fitness points in gray
    if not valid_mask.all():
        plt.scatter(
            x_values[~valid_mask],
            y_values[~valid_mask],
            c="black",
            s=110,
            alpha=1.0,           # pure dark black
            edgecolors="black",
            linewidths=0.8,
            marker="o",          # solid node
        )

    sorted_points = sorted(zip(x_values, y_values))
    if sorted_points:
        sorted_x, sorted_y = zip(*sorted_points)
        plt.plot(sorted_x, sorted_y, color="gray", linestyle="--", alpha=0.35, linewidth=1)

    def _format_label(col_name: str) -> str:
        return col_name.replace("_", " ").title()

    plt.xlabel(_format_label(x_col), fontsize=12, fontweight="bold")
    plt.ylabel(_format_label(y_col), fontsize=12, fontweight="bold")
    plt.title(f"Pareto Archive: {_format_label(x_col)} vs {_format_label(y_col)}", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Fitness Score", fontsize=11, fontweight="bold")

    top_points = plot_data.head(20).reset_index(drop=True)
    ax = plt.gca()
    for rank, row in top_points.iterrows():
        ax.annotate(
            str(rank),
            xy=(row[x_col], row[y_col]),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"),
        )

    plt.text(
        0.02,
        0.98,
        f"Points: {len(x_values)}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info("Pareto Archive plot saved to '%s'", save_path)
    plt.close()
        
def get_power(data):
    
    power = data.get("finish__power__total", None)

    if power is None:
        return None

    if not isinstance(power, (int, float)):
        return None

    return power

def get_wns_tns(data):
    
    tns = data.get("finish__timing__setup__tns", None)
    wns = data.get("finish__timing__setup__ws", None)

    if wns is None or tns is None:
        return None, None

    if not isinstance(wns, (int, float)):
        return None, None
    if not isinstance(tns, (int, float)):
        return None, None

    return wns, tns

def get_GRT_wns_tns(data):
    
    tns = data.get("globalroute__timing__setup__tns", None)
    wns = data.get("globalroute__timing__setup__ws", None)

    if wns is None or tns is None:
        return None, None

    if not isinstance(wns, (int, float)):
        return None, None
    if not isinstance(tns, (int, float)):
        return None, None

    return wns, tns

def get_area(data):
    
    area = data.get("finish__design__instance__area__stdcell", None)

    if area is None:
        return None

    if not isinstance(area, (int, float)):
        return None

    return area

def get_die_area(data):
    
    area = data.get("finish__design__die__area", None)

    if area is None:
        return None

    if not isinstance(area, (int, float)):
        return None

    return area

def get_wirelength(data):
   
    wirelength = data.get("detailedroute__route__wirelength", None)

    if wirelength is None:
        return None

    if not isinstance(wirelength, (int, float)):
        return None

    return wirelength

def get_nvp(data):
    
    nvp = data.get("finish__timing__drv__setup_violation_count", None)

    if nvp is None:
        return None

    if not isinstance(nvp, (int, float)):
        return None

    return nvp

def get_statistics(data):
    macro_num = data.get("finish__design__instance__count__macros", None)
    utilization = data.get("finish__design__instance__utilization", None)
    cell_num = data.get("finish__design__instance__count__stdcell", None)

    return macro_num, cell_num, utilization

def get_net_num(data):
    net_num = data.get("detailedroute__route__net", None)
    return net_num

def get_cellHpwl(file_content):
    try:
        # Define the regex pattern to match the specific string and capture the number
        pattern = r"\[INFO DPL-0022\] HPWL after\s+([0-9]+\.[0-9]+) u"
        
        # Search for the pattern in the file content
        match = re.search(pattern, file_content)
        
        if match:
            # Extract the matched number and convert it to float
            hpwl_value = float(match.group(1))
            return hpwl_value
        else:
            return None
    
    except re.error:
        return None

def get_usage(content):
    
    # Find the specific table
    table_start_pattern = r'\[INFO GRT-0096\] Final congestion report:'
    match = re.search(table_start_pattern, content)
    if not match:
        return None
    table_start_index = match.end()
    table_content = content[table_start_index:]
    # Find the Total line and extract the Usage percentage
    total_line_pattern = r'Total\s+\d+\s+\d+\s+(\d+\.\d+%)'
    total_match = re.search(total_line_pattern, table_content)
    if not total_match:
        raise ValueError("Total usage percentage not found in the table")
    total_usage = total_match.group(1)
    return float(total_usage.replace("%", "")) / 100

def get_RESIZE_wns_tns(content):
    tns_values = re.findall(r"tns\s+(-?\d+\.\d+)", content)
    wns_values = re.findall(r"wns\s+(-?\d+\.\d+)", content)
    if tns_values and wns_values:
        return wns_values[0], tns_values[0], wns_values[1], tns_values[1]
    return None, None, None, None


def get_grt_wl(content):
    
    # Find the specific table
    pattern = r"\[INFO GRT-0018\] Total wirelength: (\d+) um"
    match = re.search(pattern, content)
    if match:
        return int(match.group(1))
    return None

def get_total_overflow(content):
    table_start_pattern = r'\[INFO GRT-0096\] Final congestion report:'
    match = re.search(table_start_pattern, content)
    if not match:
        return None
    table_start_index = match.end()
    table_content = content[table_start_index:]

    total_line_pattern = r'Total\s+\d+\s+\d+\s+(\d+\.\d+%)\s+\d+\s+\/\s+\d+\s+\/\s+(\d+)'
    total_match = re.search(total_line_pattern, table_content)
    if not total_match:
        raise ValueError("Total overflow not found in the table")

    total_overflow = total_match.group(2)
    return int(total_overflow)

def load_Json(Json):
    try:
        with open(Json, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        
        return {}  
    
def load_file(input_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = file.read()
            return data
    except FileNotFoundError:
        
        return ""  

def get_wHPWL(log_content):
    """Extract wHPWL value from the last iteration of the log file"""
    lines = log_content.strip().split('\n')
    # Get the last non-empty line
    for line in reversed(lines):
        if line.strip() and 'wHPWL' in line:
            # Extract the value after 'wHPWL '
            parts = line.split('wHPWL')
            if len(parts) > 1:
                value_str = parts[1].split(',')[0].strip()
                return float(value_str)
    return None


def get_overflow(log_content):
    """Extract Overflow value from the last iteration of the log file"""
    lines = log_content.strip().split('\n')
    # Get the last non-empty line
    for line in reversed(lines):
        if line.strip() and 'Overflow' in line:
            # Extract the value after 'Overflow '
            parts = line.split('Overflow')
            if len(parts) > 1:
                value_str = parts[1].split(',')[0].strip()
                return float(value_str)
    return None


def get_Metric_in(finalJson, routeJson, placedpLog, gproutefile, fpjson, grouteJson, resizefile, upper_log, bottom_log, cost, key):

    if finalJson is None:
        upper_log = load_file(upper_log)
        bottom_log = load_file(bottom_log)
        metric = {}
        metric["Key"] = key
        metric["Cut_size"], metric["Area_imbalance"] = cost[0], cost[1]
        metric["U_wHPWL"] = get_wHPWL(upper_log)
        metric["B_wHPWL"] = get_wHPWL(bottom_log)
        metric["U_overflow"] = get_overflow(upper_log)
        metric["B_overflow"] = get_overflow(bottom_log)
        metric["total_wHPWL"] = metric["U_wHPWL"] + metric["B_wHPWL"]
        metric["total_overflow"] = metric["U_overflow"] + metric["B_overflow"]
        
        metric["PRE_RESIZE_WNS"], metric["PRE_RESIZE_TNS"], metric["POST_RESIZE_WNS"], metric["POST_RESIZE_TNS"] = None, None, None, None
        metric["Congestion"] = None
        metric["GRT_WL"] = None
        metric["GRT_WNS"], metric["GRT_TNS"] = None, None
        metric["DRT_WL"] = None
        metric["Final_WNS"], metric["Final_TNS"] = None, None
        metric["Power"] = None
        metric["#overflow"] = None
        metric["NVP"] = None
        
        metric["Die Area"] = None
        metric["Cell Area"] = None
        metric["#Macro"], metric["#Cells"], metric["Util"] = None, None, None
        metric["#Net"] = None
        
        return metric
    final=load_Json(finalJson)
    route=load_Json(routeJson)
    place=load_file(placedpLog)
    gproute=load_file(gproutefile)
    fp=load_Json(fpjson)
    groute=load_Json(grouteJson)
    resize=load_file(resizefile)
    upper_log=load_file(upper_log)
    bottom_log=load_file(bottom_log)

    metric = {}
    metric["Key"] = key
    metric["Cut_size"], metric["Area_imbalance"] = cost[0], cost[1]
    
    metric["U_wHPWL"] = get_wHPWL(upper_log)
    metric["B_wHPWL"] = get_wHPWL(bottom_log)
    metric["U_overflow"] = get_overflow(upper_log)
    metric["B_overflow"] = get_overflow(bottom_log)
    metric["total_wHPWL"] = metric["U_wHPWL"] + metric["B_wHPWL"]
    metric["total_overflow"] = metric["U_overflow"] + metric["B_overflow"]
    
    metric["PRE_RESIZE_WNS"], metric["PRE_RESIZE_TNS"], metric["POST_RESIZE_WNS"], metric["POST_RESIZE_TNS"] = get_RESIZE_wns_tns(resize)
    metric["Congestion"] = get_usage(gproute)
    
    metric["GRT_WL"] = get_grt_wl(gproute)
    metric["GRT_WNS"], metric["GRT_TNS"] = get_GRT_wns_tns(groute)
    metric["DRT_WL"] = get_wirelength(route)
    metric["Final_WNS"], metric["Final_TNS"] = get_wns_tns(final)
    metric["Power"] = get_power(final)
    metric["#overflow"] = get_total_overflow(gproute)
    metric["NVP"]=get_nvp(final)
    metric["Die Area"] = get_die_area(final)
    metric["Cell Area"] = get_area(final)
    metric["#Macro"], metric["#Cells"], metric["Util"] = get_statistics(final)
    metric["#Net"] = get_net_num(route)

    return metric

def get_substring_after_underscore(input_string):
    
    underscore_index = input_string.find('_')
    
    if underscore_index != -1:
        
        result = input_string[underscore_index + 1:]
        return result
    else:
        
        return input_string

def find_reports_in_dirs(base_dir):
    report_files = {}
    placement_reports = {}
    design_name = os.path.basename(base_dir)
    for solution_idx in os.listdir(base_dir):
        solution_root = os.path.join(base_dir, solution_idx)
        if os.path.isdir(solution_root):
            if os.path.exists(os.path.join(solution_root, 'openroad_logs')):
                report_files[f"{design_name}_{solution_idx}"] = {
                    "report": os.path.join(solution_root, 'openroad_logs', '6_report.json'),
                    "route": os.path.join(solution_root, 'openroad_logs', '5_2_route.json'),
                    "place": os.path.join(solution_root, 'openroad_logs', '3_5_place_dp.log'),
                    "gp_route": os.path.join(solution_root, 'openroad_logs', '5_1_grt.log'),
                    "fp": os.path.join(solution_root, 'openroad_logs', '2_1_floorplan.json'),
                    "groute": os.path.join(solution_root, 'openroad_logs', '5_1_grt.json'),
                    "resize": os.path.join(solution_root, 'openroad_logs', '3_4_place_resized.log')
                }
            else:
                report_files[f"{design_name}_{solution_idx}"] = {
                    'report': None,
                    'route': None,
                    'place': None,
                    'gp_route': None,
                    'fp': None,
                    'groute': None,
                    'resize': None
                }
            placement_reports[f"{design_name}_{solution_idx}"] = {
                "upper": os.path.join(solution_root, 'log_upper'),
                "bottom": os.path.join(solution_root, 'log_bottom_all')
            }
        elif os.path.isfile(solution_root) and solution_root.endswith('.json'):
            solution_log = solution_root
    return report_files, placement_reports, solution_log

def get_all_metric_json(report_files, placement_reports, solution_log):
    all_metric_json = {}
    solution_log = load_Json(solution_log)
    
    for case_name_idx, report_file in report_files.items():
        placement_report = placement_reports.get(case_name_idx)
        solution_idx = case_name_idx.split("_")[-1]
        key = list(solution_log["pareto_archive"]["solutions"].keys())[int(solution_idx)]
        cost = solution_log["pareto_archive"]["solutions"][key]["cost"] 
        metric = get_Metric_in(report_file["report"], report_file["route"], report_file["place"],report_file["gp_route"], report_file["fp"], report_file["groute"], report_file["resize"], placement_report["upper"], placement_report["bottom"], cost, key)
        all_metric_json[solution_idx] = metric

    return all_metric_json


def getMetrics(dir_path):

    report_files, placement_reports, solution_log = find_reports_in_dirs(dir_path)
    metric_dict=get_all_metric_json(report_files, placement_reports, solution_log)

    return metric_dict


def _to_python_scalar(value):
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and pd.isna(value):
        return None

    if pd.isna(value):
        return None

    return value


def cal_fitness_score(df, metrics):
    normalized_components = {}
    best_values = {}

    for metric in metrics:
        numeric_col = pd.to_numeric(df[metric], errors='coerce')
        col_min = numeric_col.min()
        col_max = numeric_col.max()

        if col_max <= 0:
            denom = abs(col_min)
            if denom != 0:
                normalized_series = numeric_col.abs() / denom
            else:
                # All non-NaN values are 0; set them to 0 but keep NaN as NaN
                normalized_series = numeric_col.where(numeric_col.isna(), 0.0)
            best_value = col_max
        elif col_min >= 0:
            denom = col_max
            if denom != 0:
                normalized_series = numeric_col / denom
            else:
                # All non-NaN values are 0; set them to 0 but keep NaN as NaN
                normalized_series = numeric_col.where(numeric_col.isna(), 0.0)
            best_value = col_min
        else:
            denom = max(abs(col_min), abs(col_max))
            if denom != 0:
                normalized_series = numeric_col.abs() / denom
            else:
                # All non-NaN values are 0; set them to 0 but keep NaN as NaN
                normalized_series = numeric_col.where(numeric_col.isna(), 0.0)
            abs_series = numeric_col.abs()
            best_index = abs_series.idxmin()
            best_value = numeric_col.loc[best_index]

        normalized_components[metric] = normalized_series.clip(lower=0.0, upper=1.0)
        best_values[metric] = _to_python_scalar(best_value)

    normalized_matrix = pd.DataFrame(normalized_components, index=df.index)
    # If any metric is NaN for a row, fitness will be NaN (sum propagates NaN)
    result_df = df.copy()
    # propagate NaN: if any metric is NaN for the row, fitness becomes NaN
    result_df["Fitness"] = np.sqrt((normalized_matrix ** 2).sum(axis=1, skipna=False))
    result_df = result_df.sort_values(by="Fitness", ascending=True, na_position='last').reset_index(drop=True)

    return result_df, best_values


if __name__ == "__main__":
    dir_path = os.path.join(os.path.dirname(__file__))
    dataset_name = "bp_quad"
    dir_path = os.path.join(dir_path, dataset_name)
    metric_dict = getMetrics(dir_path=dir_path)
    
    dataframes = []
    
    for solution_idx in metric_dict.keys():
        rows = []
        method_data = metric_dict[solution_idx]
        row = [solution_idx] + list(method_data.values())
        rows.append(row)
        
        df = pd.DataFrame(rows, columns=['Idx']+list(metric_dict[solution_idx].keys()))
        dataframes.append(df)
    
    final_df = pd.concat(dataframes, ignore_index=True)
    metrics_to_normalize = ["Congestion", "DRT_WL", "Final_WNS", "Final_TNS", "Power"]
    # metrics_to_normalize = ["DRT_WL"]
    
    # Create baseline row with baseline performance values
    # baseline_performance = [0.132, 3815933.0, -6.972, -9955.35, 1.048]
    # baseline_row = {col: None for col in final_df.columns}
    # baseline_row['Idx'] = 'baseline'
    # for i, metric in enumerate(metrics_to_normalize):
    #     if metric in final_df.columns:
    #         baseline_row[metric] = baseline_performance[i]
    
    # Convert baseline_row to DataFrame and append to final_df
    # baseline_df = pd.DataFrame([baseline_row])
    # final_df = pd.concat([final_df, baseline_df], ignore_index=True)
    
    final_df, best_values = cal_fitness_score(final_df, metrics_to_normalize)
    print(best_values)
    final_df.to_csv(f'{dataset_name}_final.csv', index=False)

    plot_path = os.path.join(dir_path, f"cutsize_imbalance.png")
    plot_pareto_front(
        final_df,
        x_col="Cut_size",
        y_col="Area_imbalance",
        fitness_col="Fitness",
        save_path=plot_path,
        # max_points=20,
    )