import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from scipy.spatial import distance


def load_hmsa_results_to_pareto_grid(hmsa_results_path: Path) -> Dict[Tuple[int, int], Tuple[Tuple[int, float], List[List[int]]]]:
    """
    Load HMSA results JSON file and convert to pareto_archive_grid format.
    
    Args:
        hmsa_results_path: Path to hmsa_results.json file
        
    Returns:
        Dictionary with keys (cell_x, cell_y) and values (cost, solution)
        where cost is (cutsize, imbalance) and solution is [lower_ids, upper_ids]
    """
    with open(hmsa_results_path, "r") as fp:
        solutions = json.load(fp)
    
    pareto_archive_grid = {}
    for key, value in solutions['pareto_archive']['solutions'].items():
        
        solution = value.get("solution", [[], []])
        cost = value.get("cost", [0.0, 0.0])
        
        # Convert solution to proper format
        lower_ids = [int(node_id) for node_id in solution[0]]
        upper_ids = [int(node_id) for node_id in solution[1]]
        solution = [lower_ids, upper_ids]
        
        # Convert cost to tuple (cutsize, imbalance)
        cut_size = int(cost[0]) if isinstance(cost[0], (int, float)) else 0
        imbalance = float(cost[1]) if isinstance(cost[1], (int, float)) else 0.0
        cost_tuple = (cut_size, imbalance)
        
        # Use index as grid key (since we don't have actual grid coordinates from JSON)
        # Format: (idx, 0) to match Tuple[int, int] requirement
        pareto_archive_grid[key] = (cost_tuple, solution)
    
    return pareto_archive_grid


def candidate_selection(pareto_archive_grid: Dict[Tuple[int, int], Tuple[Tuple[int, float], List[List[int]]]],
                        budget: int = 10,
                        front_ratio: float = 0.4,
                        side_percentile: float = 0.1) -> Dict[Tuple[int, int], Tuple[Tuple[int, float], List[List[int]]]]:
    """
    Selects a set of candidates from the grid archive by adaptively filtering the sides and sampling the knee and dominated cloud.
    """
    
    def _find_pareto_front(points_data: List[tuple]) -> List[tuple]:
        front = []
        for p1_data in points_data:
            is_dominated = False
            p1_cost = p1_data[1] # (cutsize, imbalance)
            
            for p2_data in points_data:
                if p1_data == p2_data: continue
                p2_cost = p2_data[1]
                if p1_cost[0] >= p2_cost[0] and p1_cost[1] >= p2_cost[1]:
                    is_dominated = True
                    break
            
            if not is_dominated:
                front.append(p1_data)
        return front
    
    all_data_points = []
    for grid_key, (cost, solution) in pareto_archive_grid.items():
        all_data_points.append((grid_key, cost, solution))
    
    if len(all_data_points) <= budget:
        print("Not enough data points to select candidates")
        return pareto_archive_grid
    
    points_with_norm = []
    costs = np.array([point[1] for point in all_data_points])
    normalized_costs = minmax_scale(costs, feature_range=(0, 1))
    
    for i, data in enumerate(all_data_points):
        points_with_norm.append((data[0], data[1], data[2], normalized_costs[i]))
    
    knee_points = []
    side_points = []
    
    for point in points_with_norm:
        norm_cost = point[3]
        is_side_point = False
        
        if norm_cost[0] <= side_percentile and norm_cost[1] >= (1 - side_percentile):
            is_side_point = True
        
        if norm_cost[1] <= side_percentile and norm_cost[0] >= (1 - side_percentile):
            is_side_point = True
        
        if is_side_point:
            side_points.append(point)
        else:
            knee_points.append(point)
        
    knee_front = _find_pareto_front(knee_points)
    knee_cloud = [point for point in knee_points if point not in knee_front]
    
    budget_front = int(budget * front_ratio)
    final_candidates_data = []
    
    # sample the front points
    if len(knee_front) > 0:
        front_norm_costs = np.array([point[3] for point in knee_front])
        n_front_clusters = min(budget_front, len(knee_front))
        
        kmeans_front = KMeans(n_clusters=n_front_clusters, random_state=0, n_init=10).fit(front_norm_costs)
        centroids_front = kmeans_front.cluster_centers_
        
        for i in range(n_front_clusters):
            cluster_points_indices = np.where(kmeans_front.labels_ == i)[0]
            if len(cluster_points_indices) == 0: continue
            
            centroid = centroids_front[i]
            distances = distance.cdist([centroid], front_norm_costs[cluster_points_indices])
            closest_point_idx = cluster_points_indices[distances.argmin()]
            final_candidates_data.append(knee_front[closest_point_idx])
    
    # sample the cloud points
    remaining_budget = budget - len(final_candidates_data)
    if len(knee_cloud) > 0 and remaining_budget > 0:
        cloud_norm_costs = np.array([point[3] for point in knee_cloud])
        n_cloud_clusters = min(remaining_budget, len(knee_cloud))
        
        kmeans_cloud = KMeans(n_clusters=n_cloud_clusters, random_state=0, n_init=10).fit(cloud_norm_costs)
        centroids_cloud = kmeans_cloud.cluster_centers_
        
        for i in range(n_cloud_clusters):
            cluster_points_indices = np.where(kmeans_cloud.labels_ == i)[0]
            if len(cluster_points_indices) == 0: continue
            
            centroid = centroids_cloud[i]
            distances = distance.cdist([centroid], cloud_norm_costs[cluster_points_indices])
            closest_point_idx = cluster_points_indices[distances.argmin()]
            final_candidates_data.append(knee_cloud[closest_point_idx])
    
    # Convert list to dictionary format matching pareto_archive_grid structure
    candidates_dict = {candidate[0]: (candidate[1], candidate[2]) for candidate in final_candidates_data}
    return candidates_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select candidates from HMSA results using K-means clustering.")
    parser.add_argument("hmsa_results", type=Path, help="Path to hmsa_results.json file from HMSA.py")
    parser.add_argument("--fitness-csv", type=Path, default=None, help="Path to final.csv file with fitness scores and ranks")
    parser.add_argument("--budget", type=int, default=10, help="Number of candidates to select (default: 10)")
    parser.add_argument("--front-ratio", type=float, default=0.4, help="Ratio of budget for front points (default: 0.4)")
    parser.add_argument("--side-percentile", type=float, default=0.1, help="Percentile threshold for side points (default: 0.1)")
    parser.add_argument("--output", type=Path, default=None, help="Path to save selected candidates JSON file")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    if not args.hmsa_results.exists():
        raise FileNotFoundError(f"HMSA results file not found: {args.hmsa_results}")
    
    logging.info(f"Loading HMSA results from {args.hmsa_results}...")
    pareto_archive_grid = load_hmsa_results_to_pareto_grid(args.hmsa_results)
    logging.info(f"Loaded {len(pareto_archive_grid)} candidates from HMSA results")
    
    logging.info(f"Running candidate selection with budget={args.budget}, front_ratio={args.front_ratio}, side_percentile={args.side_percentile}...")
    selected_candidates = candidate_selection(
        pareto_archive_grid,
        budget=args.budget,
        front_ratio=args.front_ratio,
        side_percentile=args.side_percentile
    )
    
    logging.info(f"Selected {len(selected_candidates)} candidates")
    
    # output the selected candidates with fitness and rank
    output_data = []
    df = pd.read_csv(args.fitness_csv)
    df = df.set_index("Key")
    for grid_key in selected_candidates.keys():   
        row = df.loc[grid_key]
        fitness = float(row["Fitness"])
        row_number = df.index.get_loc(grid_key)
        output_data.append((grid_key, fitness, row_number))
    print(output_data)

if __name__ == "__main__":
    main()