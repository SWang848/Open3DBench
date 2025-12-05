from __future__ import annotations

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import scipy.linalg as la
from sklearn.preprocessing import PolynomialFeatures

from HMSA import graph_construction

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from dreamplace import Params, PlaceDB


def load_candidates_from_json(json_path: Path) -> List[Tuple[str, List[List[int]], Tuple[float, float]]]:
    """
    Load candidate solutions from HMSA results JSON file.
    Returns list of (key, solution, cost) tuples.
    """
    with open(json_path, "r") as fp:
        data = json.load(fp)
    
    candidates = []
    for key, entry in data["pareto_archive"]["solutions"].items():
        raw_solution = entry.get("solution", [[], []])
        cost = entry.get("cost", [0.0, 0.0])
        
        lower_ids = [int(node_id) for node_id in raw_solution[0]]
        upper_ids = [int(node_id) for node_id in raw_solution[1]]
        cut_size = float(cost[0])
        area_imbalance = float(cost[1])
        
        candidates.append((key, [lower_ids, upper_ids], (cut_size, area_imbalance)))
    
    return candidates


def compute_global_metrics(
    partition: List[List[int]],
    cut_size: float,
    area_imbalance: float,
    num_nets: int,
    total_macros: int,
    total_area: float,
) -> Tuple[float, float, float]:
    """
    Compute the 4 basic global metrics:
    - F0: Normalized cut size = cut_size / num_nets
    - F1: Normalized area imbalance = |A1 - A2| / (A1 + A2) (already computed in JSON)
    - F2: Macro-count imbalance = |M1 - M2| / (M1 + M2)
    
    Args:
        partition: [lower_die_node_ids, upper_die_node_ids]
        cut_size: Cut size from JSON (cost[0])
        area_imbalance: Area imbalance from JSON (cost[1])
        num_nets: Total number of nets
        total_macros: Total number of macros
    
    Returns:
        Tuple of (f0, f1, f2)
    """
    lower_ids, upper_ids = partition[0], partition[1]
    
    # Compute macro counts per tier
    M1 = len(upper_ids) # Number of macros in tier 1 (upper die)
    M2 = len(lower_ids) # Number of macros in tier 2 (lower die)

    # F0: Normalized cut size
    f0 = cut_size / num_nets if num_nets > 0 else 0.0
    # F1: Normalized area imbalance = |A1 - A2| / (A1 + A2)
    f1 = area_imbalance / total_area
    # F2: Macro-count imbalance
    f2 = abs(M1 - M2) / total_macros
    
    return f0, f1, f2


def compute_cut_degree(
    G: nx.Graph,
    partition: List[List[int]],
) -> Tuple[float, float, float, float]:
    """
    Compute cut degree statistics for macros.
    Cut degree of a macro A is the number of edges incident to macro A that cross tiers.
    
    
    Args:
        G: Graph with node attributes (is_macro)
        partition: [lower_die_node_ids, upper_die_node_ids]
    
    Returns:
        f3: Minimum cut degree
        f4: Maximum cut degree
        f5: Mean cut degree
        f6: Standard deviation of cut degree
    """
    lower_ids, upper_ids = partition[0], partition[1]
    upper_die_set = set(upper_ids)
    cut_degrees = []
    
    # Iterate through all macros
    for node in G.nodes():
        if not G.nodes[node].get("is_macro", False):
            continue
        
        # Determine which tier this macro is in
        macro_in_upper = node in upper_die_set
        # Count edges to nodes in the other tier
        cut_degree = 0.0
        for neighbor in G.neighbors(node):
            neighbor_in_upper = neighbor in upper_die_set
            
            # If neighbor is in different tier, this edge crosses tiers
            if macro_in_upper != neighbor_in_upper:
                # Count number of edges between these nodes (MultiGraph can have parallel edges)
                num_edges = G.number_of_edges(node, neighbor)
                cut_degree += num_edges
        
        cut_degrees.append(cut_degree)
    
    if len(cut_degrees) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    cut_degrees_array = np.array(cut_degrees)
    # F3: Minimum cut degree    
    f3 = float(np.min(cut_degrees_array))
    # F4: Maximum cut degree
    f4 = float(np.max(cut_degrees_array))
    # F5: Mean cut degree
    f5 = float(np.mean(cut_degrees_array))
    # F6: Standard deviation of cut degree
    f6 = float(np.std(cut_degrees_array))
    
    return f3, f4, f5, f6


def extract_manual_features(
    candidates: List[Tuple[str, List[List[int]], Tuple[float, float]]],
    placedb: PlaceDB.PlaceDB,
) -> Dict[str, np.ndarray]:
    """
    Extract manual features for all candidate solutions.
    Returns a dictionary mapping candidate keys to feature vectors.
    """
    logging.info("Building graph from PlaceDB...")
    G = graph_construction(placedb)
    
    # Count total number of edges in the graph
    num_nets = G.number_of_edges()
    logging.info(f"Total number of nets (edges in graph): {num_nets}")
    
    # Count total macros
    total_macros = sum(1 for node in G.nodes() if G.nodes[node].get("is_macro", False))
    logging.info(f"Total number of macros: {total_macros}")
    
    # Count total area
    total_area = sum(G.nodes[node].get("area", 0.0) for node in G.nodes())
    logging.info(f"Total area: {total_area}")
    features_dict = {}
    
    logging.info(f"Computing features for {len(candidates)} candidates...")
    for key, partition, cost in candidates:
        cut_size, area_imbalance = cost[0], cost[1]
        f0, f1, f2 = compute_global_metrics(partition, cut_size, area_imbalance, num_nets, total_macros, total_area)
        # Compute cut degree features
        f3, f4, f5, f6 = compute_cut_degree(G, partition)
        # Combine all features: [f0, f1, f2, f3, f4, f5, f6]
        features_dict[key] = np.array([f0, f1, f2, f3, f4, f5, f6])
    logging.info(f"Extracted features for {len(features_dict)} candidates")
    
    return features_dict


def apply_polynomial_features(
    features_dict: Dict[str, np.ndarray],
    degree: int = 2,
    include_bias: bool = False,
) -> Tuple[Dict[str, np.ndarray], PolynomialFeatures]:
    """
    Apply polynomial features to the extracted manual features.
    
    Args:
        features_dict: Dictionary mapping candidate keys to feature vectors
        degree: The degree of the polynomial features (default: 2)
        include_bias: If True, include a bias (intercept) term (default: False)
    
    Returns:
        Tuple of (polynomial_features_dict, fitted_polynomial_transformer)
    """
    # Extract features in the same order as keys
    feature_keys = list(features_dict.keys())
    feature_matrix = np.array([features_dict[key] for key in feature_keys])
    # Create and fit polynomial feature transformer
    poly_transformer = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=False)
    polynomial_features = poly_transformer.fit_transform(feature_matrix)
    
    logging.info(f"  Polynomial feature shape: {polynomial_features.shape}")
    logging.info(f"  Number of original features: {feature_matrix.shape[1]}")
    logging.info(f"  Number of polynomial features: {polynomial_features.shape[1]}")
    
    # Create dictionary with polynomial features
    polynomial_features_dict = {key: polynomial_features[i] for i, key in enumerate(feature_keys)}
    
    return polynomial_features_dict, poly_transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract manual features for D-optimal design.")
    parser.add_argument("params", type=Path, help="Path to params JSON used by PlaceDB.")
    parser.add_argument("hmsa_results", type=Path, help="Path to hmsa_results.json containing candidates.")
    parser.add_argument("--output", type=Path, default=None, help="Path to save extracted features. Default: regression_results/{case_name}/manual_features.npy")
    parser.add_argument("--polynomial-degree", type=int, default=2, help="Degree of polynomial features (default: 2)")
    parser.add_argument("--include-bias", action="store_true", help="Include bias (intercept) term in polynomial features")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    case_name = args.params.stem
    out_dir = Path("./regression_results") / case_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = args.output or (out_dir / "manual_features.npy")
    
    if not args.hmsa_results.exists():
        raise FileNotFoundError(f"HMSA results file not found: {args.hmsa_results}")
    
    # Load PlaceDB
    logging.info("Loading PlaceDB...")
    params = Params.Params()
    params.load(str(args.params))
    params.placed_def_input = ""
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)
    
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    
    # Load candidates
    logging.info(f"Loading candidates from {args.hmsa_results}...")
    candidates = load_candidates_from_json(args.hmsa_results)
    logging.info(f"Loaded {len(candidates)} candidates")
    
    if len(candidates) == 0:
        raise ValueError("No candidates found in HMSA results file.")
    
    # Extract manual features
    manual_features_dict = extract_manual_features(candidates, placedb)
    
    # Apply polynomial features
    polynomial_features_dict, poly_transformer = apply_polynomial_features(
        manual_features_dict,
        degree=args.polynomial_degree,
        include_bias=args.include_bias,
    )
    
    # Save features (both original and polynomial)
    original_feature_names = [
        "f0_normalized_cut_size", "f1_normalized_area_imbalance", "f2_macro_count_imbalance", 
        "f3_min_cut_degree", "f4_max_cut_degree", "f5_mean_cut_degree", "f6_std_cut_degree"
    ]

    # Get polynomial feature names from transformer
    polynomial_feature_names = poly_transformer.get_feature_names_out(original_feature_names)
    
    # Convert to numpy arrays
    candidate_keys = list(polynomial_features_dict.keys())
    original_features_matrix = np.array([manual_features_dict[key] for key in candidate_keys])
    polynomial_features_matrix = np.array([polynomial_features_dict[key] for key in candidate_keys])
    
    # Run QR decomposition to identify linearly dependent columns
    Q, R, piv = la.qr(polynomial_features_matrix, mode="economic", pivoting=True)
    tol = 1e-10
    rank = np.sum(np.abs(np.diag(R)) > tol)
    
    independent_columns = np.sort(piv[:rank])  # Sort to preserve original column order
    dependent_columns = np.sort(piv[rank:])      # Sort for consistent logging
    
    # Log information about dropped columns (before dropping)
    if len(dependent_columns) > 0:
        dependent_feature_names = polynomial_feature_names[dependent_columns]
        logging.info(f"Dropped {len(dependent_columns)} linearly dependent columns: {dependent_columns}")
        logging.info(f"Dropped feature names: {dependent_feature_names.tolist()}")
    else:
        logging.info("No linearly dependent columns found - matrix is full rank")
    
    # Drop dependent columns from matrix and feature names (preserving original order)
    polynomial_features_matrix = polynomial_features_matrix[:, independent_columns]
    
    
    
    polynomial_feature_names = polynomial_feature_names[independent_columns]
    
    # Compute and log rank (for viewing only, not saved)
    original_rank = np.linalg.matrix_rank(original_features_matrix)
    polynomial_rank = np.linalg.matrix_rank(polynomial_features_matrix)
    logging.info(f"Feature matrix ranks: original={original_rank}/{original_features_matrix.shape[1]}, "
                 f"polynomial={polynomial_rank}/{polynomial_features_matrix.shape[1]}")
    
    output_data = {
        "candidate_keys": candidate_keys,
        "original_features": original_features_matrix,
        "polynomial_features": polynomial_features_matrix,
        "original_feature_names": original_feature_names,
        "polynomial_feature_names": polynomial_feature_names.tolist(),
        "original_feature_dim": len(original_feature_names),
        "polynomial_feature_dim": polynomial_features_matrix.shape[1],
        "polynomial_degree": args.polynomial_degree,
        "include_bias": args.include_bias,
    }
    
    np.save(output_path, output_data, allow_pickle=True)
    
    # Print sample features (original)
    if len(manual_features_dict) > 0:
        for i, (key, features) in enumerate(list(manual_features_dict.items())[:5]):
            logging.info(f"  Sample original features for '{key}': "
                        f"f0={features[0]:.2f}, f1={features[1]:.2f}, f2={features[2]:.2f}, f3={features[3]:.2f}, f4={features[4]:.2f}, f5={features[5]:.2f}, f6={features[6]:.2f}")

        # Print sample polynomial features (first few)
        for i, (key, features) in enumerate(list(polynomial_features_dict.items())[:5]):
            logging.info(f"  Sample polynomial features for '{key}': {features}")


if __name__ == "__main__":
    main()

