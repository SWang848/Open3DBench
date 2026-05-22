from __future__ import annotations

import argparse
import os
import sys
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import scipy.linalg as la
from sklearn.preprocessing import PolynomialFeatures

from algorithms.dopp._place3d_bridge import REPO_ROOT, Params, PlaceDB
from algorithms.dopp.hierarchy_multi_objective_sa import graph_construction
from algorithms.dopp.loaders import load_candidates_from_json


class ManualFeatureConstructor:
    """
    Construct manual features for candidate partitions.

    This class keeps the same feature definitions as the previous functional
    implementation, but encapsulates graph statistics and build steps in one
    object similar to ``GraphDiffusedFeatureConstructor``.
    """

    def __init__(
        self,
        placedb: PlaceDB.PlaceDB,
        candidates: List[Tuple[str, List[List[int]], Tuple[float, float]]],
    ) -> None:
        self.placedb = placedb
        self.candidates = candidates
        self.graph = graph_construction(placedb)
        self.num_nets = self.graph.number_of_edges()
        self.total_macros = sum(
            1 for node in self.graph.nodes() if self.graph.nodes[node].get("is_macro", False)
        )
        self.total_area = sum(self.graph.nodes[node].get("area", 0.0) for node in self.graph.nodes())

    @staticmethod
    def compute_global_metrics(
        partition: List[List[int]],
        cut_size: float,
        area_imbalance: float,
        num_nets: int,
        total_macros: int,
        total_area: float,
    ) -> Tuple[float, float, float]:
        """Compute global partition metrics: cut, area imbalance, macro-count imbalance."""
        lower_ids, upper_ids = partition[0], partition[1]
        upper_macro_count = len(upper_ids)
        lower_macro_count = len(lower_ids)
        normalized_cut_size = cut_size / num_nets if num_nets > 0 else 0.0
        normalized_area_imbalance = area_imbalance / total_area if total_area > 0 else 0.0
        macro_count_imbalance = abs(upper_macro_count - lower_macro_count) / total_macros
        return normalized_cut_size, normalized_area_imbalance, macro_count_imbalance

    def compute_cut_degree(self, partition: List[List[int]]) -> Tuple[float, float, float, float]:
        """Compute min/max/mean/std cut degree across macro nodes."""
        upper_die_set = set(partition[1])
        cut_degrees = []

        for node in self.graph.nodes():
            if not self.graph.nodes[node].get("is_macro", False):
                continue

            macro_in_upper = node in upper_die_set
            cut_degree = 0.0
            for neighbor in self.graph.neighbors(node):
                neighbor_in_upper = neighbor in upper_die_set
                if macro_in_upper != neighbor_in_upper:
                    cut_degree += self.graph.number_of_edges(node, neighbor)
            cut_degrees.append(cut_degree)

        if not cut_degrees:
            return 0.0, 0.0, 0.0, 0.0

        cut_degrees_array = np.array(cut_degrees)
        return (
            float(np.min(cut_degrees_array)),
            float(np.max(cut_degrees_array)),
            float(np.mean(cut_degrees_array)),
            float(np.std(cut_degrees_array)),
        )

    def compute_hierarchy_features(self, partition: List[List[int]]) -> Tuple[float, ...]:
        """Compute one hierarchy cohesion score per hierarchy cluster."""
        hierarchy_numstdnode_dict = {}
        hierarchy_macronode_dict = {}
        for node in self.graph.nodes():
            name = self.graph.nodes[node].get("name")
            hierarchy_level = name.split("__")
            depth = len(hierarchy_level)
            for idx in range(depth - 1):
                if idx == 0:
                    hierarchy_name = hierarchy_level[idx]
                else:
                    hierarchy_name = hierarchy_name + "__" + hierarchy_level[idx]

                if not self.graph.nodes[node].get("is_macro"):
                    hierarchy_numstdnode_dict[hierarchy_name] = (
                        hierarchy_numstdnode_dict.get(hierarchy_name, 0) + 1
                    )
                else:
                    hierarchy_macronode_dict.setdefault(hierarchy_name, []).append(node)

        hierarchy_features = []
        lower_die_set = set(partition[0]) if partition[0] else set()

        for key, macro_nodes in hierarchy_macronode_dict.items():
            num_stdnodes = hierarchy_numstdnode_dict.get(key, 0)
            num_macro = len(macro_nodes)
            num_macro_in_lower = sum(1 for node in macro_nodes if node in lower_die_set)
            num_macro_in_upper = num_macro - num_macro_in_lower

            if num_stdnodes == 0 and (num_macro_in_lower == 0 or num_macro_in_upper == 0):
                cohesion_score = 1.0
            elif num_stdnodes == 0 and (num_macro_in_lower > 0 and num_macro_in_upper > 0):
                cohesion_score = 1.0 - (
                    (num_macro_in_lower * num_macro_in_upper) / math.comb(num_macro, 2)
                )
            else:
                cohesion_score = (
                    num_stdnodes * num_macro_in_lower + math.comb(num_macro_in_lower, 2)
                ) / (
                    num_stdnodes * num_macro + math.comb(num_macro, 2)
                )
            hierarchy_features.append(float(cohesion_score))

        return tuple(hierarchy_features)

    def extract_manual_features(self) -> Dict[str, np.ndarray]:
        """Extract manual features for all candidates owned by this constructor."""
        logging.info("Total number of nets (edges in graph): %s", self.num_nets)
        logging.info("Total number of macros: %s", self.total_macros)
        logging.info("Total area: %s", self.total_area)
        features_dict: Dict[str, np.ndarray] = {}

        logging.info("Computing features for %d candidates...", len(self.candidates))
        for key, partition, cost in self.candidates:
            cut_size, area_imbalance = cost[0], cost[1]
            f0, f1, f2 = self.compute_global_metrics(
                partition,
                cut_size,
                area_imbalance,
                self.num_nets,
                self.total_macros,
                self.total_area,
            )
            f3, f4, f5, f6 = self.compute_cut_degree(partition)
            hierarchy_features = self.compute_hierarchy_features(partition)
            features_dict[key] = np.array(
                [f0, f1, f2, f3, f4, f5, f6] + list(hierarchy_features),
                dtype=np.float32,
            )
        logging.info("Extracted features for %d candidates", len(features_dict))
        return features_dict

    @staticmethod
    def apply_polynomial_features(
        features_dict: Dict[str, np.ndarray],
        degree: int = 2,
        include_bias: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], PolynomialFeatures]:
        """Apply polynomial expansion to manual feature vectors."""
        feature_keys = list(features_dict.keys())
        feature_matrix = np.array([features_dict[key] for key in feature_keys])
        poly_transformer = PolynomialFeatures(
            degree=degree,
            include_bias=include_bias,
            interaction_only=False,
        )
        polynomial_features = poly_transformer.fit_transform(feature_matrix)

        logging.info("  Polynomial feature shape: %s", polynomial_features.shape)
        logging.info("  Number of original features: %s", feature_matrix.shape[1])
        logging.info("  Number of polynomial features: %s", polynomial_features.shape[1])

        polynomial_features_dict = {
            key: polynomial_features[i] for i, key in enumerate(feature_keys)
        }
        return polynomial_features_dict, poly_transformer

    def build_manual_features(self) -> Dict[str, np.ndarray]:
        return self.extract_manual_features()

    def build_feature_bundle(
        self,
        polynomial_features: bool = False,
        polynomial_degree: int = 2,
        include_bias: bool = False,
    ) -> Dict:
        manual_features_dict = self.build_manual_features()
        candidate_keys = list(manual_features_dict.keys())
        if not candidate_keys:
            raise ValueError("No candidates found in HMSA results file.")

        len_hierarchy_features = len(manual_features_dict[candidate_keys[0]]) - 7
        manual_feature_names = [
            "f0_normalized_cut_size",
            "f1_normalized_area_imbalance",
            "f2_macro_count_imbalance",
            "f3_min_cut_degree",
            "f4_max_cut_degree",
            "f5_mean_cut_degree",
            "f6_std_cut_degree",
        ]
        manual_feature_names += [f"hierarchy_cohesion_{idx}" for idx in range(len_hierarchy_features)]
        manual_features_matrix = np.array([manual_features_dict[key] for key in candidate_keys])

        if polynomial_features:
            polynomial_features_dict, poly_transformer = self.apply_polynomial_features(
                manual_features_dict,
                degree=polynomial_degree,
                include_bias=include_bias,
            )
            features_dict = polynomial_features_dict
            feature_names = np.array(poly_transformer.get_feature_names_out(manual_feature_names))
        else:
            features_dict = manual_features_dict
            feature_names = np.array(manual_feature_names)

        features_matrix = np.array([features_dict[key] for key in candidate_keys])
        _, r_matrix, piv = la.qr(features_matrix, mode="economic", pivoting=True)
        tol = 1e-10
        rank = np.sum(np.abs(np.diag(r_matrix)) > tol)
        independent_columns = np.sort(piv[:rank])
        dependent_columns = np.sort(piv[rank:])

        if len(dependent_columns) > 0:
            dependent_feature_names = feature_names[dependent_columns]
            logging.info(
                "Dropped %d linearly dependent columns: %s",
                len(dependent_columns),
                dependent_columns,
            )
            logging.info("Dropped feature names: %s", dependent_feature_names.tolist())
        else:
            logging.info("No linearly dependent columns found - matrix is full rank")

        features_matrix = features_matrix[:, independent_columns]
        original_rank = np.linalg.matrix_rank(manual_features_matrix)
        current_rank = np.linalg.matrix_rank(features_matrix)
        feature_type_label = "polynomial" if polynomial_features else "manual"
        logging.info(
            "Manual feature matrix rank/dimension: %d/%d, Current feature (%s) matrix rank/dimension: %d/%d",
            original_rank,
            manual_features_matrix.shape[1],
            feature_type_label,
            current_rank,
            features_matrix.shape[1],
        )

        return {
            "feature_type": "manual_polynomial" if polynomial_features else "manual",
            "candidate_keys": candidate_keys,
            "features": features_matrix,
            "feature_dim": int(features_matrix.shape[1]),
            "metadata": {
                "base_feature_type": "manual",
                "variant": "polynomial" if polynomial_features else "raw",
                "manual_feature_dim": len(manual_feature_names),
                "polynomial_features": bool(polynomial_features),
                "polynomial_degree": polynomial_degree,
                "polynomial_include_bias": bool(include_bias),
                "current_feature_dim": int(features_matrix.shape[1]),
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract manual features for D-optimal design.")
    parser.add_argument("params", type=Path, help="Path to params JSON used by PlaceDB.")
    parser.add_argument("hmsa_results", type=Path, help="Path to hmsa_results.json containing candidates.")
    parser.add_argument("--output", type=Path, help="Path to save extracted features. Default: evaluation/regression_results/{case_name}")
    parser.add_argument("--polynomial-features", action="store_true", help="Apply polynomial features")
    parser.add_argument("--polynomial-degree", type=int, default=2, help="Degree of polynomial features (default: 2)")
    parser.add_argument("--include-bias", action="store_true", help="Include bias (intercept) term in polynomial features")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    case_name = args.params.stem
    if args.output is not None:
        out_dir = args.output
    else:
        out_dir = REPO_ROOT / "evaluation" / "regression_results" / case_name
    os.makedirs(out_dir, exist_ok=True)
    
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
    
    constructor = ManualFeatureConstructor(
        placedb=placedb,
        candidates=candidates,
    )
    output_data = constructor.build_feature_bundle(
        polynomial_features=args.polynomial_features,
        polynomial_degree=args.polynomial_degree,
        include_bias=args.include_bias,
    )
    
    np.save(os.path.join(out_dir, "manual_features.npy"), output_data, allow_pickle=True)
    logging.info("Saved manual feature bundle: shape=%s", output_data["features"].shape)


if __name__ == "__main__":
    main()

