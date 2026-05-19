from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


class GraphDiffusedFeatureConstructor:
    """
    Construct graph-diffused features from partition node features.

    Given partition node features ``X_p`` and weighted adjacency ``A``, this
    class builds ``S = D^{-1/2} (A + I) D^{-1/2}`` and returns flattened hop
    features ``[X_p, SX_p, S^2X_p]`` by default.

    ``partition_node_features`` may be either ``[n_nodes, feature_dim]`` for
    one candidate or ``[n_candidates, n_nodes, feature_dim]`` for a batch.
    """

    def __init__(
        self,
        partition_node_features: np.ndarray,
        edges: np.ndarray,
        edge_weights: Optional[np.ndarray] = None,
        self_loop_weight: float = 1.0,
    ) -> None:
        self.partition_node_features = self._validate_partition_node_features(
            partition_node_features
        )
        self.num_nodes = self.partition_node_features.shape[-2]
        self.edges = self._validate_edges(edges)
        self._validate_edge_indices(self.edges, self.num_nodes)
        self.edge_weights = self._validate_edge_weights(edge_weights, self.edges.shape[0])
        self.self_loop_weight = float(self_loop_weight)
        if self.self_loop_weight < 0.0:
            raise ValueError(
                f"self_loop_weight must be non-negative, got {self.self_loop_weight}"
            )

        self.normalized_adjacency = self.build_normalized_adjacency()

    def build_normalized_adjacency(self) -> np.ndarray:
        """
        Build ``S = D^{-1/2} (A + I) D^{-1/2}`` as a dense float32 matrix.

        ``GraphBuilder`` stores each undirected edge once, so the weighted
        adjacency is filled symmetrically before adding self-loops.
        """
        n_nodes = self.num_nodes
        adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)

        for edge_idx, (src_idx, dst_idx) in enumerate(self.edges):
            src_idx = int(src_idx)
            dst_idx = int(dst_idx)
            weight = float(self.edge_weights[edge_idx])
            adjacency[src_idx, dst_idx] += weight
            adjacency[dst_idx, src_idx] += weight

        if self.self_loop_weight != 0.0:
            adjacency += np.eye(n_nodes, dtype=np.float32) * self.self_loop_weight

        degree = adjacency.sum(axis=1)
        inv_sqrt_degree = np.zeros_like(degree, dtype=np.float32)
        nonzero_degree = degree > 0.0
        inv_sqrt_degree[nonzero_degree] = 1.0 / np.sqrt(degree[nonzero_degree])

        return (
            inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
        ).astype(np.float32, copy=False)

    def build_hop_features(self, max_hop: int = 2) -> List[np.ndarray]:
        """
        Return ``[X_p, SX_p, ..., S^max_hop X_p]``.

        For ``max_hop=2`` this captures each node's own features, weighted
        1-hop neighborhood context, and weighted 2-hop neighborhood context.
        """
        if max_hop < 0:
            raise ValueError(f"max_hop must be non-negative, got {max_hop}")

        hop_features = [self.partition_node_features]
        current_features = self.partition_node_features
        for _ in range(max_hop):
            current_features = self._diffuse_once(current_features)
            hop_features.append(current_features.astype(np.float32, copy=False))

        return hop_features

    def build_flattened_features(self, max_hop: int = 2) -> np.ndarray:
        """
        Concatenate hop features and flatten node features per candidate.

        Returns ``[n_nodes * (max_hop + 1) * d]`` for one candidate or
        ``[n_candidates, n_nodes * (max_hop + 1) * d]`` for a batch.
        """
        flattened_features = np.concatenate(
            self.build_hop_features(max_hop=max_hop),
            axis=-1,
        )
        if flattened_features.ndim == 2:
            reshaped_features = flattened_features.reshape(-1)
        else:
            reshaped_features = flattened_features.reshape(flattened_features.shape[0], -1)
        return reshaped_features.astype(np.float32, copy=False)

    def build_graph_feature_vector(self, max_hop: int = 2) -> np.ndarray:
        """
        Flatten all node-level diffused features into one graph-level vector.
        """
        return self.build_flattened_features(max_hop=max_hop)

    @staticmethod
    def _validate_partition_node_features(
        partition_node_features: np.ndarray,
    ) -> np.ndarray:
        features = np.asarray(partition_node_features, dtype=np.float32)
        if features.ndim not in (2, 3):
            raise ValueError(
                "partition_node_features must have shape "
                f"[n_nodes, feature_dim] or [n_candidates, n_nodes, feature_dim], "
                f"got {features.shape}"
            )
        if features.shape[-2] <= 0:
            raise ValueError("partition_node_features must contain at least one node")
        return features

    def _diffuse_once(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 2:
            return self.normalized_adjacency @ features
        return np.stack(
            [
                self.normalized_adjacency @ candidate_features
                for candidate_features in features
            ],
            axis=0,
        )

    @staticmethod
    def _validate_edges(edges: np.ndarray) -> np.ndarray:
        edge_array = np.asarray(edges, dtype=np.int64)
        if edge_array.size == 0:
            return edge_array.reshape(0, 2)
        if edge_array.ndim != 2 or edge_array.shape[1] != 2:
            raise ValueError(f"edges must have shape [n_edges, 2], got {edge_array.shape}")
        return edge_array

    @staticmethod
    def _validate_edge_weights(
        edge_weights: Optional[Sequence[float]],
        n_edges: int,
    ) -> np.ndarray:
        if edge_weights is None:
            return np.ones(n_edges, dtype=np.float32)

        weights = np.asarray(edge_weights, dtype=np.float32)
        if weights.shape != (n_edges,):
            raise ValueError(f"edge_weights must have shape [{n_edges}], got {weights.shape}")
        if np.any(weights < 0.0):
            raise ValueError("edge_weights must be non-negative")
        return weights

    @staticmethod
    def _validate_edge_indices(edges: np.ndarray, n_nodes: int) -> None:
        if edges.size == 0:
            return
        if np.any(edges < 0) or np.any(edges >= n_nodes):
            raise ValueError(
                f"edges contain node indices outside valid range [0, {n_nodes - 1}]"
            )


def build_graph_diffused_feature_bundle(
    benchmark: str,
    candidates_path: Path,
    max_hop: int = 2,
    def_path: Optional[Path] = None,
    upper_die_macros: Optional[str] = None,
    partition_result: Optional[str] = None,
    scale_factor: float = 1.0,
    top_k_ratio: float = 0.3,
    rand_init: bool = False,
    self_loop_weight: float = 1.0,
) -> Dict:
    """
    Build a standardized graph-diffused feature bundle.

    Returns a dictionary with the same top-level keys as manual feature
    construction so downstream D-optimal and regression code can share one
    loading path.
    """
    from algorithms.dopp.dmp_loader import DreamPlaceLoader
    from algorithms.dopp.feature_construction_manual import load_candidates_from_json
    from algorithms.dopp.graph_builder import GraphBuilder
    from algorithms.dopp.partition_graph_updater import PartitionGraphUpdater
    from algorithms.dopp.utils import _parse_partition_result, _parse_upper_die_macros

    candidates = load_candidates_from_json(candidates_path)
    candidate_keys = [key for key, _, _ in candidates]
    if not candidate_keys:
        raise ValueError("No candidates found in candidates JSON.")

    partition_params = {
        "scale_factor": scale_factor,
        "top_k_ratio": top_k_ratio,
    }
    loader = DreamPlaceLoader(
        benchmark=benchmark,
        upper_die_names=_parse_upper_die_macros(upper_die_macros),
        partition_result=_parse_partition_result(partition_result),
        def_path=str(def_path) if def_path is not None else None,
        rand_init=rand_init,
    )
    graph_builder = GraphBuilder(
        partition_params=partition_params,
        dreamplace_loader=loader,
    )
    updater = PartitionGraphUpdater(
        graph_builder=graph_builder,
        candidates_path=candidates_path,
    )
    partition_node_features = updater.build_all_partition_features()
    if partition_node_features.shape[0] != len(candidate_keys):
        raise ValueError(
            "Candidate key count does not match partition feature rows: "
            f"{len(candidate_keys)} != {partition_node_features.shape[0]}"
        )

    constructor = GraphDiffusedFeatureConstructor(
        partition_node_features=partition_node_features,
        edges=graph_builder.edges,
        edge_weights=graph_builder.edge_weights,
        self_loop_weight=self_loop_weight,
    )
    features = constructor.build_flattened_features(max_hop=max_hop)
    if features.ndim == 1:
        features = features[None, :]

    return {
        "feature_type": "graph_diffused",
        "candidate_keys": candidate_keys,
        "features": features.astype(np.float32, copy=False),
        "feature_dim": int(features.shape[1]),
        "metadata": {
            "benchmark": benchmark,
            "max_hop": max_hop,
            "self_loop_weight": self_loop_weight,
            "num_graph_nodes": int(graph_builder.node_features.shape[0]),
            "num_graph_edges": int(len(graph_builder.edge_weights)),
            "partition_feature_shape": list(partition_node_features.shape),
        },
    }
