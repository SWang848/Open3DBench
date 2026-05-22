from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA


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

        self.last_pca_explained_variance_ratio_: Optional[np.ndarray] = None
        self.last_pca_eigenvalues_: Optional[np.ndarray] = None
        self.last_full_pca_explained_variance_ratio_: Optional[np.ndarray] = None
        self.last_full_pca_eigenvalues_: Optional[np.ndarray] = None
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

    def compress_feature_dimension(
        self,
        features: np.ndarray,
        n_components: int,
    ) -> np.ndarray:
        """
        Compress a feature matrix with PCA.

        Args:
            features: ``[n_samples, feature_dim]`` or ``[feature_dim]``.
            n_components: PCA target output dimension.

        Returns:
            Compressed features with shape ``[n_samples, n_components]`` or
            ``[n_components]`` for one sample input.
        """
        features_array = np.asarray(features, dtype=np.float32)
        if features_array.ndim == 1:
            features_2d = features_array[None, :]
            squeeze_output = True
        elif features_array.ndim == 2:
            features_2d = features_array
            squeeze_output = False
        else:
            raise ValueError(
                "features must have shape [feature_dim] or [n_samples, feature_dim], "
                f"got {features_array.shape}"
            )

        max_components = min(features_2d.shape[0], features_2d.shape[1])
        if n_components > max_components:
            raise ValueError(
                "n_components must be <= min(n_samples, feature_dim): "
                f"{n_components} > {max_components}"
            )

        pca = PCA(n_components=n_components)
        compressed_features = pca.fit_transform(features_2d).astype(np.float32, copy=False)
        self.last_pca_eigenvalues_ = pca.explained_variance_.astype(np.float32, copy=False)
        self.last_pca_explained_variance_ratio_ = pca.explained_variance_ratio_.astype(
            np.float32,
            copy=False,
        )
        if squeeze_output:
            return compressed_features.reshape(-1)
        return compressed_features

    def fit_full_pca_spectrum(self, features: np.ndarray) -> None:
        """
        Fit PCA with full rank to expose eigenvalues before dimensionality reduction.

        Stores diagnostics on the instance:
        - ``last_full_pca_eigenvalues_``
        - ``last_full_pca_explained_variance_ratio_``
        """
        features_array = np.asarray(features, dtype=np.float32)
        if features_array.ndim == 1:
            features_2d = features_array[None, :]
        elif features_array.ndim == 2:
            features_2d = features_array
        else:
            raise ValueError(
                "features must have shape [feature_dim] or [n_samples, feature_dim], "
                f"got {features_array.shape}"
            )

        # PCA variance is undefined with fewer than 2 samples.
        if features_2d.shape[0] < 2:
            self.last_full_pca_eigenvalues_ = np.array([], dtype=np.float32)
            self.last_full_pca_explained_variance_ratio_ = np.array([], dtype=np.float32)
            return

        pca_full = PCA(n_components=None)
        pca_full.fit(features_2d)
        self.last_full_pca_eigenvalues_ = pca_full.explained_variance_.astype(
            np.float32,
            copy=False,
        )
        self.last_full_pca_explained_variance_ratio_ = pca_full.explained_variance_ratio_.astype(
            np.float32,
            copy=False,
        )

    def build_compressed_features(
        self,
        max_hop: int = 2,
        n_components: int = 32,
    ) -> np.ndarray:
        """
        Build flattened features and apply PCA compression.
        """
        flattened_features = self.build_flattened_features(max_hop=max_hop)
        return self.compress_feature_dimension(flattened_features, n_components=n_components)

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
    pca_components: Optional[int] = None,
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
    from algorithms.dopp.graph_builder import GraphBuilder
    from algorithms.dopp.loaders import load_candidates_from_json
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

    final_feature_type = "graph_diffused"
    metadata: Dict[str, object] = {
        "benchmark": benchmark,
        "max_hop": max_hop,
        "self_loop_weight": self_loop_weight,
        "num_graph_nodes": int(graph_builder.node_features.shape[0]),
        "num_graph_edges": int(len(graph_builder.edge_weights)),
        "partition_feature_shape": list(partition_node_features.shape),
    }

    constructor.fit_full_pca_spectrum(features)
    if constructor.last_full_pca_eigenvalues_ is not None:
        metadata["pca_full_eigenvalues"] = constructor.last_full_pca_eigenvalues_.tolist()
    if constructor.last_full_pca_explained_variance_ratio_ is not None:
        metadata["pca_full_explained_variance_ratio"] = (
            constructor.last_full_pca_explained_variance_ratio_.tolist()
        )
        metadata["pca_full_cumulative_explained_variance_ratio"] = np.cumsum(
            constructor.last_full_pca_explained_variance_ratio_
        ).tolist()

    if pca_components is not None:
        original_feature_dim = int(features.shape[1])
        features = constructor.compress_feature_dimension(features, n_components=pca_components)
        final_feature_type = "graph_diffused_pca"
        metadata["pca_components"] = int(pca_components)
        metadata["original_feature_dim"] = original_feature_dim
        if constructor.last_pca_explained_variance_ratio_ is not None:
            metadata["pca_explained_variance_ratio"] = (
                constructor.last_pca_explained_variance_ratio_.tolist()
            )
        if constructor.last_pca_eigenvalues_ is not None:
            metadata["pca_eigenvalues"] = constructor.last_pca_eigenvalues_.tolist()

    return {
        "feature_type": final_feature_type,
        "candidate_keys": candidate_keys,
        "features": features.astype(np.float32, copy=False),
        "feature_dim": int(features.shape[1]),
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract graph-diffused features for D-optimal design and regression."
    )
    parser.add_argument(
        "benchmark",
        help="Benchmark name matching Place-3D/test/or_3D/<benchmark>_3D.json",
    )
    parser.add_argument(
        "candidates_path",
        type=Path,
        help="Path to candidates.json containing candidate partitions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: evaluation/regression_results/{benchmark}",
    )
    parser.add_argument("--max-hop", type=int, default=2, help="Maximum diffusion hop")
    parser.add_argument("--self-loop-weight", type=float, default=1.0, help="Self-loop weight")
    parser.add_argument("--def-path", type=Path, default=None, help="Optional DEF file override")
    parser.add_argument("--upper-die-macros", type=str, default=None, help="Comma-separated upper-die macro names")
    parser.add_argument("--partition-result", type=str, default=None, help="Comma-separated bottom-die placedb node IDs")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="Cell area scaling factor")
    parser.add_argument("--top-k-ratio", type=float, default=0.3, help="Edge keep ratio for dense graphs")
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="Optional PCA output dimension for compressing flattened features.",
    )
    parser.add_argument("--rand-init", action="store_true", default=False, help="Enable DREAMPlace random init")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if not args.candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {args.candidates_path}")

    output_data = build_graph_diffused_feature_bundle(
        benchmark=args.benchmark,
        candidates_path=args.candidates_path,
        max_hop=args.max_hop,
        pca_components=args.pca_components,
        def_path=args.def_path,
        upper_die_macros=args.upper_die_macros,
        partition_result=args.partition_result,
        scale_factor=args.scale_factor,
        top_k_ratio=args.top_k_ratio,
        rand_init=args.rand_init,
        self_loop_weight=args.self_loop_weight,
    )

    out_dir = args.output or (
        Path(__file__).resolve().parents[2] / "evaluation" / "regression_results" / args.benchmark
    )
    os.makedirs(out_dir, exist_ok=True)
    output_path = Path(out_dir) / 'graph_diffused_features.npy'
    np.save(output_path, output_data, allow_pickle=True)
    logging.info("Saved graph-diffused feature bundle to %s", output_path)
    logging.info("Feature shape: %s", output_data["features"].shape)


if __name__ == "__main__":
    main()
