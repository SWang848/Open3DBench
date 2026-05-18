from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np


class PartitionGraphUpdater:
    """
    Load candidate partition solutions and project them onto a base graph.

    The updater assumes the graph comes from ``GraphBuilder``:
    - macro graph nodes map to placedb macro IDs
    - the last graph node is the merged cell node

    For every candidate solution:
    - macros in the configured top partition receive partition label 1
    - macros in the other partition receive partition label 0
    - the merged cell node is always fixed to label 0

    This class only builds partition-dependent node features. Static graph data
    remains owned by ``GraphBuilder``.
    """

    def __init__(
        self,
        graph_builder: Any,
        candidates_path: Path,
    ) -> None:
        self.graph_builder = graph_builder
        self.candidates_path = candidates_path

        self.base_node_features = np.array(graph_builder.node_features, dtype=np.float32, copy=True)
        self.edges = np.array(graph_builder.edges, dtype=np.int64, copy=True)
        self.edge_weights = np.array(graph_builder.edge_weights, dtype=np.float32, copy=True)
        self.cell_node_idx = int(graph_builder.get_graph_cell_node_idx())

        self.expected_macro_ids = set()
        for graph_macro_idx in graph_builder.get_graph_macro_indices():
            self.expected_macro_ids.add(int(graph_builder.graph_macro_idx_to_placedb_macro_id(graph_macro_idx)))

        self._candidate_entries = None
        self._build_hierarchy_metadata()

    def load_candidates(self) -> List[Dict[str, Any]]:
        """Load and flatten all candidate solutions from the JSON archive."""
        if self._candidate_entries is not None:
            return self._candidate_entries

        with self.candidates_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)

        pareto_archive = payload.get("pareto_archive", {})
        solutions_by_key = pareto_archive.get("solutions", {})
        if not isinstance(solutions_by_key, dict):
            raise ValueError(
                "Expected candidates JSON to contain pareto_archive.solutions as a dictionary."
            )

        candidate_entries = []
        for solution_key, entries in solutions_by_key.items():
            if not isinstance(entries, list):
                raise ValueError(
                    "Expected every pareto archive bucket to contain a list of candidate entries."
                )
            for solution_index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise ValueError("Expected each candidate entry to be a dictionary.")
                candidate_entries.append(
                    {
                        "solution_key": solution_key,
                        "solution": entry.get("solution"),
                        "proxies": entry.get("cost"),
                    }
                )

        self._candidate_entries = candidate_entries
        return self._candidate_entries

    def __len__(self) -> int:
        return len(self.load_candidates())

    def get_candidate_entry(self, candidate_index: int) -> Dict[str, Any]:
        """Return the raw flattened candidate entry."""
        return self.load_candidates()[candidate_index]

    def iter_partition_features(
        self,
        limit: Optional[int] = None,
    ) -> Iterator[np.ndarray]:
        """Iterate over candidates and yield ``[n_nodes, feature_dim]`` features."""
        for candidate_index, candidate_entry in enumerate(self.load_candidates()):
            if limit is not None and candidate_index >= limit:
                break
            yield self.build_partition_features(candidate_entry)

    def build_all_partition_features(self, limit: Optional[int] = None) -> np.ndarray:
        """Build stacked partition features with shape ``[n_candidates, n_nodes, feature_dim]``."""
        partition_features = list(self.iter_partition_features(limit=limit))
        if not partition_features:
            return np.empty((0, self.base_node_features.shape[0], 0), dtype=np.float32)
        return np.stack(partition_features, axis=0).astype(np.float32, copy=False)

    def build_partition_features(self, candidate_entry: Dict[str, Any]) -> np.ndarray:
        """Build partition-dependent node features for one candidate."""
        solution = candidate_entry["solution"]
        top_macro_ids = list(solution[1])
        partition_labels = self.build_partition_labels(top_macro_ids)
        incident_cut_net_counts = self.build_incident_cut_net_counts(partition_labels)
        flip_area_imbalance_gains = self.build_flip_area_imbalance_gains(partition_labels)
        cross_tire_cell_conectivity = self.build_cross_tire_cell_conectivity(partition_labels)
        hierarchy_cohesion_features = self.build_hierarchy_cohesion_features(partition_labels)
        partition_features = np.concatenate(
            [
                partition_labels,
                incident_cut_net_counts,
                flip_area_imbalance_gains,
                cross_tire_cell_conectivity,
                hierarchy_cohesion_features,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        return partition_features

    def build_partition_labels(self, top_macro_ids: Sequence[int]) -> np.ndarray:
        """
        Build an ``[n_nodes, 1]`` partition label array.

        Label convention:
        - 1.0: macro assigned to the top die
        - 0.0: macro assigned to the bottom die
        - 0.0: merged cell node
        """
        partition_labels = np.zeros((self.base_node_features.shape[0], 1), dtype=np.float32)
        for placedb_macro_id in top_macro_ids:
            graph_idx = self.graph_builder.placedb_macro_id_to_graph_macro_idx(int(placedb_macro_id))
            partition_labels[graph_idx, 0] = 1.0

        partition_labels[self.cell_node_idx, 0] = 0.0
        return partition_labels

    def build_incident_cut_net_counts(self, partition_labels: np.ndarray) -> np.ndarray:
        """
        Build an ``[n_nodes, 1]`` array of incident cut net counts.

        For each node, this is the sum of edge weights for incident edges whose
        endpoints belong to different partitions.
        """
        incident_cut_net_counts = np.zeros((self.base_node_features.shape[0], 1), dtype=np.float32)

        for edge_idx, (src_idx, dst_idx) in enumerate(self.edges):
            src_idx = int(src_idx)
            dst_idx = int(dst_idx)
            if partition_labels[src_idx, 0] == partition_labels[dst_idx, 0]:
                continue

            edge_weight = float(self.edge_weights[edge_idx])
            incident_cut_net_counts[src_idx, 0] += edge_weight
            incident_cut_net_counts[dst_idx, 0] += edge_weight

        return incident_cut_net_counts

    def build_flip_area_imbalance_gains(self, partition_labels: np.ndarray) -> np.ndarray:
        """
        Build an ``[n_nodes, 1]`` array of area-imbalance gains after flipping.

        The current area imbalance is ``abs(upper_die_area - bottom_die_area)``.
        For each macro node, the feature value is:

        ``current_imbalance - flipped_imbalance``

        so a positive value means flipping that macro reduces imbalance.
        The merged cell node is fixed in the bottom die, so its gain is 0.
        """
        flip_area_imbalance_gains = np.zeros((self.base_node_features.shape[0], 1), dtype=np.float32)
        node_areas = self.base_node_features[:, 0].astype(np.float32, copy=False)

        upper_die_area = float(node_areas[partition_labels[:, 0] == 1.0].sum())
        bottom_die_area = float(node_areas[partition_labels[:, 0] == 0.0].sum())
        current_signed_imbalance = upper_die_area - bottom_die_area
        current_imbalance = abs(current_signed_imbalance)

        for node_idx in range(self.base_node_features.shape[0]):
            if node_idx == self.cell_node_idx:
                continue

            node_area = float(node_areas[node_idx])
            if partition_labels[node_idx, 0] == 1.0:
                flipped_signed_imbalance = current_signed_imbalance - (2.0 * node_area)
            else:
                flipped_signed_imbalance = current_signed_imbalance + (2.0 * node_area)

            flipped_imbalance = abs(flipped_signed_imbalance)
            flip_area_imbalance_gains[node_idx, 0] = current_imbalance - flipped_imbalance

        return flip_area_imbalance_gains

    def build_cross_tire_cell_conectivity(self, partition_labels: np.ndarray) -> np.ndarray:
        """
        Build an ``[n_nodes, 1]`` array of top-partition cell connectivity.

        This copies ``num_connected_cells`` for nodes assigned to the top
        partition and sets the feature to 0 for nodes assigned to the bottom
        partition. Since the merged cell node is fixed in the bottom partition,
        its value is also 0.
        """
        num_connected_cells = self.base_node_features[:, 3:4].astype(np.float32, copy=True)
        return num_connected_cells * partition_labels

    def build_hierarchy_cohesion_features(self, partition_labels: np.ndarray) -> np.ndarray:
        """
        Build an ``[n_nodes, 1]`` array of ancestor-weighted hierarchy cohesion.

        The cluster cohesion score follows ``compute_hierarchy_features`` in
        ``feature_construction_manual.py``. Each macro may belong to multiple
        hierarchy prefix clusters, so its node feature is the weighted average
        of all valid ancestor cluster scores.
        """
        hierarchy_features = np.ones((self.base_node_features.shape[0], 1), dtype=np.float32)
        cohesion_by_prefix: Dict[str, float] = {}
        evidence_by_prefix: Dict[str, float] = {}

        for prefix, macro_ids in self.hierarchy_macronode_dict.items():
            num_stdnodes = self.hierarchy_numstdnode_dict.get(prefix, 0)
            num_macro = len(macro_ids)
            num_macro_in_upper = 0
            for placedb_macro_id in macro_ids:
                graph_idx = self.graph_builder.placedb_macro_id_to_graph_macro_idx(placedb_macro_id)
                if partition_labels[graph_idx, 0] == 1.0:
                    num_macro_in_upper += 1

            num_macro_in_lower = num_macro - num_macro_in_upper
            macro_pair_count = num_macro * (num_macro - 1) / 2.0
            evidence = (num_stdnodes * num_macro) + macro_pair_count

            if num_stdnodes == 0 and (num_macro_in_lower == 0 or num_macro_in_upper == 0):
                cohesion_score = 1.0
            elif num_stdnodes == 0 and num_macro_in_lower > 0 and num_macro_in_upper > 0:
                cohesion_score = 1.0 - (
                    (num_macro_in_lower * num_macro_in_upper) / macro_pair_count
                )
            else:
                cohesion_score = (
                    (num_stdnodes * num_macro_in_lower)
                    + (num_macro_in_lower * (num_macro_in_lower - 1) / 2.0)
                ) / evidence

            cohesion_by_prefix[prefix] = float(cohesion_score)
            evidence_by_prefix[prefix] = float(evidence)

        global_weighted_sum = 0.0
        global_weight_sum = 0.0
        for graph_macro_idx in self.graph_builder.get_graph_macro_indices():
            placedb_macro_id = int(self.graph_builder.graph_macro_idx_to_placedb_macro_id(graph_macro_idx))
            weighted_sum = 0.0
            weight_sum = 0.0

            for prefix in self.macro_to_hierarchy_prefixes.get(placedb_macro_id, []):
                evidence = evidence_by_prefix.get(prefix, 0.0)
                if evidence <= 0.0:
                    continue

                depth_bias = 1.0 + 0.4 * (self.hierarchy_prefix_depth.get(prefix, 1) - 1)
                weight = math.log1p(evidence) * depth_bias
                weighted_sum += cohesion_by_prefix[prefix] * weight
                weight_sum += weight

            if weight_sum > 0.0:
                hierarchy_features[graph_macro_idx, 0] = weighted_sum / weight_sum
                global_weighted_sum += weighted_sum
                global_weight_sum += weight_sum

        if global_weight_sum > 0.0:
            hierarchy_features[self.cell_node_idx, 0] = global_weighted_sum / global_weight_sum

        return hierarchy_features

    def _build_hierarchy_metadata(self) -> None:
        """
        Precompute hierarchy prefix membership used by hierarchy cohesion.

        Prefixes follow the existing convention: split names on ``"__"`` and
        remove the final token, which is the leaf cell or macro name.
        """
        self.hierarchy_numstdnode_dict: Dict[str, int] = {}
        self.hierarchy_macronode_dict: Dict[str, List[int]] = {}
        self.macro_to_hierarchy_prefixes: Dict[int, List[str]] = {}
        self.hierarchy_prefix_depth: Dict[str, int] = {}

        placedb = self.graph_builder.placedb
        physical_node_limit = placedb.num_physical_nodes - placedb.num_terminal_NIs

        for raw_node_name in placedb.node_names:
            node_name = self._decode_node_name(raw_node_name)
            node_id = int(placedb.node_name2id_map[node_name])
            if node_id >= physical_node_limit:
                continue

            prefixes = self._hierarchy_prefixes_from_name(node_name)
            if not prefixes:
                continue

            is_macro = node_id in self.expected_macro_ids
            if is_macro:
                self.macro_to_hierarchy_prefixes[node_id] = prefixes

            for depth, prefix in enumerate(prefixes, start=1):
                self.hierarchy_prefix_depth[prefix] = depth
                if is_macro:
                    self.hierarchy_macronode_dict.setdefault(prefix, []).append(node_id)
                else:
                    self.hierarchy_numstdnode_dict[prefix] = (
                        self.hierarchy_numstdnode_dict.get(prefix, 0) + 1
                    )

    def _hierarchy_prefixes_from_name(self, node_name: str) -> List[str]:
        hierarchy_tokens = node_name.split("__")[:-1]
        prefixes = []
        for depth in range(1, len(hierarchy_tokens) + 1):
            prefixes.append("__".join(hierarchy_tokens[:depth]))
        return prefixes

    @staticmethod
    def _decode_node_name(raw_node_name: Any) -> str:
        if isinstance(raw_node_name, bytes):
            return raw_node_name.decode("utf-8")
        return str(raw_node_name)
