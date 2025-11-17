from __future__ import annotations

import collections
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import Data

# Keep die identifiers consistent with the rest of the codebase
BOTTOM_DIE = 0
UPPER_DIE = 1
MAX_PINS_PER_NET = 20

class FrozenPrefixEncoder(nn.Module):
    def __init__(self, num_prefixes, d=64, pad_id=0, alpha=1.0, max_depth=8, seed=123):
        super().__init__()
        self.pad_id = pad_id
        self.alpha = alpha
        
        g = torch.Generator().manual_seed(seed)
        E = torch.randn(num_prefixes, d, generator=g) / (d ** 0.5)
        E[pad_id] = 0.0
        
        self.emb = nn.Embedding.from_pretrained(E, freeze=True, padding_idx=pad_id)
        self.register_buffer("pos", torch.arange(max_depth).float())
        
    def forward(self, prefix_ids):
        B, L = prefix_ids.shape
        x = self.emb(prefix_ids)
        mask = (prefix_ids != self.pad_id).float()
        
        if self.alpha != 1.0:
            w = (self.alpha ** self.pos[:L])[None, :, None]
            x = x * w
            denom = (w.squeeze(-1) * mask).sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        
        h = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        return h
        
class HierarchyEncoder:
    def __init__(self, placedb: PlaceDB):
        self.PAD = 0
        self.prefix2id = {"<PAD>": self.PAD}
        self.id2prefix = {}
        self.max_depth = 0
        self.placedb = placedb
        
    def build_hierarchy_embeddings(self) -> Dict[str, torch.Tensor]:
        prefix_id_dict = self._extract_hierarchy_prefixes()
        encoder = FrozenPrefixEncoder(
            num_prefixes=len(self.prefix2id), 
            d=64, 
            pad_id=self.PAD, 
            alpha=1.0, 
            max_depth=self.max_depth, 
            seed=123
        )
        prefix_id_seqs = torch.tensor([prefix_id_dict[key] for key in prefix_id_dict.keys()], dtype=torch.long)
        h = encoder(prefix_id_seqs)
        prefix_embedding_dict = {key: h[i] for i, key in enumerate(prefix_id_dict.keys())}
        return prefix_embedding_dict
    
    def _prefixes_from_tokens(self, tokens):
        # ['a','12','!'] -> ['a', 'a/12', 'a/12/!']
        out = []
        acc = []
        for t in tokens:
            acc.append(t)
            out.append("/".join(acc))
        return out

    def _prefix_ids_from_name(self, node_name: str) -> List[int]:
        tokens = node_name.split("__")[:-1]  # remove the last token (cell name)
        prefixes = self._prefixes_from_tokens(tokens)
        prefix_ids = [self.prefix2id[prefix] for prefix in prefixes]
        while len(prefix_ids) < self.max_depth:
            prefix_ids.append(self.PAD)
        return prefix_ids

    def _extract_hierarchy_prefixes(self) -> Dict[str, int]:
        """
        Extract hierarchy prefixes from a placedb.
        """
        hierarchy_prefixes = {}
        for node_name in self.placedb.node_names:
            node_name = node_name.decode("utf-8")
            node = self.placedb.node_name2id_map[node_name]
            if node < (self.placedb.num_physical_nodes - self.placedb.num_terminal_NIs):
                hierarchy_tokens = node_name.split("__")[:-1]  # remove the last token (cell name)
                prefixes = self._prefixes_from_tokens(hierarchy_tokens)
                self.max_depth = max(self.max_depth, len(prefixes))
                for prefix in prefixes:
                    hierarchy_prefixes[prefix] = None

        prefix_id_dict = {}
        for i, prefix in enumerate(hierarchy_prefixes.keys(), start=1):
            self.prefix2id[prefix] = i
        
        self.id2prefix = {i: p for p, i in self.prefix2id.items()}

        for node_name in self.placedb.node_names:
            node_name = node_name.decode("utf-8")
            node = self.placedb.node_name2id_map[node_name]
            if node < (self.placedb.num_physical_nodes - self.placedb.num_terminal_NIs):
                prefix_id_dict[node_name] = self._prefix_ids_from_name(node_name)
        
        return prefix_id_dict

def _edge_die_features(die_u: Optional[int], die_v: Optional[int]) -> Tuple[int, List[int]]:
    """
    Compute same-die flag and cross-die one-hot encoding.
    cross-die one-hot ordering:
        [bottom-bottom, bottom-upper, upper-bottom, upper-upper]
    """
    pair = (die_u if die_u is not None else -1, die_v if die_v is not None else -1)
    same_die = int(die_u == die_v if die_u is not None and die_v is not None else 0)

    one_hot = [0, 0, 0, 0]
    if pair == (BOTTOM_DIE, BOTTOM_DIE):
        one_hot[0] = 1
    elif pair == (BOTTOM_DIE, UPPER_DIE):
        one_hot[1] = 1
    elif pair == (UPPER_DIE, BOTTOM_DIE):
        one_hot[2] = 1
    elif pair == (UPPER_DIE, UPPER_DIE):
        one_hot[3] = 1

    return same_die, one_hot

def build_static_graph(
    placedb,
    hierarchy_embeddings: Optional[Dict[str, torch.Tensor]] = None,
) -> nx.Graph:
    G = nx.Graph()
    if hierarchy_embeddings is None:
        hierarchy_encoder = HierarchyEncoder(placedb)
        hierarchy_embedding_dict = hierarchy_encoder.build_hierarchy_embeddings()
    else:
        hierarchy_embedding_dict = hierarchy_embeddings

    node_attrs = {}
    mean_node_area = 0.0
    min_node_area = float("inf")
    max_node_area = 0.0
    num = 0

    for node_name in placedb.node_names:
        node_name = node_name.decode("utf-8")
        node = placedb.node_name2id_map[node_name]
        if node < (placedb.num_physical_nodes - placedb.num_terminal_NIs):
            G.add_node(node)
            node_area = placedb.node_size_x[node] * placedb.node_size_y[node]
            node_attrs[node] = {
                "area": node_area,
                "hierarchy_embedding": hierarchy_embedding_dict[node_name],
                "is_macro": False,
                "die": 0,
            }
            mean_node_area += node_area
            min_node_area = min(min_node_area, node_area)
            max_node_area = max(max_node_area, node_area)
            num += 1
    
    # update macro nodes attributes
    mean_node_area = mean_node_area / num
    for node_name in placedb.node_names:
        node = placedb.node_name2id_map[node_name.decode("utf-8")]
        if node < (placedb.num_physical_nodes - placedb.num_terminal_NIs):
            node_area = placedb.node_size_x[node] * placedb.node_size_y[node]
            if (node_area > (mean_node_area * 10)) and (placedb.node_size_y[node] > (placedb.row_height * 2)):
                node_attrs[node]["is_macro"] = True
                node_attrs[node]["area"] = (node_attrs[node]["area"] - min_node_area) / (max_node_area - min_node_area)
    nx.set_node_attributes(G, node_attrs)

    edge_weights: Dict[Tuple[int, int], float] = collections.defaultdict(float)
    for net_name in placedb.net_names:
        net_id = placedb.net_name2id_map[net_name.decode("utf-8")]
        pins = placedb.net2pin_map[net_id]
        connected_nodes = []

        for pin in pins:
            node_id = placedb.pin2node_map[pin]
            if node_id < placedb.num_physical_nodes - placedb.num_terminal_NIs:
                connected_nodes.append(node_id)

        if len(connected_nodes) > MAX_PINS_PER_NET:
            logging.debug(
                f"Skipping clique creation for net {net_name.decode('utf-8')} with {len(connected_nodes)} pins (>= 20)"
            )
            continue

        for idx in range(len(connected_nodes)):
            for jdx in range(idx + 1, len(connected_nodes)):
                u = connected_nodes[idx]
                v = connected_nodes[jdx]
                if u == v:
                    continue
                key = (u, v) if u < v else (v, u)
                edge_weights[key] += 1.0

    min_weight = min(edge_weights.values())
    max_weight = max(edge_weights.values())
    for (u, v), weight in edge_weights.items():
        normalized_weight = (weight - min_weight) / (max_weight - min_weight) if max_weight - min_weight > 0 else 1.0
        G.add_edge(
            u,
            v,
            weight=normalized_weight,
        )

    return G

def apply_partition_to_graph(
    base_graph: nx.Graph,
    partition_solution: List[List[int]],
) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from((node, data.copy()) for node, data in base_graph.nodes(data=True))
    G.add_edges_from((u, v, data.copy()) for u, v, data in base_graph.edges(data=True))

    upper_set = set(partition_solution[1])

    for node_id in upper_set:
        G.nodes[node_id]["die"] = 1

    return G


def build_basic_graph(
    placedb,
    partition_solution: List[List[int]],
    hierarchy_embeddings: Optional[Dict[str, torch.Tensor]] = None,
) -> nx.Graph:
    base_graph = build_static_graph(placedb, hierarchy_embeddings)
    return apply_partition_to_graph(base_graph, partition_solution)


def graph_to_pyg_base(G: nx.Graph) -> Tuple[Data, Dict[int, int]]:
    """
    Convert base graph to PyG format once. Returns the base Data and node_id->idx mapping.
    All nodes have die=0 initially.
    """
    node_list = list(G.nodes())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}  # map node id to index. pytorch geometric requires dense contiguous index.

    node_features = []
    for node_id in node_list:
        attrs: Dict = G.nodes[node_id]
        area = torch.tensor([float(attrs.get("area", 0.0))], dtype=torch.float32)
        macro_flag = torch.tensor([1.0 if attrs.get("is_macro", False) else 0.0], dtype=torch.float32)
        die = torch.tensor([0.0], dtype=torch.float32)  # Always 0 for base graph

        hierarchy_embedding = attrs.get("hierarchy_embedding")
        if hierarchy_embedding is None:
            hierarchy_embedding = torch.zeros(64, dtype=torch.float32)
        elif isinstance(hierarchy_embedding, torch.Tensor):
            hierarchy_embedding = hierarchy_embedding.float()
        else:
            hierarchy_embedding = torch.tensor(hierarchy_embedding, dtype=torch.float32)

        feature_vector = torch.cat(
            [area, macro_flag, die, hierarchy_embedding], dim=0
        )
        node_features.append(feature_vector)

    if not node_features:
        raise ValueError("Graph contains no nodes.")

    x = torch.stack(node_features, dim=0)

    edge_index_list = []
    edge_weight_list = []
    for u, v, data in G.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        weight = float(data.get("weight", 1.0))

        edge_index_list.extend([[u_idx, v_idx], [v_idx, u_idx]])
        edge_weight_list.extend([weight, weight])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    return data, node_to_idx


def update_die_in_pyg(data: Data, partition: List[List[int]], node_to_idx: Dict[int, int]) -> Data:
    """
    Update die assignments in a PyG Data object. Die feature is at index 2.
    Returns a new Data object with updated die features.
    Since base graph has all nodes at die=0, we only need to update upper_set nodes to 1.0.
    """
    upper_set = set(partition[1])
    new_data = data.clone()
    # Die feature is at index 2 (after area=0, macro_flag=1, die=2, hierarchy_embedding=3:66)
    # All nodes already have die=0 from base graph, so only update upper_set nodes
    for node_id in upper_set:
        new_data.x[node_to_idx[node_id], 2] = 1.0
    return new_data

