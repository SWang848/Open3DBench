"""
Graph Builder for Macro Partition
Constructs a graph where:
- Nodes: macros (individual nodes) + cells (merged into one node)
- Edges: connections between nodes based on netlist
- Edge weights: number of nets connecting two nodes
"""
import numpy as np
import torch
from typing import Dict, List, Set, Tuple
from torch_geometric.data import Data



class GraphBuilder:
    """
    Build graph representation for macro partition problem.
    """
    
    def __init__(self, partition_params, dreamplace_loader):
        """
        Initialize graph builder.
        
        Args:
            dreamplace_loader: lightweight object exposing `dmp_placedb`,
                `dmp_params.partition_params`, and `determine_macro()`
        """
        self.partition_params = partition_params
        self.dreamplace_loader = dreamplace_loader
        self.placedb = dreamplace_loader.dmp_placedb
        
        # Get graph construction parameters from the DREAMPlace loader if available.
        self.scale_factor = self.partition_params.get('scale_factor', 1.0)
        
        # Get top_k_ratio from loader params if available (default: 0.3, range: 0-1)
        # If total edges > top_k_ratio * n * n, keep only top top_k_ratio proportion of edges
        self.top_k_ratio = self.partition_params.get('top_k_ratio', 0.3)
        
        # Determine macros if not already determined
        if not hasattr(dreamplace_loader, 'macros'):
            dreamplace_loader.macros = None
        if dreamplace_loader.macros is None:
            dreamplace_loader.macros = dreamplace_loader.determine_macro()
        
        self.placedb_macro_ids = set(dreamplace_loader.macros)
        
        # Get all nodes and cells
        all_placedb_nodes = set()
        for node_name in self.placedb.node_names:
            if isinstance(node_name, bytes):
                node_name_str = node_name.decode('utf-8')
            else:
                node_name_str = str(node_name)
            placedb_node_id = self.placedb.node_name2id_map[node_name_str]
            all_placedb_nodes.add(placedb_node_id)
        
        self.placedb_cell_ids = all_placedb_nodes - self.placedb_macro_ids
        
        # Build graph
        self._build_graph()
    
    def _build_graph(self):
        """
        Build the graph structure.
        """
        # Node mapping: macro nodes + 1 cell node
        # macro_nodes: [0, ..., n_macros-1] map to placedb macro IDs
        # cell_node: n_macros maps to all cells
        # placedb_macro_id_to_graph_idx: maps placedb macro ID -> graph node index
        # graph_idx_to_placedb_macro_id: maps graph node index -> placedb macro ID
        self.placedb_macro_id_to_graph_idx = {placedb_macro_id: graph_idx for graph_idx, placedb_macro_id in enumerate(sorted(self.placedb_macro_ids))}
        self.graph_idx_to_placedb_macro_id = {graph_idx: placedb_macro_id for placedb_macro_id, graph_idx in self.placedb_macro_id_to_graph_idx.items()}
        self.n_macros = len(self.placedb_macro_ids)
        self.graph_cell_node_idx = self.n_macros  # Cell node is the last node
        
        # Build edge list and edge weights
        edges = []
        edge_weights = []
        
        # Build adjacency map: (graph_idx1, graph_idx2) -> net_count
        adjacency = {}
        
        # Process each net
        for placedb_net_id in range(len(self.placedb.net2pin_map)):
            # Get all pins in this net
            placedb_pin_ids = self.placedb.net2pin_map[placedb_net_id]
            
            # Get nodes connected to this net
            placedb_nodes_in_net = set()
            for placedb_pin_id in placedb_pin_ids:
                placedb_node_id = self.placedb.pin2node_map[placedb_pin_id]
                placedb_nodes_in_net.add(placedb_node_id)
            
            # Convert to graph node indices (macros -> graph_idx, cells -> graph_cell_node_idx)
            graph_node_indices = set()
            for placedb_node_id in placedb_nodes_in_net:
                if placedb_node_id in self.placedb_macro_ids:
                    graph_node_indices.add(self.placedb_macro_id_to_graph_idx[placedb_node_id])
                else:
                    graph_node_indices.add(self.graph_cell_node_idx)
            
            # Add edges between all pairs of nodes in this net
            graph_node_list = list(graph_node_indices)
            for i in range(len(graph_node_list)):
                for j in range(i + 1, len(graph_node_list)):
                    graph_idx1, graph_idx2 = graph_node_list[i], graph_node_list[j]
                    if graph_idx1 > graph_idx2:
                        graph_idx1, graph_idx2 = graph_idx2, graph_idx1
                    key = (graph_idx1, graph_idx2)
                    adjacency[key] = adjacency.get(key, 0) + 1
        
        # Convert adjacency to edge list (1-hop edges)
        for (graph_idx1, graph_idx2), weight in adjacency.items():
            edges.append([graph_idx1, graph_idx2])
            edge_weights.append(weight)
        
        # Filter edges based on threshold: if total_edges > top_k_ratio * n * n,
        # keep only top top_k_ratio proportion of edges
        total_edges = len(edges)
        n_nodes = self.n_macros + 1  # macros + 1 cell node
        threshold = self.top_k_ratio * n_nodes * n_nodes
        
        if total_edges > threshold and self.top_k_ratio > 0 and self.top_k_ratio <= 1.0:
            # Sort edges by weight (descending)
            edge_weight_pairs = list(zip(edges, edge_weights))
            edge_weight_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top k_ratio proportion of edges
            num_edges_to_keep = max(1, int(total_edges * self.top_k_ratio))
            top_k_pairs = edge_weight_pairs[:num_edges_to_keep]
            edges = [pair[0] for pair in top_k_pairs]
            edge_weights = [pair[1] for pair in top_k_pairs]
        
        self.edges = np.array(edges, dtype=np.int64)
        self.edge_weights = np.array(edge_weights, dtype=np.float32)
        
        # Build node features
        self._build_node_features()
    
    def _build_node_features(self):
        """
        Build node features for each node.
        Features: [area, width, height, num_connected_cells, num_pins]
        """
        n_nodes = self.n_macros + 1  # macros + 1 cell node
        
        features = []
        
        # Features for macro nodes
        for placedb_macro_id in sorted(self.placedb_macro_ids):
            area = float(self.placedb.node_size_x[placedb_macro_id]) * float(self.placedb.node_size_y[placedb_macro_id])
            width = float(self.placedb.node_size_x[placedb_macro_id])
            height = float(self.placedb.node_size_y[placedb_macro_id])
            
            # Count connected cells (cells connected via nets)
            connected_placedb_cells = set()
            placedb_pin_ids = self.placedb.node2pin_map[placedb_macro_id]
            for placedb_pin_id in placedb_pin_ids:
                placedb_net_id = self.placedb.pin2net_map[placedb_pin_id]
                placedb_net_pin_ids = self.placedb.net2pin_map[placedb_net_id]
                for placedb_net_pin_id in placedb_net_pin_ids:
                    placedb_net_node_id = self.placedb.pin2node_map[placedb_net_pin_id]
                    if placedb_net_node_id in self.placedb_cell_ids:
                        connected_placedb_cells.add(placedb_net_node_id)
            num_connected_cells = len(connected_placedb_cells)
            
            # Count pins
            num_pins = len(placedb_pin_ids)
            
            features.append([area, width, height, num_connected_cells, num_pins])
        
        # Features for cell node
        # Apply scale_factor to cell area
        total_cell_area = sum(
            float(self.placedb.node_size_x[placedb_cell_id]) * float(self.placedb.node_size_y[placedb_cell_id])
            for placedb_cell_id in self.placedb_cell_ids
        ) / self.scale_factor
        
        # Calculate average cell dimensions
        avg_cell_width = np.mean([float(self.placedb.node_size_x[placedb_cell_id]) for placedb_cell_id in self.placedb_cell_ids]) if len(self.placedb_cell_ids) > 0 else 0.0
        avg_cell_height = np.mean([float(self.placedb.node_size_y[placedb_cell_id]) for placedb_cell_id in self.placedb_cell_ids]) if len(self.placedb_cell_ids) > 0 else 0.0
        
        # Decompose total area into width and height based on average cell aspect ratio
        # This reflects that cell node occupies a large footprint
        if avg_cell_height > 0:
            aspect_ratio = avg_cell_width / avg_cell_height
            # width * height = total_cell_area, width / height = aspect_ratio
            # height = sqrt(total_cell_area / aspect_ratio)
            # width = height * aspect_ratio
            cell_node_height = np.sqrt(total_cell_area / aspect_ratio) if aspect_ratio > 0 else np.sqrt(total_cell_area)
            cell_node_width = cell_node_height * aspect_ratio
        else:
            # Fallback: if no cells or height is 0, use square approximation
            cell_node_width = np.sqrt(total_cell_area)
            cell_node_height = np.sqrt(total_cell_area)
        
        # Count connected cells (all cells are connected to each other conceptually)
        num_connected_cells = len(self.placedb_cell_ids)
        
        # Count pins for all cells
        num_pins = sum(len(self.placedb.node2pin_map[placedb_cell_id]) for placedb_cell_id in self.placedb_cell_ids)
        
        features.append([total_cell_area, cell_node_width, cell_node_height, num_connected_cells, num_pins])
        
        self.node_features = np.array(features, dtype=np.float32)
    
    def update_edge_weights(self, new_edge_weights):
        """
        Update edge weights dynamically.
        
        Args:
            new_edge_weights: New edge weights array of shape [n_edges]
        """
        if len(new_edge_weights) != len(self.edge_weights):
            raise ValueError(f"Edge weights length mismatch: {len(new_edge_weights)} != {len(self.edge_weights)}")
        self.edge_weights = np.array(new_edge_weights, dtype=np.float32)
    
    def update_node_features(self, new_node_features):
        """
        Update node features dynamically.
        
        Args:
            new_node_features: New node features array of shape [n_nodes, feature_dim]
        """
        if new_node_features.shape[0] != self.node_features.shape[0]:
            raise ValueError(f"Node features shape mismatch: {new_node_features.shape[0]} != {self.node_features.shape[0]}")
        self.node_features = np.array(new_node_features, dtype=np.float32)
    
    def get_pyg_data(self, device='cpu'):
        """
        Get PyTorch Geometric Data object.
        
        Args:
            device: Device to place tensors on
        
        Returns:
            Data: PyG Data object
        """
        edge_index = torch.tensor(self.edges.T, dtype=torch.long, device=device)
        edge_attr = torch.tensor(self.edge_weights, dtype=torch.float32, device=device)
        x = torch.tensor(self.node_features, dtype=torch.float32, device=device)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def get_graph_macro_indices(self):
        """
        Get list of graph macro node indices (excluding cell node).
        
        Returns:
            List[int]: Graph macro node indices [0, ..., n_macros-1]
        """
        return list(range(self.n_macros))
    
    def get_graph_cell_node_idx(self):
        """
        Get graph cell node index.
        
        Returns:
            int: Graph cell node index (n_macros)
        """
        return self.graph_cell_node_idx
    
    def graph_macro_idx_to_placedb_macro_id(self, graph_macro_idx):
        """
        Convert graph macro node index to placedb macro ID.
        
        Args:
            graph_macro_idx: Graph macro node index (0 to n_macros-1)
        
        Returns:
            int: Placedb macro ID
        """
        return self.graph_idx_to_placedb_macro_id[graph_macro_idx]
    
    def placedb_macro_id_to_graph_macro_idx(self, placedb_macro_id):
        """
        Convert placedb macro ID to graph macro node index.
        
        Args:
            placedb_macro_id: Placedb macro ID
        
        Returns:
            int: Graph macro node index (0 to n_macros-1)
        """
        return self.placedb_macro_id_to_graph_idx[placedb_macro_id]
    
    def get_area_info(self):
        """
        Get area information for partition evaluation.
        
        Returns:
            dict: Dictionary with 'macro_areas', 'cell_area', 'total_area'
        """
        macro_areas = []
        for placedb_macro_id in sorted(self.placedb_macro_ids):
            area = float(self.placedb.node_size_x[placedb_macro_id]) * float(self.placedb.node_size_y[placedb_macro_id])
            macro_areas.append(area)
        
        cell_area = sum(
            float(self.placedb.node_size_x[placedb_cell_id]) * float(self.placedb.node_size_y[placedb_cell_id])
            for placedb_cell_id in self.placedb_cell_ids
        ) / self.scale_factor
        
        total_area = sum(macro_areas) + cell_area
        
        return {
            'macro_areas': np.array(macro_areas),
            'cell_area': cell_area,
            'total_area': total_area
        }
