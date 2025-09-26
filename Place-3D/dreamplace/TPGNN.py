import networkx as nx
from networkx.utils import UnionFind
import heapq
import logging
import sys
import os
import time
import numpy as np
import re
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Add root directory to Python path for dreamplace imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# import dreamplace.ops.place_io.place_io as place_io
from dreamplace import Params, PlaceDB
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_undirected, negative_sampling
from typing import Dict, List, Tuple, Optional
import random

class TPGNNRandomWalk:
    """
    Random walk implementation following TP-GNN paper algorithm.
    Generates neighborhoods N(v) for contrastive learning.
    """
    
    def __init__(self,
                 walk_length: int = 5,  # LRW in paper
                 num_walks: int = 5):   # NRW in paper
        """
        Initialize simple random walk for TP-GNN.
        
        Args:
            walk_length: Length of each random walk
            num_walks: Number of walks per node
        """
        self.walk_length = walk_length
        self.num_walks = num_walks
        
    def build_neighborhoods(self, data: Data) -> Dict[int, set]:
        """
        Build neighborhoods N(v) for each node following TP-GNN paper.
        N(v) = all nodes visited in random walks starting from v.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Dictionary mapping node_id -> set of neighbor nodes
        """

        neighborhoods = {}
        num_nodes = data.num_nodes
        
        # Initialize empty neighborhoods
        for node in range(num_nodes):
            neighborhoods[node] = set()
        
        # Create adjacency list for faster neighbor lookup
        adj_list = {}
        for i in range(num_nodes):
            adj_list[i] = []
        
        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            adj_list[src].append(dst)
            adj_list[dst].append(src)  # Assuming undirected graph
        

        # Generate walks and build neighborhoods
        for start_node in range(num_nodes):
            for _ in range(self.num_walks):
                walk = self._single_walk(start_node, adj_list, data)
                # Add all nodes in walk to neighborhood of start_node
                for node in walk:
                    if node != start_node:  # Don't include self
                        neighborhoods[start_node].add(node)

        logging.info(f"Built neighborhoods with average size {np.mean([len(n) for n in neighborhoods.values()]):.2f}")
        return neighborhoods
    
    def neighborhoods_to_edge_tensors(self, neighborhoods: Dict[int, set], num_nodes: int, device: torch.device, num_neg_samples: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert neighborhood dictionary to PyTorch edge tensors for efficient batch processing.
        
        Args:
            neighborhoods: Dictionary mapping node_id -> set of neighbor nodes
            num_nodes: Total number of nodes in the graph
            device: Device to place tensors on
            num_neg_samples: Number of negative samples to generate. If None, defaults to number of positive edges
            
        Returns:
            pos_edge_index: [2, num_pos_edges] tensor of positive edges
            neg_edge_index: [2, num_neg_edges] tensor of negative edges (sampled)
        """
        # Build positive edge index from neighborhoods
        pos_edges = []
        for node, neighbors in neighborhoods.items():
            for neighbor in neighbors:
                pos_edges.append([node, neighbor])
        
        if not pos_edges:
            # Handle case with no positive edges
            pos_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            pos_edge_index = torch.tensor(pos_edges, dtype=torch.long, device=device).t()
            # Ensure edges are undirected for consistency
            pos_edge_index = to_undirected(pos_edge_index)
        
        # Sample negative edges that don't exist in positive edges
        try:
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=num_nodes,
                num_neg_samples=min(num_neg_samples, num_nodes * (num_nodes - 1) // 2),  # Cap at max possible edges
                method='sparse'
            )
        except Exception as e:
            # Fallback if negative sampling fails
            logging.warning(f"Negative sampling failed: {e}. Using empty negative edges.")
            neg_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        
        return pos_edge_index, neg_edge_index
    
    def _single_walk(self, start_node: int, adj_list: Dict, data: Data) -> List[int]:
        """
        Generate a simple random walk starting from start_node.
        Following the paper's algorithm - just sample 1-hop neighbors randomly.
        Avoids going back to already visited nodes (no backtracking).
        """
        walk = [start_node]
        current_node = start_node
        visited = {start_node}  # Track visited nodes to avoid backtracking
        
        for _ in range(self.walk_length - 1):
            neighbors = adj_list.get(current_node, [])
            if not neighbors:
                break
            
            # Filter out already visited neighbors to avoid backtracking
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if not unvisited_neighbors:
                # If all neighbors are visited, stop the walk
                break
            
            # Simple random selection from unvisited neighbors
            next_node = np.random.choice(unvisited_neighbors)
            walk.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        return walk


class TPGNNModel(nn.Module):
    """
    Timing-aware Placement Graph Neural Network (TP-GNN).
    Uses GNN with attention instead of simple multi-hop aggregation from paper.
    Combines GNN backbone with timing-specific features for placement optimization.
    """
    
    def __init__(self,
                 input_dim: int = 6,  # [x, y, area, avg_cap, avg_slew, avg_delay] - but we'll skip x, y
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_attention: bool = True,
                 num_heads: int = 4):
        """
        Initialize the timing-aware GNN.
        
        Args:
            input_dim: Original dimension of input features (includes x, y)
            hidden_dim: Hidden dimension size
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
            num_heads: Number of attention heads (if using attention)
        """
        super(TPGNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Input projection for non-position features only (input_dim - 2)
        # Skip x, y coordinates to avoid huge values
        self.non_pos_dim = max(1, input_dim - 2)  # At least 1 dimension
        self.input_proj = nn.Linear(self.non_pos_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if use_attention:
                layer = GATConv(hidden_dim, hidden_dim // num_heads, 
                              heads=num_heads, dropout=dropout, concat=True)
            else:
                layer = GCNConv(hidden_dim, hidden_dim)
            
            self.gnn_layers.append(layer)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Timing-specific attention
        self.timing_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def _extract_non_position_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and normalize only non-position features.
        Completely excludes x, y position features from embeddings.
        
        Args:
            x: Input features [num_nodes, num_features] = [x, y, area, avg_cap, avg_slew, avg_delay]
            
        Returns:
            Normalized non-position features [num_nodes, num_features-2]
        """
        if x.size(1) <= 2:
            # If we only have x, y coordinates, return a dummy feature on the same device
            return torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        
        # Extract features from index 2 onwards (skip x, y)
        non_position_features = x[:, 2:]  # [area, avg_cap, avg_slew, avg_delay]
        
        # Normalize these features
        mean = non_position_features.mean(dim=0, keepdim=True)
        std = non_position_features.std(dim=0, keepdim=True)
        
        # Avoid division by zero for constant features
        std = torch.clamp(std, min=1e-8)
        
        normalized_features = (non_position_features - mean) / std
        return normalized_features
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of the GNN.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Node embeddings
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # Extract only non-position features (skip x, y coordinates)
        # This avoids huge position values affecting embeddings
        non_pos_features = self._extract_non_position_features(x)
        
        # Input projection using only non-position features
        h = F.relu(self.input_proj(non_pos_features))
        h = self.dropout_layer(h)
        
        # GNN layers
        for i, (gnn_layer, batch_norm) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            h_new = gnn_layer(h, edge_index)
            h_new = batch_norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout_layer(h_new)
            
            # Residual connection
            if h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
        
        # Timing-aware attention
        h_attended, _ = self.timing_attention(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
        h = h + h_attended.squeeze(0)
        
        # Output projection
        output = self.output_proj(h)
        
        # L2 normalize embeddings to unit sphere for stable contrastive learning
        output = F.normalize(output, p=2, dim=1)
        
        return output


class NetworkXToPyGConverter:
    """
    Utility class to convert NetworkX graphs to PyTorch Geometric format.
    """
    
    @staticmethod
    def convert_graph(G: nx.Graph, 
                     node_features: List[str] = None,
                     edge_features: List[str] = None) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data object.
        
        Args:
            G: NetworkX graph
            node_features: List of node attribute names to include as features
            edge_features: List of edge attribute names to include as features
            
        Returns:
            PyTorch Geometric Data object
        """
        if node_features is None:
            node_features = ['x', 'y', 'area', 'avg_cap', 'avg_slew', 'avg_delay']
        
        # Create node feature matrix
        node_list = list(G.nodes())
        node_mapping = {node: i for i, node in enumerate(node_list)}
        num_nodes = len(node_list)
        
        # Extract node features
        x = []
        for node in node_list:
            node_data = G.nodes[node]
            features = []
            for feature in node_features:
                value = node_data.get(feature, 0.0)
                if isinstance(value, bool):
                    value = float(value)
                features.append(value)
            x.append(features)
        
        x = torch.tensor(x, dtype=torch.float)
        
        # Create edge index
        edge_list = []
        edge_weights = []
        
        for u, v, edge_data in G.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            
            edge_list.append([u_idx, v_idx])
            edge_list.append([v_idx, u_idx])  # Add reverse edge for undirected graph
            
            weight = edge_data.get('weight', 1.0)
            edge_weights.extend([weight, weight])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        # Create additional node metadata
        node_names = [G.nodes[node].get('name', f'node_{node}') for node in node_list]
        is_macro = torch.tensor([G.nodes[node].get('is_macro', False) for node in node_list], dtype=torch.bool)
        
        # Create PyG Data object
        data = Data(x=x, 
                   edge_index=edge_index, 
                   edge_attr=edge_attr,
                   num_nodes=num_nodes,
                   node_names=node_names,
                   is_macro=is_macro)
        
        return data


class TPGNNTrainer:
    """
    Training pipeline for TP-GNN following paper's unsupervised contrastive learning.
    """
    
    def __init__(self,
                 model: TPGNNModel,
                 random_walk: TPGNNRandomWalk,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 initial_lr: float = 0.001,
                 lr_decay_factor: float = 0.95,
                 lr_decay_patience: int = 5):
        """
        Initialize the trainer.
        
        Args:
            model: The GNN model
            random_walk: Random walk generator
            device: Device to run training on
            initial_lr: Initial learning rate
            lr_decay_factor: Factor by which to reduce learning rate
            lr_decay_patience: Number of epochs to wait before reducing learning rate
        """
        self.model = model.to(device)
        self.random_walk = random_walk
        self.device = device
        self.initial_lr = initial_lr
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_patience = lr_decay_patience
        
        # Optimizers
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
        
        # Learning rate scheduler - reduces LR when loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.model_optimizer, 
            mode='min', 
            factor=lr_decay_factor, 
            patience=lr_decay_patience,
            verbose=True,
            min_lr=1e-6  # Minimum learning rate
        )
        
        # Loss functions
        self.contrastive_loss = nn.CosineEmbeddingLoss()
        
    def train_epoch(self, data: Data, num_neg_samples: int = None) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            data: PyTorch Geometric Data object
            num_neg_samples: Number of negative samples per graph. If None, defaults to number of positive edges
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        data = data.to(self.device)
        
        # Build neighborhoods N(v) using random walks (following paper)
        neighborhoods = self.random_walk.build_neighborhoods(data)
        
        # Convert neighborhoods to edge tensors for efficient processing (Phase 1)
        pos_edge_index, neg_edge_index = self.random_walk.neighborhoods_to_edge_tensors(
            neighborhoods, data.num_nodes, self.device, num_neg_samples
        )
        
        # Forward pass through GNN
        embeddings = self.model(data)
        
        # Contrastive learning using vectorized approach
        contrastive_loss = self._compute_contrastive_loss(embeddings, pos_edge_index, neg_edge_index)
        
        # Combined loss
        total_loss = contrastive_loss
        
        # Backward pass
        self.model_optimizer.zero_grad()
        total_loss.backward()
        self.model_optimizer.step()
        
        return {
            'contrastive_loss': contrastive_loss.item(),
            'total_loss': total_loss.item(),
        }
    
    def _compute_contrastive_loss(self, embeddings: torch.Tensor, 
                                  pos_edge_index: torch.Tensor, 
                                  neg_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Vectorized contrastive loss computation using edge tensors.
        Eliminates nested loops by using batch tensor operations.
        
        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]
            pos_edge_index: [2, num_pos_edges] tensor of positive edges
            neg_edge_index: [2, num_neg_edges] tensor of negative edges
            
        Returns:
            Contrastive loss tensor
        """
        if pos_edge_index.size(1) == 0 and neg_edge_index.size(1) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        total_loss = 0.0
        
        # Positive loss: maximize similarity between connected nodes
        if pos_edge_index.size(1) > 0:
            # Get embeddings for source and target nodes in positive edges
            pos_src_embeddings = embeddings[pos_edge_index[0]]  # [num_pos_edges, embed_dim]
            pos_tgt_embeddings = embeddings[pos_edge_index[1]]  # [num_pos_edges, embed_dim]
            
            # Compute similarities using batch dot product
            pos_similarities = torch.sum(pos_src_embeddings * pos_tgt_embeddings, dim=1)  # [num_pos_edges]
            
            # Positive loss: -log(sigmoid(similarity)) for maximization
            pos_loss = -torch.log(torch.sigmoid(pos_similarities) + 1e-15)  # Add epsilon for numerical stability
            pos_loss = torch.mean(pos_loss)  # Average over all positive edges
            
            total_loss += pos_loss
        
        # Negative loss: minimize similarity between disconnected nodes
        if neg_edge_index.size(1) > 0:
            # Get embeddings for source and target nodes in negative edges
            neg_src_embeddings = embeddings[neg_edge_index[0]]  # [num_neg_edges, embed_dim]
            neg_tgt_embeddings = embeddings[neg_edge_index[1]]  # [num_neg_edges, embed_dim]
            
            # Compute similarities using batch dot product
            neg_similarities = torch.sum(neg_src_embeddings * neg_tgt_embeddings, dim=1)  # [num_neg_edges]
            
            # Negative loss: -log(sigmoid(-similarity)) for minimization
            neg_loss = -torch.log(torch.sigmoid(-neg_similarities) + 1e-15)  # Add epsilon for numerical stability
            neg_loss = torch.mean(neg_loss)  # Average over all negative edges
            
            total_loss += neg_loss
        
        return total_loss
    
    def evaluate(self, data: Data, targets: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            data: PyTorch Geometric Data object
            targets: Optional target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(data)
        
        metrics = {'avg_loss': 0.0}  # Placeholder for now
        return metrics
    
    def update_learning_rate(self, loss_value: float):
        """
        Update learning rate based on loss value using scheduler.
        
        Args:
            loss_value: Current loss value to use for scheduling
        """
        self.scheduler.step(loss_value)
        
    def get_current_lr(self) -> float:
        """
        Get current learning rate.
        
        Returns:
            Current learning rate
        """
        return self.model_optimizer.param_groups[0]['lr']


class TPGNN:
    """
    TP-GNN (Timing-aware Partitioning Graph Neural Network) class for 3D placement.
    
    This class implements the complete TP-GNN pipeline including:
    - Timing report parsing
    - Clique graph construction
    - Hierarchy-aware edge contraction
    - GNN training and embedding generation
    - Weighted K-means partitioning
    """
    
    def __init__(self, placedb, timing_features=None):
        """
        Initialize TPGNN with placement database and optional timing features.
        
        Args:
            placedb: PlaceDB object containing placement information
            timing_features: Dictionary mapping node IDs to timing features (optional)
        """
        self.placedb = placedb
        self.timing_features = timing_features or {}
        self.G_base = None
        self.G_contracted = None
        self.tpgnn_model = None
        self.tpgnn_results = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

    def parse_timing_report(self, timing_report_path):
        """
        Parse timing report to extract cap, slew, and delay information for each node
        @param timing_report_path: path to the timing report file
        @return: dictionary mapping node IDs to timing features
        """
        timing_features = {}
        
        with open(timing_report_path, 'r') as f:
            lines = f.readlines()
        
        current_path = False
        path_count = 0
        node_count = 0
        
        for line in lines:
            line = line.strip()
            
            # Check if this line starts a new timing path
            if line.startswith("Fanout     Cap    Slew   Delay    Time   Description"):
                current_path = True
                path_count += 1
                continue
            
            # Skip if not in a timing path section
            if not current_path:
                continue
            
            # Check if we've reached the end of a path
            if line.startswith("Startpoint:"):
                current_path = False
                continue
            
            # Skip empty lines and separator lines
            if not line or line.startswith("-"):
                continue
            
            # Parse timing line: Fanout Cap Slew Delay Time Description
            # Format: "4   15.68    0.02    0.14    0.14 ^ issue_stage_i_i_scoreboard__60955_/Q (DFFR_X2)"
            parts = line.split()
            if len(parts) >= 6:
                try:
                    if len(parts) == 6:
                        cap = 0.0
                        slew = float(parts[0]) if parts[0] != "" else 0.0
                        delay = float(parts[1]) if parts[1] != "" else 0.0
                        name = parts[-2]
                    else:
                        cap = float(parts[1]) if parts[1] != "" else 0.0
                        slew = float(parts[2]) if parts[2] != "" else 0.0
                        delay = float(parts[3]) if parts[3] != "" else 0.0
                        name = parts[-2]
                    
                    # Skip buffer lines 
                    if "(BUF_" in parts[-1]:
                        continue
                    
                    # Extract clean node name (remove pin information and cell type)
                    # Example: "issue_stage_i_i_scoreboard__60955_/Q (DFFR_X2)" -> "issue_stage_i_i_scoreboard__60955_"
                    node_match = re.search(r'([^/]+)/[a-zA-Z0-9\[\]]+', name)
                    if node_match:
                        node_name = node_match.group(1)
                        node_count += 1
                            
                        # Convert node name to node ID using placedb mapping
                        if node_name in self.placedb.node_name2id_map:
                            node_id = self.placedb.node_name2id_map[node_name]
                        
                        # Initialize node entry if not exists
                        if node_id not in timing_features:
                            timing_features[node_id] = {
                            'caps': [],
                            'slews': [],
                            'delays': [],
                                    'node_name': node_name,  # Keep node name for reference
                            # 'fanouts': []
                            }
                        
                        # Add timing values
                        timing_features[node_id]['caps'].append(cap)
                        timing_features[node_id]['slews'].append(slew)
                        timing_features[node_id]['delays'].append(delay)
                        # timing_features[node_id]['fanouts'].append(fanout)
                        
                except (ValueError, IndexError):
                    # Skip lines that can't be parsed
                    continue
    
        # Calculate average values for each node
        for node_name, features in timing_features.items():
            features['avg_cap'] = np.mean(features['caps'])
            features['avg_slew'] = np.mean(features['slews'])
            features['avg_delay'] = np.mean(features['delays'])
            # features['avg_fanout'] = np.mean(features['fanouts'])
            
        
        logging.info(f"Parsed timing features for {len(timing_features)} nodes from {timing_report_path}")
        logging.info(f"Processed {path_count} timing paths with {node_count} timing entries")
        return timing_features

    def check_timing_coverage(self):
        """
        Check how many nodes and macros have timing information.
        """
        # Get total number of nodes
        total_nodes = self.placedb.num_physical_nodes - self.placedb.num_terminal_NIs
        
        print(f"\n=== TIMING COVERAGE ANALYSIS ===")
        print(f"Total positioned components: {total_nodes}")
        print(f"Nodes with timing information: {len(self.timing_features)}")
        print(f"Timing coverage: {len(self.timing_features)/total_nodes*100:.2f}%")
        
        # Identify macros based on area threshold (using same threshold as in the code)
        macro_threshold = 1000.0
        macro_count = 0
        macro_with_timing = 0
        
        # Check each movable node - iterate through actual node names like in clique_graph_construction
        for node_name in self.placedb.node_names:
            node_name_str = node_name.decode('utf-8')
            node_id = self.placedb.node_name2id_map[node_name_str]
            
            # Only check movable nodes (exclude IO ports)
            if node_id < (self.placedb.num_physical_nodes - self.placedb.num_terminal_NIs):
                # Calculate area
                area = self.placedb.node_size_x[node_id] * self.placedb.node_size_y[node_id]
                # Check if it's a macro
                if area > macro_threshold:
                    macro_count += 1
                    # Use node ID directly for timing features lookup
                    if node_id in self.timing_features:
                        macro_with_timing += 1
        
        print(f"\n=== MACRO ANALYSIS ===")
        print(f"Total macros (area > {macro_threshold}): {macro_count}")
        print(f"Macros with timing information: {macro_with_timing}")
        if macro_count > 0:
            print(f"Macro timing coverage: {macro_with_timing/macro_count*100:.2f}%")
        else:
            print("No macros found!")
        
        # Additional statistics
        non_macro_count = total_nodes - macro_count
        non_macro_with_timing = len(self.timing_features) - macro_with_timing
        
        print(f"\n=== DETAILED BREAKDOWN ===")
        print(f"Non-macro nodes: {non_macro_count}")
        print(f"Non-macro nodes with timing: {non_macro_with_timing}")
        if non_macro_count > 0:
            print(f"Non-macro timing coverage: {non_macro_with_timing/non_macro_count*100:.2f}%")
        
        # Show some examples of nodes with timing
        print(f"\n=== SAMPLE NODES WITH TIMING ===")
        timing_node_ids = list(self.timing_features.keys())[:5]
        for node_id in timing_node_ids:
            features = self.timing_features[node_id]
            node_name = features.get('node_name', f'node_{node_id}')
            print(f"  {node_name} (ID:{node_id}): avg_cap={features['avg_cap']:.3f}, avg_slew={features['avg_slew']:.3f}, avg_delay={features['avg_delay']:.3f}")
        
        # Debug: Show the range of node IDs in timing features
        if self.timing_features:
            timing_ids = list(self.timing_features.keys())
            print(f"\n=== TIMING FEATURES DEBUG ===")
            print(f"Node ID range in timing features: {min(timing_ids)} to {max(timing_ids)}")
            print(f"Total movable nodes: {self.placedb.num_movable_nodes}")
            print(f"Node IDs in timing features: {sorted(timing_ids)[:10]}...")  # Show first 10

    def clique_graph_construction(self):
        """
        Create a clique graph based on net connectivity with Manhattan distance as edge weights
        and timing analysis features (cap, slew, delay) for each node
        """
        G = nx.Graph()
        
        # nodes with position information
        node_attrs = {}
        positioned_nodes = []
        mean_node_area = 0.
        num = 0
        
        # Extract component positions directly from the database
        for node_name in self.placedb.node_names:
            node_name_str = node_name.decode('utf-8')
            node = self.placedb.node_name2id_map[node_name_str]
            
            if node < (self.placedb.num_physical_nodes - self.placedb.num_terminal_NIs):  # exclude IO ports
                G.add_node(node)
                node_area = self.placedb.node_size_x[node] * self.placedb.node_size_y[node]
                x, y = self.placedb.node_x[node], self.placedb.node_y[node]  # Get coordinates directly from db
                
                # Initialize timing features
                avg_cap = 0.0
                avg_slew = 0.0
                avg_delay = 0.0
                # avg_fanout = 0.0
                
                # Add timing features if available - use node ID lookup
                if self.timing_features and node in self.timing_features:
                    avg_cap = self.timing_features[node]['avg_cap']
                    avg_slew = self.timing_features[node]['avg_slew']
                    avg_delay = self.timing_features[node]['avg_delay']
                    # avg_fanout = self.timing_features[node]['avg_fanout']
                
                node_attrs[node] = {
                    "is_macro": False,
                    "area": node_area,
                    "name": node_name_str,
                    "x": x,
                    "y": y,
                    "avg_cap": avg_cap,
                    "avg_slew": avg_slew,
                    "avg_delay": avg_delay,
                    # "avg_fanout": avg_fanout
                }
                positioned_nodes.append(node)
                mean_node_area += node_area
                num += 1
                
        if num == 0:
            logging.warning("No positioned components found for clique graph construction")
            return G
            
        mean_node_area = mean_node_area / num
        # detect macros
        for node in positioned_nodes:
            node_area = self.placedb.node_size_x[node] * self.placedb.node_size_y[node]
            if (node_area > (mean_node_area * 10)) and (self.placedb.node_size_y[node] > (self.placedb.row_height * 2)):
                node_attrs[node]["is_macro"] = True
                
        nx.set_node_attributes(G, node_attrs)
        logging.info(f"Created clique graph with {len(positioned_nodes)} nodes")
        # Create edges based on net connectivity with Manhattan distance as weights
        edge_count = 0
        
        # Iterate through all nets to find connected components
        edges = []
        for net_name in self.placedb.net_names:
            net = self.placedb.net_name2id_map[net_name.decode('utf-8')]
            pins = self.placedb.net2pin_map[net]
            connected_nodes = []
            
            # Find all positioned nodes connected to this net
            for pin in pins:
                node_id = self.placedb.pin2node_map[pin]
                if node_id < (self.placedb.num_physical_nodes - self.placedb.num_terminal_NIs):  # exclude IO ports
                    connected_nodes.append(node_id)

            # Only create fully connected edges if the number of pins is less than 20
            if len(connected_nodes) < 20:
                # Use combinations to generate all pairs of nodes
                edges.extend(combinations(connected_nodes, r=2))
            else:
                # Skip creating clique edges for large nets to avoid excessive edge creation
                logging.debug(f"Skipping clique creation for net {net_name.decode('utf-8')} with {len(connected_nodes)} pins (>= 20)")  
        
        # Add all edges to the graph with Manhattan distance weights
        for edge in edges:
            node1, node2 = edge[0], edge[1]
            # Calculate Manhattan distance as edge weight
            x1, y1 = self.placedb.node_x[node1], self.placedb.node_y[node1]
            x2, y2 = self.placedb.node_x[node2], self.placedb.node_y[node2]
            manhattan_dist = abs(x1 - x2) + abs(y1 - y2)
            
            # Add edge with Manhattan distance as weight
            G.add_edge(node1, node2, weight=manhattan_dist)
            edge_count += 1

        logging.info(f"Created clique graph with {len(positioned_nodes)} nodes and {edge_count} edges based on net connectivity")
        logging.info(f"Edge weights represent Manhattan distances between connected components")
        if self.timing_features:
            logging.info(f"Included timing features (cap, slew, delay) for {len(self.timing_features)} nodes")
        
        return G
    
    def hierarchy_aware_graph_construction(self, G):
        """
        Perform hierarchy-aware edge contraction on the clique graph.
        Merges nodes that share the same hierarchy based on ascending edge weights.
        
        @param G: NetworkX graph with timing features
        @return: new contracted NetworkX graph
        """
        def get_hierarchy_level(node_name):
            """Extract hierarchy level from node name (split by double underscore)"""
            parts = node_name.split("__")
            return "__".join(parts[:-1])
        
        dsu = UnionFind(G.nodes())
        pq = []
        potential_edges = []
        
        cluster_props = {
            node: {
                'name': data['name'],
                'hierarchy': get_hierarchy_level(data['name']),
                'x': data['x'],
                'y': data['y'],
                'area': data['area'],
                'avg_cap': data['avg_cap'],
                'avg_slew': data['avg_slew'],
                'avg_delay': data['avg_delay'],
                'merged_from': [node]
            } for node, data in G.nodes(data=True)
        }
        for u, v, data in G.edges(data=True):
            heapq.heappush(pq, (data['weight'], u, v))

        while pq:
            weight, u, v = heapq.heappop(pq)
            
            # find the representative of u and v, since the original node may be contracted
            u = dsu[u]
            v = dsu[v]
            if u == v:
                continue
            
            props_u = cluster_props[u]
            props_v = cluster_props[v]
            
            if props_u['hierarchy'] == props_v['hierarchy']:
                size_u = len(props_u['merged_from'])
                size_v = len(props_v['merged_from'])
                
                if size_u + size_v <=2:
                    dsu.union(u, v)
                    new_root = dsu[u]
                    old_root = v if new_root == u else u
                    
                    new_props = cluster_props[new_root]
                    old_props = cluster_props[old_root]
                    
                    new_props['x'] = (new_props['x'] + old_props['x']) / 2
                    new_props['y'] = (new_props['y'] + old_props['y']) / 2
                    new_props['area'] = new_props['area'] + old_props['area']
                    new_props['avg_cap'] = (new_props['avg_cap'] + old_props['avg_cap']) / 2
                    new_props['avg_slew'] = (new_props['avg_slew'] + old_props['avg_slew']) / 2
                    new_props['avg_delay'] = (new_props['avg_delay'] + old_props['avg_delay']) / 2
                    new_props['merged_from'] = [*new_props['merged_from'], *old_props['merged_from']]
                    del cluster_props[old_root]
                else:
                    potential_edges.append((weight, u, v))
            else:
                potential_edges.append((weight, u, v))
        
        G_contracted = nx.Graph()
        for root_id, properties in cluster_props.items():
            G_contracted.add_node(root_id, **properties)
        
        added_edges = set()
        potential_edges.sort()
        
        for _, u, v in potential_edges:
            root_u = dsu[u]
            root_v = dsu[v]
            edge_repr = tuple(sorted((root_u, root_v)))
            
            if root_u != root_v and edge_repr not in added_edges:
                final_props_u = cluster_props[root_u]
                final_props_v = cluster_props[root_v]
                new_weight = abs(final_props_u['x'] - final_props_v['x']) + abs(final_props_u['y'] - final_props_v['y'])
                G_contracted.add_edge(root_u, root_v, weight=new_weight)
                added_edges.add(edge_repr)
                
        logging.info(f"Final robust contraction completed: {G.number_of_nodes()} -> {G_contracted.number_of_nodes()} nodes, {G.number_of_edges()} -> {G_contracted.number_of_edges()} edges")
        return G_contracted

    def generate_embeddings(self, G, output_dir="./tpgnn_results", epochs=50, 
                           initial_lr=0.001, lr_decay_factor=0.95, lr_decay_patience=5):
        """
        Generate node embeddings using TP-GNN: create pipeline, train model, and extract embeddings.
        
        Args:
            G: NetworkX graph with timing features
            output_dir: Directory to save results
            epochs: Number of training epochs
            initial_lr: Initial learning rate
            lr_decay_factor: Factor by which to reduce learning rate (0.95 = 5% reduction)
            lr_decay_patience: Number of epochs to wait before reducing learning rate
            
        Returns:
            Tuple of (trained_model, results_dict)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info("Starting TP-GNN embedding generation...")
        
        # Step 1: Convert graph to PyTorch Geometric format
        converter = NetworkXToPyGConverter()
        data = converter.convert_graph(G)
        data = data.to(self.device)
        logging.info(f"Converted graph: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
        
        # Step 2: Initialize components following TP-GNN paper parameters
        model = TPGNNModel(input_dim=data.x.size(1))
        self.tpgnn_model = model.to(self.device)
        
        # Step 3: Initialize random walk and trainer
        random_walk = TPGNNRandomWalk(walk_length=5, num_walks=5)  # LRW=5, NRW=5 from paper
        trainer = TPGNNTrainer(model, random_walk, self.device, 
                              initial_lr=initial_lr, 
                              lr_decay_factor=lr_decay_factor, 
                              lr_decay_patience=lr_decay_patience)
        
        # Step 4: Training loop (unsupervised contrastive learning)
        logging.info(f"Starting TP-GNN unsupervised training for {epochs} epochs...")
        logging.info(f"Learning rate scheduler: initial_lr={initial_lr}, decay_factor={lr_decay_factor}, patience={lr_decay_patience}")
        training_losses = []
        
        for epoch in range(epochs):
            # Train for one epoch using only contrastive loss
            losses = trainer.train_epoch(data, num_neg_samples=30)
            training_losses.append(losses)
            
            # Update learning rate scheduler
            trainer.update_learning_rate(losses['total_loss'])
            current_lr = trainer.get_current_lr()
            
            print(f"Training epoch {epoch}: Contrastive Loss = {losses['contrastive_loss']:.4f}, LR = {current_lr:.6f}")
            
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: Contrastive Loss = {losses['contrastive_loss']:.4f}, LR = {current_lr:.6f}")
        
        # Step 5: Final evaluation (embedding quality)
        logging.info("Performing final evaluation...")
        eval_metrics = trainer.evaluate(data)
        logging.info(f"Final evaluation metrics: {eval_metrics}")
        
        # Step 6: Generate final embeddings
        model.eval()
        with torch.no_grad():
            final_embeddings = model(data.to(self.device))
        
        # Step 7: Save results
        results = {
            'training_losses': training_losses,
            'eval_metrics': eval_metrics,
            'final_embeddings': final_embeddings.cpu().numpy(),
            'node_names': data.node_names
        }
        
        # Save embeddings and results
        np.save(os.path.join(output_dir, 'node_embeddings.npy'), final_embeddings.cpu().numpy())
        
        # Save model checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': model.input_dim,
                'hidden_dim': model.hidden_dim,
                'output_dim': model.output_dim,
                'num_layers': model.num_layers
            }
        }, os.path.join(output_dir, 'tpgnn_checkpoint.pt'))
        
        logging.info(f"Results saved to {output_dir}")
        
        return model, results


    def analyze_gnn_embeddings(self, embeddings, node_names, is_macro, output_dir="./gnn_results"):
        """
        Analyze and visualize GNN embeddings.
        
        @param embeddings: Node embeddings from GNN
        @param node_names: List of node names
        @param is_macro: Boolean tensor indicating macro nodes
        @param output_dir: Directory to save analysis results
        """
        
        logging.info("Analyzing GNN embeddings...")
        
        # Convert to numpy if tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(is_macro, torch.Tensor):
            is_macro = is_macro.cpu().numpy()
        
        # Basic statistics
        print(f"\n=== GNN Embedding Analysis ===")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Number of nodes: {embeddings.shape[0]}")
        print(f"Number of macro nodes: {np.sum(is_macro)}")
        print(f"Embedding statistics - Mean: {np.mean(embeddings):.4f}, Std: {np.std(embeddings):.4f}")
        
        # Analyze macro vs non-macro embeddings
        macro_embeddings = embeddings[is_macro]
        non_macro_embeddings = embeddings[~is_macro]
        
        if len(macro_embeddings) > 0:
            print(f"Macro embedding statistics - Mean: {np.mean(macro_embeddings):.4f}, Std: {np.std(macro_embeddings):.4f}")
        if len(non_macro_embeddings) > 0:
            print(f"Non-macro embedding statistics - Mean: {np.mean(non_macro_embeddings):.4f}, Std: {np.std(non_macro_embeddings):.4f}")
        
        # Compute pairwise similarities for sample nodes
        if embeddings.shape[0] > 1:
            from sklearn.metrics.pairwise import cosine_similarity
            sample_indices = np.random.choice(embeddings.shape[0], min(10, embeddings.shape[0]), replace=False)
            sample_embeddings = embeddings[sample_indices]
            similarities = cosine_similarity(sample_embeddings)
            
            print(f"Sample cosine similarities (mean): {np.mean(similarities[np.triu_indices_from(similarities, k=1)]):.4f}")
        
        # Try to perform dimensionality reduction for visualization
        try:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            # PCA
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            # Create visualization
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.scatter(embeddings_2d[~is_macro, 0], embeddings_2d[~is_macro, 1], 
                    alpha=0.6, label='Standard cells', s=20)
            if np.sum(is_macro) > 0:
                plt.scatter(embeddings_2d[is_macro, 0], embeddings_2d[is_macro, 1], 
                        alpha=0.8, label='Macro cells', s=50, color='red')
            plt.title('PCA of GNN Embeddings')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.legend()
            
            # t-SNE (if dataset is not too large)
            if embeddings.shape[0] <= 1000:
                tsne = TSNE(n_components=2, random_state=42)
                embeddings_tsne = tsne.fit_transform(embeddings)
                
                plt.subplot(1, 2, 2)
                plt.scatter(embeddings_tsne[~is_macro, 0], embeddings_tsne[~is_macro, 1], 
                        alpha=0.6, label='Standard cells', s=20)
                if np.sum(is_macro) > 0:
                    plt.scatter(embeddings_tsne[is_macro, 0], embeddings_tsne[is_macro, 1], 
                            alpha=0.8, label='Macro cells', s=50, color='red')
                plt.title('t-SNE of GNN Embeddings')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'embedding_visualization.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Embedding visualization saved to {os.path.join(output_dir, 'embedding_visualization.png')}")
            
        except ImportError:
            logging.warning("sklearn and/or matplotlib not available for embedding analysis")


    def plot_training_losses(self, training_losses, output_dir="./tpgnn_results"):
        """
        Plot and save the GNN training losses.
        
        @param training_losses: List of loss dictionaries from training
        @param output_dir: Directory to save the plot
        """
        if not training_losses:
            logging.warning("No training losses to plot")
            return
        
        try:
            # Extract epochs and losses
            epochs = list(range(len(training_losses)))
            contrastive_losses = [loss['contrastive_loss'] for loss in training_losses]
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, contrastive_losses, 'b-', linewidth=2, label='Contrastive Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('TP-GNN Training Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add some styling
            plt.tight_layout()
            
            # Save the plot
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'training_loss_plot.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Training loss plot saved to {plot_path}")
            print(f"Training loss plot saved to {plot_path}")
            
        except Exception as e:
            logging.error(f"Failed to create training loss plot: {e}")

    def total_area(self, G, cut):
        upper_area = 0.
        bot_area = 0.
        for node in cut:
            if not G.nodes[node]['is_macro']:
                upper_area += G.nodes[node]['area']
            else:
                upper_area += G.nodes[node]['area']

        for node in G.nodes - cut:
            if not G.nodes[node]['is_macro']:
                bot_area += G.nodes[node]['area']
            else:
                bot_area += G.nodes[node]['area']
        return upper_area, bot_area
    
    def partition(self, G_base, G_contracted, embeddings, output_dir="./tpgnn_results"):
        """
        Partition contracted nodes into two clusters using weighted K-means based on GNN embeddings.
        Map results back to base graph and return partition for base nodes and macros.
        
        @param G_base: Base NetworkX graph (before contraction)
        @param G_contracted: Contracted NetworkX graph
        @param embeddings: GNN embeddings for contracted nodes
        @param output_dir: Directory to save partition results
        @return: Tuple of (bottom_die_node_ids, upper_die_macro_names)
                - bottom_die_node_ids: set of node IDs (standard cells + macros) placed in bottom die
                - upper_die_macro_names: list of macro names that belong to upper die
        """
        logging.info("Starting weighted K-means partitioning...")
        
        # Convert embeddings to numpy if tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # Get node areas for weighting
        node_areas = []
        
        for i, node_id in enumerate(G_contracted.nodes()):
            node_data = G_contracted.nodes[node_id]
            area = node_data.get('area', 0)
            node_areas.append(area)
            
        node_areas = np.array(node_areas)
        
        # Normalize areas for weighting (avoid division by zero)
        if np.max(node_areas) > 0:
            weights = node_areas / np.max(node_areas)
        else:
            weights = np.ones(len(node_areas))
        
        # Apply weights to embeddings (element-wise multiplication)
        weighted_embeddings = embeddings * weights.reshape(-1, 1)
        
        # Perform K-means clustering (k=2)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(weighted_embeddings)
        
        # Calculate silhouette score for cluster quality
        silhouette_avg = silhouette_score(weighted_embeddings, cluster_labels)
        logging.info(f"K-means clustering completed with silhouette score: {silhouette_avg:.4f}")
        
        # Fixed assignment: cluster 0 = bottom, cluster 1 = top
        top_cluster = 1
        bottom_cluster = 0
        
        # Map contracted macro partition results back to base macros
        upper_die_base_macro_ids = set()
        upper_die_macro_names = []
        bottom_die_base_macro_ids = set()
        bottom_die_macro_names = []
        
        for i, node_id in enumerate(G_contracted.nodes()):
            node_data = G_contracted.nodes[node_id]
            cluster_id = cluster_labels[i]
            merged_from = node_data.get('merged_from')
            for base_node_id in merged_from:
                if G_base.nodes[base_node_id].get('is_macro'):
                    if cluster_id == top_cluster:
                        upper_die_base_macro_ids.add(base_node_id)
                        base_node_data = G_base.nodes[base_node_id]
                        upper_die_macro_names.append(base_node_data.get('name'))     
                    else:  
                        bottom_die_base_macro_ids.add(base_node_id)
                        base_node_data = G_base.nodes[base_node_id]
                        bottom_die_macro_names.append(base_node_data.get('name'))
        
        # Refine: move least-connected bottom macros to upper until bottom < 25
        if len(bottom_die_base_macro_ids) >= 25:
            logging.info(f"Refining partition: {len(bottom_die_base_macro_ids)} bottom macros >= 25; moving least-connected to upper die")
            # Build (macro_id, degree) list for bottom macros
            bottom_macro_degrees = []
            for macro_id in bottom_die_base_macro_ids:
                try:
                    deg = G_base.degree(macro_id)
                except Exception:
                    deg = 0
                bottom_macro_degrees.append((macro_id, deg))
            # Sort by degree ascending (least connected first)
            bottom_macro_degrees.sort(key=lambda x: x[1])
            # Move until constraint satisfied
            idx = 0
            while len(bottom_die_base_macro_ids) >= 25 and idx < len(bottom_macro_degrees):
                macro_id, deg = bottom_macro_degrees[idx]
                idx += 1
                if macro_id not in bottom_die_base_macro_ids:
                    continue
                bottom_die_base_macro_ids.discard(macro_id)
                upper_die_base_macro_ids.add(macro_id)
                base_node_data = G_base.nodes[macro_id]
                macro_name = base_node_data.get('name')
                if macro_name not in upper_die_macro_names:
                    upper_die_macro_names.append(macro_name)
                logging.info(f"Moved macro {macro_name} (ID:{macro_id}) with degree {deg} to upper die")
            logging.info(f"Refinement complete: bottom macros now {len(bottom_die_base_macro_ids)} (<25)")
        
        # Calculate bottom die node IDs: all base nodes minus upper die macros
        bottom_die_node_ids = G_base.nodes() - upper_die_base_macro_ids
        
        # Save partition results for debugging/analysis
        os.makedirs(output_dir, exist_ok=True)
        partition_results = {
            'bottom_die_node_ids': list(bottom_die_node_ids),
            'upper_die_macro_names': upper_die_macro_names,
            'upper_die_base_macro_ids': list(upper_die_base_macro_ids),
            'cluster_labels': cluster_labels,
            'top_cluster_id': top_cluster,
            'bottom_cluster_id': bottom_cluster,
            'silhouette_score': silhouette_avg,
            'cluster_centers': kmeans.cluster_centers_,
            'node_areas': node_areas,
            'weights': weights
        }
        np.save(os.path.join(output_dir, 'partition_results.npy'), partition_results)
        
        logging.info(f"Partition results saved to {output_dir}")
        logging.info(f"Macros in upper die: {len(upper_die_base_macro_ids)}")
        logging.info(f"Macros in bottom die: {len(bottom_die_base_macro_ids)}")
        logging.info(f"Cut size: {nx.algorithms.cut_size(G_base, upper_die_base_macro_ids)}")
        upper_die_area, bottom_die_area = self.total_area(G_base, upper_die_base_macro_ids)
        logging.info(f"Upper die area: {upper_die_area}")
        logging.info(f"Bottom die area: {bottom_die_area}")
        logging.info(f"Area difference: {upper_die_area - bottom_die_area}")
        
        return bottom_die_node_ids, upper_die_macro_names


if __name__ == "__main__":
    """
    @brief main function to test TP-GNN pipeline.
    """
    logging.root.name = 'TPGNN'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    
    params = Params.Params()
    params.printWelcome()
    
    if len(sys.argv) == 1 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        params.printHelp()
        exit()
    elif len(sys.argv) != 2:
        logging.error("One input parameters in json format is required")
        params.printHelp()
        exit()

    # load parameters
    params.load(sys.argv[1])
    # params.load("test/or_2D/ariane133_2D.json")
    logging.info("parameters loaded successfully")
    
    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    # Initialize placement database
    tt = time.time()
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    
    logging.info(f"Found {placedb.num_physical_nodes - placedb.num_terminal_NIs} positioned components")
    
    # Initialize TPGNN with placement database
    tpgnn = TPGNN(placedb)
    
    # Load timing features - construct path relative to current working directory
    timing_report_path = os.path.join(os.getcwd(), params.timing_report_input)
    tpgnn.timing_features = tpgnn.parse_timing_report(timing_report_path)
    
    # Check timing coverage
    tpgnn.check_timing_coverage()
    
    # Build clique graph
    G_base = tpgnn.clique_graph_construction()
    
    # Print basic statistics of original graph
    print(f"\n=== Original Graph Statistics ===")
    print(f"Nodes: {G_base.number_of_nodes()}")
    print(f"Edges: {G_base.number_of_edges()}")
    
    # Apply hierarchy-aware edge contraction
    G_contracted = tpgnn.hierarchy_aware_graph_construction(G_base)
    
    # Print basic statistics of contracted graph
    print(f"\n=== Contracted Graph Statistics ===")
    print(f"Nodes: {G_contracted.number_of_nodes()}")
    print(f"Edges: {G_contracted.number_of_edges()}")
    print(f"Contraction ratio: {G_contracted.number_of_nodes() / G_base.number_of_nodes():.2f}")
    
    if G_base.number_of_edges() > 0:
        edge_weights = [data['weight'] for _, _, data in G_base.edges(data=True)]
        print(f"Edge weights - Min: {min(edge_weights):.2f}, Max: {max(edge_weights):.2f}, Avg: {np.mean(edge_weights):.2f}")
    
    print(f"\n=== Running TP-GNN ===")
    
    # Use the contracted graph for TP-GNN training (smaller and more manageable)
    tpgnn_model, tpgnn_results = tpgnn.generate_embeddings(G_contracted, 
                                           output_dir="./tpgnn_results", 
                                           epochs=1,
                                           initial_lr=0.001,
                                           lr_decay_factor=0.95,
                                           lr_decay_patience=5)
    
    # Analyze the embeddings
    tpgnn.analyze_gnn_embeddings(
        tpgnn_results['final_embeddings'],
        tpgnn_results['node_names'],
        # Create is_macro array based on area threshold
        np.array([G_contracted.nodes[node_id].get('area', 0) > 1000 
                    for node_id in G_contracted.nodes()]),
        output_dir="./tpgnn_results"
    )
    
    # Plot and save training losses
    tpgnn.plot_training_losses(tpgnn_results['training_losses'], output_dir="./tpgnn_results")
    
    print(f"TP-GNN training completed successfully!")
    print(f"Final contrastive loss: {tpgnn_results['training_losses'][-1]['contrastive_loss']:.4f}")
    print(f"Generated {tpgnn_results['final_embeddings'].shape[1]}-dimensional embeddings for {tpgnn_results['final_embeddings'].shape[0]} nodes")
    
        # Run partitioning using GNN embeddings
    print(f"\n=== Running Weighted K-means Partitioning ===")
    bottom_die_node_ids, upper_die_macro_names = tpgnn.partition(
        G_base,
        G_contracted, 
        tpgnn_results['final_embeddings'], 
        tpgnn_results['node_names'],
        output_dir="./tpgnn_results"
    )
    
    # Print final summary
    print(f"\n=== Final Summary ===")
    print(f"Total contracted nodes: {len(G_contracted.nodes())}")
    print(f"Total base nodes: {len(G_base.nodes())}")
    print(f"Nodes in bottom die: {len(bottom_die_node_ids)}")
    print(f"Macros in upper die: {len(upper_die_macro_names)}")
    print(f"Upper die macro names: {upper_die_macro_names}")
    print(f"Partition results saved to: ./tpgnn_results/partition_results.npy")
    
    logging.info("Total execution time: %.3f seconds" % (time.time() - tt))