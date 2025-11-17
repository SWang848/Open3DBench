from __future__ import annotations

import argparse
import json
import os
import sys
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from GraphConstruction import (
    HierarchyEncoder,
    build_static_graph,
    graph_to_pyg_base,
    update_die_in_pyg,
)
from tqdm import tqdm
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from dreamplace import Params, PlaceDB


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hmsa_results(results_path: Path) -> List[Tuple[List[List[int]], Tuple[float, float]]]:
    with open(results_path, "r") as fp:
        solutions = json.load(fp)

    if not isinstance(solutions, list):
        raise ValueError("Expected HMSA results file to contain a list of solution dicts.")

    dataset = []
    for entry in solutions:
        if not isinstance(entry, dict):
            continue

        raw_solution = entry.get("solution", [[], []])
        cost = entry.get("cost", [0.0, 0.0])
        if not isinstance(raw_solution, Sequence) or len(raw_solution) != 2:
            continue

        lower_ids = [int(node_id) for node_id in raw_solution[0]]
        upper_ids = [int(node_id) for node_id in raw_solution[1]]
        cut_size = float(cost[0])
        area_imbalance = float(cost[1])
        dataset.append(([lower_ids, upper_ids], (cut_size, area_imbalance)))
    return dataset


class HMSADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_pyg_data: Data,
        node_to_idx: Dict[int, int],
        solutions: List[Tuple[List[List[int]], Tuple[float, float]]],
        label_mean: Optional[torch.Tensor] = None,
        label_std: Optional[torch.Tensor] = None,
    ):
        self.base_pyg_data = base_pyg_data
        self.node_to_idx = node_to_idx
        self.solutions = solutions
        
        # Compute normalization stats if not provided
        if label_mean is None or label_std is None:
            labels = torch.tensor([label for _, label in solutions], dtype=torch.float32)
            self.label_mean = labels.mean(dim=0)
            self.label_std = labels.std(dim=0)
            # Avoid division by zero for constant features
            self.label_std = torch.clamp(self.label_std, min=1e-8)
        else:
            self.label_mean = label_mean
            self.label_std = label_std

    def __len__(self) -> int:
        return len(self.solutions)

    def __getitem__(self, idx: int) -> Data:
        partition, label = self.solutions[idx]
        # Generate graph on-the-fly instead of storing all in memory
        pyg_graph = update_die_in_pyg(self.base_pyg_data, partition, self.node_to_idx)
        # Normalize labels to same scale for balanced gradient updates
        label_tensor = torch.tensor(label, dtype=torch.float32)
        normalized_label = (label_tensor - self.label_mean) / self.label_std
        # y should be 1D (2,) - PyG DataLoader will stack to (batch_size, 2)
        pyg_graph.y = normalized_label
        return pyg_graph
    
    def denormalize(self, normalized_labels: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to original scale."""
        return normalized_labels * self.label_std + self.label_mean


class GraphRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 2)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        x = F.relu(self.norm1(self.conv1(x, edge_index, edge_weight=edge_weight)))
        x = F.relu(self.norm2(self.conv2(x, edge_index, edge_weight=edge_weight)))
        x = global_mean_pool(x, batch)
        x = F.relu(self.norm3(self.lin1(x)))
        return self.lin2(x)


def split_dataset(dataset: HMSADataset, train_ratio: float = 0.8, seed: int = 42):
    set_seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_idx = max(1, int(len(indices) * train_ratio))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:] or indices[:1]

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    return train_subset, val_subset


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    dataset: HMSADataset,
    epochs: int = 100,
    lr: float = 1e-3,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_state: Dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    
    # Log normalization stats
    logging.info(f"Label normalization stats:")
    logging.info(f"  Mean: cut_size={dataset.label_mean[0]:.2f}, area_imbalance={dataset.label_mean[1]:.2f}")
    logging.info(f"  Std:  cut_size={dataset.label_std[0]:.2f}, area_imbalance={dataset.label_std[1]:.2f}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch)
            # Ensure batch.y is (batch_size, 2) - PyG may concatenate to (batch_size*2,) so reshape
            if batch.y.dim() == 1:
                batch.y = batch.y.view(-1, 2)
            loss = criterion(preds, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val epoch {epoch}", leave=False):
                batch = batch.to(device)
                preds = model(batch)
                # Ensure batch.y is (batch_size, 2) - PyG may concatenate to (batch_size*2,) so reshape
                if batch.y.dim() == 1:
                    batch.y = batch.y.view(-1, 2)
                loss = criterion(preds, batch.y)
                val_loss += loss.item() * batch.num_graphs
        avg_val_loss = val_loss / len(val_loader.dataset)

        logging.info(
            f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Save checkpoint immediately when better model is found
            if checkpoint_path is not None:
                checkpoint_data = {
                    "model_state_dict": best_state,
                    "val_loss": best_val_loss,
                    "epoch": epoch,
                    "input_dim": model.conv1.in_channels,
                    "hidden_dim": model.conv1.out_channels,
                    "label_mean": dataset.label_mean.cpu(),
                    "label_std": dataset.label_std.cpu(),
                }
                torch.save(checkpoint_data, checkpoint_path)
                logging.info(f"Saved improved checkpoint (val_loss={best_val_loss:.4f}, epoch={epoch}) to {checkpoint_path}")

    return best_val_loss, best_state or {k: v.cpu().clone() for k, v in model.state_dict().items()}


def build_dataset(
    placedb: PlaceDB.PlaceDB,
    solutions: List[Tuple[List[List[int]], Tuple[float, float]]]
) -> HMSADataset:
    """
    Build a lazy dataset that generates graphs on-the-fly to avoid OOM.
    Only the base graph and solutions list are kept in memory.
    """
    logging.info("Building base graph and hierarchy embeddings...")
    hierarchy_encoder = HierarchyEncoder(placedb)
    hierarchy_embeddings = hierarchy_encoder.build_hierarchy_embeddings()
    base_graph = build_static_graph(placedb, hierarchy_embeddings)
    
    # Convert base graph to PyG once - this is the only graph kept in memory
    base_pyg_data, node_to_idx = graph_to_pyg_base(base_graph)
    
    logging.info(f"Created lazy dataset with {len(solutions)} solutions")
    # Return lazy dataset - graphs will be generated on-demand during training
    return HMSADataset(base_pyg_data, node_to_idx, solutions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train regression model on HMSA search results.")
    parser.add_argument("params", type=Path, help="Path to params JSON used by PlaceDB.")
    parser.add_argument("hmsa_results", type=Path, help="Path to hmsa_results.json generated by HMSA.py.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Path to save best model checkpoint.")
    return parser.parse_args()


def main() -> None:
    
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    set_seed()

    case_name = args.params.stem
    out_dir = Path("./regression_results") / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint_path or (out_dir / "regressor_best.pt")

    params = Params.Params()
    params.load(str(args.params))

    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    placedb = PlaceDB.PlaceDB()
    placedb(params)
    solutions = load_hmsa_results(args.hmsa_results)
    if not solutions:
        raise ValueError("No solutions found in HMSA results file.")

    dataset = build_dataset(placedb, solutions)
    train_subset, val_subset = split_dataset(dataset, args.train_ratio)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_graph: Data = dataset[0]
    model = GraphRegressor(sample_graph.num_node_features).to(device)

    best_val_loss, best_state = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        dataset=dataset,
        epochs=args.epochs,
        lr=args.learning_rate,
        checkpoint_path=checkpoint_path,
    )

    # Final confirmation - checkpoint should already be saved, but log completion
    logging.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

