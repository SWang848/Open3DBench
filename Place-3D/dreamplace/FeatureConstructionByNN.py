from __future__ import annotations

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np

from GraphConstruction import (
    HierarchyEncoder,
    build_static_graph,
    graph_to_pyg_base,
    update_die_in_pyg,
)
from Regression import GraphRegressor, build_dataset

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


class FeatureExtractor(nn.Module):
    """
    Wrapper around GraphRegressor to extract features before the final prediction layer.
    """
    def __init__(self, base_model: GraphRegressor):
        super().__init__()
        self.conv1 = base_model.conv1
        self.conv2 = base_model.conv2
        self.norm1 = base_model.norm1
        self.norm2 = base_model.norm2
        self.lin1 = base_model.lin1
        self.norm3 = base_model.norm3
        # We don't include lin2 (the final prediction layer)
    
    def forward(self, data: Data) -> torch.Tensor:
        """Extract features before the final linear layer."""
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        x = F.relu(self.norm1(self.conv1(x, edge_index, edge_weight=edge_weight)))
        x = F.relu(self.norm2(self.conv2(x, edge_index, edge_weight=edge_weight)))
        x = global_mean_pool(x, batch)
        x = F.relu(self.norm3(self.lin1(x)))
        # Return features before lin2 (final prediction layer)
        return x


def load_checkpoint(checkpoint_path: Path) -> Tuple[Dict, Dict[str, torch.Tensor], int, torch.Tensor, torch.Tensor]:
    """
    Load checkpoint and return model config, state dict, and normalization stats.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model_state = checkpoint["model_state_dict"]
    input_dim = checkpoint["input_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    label_mean = checkpoint["label_mean"]
    label_std = checkpoint["label_std"]
    val_loss = checkpoint.get("val_loss", None)
    epoch = checkpoint.get("epoch", None)
    
    config = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
    }
    
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    logging.info(f"  Model config: input_dim={input_dim}, hidden_dim={hidden_dim}")
    if val_loss is not None:
        logging.info(f"  Validation loss: {val_loss:.4f}")
    if epoch is not None:
        logging.info(f"  Epoch: {epoch}")
    
    return config, model_state, epoch, label_mean, label_std


def extract_features_and_predictions(
    candidates: List[Tuple[str, List[List[int]], Tuple[float, float]]],
    placedb: PlaceDB.PlaceDB,
    feature_extractor: FeatureExtractor,
    full_model: GraphRegressor,
    device: torch.device,
    label_mean: torch.Tensor,
    label_std: torch.Tensor,
    batch_size: int = 8,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Extract features and predictions for all candidate solutions.
    Returns a tuple of (features_dict, predictions_dict) where:
    - features_dict: maps candidate keys to feature vectors (before final layer)
    - predictions_dict: maps candidate keys to denormalized predictions (cut_size, area_imbalance)
    """
    # Build base graph and hierarchy embeddings
    logging.info("Building base graph and hierarchy embeddings...")
    hierarchy_encoder = HierarchyEncoder(placedb)
    hierarchy_embeddings = hierarchy_encoder.build_hierarchy_embeddings()
    base_graph = build_static_graph(placedb, hierarchy_embeddings)
    base_pyg_data, node_to_idx = graph_to_pyg_base(base_graph)
    
    # Create dataset-like structure for candidates
    candidate_graphs = []
    candidate_keys = []
    
    logging.info(f"Processing {len(candidates)} candidates...")
    for key, partition, _ in candidates:
        pyg_graph = update_die_in_pyg(base_pyg_data, partition, node_to_idx)
        candidate_graphs.append(pyg_graph)
        candidate_keys.append(key)
    
    # Create DataLoader for batch processing
    # IMPORTANT: num_workers=0 ensures order preservation (no multiprocessing)
    # shuffle=False also preserves order, but num_workers=0 is critical for deterministic ordering
    loader = DataLoader(candidate_graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Extract features and predictions
    feature_extractor.eval()
    full_model.eval()
    all_features = []
    all_predictions = []
    
    label_mean = label_mean.to(device)
    label_std = label_std.to(device)
    
    # Track the number of items processed to verify order preservation
    items_processed = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_size_actual = batch.num_graphs
            
            # Extract features (before final layer)
            features = feature_extractor(batch)
            all_features.append(features.cpu().numpy())
            
            # Get predictions (full model output)
            normalized_preds = full_model(batch)
            # Denormalize predictions
            denormalized_preds = normalized_preds * label_std + label_mean
            all_predictions.append(denormalized_preds.cpu().numpy())
            
            items_processed += batch_size_actual
    
    # Verify we processed all items
    assert items_processed == len(candidate_keys), \
        f"Order mismatch: processed {items_processed} items but have {len(candidate_keys)} keys"
    
    # Concatenate all features and predictions
    all_features = np.concatenate(all_features, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Verify shapes match before zipping
    assert len(all_features) == len(candidate_keys), \
        f"Feature count mismatch: {len(all_features)} features vs {len(candidate_keys)} keys"
    assert len(all_predictions) == len(candidate_keys), \
        f"Prediction count mismatch: {len(all_predictions)} predictions vs {len(candidate_keys)} keys"
    
    # Create dictionaries mapping keys to features and predictions
    # Order is preserved: DataLoader with shuffle=False and num_workers=0 processes items sequentially
    features_dict = {key: features for key, features in zip(candidate_keys, all_features)}
    predictions_dict = {key: preds for key, preds in zip(candidate_keys, all_predictions)}
    
    logging.info(f"Extracted features and predictions for {len(features_dict)} candidates")
    logging.info(f"Feature shape: {all_features.shape[1]}")
    logging.info(f"Prediction shape: {all_predictions.shape[1]} (cut_size, area_imbalance)")
    
    return features_dict, predictions_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract features from trained model for D-optimal design.")
    parser.add_argument("params", type=Path, help="Path to params JSON used by PlaceDB.")
    parser.add_argument("hmsa_results", type=Path, help="Path to hmsa_results.json containing candidates.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model checkpoint. Default: regression_results/{case_name}/regressor_best.pt")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for feature extraction.")
    parser.add_argument("--output", type=Path, default=None, help="Path to save extracted features. Default: regression_results/{case_name}/candidate_features.npy")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    case_name = args.params.stem
    out_dir = Path("./regression_results") / case_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = args.checkpoint or (out_dir / "regressor_best.pt")
    output_path = args.output or (out_dir / "candidate_features.npy")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if not args.hmsa_results.exists():
        raise FileNotFoundError(f"HMSA results file not found: {args.hmsa_results}")
    
    # Load PlaceDB
    logging.info("Loading PlaceDB...")
    params = Params.Params()
    params.load(str(args.params))
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)
    
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    
    # Load candidates
    logging.info(f"Loading candidates from {args.hmsa_results}...")
    candidates = load_candidates_from_json(args.hmsa_results)
    logging.info(f"Loaded {len(candidates)} candidates")
    
    if len(candidates) == 0:
        raise ValueError("No candidates found in HMSA results file.")
    
    # Load checkpoint
    config, model_state, epoch, label_mean, label_std = load_checkpoint(checkpoint_path)
    
    # Create full model to load weights
    full_model = GraphRegressor(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"])
    full_model.load_state_dict(model_state)
    
    # Create feature extractor (without final layer)
    feature_extractor = FeatureExtractor(full_model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device)
    full_model = full_model.to(device)
    
    # Extract features and predictions for all candidates
    features_dict, predictions_dict = extract_features_and_predictions(
        candidates,
        placedb,
        feature_extractor,
        full_model,
        device,
        label_mean,
        label_std,
        batch_size=args.batch_size,
    )
    
    # Save features and predictions
    # Save as dictionary with keys, features, and predictions
    output_data = {
        "candidate_keys": list(features_dict.keys()),
        "features": np.array([features_dict[key] for key in features_dict.keys()]),
        "predictions": np.array([predictions_dict[key] for key in predictions_dict.keys()]),
        "feature_dim": list(features_dict.values())[0].shape[0] if features_dict else 0,
        "prediction_dim": 2,  # cut_size, area_imbalance
    }
    
    np.save(output_path, output_data, allow_pickle=True)
    logging.info(f"Saved features and predictions to {output_path}")
    logging.info(f"  Number of candidates: {len(output_data['candidate_keys'])}")
    logging.info(f"  Feature dimension: {output_data['feature_dim']}")
    logging.info(f"  Features shape: {output_data['features'].shape}")
    logging.info(f"  Predictions shape: {output_data['predictions'].shape}")
    
    # Print sample predictions
    if len(predictions_dict) > 0:
        i = 0
        while i < 10:
            sample_key = list(predictions_dict.keys())[i]
            sample_pred = predictions_dict[sample_key]
            logging.info(f"  Sample prediction for '{sample_key}': cut_size={sample_pred[0]:.2f}, area_imbalance={sample_pred[1]:.2f}")
            i += 1


if __name__ == "__main__":
    main()

