from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from algorithms.dopp.loaders import load_features_from_file, load_fitness_scores_from_csv


def load_d_optimal_selection(
    d_opt_results_path: Path,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load D-optimal design weights and selected candidate keys from results file.
    Only loads candidates with non-zero weights (selected_candidates).
    
    Args:
        d_opt_results_path: Path to D-optimal results .npy file
    
    Returns:
        Tuple of (weights, selected_candidate_keys) where:
        - weights: selected weights renormalized to sum to 1
        - selected_candidate_keys: list of candidate keys with non-zero weights
        and they are one-to-one mapped
    """
    data = np.load(d_opt_results_path, allow_pickle=True).item()
    
    selected_weights = np.asarray(data["normalized_weights"], dtype=np.float32)
    candidate_keys = [str(key) for key in data["candidate_keys"]]
    selected_candidate_keys = [candidate_keys[int(idx)] for idx in data["selected_indices"]]
    return selected_weights, selected_candidate_keys


def train_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray = None,
    X_all: np.ndarray = None,
    y_all: np.ndarray = None,
) -> Tuple[LinearRegression, Dict]:
    """
    Train a linear regression model on all provided data.
    
    Args:
        X: Feature matrix (N, d) for training
        y: Target values (N,) for training
        sample_weights: Optional sample weights for weighted regression (N,)
        X_all: Optional full feature matrix for evaluation (if whole dataset is provided)
        y_all: Optional full target values for evaluation (if whole dataset is provided)
    
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logging.info(f"Training on all {X.shape[0]} samples")
    
    if sample_weights is not None:
        logging.info(f"Using weighted regression with {np.sum(sample_weights > 1e-6)} non-zero weights")
    
    # Train model on all data
    # model = Ridge(alpha=10.0)
    model = LinearRegression()
    if sample_weights is not None:
        model.fit(X, y, sample_weight=sample_weights)
    else:
        model.fit(X, y)
    
    # Evaluate on all data if provided, otherwise on training data
    if X_all is not None and y_all is not None:
        logging.info(f"Evaluating on full dataset: {X_all.shape[0]} samples")
        eval_X = X_all
        eval_y = y_all
        metric_prefix = "all"
        eval_label = "all data"
    else:
        eval_X = X
        eval_y = y
        metric_prefix = "train"
        eval_label = "training data"
    
    y_pred = model.predict(eval_X)
    mse = mean_squared_error(eval_y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(eval_y, y_pred)
    r2 = r2_score(eval_y, y_pred)
    
    metrics = {
        f"{metric_prefix}_mse": mse,
        f"{metric_prefix}_rmse": rmse,
        f"{metric_prefix}_mae": mae,
        f"{metric_prefix}_r2": r2,
    }
    
    logging.info(f"Model evaluation on {eval_label}:")
    logging.info(f"  {metric_prefix.capitalize()} RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return model, metrics


def select_training_data_from_d_optimal(
    X: np.ndarray,
    y: np.ndarray,
    candidate_keys: List[str],
    selected_candidate_keys: List[str],
    selected_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    key_to_index = {key: idx for idx, key in enumerate(candidate_keys)}
    missing_keys = [key for key in selected_candidate_keys if key not in key_to_index]
    if missing_keys:
        raise ValueError(
            "D-optimal selected keys are missing from regression feature/fitness data: "
            f"{missing_keys[:10]}"
        )

    selected_indices = [key_to_index[key] for key in selected_candidate_keys]
    return (
        X[selected_indices],
        y[selected_indices],
        np.asarray(selected_weights, dtype=np.float32),
        selected_candidate_keys,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear regression model from a feature bundle.")
    parser.add_argument("features_file", type=Path, help="Path to standardized feature bundle .npy file")
    parser.add_argument("fitness_csv", type=Path, help="Path to CSV file with fitness scores (from get_metrics.py)")
    parser.add_argument("--d-opt-results", type=Path, default=None, help="Path to D-optimal design results .npy file for weighted regression")
    parser.add_argument("--metrics", type=str, nargs="+", default=None, help="Metrics to use for fitness calculation (default: DRT_WL)")
    parser.add_argument("--output", type=Path, default=None, help="Path to save trained model. Default: evaluation/regression_results/{case_name}")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    if not args.features_file.exists():
        raise FileNotFoundError(f"Features file not found: {args.features_file}")
    if not args.fitness_csv.exists():
        raise FileNotFoundError(f"Fitness CSV file not found: {args.fitness_csv}")
    
    case_name = args.features_file.parent.name
    out_dir = args.features_file.parent
    output_path = ( args.output or out_dir ) / "linear_regressor.pkl"
    
    logging.info(f"Loading features from {args.features_file}...")
    X, candidate_keys, metadata = load_features_from_file(args.features_file)

    logging.info(f"Loading fitness scores from {args.fitness_csv}...")
    fitness_dict = load_fitness_scores_from_csv(
        args.fitness_csv,
        metrics=args.metrics,
    )

    # Filter out candidates with NaN/inf fitness values
    valid_indices = []
    for i, key in enumerate(candidate_keys):
        val = fitness_dict.get(key)
        if val is None or not np.isfinite(val):
            continue
        valid_indices.append(i)

    if len(valid_indices) != len(candidate_keys):
        dropped = len(candidate_keys) - len(valid_indices)
        logging.info(f"Dropped {dropped} candidates with NaN/inf fitness scores")

    candidate_keys = [candidate_keys[i] for i in valid_indices]
    X = X[valid_indices]
    fitness_dict = {k: fitness_dict[k] for k in candidate_keys}
    
    # Load D-optimal weights if provided
    d_opt_weights = None
    d_opt_candidate_keys = []
    if args.d_opt_results and args.d_opt_results.exists():
        logging.info(f"Loading D-optimal weights from {args.d_opt_results}...")
        selected_weights, selected_candidate_keys = load_d_optimal_selection(args.d_opt_results)
        y_all = np.array([fitness_dict[key] for key in candidate_keys], dtype=np.float32)
        X_matched, y_matched, d_opt_weights, d_opt_candidate_keys = select_training_data_from_d_optimal(
            X,
            y_all,
            candidate_keys,
            selected_candidate_keys,
            selected_weights,
        )
        best_fitness = y_matched.min()
        best_candidate_key = d_opt_candidate_keys[y_matched.argmin()]
        logging.info(f"Best fitness score found in D-optimal design: {best_fitness:.4f} for candidate {best_candidate_key}")
    else:
        logging.info("No D-optimal weights provided, using uniform weights (standard regression)")
        X_matched = X[[i for i, key in enumerate(candidate_keys)]]
        y_matched = np.array([fitness_dict[key] for key in candidate_keys])
        
    logging.info(f"Feature matrix shape: {X_matched.shape}")
    logging.info(f"Target shape: {y_matched.shape}")
    logging.info(f"Fitness score range: [{y_matched.min():.4f}, {y_matched.max():.4f}], mean: {y_matched.mean():.4f}")
    
    logging.info("Training linear regression model...")
    model, metrics = train_linear_regression(
        X_matched,
        y_matched,
        sample_weights=d_opt_weights,
        X_all=X,
        y_all=np.array([fitness_dict[key] for key in candidate_keys]),
    )
    
    # Save model
    import pickle
    model_data = {
        "model": model,
        "metrics": metrics,
        "candidate_keys": candidate_keys,
        "feature_dim": X_matched.shape[1],
        "feature_type": metadata.get("feature_type"),
        "used_weighted_regression": d_opt_weights is not None,
    }
    if args.d_opt_results and args.d_opt_results.exists():
        model_data["sample_weights"] = d_opt_weights
        model_data["d_opt_candidate_keys"] = d_opt_candidate_keys

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)
    
    logging.info(f"  Saved trained model to {output_path}")
    logging.info(f"  Model coefficients shape: {model.coef_.shape}")
    logging.info(f"  Model intercept: {model.intercept_:.4f}")

    # Predict on all solutions and calculate Kendall's tau for top 15
    logging.info("=" * 60)
    logging.info("Predicting on all solutions and calculating Kendall's tau for top 15...")
    
    # Predict on all solutions
    y_all_pred = model.predict(X)
    y_all_true = np.array([fitness_dict[key] for key in candidate_keys])
    
    # Find top 15 (lowest) predictions and true values
    k = 10
    top_k_pred_indices = np.argsort(y_all_pred)[:k]
    top_k_true_indices = np.argsort(y_all_true)[:k]
    
    # Get the candidate keys for top 15 predictions and true values
    top_k_pred_keys = [candidate_keys[i] for i in top_k_pred_indices]
    top_k_true_keys = [candidate_keys[i] for i in top_k_true_indices]
    
    # Get union of both sets (candidates that appear in either top 15)
    union_keys = list(set(top_k_pred_keys) | set(top_k_true_keys))
    
    # Create ranking maps within top 15: rank 0 = best (lowest), rank k-1 = worst of top k
    pred_rank_map = {key: rank for rank, key in enumerate(top_k_pred_keys)}
    true_rank_map = {key: rank for rank, key in enumerate(top_k_true_keys)}
    
    pred_ranks = []
    true_ranks = []
    for key in union_keys:
        pred_ranks.append(pred_rank_map.get(key, k))
        true_ranks.append(true_rank_map.get(key, k))
    
    # Calculate Kendall's tau
    tau, p_value = kendalltau(pred_ranks, true_ranks)
    
    logging.info(f"Top {k} predicted solutions (lowest fitness):")
    for i, idx in enumerate(top_k_pred_indices):
        key = candidate_keys[idx]
        pred_val = y_all_pred[idx]
        true_val = y_all_true[idx]
        if key in d_opt_candidate_keys and d_opt_weights is not None:
            logging.info(f"  {i+1:2d}. Key: {key}, Predicted: {pred_val:.4f}, True: {true_val:.4f} (D-optimal)")
        else:
            logging.info(f"  {i+1:2d}. Key: {key}, Predicted: {pred_val:.4f}, True: {true_val:.4f}")
    
    logging.info(f"\nTop {k} true solutions (lowest fitness):")
    for i, idx in enumerate(top_k_true_indices):
        key = candidate_keys[idx]
        pred_val = y_all_pred[idx]
        true_val = y_all_true[idx]
        if key in d_opt_candidate_keys and d_opt_weights is not None:
            logging.info(f"  {i+1:2d}. Key: {key}, Predicted: {pred_val:.4f}, True: {true_val:.4f} (D-optimal)")
        else:
            logging.info(f"  {i+1:2d}. Key: {key}, Predicted: {pred_val:.4f}, True: {true_val:.4f}")
    
    logging.info(f"\nKendall's tau calculation:")
    logging.info(f"  Candidates in union of top {k} (predicted or true): {len(union_keys)}")
    logging.info(f"  Kendall's tau: {tau:.4f}")
    logging.info(f"  p-value: {p_value:.4f}")
    
    # the coverage of the top 10 in coreset
    top_10_coverage_coreset = []
    top_10_coverage_prediction = []
    if args.d_opt_results and args.d_opt_results.exists():
        for key in d_opt_candidate_keys:
            if key in top_k_true_keys:
                rank = top_k_true_keys.index(key) + 1
                top_10_coverage_coreset.append(rank)
        for key in top_k_pred_keys:
            if key in top_k_true_keys:
                rank = top_k_true_keys.index(key) + 1
                top_10_coverage_prediction.append(rank)
        logging.info(f"Coverage of the top 10 in coreset: {top_10_coverage_coreset}")
        logging.info(f"Coverage of the top 10 in prediction: {top_10_coverage_prediction}")
        logging.info(f"Coverage of the top 10 in evaluation: {set(top_10_coverage_coreset+top_10_coverage_prediction)}")
if __name__ == "__main__":
    main()

