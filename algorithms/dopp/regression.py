from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from evaluation.get_metrics import cal_fitness_score


def load_features_from_file(features_path: Path) -> Tuple[np.ndarray, list, Dict]:
    """
    Load features from the output of FeatureConstructionByManual.py.
    
    Args:
        features_path: Path to the .npy file containing features
    
    Returns:
        Tuple of (feature_matrix, candidate_keys, metadata)
    """
    data = np.load(features_path, allow_pickle=True).item()
    
    candidate_keys = data["candidate_keys"]
    
    feature_matrix = data["features"]
    feature_names = data.get("feature_names", [])
    feature_dim = data.get("feature_dim", feature_matrix.shape[1])
    
    logging.info(f"Loaded manual features: shape={feature_matrix.shape}")
    logging.info(f"  Number of candidates: {len(candidate_keys)}")
    logging.info(f"  Feature dimension: {feature_dim}")
    
    metadata = {
        "candidate_keys": candidate_keys,
        "feature_names": feature_names,
        "feature_dim": feature_dim,
        "num_candidates": len(candidate_keys),
    }
    
    return feature_matrix, candidate_keys, metadata


def load_fitness_scores_from_csv(
    csv_path: Path,
    metrics: List[str] = None,
) -> Dict[str, float]:
    """
    Load fitness scores from a CSV file (output from get_metrics.py).
    
    Args:
        csv_path: Path to CSV file with metrics and fitness scores
        metrics: List of metrics to use for fitness calculation. If provided, 
                 fitness will be recalculated even if "Fitness" column exists.
    
    Returns:
        Dictionary mapping candidate keys to fitness scores
    """
    df = pd.read_csv(csv_path)
    
    # If metrics are provided, always recalculate fitness
    if metrics is not None:
        logging.info(f"Recalculating fitness scores from metrics: {metrics}")
        df_with_fitness, _ = cal_fitness_score(df, metrics)
        fitness_dict = {}
        for idx, row in df_with_fitness.iterrows():
            key_val = str(row["Key"])
            fitness_dict[key_val] = float(row["Fitness"])
        return fitness_dict
    # Otherwise, use existing Fitness column if available
    elif "Fitness" in df.columns:
        logging.info("Using existing Fitness scores from CSV file")
        fitness_dict = {}
        for idx, row in df.iterrows():
            key_val = str(row["Key"])
            fitness_dict[key_val] = float(row["Fitness"])
        return fitness_dict
    else:
        raise ValueError("CSV file does not contain 'Fitness' column and no metrics provided for calculation")


def load_d_optimal_weights(
    d_opt_results_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
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
    
    selected_weights = data["normalized_weights"]  # selected weights renormalized to sum to 1
    selected_indices = data["selected_indices"]
    
    return selected_weights, selected_indices


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


def validate_model(
    model: LinearRegression,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict:
    """
    Validate a trained model on validation data.
    
    Args:
        model: Trained LinearRegression model
        X_val: Validation feature matrix (N_val, d)
        y_val: Validation target values (N_val,)
    
    Returns:
        Dictionary with validation metrics
    """
    logging.info(f"Validating on {X_val.shape[0]} samples")
    
    # Predict on validation data
    y_val_pred = model.predict(X_val)
    
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    metrics = {
        "val_mse": val_mse,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_r2": val_r2,
    }
    
    logging.info("Model evaluation on validation data:")
    logging.info(f"  Validation RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
    
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear regression model on manual features.")
    parser.add_argument("features_file", type=Path, help="Path to features .npy file from FeatureConstructionByManual.py")
    parser.add_argument("fitness_csv", type=Path, help="Path to CSV file with fitness scores (from get_metrics.py)")
    parser.add_argument("--d-opt-results", type=Path, default=None, help="Path to D-optimal design results .npy file for weighted regression")
    parser.add_argument("--metrics", type=str, nargs="+", default=None, help="Metrics to use for fitness calculation (default: DRT_WL)")
    parser.add_argument("--output", type=Path, default=None, help="Path to save trained model. Default: evaluation/regression_results/{case_name}/linear_regressor.pkl")
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
    output_path = args.output or (out_dir / "linear_regressor.pkl")
    
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
        selected_weights, selected_indices = load_d_optimal_weights(args.d_opt_results)
        d_opt_candidate_keys = [candidate_keys[i] for i in selected_indices]
        
        X_matched = X[selected_indices]
        y_matched = np.array([fitness_dict[key] for key in d_opt_candidate_keys])
        best_fitness = y_matched.min()
        best_candidate_key = d_opt_candidate_keys[y_matched.argmin()]
        logging.info(f"Best fitness score found in D-optimal design: {best_fitness:.4f} for candidate {best_candidate_key}")
        d_opt_weights = selected_weights
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
        "feature_names": metadata["feature_names"],
        "feature_dim": X_matched.shape[1],
        "used_weighted_regression": d_opt_weights is not None,
    }
    if args.d_opt_results and args.d_opt_results.exists():
        model_data["sample_weights"] = d_opt_weights

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)
    
    logging.info(f"  Saved trained model to {output_path}")
    logging.info(f"  Model coefficients shape: {model.coef_.shape}")
    logging.info(f"  Model intercept: {model.intercept_:.4f}")
    
    # Print feature importance (top coefficients by absolute value)
    if len(metadata["feature_names"]) > 0:
        coef_abs = np.abs(model.coef_)
        top_indices = np.argsort(coef_abs)[-10:][::-1]
        logging.info("Top 10 most important features (by absolute coefficient):")
        for idx in top_indices:
            feature_name = metadata["feature_names"][idx] if idx < len(metadata["feature_names"]) else f"feature_{idx}"
            logging.info(f"  {feature_name}: {model.coef_[idx]:.6f}")

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
        coreset_keys = [candidate_keys[i] for i in selected_indices]
        for key in coreset_keys:
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

