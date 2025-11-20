import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np

def frank_wolfe_d_optimal(X, max_iter=200, step_scheme="1/t", epsilon=1e-8, verbose=False):
    """
    Frank-Wolfe for D-optimal design (approximate design on simplex of size N).

    Args:
        X : np.ndarray, shape (N, d)
            Each row is a feature vector phi(x_i)^T.
        max_iter : int
            Maximum number of iterations.
        step_scheme : str
            "1/t" for gamma_t = 2/(t+2) or "line_search" (simple backtracking).
        epsilon : float
            Jitter added to M for numerical stability.
        verbose : bool
            If True, prints objective occasionally.

    Returns:
        w : np.ndarray, shape (N,)
            Design weights on the simplex.
        history : dict
            Contains objective values over iterations.
    """
    N, d = X.shape
    # Start with uniform design
    w = np.ones(N) / N
    history = {"f": []}

    def compute_M_and_inv(w):
        # M = X^T diag(w) X
        WX = w[:, None] * X
        M = X.T @ WX
        M += epsilon * np.eye(d)
        M_inv = np.linalg.inv(M)
        return M, M_inv

    for t in range(max_iter):
        M, M_inv = compute_M_and_inv(w)

        # Objective: f(w) = -log det(M)
        sign, logdet = np.linalg.slogdet(M)
        f = -logdet
        history["f"].append(f)

        if verbose and (t % 20 == 0 or t == max_iter - 1):
            print(f"[D-opt FW] iter={t}, f=-logdet={f:.4f}")

        # Gradient: grad_i = -phi_i^T M^{-1} phi_i
        # Compute v_i = phi_i^T M^{-1} phi_i efficiently:
        XM_inv = X @ M_inv      # shape (N, d)
        quad = np.sum(XM_inv * X, axis=1)  # v_i
        grad = -quad

        # Linear minimization oracle on simplex → pick vertex with smallest grad component
        i_star = np.argmin(grad)
        s = np.zeros_like(w)
        s[i_star] = 1.0

        # Step size
        if step_scheme == "1/t":
            gamma = 2.0 / (t + 2.0)
        else:
            # Simple backtracking line search (optional)
            gamma = 1.0
            f_current = f
            for _ in range(10):
                w_trial = (1 - gamma) * w + gamma * s
                _, M_inv_trial = compute_M_and_inv(w_trial)
                WX_trial = w_trial[:, None] * X
                M_trial = X.T @ WX_trial + epsilon * np.eye(d)
                sign_trial, logdet_trial = np.linalg.slogdet(M_trial)
                f_trial = -logdet_trial
                if f_trial <= f_current:
                    break
                gamma *= 0.5

        # Update
        w = (1 - gamma) * w + gamma * s

    return w, history


def load_features_from_file(features_path: Path) -> Tuple[np.ndarray, list, Dict]:
    """
    Load features from the output of FeatureConstructionByManual.py.
    
    Args:
        features_path: Path to the .npy file containing features
    Returns:
        Tuple of (feature_matrix, candidate_keys, metadata)
        - feature_matrix: np.ndarray of shape (N, d) where N is number of candidates
        - candidate_keys: List of candidate keys in the same order as rows in feature_matrix
        - metadata: Dictionary with feature information
    """
    data = np.load(features_path, allow_pickle=True).item()
    
    candidate_keys = data["candidate_keys"]
    
    feature_matrix = data["polynomial_features"]
    feature_names = data.get("polynomial_feature_names", [])
    feature_dim = data.get("polynomial_feature_dim", feature_matrix.shape[1])
    logging.info(f"Using polynomial features: shape={feature_matrix.shape}")
    
    metadata = {
        "candidate_keys": candidate_keys,
        "feature_names": feature_names,
        "feature_dim": feature_dim,
        "num_candidates": len(candidate_keys),
    }
    
    return feature_matrix, candidate_keys, metadata


def load_labels_for_candidates(
    hmsa_results_path: Path,
    selected_candidate_keys: list,
) -> Dict[str, Tuple[float, float]]:
    """
    Load labels (costs) for selected candidates from HMSA results JSON.
    
    Args:
        hmsa_results_path: Path to hmsa_results.json
        selected_candidate_keys: List of candidate keys to extract labels for
    
    Returns:
        Dictionary mapping candidate keys to (cut_size, area_imbalance) labels
    """
    import json
    
    with open(hmsa_results_path, "r") as fp:
        data = json.load(fp)
    
    labels_dict = {}
    selected_set = set(selected_candidate_keys)
    
    for key, entry in data["pareto_archive"]["solutions"].items():
        if key in selected_set:
            cost = entry.get("cost", [0.0, 0.0])
            cut_size = float(cost[0])
            area_imbalance = float(cost[1])
            labels_dict[key] = (cut_size, area_imbalance)
    
    if len(labels_dict) != len(selected_candidate_keys):
        missing = set(selected_candidate_keys) - set(labels_dict.keys())
        logging.warning(f"Warning: {len(missing)} selected candidates not found in JSON: {list(missing)[:5]}...")
    
    return labels_dict


def select_candidates_by_weights(
    weights: np.ndarray,
    candidate_keys: list,
    top_k: int = None,
    threshold: float = None,
) -> list:
    """
    Select candidates based on D-optimal design weights.
    
    Args:
        weights: Design weights from D-optimal algorithm
        candidate_keys: List of candidate keys
        top_k: If provided, return top K candidates by weight
        threshold: If provided, return candidates with weight >= threshold
    
    Returns:
        List of selected candidate keys
    """
    if top_k is not None:
        # Select top K candidates by weight
        top_indices = np.argsort(weights)[-top_k:][::-1]
        selected = [candidate_keys[i] for i in top_indices]
        logging.info(f"Selected top {top_k} candidates by weight")
    elif threshold is not None:
        # Select candidates above threshold
        selected = [candidate_keys[i] for i in range(len(weights)) if weights[i] >= threshold]
        logging.info(f"Selected {len(selected)} candidates with weight >= {threshold}")
    else:
        # Select all candidates with non-zero weight
        # 
        # Why non-zero weights matter:
        # In D-optimal design, we maximize det(M) where M = X^T diag(w) X is the information matrix.
        # The Frank-Wolfe algorithm converges to a sparse solution where only a subset of candidates
        # receive non-zero weights. This is due to:
        # 1. Carathéodory's theorem: The optimal design has at most d(d+1)/2 + 1 support points
        #    (where d is feature dimension), so most weights will be exactly zero.
        # 2. These non-zero weight candidates are the "support points" that optimally span the
        #    feature space and maximize information content for regression.
        # 3. Candidates with zero weight don't contribute to the information matrix and can be
        #    safely excluded from the design.
        # 4. The selected candidates with non-zero weights are the ones that matter most for
        #    building an informative dataset for regression model training.
        selected = [candidate_keys[i] for i in range(len(weights)) if weights[i] > 1e-6]
        logging.info(f"Selected {len(selected)} candidates with non-zero weight")
    
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D-optimal design on extracted features.")
    parser.add_argument("features_file", type=Path, help="Path to features .npy file from FeatureConstructionByManual.py")
    parser.add_argument("--hmsa-results", type=Path, default=None, help="Path to hmsa_results.json to extract labels for selected candidates")
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum iterations for Frank-Wolfe algorithm")
    parser.add_argument("--step-scheme", type=str, default="1/t", choices=["1/t", "line_search"], help="Step size scheme")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Jitter for numerical stability")
    parser.add_argument("--output", type=Path, default=None, help="Path to save D-optimal results")
    parser.add_argument("--top-k", type=int, default=None, help="Select top K candidates by weight")
    parser.add_argument("--threshold", type=float, default=None, help="Select candidates with weight >= threshold")
    parser.add_argument("--save-training-data", action="store_true", help="Save selected candidates with features and labels for regression training")
    parser.add_argument("--verbose", action="store_true", help="Verbose output during optimization")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    if not args.features_file.exists():
        raise FileNotFoundError(f"Features file not found: {args.features_file}")
    
    # Load features
    logging.info(f"Loading features from {args.features_file}...")
    X, candidate_keys, metadata = load_features_from_file(args.features_file)
    
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Number of candidates: {len(candidate_keys)}")
    logging.info(f"Feature dimension: {X.shape[1]}")
    
    # Check if we have more features than candidates (rank issue)
    if X.shape[1] > X.shape[0]:
        logging.warning(f"Feature dimension ({X.shape[1]}) > number of candidates ({X.shape[0]}). "
                       f"Matrix may be rank-deficient. Consider reducing polynomial degree.")
    
    # Run D-optimal design
    logging.info("Running Frank-Wolfe D-optimal design algorithm...")
    logging.info(f"  Max iterations: {args.max_iter}")
    logging.info(f"  Step scheme: {args.step_scheme}")
    logging.info(f"  Epsilon: {args.epsilon}")
    
    w, history = frank_wolfe_d_optimal(
        X,
        max_iter=args.max_iter,
        step_scheme=args.step_scheme,
        epsilon=args.epsilon,
        verbose=args.verbose,
    )
    
    logging.info(f"D-optimal design completed.")
    logging.info(f"  Final objective: {history['f'][-1]:.4f}")
    logging.info(f"  Number of non-zero weights: {(w > 1e-2).sum()}")
    logging.info(f"  Max weight: {w.max():.6f}")
    logging.info(f"  Min weight: {w[w > 1e-6].min() if (w > 1e-6).any() else 0:.6f}")
    
    # Select candidates based on weights
    selected_candidates = select_candidates_by_weights(
        w,
        candidate_keys,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    
    # Load labels for selected candidates if hmsa_results is provided
    labels_dict = {}
    if args.hmsa_results and args.hmsa_results.exists():
        logging.info(f"Loading labels from {args.hmsa_results}...")
        labels_dict = load_labels_for_candidates(args.hmsa_results, selected_candidates)
        logging.info(f"Loaded labels for {len(labels_dict)} selected candidates")
    
    # Prepare output
    output_path = args.output or (args.features_file.parent / "d_optimal_results.npy")
    
    output_data = {
        "weights": w,
        "candidate_keys": candidate_keys,
        "selected_candidates": selected_candidates,
        "history": history,
        "metadata": metadata,
        "algorithm_params": {
            "max_iter": args.max_iter,
            "step_scheme": args.step_scheme,
            "epsilon": args.epsilon
        },
    }
    
    # Add labels if available
    if labels_dict:
        output_data["selected_labels"] = labels_dict
        logging.info("Labels included in output data")
    
    np.save(output_path, output_data, allow_pickle=True)
    logging.info(f"Saved D-optimal results to {output_path}")
    logging.info(f"  Selected {len(selected_candidates)} candidates")
    
    # Save training dataset if requested
    if args.save_training_data and len(selected_candidates) > 0:
        # Load original features data
        features_data = np.load(args.features_file, allow_pickle=True).item()
        
        # Extract features and labels for selected candidates
        selected_indices = [candidate_keys.index(key) for key in selected_candidates if key in candidate_keys]
        
        if len(selected_indices) > 0:
            # Get polynomial features for selected candidates
            selected_polynomial_features = features_data["polynomial_features"][selected_indices]
            selected_original_features = features_data["original_features"][selected_indices]
            
            # Get labels
            selected_labels = []
            selected_keys_with_labels = []
            for key in selected_candidates:
                if key in labels_dict:
                    selected_labels.append(labels_dict[key])
                    selected_keys_with_labels.append(key)
            
            if len(selected_labels) > 0:
                training_data = {
                    "candidate_keys": selected_keys_with_labels,
                    "polynomial_features": selected_polynomial_features[:len(selected_labels)],
                    "original_features": selected_original_features[:len(selected_labels)],
                    "labels": np.array(selected_labels),  # shape: (N, 2) where 2 = [cut_size, area_imbalance]
                    "weights": w[selected_indices[:len(selected_labels)]],
                    "feature_names": {
                        "original": features_data.get("original_feature_names", []),
                        "polynomial": features_data.get("polynomial_feature_names", []),
                    },
                }
                
                training_data_path = args.features_file.parent / "d_optimal_training_data.npy"
                np.save(training_data_path, training_data, allow_pickle=True)
                logging.info(f"Saved training dataset to {training_data_path}")
                logging.info(f"  Number of training samples: {len(selected_labels)}")
                logging.info(f"  Feature dimensions: original={selected_original_features.shape[1]}, "
                           f"polynomial={selected_polynomial_features.shape[1]}")
                logging.info(f"  Labels shape: {np.array(selected_labels).shape}")
                logging.info("  You can now use this dataset to train a regression model!")
    
    # Print top candidates
    if len(selected_candidates) > 0:
        # Get weights for selected candidates
        selected_indices = [candidate_keys.index(key) for key in selected_candidates]
        selected_weights = w[selected_indices]
        
        # Sort by weight
        sorted_pairs = sorted(zip(selected_candidates, selected_weights), key=lambda x: x[1], reverse=True)
        
        logging.info("Top selected candidates by weight:")
        for i, (key, weight) in enumerate(sorted_pairs[:10]):
            label_str = ""
            if key in labels_dict:
                cut_size, area_imbalance = labels_dict[key]
                label_str = f", cut_size={cut_size:.2f}, area_imbalance={area_imbalance:.2f}"
            logging.info(f"  {i+1}. '{key}': weight={weight:.6f}{label_str}")


if __name__ == "__main__":
    main()
