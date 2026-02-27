import argparse
import logging
from pathlib import Path
import os
from typing import Tuple, Dict

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.optimize import linprog
import pandas as pd

def scipy_d_optimal(X, epsilon=1e-8, verbose=False):
    """
    Solve the continuous D-optimal design problem with scipy.optimize.minimize.

    Args:
        X : np.ndarray, shape (N, d)
            Each row is a feature vector phi(x_i)^T.
        epsilon : float
            Small jitter added to M for numerical stability.
        verbose : bool
            If True, prints basic info.

    Returns:
        w_opt : np.ndarray, shape (N,)
            Optimal design weights on the simplex.
        result : OptimizeResult
            Raw scipy result object.
    """
    N, d = X.shape

    # Initial design: uniform weights on simplex
    w0 = np.ones(N) / N

    def info_matrix(w):
        WX = np.diag(w) @ X            # shape (N, d)
        M = X.T @ WX
        M += epsilon * np.eye(d)
        return M

    def fun(w):
        # Penalize infeasible w to keep solver away from bad regions
        # if np.any(w < 0):
        #     return 1e6 + np.sum(np.maximum(-w, 0.0))
        M = info_matrix(w)
        sign, logdet = np.linalg.slogdet(M)
        # If M is not SPD, penalize heavily
        # if sign <= 0:
        #     return 1e6
        return -logdet  # we minimize -logdet

    def jac(w):
        """
        Gradient of -log det M(w) wrt w_i:
        df/dw_i = -phi_i^T M^{-1} phi_i
        """
        M = info_matrix(w)
        M_inv = np.linalg.inv(M)
        XM_inv = X @ M_inv    
        XM_invX = XM_inv @ X.T # shape (N, d)
        # quad = np.sum(XM_inv * X, axis=1)  # phi_i^T M^{-1} phi_i
        grad = np.array([XM_invX[i,i] for i in range(XM_invX.shape[0])])
        return -1 * grad

    # Sum_i w_i = 1  (linear equality constraint)
    A = np.ones((1, N))
    linear_constraint = LinearConstraint(A, lb=[1.0], ub=[1.0])

    # Bounds: 0 <= w_i <= 1
    bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))

    result = minimize(
        fun,
        w0,
        method="SLSQP",                 # or "trust-constr"
        jac=jac,
        constraints=[linear_constraint],
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-8, "disp": verbose},
    )

    w_opt = result.x
    # Project tiny negatives to 0 and renormalize, just to be safe
    # w_opt = np.maximum(w_opt, 0.0)
    # s = w_opt.sum()
    # if s > 0:
    #     w_opt /= s
    M= info_matrix(w_opt)
    M_inv = np.linalg.inv(M)
    # A_reg = M - epsilon * np.eye(M.shape[0])
    A_reg = M
    q = np.array([X[i] @ np.linalg.solve(A_reg, X[i].T) for i in range(X.shape[0])])
    g_star = np.max(q)

    if verbose:
        print("Optimization success:", result.success)
        print("Final objective (-logdet):", fun(w_opt))
        print(f"g_star value:{g_star:.4f}")
    
    return w_opt, result


def frank_wolfe_d_optimal(X, max_iter=500, step_scheme="1/t", epsilon=1e-8, verbose=False):
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
        WX = np.diag(w) @ X            # shape (N, d)
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
        # XM_inv = X @ M_inv      # shape (N, d)
        # XM_invX = XM_inv @ X.T # shape (N, d)
        Y = np.linalg.solve(M, X.T)   # solves A @ Y = X.T
        grad = np.sum(X.T * Y, axis=0)
        # quad = np.sum(XM_inv * X, axis=1)  # phi_i^T M^{-1} phi_i
        # grad = np.array([XM_invX[i,i] for i in range(XM_invX.shape[0])])
        # quad = np.sum(XM_inv * X, axis=1)  # v_i
        grad = -1 * grad
        
        # s = linprog(grad, A_eq=np.ones((1,N)), b_eq=np.ones(1))
        # s=s.x
        # if t%100==0:
        #     logging.info(f"grad value:{grad}")
        #     logging.info(f"s value:{s}")
        #     logging.info(f"i_star value:{np.argmin(grad)}")

        # # Linear minimization oracle on simplex → pick vertex with smallest grad component
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

    # logging.info(f"det value:{np.exp(logdet_trial):.4f}")
    W = np.diag(w)
    A = X.T @ W @ X                      # (d, d)

    # Optional small regularization if A is ill-conditioned
    eps = 1e-8
    A_reg = A + eps * np.eye(A.shape[0])

    # Solve A_reg @ Y = X.T  -> Y = A_reg^{-1} X.T
    # Y = np.linalg.solve(A_reg, X.T)      # (d, N)

    # # q[i] = x_i^T A^{-1} x_i
    # q = np.sum(X.T * Y, axis=0)          # (N,)
    q = np.array([X[i] @ np.linalg.solve(A_reg, X[i].T) for i in range(X.shape[0])])
    g_star = np.max(q)
    logging.info(f"g_star value:{g_star:.4f}")
    
    return w, history


def load_features_from_file(features_path: Path, fitness_csv: Path, feature_type: str = "polynomial") -> Tuple[np.ndarray, list, Dict]:
    """
    Load features from the output of FeatureConstructionByManual.py.
    
    Args:
        features_path: Path to the .npy file containing features
        fitness_csv: Path to the CSV file with fitness scores
    Returns:
        Tuple of (feature_matrix, candidate_keys, metadata)
        - feature_matrix: np.ndarray of shape (N, d) where N is number of candidates
        - candidate_keys: List of candidate keys in the same order as rows in feature_matrix
        - metadata: Dictionary with feature information
    """
    df = pd.read_csv(fitness_csv)
    fitness_dict = {}
    for idx, row in df.iterrows():
        key_val = str(row["Key"])
        fitness_dict[key_val] = float(row["Fitness"])

    data = np.load(features_path, allow_pickle=True).item()
    candidate_keys = data["candidate_keys"]
    
    valid_indices = []
    for i, key in enumerate(candidate_keys):
        val = fitness_dict.get(key)
        if val is None or not np.isfinite(val):
            continue
        valid_indices.append(i)
    
    feature_matrix = data["features"]
    feature_names = data.get("feature_names", [])
    feature_dim = data.get("feature_dim", feature_matrix.shape[1])
    
    if not valid_indices:
        raise ValueError("No valid candidates remain after filtering NaN/inf fitness values.")
    if len(valid_indices) != len(candidate_keys):
        dropped = len(candidate_keys) - len(valid_indices)
        logging.info(f"Dropped {dropped} candidates with NaN/inf fitness scores")
    
    candidate_keys = [candidate_keys[i] for i in valid_indices]
    feature_matrix = feature_matrix[valid_indices]
    
    logging.info(f"Using {feature_type} features: shape={feature_matrix.shape}")
    
    metadata = {
        "candidate_keys": candidate_keys,
        "feature_names": feature_names,
        "feature_dim": feature_dim,
        "num_candidates": len(candidate_keys),
    }
    
    return feature_matrix, candidate_keys, metadata


def select_candidates_by_weights(
    weights: np.ndarray,
    top_k: int = None,
    threshold: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select candidates based on D-optimal design weights.
    
    Args:
        weights: Design weights from D-optimal algorithm
        candidate_keys: List of candidate keys
        top_k: If provided, return top K percentage candidates by weight
        threshold: If provided, return candidates with weight >= threshold
    
    Returns:
        Tuple of (selected_indices, normalized_weights) where normalized_weights
        are the selected weights renormalized to sum to 1
    """
    if top_k is not None:
        # Select top K percentage candidates by weight
        selected_indices = np.argsort(weights)[-int(top_k * len(weights)) :][::-1]
        logging.info(f"Selected top {top_k * 100}% candidates by weight: {selected_indices}")
    elif threshold is not None:
        # Select candidates above threshold
        selected_indices = np.where(weights >= threshold)[0]
        logging.info(f"Selected {len(selected_indices)} candidates with weight >= {threshold}")
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
        selected_indices = np.where(weights > 1e-6)[0]
        logging.info(f"Selected {len(selected_indices)} candidates with non-zero weight")
    
    # Extract selected weights and renormalize to sum to 1
    selected_weights = weights[selected_indices]
    normalized_weights = selected_weights / selected_weights.sum()
    logging.info(f"Renormalized weights sum: {normalized_weights.sum():.6f}")
    
    return selected_indices, normalized_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D-optimal design on extracted features.")
    parser.add_argument("features_file", type=Path, help="Path to features .npy file from FeatureConstructionByManual.py")
    parser.add_argument("fitness_csv", type=Path, help="Path to CSV file with fitness scores (from get_metrics.py)")
    parser.add_argument("--feature-type", type=str, default="original", choices=["polynomial", "original"], help="Type of features to use for D-optimal design")
    parser.add_argument("--method", type=str, default="scipy", choices=["frank_wolfe", "scipy"], 
                       help="Optimization method: 'frank_wolfe' or 'scipy' (default: scipy)")
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum iterations for optimization algorithm")
    parser.add_argument("--step-scheme", type=str, default="1/t", choices=["1/t", "line_search"], 
                       help="Step size scheme (only used for Frank-Wolfe method)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Jitter for numerical stability")
    parser.add_argument("--output", type=Path, default=None, help="Path to save D-optimal results")
    parser.add_argument("--top-k", type=float, default=None, help="Select top K percentage candidates by weight")
    parser.add_argument("--threshold", type=float, default=1e-6, help="Select candidates with weight >= threshold")
    parser.add_argument("--verbose", action="store_true", help="Verbose output during optimization")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()

def d_opt_objective(X, w, epsilon=1e-8):
    WX = w[:, None] * X
    M = X.T @ WX + epsilon * np.eye(X.shape[1])
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        return np.inf
    return -logdet

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    if not args.features_file.exists():
        raise FileNotFoundError(f"Features file not found: {args.features_file}")
    
    # Load features
    logging.info(f"Loading features from {args.features_file}...")
    X, candidate_keys, metadata = load_features_from_file(args.features_file, args.fitness_csv, args.feature_type)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = np.sum(S > 1e-8)
    logging.info(f"Effective rank of feature matrix: {r}")
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Number of candidates: {len(candidate_keys)}")
    logging.info(f"Feature dimension: {X.shape[1]}")
    
    # Check if we have more features than candidates (rank issue)
    if X.shape[1] > X.shape[0]:
        logging.warning(f"Feature dimension ({X.shape[1]}) > number of candidates ({X.shape[0]}). "
                       f"Matrix may be rank-deficient. Consider reducing polynomial degree.")
    # X = np.delete(X, 0, axis=1)
    print(f"Matrix rank: {np.linalg.matrix_rank(X)}")
    print(f"Feature matrix shape: {X.shape}")
    if args.method == "scipy":
        logging.info("Running scipy-based D-optimal design algorithm...")
        logging.info(f"  Max iterations: {args.max_iter}")
        logging.info(f"  Epsilon: {args.epsilon}")
        
        w, history = scipy_d_optimal(
            X,
            epsilon=args.epsilon,
            verbose=args.verbose,
        )
        f_sci = d_opt_objective(X, w, epsilon=args.epsilon)
        logging.info(f"  scipy-based objective: {f_sci:.4f}")
    else:  # frank_wolfe
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
        
        f_fw = d_opt_objective(X, w, epsilon=args.epsilon)
        logging.info(f"  frank-wolfe-based objective: {f_fw:.4f}")
        
    logging.info(f"D-optimal design completed.")
    # logging.info(f"  Final objective: {history['f'][-1]:.4f}")
    logging.info(f"  Weights: {w}")
    logging.info(f"  Number of non-zero weights: {(w > args.threshold).sum()}")
    logging.info(f"  Max weight: {w.max():.6f}")
    logging.info(f"  Min weight: {w[w > args.threshold].min() if (w > args.threshold).any() else 0:.6f}")
    
    # Select candidates based on weights
    selected_indices, normalized_weights = select_candidates_by_weights(
        w,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    
    # Prepare output
    if args.output is not None:
        output_path = args.output
    else:
        output_path = os.path.join(args.features_file.parent, "d_optimal_results.npy")
    selected_candidates = [candidate_keys[i] for i in selected_indices]

    # breakpoint()
    output_data = {
        "weights": w,
        "candidate_keys": candidate_keys,
        "selected_indices": selected_indices,
        "normalized_weights": normalized_weights,
        "history": history,
        "metadata": metadata,
        "algorithm_params": {
            "method": args.method,
            "max_iter": args.max_iter,
            "step_scheme": args.step_scheme,
            "epsilon": args.epsilon
        },
    }
    
    np.save(output_path, output_data, allow_pickle=True)
    logging.info(f"Saved D-optimal results to {output_path}")
    logging.info(f"  Selected {len(selected_candidates)} candidates")


if __name__ == "__main__":
    main()
