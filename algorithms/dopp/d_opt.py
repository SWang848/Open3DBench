import argparse
import logging
import math
from pathlib import Path
import os
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.optimize import linprog

from algorithms.dopp.loaders import load_features_from_file

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
        # Scale rows directly to avoid forming a dense N x N diagonal matrix.
        WX = w[:, None] * X            # shape (N, d)
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
        Y = np.linalg.solve(M, X.T)   # shape (d, N)
        grad = -np.sum(X * Y.T, axis=1)
        return grad

    # Sum_i w_i = 1  (linear equality constraint)
    A = np.ones((1, N))
    linear_constraint = LinearConstraint(A, lb=[1.0], ub=[1.0])

    # Bounds: 0 <= w_i <= 1
    bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))

    result = minimize(
        fun,
        w0,
        method="trust-constr",                 # or "trust-constr"
        jac=jac,
        constraints=[linear_constraint],
        bounds=bounds,
        options={
            "maxiter": 500,
            "gtol": 1e-8,
            "xtol": 1e-8,
            "verbose": 3 if verbose else 0,
        },
    )

    w_opt = result.x
    # Project tiny negatives to 0 and renormalize, just to be safe
    # w_opt = np.maximum(w_opt, 0.0)
    # s = w_opt.sum()
    # if s > 0:
    #     w_opt /= s
    M= info_matrix(w_opt)
    # A_reg = M - epsilon * np.eye(M.shape[0])
    A_reg = M
    Y = np.linalg.solve(A_reg, X.T)   # shape (d, N)
    q = np.sum(X * Y.T, axis=1)
    g_star = np.max(q)

    if verbose:
        print("Optimization success:", result.success)
        print("Final objective (-logdet):", fun(w_opt))
        print(f"g_star value:{g_star:.4f}")
    
    return w_opt, result


def frank_wolfe_d_optimal(
    X,
    tol=1e-8,
    step_scheme="1/t",
    epsilon=1e-8,
    verbose=False,
    max_iter: Optional[int] = None,
):
    """
    Frank-Wolfe for D-optimal design (approximate design on simplex of size N).

    Args:
        X : np.ndarray, shape (N, d)
            Each row is a feature vector phi(x_i)^T.
        tol : float
            Stop when |g_star - d| <= tol, where g_star = max_i x_i^T M(w)^{-1} x_i.
        step_scheme : str
            "1/t" for gamma_t = 2/(t+2), "line_search" for simple
            backtracking, or "d_opt" for the D-optimal-design exact vertex
            step.
        epsilon : float
            Jitter added to M for numerical stability.
        verbose : bool
            If True, prints objective occasionally.
        max_iter : Optional[int]
            Optional safety guard. If reached before the equivalence condition,
            the function raises instead of returning a partial design.

    Returns:
        w : np.ndarray, shape (N,)
            Design weights on the simplex.
        history : dict
            Contains objective values over iterations.
    """
    N, d = X.shape
    # Start with uniform design
    w = np.ones(N) / N
    history = {"f": [], "g_star": [], "step_size": []}

    def compute_M(w):
        # M = X^T diag(w) X, computed without forming a dense N x N matrix.
        WX = w[:, None] * X            # shape (N, d)
        M = X.T @ WX
        M += epsilon * np.eye(d)
        return M

    if max_iter is not None and max_iter <= 0:
        raise ValueError(f"max_iter must be positive or None, got {max_iter}")

    stop_reason = "not_converged"
    t = 0
    while True:
        if max_iter is not None and t >= max_iter:
            last_g = history["g_star"][-1] if history["g_star"] else float("nan")
            raise RuntimeError(
                "D-opt Frank-Wolfe reached the optional safety max_iter="
                f"{max_iter} before convergence: g_star={last_g:.4f}, "
                f"d={d}, |g_star-d|={abs(last_g - d):.4f}, tol={tol}."
            )

        M = compute_M(w)

        # Objective: f(w) = -log det(M)
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0:
            raise np.linalg.LinAlgError(
                "D-opt information matrix is not positive definite. "
                "Try reducing/conditioning the feature matrix or setting epsilon > 0."
            )
        f = -logdet
        history["f"].append(f)



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
        g_star = -np.min(grad)
        history["g_star"].append(g_star)

        if abs(g_star - d) <= tol:
            stop_reason = "equivalence_tolerance"
            break

        if verbose and t % 20 == 0:
            eigvals = np.linalg.eigvalsh(M)
            print(
                f"[D-opt FW] iter={t}, f=-logdet={f:.4f}, "
                f"|g_star - d|={abs(g_star - d):.4f}, "
                f"eig_min={eigvals.min():.4e}, eig_max={eigvals.max():.4e}"
            )

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
        if step_scheme == "d_opt" and epsilon == 0 and g_star > d and g_star > 1:
            # Exact maximizer of det((1-gamma)M + gamma x_i x_i^T)
            # along the chosen vertex direction.
            trial_gamma = (g_star - d) / (d * (g_star - 1.0))
        elif step_scheme == "1/t":
            # Keep gamma < 1 so a strictly positive start remains in the interior.
            trial_gamma = 2.0 / (t + 3.0)
        elif step_scheme == "line_search" or (step_scheme == "d_opt" and epsilon != 0):
            # Start from a full FW step and backtrack until the objective improves.
            trial_gamma = 1.0
        else:
            raise ValueError(f"Unknown step_scheme: {step_scheme!r}")

        gamma = 0.0
        f_next = f
        w_next = w
        trial_gamma = float(np.clip(trial_gamma, 0.0, 1.0 - 1e-12))
        for _ in range(50):
            if trial_gamma <= 0.0:
                break
            candidate_w = (1 - trial_gamma) * w + trial_gamma * s
            M_trial = compute_M(candidate_w)
            sign_trial, logdet_trial = np.linalg.slogdet(M_trial)
            if sign_trial > 0:
                candidate_f = -logdet_trial
                if candidate_f <= f:
                    gamma = trial_gamma
                    f_next = candidate_f
                    w_next = candidate_w
                    break
            trial_gamma *= 0.5

        improvement = f - f_next
        if improvement <= 0:
            raise RuntimeError(
                "D-opt Frank-Wolfe could not improve the log-det objective "
                f"before convergence: g_star={g_star:.4f}, d={d}, "
                f"|g_star-d|={abs(g_star - d):.4f}, tol={tol}."
            )

        # Update
        w = w_next
        history["step_size"].append(float(gamma))
        t += 1

    # logging.info(f"det value:{np.exp(logdet_trial):.4f}")
    A = compute_M(w)
    A_reg = A if epsilon > 0 else A + 1e-8 * np.eye(A.shape[0])
    Y = np.linalg.solve(A_reg, X.T)      # shape (d, N)
    q = np.sum(X * Y.T, axis=1)
    g_star = np.max(q)
    history["final_g_star"] = float(g_star)
    history["dimension"] = int(d)
    history["stop_reason"] = stop_reason
    history["converged"] = bool(abs(g_star - d) <= tol)
    logging.info(
        "g_star value: %.4f (d=%d, |g_star-d|=%.4f, stop=%s)",
        g_star,
        d,
        abs(g_star - d),
        stop_reason,
    )
    
    return w, history


def silvey_titterington_torsney_d_optimal(
    X,
    tol=1e-8,
    epsilon=1e-8,
    verbose=False,
    max_iter: Optional[int] = None,
):
    """
    Silvey-Titterington-Torsney multiplicative update for D-optimal design.

    The update is

        w_i <- w_i * (x_i^T M(w)^-1 x_i) / d

    where d is the feature dimension and M(w)=X^T diag(w) X. Starting from
    uniform positive weights keeps all design weights positive, and the update
    preserves the simplex exactly when epsilon=0. With epsilon>0, weights are
    renormalized after each update.
    """
    N, d = X.shape
    w = np.ones(N, dtype=np.float64) / N
    history = {"f": [], "g_star": [], "min_q": [], "weight_delta_l1": []}

    def compute_M(weights):
        WX = weights[:, None] * X
        M = X.T @ WX
        M += epsilon * np.eye(d)
        return M

    if max_iter is not None and max_iter <= 0:
        raise ValueError(f"max_iter must be positive or None, got {max_iter}")

    stop_reason = "not_converged"
    t = 0
    while True:
        if max_iter is not None and t >= max_iter:
            last_g = history["g_star"][-1] if history["g_star"] else float("nan")
            raise RuntimeError(
                "D-opt STT reached the optional safety max_iter="
                f"{max_iter} before convergence: g_star={last_g:.4f}, "
                f"d={d}, |g_star-d|={abs(last_g - d):.4f}, tol={tol}."
            )

        M = compute_M(w)
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0:
            raise np.linalg.LinAlgError(
                "D-opt information matrix is not positive definite. "
                "Try reducing/conditioning the feature matrix or setting epsilon > 0."
            )

        Y = np.linalg.solve(M, X.T)
        q = np.sum(X * Y.T, axis=1)
        g_star = float(np.max(q))
        min_q = float(np.min(q))

        history["f"].append(float(-logdet))
        history["g_star"].append(g_star)
        history["min_q"].append(min_q)

        if abs(g_star - d) <= tol:
            stop_reason = "equivalence_tolerance"
            break

        if verbose and t % 20 == 0:
            eigvals = np.linalg.eigvalsh(M)
            print(
                f"[D-opt STT] iter={t}, f=-logdet={-logdet:.4f}, "
                f"|g_star - d|={abs(g_star - d):.4f}, "
                f"q_min={min_q:.4f}, "
                f"eig_min={eigvals.min():.4e}, eig_max={eigvals.max():.4e}"
            )

        next_w = w * (q / d)
        weight_sum = float(next_w.sum())
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            raise RuntimeError("D-opt STT produced invalid non-positive weight sum.")
        next_w /= weight_sum

        delta_l1 = float(np.sum(np.abs(next_w - w)))
        history["weight_delta_l1"].append(delta_l1)
        w = next_w
        t += 1

    A = compute_M(w)
    Y = np.linalg.solve(A, X.T)
    q = np.sum(X * Y.T, axis=1)
    g_star = float(np.max(q))
    history["final_g_star"] = g_star
    history["dimension"] = int(d)
    history["stop_reason"] = stop_reason
    history["converged"] = bool(abs(g_star - d) <= tol)
    logging.info(
        "STT g_star value: %.4f (d=%d, |g_star-d|=%.4f, stop=%s)",
        g_star,
        d,
        abs(g_star - d),
        stop_reason,
    )
    return w, history


def qr_rank_cleanup(
    X: np.ndarray,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, List[int]]:
    """QR-pivot rank cleanup for removing linearly dependent feature columns."""
    _, r_matrix, piv = la.qr(X, mode="economic", pivoting=True)
    rank = int(np.sum(np.abs(np.diag(r_matrix)) > tol))
    independent_columns = np.sort(piv[:rank])
    dependent_columns = np.sort(piv[rank:])

    if dependent_columns.size > 0:
        logging.info(
            "D-opt features: dropped %d linearly dependent columns: %s",
            dependent_columns.size,
            dependent_columns.tolist(),
        )
    else:
        logging.info("D-opt features: full rank, no columns dropped")

    cleaned = X[:, independent_columns]
    return cleaned, independent_columns.tolist()


def select_candidates_by_weights(
    weights: np.ndarray,
    top_k: float = None,
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
        # Select top K percentage candidates by weight.
        # Use at least 10 candidates.
        k = max(10, math.ceil(top_k * len(weights)))
        k = min(k, len(weights))
        selected_indices = np.argsort(weights)[-k:][::-1]
        logging.info(f"Selected top {top_k * 100}% candidates by weight ({k} of {len(weights)}): {selected_indices}")
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
    parser.add_argument("features_file", type=Path, help="Path to standardized feature bundle .npy file")
    parser.add_argument("--fitness-csv", type=Path, default=None, help="Path to CSV file with fitness scores (from get_metrics.py)")
    parser.add_argument("--feature-type", type=str, default="original", choices=["polynomial", "original"], help="Type of features to use for D-optimal design")
    parser.add_argument(
        "--method",
        type=str,
        default="frank_wolfe",
        choices=["frank_wolfe", "stt", "scipy"],
        help="Optimization method: 'frank_wolfe', 'stt', or 'scipy'",
    )
    parser.add_argument("--tol", type=float, default=1e-2, help="Tolerance for the stopping rule |g_star - d| <= tol")
    parser.add_argument("--step-scheme", type=str, default="1/t", choices=["1/t", "line_search", "d_opt"], 
                       help="Step size scheme (only used for Frank-Wolfe method)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Jitter for numerical stability")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help=(
            "Optional Frank-Wolfe/STT safety guard. If reached before "
            "convergence, the run raises an error instead of returning a "
            "partial design."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Path to save D-optimal results. Default: evaluation/regression_results/{case_name}/d_optimal_results.npy")
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
    original_feature_dim = X.shape[1]
    X, kept_columns = qr_rank_cleanup(X)
    metadata = dict(metadata)
    metadata["original_feature_dim"] = int(original_feature_dim)
    metadata["feature_dim"] = int(X.shape[1])
    metadata["kept_columns"] = kept_columns
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
        logging.info(f"  Epsilon: {args.epsilon}")
        
        w, history = scipy_d_optimal(
            X,
            epsilon=args.epsilon,
            verbose=args.verbose,
        )
        f_sci = d_opt_objective(X, w, epsilon=args.epsilon)
        logging.info(f"  scipy-based objective: {f_sci:.4f}")
    elif args.method == "stt":
        logging.info("Running Silvey-Titterington-Torsney D-optimal design algorithm...")
        logging.info(f"  Tol: {args.tol}")
        logging.info(f"  Epsilon: {args.epsilon}")

        w, history = silvey_titterington_torsney_d_optimal(
            X,
            tol=args.tol,
            epsilon=args.epsilon,
            verbose=args.verbose,
            max_iter=args.max_iter,
        )

        f_stt = d_opt_objective(X, w, epsilon=args.epsilon)
        logging.info(f"  STT objective: {f_stt:.4f}")
    else:  # frank_wolfe
        logging.info("Running Frank-Wolfe D-optimal design algorithm...")
        logging.info(f"  Tol: {args.tol}")
        logging.info(f"  Step scheme: {args.step_scheme}")
        logging.info(f"  Epsilon: {args.epsilon}")
        
        w, history = frank_wolfe_d_optimal(
            X,
            tol=args.tol,
            step_scheme=args.step_scheme,
            epsilon=args.epsilon,
            verbose=args.verbose,
            max_iter=args.max_iter,
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
        output_path = os.path.join(args.output, "d_optimal_results.npy")
    else:
        output_path = os.path.join(args.features_file.parent, "d_optimal_results.npy")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
            "tol": args.tol,
            "step_scheme": args.step_scheme if args.method == "frank_wolfe" else None,
            "epsilon": args.epsilon,
            "max_iter": args.max_iter,
        },
    }
    
    np.save(output_path, output_data, allow_pickle=True)
    logging.info(f"Saved D-optimal results to {output_path}")
    logging.info(f"  Selected {len(selected_candidates)} candidates")


if __name__ == "__main__":
    main()
