"""Single-stage DOPP baseline with a matched oracle budget.

This runner is intentionally separate from the analysis reports. It provides a
direct comparison baseline for two-level DOPP: run one global D-opt design over
all candidate solutions, evaluate the top-weight coreset, train one global
surrogate, then evaluate the best surrogate predictions until the target budget
is reached.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from algorithms.dopp.d_opt import qr_rank_cleanup, silvey_titterington_torsney_d_optimal
from algorithms.dopp.loaders import load_features_from_file, load_fitness_scores_from_csv


def align_fitness(
    fitness_dict: Dict[str, float],
    candidate_keys: Sequence[str],
) -> np.ndarray:
    y = np.full(len(candidate_keys), np.nan, dtype=np.float64)
    missing: List[str] = []
    for idx, key in enumerate(candidate_keys):
        value = fitness_dict.get(str(key))
        if value is None or not np.isfinite(value):
            missing.append(str(key))
            continue
        y[idx] = float(value)
    if missing:
        raise ValueError(
            "Missing finite fitness for candidate keys "
            f"(showing first 10 of {len(missing)}): {missing[:10]}"
        )
    return y


def _load_two_level_reference(path: Optional[Path]) -> Optional[int]:
    """Return total oracle calls from a two-level result bundle."""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Two-level results file not found: {path}")

    bundle = np.load(path, allow_pickle=True).item()
    summary = bundle.get("summary", {})
    all_evaluated = set(int(i) for i in summary.get("all_evaluated_indices", []))
    if not all_evaluated:
        for round_key in ("round1", "round2"):
            all_evaluated.update(
                int(i) for i in bundle.get(round_key, {}).get("evaluated_indices", [])
            )

    total_budget = int(summary.get("oracle_calls", len(all_evaluated)))
    if total_budget <= 0 and all_evaluated:
        total_budget = len(all_evaluated)

    return total_budget if total_budget > 0 else None


def _resolve_budgets(
    n_candidates: int,
    total_budget_arg: Optional[int],
    dopt_budget_arg: Optional[int],
    prediction_budget_arg: Optional[int],
    dopt_budget_frac: float,
    two_level_results: Optional[Path],
) -> Tuple[int, int, int, Dict[str, object]]:
    ref_total = _load_two_level_reference(two_level_results)

    total_budget = total_budget_arg if total_budget_arg is not None else ref_total
    if total_budget is None:
        if dopt_budget_arg is not None and prediction_budget_arg is not None:
            total_budget = dopt_budget_arg + prediction_budget_arg
        else:
            raise ValueError(
                "Provide --two-level-results or --total-budget. "
                "The single-stage baseline needs a target oracle budget."
            )

    total_budget = min(int(total_budget), int(n_candidates))
    if total_budget <= 0:
        raise ValueError(f"total budget must be positive, got {total_budget}")

    if dopt_budget_arg is not None:
        dopt_budget = int(dopt_budget_arg)
    elif prediction_budget_arg is not None:
        dopt_budget = total_budget - int(prediction_budget_arg)
    else:
        if not (0.0 < dopt_budget_frac <= 1.0):
            raise ValueError(
                f"dopt budget fraction must be in (0, 1], got {dopt_budget_frac}"
            )
        dopt_budget = int(np.ceil(total_budget * float(dopt_budget_frac)))

    if dopt_budget <= 0:
        raise ValueError(f"dopt budget must be positive, got {dopt_budget}")
    if dopt_budget > total_budget:
        raise ValueError(
            f"dopt budget ({dopt_budget}) cannot exceed total budget ({total_budget})"
        )

    if prediction_budget_arg is not None:
        prediction_budget = int(prediction_budget_arg)
        if dopt_budget + prediction_budget != total_budget:
            raise ValueError(
                "Explicit --dopt-budget/--prediction-budget must sum to the "
                f"total budget: {dopt_budget} + {prediction_budget} != {total_budget}"
            )
    else:
        prediction_budget = total_budget - dopt_budget

    return total_budget, dopt_budget, prediction_budget, {
        "two_level_total_budget": ref_total,
        "dopt_budget_frac": float(dopt_budget_frac),
    }


def _build_design_features(
    X: np.ndarray,
    pca_components: int,
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    scaler = StandardScaler()
    X_std = scaler.fit_transform(np.asarray(X, dtype=np.float64))

    pca_info: Dict[str, object] = {
        "input_dim": int(X_std.shape[1]),
        "pca_components_requested": int(pca_components),
    }
    if pca_components > 0:
        pca_dim = min(int(pca_components), X_std.shape[0], X_std.shape[1])
        pca = PCA(n_components=pca_dim, random_state=random_state)
        X_design_full = pca.fit_transform(X_std)
        pca_info["pca_components_used"] = int(pca_dim)
        pca_info["pca_explained_variance_ratio"] = (
            pca.explained_variance_ratio_.astype(float).tolist()
        )
    else:
        X_design_full = X_std
        pca_info["pca_components_used"] = 0

    original_dim = int(X_design_full.shape[1])
    X_design, kept_columns = qr_rank_cleanup(X_design_full)
    # D-opt weights are invariant to full-rank linear transforms of the feature
    # columns. Using an orthonormal basis keeps the no-PCA path numerically
    # positive definite while preserving the same design problem.
    X_design, _ = np.linalg.qr(X_design, mode="reduced")
    X_design = X_design * np.sqrt(X_design.shape[0])
    pca_info["design_dim_before_cleanup"] = original_dim
    pca_info["design_dim"] = int(X_design.shape[1])
    pca_info["kept_columns"] = kept_columns
    pca_info["orthonormalized_after_cleanup"] = True
    return X_design, pca_info


def _top_weight_indices(weights: np.ndarray, count: int) -> np.ndarray:
    count_eff = min(max(int(count), 0), weights.size)
    return np.argsort(-weights, kind="stable")[:count_eff].astype(np.int64)


def _fill_from_prediction_order(
    predicted: np.ndarray,
    already_selected: Sequence[int],
    total_budget: int,
) -> np.ndarray:
    selected = set(int(i) for i in already_selected)
    picks: List[int] = []
    for idx in np.argsort(predicted, kind="stable").tolist():
        idx = int(idx)
        if idx in selected:
            continue
        selected.add(idx)
        picks.append(idx)
        if len(selected) >= total_budget:
            break
    return np.asarray(picks, dtype=np.int64)


def _coverage(
    evaluated: Sequence[int],
    y: np.ndarray,
    candidate_keys: Sequence[str],
    top_k_truth: Sequence[int],
) -> Dict[str, Dict[str, object]]:
    true_order = np.argsort(y, kind="stable")
    evaluated_set = set(int(i) for i in evaluated)
    coverage_results: Dict[str, Dict[str, object]] = {}
    for k in top_k_truth:
        k_eff = min(int(k), y.size)
        top_k_true = true_order[:k_eff].tolist()
        hits = [idx for idx in top_k_true if idx in evaluated_set]
        coverage_results[f"top_{k}"] = {
            "k": k_eff,
            "hits": len(hits),
            "true_indices": top_k_true,
            "true_keys": [candidate_keys[i] for i in top_k_true],
            "hit_indices": hits,
            "hit_keys": [candidate_keys[i] for i in hits],
        }
    return coverage_results


def run_single_stage_dopp(
    X: np.ndarray,
    y: np.ndarray,
    candidate_keys: List[str],
    total_budget: int,
    dopt_budget: int,
    prediction_budget: int,
    pca_components: int,
    stt_tol: float,
    stt_epsilon: float,
    stt_max_iter: Optional[int],
    random_state: int,
    top_k_truth: Sequence[int],
    reference_budget: Dict[str, Optional[int]],
) -> Dict[str, object]:
    X_design, feature_space = _build_design_features(
        X,
        pca_components=pca_components,
        random_state=random_state,
    )
    logging.info(
        "Single-stage design feature matrix: shape=%s, total_budget=%d, "
        "dopt_budget=%d, prediction_budget=%d",
        X_design.shape,
        total_budget,
        dopt_budget,
        prediction_budget,
    )

    weights, stt_history = silvey_titterington_torsney_d_optimal(
        X_design,
        tol=stt_tol,
        epsilon=stt_epsilon,
        verbose=True,
        max_iter=stt_max_iter,
    )
    weights = np.asarray(weights, dtype=np.float64)
    dopt_indices = _top_weight_indices(weights, dopt_budget)

    surrogate = LinearRegression()
    surrogate.fit(X_design[dopt_indices], y[dopt_indices])
    predicted = surrogate.predict(X_design).astype(np.float64, copy=False)
    surrogate_indices = _fill_from_prediction_order(
        predicted,
        already_selected=dopt_indices.tolist(),
        total_budget=total_budget,
    )

    all_evaluated = sorted(set(dopt_indices.tolist()) | set(surrogate_indices.tolist()))
    evaluated_arr = np.asarray(all_evaluated, dtype=np.int64)
    best_pos = int(np.argmin(y[evaluated_arr]))
    best_idx = int(evaluated_arr[best_pos])
    dopt_set = set(int(i) for i in dopt_indices.tolist())
    surrogate_set = set(int(i) for i in surrogate_indices.tolist())
    if best_idx in dopt_set and best_idx in surrogate_set:
        best_source = "both"
    elif best_idx in surrogate_set:
        best_source = "surrogate"
    else:
        best_source = "dopt"

    coverage = _coverage(all_evaluated, y, candidate_keys, top_k_truth)
    for k_label, payload in coverage.items():
        logging.info("Coverage %s: %d / %d", k_label, payload["hits"], payload["k"])

    return {
        "config": {
            "total_budget": int(total_budget),
            "dopt_budget": int(dopt_budget),
            "prediction_budget": int(prediction_budget),
            "pca_components": int(feature_space["pca_components_used"]),
            "dopt_method": "stt",
            "stt_tol": float(stt_tol),
            "stt_epsilon": float(stt_epsilon),
            "stt_max_iter": None if stt_max_iter is None else int(stt_max_iter),
            "random_state": int(random_state),
            "reference_budget": reference_budget,
        },
        "feature_space": feature_space,
        "dopt": {
            "weights": weights,
            "stt_history": stt_history,
            "selected_indices": dopt_indices.tolist(),
            "selected_keys": [candidate_keys[i] for i in dopt_indices.tolist()],
        },
        "surrogate": {
            "coef": getattr(surrogate, "coef_", None),
            "intercept": getattr(surrogate, "intercept_", None),
            "model_type": type(surrogate).__name__,
            "predicted_fitness": predicted,
            "selected_indices": surrogate_indices.tolist(),
            "selected_keys": [candidate_keys[i] for i in surrogate_indices.tolist()],
        },
        "summary": {
            "best_fitness": float(y[best_idx]),
            "best_solution_index": best_idx,
            "best_solution_key": candidate_keys[best_idx],
            "best_solution_source": best_source,
            "oracle_calls": int(len(all_evaluated)),
            "all_evaluated_indices": all_evaluated,
            "all_evaluated_keys": [candidate_keys[i] for i in all_evaluated],
            "dopt_oracle_calls": int(len(set(dopt_indices.tolist()))),
            "surrogate_extra_oracle_calls": int(len(set(surrogate_indices.tolist()))),
            "coverage": coverage,
        },
        "candidate_keys": candidate_keys,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-stage DOPP baseline with matched oracle budget."
    )
    parser.add_argument("features_file", type=Path)
    parser.add_argument("fitness_csv", type=Path)
    parser.add_argument(
        "--two-level-results",
        type=Path,
        default=None,
        help="Optional two_level_results.npy used to match the total oracle budget.",
    )
    parser.add_argument(
        "--total-budget",
        type=int,
        default=None,
        help="Total oracle calls. Overrides the total budget from --two-level-results.",
    )
    parser.add_argument(
        "--dopt-budget",
        type=int,
        default=None,
        help="Number of global D-opt candidates evaluated before training the surrogate.",
    )
    parser.add_argument(
        "--dopt-budget-frac",
        type=float,
        default=0.9,
        help=(
            "Default fraction of the total budget spent on global D-opt before "
            "the surrogate step. Ignored when --dopt-budget or "
            "--prediction-budget is provided."
        ),
    )
    parser.add_argument(
        "--prediction-budget",
        type=int,
        default=None,
        help="Number of extra surrogate-predicted candidates to evaluate.",
    )
    parser.add_argument("--metrics", type=str, nargs="+", default=None)
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument(
        "--stt-tol",
        "--fw-tol",
        dest="stt_tol",
        type=float,
        default=1e-2,
        help="Tolerance for STT convergence: |g_star - d| <= tol.",
    )
    parser.add_argument(
        "--fw-step-scheme",
        type=str,
        default="1/t",
        choices=["1/t", "line_search", "d_opt"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stt-epsilon",
        "--fw-epsilon",
        dest="stt_epsilon",
        type=float,
        default=0,
        help="Jitter for the STT information matrix.",
    )
    parser.add_argument(
        "--stt-max-iter",
        "--fw-max-iter",
        dest="stt_max_iter",
        type=int,
        default=None,
        help=(
            "Optional STT safety guard. If reached before "
            "convergence, the run raises an error instead of returning a "
            "partial design."
        ),
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--top-k-truth", type=int, nargs="+", default=(10, 20, 50, 100))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: <features_file dir>/single_stage/",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.features_file.exists():
        raise FileNotFoundError(f"Features file not found: {args.features_file}")
    if not args.fitness_csv.exists():
        raise FileNotFoundError(f"Fitness CSV file not found: {args.fitness_csv}")

    out_dir = args.output if args.output is not None else args.features_file.parent / "single_stage"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, candidate_keys, feature_metadata = load_features_from_file(
        args.features_file,
        fitness_csv=args.fitness_csv,
    )
    X = np.asarray(X, dtype=np.float64)
    fitness_dict = load_fitness_scores_from_csv(args.fitness_csv, metrics=args.metrics)
    y = align_fitness(fitness_dict, candidate_keys)

    total_budget, dopt_budget, prediction_budget, reference_budget = _resolve_budgets(
        n_candidates=X.shape[0],
        total_budget_arg=args.total_budget,
        dopt_budget_arg=args.dopt_budget,
        prediction_budget_arg=args.prediction_budget,
        dopt_budget_frac=args.dopt_budget_frac,
        two_level_results=args.two_level_results,
    )

    results = run_single_stage_dopp(
        X=X,
        y=y,
        candidate_keys=list(candidate_keys),
        total_budget=total_budget,
        dopt_budget=dopt_budget,
        prediction_budget=prediction_budget,
        pca_components=args.pca_components,
        stt_tol=args.stt_tol,
        stt_epsilon=args.stt_epsilon,
        stt_max_iter=args.stt_max_iter,
        random_state=args.random_state,
        top_k_truth=args.top_k_truth,
        reference_budget=reference_budget,
    )
    results["feature_metadata"] = feature_metadata

    output_path = out_dir / "single_stage_dopp_results.npy"
    np.save(output_path, results, allow_pickle=True)
    logging.info("Saved single-stage DOPP results to %s", output_path)

    summary = results["summary"]
    logging.info("=" * 60)
    logging.info(
        "Best key: %s (fitness=%.4f, source=%s)",
        summary["best_solution_key"],
        summary["best_fitness"],
        summary["best_solution_source"],
    )
    logging.info("Oracle calls: %d", summary["oracle_calls"])


if __name__ == "__main__":
    main()
