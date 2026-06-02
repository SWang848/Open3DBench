"""Markdown analysis report for the single-stage DOPP baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from algorithms.dopp.baseline_analysis_utils import (
    format_float,
    format_pct,
    load_bundle,
    safe_corr,
    write_report,
)
from algorithms.dopp.loaders import load_features_from_file, load_fitness_scores_from_csv
from algorithms.dopp.single_stage_dopp import _build_design_features, align_fitness

DOPT_NONZERO_THRESHOLD = 1e-3


def _validate_key_alignment(
    result_keys: Sequence[str],
    feature_keys: Sequence[str],
) -> None:
    if len(result_keys) != len(feature_keys):
        raise ValueError(
            "Feature file candidate count does not match single-stage result bundle: "
            f"{len(feature_keys)} != {len(result_keys)}"
        )
    mismatches = [
        (idx, str(result_keys[idx]), str(feature_keys[idx]))
        for idx in range(len(result_keys))
        if str(result_keys[idx]) != str(feature_keys[idx])
    ]
    if mismatches:
        preview = mismatches[:5]
        raise ValueError(
            "Feature file candidate order/key alignment does not match result bundle "
            f"(showing first {len(preview)}): {preview}"
        )


def _validate_metrics_keys(metrics_csv: Path, candidate_keys: Sequence[str]) -> None:
    df = pd.read_csv(metrics_csv, usecols=["Key"])
    available = set(df["Key"].astype(str).tolist())
    missing = [str(key) for key in candidate_keys if str(key) not in available]
    if missing:
        raise ValueError(
            "Metrics CSV is missing candidate keys from the single-stage result bundle "
            f"(showing first 10 of {len(missing)}): {missing[:10]}"
        )


def _rank_ascending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.int64)
    return ranks


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    sst = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if sst <= 0.0:
        return float("nan")
    sse = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - sse / sst


def _format_table_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(lambda value: "yes" if bool(value) else "no")
        elif pd.api.types.is_float_dtype(out[col]):
            col_name = str(col).lower()
            integer_like = (
                col_name in {"pred_rank", "true_rank", "n_train", "n_eval", "best_true_rank"}
                or col_name.endswith("_calls")
                or col_name.endswith("_size")
            )
            if integer_like:
                out[col] = out[col].map(
                    lambda x: str(int(round(float(x)))) if pd.notna(x) else "nan"
                )
            else:
                out[col] = out[col].map(
                    lambda x: format_float(float(x)) if pd.notna(x) else "nan"
                )
    return out


def _to_code_table(df: pd.DataFrame) -> str:
    out = _format_table_values(df)
    return "\n".join(["```text", out.to_string(index=False), "```"])


def _top_predicted_table(
    candidate_keys: Sequence[str],
    predicted: np.ndarray,
    y: np.ndarray,
    coreset: set[int],
    top_n: int = 100,
    include_coreset: bool = True,
) -> pd.DataFrame:
    pred_ranks = _rank_ascending(predicted)
    true_ranks = _rank_ascending(y)
    order = np.argsort(predicted, kind="stable")[: min(top_n, predicted.size)]
    rows = []
    for idx in order.tolist():
        idx = int(idx)
        row = {
            "candidate_key": str(candidate_keys[idx]),
            "pred_rank": int(pred_ranks[idx]),
            "true_rank": int(true_ranks[idx]),
            "pred": float(predicted[idx]),
            "true": float(y[idx]),
        }
        if include_coreset:
            row["in_coreset"] = idx in coreset
        rows.append(row)
    return pd.DataFrame(rows)


def _prediction_error_row(
    mode: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_train: int,
) -> Dict[str, object]:
    residual = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residual**2)))
    true_range = float(np.max(y_true) - np.min(y_true))
    mae = float(np.mean(np.abs(residual)))
    return {
        "mode": mode,
        "n_train": int(n_train),
        "n_eval": int(y_true.size),
        "RMSE": rmse,
        "nRMSE": rmse / true_range if true_range > 0.0 else np.nan,
        "MAE": mae,
        "R2": _r2_score(y_true, y_pred),
    }


def _ranking_row(mode: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    corr = safe_corr(y_pred, y_true)
    pred_order = np.argsort(y_pred, kind="stable")
    true_order = np.argsort(y_true, kind="stable")
    pred_top100 = pred_order[: min(100, y_pred.size)]
    true_ranks = _rank_ascending(y_true)

    row: Dict[str, object] = {
        "mode": mode,
        "Kendall tau": corr["kendall"],
        "Spearman": corr["spearman"],
    }
    for k in (10, 20, 50, 100):
        k_eff = min(k, y_true.size)
        pred_top = set(pred_order[:k_eff].tolist())
        true_top = set(true_order[:k_eff].tolist())
        row[f"Recall@{k}"] = len(pred_top & true_top) / float(k_eff)
    row["Best true rank in pred top100"] = int(true_ranks[pred_top100].min())
    return row


def _coverage_cell(summary: Dict, label: str) -> str:
    payload = summary.get("coverage", {}).get(label)
    if not payload:
        return "nan"
    return f"{int(payload.get('hits', 0))}/{int(payload.get('k', 0))}"


def _saved_budget_table(bundle: Dict, y: np.ndarray) -> pd.DataFrame:
    summary = bundle.get("summary", {})
    dopt = bundle.get("dopt", {})
    best_idx = int(summary.get("best_solution_index", -1))
    true_ranks = _rank_ascending(y)
    best_true_rank = int(true_ranks[best_idx]) if 0 <= best_idx < true_ranks.size else np.nan
    return pd.DataFrame(
        [
            {
                "oracle_calls": int(summary.get("oracle_calls", 0)),
                "coreset_size": int(len(set(int(i) for i in dopt.get("selected_indices", [])))),
                "surrogate_extra_calls": int(summary.get("surrogate_extra_oracle_calls", 0)),
                "best_fitness": float(summary.get("best_fitness", np.nan)),
                "best_true_rank": best_true_rank,
                "top10": _coverage_cell(summary, "top_10"),
                "top20": _coverage_cell(summary, "top_20"),
                "top50": _coverage_cell(summary, "top_50"),
                "top100": _coverage_cell(summary, "top_100"),
            }
        ]
    )


def _dopt_weight_markdown(bundle: Dict, n_candidates: int) -> str:
    weights = np.asarray(bundle.get("dopt", {}).get("weights", []), dtype=np.float64)
    if weights.size == 0:
        return "No saved D-opt candidate weights were found in this result bundle."

    if weights.size != n_candidates:
        logging.warning(
            "Saved D-opt weight count does not match candidate count: %d != %d",
            weights.size,
            n_candidates,
        )

    finite = weights[np.isfinite(weights)]
    if finite.size == 0:
        return "No finite saved D-opt candidate weights were found in this result bundle."

    nonzero = int(np.sum(finite > DOPT_NONZERO_THRESHOLD))
    logging.info(
        "Single-stage D-opt weight distribution: sum=%.6f, "
        "nonzero > %.3g=%d/%d, max=%.6f",
        float(np.sum(finite)),
        DOPT_NONZERO_THRESHOLD,
        nonzero,
        finite.size,
        float(np.max(finite)),
    )
    stat_rows = [
        {"stat": "sum", "value": format_float(float(np.sum(finite)), digits=6)},
        {
            "stat": f"nonzero > {DOPT_NONZERO_THRESHOLD:g}",
            "value": f"{nonzero} / {finite.size} ({format_pct(nonzero / finite.size)})",
        },
        {"stat": "min", "value": format_float(float(np.min(finite)), digits=6)},
        {"stat": "p25", "value": format_float(float(np.percentile(finite, 25)), digits=6)},
        {"stat": "median", "value": format_float(float(np.percentile(finite, 50)), digits=6)},
        {"stat": "p75", "value": format_float(float(np.percentile(finite, 75)), digits=6)},
        {"stat": "p90", "value": format_float(float(np.percentile(finite, 90)), digits=6)},
        {"stat": "max", "value": format_float(float(np.max(finite)), digits=6)},
    ]
    return "\n".join(
        [
            "Weight distribution:",
            "",
            _to_code_table(pd.DataFrame(stat_rows)),
        ]
    )


def build_report(
    bundle: Dict,
    candidate_keys: Sequence[str],
    y: np.ndarray,
    X: np.ndarray,
    results_npy: Path,
    fitness_csv: Path,
    features_file: Path,
) -> str:
    config = bundle.get("config", {})
    pca_components = int(config.get("pca_components", 20))
    random_state = int(config.get("random_state", 0))
    X_design, feature_space = _build_design_features(
        X,
        pca_components=pca_components,
        random_state=random_state,
    )
    input_dim = int(X.shape[1])
    design_dim = int(X_design.shape[1])
    pca_used = int(feature_space["pca_components_used"])
    logging.info(
        "Single-stage analysis feature dimensions: input_dim=%d, "
        "design_dim=%d, pca_components_used=%d",
        input_dim,
        design_dim,
        pca_used,
    )

    coreset = set(int(i) for i in bundle.get("dopt", {}).get("selected_indices", []))
    coreset_indices = np.asarray(sorted(coreset), dtype=np.int64)
    if coreset_indices.size == 0:
        raise ValueError("Single-stage result bundle is missing dopt.selected_indices")

    oracle_model = LinearRegression()
    oracle_model.fit(X_design, y)
    oracle_predicted = oracle_model.predict(X_design).astype(np.float64, copy=False)

    dopt_model = LinearRegression()
    dopt_model.fit(X_design[coreset_indices], y[coreset_indices])
    dopt_predicted = dopt_model.predict(X_design).astype(np.float64, copy=False)

    oracle_top_predicted = _top_predicted_table(
        candidate_keys,
        oracle_predicted,
        y,
        coreset,
        include_coreset=False,
    )
    dopt_top_predicted = _top_predicted_table(
        candidate_keys,
        dopt_predicted,
        y,
        coreset,
        include_coreset=True,
    )
    error_table = pd.DataFrame(
        [
            _prediction_error_row(
                "all-candidate oracle",
                y,
                oracle_predicted,
                n_train=y.size,
            ),
            _prediction_error_row(
                "dopt coreset",
                y,
                dopt_predicted,
                n_train=coreset_indices.size,
            ),
        ]
    )
    ranking_table = pd.DataFrame(
        [
            _ranking_row("all-candidate oracle", y, oracle_predicted),
            _ranking_row("dopt coreset", y, dopt_predicted),
        ]
    )

    lines = [
        "# Single-Stage DOPP Analysis",
        "",
        f"- Input feature dimension: {input_dim}",
        f"- Design feature dimension used by surrogate diagnostics: {design_dim}",
        f"- PCA components used before QR cleanup: {pca_used}",
        "",
        "## D-Opt Candidate Weight Summary",
        "",
        _dopt_weight_markdown(bundle, n_candidates=y.size),
        "",
        "## Metrics Summary",
        "",
        "Prediction error metrics:",
        "",
        _to_code_table(error_table),
        "",
        "Ranking and top-K metrics:",
        "",
        _to_code_table(ranking_table),
        "",
        "Saved single-stage budget summary:",
        "",
        _to_code_table(_saved_budget_table(bundle, y)),
        "",
        "Metric notes:",
        "",
        "- RMSE: Root Mean Squared Error. Typical prediction error in fitness units, with larger errors penalized more.",
        "- nRMSE: Normalized Root Mean Squared Error. RMSE divided by the range of true fitness labels.",
        "- MAE: Mean Absolute Error. Average absolute prediction error in fitness units.",
        "- R2: Coefficient of Determination. Fraction of true-label variance explained by predictions.",
        "- Kendall tau: Rank correlation based on pairwise ordering agreement across all candidates.",
        "- Spearman: Rank correlation between predicted and true ordering across all candidates.",
        "- Recall@K: Fraction of true top-K candidates recovered by the predicted top-K candidates.",
        "- Best true rank in pred top100: Best oracle rank achieved among the predicted top-100 candidates; lower is better.",
        "",
        "## Linear Surrogate Analysis",
        "",
        "- This is an all-candidate oracle training diagnostic; it uses every true fitness label and is not a realistic deployment setting.",
        f"- Results: `{results_npy}`",
        f"- Fitness CSV: `{fitness_csv}`",
        f"- Feature file: `{features_file}`",
        (
            "- Design feature path: graph-diffused features -> StandardScaler -> "
            f"global PCA-{pca_used} -> QR cleanup."
        ),
        "- Lower predicted fitness and lower true fitness are better.",
        "",
        "Predicted top-100 solutions:",
        "",
        _to_code_table(oracle_top_predicted),
        "",
        "## D-Opt Surrogate Analysis",
        "",
        "- This is the realistic single-stage surrogate diagnostic; it trains only on the global D-opt coreset selected by the saved run.",
        "- The surrogate predicts all candidates, then the table prints the lowest predicted top-100 candidates.",
        "- `in_coreset` marks candidates used to train this D-opt surrogate.",
        "",
        "Predicted top-100 solutions:",
        "",
        _to_code_table(dopt_top_predicted),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a Markdown analysis report for single-stage DOPP."
    )
    parser.add_argument("results_npy", type=Path, help="Path to single_stage_dopp_results.npy.")
    parser.add_argument("fitness_csv", type=Path, help="Matching metrics.csv.")
    parser.add_argument(
        "--features-file",
        type=Path,
        required=True,
        help="Original solution-level feature bundle used by single_stage_dopp.py.",
    )
    parser.add_argument("--metrics", type=str, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.results_npy.exists():
        raise FileNotFoundError(f"Single-stage results file not found: {args.results_npy}")
    if not args.fitness_csv.exists():
        raise FileNotFoundError(f"Fitness CSV file not found: {args.fitness_csv}")
    if not args.features_file.exists():
        raise FileNotFoundError(f"Features file not found: {args.features_file}")

    bundle = load_bundle(args.results_npy)
    candidate_keys = [str(key) for key in bundle.get("candidate_keys", [])]
    if not candidate_keys:
        raise ValueError("Single-stage result bundle is missing candidate_keys")

    _validate_metrics_keys(args.fitness_csv, candidate_keys)
    fitness_dict = load_fitness_scores_from_csv(args.fitness_csv, metrics=args.metrics)
    y = align_fitness(fitness_dict, candidate_keys)

    X, feature_keys, _ = load_features_from_file(
        args.features_file,
        fitness_csv=args.fitness_csv,
    )
    _validate_key_alignment(candidate_keys, feature_keys)

    report = build_report(
        bundle=bundle,
        candidate_keys=candidate_keys,
        y=y,
        X=np.asarray(X, dtype=np.float64),
        results_npy=args.results_npy,
        fitness_csv=args.fitness_csv,
        features_file=args.features_file,
    )

    output = args.output or args.results_npy.parent / "single_stage_dopp_analysis.md"
    write_report(output, report)
    logging.info("Wrote single-stage DOPP analysis report: %s", output)
    print(report)


if __name__ == "__main__":
    main()
