"""Region-level surrogate oracle-audit report for the two-level DOPP baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from algorithms.dopp.baseline_analysis_utils import (
    default_report_path,
    format_float,
    format_pct,
    load_inputs,
    round_region_best,
    safe_corr,
    selected_regions,
    truth_for_regions,
    write_report,
)


def _x_region(bundle: Dict) -> np.ndarray:
    x_region = bundle.get("region_features", {}).get("X_region")
    if x_region is None:
        raise ValueError("Result bundle is missing region_features.X_region")
    return np.asarray(x_region, dtype=np.float64)


def _truth_table(
    regions: List[List[int]],
    y: np.ndarray,
    candidate_keys: Sequence[str],
) -> pd.DataFrame:
    truth = truth_for_regions(regions, y, top_k_truth=(10,))
    truth = truth.rename(
        columns={
            "size": "region_size",
            "true_best_fitness": "true_label",
            "true_region_rank": "true_rank",
        }
    )
    best_keys: List[str] = []
    for idx in truth["true_best_index"].tolist():
        idx = int(idx)
        if 0 <= idx < len(candidate_keys):
            best_keys.append(str(candidate_keys[idx]))
        else:
            best_keys.append(str(idx))
    truth["best_candidate_key_or_index_in_region"] = best_keys
    truth["best_candidate_true_label"] = truth["true_label"]
    return truth


def _prediction_pool(bundle: Dict, truth: pd.DataFrame, prediction_pool: str) -> List[int]:
    valid_regions = truth.loc[truth["true_label"].notna(), "region_id"].astype(int).tolist()
    if prediction_pool == "all":
        return valid_regions

    round1 = set(selected_regions(bundle, "round1"))
    return [r for r in valid_regions if r not in round1]


def _training_labels(
    bundle: Dict,
    truth: pd.DataFrame,
    region_label_mode: str,
) -> Tuple[List[int], np.ndarray]:
    round1_regions = selected_regions(bundle, "round1")
    truth_by_region = truth.set_index("region_id")["true_label"].to_dict()
    observed_by_region = round_region_best(bundle, "round1")

    train_ids: List[int] = []
    labels: List[float] = []
    for region_id in round1_regions:
        if region_label_mode == "observed":
            label = observed_by_region.get(int(region_id))
        elif region_label_mode == "oracle":
            # Oracle labels are an analysis-only upper bound. They use the full
            # offline reward table and must not be counted as realistic PPA calls.
            label = truth_by_region.get(int(region_id))
        else:
            raise ValueError(f"Unknown region_label_mode: {region_label_mode!r}")

        if label is None or not np.isfinite(label):
            continue
        train_ids.append(int(region_id))
        labels.append(float(label))

    if not train_ids:
        raise ValueError(f"No finite training labels found for mode={region_label_mode}")
    return train_ids, np.asarray(labels, dtype=np.float64)


def _rank_within_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["predicted_rank"] = (
        out["predicted_label"].rank(method="first", ascending=True).astype(int)
    )
    return out


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    sst = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if sst <= 0:
        return float("nan")
    sse = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - sse / sst


def _mode_frame(
    bundle: Dict,
    truth: pd.DataFrame,
    prediction_pool: str,
    region_label_mode: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    train_ids, train_y = _training_labels(bundle, truth, region_label_mode)
    return _fit_and_evaluate_frame(
        bundle=bundle,
        truth=truth,
        train_ids=train_ids,
        train_y=train_y,
        prediction_pool=prediction_pool,
        region_label_mode=region_label_mode,
    )


def _all_region_oracle_frame(
    bundle: Dict,
    truth: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    valid = truth[truth["true_label"].notna()]
    train_ids = valid["region_id"].astype(int).tolist()
    train_y = valid["true_label"].to_numpy(dtype=np.float64)
    return _fit_and_evaluate_frame(
        bundle=bundle,
        truth=truth,
        train_ids=train_ids,
        train_y=train_y,
        prediction_pool="all",
        region_label_mode="all_region_oracle",
    )


def _fit_and_evaluate_frame(
    bundle: Dict,
    truth: pd.DataFrame,
    train_ids: Sequence[int],
    train_y: np.ndarray,
    prediction_pool: str,
    region_label_mode: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    model = LinearRegression()
    x_region = _x_region(bundle)
    train_arr = np.asarray(train_ids, dtype=np.int64)
    model.fit(x_region[train_arr], train_y)
    predicted_all = model.predict(x_region).astype(np.float64, copy=False)

    pool_ids = set(_prediction_pool(bundle, truth, prediction_pool))
    round1 = set(selected_regions(bundle, "round1"))
    round2 = set(selected_regions(bundle, "round2"))

    df = truth.copy()
    df["predicted_label"] = predicted_all[df["region_id"].to_numpy(dtype=np.int64)]
    df["selected_in_round1"] = df["region_id"].isin(round1)
    df["selected_in_round2"] = df["region_id"].isin(round2)
    df["in_prediction_pool"] = df["region_id"].isin(pool_ids)

    eval_df = df[
        df["in_prediction_pool"]
        & df["predicted_label"].notna()
        & df["true_label"].notna()
    ].copy()
    eval_df = _rank_within_pool(eval_df)

    summary = _metrics_for_mode(eval_df, train_ids, region_label_mode, prediction_pool)
    return eval_df, summary


def _metrics_for_mode(
    eval_df: pd.DataFrame,
    train_ids: Sequence[int],
    region_label_mode: str,
    prediction_pool: str,
    top_k: int = 10,
) -> Dict[str, object]:
    y_true = eval_df["true_label"].to_numpy(dtype=np.float64)
    y_pred = eval_df["predicted_label"].to_numpy(dtype=np.float64)
    if y_true.size == 0:
        return {
            "mode": region_label_mode,
            "prediction_pool": prediction_pool,
            "n_train": int(len(train_ids)),
            "n_eval": 0,
            "rmse": np.nan,
            "normalized_rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "kendall_tau": np.nan,
            "spearman": np.nan,
            "recall_at_10": np.nan,
            "best_true_rank_in_predicted_top10": np.nan,
        }

    residual = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residual**2)))
    true_range = float(np.max(y_true) - np.min(y_true))
    corr = safe_corr(y_pred, y_true)

    k_eff = min(top_k, len(eval_df))
    pred_top = eval_df.nsmallest(k_eff, "predicted_label")
    true_top = eval_df.nsmallest(k_eff, "true_label")
    pred_ids = set(pred_top["region_id"].astype(int).tolist())
    true_ids = set(true_top["region_id"].astype(int).tolist())

    return {
        "mode": region_label_mode,
        "prediction_pool": prediction_pool,
        "n_train": int(len(train_ids)),
        "n_eval": int(len(eval_df)),
        "rmse": rmse,
        "normalized_rmse": rmse / true_range if true_range > 0 else np.nan,
        "mae": float(np.mean(np.abs(residual))),
        "r2": _r2_score(y_true, y_pred),
        "kendall_tau": corr["kendall"],
        "spearman": corr["spearman"],
        "recall_at_10": float(len(pred_ids & true_ids) / max(k_eff, 1)),
        "best_true_rank_in_predicted_top10": int(pred_top["true_rank"].min()),
    }


def _top10_tables(eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    k_eff = min(10, len(eval_df))
    predicted_top = eval_df.nsmallest(k_eff, "predicted_label").copy()
    true_top = eval_df.nsmallest(k_eff, "true_label").copy()
    predicted_top_ids = set(predicted_top["region_id"].astype(int).tolist())
    true_top["missed_by_predicted_top10"] = ~true_top["region_id"].isin(predicted_top_ids)

    predicted_cols = [
        "predicted_rank",
        "region_id",
        "predicted_label",
        "true_label",
        "true_rank",
        "region_size",
        "selected_in_round1",
        "selected_in_round2",
        "best_candidate_key_or_index_in_region",
        "best_candidate_true_label",
    ]
    true_cols = [
        "true_rank",
        "region_id",
        "true_label",
        "predicted_label",
        "predicted_rank",
        "region_size",
        "selected_in_round1",
        "selected_in_round2",
        "missed_by_predicted_top10",
        "best_candidate_key_or_index_in_region",
        "best_candidate_true_label",
    ]
    return predicted_top[predicted_cols], true_top[true_cols]


def _selection_label(row: pd.Series) -> str:
    in_r1 = bool(row.get("selected_in_round1", False))
    in_r2 = bool(row.get("selected_in_round2", False))
    if in_r1 and in_r2:
        return "R1+R2"
    if in_r1:
        return "R1"
    if in_r2:
        return "R2"
    return "no"


def _yes_no(value: object) -> str:
    return "yes" if bool(value) else "no"


def _in_coreset_column(table: pd.DataFrame) -> pd.Series:
    return table["selected_in_round1"].map(_yes_no)


def _compact_predicted_table(predicted_top: pd.DataFrame) -> pd.DataFrame:
    compact = pd.DataFrame(
        {
            "pred_rank": predicted_top["predicted_rank"],
            "region": predicted_top["region_id"],
            "pred": predicted_top["predicted_label"],
            "true": predicted_top["true_label"],
            "true_rank": predicted_top["true_rank"],
            "size": predicted_top["region_size"],
            "in_coreset": _in_coreset_column(predicted_top),
            "selected": predicted_top.apply(_selection_label, axis=1),
            "best_candidate": predicted_top["best_candidate_key_or_index_in_region"],
            "best_candidate_true": predicted_top["best_candidate_true_label"],
        }
    )
    return compact


def _compact_true_table(true_top: pd.DataFrame) -> pd.DataFrame:
    compact = pd.DataFrame(
        {
            "true_rank": true_top["true_rank"],
            "region": true_top["region_id"],
            "true": true_top["true_label"],
            "pred": true_top["predicted_label"],
            "pred_rank": true_top["predicted_rank"],
            "missed": true_top["missed_by_predicted_top10"].map(_yes_no),
            "in_coreset": _in_coreset_column(true_top),
            "selected": true_top.apply(_selection_label, axis=1),
            "best_candidate": true_top["best_candidate_key_or_index_in_region"],
            "best_candidate_true": true_top["best_candidate_true_label"],
        }
    )
    return compact


def _metrics_markdown(metrics: Sequence[Dict[str, object]]) -> str:
    error_rows = []
    ranking_rows = []
    for item in metrics:
        error_rows.append(
            {
                "mode": _mode_name(str(item["mode"])),
                "pool": item["prediction_pool"],
                "n_train": item["n_train"],
                "n_eval": item["n_eval"],
                "RMSE": item["rmse"],
                "nRMSE": item["normalized_rmse"],
                "MAE": item["mae"],
                "R2": item["r2"],
            }
        )
        ranking_rows.append(
            {
                "mode": _mode_name(str(item["mode"])),
                "Kendall tau": item["kendall_tau"],
                "Spearman": item["spearman"],
                "Recall@10": item["recall_at_10"],
                "Best true rank in pred top10": item[
                    "best_true_rank_in_predicted_top10"
                ],
            }
        )
    return "\n".join(
        [
            "Prediction error metrics:",
            "",
            _to_code_table(pd.DataFrame(error_rows)),
            "",
            "Ranking and top-10 metrics:",
            "",
            _to_code_table(pd.DataFrame(ranking_rows)),
        ]
    )


def _metric_notes_markdown() -> str:
    return "\n".join(
        [
            "Metric notes:",
            "",
            "- RMSE: Root Mean Squared Error. Typical prediction error in fitness units, with larger errors penalized more.",
            "- nRMSE: Normalized Root Mean Squared Error. RMSE divided by the range of true region labels; lower means smaller error relative to the full quality spread.",
            "- MAE: Mean Absolute Error. Average absolute prediction error in fitness units.",
            "- R2: Coefficient of Determination. Fraction of true-label variance explained by predictions; higher is better, and negative means worse than predicting the mean.",
            "- Kendall tau: Rank correlation based on pairwise ordering agreement; higher positive values mean better ranking agreement.",
            "- Spearman: Rank correlation between predicted and true ordering; higher positive values mean better monotonic ranking agreement.",
            "- Recall@10: Fraction of true top-10 regions recovered by the predicted top-10 regions.",
            "- Best true rank in pred top10: Best oracle rank achieved among the predicted top-10 regions; lower is better.",
        ]
    )


def _mode_name(mode: str) -> str:
    if mode == "observed":
        return "observed"
    if mode == "oracle":
        return "oracle"
    if mode == "all_region_oracle":
        return "all-region oracle"
    return mode


def _format_table_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            col_name = str(col).lower()
            integer_like = (
                col_name in {"region", "region_id", "size", "region_size", "n_train", "n_eval"}
                or col_name.endswith("_rank")
                or col_name.endswith("_id")
                or col_name.startswith("n_")
                or "best true rank" in col_name
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
    return "\n".join(
        [
            "```text",
            out.to_string(index=False),
            "```",
        ]
    )


def _mode_section(mode: str, eval_df: pd.DataFrame, summary: Dict[str, object]) -> str:
    predicted_top, true_top = _top10_tables(eval_df)
    if mode == "observed":
        mode_label = "Observed-Label Surrogate"
    elif mode == "oracle":
        mode_label = "Oracle-Label Surrogate"
    elif mode == "all_region_oracle":
        mode_label = "All-Region Oracle Training Diagnostic"
    else:
        mode_label = mode

    lines = [
        f"## {mode_label}",
        "",
        "### Predicted Top-10 Regions",
        "",
        _to_code_table(_compact_predicted_table(predicted_top)),
        "",
        "### True Top-10 Regions",
        "",
        _to_code_table(_compact_true_table(true_top)),
        "",
    ]
    return "\n".join(lines)


def build_report(
    bundle: Dict,
    candidate_keys: Sequence[str],
    regions: List[List[int]],
    y: np.ndarray,
    region_label_mode: str,
    prediction_pool: str,
    results_npy: Path,
    fitness_csv: Path,
) -> str:
    modes = ["observed", "oracle"] if region_label_mode == "both" else [region_label_mode]
    truth = _truth_table(regions, y, candidate_keys)

    mode_results: List[Tuple[str, pd.DataFrame, Dict[str, object]]] = []
    for mode in modes:
        eval_df, summary = _mode_frame(
            bundle=bundle,
            truth=truth,
            prediction_pool=prediction_pool,
            region_label_mode=mode,
        )
        mode_results.append((mode, eval_df, summary))

    all_eval_df, all_summary = _all_region_oracle_frame(bundle, truth)
    mode_results.append(("all_region_oracle", all_eval_df, all_summary))

    metrics = [summary for _, _, summary in mode_results]
    round1 = selected_regions(bundle, "round1")
    round2 = selected_regions(bundle, "round2")

    lines = [
        "# Region-Level Surrogate Analysis",
        "",
        f"- Results: `{results_npy}`",
        f"- Fitness CSV: `{fitness_csv}`",
        f"- Region label mode: `{region_label_mode}`",
        f"- Prediction/evaluation pool: `{prediction_pool}` "
        "(`eligible` means non-coreset regions only; `all` includes coreset regions).",
        "- Lower predicted label and lower true label both mean better.",
        "- True oracle region label is `min(candidate fitness)` inside the region.",
        "- `true_rank` is the oracle rank among all regions; `in_coreset` marks Round-1 selected regions.",
        "- Observed region label is the best evaluated fitness found inside a Round-1 selected region.",
        "- Oracle-label mode is analysis-only and is used as an upper-bound diagnostic for label noise.",
        "- All-region oracle training is an in-sample model-capacity diagnostic; it uses all true region labels and does not use D-opt coreset selection.",
        "",
        "## Setup",
        "",
        f"- Regions: **{len(regions)}**.",
        f"- Round-1 selected regions: **{len(round1)}**.",
        f"- Round-2 newly evaluated regions in saved run: **{len(round2)}**.",
        "",
        "## Metrics Summary",
        "",
        _metrics_markdown(metrics),
        "",
        _metric_notes_markdown(),
        "",
    ]

    for mode, eval_df, summary in mode_results:
        lines.append(_mode_section(mode, eval_df, summary))

    lines.extend(
        [
            "## Main Diagnostic Question",
            "",
            "Does the region surrogate fail because the current region features/model cannot rank good regions, "
            "or because the realistic Round-1 observed labels are noisy?",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the region-level surrogate oracle-audit report.")
    parser.add_argument("results_npy", type=Path, help="Path to two_level_results.npy.")
    parser.add_argument("fitness_csv", type=Path, help="Matching metrics.csv.")
    parser.add_argument("--metrics", type=str, nargs="+", default=None)
    parser.add_argument(
        "--region-label-mode",
        choices=("observed", "oracle", "both"),
        default="observed",
        help="Training labels for the region surrogate. Oracle is analysis-only.",
    )
    parser.add_argument(
        "--prediction-pool",
        choices=("eligible", "all"),
        default="all",
        help="Regions to rank/evaluate. eligible excludes Round-1 training regions.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    bundle, candidate_keys, y, regions = load_inputs(
        args.results_npy,
        args.fitness_csv,
        args.metrics,
    )
    output = args.output or default_report_path(args.results_npy, "region_surrogate_analysis.md")
    report = build_report(
        bundle=bundle,
        candidate_keys=candidate_keys,
        regions=regions,
        y=y,
        region_label_mode=args.region_label_mode,
        prediction_pool=args.prediction_pool,
        results_npy=args.results_npy,
        fitness_csv=args.fitness_csv,
    )
    write_report(output, report)
    logging.info("Wrote region-level surrogate report: %s", output)
    print(report)


if __name__ == "__main__":
    main()
