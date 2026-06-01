"""Region quality report for the two-level DOPP baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from algorithms.dopp.baseline_analysis_utils import (
    all_selected_regions,
    default_report_path,
    format_float,
    load_inputs,
    per_region_map,
    rank_array,
    round_region_best,
    safe_corr,
    selected_regions,
    truth_for_regions,
    write_report,
)
from algorithms.dopp.d_opt import frank_wolfe_d_optimal
from algorithms.dopp.loaders import load_features_from_file


def _format_table_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(lambda value: "yes" if bool(value) else "no")
        elif pd.api.types.is_float_dtype(out[col]):
            col_name = str(col).lower()
            integer_like = (
                col_name
                in {
                    "region_id",
                    "selected_round",
                    "size",
                    "true_region_rank",
                    "true_best_global_rank",
                    "worst_rank",
                    "coreset_size",
                    "overlap",
                    "best_true_rank_in_pred_top10",
                }
                or col_name.endswith("_hits")
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


def _top_weight_indices(weights: np.ndarray, top_k_frac: float) -> np.ndarray:
    k = max(10, int(np.ceil(float(top_k_frac) * len(weights))))
    k = min(k, len(weights))
    return np.argsort(weights)[-k:][::-1].astype(np.int64)


def _top_predicted_indices(predicted: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0:
        return np.array([], dtype=np.int64)
    k_eff = min(int(top_k), predicted.size)
    return np.argsort(predicted, kind="stable")[:k_eff].astype(np.int64)


def _rank_ascending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.int64)
    return ranks


def _local_feature_space(
    X_std: np.ndarray,
    members: np.ndarray,
    inner_pca_components: int,
    random_state: int,
) -> np.ndarray:
    X_r_std = X_std[members]
    if members.size <= 1:
        return X_r_std

    local_pca_dim = min(
        inner_pca_components,
        max(1, members.size - 1),
        X_r_std.shape[1],
    )
    return PCA(n_components=local_pca_dim, random_state=random_state).fit_transform(X_r_std)


def _safe_fit_predict(train_X: np.ndarray, train_y: np.ndarray, all_X: np.ndarray) -> np.ndarray:
    model = LinearRegression()
    model.fit(train_X, train_y)
    return model.predict(all_X).astype(np.float64, copy=False)


def _validate_feature_alignment(
    result_keys: Sequence[str],
    feature_keys: Sequence[str],
) -> None:
    if len(result_keys) != len(feature_keys):
        raise ValueError(
            "Feature file candidate count does not match result bundle: "
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


def _load_candidate_proxies(
    csv_path: Path,
    candidate_keys: Sequence[str],
) -> np.ndarray:
    df = pd.read_csv(csv_path)
    required = {"Key", "Cut_size", "Area_imbalance"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Metrics CSV is missing required proxy columns: {sorted(missing_cols)}"
        )

    df = df.copy()
    df["Key"] = df["Key"].astype(str)
    df = df.drop_duplicates(subset="Key", keep="first")
    df["Cut_size"] = pd.to_numeric(df["Cut_size"], errors="coerce")
    df["Area_imbalance"] = pd.to_numeric(df["Area_imbalance"], errors="coerce")

    lookup: Dict[str, tuple[float, float]] = {}
    for row in df.itertuples(index=False):
        key = str(getattr(row, "Key"))
        cut_size = float(getattr(row, "Cut_size"))
        area_imbalance = float(getattr(row, "Area_imbalance"))
        if np.isfinite(cut_size) and np.isfinite(area_imbalance):
            lookup[key] = (cut_size, area_imbalance)

    proxies = np.full((len(candidate_keys), 2), np.nan, dtype=np.float64)
    missing_keys: List[str] = []
    for idx, key in enumerate(candidate_keys):
        values = lookup.get(str(key))
        if values is None:
            missing_keys.append(str(key))
            continue
        proxies[idx, 0] = values[0]
        proxies[idx, 1] = values[1]

    if missing_keys:
        raise ValueError(
            "Missing finite Cut_size/Area_imbalance for candidate keys "
            f"(showing first 10 of {len(missing_keys)}): {missing_keys[:10]}"
        )
    return proxies


def _add_region_proxy_ranges(
    region_truth: pd.DataFrame,
    regions: List[List[int]],
    proxies: np.ndarray,
) -> pd.DataFrame:
    table = region_truth.copy()
    cut_ranges: List[float] = []
    area_imbalance_ranges: List[float] = []
    for members_list in regions:
        members = np.asarray(members_list, dtype=np.int64)
        if members.size == 0:
            cut_ranges.append(float("nan"))
            area_imbalance_ranges.append(float("nan"))
            continue
        cut_values = proxies[members, 0]
        area_imbalance_values = proxies[members, 1]
        cut_ranges.append(float(np.max(cut_values) - np.min(cut_values)))
        area_imbalance_ranges.append(
            float(np.max(area_imbalance_values) - np.min(area_imbalance_values))
        )

    table["cut_size_range"] = cut_ranges
    table["area_imbalance_range"] = area_imbalance_ranges
    return table


def _evaluated_region_gaps(bundle: Dict, region_truth: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    evaluated_best: Dict[int, float] = {}
    for round_key in ("round1", "round2"):
        eval_map = per_region_map(bundle, round_key, "evaluated_per_region")
        for region_id, evaluated in eval_map.items():
            if evaluated:
                evaluated_best[int(region_id)] = float(y[np.asarray(evaluated, dtype=np.int64)].min())

    if not evaluated_best:
        evaluated_best.update(round_region_best(bundle, "round1"))
        evaluated_best.update(round_region_best(bundle, "round2"))

    if not evaluated_best:
        return pd.DataFrame()

    lookup = region_truth.set_index("region_id")
    rows: List[Dict[str, object]] = []
    for region_id, found_best in evaluated_best.items():
        if region_id not in lookup.index:
            continue
        true_best = float(lookup.loc[region_id, "true_best_fitness"])
        rows.append(
            {
                "region_id": int(region_id),
                "found_best_fitness": float(found_best),
                "true_best_fitness": true_best,
                "gap": float(found_best - true_best),
            }
        )
    return pd.DataFrame(rows)


def _top_pct_hits(
    regions: List[List[int]],
    y: np.ndarray,
    top_pct: float,
) -> np.ndarray:
    ranks = rank_array(y)
    cutoff = max(1, int(np.floor(y.size * float(top_pct) / 100.0)))
    hits = np.zeros(len(regions), dtype=np.int64)
    for region_id, members_list in enumerate(regions):
        members = np.asarray(members_list, dtype=np.int64)
        if members.size > 0:
            hits[region_id] = int(np.sum(ranks[members] <= cutoff))
    return hits


def _selected_round_column(bundle: Dict, region_truth: pd.DataFrame) -> pd.Series:
    selected_round = pd.Series(0, index=region_truth.index, dtype=np.int64)
    r1 = set(selected_regions(bundle, "round1"))
    r2 = set(selected_regions(bundle, "round2"))
    selected_round.loc[region_truth["region_id"].isin(r1)] = 1
    selected_round.loc[
        (region_truth["region_id"].isin(r2)) & (selected_round == 0)
    ] = 2
    return selected_round


def _per_region_table_markdown(
    bundle: Dict,
    region_truth: pd.DataFrame,
    gaps: pd.DataFrame,
    top_pct_col: str,
) -> str:
    table = region_truth.copy()
    table["selected_round"] = _selected_round_column(bundle, table)
    if len(gaps) > 0:
        table = table.merge(
            gaps[["region_id", "found_best_fitness", "gap"]],
            on="region_id",
            how="left",
        )

    preferred_cols = [
        "region_id",
        "selected_round",
        "size",
        "true_region_rank",
        "true_best_rank",
        "median_rank",
        "p10_rank",
        "p25_rank",
        "p75_rank",
        "p90_rank",
        "worst_rank",
        "true_best_fitness",
        "found_best_fitness",
        "gap",
        top_pct_col,
    ]
    top_k_cols = [c for c in table.columns if c.startswith("top_") and c.endswith("_hits")]
    cols = [c for c in preferred_cols + top_k_cols if c in table.columns]
    table = table[cols].sort_values(["true_best_rank", "region_id"]).reset_index(drop=True)
    try:
        return table.to_markdown(index=False)
    except ImportError:
        return table.to_string(index=False)


def _true_top10_basic_table(bundle: Dict, region_truth: pd.DataFrame) -> pd.DataFrame:
    table = region_truth.nsmallest(10, "true_region_rank").copy()
    table["selected_round"] = _selected_round_column(bundle, table)
    table = table.rename(
        columns={
            "true_best_fitness": "best_fitness",
            "true_worst_fitness": "worst_fitness",
            "true_best_rank": "true_best_global_rank",
            "fitness_variance": "fitness_var",
        }
    )
    cols = [
        "true_region_rank",
        "region_id",
        "selected_round",
        "best_fitness",
        "worst_fitness",
        "median_fitness",
        "true_best_global_rank",
        "worst_rank",
        "fitness_var",
        "cut_size_range",
        "area_imbalance_range",
    ]
    return table[[c for c in cols if c in table.columns]].reset_index(drop=True)


def _true_top10_inner_surrogate_table(
    bundle: Dict,
    regions: List[List[int]],
    region_truth: pd.DataFrame,
    y: np.ndarray,
    X: np.ndarray,
) -> pd.DataFrame:
    config = bundle.get("config", {})
    inner_top_k_frac = float(config.get("inner_top_k_frac", 0.2))
    inner_prediction_top_k = int(config.get("inner_prediction_top_k", 0))
    pca_components = int(config.get("pca_components", 10))
    fw_tol = float(config.get("fw_tol", 1e-2))
    fw_step_scheme = str(config.get("fw_step_scheme", "1/t"))
    fw_epsilon = float(config.get("fw_epsilon", 0.0))
    random_state = int(config.get("random_state", 0))

    X_std = StandardScaler().fit_transform(np.asarray(X, dtype=np.float64))
    inner_pca_components = min(pca_components, X_std.shape[1], X_std.shape[0])
    top10 = region_truth.nsmallest(10, "true_region_rank").copy()
    top10["selected_round"] = _selected_round_column(bundle, top10)

    rows: List[Dict[str, object]] = []
    for _, region_row in top10.iterrows():
        region_id = int(region_row["region_id"])
        members = np.asarray(regions[region_id], dtype=np.int64)
        if members.size == 0:
            continue

        X_r = _local_feature_space(
            X_std=X_std,
            members=members,
            inner_pca_components=inner_pca_components,
            random_state=random_state,
        )

        previous_logging_disable = logging.root.manager.disable
        logging.disable(logging.INFO)
        try:
            dopt_weights, _ = frank_wolfe_d_optimal(
                X_r,
                tol=fw_tol,
                step_scheme=fw_step_scheme,
                epsilon=fw_epsilon,
                verbose=False,
            )
        finally:
            logging.disable(previous_logging_disable)

        dopt_weights = np.asarray(dopt_weights, dtype=np.float64)
        dopt_local = _top_weight_indices(dopt_weights, inner_top_k_frac)
        dopt_global = members[dopt_local]
        predicted = _safe_fit_predict(
            train_X=X_r[dopt_local],
            train_y=y[dopt_global],
            all_X=X_r,
        )
        surrogate_local = _top_predicted_indices(predicted, inner_prediction_top_k)
        evaluated_local = np.unique(np.concatenate([dopt_local, surrogate_local]))
        evaluated_global = members[evaluated_local]

        true_values = y[members]
        true_ranks = _rank_ascending(true_values)
        pred_top10_local = _top_predicted_indices(predicted, 10)
        best_true_rank_in_pred_top10 = int(true_ranks[pred_top10_local].min())
        corr = safe_corr(predicted, true_values)

        true_best = float(region_row["true_best_fitness"])
        dopt_best = float(y[dopt_global].min())
        final_best = float(y[evaluated_global].min())
        rows.append(
            {
                "true_region_rank": int(region_row["true_region_rank"]),
                "region_id": region_id,
                "selected_round": int(region_row["selected_round"]),
                "coreset_size": int(dopt_local.size),
                "overlap": int(len(set(dopt_local.tolist()) & set(surrogate_local.tolist()))),
                "dopt_best": dopt_best,
                "final_best": final_best,
                "gap_after": float(final_best - true_best),
                "best_true_rank_in_pred_top10": best_true_rank_in_pred_top10,
                "spearman": corr["spearman"],
            }
        )
    return pd.DataFrame(rows)


def _true_top10_diagnostics_markdown(
    bundle: Dict,
    regions: List[List[int]],
    region_truth: pd.DataFrame,
    y: np.ndarray,
    X: Optional[np.ndarray],
) -> str:
    lines = [
        "## True Top-10 Region Diagnostics",
        "",
        "Basic region stats:",
        "",
        _to_code_table(_true_top10_basic_table(bundle, region_truth)),
        "",
    ]

    if X is None:
        lines.extend(
            [
                "Inner surrogate performance:",
                "",
                "- Skipped because `--features-file` was not provided.",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "Inner surrogate performance:",
            "",
            "- Offline reconstruction is diagnostic only and is not counted as realistic oracle usage.",
            "",
            _to_code_table(
                _true_top10_inner_surrogate_table(
                    bundle=bundle,
                    regions=regions,
                    region_truth=region_truth,
                    y=y,
                    X=X,
                )
            ),
        ]
    )
    return "\n".join(lines)


def build_report(
    bundle: Dict,
    regions: List[List[int]],
    y: np.ndarray,
    top_k_truth: Sequence[int],
    top_pct: float,
    include_table: bool,
    results_npy: Path,
    fitness_csv: Path,
    proxies: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    features_file: Optional[Path] = None,
) -> str:
    top_k_for_truth = tuple(sorted(set(int(k) for k in top_k_truth) | {10, 20, 50, 100}))
    region_truth = truth_for_regions(regions, y, top_k_for_truth)
    if proxies is not None:
        region_truth = _add_region_proxy_ranges(region_truth, regions, proxies)
    top_pct_col = f"top{top_pct:g}pct_hits"
    region_truth[top_pct_col] = _top_pct_hits(regions, y, top_pct)
    sizes = region_truth["size"].to_numpy(dtype=np.int64)
    best_ranks = region_truth["true_best_rank"].dropna().to_numpy(dtype=np.float64)
    round1_regions = set(selected_regions(bundle, "round1"))
    selected_all = set(all_selected_regions(bundle))
    round1_df = region_truth[region_truth["region_id"].isin(round1_regions)]
    selected_df = region_truth[region_truth["region_id"].isin(selected_all)]
    gaps = _evaluated_region_gaps(bundle, region_truth, y)

    corr = safe_corr(
        region_truth["true_best_rank"].to_numpy(dtype=np.float64),
        region_truth["median_rank"].to_numpy(dtype=np.float64),
    )

    lines: List[str] = [
        "# Region Quality Analysis",
        "",
        f"- Results: `{results_npy}`",
        f"- Fitness CSV: `{fitness_csv}`",
        *([f"- Feature file: `{features_file}`"] if features_file is not None else []),
        "- Lower fitness and lower rank are better.",
        "",
        "## Region Size and Quality",
        "",
        f"- Regions: **{len(region_truth)}**; candidates: **{int(sizes.sum())}**.",
        (
            f"- Region size distribution: min={int(sizes.min())}, "
            f"median={format_float(float(np.median(sizes)), 1)}, "
            f"mean={format_float(float(sizes.mean()), 2)}, max={int(sizes.max())}."
        ),
        (
            f"- Best global rank per region: min={int(np.nanmin(best_ranks))}, "
            f"median={format_float(float(np.nanmedian(best_ranks)), 1)}, "
            f"max={int(np.nanmax(best_ranks))}."
        ),
        (
            "- Correlation between a region's best rank and median candidate rank: "
            f"Spearman={format_float(corr['spearman'])}, "
            f"Kendall={format_float(corr['kendall'])}."
        ),
        "",
        "## Global Top-K Distribution",
        "",
    ]

    for k in top_k_truth:
        col = f"top_{k}_hits"
        if col not in region_truth:
            continue
        regions_with_hits = int((region_truth[col] > 0).sum())
        selected_hits = int(selected_df[col].sum()) if len(selected_df) else 0
        round1_hits = int(round1_df[col].sum()) if len(round1_df) else 0
        selected_hit_regions = (
            selected_df.loc[selected_df[col] > 0, "region_id"].astype(int).tolist()
            if len(selected_df)
            else []
        )
        lines.append(
            f"- Top-{k}: appears in {regions_with_hits} region(s); "
            f"Round-1 captures {round1_hits}/{min(int(k), int(region_truth[col].sum()))}; "
            f"Round-1+Round-2 captures {selected_hits}/{min(int(k), int(region_truth[col].sum()))}. "
            f"Selected hit regions: {selected_hit_regions[:20]}."
        )

    top_pct_hits = region_truth[top_pct_col].to_numpy(dtype=np.int64)
    selected_top_pct_hits = int(selected_df[top_pct_col].sum()) if len(selected_df) else 0
    round1_top_pct_hits = int(round1_df[top_pct_col].sum()) if len(round1_df) else 0
    lines.extend(
        [
            "",
            f"## Top-{top_pct:g}% Concentration",
            "",
            (
                f"- Global top-{top_pct:g}% candidates are spread across "
                f"{int(np.sum(top_pct_hits > 0))} region(s); "
                f"median hits per region={format_float(float(np.median(top_pct_hits)), 1)}, "
                f"max hits in one region={int(top_pct_hits.max())}."
            ),
            (
                f"- Selected-region capture of top-{top_pct:g}% candidates: "
                f"Round-1={round1_top_pct_hits}, Round-1+Round-2={selected_top_pct_hits}."
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Selected Region Quality",
            "",
            f"- Round-1 selected regions: {selected_regions(bundle, 'round1')}",
            f"- Round-2 newly evaluated regions: {selected_regions(bundle, 'round2')}",
        ]
    )
    if len(selected_df) > 0:
        lines.extend(
            [
                (
                    "- Selected-region median true region rank: "
                    f"{format_float(float(selected_df['true_region_rank'].median()), 1)} "
                    f"(all-region median {format_float(float(region_truth['true_region_rank'].median()), 1)})."
                ),
                (
                    "- Selected-region best true region rank: "
                    f"{int(selected_df['true_region_rank'].min())}."
                ),
            ]
        )
    else:
        lines.append("- No selected regions were found in the result bundle.")

    lines.extend(["", "## Found-Best Gap", ""])
    if len(gaps) > 0:
        hit_fraction = float(np.mean(np.isclose(gaps["gap"].to_numpy(dtype=np.float64), 0.0)))
        lines.extend(
            [
                (
                    "- Gap between the true best in each evaluated region and the best "
                    f"found by the baseline: mean={format_float(float(gaps['gap'].mean()))}, "
                    f"median={format_float(float(gaps['gap'].median()))}, "
                    f"max={format_float(float(gaps['gap'].max()))}."
                ),
                f"- Exact true-region-best hit rate among evaluated regions: {hit_fraction:.2%}.",
            ]
        )
    else:
        lines.append("- TODO: no evaluated region-best values were found in the result bundle.")

    lines.extend(
        [
            "",
            _true_top10_diagnostics_markdown(
                bundle=bundle,
                regions=regions,
                region_truth=region_truth,
                y=y,
                X=X,
            ),
        ]
    )

    if include_table:
        lines.extend(
            [
                "",
                "## Per-Region Table",
                "",
                _per_region_table_markdown(bundle, region_truth, gaps, top_pct_col),
            ]
        )

    lines.extend(
        [
            "",
            "## Main Questions",
            "",
            "- Are high-quality candidates concentrated in a small number of regions?",
            "- Are selected regions better than a typical/random region?",
            "- Does the best candidate in a region reflect the overall quality of that region?",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the region quality analysis report.")
    parser.add_argument("results_npy", type=Path, help="Path to two_level_results.npy.")
    parser.add_argument("fitness_csv", type=Path, help="Matching metrics.csv.")
    parser.add_argument(
        "--features-file",
        type=Path,
        default=None,
        help="Original solution-level feature bundle used by two_level_dopp.py.",
    )
    parser.add_argument("--metrics", type=str, nargs="+", default=None)
    parser.add_argument("--top-k-truth", type=int, nargs="+", default=(10, 20, 50, 100))
    parser.add_argument(
        "--top-pct",
        type=float,
        default=10.0,
        help="Top-percent cutoff for the region concentration summary.",
    )
    parser.add_argument(
        "--include-table",
        action="store_true",
        help="Embed the full per-region table in the Markdown report.",
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
    proxies = _load_candidate_proxies(args.fitness_csv, candidate_keys)
    X = None
    if args.features_file is not None:
        if not args.features_file.exists():
            raise FileNotFoundError(f"Features file not found: {args.features_file}")
        X, feature_keys, _ = load_features_from_file(
            args.features_file,
            fitness_csv=args.fitness_csv,
        )
        _validate_feature_alignment(candidate_keys, feature_keys)
    output = args.output or default_report_path(args.results_npy, "region_quality_analysis.md")
    report = build_report(
        bundle,
        regions,
        y,
        args.top_k_truth,
        args.top_pct,
        args.include_table,
        args.results_npy,
        args.fitness_csv,
        proxies=proxies,
        X=X,
        features_file=args.features_file,
    )
    write_report(output, report)
    logging.info("Wrote region quality report: %s", output)
    print(report)


if __name__ == "__main__":
    main()
