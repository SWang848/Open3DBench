"""Region quality report for the two-level DOPP baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

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


def build_report(
    bundle: Dict,
    regions: List[List[int]],
    y: np.ndarray,
    top_k_truth: Sequence[int],
    top_pct: float,
    include_table: bool,
    results_npy: Path,
    fitness_csv: Path,
) -> str:
    region_truth = truth_for_regions(regions, y, top_k_truth)
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
    bundle, _, y, regions = load_inputs(args.results_npy, args.fitness_csv, args.metrics)
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
    )
    write_report(output, report)
    logging.info("Wrote region quality report: %s", output)
    print(report)


if __name__ == "__main__":
    main()
