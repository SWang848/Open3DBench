"""Solution-level surrogate report for the two-level DOPP baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from algorithms.dopp.baseline_analysis_utils import (
    default_report_path,
    format_float,
    format_pct,
    load_inputs,
    per_region_map,
    rank_array,
    round_region_source,
    write_report,
)


def build_solution_surrogate_table(
    bundle: Dict,
    regions: List[List[int]],
    y: np.ndarray,
    local_top_k: Sequence[int],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ranks = rank_array(y)
    exact_attribution = True

    for round_key in ("round1", "round2"):
        evaluated_map = per_region_map(bundle, round_key, "evaluated_per_region")
        surrogate_map = per_region_map(bundle, round_key, "surrogate_evaluated_per_region")
        dopt_map = per_region_map(bundle, round_key, "dopt_evaluated_per_region")
        source_map = round_region_source(bundle, round_key)

        if evaluated_map and not dopt_map:
            exact_attribution = False

        for r, evaluated_global_list in evaluated_map.items():
            members = np.asarray(regions[r], dtype=np.int64)
            evaluated_global = np.asarray(evaluated_global_list, dtype=np.int64)
            surrogate_global = np.asarray(surrogate_map.get(r, []), dtype=np.int64)
            if r in dopt_map:
                dopt_global = np.asarray(dopt_map[r], dtype=np.int64)
            else:
                surrogate_set = set(int(i) for i in surrogate_global.tolist())
                dopt_global = np.asarray(
                    [idx for idx in evaluated_global if int(idx) not in surrogate_set],
                    dtype=np.int64,
                )
                if dopt_global.size == 0:
                    dopt_global = evaluated_global

            true_member_y = y[members]
            true_best_pos = int(np.argmin(true_member_y))
            true_best_idx = int(members[true_best_pos])
            true_best_fitness = float(true_member_y[true_best_pos])

            dopt_best_idx = int(dopt_global[np.argmin(y[dopt_global])])
            final_best_idx = int(evaluated_global[np.argmin(y[evaluated_global])])
            dopt_best = float(y[dopt_best_idx])
            final_best = float(y[final_best_idx])
            evaluated_set = set(int(i) for i in evaluated_global.tolist())
            dopt_set = set(int(i) for i in dopt_global.tolist())
            surrogate_set = set(int(i) for i in surrogate_global.tolist())

            row: Dict[str, object] = {
                "round": round_key,
                "region_id": int(r),
                "region_size": int(members.size),
                "dopt_count": int(len(dopt_set)),
                "surrogate_count": int(len(surrogate_set)),
                "evaluated_count": int(len(evaluated_set)),
                "dopt_surrogate_overlap": int(len(dopt_set & surrogate_set)),
                "true_best_index": true_best_idx,
                "true_best_rank": int(ranks[true_best_idx]),
                "true_best_fitness": true_best_fitness,
                "dopt_best_index": dopt_best_idx,
                "dopt_best_fitness": dopt_best,
                "final_best_index": final_best_idx,
                "final_best_fitness": final_best,
                "final_best_source": source_map.get(r, "unknown"),
                "gap_before_surrogate": float(dopt_best - true_best_fitness),
                "gap_after_surrogate": float(final_best - true_best_fitness),
                "gap_reduction": float(dopt_best - final_best),
                "improved_by_surrogate": bool(final_best + 1e-12 < dopt_best),
                "hit_true_region_best": bool(final_best_idx == true_best_idx),
            }

            member_order = members[np.argsort(y[members], kind="stable")]
            for k in local_top_k:
                k_eff = min(int(k), member_order.size)
                row[f"local_top_{k_eff}_recall"] = float(
                    len(set(member_order[:k_eff].tolist()) & evaluated_set) / max(k_eff, 1)
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df, {
            "n_regions": 0,
            "exact_attribution": exact_attribution,
            "improved_fraction": np.nan,
            "mean_gap_before": np.nan,
            "mean_gap_after": np.nan,
            "mean_gap_reduction": np.nan,
            "median_gap_reduction": np.nan,
            "hit_true_best_fraction": np.nan,
            "source_counts": {},
            "mean_dopt_count": np.nan,
            "mean_surrogate_count": np.nan,
            "mean_overlap": np.nan,
        }

    summary = {
        "n_regions": int(len(df)),
        "exact_attribution": bool(exact_attribution),
        "improved_fraction": float(df["improved_by_surrogate"].mean()),
        "mean_gap_before": float(df["gap_before_surrogate"].mean()),
        "mean_gap_after": float(df["gap_after_surrogate"].mean()),
        "mean_gap_reduction": float(df["gap_reduction"].mean()),
        "median_gap_reduction": float(df["gap_reduction"].median()),
        "hit_true_best_fraction": float(df["hit_true_region_best"].mean()),
        "source_counts": {
            str(k): int(v)
            for k, v in df["final_best_source"].value_counts(dropna=False).to_dict().items()
        },
        "mean_dopt_count": float(df["dopt_count"].mean()),
        "mean_surrogate_count": float(df["surrogate_count"].mean()),
        "mean_overlap": float(df["dopt_surrogate_overlap"].mean()),
    }
    for col in [c for c in df.columns if c.startswith("local_top_")]:
        summary[f"mean_{col}"] = float(df[col].mean())
    return df, summary


def build_report(
    bundle: Dict,
    regions: List[List[int]],
    y: np.ndarray,
    local_top_k: Sequence[int],
    results_npy: Path,
    fitness_csv: Path,
) -> str:
    _, summary = build_solution_surrogate_table(bundle, regions, y, local_top_k)
    lines = [
        "# Solution-Level Surrogate Analysis",
        "",
        f"- Results: `{results_npy}`",
        f"- Fitness CSV: `{fitness_csv}`",
        "- The report compares the best D-opt-selected candidate with the best candidate after adding local surrogate top-K predictions.",
        "",
        "## Inner Selection Behavior",
        "",
        f"- Evaluated regions: **{summary['n_regions']}**.",
        f"- Mean D-opt candidate count per region: {format_float(summary['mean_dopt_count'], 2)}.",
        f"- Mean surrogate-proposed candidate count per region: {format_float(summary['mean_surrogate_count'], 2)}.",
        f"- Mean D-opt/surrogate overlap per region: {format_float(summary['mean_overlap'], 2)}.",
    ]
    if not summary["exact_attribution"]:
        lines.append(
            "- Attribution note: this result bundle does not store exact D-opt-selected candidates; "
            "D-opt counts are reconstructed conservatively from evaluated candidates not marked as surrogate-proposed."
        )

    lines.extend(
        [
            "",
            "## Surrogate Contribution",
            "",
            f"- Fraction of evaluated regions improved by the surrogate step: {format_pct(summary['improved_fraction'])}.",
            (
                "- Mean oracle gap before/after surrogate: "
                f"{format_float(summary['mean_gap_before'])} -> {format_float(summary['mean_gap_after'])}."
            ),
            (
                "- Gap reduction: "
                f"mean={format_float(summary['mean_gap_reduction'])}, "
                f"median={format_float(summary['median_gap_reduction'])}."
            ),
            f"- Hit rate of true region-best candidate: {format_pct(summary['hit_true_best_fraction'])}.",
            f"- Final best source counts: {summary['source_counts']}.",
        ]
    )
    for key, value in summary.items():
        if key.startswith("mean_local_top_"):
            lines.append(f"- {key.replace('mean_', '')}: mean recall={format_pct(value)}.")

    lines.extend(
        [
            "",
            "## Main Questions",
            "",
            "- Does the local surrogate add useful candidates beyond D-opt?",
            "- Is the inner prediction step redundant or complementary?",
            "- Are solution-level features predictive enough within each region?",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the solution-level surrogate report.")
    parser.add_argument("results_npy", type=Path, help="Path to two_level_results.npy.")
    parser.add_argument("fitness_csv", type=Path, help="Matching metrics.csv.")
    parser.add_argument("--metrics", type=str, nargs="+", default=None)
    parser.add_argument("--local-top-k", type=int, nargs="+", default=(1, 5, 10))
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
    output = args.output or default_report_path(args.results_npy, "solution_surrogate_analysis.md")
    report = build_report(bundle, regions, y, args.local_top_k, args.results_npy, args.fitness_csv)
    write_report(output, report)
    logging.info("Wrote solution-level surrogate report: %s", output)
    print(report)


if __name__ == "__main__":
    main()
