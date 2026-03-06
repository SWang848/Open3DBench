from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import logging


# Default colors for multiple lines (distinct, colorblind-friendly)
_DEFAULT_LINE_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#9D755D", "#EECA3B", "#BAB0AC", "#76B7B2",
]


def plot_cputime_vs_composite_cost_with_improvement(
    data_df: pd.DataFrame,
    baseline_fitness: float,
    eval_times_col: str = "eval_times",
    composite_cost_col: str = "composite_cost",
    save_path: str = "./eval_times_vs_composite_cost_with_improvement.png",
    plot_name: str = "Evaluation Times vs Composite Cost (with Relative Improvement)",
    group_col: Optional[str] = None,
    x_label: str = "Evaluation Times",
) -> None:
    """
    Plot composite cost vs evaluation times with a right y-axis showing relative improvement (%).
    Supports multiple lines when group_col is specified.

    Args:
        data_df: DataFrame containing x-axis values and composite cost columns.
        baseline_fitness: Baseline composite cost value.
        eval_times_col: Column name for evaluation times (x-axis).
        composite_cost_col: Column name for composite cost score.
        save_path: Destination path for the generated plot.
        plot_name: Figure title.
        group_col: Optional column to group by; each unique value draws a separate line.
        x_label: Label for the x-axis.
    """
    if data_df.empty:
        logging.warning("data_df is empty. Skipping evaluation times vs composite cost plot.")
        return

    required_cols = {eval_times_col, composite_cost_col}
    if group_col is not None:
        required_cols.add(group_col)
    missing_cols = required_cols - set(data_df.columns)
    if missing_cols:
        logging.warning("Missing required columns %s. Skipping plot.", list(missing_cols))
        return

    data = data_df[list(required_cols)].copy()
    data[eval_times_col] = pd.to_numeric(data[eval_times_col], errors="coerce")
    data[composite_cost_col] = pd.to_numeric(data[composite_cost_col], errors="coerce")
    if group_col:
        data[group_col] = data[group_col].astype(str)
    data = data.dropna(subset=[eval_times_col, composite_cost_col])
    if data.empty:
        logging.warning("No valid evaluation times or composite cost values to plot.")
        return

    plt.figure(figsize=(10, 6))
    ax_left = plt.gca()
    ax_right = ax_left.twinx()

    if group_col is None:
        groups = [(None, data.sort_values(eval_times_col).reset_index(drop=True))]
    else:
        groups = [
            (name, grp.sort_values(eval_times_col).reset_index(drop=True))
            for name, grp in data.groupby(group_col, sort=False)
        ]

    all_costs = []
    for idx, (name, grp) in enumerate(groups):
        x_vals = grp[eval_times_col].to_numpy(dtype=float)
        y_vals = grp[composite_cost_col].to_numpy(dtype=float)
        all_costs.extend(y_vals.tolist())
        color = _DEFAULT_LINE_COLORS[idx % len(_DEFAULT_LINE_COLORS)]
        label = name if name is not None else "Composite Cost"
        ax_left.plot(
            x_vals,
            y_vals,
            color=color,
            marker="o",
            linewidth=2,
            markersize=6,
            label=label,
        )

    ax_left.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax_left.set_ylabel("Composite Cost", fontsize=12, fontweight="bold")
    ax_right.set_ylabel("Relative Improvement (%)", fontsize=12, fontweight="bold")
    ax_left.set_title(plot_name, fontsize=14, fontweight="bold")
    ax_left.grid(True, alpha=0.3)

    if all_costs:
        y_min = float(np.min(all_costs))
        y_max = float(np.max(all_costs))
        pad = max((y_max - y_min) * 0.02, 1e-6)
        ax_left.set_ylim(y_max + pad, y_min - pad)

    if len(groups) > 1:
        ax_left.legend(loc="best", fontsize=10)

    ax_right.set_ylim(ax_left.get_ylim())
    ax_right.yaxis.set_major_formatter(
        FuncFormatter(
            lambda val, _: f"{(baseline_fitness - val) / abs(baseline_fitness) * 100.0:.1f}"
        )
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info("Evaluation times vs composite cost plot with relative improvement saved to '%s'", save_path)
    plt.close()



def plot_dopt_threshold_tradeoff(
    threshold_df: pd.DataFrame,
    plot_name: str = "Top-k Percentage of Samples (By Weight) vs. Coreset Size and Best-Solution Rank",
    threshold_col: str = "top k percentage",
    coreset_col: str = "coreset_size",
    rank_col: str = "best_rank",
    save_path: str = "./dopt_threshold_tradeoff.png",
    order: str = "desc",
) -> None:
    """
    Plot D-opt threshold tradeoff: coreset size (bars) vs best-solution rank (line).

    Args:
        threshold_df: DataFrame with threshold, coreset size, and best rank columns.
        threshold_col: Column name for D-opt thresholds.
        coreset_col: Column name for coreset size.
        rank_col: Column name for best-solution rank (smaller is better).
        save_path: Destination path for the generated plot.
        order: "asc" or "desc" ordering of thresholds along the x-axis.
    """


    required_cols = {threshold_col, coreset_col, rank_col}
    data = threshold_df[list(required_cols)].copy()
    data[threshold_col] = pd.to_numeric(data[threshold_col], errors="coerce")
    data[coreset_col] = pd.to_numeric(data[coreset_col], errors="coerce")
    data[rank_col] = pd.to_numeric(data[rank_col], errors="coerce")
    data = data.dropna(subset=[threshold_col, coreset_col, rank_col])

    # ascending = True if order == "asc" else False
    # data = data.sort_values(threshold_col, ascending=ascending).reset_index(drop=True)

    thresholds = data[threshold_col].to_numpy(dtype=float)
    coreset_sizes = data[coreset_col].to_numpy(dtype=int)
    ranks = data[rank_col].to_numpy(dtype=int)

    labels = [f"{val:.0e}" if val < 1e-2 else f"{val:g}" for val in thresholds]

    plt.figure(figsize=(10, 6))
    ax_left = plt.gca()
    ax_right = ax_left.twinx()

    x_pos = np.arange(len(thresholds))
    ax_left.bar(
        x_pos,
        coreset_sizes,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )
    ax_left.bar_label(ax_left.containers[0], padding=3, fontsize=10, fontweight="bold")

    ax_right.plot(
        x_pos,
        ranks,
        color="black",
        marker="o",
        markersize=6,
        linewidth=2,
    )

    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax_left.set_xlabel("Top-k Percentage Of Samples (By Weight)", fontsize=12, fontweight="bold")
    ax_left.set_ylabel("Coreset Size(# Samples)", fontsize=12, fontweight="bold")
    ax_right.set_ylabel("Best-Solution Rank", fontsize=12, fontweight="bold")
    ax_right.invert_yaxis()
    ax_right.set_ylim(10.5, 0.5)
    tick_values = list(range(1, 11)) + [10.5]
    ax_right.yaxis.set_major_locator(FixedLocator(tick_values))
    ax_right.yaxis.set_major_formatter(
        FuncFormatter(lambda val, _: "10+" if np.isclose(val, 10.5) else f"{int(val)}")
    )

    # ax_left.grid(True, axis="y", alpha=0.3)
    ax_left.set_title(
        plot_name,
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info("D-opt threshold tradeoff plot saved to '%s'", save_path)
    plt.close()

if __name__ == "__main__":
    design_name = "bp"
    threshold_col = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    coreset_col = [7, 15, 22, 30, 37, 45]
    rank_col = [4, 2, 1, 1, 1, 1]

    design_name = "bp_be"
    coreset_col = [8, 16, 24, 32, 40, 48]
    rank_col = [3, 2, 3, 3, 3, 3]
    
    design_name = "bp_fe"
    coreset_col = [6, 12, 19, 25, 32, 38]
    rank_col = [1, 3, 3, 3, 3, 3]
    
    design_name = "bp_multi"
    coreset_col = [9, 19, 29, 39, 49, 58]
    rank_col = [3, 1, 1, 1, 1, 1]
    
    # design_name = "bp_quad"
    # coreset_col = [5, 11, 17, 22, 28, 34]
    # rank_col = [1, 1, 2, 2, 1, 1]
    
    # design_name = "swerv_wrapper"
    # coreset_col = [20, 41, 62, 83, 104, 125]
    # rank_col = [1, 1, 1, 1, 1, 1]
    
    # design_name = "ariane133"
    # coreset_col = [28, 57, 85, 114, 142, 171]
    # rank_col = [2, 1, 1, 1, 1, 1]
    
    # design_name = "ariane136"
    # coreset_col = [14, 29, 44, 59, 74, 89]
    # rank_col = [7, 6, 1, 1, 1, 1]
    
    # threshold_df = pd.DataFrame({
    #     "threshold": threshold_col,
    #     "coreset_size": coreset_col,
    #     "rank": rank_col
    # })

    # save_path = f"./{design_name}/top_k_percentage_vs_coreset_size_and_best_solution_rank.png"
    # # You can either pass the DataFrame or pass the columns as lists
    # plot_dopt_threshold_tradeoff(
    #     threshold_df,
    #     plot_name=f"{design_name}: Top-k Percentage of Samples (By Weight) vs. Coreset Size and Best-Solution Rank",
    #     threshold_col="threshold",
    #     coreset_col="coreset_size",
    #     rank_col="rank",
    #     save_path=save_path,
    # )
    
    design_name = "ariane136"
    df = pd.DataFrame({
        "eval_times": [10871, 25712, 51231, 74314, 99872],
        "composite_cost": [1.8425180337078761, 1.8504492055806137, 1.8492893518385345, 1.800856369670965, 1.800856369670965],
    })
    baseline = 1.9897829688926063

    # Single-line example:
    save_path = f"./{design_name}/eval_times_vs_composite_cost_with_improvement.png"
    plot_cputime_vs_composite_cost_with_improvement(
        df,
        baseline_fitness=baseline,
        eval_times_col="eval_times",
        composite_cost_col="composite_cost",
        save_path=save_path,
        plot_name=f"{design_name}: Evaluation Times vs Composite Cost (with Relative Improvement)",
        x_label="Evaluation Times",
    )

    # Multiple-line example (uncomment to try):
    # df_multi = pd.DataFrame({
    #     "eval_times": [10871, 25712, 51231, 74314, 99872] * 2,
    #     "composite_cost": [1.84, 1.85, 1.85, 1.80, 1.80, 1.86, 1.84, 1.83, 1.82, 1.81],
    #     "method": ["A"] * 5 + ["B"] * 5,
    # })
    # plot_cputime_vs_composite_cost_with_improvement(
    #     df_multi,
    #     baseline_fitness=baseline,
    #     eval_times_col="eval_times",
    #     composite_cost_col="composite_cost",
    #     group_col="method",
    #     save_path=f"./{design_name}/eval_times_vs_composite_cost_multi.png",
    #     plot_name=f"{design_name}: Evaluation Times vs Composite Cost (Multiple Methods)",
    # )
    
    