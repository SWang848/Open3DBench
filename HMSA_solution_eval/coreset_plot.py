import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import logging

def plot_cputime_vs_composite_cost(
    data_df: pd.DataFrame,
    baseline_fitness: float,
    cpu_col: str = "cpu_time",
    composite_cost_col: str = "composite_cost",
    save_path: str = "./cputime_vs_composite_cost.png",
    plot_name: str = "CPU Time vs Composite Cost",
) -> None:
    """
    Plot composite cost vs CPU time with a horizontal baseline.

    Args:
        data_df: DataFrame containing cpu time and composite cost columns.
        baseline_fitness: Baseline composite cost value.
        cpu_col: Column name for total CPU time (seconds).
        composite_cost_col: Column name for composite cost score.
        save_path: Destination path for the generated plot.
        plot_name: Figure title.
    """

    if data_df.empty:
        logging.warning("data_df is empty. Skipping CPU time vs composite cost plot.")
        return

    required_cols = {cpu_col, composite_cost_col}
    missing_cols = required_cols - set(data_df.columns)
    if missing_cols:
        logging.warning("Missing required columns %s. Skipping plot.", list(missing_cols))
        return

    data = data_df[list(required_cols)].copy()
    data[cpu_col] = pd.to_numeric(data[cpu_col], errors="coerce")
    data[composite_cost_col] = pd.to_numeric(data[composite_cost_col], errors="coerce")
    data = data.dropna(subset=[cpu_col, composite_cost_col])
    if data.empty:
        logging.warning("No valid CPU time or composite cost values to plot.")
        return

    data = data.sort_values(cpu_col).reset_index(drop=True)
    cpu_time = data[cpu_col].to_numpy(dtype=float)
    composite_cost = data[composite_cost_col].to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    ax.plot(
        cpu_time,
        composite_cost,
        color="#4C78A8",
        marker="o",
        linewidth=2,
        markersize=6,
        label="Composite Cost",
    )
    ax.axhline(
        baseline_fitness,
        color="#E45756",
        linestyle="--",
        linewidth=2,
        label="Baseline Composite Cost",
    )

    ax.set_xlabel("Total CPU Time (s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Composite Cost", fontsize=12, fontweight="bold")
    ax.set_title(plot_name, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info("CPU time vs composite cost plot saved to '%s'", save_path)
    plt.close()


def plot_cputime_vs_composite_cost_with_improvement(
    data_df: pd.DataFrame,
    baseline_fitness: float,
    cpu_col: str = "cpu_time",
    composite_cost_col: str = "composite_cost",
    save_path: str = "./cputime_vs_composite_cost_with_improvement.png",
    plot_name: str = "CPU Time vs Composite Cost (with Relative Improvement)",
) -> None:
    """
    Plot composite cost vs CPU time with a right y-axis showing relative improvement (%).

    Args:
        data_df: DataFrame containing cpu time and composite cost columns.
        baseline_fitness: Baseline composite cost value.
        cpu_col: Column name for total CPU time (seconds).
        composite_cost_col: Column name for composite cost score.
        save_path: Destination path for the generated plot.
        plot_name: Figure title.
    """

    if data_df.empty:
        logging.warning("data_df is empty. Skipping CPU time vs composite cost plot.")
        return

    required_cols = {cpu_col, composite_cost_col}
    missing_cols = required_cols - set(data_df.columns)
    if missing_cols:
        logging.warning("Missing required columns %s. Skipping plot.", list(missing_cols))
        return

    data = data_df[list(required_cols)].copy()
    data[cpu_col] = pd.to_numeric(data[cpu_col], errors="coerce")
    data[composite_cost_col] = pd.to_numeric(data[composite_cost_col], errors="coerce")
    data = data.dropna(subset=[cpu_col, composite_cost_col])
    if data.empty:
        logging.warning("No valid CPU time or composite cost values to plot.")
        return

    data = data.sort_values(cpu_col).reset_index(drop=True)
    cpu_time = data[cpu_col].to_numpy(dtype=float)
    composite_cost = data[composite_cost_col].to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))
    ax_left = plt.gca()
    ax_right = ax_left.twinx()

    ax_left.plot(
        cpu_time,
        composite_cost,
        color="#4C78A8",
        marker="o",
        linewidth=2,
        markersize=6,
        label="Composite Cost",
    )

    ax_left.set_xlabel("Total CPU Time (s)", fontsize=12, fontweight="bold")
    ax_left.set_ylabel("Composite Cost", fontsize=12, fontweight="bold")
    ax_right.set_ylabel("Relative Improvement (%)", fontsize=12, fontweight="bold")
    ax_left.set_title(plot_name, fontsize=14, fontweight="bold")
    ax_left.grid(True, alpha=0.3)
    y_min = float(np.min(composite_cost))
    y_max = float(np.max(composite_cost))
    pad = max((y_max - y_min) * 0.02, 1e-6)
    ax_left.set_ylim(y_max + pad, y_min - pad)


    ax_right.set_ylim(ax_left.get_ylim())
    ax_right.yaxis.set_major_formatter(
        FuncFormatter(
            lambda val, _: f"{(baseline_fitness - val) / abs(baseline_fitness) * 100.0:.1f}"
        )
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info("CPU time vs composite cost plot with relative improvement saved to '%s'", save_path)
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
    "cpu_time": [10871, 25712, 51231, 74314, 99872],
    "composite_cost": [1.8425180337078761,1.8504492055806137,1.8492893518385345,1.800856369670965,1.800856369670965],
    })
    baseline = 1.9897829688926063
    
    save_path = f"./{design_name}/cputime_vs_composite_cost.png"
    # plot_cputime_vs_composite_cost(
    #     df,
    #     baseline_fitness=baseline,
    #     cpu_col="cpu_time",
    #     composite_cost_col="composite_cost",
    #     save_path=save_path,
    #     plot_name=f"{design_name}: CPU Time vs Composite Cost",
    # )

    save_path = f"./{design_name}/cputime_vs_composite_cost_with_improvement.png"
    plot_cputime_vs_composite_cost_with_improvement(
        df,
        baseline_fitness=baseline,
        cpu_col="cpu_time",
        composite_cost_col="composite_cost",
        save_path=save_path,
        plot_name=f"{design_name}: CPU Time vs Composite Cost (with Relative Improvement)",
    )
    
    