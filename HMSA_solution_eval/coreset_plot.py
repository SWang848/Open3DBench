from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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


if __name__ == "__main__":
    design_name = "ariane136"
    baseline = 1.9897829688926063

    df = pd.DataFrame({
        "eval_times": [12, 34, 45, 67, 89] * 2,
        "composite_cost": [
            1.8425, 1.8504, 1.8493, 1.8009, 1.8009,
            1.8600, 1.8550, 1.8480, 1.8100, 1.8050,
        ],
        "method": ["A"] * 5 + ["B"] * 5,
    })
    save_path = f"./eval_times_vs_composite_cost_with_improvement.png"
    plot_cputime_vs_composite_cost_with_improvement(
        df,
        baseline_fitness=baseline,
        eval_times_col="eval_times",
        composite_cost_col="composite_cost",
        group_col="method",
        save_path=save_path,
        plot_name="Evaluation Times vs Composite Cost (with Relative Improvement)",
        x_label="Evaluation Times",
    )
