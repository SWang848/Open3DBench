from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging


# Default colors for multiple lines (distinct, colorblind-friendly)
_DEFAULT_LINE_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#9D755D", "#EECA3B", "#BAB0AC", "#76B7B2",
]


def plot_cputime_vs_composite_cost_with_improvement(
    data_df: pd.DataFrame,
    eval_times_col: str = "eval_times",
    composite_cost_col: str = "composite_cost",
    save_path: str = "./eval_times_vs_composite_cost_with_improvement.png",
    plot_name: str = "Evaluation Times vs Composite Cost",
    group_col: Optional[str] = None,
    x_label: str = "Number of Evaluations",
    y_label: str = "Composite Cost",
) -> None:
    """
    Plot composite cost vs evaluation times.
    Supports multiple subplots when group_col is specified (one subplot per group).

    Args:
        data_df: DataFrame containing x-axis values and composite cost columns.
        eval_times_col: Column name for evaluation times (x-axis).
        composite_cost_col: Column name for composite cost score.
        save_path: Destination path for the generated plot.
        plot_name: Figure title.
        group_col: Optional column to group by; each unique value gets its own subplot.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
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

    if group_col is None:
        groups = [(None, data.sort_values(eval_times_col).reset_index(drop=True))]
    else:
        groups = [
            (name, grp.sort_values(eval_times_col).reset_index(drop=True))
            for name, grp in data.groupby(group_col, sort=False)
        ]

    n_groups = len(groups)
    n_rows, n_cols = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten()

    for i in range(n_groups, n_rows * n_cols):
        axes_flat[i].set_visible(False)

    for idx, (name, grp) in enumerate(groups):
        ax = axes_flat[idx]

        x_vals = grp[eval_times_col].to_numpy(dtype=float)
        y_vals = grp[composite_cost_col].to_numpy(dtype=float)
        color = _DEFAULT_LINE_COLORS[idx % len(_DEFAULT_LINE_COLORS)]
        legend_label = name if name is not None else "Composite Cost"

        ax.plot(
            x_vals,
            y_vals,
            color=color,
            marker="o",
            linewidth=4,
            markersize=12,
            label=legend_label,
        )
        x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
        y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = max(x_range * 0.002, 1e-6)
        y_pad = max(y_range * 0.002, abs(np.mean(y_vals)) * 0.002, 0.001)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.legend(loc="best", fontsize=14)
        ax.grid(True, alpha=0.3)
    fig.supxlabel(x_label, fontsize=16, fontweight="bold")
    fig.supylabel(y_label, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info("Evaluation times vs composite cost plot saved to '%s'", save_path)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.DataFrame({
        "eval_times": [10, 13, 26, 52, 78, 104, 129, 155, 258] + [10, 10, 16, 31, 46, 61, 76, 92, 152] + [10, 13, 25, 50, 75, 100, 124, 149, 248] \
            + [10, 10, 10, 17, 25, 33, 41, 48, 81] + [10, 10, 10, 13, 20, 26, 32, 39, 64] + [10, 13, 26, 52, 77, 103, 128, 154, 256]
            + [10, 25, 50, 99, 148, 197, 245, 294, 492] + [10, 14, 27, 53, 79, 106, 132, 159, 263],
    
        "composite_cost": [
            1.7745, 1.7745, 1.7745, 1.7745, 1.7745, 1.7745, 1.7745, 1.7745, 1.7745,
            1.9199, 1.9199, 1.8559, 1.8559, 1.8559, 1.8559, 1.8559, 1.8559, 1.8559,
            1.5124, 1.5074, 1.5074, 1.5061, 1.4839, 1.4839, 1.4839, 1.4839, 1.4839,
            1.6957, 1.6957, 1.6957, 1.6957, 1.6759, 1.6759, 1.6957, 1.6502, 1.6502,
            1.6056, 1.6056, 1.6056, 1.6103, 1.6103, 1.6056, 1.6056, 1.6056, 1.6056,
            1.8317, 1.8245, 1.8144, 1.8052, 1.8052, 1.8052, 1.8052, 1.8052, 1.7997,
            1.7438, 1.7438, 1.7293, 1.7438, 1.7438, 1.7438, 1.7293, 1.7293, 1.7293,
            1.4323, 1.3961, 1.3816, 1.3816, 1.3816, 1.3726, 1.3726, 1.3726, 1.3726, 
        ],
        "method": ["ariane133"] * 9 + ["ariane136"] * 9 + ["black_parrot"] * 9 \
            + ["bp_be"] * 9 + ["bp_fe"] * 9 + ["bp_multi"] * 9 \
            + ["bp_quad"] * 9 + ["swerv_wrapper"] * 9,
    })
    save_path = f"./eval_times_vs_composite_cost_with_improvement.png"
    plot_cputime_vs_composite_cost_with_improvement(
        df,
        eval_times_col="eval_times",
        composite_cost_col="composite_cost",
        group_col="method",
        save_path=save_path,
        plot_name="Evaluation Times vs Composite Cost",
    )
