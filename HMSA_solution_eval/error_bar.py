from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import logging
import matplotlib.pyplot as plt
import numpy as np


def plot_composite_cost_error_bars(
    eval_counts: Optional[Sequence[float]] = None,
    value_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    median_values: Optional[Sequence[float]] = None,
    save_path: str = "./composite_cost_error_bar.png",
    plot_name: Optional[str] = None,
    x_label: str = "Number of Evaluations",
    y_label: str = "Composite Cost",
    point_color: str = "#4C78A8",
    error_color: str = "#4C78A8",
    capsize: float = 7.0,
    linewidth: float = 3.0,
    marker_size: float = 10.0,
    label: Optional[str] = None,
    named_data: Optional[Mapping[str, Tuple[Sequence[float], Sequence[Tuple[float, float]], Optional[Sequence[float]]]]] = None,
) -> None:
    """
    Plot composite-cost error bars from evaluation counts and value ranges.

    Args:
        eval_counts: X-axis values representing the number of evaluations for a single series.
        value_ranges: Sequence of (min_value, max_value) pairs for each evaluation count.
        median_values: Optional marker y-values for each evaluation count. If omitted, use range midpoints.
        save_path: Destination path for the generated plot.
        plot_name: Optional figure title.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        point_color: Color for the center points.
        error_color: Color for the error bars.
        capsize: Error bar cap size.
        linewidth: Error bar line width.
        marker_size: Marker size for the center points.
        label: Optional legend label for the single-series case.
        named_data: Optional mapping from series name to `(eval_counts, value_ranges, median_values)`.
    """
    if named_data is None:
        if eval_counts is None or value_ranges is None:
            raise ValueError("Provide either named_data or both eval_counts and value_ranges.")
        named_data = {label or "Series": (eval_counts, value_ranges, median_values)}

    if not named_data:
        logging.warning("No data provided. Skipping error bar plot.")
        return

    plt.figure(figsize=(9, 6))
    colors = [point_color, "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    all_x_values = []
    all_lower_bounds = []
    all_upper_bounds = []

    for idx, (series_name, series_data) in enumerate(named_data.items()):
        if len(series_data) == 2:
            series_eval_counts, series_value_ranges = series_data
            series_median_values = None
        elif len(series_data) == 3:
            series_eval_counts, series_value_ranges, series_median_values = series_data
        else:
            raise ValueError(
                f"Series '{series_name}' must provide (eval_counts, value_ranges) "
                "or (eval_counts, value_ranges, median_values)."
            )
        if len(series_eval_counts) != len(series_value_ranges):
            raise ValueError(f"Series '{series_name}' has mismatched eval_counts and value_ranges lengths.")
        if not series_eval_counts:
            continue

        x_vals = np.asarray(series_eval_counts, dtype=float)
        ranges = np.asarray(series_value_ranges, dtype=float)
        if ranges.ndim != 2 or ranges.shape[1] != 2:
            raise ValueError(f"Series '{series_name}' must use (min_value, max_value) pairs.")

        lower = np.minimum(ranges[:, 0], ranges[:, 1])
        upper = np.maximum(ranges[:, 0], ranges[:, 1])
        if series_median_values is None:
            centers = (lower + upper) / 2.0
        else:
            if len(series_median_values) != len(series_eval_counts):
                raise ValueError(f"Series '{series_name}' has mismatched median_values length.")
            centers = np.asarray(series_median_values, dtype=float)
            out_of_range_mask = (centers < lower) | (centers > upper)
            if np.any(out_of_range_mask):
                logging.warning(
                    "Series '%s' has median values outside the provided ranges; clipping them to the interval bounds.",
                    series_name,
                )
                centers = np.clip(centers, lower, upper)
        yerr = np.vstack((centers - lower, upper - centers))

        order = np.argsort(x_vals)
        x_vals = x_vals[order]
        centers = centers[order]
        yerr = yerr[:, order]
        lower = lower[order]
        upper = upper[order]

        all_x_values.extend(x_vals.tolist())
        all_lower_bounds.extend(lower.tolist())
        all_upper_bounds.extend(upper.tolist())

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.errorbar(
            x_vals,
            centers,
            yerr=yerr,
            fmt=marker,
            linestyle="none",
            color=color,
            ecolor=color if error_color == point_color else error_color,
            elinewidth=linewidth,
            capsize=capsize,
            markersize=marker_size,
            markeredgewidth=2.0,
            label=series_name,
        )

    plt.xlabel(x_label, fontsize=20, fontweight="bold")
    plt.ylabel(y_label, fontsize=20, fontweight="bold")
    if plot_name:
        plt.title(plot_name, fontsize=22, fontweight="bold")
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=18)
    if all_x_values:
        x_min = min(all_x_values)
        x_max = max(all_x_values)
        x_range = x_max - x_min
        x_pad = max(x_range * 0.02, 0.5)
        left_pad = max(x_range * 0.08, 8.0)
        ax.set_xlim(x_min - left_pad, x_max + x_pad)
    if all_lower_bounds and all_upper_bounds:
        y_min = min(all_lower_bounds)
        y_max = max(all_upper_bounds)
        y_range = y_max - y_min
        y_pad = max(y_range * 0.06, 0.002)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    filtered_ticks = [tick for tick in ax.get_xticks() if tick >= 1]
    if 1 not in filtered_ticks:
        filtered_ticks = [1] + filtered_ticks
    ax.set_xticks(sorted(set(filtered_ticks)))
    if len(named_data) > 1 or any(name != "Series" for name in named_data):
        plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info("Composite cost error bar plot saved to '%s'", save_path)
    plt.close()


if __name__ == "__main__":
    example_eval_counts = [12, 25, 51, 76, 102, 128, 153, 256]
    example_value_ranges = [
        (1.8775933750744702, 2.0123742202127409),
        (1.804543962777589, 1.9227663114748442),
        (1.8197693831041524, 1.843015930842959),
        (1.7986181342792917, 1.82599626341009),
        (1.7877561987604744, 1.8223742202127409),
        (1.7877561987604744, 1.8229248590462517),
        (1.7876181342792917, 1.812322221028705),
        (1.7877561987604744, 1.7987561987604744),
    ]
    plot_composite_cost_error_bars(
        named_data={
            "Random": (example_eval_counts, example_value_ranges, [1.907341, 1.846928, 1.831204, 1.814557, 1.806983, 1.819412, 1.804931, 1.792]),
            "DOPP": ([44], [(1.792, 1.810)], [1.804]),
            "Open3DBench": ([1], [(1.974, 2.001)], [1.99]),
            "Exhaustive": ([256], [(1.787, 1.798)], [1.792]),
        },
        save_path=str(Path(__file__).with_name("composite_cost_error_bar.png")),
    )
