from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx


def _parse_upper_die_macros(raw_value: Optional[str]) -> Optional[List[str]]:
    if raw_value is None:
        return None
    if not raw_value.strip():
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _parse_partition_result(raw_value: Optional[str]) -> Optional[List[int]]:
    if raw_value is None:
        return None
    if not raw_value.strip():
        return []
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def plot_graph_structure(
    graph_builder,
    save_path: Optional[Path] = None,
    title: str = "DOPP Graph",
    with_labels: bool = True,
):
    """
    Plot the graph topology only, with edge width proportional to edge weight.

    Node features are intentionally not visualized here.
    """
    graph = nx.Graph()
    cell_node_idx = graph_builder.get_graph_cell_node_idx()

    for node_idx in range(graph_builder.node_features.shape[0]):
        graph.add_node(node_idx, is_cell=(node_idx == cell_node_idx))

    for edge_idx, (src_idx, dst_idx) in enumerate(graph_builder.edges):
        graph.add_edge(
            int(src_idx),
            int(dst_idx),
            weight=float(graph_builder.edge_weights[edge_idx]),
        )

    positions = nx.spring_layout(graph, seed=42, weight="weight")
    edge_weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        if max_weight > min_weight:
            edge_widths = [
                0.8 + 5.2 * ((weight - min_weight) / (max_weight - min_weight))
                for weight in edge_weights
            ]
        else:
            edge_widths = [2.5 for _ in edge_weights]
    else:
        edge_widths = []

    node_colors = [
        "#d62728" if graph.nodes[node_idx]["is_cell"] else "#1f77b4"
        for node_idx in graph.nodes
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=700,
        edgecolors="black",
        linewidths=0.75,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        width=edge_widths,
        edge_color="#7f7f7f",
        alpha=0.8,
        ax=ax,
    )

    if with_labels:
        labels = {
            node_idx: ("cell" if node_idx == cell_node_idx else str(node_idx))
            for node_idx in graph.nodes
        }
        nx.draw_networkx_labels(graph, positions, labels=labels, font_size=9, ax=ax)

    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax
