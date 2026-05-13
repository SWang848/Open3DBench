import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional


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


def _format_graph_node(graph_builder, graph_node_idx: int) -> dict:
    if graph_node_idx == graph_builder.get_graph_cell_node_idx():
        return {
            "graph_node_idx": int(graph_node_idx),
            "node_type": "cell",
        }
    return {
        "graph_node_idx": int(graph_node_idx),
        "node_type": "macro",
        "placedb_macro_id": int(graph_builder.graph_macro_idx_to_placedb_macro_id(graph_node_idx)),
    }


def _get_top_heavy_edges(graph_builder, limit: int = 5) -> List[dict]:
    if len(graph_builder.edge_weights) == 0:
        return []

    ranked_edge_indices = sorted(
        range(len(graph_builder.edge_weights)),
        key=lambda idx: float(graph_builder.edge_weights[idx]),
        reverse=True,
    )[:limit]

    top_edges = []
    for edge_rank, edge_idx in enumerate(ranked_edge_indices, start=1):
        src_idx, dst_idx = graph_builder.edges[edge_idx]
        top_edges.append(
            {
                "rank": edge_rank,
                "weight": float(graph_builder.edge_weights[edge_idx]),
                "endpoints": [
                    _format_graph_node(graph_builder, int(src_idx)),
                    _format_graph_node(graph_builder, int(dst_idx)),
                ],
            }
        )
    return top_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load DREAMPlace data and build the DOPP macro graph."
    )
    parser.add_argument(
        "benchmark",
        help="Benchmark name matching Place-3D/test/or_3D/<benchmark>_3D.json",
    )
    parser.add_argument(
        "--def-path",
        type=Path,
        default=None,
        help="Optional DEF file to override the benchmark default.",
    )
    parser.add_argument(
        "--upper-die-macros",
        type=str,
        default=None,
        help="Comma-separated macro names to place in the upper die.",
    )
    parser.add_argument(
        "--partition-result",
        type=str,
        default=None,
        help="Comma-separated placedb node IDs to place in the bottom die.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Cell area scaling factor for the merged cell node.",
    )
    parser.add_argument(
        "--top-k-ratio",
        type=float,
        default=0.3,
        help="If the graph is dense, keep only this fraction of the strongest edges.",
    )
    parser.add_argument(
        "--rand-init",
        action="store_true",
        default=False,
        help="Enable DREAMPlace random center initialization.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    from algorithms.dopp.dmp_loader import DreamPlaceLoader
    from algorithms.dopp.graph_builder import GraphBuilder

    upper_die_names = _parse_upper_die_macros(args.upper_die_macros)
    partition_result = _parse_partition_result(args.partition_result)
    partition_params = {
        "scale_factor": args.scale_factor,
        "top_k_ratio": args.top_k_ratio,
    }

    loader = DreamPlaceLoader(
        benchmark=args.benchmark,
        upper_die_names=upper_die_names,
        partition_result=partition_result,
        def_path=str(args.def_path) if args.def_path is not None else None,
        rand_init=args.rand_init,
    )
    macros = loader.determine_macro()

    graph_builder = GraphBuilder(partition_params=partition_params, dreamplace_loader=loader)
    area_info = graph_builder.get_area_info()

    summary = {
        "benchmark": args.benchmark,
        "num_macros": len(macros),
        "num_graph_nodes": int(graph_builder.node_features.shape[0]),
        "num_graph_edges": int(len(graph_builder.edge_weights)),
        "node_feature_shape": list(graph_builder.node_features.shape),
        "edge_feature_shape": [int(len(graph_builder.edge_weights)), 1],
        "top_5_heavy_edges": _get_top_heavy_edges(graph_builder, limit=5),
        "cell_area": float(area_info["cell_area"]),
        "total_area": float(area_info["total_area"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
