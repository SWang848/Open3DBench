import argparse
import json
import logging
from pathlib import Path

from algorithms.dopp.utils import _parse_partition_result, _parse_upper_die_macros


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load DREAMPlace data and build the DOPP macro graph."
    )
    parser.add_argument(
        "benchmark",
        help="Benchmark name matching Place-3D/test/or_3D/<benchmark>_3D.json",
    )
    parser.add_argument(
        "candidates_path",
        type=Path,
        help="Path to candidates.json for building a partition-updated graph sample.",
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
    from algorithms.dopp.graph_diffused_feature_constructor import (
        GraphDiffusedFeatureConstructor,
    )
    from algorithms.dopp.graph_builder import GraphBuilder
    from algorithms.dopp.partition_graph_updater import PartitionGraphUpdater

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
    # plot_path = args.candidates_path.parent / f"{args.benchmark}_graph.png"
    # fig, _ = plot_graph_structure(
    #     graph_builder,
    #     save_path=plot_path,
    #     title=f"DOPP Graph: {args.benchmark}",
    # )
    # plt.close(fig)
    # logging.info("Saved graph plot to %s", plot_path)

    # summary = {
    #     "benchmark": args.benchmark,
    #     "num_macros": len(macros),
    #     "num_graph_nodes": int(graph_builder.node_features.shape[0]),
    #     "num_graph_edges": int(len(graph_builder.edge_weights)),
    #     "node_feature_shape": list(graph_builder.node_features.shape),
    #     "edge_feature_shape": [int(len(graph_builder.edge_weights)), 1],
    #     "cell_area": float(area_info["cell_area"]),
    #     "total_area": float(area_info["total_area"]),
    # }

    updater = PartitionGraphUpdater(
        graph_builder=graph_builder,
        candidates_path=args.candidates_path,
    )
    partition_node_features = updater.build_all_partition_features()

    partition_feature_summary = None
    if partition_node_features.shape[0] > 0:
        diffused_feature_constructor = GraphDiffusedFeatureConstructor(
            partition_node_features=partition_node_features,
            edges=graph_builder.edges,
            edge_weights=graph_builder.edge_weights,
        )
        diffused_partition_features = diffused_feature_constructor.build_flattened_features()
        partition_feature_summary = {
            "partition_feature_shape": list(partition_node_features.shape[1:]),
            "all_partition_feature_shape": list(partition_node_features.shape),
            "diffused_partition_feature_shape": list(diffused_partition_features.shape),
            "partition_feature_names": [
                "partition_label",
                "incident_cut_net_count",
                "flip_area_imbalance_gain",
                "cross_tire_cell_conectivity",
                "hierarchy_cohesion",
            ],
        }

    summary = {
        "benchmark": args.benchmark,
        "num_macros": len(macros),
        "num_graph_nodes": int(graph_builder.node_features.shape[0]),
        "num_graph_edges": int(len(graph_builder.edge_weights)),
        "static_node_feature_shape": list(graph_builder.node_features.shape),
        "cell_area": float(area_info["cell_area"]),
        "total_area": float(area_info["total_area"]),
        "num_partition_samples": int(partition_node_features.shape[0]),
        "partition_features": partition_feature_summary,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
