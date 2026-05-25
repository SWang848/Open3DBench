"""Per-region empirical analysis for the two-level DOPP baseline.

Given a saved ``two_level_results.npy`` (produced by ``two_level_dopp.py``)
and the matching ``metrics.csv``, this script computes a small, focused set
of per-region statistics that characterize each cluster region as a "bag of
feasible candidates":

Per region (one CSV row):
- ``size``                : number of member solutions
- ``best_rank``           : global rank (1-indexed, 1 = global best) of the
                            best member
- ``p10_rank, p25_rank, p75_rank, p90_rank``
- ``worst_rank``          : global rank of the worst member
- ``best_fitness``        : raw fitness of the best member (magnitude reference)
- ``top10pct_hits``       : count of region members in the global top 10%
                            (concentration of "gold" inside this region)
- ``rank_by_region_best`` : 1-indexed rank of this region when regions are
                            sorted by their best-member fitness (the
                            "ground-truth" region ranking the surrogate
                            is trying to learn)
- ``selected_round``      : 1, 2, or 0 (= unevaluated) -- which round picked
                            this region (if the results bundle has it)
- ``dopt_oracle_gap``     : ``dopt_best_fitness - best_fitness`` for evaluated
                            regions (how much inner D-opt under-shoots the
                            region's true best), NaN otherwise

The script writes one self-contained markdown file (``region_stats.md`` by
default) next to ``two_level_results.npy``. The file contains:
- size distribution
- where the global top-K end up (which regions contain them)
- selection quality (Round 1 vs Round 1 + Round 2 capture)
- best-vs-truth oracle gap for evaluated regions
- the full per-region table embedded inline
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from algorithms.dopp.loaders import load_fitness_scores_from_csv
from algorithms.dopp.two_level_dopp import align_fitness


# ----------------------------------------------------------------------------
# Core analysis
# ----------------------------------------------------------------------------


def _rank_array(y: np.ndarray) -> np.ndarray:
    """Return 1-indexed global ranks where rank 1 = lowest (best) fitness.

    Ties broken by ``argsort`` order (stable, deterministic).
    """
    order = np.argsort(y, kind="stable")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(y) + 1, dtype=np.int64)
    return ranks


def _percentile_int(values: np.ndarray, q: float) -> int:
    """Percentile of an integer array, returned as ``int`` (rounded)."""
    if values.size == 0:
        return -1
    return int(round(float(np.percentile(values, q))))


def _percentile_float(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def compute_region_stats(
    region_indices: List[List[int]],
    y: np.ndarray,
    top_pct_cutoff: float = 10.0,
    selected_round: Optional[Dict[int, int]] = None,
    region_dopt_best: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """Compute a concise per-region table. One row per region."""
    n_samples = y.size
    global_ranks = _rank_array(y)
    top_cutoff_rank = max(1, int(np.floor(n_samples * top_pct_cutoff / 100.0)))
    top_col = f"top{top_pct_cutoff:g}pct_hits"

    region_best: List[Tuple[int, float]] = []
    for r, members in enumerate(region_indices):
        if members:
            member_y = y[np.asarray(members, dtype=np.int64)]
            region_best.append((r, float(np.min(member_y))))
        else:
            region_best.append((r, float("inf")))
    region_best.sort(key=lambda kv: kv[1])
    rank_by_region_best: Dict[int, int] = {
        r: idx + 1 for idx, (r, _) in enumerate(region_best)
    }

    rows: List[Dict[str, object]] = []
    for r, members_list in enumerate(region_indices):
        members = np.asarray(members_list, dtype=np.int64)
        size = int(members.size)
        if size == 0:
            rows.append({"region_id": r, "size": 0})
            continue

        member_ranks = global_ranks[members]
        member_y = y[members]
        best_rank = int(member_ranks.min())
        worst_rank = int(member_ranks.max())

        row: Dict[str, object] = {
            "region_id": r,
            "size": size,
            "best_rank": best_rank,
            "p10_rank": _percentile_int(member_ranks, 10),
            "p25_rank": _percentile_int(member_ranks, 25),
            "p75_rank": _percentile_int(member_ranks, 75),
            "p90_rank": _percentile_int(member_ranks, 90),
            "worst_rank": worst_rank,
            "best_fitness": float(member_y.min()),
            top_col: int(np.sum(member_ranks <= top_cutoff_rank)),
            "rank_by_region_best": rank_by_region_best[r],
            "selected_round": (
                int(selected_round[r])
                if selected_round and r in selected_round
                else 0
            ),
        }

        if region_dopt_best and r in region_dopt_best:
            row["dopt_oracle_gap"] = float(region_dopt_best[r]) - row["best_fitness"]
        else:
            row["dopt_oracle_gap"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Global summary
# ----------------------------------------------------------------------------


def _region_capture_of_top_k(
    region_indices: List[List[int]],
    y: np.ndarray,
    k: int,
) -> Dict[str, object]:
    """For the global top-``k`` solutions, summarize which regions hold them."""
    order = np.argsort(y, kind="stable")
    top_idx = order[: min(k, y.size)]
    label_of_idx = -np.ones(y.size, dtype=np.int64)
    for r, members in enumerate(region_indices):
        if members:
            label_of_idx[np.asarray(members, dtype=np.int64)] = r
    regions_for_top = label_of_idx[top_idx]
    unique_regions = np.unique(regions_for_top[regions_for_top >= 0])
    counts = {
        int(r): int(np.sum(regions_for_top == r)) for r in unique_regions.tolist()
    }
    return {
        "k": int(min(k, y.size)),
        "regions_with_hits": int(unique_regions.size),
        "per_region_hits": counts,
    }


def compute_global_summary(
    stats_df: pd.DataFrame,
    y: np.ndarray,
    region_indices: List[List[int]],
    round1_regions: Optional[Sequence[int]] = None,
    round2_regions: Optional[Sequence[int]] = None,
    top_k_truth: Sequence[int] = (10, 20, 50, 100),
) -> Dict[str, object]:
    summary: Dict[str, object] = {}

    sizes = stats_df["size"].to_numpy()
    summary["n_regions"] = int(len(stats_df))
    summary["n_samples"] = int(y.size)
    summary["region_size"] = {
        "min": int(sizes.min()),
        "max": int(sizes.max()),
        "mean": float(sizes.mean()),
        "median": float(np.median(sizes)),
        "std": float(sizes.std()),
    }

    best_ranks = stats_df.loc[stats_df["size"] > 0, "best_rank"].to_numpy()
    summary["best_rank_per_region"] = {
        "min": int(best_ranks.min()),
        "median": float(np.median(best_ranks)),
        "max": int(best_ranks.max()),
        "p25": float(np.percentile(best_ranks, 25)),
        "p75": float(np.percentile(best_ranks, 75)),
    }

    summary["top_k_capture"] = {
        f"top_{k}": _region_capture_of_top_k(region_indices, y, k)
        for k in top_k_truth
    }

    selected_r1 = set(int(r) for r in (round1_regions or []))
    selected_r2 = set(int(r) for r in (round2_regions or []))
    selected_all = selected_r1 | selected_r2

    if selected_all:
        order = np.argsort(y, kind="stable")
        label_of_idx = -np.ones(y.size, dtype=np.int64)
        for r, members in enumerate(region_indices):
            if members:
                label_of_idx[np.asarray(members, dtype=np.int64)] = r

        cap = {}
        for k in top_k_truth:
            top_idx = order[: min(k, y.size)]
            regions_for_top = label_of_idx[top_idx]
            cap[f"top_{k}"] = {
                "k": int(min(k, y.size)),
                "captured_by_round1": int(
                    sum(int(r) in selected_r1 for r in regions_for_top.tolist())
                ),
                "captured_by_round1_or_2": int(
                    sum(int(r) in selected_all for r in regions_for_top.tolist())
                ),
            }
        summary["selection_capture"] = cap

    evaluated = stats_df.dropna(subset=["dopt_oracle_gap"])
    if len(evaluated) > 0:
        summary["oracle_gap"] = {
            "n_evaluated_regions": int(len(evaluated)),
            "mean": float(evaluated["dopt_oracle_gap"].mean()),
            "median": float(evaluated["dopt_oracle_gap"].median()),
            "max": float(evaluated["dopt_oracle_gap"].max()),
            "frac_regions_hit_true_min": float(
                (evaluated["dopt_oracle_gap"] <= 0.0 + 1e-12).mean()
            ),
        }

    return summary


# ----------------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------------


def _load_results_bundle(results_path: Path) -> Dict:
    return np.load(results_path, allow_pickle=True).item()


def _extract_region_indices(bundle: Dict) -> List[List[int]]:
    clustering = bundle.get("clustering", {})
    region_indices = clustering.get("region_indices")
    if region_indices is None:
        raise ValueError("Results bundle is missing clustering.region_indices")
    return [list(map(int, members)) for members in region_indices]


def _extract_selected_round(bundle: Dict) -> Tuple[Dict[int, int], List[int], List[int]]:
    r1 = [int(r) for r in bundle.get("round1", {}).get("selected_regions", [])]
    r2 = [int(r) for r in bundle.get("round2", {}).get("selected_regions", [])]
    selected_round: Dict[int, int] = {}
    for r in r1:
        selected_round[r] = 1
    for r in r2:
        selected_round.setdefault(r, 2)
    return selected_round, r1, r2


def _extract_region_dopt_best(bundle: Dict) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for round_key in ("round1", "round2"):
        rfm = bundle.get(round_key, {}).get("region_best_fitness", {})
        for r, v in rfm.items():
            out[int(r)] = float(v)
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Empirical per-region analysis for two-level DOPP "
            "(size, rank percentiles, top-K capture, etc.)."
        )
    )
    p.add_argument(
        "results_npy",
        type=Path,
        help="Path to two_level_results.npy produced by two_level_dopp.py.",
    )
    p.add_argument(
        "fitness_csv",
        type=Path,
        help="Path to metrics.csv with Fitness column (or use --metrics).",
    )
    p.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Recompute Fitness from these metrics columns instead of using the CSV column.",
    )
    p.add_argument(
        "--top-k-truth",
        type=int,
        nargs="+",
        default=(10, 20, 50, 100),
        help="Ks for global top-K capture summary.",
    )
    p.add_argument(
        "--top-pct",
        type=float,
        default=10.0,
        help="Top-percent cutoff (in %%) for the per-region top-K hits column.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output markdown file. Default: <results_npy parent>/region_stats.md "
            "(same folder as two_level_results.npy)."
        ),
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def _markdown_summary(stats_df: pd.DataFrame, summary: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Two-Level DOPP Region Analysis")
    lines.append("")
    lines.append(
        f"- Total candidates: **{summary['n_samples']}**, "
        f"Total regions: **{summary['n_regions']}**"
    )
    rs = summary["region_size"]
    lines.append(
        f"- Region size: min={rs['min']}, median={rs['median']:.1f}, "
        f"max={rs['max']}, mean={rs['mean']:.2f}, std={rs['std']:.2f}"
    )
    br = summary["best_rank_per_region"]
    lines.append(
        f"- Best global-rank in a region: min={br['min']}, "
        f"p25={br['p25']:.1f}, median={br['median']:.1f}, "
        f"p75={br['p75']:.1f}, max={br['max']}"
    )

    lines.append("")
    lines.append("## Where the global top-K live")
    for k_label, payload in summary["top_k_capture"].items():
        lines.append(
            f"- **{k_label}**: spread across **{payload['regions_with_hits']}** region(s); "
            f"per-region hits = {payload['per_region_hits']}"
        )

    if "selection_capture" in summary:
        lines.append("")
        lines.append("## Round 1 / Round 2 capture of true top-K")
        for k_label, payload in summary["selection_capture"].items():
            lines.append(
                f"- {k_label}: Round-1 captured **{payload['captured_by_round1']} / {payload['k']}**, "
                f"Round-1+Round-2 captured **{payload['captured_by_round1_or_2']} / {payload['k']}**"
            )

    if "oracle_gap" in summary:
        og = summary["oracle_gap"]
        lines.append("")
        lines.append("## Inner D-opt oracle gap (on evaluated regions)")
        lines.append(
            f"- evaluated regions: {og['n_evaluated_regions']}, "
            f"gap mean={og['mean']:.4f}, median={og['median']:.4f}, "
            f"max={og['max']:.4f}, "
            f"frac. of regions where D-opt found the true region-min = "
            f"{og['frac_regions_hit_true_min']:.2%}"
        )

    lines.append("")
    lines.append("## Per-region table (all regions, sorted by best_rank)")
    table_df = stats_df.sort_values(
        ["size", "best_rank"], ascending=[False, True]
    ).reset_index(drop=True)
    try:
        table_md = table_df.to_markdown(index=False)
    except ImportError:
        table_md = table_df.to_string(index=False)
    lines.append(table_md)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.results_npy.exists():
        raise FileNotFoundError(f"Results file not found: {args.results_npy}")
    if not args.fitness_csv.exists():
        raise FileNotFoundError(f"Fitness CSV not found: {args.fitness_csv}")

    out_path = (
        Path(args.output)
        if args.output is not None
        else args.results_npy.parent / "region_stats.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Loading two-level results bundle: %s", args.results_npy)
    bundle = _load_results_bundle(args.results_npy)
    region_indices = _extract_region_indices(bundle)
    selected_round, r1_regions, r2_regions = _extract_selected_round(bundle)
    region_dopt_best = _extract_region_dopt_best(bundle)

    candidate_keys = list(bundle.get("candidate_keys") or [])
    if not candidate_keys:
        raise ValueError(
            "Results bundle does not contain candidate_keys; cannot align fitness."
        )

    logging.info("Loading fitness from %s", args.fitness_csv)
    fitness_dict = load_fitness_scores_from_csv(args.fitness_csv, metrics=args.metrics)
    y = align_fitness(fitness_dict, candidate_keys).astype(np.float64)

    logging.info(
        "Computing per-region stats (n_regions=%d, n_samples=%d)",
        len(region_indices),
        y.size,
    )
    stats_df = compute_region_stats(
        region_indices=region_indices,
        y=y,
        top_pct_cutoff=float(args.top_pct),
        selected_round=selected_round,
        region_dopt_best=region_dopt_best,
    )

    summary = compute_global_summary(
        stats_df=stats_df,
        y=y,
        region_indices=region_indices,
        round1_regions=r1_regions,
        round2_regions=r2_regions,
        top_k_truth=tuple(args.top_k_truth),
    )

    md_text = _markdown_summary(stats_df, summary)
    out_path.write_text(md_text, encoding="utf-8")
    logging.info(
        "Wrote region analysis markdown: %s (regions=%d, cols=%d)",
        out_path,
        len(stats_df),
        len(stats_df.columns),
    )

    print(md_text)


if __name__ == "__main__":
    main()
