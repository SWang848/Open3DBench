"""Budget and coverage report for the two-level DOPP baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from algorithms.dopp.baseline_analysis_utils import (
    coverage_for_indices,
    coverage_text,
    default_report_path,
    format_float,
    load_bundle,
    load_inputs,
    ordered_unique,
    per_region_map,
    ranked_baseline,
    write_report,
)


def _random_candidate_baseline(
    y: np.ndarray,
    budget: int,
    top_k_truth: Sequence[int],
    rng: np.random.Generator,
    trials: int,
) -> Dict[str, object]:
    bests: List[float] = []
    cov: Dict[str, List[int]] = {f"top_{min(int(k), y.size)}": [] for k in top_k_truth}
    budget_eff = min(int(budget), y.size)
    for _ in range(trials):
        sample = rng.choice(y.size, size=budget_eff, replace=False)
        bests.append(float(y[sample].min()))
        coverage = coverage_for_indices(sample.tolist(), y, top_k_truth)
        for key, value in coverage.items():
            cov[key].append(value)
    return {
        "best_mean": float(np.mean(bests)),
        "best_std": float(np.std(bests)),
        "best_min": float(np.min(bests)),
        "coverage_mean": {k: float(np.mean(v)) for k, v in cov.items()},
    }


def _random_region_candidate_baseline(
    regions: List[List[int]],
    y: np.ndarray,
    budget: int,
    top_k_truth: Sequence[int],
    rng: np.random.Generator,
    trials: int,
) -> Dict[str, object]:
    nonempty_regions = [i for i, members in enumerate(regions) if members]
    bests: List[float] = []
    cov: Dict[str, List[int]] = {f"top_{min(int(k), y.size)}": [] for k in top_k_truth}
    budget_eff = min(int(budget), y.size)
    for _ in range(trials):
        selected = set()
        attempts = 0
        while len(selected) < budget_eff and attempts < budget_eff * 100:
            r = int(rng.choice(nonempty_regions))
            selected.add(int(rng.choice(regions[r])))
            attempts += 1
        if len(selected) < budget_eff:
            remaining = [i for i in range(y.size) if i not in selected]
            fill = rng.choice(remaining, size=budget_eff - len(selected), replace=False)
            selected.update(int(i) for i in fill.tolist())
        selected_list = list(selected)
        bests.append(float(y[np.asarray(selected_list, dtype=np.int64)].min()))
        coverage = coverage_for_indices(selected_list, y, top_k_truth)
        for key, value in coverage.items():
            cov[key].append(value)
    return {
        "best_mean": float(np.mean(bests)),
        "best_std": float(np.std(bests)),
        "best_min": float(np.min(bests)),
        "coverage_mean": {k: float(np.mean(v)) for k, v in cov.items()},
    }


def _metrics_aligned(metrics_csv: Path, candidate_keys: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    df["Key"] = df["Key"].astype(str)
    df = df.drop_duplicates(subset="Key", keep="first").set_index("Key")
    missing = [key for key in candidate_keys if key not in df.index]
    if missing:
        raise ValueError(
            "Metrics CSV is missing candidate keys needed for analysis "
            f"(showing first 10 of {len(missing)}): {missing[:10]}"
        )
    return df.loc[list(candidate_keys)].reset_index()


def _proxy_baselines(
    metrics_csv: Path,
    candidate_keys: Sequence[str],
    y: np.ndarray,
    budget: int,
    top_k_truth: Sequence[int],
) -> Dict[str, Dict[str, object]]:
    df = _metrics_aligned(metrics_csv, candidate_keys)
    budget_eff = min(int(budget), len(df))
    out: Dict[str, Dict[str, object]] = {}
    if "Cut_size" in df:
        cut = pd.to_numeric(df["Cut_size"], errors="coerce").to_numpy(dtype=np.float64)
        out["cut_size"] = ranked_baseline(np.argsort(cut)[:budget_eff].tolist(), y, top_k_truth)
    if "Area_imbalance" in df:
        imb = pd.to_numeric(df["Area_imbalance"], errors="coerce").to_numpy(dtype=np.float64)
        out["area_imbalance"] = ranked_baseline(np.argsort(imb)[:budget_eff].tolist(), y, top_k_truth)
    if "Cut_size" in df and "Area_imbalance" in df:
        cut_z = (cut - np.nanmean(cut)) / (np.nanstd(cut) + 1e-12)
        imb_z = (imb - np.nanmean(imb)) / (np.nanstd(imb) + 1e-12)
        proxy = cut_z + imb_z
        out["cut_plus_imbalance_z"] = ranked_baseline(
            np.argsort(proxy)[:budget_eff].tolist(),
            y,
            top_k_truth,
        )
    return out


def _single_stage_dopt_baseline(
    dopt_path: Optional[Path],
    candidate_keys: Sequence[str],
    y: np.ndarray,
    budget: int,
    top_k_truth: Sequence[int],
) -> Optional[Dict[str, object]]:
    if dopt_path is None or not dopt_path.exists():
        return None
    bundle = load_bundle(dopt_path)
    weights = bundle.get("weights")
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        dopt_keys = [str(k) for k in bundle.get("candidate_keys", candidate_keys)]
        if len(dopt_keys) == len(candidate_keys) and dopt_keys == list(candidate_keys):
            order = np.argsort(weights)[::-1]
            return ranked_baseline(order[: min(int(budget), len(order))].tolist(), y, top_k_truth)

        key_to_idx = {key: i for i, key in enumerate(candidate_keys)}
        ranked: List[int] = []
        for idx in np.argsort(weights)[::-1].tolist():
            if idx < len(dopt_keys) and dopt_keys[idx] in key_to_idx:
                ranked.append(key_to_idx[dopt_keys[idx]])
            if len(ranked) >= budget:
                break
        if ranked:
            return ranked_baseline(ranked, y, top_k_truth)

    selected = bundle.get("selected_indices")
    if selected is None:
        return None
    return ranked_baseline(
        [int(i) for i in selected[: min(int(budget), len(selected))]],
        y,
        top_k_truth,
    )


def build_budget_summary(
    bundle: Dict,
    regions: List[List[int]],
    metrics_csv: Path,
    candidate_keys: Sequence[str],
    y: np.ndarray,
    top_k_truth: Sequence[int],
    random_trials: int,
    random_seed: int,
    dopt_path: Optional[Path],
) -> Dict[str, object]:
    summary = bundle.get("summary", {})
    all_evaluated = [int(i) for i in summary.get("all_evaluated_indices", [])]
    if not all_evaluated:
        all_evaluated = ordered_unique(
            list(bundle.get("round1", {}).get("evaluated_indices", []))
            + list(bundle.get("round2", {}).get("evaluated_indices", []))
        )
    budget = len(set(all_evaluated))

    r1_eval = set(int(i) for i in bundle.get("round1", {}).get("evaluated_indices", []))
    r2_eval = set(int(i) for i in bundle.get("round2", {}).get("evaluated_indices", []))
    surrogate_eval = set()
    dopt_eval = set()
    exact_attribution = True
    for round_key in ("round1", "round2"):
        eval_map = per_region_map(bundle, round_key, "evaluated_per_region")
        surr_map = per_region_map(bundle, round_key, "surrogate_evaluated_per_region")
        dopt_map = per_region_map(bundle, round_key, "dopt_evaluated_per_region")
        if eval_map and not dopt_map:
            exact_attribution = False
        for r, eval_indices in eval_map.items():
            surr = set(surr_map.get(r, []))
            evaluated = set(eval_indices)
            surrogate_eval |= surr
            if r in dopt_map:
                dopt_eval |= set(dopt_map[r])
            else:
                dopt_eval |= evaluated - surr
    if not dopt_eval and not surrogate_eval:
        dopt_eval = set(all_evaluated)

    rng = np.random.default_rng(random_seed)
    baseline_dopt = _single_stage_dopt_baseline(
        dopt_path,
        candidate_keys,
        y,
        budget,
        top_k_truth,
    )
    return {
        "two_level": ranked_baseline(all_evaluated, y, top_k_truth),
        "oracle_calls": int(budget),
        "round1_calls": int(len(r1_eval)),
        "round2_calls": int(len(r2_eval)),
        "round_overlap_calls": int(len(r1_eval & r2_eval)),
        "dopt_candidate_calls": int(len(dopt_eval)),
        "surrogate_candidate_calls": int(len(surrogate_eval)),
        "surrogate_extra_calls": int(len(surrogate_eval - dopt_eval)),
        "dopt_surrogate_overlap_calls": int(len(dopt_eval & surrogate_eval)),
        "exact_attribution": bool(exact_attribution),
        "random_candidate": _random_candidate_baseline(
            y, budget, top_k_truth, rng, random_trials
        ),
        "random_region_candidate": _random_region_candidate_baseline(
            regions, y, budget, top_k_truth, rng, random_trials
        ),
        "proxy": _proxy_baselines(metrics_csv, candidate_keys, y, budget, top_k_truth),
        "single_stage_dopt": baseline_dopt,
        "single_stage_dopt_path": str(dopt_path) if dopt_path and dopt_path.exists() else "",
    }


def build_report(
    bundle: Dict,
    regions: List[List[int]],
    metrics_csv: Path,
    candidate_keys: Sequence[str],
    y: np.ndarray,
    top_k_truth: Sequence[int],
    random_trials: int,
    random_seed: int,
    dopt_path: Optional[Path],
    results_npy: Path,
) -> str:
    summary = build_budget_summary(
        bundle=bundle,
        regions=regions,
        metrics_csv=metrics_csv,
        candidate_keys=candidate_keys,
        y=y,
        top_k_truth=top_k_truth,
        random_trials=random_trials,
        random_seed=random_seed,
        dopt_path=dopt_path,
    )
    two = summary["two_level"]
    rc = summary["random_candidate"]
    rr = summary["random_region_candidate"]

    lines = [
        "# Budget and Coverage Analysis",
        "",
        f"- Results: `{results_npy}`",
        f"- Fitness CSV: `{metrics_csv}`",
        "- Lower fitness is better. All comparison baselines use the same number of oracle calls as the two-level result.",
        "",
        "## Two-Level Budget",
        "",
        (
            f"- Oracle calls: **{summary['oracle_calls']}**; "
            f"best fitness={format_float(two['best_fitness'])}; "
            f"coverage: {coverage_text(two['coverage'])}."
        ),
        (
            "- Call split: "
            f"Round-1={summary['round1_calls']}, Round-2={summary['round2_calls']}, "
            f"round overlap={summary['round_overlap_calls']}."
        ),
        (
            "- Candidate source split: "
            f"D-opt={summary['dopt_candidate_calls']}, "
            f"surrogate-proposed={summary['surrogate_candidate_calls']}, "
            f"surrogate extra unique={summary['surrogate_extra_calls']}, "
            f"D-opt/surrogate overlap={summary['dopt_surrogate_overlap_calls']}."
        ),
    ]
    if not summary["exact_attribution"]:
        lines.append(
            "- Attribution note: this result bundle does not store exact D-opt-selected candidates; "
            "candidate source counts are reconstructed conservatively."
        )

    lines.extend(
        [
            "",
            "## Same-Budget Baselines",
            "",
            (
                "- Random candidate sampling: "
                f"mean best={format_float(rc['best_mean'])} +/- {format_float(rc['best_std'])}; "
                f"best over trials={format_float(rc['best_min'])}; "
                f"mean coverage: {coverage_text(rc['coverage_mean'])}."
            ),
            (
                "- Random region plus random candidate sampling: "
                f"mean best={format_float(rr['best_mean'])} +/- {format_float(rr['best_std'])}; "
                f"best over trials={format_float(rr['best_min'])}; "
                f"mean coverage: {coverage_text(rr['coverage_mean'])}."
            ),
        ]
    )

    for name, payload in summary["proxy"].items():
        lines.append(
            f"- Proxy-only ranking ({name}): best={format_float(payload['best_fitness'])}; "
            f"coverage: {coverage_text(payload['coverage'])}."
        )
    if summary["single_stage_dopt"] is not None:
        payload = summary["single_stage_dopt"]
        lines.append(
            "- Single-stage D-opt: "
            f"best={format_float(payload['best_fitness'])}; "
            f"coverage: {coverage_text(payload['coverage'])}."
        )
    else:
        lines.append("- Single-stage D-opt: TODO (no compatible D-opt result found).")

    lines.extend(
        [
            "",
            "## Main Questions",
            "",
            "- Is two-level DOPP better than simple alternatives at the same budget?",
            "- Does the surrogate stage justify its extra evaluations?",
            "- Which stage consumes budget without improving final candidate quality?",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the budget and coverage report.")
    parser.add_argument("results_npy", type=Path, help="Path to two_level_results.npy.")
    parser.add_argument("fitness_csv", type=Path, help="Matching metrics.csv.")
    parser.add_argument("--metrics", type=str, nargs="+", default=None)
    parser.add_argument("--top-k-truth", type=int, nargs="+", default=(10, 20, 50, 100))
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--d-opt-results", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    bundle, candidate_keys, y, regions = load_inputs(
        args.results_npy,
        args.fitness_csv,
        args.metrics,
    )
    dopt_path = args.d_opt_results
    if dopt_path is None:
        candidates = [
            args.results_npy.parent / "d_optimal_results.npy",
            args.results_npy.parent.parent / "d_optimal_results.npy",
        ]
        dopt_path = next((p for p in candidates if p.exists()), None)

    output = args.output or default_report_path(args.results_npy, "budget_coverage_analysis.md")
    report = build_report(
        bundle=bundle,
        regions=regions,
        metrics_csv=args.fitness_csv,
        candidate_keys=candidate_keys,
        y=y,
        top_k_truth=args.top_k_truth,
        random_trials=max(1, int(args.random_trials)),
        random_seed=int(args.random_seed),
        dopt_path=dopt_path,
        results_npy=args.results_npy,
    )
    write_report(output, report)
    logging.info("Wrote budget and coverage report: %s", output)
    print(report)


if __name__ == "__main__":
    main()
