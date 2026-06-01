"""Shared helpers for the modular two-level DOPP baseline analyses."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import kendalltau, spearmanr
except ImportError:  # pragma: no cover - scipy is expected but analysis can degrade.
    kendalltau = None
    spearmanr = None


def load_bundle(path: Path) -> Dict:
    return np.load(path, allow_pickle=True).item()


def load_fitness_scores(csv_path: Path, metrics: Optional[List[str]] = None) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    if metrics is not None:
        from evaluation.get_metrics import cal_fitness_score

        df, _ = cal_fitness_score(df, metrics)
    elif "Fitness" not in df.columns:
        raise ValueError(
            "CSV file does not contain 'Fitness' column and no metrics were provided"
        )
    return {str(row["Key"]): float(row["Fitness"]) for _, row in df.iterrows()}


def align_fitness(fitness_dict: Dict[str, float], candidate_keys: Sequence[str]) -> np.ndarray:
    y = np.full(len(candidate_keys), np.nan, dtype=np.float64)
    missing: List[str] = []
    for idx, key in enumerate(candidate_keys):
        value = fitness_dict.get(str(key))
        if value is None or not np.isfinite(value):
            missing.append(str(key))
        else:
            y[idx] = float(value)
    if missing:
        raise ValueError(
            "Missing finite fitness for candidate keys "
            f"(showing first 10 of {len(missing)}): {missing[:10]}"
        )
    return y


def load_inputs(
    results_npy: Path,
    fitness_csv: Path,
    metrics: Optional[List[str]] = None,
) -> Tuple[Dict, List[str], np.ndarray, List[List[int]]]:
    if not results_npy.exists():
        raise FileNotFoundError(f"Results file not found: {results_npy}")
    if not fitness_csv.exists():
        raise FileNotFoundError(f"Fitness CSV not found: {fitness_csv}")

    bundle = load_bundle(results_npy)
    candidate_keys = [str(k) for k in bundle.get("candidate_keys", [])]
    if not candidate_keys:
        raise ValueError("Result bundle is missing candidate_keys")

    fitness_dict = load_fitness_scores(fitness_csv, metrics=metrics)
    y = align_fitness(fitness_dict, candidate_keys).astype(np.float64, copy=False)
    return bundle, candidate_keys, y, region_indices(bundle)


def rank_array(y: np.ndarray) -> np.ndarray:
    order = np.argsort(y, kind="stable")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(y) + 1, dtype=np.int64)
    return ranks


def safe_corr(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return {
            "spearman": float("nan"),
            "spearman_p": float("nan"),
            "kendall": float("nan"),
            "kendall_p": float("nan"),
        }
    if spearmanr is None or kendalltau is None:
        logging.warning("scipy is unavailable; Spearman/Kendall correlations set to NaN")
        return {
            "spearman": float("nan"),
            "spearman_p": float("nan"),
            "kendall": float("nan"),
            "kendall_p": float("nan"),
        }
    s_val, s_p = spearmanr(a[mask], b[mask])
    k_val, k_p = kendalltau(a[mask], b[mask])
    return {
        "spearman": float(s_val) if np.isfinite(s_val) else float("nan"),
        "spearman_p": float(s_p) if np.isfinite(s_p) else float("nan"),
        "kendall": float(k_val) if np.isfinite(k_val) else float("nan"),
        "kendall_p": float(k_p) if np.isfinite(k_p) else float("nan"),
    }


def format_float(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def format_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{100.0 * value:.2f}%"


def ordered_unique(values: Iterable[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def region_indices(bundle: Dict) -> List[List[int]]:
    raw = bundle.get("clustering", {}).get("region_indices")
    if raw is None:
        raise ValueError("Result bundle is missing clustering.region_indices")
    return [list(map(int, members)) for members in raw]


def selected_regions(bundle: Dict, round_key: str) -> List[int]:
    return [int(r) for r in bundle.get(round_key, {}).get("selected_regions", [])]


def all_selected_regions(bundle: Dict) -> List[int]:
    return ordered_unique(selected_regions(bundle, "round1") + selected_regions(bundle, "round2"))


def surrogate_selected_regions(bundle: Dict) -> List[int]:
    r2 = bundle.get("round2", {})
    selected = r2.get("surrogate_selected_regions")
    if selected is None:
        selected = r2.get("selected_regions", [])
    return [int(r) for r in selected]


def per_region_map(bundle: Dict, round_key: str, field: str) -> Dict[int, List[int]]:
    raw = bundle.get(round_key, {}).get(field, {})
    if not raw:
        return {}
    return {int(r): [int(i) for i in values] for r, values in raw.items()}


def round_region_best(bundle: Dict, round_key: str) -> Dict[int, float]:
    return {
        int(r): float(v)
        for r, v in bundle.get(round_key, {}).get("region_best_fitness", {}).items()
    }


def round_region_source(bundle: Dict, round_key: str) -> Dict[int, str]:
    return {
        int(r): str(v)
        for r, v in bundle.get(round_key, {}).get("region_best_source", {}).items()
    }


def truth_for_regions(
    regions: List[List[int]],
    y: np.ndarray,
    top_k_truth: Sequence[int],
) -> pd.DataFrame:
    ranks = rank_array(y)
    rows: List[Dict[str, object]] = []
    for r, member_list in enumerate(regions):
        members = np.asarray(member_list, dtype=np.int64)
        if members.size == 0:
            rows.append(
                {
                    "region_id": r,
                    "size": 0,
                    "true_best_index": -1,
                    "true_best_fitness": np.nan,
                    "true_worst_fitness": np.nan,
                    "median_fitness": np.nan,
                    "true_best_rank": np.nan,
                    "median_rank": np.nan,
                    "p10_rank": np.nan,
                    "p25_rank": np.nan,
                    "p75_rank": np.nan,
                    "p90_rank": np.nan,
                    "fitness_variance": np.nan,
                    "worst_rank": np.nan,
                }
            )
            continue

        member_y = y[members]
        member_ranks = ranks[members]
        p10_rank = float(np.percentile(member_ranks, 10))
        p90_rank = float(np.percentile(member_ranks, 90))
        best_pos = int(np.argmin(member_y))
        row: Dict[str, object] = {
            "region_id": r,
            "size": int(members.size),
            "true_best_index": int(members[best_pos]),
            "true_best_fitness": float(member_y[best_pos]),
            "true_worst_fitness": float(np.max(member_y)),
            "median_fitness": float(np.median(member_y)),
            "true_best_rank": int(member_ranks[best_pos]),
            "median_rank": float(np.median(member_ranks)),
            "p10_rank": p10_rank,
            "p25_rank": float(np.percentile(member_ranks, 25)),
            "p75_rank": float(np.percentile(member_ranks, 75)),
            "p90_rank": p90_rank,
            "fitness_variance": float(np.var(member_y)),
            "worst_rank": int(member_ranks.max()),
        }
        for k in top_k_truth:
            row[f"top_{k}_hits"] = int(np.sum(member_ranks <= min(int(k), y.size)))
        rows.append(row)

    df = pd.DataFrame(rows)
    valid = df["size"] > 0
    df.loc[valid, "true_region_rank"] = (
        df.loc[valid, "true_best_fitness"].rank(method="first", ascending=True).astype(int)
    )
    return df


def region_prediction_scores(bundle: Dict, n_regions: int) -> Optional[np.ndarray]:
    predicted = bundle.get("round2", {}).get("predicted_region_fitness")
    if predicted:
        out = np.full(n_regions, np.nan, dtype=np.float64)
        for r, value in predicted.items():
            out[int(r)] = float(value)
        return out

    x_region = bundle.get("region_features", {}).get("X_region")
    coef = bundle.get("surrogate", {}).get("coef")
    intercept = bundle.get("surrogate", {}).get("intercept")
    if x_region is None or coef is None or intercept is None:
        predicted_unlabeled = bundle.get("round2", {}).get("predicted_region_fitness_unlabeled")
        if not predicted_unlabeled:
            return None
        out = np.full(n_regions, np.nan, dtype=np.float64)
        for r, value in predicted_unlabeled.items():
            out[int(r)] = float(value)
        return out

    x_region = np.asarray(x_region, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64).reshape(-1)
    return (x_region @ coef + float(np.asarray(intercept).reshape(-1)[0])).astype(
        np.float64,
        copy=False,
    )


def coverage_for_indices(
    indices: Sequence[int],
    y: np.ndarray,
    top_k_truth: Sequence[int],
) -> Dict[str, int]:
    selected = set(int(i) for i in indices)
    order = np.argsort(y, kind="stable")
    out: Dict[str, int] = {}
    for k in top_k_truth:
        k_eff = min(int(k), y.size)
        out[f"top_{k_eff}"] = int(len(set(order[:k_eff].tolist()) & selected))
    return out


def best_for_indices(indices: Sequence[int], y: np.ndarray) -> Tuple[float, int]:
    if not indices:
        return float("nan"), -1
    arr = np.asarray(list(indices), dtype=np.int64)
    pos = int(np.argmin(y[arr]))
    return float(y[arr[pos]]), int(arr[pos])


def ranked_baseline(indices: Sequence[int], y: np.ndarray, top_k_truth: Sequence[int]) -> Dict[str, object]:
    best, best_idx = best_for_indices(indices, y)
    return {
        "best_fitness": best,
        "best_index": best_idx,
        "coverage": coverage_for_indices(indices, y, top_k_truth),
        "selected_count": int(len(set(int(i) for i in indices))),
    }


def coverage_text(coverage: Dict[str, object]) -> str:
    parts = []
    for key, value in coverage.items():
        if isinstance(value, (float, np.floating)):
            parts.append(f"{key}={float(value):.2f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def write_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def default_report_path(results_npy: Path, filename: str) -> Path:
    default = results_npy.parent / filename
    if default.exists():
        if os.access(default, os.W_OK):
            return default
    elif os.access(default.parent, os.W_OK):
        return default
    result_label = results_npy.parent.name or "baseline_result"
    if result_label == "two_level" and results_npy.parent.parent.name:
        result_label = f"{results_npy.parent.parent.name}_{result_label}"
    return Path("reports") / result_label / filename
