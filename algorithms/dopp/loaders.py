from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation.get_metrics import cal_fitness_score


def load_features_from_file(
    features_path: Path,
    fitness_csv: Optional[Path] = None,
    feature_type: str = "features",
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Load a standardized DOPP feature bundle.

    If ``fitness_csv`` is provided, candidates without finite fitness values are
    filtered out so the feature matrix and returned keys stay aligned.
    """
    data = np.load(features_path, allow_pickle=True).item()
    candidate_keys = [str(key) for key in data["candidate_keys"]]
    feature_matrix = np.asarray(data["features"], dtype=np.float32)
    feature_dim = int(data.get("feature_dim", feature_matrix.shape[1]))

    if fitness_csv is not None:
        fitness_dict = load_fitness_scores_from_csv(fitness_csv)
        valid_indices = [
            idx
            for idx, key in enumerate(candidate_keys)
            if key in fitness_dict and np.isfinite(fitness_dict[key])
        ]

        if len(valid_indices) != len(candidate_keys):
            dropped = len(candidate_keys) - len(valid_indices)
            logging.info("Dropped %d candidates with NaN/inf fitness scores", dropped)
            candidate_keys = [candidate_keys[idx] for idx in valid_indices]
            feature_matrix = feature_matrix[valid_indices]

    bundle_feature_type = data.get("feature_type", feature_type)
    logging.info("Loaded %s features: shape=%s", bundle_feature_type, feature_matrix.shape)
    logging.info("  Number of candidates: %d", len(candidate_keys))
    logging.info("  Feature dimension: %d", feature_dim)

    metadata = {
        "feature_type": bundle_feature_type,
        "candidate_keys": candidate_keys,
        "feature_dim": feature_dim,
        "num_candidates": len(candidate_keys),
        "metadata": data.get("metadata", {}),
    }

    return feature_matrix, candidate_keys, metadata


def load_fitness_scores_from_csv(
    csv_path: Path,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Load fitness scores from a metrics CSV.

    If ``metrics`` is provided, fitness is recalculated from those metrics even
    when the CSV already contains a ``Fitness`` column.
    """
    df = pd.read_csv(csv_path)

    if metrics is not None:
        logging.info("Recalculating fitness scores from metrics: %s", metrics)
        df, _ = cal_fitness_score(df, metrics)
    elif "Fitness" in df.columns:
        logging.info("Using existing Fitness scores from CSV file")
    else:
        raise ValueError(
            "CSV file does not contain 'Fitness' column and no metrics provided "
            "for calculation"
        )

    return {
        str(row["Key"]): float(row["Fitness"])
        for _, row in df.iterrows()
    }


def load_candidates_from_json(
    json_path: Path,
) -> List[Tuple[str, List[List[int]], Tuple[float, float]]]:
    """
    Load candidate solutions from an HMSA results JSON file.

    Returns a list of ``(key, solution, cost)`` tuples.
    """
    with open(json_path, "r") as fp:
        data = json.load(fp)

    candidates = []
    for key, entries in data["pareto_archive"]["solutions"].items():
        if isinstance(entries, list):
            iterable = [(f"{key}_{idx}", entry) for idx, entry in enumerate(entries)]
        else:
            iterable = [(str(key), entries)]

        for candidate_key, entry in iterable:
            raw_solution = entry.get("solution", [[], []])
            cost = entry.get("cost", [0.0, 0.0])

            lower_ids = [int(node_id) for node_id in raw_solution[0]]
            upper_ids = [int(node_id) for node_id in raw_solution[1]]
            cut_size = float(cost[0])
            area_imbalance = float(cost[1])

            candidates.append(
                (candidate_key, [lower_ids, upper_ids], (cut_size, area_imbalance))
            )

    return candidates
