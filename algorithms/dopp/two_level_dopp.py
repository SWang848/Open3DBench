"""Two-level DOPP baseline.

Splits the ~10k candidates into balanced regions in solution-feature space,
runs region-level D-optimal design (Round 1) to pick informative regions,
runs solution-level D-optimal design inside each picked region to query the
expensive PPA oracle (looked up from ``metrics.csv``), uses local solution
surrogates to propose additional candidates inside selected regions, trains a
region-level linear surrogate on the labeled regions, and uses it to pick a
second batch of regions (Round 2).

The implementation reuses the existing ``frank_wolfe_d_optimal`` /
``select_candidates_by_weights`` solvers from ``algorithms.dopp.d_opt`` and
shared data-loading helpers from ``algorithms.dopp.loaders``.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.linalg as la
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from algorithms.dopp.d_opt import (
    frank_wolfe_d_optimal,
    select_candidates_by_weights,
)
from algorithms.dopp.loaders import load_features_from_file, load_fitness_scores_from_csv


# ----------------------------------------------------------------------------
# Data loading helpers
# ----------------------------------------------------------------------------


def load_proxies_from_csv(
    csv_path: Path,
    candidate_keys: Sequence[str],
) -> np.ndarray:
    """Load ``Cut_size`` and ``Area_imbalance`` aligned to ``candidate_keys``.

    Returns an ``(N, 2)`` float32 array. Raises ``ValueError`` if any candidate
    key is missing or has a non-finite proxy value, so we never silently inject
    zeros into the region feature aggregation.
    """
    df = pd.read_csv(csv_path)
    required = {"Key", "Cut_size", "Area_imbalance"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Metrics CSV is missing required columns: {sorted(missing_cols)}"
        )

    df = df.copy()
    df["Key"] = df["Key"].astype(str)
    df = df.drop_duplicates(subset="Key", keep="first")
    cut = pd.to_numeric(df["Cut_size"], errors="coerce")
    imb = pd.to_numeric(df["Area_imbalance"], errors="coerce")

    lookup: Dict[str, Tuple[float, float]] = {}
    for key, c, i in zip(df["Key"].tolist(), cut.tolist(), imb.tolist()):
        if c is None or i is None:
            continue
        if not (np.isfinite(c) and np.isfinite(i)):
            continue
        lookup[key] = (float(c), float(i))

    proxies = np.zeros((len(candidate_keys), 2), dtype=np.float32)
    missing_keys: List[str] = []
    for idx, key in enumerate(candidate_keys):
        entry = lookup.get(str(key))
        if entry is None:
            missing_keys.append(str(key))
            continue
        proxies[idx, 0] = entry[0]
        proxies[idx, 1] = entry[1]

    if missing_keys:
        raise ValueError(
            "Missing finite Cut_size/Area_imbalance for candidate keys "
            f"(showing first 10 of {len(missing_keys)}): {missing_keys[:10]}"
        )

    return proxies


def align_fitness(
    fitness_dict: Dict[str, float],
    candidate_keys: Sequence[str],
) -> np.ndarray:
    """Return fitness aligned to ``candidate_keys`` as float32 array."""
    y = np.full(len(candidate_keys), np.nan, dtype=np.float32)
    missing: List[str] = []
    for idx, key in enumerate(candidate_keys):
        val = fitness_dict.get(str(key))
        if val is None or not np.isfinite(val):
            missing.append(str(key))
            continue
        y[idx] = float(val)
    if missing:
        raise ValueError(
            "Missing finite fitness for candidate keys "
            f"(showing first 10 of {len(missing)}): {missing[:10]}"
        )
    return y


# ----------------------------------------------------------------------------
# Balanced clustering
# ----------------------------------------------------------------------------


def _balanced_size_bounds(n_samples: int, n_clusters: int) -> Tuple[int, int]:
    size_min = n_samples // n_clusters
    size_max = math.ceil(n_samples / n_clusters) + 1
    return size_min, size_max


def balanced_kmeans_constrained(
    X_std: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
) -> np.ndarray:
    """Balanced KMeans via ``k_means_constrained.KMeansConstrained``."""
    try:
        from k_means_constrained import KMeansConstrained
    except ImportError as exc:
        raise ImportError(
            "balanced-method=constrained requires k-means-constrained. "
            "Install with `pip install k-means-constrained` or rerun with "
            "--balanced-method reassign."
        ) from exc

    size_min, size_max = _balanced_size_bounds(X_std.shape[0], n_clusters)
    model = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=random_state,
        n_init=10,
    )
    return model.fit_predict(X_std).astype(np.int64)


def balanced_kmeans_reassign(
    X_std: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
    max_iters: Optional[int] = None,
) -> np.ndarray:
    """Vanilla KMeans + greedy balancing reassignment (no extra dependency).

    Repeatedly moves a single point from the most-overfull cluster (donor) to
    the most-underfull cluster (receiver), picking the point in the donor that
    is *closest* to the receiver's center (i.e., the cheapest reassignment).
    Stops once every cluster size is in ``[floor(N/K), ceil(N/K) + 1]``.
    """
    size_min, size_max = _balanced_size_bounds(X_std.shape[0], n_clusters)
    n_samples = X_std.shape[0]
    if max_iters is None:
        max_iters = 10 * n_samples

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(X_std)
    labels = model.labels_.astype(np.int64).copy()
    centers = model.cluster_centers_
    counts = np.bincount(labels, minlength=n_clusters)

    for _ in range(max_iters):
        min_c = int(counts.min())
        max_c = int(counts.max())
        if min_c >= size_min and max_c <= size_max:
            break

        # Choose donor: if there is any cluster below size_min, the most-overfull
        # cluster donates; otherwise the strictly overfull (> size_max) cluster
        # donates to an under-size_max cluster.
        if min_c < size_min:
            donor = int(np.argmax(counts))
            receiver = int(np.argmin(counts))
            if counts[donor] <= size_min:
                # Nothing to spare anywhere.
                break
        else:
            donor = int(np.argmax(counts))
            # Receiver: among clusters with count < size_max, take the smallest.
            candidates = np.where(counts < size_max)[0]
            if candidates.size == 0:
                break
            receiver = int(candidates[np.argmin(counts[candidates])])
            if counts[donor] <= size_max:
                break

        members = np.where(labels == donor)[0]
        d_to_receiver = np.linalg.norm(X_std[members] - centers[receiver], axis=1)
        point = int(members[np.argmin(d_to_receiver)])
        labels[point] = receiver
        counts[donor] -= 1
        counts[receiver] += 1

    logging.info(
        "Reassignment balancing: counts min=%d max=%d (bounds [%d, %d])",
        int(counts.min()),
        int(counts.max()),
        size_min,
        size_max,
    )
    return labels


def cluster_balanced(
    X_std: np.ndarray,
    n_clusters: int,
    method: str,
    random_state: int = 0,
) -> np.ndarray:
    if method == "constrained":
        try:
            return balanced_kmeans_constrained(X_std, n_clusters, random_state)
        except ImportError as exc:
            logging.warning(
                "k-means-constrained unavailable (%s); falling back to "
                "--balanced-method reassign.",
                exc,
            )
            return balanced_kmeans_reassign(X_std, n_clusters, random_state)
    if method == "reassign":
        return balanced_kmeans_reassign(X_std, n_clusters, random_state)
    raise ValueError(f"Unknown balanced-method: {method!r}")


# ----------------------------------------------------------------------------
# Region feature construction
# ----------------------------------------------------------------------------


def build_region_features(
    proxies: np.ndarray,
    Z: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """Compute region-level feature matrix.

    Per region we stack:

    - cut min/max/mean/std (4 dims)
    - imb min/max/mean/std (4 dims)
    - region size            (1 dim)
    - PCA-10 mean            (n_components dims)
    - PCA-10 std             (n_components dims)

    Returns ``(X_region, region_sizes, region_indices)`` where
    ``region_indices[r]`` is the list of solution indices in region ``r``.
    """
    n_components = Z.shape[1]
    feat_dim = 9 + 2 * n_components
    X_region = np.zeros((n_clusters, feat_dim), dtype=np.float64)
    region_sizes = np.zeros(n_clusters, dtype=np.int64)
    region_indices: List[List[int]] = [[] for _ in range(n_clusters)]

    for idx, r in enumerate(labels.tolist()):
        region_indices[r].append(idx)

    for r in range(n_clusters):
        members = np.asarray(region_indices[r], dtype=np.int64)
        if members.size == 0:
            logging.warning("Region %d has zero members; leaving features at 0", r)
            continue

        cut_vals = proxies[members, 0]
        imb_vals = proxies[members, 1]
        z_block = Z[members]
        # std uses default ddof=0 for consistency with sklearn / numpy conventions
        if members.size == 1:
            cut_std = 0.0
            imb_std = 0.0
            pca_std = np.zeros(n_components, dtype=np.float64)
        else:
            cut_std = float(np.std(cut_vals))
            imb_std = float(np.std(imb_vals))
            pca_std = z_block.std(axis=0).astype(np.float64)

        X_region[r, 0] = float(np.min(cut_vals))
        X_region[r, 1] = float(np.max(cut_vals))
        X_region[r, 2] = float(np.mean(cut_vals))
        X_region[r, 3] = cut_std
        X_region[r, 4] = float(np.min(imb_vals))
        X_region[r, 5] = float(np.max(imb_vals))
        X_region[r, 6] = float(np.mean(imb_vals))
        X_region[r, 7] = imb_std
        X_region[r, 8] = float(members.size)
        X_region[r, 9 : 9 + n_components] = z_block.mean(axis=0)
        X_region[r, 9 + n_components : 9 + 2 * n_components] = pca_std
        region_sizes[r] = members.size

    return X_region, region_sizes, region_indices


def qr_rank_cleanup(
    X: np.ndarray,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, List[int]]:
    """QR-pivot rank cleanup (mirrors ``manual_feature_constructor``)."""
    _, r_matrix, piv = la.qr(X, mode="economic", pivoting=True)
    rank = int(np.sum(np.abs(np.diag(r_matrix)) > tol))
    independent_columns = np.sort(piv[:rank])
    dependent_columns = np.sort(piv[rank:])

    if dependent_columns.size > 0:
        logging.info(
            "Region features: dropped %d linearly dependent columns: %s",
            dependent_columns.size,
            dependent_columns.tolist(),
        )
    else:
        logging.info("Region features: full rank, no columns dropped")

    cleaned = X[:, independent_columns]
    return cleaned, independent_columns.tolist()


# ----------------------------------------------------------------------------
# Round-level helpers
# ----------------------------------------------------------------------------


def select_top_regions_by_weight(
    weights: np.ndarray,
    top_k: int,
    excluded: Optional[Sequence[int]] = None,
) -> List[int]:
    """Pick the ``top_k`` highest-weight regions, optionally excluding indices."""
    excluded_set = set(int(i) for i in excluded) if excluded else set()
    order = np.argsort(weights)[::-1]
    picked: List[int] = []
    for r in order.tolist():
        if r in excluded_set:
            continue
        picked.append(int(r))
        if len(picked) >= top_k:
            break
    return picked


def inner_dopt_select(
    X_region_solutions: np.ndarray,
    inner_top_k_frac: float,
    fw_tol: float,
    fw_step_scheme: str,
    fw_epsilon: float,
) -> np.ndarray:
    """Run Frank-Wolfe inside a region and return selected local indices."""
    if X_region_solutions.shape[0] == 1:
        return np.array([0], dtype=np.int64)

    w, _ = frank_wolfe_d_optimal(
        X_region_solutions,
        tol=fw_tol,
        step_scheme=fw_step_scheme,
        epsilon=fw_epsilon,
        verbose=False,
    )
    sel_inner, _ = select_candidates_by_weights(w, top_k=inner_top_k_frac)
    return np.asarray(sel_inner, dtype=np.int64)


def select_top_predicted_candidates(
    predicted_scores: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Pick the best predicted local indices among all candidates."""
    if top_k <= 0:
        return np.array([], dtype=np.int64)

    k_eff = min(top_k, predicted_scores.shape[0])
    return np.argsort(predicted_scores)[:k_eff].astype(np.int64, copy=False)


def evaluate_regions_via_oracle(
    region_ids: Sequence[int],
    region_indices: List[List[int]],
    X_std: np.ndarray,
    y: np.ndarray,
    inner_pca_components: int,
    inner_top_k_frac: float,
    inner_prediction_top_k: int,
    fw_tol: float,
    fw_step_scheme: str,
    fw_epsilon: float,
    random_state: int,
) -> Tuple[
    Dict[int, float],
    Dict[int, int],
    Dict[int, str],
    List[int],
    Dict[int, List[int]],
    Dict[int, List[int]],
]:
    """Per region: inner D-opt + local surrogate -> region best fitness.

    Inner D-opt uses a local PCA basis fit only on the standardized solution
    features of the current region. The selected D-opt candidates are evaluated
    with the oracle, then a local linear surrogate predicts all candidates.
    The top ``inner_prediction_top_k`` predicted candidates are also evaluated, and the
    region best is the minimum true fitness among all evaluated candidates.

    Returns:
        ``region_best_fitness``          (region -> best evaluated fitness)
        ``region_best_solution``         (region -> best global solution index)
        ``region_best_source``           (region -> dopt or surrogate)
        ``evaluated_indices``            (flat list of all evaluated indices)
        ``evaluated_per_region``         (region -> list of evaluated indices)
        ``surrogate_evaluated_per_region`` (region -> surrogate-proposed evaluated indices)
    """
    region_best_fitness: Dict[int, float] = {}
    region_best_solution: Dict[int, int] = {}
    region_best_source: Dict[int, str] = {}
    evaluated_per_region: Dict[int, List[int]] = {}
    surrogate_evaluated_per_region: Dict[int, List[int]] = {}
    evaluated_indices: List[int] = []

    for r in region_ids:
        members = np.asarray(region_indices[r], dtype=np.int64)
        if members.size == 0:
            logging.warning("Skipping empty region %d in oracle evaluation", r)
            continue

        X_r_std = X_std[members]
        if members.size > 1:
            local_pca_dim = min(
                inner_pca_components,
                max(1, members.size - 1),
                X_r_std.shape[1],
            )
            X_r = PCA(n_components=local_pca_dim, random_state=random_state).fit_transform(
                X_r_std
            )
        else:
            X_r = X_r_std
        sel_local = inner_dopt_select(
            X_r,
            inner_top_k_frac=inner_top_k_frac,
            fw_tol=fw_tol,
            fw_step_scheme=fw_step_scheme,
            fw_epsilon=fw_epsilon,
        )
        sel_global = members[sel_local]
        y_sel = y[sel_global]

        surrogate_local = np.array([], dtype=np.int64)
        local_surrogate = LinearRegression()
        local_surrogate.fit(X_r[sel_local], y_sel)
        if inner_prediction_top_k > 0:
            predicted_scores = local_surrogate.predict(X_r).astype(np.float64, copy=False)
            surrogate_local = select_top_predicted_candidates(
                predicted_scores,
                top_k=inner_prediction_top_k,
            )

        evaluated_local = np.unique(np.concatenate([sel_local, surrogate_local]))
        evaluated_global = members[evaluated_local]
        evaluated_y = y[evaluated_global]
        best_eval_pos = int(np.argmin(evaluated_y))
        best_local_pos = int(evaluated_local[best_eval_pos])
        best_global_idx = int(evaluated_global[best_eval_pos])
        surrogate_local_set = set(int(i) for i in surrogate_local.tolist())
        best_source = "surrogate" if best_local_pos in surrogate_local_set else "dopt"

        region_best_fitness[int(r)] = float(evaluated_y[best_eval_pos])
        region_best_solution[int(r)] = best_global_idx
        region_best_source[int(r)] = best_source
        evaluated_per_region[int(r)] = evaluated_global.tolist()
        surrogate_evaluated_per_region[int(r)] = members[surrogate_local].tolist()
        evaluated_indices.extend(evaluated_global.tolist())

        logging.info(
            "  Region %d: dopt_picks=%d, surrogate_picks=%d, "
            "region_best_fitness=%.4f (%s), region_size=%d",
            int(r),
            sel_global.size,
            surrogate_local.size,
            region_best_fitness[int(r)],
            region_best_source[int(r)],
            members.size,
        )

    return (
        region_best_fitness,
        region_best_solution,
        region_best_source,
        evaluated_indices,
        evaluated_per_region,
        surrogate_evaluated_per_region,
    )


def train_region_surrogate(
    X_region_labeled: np.ndarray,
    y_region_labeled: np.ndarray,
) -> object:
    """Train region-level surrogate."""
    model = LinearRegression()
    model.fit(X_region_labeled, y_region_labeled)
    return model


# ----------------------------------------------------------------------------
# Main orchestrator
# ----------------------------------------------------------------------------


def run_two_level_dopp(
    X: np.ndarray,
    y: np.ndarray,
    proxies: np.ndarray,
    candidate_keys: List[str],
    n_regions: int,
    pca_components: int,
    balanced_method: str,
    region_top_k: int,
    round2_top_k: int,
    inner_top_k_frac: float,
    inner_prediction_top_k: int,
    fw_tol: float,
    fw_step_scheme: str,
    fw_epsilon: float,
    random_state: int,
    top_k_truth: Tuple[int, ...] = (10, 20),
) -> Dict:
    """Run the full two-level DOPP pipeline and return a result dict."""
    n_samples = X.shape[0]
    if n_regions >= n_samples:
        raise ValueError(
            f"n_regions ({n_regions}) must be < number of candidates ({n_samples})"
        )
    if inner_prediction_top_k < 0:
        raise ValueError(
            "inner_prediction_top_k must be non-negative, "
            f"got {inner_prediction_top_k}"
        )

    # Step 1: standardize solution features.
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Step 2: balanced clustering.
    logging.info(
        "Running balanced clustering (method=%s, K=%d, N=%d)",
        balanced_method,
        n_regions,
        n_samples,
    )
    labels = cluster_balanced(
        X_std=X_std,
        n_clusters=n_regions,
        method=balanced_method,
        random_state=random_state,
    )

    # Step 3: PCA on standardized features.
    pca_dim = min(pca_components, X_std.shape[1], X_std.shape[0])
    if pca_dim < pca_components:
        logging.warning(
            "Requested %d PCA components but only %d available; using %d.",
            pca_components,
            pca_dim,
            pca_dim,
        )
    pca = PCA(n_components=pca_dim, random_state=random_state)
    Z = pca.fit_transform(X_std)
    logging.info(
        "Global PCA feature matrix for region stats: shape=%s",
        Z.shape,
    )

    # Step 4: region feature construction.
    X_region_full, region_sizes, region_indices = build_region_features(
        proxies=proxies,
        Z=Z,
        labels=labels,
        n_clusters=n_regions,
    )
    logging.info(
        "Region feature matrix (pre-cleanup): shape=%s",
        X_region_full.shape,
    )

    # Step 5: QR-pivot rank cleanup.
    X_region, kept_columns = qr_rank_cleanup(X_region_full)
    logging.info(
        "Region feature matrix (post-cleanup): shape=%s, kept_dim=%d",
        X_region.shape,
        X_region.shape[1],
    )

    # Step 6: Round 1 region-level D-opt.
    logging.info(
        "Round 1: Frank-Wolfe D-opt on region features (K=%d, d=%d)",
        X_region.shape[0],
        X_region.shape[1],
    )
    w_region, fw_history_region = frank_wolfe_d_optimal(
        X_region,
        tol=fw_tol,
        step_scheme=fw_step_scheme,
        epsilon=fw_epsilon,
        verbose=False,
    )
    region_top_k_eff = min(region_top_k, n_regions)
    round1_regions = select_top_regions_by_weight(w_region, top_k=region_top_k_eff)
    logging.info("Round 1: picked %d regions: %s", len(round1_regions), round1_regions)

    # Step 7: Round 1 inner D-opt + oracle eval.
    (
        region_best_fitness_r1,
        region_best_solution_r1,
        region_best_source_r1,
        evaluated_r1,
        evaluated_per_region_r1,
        surrogate_evaluated_per_region_r1,
    ) = evaluate_regions_via_oracle(
        region_ids=round1_regions,
        region_indices=region_indices,
        X_std=X_std,
        y=y,
        inner_pca_components=pca_dim,
        inner_top_k_frac=inner_top_k_frac,
        inner_prediction_top_k=inner_prediction_top_k,
        fw_tol=fw_tol,
        fw_step_scheme=fw_step_scheme,
        fw_epsilon=fw_epsilon,
        random_state=random_state,
    )

    # Step 8: train region surrogate on Round 1 labeled regions.
    labeled_region_ids = sorted(region_best_fitness_r1.keys())
    if len(labeled_region_ids) == 0:
        raise RuntimeError("Round 1 produced zero labeled regions; cannot train surrogate")

    X_region_labeled = X_region[labeled_region_ids]
    y_region_labeled = np.array(
        [region_best_fitness_r1[r] for r in labeled_region_ids], dtype=np.float64
    )
    surrogate = train_region_surrogate(
        X_region_labeled, y_region_labeled
    )

    # Step 9: Round 2 surrogate-guided selection over all regions. This mirrors
    # the inner loop: predict every item, take the top predictions, then evaluate
    # only items not already covered by the D-opt stage.
    labeled_set = set(labeled_region_ids)
    y_region_pred_all = surrogate.predict(X_region)
    round2_top_k_eff = min(max(round2_top_k, 0), n_regions)
    round2_predicted_regions = select_top_predicted_candidates(
        y_region_pred_all,
        top_k=round2_top_k_eff,
    ).tolist()
    round2_regions = [int(r) for r in round2_predicted_regions if int(r) not in labeled_set]
    logging.info(
        "Round 2: surrogate top-%d regions: %s; new evaluations: %s",
        round2_top_k_eff,
        round2_predicted_regions,
        round2_regions,
    )

    (
        region_best_fitness_r2,
        region_best_solution_r2,
        region_best_source_r2,
        evaluated_r2,
        evaluated_per_region_r2,
        surrogate_evaluated_per_region_r2,
    ) = evaluate_regions_via_oracle(
        region_ids=round2_regions,
        region_indices=region_indices,
        X_std=X_std,
        y=y,
        inner_pca_components=pca_dim,
        inner_top_k_frac=inner_top_k_frac,
        inner_prediction_top_k=inner_prediction_top_k,
        fw_tol=fw_tol,
        fw_step_scheme=fw_step_scheme,
        fw_epsilon=fw_epsilon,
        random_state=random_state,
    )

    # Aggregate evaluated solutions across both rounds.
    all_evaluated = sorted(set(evaluated_r1) | set(evaluated_r2))
    all_evaluated_keys = [candidate_keys[i] for i in all_evaluated]
    oracle_calls = len(all_evaluated)

    # Best recommended solution across evaluated oracle scores.
    region_best_fitness_all: Dict[int, float] = {}
    region_best_fitness_all.update(region_best_fitness_r1)
    region_best_fitness_all.update(region_best_fitness_r2)
    region_best_solution_all: Dict[int, int] = {}
    region_best_solution_all.update(region_best_solution_r1)
    region_best_solution_all.update(region_best_solution_r2)
    region_best_source_all: Dict[int, str] = {}
    region_best_source_all.update(region_best_source_r1)
    region_best_source_all.update(region_best_source_r2)

    if len(region_best_fitness_all) > 0:
        best_region = min(region_best_fitness_all, key=region_best_fitness_all.get)
        best_global_idx = int(region_best_solution_all[best_region])
        best_fitness = float(region_best_fitness_all[best_region])
        best_key = candidate_keys[best_global_idx]
        best_source = region_best_source_all[best_region]
    else:
        best_global_idx = -1
        best_fitness = float("nan")
        best_key = ""
        best_source = ""

    # Top-K coverage among evaluated (true ranking by `y`).
    true_order = np.argsort(y)
    coverage_results: Dict[str, Dict] = {}
    evaluated_set = set(all_evaluated)
    for k in top_k_truth:
        k_eff = min(int(k), n_samples)
        top_k_true = true_order[:k_eff].tolist()
        hits = [idx for idx in top_k_true if idx in evaluated_set]
        coverage_results[f"top_{k}"] = {
            "k": k_eff,
            "hits": len(hits),
            "true_indices": top_k_true,
            "true_keys": [candidate_keys[i] for i in top_k_true],
            "hit_indices": hits,
            "hit_keys": [candidate_keys[i] for i in hits],
        }
        logging.info(
            "True top-%d coverage in evaluated: %d / %d", k, len(hits), k_eff
        )

    logging.info("Best fitness across two rounds: %.4f", best_fitness)
    logging.info("Best solution source: %s", best_source)
    logging.info("Total oracle calls: %d", oracle_calls)

    return {
        "config": {
            "n_regions": n_regions,
            "pca_components": pca_dim,
            "balanced_method": balanced_method,
            "region_top_k": region_top_k_eff,
            "round2_top_k": round2_top_k_eff,
            "inner_top_k_frac": inner_top_k_frac,
            "inner_prediction_top_k": inner_prediction_top_k,
            "inner_dopt_feature_space": "region_local_standardized_pca",
            "fw_tol": fw_tol,
            "fw_step_scheme": fw_step_scheme,
            "fw_epsilon": fw_epsilon,
            "random_state": random_state,
        },
        "clustering": {
            "labels": labels,
            "region_sizes": region_sizes,
            "region_indices": region_indices,
        },
        "region_features": {
            "X_region_full": X_region_full,
            "X_region": X_region,
            "kept_columns": kept_columns,
        },
        "round1": {
            "weights": w_region,
            "fw_history": fw_history_region,
            "selected_regions": round1_regions,
            "region_best_fitness": region_best_fitness_r1,
            "region_best_solution": region_best_solution_r1,
            "region_best_source": region_best_source_r1,
            "evaluated_indices": evaluated_r1,
            "evaluated_per_region": evaluated_per_region_r1,
            "surrogate_evaluated_per_region": surrogate_evaluated_per_region_r1,
        },
        "round2": {
            "selected_regions": round2_regions,
            "surrogate_selected_regions": round2_predicted_regions,
            "predicted_region_fitness": {
                int(r): float(y_region_pred_all[r]) for r in range(n_regions)
            },
            "region_best_fitness": region_best_fitness_r2,
            "region_best_solution": region_best_solution_r2,
            "region_best_source": region_best_source_r2,
            "evaluated_indices": evaluated_r2,
            "evaluated_per_region": evaluated_per_region_r2,
            "surrogate_evaluated_per_region": surrogate_evaluated_per_region_r2,
        },
        "surrogate": {
            "coef": getattr(surrogate, "coef_", None),
            "intercept": getattr(surrogate, "intercept_", None),
            "model_type": type(surrogate).__name__,
        },
        "summary": {
            "best_fitness": best_fitness,
            "best_solution_index": best_global_idx,
            "best_solution_key": best_key,
            "best_solution_source": best_source,
            "oracle_calls": oracle_calls,
            "all_evaluated_indices": all_evaluated,
            "all_evaluated_keys": all_evaluated_keys,
            "region_best_fitness_all": region_best_fitness_all,
            "region_best_solution_all": region_best_solution_all,
            "region_best_source_all": region_best_source_all,
            "coverage": coverage_results,
        },
        "candidate_keys": candidate_keys,
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-level DOPP baseline (balanced regions + two-round D-opt)."
    )
    parser.add_argument(
        "features_file",
        type=Path,
        help="Path to standardized solution-level feature bundle .npy file",
    )
    parser.add_argument(
        "fitness_csv",
        type=Path,
        help="Path to metrics.csv (must contain Key, Cut_size, Area_imbalance, "
        "and either Fitness or columns used by --metrics).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Metrics to compute Fitness from (forwarded to "
        "load_fitness_scores_from_csv). If omitted, the CSV's existing Fitness "
        "column is used.",
    )
    parser.add_argument("--n-regions", type=int, default=100)
    parser.add_argument(
        "--balanced-method",
        type=str,
        default="constrained",
        choices=["constrained", "reassign"],
    )
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--region-top-k", type=int, default=10)
    parser.add_argument("--round2-top-k", type=int, default=10)
    parser.add_argument("--inner-top-k-frac", type=float, default=0.2)
    parser.add_argument(
        "--inner-prediction-top-k",
        type=int,
        default=1,
        help=(
            "Per selected region, take this many candidate solutions with the "
            "best local surrogate predictions. Candidates already selected by "
            "inner D-opt may appear here and are de-duplicated before oracle "
            "evaluation."
        ),
    )
    parser.add_argument("--fw-tol", type=float, default=1e-3)
    parser.add_argument(
        "--fw-step-scheme",
        type=str,
        default="1/t",
        choices=["1/t", "line_search"],
    )
    parser.add_argument("--fw-epsilon", type=float, default=1e-8)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: <features_file dir>/two_level/",
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
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.features_file.exists():
        raise FileNotFoundError(f"Features file not found: {args.features_file}")
    if not args.fitness_csv.exists():
        raise FileNotFoundError(f"Fitness CSV file not found: {args.fitness_csv}")

    out_dir = args.output if args.output is not None else (args.features_file.parent / "two_level")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading solution-level features from %s", args.features_file)
    X, candidate_keys, feature_metadata = load_features_from_file(
        args.features_file, fitness_csv=args.fitness_csv
    )
    X = np.asarray(X, dtype=np.float64)
    logging.info("Feature matrix shape: %s", X.shape)

    logging.info("Loading fitness from %s", args.fitness_csv)
    fitness_dict = load_fitness_scores_from_csv(args.fitness_csv, metrics=args.metrics)
    y = align_fitness(fitness_dict, candidate_keys).astype(np.float64)

    logging.info("Loading proxies (Cut_size, Area_imbalance) from %s", args.fitness_csv)
    proxies = load_proxies_from_csv(args.fitness_csv, candidate_keys).astype(np.float64)

    results = run_two_level_dopp(
        X=X,
        y=y,
        proxies=proxies,
        candidate_keys=list(candidate_keys),
        n_regions=args.n_regions,
        pca_components=args.pca_components,
        balanced_method=args.balanced_method,
        region_top_k=args.region_top_k,
        round2_top_k=args.round2_top_k,
        inner_top_k_frac=args.inner_top_k_frac,
        inner_prediction_top_k=args.inner_prediction_top_k,
        fw_tol=args.fw_tol,
        fw_step_scheme=args.fw_step_scheme,
        fw_epsilon=args.fw_epsilon,
        random_state=args.random_state,
    )

    results["feature_metadata"] = feature_metadata

    output_path = out_dir / "two_level_results.npy"
    np.save(output_path, results, allow_pickle=True)
    logging.info("Saved two-level DOPP results to %s", output_path)

    # Brief human-readable summary
    summary = results["summary"]
    logging.info("=" * 60)
    logging.info(
        "Best key: %s (fitness=%.4f, source=%s)",
        summary["best_solution_key"],
        summary["best_fitness"],
        summary["best_solution_source"],
    )
    logging.info("Oracle calls: %d", summary["oracle_calls"])
    for k_label, payload in summary["coverage"].items():
        logging.info(
            "Coverage %s: %d / %d", k_label, payload["hits"], payload["k"]
        )


if __name__ == "__main__":
    main()
