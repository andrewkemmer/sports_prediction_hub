"""Stronger candidate model families for the Auto-ML pool.

Two additions beyond the original five candidates:

  * Distance-weighted k-NN — the k nearest neighbors vote with weight 1/(d+ε)
    instead of a hard majority, which consistently lifts kNN AUC on dense
    standardized feature spaces (plain majority kNN tends to sit near 0.60).
  * L2-boosted decision stumps — shallow (depth-1) regression trees fit to the
    residuals of the previous ensemble (squared-error boosting, the classic
    tabular-data baseline). Each stump searches a deterministic random subset
    of the features, so training is reproducible with a fixed seed and needs
    only the standard library; numpy (when present) accelerates the split
    search with prefix sums.

Both return a `predict(features) -> probability` closure in the same shape as
the other candidate models, so the pipeline treats them identically.
"""

from __future__ import annotations

import heapq
import math
import random

from .logistic import standardized
from .metrics import mean, sigmoid, std

try:  # numpy is optional; only used as an accelerator for the stump search
    import numpy as _np
except Exception:  # pragma: no cover - fallback path
    _np = None


# ---------------------------------------------------------------------------
# Distance-weighted k-NN
# ---------------------------------------------------------------------------

def weighted_knn_params(train: list[dict], feature_names: list[str], k: int = 21) -> dict:
    """Serializable parameters for the distance-weighted k-NN model.

    The training set is stored as standardized vectors + labels, so the model
    can be persisted to JSON and served later (a deployable ensemble member).
    """
    stats = {}
    for f in feature_names:
        vals = [r["features"][f] for r in train]
        stats[f] = {"mean": mean(vals), "std": std(vals) or 1}
    z_train = [standardized(r["features"], feature_names, stats) for r in train]
    labels = [r["label"] for r in train]
    return {
        "k": k,
        "stats": stats,
        "zTrain": z_train,
        "labels": labels,
        "featureNames": list(feature_names),
    }


def weighted_knn_predict(params: dict, features: dict) -> float:
    """Predict from serialized weighted_knn_params (identical math to the
    closure returned by weighted_knn_model)."""
    k = params["k"]
    stats = params["stats"]
    z_train = params["zTrain"]
    labels = params["labels"]
    feature_names = params["featureNames"]
    d = len(feature_names)
    z = standardized(features, feature_names, stats)
    # heapq.nsmallest is O(n log k) — cheaper than a full sort per row.
    nn = heapq.nsmallest(
        k,
        ((sum((zt[j] - z[j]) ** 2 for j in range(d)), i) for i, zt in enumerate(z_train)),
    )
    total = 0.0
    wsum = 0.0
    for dist, i in nn:
        w = 1.0 / (dist + 1e-6)
        total += w * labels[i]
        wsum += w
    p = total / wsum if wsum > 0 else 0.5
    # A zero-distance neighbor (duplicate feature vector) could otherwise
    # yield an exact 0.0/1.0, which breaks log-loss in candidate scoring.
    return min(0.9999, max(0.0001, p))


def weighted_knn_model(train: list[dict], feature_names: list[str], k: int = 21):
    """k-nearest-neighbour classifier with inverse-distance voting.

    Nearest neighbors contribute weight 1/(distance + ε) instead of an equal
    vote, so closer (more similar) training games dominate — this removes the
    flat-majority degeneracy that keeps plain kNN AUC low.
    """
    params = weighted_knn_params(train, feature_names, k)
    return lambda features: weighted_knn_predict(params, features)


def weighted_knn_calib_preds(
    knn_train: list[dict],
    calib: list[dict],
    feature_names: list[str],
    k: int = 21,
    model=None,
) -> list[float]:
    """Calibration-set predictions for distance-weighted k-NN.

    numpy path is vectorized per calib row (argpartition + 1/d weighting); the
    pure-Python fallback reuses the scalar model (heapq.nsmallest) built once.
    """
    if _np is not None and len(calib) > 0:
        np = _np
        stats = {}
        for f in feature_names:
            vals = [r["features"][f] for r in knn_train]
            stats[f] = {"mean": mean(vals), "std": std(vals) or 1}
        means = np.array([stats[f]["mean"] for f in feature_names])
        stds = np.array([stats[f]["std"] or 1 for f in feature_names])
        z_train = np.array([[r["features"][f] for f in feature_names] for r in knn_train])
        z_train = (z_train - means) / stds
        labels = np.array([r["label"] for r in knn_train], dtype=float)
        out = []
        for r in calib:
            z = np.array([r["features"][f] for f in feature_names], dtype=float)
            z = (z - means) / stds
            dist = ((z_train - z) ** 2).sum(axis=1)
            idx = np.argpartition(dist, min(k - 1, len(dist) - 1))[: min(k, len(dist))]
            w = 1.0 / np.maximum(dist[idx], 1e-6)
            p = float((labels[idx] * w).sum() / w.sum())
            out.append(min(0.9999, max(0.0001, p)))
        return out
    if model is None:
        model = weighted_knn_model(knn_train, feature_names, k)
    return [model(r["features"]) for r in calib]


# ---------------------------------------------------------------------------
# L2-boosted decision stumps
# ---------------------------------------------------------------------------

def _best_stump_pure(feature_values: dict[str, list[float]], residuals: list[float], feats: list[str], min_leaf: int) -> dict:
    """Best depth-1 split minimizing SSE of residuals over `feats` (stdlib).

    Returns a stump dict {feature, threshold, left, right}; when no valid
    split exists (e.g. constant feature), returns a constant stump with
    feature=None so boosting can continue shrinking the residuals.
    """
    n = len(residuals)
    total_r = sum(residuals)
    total_r2 = sum(r * r for r in residuals)
    best: dict | None = None
    best_sse = float("inf")
    for f in feats:
        vals = feature_values[f]
        order = sorted(range(n), key=lambda i: vals[i])
        left_r = 0.0
        left_r2 = 0.0
        for pos, i in enumerate(order):
            left_r += residuals[i]
            left_r2 += residuals[i] * residuals[i]
            left_n = pos + 1
            right_n = n - left_n
            if left_n < min_leaf or right_n < min_leaf:
                continue
            # Thresholds must fall between two *distinct* values.
            if left_n < n and vals[order[left_n]] == vals[i]:
                continue
            right_r = total_r - left_r
            right_r2 = total_r2 - left_r2
            sse = left_r2 - left_r * left_r / left_n + right_r2 - right_r * right_r / right_n
            if sse < best_sse:
                best_sse = sse
                best = {"feature": f, "threshold": vals[i], "left": left_r / left_n, "right": right_r / right_n}
    if best is None:
        return {"feature": None, "threshold": None, "left": total_r / n, "right": total_r / n}
    return best


def _best_stump_numpy(feature_values: dict[str, list[float]], residuals: list[float], feats: list[str], min_leaf: int) -> dict:
    """Same split search as _best_stump_pure, vectorized with prefix sums.

    Candidates are scanned in ascending-value order (identical tie-breaking
    to the pure path: the first minimum SSE wins), so both paths return the
    same stump for the same input.
    """
    np = _np
    n = len(residuals)
    total_r = sum(residuals)
    total_r2 = sum(r * r for r in residuals)
    r_arr = np.array(residuals, dtype=float)
    n_left = np.arange(1, n + 1, dtype=float)
    n_right = n - n_left
    valid = (n_left >= min_leaf) & (n_right >= min_leaf)
    best: dict | None = None
    best_sse = float("inf")
    for f in feats:
        vals = feature_values[f]
        sorted_vals = np.array(vals, dtype=float)
        order = np.argsort(sorted_vals, kind="stable")
        sorted_vals = sorted_vals[order]
        r_order = r_arr[order]
        cum_r = np.cumsum(r_order)
        cum_r2 = np.cumsum(r_order * r_order)
        sse = (
            cum_r2 - cum_r * cum_r / n_left
            + (total_r2 - cum_r2) - (total_r - cum_r) * (total_r - cum_r) / n_right
        )
        sse = np.where(valid, sse, np.inf)
        # Only thresholds strictly between two distinct values are splits.
        if n > 1:
            same_next = sorted_vals[1:] == sorted_vals[:-1]
            sse = np.where(np.append(same_next, False), np.inf, sse)
        idx = int(np.argmin(sse))
        if sse[idx] < best_sse:
            best_sse = float(sse[idx])
            best = {
                "feature": f,
                "threshold": float(sorted_vals[idx]),
                "left": float(cum_r[idx] / n_left[idx]),
                "right": float((total_r - cum_r[idx]) / n_right[idx]),
            }
    if best is None:
        return {"feature": None, "threshold": None, "left": total_r / n, "right": total_r / n}
    return best


def _best_stump(feature_values, residuals, feats, min_leaf):
    if _np is not None:
        return _best_stump_numpy(feature_values, residuals, feats, min_leaf)
    return _best_stump_pure(feature_values, residuals, feats, min_leaf)


def _stump_predict(stump: dict, features: dict) -> float:
    if stump["feature"] is None:
        return stump["left"]
    return stump["left"] if features.get(stump["feature"], 0.0) <= stump["threshold"] else stump["right"]


def boosted_stumps_params(
    train: list[dict],
    feature_names: list[str],
    n_trees: int = 60,
    learning_rate: float = 0.1,
    max_features: int = 5,
    min_leaf: int | None = None,
    seed: int = 7,
) -> dict:
    """Serializable parameters for the L2-boosted decision-stump model.

    The fitted stumps + learning rate are JSON-serializable, so the model can
    be persisted and served later (a deployable ensemble member).
    """
    n = len(train)
    if min_leaf is None:
        min_leaf = max(8, n // 40)
    if n < 2 * min_leaf + 2:
        prior = mean([r["label"] for r in train])
        return {"prior": prior, "learningRate": learning_rate, "trees": []}
    features = [r["features"] for r in train]
    labels = [r["label"] for r in train]
    max_features = min(max_features, len(feature_names)) or 1
    rng = random.Random(seed)
    pred = [0.0] * n
    trees: list[dict] = []
    for _ in range(n_trees):
        feats = rng.sample(feature_names, max_features)
        feature_values = {f: [features[i].get(f, 0.0) for i in range(n)] for f in feats}
        residuals = [labels[i] - pred[i] for i in range(n)]
        stump = _best_stump(feature_values, residuals, feats, min_leaf)
        trees.append(stump)
        if stump["feature"] is not None:
            for i in range(n):
                pred[i] += learning_rate * _stump_predict(stump, features[i])
    return {"learningRate": learning_rate, "trees": trees}


def boosted_stumps_predict(params: dict, features_dict: dict) -> float:
    """Predict from serialized boosted_stumps_params (identical math to the
    closure returned by boosted_stumps_model)."""
    if "prior" in params:
        return params["prior"]
    s = 0.0
    learning_rate = params["learningRate"]
    for stump in params["trees"]:
        s += learning_rate * _stump_predict(stump, features_dict)
    return sigmoid(s)


def boosted_stumps_model(
    train: list[dict],
    feature_names: list[str],
    n_trees: int = 60,
    learning_rate: float = 0.1,
    max_features: int = 5,
    min_leaf: int | None = None,
    seed: int = 7,
):
    """L2-boosted depth-1 decision trees (squared-error boosting on {0,1} labels).

    Every stump is fit to the residuals of the previous ensemble; the final
    raw score is passed through a sigmoid (the pipeline's isotonic calibration
    then maps it to calibrated probabilities). Feature subsets per stump are
    drawn with a fixed-seed RNG, so training is fully deterministic.
    """
    params = boosted_stumps_params(
        train, feature_names, n_trees, learning_rate, max_features, min_leaf, seed
    )
    return lambda features: boosted_stumps_predict(params, features)
