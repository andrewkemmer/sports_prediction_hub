"""Logistic regression (L2, IRLS), kNN, Naive Bayes, stacking, CV.

Faithful port of the candidate-model section of src/convex/ml/model.ts.
"""

from __future__ import annotations

import math

from .metrics import (
    clamp,
    compute_auc,
    compute_brier,
    mean,
    parallel_map,
    roundn,
    sigmoid,
    std,
)

try:  # numpy is optional; only used as an accelerator for the IRLS fits
    import numpy as _np
except Exception:  # pragma: no cover - fallback path
    _np = None


def _np_sigmoid(x):
    """Numerically stable sigmoid matching engine.metrics.sigmoid, vectorized."""
    z = _np.exp(-_np.abs(x))
    return _np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def solve_linear_system(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve a small dense linear system A·x = b with partial pivoting."""
    n = len(A)
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(aug[r][col]) > abs(aug[pivot][col]):
                pivot = r
        if abs(aug[pivot][col]) < 1e-12:
            continue
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        for c in range(col, n + 1):
            aug[col][c] /= pv
        for r in range(n):
            if r == col:
                continue
            f = aug[r][col]
            if f == 0:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= f * aug[col][c]
    return [row[n] for row in aug]


def _train_logistic_vectorized(rows: list[dict], feature_names: list[str], iterations: int, w0: list[float] | None = None, lambda_: float = 0.001) -> dict | None:
    """numpy IRLS — identical math to the pure-Python path below, ~50-100x faster.

    Returns None (so the caller falls back to the reference implementation) when
    the linear solve fails, e.g. on a singular system. `w0` seeds the iteration
    with a previous fit's weights (same standardized feature space, intercept
    last) so drop-one refits converge in a couple of iterations.
    """
    np = _np
    n = len(rows)
    m = len(feature_names)
    feature_stats = {}
    for f in feature_names:
        vals = np.array([r["features"][f] for r in rows], dtype=float)
        s = float(vals.std())
        if not s or not math.isfinite(s):
            s = 1.0
        feature_stats[f] = {"mean": float(vals.mean()), "std": s}
    means = np.array([feature_stats[f]["mean"] for f in feature_names])
    stds = np.array([feature_stats[f]["std"] for f in feature_names])
    X = np.array([[r["features"][f] for f in feature_names] for r in rows], dtype=float)
    X = (X - means) / stds
    Xaug = np.column_stack([X, np.ones(n)])
    y = np.array([r["label"] for r in rows], dtype=float)
    d = m + 1  # feature columns + intercept column
    pos = float(y.sum())
    if w0 is not None and len(w0) == d:
        w = np.array(w0, dtype=float)
    else:
        w = np.zeros(d)
        w[m] = math.log((pos + 1) / (n - pos + 1))

    for _ in range(iterations):
        eta = Xaug @ w
        p = np.clip(_np_sigmoid(eta), 1e-6, 1 - 1e-6)
        weight = np.maximum(p * (1 - p), 1e-6)
        z = eta + (y - p) / weight
        wz = weight * z
        rhs = Xaug.T @ wz
        A = Xaug.T @ (Xaug * weight[:, None])
        if m > 0:
            A[np.arange(m), np.arange(m)] += lambda_  # ridge the features, not the intercept
        try:
            nxt = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(nxt)):
            break
        if float(np.max(np.abs(nxt - w))) < 1e-9:  # converged — stop exactly at the fixed point
            w = nxt
            break
        w = nxt

    return {
        "featureNames": feature_names,
        "weights": [float(v) for v in w[:m]],
        "bias": float(w[m]),
        "featureStats": feature_stats,
    }


def train_logistic(rows: list[dict], feature_names: list[str], iterations: int = 20, w0: list[float] | None = None, lambda_: float = 0.001) -> dict:
    """Newton-Raphson / IRLS ridge logistic regression (standardized features).

    `w0` seeds the iteration with a previous fit's weights (same standardized
    feature space, intercept last); used by feature selection so each drop-one
    refit starts near the optimum and converges in a couple of iterations.
    `lambda_` is the L2 ridge strength — the candidate pool trains several
    strengths (0.001 / 0.1 / 1.0) so the selector can trade bias for variance.
    Convergence is detected exactly (max weight change < 1e-9), so fewer
    iterations never change the fitted model.
    """
    if _np is not None:
        try:
            m = _train_logistic_vectorized(rows, feature_names, iterations, w0, lambda_)
            if m is not None:
                return m
        except Exception:  # pragma: no cover - fall back to the reference on any edge case
            pass

    feature_stats = {}
    for f in feature_names:
        vals = [r["features"][f] for r in rows]
        s = std(vals) or 1
        feature_stats[f] = {"mean": mean(vals), "std": s}
    X = [
        [(r["features"][f] - feature_stats[f]["mean"]) / feature_stats[f]["std"] for f in feature_names]
        for r in rows
    ]
    y = [r["label"] for r in rows]
    n = len(rows)
    m = len(feature_names)
    pos = sum(y)
    d = m + 1  # feature columns + intercept column
    Xaug = [xi + [1.0] for xi in X]
    if w0 is not None and len(w0) == d:
        w = list(w0)
    else:
        w = [0.0] * d
        w[m] = math.log((pos + 1) / (n - pos + 1))

    for _ in range(iterations):
        A = [[0.0] * d for _ in range(d)]
        rhs = [0.0] * d
        for i in range(n):
            eta = sum(w[j] * Xaug[i][j] for j in range(d))
            p = clamp(sigmoid(eta), 1e-6, 1 - 1e-6)
            weight = max(p * (1 - p), 1e-6)
            z = eta + (y[i] - p) / weight
            wz = weight * z
            xi = Xaug[i]
            for j in range(d):
                rhs[j] += wz * xi[j]
                for k in range(j, d):
                    A[j][k] += weight * xi[j] * xi[k]
        for j in range(d):
            for k in range(j + 1, d):
                A[k][j] = A[j][k]
            if j < m:
                A[j][j] += lambda_  # ridge the features, not the intercept
        nxt = solve_linear_system(A, rhs)
        if any(not math.isfinite(v) for v in nxt):
            break
        if max(abs(a - b) for a, b in zip(nxt, w)) < 1e-9:  # converged — stop exactly
            w = nxt
            break
        w = nxt

    return {
        "featureNames": feature_names,
        "weights": w[:m],
        "bias": w[m],
        "featureStats": feature_stats,
    }


def logistic_logit(model: dict, features: dict, shap: list | None = None) -> float:
    """Raw logit = bias + Σ w·z. When `shap` is a list, appends contributions."""
    logit_v = model["bias"]
    for i, f in enumerate(model["featureNames"]):
        z = (features[f] - model["featureStats"][f]["mean"]) / (model["featureStats"][f]["std"] or 1)
        logit_v += model["weights"][i] * z
        if shap is not None:
            shap.append({
                "feature": f,
                "contribution": model["weights"][i] * z,
            })
    return logit_v


def standardized(features: dict, feature_names: list[str], stats: dict) -> list[float]:
    return [(features[f] - stats[f]["mean"]) / (stats[f]["std"] or 1) for f in feature_names]


def knn_model(train: list[dict], feature_names: list[str], k: int = 21):
    """k-nearest-neighbour classifier on standardized features (majority vote)."""
    stats = {}
    for f in feature_names:
        vals = [r["features"][f] for r in train]
        stats[f] = {"mean": mean(vals), "std": std(vals) or 1}
    z_train = [standardized(r["features"], feature_names, stats) for r in train]
    labels = [r["label"] for r in train]

    def predict(features: dict) -> float:
        z = standardized(features, feature_names, stats)
        dists = sorted(
            (sum((zt[j] - z[j]) ** 2 for j in range(len(z))), labels[i])
            for i, zt in enumerate(z_train)
        )
        nn = dists[:k]
        if not nn:
            return 0.5
        return sum(x[1] for x in nn) / len(nn)

    return predict


def naive_bayes_model(train: list[dict], feature_names: list[str]):
    """Gaussian Naive Bayes classifier with Laplace-smoothed priors."""
    n = len(train)
    pos = [r for r in train if r["label"] == 1]
    neg = [r for r in train if r["label"] == 0]
    prior = (len(pos) + 1) / (n + 2)

    def cond(rows: list[dict]) -> list[dict]:
        out = []
        for f in feature_names:
            vals = [r["features"][f] for r in rows]
            v = std(vals) * std(vals) + 1e-6
            out.append({"m": mean(vals), "v": v})
        return out

    pos_stats = cond(pos)
    neg_stats = cond(neg)

    def gauss_log(x: float, s: dict) -> float:
        return -0.5 * math.log(2 * math.pi * s["v"]) - ((x - s["m"]) * (x - s["m"])) / (2 * s["v"])

    def predict(features: dict) -> float:
        log_pos = math.log(prior)
        log_neg = math.log(1 - prior)
        for j, f in enumerate(feature_names):
            log_pos += gauss_log(features[f], pos_stats[j])
            log_neg += gauss_log(features[f], neg_stats[j])
        max_log = max(log_pos, log_neg)
        p_pos = math.exp(log_pos - max_log)
        p_neg = math.exp(log_neg - max_log)
        s = p_pos + p_neg
        return p_pos / s if s > 0 else prior

    return predict


def build_stacking_weights(cand_preds: dict, labels: list[int]) -> dict:
    """Greedy forward-selection stacking over candidate model predictions."""
    names = list(cand_preds.keys())
    ranked = sorted(((n, compute_brier(cand_preds[n], labels)) for n in names), key=lambda t: t[1])
    order = [r[0] for r in ranked]
    first = cand_preds[order[0]]
    if not first:
        return {
            "preds": [],
            "brier": float("inf"),
            "weights": [{"name": n, "weight": 1 if n == order[0] else 0} for n in names],
        }
    ensemble = list(first)
    weights = {order[0]: 1.0}
    cur_brier = compute_brier(ensemble, labels)
    step = 0.05
    for i in range(1, len(order)):
        name = order[i]
        p = cand_preds[name]
        best_w = 0.0
        best_b = cur_brier
        w = step
        while w <= 1.0001:
            blend = [(1 - w) * e + w * p[j] for j, e in enumerate(ensemble)]
            b = compute_brier(blend, labels)
            if b < best_b:
                best_b = b
                best_w = w
            w += step
        if best_b < cur_brier - 0.0005:
            nxt = {k: v * (1 - best_w) for k, v in weights.items()}
            nxt[name] = best_w
            weights = nxt
            ensemble = [(1 - best_w) * e + best_w * p[j] for j, e in enumerate(ensemble)]
            cur_brier = best_b
    total = sum(weights.values()) or 1
    weight_list = [{"name": n, "weight": roundn((weights.get(n) or 0) / total, 3)} for n in names]
    return {"preds": ensemble, "brier": cur_brier, "weights": weight_list}


def cross_validate(rows: list[dict], feature_names: list[str], cv_folds: int = 5) -> dict:
    """Walk-forward 5-fold CV of the logistic model (no lookahead)."""
    n = len(rows)
    chunk_size = max(1, n // (cv_folds + 1))

    def fit_fold(f: int):
        train_end = f * chunk_size
        test_end = n if f == cv_folds else (f + 1) * chunk_size
        train = rows[:train_end]
        test = rows[train_end:test_end]
        if len(train) < 40 or len(test) < 20:
            return None
        m = train_logistic(train, feature_names)
        preds = [sigmoid(logistic_logit(m, r["features"], None)) for r in test]
        labels = [r["label"] for r in test]
        return {
            "auc": compute_auc(preds, labels),
            "brier": compute_brier(preds, labels),
            "games": len(test),
        }

    # Folds are independent walk-forward fits — run them concurrently (order
    # preserved) when numpy is present (its ops release the GIL; pure-Python
    # fits serialize under the GIL, so a pool would only add overhead).
    fold_aucs: list[float] = []
    fold_briers: list[float] = []
    games_per_fold: list[int] = []
    fold_workers = min(cv_folds, 5) if _np is not None else 1
    for fit in parallel_map(fit_fold, list(range(1, cv_folds + 1)), max_workers=fold_workers):
        if fit is None:
            continue
        fold_aucs.append(fit["auc"])
        fold_briers.append(fit["brier"])
        games_per_fold.append(fit["games"])
    return {
        "folds": len(fold_aucs),
        "aucMean": roundn(mean(fold_aucs), 3) if fold_aucs else 0,
        "aucStd": roundn(std(fold_aucs), 3) if fold_aucs else 0,
        "brierMean": roundn(mean(fold_briers), 3) if fold_briers else 0,
        "brierStd": roundn(std(fold_briers), 3) if fold_briers else 0,
        "foldAucs": [roundn(x, 3) for x in fold_aucs],
        "foldBriers": [roundn(x, 3) for x in fold_briers],
        "gamesPerFold": games_per_fold,
    }
