"""Lightweight feed-forward neural network (MLP) candidate model.

A small fully-connected network with two hidden layers and tanh activations,
trained with L2-regularized binary cross-entropy via deterministic mini-batch
gradient descent with heavy-ball momentum, early stopping on a chronological
validation slice, and Glorot-style weight initialization. It is deliberately
compact (two small hidden layers) so it generalizes on the ~1-2.5k game
training windows this pipeline sees; capacity is traded for regularization,
and the final probability is always passed through the pipeline's isotonic
calibration.

Design goals, matching the rest of the engine:

  * Deterministic — fixed-seed init and per-epoch batch shuffling, so two
    identical training runs produce byte-identical weights (the pipeline's
    determinism tests depend on this).
  * Standardized features — same z-score transform as the logistic path.
  * numpy-accelerated with a pure-Python fallback — identical math on both
    paths; numpy only makes the matmuls fast.
  * Same `predict(features) -> probability` closure shape as every other
    candidate, so the Auto-ML pool treats it identically (AUC >= 0.51
    eligibility floor, stacking, calibration).

Why a neural network here: on tabular win-probability data the MLP can model
non-linear feature interactions (e.g. pitcher-FIP edge only matters when
lineups are known) that linear logistic regression cannot. It rarely wins on
its own — logistic/Elo are strong baselines — but as an additional member of
the candidate pool and the greedy stacked ensemble it consistently adds a few
basis points of AUC and shaves Brier when it clears the floor.
"""

from __future__ import annotations

import math
import random

from .logistic import zscore
from .metrics import mean, sigmoid, std

try:  # numpy is optional; only used as an accelerator
    import numpy as _np
except Exception:  # pragma: no cover - fallback path
    _np = None


def _tanh(x: float) -> float:
    if x >= 20:
        return 1.0
    if x <= -20:
        return -1.0
    e2 = math.exp(2 * x)
    return (e2 - 1) / (e2 + 1)


def _fit_numpy(
    X: list[list[float]],
    y: list[int],
    hidden: tuple[int, int],
    epochs: int,
    batch: int,
    lr: float,
    l2: float,
    momentum: float,
    val_frac: float,
    patience: int,
    rng: random.Random,
) -> dict:
    np = _np
    n = len(X)
    Xa = np.array(X, dtype=float)
    ya = np.array(y, dtype=float)
    val_n = max(1, int(round(n * val_frac)))
    tr = Xa[: n - val_n]
    tr_y = ya[: n - val_n]
    va = Xa[n - val_n:]
    va_y = ya[n - val_n:]
    d0 = tr.shape[1]
    h1n, h2n = hidden

    def init_w(rows: int, cols: int) -> np.ndarray:
        bound = math.sqrt(6.0 / (rows + cols))
        return np.asarray([[rng.uniform(-bound, bound) for _ in range(cols)] for _ in range(rows)])

    W1 = init_w(h1n, d0)
    W2 = init_w(h2n, h1n)
    W3 = init_w(1, h2n)
    b1 = np.zeros(h1n)
    b2 = np.zeros(h2n)
    b3 = np.zeros(1)

    vW1 = np.zeros_like(W1)
    vW2 = np.zeros_like(W2)
    vW3 = np.zeros_like(W3)
    vb1 = np.zeros_like(b1)
    vb2 = np.zeros_like(b2)
    vb3 = np.zeros_like(b3)

    best = None
    best_val = float("inf")
    stagnant = 0
    m = len(tr)

    def forward(Xm: np.ndarray):
        a1 = Xm @ W1.T + b1
        h1 = np.tanh(a1)
        a2 = h1 @ W2.T + b2
        h2 = np.tanh(a2)
        z = h2 @ W3.T + b3
        return h1, h2, z

    def _stable_sigmoid(z):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    def val_loss() -> float:
        _, _, z = forward(va)
        p = np.clip(_stable_sigmoid(z), 1e-7, 1 - 1e-7)
        loss = -np.mean(va_y * np.log(p) + (1 - va_y) * np.log(1 - p))
        reg = l2 * 0.5 * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3)) / m
        return float(loss + reg)

    idx = list(range(m))
    max_norm = 5.0
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, m, batch):
            b = idx[s:s + batch]
            Xb = tr[b]
            yb = tr_y[b]
            h1, h2, z = forward(Xb)
            p = np.clip(_stable_sigmoid(z), 1e-7, 1 - 1e-7)
            dz = (p - yb.reshape(-1, 1)) / len(b)  # mean over the batch
            gW3 = dz.T @ h2 + (l2 / m) * W3
            gb3 = dz.sum(axis=0)
            dh2 = dz @ W3
            da2 = dh2 * (1 - h2 * h2)
            gW2 = da2.T @ h1 + (l2 / m) * W2
            gb2 = da2.sum(axis=0)
            dh1 = da2 @ W2
            da1 = dh1 * (1 - h1 * h1)
            gW1 = da1.T @ Xb + (l2 / m) * W1
            gb1 = da1.sum(axis=0)
            for g in (gW1, gW2, gW3, gb1, gb2, gb3):
                norm = float(np.sqrt(np.sum(g * g)))
                g *= min(1.0, max_norm / norm) if norm > max_norm else 1.0
            vW1 = momentum * vW1 - lr * gW1
            vW2 = momentum * vW2 - lr * gW2
            vW3 = momentum * vW3 - lr * gW3
            vb1 = momentum * vb1 - lr * gb1
            vb2 = momentum * vb2 - lr * gb2
            vb3 = momentum * vb3 - lr * gb3
            W1 += vW1
            W2 += vW2
            W3 += vW3
            b1 += vb1
            b2 += vb2
            b3 += vb3
        vl = val_loss()
        if vl < best_val - 1e-6:
            best_val = vl
            stagnant = 0
            best = (W1.copy(), W2.copy(), W3.copy(), b1.copy(), b2.copy(), b3.copy())
        else:
            stagnant += 1
            if stagnant >= patience:
                break
    if best is not None:
        W1, W2, W3, b1, b2, b3 = best
    return {
        "W1": W1.tolist(), "W2": W2.tolist(), "W3": W3.tolist()[0],
        "b1": b1.tolist(), "b2": b2.tolist(), "b3": float(b3[0]),
    }


def _mm(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Dense matmul (k×n @ n×m) with zip-based dot products."""
    BT = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in BT] for row in A]


def _add_row(A: list[list[float]], b: list[float]) -> list[list[float]]:
    return [[v + bias for v, bias in zip(row, b)] for row in A]


def _apply_row(A: list[list[float]], fn) -> list[list[float]]:
    return [[fn(v) for v in row] for row in A]


def _row_sums(A: list[list[float]]) -> list[float]:
    return [sum(row) for row in A]


def _fit_pure(
    X: list[list[float]],
    y: list[int],
    hidden: tuple[int, int],
    epochs: int,
    batch: int,
    lr: float,
    l2: float,
    momentum: float,
    val_frac: float,
    patience: int,
    rng: random.Random,
) -> dict:
    """Pure-Python twin of _fit_numpy: same batches, same math, batch matmuls
    via list comprehensions. Slower than numpy but correct and deterministic."""
    n = len(X)
    val_n = max(1, int(round(n * val_frac)))
    tr = X[: n - val_n]
    tr_y = y[: n - val_n]
    va = X[n - val_n:]
    va_y = y[n - val_n:]
    d0 = len(tr[0])
    h1n, h2n = hidden

    def init_w(rows: int, cols: int) -> list[list[float]]:
        bound = math.sqrt(6.0 / (rows + cols))
        return [[rng.uniform(-bound, bound) for _ in range(cols)] for _ in range(rows)]

    W1 = init_w(h1n, d0)
    W2 = init_w(h2n, h1n)
    W3 = init_w(1, h2n)
    b1 = [0.0] * h1n
    b2 = [0.0] * h2n
    b3 = [0.0]
    vW1 = [[0.0] * d0 for _ in range(h1n)]
    vW2 = [[0.0] * h1n for _ in range(h2n)]
    vW3 = [[0.0] * h2n]
    vb1 = [0.0] * h1n
    vb2 = [0.0] * h2n
    vb3 = [0.0]
    W1T = list(zip(*W1))
    W2T = list(zip(*W2))
    W3T = list(zip(*W3))

    m = len(tr)
    best = None
    best_val = float("inf")
    stagnant = 0
    max_norm = 5.0
    lr2 = l2 / m

    def forward(Xb: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[float]]:
        h1 = _apply_row(_add_row(_mm(Xb, W1T), b1), _tanh)
        h2 = _apply_row(_add_row(_mm(h1, W2T), b2), _tanh)
        z = [row[0] for row in _add_row(_mm(h2, W3T), b3)]
        return h1, h2, z

    def val_loss() -> float:
        _, _, z = forward(va)
        loss = 0.0
        for i, zv in enumerate(z):
            p = min(1 - 1e-7, max(1e-7, sigmoid(zv)))
            loss += -(va_y[i] * math.log(p) + (1 - va_y[i]) * math.log(1 - p))
        reg = 0.0
        for row in W1:
            for v in row:
                reg += v * v
        for row in W2:
            for v in row:
                reg += v * v
        for v in W3[0]:
            reg += v * v
        return loss / len(va) + l2 * 0.5 * reg / m

    for _ in range(epochs):
        order = list(range(m))
        rng.shuffle(order)
        for s in range(0, m, batch):
            b = order[s:s + batch]
            n_b = len(b)
            Xb = [tr[i] for i in b]
            yb = [tr_y[i] for i in b]
            h1, h2, z = forward(Xb)
            # Backward (mean-scaled over the batch, matching numpy).
            dz = [[(sigmoid(zv) - yv) / n_b] for zv, yv in zip(z, yb)]
            gW3 = [[sum(dzi[0] * h2i[k] for dzi, h2i in zip(dz, h2)) for k in range(h2n)]]
            gb3 = [sum(dzi[0] for dzi in dz)]
            dh2 = [[dzi[0] * W3[0][j] * (1 - h2i[j] * h2i[j]) for j in range(h2n)] for dzi, h2i in zip(dz, h2)]
            da2 = dh2
            gW2 = [[sum(da2i[j] * h1i[k] for da2i, h1i in zip(da2, h1)) for k in range(h1n)] for j in range(h2n)]
            gb2 = [sum(row[j] for row in da2) for j in range(h2n)]
            dh1 = _mm(da2, W2)
            da1 = [[dh1i[j] * (1 - h1i[j] * h1i[j]) for j in range(h1n)] for dh1i, h1i in zip(dh1, h1)]
            gW1 = [[sum(da1i[j] * Xb_i[k] for da1i, Xb_i in zip(da1, Xb)) for k in range(d0)] for j in range(h1n)]
            gb1 = [sum(row[j] for row in da1) for j in range(h1n)]
            for j in range(h1n):
                for k in range(d0):
                    gW1[j][k] += lr2 * W1[j][k]
            for j in range(h2n):
                for k in range(h1n):
                    gW2[j][k] += lr2 * W2[j][k]
            for k in range(h2n):
                gW3[0][k] += lr2 * W3[0][k]
            # Gradient clipping + momentum step. Gradients are mean-scaled
            # over the batch (dz includes 1/n_b), so the step is exactly
            # v = momentum*v - lr*g — the same update as the numpy path.
            for g in (gW1, gW2, gW3, [gb1], [gb2], [gb3]):
                norm = 0.0
                for row in g:
                    for v in row:
                        norm += v * v
                norm = math.sqrt(norm)
                scale = min(1.0, max_norm / norm) if norm > max_norm else 1.0
                for row in g:
                    for k in range(len(row)):
                        row[k] *= scale
            for j in range(h1n):
                for k in range(d0):
                    vW1[j][k] = momentum * vW1[j][k] - lr * gW1[j][k]
                    W1[j][k] += vW1[j][k]
            for j in range(h2n):
                for k in range(h1n):
                    vW2[j][k] = momentum * vW2[j][k] - lr * gW2[j][k]
                    W2[j][k] += vW2[j][k]
            for k in range(h2n):
                vW3[0][k] = momentum * vW3[0][k] - lr * gW3[0][k]
                W3[0][k] += vW3[0][k]
            for j in range(h1n):
                vb1[j] = momentum * vb1[j] - lr * gb1[j]
                b1[j] += vb1[j]
            for j in range(h2n):
                vb2[j] = momentum * vb2[j] - lr * gb2[j]
                b2[j] += vb2[j]
            vb3[0] = momentum * vb3[0] - lr * gb3[0]
            b3[0] += vb3[0]
        vl = val_loss()
        if vl < best_val - 1e-6:
            best_val = vl
            stagnant = 0
            best = (
                [row[:] for row in W1], [row[:] for row in W2], [W3[0][:]],
                b1[:], b2[:], [b3[0]],
            )
        else:
            stagnant += 1
            if stagnant >= patience:
                break
    if best is not None:
        W1, W2, W3, b1, b2, b3 = best
    return {"W1": W1, "W2": W2, "W3": W3[0], "b1": b1, "b2": b2, "b3": b3[0]}


def _predict_pure(params: dict, x: list[float], h1n: int, h2n: int) -> float:
    h1 = [_tanh(params["b1"][j] + sum(a * b for a, b in zip(params["W1"][j], x))) for j in range(h1n)]
    h2 = [_tanh(params["b2"][j] + sum(a * b for a, b in zip(params["W2"][j], h1))) for j in range(h2n)]
    z = params["b3"] + sum(a * b for a, b in zip(params["W3"], h2))
    p = sigmoid(z)
    return min(0.9999, max(0.0001, p))


def _predict_numpy(params: dict, x: list[float], h1n: int, h2n: int) -> float:
    np = _np
    xa = np.asarray(x, dtype=float)
    h1 = np.tanh(np.asarray(params["W1"]) @ xa + np.asarray(params["b1"]))
    h2 = np.tanh(np.asarray(params["W2"]) @ h1 + np.asarray(params["b2"]))
    z = float(np.asarray(params["W3"]) @ h2 + params["b3"])
    p = sigmoid(z)
    return min(0.9999, max(0.0001, p))


def mlp_params(
    train: list[dict],
    feature_names: list[str],
    hidden: tuple[int, int] = (20, 10),
    epochs: int = 40,
    batch: int = 128,
    lr: float = 0.03,
    l2: float = 1e-4,
    momentum: float = 0.9,
    val_frac: float = 0.15,
    patience: int = 6,
    seed: int = 2026,
) -> dict:
    """Serializable parameters for the two-hidden-layer MLP model.

    The standardized-feature stats + weight matrices are JSON-serializable, so
    the model can be persisted and served later (a deployable ensemble member).
    """
    n = len(train)
    d = len(feature_names)
    if n < 40 or d == 0:
        prior = mean([r["label"] for r in train]) if train else 0.5
        return {"prior": min(0.9999, max(0.0001, prior))}
    labels = [r["label"] for r in train]
    pos = sum(labels)
    if pos == 0 or pos == n:  # single-class target — predict the prior
        prior = pos / n
        return {"prior": min(0.9999, max(0.0001, prior))}

    stats = {}
    for f in feature_names:
        vals = [r["features"][f] for r in train]
        stats[f] = {"mean": mean(vals), "std": std(vals) or 1}
    # Use the same train-only winsorized z-score contract as logistic and the
    # tree ensembles (RF/XGB/LGBM). (PRUNED families — kNN, naive Bayes, and
    # boosted stumps — are no longer in the candidate pool.) This prevents a
    # malformed live weather or workload value from reaching the MLP on a
    # different scale.
    X = [[zscore(r["features"][f], stats[f]["mean"], stats[f]["std"]) for f in feature_names] for r in train]

    rng = random.Random(seed)
    h1n, h2n = hidden
    if _np is not None:
        params = _fit_numpy(X, labels, hidden, epochs, batch, lr, l2, momentum, val_frac, patience, rng)
    else:
        params = _fit_pure(X, labels, hidden, epochs, batch, lr, l2, momentum, val_frac, patience, rng)
    return {
        "stats": stats,
        "params": params,
        "hidden": [h1n, h2n],
        "featureNames": list(feature_names),
    }


def mlp_predict(member: dict, features: dict) -> float:
    """Predict from serialized mlp_params (identical math to the closure
    returned by mlp_model)."""
    if "prior" in member:
        return member["prior"]
    feature_names = member["featureNames"]
    stats = member["stats"]
    h1n, h2n = member["hidden"]
    z = [zscore(features[f], stats[f]["mean"], stats[f]["std"]) for f in feature_names]
    if _np is not None:
        return _predict_numpy(member["params"], z, h1n, h2n)
    return _predict_pure(member["params"], z, h1n, h2n)


def mlp_model(
    train: list[dict],
    feature_names: list[str],
    hidden: tuple[int, int] = (20, 10),
    epochs: int = 40,
    batch: int = 128,
    lr: float = 0.03,
    l2: float = 1e-4,
    momentum: float = 0.9,
    val_frac: float = 0.15,
    patience: int = 6,
    seed: int = 2026,
):
    """Two-hidden-layer MLP trained with L2 + early stopping (deterministic).

    Returns a `predict(features) -> probability` closure matching the other
    candidate models. With fewer than ~40 training rows (or a single-class
    target) it falls back to the empirical prior, so the pool never sees a
    degenerate fit. Hyperparameters are conservative on purpose: the pipeline
    already stacks + isotonically calibrates, so the MLP only needs to add a
    well-regularized, non-linear view of the features.
    """
    member = mlp_params(
        train, feature_names, hidden, epochs, batch, lr, l2, momentum, val_frac, patience, seed
    )
    return lambda features: mlp_predict(member, features)
