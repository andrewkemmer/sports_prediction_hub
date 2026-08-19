"""Pure-Python evaluation metrics and calibration math.

Faithful port of the metric/calibration sections of src/convex/ml/model.ts
(sigmoid, AUC, Brier, LogLoss, ECE, reliability bins, calibration curves,
isotonic PAV regression, and the Gauss-Hermite Monte Carlo adjustment).

Only the Python standard library is used so the engine runs anywhere.
"""

from __future__ import annotations

import math

try:  # numpy is optional; used only to run the true 10,000-trial Monte Carlo
    import numpy as _np
except Exception:  # pragma: no cover - quadrature fallback below
    _np = None

EPS = 1e-6


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def logit(p: float) -> float:
    q = clamp(p, EPS, 1 - EPS)
    return math.log(q / (1 - q))


def clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def mean(vals: list[float]) -> float:
    if len(vals) == 0:
        return 0.0
    return sum(vals) / len(vals)


def std(vals: list[float]) -> float:
    if len(vals) == 0:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) * (v - m) for v in vals) / len(vals))


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def parallel_map(fn, items, max_workers: int | None = None) -> list:
    """Apply `fn` over items in threads, preserving input order.

    Deterministic by construction (results come back in input order, so
    caller tie-breaking and iteration order never change) and safe when each
    item is independent (no shared mutable state). Falls back to serial for
    empty/tiny inputs. Threads overlap work that releases the GIL (numpy
    matmuls / solves); pure-Python work simply serializes under the GIL,
    which is no slower than a plain loop.
    """
    if not items:
        return []
    count = len(items)
    if max_workers is None:
        max_workers = min(8, count)
    if max_workers <= 1:
        return [fn(it) for it in items]
    from concurrent.futures import ThreadPoolExecutor

    results = [None] * count
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fn, it) for it in items]
        for i, fut in enumerate(futures):
            results[i] = fut.result()
    return results


def roundn(n: float, digits: int) -> float:
    f = 10.0 ** digits
    return round(n * f) / f


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_auc(preds: list[float], labels: list[int]) -> float:
    """Area under the ROC curve (Mann-Whitney U with tie handling)."""
    pairs = sorted(zip(preds, labels), key=lambda t: t[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - (n_pos * (n_pos + 1)) / 2) / (n_pos * n_neg)


def compute_brier(preds: list[float], labels: list[int]) -> float:
    return mean([(p - y) * (p - y) for p, y in zip(preds, labels)])


def compute_log_loss(preds: list[float], labels: list[int]) -> float:
    total = 0.0
    for p, y in zip(preds, labels):
        total += -(y * math.log(clamp(p, EPS, 1)) + (1 - y) * math.log(clamp(1 - p, EPS, 1)))
    return total / len(preds)


def spearman_rank(preds: list[float], labels: list[int]) -> float:
    """Spearman rank correlation between predictions and binary outcomes."""
    n = len(preds)
    if n < 2:
        return 0.0

    def rank(vals: list[float]) -> list[float]:
        idx = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[idx[j + 1]] == vals[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[idx[k]] = avg
            i = j + 1
        return out

    rx = rank(preds)
    ry = rank(labels)
    center = (n + 1) / 2
    num = 0.0
    dx2 = 0.0
    dy2 = 0.0
    for i in range(n):
        dx = rx[i] - center
        dy = ry[i] - center
        num += dx * dy
        dx2 += dx * dx
        dy2 += dy * dy
    if dx2 == 0 or dy2 == 0:
        return 0.0
    return num / math.sqrt(dx2 * dy2)


def confidence_bins(preds: list[float], labels: list[int]) -> list[dict]:
    """Reliability bins over confidence (max(p, 1-p)) in 5% bands 50-100%."""
    bins = [{"sumP": 0.0, "sumY": 0.0, "count": 0} for _ in range(10)]
    for i, p in enumerate(preds):
        conf = max(p, 1 - p)
        correct = 1 if (labels[i] if p >= 0.5 else 1 - labels[i]) == 1 else 0
        idx = clamp(int(math.floor((conf - 0.5) / 0.05)), 0, 9)
        bins[idx]["sumP"] += conf
        bins[idx]["sumY"] += correct
        bins[idx]["count"] += 1
    out: list[dict] = []
    for b, cell in enumerate(bins):
        if cell["count"] == 0:
            continue
        mean_predicted = cell["sumP"] / cell["count"]
        mean_actual = cell["sumY"] / cell["count"]
        out.append({
            "label": "95-100%" if b == 9 else f"{50 + b * 5}-{55 + b * 5}%",
            "meanPredicted": mean_predicted,
            "meanActual": mean_actual,
            "count": cell["count"],
            "gap": mean_actual - mean_predicted,
        })
    return out


def calibration_curve_points(preds: list[float], labels: list[int], min_count: int = 12) -> list[dict]:
    """Predicted-probability bins (5% wide) with mean predicted vs actual."""
    bins = [{"sumP": 0.0, "sumY": 0.0, "count": 0} for _ in range(20)]
    for i, p in enumerate(preds):
        idx = clamp(int(math.floor(p / 0.05)), 0, 19)
        bins[idx]["sumP"] += p
        bins[idx]["sumY"] += labels[i]
        bins[idx]["count"] += 1
    out: list[dict] = []
    for cell in bins:
        if cell["count"] < min_count:
            continue
        out.append({
            "x": cell["sumP"] / cell["count"],
            "y": cell["sumY"] / cell["count"],
            "n": cell["count"],
        })
    return out


def evaluate(preds: list[float], labels: list[int]) -> dict:
    """Full evaluation: AUC, Brier, LogLoss, ECE + reliability views."""
    bins = confidence_bins(preds, labels)
    total = len(preds)
    ece = 0.0
    for b in bins:
        ece += (b["count"] / total) * abs(b["gap"])
    distribution = [{"label": b["label"], "count": b["count"], "accuracy": b["meanActual"]} for b in bins]
    return {
        "auc": compute_auc(preds, labels),
        "brier": compute_brier(preds, labels),
        "logLoss": compute_log_loss(preds, labels),
        "ece": ece,
        "bins": bins,
        "confidenceDistribution": distribution,
        "calibrationCurve": calibration_curve_points(preds, labels),
    }


# ---------------------------------------------------------------------------
# Isotonic calibration (PAV)
# ---------------------------------------------------------------------------

def isotonic_regression(xs: list[float], ys: list[float]) -> list[dict]:
    """Pool-adjacent-violators isotonic regression -> monotone step points."""
    blocks: list[dict] = []
    for x, y in zip(xs, ys):
        blocks.append({"xSum": x, "ySum": y, "count": 1})
        while len(blocks) > 1:
            a = blocks[-2]
            b = blocks[-1]
            if a["ySum"] / a["count"] <= b["ySum"] / b["count"]:
                break
            a["xSum"] += b["xSum"]
            a["ySum"] += b["ySum"]
            a["count"] += b["count"]
            blocks.pop()
    return [{"x": b["xSum"] / b["count"], "y": b["ySum"] / b["count"]} for b in blocks]


def apply_isotonic(points: list[dict], p: float) -> float:
    """Piecewise-linear interpolation of the PAV step points."""
    if not points:
        return p
    if p <= points[0]["x"]:
        return points[0]["y"]
    if p >= points[-1]["x"]:
        return points[-1]["y"]
    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        if a["x"] <= p <= b["x"]:
            t = 0.0 if b["x"] == a["x"] else (p - a["x"]) / (b["x"] - a["x"])
            return a["y"] + t * (b["y"] - a["y"])
    return p


# ---------------------------------------------------------------------------
# Monte Carlo (stochastic component)
# ---------------------------------------------------------------------------

# 7-point Gauss-Hermite quadrature (nodes / weights for the first half + zero).
GAUSS_HERMITE_NODES = [0, 0.816287882858965, 1.673551628767471, 2.651961356835233]
GAUSS_HERMITE_WEIGHTS = [0.810264617556807, 0.425607252610128, 0.054515582819125, 0.000971781245099519]
INV_SQRT_PI = 1 / math.sqrt(math.pi)


def monte_carlo_adjust(p: float, sigma: float, trials: int = 10000) -> float:
    """E[sigmoid(logit(p) + sigma*Z)] for Z ~ N(0,1).

    With numpy this is a true `trials`-iteration Monte Carlo simulation
    (deterministic: a fixed seed is used per call, so every game prediction is
    scored with the same 10,000-draw noise realization). Without numpy it falls
    back to 7-point Gauss-Hermite quadrature — the exact analytic expectation
    of the same integral.
    """
    if sigma <= 0 or trials <= 0:
        return p
    lp = logit(p)
    if _np is not None:
        rng = _np.random.default_rng(2026)
        z = rng.standard_normal(trials)
        vals = 1.0 / (1.0 + _np.exp(-(lp + sigma * z)))
        return clamp(float(_np.mean(vals)), 0.001, 0.999)
    s = sigma * math.sqrt(2)
    total = GAUSS_HERMITE_WEIGHTS[0] * sigmoid(lp)
    for i in range(1, len(GAUSS_HERMITE_NODES)):
        node = GAUSS_HERMITE_NODES[i] * s
        total += GAUSS_HERMITE_WEIGHTS[i] * (sigmoid(lp + node) + sigmoid(lp - node))
    return clamp(total * INV_SQRT_PI, 0.001, 0.999)


def american_odds(p: float) -> int:
    q = clamp(p, 0.001, 0.999)
    if q >= 0.5:
        return -int(round((100 * q) / (1 - q)))
    return int(round((100 * (1 - q)) / q))
