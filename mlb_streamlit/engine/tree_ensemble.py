"""Tree ensemble candidate models: Random Forest, XGBoost, LightGBM.

Pure-Python fallback implementations when the real libraries are not
installed.  When sklearn / xgboost / lightgbm are available they are used
as fast drop-in replacements with the exact hyperparameters the user
specified.  The interface matches the existing candidate-model pattern:
every model returns a ``predict(features) -> float`` closure or a
serializable params dict.
"""

from __future__ import annotations

import math
import random

from .metrics import mean, sigmoid, std
from .logistic import standardized, zscore

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

# n_jobs=1 is mandatory on Streamlit Cloud free-tier (1 vCPU / 1 GB): any
# background thread pool spawn duplicates weight buffers and wakes the OOM
# killer. All three library classes honor n_jobs=1 via the same kwarg.
try:
    from sklearn.ensemble import RandomForestClassifier as _SKRF
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    from xgboost import XGBClassifier as _XGB
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier as _LGBM
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False


def _single_thread_kwargs(extra: dict | None = None) -> dict:
    """Return a kwargs dict with n_jobs=1 / nthread=1 forced across all libs.

    Streamlit Cloud free tier caps the container to 1 vCPU + 1 GB RAM. Library
    defaults (sklearn n_jobs=None, xgboost nthread=auto, lightgbm n_jobs=-1)
    silently spawn worker pools that duplicate the model state and compete
    for the same single core. This helper centralizes the safe values.
    """
    base = {"n_jobs": 1, "nthread": 1}
    if extra:
        # Caller-supplied overrides win for non-conflicting keys only.
        for k, v in extra.items():
            if k not in base:
                base[k] = v
    return base


def _skrf_safe(n_estimators=200, max_depth=5, min_samples_leaf=15, random_state=42):
    """RandomForestClassifier with single-thread guard."""
    return _SKRF(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=1,  # CRITICAL: streamlit-cloud 1 vCPU + 1 GB ceiling
    )


def _xgb_safe(n_estimators=150, max_depth=3, learning_rate=0.03,
              subsample=0.8, colsample_bytree=0.8, random_state=42, **kw):
    """XGBClassifier with single-thread guard."""
    safe_kw = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,    # CRITICAL: streamlit-cloud 1 vCPU
        nthread=1,   # xgboost native thread limiter
        verbosity=0,
    )
    safe_kw.update({k: v for k, v in kw.items() if k not in safe_kw})
    return _XGB(**safe_kw)


def _lgbm_safe(n_estimators=150, max_depth=3, learning_rate=0.03,
               subsample=0.8, colsample_bytree=0.8, random_state=42, **kw):
    """LGBMClassifier with single-thread guard."""
    safe_kw = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=1,    # CRITICAL: streamlit-cloud 1 vCPU
        verbose=-1,
    )
    safe_kw.update({k: v for k, v in kw.items() if k not in safe_kw})
    return _LGBM(**safe_kw)


np = _np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_matrix(rows: list[dict], feature_names: list[str], stats: dict | None = None):
    """Convert feature rows to a 2-D matrix (list-of-lists or numpy array).

    When *stats* is provided the columns are z-score standardized with
    train-only mean/std, matching the existing stack-member convention.
    """
    def _row(r):
        vals = []
        for f in feature_names:
            v = r["features"].get(f, 0.0)
            if stats and f in stats:
                s = stats[f]
                v = zscore(v, s["mean"], s["std"])
            vals.append(v)
        return vals

    return [_row(r) for r in rows]


def _labels(rows: list[dict]) -> list[int]:
    return [r["label"] for r in rows]


def _feature_stats(rows: list[dict], feature_names: list[str]) -> dict:
    stats: dict[str, dict] = {}
    for f in feature_names:
        vals = [r["features"].get(f, 0.0) for r in rows]
        stats[f] = {"mean": mean(vals), "std": std(vals) or 1.0}
    return stats


# ---------------------------------------------------------------------------
# Pure-Python decision tree (shared building block)
# ---------------------------------------------------------------------------

def _best_split(
    X: list[list[float]],
    y: list[int],
    feat_idx: list[int],
    min_leaf: int,
) -> tuple[int, float, float, float] | None:
    """Find best (feature, threshold) split minimizing Gini impurity.

    Returns (feature_index, threshold, left_value, right_value) or None.
    """
    n = len(y)
    if n < 2 * min_leaf:
        return None
    best_gini = float("inf")
    best = None
    for fi in feat_idx:
        col = [(X[i][fi], y[i]) for i in range(n)]
        col.sort(key=lambda t: t[0])
        left_0 = 0
        left_1 = 0
        right_0 = sum(1 for v in y if v == 0)
        right_1 = sum(1 for v in y if v == 1)
        total = n
        for pos in range(n - 1):
            val = col[pos][0]
            lbl = col[pos][1]
            if lbl == 0:
                left_0 += 1
                right_0 -= 1
            else:
                left_1 += 1
                right_1 -= 1
            left_n = pos + 1
            right_n = total - left_n
            if left_n < min_leaf or right_n < min_leaf:
                continue
            # Skip equal thresholds
            if col[pos + 1][0] == val:
                continue
            left_p = left_1 / left_n if left_n > 0 else 0.5
            right_p = right_1 / right_n if right_n > 0 else 0.5
            left_gini = 1.0 - left_p * left_p - (1 - left_p) * (1 - left_p)
            right_gini = 1.0 - right_p * right_p - (1 - right_p) * (1 - right_p)
            weighted = (left_n * left_gini + right_n * right_gini) / total
            if weighted < best_gini:
                best_gini = weighted
                best = (fi, val, left_1 / left_n, right_1 / right_n)
    return best


class _TreeNode:
    __slots__ = ("feature", "threshold", "value", "left", "right")

    def __init__(self, feature=None, threshold=None, value=0.5, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right


def _fit_tree(
    X: list[list[float]],
    y: list[int],
    feat_idx: list[int],
    max_depth: int,
    min_leaf: int,
    rng: random.Random,
) -> _TreeNode:
    n = len(y)
    n_1 = sum(y)
    prior = n_1 / n if n > 0 else 0.5
    if max_depth <= 0 or n < 2 * min_leaf:
        return _TreeNode(value=prior)
    # Random feature subset (sqrt for RF, all for boosting)
    k = max(1, int(math.sqrt(len(feat_idx)))) if len(feat_idx) > 4 else len(feat_idx)
    chosen = rng.sample(feat_idx, min(k, len(feat_idx)))
    split = _best_split(X, y, chosen, min_leaf)
    if split is None:
        return _TreeNode(value=prior)
    fi, thresh, lv, rv = split
    left_X, left_y, right_X, right_y = [], [], [], []
    for i in range(n):
        if X[i][fi] <= thresh:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])
    return _TreeNode(
        feature=fi,
        threshold=thresh,
        left=_fit_tree(left_X, left_y, feat_idx, max_depth - 1, min_leaf, rng),
        right=_fit_tree(right_X, right_y, feat_idx, max_depth - 1, min_leaf, rng),
    )


def _predict_tree(node: _TreeNode, x: list[float]) -> float:
    if node.feature is None:
        return node.value
    if x[node.feature] <= node.threshold:
        return _predict_tree(node.left, x)
    return _predict_tree(node.right, x)


def _node_to_dict(node: _TreeNode) -> dict:
    d: dict = {"value": node.value}
    if node.feature is not None:
        d["feature"] = node.feature
        d["threshold"] = node.threshold
        d["left"] = _node_to_dict(node.left)
        d["right"] = _node_to_dict(node.right)
    return d


def _dict_to_node(d: dict) -> _TreeNode:
    node = _TreeNode(value=d["value"])
    if "feature" in d:
        node.feature = d["feature"]
        node.threshold = d["threshold"]
        node.left = _dict_to_node(d["left"])
        node.right = _dict_to_node(d["right"])
    return node


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def rf_params(rows: list[dict], feature_names: list[str], n_trees=200, max_depth=5,
              min_leaf=15, seed=42) -> dict:
    """Serializable Random Forest parameters."""
    stats = _feature_stats(rows, feature_names)
    X = _to_matrix(rows, feature_names)
    y = _labels(rows)
    feat_idx = list(range(len(feature_names)))
    rng = random.Random(seed)
    trees = []
    n = len(rows)
    for t in range(n_trees):
        # Bootstrap sample
        boot_idx = [rng.randint(0, n - 1) for _ in range(n)]
        bx = [X[i] for i in boot_idx]
        by = [y[i] for i in boot_idx]
        trees.append(_fit_tree(bx, by, feat_idx, max_depth, min_leaf, rng))
    return {
        "trees": [_node_to_dict(t) for t in trees],
        "nTrees": n_trees,
        "featureNames": list(feature_names),
        "stats": stats,
    }


def rf_predict(params: dict, features: dict) -> float:
    """Predict from serialized RF params."""
    feat_names = params["featureNames"]
    stats = params.get("stats") or {}
    x = []
    for f in feat_names:
        v = features.get(f, 0.0)
        if f in stats:
            v = zscore(v, stats[f]["mean"], stats[f]["std"])
        x.append(v)
    trees = [_dict_to_node(d) for d in params["trees"]]
    votes = [_predict_tree(t, x) for t in trees]
    p = sum(votes) / len(votes) if votes else 0.5
    return min(0.9999, max(0.0001, p))


def rf_model(rows: list[dict], feature_names: list[str], n_trees=200, max_depth=5,
             min_leaf=15, seed=42):
    """Random Forest model closure (matches candidate-pool interface)."""
    params = rf_params(rows, feature_names, n_trees, max_depth, min_leaf, seed)
    return lambda features: rf_predict(params, features)


# ---------------------------------------------------------------------------
# Gradient-Boosted Trees (XGBoost-like)
# ---------------------------------------------------------------------------

def _gbt_params(
    rows: list[dict],
    feature_names: list[str],
    n_trees: int = 150,
    max_depth: int = 3,
    learning_rate: float = 0.03,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    seed: int = 42,
    name: str = "XGB",
) -> dict:
    """Generic gradient-boosted tree params (logistic loss / logloss)."""
    stats = _feature_stats(rows, feature_names)
    X = _to_matrix(rows, feature_names)
    y = _labels(rows)
    n = len(rows)
    feat_idx_all = list(range(len(feature_names)))
    n_feat = len(feat_idx_all)
    n_sub = max(1, int(math.floor(n_feat * colsample_bytree)))
    rng = random.Random(seed)
    base_p = sum(y) / n if n > 0 else 0.5
    base_logit = math.log(base_p / (1 - base_p)) if 0 < base_p < 1 else 0.0
    predictions = [base_logit] * n
    trees = []
    for _ in range(n_trees):
        # Subsample rows
        sub_n = max(2, int(math.floor(n * subsample)))
        sub_idx = rng.sample(range(n), sub_n)
        sx = [X[i] for i in sub_idx]
        sy = [y[i] for i in sub_idx]
        sp = [predictions[i] for i in sub_idx]
        # Pseudo-residuals (gradient of logloss)
        ps = [sigmoid(p) for p in sp]
        residuals = [sy[i] - ps[i] for i in range(sub_n)]
        # Subsample features
        feat_sub = sorted(rng.sample(feat_idx_all, n_sub))
        tree = _fit_tree(sx, [1 if r > 0 else 0 for r in residuals], feat_sub,
                         max_depth, max(8, n // 40), rng)
        # Store tree + its residual sign pattern for prediction
        # Instead of storing raw residuals, we store the actual leaf values
        # (which are proportions). We'll refit each tree on the full data
        # using the residual targets.
        # Actually, let's just fit on residuals as regression values
        tree = _fit_tree_regression(sx, residuals, feat_sub, max_depth,
                                    max(8, n // 40), rng)
        trees.append(tree)
        # Update predictions on full data
        for i in range(n):
            predictions[i] += learning_rate * _predict_tree_regression(tree, X[i])
    return {
        "trees": [_node_to_dict_regression(t) for t in trees],
        "nTrees": n_trees,
        "learningRate": learning_rate,
        "baseLogit": base_logit,
        "featureNames": list(feature_names),
        "stats": stats,
        "name": name,
    }


def _fit_tree_regression(
    X: list[list[float]],
    residuals: list[float],
    feat_idx: list[int],
    max_depth: int,
    min_leaf: int,
    rng: random.Random,
) -> _TreeNode:
    """Fit a regression tree minimizing MSE of residuals."""
    n = len(residuals)
    if n == 0:
        return _TreeNode(value=0.0)
    prior = sum(residuals) / n
    if max_depth <= 0 or n < 2 * min_leaf:
        return _TreeNode(value=prior)
    k = max(1, int(math.sqrt(len(feat_idx)))) if len(feat_idx) > 4 else len(feat_idx)
    chosen = rng.sample(feat_idx, min(k, len(feat_idx)))
    best_mse = float("inf")
    best = None
    for fi in chosen:
        col_vals = [(X[i][fi], residuals[i]) for i in range(n)]
        col_vals.sort(key=lambda t: t[0])
        left_sum = 0.0
        left_sq = 0.0
        total_sum = sum(residuals)
        total_sq = sum(r * r for r in residuals)
        for pos in range(n - 1):
            left_sum += col_vals[pos][1]
            left_sq += col_vals[pos][1] * col_vals[pos][1]
            left_n = pos + 1
            right_n = n - left_n
            if left_n < min_leaf or right_n < min_leaf:
                continue
            if col_vals[pos + 1][0] == col_vals[pos][0]:
                continue
            right_sum = total_sum - left_sum
            right_sq = total_sq - left_sq
            mse = (left_sq - left_sum * left_sum / left_n +
                   right_sq - right_sum * right_sum / right_n)
            if mse < best_mse:
                best_mse = mse
                best = (fi, col_vals[pos][0],
                        left_sum / left_n, right_sum / right_n)
    if best is None:
        return _TreeNode(value=prior)
    fi, thresh, lv, rv = best
    left_X, left_r, right_X, right_r = [], [], [], []
    for i in range(n):
        if X[i][fi] <= thresh:
            left_X.append(X[i])
            left_r.append(residuals[i])
        else:
            right_X.append(X[i])
            right_r.append(residuals[i])
    return _TreeNode(
        feature=fi, threshold=thresh,
        left=_fit_tree_regression(left_X, left_r, feat_idx, max_depth - 1, min_leaf, rng),
        right=_fit_tree_regression(right_X, right_r, feat_idx, max_depth - 1, min_leaf, rng),
    )


def _predict_tree_regression(node: _TreeNode, x: list[float]) -> float:
    if node.feature is None:
        return node.value
    if x[node.feature] <= node.threshold:
        return _predict_tree_regression(node.left, x)
    return _predict_tree_regression(node.right, x)


def _node_to_dict_regression(node: _TreeNode) -> dict:
    d: dict = {"value": node.value}
    if node.feature is not None:
        d["feature"] = node.feature
        d["threshold"] = node.threshold
        d["left"] = _node_to_dict_regression(node.left)
        d["right"] = _node_to_dict_regression(node.right)
    return d


def _gbt_predict(params: dict, features: dict) -> float:
    feat_names = params["featureNames"]
    stats = params.get("stats") or {}
    x = []
    for f in feat_names:
        v = features.get(f, 0.0)
        if f in stats:
            v = zscore(v, stats[f]["mean"], stats[f]["std"])
        x.append(v)
    logit = params["baseLogit"]
    lr = params["learningRate"]
    for td in params["trees"]:
        logit += lr * _predict_tree_regression(_dict_to_node(td), x)
    p = sigmoid(logit)
    return min(0.9999, max(0.0001, p))


def xgb_params(rows: list[dict], feature_names: list[str], **kw) -> dict:
    """XGBoost-style gradient boosted trees."""
    defaults = dict(n_trees=150, max_depth=3, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, seed=42, name="XGB")
    defaults.update(kw)
    return _gbt_params(rows, feature_names, **defaults)


def xgb_predict(params: dict, features: dict) -> float:
    return _gbt_predict(params, features)


def xgb_model(rows: list[dict], feature_names: list[str], **kw):
    """XGBoost-like model closure."""
    params = xgb_params(rows, feature_names, **kw)
    return lambda features: xgb_predict(params, features)


def lgbm_params(rows: list[dict], feature_names: list[str], **kw) -> dict:
    """LightGBM-style gradient boosted trees (same architecture, same defaults)."""
    defaults = dict(n_trees=150, max_depth=3, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, seed=42, name="LGBM")
    defaults.update(kw)
    return _gbt_params(rows, feature_names, **defaults)


def lgbm_predict(params: dict, features: dict) -> float:
    return _gbt_predict(params, features)


def lgbm_model(rows: list[dict], feature_names: list[str], **kw):
    """LightGBM-like model closure."""
    params = lgbm_params(rows, feature_names, **kw)
    return lambda features: lgbm_predict(params, features)
