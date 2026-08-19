"""Deployable multi-model stacking ensemble.

The Model Monitor historically reported stacking weights, but `apply_model`
could only *serve* the logistic + Elo blend — the k-NN / boosted-stump / MLP /
naive-Bayes candidates were in-memory closures that could not be serialized to
the JSON model state, so the "blend" was never actually deployed at inference.

This module closes that gap. Every candidate family is fitted to JSON-safe
parameters (`fit_stack_members`) and scored by a pure `predict_member`
function. `fit_stack` then:

  1. fits each family on a chronological fit slice (prior data only),
  2. tunes a convex-combination of those families on the trailing holdout
     (never the scored day — no lookahead), and
  3. re-fits the members on the full training history for deployment.

The result is a serializable `stack` dict that `apply_model` can serve directly,
plus the Elo blend weight used to mix the stacked probability with the Elo
rating in logit space.
"""

from __future__ import annotations

import math

from .ensemble import (
    boosted_stumps_params,
    boosted_stumps_predict,
    weighted_knn_params,
    weighted_knn_predict,
)
from .logistic import (
    build_stacking_weights,
    logistic_logit,
    naive_bayes_params,
    naive_bayes_predict,
    train_logistic,
)
from .metrics import EPS, clamp, compute_brier, logit, sigmoid
from .nn import mlp_params, mlp_predict

# The five serializable families the deployed stack can blend. Elo is handled
# separately through blendW (it is already a two-number state, not a fitted
# object). These names must match engine/model.run_model's candidate table so
# the monitor's stacking weights line up with the deployed members.
STACK_FAMILIES = [
    "Logistic regression",
    "Distance-weighted k-NN (k=21)",
    "Boosted decision stumps",
    "Neural network (MLP)",
    "Gaussian naive Bayes",
]

KNN_K = 21
KNN_TRAIN_CAP = 1500
MIN_STACK_TRAIN = 40
MIN_HOLDOUT = 20


def fit_stack_members(train: list[dict], feature_names: list[str]) -> dict:
    """Fit every deployable family on `train` and return JSON-safe parameters.

    `train` is chronological and prior-only (the caller guarantees no lookahead
    by construction). k-NN is capped to the most recent KNN_TRAIN_CAP rows so
    the serialized state stays small and prediction stays fast.
    """
    knn_train = train[-KNN_TRAIN_CAP:] if len(train) > KNN_TRAIN_CAP else train
    return {
        "Logistic regression": train_logistic(train, feature_names),
        "Distance-weighted k-NN (k=21)": weighted_knn_params(knn_train, feature_names, KNN_K),
        "Boosted decision stumps": boosted_stumps_params(train, feature_names),
        "Neural network (MLP)": mlp_params(train, feature_names),
        "Gaussian naive Bayes": naive_bayes_params(train, feature_names),
    }


def predict_member(name: str, member: dict, features: dict) -> float:
    """Score one serialized ensemble member."""
    if name == "Logistic regression":
        return sigmoid(logistic_logit(member, features, None))
    if name == "Distance-weighted k-NN (k=21)":
        return weighted_knn_predict(member, features)
    if name == "Boosted decision stumps":
        return boosted_stumps_predict(member, features)
    if name == "Neural network (MLP)":
        return mlp_predict(member, features)
    if name == "Gaussian naive Bayes":
        return naive_bayes_predict(member, features)
    raise KeyError(f"unknown stack member: {name}")


def stack_probability(stack: dict | None, features: dict) -> float:
    """Convex combination of the stack members' probabilities.

    Returns None when there is no deployable stack, so callers can fall back
    to the plain logistic path.
    """
    if not stack or not stack.get("members") or not stack.get("weights"):
        return None
    total_w = 0.0
    total_p = 0.0
    for name, weight in stack["weights"].items():
        if weight <= 0 or name not in stack["members"]:
            continue
        p = predict_member(name, stack["members"][name], features)
        total_w += weight
        total_p += weight * p
    if total_w <= 0:
        return None
    return total_p / total_w


def stack_logit(stack: dict | None, features: dict) -> float | None:
    """Logit of the stacked probability (None when the stack is not deployable)."""
    p = stack_probability(stack, features)
    if p is None:
        return None
    return logit(clamp(p, EPS, 1 - EPS))


def _elo_logit(row: dict, hfa: float) -> float:
    p = sigmoid(((row["homeElo"] + hfa - row["awayElo"]) / 400) * math.log(10))
    return logit(p)


def fit_stack(train: list[dict], feature_names: list[str], elo_hfa: float = 30.0) -> tuple[dict, float]:
    """Fit the deployable multi-model stack and the Elo blend weight.

    All tuning happens strictly on the chronological holdout (the trailing 20%
    of `train`), so the scored day is never used to pick weights. Returns
    `(stack, blend_w)`; `blend_w` mixes the stacked logit with the Elo logit in
    `apply_model`.

    Degenerate inputs (too few rows / features) fall back to a logistic-only
    stack with no Elo blend, which is always safe to serve.
    """
    n = len(train)
    logistic = train_logistic(train, feature_names) if n > 0 else None
    fallback_members = {"Logistic regression": logistic} if logistic else {}
    fallback = {
        "members": fallback_members,
        "weights": {"Logistic regression": 1.0} if logistic else {},
    }
    if n < MIN_STACK_TRAIN or len(feature_names) == 0:
        return fallback, 0.0

    split = int(math.floor(n * 0.8))
    fit_rows = train[:split]
    holdout = train[split:]
    if len(fit_rows) < MIN_HOLDOUT or len(holdout) < MIN_HOLDOUT:
        return fallback, 0.0

    fit_members = fit_stack_members(fit_rows, feature_names)
    holdout_preds = {
        name: [predict_member(name, member, r["features"]) for r in holdout]
        for name, member in fit_members.items()
    }
    labels = [r["label"] for r in holdout]
    stacking = build_stacking_weights(holdout_preds, labels)

    # Weights that actually reduced holdout Brier (greedy forward selection).
    weights_oos: dict[str, float] = {}
    for w in stacking["weights"]:
        if w.get("weight", 0.0) > 0 and w["name"] in fit_members:
            weights_oos[w["name"]] = w["weight"]
    if not weights_oos:
        weights_oos = {"Logistic regression": 1.0}
    total = sum(weights_oos.values()) or 1.0
    weights_oos = {k: v / total for k, v in weights_oos.items()}

    # Tune the Elo blend against the out-of-sample stack (fit-slice members,
    # holdout games) so the weight never sees the holdout outcomes.
    stack_oos = {"members": fit_members, "weights": weights_oos}
    stack_logits_oos = [stack_logit(stack_oos, r["features"]) for r in holdout]
    elo_logits_oos = [_elo_logit(r, elo_hfa) for r in holdout]
    best_brier = float("inf")
    blend_w = 0.0
    w = 0.0
    while w <= 1.0001:
        preds = [
            sigmoid((1 - w) * sl + w * el)
            for sl, el in zip(stack_logits_oos, elo_logits_oos)
        ]
        b = compute_brier(preds, labels)
        if b < best_brier:
            best_brier = b
            blend_w = w
        w += 0.05

    # Re-fit members on the FULL training history for deployment.
    members = fit_stack_members(train, feature_names)
    weights = {k: v for k, v in weights_oos.items() if k in members}
    # Logistic is always kept in the stack as the stable, interpretable
    # backbone (it also powers the SHAP readout in apply_model).
    if "Logistic regression" not in weights and "Logistic regression" in members:
        weights["Logistic regression"] = 0.0
    if not weights:
        weights = {"Logistic regression": 1.0}
    total = sum(weights.values()) or 1.0
    weights = {k: v / total for k, v in weights.items()}
    return {"members": members, "weights": weights}, blend_w
