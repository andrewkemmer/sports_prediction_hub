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

from .logistic import (
    build_stacking_weights,
    logistic_logit,
    train_logistic,
)
from .metrics import EPS, clamp, compute_brier, logit, sigmoid
from .nn import mlp_params, mlp_predict
from .tree_ensemble import rf_params, rf_predict, xgb_params, xgb_predict, lgbm_params, lgbm_predict

# The five serializable families the deployed stack can blend. Elo is handled
# separately through blendW (it is already a two-number state, not a fitted
# object). These names must match engine/model.run_model's candidate table so
# the monitor's stacking weights line up with the deployed members.
STACK_FAMILIES = [
    "Logistic regression",
    "Random Forest",
    "Neural network (MLP)",
    "XGBoost",
    "LightGBM",
]

MIN_STACK_TRAIN = 40
MIN_HOLDOUT = 20

# Per-family feature subsets to decorrelate errors across the stack.
# Every family below otherwise trains on the *identical* columns, so its
# errors are near-perfectly correlated with the backbone's — a convex blend
# then has nothing to contribute and the holdout-tuned weights collapse onto
# a single member.  Giving each family a different, overlapping slice of the
# feature space decorrelates their errors, which is the only reason a blend
# exists: a family that is weaker overall can still cover games the backbone
# misreads.  The structural core (Elo edge / record) is always kept so no
# member is starved of the strongest varying signals. `homeField` is retained
# only when a legacy caller supplies the full schema; production
# MODEL_FEATURE_KEYS excludes the constant row indicator before fitting.  The
# logistic backbone keeps the FULL active set for compatibility and SHAP
# output.
CORE_STACK_FAMILY = ("eloDiff", "winPctDiff", "homeField")

STACK_FAMILY_SUBSETS: dict[str, tuple[str, ...] | None] = {
    # Ratings + recent form + pitching workload: matchup-driven read.
    "Random Forest": (
        "eloDiff", "winPctDiff", "formDiff", "restDiff", "injuryDiff",
        "spFipDiff", "spEraDiff", "spK9Diff", "spWhipDiff", "spRecentDiff",
        "spTrendDiff", "spRestWorkloadDiff", "opsDiff", "teamEraDiff",
    ),
    # Pitching-vs-offense interactions + environment: the MLP's nonlinear read.
    "Neural network (MLP)": (
        "restDiff", "injuryDiff", "spFipDiff", "spEraDiff", "spK9Diff",
        "opsDiff", "teamEraDiff", "defEffDiff", "lineupKnown", "lineupOpsDiff",
        "lineupHotDiff", "parkFactor", "tempDev", "windMph",
    ),
    # Gradient-boosted trees excel at nonlinear feature interactions:
    # lineup + offense + environment gives XGBoost a complementary read.
    "XGBoost": (
        "formDiff", "opsDiff", "teamEraDiff", "defEffDiff", "parkFactor",
        "tempDev", "windMph", "lineupKnown", "lineupOpsDiff",
        "lineupWobaDiff", "lineupIsoDiff", "lineupHotDiff",
        "lineupMomentumDiff", "lineupFatigueDiff", "lineupParkInteract",
    ),
    # LightGBM sees a different slice: pitching + matchup + rest interactions.
    "LightGBM": (
        "formDiff", "restDiff", "injuryDiff", "spFipDiff", "spEraDiff",
        "spK9Diff", "spWhipDiff", "spRecentDiff", "spTrendDiff",
        "spRestWorkloadDiff", "spFipRestInteract", "eloWinPctInteract",
        "bvpOpsDiff", "platoonOpsDiff", "vsTeamOpsDiff",
    ),
}


def member_feature_names(feature_names: list[str], family: str) -> list[str]:
    """Per-family feature slice: core + that family's subset, filtered to the
    caller's active feature set. The logistic backbone always keeps the full
    active set (it powers the SHAP readout in apply_model)."""
    active = set(feature_names)
    if family == "Logistic regression":
        return list(feature_names)
    subset = STACK_FAMILY_SUBSETS.get(family)
    if not subset:
        return list(feature_names)
    chosen = [f for f in CORE_STACK_FAMILY if f in active]
    chosen += [f for f in subset if f in active and f not in chosen]
    return chosen or list(feature_names)


def fit_stack_members(train: list[dict], feature_names: list[str], mlp_epochs: int = 40) -> dict:
    """Fit every deployable family on `train` and return JSON-safe parameters.

    `train` is chronological and prior-only (the caller guarantees no lookahead
    by construction). `mlp_epochs` lets the repeated walk-forward fits trade a
    little MLP capacity for a large speedup (the MLP fit dominates backtest
    CPU); the deployed model keeps the full default.
    """
    members: dict[str, dict] = {}
    for family in STACK_FAMILIES:
        feats = member_feature_names(feature_names, family)
        if family == "Logistic regression":
            # Canonical production ridge per policy: λ=0.1. Aligns the stack
            # member with the per-date candidate pool (which already trains
            # the LR variant at λ=0.1) so the dashboard's "logistic" row is a
            # single, consistent model rather than a default-ridge vs tuned
            # mismatch.
            members[family] = train_logistic(train, feats, lambda_=0.1)
        elif family == "Random Forest":
            members[family] = rf_params(train, feats)
        elif family == "Neural network (MLP)":
            members[family] = mlp_params(train, feats, epochs=mlp_epochs)
        elif family == "XGBoost":
            members[family] = xgb_params(train, feats)
        elif family == "LightGBM":
            members[family] = lgbm_params(train, feats)
    return members


def normalize_stack_weights(weights: dict | None, member_names: list[str] | None = None) -> dict[str, float]:
    """Return one finite, non-negative simplex vector for the stack.

    Weight tuning and JSON round-tripping happen in several layers. Normalize
    once at the serving boundary so a stale/rounded payload can never make the
    execution layer treat separate families as independent 100% models.
    """
    allowed = list(member_names or (weights or {}).keys())
    positive = {
        name: float((weights or {}).get(name, 0.0))
        for name in allowed
        if isinstance((weights or {}).get(name, 0.0), (int, float))
        and math.isfinite(float((weights or {}).get(name, 0.0)))
        and float((weights or {}).get(name, 0.0)) > 0
    }
    total = sum(positive.values())
    if total <= 0:
        # Prefer the canonical lambda=0.1 label when allowed; fall back to the
        # bare logistic name for back-compat.
        for canon_name in ("Logistic regression (L2, λ=0.1)", "Logistic regression"):
            if canon_name in allowed:
                return {canon_name: 1.0}
        return {}
    return {name: value / total for name, value in positive.items()}


def predict_member(name: str, member: dict, features: dict) -> float:
    """Score one serialized ensemble member.

    The canonical production label for the logistic family is
    "Logistic regression (L2, λ=0.1)" (per policy); the bare "Logistic regression"
    alias is also accepted for backward-compat with any consumer that stored
    under the short name. Both routes hit the same logistic_logit decoder.
    """
    if name in ("Logistic regression", "Logistic regression (L2, λ=0.1)"):
        return sigmoid(logistic_logit(member, features, None))
    if name == "Random Forest":
        return rf_predict(member, features)
    if name == "Neural network (MLP)":
        return mlp_predict(member, features)
    if name == "XGBoost":
        return xgb_predict(member, features)
    if name == "LightGBM":
        return lgbm_predict(member, features)
    raise KeyError(f"unknown stack member: {name}")


def stack_probability(stack: dict | None, features: dict) -> float:
    """Convex combination of the stack members' probabilities.

    Returns None when there is no deployable stack, so callers can fall back
    to the plain logistic path.
    """
    if not stack or not stack.get("members") or not stack.get("weights"):
        return None
    weights = normalize_stack_weights(stack.get("weights"), list(stack.get("members", {}).keys()))
    total_w = sum(weights.values())
    if total_w <= 0:
        return None
    total_p = 0.0
    for name, weight in weights.items():
        if name not in stack["members"]:
            continue
        p = predict_member(name, stack["members"][name], features)
        total_p += weight * p
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


def fit_stack(train: list[dict], feature_names: list[str], elo_hfa: float = 30.0, mlp_epochs: int = 40) -> tuple[dict, float]:
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

    fit_members = fit_stack_members(fit_rows, feature_names, mlp_epochs=mlp_epochs)
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
    weights_oos = normalize_stack_weights(weights_oos, list(fit_members.keys()))

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
    members = fit_stack_members(train, feature_names, mlp_epochs=mlp_epochs)
    weights = {k: v for k, v in weights_oos.items() if k in members}
    # Logistic is always kept in the stack as the stable, interpretable
    # backbone (it also powers the SHAP readout in apply_model).
    if "Logistic regression" not in weights and "Logistic regression" in members:
        weights["Logistic regression"] = 0.0
    if not weights:
        weights = {"Logistic regression": 1.0}
    weights = normalize_stack_weights(weights, list(members.keys()))
    return {"members": members, "weights": weights}, blend_w
