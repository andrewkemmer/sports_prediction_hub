"""Point-in-time multi-signal concordance gating.

The gate is an abstention layer, not a second probability model.  It never
changes the base model probability or pick.  A gated pick is emitted only when
several independent, prior-only signal families agree with the base pick.

Signal inputs are all already present in the as-of feature row or serialized
model state: the deployed stack, Elo, record/form, pitching, offense/lineup,
and schedule/context.  Threshold selection uses only the trailing portion of
the supplied prior rows; callers use it inside each walk-forward block.
"""

from __future__ import annotations

import math

from .logistic import logistic_logit
from .metrics import clamp, sigmoid
from .stack import stack_probability

GATE_VERSION = 1
GATE_THRESHOLDS = (0.55, 0.65, 0.75, 0.85, 0.95)
GATE_MIN_SIGNALS = (3, 4, 5)
GATE_MIN_VALIDATION_ROWS = 30
GATE_MIN_COVERAGE = 0.20
GATE_MIN_LIFT = 0.005
SIGNAL_MARGIN = 0.04


# The scales turn heterogeneous feature deltas into bounded, comparable
# directional scores.  They are fixed domain scales, not learned from the
# scored game, so signal extraction cannot peek at its outcome.
_GROUP_SPECS = {
    "record": (("winPctDiff", 0.12),),
    "form": (("formDiff", 0.14),),
    "pitching": (
        ("spFipDiff", 1.25),
        ("spEraDiff", 1.50),
        ("spK9Diff", 1.20),
        ("spWhipDiff", 0.40),
        ("spRecentDiff", 1.50),
        ("spTrendDiff", 1.00),
        ("spRestWorkloadDiff", 3.00),
    ),
    "offense": (
        ("opsDiff", 0.08),
        ("teamEraDiff", 1.25),
        ("defEffDiff", 0.006),
    ),
    "lineup": (
        ("lineupOpsDiff", 0.08),
        ("lineupWobaDiff", 0.07),
        ("lineupIsoDiff", 0.05),
        ("lineupHotDiff", 0.08),
        ("lineupMomentumDiff", 0.08),
        ("lineupFatigueDiff", 3.00),
    ),
    "context": (
        ("restDiff", 2.00),
        ("injuryDiff", 3.00),
    ),
}

_SIGNAL_LABELS = {
    "stack": "Model stack",
    "elo": "Elo rating",
    "record": "Season record",
    "form": "Recent form",
    "pitching": "Pitching matchup",
    "offense": "Team offense",
    "lineup": "Starting lineup",
    "context": "Schedule / roster context",
}


def default_gate_config(reason: str = "Not enough prior-only validation history") -> dict:
    """Return a safe, serializable disabled configuration."""
    return {
        "version": GATE_VERSION,
        "enabled": False,
        "threshold": 0.75,
        "minSignals": 3,
        "minCoverage": GATE_MIN_COVERAGE,
        "tunedOn": 0,
        "validationCoverage": 0.0,
        "validationWinRate": 0.0,
        "validationBaselineWinRate": 0.0,
        "validationLift": 0.0,
        "validationAccepted": 0,
        "reason": reason,
    }


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _probability_score(probability: float | None) -> float | None:
    if not _is_number(probability):
        return None
    p = clamp(float(probability), 0.001, 0.999)
    score = 2.0 * p - 1.0
    if abs(score) < SIGNAL_MARGIN:
        return 0.0
    return clamp(score, -1.0, 1.0)


def _direction(score: float | None) -> str | None:
    if score is None or abs(score) < SIGNAL_MARGIN:
        return None
    return "home" if score > 0 else "away"


def _group_score(features: dict, specs: tuple[tuple[str, float], ...], *, lineup: bool = False) -> float | None:
    if lineup and features.get("lineupKnown") != 1:
        return None
    values: list[float] = []
    for name, scale in specs:
        value = features.get(name)
        if not _is_number(value) or not _is_number(scale) or scale == 0:
            continue
        # A zero is a real neutral signal.  If every value is zero, return
        # None so missing historical data does not count as disagreement.
        if abs(float(value)) > 1e-12:
            values.append(clamp(float(value) / abs(float(scale)), -1.0, 1.0))
    if not values:
        return None
    score = sum(values) / len(values)
    return clamp(score, -1.0, 1.0)


def _stack_score(model: dict, features: dict) -> float | None:
    probability = stack_probability(model.get("stack"), features)
    if probability is None and model.get("featureNames"):
        # Old/current states that have no serialized stack still have the
        # deployable logistic parameters.  This fallback is deterministic and
        # uses only the row's prior-only features.
        probability = sigmoid(logistic_logit(model, features, None))
    if probability is None:
        return None
    return _probability_score(probability)


def extract_signals(
    model: dict,
    features: dict,
    home_elo: float,
    away_elo: float,
) -> dict[str, dict]:
    """Extract directional signal families for one as-of game row.

    Positive scores favor the home team, negative scores favor the away team.
    A neutral/unknown signal is marked unavailable and is excluded from the
    concordance denominator.
    """
    hfa = float(model.get("eloHfa", 30.0) or 30.0)
    elo_probability = sigmoid(((float(home_elo) + hfa - float(away_elo)) / 400.0) * math.log(10))
    scores: dict[str, float | None] = {
        "stack": _stack_score(model, features),
        "elo": _probability_score(elo_probability),
        "record": _group_score(features, _GROUP_SPECS["record"]),
        "form": _group_score(features, _GROUP_SPECS["form"]),
        "pitching": _group_score(features, _GROUP_SPECS["pitching"]),
        "offense": _group_score(features, _GROUP_SPECS["offense"]),
        "lineup": _group_score(features, _GROUP_SPECS["lineup"], lineup=True),
        "context": _group_score(features, _GROUP_SPECS["context"]),
    }
    return {
        name: {
            "label": _SIGNAL_LABELS[name],
            "score": round(float(score), 6) if score is not None else None,
            "direction": _direction(score),
            "available": _direction(score) is not None,
        }
        for name, score in scores.items()
    }


def _base_prediction_from_probability(probability: float) -> dict:
    p = clamp(float(probability), 0.001, 0.999)
    home = p >= 0.5
    return {
        "homeWinProb": p,
        "awayWinProb": 1.0 - p,
        "pickTeam": "home" if home else "away",
        "pickProb": p if home else 1.0 - p,
    }


def apply_concordance_gate(
    prediction: dict,
    model: dict,
    features: dict,
    home_elo: float,
    away_elo: float,
    config: dict | None,
) -> dict:
    """Apply one serialized gate config without modifying base prediction keys."""
    cfg = config or default_gate_config("No walk-forward gate configuration")
    signals = extract_signals(model, features, home_elo, away_elo)
    available = [s for s in signals.values() if s.get("direction") in ("home", "away")]
    base_pick = prediction.get("pickTeam") or ("home" if prediction.get("homeWinProb", 0.5) >= 0.5 else "away")
    agrees = sum(1 for s in available if s["direction"] == base_pick)
    count = len(available)
    agreement = agrees / count if count else 0.0
    threshold = clamp(float(cfg.get("threshold", 0.75) or 0.75), 0.5, 1.0)
    min_signals = max(1, int(cfg.get("minSignals", 3) or 3))
    enabled = bool(cfg.get("enabled", False))
    accepted = enabled and count >= min_signals and agreement + 1e-12 >= threshold
    if not enabled:
        reason = cfg.get("reason") or "Gate disabled by prior-only validation guard"
    elif count < min_signals:
        reason = f"Only {count} independent signal(s); requires {min_signals}"
    elif accepted:
        reason = f"{agrees}/{count} independent signals agree"
    else:
        reason = f"{agrees}/{count} signals agree; threshold is {threshold:.0%}"
    pick_prob = prediction.get("pickProb")
    return {
        "gateEnabled": enabled,
        "gateAccepted": accepted,
        "gatedPickTeam": base_pick if accepted else None,
        "gatedPickProb": float(pick_prob) if accepted and _is_number(pick_prob) else None,
        "gatedHomeWinProb": float(prediction["homeWinProb"]) if accepted and _is_number(prediction.get("homeWinProb")) else None,
        "concordance": round(agreement, 6),
        "gateAgreeCount": agrees,
        "gateSignalCount": count,
        "gateThreshold": threshold,
        "gateMinSignals": min_signals,
        "gateReason": reason,
        "gateSignals": signals,
    }


def _wilson_lower(wins: int, total: int) -> float:
    if total <= 0:
        return 0.0
    z = 1.96
    p = wins / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (center - spread) / denominator


def tune_concordance_gate(
    rows: list[dict],
    model: dict,
    base_predictions: list[dict] | None = None,
    min_coverage: float = GATE_MIN_COVERAGE,
) -> dict:
    """Tune threshold/min-signals using only the supplied prior rows.

    The trailing 30% of ``rows`` is the gate-validation slice.  Rows are
    chronological and must be strictly before the walk-forward date supplied
    by the caller.  A minimum accepted count/coverage guard prevents selecting
    a spectacular but statistically meaningless 1-for-1 subset.  Selection
    maximizes the Wilson lower bound, then empirical conditional win rate, then
    coverage; no future row is inspected.
    """
    cfg = default_gate_config()
    n = len(rows)
    if n < GATE_MIN_VALIDATION_ROWS:
        cfg["reason"] = f"Only {n} prior rows; requires {GATE_MIN_VALIDATION_ROWS}"
        return cfg

    start = max(0, int(math.floor(n * 0.70)))
    validation = rows[start:]
    if len(validation) < GATE_MIN_VALIDATION_ROWS:
        cfg["reason"] = f"Only {len(validation)} gate-validation rows"
        return cfg

    if base_predictions is None or len(base_predictions) != n:
        base_predictions = []
        for row in rows:
            stack = stack_probability(model.get("stack"), row.get("features", {}))
            base_predictions.append(_base_prediction_from_probability(stack if stack is not None else 0.5))

    min_coverage = clamp(float(min_coverage), 0.05, 1.0)
    min_count = max(20, int(math.ceil(len(validation) * min_coverage)))
    baseline_correct = 0
    for i in range(start, n):
        baseline_correct += int(base_predictions[i].get("pickTeam") == ("home" if rows[i].get("label") == 1 else "away"))
    baseline_rate = baseline_correct / len(validation)

    candidates: list[dict] = []
    for threshold in GATE_THRESHOLDS:
        for min_signals in GATE_MIN_SIGNALS:
            candidate_cfg = {
                "version": GATE_VERSION,
                "enabled": True,
                "threshold": threshold,
                "minSignals": min_signals,
                "minCoverage": min_coverage,
            }
            accepted = 0
            wins = 0
            total_agreement = 0.0
            for i in range(start, n):
                row = rows[i]
                gate = apply_concordance_gate(
                    base_predictions[i],
                    model,
                    row.get("features", {}),
                    row.get("homeElo", 1500.0),
                    row.get("awayElo", 1500.0),
                    candidate_cfg,
                )
                if not gate["gateAccepted"]:
                    continue
                accepted += 1
                wins += int(("home" if row.get("label") == 1 else "away") == gate["gatedPickTeam"])
                total_agreement += gate["concordance"]
            if accepted < min_count:
                continue
            rate = wins / accepted
            candidates.append({
                "threshold": threshold,
                "minSignals": min_signals,
                "accepted": accepted,
                "coverage": accepted / len(validation),
                "winRate": rate,
                "lift": rate - baseline_rate,
                "wilsonLower": _wilson_lower(wins, accepted),
                "meanConcordance": total_agreement / accepted,
            })

    if not candidates:
        cfg["reason"] = "No threshold met the prior-only coverage guard"
        cfg["validationCoverage"] = 0.0
        cfg["validationBaselineWinRate"] = round(baseline_rate, 6)
        cfg["tunedOn"] = len(validation)
        return cfg

    best = max(
        candidates,
        key=lambda c: (c["wilsonLower"], c["winRate"], c["coverage"], -c["threshold"], -c["minSignals"]),
    )
    # If the best guarded subset did not beat the unconditional prior-only
    # record, do not manufacture a higher win rate by abstaining.  The model
    # remains fully available and the monitor explains why the gate is off.
    enabled = best["lift"] >= GATE_MIN_LIFT
    cfg.update({
        "enabled": enabled,
        "threshold": best["threshold"],
        "minSignals": best["minSignals"],
        "minCoverage": min_coverage,
        "tunedOn": len(validation),
        "validationCoverage": round(best["coverage"], 6),
        "validationWinRate": round(best["winRate"], 6),
        "validationBaselineWinRate": round(baseline_rate, 6),
        "validationLift": round(best["lift"], 6),
        "validationAccepted": best["accepted"],
        "validationWilsonLower": round(best["wilsonLower"], 6),
        "validationMeanConcordance": round(best["meanConcordance"], 6),
        "reason": (
            "Prior-only gate selected by Wilson lower bound and conditional win rate"
            if enabled else
            "Gate held out because prior-only conditional win rate did not improve"
        ),
    })
    return cfg


def summarize_gate_results(rows: list[dict]) -> dict:
    """Aggregate persisted gated outcomes separately from base metrics."""
    completed = [r for r in rows if r.get("isCorrect") is not None]
    accepted = [r for r in completed if r.get("gateAccepted") is True]
    wins = [r for r in accepted if r.get("gatedIsCorrect") is True]
    base_wins = [r for r in completed if r.get("isCorrect") is True]
    return {
        "total": len(completed),
        "accepted": len(accepted),
        "coverage": len(accepted) / len(completed) if completed else 0.0,
        "wins": len(wins),
        "losses": len(accepted) - len(wins),
        "winRate": len(wins) / len(accepted) if accepted else 0.0,
        "baseWinRate": len(base_wins) / len(completed) if completed else 0.0,
        "lift": (len(wins) / len(accepted) - len(base_wins) / len(completed)) if accepted and completed else 0.0,
        "meanConcordance": sum(float(r.get("concordance", 0.0) or 0.0) for r in accepted) / len(accepted) if accepted else 0.0,
    }
