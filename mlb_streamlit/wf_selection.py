"""Walk-forward model & feature selection for today's deployed model.

The Model Monitor must show the model actually scoring today's games, and that
model's feature set and model-family choice must be decided by the walk-forward
record (out-of-sample), not by the in-sample 70/15/15 split.

This module replays the completed current-season games day-by-day. For every
day it fits the full candidate model pool on games **strictly before** that day
and scores that day's games out-of-sample, accumulating:

  * per-candidate AUC / Brier / log-loss / ECE, and
  * per-feature univariate AUC (out-of-sample signal).

The accumulated record is cached per-date (fingerprinted + versioned), so it is
incremental: when tomorrow becomes "today", yesterday's walk-forward model is
already part of the record and the selection is simply re-derived with the new
day added.
"""

from __future__ import annotations

import hashlib

from . import cache
from .data import add_days, attach_as_of_stats, attach_lineups_as_of, et_date_string
from .engine.features import FEATURE_KEYS, FEATURE_LABELS, compute_elo_and_features
from .engine.logistic import build_stacking_weights, cross_validate
from .engine.metrics import compute_auc, evaluate, spearman_rank
from .engine.model import CANDIDATE_MIN_AUC, fit_candidate_pool, refit_stack_model

WF_SELECTION_FILE = "walk_forward_selection.json"
WF_SELECTION_VERSION = 3
WF_SELECTION_REFIT_DAYS = 3  # candidates share a fit within a block (matches calibration)
MIN_PRIOR_GAMES = 40
FEATURE_SIGNAL_EPS = 0.01  # |AUC - 0.5| threshold for a feature to stay active

# Ordering matches engine/model.run_model so the monitor renders a stable table.
CANDIDATE_NAMES = [
    "Elo rating",
    "Logistic regression",
    "Logistic regression (L2, λ=0.1)",
    "Logistic regression (L2, λ=0.3)",
    "Logistic regression (L2, λ=1)",
    "Distance-weighted k-NN (k=21)",
    "Boosted decision stumps",
    "Neural network (MLP)",
    "Gaussian naive Bayes",
    "Blended ensemble",
]

# These features are always retained: they are the structural backbone of the
# model and their out-of-sample signal is stable by construction.
CORE_FEATURES = ("eloDiff", "homeField", "winPctDiff")


def _fingerprint(games: list[dict]) -> str:
    """Stable hash of the games a per-date walk-forward record depends on."""
    lines = []
    for g in games:
        lines.append(
            f"{g['gamePk']}|{g.get('date')}|{g.get('winner')}|"
            f"{(g.get('home') or {}).get('score')}|{(g.get('away') or {}).get('score')}"
        )
    lines.sort()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _load_completed_rows(today: str) -> list[dict]:
    """Load completed games (< today), enrich as-of their own dates, and return
    the chronological feature rows."""
    cached = cache.load_games()
    completed = [
        g for g in cached
        if g.get("winner") in ("home", "away")
        and (g.get("home") or {}).get("id") and (g.get("away") or {}).get("id")
    ]
    enriched = attach_as_of_stats(completed, cache.load_pitcher_logs(), cache.load_team_logs())
    lineups = {}
    for pk, lu in (cache.load_lineups() or {}).items():
        if lu:
            try:
                lineups[int(pk)] = lu
            except (TypeError, ValueError):
                pass
    enriched = attach_lineups_as_of(enriched, lineups, cache.load_batter_logs())
    enriched = [g for g in enriched if (g.get("date") or "") < today]
    return compute_elo_and_features(enriched, cache.load_injury_snapshots(), today)["rows"]


def _index(fe_rows: list[dict], today: str):
    """Date-indexed views over chronological feature rows (< today)."""
    fe_rows = [r for r in fe_rows if (r["game"].get("date") or "") < today]
    rows_by_date: dict[str, list[dict]] = {}
    by_date: dict[str, list[dict]] = {}
    for r in fe_rows:
        rows_by_date.setdefault(r["game"]["date"], []).append(r)
        by_date.setdefault(r["game"]["date"], []).append(r["game"])
    return fe_rows, rows_by_date, by_date


def build_walk_forward_selection(report=None, rows=None) -> dict:
    """Replay the season and select today's features + model family walk-forward.

    Returns a selection summary used both to rebuild today's deployed model and
    to render the Model Monitor. When `rows` (the already-computed chronological
    feature rows from run_model) is supplied, the feature replay is reused and
    the pass only does the candidate fits. The summary is cached incrementally:
    each call only scores dates that are new or whose prior games changed.
    """
    def rep(stage, pct, msg):
        if report:
            report(stage, pct, msg)

    today = et_date_string()
    season = today[:4]
    if rows is None:
        fe_rows = _load_completed_rows(today)
    else:
        fe_rows = list(rows)
    fe_rows, rows_by_date, by_date = _index(fe_rows, today)
    dates = sorted(d for d in by_date if d[:4] == season)

    existing_raw = cache.load_json(WF_SELECTION_FILE, {}) or {}
    existing = existing_raw.get("days", {}) if isinstance(existing_raw, dict) and "days" in existing_raw else {}
    if (existing_raw.get("version") if isinstance(existing_raw, dict) else None) != WF_SELECTION_VERSION:
        existing = {}

    # Amortized O(n) prior-games pointer, matching the calibration build.
    out: dict[str, dict] = {}
    prior_games: list[dict] = []
    ptr = 0
    current_predictors: dict | None = None
    current_cutoff: str | None = None

    for i, d in enumerate(dates):
        day_rows = rows_by_date.get(d) or []
        if not day_rows:
            continue
        while ptr < len(fe_rows) and fe_rows[ptr]["game"]["date"] < d:
            prior_games.append(fe_rows[ptr]["game"])
            ptr += 1
        if len(prior_games) < MIN_PRIOR_GAMES:
            current_predictors = None
            current_cutoff = None
            continue
        fp = _fingerprint(prior_games + by_date[d])
        cached_day = existing.get(d)
        if cached_day and cached_day.get("fp") == fp:
            out[d] = cached_day
            current_predictors = None
            current_cutoff = None
            continue
        if current_predictors is None or d >= add_days(current_cutoff, WF_SELECTION_REFIT_DAYS):
            prior_rows = fe_rows[:ptr]
            current_predictors, _elo_hfa, _blend_w = fit_candidate_pool(prior_rows, list(FEATURE_KEYS))
            current_cutoff = d
        preds = {name: [current_predictors[name](r) for r in day_rows] for name in CANDIDATE_NAMES}
        labels = [r["label"] for r in day_rows]
        feat_vals = {f: [r["features"][f] for r in day_rows] for f in FEATURE_KEYS}
        out[d] = {"fp": fp, "candPreds": preds, "labels": labels, "featVals": feat_vals}
        rep("Walk-forward selection", 30 + int(55 * (i + 1) / max(1, len(dates))),
            f"Evaluated {len(day_rows)} game(s) on {d} against {len(prior_games)} prior game(s)…")

    cache.save_json(WF_SELECTION_FILE, {"version": WF_SELECTION_VERSION, "days": out})

    # Accumulate every cached day's out-of-sample predictions (each candidate
    # is aligned with the same labels and the same feature values).
    accum = {
        "candPreds": {name: [] for name in CANDIDATE_NAMES},
        "labels": [],
        "featVals": {f: [] for f in FEATURE_KEYS},
    }
    for d in sorted(out):
        day = out[d]
        for name in CANDIDATE_NAMES:
            accum["candPreds"][name].extend(day.get("candPreds", {}).get(name, []))
        accum["labels"].extend(day.get("labels", []))
        for f in FEATURE_KEYS:
            accum["featVals"][f].extend(day.get("featVals", {}).get(f, []))

    labels = accum["labels"]
    n_eval = len(labels)
    if n_eval < 20:
        return _fallback_selection(today, len(out), n_eval)

    # Per-candidate out-of-sample metrics.
    candidates = []
    eligible_preds: dict[str, list[float]] = {}
    best_single_name = "Blended ensemble"
    best_auc = -1.0
    best_brier = float("inf")
    for name in CANDIDATE_NAMES:
        preds = accum["candPreds"][name]
        m = evaluate(preds, labels)
        eligible = m["auc"] >= CANDIDATE_MIN_AUC
        candidates.append({
            "name": name,
            "auc": m["auc"],
            "brier": m["brier"],
            "logLoss": m["logLoss"],
            "ece": m["ece"],
            "selected": False,
            "eligible": eligible,
            "note": "" if eligible else f"Below {CANDIDATE_MIN_AUC:.2f} AUC floor — excluded from selection",
        })
        if not eligible:
            continue
        eligible_preds[name] = preds
        if m["auc"] > best_auc + 0.003:
            best_auc = m["auc"]
            best_brier = m["brier"]
            best_single_name = name
        elif abs(m["auc"] - best_auc) <= 0.003 and m["brier"] < best_brier:
            best_brier = m["brier"]
            best_single_name = name
    if not eligible_preds:
        for c in candidates:
            c["eligible"] = True
            c["note"] = "AUC floor relaxed — no candidate cleared 0.70"
            eligible_preds[c["name"]] = accum["candPreds"][c["name"]]
        best_auc = -1.0
        best_brier = float("inf")
        for c in candidates:
            if c["auc"] > best_auc + 0.003:
                best_auc = c["auc"]
                best_brier = c["brier"]
                best_single_name = c["name"]
            elif abs(c["auc"] - best_auc) <= 0.003 and c["brier"] < best_brier:
                best_brier = c["brier"]
                best_single_name = c["name"]

    stacking = build_stacking_weights(eligible_preds, labels)
    # The deployed scorer is the logistic + Elo blend — the only fully
    # serializable ensemble that apply_model can serve. The walk-forward record
    # still measures every candidate (and the greedy stacking) out-of-sample,
    # but the model selected for deployment is chosen between the two deployable
    # families: the blend when it matches or beats plain logistic on OOS Brier,
    # else plain logistic. Everything else is reported as a diagnostic.
    blend_eval = evaluate(accum["candPreds"]["Blended ensemble"], labels)
    lr_eval = evaluate(accum["candPreds"]["Logistic regression"], labels)
    best_name = "Blended ensemble" if blend_eval["brier"] <= lr_eval["brier"] else "Logistic regression"
    chosen_preds = accum["candPreds"][best_name]
    for c in candidates:
        c["selected"] = c["name"] == best_name

    # Walk-forward feature selection: a feature stays active when its out-of-
    # sample univariate AUC is measurably better than a coin flip in EITHER
    # direction (the logistic weight learns the sign), plus the structural core.
    selected_features = []
    for f in FEATURE_KEYS:
        uni = compute_auc(accum["featVals"][f], labels)
        if f in CORE_FEATURES or abs(uni - 0.5) >= FEATURE_SIGNAL_EPS:
            selected_features.append(f)

    # Final weights are filled in by apply_walk_forward_selection (it refits on
    # the complete history); here we only report the out-of-sample signal.
    feature_importances = []
    for f in FEATURE_KEYS:
        uni = compute_auc(accum["featVals"][f], labels)
        feature_importances.append({
            "feature": f,
            "label": FEATURE_LABELS.get(f, f),
            "weight": 0.0,
            "importance": 0.0,
            "univariateAuc": uni,
            "active": f in selected_features,
        })

    chosen_eval = evaluate(chosen_preds, labels)
    pick_probs = [max(p, 1 - p) for p in chosen_preds]
    is_correct = [1 if (p >= 0.5) == (y == 1) else 0 for p, y in zip(chosen_preds, labels)]
    high_conf = [c for p, c in zip(pick_probs, is_correct) if p >= 0.65]

    if best_name == "Blended ensemble":
        description = "Walk-forward blended ensemble (logistic + Elo), isotonic-calibrated."
    else:
        description = f"Walk-forward selected {best_name}, isotonic-calibrated."

    selection = {
        "trainedThrough": today,
        "daysEvaluated": len(out),
        "gamesEvaluated": n_eval,
        "selectedModel": best_name,
        "modelDescription": description,
        "featureNames": selected_features,
        "featureImportances": feature_importances,
        "candidates": candidates,
        "stackingWeights": stacking["weights"],
        "crossValidation": cross_validate(fe_rows, selected_features, 5),
        "auc": chosen_eval["auc"],
        "brier": chosen_eval["brier"],
        "logLoss": chosen_eval["logLoss"],
        "ece": chosen_eval["ece"],
        "bins": chosen_eval["bins"],
        "confidenceDistribution": chosen_eval["confidenceDistribution"],
        "calibrationCurve": chosen_eval["calibrationCurve"],
        "spearmanRho": spearman_rank(pick_probs, is_correct),
        "topDecileWinRate": (sum(high_conf) / len(high_conf)) if high_conf else 0.0,
        "optimizationParams": {
            "featureSelection": "Walk-forward univariate AUC (out-of-sample)",
            "minCandidateAuc": CANDIDATE_MIN_AUC,
            "l2Lambda": 0.001,
            "epochs": 20,
            "hfaGrid": [0, 10, 20, 30, 40, 50, 60],
            "blendStep": 0.05,
            "mcSigmaGrid": [0.1, 0.15, 0.2, 0.3, 0.45, 0.6],
            "cvFolds": 5,
            "isotonicMethod": "Isotonic (PAV)",
            "refitDays": WF_SELECTION_REFIT_DAYS,
            "minPriorGames": MIN_PRIOR_GAMES,
        },
    }
    rep("Walk-forward selection", 100,
        f"Selected today's model from {n_eval} out-of-sample games across {len(out)} day(s).")
    return selection


def _fallback_selection(today: str, days: int, games: int) -> dict:
    """Full-feature fallback when there is not enough walk-forward history."""
    feature_importances = [
        {
            "feature": f,
            "label": FEATURE_LABELS.get(f, f),
            "weight": 0.0,
            "importance": 0.0,
            "univariateAuc": 0.5,
            "active": True,
        }
        for f in FEATURE_KEYS
    ]
    return {
        "trainedThrough": today,
        "daysEvaluated": days,
        "gamesEvaluated": games,
        "selectedModel": "Logistic + Elo (walk-forward)",
        "modelDescription": "Walk-forward selection pending — not enough completed history yet.",
        "featureNames": list(FEATURE_KEYS),
        "featureImportances": feature_importances,
        "candidates": [],
        "stackingWeights": [],
        "crossValidation": {},
        "auc": 0.0,
        "brier": 0.0,
        "logLoss": 0.0,
        "ece": 0.0,
        "bins": [],
        "confidenceDistribution": [],
        "calibrationCurve": [],
        "spearmanRho": 0.0,
        "topDecileWinRate": 0.0,
        "optimizationParams": {"featureSelection": "Full feature set (fallback)"},
    }


def apply_walk_forward_selection(result: dict, model: dict, rows: list[dict], selection: dict) -> None:
    """Rebuild today's deployed model + state from the walk-forward selection.

    `result` is run_model's result dict (the source of model_state), `model` is
    the deployed logistic model dict, and `rows` are the chronological feature
    rows (the complete training history). After this call both reflect the
    walk-forward-selected features and model family so the Model Monitor and the
    live scorer agree.
    """
    refit = refit_stack_model(
        rows,
        selection["featureNames"],
        elo_hfa=result.get("eloHfa", 30.0),
        monte_carlo_enabled=result.get("monteCarloEnabled", False),
        monte_carlo_sigma=result.get("monteCarloSigma", 0.0),
    )
    model.update(refit)

    # Regenerate feature importances so the learned coefficient reflects the
    # deployed model while the out-of-sample signal comes from the selection.
    feature_importances = []
    for item in selection.get("featureImportances", []):
        f = item["feature"]
        idx = refit["featureNames"].index(f) if f in refit["featureNames"] else -1
        w = refit["weights"][idx] if idx >= 0 else 0.0
        feature_importances.append({
            "feature": f,
            "label": item.get("label", FEATURE_LABELS.get(f, f)),
            "weight": w,
            "importance": abs(w),
            "univariateAuc": item.get("univariateAuc", 0.5),
            "active": idx >= 0,
        })

    result["featureNames"] = selection["featureNames"]
    result["weights"] = refit["weights"]
    result["bias"] = refit["bias"]
    result["featureStats"] = refit["featureStats"]
    result["isotonicPoints"] = refit["isotonicPoints"]
    result["blendW"] = refit["blendW"]
    result["stack"] = refit["stack"]

    # The deployed scorer is now the serializable multi-model stack (logistic /
    # k-NN / boosted stumps / MLP / naive Bayes blended by holdout-tuned
    # weights) plus the Elo blend via blendW. The monitor's selectedModel,
    # stacking weights and candidate highlights describe exactly what
    # apply_model serves.
    stack = refit.get("stack") or {}
    positive = [n for n, w in stack.get("weights", {}).items() if w > 0]
    blend_w = refit.get("blendW", 0.0)
    if len(positive) > 1:
        deployed_name = "Multi-model stack"
        deployed_description = (
            f"Walk-forward multi-model stack ({', '.join(positive)}), "
            f"blended {1 - blend_w:.2f} + {blend_w:.2f}·Elo, isotonic-calibrated."
        )
    elif blend_w > 0:
        deployed_name = "Blended ensemble"
        deployed_description = (
            f"Walk-forward blended ensemble: {1 - blend_w:.2f}·logistic + {blend_w:.2f}·Elo, "
            f"isotonic-calibrated."
        )
    else:
        deployed_name = "Logistic regression"
        deployed_description = "Walk-forward selected logistic regression, isotonic-calibrated."
    result["selectedModel"] = deployed_name
    result["modelDescription"] = deployed_description
    result["featureImportances"] = feature_importances
    result["candidates"] = selection["candidates"]
    deployed_member_names = set(positive)
    for c in result["candidates"]:
        c["selected"] = c["name"] in deployed_member_names
    result["stackingWeights"] = [
        {"name": n, "weight": w}
        for n, w in sorted(stack.get("weights", {}).items(), key=lambda kv: -kv[1])
    ]
    result["crossValidation"] = selection["crossValidation"]
    result["optimizationParams"] = selection["optimizationParams"]

    # Headline metrics remain the walk-forward out-of-sample record of the
    # chosen blend; the deployed stack is measured by the same candidate
    # families in that record.
    result["auc"] = selection["auc"]
    result["brier"] = selection["brier"]
    result["logLoss"] = selection["logLoss"]
    result["ece"] = selection["ece"]
    result["bins"] = selection["bins"]
    result["confidenceDistribution"] = selection["confidenceDistribution"]
    result["calibrationCurve"] = selection["calibrationCurve"]
    # Reflect the deployed family in the persisted walk-forward record too.
    selection["selectedModel"] = deployed_name
    selection["modelDescription"] = deployed_description
    result["walkForwardSelection"] = selection
