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
import math

from . import cache
from .data import add_days, attach_as_of_stats, attach_lineups_as_of, et_date_string
from .engine.features import MODEL_FEATURE_KEYS, FEATURE_KEYS, FEATURE_LABELS, compute_elo_and_features
from .engine.gating import apply_concordance_gate, default_gate_config, summarize_gate_results, tune_concordance_gate
from .engine.logistic import cross_validate, logistic_logit, train_logistic_l1
from .engine.metrics import compute_auc, compute_brier, evaluate, roundn, sigmoid, spearman_rank
from .engine.model import CANDIDATE_MIN_AUC, apply_model, elo_prob, fit_walk_forward_step, refit_stack_model
from .engine.stack import predict_member, stack_probability

WF_SELECTION_FILE = "walk_forward_selection.json"
# Gate configuration is part of the per-date scoring recipe. Bump this when
# the gate is introduced/changed so pre-gate selection caches cannot silently
# disable the live abstention layer. Version 10 adds per-date `gateDetails`
# (the full compact gate result per game) so the walk-forward calibration
# backtest can reuse these predictions instead of re-fitting the same models.
WF_SELECTION_VERSION = 13  # schedule-clock provenance + T-1 state boundary
WF_SELECTION_REFIT_DAYS = 7  # candidates share a fit within a block (matches calibration)
WF_TRAIN_WINDOW = 2000  # rolling window of most-recent prior games for each candidate fit
WF_MLP_EPOCHS = 20  # the MLP fit dominates backtest CPU; cap it for the repeated walk-forward fits
MIN_PRIOR_GAMES = 40
FEATURE_SIGNAL_EPS = 0.01  # |AUC - 0.5| threshold for a feature to stay active

# L1 (LASSO) feature selection — the per-date walk-forward selector. The penalty
# strength is tuned on a chronological holdout by Brier (fit on Brier), and the
# stability rule keeps features selected in >= 2 of the last 3 blocks so the
# per-date feature sets don't churn.
#
# The grid extends well below the old 0.005 floor. MLB feature signals are weak
# (univariate AUC ~0.51-0.55), so the effective glmnet penalty (lambda * n,
# n ≈ WF_TRAIN_WINDOW ≈ 2000) at 0.005 is already strong enough to soft-threshold
# every non-core feature to zero — which is why the monitor was collapsing to
# the 3-feature CORE_FEATURES backbone. Lighter penalties let the Brier-tuned
# selector keep weak-but-real signals instead of throwing them away.
L1_LAMBDA_GRID = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
L1_MIN_ROWS = 100
STABILITY_K = 3
STABILITY_VOTES = 2

# Ordering matches the deployable stack families (engine.stack.STACK_FAMILIES)
# plus Elo, the ridge-logistic diagnostics, and the raw multi-model blend.
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
    "Multi-model stack",
]

# These features are always retained: they are the structural backbone of the
# model and their out-of-sample signal is stable by construction.
CORE_FEATURES = ("eloDiff", "winPctDiff")


def _l1_selected_features(prior_rows: list[dict], feature_names: list[str]) -> list[str]:
    """L1 (LASSO) logistic on prior-only rows -> features with nonzero weight.

    The penalty strength is tuned on the trailing 20% chronological holdout by
    Brier (fit on Brier). A per-feature out-of-sample univariate-AUC screen is
    unioned with the L1 survivors so an over-shrunk L1 fit cannot discard a
    feature that carries genuine signal; the structural core features are always
    retained. With too little history, fall back to the core features only.
    """
    if len(prior_rows) < L1_MIN_ROWS:
        return [f for f in feature_names if f in CORE_FEATURES]
    n = len(prior_rows)
    holdout = prior_rows[int(math.floor(n * 0.8)):]
    holdout_labels = [r["label"] for r in holdout]

    # Nested univariate signal screen: computed only on the prior-only training
    # window, so no future result influences it. A feature whose single-feature
    # AUC clears FEATURE_SIGNAL_EPS survives even when the Brier-tuned L1 lambda
    # shrinks its *joint* coefficient to zero. Without this, weak-but-real MLB
    # signals (AUC ~0.51-0.53) collapse to the 3-feature core backbone. The
    # screen uses the full prior window (not just the trailing holdout) so its
    # AUC estimate has enough rows to separate weak signal from noise.
    screen: set[str] = set()
    prior_labels = [r["label"] for r in prior_rows]
    for f in feature_names:
        auc = compute_auc([r["features"][f] for r in prior_rows], prior_labels)
        if abs(auc - 0.5) >= FEATURE_SIGNAL_EPS:
            screen.add(f)

    best_model = None
    best_brier = float("inf")
    for lam in L1_LAMBDA_GRID:
        m = train_logistic_l1(prior_rows, feature_names, lambda_l1=lam)
        if len(holdout) >= 20:
            preds = [sigmoid(logistic_logit(m, r["features"], None)) for r in holdout]
            b = compute_brier(preds, holdout_labels)
            if b < best_brier:
                best_brier = b
                best_model = m
        elif best_model is None:
            best_model = m
    m = best_model or train_logistic_l1(prior_rows, feature_names)
    sel = {f for f, w in zip(m["featureNames"], m["weights"]) if abs(w) > 1e-9}
    sel.update(screen)
    sel.update(CORE_FEATURES)
    return [f for f in feature_names if f in sel]


def _stabilize_features(current: list[str], history: list[list[str]]) -> list[str]:
    """Keep current selections plus any feature selected in >= 2 of the last 3
    blocks, plus the structural core.

    Reduces per-date churn: a feature that flickers out for one block survives
    via persistence and decays only after it stops being selected for 3 blocks.
    The current L1 selection is always trusted immediately.
    """
    recent = (history + [current])[-STABILITY_K:]
    votes: dict[str, int] = {}
    for w in recent:
        for f in w:
            votes[f] = votes.get(f, 0) + 1
    stable = set(current) | {f for f, c in votes.items() if c >= STABILITY_VOTES}
    stable.update(CORE_FEATURES)
    return [f for f in MODEL_FEATURE_KEYS if f in stable]


def _candidate_prediction(name: str, candidate_members: dict, row: dict, elo_hfa: float) -> float:
    """Score one diagnostic candidate on a single row from serialized members.

    The ridge-logistic variants are scored with logistic_logit; every other
    family goes through engine.stack.predict_member. `Multi-model stack` is the
    raw convex combination of the deployable families (before the Elo blend).
    """
    if name == "Elo rating":
        return elo_prob(row, elo_hfa)
    if name == "Multi-model stack":
        p = stack_probability(candidate_members.get("__stack"), row["features"])
        return 0.5 if p is None else p
    member = candidate_members.get(name)
    if member is None:
        return 0.5
    if name.startswith("Logistic regression"):
        return sigmoid(logistic_logit(member, row["features"], None))
    return predict_member(name, member, row["features"])


def _game_line(g: dict) -> str:
    return (
        f"{g['gamePk']}|{g.get('date')}|{g.get('winner')}|"
        f"{(g.get('home') or {}).get('score')}|{(g.get('away') or {}).get('score')}"
    )


def _row_feature_line(row: dict) -> str:
    """Stable feature digest so lineup/log changes invalidate WF caches."""
    values = row.get("features") or {}
    return "|".join(f"{name}:{values.get(name, 0.0)!r}" for name in MODEL_FEATURE_KEYS)


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
    enriched = attach_lineups_as_of(enriched, lineups, cache.load_batter_logs(), pregame_only=True)
    enriched = [g for g in enriched if (g.get("date") or "") < today]
    return compute_elo_and_features(enriched, cache.load_injury_snapshots(), today)["rows"]


def load_selection_days() -> dict:
    """Version-checked per-date walk-forward record: date -> {features, modelChoice}.

    The calibration backtest and on-demand past-date predictions read each
    date's L1-selected features (and, once the calibration pass records it,
    the stack-vs-logistic decision) from here so every dashboard serves the
    identical per-date recipe.
    """
    raw = cache.load_json(WF_SELECTION_FILE, {}) or {}
    if not isinstance(raw, dict) or raw.get("version") != WF_SELECTION_VERSION:
        return {}
    return raw.get("days", {}) or {}


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
    current_model: dict | None = None
    current_choice: dict | None = None
    current_cutoff: str | None = None
    current_features: list[str] | None = None
    current_gate: dict | None = None

    # Rolling hash of the prior-games pointer: prior_games only grows in
    # chronological order, so a running sha256 turns the per-date fingerprint
    # from O(prior games) into O(1) amortized (the full replay previously
    # re-hashed ~6000 games for every one of ~150 dates).
    prior_hash = hashlib.sha256()
    sel_history: list[list[str]] = []
    last_features: list[str] | None = None
    last_gate: dict | None = None
    for i, d in enumerate(dates):
        day_rows = rows_by_date.get(d) or []
        if not day_rows:
            continue
        while ptr < len(fe_rows) and fe_rows[ptr]["game"]["date"] < d:
            prior_games.append(fe_rows[ptr]["game"])
            prior_hash.update(
                f"{_game_line(fe_rows[ptr]['game'])}|{_row_feature_line(fe_rows[ptr])}".encode("utf-8")
            )
            ptr += 1
        if len(prior_games) < MIN_PRIOR_GAMES:
            current_model = None
            current_cutoff = None
            current_features = None
            current_gate = None
            continue
        day_hash = hashlib.sha256(                "\n".join(
                    sorted(
                        f"{_game_line(r['game'])}|{_row_feature_line(r)}"
                        for r in rows_by_date[d]
                    )
                ).encode("utf-8")

        ).hexdigest()
        fp = hashlib.sha256((prior_hash.hexdigest() + "|" + day_hash).encode("utf-8")).hexdigest()
        cached_day = existing.get(d)
        if cached_day and cached_day.get("fp") == fp:
            out[d] = cached_day
            last_features = cached_day.get("features") or last_features
            last_gate = cached_day.get("gate") or last_gate
            current_model = None
            current_cutoff = None
            current_features = None
            current_gate = None
            continue
        if current_model is None or d >= add_days(current_cutoff, WF_SELECTION_REFIT_DAYS):
            # Rolling window: fit on the most recent WF_TRAIN_WINDOW prior games
            # (still strictly before the scored day — no lookahead).
            prior_rows = fe_rows[max(0, ptr - WF_TRAIN_WINDOW):ptr]
            # Nested feature selection: L1 logistic on prior-only rows, λ tuned
            # by holdout Brier, then a stability vote across the last 3 blocks.
            current_features = _stabilize_features(
                _l1_selected_features(prior_rows, list(MODEL_FEATURE_KEYS)), sel_history
            )
            sel_history.append(current_features)
            # Nested model selection: fit the deployable stack AND plain logistic
            # on prior-only rows, then choose per date by holdout Brier.
            current_model, current_choice = fit_walk_forward_step(
                prior_rows, current_features, mlp_epochs=WF_MLP_EPOCHS
            )
            # Tune the abstention layer only on games already inside this
            # prior-only window. The scored day is never part of this fit.
            prior_base_preds = [
                apply_model(current_model, r["features"], r["homeElo"], r["awayElo"])
                for r in prior_rows
            ]
            current_gate = tune_concordance_gate(prior_rows, current_model, prior_base_preds)
            current_cutoff = d
        base_preds = [
            apply_model(current_model, r["features"], r["homeElo"], r["awayElo"])
            for r in day_rows
        ]
        chosen = [p["homeWinProb"] for p in base_preds]
        gate_results = [
            apply_concordance_gate(
                p,
                current_model,
                r["features"],
                r["homeElo"],
                r["awayElo"],
                current_gate,
            )
            for p, r in zip(base_preds, day_rows)
        ]
        cand_members = current_model.get("candidateMembers") or {}
        elo_hfa = current_model.get("eloHfa", 30.0)
        cand_preds = {
            name: [_candidate_prediction(name, cand_members, r, elo_hfa) for r in day_rows]
            for name in CANDIDATE_NAMES
        }
        labels = [r["label"] for r in day_rows]
        feat_vals = {f: [r["features"][f] for r in day_rows] for f in FEATURE_KEYS}
        out[d] = {
            "fp": fp,
            "chosenPreds": chosen,
            "candPreds": cand_preds,
            "labels": labels,
            "featVals": feat_vals,
            "features": current_features,
            "modelChoice": (current_choice or {}).get("deployed"),
            "stackBrier": (current_choice or {}).get("stackBrier"),
            "logisticBrier": (current_choice or {}).get("logisticBrier"),
            "gate": current_gate or default_gate_config(),
            "gateDetails": [
                {k: v for k, v in g.items() if k != "gateSignals"}
                for g in gate_results
            ],
            "gateAccepted": [g["gateAccepted"] for g in gate_results],
            "gateConcordance": [g["concordance"] for g in gate_results],
            "gateSignalCounts": [g["gateSignalCount"] for g in gate_results],
            "gatedCorrect": [
                bool(g["gateAccepted"] and ((r["label"] == 1 and g["gatedPickTeam"] == "home") or
                                             (r["label"] == 0 and g["gatedPickTeam"] == "away")))
                for g, r in zip(gate_results, day_rows)
            ],
        }
        last_features = current_features
        last_gate = current_gate or last_gate
        rep("Walk-forward selection", 30 + int(55 * (i + 1) / max(1, len(dates))),
            f"Evaluated {len(day_rows)} game(s) on {d} against {len(prior_games)} prior game(s)…")

    cache.save_json(WF_SELECTION_FILE, {"version": WF_SELECTION_VERSION, "days": out})

    # Accumulate every cached day's out-of-sample predictions: the per-date
    # chosen model (stack or logistic) plus each diagnostic candidate.
    accum = {
        "chosenPreds": [],
        "candPreds": {name: [] for name in CANDIDATE_NAMES},
        "labels": [],
        "featVals": {f: [] for f in FEATURE_KEYS},
        "gateAccepted": [],
        "gateConcordance": [],
        "gateSignalCounts": [],
        "gatedCorrect": [],
        "stackDays": 0,
        "logisticDays": 0,
    }
    for d in sorted(out):
        day = out[d]
        accum["chosenPreds"].extend(day.get("chosenPreds", []))
        for name in CANDIDATE_NAMES:
            accum["candPreds"][name].extend(day.get("candPreds", {}).get(name, []))
        accum["labels"].extend(day.get("labels", []))
        for f in FEATURE_KEYS:
            accum["featVals"][f].extend(day.get("featVals", {}).get(f, []))
        accum["gateAccepted"].extend(day.get("gateAccepted", []))
        accum["gateConcordance"].extend(day.get("gateConcordance", []))
        accum["gateSignalCounts"].extend(day.get("gateSignalCounts", []))
        accum["gatedCorrect"].extend(day.get("gatedCorrect", []))
        if day.get("modelChoice") == "stack":
            accum["stackDays"] += 1
        elif day.get("modelChoice") == "logistic":
            accum["logisticDays"] += 1

    labels = accum["labels"]
    n_eval = len(labels)
    if n_eval < 20:
        return _fallback_selection(today, len(out), n_eval)

    # Per-candidate out-of-sample diagnostics (the raw families plus the raw
    # multi-model blend). The headline metrics below come from the per-date
    # chosen model, not from any single candidate.
    candidates = []
    for name in CANDIDATE_NAMES:
        preds = accum["candPreds"][name]
        m = evaluate(preds, labels) if preds else {"auc": 0.0, "brier": 0.0, "logLoss": 0.0, "ece": 0.0}
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

    # Descriptive family chosen by the per-date record. The final deployed model
    # is refit on the full history by apply_walk_forward_selection using the
    # same stack-vs-logistic recipe; it overwrites these selected flags with the
    # exact stack members actually served.
    selected_model_name = (
        "Multi-model stack" if accum["stackDays"] >= accum["logisticDays"] else "Logistic regression"
    )
    # The deployed family (stack vs logistic) is chosen per date by holdout
    # Brier, but the diagnostics table must highlight the single family that
    # actually dominated the out-of-sample record — fit on Brier (risk) with
    # AUC as a sub-noise tie-break, so a strictly better holdout model (e.g.
    # the MLP) can no longer lose the "selected" row to a worse one through an
    # AUC deadlock bar.
    single_cands = [c for c in candidates if c["name"] != "Multi-model stack" and c["eligible"]]
    best_single_name = (
        min(single_cands, key=lambda c: (c["brier"], -c["auc"]))["name"]
        if single_cands else selected_model_name
    )
    for c in candidates:
        c["selected"] = c["name"] == best_single_name

    chosen_preds = accum["chosenPreds"]
    chosen_eval = evaluate(chosen_preds, labels)

    # Today's deployed feature set = the FINAL walk-forward block's L1+stability
    # selection (nested: chosen only from games strictly before that date, so no
    # future result influences it). The pooled univariate AUCs below remain as
    # the per-feature importance readout for the monitor.
    selected_features = last_features or [f for f in MODEL_FEATURE_KEYS if f in CORE_FEATURES]
    gate_config = last_gate or default_gate_config()
    gate_accepted = accum["gateAccepted"]
    gated_correct = accum["gatedCorrect"]
    accepted_count = sum(1 for accepted in gate_accepted if accepted)
    gated_wins = sum(1 for accepted, correct in zip(gate_accepted, gated_correct) if accepted and correct)
    base_correct_count = sum(
        1 for p, y in zip(chosen_preds, labels) if (p >= 0.5) == (y == 1)
    )
    gate_diagnostics = {
        "total": n_eval,
        "accepted": accepted_count,
        "coverage": accepted_count / n_eval if n_eval else 0.0,
        "wins": gated_wins,
        "losses": accepted_count - gated_wins,
        "winRate": gated_wins / accepted_count if accepted_count else 0.0,
        "baseWinRate": base_correct_count / n_eval if n_eval else 0.0,
        "lift": (gated_wins / accepted_count - base_correct_count / n_eval) if accepted_count and n_eval else 0.0,
        "meanConcordance": sum(accum["gateConcordance"]) / len(accum["gateConcordance"]) if accum["gateConcordance"] else 0.0,
        "meanSignals": sum(accum["gateSignalCounts"]) / len(accum["gateSignalCounts"]) if accum["gateSignalCounts"] else 0.0,
    }

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

    pick_probs = [max(p, 1 - p) for p in chosen_preds]
    is_correct = [1 if (p >= 0.5) == (y == 1) else 0 for p, y in zip(chosen_preds, labels)]
    high_conf = [c for p, c in zip(pick_probs, is_correct) if p >= 0.65]

    description = (
        "Walk-forward nested selection: per-date L1 (LASSO) features (λ tuned by "
        "holdout Brier) plus a univariate out-of-sample signal screen, a 3-block "
        "stability vote, stack vs logistic chosen per date by holdout Brier, "
        "isotonic-calibrated."
    )

    selection = {
        "trainedThrough": today,
        "daysEvaluated": len(out),
        "gamesEvaluated": n_eval,
        "selectedModel": selected_model_name,
        "modelDescription": description,
        "featureNames": selected_features,
        "featureImportances": feature_importances,
        "candidates": candidates,
        "stackingWeights": [],
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
        "concordanceGate": gate_config,
        "concordanceGateDiagnostics": gate_diagnostics,
        "optimizationParams": {
            "featureSelection": "Per-date L1 (LASSO) + stability vote, λ tuned by holdout Brier",
            "modelSelection": "Stack vs logistic per date by holdout Brier",
            "concordanceGate": "Prior-only threshold/min-signals tuned by Wilson lower bound and conditional win rate",
            "minCandidateAuc": CANDIDATE_MIN_AUC,
            "l1LambdaGrid": list(L1_LAMBDA_GRID),
            "stabilityWindow": STABILITY_K,
            "stabilityVotes": STABILITY_VOTES,
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
        "featureNames": list(MODEL_FEATURE_KEYS),
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
        "concordanceGate": default_gate_config(),
        "concordanceGateDiagnostics": {
            "total": games,
            "accepted": 0,
            "coverage": 0.0,
            "wins": 0,
            "losses": 0,
            "winRate": 0.0,
            "baseWinRate": 0.0,
            "lift": 0.0,
            "meanConcordance": 0.0,
            "meanSignals": 0.0,
        },
        "optimizationParams": {
            "featureSelection": "Full feature set (fallback)",
            "modelSelection": "Stack vs logistic per date by holdout Brier",
            "concordanceGate": "Disabled until prior-only validation history is sufficient",
        },
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
    # Two DISTINCT flags, one concept per row:
    #   * `selected` — the strict best single family by holdout Brier (AUC
    #     tie-break), exactly one row. "Best single" is a rank, not a weight.
    #   * `inStack`   — the fitted families carrying positive weight in the
    #     deployed stack (the allocation vector).
    # A family can be both (best single AND a stack member), but a model that
    # merely dominates the holdout can no longer collide with the deployed
    # family into two simultaneous "best" badges.
    best_single: dict | None = None
    for c in selection.get("candidates", []):
        if c.get("eligible") and c["name"] != "Multi-model stack":
            if best_single is None or (c["brier"], -c["auc"]) < (best_single["brier"], -best_single["auc"]):
                best_single = c
    for c in result["candidates"]:
        c["selected"] = best_single is not None and c["name"] == best_single["name"]
        c["inStack"] = c["name"] in deployed_member_names
    # Single normalized allocation vector across the full candidate pool:
    # deployed stack members carry their (sum-to-1) weights, every other row
    # is an explicit zero — never an implicit 100% fallback.
    result["stackingWeights"] = [
        {"name": c["name"], "weight": roundn((stack.get("weights") or {}).get(c["name"], 0.0), 3)}
        for c in result["candidates"]
    ]
    result["crossValidation"] = selection["crossValidation"]
    result["optimizationParams"] = selection["optimizationParams"]
    result["concordanceGate"] = selection.get("concordanceGate") or default_gate_config()
    result["concordanceGateDiagnostics"] = selection.get("concordanceGateDiagnostics") or {}

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
