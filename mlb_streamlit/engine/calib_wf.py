"""Point-in-time walk-forward calibration rows that REUSE the walk-forward
selection record instead of re-fitting the same block models.

The refresh pipeline replays the season twice back-to-back:

  1. ``build_walk_forward_selection`` fits a per-block model (deployable stack
     + plain logistic + MLP + gate tuning, every ``WF_REFIT_DAYS``) and stores
     every date's out-of-sample predictions, gate results and model choice.
  2. ``build_walk_forward_calibration_rows`` produces the Calibration tab's
     point-in-time view — which used to re-fit those exact same block models
     (``run_model_light``) a second time.

That second replay is what made a refresh visibly \"retrain the walk-forward
models\" again right after the selection pass just trained them. This module
builds the same rows from the selection record's stored predictions + gate
results, fitting only the cheap per-block pieces the record does not carry:
the Poisson run-scoring model and a quick logistic for the run-margin
calibration behind the totals / run-line projections. Dates the selection
record does not cover (cold start, version mismatch, too little history)
fall back to ``run_model_light`` so the path always works.

Cache: results are stored per date in ``calibration_rows_wf.json`` keyed by a
fingerprint of (prior games, day games, feature version, per-date features,
and the selection record's own per-date fingerprint) so new/changed dates are
re-scored incrementally and a change in the L1 selection or gate recipe
invalidates exactly the affected dates.
"""

from __future__ import annotations

import hashlib
import math

from .. import cache
from ..data import (
    add_days,
    attach_as_of_stats,
    attach_lineups_as_of,
    et_date_string,
)
from ..wf_selection import load_selection_days
from .features import FEATURE_KEYS, FEATURE_VERSION, MODEL_FEATURE_KEYS, compute_elo_and_features
from .logistic import train_logistic
from .metrics import logit
from .model import (
    fit_run_margin_calibration,
    run_model_light,
    simulate_runs_batch,
)
from .runs import expected_margin, expected_total, fit_run_model

CALIBRATION_WF_FILE = "calibration_rows_wf.json"
RUN_CALIB_TRIALS = 500
MIN_COMPLETED_GAMES = 40
WF_REFIT_DAYS = 7  # refit the walk-forward model every N days (games in a block share a model)
WF_TRAIN_WINDOW = 2000  # rolling window of most-recent prior games for each walk-forward fit
WF_MLP_EPOCHS = 20  # the MLP fit dominates backtest CPU; cap it for the repeated walk-forward fits
BACKTEST_CACHE_VERSION = cache.BACKTEST_CACHE_VERSION


def _margin_shift(
    run_model_state: dict,
    cal: dict,
    home_id: int,
    away_id: int,
    home_win_prob: float,
) -> float:
    """Shift the two Poisson means so the run model matches the win-prob model."""
    if not cal or (cal.get("slope") == 0 and cal.get("intercept") == 0):
        return 0.0
    base_margin = expected_margin(run_model_state, home_id, away_id)
    p = min(0.99, max(0.01, home_win_prob))
    target = cal["intercept"] + cal["slope"] * logit(p)
    return (target - base_margin) / 2


def _rows_from_stored_selection(
    day_rows: list[dict],
    stored_preds: list[float],
    stored_gate: list[dict],
    run_model_state: dict,
    run_margin_cal: dict,
) -> list[dict]:
    """Calibration rows built from the walk-forward SELECTION record.

    The selection pass already scored every game point-in-time and stored the
    out-of-sample home-win probabilities + full gate results per date. This
    rebuilds the calibration row shape around those stored predictions and
    adds the totals / run-line projections from the block's Poisson run model
    + margin calibration — no MLP/stack fit is needed.
    """
    home_ids = [r["game"]["home"]["id"] for r in day_rows]
    away_ids = [r["game"]["away"]["id"] for r in day_rows]
    lines = [expected_total(run_model_state, h, a) for h, a in zip(home_ids, away_ids)]
    shifts = [
        _margin_shift(run_model_state, run_margin_cal, h, a, p)
        for h, a, p in zip(home_ids, away_ids, stored_preds)
    ]
    projs = simulate_runs_batch(run_model_state, home_ids, away_ids, lines, shifts, RUN_CALIB_TRIALS)
    out: list[dict] = []
    for r, p, gate, proj in zip(day_rows, stored_preds, stored_gate, projs):
        g = r["game"]
        winner = g.get("winner")
        pick_home = p >= 0.5
        gated_team = gate.get("gatedPickTeam")
        out.append({
            "gamePk": g["gamePk"],
            "date": g["date"],
            "away": {"abbrev": g["away"]["abbrev"], "name": g["away"]["name"], "score": g["away"].get("score")},
            "home": {"abbrev": g["home"]["abbrev"], "name": g["home"]["name"], "score": g["home"].get("score")},
            "winner": winner,
            "pickTeam": "home" if pick_home else "away",
            "pickProb": p if pick_home else 1 - p,
            "homeWinProb": p,
            "isCorrect": pick_home == (winner == "home"),
            "isUpset": pick_home != (winner == "home"),
            **gate,
            "gatedIsCorrect": bool(
                gate.get("gateAccepted") and winner in ("home", "away") and gated_team == winner
            ),
            "predictedTotal": proj["total"],
            "homeRunLineProb": proj["homeRunLineProb"],
            "actualTotal": (g["away"].get("score") or 0) + (g["home"].get("score") or 0),
            "actualMargin": (g["home"].get("score") or 0) - (g["away"].get("score") or 0),
        })
    return out


def build_walk_forward_calibration_rows_v2(
    report=None,
    feature_names: list[str] | None = None,
    rows: list[dict] | None = None,
) -> list[dict]:
    """Strict walk-forward calibration rows — the true point-in-time backtest.

    Every completed game in the **current season** on date D is scored with a
    model trained ONLY on games strictly before D. The win probabilities, gate
    results and labels come straight from the walk-forward selection record
    (which already fit those models); only the run-scoring projections are
    computed here, from a cheap per-block Poisson model + quick logistic.

    Dates with fewer than MIN_COMPLETED_GAMES prior games are skipped, and
    per-date results are cached keyed by a fingerprint so later calls re-score
    only new/changed dates. When the selection record does not cover a date,
    the block model is fit directly with ``run_model_light`` (same recipe as
    the selection pass) so the path works even before the first refresh.
    """
    def rep(stage, pct, msg):
        if report:
            report(stage, pct, msg)

    today = et_date_string()
    if feature_names is None:
        state = cache.load_model_state()
        feature_names = (state or {}).get("featureNames") or list(MODEL_FEATURE_KEYS)
    feature_names = list(feature_names)
    if rows is not None:
        # Reuse the already-computed chronological feature rows (they include
        # today's completed games; the backtest only scores past dates).
        fe_rows = [r for r in rows if (r["game"].get("date") or "") < today]
        rows_by_date: dict[str, list[dict]] = {}
        by_date: dict[str, list[dict]] = {}
        for r in fe_rows:
            rows_by_date.setdefault(r["game"]["date"], []).append(r)
            by_date.setdefault(r["game"]["date"], []).append(r["game"])
    else:
        cached = cache.load_games()
        completed = [
            g for g in cached
            if g.get("winner") in ("home", "away")
            and (g.get("home") or {}).get("id") and (g.get("away") or {}).get("id")
        ]
        pitcher_log_cache = cache.load_pitcher_logs()
        team_log_cache = cache.load_team_logs()
        enriched = attach_as_of_stats(completed, pitcher_log_cache, team_log_cache)
        lineups = {}
        for pk, lu in (cache.load_lineups() or {}).items():
            if lu:
                try:
                    lineups[int(pk)] = lu
                except (TypeError, ValueError):
                    pass
        enriched = attach_lineups_as_of(enriched, lineups, cache.load_batter_logs(), pregame_only=False)
        enriched = [g for g in enriched if (g.get("date") or "") < today]
        if not enriched:
            return []
        fe = compute_elo_and_features(enriched, cache.load_injury_snapshots(), today)
        fe_rows = fe["rows"]
        rows_by_date = {}
        for r in fe_rows:
            rows_by_date.setdefault(r["game"]["date"], []).append(r)
        by_date = {}
        for g in enriched:
            by_date.setdefault(g["date"], []).append(g)

    if not fe_rows:
        return []

    existing_raw = cache.load_json(CALIBRATION_WF_FILE, {}) or {}
    existing = existing_raw.get("days", {}) if isinstance(existing_raw, dict) and "days" in existing_raw else {}
    if (existing_raw.get("version") if isinstance(existing_raw, dict) else None) != BACKTEST_CACHE_VERSION:
        existing = {}
    # Per-date L1-selected features + stack-vs-logistic choice from the walk-
    # forward selection record, so every date is scored by ITS point-in-time
    # feature set and model family (nested selection).
    sel_days = load_selection_days()
    out: dict[str, dict] = {}
    season = today[:4]
    dates = sorted(d for d in by_date if d[:4] == season)

    # Block cadence: reuse one block state for WF_REFIT_DAYS of dates. The
    # prior-games pointer advances once through the chronological feature rows
    # (amortized O(n) instead of an O(n) rescan per date).
    current_block: dict | None = None  # {cutoff, stored, model?, runModel, runMarginCal}
    prior_games: list[dict] = []
    ptr = 0
    prior_hash = hashlib.sha256()

    current_model_choice: str | None = None
    for i, d in enumerate(dates):
        day_rows = rows_by_date.get(d) or []
        if not day_rows:
            continue
        sel_day = sel_days.get(d) or {}
        feats_d = sel_day.get("features") or feature_names
        choice_d = sel_day.get("modelChoice") or None
        while ptr < len(fe_rows) and fe_rows[ptr]["game"]["date"] < d:
            prior_games.append(fe_rows[ptr]["game"])
            prior_hash.update(
                f"{fe_rows[ptr]['game']['gamePk']}|{fe_rows[ptr]['game'].get('date')}|"
                f"{fe_rows[ptr]['game'].get('winner')}|"
                f"{(fe_rows[ptr]['game'].get('home') or {}).get('score')}|"
                f"{(fe_rows[ptr]['game'].get('away') or {}).get('score')}".encode("utf-8")
            )
            ptr += 1
        if len(prior_games) < MIN_COMPLETED_GAMES:
            current_block = None
            continue
        day_hash = hashlib.sha256(
            "\n".join(
                sorted(
                    f"{g['gamePk']}|{g.get('date')}|{g.get('winner')}|"
                    f"{(g.get('home') or {}).get('score')}|{(g.get('away') or {}).get('score')}"
                    for g in by_date[d]
                )
            ).encode("utf-8")
        ).hexdigest()
        # The per-date feature set AND the selection record's own fingerprint
        # participate: a change in the L1 selection, the gate recipe or the
        # stored predictions invalidates that date's cached rows.
        fp = hashlib.sha256(
            (f"{prior_hash.hexdigest()}|{day_hash}|featureVersion:{FEATURE_VERSION}"
             f"|feats:{','.join(sorted(feats_d))}|selFp:{sel_day.get('fp', '')}").encode("utf-8")
        ).hexdigest()
        cached_day = existing.get(d)
        if cached_day and cached_day.get("fp") == fp:
            out[d] = cached_day
            current_block = None
            continue
        stored_preds = sel_day.get("chosenPreds") or []
        stored_gate = sel_day.get("gateDetails") or []
        stored_ok = len(stored_preds) == len(day_rows) and len(stored_gate) == len(day_rows)
        # Cache miss: fit the block state once, then reuse it for the next
        # WF_REFIT_DAYS of dates. Each fit trains on the most recent
        # WF_TRAIN_WINDOW prior games (still strictly before the scored day).
        if current_block is None or d >= add_days(current_block["cutoff"], WF_REFIT_DAYS):
            prior_rows = fe_rows[max(0, ptr - WF_TRAIN_WINDOW):ptr]
            if stored_ok:
                # Reuse the selection record: only the run-scoring model and a
                # quick logistic for the margin calibration are fit here.
                lr_quick = train_logistic(prior_rows, feats_d)
                quick_model = {
                    "featureNames": feats_d,
                    "weights": lr_quick["weights"],
                    "bias": lr_quick["bias"],
                    "featureStats": lr_quick["featureStats"],
                    "isotonicPoints": [],
                    "eloHfa": 30.0,
                    "blendW": 0.0,
                    "stack": {
                        "members": {"Logistic regression": lr_quick},
                        "weights": {"Logistic regression": 1.0},
                    },
                }
                current_block = {
                    "cutoff": d,
                    "stored": True,
                    "model": quick_model,
                    "runModel": fit_run_model(prior_games),
                    "runMarginCal": fit_run_margin_calibration(prior_rows, quick_model),
                }
            else:
                result = run_model_light(
                    prior_rows, prior_games, season, d, feats_d, mlp_epochs=WF_MLP_EPOCHS,
                    model_choice=choice_d,
                )
                current_block = {
                    "cutoff": d,
                    "stored": False,
                    "model": {
                        "featureNames": result["featureNames"],
                        "weights": result["weights"],
                        "bias": result["bias"],
                        "featureStats": result["featureStats"],
                        "isotonicPoints": result["isotonicPoints"],
                        "monteCarloSigma": result["monteCarloSigma"],
                        "monteCarloEnabled": result["monteCarloEnabled"],
                        "eloHfa": result["eloHfa"],
                        "blendW": result.get("blendW", 0.0),
                        "stack": result.get("stack", {}),
                    },
                    "runModel": result["runModel"],
                    "runLineIso": result["runLineCalibration"],
                    "runMarginCal": result["runMarginCalibration"],
                }
                current_model_choice = (result.get("modelChoice") or {}).get("deployed")
        if current_block["stored"] and stored_ok:
            cal_rows = _rows_from_stored_selection(
                day_rows,
                stored_preds,
                stored_gate,
                current_block["runModel"],
                current_block["runMarginCal"],
            )
            current_model_choice = sel_day.get("modelChoice") or current_model_choice
        else:
            cal_rows = _build_calibration_rows_fallback(
                day_rows,
                current_block["model"],
                current_block["runModel"],
                current_block.get("runLineIso") or [],
                current_block["runMarginCal"],
            )
        for r in cal_rows:
            r["trainedThrough"] = current_block["cutoff"]
            r["modelChoice"] = current_model_choice
        out[d] = {"fp": fp, "rows": cal_rows, "modelChoice": current_model_choice}
        rep("Walk-forward", 30 + int(65 * (i + 1) / max(1, len(dates))),
            f"Scored {len(cal_rows)} game(s) on {d} with a model trained on {len(prior_games)} prior game(s)…")

    cache.save_json(CALIBRATION_WF_FILE, {"version": BACKTEST_CACHE_VERSION, "days": out})
    # Record each date's deployed family in the walk-forward selection record so
    # the Model Monitor can show the per-date stack-vs-logistic decision.
    sel_raw = cache.load_json("walk_forward_selection.json", {}) or {}
    sel_days_out = sel_raw.get("days", {}) if isinstance(sel_raw, dict) else {}
    changed = False
    for d in out:
        choice = out[d].get("modelChoice")
        if choice and sel_days_out.get(d, {}).get("modelChoice") != choice:
            sel_days_out.setdefault(d, {})["modelChoice"] = choice
            changed = True
    if changed:
        sel_raw = dict(sel_raw)
        sel_raw["days"] = sel_days_out
        cache.save_json("walk_forward_selection.json", sel_raw)
    flat: list[dict] = []
    for d in sorted(out):
        flat.extend(out[d]["rows"])
    rep("Walk-forward", 100, f"Walk-forward calibration ready ({len(flat)} games scored point-in-time).")
    return flat


def _build_calibration_rows_fallback(
    day_rows: list[dict],
    model: dict,
    run_model_state: dict,
    run_line_iso: list[dict],
    run_margin_cal: dict,
) -> list[dict]:
    """Per-game calibration rows for the run_model_light fallback path.

    Mirrors refresh.build_calibration_rows for dates the walk-forward selection
    record does not cover: apply the deployed block model, attach the date's
    own gate recipe, and project the totals / run line.
    """
    from .gating import apply_concordance_gate, default_gate_config
    from .model import apply_model

    selection_raw = cache.load_json("walk_forward_selection.json", {}) or {}
    selection_days = selection_raw.get("days", {}) if isinstance(selection_raw, dict) else {}
    games = [r["game"] for r in day_rows]
    home_ids = [g["home"]["id"] for g in games]
    away_ids = [g["away"]["id"] for g in games]
    preds = [apply_model(model, r["features"], r["homeElo"], r["awayElo"]) for r in day_rows]
    projs = simulate_runs_batch(
        run_model_state,
        home_ids,
        away_ids,
        [expected_total(run_model_state, h, a) for h, a in zip(home_ids, away_ids)],
        [_margin_shift(run_model_state, run_margin_cal, h, a, p["homeWinProb"]) for h, a, p in zip(home_ids, away_ids, preds)],
        RUN_CALIB_TRIALS,
    )
    out: list[dict] = []
    for r, pred, proj in zip(day_rows, preds, projs):
        g = r["game"]
        winner = g.get("winner")
        row_gate_config = (
            (selection_days.get(g["date"], {}) or {}).get("gate")
            or default_gate_config()
        )
        gate = apply_concordance_gate(
            pred, model, r["features"], r["homeElo"], r["awayElo"], row_gate_config
        )
        compact_gate = {k: v for k, v in gate.items() if k != "gateSignals"}
        gated_correct = (
            gate["gatedPickTeam"] == winner
            if gate["gateAccepted"] and winner in ("home", "away")
            else None
        )
        out.append({
            "gamePk": g["gamePk"],
            "date": g["date"],
            "away": {"abbrev": g["away"]["abbrev"], "name": g["away"]["name"], "score": g["away"].get("score")},
            "home": {"abbrev": g["home"]["abbrev"], "name": g["home"]["name"], "score": g["home"].get("score")},
            "winner": winner,
            "pickTeam": pred["pickTeam"],
            "pickProb": pred["pickProb"],
            "homeWinProb": pred["homeWinProb"],
            "isCorrect": pred["pickTeam"] == winner,
            "isUpset": pred["pickTeam"] != winner,
            **compact_gate,
            "gatedIsCorrect": gated_correct,
            "predictedTotal": proj["total"],
            "homeRunLineProb": proj["homeRunLineProb"],
            "actualTotal": (g["away"].get("score") or 0) + (g["home"].get("score") or 0),
            "actualMargin": (g["home"].get("score") or 0) - (g["away"].get("score") or 0),
        })
    return out
