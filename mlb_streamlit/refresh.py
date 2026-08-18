"""Refresh pipeline — the Python equivalent of mlbActions.refreshModel.

Fetch schedule/boxscores/stats from the single MLB Stats API source, train
and calibrate the model, score the upcoming window, and persist everything
to the disk cache. All functions are pure stdlib so the pipeline can be
tested headlessly (`python3 mlb_streamlit/scripts/smoke_test.py`).
"""

from __future__ import annotations

import datetime as _dt
from concurrent.futures import ThreadPoolExecutor

from . import cache
from .data import (
    RECENT_WINDOW_DAYS,
    SEASON_START_MD,
    UPCOMING_WINDOW_DAYS,
    add_days,
    attach_as_of_stats,
    attach_lineups_as_of,
    enrich_with_matchups,
    et_date_string,
    fetch_all_seasons,
    fetch_batter_game_logs,
    fetch_current_injury_snapshot,
    fetch_injury_snapshots,
    fetch_lineups_for_games,
    fetch_market_odds,
    fetch_pitcher_game_logs,
    fetch_schedule_range,
    fetch_team_game_logs,
    market_odds_enabled,
    market_odds_for_game,
)
from .engine.features import build_features_for_game
from .engine.metrics import (
    apply_isotonic,
    calibration_curve_points,
    compute_auc,
    compute_brier,
    evaluate,
    logit,
    spearman_rank,
)
from .engine.model import apply_model, run_model
from .engine.runs import expected_margin, expected_total, simulate_runs

RUN_SIM_TRIALS = 10000
RUN_CALIB_TRIALS = 500
MIN_COMPLETED_GAMES = 40

SEASON_START = "2022-03-15"  # earliest calibration-range date (UI default)


# ---------------------------------------------------------------------------
# Run projections
# ---------------------------------------------------------------------------

def margin_shift_for_game(
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


def build_run_projection(
    run_model_state: dict,
    run_line_iso: list[dict],
    game: dict,
    market_total: float | None = None,
    market_run_line: float | None = None,
    trials: int = RUN_SIM_TRIALS,
    home_win_prob: float = 0.5,
    run_margin_cal: dict | None = None,
) -> dict:
    run_line = market_run_line or 1.5
    margin_shift = margin_shift_for_game(
        run_model_state,
        run_margin_cal or {},
        game["home"]["id"],
        game["away"]["id"],
        home_win_prob,
    )
    line = market_total if market_total is not None else expected_total(run_model_state, game["home"]["id"], game["away"]["id"])
    sim = simulate_runs(run_model_state, game["home"]["id"], game["away"]["id"], line, trials, run_line, margin_shift)
    home_rl = apply_isotonic(run_line_iso, sim["homeRunLineProb"]) if run_line_iso and run_line == 1.5 else sim["homeRunLineProb"]
    home = min(0.999, max(0.001, home_rl))
    return {
        "homeScore": round(sim["homeScore"], 2),
        "awayScore": round(sim["awayScore"], 2),
        "total": round(sim["total"], 2),
        "overProb": sim["overProb"],
        "underProb": sim["underProb"],
        "homeRunLineProb": home,
        "awayRunLineProb": 1 - home,
    }


# ---------------------------------------------------------------------------
# Calibration rows / summary
# ---------------------------------------------------------------------------

def build_calibration_rows(
    rows: list[dict],
    model: dict,
    run_model_state: dict,
    run_line_iso: list[dict],
    run_margin_cal: dict,
) -> list[dict]:
    """Compact per-game calibration projection (pre-game prediction vs result)."""
    from .engine.model import simulate_runs_batch

    out: list[dict] = []
    games = [r["game"] for r in rows]
    home_ids = [g["home"]["id"] for g in games]
    away_ids = [g["away"]["id"] for g in games]
    preds = [apply_model(model, r["features"], r["homeElo"], r["awayElo"]) for r in rows]
    projs = simulate_runs_batch(
        run_model_state,
        home_ids,
        away_ids,
        [expected_total(run_model_state, h, a) for h, a in zip(home_ids, away_ids)],
        [
            margin_shift_for_game(run_model_state, run_margin_cal, h, a, p["homeWinProb"])
            for h, a, p in zip(home_ids, away_ids, preds)
        ],
        RUN_CALIB_TRIALS,
    )
    for r, pred, proj in zip(rows, preds, projs):
        g = r["game"]
        winner = g.get("winner")
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
            "predictedTotal": proj["total"],
            "homeRunLineProb": proj["homeRunLineProb"],
            "actualTotal": (g["away"].get("score") or 0) + (g["home"].get("score") or 0),
            "actualMargin": (g["home"].get("score") or 0) - (g["away"].get("score") or 0),
        })
    return out


def build_calibration_summary(rows: list[dict]) -> dict:
    preds = [r["pickProb"] for r in rows]
    labels = [1 if r["isCorrect"] else 0 for r in rows]
    ev = evaluate(preds, labels)
    curve = calibration_curve_points(preds, labels, 8)
    metrics = {
        "auc": ev["auc"],
        "brier": ev["brier"],
        "logLoss": ev["logLoss"],
        "ece": ev["ece"],
        "bins": ev["bins"],
        "confidenceDistribution": ev["confidenceDistribution"],
        "calibrationCurve": curve if curve else ev["calibrationCurve"],
    }
    t_n = 0
    t_abs = 0.0
    t_sq = 0.0
    t_bias = 0.0
    rl_preds: list[float] = []
    rl_labels: list[int] = []
    for r in rows:
        if r.get("predictedTotal") is not None:
            t_n += 1
            err = r["predictedTotal"] - r["actualTotal"]
            t_abs += abs(err)
            t_sq += err * err
            t_bias += err
        if r.get("homeRunLineProb") is not None:
            rl_preds.append(r["homeRunLineProb"])
            rl_labels.append(1 if r["actualMargin"] >= 2 else 0)
    totals_metrics = {
        "n": t_n,
        "mae": t_abs / t_n if t_n > 0 else 0,
        "rmse": (t_sq / t_n) ** 0.5 if t_n > 0 else 0,
        "bias": t_bias / t_n if t_n > 0 else 0,
    }
    run_line_metrics = {"n": len(rl_preds), "auc": 0, "brier": 0, "accuracy": 0}
    if rl_preds:
        run_line_metrics = {
            "n": len(rl_preds),
            "auc": compute_auc(rl_preds, rl_labels),
            "brier": compute_brier(rl_preds, rl_labels),
            "accuracy": sum(1 for p, y in zip(rl_preds, rl_labels) if (1 if p >= 0.5 else 0) == y) / len(rl_preds),
        }
    total = len(rows)
    correct = sum(1 for r in rows if r["isCorrect"])
    return {
        "metrics": metrics,
        "totalsMetrics": totals_metrics,
        "runLineMetrics": run_line_metrics,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
    }


def build_todays_record(rows_today: list[dict], today: str) -> dict:
    completed = [r for r in rows_today if r["isCorrect"] is not None]
    total = len(completed)
    correct = sum(1 for r in completed if r["isCorrect"])
    upsets = []
    for r in completed:
        if not r["isUpset"]:
            continue
        winner_side = "home" if r["winner"] == "home" else "away"
        loser_side = "away" if winner_side == "home" else "home"
        prob = r["homeWinProb"] if winner_side == "home" else 1 - r["homeWinProb"]
        upsets.append({
            "team": r[winner_side]["abbrev"],
            "loser": r[loser_side]["abbrev"],
            "prob": round(prob * 100),
        })
    return {
        "date": today,
        "total": total,
        "completed": total,
        "wins": correct,
        "losses": total - correct,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "upsets": upsets,
    }


# ---------------------------------------------------------------------------
# Game docs
# ---------------------------------------------------------------------------

def build_game_doc(
    game: dict,
    pred: dict,
    injuries: dict | None,
    run_projection: dict | None,
    market_odds: dict | None,
) -> dict:
    # Pitcher/team stats are already attached as-of the game's own date by
    # attach_as_of_stats — never override them with a flat full-season dict.
    season = game.get("season")

    doc = {
        "gamePk": game["gamePk"],
        "date": game["date"],
        "status": game["status"],
        "detailedState": game.get("detailedState"),
        "dayNight": game["dayNight"],
        "gameDate": game["gameDate"],
        "innings": game.get("innings"),
        "venue": game.get("venue"),
        "away": game["away"],
        "home": game["home"],
        "awayPitcher": game.get("awayPitcher"),
        "homePitcher": game.get("homePitcher"),
        "winner": game.get("winner"),
        "homeWinProb": pred["homeWinProb"],
        "awayWinProb": pred["awayWinProb"],
        "pickTeam": pred["pickTeam"],
        "pickProb": pred["pickProb"],
        "edge": pred["edge"],
        "fairHomeOdds": pred["fairHomeOdds"],
        "fairAwayOdds": pred["fairAwayOdds"],
        "shap": pred["shap"],
        "homeInjuries": injuries["home"] if injuries else None,
        "awayInjuries": injuries["away"] if injuries else None,
        "season": season,
        "weather": game.get("weather"),
        "lineups": game.get("lineups"),
        "lineupStats": game.get("lineupStats"),
        "runProjection": run_projection,
        "marketOdds": market_odds,
    }
    if game.get("winner") in ("home", "away"):
        doc["isCorrect"] = pred["pickTeam"] == game["winner"]
        doc["isUpset"] = pred["pickTeam"] != game["winner"]
    return doc


def predict_for_game(game: dict, model: dict, team_state: dict) -> dict:
    return apply_model(
        model,
        build_features_for_game(game, team_state),
        team_state["elo"].get(game["home"]["id"], 1500),
        team_state["elo"].get(game["away"]["id"], 1500),
    )


def merge_pitcher(fresh: dict | None, stored: dict | None) -> dict | None:
    if not fresh:
        return stored
    if not stored:
        return fresh
    if fresh["id"] != stored["id"]:
        return fresh  # probable starter changed
    return {
        **fresh,
        "era": fresh.get("era") if fresh.get("era") is not None else stored.get("era"),
        "k9": fresh.get("k9") if fresh.get("k9") is not None else stored.get("k9"),
        "fip": fresh.get("fip") if fresh.get("fip") is not None else stored.get("fip"),
    }


def merge_raw_with_stored(fresh: dict, stored: dict) -> dict:
    return {
        **fresh,
        "awayPitcher": merge_pitcher(fresh.get("awayPitcher"), stored.get("awayPitcher")),
        "homePitcher": merge_pitcher(fresh.get("homePitcher"), stored.get("homePitcher")),
    }


# ---------------------------------------------------------------------------
# Model-state reconstruction (for on-demand date predictions)
# ---------------------------------------------------------------------------

def reconstruct_model(state: dict) -> dict:
    return {
        "featureNames": state["featureNames"],
        "weights": state["weights"],
        "bias": state["bias"],
        "featureStats": state["featureStats"],
        "isotonicPoints": state.get("isotonicPoints") or [],
        "monteCarloSigma": state.get("monteCarloSigma") or 0,
        "monteCarloEnabled": state.get("monteCarloEnabled") or False,
        "eloHfa": state.get("eloHfa") or 30,
    }


def reconstruct_team_state(rankings: list[dict]) -> dict:
    elo = {}
    form = {}
    last_game_date = {}
    records = {}
    injuries = {}
    for p in rankings:
        elo[p["teamId"]] = p["elo"]
        form[p["teamId"]] = p.get("last10WinPct") or 0.5
        last_game_date[p["teamId"]] = p.get("lastGameDate") or ""
        records[p["teamId"]] = {"wins": p.get("wins") or 0, "losses": p.get("losses") or 0}
        injuries[p["teamId"]] = p.get("injuries") or 0
    return {"elo": elo, "form": form, "lastGameDate": last_game_date, "records": records, "injuries": injuries}


# ---------------------------------------------------------------------------
# The refresh action
# ---------------------------------------------------------------------------

def run_refresh(
    report=None,
    force_full: bool = False,
    history_start: str | None = None,
) -> dict:
    """Full refresh: fetch -> enrich -> train -> predict -> persist.

    `report(stage, pct, message)` receives progress updates (used by the
    Streamlit progress bar). Returns a summary dict for the UI.
    """
    def rep(stage, pct, message):
        if report:
            report(stage, pct, message)

    now = _dt.datetime.now()
    season = str(now.year)
    today = et_date_string(now)

    rep("Reading cached data", 4, "Loading previously stored games…")
    cached_games = cache.load_games()
    cached_state = cache.load_model_state()
    injury_cache = cache.load_injury_snapshots()

    completed_cached = [g for g in cached_games if g.get("winner") in ("home", "away")]
    has_history = force_full or len(completed_cached) >= MIN_COMPLETED_GAMES

    # 1. Fetch the schedule.
    if has_history:
        rep("Fetching fresh games", 12, "Loading recent results + upcoming window…")
        fresh = fetch_schedule_range(add_days(today, -RECENT_WINDOW_DAYS), add_days(today, UPCOMING_WINDOW_DAYS))
    else:
        seasons = ["2024", "2025", season]
        rep(f"Fetching seasons 1/{len(seasons)}", 12, "First run: loading game history…")
        fresh = fetch_all_seasons(seasons, season, today)
    rep("Fetching schedule", 20, f"Loaded {len(fresh)} fresh game(s)…")

    # 2. Merge fresh over cached.
    by_pk = {g["gamePk"]: g for g in cached_games}
    for g in fresh:
        stored = by_pk.get(g["gamePk"])
        by_pk[g["gamePk"]] = merge_raw_with_stored(g, stored) if stored else g
    all_games = sorted(by_pk.values(), key=lambda g: g["gameDate"] or g["date"])
    completed = [g for g in all_games if g.get("winner") in ("home", "away")]
    if len(completed) < MIN_COMPLETED_GAMES:
        raise RuntimeError(
            f"Only {len(completed)} completed regular-season games found. Cannot train yet."
        )

    # 3-6. Fetch the independent families concurrently. Pitcher game logs, team
    #    game logs, boxscore lineups and injury snapshots depend only on the
    #    schedule, so they run in one I/O-bound wave (each family keeps its own
    #    internal thread pool). Batter game logs depend on the lineups, so they
    #    run in a second wave once the lineups are in.
    rep("Fetching game data", 24, "Loading pitcher/team logs, lineups & injuries…")
    pitcher_log_cache = cache.load_pitcher_logs()
    team_log_cache = cache.load_team_logs()
    lineup_cache = cache.load_lineups()

    pitcher_pairs = []
    seen_p = set()
    for g in all_games:
        s = g.get("season") or season
        for p in (g.get("awayPitcher"), g.get("homePitcher")):
            if p and p.get("id"):
                key = f"{p['id']}|{s}"
                if key not in seen_p:
                    seen_p.add(key)
                    pitcher_pairs.append({"id": p["id"], "season": s})
    to_fetch_pitcher = [p for p in pitcher_pairs if f"{p['id']}|{p['season']}" not in pitcher_log_cache]

    team_pairs = {}
    for g in all_games:
        s = g.get("season") or season
        team_pairs.setdefault(f"{g['home']['id']}|{s}", {"id": g["home"]["id"], "season": s})
        team_pairs.setdefault(f"{g['away']['id']}|{s}", {"id": g["away"]["id"], "season": s})
    to_fetch_team = [p for p in team_pairs.values() if f"{p['id']}|{p['season']}" not in team_log_cache]

    lineup_targets = [g for g in all_games if g["gamePk"] not in lineup_cache]
    team_ids = sorted({tid for g in completed for tid in (g["home"]["id"], g["away"]["id"]) if tid > 0})

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_pitcher = pool.submit(fetch_pitcher_game_logs, to_fetch_pitcher, pitcher_log_cache)
        f_team = pool.submit(fetch_team_game_logs, to_fetch_team, team_log_cache)
        f_lineups = pool.submit(fetch_lineups_for_games, lineup_targets, 16)
        f_injuries = pool.submit(
            fetch_injury_snapshots, team_ids, season, f"{season}-{SEASON_START_MD}", today, injury_cache
        )
        f_current_injury = pool.submit(fetch_current_injury_snapshot, team_ids, today, season)
        fresh_pitcher_logs = f_pitcher.result()
        fresh_team_logs = f_team.result()
        fetched_lineups = f_lineups.result()
        injury_snapshots = f_injuries.result()
        current_injury = f_current_injury.result()

    pitcher_log_cache.update(fresh_pitcher_logs)
    team_log_cache.update(fresh_team_logs)
    for g in lineup_targets:
        lu = fetched_lineups.get(g["gamePk"])
        if lu:
            lineup_cache[g["gamePk"]] = lu
        elif g.get("winner") in ("home", "away"):
            # Completed game with no posted lineups — final, never refetch.
            lineup_cache[g["gamePk"]] = None
    lineups = {pk: lu for pk, lu in lineup_cache.items() if lu}
    for tid, count in current_injury.items():
        lst = injury_snapshots.setdefault(str(tid), [])
        if not lst or lst[-1]["date"] != today:
            lst.append({"date": today, "count": count})
    injury_cache.update(injury_snapshots)
    rep(
        "Game data loaded",
        40,
        f"Pitcher {len(pitcher_log_cache)} · team {len(team_log_cache)} · "
        f"lineups {sum(1 for v in lineups.values() if v)} · IL {len(team_ids)} teams",
    )

    enriched = attach_as_of_stats(all_games, pitcher_log_cache, team_log_cache)

    # 6b. Batter game logs: one fetch per {batter, season}. Each batter's OPS is
    #    then accumulated as-of the game's own date, so neither a batter's future
    #    games nor a later season's numbers ever leak into a training row. This
    #    depends on the lineups just fetched, hence the second wave.
    rep("Fetching batter game logs", 44, "Loading per-game batting history…")
    batter_log_cache = cache.load_batter_logs()
    batter_ids_by_season: dict[str, set[int]] = {}
    for g in enriched:
        lu = lineups.get(g["gamePk"])
        if not lu:
            continue
        s = g.get("season") or season
        for side in (lu.get("home"), lu.get("away")):
            if not side:
                continue
            for p in side["battingOrder"] + side["bench"]:
                batter_ids_by_season.setdefault(s, set()).add(p["id"])
    for s, ids in batter_ids_by_season.items():
        fresh_logs = fetch_batter_game_logs(sorted(ids), s, batter_log_cache)
        batter_log_cache.update(fresh_logs)
    rep("Batter game logs loaded", 48, f"Cached {len(batter_log_cache)} batter-season logs.")
    enriched = attach_lineups_as_of(enriched, lineups, batter_log_cache)

    # 6c. Matchup edges (BvP / platoon / vs-team): only for games in the fresh
    #    window (real boxscore lineup + opposing starter both known), mirroring
    #    the React engine's enrichWithMatchups — no lookahead and no junk
    #    values on older history.
    rep("Loading matchups", 54, "Fetching BvP & platoon splits…")
    window_start = add_days(today, -2)
    window_end = add_days(today, UPCOMING_WINDOW_DAYS)
    window_games = [g for g in enriched if window_start <= (g.get("date") or "") <= window_end]
    matchup = enrich_with_matchups(
        window_games,
        batter_log_cache,
        cache.load_bvp_logs(),
        cache.load_platoon_logs(),
        cache.load_vs_team_logs(),
        season,
    )
    enriched_by_pk = {g["gamePk"]: g for g in matchup["games"]}
    enriched = [enriched_by_pk.get(g["gamePk"], g) for g in enriched]
    bvp_log_cache = matchup["bvpLogs"]
    platoon_log_cache = matchup["platoonLogs"]
    vs_team_log_cache = matchup["vsTeamLogs"]
    pitcher_hands_cache = matchup["pitcherHands"]
    rep(
        "Matchups loaded",
        58,
        f"BvP {len(bvp_log_cache)} · platoon {len(platoon_log_cache)} · vsTeam {len(vs_team_log_cache)}",
    )

    completed_enriched = [g for g in enriched if g.get("winner") in ("home", "away")]

    # 7. Train / calibrate / select.
    rep("Training model", 62, "Fitting models & selecting features…")
    run = run_model(completed_enriched, season, today, injury_snapshots)
    result = run["result"]
    model = run["model"]
    rows = run["rows"]
    team_state = run["teamState"]
    run_model_state = result["runModel"]
    run_line_iso = result["runLineCalibration"] or []
    run_margin_cal = result["runMarginCalibration"] or {"slope": 0, "intercept": 0}

    # 8. Market odds (best-effort).
    rep("Fetching market odds", 68, "Loading market odds (best-effort)…")
    market_odds = fetch_market_odds()
    market_odds_status = {
        "enabled": market_odds_enabled(),
        "count": len(market_odds),
        "fetchedAt": int(now.timestamp() * 1000),
    }

    # 9. Calibration rows for every completed game (as-of-time predictions).
    rep("Scoring games", 74, "Generating predictions & run simulations…")
    calibration_rows = build_calibration_rows(rows, model, run_model_state, run_line_iso, run_margin_cal)

    full_preds = [r["pickProb"] for r in calibration_rows]
    full_labels = [1 if r["isCorrect"] else 0 for r in calibration_rows]
    full_curve = calibration_curve_points(full_preds, full_labels, 12)
    full_eval = evaluate(full_preds, full_labels)
    spearman = spearman_rank(full_preds, full_labels)
    high_conf = [r for r in calibration_rows if r["pickProb"] >= 0.65]
    top_decile = (sum(1 for r in high_conf if r["isCorrect"]) / len(high_conf)) if high_conf else 0.0
    calibration_summary = build_calibration_summary(calibration_rows)

    # 10. Fresh game docs for the fetched window.
    fresh_dates = {g["date"] for g in fresh}
    rows_by_pk = {r["game"]["gamePk"]: r for r in rows}
    fresh_docs: list[dict] = []
    for g in enriched:
        if g["date"] not in fresh_dates:
            continue
        if g.get("winner") in ("home", "away"):
            row = rows_by_pk.get(g["gamePk"])
            if not row:
                continue
            pred = apply_model(model, row["features"], row["homeElo"], row["awayElo"])
            fresh_docs.append(
                build_game_doc(
                    g,
                    pred,
                    None,
                    build_run_projection(run_model_state, run_line_iso, g, None, None, RUN_CALIB_TRIALS, pred["homeWinProb"], run_margin_cal),
                    None,
                )
            )
        else:
            odds = market_odds_for_game(market_odds, g)
            pred = predict_for_game(g, model, team_state)
            fresh_docs.append(
                build_game_doc(
                    g,
                    pred,
                    {"home": team_state["injuries"].get(g["home"]["id"], 0), "away": team_state["injuries"].get(g["away"]["id"], 0)},
                    build_run_projection(
                        run_model_state,
                        run_line_iso,
                        g,
                        (odds or {}).get("total"),
                        (odds or {}).get("runLine"),
                        RUN_SIM_TRIALS,
                        pred["homeWinProb"],
                        run_margin_cal,
                    ),
                    odds,
                )
            )

    todays_record = build_todays_record([r for r in calibration_rows if r["date"] == today], today)

    # 11. Persist.
    rep("Saving model state", 92, "Persisting trained model…")
    state = {
        "key": "current",
        "trainedAt": int(now.timestamp() * 1000),
        "season": result["season"],
        "asOfDate": result["asOfDate"],
        "gamesTrained": result["gamesTrained"],
        "holdoutCount": result["holdoutCount"],
        "selectedModel": result["selectedModel"],
        "modelDescription": result["modelDescription"],
        "featureNames": result["featureNames"],
        "weights": result["weights"],
        "bias": result["bias"],
        "featureStats": result["featureStats"],
        "isotonicPoints": result["isotonicPoints"],
        "eloHfa": result["eloHfa"],
        "monteCarloEnabled": result["monteCarloEnabled"],
        "monteCarloTrials": result["monteCarloTrials"],
        "monteCarloSigma": result["monteCarloSigma"],
        "monteCarloRationale": result["monteCarloRationale"],
        "auc": full_eval["auc"],
        "brier": full_eval["brier"],
        "logLoss": full_eval["logLoss"],
        "ece": full_eval["ece"],
        "bins": full_eval["bins"],
        "confidenceDistribution": full_eval["confidenceDistribution"],
        "calibrationCurve": full_curve if full_curve else result["calibrationCurve"],
        "featureImportances": result["featureImportances"],
        "candidates": result["candidates"],
        "powerRankings": result["powerRankings"],
        "featureDrift": result["featureDrift"],
        "rollingBrier": result["rollingBrier"],
        "brierBaseline": result["brierBaseline"],
        "modelVersions": result["modelVersions"],
        "stackingWeights": result["stackingWeights"],
        "crossValidation": result["crossValidation"],
        "optimizationParams": result["optimizationParams"],
        "runModel": result["runModel"],
        "runLineCalibration": result["runLineCalibration"],
        "runMarginCalibration": result["runMarginCalibration"],
        "pitcherLogs": pitcher_log_cache,
        "teamLogs": team_log_cache,
        "batterLogs": batter_log_cache,
        "injurySnapshots": injury_cache,
        "calibrationSummary": calibration_summary,
        "spearmanRho": spearman,
        "topDecileWinRate": top_decile,
        "todaysRecord": todays_record,
        "marketOddsStatus": market_odds_status,
    }
    cache.save_model_state(state)
    cache.save_games(all_games)
    cache.save_calibration_rows(calibration_rows)
    cache.save_pitcher_logs(pitcher_log_cache)
    cache.save_team_logs(team_log_cache)
    cache.save_batter_logs(batter_log_cache)
    cache.save_bvp_logs(bvp_log_cache)
    cache.save_platoon_logs(platoon_log_cache)
    cache.save_vs_team_logs(vs_team_log_cache)
    cache.save_pitcher_hands(pitcher_hands_cache)
    cache.save_lineups(lineup_cache)
    cache.save_injury_snapshots(injury_cache)

    # 12. Update the games-by-date doc cache for the fresh window.
    docs_by_date = cache.load_docs_by_date()
    for doc in fresh_docs:
        docs_by_date[doc["date"]] = [d for d in docs_by_date.get(doc["date"], []) if d["gamePk"] != doc["gamePk"]] + [doc]
    cache.save_docs_by_date(docs_by_date)

    rep("Complete", 100, f"Refreshed {len(fresh_dates)} day(s) of predictions", )
    return {
        "season": season,
        "asOfDate": today,
        "gamesTrained": result["gamesTrained"],
        "holdoutCount": result["holdoutCount"],
        "auc": full_eval["auc"],
        "brier": full_eval["brier"],
        "logLoss": full_eval["logLoss"],
        "ece": full_eval["ece"],
        "selectedModel": result["selectedModel"],
        "monteCarloEnabled": result["monteCarloEnabled"],
        "storedGames": len(fresh_docs),
    }


def predict_date(date: str, state: dict | None = None) -> int:
    """On-demand prediction for an arbitrary date in the season (port of
    mlbActions.predictDate). Uses the stored model; fetches only that day."""
    state = state or cache.load_model_state()
    if not state:
        raise RuntimeError("Model has not been trained yet. Click refresh first.")
    model = reconstruct_model(state)
    team_state = reconstruct_team_state(state.get("powerRankings") or [])
    run_model_state = state.get("runModel")
    run_line_iso = state.get("runLineCalibration") or []
    run_margin_cal = state.get("runMarginCalibration") or {"slope": 0, "intercept": 0}
    season = state["season"]

    raw = fetch_schedule_range(date, date)

    # As-of-date stats for the requested day: game logs cached per {entity, season}.
    # Pitcher logs, team logs and boxscore lineups are independent — fetch them
    # concurrently; batter game logs follow once the lineups are known.
    pitcher_log_cache = cache.load_pitcher_logs()
    team_log_cache = cache.load_team_logs()
    pitcher_pairs = []
    for g in raw:
        s = g.get("season") or season
        for p in (g.get("awayPitcher"), g.get("homePitcher")):
            if p and p.get("id"):
                pitcher_pairs.append({"id": p["id"], "season": s})
    team_pairs = {}
    for g in raw:
        s = g.get("season") or season
        for tid in (g["home"]["id"], g["away"]["id"]):
            team_pairs.setdefault(f"{tid}|{s}", {"id": tid, "season": s})
    to_fetch_pitcher = [p for p in pitcher_pairs if f"{p['id']}|{p['season']}" not in pitcher_log_cache]
    to_fetch_team = [p for p in team_pairs.values() if f"{p['id']}|{p['season']}" not in team_log_cache]
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_pitcher = pool.submit(fetch_pitcher_game_logs, to_fetch_pitcher, pitcher_log_cache)
        f_team = pool.submit(fetch_team_game_logs, to_fetch_team, team_log_cache)
        f_lineups = pool.submit(fetch_lineups_for_games, raw, 16)
        fresh_pitcher_logs = f_pitcher.result()
        fresh_team_logs = f_team.result()
        lineups = f_lineups.result()
    pitcher_log_cache.update(fresh_pitcher_logs)
    team_log_cache.update(fresh_team_logs)

    enriched = attach_as_of_stats(raw, pitcher_log_cache, team_log_cache)

    batter_ids = []
    for lu in lineups.values():
        for side in (lu.get("home"), lu.get("away")):
            if not side:
                continue
            for p in side["battingOrder"] + side["bench"]:
                batter_ids.append(p["id"])
    batter_log_cache = cache.load_batter_logs()
    to_fetch = sorted({pid for pid in batter_ids if f"{pid}|{season}" not in batter_log_cache})
    if to_fetch:
        fresh_batter_logs = fetch_batter_game_logs(to_fetch, season, batter_log_cache)
        batter_log_cache.update(fresh_batter_logs)
    enriched = attach_lineups_as_of(enriched, lineups, batter_log_cache)

    # Matchup edges (BvP / platoon / vs-team) for the selected date's real
    # lineups, mirroring React's predictDate.
    matchup = enrich_with_matchups(
        enriched,
        batter_log_cache,
        cache.load_bvp_logs(),
        cache.load_platoon_logs(),
        cache.load_vs_team_logs(),
        season,
    )
    enriched = matchup["games"]

    market_odds = fetch_market_odds()

    docs = []
    for g in enriched:
        odds = market_odds_for_game(market_odds, g)
        pred = predict_for_game(g, model, team_state)
        docs.append(
            build_game_doc(
                g,
                pred,
                {"home": team_state["injuries"].get(g["home"]["id"], 0), "away": team_state["injuries"].get(g["away"]["id"], 0)},
                build_run_projection(
                    run_model_state,
                    run_line_iso,
                    g,
                    (odds or {}).get("total"),
                    (odds or {}).get("runLine"),
                    RUN_SIM_TRIALS,
                    pred["homeWinProb"],
                    run_margin_cal,
                ) if run_model_state else None,
                odds,
            )
        )
    # Persist the (possibly grown) log caches so the next refresh is incremental.
    cache.save_pitcher_logs(pitcher_log_cache)
    cache.save_team_logs(team_log_cache)
    cache.save_batter_logs(batter_log_cache)
    cache.save_bvp_logs(matchup["bvpLogs"])
    cache.save_platoon_logs(matchup["platoonLogs"])
    cache.save_vs_team_logs(matchup["vsTeamLogs"])
    cache.save_pitcher_hands(matchup["pitcherHands"])
    docs_by_date = cache.load_docs_by_date()
    docs_by_date[date] = docs
    cache.save_docs_by_date(docs_by_date)
    return len(docs)


def load_bundle() -> dict | None:
    """Load everything the dashboard needs from the disk cache."""
    state = cache.load_model_state()
    if not state:
        return None
    return {
        "model_state": state,
        "docs_by_date": cache.load_docs_by_date(),
        "calibration_rows": cache.load_calibration_rows(),
        "calibration_summary": state.get("calibrationSummary"),
    }
