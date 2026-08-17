"""Offline smoke test for the Streamlit migration's engine layer.

Runs entirely on the Python standard library (numpy is optional and only an
accelerator) so it can be executed in any environment without Streamlit,
Plotly, or network access:

    python3 mlb_streamlit/scripts/smoke_test.py

Covers:
  * metrics        — AUC, Brier, log-loss, ECE, calibration curves, isotonic
                     regression, Monte Carlo adjustment
  * logistic       — ridge logistic fitting, k-NN, naive Bayes, stacking
                     weights, cross-validation
  * features/elo   — chronological feature engineering + team state
  * runs           — run-scoring model fit, expected totals, Monte Carlo sims
  * Auto-ML        — the full run_model() pipeline on synthetic 2026 games
  * cache          — disk JSON round-trip

Exits non-zero on the first failure.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mlb_streamlit import cache  # noqa: E402
from mlb_streamlit.data import attach_lineups, lineup_ops  # noqa: E402
from mlb_streamlit.engine.features import (  # noqa: E402
    FEATURE_KEYS,
    build_features_for_game,
    compute_elo_and_features,
    new_state,
)
from mlb_streamlit.engine.logistic import (  # noqa: E402
    build_stacking_weights,
    cross_validate,
    knn_model,
    logistic_logit,
    naive_bayes_model,
    train_logistic,
)
from mlb_streamlit.engine.metrics import (  # noqa: E402
    american_odds,
    apply_isotonic,
    calibration_curve_points,
    compute_auc,
    compute_brier,
    compute_log_loss,
    evaluate,
    isotonic_regression,
    monte_carlo_adjust,
    sigmoid,
    spearman_rank,
)
from mlb_streamlit.engine.model import ELO_INIT, apply_model, run_model  # noqa: E402
from mlb_streamlit.engine.runs import (  # noqa: E402
    expected_margin,
    expected_total,
    fit_run_model,
    simulate_runs,
)

_CHECKS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _CHECKS
    _CHECKS += 1
    if not cond:
        raise AssertionError(f"FAIL: {name} {detail}")
    print(f"  ok: {name}")


# ---------------------------------------------------------------------------
# Synthetic 2026 season
# ---------------------------------------------------------------------------

TEAMS = [
    # id, name, abbrev, strength
    (108, "Angels", "LAA", -0.30),
    (109, "D-backs", "ARI", 0.10),
    (110, "Orioles", "BAL", 0.20),
    (111, "Red Sox", "BOS", 0.15),
    (112, "Cubs", "CHC", 0.05),
    (113, "Reds", "CIN", -0.05),
    (114, "Guardians", "CLE", 0.25),
    (115, "Rockies", "COL", -0.35),
    (116, "Tigers", "DET", 0.00),
    (117, "Astros", "HOU", 0.30),
    (118, "Royals", "KC", -0.15),
    (119, "Dodgers", "LAD", 0.35),
]
TEAM_META = {tid: {"id": tid, "name": name, "abbrev": abbrev, "strength": s} for tid, name, abbrev, s in TEAMS}
TIDS = [t[0] for t in TEAMS]


def make_games(n: int = 170, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    records: dict[int, dict] = {tid: {"wins": 0, "losses": 0} for tid in TIDS}
    games: list[dict] = []
    day = 0
    for i in range(n):
        home_id, away_id = rng.sample(TIDS, 2)
        d = date(2026, 3, 26) + timedelta(days=day)
        if i % 20 == 19:  # occasional off day
            day += 1
            d = date(2026, 3, 26) + timedelta(days=day)
        day += 1
        ymd = d.isoformat()
        sh, sa = TEAM_META[home_id]["strength"], TEAM_META[away_id]["strength"]
        p_home = 1 / (1 + math.exp(-(2.6 * (sh - sa) + 0.22)))
        p_home = min(0.9, max(0.1, p_home))
        home_wins = rng.random() < p_home
        if home_wins:
            hs = rng.randint(1, 9)
            as_ = max(0, hs - rng.randint(1, 6))
        else:
            as_ = rng.randint(1, 9)
            hs = max(0, as_ - rng.randint(1, 6))
        rec_h = records[home_id]
        rec_a = records[away_id]

        def side(tid: int, score: int, rec: dict) -> dict:
            return {
                "id": tid,
                "name": TEAM_META[tid]["name"],
                "abbrev": TEAM_META[tid]["abbrev"],
                "score": score,
                "wins": rec["wins"],
                "losses": rec["losses"],
                "ops": 0.720 + 0.28 * TEAM_META[tid]["strength"],
                "era": 4.35 - 1.6 * TEAM_META[tid]["strength"],
                "fieldingPct": 0.981 + 0.012 * TEAM_META[tid]["strength"],
            }

        games.append({
            "gamePk": 2026000000 + i,
            "date": ymd,
            "gameDate": f"{ymd}T18:00:00Z",
            "season": "2026",
            "home": side(home_id, hs, rec_h),
            "away": side(away_id, as_, rec_a),
            "winner": "home" if home_wins else "away",
            "homePitcher": {
                "id": home_id * 100 + 1,
                "name": f"{TEAM_META[home_id]['abbrev']} SP",
                "era": round(4.35 - 1.6 * TEAM_META[home_id]["strength"] + rng.uniform(-0.4, 0.4), 2),
                "fip": round(4.1 - 1.4 * TEAM_META[home_id]["strength"] + rng.uniform(-0.4, 0.4), 2),
            },
            "awayPitcher": {
                "id": away_id * 100 + 2,
                "name": f"{TEAM_META[away_id]['abbrev']} SP",
                "era": round(4.35 - 1.6 * TEAM_META[away_id]["strength"] + rng.uniform(-0.4, 0.4), 2),
                "fip": round(4.1 - 1.4 * TEAM_META[away_id]["strength"] + rng.uniform(-0.4, 0.4), 2),
            },
            "weather": {"tempF": rng.randint(55, 95), "windMph": round(rng.uniform(0, 18), 1)},
        })
        if home_wins:
            rec_h["wins"] += 1
            rec_a["losses"] += 1
        else:
            rec_h["losses"] += 1
            rec_a["wins"] += 1
    return games


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_metrics() -> None:
    print("metrics")
    preds = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    labels = [1, 1, 1, 0, 0, 0]
    check("auc perfect = 1", compute_auc(preds, labels) == 1.0)
    check("auc reversed = 0", compute_auc(list(reversed(preds)), labels) == 0.0)
    b = compute_brier([0.5] * 10, [1, 0] * 5)
    check("brier all-0.5 = 0.25", abs(b - 0.25) < 1e-9, f"got {b}")
    ll = compute_log_loss([1.0] * 5, [1] * 5)
    check("log-loss perfect = 0", abs(ll) < 1e-9, f"got {ll}")
    rho = spearman_rank([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5])
    check("spearman monotone ~ 1", abs(rho - 1.0) < 1e-9, f"got {rho}")
    rho_bin = spearman_rank([0.1, 0.2, 0.3, 0.4, 0.5], [0, 0, 0, 1, 1])
    check("spearman binary monotone", rho_bin > 0.8, f"got {rho_bin}")
    ev = evaluate(preds, labels)
    for key in ("auc", "brier", "logLoss", "ece", "bins", "confidenceDistribution", "calibrationCurve"):
        check(f"evaluate has {key}", key in ev)
    curve = calibration_curve_points(preds + [0.55, 0.45, 0.6, 0.4], labels + [1, 0, 1, 0], 2)
    check("calibration curve non-empty", len(curve) > 0)
    iso = isotonic_regression([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0, 1, 0, 1, 0, 1, 1])
    ys = [p["y"] for p in iso]
    check("isotonic monotone non-decreasing", all(b >= a for a, b in zip(ys, ys[1:])), f"{ys}")
    check("isotonic applied clamps to [0,1]",
          0 <= apply_isotonic(iso, 0.35) <= 1 and 0 <= apply_isotonic(iso, 0.99) <= 1)
    mc_lo, mc_hi = monte_carlo_adjust(0.6, 0.2), monte_carlo_adjust(0.8, 0.2)
    check("monte-carlo keeps [0,1]", 0 < mc_lo < 1 and 0 < mc_hi < 1)
    check("monte-carlo monotone", mc_hi > mc_lo, f"{mc_lo} vs {mc_hi}")
    check("american odds sign", american_odds(0.6) < 0 and american_odds(0.4) > 0,
          f"fav={american_odds(0.6)} dog={american_odds(0.4)}")


def test_logistic() -> None:
    print("logistic")
    rng = random.Random(11)
    rows = []
    for _ in range(300):
        x = rng.uniform(-2, 2)
        y = 1 if rng.random() < sigmoid(1.5 * x) else 0
        rows.append({"features": {"x": x, "noise": rng.uniform(-1, 1)}, "label": y})
    m = train_logistic(rows, ["x", "noise"], iterations=15)
    for key in ("featureNames", "weights", "bias", "featureStats"):
        check(f"logistic has {key}", key in m)
    p = sigmoid(logistic_logit(m, {"x": 2.0, "noise": 0.0}, None))
    q = sigmoid(logistic_logit(m, {"x": -2.0, "noise": 0.0}, None))
    check("logistic separates signal", p > 0.8 and q < 0.2, f"p={p} q={q}")
    check("x weight positive", m["weights"][m["featureNames"].index("x")] > 0)
    knn = knn_model(rows, ["x", "noise"])
    nb = naive_bayes_model(rows, ["x", "noise"])
    check("knn in [0,1]", 0 <= knn({"x": 1.0, "noise": 0.0}) <= 1)
    check("naive bayes in [0,1]", 0 <= nb({"x": 1.0, "noise": 0.0}) <= 1)
    preds_a = [sigmoid(logistic_logit(m, r["features"], None)) for r in rows]
    preds_b = [knn(r["features"]) for r in rows]
    stacking = build_stacking_weights({"A": preds_a, "B": preds_b}, [r["label"] for r in rows])
    check("stacking preds length", len(stacking["preds"]) == len(rows))
    wsum = sum(w["weight"] for w in stacking["weights"])
    check("stacking weights sum to 1", abs(wsum - 1.0) < 1e-6, f"sum={wsum}")
    cv = cross_validate(rows, ["x", "noise"], 5)
    for key in ("folds", "aucMean", "aucStd", "brierMean", "brierStd", "foldAucs", "foldBriers", "gamesPerFold"):
        check(f"cross_validate has {key}", key in cv)


def test_features_and_elo() -> None:
    print("features / elo")
    games = make_games(60, seed=3)
    fe = compute_elo_and_features(games)
    rows = fe["rows"]
    check("feature rows built", len(rows) == len(games), f"{len(rows)} vs {len(games)}")
    check("team state elo populated", len(fe["teamState"]["elo"]) == len(TIDS))
    for r in rows[:3]:
        check("row has all feature keys", all(f in r["features"] for f in FEATURE_KEYS))
        check("row label binary", r["label"] in (0, 1))
    upcoming = {
        "date": "2026-08-20",
        "home": {"id": 119, "name": "Dodgers", "abbrev": "LAD"},
        "away": {"id": 108, "name": "Angels", "abbrev": "LAA"},
        "homePitcher": {"era": 3.2, "fip": 3.0},
        "awayPitcher": {"era": 4.6, "fip": 4.4},
        "weather": {"tempF": 82, "windMph": 8},
    }
    feats = build_features_for_game(upcoming, fe["teamState"])
    check("upcoming features built", all(f in feats for f in FEATURE_KEYS))
    check("home field present", feats["homeField"] == 1.0)


def test_lineups() -> None:
    print("lineups")
    # lineup_ops: batting-order slots 1-4 carry 2x weight.
    lu = [
        {"ops": 0.800}, {"ops": 0.750}, {"ops": 0.700}, {"ops": 0.650},
        {"ops": 0.600}, {"ops": 0.550}, {"ops": 0.500}, {"ops": 0.450}, {"ops": 0.400},
    ]
    expected = (2 * (0.800 + 0.750 + 0.700 + 0.650) + (0.600 + 0.550 + 0.500 + 0.450 + 0.400)) / (2 * 4 + 5)
    got = lineup_ops(lu)
    check("lineup_ops top-4 double-weighted", abs(got - expected) < 1e-9, f"{got} vs {expected}")
    check("lineup_ops empty = 0", lineup_ops([]) == 0.0)
    check("lineup_ops skips missing ops", lineup_ops([{"ops": 0.800}, {}]) == 0.8)

    # Per-season attach: a batter's OPS must come from the game's own season,
    # otherwise 2026 numbers would leak into 2024 training rows.
    lineup = {
        "home": {"battingOrder": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}], "bench": []},
        "away": {"battingOrder": [{"id": 3, "name": "C"}, {"id": 4, "name": "D"}], "bench": []},
    }
    g24 = {"gamePk": 1, "date": "2024-06-01", "season": "2024",
           "home": {"id": 119, "name": "Dodgers", "abbrev": "LAD"},
           "away": {"id": 108, "name": "Angels", "abbrev": "LAA"}}
    g26 = {"gamePk": 2, "date": "2026-06-01", "season": "2026",
           "home": {"id": 119, "name": "Dodgers", "abbrev": "LAD"},
           "away": {"id": 108, "name": "Angels", "abbrev": "LAA"}}
    ops24 = {1: 0.720, 2: 0.680, 3: 0.700, 4: 0.640}
    ops26 = {1: 0.910, 2: 0.870, 3: 0.850, 4: 0.820}
    st = new_state()
    f24 = build_features_for_game(attach_lineups([g24], {1: lineup}, ops24)[0], st)
    f26 = build_features_for_game(attach_lineups([g26], {2: lineup}, ops26)[0], st)
    check("lineupKnown=1 with lineups", f24["lineupKnown"] == 1 and f26["lineupKnown"] == 1)
    check("lineupOpsDiff non-zero", f24["lineupOpsDiff"] > 0 and f26["lineupOpsDiff"] > 0)
    check("per-season OPS respected", f26["lineupOpsDiff"] > f24["lineupOpsDiff"],
          f"2024={f24['lineupOpsDiff']:.4f} 2026={f26['lineupOpsDiff']:.4f}")
    f_none = build_features_for_game(g24, st)
    check("no lineup -> feature 0", f_none["lineupOpsDiff"] == 0.0 and f_none["lineupKnown"] == 0)

    # Lineups cache round-trip preserves None (completed games with no boxscore lineups).
    marker = "_smoke_lineups.json"
    try:
        cache.save_json(marker, {"1": None, "2": lineup})
        data = cache.load_json(marker)
        check("lineups cache keeps None entries", data == {"1": None, "2": lineup}, f"{data}")
    finally:
        p = cache._path(marker)
        if p.exists():
            os.remove(p)


def test_runs_model() -> None:
    print("runs")
    games = make_games(120, seed=5)
    rm = fit_run_model(games)
    for key in ("leagueRuns", "teamOffense", "teamDefense", "parkFactor"):
        check(f"run model has {key}", key in rm)
    check("league runs sane", 3.5 < rm["leagueRuns"] < 5.5, f"{rm['leagueRuns']}")
    check("expected margin finite", math.isfinite(expected_margin(rm, 119, 108)))
    check("expected total positive", expected_total(rm, 119, 108) > 0)
    sim = simulate_runs(rm, 119, 108, 8.5, trials=2000)
    for key in ("homeScore", "awayScore", "total", "overProb", "underProb", "homeRunLineProb", "awayRunLineProb"):
        check(f"sim has {key}", key in sim)
    check("sim probs sum ~ 1", abs(sim["overProb"] + sim["underProb"] - 1.0) < 1e-6)


def test_automl_pipeline() -> None:
    print("auto-ml pipeline")
    # 500 games keeps the calibration/test splits large enough that the
    # isotonic fit does not saturate (matches real-season sample sizes).
    games = make_games(500, seed=21)
    out = run_model(games, season="2026", as_of_date="2026-08-15")
    result = out["result"]
    check("result keys", all(
        k in result for k in (
            "selectedModel", "modelDescription", "featureNames", "auc", "brier",
            "logLoss", "ece", "featureImportances", "candidates", "powerRankings",
            "featureDrift", "rollingBrier", "modelVersions", "stackingWeights",
            "crossValidation", "optimizationParams", "runModel", "monteCarloEnabled",
        )
    ))
    check("games trained > 0", result["gamesTrained"] > 0)
    check("auc above coin flip", result["auc"] >= 0.5, f"auc={result['auc']:.3f}")
    check("brier below naive", result["brier"] < 0.25, f"brier={result['brier']:.3f}")
    check("feature selection active", 2 <= len(result["featureNames"]) <= len(FEATURE_KEYS),
          f"{len(result['featureNames'])} selected")
    check("candidates trained", len(result["candidates"]) >= 5, f"{len(result['candidates'])}")
    check("candidate metrics sane", all(0.4 <= c["auc"] <= 1.0 and 0 <= c["brier"] <= 0.5 for c in result["candidates"]))
    check("power rankings sorted", all(a["elo"] >= b["elo"] for a, b in zip(result["powerRankings"], result["powerRankings"][1:])),
          "not sorted by elo")
    check("power rankings cover teams", len(result["powerRankings"]) == len(TIDS))
    check("feature importances cover keys", len(result["featureImportances"]) == len(FEATURE_KEYS))
    check("some features active", sum(1 for f in result["featureImportances"] if f["active"]) >= 1)
    check("stacking weights present", len(result["stackingWeights"]) >= 1)
    check("cross-validation folds", len(result["crossValidation"]["foldAucs"]) == 5)
    check("drift rows present", len(result["featureDrift"]) >= 1)
    check("version history present", len(result["modelVersions"]) >= 1)

    # The trained model object + prediction closure.
    model = out["model"]
    for key in ("featureNames", "weights", "bias", "featureStats", "isotonicPoints", "eloHfa", "monteCarloEnabled"):
        check(f"model has {key}", key in model)
    upcoming = {
        "date": "2026-08-20",
        "gameDate": "2026-08-20T18:00:00Z",
        "home": {"id": 119, "name": "Dodgers", "abbrev": "LAD"},
        "away": {"id": 108, "name": "Angels", "abbrev": "LAA"},
        "homePitcher": {"era": 3.2, "fip": 3.0},
        "awayPitcher": {"era": 4.6, "fip": 4.4},
        "weather": {"tempF": 82, "windMph": 8},
    }
    pred = out["predict"](upcoming)
    check("predict home prob in [0,1]", 0 <= pred["homeWinProb"] <= 1, f"{pred['homeWinProb']}")
    check("predict probs sum ~ 1", abs(pred["homeWinProb"] + pred["awayWinProb"] - 1.0) < 1e-6)
    check("predict has shap", len(pred.get("shap") or []) > 0)
    check("predict has fair odds", pred.get("fairHomeOdds") is not None)

    # apply_model directly on a feature row (as the calibration tab does).
    row = out["rows"][-1]
    applied = apply_model(model, row["features"], row["homeElo"], row["awayElo"])
    check("apply_model probability", 0 <= applied["homeWinProb"] <= 1)


def test_data_layer() -> None:
    print("data layer")
    import mlb_streamlit.data as data

    # parse_weather must handle string temp/wind AND structured wind objects
    ws = data.parse_weather({"condition": "Partly Cloudy", "temp": "72", "wind": "6 mph"})
    check("weather string wind", ws == {"condition": "Partly Cloudy", "tempF": 72.0, "windMph": 6.0}, f"{ws}")
    wo = data.parse_weather({"condition": "Dome", "temp": 72, "wind": {"speed": 9}})
    check("weather object wind", wo == {"condition": "Dome", "tempF": 72.0, "windMph": 9.0}, f"{wo}")
    check("weather none", data.parse_weather(None) is None)

    # parse_game survives a game whose weather has a string wind field
    raw = {
        "gamePk": 999001,
        "officialDate": "2024-04-14",
        "gameDate": "2024-04-14T18:00:00Z",
        "dayNight": "day",
        "gameType": "R",
        "status": {"abstractGameState": "Final", "detailedState": "Final"},
        "teams": {
            "away": {"team": {"id": 108}, "score": 3, "isWinner": False,
                     "leagueRecord": {"wins": 5, "losses": 2}},
            "home": {"team": {"id": 119}, "score": 5, "isWinner": True,
                     "leagueRecord": {"wins": 6, "losses": 1}},
        },
        "weather": {"condition": "Clear", "temp": "75", "wind": "8 mph, Out To CF"},
        "venue": {"name": "Dodger Stadium"},
    }
    p = data.parse_game(raw)
    check("parse_game handles string weather", p is not None and p["weather"]["windMph"] == 8.0, f"{p}")

    # map_limit must propagate worker exceptions (never leave None slots)
    try:
        data.map_limit([1, 2, 3], 2, lambda x: 1 / 0)
        check("map_limit propagates", False, "exception was swallowed")
    except ZeroDivisionError:
        check("map_limit propagates", True)

    # fetch_season skips a failed chunk instead of crashing the refresh
    calls = {"n": 0}

    def fake_schedule(start, end):
        calls["n"] += 1
        if start == "2024-04-14":
            raise RuntimeError("simulated statsapi 500")
        return [{"gamePk": calls["n"] * 1000 + i, "gameDate": f"{start}T18:00:00Z"} for i in range(3)]

    original = data.fetch_schedule_range
    data.fetch_schedule_range = fake_schedule
    try:
        games = data.fetch_season("2024", "2024-06-30")
    finally:
        data.fetch_schedule_range = original
    check("fetch_season skips failed chunk", len(games) == 9 and calls["n"] == 4,
          f"{len(games)} games from {calls['n']} chunks")


def test_market_odds() -> None:
    print("market odds")
    import mlb_streamlit.data as data

    calls = {"n": 0}
    original_fetch = data.fetch_json

    def fake_fetch(url: str, attempt: int = 0) -> dict:
        calls["n"] += 1
        return [
            {
                "home_team": "Los Angeles Dodgers",
                "away_team": "Los Angeles Angels",
                "commence_time": "2026-08-20T23:10:00Z",
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "markets": [
                            {"key": "h2h", "outcomes": [
                                {"name": "Los Angeles Dodgers", "price": -190},
                                {"name": "Los Angeles Angels", "price": 165},
                            ]},
                        ],
                    },
                    {
                        "key": "pinnacle",
                        "markets": [
                            {"key": "h2h", "outcomes": [
                                {"name": "Los Angeles Dodgers", "price": -180},
                                {"name": "Los Angeles Angels", "price": 155},
                            ]},
                            {"key": "totals", "outcomes": [
                                {"name": "Over", "price": -110, "point": 8.5},
                                {"name": "Under", "price": -110, "point": 8.5},
                            ]},
                            {"key": "spreads", "outcomes": [
                                {"name": "Los Angeles Dodgers", "price": -115, "point": -1.5},
                                {"name": "Los Angeles Angels", "price": -105, "point": 1.5},
                            ]},
                        ],
                    },
                ],
            }
        ]

    old_key = os.environ.get("THE_ODDS_API_KEY")
    old_cached = cache.load_market_odds()
    # Isolate the test from any real odds snapshot already in the disk cache
    # (e.g. one fetched live with the user's key), so the mocked path is exercised.
    cache_p = cache._path("market_odds.json")
    if cache_p.exists():
        cache_p.unlink()
    try:
        os.environ.pop("THE_ODDS_API_KEY", None)
        data.fetch_json = fake_fetch
        check("no key -> empty odds", data.fetch_market_odds() == {})
        check("no key -> no HTTP call", calls["n"] == 0)

        os.environ["THE_ODDS_API_KEY"] = "test-key"
        odds = data.fetch_market_odds()
        key = "2026-08-20|Los Angeles Dodgers|Los Angeles Angels"
        check("odds parsed", len(odds) == 1 and key in odds)
        entry = odds[key]
        for k in ("homeMoneyline", "awayMoneyline", "total", "overPrice",
                  "underPrice", "runLine", "homeRunLinePrice", "awayRunLinePrice"):
            check(f"odds entry has {k}", k in entry and entry[k] is not None)
        check("odds values sane", entry["homeMoneyline"] == -180 and entry["runLine"] == 1.5)

        calls["n"] = 0
        check("cached within TTL", data.fetch_market_odds() == odds and calls["n"] == 0)

        cache.save_market_odds({"fetchedAt": int(time.time() * 1000) - 7200_000, "odds": odds})
        calls["n"] = 0
        check("stale cache refetches", data.fetch_market_odds() == odds and calls["n"] == 1)

        def boom(url: str, attempt: int = 0) -> dict:
            raise RuntimeError("network down")

        data.fetch_json = boom
        check("failed fetch falls back to cache", data.fetch_market_odds() == odds)
        data.fetch_json = fake_fetch

        bm = data.pick_bookmaker([
            {"key": "fanduel", "markets": []},
            {"key": "pinnacle", "markets": []},
        ])
        check("pick_bookmaker prefers pinnacle", bm["key"] == "pinnacle")

        game = {"date": "2026-08-20", "home": {"id": 119}, "away": {"id": 108}}
        matched = data.market_odds_for_game(odds, game)
        check("market_odds_for_game resolves full names",
              matched is not None and matched.get("homeMoneyline") == -180)
    finally:
        data.fetch_json = original_fetch
        if old_key is None:
            os.environ.pop("THE_ODDS_API_KEY", None)
        else:
            os.environ["THE_ODDS_API_KEY"] = old_key
        if old_cached:
            cache.save_market_odds(old_cached)
        elif cache_p.exists():
            cache_p.unlink()


def test_cache_roundtrip() -> None:
    print("cache")
    marker = "_smoke_test.json"
    try:
        cache.save_json(marker, {"hello": 42, "list": [1, 2, 3]})
        data = cache.load_json(marker)
        check("cache round-trip", data == {"hello": 42, "list": [1, 2, 3]}, f"{data}")
        check("cache missing returns default", cache.load_json("_nope.json", "dflt") == "dflt")
    finally:
        p = cache._path(marker)
        if p.exists():
            os.remove(p)


def main() -> int:
    print(f"smoke test — python {sys.version.split()[0]}")
    test_metrics()
    test_logistic()
    test_features_and_elo()
    test_lineups()
    test_runs_model()
    test_data_layer()
    test_market_odds()
    test_automl_pipeline()
    test_cache_roundtrip()
    print(f"\nAll {_CHECKS} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
