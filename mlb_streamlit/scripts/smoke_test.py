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
from mlb_streamlit.data import (  # noqa: E402
    attach_lineups_as_of,
    attach_matchups,
    enrich_with_matchups,
    lineup_ops,
    matchup_lineup_mean,
    matchup_ops,
    matchup_pa,
)
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

    # Warm-start IRLS must reach the SAME fixed point as a cold fit (convergence
    # is detected exactly, so fewer iterations never change the model), and
    # converge in far fewer iterations from a good seed.
    w1 = m["weights"] + [m["bias"]]
    m_ws = train_logistic(rows, ["x", "noise"], iterations=20, w0=w1)
    w2 = m_ws["weights"] + [m_ws["bias"]]
    check("warm-start (20 iters) matches cold fit",
          all(abs(a - b) < 1e-6 for a, b in zip(w1, w2)), f"{w1} vs {w2}")
    m_fast = train_logistic(rows, ["x", "noise"], iterations=3, w0=w1)
    w3 = m_fast["weights"] + [m_fast["bias"]]
    check("warm-start converges in few iterations",
          all(abs(a - b) < 1e-4 for a, b in zip(w1, w3)), f"{w1} vs {w3}")
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

    # As-of-date attach: a batter's OPS must come from the game's OWN season
    # (2024 rows never see 2026 numbers) and only from games STRICTLY BEFORE
    # the game's date (a June game never sees the batter's July games).
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

    def ent(pid, season, h, tb, dbl=0, hr=0, bb=0):
        return {f"{pid}|{season}": [{"d": f"{season}-05-01", "ab": 4, "h": h, "bb": bb, "ibb": 0,
                                     "hbp": 0, "sf": 0, "tb": tb, "2b": dbl, "3b": 0, "hr": hr}]}

    batter_logs = {}
    # 2024: home hits 2/4 w/ a double (OPS 1.25) vs away singles (OPS .50)
    for pid in (1, 2):
        batter_logs.update(ent(pid, "2024", 2, 3, dbl=1))
    for pid in (3, 4):
        batter_logs.update(ent(pid, "2024", 1, 1))
    # 2026: home much hotter (HRs) vs away cold singles
    batter_logs.update(ent(1, "2026", 3, 6, dbl=1, hr=1))
    batter_logs.update(ent(2, "2026", 2, 4, dbl=1, hr=1))
    for pid in (3, 4):
        batter_logs.update(ent(pid, "2026", 1, 1))
    st = new_state()
    f24 = build_features_for_game(attach_lineups_as_of([g24], {1: lineup}, batter_logs)[0], st)
    f26 = build_features_for_game(attach_lineups_as_of([g26], {2: lineup}, batter_logs)[0], st)
    check("lineupKnown=1 with lineups", f24["lineupKnown"] == 1 and f26["lineupKnown"] == 1)
    check("lineupOpsDiff non-zero", f24["lineupOpsDiff"] > 0 and f26["lineupOpsDiff"] > 0)
    check("per-season OPS respected", f26["lineupOpsDiff"] > f24["lineupOpsDiff"],
          f"2024={f24['lineupOpsDiff']:.4f} 2026={f26['lineupOpsDiff']:.4f}")
    check("lineupWobaDiff non-zero", f24["lineupWobaDiff"] > 0 and f26["lineupWobaDiff"] > 0)
    check("lineupIsoDiff non-zero", f24["lineupIsoDiff"] > 0 and f26["lineupIsoDiff"] > 0)
    check("lineupHotDiff non-zero", f24["lineupHotDiff"] > 0 and f26["lineupHotDiff"] > 0)
    check("per-season wOBA respected", f26["lineupWobaDiff"] > f24["lineupWobaDiff"],
          f"2024={f24['lineupWobaDiff']:.4f} 2026={f26['lineupWobaDiff']:.4f}")

    # No-lookahead: a lineup whose batters ONLY have post-game entries is
    # unknown (all lineup features stay 0) — future games never count.
    leaky = {
        f"{pid}|2024": [{"d": "2024-06-15", "ab": 4, "h": 4, "bb": 0, "ibb": 0, "hbp": 0,
                          "sf": 0, "tb": 8, "2b": 2, "3b": 0, "hr": 1}]
        for pid in (1, 2, 3, 4)
    }
    f_leak = build_features_for_game(attach_lineups_as_of([g24], {1: lineup}, leaky)[0], st)
    check("no-lookahead lineup (post-game entries excluded)",
          f_leak["lineupKnown"] == 0 and f_leak["lineupOpsDiff"] == 0.0
          and f_leak["lineupWobaDiff"] == 0.0 and f_leak["lineupIsoDiff"] == 0.0
          and f_leak["lineupHotDiff"] == 0.0,
          f"known={f_leak['lineupKnown']} ops={f_leak['lineupOpsDiff']:.4f} woba={f_leak['lineupWobaDiff']:.4f}")

    f_none = build_features_for_game(g24, st)
    check("no lineup -> feature 0", f_none["lineupOpsDiff"] == 0.0 and f_none["lineupKnown"] == 0)

    # Trajectory: batter momentum (recent-10 OPS minus season OPS) and 7-day
    # fatigue, gated on >= 5 prior games, attached per-batter and
    # slot-weighted into the lineup features (no lookahead).
    def mk_traj(cold: bool, d: str) -> dict:
        h = 1 if cold else 3
        tb = 1 if cold else 7
        return {"d": d, "ab": 4, "h": h, "bb": 0, "ibb": 0, "hbp": 0, "sf": 0,
                "tb": tb, "2b": 0, "3b": 0, "hr": 0 if cold else 1}

    home_days = [(date(2026, 5, 1) + timedelta(days=i)).isoformat() for i in range(14)]  # 05-01..05-14
    away_days = [(date(2026, 4, 20) + timedelta(days=i)).isoformat() for i in range(14)]  # 04-20..05-03
    traj_logs = {}
    # Home batters: 9 cold games then 5 hot (05-10..05-14) -> positive
    # momentum; games7 = 2 (05-13/05-14 inside the 7-day window).
    for pid in (11, 12):
        traj_logs[f"{pid}|2026"] = [mk_traj(True, d) for d in home_days[:9]] + [mk_traj(False, d) for d in home_days[9:]]
    # Away batters: 14 cold games all before 05-03 -> momentum 0, games7 = 0.
    for pid in (21, 22):
        traj_logs[f"{pid}|2026"] = [mk_traj(True, d) for d in away_days]
    traj_lineup = {
        "home": {"battingOrder": [{"id": 11, "name": "H1"}, {"id": 12, "name": "H2"}], "bench": []},
        "away": {"battingOrder": [{"id": 21, "name": "A1"}, {"id": 22, "name": "A2"}], "bench": []},
    }
    tg = {"gamePk": 9, "date": "2026-05-20", "season": "2026",
          "home": {"id": 119}, "away": {"id": 108}}
    attached_t = attach_lineups_as_of([tg], {9: traj_lineup}, traj_logs)[0]
    hs = attached_t["lineupStats"]["home"]
    as_ = attached_t["lineupStats"]["away"]
    check("lineup momentum positive (heating up)", hs["momentum"] == 0.286, f"{hs['momentum']}")
    check("lineup momentum flat for cold team", as_["momentum"] == 0.0, f"{as_['momentum']}")
    check("lineup fatigue games7 (2 games in window)", hs["games7"] == 2.0, f"{hs['games7']}")
    check("lineup fatigue 0 for rested team", as_["games7"] == 0.0, f"{as_['games7']}")
    feats_t = build_features_for_game(attached_t, new_state())
    check("lineupMomentumDiff = home - away", feats_t["lineupMomentumDiff"] == 0.286,
          f"{feats_t['lineupMomentumDiff']}")
    check("lineupFatigueDiff = away - home (lower-better)", feats_t["lineupFatigueDiff"] == -2.0,
          f"{feats_t['lineupFatigueDiff']}")
    # Gating: fewer than 5 prior games -> momentum stays 0 (no junk values).
    sparse = {f"{pid}|2026": [mk_traj(True, d) for d in home_days[:3]] for pid in (11, 12, 21, 22)}
    sparse_lu = attach_lineups_as_of([tg], {9: traj_lineup}, sparse)[0]
    check("momentum gated on >= 5 prior games",
          sparse_lu["lineupStats"]["home"]["momentum"] == 0.0
          and sparse_lu["lineupStats"]["away"]["momentum"] == 0.0)

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


def test_matchups() -> None:
    print("matchups (BvP / platoon / vs-team)")
    import mlb_streamlit.data as data
    from mlb_streamlit.engine.features import build_features_for_game

    # 1. Feature registration: the three matchup edges are part of the
    #    candidate set and flow through Auto-ML selection.
    from mlb_streamlit.engine.features import FEATURE_LABELS

    for f in ("bvpOpsDiff", "platoonOpsDiff", "vsTeamOpsDiff"):
        check(f"feature key registered: {f}", f in FEATURE_KEYS)
        check(f"feature label present: {f}", f in FEATURE_LABELS)

    # 2. statsapi stat-block parsing (OPS direct + OBP/SLG fallback + PA fallback).
    check("matchup_ops direct", matchup_ops({"ops": "0.850", "obp": ".320", "slg": ".420"}) == 0.85)
    check("matchup_ops obp+slg fallback", matchup_ops({"ops": ".000", "obp": ".300", "slg": ".400"}) == 0.7)
    check("matchup_ops none", matchup_ops({"ops": ".000", "obp": ".000", "slg": ".000"}) is None)
    check("matchup_pa direct", matchup_pa({"plateAppearances": "41", "atBats": 40}) == 41.0)
    check("matchup_pa fallback", matchup_pa({"atBats": 40, "baseOnBalls": 5, "hitByPitch": 1, "sacFlies": 2}) == 48.0)

    # 3. matchup_lineup_mean: slots 1-4 double-weighted; PA saturation caps the
    #    influence of tiny BvP samples (pa 0 rows are excluded entirely).
    lu = [{"bvpOPS": 0.8}, {"bvpOPS": 0.4}, {"bvpOPS": 0.4}, {"bvpOPS": 0.4}, {"bvpOPS": 0.4}]
    exp = (2 * (0.8 + 0.4 + 0.4 + 0.4) + 0.4) / 9
    check("matchup_lineup_mean top-4 double-weighted",
          abs(matchup_lineup_mean(lu, "bvpOPS") - round(exp, 3)) < 1e-9, f"{matchup_lineup_mean(lu, 'bvpOPS')}")
    sat = [{"bvpOPS": 0.9, "bvpPA": 1000}, {"bvpOPS": 0.5, "bvpPA": 0}]
    check("matchup_lineup_mean PA saturation", matchup_lineup_mean(sat, "bvpOPS", "bvpPA") == 0.9,
          f"{matchup_lineup_mean(sat, 'bvpOPS', 'bvpPA')}")
    check("matchup_lineup_mean empty = 0", matchup_lineup_mean([], "bvpOPS") == 0.0)

    # 4. attach_matchups gating + shrinkage.
    #    Home batter 11 vs away RHP 902: BvP 5 PA @ 1.000 OPS, shrunk toward
    #    season OPS 0.700 -> (5 + 15*0.7) / 20 = 0.775. Platoon uses the
    #    starter's hand (vsRight). vsTeam targets the opposing team (108).
    lineup = {
        "home": {"battingOrder": [{"id": 11, "name": "H1"}, {"id": 12, "name": "H2"}], "bench": []},
        "away": {"battingOrder": [{"id": 21, "name": "A1"}, {"id": 22, "name": "A2"}], "bench": []},
    }
    game = {
        "gamePk": 1,
        "date": "2026-06-01",
        "season": "2026",
        "home": {"id": 119, "name": "Dodgers", "abbrev": "LAD"},
        "away": {"id": 108, "name": "Angels", "abbrev": "LAA"},
        "homePitcher": {"id": 901, "name": "P1", "pitchHand": "L"},
        "awayPitcher": {"id": 902, "name": "P2", "pitchHand": "R"},
        "lineups": lineup,
        "lineupStats": {"home": {"known": True, "ops": 0.8}, "away": {"known": True, "ops": 0.7}},
    }
    season_log = {"d": "2026-05-01", "ab": 10, "h": 3, "bb": 0, "ibb": 0, "hbp": 0, "sf": 0,
                  "tb": 4, "2b": 1, "3b": 0, "hr": 0}  # OPS 0.300 + 0.400 = 0.700
    batter_logs = {f"{pid}|2026": [season_log] for pid in (11, 12, 21, 22)}
    bvp_cache = {"11|902": {"pa": 5, "ops": 1.0}}
    platoon_cache = {"11|2026": {"vsLeft": {"pa": 30, "ops": 0.600}, "vsRight": {"pa": 40, "ops": 0.850}}}
    vs_team_cache = {"11|108|2026": {"pa": 40, "ops": 0.900}}
    attached = attach_matchups([game], batter_logs, bvp_cache, platoon_cache, vs_team_cache, {})[0]
    home_stats = attached["lineupStats"]["home"]
    away_stats = attached["lineupStats"]["away"]
    check("bvpOPS shrunk toward season OPS", home_stats["bvpOps"] == 0.775, f"{home_stats['bvpOps']}")
    check("platoon picks starter's hand (vsRight)", home_stats["platoonOps"] == 0.85, f"{home_stats['platoonOps']}")
    check("vsTeam targets opposing team", home_stats["vsTeamOps"] == 0.9, f"{home_stats['vsTeamOps']}")
    check("bvpPA summed across order", home_stats["bvpPA"] == 5.0, f"{home_stats['bvpPA']}")
    check("away side has no matchup data -> 0",
          away_stats["bvpOps"] == 0.0 and away_stats["platoonOps"] == 0.0 and away_stats["vsTeamOps"] == 0.0,
          f"{away_stats}")
    check("per-batter matchup keys attached",
          attached["lineups"]["home"]["battingOrder"][0]["bvpOPS"] == 0.775
          and attached["lineups"]["home"]["battingOrder"][0]["platoonOPS"] == 0.85
          and attached["lineups"]["home"]["battingOrder"][0]["vsTeamOPS"] == 0.9
          and attached["lineups"]["home"]["battingOrder"][0]["bvpPA"] == 5.0)
    feats = build_features_for_game(attached, new_state())
    check("bvpOpsDiff = home - away", feats["bvpOpsDiff"] == 0.775, f"{feats['bvpOpsDiff']}")
    check("platoonOpsDiff = home - away", feats["platoonOpsDiff"] == 0.85, f"{feats['platoonOpsDiff']}")
    check("vsTeamOpsDiff = home - away", feats["vsTeamOpsDiff"] == 0.9, f"{feats['vsTeamOpsDiff']}")

    # No opposing starter known -> BvP / platoon stay 0 (vsTeam still works).
    no_starter = {
        **game,
        "homePitcher": None,
        "awayPitcher": None,
        "lineups": lineup,
        "lineupStats": {"home": {"known": True, "ops": 0.8}, "away": {"known": True, "ops": 0.7}},
    }
    ns = attach_matchups([no_starter], batter_logs, bvp_cache, platoon_cache, vs_team_cache, {})[0]
    check("no starter -> bvp/platoon 0, vsTeam intact",
          ns["lineupStats"]["home"]["bvpOps"] == 0.0
          and ns["lineupStats"]["home"]["platoonOps"] == 0.0
          and ns["lineupStats"]["home"]["vsTeamOps"] == 0.9)

    # No real boxscore lineup -> game passes through untouched (fresh-window
    # gating: older history never gets junk matchup values).
    bare = {"gamePk": 2, "date": "2026-03-20", "season": "2026",
            "home": {"id": 119}, "away": {"id": 108}}
    passthrough = attach_matchups([bare], batter_logs, bvp_cache, platoon_cache, vs_team_cache, {})[0]
    check("no lineup -> passthrough", passthrough == bare)
    no_ls_src = {k: v for k, v in game.items() if k != "lineupStats"}
    no_ls = attach_matchups([no_ls_src], batter_logs, bvp_cache, platoon_cache, vs_team_cache, {})[0]
    check("lineup without lineupStats -> lineups attached only",
          no_ls.get("lineupStats") is None and no_ls["lineups"]["home"]["battingOrder"][0].get("bvpOPS") == 0.775)

    # 5. enrich_with_matchups: pair gathering + per-season caching, offline.
    calls = {"bvp": [], "platoon": [], "vs": [], "hands": []}
    orig = (data.fetch_bvp_stats, data.fetch_platoon_splits, data.fetch_vs_team_stats, data.fetch_pitcher_hands)

    def fake_bvp(pairs, cached=None):
        calls["bvp"].append(pairs)
        return {"11|902": {"pa": 5, "ops": 1.0}}

    def fake_platoon(pairs, cached=None):
        calls["platoon"].append((pairs, cached))
        return {"11|2026": {"vsRight": {"pa": 40, "ops": 0.850}}}

    def fake_vs(pairs, cached=None):
        calls["vs"].append(pairs)
        return {"11|108|2026": {"pa": 40, "ops": 0.900}}

    def fake_hands(ids):
        calls["hands"].append(ids)
        return {}

    data.fetch_bvp_stats = fake_bvp
    data.fetch_platoon_splits = fake_platoon
    data.fetch_vs_team_stats = fake_vs
    data.fetch_pitcher_hands = fake_hands
    try:
        result = enrich_with_matchups([game], batter_logs, {}, {}, {}, "2026")
    finally:
        (data.fetch_bvp_stats, data.fetch_platoon_splits, data.fetch_vs_team_stats,
         data.fetch_pitcher_hands) = orig
    check("bvp pairs = each batter vs opposing starter", len(calls["bvp"][0]) == 4, f"{calls['bvp'][0]}")
    check("platoon pairs = each batter", len(calls["platoon"][0][0]) == 4)
    check("vs-team pairs = each batter vs opposing team", len(calls["vs"][0]) == 4)
    check("no hand fetch when starters carry pitchHand", calls["hands"] == [[]],
          f"{calls['hands']}")
    check("enrich merges fetched caches", result["bvpLogs"]["11|902"] == {"pa": 5, "ops": 1.0}
          and result["platoonLogs"]["11|2026"]["vsRight"]["ops"] == 0.85
          and result["vsTeamLogs"]["11|108|2026"]["ops"] == 0.9)
    check("enrich attaches games", result["games"][0]["lineupStats"]["home"]["bvpOps"] == 0.775)

    # Past-season splits are reused from the cache (cached= passed through);
    # current-season splits always refetch (cached={}).
    past = {"gamePk": 3, "date": "2025-06-01", "season": "2025",
            "home": {"id": 119}, "away": {"id": 108},
            "homePitcher": {"id": 901, "pitchHand": "L"},
            "awayPitcher": {"id": 902, "pitchHand": "R"},
            "lineups": lineup,
            "lineupStats": {"home": {"known": True, "ops": 0.8}, "away": {"known": True, "ops": 0.7}}}
    calls = {"bvp": [], "platoon": [], "vs": [], "hands": []}
    data.fetch_platoon_splits = fake_platoon
    data.fetch_vs_team_stats = fake_vs
    data.fetch_bvp_stats = fake_bvp
    data.fetch_pitcher_hands = fake_hands
    try:
        enrich_with_matchups([past], batter_logs, {}, {"11|2025": {"vsRight": {"pa": 9, "ops": 0.9}}}, {}, "2026")
    finally:
        data.fetch_platoon_splits = orig[1]
        data.fetch_vs_team_stats = orig[2]
        data.fetch_bvp_stats = orig[0]
        data.fetch_pitcher_hands = orig[3]
    cached_args = [c for c in calls["platoon"] if c[1] == {"11|2025": {"vsRight": {"pa": 9, "ops": 0.9}}}]
    fresh_args = [c for c in calls["platoon"] if c[1] == {}]
    check("past-season splits reuse cache (pairs passed through)",
          len(cached_args) == 1 and len(cached_args[0][0]) == 4,
          f"platoon calls={calls['platoon']}")
    check("current-season splits refetch (cached={})",
          len(fresh_args) == 1 and fresh_args[0][0] == [],
          f"platoon calls={calls['platoon']}")

    # 6. pitcher-hand fallback fetch when the schedule hydrate omits the hand.
    no_hand = {"gamePk": 4, "date": "2026-06-02", "season": "2026",
               "home": {"id": 119}, "away": {"id": 108},
               "homePitcher": {"id": 901}, "awayPitcher": {"id": 902},
               "lineups": lineup,
               "lineupStats": {"home": {"known": True, "ops": 0.8}, "away": {"known": True, "ops": 0.7}}}
    calls = {"bvp": [], "platoon": [], "vs": [], "hands": []}
    data.fetch_pitcher_hands = fake_hands
    try:
        enrich_with_matchups([no_hand], batter_logs, {}, {}, {}, "2026")
    finally:
        data.fetch_pitcher_hands = orig[3]
    check("starters without hand trigger fetch",
          calls["hands"] == [[902, 901]] or calls["hands"] == [[901, 902]],
          f"{calls['hands']}")

    # 7. BvP TTL: entries fetched within the TTL skip the refetch (career
    #    totals barely move); stale entries refetch; pre-TTL legacy caches
    #    (no timestamp) are treated as fresh so deploy doesn't mass-refetch.
    calls = {"n": 0}

    def fake_bvp_json(url, attempt=0):
        calls["n"] += 1
        return {"stats": [{"splits": [{"stat": {"plateAppearances": "40", "ops": ".850"}}]}]}

    orig_json = data.fetch_json
    data.fetch_json = fake_bvp_json
    try:
        out = data.fetch_bvp_stats([{"batterId": 11, "pitcherId": 902}], {})
        check("bvp fetched on empty cache", calls["n"] == 1 and "11|902" in out
              and out["11|902"].get("fetchedAt") is not None, f"{out} calls={calls['n']}")
        calls["n"] = 0
        data.fetch_bvp_stats([{"batterId": 11, "pitcherId": 902}], out)
        check("bvp TTL skips fresh refetch", calls["n"] == 0, f"calls={calls['n']}")
        stale = {"11|902": {"pa": 5, "ops": 1.0, "fetchedAt": int(time.time() * 1000) - 48 * 3600 * 1000}}
        calls["n"] = 0
        data.fetch_bvp_stats([{"batterId": 11, "pitcherId": 902}], stale)
        check("bvp TTL refetches stale entries", calls["n"] == 1, f"calls={calls['n']}")
        legacy = {"11|902": {"pa": 5, "ops": 1.0}}
        calls["n"] = 0
        data.fetch_bvp_stats([{"batterId": 11, "pitcherId": 902}], legacy)
        check("bvp legacy cache (no timestamp) treated fresh", calls["n"] == 0, f"calls={calls['n']}")
    finally:
        data.fetch_json = orig_json

def test_as_of_stats() -> None:
    print("as-of-date stats")
    import mlb_streamlit.data as data

    # pitcher_as_of: only starts strictly before the target date count.
    log = [
        {"d": "2026-04-05", "ip": 6.0, "er": 2, "so": 5, "bb": 2, "hbp": 0, "hr": 1, "h": 3},
        {"d": "2026-04-08", "ip": 5.0, "er": 3, "so": 4, "bb": 1, "hbp": 0, "hr": 1, "h": 5},
        {"d": "2026-04-12", "ip": 7.0, "er": 0, "so": 8, "bb": 1, "hbp": 0, "hr": 0, "h": 2},
        {"d": "2026-04-14", "ip": 4.0, "er": 4, "so": 3, "bb": 3, "hbp": 0, "hr": 2, "h": 6},
        {"d": "2026-04-19", "ip": 5.0, "er": 6, "so": 3, "bb": 4, "hbp": 1, "hr": 2, "h": 4},  # after target
    ]
    a = data.pitcher_as_of(log, "2026-04-15")
    ip = 6 + 5 + 7 + 4
    er = 2 + 3 + 0 + 4
    so = 5 + 4 + 8 + 3
    bb = 2 + 1 + 1 + 3
    h = 3 + 5 + 2 + 6
    check("pitcher_as_of ERA", a["era"] == round(er * 9 / ip, 2), f"{a}")
    check("pitcher_as_of K9", a["k9"] == round(so * 9 / ip, 2), f"{a}")
    exp_fip = (13 * (1 + 1 + 0 + 2) + 3 * bb - 2 * so) / ip + 3.1
    check("pitcher_as_of FIP", a["fip"] == round(exp_fip, 2), f"{a}")
    check("pitcher_as_of WHIP", a["whip"] == round((bb + h) / ip, 2), f"{a}")
    # recentEra covers only the last 3 starts before the target (04-08/12/14).
    r_ip = 5 + 7 + 4
    r_er = 3 + 0 + 4
    check("pitcher_as_of recentEra (last 3 starts)", a["recentEra"] == round(r_er * 9 / r_ip, 2), f"{a}")

    # Trajectory fields: rest since the last start, workload (IP, last 3
    # starts) and the FIP trend slope over the last 5 starts (>= 3 required) —
    # all strictly before the target date, from the same cached log.
    check("pitcher_as_of restDays (last start 04-14 vs 04-15)", a.get("restDays") == 1, f"{a.get('restDays')}")
    check("pitcher_as_of workload (IP last 3 starts)", a.get("workload") == 16.0, f"{a.get('workload')}")
    check("pitcher_as_of trendFip present", isinstance(a.get("trendFip"), (int, float)))
    check("pitcher_as_of trendFip sign (terrible final start -> worsening)", a.get("trendFip", 0) > 0,
          f"{a.get('trendFip')}")
    improving = [
        {"d": "2026-04-01", "ip": 6.0, "er": 3, "so": 4, "bb": 2, "hbp": 0, "hr": 2, "h": 6},
        {"d": "2026-04-06", "ip": 6.0, "er": 2, "so": 5, "bb": 2, "hbp": 0, "hr": 1, "h": 5},
        {"d": "2026-04-11", "ip": 6.0, "er": 1, "so": 6, "bb": 1, "hbp": 0, "hr": 0, "h": 4},
        {"d": "2026-04-16", "ip": 6.0, "er": 0, "so": 8, "bb": 1, "hbp": 0, "hr": 0, "h": 2},
    ]
    b = data.pitcher_as_of(improving, "2026-04-20")
    check("pitcher_as_of trendFip negative when improving", (b.get("trendFip") or 0) < 0, f"{b.get('trendFip')}")
    check("pitcher_as_of restDays when well rested", b.get("restDays") == 4, f"{b.get('restDays')}")
    check("pitcher_as_of empty", data.pitcher_as_of([], "2026-04-15") == {})
    check("pitcher_as_of no prior games", data.pitcher_as_of(log, "2026-04-01") == {})

    # batter_ops_as_of: OPS from OBP/SLG components, strictly before the date.
    blog = [
        {"d": "2026-04-05", "ab": 4, "h": 1, "bb": 0, "ibb": 0, "hbp": 0, "sf": 0, "tb": 2, "2b": 1, "3b": 0, "hr": 0},
        {"d": "2026-04-06", "ab": 4, "h": 2, "bb": 1, "ibb": 0, "hbp": 0, "sf": 0, "tb": 3, "2b": 1, "3b": 0, "hr": 0},
        {"d": "2026-04-20", "ab": 4, "h": 4, "bb": 0, "ibb": 0, "hbp": 0, "sf": 0, "tb": 8, "2b": 2, "3b": 0, "hr": 0},  # after
    ]
    ops = data.batter_ops_as_of(blog, "2026-04-20")
    # Both prior games: 8 AB, 3 H, 1 BB, 5 TB -> OBP 4/9, SLG 5/8.
    check("batter_ops_as_of excludes post-date games", ops is not None and ops == round(0.625 + 4 / 9, 3), f"{ops}")
    # wOBA: game 1 is a double (1.27/4); game 2 is a single + walk ((0.69+0.89+1.27)/5).
    check("batter_woba_as_of", data.batter_woba_as_of(blog, "2026-04-20") == round((1.27 + 2.85) / 9, 3))
    check("batter_iso_as_of", data.batter_iso_as_of(blog, "2026-04-20") == 0.25)
    check("batter_recent_ops_as_of (window)", data.batter_recent_ops_as_of(blog, "2026-04-20") == round(0.625 + 4 / 9, 3))
    check("batter_ops_as_of none without prior games", data.batter_ops_as_of(blog, "2026-03-01") is None)

    # team_as_of: ops/era/fielding accumulated per group, strictly before date.
    tlog = {
        "hitting": [
            {"d": "2026-04-05", "ab": 40, "h": 10, "bb": 4, "hbp": 1, "sf": 1, "tb": 16},
            {"d": "2026-04-06", "ab": 36, "h": 12, "bb": 2, "hbp": 0, "sf": 0, "tb": 20},
            {"d": "2026-05-01", "ab": 40, "h": 16, "bb": 2, "hbp": 0, "sf": 0, "tb": 28},  # after
        ],
        "pitching": [
            {"d": "2026-04-05", "ip": 9.0, "er": 3, "so": 8, "bb": 2, "hbp": 0, "hr": 1, "h": 5},
            {"d": "2026-05-01", "ip": 9.0, "er": 9, "so": 8, "bb": 2, "hbp": 0, "hr": 3, "h": 9},  # after
        ],
        "fielding": [
            {"d": "2026-04-05", "po": 27, "a": 9, "e": 0},
            {"d": "2026-05-01", "po": 27, "a": 9, "e": 3},  # after
        ],
    }
    t = data.team_as_of(tlog, "2026-04-10")
    # Both 04-05 and 04-06 hitting games are before the target date; 05-01 is not.
    exp_ops = (22 + 6 + 1) / (76 + 6 + 1 + 1) + 36 / 76
    check("team_as_of ops", t["ops"] == round(exp_ops, 3), f"{t}")
    check("team_as_of era", t["era"] == round(3 * 9 / 9, 2), f"{t}")
    check("team_as_of k9", t["k9"] == 8.0, f"{t}")
    check("team_as_of whip", t["whip"] == round((2 + 5) / 9, 2), f"{t}")
    check("team_as_of fieldingPct", t["fieldingPct"] == 1.0, f"{t}")
    check("team_as_of empty", data.team_as_of(None, "2026-04-10") == {})

    # attach_as_of_stats: a game's stats reflect its OWN date, not today's.
    games = [
        {"gamePk": 1, "date": "2026-04-10", "season": "2026",
         "home": {"id": 119, "name": "Dodgers", "abbrev": "LAD"},
         "away": {"id": 108, "name": "Angels", "abbrev": "LAA"},
         "homePitcher": {"id": 1, "name": "A"}, "awayPitcher": {"id": 2, "name": "B"}},
        {"gamePk": 2, "date": "2026-05-10", "season": "2026",
         "home": {"id": 119, "name": "Dodgers", "abbrev": "LAD"},
         "away": {"id": 108, "name": "Angels", "abbrev": "LAA"},
         "homePitcher": {"id": 1, "name": "A"}, "awayPitcher": {"id": 2, "name": "B"}},
    ]
    p_logs = {
        "1|2026": [{"d": "2026-04-01", "ip": 6.0, "er": 2, "so": 5, "bb": 2, "hbp": 0, "hr": 1},
                    {"d": "2026-05-01", "ip": 7.0, "er": 7, "so": 8, "bb": 2, "hbp": 0, "hr": 3}],
        "2|2026": [{"d": "2026-04-01", "ip": 6.0, "er": 2, "so": 5, "bb": 2, "hbp": 0, "hr": 1},
                    {"d": "2026-05-01", "ip": 7.0, "er": 7, "so": 8, "bb": 2, "hbp": 0, "hr": 3}],
    }
    t_logs = {"119|2026": tlog, "108|2026": tlog}
    g1, g2 = data.attach_as_of_stats(games, p_logs, t_logs)
    check("attach as-of pitcher (early game)", g1["homePitcher"]["era"] == 3.0, f"{g1['homePitcher']}")
    check("attach as-of pitcher (later game)", g2["homePitcher"]["era"] == round(9 * 9 / 13, 2), f"{g2['homePitcher']}")
    check("attach as-of team ops differs by date", g1["home"]["ops"] != g2["home"]["ops"],
          f"{g1['home'].get('ops')} vs {g2['home'].get('ops')}")
    check("attach as-of keeps team ids", g1["home"]["id"] == 119 and g1["away"]["id"] == 108)
    # Optimization regression coverage — the pointer-based as-of attachment,
    # the data fingerprint, and the no-retrain fast refresh path. Invoked
    # from here because the file tail (main) is outside the patch tool's
    # index; each test reports through the shared check() counter.
    test_as_of_pointer_equiv()
    test_data_fingerprint()
    test_refresh_fast_path()


def test_as_of_pointer_equiv() -> None:
    print("as-of pointer equivalence")
    import mlb_streamlit.data as data

    def mk_game(pk: int, ymd: str, hp: int, ap: int) -> dict:
        return {
            "gamePk": pk,
            "date": ymd,
            "gameDate": f"{ymd}T18:00:00Z",
            "season": "2026",
            "home": {"id": 119, "name": "Dodgers", "abbrev": "LAD", "score": 3},
            "away": {"id": 108, "name": "Angels", "abbrev": "LAA", "score": 2},
            "homePitcher": {"id": hp, "name": f"SP{hp}"},
            "awayPitcher": {"id": ap, "name": f"SP{ap}"},
            "winner": "home",
        }

    games = [
        mk_game(1, "2026-04-01", 11, 21),
        mk_game(2, "2026-04-03", 11, 22),
        mk_game(3, "2026-04-05", 12, 21),
        mk_game(4, "2026-04-08", 12, 22),
        mk_game(5, "2026-04-10", 11, 21),
    ]
    games = [games[4], games[1], games[3], games[0], games[2]]  # unsorted input

    pitcher_logs: dict[str, list[dict]] = {}
    for pid in (11, 12, 21, 22):
        pitcher_logs[f"{pid}|2026"] = [
            {"d": "2026-03-26", "ip": 6.0, "er": 2, "so": 5, "bb": 2, "hbp": 0, "hr": 1, "h": 3},
            {"d": "2026-03-31", "ip": 7.0, "er": 1, "so": 8, "bb": 1, "hbp": 1, "hr": 0, "h": 2},
            {"d": "2026-04-04", "ip": 5.0, "er": 3, "so": 4, "bb": 3, "hbp": 0, "hr": 1, "h": 5},
            {"d": "2026-04-07", "ip": 6.0, "er": 4, "so": 6, "bb": 2, "hbp": 0, "hr": 2, "h": 6},
        ]
    team_logs: dict[str, dict] = {}
    for tid in (108, 119):
        team_logs[f"{tid}|2026"] = {
            "hitting": [
                {"d": "2026-03-26", "ab": 34, "h": 9, "bb": 3, "hbp": 1, "sf": 1, "tb": 14},
                {"d": "2026-03-31", "ab": 36, "h": 10, "bb": 2, "hbp": 0, "sf": 0, "tb": 16},
                {"d": "2026-04-04", "ab": 35, "h": 8, "bb": 4, "hbp": 0, "sf": 1, "tb": 12},
            ],
            "pitching": [
                {"d": "2026-03-26", "ip": 9.0, "er": 3, "so": 8, "bb": 2, "h": 7},
                {"d": "2026-03-31", "ip": 9.0, "er": 2, "so": 9, "bb": 1, "h": 5},
                {"d": "2026-04-04", "ip": 9.0, "er": 4, "so": 7, "bb": 3, "h": 8},
            ],
            "fielding": [
                {"d": "2026-03-26", "po": 27, "a": 9, "e": 1},
                {"d": "2026-03-31", "po": 27, "a": 10, "e": 0},
                {"d": "2026-04-04", "po": 27, "a": 8, "e": 2},
            ],
        }

    def naive_attach(games: list[dict]) -> list[dict]:
        out = []
        for g in games:
            season = g.get("season")
            hp = g.get("homePitcher")
            ap = g.get("awayPitcher")
            if hp and hp.get("id"):
                hp = {**hp, **data.pitcher_as_of(pitcher_logs.get(f"{hp['id']}|{season}"), g["date"])}
            if ap and ap.get("id"):
                ap = {**ap, **data.pitcher_as_of(pitcher_logs.get(f"{ap['id']}|{season}"), g["date"])}
            out.append({
                **g,
                "awayPitcher": ap,
                "homePitcher": hp,
                "home": {**g["home"], **data.team_as_of(team_logs.get(f"{g['home']['id']}|{season}"), g["date"])},
                "away": {**g["away"], **data.team_as_of(team_logs.get(f"{g['away']['id']}|{season}"), g["date"])},
            })
        return out

    ptr = data.attach_as_of_stats([dict(g) for g in games], pitcher_logs, team_logs)
    naive = naive_attach(games)
    check("as-of stats: pointer == naive on unsorted games", ptr == naive, "output mismatch")

    def batter(bid: int) -> dict:
        return {"id": bid, "name": f"B{bid}", "pos": "OF"}

    home9 = [batter(1001 + i) for i in range(9)]
    away9 = [batter(2001 + i) for i in range(9)]
    lineups = {
        1: {"home": {"battingOrder": home9, "bench": [batter(3001)]}, "away": {"battingOrder": away9, "bench": []}},
        4: {"home": {"battingOrder": home9, "bench": []}, "away": {"battingOrder": away9, "bench": []}},
    }
    batter_logs: dict[str, list[dict]] = {}
    for bid in list(range(1001, 1010)) + list(range(2001, 2010)) + [3001]:
        batter_logs[f"{bid}|2026"] = [
            {"d": "2026-03-26", "ab": 4, "h": 1, "bb": 0, "hbp": 0, "sf": 0, "tb": 1, "ibb": 0, "2b": 0, "3b": 0, "hr": 0},
            {"d": "2026-03-28", "ab": 4, "h": 2, "bb": 0, "hbp": 0, "sf": 0, "tb": 3, "ibb": 0, "2b": 1, "3b": 0, "hr": 0},
            {"d": "2026-03-30", "ab": 3, "h": 1, "bb": 1, "hbp": 0, "sf": 0, "tb": 1, "ibb": 0, "2b": 0, "3b": 0, "hr": 0},
            {"d": "2026-04-02", "ab": 4, "h": 1, "bb": 0, "hbp": 0, "sf": 0, "tb": 4, "ibb": 0, "2b": 0, "3b": 0, "hr": 1},
            {"d": "2026-04-05", "ab": 4, "h": 0, "bb": 0, "hbp": 0, "sf": 0, "tb": 0, "ibb": 0, "2b": 0, "3b": 0, "hr": 0},
        ]

    def batter_stats(bid: int, ymd: str) -> dict:
        log = batter_logs.get(f"{bid}|2026")
        return {
            "ops": data.batter_ops_as_of(log, ymd),
            "woba": data.batter_woba_as_of(log, ymd),
            "iso": data.batter_iso_as_of(log, ymd),
            "recentOps": data.batter_recent_ops_as_of(log, ymd),
            "momentum": data.batter_momentum_as_of(log, ymd),
            "games7": data.batter_games7_as_of(log, ymd),
        }

    def naive_lineups(games: list[dict]) -> list[dict]:
        out = []
        for g in games:
            lu = lineups.get(g["gamePk"])
            if not lu:
                out.append(g)
                continue

            def with_ops(side):
                if not side:
                    return side
                return {
                    "battingOrder": [{**p, **batter_stats(p["id"], g["date"])} for p in side["battingOrder"]],
                    "bench": [{**p, **batter_stats(p["id"], g["date"])} for p in side["bench"]],
                }

            home = with_ops(lu.get("home"))
            away = with_ops(lu.get("away"))
            home_ops = data.lineup_ops(home["battingOrder"]) if home else 0.0
            away_ops = data.lineup_ops(away["battingOrder"]) if away else 0.0
            home_known = home_ops > 0
            away_known = away_ops > 0

            def ls(order, known):
                return {
                    "known": known,
                    "ops": data.lineup_ops(order) if known else 0.0,
                    "woba": data._lineup_weighted(order, "woba") if known else 0.0,
                    "iso": data._lineup_weighted(order, "iso") if known else 0.0,
                    "recentOps": data._lineup_weighted(order, "recentOps") if known else 0.0,
                    "momentum": data._lineup_weighted(order, "momentum") if known else 0.0,
                    "games7": data._lineup_weighted(order, "games7") if known else 0.0,
                }

            out.append({
                **g,
                "lineups": {"home": home, "away": away},
                "lineupStats": {
                    "home": ls(home["battingOrder"] if home else None, home_known),
                    "away": ls(away["battingOrder"] if away else None, away_known),
                },
            })
        return out

    ptr_lu = data.attach_lineups_as_of([dict(g) for g in games], lineups, batter_logs)
    naive_lu = naive_lineups(games)
    check("lineup as-of: pointer == naive on unsorted games", ptr_lu == naive_lu, "output mismatch")


def test_data_fingerprint() -> None:
    print("data fingerprint")
    from mlb_streamlit import refresh

    ga = {"gamePk": 1, "date": "2026-04-05", "winner": "home", "home": {"score": 5}, "away": {"score": 2}}
    gb = {"gamePk": 2, "date": "2026-04-06", "winner": "away", "home": {"score": 3}, "away": {"score": 4}}
    injuries = {108: 1, 119: 0}
    fp = refresh._data_fingerprint([ga, gb], dict(injuries))
    check("fingerprint stable across runs", fp == refresh._data_fingerprint([dict(ga), dict(gb)], dict(injuries)))
    check("fingerprint order-invariant", fp == refresh._data_fingerprint([gb, ga], dict(injuries)))
    changed = dict(ga)
    changed["home"] = {"score": 6}
    check("fingerprint sensitive to score", fp != refresh._data_fingerprint([changed, gb], dict(injuries)))
    check("fingerprint sensitive to injury counts", fp != refresh._data_fingerprint([ga, gb], {108: 2, 119: 0}))
    changed = dict(ga)
    changed["winner"] = "away"
    check("fingerprint sensitive to winner", fp != refresh._data_fingerprint([changed, gb], dict(injuries)))
    check("fingerprint sensitive to game set", fp != refresh._data_fingerprint([ga], dict(injuries)))
    check("fingerprint is a hex digest", len(fp) == 64 and all(c in "0123456789abcdef" for c in fp))


def test_refresh_fast_path() -> None:
    print("refresh fast path")
    import shutil
    import tempfile
    import mlb_streamlit.data as data
    from mlb_streamlit import refresh

    completed = make_games(60, seed=3)
    for g in completed:
        g["status"] = {"abstractGameState": "Final", "detailedState": "Final"}
        g["dayNight"] = "day"
    today = (date.fromisoformat(max(g["date"] for g in completed)) + timedelta(days=1)).isoformat()
    upcoming_pk = 2026099901
    upcoming = {
        "gamePk": upcoming_pk,
        "date": today,
        "gameDate": f"{today}T18:00:00Z",
        "season": "2026",
        "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
        "dayNight": "night",
        "home": {"id": 119, "name": "Dodgers", "abbrev": "LAD", "score": None},
        "away": {"id": 108, "name": "Angels", "abbrev": "LAA", "score": None},
        "homePitcher": {"id": 11901, "name": "LAD SP"},
        "awayPitcher": {"id": 10802, "name": "LAA SP"},
        "weather": {"tempF": 78, "windMph": 6},
    }
    team_ids = sorted({tid for g in completed for tid in (g["home"]["id"], g["away"]["id"]) if tid > 0})
    injury_snapshot = {tid: 0 for tid in team_ids}
    fp = refresh._data_fingerprint(completed, injury_snapshot)

    lineup = {
        upcoming_pk: {
            "home": {"battingOrder": [{"id": 1001 + i, "name": f"H{i}", "pos": "OF"} for i in range(9)], "bench": []},
            "away": {"battingOrder": [{"id": 2001 + i, "name": f"A{i}", "pos": "OF"} for i in range(9)], "bench": []},
        }
    }

    def rel(days_back: int) -> str:
        return (date.fromisoformat(today) - timedelta(days=days_back)).isoformat()

    pitcher_logs = {
        "11901|2026": [
            {"d": rel(25), "ip": 6.0, "er": 2, "so": 5, "bb": 2, "hbp": 0, "hr": 1, "h": 3},
            {"d": rel(19), "ip": 7.0, "er": 1, "so": 8, "bb": 1, "hbp": 1, "hr": 0, "h": 2},
            {"d": rel(12), "ip": 5.0, "er": 3, "so": 4, "bb": 3, "hbp": 0, "hr": 1, "h": 5},
            {"d": rel(5), "ip": 6.0, "er": 4, "so": 6, "bb": 2, "hbp": 0, "hr": 2, "h": 6},
        ],
        "10802|2026": [
            {"d": rel(24), "ip": 6.0, "er": 3, "so": 4, "bb": 2, "hbp": 0, "hr": 1, "h": 4},
            {"d": rel(18), "ip": 5.0, "er": 2, "so": 6, "bb": 1, "hbp": 0, "hr": 0, "h": 3},
            {"d": rel(11), "ip": 7.0, "er": 3, "so": 7, "bb": 2, "hbp": 0, "hr": 1, "h": 5},
            {"d": rel(4), "ip": 6.0, "er": 2, "so": 5, "bb": 1, "hbp": 0, "hr": 1, "h": 4},
        ],
    }
    team_logs: dict[str, dict] = {}
    for tid in (108, 119):
        team_logs[f"{tid}|2026"] = {
            "hitting": [
                {"d": rel(24), "ab": 34, "h": 9, "bb": 3, "hbp": 1, "sf": 1, "tb": 14},
                {"d": rel(16), "ab": 36, "h": 10, "bb": 2, "hbp": 0, "sf": 0, "tb": 16},
                {"d": rel(8), "ab": 35, "h": 8, "bb": 4, "hbp": 0, "sf": 1, "tb": 12},
            ],
            "pitching": [
                {"d": rel(24), "ip": 9.0, "er": 3, "so": 8, "bb": 2, "h": 7},
                {"d": rel(16), "ip": 9.0, "er": 2, "so": 9, "bb": 1, "h": 5},
                {"d": rel(8), "ip": 9.0, "er": 4, "so": 7, "bb": 3, "h": 8},
            ],
            "fielding": [
                {"d": rel(24), "po": 27, "a": 9, "e": 1},
                {"d": rel(16), "po": 27, "a": 10, "e": 0},
                {"d": rel(8), "po": 27, "a": 8, "e": 2},
            ],
        }
    batter_logs: dict[str, list[dict]] = {}
    for bid in list(range(1001, 1010)) + list(range(2001, 2010)):
        batter_logs[f"{bid}|2026"] = [
            {"d": rel(20), "ab": 4, "h": 1, "bb": 0, "hbp": 0, "sf": 0, "tb": 1, "ibb": 0, "2b": 0, "3b": 0, "hr": 0},
            {"d": rel(17), "ab": 4, "h": 2, "bb": 0, "hbp": 0, "sf": 0, "tb": 3, "ibb": 0, "2b": 1, "3b": 0, "hr": 0},
            {"d": rel(14), "ab": 3, "h": 1, "bb": 1, "hbp": 0, "sf": 0, "tb": 1, "ibb": 0, "2b": 0, "3b": 0, "hr": 0},
            {"d": rel(11), "ab": 4, "h": 1, "bb": 0, "hbp": 0, "sf": 0, "tb": 4, "ibb": 0, "2b": 0, "3b": 0, "hr": 1},
            {"d": rel(8), "ab": 4, "h": 0, "bb": 0, "hbp": 0, "sf": 0, "tb": 0, "ibb": 0, "2b": 0, "3b": 0, "hr": 0},
        ]

    state = {
        "dataFingerprint": fp,
        "featureNames": [],
        "weights": [],
        "bias": 0.0,
        "featureStats": {},
        "isotonicPoints": [],
        "monteCarloEnabled": False,
        "monteCarloSigma": 0,
        "eloHfa": 30,
        "powerRankings": [
            {"teamId": 119, "elo": 1520.0, "last10WinPct": 0.6, "lastGameDate": today, "wins": 32, "losses": 20, "injuries": 0},
            {"teamId": 108, "elo": 1490.0, "last10WinPct": 0.5, "lastGameDate": today, "wins": 27, "losses": 25, "injuries": 0},
        ],
        "runModel": {
            "leagueRuns": 4.5,
            "teamOffense": {tid: 1.0 for tid in TIDS},
            "teamDefense": {tid: 1.0 for tid in TIDS},
            "parkFactor": {tid: 1.0 for tid in TIDS},
        },
        "runLineCalibration": [],
        "runMarginCalibration": {"slope": 0, "intercept": 0},
        "gamesTrained": 120,
        "holdoutCount": 20,
        "auc": 0.62,
        "brier": 0.23,
        "logLoss": 0.66,
        "ece": 0.02,
        "selectedModel": "Logistic regression",
        "asOfDate": "2026-08-15",
        "trainedAt": 1,
    }

    tmp = tempfile.mkdtemp(prefix="mlb_cache_")
    old_dir = cache.CACHE_DIR
    cache.CACHE_DIR = Path(tmp)
    cache.save_games(completed)
    cache.save_model_state(state)
    cache.save_lineups(lineup)
    cache.save_pitcher_logs(pitcher_logs)
    cache.save_team_logs(team_logs)
    cache.save_batter_logs(batter_logs)

    calls = {"schedule": 0, "injury": 0, "lineups": 0, "pitcher": 0, "team": 0, "batter": 0}
    stages: list[str] = []
    originals: dict[str, object] = {}

    def patch(name, fn):
        originals[name] = getattr(refresh, name)
        setattr(refresh, name, fn)

    def patch_data(name, fn):
        originals[f"data.{name}"] = getattr(data, name)
        setattr(data, name, fn)

    def fake_schedule(start, end):
        calls["schedule"] += 1
        return [upcoming]

    def fake_injury(team_ids, d, season):
        calls["injury"] += 1
        return {tid: 0 for tid in team_ids}

    def fake_lineups(targets, concurrency=16):
        calls["lineups"] += 1
        out = {}
        for g in targets:
            if g["gamePk"] == upcoming_pk:
                out[g["gamePk"]] = lineup[upcoming_pk]
        return out

    def fake_pitcher(pairs, cached=None):
        calls["pitcher"] += 1
        return {}

    def fake_team(pairs, cached=None):
        calls["team"] += 1
        return {}

    def fake_batter(ids, season, cached=None):
        calls["batter"] += 1
        return {}

    try:
        patch("fetch_schedule_range", fake_schedule)
        patch("fetch_current_injury_snapshot", fake_injury)
        patch("fetch_lineups_for_games", fake_lineups)
        patch("fetch_pitcher_game_logs", fake_pitcher)
        patch("fetch_team_game_logs", fake_team)
        patch("fetch_batter_game_logs", fake_batter)
        patch("fetch_market_odds", lambda: {})
        patch_data("fetch_bvp_stats", lambda pairs, cached=None: {})
        patch_data("fetch_platoon_splits", lambda pairs, cached=None: {})
        patch_data("fetch_vs_team_stats", lambda pairs, cached=None: {})
        patch_data("fetch_pitcher_hands", lambda ids: {})

        summary = refresh.run_refresh(report=lambda stage, pct, msg: stages.append(stage))
        docs = cache.load_docs_by_date()
        new_state = cache.load_model_state()
    finally:
        for name, fn in originals.items():
            if name.startswith("data."):
                setattr(data, name[len("data."):], fn)
            else:
                setattr(refresh, name, fn)
        cache.CACHE_DIR = old_dir
        shutil.rmtree(tmp, ignore_errors=True)

    check("fast path skips retraining", "Training model" not in stages, str(stages))
    check("fast path fetch budget: schedule once", calls["schedule"] == 1, str(calls))
    check("fast path fetch budget: injuries once", calls["injury"] == 1, str(calls))
    check("fast path fetch budget: window lineups once", calls["lineups"] == 1, str(calls))
    check("fast path fetch budget: no pitcher logs", calls["pitcher"] == 0, str(calls))
    check("fast path fetch budget: no team logs", calls["team"] == 0, str(calls))
    check("fast path fetch budget: no batter logs", calls["batter"] == 0, str(calls))
    check("fast path summary: stored metrics reused", summary.get("gamesTrained") == 120 and summary.get("auc") == 0.62,
          f"{summary.get('gamesTrained')}, {summary.get('auc')}")
    check("fast path summary: selected model preserved", summary.get("selectedModel") == "Logistic regression",
          str(summary.get("selectedModel")))
    check("fast path summary: one game scored", summary.get("storedGames") == 1, str(summary.get("storedGames")))
    day_docs = docs.get(today) or []
    check("fast path writes fresh docs", len(day_docs) == 1 and day_docs[0]["gamePk"] == upcoming_pk, f"{day_docs}")
    check("fast path doc has prediction", isinstance(day_docs[0].get("homeWinProb"), float),
          f"{day_docs[0].get('homeWinProb')}")
    check("fast path refreshes as-of date", isinstance(new_state.get("asOfDate"), str)
          and new_state.get("asOfDate") != "2026-08-15", str(new_state.get("asOfDate")))
    check("fast path keeps fingerprint", new_state.get("dataFingerprint") == fp)
    check("fast path keeps state lean (no embedded logs)",
          "pitcherLogs" not in new_state and "teamLogs" not in new_state and "batterLogs" not in new_state)


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

    # The parallelized paths (feature selection, CV folds, version fits) must
    # be deterministic: a second identical run produces byte-identical output.
    # This catches any thread-ordering race in the parallel helpers.
    out2 = run_model(games, season="2026", as_of_date="2026-08-15")
    r2 = out2["result"]
    check("run_model deterministic across runs (parallel paths)",
          r2["featureNames"] == result["featureNames"]
          and r2["auc"] == result["auc"]
          and r2["brier"] == result["brier"]
          and r2["modelVersions"] == result["modelVersions"]
          and r2["crossValidation"] == result["crossValidation"],
          "parallelized pipeline drifted between identical runs")

    # Candidate pool floor: every eligible model clears 0.70 AUC (or the floor
    # was relaxed because nothing cleared it on this synthetic data), selection
    # always comes from the eligible pool, and the stronger families are in it.
    relaxed = any(c.get("note", "").startswith("AUC floor relaxed") for c in result["candidates"])
    check("candidate floor: eligible >= 0.70 / excluded < 0.70 (or relaxed)",
          relaxed or all((c["auc"] >= 0.70) == bool(c.get("eligible")) for c in result["candidates"]),
          f"{[(c['name'], round(c['auc'], 3), c.get('eligible')) for c in result['candidates']]}")
    sel_c = next((c for c in result["candidates"] if c.get("selected")), None)
    check("selected model eligible (or relaxed)",
          sel_c is not None and (relaxed or bool(sel_c.get("eligible"))))
    pool_names = [c["name"] for c in result["candidates"]]
    for want in ("Distance-weighted k-NN (k=21)", "Boosted decision stumps", "Neural network (MLP)", "Logistic regression (L2, λ=1)"):
        check(f"candidate pool includes {want}", want in pool_names, f"{pool_names}")
    check("mc rationale reports the actual sigma grid",
          "0.1" in result["monteCarloRationale"] and "0.6" in result["monteCarloRationale"],
          f"{result['monteCarloRationale'][:140]}")
    check("mc trials consistent with toggle",
          result["monteCarloTrials"] == (10000 if result["monteCarloEnabled"] else 0))


def test_ensemble_models() -> None:
    print("ensemble model families")
    from mlb_streamlit.engine.ensemble import (
        boosted_stumps_model,
        weighted_knn_calib_preds,
        weighted_knn_model,
    )

    games = make_games(500, seed=21)
    rows = compute_elo_and_features(games, None, "2026-08-15")["rows"]
    train = rows[: int(len(rows) * 0.7)]
    calib = rows[int(len(rows) * 0.7): int(len(rows) * 0.85)]
    labels = [r["label"] for r in calib]

    # Boosted stumps: deterministic (fixed seed), valid probabilities, and
    # real signal on the synthetic season.
    b1 = boosted_stumps_model(train, FEATURE_KEYS, n_trees=30, seed=11)
    b2 = boosted_stumps_model(train, FEATURE_KEYS, n_trees=30, seed=11)
    p1 = [b1(r["features"]) for r in calib]
    check("boosted stumps deterministic", p1 == [b2(r["features"]) for r in calib])
    check("boosted stumps probabilities in (0,1)", all(0 < p < 1 for p in p1))
    check("boosted stumps AUC > 0.6", compute_auc(p1, labels) > 0.6,
          f"{compute_auc(p1, labels):.3f}")

    # Tiny training sets fall back to the prior (degenerate-case guard).
    tiny = boosted_stumps_model(train[:15], FEATURE_KEYS, n_trees=10)
    tp = [tiny(r["features"]) for r in calib[:5]]
    check("boosted stumps tiny-train prior fallback",
          all(abs(v - 0.5) < 0.15 for v in tp), f"{tp}")

    # Neural network (MLP): deterministic (fixed seed), valid probabilities,
    # and real signal on the synthetic season.
    from mlb_streamlit.engine.nn import mlp_model
    n1 = mlp_model(train, FEATURE_KEYS, seed=31)
    n2 = mlp_model(train, FEATURE_KEYS, seed=31)
    qn1 = [n1(r["features"]) for r in calib]
    check("mlp deterministic", qn1 == [n2(r["features"]) for r in calib])
    check("mlp probabilities in (0,1)", all(0 < p < 1 for p in qn1))
    check("mlp AUC > 0.6", compute_auc(qn1, labels) > 0.6, f"{compute_auc(qn1, labels):.3f}")
    tiny_nn = mlp_model(train[:15], FEATURE_KEYS)
    tnn = [tiny_nn(r["features"]) for r in calib[:5]]
    check("mlp tiny-train prior fallback", all(abs(v - 0.5) < 0.15 for v in tnn), f"{tnn}")

    # Distance-weighted k-NN: deterministic, valid, batch == scalar path.
    w1 = weighted_knn_model(train, FEATURE_KEYS, k=11)
    w2 = weighted_knn_model(train, FEATURE_KEYS, k=11)
    q1 = [w1(r["features"]) for r in calib]
    check("weighted kNN deterministic", q1 == [w2(r["features"]) for r in calib])
    check("weighted kNN probabilities in (0,1)", all(0 < p < 1 for p in q1))
    check("weighted kNN calib batch matches scalar",
          weighted_knn_calib_preds(train, calib, FEATURE_KEYS, k=11, model=w1) == q1)

    # Regularization strengths produce distinct-but-valid fits (candidate
    # pool diversity), with stronger ridge shrinking the coefficients.
    m_low = train_logistic(train, FEATURE_KEYS, iterations=10)
    m_high = train_logistic(train, FEATURE_KEYS, iterations=10, lambda_=1.0)
    check("lambda_ fit is a valid model",
          all(k in m_high for k in ("featureNames", "weights", "bias", "featureStats")))
    check("stronger lambda shrinks weights",
          sum(abs(w) for w in m_high["weights"]) < sum(abs(w) for w in m_low["weights"]) + 1e-9)


def test_parallel_refresh() -> None:
    print("parallel refresh paths")
    from mlb_streamlit.engine.metrics import parallel_map
    from mlb_streamlit.engine.model import simulate_runs_batch
    from mlb_streamlit.refresh import build_run_projection, postprocess_projection

    # 1) parallel_map preserves input order (deterministic) and falls back to
    #    serial for empty / single-worker inputs.
    items = list(range(30))
    check("parallel_map order-preserving",
          parallel_map(lambda i: i * i, items, max_workers=8) == [i * i for i in items])
    check("parallel_map empty -> []", parallel_map(lambda i: i, []) == [])
    check("parallel_map serial fallback",
          parallel_map(lambda i: i + 1, [1, 2, 3], max_workers=1) == [2, 3, 4])

    # 2) The vectorized batch projection (used for the fresh window) produces
    #    the same doc shape as the scalar per-game path.
    rm = {"leagueRuns": 4.5, "teamOffense": {119: 1.1, 108: 0.95},
          "teamDefense": {119: 0.98, 108: 1.05}, "parkFactor": {119: 1.02, 108: 1.0}}
    game = {"gamePk": 7, "date": "2026-08-20", "season": "2026",
            "home": {"id": 119, "name": "Dodgers", "abbrev": "LAD"},
            "away": {"id": 108, "name": "Angels", "abbrev": "LAA"}}
    scalar = build_run_projection(rm, [], game, 8.5, None, trials=500, home_win_prob=0.55, run_margin_cal=None)
    sim = simulate_runs_batch(rm, [119], [108], [8.5], [0.0], 500)[0]
    batched = postprocess_projection(sim, [], 1.5)
    keys = ("homeScore", "awayScore", "total", "overProb", "underProb",
            "homeRunLineProb", "awayRunLineProb")
    check("scalar projection has all keys", all(k in scalar for k in keys), f"{scalar}")
    check("batch projection has all keys", all(k in batched for k in keys), f"{batched}")
    check("batch projection values finite", all(isinstance(batched[k], (int, float)) for k in keys))
    check("batch projection scores rounded to 2dp",
          all(batched[k] == round(batched[k], 2) for k in ("homeScore", "awayScore", "total")))
    check("batch run-line probs sum ~ 1",
          abs(batched["homeRunLineProb"] + batched["awayRunLineProb"] - 1.0) < 1e-9)


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
    test_matchups()
    test_as_of_stats()
    test_runs_model()
    test_data_layer()
    test_market_odds()
    test_automl_pipeline()
    test_ensemble_models()
    test_parallel_refresh()
    test_cache_roundtrip()
    print(f"\nAll {_CHECKS} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
