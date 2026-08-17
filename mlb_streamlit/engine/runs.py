"""Run-scoring model: Poisson offense/defense factors + Monte Carlo.

Faithful port of src/convex/ml/runs.ts. Each team's runs are modeled as
independent Poisson variables with mean = leagueAvg * offense * defense * park,
then a deterministic xorshift PRNG drives the Monte Carlo simulation so
results are reproducible.
"""

from __future__ import annotations

import math


def clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def make_rand(seed: int):
    """Deterministic xorshift PRNG so Monte Carlo results are reproducible."""
    s = (seed & 0xFFFFFFFF) or 0x12345678

    def rand() -> float:
        nonlocal s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        s &= 0xFFFFFFFF
        return s / 4294967296.0

    return rand


def poisson(lambda_: float, rand) -> int:
    """Knuth's Poisson sampler."""
    if lambda_ <= 0:
        return 0
    L = math.exp(-lambda_)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rand()
        if p <= L:
            break
    return k - 1


def fit_run_model(games: list[dict]) -> dict:
    """Fit league average + per-team offense/defense/park factors from scores."""
    team_games: dict[int, int] = {}
    team_scored: dict[int, float] = {}
    team_allowed: dict[int, float] = {}
    park_runs: dict[int, dict] = {}
    total_runs = 0
    total_games = 0

    for g in games:
        if g["winner"] not in ("home", "away"):
            continue
        hs = g["home"].get("score")
        as_ = g["away"].get("score")
        if not isinstance(hs, (int, float)) or not isinstance(as_, (int, float)):
            continue
        home_id = g["home"]["id"]
        away_id = g["away"]["id"]
        team_games[home_id] = team_games.get(home_id, 0) + 1
        team_games[away_id] = team_games.get(away_id, 0) + 1
        team_scored[home_id] = team_scored.get(home_id, 0) + hs
        team_allowed[home_id] = team_allowed.get(home_id, 0) + as_
        team_scored[away_id] = team_scored.get(away_id, 0) + as_
        team_allowed[away_id] = team_allowed.get(away_id, 0) + hs
        pr = park_runs.get(home_id, {"runs": 0, "games": 0})
        pr["runs"] += hs + as_
        pr["games"] += 1
        park_runs[home_id] = pr
        total_runs += hs + as_
        total_games += 1

    league_runs = total_runs / (2 * total_games) if total_games > 0 else 4.4

    team_offense: dict[int, float] = {}
    team_defense: dict[int, float] = {}
    park_factor: dict[int, float] = {}
    for tid, g_count in team_games.items():
        team_offense[tid] = clamp((team_scored.get(tid, 0) / g_count) / league_runs, 0.7, 1.3)
        team_defense[tid] = clamp((team_allowed.get(tid, 0) / g_count) / league_runs, 0.7, 1.3)
    for tid, p in park_runs.items():
        park_factor[tid] = clamp(p["runs"] / p["games"] / (2 * league_runs), 0.85, 1.15)

    return {
        "leagueRuns": league_runs,
        "teamOffense": team_offense,
        "teamDefense": team_defense,
        "parkFactor": park_factor,
    }


def expected_margin(model: dict, home_id: int, away_id: int) -> float:
    park_mul = model["parkFactor"].get(home_id, 1)
    lr = model["leagueRuns"]
    lambda_home = lr * model["teamOffense"].get(home_id, 1) * model["teamDefense"].get(away_id, 1) * park_mul
    lambda_away = lr * model["teamOffense"].get(away_id, 1) * model["teamDefense"].get(home_id, 1) * park_mul
    return lambda_home - lambda_away


def expected_total(model: dict, home_id: int, away_id: int) -> float:
    park_mul = model["parkFactor"].get(home_id, 1)
    lr = model["leagueRuns"]
    return (
        lr
        * (
            model["teamOffense"].get(home_id, 1) * model["teamDefense"].get(away_id, 1)
            + model["teamOffense"].get(away_id, 1) * model["teamDefense"].get(home_id, 1)
        )
        * park_mul
    )


def simulate_runs(
    model: dict,
    home_id: int,
    away_id: int,
    line: float,
    trials: int = 10000,
    run_line: float = 1.5,
    margin_shift: float = 0.0,
) -> dict:
    """Monte Carlo run simulation. `marginShift` reconciles scores with the
    win-probability model (total preserved)."""
    offense = model["teamOffense"]
    defense = model["teamDefense"]
    park = model["parkFactor"]
    park_mul = park.get(home_id, 1)
    base_home = model["leagueRuns"] * offense.get(home_id, 1) * defense.get(away_id, 1) * park_mul
    base_away = model["leagueRuns"] * offense.get(away_id, 1) * defense.get(home_id, 1) * park_mul
    shift = clamp(margin_shift, -min(base_home, base_away) + 0.05, min(base_home, base_away) - 0.05)
    lambda_home = base_home + shift
    lambda_away = base_away - shift
    cover_threshold = math.ceil(run_line)  # 1.5 -> 2, 2.5 -> 3

    rand = make_rand((home_id * 1000003 + away_id * 7919) & 0xFFFFFFFF)
    home_sum = 0
    away_sum = 0
    over = 0
    under = 0
    home_cover = 0
    away_cover = 0
    for _ in range(trials):
        hs = poisson(lambda_home, rand)
        as_ = poisson(lambda_away, rand)
        home_sum += hs
        away_sum += as_
        total = hs + as_
        if total > line:
            over += 1
        elif total < line:
            under += 1
        margin = hs - as_
        if margin >= cover_threshold:
            home_cover += 1
        else:
            away_cover += 1

    over_under = over + under
    return {
        "homeScore": home_sum / trials,
        "awayScore": away_sum / trials,
        "total": (home_sum + away_sum) / trials,
        "overProb": (over / over_under) if over_under > 0 else 0.5,
        "underProb": (under / over_under) if over_under > 0 else 0.5,
        "homeRunLineProb": home_cover / trials,
        "awayRunLineProb": away_cover / trials,
    }
