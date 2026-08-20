"""Feature engineering + chronological Elo ratings.

Faithful port of the feature/state sections of src/convex/ml/model.ts.
All features are computed strictly as-of game time (no lookahead).
"""

from __future__ import annotations

import datetime as _dt

from .teams import PARK_FACTORS
from .metrics import clamp

ELO_INIT = 1500.0
ELO_HFA_UPDATE = 30.0  # home advantage baked into Elo updates only

# Bump whenever feature engineering changes so refresh fingerprints and every
# cached prediction/backtest are invalidated and rebuilt point-in-time.
FEATURE_VERSION = 9

# Fixed physical scales keep live weather and interaction terms in the same
# numeric domain as the training matrix. Logistic/MLP/kNN still fit their own
# train-only z-scores, but these bounds prevent raw API units (°F, mph, innings)
# from leaking into drift views or dominating non-linear members.
TEMP_REFERENCE_F = 72.0
TEMP_SCALE_F = 15.0
WIND_SCALE_MPH = 15.0
LINEUP_FATIGUE_GAMES = 7.0
FATIGUE_REST_DAYS = 4.0
FATIGUE_WORKLOAD_IP = 18.0

# This dataset is framed as one home-vs-away row per game. `homeField` is a
# structural reference row indicator (always 1), not a varying observation;
# it is retained in the serialized schema for compatibility but excluded from
# PSI/drift statistics below.
STRUCTURAL_FEATURES = frozenset(("homeField",))

# Canonical feature order (27 features). Every feature is computed as-of the
# game's own date (no lookahead) and flows into the ML candidate set; greedy
# backward elimination decides which ones the final model actually uses.
FEATURE_KEYS = [
    "eloDiff",
    "winPctDiff",
    "formDiff",
    "restDiff",
    "injuryDiff",
    "homeField",
    "spFipDiff",
    "spEraDiff",
    "opsDiff",
    "teamEraDiff",
    "defEffDiff",
    "parkFactor",
    "tempDev",
    "windMph",
    "lineupKnown",
    "lineupOpsDiff",
    "lineupWobaDiff",
    "lineupIsoDiff",
    "lineupHotDiff",
    "bvpOpsDiff",
    "platoonOpsDiff",
    "vsTeamOpsDiff",
    "spK9Diff",
    "spWhipDiff",
    "spRecentDiff",
    "teamK9Diff",
    "teamWhipDiff",
    "spTrendDiff",
    "spRestWorkloadDiff",
    "lineupMomentumDiff",
    "lineupFatigueDiff",
    "eloWinPctInteract",
    "spFipRestInteract",
    "lineupParkInteract",
]

# `homeField` is retained in FEATURE_KEYS for schema/UI compatibility, but it
# is a constant reference column in this one-row-per-game home perspective and
# must not enter a fitted statistical matrix alongside the intercept.
MODEL_FEATURE_KEYS = [f for f in FEATURE_KEYS if f not in STRUCTURAL_FEATURES]

FEATURE_LABELS = {
    "eloDiff": "Elo rating edge",
    "winPctDiff": "Win % edge",
    "formDiff": "Recent form (L10)",
    "restDiff": "Rest advantage",
    "injuryDiff": "Injury edge (IL)",
    "homeField": "Home field",
    "spFipDiff": "Starting Pitcher FIP / xERA Delta",
    "spEraDiff": "Starting Pitcher ERA Delta",
    "opsDiff": "Team OPS edge",
    "teamEraDiff": "Bullpen / Staff ERA edge",
    "defEffDiff": "Defensive efficiency edge",
    "parkFactor": "Ballpark factor",
    "tempDev": "Weather temperature",
    "windMph": "Weather wind",
    "lineupKnown": "Lineup data available",
    "lineupOpsDiff": "Starting-9 OPS edge",
    "lineupWobaDiff": "Starting-9 wOBA edge",
    "lineupIsoDiff": "Starting-9 ISO edge",
    "lineupHotDiff": "Starting-9 hot streak (L10 OPS)",
    "bvpOpsDiff": "Batter vs. Pitcher (BvP) edge",
    "platoonOpsDiff": "Platoon split edge (vs starter's hand)",
    "vsTeamOpsDiff": "Batter vs. team split edge",
    "spK9Diff": "Starting Pitcher K/9 Delta",
    "spWhipDiff": "Starting Pitcher WHIP Delta",
    "spRecentDiff": "Starting Pitcher last-3-start ERA Delta",
    "teamK9Diff": "Staff K/9 edge",
    "teamWhipDiff": "Staff WHIP edge",
    "spTrendDiff": "Starter recent-form trajectory (FIP slope)",
    "spRestWorkloadDiff": "Starter short-rest × workload interaction",
    "lineupMomentumDiff": "Starting-9 momentum (recent OPS vs season)",
    "lineupFatigueDiff": "Starting-9 fatigue (games in last 7 days)",
    "eloWinPctInteract": "Elo edge × win-% edge (rating × record)",
    "spFipRestInteract": "Starter FIP edge × rest advantage",
    "lineupParkInteract": "Lineup OPS edge × ballpark factor",
}


def shift_date(ymd: str, days: int) -> str:
    if not ymd:
        return ""
    try:
        d = _dt.date.fromisoformat(ymd)
    except ValueError:
        return ymd
    d = d + _dt.timedelta(days=days)
    return d.isoformat()


def days_between(from_ymd: str, to_ymd: str) -> int:
    if not from_ymd or not to_ymd:
        return 4
    try:
        a = _dt.date.fromisoformat(from_ymd)
        b = _dt.date.fromisoformat(to_ymd)
    except ValueError:
        return 4
    return (b - a).days


def new_state() -> dict:
    return {
        "elo": {},
        "formHistory": {},
        "lastGameDate": {},
        "records": {},
        "injuries": {},
        "runDiff": {},
        "homeRecords": {},
        "awayRecords": {},
    }


def form_of(state: dict, team_id: int) -> float:
    h = state["formHistory"].get(team_id)
    if not h:
        return 0.5
    return sum(h) / len(h)


def starter_delta(home_pitcher, away_pitcher, key: str) -> float:
    """Starting-pitcher delta (away - home) so positive values favor home."""
    away = (away_pitcher or {}).get(key)
    home = (home_pitcher or {}).get(key)
    if away is None or home is None:
        return 0.0
    return away - home


def _edge(home, away, lower_better: bool = False) -> float:
    """Signed edge so positive values favor home (0 when either side is missing)."""
    if not isinstance(home, (int, float)) or not isinstance(away, (int, float)):
        return 0.0
    return (away - home) if lower_better else (home - away)


def _finite_float(value) -> float | None:
    """Coerce numeric/string API values without admitting NaN or infinities."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _sp_fatigue(pitcher) -> float:
    """Normalized short-rest × recent-workload interaction in ``[0, 1]``.

    The previous product ``rest deficit * innings`` was unbounded (a starter
    with a large recent workload could create a 100+ feature value), which
    made the live interaction distribution incomparable with the baseline.
    Both factors now use fixed baseball-domain scales and are clipped before
    multiplication. Missing prior starts remain neutral at zero.
    """
    rest = (pitcher or {}).get("restDays")
    workload = (pitcher or {}).get("workload")
    if not isinstance(rest, (int, float)) or not isinstance(workload, (int, float)):
        return 0.0
    rest_deficit = clamp((5.0 - rest) / FATIGUE_REST_DAYS, 0.0, 1.0)
    workload_factor = clamp(workload / FATIGUE_WORKLOAD_IP, 0.0, 1.0)
    return round(rest_deficit * workload_factor, 4)


def build_features(game: dict, state: dict) -> dict:
    home_elo = state["elo"].get(game["home"]["id"], ELO_INIT)
    away_elo = state["elo"].get(game["away"]["id"], ELO_INIT)

    home_rec = state["records"].get(game["home"]["id"], {"wins": 0, "losses": 0})
    away_rec = state["records"].get(game["away"]["id"], {"wins": 0, "losses": 0})
    # Point-in-time records: always use the chronologically accumulated state,
    # never the schedule's leagueRecord. For a completed game the leagueRecord
    # is the team's record AFTER that game, so using it would leak the game's
    # own outcome into winPctDiff (the source of the old "99%" past picks).
    home_wins = home_rec["wins"]
    home_losses = home_rec["losses"]
    away_wins = away_rec["wins"]
    away_losses = away_rec["losses"]
    home_wp = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0.5
    away_wp = away_wins / (away_wins + away_losses) if (away_wins + away_losses) > 0 else 0.5

    home_rest = clamp(days_between(state["lastGameDate"].get(game["home"]["id"], ""), game["date"]), 0, 10)
    away_rest = clamp(days_between(state["lastGameDate"].get(game["away"]["id"], ""), game["date"]), 0, 10)

    home_ops = game["home"].get("ops")
    away_ops = game["away"].get("ops")
    home_team_era = game["home"].get("era")
    away_team_era = game["away"].get("era")
    home_fielding = game["home"].get("fieldingPct")
    away_fielding = game["away"].get("fieldingPct")
    home_k9 = game["home"].get("k9")
    away_k9 = game["away"].get("k9")
    home_whip = game["home"].get("whip")
    away_whip = game["away"].get("whip")
    temp_f = _finite_float((game.get("weather") or {}).get("tempF"))
    wind = _finite_float((game.get("weather") or {}).get("windMph"))

    # Actual starting-9 / bench data. When lineups are missing (older
    # historical games) the feature defaults to 0 with lineupKnown = 0.
    lineup_home = (game.get("lineupStats") or {}).get("home")
    lineup_away = (game.get("lineupStats") or {}).get("away")
    home_lineup_known = 1 if (lineup_home or {}).get("known") is True else 0
    away_lineup_known = 1 if (lineup_away or {}).get("known") is True else 0
    lineup_known = 1 if home_lineup_known == 1 and away_lineup_known == 1 else 0
    # Matchup splits (BvP / platoon / vs-team) are fetched as-of *now* with no
    # as-of filter. A decided game must never consume them, or its own (and
    # later) results would leak into the prediction.
    matchup_known = game.get("winner") not in ("home", "away")
    if isinstance((lineup_home or {}).get("ops"), (int, float)) and isinstance((lineup_away or {}).get("ops"), (int, float)):
        lineup_ops_diff = lineup_home["ops"] - lineup_away["ops"]
    else:
        lineup_ops_diff = 0.0

    return {
        "eloDiff": (home_elo - away_elo) / 100,
        "winPctDiff": home_wp - away_wp,
        "formDiff": form_of(state, game["home"]["id"]) - form_of(state, game["away"]["id"]),
        "restDiff": clamp(home_rest - away_rest, -4, 4),
        "injuryDiff": (state["injuries"].get(game["away"]["id"], 0)) - (state["injuries"].get(game["home"]["id"], 0)),
        "homeField": 1.0,
        "spFipDiff": starter_delta(game.get("homePitcher"), game.get("awayPitcher"), "fip"),
        "spEraDiff": starter_delta(game.get("homePitcher"), game.get("awayPitcher"), "era"),
        "opsDiff": (home_ops - away_ops) if isinstance(home_ops, (int, float)) and isinstance(away_ops, (int, float)) else 0.0,
        "teamEraDiff": (away_team_era - home_team_era) if isinstance(away_team_era, (int, float)) and isinstance(home_team_era, (int, float)) else 0.0,
        "defEffDiff": (home_fielding - away_fielding) if isinstance(home_fielding, (int, float)) and isinstance(away_fielding, (int, float)) else 0.0,
        "parkFactor": PARK_FACTORS.get(game["home"]["id"], 1.0),
        # Store weather in bounded, dimensionless units. The raw API payload
        # remains Fahrenheit/mph in the game document; only the model vector is
        # normalized here so baseline and live feeds share one contract.
        "tempDev": clamp((temp_f - TEMP_REFERENCE_F) / TEMP_SCALE_F, -3.0, 3.0)
        if temp_f is not None else 0.0,
        "windMph": clamp(wind / WIND_SCALE_MPH, 0.0, 3.0)
        if wind is not None else 0.0,
        "lineupKnown": lineup_known,
        "lineupOpsDiff": lineup_ops_diff,
        "lineupWobaDiff": _edge((lineup_home or {}).get("woba"), (lineup_away or {}).get("woba")),
        "lineupIsoDiff": _edge((lineup_home or {}).get("iso"), (lineup_away or {}).get("iso")),
        "lineupHotDiff": _edge((lineup_home or {}).get("recentOps"), (lineup_away or {}).get("recentOps")),
        # Matchup edges: career BvP OPS, season platoon OPS vs the starter's
        # throwing hand, season OPS vs the opposing team — PA-saturated,
        # slot-weighted means over the real starting 9 (0 + lineupKnown = 0
        # when no boxscore lineup / opposing starter is known). Zeroed for
        # decided games so as-of-now splits can never leak a result back in.
        "bvpOpsDiff": _edge((lineup_home or {}).get("bvpOps"), (lineup_away or {}).get("bvpOps")) if matchup_known else 0.0,
        "platoonOpsDiff": _edge((lineup_home or {}).get("platoonOps"), (lineup_away or {}).get("platoonOps")) if matchup_known else 0.0,
        "vsTeamOpsDiff": _edge((lineup_home or {}).get("vsTeamOps"), (lineup_away or {}).get("vsTeamOps")) if matchup_known else 0.0,
        "spK9Diff": _edge((game.get("homePitcher") or {}).get("k9"), (game.get("awayPitcher") or {}).get("k9")),
        "spWhipDiff": _edge((game.get("homePitcher") or {}).get("whip"), (game.get("awayPitcher") or {}).get("whip"), True),
        "spRecentDiff": _edge((game.get("homePitcher") or {}).get("recentEra"), (game.get("awayPitcher") or {}).get("recentEra"), True),
        "teamK9Diff": _edge(home_k9, away_k9),
        "teamWhipDiff": _edge(home_whip, away_whip, True),
        # Trajectory features from the cached game logs (no lookahead, and
        # only populated where the underlying data really exists): starter
        # FIP trend slope (lower = improving), the short-rest x workload
        # interaction, and lineup-level batter momentum / fatigue.
        "spTrendDiff": _edge((game.get("homePitcher") or {}).get("trendFip"), (game.get("awayPitcher") or {}).get("trendFip"), True),
        "spRestWorkloadDiff": _edge(_sp_fatigue(game.get("homePitcher")), _sp_fatigue(game.get("awayPitcher")), True),
        "lineupMomentumDiff": _edge((lineup_home or {}).get("momentum"), (lineup_away or {}).get("momentum")),
        "lineupFatigueDiff": clamp(
            _edge((lineup_home or {}).get("games7"), (lineup_away or {}).get("games7"), True)
            / LINEUP_FATIGUE_GAMES,
            -1.0,
            1.0,
        ),
        # Interaction features (combinations of the point-in-time edges above,
        # so they are themselves strictly prior to first pitch).
        "eloWinPctInteract": ((home_elo - away_elo) / 100) * (home_wp - away_wp),
        "spFipRestInteract": clamp(
            (starter_delta(game.get("homePitcher"), game.get("awayPitcher"), "fip") / 3.0)
            * (clamp(home_rest - away_rest, -4, 4) / 4.0),
            -3.0,
            3.0,
        ),
        "lineupParkInteract": lineup_ops_diff * (PARK_FACTORS.get(game["home"]["id"], 1.0) - 1.0),
    }


def update_state(state: dict, game: dict) -> None:
    home = game["home"]["id"]
    away = game["away"]["id"]
    home_elo = state["elo"].get(home, ELO_INIT)
    away_elo = state["elo"].get(away, ELO_INIT)

    expected_home = 1 / (1 + 10 ** -(((home_elo + ELO_HFA_UPDATE) - away_elo) / 400))
    home_actual = 1 if game["winner"] == "home" else 0
    margin = abs((game["home"].get("score") or 0) - (game["away"].get("score") or 0))
    k = 24 * math_sqrt(max(1, margin))
    delta = k * (home_actual - expected_home)
    state["elo"][home] = home_elo + delta
    state["elo"][away] = away_elo - delta

    hh = state["formHistory"].setdefault(home, [])
    hh.append(home_actual)
    if len(hh) > 10:
        hh.pop(0)
    ah = state["formHistory"].setdefault(away, [])
    ah.append(1 - home_actual)
    if len(ah) > 10:
        ah.pop(0)

    hr = state["records"].setdefault(home, {"wins": 0, "losses": 0})
    ar = state["records"].setdefault(away, {"wins": 0, "losses": 0})
    if home_actual == 1:
        hr["wins"] += 1
        ar["losses"] += 1
    else:
        hr["losses"] += 1
        ar["wins"] += 1

    h_score = game["home"].get("score") or 0
    a_score = game["away"].get("score") or 0
    state["runDiff"][home] = state["runDiff"].get(home, 0) + (h_score - a_score)
    state["runDiff"][away] = state["runDiff"].get(away, 0) + (a_score - h_score)

    h_home = state["homeRecords"].setdefault(home, {"wins": 0, "losses": 0})
    a_away = state["awayRecords"].setdefault(away, {"wins": 0, "losses": 0})
    if home_actual == 1:
        h_home["wins"] += 1
        a_away["losses"] += 1
    else:
        h_home["losses"] += 1
        a_away["wins"] += 1

    state["lastGameDate"][home] = game["date"]
    state["lastGameDate"][away] = game["date"]


def math_sqrt(x: float) -> float:
    import math
    return math.sqrt(x)


def lookup_injuries(team_id: int, date: str, snapshots) -> int:
    """Most recent injury snapshot on or before `date` (no lookahead)."""
    if not snapshots:
        return 0
    # JSON object keys are strings; accept both in-memory integer maps and
    # disk-loaded snapshots so injury features do not silently become zero
    # after a restart.
    lst = snapshots.get(team_id) or snapshots.get(str(team_id))
    if not lst:
        return 0
    best = 0
    for s in lst:
        if s["date"] > date:
            break
        best = s["count"]
    return best


def compute_elo_and_features(games: list[dict], injury_snapshots=None, latest_date: str | None = None):
    """Chronological pass producing feature rows + current team state."""
    sorted_games = sorted(games, key=lambda g: g.get("gameDate") or g.get("date") or "")
    state = new_state()
    rows: list[dict] = []
    for game in sorted_games:
        if game["winner"] not in ("home", "away"):
            continue
        # Defensive guard: drop only genuinely malformed cache rows (no team
        # ids) — never actual results. Every decided game with valid team ids
        # still produces a training row.
        if not (game.get("home") or {}).get("id") or not (game.get("away") or {}).get("id"):
            continue
        state["injuries"][game["home"]["id"]] = lookup_injuries(game["home"]["id"], game["date"], injury_snapshots)
        state["injuries"][game["away"]["id"]] = lookup_injuries(game["away"]["id"], game["date"], injury_snapshots)
        features = build_features(game, state)
        rows.append({
            "game": game,
            "features": features,
            "homeElo": state["elo"].get(game["home"]["id"], ELO_INIT),
            "awayElo": state["elo"].get(game["away"]["id"], ELO_INIT),
            "label": 1 if game["winner"] == "home" else 0,
        })
        update_state(state, game)
    # Refresh injury counts to the latest snapshot so upcoming-game
    # predictions use current roster state.
    if latest_date and injury_snapshots:
        for tid in list(state["elo"].keys()):
            state["injuries"][tid] = lookup_injuries(tid, latest_date, injury_snapshots)
    team_state = {
        "elo": state["elo"],
        "form": {},
        "lastGameDate": state["lastGameDate"],
        "records": state["records"],
        "injuries": state["injuries"],
    }
    for tid in state["formHistory"]:
        team_state["form"][tid] = form_of(state, tid)
    team_stats = {
        "runDiff": state["runDiff"],
        "homeRecords": state["homeRecords"],
        "awayRecords": state["awayRecords"],
    }
    return {"rows": rows, "teamState": team_state, "teamStats": team_stats}


def build_features_for_game(game: dict, team_state: dict) -> dict:
    """Features for a not-yet-seen game given the current team state."""
    mut = new_state()
    mut["elo"] = team_state["elo"]
    mut["lastGameDate"] = team_state["lastGameDate"]
    mut["records"] = team_state["records"]
    mut["injuries"] = team_state["injuries"]
    # Provide form as a synthetic 10-game history so build_features reuses it.
    for tid, p in (team_state.get("form") or {}).items():
        wins = int(round(p * 10))
        mut["formHistory"][tid] = [1 if i < wins else 0 for i in range(10)]
    return build_features(game, mut)
