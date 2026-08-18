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
]

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


def build_features(game: dict, state: dict) -> dict:
    home_elo = state["elo"].get(game["home"]["id"], ELO_INIT)
    away_elo = state["elo"].get(game["away"]["id"], ELO_INIT)

    home_rec = state["records"].get(game["home"]["id"], {"wins": 0, "losses": 0})
    away_rec = state["records"].get(game["away"]["id"], {"wins": 0, "losses": 0})
    home_wins = game["home"].get("wins") if game["home"].get("wins") is not None else home_rec["wins"]
    home_losses = game["home"].get("losses") if game["home"].get("losses") is not None else home_rec["losses"]
    away_wins = game["away"].get("wins") if game["away"].get("wins") is not None else away_rec["wins"]
    away_losses = game["away"].get("losses") if game["away"].get("losses") is not None else away_rec["losses"]
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
    temp_f = (game.get("weather") or {}).get("tempF")
    wind = (game.get("weather") or {}).get("windMph")

    # Actual starting-9 / bench data. When lineups are missing (older
    # historical games) the feature defaults to 0 with lineupKnown = 0.
    lineup_home = (game.get("lineupStats") or {}).get("home")
    lineup_away = (game.get("lineupStats") or {}).get("away")
    home_lineup_known = 1 if (lineup_home or {}).get("known") is True else 0
    away_lineup_known = 1 if (lineup_away or {}).get("known") is True else 0
    lineup_known = 1 if home_lineup_known == 1 and away_lineup_known == 1 else 0
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
        "tempDev": (temp_f - 72) if isinstance(temp_f, (int, float)) else 0.0,
        "windMph": wind if isinstance(wind, (int, float)) else 0.0,
        "lineupKnown": lineup_known,
        "lineupOpsDiff": lineup_ops_diff,
        "lineupWobaDiff": _edge((lineup_home or {}).get("woba"), (lineup_away or {}).get("woba")),
        "lineupIsoDiff": _edge((lineup_home or {}).get("iso"), (lineup_away or {}).get("iso")),
        "lineupHotDiff": _edge((lineup_home or {}).get("recentOps"), (lineup_away or {}).get("recentOps")),
        # Matchup edges: career BvP OPS, season platoon OPS vs the starter's
        # throwing hand, season OPS vs the opposing team — PA-saturated,
        # slot-weighted means over the real starting 9 (0 + lineupKnown = 0
        # when no boxscore lineup / opposing starter is known).
        "bvpOpsDiff": _edge((lineup_home or {}).get("bvpOps"), (lineup_away or {}).get("bvpOps")),
        "platoonOpsDiff": _edge((lineup_home or {}).get("platoonOps"), (lineup_away or {}).get("platoonOps")),
        "vsTeamOpsDiff": _edge((lineup_home or {}).get("vsTeamOps"), (lineup_away or {}).get("vsTeamOps")),
        "spK9Diff": _edge((game.get("homePitcher") or {}).get("k9"), (game.get("awayPitcher") or {}).get("k9")),
        "spWhipDiff": _edge((game.get("homePitcher") or {}).get("whip"), (game.get("awayPitcher") or {}).get("whip"), True),
        "spRecentDiff": _edge((game.get("homePitcher") or {}).get("recentEra"), (game.get("awayPitcher") or {}).get("recentEra"), True),
        "teamK9Diff": _edge(home_k9, away_k9),
        "teamWhipDiff": _edge(home_whip, away_whip, True),
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
    lst = snapshots.get(team_id)
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
