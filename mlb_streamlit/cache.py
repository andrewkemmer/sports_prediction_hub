"""Disk-backed JSON cache for the Streamlit dashboard.

Fully self-contained: everything the dashboard needs lives in this directory
as JSON files, so the app survives restarts and refreshes are incremental
(only new dates are fetched from the MLB Stats API).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "cache"

_GAMES = "games.json"
_MODEL_STATE = "model_state.json"
_CALIBRATION = "calibration_rows.json"
_PITCHER_STATS = "pitcher_stats.json"
_TEAM_STATS = "team_stats.json"
_PLAYER_OPS = "player_ops.json"
_PITCHER_LOGS = "pitcher_game_logs.json"
_TEAM_LOGS = "team_game_logs.json"
_BATTER_LOGS = "batter_game_logs.json"
_BVP_LOGS = "bvp_logs.json"
_PLATOON_LOGS = "platoon_logs.json"
_VS_TEAM_LOGS = "vs_team_logs.json"
_PITCHER_HANDS = "pitcher_hands.json"
_LINEUPS = "lineups.json"
_INJURIES = "injury_snapshots.json"
_DOCS_BY_DATE = "docs_by_date.json"
_MARKET_ODDS = "market_odds.json"


def _path(name: str) -> Path:
    return CACHE_DIR / name


def load_json(name: str, default=None):
    p = _path(name)
    if not p.exists():
        return default
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 — corrupt cache is treated as absent
        return default


def save_json(name: str, data) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(name).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, _path(name))


def has(name: str) -> bool:
    return _path(name).exists()


# ---------------------------------------------------------------------------
# Typed accessors
# ---------------------------------------------------------------------------

def load_games() -> list[dict]:
    return load_json(_GAMES, []) or []


def save_games(games: list[dict]) -> None:
    save_json(_GAMES, games)


def load_model_state() -> dict | None:
    return load_json(_MODEL_STATE, None)


def save_model_state(state: dict) -> None:
    save_json(_MODEL_STATE, state)


def load_calibration_rows() -> list[dict]:
    return load_json(_CALIBRATION, []) or []


def save_calibration_rows(rows: list[dict]) -> None:
    save_json(_CALIBRATION, rows)


def load_pitcher_stats() -> dict:
    return load_json(_PITCHER_STATS, {}) or {}


def save_pitcher_stats(stats: dict) -> None:
    save_json(_PITCHER_STATS, stats)


def load_team_stats() -> dict:
    return load_json(_TEAM_STATS, {}) or {}


def save_team_stats(stats: dict) -> None:
    save_json(_TEAM_STATS, stats)


def load_player_ops() -> dict:
    return load_json(_PLAYER_OPS, {}) or {}


def save_player_ops(stats: dict) -> None:
    save_json(_PLAYER_OPS, stats)


def load_pitcher_logs() -> dict:
    """{id|season} -> sorted per-game pitching entries (as-of-date accumulation)."""
    return load_json(_PITCHER_LOGS, {}) or {}


def save_pitcher_logs(stats: dict) -> None:
    save_json(_PITCHER_LOGS, stats)


def load_team_logs() -> dict:
    """{id|season} -> {"hitting": [...], "pitching": [...], "fielding": [...]}."""
    return load_json(_TEAM_LOGS, {}) or {}


def save_team_logs(stats: dict) -> None:
    save_json(_TEAM_LOGS, stats)


def load_batter_logs() -> dict:
    """{id|season} -> sorted per-game hitting entries (as-of-date accumulation)."""
    return load_json(_BATTER_LOGS, {}) or {}


def save_batter_logs(stats: dict) -> None:
    save_json(_BATTER_LOGS, stats)


def load_bvp_logs() -> dict:
    """{batterId|pitcherId} -> career BvP {"pa", "ops"}."""
    return load_json(_BVP_LOGS, {}) or {}


def save_bvp_logs(logs: dict) -> None:
    save_json(_BVP_LOGS, logs)


def load_platoon_logs() -> dict:
    """{batterId|season} -> {"vsLeft": {"pa", "ops"}, "vsRight": {"pa", "ops"}}."""
    return load_json(_PLATOON_LOGS, {}) or {}


def save_platoon_logs(logs: dict) -> None:
    save_json(_PLATOON_LOGS, logs)


def load_vs_team_logs() -> dict:
    """{batterId|teamId|season} -> season {"pa", "ops"} vs that team."""
    return load_json(_VS_TEAM_LOGS, {}) or {}


def save_vs_team_logs(logs: dict) -> None:
    save_json(_VS_TEAM_LOGS, logs)


def load_pitcher_hands() -> dict:
    """{pitcherId} -> "L" | "R" (throwing hand for platoon features)."""
    return load_json(_PITCHER_HANDS, {}) or {}


def save_pitcher_hands(hands: dict) -> None:
    save_json(_PITCHER_HANDS, hands)


def load_lineups() -> dict:
    """{gamePk} -> parsed lineup, or None for completed games with no posted lineups."""
    return load_json(_LINEUPS, {}) or {}


def save_lineups(lineups: dict) -> None:
    save_json(_LINEUPS, lineups)


def load_injury_snapshots() -> dict:
    return load_json(_INJURIES, {}) or {}


def save_injury_snapshots(snapshots: dict) -> None:
    save_json(_INJURIES, snapshots)


def load_docs_by_date() -> dict:
    return load_json(_DOCS_BY_DATE, {}) or {}


def save_docs_by_date(docs: dict) -> None:
    save_json(_DOCS_BY_DATE, docs)


def load_market_odds() -> dict:
    return load_json(_MARKET_ODDS, {}) or {}


def save_market_odds(payload: dict) -> None:
    save_json(_MARKET_ODDS, payload)


def cache_size_bytes() -> int:
    total = 0
    if CACHE_DIR.exists():
        for p in CACHE_DIR.iterdir():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total
