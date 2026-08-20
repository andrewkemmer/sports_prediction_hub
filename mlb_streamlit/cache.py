"""Disk-backed JSON cache for the Streamlit dashboard.

Fully self-contained: everything the dashboard needs lives in this directory
as JSON files, so the app survives restarts and refreshes are incremental
(only new dates are fetched from the MLB Stats API).

Files are written compact (no whitespace) and atomically (tmp + os.replace).
`save_many` writes several independent files concurrently — the refresh
pipeline persists ~12 caches per run, and the writes are pure I/O with no
shared state, so a small thread pool hides their latency.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "cache"

_GAMES = "games.json"
_MODEL_STATE = "model_state.json"
_CALIBRATION = "calibration_rows.json"
_CALIBRATION_WF = "calibration_rows_wf.json"  # {date: {"fp": ..., "rows": [...]}}
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

# Bump to invalidate stale backtest caches (calibration_rows_wf.json).
# refresh.py reads this value as its source of truth.
BACKTEST_CACHE_VERSION = 15  # invalidate rows built before schedule-clock provenance


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
    # Keep the raw writer safe for refresh callers that persist several files
    # through save_many: lineups must never bypass the provenance gate.
    if name == _LINEUPS:
        data = _sanitize_lineup_cache_payload(data)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(name).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, _path(name))


def save_many(items: list[tuple[str, object]], workers: int = 4) -> None:
    """Write several independent cache files concurrently (atomic per file).

    Deterministic by construction: each file is written independently and
    replaced atomically, so a crash leaves every file either old or new —
    never half-written. Falls back to serial for tiny batches.
    """
    if len(items) <= 1:
        for name, data in items:
            save_json(name, data)
        return
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as ex:
        futures = [ex.submit(save_json, name, data) for name, data in items]
        for fut in futures:
            fut.result()  # propagate the first write failure


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
    # Mark states written after the concordance-gate rollout as current even
    # when a test fixture or a hand-created legacy state omitted the optional
    # field. Existing on-disk pre-gate states are still rejected by the refresh
    # fast-path until a full refresh rebuilds them.
    payload = dict(state)
    if "concordanceGate" not in payload:
        from .engine.gating import default_gate_config

        payload["concordanceGate"] = default_gate_config(
            "Gate configuration not present; awaiting prior-only tuning"
        )
    save_json(_MODEL_STATE, payload)


def load_calibration_rows() -> list[dict]:
    return load_json(_CALIBRATION, []) or []


def save_calibration_rows(rows: list[dict]) -> None:
    save_json(_CALIBRATION, rows)


def load_calibration_rows_wf() -> dict:
    """Walk-forward calibration cache: date -> {"fp", "rows"}.

    The payload wraps the per-date map under a ``days`` key plus a top-level
    ``version``. Rows stamped with an older version are treated as absent so
    the calibration tab rebuilds them rather than rendering stale results.
    """
    raw = load_json(_CALIBRATION_WF, {}) or {}
    if isinstance(raw, dict) and raw.get("version") == BACKTEST_CACHE_VERSION:
        return raw.get("days") or {}
    return {}


def save_calibration_rows_wf(days: dict) -> None:
    save_json(_CALIBRATION_WF, days)


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


def _trusted_lineup_cache_value(value) -> bool:
    """Reject legacy/final-boxscore lineup payloads at the cache boundary."""
    if not isinstance(value, dict):
        return False
    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        return False
    if provenance.get("source") != "mlb_statsapi_pregame_boxscore" or provenance.get("isPregame") is not True:
        return False
    try:
        captured = datetime.fromisoformat(str(provenance["capturedAt"]).replace("Z", "+00:00"))
        first = datetime.fromisoformat(str(provenance["firstPitchAt"]).replace("Z", "+00:00"))
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=timezone.utc)
        if first.tzinfo is None:
            first = first.replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return False
    return captured < first and bool(
        (value.get("home") or {}).get("battingOrder")
        and (value.get("away") or {}).get("battingOrder")
    )


def _sanitize_lineup_cache_payload(value) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if _trusted_lineup_cache_value(item)
    }


def load_lineups() -> dict:
    """Return only explicitly pre-first-pitch lineup snapshots.

    JSON object keys are always strings on disk. Legacy entries without an
    auditable capture timestamp (including final boxscore payloads and None
    negative-cache markers) are intentionally discarded instead of being
    treated as historical pre-game data.
    """
    raw = load_json(_LINEUPS, {}) or {}
    if not isinstance(raw, dict):
        return {}
    normalized: dict = {}
    for key, value in raw.items():
        try:
            game_pk = int(key)
        except (TypeError, ValueError):
            continue
        if _trusted_lineup_cache_value(value):
            normalized[game_pk] = value
    return normalized


def save_lineups(lineups: dict) -> None:
    # save_json applies the same conservative filter used by save_many.
    save_json(_LINEUPS, lineups or {})


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
