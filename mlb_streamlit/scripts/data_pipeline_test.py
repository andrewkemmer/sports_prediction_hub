"""Offline regression tests for the upstream feature-data contract.

Run from the repository root with:
    python3 mlb_streamlit/scripts/data_pipeline_test.py
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mlb_streamlit import cache  # noqa: E402
from mlb_streamlit.data import attach_lineups_as_of, parse_boxscore_lineup  # noqa: E402
from mlb_streamlit.engine.ensemble import boosted_stumps_params, boosted_stumps_predict  # noqa: E402
from mlb_streamlit.engine.features import (  # noqa: E402
    FEATURE_KEYS,
    build_features,
    new_state,
)
from mlb_streamlit.engine.model import compute_feature_drift  # noqa: E402
from mlb_streamlit.engine.nn import mlp_params, mlp_predict  # noqa: E402


_CHECKS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _CHECKS
    _CHECKS += 1
    if not condition:
        raise AssertionError(f"FAIL: {name} {detail}")
    print(f"  ok: {name}")


def make_boxscore() -> dict:
    def side(prefix: int) -> dict:
        batter_ids = [prefix + i for i in range(1, 10)]
        players = {
            f"ID{pid}": {
                "person": {"id": pid, "fullName": f"Player {pid}"},
                "position": {"abbreviation": "CF"},
            }
            for pid in batter_ids
        }
        players[f"ID{prefix + 99}"] = {
            "person": {"id": prefix + 99, "fullName": "Bench bat"},
            "position": {"abbreviation": "1B"},
        }
        players[f"ID{prefix + 199}"] = {
            "person": {"id": prefix + 199, "fullName": "Starter"},
            "position": {"abbreviation": "P"},
        }
        return {
            # This is the current Stats API shape; the parser used to only
            # inspect the legacy team-level battingOrder field.
            "batters": batter_ids,
            "pitchers": [prefix + 199],
            "players": players,
        }

    return {"teams": {"home": side(1000), "away": side(2000)}}


def make_game() -> dict:
    return {
        "gamePk": 2026000001,
        "date": "2026-04-20",
        "gameDate": "2026-04-20T18:00:00Z",
        "season": "2026",
        "winner": None,
        "home": {"id": 119, "name": "Home", "abbrev": "H"},
        "away": {"id": 108, "name": "Away", "abbrev": "A"},
        "homePitcher": {"id": 9001, "pitchHand": "R", "restDays": 3, "workload": 20.0},
        "awayPitcher": {"id": 9002, "pitchHand": "L", "restDays": 5, "workload": 8.0},
        "weather": {"tempF": "102", "windMph": "24"},
    }


def make_logs(prefix: int) -> dict:
    logs = {}
    for player_id in range(prefix + 1, prefix + 10):
        entries = []
        for i in range(18):
            d = date(2026, 4, 2) + timedelta(days=i)
            # The last six games are intentionally stronger so the recent-10
            # OPS differs from the season-to-date OPS.
            hits = 1 if i < 12 else 3
            total_bases = 2 if i < 12 else 7
            entries.append({
                "d": d.isoformat(),
                "ab": 4.0,
                "h": float(hits),
                "bb": 1.0,
                "ibb": 0.0,
                "hbp": 0.0,
                "sf": 0.0,
                "tb": float(total_bases),
                "2b": 1.0,
                "3b": 0.0,
                "hr": 0.0,
                "teamId": 119 if prefix == 1000 else 108,
                "opponentId": 108 if prefix == 1000 else 119,
            })
        logs[f"{player_id}|2026"] = entries
    return logs


def test_pipeline() -> None:
    print("lineup parser and PIT feature population")
    lineup = parse_boxscore_lineup(make_boxscore())
    check("boxscore parser returns both sides", lineup is not None)
    check("canonical batters array yields nine starters",
          len(lineup["home"]["battingOrder"]) == 9
          and len(lineup["away"]["battingOrder"]) == 9)
    check("bench remains separate",
          len(lineup["home"]["bench"]) == 1 and lineup["home"]["bench"][0]["id"] == 1099)

    game = make_game()
    enriched = attach_lineups_as_of(
        [game],
        {game["gamePk"]: lineup},
        {**make_logs(1000), **make_logs(2000)},
        pregame_only=False,
    )[0]
    home_stats = enriched["lineupStats"]["home"]
    away_stats = enriched["lineupStats"]["away"]
    check("historical lineup is marked known", home_stats["known"] and away_stats["known"])
    for key in ("ops", "woba", "iso", "recentOps", "momentum"):
        check(f"lineup {key} populated", home_stats[key] != 0 and away_stats[key] != 0,
              f"{home_stats[key]} / {away_stats[key]}")
    check("lineup fatigue populated", home_stats["games7"] > 0 and away_stats["games7"] > 0)

    state = new_state()
    features = build_features(enriched, state)
    check("feature vector contains the canonical keys", all(k in features for k in FEATURE_KEYS))
    check("weather temperature is dimensionless and bounded", -3 <= features["tempDev"] <= 3,
          str(features["tempDev"]))
    check("weather wind is bounded", 0 <= features["windMph"] <= 3,
          str(features["windMph"]))
    check("starter fatigue interaction is bounded", -1 <= features["spRestWorkloadDiff"] <= 1,
          str(features["spRestWorkloadDiff"]))
    check("lineup fatigue interaction is bounded", -1 <= features["lineupFatigueDiff"] <= 1,
          str(features["lineupFatigueDiff"]))

    drift_rows = [
        {"features": {"homeField": 1.0, "tempDev": float(i) / 10}}
        for i in range(10)
    ]
    drift = compute_feature_drift(drift_rows, ["homeField", "tempDev"])
    check("structural homeField is excluded from PSI", all(d["feature"] != "homeField" for d in drift))
    check("non-structural drift remains visible", [d["feature"] for d in drift] == ["tempDev"])

    print("cache key and common scaling contract")
    tmp = tempfile.mkdtemp(prefix="mlb_pipeline_cache_")
    old_cache_dir = cache.CACHE_DIR
    cache.CACHE_DIR = Path(tmp)
    try:
        cache.save_lineups({"123": lineup, 456: lineup})
        loaded = cache.load_lineups()
        check("lineup cache normalizes JSON string keys", 123 in loaded and 456 in loaded)
        check("lineup cache retains actual payload", len(loaded[123]["home"]["battingOrder"]) == 9)
    finally:
        cache.CACHE_DIR = old_cache_dir
        shutil.rmtree(tmp, ignore_errors=True)

    rows = []
    for i in range(50):
        x = -2.0 + 4.0 * i / 49
        rows.append({"features": {"x": x, "y": math.sin(x)}, "label": 1 if x > 0 else 0})
    stump = boosted_stumps_params(rows, ["x", "y"], n_trees=2, max_features=2, min_leaf=3)
    check("boosted stumps persist train-only scaling stats", set(stump["stats"]) == {"x", "y"})
    check("boosted stumps remain finite on an extreme live vector",
          math.isfinite(boosted_stumps_predict(stump, {"x": 99999, "y": -99999})))
    mlp = mlp_params(rows, ["x", "y"], epochs=1, patience=1, batch=16)
    check("MLP persists the same train-only scaling stats", set(mlp.get("stats", {})) == {"x", "y"})
    check("MLP remains finite on an extreme live vector",
          math.isfinite(mlp_predict(mlp, {"x": 99999, "y": -99999})))


if __name__ == "__main__":
    test_pipeline()
    print(f"\nAll {_CHECKS} data-pipeline checks passed.")
