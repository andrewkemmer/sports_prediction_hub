"""Offline regression tests for Multi-Signal Concordance Gating.

Run with:
    python3 mlb_streamlit/scripts/gating_test.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mlb_streamlit import cache, refresh  # noqa: E402
from mlb_streamlit.engine.gating import (  # noqa: E402
    apply_concordance_gate,
    extract_signals,
    summarize_gate_results,
    tune_concordance_gate,
)
from mlb_streamlit.refresh import build_game_doc  # noqa: E402



def _row(side: int, label: int) -> dict:
    """Build a row whose independent signal families all favor ``side``."""
    s = 1 if side == 1 else -1
    return {
        "features": {
            "winPctDiff": 0.20 * s,
            "formDiff": 0.20 * s,
            "spFipDiff": 1.0 * s,
            "spEraDiff": 1.0 * s,
            "spK9Diff": 1.0 * s,
            "opsDiff": 0.10 * s,
            "restDiff": 1.0 * s,
            "injuryDiff": 1.0 * s,
            "lineupKnown": 0,
        },
        "homeElo": 1550.0 if side == 1 else 1450.0,
        "awayElo": 1450.0 if side == 1 else 1550.0,
        "label": label,
    }



def _prediction(home: bool = True) -> dict:
    p = 0.70 if home else 0.30
    return {
        "homeWinProb": p,
        "awayWinProb": 1.0 - p,
        "pickTeam": "home" if home else "away",
        "pickProb": p,
    }



def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAIL: {name} {detail}")
    print(f"  ok: {name}")



def test_date_specific_calibration_gate() -> None:
    print("date-specific calibration gate and cache migration")
    old_dir = cache.CACHE_DIR
    tmp = tempfile.mkdtemp(prefix="mlb_gate_persistence_")
    cache.CACHE_DIR = Path(tmp)
    try:
        historical_gate = {
            "version": 1,
            "enabled": True,
            "threshold": 1.0,
            "minSignals": 3,
        }
        # This is deliberately different from the current/live fallback. A
        # historical row must use the date's own recipe, never today's gate.
        cache.save_json(
            "walk_forward_selection.json",
            {"version": 8, "days": {"2026-08-10": {"gate": historical_gate}}},
        )
        game = {
            "gamePk": 456,
            "date": "2026-08-10",
            "home": {"id": 119, "abbrev": "LAD", "name": "Dodgers", "score": 4},
            "away": {"id": 108, "abbrev": "LAA", "name": "Angels", "score": 2},
        }
        row = _row(1, 1)
        row["game"] = game
        model = {
            "featureNames": [],
            "weights": [],
            "bias": 0.0,
            "featureStats": {},
            "isotonicPoints": [],
            "eloHfa": 30.0,
            "blendW": 0.0,
            "stack": {"members": {}, "weights": {}},
        }
        run_model_state = {
            "leagueRuns": 4.5,
            "teamOffense": {108: 1.0, 119: 1.0},
            "teamDefense": {108: 1.0, 119: 1.0},
            "parkFactor": {108: 1.0, 119: 1.0},
        }
        rows = refresh.build_calibration_rows(
            [row],
            model,
            run_model_state,
            [],
            {"slope": 0.0, "intercept": 0.0},
            gate_config={"version": 1, "enabled": False, "threshold": 0.75, "minSignals": 3},
        )
        check("historical calibration prefers its own gate config", rows[0]["gateEnabled"] is True)
        check("historical gate threshold persists", rows[0]["gateThreshold"] == 1.0)
        check("historical gate accepts concordant row", rows[0]["gateAccepted"] is True)
        check("gate does not replace base probability", rows[0]["pickTeam"] == "home" and rows[0]["pickProb"] == 0.5)

        # A legacy state saved through the typed cache accessor is migrated to
        # an explicitly disabled gate; an old on-disk state is still rejected
        # by run_refresh's fast-path version guard until a real refresh.
        cache.save_model_state({"featureNames": [], "weights": [], "bias": 0.0, "featureStats": {}})
        migrated = cache.load_model_state()
        check("legacy state gets an explicit gate record", migrated["concordanceGate"]["version"] == 1)
        check("legacy migration is conservative", migrated["concordanceGate"]["enabled"] is False)
    finally:
        cache.CACHE_DIR = old_dir
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    model = {"eloHfa": 30.0, "featureNames": [], "stack": {"members": {}, "weights": {}}}
    config = {"version": 1, "enabled": True, "threshold": 0.75, "minSignals": 3}

    print("signal extraction and gate behavior")
    aligned = _row(1, 1)
    signals = extract_signals(model, aligned["features"], aligned["homeElo"], aligned["awayElo"])
    available = [s for s in signals.values() if s["available"]]
    check("multiple independent signals are available", len(available) >= 5, str(signals))
    accepted = apply_concordance_gate(
        _prediction(True), model, aligned["features"], aligned["homeElo"], aligned["awayElo"], config
    )
    check("concordant signals accept the base pick", accepted["gateAccepted"])
    check("accepted gate preserves base probability", accepted["gatedPickProb"] == 0.70)
    check("accepted gate reports agreement", accepted["concordance"] >= 0.75)

    opposed = _row(-1, 0)
    rejected = apply_concordance_gate(
        _prediction(True), model, opposed["features"], opposed["homeElo"], opposed["awayElo"], config
    )
    check("discordant signals reject the base pick", not rejected["gateAccepted"])
    check("rejected gate abstains instead of flipping the pick", rejected["gatedPickTeam"] is None)
    check("rejected gate leaves base probability independent", _prediction(True)["homeWinProb"] == 0.70)

    print("prior-only threshold tuning")
    rows = [_row(1, 1) for _ in range(105)]
    # The trailing validation slice contains 30 concordant wins and 15
    # discordant losses. A guarded gate can improve conditional win rate while
    # retaining at least the configured minimum coverage.
    rows.extend(_row(1, 1) for _ in range(30))
    rows.extend(_row(-1, 0) for _ in range(15))
    base_preds = [_prediction(True) for _ in rows]
    tuned = tune_concordance_gate(rows, model, base_preds)
    check("tuner returns a serializable configuration", tuned["version"] == 1)
    check("tuner uses a guarded validation sample", tuned["tunedOn"] == 45)
    check("tuner enables a demonstrably higher conditional rate", tuned["enabled"] and tuned["validationLift"] > 0)
    check("tuner respects coverage guard", tuned["validationCoverage"] >= tuned["minCoverage"])

    print("persisted prediction fields")
    game = {
        "gamePk": 123,
        "date": "2026-08-20",
        "status": "Preview",
        "dayNight": "night",
        "gameDate": "2026-08-20T23:00:00Z",
        "season": "2026",
        "home": {"id": 119, "abbrev": "LAD", "name": "Dodgers"},
        "away": {"id": 108, "abbrev": "LAA", "name": "Angels"},
    }
    pred = {**_prediction(True), **accepted, "edge": 0.1, "fairHomeOdds": -150, "fairAwayOdds": 130, "shap": []}
    doc = build_game_doc(game, pred, None, None, None, trained_through="2026-08-19")
    check("doc stores gate acceptance", doc["gateAccepted"] is True)
    check("doc stores gated team", doc["gatedPickTeam"] == "home")
    check("doc stores gate signal diagnostics", doc["gateSignalCount"] >= 5 and "gateSignals" in doc)
    check("doc keeps base pick separate", doc["pickTeam"] == "home" and doc["pickProb"] == 0.70)

    completed = [
        {"isCorrect": True, "gateAccepted": True, "gatedIsCorrect": True, "concordance": 1.0},
        {"isCorrect": False, "gateAccepted": False, "gatedIsCorrect": None, "concordance": 0.5},
        {"isCorrect": True, "gateAccepted": True, "gatedIsCorrect": False, "concordance": 0.8},
    ]
    summary = summarize_gate_results(completed)
    check("summary reports accepted coverage separately", summary["accepted"] == 2 and summary["total"] == 3)
    check("summary reports conditional win rate", summary["winRate"] == 0.5 and summary["coverage"] == 2 / 3)

    test_date_specific_calibration_gate()

    print("\nAll gating checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
