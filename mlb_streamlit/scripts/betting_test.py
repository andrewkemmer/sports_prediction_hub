"""Offline regression tests for the PIT market execution layer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mlb_streamlit.data import attach_lineups_as_of  # noqa: E402
from mlb_streamlit.engine.betting import (  # noqa: E402
    american_implied_probability,
    american_to_decimal,
    build_bet_decision,
    no_vig_two_way,
    stamp_market_odds,
    summarize_bet_decisions,
)


CHECKS = 0


def check(name: str, condition: bool) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(f"FAIL: {name}")
    print(f"  ok: {name}")


def main() -> None:
    print("market execution")
    check("negative American odds convert to decimal", abs(american_to_decimal(-110) - (1 + 100 / 110)) < 1e-12)
    check("positive American odds convert to decimal", abs(american_to_decimal(150) - 2.5) < 1e-12)
    check("raw implied probability", abs(american_implied_probability(-110) - (110 / 210)) < 1e-12)

    market = no_vig_two_way(-110, -110)
    check("no-vig two-way normalizes", market is not None and abs(market["home"] - 0.5) < 1e-12)
    check("overround is retained", market is not None and market["overround"] > 0)

    prediction = {"homeWinProb": 0.60, "awayWinProb": 0.40, "pickTeam": "home", "pickProb": 0.60}
    quote = {
        "homeMoneyline": -110,
        "awayMoneyline": -110,
        "snapshotAt": "2026-08-19T12:00:00Z",
    }
    now_ms = 1787140800000  # 2026-08-19T12:00:00Z
    decision = build_bet_decision(
        prediction,
        quote,
        game_date="2026-08-19T18:00:00Z",
        game_status="Scheduled",
        now_ms=now_ms,
    )
    check("positive-EV quote is available", decision["available"] is True)
    check("positive-EV quote is recommended", decision["recommended"] is True and decision["side"] == "home")
    check("base model probability is unchanged", prediction["homeWinProb"] == 0.60)
    check("quarter Kelly is hard capped", 0 < decision["recommendedStakeFraction"] <= 0.01)
    check("no-vig edge is positive", decision["edge"] > 0.02)
    check("expected value is positive", decision["expectedValue"] > 0.01)

    blocked_prediction = {**prediction, "gateEnabled": True, "gateAccepted": False, "concordance": 0.5}
    blocked = build_bet_decision(
        blocked_prediction,
        quote,
        game_date="2026-08-19T18:00:00Z",
        game_status="Scheduled",
        now_ms=now_ms,
    )
    check("enabled gate remains a wager filter", blocked["available"] is True and blocked["recommended"] is False)
    check("gate filter is recorded", blocked["gateRequired"] is True and blocked["gateAccepted"] is False)
    check("gate abstention explains PASS", "Concordance gate" in blocked["reason"])
    accepted = build_bet_decision(
        {**prediction, "gateEnabled": True, "gateAccepted": True, "concordance": 1.0},
        quote,
        game_date="2026-08-19T18:00:00Z",
        game_status="Scheduled",
        now_ms=now_ms,
    )
    check("accepted gate allows positive EV", accepted["recommended"] is True)
    disabled_gate = build_bet_decision(
        {**prediction, "gateEnabled": False, "gateAccepted": False},
        quote,
        game_date="2026-08-19T18:00:00Z",
        game_status="Scheduled",
        now_ms=now_ms,
    )
    check("held-out gate does not block market execution", disabled_gate["recommended"] is True)

    missing_timestamp = build_bet_decision(
        prediction,
        {"homeMoneyline": -110, "awayMoneyline": -110},
        game_date="2026-08-19T18:00:00Z",
        game_status="Scheduled",
        now_ms=now_ms,
    )
    check("missing timestamp safely passes", missing_timestamp["recommended"] is False)
    check("missing timestamp explains PIT refusal", "timestamp" in missing_timestamp["reason"])

    after_start = build_bet_decision(
        prediction,
        {**quote, "snapshotAt": "2026-08-19T19:00:00Z"},
        game_date="2026-08-19T18:00:00Z",
        game_status="Scheduled",
        now_ms=now_ms,
    )
    check("post-start quote safely passes", after_start["recommended"] is False)
    check("post-start quote explains refusal", "after game start" in after_start["reason"])

    final_game = build_bet_decision(
        prediction,
        quote,
        game_date="2026-08-19T18:00:00Z",
        game_status="Final",
        now_ms=now_ms,
    )
    check("completed game cannot create a wager", final_game["recommended"] is False)

    stale = build_bet_decision(
        prediction,
        quote,
        game_date="2026-08-20T18:00:00Z",
        game_status="Scheduled",
        now_ms=now_ms + 7 * 60 * 60 * 1000,
    )
    check("stale quote safely passes", stale["recommended"] is False)
    check("stale quote explains refusal", "stale" in stale["reason"])

    stamped = stamp_market_odds({"game": {"homeMoneyline": -110}}, "2026-08-19T12:00:00Z")
    check("snapshot stamping is local and serializable", stamped["game"]["snapshotAt"] == "2026-08-19T12:00:00Z")

    lineup = {
        "home": {"battingOrder": [{"id": 10, "name": "Home hitter"}], "bench": []},
        "away": {"battingOrder": [{"id": 11, "name": "Away hitter"}], "bench": []},
    }
    completed_game = {
        "gamePk": 1,
        "date": "2026-08-19",
        "gameDate": "2026-08-19T18:00:00Z",
        "season": "2026",
        "status": "Final",
        "winner": "home",
        "home": {"id": 1},
        "away": {"id": 2},
    }
    scheduled_game = {**completed_game, "gamePk": 2, "status": "Scheduled", "winner": None}
    guarded = attach_lineups_as_of([completed_game], {1: lineup}, {})[0]
    live_lineup = attach_lineups_as_of([scheduled_game], {2: lineup}, {})[0]
    check("completed boxscore lineup is excluded by default", "lineups" not in guarded)
    check("scheduled lineup remains available", "lineups" in live_lineup)

    summary = summarize_bet_decisions([
        {"betDecision": decision},
        {"betDecision": missing_timestamp},
    ])
    check("summary counts only available markets", summary["marketAvailable"] == 1)
    check("summary counts recommended bets", summary["recommended"] == 1)
    check("summary exposes coverage", summary["coverage"] == 0.5)

    print(f"\nAll market execution checks passed ({CHECKS}).")


if __name__ == "__main__":
    main()
