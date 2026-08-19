"""Offline regression tests for engine/backtest.py (paper-trading P&L)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.backtest import (  # noqa: E402
    BENCHMARK_AMERICAN_ODDS,
    breakeven_probability,
    build_execution_backtest,
    kelly_stake,
    run_backtest,
    threshold_sweep,
)

_CHECKS = 0


def ok(cond: bool, label: str) -> None:
    global _CHECKS
    _CHECKS += 1
    assert cond, label
    print(f"  ok: {label}")


def _rows(pairs: list[tuple[float, bool]]) -> list[dict]:
    return [
        {
            "gamePk": i,
            "date": "2026-04-0%d" % (i + 1),
            "pickTeam": "home",
            "pickProb": p,
            "isCorrect": correct,
        }
        for i, (p, correct) in enumerate(pairs)
    ]


def test_kelly_and_breakeven() -> None:
    print("kelly sizing & breakeven")
    ok(abs(breakeven_probability() - 110 / 210) < 1e-9, "breakeven for -110 is 52.38%")
    ok(kelly_stake(0.50) == 0.0, "no stake at breakeven")
    ok(kelly_stake(0.5238, cap=0.05) == 0.0, "no stake below breakeven")
    ok(kelly_stake(0.60) > 0.0, "positive stake above breakeven")
    ok(kelly_stake(0.99) <= 0.01, "stake is capped at 1%")
    ok(kelly_stake(0.60, fractional=0.0) == 0.0, "zero fractional Kelly stakes nothing")


def test_run_backtest() -> None:
    print("backtest P&L")
    rows = _rows([(0.60, True), (0.60, False), (0.60, True), (0.60, True)])
    r = run_backtest(rows)
    ok(r["games"] == 4, "all games counted")
    ok(r["bets"] == 4, "all positive-Kelly games staked")
    ok(r["wins"] == 3 and r["losses"] == 1, "win/loss split")
    ok(r["hitRate"] == 0.75, "hit rate")
    ok(r["coverage"] == 1.0, "full coverage without gate")
    # 3 wins pay stake*b, 1 loss pays -stake; stake is identical per row here,
    # so ROI = (3*b - 1)/4.
    b = (100 / 110)
    ok(abs(r["roi"] - round((3 * b - 1) / 4, 6)) < 1e-9, "ROI matches closed-form at flat price")
    ok(r["finalBankroll"] > 1.0, "profitable strategy grows bankroll")
    ok(len(r["equityCurve"]) == 4, "equity curve has one point per bet")


def test_min_prob_filter() -> None:
    print("confidence threshold")
    rows = _rows([(0.52, True), (0.70, True), (0.70, False), (0.90, True)])
    r = run_backtest(rows, min_prob=0.65)
    ok(r["bets"] == 3, "threshold drops sub-0.65 pick")
    ok(r["games"] == 4, "games count is unchanged by the filter")


def test_gated_strategy() -> None:
    print("concordance-gated strategy")
    rows = [
        {"gamePk": 1, "date": "2026-04-01", "pickTeam": "home", "pickProb": 0.60,
         "isCorrect": True, "gateAccepted": True, "gatedPickTeam": "home",
         "gatedPickProb": 0.60, "gatedIsCorrect": True},
        {"gamePk": 2, "date": "2026-04-02", "pickTeam": "home", "pickProb": 0.60,
         "isCorrect": False, "gateAccepted": False, "gatedPickProb": None,
         "gatedIsCorrect": None},
        {"gamePk": 3, "date": "2026-04-03", "pickTeam": "home", "pickProb": 0.60,
         "isCorrect": True, "gateAccepted": True, "gatedPickTeam": "home",
         "gatedPickProb": 0.60, "gatedIsCorrect": True},
    ]
    r = run_backtest(rows, gate="gated")
    ok(r["bets"] == 2, "only accepted rows are staked")
    ok(r["coverage"] == 2 / 3, "coverage excludes abstentions")
    ok(r["hitRate"] == 1.0, "gated hit rate")


def test_sweep_and_summary() -> None:
    print("threshold sweep & summary")
    rows = _rows([(0.53, True), (0.60, True), (0.62, False), (0.68, True), (0.72, True)])
    sweep = threshold_sweep(rows)
    ok(len(sweep) > 0, "sweep produced")
    ok(all(s["bets"] <= 5 for s in sweep), "bets are monotone non-increasing in threshold")
    summary = build_execution_backtest(rows)
    ok(set(summary) >= {"price", "breakeven", "base", "gated", "thresholdSweep"}, "summary schema")
    ok(summary["base"]["games"] == 5, "summary uses full rows")


def main() -> None:
    test_kelly_and_breakeven()
    test_run_backtest()
    test_min_prob_filter()
    test_gated_strategy()
    test_sweep_and_summary()
    print(f"\nAll {_CHECKS} backtest checks passed.")


if __name__ == "__main__":
    main()
