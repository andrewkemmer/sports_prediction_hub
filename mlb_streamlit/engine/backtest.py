"""Point-in-time paper-trading backtest for the execution layer.

The market-mapping module (`engine/betting.py`) refuses to reconstruct
historical ROI from today's live odds feed, which is correct — a bet can only
be judged against the price it was actually offered.  This module closes that
loop for the *model* side by replaying the walk-forward calibration record
against an explicit, conservative price model.

Default benchmark: a flat -110 two-way market.  That is the standard
"can the model beat the vig" test: no historical closing line is required, and
a model that cannot clear -110 has no business being traded against sharper
closing lines.  The walk-forward rows already carry point-in-time (isotonic-
calibrated) probabilities plus the per-date concordance-gate decision, so the
backtest uses exactly the numbers the dashboard would have shown before each
game — no lookahead.

Sizing is quarter-Kelly, hard-capped at 1% of bankroll per position (matching
the live execution layer).  Kelly here is a *diagnostic* — it sizes the bet
proportionally to the model's edge over -110 — not a claim that the model's
uncertainty is fully known.
"""

from __future__ import annotations

import math

from .betting import american_to_decimal
from .metrics import clamp

BACKTEST_VERSION = 1
BENCHMARK_AMERICAN_ODDS = -110
FRACTIONAL_KELLY = 0.25
MAX_STAKE_FRACTION = 0.01

# A sweep of confidence thresholds.  This is the quantified version of
# "Multi-Signal Concordance Gating": each row shows how much win rate is
# gained by raising the bar, and what that costs in coverage and ROI.
DEFAULT_THRESHOLDS = (0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70)


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _wilson_lower(wins: int, total: int) -> float:
    if total <= 0:
        return 0.0
    z = 1.96
    p = wins / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (center - spread) / denominator


def breakeven_probability(american: float = BENCHMARK_AMERICAN_ODDS) -> float:
    """Implied breakeven win rate for a flat American price (0.5238 at -110)."""
    decimal = american_to_decimal(american)
    return 1.0 / decimal if decimal and decimal > 1.0 else 0.0


def kelly_stake(
    probability: float,
    american: float = BENCHMARK_AMERICAN_ODDS,
    *,
    fractional: float = FRACTIONAL_KELLY,
    cap: float = MAX_STAKE_FRACTION,
) -> float:
    """Quarter-Kelly stake fraction for a single position at a flat price."""
    decimal = american_to_decimal(american)
    p = clamp(float(probability), 0.001, 0.999)
    if not decimal or decimal <= 1.0:
        return 0.0
    b = decimal - 1.0
    q = 1.0 - p
    full_kelly = (p * b - q) / b
    if full_kelly <= 0:
        return 0.0
    return min(clamp(float(cap), 0.0, 0.05), float(fractional) * full_kelly)


def _bet_row(row: dict, gate: str) -> dict | None:
    """Resolve one walk-forward row into an executable (side, prob, correct)."""
    if gate == "gated":
        if row.get("gateAccepted") is not True:
            return None
        prob = _number(row.get("gatedPickProb", row.get("pickProb")))
        correct = row.get("gatedIsCorrect")
    else:
        prob = _number(row.get("pickProb"))
        correct = row.get("isCorrect")
    if prob is None or not isinstance(correct, bool):
        return None
    return {
        "date": row.get("date") or "",
        "pickTeam": row.get("gatedPickTeam") if gate == "gated" else row.get("pickTeam"),
        "prob": prob,
        "correct": correct,
    }


def run_backtest(
    rows: list[dict],
    *,
    american: float = BENCHMARK_AMERICAN_ODDS,
    fractional: float = FRACTIONAL_KELLY,
    cap: float = MAX_STAKE_FRACTION,
    min_prob: float = 0.0,
    gate: str = "off",
    start_bankroll: float = 1.0,
) -> dict:
    """Replay a betting strategy over chronological calibration rows.

    ``gate`` is ``"off"`` (base model) or ``"gated"`` (only rows the per-date
    concordance gate accepted).  ``min_prob`` additionally filters the base
    strategy to picks at or above a confidence floor.
    """
    ordered = sorted(
        rows,
        key=lambda r: (r.get("date") or "", str(r.get("gamePk") or "")),
    )
    decimal = american_to_decimal(american)
    b = (decimal - 1.0) if decimal else 0.0
    breakeven = 1.0 / decimal if decimal and decimal > 1.0 else 0.0

    equity = start_bankroll
    peak = equity
    max_drawdown = 0.0
    wins = 0
    losses = 0
    gross_win = 0.0
    gross_loss = 0.0
    total_staked = 0.0
    per_bet_pnl: list[float] = []
    equity_curve: list[dict] = []

    for row in ordered:
        bet = _bet_row(row, gate)
        if bet is None:
            continue
        if gate == "off" and bet["prob"] < float(min_prob):
            continue
        stake = kelly_stake(bet["prob"], american, fractional=fractional, cap=cap)
        if stake <= 0:
            continue
        pnl = stake * b if bet["correct"] else -stake
        equity += pnl
        max_drawdown = max(max_drawdown, (peak - equity) / peak if peak > 0 else 0.0)
        peak = max(peak, equity)
        if bet["correct"]:
            wins += 1
            gross_win += stake * b
        else:
            losses += 1
            gross_loss += stake
        total_staked += stake
        per_bet_pnl.append(pnl)
        equity_curve.append({"date": bet["date"], "team": bet["pickTeam"], "bankroll": round(equity, 6)})

    bets = wins + losses
    hit_rate = wins / bets if bets else 0.0
    roi = (gross_win - gross_loss) / total_staked if total_staked else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    mean_pnl = sum(per_bet_pnl) / len(per_bet_pnl) if per_bet_pnl else 0.0
    var_pnl = (
        sum((x - mean_pnl) ** 2 for x in per_bet_pnl) / len(per_bet_pnl)
        if per_bet_pnl else 0.0
    )
    sharpe = mean_pnl / math.sqrt(var_pnl) if var_pnl > 1e-12 else 0.0

    return {
        "version": BACKTEST_VERSION,
        "american": float(american),
        "breakeven": round(breakeven, 6),
        "fractionalKelly": float(fractional),
        "maxStakeFraction": float(cap),
        "gate": gate,
        "minProb": round(float(min_prob), 6),
        "games": len(ordered),
        "bets": bets,
        "coverage": bets / len(ordered) if ordered else 0.0,
        "wins": wins,
        "losses": losses,
        "hitRate": round(hit_rate, 6),
        "wilsonLower": round(_wilson_lower(wins, bets), 6),
        "roi": round(roi, 6),
        "profitFactor": round(profit_factor, 4) if math.isfinite(profit_factor) else None,
        "totalStaked": round(total_staked, 6),
        "netProfit": round(gross_win - gross_loss, 6),
        "finalBankroll": round(equity, 6),
        "bankrollGrowth": round(equity - start_bankroll, 6),
        "maxDrawdown": round(max_drawdown, 6),
        "perBetSharpe": round(sharpe, 4),
        "equityCurve": equity_curve,
    }


def threshold_sweep(
    rows: list[dict],
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    *,
    american: float = BENCHMARK_AMERICAN_ODDS,
) -> list[dict]:
    """Base-model ROI / win-rate tradeoff across confidence thresholds."""
    out: list[dict] = []
    for t in thresholds:
        r = run_backtest(rows, american=american, min_prob=t, gate="off")
        out.append({
            "minProb": t,
            "bets": r["bets"],
            "coverage": r["coverage"],
            "hitRate": r["hitRate"],
            "roi": r["roi"],
            "maxDrawdown": r["maxDrawdown"],
            "finalBankroll": r["finalBankroll"],
        })
    return out


def build_execution_backtest(rows: list[dict]) -> dict:
    """Full paper-trading summary: base vs gated strategies plus the sweep."""
    return {
        "version": BACKTEST_VERSION,
        "price": BENCHMARK_AMERICAN_ODDS,
        "breakeven": breakeven_probability(),
        "base": run_backtest(rows, gate="off"),
        "gated": run_backtest(rows, gate="gated"),
        "thresholdSweep": threshold_sweep(rows),
    }
