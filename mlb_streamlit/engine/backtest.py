"""Point-in-time, market-agnostic execution backtest.

The calibration pipeline emits predictions, not historical prices.  This module
therefore accepts an explicit ``closing_odds`` field per market row and uses a
transparent 1.9091 decimal (-110) fallback only when an older historical row
has no close.  It never substitutes today's live quote for a historical close.

Every settled execution row is normalized to one of MONEYLINE, TOTAL, or
RUN_LINE and is paid out with the same universal rule:

    win  -> stake * (closing_odds - 1.0)
    loss -> -stake

Kelly sizing uses that row's decimal price.  The legacy ``american`` argument
remains available for old moneyline fixtures and tests.
"""

from __future__ import annotations

import math

from .betting import american_to_decimal
from .markets import (
    DEFAULT_DECIMAL_CLOSING_ODDS,
    MARKET_ARCHITECTURE_METADATA,
    MARKET_TYPES,
    build_market_payload,
    expand_market_rows,
    normalize_decimal_odds,
    normalize_market_row,
    normalize_market_type,
)
from .metrics import clamp

BACKTEST_VERSION = 2
BENCHMARK_AMERICAN_ODDS = -110
FRACTIONAL_KELLY = 0.25
MAX_STAKE_FRACTION = 0.01
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


def breakeven_probability(american: float = BENCHMARK_AMERICAN_ODDS, *, closing_odds: float | None = None) -> float:
    """Implied breakeven win rate for either a legacy American or decimal price."""
    decimal = closing_odds if closing_odds is not None else american_to_decimal(american)
    return 1.0 / decimal if decimal and decimal > 1.0 else 0.0


def _decimal_for_stake(american: float, closing_odds: float | None) -> float:
    if closing_odds is not None:
        return normalize_decimal_odds(closing_odds)[0]
    return normalize_decimal_odds(american_to_decimal(american))[0]


def kelly_stake(
    probability: float,
    american: float = BENCHMARK_AMERICAN_ODDS,
    *,
    closing_odds: float | None = None,
    fractional: float = FRACTIONAL_KELLY,
    cap: float = MAX_STAKE_FRACTION,
) -> float:
    """Quarter-Kelly stake fraction using the row's dynamic decimal price."""
    decimal = _decimal_for_stake(american, closing_odds)
    p = clamp(float(probability), 0.001, 0.999)
    if decimal <= 1.0:
        return 0.0
    b = decimal - 1.0
    q = 1.0 - p
    full_kelly = (p * b - q) / b
    if full_kelly <= 0:
        return 0.0
    return min(clamp(float(cap), 0.0, 0.05), max(0.0, float(fractional)) * full_kelly)


def _settled(value) -> bool:
    return value in (0, 1, False, True)


def _bet_row(row: dict, gate: str) -> dict | None:
    """Resolve a canonical row to a probability, PIT price, and binary result."""
    if gate == "gated":
        accepted = row.get("market_gate_accepted")
        if accepted is None:
            accepted = row.get("marketGateAccepted")
        if accepted is None:
            accepted = row.get("gateAccepted")
        if accepted is not True:
            return None
        probability = _number(row.get("gated_model_probability"))
        is_win = row.get("gated_is_win")
        if probability is None:
            probability = _number(
                row.get(
                    "gatedPickProb",
                    row.get("pickProb", row.get("model_probability", row.get("modelProb"))),
                )
            )
        if is_win is None:
            is_win = row.get("gatedIsCorrect", row.get("is_win", row.get("isWin")))
    else:
        probability = _number(row.get("model_probability", row.get("modelProb", row.get("pickProb"))))
        is_win = row.get("is_win", row.get("isWin", row.get("isCorrect")))
    if probability is None or not _settled(is_win):
        return None
    close, _ = normalize_decimal_odds(row.get("closing_odds", row.get("closingOdds")))
    return {
        "date": row.get("date") or "",
        "gamePk": row.get("gamePk"),
        "marketType": row.get("market_type", row.get("marketType", "MONEYLINE")),
        "marketLine": row.get("market_line", row.get("marketLine", 0.0)),
        "modelSide": row.get("model_side", row.get("modelSide")),
        "pickTeam": row.get("pickTeam"),
        "prob": probability,
        "isWin": 1 if bool(is_win) else 0,
        "closingOdds": close,
    }


def _rows_for_backtest(
    rows: list[dict],
    market_type: str | None = None,
    *,
    legacy_american: float = BENCHMARK_AMERICAN_ODDS,
) -> list[dict]:
    # Direct callers of the legacy moneyline API supplied one game row with
    # only ``pickProb``/``isCorrect``. Keep that contract moneyline-only; the
    # market migration expands legacy rows to all three vectors only at the
    # explicit serialization boundary (``build_execution_backtest``).
    has_explicit_market = any(
        isinstance(row, dict)
        and normalize_market_type(row.get("market_type") or row.get("marketType")) is not None
        for row in rows or []
    )
    if market_type is None and not has_explicit_market:
        canonical = [
            normalize_market_row(
                {
                    **row,
                    "market_type": "MONEYLINE",
                    # Compatibility fixtures predate the market-centric
                    # schema. Preserve their exact American -110 conversion;
                    # migrated historical rows still use the explicit 1.9091
                    # fallback in normalize_market_row.
                    "closing_odds": american_to_decimal(legacy_american),
                },
                "MONEYLINE",
            )
            for row in rows or []
            if isinstance(row, dict)
        ]
    else:
        canonical = expand_market_rows(rows)
    if market_type:
        target = normalize_market_type(market_type) or market_type.upper()
        canonical = [r for r in canonical if r.get("market_type") == target]
    return canonical


def _summary_rows(rows: list[dict]) -> tuple[list[dict], int]:
    ordered = sorted(rows, key=lambda r: (r.get("date") or "", str(r.get("gamePk") or ""), r.get("market_type", "")))
    return ordered, len(ordered)


def run_backtest(
    rows: list[dict],
    *,
    american: float = BENCHMARK_AMERICAN_ODDS,
    fractional: float = FRACTIONAL_KELLY,
    cap: float = MAX_STAKE_FRACTION,
    min_prob: float = 0.0,
    gate: str = "off",
    start_bankroll: float = 1.0,
    market_type: str | None = None,
) -> dict:
    """Replay a strategy over canonical market rows in chronological order.

    ``gate`` is ``"off"`` for the base strategy or ``"gated"`` for rows
    accepted by that row's isolated market-local gate.  Pushes and incomplete
    outcomes are excluded from bets, never converted into losses.
    """
    ordered, raw_games = _summary_rows(
        _rows_for_backtest(rows, market_type, legacy_american=american)
    )
    equity = float(start_bankroll)
    peak = equity
    max_drawdown = 0.0
    wins = losses = 0
    gross_win = gross_loss = total_staked = 0.0
    per_bet_pnl: list[float] = []
    equity_curve: list[dict] = []

    for row in ordered:
        bet = _bet_row(row, gate)
        if bet is None or (gate == "off" and bet["prob"] < float(min_prob)):
            continue
        stake = kelly_stake(
            bet["prob"],
            american,
            closing_odds=bet["closingOdds"],
            fractional=fractional,
            cap=cap,
        )
        if stake <= 0:
            continue
        # Universal market payout: no team- or market-specific branch exists
        # below this line.
        pnl = stake * (bet["closingOdds"] - 1.0) if bet["isWin"] == 1 else -stake
        equity += pnl
        max_drawdown = max(max_drawdown, (peak - equity) / peak if peak > 0 else 0.0)
        peak = max(peak, equity)
        if bet["isWin"] == 1:
            wins += 1
            gross_win += pnl
        else:
            losses += 1
            gross_loss += stake
        total_staked += stake
        per_bet_pnl.append(pnl)
        equity_curve.append({
            "date": bet["date"],
            "gamePk": bet["gamePk"],
            "marketType": bet["marketType"],
            "marketLine": bet["marketLine"],
            "side": bet["modelSide"],
            "closingOdds": bet["closingOdds"],
            "pnl": round(pnl, 6),
            "bankroll": round(equity, 6),
        })

    bets = wins + losses
    hit_rate = wins / bets if bets else 0.0
    roi = (gross_win - gross_loss) / total_staked if total_staked else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    mean_pnl = sum(per_bet_pnl) / len(per_bet_pnl) if per_bet_pnl else 0.0
    variance = sum((x - mean_pnl) ** 2 for x in per_bet_pnl) / len(per_bet_pnl) if per_bet_pnl else 0.0
    return {
        "version": BACKTEST_VERSION,
        "architectureMetadata": MARKET_ARCHITECTURE_METADATA,
        "american": float(american),
        "defaultClosingOdds": DEFAULT_DECIMAL_CLOSING_ODDS,
        "breakeven": round(breakeven_probability(american), 6),
        "fractionalKelly": float(fractional),
        "maxStakeFraction": float(cap),
        "gate": gate,
        "marketType": market_type.upper() if market_type else "GLOBAL",
        "minProb": round(float(min_prob), 6),
        "games": raw_games,
        "bets": bets,
        "coverage": bets / raw_games if raw_games else 0.0,
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
        "perBetSharpe": round(mean_pnl / math.sqrt(variance), 4) if variance > 1e-12 else 0.0,
        "equityCurve": equity_curve,
    }


def threshold_sweep(
    rows: list[dict],
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    *,
    american: float = BENCHMARK_AMERICAN_ODDS,
    market_type: str | None = None,
) -> list[dict]:
    """Confidence-threshold sweep for a global or isolated market track."""
    return [
        {
            "minProb": threshold,
            **{
                key: value
                for key, value in run_backtest(
                    rows,
                    american=american,
                    min_prob=threshold,
                    gate="off",
                    market_type=market_type,
                ).items()
                if key in {"bets", "coverage", "hitRate", "roi", "maxDrawdown", "finalBankroll"}
            },
        }
        for threshold in thresholds
    ]


def _legacy_moneyline_rows(rows: list[dict]) -> list[dict]:
    """Keep the old ``base``/``gated`` keys useful for existing callers."""
    return [r for r in expand_market_rows(rows) if r.get("market_type") == "MONEYLINE"]


def build_execution_backtest(rows: list[dict]) -> dict:
    """Build global plus per-market paper-trading payloads.

    Legacy callers still receive ``base``, ``gated`` and ``thresholdSweep``
    for moneyline. New callers receive the normalized execution rows, a global
    summary, three independent market slices, and isolated stack/gate tracks.
    """
    payload = build_market_payload(rows)
    # ``build_market_payload`` returns a separate copy per track, each already
    # annotated with its own market-local gate decision. Recombine those rows
    # without allowing one market's gate state to overwrite another's.
    execution_rows = [
        row
        for market_type in MARKET_TYPES
        for row in payload["tracks"][market_type]["rows"]
    ]
    legacy_ml = _legacy_moneyline_rows(rows)
    global_base = run_backtest(execution_rows, gate="off")
    global_gated = run_backtest(execution_rows, gate="gated")
    market_summaries: dict[str, dict] = {}
    for market_type in MARKET_TYPES:
        track_rows = payload["tracks"][market_type]["rows"]
        market_summaries[market_type] = {
            "marketType": market_type,
            "base": run_backtest(track_rows, gate="off", market_type=market_type),
            "gated": run_backtest(track_rows, gate="gated", market_type=market_type),
            "thresholdSweep": threshold_sweep(track_rows, market_type=market_type),
            "track": {
                key: value
                for key, value in payload["tracks"][market_type].items()
                if key not in {"rows", "settledRows"}
            },
        }
    # For a legacy list, preserve the historical one-row-per-game moneyline
    # summary expected by the existing offline tests. A genuinely market
    # centric list uses the global all-market summary for the compatibility key.
    is_legacy = not any(
        isinstance(r, dict)
        and normalize_market_type(r.get("market_type") or r.get("marketType")) is not None
        for r in rows
    )
    compatibility_base_rows = legacy_ml if is_legacy else execution_rows
    compatibility_base = run_backtest(compatibility_base_rows, gate="off")
    compatibility_gated = run_backtest(compatibility_base_rows, gate="gated")
    return {
        "version": BACKTEST_VERSION,
        "architectureMetadata": MARKET_ARCHITECTURE_METADATA,
        "price": BENCHMARK_AMERICAN_ODDS,
        "defaultClosingOdds": DEFAULT_DECIMAL_CLOSING_ODDS,
        "breakeven": breakeven_probability(),
        "base": compatibility_base,
        "gated": compatibility_gated,
        "thresholdSweep": threshold_sweep(compatibility_base_rows),
        "global": {"base": global_base, "gated": global_gated, "thresholdSweep": threshold_sweep(execution_rows)},
        "markets": market_summaries,
        "executionRows": execution_rows,
        "marketTracks": payload["summaries"],
    }
