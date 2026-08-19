"""PIT-safe market mapping and bet execution diagnostics.

This module is deliberately downstream of model inference.  It never changes
model probabilities, is never used to train or calibrate the win model, and
refuses to issue a bet when the market snapshot cannot be proven to be
pre-game.  Historical ROI cannot be reconstructed from the current live odds
feed; callers therefore pass no odds for historical rows.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone

from .metrics import clamp

BETTING_VERSION = 1
MIN_MODEL_EDGE = 0.02  # model probability minus no-vig market probability
MIN_EXPECTED_VALUE = 0.01  # expected profit per unit stake
FRACTIONAL_KELLY = 0.25
MAX_STAKE_FRACTION = 0.01  # hard 1% bankroll cap before portfolio limits
MAX_MARKET_AGE_SECONDS = 6 * 60 * 60


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def american_to_decimal(odds) -> float | None:
    """Convert American odds to decimal odds; 0/invalid prices are rejected."""
    value = _number(odds)
    if value is None or value == 0:
        return None
    return 1.0 + (100.0 / abs(value) if value < 0 else value / 100.0)


def american_implied_probability(odds) -> float | None:
    """Raw (vig-inclusive) implied probability for an American price."""
    decimal = american_to_decimal(odds)
    return 1.0 / decimal if decimal and decimal > 1.0 else None


def no_vig_two_way(home_odds, away_odds) -> dict | None:
    """Return normalized two-way probabilities and overround."""
    home_raw = american_implied_probability(home_odds)
    away_raw = american_implied_probability(away_odds)
    if home_raw is None or away_raw is None:
        return None
    total = home_raw + away_raw
    if total <= 0:
        return None
    return {
        "homeRaw": home_raw,
        "awayRaw": away_raw,
        "home": home_raw / total,
        "away": away_raw / total,
        "overround": total - 1.0,
    }


def _timestamp_ms(value) -> float | None:
    numeric = _number(value)
    if numeric is not None:
        # Accept seconds as well as milliseconds for hand-built/test payloads.
        return numeric * 1000.0 if numeric < 10_000_000_000 else numeric
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp() * 1000.0
        except ValueError:
            return None
    return None


def _game_start_ms(game_date) -> float | None:
    return _timestamp_ms(game_date)


def _base_result(reason: str, *, min_edge: float, now_ms: float) -> dict:
    return {
        "version": BETTING_VERSION,
        "available": False,
        "recommended": False,
        "action": "PASS",
        "side": None,
        "team": None,
        "offeredOdds": None,
        "modelProb": None,
        "marketImpliedProb": None,
        "marketNoVigProb": None,
        "edge": None,
        "expectedValue": None,
        "kellyFraction": 0.0,
        "recommendedStakeFraction": 0.0,
        "fractionalKelly": FRACTIONAL_KELLY,
        "maxStakeFraction": MAX_STAKE_FRACTION,
        "overround": None,
        "snapshotAt": None,
        "snapshotAgeSeconds": None,
        "minEdge": min_edge,
        "minExpectedValue": MIN_EXPECTED_VALUE,
        "asOfMs": now_ms,
        "gateRequired": False,
        "gateAccepted": None,
        "gateConcordance": None,
        "reason": reason,
    }


def build_bet_decision(
    prediction: dict,
    market_odds: dict | None,
    *,
    game_date: str | None = None,
    game_status: str | None = None,
    min_edge: float = MIN_MODEL_EDGE,
    min_expected_value: float = MIN_EXPECTED_VALUE,
    fractional_kelly: float = FRACTIONAL_KELLY,
    max_stake_fraction: float = MAX_STAKE_FRACTION,
    now_ms: float | None = None,
) -> dict:
    """Map one pre-game prediction to a conservative executable decision.

    ``market_odds`` must contain a ``snapshotAt`` timestamp.  The snapshot must
    be before ``game_date`` and no older than ``MAX_MARKET_AGE_SECONDS`` at
    decision time.  ``homeMoneyline``/``awayMoneyline`` are the sharp-market
    benchmark prices; ``bestHomeMoneyline``/``bestAwayMoneyline`` are optional
    executable best prices and fall back to the benchmark.

    The returned ``edge`` is against the no-vig benchmark, while
    ``expectedValue`` is calculated from the executable offered price.  Kelly
    is quarter-Kelly and hard-capped; it is a sizing diagnostic, not a claim
    that the model's uncertainty is fully known.
    """
    now = float(now_ms if now_ms is not None else time.time() * 1000.0)
    edge_floor = clamp(float(min_edge), 0.0, 0.50)
    ev_floor = max(0.0, float(min_expected_value))
    fractional = clamp(float(fractional_kelly), 0.0, 1.0)
    stake_cap = clamp(float(max_stake_fraction), 0.0, 0.05)
    out = _base_result("No pre-game market snapshot", min_edge=edge_floor, now_ms=now)
    out["minExpectedValue"] = ev_floor
    gate_required = prediction.get("gateEnabled") is True
    gate_accepted = prediction.get("gateAccepted") is True if gate_required else None
    out["gateRequired"] = gate_required
    out["gateAccepted"] = gate_accepted
    out["gateConcordance"] = prediction.get("concordance")

    if not isinstance(market_odds, dict):
        return out
    status = (game_status or "").strip().lower()
    if status and status not in {"scheduled", "preview", "pre-game", "pregame"}:
        out["reason"] = f"Game status is {game_status}; no pre-game wager"
        return out

    snapshot_ms = _timestamp_ms(market_odds.get("snapshotAt"))
    if snapshot_ms is None:
        out["reason"] = "Market timestamp missing; refusing a non-PIT wager"
        return out
    start_ms = _game_start_ms(game_date)
    if start_ms is not None and snapshot_ms > start_ms + 1000.0:
        out["reason"] = "Market snapshot is after game start"
        out["snapshotAt"] = snapshot_ms
        return out
    age_seconds = max(0.0, (now - snapshot_ms) / 1000.0)
    if age_seconds > MAX_MARKET_AGE_SECONDS:
        out["reason"] = "Market snapshot is stale"
        out["snapshotAt"] = snapshot_ms
        out["snapshotAgeSeconds"] = round(age_seconds, 1)
        return out

    benchmark_home = market_odds.get("benchmarkHomeMoneyline", market_odds.get("homeMoneyline"))
    benchmark_away = market_odds.get("benchmarkAwayMoneyline", market_odds.get("awayMoneyline"))
    executable_home = market_odds.get("bestHomeMoneyline", market_odds.get("homeMoneyline"))
    executable_away = market_odds.get("bestAwayMoneyline", market_odds.get("awayMoneyline"))
    market = no_vig_two_way(benchmark_home, benchmark_away)
    home_decimal = american_to_decimal(executable_home)
    away_decimal = american_to_decimal(executable_away)
    if market is None or home_decimal is None or away_decimal is None:
        out["reason"] = "Incomplete two-way moneyline market"
        out["snapshotAt"] = snapshot_ms
        out["snapshotAgeSeconds"] = round(age_seconds, 1)
        return out

    model_home = _number(prediction.get("homeWinProb"))
    if model_home is None:
        out["reason"] = "Model probability unavailable"
        return out
    model_home = clamp(model_home, 0.001, 0.999)
    candidates = [
        {
            "side": "home",
            "modelProb": model_home,
            "marketRaw": market["homeRaw"],
            "marketNoVig": market["home"],
            "odds": float(executable_home),
            "decimal": home_decimal,
            "book": market_odds.get("bestHomeBook", market_odds.get("bookmaker")),
        },
        {
            "side": "away",
            "modelProb": 1.0 - model_home,
            "marketRaw": market["awayRaw"],
            "marketNoVig": market["away"],
            "odds": float(executable_away),
            "decimal": away_decimal,
            "book": market_odds.get("bestAwayBook", market_odds.get("bookmaker")),
        },
    ]
    for candidate in candidates:
        candidate["edge"] = candidate["modelProb"] - candidate["marketNoVig"]
        candidate["expectedValue"] = candidate["modelProb"] * candidate["decimal"] - 1.0
        b = candidate["decimal"] - 1.0
        candidate["kelly"] = max(0.0, candidate["expectedValue"] / b) if b > 0 else 0.0
    best = max(candidates, key=lambda c: (c["expectedValue"], c["edge"], c["modelProb"]))
    gate_blocked = gate_required and not gate_accepted
    passes = (
        not gate_blocked
        and best["edge"] >= edge_floor
        and best["expectedValue"] >= ev_floor
    )
    kelly = min(stake_cap, fractional * best["kelly"]) if passes else 0.0
    if gate_blocked:
        reason = "Concordance gate abstains; no wager is eligible"
    elif passes:
        reason = "Positive EV after no-vig and PIT price checks"
    elif best["expectedValue"] < ev_floor:
        reason = "Best executable price does not clear the EV guard"
    else:
        reason = "Model edge does not clear the no-vig guard"
    return {
        **out,
        "available": True,
        "recommended": passes,
        "action": "BET" if passes else "PASS",
        "side": best["side"] if passes else None,
        "team": best["side"] if passes else None,
        "offeredOdds": best["odds"] if passes else None,
        "candidateSide": best["side"],
        "candidateOdds": best["odds"],
        "candidateExpectedValue": best["expectedValue"],
        "candidateEdge": best["edge"],
        "bookmaker": best["book"],
        "modelProb": best["modelProb"],
        "marketImpliedProb": best["marketRaw"],
        "marketNoVigProb": best["marketNoVig"],
        "edge": best["edge"],
        "expectedValue": best["expectedValue"],
        "kellyFraction": best["kelly"],
        "recommendedStakeFraction": kelly,
        "overround": market["overround"],
        "snapshotAt": snapshot_ms,
        "snapshotAgeSeconds": round(age_seconds, 1),
        "reason": reason,
        "homeExecutableOdds": float(executable_home),
        "awayExecutableOdds": float(executable_away),
    }


def stamp_market_odds(odds_map: dict | None, snapshot_at) -> dict:
    """Attach the cache's original fetch time to each quote without rewriting it.

    The Odds API cache stores the timestamp at the payload level for efficient
    persistence.  The execution layer uses a per-game field so the PIT check
    remains local and explicit; an absent timestamp is intentionally left
    absent and causes a safe PASS.
    """
    if not isinstance(odds_map, dict):
        return {}
    if _timestamp_ms(snapshot_at) is None:
        return dict(odds_map)
    stamped = {}
    for key, value in odds_map.items():
        stamped[key] = {**value, "snapshotAt": snapshot_at} if isinstance(value, dict) else value
    return stamped


def summarize_bet_decisions(docs: list[dict]) -> dict:
    """Summarize live execution opportunities without treating passes as losses."""
    decisions = [d.get("betDecision") for d in docs if isinstance(d.get("betDecision"), dict)]
    available = [d for d in decisions if d.get("available")]
    bets = [d for d in available if d.get("recommended")]
    evs = [float(d["expectedValue"]) for d in bets if _number(d.get("expectedValue")) is not None]
    return {
        "version": BETTING_VERSION,
        "total": len(decisions),
        "marketAvailable": len(available),
        "recommended": len(bets),
        "coverage": len(bets) / len(decisions) if decisions else 0.0,
        "meanExpectedValue": sum(evs) / len(evs) if evs else 0.0,
        "meanStakeFraction": (
            sum(float(d.get("recommendedStakeFraction", 0.0) or 0.0) for d in bets) / len(bets)
            if bets else 0.0
        ),
    }
