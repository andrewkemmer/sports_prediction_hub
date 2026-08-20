"""Market-centric execution and calibration helpers for the Streamlit app.

The model is trained from the MLB Stats API, while market prices are an
optional downstream execution input.  This module keeps the three market tasks
separate and makes the distinction explicit in every execution row:

* ``MONEYLINE`` -- HOME/AWAY at line 0.0
* ``TOTAL`` -- OVER/UNDER at a runs total
* ``RUN_LINE`` -- HOME/AWAY at a run line

Historical sharp closing prices are not available from the live odds endpoint.
When a historical row does not carry an explicit decimal closing price, the
row is marked with the transparent flat -110 benchmark fallback (1.9091
Decimal) rather than silently treating a current quote as a historical close.
All helpers are pure and use only the supplied point-in-time row data.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from .logistic import build_stacking_weights
from .metrics import compute_auc, compute_brier, evaluate, roundn

MARKET_TYPES = ("MONEYLINE", "TOTAL", "RUN_LINE")
DEFAULT_DECIMAL_CLOSING_ODDS = 1.9091  # -110 benchmark
MARKET_ARCHITECTURE_METADATA = "Full-Market CLV / Multi-Line Architecture Active"
MARKET_SCHEMA_VERSION = 1


def _number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_market_type(value) -> str | None:
    """Normalize legacy market labels to the persisted enum."""
    if not isinstance(value, str):
        return None
    key = value.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "ML": "MONEYLINE",
        "MONEY_LINE": "MONEYLINE",
        "MONEYLINE": "MONEYLINE",
        "TOTALS": "TOTAL",
        "GAME_TOTAL": "TOTAL",
        "GAME_TOTALS": "TOTAL",
        "TOTAL": "TOTAL",
        "RUNLINE": "RUN_LINE",
        "RUN_LINES": "RUN_LINE",
        "SPREAD": "RUN_LINE",
        "SPREADS": "RUN_LINE",
        "RUN_LINE": "RUN_LINE",
    }
    return aliases.get(key)


def american_to_decimal(odds) -> float | None:
    """Convert an American price when a legacy close is explicitly labelled."""
    value = _number(odds)
    if value is None or value == 0:
        return None
    return 1.0 + (100.0 / abs(value) if value < 0 else value / 100.0)


def normalize_decimal_odds(value, fallback: float = DEFAULT_DECIMAL_CLOSING_ODDS) -> tuple[float, bool]:
    """Return a valid decimal price and whether the fallback was required."""
    number = _number(value)
    if number is not None and number > 1.0:
        return number, False
    return float(fallback), True


def _timestamp(value) -> float | None:
    number = _number(value)
    if number is not None:
        return number / 1000.0 if abs(number) > 10_000_000_000 else number
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            return None
    return None


def _is_pit_row(row: dict) -> bool:
    """Reject an explicitly post-start market snapshot.

    Date-only historical rows are accepted because an explicit ``closing_odds``
    value is already a historical source field.  If both clocks are present,
    the quote must precede the scheduled first pitch.
    """
    quote = row.get("marketOdds") or row.get("market_odds")
    quote = quote if isinstance(quote, dict) else {}
    snapshot = _timestamp(
        row.get("market_snapshot_at")
        or row.get("marketSnapshotAt")
        or row.get("snapshotAt")
        or quote.get("market_snapshot_at")
        or quote.get("marketSnapshotAt")
        or quote.get("snapshotAt")
    )
    target = _timestamp(
        row.get("gameDate")
        or row.get("game_date")
        or row.get("firstPitchAt")
        or quote.get("gameDate")
        or quote.get("game_date")
        or quote.get("firstPitchAt")
    )
    # An explicit snapshot without a target clock cannot be proven pre-game;
    # reject it rather than accidentally treating a post-start quote as a
    # historical close. Date-only rows with no snapshot remain acceptable
    # because their explicit close is already a historical source field.
    if snapshot is None:
        return True
    if target is None:
        return False
    return snapshot < target


def _market_key(market_type: str) -> str:
    return market_type.lower().replace("_", "")


def _nested_value(value, market_type: str, side: str | None = None):
    """Read flat, market-keyed, or market-and-side-keyed legacy payloads."""
    if not isinstance(value, dict):
        return value
    market_keys = [market_type, market_type.lower(), _market_key(market_type)]
    side_keys = []
    if side:
        side_key = side.upper()
        side_keys = [side_key, side.lower(), side_key.replace("_", "")]
        side_keys.extend([
            f"{market_type}_{side_key}",
            f"{market_type.lower()}_{side.lower()}",
            f"{_market_key(market_type)}_{side.lower()}",
        ])
    for key in market_keys + side_keys:
        if key not in value:
            continue
        found = value[key]
        if isinstance(found, dict) and side:
            for nested_key in side_keys[:3]:
                if nested_key in found:
                    return found[nested_key]
            return None
        return found
    return None


def closing_odds_value(row: dict, market_type: str, side: str | None = None):
    """Find an explicitly supplied decimal historical close.

    Deliberately does *not* use ``homeMoneyline``/``overPrice``/spread prices:
    those fields are live quotes, not historical closing prices.  A legacy
    ``americanClosingOdds`` field is converted only because it is explicitly
    labelled as a close.
    """
    sources = [row]
    quote = row.get("marketOdds") or row.get("market_odds")
    if isinstance(quote, dict):
        sources.append(quote)
    names = (
        "closing_odds",
        "closingOdds",
        "decimal_closing_odds",
        "decimalClosingOdds",
        "sharpClosingOdds",
        "sharp_closing_odds",
    )
    american_names = ("americanClosingOdds", "american_closing_odds")
    for source in sources:
        for name in names:
            if name in source:
                value = _nested_value(source[name], market_type, side)
                if value is not None:
                    return value
        for name in american_names:
            if name in source:
                value = _nested_value(source[name], market_type, side)
                converted = american_to_decimal(value)
                if converted is not None:
                    return converted
        # Flat per-market field names are common in CSV migrations.
        for prefix in (market_type, market_type.lower(), _market_key(market_type)):
            for suffix in ("ClosingOdds", "_closing_odds", "_closingOdds"):
                name = f"{prefix}{suffix}"
                if name in source:
                    return source[name]
    return None


def _market_quote(row: dict) -> dict:
    quote = row.get("marketOdds") or row.get("market_odds")
    return quote if isinstance(quote, dict) else {}


def _value_for_market(row: dict, market_type: str, names: tuple[str, ...]):
    quote = _market_quote(row)
    for source in (row, quote):
        for name in names:
            if name in source:
                value = source[name]
                if isinstance(value, dict):
                    value = _nested_value(value, market_type)
                if value is not None:
                    return value
    return None


def _market_line(row: dict, market_type: str, predicted_total: float | None = None) -> tuple[float, str]:
    if market_type == "MONEYLINE":
        return 0.0, "moneyline_reference"
    if market_type == "TOTAL":
        value = _value_for_market(
            row,
            market_type,
            ("market_line", "marketLine", "totalLine", "total_line", "line", "total"),
        )
        line = _number(value)
        if line is not None:
            return line, "source_market_line"
        # A model projection is a non-market fallback, never an actual result.
        # Keep it explicit so a dashboard cannot mistake it for a sharp line.
        if predicted_total is not None:
            return round(predicted_total * 2.0) / 2.0, "model_projection_fallback"
        return 0.0, "missing_line_fallback"
    value = _value_for_market(
        row,
        market_type,
        ("market_line", "marketLine", "runLine", "run_line", "spread", "line"),
    )
    line = _number(value)
    return (line, "source_market_line") if line is not None else (1.5, "standard_run_line_fallback")


def _probability_for_market(row: dict, market_type: str, side: str | None, predicted_total: float | None, line: float) -> float:
    if market_type == "MONEYLINE":
        value = row.get("model_probability", row.get("modelProb", row.get("pickProb")))
    elif market_type == "TOTAL":
        over = _number(row.get("overProb", row.get("over_probability")))
        under = _number(row.get("underProb", row.get("under_probability")))
        if side == "OVER":
            value = over
        elif side == "UNDER":
            value = under
        else:
            value = None
        # A legacy moneyline ``pickProb`` is deliberately not a fallback for
        # totals. Cross-market probability bleed would make the totals track
        # appear calibrated by the moneyline model. A canonical market row may
        # provide ``model_probability`` explicitly; otherwise remain neutral.
        if value is None:
            value = row.get("model_probability", row.get("modelProb"))
        if value is None:
            value = 0.5
    else:
        value = row.get("homeRunLineProb", row.get("home_run_line_prob"))
        if value is None:
            value = row.get("model_probability", row.get("modelProb"))
        if value is None:
            value = 0.5
        if side == "AWAY":
            value = 1.0 - float(value) if _number(value) is not None else None
    probability = _number(value)
    return min(0.999, max(0.001, probability)) if probability is not None else 0.5


def _side_for_market(row: dict, market_type: str, predicted_total: float | None, line: float) -> str:
    if market_type == "MONEYLINE":
        side = row.get("model_side", row.get("modelSide"))
        if isinstance(side, str) and side.upper() in ("HOME", "AWAY"):
            return side.upper()
        pick = row.get("pickTeam")
        return "HOME" if str(pick).lower() == "home" else "AWAY"
    if market_type == "TOTAL":
        side = row.get("model_side", row.get("modelSide"))
        if isinstance(side, str) and side.upper() in ("OVER", "UNDER"):
            return side.upper()
        over = _number(row.get("overProb", row.get("over_probability")))
        if over is not None:
            return "OVER" if over >= 0.5 else "UNDER"
        return "OVER" if (predicted_total is not None and predicted_total > line) else "UNDER"
    side = row.get("model_side", row.get("modelSide"))
    if isinstance(side, str) and side.upper() in ("HOME", "AWAY"):
        return side.upper()
    prob = _number(row.get("homeRunLineProb"))
    return "HOME" if prob is None or prob >= 0.5 else "AWAY"


def _settle_market(row: dict, market_type: str, side: str, line: float):
    # Explicit settlement is authoritative for migrated execution rows. When
    # it is absent, derive the result from the market-specific final score
    # fields below; this keeps fixture/legacy rows usable without inventing a
    # payout from a missing total or margin.
    value = row.get("is_win", row.get("isWin", row.get("isCorrect")))
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if market_type == "MONEYLINE":
        return None
    actual_total = _number(row.get("actualTotal", row.get("actual_total")))
    if market_type == "TOTAL":
        if actual_total is None:
            return None
        if abs(actual_total - line) < 1e-9:
            return None  # push; no binary payout is invented
        won = actual_total > line if side == "OVER" else actual_total < line
        return 1 if won else 0
    actual_margin = _number(row.get("actualMargin", row.get("actual_margin")))
    if actual_margin is None:
        return None
    # ``market_line`` is signed for the selected side: HOME -1.5 means the
    # home team must win by more than 1.5; AWAY +1.5 means the away team may
    # lose by fewer than 1.5. This also handles whole-run pushes explicitly.
    selected_margin = actual_margin if side == "HOME" else -actual_margin
    margin_after_line = selected_margin + line
    if abs(margin_after_line) < 1e-9:
        return None
    return 1 if margin_after_line > 0 else 0


def _candidate_predictions(row: dict, market_type: str, side: str, probability: float) -> dict[str, float]:
    source = row.get("candidatePredictions") or row.get("candidate_predictions")
    out: dict[str, float] = {}
    if isinstance(source, dict):
        for name, value in source.items():
            number = _number(value)
            if number is not None:
                out[str(name)] = min(0.999, max(0.001, number))
    if out:
        return out
    label = {
        "MONEYLINE": "Deployed moneyline model",
        "TOTAL": "Poisson total model",
        "RUN_LINE": "Poisson run-line model",
    }[market_type]
    return {label: probability}


def normalize_market_row(row: dict, market_type: str | None = None) -> dict:
    """Normalize one execution row and enforce the odds/PIT guardrails."""
    raw_type = market_type or row.get("market_type") or row.get("marketType")
    normalized_type = normalize_market_type(raw_type) or "MONEYLINE"
    predicted_total = _number(row.get("predictedTotal", row.get("predicted_total")))
    line, line_source = _market_line(row, normalized_type, predicted_total)
    side = _side_for_market(row, normalized_type, predicted_total, line)
    if normalized_type == "RUN_LINE" and line_source == "standard_run_line_fallback":
        # The default MLB convention is HOME -1.5 / AWAY +1.5. Keep the
        # stored line signed so settlement and downstream serializers agree.
        line = -1.5 if side == "HOME" else 1.5
    probability = _probability_for_market(row, normalized_type, side, predicted_total, line)
    explicit_close = closing_odds_value(row, normalized_type, side)
    closing_odds, used_fallback = normalize_decimal_odds(explicit_close)
    if not _is_pit_row(row):
        # A post-start quote cannot be used as a historical close.
        closing_odds, used_fallback = DEFAULT_DECIMAL_CLOSING_ODDS, True
    is_win = _settle_market(row, normalized_type, side, line)
    source = "flat_minus_110_fallback" if used_fallback else "historical_sharp_close"
    quote = row.get("marketOdds") or row.get("market_odds")
    quote = quote if isinstance(quote, dict) else {}
    snapshot = (
        row.get("market_snapshot_at")
        or row.get("marketSnapshotAt")
        or row.get("snapshotAt")
        or quote.get("market_snapshot_at")
        or quote.get("marketSnapshotAt")
        or quote.get("snapshotAt")
    )
    normalized = dict(row)
    normalized.update({
        "market_type": normalized_type,
        "marketType": normalized_type,
        "market_line": float(line),
        "marketLine": float(line),
        "model_side": side,
        "modelSide": side,
        "model_probability": probability,
        "modelProb": probability,
        "is_win": is_win,
        "isWin": is_win,
        "closing_odds": float(closing_odds),
        "closingOdds": float(closing_odds),
        "closing_odds_source": source,
        "market_line_source": line_source,
        "market_snapshot_at": snapshot,
        "pit_safe": _is_pit_row(row),
        "candidatePredictions": _candidate_predictions(row, normalized_type, side, probability),
    })
    return normalized


def market_rows_for_calibration(row: dict) -> list[dict]:
    """Expand one completed calibration row into the three market vectors."""
    # The walk-forward selector stores home-side candidate probabilities. Turn
    # them into probabilities for the selected HOME/AWAY side without mixing
    # candidates between market tracks.
    money_side = _side_for_market(row, "MONEYLINE", None, 0.0)
    raw_candidates = (
        row.get("candPreds")
        or row.get("candidatePreds")
        or row.get("candidatePredictions")
        or row.get("candidate_predictions")
    )
    candidate_predictions = None
    if isinstance(raw_candidates, dict):
        candidate_predictions = {
            str(name): (float(value) if money_side == "HOME" else 1.0 - float(value))
            for name, value in raw_candidates.items()
            if _number(value) is not None
        }

    total_line, total_line_source = _market_line(row, "TOTAL", _number(row.get("predictedTotal")))
    total_prob = _probability_for_market(
        row,
        "TOTAL",
        _side_for_market(row, "TOTAL", _number(row.get("predictedTotal")), total_line),
        _number(row.get("predictedTotal")),
        total_line,
    )
    run_line, run_line_source = _market_line(row, "RUN_LINE", None)
    run_side = _side_for_market(row, "RUN_LINE", None, run_line)
    if run_line_source == "standard_run_line_fallback":
        run_line = -1.5 if run_side == "HOME" else 1.5
    run_prob = _probability_for_market(row, "RUN_LINE", run_side, None, run_line)

    common = {
        "gamePk": row.get("gamePk"),
        "date": row.get("date"),
        "gameDate": row.get("gameDate"),
        "home": row.get("home"),
        "away": row.get("away"),
        "winner": row.get("winner"),
        "actualTotal": row.get("actualTotal"),
        "actualMargin": row.get("actualMargin"),
        "market_snapshot_at": row.get("market_snapshot_at") or row.get("marketSnapshotAt"),
    }
    money = normalize_market_row({
        **common,
        "market_type": "MONEYLINE",
        "model_side": money_side,
        "model_probability": row.get("pickProb"),
        "is_win": row.get("isCorrect"),
        "candidatePredictions": candidate_predictions,
        "closing_odds": row.get(
            "moneylineClosingOdds",
            row.get("moneyline_closing_odds", row.get("closingOdds", row.get("closing_odds"))),
        ),
    })
    gated_side = row.get("gatedPickTeam")
    gated_prob = _number(row.get("gatedPickProb"))
    money["gated_model_side"] = str(gated_side).upper() if gated_side in ("home", "away", "HOME", "AWAY") else None
    money["gated_model_probability"] = gated_prob
    money["gated_is_win"] = (1 if row.get("gatedIsCorrect") is True else 0) if row.get("gatedIsCorrect") is not None else None
    money["gate_accepted"] = row.get("gateAccepted") is True

    total_side = _side_for_market(row, "TOTAL", _number(row.get("predictedTotal")), total_line)
    total = normalize_market_row({
        **common,
        "market_type": "TOTAL",
        "market_line": total_line,
        "model_side": total_side,
        "model_probability": total_prob,
        "overProb": row.get("overProb"),
        "underProb": row.get("underProb"),
        # Do not settle a totals bet against a rounded model projection when
        # the historical market line was never captured. The row remains in
        # the payload with a float line, but execution correctly skips it.
        "is_win": (
            _settle_market(row, "TOTAL", total_side, total_line)
            if total_line_source == "source_market_line" else None
        ),
        "closing_odds": row.get(
            "totalClosingOdds",
            row.get("total_closing_odds", row.get("closingOdds", row.get("closing_odds"))),
        ),
    })
    total["gate_accepted"] = False

    run = normalize_market_row({
        **common,
        "market_type": "RUN_LINE",
        "market_line": run_line,
        "model_side": run_side,
        "model_probability": run_prob,
        "homeRunLineProb": row.get("homeRunLineProb"),
        "is_win": _settle_market(row, "RUN_LINE", run_side, run_line),
        "closing_odds": row.get(
            "runLineClosingOdds",
            row.get("run_line_closing_odds", row.get("closingOdds", row.get("closing_odds"))),
        ),
    })
    run["gate_accepted"] = False
    return [money, total, run]


def expand_market_rows(rows: list[dict]) -> list[dict]:
    """Return canonical rows, accepting both legacy game rows and new vectors."""
    expanded: list[dict] = []
    for row in rows or []:
        nested = row.get("marketRows") or row.get("market_rows")
        if isinstance(nested, list) and nested:
            expanded.extend(normalize_market_row(item) for item in nested if isinstance(item, dict))
            continue
        if normalize_market_type(row.get("market_type") or row.get("marketType")):
            expanded.append(normalize_market_row(row))
        else:
            expanded.extend(market_rows_for_calibration(row))
    return expanded


def normalize_weights(weights: dict[str, float] | None, names: list[str]) -> dict[str, float]:
    clean = {
        name: max(0.0, float((weights or {}).get(name, 0.0)))
        for name in names
        if _number((weights or {}).get(name, 0.0)) is not None
    }
    total = sum(clean.values())
    if total <= 0:
        return {names[0]: 1.0} if names else {}
    return {name: value / total for name, value in clean.items() if value > 0}


def normalized_weight_rows(weights: dict[str, float] | None, names: list[str]) -> list[dict]:
    """Serialize one exact simplex vector across the complete candidate pool.

    Every candidate is retained in the payload, including explicit zeroes. A
    final drift correction makes the floating-point sum exactly 1.0 so UI and
    execution serializers cannot accidentally show multiple 100% allocations
    or a vector that is slightly off the convex-combination constraint.
    """
    if not names:
        return []
    values = {
        name: max(0.0, float((weights or {}).get(name, 0.0)))
        if _number((weights or {}).get(name, 0.0)) is not None else 0.0
        for name in names
    }
    total = sum(values.values())
    if total <= 0:
        values = {name: (1.0 if index == 0 else 0.0) for index, name in enumerate(names)}
    else:
        values = {name: value / total for name, value in values.items()}
    drift = 1.0 - sum(values.values())
    anchor = next((name for name in reversed(names) if values.get(name, 0.0) > 0), names[0])
    values[anchor] = max(0.0, values.get(anchor, 0.0) + drift)
    return [{"name": name, "weight": values.get(name, 0.0)} for name in names]


def _wilson_lower(wins: int, total: int) -> float:
    if total <= 0:
        return 0.0
    z = 1.96
    p = wins / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (center - spread) / denominator


def tune_market_gate(rows: list[dict]) -> dict:
    """Tune a market-local confidence gate on the trailing PIT validation slice."""
    settled = [r for r in rows if r.get("is_win") in (0, 1) and _number(r.get("model_probability")) is not None]
    base = {
        "version": 1,
        "enabled": False,
        "threshold": 0.60,
        "minSignals": 1,
        "validationRows": 0,
        "validationCoverage": 0.0,
        "validationWinRate": 0.0,
        "validationBaselineWinRate": 0.0,
        "validationLift": 0.0,
        "reason": "Not enough market-specific PIT validation history",
    }
    if len(settled) < 20:
        return base
    start = max(0, int(len(settled) * 0.70))
    validation = settled[start:]
    baseline = sum(int(r["is_win"] == 1) for r in validation) / len(validation)
    candidates = []
    for threshold in (0.52, 0.55, 0.60, 0.65, 0.70):
        accepted = [r for r in validation if max(r["model_probability"], 1.0 - r["model_probability"]) >= threshold]
        if len(accepted) < max(5, int(math.ceil(len(validation) * 0.20))):
            continue
        wins = sum(int(r["is_win"] == 1) for r in accepted)
        rate = wins / len(accepted)
        candidates.append({
            "threshold": threshold,
            "accepted": len(accepted),
            "coverage": len(accepted) / len(validation),
            "winRate": rate,
            "lift": rate - baseline,
            "wilsonLower": _wilson_lower(wins, len(accepted)),
        })
    if not candidates:
        base.update({"validationRows": len(validation), "validationBaselineWinRate": baseline})
        return base
    best = max(candidates, key=lambda c: (c["wilsonLower"], c["winRate"], c["coverage"], -c["threshold"]))
    enabled = best["lift"] >= 0.005
    base.update({
        "enabled": enabled,
        "threshold": best["threshold"],
        "validationRows": len(validation),
        "validationCoverage": best["coverage"],
        "validationWinRate": best["winRate"],
        "validationBaselineWinRate": baseline,
        "validationLift": best["lift"],
        "validationAccepted": best["accepted"],
        "validationWilsonLower": best["wilsonLower"],
        "reason": "Market-local prior-only confidence gate selected" if enabled else "Market-local gate held out; no guarded lift",
    })
    return base


def apply_market_gate(row: dict, config: dict) -> bool:
    probability = _number(row.get("model_probability"))
    if probability is None or not config.get("enabled"):
        return False
    return max(probability, 1.0 - probability) >= float(config.get("threshold", 0.60))


def build_market_track(rows: list[dict], market_type: str) -> dict:
    """Build one isolated selection/stack/gate track for one market."""
    normalized_type = normalize_market_type(market_type) or market_type
    market_rows = [r for r in rows if r.get("market_type") == normalized_type]
    settled = [r for r in market_rows if r.get("is_win") in (0, 1)]
    labels = [int(r["is_win"]) for r in settled]
    candidate_names = sorted({
        name
        for r in settled
        for name in (r.get("candidatePredictions") or {}).keys()
    })
    candidate_preds = {
        name: [float((r.get("candidatePredictions") or {}).get(name, r.get("model_probability", 0.5))) for r in settled]
        for name in candidate_names
    }
    candidates = []
    for name in candidate_names:
        metrics = evaluate(candidate_preds[name], labels) if labels else {"auc": 0.0, "brier": 0.0, "logLoss": 0.0, "ece": 0.0}
        eligible = metrics["auc"] >= 0.51
        candidates.append({
            "name": name,
            "auc": metrics["auc"],
            "brier": metrics["brier"],
            "logLoss": metrics["logLoss"],
            "ece": metrics["ece"],
            "eligible": eligible,
            "selected": False,
            "note": "" if eligible else "Below 0.51 AUC floor",
        })
    if candidates:
        eligible = [c for c in candidates if c["eligible"]]
        if not eligible:
            # Small or flat market slices should still expose a valid deployed
            # candidate rather than label every model excluded and then mark
            # one of those excluded rows as selected.
            for candidate in candidates:
                candidate["eligible"] = True
                candidate["note"] = "AUC floor relaxed — no market candidate cleared 0.51"
            eligible = candidates
        best = min(eligible, key=lambda c: (c["brier"], -c["auc"], c["name"]))
        best["selected"] = True
    if candidate_preds and labels:
        stacking = build_stacking_weights(candidate_preds, labels)
        weights = {
            item["name"]: float(item.get("weight", 0.0))
            for item in stacking.get("weights", [])
        }
    else:
        stacking = {"brier": 0.0, "weights": []}
        weights = {}
    weights = normalize_weights(weights, candidate_names)
    # Serialize one global convex allocation vector for this market. Keep
    # zero-weight diagnostics in the payload, then correct any floating-point
    # drift on the last active member so the vector sums to exactly 1.0.
    stacking_weights = [
        {"name": name, "weight": float(weights.get(name, 0.0))}
        for name in candidate_names
    ]
    weight_total = sum(item["weight"] for item in stacking_weights)
    if stacking_weights and weight_total > 0:
        for item in stacking_weights:
            item["weight"] /= weight_total
        drift = 1.0 - sum(item["weight"] for item in stacking_weights)
        anchor = next((item for item in reversed(stacking_weights) if item["weight"] > 0), stacking_weights[0])
        anchor["weight"] = max(0.0, anchor["weight"] + drift)
    gate = tune_market_gate(market_rows)
    for row in market_rows:
        row["market_gate_accepted"] = apply_market_gate(row, gate)
        row["marketGateAccepted"] = row["market_gate_accepted"]
    if settled:
        predictions = [float(r.get("model_probability", 0.5)) for r in settled]
        model_metrics = evaluate(predictions, labels)
    else:
        model_metrics = {"auc": 0.0, "brier": 0.0, "logLoss": 0.0, "ece": 0.0, "bins": []}
    return {
        "marketType": normalized_type,
        "market_type": normalized_type,
        "selectedModel": next((c["name"] for c in candidates if c.get("selected")), "No settled model"),
        "candidates": candidates,
        "stackingWeights": stacking_weights,
        "stackBrier": stacking.get("brier", 0.0),
        "metrics": model_metrics,
        "gate": gate,
        "gateState": "ACTIVE" if gate.get("enabled") else "HELD_OUT",
        "rows": market_rows,
        "settledRows": len(settled),
        "fitScope": "PIT calibration rows; market tracks are isolated and never used to cross-fit another market",
    }


def build_market_payload(rows: list[dict]) -> dict:
    """Build the serialized multi-market payload used by refresh and Streamlit."""
    expanded = expand_market_rows(rows)
    tracks = {
        market_type: build_market_track(
            [dict(row) for row in expanded], market_type
        )
        for market_type in MARKET_TYPES
    }
    summaries = {
        market_type: {
            "marketType": market_type,
            "selectedModel": track["selectedModel"],
            "metrics": track["metrics"],
            "gate": track["gate"],
            "stackingWeights": track["stackingWeights"],
            "settledRows": track["settledRows"],
        }
        for market_type, track in tracks.items()
    }
    return {
        "version": MARKET_SCHEMA_VERSION,
        "architectureMetadata": MARKET_ARCHITECTURE_METADATA,
        "marketTypes": list(MARKET_TYPES),
        "tracks": tracks,
        "summaries": summaries,
        "rows": expanded,
    }
