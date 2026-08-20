#!/usr/bin/env python3
"""
Export a single consolidated dashboard_payload.json from the Streamlit
cache files for consumption by the React CDN dashboard AND the refactored
Streamlit app (payload mode).

Run from the repo root:
    python3 mlb_streamlit/scripts/export_dashboard_payload.py

Produces:
    mlb_streamlit/cache/dashboard_payload.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mlb_streamlit import cache  # noqa: E402


def _games_by_date(games: list[dict]) -> dict[str, list[dict]]:
    by_date: dict[str, list[dict]] = {}
    for g in games:
        d = g.get("date", "")
        if d:
            by_date.setdefault(d, []).append(g)
    return by_date


def _game_cards(games: list[dict]) -> dict[str, dict]:
    return {str(g["gamePk"]): g for g in games if "gamePk" in g}


def _compute_moneyline_metrics(games: list[dict]) -> dict:
    completed = [
        g for g in games
        if g.get("winner") in ("home", "away")
        and (g.get("home") or {}).get("id")
        and (g.get("away") or {}).get("id")
    ]
    total = len(completed)
    correct = sum(1 for g in completed if g.get("isCorrect"))
    return {"total": total, "correct": correct, "accuracy": correct / total if total > 0 else 0}


def _compute_totals_metrics(games: list[dict]) -> dict:
    t_n = t_abs = t_sq = t_bias = 0
    completed = [g for g in games if g.get("winner") in ("home", "away")]
    for g in completed:
        rp = g.get("runProjection") or {}
        predicted = rp.get("total")
        if isinstance(predicted, (int, float)):
            actual = (g.get("away", {}).get("score") or 0) + (g.get("home", {}).get("score") or 0)
            err = predicted - actual
            t_n += 1; t_abs += abs(err); t_sq += err * err; t_bias += err
    return {"n": t_n, "mae": t_abs / t_n if t_n else 0, "rmse": (t_sq / t_n) ** 0.5 if t_n else 0, "bias": t_bias / t_n if t_n else 0}


def _compute_runline_metrics(games: list[dict]) -> dict:
    preds, labels = [], []
    for g in games:
        if g.get("winner") not in ("home", "away"):
            continue
        rp = g.get("runProjection") or {}
        hrlp = rp.get("homeRunLineProb")
        if not isinstance(hrlp, (int, float)):
            continue
        margin = (g.get("home", {}).get("score") or 0) - (g.get("away", {}).get("score") or 0)
        preds.append(hrlp)
        labels.append(1 if margin >= 2 else 0)
    n = len(preds)
    if n == 0:
        return {"n": 0, "auc": 0, "brier": 0, "accuracy": 0}
    sorted_pairs = sorted(zip(preds, labels), key=lambda x: x[0])
    pos = sum(l for _, l in sorted_pairs)
    neg = n - pos
    auc = 0.0
    if pos > 0 and neg > 0:
        rank_sum = 0.0
        i = 0
        while i < n:
            j = i
            while j < n and sorted_pairs[j][0] == sorted_pairs[i][0]:
                j += 1
            avg_rank = (i + j - 1) / 2 + 1
            for k in range(i, j):
                if sorted_pairs[k][1] == 1:
                    rank_sum += avg_rank
            i = j
        auc = (rank_sum - pos * (pos + 1) / 2) / (pos * neg)
    brier = sum((p - l) ** 2 for p, l in zip(preds, labels)) / n
    correct = sum(1 for p, l in zip(preds, labels) if (1 if p >= 0.5 else 0) == l)
    return {"n": n, "auc": auc, "brier": brier, "accuracy": correct / n}


def main() -> None:
    print("Loading cache files...")
    model_state = cache.load_model_state()
    games = cache.load_games()

    if not model_state:
        print("ERROR: model_state.json not found. Run a refresh first.")
        sys.exit(1)
    if not games:
        print("ERROR: games.json not found. Run a refresh first.")
        sys.exit(1)

    print(f"  model_state: {len(str(model_state))} bytes")
    print(f"  games: {len(games)} games")

    ml_metrics = _compute_moneyline_metrics(games)
    totals_metrics = _compute_totals_metrics(games)
    runline_metrics = _compute_runline_metrics(games)

    # Calibration rows (in-sample + walk-forward)
    cal_rows = cache.load_json("calibration_rows.json", []) or []
    cal_rows_wf = cache.load_json("calibration_rows_wf.json", []) or []
    print(f"  calibration_rows: {len(cal_rows)} rows")
    print(f"  calibration_rows_wf: {len(cal_rows_wf)} rows")

    payload = {
        "modelState": model_state,
        "gamesByDate": _games_by_date(games),
        "gameCards": _game_cards(games),
        "calibrationBins": model_state.get("bins", []),
        "confidenceDistribution": model_state.get("confidenceDistribution", []),
        "calibrationCurve": model_state.get("calibrationCurve", []),
        "totalsMetrics": totals_metrics,
        "runLineMetrics": runline_metrics,
        "moneylineTotal": ml_metrics["total"],
        "moneylineCorrect": ml_metrics["correct"],
        "moneylineAccuracy": ml_metrics["accuracy"],
        "calibrationRows": cal_rows,
        "calibrationRowsWf": cal_rows_wf,
    }

    out_path = cache.CACHE_DIR / "dashboard_payload.json"
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp_path, out_path)

    size_kb = out_path.stat().st_size / 1024
    print(f"\nWrote {out_path} ({size_kb:.0f} KB)")
    print(f"  Moneyline: {ml_metrics['total']} games, {ml_metrics['correct']} correct ({ml_metrics['accuracy']:.1%})")
    print(f"  Totals: {totals_metrics['n']} games, MAE {totals_metrics['mae']:.2f}")
    print(f"  Run Line: {runline_metrics['n']} games, AUC {runline_metrics['auc']:.3f}")
    print(f"\nPush to GitHub:")
    print(f"  git add -f {out_path}")
    print(f"  git commit -m 'Update dashboard payload'")
    print(f"  git push origin main")


if __name__ == "__main__":
    main()
