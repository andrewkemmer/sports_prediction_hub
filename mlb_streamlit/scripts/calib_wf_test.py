"""Tests for the optimized refresh + stacking changes.

Covers:
  1. build_stacking_weights blends genuinely diverse members and collapses
     redundant ones (the greedy forward-selection deadlock fix).
  2. Deployable stack members train on diverse per-family feature subsets.
  3. run_model_lean + decide_monte_carlo produce the complete result shape.
  4. build_walk_forward_calibration_rows_v2 REUSES the walk-forward selection
     record (no run_model_light fit per block) and falls back to fitting when
     the record is missing.

Runs offline on the standard library + numpy:

    python3 mlb_streamlit/scripts/calib_wf_test.py
"""

from __future__ import annotations

import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mlb_streamlit import cache, wf_selection  # noqa: E402
from mlb_streamlit.engine import calib_wf  # noqa: E402
from mlb_streamlit.engine.features import FEATURE_KEYS, compute_elo_and_features  # noqa: E402
from mlb_streamlit.engine.logistic import build_stacking_weights  # noqa: E402
from mlb_streamlit.engine.model import decide_monte_carlo, run_model_lean  # noqa: E402
from mlb_streamlit.engine.stack import fit_stack_members  # noqa: E402
from mlb_streamlit.scripts.nested_selection_test import _make_rows, _stub_fit_walk_forward_step  # noqa: E402
from mlb_streamlit.scripts.smoke_test import check, make_games  # noqa: E402


def test_stacking_blend_diversity() -> None:
    print("greedy stacking blends diverse members, collapses redundant ones")
    labels = [i % 2 for i in range(400)]

    # Two perfectly ANTI-correlated members: each is right on one half and
    # wrong on the other (Brier ~0.41), so their 50/50 blend is a coin flip
    # (Brier 0.25) — the greedy selector must not deadlock on the first pick.
    def _p(cover_even: bool, i: int, y: int) -> float:
        if (i % 2 == 0) == cover_even:
            return 0.9 if y == 1 else 0.1
        return 0.1 if y == 1 else 0.9

    a = [_p(True, i, y) for i, y in enumerate(labels)]
    b = [_p(False, i, y) for i, y in enumerate(labels)]
    redundant = list(a)
    r = build_stacking_weights({"A": a, "B": b, "C": redundant}, labels)
    weights = {w["name"]: w["weight"] for w in r["weights"]}
    check("diverse members both get weight", weights["A"] > 0.2 and weights["B"] > 0.2, f"{weights}")
    check("redundant member gets zero weight", weights["C"] == 0.0, f"{weights}")
    check("blend beats each single member", r["brier"] < 0.30, f"{r['brier']:.4f}")

    # Identical members must collapse to the single best (honest no-blend).
    d = [max(0.01, min(0.99, 0.5 + 0.4 * (1 if y else -1))) for y in labels]
    r2 = build_stacking_weights({"D": d, "E": list(d)}, labels)
    w2 = {w["name"]: w["weight"] for w in r2["weights"]}
    check("redundant pool collapses to one member", w2["D"] == 1.0 and w2["E"] == 0.0, f"{w2}")


def test_stack_member_diversity() -> None:
    print("deployable stack member feature subsets")
    rows = _make_rows(300)
    feats = list(FEATURE_KEYS)
    members = fit_stack_members(rows, feats, mlp_epochs=2)
    full = set(feats)
    core = {"eloDiff", "homeField", "winPctDiff"}
    check("five families fitted", len(members) == 5, f"{len(members)}")
    check("logistic backbone keeps the full feature set",
          set(members["Logistic regression"]["featureNames"]) == full)
    rf_feats = set(members["Random Forest"]["featureNames"])
    mlp_feats = set(members["Neural network (MLP)"]["featureNames"])
    check("RF member trains on a reduced subset", rf_feats < full and len(rf_feats) >= 3,
          f"{len(rf_feats)}/{len(full)}")
    check("family subsets differ", rf_feats != mlp_feats)
    check("structural core retained in every subset",
          all(core <= set(m["featureNames"]) for m in members.values()))


def test_lean_and_mc() -> None:
    print("lean training pass + Monte Carlo decision")
    games = make_games(120, seed=5)
    run = run_model_lean(games, "2026", "2026-09-01")
    result = run["result"]
    for key in ("runModel", "runLineCalibration", "runMarginCalibration", "powerRankings",
                "featureDrift", "rollingBrier", "brierBaseline", "modelVersions",
                "featureImportances", "gamesTrained", "holdoutCount", "eloHfa",
                "monteCarloEnabled", "monteCarloSigma"):
        check(f"lean result has {key}", key in result)
    check("lean rows present", len(run["rows"]) > 0, f"{len(run['rows'])}")
    check("lean model is logistic-only",
          set(run["model"]["stack"]["weights"]) == {"Logistic regression"})
    mc = decide_monte_carlo(run["rows"], run["model"])
    for key in ("sigma", "enabled", "trials", "rationale"):
        check(f"MC decision has {key}", key in mc)
    check("MC deterministic decision", mc["enabled"] in (True, False))


def test_calibration_reuses_selection() -> None:
    print("walk-forward calibration reuses the selection record")
    real_fit = wf_selection.fit_walk_forward_step
    real_light = calib_wf.run_model_light
    real_dir = cache.CACHE_DIR
    real_min_wf = wf_selection.MIN_PRIOR_GAMES
    real_min_cal = calib_wf.MIN_COMPLETED_GAMES
    tmp = tempfile.mkdtemp(prefix="mlb_calib_wf_")
    cache.CACHE_DIR = Path(tmp)
    calls = {"n": 0}

    def counting_light(rows, completed_games, season, as_of_date, feature_names=None,
                       mlp_epochs=40, model_choice=None):
        calls["n"] += 1
        return {
            "season": season, "asOfDate": as_of_date,
            "gamesTrained": len(rows), "holdoutCount": 0,
            "selectedModel": "stub", "modelDescription": "stub",
            "featureNames": list(feature_names or []),
            "weights": [0.0] * len(feature_names or []), "bias": 0.0,
            "featureStats": {f: {"mean": 0.0, "std": 1.0} for f in (feature_names or [])},
            "isotonicPoints": [], "eloHfa": 30,
            "blendW": 0.0, "stack": {"members": {}, "weights": {}},
            "monteCarloEnabled": False, "monteCarloTrials": 0,
            "monteCarloSigma": 0.0, "auc": 0.5, "brier": 0.25,
            "logLoss": 0.69, "ece": 0.0,
            "modelChoice": {"deployed": model_choice or "logistic", "stackBrier": 0.25, "logisticBrier": 0.24},
            "runModel": {"parkFactor": {}, "leagueRuns": 4.5, "teamOffense": {}, "teamDefense": {}},
            "runLineCalibration": [],
            "runMarginCalibration": {"slope": 0.0, "intercept": 0.0},
        }

    wf_selection.fit_walk_forward_step = _stub_fit_walk_forward_step
    calib_wf.run_model_light = counting_light
    wf_selection.MIN_PRIOR_GAMES = 5
    calib_wf.MIN_COMPLETED_GAMES = 5
    try:
        games = make_games(48, seed=13)
        rows = compute_elo_and_features(games)["rows"]
        # Populate the selection record (per-date stored predictions + gate
        # details) with the stub block model.
        wf_selection.build_walk_forward_selection(rows=rows)
        days = wf_selection.load_selection_days()
        check("selection record has stored predictions",
              any(len((d.get("chosenPreds") or [])) > 0 for d in days.values()))
        check("selection record has gate details",
              any(len((d.get("gateDetails") or [])) > 0 for d in days.values()))

        flat = calib_wf.build_walk_forward_calibration_rows_v2(rows=rows)
        check("reused calibration rows produced", len(flat) > 0, f"{len(flat)}")
        check("reuse path does NOT re-fit block models", calls["n"] == 0, f"{calls['n']} fits")
        check("every row records its model family",
              all(r.get("modelChoice") in ("stack", "logistic") for r in flat))
        sample = flat[0]
        for key in ("pickTeam", "pickProb", "homeWinProb", "isCorrect", "gateAccepted",
                    "predictedTotal", "homeRunLineProb", "actualTotal", "actualMargin",
                    "trainedThrough"):
            check(f"row has {key}", key in sample)

        # Remove the selection record -> the fallback path must re-fit.
        cache.save_json("walk_forward_selection.json", {"version": 0, "days": {}})
        before = calls["n"]
        flat2 = calib_wf.build_walk_forward_calibration_rows_v2(rows=rows)
        check("fallback path still produces rows", len(flat2) > 0, f"{len(flat2)}")
        check("fallback path fits block models", calls["n"] > before, f"{calls['n'] - before} fits")
    finally:
        wf_selection.fit_walk_forward_step = real_fit
        calib_wf.run_model_light = real_light
        wf_selection.MIN_PRIOR_GAMES = real_min_wf
        calib_wf.MIN_COMPLETED_GAMES = real_min_cal
        cache.CACHE_DIR = real_dir
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_stacking_blend_diversity()
    test_stack_member_diversity()
    test_lean_and_mc()
    test_calibration_reuses_selection()
    print("\nAll calibration-wf checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
