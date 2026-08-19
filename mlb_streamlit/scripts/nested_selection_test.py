"""Tests for the nested (per-date) walk-forward selection.

Covers the L1 feature selector, the stability vote, the per-date stack-vs-\nlogistic model choice, and that every dashboard's per-date model uses its own\nprior-only features + family. Runs offline on the standard library + numpy.

    python3 mlb_streamlit/scripts/nested_selection_test.py
"""

from __future__ import annotations

import math
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mlb_streamlit import cache, refresh, wf_selection  # noqa: E402
from mlb_streamlit.engine import model as emod  # noqa: E402
from mlb_streamlit.engine.features import FEATURE_KEYS, compute_elo_and_features  # noqa: E402
from mlb_streamlit.engine.logistic import _soft_threshold, train_logistic_l1  # noqa: E402
from mlb_streamlit.engine.model import fit_deployable_model  # noqa: E402
from mlb_streamlit.scripts.smoke_test import check, make_games  # noqa: E402


def _make_rows(n: int, seed: int = 11) -> list[dict]:
    random.seed(seed)
    feats = list(FEATURE_KEYS)
    rows = []
    for _ in range(n):
        f = {k: random.uniform(-2, 2) for k in feats}
        s = 0.8 * f["eloDiff"] + 0.6 * f["winPctDiff"] - 0.4 * f["homeField"] + 0.2 * f["spFipDiff"]
        p = 1 / (1 + math.exp(-s))
        rows.append(
            {"label": 1 if random.random() < p else 0, "features": f, "homeElo": 1500.0, "awayElo": 1500.0}
        )
    return rows


def _stub_fit_walk_forward_step(train, feature_names, mlp_epochs=40):
    """Return a trivial serializable per-date model (logistic chosen)."""
    model = {
        "featureNames": list(feature_names),
        "weights": [0.0] * len(feature_names),
        "bias": 0.0,
        "featureStats": {f: {"mean": 0.0, "std": 1.0} for f in feature_names},
        "isotonicPoints": [],
        "eloHfa": 30.0,
        "blendW": 0.0,
        "stack": {"members": {}, "weights": {}},
        "candidateMembers": {},
        "monteCarloSigma": 0.0,
        "monteCarloEnabled": False,
    }
    choice = {"deployed": "logistic", "stackBrier": 0.25, "logisticBrier": 0.24}
    return model, choice


def test_l1_selector() -> None:
    print("l1 selector")
    rows = _make_rows(1500)
    m = train_logistic_l1(rows, list(FEATURE_KEYS), lambda_l1=0.05)
    nz = [f for f, w in zip(m["featureNames"], m["weights"]) if abs(w) > 1e-9]
    check("L1 returns same dict shape as train_logistic",
          {"featureNames", "weights", "bias", "featureStats"} <= set(m))
    check("L1 produces sparse weights", 0 < len(nz) < len(FEATURE_KEYS), f"{len(nz)}/{len(FEATURE_KEYS)}")
    check("L1 keeps the strong-signal features",
          {"eloDiff", "winPctDiff", "homeField"} <= set(nz))
    check("L1 dropped features have exactly 0.0 weight",
          all(abs(w) < 1e-9 for f, w in zip(m["featureNames"], m["weights"]) if f not in nz))
    # soft-threshold correctness
    check("soft threshold positive", _soft_threshold(0.5, 0.2) == 0.3)
    check("soft threshold negative", _soft_threshold(-0.5, 0.2) == -0.3)
    check("soft threshold zeroes small", _soft_threshold(0.1, 0.2) == 0.0)


def test_stability() -> None:
    print("stability vote")
    # A feature selected in 2 of the last 3 blocks survives even when the
    # current block's L1 dropped it; a one-off flicker does not persist.
    core = set(wf_selection.CORE_FEATURES)
    current = ["eloDiff", "homeField", "winPctDiff", "spFipDiff"]
    history = [["eloDiff", "homeField", "winPctDiff", "spEraDiff"],
               ["eloDiff", "homeField", "winPctDiff", "spEraDiff", "opsDiff"]]
    stable = set(wf_selection._stabilize_features(current, history))
    check("stability keeps current selections", {"spFipDiff"} <= stable)
    check("stability keeps persistent feature", {"spEraDiff"} <= stable)
    check("stability drops one-off flicker", "opsDiff" not in stable)
    check("stability always keeps core", core <= stable)


def test_early_core_fallback() -> None:
    print("early-season core fallback")
    rows = _make_rows(50)  # below L1_MIN_ROWS
    sel = wf_selection._l1_selected_features(rows, list(FEATURE_KEYS))
    check("early season selects only core features",
          set(sel) == set(wf_selection.CORE_FEATURES), f"{sel}")


def test_deployable_choice() -> None:
    print("deployable stack vs logistic")
    rows = _make_rows(900)
    m, c = fit_deployable_model(rows, list(FEATURE_KEYS)[:12])
    check("auto choice is stack or logistic", c["deployed"] in ("stack", "logistic"), c["deployed"])
    check("choice carries both Briers", "stackBrier" in c and "logisticBrier" in c)
    check("deployed stack is serializable",
          set(m["stack"]["members"]) == {"Logistic regression"} or len(m["stack"]["members"]) > 1)
    m2, c2 = fit_deployable_model(rows, list(FEATURE_KEYS)[:12], model_choice="logistic")
    check("forced logistic deploys logistic-only stack",
          c2["deployed"] == "logistic"
          and set(m2["stack"]["weights"]) == {"Logistic regression"})
    m3, c3 = fit_deployable_model(rows, list(FEATURE_KEYS)[:12], model_choice="stack")
    check("forced stack respected", c3["deployed"] == "stack")


def test_walk_forward_records_features() -> None:
    print("walk-forward per-date features")
    real_fit = wf_selection.fit_walk_forward_step
    real_dir = cache.CACHE_DIR
    tmp = tempfile.mkdtemp(prefix="mlb_nested_")
    cache.CACHE_DIR = Path(tmp)
    calls = {"n": 0}

    def counting_fit(train, feature_names, mlp_epochs=40):
        calls["n"] += 1
        return _stub_fit_walk_forward_step(train, feature_names, mlp_epochs=mlp_epochs)

    wf_selection.fit_walk_forward_step = counting_fit
    try:
        games = make_games(120, seed=7)
        rows = compute_elo_and_features(games)["rows"]
        sel = wf_selection.build_walk_forward_selection(rows=rows)
        check("selection produced", sel is not None)
        check("final features include core", set(wf_selection.CORE_FEATURES) <= set(sel["featureNames"]))
        check("final features are a real subset", set(sel["featureNames"]) <= set(FEATURE_KEYS))
        days = wf_selection.load_selection_days()
        check("selection record has per-date features",
              any(d.get("features") for d in days.values()))
        check("final features match last recorded date",
              sel["featureNames"] == list(days[sorted(days)[-1]].get("features")))
    finally:
        wf_selection.fit_walk_forward_step = real_fit
        cache.CACHE_DIR = real_dir
        shutil.rmtree(tmp, ignore_errors=True)


def test_calibration_records_choice() -> None:
    print("calibration per-date model choice")
    real_light = refresh.run_model_light
    real_dir = cache.CACHE_DIR
    real_min = refresh.MIN_COMPLETED_GAMES
    real_sim = emod.simulate_runs_batch
    tmp = tempfile.mkdtemp(prefix="mlb_nested_cal_")
    cache.CACHE_DIR = Path(tmp)

    def stub_light(rows, completed_games, season, as_of_date, feature_names=None, mlp_epochs=40, model_choice=None):
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
            "runModel": {"parkFactor": {}, "leagueRuns": 4.5,
                         "teamOffense": {}, "teamDefense": {}},
            "runLineCalibration": [],
            "runMarginCalibration": {"slope": 0.0, "intercept": 0.0},
        }

    def fast_sim(state, home_ids, away_ids, totals, margins, trials):
        return [{"total": 8.5, "homeRunLineProb": 0.52} for _ in home_ids]

    refresh.run_model_light = stub_light
    refresh.MIN_COMPLETED_GAMES = 5
    emod.simulate_runs_batch = fast_sim
    try:
        games = make_games(48, seed=13)
        rows = compute_elo_and_features(games)["rows"]
        # Populate per-date features in the selection record first.
        real_fit = wf_selection.fit_walk_forward_step
        wf_selection.fit_walk_forward_step = _stub_fit_walk_forward_step
        try:
            wf_selection.build_walk_forward_selection(rows=rows)
        finally:
            wf_selection.fit_walk_forward_step = real_fit
        cal = refresh.build_walk_forward_calibration_rows(rows=rows)
        check("calibration rows produced", len(cal) > 0, f"{len(cal)}")
        check("every row records its model family",
              all(r.get("modelChoice") in ("stack", "logistic") for r in cal),
              f"{set(r.get('modelChoice') for r in cal)}")
        days = wf_selection.load_selection_days()
        check("selection record carries per-date model choice",
              any(d.get("modelChoice") in ("stack", "logistic") for d in days.values()))
    finally:
        refresh.run_model_light = real_light
        refresh.MIN_COMPLETED_GAMES = real_min
        emod.simulate_runs_batch = real_sim
        cache.CACHE_DIR = real_dir
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_l1_selector()
    test_stability()
    test_early_core_fallback()
    test_deployable_choice()
    test_walk_forward_records_features()
    test_calibration_records_choice()
    print("\nAll nested-selection checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
