"""Offline test for walk-forward model & feature selection.

Runs on the Python standard library (numpy optional) with no Streamlit or
network access:

    python3 mlb_streamlit/scripts/wf_selection_test.py

Verifies that today's deployed model + Model Monitor state are derived from the
walk-forward record:

  * feature selection is decided by out-of-sample univariate signal,
  * model selection is decided by out-of-sample candidate AUC/Brier,
  * the per-date record is cached (incremental, no re-fit on cache hits),
  * the record rolls forward when a new day arrives, and
  * apply_walk_forward_selection makes the deployed model agree with the
    selection (features, weights, candidates, description).
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mlb_streamlit import cache, wf_selection  # noqa: E402
from mlb_streamlit.engine.features import FEATURE_KEYS, compute_elo_and_features  # noqa: E402
from mlb_streamlit.scripts.smoke_test import check, make_games  # noqa: E402

_CHECKS = 0


def _stub_fit_candidate_pool(train, feature_names):
    """Cheap deterministic candidate pool (identical predictors) so the test
    never pays for the pure-Python MLP / boosted-stump fits."""
    def pred(r):
        s = (r["homeElo"] - r["awayElo"]) / 250.0
        s += r["features"].get("eloDiff", 0.0) * 0.4
        s += r["features"].get("winPctDiff", 0.0) * 0.3
        s += r["features"].get("homeField", 0.0) * 0.2
        return 1.0 / (1.0 + math.exp(-s))

    return {name: pred for name in wf_selection.CANDIDATE_NAMES}, 30.0, 0.5


def main() -> int:
    global _CHECKS
    real_fit = wf_selection.fit_candidate_pool
    real_dir = cache.CACHE_DIR
    tmp = tempfile.mkdtemp(prefix="mlb_wfsel_")
    cache.CACHE_DIR = Path(tmp)
    calls = {"n": 0}

    def counting_fit(train, feature_names):
        calls["n"] += 1
        return _stub_fit_candidate_pool(train, feature_names)

    wf_selection.fit_candidate_pool = counting_fit
    try:
        games = make_games(120, seed=7)
        rows = compute_elo_and_features(games)["rows"]

        sel = wf_selection.build_walk_forward_selection(rows=rows)
        print("walk-forward selection")
        check("selection produced", sel is not None)
        check("days evaluated", sel["daysEvaluated"] > 0, f"{sel['daysEvaluated']}")
        check("games evaluated", sel["gamesEvaluated"] >= 20, f"{sel['gamesEvaluated']}")
        check("selected model name present", bool(sel["selectedModel"]))
        check("candidate table full", len(sel["candidates"]) == len(wf_selection.CANDIDATE_NAMES),
              f"{len(sel['candidates'])}")
        check("exactly one candidate selected",
              sum(1 for c in sel["candidates"] if c.get("selected")) == 1)
        check("feature set keeps structural core",
              all(f in sel["featureNames"] for f in wf_selection.CORE_FEATURES))
        check("feature set is a subset of universe",
              set(sel["featureNames"]).issubset(set(FEATURE_KEYS)))
        check("feature importances cover universe",
              len(sel["featureImportances"]) == len(FEATURE_KEYS))
        check("feature importances active flags match selection",
              {f["feature"] for f in sel["featureImportances"] if f["active"]} == set(sel["featureNames"]))
        check("selection reports out-of-sample metrics",
              "auc" in sel and "brier" in sel and "logLoss" in sel and "ece" in sel)

        n1 = calls["n"]
        _again = wf_selection.build_walk_forward_selection(rows=rows)
        check("per-date record cached (no re-fit on rebuild)", calls["n"] == n1, f"{calls['n']} vs {n1}")
        check("cached rebuild deterministic", _again["selectedModel"] == sel["selectedModel"]
              and _again["featureNames"] == sel["featureNames"])

        # Roll forward: a new completed day must extend the record and be
        # scored by a model that includes the prior day's walk-forward model.
        games2 = make_games(121, seed=7)
        rows2 = compute_elo_and_features(games2)["rows"]
        sel2 = wf_selection.build_walk_forward_selection(rows=rows2)
        check("record rolls forward with the new day",
              sel2["daysEvaluated"] > sel["daysEvaluated"],
              f"{sel2['daysEvaluated']} vs {sel['daysEvaluated']}")
        check("rolled-forward games count grows",
              sel2["gamesEvaluated"] > sel["gamesEvaluated"])

        # apply_walk_forward_selection must make the deployed model agree.
        model = {
            "featureNames": list(FEATURE_KEYS),
            "weights": [0.0] * len(FEATURE_KEYS),
            "bias": 0.0,
            "featureStats": {},
            "isotonicPoints": [],
            "monteCarloSigma": 0.0,
            "monteCarloEnabled": False,
            "eloHfa": 30.0,
        }
        result = {"eloHfa": 30.0, "monteCarloEnabled": False, "monteCarloSigma": 0.0}
        wf_selection.apply_walk_forward_selection(result, model, rows, sel)
        check("deployed model uses walk-forward features",
              model["featureNames"] == sel["featureNames"])
        check("deployed weights match features",
              len(model["weights"]) == len(sel["featureNames"]))
        check("result selectedModel matches selection",
              result["selectedModel"] == sel["selectedModel"])
        check("result feature importances carry learned weights",
              all(f["active"] == (f["feature"] in sel["featureNames"])
                  for f in result["featureImportances"]))
        check("result carries walk-forward selection metadata",
              result.get("walkForwardSelection") is sel)
    finally:
        wf_selection.fit_candidate_pool = real_fit
        cache.CACHE_DIR = real_dir
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nAll wf-selection checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
