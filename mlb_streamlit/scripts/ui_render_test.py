"""Targeted regression tests for the Streamlit UI layer.

Catches the two crash classes that broke the deployed app once a model was
trained:

1. Duplicate-keyword crash in the Calibration charts:
   `fig.update_layout(**_base_layout(...), xaxis=...)` raised
   `TypeError: got multiple values for keyword argument 'xaxis'`.
2. `st.segmented_control` returning `None` before the first interaction:
   `_automl_panel` called `sub.startswith(...)` on it and raised
   `AttributeError: 'NoneType' object has no attribute 'startswith'`.

Runs with no third-party packages installed by stubbing `streamlit` and
`plotly.graph_objects`. The `segmented_control` stub reproduces the real
widget's None-on-first-render behavior, so the panel tests fail if the guard
or default is ever removed. The app module runs `main()` at import (Streamlit
script convention), so this test execs the source minus the trailing `main()`
call to reach the pure UI functions without executing the app.

Run:  python3 mlb_streamlit/scripts/ui_render_test.py
"""

from __future__ import annotations

import datetime as _dt
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKS: list[str] = []


def check(name: str) -> None:
    CHECKS.append(name)
    print(f"  ok: {name}")


# ---------------------------------------------------------------------------
# streamlit stub — faithful enough to exercise the panels end to end
# ---------------------------------------------------------------------------


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


class _Ctx:
    def __init__(self, val=None) -> None:
        self.val = val

    def __enter__(self):
        return self.val

    def __exit__(self, *exc) -> bool:
        return False


class _Progress:
    def progress(self, pct, text=None) -> None:
        pass

    def empty(self) -> None:
        pass


_SS = SessionState()
_MARKDOWN: list[str] = []


def _segmented_control(label, options, key=None, default=None):
    if key is not None and key in _SS:
        return _SS[key]
    val = default if default is not None else None  # real widget: None until interaction
    if key is not None:
        _SS[key] = val
    return val


def _columns(spec, **kw):
    if isinstance(spec, int):
        n = spec
    elif isinstance(spec, (list, tuple)):
        n = len(spec)
    else:
        n = 1
    return [_Ctx() for _ in range(n)]


def _date_input(label, key=None, **kw):
    if key is not None and key in _SS:
        return _SS[key]
    val = _dt.date.today()
    if key is not None:
        _SS[key] = val
    return val


def _text_input(label, key=None, **kw):
    if key is not None and key in _SS:
        return _SS[key]
    val = ""
    if key is not None:
        _SS[key] = val
    return val


_st = types.ModuleType("streamlit")
_st.set_page_config = lambda *a, **k: None
_st.markdown = lambda html, **k: _MARKDOWN.append(html)
_st.session_state = _SS
_st.container = lambda *a, **k: _Ctx()
_st.query_params = {}
_st.segmented_control = _segmented_control
_st.columns = _columns
_st.tabs = lambda labels: [_Ctx() for _ in labels]
_st.date_input = _date_input
_st.text_input = _text_input
_st.button = lambda *a, **k: False
_st.progress = lambda *a, **k: _Progress()
_st.info = lambda *a, **k: None
_st.caption = lambda *a, **k: None
_st.error = lambda *a, **k: None
_st.toast = lambda *a, **k: None
_st.plotly_chart = lambda *a, **k: None
_st.rerun = lambda: None
_st.spinner = lambda *a, **k: _Ctx()
_st.warning = lambda *a, **k: None
_st.expander = lambda *a, **k: _Ctx()
sys.modules["streamlit"] = _st

# ---------------------------------------------------------------------------
# plotly.graph_objects stub
# ---------------------------------------------------------------------------
_plotly = types.ModuleType("plotly")
_go = types.ModuleType("plotly.graph_objects")


class Figure:
    def __init__(self) -> None:
        self.traces: list = []
        self.layout: dict = {}

    def add_trace(self, trace) -> None:
        self.traces.append(trace)

    def update_layout(self, **kwargs) -> None:
        self.layout.update(kwargs)


class Scatter:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class Bar:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


_go.Figure = Figure
_go.Scatter = Scatter
_go.Bar = Bar
_plotly.graph_objects = _go
sys.modules["plotly"] = _plotly
sys.modules["plotly.graph_objects"] = _go

# ---------------------------------------------------------------------------
# Load app.py without running its trailing main()
# ---------------------------------------------------------------------------
APP_PATH = ROOT / "mlb_streamlit" / "app.py"
_src = APP_PATH.read_text(encoding="utf-8").rstrip()
_tail = "\nmain()"
assert _src.endswith(_tail), f"unexpected app.py tail: ...{_src[-120:]!r}"
_src = _src[: -len(_tail)]

_ns: dict = {"__file__": str(APP_PATH), "__name__": "mlb_streamlit_app_under_test"}
exec(compile(_src, str(APP_PATH), "exec"), _ns)  # noqa: S102 (test harness)
app = types.SimpleNamespace(**{k: v for k, v in _ns.items() if not k.startswith("__")})


# ---------------------------------------------------------------------------
# Regression 1: the three calibration chart builders
# ---------------------------------------------------------------------------

curve = [
    {"x": 0.48, "y": 0.46, "n": 14},
    {"x": 0.55, "y": 0.56, "n": 31},
    {"x": 0.65, "y": 0.66, "n": 22},
    {"x": 0.75, "y": 0.78, "n": 18},
    {"x": 0.82, "y": 0.84, "n": 9},
]

fig = app.calibration_scatter(curve, 94)
assert isinstance(fig, Figure)
assert fig.layout.get("showlegend") is False
assert "xaxis" in fig.layout and "yaxis" in fig.layout
assert fig.layout["height"] == 300
check("calibration_scatter renders (was TypeError: multiple values for xaxis)")

fig2 = app.confidence_chart([
    {"label": "50-60%", "count": 40, "accuracy": 0.54},
    {"label": "60-70%", "count": 33, "accuracy": 0.66},
    {"label": "70-80%", "count": 21, "accuracy": 0.76},
])
assert isinstance(fig2, Figure)
assert "yaxis2" in fig2.layout
check("confidence_chart renders (was TypeError: multiple values for xaxis)")

fig3 = app.rolling_brier_chart(
    [{"date": "2026-08-10", "brier": 0.24}, {"date": "2026-08-16", "brier": 0.21}],
    0.26,
)
assert isinstance(fig3, Figure)
assert len(fig3.traces) == 2
check("rolling_brier_chart renders (was TypeError: multiple values for xaxis)")


# ---------------------------------------------------------------------------
# Synthetic trained state — mirrors what run_refresh persists
# ---------------------------------------------------------------------------

def _cal_rows(n: int = 36) -> list[dict]:
    rows = []
    rng = 0
    for i in range(n):
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        prob = 0.45 + (rng % 40) / 100.0
        correct = (rng // 7) % 2 == 0
        margin = 1 + (rng % 5)
        rows.append({
            "date": f"2026-08-{(i % 28) + 1:02d}",
            "away": {"abbrev": "LAD", "name": "Los Angeles Dodgers", "score": 3 + i % 5},
            "home": {"abbrev": "SF", "name": "San Francisco Giants", "score": 2 + (i + 1) % 5},
            "pickTeam": "home" if prob >= 0.5 else "away",
            "pickProb": prob,
            "winner": "home" if correct else "away",
            "isCorrect": correct,
            "predictedTotal": 8.5 + (i % 3) * 0.5,
            "actualTotal": 8 + i % 5,
            "homeRunLineProb": 0.45 + (rng % 30) / 100.0,
            "actualMargin": margin if correct else -margin,
        })
    return rows


MS = {
    "asOfDate": "2026-08-16",
    "season": 2026,
    "trainedAt": 1784470800000,  # Aug 16, 2026 11:20 PM ET
    "gamesTrained": 2400,
    "selectedModel": "Logistic regression",
    "auc": 0.638,
    "brier": 0.238,
    "spearmanRho": 0.71,
    "topDecileWinRate": 0.66,
    "monteCarloEnabled": True,
    "monteCarloSigma": 0.08,
    "monteCarloRationale": "Gaussian logit-noise expectation selected by walk-forward CV.",
    "candidates": [
        {"name": "Ridge Logistic", "auc": 0.638, "brier": 0.238, "selected": True},
        {"name": "k-NN", "auc": 0.612, "brier": 0.249, "selected": False},
        {"name": "Naive Bayes", "auc": 0.597, "brier": 0.258, "selected": False},
    ],
    "stackingWeights": [
        {"name": "Ridge Logistic", "weight": 0.7},
        {"name": "k-NN", "weight": 0.3},
        {"name": "Naive Bayes", "weight": 0.0},
    ],
    "featureImportances": [
        {"feature": "eloDiff", "label": "Power ranking gap", "weight": 0.42, "importance": 0.031, "active": True},
        {"feature": "homeField", "label": "Home-field edge", "weight": 0.18, "importance": 0.014, "active": True},
        {"feature": "spEraDiff", "label": "Pitcher ERA (starters)", "weight": 0.24, "importance": 0.019, "active": True},
        {"feature": "restDiff", "label": "Rest days", "weight": 0.09, "importance": 0.007, "active": True},
        {"feature": "formDiff", "label": "Recent form (14d)", "weight": 0.07, "importance": 0.006, "active": True},
        {"feature": "bullpenFatigue", "label": "Bullpen fatigue", "weight": 0.0, "importance": 0.001, "active": False},
        {"feature": "tempDev", "label": "Weather (temp)", "weight": 0.0, "importance": 0.0004, "active": False},
    ],
    "optimizationParams": {
        "featureSelection": "Greedy backward elimination",
        "l2Lambda": 0.05,
        "epochs": 12,
        "hfaGrid": [0.02, 0.04],
        "blendStep": 0.05,
        "mcSigmaGrid": [0.0, 0.05, 0.1],
        "isotonicMethod": "PAV (isotonic regression)",
        "cvFolds": 5,
    },
    "crossValidation": {},
    "featureDrift": [
        {"label": "Power ranking gap", "currentMean": 0.41, "baselineMean": 0.39, "psi": 0.012, "status": "OK"},
        {"label": "Pitcher ERA", "currentMean": 4.02, "baselineMean": 3.98, "psi": 0.021, "status": "WARN"},
    ],
    "rollingBrier": [
        {"date": "2026-08-01", "brier": 0.26},
        {"date": "2026-08-08", "brier": 0.24},
        {"date": "2026-08-16", "brier": 0.22},
    ],
    "brierBaseline": 0.245,
    "modelVersions": [
        {"version": "v1.4.2", "date": "2026-08-16", "auc": 0.638, "brier": 0.238, "notes": "Stacking + isotonic"},
        {"version": "v1.4.1", "date": "2026-08-15", "auc": 0.631, "brier": 0.242, "notes": "MC σ tuned"},
    ],
    "todaysRecord": {"upsets": [], "date": "2026-08-16", "wins": 5, "losses": 2},
    "featureNames": ["Power ranking gap", "Home-field edge", "Pitcher ERA", "Rest days", "Recent form"],
    "powerRankings": [
        {"teamId": 119, "name": "Los Angeles", "abbrev": "LAD", "elo": 1580, "wins": 72, "losses": 44, "winPct": 0.621, "runDiff": 128, "last10WinPct": 0.7, "homeWinPct": 0.655, "awayWinPct": 0.588},
        {"teamId": 137, "name": "San Francisco", "abbrev": "SF", "elo": 1542, "wins": 68, "losses": 48, "winPct": 0.586, "runDiff": 54, "last10WinPct": 0.6, "homeWinPct": 0.619, "awayWinPct": 0.553},
        {"teamId": 147, "name": "New York", "abbrev": "NYY", "elo": 1521, "wins": 65, "losses": 51, "winPct": 0.560, "runDiff": 39, "last10WinPct": 0.5, "homeWinPct": 0.593, "awayWinPct": 0.528},
    ],
}


def _game_docs() -> dict[str, list[dict]]:
    """Two synthetic game docs (one final, one preview) for the games tab."""
    return {
        "2026-08-16": [
            {
                "gamePk": 750001,
                "status": "Final",
                "dayNight": "night",
                "gameDate": "2026-08-16T19:05:00Z",
                "innings": 9,
                "away": {"id": 119, "abbrev": "LAD", "name": "Los Angeles", "score": 5, "wins": 72, "losses": 44},
                "home": {"id": 137, "abbrev": "SF", "name": "San Francisco", "score": 3, "wins": 68, "losses": 48},
                "awayWinProb": 0.39, "homeWinProb": 0.61, "pickProb": 0.61,
                "pickTeam": "home", "winner": "home", "isCorrect": True, "isUpset": False,
                "homePitcher": {"name": "Logan Webb", "era": 3.21, "k9": 8.4},
                "awayPitcher": {"name": "Yoshinobu Yamamoto", "era": 2.98, "k9": 10.1},
                "runProjection": {"homeScore": 4.6, "awayScore": 3.9, "total": 8.5, "overProb": 0.48, "underProb": 0.52, "homeRunLineProb": 0.44, "awayRunLineProb": 0.56},
                "venue": "Oracle Park", "weather": {"tempF": 68, "windMph": 9},
                "homeInjuries": 1, "awayInjuries": 0,
                "fairHomeOdds": -156, "fairAwayOdds": 136, "edge": 0.02,
                "marketOdds": {"homeMoneyline": -165, "awayMoneyline": 145, "total": 8.0, "overPrice": -110, "underPrice": -110, "runLine": 1.5, "homeRunLinePrice": 120, "awayRunLinePrice": -140},
                "lineups": {"home": {"battingOrder": [{"name": "Posey", "pos": "C"}], "bench": []}, "away": {"battingOrder": [{"name": "Ohtani", "pos": "DH"}], "bench": []}},
                "shap": [{"feature": "eloDiff", "label": "Power ranking gap", "value": 0.4, "contribution": 0.12}],
            },
            {
                "gamePk": 750002,
                "status": "Preview",
                "dayNight": "day",
                "gameDate": "2026-08-17T13:05:00Z",
                "away": {"id": 121, "abbrev": "NYM", "name": "New York", "wins": 60, "losses": 56},
                "home": {"id": 147, "abbrev": "NYY", "name": "New York", "wins": 65, "losses": 51},
                "awayWinProb": 0.53, "homeWinProb": 0.47, "pickProb": 0.53,
                "pickTeam": "away",
                "homePitcher": {"name": "Carlos Rodón", "era": 3.55, "k9": 9.2},
                "awayPitcher": {"name": "Kodai Senga", "era": 3.31, "k9": 10.8},
                "runProjection": {"homeScore": 4.1, "awayScore": 4.4, "total": 8.5, "overProb": 0.51, "underProb": 0.49, "homeRunLineProb": 0.46, "awayRunLineProb": 0.54},
                "venue": "Yankee Stadium", "weather": {"tempF": 82, "windMph": 6},
                "homeInjuries": 2, "awayInjuries": 1,
                "fairHomeOdds": 112, "fairAwayOdds": -132, "edge": -0.01,
                "lineups": {},
                "shap": [],
            },
        ]
    }


BUNDLE = {"model_state": MS, "calibration_rows": _cal_rows(), "docs_by_date": _game_docs()}

# ---------------------------------------------------------------------------
# Regression 2: panels with segmented_control returning None (fresh session)
# ---------------------------------------------------------------------------

_MARKDOWN.clear()
_SS.clear()
app._automl_panel(MS)
assert _SS.get("automl_sub") is not None, "automl_sub should default to an option"
assert any("Empirical Feature Selection" in h for h in _MARKDOWN), "Auto-ML panel body rendered"
check("_automl_panel renders on fresh session (was AttributeError: None.startswith)")

_MARKDOWN.clear()
_SS.clear()
app.monitor_tab(BUNDLE)
assert _SS.get("mon_sub") == "Auto-ML Selection & Weights", "mon_sub should default to Auto-ML"
assert any("Drift Monitor" in h for h in _MARKDOWN), "monitor body rendered"  # title is HTML-escaped: &amp;
assert any("Feature Drift Analysis" in h for h in _MARKDOWN), "drift table rendered"
check("monitor_tab renders on fresh session (no crash, correct default panel)")

_MARKDOWN.clear()
_SS.clear()
app.calibration_tab(BUNDLE)
assert _SS.get("cal_view") == "Moneyline", "cal_view should default to Moneyline"
assert any("Model Calibration Dashboard" in h for h in _MARKDOWN), "calibration header rendered"
assert any("Calibration Curve" in h for h in _MARKDOWN), "calibration curve section rendered"
check("calibration_tab renders Moneyline view on fresh session (no crash)")

# ---------------------------------------------------------------------------
# Regression 3: layout parity — header nav, games tab, power rankings
# ---------------------------------------------------------------------------

_MARKDOWN.clear()
_SS.clear()
app.render_header("games")
joined = "\n".join(_MARKDOWN)
for label in ("MLB Predictions", "Today's Games", "Power Rankings", "Calibration", "Model Monitor", "Refresh"):
    assert label in joined, f"header missing {label!r}"
check("render_header renders sticky nav + refresh link (no crash)")

_MARKDOWN.clear()
_SS.clear()
_SS["games_date"] = _dt.date(2026, 8, 16)
_SS["games_filter"] = "All Games (2)"
_SS["bundle"] = BUNDLE
app.games_tab(BUNDLE)
joined = "\n".join(_MARKDOWN)
assert _SS.get("games_filter", "").startswith("All Games"), "games filter should default to All Games"
assert "2 of 2 games shown" in joined, "summary chips row rendered"
assert "Predicted score" in joined, "game card run projection rendered"
assert "Final" in joined, "final-status pill rendered on completed game"
assert "LAD" in joined and "NYM" in joined, "both game cards rendered"
check("games_tab renders summary row + date selector + game cards (no crash)")

_MARKDOWN.clear()
_SS.clear()
app.rankings_tab(BUNDLE)
joined = "\n".join(_MARKDOWN)
assert "Power Rankings" in joined, "rankings header rendered"
assert "LAD" in joined and "SF" in joined, "ranking rows rendered"
assert "Run Diff" in joined, "rankings table columns rendered"
# Regression: the run-diff cell used to carry a stray quote that closed the
# style attribute before the value, so +128/-54 never displayed.
assert "+128" in joined, "run-diff value renders inside its span"
check("rankings_tab renders Elo power rankings table (no crash)")

print(f"\nAll {len(CHECKS)} UI render checks passed.")
