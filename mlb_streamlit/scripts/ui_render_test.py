"""Targeted regression test for the Streamlit UI chart builders.

Catches the duplicate-keyword crash that broke the deployed app's Calibration
tab once a model was trained: `fig.update_layout(**_base_layout(...), xaxis=...)`
raised `TypeError: got multiple values for keyword argument 'xaxis'` because
`_base_layout()` already returned xaxis/yaxis keys.

Runs with no third-party packages installed by stubbing `streamlit` and
`plotly.graph_objects`. Python itself raises TypeError on duplicate kwargs at
the call site, so the stub needs no special logic to detect the bug.

The app module runs `main()` at import (Streamlit script convention), so this
test execs the source minus the trailing `main()` call to reach the pure chart
builders without executing the UI.

Run:  python3 mlb_streamlit/scripts/ui_render_test.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CHECKS: list[str] = []


def check(name: str) -> None:
    CHECKS.append(name)
    print(f"  ok: {name}")


# --- stub streamlit (only module-level calls are set_page_config/markdown) ---
_st = types.ModuleType("streamlit")
_st.set_page_config = lambda *a, **k: None
_st.markdown = lambda *a, **k: None
_st.session_state = {}
sys.modules["streamlit"] = _st

# --- stub plotly.graph_objects ----------------------------------------------
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

# --- load app.py without running its trailing main() ------------------------
APP_PATH = ROOT / "mlb_streamlit" / "app.py"
_src = APP_PATH.read_text(encoding="utf-8").rstrip()
_tail = "\nmain()"
assert _src.endswith(_tail), f"unexpected app.py tail: ...{_src[-120:]!r}"
_src = _src[: -len(_tail)]

_ns: dict = {"__file__": str(APP_PATH), "__name__": "mlb_streamlit_app_under_test"}
exec(compile(_src, str(APP_PATH), "exec"), _ns)  # noqa: S102 (test harness)
app = types.SimpleNamespace(**{k: v for k, v in _ns.items() if not k.startswith("__")})

# --- the three chart builders that crashed on the deployed app ---------------

curve = [
    {"x": 0.48, "y": 0.46, "n": 14},
    {"x": 0.55, "y": 0.56, "n": 31},
    {"x": 0.65, "y": 0.66, "n": 22},
    {"x": 0.75, "y": 0.78, "n": 18},
    {"x": 0.82, "y": 0.84, "n": 9},
]

fig = app.calibration_scatter(curve, 94)
assert isinstance(fig, Figure), "calibration_scatter returned a Figure"
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
assert fig2.layout["yaxis2"]["side"] == "right"
check("confidence_chart renders (was TypeError: multiple values for xaxis)")

fig3 = app.rolling_brier_chart(
    [{"date": "2026-08-10", "brier": 0.24}, {"date": "2026-08-16", "brier": 0.21}],
    0.26,
)
assert isinstance(fig3, Figure)
assert fig3.layout["yaxis"]["title"] == "Brier"
assert len(fig3.traces) == 2
check("rolling_brier_chart renders (was TypeError: multiple values for xaxis)")

print(f"\nAll {len(CHECKS)} UI render checks passed.")
