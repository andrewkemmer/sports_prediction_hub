"""MLB Predictions — Streamlit dashboard.

Self-contained Python migration of the React/Convex dashboard:
  1. Today's Games — date selector, filter pills, game cards with model
     predictions, run projections, lineups, SHAP contributions.
  2. Power Rankings — Elo-based team table.
  3. Calibration — moneyline / totals / run-line reliability views over a
     selectable date range, with a full game-history table.
  4. Model Monitor — Auto-ML selection, feature importance, ensemble
     architecture, drift monitoring, rolling Brier and version history.

A single Refresh button re-pulls data from the MLB Stats API, retrains and
recalibrates the model, and re-scores the upcoming window.

Run:  pip install -r mlb_streamlit/requirements.txt
      streamlit run mlb_streamlit/app.py
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

from mlb_streamlit import cache, ui
from mlb_streamlit.data import et_date_string, market_odds_enabled
from mlb_streamlit.engine.metrics import (
    calibration_curve_points,
    compute_auc,
    compute_brier,
    evaluate,
)
from mlb_streamlit.engine.teams import team_meta
from mlb_streamlit.refresh import (
    PREDICTION_VERSION,
    SEASON_START,
    build_walk_forward_calibration_rows,
    load_bundle,
    power_rankings_as_of,
    predict_date,
    run_refresh,
)

# ---------------------------------------------------------------------------
# Page setup + theme CSS
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MLB Predictions — 2026", page_icon="⚾", layout="wide")

_CSS = """
<style>
.stApp { background: #0a0d12; }
[data-testid="stHeader"] { background: rgba(10,13,18,0.8); backdrop-filter: blur(8px); border-bottom: 1px solid rgba(255,255,255,0.07); }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
.block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1280px; }
div[data-testid="stExpander"] details { border: 1px solid rgba(255,255,255,0.09); border-radius: 12px; background: #12161c; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color: rgba(255,255,255,0.09) !important; border-radius: 16px; }
.stButton > button[kind="primary"] { background: #427ff7; border-radius: 8px; font-weight: 600; }
/* Sticky app header — the first row of columns (React-style sticky nav) */
[data-testid="stHorizontalBlock"]:first-of-type {
  position: sticky !important;
  top: 0;
  z-index: 100;
  background: rgba(10,13,18,0.85);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid rgba(255,255,255,0.07);
  padding: 10px 2px;
  margin-bottom: 16px;
}
/* Header nav tabs render as text links with an underline bar when active (React-style) */
div[class*="st-key-nav_"] { display: flex; justify-content: center; }
div[class*="st-key-nav_"] button {
  background: transparent !important;
  border: none !important;
  border-radius: 8px;
  color: #8b939f;
  font-size: 14px;
  font-weight: 500;
  padding: 6px 10px;
  box-shadow: none !important;
}
div[class*="st-key-nav_"] button:hover { color: #e5e8ec !important; background: rgba(255,255,255,0.04) !important; }
div[class*="st-key-nav_active_"] button {
  color: #e5e8ec !important;
  border-bottom: 2px solid #427ff7 !important;
  border-radius: 0 !important;
}
/* Header refresh button (React-style bordered pill) */
div[class*="st-key-header_refresh"] { display: flex; justify-content: flex-end; }
div[class*="st-key-header_refresh"] button {
  border: 1px solid rgba(255,255,255,0.09) !important;
  background: #12161c !important;
  border-radius: 8px;
  color: #e5e8ec;
  font-size: 12px;
  font-weight: 500;
  padding: 6px 12px;
  box-shadow: none !important;
}
div[class*="st-key-header_refresh"] button:hover { border-color: rgba(255,255,255,0.18) !important; }
/* Prev / next day square buttons (React-style) */
div[class*="st-key-date_prev"], div[class*="st-key-date_next"] { display: flex; justify-content: center; }
div[class*="st-key-date_prev"] button, div[class*="st-key-date_next"] button {
  width: 32px !important;
  min-width: 32px !important;
  height: 32px !important;
  min-height: 32px !important;
  padding: 0 !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  background: #12161c !important;
  border-radius: 8px !important;
  color: #8b939f !important;
  font-size: 16px !important;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-shadow: none !important;
}
div[class*="st-key-date_prev"] button:hover, div[class*="st-key-date_next"] button:hover { color: #e5e8ec !important; border-color: rgba(255,255,255,0.2) !important; }
/* Date pill opens a calendar popover (React-style blue-tinted rounded pill) */
button[data-testid="stPopoverButton"] {
  border-radius: 999px;
  border: 1px solid rgba(59,130,246,0.30);
  background: rgba(59,130,246,0.10);
  color: #e5e8ec;
  font-weight: 500;
  font-size: 14px;
  padding: 8px 20px;
  white-space: nowrap;
  box-shadow: none !important;
}
button[data-testid="stPopoverButton"]:hover { border-color: rgba(59,130,246,0.50) !important; }
/* Date inputs render as rounded pills (React-style, blue-tinted like the calendar trigger) */
div[data-testid="stDateInput"] > label { display: none; }
div[data-testid="stDateInput"] [data-baseweb="input"] { border-radius: 9999px; }
div[data-testid="stDateInput"] input { border-radius: 9999px; border: 1px solid rgba(59,130,246,0.30); background: rgba(59,130,246,0.10); color: #e5e8ec; font-weight: 500; font-size: 14px; }
div[data-testid="stDateInput"] input:hover { border-color: rgba(59,130,246,0.50); }
/* Segmented controls render as pill toggles (React-style) */
div[data-testid="stSegmentedControl"] > label { display: none; }
div[data-testid="stSegmentedControl"] { gap: 6px; }
div[data-testid="stSegmentedControl"] button { border-radius: 9999px; font-size: 12px; font-weight: 600; border: 1px solid rgba(255,255,255,0.09); background: #12161c; color: #8b939f; }
div[data-testid="stSegmentedControl"] button:hover { color: #e5e8ec; }
div[data-testid="stSegmentedControl"] button[aria-checked="true"], div[data-testid="stSegmentedControl"] button[aria-selected="true"] { background: #427ff7 !important; border-color: #427ff7 !important; color: #fff !important; }
/* Model Monitor tab uses cyan accents in the React app (Run Auto-ML button + sub-tab toggles) */
div[class*="st-key-mon_sub"] [data-testid="stSegmentedControl"] button[aria-checked="true"],
div[class*="st-key-mon_sub"] [data-testid="stSegmentedControl"] button[aria-selected="true"],
div[class*="st-key-automl_sub"] [data-testid="stSegmentedControl"] button[aria-checked="true"],
div[class*="st-key-automl_sub"] [data-testid="stSegmentedControl"] button[aria-selected="true"] { background: #22d3ee !important; border-color: #22d3ee !important; color: #083344 !important; }
div[class*="st-key-automl_btn"] button[kind="primary"] { background: #22d3ee; border-color: #22d3ee; color: #083344; }
div[class*="st-key-automl_btn"] button[kind="primary"]:hover { background: #22d3ee; color: #083344; opacity: .9; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

ET = _dt.timezone(_dt.timedelta(hours=-4))


def _et_zone(now: _dt.datetime) -> _dt.tzinfo:
    return _dt.timezone(_dt.timedelta(hours=-4 if 4 <= now.month <= 10 else -5))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_pct(p, digits=0) -> str:
    try:
        return f"{float(p) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_prob(p, digits=3) -> str:
    try:
        return f"{float(p):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_american(odds) -> str:
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return "—"
    return f"+{odds:.0f}" if odds > 0 else f"{odds:.0f}"


def fmt_signed(v, digits=2) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{digits}f}"


def short_date(ymd: str) -> str:
    try:
        return _dt.date.fromisoformat(ymd).strftime("%b %d")
    except ValueError:
        return ymd


def fmt_date_short(ymd: str) -> str:
    try:
        return _dt.date.fromisoformat(ymd).strftime("%a, %b %d")
    except ValueError:
        return ymd


def fmt_date_long(ymd: str) -> str:
    try:
        return _dt.date.fromisoformat(ymd).strftime("%A, %B %d, %Y")
    except ValueError:
        return ymd


def fmt_trained_at(ms: int) -> str:
    try:
        d = _dt.datetime.fromtimestamp(ms / 1000).astimezone(_et_zone(_dt.datetime.now()))
        return d.strftime("%b %d, %I:%M %p ET").lstrip("0")
    except (TypeError, ValueError, OSError):
        return "—"


def fmt_time_et(iso: str) -> str:
    try:
        d = _dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return d.astimezone(ET).strftime("%-I:%M %p")
    except (TypeError, ValueError):
        return "TBD"


def fmt_number(n) -> str:
    try:
        return f"{int(round(float(n))):,}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Charts (plotly)
# ---------------------------------------------------------------------------

def _base_layout(height: int = 300) -> dict:
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ui.TEXT, size=12),
        margin=dict(l=10, r=14, t=10, b=10),
        # NOTE: do not add xaxis/yaxis here — callers always pass their own
        # axis config, and `update_layout(**_base_layout(...), xaxis=...)`
        # would raise TypeError: got multiple values for keyword argument.
        hoverlabel=dict(bgcolor="#161b22", bordercolor="rgba(255,255,255,0.1)", font=dict(color=ui.TEXT)),
    )


def calibration_scatter(curve: list[dict], total: int) -> go.Figure:
    fig = go.Figure()
    xs = [p["x"] for p in curve]
    ys = [p["y"] for p in curve]
    fig.add_trace(go.Scatter(
        x=[0.45, 0.85], y=[0.45, 0.85], mode="lines",
        line=dict(color=ui.MUTED, dash="dash", width=1), name="Perfect calibration",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(color=ui.ACCENT, size=9, line=dict(color="rgba(255,255,255,0.35)", width=1)),
        customdata=[[p["n"]] for p in curve],
        hovertemplate="Predicted %{x:.3f}<br>Actual %{y:.3f}<br>n=%{customdata[0]}<extra></extra>",
        name=f"Model (n={total})",
    ))
    fig.update_layout(
        **_base_layout(300),
        xaxis=dict(title="Mean Predicted Probability", range=[0.45, 0.85], tickvals=[0.48, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82], tickformat=".2f", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Mean Actual Win Rate", range=[0.45, 0.85], tickvals=[0.48, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82], tickformat=".2f", gridcolor="rgba(255,255,255,0.06)"),
        showlegend=False,
    )
    return fig


def confidence_chart(distribution: list[dict]) -> go.Figure:
    fig = go.Figure()
    labels = [d["label"] for d in distribution]
    counts = [d["count"] for d in distribution]
    accs = [d["accuracy"] for d in distribution]
    fig.add_trace(go.Bar(
        y=counts, x=labels, name="Game Count", yaxis="y",
        marker_color="rgba(77,125,255,0.45)", marker_line_width=0,
    ))
    fig.add_trace(go.Scatter(
        y=accs, x=labels, name="Accuracy", yaxis="y2", mode="lines+markers",
        line=dict(color=ui.EMERALD, width=2), marker=dict(size=5, color=ui.EMERALD),
        hovertemplate="%{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(300),
        xaxis=dict(title="", tickangle=-30),
        yaxis=dict(title="Game count", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis2=dict(title="Accuracy %", overlaying="y", side="right", range=[0.45, 0.8],
                    tickformat=".0%", showgrid=False, tickfont=dict(color=ui.MUTED)),
        showlegend=False,
    )
    return fig


def rolling_brier_chart(points: list[dict], baseline: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=[baseline] * max(1, len(points)), x=[p["date"] for p in points] or [""],
        mode="lines", line=dict(color=ui.MUTED, dash="dash", width=1), name="Baseline (prior version)",
    ))
    fig.add_trace(go.Scatter(
        x=[p["date"] for p in points], y=[p["brier"] for p in points],
        mode="lines+markers", line=dict(color=ui.ORANGE, width=2),
        marker=dict(size=4, color=ui.ORANGE),
        hovertemplate="%{x|%b %d}<br>Brier %{y:.3f}<extra></extra>",
        name="Brier Score",
    ))
    fig.update_layout(
        **_base_layout(300),
        xaxis=dict(title=""),
        yaxis=dict(title="Brier", gridcolor="rgba(255,255,255,0.06)"),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

def do_refresh() -> None:
    """Run the full refresh pipeline with a live progress bar."""
    bar = st.progress(4, text="Starting refresh…")

    def report(stage: str, pct: int, message: str) -> None:
        bar.progress(min(100, max(0, int(pct))), text=f"{stage} — {message}")

    try:
        summary = run_refresh(report)
    except Exception as e:  # noqa: BLE001
        bar.empty()
        st.error(f"Refresh failed: {e}")
        return
    st.session_state.bundle = load_bundle()
    st.session_state.pop("requested_dates", None)
    bar.progress(100, text="Complete")
    st.toast(
        f"Model refreshed · {fmt_number(summary['gamesTrained'])} games trained · "
        f"AUC {summary['auc']:.3f} · Brier {summary['brier']:.3f}"
    )
    st.rerun()


# ---------------------------------------------------------------------------
# Header + empty state
# ---------------------------------------------------------------------------

TAB_IDS: list[tuple[str, str]] = [
    ("games", "Today's Games"),
    ("rankings", "Power Rankings"),
    ("calibration", "Calibration"),
    ("monitor", "Model Monitor"),
]

def render_header(active_tab: str) -> None:
    """Sticky header matching the React dashboard: mark + title, nav tabs, refresh.

    Navigation uses in-app buttons + session state. Streamlit rewrites every
    markdown anchor so it opens in a new tab, so the React-style nav cannot use
    <a> links — clicking a header must switch tabs in place.
    """
    cols = st.columns([2.2, 1.7, 1.6, 1.25, 1.6, 1.15], vertical_alignment="center")
    with cols[0]:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;'>"
            f"{ui.baseball_mark()}"
            f"<span style='font-size:16px;font-weight:700;color:{ui.TEXT};letter-spacing:-.01em;white-space:nowrap;'>MLB Predictions</span></div>",
            unsafe_allow_html=True,
        )
    for col, tid, label in (
        (cols[1], "games", "Today's Games"),
        (cols[2], "rankings", "Power Rankings"),
        (cols[3], "calibration", "Calibration"),
        (cols[4], "monitor", "Model Monitor"),
    ):
        with col:
            key = f"nav_active_{tid}" if tid == active_tab else f"nav_{tid}"
            if st.button(label, key=key):
                st.session_state.active_tab = tid
                st.rerun()
    with cols[5]:
        if st.button("⟳ Refresh", key="header_refresh"):
            st.session_state.refresh_requested = True
            st.rerun()


def render_empty_state() -> None:
    st.markdown("<div style='height:12vh'></div>", unsafe_allow_html=True)
    c = st.columns([1, 2, 1])[1]
    with c:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='display:inline-flex;width:56px;height:56px;border-radius:16px;align-items:center;justify-content:center;"
            f"border:1px solid {ui.BORDER};background:{ui._card_bg()};font-size:26px;'>🔄</div>"
            f"<h2 style='margin-top:18px;font-size:20px;font-weight:700;color:{ui.TEXT}'>Train your prediction model</h2>"
            f"<p style='margin:10px auto 0;max-width:440px;font-size:13px;color:{ui.MUTED};line-height:1.6'>"
            f"Pull every 2026 regular-season game from the MLB Stats API, fit and calibrate the model, "
            f"and generate win probabilities for the rest of the season.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh & train model", type="primary", use_container_width=True, key="train_btn"):
            do_refresh()


# ---------------------------------------------------------------------------
# Tab 1 — Today's Games
# ---------------------------------------------------------------------------

def _games_for_date(bundle, ymd: str) -> list[dict]:
    return bundle["docs_by_date"].get(ymd, []) or []


def _filter_games(games: list[dict], f: str) -> list[dict]:
    if f == "Final":
        return [g for g in games if g["status"] == "Final"]
    if f == "Live":
        return [g for g in games if g["status"] == "Live"]
    if f == "Upcoming":
        return [g for g in games if g["status"] in ("Preview", "Scheduled")]
    return games


def _doc_is_point_in_time(doc: dict, ymd: str, today: str) -> bool:
    """A doc for date D is only displayable when the model that scored it was
    trained exactly through D (walk-forward) — or through today for D itself.
    Anything else (legacy 99% docs, fresh-window in-sample writes, older
    pipeline versions) is never shown."""
    expected = ymd if ymd < today else today
    return bool(doc) and doc.get("predictionVersion") == PREDICTION_VERSION \
        and doc.get("trainedThrough") == expected


def _ensure_games_for_date(bundle) -> None:
    """On-demand prediction for the selected date (port of predictDate).

    Every doc shown for a date must be point-in-time: scored by a model
    trained only on games before that date (walk-forward), or by the deployed
    model for today. Docs that fail the contract — older pipeline versions,
    in-sample fresh-window writes, stale 99% results — are re-scored on view.
    """
    ms = bundle["model_state"]
    today = ms.get("asOfDate") or et_date_string()
    ymd = st.session_state.games_date.isoformat()
    cached_docs = bundle["docs_by_date"].get(ymd)
    if cached_docs and all(_doc_is_point_in_time(d, ymd, today) for d in cached_docs):
        return
    if "requested_dates" not in st.session_state:
        st.session_state.requested_dates = set()
    if ymd in st.session_state.requested_dates:
        return
    st.session_state.requested_dates.add(ymd)
    progress = st.progress(0, text=f"Building predictions for {fmt_date_long(ymd)}…")
    try:
        predict_date(
            ymd,
            ms,
            report=lambda stage, pct, msg: progress.progress(max(0, min(int(pct), 99)), text=msg or stage),
        )
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not load predictions for {ymd}: {e}")
        st.session_state.requested_dates.discard(ymd)
        return
    finally:
        progress.empty()
    st.session_state.bundle = load_bundle()
    st.rerun()


def _game_card(game: dict, col) -> None:
    home_color = team_meta(game["home"]["id"])["color"]
    away_color = team_meta(game["away"]["id"])["color"]
    is_final = game["status"] == "Final"
    is_live = game["status"] == "Live"
    is_coin_flip = game.get("pickProb", 0) <= 0.54
    winner_abbrev = (
        game["home"]["abbrev"] if game.get("winner") == "home"
        else game["away"]["abbrev"] if game.get("winner") == "away" else None
    )
    pick_abbrev = game["home"]["abbrev"] if game.get("pickTeam") == "home" else game["away"]["abbrev"]

    pills = []
    if is_coin_flip:
        pills.append(ui.pill("Coin flip", ui.AMBER, "rgba(252,211,77,0.15)"))
    if game.get("isUpset"):
        pills.append(ui.pill("Upset", ui.AMBER, "rgba(252,211,77,0.15)"))
    if game.get("isCorrect") is False:
        pills.append(ui.pill("Miss", ui.ROSE, "rgba(251,113,133,0.15)"))
    if game.get("isCorrect") is True:
        pills.append(ui.pill("Correct pick", ui.EMERALD, "rgba(52,211,153,0.15)"))
    if is_live:
        pills.append(ui.pill("Live", ui.EMERALD, "rgba(52,211,153,0.15)"))
    if is_final:
        inn = f" (F/{game['innings']})" if game.get("innings") else ""
        pills.append(ui.pill(f"Final{inn}", "#5eead4", "rgba(45,212,191,0.12)"))

    with col:
        with st.container(border=True):
            # Header
            center_text = (
                f"F/{game['innings']}" if is_final and game.get("innings")
                else "Live" if is_live
                else fmt_time_et(game["gameDate"])
            )
            st.markdown(
                f"<div style='display:flex;align-items:center;justify-content:space-between;'>"
                f"<span style='font-size:12px;font-weight:500;color:{ui.MUTED}'>"
                f"{'Day Game' if game['dayNight'] == 'day' else 'Night Game'}</span>"
                f"<span style='display:flex;gap:5px;'>{''.join(pills)}</span></div>",
                unsafe_allow_html=True,
            )
            # Scoreboard
            away_score = game["away"].get("score")
            home_score = game["home"].get("score")
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;padding:12px 4px 6px;'>"
                f"<div style='text-align:left;'><div style='font-size:30px;font-weight:700;color:{ui.TEXT};"
                f"font-variant-numeric:tabular-nums;'>{away_score if away_score is not None else '—'}</div>"
                f"<div style='font-size:12px;font-weight:500;color:{ui.MUTED}'>{game['away']['abbrev']}</div></div>"
                f"<div style='text-align:center;font-size:12px;font-weight:500;color:{ui.MUTED}'>{center_text}</div>"
                f"<div style='text-align:right;'><div style='font-size:30px;font-weight:700;color:{ui.TEXT};"
                f"font-variant-numeric:tabular-nums;'>{home_score if home_score is not None else '—'}</div>"
                f"<div style='font-size:12px;font-weight:500;color:{ui.MUTED}'>{game['home']['abbrev']}</div></div></div>",
                unsafe_allow_html=True,
            )

            def team_row(side_key: str, prob: float, is_pick: bool, is_home: bool):
                team = game[side_key]
                color = home_color if is_home else away_color
                rec = ""
                if team.get("wins") is not None and team.get("losses") is not None:
                    rec = f"{team['wins']}-{team['losses']}"
                pick_pill = ui.pill("Pick", ui.ACCENT, "rgba(77,125,255,0.18)") if is_pick else ""
                st.markdown(
                    f"<div style='padding:4px 2px;'>"
                    f"<div style='display:flex;align-items:center;gap:10px;'>"
                    f"<span style='width:4px;height:36px;border-radius:999px;background:{color};'></span>"
                    f"<div style='flex:1;min-width:0;'>"
                    f"<div style='display:flex;align-items:center;gap:8px;'>"
                    f"<span style='font-size:14px;font-weight:600;color:{ui.TEXT};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{team['name']}</span>"
                    f"<span style='font-size:11px;font-weight:500;color:{ui.MUTED}'>{team['abbrev']}{(' · ' + rec) if rec else ''}</span>"
                    f"{pick_pill}</div>"
                    f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{ui.MUTED}'>"
                    f"{'Home' if is_home else 'Away'}</div></div>"
                    f"<span style='font-size:14px;font-weight:700;color:{color};font-variant-numeric:tabular-nums;'>{fmt_pct(prob)}</span></div>"
                    f"<div style='margin-top:6px;'>{ui.bar(prob * 100, color)}</div></div>",
                    unsafe_allow_html=True,
                )

            team_row("home", game["homeWinProb"], game.get("pickTeam") == "home", True)
            team_row("away", game["awayWinProb"], game.get("pickTeam") == "away", False)

            st.markdown(
                f"<div style='text-align:center;font-size:11px;color:{ui.MUTED};padding:2px 0 6px;'>"
                f"Pre-game: {game['home']['abbrev']} {fmt_pct(game['homeWinProb'])} vs "
                f"{game['away']['abbrev']} {fmt_pct(game['awayWinProb'])}</div>",
                unsafe_allow_html=True,
            )

            # Pitchers
            hp = game.get("homePitcher") or {}
            ap = game.get("awayPitcher") or {}
            p1, p2 = st.columns(2)
            for c, p, label in ((p1, hp, game["home"]["abbrev"]), (p2, ap, game["away"]["abbrev"])):
                with c:
                    era = f"ERA {p['era']:.2f}" if p.get("era") is not None else "ERA —"
                    k9 = f"K/9 {p['k9']:.1f}" if p.get("k9") is not None else ""
                    st.markdown(
                        f"<div style='background:rgba(255,255,255,0.02);border:1px solid {ui.BORDER};"
                        f"border-radius:10px;padding:7px 10px;'>"
                        f"<div style='font-size:12px;font-weight:500;color:{ui.TEXT};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{p.get('name') or 'TBD'} <span style='color:{ui.MUTED}'>({label})</span></div>"
                        f"<div style='font-size:11px;color:{ui.MUTED}'> {era} {('  ' + k9) if k9 else ''}</div></div>",
                        unsafe_allow_html=True,
                    )

            # Run projection
            rp = game.get("runProjection")
            if rp:
                market = game.get("marketOdds") or {}
                total_label = f"{market['total']:.1f}" if market.get("total") is not None else f"{rp['total']:.1f}"
                total_prices = ""
                if market.get("overPrice") is not None:
                    total_prices = f" (O {fmt_american(market['overPrice'])} / U {fmt_american(market.get('underPrice', 0))})"
                rl_label = f"±{market['runLine']:.1f}".replace(".0", "") if market.get("runLine") is not None else "±1.5"
                rl_prices = ""
                if market.get("homeRunLinePrice") is not None:
                    rl_prices = f" ({game['home']['abbrev']} {fmt_american(market['homeRunLinePrice'])} / {game['away']['abbrev']} {fmt_american(market.get('awayRunLinePrice', 0))})"
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.02);border:1px solid {ui.BORDER};border-radius:10px;padding:8px 10px;margin-top:8px;'>"
                    f"<div style='display:flex;justify-content:space-between;font-size:11px;'>"
                    f"<span style='color:{ui.MUTED}'>Predicted score</span>"
                    f"<span style='font-weight:600;color:{ui.TEXT}'>HOME {rp['homeScore']:.1f} – {rp['awayScore']:.1f} AWAY</span></div>"
                    f"<div style='display:flex;justify-content:space-between;font-size:11px;margin-top:5px;'>"
                    f"<span style='color:{ui.MUTED}'>Total {total_label}{total_prices}</span>"
                    f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums;'>O {fmt_pct(rp['overProb'])} · U {fmt_pct(rp['underProb'])}</span></div>"
                    f"<div style='display:flex;justify-content:space-between;font-size:11px;margin-top:5px;'>"
                    f"<span style='color:{ui.MUTED}'>Run line {rl_label}{rl_prices}</span>"
                    f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums;'>{game['home']['abbrev']} {fmt_pct(rp['homeRunLineProb'])} · {game['away']['abbrev']} {fmt_pct(rp['awayRunLineProb'])}</span></div></div>",
                    unsafe_allow_html=True,
                )

            # Venue / weather / IL / odds / edge
            meta_bits = []
            if game.get("venue"):
                meta_bits.append(game["venue"])
            w = game.get("weather") or {}
            if w.get("tempF") is not None:
                meta_bits.append(f"{w['tempF']:.0f}°F")
            if w.get("windMph") is not None:
                meta_bits.append(f"{w['windMph']:.0f} mph wind")
            if meta_bits:
                st.markdown(
                    f"<div style='margin-top:8px;font-size:11px;color:{ui.MUTED}'>📍 {' · '.join(meta_bits)}</div>",
                    unsafe_allow_html=True,
                )
            if game.get("homeInjuries") is not None or game.get("awayInjuries") is not None:
                st.markdown(
                    f"<div style='margin-top:4px;font-size:11px;color:{ui.MUTED}'>"
                    f"On IL: {game['home']['abbrev']} {game.get('homeInjuries') or 0} · "
                    f"{game['away']['abbrev']} {game.get('awayInjuries') or 0}</div>",
                    unsafe_allow_html=True,
                )

            market = game.get("marketOdds") or {}
            if market.get("homeMoneyline") is not None:
                ml_text = (
                    f"ML: {game['home']['abbrev']} {fmt_american(market['homeMoneyline'])} "
                    f"{game['away']['abbrev']} {fmt_american(market.get('awayMoneyline', 0))}"
                )
            else:
                ml_text = (
                    f"Fair ML: {game['home']['abbrev']} {fmt_american(game.get('fairHomeOdds'))} "
                    f"{game['away']['abbrev']} {fmt_american(game.get('fairAwayOdds'))}"
                )
            edge = game.get("edge") or 0
            edge_color = ui.EMERALD if edge >= 0 else ui.ROSE
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-top:8px;font-size:11px;'>"
                f"<span style='color:{ui.MUTED}'>{ml_text}</span>"
                f"<span style='font-weight:500;color:{edge_color};font-variant-numeric:tabular-nums;'>Edge: {fmt_signed(edge)}</span></div>",
                unsafe_allow_html=True,
            )

            # Lineups expander
            lu = game.get("lineups")
            has_lineups = bool(lu and ((lu.get("home") or {}).get("battingOrder") or (lu.get("away") or {}).get("battingOrder")))
            with st.expander("Starting lineups", expanded=False):
                if not has_lineups:
                    st.caption("Lineup not posted yet.")
                else:
                    c1, c2 = st.columns(2)
                    for c, side_key, label, color in (
                        (c1, "home", game["home"]["abbrev"], home_color),
                        (c2, "away", game["away"]["abbrev"], away_color),
                    ):
                        side = (lu or {}).get(side_key) or {}
                        order = side.get("battingOrder") or []
                        bench = side.get("bench") or []
                        with c:
                            st.markdown(
                                f"<div style='font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:{color};margin-bottom:4px;'>"
                                f"{label} {'(Home)' if side_key == 'home' else '(Away)'}</div>",
                                unsafe_allow_html=True,
                            )
                            if order:
                                for i, p in enumerate(order):
                                    st.markdown(
                                        f"<div style='font-size:12px;display:flex;gap:6px;'>"
                                        f"<span style='width:14px;color:{ui.MUTED};text-align:right;'>{i + 1}</span>"
                                        f"<span style='color:{ui.TEXT};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{p['name']}</span>"
                                        f"<span style='margin-left:auto;font-size:10px;text-transform:uppercase;color:{ui.MUTED}'>{p.get('pos') or ''}</span></div>",
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.caption("Lineup not posted yet")
                            if bench:
                                names = ", ".join(p["name"] for p in bench[:5])
                                more = f" +{len(bench) - 5} more" if len(bench) > 5 else ""
                                st.caption(f"Bench: {names}{more}")

            # SHAP expander
            shap = game.get("shap") or []
            if shap:
                max_shap = max((abs(s["contribution"]) for s in shap), default=1e-6)
                with st.expander("SHAP features", expanded=False):
                    for s in shap:
                        positive = s["contribution"] >= 0
                        color = ui.EMERALD if positive else ui.ROSE
                        width = max(4.0, min(100.0, abs(s["contribution"]) / max_shap * 100))
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:10px;font-size:12px;padding:2px 0;'>"
                            f"<span style='width:120px;color:{ui.MUTED};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{s['label']}</span>"
                            f"<div style='flex:1;'>{ui.bar(width, color, '8px')}</div>"
                            f"<span style='width:52px;text-align:right;color:{color};font-weight:500;font-variant-numeric:tabular-nums;'>{fmt_signed(s['contribution'], 2)}</span></div>",
                            unsafe_allow_html=True,
                        )

            # Deep dive expander
            with st.expander("Deep dive", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{ui.MUTED}'>Home win prob</div>"
                        f"<div style='font-size:20px;font-weight:700;color:{home_color}'>{fmt_pct(game['homeWinProb'])}</div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{ui.MUTED}'>Away win prob</div>"
                        f"<div style='font-size:20px;font-weight:700;color:{away_color}'>{fmt_pct(game['awayWinProb'])}</div>",
                        unsafe_allow_html=True,
                    )
                rows = [
                    ("Fair ML " + game["home"]["abbrev"], fmt_american(game.get("fairHomeOdds"))),
                    ("Fair ML " + game["away"]["abbrev"], fmt_american(game.get("fairAwayOdds"))),
                    ("Model edge vs Elo baseline", fmt_signed(game.get("edge") or 0, 3)),
                ]
                for label, value in rows:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;font-size:12px;padding:4px 0;border-bottom:1px solid {ui.BORDER}'>"
                        f"<span style='color:{ui.MUTED}'>{label}</span><span style='font-weight:600;color:{ui.TEXT}'>{value}</span></div>",
                        unsafe_allow_html=True,
                    )
                if shap:
                    st.markdown(
                        f"<div style='font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:{ui.MUTED};margin-top:10px;'>Feature contributions (logit space)</div>",
                        unsafe_allow_html=True,
                    )
                    for s in shap:
                        c = ui.EMERALD if s["contribution"] >= 0 else ui.ROSE
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;font-size:12px;padding:2px 0;'>"
                            f"<span style='color:{ui.MUTED}'>{s['label']}</span>"
                            f"<span style='color:{ui.MUTED}'>x={s['value']:.2f}</span>"
                            f"<span style='color:{c};font-weight:500;width:56px;text-align:right;'>{fmt_signed(s['contribution'], 3)}</span></div>",
                            unsafe_allow_html=True,
                        )
                st.markdown(
                    f"<div style='font-size:11px;color:{ui.MUTED};margin-top:8px;'>"
                    f"On IL: {game['home']['abbrev']} {game.get('homeInjuries') or 0} · {game['away']['abbrev']} {game.get('awayInjuries') or 0}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.02);border:1px solid {ui.BORDER};border-radius:10px;padding:8px 10px;margin-top:8px;font-size:12px;color:{ui.MUTED}'>"
                    f"<span style='color:{ui.TEXT};font-weight:500'>Starting pitchers</span> — "
                    f"{hp.get('name') or 'TBD'} (home, ERA {hp.get('era', '—') if hp.get('era') is not None else '—'}) · "
                    f"{ap.get('name') or 'TBD'} (away, ERA {ap.get('era', '—') if ap.get('era') is not None else '—'})</div>",
                    unsafe_allow_html=True,
                )

            # Result banner
            if winner_abbrev:
                correct = game.get("isCorrect")
                bg = "rgba(52,211,153,0.10)" if correct else "rgba(251,113,133,0.10)"
                color = ui.EMERALD if correct else ui.ROSE
                text = (
                    f"{winner_abbrev} Won — Model Correct" if correct
                    else f"{winner_abbrev} Won — Upset! Model picked {pick_abbrev}"
                )
                st.markdown(
                    f"<div style='text-align:center;padding:9px;border-radius:10px;margin-top:10px;"
                    f"font-size:12px;font-weight:600;color:{color};background:{bg};'>"
                    f"{'✓' if correct else '✗'} {text}</div>",
                    unsafe_allow_html=True,
                )


def games_tab(bundle) -> None:
    ms = bundle["model_state"]
    today = ms["asOfDate"]
    season = int(ms["season"])

    if "games_date" not in st.session_state:
        st.session_state.games_date = _dt.date.fromisoformat(today)
    if "games_filter" not in st.session_state:
        st.session_state.games_filter = "All Games"

    _ensure_games_for_date(bundle)
    bundle = st.session_state.bundle
    ms = bundle["model_state"]

    ymd = st.session_state.games_date.isoformat()
    games = _games_for_date(bundle, ymd)
    # Never display a prediction that isn't point-in-time (stale docs survive
    # only if regeneration failed; the warning was already shown above).
    games = [g for g in games if _doc_is_point_in_time(g, ymd, today)]
    filtered = _filter_games(games, str(st.session_state.games_filter).split(" (")[0])
    night_count = sum(1 for g in games if g["dayNight"] == "night")
    record = ms.get("todaysRecord") or {}

    # Row 1: summary chips (mirrors the React games-tab header row)
    row1 = [f"<span style='font-size:14px;font-weight:600;color:{ui.TEXT};'>{fmt_date_short(ymd)}</span>"]
    if games:
        row1.append(
            f"<span style='font-size:14px;color:{ui.MUTED};'>{len(filtered)} of {len(games)} games shown</span>"
        )
    if night_count > 0:
        row1.append(
            f"<span style='background:rgba(59,130,246,0.15);color:#93c5fd;border-radius:999px;"
            f"padding:4px 10px;font-size:12px;font-weight:500;'>{night_count} evening games begin 7 PM ET+</span>"
        )
    if ymd == today and record.get("total", 0) > 0:
        row1.append(
            f"<span style='background:rgba(52,211,153,0.15);color:#6ee7b7;border-radius:999px;"
            f"padding:4px 10px;font-size:12px;font-weight:600;'>✓ {record['wins']}-{record['losses']} Today</span>"
        )
        row1.append(
            f"<span style='font-size:12px;color:{ui.MUTED};'>{fmt_pct(record['accuracy'], 1)} accuracy</span>"
        )
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;gap:8px;'>{''.join(row1)}</div>",
        unsafe_allow_html=True,
    )

    # Row 2: centered date selector (prev square / date pill / next square)
    lo, hi = _dt.date(season, 2, 1), _dt.date(season, 11, 15)
    c_l, c_mid, c_r = st.columns([1, 2, 1], vertical_alignment="center")
    with c_l:
        st.markdown("")
    with c_mid:
        c_prev, c_pill, c_next = st.columns([1, 6, 1], vertical_alignment="center")
        with c_prev:
            if st.button("‹", key="date_prev", help="Previous day"):
                st.session_state.games_date = max(lo, min(hi, st.session_state.games_date - _dt.timedelta(days=1)))
                st.rerun()
        with c_pill:
            # React-style pill: long date label + calendar in a popover. The
            # date_input `format` is restricted by Streamlit to YYYY/MM/DD-style
            # patterns — the old "ddd, MMM D, YYYY" string crashed the app with
            # StreamlitAPIException on every render of this tab.
            with st.popover(fmt_date_long(ymd)):
                st.date_input(
                    "Game date",
                    key="games_date",
                    min_value=lo,
                    max_value=hi,
                    format="YYYY-MM-DD",
                    label_visibility="collapsed",
                )
        with c_next:
            if st.button("›", key="date_next", help="Next day"):
                st.session_state.games_date = max(lo, min(hi, st.session_state.games_date + _dt.timedelta(days=1)))
                st.rerun()
    with c_r:
        st.markdown("")

    # Row 3: filter pills with counts
    counts = {
        "All Games": len(games),
        "Final": sum(1 for g in games if g["status"] == "Final"),
        "Live": sum(1 for g in games if g["status"] == "Live"),
        "Upcoming": sum(1 for g in games if g["status"] in ("Preview", "Scheduled")),
    }
    options = [f"{k} ({counts[k]})" for k in counts if k == "All Games" or counts[k] > 0]
    stored = st.session_state.get("games_filter")
    raw_cur = stored.split(" (")[0] if isinstance(stored, str) else "All Games"
    if raw_cur != "All Games" and counts.get(raw_cur, 0) == 0:
        raw_cur = "All Games"
    display_cur = f"{raw_cur} ({counts[raw_cur]})"
    if not any(display_cur == o for o in options):
        display_cur = options[0]
    st.session_state.games_filter = display_cur
    selected = st.segmented_control("Filter", options, key="games_filter")
    raw_filter = selected.split(" (")[0] if isinstance(selected, str) else "All Games"
    filtered = _filter_games(games, raw_filter)

    # Content
    if not games:
        st.markdown(
            f"<div style='text-align:center;padding:80px 0;color:{ui.MUTED};font-size:14px;'>"
            f"No games found for {fmt_date_long(ymd)}.</div>",
            unsafe_allow_html=True,
        )
        if st.button("🔄 Refresh data", key="refresh_data_btn"):
            do_refresh()
        return
    if not filtered:
        st.markdown(
            f"<div style='text-align:center;padding:80px 0;color:{ui.MUTED};font-size:14px;'>"
            f"No {st.session_state.games_filter.lower()} games on {fmt_date_long(ymd)}.</div>",
            unsafe_allow_html=True,
        )
        return

    for i in range(0, len(filtered), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(filtered):
                _game_card(filtered[i + j], col)


# ---------------------------------------------------------------------------
# Tab 2 — Power Rankings
# ---------------------------------------------------------------------------

def rankings_tab(bundle) -> None:
    ms = bundle["model_state"]
    today = ms["asOfDate"]
    if "rank_date" not in st.session_state:
        st.session_state.rank_date = _dt.date.fromisoformat(today)
    if st.session_state.rank_date > _dt.date.fromisoformat(today):
        st.session_state.rank_date = _dt.date.fromisoformat(today)
    sel = st.session_state.rank_date.isoformat()
    if sel < today:
        # Point-in-time rankings: the Elo table as it stood before that date
        # (cached walk-forward model for that date); stored rankings as fallback.
        asof = power_rankings_as_of(sel)
        rankings = asof if asof else (ms.get("powerRankings") or [])
        label = (f"As of {fmt_date_long(sel)} · point-in-time (trained only on prior games)"
                 if asof else f"As of {fmt_date_long(sel)} · back to current (insufficient history)")
    else:
        rankings = ms.get("powerRankings") or []
        label = f"As of {fmt_date_long(today)} · current"
    with st.container(border=True):
        c_lab, c_pick, c_info = st.columns([1.2, 2.0, 5.0], vertical_alignment="center")
        with c_lab:
            st.markdown(
                f"<span style='font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;"
                f"color:{ui.MUTED}'>As of date</span>",
                unsafe_allow_html=True,
            )
        with c_pick:
            st.date_input(
                "Rankings as of",
                key="rank_date",
                max_value=_dt.date.fromisoformat(today),
                label_visibility="collapsed",
            )
        with c_info:
            st.markdown(
                f"<span style='font-size:12px;color:{ui.MUTED}'>{label}</span>",
                unsafe_allow_html=True,
            )
    ui.section(
        "Power Rankings",
        f"Elo-based power rankings · {label} · All {len(rankings)} teams",
    )
    rows = []
    for i, r in enumerate(rankings):
        meta = team_meta(r["teamId"])
        elo_color = ui.CYAN if i < 5 else ui.AMBER if i < 10 else ui.TEXT
        run_diff = r.get("runDiff") or 0
        rd_color = ui.EMERALD if run_diff > 0 else ui.ROSE if run_diff < 0 else ui.MUTED
        l10_wins = round(r.get("last10WinPct", 0.5) * 10)
        rows.append([
            f"<span style='color:{ui.MUTED};font-variant-numeric:tabular-nums;'>{i + 1}</span>",
            f"<span style='display:inline-block;width:4px;height:18px;border-radius:999px;background:{meta['color']};vertical-align:middle;margin-right:8px;'></span>"
            f"<b style='color:{ui.TEXT}'>{r['name']}</b> <span style='color:{ui.MUTED};font-size:11px'>{r['abbrev']}</span>",
            f"<b style='color:{elo_color};font-variant-numeric:tabular-nums;'>{round(r['elo'])}</b>",
            f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums;'>{r['wins']}-{r['losses']}</span>",
            f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums;'>{r['winPct']:.3f}".replace("0.", ".") + "</span>",
            f"<span style='color:{rd_color};font-variant-numeric:tabular-nums;'>{('+' if run_diff > 0 else '')}{run_diff}</span>",
            f"<span style='color:{ui.MUTED};font-variant-numeric:tabular-nums;'>{l10_wins}-{10 - l10_wins}</span>",
            f"<span style='color:{ui.MUTED};font-variant-numeric:tabular-nums;'>{r['homeWinPct']:.3f}".replace("0.", ".") + "</span>",
            f"<span style='color:{ui.MUTED};font-variant-numeric:tabular-nums;'>{r['awayWinPct']:.3f}".replace("0.", ".") + "</span>",
        ])
    st.markdown(
        f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:8px 12px;overflow-x:auto;'>"
        + ui.html_table(
            ["Rank", "Team", "Elo", "W-L", "Pct", "Run Diff", "L10", "Home%", "Away%"],
            rows,
            align=["left", "left", "right", "right", "right", "right", "right", "right", "right"],
        )
        + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 3 — Calibration
# ---------------------------------------------------------------------------

def _calibration_range(rows: list[dict], fallback: str) -> tuple[_dt.date, _dt.date]:
    if not rows:
        return _dt.date.fromisoformat(fallback), _dt.date.fromisoformat(fallback)
    dates = sorted(r["date"] for r in rows)
    return _dt.date.fromisoformat(dates[0]), _dt.date.fromisoformat(dates[-1])


def _moneyline_metrics(rows: list[dict]) -> dict:
    preds = [r["pickProb"] for r in rows]
    labels = [1 if r["isCorrect"] else 0 for r in rows]
    ev = evaluate(preds, labels)
    curve = calibration_curve_points(preds, labels, 12)
    return {"ev": ev, "curve": curve if curve else ev["calibrationCurve"], "n": len(rows)}


def _totals_metrics(rows: list[dict]) -> dict:
    n = 0
    s_abs = s_sq = s_bias = 0.0
    for r in rows:
        if r.get("predictedTotal") is None:
            continue
        n += 1
        err = r["predictedTotal"] - r["actualTotal"]
        s_abs += abs(err)
        s_sq += err * err
        s_bias += err
    return {
        "n": n,
        "mae": s_abs / n if n else 0,
        "rmse": (s_sq / n) ** 0.5 if n else 0,
        "bias": s_bias / n if n else 0,
    }


def _runline_metrics(rows: list[dict]) -> dict:
    preds = [r["homeRunLineProb"] for r in rows if r.get("homeRunLineProb") is not None]
    labels = [1 if r["actualMargin"] >= 2 else 0 for r in rows if r.get("homeRunLineProb") is not None]
    if not preds:
        return {"n": 0, "auc": 0, "brier": 0, "accuracy": 0}
    return {
        "n": len(preds),
        "auc": compute_auc(preds, labels),
        "brier": compute_brier(preds, labels),
        "accuracy": sum(1 for p, y in zip(preds, labels) if (1 if p >= 0.5 else 0) == y) / len(preds),
    }


def _reliability_rows(bins: list[dict]) -> str:
    rows = []
    for b in bins:
        gap = b["gap"]
        gap_color = ui.EMERALD if abs(gap) < 0.015 else ui.AMBER if abs(gap) < 0.025 else ui.ROSE
        rows.append([
            f"<span style='color:{ui.TEXT};font-weight:500'>{b['label']}</span>",
            f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{b['meanPredicted']:.3f}</span>",
            f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{b['meanActual']:.3f}</span>",
            f"<span style='color:{ui.MUTED};font-variant-numeric:tabular-nums'>{b['count']}</span>",
            f"<span style='color:{gap_color};font-variant-numeric:tabular-nums'>{('+' if gap >= 0 else '')}{gap:.3f}</span>",
        ])
    return ui.html_table(
        ["Bucket", "Mean predicted", "Mean actual", "Count", "Gap"],
        rows,
        align=["left", "right", "right", "right", "right"],
    )


def _game_history_rows(view: str, rows: list[dict]) -> tuple[list[str], list[list], list[str]]:
    if view == "Moneyline":
        headers = ["Date", "Matchup", "Final", "Predicted", "Actual", "Result"]
        align = ["left", "left", "right", "right", "right", "right"]
        out = []
        for r in rows:
            pick = r["home"]["abbrev"] if r["pickTeam"] == "home" else r["away"]["abbrev"]
            winner = r["home"]["abbrev"] if r["winner"] == "home" else r["away"]["abbrev"]
            result = (
                f"<span style='background:rgba(52,211,153,0.15);color:{ui.EMERALD};border-radius:999px;padding:2px 8px;font-size:11px;font-weight:600;'>✓ Correct</span>"
                if r["isCorrect"]
                else f"<span style='background:rgba(251,113,133,0.15);color:{ui.ROSE};border-radius:999px;padding:2px 8px;font-size:11px;font-weight:600;'>✗ Upset</span>"
            )
            out.append([
                f"<span style='color:{ui.MUTED}'>{short_date(r['date'])}</span>",
                f"<b style='color:{ui.TEXT}'>{r['away']['abbrev']}</b><span style='color:{ui.MUTED}'> @ </span><b style='color:{ui.TEXT}'>{r['home']['abbrev']}</b>",
                f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{r['away'].get('score') if r['away'].get('score') is not None else '—'} – {r['home'].get('score') if r['home'].get('score') is not None else '—'}</span>",
                f"<span style='color:{ui.MUTED};font-variant-numeric:tabular-nums'>{pick} {fmt_pct(r['pickProb'])}</span>",
                f"<span style='color:{ui.TEXT};font-weight:500'>{winner}</span>",
                result,
            ])
        return headers, out, align

    if view == "Game Totals":
        headers = ["Date", "Matchup", "Predicted Total", "Actual Total", "Diff"]
        align = ["left", "left", "right", "right", "right"]
        out = []
        for r in rows:
            pt = r.get("predictedTotal")
            at = r.get("actualTotal")
            if pt is None:
                diff = "—"
            else:
                d = pt - at
                diff = (
                    f"<span style='color:{ui.ROSE};font-weight:600;font-variant-numeric:tabular-nums'>{('+' if d >= 0 else '')}{d:.1f}</span>"
                    if d >= 0 else
                    f"<span style='color:{ui.EMERALD};font-weight:600;font-variant-numeric:tabular-nums'>{d:.1f}</span>"
                )
            out.append([
                f"<span style='color:{ui.MUTED}'>{short_date(r['date'])}</span>",
                f"<b style='color:{ui.TEXT}'>{r['away']['abbrev']}</b><span style='color:{ui.MUTED}'> @ </span><b style='color:{ui.TEXT}'>{r['home']['abbrev']}</b>",
                f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{pt:.1f}" if pt is not None else "—",
                f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{at if at is not None else '—'}</span>",
                diff,
            ])
        return headers, out, align

    headers = ["Date", "Matchup", "Home Cover Prob", "Margin", "Result"]
    align = ["left", "left", "right", "right", "right"]
    out = []
    for r in rows:
        prob = r.get("homeRunLineProb")
        margin = r.get("actualMargin")
        if margin is None:
            result = "—"
        elif margin >= 2:
            result = (
                f"<span style='background:rgba(52,211,153,0.15);color:{ui.EMERALD};border-radius:999px;padding:2px 8px;font-size:11px;font-weight:600;'>✓ Home covers</span>"
            )
        else:
            result = (
                f"<span style='background:rgba(77,125,255,0.18);color:{ui.ACCENT};border-radius:999px;padding:2px 8px;font-size:11px;font-weight:600;'>Away covers</span>"
            )
        out.append([
            f"<span style='color:{ui.MUTED}'>{short_date(r['date'])}</span>",
            f"<b style='color:{ui.TEXT}'>{r['away']['abbrev']}</b><span style='color:{ui.MUTED}'> @ </span><b style='color:{ui.TEXT}'>{r['home']['abbrev']}</b>",
            f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{fmt_pct(prob, 1) if prob is not None else '—'}</span>",
            f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{('+' if margin is not None and margin >= 0 else '') + str(margin) if margin is not None else '—'}</span>",
            result,
        ])
    return headers, out, align


def calibration_tab(bundle) -> None:
    ms = bundle["model_state"]
    rows_in = bundle["calibration_rows"]
    rows_wf = bundle.get("calibration_rows_wf") or []

    # Method toggle: full-season (in-sample) vs strict walk-forward backtest.
    if "cal_method" not in st.session_state:
        # Point-in-time is the default: every prediction shown must have been
        # made with only prior knowledge (walk-forward), not the in-sample fit.
        st.session_state.cal_method = "Walk-forward (point-in-time)"
    st.segmented_control(
        "Method",
        ["Full-season (in-sample)", "Walk-forward (point-in-time)"],
        key="cal_method",
        default=st.session_state.cal_method,
    )
    method = st.session_state.cal_method
    rows = rows_wf if method == "Walk-forward (point-in-time)" else rows_in
    total_games = len(rows)

    # Header: title → n pill → subtitle (mirrors the React calibration header)
    trained_at = fmt_trained_at(ms.get("trainedAt", 0))
    st.markdown(
        f"<h2 style='margin:0;font-size:20px;font-weight:700;letter-spacing:-.01em;color:{ui.TEXT}'>"
        f"Model Calibration Dashboard</h2>",
        unsafe_allow_html=True,
    )
    if method == "Walk-forward (point-in-time)":
        st.markdown(
            f"<div style='margin-top:10px;'>{ui.pill(f'n = {fmt_number(len(rows_wf))} games scored point-in-time · each day trained on prior games only', ui.ACCENT, 'rgba(77,125,255,0.15)')}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='margin-top:10px;'>{ui.pill(f'n = {fmt_number(total_games)} games in range · Trained {trained_at} ET', ui.MUTED, 'rgba(255,255,255,0.05)')}</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<p style='margin:10px 0 0;font-size:13px;color:{ui.MUTED};line-height:1.5'>"
        "Assessing prediction reliability and accuracy for moneyline, game totals, and run lines — "
        "computed on one side per game.</p>",
        unsafe_allow_html=True,
    )

    # First run of the walk-forward backtest: build automatically once per
    # session (cached per date afterwards), with a manual fallback button.
    if method == "Walk-forward (point-in-time)" and not rows_wf:
        if not st.session_state.get("cal_wf_attempted", False):
            st.session_state.cal_wf_attempted = True
            progress = st.progress(0, text="Building walk-forward calibration (first time only)...")
            try:
                build_walk_forward_calibration_rows(
                    report=lambda stage, pct, msg: progress.progress(
                        max(0, min(int(pct), 99)), text=msg or stage
                    ),
                )
            except Exception as e:  # noqa: BLE001
                st.warning(f"Walk-forward build failed: {e}")
            finally:
                progress.empty()
            nb = load_bundle()
            if nb is not None:
                st.session_state.bundle = nb
            st.rerun()
        st.markdown(
            f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;"
            f"padding:16px;margin-top:12px;'>"
            f"<h3 style='margin:0;font-size:14px;font-weight:600;color:{ui.TEXT}'>Build the true backtest</h3>"
            f"<p style='margin:6px 0 0;font-size:13px;color:{ui.MUTED};line-height:1.5'>"
            f"Every completed game gets scored by a fresh model trained only on games before it — "
            f"no game is ever predicted by a model that saw its result. Runs once (cached per date, "
            f"incremental afterwards); the first build takes a few minutes.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Build walk-forward calibration", key="cal_build_wf", type="primary", use_container_width=True):
            progress = st.progress(0, text="Building walk-forward calibration…")
            try:
                build_walk_forward_calibration_rows(
                    report=lambda stage, pct, msg: progress.progress(
                        max(0, min(int(pct), 99)), text=msg or stage
                    ),
                )
            except Exception as e:  # noqa: BLE001
                st.warning(f"Walk-forward build failed: {e}")
            finally:
                progress.empty()
            nb = load_bundle()
            if nb is not None:
                st.session_state.bundle = nb
            st.rerun()

    min_date, max_date = _calibration_range(rows, ms.get("asOfDate") or "")

    if "cal_view" not in st.session_state:
        st.session_state.cal_view = "Moneyline"
    if "cal_start" not in st.session_state:
        st.session_state.cal_start = min_date
    if "cal_end" not in st.session_state:
        st.session_state.cal_end = max_date
    if "cal_search" not in st.session_state:
        st.session_state.cal_search = ""
    if "cal_pages" not in st.session_state:
        st.session_state.cal_pages = 1

    # View toggle (pill style, matches React)
    st.segmented_control(
        "View",
        ["Moneyline", "Game Totals", "Run Lines (-1.5 / +1.5)"],
        key="cal_view",
        default="Moneyline",
    )
    view = st.session_state.cal_view

    # Range selector card (Range label + start → end + counts, matches React)
    prev_start = st.session_state.cal_start
    prev_end = st.session_state.cal_end
    if prev_start > prev_end:
        prev_start, prev_end = prev_end, prev_start
    preview = [r for r in rows if prev_start.isoformat() <= r["date"] <= prev_end.isoformat()]
    preview_acc = sum(1 for r in preview if r["isCorrect"]) / len(preview) if preview else 0
    with st.container(border=True):
        c_label, c_start, c_arrow, c_end, c_info = st.columns([0.6, 1.7, 0.3, 1.7, 3.6], vertical_alignment="center")
        with c_label:
            st.markdown(
                f"<span style='font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;"
                f"color:{ui.MUTED}'>Range</span>",
                unsafe_allow_html=True,
            )
        with c_start:
            st.date_input(
                "Start",
                key="cal_start",
                min_value=_dt.date.fromisoformat(SEASON_START),
                max_value=st.session_state.cal_end,
                label_visibility="collapsed",
            )
        with c_arrow:
            st.markdown(
                f"<div style='text-align:center;color:{ui.MUTED};'>→</div>",
                unsafe_allow_html=True,
            )
        with c_end:
            st.date_input(
                "End",
                key="cal_end",
                min_value=st.session_state.cal_start,
                max_value=max_date,
                label_visibility="collapsed",
            )
        with c_info:
            st.markdown(
                f"<span style='font-size:12px;color:{ui.MUTED}'>"
                f"{fmt_number(len(preview))} completed game{'s' if len(preview) != 1 else ''} · "
                f"{fmt_pct(preview_acc, 1)} accuracy</span>",
                unsafe_allow_html=True,
            )

    if st.session_state.cal_start > st.session_state.cal_end:
        st.session_state.cal_start, st.session_state.cal_end = st.session_state.cal_end, st.session_state.cal_start

    start_ymd = st.session_state.cal_start.isoformat()
    end_ymd = st.session_state.cal_end.isoformat()
    in_range = [r for r in rows if start_ymd <= r["date"] <= end_ymd]
    if not in_range:
        st.markdown(
            f"<div style='text-align:center;padding:28px 0;color:{ui.MUTED};font-size:13px;'>"
            f"No point-in-time results in this range. Build the walk-forward backtest above, "
            f"or widen the date range.</div>",
            unsafe_allow_html=True,
        )
        return

    # Today's record card
    record = ms.get("todaysRecord") or {}
    if view == "Moneyline" and record.get("total", 0) > 0:
        rec_pill = ui.pill(f"{record['wins']}-{record['losses']}", ui.EMERALD, "rgba(52,211,153,0.15)")
        st.markdown(
            f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:14px 16px;margin-top:12px;'>"
            f"<span style='font-size:13px;color:{ui.TEXT}'>Today's Record: </span>"
            f"{rec_pill} "
            f"<span style='font-size:12px;color:{ui.MUTED}'> {record['completed']} completed games · {record['correct']} correct picks "
            f"({fmt_pct(record['accuracy'], 1)}) · {len(record['upsets'])} upsets</span></div>",
            unsafe_allow_html=True,
        )
        if record.get("upsets"):
            upsets_html = " ".join(
                ui.pill(f"⚡ {u['team']} {u['prob']}% upset", ui.AMBER, "rgba(252,211,77,0.15)") for u in record["upsets"]
            )
            st.markdown(
                f"<div style='display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;'>{upsets_html}</div>",
                unsafe_allow_html=True,
            )

    if view == "Moneyline":
        m = _moneyline_metrics(in_range)
        ev = m["ev"]
        total = m["n"]
        # Calibration curve
        st.markdown(
            f"<div style='display:flex;align-items:center;justify-content:space-between;margin-top:10px;'>"
            f"<span style='font-size:14px;font-weight:600;color:{ui.TEXT}'>Calibration Curve</span>"
            f"<span style='display:flex;gap:14px;'>{ui.legend(ui.ACCENT, f'Model (n={fmt_number(total)})')}{ui.legend(ui.MUTED, 'Perfect calibration', dashed=True)}</span></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(calibration_scatter(m["curve"], total), use_container_width=True)

        # Metric cards
        cards = [
            ("AUC-ROC", fmt_prob(ev["auc"]), ui.CYAN, ""),
            ("Brier Score", fmt_prob(ev["brier"]), ui.EMERALD, ""),
            ("Log-Loss", fmt_prob(ev["logLoss"]), ui.AMBER, "Penalizes confidence"),
            ("Cal. Error", fmt_prob(ev["ece"]), ui.PURPLE, "ECE metric"),
        ]
        cols = st.columns(4)
        for col, (label, value, color, sub) in zip(cols, cards):
            with col:
                st.markdown(ui.metric_card(label, value, color, sub), unsafe_allow_html=True)
        st.caption(
            "Metrics are computed on the predicted favorite only (probability > 50%), one side per game, over the selected date range."
        )

        # Confidence distribution
        st.markdown(
            f"<div style='display:flex;align-items:center;justify-content:space-between;margin-top:14px;'>"
            f"<span style='font-size:14px;font-weight:600;color:{ui.TEXT}'>Prediction Confidence Distribution &amp; Accuracy</span>"
            f"<span style='display:flex;gap:14px;'>{ui.legend(ui.EMERALD, 'Actual accuracy %')}{ui.legend('rgba(77,125,255,0.45)', 'Game count')}</span></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(confidence_chart(ev["confidenceDistribution"]), use_container_width=True)

        # Reliability table
        st.markdown(
            f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:16px;margin-top:14px;'>"
            f"<h3 style='margin:0 0 12px;font-size:14px;font-weight:600;color:{ui.TEXT}'>Reliability Diagram — Binned Data</h3>"
            + (_reliability_rows(ev["bins"]) if ev["bins"] else "<div style='color:#8b939f;font-size:13px;padding:20px 0;text-align:center'>No binned data for this range.</div>")
            + "</div>",
            unsafe_allow_html=True,
        )
    elif view == "Game Totals":
        tm = _totals_metrics(in_range)
        cards = [
            ("Mean Abs. Error", f"{tm['mae']:.2f}", ui.CYAN, "Runs"),
            ("RMSE", f"{tm['rmse']:.2f}", ui.EMERALD, "Runs"),
            ("Bias", f"{tm['bias']:+.2f}", ui.AMBER, "Predicted − actual"),
            ("Games", fmt_number(tm["n"]), ui.PURPLE, "With run projection"),
        ]
        cols = st.columns(4)
        for col, (label, value, color, sub) in zip(cols, cards):
            with col:
                st.markdown(ui.metric_card(label, value, color, sub), unsafe_allow_html=True)
        st.caption("Predicted combined runs (both teams) versus the actual final score total over the selected range.")
    else:
        rl = _runline_metrics(in_range)
        cards = [
            ("Run-Line AUC", fmt_prob(rl["auc"]), ui.CYAN, ""),
            ("Brier Score", fmt_prob(rl["brier"]), ui.EMERALD, ""),
            ("Cover Accuracy", fmt_pct(rl["accuracy"], 1), ui.AMBER, "Home −1.5 covers"),
            ("Games", fmt_number(rl["n"]), ui.PURPLE, "With run projection"),
        ]
        cols = st.columns(4)
        for col, (label, value, color, sub) in zip(cols, cards):
            with col:
                st.markdown(ui.metric_card(label, value, color, sub), unsafe_allow_html=True)
        st.caption(
            "Home team covers −1.5 when it wins by 2+ runs; away team covers +1.5 otherwise. One side per game, calibrated probabilities."
        )

    # Game history table
    headers, history_rows, align = _game_history_rows(view, in_range)
    correct = sum(1 for r in in_range if r["isCorrect"])
    if view == "Moneyline":
        sub = f"{fmt_number(len(in_range))} games · {fmt_number(correct)} correct picks ({fmt_pct(len(in_range) and correct / len(in_range) or 0, 1)})"
    elif view == "Game Totals":
        sub = f"{fmt_number(_totals_metrics(in_range)['n'])} games with projections"
    else:
        sub = f"{fmt_number(_runline_metrics(in_range)['n'])} games with projections"

    st.markdown(
        f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:16px;margin-top:14px;'>"
        f"<div style='display:flex;align-items:flex-end;justify-content:space-between;gap:10px;flex-wrap:wrap;'>"
        f"<div><h3 style='margin:0;font-size:14px;font-weight:600;color:{ui.TEXT}'>Game History — "
        f"{'Predicted vs Actual' if view == 'Moneyline' else 'Predicted vs Actual Totals' if view == 'Game Totals' else 'Run Line Results'}</h3>"
        f"<p style='margin:4px 0 0;font-size:12px;color:{ui.MUTED}'>{sub}</p></div></div>",
        unsafe_allow_html=True,
    )
    search = st.text_input("Filter by team…", key="cal_search", placeholder="Filter by team…")
    q = search.strip().lower()
    if q:
        # Filter the source rows (not the rendered HTML) so the search stays
        # correct even when two games render identical cells.
        in_range = [
            r for r in in_range
            if q in f"{r['away']['name']} {r['away']['abbrev']} {r['home']['name']} {r['home']['abbrev']}".lower()
        ]
        headers, history_rows, align = _game_history_rows(view, in_range)
    page_size = 100
    pages = st.session_state.cal_pages
    visible = history_rows[: page_size * pages]
    st.markdown(
        ui.html_table(headers, visible, align=align) + "</div>",
        unsafe_allow_html=True,
    )
    if len(history_rows) > page_size * pages:
        if st.button("Load more games", key="cal_load_more", use_container_width=True):
            st.session_state.cal_pages += 1
            st.rerun()
    if not history_rows:
        st.markdown(
            f"<div style='text-align:center;padding:24px 0;color:{ui.MUTED};font-size:13px;'>"
            f"{'No games match your filter.' if q else 'No completed games in this range — adjust the dates or click Refresh.'}</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 4 — Model Monitor
# ---------------------------------------------------------------------------

def _feature_item(rank: int, f: dict, total_weight: float, max_imp: float) -> str:
    weight_pct = round(abs(f["weight"]) / total_weight * 100) if total_weight else 0
    bar_pct = round(f["importance"] / max_imp * 100) if max_imp else 0
    return (
        f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:14px 16px;margin-bottom:10px;'>"
        f"<div style='display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap;'>"
        f"<div style='display:flex;align-items:flex-start;gap:10px;'>"
        f"<span style='color:{ui.CYAN};font-weight:700;font-size:14px;'>#{rank}</span>"
        f"<div><div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>"
        f"<span style='font-size:14px;font-weight:600;color:{ui.TEXT}'>{f['label']}</span>"
        f"<span style='background:rgba(77,125,255,0.15);color:{ui.ACCENT};border-radius:999px;padding:1px 8px;font-size:10px;font-weight:600;'>"
        f"{_FEATURE_CATEGORY.get(f['feature'], 'Model Feature')}</span>"
        f"<span style='background:rgba(52,211,153,0.15);color:{ui.EMERALD};border-radius:999px;padding:1px 8px;font-size:10px;font-weight:600;'>Selected by ML</span></div>"
        f"<p style='margin:6px 0 0;font-size:12px;color:{ui.MUTED};line-height:1.5;max-width:640px;'>"
        f"{_FEATURE_DESCRIPTIONS.get(f['feature'], 'Automatically selected predictive feature.')}</p></div></div>"
        f"<div style='text-align:right;'><div style='font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:{ui.MUTED}'>ML Learned Weight</div>"
        f"<div style='font-size:18px;font-weight:700;color:{ui.CYAN}'> {weight_pct}%</div></div></div>"
        f"<div style='margin-top:10px;'>{ui.bar(bar_pct, ui.CYAN)}</div></div>"
    )


_FEATURE_CATEGORY = {
    "eloDiff": "Team Strength",
    "winPctDiff": "Team Strength",
    "formDiff": "Recent Form",
    "restDiff": "Schedule",
    "injuryDiff": "Roster",
    "homeField": "Context",
    "spFipDiff": "Starting Pitcher",
    "spEraDiff": "Starting Pitcher",
    "opsDiff": "Hitting",
    "teamEraDiff": "Pitching Staff",
    "defEffDiff": "Defense",
    "parkFactor": "Ballpark",
    "tempDev": "Weather",
    "windMph": "Weather",
    "lineupKnown": "Lineup",
    "lineupOpsDiff": "Lineup",
}

_FEATURE_DESCRIPTIONS = {
    "eloDiff": "Chronological Elo rating gap, adjusted for home field and margin of victory.",
    "winPctDiff": "Season-to-date win-percentage differential between the two clubs.",
    "formDiff": "Last-10-game win-rate differential capturing current momentum.",
    "restDiff": "Days-of-rest advantage, accounting for fatigue and bullpen availability.",
    "injuryDiff": "Injured-list count edge from the latest roster snapshots.",
    "homeField": "Home-field advantage term.",
    "spFipDiff": "Fielding-independent pitching edge (FIP/xERA-style), computed from strikeouts, walks and home runs — strips fielding luck and BABIP variance to quantify true run-prevention expectation.",
    "spEraDiff": "Season ERA differential between the two projected starting pitchers.",
    "opsDiff": "Season-to-date team OPS edge — a consolidated measure of on-base and slugging production for the projected lineups.",
    "teamEraDiff": "Season-to-date team ERA edge, capturing rotation and bullpen run prevention beyond the two starters.",
    "defEffDiff": "Season fielding-percentage edge (defensive-efficiency proxy) between the two clubs.",
    "parkFactor": "Home ballpark run factor — values above 1 favor hitters and inflate expected totals.",
    "tempDev": "Game-time temperature deviation from 72°F, a proxy for air density and carry.",
    "windMph": "Game-time wind speed in mph, affecting fly-ball carry and scoring environment.",
    "lineupKnown": "Whether actual lineup data is available for this game.",
    "lineupOpsDiff": "Weighted mean OPS of the actual starting nine — home minus away.",
}


def _automl_panel(ms: dict) -> None:
    candidates = ms.get("candidates") or []
    selected = next((c for c in candidates if c.get("selected")), candidates[0] if candidates else None)
    ensemble_auc = selected["auc"] if selected else ms.get("auc", 0)
    ensemble_brier = selected["brier"] if selected else ms.get("brier", 0)
    spearman = ms.get("spearmanRho") or 0
    top_decile = ms.get("topDecileWinRate") or 0

    st.markdown(
        f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:18px;'>"
        f"<div style='display:flex;align-items:flex-start;justify-content:space-between;gap:14px;flex-wrap:wrap;'>"
        f"<div style='flex:1;min-width:280px;'>"
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;'>"
        f"{ui.pill('✨ Automated Machine Learning (Auto-ML)', ui.CYAN, 'rgba(34,211,238,0.15)')}"
        f"{ui.pill('5-Fold Cross-Validated', ui.EMERALD, 'rgba(52,211,153,0.15)')}</div>"
        f"<h3 style='margin:0;font-size:17px;font-weight:700;color:{ui.TEXT}'>Empirical Feature Selection &amp; Model Stacking Optimizer</h3>"
        f"<p style='margin:8px 0 0;font-size:13px;color:{ui.MUTED};line-height:1.6;max-width:720px;'>"
        f"Machine Learning algorithms automatically assess feature predictive signal (via L2-regularized logistic regression "
        f"with greedy backward elimination) and solve for optimal ensemble weights by minimizing calibration-set Brier loss to "
        f"maximize out-of-sample AUC (&gt; 0.70 floor) while enforcing monotonic probability calibration.</p></div></div></div>",
        unsafe_allow_html=True,
    )
    if st.button("🔄 Run Auto-ML Optimization", type="primary", key="automl_btn"):
        # Streamlit forbids progress bars / toasts inside on_click callbacks,
        # so defer the full pipeline to the main flow via a session flag.
        st.session_state.run_automl = True

    cards = [
        ("Ensemble AUC-ROC", f"{ensemble_auc:.3f}", ui.CYAN,
         "> 0.75 Goal Met" if ensemble_auc >= 0.75 else "Below 0.75 goal",
         ui.EMERALD if ensemble_auc >= 0.75 else ui.AMBER,
         "High single-game MLB separation"),
        ("Ensemble Brier Loss", f"{ensemble_brier:.3f}", ui.EMERALD,
         f"{ensemble_brier - 0.25:+.3f} vs naive baseline", ui.EMERALD,
         "Lower is better (risk)"),
        ("Rank Correlation (Spearman ρ)", f"{spearman:.3f}", ui.PURPLE,
         "Strict monotonic fidelity", "#d8b4fe", "Probability vs outcome ordering"),
        ("Top Decile Win Rate (>65%)", fmt_pct(top_decile, 1), ui.AMBER,
         "Highest confidence picks", "#fcd34d", "Win rate of >65% favorites"),
    ]
    cols = st.columns(4)
    for col, (label, value, color, sub, sub_color, note) in zip(cols, cards):
        with col:
            st.markdown(
                f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:16px;'>"
                f"<div style='font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:{ui.MUTED}'>{label}</div>"
                f"<div style='margin-top:8px;font-size:26px;font-weight:700;color:{color};font-variant-numeric:tabular-nums;'>{value}</div>"
                f"<div style='margin-top:4px;font-size:12px;font-weight:500;color:{sub_color}'>{sub}</div>"
                f"<div style='margin-top:4px;font-size:11px;color:{ui.MUTED}'>{note}</div></div>",
                unsafe_allow_html=True,
            )

    features = ms.get("featureImportances") or []
    active = [f for f in features if f.get("active") is not False]
    inactive = [f for f in features if f.get("active") is False]
    total_weight = sum(abs(f["weight"]) for f in active) or 1
    max_imp = max((f["importance"] for f in active), default=1e-6)
    sorted_features = sorted(active, key=lambda f: -f["importance"])

    _automl_options = [
        f"Learned Feature Decisions ({len(active)}/{len(features)} Active)",
        f"Optimal Model Stacking Weights ({len(candidates)} Models)",
        "Optimization Parameters",
        f"Cross-Validation on {fmt_number(ms.get('gamesTrained', 0))} games",
    ]
    sub = st.segmented_control("Auto-ML detail", _automl_options, key="automl_sub", default=_automl_options[0])
    if not sub:
        # segmented_control returns None until the user interacts with it;
        # never call methods on it before this guard.
        sub = _automl_options[0]

    if sub.startswith("Learned Feature Decisions"):
        st.info(
            "**How Machine Learning Decided Feature Inclusion & Weights:** "
            "L2-regularized logistic regression with greedy backward elimination evaluated each candidate "
            "feature set. Features were retained only when they measurably reduced calibration-set Brier loss; "
            "the final coefficients are the learned weights."
        )
        if inactive:
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.02);border:1px solid {ui.BORDER};border-radius:12px;"
                f"padding:10px 14px;font-size:12px;color:{ui.MUTED};margin-bottom:10px;'>"
                f"<b style='color:{ui.TEXT}'>Dropped by ML:</b> {' · '.join(f['label'] for f in inactive)}</div>",
                unsafe_allow_html=True,
            )
        for i, f in enumerate(sorted_features):
            st.markdown(_feature_item(i + 1, f, total_weight, max_imp), unsafe_allow_html=True)
        if not sorted_features:
            st.caption("No feature decisions yet — run Auto-ML optimization.")
    elif sub.startswith("Optimal Model Stacking"):
        weights = ms.get("stackingWeights") or []
        rows = []
        for c in candidates:
            w = next((x["weight"] for x in weights if x["name"] == c["name"]), (1 if c.get("selected") else 0))
            rows.append(
                f"<div style='margin-bottom:12px;'>"
                f"<div style='display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;'>"
                f"<div style='display:flex;align-items:center;gap:8px;'>"
                f"<span style='font-size:13px;font-weight:500;color:{ui.TEXT}'>{c['name']}</span>"
                + (ui.pill("Best single", ui.EMERALD, "rgba(52,211,153,0.15)") if c.get("selected") else "")
                + (ui.pill("Excluded", ui.AMBER, "rgba(252,211,77,0.12)") if not c.get("eligible") else "")
                + f"</div>"
                f"<div style='display:flex;gap:12px;font-size:12px;color:{ui.MUTED};font-variant-numeric:tabular-nums;'>"
                f"<span>AUC {c['auc']:.3f}</span><span>Brier {c['brier']:.3f}</span>"
                f"<b style='color:{ui.CYAN}'>{round(w * 100)}%</b></div></div>"
                f"<div style='margin-top:6px;'>{ui.bar(w * 100, ui.CYAN, '8px')}</div></div>"
            )
        st.markdown(
            f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:18px;'>"
            f"<h3 style='margin:0;font-size:14px;font-weight:600;color:{ui.TEXT}'>Optimal Model Stacking Weights</h3>"
            f"<p style='margin:6px 0 14px;font-size:12px;color:{ui.MUTED};line-height:1.5;max-width:760px;'>"
            f"Greedy forward-selection solves for convex-combination weights that minimize calibration-set Brier loss. "
            f"Only models that measurably reduce risk are added to the stack; the remainder carry zero weight.</p>"
            + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )
    elif sub.startswith("Optimization Parameters"):
        params = ms.get("optimizationParams") or {}
        param_rows = [
            ("Feature selection", params.get("featureSelection", "—")),
            ("Candidate pool floor", f"AUC ≥ {params.get('minCandidateAuc', '—')}"),
            ("Regularization", f"L2 λ = {params.get('l2Lambda', '—')}"),
            ("Optimizer", f"Newton–Raphson (IRLS) · {params.get('epochs', '—')} iterations"),
            ("Home-field grid", ", ".join(str(x) for x in params.get("hfaGrid", []))),
            ("Stacking blend step", str(params.get("blendStep", "—"))),
            ("Monte Carlo σ grid", ", ".join(str(x) for x in params.get("mcSigmaGrid", []))),
            ("Calibration", params.get("isotonicMethod", "—")),
            ("Cross-validation folds", str(params.get("cvFolds", "—"))),
        ]
        table_rows = [
            [f"<span style='color:{ui.MUTED}'>{label}</span>", f"<b style='color:{ui.TEXT};font-variant-numeric:tabular-nums;'>{value}</b>"]
            for label, value in param_rows
        ]
        st.markdown(
            f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:18px;'>"
            f"<h3 style='margin:0;font-size:14px;font-weight:600;color:{ui.TEXT}'>Optimization Parameters</h3>"
            f"<p style='margin:6px 0 12px;font-size:12px;color:{ui.MUTED}'>Hyperparameters and search grids used by the Auto-ML optimizer on the most recent training run.</p>"
            + ui.html_table(["Parameter", "Value"], table_rows, align=["left", "right"]) + "</div>",
            unsafe_allow_html=True,
        )
    else:  # Cross-validation
        cv = ms.get("crossValidation") or {}
        if not cv:
            st.caption("Cross-validation metrics will appear after the next Auto-ML run.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:16px;'>"
                    f"<div style='font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:{ui.MUTED}'>Mean AUC</div>"
                    f"<div style='margin-top:6px;font-size:22px;font-weight:700;color:{ui.CYAN}'>{cv.get('aucMean', 0):.3f} ± {cv.get('aucStd', 0):.3f}</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:16px;'>"
                    f"<div style='font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:{ui.MUTED}'>Mean Brier</div>"
                    f"<div style='margin-top:6px;font-size:22px;font-weight:700;color:{ui.EMERALD}'>{cv.get('brierMean', 0):.3f} ± {cv.get('brierStd', 0):.3f}</div></div>",
                    unsafe_allow_html=True,
                )
            cv_rows = [
                [f"<span style='color:{ui.TEXT};font-weight:500'>Fold {i + 1}</span>",
                 f"<span style='color:{ui.MUTED};font-variant-numeric:tabular-nums'>{fmt_number(cv.get('gamesPerFold', [])[i] if i < len(cv.get('gamesPerFold', [])) else 0)}</span>",
                 f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{auc:.3f}</span>",
                 f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{cv.get('foldBriers', [])[i]:.3f}" if i < len(cv.get('foldBriers', [])) else "—</span>"]
                for i, auc in enumerate(cv.get("foldAucs", []))
            ]
            st.markdown(
                f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:18px;margin-top:10px;'>"
                f"<h3 style='margin:0;font-size:14px;font-weight:600;color:{ui.TEXT}'>{cv.get('folds', 0)}-Fold Cross-Validation ({fmt_number(ms.get('gamesTrained', 0))} games)</h3>"
                f"<p style='margin:6px 0 12px;font-size:12px;color:{ui.MUTED};line-height:1.5'>Walk-forward folds train only on prior data so out-of-sample AUC and Brier are never inflated by lookahead. Reported mean ± standard deviation across folds.</p>"
                + ui.html_table(["Fold", "Games", "AUC", "Brier"], cv_rows, align=["left", "right", "right", "right"]) + "</div>",
                unsafe_allow_html=True,
            )


def _pfi_panel(ms: dict) -> None:
    features = sorted(ms.get("featureImportances") or [], key=lambda f: -f["univariateAuc"])
    max_w = max((abs(f["weight"]) for f in features), default=1e-6)
    rows_html = []
    for i, f in enumerate(features):
        bar_pct = max(3.0, min(100.0, abs(f["weight"]) / max_w * 100))
        rows_html.append(
            f"<div style='margin-bottom:12px;'>"
            f"<div style='display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;'>"
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<span style='font-size:12px;font-weight:700;color:{ui.MUTED}'>#{i + 1}</span>"
            f"<span style='font-size:13px;font-weight:500;color:{ui.TEXT}'>{f['label']}</span>"
            f"<span style='background:rgba(77,125,255,0.15);color:{ui.ACCENT};border-radius:999px;padding:1px 8px;font-size:10px;font-weight:600;'>"
            f"{_FEATURE_CATEGORY.get(f['feature'], 'Model Feature')}</span></div>"
            f"<div style='display:flex;gap:12px;font-size:12px;color:{ui.MUTED};font-variant-numeric:tabular-nums;'>"
            f"<span>PFI AUC {f['univariateAuc']:.3f}</span><b style='color:{ui.CYAN}'>w = {f['weight']:.3f}</b></div></div>"
            f"<div style='margin-top:6px;'>{ui.bar(bar_pct, ui.CYAN, '8px')}</div></div>"
        )
    st.markdown(
        f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:18px;'>"
        f"<h3 style='margin:0;font-size:14px;font-weight:600;color:{ui.TEXT}'>Feature Importance (Permutation Feature Importance)</h3>"
        f"<p style='margin:6px 0 14px;font-size:12px;color:{ui.MUTED};line-height:1.5'>Each feature is ranked by its isolated out-of-sample predictive signal (univariate AUC) and its learned coefficient in the logistic ensemble. Bars show the standardized coefficient magnitude.</p>"
        + "".join(rows_html) + "</div>",
        unsafe_allow_html=True,
    )


def _ensemble_panel(ms: dict) -> None:
    candidates = ms.get("candidates") or []
    steps = [
        f"{len(ms.get('featureImportances') or ms.get('featureNames') or [])} Features",
        f"{max(1, len([c for c in candidates if c.get('eligible')]))} Candidate Models",
        "Stacked Ensemble",
        "Isotonic Calibration",
        "Monte Carlo",
        "Win Probability",
    ]
    steps_html = " ".join(
        f"<span style='background:rgba(255,255,255,0.02);border:1px solid {ui.BORDER};border-radius:10px;padding:6px 12px;font-size:12px;font-weight:500;color:{ui.TEXT}'>{s}</span>"
        + ("<span style='color:#8b939f'> → </span>" if i < len(steps) - 1 else "")
        for i, s in enumerate(steps)
    )
    st.markdown(
        f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:18px;'>"
        f"<h3 style='margin:0;font-size:14px;font-weight:600;color:{ui.TEXT}'>Ensemble Architecture</h3>"
        f"<p style='margin:6px 0 14px;font-size:12px;color:{ui.MUTED};line-height:1.5'>{ms.get('modelDescription', '')}</p>"
        f"<div style='display:flex;flex-wrap:wrap;gap:6px;align-items:center;'>{steps_html}</div></div>",
        unsafe_allow_html=True,
    )

    cand_rows = []
    for c in candidates:
        if c.get("selected"):
            status = ui.pill("Selected", ui.EMERALD, "rgba(52,211,153,0.15)")
        elif not c.get("eligible"):
            status = ui.pill("Excluded (<0.70)", ui.AMBER, "rgba(252,211,77,0.12)")
        else:
            status = "<span style='color:#8b939f;font-size:12px'>—</span>"
        cand_rows.append([
            f"<b style='color:{ui.TEXT}'>{c['name']}</b>",
            f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{c['auc']:.3f}</span>",
            f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{c['brier']:.3f}</span>",
            f"<span style='color:{ui.MUTED};font-variant-numeric:tabular-nums'>{c['logLoss']:.3f}</span>",
            f"<span style='color:{ui.MUTED};font-variant-numeric:tabular-nums'>{c['ece']:.3f}</span>",
            status,
        ])
    st.markdown(
        f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:18px;margin-top:12px;'>"
        f"<h3 style='margin:0 0 12px;font-size:14px;font-weight:600;color:{ui.TEXT}'>Candidate Models (Cross-Validated)</h3>"
        + ui.html_table(
            ["Model", "AUC", "Brier", "Log-Loss", "ECE", "Status"],
            cand_rows,
            align=["left", "right", "right", "right", "right", "right"],
        ) + "</div>",
        unsafe_allow_html=True,
    )

    mc = ms.get("monteCarloEnabled")
    mc_color = ui.EMERALD if mc else ui.TEXT
    mc_bg = "rgba(52,211,153,0.10)" if mc else "rgba(255,255,255,0.02)"
    mc_border = "rgba(52,211,153,0.25)" if mc else ui.BORDER
    mc_line = (
        f"Enabled — σ = {ms.get('monteCarloSigma', 0):.2f} (Gaussian logit-noise expectation)"
        if mc else "Disabled — deterministic point estimates"
    )
    st.markdown(
        f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:18px;margin-top:12px;'>"
        f"<h3 style='margin:0;font-size:14px;font-weight:600;color:{ui.TEXT}'>Stochastic Component (Monte Carlo)</h3>"
        f"<div style='display:flex;align-items:center;gap:8px;border:1px solid {mc_border};background:{mc_bg};border-radius:10px;padding:10px 12px;margin-top:10px;'>"
        f"<span style='color:{mc_color};font-weight:600;font-size:13px'>{'✓' if mc else '⚠'}</span>"
        f"<span style='color:{mc_color};font-size:13px;font-weight:500'>{mc_line}</span></div>"
        f"<p style='margin:10px 0 0;font-size:13px;color:{ui.MUTED};line-height:1.6'>{ms.get('monteCarloRationale', '')}</p></div>",
        unsafe_allow_html=True,
    )


def monitor_tab(bundle) -> None:
    ms = bundle["model_state"]
    record = ms.get("todaysRecord") or {}
    ui.section(
        "Auto-ML & Model Monitor",
        "Automated feature selection, regularized weighting, model ensemble optimization (AUC > 0.60), and calibration diagnostics.",
    )

    sub = st.segmented_control(
        "Monitor view",
        ["Auto-ML Selection & Weights", "Feature Importance (PFI)", "Ensemble Architecture"],
        key="mon_sub",
        default="Auto-ML Selection & Weights",
    )
    if not sub:
        sub = "Auto-ML Selection & Weights"
    if sub == "Auto-ML Selection & Weights":
        _automl_panel(ms)
    elif sub == "Feature Importance (PFI)":
        _pfi_panel(ms)
    else:
        _ensemble_panel(ms)

    # Drift section
    st.markdown(f"<div style='height:18px'></div>", unsafe_allow_html=True)
    ui.section("Model & Data Drift Monitor", "Tracking model health, feature drift, and performance over time")

    drift = ms.get("featureDrift") or []
    warns = [d for d in drift if d["status"] == "WARN"]
    first_warn = warns[0] if warns else None
    trained_at = ms.get("trainedAt", 0)
    now_ms = int(_dt.datetime.now().timestamp() * 1000)
    days_ago = max(0, (now_ms - trained_at) // 86400000) if trained_at else 0
    next_retrain = trained_at + 86400000
    cols = st.columns(3)
    for col, dot, label, value, sub_text, sub_color in [
        (cols[0], ui.EMERALD, "Last Retrain", fmt_trained_at(trained_at),
         f"Model healthy — {'today' if days_ago == 0 else f'{days_ago} day' + ('s' if days_ago != 1 else '') + ' ago'}", ui.EMERALD),
        (cols[1], ui.ACCENT, "Next Retrain", fmt_trained_at(next_retrain),
         "Nightly schedule — tonight", ui.MUTED),
        (cols[2], ui.AMBER if warns else ui.EMERALD, "Drift Alerts", f"{len(warns)} Warning" + ("s" if len(warns) != 1 else ""),
         f"{first_warn['label']} — elevated PSI" if first_warn else "All features stable",
         ui.AMBER if first_warn else ui.EMERALD),
    ]:
        with col:
            st.markdown(ui.stat_card(dot, label, value, sub_text, sub_color), unsafe_allow_html=True)

    upsets = record.get("upsets") or []
    if upsets or record:
        upset_text = ", ".join(f"{u['team']} over {u['loser']} at {u['prob']}%" for u in upsets)
        st.markdown(
            f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:16px;margin-top:12px;'>"
            f"<div style='font-size:13px;font-weight:700;color:{ui.AMBER}'>⚠ Upset Monitoring Note — {short_date(record.get('date') or ms.get('asOfDate', ''))}</div>"
            f"<p style='margin:8px 0 0;font-size:13px;color:{ui.MUTED};line-height:1.6'>"
            + (
                f"{len(upsets)} upset{'s' if len(upsets) != 1 else ''} today ({upset_text}) — monitoring for regime shift. "
                f"Model went {record.get('wins', 0)}-{record.get('losses', 0)} overall but high-confidence picks (>65%) showed vulnerability. "
                f"Will assess after tonight's retrain."
                if upsets
                else "No upsets recorded — monitoring for regime shift on tonight's retrain."
            )
            + "</p></div>",
            unsafe_allow_html=True,
        )

    drift_rows = [
        [
            f"<span style='color:{ui.TEXT};font-weight:500'>{d['label']}</span>",
            f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{d['currentMean']:.3f}</span>",
            f"<span style='color:{ui.MUTED};font-variant-numeric:tabular-nums'>{d['baselineMean']:.3f}</span>",
            f"<span style='color:{ui.AMBER if d['status'] == 'WARN' else ui.TEXT};font-variant-numeric:tabular-nums'>{d['psi']:.3f}</span>",
            ui.pill(d["status"], ui.AMBER if d["status"] == "WARN" else ui.EMERALD,
                    "rgba(252,211,77,0.15)" if d["status"] == "WARN" else "rgba(52,211,153,0.15)"),
        ]
        for d in drift
    ]
    st.markdown(
        f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:18px;margin-top:12px;'>"
        f"<h3 style='margin:0 0 12px;font-size:14px;font-weight:600;color:{ui.TEXT}'>Feature Drift Analysis (PSI Scores)</h3>"
        + (ui.html_table(["Feature", "Current Mean", "Baseline Mean", "PSI", "Status"], drift_rows,
                          align=["left", "right", "right", "right", "right"]) if drift_rows
           else "<div style='color:#8b939f;font-size:13px;padding:20px 0;text-align:center'>No drift data yet.</div>")
        + "</div>",
        unsafe_allow_html=True,
    )

    rolling = ms.get("rollingBrier") or []
    baseline = ms.get("brierBaseline") or ms.get("brier") or 0
    st.markdown(
        f"<div style='display:flex;align-items:center;justify-content:space-between;margin-top:14px;'>"
        f"<span style='font-size:14px;font-weight:600;color:{ui.TEXT}'>Rolling Brier Score (Last 30 Days)</span>"
        f"<span style='display:flex;gap:14px;'>{ui.legend(ui.ORANGE, 'Brier Score')}{ui.legend(ui.MUTED, 'Baseline (prior version)', dashed=True)}</span></div>",
        unsafe_allow_html=True,
    )
    if not rolling:
        st.markdown(
            f"<div style='display:flex;align-items:center;justify-content:center;height:200px;color:{ui.MUTED};font-size:13px;'>No rolling risk data yet.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.plotly_chart(rolling_brier_chart(rolling, baseline), use_container_width=True)

    versions = ms.get("modelVersions") or []
    version_rows = [
        [
            f"<b style='color:{ui.CYAN}'>{v['version']}</b>",
            f"<span style='color:{ui.MUTED}'>{short_date(v['date'])}</span>",
            f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{v['auc']:.3f}</span>",
            f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{v['brier']:.3f}</span>",
            f"<span style='color:{ui.MUTED}'>{v['notes']}</span>",
        ]
        for v in versions
    ]
    st.markdown(
        f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:18px;margin-top:14px;'>"
        f"<h3 style='margin:0 0 12px;font-size:14px;font-weight:600;color:{ui.TEXT}'>Model Version History</h3>"
        + (ui.html_table(["Version", "Date", "AUC", "Brier", "Notes"], version_rows,
                          align=["left", "left", "right", "right", "left"]) if version_rows
           else "<div style='color:#8b939f;font-size:13px;padding:20px 0;text-align:center'>No version history yet.</div>")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        f"Model: **{ms.get('selectedModel', '')}** · {len(ms.get('featureNames') or [])} features selected · "
        f"Monte Carlo {'enabled' if ms.get('monteCarloEnabled') else 'disabled'} · "
        f"Data: **statsapi.mlb.com** (single consolidated source)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if "bundle" not in st.session_state:
        st.session_state.bundle = load_bundle()

    # Auto-ML button sets this flag (callbacks cannot render a progress bar);
    # run the pipeline here in the main flow, then rerun to show fresh results.
    if st.session_state.get("run_automl", False):
        st.session_state["run_automl"] = False
        do_refresh()
        return

    # Header Refresh button — same flow as the Auto-ML flag.
    if st.session_state.get("refresh_requested", False):
        del st.session_state["refresh_requested"]
        do_refresh()
        return

    active_tab = st.session_state.get("active_tab", "games")
    bundle = st.session_state.bundle

    render_header(active_tab)

    if bundle is None:
        render_empty_state()
        return

    if active_tab == "games":
        games_tab(bundle)
    elif active_tab == "rankings":
        rankings_tab(bundle)
    elif active_tab == "calibration":
        calibration_tab(bundle)
    else:
        monitor_tab(bundle)


main()
