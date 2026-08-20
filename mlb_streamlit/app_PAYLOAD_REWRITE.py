"""MLB Predictions — Streamlit dashboard (CDN/Payload mode).

This version loads ALL data from a single pre-computed JSON payload
(generated headlessly in Google Colab). It contains ZERO model training,
ZERO threading, ZERO background warming. The entire file is a pure
presentation layer.

Payload file: cache/dashboard_payload.json

Run:  streamlit run mlb_streamlit/app.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from mlb_streamlit import ui
from mlb_streamlit.engine.markets import (
    MARKET_ARCHITECTURE_METADATA,
    MARKET_TYPES,
    expand_market_rows,
)
from mlb_streamlit.engine.metrics import (
    calibration_curve_points,
    compute_auc,
    compute_brier,
    evaluate,
)
from mlb_streamlit.engine.model import CANDIDATE_MIN_AUC
from mlb_streamlit.engine.teams import team_meta
from mlb_streamlit.engine.backtest import build_execution_backtest

# ---------------------------------------------------------------------------
# Payload loading — the ONLY data source
# ---------------------------------------------------------------------------

PAYLOAD_FILE = Path(__file__).resolve().parent / "cache" / "dashboard_payload.json"


@st.cache_data(ttl=3600, show_spinner=False)
def load_payload() -> dict | None:
    """Load the pre-computed dashboard payload from disk.

    Cached for 1 hour via Streamlit's native cache. The payload is produced
    by the Colab pipeline (export_dashboard_payload.py) and contains:
      modelState, gamesByDate, calibrationRows, calibrationRowsWf,
      calibrationBins, calibrationCurve, confidenceDistribution,
      totalsMetrics, runLineMetrics, moneylineTotal/Correct/Accuracy.
    """
    if not PAYLOAD_FILE.exists():
        return None
    try:
        with open(PAYLOAD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _invalidate_payload_cache() -> None:
    """Drop the Streamlit cache so the next read picks up a fresh payload."""
    load_payload.clear()


# ---------------------------------------------------------------------------
# Engine version footer (no git dependency — reads from payload)
# ---------------------------------------------------------------------------

def _render_engine_footer(payload: dict | None) -> None:
    """Sidebar footer showing engine version from the payload metadata."""
    ms = (payload or {}).get("modelState") or {}
    version = ms.get("WF_SELECTION_VERSION", "?")
    trained_at = ms.get("trainedAt")
    trained_str = ""
    if trained_at:
        try:
            d = _dt.datetime.fromtimestamp(trained_at / 1000)
            trained_str = f" · Trained {d.strftime('%b %d, %Y')}"
        except (TypeError, ValueError, OSError):
            pass
    st.sidebar.caption(f"Engine Version: v{version}{trained_str}")


# ---------------------------------------------------------------------------
# Page setup + theme CSS (identical to original)
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
[data-testid="stHorizontalBlock"]:first-of-type {
  position: sticky !important; top: 0; z-index: 100;
  background: rgba(10,13,18,0.85); backdrop-filter: blur(8px);
  border-bottom: 1px solid rgba(255,255,255,0.07); padding: 10px 2px; margin-bottom: 16px;
}
div[class*="st-key-nav_"] { display: flex; justify-content: center; }
div[class*="st-key-nav_"] button {
  background: transparent !important; border: none !important; border-radius: 8px;
  color: #8b939f; font-size: 14px; font-weight: 500; padding: 6px 10px; box-shadow: none !important;
}
div[class*="st-key-nav_"] button:hover { color: #e5e8ec !important; background: rgba(255,255,255,0.04) !important; }
div[class*="st-key-nav_active_"] button {
  color: #e5e8ec !important; border-bottom: 2px solid #427ff7 !important; border-radius: 0 !important;
}
div[class*="st-key-header_refresh"] { display: flex; justify-content: flex-end; }
div[class*="st-key-header_refresh"] button {
  border: 1px solid rgba(255,255,255,0.09) !important; background: #12161c !important;
  border-radius: 8px; color: #e5e8ec; font-size: 12px; font-weight: 500; padding: 6px 12px; box-shadow: none !important;
}
div[class*="st-key-header_refresh"] button:hover { border-color: rgba(255,255,255,0.18) !important; }
div[class*="st-key-date_prev"], div[class*="st-key-date_next"] { display: flex; justify-content: center; }
div[class*="st-key-date_prev"] button, div[class*="st-key-date_next"] button {
  width: 32px !important; min-width: 32px !important; height: 32px !important;
  padding: 0 !important; border: 1px solid rgba(255,255,255,0.09) !important;
  background: #12161c !important; border-radius: 8px !important; color: #8b939f !important;
  font-size: 16px !important; display: inline-flex; align-items: center; justify-content: center;
  box-shadow: none !important;
}
div[class*="st-key-date_prev"] button:hover, div[class*="st-key-date_next"] button:hover {
  color: #e5e8ec !important; border-color: rgba(255,255,255,0.2) !important;
}
button[data-testid="stPopoverButton"] {
  border-radius: 999px; border: 1px solid rgba(59,130,246,0.30);
  background: rgba(59,130,246,0.10); color: #e5e8ec; font-weight: 500;
  font-size: 14px; padding: 8px 20px; white-space: nowrap; box-shadow: none !important;
}
button[data-testid="stPopoverButton"]:hover { border-color: rgba(59,130,246,0.50) !important; }
div[data-testid="stSegmentedControl"] > label { display: none; }
div[data-testid="stSegmentedControl"] { gap: 6px; }
div[data-testid="stSegmentedControl"] button {
  border-radius: 9999px; font-size: 12px; font-weight: 600;
  border: 1px solid rgba(255,255,255,0.09); background: #12161c; color: #8b939f;
}
div[data-testid="stSegmentedControl"] button:hover { color: #e5e8ec; }
div[data-testid="stSegmentedControl"] button[aria-checked="true"],
div[data-testid="stSegmentedControl"] button[aria-selected="true"] {
  background: #427ff7 !important; border-color: #427ff7 !important; color: #fff !important;
}
div[class*="st-key-mon_sub"] [data-testid="stSegmentedControl"] button[aria-checked="true"],
div[class*="st-key-automl_sub"] [data-testid="stSegmentedControl"] button[aria-checked="true"] {
  background: #22d3ee !important; border-color: #22d3ee !important; color: #083344 !important;
}
div[class*="st-key-automl_btn"] button[kind="primary"] {
  background: #22d3ee; border-color: #22d3ee; color: #083344;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

ET = _dt.timezone(_dt.timedelta(hours=-4))


def _et_zone(now: _dt.datetime) -> _dt.tzinfo:
    return _dt.timezone(_dt.timedelta(hours=-4 if 4 <= now.month <= 10 else -5))


# ---------------------------------------------------------------------------
# Formatting helpers (unchanged)
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
# Charts (plotly) — unchanged
# ---------------------------------------------------------------------------

def _base_layout(height: int = 300) -> dict:
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ui.TEXT, size=12),
        margin=dict(l=10, r=14, t=10, b=10),
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
        xaxis=dict(title="Mean Predicted Probability", range=[0.45, 0.85], tickformat=".2f", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Mean Actual Win Rate", range=[0.45, 0.85], tickformat=".2f", gridcolor="rgba(255,255,255,0.06)"),
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
        mode="lines", line=dict(color=ui.MUTED, dash="dash", width=1), name="Baseline",
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
# Header (no refresh button — data is static from payload)
# ---------------------------------------------------------------------------

TAB_IDS = [("games", "Today's Games"), ("rankings", "Power Rankings"),
           ("calibration", "Calibration"), ("monitor", "Model Monitor")]


def render_header(active_tab: str) -> None:
    cols = st.columns([2.2, 1.7, 1.6, 1.25, 1.6], vertical_alignment="center")
    with cols[0]:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;'>"
            f"{ui.baseball_mark()}"
            f"<span style='font-size:16px;font-weight:700;color:{ui.TEXT};letter-spacing:-.01em;'>MLB Predictions</span></div>",
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
    if active_tab == "monitor":
        bundle = st.session_state.get("bundle") or {}
        ms = bundle.get("modelState") or {}
        cfg = ms.get("concordanceGate") or {}
        diag = ms.get("concordanceGateDiagnostics") or {}
        st.markdown(
            f"<div style='margin:4px 0 14px;background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:12px;padding:10px 14px;'>"
            f"<span style='font-size:11px;font-weight:700;color:{ui.CYAN};text-transform:uppercase;letter-spacing:.08em;'>Concordance gate</span>"
            f"<span style='font-size:12px;color:{ui.TEXT};margin-left:10px;'>"
            f"{'active' if cfg.get('enabled') else 'held out'} · threshold {float(cfg.get('threshold', .75)):.0%} · "
            f"{fmt_pct(diag.get('winRate', 0), 1)} conditional win rate · "
            f"{fmt_pct(diag.get('coverage', 0), 1)} coverage · "
            f"{diag.get('accepted', 0)} accepted of {diag.get('total', 0)}</span></div>",
            unsafe_allow_html=True,
        )


def render_empty_state() -> None:
    st.markdown("<div style='height:12vh'></div>", unsafe_allow_html=True)
    c = st.columns([1, 2, 1])[1]
    with c:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='display:inline-flex;width:56px;height:56px;border-radius:16px;align-items:center;justify-content:center;"
            f"border:1px solid {ui.BORDER};background:{ui._card_bg()};font-size:26px;'>⚾</div>"
            f"<h2 style='margin-top:18px;font-size:20px;font-weight:700;color:{ui.TEXT}'>MLB Predictions Dashboard</h2>"
            f"<p style='margin:10px auto 0;max-width:440px;font-size:13px;color:{ui.MUTED};line-height:1.6'>"
            f"Pre-computed model predictions are loaded from the CDN payload. "
            f"No data file was found at the expected path.</p></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 1 — Today's Games
# ---------------------------------------------------------------------------

def _games_for_date(bundle, ymd: str) -> list[dict]:
    return (bundle.get("gamesByDate") or {}).get(ymd, []) or []


def _filter_games(games: list[dict], f: str) -> list[dict]:
    if f == "Final":
        return [g for g in games if g["status"] == "Final"]
    if f == "Live":
        return [g for g in games if g["status"] == "Live"]
    if f == "Upcoming":
        return [g for g in games if g["status"] in ("Preview", "Scheduled")]
    return games


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
    gate_enabled = game.get("gateEnabled") is True
    gate_accepted = game.get("gateAccepted") is True
    gated_team = game.get("gatedPickTeam")
    gated_abbrev = (
        game["home"]["abbrev"] if gated_team == "home"
        else game["away"]["abbrev"] if gated_team == "away" else None
    )

    pills = []
    if is_coin_flip:
        pills.append(ui.pill("Coin flip", ui.AMBER, "rgba(252,211,77,0.15)"))
    if game.get("isUpset"):
        pills.append(ui.pill("Upset", ui.AMBER, "rgba(252,211,77,0.15)"))
    if game.get("isCorrect") is False:
        pills.append(ui.pill("Miss", ui.ROSE, "rgba(251,113,133,0.15)"))
    if game.get("isCorrect") is True:
        pills.append(ui.pill("Correct pick", ui.EMERALD, "rgba(52,211,153,0.15)"))
    if gate_accepted:
        pills.append(ui.pill("Gated pick", ui.EMERALD, "rgba(52,211,153,0.15)"))
    elif gate_enabled:
        pills.append(ui.pill("No gated play", ui.AMBER, "rgba(252,211,77,0.15)"))
    if is_live:
        pills.append(ui.pill("Live", ui.EMERALD, "rgba(52,211,153,0.15)"))
    if is_final:
        inn = f" (F/{game['innings']})" if game.get("innings") else ""
        pills.append(ui.pill(f"Final{inn}", "#5eead4", "rgba(45,212,191,0.12)"))

    with col:
        with st.container(border=True):
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
                model_pill = ui.pill("Model pick", ui.ACCENT, "rgba(77,125,255,0.18)") if is_pick else ""
                gate_pill = ui.pill("Gate pick", ui.EMERALD, "rgba(52,211,153,0.15)") if gated_team == side_key else ""
                pick_pill = model_pill + gate_pill
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

            gate_text = ""
            if gate_enabled:
                gate_text = (
                    f"Concordance gate: <b style='color:{ui.EMERALD}'>ACCEPTED — {gated_abbrev}</b>"
                    if gate_accepted else
                    f"Concordance gate: <b style='color:{ui.AMBER}'>ABSTAIN</b> · "
                    f"{game.get('gateAgreeCount', 0)}/{game.get('gateSignalCount', 0)} signals agree"
                )
            st.markdown(
                f"<div style='text-align:center;font-size:11px;color:{ui.MUTED};padding:2px 0 6px;'>"
                f"Pre-game: {game['home']['abbrev']} {fmt_pct(game['homeWinProb'])} vs "
                f"{game['away']['abbrev']} {fmt_pct(game['awayWinProb'])}"
                f"{(' · ' + gate_text) if gate_text else ''}</div>",
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

            # Bet decision
            bet = game.get("betDecision") or {}
            if bet.get("available"):
                bet_color = ui.EMERALD if bet.get("recommended") else ui.AMBER
                bet_label = "BET" if bet.get("recommended") else "PASS"
                bet_team = bet.get("team") or bet.get("candidateSide")
                bet_abbrev = (
                    game["home"]["abbrev"] if bet_team == "home"
                    else game["away"]["abbrev"] if bet_team == "away" else "best side"
                )
                quoted_odds = bet.get("offeredOdds") if bet.get("offeredOdds") is not None else bet.get("candidateOdds")
                odds_text = fmt_american(quoted_odds) if quoted_odds is not None else "—"
                ev_text = f"{float(bet.get('expectedValue') or 0):+.1%} EV"
                edge_text = f"{float(bet.get('edge') or 0):+.1%} vs no-vig"
                stake_text = f"quarter-Kelly {float(bet.get('recommendedStakeFraction') or 0):.2%} bankroll"
                st.markdown(
                    f"<div style='margin-top:8px;border:1px solid {bet_color};background:rgba(255,255,255,0.025);border-radius:10px;padding:9px 10px;'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<span style='font-size:11px;font-weight:700;letter-spacing:.08em;color:{bet_color}'>MARKET ACTION · {bet_label}</span>"
                    f"<span style='font-size:12px;font-weight:700;color:{ui.TEXT}'>{bet_abbrev} {odds_text}</span></div>"
                    f"<div style='margin-top:4px;font-size:11px;color:{ui.MUTED}'>{ev_text} · {edge_text} · {stake_text}</div>"
                    f"<div style='margin-top:3px;font-size:10px;color:{ui.MUTED}'>{bet.get('reason', '')}</div></div>",
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
    ms = bundle["modelState"]
    today = ms["asOfDate"]
    season = int(ms["season"])

    if "games_date" not in st.session_state:
        st.session_state.games_date = _dt.date.fromisoformat(today)
    if "games_filter" not in st.session_state:
        st.session_state.games_filter = "All Games"

    ymd = st.session_state.games_date.isoformat()
    games = _games_for_date(bundle, ymd)
    filtered = _filter_games(games, str(st.session_state.games_filter).split(" (")[0])
    night_count = sum(1 for g in games if g["dayNight"] == "night")
    record = ms.get("todaysRecord") or {}

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
            with st.popover(fmt_date_long(ymd)):
                st.date_input(
                    "Game date", key="games_date", min_value=lo, max_value=hi,
                    format="YYYY-MM-DD", label_visibility="collapsed",
                )
        with c_next:
            if st.button("›", key="date_next", help="Next day"):
                st.session_state.games_date = max(lo, min(hi, st.session_state.games_date + _dt.timedelta(days=1)))
                st.rerun()
    with c_r:
        st.markdown("")

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

    if not games:
        st.markdown(
            f"<div style='text-align:center;padding:80px 0;color:{ui.MUTED};font-size:14px;'>"
            f"No games found for {fmt_date_long(ymd)}.</div>",
            unsafe_allow_html=True,
        )
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
# Tab 2 — Power Rankings (unchanged)
# ---------------------------------------------------------------------------

def rankings_tab(bundle) -> None:
    ms = bundle["modelState"]
    today = ms["asOfDate"]
    rankings = ms.get("powerRankings") or []
    label = f"As of {fmt_date_long(today)} · current"
    ui.section("Power Rankings", f"Elo-based power rankings · {label} · All {len(rankings)} teams")
    rows = []
    for i, r in enumerate(rankings):
        meta = team_meta(r["teamId"])
        elo_color = ui.CYAN if i < 5 else ui.AMBER if i < 10 else ui.TEXT
        run_diff = r.get("runDiff") or 0
        rd_color = ui.EMERALD if run_diff > 0 else ui.ROSE if run_diff < 0 else ui.MUTED
        l10_wins = round(r.get("last10WinPct", 0.5) * 10)
        rows.append([
            f"<span style='color:{ui.MUTED};font-variant-numeric:tabular-nums;'>{i + 1}</span>",
            f"<span style='display:inline-block;width:4px;height:18px;border-radius:999px;background:{meta['color']};vertical-align:middle;margin-right:8px;'>"
            f"</span><b style='color:{ui.TEXT}'>{r['name']}</b> <span style='color:{ui.MUTED};font-size:11px'>{r['abbrev']}</span>",
            f"<b style='color:{elo_color};font-variant-numeric:tabular-nums;'>{round(r['elo'])}</b>",
            f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums;'>{r['wins']}-{r['losses']}</span>",
            f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums;'>{r['winPct']:.3f}</span>",
            f"<span style='color:{rd_color};font-variant-numeric:tabular-nums;'>{('+' if run_diff > 0 else '')}{run_diff}</span>",
            f"<span style='color:{ui.MUTED};font-variant-numeric:tabular-nums;'>{l10_wins}-{10 - l10_wins}</span>",
        ])
    st.markdown(
        f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:8px 12px;overflow-x:auto;'>"
        + ui.html_table(
            ["Rank", "Team", "Elo", "W-L", "Pct", "Run Diff", "L10"],
            rows, align=["left", "left", "right", "right", "right", "right", "right"],
        ) + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 3 — Calibration (reads from payload)
# ---------------------------------------------------------------------------

def calibration_tab(bundle) -> None:
    ms = bundle["modelState"]
    rows_in = bundle.get("calibrationRows") or []
    rows_wf = bundle.get("calibrationRowsWf") or []

    if "cal_method" not in st.session_state:
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

    trained_at = fmt_trained_at(ms.get("trainedAt", 0))
    st.markdown(
        f"<h2 style='margin:0;font-size:20px;font-weight:700;letter-spacing:-.01em;color:{ui.TEXT}'>"
        f"Model Calibration Dashboard</h2>",
        unsafe_allow_html=True,
    )
    if method == "Walk-forward (point-in-time)":
        st.markdown(
            f"<div style='margin-top:10px;'>{ui.pill(f'n = {fmt_number(len(rows_wf))} games scored point-in-time', ui.ACCENT, 'rgba(77,125,255,0.15)')}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='margin-top:10px;'>{ui.pill(f'n = {fmt_number(total_games)} games in range · Trained {trained_at} ET', ui.MUTED, 'rgba(255,255,255,0.05)')}</div>",
            unsafe_allow_html=True,
        )

    if not rows:
        st.markdown(
            f"<div style='text-align:center;padding:28px 0;color:{ui.MUTED};font-size:13px;'>"
            f"No calibration data in the payload. Re-export from Colab.</div>",
            unsafe_allow_html=True,
        )
        return

    min_date = _dt.date.fromisoformat(min(r["date"] for r in rows))
    max_date = _dt.date.fromisoformat(max(r["date"] for r in rows))

    if "cal_start" not in st.session_state:
        st.session_state.cal_start = min_date
    if "cal_end" not in st.session_state:
        st.session_state.cal_end = max_date

    st.segmented_control("View", ["Moneyline", "Game Totals", "Run Lines (-1.5 / +1.5)"], key="cal_view", default="Moneyline")
    view = st.session_state.cal_view

    start_ymd = st.session_state.cal_start.isoformat()
    end_ymd = st.session_state.cal_end.isoformat()
    in_range = [r for r in rows if start_ymd <= r["date"] <= end_ymd]

    if not in_range:
        st.markdown(
            f"<div style='text-align:center;padding:28px 0;color:{ui.MUTED};font-size:13px;'>No data in this range.</div>",
            unsafe_allow_html=True,
        )
        return

    if view == "Moneyline":
        preds = [r["pickProb"] for r in in_range]
        labels = [1 if r["isCorrect"] else 0 for r in in_range]
        ev = evaluate(preds, labels)
        curve = calibration_curve_points(preds, labels, 12) or ev["calibrationCurve"]
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
        st.plotly_chart(calibration_scatter(curve, len(in_range)), use_container_width=True)
        st.plotly_chart(confidence_chart(ev["confidenceDistribution"]), use_container_width=True)
    elif view == "Game Totals":
        n = t_abs = t_sq = t_bias = 0
        for r in in_range:
            rp = r.get("runProjection") or {}
            pt = rp.get("total")
            if pt is not None:
                actual = (r.get("away", {}).get("score") or 0) + (r.get("home", {}).get("score") or 0)
                err = pt - actual
                n += 1; t_abs += abs(err); t_sq += err * err; t_bias += err
        cards = [
            ("Mean Abs. Error", f"{t_abs/n:.2f}" if n else "—", ui.CYAN, "Runs"),
            ("RMSE", f"{(t_sq/n)**0.5:.2f}" if n else "—", ui.EMERALD, "Runs"),
            ("Bias", f"{t_bias/n:+.2f}" if n else "—", ui.AMBER, "Predicted − actual"),
            ("Games", fmt_number(n), ui.PURPLE, "With run projection"),
        ]
        cols = st.columns(4)
        for col, (label, value, color, sub) in zip(cols, cards):
            with col:
                st.markdown(ui.metric_card(label, value, color, sub), unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center;padding:28px 0;color:{ui.MUTED};font-size:13px;'>Run-line view coming soon.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 4 — Model Monitor (reads from payload)
# ---------------------------------------------------------------------------

_FEATURE_CATEGORY = {
    "eloDiff": "Team Strength", "winPctDiff": "Team Strength", "formDiff": "Recent Form",
    "restDiff": "Schedule", "injuryDiff": "Roster", "homeField": "Context",
    "spFipDiff": "Starting Pitcher", "spEraDiff": "Starting Pitcher",
    "opsDiff": "Hitting", "teamEraDiff": "Pitching Staff", "defEffDiff": "Defense",
    "parkFactor": "Ballpark", "tempDev": "Weather", "windMph": "Weather",
    "lineupKnown": "Lineup", "lineupOpsDiff": "Lineup",
}


def monitor_tab(bundle) -> None:
    ms = bundle["modelState"]
    ui.section("Auto-ML & Model Monitor", "Model selection, feature importance, and drift diagnostics (from pre-computed payload).")

    sub = st.segmented_control(
        "Monitor view",
        ["Auto-ML Selection & Weights", "Feature Importance (PFI)", "Ensemble Architecture"],
        key="mon_sub",
        default="Auto-ML Selection & Weights",
    ) or "Auto-ML Selection & Weights"

    candidates = ms.get("candidates") or []
    ensemble_auc = ms.get("auc", 0)
    ensemble_brier = ms.get("brier", 0)

    if sub == "Auto-ML Selection & Weights":
        cards = [
            ("Ensemble AUC-ROC", f"{ensemble_auc:.3f}", ui.CYAN, "> 0.75 Goal Met" if ensemble_auc >= 0.75 else "Below 0.75 goal"),
            ("Ensemble Brier Loss", f"{ensemble_brier:.3f}", ui.EMERALD, f"{ensemble_brier - 0.25:+.3f} vs naive baseline"),
        ]
        cols = st.columns(2)
        for col, (label, value, color, sub_text) in zip(cols, cards):
            with col:
                st.markdown(
                    f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:16px;'>"
                    f"<div style='font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:{ui.MUTED}'>{label}</div>"
                    f"<div style='margin-top:8px;font-size:26px;font-weight:700;color:{color};font-variant-numeric:tabular-nums;'>{value}</div>"
                    f"<div style='margin-top:4px;font-size:12px;color:{ui.MUTED}'>{sub_text}</div></div>",
                    unsafe_allow_html=True,
                )
        weights = ms.get("stackingWeights") or []
        for c in candidates:
            w = next((x["weight"] for x in weights if x["name"] == c["name"]), 0.0)
            pills = ""
            if c.get("selected"):
                pills += ui.pill("Best single", ui.EMERALD, "rgba(52,211,153,0.15)")
            if c.get("inStack"):
                pills += ui.pill("In stack", ui.CYAN, "rgba(34,211,238,0.15)")
            if not c.get("eligible"):
                pills += ui.pill("Excluded", ui.AMBER, "rgba(252,211,77,0.12)")
            st.markdown(
                f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:12px;padding:12px;margin-bottom:8px;'>"
                f"<div style='display:flex;align-items:center;justify-content:space-between;'>"
                f"<span style='font-size:13px;font-weight:500;color:{ui.TEXT}'>{c['name']}</span> {pills}"
                f"<span style='font-size:12px;color:{ui.CYAN};font-weight:600;'>{round(w * 100)}%</span></div>"
                f"<div style='margin-top:6px;'>{ui.bar(w * 100, ui.CYAN, '8px')}</div></div>",
                unsafe_allow_html=True,
            )

    elif sub == "Feature Importance (PFI)":
        features = sorted(ms.get("featureImportances") or [], key=lambda f: -f.get("importance", 0))
        max_imp = max((f.get("importance", 0) for f in features), default=1e-6)
        for i, f in enumerate(features):
            bar_pct = max(3.0, min(100.0, f.get("importance", 0) / max_imp * 100))
            st.markdown(
                f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:12px;padding:12px;margin-bottom:8px;'>"
                f"<div style='display:flex;align-items:center;justify-content:space-between;'>"
                f"<span style='font-size:13px;font-weight:500;color:{ui.TEXT}'>#{i+1} {f['label']}</span>"
                f"<span style='font-size:12px;color:{ui.CYAN};font-weight:600;'>{f.get('importance', 0):.3f}</span></div>"
                f"<div style='margin-top:6px;'>{ui.bar(bar_pct, ui.CYAN, '8px')}</div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:18px;'>"
            f"<h3 style='margin:0;font-size:14px;font-weight:600;color:{ui.TEXT}'>Ensemble Architecture</h3>"
            f"<p style='margin:6px 0 0;font-size:13px;color:{ui.MUTED}'>{ms.get('modelDescription', '')}</p></div>",
            unsafe_allow_html=True,
        )

    # Drift
    drift = ms.get("featureDrift") or []
    if drift:
        drift_rows = [
            [
                f"<span style='color:{ui.TEXT};font-weight:500'>{d['label']}</span>",
                f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{d.get('currentMean', 0):.3f}</span>",
                f"<span style='color:{ui.MUTED};font-variant-numeric:tabular-nums'>{d.get('baselineMean', 0):.3f}</span>",
                f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{d.get('psi', 0):.3f}</span>",
                ui.pill(d["status"], ui.ROSE if d["status"] == "ALERT" else ui.AMBER if d["status"] == "WARN" else ui.EMERALD,
                        "rgba(251,113,133,0.15)" if d["status"] == "ALERT" else "rgba(252,211,77,0.15)" if d["status"] == "WARN" else "rgba(52,211,153,0.15)"),
            ] for d in drift
        ]
        st.markdown(
            f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:18px;margin-top:12px;'>"
            f"<h3 style='margin:0 0 12px;font-size:14px;font-weight:600;color:{ui.TEXT}'>Feature Drift Analysis (PSI Scores)</h3>"
            + ui.html_table(["Feature", "Current Mean", "Baseline Mean", "PSI", "Status"], drift_rows,
                            align=["left", "right", "right", "right", "right"])
            + "</div>",
            unsafe_allow_html=True,
        )

    rolling = ms.get("rollingBrier") or []
    if rolling:
        baseline = ms.get("brierBaseline") or ms.get("brier") or 0
        st.plotly_chart(rolling_brier_chart(rolling, baseline), use_container_width=True)

    versions = ms.get("modelVersions") or []
    if versions:
        version_rows = [
            [f"<b style='color:{ui.CYAN}'>{v['version']}</b>",
             f"<span style='color:{ui.MUTED}'>{short_date(v['date'])}</span>",
             f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{v['auc']:.3f}</span>",
             f"<span style='color:{ui.TEXT};font-variant-numeric:tabular-nums'>{v['brier']:.3f}</span>",
             f"<span style='color:{ui.MUTED}'>{v.get('notes', '')}</span>"]
            for v in versions
        ]
        st.markdown(
            f"<div style='background:{ui._card_bg()};border:1px solid {ui.BORDER};border-radius:16px;padding:18px;margin-top:14px;'>"
            f"<h3 style='margin:0 0 12px;font-size:14px;font-weight:600;color:{ui.TEXT}'>Model Version History</h3>"
            + ui.html_table(["Version", "Date", "AUC", "Brier", "Notes"], version_rows,
                            align=["left", "left", "right", "right", "left"])
            + "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main — pure presentation, zero background work
# ---------------------------------------------------------------------------

def main() -> None:
    payload = load_payload()

    _render_engine_footer(payload)

    active_tab = st.session_state.get("active_tab", "games")
    render_header(active_tab)

    if payload is None:
        render_empty_state()
        return

    # Map payload to the "bundle" key the tabs expect
    # Tabs read: bundle["modelState"], bundle["gamesByDate"], bundle["calibrationRows"], etc.
    bundle = payload

    if active_tab == "games":
        games_tab(bundle)
    elif active_tab == "rankings":
        rankings_tab(bundle)
    elif active_tab == "calibration":
        calibration_tab(bundle)
    else:
        monitor_tab(bundle)


main()
