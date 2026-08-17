"""HTML rendering helpers for the Streamlit dashboard.

The React app's card/table styling is recreated with scoped inline CSS so the
Streamlit dashboard keeps the same polished product look.
"""

from __future__ import annotations

import html as _html

ACCENT = "#427ff7"
CYAN = "#22d3ee"
EMERALD = "#34d399"
AMBER = "#fcd34d"
ROSE = "#fb7185"
PURPLE = "#e879f9"
ORANGE = "#fb923c"
MUTED = "#8b939f"
TEXT = "#e5e8ec"
BORDER = "rgba(255,255,255,0.09)"


def esc(value) -> str:
    return _html.escape(str(value if value is not None else ""))


def pill(text: str, color: str, bg: str, icon: str = "") -> str:
    icon_html = f"<span style='font-size:10px'>{icon}</span>&nbsp;" if icon else ""
    return (
        f"<span style='display:inline-flex;align-items:center;gap:4px;border-radius:999px;"
        f"padding:2px 9px;font-size:10px;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:.04em;white-space:nowrap;color:{color};background:{bg};"
        f"border:1px solid {bg}99;'>{icon_html}{esc(text)}</span>"
    )


def metric_card(label: str, value: str, color: str, sub: str = "", decimals: int = 3) -> str:
    sub_html = f"<div style='margin-top:4px;font-size:11px;color:{MUTED}'>{esc(sub)}</div>" if sub else ""
    return (
        f"<div style='background:{_card_bg()};border:1px solid {BORDER};border-radius:16px;"
        f"padding:18px 12px;text-align:center;height:100%;'>"
        f"<div style='font-size:10px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:{MUTED}'>{esc(label)}</div>"
        f"<div style='margin-top:8px;font-size:30px;font-weight:700;font-variant-numeric:tabular-nums;color:{color}'>{esc(value)}</div>"
        f"{sub_html}</div>"
    )


def stat_card(dot_color: str, label: str, value: str, sub: str, sub_color: str) -> str:
    return (
        f"<div style='background:{_card_bg()};border:1px solid {BORDER};border-radius:16px;padding:16px;'>"
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<span style='width:8px;height:8px;border-radius:50%;background:{dot_color};display:inline-block'></span>"
        f"<span style='font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:{MUTED}'>{esc(label)}</span>"
        f"</div>"
        f"<div style='margin-top:8px;font-size:19px;font-weight:700;color:{TEXT}'>{esc(value)}</div>"
        f"<div style='margin-top:4px;font-size:12px;color:{sub_color}'>{esc(sub)}</div>"
        f"</div>"
    )


def bar(pct: float, color: str, height: str = "6px") -> str:
    pct = max(0.0, min(100.0, pct))
    return (
        f"<div style='height:{height};width:100%;background:rgba(255,255,255,0.06);border-radius:999px;overflow:hidden;'>"
        f"<div style='height:100%;width:{pct:.1f}%;background:{color};border-radius:999px;'></div></div>"
    )


def html_table(headers: list[str], rows: list[list], align: list[str] | None = None) -> str:
    """Render a table. Each cell is an HTML string (use esc() for plain text)."""
    align = align or ["left"] * len(headers)
    th = "".join(
        f"<th style='padding:8px 12px;font-size:10px;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:.08em;color:{MUTED};text-align:{a};white-space:nowrap;'>{h}</th>"
        for h, a in zip(headers, align)
    )
    trs = []
    for row in rows:
        tds = "".join(
            f"<td style='padding:9px 12px;font-size:13px;text-align:{a};white-space:nowrap;"
            f"border-top:1px solid {BORDER};'>{c}</td>"
            for c, a in zip(row, align)
        )
        trs.append(f"<tr style='transition:background .15s;'>{tds}</tr>")
    return (
        f"<div style='overflow-x:auto;'><table style='width:100%;border-collapse:collapse;"
        f"font-size:13px;color:{TEXT};'><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>"
    )


def card(title: str, body: str, subtitle: str = "") -> str:
    sub = f"<p style='margin:4px 0 0;font-size:12px;color:{MUTED};line-height:1.5'>{subtitle}</p>" if subtitle else ""
    return (
        f"<div style='background:{_card_bg()};border:1px solid {BORDER};border-radius:16px;padding:18px;margin-bottom:14px;'>"
        f"<h3 style='margin:0;font-size:14px;font-weight:600;color:{TEXT}'>{esc(title)}</h3>{sub}{body}</div>"
    )


def legend(color: str, label: str, dashed: bool = False) -> str:
    swatch = (
        f"<span style='display:inline-block;width:14px;height:0;border-top:2px dashed {color};vertical-align:middle'></span>"
        if dashed
        else f"<span style='display:inline-block;width:11px;height:11px;border-radius:3px;background:{color};vertical-align:middle'></span>"
    )
    return f"<span style='display:inline-flex;align-items:center;gap:6px;font-size:11px;color:{MUTED}'>{swatch}{esc(label)}</span>"


def _card_bg() -> str:
    return "#12161c"


def section(title: str, subtitle: str = "") -> None:
    import streamlit as st

    sub = f"<p style='margin:6px 0 0;font-size:13px;color:{MUTED}'>{esc(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"<h2 style='font-size:20px;font-weight:700;letter-spacing:-.01em;color:{TEXT};margin:0'>{esc(title)}</h2>{sub}",
        unsafe_allow_html=True,
    )
