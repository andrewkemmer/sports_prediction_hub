"""Probe The Odds API with a real key.

Usage:
    THE_ODDS_API_KEY=<key> python3 mlb_streamlit/scripts/probe_odds.py

Verifies the raw API response (status, event shape) and then exercises the
dashboard's own parsing/caching path (data.fetch_market_odds), printing the
resulting date|home|away entries with their parsed prices.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mlb_streamlit import cache  # noqa: E402
from mlb_streamlit.data import ODDS_BASE, fetch_market_odds  # noqa: E402

api_key = os.environ.get("THE_ODDS_API_KEY", "")
if not api_key:
    print("No THE_ODDS_API_KEY in environment.")
    sys.exit(2)

url = (
    f"{ODDS_BASE}/?apiKey={api_key}"
    f"&regions=us&markets=h2h,totals,spreads&oddsFormat=american"
)
req = urllib.request.Request(url, headers={"User-Agent": "FreebuffMLB/1.0"})
try:
    with urllib.request.urlopen(req, timeout=20) as resp:
        status = resp.status
        payload = resp.read().decode("utf-8")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.reason}")
    print(e.read().decode("utf-8", "replace")[:400])
    sys.exit(1)
except urllib.error.URLError as e:
    print(f"Network error: {e.reason}")
    sys.exit(1)

import json  # noqa: E402

events = json.loads(payload)
print(f"HTTP {status} · {len(events)} MLB events")
for ev in events[:5]:
    print(
        f"  {ev.get('commence_time', '')[:10]}  "
        f"{ev.get('away_team')} @ {ev.get('home_team')}  "
        f"books: {[b.get('key') for b in ev.get('bookmakers') or []]}"
    )

print("\n--- dashboard fetch_market_odds() ---")
before = cache.load_market_odds()
odds = fetch_market_odds()
print(f"enabled: {bool(api_key)} · parsed entries: {len(odds)}")
for key in sorted(odds)[:8]:
    o = odds[key]
    print(
        f"  {key}: ML {o.get('homeMoneyline')}/{o.get('awayMoneyline')} · "
        f"total {o.get('total')} (O {o.get('overPrice')}/U {o.get('underPrice')}) · "
        f"RL {o.get('runLine')} ({o.get('homeRunLinePrice')}/{o.get('awayRunLinePrice')})"
    )
after = cache.load_market_odds()
print(f"\ncache written: {bool(after and after.get('odds'))} "
      f"(was cached before: {bool(before and before.get('odds'))})")
