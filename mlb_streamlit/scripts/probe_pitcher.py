"""Inspect raw pitcher stats endpoint response."""
from __future__ import annotations

import json
import urllib.request

BASE = "https://statsapi.mlb.com"


def get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "FreebuffMLB/1.0"})
    with urllib.request.urlopen(req, timeout=12) as r:
        return json.load(r)


s = get(f"{BASE}/api/v1/schedule?sportId=1&startDate=2026-08-15&endDate=2026-08-17&hydrate=probablePitcher")
games = [g for d in s["dates"] for g in d["games"]]
prob = [g for g in games if g["teams"]["home"].get("probablePitcher") or g["teams"]["away"].get("probablePitcher")]
for g in prob[:3]:
    for side in ("home", "away"):
        pp = g["teams"][side].get("probablePitcher")
        if not pp:
            continue
        st = get(f"{BASE}/api/v1/people/{pp['id']}/stats?stats=season&group=pitching&season=2026")
        print("=" * 60)
        print("pitcher:", pp["fullName"], pp["id"], "| stats keys:", list(st.keys()))
        print("stats len:", len(st.get("stats") or []), "| totalSplits:", st.get("totalSplits"))
        for block in st.get("stats") or []:
            print("  block:", block.get("group", {}).get("displayName"), "| splits:", len(block.get("splits") or []))
            if block.get("splits"):
                stat = block["splits"][0]["stat"]
                print("  stat:", {k: stat.get(k) for k in ["era", "strikeoutsPer9Inn", "inningsPitched", "homeRuns", "baseOnBalls", "hitByPitch", "strikeOuts", "gamesPlayed"]})
