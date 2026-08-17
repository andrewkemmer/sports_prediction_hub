"""Probe the MLB Stats API endpoints used by the data layer (stdlib only)."""
from __future__ import annotations

import json
import sys
import urllib.request

BASE = "https://statsapi.mlb.com"


def get(url: str, timeout: int = 12):
    req = urllib.request.Request(url, headers={"User-Agent": "FreebuffMLB/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def step(name, fn):
    try:
        fn()
        print(f"[ok] {name}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        raise


def main() -> None:
    s = get(f"{BASE}/api/v1/schedule?sportId=1&startDate=2026-08-15&endDate=2026-08-17&hydrate=probablePitcher,linescore,weather")
    games = [g for d in s["dates"] for g in d["games"]]
    final = [g for g in games if g["status"]["abstractGameState"] == "Final"]
    print(f"games in window: {len(games)}  final: {len(final)}")

    def probe_boxscore():
        pk = final[0]["gamePk"]
        b = get(f"{BASE}/api/v1/game/{pk}/boxscore")
        home = b["teams"]["home"]
        order = home.get("battingOrder", [])
        print(f"game {pk} home battingOrder len: {len(order)} sample: {order[:4]}")
        key = f"ID{order[0]}" if order else None
        if key and key in home["players"]:
            p = home["players"][key]
            print("first batter:", p["person"]["fullName"], p["position"]["abbreviation"])

    step("boxscore", probe_boxscore)

    def probe_pitcher():
        prob = [g for g in games if g["teams"]["home"].get("probablePitcher")]
        pp = prob[0]["teams"]["home"]["probablePitcher"]
        st = get(f"{BASE}/api/v1/people/{pp['id']}/stats?stats=season&group=pitching&season=2026")
        blocks = st.get("stats") or []
        print(f"pitcher {pp['fullName']} blocks: {len(blocks)}")
        splits = (blocks[0].get("splits") or []) if blocks else []
        print(f"splits: {len(splits)}")
        if splits:
            stat = splits[0]["stat"]
            print("pitcher stats:", {k: stat.get(k) for k in ["era", "strikeoutsPer9Inn", "inningsPitched", "homeRuns", "baseOnBalls", "hitByPitch", "strikeOuts"]})

    step("pitcher stats", probe_pitcher)

    def probe_team_stats():
        ts = get(f"{BASE}/api/v1/teams/108/stats?stats=season&group=hitting,pitching,fielding&season=2026")
        for block in ts["stats"]:
            g = block["group"]["displayName"]
            splits = block.get("splits") or []
            print(f"team stat {g} splits: {len(splits)}")
            if splits:
                st2 = splits[0]["stat"]
                print(" ", {k: st2.get(k) for k in ["ops", "era", "fielding"]})

    step("team stats", probe_team_stats)

    def probe_roster():
        ro = get(f"{BASE}/api/v1/teams/108/roster?rosterType=40Man&season=2026&date=2026-08-17")
        roster = ro.get("roster") or []
        print("roster entries:", len(roster), "status sample:", roster[0].get("status") if roster else None)

    step("roster", probe_roster)

    def probe_player_hitting():
        ps = get(f"{BASE}/api/v1/people/660271/stats?stats=season&group=hitting&season=2026")
        splits = (ps["stats"][0].get("splits") or []) if ps.get("stats") else []
        print("player ops:", splits[0]["stat"].get("ops") if splits else None)

    step("player hitting", probe_player_hitting)

    print("ALL PROBES OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        sys.exit(1)
