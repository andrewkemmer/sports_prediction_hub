"""MLB Stats API client — the single consolidated data source.

Faithful port of the fetch layer in src/convex/mlbActions.ts. Everything comes
from statsapi.mlb.com (market odds are an optional best-effort addition read
from THE_ODDS_API_KEY). Uses only the Python standard library so the full
pipeline can run and be tested anywhere.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

from . import cache
from .engine.teams import team_meta

MLB_BASE = "https://statsapi.mlb.com"
SEASON_START_MD = "03-15"
UPCOMING_WINDOW_DAYS = 3
RECENT_WINDOW_DAYS = 7
PAST_SEASON_END_MD = "11-01"
FIP_CONSTANT = 3.1
INJURY_SNAPSHOT_DAYS = 28
_USER_AGENT = "FreebuffMLB/1.0"


# ---------------------------------------------------------------------------
# HTTP / date helpers
# ---------------------------------------------------------------------------

def fetch_json(url: str, attempt: int = 0) -> dict:
    """GET + parse JSON with a small retry/backoff loop."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001
        if attempt < 2:
            time.sleep(0.5 * (attempt + 1))
            return fetch_json(url, attempt + 1)
        raise


def map_limit(items: list, limit: int, fn):
    """Run `fn` over items with bounded concurrency, preserving order.

    Worker exceptions are NOT silently swallowed: after all workers finish,
    the first exception is re-raised so callers never see None placeholders.
    """
    results = [None] * len(items)
    errors: list[BaseException] = []
    if not items:
        return results

    def worker(idx):
        try:
            results[idx] = fn(items[idx])
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    with ThreadPoolExecutor(max_workers=max(1, min(limit, len(items)))) as ex:
        for i in range(len(items)):
            ex.submit(worker, i)
    if errors:
        raise errors[0]
    return results


def to_ymd(d: date) -> str:
    return d.isoformat()


def add_days(ymd: str, days: int) -> str:
    return (date.fromisoformat(ymd) + timedelta(days=days)).isoformat()


def et_date_string(now: datetime | None = None) -> str:
    """Today's date in the America/New_York timezone as YYYY-MM-DD."""
    now = now or datetime.now()
    return now.astimezone(timezone(timedelta(hours=-4) if _edt(now) else timedelta(hours=-5))).date().isoformat()


def _edt(now: datetime) -> bool:
    """Rough EDT check (Mar-Nov). Good enough for the dashboard's as-of date."""
    m = now.month
    return 4 <= m <= 10


def date_ranges(start: str, end: str, chunk_days: int) -> list[dict]:
    ranges = []
    end_d = date.fromisoformat(end)
    cur = date.fromisoformat(start)
    while cur <= end_d:
        next_d = cur + timedelta(days=chunk_days - 1)
        chunk_end = min(next_d, end_d)
        ranges.append({"start": cur.isoformat(), "end": chunk_end.isoformat()})
        cur = next_d + timedelta(days=1)
    return ranges


def schedule_url(start: str, end: str) -> str:
    return (
        f"{MLB_BASE}/api/v1/schedule?sportId=1&startDate={start}&endDate={end}"
        f"&hydrate=probablePitcher,linescore,weather"
    )


# ---------------------------------------------------------------------------
# Schedule parsing
# ---------------------------------------------------------------------------

def _first_float(value) -> float | None:
    """First float from a scalar or string like '72' or '6 mph' (or None)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        return float(m.group()) if m else None
    return None


def parse_weather(w) -> dict | None:
    """Normalize the MLB API weather blob.

    The API returns `temp`/`wind` as strings ("72", "6 mph, Out To CF") on
    some games and structured objects ({"speed": 6}) on others; both shapes
    are handled so a single bad game never breaks the whole refresh.
    """
    if not w:
        return None
    out: dict = {}
    wind_val = w.get("wind")
    if isinstance(wind_val, dict):
        wind = _first_float(wind_val.get("speed"))
    else:
        wind = _first_float(wind_val)
    temp = _first_float(w.get("temp"))
    if isinstance(w.get("condition"), str):
        out["condition"] = w["condition"]
    if temp is not None:
        out["tempF"] = temp
    if wind is not None:
        out["windMph"] = wind
    return out if out else None


def parse_game(g: dict) -> dict | None:
    away = (g.get("teams") or {}).get("away")
    home = (g.get("teams") or {}).get("home")
    if not away or not home or not away.get("team", {}).get("id") or not home.get("team", {}).get("id"):
        return None
    away_meta = team_meta(away["team"]["id"])
    home_meta = team_meta(home["team"]["id"])
    winner = None
    if away.get("isWinner") is True:
        winner = "away"
    elif home.get("isWinner") is True:
        winner = "home"
    innings = None
    cur = (g.get("linescore") or {}).get("currentInning")
    if isinstance(cur, (int, float)) and cur > 9:
        innings = int(cur)

    def team_side(side: dict, meta: dict) -> dict:
        lr = side.get("leagueRecord") or {}
        rec = {
            "id": side["team"]["id"],
            "abbrev": meta["abbrev"],
            "name": meta["name"],
        }
        if isinstance(side.get("score"), (int, float)):
            rec["score"] = side["score"]
        if isinstance(lr.get("wins"), (int, float)):
            rec["wins"] = lr["wins"]
        if isinstance(lr.get("losses"), (int, float)):
            rec["losses"] = lr["losses"]
        return rec

    def pitcher(side: dict) -> dict | None:
        pp = side.get("probablePitcher")
        if not pp or not pp.get("id"):
            return None
        return {"id": pp["id"], "name": pp.get("fullName") or ""}

    status = (g.get("status") or {}).get("abstractGameState") or "Scheduled"
    return {
        "gamePk": g["gamePk"],
        "date": g.get("officialDate") or (g.get("gameDate") or "")[:10],
        "gameDate": g.get("gameDate") or "",
        "dayNight": g.get("dayNight") or "day",
        "status": status,
        "detailedState": (g.get("status") or {}).get("detailedState"),
        "away": team_side(away, away_meta),
        "home": team_side(home, home_meta),
        "awayPitcher": pitcher(away),
        "homePitcher": pitcher(home),
        "venue": (g.get("venue") or {}).get("name"),
        "innings": innings,
        "winner": winner,
        "season": g.get("season"),
        "weather": parse_weather(g.get("weather")),
    }


def parse_schedule(data: dict) -> list[dict]:
    out = []
    for d in data.get("dates") or []:
        for g in d.get("games") or []:
            if g.get("gameType") != "R":
                continue
            parsed = parse_game(g)
            if parsed:
                out.append(parsed)
    return out


def fetch_schedule_range(start: str, end: str) -> list[dict]:
    return parse_schedule(fetch_json(schedule_url(start, end)))


def _schedule_chunk(start: str, end: str) -> list[dict]:
    """Fetch one schedule window; return [] on failure so a bad window is
    skipped instead of killing the whole refresh."""
    try:
        return fetch_schedule_range(start, end)
    except Exception:  # noqa: BLE001 — transient API errors degrade gracefully
        return []


def fetch_season(season: str, through_date: str) -> list[dict]:
    start = f"{season}-{SEASON_START_MD}"
    ranges = date_ranges(start, through_date, 30)
    seen: dict[int, dict] = {}
    results = map_limit(ranges, 8, lambda r: _schedule_chunk(r["start"], r["end"]))
    for games in results:
        for g in games or []:
            seen[g["gamePk"]] = g
    return sorted(seen.values(), key=lambda g: g["gameDate"])


def fetch_all_seasons(seasons: list[str], current_season: str, through_date: str) -> list[dict]:
    all_games = []
    results = map_limit(
        seasons,
        min(3, len(seasons)),
        lambda s: fetch_season(s, through_date if s == current_season else f"{s}-{PAST_SEASON_END_MD}"),
    )
    for games in results:
        all_games.extend(games)
    return sorted(all_games, key=lambda g: g["gameDate"])


# ---------------------------------------------------------------------------
# Pitcher season stats (ERA / K9 / FIP)
# ---------------------------------------------------------------------------

def stat_number(value) -> float | None:
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, (int, float)) and value == value:  # not NaN
        return float(value)
    return None


def innings_pitched_value(ip) -> float:
    """'84.1' -> 84.333... (baseball innings convention)."""
    if isinstance(ip, (int, float)):
        return float(ip)
    if isinstance(ip, str):
        try:
            whole, _, frac = ip.partition(".")
            return float(whole or 0) + (1 / 3 if frac == "1" else 2 / 3 if frac == "2" else 0)
        except ValueError:
            return 0.0
    return 0.0


def fetch_pitcher_stats(pairs: list[dict], cached: dict | None = None) -> dict:
    """{id|season} -> {era, k9, fip}. Skips keys already in `cached`."""
    cached = cached or {}
    seen = set()
    unique = []
    for p in pairs:
        key = f"{p['id']}|{p['season']}"
        if p["id"] <= 0 or key in seen or key in cached:
            continue
        seen.add(key)
        unique.append(p)
    out: dict = {}

    def fetch(p):
        key = f"{p['id']}|{p['season']}"
        try:
            data = fetch_json(
                f"{MLB_BASE}/api/v1/people/{p['id']}/stats?stats=season&group=pitching&season={p['season']}"
            )
            splits = (data.get("stats") or [{}])[0].get("splits") or []
            if not splits:
                return
            stat = splits[0]["stat"]
            era = stat_number(stat.get("era"))
            k9 = stat_number(stat.get("strikeoutsPer9Inn"))
            hr = stat_number(stat.get("homeRuns")) or 0
            bb = stat_number(stat.get("baseOnBalls")) or 0
            hbp = stat_number(stat.get("hitByPitch")) or 0
            so = stat_number(stat.get("strikeOuts")) or 0
            ip = innings_pitched_value(stat.get("inningsPitched"))
            fip = None
            if ip > 0:
                fip = (13 * hr + 3 * (bb + hbp) - 2 * so) / ip + FIP_CONSTANT
            rec = {}
            if era is not None:
                rec["era"] = era
            if k9 is not None:
                rec["k9"] = k9
            if fip is not None:
                rec["fip"] = round(fip, 2)
            if rec:
                out[key] = rec
        except Exception:  # noqa: BLE001 — individual pitcher failures are non-fatal
            pass

    map_limit(unique, 16, fetch)
    return out


def attach_pitcher_stats(games: list[dict], stats: dict) -> list[dict]:
    out = []
    for g in games:
        season = g.get("season")
        def with_stats(p):
            if not p:
                return p
            key = f"{p['id']}|{season}"
            return {**p, **stats.get(key, {})}
        out.append({
            **g,
            "awayPitcher": with_stats(g.get("awayPitcher")),
            "homePitcher": with_stats(g.get("homePitcher")),
        })
    return out


# ---------------------------------------------------------------------------
# As-of-date stats (per-game game logs, accumulated with no lookahead)
#
# The season endpoints above return full-season totals that leak information
# from AFTER a game's date into that game's features (a July game would see
# August stats). For honest training the pipeline accumulates each entity's
# per-game game log strictly BEFORE the target game's date. Logs are cached
# once per {id|season}, so any as-of date is a cheap local sum.
# ---------------------------------------------------------------------------

def _num(value) -> float:
    """Best-effort float conversion (0 on junk) for compact log entries."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _compact_pitcher_entry(split: dict) -> dict | None:
    st = split.get("stat") or {}
    ip = innings_pitched_value(st.get("inningsPitched"))
    if ip <= 0:
        return None
    return {
        "d": split.get("date") or "",
        "ip": round(ip, 3),
        "er": _num(st.get("earnedRuns")),
        "so": _num(st.get("strikeOuts")),
        "bb": _num(st.get("baseOnBalls")),
        "hbp": _num(st.get("hitByPitch")),
        "hr": _num(st.get("homeRuns")),
    }


def fetch_pitcher_game_logs(pairs: list[dict], cached: dict | None = None) -> dict:
    """{id|season} -> sorted compact per-game pitching entries (gameLog)."""
    cached = cached or {}
    seen = set()
    unique = []
    for p in pairs:
        key = f"{p['id']}|{p['season']}"
        if p["id"] <= 0 or key in seen or key in cached:
            continue
        seen.add(key)
        unique.append(p)
    out: dict = {}

    def fetch(p):
        key = f"{p['id']}|{p['season']}"
        try:
            data = fetch_json(
                f"{MLB_BASE}/api/v1/people/{p['id']}/stats?stats=gameLog&group=pitching&season={p['season']}&gameType=R"
            )
            splits = (data.get("stats") or [{}])[0].get("splits") or []
            entries = [e for e in (_compact_pitcher_entry(s) for s in splits) if e and e["d"]]
            entries.sort(key=lambda e: e["d"])
            if entries:
                out[key] = entries
        except Exception:  # noqa: BLE001 — individual pitcher failures are non-fatal
            pass

    map_limit(unique, 16, fetch)
    return out


def pitcher_as_of(entries: list[dict] | None, ymd: str) -> dict:
    """Season ERA / K9 / FIP accumulated strictly before `ymd` (no lookahead)."""
    if not entries:
        return {}
    ip = er = so = bb = hbp = hr = 0.0
    for e in entries:  # date-sorted; stop at the first game on/after the target date
        if e["d"] >= ymd:
            break
        ip += e["ip"]
        er += e["er"]
        so += e["so"]
        bb += e["bb"]
        hbp += e["hbp"]
        hr += e["hr"]
    if ip <= 0:
        return {}
    return {
        "era": round(er * 9 / ip, 2),
        "k9": round(so * 9 / ip, 2),
        "fip": round((13 * hr + 3 * (bb + hbp) - 2 * so) / ip + FIP_CONSTANT, 2),
    }


# ---------------------------------------------------------------------------
# Team season stats (OPS / ERA / fielding%)
# ---------------------------------------------------------------------------

def fetch_team_season_stats(pairs: list[dict], cached: dict | None = None) -> dict:
    cached = cached or {}
    seen = set()
    unique = []
    for p in pairs:
        key = f"{p['id']}|{p['season']}"
        if p["id"] <= 0 or key in seen or key in cached:
            continue
        seen.add(key)
        unique.append(p)
    out: dict = {}

    def fetch(p):
        key = f"{p['id']}|{p['season']}"
        try:
            data = fetch_json(
                f"{MLB_BASE}/api/v1/teams/{p['id']}/stats?stats=season&group=hitting,pitching,fielding&season={p['season']}"
            )
            result = {}
            for block in data.get("stats") or []:
                group = (block.get("group") or {}).get("displayName")
                splits = block.get("splits") or []
                if not splits:
                    continue
                stat = splits[0]["stat"]
                if group == "hitting":
                    v = stat_number(stat.get("ops"))
                    if v is not None:
                        result["ops"] = v
                elif group == "pitching":
                    v = stat_number(stat.get("era"))
                    if v is not None:
                        result["era"] = v
                elif group == "fielding":
                    v = stat_number(stat.get("fielding"))
                    if v is not None:
                        result["fieldingPct"] = v
            if result:
                out[key] = result
        except Exception:  # noqa: BLE001
            pass

    map_limit(unique, 16, fetch)
    return out


def attach_team_season_stats(games: list[dict], stats: dict) -> list[dict]:
    out = []
    for g in games:
        season = g.get("season")
        home_stats = stats.get(f"{g['home']['id']}|{season}") or {}
        away_stats = stats.get(f"{g['away']['id']}|{season}") or {}
        out.append({
            **g,
            "home": {**g["home"], **home_stats},
            "away": {**g["away"], **away_stats},
        })
    return out


def _compact_hitting_entry(split: dict) -> dict | None:
    st = split.get("stat") or {}
    return {
        "d": split.get("date") or "",
        "ab": _num(st.get("atBats")),
        "h": _num(st.get("hits")),
        "bb": _num(st.get("baseOnBalls")),
        "hbp": _num(st.get("hitByPitch")),
        "sf": _num(st.get("sacFlies")),
        "tb": _num(st.get("totalBases")),
    }


def _compact_fielding_entry(split: dict) -> dict | None:
    st = split.get("stat") or {}
    return {"d": split.get("date") or "", "po": _num(st.get("putOuts")), "a": _num(st.get("assists")), "e": _num(st.get("errors"))}


def fetch_team_game_logs(pairs: list[dict], cached: dict | None = None) -> dict:
    """{id|season} -> {"hitting": [...], "pitching": [...], "fielding": [...]} (gameLog)."""
    cached = cached or {}
    seen = set()
    unique = []
    for p in pairs:
        key = f"{p['id']}|{p['season']}"
        if p["id"] <= 0 or key in seen or key in cached:
            continue
        seen.add(key)
        unique.append(p)
    out: dict = {}

    def fetch(p):
        key = f"{p['id']}|{p['season']}"
        try:
            data = fetch_json(
                f"{MLB_BASE}/api/v1/teams/{p['id']}/stats?stats=gameLog&group=hitting,pitching,fielding&season={p['season']}&gameType=R"
            )
            result: dict = {}
            for block in data.get("stats") or []:
                group = (block.get("group") or {}).get("displayName") or ""
                splits = block.get("splits") or []
                if group == "hitting":
                    entries = [e for e in (_compact_hitting_entry(s) for s in splits) if e and e["d"]]
                elif group == "pitching":
                    entries = [e for e in (_compact_pitcher_entry(s) for s in splits) if e and e["d"]]
                elif group == "fielding":
                    entries = [e for e in (_compact_fielding_entry(s) for s in splits) if e and e["d"]]
                else:
                    continue
                entries.sort(key=lambda e: e["d"])
                result[group] = entries
            if result:
                out[key] = result
        except Exception:  # noqa: BLE001 — individual team failures are non-fatal
            pass

    map_limit(unique, 16, fetch)
    return out


def _ops_from_hitting(ab: float, h: float, bb: float, hbp: float, sf: float, tb: float) -> float | None:
    if ab + bb + hbp + sf <= 0:
        return None
    obp = (h + bb + hbp) / (ab + bb + hbp + sf)
    slg = tb / ab if ab > 0 else 0.0
    return obp + slg


def team_as_of(log: dict | None, ymd: str) -> dict:
    """Team OPS / staff ERA / fielding pct accumulated strictly before `ymd`."""
    if not log:
        return {}
    out: dict = {}

    ab = h = bb = hbp = sf = tb = 0.0
    for e in log.get("hitting") or []:
        if e["d"] >= ymd:
            break
        ab += e["ab"]
        h += e["h"]
        bb += e["bb"]
        hbp += e["hbp"]
        sf += e["sf"]
        tb += e["tb"]
    ops = _ops_from_hitting(ab, h, bb, hbp, sf, tb)
    if ops is not None:
        out["ops"] = round(ops, 3)

    ip = er = 0.0
    for e in log.get("pitching") or []:
        if e["d"] >= ymd:
            break
        ip += e["ip"]
        er += e["er"]
    if ip > 0:
        out["era"] = round(er * 9 / ip, 2)

    po = a = err = 0.0
    for e in log.get("fielding") or []:
        if e["d"] >= ymd:
            break
        po += e["po"]
        a += e["a"]
        err += e["e"]
    chances = po + a + err
    if chances > 0:
        out["fieldingPct"] = round((po + a) / chances, 3)

    return out


def attach_as_of_stats(games: list[dict], pitcher_logs: dict, team_logs: dict) -> list[dict]:
    """Attach per-game as-of-date pitcher + team stats (no lookahead)."""
    out = []
    for g in games:
        season = g.get("season")
        ymd = g["date"]

        def pitcher_stats(p):
            if not p:
                return p
            return {**p, **pitcher_as_of(pitcher_logs.get(f"{p['id']}|{season}"), ymd)}

        out.append({
            **g,
            "awayPitcher": pitcher_stats(g.get("awayPitcher")),
            "homePitcher": pitcher_stats(g.get("homePitcher")),
            "home": {**g["home"], **team_as_of(team_logs.get(f"{g['home']['id']}|{season}"), ymd)},
            "away": {**g["away"], **team_as_of(team_logs.get(f"{g['away']['id']}|{season}"), ymd)},
        })
    return out


# ---------------------------------------------------------------------------
# Injury data (MLB Stats API rosters)
# ---------------------------------------------------------------------------

def is_injured_status(status: dict | None) -> bool:
    if not status:
        return False
    code = str(status.get("code") or "")
    description = str(status.get("description") or "").lower()
    return (
        code.startswith("D")
        or code.startswith("IL")
        or "injured" in description
        or "day-to-day" in description
    )


def fetch_injury_count(team_id: int, d: str, season: str) -> int:
    try:
        data = fetch_json(
            f"{MLB_BASE}/api/v1/teams/{team_id}/roster?rosterType=40Man&season={season}&date={d}"
        )
        return sum(1 for entry in data.get("roster") or [] if is_injured_status(entry.get("status")))
    except Exception:  # noqa: BLE001
        return 0


def fetch_injury_snapshots(
    team_ids: list[int],
    season: str,
    start_date: str,
    end_date: str,
    previous: dict | None = None,
) -> dict:
    """Per-team IL snapshots at ~28-day intervals; reuses cached dates."""
    previous = previous or {}
    dates = []
    cursor = start_date
    while cursor <= end_date:
        dates.append(cursor)
        cursor = add_days(cursor, INJURY_SNAPSHOT_DAYS)
    if not dates or dates[-1] != end_date:
        dates.append(end_date)

    out: dict[int, list[dict]] = {}
    jobs: list[tuple[int, str]] = []
    for team_id in team_ids:
        cached_list = list(previous.get(str(team_id)) or previous.get(team_id) or [])
        have = {s["date"] for s in cached_list}
        for d in dates:
            if d not in have:
                jobs.append((team_id, d))
        out[team_id] = cached_list

    results = map_limit(
        jobs,
        16,
        lambda job: {"teamId": job[0], "date": job[1], "count": fetch_injury_count(job[0], job[1], season)},
    )
    for r in results:
        out.setdefault(r["teamId"], []).append({"date": r["date"], "count": r["count"]})
    for lst in out.values():
        lst.sort(key=lambda s: s["date"])
    return {str(k): v for k, v in out.items()}


def fetch_current_injury_snapshot(team_ids: list[int], d: str, season: str) -> dict:
    pairs = sorted(set(tid for tid in team_ids if tid > 0))
    results = map_limit(pairs, 16, lambda tid: (tid, fetch_injury_count(tid, d, season)))
    return {tid: count for tid, count in results}


# ---------------------------------------------------------------------------
# Lineups (actual starting 9 + bench, from the per-game boxscore)
# ---------------------------------------------------------------------------

def fetch_game_lineup(game_pk: int) -> dict | None:
    """Actual lineup from the boxscore. None when lineups are not posted yet."""
    try:
        req = urllib.request.Request(f"{MLB_BASE}/api/v1/game/{game_pk}/boxscore", headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
    except Exception:  # noqa: BLE001 — 404/empty boxscore is the expected pre-lineup state
        return None
    teams = data.get("teams")
    if not teams or not teams.get("home") or not teams.get("away"):
        return None

    def parse_side(side: dict) -> dict:
        batting_order = []
        bench = []
        order_ids = set(side.get("battingOrder") or [])
        order_slot = {}
        players = side.get("players") or {}
        for key, p in players.items():
            person = p.get("person") or {}
            pid = person.get("id")
            if not isinstance(pid, int):
                continue
            pos = (p.get("position") or {}).get("abbreviation")
            if pid in order_ids:
                try:
                    slot = int(p.get("battingOrder"))
                except (TypeError, ValueError):
                    slot = 0
                if slot > 0:
                    order_slot[pid] = slot
                batting_order.append({"id": pid, "name": person.get("fullName") or "", "pos": pos})
            elif pos != "P":
                bench.append({"id": pid, "name": person.get("fullName") or "", "pos": pos})
        batting_order.sort(key=lambda p: order_slot.get(p["id"], 99))
        return {"battingOrder": batting_order, "bench": bench}

    return {"home": parse_side(teams["home"]), "away": parse_side(teams["away"])}


def fetch_lineups_for_games(games: list[dict], concurrency: int = 16) -> dict:
    out: dict[int, dict] = {}
    seen = set()
    targets = []
    for g in games:
        if g["gamePk"] in seen:
            continue
        seen.add(g["gamePk"])
        targets.append(g)

    def fetch(g):
        lu = fetch_game_lineup(g["gamePk"])
        if (
            lu
            and (lu.get("home") or {}).get("battingOrder")
            and (lu.get("away") or {}).get("battingOrder")
        ):
            return g["gamePk"], lu
        return g["gamePk"], None

    for pk, lu in map_limit(targets, concurrency, fetch):
        if lu:
            out[pk] = lu
    return out


def lineup_ops(lineup: list[dict] | None) -> float:
    if not lineup:
        return 0.0
    total = 0.0
    w = 0
    for i, p in enumerate(lineup):
        ops = p.get("ops")
        if not isinstance(ops, (int, float)):
            continue
        weight = 2 if i < 4 else 1
        total += ops * weight
        w += weight
    return total / w if w > 0 else 0.0


def attach_lineups(games: list[dict], lineups: dict, player_ops: dict) -> list[dict]:
    out = []
    for g in games:
        lu = lineups.get(g["gamePk"])
        if not lu:
            out.append(g)
            continue

        def with_ops(side):
            if not side:
                return side
            return {
                "battingOrder": [{**p, "ops": player_ops.get(p["id"])} for p in side["battingOrder"]],
                "bench": [{**p, "ops": player_ops.get(p["id"])} for p in side["bench"]],
            }

        home = with_ops(lu.get("home"))
        away = with_ops(lu.get("away"))
        home_ops = lineup_ops(home["battingOrder"]) if home else 0.0
        away_ops = lineup_ops(away["battingOrder"]) if away else 0.0
        out.append({
            **g,
            "lineups": {"home": home, "away": away},
            "lineupStats": {
                "home": {"known": home_ops > 0, "ops": home_ops},
                "away": {"known": away_ops > 0, "ops": away_ops},
            },
        })
    return out


def fetch_player_season_ops(ids: list[int], season: str, cached: dict | None = None) -> dict:
    """{id} -> season OPS for batters (skips ids already in the cache)."""
    cached = cached or {}
    unique = sorted({pid for pid in ids if pid > 0 and f"{pid}|{season}" not in cached})
    out: dict = {}

    def fetch(pid):
        try:
            data = fetch_json(
                f"{MLB_BASE}/api/v1/people/{pid}/stats?stats=season&group=hitting&season={season}"
            )
            splits = (data.get("stats") or [{}])[0].get("splits") or []
            if not splits:
                return
            ops = stat_number(splits[0]["stat"].get("ops"))
            if ops is not None:
                out[pid] = ops
        except Exception:  # noqa: BLE001
            pass

    map_limit(unique, 24, fetch)
    return out


def fetch_batter_game_logs(ids: list[int], season: str, cached: dict | None = None) -> dict:
    """{pid|season} -> sorted compact per-game hitting entries (gameLog)."""
    cached = cached or {}
    unique = sorted({pid for pid in ids if pid > 0 and f"{pid}|{season}" not in cached})
    out: dict = {}

    def fetch(pid):
        key = f"{pid}|{season}"
        try:
            data = fetch_json(
                f"{MLB_BASE}/api/v1/people/{pid}/stats?stats=gameLog&group=hitting&season={season}&gameType=R"
            )
            splits = (data.get("stats") or [{}])[0].get("splits") or []
            entries = [e for e in (_compact_hitting_entry(s) for s in splits) if e and e["d"]]
            entries.sort(key=lambda e: e["d"])
            if entries:
                out[key] = entries
        except Exception:  # noqa: BLE001 — individual batter failures are non-fatal
            pass

    map_limit(unique, 24, fetch)
    return out


def batter_ops_as_of(entries: list[dict] | None, ymd: str) -> float | None:
    """Batter season OPS accumulated strictly before `ymd`; None with no prior games."""
    if not entries:
        return None
    ab = h = bb = hbp = sf = tb = 0.0
    for e in entries:  # date-sorted; stop at the first game on/after the target date
        if e["d"] >= ymd:
            break
        ab += e["ab"]
        h += e["h"]
        bb += e["bb"]
        hbp += e["hbp"]
        sf += e["sf"]
        tb += e["tb"]
    ops = _ops_from_hitting(ab, h, bb, hbp, sf, tb)
    return round(ops, 3) if ops is not None else None


def attach_lineups_as_of(games: list[dict], lineups: dict, batter_logs: dict) -> list[dict]:
    """Attach lineups with each batter's OPS as-of the game's own date (no lookahead)."""
    out = []
    for g in games:
        lu = lineups.get(g["gamePk"])
        if not lu:
            out.append(g)
            continue
        season = g.get("season")
        ymd = g["date"]

        def with_ops(side):
            if not side:
                return side
            return {
                "battingOrder": [
                    {**p, "ops": batter_ops_as_of(batter_logs.get(f"{p['id']}|{season}"), ymd)}
                    for p in side["battingOrder"]
                ],
                "bench": [
                    {**p, "ops": batter_ops_as_of(batter_logs.get(f"{p['id']}|{season}"), ymd)}
                    for p in side["bench"]
                ],
            }

        home = with_ops(lu.get("home"))
        away = with_ops(lu.get("away"))
        home_ops = lineup_ops(home["battingOrder"]) if home else 0.0
        away_ops = lineup_ops(away["battingOrder"]) if away else 0.0
        out.append({
            **g,
            "lineups": {"home": home, "away": away},
            "lineupStats": {
                "home": {"known": home_ops > 0, "ops": home_ops},
                "away": {"known": away_ops > 0, "ops": away_ops},
            },
        })
    return out


# ---------------------------------------------------------------------------
# Market odds (optional — reads THE_ODDS_API_KEY)
# ---------------------------------------------------------------------------

ODDS_BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
ODDS_TTL_SECONDS = 3600  # reuse cached market odds within an hour


def market_odds_enabled() -> bool:
    """True when THE_ODDS_API_KEY is configured (live market prices)."""
    return bool(os.environ.get("THE_ODDS_API_KEY"))


def odds_num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def pick_bookmaker(bookmakers: list | None):
    if not bookmakers:
        return None
    for key in ("pinnacle", "draftkings", "fanduel", "betmgm"):
        for b in bookmakers:
            if (b or {}).get("key") == key:
                return b
    return bookmakers[0]


def odds_from_event(event: dict) -> dict | None:
    home = event.get("home_team")
    away = event.get("away_team")
    if not isinstance(home, str) or not isinstance(away, str):
        return None
    book = pick_bookmaker(event.get("bookmakers"))
    markets = (book or {}).get("markets") or []
    odds: dict = {}
    for m in markets:
        key = m.get("key")
        outcomes = m.get("outcomes") or []
        if key == "h2h":
            for o in outcomes:
                if o.get("name") == home:
                    odds["homeMoneyline"] = odds_num(o.get("price"))
                elif o.get("name") == away:
                    odds["awayMoneyline"] = odds_num(o.get("price"))
        elif key == "totals":
            for o in outcomes:
                pt = odds_num(o.get("point"))
                if pt is not None:
                    odds["total"] = pt
                if o.get("name") == "Over":
                    odds["overPrice"] = odds_num(o.get("price"))
                elif o.get("name") == "Under":
                    odds["underPrice"] = odds_num(o.get("price"))
        elif key == "spreads":
            for o in outcomes:
                pt = odds_num(o.get("point"))
                if pt is not None:
                    odds["runLine"] = abs(pt)
                if o.get("name") == home:
                    odds["homeRunLinePrice"] = odds_num(o.get("price"))
                elif o.get("name") == away:
                    odds["awayRunLinePrice"] = odds_num(o.get("price"))
    if not odds:
        return None
    odds["source"] = "The Odds API"
    commence = event.get("commence_time")
    d = ""
    if commence:
        try:
            d = et_date_string(datetime.fromisoformat(commence.replace("Z", "+00:00")))
        except ValueError:
            d = commence[:10]
    return {"date": d, "home": home, "away": away, "odds": odds}


def fetch_market_odds() -> dict:
    """"date|homeFullName|awayFullName" -> MarketOdds (best-effort, empty w/o key).

    Reads THE_ODDS_API_KEY from the environment. Without the key the map is
    empty and the UI shows model-derived fair odds. Results are cached on disk
    for ODDS_TTL_SECONDS so on-demand date lookups don't burn the free-tier
    request budget; a failed fetch falls back to the last good snapshot.
    """
    if not market_odds_enabled():
        return {}
    cached = cache.load_market_odds()
    fetched_at = cached.get("fetchedAt") or 0
    if cached.get("odds") and time.time() * 1000 - fetched_at < ODDS_TTL_SECONDS * 1000:
        return cached["odds"]
    out: dict = {}
    api_key = os.environ["THE_ODDS_API_KEY"]
    try:
        url = (
            f"{ODDS_BASE}/?apiKey={urllib.parse.quote(api_key)}"
            f"&regions=us&markets=h2h,totals,spreads&oddsFormat=american"
        )
        data = fetch_json(url)
        for event in data or []:
            parsed = odds_from_event(event)
            if parsed and parsed["date"]:
                out[f"{parsed['date']}|{parsed['home']}|{parsed['away']}"] = parsed["odds"]
    except Exception:  # noqa: BLE001 — market odds are best-effort
        return cached.get("odds") or {}
    if out:
        cache.save_market_odds({"fetchedAt": int(time.time() * 1000), "odds": out})
    return out


def market_odds_for_game(odds_map: dict, game: dict) -> dict | None:
    if not odds_map:
        return None
    home_full = team_meta(game["home"]["id"])["fullName"]
    away_full = team_meta(game["away"]["id"])["fullName"]
    return (
        odds_map.get(f"{game['date']}|{home_full}|{away_full}")
        or odds_map.get(f"{game['date']}|{away_full}|{home_full}")
    )
