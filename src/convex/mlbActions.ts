"use node";

import { v } from "convex/values";
import { api, internal } from "./_generated/api";
import { action } from "./_generated/server";
import {
  applyIsotonic,
  applyModel,
  buildFeaturesForGame,
  calibrationCurvePoints,
  computeAuc,
  computeBrier,
  evaluate,
  logit,
  runModel,
  spearmanRank,
} from "./ml/model";
import { expectedMargin, expectedTotal, RunModel, simulateRuns } from "./ml/runs";
import { teamMeta } from "./ml/teams";
import {
  CalibrationSummary,
  TeamInfo,
  FeatureRow,
  GameDoc,
  GameWeather,
  InjurySnapshot,
  LineupData,
  LineupPlayer,
  MarketOdds,
  PitcherInfo,
  PowerRanking,
  RawGame,
  RunMarginCalibration,
  RunProjection,
  TeamState,
  TodaysRecord,
  TrainedModel,
} from "./ml/types";

const MLB_BASE = "https://statsapi.mlb.com";
const SEASON_START_MD = "03-15";
const UPCOMING_WINDOW_DAYS = 3;
const RECENT_WINDOW_DAYS = 7;
const TRAIN_SEASONS = ["2022", "2023", "2024", "2025"];
const PAST_SEASON_END_MD = "11-01";
const RUN_SIM_TRIALS = 10000;
const RUN_CALIB_TRIALS = 500;

// ---------------------------------------------------------------------------
// HTTP / date helpers
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function fetchJson(url: string, attempt = 0): Promise<any> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 25000);
    const res = await fetch(url, {
      signal: controller.signal,
      headers: { "User-Agent": "FreebuffMLB/1.0" },
    });
    clearTimeout(timer);
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    return await res.json();
  } catch (e) {
    if (attempt < 2) {
      await sleep(500 * (attempt + 1));
      return fetchJson(url, attempt + 1);
    }
    throw e;
  }
}

function toYmd(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function etDateString(d: Date): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(d);
}

function addDays(ymd: string, days: number): string {
  const d = new Date(`${ymd}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return toYmd(d);
}

function dateRanges(start: string, end: string, chunkDays: number): { start: string; end: string }[] {
  const ranges: { start: string; end: string }[] = [];
  const endD = new Date(`${end}T00:00:00Z`);
  let cur = new Date(`${start}T00:00:00Z`);
  while (cur <= endD) {
    const next = new Date(cur.getTime() + (chunkDays - 1) * 86400000);
    const chunkEnd = next > endD ? endD : next;
    ranges.push({ start: toYmd(cur), end: toYmd(chunkEnd) });
    cur = new Date(next.getTime() + 86400000);
  }
  return ranges;
}

async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let idx = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (idx < items.length) {
      const i = idx++;
      results[i] = await fn(items[i]);
    }
  });
  await Promise.all(workers);
  return results;
}

/**
 * Best-effort refresh-progress writer. It must never throw: an unhandled
 * rejection from a failed progress write would surface as `Uncaught
 * unhandledRejection` and kill the whole refresh action.
 */
function progressReporter(ctx: any) {
  return async (
    stage: string,
    pct: number,
    message: string,
    extra: { done?: boolean; error?: string } = {},
  ) => {
    try {
      await ctx.runMutation(internal.mlb.setRefreshProgress, {
        stage,
        pct,
        message,
        ...extra,
      });
    } catch {
      // ignore — progress is informational
    }
  };
}

// ---------------------------------------------------------------------------
// Schedule parsing
// ---------------------------------------------------------------------------

function scheduleUrl(start: string, end: string): string {
  return `${MLB_BASE}/api/v1/schedule?sportId=1&startDate=${start}&endDate=${end}&hydrate=probablePitcher,linescore,weather`;
}

/** Parse a probable starter, keeping the throwing hand for platoon features. */
function parsePitcher(p: any): PitcherInfo | undefined {
  if (!p?.id) return undefined;
  const out: PitcherInfo = { id: p.id, name: p.fullName ?? "" };
  const code = p.pitchHand?.code ?? "";
  const desc = p.pitchHand?.description ?? "";
  if (code === "L" || code === "R") out.pitchHand = code;
  else if (/^L/i.test(desc)) out.pitchHand = "L";
  else if (/^R/i.test(desc)) out.pitchHand = "R";
  return out;
}

function parseGame(g: any): RawGame | null {
  const away = g.teams?.away;
  const home = g.teams?.home;
  if (!away?.team?.id || !home?.team?.id) return null;
  const awayMeta = teamMeta(away.team.id);
  const homeMeta = teamMeta(home.team.id);
  let winner: "home" | "away" | undefined;
  if (away.isWinner === true) winner = "away";
  else if (home.isWinner === true) winner = "home";
  const innings = typeof g.linescore?.currentInning === "number" ? g.linescore.currentInning : 0;
  return {
    gamePk: g.gamePk,
    date: g.officialDate ?? (g.gameDate ? g.gameDate.slice(0, 10) : ""),
    gameDate: g.gameDate ?? "",
    dayNight: g.dayNight ?? "day",
    status: g.status?.abstractGameState ?? "Scheduled",
    detailedState: g.status?.detailedState,
    away: {
      id: away.team.id,
      abbrev: awayMeta.abbrev,
      name: awayMeta.name,
      score: typeof away.score === "number" ? away.score : undefined,
      wins: typeof away.leagueRecord?.wins === "number" ? away.leagueRecord.wins : undefined,
      losses: typeof away.leagueRecord?.losses === "number" ? away.leagueRecord.losses : undefined,
    },
    home: {
      id: home.team.id,
      abbrev: homeMeta.abbrev,
      name: homeMeta.name,
      score: typeof home.score === "number" ? home.score : undefined,
      wins: typeof home.leagueRecord?.wins === "number" ? home.leagueRecord.wins : undefined,
      losses: typeof home.leagueRecord?.losses === "number" ? home.leagueRecord.losses : undefined,
    },
    awayPitcher: parsePitcher(away.probablePitcher),
    homePitcher: parsePitcher(home.probablePitcher),
    venue: g.venue?.name,
    innings: innings > 9 ? innings : undefined,
    winner,
    season: g.season,
    weather: parseWeather(g.weather),
  };
}

function parseWeather(w: any): GameWeather | undefined {
  if (!w) return undefined;
  const temp = typeof w.temp === "string" ? parseFloat(w.temp) : w.temp;
  const wind = typeof w.wind?.speed === "string" ? parseFloat(w.wind.speed) : w.wind?.speed;
  const out: GameWeather = {};
  if (typeof w.condition === "string") out.condition = w.condition;
  if (Number.isFinite(temp)) out.tempF = temp;
  if (Number.isFinite(wind)) out.windMph = wind;
  return Object.keys(out).length > 0 ? out : undefined;
}

function parseSchedule(data: any): RawGame[] {
  const out: RawGame[] = [];
  for (const d of data?.dates ?? []) {
    for (const g of d.games ?? []) {
      if (g.gameType !== "R") continue;
      const parsed = parseGame(g);
      if (parsed) out.push(parsed);
    }
  }
  return out;
}

async function fetchScheduleRange(start: string, end: string): Promise<RawGame[]> {
  const data = await fetchJson(scheduleUrl(start, end));
  return parseSchedule(data);
}

async function fetchSeason(season: string, throughDate: string): Promise<RawGame[]> {
  const start = `${season}-${SEASON_START_MD}`;
  const ranges = dateRanges(start, throughDate, 30);
  const seen = new Map<number, RawGame>();
  await mapLimit(ranges, 8, async (r) => {
    const games = await fetchScheduleRange(r.start, r.end);
    for (const g of games) seen.set(g.gamePk, g);
  });
  return [...seen.values()].sort((a, b) => (a.gameDate < b.gameDate ? -1 : 1));
}

/** Fetch every training season's schedule window in parallel. */
async function fetchAllSeasons(
  seasons: string[],
  currentSeason: string,
  throughDate: string,
  onProgress?: (done: number, total: number) => void): Promise<RawGame[]> {
  const all: RawGame[] = [];
  let completed = 0;
  await mapLimit(seasons, Math.min(3, seasons.length), async (s) => {
    const end = s === currentSeason ? throughDate : `${s}-${PAST_SEASON_END_MD}`;
    const games = await fetchSeason(s, end);
    all.push(...games);
    completed += 1;
    if (onProgress) onProgress(completed, seasons.length);
  });
  return all.sort((a, b) => (a.gameDate < b.gameDate ? -1 : 1));
}

/** "187.2" → 187.666… (baseball innings convention: .1 = ⅓, .2 = ⅔). */
function inningsPitchedValue(ip: unknown): number {
  if (typeof ip === "number") return ip;
  if (typeof ip === "string") {
    const [whole, frac] = ip.split(".");
    return (Number(whole) || 0) + (frac === "1" ? 1 / 3 : frac === "2" ? 2 / 3 : 0);
  }
  return 0;
}

/** FIP constant keeps values ERA-like; it cancels out in the feature delta. */
const FIP_CONSTANT = 3.1;

// ---------------------------------------------------------------------------
// As-of-date game-log stats (pitcher / team / batter) — no lookahead
//
// The season-total endpoints leak information from AFTER a game's date into
// that game's features (a July game would see August stats). For honest
// training the pipeline accumulates each entity's per-game game log strictly
// BEFORE the target game's date. Logs are cached on the model state, so any
// as-of date is a cheap local sum and refreshes only fetch missing seasons.
// ---------------------------------------------------------------------------

interface PitcherLogEntry {
  d: string; // game date YYYY-MM-DD
  ip: number;
  er: number;
  h: number;
  so: number;
  bb: number;
  hbp: number;
  hr: number;
}

interface HittingLogEntry {
  d: string;
  ab: number;
  h: number;
  bb: number;
  ibb: number;
  hbp: number;
  sf: number;
  tb: number;
  "2b": number;
  "3b": number;
  hr: number;
}

interface FieldingLogEntry {
  d: string;
  po: number;
  a: number;
  e: number;
}

interface TeamGameLog {
  hitting?: HittingLogEntry[];
  pitching?: PitcherLogEntry[];
  fielding?: FieldingLogEntry[];
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function round3(n: number): number {
  return Math.round(n * 1000) / 1000;
}

function logNum(value: unknown): number {
  if (typeof value === "string") {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : 0;
  }
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function compactPitcherEntry(split: any): PitcherLogEntry | null {
  const st = split?.stat ?? {};
  const ip = inningsPitchedValue(st.inningsPitched);
  if (ip <= 0) return null;
  return {
    d: split?.date ?? "",
    ip: Math.round(ip * 1000) / 1000,
    er: logNum(st.earnedRuns),
    h: logNum(st.hits),
    so: logNum(st.strikeOuts),
    bb: logNum(st.baseOnBalls),
    hbp: logNum(st.hitByPitch),
    hr: logNum(st.homeRuns),
  };
}

function compactHittingEntry(split: any): HittingLogEntry | null {
  const st = split?.stat ?? {};
  return {
    d: split?.date ?? "",
    ab: logNum(st.atBats),
    h: logNum(st.hits),
    bb: logNum(st.baseOnBalls),
    ibb: logNum(st.intentionalWalks),
    hbp: logNum(st.hitByPitch),
    sf: logNum(st.sacFlies),
    tb: logNum(st.totalBases),
    "2b": logNum(st.doubles),
    "3b": logNum(st.triples),
    hr: logNum(st.homeRuns),
  };
}

function compactFieldingEntry(split: any): FieldingLogEntry | null {
  const st = split?.stat ?? {};
  return {
    d: split?.date ?? "",
    po: logNum(st.putOuts),
    a: logNum(st.assists),
    e: logNum(st.errors),
  };
}

/** Fetch per-game pitching logs for `{id|season}` pairs, reusing the cache. */
async function fetchPitcherGameLogs(
  pairs: { id: number; season: string }[],
  cached: Record<string, PitcherLogEntry[]> = {},
): Promise<Record<string, PitcherLogEntry[]>> {
  const out: Record<string, PitcherLogEntry[]> = {};
  const seen = new Set<string>();
  const unique = pairs.filter((p) => {
    const key = `${p.id}|${p.season}`;
    if (p.id <= 0 || seen.has(key) || cached[key]) return false;
    seen.add(key);
    return true;
  });
  await mapLimit(unique, 16, async (p) => {
    try {
      const data = await fetchJson(
        `${MLB_BASE}/api/v1/people/${p.id}/stats?stats=gameLog&group=pitching&season=${p.season}&gameType=R`,
      );
      const splits = data?.stats?.[0]?.splits ?? [];
      const entries = splits
        .map((s: any) => compactPitcherEntry(s))
        .filter((e: PitcherLogEntry | null): e is PitcherLogEntry => !!e && !!e.d);
      entries.sort((a: PitcherLogEntry, b: PitcherLogEntry) => (a.d < b.d ? -1 : 1));
      if (entries.length > 0) out[`${p.id}|${p.season}`] = entries;
    } catch {
      // individual log failures are non-fatal
    }
  });
  return out;
}

/** Fetch per-game team logs (hitting / pitching / fielding), reusing the cache. */
async function fetchTeamGameLogs(
  pairs: { id: number; season: string }[],
  cached: Record<string, TeamGameLog> = {},
): Promise<Record<string, TeamGameLog>> {
  const out: Record<string, TeamGameLog> = {};
  const seen = new Set<string>();
  const unique = pairs.filter((p) => {
    const key = `${p.id}|${p.season}`;
    if (p.id <= 0 || seen.has(key) || cached[key]) return false;
    seen.add(key);
    return true;
  });
  await mapLimit(unique, 16, async (p) => {
    try {
      const data = await fetchJson(
        `${MLB_BASE}/api/v1/teams/${p.id}/stats?stats=gameLog&group=hitting,pitching,fielding&season=${p.season}&gameType=R`,
      );
      const result: TeamGameLog = {};
      for (const block of data?.stats ?? []) {
        const group = block?.group?.displayName ?? "";
        const splits = block?.splits ?? [];
        if (group === "hitting") {
          result.hitting = splits
            .map((s: any) => compactHittingEntry(s))
            .filter((e: HittingLogEntry | null): e is HittingLogEntry => !!e && !!e.d);
        } else if (group === "pitching") {
          result.pitching = splits
            .map((s: any) => compactPitcherEntry(s))
            .filter((e: PitcherLogEntry | null): e is PitcherLogEntry => !!e && !!e.d);
        } else if (group === "fielding") {
          result.fielding = splits
            .map((s: any) => compactFieldingEntry(s))
            .filter((e: FieldingLogEntry | null): e is FieldingLogEntry => !!e && !!e.d);
        }
      }
      if (Object.keys(result).length > 0) out[`${p.id}|${p.season}`] = result;
    } catch {
      // non-fatal
    }
  });
  return out;
}

/** Fetch per-game batting logs for batters in a season, reusing the cache. */
async function fetchBatterGameLogs(
  ids: number[],
  season: string,
  cached: Record<string, HittingLogEntry[]> = {},
): Promise<Record<string, HittingLogEntry[]>> {
  const out: Record<string, HittingLogEntry[]> = {};
  const unique = [...new Set(ids)].filter((id) => id > 0 && !cached[`${id}|${season}`]);
  await mapLimit(unique, 24, async (id) => {
    try {
      const data = await fetchJson(
        `${MLB_BASE}/api/v1/people/${id}/stats?stats=gameLog&group=hitting&season=${season}&gameType=R`,
      );
      const splits = data?.stats?.[0]?.splits ?? [];
      const entries = splits
        .map((s: any) => compactHittingEntry(s))
        .filter((e: HittingLogEntry | null): e is HittingLogEntry => !!e && !!e.d);
      entries.sort((a: HittingLogEntry, b: HittingLogEntry) => (a.d < b.d ? -1 : 1));
      if (entries.length > 0) out[`${id}|${season}`] = entries;
    } catch {
      // non-fatal
    }
  });
  return out;
}

/**
 * Season ERA / K9 / FIP / WHIP + recent-form ERA, strictly before `ymd`.
 * `recentEra` covers the pitcher's last `recentStarts` starts (hot/cold
 * starter signal).
 */
function pitcherAsOf(entries: PitcherLogEntry[] | undefined, ymd: string, recentStarts = 3): Partial<PitcherInfo> {
  if (!entries || entries.length === 0) return {};
  let ip = 0;
  let er = 0;
  let so = 0;
  let bb = 0;
  let hbp = 0;
  let hr = 0;
  let h = 0;
  const recent: PitcherLogEntry[] = [];
  for (const e of entries) {
    if (e.d >= ymd) break;
    ip += e.ip;
    er += e.er;
    so += e.so;
    bb += e.bb;
    hbp += e.hbp;
    hr += e.hr;
    h += e.h;
    recent.push(e);
    if (recent.length > recentStarts) recent.shift();
  }
  if (ip <= 0) return {};
  const out: Partial<PitcherInfo> = {
    era: round2((er * 9) / ip),
    k9: round2((so * 9) / ip),
    fip: round2((13 * hr + 3 * (bb + hbp) - 2 * so) / ip + FIP_CONSTANT),
    whip: round2((bb + h) / ip),
  };
  const rIp = recent.reduce((s, e) => s + e.ip, 0);
  const rEr = recent.reduce((s, e) => s + e.er, 0);
  if (rIp > 0) out.recentEra = round2((rEr * 9) / rIp);
  return out;
}

function opsFromHitting(ab: number, h: number, bb: number, hbp: number, sf: number, tb: number): number | undefined {
  if (ab + bb + hbp + sf <= 0) return undefined;
  const obp = (h + bb + hbp) / (ab + bb + hbp + sf);
  const slg = ab > 0 ? tb / ab : 0;
  return obp + slg;
}

// FanGraphs-style wOBA weights (2019+ scale). Higher is better for hitters.
const WOBA_UBB = 0.69;
const WOBA_HBP = 0.72;
const WOBA_1B = 0.89;
const WOBA_2B = 1.27;
const WOBA_3B = 1.62;
const WOBA_HR = 2.1;

interface HittingTotals {
  ab: number;
  h: number;
  bb: number;
  ibb: number;
  hbp: number;
  sf: number;
  tb: number;
  "2b": number;
  "3b": number;
  hr: number;
}

function wobaFromHitting(t: HittingTotals): number | undefined {
  const ubb = Math.max(0, t.bb - t.ibb);
  const den = t.ab + ubb + t.sf + t.hbp;
  if (den <= 0) return undefined;
  const singles = Math.max(0, t.h - t["2b"] - t["3b"] - t.hr);
  const num =
    WOBA_UBB * ubb +
    WOBA_HBP * t.hbp +
    WOBA_1B * singles +
    WOBA_2B * t["2b"] +
    WOBA_3B * t["3b"] +
    WOBA_HR * t.hr;
  return num / den;
}

function isoFromHitting(h: number, tb: number, ab: number): number | undefined {
  if (ab <= 0) return undefined;
  return (tb - h) / ab;
}

function hittingTotalsAsOf(entries: HittingLogEntry[] | undefined, ymd: string): HittingTotals {
  const t: HittingTotals = { ab: 0, h: 0, bb: 0, ibb: 0, hbp: 0, sf: 0, tb: 0, "2b": 0, "3b": 0, hr: 0 };
  for (const e of entries ?? []) {
    if (e.d >= ymd) break;
    t.ab += e.ab;
    t.h += e.h;
    t.bb += e.bb;
    t.ibb += e.ibb;
    t.hbp += e.hbp;
    t.sf += e.sf;
    t.tb += e.tb;
    t["2b"] += e["2b"];
    t["3b"] += e["3b"];
    t.hr += e.hr;
  }
  return t;
}

/** Batter season OPS accumulated strictly before `ymd`; undefined with no prior games. */
function batterOpsAsOf(entries: HittingLogEntry[] | undefined, ymd: string): number | undefined {
  const t = hittingTotalsAsOf(entries, ymd);
  const v = opsFromHitting(t.ab, t.h, t.bb, t.hbp, t.sf, t.tb);
  return v === undefined ? undefined : round3(v);
}

/** Batter wOBA accumulated strictly before `ymd`; undefined with no prior games. */
function batterWobaAsOf(entries: HittingLogEntry[] | undefined, ymd: string): number | undefined {
  const v = wobaFromHitting(hittingTotalsAsOf(entries, ymd));
  return v === undefined ? undefined : round3(v);
}

/** Batter isolated power accumulated strictly before `ymd`; undefined with no prior games. */
function batterIsoAsOf(entries: HittingLogEntry[] | undefined, ymd: string): number | undefined {
  const t = hittingTotalsAsOf(entries, ymd);
  const v = isoFromHitting(t.h, t.tb, t.ab);
  return v === undefined ? undefined : round3(v);
}

/** OPS over the batter's last `window` games before `ymd` (hot-streak signal). */
function batterRecentOpsAsOf(entries: HittingLogEntry[] | undefined, ymd: string, window = 10): number | undefined {
  if (!entries || entries.length === 0) return undefined;
  const recent = entries.filter((e) => e.d < ymd).slice(-window);
  if (recent.length === 0) return undefined;
  const t = hittingTotalsAsOf(recent, "9999-12-31");
  const v = opsFromHitting(t.ab, t.h, t.bb, t.hbp, t.sf, t.tb);
  return v === undefined ? undefined : round3(v);
}

/**
 * Team OPS / staff ERA / K9 / WHIP / fielding pct accumulated strictly before
 * `ymd` (no lookahead).
 */
function teamAsOf(log: TeamGameLog | undefined, ymd: string): Partial<TeamInfo> {
  if (!log) return {};
  const out: Partial<TeamInfo> = {};

  let ab = 0;
  let h = 0;
  let bb = 0;
  let hbp = 0;
  let sf = 0;
  let tb = 0;
  for (const e of log.hitting ?? []) {
    if (e.d >= ymd) break;
    ab += e.ab;
    h += e.h;
    bb += e.bb;
    hbp += e.hbp;
    sf += e.sf;
    tb += e.tb;
  }
  const ops = opsFromHitting(ab, h, bb, hbp, sf, tb);
  if (ops !== undefined) out.ops = round3(ops);

  let ip = 0;
  let er = 0;
  let so = 0;
  let pbb = 0;
  let ph = 0;
  for (const e of log.pitching ?? []) {
    if (e.d >= ymd) break;
    ip += e.ip;
    er += e.er;
    so += e.so;
    pbb += e.bb;
    ph += e.h;
  }
  if (ip > 0) {
    out.era = round2((er * 9) / ip);
    out.k9 = round2((so * 9) / ip);
    out.whip = round2((pbb + ph) / ip);
  }

  let po = 0;
  let a = 0;
  let err = 0;
  for (const e of log.fielding ?? []) {
    if (e.d >= ymd) break;
    po += e.po;
    a += e.a;
    err += e.e;
  }
  const chances = po + a + err;
  if (chances > 0) out.fieldingPct = round3((po + a) / chances);
  return out;
}

/** Attach per-game as-of-date pitcher + team stats (no lookahead). */
function attachAsOfStats(
  games: RawGame[],
  pitcherLogs: Record<string, PitcherLogEntry[]>,
  teamLogs: Record<string, TeamGameLog>,
): RawGame[] {
  return games.map((g) => {
    const season = g.season ?? "";
    const ymd = g.date;
    const withPitcher = (p?: PitcherInfo): PitcherInfo | undefined => {
      if (!p) return p;
      return { ...p, ...pitcherAsOf(pitcherLogs[`${p.id}|${season}`], ymd) };
    };
    return {
      ...g,
      awayPitcher: withPitcher(g.awayPitcher),
      homePitcher: withPitcher(g.homePitcher),
      home: { ...g.home, ...teamAsOf(teamLogs[`${g.home.id}|${season}`], ymd) },
      away: { ...g.away, ...teamAsOf(teamLogs[`${g.away.id}|${season}`], ymd) },
    };
  });
}

// ---------------------------------------------------------------------------
// Market odds (The Odds API) — optional, reads THE_ODDS_API_KEY
// ---------------------------------------------------------------------------

const ODDS_API_KEY = process.env.THE_ODDS_API_KEY;
const ODDS_BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds";

function oddsNum(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  return undefined;
}

function pickBookmaker(bookmakers: any[]): any | undefined {
  if (!Array.isArray(bookmakers) || bookmakers.length === 0) return undefined;
  const preferred = ["pinnacle", "draftkings", "fanduel", "betmgm"];
  for (const p of preferred) {
    const b = bookmakers.find((x) => x?.key === p);
    if (b) return b;
  }
  return bookmakers[0];
}

function oddsFromEvent(event: any): { date: string; home: string; away: string; odds: MarketOdds } | null {
  const home = event?.home_team;
  const away = event?.away_team;
  if (typeof home !== "string" || typeof away !== "string") return null;
  const book = pickBookmaker(event?.bookmakers);
  const markets = book?.markets;
  if (!Array.isArray(markets)) return null;
  const odds: MarketOdds = {};
  for (const m of markets) {
    const key = m?.key;
    const outcomes: any[] = m?.outcomes ?? [];
    if (key === "h2h") {
      for (const o of outcomes) {
        if (o?.name === home) odds.homeMoneyline = oddsNum(o.price);
        else if (o?.name === away) odds.awayMoneyline = oddsNum(o.price);
      }
    } else if (key === "totals") {
      for (const o of outcomes) {
        const pt = oddsNum(o?.point);
        if (pt !== undefined) odds.total = pt;
        if (o?.name === "Over") odds.overPrice = oddsNum(o.price);
        if (o?.name === "Under") odds.underPrice = oddsNum(o.price);
      }
    } else if (key === "spreads") {
      for (const o of outcomes) {
        const pt = oddsNum(o?.point);
        if (pt !== undefined) odds.runLine = Math.abs(pt);
        if (o?.name === home) odds.homeRunLinePrice = oddsNum(o.price);
        else if (o?.name === away) odds.awayRunLinePrice = oddsNum(o.price);
      }
    }
  }
  if (Object.keys(odds).length === 0) return null;
  odds.source = "The Odds API";
  const commence = event?.commence_time;
  const date = commence ? etDateString(new Date(commence)) : "";
  return { date, home, away, odds };
}

async function fetchMarketOdds(): Promise<Map<string, MarketOdds>> {
  const out = new Map<string, MarketOdds>();
  if (!ODDS_API_KEY) return out;
  try {
    const url = `${ODDS_BASE}/?apiKey=${encodeURIComponent(ODDS_API_KEY)}&regions=us&markets=h2h,totals,spreads&oddsFormat=american`;
    const data = await fetchJson(url);
    for (const event of data ?? []) {
      const parsed = oddsFromEvent(event);
      if (!parsed || !parsed.date) continue;
      out.set(`${parsed.date}|${parsed.home}|${parsed.away}`, parsed.odds);
    }
  } catch {
    // market odds are best-effort; never fail a model refresh because of them
  }
  return out;
}

function marketOddsForGame(odds: Map<string, MarketOdds>, game: RawGame): MarketOdds | undefined {
  if (odds.size === 0) return undefined;
  const homeFull = teamMeta(game.home.id).fullName;
  const awayFull = teamMeta(game.away.id).fullName;
  return odds.get(`${game.date}|${homeFull}|${awayFull}`) ?? odds.get(`${game.date}|${awayFull}|${homeFull}`);
}

// ---------------------------------------------------------------------------
// Run projections (predicted scores, totals, run lines)
// ---------------------------------------------------------------------------

/**
 * Shift the two Poisson means (total preserved) so the run-scoring model's
 * expected margin matches the win-probability model. This prevents an
 * underdog being shown to score more runs than the favorite.
 */
function marginShiftForGame(
  runModel: RunModel,
  cal: RunMarginCalibration,
  homeId: number,
  awayId: number,
  homeWinProb: number): number {
  if (!cal || (cal.slope === 0 && cal.intercept === 0)) return 0;
  const baseMargin = expectedMargin(runModel, homeId, awayId);
  const p = Math.min(0.99, Math.max(0.01, homeWinProb));
  const targetMargin = cal.intercept + cal.slope * logit(p);
  return (targetMargin - baseMargin) / 2;
}

function buildRunProjection(
  runModel: RunModel,
  runLineIso: { x: number; y: number }[],
  game: RawGame,
  marketTotal?: number,
  marketRunLine?: number,
  trials = RUN_SIM_TRIALS,
  homeWinProb = 0.5,
  runMarginCal: RunMarginCalibration = { slope: 0, intercept: 0 }): RunProjection {
  const runLine = marketRunLine ?? 1.5;
  const marginShift = marginShiftForGame(runModel, runMarginCal, game.home.id, game.away.id, homeWinProb);
  const line = marketTotal ?? expectedTotal(runModel, game.home.id, game.away.id);
  const sim = simulateRuns(runModel, game.home.id, game.away.id, line, trials, runLine, marginShift);
  // Isotonic calibration is fit on the ±1.5 run line; other lines use raw MC.
  const homeRL =
    runLineIso.length > 0 && runLine === 1.5
      ? applyIsotonic(runLineIso, sim.homeRunLineProb)
      : sim.homeRunLineProb;
  const home = Math.min(0.999, Math.max(0.001, homeRL));
  return {
    homeScore: Math.round(sim.homeScore * 100) / 100,
    awayScore: Math.round(sim.awayScore * 100) / 100,
    total: Math.round(sim.total * 100) / 100,
    overProb: sim.overProb,
    underProb: sim.underProb,
    homeRunLineProb: home,
    awayRunLineProb: 1 - home,
  };
}

// ---------------------------------------------------------------------------
// Injury data (single consolidated source: MLB Stats API rosters)
// ---------------------------------------------------------------------------

const INJURY_SNAPSHOT_DAYS = 28;

/** Whether a roster entry's status indicates an injured-list / day-to-day stint. */
function isInjuredStatus(status: { code?: string; description?: string } | undefined): boolean {
  if (!status) return false;
  const code = typeof status.code === "string" ? status.code : "";
  const description = typeof status.description === "string" ? status.description.toLowerCase() : "";
  return (
    /^D/.test(code) ||
    /^IL/.test(code) ||
    description.includes("injured") ||
    description.includes("day-to-day")
  );
}

async function fetchInjuryCount(teamId: number, date: string, season: string): Promise<number> {
  try {
    const data = await fetchJson(
      `${MLB_BASE}/api/v1/teams/${teamId}/roster?rosterType=40Man&season=${season}&date=${date}`,
    );
    let count = 0;
    for (const entry of data?.roster ?? []) {
      if (isInjuredStatus(entry.status)) count += 1;
    }
    return count;
  } catch {
    return 0;
  }
}

/**
 * Weekly per-team injured-list snapshots (plus the final as-of date) so the
 * injury feature is available historically without lookahead bias.
 */
async function fetchInjurySnapshots(
  teamIds: number[],
  season: string,
  startDate: string,
  endDate: string,
  previous?: Record<string, InjurySnapshot[]>): Promise<Map<number, InjurySnapshot[]>> {
  const dates: string[] = [];
  let cursor = startDate;
  while (cursor <= endDate) {
    dates.push(cursor);
    cursor = addDays(cursor, INJURY_SNAPSHOT_DAYS);
  }
  if (dates[dates.length - 1] !== endDate) dates.push(endDate);

  // Reuse snapshots cached on the model state and only fetch missing dates,
  // so an on-demand refresh makes ~30 roster requests instead of ~360+.
  const out = new Map<number, InjurySnapshot[]>();
  const jobs: { teamId: number; date: string }[] = [];
  for (const teamId of teamIds) {
    const cached = previous?.[String(teamId)] ?? [];
    const have = new Set(cached.map((s) => s.date));
    const list = [...cached];
    for (const date of dates) {
      if (!have.has(date)) jobs.push({ teamId, date });
    }
    out.set(teamId, list);
  }

  const results = await mapLimit(jobs, 16, async (job) => ({
    teamId: job.teamId,
    date: job.date,
    count: await fetchInjuryCount(job.teamId, job.date, season),
  }));

  for (const r of results) {
    out.get(r.teamId)!.push({ date: r.date, count: r.count });
  }
  for (const list of out.values()) list.sort((a, b) => (a.date < b.date ? -1 : 1));
  return out;
}

/**
 * Apply the result of a freshly completed game to a team state (Elo, wins
 * / losses, lastGameDate). Called in chronological order during the fast
 * refresh path so pre-game Elo matches the previous state's post-game Elo
 * going into the next game on the same day.
 */
function applyGameResultToTeamState(
  teamState: TeamState,
  game: RawGame,
  eloHfa = 30,
): void {
  if (game.winner !== "home" && game.winner !== "away") return;
  const homeElo = teamState.elo[game.home.id] ?? 1500;
  const awayElo = teamState.elo[game.away.id] ?? 1500;
  const expectedHome = 1 / (1 + Math.pow(10, -((homeElo + eloHfa) - awayElo) / 400));
  const homeActual = game.winner === "home" ? 1 : 0;
  const margin = Math.abs((game.home.score ?? 0) - (game.away.score ?? 0));
  const k = 24 * Math.sqrt(Math.max(1, margin));
  const delta = k * (homeActual - expectedHome);
  teamState.elo[game.home.id] = homeElo + delta;
  teamState.elo[game.away.id] = awayElo - delta;

  const hr = teamState.records[game.home.id] ?? { wins: 0, losses: 0 };
  const ar = teamState.records[game.away.id] ?? { wins: 0, losses: 0 };
  if (homeActual === 1) {
    hr.wins += 1;
    ar.losses += 1;
  } else {
    hr.losses += 1;
    ar.wins += 1;
  }
  teamState.records[game.home.id] = hr;
  teamState.records[game.away.id] = ar;
  teamState.lastGameDate[game.home.id] = game.date;
  teamState.lastGameDate[game.away.id] = game.date;
}

/** Lightweight current-day injury snapshot (one roster call per team). */
async function fetchCurrentInjurySnapshot(
  teamIds: number[],
  date: string,
  season: string): Promise<Map<number, number>> {
  const pairs = [...new Set(teamIds)].filter((id) => id > 0);
  const out = new Map<number, number>();
  await mapLimit(pairs, 16, async (teamId) => {
    out.set(teamId, await fetchInjuryCount(teamId, date, season));
  });
  return out;
}

/**
 * Fast refresh path: when a trained model already exists, just update the
 * team state from the last week of new results and re-predict the upcoming
 * window. Skips loadStoredGames, full schedule history, retraining, and
 * bulk injury history fetches — the action completes in seconds, not
 * minutes. The model weights stay frozen as of the last retrain. Pitcher /
 * team / batter stats come from the cached as-of game logs (only missing
 * logs are fetched), so every feature stays strictly no-lookahead.
 */
async function fastRefresh(
  ctx: any,
  previous: any,
  season: string,
  today: string,
  report: (stage: string, pct: number, message: string, extra?: { done?: boolean; error?: string }) => Promise<void>,
): Promise<void> {
  await report("Reading previous model", 6, "Reusing the trained model…");
  const model = reconstructModel(previous);
  const runModelState = previous.runModel as RunModel | undefined;
  const runLineIso = (previous.runLineCalibration ?? []) as { x: number; y: number }[];
  const runMarginCal = (previous.runMarginCalibration ?? { slope: 0, intercept: 0 }) as RunMarginCalibration;
  const pitcherLogs = (previous.pitcherLogs ?? {}) as Record<string, PitcherLogEntry[]>;
  const teamLogs = (previous.teamLogs ?? {}) as Record<string, TeamGameLog>;
  const batterLogs = (previous.batterLogs ?? {}) as Record<string, HittingLogEntry[]>;

  await report("Fetching fresh games", 18, "Loading the last week + upcoming schedule…");
  const freshRaw = await fetchScheduleRange(
    addDays(today, -RECENT_WINDOW_DAYS),
    addDays(today, UPCOMING_WINDOW_DAYS),
  );

  if (freshRaw.length === 0) {
    await report("No new games", 100, "Up to date; nothing to refresh.", { done: true });
    return;
  }

  await report("Pitcher logs", 28, "Loading as-of stats for fresh window starters…");
  const freshPitcherPairs: { id: number; season: string }[] = [];
  const seenPitcher = new Set<string>();
  for (const g of freshRaw) {
    const s = g.season ?? season;
    for (const p of [g.awayPitcher, g.homePitcher]) {
      if (p && p.id > 0) {
        const k = `${p.id}|${s}`;
        if (!seenPitcher.has(k)) {
          seenPitcher.add(k);
          freshPitcherPairs.push({ id: p.id, season: s });
        }
      }
    }
  }
  const newPitcherLogs = await fetchPitcherGameLogs(freshPitcherPairs, pitcherLogs);
  const pitcherLogsCache: Record<string, PitcherLogEntry[]> = { ...pitcherLogs, ...newPitcherLogs };

  await report("Team logs", 40, "Refreshing as-of team game logs…");
  const teamSeasonPairs = new Map<string, { id: number; season: string }>();
  for (const g of freshRaw) {
    const s = g.season ?? season;
    const hk = `${g.home.id}|${s}`;
    const ak = `${g.away.id}|${s}`;
    if (!teamSeasonPairs.has(hk)) teamSeasonPairs.set(hk, { id: g.home.id, season: s });
    if (!teamSeasonPairs.has(ak)) teamSeasonPairs.set(ak, { id: g.away.id, season: s });
  }
  const teamStatsToFetch = [...teamSeasonPairs.values()].filter((p) => {
    const key = `${p.id}|${p.season}`;
    if (p.season === season) return true;
    return teamLogs[key] === undefined;
  });
  const newTeamLogs = await fetchTeamGameLogs(teamStatsToFetch, teamLogs);
  const teamLogsCache: Record<string, TeamGameLog> = { ...teamLogs, ...newTeamLogs };

  await report("Updating team state", 48, "Applying Elo + records from new results…");
  const teamState = reconstructTeamState((previous.powerRankings ?? []) as PowerRanking[]);
  const freshCompleted = freshRaw
    .filter((g) => g.winner === "home" || g.winner === "away")
    .sort((a, b) => (a.gameDate < b.gameDate ? -1 : 1));
  for (const g of freshCompleted) applyGameResultToTeamState(teamState, g, model.eloHfa ?? 30);

  await report("Loading lineups", 58, "Loading actual starting lineups…");
  const lineupGames = freshRaw.filter(
    (g) => g.date >= addDays(today, -2) && g.date <= addDays(today, UPCOMING_WINDOW_DAYS),
  );
  const lineups = await fetchLineupsForGames(lineupGames, 16);
  const batterIds: number[] = [];
  for (const lu of lineups.values()) {
    for (const side of [lu.home, lu.away]) {
      if (!side) continue;
      for (const p of [...side.battingOrder, ...side.bench]) batterIds.push(p.id);
    }
  }
  const newBatterLogs = await fetchBatterGameLogs(batterIds, season, batterLogs);
  const batterLogsCache: Record<string, HittingLogEntry[]> = { ...batterLogs, ...newBatterLogs };

  await report("Loading matchups", 66, "Fetching BvP & platoon splits…");
  const lineupsEnriched = attachLineupsAsOf(freshRaw, lineups, batterLogsCache);
  const matchup = await enrichWithMatchups(
    lineupsEnriched,
    batterLogsCache,
    (previous.bvpLogs ?? {}) as BvpCache,
    (previous.platoonLogs ?? {}) as PlatoonCache,
    (previous.vsTeamLogs ?? {}) as VsTeamCache,
    season,
  );
  const enrichedWithLineups = matchup.games;
  const bvpLogsCache = matchup.bvpLogs;
  const platoonLogsCache = matchup.platoonLogs;
  const vsTeamLogsCache = matchup.vsTeamLogs;

  await report("Loading injuries", 68, "Loading current IL snapshot…");
  const teamIds = [...new Set(freshRaw.flatMap((g) => [g.home.id, g.away.id]))];
  const currentInjury = await fetchCurrentInjurySnapshot(teamIds, today, season);
  for (const [id, count] of currentInjury) teamState.injuries[id] = count;

  await report("Predicting upcoming", 78, "Scoring the upcoming schedule…");
  const enriched = attachAsOfStats(freshRaw, pitcherLogsCache, teamLogsCache);
  const marketOdds = await fetchMarketOdds();
  const freshByPk = new Map<number, RawGame>(enrichedWithLineups.map((g) => [g.gamePk, g]));

  // Upcoming games (no result yet) get a fresh pre-game prediction built from
  // the current team state — which already includes every result applied above.
  const emptyPitcherStats = new Map<string, PitcherSeasonStats>();
  const freshDocsByPk = new Map<number, GameDoc>();
  for (const g of enrichedWithLineups) {
    if (g.winner === "home" || g.winner === "away") continue;
    const odds = marketOddsForGame(marketOdds, g);
    const pred = predictionFor(g, model, teamState);
    freshDocsByPk.set(
      g.gamePk,
      buildGameDoc(
        g,
        pred,
        emptyPitcherStats,
        { home: teamState.injuries[g.home.id] ?? 0, away: teamState.injuries[g.away.id] ?? 0 },
        runModelState ? buildRunProjection(runModelState, runLineIso, g, odds?.total, odds?.runLine, RUN_SIM_TRIALS, pred.homeWinProb, runMarginCal) : undefined,
        odds,
      ),
    );
  }

  await report("Saving", 92, "Persisting refreshed predictions…");
  const datesToWrite = [...new Set(enrichedWithLineups.map((g) => g.date))];
  const newlyCompleted: GameDoc[] = [];
  const calibrationRowsByDate = new Map<string, CalibrationRow[]>();
  await mapLimit(datesToWrite, 8, async (date) => {
    const existingDocs = (await ctx.runQuery(api.mlb.getGamesByDate, { date })) as GameDoc[];
    const existingByPk = new Map<number, GameDoc>(existingDocs.map((d) => [d.gamePk, d]));
    const merged: GameDoc[] = [];

    // 1. Stored docs: record results on games that have now completed (without
    //    re-predicting them, so pre-game predictions stay honest for calibration).
    for (const [pk, doc] of existingByPk) {
      const fresh = freshByPk.get(pk);
      if (fresh && (fresh.winner === "home" || fresh.winner === "away") && !doc.winner) {
        const updated: GameDoc = {
          ...doc,
          winner: fresh.winner,
          away: { ...doc.away, score: fresh.away.score },
          home: { ...doc.home, score: fresh.home.score },
        };
        updated.isCorrect = updated.pickTeam === fresh.winner;
        updated.isUpset = updated.pickTeam !== fresh.winner;
        newlyCompleted.push(updated);
        merged.push(updated);
      } else if (!fresh) {
        // Game from a previous fetch not in this window — keep the stored doc.
        merged.push(doc);
      } else if (doc.winner) {
        // Already completed and recorded in a previous refresh — keep as-is.
        merged.push(doc);
      }
      // Stored upcoming docs are replaced by the fresh prediction below.
    }

    // 2. Fresh docs for this date (upcoming predictions; completed games
    //    without a stored doc would be added here too if present).
    for (const g of enrichedWithLineups) {
      if (g.date !== date) continue;
      const doc = freshDocsByPk.get(g.gamePk);
      if (doc) merged.push(doc);
    }

    await ctx.runMutation(internal.mlb.replaceGamesForDate, { date, games: merged });

    // Keep the compact calibration projection in sync for this date.
    const completedMerged = merged.filter((d) => d.winner === "home" || d.winner === "away");
    if (completedMerged.length > 0) {
      calibrationRowsByDate.set(date, completedMerged.map(calibrationRowFromDoc));
    }
  });

  // Today's record from the stored (now completed) games for today.
  const todayStored = (await ctx.runQuery(api.mlb.getGamesByDate, { date: today })) as GameDoc[];
  const completedToday = todayStored.filter((d) => d.winner === "home" || d.winner === "away");
  const todaysRecord =
    completedToday.length > 0 || newlyCompleted.length === 0
      ? buildTodaysRecord(completedToday, today)
      : previous.todaysRecord;

  // Calibration projection: rewrite the fresh window's compact rows every
  // refresh. The one-time full-history backfill is NOT done here — it runs as
  // a separate scheduled action (backfillCalibration) so this action stays
  // well inside Convex's 10-minute Node action limit.
  await report("Updating calibration", 94, "Writing calibration rows for the fresh window…");
  const calibrationDates = [...calibrationRowsByDate.keys()];
  const calibrationTotal = calibrationDates.length;
  // Batch ~40 dates per mutation so the fresh window's rows are a single
  // mutation instead of dozens of action→mutation round trips.
  const dateGroups: { date: string; rows: CalibrationRow[] }[][] = [];
  for (let i = 0; i < calibrationTotal; i += 40) {
    const group: { date: string; rows: CalibrationRow[] }[] = [];
    for (const date of calibrationDates.slice(i, i + 40)) {
      group.push({ date, rows: calibrationRowsByDate.get(date)! });
    }
    dateGroups.push(group);
  }
  let calibrationWritten = 0;
  await mapLimit(dateGroups, 6, async (group) => {
    await ctx.runMutation(internal.mlb.bulkReplaceCalibration, { groups: group });
    calibrationWritten += group.length;
    await report(
      "Updating calibration",
      calibrationTotal > 0 ? 95 + Math.floor((calibrationWritten / calibrationTotal) * 4) : 95,
      `Writing calibration rows — ${calibrationWritten.toLocaleString()}/${calibrationTotal.toLocaleString()} dates…`,
    );
  });
  // The full-history summary is maintained by the scheduled backfill action.
  const calibrationSummary = previous.calibrationSummary;

  // Compute refreshed powerRankings, preserving everything the model didn't change.
  const newPowerRankings: PowerRanking[] = (
    (previous.powerRankings ?? []) as PowerRanking[]
  ).map((r) => {
    const rec = teamState.records[r.teamId] ?? { wins: r.wins, losses: r.losses };
    return {
      ...r,
      elo: teamState.elo[r.teamId] ?? r.elo,
      wins: rec.wins,
      losses: rec.losses,
      winPct: rec.wins + rec.losses > 0 ? rec.wins / (rec.wins + rec.losses) : r.winPct,
      lastGameDate: teamState.lastGameDate[r.teamId] ?? r.lastGameDate,
      injuries: teamState.injuries[r.teamId] ?? r.injuries,
    };
  });

  await ctx.runMutation(internal.mlb.replaceModelState, {
    state: {
      ...previous,
      key: "current",
      trainedAt: Date.now(),
      asOfDate: today,
      powerRankings: newPowerRankings,
      pitcherLogs: pitcherLogsCache,
      teamLogs: teamLogsCache,
      batterLogs: batterLogsCache,
      bvpLogs: bvpLogsCache,
      platoonLogs: platoonLogsCache,
      vsTeamLogs: vsTeamLogsCache,
      calibrationSummary,
      todaysRecord,
    },
  });

  await report("Complete", 100, `Refreshed ${datesToWrite.length} day${datesToWrite.length === 1 ? "" : "s"} of predictions`, { done: true });
}

// ---------------------------------------------------------------------------
// Lineups (actual starting 9 + bench, from the per-game boxscore)
// ---------------------------------------------------------------------------

/**
 * Fetch a game's actual lineup from the lightweight boxscore endpoint.
 * Returns undefined when the boxscore has no lineup data yet (lineups are
 * posted ~2-3 hours before first pitch, and the endpoint 404s until then).
 */
async function fetchGameLineup(gamePk: number): Promise<LineupData | undefined> {
  try {
    // No retry loop here: a 404/empty boxscore is the *expected* state for
    // games whose lineups are not posted yet, and retrying would add seconds
    // per scheduled game.
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    const res = await fetch(`${MLB_BASE}/api/v1/game/${gamePk}/boxscore`, {
      signal: controller.signal,
      headers: { "User-Agent": "FreebuffMLB/1.0" },
    });
    clearTimeout(timer);
    if (!res.ok) return undefined;
    const data = await res.json();
    const teams = data?.teams;
    if (!teams?.home || !teams?.away) return undefined;
    const parseSide = (side: any) => {
      const battingOrder: LineupPlayer[] = [];
      const bench: LineupPlayer[] = [];
      const orderIdToSlot = new Map<number, number>();
      const orderIds = new Set<number>((side.battingOrder ?? []) as number[]);
      const players = (side.players ?? {}) as Record<string, any>;
      for (const key of Object.keys(players)) {
        const p = players[key];
        const id = p?.person?.id;
        if (typeof id !== "number") continue;
        const pos = p?.position?.abbreviation;
        if (orderIds.has(id)) {
          const slot = typeof p?.battingOrder === "string" ? parseInt(p.battingOrder, 10) : 0;
          if (slot > 0) orderIdToSlot.set(id, slot);
          battingOrder.push({ id, name: p.person.fullName ?? "", pos });
        } else if (pos !== "P") {
          bench.push({ id, name: p.person.fullName ?? "", pos });
        }
      }
      battingOrder.sort((a, b) => (orderIdToSlot.get(a.id) ?? 99) - (orderIdToSlot.get(b.id) ?? 99));
      return { battingOrder, bench };
    };
    return { home: parseSide(teams.home), away: parseSide(teams.away) };
  } catch {
    return undefined; // lineups not posted yet / no boxscore for scheduled games
  }
}

/** Fetch lineups for many games with a bounded concurrency. */
async function fetchLineupsForGames(
  games: RawGame[],
  concurrency = 16,
  onProgress?: (done: number, total: number) => void): Promise<Map<number, LineupData>> {
  const out = new Map<number, LineupData>();
  const seen = new Set<number>();
  const targets = games.filter((g) => {
    if (seen.has(g.gamePk)) return false;
    seen.add(g.gamePk);
    return true;
  });
  let done = 0;
  await mapLimit(targets, concurrency, async (g) => {
    const lu = await fetchGameLineup(g.gamePk);
    if (lu && (lu.home?.battingOrder.length ?? 0) > 0 && (lu.away?.battingOrder.length ?? 0) > 0) {
      out.set(g.gamePk, lu);
    }
    done += 1;
    if (onProgress && done % 10 === 0) onProgress(done, targets.length);
  });
  if (onProgress) onProgress(targets.length, targets.length);
  return out;
}

/** Weighted mean of a batter stat across a starting 9 — slots 1-4 count double. */
function lineupMean(lineup: LineupPlayer[] | undefined, key: "ops" | "woba" | "iso" | "recentOps"): number {
  if (!lineup || lineup.length === 0) return 0;
  let sum = 0;
  let w = 0;
  for (let i = 0; i < lineup.length; i++) {
    const v = lineup[i][key];
    if (typeof v !== "number") continue;
    const weight = i < 4 ? 2 : 1;
    sum += v * weight;
    w += weight;
  }
  return w > 0 ? sum / w : 0;
}

/**
 * Attach lineup data + as-of-date lineup-strength features (OPS / wOBA / ISO /
 * L10 hot streak) computed strictly from each batter's game log before the
 * game date — no lookahead. `batterLogs` is the `${id}|${season}` → game-log
 * cache stored on the model state.
 */
function attachLineupsAsOf(
  games: RawGame[],
  lineups: Map<number, LineupData>,
  batterLogs: Record<string, HittingLogEntry[]>,
): RawGame[] {
  return games.map((g) => {
    const lu = lineups.get(g.gamePk);
    if (!lu) return g;
    const season = g.season ?? "";
    const withStats = (side?: { battingOrder: LineupPlayer[]; bench: LineupPlayer[] }) => {
      if (!side) return side;
      const map = (p: LineupPlayer): LineupPlayer => {
        const log = batterLogs[`${p.id}|${season}`];
        return {
          ...p,
          ops: batterOpsAsOf(log, g.date),
          woba: batterWobaAsOf(log, g.date),
          iso: batterIsoAsOf(log, g.date),
          recentOps: batterRecentOpsAsOf(log, g.date),
        };
      };
      return {
        battingOrder: side.battingOrder.map(map),
        bench: side.bench.map(map),
      };
    };
    const home = withStats(lu.home);
    const away = withStats(lu.away);
    const homeOps = lineupMean(home?.battingOrder, "ops");
    const awayOps = lineupMean(away?.battingOrder, "ops");
    return {
      ...g,
      lineups: { home, away },
      lineupStats: {
        home: {
          known: homeOps > 0,
          ops: homeOps,
          woba: lineupMean(home?.battingOrder, "woba"),
          iso: lineupMean(home?.battingOrder, "iso"),
          recentOps: lineupMean(home?.battingOrder, "recentOps"),
        },
        away: {
          known: awayOps > 0,
          ops: awayOps,
          woba: lineupMean(away?.battingOrder, "woba"),
          iso: lineupMean(away?.battingOrder, "iso"),
          recentOps: lineupMean(away?.battingOrder, "recentOps"),
        },
      },
    };
  });
}

// ---------------------------------------------------------------------------
// Batter-vs-pitcher matchups + platoon splits (MLB Stats API vsPlayer /
// statSplits / vsTeam) — attached only where the real boxscore lineup AND the
// opposing starter are both known (fresh window + upcoming games), mirroring
// how the lineup-strength features are populated.
// ---------------------------------------------------------------------------

interface MatchupStat {
  pa: number; // plate appearances (sample size)
  ops: number;
}

/** Career BvP cache: `${batterId}|${pitcherId}` → career OPS/PA vs that pitcher. */
type BvpCache = Record<string, MatchupStat>;

/** Season platoon cache: `${batterId}|${season}` → OPS vs LHP / RHP. */
type PlatoonCache = Record<string, { vsLeft?: MatchupStat; vsRight?: MatchupStat }>;

/** Season vs-team cache: `${batterId}|${teamId}|${season}` → OPS vs that team. */
type VsTeamCache = Record<string, MatchupStat>;

/** OPS from a statsapi hitting `stat` block (falls back to OBP + SLG). */
function matchupOps(stat: any): number | undefined {
  const ops = logNum(stat?.ops);
  if (ops > 0) return round3(ops);
  const obp = logNum(stat?.obp);
  const slg = logNum(stat?.slg);
  if (obp + slg > 0) return round3(obp + slg);
  return undefined;
}

/** Plate appearances from a statsapi hitting `stat` block. */
function matchupPa(stat: any): number {
  const pa = logNum(stat?.plateAppearances);
  if (pa > 0) return pa;
  return logNum(stat?.atBats) + logNum(stat?.baseOnBalls) + logNum(stat?.hitByPitch) + logNum(stat?.sacFlies);
}

/** Career batter-vs-pitcher totals (`stats=vsPlayer`, no season → career). */
async function fetchBvpStats(
  pairs: { batterId: number; pitcherId: number }[],
  cached: BvpCache = {},
): Promise<BvpCache> {
  const out: BvpCache = {};
  const seen = new Set<string>();
  const unique = pairs.filter((p) => {
    const key = `${p.batterId}|${p.pitcherId}`;
    if (p.batterId <= 0 || p.pitcherId <= 0 || seen.has(key) || cached[key]) return false;
    seen.add(key);
    return true;
  });
  await mapLimit(unique, 20, async (p) => {
    try {
      const data = await fetchJson(
        `${MLB_BASE}/api/v1/people/${p.batterId}/stats?stats=vsPlayer&opposingPlayerId=${p.pitcherId}&group=hitting`,
      );
      const stat = data?.stats?.[0]?.splits?.[0]?.stat;
      const ops = matchupOps(stat);
      const pa = matchupPa(stat);
      if (typeof ops === "number" && pa > 0) {
        out[`${p.batterId}|${p.pitcherId}`] = { pa, ops };
      }
    } catch {
      // non-fatal
    }
  });
  return out;
}

/** Season platoon splits (`stats=statSplits&sitCodes=vl,vr`). */
async function fetchPlatoonSplits(
  pairs: { id: number; season: string }[],
  cached: PlatoonCache = {},
): Promise<PlatoonCache> {
  const out: PlatoonCache = {};
  const seen = new Set<string>();
  const unique = pairs.filter((p) => {
    const key = `${p.id}|${p.season}`;
    if (p.id <= 0 || seen.has(key) || cached[key]) return false;
    seen.add(key);
    return true;
  });
  await mapLimit(unique, 20, async (p) => {
    try {
      const data = await fetchJson(
        `${MLB_BASE}/api/v1/people/${p.id}/stats?stats=statSplits&sitCodes=vl,vr&group=hitting&season=${p.season}`,
      );
      const entry: PlatoonCache[string] = {};
      for (const s of data?.stats?.[0]?.splits ?? []) {
        const code = s?.split?.sitCode ?? "";
        const ops = matchupOps(s?.stat);
        const pa = matchupPa(s?.stat);
        if (typeof ops !== "number" || pa <= 0) continue;
        const stat: MatchupStat = { pa, ops };
        if (code === "vl") entry.vsLeft = stat;
        else if (code === "vr") entry.vsRight = stat;
      }
      if (entry.vsLeft || entry.vsRight) out[`${p.id}|${p.season}`] = entry;
    } catch {
      // non-fatal
    }
  });
  return out;
}

/** Season batter-vs-team totals (`stats=vsTeam`). */
async function fetchVsTeamStats(
  pairs: { batterId: number; teamId: number; season: string }[],
  cached: VsTeamCache = {},
): Promise<VsTeamCache> {
  const out: VsTeamCache = {};
  const seen = new Set<string>();
  const unique = pairs.filter((p) => {
    const key = `${p.batterId}|${p.teamId}|${p.season}`;
    if (p.batterId <= 0 || p.teamId <= 0 || seen.has(key) || cached[key]) return false;
    seen.add(key);
    return true;
  });
  await mapLimit(unique, 20, async (p) => {
    try {
      const data = await fetchJson(
        `${MLB_BASE}/api/v1/people/${p.batterId}/stats?stats=vsTeam&opposingTeamId=${p.teamId}&group=hitting&season=${p.season}`,
      );
      const stat = data?.stats?.[0]?.splits?.[0]?.stat;
      const ops = matchupOps(stat);
      const pa = matchupPa(stat);
      if (typeof ops === "number" && pa > 0) {
        out[`${p.batterId}|${p.teamId}|${p.season}`] = { pa, ops };
      }
    } catch {
      // non-fatal
    }
  });
  return out;
}

/** Throwing hand for starters the schedule hydrate didn't include. */
async function fetchPitcherHands(ids: number[]): Promise<Record<number, "L" | "R">> {
  const out: Record<number, "L" | "R"> = {};
  const unique = [...new Set(ids)].filter((id) => id > 0);
  await mapLimit(unique, 12, async (id) => {
    try {
      const data = await fetchJson(`${MLB_BASE}/api/v1/people/${id}`);
      const person = data?.people?.[0];
      const code = person?.pitchHand?.code ?? "";
      const desc = person?.pitchHand?.description ?? "";
      if (code === "L" || code === "R") out[id] = code;
      else if (/^L/i.test(desc)) out[id] = "L";
      else if (/^R/i.test(desc)) out[id] = "R";
    } catch {
      // non-fatal
    }
  });
  return out;
}

/**
 * Slot-weighted mean (slots 1-4 double) of a matchup stat across the starting
 * 9, with PA saturation so tiny BvP samples can't dominate. Returns 0 when no
 * batter in the order has data (the model reads 0 + lineupKnown = "no matchup
 * data").
 */
function matchupLineupMean(
  lineup: LineupPlayer[] | undefined,
  key: "bvpOPS" | "platoonOPS" | "vsTeamOPS",
  paKey?: "bvpPA",
): number {
  if (!lineup || lineup.length === 0) return 0;
  let sum = 0;
  let w = 0;
  for (let i = 0; i < lineup.length; i++) {
    const v = lineup[i][key];
    if (typeof v !== "number") continue;
    let weight = i < 4 ? 2 : 1;
    if (paKey) {
      const pa = lineup[i][paKey] ?? 0;
      weight *= Math.min(1, pa / 20);
    }
    sum += v * weight;
    w += weight;
  }
  return w > 0 ? round3(sum / w) : 0;
}

/** Total career PA in the BvP matchup across the starting 9 (display only). */
function lineupMatchupPa(lineup: LineupPlayer[] | undefined): number {
  if (!lineup) return 0;
  return lineup.reduce((s, p) => s + (p.bvpPA ?? 0), 0);
}

/**
 * Attach batter-vs-pitcher (career vsPlayer), platoon (vs the starter's
 * handedness) and batter-vs-team splits to games that have a real boxscore
 * lineup. BvP OPS is shrunk toward the batter's as-of season OPS (empirical-
 * Bayes style) so a handful of plate appearances can't dominate; the lineup
 * edge is the PA-saturated, slot-weighted mean over the starting 9.
 */
function attachMatchups(
  games: RawGame[],
  batterLogs: Record<string, HittingLogEntry[]>,
  bvpCache: BvpCache,
  platoonCache: PlatoonCache,
  vsTeamCache: VsTeamCache,
  pitcherHands: Record<number, "L" | "R">,
): RawGame[] {
  const SHRINK_PA = 15; // BvP PA count that pulls the blend halfway to season OPS
  return games.map((g) => {
    const lu = g.lineups;
    if (!lu) return g;
    const season = g.season ?? "";
    const ymd = g.date;
    const withMatchup =
      (starter: PitcherInfo | undefined, opponentTeamId: number) =>
      (side?: { battingOrder: LineupPlayer[]; bench: LineupPlayer[] }) => {
        if (!side) return side;
        const map = (p: LineupPlayer): LineupPlayer => {
          const seasonOps = batterOpsAsOf(batterLogs[`${p.id}|${season}`], ymd);
          const bvp = starter ? bvpCache[`${p.id}|${starter.id}`] : undefined;
          let bvpOPS: number | undefined;
          if (bvp && bvp.pa > 0) {
            bvpOPS =
              typeof seasonOps === "number"
                ? (bvp.ops * bvp.pa + seasonOps * SHRINK_PA) / (bvp.pa + SHRINK_PA)
                : bvp.ops;
            bvpOPS = round3(bvpOPS);
          }
          const hand = starter?.pitchHand ?? pitcherHands[starter?.id ?? -1];
          const platoon = platoonCache[`${p.id}|${season}`];
          const platoonOps =
            hand === "L" ? platoon?.vsLeft?.ops : hand === "R" ? platoon?.vsRight?.ops : undefined;
          const vsTeam = vsTeamCache[`${p.id}|${opponentTeamId}|${season}`];
          return {
            ...p,
            bvpOPS,
            bvpPA: bvp?.pa,
            platoonOPS: typeof platoonOps === "number" ? round3(platoonOps) : undefined,
            vsTeamOPS: typeof vsTeam?.ops === "number" ? round3(vsTeam.ops) : undefined,
          };
        };
        return {
          battingOrder: side.battingOrder.map(map),
          bench: side.bench.map(map),
        };
      };
    const home = withMatchup(g.awayPitcher, g.away.id)(lu.home);
    const away = withMatchup(g.homePitcher, g.home.id)(lu.away);
    if (!g.lineupStats) return { ...g, lineups: { home, away } };
    const extend = (
      side: {
        known: boolean;
        ops: number;
        woba: number;
        iso: number;
        recentOps: number;
        bvpOps?: number;
        bvpPA?: number;
        platoonOps?: number;
        vsTeamOps?: number;
      },
      batters: LineupPlayer[] | undefined,
    ) => ({
      ...side,
      bvpOps: matchupLineupMean(batters, "bvpOPS", "bvpPA"),
      bvpPA: lineupMatchupPa(batters),
      platoonOps: matchupLineupMean(batters, "platoonOPS"),
      vsTeamOps: matchupLineupMean(batters, "vsTeamOPS"),
    });
    return {
      ...g,
      lineups: { home, away },
      lineupStats: {
        home: extend(g.lineupStats.home, home?.battingOrder),
        away: extend(g.lineupStats.away, away?.battingOrder),
      },
    };
  });
}

/**
 * Gather the matchup pairs implied by the window's real lineups, fetch the
 * missing statsapi data (career BvP always refetched; current-season splits
 * refetched, past-season totals reused from the cache) and attach everything.
 */
async function enrichWithMatchups(
  games: RawGame[],
  batterLogs: Record<string, HittingLogEntry[]>,
  bvpCache: BvpCache,
  platoonCache: PlatoonCache,
  vsTeamCache: VsTeamCache,
  season: string,
): Promise<{
  games: RawGame[];
  bvpLogs: BvpCache;
  platoonLogs: PlatoonCache;
  vsTeamLogs: VsTeamCache;
}> {
  const bvpPairs: { batterId: number; pitcherId: number }[] = [];
  const platoonPairs: { id: number; season: string }[] = [];
  const vsTeamPairs: { batterId: number; teamId: number; season: string }[] = [];
  const pitchersNeedingHand: number[] = [];
  const seenBvp = new Set<string>();
  const seenPlatoon = new Set<string>();
  const seenVsTeam = new Set<string>();
  const seenHand = new Set<number>();
  for (const g of games) {
    const lu = g.lineups;
    if (!lu) continue;
    const s = g.season ?? season;
    const matchups: [LineupPlayer[] | undefined, PitcherInfo | undefined, number][] = [
      [lu.home?.battingOrder, g.awayPitcher, g.away.id],
      [lu.away?.battingOrder, g.homePitcher, g.home.id],
    ];
    for (const [batters, starter, oppTeamId] of matchups) {
      if (!batters) continue;
      for (const p of batters) {
        if (starter?.id) {
          const bKey = `${p.id}|${starter.id}`;
          if (!seenBvp.has(bKey)) {
            seenBvp.add(bKey);
            bvpPairs.push({ batterId: p.id, pitcherId: starter.id });
          }
          if (!starter.pitchHand && !seenHand.has(starter.id)) {
            seenHand.add(starter.id);
            pitchersNeedingHand.push(starter.id);
          }
        }
        const pKey = `${p.id}|${s}`;
        if (!seenPlatoon.has(pKey)) {
          seenPlatoon.add(pKey);
          platoonPairs.push({ id: p.id, season: s });
        }
        const vKey = `${p.id}|${oppTeamId}|${s}`;
        if (!seenVsTeam.has(vKey)) {
          seenVsTeam.add(vKey);
          vsTeamPairs.push({ batterId: p.id, teamId: oppTeamId, season: s });
        }
      }
    }
  }

  // Career BvP is always refetched for the window so repeat matchups stay
  // fresh; season splits refetch for the current season (they change daily)
  // and reuse cached totals for past seasons (they are complete).
  const newBvp = await fetchBvpStats(bvpPairs, {});
  const bvpLogs: BvpCache = { ...bvpCache, ...newBvp };
  const curPlatoon = platoonPairs.filter((p) => p.season === season);
  const pastPlatoon = platoonPairs.filter((p) => p.season !== season);
  const newPlatoon = await fetchPlatoonSplits(curPlatoon, {});
  const reusedPlatoon = await fetchPlatoonSplits(pastPlatoon, platoonCache);
  const platoonLogs: PlatoonCache = { ...platoonCache, ...reusedPlatoon, ...newPlatoon };
  const curVsTeam = vsTeamPairs.filter((p) => p.season === season);
  const pastVsTeam = vsTeamPairs.filter((p) => p.season !== season);
  const newVsTeam = await fetchVsTeamStats(curVsTeam, {});
  const reusedVsTeam = await fetchVsTeamStats(pastVsTeam, vsTeamCache);
  const vsTeamLogs: VsTeamCache = { ...vsTeamCache, ...reusedVsTeam, ...newVsTeam };
  const pitcherHands = await fetchPitcherHands(pitchersNeedingHand);

  return {
    games: attachMatchups(games, batterLogs, bvpLogs, platoonLogs, vsTeamLogs, pitcherHands),
    bvpLogs,
    platoonLogs,
    vsTeamLogs,
  };
}

// ---------------------------------------------------------------------------
// Model state reconstruction (for on-demand date predictions)
// ---------------------------------------------------------------------------

function reconstructModel(state: any): TrainedModel {
  return {
    featureNames: state.featureNames as TrainedModel["featureNames"],
    weights: state.weights as number[],
    bias: state.bias as number,
    featureStats: state.featureStats as TrainedModel["featureStats"],
    isotonicPoints: state.isotonicPoints as TrainedModel["isotonicPoints"],
    monteCarloSigma: state.monteCarloSigma as number,
    monteCarloEnabled: state.monteCarloEnabled as boolean,
    eloHfa: state.eloHfa as number,
  };
}

function reconstructTeamState(rankings: PowerRanking[]): TeamState {
  const elo: Record<number, number> = {};
  const form: Record<number, number> = {};
  const lastGameDate: Record<number, string> = {};
  const records: Record<number, { wins: number; losses: number }> = {};
  const injuries: Record<number, number> = {};
  for (const p of rankings) {
    elo[p.teamId] = p.elo;
    form[p.teamId] = p.last10WinPct;
    lastGameDate[p.teamId] = p.lastGameDate;
    records[p.teamId] = { wins: p.wins, losses: p.losses };
    injuries[p.teamId] = p.injuries;
  }
  return { elo, form, lastGameDate, records, injuries };
}

// ---------------------------------------------------------------------------
// Game document building
// ---------------------------------------------------------------------------

/** Season totals (legacy display fallback; as-of stats on the game win). */
interface PitcherSeasonStats {
  era?: number;
  k9?: number;
  fip?: number;
}

function buildGameDoc(
  game: RawGame,
  pred: { homeWinProb: number; awayWinProb: number; pickTeam: "home" | "away"; pickProb: number; shap: GameDoc["shap"]; edge: number; fairHomeOdds: number; fairAwayOdds: number },
  pitcherStats: Map<string, PitcherSeasonStats>,
  injuries?: { home: number; away: number },
  runProjection?: RunProjection,
  marketOdds?: MarketOdds): GameDoc {
  // As-of-date stats attached to the game (attachAsOfStats) win over the
  // season-total map: the map only fills gaps so displayed ERA / K9 / FIP never
  // leak post-game data into a game's own prediction.
  const withPitcherStats = (p?: PitcherInfo): PitcherInfo | undefined => {
    if (!p) return p;
    const st = pitcherStats.get(`${p.id}|${game.season}`);
    return {
      ...p,
      era: typeof p.era === "number" ? p.era : st?.era,
      k9: typeof p.k9 === "number" ? p.k9 : st?.k9,
      fip: typeof p.fip === "number" ? p.fip : st?.fip,
    };
  };
  const awayPitcher = withPitcherStats(game.awayPitcher);
  const homePitcher = withPitcherStats(game.homePitcher);

  const doc: GameDoc = {
    gamePk: game.gamePk,
    date: game.date,
    status: game.status,
    detailedState: game.detailedState,
    dayNight: game.dayNight,
    gameDate: game.gameDate,
    innings: game.innings,
    venue: game.venue,
    away: game.away,
    home: game.home,
    awayPitcher,
    homePitcher,
    winner: game.winner,
    homeWinProb: pred.homeWinProb,
    awayWinProb: pred.awayWinProb,
    pickTeam: pred.pickTeam,
    pickProb: pred.pickProb,
    edge: pred.edge,
    fairAwayOdds: pred.fairAwayOdds,
    fairHomeOdds: pred.fairHomeOdds,
    shap: pred.shap,
    homeInjuries: injuries?.home,
    awayInjuries: injuries?.away,
    season: game.season,
    weather: game.weather,
    lineups: game.lineups,
    lineupStats: game.lineupStats,
    runProjection,
    marketOdds,
  };
  if (game.winner === "home" || game.winner === "away") {
    doc.isCorrect = pred.pickTeam === game.winner;
    doc.isUpset = pred.pickTeam !== game.winner;
  }
  return doc;
}

function predictionFor(game: RawGame, model: TrainedModel, teamState: TeamState) {
  return applyModel(
    model,
    buildFeaturesForGame(game, teamState),
    teamState.elo[game.home.id] ?? 1500,
    teamState.elo[game.away.id] ?? 1500,
  );
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/** Convert a stored game document back into a raw game for retraining. */
function gameDocToRaw(doc: any): RawGame {
  return {
    gamePk: doc.gamePk,
    date: doc.date,
    gameDate: doc.gameDate,
    dayNight: doc.dayNight,
    status: doc.status,
    detailedState: doc.detailedState,
    away: doc.away,
    home: doc.home,
    awayPitcher: doc.awayPitcher,
    homePitcher: doc.homePitcher,
    venue: doc.venue,
    innings: doc.innings,
    winner: doc.winner,
    season: doc.season,
    weather: doc.weather,
    lineups: doc.lineups,
    lineupStats: doc.lineupStats,
  };
}

/** Prefer stored pitcher stats when a fresh API game re-fetches the same starter. */
function mergePitcher(fresh?: PitcherInfo, stored?: PitcherInfo): PitcherInfo | undefined {
  if (!fresh) return stored;
  if (!stored) return fresh;
  if (fresh.id !== stored.id) return fresh; // probable starter changed
  return {
    ...fresh,
    era: typeof fresh.era === "number" ? fresh.era : stored.era,
    k9: typeof fresh.k9 === "number" ? fresh.k9 : stored.k9,
    fip: typeof fresh.fip === "number" ? fresh.fip : stored.fip,
    whip: typeof fresh.whip === "number" ? fresh.whip : stored.whip,
    recentEra: typeof fresh.recentEra === "number" ? fresh.recentEra : stored.recentEra,
  };
}

/** Merge a freshly fetched game over its stored copy without discarding cached stats. */
function mergeRawWithStored(fresh: RawGame, stored: RawGame): RawGame {
  return {
    ...fresh,
    awayPitcher: mergePitcher(fresh.awayPitcher, stored.awayPitcher),
    homePitcher: mergePitcher(fresh.homePitcher, stored.homePitcher),
  };
}

/**
 * Load stored games for a date range in bounded pages (fast, no external
 * API). Page size is 300 so full game docs never blow Convex's per-query
 * response limit; loading only 2024→today skips irrelevant older history.
 */
async function loadStoredGames(ctx: any, startDate: string, endDate: string): Promise<RawGame[]> {
  const all: RawGame[] = [];
  let cursor: string | null = null;
  do {
    const page: { games: any[]; cursor: string | null } = await ctx.runQuery(
      internal.mlb.getGamesByDateRange,
      { startDate, endDate, cursor, limit: 300 },
    );
    for (const g of page.games) all.push(gameDocToRaw(g));
    cursor = page.cursor;
  } while (cursor);
  return all;
}

/** Map a stored game document to the compact calibration projection row. */
function calibrationRowFromDoc(d: GameDoc): CalibrationRow {
  return {
    gamePk: d.gamePk,
    date: d.date,
    away: { abbrev: d.away.abbrev, name: d.away.name, score: d.away.score },
    home: { abbrev: d.home.abbrev, name: d.home.name, score: d.home.score },
    winner: d.winner,
    pickTeam: d.pickTeam,
    pickProb: d.pickProb,
    homeWinProb: d.homeWinProb,
    isCorrect: d.isCorrect,
    isUpset: d.isUpset,
    predictedTotal: d.runProjection?.total,
    homeRunLineProb: d.runProjection?.homeRunLineProb,
    actualTotal: (d.away.score ?? 0) + (d.home.score ?? 0),
    actualMargin: (d.home.score ?? 0) - (d.away.score ?? 0),
  };
}

/** Precomputed full-range calibration metrics stored on the model state. */
function buildCalibrationSummary(rows: CalibrationRow[]): CalibrationSummary {
  const preds = rows.map((d) => d.pickProb);
  const labels = rows.map((d) => (d.isCorrect ? 1 : 0));
  const evalResult = evaluate(preds, labels);
  const curve = calibrationCurvePoints(preds, labels, 8);
  const metrics = {
    auc: evalResult.auc,
    brier: evalResult.brier,
    logLoss: evalResult.logLoss,
    ece: evalResult.ece,
    bins: evalResult.bins,
    confidenceDistribution: evalResult.confidenceDistribution,
    calibrationCurve: curve.length > 0 ? curve : evalResult.calibrationCurve,
  };

  let tN = 0;
  let tAbs = 0;
  let tSq = 0;
  let tBias = 0;
  const rlPreds: number[] = [];
  const rlLabels: number[] = [];
  for (const d of rows) {
    if (typeof d.predictedTotal === "number") {
      tN += 1;
      const err = d.predictedTotal - d.actualTotal;
      tAbs += Math.abs(err);
      tSq += err * err;
      tBias += err;
    }
    if (typeof d.homeRunLineProb === "number") {
      rlPreds.push(d.homeRunLineProb);
      rlLabels.push(d.actualMargin >= 2 ? 1 : 0);
    }
  }

  const totalsMetrics = {
    n: tN,
    mae: tN > 0 ? tAbs / tN : 0,
    rmse: tN > 0 ? Math.sqrt(tSq / tN) : 0,
    bias: tN > 0 ? tBias / tN : 0,
  };
  let runLineMetrics = { n: rlPreds.length, auc: 0, brier: 0, accuracy: 0 };
  if (rlPreds.length > 0) {
    runLineMetrics = {
      n: rlPreds.length,
      auc: computeAuc(rlPreds, rlLabels),
      brier: computeBrier(rlPreds, rlLabels),
      accuracy:
        rlPreds.filter((p, i) => (p >= 0.5 ? 1 : 0) === rlLabels[i]).length / rlPreds.length,
    };
  }

  const total = rows.length;
  const correct = rows.filter((d) => d.isCorrect).length;
  return {
    metrics,
    totalsMetrics,
    runLineMetrics,
    total,
    correct,
    accuracy: total > 0 ? correct / total : 0,
  };
}

/** Compact calibration row (≈1/10th of a game doc) for calibration queries. */
interface CalibrationRow {
  gamePk: number;
  date: string;
  away: { abbrev: string; name: string; score?: number };
  home: { abbrev: string; name: string; score?: number };
  winner?: "home" | "away";
  pickTeam: "home" | "away";
  pickProb: number;
  homeWinProb: number;
  isCorrect?: boolean;
  isUpset?: boolean;
  predictedTotal?: number;
  homeRunLineProb?: number;
  actualTotal: number;
  actualMargin: number;
}

/** Build compact calibration rows from the training rows (no full GameDocs). */
function buildCalibrationRows(
  rows: FeatureRow[],
  model: TrainedModel,
  runModelState: RunModel,
  runLineIso: { x: number; y: number }[],
  runMarginCal: RunMarginCalibration): CalibrationRow[] {
  return rows.map((row: FeatureRow) => {
    const pred = applyModel(model, row.features, row.homeElo, row.awayElo);
    const g = row.game;
    const proj = buildRunProjection(
      runModelState,
      runLineIso,
      g,
      undefined,
      undefined,
      RUN_CALIB_TRIALS,
      pred.homeWinProb,
      runMarginCal,
    );
    return {
      gamePk: g.gamePk,
      date: g.date,
      away: { abbrev: g.away.abbrev, name: g.away.name, score: g.away.score },
      home: { abbrev: g.home.abbrev, name: g.home.name, score: g.home.score },
      winner: g.winner,
      pickTeam: pred.pickTeam,
      pickProb: pred.pickProb,
      homeWinProb: pred.homeWinProb,
      isCorrect: pred.pickTeam === g.winner,
      isUpset: pred.pickTeam !== g.winner,
      predictedTotal: proj.total,
      homeRunLineProb: proj.homeRunLineProb,
      actualTotal: (g.away.score ?? 0) + (g.home.score ?? 0),
      actualMargin: (g.home.score ?? 0) - (g.away.score ?? 0),
    };
  });
}

export const refreshModel = action({
  args: {},
  handler: async (ctx) => {
    // Progress reporting is best-effort: it must never abort the refresh
    // itself (e.g. on an OCC conflict from a second writer on the shared
    // refreshProgress doc). Swallow write failures and keep going.
    const report = progressReporter(ctx);

    const now = new Date();
    const season = String(now.getFullYear());
    const today = etDateString(now);

    try {
      const previousState: any = await ctx.runQuery(internal.mlb.getLatestModelState, {});

      // 0a. ATOMIC CLAIM: only one refresh may run at a time. The claim is a
      //     single transaction (claimRefresh), so two concurrent refreshModel
      //     actions (double-click, multiple tabs, reconnect) cannot both win —
      //     the loser returns `alreadyRunning` instead of racing the winner on
      //     the shared progress doc (which caused OptimisticConcurrencyControl
      //     failures that killed the action). The claim also writes the first
      //     progress report atomically.
      const claim = await ctx.runMutation(internal.mlb.claimRefresh, {});
      if (!claim.claimed) {
        return {
          fast: true,
          alreadyRunning: true,
          season,
          asOfDate: today,
          gamesTrained: previousState?.gamesTrained ?? 0,
          holdoutCount: previousState?.holdoutCount ?? 0,
          auc: previousState?.auc ?? 0,
          brier: previousState?.brier ?? 0,
          logLoss: previousState?.logLoss ?? 0,
          ece: previousState?.ece ?? 0,
          selectedModel: previousState?.selectedModel ?? "",
          monteCarloEnabled: previousState?.monteCarloEnabled ?? false,
          storedGames: 0,
        };
      }

      // 0. FAST PATH: a trained model already exists, so skip the full reload
      //    + retrain (the multi-minute path) and just refresh the recent
      //    results + upcoming predictions from the live API. The model weights
      //    stay frozen as of the last retrain.
      if (previousState) {
        await fastRefresh(ctx, previousState, season, today, report);
        // One-time calibration-history backfill runs as its own SCHEDULED
        // action after the refresh returns. Inline, it scans the whole games
        // table and pushed this action past Convex's 10-minute Node action
        // timeout — the reason refreshes kept dying with the bar frozen at
        // "Backfilling calibration 94%". The scheduled action gets its own
        // execution budget and reports its own progress.
        // Schedule (or resume) the one-time calibration backfill unless its
        // chain has completed OR is already actively running (a fresh,
        // not-done state means the self-scheduling chain is alive and will
        // finish on its own — scheduling again would start a second chain).
        const backfillState: any = await ctx.runQuery(internal.mlb.getCalibrationBackfill, {});
        const backfillChainAlive =
          backfillState &&
          !backfillState.done &&
          Date.now() - backfillState.updatedAt < 3 * 60_000;
        if (!backfillState?.done && !backfillChainAlive) {
          await ctx.scheduler.runAfter(0, internal.backfill.backfillCalibration, {});
        }
        return {
          fast: true,
          season,
          asOfDate: today,
          gamesTrained: previousState.gamesTrained ?? 0,
          holdoutCount: previousState.holdoutCount ?? 0,
          auc: previousState.auc ?? 0,
          brier: previousState.brier ?? 0,
          logLoss: previousState.logLoss ?? 0,
          ece: previousState.ece ?? 0,
          selectedModel: previousState.selectedModel ?? "",
          monteCarloEnabled: previousState.monteCarloEnabled ?? false,
          storedGames: 0,
        };
      }

      // 1. Reuse previously stored games so complete seasons are not re-fetched
      //    from the external API on every refresh (this was the ~5-minute cost).
      //    Only 2024 → today is loaded: older seasons add little to the fit and
      //    trimming them keeps the read well inside Convex's per-action budget.
      const storedGames = await loadStoredGames(ctx, `${Number(season) - 2}-03-15`, today);
      const storedCompleted = storedGames.filter(
        (g) => g.winner === "home" || g.winner === "away");

      // 2. Fetch only the recent window + upcoming games. On a cold start (empty
      //    database) fall back to the full 2024–2026 fetch exactly once.
      const hasFullHistory = storedCompleted.length >= 2000;
      const coldStartSeasons = ["2024", "2025"];
      const coldStartList = [...coldStartSeasons, season];
      await report(
        "Fetching schedule",
        14,
        hasFullHistory
          ? "Fetching recent results & upcoming games…"
          : `First run: fetching ${coldStartList.join("–")} game history…`);

      const stageTicker = (stageLabel: string, total: number, startPct: number, endPct: number) =>
        (completed: number) => {
          const c = Math.min(completed, total);
          const frac = total > 0 ? c / total : 1;
          ctx.runMutation(internal.mlb.setRefreshProgress, {
            stage: `${stageLabel} ${c}/${total}`,
            pct: startPct + Math.floor(frac * (endPct - startPct)),
            message: `Loading ${stageLabel.toLowerCase()} (${c}/${total})…`,
          }).catch(() => {});
        };

      let freshRaw: RawGame[];
      if (hasFullHistory) {
        freshRaw = await fetchScheduleRange(
          addDays(today, -RECENT_WINDOW_DAYS),
          addDays(today, UPCOMING_WINDOW_DAYS),
        );
      } else {
        await report(
          `Fetching seasons 1/${coldStartList.length}`,
          18,
          `Loading first season of ${coldStartList.length}…`,
        );
        const seasonTicker = stageTicker("Fetching seasons", coldStartList.length, 18, 28);
        freshRaw = await fetchAllSeasons(coldStartList, season, today, seasonTicker);
      }

      // 3. Merge fresh API data over stored data (fresh wins by gamePk).
      const byPk = new Map<number, RawGame>();
      for (const g of storedGames) byPk.set(g.gamePk, g);
      for (const g of freshRaw) {
        const stored = byPk.get(g.gamePk);
        byPk.set(g.gamePk, stored ? mergeRawWithStored(g, stored) : g);
      }
      const allRaw = [...byPk.values()];

      const completed = allRaw.filter((g) => g.winner === "home" || g.winner === "away");
      if (completed.length < 40) {
        throw new Error(
          `Only ${completed.length} completed regular-season games found. Cannot train yet.`,
        );
      }

      // 4. As-of-date game logs for pitchers + teams across the training window
      //    (accumulated strictly before each game's date — no lookahead). Logs
      //    are cached on the model state, so refreshes only fetch missing
      //    seasons and any as-of date is a cheap local sum.
      await report("Fetching game logs", 28, "Loading as-of pitcher / team game logs…");
      const previousPitcherLogs = (previousState?.pitcherLogs ?? {}) as Record<string, PitcherLogEntry[]>;
      const previousTeamLogs = (previousState?.teamLogs ?? {}) as Record<string, TeamGameLog>;
      const previousBatterLogs = (previousState?.batterLogs ?? {}) as Record<string, HittingLogEntry[]>;
      const pitcherPairs: { id: number; season: string }[] = [];
      const seenPitcher = new Set<string>();
      for (const g of allRaw) {
        const s = g.season ?? season;
        if (g.awayPitcher) {
          const k = `${g.awayPitcher.id}|${s}`;
          if (!seenPitcher.has(k)) {
            seenPitcher.add(k);
            pitcherPairs.push({ id: g.awayPitcher.id, season: s });
          }
        }
        if (g.homePitcher) {
          const k = `${g.homePitcher.id}|${s}`;
          if (!seenPitcher.has(k)) {
            seenPitcher.add(k);
            pitcherPairs.push({ id: g.homePitcher.id, season: s });
          }
        }
      }
      const newPitcherLogs = await fetchPitcherGameLogs(pitcherPairs, previousPitcherLogs);
      const pitcherLogs: Record<string, PitcherLogEntry[]> = { ...previousPitcherLogs, ...newPitcherLogs };
      await report(
        "Pitcher logs loaded",
        34,
        `Fetched ${Object.keys(newPitcherLogs).length} pitcher game logs.`);

      // 5. Team game logs (hitting / pitching / fielding) per season. Current
      //    season always refreshes (new games accumulate); past seasons reuse
      //    the cached logs on the model state.
      await report("Fetching team logs", 40, "Loading as-of team game logs…");
      const teamSeasonPairs = new Map<string, { id: number; season: string }>();
      for (const g of allRaw) {
        const s = g.season ?? season;
        const hk = `${g.home.id}|${s}`;
        const ak = `${g.away.id}|${s}`;
        if (!teamSeasonPairs.has(hk)) teamSeasonPairs.set(hk, { id: g.home.id, season: s });
        if (!teamSeasonPairs.has(ak)) teamSeasonPairs.set(ak, { id: g.away.id, season: s });
      }
      const teamStatsToFetch = [...teamSeasonPairs.values()].filter((p) => {
        const key = `${p.id}|${p.season}`;
        return p.season === season || previousTeamLogs[key] === undefined;
      });
      const newTeamLogs = await fetchTeamGameLogs(teamStatsToFetch, previousTeamLogs);
      const teamLogs: Record<string, TeamGameLog> = { ...previousTeamLogs, ...newTeamLogs };
      await report(
        "Team logs loaded",
        44,
        `Fetched ${Object.keys(newTeamLogs).length} team-season game logs.`);
      const enriched = attachAsOfStats(allRaw, pitcherLogs, teamLogs);

      // 6. Actual starting lineups (last 2 days + upcoming window): fetch the
      //    boxscore for each game, attach the starting 9 + bench, and pull each
      //    batter's as-of game log so the model gets real lineup-strength
      //    features (weighted OPS / wOBA / ISO / L10 hot streak — no lookahead).
      await report("Fetching lineups", 46, "Loading actual starting lineups…");
      const lineupGames = enriched.filter(
        (g) => g.date >= addDays(today, -2) && g.date <= addDays(today, UPCOMING_WINDOW_DAYS));
      const lineupTicker = stageTicker("Lineups", lineupGames.length, 46, 52);
      const lineups = await fetchLineupsForGames(lineupGames, 16, lineupTicker);
      const batterIds: number[] = [];
      for (const lu of lineups.values()) {
        for (const side of [lu.home, lu.away]) {
          if (!side) continue;
          for (const p of [...side.battingOrder, ...side.bench]) batterIds.push(p.id);
        }
      }
      const newBatterLogs = await fetchBatterGameLogs(batterIds, season, previousBatterLogs);
      const batterLogs: Record<string, HittingLogEntry[]> = { ...previousBatterLogs, ...newBatterLogs };
      const lineupsEnriched = attachLineupsAsOf(enriched, lineups, batterLogs);

      // 6b. Batter-vs-pitcher matchups + platoon / vs-team splits for the fresh
      //     window (real lineups + opposing starters only, like the lineup
      //     features). Career BvP is always refetched; current-season splits
      //     refetch, past-season totals reuse the cached values.
      await report("Fetching matchups", 52, "Loading BvP & platoon splits…");
      const matchup = await enrichWithMatchups(
        lineupsEnriched,
        batterLogs,
        (previousState?.bvpLogs ?? {}) as BvpCache,
        (previousState?.platoonLogs ?? {}) as PlatoonCache,
        (previousState?.vsTeamLogs ?? {}) as VsTeamCache,
        season,
      );
      const enrichedWithLineups = matchup.games;
      const bvpLogs = matchup.bvpLogs;
      const platoonLogs = matchup.platoonLogs;
      const vsTeamLogs = matchup.vsTeamLogs;

      // 7. Pull as-of-time injured-list snapshots for every team that has played,
      //    then train, select features/model, calibrate, and decide on Monte Carlo.
      //    On a cold start we skip this: the injury feature benefits from a multi-
      //    month history, and saving the bulk fetch here is what was leaving the
      //    action hanging past 5 minutes before the user reconnected.
      await report("Fetching injury data", 54, hasFullHistory ? "Loading injured-list snapshots…" : "Skipping injury history on first run");
      const teamIds = [...new Set(completed.flatMap((g) => [g.home.id, g.away.id]))];
      const previousInjurySnapshots =
        previousState?.season === season
          ? ((previousState.injurySnapshots ?? {}) as Record<string, InjurySnapshot[]>)
          : {};
      const injurySnapshots = hasFullHistory
        ? await fetchInjurySnapshots(
            teamIds,
            season,
            `${season}-${SEASON_START_MD}`,
            today,
            previousInjurySnapshots,
          )
        : new Map<number, InjurySnapshot[]>();

      const completedEnriched = enrichedWithLineups.filter(
        (g) => g.winner === "home" || g.winner === "away",
      );

      await report("Training model", 64, "Fitting models & selecting features…");
      const run = runModel(completedEnriched, { season, asOfDate: today }, injurySnapshots);
      const { result, model, rows, teamState } = run;
      const runModelState = result.runModel;
      const runLineIso = result.runLineCalibration ?? [];
      const runMarginCal = result.runMarginCalibration ?? { slope: 0, intercept: 0 };

      await report("Fetching market odds", 72, "Loading market odds (best-effort)…");
      // Market odds (best-effort; empty map when THE_ODDS_API_KEY is unset).
      const marketOdds = await fetchMarketOdds();

      await report("Scoring games", 80, "Generating predictions & run simulations…");
      // 7. As-of-time predictions for every completed game (historical results),
      //    plus run-scoring projections for the totals / run-line views. Only
      //    compact calibration rows are built here — the full GameDoc (with
      //    SHAP + market odds) is only materialized for the fresh window below,
      //    which keeps the refresh from constructing ~7k heavyweight docs.
      const calibrationRows = buildCalibrationRows(
        rows,
        model,
        runModelState,
        runLineIso,
        runMarginCal,
      );

      // Descriptive reliability / calibration views over the full dataset.
      // Favorite framing: one side per game (predicted probability > 50%) vs. outcome.
      const fullPreds = calibrationRows.map((d) => d.pickProb);
      const fullLabels = calibrationRows.map((d) => (d.isCorrect ? 1 : 0));
      const fullCurve = calibrationCurvePoints(fullPreds, fullLabels, 12);
      const fullEval = evaluate(fullPreds, fullLabels);
      const spearmanRho = spearmanRank(fullPreds, fullLabels);
      const highConf = calibrationRows.filter((d) => d.pickProb >= 0.65);
      const topDecileWinRate = highConf.length > 0 ? highConf.filter((d) => d.isCorrect).length / highConf.length : 0;
      const calibrationSummary = buildCalibrationSummary(calibrationRows);

      // 8. Build fresh docs only for the recent/upcoming window so we never
      //    rewrite thousands of unchanged historical dates on refresh.
      const freshDates = new Set(freshRaw.map((g) => g.date));
      const rowsByPk = new Map(rows.map((r) => [r.game.gamePk, r]));
      const freshDocs: GameDoc[] = [];
      const emptyPitcherStats = new Map<string, PitcherSeasonStats>();
      for (const g of enrichedWithLineups) {
        if (!freshDates.has(g.date)) continue;
        if (g.winner === "home" || g.winner === "away") {
          const row = rowsByPk.get(g.gamePk);
          if (!row) continue;
          const pred = applyModel(model, row.features, row.homeElo, row.awayElo);
          freshDocs.push(
            buildGameDoc(
              g,
              pred,
              emptyPitcherStats,
              undefined,
              buildRunProjection(runModelState, runLineIso, g, undefined, undefined, RUN_CALIB_TRIALS, pred.homeWinProb, runMarginCal),
            ),
          );
        } else {
          const odds = marketOddsForGame(marketOdds, g);
          const pred = predictionFor(g, model, teamState);
          freshDocs.push(
            buildGameDoc(
              g,
              pred,
              emptyPitcherStats,
              { home: teamState.injuries[g.home.id] ?? 0, away: teamState.injuries[g.away.id] ?? 0 },
              buildRunProjection(runModelState, runLineIso, g, odds?.total, odds?.runLine, RUN_SIM_TRIALS, pred.homeWinProb, runMarginCal),
              odds,
            ),
          );
        }
      }

      // 9. Today's record.
      const todaysRecord = buildTodaysRecord(calibrationRows as unknown as GameDoc[], today);

      await report("Saving model state", 92, "Persisting trained model…");
      // 10. Store the model state (singleton).
      await ctx.runMutation(internal.mlb.replaceModelState, {
        state: {
          key: "current",
          trainedAt: Date.now(),
          season: result.season,
          asOfDate: result.asOfDate,
          gamesTrained: result.gamesTrained,
          holdoutCount: result.holdoutCount,
          selectedModel: result.selectedModel,
          modelDescription: result.modelDescription,
          featureNames: result.featureNames,
          weights: result.weights,
          bias: result.bias,
          featureStats: result.featureStats,
          isotonicPoints: result.isotonicPoints,
          eloHfa: result.eloHfa,
          monteCarloEnabled: result.monteCarloEnabled,
          monteCarloTrials: result.monteCarloTrials,
          monteCarloSigma: result.monteCarloSigma,
          monteCarloRationale: result.monteCarloRationale,
          auc: fullEval.auc,
          brier: fullEval.brier,
          logLoss: fullEval.logLoss,
          ece: fullEval.ece,
          bins: fullEval.bins,
          confidenceDistribution: fullEval.confidenceDistribution,
          calibrationCurve: fullCurve.length > 0 ? fullCurve : result.calibrationCurve,
          featureImportances: result.featureImportances,
          candidates: result.candidates,
          powerRankings: result.powerRankings,
          featureDrift: result.featureDrift,
          rollingBrier: result.rollingBrier,
          brierBaseline: result.brierBaseline,
          modelVersions: result.modelVersions,
          stackingWeights: result.stackingWeights,
          crossValidation: result.crossValidation,
          optimizationParams: result.optimizationParams,
          runModel: result.runModel,
          runLineCalibration: result.runLineCalibration,
          runMarginCalibration: runMarginCal,
          pitcherLogs,
          teamLogs,
          batterLogs,
          bvpLogs,
          platoonLogs,
          vsTeamLogs,
          injurySnapshots: Object.fromEntries(injurySnapshots),
          calibrationSummary,
          spearmanRho,
          topDecileWinRate,
          todaysRecord,
        },
      });

      await report("Saving games", 97, "Persisting game predictions…");
      // 11. Store fresh game docs grouped by date (recent window + upcoming only).
      const byDate = new Map<string, GameDoc[]>();
      for (const doc of freshDocs) {
        const list = byDate.get(doc.date) ?? [];
        list.push(doc);
        byDate.set(doc.date, list);
      }
      await mapLimit([...byDate.keys()], 8, (date) =>
        ctx.runMutation(internal.mlb.replaceGamesForDate, { date, games: byDate.get(date)! }),
      );

      // 12. Store the lightweight calibration projection. On the first run after
      //    this feature ships (or a cold start) backfill every completed date;
      //    afterward only the fresh window is rewritten.
      const needsCalibrationBackfill = !previousState?.calibrationSummary;
      const calibrationDates = needsCalibrationBackfill
        ? new Set(calibrationRows.map((r) => r.date))
        : freshDates;
      const calibrationByDate = new Map<string, CalibrationRow[]>();
      for (const row of calibrationRows) {
        if (!calibrationDates.has(row.date)) continue;
        const list = calibrationByDate.get(row.date) ?? [];
        list.push(row);
        calibrationByDate.set(row.date, list);
      }
      await mapLimit([...calibrationByDate.keys()], 8, (date) =>
        ctx.runMutation(internal.mlb.replaceCalibrationForDate, { date, rows: calibrationByDate.get(date)! }),
      );

      await report("Complete", 100, "Model refreshed", { done: true });

      return {
        season,
        asOfDate: today,
        gamesTrained: result.gamesTrained,
        holdoutCount: result.holdoutCount,
        auc: result.auc,
        brier: result.brier,
        logLoss: result.logLoss,
        ece: result.ece,
        selectedModel: result.selectedModel,
        monteCarloEnabled: result.monteCarloEnabled,
        storedGames: freshDocs.length,
      };
    } catch (e) {
      const message = e instanceof Error ? e.message : "Unknown error";
      await report("Refresh failed", 100, message, { done: true, error: message }).catch(() => {});
      throw e;
    }
  },
});

export const predictDate = action({
  args: { date: v.string() },
  handler: async (ctx, args) => {
    const state = await ctx.runQuery(internal.mlb.getLatestModelState, {});
    if (!state) {
      throw new Error("Model has not been trained yet. Click refresh first.");
    }
    const model = reconstructModel(state);
    const teamState = reconstructTeamState(state.powerRankings as PowerRanking[]);
    const runModelState = state.runModel as RunModel | undefined;
    const runLineIso = (state.runLineCalibration ?? []) as { x: number; y: number }[];
    const runMarginCal = (state.runMarginCalibration ?? { slope: 0, intercept: 0 }) as RunMarginCalibration;
    const season = state.season as string;

    const raw = await fetchScheduleRange(args.date, args.date);
    const pitcherPairs: { id: number; season: string }[] = [];
    for (const g of raw) {
      const s = g.season ?? season;
      if (g.awayPitcher) pitcherPairs.push({ id: g.awayPitcher.id, season: s });
      if (g.homePitcher) pitcherPairs.push({ id: g.homePitcher.id, season: s });
    }

    // As-of game logs are cached on the model state; only missing seasons are
    // fetched, and stats accumulate strictly before the selected game date.
    const storedPitcherLogs = (state.pitcherLogs ?? {}) as Record<string, PitcherLogEntry[]>;
    const storedTeamLogs = (state.teamLogs ?? {}) as Record<string, TeamGameLog>;
    const storedBatterLogs = (state.batterLogs ?? {}) as Record<string, HittingLogEntry[]>;
    const newPitcherLogs = await fetchPitcherGameLogs(pitcherPairs, storedPitcherLogs);
    const pitcherLogs: Record<string, PitcherLogEntry[]> = { ...storedPitcherLogs, ...newPitcherLogs };

    const teamPairs = new Map<string, { id: number; season: string }>();
    for (const g of raw) {
      const s = g.season ?? season;
      for (const id of [g.home.id, g.away.id]) {
        const key = `${id}|${s}`;
        if (!teamPairs.has(key)) teamPairs.set(key, { id, season: s });
      }
    }
    const newTeamLogs = await fetchTeamGameLogs([...teamPairs.values()], storedTeamLogs);
    const teamLogs: Record<string, TeamGameLog> = { ...storedTeamLogs, ...newTeamLogs };
    const enriched = attachAsOfStats(raw, pitcherLogs, teamLogs);

    // Actual starting lineups for the selected date (boxscore), with each
    // batter's as-of stats (OPS / wOBA / ISO / L10) pulled from the cached
    // game-log map + fresh fetches.
    const lineups = await fetchLineupsForGames(enriched, 16);
    const batterIds: number[] = [];
    for (const lu of lineups.values()) {
      for (const side of [lu.home, lu.away]) {
        if (!side) continue;
        for (const p of [...side.battingOrder, ...side.bench]) batterIds.push(p.id);
      }
    }
    const newBatterLogs = await fetchBatterGameLogs(batterIds, season, storedBatterLogs);
    const batterLogs: Record<string, HittingLogEntry[]> = { ...storedBatterLogs, ...newBatterLogs };
    const lineupsEnriched = attachLineupsAsOf(enriched, lineups, batterLogs);

    // BvP / platoon / vs-team splits for the selected date's real lineups.
    const matchup = await enrichWithMatchups(
      lineupsEnriched,
      batterLogs,
      (state.bvpLogs ?? {}) as BvpCache,
      (state.platoonLogs ?? {}) as PlatoonCache,
      (state.vsTeamLogs ?? {}) as VsTeamCache,
      season,
    );
    const enrichedWithLineups = matchup.games;

    const marketOdds = await fetchMarketOdds();
    const emptyPitcherStats = new Map<string, PitcherSeasonStats>();

    const docs = enrichedWithLineups.map((g) => {
      const odds = marketOddsForGame(marketOdds, g);
      const pred = predictionFor(g, model, teamState);
      return buildGameDoc(
        g,
        pred,
        emptyPitcherStats,
        { home: teamState.injuries[g.home.id] ?? 0, away: teamState.injuries[g.away.id] ?? 0 },
        runModelState ? buildRunProjection(runModelState, runLineIso, g, odds?.total, odds?.runLine, RUN_SIM_TRIALS, pred.homeWinProb, runMarginCal) : undefined,
        odds,
      );
    });
    await ctx.runMutation(internal.mlb.replaceGamesForDate, { date: args.date, games: docs });
    return { games: docs.length };
  },
});

function buildTodaysRecord(completedDocs: GameDoc[], today: string): TodaysRecord {
  const todays = completedDocs.filter((d) => d.date === today);
  const total = todays.length;
  const correct = todays.filter((d) => d.isCorrect).length;
  const upsets = todays
    .filter((d) => d.isUpset)
    .map((d) => ({
      team: d.winner === "home" ? d.home.abbrev : d.away.abbrev,
      loser: d.winner === "home" ? d.away.abbrev : d.home.abbrev,
      prob: Math.round((d.winner === "home" ? d.homeWinProb : d.awayWinProb) * 100),
    }));
  return {
    date: today,
    total,
    completed: total,
    wins: correct,
    losses: total - correct,
    correct,
    accuracy: total > 0 ? correct / total : 0,
    upsets,
  };
}
