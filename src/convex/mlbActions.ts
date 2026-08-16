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

/** True if `g.date` falls within the last N days of `today`. */
function isWithinRecentWind(g: RawGame, today: string, days: number): boolean {
  const start = addDays(today, -days);
  return g.date >= start && g.date <= today;
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

// ---------------------------------------------------------------------------
// Schedule parsing
// ---------------------------------------------------------------------------

function scheduleUrl(start: string, end: string): string {
  return `${MLB_BASE}/api/v1/schedule?sportId=1&startDate=${start}&endDate=${end}&hydrate=probablePitcher,linescore,weather`;
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
    awayPitcher: away.probablePitcher
      ? { id: away.probablePitcher.id, name: away.probablePitcher.fullName }
      : undefined,
    homePitcher: home.probablePitcher
      ? { id: home.probablePitcher.id, name: home.probablePitcher.fullName }
      : undefined,
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

// ---------------------------------------------------------------------------
// Pitcher stats (season ERA / K/9) — display + matchup context
// ---------------------------------------------------------------------------

interface PitcherSeasonStats {
  era?: number;
  k9?: number;
  fip?: number;
}

function statNumber(value: unknown): number | undefined {
  if (typeof value === "string") {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : undefined;
  }
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
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

async function fetchPitcherStats(
  pairs: { id: number; season: string }[]): Promise<Map<string, PitcherSeasonStats>> {
  const out = new Map<string, PitcherSeasonStats>();
  const seen = new Set<string>();
  const unique = pairs.filter((p) => {
    const key = `${p.id}|${p.season}`;
    if (p.id <= 0 || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  await mapLimit(unique, 16, async (p) => {
    try {
      const data = await fetchJson(
        `${MLB_BASE}/api/v1/people/${p.id}/stats?stats=season&group=pitching&season=${p.season}`);
      const stat = data?.stats?.[0]?.splits?.[0]?.stat;
      if (stat) {
        const era = statNumber(stat.era);
        const k9 = statNumber(stat.strikeoutsPer9Inn);
        const hr = statNumber(stat.homeRuns) ?? 0;
        const bb = statNumber(stat.baseOnBalls) ?? 0;
        const hbp = statNumber(stat.hitByPitch) ?? 0;
        const so = statNumber(stat.strikeOuts) ?? 0;
        const ip = inningsPitchedValue(stat.inningsPitched);
        const fip = ip > 0 ? (13 * hr + 3 * (bb + hbp) - 2 * so) / ip + FIP_CONSTANT : undefined;
        out.set(`${p.id}|${p.season}`, {
          era,
          k9,
          fip: fip === undefined ? undefined : Math.round(fip * 100) / 100,
        });
      }
    } catch {
      // ignore individual pitcher stat failures
    }
  });
  return out;
}

/** Attach ERA/K9/FIP to each game's probable pitchers in place (new array). */
function attachPitcherStats(
  games: RawGame[],
  stats: Map<string, PitcherSeasonStats>): RawGame[] {
  return games.map((g) => ({
    ...g,
    awayPitcher: g.awayPitcher
      ? { ...g.awayPitcher, ...(stats.get(`${g.awayPitcher.id}|${g.season}`) ?? {}) }
      : undefined,
    homePitcher: g.homePitcher
      ? { ...g.homePitcher, ...(stats.get(`${g.homePitcher.id}|${g.season}`) ?? {}) }
      : undefined,
  }));
}

// ---------------------------------------------------------------------------
// Team season stats (OPS / ERA / fielding%) — statsapi-only features
// ---------------------------------------------------------------------------

interface TeamSeasonStats {
  ops?: number;
  era?: number;
  fieldingPct?: number;
}

async function fetchTeamSeasonStats(
  pairs: { id: number; season: string }[]): Promise<Map<string, TeamSeasonStats>> {
  const out = new Map<string, TeamSeasonStats>();
  const seen = new Set<string>();
  const unique = pairs.filter((p) => {
    const key = `${p.id}|${p.season}`;
    if (p.id <= 0 || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  await mapLimit(unique, 16, async (p) => {
    try {
      const data = await fetchJson(
        `${MLB_BASE}/api/v1/teams/${p.id}/stats?stats=season&group=hitting,pitching,fielding&season=${p.season}`);
      const stats = (data?.stats ?? []) as any[];
      const result: TeamSeasonStats = {};
      for (const block of stats) {
        const group = block?.group?.displayName;
        const stat = block?.splits?.[0]?.stat;
        if (!stat) continue;
        if (group === "hitting") result.ops = statNumber(stat.ops);
        else if (group === "pitching") result.era = statNumber(stat.era);
        else if (group === "fielding") result.fieldingPct = statNumber(stat.fielding);
      }
      if (Object.keys(result).length > 0) out.set(`${p.id}|${p.season}`, result);
    } catch {
      // ignore individual team stat failures
    }
  });
  return out;
}

/** Attach season team OPS / ERA / fielding% to each game's two teams. */
function attachTeamSeasonStats(
  games: RawGame[],
  stats: Record<string, TeamSeasonStats>): RawGame[] {
  return games.map((g) => {
    const homeStats = stats[`${g.home.id}|${g.season}`];
    const awayStats = stats[`${g.away.id}|${g.season}`];
    return {
      ...g,
      home: { ...g.home, ...(homeStats ?? {}) },
      away: { ...g.away, ...(awayStats ?? {}) },
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
 * minutes. The model weights stay frozen as of the last retrain.
 */
async function fastRefresh(
  ctx: any,
  previous: any,
  season: string,
  today: string,
  report: (stage: string, pct: number, message: string, extra?: { done?: boolean; error?: string }) => Promise<void>): Promise<void> {
  await report("Reading previous model", 6, "Reusing the trained model…");
  const model = reconstructModel(previous);
  const runModelState = previous.runModel as RunModel | undefined;
  const runLineIso = (previous.runLineCalibration ?? []) as { x: number; y: number }[];
  const runMarginCal = (previous.runMarginCalibration ?? { slope: 0, intercept: 0 }) as RunMarginCalibration;
  const previousTeamStats = (previous.teamSeasonStats ?? {}) as Record<string, TeamSeasonStats>;
  const previousPlayerOps = (previous.playerOps ?? {}) as Record<string, number>;

  await report("Fetching fresh games", 18, "Loading the last week + upcoming schedule…");
  const freshRaw = await fetchScheduleRange(
    addDays(today, -RECENT_WINDOW_DAYS),
    addDays(today, UPCOMING_WINDOW_DAYS),
  );

  if (freshRaw.length === 0) {
    await report("No new games", 100, "Up to date; nothing to refresh.", { done: true });
    return;
  }

  await report("Pitcher stats", 28, "Loading ERA / K9 for fresh window starters…");
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
  const freshPitcherStats = await fetchPitcherStats(freshPitcherPairs);

  await report("Team stats", 40, "Refreshing current-season team batting / pitching stats…");
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
    return previousTeamStats[key] === undefined;
  });
  const freshTeamStats = await fetchTeamSeasonStats(teamStatsToFetch);
  const teamSeasonStats: Record<string, TeamSeasonStats> = {
    ...previousTeamStats,
    ...Object.fromEntries(freshTeamStats),
  };

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
  const playerOps = await fetchPlayerSeasonOps(batterIds, season, previousPlayerOps);
  const playerOpsCache: Record<string, number> = {
    ...previousPlayerOps,
    ...Object.fromEntries([...playerOps].map(([id, ops]) => [`${id}|${season}`, ops])),
  };

  await report("Loading injuries", 68, "Loading current IL snapshot…");
  const teamIds = [...new Set(freshRaw.flatMap((g) => [g.home.id, g.away.id]))];
  const currentInjury = await fetchCurrentInjurySnapshot(teamIds, today, season);
  for (const [id, count] of currentInjury) teamState.injuries[id] = count;

  await report("Predicting upcoming", 78, "Scoring the upcoming schedule…");
  const enriched = attachTeamSeasonStats(attachPitcherStats(freshRaw, freshPitcherStats), teamSeasonStats);
  const enrichedWithLineups = attachLineups(enriched, lineups, playerOps);
  const marketOdds = await fetchMarketOdds();
  const freshByPk = new Map<number, RawGame>(enrichedWithLineups.map((g) => [g.gamePk, g]));

  // Upcoming games (no result yet) get a fresh pre-game prediction built from
  // the current team state — which already includes every result applied above.
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
        freshPitcherStats,
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
  // refresh. DBs that predate the calibration feature have no rows and no
  // summary, so the first refresh after this ships does a one-time full
  // backfill from stored game docs (bounded pages) — after that, only the
  // fresh window is rewritten and reads short-circuit on the summary.
  const needsCalibrationBackfill =
    !previous.calibrationSummary || (previous.calibrationSummary?.total ?? 0) === 0;
  await report(
    needsCalibrationBackfill ? "Backfilling calibration" : "Updating calibration",
    94,
    needsCalibrationBackfill
      ? "Building calibration history from stored games…"
      : "Writing calibration rows for the fresh window…",
  );
  if (needsCalibrationBackfill) {
    const storedDocs = await loadStoredDocs(ctx, "2022-03-15", today);
    const backfillByDate = new Map<string, CalibrationRow[]>();
    for (const d of storedDocs) {
      if (d.winner !== "home" && d.winner !== "away") continue;
      if (typeof d.pickProb !== "number") continue;
      const list = backfillByDate.get(d.date) ?? [];
      list.push(calibrationRowFromDoc(d));
      backfillByDate.set(d.date, list);
    }
    for (const [date, rows] of backfillByDate) calibrationRowsByDate.set(date, rows);
  }
  const calibrationDates = [...calibrationRowsByDate.keys()];
  await mapLimit(calibrationDates, 8, (date) =>
    ctx.runMutation(internal.mlb.replaceCalibrationForDate, {
      date,
      rows: calibrationRowsByDate.get(date)!,
    }),
  );
  const calibrationRows = [...calibrationRowsByDate.values()].flat();
  const calibrationSummary =
    calibrationRows.length > 0 ? buildCalibrationSummary(calibrationRows) : previous.calibrationSummary;

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
      teamSeasonStats,
      playerOps: playerOpsCache,
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

/** Season hitting OPS for a set of players, reusing the cached map. */
async function fetchPlayerSeasonOps(
  ids: number[],
  season: string,
  cached: Record<string, number> = {}): Promise<Map<number, number>> {
  const out = new Map<number, number>();
  const unique = [...new Set(ids)].filter((id) => id > 0 && typeof cached[`${id}|${season}`] !== "number");
  await mapLimit(unique, 24, async (id) => {
    try {
      const data = await fetchJson(
        `${MLB_BASE}/api/v1/people/${id}/stats?stats=season&group=hitting&season=${season}`);
      const stat = data?.stats?.[0]?.splits?.[0]?.stat;
      const ops = statNumber(stat?.ops);
      if (ops !== undefined) out.set(id, ops);
    } catch {
      // individual player stat failures are non-fatal
    }
  });
  return out;
}

/** Weighted mean OPS of a lineup — slots 1-4 (most PAs) count double. */
function lineupOps(lineup: LineupPlayer[] | undefined): number {
  if (!lineup || lineup.length === 0) return 0;
  let sum = 0;
  let w = 0;
  for (let i = 0; i < lineup.length; i++) {
    const ops = lineup[i].ops;
    if (typeof ops !== "number") continue;
    const weight = i < 4 ? 2 : 1;
    sum += ops * weight;
    w += weight;
  }
  return w > 0 ? sum / w : 0;
}

/**
 * Attach lineup data + computed lineup-strength features to games that have a
 * fetched boxscore lineup. `playerOps` is a game-season map of player OPS.
 */
function attachLineups(
  games: RawGame[],
  lineups: Map<number, LineupData>,
  playerOps: Map<number, number>): RawGame[] {
  return games.map((g) => {
    const lu = lineups.get(g.gamePk);
    if (!lu) return g;
    const withOps = (side?: { battingOrder: LineupPlayer[]; bench: LineupPlayer[] }) => {
      if (!side) return side;
      return {
        battingOrder: side.battingOrder.map((p) => ({ ...p, ops: playerOps.get(p.id) })),
        bench: side.bench.map((p) => ({ ...p, ops: playerOps.get(p.id) })),
      };
    };
    const home = withOps(lu.home);
    const away = withOps(lu.away);
    const homeOps = lineupOps(home?.battingOrder);
    const awayOps = lineupOps(away?.battingOrder);
    return {
      ...g,
      lineups: { home, away },
      lineupStats: {
        home: { known: homeOps > 0, ops: homeOps },
        away: { known: awayOps > 0, ops: awayOps },
      },
    };
  });
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

function buildGameDoc(
  game: RawGame,
  pred: { homeWinProb: number; awayWinProb: number; pickTeam: "home" | "away"; pickProb: number; shap: GameDoc["shap"]; edge: number; fairHomeOdds: number; fairAwayOdds: number },
  pitcherStats: Map<string, PitcherSeasonStats>,
  injuries?: { home: number; away: number },
  runProjection?: RunProjection,
  marketOdds?: MarketOdds): GameDoc {
  const awayPitcher: PitcherInfo | undefined = game.awayPitcher
    ? { ...game.awayPitcher, ...(pitcherStats.get(`${game.awayPitcher.id}|${game.season}`) ?? {}) }
    : undefined;
  const homePitcher: PitcherInfo | undefined = game.homePitcher
    ? { ...game.homePitcher, ...(pitcherStats.get(`${game.homePitcher.id}|${game.season}`) ?? {}) }
    : undefined;

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

/**
 * Load stored game DOCUMENTS for a date range in bounded pages, keeping the
 * prediction fields (pickProb / isCorrect / runProjection) intact. Used for
 * the one-time calibration backfill.
 */
async function loadStoredDocs(ctx: any, startDate: string, endDate: string): Promise<GameDoc[]> {
  const all: GameDoc[] = [];
  let cursor: string | null = null;
  do {
    const page: { games: GameDoc[]; cursor: string | null } = await ctx.runQuery(
      internal.mlb.getGamesByDateRange,
      { startDate, endDate, cursor, limit: 300 },
    );
    all.push(...page.games);
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
    const report = async (
      stage: string,
      pct: number,
      message: string,
      extra: { done?: boolean; error?: string } = {},
    ) => {
      await ctx.runMutation(internal.mlb.setRefreshProgress, {
        stage,
        pct,
        message,
        ...extra,
      });
    };

    const now = new Date();
    const season = String(now.getFullYear());
    const today = etDateString(now);

    try {
      await report("Loading stored games", 4, "Reading previously stored games…");
      const previousState: any = await ctx.runQuery(internal.mlb.getLatestModelState, {});

      // 0. FAST PATH: a trained model already exists, so skip the full reload
      //    + retrain (the multi-minute path) and just refresh the recent
      //    results + upcoming predictions from the live API. The model weights
      //    stay frozen as of the last retrain.
      if (previousState) {
        await fastRefresh(ctx, previousState, season, today, report);
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
          void ctx.runMutation(internal.mlb.setRefreshProgress, {
            stage: `${stageLabel} ${c}/${total}`,
            pct: startPct + Math.floor(frac * (endPct - startPct)),
            message: `Loading ${stageLabel.toLowerCase()} (${c}/${total})…`,
          });
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

      // 4. Fetch pitcher stats only for games whose starters still lack them
      //    (stored games already carry their ERA / K9 / FIP). On a cold start we
      //    pull stats for the current-season starters only — older-season pitcher
      //    deltas are folded into the team pitching/ERA feature instead.
      await report("Fetching pitcher stats", 28, "Loading starting-pitcher ERA / FIP…");
      const statsSeasons = new Set([season, String(Number(season) - 1)]);
      const needsStats = allRaw
        .filter(
          (g) =>
            statsSeasons.has(g.season ?? season) &&
            ((g.awayPitcher && typeof g.awayPitcher.era !== "number") ||
              (g.homePitcher && typeof g.homePitcher.era !== "number")),
        )
        .filter((g) => hasFullHistory || g.season === season || isWithinRecentWind(g, today, 30));
      const pitcherPairs: { id: number; season: string }[] = [];
      const seenPitcher = new Set<string>();
      for (const g of needsStats) {
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
      const pitcherStats = await fetchPitcherStats(pitcherPairs);
      await report(
        "Pitcher stats loaded",
        34,
        `Loaded ${pitcherStats.size} pitcher seasons.`);

      // 5. Fetch season team OPS / ERA / fielding% (statsapi-only features).
      //    Past seasons are reused from the previously stored model state, so a
      //    normal refresh only refreshes the current season (~30 requests). On
      //    a cold start we still need current-season + the previous year only;
      //    the deeper history is folded into the team pitching/ERA feature by
      //    the run model.
      await report("Fetching team stats", 40, "Loading team OPS / ERA / fielding…");
      const previousTeamStats = (previousState?.teamSeasonStats ?? {}) as Record<string, TeamSeasonStats>;
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
        if (hasFullHistory) return p.season === season || previousTeamStats[key] === undefined;
        // Cold start — only current + previous season; historic team deltas aren't
        // useful features this late into the model anyway.
        return key.endsWith(`|${season}`) || key.endsWith(`|${String(Number(season) - 1)}`);
      });
      const freshTeamStats = await fetchTeamSeasonStats(teamStatsToFetch);
      const teamSeasonStats: Record<string, TeamSeasonStats> = {
        ...previousTeamStats,
        ...Object.fromEntries(freshTeamStats),
      };
      await report(
        "Team stats loaded",
        44,
        `Loaded ${freshTeamStats.size} team-season stat blocks.`);
      const enriched = attachTeamSeasonStats(attachPitcherStats(allRaw, pitcherStats), teamSeasonStats);

      // 6. Actual starting lineups (last 2 days + upcoming window): fetch the
      //    boxscore for each game, attach the starting 9 + bench, and pull each
      //    batter's season OPS so the model gets a real lineup-strength feature.
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
      const previousPlayerOps = (previousState?.playerOps ?? {}) as Record<string, number>;
      const playerOps = await fetchPlayerSeasonOps(batterIds, season, previousPlayerOps);
      const playerOpsCache: Record<string, number> = {
        ...previousPlayerOps,
        ...Object.fromEntries([...playerOps].map(([id, ops]) => [`${id}|${season}`, ops])),
      };
      const enrichedWithLineups = attachLineups(enriched, lineups, playerOps);

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
              pitcherStats,
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
              pitcherStats,
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
          teamSeasonStats,
          injurySnapshots: Object.fromEntries(injurySnapshots),
          playerOps: playerOpsCache,
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
    const pitcherStats = await fetchPitcherStats(pitcherPairs);

    // Team season stats are cached on the model state; refresh only any current
    // season team that is still missing before predicting a date.
    const storedTeamStats = (state.teamSeasonStats ?? {}) as Record<string, TeamSeasonStats>;
    const teamPairs = new Map<string, { id: number; season: string }>();
    for (const g of raw) {
      const s = g.season ?? season;
      for (const id of [g.home.id, g.away.id]) {
        const key = `${id}|${s}`;
        if (!teamPairs.has(key)) teamPairs.set(key, { id, season: s });
      }
    }
    const missingTeamStats = [...teamPairs.values()].filter(
      (p) => storedTeamStats[`${p.id}|${p.season}`] === undefined,
    );
    const freshTeamStats = await fetchTeamSeasonStats(missingTeamStats);
    const teamSeasonStats: Record<string, TeamSeasonStats> = {
      ...storedTeamStats,
      ...Object.fromEntries(freshTeamStats),
    };
    const enriched = attachTeamSeasonStats(attachPitcherStats(raw, pitcherStats), teamSeasonStats);

    // Actual starting lineups for the selected date (boxscore), with each
    // batter's season OPS pulled from the cached map + fresh fetches.
    const lineups = await fetchLineupsForGames(enriched, 16);
    const batterIds: number[] = [];
    for (const lu of lineups.values()) {
      for (const side of [lu.home, lu.away]) {
        if (!side) continue;
        for (const p of [...side.battingOrder, ...side.bench]) batterIds.push(p.id);
      }
    }
    const previousPlayerOps = (state.playerOps ?? {}) as Record<string, number>;
    const playerOps = await fetchPlayerSeasonOps(batterIds, season, previousPlayerOps);
    const enrichedWithLineups = attachLineups(enriched, lineups, playerOps);

    const marketOdds = await fetchMarketOdds();

    const docs = enrichedWithLineups.map((g) => {
      const odds = marketOddsForGame(marketOdds, g);
      const pred = predictionFor(g, model, teamState);
      return buildGameDoc(
        g,
        pred,
        pitcherStats,
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
