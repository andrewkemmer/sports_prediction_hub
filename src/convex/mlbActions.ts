"use node";

import { v } from "convex/values";
import { internal } from "./_generated/api";
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

/** Fetch the full training window: 2022–2025 full seasons + 2026 through today. */
async function fetchAllSeasons(currentSeason: string, throughDate: string): Promise<RawGame[]> {
  const seasons = [...TRAIN_SEASONS, currentSeason];
  const all: RawGame[] = [];
  for (const s of seasons) {
    const end = s === currentSeason ? throughDate : `${s}-${PAST_SEASON_END_MD}`;
    const games = await fetchSeason(s, end);
    all.push(...games);
  }
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
  pairs: { id: number; season: string }[],
): Promise<Map<string, PitcherSeasonStats>> {
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
        `${MLB_BASE}/api/v1/people/${p.id}/stats?stats=season&group=pitching&season=${p.season}`,
      );
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
  stats: Map<string, PitcherSeasonStats>,
): RawGame[] {
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
  pairs: { id: number; season: string }[],
): Promise<Map<string, TeamSeasonStats>> {
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
        `${MLB_BASE}/api/v1/teams/${p.id}/stats?stats=season&group=hitting,pitching,fielding&season=${p.season}`,
      );
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
  stats: Record<string, TeamSeasonStats>,
): RawGame[] {
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
  homeWinProb: number,
): number {
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
  runMarginCal: RunMarginCalibration = { slope: 0, intercept: 0 },
): RunProjection {
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
  previous?: Record<string, InjurySnapshot[]>,
): Promise<Map<number, InjurySnapshot[]>> {
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
  marketOdds?: MarketOdds,
): GameDoc {
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

/** Load every stored game in bounded pages (fast, no external API). */
async function loadStoredGames(ctx: any): Promise<RawGame[]> {
  const all: RawGame[] = [];
  let cursor: string | null = null;
  do {
    const page: { games: any[]; cursor: string | null } = await ctx.runQuery(
      internal.mlb.getGamesPage,
      { cursor, limit: 1000 },
    );
    for (const g of page.games) all.push(gameDocToRaw(g));
    cursor = page.cursor;
  } while (cursor);
  return all;
}

/** Precomputed full-range calibration metrics stored on the model state. */
function buildCalibrationSummary(completedDocs: GameDoc[]): CalibrationSummary {
  const preds = completedDocs.map((d) => d.pickProb);
  const labels = completedDocs.map((d) => (d.isCorrect ? 1 : 0));
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
  for (const d of completedDocs) {
    const predictedTotal = d.runProjection?.total;
    if (typeof predictedTotal === "number") {
      const actual = (d.away.score ?? 0) + (d.home.score ?? 0);
      tN += 1;
      const err = predictedTotal - actual;
      tAbs += Math.abs(err);
      tSq += err * err;
      tBias += err;
    }
    const homeRunLineProb = d.runProjection?.homeRunLineProb;
    if (typeof homeRunLineProb === "number") {
      const margin = (d.home.score ?? 0) - (d.away.score ?? 0);
      rlPreds.push(homeRunLineProb);
      rlLabels.push(margin >= 2 ? 1 : 0);
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

  const total = completedDocs.length;
  const correct = completedDocs.filter((d) => d.isCorrect).length;
  return {
    metrics,
    totalsMetrics,
    runLineMetrics,
    total,
    correct,
    accuracy: total > 0 ? correct / total : 0,
  };
}

/** Lightweight projection (1/10th of a game doc) for calibration queries. */
function calibrationRowFromDoc(d: GameDoc) {
  return {
    gamePk: d.gamePk,
    date: d.date,
    away: { abbrev: d.away.abbrev, name: d.away.name, score: d.away.score },
    home: { abbrev: d.home.abbrev, name: d.home.name, score: d.home.score },
    winner: d.winner,
    pickTeam: d.pickTeam,
    pickProb: d.pickProb,
    isCorrect: d.isCorrect,
    isUpset: d.isUpset,
    predictedTotal: d.runProjection?.total,
    homeRunLineProb: d.runProjection?.homeRunLineProb,
    actualTotal: (d.away.score ?? 0) + (d.home.score ?? 0),
    actualMargin: (d.home.score ?? 0) - (d.away.score ?? 0),
  };
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
      const previousState = await ctx.runQuery(internal.mlb.getLatestModelState, {});

      // 1. Reuse previously stored games so complete seasons are not re-fetched
      //    from the external API on every refresh (this was the ~5-minute cost).
      const storedGames = await loadStoredGames(ctx);
      const storedCompleted = storedGames.filter(
        (g) => g.winner === "home" || g.winner === "away",
      );

      // 2. Fetch only the recent window + upcoming games. On a cold start (empty
      //    database) fall back to the full 2022–2026 fetch exactly once.
      const hasFullHistory = storedCompleted.length >= 2000;
      await report(
        "Fetching schedule",
        14,
        hasFullHistory
          ? "Fetching recent results & upcoming games…"
          : "First run: fetching 2022–2026 game history…",
      );
      const freshRaw = hasFullHistory
        ? await fetchScheduleRange(addDays(today, -RECENT_WINDOW_DAYS), addDays(today, UPCOMING_WINDOW_DAYS))
        : await fetchAllSeasons(season, today);

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
      //    limit this to the current + previous season so the first refresh stays
      //    fast; older seasons rely on the team pitching/ERA features instead.
      await report("Fetching pitcher stats", 28, "Loading starting-pitcher ERA / FIP…");
      const statsSeasons = new Set([season, String(Number(season) - 1)]);
      const needsStats = allRaw.filter(
        (g) =>
          statsSeasons.has(g.season ?? season) &&
          ((g.awayPitcher && typeof g.awayPitcher.era !== "number") ||
            (g.homePitcher && typeof g.homePitcher.era !== "number")),
      );
      const pitcherPairs: { id: number; season: string }[] = [];
      for (const g of needsStats) {
        const s = g.season ?? season;
        if (g.awayPitcher) pitcherPairs.push({ id: g.awayPitcher.id, season: s });
        if (g.homePitcher) pitcherPairs.push({ id: g.homePitcher.id, season: s });
      }
      const pitcherStats = await fetchPitcherStats(pitcherPairs);

      // 5. Fetch season team OPS / ERA / fielding% (statsapi-only features).
      //    Past seasons are reused from the previously stored model state, so a
      //    normal refresh only refreshes the current season (~30 requests).
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
        return p.season === season || previousTeamStats[key] === undefined;
      });
      const freshTeamStats = await fetchTeamSeasonStats(teamStatsToFetch);
      const teamSeasonStats: Record<string, TeamSeasonStats> = {
        ...previousTeamStats,
        ...Object.fromEntries(freshTeamStats),
      };
      const enriched = attachTeamSeasonStats(attachPitcherStats(allRaw, pitcherStats), teamSeasonStats);
      const completedEnriched = enriched.filter(
        (g) => g.winner === "home" || g.winner === "away",
      );

      // 6. Pull as-of-time injured-list snapshots for every team that has played,
      //    then train, select features/model, calibrate, and decide on Monte Carlo.
      await report("Fetching injury data", 50, "Loading injured-list snapshots…");
      const teamIds = [...new Set(completed.flatMap((g) => [g.home.id, g.away.id]))];
      const previousInjurySnapshots =
        previousState?.season === season
          ? ((previousState.injurySnapshots ?? {}) as Record<string, InjurySnapshot[]>)
          : {};
      const injurySnapshots = await fetchInjurySnapshots(
        teamIds,
        season,
        `${season}-${SEASON_START_MD}`,
        today,
        previousInjurySnapshots,
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
      //    plus run-scoring projections for the totals / run-line views.
      const completedDocs: GameDoc[] = rows.map((row: FeatureRow) => {
        const pred = applyModel(model, row.features, row.homeElo, row.awayElo);
        return buildGameDoc(
          row.game,
          pred,
          pitcherStats,
          undefined,
          buildRunProjection(runModelState, runLineIso, row.game, undefined, undefined, RUN_CALIB_TRIALS, pred.homeWinProb, runMarginCal),
        );
      });

      // Descriptive reliability / calibration views over the full 2022–2026 dataset.
      // Favorite framing: one side per game (predicted probability > 50%) vs. outcome.
      const fullPreds = completedDocs.map((d) => d.pickProb);
      const fullLabels = completedDocs.map((d) => (d.isCorrect ? 1 : 0));
      const fullCurve = calibrationCurvePoints(fullPreds, fullLabels, 12);
      const fullEval = evaluate(fullPreds, fullLabels);
      const spearmanRho = spearmanRank(fullPreds, fullLabels);
      const highConf = completedDocs.filter((d) => d.pickProb >= 0.65);
      const topDecileWinRate = highConf.length > 0 ? highConf.filter((d) => d.isCorrect).length / highConf.length : 0;
      const calibrationSummary = buildCalibrationSummary(completedDocs);
      const calibrationRows = completedDocs.map(calibrationRowFromDoc);

      // 8. Build fresh docs only for the recent/upcoming window so we never
      //    rewrite thousands of unchanged historical dates on refresh.
      const freshDates = new Set(freshRaw.map((g) => g.date));
      const rowsByPk = new Map(rows.map((r) => [r.game.gamePk, r]));
      const freshDocs: GameDoc[] = [];
      for (const g of enriched) {
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
      const todaysRecord = buildTodaysRecord(completedDocs, today);

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
      const calibrationByDate = new Map<string, ReturnType<typeof calibrationRowFromDoc>[]>();
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

    const marketOdds = await fetchMarketOdds();

    const docs = enriched.map((g) => {
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
