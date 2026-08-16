"use node";

import { v } from "convex/values";
import { internal } from "./_generated/api";
import { action } from "./_generated/server";
import {
  applyIsotonic,
  applyModel,
  buildFeaturesForGame,
  calibrationCurvePoints,
  evaluate,
  runModel,
  spearmanRank,
} from "./ml/model";
import { RunModel, simulateRuns } from "./ml/runs";
import { teamMeta } from "./ml/teams";
import {
  FeatureRow,
  GameDoc,
  GameWeather,
  InjurySnapshot,
  MarketOdds,
  PitcherInfo,
  PowerRanking,
  RawGame,
  RunProjection,
  TeamState,
  TodaysRecord,
  TrainedModel,
} from "./ml/types";

const MLB_BASE = "https://statsapi.mlb.com";
const SEASON_START_MD = "03-15";
const UPCOMING_WINDOW_DAYS = 3;
const TRAIN_SEASONS = ["2022", "2023", "2024", "2025"];
const PAST_SEASON_END_MD = "11-01";
const RUN_SIM_TRIALS = 10000;
const RUN_CALIB_TRIALS = 2000;

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
  const ranges = dateRanges(start, throughDate, 10);
  const seen = new Map<number, RawGame>();
  await mapLimit(ranges, 4, async (r) => {
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
  await mapLimit(unique, 8, async (p) => {
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

function buildRunProjection(
  runModel: RunModel,
  runLineIso: { x: number; y: number }[],
  game: RawGame,
  marketTotal?: number,
  trials = RUN_SIM_TRIALS,
): RunProjection {
  const first = simulateRuns(runModel, game.home.id, game.away.id, 0, trials);
  const line = marketTotal ?? first.total;
  const sim = line === 0 ? first : simulateRuns(runModel, game.home.id, game.away.id, line, trials);
  const homeRL = runLineIso.length > 0 ? applyIsotonic(runLineIso, sim.homeRunLineProb) : sim.homeRunLineProb;
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

const INJURY_SNAPSHOT_DAYS = 7;

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
): Promise<Map<number, InjurySnapshot[]>> {
  const dates: string[] = [];
  let cursor = startDate;
  while (cursor <= endDate) {
    dates.push(cursor);
    cursor = addDays(cursor, INJURY_SNAPSHOT_DAYS);
  }
  if (dates[dates.length - 1] !== endDate) dates.push(endDate);

  const jobs: { teamId: number; date: string }[] = [];
  for (const teamId of teamIds) {
    for (const date of dates) jobs.push({ teamId, date });
  }

  const results = await mapLimit(jobs, 6, async (job) => ({
    teamId: job.teamId,
    date: job.date,
    count: await fetchInjuryCount(job.teamId, job.date, season),
  }));

  const out = new Map<number, InjurySnapshot[]>();
  for (const r of results) {
    const list = out.get(r.teamId) ?? [];
    list.push({ date: r.date, count: r.count });
    out.set(r.teamId, list);
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

export const refreshModel = action({
  args: {},
  handler: async (ctx) => {
    const now = new Date();
    const season = String(now.getFullYear());
    const today = etDateString(now);

    // 1. Fetch 2022–2025 full seasons + 2026 through today (regular season only).
    const allGames = await fetchAllSeasons(season, today);
    const completed = allGames.filter((g) => g.winner === "home" || g.winner === "away");
    if (completed.length < 40) {
      throw new Error(
        `Only ${completed.length} completed regular-season games found in MLB Stats API. Cannot train yet.`,
      );
    }

    // 1b. Fetch the upcoming window (current season) for on-demand predictions.
    const windowEnd = addDays(today, UPCOMING_WINDOW_DAYS);
    const upcomingRaw = await fetchScheduleRange(today, windowEnd);

    // 1c. Season pitching stats (ERA / K9 / FIP) for every starter in scope,
    //     keyed by (pitcher, season) so past seasons use that season's stats.
    const pitcherPairs: { id: number; season: string }[] = [];
    for (const g of [...completed, ...upcomingRaw]) {
      const s = g.season ?? season;
      if (g.awayPitcher) pitcherPairs.push({ id: g.awayPitcher.id, season: s });
      if (g.homePitcher) pitcherPairs.push({ id: g.homePitcher.id, season: s });
    }
    const pitcherStats = await fetchPitcherStats(pitcherPairs);
    const completedEnriched = attachPitcherStats(completed, pitcherStats);

    // 2. Pull as-of-time injured-list snapshots for every team that has played,
    //    then train, select features/model, calibrate, and decide on Monte Carlo.
    const teamIds = [...new Set(completed.flatMap((g) => [g.home.id, g.away.id]))];
    const injurySnapshots = await fetchInjurySnapshots(
      teamIds,
      season,
      `${season}-${SEASON_START_MD}`,
      today,
    );
    const run = runModel(completedEnriched, { season, asOfDate: today }, injurySnapshots);
    const { result, model, rows, teamState } = run;
    const runModelState = result.runModel;
    const runLineIso = result.runLineCalibration ?? [];

    // Market odds (best-effort; empty map when THE_ODDS_API_KEY is unset).
    const marketOdds = await fetchMarketOdds();

    // 3. As-of-time predictions for every completed game (historical results),
    //    plus run-scoring projections for the totals / run-line views.
    const completedDocs: GameDoc[] = rows.map((row: FeatureRow) =>
      buildGameDoc(
        row.game,
        applyModel(model, row.features, row.homeElo, row.awayElo),
        pitcherStats,
        undefined,
        buildRunProjection(runModelState, runLineIso, row.game, undefined, RUN_CALIB_TRIALS),
      ),
    );

    // Descriptive reliability / calibration views over the full 2022–2026 dataset.
    // Favorite framing: one side per game (predicted probability > 50%) vs. outcome.
    const fullPreds = completedDocs.map((d) => d.pickProb);
    const fullLabels = completedDocs.map((d) => (d.isCorrect ? 1 : 0));
    const fullCurve = calibrationCurvePoints(fullPreds, fullLabels, 12);
    const fullEval = evaluate(fullPreds, fullLabels);
    const spearmanRho = spearmanRank(fullPreds, fullLabels);
    const highConf = completedDocs.filter((d) => d.pickProb >= 0.65);
    const topDecileWinRate = highConf.length > 0 ? highConf.filter((d) => d.isCorrect).length / highConf.length : 0;

    // 4. Upcoming games (today through +3 days) with starter stat context,
    //    run projections, and market odds.
    const upcomingEnriched = attachPitcherStats(upcomingRaw, pitcherStats);
    const upcomingDocs: GameDoc[] = upcomingEnriched
      .filter((g) => g.winner !== "home" && g.winner !== "away")
      .map((g) => {
        const odds = marketOddsForGame(marketOdds, g);
        return buildGameDoc(
          g,
          predictionFor(g, model, teamState),
          pitcherStats,
          { home: teamState.injuries[g.home.id] ?? 0, away: teamState.injuries[g.away.id] ?? 0 },
          buildRunProjection(runModelState, runLineIso, g, odds?.total, RUN_SIM_TRIALS),
          odds,
        );
      });

    // 5. Today's record.
    const todaysRecord = buildTodaysRecord(completedDocs, today);

    // 6. Store the model state (singleton).
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
        spearmanRho,
        topDecileWinRate,
        todaysRecord,
      },
    });

    // 7. Store game docs grouped by date.
    const byDate = new Map<string, GameDoc[]>();
    for (const doc of [...completedDocs, ...upcomingDocs]) {
      const list = byDate.get(doc.date) ?? [];
      list.push(doc);
      byDate.set(doc.date, list);
    }
    const dates = [...byDate.keys()];
    await mapLimit(dates, 8, (date) =>
      ctx.runMutation(internal.mlb.replaceGamesForDate, { date, games: byDate.get(date)! }),
    );

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
      storedGames: completedDocs.length + upcomingDocs.length,
    };
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
    const season = state.season as string;

    const raw = await fetchScheduleRange(args.date, args.date);
    const pitcherPairs: { id: number; season: string }[] = [];
    for (const g of raw) {
      const s = g.season ?? season;
      if (g.awayPitcher) pitcherPairs.push({ id: g.awayPitcher.id, season: s });
      if (g.homePitcher) pitcherPairs.push({ id: g.homePitcher.id, season: s });
    }
    const pitcherStats = await fetchPitcherStats(pitcherPairs);
    const enriched = attachPitcherStats(raw, pitcherStats);

    const marketOdds = await fetchMarketOdds();

    const docs = enriched.map((g) => {
      const odds = marketOddsForGame(marketOdds, g);
      return buildGameDoc(
        g,
        predictionFor(g, model, teamState),
        pitcherStats,
        { home: teamState.injuries[g.home.id] ?? 0, away: teamState.injuries[g.away.id] ?? 0 },
        runModelState ? buildRunProjection(runModelState, runLineIso, g, odds?.total, RUN_SIM_TRIALS) : undefined,
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
