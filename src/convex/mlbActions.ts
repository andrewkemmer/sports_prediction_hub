"use node";

import { v } from "convex/values";
import { internal } from "./_generated/api";
import { action } from "./_generated/server";
import {
  applyModel,
  buildFeaturesForGame,
  calibrationCurvePoints,
  evaluate,
  runModel,
} from "./ml/model";
import { teamMeta } from "./ml/teams";
import {
  FeatureRow,
  GameDoc,
  InjurySnapshot,
  PitcherInfo,
  PowerRanking,
  RawGame,
  TeamState,
  TodaysRecord,
  TrainedModel,
} from "./ml/types";

const MLB_BASE = "https://statsapi.mlb.com";
const SEASON_START_MD = "03-15";
const UPCOMING_WINDOW_DAYS = 3;

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
  return `${MLB_BASE}/api/v1/schedule?sportId=1&startDate=${start}&endDate=${end}&hydrate=probablePitcher,linescore`;
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
  };
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

// ---------------------------------------------------------------------------
// Pitcher stats (season ERA / K/9) — display + matchup context
// ---------------------------------------------------------------------------

async function fetchPitcherStats(ids: number[], season: string): Promise<Map<number, { era?: number; k9?: number }>> {
  const out = new Map<number, { era?: number; k9?: number }>();
  const unique = [...new Set(ids)].filter((id) => id > 0);
  await mapLimit(unique, 6, async (id) => {
    try {
      const data = await fetchJson(
        `${MLB_BASE}/api/v1/people/${id}/stats?stats=season&group=pitching&season=${season}`,
      );
      const stat = data?.stats?.[0]?.splits?.[0]?.stat;
      if (stat) {
        const era = typeof stat.era === "string" ? parseFloat(stat.era) : stat.era;
        const k9 =
          typeof stat.strikeoutsPer9Inn === "string"
            ? parseFloat(stat.strikeoutsPer9Inn)
            : stat.strikeoutsPer9Inn;
        out.set(id, {
          era: Number.isFinite(era) ? era : undefined,
          k9: Number.isFinite(k9) ? k9 : undefined,
        });
      }
    } catch {
      // ignore individual pitcher stat failures
    }
  });
  return out;
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
  pitcherStats: Map<number, { era?: number; k9?: number }>,
  injuries?: { home: number; away: number },
): GameDoc {
  const awayPitcher: PitcherInfo | undefined = game.awayPitcher
    ? { ...game.awayPitcher, ...(pitcherStats.get(game.awayPitcher.id) ?? {}) }
    : undefined;
  const homePitcher: PitcherInfo | undefined = game.homePitcher
    ? { ...game.homePitcher, ...(pitcherStats.get(game.homePitcher.id) ?? {}) }
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

    // 1. Fetch every regular-season game through today (single consolidated source).
    const allGames = await fetchSeason(season, today);
    const completed = allGames.filter((g) => g.winner === "home" || g.winner === "away");
    if (completed.length < 40) {
      throw new Error(
        `Only ${completed.length} completed ${season} regular-season games found in MLB Stats API. Cannot train yet.`,
      );
    }

    // 2. Pull as-of-time injured-list snapshots for every team that has played,
    //    then train, select features/model, calibrate, and decide on Monte Carlo.
    const teamIds = [...new Set(completed.flatMap((g) => [g.home.id, g.away.id]))];
    const injurySnapshots = await fetchInjurySnapshots(
      teamIds,
      season,
      `${season}-${SEASON_START_MD}`,
      today,
    );
    const run = runModel(completed, { season, asOfDate: today }, injurySnapshots);
    const { result, model, rows, teamState } = run;

    // 3. As-of-time predictions for every completed game (historical results).
    const completedDocs: GameDoc[] = rows.map((row: FeatureRow) =>
      buildGameDoc(
        row.game,
        applyModel(model, row.features, row.homeElo, row.awayElo),
        new Map(),
      ),
    );

    // Descriptive reliability / calibration views over the full 2026 season.
    // Favorite framing: one side per game (predicted probability > 50%) vs. outcome.
    const fullPreds = completedDocs.map((d) => d.pickProb);
    const fullLabels = completedDocs.map((d) => (d.isCorrect ? 1 : 0));
    const fullCurve = calibrationCurvePoints(fullPreds, fullLabels, 12);
    const fullEval = evaluate(fullPreds, fullLabels);

    // 4. Upcoming games (today through +3 days) with starter stat context.
    const windowEnd = addDays(today, UPCOMING_WINDOW_DAYS);
    const upcomingRaw = await fetchScheduleRange(today, windowEnd);
    const pitcherIds: number[] = [];
    for (const g of upcomingRaw) {
      if (g.awayPitcher) pitcherIds.push(g.awayPitcher.id);
      if (g.homePitcher) pitcherIds.push(g.homePitcher.id);
    }
    const pitcherStats = await fetchPitcherStats(pitcherIds, season);
    const upcomingDocs: GameDoc[] = upcomingRaw
      .filter((g) => g.winner !== "home" && g.winner !== "away")
      .map((g) =>
        buildGameDoc(g, predictionFor(g, model, teamState), pitcherStats, {
          home: teamState.injuries[g.home.id] ?? 0,
          away: teamState.injuries[g.away.id] ?? 0,
        }),
      );

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

    const raw = await fetchScheduleRange(args.date, args.date);
    const pitcherIds: number[] = [];
    for (const g of raw) {
      if (g.awayPitcher) pitcherIds.push(g.awayPitcher.id);
      if (g.homePitcher) pitcherIds.push(g.homePitcher.id);
    }
    const pitcherStats = await fetchPitcherStats(pitcherIds, state.season as string);

    const docs = raw.map((g) =>
      buildGameDoc(g, predictionFor(g, model, teamState), pitcherStats, {
        home: teamState.injuries[g.home.id] ?? 0,
        away: teamState.injuries[g.away.id] ?? 0,
      }),
    );
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
