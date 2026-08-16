import { v } from "convex/values";
import { internalMutation, internalQuery, query, QueryCtx } from "./_generated/server";
import { calibrationCurvePoints, computeAuc, computeBrier, evaluate } from "./ml/model";
import type { CalibrationBin, ConfidencePoint, CurvePoint } from "./ml/types";

// Latest trained model (singleton key = "current").
export const getModelState = query({
  args: {},
  handler: async (ctx) =>
    ctx.db
      .query("modelState")
      .withIndex("by_key", (q) => q.eq("key", "current"))
      .first(),
});

// Games for a single date (completed or predicted).
export const getGamesByDate = query({
  args: { date: v.string() },
  handler: async (ctx, args) =>
    ctx.db
      .query("games")
      .withIndex("by_date", (q) => q.eq("date", args.date))
      .collect(),
});

// Completed-game row used by both calibration metrics and the game history table.
interface CalibrationRow {
  gamePk: number;
  date: string;
  away: { abbrev: string; name: string; score?: number };
  home: { abbrev: string; name: string; score?: number };
  winner?: string;
  pickTeam: string;
  pickProb: number;
  isCorrect?: boolean;
  isUpset?: boolean;
  predictedTotal?: number;
  homeRunLineProb?: number;
  actualTotal?: number;
  actualMargin?: number;
}

async function loadCalibrationRows(
  ctx: QueryCtx,
  startDate: string,
  endDate: string,
): Promise<CalibrationRow[]> {
  const rows: CalibrationRow[] = [];
  let cursor: string | null = null;
  do {
    const page = await ctx.db
      .query("games")
      .withIndex("by_date", (q) => q.gte("date", startDate).lte("date", endDate))
      .paginate({ numItems: 1000, cursor });
    for (const g of page.page) {
      if (g.winner !== "home" && g.winner !== "away") continue;
      rows.push({
        gamePk: g.gamePk,
        date: g.date,
        away: { abbrev: g.away.abbrev, name: g.away.name, score: g.away.score },
        home: { abbrev: g.home.abbrev, name: g.home.name, score: g.home.score },
        winner: g.winner,
        pickTeam: g.pickTeam,
        pickProb: g.pickProb,
        isCorrect: g.isCorrect,
        isUpset: g.isUpset,
        predictedTotal: g.runProjection?.total,
        homeRunLineProb: g.runProjection?.homeRunLineProb,
        actualTotal: (g.away.score ?? 0) + (g.home.score ?? 0),
        actualMargin: (g.home.score ?? 0) - (g.away.score ?? 0),
      });
    }
    cursor = page.continueCursor;
  } while (cursor !== null);

  return rows.sort((a, b) =>
    a.date < b.date ? 1 : a.date > b.date ? -1 : a.gamePk < b.gamePk ? 1 : -1,
  );
}

// Calibration metrics computed server-side over the full range so the client
// only receives a small aggregate (never thousands of game rows at once).
export const getCalibrationResults = query({
  args: { startDate: v.string(), endDate: v.string() },
  handler: async (ctx, args) => {
    const games = await loadCalibrationRows(ctx, args.startDate, args.endDate);
    const total = games.length;
    const correct = games.filter((g) => g.isCorrect).length;

    // Moneyline (favorite framing) metrics.
    let metrics = {
      auc: 0,
      brier: 0,
      logLoss: 0,
      ece: 0,
      bins: [] as CalibrationBin[],
      confidenceDistribution: [] as ConfidencePoint[],
      calibrationCurve: [] as CurvePoint[],
    };
    if (total > 0) {
      const preds = games.map((g) => g.pickProb);
      const labels = games.map((g) => (g.isCorrect ? 1 : 0));
      const evalResult = evaluate(preds, labels);
      const curve = calibrationCurvePoints(preds, labels, 8);
      metrics = {
        auc: evalResult.auc,
        brier: evalResult.brier,
        logLoss: evalResult.logLoss,
        ece: evalResult.ece,
        bins: evalResult.bins,
        confidenceDistribution: evalResult.confidenceDistribution,
        calibrationCurve: curve.length > 0 ? curve : evalResult.calibrationCurve,
      };
    }

    // Totals (predicted vs actual combined runs) metrics.
    let totalsMetrics = { n: 0, mae: 0, rmse: 0, bias: 0 };
    {
      const rows = games.filter((g) => typeof g.predictedTotal === "number");
      totalsMetrics.n = rows.length;
      if (rows.length > 0) {
        let absSum = 0;
        let sqSum = 0;
        let biasSum = 0;
        for (const g of rows) {
          const err = (g.predictedTotal as number) - (g.actualTotal as number);
          absSum += Math.abs(err);
          sqSum += err * err;
          biasSum += err;
        }
        totalsMetrics.mae = absSum / rows.length;
        totalsMetrics.rmse = Math.sqrt(sqSum / rows.length);
        totalsMetrics.bias = biasSum / rows.length;
      }
    }

    // Run-line (±1.5) metrics: home covers when it wins by 2+.
    let runLineMetrics = { n: 0, auc: 0, brier: 0, accuracy: 0 };
    {
      const rows = games.filter(
        (g) => typeof g.homeRunLineProb === "number" && typeof g.actualMargin === "number",
      );
      runLineMetrics.n = rows.length;
      if (rows.length > 0) {
        const preds = rows.map((g) => g.homeRunLineProb as number);
        const labels = rows.map((g) => ((g.actualMargin as number) >= 2 ? 1 : 0));
        runLineMetrics.auc = computeAuc(preds, labels);
        runLineMetrics.brier = computeBrier(preds, labels);
        const hits = rows.filter((g, i) => (((g.homeRunLineProb as number) >= 0.5 ? 1 : 0) === labels[i])).length;
        runLineMetrics.accuracy = hits / rows.length;
      }
    }

    return {
      metrics,
      totalsMetrics,
      runLineMetrics,
      total,
      correct,
      accuracy: total > 0 ? correct / total : 0,
    };
  },
});

// Paginated game history for the calibration table.
export const getCalibrationGames = query({
  args: {
    startDate: v.string(),
    endDate: v.string(),
    cursor: v.optional(v.string()),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const rows = await loadCalibrationRows(ctx, args.startDate, args.endDate);
    const limit = Math.max(1, Math.min(args.limit ?? 100, 300));
    let start = 0;
    if (args.cursor) {
      const [cd, cpk] = args.cursor.split("|");
      const pk = Number(cpk);
      start = rows.findIndex((g) => g.date < cd || (g.date === cd && g.gamePk < pk));
      if (start === -1) start = rows.length;
    }
    const page = rows.slice(start, start + limit);
    const cursor =
      start + limit < rows.length
        ? `${page[page.length - 1].date}|${page[page.length - 1].gamePk}`
        : null;
    return { games: page, cursor };
  },
});

export const getLatestModelState = internalQuery({
  args: {},
  handler: async (ctx) =>
    ctx.db
      .query("modelState")
      .withIndex("by_key", (q) => q.eq("key", "current"))
      .first(),
});

export const replaceModelState = internalMutation({
  args: { state: v.any() },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("modelState")
      .withIndex("by_key", (q) => q.eq("key", "current"))
      .first();
    const doc = { ...args.state, key: "current" };
    if (existing) {
      await ctx.db.patch(existing._id, doc);
    } else {
      await ctx.db.insert("modelState", doc);
    }
  },
});

export const replaceGamesForDate = internalMutation({
  args: { date: v.string(), games: v.array(v.any()) },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("games")
      .withIndex("by_date", (q) => q.eq("date", args.date))
      .collect();
    for (const g of existing) await ctx.db.delete(g._id);
    for (const g of args.games) await ctx.db.insert("games", g);
  },
});

export const clearGames = internalMutation({
  args: {},
  handler: async (ctx) => {
    const all = await ctx.db.query("games").collect();
    for (const g of all) await ctx.db.delete(g._id);
  },
});
