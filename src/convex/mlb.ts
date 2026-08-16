import { v } from "convex/values";
import { internalMutation, internalQuery, query } from "./_generated/server";
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

// Load all stored games in bounded pages. The refresh action uses this to
// reuse already-stored seasons instead of re-fetching them from the API.
export const getGamesPage = internalQuery({
  args: { cursor: v.union(v.string(), v.null()), limit: v.optional(v.number()) },
  handler: async (ctx, args) => {
    const page = await ctx.db
      .query("games")
      .paginate({ numItems: Math.min(args.limit ?? 1000, 2000), cursor: args.cursor ?? null });
    return { games: page.page, cursor: page.continueCursor };
  },
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

// Minimal structural view of a stored game doc used for calibration.
interface CalibrationGame {
  gamePk: number;
  date: string;
  away: { abbrev: string; name: string; score?: number };
  home: { abbrev: string; name: string; score?: number };
  winner?: string;
  pickTeam: string;
  pickProb: number;
  isCorrect?: boolean;
  isUpset?: boolean;
  runProjection?: { total?: number; homeRunLineProb?: number };
}

function toCalibrationRow(g: CalibrationGame): CalibrationRow {
  return {
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
  };
}

// Calibration metrics computed over the selected date range. Uses a single
// (non-paginated) collection so Convex's single-paginated-query rule is never
// triggered; only small aggregates are returned to the client.
export const getCalibrationResults = query({
  args: { startDate: v.string(), endDate: v.string() },
  handler: async (ctx, args) => {
    const games = await ctx.db
      .query("games")
      .withIndex("by_date", (q) => q.gte("date", args.startDate).lte("date", args.endDate))
      .collect();

    const preds: number[] = [];
    const labels: number[] = [];
    let total = 0;
    let correct = 0;
    let tN = 0;
    let tAbs = 0;
    let tSq = 0;
    let tBias = 0;
    const rlPreds: number[] = [];
    const rlLabels: number[] = [];

    for (const g of games) {
      if (g.winner !== "home" && g.winner !== "away") continue;
      total += 1;
      if (g.isCorrect) correct += 1;
      preds.push(g.pickProb);
      labels.push(g.isCorrect ? 1 : 0);

      const predictedTotal = g.runProjection?.total;
      if (typeof predictedTotal === "number") {
        tN += 1;
        const err = predictedTotal - ((g.away.score ?? 0) + (g.home.score ?? 0));
        tAbs += Math.abs(err);
        tSq += err * err;
        tBias += err;
      }

      const homeRunLineProb = g.runProjection?.homeRunLineProb;
      if (typeof homeRunLineProb === "number") {
        const margin = (g.home.score ?? 0) - (g.away.score ?? 0);
        rlPreds.push(homeRunLineProb);
        rlLabels.push(margin >= 2 ? 1 : 0);
      }
    }

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
    cursor: v.union(v.string(), v.null()),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    // Paginate directly on the date index (newest first) so each call reads
    // only a small page instead of the entire multi-season dataset.
    const limit = Math.max(1, Math.min(args.limit ?? 100, 1000));
    const page = await ctx.db
      .query("games")
      .withIndex("by_date", (q) => q.gte("date", args.startDate).lte("date", args.endDate))
      .order("desc")
      .paginate({ numItems: limit, cursor: args.cursor ?? null });
    const games: CalibrationRow[] = [];
    for (const g of page.page) {
      if (g.winner !== "home" && g.winner !== "away") continue;
      games.push(toCalibrationRow(g));
    }
    return { games, cursor: page.continueCursor };
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
