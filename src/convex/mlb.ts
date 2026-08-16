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

// Load stored games for a date range in bounded pages. The refresh action uses
// this to reuse already-stored seasons instead of re-fetching them from the
// API. Pages are kept at 300 docs so full game documents (with SHAP + run
// projections) never push a single query response over Convex's limit.
export const getGamesByDateRange = internalQuery({
  args: {
    startDate: v.string(),
    endDate: v.string(),
    cursor: v.union(v.string(), v.null()),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const page = await ctx.db
      .query("games")
      .withIndex("by_date", (q) => q.gte("date", args.startDate).lte("date", args.endDate))
      .order("asc")
      .paginate({ numItems: Math.min(args.limit ?? 300, 500), cursor: args.cursor ?? null });
    return { games: page.page, cursor: page.continueCursor };
  },
});

// Calibration metrics computed over the selected date range. The default
// full-range view is served from a precomputed summary stored on the model
// state; narrower ranges are computed from the lightweight `calibration`
// projection so full-range reads never approach Convex's transaction limit.
export const getCalibrationResults = query({
  args: { startDate: v.string(), endDate: v.string() },
  handler: async (ctx, args) => {
    const state = await ctx.db
      .query("modelState")
      .withIndex("by_key", (q) => q.eq("key", "current"))
      .first();
    if (
      state?.calibrationSummary &&
      (state.calibrationSummary as { total?: number }).total !== 0 &&
      args.startDate <= "2022-03-15" &&
      typeof state.asOfDate === "string" &&
      args.endDate >= state.asOfDate
    ) {
      return state.calibrationSummary;
    }

    const games = await ctx.db
      .query("calibration")
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

      const predictedTotal = g.predictedTotal;
      if (typeof predictedTotal === "number") {
        tN += 1;
        const actual = g.actualTotal ?? (g.away.score ?? 0) + (g.home.score ?? 0);
        const err = predictedTotal - actual;
        tAbs += Math.abs(err);
        tSq += err * err;
        tBias += err;
      }

      const homeRunLineProb = g.homeRunLineProb;
      if (typeof homeRunLineProb === "number") {
        const margin = g.actualMargin ?? (g.home.score ?? 0) - (g.away.score ?? 0);
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

// Paginated game history for the calibration table, served from the
// lightweight calibration projection.
export const getCalibrationGames = query({
  args: {
    startDate: v.string(),
    endDate: v.string(),
    cursor: v.union(v.string(), v.null()),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const limit = Math.max(1, Math.min(args.limit ?? 100, 1000));
    const page = await ctx.db
      .query("calibration")
      .withIndex("by_date", (q) => q.gte("date", args.startDate).lte("date", args.endDate))
      .order("desc")
      .paginate({ numItems: limit, cursor: args.cursor ?? null });
    return { games: page.page, cursor: page.continueCursor };
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

// Live progress for the long-running refresh action. The client subscribes to
// this while the server-side action runs so the refresh button can show a real
// progress bar instead of an indeterminate spinner.
export const getRefreshProgress = query({
  args: {},
  handler: async (ctx) =>
    ctx.db
      .query("refreshProgress")
      .withIndex("by_key", (q) => q.eq("key", "current"))
      .first(),
});

export const setRefreshProgress = internalMutation({
  args: {
    stage: v.string(),
    pct: v.number(),
    message: v.string(),
    done: v.optional(v.boolean()),
    error: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("refreshProgress")
      .withIndex("by_key", (q) => q.eq("key", "current"))
      .first();
    const doc = {
      key: "current",
      stage: args.stage,
      pct: args.pct,
      message: args.message,
      startedAt: existing?.startedAt ?? Date.now(),
      updatedAt: Date.now(),
      done: args.done ?? false,
      error: args.error,
    };
    if (existing) await ctx.db.patch(existing._id, doc);
    else await ctx.db.insert("refreshProgress", doc);
  },
});

export const replaceModelState = internalMutation({
  args: { state: v.any() },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("modelState")
      .withIndex("by_key", (q) => q.eq("key", "current"))
      .first();
    const { _id: _ignored, _creationTime: _ct, ...state } = args.state as Record<string, unknown>;
    const doc = { ...state, key: "current" } as any;
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
    for (const g of args.games) {
      // Stored docs (re-read from the DB) carry Convex's own `_id` and
      // `_creationTime`; strip them so insert() can mint fresh ones.
      const { _id: _ignored, _creationTime: _ct, ...rest } = g as Record<string, unknown>;
      await ctx.db.insert("games", rest as any);
    }
  },
});

export const replaceCalibrationForDate = internalMutation({
  args: { date: v.string(), rows: v.array(v.any()) },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("calibration")
      .withIndex("by_date", (q) => q.eq("date", args.date))
      .collect();
    for (const r of existing) await ctx.db.delete(r._id);
    for (const r of args.rows) {
      const { _id: _ignored, _creationTime: _ct, ...rest } = r as Record<string, unknown>;
      await ctx.db.insert("calibration", rest as any);
    }
  },
});

export const clearGames = internalMutation({
  args: {},
  handler: async (ctx) => {
    const all = await ctx.db.query("games").collect();
    for (const g of all) await ctx.db.delete(g._id);
  },
});
