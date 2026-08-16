import { v } from "convex/values";
import { internalMutation, internalQuery, query } from "./_generated/server";
import { calibrationCurvePoints, evaluate } from "./ml/model";

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

// Calibration dashboard data: completed games in a date range plus favorite-framed
// calibration metrics (one side per game — predicted probability > 50%).
export const getCalibrationResults = query({
  args: { startDate: v.string(), endDate: v.string() },
  handler: async (ctx, args) => {
    const all = await ctx.db.query("games").collect();
    const games = all
      .filter(
        (g) =>
          (g.winner === "home" || g.winner === "away") &&
          g.date >= args.startDate &&
          g.date <= args.endDate,
      )
      .sort((a, b) => (a.date < b.date ? 1 : -1))
      .map((g) => ({
        gamePk: g.gamePk,
        date: g.date,
        away: { abbrev: g.away.abbrev, name: g.away.name, score: g.away.score },
        home: { abbrev: g.home.abbrev, name: g.home.name, score: g.home.score },
        winner: g.winner,
        pickTeam: g.pickTeam,
        pickProb: g.pickProb,
        isCorrect: g.isCorrect,
        isUpset: g.isUpset,
      }));

    const total = games.length;
    const correct = games.filter((g) => g.isCorrect).length;

    if (total === 0) {
      return {
        games,
        metrics: {
          auc: 0,
          brier: 0,
          logLoss: 0,
          ece: 0,
          bins: [],
          confidenceDistribution: [],
          calibrationCurve: [],
        },
        total: 0,
        correct: 0,
        accuracy: 0,
      };
    }

    const preds = games.map((g) => g.pickProb);
    const labels = games.map((g) => (g.isCorrect ? 1 : 0));
    const evalResult = evaluate(preds, labels);
    const curve = calibrationCurvePoints(preds, labels, 8);

    return {
      games,
      metrics: {
        auc: evalResult.auc,
        brier: evalResult.brier,
        logLoss: evalResult.logLoss,
        ece: evalResult.ece,
        bins: evalResult.bins,
        confidenceDistribution: evalResult.confidenceDistribution,
        calibrationCurve: curve.length > 0 ? curve : evalResult.calibrationCurve,
      },
      total,
      correct,
      accuracy: correct / total,
    };
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
