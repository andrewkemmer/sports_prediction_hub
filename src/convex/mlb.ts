import { v } from "convex/values";
import { internalMutation, internalQuery, query } from "./_generated/server";

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

// Slim completed-game results (predicted vs actual) for the calibration dashboard.
export const getCompletedGameResults = query({
  args: {},
  handler: async (ctx) => {
    const all = await ctx.db.query("games").collect();
    return all
      .filter((g) => g.winner === "home" || g.winner === "away")
      .sort((a, b) => (a.date < b.date ? 1 : -1))
      .map((g) => ({
        gamePk: g.gamePk,
        date: g.date,
        away: { abbrev: g.away.abbrev, name: g.away.name, score: g.away.score },
        home: { abbrev: g.home.abbrev, name: g.home.name, score: g.home.score },
        winner: g.winner,
        pickTeam: g.pickTeam,
        pickProb: g.pickProb,
        homeWinProb: g.homeWinProb,
        awayWinProb: g.awayWinProb,
        isCorrect: g.isCorrect,
        isUpset: g.isUpset,
      }));
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
