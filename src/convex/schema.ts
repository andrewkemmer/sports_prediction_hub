import { authTables } from "@convex-dev/auth/server";
import { defineSchema, defineTable } from "convex/server";
import { Infer, v } from "convex/values";

// default user roles. can add / remove based on the project as needed
export const ROLES = {
  ADMIN: "admin",
  USER: "user",
  MEMBER: "member",
} as const;

export const roleValidator = v.union(
  v.literal(ROLES.ADMIN),
  v.literal(ROLES.USER),
  v.literal(ROLES.MEMBER),
);
export type Role = Infer<typeof roleValidator>;

const teamInfo = v.object({
  id: v.number(),
  abbrev: v.string(),
  name: v.string(),
  score: v.optional(v.number()),
  wins: v.optional(v.number()),
  losses: v.optional(v.number()),
  ops: v.optional(v.number()),
  era: v.optional(v.number()),
  fieldingPct: v.optional(v.number()),
});

const pitcherInfo = v.object({
  id: v.number(),
  name: v.string(),
  era: v.optional(v.number()),
  k9: v.optional(v.number()),
  fip: v.optional(v.number()),
});

const shapContribution = v.object({
  feature: v.string(),
  label: v.string(),
  value: v.number(),
  contribution: v.number(),
});

const schema = defineSchema(
  {
    // default auth tables using convex auth.
    ...authTables, // do not remove or modify

    // the users table is the default users table that is brought in by the authTables
    users: defineTable({
      name: v.optional(v.string()), // name of the user. do not remove
      image: v.optional(v.string()), // image of the user. do not remove
      email: v.optional(v.string()), // email of the user. do not remove
      emailVerificationTime: v.optional(v.number()), // email verification time. do not remove
      isAnonymous: v.optional(v.boolean()), // is the user anonymous. do not remove

      role: v.optional(roleValidator), // role of the user. do not remove
    }).index("email", ["email"]), // index for the email. do not remove or modify

    // A single game (completed or upcoming) with its model prediction.
    games: defineTable({
      gamePk: v.number(),
      date: v.string(), // official date YYYY-MM-DD
      status: v.string(), // Final | Live | Preview | Scheduled | Postponed ...
      detailedState: v.optional(v.string()),
      dayNight: v.string(),
      gameDate: v.string(), // ISO timestamp
      innings: v.optional(v.number()), // currentInning when past 9
      venue: v.optional(v.string()),
      away: teamInfo,
      home: teamInfo,
      awayPitcher: v.optional(pitcherInfo),
      homePitcher: v.optional(pitcherInfo),
      winner: v.optional(v.string()), // "home" | "away"
      homeWinProb: v.number(),
      awayWinProb: v.number(),
      pickTeam: v.string(), // "home" | "away"
      pickProb: v.number(), // probability of the pick (>= 0.5)
      isUpset: v.optional(v.boolean()),
      isCorrect: v.optional(v.boolean()),
      edge: v.optional(v.number()), // model vs Elo baseline edge
      fairAwayOdds: v.optional(v.number()),
      fairHomeOdds: v.optional(v.number()),
      shap: v.optional(v.array(shapContribution)),
      homeInjuries: v.optional(v.number()),
      awayInjuries: v.optional(v.number()),
      season: v.optional(v.string()),
      weather: v.optional(v.any()),
      lineups: v.optional(v.any()), // actual starting 9 + bench (boxscore)
      lineupStats: v.optional(v.any()), // aggregated lineup-strength features
      runProjection: v.optional(v.any()),
      marketOdds: v.optional(v.any()),
    }).index("by_date", ["date"]),

    // Lean per-game calibration projection. Kept small (~1/10 the size of a
    // game doc) so full-range calibration queries never approach Convex's
    // per-transaction read limit.
    calibration: defineTable({
      gamePk: v.number(),
      date: v.string(),
      away: v.object({ abbrev: v.string(), name: v.string(), score: v.optional(v.number()) }),
      home: v.object({ abbrev: v.string(), name: v.string(), score: v.optional(v.number()) }),
      winner: v.optional(v.string()),
      pickTeam: v.string(),
      pickProb: v.number(),
      isCorrect: v.optional(v.boolean()),
      isUpset: v.optional(v.boolean()),
      predictedTotal: v.optional(v.number()),
      homeRunLineProb: v.optional(v.number()),
      actualTotal: v.optional(v.number()),
      actualMargin: v.optional(v.number()),
    }).index("by_date", ["date"]),

    // Resume state for the one-time calibration-history backfill. The backfill
    // runs as a self-scheduling chain of small steps (one page per step); this
    // doc persists the last pagination cursor so a killed step resumes from
    // where it stopped instead of re-scanning the whole table.
    calibrationBackfill: defineTable({
      key: v.string(),
      cursor: v.union(v.string(), v.null()),
      scanned: v.number(),
      done: v.boolean(),
      updatedAt: v.number(),
      error: v.optional(v.string()),
    }).index("by_key", ["key"]),

    // Live progress for the long-running model refresh action. The client
    // subscribes to this so the refresh button can show a real progress bar.
    refreshProgress: defineTable({
      key: v.string(),
      stage: v.string(),
      pct: v.number(),
      message: v.string(),
      startedAt: v.number(),
      updatedAt: v.number(),
      done: v.boolean(),
      error: v.optional(v.string()),
    }).index("by_key", ["key"]),

    // Singleton document (key = "current") describing the trained model.
    modelState: defineTable({
      key: v.string(),
      trainedAt: v.number(),
      season: v.string(),
      asOfDate: v.string(),
      gamesTrained: v.number(),
      holdoutCount: v.number(),
      selectedModel: v.string(),
      modelDescription: v.string(),
      featureNames: v.array(v.string()),
      weights: v.array(v.number()),
      bias: v.number(),
      featureStats: v.any(), // { feature: { mean, std } }
      isotonicPoints: v.any(), // [{ x, y }]
      eloHfa: v.number(),
      monteCarloEnabled: v.boolean(),
      monteCarloTrials: v.number(),
      monteCarloSigma: v.number(),
      monteCarloRationale: v.string(),
      auc: v.number(),
      brier: v.number(),
      logLoss: v.number(),
      ece: v.number(),
      bins: v.any(), // reliability bins (confidence)
      confidenceDistribution: v.any(), // per-bin count + accuracy
      calibrationCurve: v.any(), // home win prob bins
      featureImportances: v.any(),
      candidates: v.any(),
      powerRankings: v.any(),
      featureDrift: v.optional(v.any()),
      rollingBrier: v.optional(v.any()),
      brierBaseline: v.optional(v.number()),
      modelVersions: v.optional(v.any()),
      spearmanRho: v.optional(v.number()),
      topDecileWinRate: v.optional(v.number()),
      runModel: v.optional(v.any()),
      runLineCalibration: v.optional(v.any()),
      runMarginCalibration: v.optional(v.any()),
      teamSeasonStats: v.optional(v.any()),
      injurySnapshots: v.optional(v.any()),
      playerOps: v.optional(v.any()), // cached per-player season OPS (`id|season` → OPS)
      calibrationSummary: v.optional(v.any()),
      todaysRecord: v.any(),
    }).index("by_key", ["key"]),
  },
  {
    schemaValidation: false,
  },
);

export default schema;
