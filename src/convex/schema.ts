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
      runProjection: v.optional(v.any()),
      marketOdds: v.optional(v.any()),
    }).index("by_date", ["date"]),

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
      todaysRecord: v.any(),
    }).index("by_key", ["key"]),
  },
  {
    schemaValidation: false,
  },
);

export default schema;
