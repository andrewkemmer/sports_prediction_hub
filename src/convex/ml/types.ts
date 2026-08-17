// Shared pure-TS types for the MLB prediction engine.
// Used by the Convex backend (actions) and the frontend. No node/convex imports.

export interface PitcherInfo {
  id: number;
  name: string;
  pitchHand?: "L" | "R"; // throwing handedness (drives the platoon-split feature)
  era?: number;
  k9?: number;
  fip?: number; // fielding-independent pitching (computed from K / BB / HR / IP)
  whip?: number; // walks + hits per inning pitched (as-of-date)
  recentEra?: number; // ERA over the last 3 starts (as-of-date hot/cold signal)
}

export interface TeamInfo {
  id: number;
  abbrev: string;
  name: string;
  score?: number;
  wins?: number;
  losses?: number;
  ops?: number; // season team OPS (hitting / batter production)
  era?: number; // season team ERA (pitching staff incl. bullpen)
  fieldingPct?: number; // season fielding percentage (defensive-efficiency proxy)
  k9?: number; // staff strikeouts per 9 innings (as-of-date)
  whip?: number; // staff walks + hits per inning pitched (as-of-date)
}

export interface ShapContribution {
  feature: string;
  label: string;
  value: number;
  contribution: number;
}

/** One player in a lineup: batting order slot, position, name. */
export interface LineupPlayer {
  id: number;
  name: string;
  pos?: string;
  ops?: number; // OPS as-of the game date (no lookahead)
  woba?: number; // wOBA as-of the game date
  iso?: number; // isolated power as-of the game date
  recentOps?: number; // OPS over the last 10 games (hot-streak signal)
  // Batter-vs-pitcher + platoon matchup data (MLB Stats API vsPlayer / splits).
  bvpOPS?: number; // career OPS vs the opposing starter, shrunk toward season OPS
  bvpPA?: number; // career plate appearances vs the opposing starter (sample size)
  platoonOPS?: number; // season OPS vs the starter's handedness (L/R split)
  vsTeamOPS?: number; // season OPS vs the opposing team
}

/** Actual starting 9 + available bench for a game (from the boxscore). */
export interface LineupData {
  home?: { battingOrder: LineupPlayer[]; bench: LineupPlayer[] };
  away?: { battingOrder: LineupPlayer[]; bench: LineupPlayer[] };
}

/** Aggregated lineup-strength inputs used as model features. */
export interface LineupStats {
  // Weighted means over the starting 9 (slots 1-4 double-weighted), computed
  // strictly as-of the game date.
  home: {
    known: boolean;
    ops: number;
    woba: number;
    iso: number;
    recentOps: number;
    // Matchup edges (present when the real lineup AND the opposing starter are
    // known): career BvP OPS (PA-saturated), season platoon OPS vs the starter's
    // hand, and season OPS vs the opposing team.
    bvpOps?: number;
    bvpPA?: number; // total career PA in the matchup across the starting 9
    platoonOps?: number;
    vsTeamOps?: number;
  };
  away: {
    known: boolean;
    ops: number;
    woba: number;
    iso: number;
    recentOps: number;
    bvpOps?: number;
    bvpPA?: number;
    platoonOps?: number;
    vsTeamOps?: number;
  };
}

/** A parsed schedule game straight from the MLB Stats API. */
export interface RawGame {
  gamePk: number;
  date: string; // official date YYYY-MM-DD
  gameDate: string; // ISO timestamp
  dayNight: string;
  status: string; // Final | Live | Preview | Scheduled | Postponed ...
  detailedState?: string;
  away: TeamInfo;
  home: TeamInfo;
  awayPitcher?: PitcherInfo;
  homePitcher?: PitcherInfo;
  venue?: string;
  innings?: number;
  winner?: "home" | "away";
  season?: string; // season year (e.g. "2024")
  weather?: GameWeather;
  lineups?: LineupData; // actual starting 9 + bench when available
  lineupStats?: LineupStats; // computed lineup-strength features
}

/** Game-time weather conditions (MLB Stats API `hydrate=weather`). */
export interface GameWeather {
  condition?: string;
  tempF?: number;
  windMph?: number;
}

/** Predicted score / totals / run-line projection from the run model. */
export interface RunProjection {
  homeScore: number; // mean runs
  awayScore: number; // mean runs
  total: number; // mean combined runs
  overProb: number; // P(total > line)
  underProb: number; // P(total < line)
  homeRunLineProb: number; // P(home wins by 2+, i.e. covers -1.5)
  awayRunLineProb: number; // P(away loses by <=1 or wins, i.e. covers +1.5)
}

/** Market odds from a third-party odds provider (e.g. The Odds API). */
export interface MarketOdds {
  homeMoneyline?: number; // american odds
  awayMoneyline?: number;
  total?: number; // over/under line
  overPrice?: number;
  underPrice?: number;
  runLine?: number; // e.g. 1.5
  homeRunLinePrice?: number;
  awayRunLinePrice?: number;
  source?: string;
}

/** The feature values used by the model for a single game. */
export interface FeatureValues {
  eloDiff: number; // (homeElo - awayElo) / 100
  winPctDiff: number; // home win% - away win%
  formDiff: number; // home last-10 win% - away last-10 win%
  restDiff: number; // home rest days - away rest days
  injuryDiff: number; // away injured-list count - home injured-list count (positive favors home)
  homeField: number; // always 1
  spFipDiff: number; // away SP FIP - home SP FIP (positive favors home)
  spEraDiff: number; // away SP ERA - home SP ERA (positive favors home)
  spK9Diff: number; // home SP K/9 - away SP K/9 (positive favors home)
  spWhipDiff: number; // away SP WHIP - home SP WHIP (positive favors home)
  spRecentDiff: number; // away SP last-3-start ERA - home SP last-3-start ERA (positive favors home)
  opsDiff: number; // home team OPS - away team OPS (positive favors home)
  teamEraDiff: number; // away team ERA - home team ERA (positive favors home)
  teamK9Diff: number; // home staff K/9 - away staff K/9 (positive favors home)
  teamWhipDiff: number; // away staff WHIP - home staff WHIP (positive favors home)
  defEffDiff: number; // home fieldingPct - away fieldingPct (positive favors home)
  parkFactor: number; // home ballpark run factor (>1 hitter-friendly)
  tempDev: number; // game temperature deviation from 72°F
  windMph: number; // game wind speed
  lineupKnown: number; // 1 when actual lineup data is available for this game
  lineupOpsDiff: number; // home starting-9 weighted OPS - away (positive favors home)
  lineupWobaDiff: number; // home starting-9 wOBA - away (positive favors home)
  lineupIsoDiff: number; // home starting-9 ISO - away (positive favors home)
  lineupHotDiff: number; // home starting-9 L10 OPS - away (positive favors home)
  bvpOpsDiff: number; // home starting-9 career BvP OPS - away (positive favors home)
  platoonOpsDiff: number; // home starting-9 OPS vs starter's hand - away (positive favors home)
  vsTeamOpsDiff: number; // home starting-9 OPS vs away team - away starting-9 OPS vs home team
}

export const FEATURE_KEYS = [
  "eloDiff",
  "winPctDiff",
  "formDiff",
  "restDiff",
  "injuryDiff",
  "homeField",
  "spFipDiff",
  "spEraDiff",
  "spK9Diff",
  "spWhipDiff",
  "spRecentDiff",
  "opsDiff",
  "teamEraDiff",
  "teamK9Diff",
  "teamWhipDiff",
  "defEffDiff",
  "parkFactor",
  "tempDev",
  "windMph",
  "lineupKnown",
  "lineupOpsDiff",
  "lineupWobaDiff",
  "lineupIsoDiff",
  "lineupHotDiff",
  "bvpOpsDiff",
  "platoonOpsDiff",
  "vsTeamOpsDiff",
] as const;

export type FeatureKey = (typeof FEATURE_KEYS)[number];

export const FEATURE_LABELS: Record<FeatureKey, string> = {
  eloDiff: "Elo rating edge",
  winPctDiff: "Win % edge",
  formDiff: "Recent form (L10)",
  restDiff: "Rest advantage",
  injuryDiff: "Injury edge (IL)",
  homeField: "Home field",
  spFipDiff: "Starting Pitcher FIP / xERA Delta",
  spEraDiff: "Starting Pitcher ERA Delta",
  spK9Diff: "Starting Pitcher K/9 Delta",
  spWhipDiff: "Starting Pitcher WHIP Delta",
  spRecentDiff: "Starting Pitcher last-3-start ERA Delta",
  opsDiff: "Team OPS edge",
  teamEraDiff: "Bullpen / Staff ERA edge",
  teamK9Diff: "Staff K/9 edge",
  teamWhipDiff: "Staff WHIP edge",
  defEffDiff: "Defensive efficiency edge",
  parkFactor: "Ballpark factor",
  tempDev: "Weather temperature",
  windMph: "Weather wind",
  lineupKnown: "Lineup data available",
  lineupOpsDiff: "Starting-9 OPS edge",
  lineupWobaDiff: "Starting-9 wOBA edge",
  lineupIsoDiff: "Starting-9 ISO edge",
  lineupHotDiff: "Starting-9 hot streak (L10 OPS)",
  bvpOpsDiff: "Batter vs. Pitcher (BvP) edge",
  platoonOpsDiff: "Platoon split edge (vs starter's hand)",
  vsTeamOpsDiff: "Batter vs. team split edge",
};

/** A training row: features computed strictly as-of the game time. */
export interface FeatureRow {
  game: RawGame;
  features: FeatureValues;
  homeElo: number;
  awayElo: number;
  label: number; // 1 if home team won, else 0
}

/** Current per-team state used to build features for new (upcoming) games. */
export interface TeamState {
  elo: Record<number, number>;
  form: Record<number, number>; // last-10 win percentage
  lastGameDate: Record<number, string>;
  records: Record<number, { wins: number; losses: number }>;
  injuries: Record<number, number>; // players currently on the injured list
}

export interface CalibrationBin {
  label: string;
  meanPredicted: number;
  meanActual: number;
  count: number;
  gap: number;
}

export interface ConfidencePoint {
  label: string;
  count: number;
  accuracy: number;
}

export interface CurvePoint {
  x: number;
  y: number;
  n: number;
}

export interface FeatureImportance {
  feature: FeatureKey;
  label: string;
  weight: number; // standardized logistic coefficient (0 when not selected)
  importance: number; // |weight|
  univariateAuc: number;
  active: boolean; // retained by ML feature selection
}

export interface CandidateModel {
  name: string;
  auc: number;
  brier: number;
  logLoss: number;
  ece: number;
  selected: boolean;
  note: string;
}

export interface PowerRanking {
  teamId: number;
  name: string;
  abbrev: string;
  elo: number;
  wins: number;
  losses: number;
  winPct: number;
  last10WinPct: number;
  lastGameDate: string;
  injuries: number;
  runDiff: number;
  homeWinPct: number;
  awayWinPct: number;
}

/** A team's injured-list count captured on a given date. */
export interface InjurySnapshot {
  date: string;
  count: number;
}

export interface UpsetItem {
  team: string; // winning team abbrev
  loser: string; // losing team abbrev
  prob: number;
}

export interface TodaysRecord {
  date: string;
  total: number;
  completed: number;
  wins: number;
  losses: number;
  correct: number;
  accuracy: number;
  upsets: UpsetItem[];
}

export interface FeatureDriftItem {
  feature: string;
  label: string;
  currentMean: number;
  baselineMean: number;
  psi: number;
  status: "OK" | "WARN";
}

export interface RollingBrierPoint {
  date: string;
  brier: number;
}

export interface ModelVersion {
  version: string;
  date: string;
  auc: number;
  brier: number;
  notes: string;
}

/** A trained, deployable model (reconstructable from modelState). */
export interface TrainedModel {
  featureNames: FeatureKey[];
  weights: number[];
  bias: number;
  featureStats: Record<FeatureKey, { mean: number; std: number }>;
  isotonicPoints: { x: number; y: number }[];
  monteCarloSigma: number;
  monteCarloEnabled: boolean;
  eloHfa: number;
}

/** A game document as stored in the `games` table and rendered on cards. */
export interface GameDoc {
  gamePk: number;
  date: string;
  status: string;
  detailedState?: string;
  dayNight: string;
  gameDate: string;
  innings?: number;
  venue?: string;
  away: TeamInfo;
  home: TeamInfo;
  awayPitcher?: PitcherInfo;
  homePitcher?: PitcherInfo;
  winner?: "home" | "away";
  homeWinProb: number;
  awayWinProb: number;
  pickTeam: "home" | "away";
  pickProb: number;
  isUpset?: boolean;
  isCorrect?: boolean;
  edge?: number;
  fairAwayOdds?: number;
  fairHomeOdds?: number;
  shap?: ShapContribution[];
  homeInjuries?: number;
  awayInjuries?: number;
  season?: string;
  weather?: GameWeather;
  lineups?: LineupData;
  lineupStats?: LineupStats;
  runProjection?: RunProjection;
  marketOdds?: MarketOdds;
}

/** Ensemble stacking weight for one candidate model. */
export interface StackingWeight {
  name: string;
  weight: number; // 0..1 share of the final ensemble (sums to 1)
}

/** 5-fold cross-validation diagnostics for the final model family. */
export interface CrossValidationResult {
  folds: number;
  aucMean: number;
  aucStd: number;
  brierMean: number;
  brierStd: number;
  foldAucs: number[];
  foldBriers: number[];
  gamesPerFold: number[];
}

/** Hyperparameters and search grids used by the Auto-ML optimizer. */
export interface OptimizationParams {
  learningRate: number;
  l2Lambda: number;
  epochs: number;
  hfaGrid: number[];
  blendStep: number;
  mcSigmaGrid: number[];
  cvFolds: number;
  isotonicMethod: string;
  featureSelection: string;
}

/**
 * Mapping from win probability to expected run margin, used to reconcile the
 * run-scoring model's predicted scores with the win-probability model.
 * `margin = intercept + slope * logit(homeWinProb)`.
 */
export interface RunMarginCalibration {
  slope: number;
  intercept: number;
}

/** Precomputed full-range calibration metrics (stored on the model state). */
export interface CalibrationSummary {
  metrics: {
    auc: number;
    brier: number;
    logLoss: number;
    ece: number;
    bins: CalibrationBin[];
    confidenceDistribution: ConfidencePoint[];
    calibrationCurve: CurvePoint[];
  };
  totalsMetrics: { n: number; mae: number; rmse: number; bias: number };
  runLineMetrics: { n: number; auc: number; brier: number; accuracy: number };
  total: number;
  correct: number;
  accuracy: number;
}
