// Shared pure-TS types for the MLB prediction engine.
// Used by the Convex backend (actions) and the frontend. No node/convex imports.

export interface PitcherInfo {
  id: number;
  name: string;
  era?: number;
  k9?: number;
}

export interface TeamInfo {
  id: number;
  abbrev: string;
  name: string;
  score?: number;
  wins?: number;
  losses?: number;
}

export interface ShapContribution {
  feature: string;
  label: string;
  value: number;
  contribution: number;
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
}

/** The feature values used by the model for a single game. */
export interface FeatureValues {
  eloDiff: number; // (homeElo - awayElo) / 100
  winPctDiff: number; // home win% - away win%
  formDiff: number; // home last-10 win% - away last-10 win%
  restDiff: number; // home rest days - away rest days
  injuryDiff: number; // away injured-list count - home injured-list count (positive favors home)
  homeField: number; // always 1
}

export const FEATURE_KEYS = [
  "eloDiff",
  "winPctDiff",
  "formDiff",
  "restDiff",
  "injuryDiff",
  "homeField",
] as const;

export type FeatureKey = (typeof FEATURE_KEYS)[number];

export const FEATURE_LABELS: Record<FeatureKey, string> = {
  eloDiff: "Elo rating edge",
  winPctDiff: "Win % edge",
  formDiff: "Recent form (L10)",
  restDiff: "Rest advantage",
  injuryDiff: "Injury edge (IL)",
  homeField: "Home field",
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
  weight: number; // standardized logistic coefficient
  importance: number; // |weight|
  univariateAuc: number;
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
}
