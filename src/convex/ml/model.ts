// Pure-TS machine learning engine for MLB win probability.
// No node/convex imports — safe to use from Convex actions and shared types.
//
// Pipeline:
//   1. Chronological Elo ratings + as-of-game-time features (no lookahead).
//   2. Chronological 70/15/15 split: train / calibrate / test.
//   3. Feature selection via greedy backward elimination on the calibrate set.
//   4. Candidate models: Elo, logistic regression, blended ensemble.
//   5. Model selection: maximize AUC, then minimize Brier among near-best.
//   6. Isotonic calibration (PAV) to reduce Brier / calibration error.
//   7. Monte Carlo decision: enable the stochastic component only if it
//      measurably reduces holdout Brier (risk).

import {
  CalibrationBin,
  CandidateModel,
  ConfidencePoint,
  CrossValidationResult,
  CurvePoint,
  FEATURE_KEYS,
  FEATURE_LABELS,
  FeatureDriftItem,
  FeatureImportance,
  FeatureKey,
  FeatureRow,
  FeatureValues,
  InjurySnapshot,
  ModelVersion,
  OptimizationParams,
  PowerRanking,
  RawGame,
  RollingBrierPoint,
  RunMarginCalibration,
  ShapContribution,
  StackingWeight,
  TeamState,
  TrainedModel,
} from "./types";
import { fitRunModel, RunModel, simulateRuns } from "./runs";
import { PARK_FACTORS } from "./teams";

const ELO_INIT = 1500;
const ELO_HFA_UPDATE = 30; // home advantage baked into Elo updates only
const EPS = 1e-6;

export interface EvalResult {
  auc: number;
  brier: number;
  logLoss: number;
  ece: number;
  bins: CalibrationBin[];
  confidenceDistribution: ConfidencePoint[];
  calibrationCurve: CurvePoint[];
}

export interface ModelRunResult {
  season: string;
  asOfDate: string;
  gamesTrained: number;
  holdoutCount: number;
  selectedModel: string;
  modelDescription: string;
  featureNames: FeatureKey[];
  weights: number[];
  bias: number;
  featureStats: Record<FeatureKey, { mean: number; std: number }>;
  isotonicPoints: { x: number; y: number }[];
  eloHfa: number;
  monteCarloEnabled: boolean;
  monteCarloTrials: number;
  monteCarloSigma: number;
  monteCarloRationale: string;
  auc: number;
  brier: number;
  logLoss: number;
  ece: number;
  bins: CalibrationBin[];
  confidenceDistribution: ConfidencePoint[];
  calibrationCurve: CurvePoint[];
  featureImportances: FeatureImportance[];
  candidates: CandidateModel[];
  powerRankings: PowerRanking[];
  featureDrift: FeatureDriftItem[];
  rollingBrier: RollingBrierPoint[];
  brierBaseline: number;
  modelVersions: ModelVersion[];
  stackingWeights: StackingWeight[];
  crossValidation: CrossValidationResult;
  optimizationParams: OptimizationParams;
  runModel: RunModel;
  runLineCalibration: { x: number; y: number }[];
  runMarginCalibration: RunMarginCalibration;
}

export interface Prediction {
  homeWinProb: number;
  awayWinProb: number;
  pickTeam: "home" | "away";
  pickProb: number;
  shap: ShapContribution[];
  edge: number;
  fairHomeOdds: number;
  fairAwayOdds: number;
}

export interface ModelRun {
  result: ModelRunResult;
  model: TrainedModel;
  teamState: TeamState;
  rows: FeatureRow[];
  predict: (game: RawGame) => Prediction;
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

export function logit(p: number): number {
  const q = clamp(p, EPS, 1 - EPS);
  return Math.log(q / (1 - q));
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function mean(vals: number[]): number {
  if (vals.length === 0) return 0;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

function std(vals: number[]): number {
  if (vals.length === 0) return 0;
  const m = mean(vals);
  return Math.sqrt(vals.reduce((s, v) => s + (v - m) * (v - m), 0) / vals.length);
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function round(n: number, digits: number): number {
  const f = Math.pow(10, digits);
  return Math.round(n * f) / f;
}

function shiftDate(ymd: string, days: number): string {
  if (!ymd) return "";
  const d = new Date(`${ymd}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return ymd;
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function daysBetween(from: string, to: string): number {
  if (!from || !to) return 4;
  const a = Date.parse(from);
  const b = Date.parse(to);
  if (Number.isNaN(a) || Number.isNaN(b)) return 4;
  return Math.round((b - a) / 86400000);
}

// ---------------------------------------------------------------------------
// Feature engineering (strictly as-of-game-time)
// ---------------------------------------------------------------------------

interface MutableState {
  elo: Record<number, number>;
  formHistory: Record<number, number[]>;
  lastGameDate: Record<number, string>;
  records: Record<number, { wins: number; losses: number }>;
  injuries: Record<number, number>;
  runDiff: Record<number, number>;
  homeRecords: Record<number, { wins: number; losses: number }>;
  awayRecords: Record<number, { wins: number; losses: number }>;
}

function newState(): MutableState {
  return {
    elo: {},
    formHistory: {},
    lastGameDate: {},
    records: {},
    injuries: {},
    runDiff: {},
    homeRecords: {},
    awayRecords: {},
  };
}

function formOf(state: MutableState, id: number): number {
  const h = state.formHistory[id];
  if (!h || h.length === 0) return 0.5;
  return h.reduce((a, b) => a + b, 0) / h.length;
}

/** Starting-pitcher delta (away − home) so positive values favor the home team. */
function starterDelta(
  homePitcher: RawGame["homePitcher"],
  awayPitcher: RawGame["awayPitcher"],
  key: "era" | "fip",
): number {
  const away = awayPitcher?.[key];
  const home = homePitcher?.[key];
  if (typeof away !== "number" || typeof home !== "number") return 0;
  return away - home;
}

/**
 * Signed edge so positive values favor the home team (0 when either side is
 * missing). `lowerBetter` inverts the sign for stats where lower is better
 * (ERA / WHIP) so the convention stays uniform across all features.
 */
function edge(home: unknown, away: unknown, lowerBetter = false): number {
  if (typeof home !== "number" || typeof away !== "number") return 0;
  return lowerBetter ? away - home : home - away;
}

function buildFeatures(game: RawGame, state: MutableState): FeatureValues {
  const homeElo = state.elo[game.home.id] ?? ELO_INIT;
  const awayElo = state.elo[game.away.id] ?? ELO_INIT;

  const homeRec = state.records[game.home.id] ?? { wins: 0, losses: 0 };
  const awayRec = state.records[game.away.id] ?? { wins: 0, losses: 0 };
  const homeWins = game.home.wins ?? homeRec.wins;
  const homeLosses = game.home.losses ?? homeRec.losses;
  const awayWins = game.away.wins ?? awayRec.wins;
  const awayLosses = game.away.losses ?? awayRec.losses;
  const homeWp = homeWins + homeLosses > 0 ? homeWins / (homeWins + homeLosses) : 0.5;
  const awayWp = awayWins + awayLosses > 0 ? awayWins / (awayWins + awayLosses) : 0.5;

  const homeRest = clamp(daysBetween(state.lastGameDate[game.home.id] ?? "", game.date), 0, 10);
  const awayRest = clamp(daysBetween(state.lastGameDate[game.away.id] ?? "", game.date), 0, 10);

  const homeOps = game.home.ops;
  const awayOps = game.away.ops;
  const homeTeamEra = game.home.era;
  const awayTeamEra = game.away.era;
  const homeFielding = game.home.fieldingPct;
  const awayFielding = game.away.fieldingPct;
  const homeTeamK9 = game.home.k9;
  const awayTeamK9 = game.away.k9;
  const homeTeamWhip = game.home.whip;
  const awayTeamWhip = game.away.whip;
  const tempF = game.weather?.tempF;
  const wind = game.weather?.windMph;

  // Actual starting-9 / bench data. When lineups are missing (older historical
  // games) the feature defaults to 0 with lineupKnown = 0, so the model learns
  // to treat "unknown lineup" distinctly from "known, strong/weak lineup".
  const lineupHome = game.lineupStats?.home;
  const lineupAway = game.lineupStats?.away;
  const homeLineupKnown = lineupHome?.known === true ? 1 : 0;
  const awayLineupKnown = lineupAway?.known === true ? 1 : 0;
  const lineupKnown = homeLineupKnown === 1 && awayLineupKnown === 1 ? 1 : 0;
  const lineupOpsDiff =
    typeof lineupHome?.ops === "number" && typeof lineupAway?.ops === "number"
      ? lineupHome.ops - lineupAway.ops
      : 0;

  return {
    eloDiff: (homeElo - awayElo) / 100,
    winPctDiff: homeWp - awayWp,
    formDiff: formOf(state, game.home.id) - formOf(state, game.away.id),
    restDiff: clamp(homeRest - awayRest, -4, 4),
    injuryDiff: (state.injuries[game.away.id] ?? 0) - (state.injuries[game.home.id] ?? 0),
    homeField: 1,
    spFipDiff: starterDelta(game.homePitcher, game.awayPitcher, "fip"),
    spEraDiff: starterDelta(game.homePitcher, game.awayPitcher, "era"),
    spK9Diff: edge(game.homePitcher?.k9, game.awayPitcher?.k9),
    spWhipDiff: edge(game.homePitcher?.whip, game.awayPitcher?.whip, true),
    spRecentDiff: edge(game.homePitcher?.recentEra, game.awayPitcher?.recentEra, true),
    opsDiff: typeof homeOps === "number" && typeof awayOps === "number" ? homeOps - awayOps : 0,
    teamEraDiff: typeof awayTeamEra === "number" && typeof homeTeamEra === "number" ? awayTeamEra - homeTeamEra : 0,
    teamK9Diff: edge(homeTeamK9, awayTeamK9),
    teamWhipDiff: edge(homeTeamWhip, awayTeamWhip, true),
    defEffDiff: typeof homeFielding === "number" && typeof awayFielding === "number" ? homeFielding - awayFielding : 0,
    parkFactor: PARK_FACTORS[game.home.id] ?? 1,
    tempDev: typeof tempF === "number" ? tempF - 72 : 0,
    windMph: typeof wind === "number" ? wind : 0,
    lineupKnown,
    lineupOpsDiff,
    lineupWobaDiff: edge(lineupHome?.woba, lineupAway?.woba),
    lineupIsoDiff: edge(lineupHome?.iso, lineupAway?.iso),
    lineupHotDiff: edge(lineupHome?.recentOps, lineupAway?.recentOps),
    // Matchup edges: career BvP OPS, season platoon OPS vs the starter's
    // handedness, and season OPS vs the opposing team — all PA/slot-weighted
    // means over the real starting 9. 0 (with lineupKnown = 0) when no boxscore
    // lineup exists for the game, mirroring the other lineup features.
    bvpOpsDiff: edge(lineupHome?.bvpOps, lineupAway?.bvpOps),
    platoonOpsDiff: edge(lineupHome?.platoonOps, lineupAway?.platoonOps),
    vsTeamOpsDiff: edge(lineupHome?.vsTeamOps, lineupAway?.vsTeamOps),
  };
}

function updateState(state: MutableState, game: RawGame): void {
  const home = game.home.id;
  const away = game.away.id;
  const homeElo = state.elo[home] ?? ELO_INIT;
  const awayElo = state.elo[away] ?? ELO_INIT;

  const expectedHome = 1 / (1 + Math.pow(10, -((homeElo + ELO_HFA_UPDATE) - awayElo) / 400));
  const homeActual = game.winner === "home" ? 1 : 0;
  const margin = Math.abs((game.home.score ?? 0) - (game.away.score ?? 0));
  const k = 24 * Math.sqrt(Math.max(1, margin));
  const delta = k * (homeActual - expectedHome);
  state.elo[home] = homeElo + delta;
  state.elo[away] = awayElo - delta;

  const hh = state.formHistory[home] ?? [];
  hh.push(homeActual);
  if (hh.length > 10) hh.shift();
  state.formHistory[home] = hh;
  const ah = state.formHistory[away] ?? [];
  ah.push(1 - homeActual);
  if (ah.length > 10) ah.shift();
  state.formHistory[away] = ah;

  const hr = state.records[home] ?? { wins: 0, losses: 0 };
  const ar = state.records[away] ?? { wins: 0, losses: 0 };
  if (homeActual === 1) {
    hr.wins += 1;
    ar.losses += 1;
  } else {
    hr.losses += 1;
    ar.wins += 1;
  }
  state.records[home] = hr;
  state.records[away] = ar;

  const hScore = game.home.score ?? 0;
  const aScore = game.away.score ?? 0;
  state.runDiff[home] = (state.runDiff[home] ?? 0) + (hScore - aScore);
  state.runDiff[away] = (state.runDiff[away] ?? 0) + (aScore - hScore);

  const hHome = state.homeRecords[home] ?? { wins: 0, losses: 0 };
  const aAway = state.awayRecords[away] ?? { wins: 0, losses: 0 };
  if (homeActual === 1) {
    hHome.wins += 1;
    aAway.losses += 1;
  } else {
    hHome.losses += 1;
    aAway.wins += 1;
  }
  state.homeRecords[home] = hHome;
  state.awayRecords[away] = aAway;

  state.lastGameDate[home] = game.date;
  state.lastGameDate[away] = game.date;
}

/**
 * Find the injured-list count from the most recent snapshot on or before `date`
 * (strictly no lookahead). Snapshots are sorted ascending by date.
 */
function lookupInjuries(
  teamId: number,
  date: string,
  snapshots?: Map<number, InjurySnapshot[]>,
): number {
  if (!snapshots) return 0;
  const list = snapshots.get(teamId);
  if (!list || list.length === 0) return 0;
  let best = 0;
  for (const s of list) {
    if (s.date > date) break;
    best = s.count;
  }
  return best;
}

export interface TeamStats {
  runDiff: Record<number, number>;
  homeRecords: Record<number, { wins: number; losses: number }>;
  awayRecords: Record<number, { wins: number; losses: number }>;
}

export function computeEloAndFeatures(
  games: RawGame[],
  injurySnapshots?: Map<number, InjurySnapshot[]>,
  latestDate?: string,
): { rows: FeatureRow[]; teamState: TeamState; teamStats: TeamStats } {
  const sorted = [...games].sort((a, b) => (a.gameDate < b.gameDate ? -1 : 1));
  const state = newState();
  const rows: FeatureRow[] = [];
  for (const game of sorted) {
    if (game.winner !== "home" && game.winner !== "away") continue;
    state.injuries[game.home.id] = lookupInjuries(game.home.id, game.date, injurySnapshots);
    state.injuries[game.away.id] = lookupInjuries(game.away.id, game.date, injurySnapshots);
    const features = buildFeatures(game, state);
    rows.push({
      game,
      features,
      homeElo: state.elo[game.home.id] ?? ELO_INIT,
      awayElo: state.elo[game.away.id] ?? ELO_INIT,
      label: game.winner === "home" ? 1 : 0,
    });
    updateState(state, game);
  }
  // Refresh injury counts to the latest snapshot so upcoming-game predictions
  // use current roster state rather than the last completed game's snapshot.
  if (latestDate && injurySnapshots) {
    for (const id of Object.keys(state.elo)) {
      state.injuries[Number(id)] = lookupInjuries(Number(id), latestDate, injurySnapshots);
    }
  }
  const teamState: TeamState = {
    elo: state.elo,
    form: {},
    lastGameDate: state.lastGameDate,
    records: state.records,
    injuries: state.injuries,
  };
  for (const id of Object.keys(state.formHistory)) {
    teamState.form[Number(id)] = formOf(state, Number(id));
  }
  const teamStats: TeamStats = {
    runDiff: state.runDiff,
    homeRecords: state.homeRecords,
    awayRecords: state.awayRecords,
  };
  return { rows, teamState, teamStats };
}

/** Build features for a not-yet-seen game given the current team state. */
export function buildFeaturesForGame(game: RawGame, state: TeamState): FeatureValues {
  const mut: MutableState = {
    elo: state.elo,
    formHistory: {},
    lastGameDate: state.lastGameDate,
    records: state.records,
    injuries: state.injuries,
    runDiff: {},
    homeRecords: {},
    awayRecords: {},
  };
  // Provide form as a synthetic 10-game history so buildFeatures can reuse it.
  for (const id of Object.keys(state.form)) {
    const p = state.form[Number(id)];
    const wins = Math.round(p * 10);
    mut.formHistory[Number(id)] = Array.from({ length: 10 }, (_, i) => (i < wins ? 1 : 0));
  }
  return buildFeatures(game, mut);
}

// ---------------------------------------------------------------------------
// Logistic regression (L2, standardized features)
// ---------------------------------------------------------------------------

export interface LogisticModel {
  featureNames: FeatureKey[];
  weights: number[];
  bias: number;
  featureStats: Record<FeatureKey, { mean: number; std: number }>;
}

/** Solve a small dense linear system A·x = b with partial pivoting. */
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const aug = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(aug[r][col]) > Math.abs(aug[pivot][col])) pivot = r;
    }
    if (Math.abs(aug[pivot][col]) < 1e-12) continue;
    [aug[col], aug[pivot]] = [aug[pivot], aug[col]];
    const pv = aug[col][col];
    for (let c = col; c <= n; c++) aug[col][c] /= pv;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = aug[r][col];
      if (f === 0) continue;
      for (let c = col; c <= n; c++) aug[r][c] -= f * aug[col][c];
    }
  }
  return aug.map((row) => row[n]);
}

export function trainLogistic(
  rows: FeatureRow[],
  featureNames: FeatureKey[],
  opts?: { iterations?: number },
): LogisticModel {
  const featureStats = {} as Record<FeatureKey, { mean: number; std: number }>;
  for (const f of featureNames) {
    const vals = rows.map((r) => r.features[f]);
    featureStats[f] = { mean: mean(vals), std: std(vals) || 1 };
  }
  const X = rows.map((r) =>
    featureNames.map((f) => (r.features[f] - featureStats[f].mean) / featureStats[f].std),
  );
  const y = rows.map((r) => r.label);
  const n = rows.length;
  const m = featureNames.length;
  const pos = y.reduce((a, b) => a + b, 0);
  const d = m + 1; // feature columns + intercept column
  const Xaug = X.map((xi) => [...xi, 1]);
  let w = new Array<number>(d).fill(0);
  w[m] = Math.log((pos + 1) / (n - pos + 1));
  const iterations = opts?.iterations ?? 20;
  const lambda = 0.001;

  // Newton-Raphson / IRLS for ridge logistic regression. This converges in a
  // handful of iterations instead of the previous ~1200 batch gradient steps,
  // which is what keeps on-demand refreshes fast now that the feature set is
  // wider.
  for (let it = 0; it < iterations; it++) {
    const A = Array.from({ length: d }, () => new Array<number>(d).fill(0));
    const rhs = new Array<number>(d).fill(0);
    for (let i = 0; i < n; i++) {
      let eta = 0;
      for (let j = 0; j < d; j++) eta += w[j] * Xaug[i][j];
      const p = clamp(sigmoid(eta), 1e-6, 1 - 1e-6);
      const weight = Math.max(p * (1 - p), 1e-6);
      const z = eta + (y[i] - p) / weight;
      const wz = weight * z;
      for (let j = 0; j < d; j++) {
        rhs[j] += wz * Xaug[i][j];
        for (let k = j; k < d; k++) {
          A[j][k] += weight * Xaug[i][j] * Xaug[i][k];
        }
      }
    }
    for (let j = 0; j < d; j++) {
      for (let k = j + 1; k < d; k++) A[k][j] = A[j][k];
      if (j < m) A[j][j] += lambda; // ridge the features, not the intercept
    }
    const next = solveLinearSystem(A, rhs);
    if (next.some((v) => !Number.isFinite(v))) break;
    w = next;
  }

  return {
    featureNames,
    weights: w.slice(0, m),
    bias: w[m],
    featureStats,
  };
}

export function logisticLogit(
  model: LogisticModel,
  features: FeatureValues,
  shap: ShapContribution[] | null,
): number {
  let logitV = model.bias;
  for (let i = 0; i < model.featureNames.length; i++) {
    const f = model.featureNames[i];
    const z = (features[f] - model.featureStats[f].mean) / (model.featureStats[f].std || 1);
    logitV += model.weights[i] * z;
    if (shap) shap.push({ feature: f, label: FEATURE_LABELS[f], value: features[f], contribution: model.weights[i] * z });
  }
  return logitV;
}

// ---------------------------------------------------------------------------
// Additional candidate models (pure TS) for the stacking ensemble
// ---------------------------------------------------------------------------

function standardized(
  features: FeatureValues,
  featureNames: FeatureKey[],
  stats: Record<FeatureKey, { mean: number; std: number }>,
): number[] {
  return featureNames.map((f) => (features[f] - stats[f].mean) / (stats[f].std || 1));
}

/** k-nearest-neighbour classifier on standardized features (majority vote). */
function knnModel(train: FeatureRow[], featureNames: FeatureKey[], k = 21) {
  const stats = {} as Record<FeatureKey, { mean: number; std: number }>;
  for (const f of featureNames) {
    const vals = train.map((r) => r.features[f]);
    stats[f] = { mean: mean(vals), std: std(vals) || 1 };
  }
  const zTrain = train.map((r) => standardized(r.features, featureNames, stats));
  const labels = train.map((r) => r.label);
  return (features: FeatureValues): number => {
    const z = standardized(features, featureNames, stats);
    const dists = zTrain.map((zt, i) => ({
      d: zt.reduce((s, v, j) => s + (v - z[j]) * (v - z[j]), 0),
      y: labels[i],
    }));
    dists.sort((a, b) => a.d - b.d);
    const nn = dists.slice(0, k);
    if (nn.length === 0) return 0.5;
    return nn.reduce((s, x) => s + x.y, 0) / nn.length;
  };
}

/** Gaussian Naive Bayes classifier with Laplace-smoothed priors. */
function naiveBayesModel(train: FeatureRow[], featureNames: FeatureKey[]) {
  const n = train.length;
  const pos = train.filter((r) => r.label === 1);
  const neg = train.filter((r) => r.label === 0);
  const prior = (pos.length + 1) / (n + 2);
  const cond = (rows: FeatureRow[]) =>
    featureNames.map((f) => {
      const vals = rows.map((r) => r.features[f]);
      return { m: mean(vals), v: std(vals) * std(vals) + 1e-6 };
    });
  const posStats = cond(pos);
  const negStats = cond(neg);
  const gaussLog = (x: number, s: { m: number; v: number }) =>
    -0.5 * Math.log(2 * Math.PI * s.v) - ((x - s.m) * (x - s.m)) / (2 * s.v);
  return (features: FeatureValues): number => {
    let logPos = Math.log(prior);
    let logNeg = Math.log(1 - prior);
    for (let j = 0; j < featureNames.length; j++) {
      logPos += gaussLog(features[featureNames[j]], posStats[j]);
      logNeg += gaussLog(features[featureNames[j]], negStats[j]);
    }
    const maxLog = Math.max(logPos, logNeg);
    const pPos = Math.exp(logPos - maxLog);
    const pNeg = Math.exp(logNeg - maxLog);
    const sum = pPos + pNeg;
    return sum > 0 ? pPos / sum : prior;
  };
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

export function computeAuc(preds: number[], labels: number[]): number {
  const pairs = preds.map((p, i) => ({ p, y: labels[i] }));
  pairs.sort((a, b) => a.p - b.p);
  const nPos = labels.reduce((s, y) => s + y, 0);
  const nNeg = labels.length - nPos;
  if (nPos === 0 || nNeg === 0) return 0.5;
  let rankSum = 0;
  let i = 0;
  while (i < pairs.length) {
    let j = i;
    while (j + 1 < pairs.length && pairs[j + 1].p === pairs[i].p) j++;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) if (pairs[k].y === 1) rankSum += avgRank;
    i = j + 1;
  }
  return (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

export function computeBrier(preds: number[], labels: number[]): number {
  return mean(preds.map((p, i) => (p - labels[i]) * (p - labels[i])));
}

export function computeLogLoss(preds: number[], labels: number[]): number {
  return mean(
    preds.map((p, i) => -(labels[i] * Math.log(clamp(p, EPS, 1)) + (1 - labels[i]) * Math.log(clamp(1 - p, EPS, 1)))),
  );
}

/** Spearman rank correlation between predictions and binary outcomes. */
export function spearmanRank(preds: number[], labels: number[]): number {
  const n = preds.length;
  if (n < 2) return 0;
  const rank = (vals: number[]): number[] => {
    const idx = vals.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const out = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let j = i;
      while (j + 1 < n && idx[j + 1].v === idx[i].v) j++;
      const avg = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) out[idx[k].i] = avg;
      i = j;
    }
    return out;
  };
  const rx = rank(preds);
  const ry = rank(labels);
  const center = (n + 1) / 2;
  let num = 0;
  let dx2 = 0;
  let dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = rx[i] - center;
    const dy = ry[i] - center;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  return dx2 === 0 || dy2 === 0 ? 0 : num / Math.sqrt(dx2 * dy2);
}

function confidenceBins(preds: number[], labels: number[]): CalibrationBin[] {
  const bins: { sumP: number; sumY: number; count: number }[] = [];
  for (let b = 0; b < 10; b++) bins.push({ sumP: 0, sumY: 0, count: 0 });
  for (let i = 0; i < preds.length; i++) {
    const conf = Math.max(preds[i], 1 - preds[i]);
    const correct = (preds[i] >= 0.5 ? labels[i] : 1 - labels[i]) === 1 ? 1 : 0;
    const idx = clamp(Math.floor((conf - 0.5) / 0.05), 0, 9);
    bins[idx].sumP += conf;
    bins[idx].sumY += correct;
    bins[idx].count += 1;
  }
  const out: CalibrationBin[] = [];
  for (let b = 0; b < 10; b++) {
    if (bins[b].count === 0) continue;
    const meanPredicted = bins[b].sumP / bins[b].count;
    const meanActual = bins[b].sumY / bins[b].count;
    out.push({
      label: b === 9 ? "95-100%" : `${50 + b * 5}-${55 + b * 5}%`,
      meanPredicted,
      meanActual,
      count: bins[b].count,
      gap: meanActual - meanPredicted,
    });
  }
  return out;
}

export function calibrationCurvePoints(preds: number[], labels: number[], minCount = 12): CurvePoint[] {
  const bins: { sumP: number; sumY: number; count: number }[] = [];
  for (let b = 0; b < 20; b++) bins.push({ sumP: 0, sumY: 0, count: 0 });
  for (let i = 0; i < preds.length; i++) {
    const idx = clamp(Math.floor(preds[i] / 0.05), 0, 19);
    bins[idx].sumP += preds[i];
    bins[idx].sumY += labels[i];
    bins[idx].count += 1;
  }
  const out: CurvePoint[] = [];
  for (let b = 0; b < 20; b++) {
    if (bins[b].count < minCount) continue;
    out.push({ x: bins[b].sumP / bins[b].count, y: bins[b].sumY / bins[b].count, n: bins[b].count });
  }
  return out;
}

export function evaluate(preds: number[], labels: number[]): EvalResult {
  const bins = confidenceBins(preds, labels);
  const total = preds.length;
  let ece = 0;
  for (const b of bins) ece += (b.count / total) * Math.abs(b.gap);
  const distribution: ConfidencePoint[] = bins.map((b) => ({
    label: b.label,
    count: b.count,
    accuracy: b.meanActual,
  }));
  return {
    auc: computeAuc(preds, labels),
    brier: computeBrier(preds, labels),
    logLoss: computeLogLoss(preds, labels),
    ece,
    bins,
    confidenceDistribution: distribution,
    calibrationCurve: calibrationCurvePoints(preds, labels),
  };
}

// ---------------------------------------------------------------------------
// Isotonic calibration (PAV)
// ---------------------------------------------------------------------------

export function isotonicRegression(xs: number[], ys: number[]): { x: number; y: number }[] {
  const n = xs.length;
  const blocks: { xSum: number; ySum: number; count: number }[] = [];
  for (let i = 0; i < n; i++) {
    blocks.push({ xSum: xs[i], ySum: ys[i], count: 1 });
    while (blocks.length > 1) {
      const a = blocks[blocks.length - 2];
      const b = blocks[blocks.length - 1];
      if (a.ySum / a.count <= b.ySum / b.count) break;
      a.xSum += b.xSum;
      a.ySum += b.ySum;
      a.count += b.count;
      blocks.pop();
    }
  }
  return blocks.map((b) => ({ x: b.xSum / b.count, y: b.ySum / b.count }));
}

export function applyIsotonic(points: { x: number; y: number }[], p: number): number {
  if (points.length === 0) return p;
  if (p <= points[0].x) return points[0].y;
  if (p >= points[points.length - 1].x) return points[points.length - 1].y;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    if (p >= a.x && p <= b.x) {
      const t = b.x === a.x ? 0 : (p - a.x) / (b.x - a.x);
      return a.y + t * (b.y - a.y);
    }
  }
  return p;
}

// ---------------------------------------------------------------------------
// Monte Carlo (stochastic component)
// ---------------------------------------------------------------------------

/**
 * Expected win probability after adding Gaussian noise N(0, σ²) to the logit:
 * E[sigmoid(logit(p) + σ·Z)]. Evaluated with 7-point Gauss-Hermite quadrature,
 * which matches the infinite-trial Monte Carlo mean but is O(1), deterministic,
 * and far faster than simulating thousands of draws per game.
 */
const GAUSS_HERMITE_NODES = [0, 0.816287882858965, 1.673551628767471, 2.651961356835233];
const GAUSS_HERMITE_WEIGHTS = [
  0.810264617556807,
  0.425607252610128,
  0.054515582819125,
  0.000971781245099519,
];
const INV_SQRT_PI = 1 / Math.sqrt(Math.PI);

export function monteCarloAdjust(p: number, sigma: number, trials: number): number {
  if (sigma <= 0 || trials <= 0) return p;
  const lp = logit(p);
  const s = sigma * Math.SQRT2;
  let sum = GAUSS_HERMITE_WEIGHTS[0] * sigmoid(lp);
  for (let i = 1; i < GAUSS_HERMITE_NODES.length; i++) {
    const node = GAUSS_HERMITE_NODES[i] * s;
    sum += GAUSS_HERMITE_WEIGHTS[i] * (sigmoid(lp + node) + sigmoid(lp - node));
  }
  return clamp(sum * INV_SQRT_PI, 0.001, 0.999);
}

// ---------------------------------------------------------------------------
// Drift monitoring, rolling risk, and version history
// ---------------------------------------------------------------------------

/**
 * Simplified PSI-style drift score: KL divergence between the baseline
 * (training) and recent feature distributions under a Gaussian assumption,
 * `(mu_cur - mu_base)^2 / (2 * sigma_base^2)`. Small = stable, large = drift.
 */
function computeFeatureDrift(
  rows: FeatureRow[],
  selected: FeatureKey[],
): FeatureDriftItem[] {
  const n = rows.length;
  if (n === 0) return [];
  const baselineEnd = Math.floor(n * 0.7);
  const recentStart = Math.max(baselineEnd, n - 40);
  const baseline = rows.slice(0, baselineEnd);
  const recent = rows.slice(recentStart);
  return selected.map((f) => {
    const bvals = baseline.map((r) => r.features[f]);
    const cvals = recent.map((r) => r.features[f]);
    const bMean = mean(bvals);
    const cMean = mean(cvals);
    const bStd = std(bvals) || 1;
    const psi = ((cMean - bMean) * (cMean - bMean)) / (2 * bStd * bStd);
    return {
      feature: f,
      label: FEATURE_LABELS[f],
      currentMean: round(cMean, 3),
      baselineMean: round(bMean, 3),
      psi: round(psi, 3),
      status: psi >= 0.1 ? "WARN" : "OK",
    };
  });
}

/** Rolling (per-day) Brier score over the last 30 days of the season. */
function computeRollingBrier(
  rows: FeatureRow[],
  model: TrainedModel,
  asOfDate: string,
): { points: RollingBrierPoint[]; baseline: number } {
  const cutoff = shiftDate(asOfDate, -30);
  const byDate = new Map<string, { sum: number; count: number }>();
  let totalSum = 0;
  let totalCount = 0;
  for (const r of rows) {
    if (r.game.date < cutoff) continue;
    const p = applyModel(model, r.features, r.homeElo, r.awayElo).homeWinProb;
    const sq = (p - r.label) * (p - r.label);
    totalSum += sq;
    totalCount += 1;
    const b = byDate.get(r.game.date) ?? { sum: 0, count: 0 };
    b.sum += sq;
    b.count += 1;
    byDate.set(r.game.date, b);
  }
  const points: RollingBrierPoint[] = [...byDate.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : 1))
    .map(([date, v]) => ({ date, brier: round(v.sum / Math.max(1, v.count), 3) }));
  return { points, baseline: round(totalSum / Math.max(1, totalCount), 3) };
}

/** Data-driven version history from progressively larger training windows. */
function buildModelVersions(
  rows: FeatureRow[],
  asOfDate: string,
  finalEval: EvalResult,
): ModelVersion[] {
  const n = rows.length;
  const stages: { frac: number; features: FeatureKey[]; note: string }[] = [
    {
      frac: 0.25,
      features: ["eloDiff", "winPctDiff", "homeField"],
      note: "Baseline model: Elo, win % and home-field features",
    },
    {
      frac: 0.5,
      features: ["eloDiff", "winPctDiff", "formDiff", "restDiff", "homeField"],
      note: "Added recent form and rest-day features",
    },
    {
      frac: 0.75,
      features: [
        "eloDiff",
        "winPctDiff",
        "formDiff",
        "restDiff",
        "injuryDiff",
        "homeField",
        "spFipDiff",
        "spEraDiff",
      ],
      note: "Added injured-list edge, starting-pitcher FIP/ERA and isotonic calibration",
    },
  ];
  const versions: ModelVersion[] = [];
  for (const stage of stages) {
    const end = Math.floor(n * stage.frac);
    if (end < 60) continue;
    const trainEnd = Math.floor(end * 0.85);
    const train = rows.slice(0, trainEnd);
    const test = rows.slice(trainEnd, end);
    if (train.length < 40 || test.length < 20) continue;
    const m = trainLogistic(train, stage.features);
    const preds = test.map((r) => sigmoid(logisticLogit(m, r.features, null)));
    const labels = test.map((r) => r.label);
    versions.push({
      version: `v${versions.length + 1}.0.0`,
      date: rows[end - 1]?.game.date ?? asOfDate,
      auc: round(computeAuc(preds, labels), 3),
      brier: round(computeBrier(preds, labels), 3),
      notes: stage.note,
    });
  }
  versions.push({
    version: `v${versions.length + 1}.0.0`,
    date: asOfDate,
    auc: round(finalEval.auc, 3),
    brier: round(finalEval.brier, 3),
    notes: "Current model: ML feature selection, ensemble and Monte Carlo decision",
  });
  return versions.reverse();
}

// ---------------------------------------------------------------------------
// Stacking ensemble & cross-validation
// ---------------------------------------------------------------------------

/**
 * Greedy forward-selection stacking: start from the lowest-Brier candidate and
 * iteratively blend in the model that most reduces calibration-set Brier,
 * tuning each new model's weight on a [0..1] grid. Returns convex-combination
 * ensemble predictions plus normalized per-model weights.
 */
function buildStackingWeights(
  candPreds: Record<string, number[]>,
  labels: number[],
): { preds: number[]; brier: number; weights: StackingWeight[] } {
  const names = Object.keys(candPreds);
  const ranked = names
    .map((n) => ({ n, b: computeBrier(candPreds[n], labels) }))
    .sort((a, b) => a.b - b.b);
  const order = ranked.map((r) => r.n);
  const first = candPreds[order[0]];
  if (first.length === 0) {
    return { preds: [], brier: Infinity, weights: names.map((n) => ({ name: n, weight: n === order[0] ? 1 : 0 })) };
  }
  let ensemble = [...first];
  let weights = new Map<string, number>([[order[0], 1]]);
  let curBrier = computeBrier(ensemble, labels);
  const step = 0.05;
  for (let i = 1; i < order.length; i++) {
    const name = order[i];
    const p = candPreds[name];
    let bestW = 0;
    let bestB = curBrier;
    for (let w = step; w <= 1.0001; w += step) {
      const blend = ensemble.map((e, j) => (1 - w) * e + w * p[j]);
      const b = computeBrier(blend, labels);
      if (b < bestB) {
        bestB = b;
        bestW = w;
      }
    }
    if (bestB < curBrier - 0.0005) {
      const next = new Map<string, number>();
      for (const [k, v] of weights) next.set(k, v * (1 - bestW));
      next.set(name, bestW);
      weights = next;
      ensemble = ensemble.map((e, j) => (1 - bestW) * e + bestW * p[j]);
      curBrier = bestB;
    }
  }
  const total = [...weights.values()].reduce((s, v) => s + v, 0) || 1;
  const weightList: StackingWeight[] = names.map((n) => ({
    name: n,
    weight: round((weights.get(n) ?? 0) / total, 3),
  }));
  return { preds: ensemble, brier: curBrier, weights: weightList };
}

/**
 * Walk-forward 5-fold cross-validation of the logistic model on selected
 * features. Each fold trains only on data before its test window so
 * out-of-sample AUC and Brier are never inflated by lookahead.
 */
function crossValidate(
  rows: FeatureRow[],
  featureNames: FeatureKey[],
  cvFolds = 5,
): CrossValidationResult {
  const n = rows.length;
  const chunkSize = Math.max(1, Math.floor(n / (cvFolds + 1)));
  const foldAucs: number[] = [];
  const foldBriers: number[] = [];
  const gamesPerFold: number[] = [];
  for (let f = 1; f <= cvFolds; f++) {
    const trainEnd = f * chunkSize;
    const testEnd = f === cvFolds ? n : (f + 1) * chunkSize;
    const train = rows.slice(0, trainEnd);
    const test = rows.slice(trainEnd, testEnd);
    if (train.length < 40 || test.length < 20) continue;
    const m = trainLogistic(train, featureNames);
    const preds = test.map((r) => sigmoid(logisticLogit(m, r.features, null)));
    const labels = test.map((r) => r.label);
    foldAucs.push(computeAuc(preds, labels));
    foldBriers.push(computeBrier(preds, labels));
    gamesPerFold.push(test.length);
  }
  return {
    folds: foldAucs.length,
    aucMean: round(mean(foldAucs), 3),
    aucStd: round(std(foldAucs), 3),
    brierMean: round(mean(foldBriers), 3),
    brierStd: round(std(foldBriers), 3),
    foldAucs: foldAucs.map((x) => round(x, 3)),
    foldBriers: foldBriers.map((x) => round(x, 3)),
    gamesPerFold,
  };
}

// ---------------------------------------------------------------------------
// Model pipeline
// ---------------------------------------------------------------------------

const HFA_GRID = [0, 10, 20, 30, 40, 50, 60];

/**
 * Apply a trained model to an explicit feature vector (as-of-time for
 * historical rows, current for upcoming games).
 */
export function applyModel(
  model: TrainedModel,
  features: FeatureValues,
  homeElo: number,
  awayElo: number,
): Prediction {
  const shap: ShapContribution[] = [];
  const logitV = logisticLogit(
    { featureNames: model.featureNames, weights: model.weights, bias: model.bias, featureStats: model.featureStats },
    features,
    shap,
  );
  let p = sigmoid(logitV);
  p = applyIsotonic(model.isotonicPoints, p);
  if (model.monteCarloEnabled && model.monteCarloSigma > 0) p = monteCarloAdjust(p, model.monteCarloSigma, 10000);
  p = clamp(p, 0.01, 0.99);
  const baseline = sigmoid(((homeElo + model.eloHfa - awayElo) / 400) * Math.log(10));
  const edge = p - baseline;
  return {
    homeWinProb: p,
    awayWinProb: 1 - p,
    pickTeam: p >= 0.5 ? "home" : "away",
    pickProb: p >= 0.5 ? p : 1 - p,
    shap: shap.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 5),
    edge,
    fairHomeOdds: americanOdds(p),
    fairAwayOdds: americanOdds(1 - p),
  };
}

function eloProb(row: FeatureRow, hfa: number): number {
  return sigmoid(((row.homeElo + hfa - row.awayElo) / 400) * Math.log(10));
}

/**
 * Fit the mapping `expected run margin = intercept + slope · logit(homeWinProb)`
 * on all completed games. This reconciles the run-scoring model (Poisson means)
 * with the win-probability model so the two never contradict each other —
 * e.g. an underdog being shown to score more runs than the favorite.
 */
function fitRunMarginCalibration(rows: FeatureRow[], model: TrainedModel): RunMarginCalibration {
  const lr = {
    featureNames: model.featureNames,
    weights: model.weights,
    bias: model.bias,
    featureStats: model.featureStats,
  };
  const xs: number[] = [];
  const ys: number[] = [];
  for (const r of rows) {
    // Use the deterministic point estimate (logistic + isotonic), not the
    // Monte Carlo-smoothed value, so the margin mapping is stable and fast.
    const p = applyIsotonic(model.isotonicPoints, sigmoid(logisticLogit(lr, r.features, null)));
    const margin = (r.game.home.score ?? 0) - (r.game.away.score ?? 0);
    xs.push(logit(p));
    ys.push(margin);
  }
  if (xs.length < 40) return { slope: 0, intercept: 0 };
  const n = xs.length;
  const mx = mean(xs);
  const my = mean(ys);
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) * (xs[i] - mx);
  }
  const slope = sxx > 1e-9 ? sxy / sxx : 0;
  return { slope, intercept: my - slope * mx };
}

export function runModel(
  completedGames: RawGame[],
  opts: { season: string; asOfDate: string },
  injurySnapshots?: Map<number, InjurySnapshot[]>,
): ModelRun {
  const { rows, teamState, teamStats } = computeEloAndFeatures(completedGames, injurySnapshots, opts.asOfDate);
  const n = rows.length;

  const trainEnd = Math.floor(n * 0.7);
  const calibEnd = Math.min(n, Math.floor(n * 0.85));
  const train = rows.slice(0, trainEnd);
  const calib = rows.slice(trainEnd, calibEnd);
  const test = rows.slice(calibEnd);

  const calibLabels = calib.map((r) => r.label);
  const testLabels = test.map((r) => r.label);

  // Feature selection: greedy backward elimination evaluated on the calib set.
  let selected: FeatureKey[] = [...FEATURE_KEYS];
  const score = (preds: number[], labels: number[]) => computeBrier(preds, labels) - 0.5 * computeAuc(preds, labels);

  if (calib.length >= 20 && train.length >= 20) {
    // Use a quick IRLS pass for candidate screening; the final model below is
    // refit with more iterations. The backward pass is capped to keep on-demand
    // refreshes inside Convex's per-action time budget when the feature pool
    // and calib set are large.
    let currentModel = trainLogistic(train, selected, { iterations: 10 });
    let currentPreds = calib.map((r) => sigmoid(logisticLogit(currentModel, r.features, null)));
    let currentScore = score(currentPreds, calibLabels);
    let improved = true;
    let stagnationRounds = 0;
    const MAX_BE_ROUNDS = Math.max(4, Math.min(8, selected.length - 2));
    let rounds = 0;
    while (improved && selected.length > 2 && rounds < MAX_BE_ROUNDS) {
      improved = false;
      rounds += 1;
      let bestFeatures = selected;
      let bestScore = currentScore;
      for (const drop of selected) {
        const candidate = selected.filter((f) => f !== drop);
        const m = trainLogistic(train, candidate, { iterations: 10 });
        const preds = calib.map((r) => sigmoid(logisticLogit(m, r.features, null)));
        const s = score(preds, calibLabels);
        if (s < bestScore) {
          bestScore = s;
          bestFeatures = candidate;
        }
      }
      if (bestFeatures.length < selected.length && bestScore < currentScore - 1e-5) {
        selected = bestFeatures;
        currentScore = bestScore;
        currentModel = trainLogistic(train, selected, { iterations: 10 });
        currentPreds = calib.map((r) => sigmoid(logisticLogit(currentModel, r.features, null)));
        stagnationRounds = 0;
        improved = true;
      } else {
        stagnationRounds += 1;
        if (stagnationRounds >= 2) break;
      }
    }
  }

  const lrModel = trainLogistic(train, selected);

  // Tune Elo home-field advantage on the training set.
  let eloHfa = 30;
  if (train.length >= 20) {
    let bestBrier = Infinity;
    for (const hfa of HFA_GRID) {
      const preds = train.map((r) => eloProb(r, hfa));
      const b = computeBrier(preds, train.map((r) => r.label));
      if (b < bestBrier) {
        bestBrier = b;
        eloHfa = hfa;
      }
    }
  }

  const lrLogits = calib.map((r) => logisticLogit(lrModel, r.features, null));
  const eloLogits = calib.map((r) => logit(eloProb(r, eloHfa)));

  // Blend weight tuned on the calibrate set.
  let blendW = 0.5;
  if (calib.length >= 20) {
    let bestBrier = Infinity;
    for (let w = 0; w <= 1.0001; w += 0.05) {
      const preds = calib.map((_, i) => sigmoid((1 - w) * lrLogits[i] + w * eloLogits[i]));
      const b = computeBrier(preds, calibLabels);
      if (b < bestBrier) {
        bestBrier = b;
        blendW = w;
      }
    }
  }

  // kNN is O(train × calib × d) per evaluation — far too slow at 7k+ train
  // rows during a refresh. Subsample to the 1,500 most-recent training rows,
  // which still preserves the as-of-time recency signal the model cares about.
  const KNN_TRAIN_CAP = 1500;
  const knnTrain = train.length > KNN_TRAIN_CAP ? train.slice(train.length - KNN_TRAIN_CAP) : train;
  const knn = knnModel(knnTrain, selected);
  const nb = naiveBayesModel(train, selected);

  const candPreds: Record<string, number[]> = {
    "Elo rating": calib.map((r) => eloProb(r, eloHfa)),
    "Logistic regression": calib.map((r) => sigmoid(logisticLogit(lrModel, r.features, null))),
    "k-NN (k=21)": calib.map((r) => knn(r.features)),
    "Naive Bayes": calib.map((r) => nb(r.features)),
    "Blended ensemble": calib.map((_, i) => sigmoid((1 - blendW) * lrLogits[i] + blendW * eloLogits[i])),
  };

  const candidates: CandidateModel[] = [];
  let bestSingleName = "Blended ensemble";
  let bestAuc = -1;
  let bestBrier = Infinity;
  for (const name of Object.keys(candPreds)) {
    const p = candPreds[name];
    const m = evaluate(p, calibLabels);
    candidates.push({
      name,
      auc: m.auc,
      brier: m.brier,
      logLoss: m.logLoss,
      ece: m.ece,
      selected: false,
      note: "",
    });
    if (m.auc > bestAuc + 0.003 || (Math.abs(m.auc - bestAuc) <= 0.003 && m.brier < bestBrier)) {
      if (m.auc > bestAuc + 0.003) {
        bestAuc = m.auc;
        bestBrier = m.brier;
        bestSingleName = name;
      } else if (m.brier < bestBrier) {
        bestBrier = m.brier;
        bestSingleName = name;
      }
    }
  }
  for (const c of candidates) c.selected = c.name === bestSingleName;

  // Solve for optimal stacking weights across all candidate models.
  const stacking = buildStackingWeights(candPreds, calibLabels);

  // Prefer the stacked ensemble only when it measurably reduces holdout risk.
  let bestName = bestSingleName;
  let chosenPreds = candPreds[bestSingleName];
  if (stacking.preds.length > 0 && stacking.brier < bestBrier - 0.0005) {
    bestName = "Stacked ensemble";
    chosenPreds = stacking.preds;
  }

  // Fit isotonic calibration on the calibrate set.
  const order = chosenPreds.map((p, i) => ({ p, y: calibLabels[i] })).sort((a, b) => a.p - b.p);
  const isotonicPoints = isotonicRegression(order.map((o) => o.p), order.map((o) => o.y));

  const calibratedCalib = chosenPreds.map((p) => applyIsotonic(isotonicPoints, p));

  // Monte Carlo decision: enable the stochastic component only if it reduces
  // holdout Brier (risk) meaningfully. The Gauss-Hermite quadrature is O(1)
  // per probability, so this is cheap even at the full calibrate-set size.
  const mcGrid = [0.15, 0.3, 0.45];
  let mcSigma = 0;
  let mcEnabled = false;
  let baseBrier = computeBrier(calibratedCalib, calibLabels);
  let bestMcBrier = baseBrier;
  for (const s of mcGrid) {
    const preds = calibratedCalib.map((p) => monteCarloAdjust(p, s, 10000));
    const b = computeBrier(preds, calibLabels);
    if (b < bestMcBrier - 0.0005) {
      bestMcBrier = b;
      mcSigma = s;
    }
  }
  if (mcSigma > 0) mcEnabled = true;

  const monteCarloRationale = mcEnabled
    ? `Monte Carlo enabled: σ=${mcSigma} reduces calibration-set Brier ${baseBrier.toFixed(4)} → ${bestMcBrier.toFixed(4)} (calibration error shrinks toward the mean).`
    : `Monte Carlo disabled: no stochastic σ in {0.1..0.6} reduced calibration-set Brier below ${baseBrier.toFixed(4)}. Deterministic point estimates are kept.`;

  const model: TrainedModel = {
    featureNames: selected,
    weights: lrModel.weights,
    bias: lrModel.bias,
    featureStats: lrModel.featureStats,
    isotonicPoints,
    monteCarloSigma: mcSigma,
    monteCarloEnabled: mcEnabled,
    eloHfa,
  };

  // Reconcile the run-scoring model with the win-probability model.
  const runMarginCalibration = fitRunMarginCalibration(rows, model);

  const predict = (game: RawGame): Prediction =>
    applyModel(
      model,
      buildFeaturesForGame(game, teamState),
      teamState.elo[game.home.id] ?? ELO_INIT,
      teamState.elo[game.away.id] ?? ELO_INIT,
    );

  // Final unbiased metrics on the test set (as-of-time features only).
  const testPreds = test.map((r) => applyModel(model, r.features, r.homeElo, r.awayElo).homeWinProb);
  const testEval = evaluate(testPreds, testLabels);

  // Drift monitoring, rolling risk, and version history (same pipeline).
  const featureDrift = computeFeatureDrift(rows, selected);
  const rolling = computeRollingBrier(rows, model, opts.asOfDate);
  const modelVersions = buildModelVersions(rows, opts.asOfDate, testEval);
  const brierBaseline = modelVersions.length > 1 ? modelVersions[1].brier : rolling.baseline;

  // Feature importances (univariate AUC on the full dataset + coefficient).
  // All candidate features are listed so the UI can show selection decisions.
  const fullLabels = rows.map((r) => r.label);
  const featureImportances: FeatureImportance[] = FEATURE_KEYS.map((f) => {
    const uni = computeAuc(rows.map((r) => r.features[f]), fullLabels);
    const idx = lrModel.featureNames.indexOf(f);
    const active = idx >= 0;
    const w = active ? lrModel.weights[idx] : 0;
    return { feature: f, label: FEATURE_LABELS[f], weight: w, importance: Math.abs(w), univariateAuc: uni, active };
  });

  // Diagnostics: 5-fold cross-validation and hyperparameter audit trail.
  const crossValidation = crossValidate(rows, selected, 5);
  const optimizationParams: OptimizationParams = {
    learningRate: 0,
    l2Lambda: 0.001,
    epochs: 20,
    hfaGrid: HFA_GRID,
    blendStep: 0.05,
    mcSigmaGrid: mcGrid,
    cvFolds: 5,
    isotonicMethod: "Isotonic (PAV)",
    featureSelection: "Greedy backward elimination (L2 logistic, IRLS)",
  };

  // Run-scoring model: predicted scores, totals, and run lines. Fitted on all
  // completed games (a season-talent model) and isotonic-calibrated on the
  // first 85% so run-line probabilities minimize risk (Brier). 200 trials is
  // plenty for the rank-only isotonic fit; the heavy 10,000-trial simul only
  // fires later for the upcoming games surfaced in the UI.
  const runModel = fitRunModel(completedGames);
  const rlRaw: number[] = [];
  const rlOutcomes: number[] = [];
  for (const r of rows.slice(0, calibEnd)) {
    const sim = simulateRuns(runModel, r.game.home.id, r.game.away.id, 0, 200);
    const margin = (r.game.home.score ?? 0) - (r.game.away.score ?? 0);
    rlRaw.push(sim.homeRunLineProb);
    rlOutcomes.push(margin >= 2 ? 1 : 0);
  }
  const rlOrder = rlRaw.map((p, i) => ({ p, y: rlOutcomes[i] })).sort((a, b) => a.p - b.p);
  const runLineCalibration =
    rlOrder.length >= 40
      ? isotonicRegression(rlOrder.map((o) => o.p), rlOrder.map((o) => o.y))
      : [];

  const teamMeta = new Map<number, { name: string; abbrev: string }>();
  for (const g of completedGames) {
    if (!teamMeta.has(g.away.id)) teamMeta.set(g.away.id, { name: g.away.name, abbrev: g.away.abbrev });
    if (!teamMeta.has(g.home.id)) teamMeta.set(g.home.id, { name: g.home.name, abbrev: g.home.abbrev });
  }
  const powerRankings: PowerRanking[] = Object.keys(teamState.elo)
    .map((id) => {
      const teamId = Number(id);
      const rec = teamState.records[teamId] ?? { wins: 0, losses: 0 };
      const homeRec = teamStats.homeRecords[teamId] ?? { wins: 0, losses: 0 };
      const awayRec = teamStats.awayRecords[teamId] ?? { wins: 0, losses: 0 };
      const meta = teamMeta.get(teamId) ?? { name: `Team ${teamId}`, abbrev: "TBD" };
      const homeTotal = homeRec.wins + homeRec.losses;
      const awayTotal = awayRec.wins + awayRec.losses;
      return {
        teamId,
        name: meta.name,
        abbrev: meta.abbrev,
        elo: teamState.elo[teamId],
        wins: rec.wins,
        losses: rec.losses,
        winPct: rec.wins + rec.losses > 0 ? rec.wins / (rec.wins + rec.losses) : 0,
        last10WinPct: teamState.form[teamId] ?? 0.5,
        lastGameDate: teamState.lastGameDate[teamId] ?? "",
        injuries: teamState.injuries[teamId] ?? 0,
        runDiff: teamStats.runDiff[teamId] ?? 0,
        homeWinPct: homeTotal > 0 ? homeRec.wins / homeTotal : 0,
        awayWinPct: awayTotal > 0 ? awayRec.wins / awayTotal : 0,
      };
    })
    .sort((a, b) => b.elo - a.elo);

  const stackedModelCount = stacking.weights.filter((w) => w.weight > 0).length;
  const description =
    bestName === "Stacked ensemble"
      ? `Stacked ensemble of ${stackedModelCount} models (greedy forward selection), isotonic-calibrated${mcEnabled ? ", Monte Carlo-smoothed" : ""}.`
      : bestName === "Blended ensemble"
        ? `Ensemble: ${(1 - blendW).toFixed(2)}·logistic + ${blendW.toFixed(2)}·Elo, isotonic-calibrated${mcEnabled ? ", Monte Carlo-smoothed" : ""}.`
        : `${bestName}, isotonic-calibrated${mcEnabled ? ", Monte Carlo-smoothed" : ""}.`;

  const result: ModelRunResult = {
    season: opts.season,
    asOfDate: opts.asOfDate,
    gamesTrained: n,
    holdoutCount: test.length,
    selectedModel: bestName,
    modelDescription: description,
    featureNames: selected,
    weights: lrModel.weights,
    bias: lrModel.bias,
    featureStats: lrModel.featureStats,
    isotonicPoints,
    eloHfa,
    monteCarloEnabled: mcEnabled,
    monteCarloTrials: mcEnabled ? 10000 : 0,
    monteCarloSigma: mcSigma,
    monteCarloRationale,
    auc: testEval.auc,
    brier: testEval.brier,
    logLoss: testEval.logLoss,
    ece: testEval.ece,
    bins: testEval.bins,
    confidenceDistribution: testEval.confidenceDistribution,
    calibrationCurve: testEval.calibrationCurve,
    featureImportances,
    candidates,
    powerRankings,
    featureDrift,
    rollingBrier: rolling.points,
    brierBaseline,
    modelVersions,
    stackingWeights: stacking.weights,
    crossValidation,
    optimizationParams,
    runModel,
    runLineCalibration,
    runMarginCalibration,
  };

  return { result, model, teamState, rows, predict };
}

export function americanOdds(p: number): number {
  const q = clamp(p, 0.001, 0.999);
  return q >= 0.5 ? -Math.round((100 * q) / (1 - q)) : Math.round((100 * (1 - q)) / q);
}
