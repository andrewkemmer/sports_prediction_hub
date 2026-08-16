// UI-facing types for the model state and game documents.
import type {
  CalibrationBin,
  CalibrationSummary,
  CandidateModel,
  ConfidencePoint,
  CrossValidationResult,
  CurvePoint,
  FeatureDriftItem,
  FeatureImportance,
  GameDoc,
  MarketOdds,
  ModelVersion,
  OptimizationParams,
  PowerRanking,
  RollingBrierPoint,
  RunProjection,
  StackingWeight,
  TodaysRecord,
} from "@/convex/ml/types";
import type { RunModel } from "@/convex/ml/runs";

export type {
  CalibrationBin,
  CandidateModel,
  ConfidencePoint,
  CrossValidationResult,
  CurvePoint,
  FeatureDriftItem,
  FeatureImportance,
  GameDoc,
  MarketOdds,
  ModelVersion,
  OptimizationParams,
  PowerRanking,
  RollingBrierPoint,
  RunProjection,
  StackingWeight,
  TodaysRecord,
};
export type { RunModel };

export interface RefreshProgressDoc {
  key: string;
  stage: string;
  pct: number;
  message: string;
  startedAt: number;
  updatedAt: number;
  done: boolean;
  error?: string;
}

export interface ModelStateDoc {
  key: string;
  trainedAt: number;
  season: string;
  asOfDate: string;
  gamesTrained: number;
  holdoutCount: number;
  selectedModel: string;
  modelDescription: string;
  featureNames: string[];
  weights: number[];
  bias: number;
  featureStats: Record<string, { mean: number; std: number }>;
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
  featureDrift?: FeatureDriftItem[];
  rollingBrier?: RollingBrierPoint[];
  brierBaseline?: number;
  modelVersions?: ModelVersion[];
  stackingWeights?: StackingWeight[];
  crossValidation?: CrossValidationResult;
  optimizationParams?: OptimizationParams;
  runModel?: RunModel;
  runLineCalibration?: { x: number; y: number }[];
  runMarginCalibration?: { slope: number; intercept: number };
  teamSeasonStats?: Record<string, { ops?: number; era?: number; fieldingPct?: number }>;
  calibrationSummary?: CalibrationSummary;
  spearmanRho?: number;
  topDecileWinRate?: number;
  todaysRecord: TodaysRecord;
}
