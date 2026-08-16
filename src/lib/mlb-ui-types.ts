// UI-facing types for the model state and game documents.
import type {
  CalibrationBin,
  CandidateModel,
  ConfidencePoint,
  CurvePoint,
  FeatureDriftItem,
  FeatureImportance,
  GameDoc,
  ModelVersion,
  PowerRanking,
  RollingBrierPoint,
  TodaysRecord,
} from "@/convex/ml/types";

export type {
  CalibrationBin,
  CandidateModel,
  ConfidencePoint,
  CurvePoint,
  FeatureDriftItem,
  FeatureImportance,
  GameDoc,
  ModelVersion,
  PowerRanking,
  RollingBrierPoint,
  TodaysRecord,
};

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
  spearmanRho?: number;
  topDecileWinRate?: number;
  todaysRecord: TodaysRecord;
}
