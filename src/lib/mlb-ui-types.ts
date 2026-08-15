// UI-facing types for the model state and game documents.
import type {
  CalibrationBin,
  CandidateModel,
  ConfidencePoint,
  CurvePoint,
  FeatureImportance,
  GameDoc,
  PowerRanking,
  TodaysRecord,
} from "@/convex/ml/types";

export type {
  CalibrationBin,
  CandidateModel,
  ConfidencePoint,
  CurvePoint,
  FeatureImportance,
  GameDoc,
  PowerRanking,
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
  todaysRecord: TodaysRecord;
}
