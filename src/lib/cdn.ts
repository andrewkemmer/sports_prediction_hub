/**
 * CDN data source for the MLB prediction dashboard.
 *
 * All dashboard data (model state, game predictions, calibration metrics)
 * is served from a single static JSON file on GitHub's CDN. This replaces
 * every Convex query/action that previously fed the UI.
 */

import type { GameDoc, CalibrationBin, ConfidencePoint, CurvePoint, CalibrationSummary } from "@/convex/ml/types";
import type { ModelStateDoc } from "@/lib/mlb-ui-types";

// ─── CDN URL ────────────────────────────────────────────────────────────
// Replace this with the actual raw.githubusercontent.com URL pointing to
// the pre-computed JSON artifact (produced by the Python backend's export
// step). The JSON shape must match CdnPayload below.
export const CDN_URL = "https://githubusercontent.com";

// ─── CDN payload shape ──────────────────────────────────────────────────
// This is the top-level JSON structure the CDN endpoint returns.
export interface CdnPayload {
  /** The full model state singleton (replaces getModelState) */
  modelState: ModelStateDoc | null;

  /** Games indexed by date string (e.g. "2026-04-01") — replaces getGamesByDate */
  gamesByDate: Record<string, GameDoc[]>;

  /** All individual game cards keyed by gamePk — replaces getCalibrationGames */
  gameCards: Record<string, GameDoc>;

  /**
   * Pre-computed calibration summary for the full training range.
   * Computed once by the Python backend over all completed games.
   * This replaces the server-side getCalibrationResults for the full range.
   * For narrower date ranges the CalibrationTab computes metrics client-side
   * from the full gameCards pool.
   */
  calibrationSummary: CalibrationSummary;

  /** Calibration bins for the full training range */
  calibrationBins: CalibrationBin[];
  /** Confidence distribution points */
  confidenceDistribution: ConfidencePoint[];
  /** Calibration curve points */
  calibrationCurve: CurvePoint[];
  /** Totals calibration metrics */
  totalsMetrics: { n: number; mae: number; rmse: number; bias: number };
  /** Run-line calibration metrics */
  runLineMetrics: { n: number; auc: number; brier: number; accuracy: number };
  /** Moneyline total / correct / accuracy */
  moneylineTotal: number;
  moneylineCorrect: number;
  moneylineAccuracy: number;
}
