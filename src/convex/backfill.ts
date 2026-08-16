"use node";

import { internalAction } from "./_generated/server";
import { internal } from "./_generated/api";
import { calibrationCurvePoints, computeAuc, computeBrier, evaluate } from "./ml/model";

// ---------------------------------------------------------------------------
// Compact calibration row + summary computation (mirrors mlbActions.ts)
// ---------------------------------------------------------------------------

interface CalibrationRow {
  gamePk: number;
  date: string;
  away: { abbrev: string; name: string; score?: number };
  home: { abbrev: string; name: string; score?: number };
  winner?: "home" | "away";
  pickTeam: "home" | "away";
  pickProb: number;
  homeWinProb: number;
  isCorrect?: boolean;
  isUpset?: boolean;
  predictedTotal?: number;
  homeRunLineProb?: number;
  actualTotal: number;
  actualMargin: number;
}

interface CalibrationSummary {
  metrics: {
    auc: number;
    brier: number;
    logLoss: number;
    ece: number;
    bins: unknown[];
    confidenceDistribution: unknown[];
    calibrationCurve: unknown[];
  };
  totalsMetrics: { n: number; mae: number; rmse: number; bias: number };
  runLineMetrics: { n: number; auc: number; brier: number; accuracy: number };
  total: number;
  correct: number;
  accuracy: number;
}

function buildCalibrationSummary(rows: CalibrationRow[]): CalibrationSummary {
  const preds = rows.map((d) => d.pickProb);
  const labels = rows.map((d) => (d.isCorrect ? 1 : 0));
  const evalResult = evaluate(preds, labels);
  const curve = calibrationCurvePoints(preds, labels, 8);
  const metrics = {
    auc: evalResult.auc,
    brier: evalResult.brier,
    logLoss: evalResult.logLoss,
    ece: evalResult.ece,
    bins: evalResult.bins,
    confidenceDistribution: evalResult.confidenceDistribution,
    calibrationCurve: curve.length > 0 ? curve : evalResult.calibrationCurve,
  };

  let tN = 0;
  let tAbs = 0;
  let tSq = 0;
  let tBias = 0;
  const rlPreds: number[] = [];
  const rlLabels: number[] = [];
  for (const d of rows) {
    if (typeof d.predictedTotal === "number") {
      tN += 1;
      const err = d.predictedTotal - d.actualTotal;
      tAbs += Math.abs(err);
      tSq += err * err;
      tBias += err;
    }
    if (typeof d.homeRunLineProb === "number") {
      rlPreds.push(d.homeRunLineProb);
      rlLabels.push(d.actualMargin >= 2 ? 1 : 0);
    }
  }

  const totalsMetrics = {
    n: tN,
    mae: tN > 0 ? tAbs / tN : 0,
    rmse: tN > 0 ? Math.sqrt(tSq / tN) : 0,
    bias: tN > 0 ? tBias / tN : 0,
  };
  let runLineMetrics = { n: rlPreds.length, auc: 0, brier: 0, accuracy: 0 };
  if (rlPreds.length > 0) {
    runLineMetrics = {
      n: rlPreds.length,
      auc: computeAuc(rlPreds, rlLabels),
      brier: computeBrier(rlPreds, rlLabels),
      accuracy:
        rlPreds.filter((p, i) => (p >= 0.5 ? 1 : 0) === rlLabels[i]).length / rlPreds.length,
    };
  }

  const total = rows.length;
  const correct = rows.filter((d) => d.isCorrect).length;
  return {
    metrics,
    totalsMetrics,
    runLineMetrics,
    total,
    correct,
    accuracy: total > 0 ? correct / total : 0,
  };
}

// ---------------------------------------------------------------------------
// Progress reporting (best-effort — must never kill the action)
// ---------------------------------------------------------------------------

async function report(
  ctx: any,
  stage: string,
  pct: number,
  message: string,
  extra: { done?: boolean; error?: string } = {},
): Promise<void> {
  try {
    await ctx.runMutation(internal.mlb.setRefreshProgress, {
      stage,
      pct,
      message,
      ...extra,
    });
  } catch {
    // ignore — progress is informational
  }
}

// ---------------------------------------------------------------------------
// Backfill action
// ---------------------------------------------------------------------------

/**
 * Builds the compact calibration projection for the FULL stored history and
 * saves the precomputed summary. This used to run inline inside refreshModel,
 * but scanning the whole games table on top of the refresh's API work pushed
 * the action past Convex's 10-minute Node action limit (refreshes died with
 * the bar frozen at "Backfilling calibration 94%"). As a scheduled action it
 * gets its own execution budget, and it only ever runs until the summary
 * exists — after that refreshes skip it entirely.
 */
export const backfillCalibration = internalAction({
  args: {},
  handler: async (ctx) => {
    try {
      const previous: any = await ctx.runQuery(internal.mlb.getLatestModelState, {});
      if (previous?.calibrationSummary && (previous.calibrationSummary?.total ?? 0) > 0) {
        return { backfilled: 0, skipped: true };
      }
      const today = new Intl.DateTimeFormat("en-CA", {
        timeZone: "America/New_York",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
      }).format(new Date());
      await report(ctx, "Backfilling calibration", 94, "Building calibration history from stored games…");

      // Compact paginated read of the games table — only the calibration
      // fields are returned (~1/20th the size of a full game doc).
      const backfillByDate = new Map<string, CalibrationRow[]>();
      let cursor: string | null = null;
      let scanned = 0;
      do {
        const page: any = await ctx.runQuery(
          internal.mlb.getCalibrationProjection,
          { startDate: "2022-03-15", endDate: today, cursor, limit: 500 },
        );
        for (const r of page.games) {
          if (r.winner !== "home" && r.winner !== "away") continue;
          if (typeof r.pickProb !== "number") continue;
          const list = backfillByDate.get(r.date) ?? [];
          list.push(r);
          backfillByDate.set(r.date, list);
        }
        cursor = page.cursor;
        scanned += page.games.length;
        await report(
          ctx,
          "Backfilling calibration",
          94,
          `Building calibration history — ${scanned.toLocaleString()} games scanned…`,
        );
      } while (cursor);

      // Batch ~40 dates per mutation (~18 mutations for ~700 dates).
      const calibrationDates = [...backfillByDate.keys()];
      const dateGroups: { date: string; rows: CalibrationRow[] }[][] = [];
      for (let i = 0; i < calibrationDates.length; i += 40) {
        const group: { date: string; rows: CalibrationRow[] }[] = [];
        for (const date of calibrationDates.slice(i, i + 40)) {
          group.push({ date, rows: backfillByDate.get(date)! });
        }
        dateGroups.push(group);
      }
      await report(
        ctx,
        "Backfilling calibration",
        95,
        `Writing calibration rows for ${calibrationDates.length.toLocaleString()} dates…`,
      );
      let written = 0;
      const workers = Array.from({ length: Math.min(6, dateGroups.length) }, async () => {
        while (dateGroups.length > 0) {
          const group = dateGroups.shift()!;
          await ctx.runMutation(internal.mlb.bulkReplaceCalibration, { groups: group });
          written += group.length;
          await report(
            ctx,
            "Backfilling calibration",
            calibrationDates.length > 0
              ? 95 + Math.floor((written / calibrationDates.length) * 4)
              : 95,
            `Writing calibration rows — ${written.toLocaleString()}/${calibrationDates.length.toLocaleString()} dates…`,
          );
        }
      });
      await Promise.all(workers);

      const rows = [...backfillByDate.values()].flat();
      const summary = rows.length > 0 ? buildCalibrationSummary(rows) : undefined;
      if (summary) {
        await ctx.runMutation(internal.mlb.setCalibrationSummary, { summary });
      }
      await report(
        ctx,
        "Complete",
        100,
        `Calibration history ready — ${rows.length.toLocaleString()} games`,
        { done: true },
      );
      return { backfilled: rows.length };
    } catch (e) {
      const message = e instanceof Error ? e.message : "Unknown error";
      await report(ctx, "Backfill failed", 100, message, { done: true, error: message }).catch(() => {});
      throw e;
    }
  },
});
