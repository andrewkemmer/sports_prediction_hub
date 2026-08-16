"use node";

import { internalAction } from "./_generated/server";
import { internal } from "./_generated/api";

// Pages are kept small: the projection query reads full game docs (Convex
// can't project fields), and big pages were slow enough to push the old
// single-shot backfill past Convex's 10-minute Node action limit. 200 docs
// per page keeps every step light and fast.
const BACKFILL_PAGE = 200;

/** Best-effort progress writer — must never kill the action. */
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

function etDateString(d: Date): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(d);
}

/**
 * Builds the compact calibration projection for the FULL stored history.
 *
 * It runs as a self-scheduling chain of small steps: each invocation reads
 * one 200-game page, writes those dates' calibration rows immediately, saves
 * its pagination cursor, then schedules the next step. The original single
 * action scanned the whole table in one shot and repeatedly died at Convex's
 * 10-minute action limit (the bar froze at "Backfilling calibration 94%" with
 * `done` never written). With per-page work plus a persisted resume cursor,
 * any killed step is simply resumed by the next refresh — the chain always
 * converges, and the table is filled idempotently (dates are delete+rewrite,
 * so partial progress is safe).
 *
 * The full-range calibration metrics are computed on read from the compact
 * `calibration` table, so this action never needs to precompute a summary.
 *
 * No in-chain dedupe: each step writes the state doc, so a freshness-based
 * guard would make every step see its own previous step's write and skip
 * itself (that bug killed the chain after one step). Duplicate-chain
 * protection lives in refreshModel, which only schedules a new step when no
 * chain is currently active.
 */
export const backfillCalibration = internalAction({
  args: {},
  handler: async (ctx) => {
    try {
      const state: any = await ctx.runQuery(internal.mlb.getCalibrationBackfill, {});
      if (state?.done) return { skipped: true };

      const today = etDateString(new Date());
      const cursor = state?.cursor ?? null;
      const scannedBase = state?.scanned ?? 0;

      // 1. One page of the games table (compact projection).
      const page: any = await ctx.runQuery(internal.mlb.getCalibrationProjection, {
        startDate: "2022-03-15",
        endDate: today,
        cursor,
        limit: BACKFILL_PAGE,
      });

      // 2. Write this page's rows immediately (idempotent per date).
      const byDate = new Map<string, any[]>();
      for (const r of page.games) {
        if (r.winner !== "home" && r.winner !== "away") continue;
        if (typeof r.pickProb !== "number") continue;
        const list = byDate.get(r.date) ?? [];
        list.push(r);
        byDate.set(r.date, list);
      }
      const dates = [...byDate.keys()];
      const groups: { date: string; rows: any[] }[][] = [];
      for (let i = 0; i < dates.length; i += 40) {
        groups.push(dates.slice(i, i + 40).map((date) => ({ date, rows: byDate.get(date)! })));
      }
      for (const group of groups) {
        await ctx.runMutation(internal.mlb.bulkReplaceCalibration, { groups: group });
      }

      const scanned = scannedBase + page.games.length;

      // 3. Persist the cursor so the next step resumes here.
      await ctx.runMutation(internal.mlb.saveCalibrationBackfill, {
        cursor: page.cursor,
        scanned,
        done: false,
        error: undefined,
      });

      // 4. Progress — the count advances every step so the banner stays live.
      await report(
        ctx,
        "Backfilling calibration",
        94,
        `Building calibration history — ${scanned.toLocaleString()} games scanned…`,
      );

      // 5. More pages → schedule the next step and return; else finish.
      //    Terminate on cursor null OR an EMPTY page: Convex pagination
      //    returns an empty page with a non-null cursor at the end of a range
      //    query, so checking only `cursor` loops forever (that is what kept
      //    the original backfill spinning on "11,586 games scanned…").
      const finished = !page.cursor || page.games.length === 0;
      if (!finished) {
        await ctx.scheduler.runAfter(0, internal.backfill.backfillCalibration, {});
        return { continued: true, scanned };
      }
      await ctx.runMutation(internal.mlb.saveCalibrationBackfill, {
        cursor: null,
        scanned,
        done: true,
        error: undefined,
      });
      await report(
        ctx,
        "Complete",
        100,
        `Calibration history ready — ${scanned.toLocaleString()} games scanned`,
        { done: true },
      );
      return { backfilled: scanned };
    } catch (e) {
      const message = e instanceof Error ? e.message : "Unknown error";
      await report(ctx, "Backfill failed", 100, message, { done: true, error: message }).catch(() => {});
      // The resume cursor was already persisted before the failing step, so a
      // later refresh resumes from where the chain stopped.
      throw e;
    }
  },
});
