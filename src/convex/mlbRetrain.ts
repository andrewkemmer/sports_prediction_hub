"use node";

import { action } from "./_generated/server";
import { api, internal } from "./_generated/api";

interface RetrainResult {
  gamesTrained: number;
  auc: number;
  brier: number;
  storedGames: number;
}

/**
 * Hard-reset retrain: clears the stored model and game history, then runs the
 * full refresh pipeline from scratch. Because the database is now empty the
 * refresh performs a true cold start — it re-fetches the season history,
 * re-fits the model, and regenerates every stored prediction using the
 * current point-in-time feature logic (no lookahead into a game's own result).
 *
 * This is wired to the Model Monitor "Run Auto-ML Optimization" button. The
 * header Refresh button keeps using `refreshModel` directly, which stays on
 * the fast incremental path and reuses the trained weights.
 */
export const retrainModel = action({
  args: {},
  handler: async (ctx): Promise<RetrainResult> => {
    await ctx.runMutation(internal.mlb.clearModelState, {});

    // Delete the stored game history in bounded pages so we never issue one
    // oversized transaction (a multi-season table can exceed Convex's per-
    // transaction read/write bandwidth limits in a single mutation).
    const ids: any[] = [];
    let cursor: string | null = null;
    do {
      const page: { ids: any[]; cursor: string | null } = await ctx.runQuery(internal.mlb.getGameIdsRange, {
        startDate: "0000-00-00",
        endDate: "9999-99-99",
        cursor,
        limit: 500,
      });
      ids.push(...page.ids);
      cursor = page.cursor;
    } while (cursor);

    for (let i = 0; i < ids.length; i += 300) {
      await ctx.runMutation(internal.mlb.deleteGamesByIds, {
        ids: ids.slice(i, i + 300),
      });
    }

    // The full rebuild writes calibration rows inline, so mark the one-time
    // backfill complete to avoid a redundant scheduled pass afterwards.
    await ctx.runMutation(internal.mlb.saveCalibrationBackfill, {
      cursor: null,
      scanned: 0,
      done: true,
    });

    const result: unknown = await ctx.runAction(api.mlbActions.refreshModel, {});
    return result as RetrainResult;
  },
});
