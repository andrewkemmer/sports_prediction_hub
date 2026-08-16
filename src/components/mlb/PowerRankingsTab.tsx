import { teamMeta } from "@/convex/ml/teams";
import type { ModelStateDoc } from "@/lib/mlb-ui-types";
import { formatDateShort } from "@/lib/format";
import { cn } from "@/lib/utils";

function eloColor(rank: number): string {
  if (rank <= 5) return "text-cyan-300";
  if (rank <= 10) return "text-amber-300";
  return "text-muted-foreground";
}

export function PowerRankingsTab({ modelState }: { modelState: ModelStateDoc }) {
  const rankings = (modelState.powerRankings ?? []).slice(0, 15);

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-xl font-bold tracking-tight">Power Rankings</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Current Elo-based power rankings · As of {formatDateShort(modelState.asOfDate)} · Top{" "}
          {rankings.length} teams
        </p>
      </div>

      <div className="overflow-hidden rounded-2xl border border-border bg-card">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead>
              <tr className="border-b border-border/70 bg-white/[0.02] text-left text-[11px] uppercase tracking-wider text-muted-foreground">
                <th className="px-4 py-3 font-medium">Rank</th>
                <th className="px-4 py-3 font-medium">Team</th>
                <th className="px-4 py-3 text-right font-medium">Elo</th>
                <th className="px-4 py-3 text-right font-medium">W-L</th>
                <th className="px-4 py-3 text-right font-medium">Pct</th>
                <th className="px-4 py-3 text-right font-medium">Run Diff</th>
                <th className="px-4 py-3 text-right font-medium">L10</th>
                <th className="px-4 py-3 text-right font-medium">Home%</th>
                <th className="px-4 py-3 text-right font-medium">Away%</th>
              </tr>
            </thead>
            <tbody>
              {rankings.map((r, i) => {
                const color = teamMeta(r.teamId).color;
                const runDiff = r.runDiff ?? 0;
                const homeWinPct = r.homeWinPct ?? 0;
                const awayWinPct = r.awayWinPct ?? 0;
                return (
                  <tr
                    key={r.teamId}
                    className="border-b border-border/50 last:border-0 hover:bg-white/[0.02]"
                  >
                    <td className="px-4 py-2.5 tabular-nums text-muted-foreground">{i + 1}</td>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-2.5">
                        <span className="h-5 w-1 shrink-0 rounded-full" style={{ backgroundColor: color }} />
                        <div className="leading-tight">
                          <div className="font-medium text-foreground">{r.name}</div>
                          <div className="text-xs text-muted-foreground">{r.abbrev}</div>
                        </div>
                      </div>
                    </td>
                    <td className={cn("px-4 py-2.5 text-right font-semibold tabular-nums", eloColor(i + 1))}>
                      {Math.round(r.elo)}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-foreground">
                      {r.wins}-{r.losses}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-foreground">
                      {r.winPct.toFixed(3).replace("0.", ".")}
                    </td>
                    <td
                      className={cn(
                        "px-4 py-2.5 text-right tabular-nums",
                        runDiff > 0 ? "text-emerald-400" : runDiff < 0 ? "text-rose-400" : "text-muted-foreground",
                      )}
                    >
                      {runDiff > 0 ? "+" : ""}
                      {runDiff}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-muted-foreground">
                      {Math.round(r.last10WinPct * 10)}-{10 - Math.round(r.last10WinPct * 10)}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-muted-foreground">
                      {homeWinPct.toFixed(3).replace("0.", ".")}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-muted-foreground">
                      {awayWinPct.toFixed(3).replace("0.", ".")}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
