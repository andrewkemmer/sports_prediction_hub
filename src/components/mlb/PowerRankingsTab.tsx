import { teamMeta } from "@/convex/ml/teams";
import type { ModelStateDoc } from "@/lib/mlb-ui-types";
import { formatPct } from "@/lib/format";

export function PowerRankingsTab({ modelState }: { modelState: ModelStateDoc }) {
  const rankings = modelState.powerRankings ?? [];

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-xl font-bold tracking-tight">Power Rankings</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Elo ratings computed chronologically from every 2026 regular-season game, adjusted for
          home field and margin of victory.
        </p>
      </div>

      <div className="overflow-hidden rounded-2xl border border-border bg-card">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="border-b border-border/70 text-left text-[11px] uppercase tracking-wider text-muted-foreground">
                <th className="px-4 py-3 font-medium">#</th>
                <th className="px-4 py-3 font-medium">Team</th>
                <th className="px-4 py-3 text-right font-medium">Record</th>
                <th className="px-4 py-3 text-right font-medium">Win %</th>
                <th className="px-4 py-3 text-right font-medium">Last 10</th>
                <th className="px-4 py-3 text-right font-medium">IL</th>
                <th className="px-4 py-3 text-right font-medium">Elo</th>
              </tr>
            </thead>
            <tbody>
              {rankings.map((r, i) => {
                const color = teamMeta(r.teamId).color;
                return (
                  <tr key={r.teamId} className="border-b border-border/50 last:border-0 hover:bg-white/[0.02]">
                    <td className="px-4 py-2.5 text-muted-foreground tabular-nums">{i + 1}</td>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-2">
                        <span className="size-2.5 rounded-full" style={{ backgroundColor: color }} />
                        <span className="font-medium text-foreground">{r.name}</span>
                        <span className="text-xs text-muted-foreground">{r.abbrev}</span>
                      </div>
                    </td>
                    <td className="px-4 py-2.5 text-right text-muted-foreground tabular-nums">
                      {r.wins}-{r.losses}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-foreground">
                      {formatPct(r.winPct, 1)}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-muted-foreground">
                      {formatPct(r.last10WinPct)}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-muted-foreground">
                      {r.injuries > 0 ? r.injuries : "—"}
                    </td>
                    <td className="px-4 py-2.5 text-right font-semibold tabular-nums text-foreground">
                      {Math.round(r.elo)}
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
