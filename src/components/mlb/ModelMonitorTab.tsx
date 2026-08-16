import type { ModelStateDoc } from "@/lib/mlb-ui-types";
import { formatMonthDayYear } from "@/lib/format";
import { cn } from "@/lib/utils";
import { AlertTriangle } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const tooltipStyle = {
  backgroundColor: "#161b22",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 8,
  color: "#e6e8ee",
  fontSize: 12,
};

function shortDate(ymd: string): string {
  const d = new Date(`${ymd}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return ymd;
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

function shortMonthDay(ymd: string): string {
  const d = new Date(`${ymd}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return ymd;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

function StatCard({
  dot,
  label,
  value,
  sub,
  subClass,
}: {
  dot: string;
  label: string;
  value: string;
  sub: string;
  subClass: string;
}) {
  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="flex items-center gap-2">
        <span className={cn("size-2 rounded-full", dot)} />
        <span className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">{label}</span>
      </div>
      <div className="mt-2 text-xl font-bold text-foreground">{value}</div>
      <div className={cn("mt-1 text-xs", subClass)}>{sub}</div>
    </div>
  );
}

export function ModelMonitorTab({ modelState }: { modelState: ModelStateDoc }) {
  const drift = modelState.featureDrift ?? [];
  const rolling = modelState.rollingBrier ?? [];
  const versions = modelState.modelVersions ?? [];
  const record = modelState.todaysRecord;

  const now = Date.now();
  const daysAgo = Math.max(0, Math.floor((now - modelState.trainedAt) / 86400000));
  const nextRetrain = modelState.trainedAt + 86400000;

  const warns = drift.filter((d) => d.status === "WARN");
  const firstWarn = warns[0];

  const upsets = record?.upsets ?? [];
  const upsetText = upsets
    .map((u) => `${u.team} over ${u.loser} at ${u.prob}%`)
    .join(", ");

  return (
    <div className="flex flex-col gap-5">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold tracking-tight">Model &amp; Data Drift Monitor</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Tracking model health, feature drift, and performance over time
        </p>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <StatCard
          dot="bg-emerald-500"
          label="Last Retrain"
          value={formatMonthDayYear(modelState.trainedAt)}
          sub={`Model healthy — ${daysAgo === 0 ? "today" : `${daysAgo} day${daysAgo === 1 ? "" : "s"} ago`}`}
          subClass="text-emerald-400"
        />
        <StatCard
          dot="bg-blue-500"
          label="Next Retrain"
          value={formatMonthDayYear(nextRetrain)}
          sub="Nightly schedule — tonight"
          subClass="text-muted-foreground"
        />
        <StatCard
          dot={warns.length > 0 ? "bg-amber-500" : "bg-emerald-500"}
          label="Drift Alerts"
          value={`${warns.length} Warning${warns.length === 1 ? "" : "s"}`}
          sub={firstWarn ? `${firstWarn.label} — elevated PSI` : "All features stable"}
          subClass={firstWarn ? "text-amber-400" : "text-emerald-400"}
        />
      </div>

      {/* Upset monitoring note */}
      {(upsets.length > 0 || record) && (
        <div className="rounded-2xl border border-border bg-card p-5">
          <div className="flex items-center gap-2 text-sm font-bold text-amber-400">
            <AlertTriangle className="size-4" />
            Upset Monitoring Note — {shortDate(record?.date ?? modelState.asOfDate)}
          </div>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            {upsets.length > 0 ? (
              <>
                {upsets.length} upset{upsets.length === 1 ? "" : "s"} today ({upsetText}) — monitoring for
                regime shift. Model went {record?.wins}-{record?.losses} overall but high-confidence picks
                (&gt;65%) showed vulnerability. Will assess after tonight's retrain.
              </>
            ) : (
              <>No upsets recorded — monitoring for regime shift on tonight's retrain.</>
            )}
          </p>
        </div>
      )}

      {/* Feature drift analysis */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <h3 className="mb-4 text-sm font-semibold text-foreground">Feature Drift Analysis (PSI Scores)</h3>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="border-b border-border/70 text-left text-[11px] uppercase tracking-wider text-muted-foreground">
                <th className="pb-2 font-medium">Feature</th>
                <th className="pb-2 text-right font-medium">Current Mean</th>
                <th className="pb-2 text-right font-medium">Baseline Mean</th>
                <th className="pb-2 text-right font-medium">PSI</th>
                <th className="pb-2 text-right font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {drift.map((d) => (
                <tr key={d.feature} className="border-b border-border/50 last:border-0">
                  <td className="py-2.5 font-medium text-foreground">{d.label}</td>
                  <td className="py-2.5 text-right tabular-nums text-foreground">{d.currentMean.toFixed(3)}</td>
                  <td className="py-2.5 text-right tabular-nums text-muted-foreground">{d.baselineMean.toFixed(3)}</td>
                  <td className={cn("py-2.5 text-right tabular-nums", d.status === "WARN" ? "text-amber-400" : "text-foreground")}>
                    {d.psi.toFixed(3)}
                  </td>
                  <td className="py-2.5 text-right">
                    <span
                      className={cn(
                        "rounded-full px-2 py-0.5 text-xs font-semibold",
                        d.status === "WARN" ? "bg-amber-500/15 text-amber-300" : "bg-emerald-500/15 text-emerald-300",
                      )}
                    >
                      {d.status}
                    </span>
                  </td>
                </tr>
              ))}
              {drift.length === 0 && (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-sm text-muted-foreground">
                    No drift data yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Rolling Brier score */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-foreground">Rolling Brier Score (Last 30 Days)</h3>
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="size-2.5 rounded-sm bg-orange-400" /> Brier Score
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-0 w-4 border-t border-dashed border-muted-foreground" /> Baseline (prior version)
            </span>
          </div>
        </div>
        {rolling.length === 0 ? (
          <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
            No rolling risk data yet.
          </div>
        ) : (
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rolling} margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={shortMonthDay}
                stroke="#8b93a7"
                tick={{ fill: "#8b93a7", fontSize: 10 }}
                minTickGap={24}
              />
              <YAxis
                domain={["dataMin - 0.01", "dataMax + 0.01"]}
                stroke="#8b93a7"
                tick={{ fill: "#8b93a7", fontSize: 10 }}
                tickFormatter={(v: number) => v.toFixed(3)}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                labelFormatter={(l: string) => shortDate(l)}
                formatter={(value: number | string) => Number(value).toFixed(3)}
              />
              <ReferenceLine
                y={modelState.brierBaseline ?? modelState.brier}
                stroke="#8b93a7"
                strokeDasharray="4 4"
              />
              <Line
                type="monotone"
                dataKey="brier"
                name="Brier Score"
                stroke="#fb923c"
                strokeWidth={2}
                dot={{ r: 3, fill: "#fb923c", strokeWidth: 0 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        )}
      </div>

      {/* Model version history */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <h3 className="mb-4 text-sm font-semibold text-foreground">Model Version History</h3>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[600px] text-sm">
            <thead>
              <tr className="border-b border-border/70 text-left text-[11px] uppercase tracking-wider text-muted-foreground">
                <th className="pb-2 font-medium">Version</th>
                <th className="pb-2 font-medium">Date</th>
                <th className="pb-2 text-right font-medium">AUC</th>
                <th className="pb-2 text-right font-medium">Brier</th>
                <th className="pb-2 font-medium">Notes</th>
              </tr>
            </thead>
            <tbody>
              {versions.map((v) => (
                <tr key={v.version} className="border-b border-border/50 last:border-0">
                  <td className="py-2.5 font-semibold text-cyan-300">{v.version}</td>
                  <td className="py-2.5 text-muted-foreground">{shortDate(v.date)}</td>
                  <td className="py-2.5 text-right tabular-nums text-foreground">{v.auc.toFixed(3)}</td>
                  <td className="py-2.5 text-right tabular-nums text-foreground">{v.brier.toFixed(3)}</td>
                  <td className="py-2.5 text-muted-foreground">{v.notes}</td>
                </tr>
              ))}
              {versions.length === 0 && (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-sm text-muted-foreground">
                    No version history yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model summary + data source footnote */}
      <p className="text-xs leading-5 text-muted-foreground">
        Model: <span className="text-foreground">{modelState.selectedModel}</span> ·{" "}
        {modelState.featureNames.length} features selected · Monte Carlo{" "}
        {modelState.monteCarloEnabled ? "enabled" : "disabled"} · Data:{" "}
        <span className="text-foreground">statsapi.mlb.com</span> (single consolidated source)
      </p>
    </div>
  );
}
