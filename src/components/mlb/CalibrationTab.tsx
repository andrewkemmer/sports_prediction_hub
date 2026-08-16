import { api } from "@/convex/_generated/api";
import type { CalibrationBin, ConfidencePoint, CurvePoint, ModelStateDoc } from "@/lib/mlb-ui-types";
import { formatNumber, formatPct, formatTrainedAt } from "@/lib/format";
import { cn } from "@/lib/utils";
import { useQuery } from "convex/react";
import { Check, X, Zap } from "lucide-react";
import { useState } from "react";
import {
  Area,
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface GameResultRow {
  gamePk: number;
  date: string;
  away: { abbrev: string; name: string; score?: number };
  home: { abbrev: string; name: string; score?: number };
  winner?: "home" | "away";
  pickTeam: "home" | "away";
  pickProb: number;
  isCorrect?: boolean;
  isUpset?: boolean;
}

interface CalibrationQueryResult {
  games: GameResultRow[];
  metrics: {
    auc: number;
    brier: number;
    logLoss: number;
    ece: number;
    bins: CalibrationBin[];
    confidenceDistribution: ConfidencePoint[];
    calibrationCurve: CurvePoint[];
  };
  total: number;
  correct: number;
  accuracy: number;
}

function shortDate(ymd: string): string {
  const d = new Date(`${ymd}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return ymd;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

const tooltipStyle = {
  backgroundColor: "#161b22",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 8,
  color: "#e6e8ee",
  fontSize: 12,
};

function MetricCard({
  label,
  value,
  color,
  sub,
}: {
  label: string;
  value: number;
  color: string;
  sub?: string;
}) {
  return (
    <div className="rounded-2xl border border-border bg-card p-6">
      <div className="text-center text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <div className="mt-3 text-center text-4xl font-bold tabular-nums" style={{ color }}>
        {value.toFixed(3)}
      </div>
      {sub && <div className="mt-1 text-center text-xs text-muted-foreground">{sub}</div>}
    </div>
  );
}

function gapColor(gap: number): string {
  const a = Math.abs(gap);
  if (a < 0.015) return "text-emerald-400";
  if (a < 0.025) return "text-amber-400";
  return "text-rose-400";
}

export function CalibrationTab({ modelState }: { modelState: ModelStateDoc }) {
  const record = modelState.todaysRecord;
  const seasonStart = `${modelState.season}-03-15`;
  const [startDate, setStartDate] = useState(seasonStart);
  const [endDate, setEndDate] = useState(modelState.asOfDate);
  const [visibleCount, setVisibleCount] = useState(25);
  const [search, setSearch] = useState("");

  const results = useQuery(api.mlb.getCalibrationResults, {
    startDate,
    endDate,
  }) as CalibrationQueryResult | undefined;

  const metrics = results?.metrics;
  const bins: CalibrationBin[] = metrics?.bins ?? [];
  const distribution = metrics?.confidenceDistribution ?? [];
  const curve = metrics?.calibrationCurve ?? [];
  const auc = metrics?.auc ?? modelState.auc;
  const brier = metrics?.brier ?? modelState.brier;
  const logLoss = metrics?.logLoss ?? modelState.logLoss;
  const ece = metrics?.ece ?? modelState.ece;

  const allGames = results?.games ?? [];
  const totalGames = results?.total ?? allGames.length;
  const correctCount = results?.correct ?? allGames.filter((g) => g.isCorrect).length;
  const q = search.trim().toLowerCase();
  const filteredGames = q
    ? allGames.filter((g) =>
        `${g.away.name} ${g.away.abbrev} ${g.home.name} ${g.home.abbrev}`.toLowerCase().includes(q),
      )
    : allGames;
  const visibleGames = filteredGames.slice(0, visibleCount);

  const onStartChange = (v: string) => {
    setStartDate(v);
    if (v && endDate && v > endDate) setEndDate(v);
  };
  const onEndChange = (v: string) => {
    setEndDate(v);
    if (v && startDate && v < startDate) setStartDate(v);
  };

  return (
    <div className="flex flex-col gap-5">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold tracking-tight">Model Calibration Dashboard</h2>
        <span className="mt-3 inline-block rounded-full border border-border bg-card px-3 py-1 text-xs text-muted-foreground">
          n = {formatNumber(totalGames)} games in range · Trained {formatTrainedAt(modelState.trainedAt)} ET
        </span>
        <p className="mt-3 text-sm text-muted-foreground">
          Assessing prediction reliability and accuracy across probability buckets for the predicted
          favorite (one side per game).
        </p>

        {/* Date range selector */}
        <div className="mt-4 flex flex-wrap items-center gap-2 rounded-2xl border border-border bg-card p-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Range
          </span>
          <input
            type="date"
            value={startDate}
            min={seasonStart}
            max={endDate}
            onChange={(e) => onStartChange(e.target.value)}
            className="h-9 cursor-pointer rounded-lg border border-border bg-background px-3 text-sm text-foreground outline-none [color-scheme:dark] focus:border-ring/50"
          />
          <span className="text-muted-foreground">→</span>
          <input
            type="date"
            value={endDate}
            min={startDate}
            max={modelState.asOfDate}
            onChange={(e) => onEndChange(e.target.value)}
            className="h-9 cursor-pointer rounded-lg border border-border bg-background px-3 text-sm text-foreground outline-none [color-scheme:dark] focus:border-ring/50"
          />
          <span className="text-xs text-muted-foreground">
            {formatNumber(totalGames)} completed game{totalGames === 1 ? "" : "s"} ·{" "}
            {formatPct(results?.accuracy ?? 0, 1)} accuracy
          </span>
        </div>
      </div>

      {/* Today's record */}
      {record && record.total > 0 && (
        <div className="rounded-2xl border border-border bg-card p-5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-foreground">Today's Record:</span>
            <span className="flex items-center gap-1 rounded-full bg-emerald-500/15 px-2.5 py-0.5 text-xs font-semibold text-emerald-300">
              {record.wins}-{record.losses}
            </span>
            <span className="text-sm text-muted-foreground">
              {record.completed} completed games · {record.correct} correct picks (
              {formatPct(record.accuracy, 1)}) · {record.upsets.length} upsets
            </span>
          </div>
          {record.upsets.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {record.upsets.map((u, i) => (
                <span
                  key={i}
                  className="flex items-center gap-1 rounded-full bg-amber-500/15 px-2.5 py-0.5 text-xs text-amber-400"
                >
                  <Zap className="size-3" />
                  {u.team} {u.prob}% upset
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Calibration curve */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-foreground">Calibration Curve</h3>
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="size-2.5 rounded-sm bg-[#4d7fff]" /> Model (n={formatNumber(totalGames)})
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-0 w-4 border-t border-dashed border-muted-foreground" /> Perfect calibration
            </span>
          </div>
        </div>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="x"
                name="Predicted"
                domain={[0.45, 0.85]}
                ticks={[0.48, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82]}
                tickFormatter={(v: number) => v.toFixed(2)}
                stroke="#8b93a7"
                tick={{ fill: "#8b93a7", fontSize: 11 }}
                label={{ value: "Mean Predicted Probability", position: "insideBottom", offset: -2, fill: "#8b93a7", fontSize: 11 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="Actual"
                domain={[0.45, 0.85]}
                ticks={[0.48, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82]}
                tickFormatter={(v: number) => v.toFixed(2)}
                stroke="#8b93a7"
                tick={{ fill: "#8b93a7", fontSize: 11 }}
                label={{ value: "Mean Actual Win Rate", angle: -90, position: "insideLeft", offset: 10, fill: "#8b93a7", fontSize: 11 }}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(value: number | string) => Number(value).toFixed(3)}
                labelFormatter={() => ""}
              />
              <ReferenceLine
                segment={[
                  { x: 0.48, y: 0.48 },
                  { x: 0.82, y: 0.82 },
                ]}
                stroke="#8b93a7"
                strokeDasharray="4 4"
              />
              <Scatter data={curve} fill="#4d7fff" shape="circle" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <MetricCard label="AUC-ROC" value={auc} color="#22d3ee" />
        <MetricCard label="Brier Score" value={brier} color="#34d399" />
        <MetricCard label="Log-Loss" value={logLoss} color="#fcd34d" sub="Penalizes confidence" />
        <MetricCard label="Cal. Error" value={ece} color="#e879f9" sub="ECE metric" />
      </div>
      <p className="-mt-1 text-xs text-muted-foreground">
        Metrics are computed on the predicted favorite only (probability &gt; 50%), one side per
        game, over the selected date range.
      </p>

      {/* Confidence distribution & accuracy */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-foreground">Prediction Confidence Distribution &amp; Accuracy</h3>
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="size-2.5 rounded-sm bg-emerald-400" /> Actual accuracy %
            </span>
            <span className="flex items-center gap-1.5">
              <span className="size-2.5 rounded-sm bg-blue-500/60" /> Game count
            </span>
          </div>
        </div>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={distribution} margin={{ top: 8, right: 0, bottom: 8, left: -12 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
              <XAxis dataKey="label" stroke="#8b93a7" tick={{ fill: "#8b93a7", fontSize: 10 }} />
              <YAxis
                yAxisId="left"
                stroke="#8b93a7"
                tick={{ fill: "#8b93a7", fontSize: 10 }}
                allowDecimals={false}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                domain={[0.45, 0.8]}
                ticks={[0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]}
                tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                stroke="#8b93a7"
                tick={{ fill: "#8b93a7", fontSize: 10 }}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(value: number | string, name: string) =>
                  name === "accuracy" ? `${(Number(value) * 100).toFixed(1)}%` : value
                }
              />
              <Bar yAxisId="left" dataKey="count" name="Game Count" fill="rgba(77,125,255,0.45)" radius={[4, 4, 0, 0]} />
              <Area
                yAxisId="right"
                dataKey="accuracy"
                name="accuracy"
                stroke="none"
                fill="#34d399"
                fillOpacity={0.12}
              />
              <Line
                yAxisId="right"
                dataKey="accuracy"
                name="accuracy"
                stroke="#34d399"
                strokeWidth={2}
                dot={{ r: 3, fill: "#34d399", strokeWidth: 0 }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Reliability diagram */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <h3 className="mb-4 text-sm font-semibold text-foreground">Reliability Diagram — Binned Data</h3>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-sm">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-wider text-muted-foreground">
                <th className="pb-2 font-medium">Bucket</th>
                <th className="pb-2 text-right font-medium">Mean predicted</th>
                <th className="pb-2 text-right font-medium">Mean actual</th>
                <th className="pb-2 text-right font-medium">Count</th>
                <th className="pb-2 text-right font-medium">Gap</th>
              </tr>
            </thead>
            <tbody>
              {bins.map((b) => (
                <tr key={b.label} className="border-t border-border/70">
                  <td className="py-2.5 font-medium text-foreground">{b.label}</td>
                  <td className="py-2.5 text-right tabular-nums text-foreground">{b.meanPredicted.toFixed(3)}</td>
                  <td className="py-2.5 text-right tabular-nums text-foreground">{b.meanActual.toFixed(3)}</td>
                  <td className="py-2.5 text-right tabular-nums text-muted-foreground">{b.count}</td>
                  <td className={cn("py-2.5 text-right tabular-nums", gapColor(b.gap))}>
                    {b.gap >= 0 ? "+" : ""}
                    {b.gap.toFixed(3)}
                  </td>
                </tr>
              ))}
              {bins.length === 0 && (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-sm text-muted-foreground">
                    No binned data for this range.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Game history: predicted vs actual */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-foreground">Game History — Predicted vs Actual</h3>
            <p className="mt-1 text-xs text-muted-foreground">
              {formatNumber(totalGames)} games · {formatNumber(correctCount)} correct picks (
              {formatPct(totalGames > 0 ? correctCount / totalGames : 0, 1)})
            </p>
          </div>
          <input
            type="text"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setVisibleCount(25);
            }}
            placeholder="Filter by team…"
            className="h-9 w-full max-w-xs rounded-lg border border-border bg-background px-3 text-sm text-foreground outline-none placeholder:text-muted-foreground focus:border-ring/50"
          />
        </div>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="border-b border-border/70 text-left text-[11px] uppercase tracking-wider text-muted-foreground">
                <th className="pb-2 pr-4 font-medium">Date</th>
                <th className="pb-2 pr-4 font-medium">Matchup</th>
                <th className="pb-2 pr-4 text-right font-medium">Final</th>
                <th className="pb-2 pr-4 text-right font-medium">Predicted</th>
                <th className="pb-2 pr-4 text-right font-medium">Actual</th>
                <th className="pb-2 text-right font-medium">Result</th>
              </tr>
            </thead>
            <tbody>
              {visibleGames.map((g) => {
                const pickAbbrev = g.pickTeam === "home" ? g.home.abbrev : g.away.abbrev;
                const winnerAbbrev = g.winner === "home" ? g.home.abbrev : g.away.abbrev;
                return (
                  <tr key={g.gamePk} className="border-b border-border/50 last:border-0">
                    <td className="whitespace-nowrap py-2 pr-4 text-muted-foreground">{shortDate(g.date)}</td>
                    <td className="whitespace-nowrap py-2 pr-4">
                      <span className="font-medium text-foreground">{g.away.abbrev}</span>
                      <span className="text-muted-foreground"> @ </span>
                      <span className="font-medium text-foreground">{g.home.abbrev}</span>
                    </td>
                    <td className="whitespace-nowrap py-2 pr-4 text-right tabular-nums text-foreground">
                      {g.away.score ?? "—"} – {g.home.score ?? "—"}
                    </td>
                    <td className="whitespace-nowrap py-2 pr-4 text-right tabular-nums text-muted-foreground">
                      {pickAbbrev} {formatPct(g.pickProb)}
                    </td>
                    <td className="whitespace-nowrap py-2 pr-4 text-right tabular-nums text-foreground">
                      {winnerAbbrev}
                    </td>
                    <td className="whitespace-nowrap py-2 text-right">
                      {g.isCorrect ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs font-semibold text-emerald-300">
                          <Check className="size-3" /> Correct
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded-full bg-rose-500/15 px-2 py-0.5 text-xs font-semibold text-rose-300">
                          <X className="size-3" /> Upset
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
              {visibleGames.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-sm text-muted-foreground">
                    {allGames.length === 0
                      ? "No completed games in this range — adjust the dates or click Refresh."
                      : "No games match your filter."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {visibleCount < filteredGames.length && (
          <button
            type="button"
            onClick={() => setVisibleCount((c) => c + 50)}
            className="mt-3 w-full cursor-pointer rounded-lg border border-border/80 bg-white/[0.02] py-2 text-xs font-medium text-blue-400 transition-colors hover:bg-blue-500/10"
          >
            Show more ({formatNumber(filteredGames.length - visibleCount)} remaining)
          </button>
        )}
      </div>
    </div>
  );
}
