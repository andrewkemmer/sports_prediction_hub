import type { CalibrationBin, ConfidencePoint, CurvePoint, ModelStateDoc } from "@/lib/mlb-ui-types";
import type { GameDoc } from "@/convex/ml/types";
import { formatNumber, formatPct, formatTrainedAt } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Check, X, Zap } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
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
  predictedTotal?: number;
  homeRunLineProb?: number;
  actualTotal?: number;
  actualMargin?: number;
}

interface CalibrationTabProps {
  modelState: ModelStateDoc;
  gameCards: Record<string, GameDoc>;
  calibrationBins: CalibrationBin[];
  confidenceDistribution: ConfidencePoint[];
  calibrationCurve: CurvePoint[];
  totalsMetrics: { n: number; mae: number; rmse: number; bias: number };
  runLineMetrics: { n: number; auc: number; brier: number; accuracy: number };
  moneylineTotal: number;
  moneylineCorrect: number;
  moneylineAccuracy: number;
}

type CalibView = "moneyline" | "totals" | "runline";

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
  decimals = 3,
}: {
  label: string;
  value: number;
  color: string;
  sub?: string;
  decimals?: number;
}) {
  return (
    <div className="rounded-2xl border border-border bg-card p-6">
      <div className="text-center text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <div className="mt-3 text-center text-4xl font-bold tabular-nums" style={{ color }}>
        {Number.isFinite(value) ? value.toFixed(decimals) : "—"}
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

export function CalibrationTab({
  modelState,
  gameCards,
  calibrationBins,
  confidenceDistribution,
  calibrationCurve,
  totalsMetrics: fullRangeTotalsMetrics,
  runLineMetrics: fullRangeRunLineMetrics,
  moneylineTotal,
  moneylineCorrect,
  moneylineAccuracy,
}: CalibrationTabProps) {
  const record = modelState.todaysRecord;
  const trainStart = "2022-03-15";
  const [view, setView] = useState<CalibView>("moneyline");
  const [startDate, setStartDate] = useState(trainStart);
  const [endDate, setEndDate] = useState(modelState.asOfDate);
  const [search, setSearch] = useState("");

  // ── Client-side calibration game filtering ──────────────────────────────
  // Convert the flat gameCards map to an array filtered by date range.
  const allCalibGames = useMemo(() => {
    const cards = Object.values(gameCards);
    return cards.filter((g) => {
      if (!g.date) return false;
      if (g.date < startDate || g.date > endDate) return false;
      // Only include completed games with a valid winner
      return g.winner === "home" || g.winner === "away";
    });
  }, [gameCards, startDate, endDate]);

  // Compute calibration metrics over the filtered date range.
  const computedMetrics = useMemo(() => {
    const preds: number[] = [];
    const labels: number[] = [];
    let tN = 0;
    let tAbs = 0;
    let tSq = 0;
    let tBias = 0;
    const rlPreds: number[] = [];
    const rlLabels: number[] = [];

    for (const g of allCalibGames) {
      preds.push(g.pickProb);
      labels.push(g.isCorrect ? 1 : 0);

      const predictedTotal = (g.runProjection as { total?: number } | undefined)?.total;
      if (typeof predictedTotal === "number") {
        tN += 1;
        const actual = (g.away.score ?? 0) + (g.home.score ?? 0);
        const err = predictedTotal - actual;
        tAbs += Math.abs(err);
        tSq += err * err;
        tBias += err;
      }

      const homeRunLineProb = (g.runProjection as { homeRunLineProb?: number } | undefined)
        ?.homeRunLineProb;
      if (typeof homeRunLineProb === "number") {
        const margin = (g.home.score ?? 0) - (g.away.score ?? 0);
        rlPreds.push(homeRunLineProb);
        rlLabels.push(margin >= 2 ? 1 : 0);
      }
    }

    // For the full range, use the pre-computed summary from CDN props
    const isFullRange =
      startDate <= trainStart && endDate >= modelState.asOfDate;

    if (isFullRange) {
      return {
        metrics: {
          auc: modelState.auc,
          brier: modelState.brier,
          logLoss: modelState.logLoss,
          ece: modelState.ece,
          bins: calibrationBins,
          confidenceDistribution,
          calibrationCurve,
        },
        totalsMetrics: fullRangeTotalsMetrics,
        runLineMetrics: fullRangeRunLineMetrics,
        total: moneylineTotal,
        correct: moneylineCorrect,
        accuracy: moneylineAccuracy,
      };
    }

    // For narrower ranges, compute metrics from the filtered games
    const total = allCalibGames.length;
    const correct = allCalibGames.filter((g) => g.isCorrect).length;

    // Simple AUC via trapezoidal rule (replaces server-side computeAuc)
    let auc = 0;
    if (preds.length >= 2) {
      const sorted = preds
        .map((p, i) => ({ p, l: labels[i] }))
        .sort((a, b) => a.p - b.p);
      const pos = sorted.filter((x) => x.l === 1).length;
      const neg = sorted.length - pos;
      if (pos > 0 && neg > 0) {
        let rankSum = 0;
        let i = 0;
        while (i < sorted.length) {
          let j = i;
          while (j < sorted.length && sorted[j].p === sorted[i].p) j++;
          const avgRank = (i + j - 1) / 2 + 1;
          for (let k = i; k < j; k++) {
            if (sorted[k].l === 1) rankSum += avgRank;
          }
          i = j;
        }
        auc = (rankSum - (pos * (pos + 1)) / 2) / (pos * neg);
      }
    }

    const brier =
      preds.length > 0
        ? preds.reduce((s, p, i) => s + (p - labels[i]) ** 2, 0) / preds.length
        : 0;
    const logLoss =
      preds.length > 0
        ? preds.reduce((s, p, i) => {
            const pClamped = Math.min(Math.max(p, 1e-15), 1 - 1e-15);
            return s - (labels[i] * Math.log(pClamped) + (1 - labels[i]) * Math.log(1 - pClamped));
          }, 0) / preds.length
        : 0;
    // ECE
    const nBins = 8;
    const bins: CalibrationBin[] = [];
    for (let b = 0; b < nBins; b++) {
      const lo = b / nBins;
      const hi = (b + 1) / nBins;
      const inBin = preds.map((p, i) => ({ p, l: labels[i] })).filter((x) => x.p >= lo && x.p < hi);
      if (inBin.length > 0) {
        const meanP = inBin.reduce((s, x) => s + x.p, 0) / inBin.length;
        const meanA = inBin.reduce((s, x) => s + x.l, 0) / inBin.length;
        bins.push({
          label: `${(lo * 100).toFixed(0)}–${(hi * 100).toFixed(0)}%`,
          meanPredicted: meanP,
          meanActual: meanA,
          count: inBin.length,
          gap: meanP - meanA,
        });
      }
    }

    return {
      metrics: {
        auc,
        brier,
        logLoss,
        ece: bins.length > 0 ? bins.reduce((s, b) => s + Math.abs(b.gap) * b.count, 0) / total : 0,
        bins,
        confidenceDistribution: confidenceDistribution,
        calibrationCurve: calibrationCurve,
      },
      totalsMetrics: {
        n: tN,
        mae: tN > 0 ? tAbs / tN : 0,
        rmse: tN > 0 ? Math.sqrt(tSq / tN) : 0,
        bias: tN > 0 ? tBias / tN : 0,
      },
      runLineMetrics: {
        n: rlPreds.length,
        auc: 0,
        brier: 0,
        accuracy: rlPreds.length > 0
          ? rlPreds.filter((p, i) => (p >= 0.5 ? 1 : 0) === rlLabels[i]).length / rlPreds.length
          : 0,
      },
      total,
      correct,
      accuracy: total > 0 ? correct / total : 0,
    };
  }, [
    allCalibGames, startDate, endDate, modelState, calibrationBins,
    confidenceDistribution, calibrationCurve, fullRangeTotalsMetrics,
    fullRangeRunLineMetrics, moneylineTotal, moneylineCorrect, moneylineAccuracy,
  ]);

  const metrics = computedMetrics.metrics;
  const totalsMetrics = computedMetrics.totalsMetrics;
  const runLineMetrics = computedMetrics.runLineMetrics;
  const bins: CalibrationBin[] = metrics?.bins ?? [];
  const distribution = metrics?.confidenceDistribution ?? [];
  const curve = metrics?.calibrationCurve ?? [];
  const auc = metrics?.auc ?? modelState.auc;
  const brier = metrics?.brier ?? modelState.brier;
  const logLoss = metrics?.logLoss ?? modelState.logLoss;
  const ece = metrics?.ece ?? modelState.ece;

  const allGames = allCalibGames as unknown as GameResultRow[];
  const totalGames = computedMetrics.total;
  const correctCount = computedMetrics.correct;
  const q = search.trim().toLowerCase();
  const filteredGames = q
    ? allGames.filter((g) =>
        `${g.away.name} ${g.away.abbrev} ${g.home.name} ${g.home.abbrev}`.toLowerCase().includes(q),
      )
    : allGames;
  const visibleGames = filteredGames;

  const resetPagination = () => {
    // No-op: client-side, all games are already loaded
  };

  const onStartChange = (v: string) => {
    setStartDate(v);
    if (v && endDate && v > endDate) setEndDate(v);
  };
  const onEndChange = (v: string) => {
    setEndDate(v);
    if (v && startDate && v < startDate) setStartDate(v);
  };

  const views: { id: CalibView; label: string }[] = [
    { id: "moneyline", label: "Moneyline" },
    { id: "totals", label: "Game Totals" },
    { id: "runline", label: "Run Lines (-1.5 / +1.5)" },
  ];

  return (
    <div className="flex flex-col gap-5">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold tracking-tight">Model Calibration Dashboard</h2>
        <span className="mt-3 inline-block rounded-full border border-border bg-card px-3 py-1 text-xs text-muted-foreground">
          n = {formatNumber(totalGames)} games in range · Trained {formatTrainedAt(modelState.trainedAt)} ET
        </span>
        <p className="mt-3 text-sm text-muted-foreground">
          Assessing prediction reliability and accuracy for moneyline, game totals, and run lines —
          computed on one side per game.
        </p>

        {/* View toggle */}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          {views.map((v) => (
            <button
              key={v.id}
              type="button"
              onClick={() => setView(v.id)}
              className={cn(
                "cursor-pointer rounded-full px-4 py-1.5 text-xs font-semibold transition-colors",
                view === v.id
                  ? "bg-primary text-primary-foreground"
                  : "border border-border bg-card text-muted-foreground hover:text-foreground",
              )}
            >
              {v.label}
            </button>
          ))}
        </div>

        {/* Date range selector */}
        <div className="mt-3 flex flex-wrap items-center gap-2 rounded-2xl border border-border bg-card p-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Range
          </span>
          <input
            type="date"
            value={startDate}
            min={trainStart}
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
            {formatPct(computedMetrics.accuracy, 1)} accuracy
          </span>
        </div>
      </div>

      {/* Today's record */}
      {view === "moneyline" && record && record.total > 0 && (
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

      {/* ----- Moneyline view ----- */}
      {view === "moneyline" && (
        <>
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
        </>
      )}

      {/* ----- Totals view ----- */}
      {view === "totals" && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard label="Mean Abs. Error" value={totalsMetrics?.mae ?? 0} color="#22d3ee" decimals={2} sub="Runs" />
            <MetricCard label="RMSE" value={totalsMetrics?.rmse ?? 0} color="#34d399" decimals={2} sub="Runs" />
            <MetricCard label="Bias" value={totalsMetrics?.bias ?? 0} color="#fcd34d" decimals={2} sub="Predicted − actual" />
            <MetricCard label="Games" value={totalsMetrics?.n ?? 0} color="#c084fc" decimals={0} sub="With run projection" />
          </div>
          <p className="-mt-1 text-xs text-muted-foreground">
            Predicted combined runs (both teams) versus the actual final score total over the selected
            range.
          </p>
        </>
      )}

      {/* ----- Run line view ----- */}
      {view === "runline" && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard label="Run-Line AUC" value={runLineMetrics?.auc ?? 0} color="#22d3ee" />
            <MetricCard label="Brier Score" value={runLineMetrics?.brier ?? 0} color="#34d399" />
            <MetricCard label="Cover Accuracy" value={runLineMetrics?.accuracy ?? 0} color="#fcd34d" sub="Home −1.5 covers" />
            <MetricCard label="Games" value={runLineMetrics?.n ?? 0} color="#c084fc" decimals={0} sub="With run projection" />
          </div>
          <p className="-mt-1 text-xs text-muted-foreground">
            Home team covers −1.5 when it wins by 2+ runs; away team covers +1.5 otherwise. One side per
            game, calibrated probabilities.
          </p>
        </>
      )}

      {/* Game history table (view-specific columns) */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-foreground">
              {view === "moneyline"
                ? "Game History — Predicted vs Actual"
                : view === "totals"
                  ? "Game History — Predicted vs Actual Totals"
                  : "Game History — Run Line Results"}
            </h3>
            <p className="mt-1 text-xs text-muted-foreground">
              {view === "moneyline"
                ? `${formatNumber(totalGames)} games · ${formatNumber(correctCount)} correct picks (${formatPct(totalGames > 0 ? correctCount / totalGames : 0, 1)})`
                : `${formatNumber(view === "totals" ? totalsMetrics?.n ?? 0 : runLineMetrics?.n ?? 0)} games with projections`}
            </p>
          </div>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
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
                {view === "moneyline" && (
                  <>
                    <th className="pb-2 pr-4 text-right font-medium">Final</th>
                    <th className="pb-2 pr-4 text-right font-medium">Predicted</th>
                    <th className="pb-2 pr-4 text-right font-medium">Actual</th>
                    <th className="pb-2 text-right font-medium">Result</th>
                  </>
                )}
                {view === "totals" && (
                  <>
                    <th className="pb-2 pr-4 text-right font-medium">Predicted Total</th>
                    <th className="pb-2 pr-4 text-right font-medium">Actual Total</th>
                    <th className="pb-2 text-right font-medium">Diff</th>
                  </>
                )}
                {view === "runline" && (
                  <>
                    <th className="pb-2 pr-4 text-right font-medium">Home Cover Prob</th>
                    <th className="pb-2 pr-4 text-right font-medium">Margin</th>
                    <th className="pb-2 text-right font-medium">Result</th>
                  </>
                )}
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
                    {view === "moneyline" && (
                      <>
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
                      </>
                    )}
                    {view === "totals" && (
                      <>
                        <td className="whitespace-nowrap py-2 pr-4 text-right tabular-nums text-foreground">
                          {g.predictedTotal !== undefined ? g.predictedTotal.toFixed(1) : "—"}
                        </td>
                        <td className="whitespace-nowrap py-2 pr-4 text-right tabular-nums text-foreground">
                          {g.actualTotal ?? "—"}
                        </td>
                        <td className="whitespace-nowrap py-2 text-right tabular-nums">
                          {g.predictedTotal !== undefined && g.actualTotal !== undefined ? (
                            <span className={cn("font-semibold", g.predictedTotal - g.actualTotal >= 0 ? "text-rose-400" : "text-emerald-400")}>
                              {g.predictedTotal - g.actualTotal >= 0 ? "+" : ""}
                              {(g.predictedTotal - g.actualTotal).toFixed(1)}
                            </span>
                          ) : (
                            "—"
                          )}
                        </td>
                      </>
                    )}
                    {view === "runline" && (
                      <>
                        <td className="whitespace-nowrap py-2 pr-4 text-right tabular-nums text-foreground">
                          {g.homeRunLineProb !== undefined ? formatPct(g.homeRunLineProb, 1) : "—"}
                        </td>
                        <td className="whitespace-nowrap py-2 pr-4 text-right tabular-nums text-foreground">
                          {g.actualMargin !== undefined ? (g.actualMargin >= 0 ? "+" : "") + g.actualMargin : "—"}
                        </td>
                        <td className="whitespace-nowrap py-2 text-right">
                          {g.actualMargin === undefined ? (
                            "—"
                          ) : g.actualMargin >= 2 ? (
                            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs font-semibold text-emerald-300">
                              <Check className="size-3" /> Home covers
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 rounded-full bg-blue-500/15 px-2 py-0.5 text-xs font-semibold text-blue-300">
                              Away covers
                            </span>
                          )}
                        </td>
                      </>
                    )}
                  </tr>
                );
              })}
              {visibleGames.length === 0 && (
                <tr>
                  <td
                    colSpan={view === "moneyline" ? 6 : 5}
                    className="py-8 text-center text-sm text-muted-foreground"
                  >
                    No games found in this range.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
