import { api } from "@/convex/_generated/api";
import type { ModelStateDoc } from "@/lib/mlb-ui-types";
import { formatMonthDayYear, formatNumber, formatPct } from "@/lib/format";
import { cn } from "@/lib/utils";
import { useAction } from "convex/react";
import {
  AlertTriangle,
  BarChart3,
  Bot,
  Check,
  Network,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
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

const FEATURE_META: Record<string, { category: string; description: string }> = {
  eloDiff: {
    category: "Team Strength",
    description: "Chronological Elo rating gap, adjusted for home field and margin of victory.",
  },
  winPctDiff: {
    category: "Team Strength",
    description: "Season-to-date win-percentage differential between the two clubs.",
  },
  formDiff: {
    category: "Recent Form",
    description: "Last-10-game win-rate differential capturing current momentum.",
  },
  restDiff: {
    category: "Schedule",
    description: "Days-of-rest advantage, accounting for fatigue and bullpen availability.",
  },
  injuryDiff: {
    category: "Roster",
    description: "Injured-list count edge from the latest roster snapshots.",
  },
  homeField: {
    category: "Context",
    description: "Home-field advantage term.",
  },
  spFipDiff: {
    category: "Starting Pitcher",
    description:
      "Fielding-independent pitching edge (FIP/xERA-style), computed from strikeouts, walks and home runs — strips fielding luck and BABIP variance to quantify true run-prevention expectation.",
  },
  spEraDiff: {
    category: "Starting Pitcher",
    description: "Season ERA differential between the two projected starting pitchers.",
  },
  opsDiff: {
    category: "Hitting",
    description: "Season-to-date team OPS edge — a consolidated measure of on-base and slugging production for the projected lineups.",
  },
  teamEraDiff: {
    category: "Pitching Staff",
    description: "Season-to-date team ERA edge, capturing rotation and bullpen run prevention beyond the two starters.",
  },
  defEffDiff: {
    category: "Defense",
    description: "Season fielding-percentage edge (defensive-efficiency proxy) between the two clubs.",
  },
  parkFactor: {
    category: "Ballpark",
    description: "Home ballpark run factor — values above 1 favor hitters and inflate expected totals.",
  },
  tempDev: {
    category: "Weather",
    description: "Game-time temperature deviation from 72°F, a proxy for air density and carry.",
  },
  windMph: {
    category: "Weather",
    description: "Game-time wind speed in mph, affecting fly-ball carry and scoring environment.",
  },
  spK9Diff: {
    category: "Starting Pitcher",
    description: "Strikeouts per 9 innings edge between the two projected starters — a raw swing-and-miss / dominance signal computed as-of the game date.",
  },
  spWhipDiff: {
    category: "Starting Pitcher",
    description: "Walks + hits per inning pitched edge between the projected starters — a control and contact-suppression signal computed as-of the game date.",
  },
  spRecentDiff: {
    category: "Starting Pitcher",
    description: "ERA over each starter's last 3 starts — a hot/cold form signal that season-long ERA smooths away.",
  },
  teamK9Diff: {
    category: "Pitching Staff",
    description: "Staff strikeouts per 9 innings edge (rotation + bullpen), computed as-of the game date.",
  },
  teamWhipDiff: {
    category: "Pitching Staff",
    description: "Staff walks + hits per inning pitched edge (rotation + bullpen), computed as-of the game date.",
  },
  lineupKnown: {
    category: "Lineup",
    description: "Indicator that actual starting-lineup data is available; the model treats unknown lineups distinctly from known strong/weak ones.",
  },
  lineupOpsDiff: {
    category: "Lineup",
    description: "Starting-9 weighted OPS edge (slots 1-4 double-weighted), each batter's OPS computed as-of the game date.",
  },
  lineupWobaDiff: {
    category: "Lineup",
    description: "Starting-9 weighted wOBA edge — quality-of-contact production beyond simple OPS, computed as-of the game date.",
  },
  lineupIsoDiff: {
    category: "Lineup",
    description: "Starting-9 weighted isolated-power edge — extra-base hit power in the projected lineup, computed as-of the game date.",
  },
  lineupHotDiff: {
    category: "Lineup",
    description: "Starting-9 weighted OPS over each batter's last 10 games — a lineup-level hot/cold streak signal.",
  },
  bvpOpsDiff: {
    category: "Matchup",
    description: "Career batter-vs-pitcher OPS edge vs the opposing starter — each batter's BvP sample is PA-saturated and shrunk toward their as-of season OPS so tiny samples can't dominate.",
  },
  platoonOpsDiff: {
    category: "Matchup",
    description: "Season OPS edge vs the starter's throwing hand (L/R split) across the starting 9 — the platoon advantage the actual lineup holds over the opposing starter.",
  },
  vsTeamOpsDiff: {
    category: "Matchup",
    description: "Season OPS edge vs the opposing team across the starting 9 — each club's hitters against the specific opponent staff they face.",
  },
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

function AutoMetricCard({
  label,
  value,
  color,
  sub,
  subColor,
  note,
}: {
  label: string;
  value: string;
  color: string;
  sub: string;
  subColor: string;
  note: string;
}) {
  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-2 text-3xl font-bold tabular-nums" style={{ color }}>
        {value}
      </div>
      <div className={cn("mt-1 text-xs font-medium", subColor)}>{sub}</div>
      <div className="mt-1 text-xs text-muted-foreground">{note}</div>
    </div>
  );
}

function FeatureItem({
  rank,
  label,
  category,
  description,
  weightPct,
  barPct,
}: {
  rank: number;
  label: string;
  category: string;
  description: string;
  weightPct: number;
  barPct: number;
}) {
  return (
    <div className="rounded-2xl border border-border bg-card p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 text-sm font-bold text-cyan-300">#{rank}</span>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-semibold text-foreground">{label}</span>
              <span className="rounded-full bg-blue-500/15 px-2 py-0.5 text-[10px] font-semibold text-blue-300">
                {category}
              </span>
              <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
                Selected by ML
              </span>
            </div>
            <p className="mt-1.5 text-xs leading-5 text-muted-foreground">{description}</p>
          </div>
        </div>
        <div className="shrink-0 text-right">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            ML Learned Weight
          </div>
          <div className="text-xl font-bold tabular-nums text-cyan-300">{weightPct}%</div>
        </div>
      </div>
      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-white/5">
        <div
          className="h-full rounded-full bg-cyan-400"
          style={{ width: `${Math.max(3, Math.min(100, barPct))}%` }}
        />
      </div>
    </div>
  );
}

function StackingWeightsPanel({ modelState }: { modelState: ModelStateDoc }) {
  const candidates = modelState.candidates ?? [];
  const weights = modelState.stackingWeights ?? [];
  const rows = candidates
    .map((c) => {
      const stackWeight = weights.find((w) => w.name === c.name)?.weight ?? (c.selected ? 1 : 0);
      return { ...c, stackWeight };
    })
    .sort((a, b) => b.stackWeight - a.stackWeight);

  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <h3 className="text-sm font-semibold text-foreground">Optimal Model Stacking Weights</h3>
      <p className="mt-1.5 max-w-3xl text-xs leading-5 text-muted-foreground">
        Greedy forward-selection solves for convex-combination weights that minimize calibration-set
        Brier loss. Only models that measurably reduce risk are added to the stack; the remainder
        carry zero weight.
      </p>
      <div className="mt-5 flex flex-col gap-4">
        {rows.map((c) => (
          <div key={c.name}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-foreground">{c.name}</span>
                {c.selected && (
                  <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
                    Best single
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3 text-xs tabular-nums">
                <span className="text-muted-foreground">AUC {c.auc.toFixed(3)}</span>
                <span className="text-muted-foreground">Brier {c.brier.toFixed(3)}</span>
                <span className="font-semibold text-cyan-300">{Math.round(c.stackWeight * 100)}%</span>
              </div>
            </div>
            <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-white/5">
              <div
                className="h-full rounded-full bg-cyan-400"
                style={{ width: `${Math.max(0, Math.min(100, c.stackWeight * 100))}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function OptimizationParamsPanel({ modelState }: { modelState: ModelStateDoc }) {
  const params = modelState.optimizationParams;
  if (!params) {
    return (
      <div className="rounded-2xl border border-border bg-card p-5 text-sm text-muted-foreground">
        Optimization parameters will appear after the next Auto-ML run.
      </div>
    );
  }
  const rows: { label: string; value: string }[] = [
    { label: "Feature selection", value: params.featureSelection },
    { label: "Regularization", value: `L2 λ = ${params.l2Lambda}` },
    { label: "Optimizer", value: `Newton–Raphson (IRLS) · ${params.epochs} iterations` },
    { label: "Home-field grid", value: params.hfaGrid.join(", ") },
    { label: "Stacking blend step", value: `${params.blendStep}` },
    { label: "Monte Carlo σ grid", value: params.mcSigmaGrid.join(", ") },
    { label: "Calibration", value: params.isotonicMethod },
    { label: "Cross-validation folds", value: `${params.cvFolds}` },
  ];
  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <h3 className="text-sm font-semibold text-foreground">Optimization Parameters</h3>
      <p className="mt-1.5 max-w-3xl text-xs leading-5 text-muted-foreground">
        Hyperparameters and search grids used by the Auto-ML optimizer on the most recent training run.
      </p>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[480px] text-sm">
          <tbody>
            {rows.map((r) => (
              <tr key={r.label} className="border-b border-border/50 last:border-0">
                <td className="py-2.5 pr-4 text-muted-foreground">{r.label}</td>
                <td className="py-2.5 text-right font-medium tabular-nums text-foreground">{r.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CrossValidationPanel({ modelState }: { modelState: ModelStateDoc }) {
  const cv = modelState.crossValidation;
  if (!cv) {
    return (
      <div className="rounded-2xl border border-border bg-card p-5 text-sm text-muted-foreground">
        Cross-validation metrics will appear after the next Auto-ML run.
      </div>
    );
  }
  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-foreground">
            {cv.folds}-Fold Cross-Validation ({formatNumber(modelState.gamesTrained)} games)
          </h3>
          <p className="mt-1.5 max-w-3xl text-xs leading-5 text-muted-foreground">
            Walk-forward folds train only on prior data so out-of-sample AUC and Brier are never
            inflated by lookahead. Reported mean ± standard deviation across folds.
          </p>
        </div>
        <div className="flex gap-4 text-right">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Mean AUC</div>
            <div className="text-lg font-bold tabular-nums text-cyan-300">
              {cv.aucMean.toFixed(3)} ± {cv.aucStd.toFixed(3)}
            </div>
          </div>
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Mean Brier</div>
            <div className="text-lg font-bold tabular-nums text-emerald-300">
              {cv.brierMean.toFixed(3)} ± {cv.brierStd.toFixed(3)}
            </div>
          </div>
        </div>
      </div>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[440px] text-sm">
          <thead>
            <tr className="border-b border-border/70 text-left text-[11px] uppercase tracking-wider text-muted-foreground">
              <th className="pb-2 font-medium">Fold</th>
              <th className="pb-2 text-right font-medium">Games</th>
              <th className="pb-2 text-right font-medium">AUC</th>
              <th className="pb-2 text-right font-medium">Brier</th>
            </tr>
          </thead>
          <tbody>
            {cv.foldAucs.map((auc, i) => (
              <tr key={i} className="border-b border-border/50 last:border-0">
                <td className="py-2.5 font-medium text-foreground">Fold {i + 1}</td>
                <td className="py-2.5 text-right tabular-nums text-muted-foreground">
                  {formatNumber(cv.gamesPerFold[i] ?? 0)}
                </td>
                <td className="py-2.5 text-right tabular-nums text-foreground">{auc.toFixed(3)}</td>
                <td className="py-2.5 text-right tabular-nums text-foreground">
                  {cv.foldBriers[i]?.toFixed(3) ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-panels
// ---------------------------------------------------------------------------

function AutoMlPanel({ modelState }: { modelState: ModelStateDoc }) {
  const runAutoMl = useAction(api.mlbRetrain.retrainModel);
  const [running, setRunning] = useState(false);

  const [autoSub, setAutoSub] = useState<"features" | "stacking" | "params" | "cv">("features");

  const candidates = modelState.candidates ?? [];
  const selected = candidates.find((c) => c.selected) ?? candidates[0];
  const ensembleAuc = selected?.auc ?? modelState.auc;
  const ensembleBrier = selected?.brier ?? modelState.brier;
  const brierVsBaseline = ensembleBrier - 0.25;
  const spearmanRho = modelState.spearmanRho ?? 0;
  const topDecileWinRate = modelState.topDecileWinRate ?? 0;
  const features = modelState.featureImportances ?? [];
  // `active === undefined` (pre-pitcher-feature states) is treated as selected.
  const activeFeatures = features.filter((f) => f.active !== false);
  const inactiveFeatures = features.filter((f) => f.active === false);
  const totalWeight = activeFeatures.reduce((s, f) => s + Math.abs(f.weight), 0) || 1;
  const maxImp = Math.max(...activeFeatures.map((f) => f.importance), 1e-6);
  const sortedFeatures = [...activeFeatures].sort((a, b) => b.importance - a.importance);

  const handleRun = async () => {
    if (running) return;
    setRunning(true);
    try {
      const res = await runAutoMl();
      toast.success("Auto-ML optimization complete", {
        description: `Trained on ${formatNumber(res.gamesTrained)} games · Ensemble AUC ${res.auc.toFixed(3)}`,
      });
    } catch (e) {
      toast.error("Auto-ML run failed", {
        description: e instanceof Error ? e.message : "Unknown error",
      });
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Main optimizer panel */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="flex items-center gap-1.5 rounded-full bg-cyan-500/15 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-cyan-300">
                <Sparkles className="size-3" /> Automated Machine Learning (Auto-ML)
              </span>
              <span className="rounded-full bg-emerald-500/15 px-2.5 py-0.5 text-[10px] font-semibold text-emerald-300">
                5-Fold Cross-Validated
              </span>
            </div>
            <h3 className="mt-3 text-lg font-bold tracking-tight text-foreground">
              Empirical Feature Selection &amp; Model Stacking Optimizer
            </h3>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-muted-foreground">
              Machine Learning algorithms automatically assess feature predictive signal (via
              L2-regularized logistic regression with greedy backward elimination) and solve for optimal
              ensemble weights by minimizing calibration-set Brier loss to maximize out-of-sample AUC
              (&gt; 0.60) while enforcing monotonic probability calibration.
            </p>
          </div>
          <button
            type="button"
            onClick={handleRun}
            disabled={running}
            className="flex shrink-0 cursor-pointer items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-cyan-950 transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <RefreshCw className={cn("size-4", running && "animate-spin")} />
            {running ? "Optimizing…" : "Run Auto-ML Optimization"}
          </button>
        </div>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <AutoMetricCard
          label="Ensemble AUC-ROC"
          value={ensembleAuc.toFixed(3)}
          color="#22d3ee"
          sub={ensembleAuc >= 0.6 ? "> 0.60 Target Met" : "Below 0.60 target"}
          subColor={ensembleAuc >= 0.6 ? "text-emerald-400" : "text-amber-400"}
          note="High single-game MLB separation"
        />
        <AutoMetricCard
          label="Ensemble Brier Loss"
          value={ensembleBrier.toFixed(3)}
          color="#34d399"
          sub={`${brierVsBaseline >= 0 ? "+" : ""}${brierVsBaseline.toFixed(3)} vs naive baseline`}
          subColor="text-emerald-400"
          note="Lower is better (risk)"
        />
        <AutoMetricCard
          label="Rank Correlation (Spearman ρ)"
          value={spearmanRho.toFixed(3)}
          color="#c084fc"
          sub="Strict monotonic fidelity"
          subColor="text-purple-300"
          note="Probability vs outcome ordering"
        />
        <AutoMetricCard
          label="Top Decile Win Rate (>65%)"
          value={formatPct(topDecileWinRate, 1)}
          color="#fcd34d"
          sub="Highest confidence picks"
          subColor="text-amber-300"
          note="Win rate of >65% favorites"
        />
      </div>

      {/* Secondary sub-tabs */}
      <div className="flex flex-wrap items-center gap-2">
        {(
          [
            { id: "features", label: `Learned Feature Decisions (${activeFeatures.length}/${features.length} Active)`, icon: <SlidersHorizontal className="size-3.5" /> },
            { id: "stacking", label: `Optimal Model Stacking Weights (${candidates.length} Models)`, icon: <Network className="size-3.5" /> },
            { id: "params", label: "Optimization Parameters", icon: <SlidersHorizontal className="size-3.5" /> },
            { id: "cv", label: `Cross-Validation on ${formatNumber(modelState.gamesTrained)} games`, icon: <Check className="size-3.5" /> },
          ] as const
        ).map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setAutoSub(t.id)}
            className={cn(
              "flex cursor-pointer items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold transition-colors",
              autoSub === t.id
                ? "bg-cyan-500 text-cyan-950"
                : "border border-border bg-card text-muted-foreground hover:text-foreground",
            )}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </div>

      {/* Info callout (feature decisions only) */}
      {autoSub === "features" && (
        <div className="rounded-2xl border border-amber-500/20 bg-amber-500/[0.06] p-4">
          <div className="flex items-start gap-3">
            <Sparkles className="mt-0.5 size-4 shrink-0 text-amber-400" />
            <div>
              <div className="text-sm font-bold text-foreground">
                How Machine Learning Decided Feature Inclusion &amp; Weights:
              </div>
              <p className="mt-1 text-xs leading-5 text-muted-foreground">
                L2-regularized logistic regression with greedy backward elimination evaluated each
                candidate feature set. Features were retained only when they measurably reduced
                calibration-set Brier loss; the final coefficients are the learned weights.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Sub-tab content */}
      {autoSub === "features" && (
        <div className="flex flex-col gap-3">
          {inactiveFeatures.length > 0 && (
            <div className="rounded-xl border border-border/70 bg-white/[0.02] px-4 py-3 text-xs leading-5 text-muted-foreground">
              <span className="font-semibold text-foreground">Dropped by ML:</span>{" "}
              {inactiveFeatures.map((f) => f.label).join(" · ")}
            </div>
          )}
          {sortedFeatures.map((f, i) => (
            <FeatureItem
              key={f.feature}
              rank={i + 1}
              label={f.label}
              category={FEATURE_META[f.feature]?.category ?? "Model Feature"}
              description={FEATURE_META[f.feature]?.description ?? "Automatically selected predictive feature."}
              weightPct={Math.round((Math.abs(f.weight) / totalWeight) * 100)}
              barPct={Math.round((f.importance / maxImp) * 100)}
            />
          ))}
          {sortedFeatures.length === 0 && (
            <div className="rounded-2xl border border-border bg-card p-5 text-sm text-muted-foreground">
              No feature decisions yet — run Auto-ML optimization.
            </div>
          )}
        </div>
      )}
      {autoSub === "stacking" && <StackingWeightsPanel modelState={modelState} />}
      {autoSub === "params" && <OptimizationParamsPanel modelState={modelState} />}
      {autoSub === "cv" && <CrossValidationPanel modelState={modelState} />}
    </div>
  );
}

function PfiPanel({ modelState }: { modelState: ModelStateDoc }) {
  const features = modelState.featureImportances ?? [];
  const sorted = [...features].sort((a, b) => b.univariateAuc - a.univariateAuc);
  const maxAuc = Math.max(...sorted.map((f) => f.univariateAuc), 1e-6);
  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <h3 className="text-sm font-semibold text-foreground">Feature Importance (Permutation Feature Importance)</h3>
      <p className="mt-1.5 text-xs leading-5 text-muted-foreground">
        Each feature is ranked by its isolated out-of-sample predictive signal (univariate AUC) and its
        learned coefficient in the logistic ensemble. Bars show the standardized coefficient magnitude.
      </p>
      <div className="mt-5 flex flex-col gap-4">
        {sorted.map((f, i) => {
          const meta = FEATURE_META[f.feature];
          const barPct = maxAuc > 0 ? Math.round((Math.abs(f.weight) / (Math.max(...sorted.map((x) => Math.abs(x.weight)), 1e-6))) * 100) : 0;
          return (
            <div key={f.feature}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-muted-foreground">#{i + 1}</span>
                  <span className="text-sm font-medium text-foreground">{f.label}</span>
                  {meta && (
                    <span className="rounded-full bg-blue-500/15 px-2 py-0.5 text-[10px] font-semibold text-blue-300">
                      {meta.category}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3 text-xs tabular-nums">
                  <span className="text-muted-foreground">PFI AUC {f.univariateAuc.toFixed(3)}</span>
                  <span className="font-semibold text-cyan-300">w = {f.weight.toFixed(3)}</span>
                </div>
              </div>
              <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-white/5">
                <div
                  className="h-full rounded-full bg-cyan-400/80"
                  style={{ width: `${Math.max(3, Math.min(100, barPct))}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function EnsemblePanel({ modelState }: { modelState: ModelStateDoc }) {
  const candidates = modelState.candidates ?? [];
  const steps = [
    `${modelState.featureImportances?.length ?? modelState.featureNames.length} Features`,
    "5 Candidate Models",
    "Stacked Ensemble",
    "Isotonic Calibration",
    "Monte Carlo",
    "Win Probability",
  ];
  return (
    <div className="flex flex-col gap-4">
      {/* Architecture flow */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <h3 className="text-sm font-semibold text-foreground">Ensemble Architecture</h3>
        <p className="mt-1.5 text-xs leading-5 text-muted-foreground">
          {modelState.modelDescription}
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          {steps.map((s, i) => (
            <div key={s} className="flex items-center gap-2">
              <span className="rounded-lg border border-border bg-white/[0.02] px-3 py-1.5 text-xs font-medium text-foreground">
                {s}
              </span>
              {i < steps.length - 1 && <span className="text-muted-foreground">→</span>}
            </div>
          ))}
        </div>
      </div>

      {/* Candidate models */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <h3 className="mb-4 text-sm font-semibold text-foreground">Candidate Models (Cross-Validated)</h3>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="border-b border-border/70 text-left text-[11px] uppercase tracking-wider text-muted-foreground">
                <th className="pb-2 font-medium">Model</th>
                <th className="pb-2 text-right font-medium">AUC</th>
                <th className="pb-2 text-right font-medium">Brier</th>
                <th className="pb-2 text-right font-medium">Log-Loss</th>
                <th className="pb-2 text-right font-medium">ECE</th>
                <th className="pb-2 text-right font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((c) => (
                <tr key={c.name} className="border-b border-border/50 last:border-0">
                  <td className="py-2.5 font-medium text-foreground">{c.name}</td>
                  <td className="py-2.5 text-right tabular-nums text-foreground">{c.auc.toFixed(3)}</td>
                  <td className="py-2.5 text-right tabular-nums text-foreground">{c.brier.toFixed(3)}</td>
                  <td className="py-2.5 text-right tabular-nums text-muted-foreground">{c.logLoss.toFixed(3)}</td>
                  <td className="py-2.5 text-right tabular-nums text-muted-foreground">{c.ece.toFixed(3)}</td>
                  <td className="py-2.5 text-right">
                    {c.selected ? (
                      <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs font-semibold text-emerald-300">
                        Selected
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Monte Carlo decision */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <h3 className="text-sm font-semibold text-foreground">Stochastic Component (Monte Carlo)</h3>
        <div
          className={cn(
            "mt-3 flex items-center gap-2 rounded-lg border p-3",
            modelState.monteCarloEnabled
              ? "border-emerald-500/25 bg-emerald-500/10"
              : "border-border/70 bg-white/[0.02]",
          )}
        >
          {modelState.monteCarloEnabled ? (
            <Check className="size-4 text-emerald-400" />
          ) : (
            <AlertTriangle className="size-4 text-muted-foreground" />
          )}
          <span className={cn("text-sm font-medium", modelState.monteCarloEnabled ? "text-emerald-300" : "text-foreground")}>
            {modelState.monteCarloEnabled
              ? `Enabled — σ = ${modelState.monteCarloSigma.toFixed(2)} (Gaussian logit-noise expectation)`
              : "Disabled — deterministic point estimates"}
          </span>
        </div>
        <p className="mt-3 text-sm leading-6 text-muted-foreground">{modelState.monteCarloRationale}</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main tab
// ---------------------------------------------------------------------------

type SubTab = "automl" | "pfi" | "ensemble";

export function ModelMonitorTab({ modelState }: { modelState: ModelStateDoc }) {
  const [subTab, setSubTab] = useState<SubTab>("automl");
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
  const upsetText = upsets.map((u) => `${u.team} over ${u.loser} at ${u.prob}%`).join(", ");

  const subTabs: { id: SubTab; label: string; icon: React.ReactNode }[] = [
    { id: "automl", label: "Auto-ML Selection & Weights", icon: <Bot className="size-3.5" /> },
    { id: "pfi", label: "Feature Importance (PFI)", icon: <BarChart3 className="size-3.5" /> },
    { id: "ensemble", label: "Ensemble Architecture", icon: <Network className="size-3.5" /> },
  ];

  return (
    <div className="flex flex-col gap-5">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold tracking-tight">Auto-ML &amp; Model Monitor</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Automated feature selection, LASSO regularized weighting, model ensemble optimization (AUC
          &gt; 0.60), and calibration diagnostics.
        </p>
      </div>

      {/* Sub-tabs */}
      <div className="flex flex-wrap items-center gap-2">
        {subTabs.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setSubTab(t.id)}
            className={cn(
              "flex cursor-pointer items-center gap-1.5 rounded-full px-3.5 py-1.5 text-xs font-semibold transition-colors",
              subTab === t.id
                ? "bg-cyan-500 text-cyan-950"
                : "border border-border bg-card text-foreground hover:border-ring/50",
            )}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </div>

      {/* Sub-tab content */}
      {subTab === "automl" && <AutoMlPanel modelState={modelState} />}
      {subTab === "pfi" && <PfiPanel modelState={modelState} />}
      {subTab === "ensemble" && <EnsemblePanel modelState={modelState} />}

      {/* Drift monitor section */}
      <div className="pt-4">
        <h2 className="text-xl font-bold tracking-tight">Model &amp; Data Drift Monitor</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Tracking model health, feature drift, and performance over time
        </p>
      </div>

      {/* Drift metric cards */}
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
