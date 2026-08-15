import type { ModelStateDoc } from "@/lib/mlb-ui-types";
import { formatTrainedAt } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Check, Database, Dices, Sparkles, X } from "lucide-react";

function Section({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="mb-4 flex items-center gap-2">
        {icon}
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      </div>
      {children}
    </div>
  );
}

export function ModelMonitorTab({ modelState }: { modelState: ModelStateDoc }) {
  const candidates = modelState.candidates ?? [];
  const features = modelState.featureImportances ?? [];

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-xl font-bold tracking-tight">Model Monitor</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          How the model was trained, which features and model were selected, and whether a
          stochastic (Monte Carlo) component was adopted.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Selected model */}
        <Section title="Selected Model" icon={<Sparkles className="size-4 text-primary" />}>
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-primary/15 px-2.5 py-0.5 text-xs font-semibold text-primary">
              {modelState.selectedModel}
            </span>
            <span className="text-xs text-muted-foreground">Trained {formatTrainedAt(modelState.trainedAt)} ET</span>
          </div>
          <p className="mt-3 text-sm leading-6 text-muted-foreground">{modelState.modelDescription}</p>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="rounded-lg border border-border/70 p-3">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Training games</div>
              <div className="mt-1 text-lg font-semibold tabular-nums">{modelState.gamesTrained}</div>
            </div>
            <div className="rounded-lg border border-border/70 p-3">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Holdout (test)</div>
              <div className="mt-1 text-lg font-semibold tabular-nums">{modelState.holdoutCount}</div>
            </div>
          </div>
        </Section>

        {/* Monte Carlo decision */}
        <Section
          title="Stochastic Component (Monte Carlo)"
          icon={<Dices className="size-4 text-amber-400" />}
        >
          <div
            className={cn(
              "flex items-center gap-2 rounded-lg border p-3",
              modelState.monteCarloEnabled
                ? "border-emerald-500/25 bg-emerald-500/10"
                : "border-border/70 bg-white/[0.02]",
            )}
          >
            {modelState.monteCarloEnabled ? (
              <Check className="size-4 text-emerald-400" />
            ) : (
              <X className="size-4 text-muted-foreground" />
            )}
            <span className={cn("text-sm font-medium", modelState.monteCarloEnabled ? "text-emerald-300" : "text-foreground")}>
              {modelState.monteCarloEnabled
                ? `Enabled — σ = ${modelState.monteCarloSigma.toFixed(2)}, ${modelState.monteCarloTrials} trials`
                : "Disabled — deterministic point estimates"}
            </span>
          </div>
          <p className="mt-3 text-sm leading-6 text-muted-foreground">{modelState.monteCarloRationale}</p>
        </Section>
      </div>

      {/* Candidate models */}
      <Section title="Model Selection (candidate comparison)">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="border-b border-border/70 text-left text-[11px] uppercase tracking-wider text-muted-foreground">
                <th className="pb-2 font-medium">Model</th>
                <th className="pb-2 text-right font-medium">AUC</th>
                <th className="pb-2 text-right font-medium">Brier</th>
                <th className="pb-2 text-right font-medium">Log-loss</th>
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
      </Section>

      {/* Feature selection */}
      <Section title="Feature Selection (greedy backward elimination)">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {features.map((f) => {
            const maxImp = Math.max(...features.map((x) => x.importance), 1e-6);
            return (
              <div key={f.feature} className="rounded-lg border border-border/70 p-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">{f.label}</span>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    uni AUC {f.univariateAuc.toFixed(3)}
                  </span>
                </div>
                <div className="mt-2.5 flex items-center gap-2.5">
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/5">
                    <div
                      className="h-full rounded-full bg-primary/70"
                      style={{ width: `${Math.max(4, (f.importance / maxImp) * 100)}%` }}
                    />
                  </div>
                  <span className="w-20 text-right text-[11px] tabular-nums text-muted-foreground">
                    w = {f.weight.toFixed(3)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </Section>

      {/* Data source */}
      <Section title="Data Source" icon={<Database className="size-4 text-teal-400" />}>
        <p className="text-sm leading-6 text-muted-foreground">
          All historical game, score, standings, probable-pitcher, player statistics, and weekly
          injured-list rosters come from a single consolidated source — the official MLB Stats API (
          <span className="font-medium text-foreground">statsapi.mlb.com</span>). No secondary data
          sources are used. The model retrains and regenerates predictions on demand whenever you
          click <span className="font-medium text-foreground">Refresh</span>.
        </p>
      </Section>
    </div>
  );
}
