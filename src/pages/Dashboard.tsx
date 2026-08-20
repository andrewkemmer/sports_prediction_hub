import type { GameDoc, ModelStateDoc, RefreshProgressDoc } from "@/lib/mlb-ui-types";
import { addDaysYmd, formatDateLong, todayET } from "@/lib/format";
import { useAuth } from "@/hooks/use-auth";
import { useCdnData } from "@/hooks/use-cdn-data";
import { CalibrationTab } from "@/components/mlb/CalibrationTab";
import { GamesTab } from "@/components/mlb/GamesTab";
import { ModelMonitorTab } from "@/components/mlb/ModelMonitorTab";
import { PowerRankingsTab } from "@/components/mlb/PowerRankingsTab";
import { Progress } from "@/components/ui/progress";
import { LogOut, RefreshCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

type Tab = "games" | "rankings" | "calibration" | "monitor";

const TABS: { id: Tab; label: string }[] = [
  { id: "games", label: "Today's Games" },
  { id: "rankings", label: "Power Rankings" },
  { id: "calibration", label: "Calibration" },
  { id: "monitor", label: "Model Monitor" },
];

function BaseballMark() {
  return (
    <span className="flex size-9 items-center justify-center rounded-full bg-gradient-to-br from-rose-500/30 to-red-600/20 ring-1 ring-rose-500/30">
      <svg viewBox="0 0 24 24" className="size-5" aria-hidden>
        <circle cx="12" cy="12" r="9" fill="#f8fafc" />
        <path
          d="M4.5 7.5c2.6-1.6 5.8-1.7 8.4-.2M19.5 16.5c-2.6 1.6-5.8 1.7-8.4.2"
          fill="none"
          stroke="#e11d48"
          strokeWidth="1.4"
          strokeLinecap="round"
        />
        <path
          d="M4.5 16.5c2.6 1.6 5.8 1.7 8.4.2M19.5 7.5c-2.6-1.6-5.8-1.7-8.4-.2"
          fill="none"
          stroke="#e11d48"
          strokeWidth="1.4"
          strokeLinecap="round"
        />
      </svg>
    </span>
  );
}

export default function Dashboard() {
  const { user, signOut } = useAuth();
  const { payload, loading: cdnLoading, error: cdnError, fetchedAt } = useCdnData();

  const modelState = payload?.modelState ?? null;

  const [tab, setTab] = useState<Tab>("games");
  const [refreshing, setRefreshing] = useState(false);
  const [selectedDate, setSelectedDate] = useState<string>(todayET());
  const [dateLoading, setDateLoading] = useState(false);

  // Games for the selected date, sourced from CDN
  const games = payload?.gamesByDate[selectedDate] ?? undefined;
  const loading = cdnLoading || dateLoading;

  // Stable empty set for date-request guard
  const requestedDates = useRef(new Set<string>());

  // Once data loads for a date that had no games, show a message
  useEffect(() => {
    if (!payload) return;
    if (cdnLoading) return;
    if (games !== undefined) return;
    if (requestedDates.current.has(selectedDate)) return;
    requestedDates.current.add(selectedDate);
    toast.info("No data available", {
      description: `No games found for ${formatDateLong(selectedDate)}.`,
    });
  }, [selectedDate, games, payload, cdnLoading]);

  const handleRefresh = async () => {
    // CDN data is static — refresh re-fetches the same JSON file
    toast.info("Re-fetching data", {
      description: "Pulling the latest predictions from the CDN…",
    });
    window.location.reload();
  };

  const handleSignOut = async () => {
    await signOut();
  };

  return (
    <main className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-20 border-b border-border/70 bg-background/80 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-2.5">
            <BaseballMark />
            <h1 className="text-base font-bold tracking-tight">MLB Predictions</h1>
          </div>

          <nav className="flex flex-wrap items-center gap-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                className={cn(
                  "relative shrink-0 cursor-pointer rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  tab === t.id ? "text-foreground" : "text-muted-foreground hover:text-foreground",
                )}
              >
                {t.label}
                {tab === t.id && (
                  <span className="absolute inset-x-2 -bottom-0.5 h-0.5 rounded-full bg-primary" />
                )}
              </button>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={handleRefresh}
              disabled={cdnLoading}
              className="flex cursor-pointer items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-medium text-foreground transition-colors hover:border-ring/50 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <RefreshCw className={cn("size-3.5", cdnLoading && "animate-spin")} />
              {cdnLoading ? "Refreshing…" : "Refresh"}
            </button>
            <button
              type="button"
              onClick={handleSignOut}
              className="flex cursor-pointer items-center gap-1.5 rounded-lg border border-border bg-card px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
              title={user?.name ? `Signed in as ${user.name} — sign out` : "Sign out"}
            >
              <LogOut className="size-3.5" />
            </button>
          </div>
        </div>
      </header>

      {/* Content */}
      <div className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6">
        {!modelState ? (
          <EmptyState loading={cdnLoading} onRefresh={handleRefresh} error={cdnError} />
        ) : (
          <>
            {tab === "games" && (
              <GamesTab
                games={(games ?? []) as GameDoc[]}
                modelState={modelState as unknown as ModelStateDoc}
                selectedDate={selectedDate}
                onDateChange={setSelectedDate}
                onPrev={() => setSelectedDate((d) => addDaysYmd(d, -1))}
                onNext={() => setSelectedDate((d) => addDaysYmd(d, 1))}
                loading={loading}
                onRefresh={handleRefresh}
              />
            )}
            {tab === "rankings" && (
              <PowerRankingsTab modelState={modelState as unknown as ModelStateDoc} />
            )}
            {tab === "calibration" && payload && (
              <CalibrationTab
                modelState={modelState as unknown as ModelStateDoc}
                gameCards={payload.gameCards}
                calibrationBins={payload.calibrationBins}
                confidenceDistribution={payload.confidenceDistribution}
                calibrationCurve={payload.calibrationCurve}
                totalsMetrics={payload.totalsMetrics}
                runLineMetrics={payload.runLineMetrics}
                moneylineTotal={payload.moneylineTotal}
                moneylineCorrect={payload.moneylineCorrect}
                moneylineAccuracy={payload.moneylineAccuracy}
              />
            )}
            {tab === "monitor" && (
              <ModelMonitorTab modelState={modelState as unknown as ModelStateDoc} />
            )}
          </>
        )}
      </div>
    </main>
  );
}

function EmptyState({ loading, onRefresh, error }: { loading: boolean; onRefresh: () => void; error?: string | null }) {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center text-center">
      <div className="flex size-14 items-center justify-center rounded-2xl border border-border bg-card">
        <RefreshCw className="size-6 text-primary" />
      </div>
      <h2 className="mt-5 text-xl font-bold tracking-tight">Train your prediction model</h2>
      <p className="mt-2 max-w-md text-sm text-muted-foreground">
        Pull every 2026 regular-season game from the MLB Stats API, fit and calibrate the model,
        and generate win probabilities for the rest of the season.
      </p>
      {error && (
        <p className="mt-2 max-w-md text-sm text-rose-400">{error}</p>
      )}
      <button
        type="button"
        onClick={onRefresh}
        disabled={loading}
        className="mt-6 flex cursor-pointer items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
      >
        <RefreshCw className={cn("size-4", loading && "animate-spin")} />
        {loading ? "Fetching data…" : "Refresh & train model"}
      </button>
    </div>
  );
}
