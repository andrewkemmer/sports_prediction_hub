import { api } from "@/convex/_generated/api";
import type { GameDoc, ModelStateDoc, RefreshProgressDoc } from "@/lib/mlb-ui-types";
import { addDaysYmd, formatDateLong, todayET } from "@/lib/format";
import { useAuth } from "@/hooks/use-auth";
import { CalibrationTab } from "@/components/mlb/CalibrationTab";
import { GamesTab } from "@/components/mlb/GamesTab";
import { ModelMonitorTab } from "@/components/mlb/ModelMonitorTab";
import { PowerRankingsTab } from "@/components/mlb/PowerRankingsTab";
import { Progress } from "@/components/ui/progress";
import { useAction, useQuery } from "convex/react";
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
  const modelState = useQuery(api.mlb.getModelState);
  const refreshProgress = useQuery(api.mlb.getRefreshProgress);
  const refresh = useAction(api.mlbActions.refreshModel);
  const predictDate = useAction(api.mlbActions.predictDate);

  const [tab, setTab] = useState<Tab>("games");
  const [refreshing, setRefreshing] = useState(false);
  const [selectedDate, setSelectedDate] = useState<string>(todayET());
  const [dateLoading, setDateLoading] = useState(false);

  const games = useQuery(api.mlb.getGamesByDate, { date: selectedDate });
  const requestedDates = useRef(new Set<string>());
  // Timestamp of the most recent refresh click, used to tell a *new* run's
  // progress writes apart from a previous run's stale `done` doc.
  const refreshStartedAt = useRef(0);

  // The refresh action reports its stage through the `refreshProgress` doc so
  // the UI can render a real progress bar instead of an indeterminate spinner.
  // A progress doc that has not updated in 3 minutes while no refresh is in
  // flight locally is treated as stale (the server action died without writing
  // `done`), so a dead run never pins the UI at 4% forever.
  const progressStale = refreshProgress && !refreshProgress.done && Date.now() - refreshProgress.updatedAt > 3 * 60_000;
  const activeProgress = refreshing || (!!refreshProgress && !refreshProgress.done && !progressStale);
  // True when a refresh is running on the server even if this page reloaded
  // mid-flight (the local `refreshing` state resets on remount).
  const serverRefreshing = !!refreshProgress && !refreshProgress.done && !progressStale;
  const progressForUi: RefreshProgressDoc | null =
    refreshing && (!refreshProgress || refreshProgress.done) ? null : (refreshProgress ?? null);
  const showProgress = activeProgress;
  // While a refresh IS in flight locally, warn if the server hasn't checked in
  // for a while so the user knows it may be stuck rather than silently waiting.
  const progressStalled =
    refreshing && !!refreshProgress && !refreshProgress.done && Date.now() - refreshProgress.updatedAt > 2 * 60_000;
  // Show a stale-but-failed progress banner so the user can see the prior
  // refresh's failure reason after a dropped WebSocket reconnect.
  const lastFailedProgress =
    refreshProgress && refreshProgress.done && refreshProgress.error && Date.now() - refreshProgress.updatedAt < 30 * 60_000;
  const lastFailedMessage = refreshProgress?.error ?? null;

  // On-demand predictions for a date that has no stored games yet.
  useEffect(() => {
    if (!modelState) return;
    if (games === undefined) return; // still loading
    if (games.length > 0) return;
    if (requestedDates.current.has(selectedDate)) return;
    requestedDates.current.add(selectedDate);
    setDateLoading(true);
    predictDate({ date: selectedDate })
      .catch(() => {
        requestedDates.current.delete(selectedDate);
        toast.error("Could not load predictions", {
          description: `No data for ${formatDateLong(selectedDate)}.`,
        });
      })
      .finally(() => setDateLoading(false));
  }, [selectedDate, games, modelState, predictDate]);

  // Once the server-side action marks the progress doc `done` — on normal
  // success, or later on after a dropped WebSocket where the action kept
  // running server-side — exit the refreshing state. Without this, a
  // successful refresh left `refreshing` true forever: the button stayed on
  // "Refreshing…" and the progress bar pinned at its default "4%" text.
  useEffect(() => {
    if (refreshing && refreshProgress?.done && refreshProgress.updatedAt >= refreshStartedAt.current) {
      setRefreshing(false);
    }
  }, [refreshing, refreshProgress]);

  const handleRefresh = async () => {
    if (refreshing || serverRefreshing) return;
    refreshStartedAt.current = Date.now();
    setRefreshing(true);
    try {
      const res = await refresh({});
      if ((res as { alreadyRunning?: boolean }).alreadyRunning) {
        // A refresh was already in flight server-side (e.g. after a page
        // reload); nothing new was started — the banner keeps showing its
        // live progress.
        toast.info("Refresh already in progress", {
          description: "A refresh is running on the server — progress shown above.",
        });
        setRefreshing(false);
      } else {
        toast.success("Model refreshed", {
          description: `Trained on ${res.gamesTrained} games · AUC ${res.auc.toFixed(3)} · Brier ${res.brier.toFixed(3)}`,
        });
        // Reset the auto-fetch guard so today's view is live.
        requestedDates.current.delete(selectedDate);
        setRefreshing(false);
      }
    } catch (e) {
      // The Convex client throws "Connection lost while action was in flight"
      // when the browser tab is backgrounded or the WebSocket re-establishes
      // mid-flight. The action itself runs to completion on the server and
      // writes its progress + result there, so don't surface a hard "failure"
      // to the user in that case — keep the live progress bar visible.
      const message = e instanceof Error ? e.message : "Unknown error";
      const lostConnection = /connection lost|connection.*closed|client.*closed/i.test(message);
      if (!lostConnection) {
        toast.error("Refresh failed", { description: message });
        setRefreshing(false);
      } else {
        toast.info("Refresh still running on the server", {
          description: "The browser dropped its connection, but the action is still executing. Progress will resume when it completes.",
        });
        // Don't clear `refreshing` — the progress bar stays visible until the
        // server action finishes and updates `refreshProgress.done`.
      }
    }
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
              disabled={refreshing || serverRefreshing}
              className="flex cursor-pointer items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-medium text-foreground transition-colors hover:border-ring/50 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <RefreshCw className={cn("size-3.5", (refreshing || serverRefreshing) && "animate-spin")} />
              {refreshing || serverRefreshing ? "Refreshing…" : "Refresh"}
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
        {showProgress && <RefreshProgressBar progress={progressForUi} stalled={progressStalled} />}
        {lastFailedProgress && lastFailedMessage && (
          <div className="mb-4 rounded-xl border border-rose-500/40 bg-rose-500/5 p-3 text-sm text-rose-700 dark:text-rose-300">
            <span className="font-semibold">Last refresh failed: </span>
            {lastFailedMessage}
          </div>
        )}
        {!modelState ? (
          <EmptyState refreshing={refreshing} onRefresh={handleRefresh} />
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
                loading={dateLoading || games === undefined}
                onRefresh={handleRefresh}
              />
            )}
            {tab === "rankings" && (
              <PowerRankingsTab modelState={modelState as unknown as ModelStateDoc} />
            )}
            {tab === "calibration" && (
              <CalibrationTab modelState={modelState as unknown as ModelStateDoc} />
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

function RefreshProgressBar({ progress, stalled }: { progress: RefreshProgressDoc | null; stalled?: boolean }) {
  const pct = progress?.pct ?? 4;
  const stage = progress?.stage ?? "Starting refresh";
  const message = progress?.message ?? "Processing on the server…";
  return (
    <div className="mb-4 rounded-xl border border-border bg-card/70 p-4 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <RefreshCw className="size-3.5 animate-spin text-primary" />
          <span className="text-sm font-medium">{stage}</span>
        </div>
        <span className="text-xs font-medium tabular-nums text-muted-foreground">
          {Math.round(pct)}%
        </span>
      </div>
      <Progress value={pct} className="mt-2.5" />
      <p className="mt-2 text-xs text-muted-foreground">{message}</p>
      {stalled && (
        <p className="mt-2 text-xs font-medium text-amber-400">
          Still working — this is taking longer than usual. The server may be slow to reach the MLB API;
          you can wait for it to finish or click Refresh again to restart.
        </p>
      )}
    </div>
  );
}

function EmptyState({ refreshing, onRefresh }: { refreshing: boolean; onRefresh: () => void }) {
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
      <button
        type="button"
        onClick={onRefresh}
        disabled={refreshing}
        className="mt-6 flex cursor-pointer items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
      >
        <RefreshCw className={cn("size-4", refreshing && "animate-spin")} />
        {refreshing ? "Fetching data & training…" : "Refresh & train model"}
      </button>
    </div>
  );
}
