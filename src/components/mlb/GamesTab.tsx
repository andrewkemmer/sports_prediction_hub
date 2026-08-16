import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import type { GameDoc, ModelStateDoc } from "@/lib/mlb-ui-types";
import { formatDateLong, formatDateShort, formatPct } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Check, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";
import { useState } from "react";
import { GameCard } from "./GameCard";

type Filter = "all" | "final" | "live" | "upcoming";

function toYmd(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

export function GamesTab({
  games,
  modelState,
  selectedDate,
  onDateChange,
  onPrev,
  onNext,
  loading,
  onRefresh,
}: {
  games: GameDoc[];
  modelState: ModelStateDoc;
  selectedDate: string;
  onDateChange: (date: string) => void;
  onPrev: () => void;
  onNext: () => void;
  loading: boolean;
  onRefresh: () => void;
}) {
  const [filter, setFilter] = useState<Filter>("all");

  const final = games.filter((g) => g.status === "Final");
  const live = games.filter((g) => g.status === "Live");
  const upcoming = games.filter((g) => g.status === "Preview" || g.status === "Scheduled");
  const filtered =
    filter === "final" ? final : filter === "live" ? live : filter === "upcoming" ? upcoming : games;

  const record = modelState.todaysRecord;
  const nightCount = games.filter((g) => g.dayNight === "night").length;

  const allFilters: { id: Filter; label: string; count: number }[] = [
    { id: "all", label: "All Games", count: games.length },
    { id: "final", label: "Final", count: final.length },
    { id: "live", label: "Live", count: live.length },
    { id: "upcoming", label: "Upcoming", count: upcoming.length },
  ];
  const filters = allFilters.filter((f) => f.id === "all" || f.count > 0);

  return (
    <div className="flex flex-col gap-4">
      {/* Row 1: summary / status bar */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold text-foreground">{formatDateShort(selectedDate)}</span>
        {games.length > 0 && (
          <span className="text-sm text-muted-foreground">
            {filtered.length} of {games.length} games shown
          </span>
        )}
        {nightCount > 0 && (
          <span className="rounded-full bg-blue-500/15 px-2.5 py-1 text-xs font-medium text-blue-300">
            {nightCount} evening games begin 7 PM ET+
          </span>
        )}
        {record && record.total > 0 && (
          <>
            <span className="flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-2.5 py-1 text-xs font-semibold text-emerald-300">
              <Check className="size-3.5" />
              {record.wins}-{record.losses} Today
            </span>
            <span className="text-xs text-muted-foreground">{formatPct(record.accuracy, 1)} accuracy</span>
          </>
        )}
      </div>

      {/* Row 2: centered date selector */}
      <div className="flex items-center justify-center gap-2">
        <button
          type="button"
          onClick={onPrev}
          className="flex size-8 cursor-pointer items-center justify-center rounded-lg border border-border bg-card text-muted-foreground transition-colors hover:text-foreground"
        >
          <ChevronLeft className="size-4" />
        </button>
        <Popover>
          <PopoverTrigger asChild>
            <button
              type="button"
              className="cursor-pointer rounded-full border border-blue-500/30 bg-blue-500/10 px-5 py-2 text-sm font-medium text-foreground transition-colors hover:border-blue-500/50"
            >
              {formatDateLong(selectedDate)}
            </button>
          </PopoverTrigger>
          <PopoverContent className="w-auto p-0" align="center">
            <Calendar
              mode="single"
              selected={new Date(`${selectedDate}T00:00:00`)}
              onSelect={(d) => d && onDateChange(toYmd(d))}
            />
          </PopoverContent>
        </Popover>
        <button
          type="button"
          onClick={onNext}
          className="flex size-8 cursor-pointer items-center justify-center rounded-lg border border-border bg-card text-muted-foreground transition-colors hover:text-foreground"
        >
          <ChevronRight className="size-4" />
        </button>
      </div>

      {/* Row 3: filter pills */}
      <div className="flex items-center gap-1.5">
        {filters.map((f) => (
          <button
            key={f.id}
            type="button"
            onClick={() => setFilter(f.id)}
            className={cn(
              "cursor-pointer rounded-full px-3 py-1 text-xs font-medium transition-colors",
              filter === f.id
                ? "bg-primary text-primary-foreground"
                : "border border-border bg-card text-muted-foreground hover:text-foreground",
            )}
          >
            {f.label} ({f.count})
          </button>
        ))}
      </div>

      {/* Content */}
      {loading ? (
        <div className="flex min-h-[40vh] items-center justify-center">
          <Loader2 className="size-6 animate-spin text-muted-foreground" />
        </div>
      ) : filtered.length === 0 ? (
        <div className="flex min-h-[40vh] flex-col items-center justify-center text-center">
          <p className="text-sm text-muted-foreground">No games found for {formatDateLong(selectedDate)}.</p>
          <button
            type="button"
            onClick={onRefresh}
            className="mt-3 cursor-pointer text-sm font-medium text-primary hover:underline"
          >
            Refresh data
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {filtered.map((game) => (
            <GameCard key={game.gamePk} game={game} />
          ))}
        </div>
      )}
    </div>
  );
}
