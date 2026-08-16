import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { teamMeta } from "@/convex/ml/teams";
import type { GameDoc, ShapContribution, TeamInfo } from "@/convex/ml/types";
import {
  formatAmerican,
  formatPct,
  formatProb,
  formatSigned,
  formatTimeET,
} from "@/lib/format";
import { cn } from "@/lib/utils";
import {
  Activity,
  Check,
  ChevronDown,
  Coins,
  HeartPulse,
  MapPin,
  Search,
  Trophy,
  X,
  Zap,
} from "lucide-react";
import { useState } from "react";

function Pill({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide whitespace-nowrap",
        className,
      )}
    >
      {children}
    </span>
  );
}

function teamColor(id: number): string {
  return teamMeta(id).color;
}

function TeamRow({
  team,
  prob,
  isPick,
  isWinner,
  isHome,
}: {
  team: TeamInfo;
  prob: number;
  isPick: boolean;
  isWinner: boolean;
  isHome: boolean;
}) {
  const color = teamColor(team.id);
  const record = team.wins !== undefined && team.losses !== undefined ? `${team.wins}-${team.losses}` : null;
  return (
    <div className="px-4 py-2">
      <div className="flex items-center gap-2.5">
        <span className="h-9 w-1 shrink-0 rounded-full" style={{ backgroundColor: color }} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-semibold text-foreground">{team.name}</span>
            <span className="text-[11px] font-medium text-muted-foreground">{team.abbrev}</span>
            {record && <span className="text-[11px] text-muted-foreground">{record}</span>}
            {isPick && <Pill className="bg-blue-500/15 text-blue-400">Pick</Pill>}
            {isWinner && <Trophy className="size-3.5 text-amber-400" />}
          </div>
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70">
            {isHome ? "Home" : "Away"}
          </div>
        </div>
        <span className="text-sm font-bold tabular-nums" style={{ color }}>
          {formatPct(prob)}
        </span>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-white/5">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${Math.round(prob * 100)}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function ShapRow({ item, max }: { item: ShapContribution; max: number }) {
  const positive = item.contribution >= 0;
  const width = max > 0 ? Math.max(4, Math.min(100, (Math.abs(item.contribution) / max) * 100)) : 0;
  return (
    <div className="flex items-center gap-2.5">
      <span className="w-28 shrink-0 truncate text-[11px] text-muted-foreground">{item.label}</span>
      <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-white/5">
        <div
          className={cn("absolute inset-y-0 left-0 rounded-full", positive ? "bg-emerald-400/80" : "bg-rose-400/80")}
          style={{ width: `${width}%` }}
        />
      </div>
      <span
        className={cn(
          "w-12 shrink-0 text-right text-[11px] font-medium tabular-nums",
          positive ? "text-emerald-400" : "text-rose-400",
        )}
      >
        {formatSigned(item.contribution, 2)}
      </span>
    </div>
  );
}

function DeepDive({ game }: { game: GameDoc }) {
  const awayColor = teamColor(game.away.id);
  const homeColor = teamColor(game.home.id);
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button
          type="button"
          className="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-lg border border-border/80 bg-white/[0.02] py-2 text-xs font-medium text-blue-400 transition-colors hover:bg-blue-500/10"
        >
          <Search className="size-3.5" />
          Deep Dive
        </button>
      </DialogTrigger>
      <DialogContent className="max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-base">
            {game.away.name} vs {game.home.name}
          </DialogTitle>
          <DialogDescription className="text-xs">
            {game.venue ? `${game.venue} · ` : ""}
            Model breakdown and feature contributions
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-lg border border-border/70 bg-white/[0.02] p-3">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Home win prob</div>
            <div className="mt-1 text-xl font-bold tabular-nums" style={{ color: homeColor }}>
              {formatPct(game.homeWinProb)}
            </div>
          </div>
          <div className="rounded-lg border border-border/70 bg-white/[0.02] p-3">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Away win prob</div>
            <div className="mt-1 text-xl font-bold tabular-nums" style={{ color: awayColor }}>
              {formatPct(game.awayWinProb)}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center justify-between rounded-lg border border-border/70 px-3 py-2">
            <span className="text-muted-foreground">Fair ML {game.home.abbrev}</span>
            <span className="font-semibold tabular-nums">{formatAmerican(game.fairHomeOdds ?? 0)}</span>
          </div>
          <div className="flex items-center justify-between rounded-lg border border-border/70 px-3 py-2">
            <span className="text-muted-foreground">Fair ML {game.away.abbrev}</span>
            <span className="font-semibold tabular-nums">{formatAmerican(game.fairAwayOdds ?? 0)}</span>
          </div>
          <div className="col-span-2 flex items-center justify-between rounded-lg border border-border/70 px-3 py-2">
            <span className="text-muted-foreground">Model edge vs Elo baseline</span>
            <span className={cn("font-semibold tabular-nums", (game.edge ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400")}>
              {formatSigned(game.edge ?? 0, 3)}
            </span>
          </div>
        </div>

        {game.shap && game.shap.length > 0 && (
          <div>
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Feature contributions (logit space)
            </div>
            <div className="space-y-1.5">
              {game.shap.map((s) => (
                <div key={s.feature} className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">{s.label}</span>
                  <span className="tabular-nums text-muted-foreground">x={s.value.toFixed(2)}</span>
                  <span className={cn("w-14 text-right font-medium tabular-nums", s.contribution >= 0 ? "text-emerald-400" : "text-rose-400")}>
                    {formatSigned(s.contribution, 3)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {(game.homeInjuries !== undefined || game.awayInjuries !== undefined) && (
          <div className="flex items-center justify-between rounded-lg border border-border/70 px-3 py-2 text-xs">
            <span className="flex items-center gap-1.5 text-muted-foreground">
              <HeartPulse className="size-3.5" /> Players on IL
            </span>
            <span className="font-semibold tabular-nums">
              {game.home.abbrev} {game.homeInjuries ?? 0} · {game.away.abbrev} {game.awayInjuries ?? 0}
            </span>
          </div>
        )}

        <div className="rounded-lg border border-border/70 bg-white/[0.02] p-3 text-xs text-muted-foreground">
          <div className="mb-1.5 font-medium text-foreground">Starting pitchers</div>
          <div className="flex items-center justify-between">
            <span>{game.homePitcher?.name ?? "TBD"} (home)</span>
            <span className="tabular-nums">
              {game.homePitcher?.era !== undefined ? `ERA ${game.homePitcher.era.toFixed(2)}` : "ERA —"}
              {game.homePitcher?.k9 !== undefined ? ` · K/9 ${game.homePitcher.k9.toFixed(1)}` : ""}
            </span>
          </div>
          <div className="mt-1 flex items-center justify-between">
            <span>{game.awayPitcher?.name ?? "TBD"} (away)</span>
            <span className="tabular-nums">
              {game.awayPitcher?.era !== undefined ? `ERA ${game.awayPitcher.era.toFixed(2)}` : "ERA —"}
              {game.awayPitcher?.k9 !== undefined ? ` · K/9 ${game.awayPitcher.k9.toFixed(1)}` : ""}
            </span>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function GameCard({ game }: { game: GameDoc }) {
  const [showShap, setShowShap] = useState(true);
  const homeColor = teamColor(game.home.id);
  const awayColor = teamColor(game.away.id);
  const isFinal = game.status === "Final";
  const isLive = game.status === "Live";
  const winnerName = game.winner === "home" ? game.home.abbrev : game.winner === "away" ? game.away.abbrev : null;
  const pickName = game.pickTeam === "home" ? game.home.abbrev : game.away.abbrev;
  const isCoinFlip = game.pickProb <= 0.54;
  const maxShap = game.shap?.reduce((m, s) => Math.max(m, Math.abs(s.contribution)), 0) ?? 0;

  return (
    <div
      className={cn(
        "flex flex-col overflow-hidden rounded-2xl border bg-card",
        game.isCorrect === true && "border-emerald-500/30",
        game.isUpset === true && "border-rose-500/30",
        game.isCorrect === undefined && "border-border",
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 pt-3.5">
        <span className="text-xs font-medium text-muted-foreground">
          {game.dayNight === "day" ? "Day Game" : "Night Game"}
        </span>
        <div className="flex items-center gap-1.5">
          {isCoinFlip && (
            <Pill className="bg-amber-500/15 text-amber-400">
              <Coins className="size-3" /> Coin flip
            </Pill>
          )}
          {game.isUpset && (
            <Pill className="bg-amber-500/15 text-amber-400">
              <Zap className="size-3" /> Upset
            </Pill>
          )}
          {game.isCorrect === false && (
            <Pill className="bg-rose-500/15 text-rose-400">
              <X className="size-3" /> Miss
            </Pill>
          )}
          {game.isCorrect === true && (
            <Pill className="bg-emerald-500/15 text-emerald-400">
              <Check className="size-3" /> Correct pick
            </Pill>
          )}
          {isLive && (
            <Pill className="bg-emerald-500/15 text-emerald-400">
              <span className="relative flex size-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex size-1.5 rounded-full bg-emerald-400" />
              </span>
              Live
            </Pill>
          )}
          {isFinal && <Pill className="bg-teal-500/15 text-teal-300">Final{game.innings ? ` (F/${game.innings})` : ""}</Pill>}
        </div>
      </div>

      {/* Scoreboard */}
      <div className="flex items-center justify-between px-6 py-3">
        <div className="flex flex-col items-start leading-none">
          <span className="text-3xl font-bold tabular-nums text-foreground">
            {game.away.score !== undefined ? game.away.score : "—"}
          </span>
          <span className="mt-1 text-xs font-medium text-muted-foreground">{game.away.abbrev}</span>
        </div>
        <div className="text-center text-xs font-medium text-muted-foreground">
          {isFinal ? (game.innings ? `F/${game.innings}` : "F") : isLive ? "Live" : formatTimeET(game.gameDate)}
        </div>
        <div className="flex flex-col items-end leading-none">
          <span className="text-3xl font-bold tabular-nums text-foreground">
            {game.home.score !== undefined ? game.home.score : "—"}
          </span>
          <span className="mt-1 text-xs font-medium text-muted-foreground">{game.home.abbrev}</span>
        </div>
      </div>

      {/* Team rows (home first) */}
      <div className="flex flex-col">
        <TeamRow
          team={game.home}
          prob={game.homeWinProb}
          isPick={game.pickTeam === "home"}
          isWinner={game.winner === "home"}
          isHome
        />
        <TeamRow
          team={game.away}
          prob={game.awayWinProb}
          isPick={game.pickTeam === "away"}
          isWinner={game.winner === "away"}
          isHome={false}
        />
      </div>

      <div className="px-4 pb-3 text-center text-[11px] text-muted-foreground">
        Pre-game: {game.home.abbrev} {formatPct(game.homeWinProb)} vs {game.away.abbrev}{" "}
        {formatPct(game.awayWinProb)}
      </div>

      {/* Pitchers */}
      <div className="grid grid-cols-2 gap-2 px-4">
        <div className="rounded-lg border border-border/60 bg-white/[0.02] px-3 py-2">
          <div className="truncate text-xs font-medium text-foreground">{game.homePitcher?.name ?? "TBD"}</div>
          <div className="mt-0.5 text-[11px] text-muted-foreground">
            {game.homePitcher?.era !== undefined ? `ERA ${game.homePitcher.era.toFixed(2)}` : "ERA —"}
            {game.homePitcher?.k9 !== undefined ? `  K/9 ${game.homePitcher.k9.toFixed(1)}` : ""}
          </div>
        </div>
        <div className="rounded-lg border border-border/60 bg-white/[0.02] px-3 py-2">
          <div className="truncate text-xs font-medium text-foreground">{game.awayPitcher?.name ?? "TBD"}</div>
          <div className="mt-0.5 text-[11px] text-muted-foreground">
            {game.awayPitcher?.era !== undefined ? `ERA ${game.awayPitcher.era.toFixed(2)}` : "ERA —"}
            {game.awayPitcher?.k9 !== undefined ? `  K/9 ${game.awayPitcher.k9.toFixed(1)}` : ""}
          </div>
        </div>
      </div>

      {/* Run model: predicted score, totals, run line */}
      {game.runProjection && (
        <div className="mt-2.5 px-4">
          <div className="rounded-lg border border-border/60 bg-white/[0.02] px-3 py-2">
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-muted-foreground">Predicted score</span>
              <span className="font-semibold tabular-nums text-foreground">
                {game.home.abbrev} {game.runProjection.homeScore.toFixed(1)} – {game.away.abbrev}{" "}
                {game.runProjection.awayScore.toFixed(1)}
              </span>
            </div>
            <div className="mt-1.5 flex items-center justify-between gap-2 text-[11px]">
              <span className="text-muted-foreground">
                Total{" "}
                {game.marketOdds?.total !== undefined ? game.marketOdds.total : game.runProjection.total.toFixed(1)}
                {game.marketOdds?.overPrice !== undefined && (
                  <span className="text-muted-foreground/80">
                    {" "}
                    (O {formatAmerican(game.marketOdds.overPrice)} / U {formatAmerican(game.marketOdds.underPrice ?? 0)})
                  </span>
                )}
              </span>
              <span className="tabular-nums text-foreground">
                O {formatPct(game.runProjection.overProb, 0)} · U {formatPct(game.runProjection.underProb, 0)}
              </span>
            </div>
            <div className="mt-1.5 flex items-center justify-between gap-2 text-[11px]">
              <span className="text-muted-foreground">
                Run line {game.marketOdds?.runLine !== undefined ? `±${game.marketOdds.runLine}` : "±1.5"}
                {game.marketOdds?.homeRunLinePrice !== undefined && (
                  <span className="text-muted-foreground/80">
                    {" "}
                    ({game.home.abbrev} {formatAmerican(game.marketOdds.homeRunLinePrice)} / {game.away.abbrev}{" "}
                    {formatAmerican(game.marketOdds.awayRunLinePrice ?? 0)})
                  </span>
                )}
              </span>
              <span className="tabular-nums text-foreground">
                {game.home.abbrev} {formatPct(game.runProjection.homeRunLineProb, 0)} · {game.away.abbrev}{" "}
                {formatPct(game.runProjection.awayRunLineProb, 0)}
              </span>
            </div>
          </div>
        </div>
      )}

      {game.venue && (
        <div className="mt-2.5 flex items-center gap-1.5 px-4 text-[11px] text-muted-foreground">
          <MapPin className="size-3.5" />
          {game.venue}
          {game.weather?.tempF !== undefined && ` · ${game.weather.tempF}°F`}
          {game.weather?.windMph !== undefined && ` · ${game.weather.windMph} mph wind`}
        </div>
      )}

      {(game.homeInjuries !== undefined || game.awayInjuries !== undefined) && (
        <div className="mt-2 flex items-center gap-1.5 px-4 text-[11px] text-muted-foreground">
          <HeartPulse className="size-3.5" />
          On IL: {game.home.abbrev} {game.homeInjuries ?? 0} · {game.away.abbrev}{" "}
          {game.awayInjuries ?? 0}
        </div>
      )}

      {/* Odds / edge */}
      <div className="mt-2 flex items-center justify-between px-4 pb-3 text-[11px]">
        <span className="text-muted-foreground">
          {game.marketOdds?.homeMoneyline !== undefined ? (
            <>
              ML: {game.home.abbrev} {formatAmerican(game.marketOdds.homeMoneyline)} {game.away.abbrev}{" "}
              {formatAmerican(game.marketOdds.awayMoneyline ?? 0)}
            </>
          ) : (
            <>
              Fair ML: {game.home.abbrev} {formatAmerican(game.fairHomeOdds ?? 0)} {game.away.abbrev}{" "}
              {formatAmerican(game.fairAwayOdds ?? 0)}
            </>
          )}
        </span>
        <span className={cn("font-medium tabular-nums", (game.edge ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400")}>
          Edge: {formatSigned(game.edge ?? 0, 2)}
        </span>
      </div>

      {/* SHAP features */}
      {game.shap && game.shap.length > 0 && (
        <div className="border-t border-border/70 px-4 py-3">
          <button
            type="button"
            className="flex w-full cursor-pointer items-center justify-between"
            onClick={() => setShowShap((v) => !v)}
          >
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              <Activity className="size-3.5" />
              SHAP Features
            </span>
            <ChevronDown className={cn("size-4 text-muted-foreground transition-transform", showShap && "rotate-180")} />
          </button>
          {showShap && (
            <div className="mt-3 space-y-2">
              {game.shap.map((s) => (
                <ShapRow key={s.feature} item={s} max={maxShap} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Deep dive */}
      <div className="px-4 pb-3 pt-1">
        <DeepDive game={game} />
      </div>

      {/* Result banner */}
      {winnerName && (
        <div
          className={cn(
            "mt-auto flex items-center justify-center gap-1.5 px-4 py-2.5 text-xs font-semibold",
            game.isCorrect ? "bg-emerald-500/10 text-emerald-300" : "bg-rose-500/10 text-rose-300",
          )}
        >
          {game.isCorrect ? <Check className="size-3.5" /> : <X className="size-3.5" />}
          {game.isCorrect
            ? `${winnerName} Won — Model Correct`
            : `${winnerName} Won — Upset! Model picked ${pickName}`}
        </div>
      )}
    </div>
  );
}
