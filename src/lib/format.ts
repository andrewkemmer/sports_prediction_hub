// Formatting helpers for dates, probabilities, and odds.

export function formatPct(p: number, digits = 0): string {
  if (!Number.isFinite(p)) return "—";
  return `${(p * 100).toFixed(digits)}%`;
}

export function formatProb(p: number, digits = 3): string {
  if (!Number.isFinite(p)) return "—";
  return p.toFixed(digits);
}

export function formatAmerican(odds: number): string {
  if (!Number.isFinite(odds)) return "—";
  return odds > 0 ? `+${odds}` : `${odds}`;
}

export function formatSigned(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

export function formatDateLong(ymd: string): string {
  const d = new Date(`${ymd}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return ymd;
  return d.toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function formatDateShort(ymd: string): string {
  const d = new Date(`${ymd}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return ymd;
  return d.toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function formatTimeET(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "TBD";
  return d.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/New_York",
  });
}

export function todayET(): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
}

export function addDaysYmd(ymd: string, days: number): string {
  const d = new Date(`${ymd}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

export function formatTrainedAt(ms: number): string {
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/New_York",
  });
}
