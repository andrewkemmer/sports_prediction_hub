import { motion } from "framer-motion";
import { useAuth } from "@/hooks/use-auth";
import {
  Activity,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  Database,
  LineChart,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { Link } from "react-router";

const FEATURES = [
  {
    icon: <Activity className="size-5 text-blue-400" />,
    title: "Live win probabilities",
    body: "Every regular-season game gets a home/away win probability refreshed on demand from the MLB Stats API.",
  },
  {
    icon: <BrainCircuit className="size-5 text-violet-400" />,
    title: "Machine-learned model",
    body: "Elo ratings, logistic regression, feature selection, and an ensemble — tuned to maximize AUC and minimize Brier score.",
  },
  {
    icon: <ShieldCheck className="size-5 text-emerald-400" />,
    title: "Calibrated & tested",
    body: "Isotonic calibration keeps predictions honest, with reliability diagrams and a holdout test set for every metric.",
  },
  {
    icon: <Database className="size-5 text-teal-400" />,
    title: "One data source",
    body: "Schedules, scores, standings, and pitcher stats all flow from a single consolidated official API.",
  },
];

function BaseballMark() {
  return (
    <span className="flex size-9 items-center justify-center rounded-xl bg-gradient-to-br from-rose-500/30 to-red-600/20 ring-1 ring-rose-500/30">
      <svg viewBox="0 0 24 24" className="size-5" aria-hidden>
        <circle cx="12" cy="12" r="9" fill="#f8fafc" />
        <path d="M4.5 7.5c2.6-1.6 5.8-1.7 8.4-.2M19.5 16.5c-2.6 1.6-5.8 1.7-8.4.2" fill="none" stroke="#e11d48" strokeWidth="1.4" strokeLinecap="round" />
        <path d="M4.5 16.5c2.6 1.6 5.8 1.7 8.4.2M19.5 7.5c-2.6-1.6-5.8-1.7-8.4-.2" fill="none" stroke="#e11d48" strokeWidth="1.4" strokeLinecap="round" />
      </svg>
    </span>
  );
}

export default function Landing() {
  const { isAuthenticated } = useAuth();

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Nav */}
      <header className="sticky top-0 z-20 border-b border-border/70 bg-background/80 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
          <Link to="/" className="flex items-center gap-2.5">
            <BaseballMark />
            <span className="text-sm font-bold tracking-tight">MLB Predictions</span>
          </Link>
          <div className="flex items-center gap-2">
            <Link
              to={isAuthenticated ? "/dashboard" : "/auth?returnTo=%2Fdashboard"}
              className="rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90"
            >
              {isAuthenticated ? "Open dashboard" : "Get started"}
            </Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="mx-auto flex w-full max-w-6xl flex-col items-center px-4 pb-20 pt-16 text-center sm:px-6 sm:pt-24">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-card px-3 py-1 text-xs text-muted-foreground">
            <Sparkles className="size-3.5 text-blue-400" />
            2026 season · machine-learned predictions
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.08 }}
          className="mt-6 max-w-3xl text-4xl font-bold leading-tight tracking-tight sm:text-6xl"
        >
          Sharper MLB win probabilities,{" "}
          <span className="text-blue-400">calibrated by machine learning</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.16 }}
          className="mt-5 max-w-xl text-base leading-7 text-muted-foreground sm:text-lg"
        >
          A prediction and calibration dashboard that trains on real 2026 game data, selects its own
          features and model, and reports honest AUC and Brier scores — refreshed on demand.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.24 }}
          className="mt-8 flex flex-col items-center gap-3 sm:flex-row"
        >
          <Link
            to={isAuthenticated ? "/dashboard" : "/auth?returnTo=%2Fdashboard"}
            className="group flex items-center gap-2 rounded-lg bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90"
          >
            {isAuthenticated ? "Open your dashboard" : "Open the dashboard"}
            <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
          </Link>
          <a
            href="#features"
            className="flex items-center gap-2 rounded-lg border border-border bg-card px-6 py-3 text-sm font-medium text-foreground transition-colors hover:border-ring/50"
          >
            See how it works
          </a>
        </motion.div>

        {/* Stats strip */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.32 }}
          className="mt-14 grid w-full max-w-3xl grid-cols-1 gap-3 sm:grid-cols-3"
        >
          {[
            { icon: <BarChart3 className="size-4 text-cyan-400" />, label: "AUC-ROC", value: "Maximized", sub: "rank discrimination" },
            { icon: <LineChart className="size-4 text-emerald-400" />, label: "Brier score", value: "Minimized", sub: "calibration risk" },
            { icon: <Database className="size-4 text-teal-400" />, label: "MLB Stats API", value: "Single source", sub: "official data" },
          ].map((s) => (
            <div key={s.label} className="rounded-2xl border border-border bg-card p-4 text-left">
              <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-muted-foreground">
                {s.icon} {s.label}
              </div>
              <div className="mt-1.5 text-lg font-bold">{s.value}</div>
              <div className="text-xs text-muted-foreground">{s.sub}</div>
            </div>
          ))}
        </motion.div>
      </section>

      {/* Features */}
      <section id="features" className="border-t border-border/70">
        <div className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-6">
          <motion.h2
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-center text-2xl font-bold tracking-tight sm:text-3xl"
          >
            Built for accuracy, built for trust
          </motion.h2>
          <div className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {FEATURES.map((f, i) => (
              <motion.div
                key={f.title}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.45, delay: i * 0.06 }}
                className="rounded-2xl border border-border bg-card p-5"
              >
                <div className="flex size-10 items-center justify-center rounded-xl border border-border/70 bg-white/[0.02]">
                  {f.icon}
                </div>
                <h3 className="mt-4 text-sm font-semibold text-foreground">{f.title}</h3>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">{f.body}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-border/70">
        <div className="mx-auto w-full max-w-6xl px-4 py-16 text-center sm:px-6">
          <motion.h2
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-2xl font-bold tracking-tight sm:text-3xl"
          >
            Pick a date, hit refresh, see the model think
          </motion.h2>
          <p className="mx-auto mt-3 max-w-lg text-sm leading-6 text-muted-foreground">
            Browse any date left in the 2026 season. Every probability is backed by the latest
            historical data and a calibrated, tested model.
          </p>
          <Link
            to={isAuthenticated ? "/dashboard" : "/auth?returnTo=%2Fdashboard"}
            className="mt-7 inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90"
          >
            Launch the dashboard
            <ArrowRight className="size-4" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/70">
        <div className="mx-auto flex w-full max-w-6xl flex-col items-center justify-between gap-3 px-4 py-6 text-xs text-muted-foreground sm:flex-row sm:px-6">
          <div className="flex items-center gap-2">
            <BaseballMark />
            <span>MLB Predictions</span>
          </div>
          <span>Data via MLB Stats API · Predictions for entertainment</span>
        </div>
      </footer>
    </div>
  );
}
