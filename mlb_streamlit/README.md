# MLB Predictions — Streamlit Dashboard (2026)

A fully self-contained Python migration of the MLB prediction dashboard. It
replicates the four tabs of the React/Convex app — **Today's Games**, **Power
Rankings**, **Calibration**, and **Model Monitor** — and is driven entirely by
the public MLB Stats API (`statsapi.mlb.com`) as its single consolidated data
source.

Everything lives in this directory and runs on a plain Python interpreter:

| File | Purpose |
| --- | --- |
| `app.py` | The Streamlit app (4 tabs + refresh flow) |
| `data.py` | MLB Stats API client (schedule, pitchers, teams, lineups, injuries, odds) |
| `cache.py` | Disk-backed JSON cache so refreshes are incremental |
| `refresh.py` | The refresh pipeline: fetch → enrich → train → predict → persist |
| `engine/` | Pure-stdlib ML engine (features, logistic, metrics, model, runs, teams) |
| `engine/model.py` | Auto-ML: feature selection, model stacking, calibration, Monte Carlo decision |
| `engine/metrics.py` | AUC, Brier, log-loss, ECE, isotonic regression, calibration curves |
| `scripts/smoke_test.py` | Offline engine test (no Streamlit/network required) |

## Run it

```bash
pip install -r mlb_streamlit/requirements.txt
streamlit run mlb_streamlit/app.py
```

On first launch click **Refresh & train**: the app pulls every completed 2026
game plus the upcoming window from the MLB Stats API, enriches it with pitcher
ERA/FIP, team OPS/ERA/fielding, starting lineups, injured-list snapshots, and
market odds, then runs the full Auto-ML pipeline and generates win
probabilities for the rest of the season. Later refreshes are incremental —
only new dates are re-fetched.

**Market odds (optional):** set the `THE_ODDS_API_KEY` env var to show live
sportsbook prices (moneyline / totals / run lines) on the game cards. Without
it the dashboard uses model-derived fair odds and shows an amber status chip in
the header. Odds snapshots are cached on disk for an hour, so on-demand date
lookups stay inside The Odds API free-tier limits; a failed fetch falls back to
the last good snapshot.

## Deploy

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for two supported paths:

- **Streamlit Community Cloud** — point it at `mlb_streamlit/app.py`; the
  repo-root `requirements.txt` installs dependencies, no API keys needed (the
  MLB Stats API is public).
- **Docker** — `docker build -t mlb-streamlit . && docker run -p 8501:8501 mlb-streamlit`
  (plus a `docker-compose.yml` with a persistent cache volume).

An optional `THE_ODDS_API_KEY` (The Odds API) enables live sportsbook
moneyline/total/run-line prices; without it the app shows model-derived fair
odds.

## The refresh button

The single **Refresh & train** button (header) and the **Run Auto-ML
Optimization** button (Model Monitor tab) both run `refresh.run_refresh()` with
a live progress bar. The pipeline:

1. Fetch schedule (recent + upcoming window, or full seasons on a cold start).
2. Merge with the disk cache (completed games are never re-fetched).
3. Pull starting-pitcher stats, team season stats, lineups + player OPS, and
   injury snapshots (all from `statsapi.mlb.com`; odds are a best-effort extra).
4. Train the model (see below), calibrate, and re-score the upcoming window.
5. Persist `model_state.json` + per-date predictions; the UI re-renders.

Any date the user picks in the Games tab is predicted on demand via
`refresh.predict_date()` using the stored model — no full refresh required.

## The model (Auto-ML)

- **Features** (31): Elo gap, win %, recent form, rest, IL counts, home field,
  starting-pitcher FIP/ERA/K9/WHIP/last-3 ERA, team OPS/ERA/K9/WHIP/fielding,
  park factor, temperature, wind, lineup OPS/wOBA/ISO/hot streak, BvP /
  platoon / vs-team matchup edges — plus trajectory features from the cached
  game logs: starter FIP trend slope, starter short-rest × workload
  interaction, lineup batter momentum (recent OPS vs season) and 7-day
  fatigue. Trajectory features populate only where real data exists (same
  fresh-window gating as lineups), so older history never gets junk values.
- **Feature selection**: L2-regularized logistic regression (Newton–Raphson /
  IRLS) with greedy backward elimination on a calibration split.
- **Candidates**: Elo, logistic regression (3 ridge strengths), distance-
  weighted k-NN, L2-boosted decision stumps, a compact two-hidden-layer
  neural network (MLP, deterministic seed, L2 + early stopping), and a
  blended ensemble with an Elo-weight tuned on the calibration set. Only
  candidates clearing a 0.70 calibration AUC floor are eligible for
  selection/stacking.
- **Stacking**: greedy forward selection solves convex combination weights that
  minimize calibration-set Brier loss (high AUC, low risk).
- **Calibration**: isotonic regression (PAV) enforced monotonic on the
  calibration set; ECE is reported.
- **Monte Carlo**: a Gaussian logit-noise sigma grid is tested; the stochastic
  component is enabled only when it measurably reduces calibration-set Brier.
- **Validation**: 5-fold walk-forward cross-validation reports out-of-sample
  AUC/Brier per fold; final metrics are computed on a held-out test slice.
- **Monitoring**: per-feature PSI drift, rolling 30-day Brier, and model
  version history are persisted with each training run.

## Testing

```bash
python3 mlb_streamlit/scripts/smoke_test.py          # engine + data pipeline (233 checks)
python3 mlb_streamlit/scripts/ui_render_test.py       # Streamlit UI panels with stubbed streamlit/plotly
```

The smoke test exercises the whole engine — metrics, logistic fitting,
cross-validation, feature/Elo engineering, the run-scoring model, the complete
Auto-ML pipeline on synthetic 2026 games, and a cache round-trip — with only
the Python standard library.
