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

- **Features** (16): Elo gap, win %, recent form, rest, IL counts, home field,
  starting-pitcher FIP/ERA, team OPS/ERA/fielding, park factor, temperature,
  wind, lineup-known and lineup OPS.
- **Feature selection**: L2-regularized logistic regression (Newton–Raphson /
  IRLS) with greedy backward elimination on a calibration split.
- **Candidates**: Elo, logistic regression, k-NN (k=21), naive Bayes, and a
  blended ensemble with an Elo-weight tuned on the calibration set.
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
python3 mlb_streamlit/scripts/smoke_test.py
```

The smoke test exercises the whole engine — metrics, logistic fitting,
cross-validation, feature/Elo engineering, the run-scoring model, the complete
Auto-ML pipeline on synthetic 2026 games, and a cache round-trip — with only
the Python standard library.
