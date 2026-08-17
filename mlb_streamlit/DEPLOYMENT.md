# Deployment

Two supported paths: **Streamlit Community Cloud** (free, zero-ops) and
**Docker** (any host, including Hugging Face Spaces with the Docker SDK).

## Option A — Streamlit Community Cloud (recommended)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** →
   select the repo and branch.
3. Set the **Main file path** to `mlb_streamlit/app.py`.
4. Advanced settings:
   - **Python version**: 3.11
   - Dependencies are installed automatically from the repo-root
     `requirements.txt`, which redirects to `mlb_streamlit/requirements.txt`.
     (No need to paste packages manually.)
5. Deploy.

**No secrets are required.** All game/player data comes from the public MLB
Stats API (`statsapi.mlb.com`).

Optional — real market odds: the dashboard shows model-derived *fair* odds
(moneyline, totals, run line) out of the box. To also display live sportsbook
prices, get a free key from [The Odds API](https://the-odds-api.com) and add it
under **Settings → Secrets**:

```
THE_ODDS_API_KEY = "your-key-here"
```

The app degrades gracefully when the key is absent or the odds call fails —
predictions and calibration never depend on it.

Notes:
- Community Cloud's filesystem is ephemeral and the app may sleep between
  sessions. The disk cache rebuilds on demand: click **Refresh & train** on
  first load to pull the season, and again whenever you want the freshest
  results.
- Startup is fast (the app only reads JSON on load — no network calls until
  you click refresh), so the health check passes immediately.

## Option B — Docker

Build and run from the `mlb_streamlit/` directory:

```bash
docker build -t mlb-streamlit .
docker run -d --name mlb-streamlit -p 8501:8501 \
  -e THE_ODDS_API_KEY=your-key-here \
  -v mlb_cache:/app/cache \
  mlb-streamlit
```

Or with Compose (reads `THE_ODDS_API_KEY` from your shell env):

```bash
docker compose up -d
```

- App: http://localhost:8501
- Health: `curl http://localhost:8501/_stcore/health` → `ok`
- The `mlb_cache` volume persists the JSON cache so refreshes stay
  incremental across container restarts.

### Verify before deploying

```bash
python3 mlb_streamlit/scripts/smoke_test.py   # engine, offline
docker build -t mlb-streamlit mlb_streamlit/  # image
```
