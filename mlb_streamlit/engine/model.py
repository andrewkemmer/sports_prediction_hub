"""Full Auto-ML pipeline: feature selection, model selection, calibration.

Faithful port of `runModel` from src/convex/ml/model.ts. The pipeline is:

  1. Chronological Elo ratings + as-of-game-time features (no lookahead).
  2. Chronological 70/15/15 split: train / calibrate / test.
  3. Feature selection via greedy backward elimination on the calibrate set.
  4. Candidate models drawn from the active pipeline configuration;
     only candidates clearing CANDIDATE_MIN_AUC are eligible for stacking.
  5. Model selection: lexicographic (Brier, -AUC) for best single family;
     greedy-forward selection for the stacked ensemble.
  6. Isotonic calibration (PAV) to reduce Brier / calibration error.
  7. Monte Carlo decision: enable the stochastic component only if it
     measurably reduces holdout Brier (risk).

Only the Python standard library is required; when numpy is available it is
used to accelerate the Poisson Monte Carlo and k-NN paths.
"""

from __future__ import annotations

import math
import warnings

from .features import (
    FEATURE_KEYS,
    FEATURE_LABELS,
    MODEL_FEATURE_KEYS,
    STRUCTURAL_FEATURES,
    build_features_for_game,
    compute_elo_and_features,
)
from .logistic import (
    build_stacking_weights,
    cross_validate,
    logistic_logit,
    train_logistic,
)
from .tree_ensemble import rf_model, xgb_model, lgbm_model
from .markets import normalized_weight_rows
from .metrics import (
    apply_isotonic,
    calibration_curve_points,
    clamp,
    compute_auc,
    compute_brier,
    evaluate,
    isotonic_regression,
    logit,
    mean,
    monte_carlo_adjust,
    parallel_map,
    roundn,
    sigmoid,
    std,
)
from .nn import mlp_model
from .runs import expected_margin, expected_total, fit_run_model, simulate_runs
from .stack import fit_stack, stack_logit
from .teams import team_meta

try:  # numpy is optional; only used as an accelerator
    import numpy as _np
except Exception:  # pragma: no cover - fallback path
    _np = None

ELO_INIT = 1500.0
HFA_GRID = [0, 10, 20, 30, 40, 50, 60]
KNN_TRAIN_CAP = 1500
# AUC floor for the candidate pool: only models that clear it are eligible for
# selection / stacking. MLB single-game win-probability models sit around
# 0.52-0.55 AUC out-of-sample, so a hard 0.70 gate (a relic from a different
# data scale) wiped out the entire pool before the stacker could run. 0.51
# keeps the gate honest (above the 0.5 coin-flip) without bottlenecking the
# ensemble to a single candidate.
CANDIDATE_MIN_AUC = 0.51
MC_GRID = [0.1, 0.15, 0.2, 0.3, 0.45, 0.6]
# Ridge strengths tried for the deployed logistic (selected on the calibrate set).
LAMBDA_GRID = [0.001, 0.01, 0.1, 0.3, 1.0]

np = _np


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def elo_prob(row: dict, hfa: float) -> float:
    return sigmoid(((row["homeElo"] + hfa - row["awayElo"]) / 400) * math.log(10))


def apply_model(model: dict, features: dict, home_elo: float, away_elo: float) -> dict:
    """Apply a trained model to a feature vector -> full prediction object.

    When a serializable multi-model `stack` is present it is served directly
    (a convex combination of logistic / k-NN / boosted stumps / MLP / naive
    Bayes), then blended with the Elo logit via `blendW`. The logistic member
    still powers the SHAP readout as the interpretable linear approximation.
    """
    shap: list[dict] = []
    logistic_v = None
    if model.get("featureNames"):
        logistic_v = logistic_logit(model, features, shap)
    stack_v = stack_logit(model.get("stack"), features)
    if stack_v is not None:
        if logistic_v is not None:
            shap.append({
                "feature": "_stackBlend",
                "label": "Model stack blend",
                "contribution": stack_v - logistic_v,
                "value": 0.0,
            })
        logit_v = stack_v
    elif logistic_v is not None:
        logit_v = logistic_v
    else:
        logit_v = 0.0
    baseline = sigmoid(((home_elo + model["eloHfa"] - away_elo) / 400) * math.log(10))
    blend_w = model.get("blendW", 0.0) or 0.0
    if blend_w > 0:
        elo_logit_v = logit(baseline)
        shap.append({
            "feature": "_eloBlend",
            "label": "Elo rating blend",
            "contribution": blend_w * elo_logit_v,
            "value": baseline,
        })
        logit_v = (1 - blend_w) * logit_v + blend_w * elo_logit_v
    for s in shap:
        if s["feature"] in ("_eloBlend", "_stackBlend"):
            continue
        s["label"] = FEATURE_LABELS.get(s["feature"], s["feature"])
        s["value"] = features.get(s["feature"], 0.0)
    p = sigmoid(logit_v)
    p = apply_isotonic(model["isotonicPoints"], p)
    if model.get("monteCarloEnabled") and (model.get("monteCarloSigma") or 0) > 0:
        p = monte_carlo_adjust(p, model["monteCarloSigma"], 10000)
    p = clamp(p, 0.01, 0.99)
    edge = p - baseline
    shap.sort(key=lambda s: -abs(s["contribution"]))
    shap = shap[:5]
    return {
        "homeWinProb": p,
        "awayWinProb": 1 - p,
        "pickTeam": "home" if p >= 0.5 else "away",
        "pickProb": p if p >= 0.5 else 1 - p,
        "shap": shap,
        "edge": edge,
        "fairHomeOdds": _american_odds(p),
        "fairAwayOdds": _american_odds(1 - p),
    }


def _american_odds(p: float) -> int:
    q = clamp(p, 0.001, 0.999)
    if q >= 0.5:
        return -int(round((100 * q) / (1 - q)))
    return int(round((100 * (1 - q)) / q))


# ---------------------------------------------------------------------------
# Drift monitoring, rolling risk, and version history
# ---------------------------------------------------------------------------

def compute_feature_drift(rows: list[dict], selected: list[str]) -> list[dict]:
    """PSI-style drift: (mu_cur - mu_base)^2 / (2 * sigma_base^2)."""
    n = len(rows)
    if n == 0:
        return []
    baseline_end = int(math.floor(n * 0.7))
    recent_start = max(baseline_end, n - 40)
    baseline = rows[:baseline_end]
    recent = rows[recent_start:]
    out: list[dict] = []
    for f in selected:
        # The home-vs-away table has one canonical home perspective, so
        # ``homeField`` is a structural row indicator (always 1.0). Including
        # it in PSI creates a meaningless zero-variance drift row and can mask
        # real upstream changes; it remains available to legacy model payloads
        # but is intentionally excluded from statistical drift tracking.
        if f in STRUCTURAL_FEATURES:
            continue
        bvals = [r["features"][f] for r in baseline]
        cvals = [r["features"][f] for r in recent]
        b_mean = mean(bvals)
        c_mean = mean(cvals)
        b_std = std(bvals) or 1
        psi = ((c_mean - b_mean) * (c_mean - b_mean)) / (2 * b_std * b_std)
        out.append({
            "feature": f,
            "label": FEATURE_LABELS.get(f, f),
            "currentMean": roundn(c_mean, 3),
            "baselineMean": roundn(b_mean, 3),
            "psi": roundn(psi, 3),
            "status": "WARN" if psi >= 0.1 else "OK",
        })
    return out


def compute_rolling_brier(rows: list[dict], model: dict, as_of_date: str) -> dict:
    """Rolling (per-day) Brier score over the last 30 days."""
    from .features import shift_date

    cutoff = shift_date(as_of_date, -30)
    by_date: dict[str, dict] = {}
    total_sum = 0.0
    total_count = 0
    for r in rows:
        if r["game"]["date"] < cutoff:
            continue
        p = apply_model(model, r["features"], r["homeElo"], r["awayElo"])["homeWinProb"]
        sq = (p - r["label"]) * (p - r["label"])
        total_sum += sq
        total_count += 1
        b = by_date.setdefault(r["game"]["date"], {"sum": 0.0, "count": 0})
        b["sum"] += sq
        b["count"] += 1
    points = [
        {"date": d, "brier": roundn(v["sum"] / max(1, v["count"]), 3)}
        for d, v in sorted(by_date.items())
    ]
    return {"points": points, "baseline": roundn(total_sum / max(1, total_count), 3)}


def build_model_versions(rows: list[dict], as_of_date: str, final_eval: dict) -> list[dict]:
    """Data-driven version history from progressively larger training windows."""
    n = len(rows)
    stages = [
        (0.25, ["eloDiff", "winPctDiff", "homeField"],
         "Baseline model: Elo, win % and home-field features"),
        (0.5, ["eloDiff", "winPctDiff", "formDiff", "restDiff", "homeField"],
         "Added recent form and rest-day features"),
        (0.75, ["eloDiff", "winPctDiff", "formDiff", "restDiff", "injuryDiff",
                "homeField", "spFipDiff", "spEraDiff"],
         "Added injured-list edge, starting-pitcher FIP/ERA and isotonic calibration"),
    ]
    def fit_stage(args):
        frac, features, note = args
        end = int(math.floor(n * frac))
        if end < 60:
            return None
        train_end = int(math.floor(end * 0.85))
        train = rows[:train_end]
        test = rows[train_end:end]
        if len(train) < 40 or len(test) < 20:
            return None
        m = train_logistic(train, features)
        preds = [sigmoid(logistic_logit(m, r["features"], None)) for r in test]
        labels = [r["label"] for r in test]
        return {
            "date": rows[end - 1]["game"]["date"] if end - 1 < len(rows) else as_of_date,
            "auc": roundn(compute_auc(preds, labels), 3),
            "brier": roundn(compute_brier(preds, labels), 3),
            "notes": note,
        }

    # The three stage fits are independent — run them concurrently (only when
    # numpy is present; pure-Python fits serialize under the GIL anyway),
    # then number the surviving stages in order (identical versioning).
    versions: list[dict] = []
    for fit in parallel_map(fit_stage, stages, max_workers=len(stages) if np is not None else 1):
        if fit is None:
            continue
        versions.append({"version": f"v{len(versions) + 1}.0.0", **fit})
    versions.append({
        "version": f"v{len(versions) + 1}.0.0",
        "date": as_of_date,
        "auc": roundn(final_eval["auc"], 3),
        "brier": roundn(final_eval["brier"], 3),
        "notes": "Current model: ML feature selection, ensemble and Monte Carlo decision",
    })
    versions.reverse()
    return versions


# ---------------------------------------------------------------------------
# Run-margin reconciliation (win-probability model <-> run-scoring model)
# ---------------------------------------------------------------------------

def fit_run_margin_calibration(rows: list[dict], model: dict) -> dict:
    """margin = intercept + slope * logit(homeWinProb), fit on completed games."""
    xs: list[float] = []
    ys: list[float] = []
    for r in rows:
        p = apply_model(model, r["features"], r["homeElo"], r["awayElo"])["homeWinProb"]
        margin = (r["game"]["home"].get("score") or 0) - (r["game"]["away"].get("score") or 0)
        xs.append(logit(p))
        ys.append(margin)
    if len(xs) < 40:
        return {"slope": 0, "intercept": 0}
    n = len(xs)
    mx = mean(xs)
    my = mean(ys)
    sxy = 0.0
    sxx = 0.0
    for x, y in zip(xs, ys):
        sxy += (x - mx) * (y - my)
        sxx += (x - mx) * (x - mx)
    slope = sxy / sxx if sxx > 1e-9 else 0.0
    return {"slope": slope, "intercept": my - slope * mx}


# ---------------------------------------------------------------------------
# Accelerated Monte Carlo (numpy optional; identical semantics to runs.simulate_runs)
# ---------------------------------------------------------------------------

def simulate_runs_batch(
    run_model_state: dict,
    home_ids: list[int],
    away_ids: list[int],
    lines: list[float],
    margin_shifts: list[float],
    trials: int,
    run_line: float = 1.5,
    seed: int = 2026,
) -> list[dict]:
    """Vectorized Monte Carlo run simulation for many matchups at once.

    Falls back to the pure-Python per-game loop when numpy is unavailable.
    Semantics match engine.runs.simulate_runs exactly.
    """
    if np is None:
        return [
            simulate_runs(run_model_state, h, a, line, trials, run_line, shift)
            for h, a, line, shift in zip(home_ids, away_ids, lines, margin_shifts)
        ]

    offense = run_model_state["teamOffense"]
    defense = run_model_state["teamDefense"]
    park = run_model_state["parkFactor"]
    lr = run_model_state["leagueRuns"]
    cover_threshold = math.ceil(run_line)

    n = len(home_ids)
    park_mul = np.array([park.get(h, 1.0) for h in home_ids])
    off_h = np.array([offense.get(h, 1.0) for h in home_ids])
    def_a = np.array([defense.get(a, 1.0) for a in away_ids])
    off_a = np.array([offense.get(a, 1.0) for a in away_ids])
    def_h = np.array([defense.get(h, 1.0) for h in home_ids])
    base_home = lr * off_h * def_a * park_mul
    base_away = lr * off_a * def_h * park_mul
    lo = -np.minimum(base_home, base_away) + 0.05
    hi = np.minimum(base_home, base_away) - 0.05
    shift = np.clip(np.array(margin_shifts, dtype=float), lo, hi)
    lambda_home = np.clip(base_home + shift, 0, None)
    lambda_away = np.clip(base_away - shift, 0, None)
    line_arr = np.array(lines, dtype=float)

    rng = np.random.default_rng(seed)
    hs = rng.poisson(lambda_home[:, None], size=(n, trials))
    as_ = rng.poisson(lambda_away[:, None], size=(n, trials))
    totals = hs + as_
    margins = hs - as_

    home_score = hs.mean(axis=1)
    away_score = as_.mean(axis=1)
    total_mean = totals.mean(axis=1)
    over = (totals > line_arr[:, None]).sum(axis=1)
    under = (totals < line_arr[:, None]).sum(axis=1)
    over_under = over + under
    over_prob = np.where(over_under > 0, over / np.maximum(over_under, 1), 0.5)
    under_prob = np.where(over_under > 0, under / np.maximum(over_under, 1), 0.5)
    home_cover = (margins >= cover_threshold).sum(axis=1) / trials
    away_cover = 1 - home_cover

    out = []
    for i in range(n):
        out.append({
            "homeScore": round(float(home_score[i]), 6),
            "awayScore": round(float(away_score[i]), 6),
            "total": round(float(total_mean[i]), 6),
            "overProb": round(float(over_prob[i]), 6),
            "underProb": round(float(under_prob[i]), 6),
            "homeRunLineProb": round(float(home_cover[i]), 6),
            "awayRunLineProb": round(float(away_cover[i]), 6),
        })
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_model(
    completed_games: list[dict],
    season: str,
    as_of_date: str,
    injury_snapshots: dict | None = None,
) -> dict:
    """Train, select features, calibrate, and decide on Monte Carlo."""
    from .features import shift_date

    # numpy accelerates the IRLS fits, k-NN and Monte Carlo by 50-100x; make
    # the pure-Python fallback visible instead of silently running slow.
    if _np is None:
        warnings.warn(
            "numpy is not installed — the ML pipeline is running in its pure-Python "
            "fallback (roughly 50-100x slower). Install numpy>=1.24.0 to accelerate "
            "model training, k-NN, and Monte Carlo simulation.",
            RuntimeWarning,
            stacklevel=2,
        )

    # 1. Features / Elo (chronological, as-of-game-time).
    fe = compute_elo_and_features(completed_games, injury_snapshots, as_of_date)
    rows = fe["rows"]
    team_state = fe["teamState"]
    team_stats = fe["teamStats"]
    n = len(rows)

    train_end = int(math.floor(n * 0.7))
    calib_end = min(n, int(math.floor(n * 0.85)))
    train = rows[:train_end]
    calib = rows[train_end:calib_end]
    test = rows[calib_end:]

    calib_labels = [r["label"] for r in calib]
    test_labels = [r["label"] for r in test]

    def score_fn(preds: list[float], labels: list[int]) -> float:
        return compute_brier(preds, labels) - 0.5 * compute_auc(preds, labels)

    # 2. Feature selection: greedy backward elimination on the calib set.
    #    Every candidate fit reuses the accepted model's weights as its IRLS
    #    seed (dropping one feature only removes that column; the rest are
    #    already near-optimal), so each refit converges in a couple of
    #    iterations. No feature is removed from the candidate set — the same
    #    full FEATURE_KEYS universe always enters selection.
    selected = list(MODEL_FEATURE_KEYS)
    if len(calib) >= 20 and len(train) >= 20:
        current_model = train_logistic(train, selected, iterations=10)
        current_preds = [sigmoid(logistic_logit(current_model, r["features"], None)) for r in calib]
        current_score = score_fn(current_preds, calib_labels)
        improved = True
        stagnation_rounds = 0
        max_rounds = max(4, min(8, len(selected) - 2))
        rounds = 0
        while improved and len(selected) > 2 and rounds < max_rounds:
            improved = False
            rounds += 1
            # All drop-one candidates of a round are independent (each starts
            # from the accepted model's weights), so the fits run concurrently
            # and the results come back in order — the selection below is
            # byte-identical to a serial pass.
            fit_jobs: list[tuple[list[str], list[float]]] = []
            for drop in selected:
                drop_idx = selected.index(drop)
                candidate = [f for f in selected if f != drop]
                w0 = current_model["weights"][:drop_idx] + current_model["weights"][drop_idx + 1:] + [current_model["bias"]]
                fit_jobs.append((candidate, w0))

            def fit_candidate(args):
                candidate, w0 = args
                m = train_logistic(train, candidate, iterations=10, w0=w0)
                preds = [sigmoid(logistic_logit(m, r["features"], None)) for r in calib]
                return candidate, m, score_fn(preds, calib_labels)

            best_features = selected
            best_score = current_score
            best_model = current_model
            # Threads only overlap real work when numpy releases the GIL; in
            # the pure-Python fallback a pool just adds switching overhead, so
            # it stays serial there.
            fit_workers = min(8, max(2, len(fit_jobs))) if np is not None else 1
            for candidate, m, s in parallel_map(fit_candidate, fit_jobs, max_workers=fit_workers):
                if s < best_score:
                    best_score = s
                    best_features = candidate
                    best_model = m
            if len(best_features) < len(selected) and best_score < current_score - 1e-5:
                selected = best_features
                current_score = best_score
                current_model = best_model
                current_preds = [sigmoid(logistic_logit(current_model, r["features"], None)) for r in calib]
                stagnation_rounds = 0
                improved = True
            else:
                stagnation_rounds += 1
                if stagnation_rounds >= 2:
                    break

    # 2b. Tune the logistic ridge strength on the calibrate set. A small grid
    #     lets the selector trade bias for variance; the tuned model is the one
    #     actually deployed (the old pipeline always deployed λ=0.001 even when
    #     a different ridge scored better out-of-sample).
    lr_model = train_logistic(train, selected)
    best_lambda = 0.001
    if len(calib) >= 20 and len(train) >= 20:
        best_lambda_score = score_fn(
            [sigmoid(logistic_logit(lr_model, r["features"], None)) for r in calib],
            calib_labels,
        )
        for lam in LAMBDA_GRID:
            cand_model = train_logistic(train, selected, lambda_=lam)
            s = score_fn(
                [sigmoid(logistic_logit(cand_model, r["features"], None)) for r in calib],
                calib_labels,
            )
            if s < best_lambda_score - 1e-6:
                best_lambda_score = s
                lr_model = cand_model
                best_lambda = lam

    # 3. Tune Elo home-field advantage on the training set.
    elo_hfa = 30.0
    if len(train) >= 20:
        best_brier = float("inf")
        for hfa in HFA_GRID:
            preds = [elo_prob(r, hfa) for r in train]
            b = compute_brier(preds, [r["label"] for r in train])
            if b < best_brier:
                best_brier = b
                elo_hfa = hfa

    # 4. Blend weight (logistic + Elo logits) tuned on the calib set.
    lr_logits = [logistic_logit(lr_model, r["features"], None) for r in calib]
    elo_logits = [logit(elo_prob(r, elo_hfa)) for r in calib]
    blend_w = 0.5
    if len(calib) >= 20:
        best_brier = float("inf")
        w = 0.0
        while w <= 1.0001:
            preds = [sigmoid((1 - w) * l + w * e) for l, e in zip(lr_logits, elo_logits)]
            b = compute_brier(preds, calib_labels)
            if b < best_brier:
                best_brier = b
                blend_w = w
            w += 0.05

    # 5. Candidate models. The pool deliberately spans model families:
    #    regularized logistic (the interpretable backbone), a compact
    #    two-hidden-layer neural network (MLP), Random Forest, XGBoost, and
    #    LightGBM.  Only candidates that clear the AUC floor
    #    (CANDIDATE_MIN_AUC) are eligible for selection / stacking, so every
    #    model the selector chooses among is strong; the rest stay visible
    #    with an "excluded" note for transparency.
    lr_strong = train_logistic(train, selected, lambda_=0.1)
    nn = mlp_model(train, selected)
    rf = rf_model(train, selected)
    xgb = xgb_model(train, selected)
    lgbm = lgbm_model(train, selected)

    cand_preds: dict[str, list[float]] = {
        "Logistic regression": [sigmoid(logistic_logit(lr_strong, r["features"], None)) for r in calib],
        "Neural network (MLP)": [nn(r["features"]) for r in calib],
        "Random Forest": [rf(r["features"]) for r in calib],
        "XGBoost": [xgb(r["features"]) for r in calib],
        "LightGBM": [lgbm(r["features"]) for r in calib],
    }

    candidates: list[dict] = []
    eligible_preds: dict[str, list[float]] = {}
    for name, p in cand_preds.items():
        m = evaluate(p, calib_labels)
        eligible = m["auc"] >= CANDIDATE_MIN_AUC
        candidates.append({
            "name": name,
            "auc": m["auc"],
            "brier": m["brier"],
            "logLoss": m["logLoss"],
            "ece": m["ece"],
            "selected": False,
            "eligible": eligible,
            "note": "" if eligible else f"Below {CANDIDATE_MIN_AUC:.2f} AUC floor — excluded from selection",
        })
        if eligible:
            eligible_preds[name] = p
    if not eligible_preds:
        # Safety valve: if nothing clears the floor (e.g. a pathological small
        # calibration set), relax it so the chosen model is genuinely the best
        # of what exists instead of an empty pool.
        for c in candidates:
            c["eligible"] = True
            c["note"] = f"AUC floor relaxed — no candidate cleared {CANDIDATE_MIN_AUC:.2f}"
            eligible_preds[c["name"]] = cand_preds[c["name"]]

    # Best single family, fit on Brier (the risk metric) with AUC as a
    # sub-noise tie-break. The old rule replaced a candidate only when its AUC
    # beat the leader by +0.003 — on a ~400-game holdout that bar sits inside
    # the noise itself, so the first family to tie deadlocked at 100% weight
    # even when a strictly-better family (e.g. the MLP) sat just below the bar.
    # Lexicographic (Brier, -AUC) selection is exact and deadlock-free.
    best_single_name = min(
        (c for c in candidates if c["eligible"]),
        key=lambda c: (c["brier"], -c["auc"]),
    )["name"]
    best_brier = next(c["brier"] for c in candidates if c["name"] == best_single_name)
    for c in candidates:
        c["selected"] = c["name"] == best_single_name

    # 6. Stacking ensemble (greedy forward selection over eligible models only).
    stacking = build_stacking_weights(eligible_preds, calib_labels)

    best_name = best_single_name
    chosen_preds = eligible_preds[best_single_name]
    if len(stacking["preds"]) > 0 and stacking["brier"] < best_brier - 0.0005:
        best_name = "Stacked ensemble"
        chosen_preds = stacking["preds"]

    # 7. Deployable multi-model stack (serializable) + isotonic calibration.
    #    Fit every family on the full chronological history, tune family
    #    weights + the Elo blend on a chronological holdout, then fit isotonic
    #    on the exact blend apply_model serves (stack + Elo) — train == serve.
    stack, stack_blend_w = fit_stack(rows, selected, elo_hfa)

    # Stack-membership highlight for the monitor. `selected` above is the
    # strict best single family by holdout Brier (a rank, exactly one row);
    # `inStack` marks the fitted families carrying positive weight in the
    # deployed stack (the allocation vector). A family can be both, but the
    # two concepts never merge into duplicate "best" badges.
    positive_stack = {nm for nm, w in (stack.get("weights") or {}).items() if (w or 0) > 0}
    for c in candidates:
        c["inStack"] = c["name"] in positive_stack

    def _served_logit(r: dict) -> float:
        sl = stack_logit(stack, r["features"])
        el = logit(elo_prob(r, elo_hfa))
        return (1 - stack_blend_w) * sl + stack_blend_w * el if stack_blend_w > 0 else sl

    blend_preds = [sigmoid(_served_logit(r)) for r in calib]
    order = sorted(zip(blend_preds, calib_labels), key=lambda t: t[0])
    isotonic_points = isotonic_regression([o[0] for o in order], [o[1] for o in order])

    calibrated_calib = [apply_isotonic(isotonic_points, p) for p in blend_preds]

    # 8. Monte Carlo decision: enable only if it reduces Brier meaningfully.
    #    The stochastic component is a 7-point Gauss-Hermite quadrature
    #    expectation (O(1) per probability), evaluated at every σ in the grid
    #    over the calibration set; the best-scoring σ is reported even when
    #    it does not clear the improvement bar, so the rationale is exact.
    mc_sigma = 0.0
    mc_enabled = False
    base_brier = compute_brier(calibrated_calib, calib_labels)
    best_mc_brier = base_brier
    best_sigma = 0.0
    for s in MC_GRID:
        preds = [monte_carlo_adjust(p, s, 10000) for p in calibrated_calib]
        b = compute_brier(preds, calib_labels)
        if b < best_mc_brier:
            best_mc_brier = b
            best_sigma = s
        if b < base_brier - 0.0005:
            best_mc_brier = b
            mc_sigma = s
    if mc_sigma > 0:
        mc_enabled = True
    grid_text = ", ".join(f"{s}" for s in MC_GRID)
    if mc_enabled:
        mc_rationale = (
            f"Monte Carlo enabled: σ={mc_sigma} (grid {{{grid_text}}}) reduces calibration-set Brier "
            f"{base_brier:.4f} → {best_mc_brier:.4f} (calibration error shrinks toward the mean)."
        )
    else:
        mc_rationale = (
            f"Monte Carlo disabled: no stochastic σ in {{{grid_text}}} reduced calibration-set Brier "
            f"below {base_brier:.4f} (best {best_mc_brier:.4f} at σ={best_sigma}). "
            "Deterministic point estimates are kept."
        )

    model: dict = {
        "featureNames": selected,
        "weights": lr_model["weights"],
        "bias": lr_model["bias"],
        "featureStats": lr_model["featureStats"],
        "isotonicPoints": isotonic_points,
        "monteCarloSigma": mc_sigma,
        "monteCarloEnabled": mc_enabled,
        "eloHfa": elo_hfa,
        "blendW": stack_blend_w,
        "stack": stack,
    }

    # 9. Reconcile run-scoring model with win-probability model.
    run_margin_calibration = fit_run_margin_calibration(rows, model)

    def predict(game: dict) -> dict:
        return apply_model(
            model,
            build_features_for_game(game, team_state),
            team_state["elo"].get(game["home"]["id"], ELO_INIT),
            team_state["elo"].get(game["away"]["id"], ELO_INIT),
        )

    # 10. Final unbiased metrics on the test set.
    test_preds = [apply_model(model, r["features"], r["homeElo"], r["awayElo"])["homeWinProb"] for r in test]
    test_eval = evaluate(test_preds, test_labels)

    # 11. Drift / rolling risk / version history.
    feature_drift = compute_feature_drift(rows, selected)
    rolling = compute_rolling_brier(rows, model, as_of_date)
    model_versions = build_model_versions(rows, as_of_date, test_eval)
    brier_baseline = model_versions[1]["brier"] if len(model_versions) > 1 else rolling["baseline"]

    # 12. Feature importances (univariate AUC + learned coefficient).
    full_labels = [r["label"] for r in rows]
    feature_importances: list[dict] = []
    for f in FEATURE_KEYS:
        uni = compute_auc([r["features"][f] for r in rows], full_labels)
        idx = lr_model["featureNames"].index(f) if f in lr_model["featureNames"] else -1
        active = idx >= 0
        w = lr_model["weights"][idx] if active else 0.0
        feature_importances.append({
            "feature": f,
            "label": FEATURE_LABELS.get(f, f),
            "weight": w,
            "importance": abs(w),
            "univariateAuc": uni,
            "active": active,
        })

    # 13. Diagnostics: cross-validation + hyperparameter audit trail.
    cross_validation = cross_validate(rows, selected, 5)
    optimization_params = {
        "learningRate": 0,
        "l2Lambda": best_lambda,
        "epochs": 20,
        "hfaGrid": list(HFA_GRID),
        "blendStep": 0.05,
        "mcSigmaGrid": list(MC_GRID),
        "cvFolds": 5,
        "isotonicMethod": "Isotonic (PAV)",
        "featureSelection": "Greedy backward elimination (L2 logistic, IRLS)",
        "minCandidateAuc": CANDIDATE_MIN_AUC,
        "nnHidden": [20, 10],
        "nnEpochs": 40,
        "nnLr": 0.03,
        "nnL2": 1e-4,
        "nnEarlyStopPatience": 6,
    }

    # 14. Run-scoring model + run-line calibration.
    run_model_state = fit_run_model(completed_games)
    calib_rows = rows[:calib_end]
    if np is not None and len(calib_rows) > 0:
        sims = simulate_runs_batch(
            run_model_state,
            [r["game"]["home"]["id"] for r in calib_rows],
            [r["game"]["away"]["id"] for r in calib_rows],
            [0.0] * len(calib_rows),
            [0.0] * len(calib_rows),
            200,
        )
    else:
        sims = [
            simulate_runs(run_model_state, r["game"]["home"]["id"], r["game"]["away"]["id"], 0, 200)
            for r in calib_rows
        ]
    rl_pairs: list[tuple[float, int]] = []
    for r, sim in zip(calib_rows, sims):
        margin = (r["game"]["home"].get("score") or 0) - (r["game"]["away"].get("score") or 0)
        rl_pairs.append((sim["homeRunLineProb"], 1 if margin >= 2 else 0))
    rl_pairs.sort(key=lambda t: t[0])
    run_line_calibration = (
        isotonic_regression([t[0] for t in rl_pairs], [t[1] for t in rl_pairs])
        if len(rl_pairs) >= 40 else []
    )

    # 15. Power rankings (Elo-based).
    meta_map: dict[int, dict] = {}
    for g in completed_games:
        meta_map.setdefault(g["away"]["id"], {"name": g["away"]["name"], "abbrev": g["away"]["abbrev"]})
        meta_map.setdefault(g["home"]["id"], {"name": g["home"]["name"], "abbrev": g["home"]["abbrev"]})
    power_rankings: list[dict] = []
    for tid in sorted(team_state["elo"].keys()):
        rec = team_state["records"].get(tid, {"wins": 0, "losses": 0})
        home_rec = team_stats["homeRecords"].get(tid, {"wins": 0, "losses": 0})
        away_rec = team_stats["awayRecords"].get(tid, {"wins": 0, "losses": 0})
        meta = meta_map.get(tid, {"name": f"Team {tid}", "abbrev": "TBD"})
        home_total = home_rec["wins"] + home_rec["losses"]
        away_total = away_rec["wins"] + away_rec["losses"]
        wins = rec["wins"]
        losses = rec["losses"]
        power_rankings.append({
            "teamId": tid,
            "name": meta["name"],
            "abbrev": meta["abbrev"],
            "elo": team_state["elo"][tid],
            "wins": wins,
            "losses": losses,
            "winPct": wins / (wins + losses) if (wins + losses) > 0 else 0.0,
            "last10WinPct": team_state["form"].get(tid, 0.5),
            "lastGameDate": team_state["lastGameDate"].get(tid, ""),
            "injuries": team_state["injuries"].get(tid, 0),
            "runDiff": team_stats["runDiff"].get(tid, 0),
            "homeWinPct": home_rec["wins"] / home_total if home_total > 0 else 0.0,
            "awayWinPct": away_rec["wins"] / away_total if away_total > 0 else 0.0,
        })
    power_rankings.sort(key=lambda r: -r["elo"])

    # 16. Description — always describes the deployed scorer (the logistic +
    #     Elo blend stored in blendW), so selectedModel and apply_model never
    #     disagree about what is actually scoring today's games.
    positive_members = [n for n, w in stack["weights"].items() if w > 0]
    if len(positive_members) > 1:
        deployed_name = "Multi-model stack"
        description = (
            f"Multi-model stack ({', '.join(positive_members)}), blended "
            f"{1 - stack_blend_w:.2f} + {stack_blend_w:.2f}·Elo, "
            f"isotonic-calibrated{', Monte Carlo-smoothed' if mc_enabled else ''}."
        )
    elif stack_blend_w > 0:
        deployed_name = "Blended ensemble"
        description = (
            f"Ensemble: {1 - stack_blend_w:.2f}·logistic + {stack_blend_w:.2f}·Elo, "
            f"isotonic-calibrated{', Monte Carlo-smoothed' if mc_enabled else ''}."
        )
    else:
        deployed_name = "Logistic regression"
        description = (
            f"Logistic regression (L2, λ={best_lambda}), "
            f"isotonic-calibrated{', Monte Carlo-smoothed' if mc_enabled else ''}."
        )

    result = {
        "season": season,
        "asOfDate": as_of_date,
        "gamesTrained": n,
        "holdoutCount": len(test),
        "selectedModel": deployed_name,
        "modelDescription": description,
        "featureNames": selected,
        "weights": lr_model["weights"],
        "bias": lr_model["bias"],
        "featureStats": lr_model["featureStats"],
        "isotonicPoints": isotonic_points,
        "eloHfa": elo_hfa,
        "blendW": stack_blend_w,
        "stack": stack,
        "monteCarloEnabled": mc_enabled,
        "monteCarloTrials": 10000 if mc_enabled else 0,
        "monteCarloSigma": mc_sigma,
        "monteCarloRationale": mc_rationale,
        "auc": test_eval["auc"],
        "brier": test_eval["brier"],
        "logLoss": test_eval["logLoss"],
        "ece": test_eval["ece"],
        "bins": test_eval["bins"],
        "confidenceDistribution": test_eval["confidenceDistribution"],
        "calibrationCurve": test_eval["calibrationCurve"],
        "featureImportances": feature_importances,
        "candidates": candidates,
        "powerRankings": power_rankings,
        "featureDrift": feature_drift,
        "rollingBrier": rolling["points"],
        "brierBaseline": brier_baseline,
        "modelVersions": model_versions,
        # Single normalized allocation vector across the full candidate pool:
        # deployed stack members carry their (sum-to-1) weights, every other
        # row is an explicit zero — never an implicit 100% fallback.
        "stackingWeights": normalized_weight_rows(
            stack.get("weights") or {},
            [c["name"] for c in candidates],
        ),
        "crossValidation": cross_validation,
        "optimizationParams": optimization_params,
        "runModel": run_model_state,
        "runLineCalibration": run_line_calibration,
        "runMarginCalibration": run_margin_calibration,
    }

    return {
        "result": result,
        "model": model,
        "teamState": team_state,
        "rows": rows,
        "predict": predict,
    }


def decide_monte_carlo(rows: list[dict], model: dict) -> dict:
    """Decide the Monte Carlo smoothing sigma on the deployed model itself.

    The walk-forward selection rebuilds today's model *after* the in-sample
    pipeline, so a sigma tuned on the pre-selection fit would describe the
    wrong model. This evaluates the same 7-point Gauss-Hermite sigma grid on
    the exact model `apply_model` serves (stack + Elo blend + isotonic), on
    the chronological calibration slice — the stochastic component is enabled
    only when it measurably reduces Brier (risk), matching run_model's rule.
    """
    n = len(rows)
    calib = rows[:min(n, int(math.floor(n * 0.85)))]
    labels = [r["label"] for r in calib]
    preds = [
        apply_model(model, r["features"], r["homeElo"], r["awayElo"])["homeWinProb"]
        for r in calib
    ]
    base_brier = compute_brier(preds, labels)
    mc_sigma = 0.0
    best_brier = base_brier
    best_sigma = 0.0
    for s in MC_GRID:
        b = compute_brier([monte_carlo_adjust(p, s, 10000) for p in preds], labels)
        if b < best_brier:
            best_brier = b
            best_sigma = s
        if b < base_brier - 0.0005:
            best_brier = b
            mc_sigma = s
    enabled = mc_sigma > 0
    grid_text = ", ".join(f"{s}" for s in MC_GRID)
    if enabled:
        rationale = (
            f"Monte Carlo enabled: σ={mc_sigma} (grid {{{grid_text}}}) reduces calibration-set Brier "
            f"{base_brier:.4f} → {best_brier:.4f} (calibration error shrinks toward the mean)."
        )
    else:
        rationale = (
            f"Monte Carlo disabled: no stochastic σ in {{{grid_text}}} reduced calibration-set Brier "
            f"below {base_brier:.4f} (best {best_brier:.4f} at σ={best_sigma}). "
            "Deterministic point estimates are kept."
        )
    return {
        "sigma": mc_sigma,
        "enabled": enabled,
        "trials": 10000 if enabled else 0,
        "rationale": rationale,
    }


def run_model_lean(
    completed_games: list[dict],
    season: str,
    as_of_date: str,
    injury_snapshots: dict | None = None,
) -> dict:
    """Cheap training pass for the refresh pipeline.

    The refresh pipeline immediately follows this with the walk-forward
    selection, which REPLACES the feature set, stack, isotonic points,
    candidates, stacking weights, cross-validation and headline metrics via
    apply_walk_forward_selection. Fitting all of those twice (here and again
    on the walk-forward-selected recipe) was the dominant "retrains the
    walk-forward models multiple times" cost of a refresh. This pass keeps
    only what the walk-forward selection does not rebuild:

      * the chronological feature rows + as-of team state (Elo/records/injuries),
      * home-field advantage, a quick logistic + Elo blend + isotonic (used
        for the run-margin / run-line calibrations and the diagnostics below),
      * the run-scoring model state, power rankings, feature drift / rolling
        Brier / version history and feature importances.

    It skips greedy backward elimination, the ridge-lambda grid, the candidate
    pool, the deployable stack fit, Monte Carlo and cross-validation — the
    walk-forward selection owns all of those. Monte Carlo is decided on the
    final deployed model by `decide_monte_carlo` right after the selection
    pass replaces `model`.
    """
    fe = compute_elo_and_features(completed_games, injury_snapshots, as_of_date)
    rows = fe["rows"]
    team_state = fe["teamState"]
    team_stats = fe["teamStats"]
    n = len(rows)
    selected = list(MODEL_FEATURE_KEYS)

    train_end = int(math.floor(n * 0.7))
    calib_end = min(n, int(math.floor(n * 0.85)))
    train = rows[:train_end]
    calib = rows[train_end:calib_end]
    test = rows[calib_end:]
    calib_labels = [r["label"] for r in calib]
    test_labels = [r["label"] for r in test]

    # Home-field advantage tuned on the training slice (identical to run_model).
    elo_hfa = 30.0
    if len(train) >= 20:
        best_brier = float("inf")
        for hfa in HFA_GRID:
            b = compute_brier([elo_prob(r, hfa) for r in train], [r["label"] for r in train])
            if b < best_brier:
                best_brier = b
                elo_hfa = hfa

    lr_model = train_logistic(train, selected)
    # Logistic + Elo blend weight tuned on the calibration slice.
    blend_w = 0.5
    if len(calib) >= 20:
        lr_logits = [logistic_logit(lr_model, r["features"], None) for r in calib]
        elo_logits = [logit(elo_prob(r, elo_hfa)) for r in calib]
        best_brier = float("inf")
        w = 0.0
        while w <= 1.0001:
            preds = [sigmoid((1 - w) * l + w * e) for l, e in zip(lr_logits, elo_logits)]
            b = compute_brier(preds, calib_labels)
            if b < best_brier:
                best_brier = b
                blend_w = w
            w += 0.05

    def _served_logit(r: dict) -> float:
        l = logistic_logit(lr_model, r["features"], None)
        e = logit(elo_prob(r, elo_hfa))
        return (1 - blend_w) * l + blend_w * e if blend_w > 0 else l

    order = sorted(zip([sigmoid(_served_logit(r)) for r in rows], [r["label"] for r in rows]), key=lambda t: t[0])
    isotonic_points = isotonic_regression([o[0] for o in order], [o[1] for o in order])

    model: dict = {
        "featureNames": selected,
        "weights": lr_model["weights"],
        "bias": lr_model["bias"],
        "featureStats": lr_model["featureStats"],
        "isotonicPoints": isotonic_points,
        "eloHfa": elo_hfa,
        "blendW": blend_w,
        "stack": {
            "members": {"Logistic regression": lr_model},
            "weights": {"Logistic regression": 1.0},
        },
        "monteCarloSigma": 0.0,
        "monteCarloEnabled": False,
    }

    run_margin_calibration = fit_run_margin_calibration(rows, model)
    run_model_state = fit_run_model(completed_games)

    # Run-line calibration (200-trial sims over the calibration slice), mirroring
    # run_model's step 14 exactly.
    calib_rows = rows[:calib_end]
    if np is not None and len(calib_rows) > 0:
        sims = simulate_runs_batch(
            run_model_state,
            [r["game"]["home"]["id"] for r in calib_rows],
            [r["game"]["away"]["id"] for r in calib_rows],
            [0.0] * len(calib_rows),
            [0.0] * len(calib_rows),
            200,
        )
    else:
        sims = [
            simulate_runs(run_model_state, r["game"]["home"]["id"], r["game"]["away"]["id"], 0, 200)
            for r in calib_rows
        ]
    rl_pairs: list[tuple[float, int]] = []
    for r, sim in zip(calib_rows, sims):
        margin = (r["game"]["home"].get("score") or 0) - (r["game"]["away"].get("score") or 0)
        rl_pairs.append((sim["homeRunLineProb"], 1 if margin >= 2 else 0))
    rl_pairs.sort(key=lambda t: t[0])
    run_line_calibration = (
        isotonic_regression([t[0] for t in rl_pairs], [t[1] for t in rl_pairs])
        if len(rl_pairs) >= 40 else []
    )

    test_preds = [
        apply_model(model, r["features"], r["homeElo"], r["awayElo"])["homeWinProb"]
        for r in test
    ]
    test_eval = evaluate(test_preds, test_labels)

    feature_drift = compute_feature_drift(rows, selected)
    rolling = compute_rolling_brier(rows, model, as_of_date)
    model_versions = build_model_versions(rows, as_of_date, test_eval)
    brier_baseline = model_versions[1]["brier"] if len(model_versions) > 1 else rolling["baseline"]
    power_rankings = build_power_rankings(completed_games, team_state, team_stats)

    full_labels = [r["label"] for r in rows]
    feature_importances: list[dict] = []
    for f in FEATURE_KEYS:
        uni = compute_auc([r["features"][f] for r in rows], full_labels)
        idx = lr_model["featureNames"].index(f) if f in lr_model["featureNames"] else -1
        active = idx >= 0
        w = lr_model["weights"][idx] if active else 0.0
        feature_importances.append({
            "feature": f,
            "label": FEATURE_LABELS.get(f, f),
            "weight": w,
            "importance": abs(w),
            "univariateAuc": uni,
            "active": active,
        })

    result = {
        "season": season,
        "asOfDate": as_of_date,
        "gamesTrained": n,
        "holdoutCount": len(test),
        "selectedModel": "Walk-forward selection (lean pass)",
        "modelDescription": "Temporary in-sample fit; replaced by the walk-forward-selected model.",
        "featureNames": selected,
        "weights": lr_model["weights"],
        "bias": lr_model["bias"],
        "featureStats": lr_model["featureStats"],
        "isotonicPoints": isotonic_points,
        "eloHfa": elo_hfa,
        "blendW": blend_w,
        "stack": model["stack"],
        "monteCarloEnabled": False,
        "monteCarloTrials": 0,
        "monteCarloSigma": 0.0,
        "monteCarloRationale": "Monte Carlo is decided on the deployed model after walk-forward selection.",
        "auc": test_eval["auc"],
        "brier": test_eval["brier"],
        "logLoss": test_eval["logLoss"],
        "ece": test_eval["ece"],
        "bins": test_eval["bins"],
        "confidenceDistribution": test_eval["confidenceDistribution"],
        "calibrationCurve": test_eval["calibrationCurve"],
        "featureImportances": feature_importances,
        "candidates": [],
        "powerRankings": power_rankings,
        "featureDrift": feature_drift,
        "rollingBrier": rolling["points"],
        "brierBaseline": brier_baseline,
        "modelVersions": model_versions,
        "stackingWeights": [],
        "crossValidation": {},
        "optimizationParams": {
            "featureSelection": "Walk-forward nested L1 + stability (applied after this lean pass)",
            "modelSelection": "Stack vs logistic per date by holdout Brier",
            "isotonicMethod": "Isotonic (PAV)",
        },
        "runModel": run_model_state,
        "runLineCalibration": run_line_calibration,
        "runMarginCalibration": run_margin_calibration,
    }

    return {
        "result": result,
        "model": model,
        "teamState": team_state,
        "rows": rows,
    }


def build_power_rankings(
    completed_games: list[dict],
    team_state: dict,
    team_stats: dict,
) -> list[dict]:
    """Elo power-ranking table from a chronological feature pass.

    Extracted from `run_model` so the lightweight walk-forward path can build
    the same as-of rankings without re-running the full Auto-ML pipeline.
    """
    meta_map: dict[int, dict] = {}
    for g in completed_games:
        meta_map.setdefault(g["away"]["id"], {"name": g["away"]["name"], "abbrev": g["away"]["abbrev"]})
        meta_map.setdefault(g["home"]["id"], {"name": g["home"]["name"], "abbrev": g["home"]["abbrev"]})
    power_rankings: list[dict] = []
    for tid in sorted(team_state["elo"].keys()):
        rec = team_state["records"].get(tid, {"wins": 0, "losses": 0})
        home_rec = team_stats["homeRecords"].get(tid, {"wins": 0, "losses": 0})
        away_rec = team_stats["awayRecords"].get(tid, {"wins": 0, "losses": 0})
        meta = meta_map.get(tid, {"name": f"Team {tid}", "abbrev": "TBD"})
        home_total = home_rec["wins"] + home_rec["losses"]
        away_total = away_rec["wins"] + away_rec["losses"]
        wins = rec["wins"]
        losses = rec["losses"]
        power_rankings.append({
            "teamId": tid,
            "name": meta["name"],
            "abbrev": meta["abbrev"],
            "elo": team_state["elo"][tid],
            "wins": wins,
            "losses": losses,
            "winPct": wins / (wins + losses) if (wins + losses) > 0 else 0.0,
            "last10WinPct": team_state["form"].get(tid, 0.5),
            "lastGameDate": team_state["lastGameDate"].get(tid, ""),
            "injuries": team_state["injuries"].get(tid, 0),
            "runDiff": team_stats["runDiff"].get(tid, 0),
            "homeWinPct": home_rec["wins"] / home_total if home_total > 0 else 0.0,
            "awayWinPct": away_rec["wins"] / away_total if away_total > 0 else 0.0,
        })
    power_rankings.sort(key=lambda r: -r["elo"])
    return power_rankings


def run_model_light(
    rows: list[dict],
    completed_games: list[dict],
    season: str,
    as_of_date: str,
    feature_names: list[str] | None = None,
    mlp_epochs: int = 40,
    model_choice: str | None = None,
) -> dict:
    """Cheap point-in-time refit for walk-forward backtests.

    Reuses precomputed chronological feature rows (`rows`) and fits the same
    core production recipe — the deployable multi-model stack (logistic / k-NN
    / boosted stumps / MLP / naive Bayes blended by holdout-tuned weights) OR a
    pure logistic, chosen per-date by holdout Brier (fit_deployable_model) —
    plus Elo blend + isotonic calibration + Poisson run-scoring. It skips the
    expensive Auto-ML layers (greedy backward elimination, the full candidate
    pool, Monte Carlo grid, drift and cross-validation diagnostics). It trains
    strictly on prior rows, so no future outcome leaks into the fitted model.

    `feature_names` defaults to the full FEATURE_KEYS universe; the walk-
    forward calibration passes each date's L1-selected features so every
    dashboard scores with the identical feature set. `model_choice` forces the
    stack-vs-logistic decision when the walk-forward record already made it.
    """
    n = len(rows)
    train_end = int(math.floor(n * 0.7))
    calib_end = min(n, int(math.floor(n * 0.85)))
    train = rows[:train_end]
    test = rows[calib_end:]

    selected = list(MODEL_FEATURE_KEYS) if feature_names is None else [
        f for f in feature_names if f not in STRUCTURAL_FEATURES
    ]

    elo_hfa = 30.0
    if len(train) >= 20:
        best_brier = float("inf")
        for hfa in HFA_GRID:
            preds = [elo_prob(r, hfa) for r in train]
            b = compute_brier(preds, [r["label"] for r in train])
            if b < best_brier:
                best_brier = b
                elo_hfa = hfa

    # Deployable model + per-date family choice: fit the stack AND a pure
    # logistic on the prior history, tune each's Elo blend on a chronological
    # holdout, deploy whichever has the lower Brier on the trailing 15% (fit
    # on Brier), then fit isotonic on the exact served blend — train == serve.
    model, model_choice = fit_deployable_model(
        rows, selected, elo_hfa, mlp_epochs=mlp_epochs, model_choice=model_choice
    )
    model["monteCarloSigma"] = 0.0
    model["monteCarloEnabled"] = False

    run_margin_calibration = fit_run_margin_calibration(rows, model)
    run_model_state = fit_run_model(completed_games)

    calib_rows = rows[:calib_end]
    if np is not None and len(calib_rows) > 0:
        sims = simulate_runs_batch(
            run_model_state,
            [r["game"]["home"]["id"] for r in calib_rows],
            [r["game"]["away"]["id"] for r in calib_rows],
            [0.0] * len(calib_rows),
            [0.0] * len(calib_rows),
            200,
        )
    else:
        sims = [
            simulate_runs(run_model_state, r["game"]["home"]["id"], r["game"]["away"]["id"], 0, 200)
            for r in calib_rows
        ]
    rl_pairs: list[tuple[float, int]] = []
    for r, sim in zip(calib_rows, sims):
        margin = (r["game"]["home"].get("score") or 0) - (r["game"]["away"].get("score") or 0)
        rl_pairs.append((sim["homeRunLineProb"], 1 if margin >= 2 else 0))
    rl_pairs.sort(key=lambda t: t[0])
    run_line_calibration = (
        isotonic_regression([t[0] for t in rl_pairs], [t[1] for t in rl_pairs])
        if len(rl_pairs) >= 40 else []
    )

    test_labels = [r["label"] for r in test]
    test_preds = [
        apply_model(model, r["features"], r["homeElo"], r["awayElo"])["homeWinProb"]
        for r in test
    ]
    test_eval = evaluate(test_preds, test_labels)

    stack = model.get("stack") or {}
    positive_members = [nm for nm, w in stack.get("weights", {}).items() if w > 0]
    blend_w = model.get("blendW", 0.0)
    deployed = model_choice.get("deployed", "stack")
    if deployed == "stack" and len(positive_members) > 1:
        selected_model_name = "Multi-model stack (point-in-time)"
        description = (
            f"Point-in-time multi-model stack ({', '.join(positive_members)}), "
            f"blended {1 - blend_w:.2f} + {blend_w:.2f}·Elo, isotonic-calibrated."
        )
    elif deployed == "stack" or blend_w > 0:
        selected_model_name = "Blended ensemble (point-in-time)"
        description = (
            f"Point-in-time ensemble: {1 - blend_w:.2f}·logistic + {blend_w:.2f}·Elo, "
            "isotonic-calibrated."
        )
    else:
        selected_model_name = "Logistic regression (point-in-time)"
        description = "Point-in-time logistic + Elo, isotonic-calibrated."

    return {
        "season": season,
        "asOfDate": as_of_date,
        "gamesTrained": n,
        "holdoutCount": len(test),
        "selectedModel": selected_model_name,
        "modelDescription": description,
        "featureNames": selected,
        "weights": model.get("weights", []),
        "bias": model.get("bias", 0.0),
        "featureStats": model.get("featureStats", {}),
        "isotonicPoints": model.get("isotonicPoints", []),
        "eloHfa": elo_hfa,
        "blendW": blend_w,
        "stack": stack,
        "modelChoice": model_choice,
        "monteCarloEnabled": False,
        "monteCarloTrials": 0,
        "monteCarloSigma": 0.0,
        "auc": test_eval["auc"],
        "brier": test_eval["brier"],
        "logLoss": test_eval["logLoss"],
        "ece": test_eval["ece"],
        "runModel": run_model_state,
        "runLineCalibration": run_line_calibration,
        "runMarginCalibration": run_margin_calibration,
        "model": model,
    }


def fit_candidate_pool(train: list[dict], feature_names: list[str], mlp_epochs: int = 40) -> tuple[dict, float, float]:
    """Fit the full candidate model pool on `train` (chronological, prior-only).

    Used by the walk-forward selection pass: every model family is fit on games
    strictly before a target day, then scored on that day out-of-sample.

    Returns (predictors, elo_hfa, blend_w):
      predictors — name -> predict(features dict) -> probability
      elo_hfa    — home-field advantage tuned on `train`
      blend_w    — logistic/Elo blend weight tuned on a chronological holdout
                   of `train` (never the scored day), so nothing leaks.

    `mlp_epochs` caps the MLP's training length; the walk-forward pass lowers
    it because the MLP fit dominates backtest CPU and the MLP is only a
    diagnostic candidate there (the deployable stack's own MLP member keeps
    its full epochs).
    """
    n = len(train)
    prior = 0.5
    if n == 0:
        return (
            {
                "Logistic regression": lambda f: prior,
                "Neural network (MLP)": lambda f: prior,
                "Random Forest": lambda f: prior,
                "XGBoost": lambda f: prior,
                "LightGBM": lambda f: prior,
            },
            30.0,
            0.5,
        )

    # Elo home-field advantage tuned on the prior window.
    elo_hfa = 30.0
    if n >= 20:
        best_brier = float("inf")
        for hfa in HFA_GRID:
            preds = [elo_prob(r, hfa) for r in train]
            b = compute_brier(preds, [r["label"] for r in train])
            if b < best_brier:
                best_brier = b
                elo_hfa = hfa

    lr_strong = train_logistic(train, feature_names, lambda_=0.1)
    nn = mlp_model(train, feature_names, epochs=mlp_epochs)
    rf = rf_model(train, feature_names)
    xgb = xgb_model(train, feature_names)
    lgbm = lgbm_model(train, feature_names)

    # Blend weight tuned on the trailing 20% of the prior window (chronological
    # holdout) — mirrors run_model's calib-set blend tuning, but stays strictly
    # inside prior data so the scored day is never used to tune the blend.
    blend_w = 0.5
    split = int(math.floor(n * 0.8))
    blend_train = train[:split]
    blend_calib = train[split:]
    if len(blend_calib) >= 20 and len(blend_train) >= 20:
        lr_blend = train_logistic(blend_train, feature_names)
        lr_logits = [logistic_logit(lr_blend, r["features"], None) for r in blend_calib]
        elo_logits = [logit(elo_prob(r, elo_hfa)) for r in blend_calib]
        labels = [r["label"] for r in blend_calib]
        best_brier = float("inf")
        w = 0.0
        while w <= 1.0001:
            preds = [sigmoid((1 - w) * l + w * e) for l, e in zip(lr_logits, elo_logits)]
            b = compute_brier(preds, labels)
            if b < best_brier:
                best_brier = b
                blend_w = w
            w += 0.05

    predictors = {
        "Logistic regression": lambda r: sigmoid(logistic_logit(lr_strong, r["features"], None)),
        "Neural network (MLP)": lambda r: nn(r["features"]),
        "Random Forest": lambda r: rf(r["features"]),
        "XGBoost": lambda r: xgb(r["features"]),
        "LightGBM": lambda r: lgbm(r["features"]),
    }
    return predictors, elo_hfa, blend_w


def _tune_elo_blend(served_logits: list[float], holdout: list[dict], elo_hfa: float) -> float:
    """Tune the Elo blend weight on a chronological holdout by Brier."""
    elo_logits = [logit(elo_prob(r, elo_hfa)) for r in holdout]
    labels = [r["label"] for r in holdout]
    best_brier = float("inf")
    best_w = 0.0
    w = 0.0
    while w <= 1.0001:
        preds = [sigmoid((1 - w) * sl + w * el) for sl, el in zip(served_logits, elo_logits)]
        b = compute_brier(preds, labels)
        if b < best_brier:
            best_brier = b
            best_w = w
        w += 0.05
    return best_w


def fit_deployable_model(
    rows: list[dict],
    feature_names: list[str],
    elo_hfa: float = 30.0,
    mlp_epochs: int = 40,
    model_choice: str | None = None,
) -> tuple[dict, dict]:
    """Fit the deployable stack AND a pure logistic on `rows` (strictly prior-
    only) and deploy whichever has the lower Brier on the trailing 15% of
    `rows` — model selection on Brier, the user's risk metric.

    Each family's Elo blend is tuned on the trailing 20% chronological holdout
    (never a future game). `model_choice` ("stack" | "logistic") forces the
    decision when the walk-forward record already made it, so the calibration
    backtest and today's deployed model serve the identical recipe. Isotonic
    calibration is fit on the exact blend apply_model serves (train == serve).

    Returns (model, choice): model carries featureNames / weights / bias /
    featureStats / isotonicPoints / eloHfa / blendW / stack; choice carries the
    deployed family and both holdout Briers for the monitor.
    """
    selected = list(feature_names)
    stack, blend_w = fit_stack(rows, selected, elo_hfa, mlp_epochs=mlp_epochs)
    full_stack = stack
    full_members = dict(stack.get("members") or {})
    lr = train_logistic(rows, selected)
    n = len(rows)

    # Every deployable family is retained for the walk-forward candidate table,
    # even when the per-date selector picks the pure logistic family.
    # `__stack` keeps the full multi-model stack so the monitor can show the
    # raw blend alongside the chosen family.
    candidate_members = dict(full_members)
    candidate_members["Logistic regression"] = lr
    candidate_members["__stack"] = {"members": full_members, "weights": full_stack.get("weights", {})}

    lr_blend_w = 0.0
    holdout = rows[int(math.floor(n * 0.8)):] if n >= 40 else []
    if len(holdout) >= 20:
        lr_blend_w = _tune_elo_blend(
            [logistic_logit(lr, r["features"], None) for r in holdout], holdout, elo_hfa
        )

    test = rows[min(n, int(math.floor(n * 0.85))):]
    labels = [r["label"] for r in test]

    def _served(r: dict, bw: float, logits_fn) -> float:
        sl = logits_fn(r)
        el = logit(elo_prob(r, elo_hfa))
        return sigmoid((1 - bw) * sl + bw * el if bw > 0 else sl)

    stack_brier = float("inf")
    lr_brier = float("inf")
    if len(labels) >= 20:
        stack_brier = compute_brier(
            [_served(r, blend_w, lambda r: stack_logit(stack, r["features"])) for r in test], labels
        )
        lr_brier = compute_brier(
            [_served(r, lr_blend_w, lambda r: logistic_logit(lr, r["features"], None)) for r in test],
            labels,
        )
    if model_choice is None:
        model_choice = "logistic" if lr_brier < stack_brier - 1e-4 else "stack"

    if model_choice == "logistic":
        stack = {"members": {"Logistic regression": lr}, "weights": {"Logistic regression": 1.0}}
        blend_w = lr_blend_w
        lr_member = lr
    else:
        lr_member = stack["members"].get("Logistic regression") or {}

    served_preds = [
        _served(r, blend_w, lambda r: stack_logit(stack, r["features"])) for r in rows
    ]
    all_labels = [r["label"] for r in rows]
    order = sorted(zip(served_preds, all_labels), key=lambda t: t[0])
    isotonic_points = isotonic_regression([o[0] for o in order], [o[1] for o in order])

    model = {
        "featureNames": selected,
        "weights": lr_member.get("weights", []),
        "bias": lr_member.get("bias", 0.0),
        "featureStats": lr_member.get("featureStats", {}),
        "isotonicPoints": isotonic_points,
        "eloHfa": elo_hfa,
        "blendW": blend_w,
        "stack": stack,
        "candidateMembers": candidate_members,
    }
    choice = {"deployed": model_choice, "stackBrier": stack_brier, "logisticBrier": lr_brier}
    return model, choice


def fit_walk_forward_step(
    train: list[dict],
    feature_names: list[str],
    mlp_epochs: int = 20,
) -> tuple[dict, dict]:
    """Fit one walk-forward block's model for the selection pass.

    Mirrors `run_model_light`'s win-probability recipe exactly so every
    dashboard's point-in-time model is identical: home-field advantage is tuned
    on the first 70% of `train` and the deployable stack-vs-logistic family is
    then chosen on the trailing 15% by holdout Brier (`fit_deployable_model`).
    It skips the Poisson run-scoring layers the selection pass does not need.

    `train` is chronological and strictly prior-only (games before the scored
    day), so nothing from the scored day or any future event leaks in.
    """
    n = len(train)
    elo_hfa = 30.0
    hfa_rows = train[:int(math.floor(n * 0.7))]
    if len(hfa_rows) >= 20:
        best_brier = float("inf")
        for hfa in HFA_GRID:
            preds = [elo_prob(r, hfa) for r in hfa_rows]
            b = compute_brier(preds, [r["label"] for r in hfa_rows])
            if b < best_brier:
                best_brier = b
                elo_hfa = hfa
    model, choice = fit_deployable_model(
        train, feature_names, elo_hfa, mlp_epochs=mlp_epochs
    )
    # The selection pass scores deterministic point estimates exactly like the
    # calibration backtest (Monte Carlo is decided later, on the deployed model).
    model["monteCarloSigma"] = 0.0
    model["monteCarloEnabled"] = False
    return model, choice


def refit_stack_model(
    rows: list[dict],
    feature_names: list[str],
    elo_hfa: float = 30.0,
    monte_carlo_enabled: bool = False,
    monte_carlo_sigma: float = 0.0,
    model_choice: str | None = None,
) -> dict:
    """Fit the production model on ALL `rows` with `feature_names`.

    Used after walk-forward feature selection to rebuild today's deployed model
    on the complete training history using only the features the walk-forward
    record selected. Deploys the stack or a pure logistic by holdout Brier
    (fit_deployable_model), tunes the Elo blend on a chronological holdout of
    `rows` (never a future game) and fits isotonic calibration on the exact
    blend apply_model serves (stack + Elo), so train == serve. The walk-forward
    selection itself supplies the unbiased out-of-sample metrics.
    """
    model, _choice = fit_deployable_model(rows, feature_names, elo_hfa, model_choice=model_choice)
    model["monteCarloSigma"] = monte_carlo_sigma
    model["monteCarloEnabled"] = monte_carlo_enabled
    model["modelChoice"] = _choice
    return model


def refit_logistic_model(
    rows: list[dict],
    feature_names: list[str],
    elo_hfa: float = 30.0,
    monte_carlo_enabled: bool = False,
    monte_carlo_sigma: float = 0.0,
) -> dict:
    """Backward-compatible alias for refit_stack_model."""
    return refit_stack_model(
        rows, feature_names, elo_hfa, monte_carlo_enabled, monte_carlo_sigma
    )
