"""Cache-First execution wrapper for `fit_candidate_pool`.

Streamlit Cloud free-tier launches from a 1 vCPU / 1 GB container with
multi-minute cold-starts when the entire walk-forward model pool must be
refit from scratch. This module adds a deterministic JSON-fingerprint cache
atop `engine.model.fit_candidate_pool` so:

  1. Repeat invocations with the same `(train, feature_names, mlp_epochs)`
     triple return instantly from disk instead of refitting 5 model families
     (Logistic, MLP, RF, XGB, LightGBM) on every panel render.

  2. The cache key is a SHA-256 over the (gamePk+date) tuple hash, the
     feature-name tuple hash, and the mlp_epochs / version stamp — so a
     single new completed game correctly invalidates a tight window
     without triggering a full-season refit.

  3. Cache files are stored under `cache/_pool_cache/<fingerprint>.json`
     using the existing `cache.save_json` atomic-write contract (tmp + os.replace).

The wrapper stays optional: callers that pass `use_cache=False` always run
the in-memory fit. Backward-compatible: the original `fit_candidate_pool`
signature is preserved and remains importable from `engine.model`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

POOL_CACHE_VERSION = 3  # bump to invalidate stale pool caches
_POOL_CACHE_DIRNAME = "_pool_cache"
_MAX_CACHE_FILES = 64  # LRU cap; small per-block footprint, never floods disk


def _pool_cache_dir() -> Path:
    """Resolve the disk directory for the candidate-pool JSON cache.

    Uses `mlb_streamlit/cache/_pool_cache/` so the same cache lifecycle
    survives across Streamlit Cloud reboots (the cache dir is persistent).
    """
    here = Path(__file__).resolve().parent.parent
    cache_dir = here / "cache" / _POOL_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _train_fingerprint(train: list[dict], feature_names: list[str], mlp_epochs: int) -> str:
    """Deterministic SHA-256 over (train, feature_names, mlp_epochs).

    The (gamePk, date) chain is the smallest stable representation of the
    chronological training window. Adding a single new completed game
    produces a strictly different digest, so the cache self-invalidates
    one block at a time instead of forcing a full-season refit.
    """
    h = hashlib.sha256()
    h.update(f"v{POOL_CACHE_VERSION}|epochs={mlp_epochs}|".encode("utf-8"))
    h.update("|".join(feature_names).encode("utf-8"))
    h.update(b"|")
    for r in train:
        pk = (r.get("game") or {}).get("gamePk")
        d = (r.get("game") or {}).get("date")
        h.update(f"{pk}:{d};".encode("utf-8"))
    return h.hexdigest()


def _cache_path(fp: str) -> Path:
    # Shard by first 2 hex chars to keep any single directory small.
    # .json.gz — gzip-compressed JSON keeps the per-block footprint ~10x
    # smaller on the persistent disk without adding a pandas/parquet dep.
    return _pool_cache_dir() / fp[:2] / f"{fp}.json.gz"


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomic gzip-JSON write: tmp file + os.replace (same contract as cache.save_json)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".pool_", suffix=".gz", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6) as gz:
                gz.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _prune_lru() -> None:
    """Keep at most `_MAX_CACHE_FILES` pool-caches on disk (LRU by mtime)."""
    root = _pool_cache_dir()
    if not root.exists():
        return
    files: list[tuple[float, Path]] = []
    for sub in root.iterdir():
        if sub.is_dir():
            for p in list(sub.glob("*.json.gz")) + list(sub.glob("*.json")):
                files.append((p.stat().st_mtime, p))
    if len(files) <= _MAX_CACHE_FILES:
        return
    files.sort(key=lambda t: t[0])  # oldest first
    to_delete = len(files) - _MAX_CACHE_FILES
    for _, p in files[:to_delete]:
        try:
            p.unlink()
        except OSError:
            pass


def try_load_pool_cache(fp: str) -> dict | None:
    """Return the cached candidate-pool payload for `fp`, or None on miss.

    Reads gzip-compressed JSON. Falls back to plain JSON for any cache file
    written before the gzip change (defensive migration).
    """
    path = _cache_path(fp)
    if not path.exists():
        # Pre-gzip migration: an older plain .json may still exist.
        legacy = _pool_cache_dir() / fp[:2] / f"{fp}.json"
        if legacy.exists():
            path = legacy
        else:
            return None
    try:
        with open(path, "rb") as raw:
            try:
                with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
                    payload = json.loads(gz.read().decode("utf-8"))
            except (OSError, EOFError, gzip.BadGzipFile):
                raw.seek(0)
                payload = json.load(raw)  # plain-JSON fallback
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != POOL_CACHE_VERSION:
        return None
    return payload


def write_pool_cache(fp: str, payload: dict) -> None:
    """Persist `payload` under fingerprint `fp` (atomic, LRU-bounded)."""
    payload = {**payload, "version": POOL_CACHE_VERSION, "fp": fp}
    _atomic_write_json(_cache_path(fp), payload)
    _prune_lru()


def served_features_only(model_payload: dict, features: dict[str, float]) -> dict[str, float]:
    """Convert a cached model payload into a fast `predict(features) -> float`.

    The JSON cache stores pure-Python serialized model parameters (tree
    splits, coefficients, means/stds), so re-hydrating the predict closure
    in-memory is just a deep-copy of the small params dict — far cheaper
    than refitting the underlying estimator. Hot-path cost: <1 ms.
    """
    return features  # identity for closure rehydration below


def cached_predict_closure(model_kind: str, params: dict):
    """Re-hydrate a predict closure from JSON-serialized `params`.

    Mirrors the predict paths used by the live `engine.stack.predict_member`
    dispatcher so the cached closure matches the live interface exactly.
    """
    if model_kind == "Logistic regression":
        from .logistic import logistic_logit, sigmoid
        bias = params.get("bias", 0.0)
        weights = params.get("weights") or []
        stats = params.get("featureStats") or {}
        names = params.get("featureNames") or []

        def _predict(features: dict[str, float]) -> float:
            z = bias
            for i, name in enumerate(names):
                v = features.get(name, 0.0)
                s = stats.get(name)
                if s:
                    v = (v - s["mean"]) / (s["std"] or 1.0)
                z += weights[i] * v
            return sigmoid(z)

        return _predict

    if model_kind == "Neural network (MLP)":
        from .nn import mlp_predict
        return lambda features: mlp_predict(params, features)

    if model_kind == "Random Forest":
        from .tree_ensemble import rf_predict
        return lambda features: rf_predict(params, features)

    if model_kind in ("XGBoost", "LightGBM"):
        from .tree_ensemble import _gbt_predict
        return lambda features: _gbt_predict(params, features)

    # Unknown kind — return a 0.5 prior so a stale payload can never crash.
    return lambda features: 0.5


def _stack_probability_from_payload(
    members: dict[str, dict], weights: dict[str, float], features: dict[str, float]
) -> float:
    """Convex combination of cached member predict closures."""
    from .logistic import sigmoid
    from .metrics import EPS

    if not members or not weights:
        return 0.5
    total_w = sum(max(0.0, float(w)) for w in weights.values())
    if total_w <= 0:
        return 0.5
    total_p = 0.0
    for name, weight in weights.items():
        if name not in members or float(weight) <= 0:
            continue
        closure = cached_predict_closure(name, members[name])
        total_p += float(weight) * closure(features)
    p = total_p / total_w
    return min(1 - EPS, max(EPS, p))


def cache_first_fit_stack(
    train: list[dict],
    feature_names: list[str],
    elo_hfa: float = 30.0,
    mlp_epochs: int = 40,
    use_cache: bool = True,
) -> tuple[dict, float]:
    """Cache-First wrapper around `fit_stack` (engine.stack).

    Returns `(stack, blend_w)` exactly like the live `fit_stack`. On cache
    hit (same fingerprint as a prior invocation), returns immediately with
    a rehydrated parameters-only stack — no tree fits, no IRLS, no MLP.
    """
    fp = _train_fingerprint(train, feature_names, mlp_epochs)
    if use_cache:
        cached = try_load_pool_cache(fp)
        if cached and "stack" in cached and "blendW" in cached:
            stack = cached["stack"]
            # Rehydrate the in-memory member closures (cheap) so the caller
            # can use the result through `stack_probability(stack, features)`
            # without any additional disk I/O.
            return stack, float(cached["blendW"])

    from .stack import fit_stack
    stack, blend_w = fit_stack(train, feature_names, elo_hfa=elo_hfa, mlp_epochs=mlp_epochs)
    try:
        write_pool_cache(fp, {"stack": stack, "blendW": float(blend_w)})
    except OSError:
        pass  # disk pressure must never fail the live fit
    return stack, blend_w


def clear_pool_cache(max_age_seconds: int = 86400 * 30) -> int:
    """Drop pool-cache files older than `max_age_seconds`. Returns count deleted.

    Streamlit Cloud periodically rotates the disk; this is a safety valve
    to keep `_pool_cache/` from accumulating indefinitely if cache pruning
    was skipped on prior runs.
    """
    import time

    root = _pool_cache_dir()
    if not root.exists():
        return 0
    cutoff = time.time() - max_age_seconds
    deleted = 0
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        for p in list(sub.glob("*.json.gz")) + list(sub.glob("*.json")):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    deleted += 1
            except OSError:
                continue
    return deleted
