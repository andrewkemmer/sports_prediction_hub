"""One-shot cache warmer for Streamlit Cloud.

Run this from the project root BEFORE the first user opens the app:

    python3 mlb_streamlit/scripts/warm_cache.py

It executes the full refresh pipeline headlessly (fetch -> enrich -> train ->
predict -> persist) so every disk artifact the dashboard reads on startup
(games, model_state, calibration rows, walk-forward selection at the current
WF_SELECTION_VERSION, pool-cache fingerprints) exists and is current when the
first browser session connects. After this runs once, the app's Rehydrate-First
gate (`app._ui_data_fingerprint` / `_bundle_is_fresh`) matches on the first
click and the dashboard renders in under a second instead of replaying the
5-minute walk-forward.

Usage:
    python3 mlb_streamlit/scripts/warm_cache.py            # warm only what changed
    python3 mlb_streamlit/scripts/warm_cache.py --force    # full rebuild (ignore caches)

Exit code 0 on success, 1 on failure. Prints the resulting data fingerprint
and the walk-forward selection version so you can verify the warm state.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mlb_streamlit import cache  # noqa: E402
from mlb_streamlit.refresh import run_refresh  # noqa: E402
from mlb_streamlit.wf_selection import WF_SELECTION_VERSION  # noqa: E402


def main() -> int:
    force = "--force" in sys.argv
    started = time.time()

    def report(stage: str, pct: int, message: str) -> None:
        print(f"  [{pct:3d}%] {stage}: {message}")

    print(f"Warming cache (force_full={force})…")
    try:
        summary = run_refresh(report=report, force_full=force)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    state = cache.load_model_state() or {}
    fp = state.get("dataFingerprint", "?")[:16]
    elapsed = time.time() - started
    print(f"OK in {elapsed:.1f}s")
    print(f"  gamesTrained : {summary.get('gamesTrained')}")
    print(f"  AUC / Brier  : {summary.get('auc'):.3f} / {summary.get('brier'):.3f}")
    print(f"  dataFingerprint: {fp}…")
    print(f"  WF_SELECTION_VERSION: {WF_SELECTION_VERSION}")
    print("The app will now rehydrate from these artifacts in <1 s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
