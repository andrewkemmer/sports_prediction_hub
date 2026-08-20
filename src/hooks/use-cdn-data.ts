/**
 * useCdnData — single hook that fetches all MLB dashboard data from the
 * GitHub CDN. Replaces every Convex query and action previously used by
 * Dashboard, CalibrationTab, ModelMonitorTab, and PowerRankingsTab.
 */

import { useCallback, useEffect, useState } from "react";
import { CDN_URL, type CdnPayload } from "@/lib/cdn";

interface CdnDataState {
  /** Full CDN payload — null until the fetch completes */
  payload: CdnPayload | null;
  /** True while the initial fetch is in flight */
  loading: boolean;
  /** Error message if the fetch failed */
  error: string | null;
  /** Timestamp of the last successful fetch */
  fetchedAt: number | null;
}

export function useCdnData(): CdnDataState & { refetch: () => void } {
  const [state, setState] = useState<CdnDataState>({
    payload: null,
    loading: true,
    error: null,
    fetchedAt: null,
  });

  const fetchData = useCallback(async () => {
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const res = await fetch(CDN_URL);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload: CdnPayload = await res.json();
      setState({
        payload,
        loading: false,
        error: null,
        fetchedAt: Date.now(),
      });
    } catch (err) {
      console.error("Error rehydrating dashboard matrix:", err);
      setState((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : "Unknown error",
      }));
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { ...state, refetch: fetchData };
}
