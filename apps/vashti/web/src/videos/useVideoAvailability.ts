import { useEffect, useState } from "react";
import { ApiRequestError, requestJson } from "../api";
import type { BackendModelGroup } from "../types";
import type { VideoHost } from "./types";

export function useVideoAvailability(userId: string, modelGroups: BackendModelGroup[]) {
  const [state, setState] = useState({ userId, available: false });

  useEffect(() => {
    let current = true;
    let pending = false;
    async function refresh() {
      if (pending) return;
      pending = true;
      try {
        const hosts = await requestJson<VideoHost[]>("/api/videos/hosts");
        if (current) setState({ userId, available: hosts.length > 0 });
      } catch (error) {
        if (current && error instanceof ApiRequestError && [401, 403, 404].includes(error.status)) {
          setState({ userId, available: false });
        }
        // A transient connection failure should not make navigation disappear.
      } finally {
        pending = false;
      }
    }
    const visible = () => { if (!document.hidden) void refresh(); };
    void refresh();
    const timer = window.setInterval(visible, 60000);
    window.addEventListener("focus", visible);
    document.addEventListener("visibilitychange", visible);
    return () => {
      current = false;
      window.clearInterval(timer);
      window.removeEventListener("focus", visible);
      document.removeEventListener("visibilitychange", visible);
    };
  }, [userId, modelGroups]);

  return state.userId === userId && state.available;
}
