import { useEffect, useRef, useState } from "react";
import { assistantApi, activeStatuses } from "./api";
import { actionCursor, mergeActivity } from "./activity";
import type { RunResponse } from "./types";

export function useRun(id: string | null, refresh: number) {
  const [data, setData] = useState<RunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const after = useRef(0);
  const actionsAfter = useRef(-1);
  const dataRef = useRef<RunResponse | null>(null);
  const lastId = useRef(id);
  if (lastId.current !== id) { lastId.current = id; after.current = 0; actionsAfter.current = -1; dataRef.current = null; }
  useEffect(() => { setData(null); setError(null); }, [id]);
  useEffect(() => {
    if (!id) return;
    let current = true, timer = 0, inFlight = false;
    async function poll() {
      if (inFlight) return;
      inFlight = true;
      try {
        const response = await assistantApi.run(id!, after.current, actionsAfter.current);
        if (!current) return;
        after.current = response.events[response.events.length - 1]?.id ?? after.current;
        dataRef.current = mergeActivity(dataRef.current, response);
        actionsAfter.current = actionCursor(dataRef.current.actions);
        setData(dataRef.current);
        setError(null);
        if (response.more_events || activeStatuses.has(response.run.status) || response.actions.some(a => a.status === "executing")) timer = window.setTimeout(poll, response.more_events ? 0 : document.hidden ? 10000 : 1200);
      } catch (e) { if (current) { setError(e instanceof Error ? e.message : "Request failed"); timer = window.setTimeout(poll, 5000); } }
      finally { inFlight = false; }
    }
    void poll();
    const focus = () => { clearTimeout(timer); timer = window.setTimeout(poll, 0); };
    window.addEventListener("focus", focus);
    return () => { current = false; clearTimeout(timer); window.removeEventListener("focus", focus); };
  }, [id, refresh]);
  return { data: data?.run.id === id ? data : null, error };
}
