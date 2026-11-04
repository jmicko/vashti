import type { AssistantAction, AssistantEvent, RunResponse } from "./types";

// Advance only over an immutable, contiguous prefix, never over a pending action.
export function actionCursor(actions: AssistantAction[]) {
  let cursor = -1;
  for (const action of actions) {
    if (action.ordinal !== cursor + 1 || !(action.status === "uncertain" || (["completed", "rejected"].includes(action.status) && action.finished_at !== null))) break;
    cursor = action.ordinal;
  }
  return cursor;
}

export function mergeActivity(previous: RunResponse | null, next: RunResponse): RunResponse {
  if (!previous || previous.run.id !== next.run.id) return next;
  const updated = new Map(next.actions.map(a => [a.id, a]));
  const actions = previous.actions.map(action => {
    const replacement = updated.get(action.id);
    updated.delete(action.id);
    return replacement && JSON.stringify(replacement) !== JSON.stringify(action) ? replacement : action;
  });
  actions.push(...updated.values());
  actions.sort((a, b) => a.ordinal - b.ordinal);
  return { ...next, events: [...previous.events, ...next.events.filter(e => !previous.events.some(p => p.id === e.id))], actions };
}

export function activityEntries(events: AssistantEvent[], actions: AssistantAction[]) {
  const linked = new Set<string>();
  const entries: { key: string; event: AssistantEvent | null; action: AssistantAction | null; order: number }[] = [];
  for (const event of events) {
    if (event.kind === "action") {
      let id: unknown;
      try { id = JSON.parse(event.data_json ?? "{}").action_id; } catch { continue; }
      const action = actions.find(a => a.id === id);
      if (action) { linked.add(action.id); entries.push({ key: `a${action.id}`, event: null, action, order: event.id }); }
    } else if (["user", "assistant", "decision", "status"].includes(event.kind)) entries.push({ key: `e${event.id}`, event, action: null, order: event.id });
  }
  // Older checkpoints predate explicit action events. Keep their audit visible.
  for (const action of actions) if (!linked.has(action.id)) {
    const before = events.filter(e => e.created_at <= action.requested_at);
    entries.push({ key: `a${action.id}`, event: null, action, order: (before[before.length - 1]?.id ?? 0) + .5 + action.ordinal / 10000 });
  }
  return entries.sort((a, b) => a.order - b.order);
}
