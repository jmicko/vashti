import { useEffect, useRef, useState } from "react";
import { privateImageGuard, readAssistantDraft, writeAssistantDraft } from "../privateChatStore";

// Serialize local writes so a slow encryption cannot overwrite a newer draft.
export function useAssistantDraft<T>(key: string, initial: T) {
  const [value, setValue] = useState(initial);
  const [ready, setReady] = useState(false);
  const [storageError, setStorageError] = useState<string | null>(null);
  const queue = useRef(Promise.resolve());
  const initialRef = useRef(initial);
  const guardRef = useRef<(() => void) | null>(null);
  useEffect(() => {
    let current = true;
    try { guardRef.current = privateImageGuard(); } catch { /* Editing still works without local storage. */ }
    void readAssistantDraft<T>(key).then(draft => {
      if (current) setValue(draft ?? initialRef.current);
    }).catch(() => { if (current) setStorageError("Draft storage unavailable. Keep this page open until saved."); })
      .finally(() => { if (current) setReady(true); });
    return () => { current = false; };
  }, [key]);
  function persist(next: T | null) {
    const guard = guardRef.current;
    queue.current = queue.current.catch(() => undefined).then(async () => {
      if (!guard) throw new Error("Device storage unavailable");
      guard();
      await writeAssistantDraft(key, next);
    }).catch(() => setStorageError("Draft storage unavailable. Keep this page open until saved."));
    return queue.current;
  }
  return {
    value, ready, storageError,
    change(next: T) { setValue(next); return persist(next); },
    async clear() { await persist(null); },
    reset(next: T) { setValue(next); void persist(null); }
  };
}
