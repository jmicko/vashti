import { useCallback, useEffect, useRef, useState } from "react";
import { imageError, requestJson, notifyImageFlows, writeImage } from "./api";
import { useImagePath, deviceImagePath } from "./storage";
import { advancedImageSeed, clearPromptAfterSend, hasImageAncestry } from "./flow";
import {
  activeRun,
  newDraft,
  type Draft,
  type ImageResult,
  type ImageSession,
  type Preferences,
  type Run,
  type RunRequest,
} from "./types";

const merge = <T extends { id: string }>(newer: T[], older: T[]) => [
  ...newer,
  ...older.filter((value) => !newer.some((item) => item.id === value.id)),
];
export function useImageFlow(
  sessionId: string | undefined,
  initialEditing: boolean,
  preferences: Preferences,
) {
  const imagePath = useImagePath();
  const [session, setSession] = useState<ImageSession | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [results, setResults] = useState<ImageResult[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [leaf, setLeaf] = useState<string | null>(null);
  const [more, setMore] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState(false);
  const [persisted, setPersisted] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [editMode, setEditModeState] = useState<"iterative" | "original">(
    "iterative",
  );
  const modeRef = useRef<"iterative" | "original">("iterative");
  const live = useRef(true);
  const valueRef = useRef<Draft | null>(null);
  const saved = useRef<ImageSession | null>(null);
  const queue = useRef<Promise<unknown>>(Promise.resolve());
  const selectionQueue = useRef<Promise<unknown>>(Promise.resolve());
  const pendingSelections = useRef(0);
  const signature = useRef("");
  const leafRef = useRef<string | null>(null);
  const selectionEpoch = useRef(0);
  const historyEpoch = useRef(0);
  const runRef = useRef<Run[]>([]);
  runRef.current = runs;
  const resultRef = useRef<ImageResult[]>([]);
  resultRef.current = results;
  const runOffset = useRef(0),
    resultOffset = useRef(0);
  const hasOlderRuns = useRef(false),
    hasOlderResults = useRef(false);
  const completed = useRef(new Set<string>());
  const watching = useRef(new Map<string, number>());
  const dispatchedSeeds = useRef(new Map<string, string>());
  const submitting = useRef(false);
  const submission = useRef<{ fingerprint: string; id: string } | null>(null);
  function patch(patch: Partial<Draft>) {
    if (!valueRef.current) return;
    const next = { ...valueRef.current, ...patch };
    valueRef.current = next;
    setDraft(next);
  }
  function chooseLeaf(id: string | null) {
    selectionEpoch.current++;
    leafRef.current = id;
    setLeaf(id);
  }
  function followRun(run: Run) {
    chooseLeaf(run.request.draft.parent_result_id);
    watching.current.set(run.id, selectionEpoch.current);
  }
  function setEditMode(mode: "iterative" | "original") {
    modeRef.current = mode;
    setEditModeState(mode);
    const current = valueRef.current;
    if (mode === "original" && current)
      patch({ input_ids: [...current.original_input_ids] });
    if (mode === "iterative" && current && leafRef.current) {
      const selected = resultRef.current.find(
        (result) => result.id === leafRef.current,
      );
      if (selected)
        patch({
          input_ids: [selected.asset_id],
          parent_result_id: selected.id,
        });
    }
  }
  const refresh = useCallback(async (id: string, older = false) => {
    const epoch = historyEpoch.current;
    const selection = selectionEpoch.current;
    const [nextRuns, nextResults] = await Promise.all([
      requestJson<Run[]>(
        `${imagePath}/sessions/${id}/runs?offset=${older ? runOffset.current : 0}`,
      ),
      requestJson<ImageResult[]>(
        `${imagePath}/sessions/${id}/results?offset=${older ? resultOffset.current : 0}`,
      ),
    ]);
    if (!live.current || historyEpoch.current !== epoch)
      return { runs: [], results: [] };
    if (older) {
      runRef.current = merge(runRef.current, nextRuns);
      resultRef.current = merge(resultRef.current, nextResults);
      setRuns(runRef.current);
      setResults(resultRef.current);
      runOffset.current += nextRuns.length;
      resultOffset.current += nextResults.length;
    } else {
      const addedRuns = nextRuns.filter(
        (value) => !runRef.current.some((run) => run.id === value.id),
      ).length;
      const addedResults = nextResults.filter(
        (value) => !resultRef.current.some((result) => result.id === value.id),
      ).length;
      runRef.current = merge(nextRuns, runRef.current);
      resultRef.current = merge(
        nextResults.map((result) => {
          const local = resultRef.current.find(
            (value) => value.id === result.id,
          );
          return local &&
            (pendingSelections.current || selection !== selectionEpoch.current)
            ? { ...result, selected: local.selected }
            : result;
        }),
        resultRef.current,
      );
      setRuns(runRef.current);
      setResults(resultRef.current);
      runOffset.current = runOffset.current
        ? runOffset.current + addedRuns
        : nextRuns.length;
      resultOffset.current = resultOffset.current
        ? resultOffset.current + addedResults
        : nextResults.length;
    }
    if (older || runOffset.current <= 100)
      hasOlderRuns.current = nextRuns.length === 100;
    if (older || resultOffset.current <= 100)
      hasOlderResults.current = nextResults.length === 100;
    setMore(hasOlderRuns.current || hasOlderResults.current);
    return { runs: nextRuns, results: nextResults };
  }, []);
  useEffect(() => {
    live.current = true;
    let mounted = true;
    async function load() {
      if (sessionId) {
        const current = await requestJson<ImageSession>(
          `${imagePath}/sessions/${sessionId}`,
        );
        if (!mounted) return;
        saved.current = current;
        setSession(current);
        valueRef.current = current.draft;
        setDraft(current.draft);
        modeRef.current =
          current.mode === "original" ? "original" : "iterative";
        setEditModeState(modeRef.current);
        signature.current = JSON.stringify(current.draft);
        setPersisted(signature.current);
        const history = await refresh(current.id);
        if (!mounted) return;
        leafRef.current =
          current.active_result_id ??
          history.results.find((value) => value.selected)?.id ??
          null;
        setLeaf(leafRef.current);
        while (
          mounted &&
          !hasImageAncestry(
            runRef.current,
            resultRef.current,
            leafRef.current,
          ) &&
          (hasOlderRuns.current || hasOlderResults.current)
        )
          await refresh(current.id, true);
        for (const run of runRef.current)
          if (!activeRun(run)) completed.current.add(run.id);
      } else {
        const value = newDraft(
          initialEditing ? preferences.edit : preferences.generate,
        );
        valueRef.current = value;
        setDraft(value);
        signature.current = JSON.stringify(value);
        setPersisted(signature.current);
        modeRef.current = preferences.edit.edit_mode ?? "iterative";
        setEditModeState(modeRef.current);
      }
      if (mounted) setLoaded(true);
    }
    void load().catch((error) => {
      if (mounted) setError(imageError(error));
    });
    return () => {
      mounted = false;
      live.current = false;
      historyEpoch.current++;
      if (imagePath === deviceImagePath && saved.current) {
        const id = saved.current.id;
        void import("./deviceRepository").then(module => module.stopDeviceImageFlow(id)).catch(() => {});
      }
    };
  }, []);
  async function save(value: Draft = valueRef.current!) {
    const snapshot = structuredClone(value);
    const editMode = modeRef.current;
    const task = queue.current
      .catch(() => {})
      .then(async () => {
        const current = saved.current;
        const mode =
          snapshot.recipe.workflow === "qwen-image-2.1-edit/v1"
            ? editMode
            : "generate";
        if (
          current &&
          JSON.stringify(snapshot) === signature.current &&
          current.mode === mode
        )
          return current;
        const updated = await writeImage<ImageSession>(
          current
            ? `${imagePath}/sessions/${current.id}`
            : `${imagePath}/sessions`,
          { mode, draft: snapshot, revision: current?.revision },
          current ? "PUT" : "POST",
        );
        saved.current = updated;
        signature.current = JSON.stringify(updated.draft);
        const latest = valueRef.current;
        if (
          latest?.recipe.prompt === snapshot.recipe.prompt &&
          latest.prompt_version_id === snapshot.prompt_version_id &&
          latest.prompt_version_id !== updated.draft.prompt_version_id
        ) {
          const next = {
            ...latest,
            prompt_version_id: updated.draft.prompt_version_id,
          };
          valueRef.current = next;
          if (live.current) setDraft(next);
        }
        notifyImageFlows();
        if (live.current) {
          setSession(updated);
          setPersisted(signature.current);
        }
        return updated;
      });
    queue.current = task;
    return task;
  }
  const draftSignature = JSON.stringify(draft);
  const flush = useRef<(() => Promise<unknown>) | null>(null);
  flush.current =
    draft &&
    (draftSignature !== signature.current ||
      (session?.mode !== "generate" && session?.mode !== editMode)) &&
    (saved.current || draft.recipe.prompt.trim() || draft.input_ids.length)
      ? () => {
          const current = valueRef.current;
          if (
            !current ||
            (JSON.stringify(current) === signature.current &&
              (saved.current?.mode === "generate" ||
                saved.current?.mode === modeRef.current))
          )
            return Promise.resolve();
          return save(current);
        }
      : null;
  useEffect(() => {
    if (!flush.current) return;
    const timer = setTimeout(() => {
      setSaving(true);
      void flush
        .current?.()
        .catch((error) => {
          if (live.current) setError(imageError(error));
        })
        .finally(() => {
          if (live.current) setSaving(false);
        });
    }, 650);
    return () => clearTimeout(timer);
  }, [draftSignature, persisted, editMode]);
  useEffect(
    () => () => {
      void flush.current?.().catch(() => {});
    },
    [],
  );
  useEffect(() => {
    if (!flush.current) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [draftSignature, persisted]);
  async function guard(task: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await task();
    } catch (error) {
      if (live.current) setError(imageError(error));
    } finally {
      if (live.current) setBusy(false);
    }
  }
  const active = runs.find(activeRun);
  const watched =
    active ??
    runs.find(
      (run) => watching.current.has(run.id) && !completed.current.has(run.id),
    );
  function acceptResults(
    next: Run,
    previous: Run | undefined,
    candidates: ImageResult[],
  ) {
    if (!candidates.length) return;
    if (watching.current.get(next.id) !== selectionEpoch.current) {
      setNotice(candidates[0].id);
      return;
    }
    const selected = candidates[0];
    if (
      !previous?.completed_count ||
      leafRef.current === next.request.draft.parent_result_id ||
      resultRef.current.some(
        (value) =>
          value.id === leafRef.current &&
          candidates.some((candidate) => candidate.group_id === value.group_id),
      )
    ) {
      leafRef.current = selected.id;
      setLeaf(selected.id);
    }
    const current = valueRef.current;
    if (!current) return;
    const seed = advancedImageSeed(
      next,
      previous?.completed_count ?? 0,
      current,
      dispatchedSeeds.current.get(next.id),
    );
    if (seed !== null) patch({ recipe: { ...current.recipe, seed } });
    if (next.request.retry_result_id) return;
    const lastInput = resultRef.current.find(
      (value) => value.run_id === next.id,
    )?.asset_id;
    const sameInputs =
      JSON.stringify(current.input_ids) ===
        JSON.stringify(next.request.draft.input_ids) ||
      (lastInput &&
        current.input_ids.length === 1 &&
        current.input_ids[0] === lastInput);
    if (
      modeRef.current === "iterative" &&
      current.recipe.workflow === "qwen-image-2.1-edit/v1" &&
      sameInputs &&
      (current.parent_result_id === next.request.draft.parent_result_id ||
        resultRef.current.some(
          (value) =>
            value.run_id === next.id && value.id === current.parent_result_id,
        ))
    ) {
      patch({ parent_result_id: selected.id, input_ids: [selected.asset_id] });
    }
  }
  useEffect(() => {
    if (!loaded || !watched || !session) return;
    let mounted = true;
    let timer: ReturnType<typeof setTimeout>;
    const id = watched.id,
      sessionKey = session.id;
    if (!watching.current.has(id))
      watching.current.set(id, selectionEpoch.current);
    async function tick() {
      try {
        const next = await requestJson<Run>(`${imagePath}/runs/${id}`);
        if (!mounted || !live.current) return;
        const previous = runRef.current.find((value) => value.id === id);
        setRuns((values) =>
          values.map((value) => (value.id === id ? next : value)),
        );
        if (
          next.completed_count !== previous?.completed_count ||
          !activeRun(next)
        ) {
          const latest = await refresh(sessionKey);
          if (!mounted || !live.current) return;
          const candidates = [...latest.results]
            .reverse()
            .filter((value) => value.run_id === id);
          acceptResults(next, previous, candidates);
          if (!activeRun(next) && !completed.current.has(id)) {
            completed.current.add(id);
            notifyImageFlows();
            const metadata = await requestJson<ImageSession>(
              `${imagePath}/sessions/${sessionKey}`,
            );
            if (mounted && live.current) {
              setSession(metadata);
              if (saved.current?.id === metadata.id)
                saved.current = { ...saved.current, title: metadata.title };
            }
          }
        }
        if (mounted && activeRun(next))
          timer = setTimeout(() => void tick(), 900);
      } catch (error) {
        if (mounted) {
          setError(imageError(error));
          timer = setTimeout(() => void tick(), 2500);
        }
      }
    }
    void tick();
    return () => {
      mounted = false;
      clearTimeout(timer);
    };
  }, [loaded, watched?.id, session?.id, refresh]);
  async function submit(
    value: Draft,
    improveOnly = false,
    retry?: ImageResult,
    confirmed = false,
  ) {
    if (submitting.current || active) return null;
    submitting.current = true;
    setBusy(true);
    setError("");
    try {
      const current = await save();
      const fingerprint = JSON.stringify([
        current.id,
        value,
        improveOnly,
        retry?.id,
        confirmed,
      ]);
      if (submission.current?.fingerprint !== fingerprint)
        submission.current = { fingerprint, id: crypto.randomUUID() };
      const request: RunRequest = {
        client_request_id: submission.current.id,
        draft: value,
        improve_only: improveOnly,
        retry_result_id: retry?.id ?? null,
        fixed_endless_confirmed: confirmed,
      };
      const dispatchSeed = valueRef.current?.recipe.seed;
      const run = await writeImage<Run>(
        `${imagePath}/sessions/${current.id}/runs`,
        request,
      );
      submission.current = null;
      watching.current.set(run.id, selectionEpoch.current);
      if (dispatchSeed !== undefined)
        dispatchedSeeds.current.set(run.id, dispatchSeed);
      if (live.current) setRuns((values) => merge([run], values));
      if (valueRef.current && clearPromptAfterSend(request, current.mode, valueRef.current.recipe.prompt)) {
        patch({ recipe: { ...valueRef.current.recipe, prompt: "" } });
        await save();
      }
      if (!activeRun(run)) {
        const history = await refresh(current.id);
        const result = [...history.results]
          .reverse()
          .find((value) => value.run_id === run.id);
        if (result) acceptResults(run, undefined, [result]);
      }
      return current.id;
    } catch (error) {
      if (live.current) setError(imageError(error));
      return null;
    } finally {
      submitting.current = false;
      if (live.current) setBusy(false);
    }
  }
  async function select(result: ImageResult, selectedLeaf: string) {
    chooseLeaf(selectedLeaf);
    const epoch = selectionEpoch.current;
    const selected = new Map<string, string>();
    let cursor: string | null = selectedLeaf;
    const visited = new Set<string>();
    while (cursor && !visited.has(cursor)) {
      visited.add(cursor);
      const value = resultRef.current.find((value) => value.id === cursor);
      if (!value) break;
      selected.set(value.group_id, value.id);
      cursor = value.parent_result_id;
    }
    setResults((values) =>
      values.map((value) =>
        selected.has(value.group_id)
          ? { ...value, selected: value.id === selected.get(value.group_id) }
          : value,
      ),
    );
    const current = valueRef.current;
    const last =
      resultRef.current.find((value) => value.id === selectedLeaf) ?? result;
    if (current)
      patch({
        parent_result_id: last.id,
        ...(current.recipe.workflow === "qwen-image-2.1-edit/v1" &&
        modeRef.current === "iterative"
          ? { input_ids: [last.asset_id] }
          : {}),
      });
    pendingSelections.current++;
    const task = selectionQueue.current
      .catch(() => {})
      .then(async () => {
        try {
          await writeImage(`${imagePath}/results/${last.id}/select`, {});
          if (live.current && epoch === selectionEpoch.current) await save();
        } catch (error) {
          if (live.current && epoch === selectionEpoch.current) {
            setError(imageError(error));
            if (saved.current) {
              const current = await requestJson<ImageSession>(
                `${imagePath}/sessions/${saved.current.id}`,
              );
              leafRef.current = current.active_result_id;
              setLeaf(current.active_result_id);
            }
          }
        } finally {
          pendingSelections.current--;
        }
        if (!pendingSelections.current && saved.current && live.current)
          await refresh(saved.current.id);
      });
    selectionQueue.current = task;
    await task.catch((error) => {
      if (live.current) setError(imageError(error));
    });
  }
  async function stop() {
    if (!active) return;
    await guard(async () => {
      const next = await writeImage<Run>(
        `${imagePath}/runs/${active.id}/stop`,
        {},
      );
      setRuns((values) =>
        values.map((value) => (value.id === next.id ? next : value)),
      );
    });
  }
  return {
    session,
    draft,
    results,
    runs,
    leaf,
    more,
    error,
    busy,
    saving,
    notice,
    active,
    persisted,
    loaded,
    editMode,
    setEditMode,
    getDraft: () => valueRef.current,
    patch,
    save,
    guard,
    submit,
    select,
    stop,
    chooseLeaf,
    followRun,
    setNotice,
    setError,
    loadOlder: () =>
      saved.current
        ? guard(async () => {
            await refresh(saved.current!.id, true);
          })
        : Promise.resolve(),
  };
}
