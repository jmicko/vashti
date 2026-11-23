import {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";
import {
  ArrowLeft,
  Bold,
  Check,
  Code2,
  Download,
  Eye,
  FilePenLine,
  FileText,
  GitCompare,
  Heading2,
  History,
  Italic,
  List,
  MessageSquare,
  Plus,
  Redo2,
  RotateCcw,
  Search,
  SendHorizontal,
  Trash2,
  Undo2,
  X,
} from "lucide-react";
import { ActionMenu } from "../ActionMenu";
import { ConfirmDialog } from "../common";
import { saveFile } from "../saveFile";
import { privateId, privateImageGuard } from "../privateChatStore";
import { CanvasEditor, type CanvasEditorHandle } from "./Editor";
import {
  canvasHistory,
  canvasVersion,
  clearCanvasDraft,
  loadCanvas,
  mutateCanvas,
  readCanvasDrafts,
  saveCanvasDraft,
  type CanvasDraft,
} from "./repository";
import {
  canvasLanguages,
  canvasScopeKey,
  validateCanvas,
  type CanvasDocument,
  type CanvasLanguage,
  type CanvasMode,
  type CanvasScope,
  type CanvasVersion,
  type CanvasVersionSummary,
  type CanvasWorkspace,
} from "./types";
import type { CanvasOpenRequest } from "./useCanvas";
import "./canvas.css";

const Markdown = lazy(() =>
  import("../MarkdownContent").then((module) => ({
    default: module.MarkdownContent,
  })),
);
type Context = { mode: CanvasMode; chatId: string; scope: CanvasScope };
type PanelProps = Context & {
  request: CanvasOpenRequest | null;
  onConsumed: (key: number) => void;
  onClose: () => void;
  onAsk: (prompt: string) => Promise<boolean | void>;
  isGenerating: boolean;
};
const message = (error: unknown) =>
  error instanceof Error ? error.message : "Canvas is unavailable";
const same = (
  a: Pick<CanvasDraft, "title" | "language" | "content">,
  b: Pick<CanvasDraft, "title" | "language" | "content">,
) =>
  a.title === b.title && a.language === b.language && a.content === b.content;
const draftFor = (v: CanvasVersion): CanvasDraft => ({
  title: v.title,
  language: v.language,
  content: v.content,
  expected_version_id: v.id,
  edit_session_id: privateId("canvas-edit"),
  updated_at: Date.now(),
});
const languageLabel = (language: CanvasLanguage) =>
  ({
    markdown: "Markdown",
    text: "Plain text",
    javascript: "JavaScript",
    typescript: "TypeScript",
    python: "Python",
    rust: "Rust",
    html: "HTML",
    css: "CSS",
    json: "JSON",
    sql: "SQL",
  })[language];

export function CanvasWorkspacePanel(props: PanelProps) {
  const sessionGuard = useRef(privateImageGuard()).current;
  const [workspace, setWorkspace] = useState<CanvasWorkspace | null>(null),
    [selected, setSelected] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [showTrash, setShowTrash] = useState(false),
    [reference, setReference] = useState<CanvasVersion | null>(null),
    [requestedVersion, setRequestedVersion] = useState<string | null>(null);
  const mounted = useRef(true),
    loadSequence = useRef(0),
    handled = useRef(new Set<number>()),
    scopeKey = canvasScopeKey(props.chatId, props.scope);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      loadSequence.current++;
    };
  }, []);
  const reload = useCallback(async () => {
    const seq = ++loadSequence.current;
    try {
      sessionGuard();
      const result = await loadCanvas(props.mode, props.chatId, props.scope);
      if (mounted.current && seq === loadSequence.current) {
        setWorkspace(result);
        setError("");
      }
    } catch (error) {
      if (mounted.current && seq === loadSequence.current)
        setError(message(error));
    }
  }, [props.mode, props.chatId, scopeKey]);
  useEffect(() => {
    void reload();
    const refresh = (event: Event) => {
      const detail = (event as CustomEvent).detail;
      if (detail?.chat === props.chatId && detail?.mode === props.mode)
        void reload();
    };
    window.addEventListener("vashti:canvas-changed", refresh);
    return () => window.removeEventListener("vashti:canvas-changed", refresh);
  }, [reload]);
  useEffect(() => {
    if (!props.isGenerating) return;
    const timer = window.setInterval(() => void reload(), 2000);
    return () => window.clearInterval(timer);
  }, [props.isGenerating, reload]);
  async function create(
    title: string,
    language: CanvasLanguage,
    content = "",
    document_id: string = crypto.randomUUID(),
  ) {
    setBusy(true);
    setError("");
    try {
      sessionGuard();
      const result = await mutateCanvas(props.mode, props.chatId, props.scope, {
        operation: "create",
        document_id,
        title,
        language,
        content,
      });
      if (mounted.current) {
        setWorkspace(result);
        setSelected(document_id);
        setReference(null);
        setRequestedVersion(null);
      }
      return true;
    } catch (error) {
      if (mounted.current) setError(message(error));
      return false;
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  useEffect(() => {
    const request = props.request;
    if (!request || handled.current.has(request.key)) return;
    handled.current.add(request.key);
    props.onConsumed(request.key);
    if (request.copy) {
      const copy = request.copy;
      void (async () => {
        let language: CanvasLanguage = "markdown",
          content = copy.content;
        const { fromMarkdown } = await import("mdast-util-from-markdown"),
          tree = fromMarkdown(content);
        if (tree.children.length === 1 && tree.children[0].type === "code") {
          const node = tree.children[0],
            aliases: Record<string, CanvasLanguage> = {
              js: "javascript",
              jsx: "javascript",
              ts: "typescript",
              tsx: "typescript",
              py: "python",
              rs: "rust",
              md: "markdown",
              txt: "text",
            };
          language =
            aliases[node.lang ?? ""] ??
            (canvasLanguages.includes(node.lang as CanvasLanguage)
              ? (node.lang as CanvasLanguage)
              : "text");
          content = node.value;
        }
        if (mounted.current)
          await create(copy.title, language, content, copy.document_id);
      })().catch((error) => {
        if (mounted.current) setError(message(error));
      });
    }
    if (request.reference) {
      const ref = request.reference;
      setSelected(ref.document_id);
      setRequestedVersion(ref.version_id);
      void canvasVersion(
        props.mode,
        props.chatId,
        ref.document_id,
        ref.version_id,
      )
        .then((value) => {
          if (mounted.current) setReference(value);
        })
        .catch((error) => {
          if (mounted.current) setError(message(error));
        });
    }
  }, [props.request]);
  const documents = workspace?.documents ?? [],
    active = documents.filter((doc) => !doc.trashed),
    doc = documents.find((doc) => doc.id === selected) ?? active[0] ?? null;
  const foreignReference =
    reference &&
    reference.document_id === selected &&
    !documents.some((doc) => doc.id === selected);
  function choose(id: string) {
    setSelected(id);
    setReference(null);
    setRequestedVersion(null);
  }
  function resize(event: React.PointerEvent<HTMLDivElement>) {
    if (event.pointerType === "touch") return;
    const handle = event.currentTarget,
      root = handle.closest<HTMLElement>(".chat-view");
    if (!root) return;
    const update = (x: number) => {
      const rect = root.getBoundingClientRect();
      root.style.setProperty(
        "--canvas-width",
        `${Math.max(360, Math.min(rect.width - 280, rect.right - x))}px`,
      );
    };
    handle.setPointerCapture(event.pointerId);
    handle.onpointermove = (e) => update(e.clientX);
    handle.onpointerup = handle.onpointercancel = () => {
      handle.onpointermove = null;
      handle.onpointerup = null;
      handle.onpointercancel = null;
    };
    event.preventDefault();
  }
  return (
    <aside className="canvas-workspace" aria-label="Canvas workspace">
      <div
        className="canvas-resizer"
        role="separator"
        aria-label="Resize canvas"
        aria-orientation="vertical"
        tabIndex={0}
        onPointerDown={resize}
        onKeyDown={(event) => {
          if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
          event.preventDefault();
          const root = event.currentTarget.closest<HTMLElement>(".chat-view");
          if (root) {
            const width =
              event.currentTarget.parentElement!.getBoundingClientRect().width;
            root.style.setProperty(
              "--canvas-width",
              `${Math.max(360, Math.min(root.clientWidth - 280, width + (event.key === "ArrowLeft" ? 32 : -32)))}px`,
            );
          }
        }}
      />
      <header className="canvas-header">
        <FilePenLine aria-hidden="true" />
        <select
          aria-label="Open canvas"
          value={foreignReference ? "reference" : (doc?.id ?? "")}
          onChange={(e) => choose(e.target.value)}
        >
          {!doc && !foreignReference && <option value="">Canvas</option>}
          {foreignReference && (
            <option value="reference">{reference.title}</option>
          )}
          {active.map((item) => (
            <option key={item.id} value={item.id}>
              {item.version.title}
            </option>
          ))}
          {doc?.trashed && (
            <option value={doc.id}>{doc.version.title} (removed)</option>
          )}
        </select>
        <ActionMenu label="New canvas" icon={<Plus />} disabled={busy}>
          <button
            type="button"
            onClick={() => void create("Untitled document", "markdown")}
          >
            <FileText />
            <span>Document</span>
          </button>
          <button
            type="button"
            onClick={() => void create("Untitled code", "text")}
          >
            <Code2 />
            <span>Code file</span>
          </button>
        </ActionMenu>
        <ActionMenu label="Canvas library">
          <button type="button" onClick={() => setShowTrash((value) => !value)}>
            <Trash2 />
            <span>Removed canvases</span>
          </button>
        </ActionMenu>
        <button
          type="button"
          className="message-icon-button"
          title="Back to chat"
          aria-label="Back to chat"
          onClick={props.onClose}
        >
          <ArrowLeft />
        </button>
      </header>
      {error && (
        <div className="canvas-error" role="alert">
          <span>{error}</span>
          <button
            type="button"
            className="message-icon-button"
            title="Retry loading canvas"
            aria-label="Retry loading canvas"
            onClick={() => void reload()}
          >
            <RotateCcw />
          </button>
        </div>
      )}
      {showTrash ? (
        <section className="canvas-library">
          <header>
            <button
              type="button"
              className="message-icon-button"
              aria-label="Back to canvas"
              onClick={() => setShowTrash(false)}
            >
              <ArrowLeft />
            </button>
            <h3>Removed canvases</h3>
          </header>
          {documents
            .filter((d) => d.trashed)
            .map((item) => (
              <div className="canvas-library-row" key={item.id}>
                <span>{item.version.title}</span>
                <button
                  type="button"
                  title={`Restore ${item.version.title}`}
                  aria-label={`Restore ${item.version.title}`}
                  onClick={() =>
                    void Promise.resolve()
                      .then(() => {
                        sessionGuard();
                        return mutateCanvas(
                          props.mode,
                          props.chatId,
                          props.scope,
                          {
                            operation: "trash",
                            document_id: item.id,
                            expected_version_id: item.version.id,
                            trashed: false,
                          },
                        );
                      })
                      .then((result) => {
                        setWorkspace(result);
                        choose(item.id);
                        setShowTrash(false);
                      })
                      .catch((error) => setError(message(error)))
                  }
                >
                  <RotateCcw />
                </button>
              </div>
            ))}
          {!documents.some((d) => d.trashed) && <p>No removed canvases.</p>}
        </section>
      ) : foreignReference ? (
        <section className="canvas-reference-view">
          <div className="canvas-view-heading">
            <span>Earlier branch · v{reference.version_number}</span>
            <button
              type="button"
              onClick={() =>
                void create(
                  reference.title,
                  reference.language,
                  reference.content,
                )
              }
            >
              <Plus />
              <span>Copy to this branch</span>
            </button>
          </div>
          <CanvasEditor
            value={reference.content}
            language={reference.language}
            readOnly
            onChange={() => {}}
            onSelection={() => {}}
          />
        </section>
      ) : doc ? (
        <CanvasDocumentEditor
          key={`${props.mode}:${scopeKey}:${doc.id}`}
          {...props}
          document={doc}
          workspace={workspace!}
          onWorkspace={setWorkspace}
          initialVersion={
            requestedVersion && reference?.document_id === doc.id
              ? reference
              : null
          }
        />
      ) : (
        <div className="canvas-empty">
          {workspace ? (
            <>
              <FilePenLine />
              <h3>Canvas</h3>
              <button
                type="button"
                disabled={busy}
                onClick={() => void create("Untitled document", "markdown")}
              >
                <Plus />
                <span>New document</span>
              </button>
              <button
                type="button"
                className="secondary-button"
                disabled={busy}
                onClick={() => void create("Untitled code", "text")}
              >
                <Code2 />
                <span>New code file</span>
              </button>
            </>
          ) : (
            <p role="status">Loading canvas...</p>
          )}
        </div>
      )}
    </aside>
  );
}

type EditorProps = PanelProps & {
  document: CanvasDocument;
  workspace: CanvasWorkspace;
  onWorkspace: (workspace: CanvasWorkspace) => void;
  initialVersion: CanvasVersion | null;
};
function CanvasDocumentEditor(props: EditorProps) {
  const { document: doc } = props;
  const sessionGuard = useRef(privateImageGuard()).current;
  const [draft, setDraft] = useState(() => draftFor(doc.version)),
    [baseline, setBaseline] = useState(doc.version),
    [ready, setReady] = useState(false),
    [status, setStatus] = useState("Saved"),
    [error, setError] = useState(""),
    [conflict, setConflict] = useState(false),
    [localSaved, setLocalSaved] = useState(false);
  const [view, setView] = useState<
      "edit" | "preview" | "history" | "version" | "diff" | "comments"
    >("edit"),
    [selectedVersion, setSelectedVersion] = useState<CanvasVersion | null>(
      null,
    ),
    [versions, setVersions] = useState<CanvasVersionSummary[]>([]),
    [selection, setSelection] = useState(""),
    [instruction, setInstruction] = useState(""),
    [asking, setAsking] = useState(false),
    [renaming, setRenaming] = useState(false),
    [comment, setComment] = useState(""),
    [restoreConfirm, setRestoreConfirm] = useState<string | null>(null),
    [rebaseConfirm, setRebaseConfirm] = useState<string | null>(null),
    [recovered, setRecovered] = useState<CanvasDraft[]>([]);
  const editor = useRef<CanvasEditorHandle>(null),
    alive = useRef(true),
    draftRef = useRef(draft),
    baseRef = useRef(baseline),
    docRef = useRef(doc),
    conflictRef = useRef(conflict),
    saving = useRef<Promise<boolean> | null>(null),
    draftQueue = useRef(Promise.resolve());
  draftRef.current = draft;
  baseRef.current = baseline;
  docRef.current = doc;
  conflictRef.current = conflict;
  const dirty = !same(draft, baseline),
    context = { mode: props.mode, chatId: props.chatId, scope: props.scope };
  function replaceDraft(value: CanvasDraft) {
    draftRef.current = value;
    setDraft(value);
  }
  function replaceBase(value: CanvasVersion) {
    baseRef.current = value;
    setBaseline(value);
  }
  const persist = (snapshot: CanvasDraft) => {
    const next = draftQueue.current
      .catch(() => undefined)
      .then(() => {
        sessionGuard();
        return saveCanvasDraft(
          context.mode,
          context.chatId,
          context.scope,
          doc.id,
          snapshot,
        );
      });
    draftQueue.current = next;
    return next.then(
      () => {
        if (alive.current && same(snapshot, draftRef.current))
          setLocalSaved(true);
      },
      (error) => {
        if (alive.current) {
          setLocalSaved(false);
          setError(`Device draft: ${message(error)}`);
        }
        throw error;
      },
    );
  };
  function clearPersisted(snapshot: CanvasDraft) {
    const next = draftQueue.current
      .catch(() => undefined)
      .then(() => {
        sessionGuard();
        return clearCanvasDraft(
          props.mode,
          props.chatId,
          props.scope,
          doc.id,
          snapshot,
        );
      });
    draftQueue.current = next;
    return next;
  }
  useEffect(() => {
    alive.current = true;
    void readCanvasDrafts(props.mode, props.chatId, props.scope, doc.id)
      .then(async (values) => {
        if (!alive.current) return;
        const changed: CanvasDraft[] = [];
        for (const value of values) {
          if (same(value, doc.version))
            await clearCanvasDraft(
              props.mode,
              props.chatId,
              props.scope,
              doc.id,
              value,
            );
          else changed.push(value);
        }
        if (!alive.current) return;
        setRecovered(changed);
        if (changed[0]) {
          replaceDraft(changed[0]);
          setLocalSaved(true);
          setStatus("Recovered draft");
          if (changed[0].expected_version_id !== doc.version.id) {
            setConflict(true);
            conflictRef.current = true;
          }
        }
        setReady(true);
      })
      .catch((error) => {
        if (alive.current) {
          setError(message(error));
          setReady(true);
        }
      });
    return () => {
      alive.current = false;
      if (!same(draftRef.current, baseRef.current))
        void persist({ ...draftRef.current, updated_at: Date.now() }).catch(
          () => undefined,
        );
    };
  }, []);
  useEffect(() => {
    if (doc.version.id === baseRef.current.id) return;
    if (saving.current) return;
    if (
      same(draftRef.current, baseRef.current) ||
      same(draftRef.current, doc.version)
    ) {
      const next = {
        ...draftRef.current,
        title: doc.version.title,
        language: doc.version.language,
        content: doc.version.content,
        expected_version_id: doc.version.id,
      };
      replaceBase(doc.version);
      replaceDraft(next);
      setConflict(false);
      conflictRef.current = false;
      setStatus(
        doc.version.actor === "model"
          ? `Updated by ${doc.version.model_name ?? "model"}`
          : "Saved",
      );
    } else {
      setConflict(true);
      conflictRef.current = true;
      setStatus("Canvas changed");
    }
  }, [doc.version.id]);
  useEffect(() => {
    if (props.initialVersion && props.initialVersion.id !== doc.version.id) {
      setSelectedVersion(props.initialVersion);
      setView("version");
    }
  }, [props.initialVersion?.id]);
  async function flush(attempts = 2): Promise<boolean> {
    if (saving.current) {
      await saving.current;
      if (same(draftRef.current, baseRef.current)) return true;
      if (conflictRef.current) return false;
      return attempts > 0 ? flush(attempts - 1) : false;
    }
    const snapshot = { ...draftRef.current };
    if (same(snapshot, baseRef.current)) return true;
    if (conflictRef.current) return false;
    const task = (async () => {
      if (alive.current) {
        setStatus("Saving...");
        setError("");
      }
      try {
        sessionGuard();
        validateCanvas(snapshot.title, snapshot.language, snapshot.content);
        await persist(snapshot);
        sessionGuard();
        const workspace = await mutateCanvas(
          props.mode,
          props.chatId,
          props.scope,
          {
            operation: "update",
            document_id: doc.id,
            expected_version_id: snapshot.expected_version_id,
            title: snapshot.title,
            language: snapshot.language,
            content: snapshot.content,
            edit_session_id: snapshot.edit_session_id,
          },
        );
        const saved = workspace.documents.find((d) => d.id === doc.id)!.version;
        let draftError = "";
        try {
          await clearPersisted(snapshot);
        } catch (error) {
          draftError = `Device draft: ${message(error)}`;
        }
        // Read the live draft after awaits; never replace typing made during a save.
        const newer = !same(draftRef.current, snapshot),
          next = { ...draftRef.current, expected_version_id: saved.id };
        baseRef.current = saved;
        draftRef.current = next;
        if (alive.current) {
          replaceBase(saved);
          replaceDraft(next);
          setStatus(newer ? "Unsaved changes" : "Saved");
          setConflict(false);
          conflictRef.current = false;
          props.onWorkspace(workspace);
          if (draftError) setError(draftError);
        }
        if (newer) await persist(next).catch(() => undefined);
        return true;
      } catch (error) {
        if (alive.current) {
          setError(message(error));
          setStatus("Not saved");
          if (
            message(error).includes("canvas changed") ||
            message(error).includes("Canvas changed")
          ) {
            setConflict(true);
            conflictRef.current = true;
          }
        }
        return false;
      }
    })();
    saving.current = task;
    try {
      return await task;
    } finally {
      saving.current = null;
    }
  }
  const actions = useRef({ flush, persist });
  actions.current = { flush, persist };
  useEffect(() => {
    if (!ready || !dirty) return;
    const snapshot = { ...draft, updated_at: Date.now() };
    const local = window.setTimeout(
      () => void actions.current.persist(snapshot).catch(() => undefined),
      250,
    );
    const server = window.setTimeout(() => {
      if (!conflictRef.current) void actions.current.flush();
    }, 900);
    return () => {
      window.clearTimeout(local);
      window.clearTimeout(server);
    };
  }, [
    ready,
    draft.title,
    draft.language,
    draft.content,
    draft.expected_version_id,
  ]);
  useEffect(() => {
    if (!dirty) return;
    const unload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    const hidden = () => {
      if (document.visibilityState === "hidden")
        void actions.current
          .persist({ ...draftRef.current, updated_at: Date.now() })
          .catch(() => undefined);
    };
    window.addEventListener("beforeunload", unload);
    document.addEventListener("visibilitychange", hidden);
    return () => {
      window.removeEventListener("beforeunload", unload);
      document.removeEventListener("visibilitychange", hidden);
    };
  }, [dirty]);
  useEffect(() => {
    const retry = () => {
      if (!conflictRef.current) void actions.current.flush();
    };
    window.addEventListener("online", retry);
    return () => window.removeEventListener("online", retry);
  }, []);
  function change(
    value: Partial<Pick<CanvasDraft, "content" | "title" | "language">>,
  ) {
    if (!ready || doc.trashed) return;
    const current = draftRef.current,
      session =
        Date.now() - current.updated_at > 600_000
          ? privateId("canvas-edit")
          : current.edit_session_id;
    replaceDraft({
      ...current,
      ...value,
      edit_session_id: session,
      updated_at: Date.now(),
    });
    setStatus("Unsaved changes");
    setLocalSaved(false);
  }
  async function showHistory() {
    sessionGuard();
    if (dirty) await persist(draftRef.current);
    if (!conflictRef.current) await flush();
    setVersions(await canvasHistory(props.mode, props.chatId, doc.id));
    setRecovered(
      await readCanvasDrafts(props.mode, props.chatId, props.scope, doc.id),
    );
    setView("history");
  }
  async function compare(previous?: CanvasVersion) {
    sessionGuard();
    const base =
      previous ??
      (baseRef.current.parent_id
        ? await canvasVersion(
            props.mode,
            props.chatId,
            doc.id,
            baseRef.current.parent_id,
          )
        : null);
    if (base) {
      setSelectedVersion(base);
      setView("diff");
    }
  }
  async function restore() {
    const expected = restoreConfirm;
    setRestoreConfirm(null);
    if (!selectedVersion || !expected) return;
    try {
      await persist(draftRef.current);
      const workspace = await mutateCanvas(
        props.mode,
        props.chatId,
        props.scope,
        {
          operation: "restore",
          document_id: doc.id,
          expected_version_id: expected,
          version_id: selectedVersion.id,
        },
      );
      const version = workspace.documents.find((d) => d.id === doc.id)!.version;
      await clearCanvasDraft(
        props.mode,
        props.chatId,
        props.scope,
        doc.id,
        draftRef.current,
      );
      replaceBase(version);
      replaceDraft(draftFor(version));
      setConflict(false);
      conflictRef.current = false;
      setStatus("Saved");
      props.onWorkspace(workspace);
      setView("edit");
    } catch (error) {
      setError(message(error));
    }
  }
  async function rebase() {
    const expected = rebaseConfirm;
    setRebaseConfirm(null);
    const latest = docRef.current.version;
    if (latest.id !== expected) {
      setError(
        "The canvas changed again. Compare the latest version before replacing it.",
      );
      return;
    }
    const previous = draftRef.current;
    replaceBase(latest);
    replaceDraft({
      ...draftRef.current,
      expected_version_id: latest.id,
      edit_session_id: privateId("canvas-edit"),
    });
    setConflict(false);
    conflictRef.current = false;
    if (await flush())
      await clearCanvasDraft(
        props.mode,
        props.chatId,
        props.scope,
        doc.id,
        previous,
      );
  }
  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!instruction.trim() || asking || props.isGenerating) return;
    setAsking(true);
    try {
      if (!(await flush()))
        throw new Error(
          "Save or resolve this draft before asking the model to edit it",
        );
      const v = baseRef.current,
        quote = selection.slice(0, 8000),
        prompt = `${instruction.trim()}\n\nCanvas: ${v.title}\nDocument ID: ${doc.id}\nVersion ID: ${v.id}${quote ? `\nApply the request to this selected passage:\n<selected_text>\n${quote}\n</selected_text>` : "\nApply the request to this canvas."}`;
      sessionGuard();
      const result = await props.onAsk(prompt);
      if (result !== false) {
        setInstruction("");
        setSelection("");
      }
    } catch (error) {
      setError(message(error));
    } finally {
      setAsking(false);
    }
  }
  async function addComment() {
    try {
      if (!comment.trim()) return;
      if (!(await flush())) return;
      const workspace = await mutateCanvas(
        props.mode,
        props.chatId,
        props.scope,
        {
          operation: "comment",
          document_id: doc.id,
          expected_version_id: baseRef.current.id,
          quote: selection.slice(0, 8000),
          content: comment,
        },
      );
      props.onWorkspace(workspace);
      setComment("");
    } catch (error) {
      setError(message(error));
    }
  }
  async function download() {
    const ext: Record<CanvasLanguage, string> = {
      markdown: "md",
      text: "txt",
      javascript: "js",
      typescript: "ts",
      python: "py",
      rust: "rs",
      html: "html",
      css: "css",
      json: "json",
      sql: "sql",
    };
    const name = /\.[a-z0-9]{1,8}$/i.test(draft.title)
      ? draft.title
      : `${draft.title}.${ext[draft.language]}`;
    await saveFile(
      new Blob([draft.content], { type: "text/plain" }),
      name,
      "text/plain",
    );
  }
  const safe = (action: () => void | Promise<void>) => () => {
    try {
      sessionGuard();
      void Promise.resolve(action()).catch((error) => setError(message(error)));
    } catch (error) {
      setError(message(error));
    }
  };
  const comments = props.workspace.comments.filter(
      (c) => c.document_id === doc.id,
    ),
    currentComments = comments.filter(
      (c) => c.version_id === baseline.id && !c.resolved,
    );
  const prose = draft.language === "markdown",
    editing = view === "edit",
    historic = view === "version" && selectedVersion;
  return (
    <section
      className="canvas-document"
      aria-label={draft.title}
      onKeyDown={(event) => {
        if (
          (event.ctrlKey || event.metaKey) &&
          event.key.toLowerCase() === "s"
        ) {
          event.preventDefault();
          void flush();
        }
      }}
    >
      <div
        className="canvas-toolbar"
        role="toolbar"
        aria-label="Canvas editing"
      >
        <button
          type="button"
          className="message-icon-button"
          title="Undo"
          aria-label="Undo"
          disabled={!editing || doc.trashed}
          onClick={() => editor.current?.undo()}
        >
          <Undo2 />
        </button>
        <button
          type="button"
          className="message-icon-button"
          title="Redo"
          aria-label="Redo"
          disabled={!editing || doc.trashed}
          onClick={() => editor.current?.redo()}
        >
          <Redo2 />
        </button>
        {prose ? (
          <>
            <button
              type="button"
              className="message-icon-button"
              title="Bold"
              aria-label="Bold"
              disabled={!editing}
              onClick={() => editor.current?.wrap("**", "**")}
            >
              <Bold />
            </button>
            <button
              type="button"
              className="message-icon-button"
              title="Italic"
              aria-label="Italic"
              disabled={!editing}
              onClick={() => editor.current?.wrap("*", "*")}
            >
              <Italic />
            </button>
            <button
              type="button"
              className="message-icon-button"
              title="Heading"
              aria-label="Heading"
              disabled={!editing}
              onClick={() => editor.current?.prefix("## ")}
            >
              <Heading2 />
            </button>
            <button
              type="button"
              className="message-icon-button"
              title="Bulleted list"
              aria-label="Bulleted list"
              disabled={!editing}
              onClick={() => editor.current?.prefix("- ")}
            >
              <List />
            </button>
          </>
        ) : (
          <select
            aria-label="Canvas language"
            value={draft.language}
            disabled={!editing || doc.trashed}
            onChange={(e) =>
              change({ language: e.target.value as CanvasLanguage })
            }
          >
            {canvasLanguages.map((language) => (
              <option key={language} value={language}>
                {languageLabel(language)}
              </option>
            ))}
          </select>
        )}
        {prose && (
          <button
            type="button"
            className="message-icon-button"
            title={view === "preview" ? "Edit Markdown" : "Preview"}
            aria-label={view === "preview" ? "Edit Markdown" : "Preview"}
            aria-pressed={view === "preview"}
            onClick={() => setView(view === "preview" ? "edit" : "preview")}
          >
            <Eye />
          </button>
        )}
        <button
          type="button"
          className="message-icon-button"
          title="Comments"
          aria-label={`Comments${currentComments.length ? `, ${currentComments.length} unresolved` : ""}`}
          aria-pressed={view === "comments"}
          onClick={() => setView(view === "comments" ? "edit" : "comments")}
        >
          <MessageSquare />
          {currentComments.length > 0 && (
            <small>{currentComments.length}</small>
          )}
        </button>
        <ActionMenu label="Canvas actions">
          <button
            type="button"
            disabled={!ready || doc.trashed}
            onClick={() => {
              setView("edit");
              setRenaming(true);
            }}
          >
            <FilePenLine />
            <span>Rename</span>
          </button>
          <button type="button" onClick={safe(showHistory)}>
            <History />
            <span>History</span>
          </button>
          <button
            type="button"
            disabled={!baseline.parent_id}
            onClick={safe(() => compare())}
          >
            <GitCompare />
            <span>Review last change</span>
          </button>
          <button type="button" onClick={safe(download)}>
            <Download />
            <span>Download</span>
          </button>
          <button
            type="button"
            onClick={() => {
              setView("edit");
              editor.current?.search();
            }}
          >
            <Search />
            <span>Find</span>
          </button>
          {prose && (
            <button type="button" onClick={() => change({ language: "text" })}>
              <Code2 />
              <span>Change to code / plain text</span>
            </button>
          )}
          <button
            type="button"
            className="danger-button"
            onClick={safe(async () => {
              if (!(await flush())) return;
              props.onWorkspace(
                await mutateCanvas(props.mode, props.chatId, props.scope, {
                  operation: "trash",
                  document_id: doc.id,
                  expected_version_id: baseRef.current.id,
                  trashed: !doc.trashed,
                }),
              );
            })}
          >
            <Trash2 />
            <span>{doc.trashed ? "Restore canvas" : "Remove from branch"}</span>
          </button>
        </ActionMenu>
      </div>
      {renaming && (
        <form
          className="canvas-rename"
          onSubmit={(event) => {
            event.preventDefault();
            setRenaming(false);
            void flush();
          }}
        >
          <input
            aria-label="Canvas title"
            disabled={!ready || doc.trashed}
            value={draft.title}
            autoFocus
            onChange={(e) => change({ title: e.target.value })}
          />
          <button
            type="submit"
            aria-label="Done renaming"
            title="Done renaming"
          >
            <Check />
          </button>
        </form>
      )}
      {conflict && (
        <div className="canvas-conflict" role="alert">
          <span>Your draft differs from the current canvas.</span>
          <button
            type="button"
            onClick={() => {
              setSelectedVersion(docRef.current.version);
              setView("diff");
            }}
          >
            Compare
          </button>
          <button
            type="button"
            onClick={() => setRebaseConfirm(docRef.current.version.id)}
          >
            Save my draft instead
          </button>
        </div>
      )}
      {error && (
        <p role="alert" className="canvas-error">
          {error}
        </p>
      )}
      {doc.trashed && (
        <p className="canvas-error">
          Removed from this branch. Restore it to edit.
        </p>
      )}
      <div className="canvas-body">
        <div className="canvas-editor-container" hidden={!editing}>
          <CanvasEditor
            ref={editor}
            value={draft.content}
            language={draft.language}
            readOnly={!ready || doc.trashed}
            onChange={(content) => change({ content })}
            onSelection={setSelection}
          />
        </div>
        {view === "preview" && (
          <div className="canvas-preview">
            <Suspense fallback={<p role="status">Loading preview...</p>}>
              <Markdown content={draft.content} deferImages />
            </Suspense>
          </div>
        )}
        {view === "history" && (
          <div className="canvas-history">
            <div className="canvas-view-heading">
              <button
                type="button"
                className="message-icon-button"
                aria-label="Back to editor"
                onClick={() => setView("edit")}
              >
                <ArrowLeft />
              </button>
              <h3>History</h3>
            </div>
            {versions.map((version) => (
              <button
                type="button"
                key={version.id}
                className="canvas-history-entry"
                onClick={safe(async () => {
                  setSelectedVersion(
                    await canvasVersion(
                      props.mode,
                      props.chatId,
                      doc.id,
                      version.id,
                    ),
                  );
                  setView("version");
                })}
              >
                <span>
                  v{version.version_number} · {version.model_name ?? "You"}
                  {version.id === baseline.id ? " · Current" : ""}
                </span>
                <small>
                  {new Date(version.created_at * 1000).toLocaleString()}
                </small>
              </button>
            ))}
            {versions.length > 0 && versions.length % 50 === 0 && (
              <button
                type="button"
                className="secondary-button"
                onClick={safe(async () => {
                  const older = await canvasHistory(
                    props.mode,
                    props.chatId,
                    doc.id,
                    versions[versions.length - 1].version_number,
                  );
                  setVersions((current) => current.concat(older));
                })}
              >
                Load older versions
              </button>
            )}
            {recovered.length > 1 && (
              <>
                <h4>Device drafts</h4>
                {recovered.map((value) => (
                  <button
                    key={value.edit_session_id}
                    type="button"
                    className="canvas-history-entry"
                    onClick={() => {
                      replaceDraft(value);
                      setConflict(value.expected_version_id !== baseline.id);
                      conflictRef.current =
                        value.expected_version_id !== baseline.id;
                      setView("edit");
                    }}
                  >
                    <span>{value.title}</span>
                    <small>{new Date(value.updated_at).toLocaleString()}</small>
                  </button>
                ))}
              </>
            )}
          </div>
        )}
        {historic && (
          <div className="canvas-version-view">
            <div className="canvas-view-heading">
              <button
                type="button"
                className="message-icon-button"
                aria-label="Back to history"
                onClick={() => setView("history")}
              >
                <ArrowLeft />
              </button>
              <span>
                v{selectedVersion.version_number} ·{" "}
                {selectedVersion.model_name ?? "You"}
              </span>
              <button type="button" onClick={() => setView("diff")}>
                <GitCompare />
                <span>Compare</span>
              </button>
              <button
                type="button"
                onClick={() => setRestoreConfirm(docRef.current.version.id)}
              >
                <RotateCcw />
                <span>Restore</span>
              </button>
            </div>
            <CanvasEditor
              key={selectedVersion.id}
              value={selectedVersion.content}
              language={selectedVersion.language}
              readOnly
              onChange={() => {}}
              onSelection={() => {}}
            />
          </div>
        )}
        {view === "diff" && selectedVersion && (
          <div className="canvas-version-view">
            <div className="canvas-view-heading">
              <button
                type="button"
                className="message-icon-button"
                aria-label="Back to editor"
                onClick={() => setView("edit")}
              >
                <ArrowLeft />
              </button>
              <span>
                {conflict
                  ? "Current canvas compared with your draft"
                  : `v${selectedVersion.version_number} compared with ${dirty ? "your draft" : `v${baseline.version_number}`}`}
              </span>
            </div>
            <CanvasEditor
              key={`diff:${selectedVersion.id}`}
              value={draft.content}
              original={selectedVersion.content}
              language={draft.language}
              readOnly
              onChange={() => {}}
              onSelection={() => {}}
            />
          </div>
        )}
        {view === "comments" && (
          <div className="canvas-comments">
            <div className="canvas-view-heading">
              <button
                type="button"
                className="message-icon-button"
                aria-label="Back to editor"
                onClick={() => setView("edit")}
              >
                <ArrowLeft />
              </button>
              <h3>Comments</h3>
            </div>
            {comments.map((value) => (
              <article className="canvas-comment" key={value.id}>
                <header>
                  <strong>{value.model_name ?? "You"}</strong>
                  <small>
                    {value.version_id !== baseline.id
                      ? "Earlier revision"
                      : value.resolved
                        ? "Resolved"
                        : ""}
                  </small>
                  <button
                    type="button"
                    className="message-icon-button"
                    title={
                      value.resolved ? "Reopen comment" : "Resolve comment"
                    }
                    aria-label={
                      value.resolved ? "Reopen comment" : "Resolve comment"
                    }
                    onClick={safe(async () =>
                      props.onWorkspace(
                        await mutateCanvas(
                          props.mode,
                          props.chatId,
                          props.scope,
                          {
                            operation: "resolve_comment",
                            comment_id: value.id,
                            resolved: !value.resolved,
                          },
                        ),
                      ),
                    )}
                  >
                    <Check />
                  </button>
                </header>
                {value.quote && <blockquote>{value.quote}</blockquote>}
                <p>{value.content}</p>
              </article>
            ))}
            <form
              onSubmit={(event) => {
                event.preventDefault();
                void addComment();
              }}
            >
              <textarea
                aria-label="New canvas comment"
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                rows={3}
              />
              <button type="submit" disabled={!comment.trim() || doc.trashed}>
                <Plus />
                <span>Add comment</span>
              </button>
            </form>
          </div>
        )}
      </div>
      <footer className="canvas-footer">
        <div className="canvas-save-status">
          <small role="status">
            {status}
            {dirty && localSaved ? " · Device draft saved" : ""}
          </small>
          <small>v{baseline.version_number}</small>
          {baseline.actor === "model" && baseline.parent_id && (
            <button
              type="button"
              className="message-icon-button"
              title="Review model changes"
              aria-label="Review model changes"
              onClick={safe(() => compare())}
            >
              <GitCompare />
            </button>
          )}
          {status === "Not saved" && !conflict && (
            <button
              type="button"
              className="message-icon-button"
              title="Retry save"
              aria-label="Retry save"
              onClick={() => void flush()}
            >
              <RotateCcw />
            </button>
          )}
        </div>
        {selection && (
          <div className="canvas-selection">
            <span>
              {Array.from(selection).length.toLocaleString()} characters
              selected
            </span>
            <button
              type="button"
              className="message-icon-button"
              title="Clear selection"
              aria-label="Clear selection"
              onClick={() => setSelection("")}
            >
              <X />
            </button>
          </div>
        )}
        <form className="canvas-ask" onSubmit={ask}>
          <input
            aria-label={
              selection ? "Ask AI about selection" : "Ask AI about canvas"
            }
            placeholder={selection ? "Ask about selection" : "Ask about canvas"}
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            disabled={!ready || asking || doc.trashed}
          />
          <button
            type="submit"
            title="Send to chat"
            aria-label="Send to chat"
            disabled={
              !ready ||
              !instruction.trim() ||
              asking ||
              props.isGenerating ||
              doc.trashed
            }
          >
            <SendHorizontal />
          </button>
        </form>
      </footer>
      {restoreConfirm && (
        <ConfirmDialog
          title="Restore canvas version?"
          message="This creates a new revision from the selected version. Later versions remain in history."
          confirmLabel="Restore"
          onCancel={() => setRestoreConfirm(null)}
          onConfirm={() => void restore()}
        />
      )}
      {rebaseConfirm && (
        <ConfirmDialog
          title="Save your draft instead?"
          message="Your draft will replace the current canvas as a new revision. The current version will remain in history."
          confirmLabel="Save draft"
          onCancel={() => setRebaseConfirm(null)}
          onConfirm={() => void rebase()}
        />
      )}
    </section>
  );
}
