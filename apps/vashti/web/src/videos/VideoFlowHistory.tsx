import { useEffect, useMemo, useRef, useState } from "react";
import { Film, Pencil, Search, Trash2 } from "lucide-react";
import { ActionMenu } from "../ActionMenu";
import { ImagePanel } from "../images/controls";
import { videoRepository, type VideoStorage } from "./repository";
import type { VideoSession } from "./types";
export function VideoFlowHistory({
  storageMode,
  currentId,
  onOpen,
  onRemoved,
}: {
  storageMode: VideoStorage;
  currentId?: string;
  onOpen: (id: string) => void;
  onRemoved: (id: string) => void;
}) {
  const repo = useMemo(() => videoRepository(storageMode), [storageMode]);
  const generation = useRef(0);
  const [loading, setLoading] = useState(true);
  const [sessions, setSessions] = useState<VideoSession[]>([]),
    [search, setSearch] = useState(""),
    [query, setQuery] = useState(""),
    [error, setError] = useState<string | null>(null),
    [more, setMore] = useState(false);
  const [edit, setEdit] = useState<VideoSession | null>(null),
    [title, setTitle] = useState(""),
    [remove, setRemove] = useState<VideoSession | null>(null);
  useEffect(() => {
    const t = setTimeout(() => setQuery(search.trim()), 180);
    return () => clearTimeout(t);
  }, [search]);
  useEffect(() => {
    let current = true;
    const load = () => {
      const request = ++generation.current;
      setLoading(true);
      void repo
        .list(query)
        .then((v) => {
          repo.guard();
          if (current && request === generation.current) {
            setSessions(v);
            setMore(v.length === 50);
            setError(null);
          }
        })
        .catch((e) => {
          if (current && request === generation.current)
            setError(
              e instanceof Error ? e.message : "Could not load video flows",
            );
        })
        .finally(() => {
          if (current && request === generation.current) setLoading(false);
        });
    };
    load();
    window.addEventListener("vashti:image-flows-changed", load);
    const t = setInterval(load, 30000);
    return () => {
      current = false;
      generation.current++;
      clearInterval(t);
      window.removeEventListener("vashti:image-flows-changed", load);
    };
  }, [repo, query]);
  async function action(work: () => Promise<void>) {
    try {
      await work();
      repo.guard();
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    }
  }
  return (
    <>
      <p className="eyebrow">Previous Video Flows</p>
      <label className="history-search">
        <Search size={16} />
        <input
          aria-label="Search video conversations"
          placeholder="Search video conversations"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </label>
      {error && <p role="alert">{error}</p>}
      <div className="chat-link-list">
        {sessions.map((s) => (
          <div
            className={`chat-link-row${currentId === s.id ? " chat-link-row-active" : ""}`}
            key={s.id}
          >
            <button
              type="button"
              className="chat-link"
              onClick={() => onOpen(s.id)}
            >
              <Film className="image-flow-avatar" />
              <span className="chat-link-copy">
                <span className="chat-title-line">
                  <span>{s.title}</span>
                </span>
                <small>Video</small>
              </span>
            </button>
            <ActionMenu label={`Options for ${s.title}`}>
              <button
                type="button"
                onClick={() => {
                  setEdit(s);
                  setTitle(s.title);
                }}
              >
                <Pencil />
                Rename
              </button>
              <button type="button" onClick={() => setRemove(s)}>
                <Trash2 />
                Delete flow
              </button>
            </ActionMenu>
          </div>
        ))}
      </div>
      {more && (
        <button
          type="button"
          disabled={loading}
          onClick={() =>
            void action(async () => {
              const request = generation.current;
              setLoading(true);
              try {
                const values = await repo.list(query, sessions.length);
                if (request !== generation.current) return;
                setSessions((v) => [
                  ...v,
                  ...values.filter((n) => !v.some((p) => p.id === n.id)),
                ]);
                setMore(values.length === 50);
              } finally {
                if (request === generation.current) setLoading(false);
              }
            })
          }
        >
          Load more flows
        </button>
      )}
      {!sessions.length && !error && (
        <p role="status">
          {loading ? "Loading video flows..." : "No video flows"}
        </p>
      )}
      {edit && (
        <ImagePanel title="Rename video flow" onClose={() => setEdit(null)}>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void action(async () => {
                await repo.rename(edit.id, title.trim());
                setEdit(null);
              });
            }}
          >
            <input
              aria-label="Video flow title"
              value={title}
              maxLength={200}
              onChange={(e) => setTitle(e.target.value)}
            />
            <button type="submit" disabled={!title.trim()}>
              Save
            </button>
          </form>
        </ImagePanel>
      )}
      {remove && (
        <ImagePanel title="Delete video flow" onClose={() => setRemove(null)}>
          <p>
            Delete “{remove.title}” and all its clips? This cannot be undone.
          </p>
          <button
            type="button"
            onClick={() =>
              void action(async () => {
                await repo.remove(remove.id);
                onRemoved(remove.id);
                setRemove(null);
              })
            }
          >
            Delete
          </button>
        </ImagePanel>
      )}
    </>
  );
}
