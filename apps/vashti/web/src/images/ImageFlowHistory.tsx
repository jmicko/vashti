import { useEffect, useState } from "react";
import { Image, MoreHorizontal, Pencil, Trash2, X } from "lucide-react";
import { requestJson } from "../api";
import { imageError, imagePath, notifyImageFlows, writeImage } from "./api";
import type { ImageSession } from "./types";
import { ImagePanel } from "./controls";

export function ImageFlowHistory({
  currentId,
  onOpen,
  onRemoved,
}: {
  currentId?: string;
  onOpen: (id: string) => void;
  onRemoved: (id: string) => void;
}) {
  const [sessions, setSessions] = useState<ImageSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [menu, setMenu] = useState<string | null>(null);
  const [rename, setRename] = useState<ImageSession | null>(null);
  const [title, setTitle] = useState("");
  const [remove, setRemove] = useState<ImageSession | null>(null);
  const [busy, setBusy] = useState(false);
  const [more, setMore] = useState(false);
  useEffect(() => {
    let live = true;
    const refresh = () =>
      void requestJson<ImageSession[]>(`${imagePath}/sessions`)
        .then((values) => {
          if (live) {
            setSessions((previous) => [
              ...values,
              ...previous
                .slice(50)
                .filter(
                  (value) => !values.some((current) => current.id === value.id),
                ),
            ]);
            setMore((previous) => (values.length < 50 ? false : previous));
            setLoading(false);
            setError("");
          }
        })
        .catch((error) => {
          if (live) {
            setError(imageError(error));
            setLoading(false);
          }
        });
    void requestJson<ImageSession[]>(`${imagePath}/sessions`)
      .then((values) => {
        if (live) {
          setSessions(values);
          setMore(values.length === 50);
          setLoading(false);
        }
      })
      .catch((error) => {
        if (live) {
          setError(imageError(error));
          setLoading(false);
        }
      });
    window.addEventListener("vashti:image-flows-changed", refresh);
    window.addEventListener("focus", refresh);
    const timer = setInterval(refresh, 30000);
    return () => {
      live = false;
      clearInterval(timer);
      window.removeEventListener("vashti:image-flows-changed", refresh);
      window.removeEventListener("focus", refresh);
    };
  }, []);
  useEffect(() => {
    if (!menu) return;
    const outside = (event: PointerEvent) => {
      if (
        !(event.target instanceof Element) ||
        !event.target.closest(".image-flow-menu,.chat-menu-button")
      )
        setMenu(null);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenu(null);
    };
    window.addEventListener("pointerdown", outside);
    window.addEventListener("keydown", escape);
    return () => {
      window.removeEventListener("pointerdown", outside);
      window.removeEventListener("keydown", escape);
    };
  }, [menu]);
  async function action(task: () => Promise<void>, notify = true) {
    setBusy(true);
    setError("");
    try {
      await task();
      if (notify) notifyImageFlows();
    } catch (error) {
      setError(imageError(error));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <p className="eyebrow">Previous Image Flows</p>
      {error && <p role="alert">{error}</p>}
      {loading && !sessions.length ? (
        <p role="status">Loading flows...</p>
      ) : !sessions.length ? (
        <p>No image flows yet</p>
      ) : (
        <div className="chat-link-list">
          {sessions.map((session) => (
            <div
              className={`chat-link-row ${currentId === session.id ? "chat-link-row-active" : ""}`}
              key={session.id}
            >
              {rename?.id === session.id ? (
                <form
                  className="image-flow-rename"
                  onSubmit={(event) => {
                    event.preventDefault();
                    void action(async () => {
                      await writeImage(
                        `${imagePath}/sessions/${session.id}/title`,
                        { title: title.trim() },
                        "PATCH",
                      );
                      setRename(null);
                    });
                  }}
                >
                  <input
                    autoFocus
                    aria-label="Flow title"
                    maxLength={200}
                    value={title}
                    onChange={(event) => setTitle(event.target.value)}
                  />
                  <button type="submit" disabled={busy || !title.trim()}>
                    Save
                  </button>
                  <button
                    type="button"
                    className="icon-button"
                    aria-label="Cancel rename"
                    onClick={() => setRename(null)}
                  >
                    <X />
                  </button>
                </form>
              ) : (
                <>
                  <button
                    className={`chat-link${currentId === session.id ? " chat-link-active" : ""}`}
                    type="button"
                    onClick={() => {
                      setMenu(null);
                      onOpen(session.id);
                    }}
                  >
                    <Image className="image-flow-avatar" />
                    <span className="chat-link-copy">
                      <span className="chat-title-line">
                        <span>{session.title}</span>
                      </span>
                      <small>
                        {session.mode === "generate"
                          ? "Image flow"
                          : "Image edits"}
                      </small>
                    </span>
                  </button>
                  <button
                    type="button"
                    className="chat-menu-button"
                    aria-label={`Options for ${session.title}`}
                    aria-expanded={menu === session.id}
                    onClick={() =>
                      setMenu(menu === session.id ? null : session.id)
                    }
                  >
                    <MoreHorizontal />
                  </button>
                  {menu === session.id && (
                    <div className="chat-row-menu image-flow-menu">
                      <button
                        type="button"
                        onClick={() => {
                          setMenu(null);
                          setRename(session);
                          setTitle(session.title);
                        }}
                      >
                        <Pencil />
                        Rename
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setMenu(null);
                          setRemove(session);
                        }}
                      >
                        <Trash2 />
                        Remove from history
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>
          ))}
          {more && (
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                void action(async () => {
                  const older = await requestJson<ImageSession[]>(
                    `${imagePath}/sessions?offset=${sessions.length}`,
                  );
                  setSessions((values) => [
                    ...values,
                    ...older.filter(
                      (value) =>
                        !values.some((existing) => existing.id === value.id),
                    ),
                  ]);
                  setMore(older.length === 50);
                }, false)
              }
            >
              Load more flows
            </button>
          )}
        </div>
      )}
      {remove && (
        <ImagePanel
          title="Remove image flow"
          onClose={() => {
            if (!busy) setRemove(null);
          }}
        >
          <p>
            Remove "{remove.title}" from history? Its images stay in Gallery.
          </p>
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              void action(async () => {
                await requestJson(`${imagePath}/sessions/${remove.id}`, {
                  method: "DELETE",
                });
                onRemoved(remove.id);
                setRemove(null);
              })
            }
          >
            Remove
          </button>
          <button type="button" disabled={busy} onClick={() => setRemove(null)}>
            Cancel
          </button>
        </ImagePanel>
      )}
    </>
  );
}
