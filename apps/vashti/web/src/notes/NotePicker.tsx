import { useEffect, useRef, useState } from "react";
import { Check, NotebookPen, Search } from "lucide-react";
import { searchServerNotes, type NoteSearchFunction } from "./repository";
import type { NoteSummary } from "./types";

export function useNoteSearch(
  query: string,
  enabled: boolean,
  searchNotes: NoteSearchFunction = searchServerNotes
) {
  const [notes, setNotes] = useState<NoteSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (!enabled) {
      setNotes([]);
      setIsLoading(false);
      setError(null);
      return;
    }

    const requestId = ++requestIdRef.current;
    const timer = window.setTimeout(() => {
      setIsLoading(true);
      setError(null);
      void searchNotes(query)
        .then((response) => {
          if (requestId === requestIdRef.current) {
            setNotes(response.notes);
          }
        })
        .catch((loadError) => {
          if (requestId === requestIdRef.current) {
            setNotes([]);
            setError(loadError instanceof Error ? loadError.message : "Failed to search notes");
          }
        })
        .finally(() => {
          if (requestId === requestIdRef.current) {
            setIsLoading(false);
          }
        });
    }, 120);

    return () => window.clearTimeout(timer);
  }, [enabled, query, searchNotes]);

  return { notes, isLoading, error };
}

export function ComposerNotePicker({
  notes,
  selectedNoteIds,
  activeIndex,
  isLoading,
  error,
  onActiveIndexChange,
  onSelect
}: {
  notes: NoteSummary[];
  selectedNoteIds: Set<string>;
  activeIndex: number;
  isLoading: boolean;
  error: string | null;
  onActiveIndexChange: (index: number) => void;
  onSelect: (note: NoteSummary) => void;
}) {
  return (
    <div className="composer-note-picker" role="listbox" aria-label="Attach a note">
      <header>
        <NotebookPen />
        <strong>Notes</strong>
      </header>
      {isLoading && notes.length === 0 ? (
        <p className="composer-note-picker-state">Searching...</p>
      ) : error ? (
        <p className="composer-note-picker-state error">{error}</p>
      ) : notes.length === 0 ? (
        <p className="composer-note-picker-state">No matching notes</p>
      ) : (
        <div className="composer-note-picker-list">
          {notes.map((note, index) => {
            const selected = selectedNoteIds.has(note.id);
            return (
              <button
                type="button"
                role="option"
                aria-selected={selected}
                className={index === activeIndex ? "composer-note-option active" : "composer-note-option"}
                key={note.id}
                onPointerEnter={() => onActiveIndexChange(index)}
                onPointerDown={(event) => event.preventDefault()}
                onClick={() => onSelect(note)}
              >
                <Search aria-hidden="true" />
                <span>
                  <strong>{note.title}</strong>
                  <small>{note.excerpt || "Empty note"}</small>
                </span>
                {selected && <Check className="composer-note-option-check" aria-hidden="true" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
