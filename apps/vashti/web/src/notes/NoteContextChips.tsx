import { NotebookPen, X } from "lucide-react";
import type { NoteContextSelection } from "../types";

export function NoteContextChips({
  notes,
  className = "",
  onRemove
}: {
  notes: NoteContextSelection[];
  className?: string;
  onRemove?: (noteId: string) => void;
}) {
  if (notes.length === 0) {
    return null;
  }

  return (
    <div className={["note-context-chips", className].filter(Boolean).join(" ")}>
      {notes.map((note) => (
        <span className="attachment-chip note-context-chip" key={`${note.note_id}:${note.source}`}>
          <NotebookPen />
          <span>{note.title}</span>
          <small>v{note.version_number}</small>
          {onRemove && (
            <button
              type="button"
              className="message-icon-button"
              aria-label={`Remove ${note.title}`}
              onClick={() => onRemove(note.note_id)}
            >
              <X />
            </button>
          )}
        </span>
      ))}
    </div>
  );
}
