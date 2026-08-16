import { ChevronDown, NotebookPen } from "lucide-react";
import type { NoteToolReference } from "./toolReferences";

export function MessageNoteReferences({
  references,
  onOpen
}: {
  references: NoteToolReference[];
  onOpen: (noteId: string) => void;
}) {
  if (references.length === 0) {
    return null;
  }

  if (references.length === 1) {
    const reference = references[0];
    return (
      <div className="message-note-references">
        <NoteReferenceButton reference={reference} onOpen={onOpen} />
      </div>
    );
  }

  return (
    <details className="message-note-references message-note-references-multiple">
      <summary>
        <NotebookPen />
        <span>{references.length} referenced notes</span>
        <ChevronDown className="message-note-references-chevron" />
      </summary>
      <div className="message-note-reference-list">
        {references.map((reference) => (
          <NoteReferenceButton
            key={reference.noteId}
            reference={reference}
            onOpen={onOpen}
          />
        ))}
      </div>
    </details>
  );
}

function NoteReferenceButton({
  reference,
  onOpen
}: {
  reference: NoteToolReference;
  onOpen: (noteId: string) => void;
}) {
  return (
    <button
      type="button"
      className="message-note-reference-button"
      title={`Open note: ${reference.title}`}
      aria-label={
        reference.version === undefined
          ? `Open note: ${reference.title}`
          : `Open note: ${reference.title}, version ${reference.version}`
      }
      onClick={() => onOpen(reference.noteId)}
    >
      <NotebookPen />
      <span>{reference.title}</span>
      {reference.version !== undefined && <small>v{reference.version}</small>}
    </button>
  );
}
