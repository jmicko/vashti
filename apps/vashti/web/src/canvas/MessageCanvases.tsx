import { FilePenLine } from "lucide-react";
import type { CanvasReference } from "./types";
export function MessageCanvases({
  references,
  onOpen,
}: {
  references: CanvasReference[];
  onOpen: (reference: CanvasReference) => void;
}) {
  if (!references.length) return null;
  return (
    <div className="message-canvas-references">
      {references.map((ref) => (
        <button
          type="button"
          key={ref.document_id}
          title={`Open canvas: ${ref.title}, version ${ref.version_number}`}
          onClick={() => onOpen(ref)}
        >
          <FilePenLine />
          <span>{ref.title}</span>
          <small>v{ref.version_number}</small>
        </button>
      ))}
    </div>
  );
}
