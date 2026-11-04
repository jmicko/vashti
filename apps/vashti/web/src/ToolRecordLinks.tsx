import { createContext, useContext } from "react";
import { BrainCircuit, MessagesSquare } from "lucide-react";
import { toolRecordPath, toolRecords, type ToolRecord } from "./toolRecords";
import type { ToolUsageRecord } from "./types";

export const ToolRecordNavigation = createContext<
  ((record: ToolRecord) => void) | null
>(null);

export function ToolRecordLinks({ usage }: { usage: ToolUsageRecord }) {
  const open = useContext(ToolRecordNavigation);
  const records = toolRecords(usage);
  if (!records.length) return null;
  return (
    <nav className="tool-record-links" aria-label="Referenced records">
      {records.map((record) => (
        <a
          key={`${record.kind}:${record.id}`}
          href={toolRecordPath(record)}
          title={`Open ${record.kind === "memory" ? "memory" : "conversation"}: ${record.label}`}
          onClick={(event) => {
            if (
              !open ||
              event.button !== 0 ||
              event.metaKey ||
              event.ctrlKey ||
              event.shiftKey ||
              event.altKey
            )
              return;
            event.preventDefault();
            open(record);
          }}
        >
          {record.kind === "memory" ? <BrainCircuit /> : <MessagesSquare />}
          <span className="tool-record-copy">
            <span>{record.label}</span>
            {record.excerpt && <small>{record.excerpt}</small>}
          </span>
        </a>
      ))}
    </nav>
  );
}
