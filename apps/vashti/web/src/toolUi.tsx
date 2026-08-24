import { BrainCircuit, FileText, NotebookPen, Search } from "lucide-react";

export function toolIcon(toolName: string) {
  if (toolName === "notes" || toolName.endsWith("_note") || toolName.endsWith("_notes")) {
    return <NotebookPen />;
  }
  if (
    toolName === "memories" ||
    toolName.endsWith("_memories") ||
    toolName.includes("memory") ||
    toolName === "remember"
  ) {
    return <BrainCircuit />;
  }
  return toolName.includes("search") ? <Search /> : <FileText />;
}
