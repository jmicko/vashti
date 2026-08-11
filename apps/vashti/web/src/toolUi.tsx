import { FileText, NotebookPen, Search } from "lucide-react";

export function toolIcon(toolName: string) {
  if (toolName === "notes" || toolName.endsWith("_note") || toolName.endsWith("_notes")) {
    return <NotebookPen />;
  }
  return toolName.includes("search") ? <Search /> : <FileText />;
}
