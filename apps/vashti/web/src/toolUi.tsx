import { BrainCircuit, FileText, MessageSquareText, NotebookPen, Search } from "lucide-react";

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
  if (toolName === "chat_history" || toolName.endsWith("_chat_history")) {
    return <MessageSquareText />;
  }
  return toolName.includes("search") ? <Search /> : <FileText />;
}
