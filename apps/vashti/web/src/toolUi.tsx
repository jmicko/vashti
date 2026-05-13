import { FileText, Search } from "lucide-react";

export function toolIcon(toolName: string) {
  return toolName.includes("search") ? <Search /> : <FileText />;
}
