import type { ArchiveSummary, ImportConversation } from "./chatgpt";
export class ArchiveClient {
  private worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  private sequence = 0;
  private closed = false;
  private calls = new Map<number, { resolve: (value: any) => void; reject: (error: Error) => void }>();
  constructor(onProgress: (count: number) => void) {
    this.worker.onmessage = event => {
      if ("progress" in event.data) { onProgress(event.data.progress); return; }
      const call = this.calls.get(event.data.id); this.calls.delete(event.data.id);
      if (event.data.error) call?.reject(new Error(event.data.error)); else call?.resolve(event.data.value);
    };
    this.worker.onerror = event => { event.preventDefault(); this.close(); };
    this.worker.onmessageerror = () => this.close();
  }
  private call<T>(action: string, extra: Record<string, unknown> = {}): Promise<T> {
    if (this.closed) return Promise.reject(new Error("Import reader stopped. Select the ZIP again to resume."));
    const id = ++this.sequence;
    return new Promise((resolve, reject) => { this.calls.set(id, { resolve, reject }); this.worker.postMessage({ id, action, ...extra }); });
  }
  open(file: File) { return this.call<ArchiveSummary>("open", { file }); }
  next() { return this.call<ImportConversation | null>("next"); }
  asset(path: string) { return this.call<Blob>("asset", { path }); }
  close() {
    this.closed = true;
    this.worker.terminate();
    for (const call of this.calls.values()) call.reject(new Error("Import stopped"));
    this.calls.clear();
  }
}
