import { ChatGptArchive } from "./archive";
let archive: ChatGptArchive | null = null;
let lane = Promise.resolve();
self.onmessage = (event: MessageEvent) => {
  const { id, action, file, path } = event.data;
  lane = lane.then(async () => {
    try {
      let value: unknown;
      if (action === "open") {
        await archive?.close(); archive = new ChatGptArchive(file);
        value = await archive.open(count => { if (count % 25 === 0) self.postMessage({ progress: count }); });
      } else if (action === "next") value = await archive?.next();
      else if (action === "asset") value = await archive?.asset(path);
      else { await archive?.close(); archive = null; }
      self.postMessage({ id, value });
    } catch {
      // Library/JSON errors can contain personal filenames or message snippets.
      self.postMessage({ id, error: "The export could not be read. It may be damaged, password-protected, or outside the import safety limits." });
    }
  });
};
