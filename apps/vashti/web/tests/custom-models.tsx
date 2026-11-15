// Isolated device vault and blocked transport: no real account or inference host is used.
import React, { useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { UnsavedSettingsDialog } from "../src/settingsControls";
import { CustomModelsSection } from "../src/settingsPersonas";
import { setPrivateStorageUser } from "../src/privateChatStore";
import type { CustomModelEditorLocation, SettingsGuard } from "../src/types";
import "../src/styles.css";

const json = (value: unknown) => new Response(JSON.stringify(value), {
  headers: { "Content-Type": "application/json" }
});
window.fetch = async input => {
  const url = new URL(String(input), location.origin);
  if (url.pathname === "/api/personas") return json({ personas: [] });
  if (url.pathname === "/api/models/inference-defaults") return json({ temperature: 1, top_p: 0.95, seed: -1 });
  if (url.pathname === "/api/models") return json({ backends: [{
    backend: { id: "fixture", name: "Simulated host" },
    models: [{ name: "long-model-name-for-narrow-screen-layout-testing:latest", capabilities: ["completion"] }]
  }] });
  throw new Error(`Fixture blocked ${url.pathname}`);
};
setPrivateStorageUser("custom-model-fixture", {
  user_id: "custom-model-fixture", key_material: btoa("c".repeat(32))
}, "custom-model-fixture");

function Fixture() {
  const [editor, setEditor] = useState<CustomModelEditorLocation | undefined>({});
  const [pending, setPending] = useState<{ editor?: CustomModelEditorLocation } | null>(null);
  const [busy, setBusy] = useState(false);
  const guard = useRef<SettingsGuard | null>(null);
  function navigate(next?: CustomModelEditorLocation) {
    if (guard.current?.isDirty) setPending({ editor: next }); else setEditor(next);
  }
  return <main style={{ height: "100dvh", display: "grid", gridTemplateRows: "auto minmax(0,1fr)" }}>
    <header className="app-header"><strong>Custom model fixture</strong></header>
    <div className="settings-content"><div className="settings-section custom-model-editor-view">
      <CustomModelsSection key={editor ? `${editor.storageMode}:${editor.personaId ?? "new"}` : "list"}
        editor={editor} onOpenEditor={navigate} onGuardChange={value => { guard.current = value; }}
        onPersonasChanged={async () => {}} onPrivatePersonasChanged={async () => {}} />
    </div></div>
    {pending && <UnsavedSettingsDialog
      isSaving={busy} onCancel={() => setPending(null)} onSave={() => {
        setBusy(true);
        void guard.current?.save().then(saved => {
          if (saved) { setEditor(pending.editor); setPending(null); }
        }).finally(() => setBusy(false));
      }} onDiscard={() => {
        guard.current?.discard(); setEditor(pending.editor); setPending(null);
      }} />}
  </main>;
}
createRoot(document.getElementById("root")!).render(<Fixture />);
