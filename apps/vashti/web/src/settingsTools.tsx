import {
  FormEvent,
  useCallback,
  useEffect,
  useState
} from "react";
import {
  BrainCircuit,
  FileText,
  MessageSquareText,
  NotebookPen,
  Save,
  Search,
  Wrench
} from "lucide-react";
import { requestJson } from "./api";
import { ConfirmDialog } from "./common";
import { PermissionTagEditor } from "./permissionTags";
import {
  SettingsPanel,
  SettingsSaveBanner,
  ToggleSwitch,
  ToolPromptEditor
} from "./settingsControls";
import {
  permissionTagPayload,
  permissionTagSetsEqual
} from "./settingsModelHelpers";
import type { AdminBackendModelGroup, AdminModelsResponse, PermissionTag, ToolSettings } from "./types";

export function ToolsSettingsPanel({ onToolsChanged }: { onToolsChanged: () => Promise<void> }) {
  const [settings, setSettings] = useState<ToolSettings | null>(null);
  const [toolsEnabled, setToolsEnabled] = useState(false);
  const [ollamaSearchEnabled, setOllamaSearchEnabled] = useState(false);
  const [ollamaFetchEnabled, setOllamaFetchEnabled] = useState(false);
  const [ollamaApiKey, setOllamaApiKey] = useState("");
  const [braveSearchEnabled, setBraveSearchEnabled] = useState(false);
  const [braveApiKey, setBraveApiKey] = useState("");
  const [directFetchEnabled, setDirectFetchEnabled] = useState(false);
  const [notesSemanticSearchEnabled, setNotesSemanticSearchEnabled] = useState(false);
  const [notesEmbeddingBackendId, setNotesEmbeddingBackendId] = useState("");
  const [notesEmbeddingModel, setNotesEmbeddingModel] = useState("");
  const [embeddingBackends, setEmbeddingBackends] = useState<AdminBackendModelGroup[]>([]);
  const [toolSystemPrompt, setToolSystemPrompt] = useState("");
  const [webSearchToolPrompt, setWebSearchToolPrompt] = useState("");
  const [webFetchToolPrompt, setWebFetchToolPrompt] = useState("");
  const [availableTags, setAvailableTags] = useState<PermissionTag[]>([]);
  const [defaultToolTags, setDefaultToolTags] = useState<PermissionTag[]>([]);
  const [toolPermissionTags, setToolPermissionTags] = useState<Record<string, PermissionTag[]>>({});
  const [clearKeyTarget, setClearKeyTarget] = useState<"ollama" | "brave" | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isClearingKey, setIsClearingKey] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isDirty = Boolean(
    settings &&
      (toolsEnabled !== settings.tools_enabled ||
        ollamaSearchEnabled !== settings.ollama_web_search_enabled ||
        ollamaFetchEnabled !== settings.ollama_web_fetch_enabled ||
        braveSearchEnabled !== settings.brave_search_enabled ||
        directFetchEnabled !== settings.direct_web_fetch_enabled ||
        notesSemanticSearchEnabled !== settings.notes_semantic_search_enabled ||
        notesEmbeddingBackendId !== (settings.notes_embedding_backend_id ?? "") ||
        notesEmbeddingModel !== (settings.notes_embedding_model ?? "") ||
        ollamaApiKey.trim() ||
        braveApiKey.trim() ||
        toolSystemPrompt !== settings.tool_system_prompt ||
        webSearchToolPrompt !== settings.web_search_tool_prompt ||
        webFetchToolPrompt !== settings.web_fetch_tool_prompt ||
        !permissionTagSetsEqual(defaultToolTags, settings.default_tool_permission_tags) ||
        !toolPermissionSetsEqual(toolPermissionTags, settings.tool_permissions))
  );
  const toolsEnabledChanged = Boolean(settings && toolsEnabled !== settings.tools_enabled);
  const ollamaSearchChanged = Boolean(
    settings && ollamaSearchEnabled !== settings.ollama_web_search_enabled
  );
  const ollamaFetchChanged = Boolean(
    settings && ollamaFetchEnabled !== settings.ollama_web_fetch_enabled
  );
  const braveSearchChanged = Boolean(
    settings && braveSearchEnabled !== settings.brave_search_enabled
  );
  const directFetchChanged = Boolean(
    settings && directFetchEnabled !== settings.direct_web_fetch_enabled
  );
  const notesSemanticChanged = Boolean(
    settings && notesSemanticSearchEnabled !== settings.notes_semantic_search_enabled
  );
  const ollamaKeyChanged = Boolean(ollamaApiKey.trim());
  const braveKeyChanged = Boolean(braveApiKey.trim());
  const toolSystemPromptChanged = Boolean(
    settings && toolSystemPrompt !== settings.tool_system_prompt
  );
  const webSearchToolPromptChanged = Boolean(
    settings && webSearchToolPrompt !== settings.web_search_tool_prompt
  );
  const webFetchToolPromptChanged = Boolean(
    settings && webFetchToolPrompt !== settings.web_fetch_tool_prompt
  );

  function applyToolSettings(response: ToolSettings) {
    setSettings(response);
    setToolsEnabled(response.tools_enabled);
    setOllamaSearchEnabled(response.ollama_web_search_enabled);
    setOllamaFetchEnabled(response.ollama_web_fetch_enabled);
    setOllamaApiKey("");
    setBraveSearchEnabled(response.brave_search_enabled);
    setBraveApiKey("");
    setDirectFetchEnabled(response.direct_web_fetch_enabled);
    setNotesSemanticSearchEnabled(response.notes_semantic_search_enabled);
    setNotesEmbeddingBackendId(response.notes_embedding_backend_id ?? "");
    setNotesEmbeddingModel(response.notes_embedding_model ?? "");
    setToolSystemPrompt(response.tool_system_prompt);
    setWebSearchToolPrompt(response.web_search_tool_prompt);
    setWebFetchToolPrompt(response.web_fetch_tool_prompt);
    setAvailableTags(response.available_tags);
    setDefaultToolTags(response.default_tool_permission_tags);
    setToolPermissionTags(
      Object.fromEntries(
        response.tool_permissions.map((tool) => [tool.tool_id, tool.permission_tags])
      )
    );
  }

  const loadToolSettings = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [response, models] = await Promise.all([
        requestJson<ToolSettings>("/api/settings/tools"),
        requestJson<AdminModelsResponse>("/api/admin/models")
      ]);
      applyToolSettings(response);
      setEmbeddingBackends(models.backends);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load tool settings");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadToolSettings();
  }, [loadToolSettings]);

  useEffect(() => {
    if (!status) {
      return;
    }

    const timeout = window.setTimeout(() => setStatus(null), 3000);
    return () => window.clearTimeout(timeout);
  }, [status]);

  useEffect(() => {
    if (isDirty && status) {
      setStatus(null);
    }
  }, [isDirty, status]);

  async function saveToolSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSaving(true);
    setStatus(null);
    setError(null);

    const payload: Record<string, unknown> = {
      tools_enabled: toolsEnabled,
      ollama_web_search_enabled: ollamaSearchEnabled,
      ollama_web_fetch_enabled: ollamaFetchEnabled,
      clear_ollama_api_key: false,
      brave_search_enabled: braveSearchEnabled,
      clear_brave_search_api_key: false,
      direct_web_fetch_enabled: directFetchEnabled,
      notes_semantic_search_enabled: notesSemanticSearchEnabled,
      notes_embedding_backend_id: notesEmbeddingBackendId || null,
      notes_embedding_model: notesEmbeddingModel || null,
      tool_system_prompt: toolSystemPrompt,
      web_search_tool_prompt: webSearchToolPrompt,
      web_fetch_tool_prompt: webFetchToolPrompt,
      default_tool_permission_tags: permissionTagPayload(defaultToolTags),
      tool_permissions: Object.entries(toolPermissionTags).map(([toolId, tags]) => ({
        tool_id: toolId,
        permission_tags: permissionTagPayload(tags)
      }))
    };
    if (ollamaApiKey.trim()) {
      payload.ollama_api_key = ollamaApiKey.trim();
    }
    if (braveApiKey.trim()) {
      payload.brave_search_api_key = braveApiKey.trim();
    }

    try {
      const response = await requestJson<ToolSettings>("/api/settings/tools", {
        method: "PATCH",
        body: JSON.stringify(payload)
      });
      applyToolSettings(response);
      await onToolsChanged();
      setStatus("Tool settings saved.");
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save tool settings");
    } finally {
      setIsSaving(false);
    }
  }

  function revertToolSettings() {
    if (!settings) {
      return;
    }
    applyToolSettings(settings);
    setStatus(null);
    setError(null);
  }

  async function clearToolKey(target: "ollama" | "brave") {
    setIsClearingKey(true);
    setStatus(null);
    setError(null);

    try {
      const response = await requestJson<ToolSettings>("/api/settings/tools", {
        method: "PATCH",
        body: JSON.stringify({
          tools_enabled: toolsEnabled,
          ollama_web_search_enabled: ollamaSearchEnabled,
          ollama_web_fetch_enabled: ollamaFetchEnabled,
          clear_ollama_api_key: target === "ollama",
          brave_search_enabled: braveSearchEnabled,
          clear_brave_search_api_key: target === "brave",
          direct_web_fetch_enabled: directFetchEnabled,
          notes_semantic_search_enabled: notesSemanticSearchEnabled,
          notes_embedding_backend_id: notesEmbeddingBackendId || null,
          notes_embedding_model: notesEmbeddingModel || null,
          tool_system_prompt: toolSystemPrompt,
          web_search_tool_prompt: webSearchToolPrompt,
          web_fetch_tool_prompt: webFetchToolPrompt,
          default_tool_permission_tags: permissionTagPayload(defaultToolTags),
          tool_permissions: Object.entries(toolPermissionTags).map(([toolId, tags]) => ({
            tool_id: toolId,
            permission_tags: permissionTagPayload(tags)
          }))
        })
      });
      applyToolSettings(response);
      await onToolsChanged();
      setStatus(target === "ollama" ? "Ollama API key cleared." : "Brave Search API key cleared.");
      setClearKeyTarget(null);
    } catch (clearError) {
      setError(clearError instanceof Error ? clearError.message : "Failed to clear API key");
    } finally {
      setIsClearingKey(false);
    }
  }

  return (
    <SettingsPanel eyebrow="Admin" title="Tools" className="tools-settings-section">
      {error && <p className="error">{error}</p>}
      {!settings && isLoading && <p className="status-message">Loading tool settings...</p>}

      {settings && (
        <form
          className="settings-form settings-form-with-banner tools-settings-form"
          onSubmit={saveToolSettings}
        >
          <SettingsSaveBanner
            isDirty={isDirty}
            status={status}
            dirtyTitle="Unsaved tool changes"
            dirtyDescription="Save to apply these settings to future generations."
            savedDescription="Saved changes are active for future generations."
          >
            <button
              type="button"
              className="secondary-button"
              disabled={isSaving}
              onClick={revertToolSettings}
            >
              Revert
            </button>
            <button type="submit" disabled={isSaving}>
              <Save />
              <span>{isSaving ? "Saving..." : "Save"}</span>
            </button>
          </SettingsSaveBanner>

          <section className="settings-subsection">
            <ToggleSwitch
              icon={<Wrench />}
              label="Enable tools globally"
              description="When enabled, Vashti only sends tool schemas to Ollama models whose capabilities include tools."
              checked={toolsEnabled}
              isChanged={toolsEnabledChanged}
              onChange={setToolsEnabled}
            />
            <PermissionTagEditor
              label="Default tool tags"
              tags={defaultToolTags}
              availableTags={availableTags}
              onChange={setDefaultToolTags}
            />
            <details className="tool-details">
              <summary>Tool instructions</summary>
              <label
                className={
                  toolSystemPromptChanged ? "setting-field setting-field-changed" : "setting-field"
                }
              >
                <span>Tool system prompt</span>
                <textarea
                  value={toolSystemPrompt}
                  onChange={(event) => setToolSystemPrompt(event.target.value)}
                />
              </label>
              <button
                type="button"
                className="secondary-button prompt-reset-button"
                disabled={!settings || toolSystemPrompt === settings.default_tool_system_prompt}
                onClick={() => setToolSystemPrompt(settings.default_tool_system_prompt)}
              >
                Reset to Default
              </button>
            </details>
          </section>

          <section className="settings-subsection">
            <div>
              <p className="eyebrow">Vashti</p>
              <h2>Notes</h2>
              <p className="status-message">
                Controls who may use Notes on this server. People choose what models can do with
                their notes in Personal → Notes.
              </p>
            </div>
            <div className="tool-setting-heading">
              <NotebookPen aria-hidden="true" />
              <div>
                <strong>Notes tools</strong>
                <p>Search, read, create, edit, and move permitted notes to trash.</p>
              </div>
            </div>
            <PermissionTagEditor
              label="Tags"
              tags={toolPermissionTags.notes ?? []}
              availableTags={availableTags}
              onChange={(tags) =>
                setToolPermissionTags((current) => ({
                  ...current,
                  notes: tags
                }))
              }
            />
          </section>

          <section className="settings-subsection">
            <div>
              <p className="eyebrow">Vashti</p>
              <h2>Past Chats</h2>
              <p className="status-message">
                Controls who may let models search their completed server chat history. Each person
                opts in from Personal → Memories.
              </p>
            </div>
            <div className="tool-setting-heading">
              <MessageSquareText aria-hidden="true" />
              <div>
                <strong>Past-chat tools</strong>
                <p>Search historical messages and read the matching branch context.</p>
              </div>
            </div>
            <PermissionTagEditor
              label="Tags"
              tags={toolPermissionTags.chat_history ?? []}
              availableTags={availableTags}
              onChange={(tags) =>
                setToolPermissionTags((current) => ({
                  ...current,
                  chat_history: tags
                }))
              }
            />
          </section>

          <section className="settings-subsection">
            <div>
              <p className="eyebrow">Vashti</p>
              <h2>Memories</h2>
              <p className="status-message">
                Controls who may use Memories on this server. People choose the allowed memory
                actions and model scope in Personal → Memories.
              </p>
            </div>
            <div className="tool-setting-heading">
              <BrainCircuit aria-hidden="true" />
              <div>
                <strong>Memory tools</strong>
                <p>Search, read, create, update, and forget permitted memories.</p>
              </div>
            </div>
            <PermissionTagEditor
              label="Tags"
              tags={toolPermissionTags.memories ?? []}
              availableTags={availableTags}
              onChange={(tags) =>
                setToolPermissionTags((current) => ({
                  ...current,
                  memories: tags
                }))
              }
            />
          </section>

          <section className="settings-subsection">
            <div>
              <p className="eyebrow">Knowledge</p>
              <h2>Meaning-Based Search</h2>
              <p className="status-message">
                One local Ollama embedding model indexes Notes, Memories, and completed server
                chats. A dedicated backend can keep this work away from chat inference.
              </p>
            </div>
            <details className="tool-details">
              <summary>Embedding model and index status</summary>
              <ToggleSwitch
                icon={<Search />}
                label="Meaning-based knowledge search"
                description="Find notes, memories, and past chats that use different words but have a similar meaning. Keyword search remains available alongside it."
                checked={notesSemanticSearchEnabled}
                isChanged={notesSemanticChanged}
                onChange={setNotesSemanticSearchEnabled}
              />
              <label className="setting-field">
                <span>Embedding backend</span>
                <select
                  value={notesEmbeddingBackendId}
                  onChange={(event) => {
                    const backendId = event.target.value;
                    const backend = embeddingBackends.find(
                      (candidate) => candidate.backend.id === backendId
                    );
                    setNotesEmbeddingBackendId(backendId);
                    if (!backend?.models.some((model) => model.name === notesEmbeddingModel)) {
                      setNotesEmbeddingModel(backend?.models[0]?.name ?? "");
                    }
                  }}
                >
                  <option value="">Select backend</option>
                  {embeddingBackends.map((backend) => (
                    <option key={backend.backend.id} value={backend.backend.id}>
                      {backend.backend.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="setting-field">
                <span>Embedding model</span>
                <select
                  value={notesEmbeddingModel}
                  disabled={!notesEmbeddingBackendId}
                  onChange={(event) => setNotesEmbeddingModel(event.target.value)}
                >
                  <option value="">Select model</option>
                  {(embeddingBackends.find(
                    (backend) => backend.backend.id === notesEmbeddingBackendId
                  )?.models ?? []).map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </label>
              <p className="status-message">
                Notes: {settings.notes_indexed_chunks} indexed chunks
                {settings.notes_pending_index_count > 0
                  ? ` · ${settings.notes_pending_index_count} waiting`
                  : " · current"}
                <br />
                Memories: {settings.memories_indexed_count} indexed
                {settings.memories_pending_index_count > 0
                  ? ` · ${settings.memories_pending_index_count} waiting`
                  : " · current"}
                <br />
                Past chats: {settings.conversations_indexed_count} indexed
                {settings.conversations_pending_index_count > 0
                  ? ` · ${settings.conversations_pending_index_count} waiting`
                  : " · current"}
              </p>
              {settings.notes_embedding_last_error && (
                <p className="error">Last Notes indexing error: {settings.notes_embedding_last_error}</p>
              )}
              {settings.memories_embedding_last_error && (
                <p className="error">
                  Last Memories indexing error: {settings.memories_embedding_last_error}
                </p>
              )}
              {settings.conversations_embedding_last_error && (
                <p className="error">
                  Last past-chat indexing error: {settings.conversations_embedding_last_error}
                </p>
              )}
            </details>
          </section>

          <section className="settings-subsection">
            <div>
              <p className="eyebrow">Ollama</p>
              <h2>Web Search and Fetch</h2>
              <p className="status-message">
                Uses Ollama's hosted web search and web fetch APIs. Requires an Ollama API key.
              </p>
            </div>
            <ToggleSwitch
              icon={<Search />}
              label="Ollama web search"
              description="Search the web through Ollama's hosted search API."
              checked={ollamaSearchEnabled}
              isChanged={ollamaSearchChanged}
              onChange={setOllamaSearchEnabled}
            />
            <PermissionTagEditor
              label="Tags"
              tags={toolPermissionTags.ollama_web_search ?? []}
              availableTags={availableTags}
              onChange={(tags) =>
                setToolPermissionTags((current) => ({
                  ...current,
                  ollama_web_search: tags
                }))
              }
            />
            <ToggleSwitch
              icon={<FileText />}
              label="Ollama web fetch"
              description="Fetch public pages through Ollama's hosted fetch API."
              checked={ollamaFetchEnabled}
              isChanged={ollamaFetchChanged}
              onChange={setOllamaFetchEnabled}
            />
            <PermissionTagEditor
              label="Tags"
              tags={toolPermissionTags.ollama_web_fetch ?? []}
              availableTags={availableTags}
              onChange={(tags) =>
                setToolPermissionTags((current) => ({
                  ...current,
                  ollama_web_fetch: tags
                }))
              }
            />
            <details className="tool-details">
              <summary>API key and tool prompts</summary>
              <label
                className={
                  ollamaKeyChanged ? "setting-field setting-field-changed" : "setting-field"
                }
              >
                <span>Ollama API key</span>
                <input
                  type="password"
                  autoComplete="off"
                  value={ollamaApiKey}
                  onChange={(event) => setOllamaApiKey(event.target.value)}
                  placeholder={
                    settings.ollama_api_key_configured
                      ? "Configured; enter a new key to replace"
                      : "Not configured"
                  }
                />
              </label>
              {settings.ollama_api_key_configured && (
                <button
                  type="button"
                  className="danger-button key-clear-button"
                  onClick={() => setClearKeyTarget("ollama")}
                >
                  Clear Ollama API Key
                </button>
              )}
              <ToolPromptEditor
                label="web_search tool prompt"
                value={webSearchToolPrompt}
                defaultValue={settings.default_web_search_tool_prompt}
                isChanged={webSearchToolPromptChanged}
                onChange={setWebSearchToolPrompt}
              />
              <ToolPromptEditor
                label="web_fetch tool prompt"
                value={webFetchToolPrompt}
                defaultValue={settings.default_web_fetch_tool_prompt}
                isChanged={webFetchToolPromptChanged}
                onChange={setWebFetchToolPrompt}
              />
            </details>
          </section>

          <section className="settings-subsection">
            <div>
              <p className="eyebrow">Brave</p>
              <h2>Search API</h2>
              <p className="status-message">
                Uses Brave Search for result lists. Page fetching is handled separately.
              </p>
            </div>
            <ToggleSwitch
              icon={<Search />}
              label="Brave web search"
              description="Search with Brave Search and return compact result lists."
              checked={braveSearchEnabled}
              isChanged={braveSearchChanged}
              onChange={setBraveSearchEnabled}
            />
            <PermissionTagEditor
              label="Tags"
              tags={toolPermissionTags.brave_web_search ?? []}
              availableTags={availableTags}
              onChange={(tags) =>
                setToolPermissionTags((current) => ({
                  ...current,
                  brave_web_search: tags
                }))
              }
            />
            <details className="tool-details">
              <summary>API key and tool prompt</summary>
              <label
                className={braveKeyChanged ? "setting-field setting-field-changed" : "setting-field"}
              >
                <span>Brave Search API key</span>
                <input
                  type="password"
                  autoComplete="off"
                  value={braveApiKey}
                  onChange={(event) => setBraveApiKey(event.target.value)}
                  placeholder={
                    settings.brave_search_api_key_configured
                      ? "Configured; enter a new key to replace"
                      : "Not configured"
                  }
                />
              </label>
              {settings.brave_search_api_key_configured && (
                <button
                  type="button"
                  className="danger-button key-clear-button"
                  onClick={() => setClearKeyTarget("brave")}
                >
                  Clear Brave Search API Key
                </button>
              )}
              <ToolPromptEditor
                label="web_search tool prompt"
                value={webSearchToolPrompt}
                defaultValue={settings.default_web_search_tool_prompt}
                isChanged={webSearchToolPromptChanged}
                onChange={setWebSearchToolPrompt}
              />
            </details>
          </section>

          <section className="settings-subsection">
            <div>
              <p className="eyebrow">Fetch</p>
              <h2>Direct Page Fetch</h2>
              <p className="status-message">
                Lets Vashti fetch public HTTP/HTTPS pages directly. Private and local network
                addresses are blocked.
              </p>
            </div>
            <ToggleSwitch
              icon={<FileText />}
              label="Direct page fetch"
              description="Fetch public HTTP/HTTPS pages from the Vashti server."
              checked={directFetchEnabled}
              isChanged={directFetchChanged}
              onChange={setDirectFetchEnabled}
            />
            <PermissionTagEditor
              label="Tags"
              tags={toolPermissionTags.direct_web_fetch ?? []}
              availableTags={availableTags}
              onChange={(tags) =>
                setToolPermissionTags((current) => ({
                  ...current,
                  direct_web_fetch: tags
                }))
              }
            />
            <details className="tool-details">
              <summary>Tool prompt</summary>
              <ToolPromptEditor
                label="web_fetch tool prompt"
                value={webFetchToolPrompt}
                defaultValue={settings.default_web_fetch_tool_prompt}
                isChanged={webFetchToolPromptChanged}
                onChange={setWebFetchToolPrompt}
              />
            </details>
          </section>
        </form>
      )}

      {clearKeyTarget && (
        <ConfirmDialog
          title="Clear API Key"
          message={
            clearKeyTarget === "ollama"
              ? "Clear the stored Ollama API key? Ollama web search and fetch will stop working until a new key is saved."
              : "Clear the stored Brave Search API key? Brave web search will stop working until a new key is saved."
          }
          confirmLabel="Clear Key"
          isBusy={isClearingKey}
          onCancel={() => setClearKeyTarget(null)}
          onConfirm={() => void clearToolKey(clearKeyTarget)}
        />
      )}
    </SettingsPanel>
  );
}

function toolPermissionSetsEqual(
  draft: Record<string, PermissionTag[]>,
  saved: ToolSettings["tool_permissions"]
) {
  const savedByTool = new Map(saved.map((tool) => [tool.tool_id, tool.permission_tags]));
  const toolIds = new Set([...Object.keys(draft), ...savedByTool.keys()]);

  for (const toolId of toolIds) {
    if (!permissionTagSetsEqual(draft[toolId] ?? [], savedByTool.get(toolId) ?? [])) {
      return false;
    }
  }

  return true;
}
