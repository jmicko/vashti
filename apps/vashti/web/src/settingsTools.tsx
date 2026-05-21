import {
  FormEvent,
  useCallback,
  useEffect,
  useState
} from "react";
import { FileText, Save, Search, Wrench } from "lucide-react";
import { requestJson } from "./api";
import { ConfirmDialog } from "./common";
import { PermissionTagEditor } from "./permissionTags";
import {
  SettingsPanel,
  SettingsSaveBanner,
  ToggleSwitch,
  ToolPromptEditor
} from "./settingsControls";
import { permissionTagPayload } from "./settingsModelHelpers";
import type { PermissionTag, ToolSettings } from "./types";

export function ToolsSettingsPanel({ onToolsChanged }: { onToolsChanged: () => Promise<void> }) {
  const [settings, setSettings] = useState<ToolSettings | null>(null);
  const [toolsEnabled, setToolsEnabled] = useState(false);
  const [ollamaSearchEnabled, setOllamaSearchEnabled] = useState(false);
  const [ollamaFetchEnabled, setOllamaFetchEnabled] = useState(false);
  const [ollamaApiKey, setOllamaApiKey] = useState("");
  const [braveSearchEnabled, setBraveSearchEnabled] = useState(false);
  const [braveApiKey, setBraveApiKey] = useState("");
  const [directFetchEnabled, setDirectFetchEnabled] = useState(false);
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
        ollamaApiKey.trim() ||
        braveApiKey.trim() ||
        toolSystemPrompt !== settings.tool_system_prompt ||
        webSearchToolPrompt !== settings.web_search_tool_prompt ||
        webFetchToolPrompt !== settings.web_fetch_tool_prompt ||
        JSON.stringify(permissionTagPayload(defaultToolTags)) !==
          JSON.stringify(permissionTagPayload(settings.default_tool_permission_tags)) ||
        JSON.stringify(
          Object.fromEntries(
            Object.entries(toolPermissionTags).map(([toolId, tags]) => [
              toolId,
              permissionTagPayload(tags)
            ])
          )
        ) !==
          JSON.stringify(
            Object.fromEntries(
              settings.tool_permissions.map((tool) => [
                tool.tool_id,
                permissionTagPayload(tool.permission_tags)
              ])
            )
          ))
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
      const response = await requestJson<ToolSettings>("/api/settings/tools");
      applyToolSettings(response);
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
