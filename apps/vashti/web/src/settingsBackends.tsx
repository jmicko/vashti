import {
  FormEvent,
  useCallback,
  useEffect,
  useState
} from "react";
import { Plus, RefreshCw, Search, Server } from "lucide-react";
import { requestJson } from "./api";
import { isLocalBackend } from "./modelSelection";
import { ConfirmDialog, RetroLoader } from "./common";
import { AdminModelsAccessPanel } from "./settingsModels";
import type {
  Backend,
  BackendsResponse,
  DetectLocalhostResponse
} from "./types";

export function BackendsPanel({ onBackendsChanged }: { onBackendsChanged: () => Promise<void> }) {
  const [backends, setBackends] = useState<Backend[]>([]);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isRefreshingModels, setIsRefreshingModels] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [busyBackendId, setBusyBackendId] = useState<string | null>(null);
  const [editingBackend, setEditingBackend] = useState<Backend | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Backend | null>(null);
  const [newName, setNewName] = useState("");
  const [newBaseUrl, setNewBaseUrl] = useState("");
  const [modelRefreshRequest, setModelRefreshRequest] = useState(0);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadBackends = useCallback(async () => {
    setIsRefreshing(true);
    setError(null);

    try {
      const response = await requestJson<BackendsResponse>("/api/backends");
      setBackends(response.backends);
      setHasLoaded(true);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load backends");
      setBackends([]);
      setEditingBackend(null);
      setDeleteTarget(null);
      setHasLoaded(true);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadBackends();
  }, [loadBackends]);

  async function createBackend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsCreating(true);
    setError(null);
    setStatus(null);

    try {
      await requestJson("/api/backends", {
        method: "POST",
        body: JSON.stringify({ name: newName, base_url: newBaseUrl })
      });
      setNewName("");
      setNewBaseUrl("");
      setStatus("Backend added.");
      await loadBackends();
      await onBackendsChanged();
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "Failed to add backend");
    } finally {
      setIsCreating(false);
    }
  }

  async function updateBackend(
    backendId: string,
    body: Partial<Pick<Backend, "name" | "base_url" | "is_enabled">>
  ) {
    setBusyBackendId(backendId);
    setError(null);
    setStatus(null);

    try {
      await requestJson(`/api/backends/${backendId}`, {
        method: "PATCH",
        body: JSON.stringify(body)
      });
      setEditingBackend(null);
      setStatus("Backend updated.");
      await loadBackends();
      await onBackendsChanged();
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : "Failed to update backend");
    } finally {
      setBusyBackendId(null);
    }
  }

  async function deleteBackend(backendId: string) {
    setBusyBackendId(backendId);
    setError(null);
    setStatus(null);

    try {
      await requestJson(`/api/backends/${backendId}`, { method: "DELETE" });
      setDeleteTarget(null);
      setStatus("Backend deleted.");
      await loadBackends();
      await onBackendsChanged();
    } catch (deleteError) {
      setDeleteTarget(null);
      setError(deleteError instanceof Error ? deleteError.message : "Failed to delete backend");
    } finally {
      setBusyBackendId(null);
    }
  }

  async function detectLocalhost() {
    setIsDetecting(true);
    setError(null);
    setStatus(null);

    try {
      const response = await requestJson<DetectLocalhostResponse>("/api/backends/detect-localhost", {
        method: "POST"
      });
      setStatus(
        response.detected.length === 0
          ? "No local Ollama backend found."
          : `Detected ${response.detected.map((backend) => backend.name).join(", ")}.`
      );
      await loadBackends();
      await onBackendsChanged();
    } catch (detectError) {
      setError(detectError instanceof Error ? detectError.message : "Localhost detection failed");
    } finally {
      setIsDetecting(false);
    }
  }

  async function scanLocalNetwork() {
    setIsScanning(true);
    setError(null);
    setStatus(null);

    try {
      const response = await requestJson<DetectLocalhostResponse>(
        "/api/backends/scan-local-network",
        {
          method: "POST"
        }
      );
      setStatus(
        response.detected.length === 0
          ? "No Ollama backends found on the local network."
          : `Detected ${response.detected.map((backend) => backend.name).join(", ")}.`
      );
      await loadBackends();
      await onBackendsChanged();
    } catch (scanError) {
      setError(scanError instanceof Error ? scanError.message : "Local network scan failed");
    } finally {
      setIsScanning(false);
    }
  }

  const hasLocalBackend = backends.some((backend) => isLocalBackend(backend.base_url));

  return (
    <div className="settings-section">
      <div className="section-header">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>Backends</h1>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="secondary-button refresh-button"
            onClick={() => void loadBackends()}
            disabled={isRefreshing}
          >
            {isRefreshing ? <RetroLoader /> : <RefreshCw />}
            <span>{isRefreshing ? "Loading" : "Refresh Backends"}</span>
          </button>
          <button
            type="button"
            className="secondary-button refresh-button"
            onClick={() => setModelRefreshRequest((current) => current + 1)}
            disabled={isRefreshingModels}
          >
            {isRefreshingModels ? <RetroLoader /> : <RefreshCw />}
            <span>{isRefreshingModels ? "Checking" : "Refresh Models"}</span>
          </button>
          {!hasLocalBackend && (
            <button
              type="button"
              className="secondary-button"
              onClick={() => void detectLocalhost()}
              disabled={isDetecting}
            >
              {isDetecting ? <RetroLoader /> : <Search />}
              <span>{isDetecting ? "Detecting" : "Detect Localhost"}</span>
            </button>
          )}
          <button
            type="button"
            className="secondary-button"
            onClick={() => void scanLocalNetwork()}
            disabled={isScanning}
          >
            {isScanning ? <RetroLoader /> : <Server />}
            <span>{isScanning ? "Scanning" : "Scan Network"}</span>
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}
      {status && <p className="status-message">{status}</p>}

      <form className="settings-form backend-create-form" onSubmit={createBackend}>
        <label>
          <span>Name</span>
          <input
            required
            value={newName}
            onChange={(event) => setNewName(event.target.value)}
            placeholder="hostname"
          />
        </label>
        <label>
          <span>Base URL</span>
          <input
            required
            value={newBaseUrl}
            onChange={(event) => setNewBaseUrl(event.target.value)}
            placeholder="http://127.0.0.1:11434"
          />
        </label>
        <button type="submit" disabled={isCreating}>
          <Plus />
          <span>{isCreating ? "Adding..." : "Add Backend"}</span>
        </button>
      </form>

      {!hasLoaded && <p className="status-message">Loading backends...</p>}
      {hasLoaded && backends.length === 0 && (
        <p className="status-message">No Ollama backends configured.</p>
      )}

      <AdminModelsAccessPanel
        backends={backends}
        editingBackend={editingBackend}
        backendBusyId={busyBackendId}
        onCancelBackendEdit={() => setEditingBackend(null)}
        onDeleteBackend={setDeleteTarget}
        onEditBackend={setEditingBackend}
        onSaveBackend={updateBackend}
        onToggleBackend={(backendId, nextEnabled) =>
          void updateBackend(backendId, { is_enabled: nextEnabled })
        }
        modelRefreshRequest={modelRefreshRequest}
        onModelRefreshStateChange={setIsRefreshingModels}
        onModelsChanged={onBackendsChanged}
      />

      {deleteTarget && (
        <ConfirmDialog
          title="Delete Backend"
          message={`Delete ${deleteTarget.name}? This cannot be undone.`}
          confirmLabel="Delete"
          isBusy={busyBackendId === deleteTarget.id}
          onCancel={() => setDeleteTarget(null)}
          onConfirm={() => void deleteBackend(deleteTarget.id)}
        />
      )}
    </div>
  );
}
