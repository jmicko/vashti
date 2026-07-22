import {
  createContext,
  type FormEvent,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState
} from "react";
import { Check, Pencil, Plus, Server, Trash2, X } from "lucide-react";
import { BrandMark, RetroLoader } from "./common";
import { invokeNative, isNativeRuntime, setNativeAssetNamespace } from "./runtime";

export type NativeConnection = {
  id: string;
  name: string;
  base_url: string;
  instance_id: string;
  api_version: number;
  allow_insecure_http: boolean;
};

type NativeConnectionSnapshot = {
  active_connection_id: string | null;
  connections: NativeConnection[];
};

type ConnectionInput = {
  name: string;
  base_url: string;
  allow_insecure_http: boolean;
};

type NativeConnectionContextValue = {
  isNative: boolean;
  activeConnection: NativeConnection | null;
  connections: NativeConnection[];
  addConnection(input: ConnectionInput): Promise<void>;
  updateConnection(id: string, input: ConnectionInput): Promise<void>;
  removeConnection(id: string): Promise<void>;
  selectConnection(id: string): Promise<void>;
  syncActiveIdentity(instanceId: string, apiVersion: number): Promise<boolean>;
};

const browserValue: NativeConnectionContextValue = {
  isNative: false,
  activeConnection: null,
  connections: [],
  async addConnection() {},
  async updateConnection() {},
  async removeConnection() {},
  async selectConnection() {},
  async syncActiveIdentity() {
    return false;
  }
};

const NativeConnectionContext = createContext<NativeConnectionContextValue>(browserValue);

export function NativeConnectionProvider({ children }: { children: ReactNode }) {
  const native = isNativeRuntime();
  const [snapshot, setSnapshot] = useState<NativeConnectionSnapshot | null>(
    native ? null : { active_connection_id: null, connections: [] }
  );
  const [error, setError] = useState<string | null>(null);

  const loadConnections = useCallback(async () => {
    if (!native) {
      return;
    }
    setError(null);
    try {
      setSnapshot(await invokeNative<NativeConnectionSnapshot>("native_list_connections"));
    } catch (loadError) {
      setError(errorMessage(loadError, "Could not load saved servers"));
    }
  }, [native]);

  useEffect(() => {
    void loadConnections();
  }, [loadConnections]);

  const activeConnection = useMemo(
    () =>
      snapshot?.connections.find(
        (connection) => connection.id === snapshot.active_connection_id
      ) ?? null,
    [snapshot]
  );

  useEffect(() => {
    setNativeAssetNamespace(
      activeConnection ? `${activeConnection.instance_id}-${activeConnection.id}` : null
    );
  }, [activeConnection]);

  const applySnapshot = useCallback((nextSnapshot: NativeConnectionSnapshot) => {
    setSnapshot(nextSnapshot);
    setError(null);
  }, []);

  const contextValue = useMemo<NativeConnectionContextValue>(
    () => ({
      isNative: native,
      activeConnection,
      connections: snapshot?.connections ?? [],
      async addConnection(input) {
        applySnapshot(
          await invokeNative<NativeConnectionSnapshot>("native_add_connection", { input })
        );
      },
      async updateConnection(id, input) {
        const next = await invokeNative<NativeConnectionSnapshot>(
          "native_update_connection",
          { id, input }
        );
        if (id === activeConnection?.id) {
          window.location.reload();
          return;
        }
        applySnapshot(next);
      },
      async removeConnection(id) {
        const next = await invokeNative<NativeConnectionSnapshot>(
          "native_remove_connection",
          { id }
        );
        applySnapshot(next);
        window.location.reload();
      },
      async selectConnection(id) {
        if (id === activeConnection?.id) {
          return;
        }
        await invokeNative<NativeConnectionSnapshot>("native_select_connection", { id });
        window.location.reload();
      },
      async syncActiveIdentity(instanceId, apiVersion) {
        return invokeNative<boolean>("native_sync_active_identity", {
          instanceId,
          apiVersion
        });
      }
    }),
    [activeConnection, applySnapshot, native, snapshot?.connections]
  );

  if (!native) {
    return (
      <NativeConnectionContext.Provider value={browserValue}>
        {children}
      </NativeConnectionContext.Provider>
    );
  }

  if (!snapshot && !error) {
    return <NativeConnectionLoading />;
  }

  if (!snapshot || !activeConnection) {
    return (
      <NativeConnectionContext.Provider value={contextValue}>
        <NativeConnectionSetup error={error} onRetry={loadConnections} />
      </NativeConnectionContext.Provider>
    );
  }

  return (
    <NativeConnectionContext.Provider value={contextValue}>
      {children}
    </NativeConnectionContext.Provider>
  );
}

export function useNativeConnections() {
  return useContext(NativeConnectionContext);
}

export function NativeConnectionsSettings() {
  const {
    isNative,
    activeConnection,
    connections,
    addConnection,
    updateConnection,
    removeConnection,
    selectConnection
  } = useNativeConnections();
  const [isAdding, setIsAdding] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isNative) {
    return null;
  }

  async function run(action: () => Promise<void>) {
    setIsBusy(true);
    setError(null);
    try {
      await action();
      setIsAdding(false);
      setEditingId(null);
    } catch (actionError) {
      setError(errorMessage(actionError, "Could not update the server list"));
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <section className="settings-subsection native-connections-section">
      <div className="settings-subsection-heading-row">
        <div>
          <p className="eyebrow">Android App</p>
          <h2>Servers</h2>
        </div>
        <button
          type="button"
          className="secondary-button compact-button"
          disabled={isBusy || isAdding}
          onClick={() => {
            setEditingId(null);
            setIsAdding(true);
          }}
        >
          <Plus aria-hidden="true" />
          <span>Add</span>
        </button>
      </div>
      <p className="status-message">
        The Android app can connect to more than one Vashti server. Switching servers reloads
        the app and keeps each server's local data separate.
      </p>
      {error && <p className="error">{error}</p>}
      <div className="native-connection-list">
        {connections.map((connection) =>
          editingId === connection.id ? (
            <ConnectionForm
              key={connection.id}
              initial={connection}
              submitLabel="Save"
              disabled={isBusy}
              onCancel={() => setEditingId(null)}
              onSubmit={(input) => run(() => updateConnection(connection.id, input))}
            />
          ) : (
            <article
              key={connection.id}
              className={
                connection.id === activeConnection?.id
                  ? "native-connection-row native-connection-row-active"
                  : "native-connection-row"
              }
            >
              <button
                type="button"
                className="native-connection-select"
                disabled={isBusy || connection.id === activeConnection?.id}
                onClick={() => void run(() => selectConnection(connection.id))}
              >
                <Server aria-hidden="true" />
                <span>
                  <strong>{connection.name}</strong>
                  <small>{connection.base_url}</small>
                </span>
                {connection.id === activeConnection?.id && (
                  <span className="capability-chip">Current</span>
                )}
              </button>
              <div className="native-connection-actions">
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`Edit ${connection.name}`}
                  title="Edit server"
                  disabled={isBusy}
                  onClick={() => {
                    setIsAdding(false);
                    setEditingId(connection.id);
                  }}
                >
                  <Pencil aria-hidden="true" />
                </button>
                <button
                  type="button"
                  className="icon-button danger-button"
                  aria-label={`Remove ${connection.name}`}
                  title="Remove server"
                  disabled={isBusy}
                  onClick={() => {
                    if (window.confirm(`Remove ${connection.name} from this device?`)) {
                      void run(() => removeConnection(connection.id));
                    }
                  }}
                >
                  <Trash2 aria-hidden="true" />
                </button>
              </div>
            </article>
          )
        )}
      </div>
      {isAdding && (
        <ConnectionForm
          submitLabel="Add Server"
          disabled={isBusy}
          onCancel={() => setIsAdding(false)}
          onSubmit={(input) => run(() => addConnection(input))}
        />
      )}
    </section>
  );
}

function NativeConnectionLoading() {
  return (
    <main className="auth-page native-connection-page">
      <section className="auth-panel native-connection-panel">
        <BrandMark />
        <h1>Loading Servers</h1>
        <RetroLoader />
      </section>
    </main>
  );
}

function NativeConnectionSetup({
  error,
  onRetry
}: {
  error: string | null;
  onRetry: () => Promise<void>;
}) {
  const { addConnection } = useNativeConnections();
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  async function submit(input: ConnectionInput) {
    setIsSaving(true);
    setSaveError(null);
    try {
      await addConnection(input);
    } catch (submitError) {
      setSaveError(errorMessage(submitError, "Could not connect to that Vashti server"));
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <main className="auth-page native-connection-page">
      <section className="auth-panel native-connection-panel">
        <BrandMark />
        <div>
          <p className="eyebrow">Android App</p>
          <h1>Connect to Vashti</h1>
          <p>Enter the address you use to open your Vashti server.</p>
        </div>
        {(error || saveError) && <p className="error">{saveError ?? error}</p>}
        <ConnectionForm submitLabel="Connect" disabled={isSaving} onSubmit={submit} />
        {error && (
          <button type="button" className="secondary-button" onClick={() => void onRetry()}>
            Retry Loading
          </button>
        )}
      </section>
    </main>
  );
}

function ConnectionForm({
  initial,
  submitLabel,
  disabled,
  onCancel,
  onSubmit
}: {
  initial?: NativeConnection;
  submitLabel: string;
  disabled: boolean;
  onCancel?: () => void;
  onSubmit: (input: ConnectionInput) => Promise<void> | void;
}) {
  const [name, setName] = useState(initial?.name ?? "My Vashti");
  const [baseUrl, setBaseUrl] = useState(initial?.base_url ?? "");
  const [allowHttp, setAllowHttp] = useState(initial?.allow_insecure_http ?? false);
  const usesHttp = baseUrl.trim().toLowerCase().startsWith("http://");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void onSubmit({
      name: name.trim(),
      base_url: baseUrl.trim(),
      allow_insecure_http: allowHttp
    });
  }

  return (
    <form className="native-connection-form" onSubmit={handleSubmit}>
      <label>
        <span>Name</span>
        <input
          required
          value={name}
          maxLength={80}
          placeholder="Home server"
          disabled={disabled}
          onChange={(event) => setName(event.target.value)}
        />
      </label>
      <label>
        <span>Server URL</span>
        <input
          required
          type="url"
          inputMode="url"
          autoCapitalize="none"
          autoCorrect="off"
          value={baseUrl}
          placeholder="https://chat.example.com"
          disabled={disabled}
          onChange={(event) => {
            setBaseUrl(event.target.value);
            if (!event.target.value.trim().toLowerCase().startsWith("http://")) {
              setAllowHttp(false);
            }
          }}
        />
      </label>
      {usesHttp && (
        <label className="native-http-warning">
          <input
            type="checkbox"
            required
            checked={allowHttp}
            disabled={disabled}
            onChange={(event) => setAllowHttp(event.target.checked)}
          />
          <span>
            Allow unencrypted HTTP. Use this only for a trusted local network; passwords and
            chats can be visible to other devices on that network.
          </span>
        </label>
      )}
      <div className="native-connection-form-actions">
        {onCancel && (
          <button type="button" className="secondary-button" disabled={disabled} onClick={onCancel}>
            <X aria-hidden="true" />
            <span>Cancel</span>
          </button>
        )}
        <button type="submit" disabled={disabled || (usesHttp && !allowHttp)}>
          <Check aria-hidden="true" />
          <span>{disabled ? "Connecting..." : submitLabel}</span>
        </button>
      </div>
    </form>
  );
}

function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : typeof error === "string" ? error : fallback;
}
