import { useCallback, useEffect, useRef, useState } from "react";
import {
  BrainCircuit,
  Check,
  Clock3,
  HardDrive,
  LoaderCircle,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Server,
  Trash2,
  X
} from "lucide-react";
import { ConfirmDialog } from "../common";
import { ToggleSwitch } from "../settingsControls";
import type { SettingsGuard } from "../types";
import {
  forgetProfile,
  getProfile,
  processProfile,
  profileHistory,
  profileModels,
  saveProfileFact,
  saveProfileSettings
} from "./api";
import { allowsProfile, factSources } from "./core";
import { models as deviceModels } from "./device";
import type {
  PersonalProfile,
  ProfileFact,
  ProfileMode,
  ProfileModel,
  ProfileSettings,
  ProfileVersion
} from "./types";

export function PersonalizationPanel({
  onGuardChange,
  onOpenSource
}: {
  onGuardChange: (guard: SettingsGuard | null) => void;
  onOpenSource?: (chatId: string, mode: ProfileMode) => void;
}) {
  const [mode, setMode] = useState<ProfileMode>("server");
  const [profile, setProfile] = useState<PersonalProfile | null>(null);
  const [draft, setDraft] = useState<ProfileSettings | null>(null);
  const [models, setModels] = useState<ProfileModel[]>([]);
  const [busy, setBusy] = useState(false);
  const [processingNow, setProcessingNow] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [editing, setEditing] = useState<{
    id: string | null;
    topic: string;
    content: string;
  } | null>(null);
  const [history, setHistory] = useState<{
    fact: ProfileFact;
    versions: ProfileVersion[];
  } | null>(null);
  const [confirm, setConfirm] = useState<{
    id: string | null;
    title: string;
  } | null>(null);
  const [search, setSearch] = useState("");
  const [modelSearch, setModelSearch] = useState("");
  const request = useRef(0);
  const dirty =
    !!profile &&
    !!draft &&
    (JSON.stringify(profile.settings) !== JSON.stringify(draft) || editing !== null);

  const adopt = useCallback((value: PersonalProfile) => {
    setProfile(value);
    setDraft(structuredClone(value.settings));
    setEditing(null);
    setHistory(null);
  }, []);

  const load = useCallback(async () => {
    const id = ++request.current;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const [value, catalog] = await Promise.all([
        getProfile(mode),
        mode === "device"
          ? deviceModels().catch(() => {
              if (id === request.current)
                setNotice("Model list unavailable. Saved device facts are still accessible.");
              return [];
            })
          : profileModels()
      ]);
      if (id !== request.current) return;
      adopt(value);
      setModels(catalog);
    } catch (e) {
      if (id === request.current) setError(message(e));
    } finally {
      if (id === request.current) setBusy(false);
    }
  }, [adopt, mode]);

  function selectMode(next: ProfileMode) {
    if (next === mode) return;
    request.current++;
    setProfile(null);
    setDraft(null);
    setModels([]);
    setEditing(null);
    setHistory(null);
    setConfirm(null);
    setError(null);
    setNotice(null);
    setSearch("");
    setBusy(true);
    setMode(next);
  }

  useEffect(() => {
    void load();
    return () => {
      request.current++;
    };
  }, [load]);

  const save = useCallback(async (): Promise<boolean> => {
    if (!profile || !draft) return false;
    const id = request.current;
    setBusy(true);
    setError(null);
    try {
      let value = profile;
      if (JSON.stringify(profile.settings) !== JSON.stringify(draft)) {
        value = await saveProfileSettings(mode, value, draft);
        if (id === request.current) setProfile(value);
      }
      if (editing)
        value = await saveProfileFact(mode, value, editing.id, editing.topic, editing.content);
      if (id !== request.current) return false;
      adopt(value);
      setNotice("Profile saved.");
      return true;
    } catch (e) {
      if (id === request.current) setError(message(e));
      return false;
    } finally {
      if (id === request.current) setBusy(false);
    }
  }, [adopt, draft, editing, mode, profile]);

  const discard = useCallback(() => {
    if (profile) {
      setDraft(structuredClone(profile.settings));
      setEditing(null);
    }
  }, [profile]);
  useEffect(() => {
    onGuardChange({ isDirty: dirty, save, discard });
  }, [dirty, discard, onGuardChange, save]);
  useEffect(() => () => onGuardChange(null), [onGuardChange]);
  useEffect(() => {
    if (dirty) return;
    const listener = () => {
      void load();
    };
    window.addEventListener("vashti:personalization-changed", listener);
    return () => window.removeEventListener("vashti:personalization-changed", listener);
  }, [dirty, load]);
  useEffect(() => {
    if (!profile?.settings.enabled || dirty || busy || mode !== "server") return;
    let stopped = false;
    const timer = window.setInterval(() => {
      void getProfile(mode)
        .then((value) => {
          if (
            !stopped &&
            (value.revision !== profile.revision ||
              value.requested !== profile.requested ||
              value.last_attempt_at !== profile.last_attempt_at ||
              value.last_error !== profile.last_error)
          )
            adopt(value);
        })
        .catch(() => undefined);
    }, 3000);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [adopt, busy, dirty, mode, profile]);

  function update(patch: Partial<ProfileSettings>) {
    setDraft((current) => (current ? { ...current, ...patch } : current));
    setNotice(null);
  }

  async function process() {
    if (!profile) return;
    setBusy(true);
    setProcessingNow(true);
    setError(null);
    try {
      const value = await processProfile(mode, profile);
      adopt(value);
      setNotice(
        mode === "server" || value.requested
          ? "Queued. Processing waits for active generations to finish."
          : "Device profile is up to date."
      );
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
      setProcessingNow(false);
    }
  }

  async function forget() {
    if (!profile || !confirm) return;
    setBusy(true);
    setError(null);
    try {
      adopt(await forgetProfile(mode, profile, confirm.id));
      setConfirm(null);
      setNotice("Forgotten. Previous sources will not recreate this information.");
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }

  async function showHistory(fact: ProfileFact) {
    setBusy(true);
    setError(null);
    try {
      setHistory({ fact, versions: await profileHistory(mode, fact.id) });
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }

  const processors = models.filter((m) => m.key.startsWith("base:"));
  const processorKey =
    draft?.processor_backend_id && draft.processor_model
      ? `base:${draft.processor_backend_id}:${draft.processor_model}`
      : "";
  const processor = processors.find((m) => m.key === processorKey);
  const visibleFacts =
    profile?.facts.filter((f) =>
      `${f.topic} ${f.content}`.toLowerCase().includes(search.toLowerCase())
    ) ?? [];

  return (
    <section
      className="personalization-section"
      aria-label="Automatic personalization"
      aria-busy={busy}
    >
      <header className="profile-heading">
        <div>
          <h2>
            <BrainCircuit aria-hidden="true" />
            Personal Profile
          </h2>
          <small>
            {profile
              ? `${profile.facts.length} facts · ${profile.settings.enabled ? "On" : "Off"}`
              : busy
                ? "Loading profile..."
                : "Profile unavailable"}
          </small>
        </div>
        <button
          className="icon-button"
          type="button"
          title="Refresh profile"
          aria-label="Refresh profile"
          disabled={busy || dirty}
          onClick={() => void load()}
        >
          <RefreshCw aria-hidden="true" />
        </button>
      </header>
      <div className="profile-storage" role="group" aria-label="Profile storage">
        <button
          type="button"
          className={mode === "server" ? "active" : "secondary-button"}
          aria-pressed={mode === "server"}
          disabled={busy || dirty}
          onClick={() => selectMode("server")}
        >
          <Server aria-hidden="true" />
          Server
        </button>
        <button
          type="button"
          className={mode === "device" ? "active" : "secondary-button"}
          aria-pressed={mode === "device"}
          disabled={busy || dirty}
          onClick={() => selectMode("device")}
        >
          <HardDrive aria-hidden="true" />
          This device
        </button>
      </div>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {notice && (
        <p className="muted" role="status">
          {notice}
        </p>
      )}
      {draft && profile && (
        <>
          <ToggleSwitch
            label="Automatic personalization"
            checked={draft.enabled}
            disabled={busy}
            onChange={(enabled) => update({ enabled })}
          />
          <div className="profile-processor">
            <label htmlFor="profile-processor">Processing model</label>
            <select
              id="profile-processor"
              value={processorKey}
              disabled={busy}
              onChange={(e) => {
                const selected = processors.find((m) => m.key === e.target.value);
                update({
                  processor_backend_id: selected?.backend_id ?? null,
                  processor_model: selected?.model_name ?? null
                });
              }}
            >
              <option value="">Not selected</option>
              {processorKey && !processor && (
                <option value={processorKey}>Unavailable: {draft.processor_model}</option>
              )}
              {processors.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.name}
                  {m.cloud ? " (cloud)" : ""}
                </option>
              ))}
            </select>
            {(processor?.cloud || draft.allow_cloud_processor) && (
              <ToggleSwitch
                label="Allow cloud processing"
                description="Eligible conversation text will be sent to this processor."
                checked={draft.allow_cloud_processor}
                disabled={busy}
                onChange={(allow_cloud_processor) => update({ allow_cloud_processor })}
              />
            )}
            <small className="muted">
              {mode === "device"
                ? "Encrypted on this device. Processing runs while the app is open."
                : "Stored on this server. Processing can run while you are away."}
            </small>
          </div>
          <details className="profile-disclosure">
            <summary>Model access and processing</summary>
            <div className="profile-access">
              <ToggleSwitch
                label="Local models"
                checked={draft.allow_local}
                disabled={busy}
                onChange={(allow_local) => update({ allow_local })}
              />
              <ToggleSwitch
                label="Cloud models"
                checked={draft.allow_cloud}
                disabled={busy}
                onChange={(allow_cloud) => update({ allow_cloud })}
              />
              <ToggleSwitch
                label="Character models"
                checked={draft.allow_characters}
                disabled={busy}
                onChange={(allow_characters) => update({ allow_characters })}
              />
              <small className="muted">
                Excluded models neither contribute to nor receive this profile. Individual choices
                override the group switches.
              </small>
              {models.length > 8 && (
                <input
                  type="search"
                  aria-label="Filter profile model access"
                  placeholder="Filter models"
                  value={modelSearch}
                  onChange={(event) => setModelSearch(event.target.value)}
                />
              )}
              {Object.keys(draft.model_overrides).length > 0 && (
                <button
                  type="button"
                  className="secondary-button"
                  disabled={busy}
                  onClick={() => update({ model_overrides: {} })}
                >
                  <RotateCcw aria-hidden="true" />
                  Clear individual choices
                </button>
              )}
              <div className="profile-model-list">
                {models
                  .filter((m) =>
                    `${m.name} ${m.cloud ? "cloud" : "local"} ${m.character ? "character" : ""}`
                      .toLowerCase()
                      .includes(modelSearch.toLowerCase())
                  )
                  .map((m) => (
                    <label key={m.key} className="profile-model-row">
                      <span>
                        {m.name}
                        <small>
                          {m.cloud ? "Cloud" : "Local"}
                          {m.character ? " · Character" : ""}
                        </small>
                      </span>
                      <select
                        aria-label={`Profile access for ${m.name}`}
                        disabled={busy}
                        value={
                          draft.model_overrides[m.key] === undefined
                            ? "default"
                            : draft.model_overrides[m.key]
                              ? "allow"
                              : "exclude"
                        }
                        onChange={(e) => {
                          const overrides = { ...draft.model_overrides };
                          if (e.target.value === "default") delete overrides[m.key];
                          else overrides[m.key] = e.target.value === "allow";
                          update({ model_overrides: overrides });
                        }}
                      >
                        <option value="default">
                          Default:{" "}
                          {allowsProfile({ ...draft, enabled: true, model_overrides: {} }, m)
                            ? "allow"
                            : "exclude"}
                        </option>
                        <option value="allow">Allow</option>
                        <option value="exclude">Exclude</option>
                      </select>
                    </label>
                  ))}
              </div>
              <label className="profile-number">
                Idle time (minutes)
                <input
                  type="number"
                  min="1"
                  max="120"
                  value={draft.idle_minutes}
                  disabled={busy}
                  onChange={(e) => update({ idle_minutes: Number(e.target.value) })}
                />
              </label>
              <label className="profile-number">
                Context budget (characters)
                <input
                  type="number"
                  min="1000"
                  max="16000"
                  step="500"
                  value={draft.context_chars}
                  disabled={busy}
                  onChange={(e) => update({ context_chars: Number(e.target.value) })}
                />
              </label>
            </div>
          </details>
          {dirty && (
            <div className="profile-actions">
              <button type="button" disabled={busy} onClick={() => void save()}>
                <Save aria-hidden="true" />
                Save profile
              </button>
              <button type="button" className="secondary-button" disabled={busy} onClick={discard}>
                <RotateCcw aria-hidden="true" />
                Revert
              </button>
            </div>
          )}
          <div className="profile-heading profile-list-heading">
            <h3>Profile facts</h3>
            <div className="profile-actions">
              <button
                className="icon-button"
                type="button"
                title="Add profile fact"
                aria-label="Add profile fact"
                disabled={busy || dirty}
                onClick={() => setEditing({ id: null, topic: "", content: "" })}
              >
                <Plus aria-hidden="true" />
              </button>
              <button
                type="button"
                className="secondary-button"
                disabled={
                  busy ||
                  dirty ||
                  (profile.requested && !profile.last_error) ||
                  !draft.enabled ||
                  !processor ||
                  (processor.cloud && !draft.allow_cloud_processor)
                }
                onClick={() => void process()}
              >
                {processingNow ? (
                  <LoaderCircle aria-hidden="true" className="spin" />
                ) : (
                  <Play aria-hidden="true" />
                )}
                {processingNow
                  ? "Processing..."
                  : profile.last_error
                    ? "Retry processing"
                    : profile.requested
                      ? "Queued"
                      : "Process now"}
              </button>
            </div>
          </div>
          <small className="muted">
            {profile.last_processed_at
              ? `Last processed ${new Date(profile.last_processed_at * 1000).toLocaleString()}`
              : "Not processed yet"}
            {profile.requested && !profile.last_error ? " · Queued" : ""}
            {profile.last_error ? ` · ${processingError(profile.last_error)}` : ""}
          </small>
          {editing && (
            <div className="profile-fact-editor">
              <label>
                Topic
                <input
                  aria-label="Profile topic"
                  maxLength={100}
                  value={editing.topic}
                  disabled={busy}
                  placeholder="food.favorite"
                  onChange={(e) => setEditing({ ...editing, topic: e.target.value })}
                />
              </label>
              <label>
                Fact
                <textarea
                  aria-label="Profile fact"
                  rows={4}
                  maxLength={600}
                  value={editing.content}
                  disabled={busy}
                  onChange={(e) => setEditing({ ...editing, content: e.target.value })}
                />
              </label>
              <div className="profile-actions">
                <button type="button" disabled={busy} onClick={() => void save()}>
                  <Check aria-hidden="true" />
                  Save
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  disabled={busy}
                  onClick={() => setEditing(null)}
                >
                  <X aria-hidden="true" />
                  Cancel
                </button>
              </div>
            </div>
          )}
          {profile.facts.length > 5 && (
            <input
              type="search"
              aria-label="Search profile facts"
              placeholder="Search profile"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          )}
          <ul className="profile-facts">
            {visibleFacts.map((fact) => (
              <li key={fact.id}>
                <div className="profile-fact-heading">
                  <span>
                    {fact.topic.replace(/_/g, " ").replace(/\./g, " / ")}
                    <small>
                      {fact.locked ? "Saved by you" : "Automatic"} · v{fact.version}
                    </small>
                  </span>
                  <div className="profile-actions">
                    <button
                      className="icon-button"
                      type="button"
                      title="Edit fact"
                      aria-label={`Edit ${fact.topic}`}
                      disabled={busy || dirty}
                      onClick={() =>
                        setEditing({
                          id: fact.id,
                          topic: fact.topic,
                          content: fact.content
                        })
                      }
                    >
                      <Pencil aria-hidden="true" />
                    </button>
                    <button
                      className="icon-button"
                      type="button"
                      title="Fact history"
                      aria-label={`History of ${fact.topic}`}
                      disabled={busy || dirty}
                      onClick={() => void showHistory(fact)}
                    >
                      <Clock3 aria-hidden="true" />
                    </button>
                    <button
                      className="icon-button danger-button"
                      type="button"
                      title="Forget fact"
                      aria-label={`Forget ${fact.topic}`}
                      disabled={busy || dirty}
                      onClick={() => setConfirm({ id: fact.id, title: "Forget this fact?" })}
                    >
                      <Trash2 aria-hidden="true" />
                    </button>
                  </div>
                </div>
                <p>{fact.content}</p>
                {factSources(fact).length > 0 && (
                  <details>
                    <summary>Sources</summary>
                    {factSources(fact).map((source, index) => (
                      <div className="profile-source" key={`${source.revision_id}:${index}`}>
                        <blockquote>{source.quote}</blockquote>
                        <a
                          href={`/app/${mode === "device" ? "private-chats" : "chats"}/${encodeURIComponent(source.chat_id)}`}
                          onClick={(event) => {
                            if (
                              onOpenSource &&
                              !event.ctrlKey &&
                              !event.metaKey &&
                              !event.shiftKey
                            ) {
                              event.preventDefault();
                              onOpenSource(source.chat_id, mode);
                            }
                          }}
                        >
                          Open conversation
                        </a>
                      </div>
                    ))}
                  </details>
                )}
              </li>
            ))}
          </ul>
          {!profile.facts.length && <p className="muted">No facts saved.</p>}
          {history && (
            <section className="profile-history" aria-label="Profile fact history">
              <div className="profile-heading">
                <h3>{history.fact.topic} history</h3>
                <button
                  className="icon-button"
                  type="button"
                  title="Close history"
                  aria-label="Close fact history"
                  onClick={() => setHistory(null)}
                >
                  <X aria-hidden="true" />
                </button>
              </div>
              {history.versions.map((version) => (
                <article key={version.version}>
                  <small>
                    v{version.version} · {version.actor === "human" ? "You" : "Processor"} ·{" "}
                    {new Date(version.created_at * 1000).toLocaleString()}
                  </small>
                  <p>{version.content}</p>
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={busy || dirty || version.version === history.fact.version}
                    onClick={() => {
                      setEditing({
                        id: history.fact.id,
                        topic: history.fact.topic,
                        content: version.content
                      });
                      setHistory(null);
                    }}
                  >
                    <RotateCcw aria-hidden="true" />
                    Use this version
                  </button>
                </article>
              ))}
            </section>
          )}
          {!!profile.facts.length && (
            <button
              type="button"
              className="secondary-button danger-button"
              disabled={busy || dirty}
              onClick={() => setConfirm({ id: null, title: "Forget the entire profile?" })}
            >
              <Trash2 aria-hidden="true" />
              Forget all facts
            </button>
          )}
        </>
      )}
      {confirm && (
        <ConfirmDialog
          title={confirm.title}
          message="This removes the fact and its history. Existing source text will not recreate it automatically. Your conversations will stay intact."
          confirmLabel="Forget"
          isBusy={busy}
          onConfirm={() => void forget()}
          onCancel={() => setConfirm(null)}
        />
      )}
    </section>
  );
}

function message(error: unknown) {
  return error instanceof Error ? error.message : "Could not update the profile.";
}

function processingError(code: string) {
  if (code === "processor_invalid_evidence")
    return "The last batch was rejected: source quotes did not match. Retry or choose a stronger processor.";
  if (code === "processor_invalid_schema" || code === "processor_invalid_json")
    return "The processor returned an invalid fact format. Retry or choose another model.";
  if (code === "processor_incomplete")
    return "The processor did not finish its last batch. Retry or choose another model.";
  return "Processing needs attention; check the model and retry.";
}
