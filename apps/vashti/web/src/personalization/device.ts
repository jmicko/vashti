import { requestJson } from "../api";
import {
  listPrivateChats,
  listPrivateMessages,
  listPrivatePersonas,
  listPrivatePersonaVersions,
  privateImageGuard,
  privateProfileIdentity,
  readPrivateProfile,
  writePrivateProfile
} from "../privateChatStore";
import {
  allowsProfile,
  contextWeight,
  defaultProfile,
  factSources,
  fitProfileContext,
  profileContext,
  validateFact
} from "./core";
import type {
  DeviceProfile,
  InputSource,
  ProfileFact,
  ProfileModel,
  ProfileSettings,
  ProfileSource
} from "./types";

export async function getProfile(): Promise<DeviceProfile> {
  return (
    (await readPrivateProfile<DeviceProfile>()) ?? {
      ...defaultProfile(),
      processed: [],
      suppressed: [],
      history: {}
    }
  );
}

async function save(profile: DeviceProfile, expected: number) {
  profile.revision = expected + 1;
  await writePrivateProfile(profile, expected);
  return profile;
}

export async function models(): Promise<ProfileModel[]> {
  const guard = privateImageGuard();
  const base = await requestJson<ProfileModel[]>("/api/personalization/models");
  const personas = await listPrivatePersonas();
  guard();
  return [
    ...base.filter((m) => m.key.startsWith("base:")),
    ...personas.map((p) => {
      const v = p.current_version;
      const parent = base.find(
        (m) =>
          m.backend_id === v.base_backend_id &&
          m.model_name === v.base_model_name &&
          m.key.startsWith("base:")
      );
      return {
        key: `persona:${p.id}`,
        name: v.display_name,
        backend_id: v.base_backend_id,
        model_name: v.base_model_name,
        cloud: parent?.cloud ?? true,
        character: v.model_type === "character"
      };
    })
  ];
}

export async function saveSettings(expected: number, settings: ProfileSettings) {
  const guard = privateImageGuard();
  const profile = await getProfile();
  if (profile.revision !== expected)
    throw new Error("Your device profile changed. Refresh and try again.");
  if (settings.processor_backend_id || settings.processor_model) {
    const selected = (await models()).find(
      (m) =>
        m.key.startsWith("base:") &&
        m.backend_id === settings.processor_backend_id &&
        m.model_name === settings.processor_model
    );
    if (!selected) throw new Error("Choose an available processing model.");
    if (selected.cloud && !settings.allow_cloud_processor)
      throw new Error("Cloud processing needs separate permission.");
  }
  if (
    settings.idle_minutes < 1 ||
    settings.idle_minutes > 120 ||
    settings.context_chars < 1000 ||
    settings.context_chars > 16000
  )
    throw new Error("Check idle time and context size.");
  guard();
  return save({ ...profile, settings, last_error: null }, expected);
}

export async function editFact(
  expected: number,
  id: string | null,
  topic: string,
  content: string
) {
  validateFact(topic, content);
  const guard = privateImageGuard();
  const profile = await getProfile();
  if (profile.revision !== expected)
    throw new Error("Your device profile changed. Refresh and try again.");
  const old = profile.facts.find((f) => f.id === id);
  if (id && !old) throw new Error("Profile fact not found.");
  if (profile.facts.some((f) => f.topic === topic && f.id !== id))
    throw new Error("A fact with this topic already exists.");
  if (!old && profile.facts.length >= 200)
    throw new Error("The profile is full. Remove a fact first.");
  const now = Math.floor(Date.now() / 1000);
  const fact: ProfileFact = {
    id: old?.id ?? crypto.randomUUID(),
    topic,
    content: content.trim(),
    locked: true,
    version: (old?.version ?? 0) + 1,
    sources_json: old?.sources_json ?? "[]",
    created_at: old?.created_at ?? now,
    updated_at: now
  };
  if (old) profile.suppressed = [...new Set([...profile.suppressed, old.topic])];
  profile.facts = [...profile.facts.filter((f) => f.id !== id), fact];
  profile.history[fact.id] = [
    {
      version: fact.version,
      content: fact.content,
      sources_json: fact.sources_json,
      actor: "human" as const,
      created_at: now
    },
    ...(profile.history[fact.id] ?? [])
  ].slice(0, 20);
  guard();
  return save(profile, expected);
}

export async function forget(expected: number, id: string | null) {
  const guard = privateImageGuard();
  const profile = await getProfile();
  if (profile.revision !== expected)
    throw new Error("Your device profile changed. Refresh and try again.");
  const removed = profile.facts.filter((f) => !id || f.id === id);
  profile.suppressed = [...new Set([...profile.suppressed, ...removed.map((f) => f.topic)])];
  const processed = new Set(profile.processed);
  for (const fact of removed) {
    for (const source of factSources(fact)) processed.add(source.revision_id);
    delete profile.history[fact.id];
  }
  if (!id) {
    for (const chat of await listPrivateChats()) {
      for (const message of await listPrivateMessages(chat.id)) {
        for (const revision of message.revisions) processed.add(revision.id);
      }
    }
  }
  profile.processed = [...processed];
  profile.facts = profile.facts.filter((f) => id && f.id !== id);
  guard();
  return save(profile, expected);
}

export async function history(id: string) {
  return (await getProfile()).history[id] ?? [];
}

async function eligibleSources(
  profile: DeviceProfile,
  catalog: ProfileModel[],
  unprocessed: boolean
): Promise<InputSource[]> {
  const result: InputSource[] = [];
  const processed = new Set(profile.processed);
  const chats = await listPrivateChats();
  for (const chat of [...chats].sort((a, b) => a.created_at - b.created_at)) {
    const messages = await listPrivateMessages(chat.id);
    for (const message of messages) {
      const revision = message.active_revision;
      if (
        message.role !== "user" ||
        message.is_deleted ||
        message.status !== "complete" ||
        !revision ||
        (unprocessed && processed.has(revision.id))
      )
        continue;
      const response = messages
        .filter((m) => m.parent_message_id === message.id && m.role === "assistant")
        .sort((a, b) => a.created_at - b.created_at)[0];
      const personaId = response ? response.persona_id : chat.persona_id;
      const backend = response?.backend_id ?? chat.default_backend_id;
      const name = response?.model_name ?? chat.default_model_name;
      const key = personaId ? `persona:${personaId}` : `base:${backend}:${name}`;
      const model = catalog.find((m) => m.key === key);
      const base = catalog.find((m) => m.key === `base:${backend}:${name}`);
      if (!model || !base) continue;
      const versionId = response ? response.persona_version_id : chat.persona_version_id;
      const version =
        personaId && versionId
          ? (await listPrivatePersonaVersions(personaId)).find((v) => v.id === versionId)
          : null;
      const character =
        model.character || (!!personaId && (!version || version.model_type === "character"));
      if (!allowsProfile(profile.settings, { ...model, cloud: base.cloud, character })) continue;
      result.push({
        message_id: message.id,
        revision_id: revision.id,
        chat_id: chat.id,
        model_key: key,
        text: revision.content_text.slice(0, 6000),
        created_at: revision.created_at
      });
    }
  }
  return result.sort(
    (a, b) => a.created_at - b.created_at || a.revision_id.localeCompare(b.revision_id)
  );
}

export async function deviceContext(
  key: string,
  backend: string,
  name: string,
  character: boolean,
  query: string,
  personaVersion?: string | null
) {
  const guard = privateImageGuard();
  const profile = await getProfile();
  if (!profile.settings.enabled || !profile.facts.length) return null;
  const catalog = await models();
  const base = catalog.find((m) => m.key === `base:${backend}:${name}`);
  const selected = catalog.find((m) => m.key === key);
  if (key.startsWith("persona:")) {
    const version = (await listPrivatePersonaVersions(key.slice(8))).find(
      (v) => v.id === personaVersion
    );
    character ||= !version || version.model_type === "character";
  }
  if (
    !base ||
    !selected ||
    !allowsProfile(profile.settings, { ...selected, cloud: base.cloud, character })
  )
    return null;
  const sources = await eligibleSources(profile, catalog, false);
  const valid = profile.facts.filter(
    (f) =>
      f.locked ||
      factSources(f).some((s) =>
        sources.some(
          (v) =>
            v.revision_id === s.revision_id &&
            v.message_id === s.message_id &&
            v.chat_id === s.chat_id &&
            v.model_key === s.model_key &&
            v.text.includes(s.quote)
        )
      )
  );
  guard();
  return profileContext(valid, query, profile.settings.context_chars);
}

export async function applyProfileToPrompt<T extends import("./types").ProfilePromptPayload>(
  payload: T,
  recipient: { key: string; character: boolean; personaVersion?: string | null }
): Promise<T> {
  const guard = privateImageGuard();
  const profile = await deviceContext(
    recipient.key,
    payload.backend_id,
    payload.model_name,
    recipient.character,
    [...payload.messages].reverse().find((message) => message.role === "user")?.content_text ?? "",
    recipient.personaVersion
  );
  guard();
  const window = payload.inference_settings?.num_ctx ?? 4096;
  const used = payload.messages.reduce(
    (total, message) => total + contextWeight(message.content_text) + 32,
    0
  );
  const context = profile
    ? fitProfileContext(profile, Math.max(0, Math.min(window / 4, window - used - 512)))
    : null;
  if (!context) return payload;
  const messages = [...payload.messages];
  const position = messages.findIndex((message) => message.role !== "system");
  messages.splice(position < 0 ? messages.length : position, 0, {
    role: "user",
    content_text: context
  });
  return { ...payload, messages };
}

let processing = false;
let lastActivity = Date.now();
let lastEmptyScan = 0;
let streaming = 0;

export function profileGenerationActive(active: boolean) {
  streaming = Math.max(0, streaming + (active ? 1 : -1));
  lastActivity = Date.now();
}

export async function processNow() {
  if (processing) throw new Error("Device processing is already running.");
  if (streaming) throw new Error("Wait for the current generation to finish.");
  const guard = privateImageGuard();
  const identity = privateProfileIdentity();
  processing = true;
  let before: DeviceProfile | null = null;
  try {
    before = await getProfile();
    const settings = before.settings;
    if (!settings.enabled || !settings.processor_backend_id || !settings.processor_model)
      throw new Error("Enable personalization and choose a processing model first.");
    const catalog = await models();
    const all = await eligibleSources(before, catalog, true);
    let chars = 0;
    const sources: InputSource[] = [];
    for (const source of all) {
      if (sources.length === 16 || chars + [...source.text].length > 24000) break;
      chars += [...source.text].length;
      sources.push(source);
    }
    guard();
    if (!sources.length) {
      lastEmptyScan = Date.now();
      return before.requested
        ? await save({ ...before, requested: false }, before.revision)
        : before;
    }
    const now = Math.floor(Date.now() / 1000);
    const result = await requestJson<{
      facts: Array<{ topic: string; content: string; sources: ProfileSource[] }>;
    }>("/api/personalization/device-extract", {
      method: "POST",
      expectedServerInstanceId: identity.instance_id,
      body: JSON.stringify({
        expected_user_id: identity.user_id,
        backend_id: settings.processor_backend_id,
        model_name: settings.processor_model,
        allow_cloud_processor: settings.allow_cloud_processor,
        sources,
        topics: before.facts.map((f) => f.topic)
      })
    });
    guard();
    if (streaming) return before;
    const valid = await eligibleSources(before, await models(), false);
    for (const fact of result.facts) {
      validateFact(fact.topic, fact.content);
      if (
        !fact.sources.length ||
        fact.sources.some(
          (s) =>
            !valid.some(
              (v) =>
                v.revision_id === s.revision_id &&
                v.message_id === s.message_id &&
                v.text.includes(s.quote)
            )
        )
      )
        throw new Error("A source changed during processing.");
      if (before.suppressed.includes(fact.topic)) continue;
      const old = before.facts.find((f) => f.topic === fact.topic);
      if (
        old?.locked ||
        old?.content === fact.content ||
        (old &&
          factSources(old).some(
            (s) => s.created_at > Math.max(...fact.sources.map((v) => v.created_at))
          ))
      )
        continue;
      if (!old && before.facts.length >= 200) continue;
      const next: ProfileFact = {
        id: old?.id ?? crypto.randomUUID(),
        topic: fact.topic,
        content: fact.content,
        locked: false,
        version: (old?.version ?? 0) + 1,
        sources_json: JSON.stringify(fact.sources),
        created_at: old?.created_at ?? now,
        updated_at: now
      };
      before.facts = [...before.facts.filter((f) => f.id !== next.id), next];
      before.history[next.id] = [
        {
          version: next.version,
          content: next.content,
          sources_json: next.sources_json,
          actor: "processor" as const,
          created_at: now
        },
        ...(before.history[next.id] ?? [])
      ].slice(0, 20);
    }
    // Prune IDs for deleted/superseded messages; keep a content-free processed ledger.
    before.processed = [...new Set([...before.processed, ...sources.map((s) => s.revision_id)])];
    before.last_attempt_at = now;
    before.last_processed_at = now;
    before.last_error = null;
    const processedIds = new Set(before.processed);
    before.requested = valid.some((source) => !processedIds.has(source.revision_id));
    guard();
    return await save(before, before.revision);
  } catch (error) {
    guard();
    // Never persist raw provider output or sensitive request details in status.
    const current = await getProfile();
    if (
      before &&
      current.revision === before.revision &&
      error instanceof Error &&
      "code" in error &&
      error.code === "processor_busy"
    ) {
      return await save(
        {
          ...current,
          last_attempt_at: Math.floor(Date.now() / 1000),
          last_error: null,
          requested: true
        },
        current.revision
      );
    }
    if (before && current.revision === before.revision)
      await save(
        {
          ...current,
          last_attempt_at: Math.floor(Date.now() / 1000),
          last_error: "processing_failed"
        },
        current.revision
      );
    throw error;
  } finally {
    processing = false;
  }
}

export function startDeviceProfileWorker() {
  lastEmptyScan = 0;
  let stopped = false;
  let polling = false;
  const activity = () => {
    lastActivity = Date.now();
  };
  for (const name of ["pointerdown", "keydown", "touchstart"])
    window.addEventListener(name, activity, { passive: true });
  const timer = window.setInterval(async () => {
    if (stopped || polling || processing || streaming) return;
    polling = true;
    try {
      const profile = await getProfile();
      if (
        stopped ||
        !profile.settings.enabled ||
        !profile.settings.processor_model ||
        (!profile.requested && lastEmptyScan >= lastActivity) ||
        (!profile.requested && Date.now() - lastActivity < profile.settings.idle_minutes * 60000) ||
        (profile.last_attempt_at &&
          Date.now() / 1000 - profile.last_attempt_at <
            (profile.requested && !profile.last_error ? 10 : 120))
      )
        return;
      await processNow();
    } catch {
      /* Status is available in Memories; background work must not disrupt chat. */
    } finally {
      polling = false;
    }
  }, 15000);
  return () => {
    stopped = true;
    window.clearInterval(timer);
    for (const name of ["pointerdown", "keydown", "touchstart"])
      window.removeEventListener(name, activity);
  };
}
