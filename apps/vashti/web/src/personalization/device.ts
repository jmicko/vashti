import { requestJson } from "../api";
import {
  getPrivateChat,
  listPrivateChats,
  listPrivateMessages,
  listPrivatePersonas,
  listPrivatePersonaVersions,
  privateImageGuard,
  privateProfileIdentity,
  privateProfileSourceEpoch,
  readPrivateProfile,
  writePrivateProfile
} from "../privateChatStore";
import {
  allowsProfile,
  candidateTopic,
  contextWeight,
  defaultProfile,
  factSources,
  fitProfileContext,
  profileContext,
  topicToken,
  topicTokens,
  validateFact
} from "./core";
import type {
  DeviceProfile,
  InputSource,
  ProfileFact,
  ProfileModel,
  ProfileNotice,
  ProfileSettings,
  ProfileSource,
  RebuildProgress
} from "./types";

const IMPORT_SOURCE = "import:chatgpt";
type SourceRef = Pick<InputSource, "chat_id" | "message_id" | "revision_id" | "model_key">;

export async function getProfile(): Promise<DeviceProfile> {
  const guard = privateImageGuard();
  const profile = (await readPrivateProfile<DeviceProfile>()) ?? {
    ...defaultProfile(),
    processed: [],
    suppressed: [],
    history: {}
  };
  guard();
  profile.settings = { ...defaultProfile().settings, ...profile.settings };
  profile.rebuild_id ??= null;
  if (profile.suppressed.some((topic) => !topic.startsWith("sha256:"))) {
    try {
      return await save(
        { ...profile, suppressed: profile.suppressed.map(topicToken) },
        profile.revision
      );
    } catch (error) {
      guard();
      const current = await readPrivateProfile<DeviceProfile>();
      guard();
      if (current && current.revision !== profile.revision) return getProfile();
      throw error;
    }
  }
  return profile;
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
  if (old) profile.suppressed = [...new Set([...profile.suppressed, ...topicTokens(old.topic)])];
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
  profile.suppressed = [
    ...new Set([...profile.suppressed, ...removed.flatMap((f) => topicTokens(f.topic))])
  ];
  const processed = new Set(profile.processed);
  const forgotten = new Set(profile.forgotten ?? []);
  for (const fact of removed) {
    for (const source of factSources(fact)) forgotten.add(source.revision_id);
    delete profile.history[fact.id];
  }
  if (profile.rebuild_draft) {
    const suppressed = new Set(profile.suppressed);
    const draft = profile.rebuild_draft;
    draft.facts = draft.facts.filter((staged) => {
      if (!isSuppressed(suppressed, staged.topic)) return true;
      delete draft.history[staged.id];
      return false;
    });
  }
  if (!id) {
    for (const chat of await listPrivateChats()) {
      for (const message of await listPrivateMessages(chat.id)) {
        for (const revision of message.revisions) forgotten.add(revision.id);
      }
    }
    profile.rebuild_id = null;
    delete profile.rebuild_draft;
    profile.requested = false;
  }
  for (const revision of forgotten) processed.add(revision);
  profile.forgotten = [...forgotten];
  profile.processed = [...processed];
  profile.facts = profile.facts.filter((f) => id && f.id !== id);
  guard();
  return save(profile, expected);
}

export async function history(id: string) {
  return (await getProfile()).history[id] ?? [];
}

export async function rebuild(expected: number) {
  const guard = privateImageGuard();
  const profile = await getProfile();
  if (profile.revision !== expected) throw new Error("Your device profile changed. Refresh and try again.");
  const settings = profile.settings;
  if (!settings.enabled || !settings.processor_backend_id || !settings.processor_model)
    throw new Error("Enable personalization and choose a processing model first.");
  if (profile.rebuild_id) throw new Error("A profile rebuild is already queued.");
  const processor = (await models()).find((model) => model.key === `base:${settings.processor_backend_id}:${settings.processor_model}`);
  if (!processor || (processor.cloud && !settings.allow_cloud_processor))
    throw new Error("Choose an available, permitted processing model first.");
  guard();
  profile.rebuild_draft = { facts: [], history: {}, processed: [] };
  profile.rebuild_id = crypto.randomUUID();
  profile.requested = false;
  profile.last_attempt_at = null;
  profile.last_error = null;
  sourceQueue = undefined;
  guard();
  return save(profile, expected);
}

export async function cancelRebuild(id: string): Promise<DeviceProfile> {
  const guard = privateImageGuard();
  for (let attempt = 0; attempt < 3; attempt++) {
    const profile = await getProfile();
    guard();
    if (profile.rebuild_id !== id) throw new Error("This rebuild is no longer running.");
    const expected = profile.revision;
    profile.rebuild_id = null;
    delete profile.rebuild_draft;
    profile.requested = false;
    profile.last_attempt_at = Math.floor(Date.now() / 1000);
    profile.last_error = null;
    try {
      const saved = await save(profile, expected);
      sourceQueue = undefined;
      if (activeReplay?.id === id) activeReplay.abort.abort();
      return saved;
    } catch (error) {
      guard();
      if (attempt === 2 || (await getProfile()).revision === expected) throw error;
    }
  }
  throw new Error("Your device profile changed. Refresh and try again.");
}

export async function rebuildProgress(): Promise<RebuildProgress | null> {
  const profile = await getProfile();
  if (!profile.rebuild_id) return null;
  const phase = !profile.settings.enabled || !profile.settings.processor_model ? "paused"
    : profile.last_error ? "needs_attention"
    : streaming ? "waiting_for_generation"
    : activeReplay?.id === profile.rebuild_id ? "replaying"
    : waitingForIdle(profile) ? "waiting_for_idle"
    : profile.last_attempt_at ? "replaying" : "queued";
  return { id: profile.rebuild_id, processed: profile.rebuild_draft?.processed.length ?? 0, phase };
}

export async function profileNotices(): Promise<ProfileNotice[]> {
  return (await readPrivateProfile<DeviceProfile>())?.notices ?? [];
}

export async function readNotice(id: string) {
  const guard = privateImageGuard(), profile = await getProfile();
  const notice = profile.notices?.find((item) => item.id === id);
  if (!notice || notice.read_at !== null) return;
  notice.read_at = Math.floor(Date.now() / 1000);
  guard();
  await save(profile, profile.revision);
}

function effectiveLedger(profile: DeviceProfile) {
  return profile.rebuild_id
    ? [...(profile.forgotten ?? []), ...(profile.rebuild_draft?.processed ?? [])]
    : profile.processed;
}

function workingFacts(profile: DeviceProfile) {
  if (!profile.rebuild_id) return profile.facts;
  const manual = profile.facts.filter((fact) => fact.locked);
  return [...manual, ...(profile.rebuild_draft?.facts ?? []).filter((fact) => !manual.some((item) => item.topic === fact.topic))];
}

function supportedEvidence(sources: ProfileSource[], checked: Map<string, InputSource>) {
  return sources.filter((source) => {
    const current = checked.get(source.revision_id);
    return current?.chat_id === source.chat_id && current.message_id === source.message_id
      && current.model_key === source.model_key && current.text.includes(source.quote);
  });
}

function isSuppressed(suppressed: Set<string>, topic: string) {
  return topicTokens(topic).some((token) => suppressed.has(token));
}

async function finishScan(profile: DeviceProfile, catalog: ProfileModel[]) {
  if (profile.rebuild_id) {
    const draft = profile.rebuild_draft;
    const facts = workingFacts(profile);
    const scoped = scopeSources(facts.filter((fact) => !fact.locked).flatMap((fact) => [
      ...factSources(fact), ...(draft?.history[fact.id] ?? []).flatMap(factSources)
    ]));
    const sources = new Map((await eligibleSources(profile, catalog, false, scoped)).map((entry) => [entry.revision_id, entry]));
    const nextHistory: DeviceProfile["history"] = {};
    const nextFacts: ProfileFact[] = [];
    const suppressed = new Set(profile.suppressed);
    for (const fact of facts) {
      if (nextFacts.length === 200) break;
      if (fact.locked) {
        nextFacts.push(fact);
        nextHistory[fact.id] = profile.history[fact.id] ?? [];
        continue;
      }
      if (isSuppressed(suppressed, fact.topic)) continue;
      const valid = supportedEvidence(factSources(fact), sources);
      if (!valid.length) continue;
      nextFacts.push({ ...fact, sources_json: JSON.stringify(valid) });
      nextHistory[fact.id] = (draft?.history[fact.id] ?? []).flatMap((version) => {
        const evidence = version.version === fact.version ? valid : supportedEvidence(factSources(version), sources);
        return evidence.length ? [{ ...version, sources_json: JSON.stringify(evidence) }] : [];
      });
    }
    profile.facts = nextFacts;
    profile.history = nextHistory;
    profile.processed = [...new Set(effectiveLedger(profile))];
    delete profile.rebuild_draft;
    profile.last_processed_at = Math.floor(Date.now() / 1000);
    const prior = profile.notices ?? [];
    profile.notices = [
      { id: profile.rebuild_id, created_at: Math.floor(Date.now() / 1000), read_at: null },
      ...prior.filter((item) => item.read_at === null),
      ...prior.filter((item) => item.read_at !== null).slice(0, 100)
    ];
    profile.rebuild_id = null;
  }
  profile.requested = false;
  profile.last_error = null;
}

async function eligibleSources(
  profile: DeviceProfile,
  catalog: ProfileModel[],
  unprocessed: boolean,
  sourcesByChat?: ReadonlyMap<string, ReadonlySet<string>>
): Promise<InputSource[]> {
  const result: InputSource[] = [];
  const processed = new Set(effectiveLedger(profile));
  const chats = sourcesByChat
    ? (await Promise.all([...sourcesByChat.keys()].map(getPrivateChat))).filter((chat) => chat !== null)
    : await listPrivateChats();
  for (const chat of [...chats].sort((a, b) => a.created_at - b.created_at)) {
    if (chat.do_not_remember) continue;
    const messageIds = sourcesByChat?.get(chat.id);
    const messages = await listPrivateMessages(chat.id);
    for (const message of messages) {
      const revision = message.active_revision;
      if (
        (messageIds && !messageIds.has(message.id)) ||
        message.role !== "user" ||
        message.is_deleted ||
        message.status !== "complete" ||
        !revision ||
        (unprocessed && processed.has(revision.id))
      )
        continue;
      if (!revision.content_text.trim()) continue;
      if (message.import_source === "chatgpt") {
        if (profile.settings.learn_from_imports && chat.allow_profile_learning !== false && revision.source === "original") {
          result.push({
            message_id: message.id, revision_id: revision.id, chat_id: chat.id,
            model_key: IMPORT_SOURCE, text: revision.content_text.slice(0, 6000), created_at: revision.created_at
          });
        }
        continue;
      }
      const snapshot = revision.profile_source;
      // Old edits have no trustworthy model snapshot. Never inherit an earlier revision's model.
      if (!snapshot && revision.source !== "original") continue;
      const response = snapshot ? undefined : messages
        .filter((m) => m.parent_message_id === message.id && m.role === "assistant")
        .sort((a, b) => a.created_at - b.created_at)[0];
      if (!snapshot && !response) continue;
      const personaId = snapshot
        ? snapshot.persona_id
        : response
          ? response.persona_id
          : chat.persona_id;
      const backend = snapshot?.backend_id ?? response?.backend_id ?? chat.default_backend_id;
      const name = snapshot?.model_name ?? response?.model_name ?? chat.default_model_name;
      const key = personaId ? `persona:${personaId}` : `base:${backend}:${name}`;
      const model = catalog.find((m) => m.key === key);
      const base = catalog.find((m) => m.key === `base:${backend}:${name}`);
      if (!model || !base) continue;
      const versionId = snapshot
        ? snapshot.persona_version_id
        : response
          ? response.persona_version_id
          : chat.persona_version_id;
      const version =
        personaId && versionId
          ? (await listPrivatePersonaVersions(personaId)).find((v) => v.id === versionId)
          : null;
      const character =
        model.character || (!!personaId && (!version || version.model_type === "character"));
      if (
        !allowsProfile(profile.settings, {
          ...model,
          cloud: base.cloud,
          character
        })
      )
        continue;
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

function scopeSources(sources: Iterable<Pick<SourceRef, "chat_id" | "message_id">>) {
  const result = new Map<string, Set<string>>();
  for (const source of sources) {
    let ids = result.get(source.chat_id);
    if (!ids) result.set(source.chat_id, ids = new Set());
    ids.add(source.message_id);
  }
  return result;
}

let sourceQueue: { epoch: string; policy: string; refs: SourceRef[] } | undefined;

async function pendingSources(profile: DeviceProfile, catalog: ProfileModel[]) {
  const epoch = privateProfileSourceEpoch();
  const policy = JSON.stringify([profile.settings, profile.rebuild_id, catalog]);
  if (sourceQueue?.epoch !== epoch || sourceQueue.policy !== policy) {
    const guard = privateImageGuard();
    const refs = (await eligibleSources(profile, catalog, true)).map(({ chat_id, message_id, revision_id, model_key }) =>
      ({ chat_id, message_id, revision_id, model_key }));
    guard();
    sourceQueue = { epoch, policy, refs };
  }
  const processed = new Set(effectiveLedger(profile));
  return sourceQueue.refs.filter((source) => !processed.has(source.revision_id));
}

async function replayHints(profile: DeviceProfile, catalog: ProfileModel[], sources: InputSource[], facts: ProfileFact[]) {
  const words = new Set(sources.flatMap((source) => source.text.toLowerCase().split(/[^\p{L}\p{N}]+/u)).filter((word) => word.length >= 4));
  const ranked = facts.filter((fact) => !fact.locked).map((fact) => ({ fact, score: [...words].filter((word) => `${fact.topic} ${fact.content}`.toLowerCase().includes(word)).length }))
    .sort((a, b) => b.score - a.score || b.fact.updated_at - a.fact.updated_at).slice(0, 12).map(({ fact }) => fact);
  const ancestors = new Map<number, Array<{ chat_id: string; message_id: string }>>();
  const messagesByChat = new Map<string, Awaited<ReturnType<typeof listPrivateMessages>>>();
  for (const [index, source] of sources.entries()) {
    if (index > 0 && sources[index - 1].chat_id === source.chat_id) continue;
    let messages = messagesByChat.get(source.chat_id);
    if (!messages) {
      messages = await listPrivateMessages(source.chat_id);
      messagesByChat.set(source.chat_id, messages);
    }
    const byId = new Map(messages.map((message) => [message.id, message]));
    const prior: Array<{ chat_id: string; message_id: string }> = [];
    let parent = byId.get(source.message_id)?.parent_message_id;
    for (let depth = 0; parent && depth < 8; depth++) {
      const message = byId.get(parent);
      if (!message || message.is_deleted) break;
      if (message.role === "user") prior.push({ chat_id: source.chat_id, message_id: message.id });
      parent = message.parent_message_id;
    }
    ancestors.set(index, prior);
  }
  const checked = await eligibleSources(profile, catalog, false, scopeSources([
    ...ranked.flatMap(factSources), ...[...ancestors.values()].flat()
  ]));
  const byRevision = new Map(checked.map((source) => [source.revision_id, source]));
  const byMessage = new Map(checked.map((source) => [source.message_id, source]));
  const forgotten = new Set(profile.forgotten);
  const suppressed = new Set(profile.suppressed);
  let chars = 0;
  return {
    facts: ranked.filter((fact) => !isSuppressed(suppressed, fact.topic) && supportedEvidence(factSources(fact), byRevision).length)
      .map((fact) => ({ topic: fact.topic, content: [...fact.content].slice(0, 320).join("") })),
    earlier_turns: [...ancestors].flatMap(([index, items]) => {
      const excerpts: string[] = [];
      for (const item of items) {
        if (excerpts.length === 2 || chars >= 2400) break;
        const prior = byMessage.get(item.message_id);
        if (prior && prior.created_at <= sources[index].created_at && !forgotten.has(prior.revision_id)) {
          const excerpt = [...prior.text].slice(0, Math.min(240, 2400 - chars));
          chars += excerpt.length;
          excerpts.unshift(excerpt.join(""));
        }
      }
      return excerpts.length ? [{ source: index, excerpts }] : [];
    })
  };
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
    !allowsProfile(profile.settings, {
      ...selected,
      cloud: base.cloud,
      character
    })
  )
    return null;
  // Validate only the saved citations when building a foreground prompt.
  const sourcesByChat = scopeSources(profile.facts.filter((fact) => !fact.locked).flatMap(factSources));
  const sources = await eligibleSources(profile, catalog, false, sourcesByChat);
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
  recipient: {
    key: string;
    character: boolean;
    personaVersion?: string | null;
  }
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
let activeReplay: { id: string; abort: AbortController } | undefined;
let lastActivity = Date.now();
let lastEmptyScan = 0;
let lastEmptyEpoch = "";
let streaming = 0;

function waitingForIdle(profile: DeviceProfile) {
  return !profile.requested && Date.now() - lastActivity < profile.settings.idle_minutes * 60000;
}

export function profileGenerationActive(active: boolean) {
  streaming = Math.max(0, streaming + (active ? 1 : -1));
  lastActivity = Date.now();
}

export async function processNow(manual = true) {
  if (processing) throw new Error("Device processing is already running.");
  if (streaming) throw new Error("Wait for the current generation to finish.");
  const guard = privateImageGuard();
  const identity = privateProfileIdentity();
  processing = true;
  let before: DeviceProfile | null = null;
  try {
    before = await getProfile();
    const abort = before.rebuild_id ? new AbortController() : undefined;
    if (before.rebuild_id && abort) activeReplay = { id: before.rebuild_id, abort };
    const settings = before.settings;
    if (!settings.enabled || !settings.processor_backend_id || !settings.processor_model)
      throw new Error("Enable personalization and choose a processing model first.");
    const catalog = await models();
    let refs = (await pendingSources(before, catalog)).slice(0, 16);
    let all = await eligibleSources(before, catalog, true, scopeSources(refs));
    const matches = (source: InputSource) => refs.some((ref) => ref.revision_id === source.revision_id && ref.model_key === source.model_key);
    if (all.filter(matches).length !== refs.length) {
      // Another tab may have edited one of the queued messages.
      sourceQueue = undefined;
      refs = (await pendingSources(before, catalog)).slice(0, 16);
      all = await eligibleSources(before, catalog, true, scopeSources(refs));
    }
    let chars = 0;
    const sources: InputSource[] = [];
    for (const source of all) {
      if (!matches(source)) continue;
      if (sources.length === 16 || chars + [...source.text].length > 24000) break;
      chars += [...source.text].length;
      sources.push(source);
    }
    guard();
    if (streaming) return before;
    if (!sources.length) {
      lastEmptyScan = Date.now();
      lastEmptyEpoch = privateProfileSourceEpoch();
      sourceQueue = undefined;
      if (!before.requested && !before.rebuild_id && !before.last_error) return before;
      await finishScan(before, catalog);
      return await save(before, before.revision);
    }
    let facts = workingFacts(before);
    const replay = await replayHints(before, catalog, sources, facts);
    const current = await getProfile();
    if (streaming || current.revision !== before.revision || abort?.signal.aborted) return current;
    guard();
    const now = Math.floor(Date.now() / 1000);
    const result = await requestJson<{
      facts: Array<{
        topic: string;
        replaces_topic?: string;
        content: string;
        sources: ProfileSource[];
      }>;
    }>("/api/personalization/device-extract", {
      method: "POST",
      expectedServerInstanceId: identity.instance_id,
      signal: abort?.signal,
      body: JSON.stringify({
        expected_user_id: identity.user_id,
        backend_id: settings.processor_backend_id,
        model_name: settings.processor_model,
        allow_cloud_processor: settings.allow_cloud_processor,
        sources,
        topics: facts.slice(0, 200).map((f) => f.topic),
        replay
      })
    });
    guard();
    if (abort?.signal.aborted) return getProfile();
    if (streaming) return before;
    const valid = await eligibleSources(before, await models(), false, scopeSources(sources));
    const draft = before.rebuild_id ? (before.rebuild_draft ??= { facts: [], history: {}, processed: [] }) : undefined;
    const history = draft?.history ?? before.history;
    const suppressed = new Set(before.suppressed);
    const sourceTime = (fact: (typeof result.facts)[number]) => Math.max(...fact.sources.map((source) => source.created_at));
    for (const fact of result.facts.sort((a, b) => sourceTime(a) - sourceTime(b))) {
      validateFact(fact.topic, fact.content);
      if (fact.replaces_topic !== undefined) validateFact(fact.replaces_topic, fact.content);
      if (
        !fact.sources.length ||
        fact.sources.some(
          (s) =>
            !valid.some(
              (v) =>
                v.chat_id === s.chat_id &&
                v.revision_id === s.revision_id &&
                v.message_id === s.message_id &&
                v.model_key === s.model_key &&
                sources.some((sent) => sent.revision_id === s.revision_id && sent.model_key === s.model_key) &&
                v.text.includes(s.quote)
            )
        )
      )
        throw new Error("A source changed during processing.");
      const topic = candidateTopic(fact.topic, fact.replaces_topic, facts);
      if (!topic || isSuppressed(suppressed, fact.topic) || isSuppressed(suppressed, topic)) continue;
      const old = facts.find((f) => f.topic === topic);
      if (old && !old.locked && old.content === fact.content) {
        const seen = new Set<string>();
        const sources = [...fact.sources, ...factSources(old)]
          .sort((a, b) => b.created_at - a.created_at)
          .filter((source) => {
            if (seen.has(source.revision_id)) return false;
            seen.add(source.revision_id);
            return true;
          })
          .slice(0, 3);
        facts = facts.map((current) =>
          current.id === old.id
            ? {
                ...current,
                sources_json: JSON.stringify(sources),
                updated_at: now
              }
            : current
        );
        continue;
      }
      if (
        old?.locked ||
        (old &&
          factSources(old).some(
            (s) => s.created_at > sourceTime(fact)
          ))
      )
        continue;
      if (!old && facts.length >= 200) {
        if (!draft) continue;
        const time = Math.max(...fact.sources.map((entry) => entry.created_at));
        const victim = facts.filter((entry) => !entry.locked)
          .map((entry) => ({ entry, time: Math.max(0, ...factSources(entry).map((source) => source.created_at)) }))
          .filter((entry) => entry.time < time)
          .sort((a, b) => a.time - b.time || a.entry.topic.localeCompare(b.entry.topic))[0]?.entry;
        if (!victim) continue;
        facts = facts.filter((entry) => entry.id !== victim.id);
        delete history[victim.id];
      }
      const next: ProfileFact = {
        id: old?.id ?? crypto.randomUUID(),
        topic,
        content: fact.content,
        locked: false,
        version: (old?.version ?? 0) + 1,
        sources_json: JSON.stringify(fact.sources),
        created_at: old?.created_at ?? now,
        updated_at: now
      };
      facts = [...facts.filter((f) => f.id !== next.id), next];
      history[next.id] = [
        {
          version: next.version,
          content: next.content,
          sources_json: next.sources_json,
          actor: "processor" as const,
          created_at: now
        },
        ...(history[next.id] ?? [])
      ].slice(0, 20);
    }
    // Retain only revision IDs in the processed ledger, never source text.
    const newIds = sources.map((s) => s.revision_id);
    if (draft) {
      draft.facts = facts.filter((fact) => !fact.locked);
      draft.processed = [...new Set([...draft.processed, ...newIds])];
    } else {
      before.facts = facts;
      before.processed = [...new Set([...before.processed, ...newIds])];
    }
    before.last_attempt_at = now;
    if (!draft) before.last_processed_at = now;
    before.last_error = null;
    const hasRemaining = (await pendingSources(before, catalog)).length > 0;
    before.requested = (manual || before.requested) && hasRemaining;
    if (!hasRemaining) await finishScan(before, catalog);
    guard();
    return await save(before, before.revision);
  } catch (error) {
    guard();
    // Never persist raw provider output or sensitive request details in status.
    const current = await getProfile();
    if (before?.rebuild_id && current.rebuild_id !== before.rebuild_id) return current;
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
          requested: current.requested || manual
        },
        current.revision
      );
    }
    const code =
      error instanceof Error &&
      "code" in error &&
      typeof error.code === "string" &&
      [
        "processor_invalid_evidence",
        "processor_invalid_schema",
        "processor_invalid_json",
        "processor_incomplete",
        "processor_unavailable"
      ].includes(error.code)
        ? error.code
        : "processing_failed";
    if (before && current.revision === before.revision)
      await save(
        {
          ...current,
          last_attempt_at: Math.floor(Date.now() / 1000),
          last_error: code
        },
        current.revision
      );
    throw error;
  } finally {
    activeReplay = undefined;
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
        (!profile.requested && !profile.rebuild_id && lastEmptyScan >= lastActivity && lastEmptyEpoch === privateProfileSourceEpoch()) ||
        waitingForIdle(profile) ||
        (profile.last_attempt_at &&
          Date.now() / 1000 - profile.last_attempt_at <
            ((profile.requested || profile.rebuild_id) && !profile.last_error ? 10 : 120))
      )
        return;
      await processNow(false);
    } catch {
      /* Status is available in Memories; background work must not disrupt chat. */
    } finally {
      polling = false;
    }
  }, 15000);
  return () => {
    stopped = true;
    sourceQueue = undefined;
    window.clearInterval(timer);
    for (const name of ["pointerdown", "keydown", "touchstart"])
      window.removeEventListener(name, activity);
  };
}
