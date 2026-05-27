import { gcm as aesGcm } from "@noble/ciphers/aes.js";
import type { ChatInferenceSettings, MessageStats } from "./types";

const LEGACY_DB_NAME = "vashti-private-local";
const DB_NAME_PREFIX = "vashti-private-local";
const DB_VERSION = 3;
const CHAT_STORE = "private_chats";
const MESSAGE_STORE = "private_messages";
const PERSONA_STORE = "private_personas";
const PERSONA_VERSION_STORE = "private_persona_versions";
const HOSTED_CHAT_CACHE_STORE = "hosted_chat_cache";
const MODEL_CACHE_STORE = "model_cache";
const MODEL_CACHE_ID = "model-picker";

type PrivateVaultKeyResponse = {
  user_id: string;
  key_material: string;
};

type EncryptedPayload = {
  v: 1;
  iv: string;
  data: string;
};

type PrivateStoreRecord = {
  id: string;
  chat_id?: string;
  persona_id?: string;
  created_at?: number;
  updated_at?: number;
  last_message_at?: number;
  encrypted_payload?: EncryptedPayload;
};

export type PrivateChatSummary = {
  id: string;
  title: string;
  default_backend_id: string;
  backend_name: string;
  default_model_name: string;
  persona_id?: string | null;
  persona_version_id?: string | null;
  persona_name?: string | null;
  system_prompt_override?: string | null;
  inference_settings?: ChatInferenceSettings;
  active_root_message_id: string | null;
  created_at: number;
  updated_at: number;
  last_message_at: number;
  message_count: number;
};

export type PrivateChatDetail = {
  id: string;
  title: string;
  default_backend_id: string;
  backend_name: string;
  default_model_name: string;
  persona_id?: string | null;
  persona_version_id?: string | null;
  persona_name?: string | null;
  system_prompt_override?: string | null;
  inference_settings?: ChatInferenceSettings;
  active_root_message_id: string | null;
  created_at: number;
  updated_at: number;
  last_message_at: number;
};

export type PrivateChatMessageRevision = {
  id: string;
  content_text: string;
  thinking_text: string;
  source: string;
  created_at: number;
};

export type PrivateChatAttachment = {
  id: string;
  chat_id?: string;
  message_id: string | null;
  revision_id: string | null;
  original_filename: string;
  mime_type: string;
  size_bytes: number;
  attachment_kind: string;
  created_at?: number;
  data_url?: string;
  text_content?: string;
};

export type PrivateChatMessage = {
  id: string;
  chat_id: string;
  parent_message_id: string | null;
  active_child_message_id: string | null;
  active_revision_id: string | null;
  role: string;
  status: string;
  is_deleted: boolean;
  backend_id: string | null;
  model_name: string | null;
  persona_id?: string | null;
  persona_version_id?: string | null;
  persona_name_snapshot?: string | null;
  think_mode: string | null;
  done_reason: string | null;
  error_text: string | null;
  stats?: MessageStats | null;
  started_at: number | null;
  completed_at: number | null;
  created_at: number;
  updated_at: number;
  active_revision: PrivateChatMessageRevision | null;
  revisions: PrivateChatMessageRevision[];
  revision_count: number;
  attachments: PrivateChatAttachment[];
};

export type CreatePrivateChatParams = {
  title: string;
  backendId: string;
  backendName: string;
  modelName: string;
  personaId?: string | null;
  personaVersionId?: string | null;
  personaName?: string | null;
  systemPromptOverride?: string | null;
  inferenceSettings?: ChatInferenceSettings;
};

export type CreatePrivateMessageParams = {
  chatId: string;
  parentMessageId: string | null;
  role: string;
  contentText: string;
  thinkingText?: string;
  status?: string;
  backendId?: string | null;
  modelName?: string | null;
  personaId?: string | null;
  personaVersionId?: string | null;
  personaNameSnapshot?: string | null;
  thinkMode?: string | null;
  createdAt?: number;
};

export type PrivatePersonaVersion = {
  id: string;
  persona_id: string;
  version_number: number;
  display_name: string;
  avatar_attachment_id: string | null;
  base_backend_id: string;
  base_backend_name: string;
  base_model_name: string;
  system_prompt: string;
  tool_policy_json: string | null;
  source_persona_id?: string | null;
  source_persona_version_id?: string | null;
  created_at: number;
};

export type PrivatePersona = {
  id: string;
  current_version_id: string;
  created_at: number;
  updated_at: number;
  current_version: PrivatePersonaVersion;
};

export type SavePrivatePersonaParams = {
  displayName: string;
  baseBackendId: string;
  baseBackendName: string;
  baseModelName: string;
  systemPrompt: string;
  avatarAttachmentId?: string | null;
  toolPolicyJson?: string | null;
  sourcePersonaId?: string | null;
  sourcePersonaVersionId?: string | null;
};

export type CachedHostedChat<
  TChat extends { id: string; updated_at: number },
  TMessage
> = {
  id: string;
  chat: TChat;
  active_root_message_id: string | null;
  messages: TMessage[];
  updated_at: number;
  cached_at: number;
};

let currentUserId: string | null = null;
let dbPromise: Promise<IDBDatabase> | null = null;
let vaultKeyPromise: Promise<PrivateVaultKeyResponse> | null = null;
let vaultKeyBytesPromise: Promise<Uint8Array> | null = null;
let webCryptoKeyPromise: Promise<CryptoKey> | null = null;
let legacyMigrationPromise: Promise<void> | null = null;

export function setPrivateStorageUser(userId: string) {
  if (currentUserId === userId) {
    return;
  }

  currentUserId = userId;
  vaultKeyPromise = null;
  vaultKeyBytesPromise = null;
  webCryptoKeyPromise = null;
  legacyMigrationPromise = null;
  if (dbPromise) {
    void dbPromise.then((db) => db.close()).catch(() => undefined);
    dbPromise = null;
  }
}

export function resetPrivateStorageUser() {
  currentUserId = null;
  vaultKeyPromise = null;
  vaultKeyBytesPromise = null;
  webCryptoKeyPromise = null;
  legacyMigrationPromise = null;
  if (dbPromise) {
    void dbPromise.then((db) => db.close()).catch(() => undefined);
    dbPromise = null;
  }
}

export function unixTimestamp() {
  return Math.floor(Date.now() / 1000);
}

export function privateId(prefix: string) {
  return `${prefix}-${randomId()}`;
}

function randomId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }

  const bytes = new Uint8Array(16);
  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(bytes);
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256);
    }
  }

  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = [...bytes].map((byte) => byte.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex
    .slice(6, 8)
    .join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
}

export function createPrivateMessage({
  chatId,
  parentMessageId,
  role,
  contentText,
  thinkingText = "",
  status = "complete",
  backendId = null,
  modelName = null,
  personaId = null,
  personaVersionId = null,
  personaNameSnapshot = null,
  thinkMode = null,
  createdAt = unixTimestamp()
}: CreatePrivateMessageParams): PrivateChatMessage {
  const revision = {
    id: privateId("private-revision"),
    content_text: contentText,
    thinking_text: thinkingText,
    source: "original",
    created_at: createdAt
  };

  return {
    id: privateId("private-message"),
    chat_id: chatId,
    parent_message_id: parentMessageId,
    active_child_message_id: null,
    active_revision_id: revision.id,
    role,
    status,
    is_deleted: false,
    backend_id: backendId,
    model_name: modelName,
    persona_id: personaId,
    persona_version_id: personaVersionId,
    persona_name_snapshot: personaNameSnapshot,
    think_mode: thinkMode,
    done_reason: null,
    error_text: null,
    stats: null,
    started_at: status === "streaming" ? createdAt : null,
    completed_at: status === "streaming" ? null : createdAt,
    created_at: createdAt,
    updated_at: createdAt,
    active_revision: revision,
    revisions: [revision],
    revision_count: 1,
    attachments: []
  };
}

export async function listPrivateChats(): Promise<PrivateChatSummary[]> {
  const db = await openPrivateDb();
  const [chats, messages] = await Promise.all([
    getAllPrivateRecords<PrivateChatDetail>(db, CHAT_STORE),
    getAllPrivateRecords<PrivateChatMessage>(db, MESSAGE_STORE)
  ]);
  const messageCounts = new Map<string, number>();
  for (const message of messages) {
    messageCounts.set(message.chat_id, (messageCounts.get(message.chat_id) ?? 0) + 1);
  }

  return chats
    .map((chat) => ({
      ...chat,
      message_count: messageCounts.get(chat.id) ?? 0
    }))
    .sort((left, right) => right.last_message_at - left.last_message_at || left.title.localeCompare(right.title));
}

export async function createPrivateChat({
  title,
  backendId,
  backendName,
  modelName,
  personaId = null,
  personaVersionId = null,
  personaName = null,
  systemPromptOverride = null,
  inferenceSettings = {}
}: CreatePrivateChatParams): Promise<PrivateChatDetail> {
  const now = unixTimestamp();
  const chat: PrivateChatDetail = {
    id: privateId("private-chat"),
    title,
    default_backend_id: backendId,
    backend_name: backendName,
    default_model_name: modelName,
    persona_id: personaId,
    persona_version_id: personaVersionId,
    persona_name: personaName,
    system_prompt_override: systemPromptOverride,
    inference_settings: inferenceSettings,
    active_root_message_id: null,
    created_at: now,
    updated_at: now,
    last_message_at: now
  };

  await savePrivateChat(chat);
  return chat;
}

export async function getPrivateChat(chatId: string): Promise<PrivateChatDetail | null> {
  const db = await openPrivateDb();
  const tx = db.transaction(CHAT_STORE, "readonly");
  const chat = await requestResult<PrivateStoreRecord | PrivateChatDetail | undefined>(
    tx.objectStore(CHAT_STORE).get(chatId)
  );
  await transactionDone(tx);
  return chat ? readPrivateRecord<PrivateChatDetail>(chat) : null;
}

export async function savePrivateChat(chat: PrivateChatDetail): Promise<void> {
  const db = await openPrivateDb();
  const record = await privateStoreRecord(chat, {
    created_at: chat.created_at,
    updated_at: chat.updated_at,
    last_message_at: chat.last_message_at
  });
  const tx = db.transaction(CHAT_STORE, "readwrite");
  tx.objectStore(CHAT_STORE).put(record);
  await transactionDone(tx);
}

export async function renamePrivateChat(chatId: string, title: string): Promise<PrivateChatDetail> {
  const chat = await getPrivateChat(chatId);
  if (!chat) {
    throw new Error("Private chat not found");
  }

  const updated = {
    ...chat,
    title,
    updated_at: unixTimestamp()
  };
  await savePrivateChat(updated);
  return updated;
}

export async function deletePrivateChat(chatId: string): Promise<void> {
  const db = await openPrivateDb();
  const tx = db.transaction([CHAT_STORE, MESSAGE_STORE], "readwrite");
  tx.objectStore(CHAT_STORE).delete(chatId);

  const index = tx.objectStore(MESSAGE_STORE).index("chat_id");
  const cursorRequest = index.openCursor(IDBKeyRange.only(chatId));
  cursorRequest.onsuccess = () => {
    const cursor = cursorRequest.result;
    if (!cursor) {
      return;
    }

    cursor.delete();
    cursor.continue();
  };

  await transactionDone(tx);
}

export async function listPrivateMessages(chatId: string): Promise<PrivateChatMessage[]> {
  const db = await openPrivateDb();
  const tx = db.transaction(MESSAGE_STORE, "readonly");
  const records = await requestResult<Array<PrivateStoreRecord | PrivateChatMessage>>(
    tx.objectStore(MESSAGE_STORE).index("chat_id").getAll(chatId)
  );
  await transactionDone(tx);
  const messages = await Promise.all(
    records.map((record) => readPrivateRecord<PrivateChatMessage>(record))
  );
  return messages.map(normalizePrivateMessage).sort(compareMessagesByCreatedAt);
}

export async function savePrivateMessage(message: PrivateChatMessage): Promise<void> {
  const db = await openPrivateDb();
  const record = await privateStoreRecord(message, {
    chat_id: message.chat_id,
    created_at: message.created_at,
    updated_at: message.updated_at
  });
  const tx = db.transaction(MESSAGE_STORE, "readwrite");
  tx.objectStore(MESSAGE_STORE).put(record);
  await transactionDone(tx);
}

export async function savePrivateMessages(messages: PrivateChatMessage[]): Promise<void> {
  if (messages.length === 0) {
    return;
  }

  const db = await openPrivateDb();
  const records = await Promise.all(
    messages.map((message) =>
      privateStoreRecord(message, {
        chat_id: message.chat_id,
        created_at: message.created_at,
        updated_at: message.updated_at
      })
    )
  );
  const tx = db.transaction(MESSAGE_STORE, "readwrite");
  const store = tx.objectStore(MESSAGE_STORE);
  for (const record of records) {
    store.put(record);
  }
  await transactionDone(tx);
}

export async function getCachedHostedChat<
  TChat extends { id: string; updated_at: number },
  TMessage
>(chatId: string): Promise<CachedHostedChat<TChat, TMessage> | null> {
  const db = await openPrivateDb();
  const tx = db.transaction(HOSTED_CHAT_CACHE_STORE, "readonly");
  const record = await requestResult<
    PrivateStoreRecord | CachedHostedChat<TChat, TMessage> | undefined
  >(tx.objectStore(HOSTED_CHAT_CACHE_STORE).get(chatId));
  await transactionDone(tx);
  return record ? readPrivateRecord<CachedHostedChat<TChat, TMessage>>(record) : null;
}

export async function saveCachedHostedChat<
  TChat extends { id: string; updated_at: number },
  TMessage
>({
  chat,
  active_root_message_id,
  messages
}: {
  chat: TChat;
  active_root_message_id: string | null;
  messages: TMessage[];
}): Promise<void> {
  const cache: CachedHostedChat<TChat, TMessage> = {
    id: chat.id,
    chat,
    active_root_message_id,
    messages,
    updated_at: chat.updated_at,
    cached_at: unixTimestamp()
  };
  const db = await openPrivateDb();
  const record = await privateStoreRecord(cache, {
    updated_at: cache.updated_at
  });
  const tx = db.transaction(HOSTED_CHAT_CACHE_STORE, "readwrite");
  tx.objectStore(HOSTED_CHAT_CACHE_STORE).put(record);
  await transactionDone(tx);
}

export async function getCachedModelState<T>(): Promise<T | null> {
  const db = await openPrivateDb();
  const tx = db.transaction(MODEL_CACHE_STORE, "readonly");
  const record = await requestResult<
    PrivateStoreRecord | { id: string; state: T; cached_at: number } | undefined
  >(tx.objectStore(MODEL_CACHE_STORE).get(MODEL_CACHE_ID));
  await transactionDone(tx);
  if (!record) {
    return null;
  }

  const cache = await readPrivateRecord<{ id: string; state: T; cached_at: number }>(record);
  return cache.state;
}

export async function saveCachedModelState<T>(state: T): Promise<void> {
  const db = await openPrivateDb();
  const record = await privateStoreRecord({
    id: MODEL_CACHE_ID,
    state,
    cached_at: unixTimestamp()
  });
  const tx = db.transaction(MODEL_CACHE_STORE, "readwrite");
  tx.objectStore(MODEL_CACHE_STORE).put(record);
  await transactionDone(tx);
}

export async function listPrivatePersonas(): Promise<PrivatePersona[]> {
  const db = await openPrivateDb();
  const [personas, versions] = await Promise.all([
    getAllPrivateRecords<Omit<PrivatePersona, "current_version">>(db, PERSONA_STORE),
    getAllPrivateRecords<PrivatePersonaVersion>(db, PERSONA_VERSION_STORE)
  ]);
  const versionsById = new Map(versions.map((version) => [version.id, version]));

  return personas
    .map((persona) => {
      const currentVersion = versionsById.get(persona.current_version_id);
      if (!currentVersion) {
        return null;
      }

      return {
        ...persona,
        current_version: currentVersion
      };
    })
    .filter((persona): persona is PrivatePersona => Boolean(persona))
    .sort((left, right) =>
      left.current_version.display_name.localeCompare(right.current_version.display_name)
    );
}

export async function createPrivatePersona(
  params: SavePrivatePersonaParams
): Promise<PrivatePersona> {
  const now = unixTimestamp();
  const personaId = privateId("private-persona");
  const versionId = privateId("private-persona-version");
  const version = privatePersonaVersionFromParams({
    params,
    personaId,
    versionId,
    versionNumber: 1,
    now
  });
  const persona = {
    id: personaId,
    current_version_id: versionId,
    created_at: now,
    updated_at: now
  };

  const db = await openPrivateDb();
  const personaRecord = await privateStoreRecord(persona, {
    created_at: persona.created_at,
    updated_at: persona.updated_at
  });
  const versionRecord = await privateStoreRecord(version, {
    persona_id: version.persona_id,
    created_at: version.created_at
  });
  const tx = db.transaction([PERSONA_STORE, PERSONA_VERSION_STORE], "readwrite");
  tx.objectStore(PERSONA_STORE).put(personaRecord);
  tx.objectStore(PERSONA_VERSION_STORE).put(versionRecord);
  await transactionDone(tx);

  return {
    ...persona,
    current_version: version
  };
}

export async function updatePrivatePersona(
  personaId: string,
  params: SavePrivatePersonaParams
): Promise<PrivatePersona> {
  const persona = await getPrivatePersona(personaId);
  if (!persona) {
    throw new Error("Private persona not found on this device");
  }

  const versions = await listPrivatePersonaVersions(personaId);
  const nextVersionNumber =
    versions.reduce((max, version) => Math.max(max, version.version_number), 0) + 1;
  const now = unixTimestamp();
  const versionId = privateId("private-persona-version");
  const version = privatePersonaVersionFromParams({
    params,
    personaId,
    versionId,
    versionNumber: nextVersionNumber,
    now
  });
  const updatedPersona = {
    id: persona.id,
    current_version_id: versionId,
    created_at: persona.created_at,
    updated_at: now
  };

  const db = await openPrivateDb();
  const personaRecord = await privateStoreRecord(updatedPersona, {
    created_at: updatedPersona.created_at,
    updated_at: updatedPersona.updated_at
  });
  const versionRecord = await privateStoreRecord(version, {
    persona_id: version.persona_id,
    created_at: version.created_at
  });
  const tx = db.transaction([PERSONA_STORE, PERSONA_VERSION_STORE], "readwrite");
  tx.objectStore(PERSONA_STORE).put(personaRecord);
  tx.objectStore(PERSONA_VERSION_STORE).put(versionRecord);
  await transactionDone(tx);

  return {
    ...updatedPersona,
    current_version: version
  };
}

export async function deletePrivatePersona(personaId: string): Promise<void> {
  const db = await openPrivateDb();
  const tx = db.transaction([PERSONA_STORE, PERSONA_VERSION_STORE], "readwrite");
  tx.objectStore(PERSONA_STORE).delete(personaId);

  const index = tx.objectStore(PERSONA_VERSION_STORE).index("persona_id");
  const cursorRequest = index.openCursor(IDBKeyRange.only(personaId));
  cursorRequest.onsuccess = () => {
    const cursor = cursorRequest.result;
    if (!cursor) {
      return;
    }

    cursor.delete();
    cursor.continue();
  };

  await transactionDone(tx);
}

export async function getPrivatePersona(personaId: string): Promise<PrivatePersona | null> {
  const db = await openPrivateDb();
  const tx = db.transaction([PERSONA_STORE, PERSONA_VERSION_STORE], "readonly");
  const personaRecord = await requestResult<
    PrivateStoreRecord | Omit<PrivatePersona, "current_version"> | undefined
  >(
    tx.objectStore(PERSONA_STORE).get(personaId)
  );
  if (!personaRecord) {
    await transactionDone(tx);
    return null;
  }
  const persona = await readPrivateRecord<Omit<PrivatePersona, "current_version">>(personaRecord);

  const versionRecord = await requestResult<PrivateStoreRecord | PrivatePersonaVersion | undefined>(
    tx.objectStore(PERSONA_VERSION_STORE).get(persona.current_version_id)
  );
  await transactionDone(tx);
  const version = versionRecord
    ? await readPrivateRecord<PrivatePersonaVersion>(versionRecord)
    : null;
  return version ? { ...persona, current_version: version } : null;
}

export async function listPrivatePersonaVersions(
  personaId: string
): Promise<PrivatePersonaVersion[]> {
  const db = await openPrivateDb();
  const tx = db.transaction(PERSONA_VERSION_STORE, "readonly");
  const records = await requestResult<Array<PrivateStoreRecord | PrivatePersonaVersion>>(
    tx.objectStore(PERSONA_VERSION_STORE).index("persona_id").getAll(personaId)
  );
  await transactionDone(tx);
  const versions = await Promise.all(
    records.map((record) => readPrivateRecord<PrivatePersonaVersion>(record))
  );
  return versions.sort((left, right) => left.version_number - right.version_number);
}

async function openPrivateDb(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(privateDbName(), DB_VERSION);
      request.onerror = () => reject(request.error ?? new Error("Failed to open private storage"));
      request.onupgradeneeded = () => {
        const db = request.result;
        ensurePrivateStores(db);
      };
      request.onsuccess = () => resolve(request.result);
    });
  }

  const db = await dbPromise;
  if (!legacyMigrationPromise) {
    legacyMigrationPromise = migrateLegacyPrivateDb(db).catch(() => undefined);
  }
  await legacyMigrationPromise;
  return db;
}

function privateDbName() {
  if (!currentUserId) {
    throw new Error("Private storage is not ready");
  }

  return `${DB_NAME_PREFIX}-${currentUserId}`;
}

function ensurePrivateStores(db: IDBDatabase) {
  if (!db.objectStoreNames.contains(CHAT_STORE)) {
    db.createObjectStore(CHAT_STORE, { keyPath: "id" });
  }
  if (!db.objectStoreNames.contains(MESSAGE_STORE)) {
    const messageStore = db.createObjectStore(MESSAGE_STORE, { keyPath: "id" });
    messageStore.createIndex("chat_id", "chat_id", { unique: false });
  }
  if (!db.objectStoreNames.contains(PERSONA_STORE)) {
    db.createObjectStore(PERSONA_STORE, { keyPath: "id" });
  }
  if (!db.objectStoreNames.contains(PERSONA_VERSION_STORE)) {
    const personaVersionStore = db.createObjectStore(PERSONA_VERSION_STORE, {
      keyPath: "id"
    });
    personaVersionStore.createIndex("persona_id", "persona_id", { unique: false });
  }
  if (!db.objectStoreNames.contains(HOSTED_CHAT_CACHE_STORE)) {
    db.createObjectStore(HOSTED_CHAT_CACHE_STORE, { keyPath: "id" });
  }
  if (!db.objectStoreNames.contains(MODEL_CACHE_STORE)) {
    db.createObjectStore(MODEL_CACHE_STORE, { keyPath: "id" });
  }
}

async function migrateLegacyPrivateDb(targetDb: IDBDatabase) {
  if (privateDbName() === LEGACY_DB_NAME) {
    return;
  }

  const targetChats = await getAllRaw(targetDb, CHAT_STORE);
  const targetMessages = await getAllRaw(targetDb, MESSAGE_STORE);
  const targetPersonas = await getAllRaw(targetDb, PERSONA_STORE);
  const targetVersions = await getAllRaw(targetDb, PERSONA_VERSION_STORE);
  if (
    targetChats.length > 0 ||
    targetMessages.length > 0 ||
    targetPersonas.length > 0 ||
    targetVersions.length > 0
  ) {
    return;
  }

  const legacyDb = await openDbByName(LEGACY_DB_NAME);
  try {
    const [chats, messages, personas, versions] = await Promise.all([
      getAllRaw(legacyDb, CHAT_STORE),
      getAllRaw(legacyDb, MESSAGE_STORE),
      getAllRaw(legacyDb, PERSONA_STORE),
      getAllRaw(legacyDb, PERSONA_VERSION_STORE)
    ]);
    if (
      chats.length === 0 &&
      messages.length === 0 &&
      personas.length === 0 &&
      versions.length === 0
    ) {
      return;
    }

    const chatRecords = await Promise.all(
      chats.map(async (record) => {
        const chat = await readPrivateRecord<PrivateChatDetail>(record);
        return privateStoreRecord(chat, {
          created_at: chat.created_at,
          updated_at: chat.updated_at,
          last_message_at: chat.last_message_at
        });
      })
    );
    const messageRecords = await Promise.all(
      messages.map(async (record) => {
        const message = await readPrivateRecord<PrivateChatMessage>(record);
        return privateStoreRecord(message, {
          chat_id: message.chat_id,
          created_at: message.created_at,
          updated_at: message.updated_at
        });
      })
    );
    const personaRecords = await Promise.all(
      personas.map(async (record) => {
        const persona = await readPrivateRecord<Omit<PrivatePersona, "current_version">>(record);
        return privateStoreRecord(persona, {
          created_at: persona.created_at,
          updated_at: persona.updated_at
        });
      })
    );
    const versionRecords = await Promise.all(
      versions.map(async (record) => {
        const version = await readPrivateRecord<PrivatePersonaVersion>(record);
        return privateStoreRecord(version, {
          persona_id: version.persona_id,
          created_at: version.created_at
        });
      })
    );

    const tx = targetDb.transaction(
      [CHAT_STORE, MESSAGE_STORE, PERSONA_STORE, PERSONA_VERSION_STORE],
      "readwrite"
    );
    for (const record of chatRecords) {
      tx.objectStore(CHAT_STORE).put(record);
    }
    for (const record of messageRecords) {
      tx.objectStore(MESSAGE_STORE).put(record);
    }
    for (const record of personaRecords) {
      tx.objectStore(PERSONA_STORE).put(record);
    }
    for (const record of versionRecords) {
      tx.objectStore(PERSONA_VERSION_STORE).put(record);
    }
    await transactionDone(tx);
  } finally {
    legacyDb.close();
  }

  await deleteDbByName(LEGACY_DB_NAME).catch(() => undefined);
}

function openDbByName(name: string): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name, DB_VERSION);
    request.onerror = () => reject(request.error ?? new Error("Failed to open private storage"));
    request.onupgradeneeded = () => {
      ensurePrivateStores(request.result);
    };
    request.onsuccess = () => resolve(request.result);
  });
}

function deleteDbByName(name: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.deleteDatabase(name);
    request.onerror = () => reject(request.error ?? new Error("Failed to delete private storage"));
    request.onsuccess = () => resolve();
    request.onblocked = () => resolve();
  });
}

async function getAllPrivateRecords<T>(db: IDBDatabase, storeName: string): Promise<T[]> {
  const records = await getAllRaw(db, storeName);
  return Promise.all(records.map((record) => readPrivateRecord<T>(record)));
}

async function getAllRaw(db: IDBDatabase, storeName: string): Promise<Array<PrivateStoreRecord | any>> {
  const tx = db.transaction(storeName, "readonly");
  const values = await requestResult<Array<PrivateStoreRecord | any>>(tx.objectStore(storeName).getAll());
  await transactionDone(tx);
  return values;
}

async function privateStoreRecord<T extends { id: string }>(
  value: T,
  metadata: Omit<PrivateStoreRecord, "id" | "encrypted_payload"> = {}
): Promise<PrivateStoreRecord> {
  return {
    id: value.id,
    ...metadata,
    encrypted_payload: await encryptJson(value)
  };
}

async function readPrivateRecord<T>(record: PrivateStoreRecord | T): Promise<T> {
  if (hasEncryptedPayload(record)) {
    return decryptJson<T>(record.encrypted_payload);
  }

  return record as T;
}

function hasEncryptedPayload(record: unknown): record is PrivateStoreRecord & { encrypted_payload: EncryptedPayload } {
  return (
    typeof record === "object" &&
    record !== null &&
    "encrypted_payload" in record &&
    typeof (record as PrivateStoreRecord).encrypted_payload?.data === "string"
  );
}

async function encryptJson(value: unknown): Promise<EncryptedPayload> {
  const iv = randomBytes(12);
  const encoded = new TextEncoder().encode(JSON.stringify(value));
  const encrypted = await encryptBytes(encoded, iv);

  return {
    v: 1,
    iv: bytesToBase64(iv),
    data: bytesToBase64(new Uint8Array(encrypted))
  };
}

async function decryptJson<T>(payload: EncryptedPayload): Promise<T> {
  const iv = base64ToBytes(payload.iv);
  const encrypted = base64ToBytes(payload.data);
  const decrypted = await decryptBytes(encrypted, iv);
  return JSON.parse(new TextDecoder().decode(decrypted)) as T;
}

async function encryptBytes(plaintext: Uint8Array, iv: Uint8Array): Promise<Uint8Array> {
  if (globalThis.crypto?.subtle) {
    const key = await privateWebCryptoKey();
    const encrypted = await globalThis.crypto.subtle.encrypt(
      { name: "AES-GCM", iv: bufferSource(iv) },
      key,
      bufferSource(plaintext)
    );
    return new Uint8Array(encrypted);
  }

  const key = await privateVaultKeyBytes();
  return aesGcm(key, iv).encrypt(plaintext);
}

async function decryptBytes(encrypted: Uint8Array, iv: Uint8Array): Promise<Uint8Array> {
  if (globalThis.crypto?.subtle) {
    const key = await privateWebCryptoKey();
    const decrypted = await globalThis.crypto.subtle.decrypt(
      { name: "AES-GCM", iv: bufferSource(iv) },
      key,
      bufferSource(encrypted)
    );
    return new Uint8Array(decrypted);
  }

  const key = await privateVaultKeyBytes();
  return aesGcm(key, iv).decrypt(encrypted);
}

async function privateWebCryptoKey(): Promise<CryptoKey> {
  if (!webCryptoKeyPromise) {
    webCryptoKeyPromise = privateVaultKeyBytes().then((keyBytes) =>
      globalThis.crypto.subtle.importKey(
        "raw",
        bufferSource(keyBytes),
        "AES-GCM",
        false,
        ["encrypt", "decrypt"]
      )
    );
  }

  return webCryptoKeyPromise;
}

async function privateVaultKeyBytes(): Promise<Uint8Array> {
  if (!vaultKeyBytesPromise) {
    vaultKeyBytesPromise = privateVaultKey().then((vaultKey) =>
      base64ToBytes(vaultKey.key_material)
    );
  }

  return vaultKeyBytesPromise;
}

async function privateVaultKey(): Promise<PrivateVaultKeyResponse> {
  if (!vaultKeyPromise) {
    vaultKeyPromise = fetch("/api/private/vault-key", { credentials: "include" }).then(
      async (response) => {
        if (!response.ok) {
          throw new Error(await responseErrorMessage(response));
        }

        const vaultKey = (await response.json()) as PrivateVaultKeyResponse;
        if (currentUserId && vaultKey.user_id !== currentUserId) {
          throw new Error("Private storage user changed");
        }

        return vaultKey;
      }
    );
  }

  return vaultKeyPromise;
}

async function responseErrorMessage(response: Response) {
  try {
    const body = (await response.json()) as { error?: { message?: string } };
    return body.error?.message ?? `Request failed with status ${response.status}`;
  } catch {
    return `Request failed with status ${response.status}`;
  }
}

function bytesToBase64(bytes: Uint8Array) {
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }

  return btoa(binary);
}

function randomBytes(length: number) {
  if (!globalThis.crypto?.getRandomValues) {
    throw new Error("Private chats are unavailable in this browser context.");
  }

  return globalThis.crypto.getRandomValues(new Uint8Array(length));
}

function bufferSource(bytes: Uint8Array): BufferSource {
  return bytes as unknown as BufferSource;
}

function base64ToBytes(value: string) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return bytes;
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB transaction aborted"));
    transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB transaction failed"));
  });
}

function compareMessagesByCreatedAt(left: PrivateChatMessage, right: PrivateChatMessage) {
  return left.created_at - right.created_at || left.id.localeCompare(right.id);
}

function normalizePrivateMessage(message: PrivateChatMessage): PrivateChatMessage {
  return {
    ...message,
    stats: message.stats ?? null,
    attachments: message.attachments ?? []
  };
}

function privatePersonaVersionFromParams({
  params,
  personaId,
  versionId,
  versionNumber,
  now
}: {
  params: SavePrivatePersonaParams;
  personaId: string;
  versionId: string;
  versionNumber: number;
  now: number;
}): PrivatePersonaVersion {
  return {
    id: versionId,
    persona_id: personaId,
    version_number: versionNumber,
    display_name: params.displayName.trim(),
    avatar_attachment_id: params.avatarAttachmentId ?? null,
    base_backend_id: params.baseBackendId,
    base_backend_name: params.baseBackendName,
    base_model_name: params.baseModelName,
    system_prompt: params.systemPrompt.trim(),
    tool_policy_json: params.toolPolicyJson ?? null,
    source_persona_id: params.sourcePersonaId ?? null,
    source_persona_version_id: params.sourcePersonaVersionId ?? null,
    created_at: now
  };
}
