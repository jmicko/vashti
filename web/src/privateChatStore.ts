const DB_NAME = "vashti-private-local";
const DB_VERSION = 2;
const CHAT_STORE = "private_chats";
const MESSAGE_STORE = "private_messages";
const PERSONA_STORE = "private_personas";
const PERSONA_VERSION_STORE = "private_persona_versions";

export type PrivateChatSummary = {
  id: string;
  title: string;
  default_backend_id: string;
  backend_name: string;
  default_model_name: string;
  persona_id?: string | null;
  persona_version_id?: string | null;
  persona_name?: string | null;
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

let dbPromise: Promise<IDBDatabase> | null = null;

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
    getAll<PrivateChatDetail>(db, CHAT_STORE),
    getAll<PrivateChatMessage>(db, MESSAGE_STORE)
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
  personaName = null
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
  const chat = await requestResult<PrivateChatDetail | undefined>(
    tx.objectStore(CHAT_STORE).get(chatId)
  );
  await transactionDone(tx);
  return chat ?? null;
}

export async function savePrivateChat(chat: PrivateChatDetail): Promise<void> {
  const db = await openPrivateDb();
  const tx = db.transaction(CHAT_STORE, "readwrite");
  tx.objectStore(CHAT_STORE).put(chat);
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
  const messages = await requestResult<PrivateChatMessage[]>(
    tx.objectStore(MESSAGE_STORE).index("chat_id").getAll(chatId)
  );
  await transactionDone(tx);
  return messages.map(normalizePrivateMessage).sort(compareMessagesByCreatedAt);
}

export async function savePrivateMessage(message: PrivateChatMessage): Promise<void> {
  const db = await openPrivateDb();
  const tx = db.transaction(MESSAGE_STORE, "readwrite");
  tx.objectStore(MESSAGE_STORE).put(message);
  await transactionDone(tx);
}

export async function savePrivateMessages(messages: PrivateChatMessage[]): Promise<void> {
  if (messages.length === 0) {
    return;
  }

  const db = await openPrivateDb();
  const tx = db.transaction(MESSAGE_STORE, "readwrite");
  const store = tx.objectStore(MESSAGE_STORE);
  for (const message of messages) {
    store.put(message);
  }
  await transactionDone(tx);
}

export async function listPrivatePersonas(): Promise<PrivatePersona[]> {
  const db = await openPrivateDb();
  const [personas, versions] = await Promise.all([
    getAll<Omit<PrivatePersona, "current_version">>(db, PERSONA_STORE),
    getAll<PrivatePersonaVersion>(db, PERSONA_VERSION_STORE)
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
  const tx = db.transaction([PERSONA_STORE, PERSONA_VERSION_STORE], "readwrite");
  tx.objectStore(PERSONA_STORE).put(persona);
  tx.objectStore(PERSONA_VERSION_STORE).put(version);
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
  const tx = db.transaction([PERSONA_STORE, PERSONA_VERSION_STORE], "readwrite");
  tx.objectStore(PERSONA_STORE).put(updatedPersona);
  tx.objectStore(PERSONA_VERSION_STORE).put(version);
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
  const persona = await requestResult<Omit<PrivatePersona, "current_version"> | undefined>(
    tx.objectStore(PERSONA_STORE).get(personaId)
  );
  if (!persona) {
    await transactionDone(tx);
    return null;
  }

  const version = await requestResult<PrivatePersonaVersion | undefined>(
    tx.objectStore(PERSONA_VERSION_STORE).get(persona.current_version_id)
  );
  await transactionDone(tx);
  return version ? { ...persona, current_version: version } : null;
}

async function listPrivatePersonaVersions(personaId: string): Promise<PrivatePersonaVersion[]> {
  const db = await openPrivateDb();
  const tx = db.transaction(PERSONA_VERSION_STORE, "readonly");
  const versions = await requestResult<PrivatePersonaVersion[]>(
    tx.objectStore(PERSONA_VERSION_STORE).index("persona_id").getAll(personaId)
  );
  await transactionDone(tx);
  return versions.sort((left, right) => left.version_number - right.version_number);
}

async function openPrivateDb(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onerror = () => reject(request.error ?? new Error("Failed to open private storage"));
      request.onupgradeneeded = () => {
        const db = request.result;
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
      };
      request.onsuccess = () => resolve(request.result);
    });
  }

  return dbPromise;
}

async function getAll<T>(db: IDBDatabase, storeName: string): Promise<T[]> {
  const tx = db.transaction(storeName, "readonly");
  const values = await requestResult<T[]>(tx.objectStore(storeName).getAll());
  await transactionDone(tx);
  return values;
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
