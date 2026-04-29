const DB_NAME = "vashti-private-local";
const DB_VERSION = 1;
const CHAT_STORE = "private_chats";
const MESSAGE_STORE = "private_messages";

export type PrivateChatSummary = {
  id: string;
  title: string;
  default_backend_id: string;
  backend_name: string;
  default_model_name: string;
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
  thinkMode?: string | null;
  createdAt?: number;
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
  modelName
}: CreatePrivateChatParams): Promise<PrivateChatDetail> {
  const now = unixTimestamp();
  const chat: PrivateChatDetail = {
    id: privateId("private-chat"),
    title,
    default_backend_id: backendId,
    backend_name: backendName,
    default_model_name: modelName,
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
