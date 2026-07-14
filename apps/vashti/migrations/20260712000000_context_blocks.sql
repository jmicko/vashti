CREATE TABLE context_categories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    selection_mode TEXT NOT NULL DEFAULT 'single'
        CHECK (selection_mode IN ('single', 'multiple')),
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE UNIQUE INDEX idx_context_categories_user_name
    ON context_categories(user_id, name COLLATE NOCASE);
CREATE INDEX idx_context_categories_user_sort
    ON context_categories(user_id, sort_order, created_at);

CREATE TABLE context_blocks (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category_id TEXT REFERENCES context_categories(id) ON DELETE SET NULL,
    current_version_id TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    deleted_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE context_block_versions (
    id TEXT PRIMARY KEY,
    block_id TEXT NOT NULL REFERENCES context_blocks(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(block_id, version_number)
);

CREATE INDEX idx_context_blocks_user_category
    ON context_blocks(user_id, category_id, sort_order, created_at);
CREATE INDEX idx_context_block_versions_block
    ON context_block_versions(block_id, version_number);

CREATE TABLE chat_context_blocks (
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    block_version_id TEXT NOT NULL REFERENCES context_block_versions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    PRIMARY KEY (chat_id, block_version_id),
    UNIQUE (chat_id, position)
);

CREATE INDEX idx_chat_context_blocks_chat_position
    ON chat_context_blocks(chat_id, position);

CREATE TABLE chat_message_context_blocks (
    message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    block_version_id TEXT NOT NULL REFERENCES context_block_versions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    PRIMARY KEY (message_id, block_version_id),
    UNIQUE (message_id, position)
);

CREATE INDEX idx_chat_message_context_blocks_message_position
    ON chat_message_context_blocks(message_id, position);
