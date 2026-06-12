import {
  PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import {
  Copy,
  Lock,
  MessageSquare,
  MoreHorizontal,
  Pencil,
  Trash2,
  X
} from "lucide-react";
import { BrandMark } from "./common";
import { ModelAvatar } from "./ModelAvatar";
import type {
  PrivateChatSummary,
  PrivatePersonaVersion
} from "./privateChatStore";
import type { ChatSummary, Page, PersonaVersion } from "./types";

export function Sidebar({
  chats,
  privateChats,
  currentChatId,
  currentPrivateChatId,
  currentPage,
  isOpen,
  isLoading,
  isLoadingPrivateChats,
  personaVersions,
  privatePersonaVersions,
  onClose,
  onDeleteChat,
  onDeletePrivateChat,
  onOpenChat,
  onOpenPrivateChat,
  onRenameChat,
  onRenamePrivateChat
}: {
  chats: ChatSummary[];
  privateChats: PrivateChatSummary[];
  currentChatId: string | null;
  currentPrivateChatId: string | null;
  currentPage: Page;
  isOpen: boolean;
  isLoading: boolean;
  isLoadingPrivateChats: boolean;
  personaVersions: PersonaVersion[];
  privatePersonaVersions: PrivatePersonaVersion[];
  onClose: () => void;
  onDeleteChat: (chat: ChatSummary) => void;
  onDeletePrivateChat: (chat: PrivateChatSummary) => void;
  onOpenChat: (chatId?: string) => void;
  onOpenPrivateChat: (chatId: string) => void;
  onRenameChat: (chatId: string, title: string) => Promise<void>;
  onRenamePrivateChat: (chatId: string, title: string) => Promise<void>;
}) {
  const [openMenuChatId, setOpenMenuChatId] = useState<string | null>(null);
  const [editingChatId, setEditingChatId] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      setOpenMenuChatId(null);
    }
  }, [isOpen]);

  useEffect(() => {
    if (!openMenuChatId) {
      return;
    }

    function closeOnOutsidePointer(event: PointerEvent) {
      const target = event.target;
      if (
        target instanceof Element &&
        target.closest(".chat-row-menu, .chat-menu-button")
      ) {
        return;
      }

      setOpenMenuChatId(null);
    }

    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpenMenuChatId(null);
      }
    }

    window.addEventListener("pointerdown", closeOnOutsidePointer);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("pointerdown", closeOnOutsidePointer);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [openMenuChatId]);

  const combinedChats = useMemo(
    () =>
      [
        ...chats.map((chat) => ({ chat, isPrivate: false })),
        ...privateChats.map((chat) => ({ chat, isPrivate: true }))
      ].sort(
        (left, right) =>
          right.chat.last_message_at - left.chat.last_message_at ||
          left.chat.title.localeCompare(right.chat.title)
      ),
    [chats, privateChats]
  );
  const isLoadingAnyChats = isLoading || isLoadingPrivateChats;

  return (
    <aside className="sidebar">
      <div className="sidebar-static">
        <div className="sidebar-header">
          <BrandMark compact />
          <button
            type="button"
            className="icon-button mobile-only"
            aria-label="Close sidebar"
            onClick={() => {
              setOpenMenuChatId(null);
              onClose();
            }}
          >
            <X />
          </button>
        </div>
        <button
          type="button"
          className={
            currentPage === "chat" && !currentChatId
              ? "nav-button nav-button-active"
              : "nav-button"
          }
          onClick={() => {
            setOpenMenuChatId(null);
            onOpenChat();
          }}
        >
          <MessageSquare />
          <span>Chats</span>
        </button>
      </div>
      <div className="chat-history">
        <p className="eyebrow">Previous Chats</p>
        {isLoadingAnyChats && combinedChats.length === 0 ? (
          <p>Loading chats...</p>
        ) : combinedChats.length === 0 ? (
          <p>No chats yet</p>
        ) : (
          <div className="chat-link-list">
            {combinedChats.map(({ chat, isPrivate }) => {
              const version = chat.persona_version_id
                ? (isPrivate ? privatePersonaVersions : personaVersions).find(
                    (candidate) => candidate.id === chat.persona_version_id
                  )
                : null;
              return (
                <ChatListItem
                key={chat.id}
                chat={chat}
                avatar={
                  version
                    ? {
                        displayName: version.display_name,
                        assetId: isPrivate ? null : version.avatar_asset_id,
                        privateAssetId: isPrivate ? version.avatar_asset_id : null,
                        cropX: version.avatar_crop_x,
                        cropY: version.avatar_crop_y,
                        cropSize: version.avatar_crop_size
                      }
                    : null
                }
                isActive={
                  isPrivate ? currentPrivateChatId === chat.id : currentChatId === chat.id
                }
                isEditing={editingChatId === chat.id}
                isMenuOpen={openMenuChatId === chat.id}
                isPrivate={isPrivate}
                onCancelEditing={() => setEditingChatId(null)}
                onCloseMenu={() => setOpenMenuChatId(null)}
                onDelete={() => {
                  setOpenMenuChatId(null);
                  if (isPrivate) {
                    onDeletePrivateChat(chat as PrivateChatSummary);
                  } else {
                    onDeleteChat(chat as ChatSummary);
                  }
                }}
                onOpen={() => {
                  setOpenMenuChatId(null);
                  if (isPrivate) {
                    onOpenPrivateChat(chat.id);
                  } else {
                    onOpenChat(chat.id);
                  }
                }}
                onOpenMenu={() => setOpenMenuChatId(chat.id)}
                onRename={(title) =>
                  isPrivate
                    ? onRenamePrivateChat(chat.id, title)
                    : onRenameChat(chat.id, title)
                }
                onStartEditing={() => {
                  setOpenMenuChatId(null);
                  setEditingChatId(chat.id);
                }}
                onToggleMenu={() =>
                  setOpenMenuChatId((current) => (current === chat.id ? null : chat.id))
                }
                />
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}

function ChatListItem({
  chat,
  avatar,
  isActive,
  isEditing,
  isMenuOpen,
  isPrivate = false,
  onCancelEditing,
  onCloseMenu,
  onDelete,
  onOpen,
  onOpenMenu,
  onRename,
  onStartEditing,
  onToggleMenu
}: {
  chat: ChatSummary | PrivateChatSummary;
  avatar: {
    displayName: string;
    assetId: string | null;
    privateAssetId: string | null;
    cropX: number;
    cropY: number;
    cropSize: number;
  } | null;
  isActive: boolean;
  isEditing: boolean;
  isMenuOpen: boolean;
  isPrivate?: boolean;
  onCancelEditing: () => void;
  onCloseMenu: () => void;
  onDelete: () => void;
  onOpen: () => void;
  onOpenMenu: () => void;
  onRename: (title: string) => Promise<void>;
  onStartEditing: () => void;
  onToggleMenu: () => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const longPressTimerRef = useRef<number | null>(null);
  const suppressClickRef = useRef(false);
  const [draftTitle, setDraftTitle] = useState(chat.title);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (isEditing) {
      setDraftTitle(chat.title);
      window.requestAnimationFrame(() => {
        inputRef.current?.focus();
        inputRef.current?.select();
      });
    }
  }, [chat.title, isEditing]);

  async function finishRename() {
    const title = draftTitle.trim();
    if (!title || title === chat.title) {
      onCancelEditing();
      return;
    }

    setIsSaving(true);
    try {
      await onRename(title);
      onCancelEditing();
    } finally {
      setIsSaving(false);
    }
  }

  function cancelLongPress() {
    if (longPressTimerRef.current) {
      window.clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }
  }

  function startLongPress(event: ReactPointerEvent<HTMLDivElement>) {
    if (isEditing || event.pointerType === "mouse") {
      return;
    }

    cancelLongPress();
    suppressClickRef.current = false;
    longPressTimerRef.current = window.setTimeout(() => {
      suppressClickRef.current = true;
      onOpenMenu();
    }, 520);
  }

  async function copyTitle() {
    await navigator.clipboard.writeText(chat.title);
    onCloseMenu();
  }

  return (
    <div
      className={isActive ? "chat-link-row chat-link-row-active" : "chat-link-row"}
      onContextMenu={(event) => {
        if (!isEditing) {
          event.preventDefault();
          onOpenMenu();
        }
      }}
      onPointerCancel={cancelLongPress}
      onPointerDown={startLongPress}
      onPointerLeave={cancelLongPress}
      onPointerUp={cancelLongPress}
    >
      {isEditing ? (
        <input
          ref={inputRef}
          className="chat-title-input"
          disabled={isSaving}
          value={draftTitle}
          onBlur={() => void finishRename()}
          onChange={(event) => setDraftTitle(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              event.currentTarget.blur();
            }
            if (event.key === "Escape") {
              onCancelEditing();
            }
          }}
        />
      ) : (
        <button
          type="button"
          className={isActive ? "chat-link chat-link-active" : "chat-link"}
          onClick={(event) => {
            if (suppressClickRef.current) {
              event.preventDefault();
              suppressClickRef.current = false;
              return;
            }
            onOpen();
          }}
        >
          {avatar && (
            <ModelAvatar
              displayName={avatar.displayName}
              assetId={avatar.assetId}
              privateAssetId={avatar.privateAssetId}
              cropX={avatar.cropX}
              cropY={avatar.cropY}
              cropSize={avatar.cropSize}
              className="model-avatar-chat-list"
            />
          )}
          <span className="chat-link-copy">
          <span className="chat-title-line">
            {isPrivate && <Lock />}
            <span>{chat.title}</span>
          </span>
          <small>
            {isPrivate
              ? `Private · ${"persona_name" in chat && chat.persona_name ? chat.persona_name : chat.default_model_name}`
              : "persona_name" in chat && chat.persona_name
                ? `Custom · ${chat.persona_name}`
                : chat.default_model_name}
          </small>
          </span>
        </button>
      )}
      <button
        type="button"
        className="chat-menu-button"
        aria-label={`Open menu for ${chat.title}`}
        aria-expanded={isMenuOpen}
        onPointerDown={(event) => event.stopPropagation()}
        onClick={onToggleMenu}
      >
        <MoreHorizontal />
      </button>
      {isMenuOpen && (
        <div className="chat-row-menu" onPointerDown={(event) => event.stopPropagation()}>
          <button
            type="button"
            className="menu-item"
            onClick={onStartEditing}
          >
            <Pencil />
            <span>Rename</span>
          </button>
          <button
            type="button"
            className="menu-item"
            onClick={() => void copyTitle()}
          >
            <Copy />
            <span>Copy Title</span>
          </button>
          <button
            type="button"
            className="menu-item danger-button"
            onClick={() => {
              onCloseMenu();
              onDelete();
            }}
          >
            <Trash2 />
            <span>Delete</span>
          </button>
        </div>
      )}
    </div>
  );
}
