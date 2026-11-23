import {
  PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import {
  Archive,
  ArrowLeft,
  Undo2,
  Copy,
  ChevronDown,
  Bot,
  Film,
  Lock,
  MessageSquarePlus,
  MessageSquare,
  Search,
  RefreshCw,
  MoreHorizontal,
  NotebookPen,
  Images,
  Pencil,
  Trash2,
  X
} from "lucide-react";
import { BrandMark } from "./common";
import { ModelAvatar } from "./ModelAvatar";
import { ImageFlowHistory } from "./images/ImageFlowHistory";
import type {
  PrivateChatSummary,
  PrivatePersonaVersion
} from "./privateChatStore";
import type { BackendModelGroup, ChatSummary, PersonaVersion } from "./types";
import { VideoFlowHistory } from "./videos/VideoFlowHistory";

export function Sidebar({
  chats,
  privateChats,
  currentChatId,
  currentPrivateChatId,
  isNotesActive,
  isImagesActive,
  isVideosActive = false,
  isAssistantActive = false,
  assistantAttention = 0,
  currentImageFlowId,
  imageStorageMode,
  onOpenImageFlow,
  onRemoveImageFlow,
  isOpen,
  isLoading,
  isLoadingPrivateChats,
  showingArchived,
  historyError,
  onToggleArchived,
  onArchiveChat,
  onRetryHistory,
  personaVersions,
  privatePersonaVersions,
  modelGroups,
  onClose,
  onDeleteChat,
  onDeletePrivateChat,
  onNewChat,
  onOpenNotes,
  onOpenImages,
  onOpenVideos,
  onOpenAssistant,
  onOpenChat,
  onOpenPrivateChat,
  onRenameChat,
  onRenamePrivateChat
}: {
  chats: ChatSummary[];
  privateChats: PrivateChatSummary[];
  currentChatId: string | null;
  currentPrivateChatId: string | null;
  isNotesActive: boolean;
  isImagesActive: boolean;
  isVideosActive?: boolean;
  isAssistantActive?: boolean;
  assistantAttention?: number;
  currentImageFlowId?: string;
  imageStorageMode: "server" | "device";
  onOpenImageFlow: (id: string) => void;
  onRemoveImageFlow: (id: string) => void;
  isOpen: boolean;
  isLoading: boolean;
  isLoadingPrivateChats: boolean;
  showingArchived: boolean;
  historyError: string | null;
  onToggleArchived: () => void;
  onArchiveChat: (chat: ChatSummary | PrivateChatSummary, isPrivate: boolean, archived: boolean) => Promise<void>;
  onRetryHistory: () => void;
  personaVersions: PersonaVersion[];
  privatePersonaVersions: PrivatePersonaVersion[];
  modelGroups: BackendModelGroup[];
  onClose: () => void;
  onDeleteChat: (chat: ChatSummary) => void;
  onDeletePrivateChat: (chat: PrivateChatSummary) => void;
  onNewChat: () => void;
  onOpenNotes: () => void;
  onOpenImages: () => void;
  onOpenVideos?: () => void;
  onOpenAssistant?: () => void;
  onOpenChat: (chatId?: string) => void;
  onOpenPrivateChat: (chatId: string) => void;
  onRenameChat: (chatId: string, title: string) => Promise<void>;
  onRenamePrivateChat: (chatId: string, title: string) => Promise<void>;
}) {
  const [openMenuChatId, setOpenMenuChatId] = useState<string | null>(null);
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [visibleCount, setVisibleCount] = useState(100);
  useEffect(() => setVisibleCount(100), [search, showingArchived]);
  const [narrow, setNarrow] = useState(() => window.matchMedia("(max-width: 720px)").matches);
  useEffect(() => {
    const query = window.matchMedia("(max-width: 720px)");
    const change = () => setNarrow(query.matches);
    query.addEventListener("change", change);
    return () => query.removeEventListener("change", change);
  }, []);

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
        ...privateChats.filter((chat) => (chat.archived_at != null) === showingArchived)
          .map((chat) => ({ chat, isPrivate: true }))
      ].sort(
        (left, right) =>
          right.chat.last_message_at - left.chat.last_message_at ||
          left.chat.title.localeCompare(right.chat.title)
      ),
    [chats, privateChats, showingArchived]
  );
  const isLoadingAnyChats = isLoading || isLoadingPrivateChats;
  const filteredChats = combinedChats.filter(({ chat }) => chat.title.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()));

  return (
    <aside className="sidebar" inert={narrow && !isOpen ? true : undefined} aria-label="Navigation">
      <div className="sidebar-static">
        <div className="sidebar-header">
          <button
            type="button"
            className="sidebar-brand-button"
            aria-label="Go to home"
            title="Home"
            onClick={() => {
              setOpenMenuChatId(null);
              onOpenChat();
            }}
          >
            <BrandMark compact />
          </button>
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
        <nav className="sidebar-primary-nav" aria-label="Workspace">
          <button
            type="button"
            className="primary-action sidebar-new-chat"
            onClick={() => {
              setOpenMenuChatId(null);
              onNewChat();
            }}
          >
            <MessageSquarePlus />
            <span>{isImagesActive || isVideosActive ? "New Flow" : "New Chat"}</span>
          </button>
          {(isImagesActive || isVideosActive || isNotesActive || isAssistantActive) && <button type="button" className="sidebar-section-button" onClick={() => onOpenChat()}><MessageSquare /><span>Chat</span></button>}
          {onOpenAssistant && <button type="button" className={isAssistantActive ? "sidebar-section-button sidebar-section-button-active" : "sidebar-section-button"} aria-current={isAssistantActive ? "page" : undefined} aria-label={assistantAttention ? `Assistant, ${assistantAttention === 1 ? "1 item needs" : `${assistantAttention} items need`} attention` : undefined} onClick={onOpenAssistant}><Bot /><span>Assistant</span>{assistantAttention > 0 && <small className="assistant-attention-count" aria-hidden="true">{assistantAttention > 99 ? "99+" : assistantAttention}</small>}</button>}
          <button type="button" className={isImagesActive ? "sidebar-section-button sidebar-section-button-active" : "sidebar-section-button"} aria-current={isImagesActive ? "page" : undefined} onClick={() => { setOpenMenuChatId(null); onOpenImages(); }}><Images /><span>Images</span></button>
          {onOpenVideos && (
            <button
              type="button"
              className={isVideosActive ? "sidebar-section-button sidebar-section-button-active" : "sidebar-section-button"}
              aria-current={isVideosActive ? "page" : undefined}
              onClick={() => { setOpenMenuChatId(null); onOpenVideos(); }}
            >
              <Film /><span>Video</span>
            </button>
          )}
          <button
            type="button"
            className={isNotesActive ? "sidebar-section-button sidebar-section-button-active" : "sidebar-section-button"}
            aria-current={isNotesActive ? "page" : undefined}
            onClick={() => {
              setOpenMenuChatId(null);
              onOpenNotes();
            }}
          >
            <NotebookPen />
            <span>Notes</span>
          </button>
        </nav>
      </div>
      <div className="chat-history" hidden={isAssistantActive}>
        {isVideosActive ? <VideoFlowHistory key={imageStorageMode} storageMode={imageStorageMode} currentId={currentImageFlowId} onOpen={onOpenImageFlow} onRemoved={onRemoveImageFlow}/> : isImagesActive ? <ImageFlowHistory key={imageStorageMode} storageMode={imageStorageMode} currentId={currentImageFlowId} onOpen={onOpenImageFlow} onRemoved={onRemoveImageFlow} /> : <>
        <div className="history-heading">
          <p className="eyebrow">{showingArchived ? "Archived Chats" : "Previous Chats"}</p>
          <button type="button" className="icon-button" aria-pressed={showingArchived}
            aria-label={showingArchived ? "Show previous chats" : "Show archived chats"}
            title={showingArchived ? "Show previous chats" : "Show archived chats"}
            onClick={() => { setOpenMenuChatId(null); setEditingChatId(null); setSearch(""); onToggleArchived(); }}>
            {showingArchived ? <ArrowLeft /> : <Archive />}
          </button>
        </div>
        {historyError && <div className="history-error" role="alert"><span>{historyError}</span><button type="button" className="icon-button" aria-label="Retry loading chats" title="Retry loading chats" onClick={onRetryHistory}><RefreshCw /></button></div>}
        {(showingArchived || combinedChats.length > 5 || search) && <label className="history-search"><Search size={16} /><input aria-label="Search chats" placeholder="Search chats" value={search} onChange={(event) => setSearch(event.target.value)} /></label>}
        {isLoadingAnyChats && combinedChats.length === 0 ? (
          <p>Loading chats...</p>
        ) : historyError && combinedChats.length === 0 ? null : combinedChats.length === 0 ? (
          <p>{showingArchived ? "No archived chats" : "No chats yet"}</p>
        ) : (
          <div className="chat-link-list">
            {!filteredChats.length && <p className="history-empty" role="status">No matching chats</p>}
            {filteredChats.slice(0, visibleCount).map(({ chat, isPrivate }) => {
              const version = chat.persona_version_id
                ? (isPrivate ? privatePersonaVersions : personaVersions).find(
                    (candidate) => candidate.id === chat.persona_version_id
                  )
                : null;
              const model = !version
                ? modelGroups
                    .find((group) => group.backend.id === chat.default_backend_id)
                    ?.models.find((candidate) => candidate.name === chat.default_model_name)
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
                    : model
                      ? {
                          displayName: model.name,
                          assetId: model.avatar_asset_id,
                          privateAssetId: null,
                          cropX: model.avatar_crop_x,
                          cropY: model.avatar_crop_y,
                          cropSize: model.avatar_crop_size
                        }
                      : null
                }
                isActive={
                  isPrivate ? currentPrivateChatId === chat.id : currentChatId === chat.id
                }
                isEditing={editingChatId === chat.id}
                isMenuOpen={openMenuChatId === chat.id}
                isPrivate={isPrivate}
                isArchived={showingArchived}
                onArchive={() => onArchiveChat(chat, isPrivate, !showingArchived)}
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
            {filteredChats.length > visibleCount && <button type="button" className="secondary-button" onClick={() => setVisibleCount(count => count + 100)}><ChevronDown /><span>Show more chats</span></button>}
          </div>
        )}
        </>}
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
  isArchived,
  onArchive,
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
  isArchived: boolean;
  onArchive: () => Promise<void>;
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
  const [isArchiving, setIsArchiving] = useState(false);

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
            disabled={isArchiving}
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
          <button type="button" className="menu-item" disabled={isArchiving}
            onClick={() => {
              setIsArchiving(true);
              void onArchive().then(onCloseMenu).catch(() => undefined).finally(() => setIsArchiving(false));
            }}>
            {isArchived ? <Undo2 /> : <Archive />}
            <span>{isArchiving ? (isArchived ? "Restoring..." : "Archiving...") : isArchived ? "Restore" : "Archive"}</span>
          </button>
          <button
            type="button"
            className="menu-item danger-button"
            disabled={isArchiving}
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
