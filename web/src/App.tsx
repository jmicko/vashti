import {
  FormEvent,
  isValidElement,
  PointerEvent as ReactPointerEvent,
  ReactNode,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import {
  ChevronLeft,
  ChevronRight,
  Cog,
  Copy,
  LogOut,
  Menu,
  MessageSquare,
  MessageSquarePlus,
  MoreHorizontal,
  Pencil,
  Plus,
  Power,
  RefreshCw,
  Server,
  Save,
  Search,
  SendHorizontal,
  Settings as SettingsIcon,
  Square,
  Trash2,
  UserRound,
  Users,
  X
} from "lucide-react";

type User = {
  id: string;
  username: string;
  email: string | null;
  role: string;
};

type RegisteredUser = User & {
  is_disabled: boolean;
};

type AdminUser = User & {
  is_disabled: boolean;
  created_at: number;
  updated_at: number;
  last_login_at: number | null;
};

type SessionResponse = {
  is_authenticated: boolean;
  user: User | null;
  can_create_account: boolean;
};

type RegisterResponse = {
  requires_approval: boolean;
  user: RegisteredUser;
};

type AdminUsersResponse = {
  users: AdminUser[];
};

type Backend = {
  id: string;
  name: string;
  base_url: string;
  is_enabled: boolean;
  last_health_status: string | null;
  last_error: string | null;
};

type BackendsResponse = {
  backends: Backend[];
};

type DetectLocalhostResponse = {
  detected: Array<{
    name: string;
    base_url: string;
  }>;
};

type ModelInfo = {
  name: string;
  supports_images: boolean;
};

type BackendModelGroup = {
  backend: {
    id: string;
    name: string;
  };
  models: ModelInfo[];
};

type ModelsResponse = {
  backends: BackendModelGroup[];
};

type ChatSummary = {
  id: string;
  title: string;
  default_backend_id: string;
  backend_name: string;
  default_model_name: string;
  updated_at: number;
  last_message_at: number;
  message_count: number;
};

type ChatDetail = {
  id: string;
  title: string;
  default_backend_id: string;
  backend_name: string;
  default_model_name: string;
  active_root_message_id: string | null;
  created_at: number;
  updated_at: number;
};

type ChatResponse = {
  chat: ChatDetail;
};

type ListChatsResponse = {
  chats: ChatSummary[];
};

type ChatMessageRevision = {
  id: string;
  content_text: string;
  thinking_text: string;
  source: string;
  created_at: number;
};

type ChatMessage = {
  id: string;
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
  active_revision: ChatMessageRevision | null;
  revisions: ChatMessageRevision[];
  revision_count: number;
  attachments: unknown[];
};

type ListMessagesResponse = {
  active_root_message_id: string | null;
  messages: ChatMessage[];
};

type MessageResponse = {
  message: ChatMessage;
};

type GenerateEvent =
  | {
      type: "message_start";
      user_message: ChatMessage | null;
      assistant_message: ChatMessage;
    }
  | {
      type: "thinking_delta";
      assistant_message_id: string;
      delta: string;
    }
  | {
      type: "content_delta";
      assistant_message_id: string;
      delta: string;
    }
  | {
      type: "message_done";
      assistant_message_id: string;
      done_reason: string | null;
    }
  | {
      type: "chat_title";
      chat_id: string;
      title: string;
    }
  | {
      type: "message_stopped";
      assistant_message_id: string;
    }
  | {
      type: "error";
      assistant_message_id: string | null;
      message: string;
    };

type AppSettings = {
  allow_signup: boolean;
  signup_limit: number;
  signup_count: number;
  max_upload_bytes: number;
  request_timeout_ms: number;
};

type LoadState =
  | { status: "loading" }
  | { status: "loaded"; session: SessionResponse }
  | { status: "error"; message: string };

type FormState = {
  isSubmitting: boolean;
  error: string | null;
};

type ApiError = {
  error?: {
    code?: string;
    message?: string;
  };
};

type Page = "chat" | "settings";
type SettingsSection = "profile" | "users" | "app" | "backends";
type AppRoute = { page: "chat"; chatId?: string } | { page: "settings"; section: SettingsSection };
type AppSettingsGuard = {
  isDirty: boolean;
  save: () => Promise<boolean>;
  discard: () => void;
};
type AutoScrollMode = "top" | "bottom" | "paused";
type BranchScrollAnchor = {
  messageId: string;
  topOffset: number;
};
type MessageVersion = {
  message: ChatMessage;
  revision: ChatMessageRevision;
};
type VersionInfo = {
  index: number;
  total: number;
  canPrevious: boolean;
  canNext: boolean;
  onPrevious: () => void;
  onNext: () => void;
};

const settingsSections: SettingsSection[] = ["profile", "users", "backends", "app"];
const rootSiblingGroupKey = "__root__";

function isSettingsSection(value: string | undefined): value is SettingsSection {
  return settingsSections.includes(value as SettingsSection);
}

function routeFromLocation(): AppRoute {
  const path = window.location.pathname;

  if (path.startsWith("/app/settings")) {
    const section = path.split("/")[3];
    return { page: "settings", section: isSettingsSection(section) ? section : "profile" };
  }

  if (path.startsWith("/app/chats/")) {
    const chatId = path.split("/")[3];
    if (chatId) {
      return { page: "chat", chatId };
    }
  }

  return { page: "chat" };
}

function pathForRoute(route: AppRoute) {
  if (route.page === "settings") {
    return `/app/settings/${route.section}`;
  }

  if (route.chatId) {
    return `/app/chats/${route.chatId}`;
  }

  return "/app";
}

function routesEqual(left: AppRoute, right: AppRoute) {
  return pathForRoute(left) === pathForRoute(right);
}

function modelValue(backendId: string, modelName: string) {
  return `${backendId}:${modelName}`;
}

function modelParts(value: string) {
  const separator = value.indexOf(":");
  if (separator < 1) {
    return null;
  }

  return {
    backendId: value.slice(0, separator),
    modelName: value.slice(separator + 1)
  };
}

function isLocalBackend(baseUrl: string) {
  try {
    const url = new URL(baseUrl);
    return (
      (url.hostname === "localhost" || url.hostname === "127.0.0.1") &&
      url.port === "11434"
    );
  } catch {
    return false;
  }
}

export default function App() {
  const [state, setState] = useState<LoadState>({ status: "loading" });

  const loadSession = useCallback(async () => {
    setState({ status: "loading" });
    try {
      const session = await requestJson<SessionResponse>("/api/auth/session");
      setState({ status: "loaded", session });
    } catch (error) {
      setState({
        status: "error",
        message: error instanceof Error ? error.message : "Session request failed"
      });
    }
  }, []);

  useEffect(() => {
    void loadSession();
  }, [loadSession]);

  if (state.status === "loading") {
    return (
      <main className="auth-page">
        <section className="auth-panel">
          <BrandMark />
          <h1>Loading</h1>
        </section>
      </main>
    );
  }

  if (state.status === "error") {
    return (
      <main className="auth-page">
        <section className="auth-panel">
          <BrandMark />
          <h1>Cannot Load Session</h1>
          <p className="error">{state.message}</p>
          <button type="button" onClick={() => void loadSession()}>
            Retry
          </button>
        </section>
      </main>
    );
  }

  if (!state.session.is_authenticated || !state.session.user) {
    return (
      <AuthScreen
        canCreateAccount={state.session.can_create_account}
        onSessionChanged={loadSession}
      />
    );
  }

  return <AppShell user={state.session.user} onSessionChanged={loadSession} />;
}

function AuthScreen({
  canCreateAccount,
  onSessionChanged
}: {
  canCreateAccount: boolean;
  onSessionChanged: () => Promise<void>;
}) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [pendingUser, setPendingUser] = useState<RegisteredUser | null>(null);

  useEffect(() => {
    if (!canCreateAccount && mode === "register") {
      setMode("login");
    }
  }, [canCreateAccount, mode]);

  return (
    <main className="auth-page">
      <section className="auth-panel">
        <BrandMark />
        {pendingUser ? (
          <PendingApproval
            username={pendingUser.username}
            onBack={() => {
              setPendingUser(null);
              setMode("login");
            }}
          />
        ) : mode === "login" ? (
          <LoginForm
            canCreateAccount={canCreateAccount}
            onCreateAccount={() => setMode("register")}
            onSessionChanged={onSessionChanged}
          />
        ) : canCreateAccount ? (
          <RegisterForm
            onBackToLogin={() => setMode("login")}
            onPendingApproval={setPendingUser}
            onSessionChanged={onSessionChanged}
          />
        ) : (
          <LoginForm
            canCreateAccount={false}
            onCreateAccount={() => setMode("register")}
            onSessionChanged={onSessionChanged}
          />
        )}
      </section>
    </main>
  );
}

function LoginForm({
  canCreateAccount,
  onCreateAccount,
  onSessionChanged
}: {
  canCreateAccount: boolean;
  onCreateAccount: () => void;
  onSessionChanged: () => Promise<void>;
}) {
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [formState, setFormState] = useState<FormState>({
    isSubmitting: false,
    error: null
  });

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormState({ isSubmitting: true, error: null });

    try {
      await requestJson("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ identifier, password })
      });
      await onSessionChanged();
    } catch (error) {
      setFormState({
        isSubmitting: false,
        error: error instanceof Error ? error.message : "Login failed"
      });
    }
  }

  return (
    <>
      <h1>Sign In</h1>
      <form onSubmit={submit}>
        <label>
          <span>Username or Email</span>
          <input
            autoComplete="username"
            autoFocus
            required
            value={identifier}
            onChange={(event) => setIdentifier(event.target.value)}
          />
        </label>
        <label>
          <span>Password</span>
          <input
            autoComplete="current-password"
            required
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        {formState.error && <p className="error">{formState.error}</p>}
        <button type="submit" disabled={formState.isSubmitting}>
          {formState.isSubmitting ? "Signing In..." : "Sign In"}
        </button>
      </form>
      {canCreateAccount && (
        <button type="button" className="text-button" onClick={onCreateAccount}>
          Create Account
        </button>
      )}
    </>
  );
}

function RegisterForm({
  onBackToLogin,
  onPendingApproval,
  onSessionChanged
}: {
  onBackToLogin: () => void;
  onPendingApproval: (user: RegisteredUser) => void;
  onSessionChanged: () => Promise<void>;
}) {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [formState, setFormState] = useState<FormState>({
    isSubmitting: false,
    error: null
  });

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormState({ isSubmitting: true, error: null });

    try {
      const response = await requestJson<RegisterResponse>("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({
          username,
          email: email.trim() === "" ? null : email,
          password
        })
      });

      if (response.requires_approval) {
        onPendingApproval(response.user);
        return;
      }

      await onSessionChanged();
    } catch (error) {
      setFormState({
        isSubmitting: false,
        error: error instanceof Error ? error.message : "Account creation failed"
      });
    }
  }

  return (
    <>
      <h1>Create Account</h1>
      <form onSubmit={submit}>
        <label>
          <span>Username</span>
          <input
            autoComplete="username"
            autoFocus
            required
            value={username}
            onChange={(event) => setUsername(event.target.value)}
          />
        </label>
        <label>
          <span>Email</span>
          <input
            autoComplete="email"
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
        </label>
        <label>
          <span>Password</span>
          <input
            autoComplete="new-password"
            required
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        {formState.error && <p className="error">{formState.error}</p>}
        <button type="submit" disabled={formState.isSubmitting}>
          {formState.isSubmitting ? "Creating..." : "Create Account"}
        </button>
      </form>
      <button type="button" className="text-button" onClick={onBackToLogin}>
        Back to Sign In
      </button>
    </>
  );
}

function PendingApproval({ username, onBack }: { username: string; onBack: () => void }) {
  return (
    <>
      <h1>Pending Approval</h1>
      <p className="status-message">{username} is waiting for admin approval.</p>
      <button type="button" onClick={onBack}>
        Back to Sign In
      </button>
    </>
  );
}

function AppShell({
  user,
  onSessionChanged
}: {
  user: User;
  onSessionChanged: () => Promise<void>;
}) {
  const [route, setRoute] = useState<AppRoute>(() => routeFromLocation());
  const routeRef = useRef(route);
  const appSettingsGuardRef = useRef<AppSettingsGuard | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSettingsMenuOpen, setIsSettingsMenuOpen] = useState(false);
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const [pendingNavigation, setPendingNavigation] = useState<AppRoute | null>(null);
  const [isSavingPendingNavigation, setIsSavingPendingNavigation] = useState(false);
  const [modelGroups, setModelGroups] = useState<BackendModelGroup[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [isLoadingChats, setIsLoadingChats] = useState(false);
  const [isCreatingChat, setIsCreatingChat] = useState(false);
  const [chatDeleteTarget, setChatDeleteTarget] = useState<ChatSummary | null>(null);
  const [isDeletingChat, setIsDeletingChat] = useState(false);
  const [queuedPrompt, setQueuedPrompt] = useState<{ chatId: string; prompt: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const isAdmin = user.role === "admin";
  const page = route.page;
  const isSettingsPage = page === "settings";
  const settingsSection = route.page === "settings" ? route.section : "profile";
  const currentChatId = route.page === "chat" ? route.chatId ?? null : null;

  useEffect(() => {
    routeRef.current = route;
  }, [route]);

  const updateAppSettingsGuard = useCallback((guard: AppSettingsGuard | null) => {
    appSettingsGuardRef.current = guard;
  }, []);

  const loadModels = useCallback(async () => {
    setIsLoadingModels(true);
    setModelError(null);

    try {
      const response = await requestJson<ModelsResponse>("/api/models");
      setModelGroups(response.backends);
      setSelectedModel((current) => {
        const values = response.backends.flatMap((group) =>
          group.models.map((model) => modelValue(group.backend.id, model.name))
        );

        return current && values.includes(current) ? current : values[0] ?? "";
      });
    } catch (loadError) {
      setModelGroups([]);
      setSelectedModel("");
      setModelError(loadError instanceof Error ? loadError.message : "Failed to load models");
    } finally {
      setIsLoadingModels(false);
    }
  }, []);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  const loadChats = useCallback(async () => {
    setIsLoadingChats(true);

    try {
      const response = await requestJson<ListChatsResponse>("/api/chats");
      setChats(response.chats);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load chats");
    } finally {
      setIsLoadingChats(false);
    }
  }, []);

  useEffect(() => {
    void loadChats();
  }, [loadChats]);

  useEffect(() => {
    function handlePopState() {
      const nextRoute = routeFromLocation();
      const currentRoute = routeRef.current;

      if (shouldGuardNavigation(currentRoute, nextRoute)) {
        window.history.pushState(null, "", pathForRoute(currentRoute));
        setPendingNavigation(nextRoute);
        setIsSettingsMenuOpen(false);
        setIsSidebarOpen(false);
        return;
      }

      setRoute(nextRoute);
      setIsSettingsMenuOpen(false);
      setIsSidebarOpen(false);
    }

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  function shouldGuardNavigation(currentRoute: AppRoute, nextRoute: AppRoute) {
    return (
      currentRoute.page === "settings" &&
      currentRoute.section === "app" &&
      Boolean(appSettingsGuardRef.current?.isDirty) &&
      !routesEqual(currentRoute, nextRoute)
    );
  }

  function applyNavigation(nextRoute: AppRoute) {
    const nextPath = pathForRoute(nextRoute);
    if (window.location.pathname !== nextPath) {
      window.history.pushState(null, "", nextPath);
    }
    setRoute(nextRoute);
    setIsSettingsMenuOpen(false);
    setIsSidebarOpen(false);
  }

  function navigate(nextRoute: AppRoute) {
    if (shouldGuardNavigation(route, nextRoute)) {
      setPendingNavigation(nextRoute);
      setIsSettingsMenuOpen(false);
      setIsSidebarOpen(false);
      return;
    }

    applyNavigation(nextRoute);
  }

  function openSettings(section: SettingsSection = "profile") {
    navigate({ page: "settings", section });
  }

  function openChat(chatId?: string) {
    navigate(chatId ? { page: "chat", chatId } : { page: "chat" });
  }

  function activateSettingsControl() {
    if (isSettingsPage) {
      openChat();
      return;
    }

    setIsSettingsMenuOpen((open) => !open);
  }

  async function logout() {
    setIsLoggingOut(true);
    setError(null);

    try {
      await requestJson("/api/auth/logout", { method: "POST" });
      await onSessionChanged();
    } catch (logoutError) {
      setIsLoggingOut(false);
      setError(logoutError instanceof Error ? logoutError.message : "Logout failed");
    }
  }

  async function saveAndContinueNavigation() {
    const nextRoute = pendingNavigation;
    const guard = appSettingsGuardRef.current;
    if (!nextRoute || !guard) {
      setPendingNavigation(null);
      return;
    }

    setIsSavingPendingNavigation(true);
    try {
      const saved = await guard.save();
      if (saved) {
        setPendingNavigation(null);
        applyNavigation(nextRoute);
      }
    } finally {
      setIsSavingPendingNavigation(false);
    }
  }

  function discardAndContinueNavigation() {
    const nextRoute = pendingNavigation;
    appSettingsGuardRef.current?.discard();
    setPendingNavigation(null);

    if (nextRoute) {
      applyNavigation(nextRoute);
    }
  }

  async function createChatFromPrompt(prompt: string) {
    const selected = modelParts(selectedModel);
    if (!selected) {
      setError("Select a model before starting a chat");
      return;
    }

    setIsCreatingChat(true);
    setError(null);

    try {
      const response = await requestJson<ChatResponse>("/api/chats", {
        method: "POST",
        body: JSON.stringify({
          title: "New Chat",
          default_backend_id: selected.backendId,
          default_model_name: selected.modelName
        })
      });

      if (prompt.trim()) {
        setQueuedPrompt({ chatId: response.chat.id, prompt });
      }

      await loadChats();
      openChat(response.chat.id);
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "Failed to create chat");
    } finally {
      setIsCreatingChat(false);
    }
  }

  async function renameChat(chatId: string, title: string) {
    setError(null);
    try {
      const response = await requestJson<ChatResponse>(`/api/chats/${chatId}`, {
        method: "PATCH",
        body: JSON.stringify({ title })
      });
      setChats((current) =>
        current.map((chat) =>
          chat.id === chatId ? { ...chat, title: response.chat.title } : chat
        )
      );
    } catch (renameError) {
      setError(renameError instanceof Error ? renameError.message : "Failed to rename chat");
      throw renameError;
    }
  }

  async function deleteSelectedChat() {
    const chat = chatDeleteTarget;
    if (!chat) {
      return;
    }

    setIsDeletingChat(true);
    setError(null);
    try {
      await requestJson(`/api/chats/${chat.id}`, { method: "DELETE" });
      setChats((current) => current.filter((item) => item.id !== chat.id));
      setChatDeleteTarget(null);
      if (currentChatId === chat.id) {
        openChat();
      }
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Failed to delete chat");
    } finally {
      setIsDeletingChat(false);
    }
  }

  return (
    <main className={isSidebarOpen ? "app-shell sidebar-open" : "app-shell"}>
      <Sidebar
        chats={chats}
        currentChatId={currentChatId}
        currentPage={page}
        isOpen={isSidebarOpen}
        isLoading={isLoadingChats}
        onClose={() => setIsSidebarOpen(false)}
        onDeleteChat={setChatDeleteTarget}
        onOpenChat={openChat}
        onRenameChat={renameChat}
      />
      <button
        type="button"
        className="sidebar-backdrop"
        aria-label="Close sidebar"
        onClick={() => setIsSidebarOpen(false)}
      />
      <section className="main-pane">
        <header className="topbar">
          <div className="topbar-left">
            <button
              type="button"
              className="icon-button mobile-only"
              aria-label="Open sidebar"
              onClick={() => setIsSidebarOpen(true)}
            >
              <Menu />
            </button>
            <ModelPicker
              groups={modelGroups}
              isLoading={isLoadingModels}
              error={modelError}
              value={selectedModel}
              onChange={setSelectedModel}
            />
          </div>
          <div className="topbar-right">
            <button
              type="button"
              className="primary-action"
              disabled={isCreatingChat || !selectedModel}
              onClick={() => void createChatFromPrompt("")}
            >
              <MessageSquarePlus />
              <span>{isCreatingChat ? "Creating..." : "New Chat"}</span>
            </button>
            <div className="settings-menu-wrap">
              <button
                type="button"
                className="icon-button"
                aria-label={
                  isSettingsPage
                    ? "Close settings"
                    : isSettingsMenuOpen
                      ? "Close settings menu"
                      : "Open settings menu"
                }
                aria-expanded={isSettingsMenuOpen}
                onClick={activateSettingsControl}
              >
                {isSettingsPage || isSettingsMenuOpen ? <X /> : <Cog />}
              </button>
              {isSettingsMenuOpen && !isSettingsPage && (
                <div className="settings-menu">
                  <button
                    type="button"
                    className="menu-item"
                    onClick={() => openSettings(settingsSection)}
                  >
                    <SettingsIcon />
                    <span>Settings</span>
                  </button>
                  <button
                    type="button"
                    className="menu-item"
                    onClick={() => void logout()}
                    disabled={isLoggingOut}
                  >
                    <LogOut />
                    <span>{isLoggingOut ? "Signing Out..." : "Sign Out"}</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>
        {page === "settings" ? (
          <SettingsPage
            currentUser={user}
            activeSection={settingsSection}
            onBackendsChanged={loadModels}
            onAppSettingsGuardChange={updateAppSettingsGuard}
            onSelectSection={(section) => openSettings(section)}
            isAdmin={isAdmin}
          />
        ) : (
          currentChatId ? (
            <ChatView
              chatId={currentChatId}
              error={error}
              queuedPrompt={queuedPrompt?.chatId === currentChatId ? queuedPrompt.prompt : null}
              selectedModel={selectedModel}
              externalTitle={chats.find((chat) => chat.id === currentChatId)?.title ?? null}
              onChatsChanged={loadChats}
              onModelSelected={setSelectedModel}
              onQueuedPromptConsumed={() => setQueuedPrompt(null)}
            />
          ) : (
            <ChatHome
              error={error}
              isCreating={isCreatingChat}
              selectedModel={selectedModel}
              onCreateChat={createChatFromPrompt}
            />
          )
        )}
      </section>
      {pendingNavigation && (
        <UnsavedSettingsDialog
          isSaving={isSavingPendingNavigation}
          onCancel={() => setPendingNavigation(null)}
          onDiscard={discardAndContinueNavigation}
          onSave={() => void saveAndContinueNavigation()}
        />
      )}
      {chatDeleteTarget && (
        <ConfirmDialog
          title="Delete Chat"
          message={`Delete "${chatDeleteTarget.title}"? This will remove the chat and its messages.`}
          confirmLabel="Delete"
          isBusy={isDeletingChat}
          onCancel={() => setChatDeleteTarget(null)}
          onConfirm={() => void deleteSelectedChat()}
        />
      )}
    </main>
  );
}

function Sidebar({
  chats,
  currentChatId,
  currentPage,
  isOpen,
  isLoading,
  onClose,
  onDeleteChat,
  onOpenChat,
  onRenameChat
}: {
  chats: ChatSummary[];
  currentChatId: string | null;
  currentPage: Page;
  isOpen: boolean;
  isLoading: boolean;
  onClose: () => void;
  onDeleteChat: (chat: ChatSummary) => void;
  onOpenChat: (chatId?: string) => void;
  onRenameChat: (chatId: string, title: string) => Promise<void>;
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

  return (
    <aside className="sidebar">
      <div>
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
        <div className="chat-history">
          <p className="eyebrow">Previous Chats</p>
          {isLoading && chats.length === 0 ? (
            <p>Loading chats...</p>
          ) : chats.length === 0 ? (
            <p>No chats yet</p>
          ) : (
            <div className="chat-link-list">
              {chats.map((chat) => (
                <ChatListItem
                  key={chat.id}
                  chat={chat}
                  isActive={currentChatId === chat.id}
                  isEditing={editingChatId === chat.id}
                  isMenuOpen={openMenuChatId === chat.id}
                  onCancelEditing={() => setEditingChatId(null)}
                  onCloseMenu={() => setOpenMenuChatId(null)}
                  onDelete={() => {
                    setOpenMenuChatId(null);
                    onDeleteChat(chat);
                  }}
                  onOpen={() => {
                    setOpenMenuChatId(null);
                    onOpenChat(chat.id);
                  }}
                  onOpenMenu={() => setOpenMenuChatId(chat.id)}
                  onRename={(title) => onRenameChat(chat.id, title)}
                  onStartEditing={() => {
                    setOpenMenuChatId(null);
                    setEditingChatId(chat.id);
                  }}
                  onToggleMenu={() =>
                    setOpenMenuChatId((current) => (current === chat.id ? null : chat.id))
                  }
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}

function ChatListItem({
  chat,
  isActive,
  isEditing,
  isMenuOpen,
  onCancelEditing,
  onCloseMenu,
  onDelete,
  onOpen,
  onOpenMenu,
  onRename,
  onStartEditing,
  onToggleMenu
}: {
  chat: ChatSummary;
  isActive: boolean;
  isEditing: boolean;
  isMenuOpen: boolean;
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
          <span>{chat.title}</span>
          <small>{chat.default_model_name}</small>
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

function ChatHome({
  error,
  isCreating,
  selectedModel,
  onCreateChat
}: {
  error: string | null;
  isCreating: boolean;
  selectedModel: string;
  onCreateChat: (prompt: string) => Promise<void>;
}) {
  return (
    <div className="chat-home">
      <div className="chat-home-inner">
        <BrandMark compact />
        <StartChatComposer
          isBusy={isCreating}
          isDisabled={!selectedModel}
          placeholder={selectedModel ? "Message Vashti" : "Select a model to start"}
          onSubmit={onCreateChat}
        />
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}

function ChatView({
  chatId,
  error,
  externalTitle,
  queuedPrompt,
  selectedModel,
  onChatsChanged,
  onModelSelected,
  onQueuedPromptConsumed
}: {
  chatId: string;
  error: string | null;
  externalTitle: string | null;
  queuedPrompt: string | null;
  selectedModel: string;
  onChatsChanged: () => Promise<void>;
  onModelSelected: (value: string) => void;
  onQueuedPromptConsumed: () => void;
}) {
  const [chat, setChat] = useState<ChatDetail | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeAssistantId, setActiveAssistantId] = useState<string | null>(null);
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);
  const [busyMessageId, setBusyMessageId] = useState<string | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ChatMessage | null>(null);
  const messageListRef = useRef<HTMLElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const generationRunRef = useRef(0);
  const autoScrollModeRef = useRef<AutoScrollMode>("top");
  const userScrollIntentRef = useRef(false);
  const userScrollIntentTimeoutRef = useRef<number | null>(null);
  const scrollToBottomAfterLoadRef = useRef(false);
  const branchScrollAnchorRef = useRef<BranchScrollAnchor | null>(null);
  const thinkingStartedAtRef = useRef(new Map<string, number>());
  const [thinkingDurations, setThinkingDurations] = useState<Record<string, number>>({});
  const visibleMessages = useMemo(
    () => activePathMessages(messages, chat?.active_root_message_id ?? null),
    [chat?.active_root_message_id, messages]
  );
  const siblingGroups = useMemo(() => groupMessagesByParent(messages), [messages]);

  const loadChat = useCallback(async () => {
    scrollToBottomAfterLoadRef.current = true;
    setIsLoading(true);
    setLoadError(null);

    try {
      const [chatResponse, messageResponse] = await Promise.all([
        requestJson<ChatResponse>(`/api/chats/${chatId}`),
        requestJson<ListMessagesResponse>(`/api/chats/${chatId}/messages`)
      ]);

      setChat({
        ...chatResponse.chat,
        active_root_message_id: messageResponse.active_root_message_id
      });
      thinkingStartedAtRef.current.clear();
      setThinkingDurations({});
      setMessages(messageResponse.messages);
      const latestModel = latestAssistantModelValue(
        activePathMessages(messageResponse.messages, messageResponse.active_root_message_id)
      );
      onModelSelected(
        latestModel ??
          modelValue(chatResponse.chat.default_backend_id, chatResponse.chat.default_model_name)
      );
    } catch (chatError) {
      setLoadError(chatError instanceof Error ? chatError.message : "Failed to load chat");
    } finally {
      setIsLoading(false);
    }
  }, [chatId, onModelSelected]);

  const streamAssistantResponse = useCallback(
    async (path: string, body: unknown) => {
      const runId = generationRunRef.current + 1;
      generationRunRef.current = runId;
      const controller = new AbortController();
      abortRef.current = controller;
      autoScrollModeRef.current = "top";
      setIsGenerating(true);
      setGenerationError(null);

      try {
        const response = await fetch(path, {
          method: "POST",
          credentials: "include",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal
        });

        if (!response.ok) {
          throw new Error(await responseErrorMessage(response));
        }

        if (!response.body) {
          throw new Error("Generation stream was empty");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) {
              continue;
            }
            applyGenerateEvent(JSON.parse(trimmed) as GenerateEvent, runId);
          }
        }

        buffer += decoder.decode();
        const trailing = buffer.trim();
        if (trailing) {
          applyGenerateEvent(JSON.parse(trailing) as GenerateEvent, runId);
        }

        await onChatsChanged();
      } catch (generateError) {
        if (generateError instanceof DOMException && generateError.name === "AbortError") {
          return;
        }

        setGenerationError(
          generateError instanceof Error ? generateError.message : "Generation failed"
        );
      } finally {
        if (generationRunRef.current === runId) {
          setIsGenerating(false);
          setActiveAssistantId(null);
          abortRef.current = null;
        }
      }
    },
    [onChatsChanged]
  );

  const generate = useCallback(
    async (prompt: string) => {
      if (isGenerating) {
        return;
      }

      const selected = modelParts(selectedModel);
      await streamAssistantResponse(`/api/chats/${chatId}/generate`, {
        user_message: { content_text: prompt },
        backend_id: selected?.backendId ?? null,
        model_name: selected?.modelName ?? null,
        think_mode: null,
        attachments: []
      });
    },
    [chatId, isGenerating, selectedModel, streamAssistantResponse]
  );

  useEffect(() => {
    void loadChat();
  }, [loadChat]);

  useEffect(() => {
    if (!externalTitle) {
      return;
    }

    setChat((current) =>
      current && current.title !== externalTitle ? { ...current, title: externalTitle } : current
    );
  }, [externalTitle]);

  useEffect(() => {
    const latestModel = latestAssistantModelValue(visibleMessages);
    if (latestModel) {
      onModelSelected(latestModel);
    }
  }, [onModelSelected, visibleMessages]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (userScrollIntentTimeoutRef.current) {
        window.clearTimeout(userScrollIntentTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!queuedPrompt || isLoading || !chat || isGenerating) {
      return;
    }

    onQueuedPromptConsumed();
    void generate(queuedPrompt);
  }, [chat, generate, isGenerating, isLoading, onQueuedPromptConsumed, queuedPrompt]);

  useEffect(() => {
    if (!pendingPrompt || isGenerating || isLoading || !chat) {
      return;
    }

    const prompt = pendingPrompt;
    setPendingPrompt(null);
    void generate(prompt);
  }, [chat, generate, isGenerating, isLoading, pendingPrompt]);

  useLayoutEffect(() => {
    if (isLoading) {
      return;
    }

    const list = messageListRef.current;
    if (!list) {
      return;
    }

    const messageList = list;
    const finalMessageId = visibleMessages[visibleMessages.length - 1]?.id ?? null;
    const finalMessage = finalMessageId
      ? messageList.querySelector<HTMLElement>(`[data-message-id="${finalMessageId}"]`)
      : null;

    function updateMessageListBuffer() {
      if (!finalMessage) {
        messageList.style.setProperty("--message-list-buffer-height", "0px");
        return;
      }

      const styles = window.getComputedStyle(messageList);
      const topPadding = Number.parseFloat(styles.paddingTop) || 0;
      const bottomPadding = Number.parseFloat(styles.paddingBottom) || 0;
      const rowGap = Number.parseFloat(styles.rowGap || styles.gap) || 0;
      const availableSpace =
        messageList.clientHeight -
        finalMessage.getBoundingClientRect().height -
        topPadding -
        bottomPadding -
        rowGap;
      const bufferHeight = Math.max(0, Math.floor(availableSpace));
      messageList.style.setProperty("--message-list-buffer-height", `${bufferHeight}px`);
    }

    updateMessageListBuffer();

    const resizeObserver =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(updateMessageListBuffer);
    resizeObserver?.observe(messageList);
    if (finalMessage) {
      resizeObserver?.observe(finalMessage);
    }
    window.addEventListener("resize", updateMessageListBuffer);
    window.visualViewport?.addEventListener("resize", updateMessageListBuffer);

    return () => {
      resizeObserver?.disconnect();
      window.removeEventListener("resize", updateMessageListBuffer);
      window.visualViewport?.removeEventListener("resize", updateMessageListBuffer);
    };
  }, [chatId, isLoading, visibleMessages]);

  useLayoutEffect(() => {
    if (isLoading || !scrollToBottomAfterLoadRef.current) {
      return;
    }

    const list = messageListRef.current;
    if (!list) {
      return;
    }

    const lastMessage = visibleMessages[visibleMessages.length - 1];
    const lastMessageElement = lastMessage
      ? list.querySelector<HTMLElement>(`[data-message-id="${lastMessage.id}"]`)
      : null;
    if (lastMessageElement) {
      lastMessageElement.scrollIntoView({ block: "end" });
    } else {
      list.scrollTop = 0;
    }
    scrollToBottomAfterLoadRef.current = false;
  }, [chatId, isLoading, visibleMessages]);

  useLayoutEffect(() => {
    const anchor = branchScrollAnchorRef.current;
    if (!anchor || isLoading) {
      return;
    }

    const list = messageListRef.current;
    const messageElement = list?.querySelector<HTMLElement>(
      `[data-message-id="${anchor.messageId}"]`
    );
    if (!list || !messageElement) {
      branchScrollAnchorRef.current = null;
      return;
    }

    const anchorElement =
      messageElement.querySelector<HTMLElement>(".version-switcher") ?? messageElement;
    const nextTopOffset = anchorElement.getBoundingClientRect().top - list.getBoundingClientRect().top;
    list.scrollTop += nextTopOffset - anchor.topOffset;
    branchScrollAnchorRef.current = null;
  }, [isLoading, visibleMessages]);

  useLayoutEffect(() => {
    if (!isGenerating || !activeAssistantId || autoScrollModeRef.current === "paused") {
      return;
    }

    const list = messageListRef.current;
    const activeMessage = list?.querySelector<HTMLElement>(
      `[data-message-id="${activeAssistantId}"]`
    );
    if (!list || !activeMessage) {
      return;
    }

    const listRect = list.getBoundingClientRect();
    const messageRect = activeMessage.getBoundingClientRect();
    const topOffset = messageRect.top - listRect.top;
    const bottomOverflow = messageRect.bottom - listRect.bottom;

    if (autoScrollModeRef.current === "bottom") {
      activeMessage.scrollIntoView({ block: "end" });
      return;
    }

    if (bottomOverflow <= 0 || topOffset <= 8) {
      return;
    }

    list.scrollTop += Math.min(bottomOverflow + 16, topOffset - 8);
  }, [activeAssistantId, isGenerating, visibleMessages]);

  function noteUserScrollIntent() {
    if (!isGenerating) {
      return;
    }

    userScrollIntentRef.current = true;
    if (userScrollIntentTimeoutRef.current) {
      window.clearTimeout(userScrollIntentTimeoutRef.current);
    }
    userScrollIntentTimeoutRef.current = window.setTimeout(() => {
      userScrollIntentRef.current = false;
    }, 240);

    window.requestAnimationFrame(updateAutoScrollModeFromScrollPosition);
  }

  function handleMessageListScroll() {
    if (!isGenerating || !userScrollIntentRef.current) {
      return;
    }

    updateAutoScrollModeFromScrollPosition();
  }

  function updateAutoScrollModeFromScrollPosition() {
    const list = messageListRef.current;
    if (!list) {
      return;
    }

    const activeMessage = activeAssistantId
      ? list.querySelector<HTMLElement>(`[data-message-id="${activeAssistantId}"]`)
      : null;
    if (!activeMessage) {
      const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight <= 12;
      autoScrollModeRef.current = atBottom ? "bottom" : "paused";
      return;
    }

    const listRect = list.getBoundingClientRect();
    const messageRect = activeMessage.getBoundingClientRect();
    const bottomOverflow = messageRect.bottom - listRect.bottom;
    if (bottomOverflow > 12) {
      autoScrollModeRef.current = "paused";
      return;
    }

    const topOffset = messageRect.top - listRect.top;
    autoScrollModeRef.current = topOffset <= 8 ? "bottom" : "top";
  }

  function clearThinkingDuration(messageId: string) {
    thinkingStartedAtRef.current.delete(messageId);
    setThinkingDurations((current) => {
      if (!(messageId in current)) {
        return current;
      }

      const next = { ...current };
      delete next[messageId];
      return next;
    });
  }

  function markThinkingStarted(messageId: string) {
    if (!thinkingStartedAtRef.current.has(messageId)) {
      thinkingStartedAtRef.current.set(messageId, Date.now());
    }
  }

  function finishThinkingDuration(messageId: string) {
    const startedAt = thinkingStartedAtRef.current.get(messageId);
    if (!startedAt) {
      return;
    }

    thinkingStartedAtRef.current.delete(messageId);
    const durationSeconds = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
    setThinkingDurations((current) =>
      current[messageId] ? current : { ...current, [messageId]: durationSeconds }
    );
  }

  function applyGenerateEvent(event: GenerateEvent, runId: number) {
    if (generationRunRef.current !== runId && event.type !== "chat_title") {
      return;
    }

    switch (event.type) {
      case "message_start":
        autoScrollModeRef.current = "top";
        setActiveAssistantId(event.assistant_message.id);
        clearThinkingDuration(event.assistant_message.id);
        if (event.user_message && !event.user_message.parent_message_id) {
          setChat((current) =>
            current ? { ...current, active_root_message_id: event.user_message?.id ?? null } : current
          );
        }
        setMessages((current) => {
          const parentUpdates = new Map<string, string>();
          if (event.user_message?.parent_message_id) {
            parentUpdates.set(event.user_message.parent_message_id, event.user_message.id);
          }
          if (event.assistant_message.parent_message_id) {
            parentUpdates.set(
              event.assistant_message.parent_message_id,
              event.assistant_message.id
            );
          }

          const incomingIds = new Set(
            [event.user_message?.id, event.assistant_message.id].filter(
              (id): id is string => Boolean(id)
            )
          );
          const now = Math.floor(Date.now() / 1000);
          const existingMessages = current
            .filter((message) => !incomingIds.has(message.id))
            .map((message) => {
              const activeChildId = parentUpdates.get(message.id);
              return activeChildId
                ? { ...message, active_child_message_id: activeChildId, updated_at: now }
                : message;
            });

          return [
            ...existingMessages,
            ...(event.user_message ? [event.user_message] : []),
            event.assistant_message
          ];
        });
        break;
      case "thinking_delta":
        markThinkingStarted(event.assistant_message_id);
        appendMessageText(event.assistant_message_id, "thinking_text", event.delta);
        break;
      case "content_delta":
        finishThinkingDuration(event.assistant_message_id);
        appendMessageText(event.assistant_message_id, "content_text", event.delta);
        break;
      case "message_done":
        finishThinkingDuration(event.assistant_message_id);
        updateMessageStatus(event.assistant_message_id, "complete", event.done_reason);
        if (generationRunRef.current === runId) {
          setIsGenerating(false);
          setActiveAssistantId(null);
          abortRef.current = null;
        }
        break;
      case "chat_title":
        setChat((current) =>
          current && current.id === event.chat_id ? { ...current, title: event.title } : current
        );
        break;
      case "message_stopped":
        finishThinkingDuration(event.assistant_message_id);
        updateMessageStatus(event.assistant_message_id, "stopped", "stopped");
        break;
      case "error":
        setGenerationError(event.message);
        if (event.assistant_message_id) {
          finishThinkingDuration(event.assistant_message_id);
          updateMessageStatus(event.assistant_message_id, "error", "error");
        }
        break;
    }
  }

  function appendMessageText(
    messageId: string,
    field: "content_text" | "thinking_text",
    delta: string
  ) {
    setMessages((current) =>
      current.map((message) => {
        if (message.id !== messageId) {
          return message;
        }

        const activeRevision = message.active_revision ?? {
          id: message.active_revision_id ?? `${message.id}-active`,
          content_text: "",
          thinking_text: "",
          source: "original",
          created_at: message.created_at
        };

        return {
          ...message,
          status: "streaming",
          active_revision: {
            ...activeRevision,
            [field]: activeRevision[field] + delta
          },
          revisions: updateRevisionList(message.revisions, {
            ...activeRevision,
            [field]: activeRevision[field] + delta
          })
        };
      })
    );
  }

  function updateMessageStatus(messageId: string, status: string, doneReason: string | null) {
    setMessages((current) =>
      current.map((message) =>
        message.id === messageId
          ? { ...message, status, done_reason: doneReason, completed_at: Math.floor(Date.now() / 1000) }
          : message
      )
    );
  }

  async function stopGeneration() {
    const assistantId = activeAssistantId;
    generationRunRef.current += 1;
    if (!assistantId) {
      abortRef.current?.abort();
      setIsGenerating(false);
      setActiveAssistantId(null);
      abortRef.current = null;
      return;
    }

    try {
      await requestJson(`/api/chats/${chatId}/messages/${assistantId}/stop`, { method: "POST" });
      updateMessageStatus(assistantId, "stopped", "stopped");
    } catch (stopError) {
      setGenerationError(stopError instanceof Error ? stopError.message : "Failed to stop");
    } finally {
      abortRef.current?.abort();
      setIsGenerating(false);
      setActiveAssistantId(null);
    }
  }

  async function submitPrompt(prompt: string) {
    if (isGenerating) {
      setPendingPrompt(prompt);
      await stopGeneration();
      return;
    }

    await generate(prompt);
  }

  function replaceMessage(nextMessage: ChatMessage) {
    setMessages((current) =>
      current.map((message) => (message.id === nextMessage.id ? nextMessage : message))
    );
  }

  async function copyMessage(message: ChatMessage) {
    const content = message.active_revision?.content_text ?? "";
    if (!content || message.is_deleted) {
      return;
    }

    await navigator.clipboard.writeText(content);
    setCopiedMessageId(message.id);
    window.setTimeout(() => {
      setCopiedMessageId((current) => (current === message.id ? null : current));
    }, 1200);
  }

  async function editMessage(message: ChatMessage, contentText: string) {
    setBusyMessageId(message.id);
    setGenerationError(null);

    try {
      const response = await requestJson<MessageResponse>(
        `/api/chats/${chatId}/messages/${message.id}`,
        {
          method: "PATCH",
          body: JSON.stringify({ content_text: contentText })
        }
      );
      replaceMessage(response.message);
      await onChatsChanged();
    } catch (editError) {
      setGenerationError(editError instanceof Error ? editError.message : "Failed to edit message");
      throw editError;
    } finally {
      setBusyMessageId(null);
    }
  }

  async function branchMessage(message: ChatMessage, contentText: string) {
    if (isGenerating || message.role !== "user") {
      return;
    }

    const selected = modelParts(selectedModel);
    await streamAssistantResponse(`/api/chats/${chatId}/messages/${message.id}/branch`, {
      content_text: contentText,
      backend_id: selected?.backendId ?? null,
      model_name: selected?.modelName ?? null,
      think_mode: null,
      attachments: []
    });
  }

  async function deleteMessage(message: ChatMessage) {
    setBusyMessageId(message.id);
    setGenerationError(null);

    try {
      const response = await requestJson<MessageResponse>(
        `/api/chats/${chatId}/messages/${message.id}`,
        { method: "DELETE" }
      );
      replaceMessage(response.message);
      setDeleteTarget(null);
      await onChatsChanged();
    } catch (deleteError) {
      setGenerationError(
        deleteError instanceof Error ? deleteError.message : "Failed to delete message"
      );
    } finally {
      setBusyMessageId(null);
    }
  }

  async function regenerateMessage(message: ChatMessage) {
    if (isGenerating || message.role !== "assistant") {
      return;
    }

    const selected = modelParts(selectedModel);
    await streamAssistantResponse(`/api/chats/${chatId}/messages/${message.id}/regenerate`, {
      backend_id: selected?.backendId ?? message.backend_id,
      model_name: selected?.modelName ?? message.model_name,
      think_mode: null,
      attachments: []
    });
  }

  async function selectVersion(currentMessage: ChatMessage, version: MessageVersion) {
    const nextMessage = version.message;
    const nextRevision = version.revision;
    const isSameMessage = currentMessage.id === nextMessage.id;
    const isSameRevision = nextMessage.active_revision_id === nextRevision.id;
    if ((isSameMessage && isSameRevision) || isGenerating) {
      return;
    }

    const list = messageListRef.current;
    const messageElement = list?.querySelector<HTMLElement>(
      `[data-message-id="${currentMessage.id}"]`
    );
    const anchorElement =
      messageElement?.querySelector<HTMLElement>(".version-switcher") ?? messageElement;
    if (list && anchorElement) {
      branchScrollAnchorRef.current = {
        messageId: nextMessage.id,
        topOffset: anchorElement.getBoundingClientRect().top - list.getBoundingClientRect().top
      };
    }

    setBusyMessageId(currentMessage.id);
    setGenerationError(null);

    try {
      if (!isSameMessage) {
        if (currentMessage.parent_message_id) {
          const response = await requestJson<MessageResponse>(
            `/api/chats/${chatId}/messages/${currentMessage.parent_message_id}/active-child`,
            {
              method: "PATCH",
              body: JSON.stringify({ active_child_message_id: nextMessage.id })
            }
          );
          replaceMessage(response.message);
        } else {
          const response = await requestJson<ChatResponse>(`/api/chats/${chatId}/active-root`, {
            method: "PATCH",
            body: JSON.stringify({ active_root_message_id: nextMessage.id })
          });
          setChat(response.chat);
        }

        if (nextMessage.role === "assistant" && nextMessage.backend_id && nextMessage.model_name) {
          onModelSelected(modelValue(nextMessage.backend_id, nextMessage.model_name));
        }
      }

      if (!isSameRevision) {
        const response = await requestJson<MessageResponse>(
          `/api/chats/${chatId}/messages/${nextMessage.id}/active-revision`,
          {
            method: "PATCH",
            body: JSON.stringify({ active_revision_id: nextRevision.id })
          }
        );
        replaceMessage(response.message);
      }
    } catch (selectError) {
      setGenerationError(
        selectError instanceof Error ? selectError.message : "Failed to switch version"
      );
    } finally {
      setBusyMessageId(null);
    }
  }

  function versionInfoFor(message: ChatMessage): VersionInfo | null {
    const versions = versionsForMessage(message);
    if (versions.length < 2 || !message.active_revision_id) {
      return null;
    }

    const index = versions.findIndex(
      (version) =>
        version.message.id === message.id && version.revision.id === message.active_revision_id
    );
    if (index < 0) {
      return null;
    }

    const previousVersion = versions[index - 1] ?? null;
    const nextVersion = versions[index + 1] ?? null;

    return {
      index,
      total: versions.length,
      canPrevious: Boolean(previousVersion),
      canNext: Boolean(nextVersion),
      onPrevious: () => {
        if (previousVersion) {
          void selectVersion(message, previousVersion);
        }
      },
      onNext: () => {
        if (nextVersion) {
          void selectVersion(message, nextVersion);
        }
      }
    };
  }

  function versionsForMessage(message: ChatMessage) {
    const siblings = siblingGroups.get(parentGroupKey(message.parent_message_id)) ?? [];
    return siblings
      .flatMap((sibling) =>
        revisionsForMessage(sibling).map((revision) => ({
          message: sibling,
          revision
        }))
      )
      .sort(compareVersionsByCreatedAt);
  }

  return (
    <div className="chat-view">
      {isLoading ? (
        <div className="empty-state">
          <RetroLoader />
        </div>
      ) : loadError ? (
        <div className="empty-state">
          <p className="error">{loadError}</p>
        </div>
      ) : chat ? (
        <>
          <header className="chat-titlebar">
            <div>
              <h1>{chat.title}</h1>
              <p>
                {chat.backend_name} / {chat.default_model_name}
              </p>
            </div>
          </header>
          <section
            className={
              visibleMessages.length > 0 ? "message-list message-list-buffered" : "message-list"
            }
            aria-label="Chat messages"
            ref={messageListRef}
            onScroll={handleMessageListScroll}
            onTouchMove={noteUserScrollIntent}
            onWheel={noteUserScrollIntent}
          >
            {visibleMessages.length === 0 ? (
              <div className="message-list-empty">
                <p>Ready for messages.</p>
              </div>
            ) : (
              visibleMessages.map((message) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  versionInfo={versionInfoFor(message)}
                  copied={copiedMessageId === message.id}
                  isBusy={busyMessageId === message.id}
                  isGenerating={isGenerating}
                  thinkingDurationSeconds={thinkingDurations[message.id] ?? null}
                  onCopy={copyMessage}
                  onDelete={setDeleteTarget}
                  onBranch={branchMessage}
                  onEdit={editMessage}
                  onRegenerate={regenerateMessage}
                />
              ))
            )}
          </section>
          <div className="chat-composer-wrap">
            <StartChatComposer
              isBusy={isGenerating}
              isDisabled={!selectedModel}
              isGenerating={isGenerating}
              placeholder={selectedModel ? "Message Vashti" : "Select a model to continue"}
              onStop={stopGeneration}
              onSubmit={submitPrompt}
            />
          </div>
        </>
      ) : null}
      {(error || generationError) && (
        <p className="error chat-view-error">{generationError ?? error}</p>
      )}
      {deleteTarget && (
        <ConfirmDialog
          title="Delete Message"
          message="Delete this message? Its text and thinking content will be scrubbed, but the chat path will remain intact."
          confirmLabel="Delete"
          isBusy={busyMessageId === deleteTarget.id}
          onCancel={() => setDeleteTarget(null)}
          onConfirm={() => void deleteMessage(deleteTarget)}
        />
      )}
    </div>
  );
}

function MessageBubble({
  message,
  versionInfo,
  copied,
  isBusy,
  isGenerating,
  thinkingDurationSeconds,
  onCopy,
  onDelete,
  onBranch,
  onEdit,
  onRegenerate
}: {
  message: ChatMessage;
  versionInfo: VersionInfo | null;
  copied: boolean;
  isBusy: boolean;
  isGenerating: boolean;
  thinkingDurationSeconds: number | null;
  onCopy: (message: ChatMessage) => Promise<void>;
  onDelete: (message: ChatMessage) => void;
  onBranch: (message: ChatMessage, contentText: string) => Promise<void>;
  onEdit: (message: ChatMessage, contentText: string) => Promise<void>;
  onRegenerate: (message: ChatMessage) => Promise<void>;
}) {
  const content = message.is_deleted
    ? "Message deleted"
    : message.active_revision?.content_text.trim() || "";
  const thinking = message.active_revision?.thinking_text.trim();
  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState(content);

  useEffect(() => {
    if (!isEditing) {
      setDraft(content);
    }
  }, [content, isEditing]);

  async function saveEdit() {
    await onEdit(message, draft);
    setIsEditing(false);
  }

  async function sendEdit() {
    setIsEditing(false);
    await onBranch(message, draft);
  }

  return (
    <article
      className={`message-bubble message-bubble-${message.role}`}
      data-message-id={message.id}
    >
      <div className="message-header">
        <p className="message-role">{messageLabel(message)}</p>
        {versionInfo && (
          <VersionSwitcher
            versionInfo={versionInfo}
            isBusy={isBusy}
            isGenerating={isGenerating}
          />
        )}
      </div>
      {thinking && !message.is_deleted && (
        <details className="message-thinking">
          <summary>{thinkingSummary(message, thinkingDurationSeconds)}</summary>
          <p>{thinking}</p>
        </details>
      )}
      {isEditing ? (
        <div className="message-edit">
          <textarea value={draft} onChange={(event) => setDraft(event.target.value)} rows={5} />
          <div className="message-actions">
            <button type="button" className="secondary-button" disabled={isBusy} onClick={() => setIsEditing(false)}>
              <X />
              <span>Cancel</span>
            </button>
            <button type="button" disabled={isBusy || draft.trim() === ""} onClick={() => void saveEdit()}>
              <Save />
              <span>{isBusy ? "Saving..." : "Save"}</span>
            </button>
            {message.role === "user" && (
              <button
                type="button"
                disabled={isBusy || isGenerating || draft.trim() === ""}
                onClick={() => void sendEdit()}
              >
                <SendHorizontal />
                <span>Send</span>
              </button>
            )}
          </div>
        </div>
      ) : content ? (
        <MarkdownContent content={content} />
      ) : (
        <p>{message.status === "streaming" ? <RetroLoader /> : "No content"}</p>
      )}
      {!isEditing && (
        <div className="message-actions">
          {message.role === "assistant" && (
            <button
              type="button"
              className="message-icon-button"
              title="Regenerate"
              aria-label="Regenerate"
              disabled={isBusy || isGenerating || message.status === "streaming"}
              onClick={() => void onRegenerate(message)}
            >
              <RefreshCw />
            </button>
          )}
          <button
            type="button"
            className="message-icon-button"
            title="Copy"
            aria-label="Copy"
            disabled={message.is_deleted || content === ""}
            onClick={() => void onCopy(message)}
          >
            <Copy />
            {copied && <span>Copied</span>}
          </button>
          <button
            type="button"
            className="message-icon-button"
            title="Edit"
            aria-label="Edit"
            disabled={isBusy || message.is_deleted || message.status === "streaming"}
            onClick={() => setIsEditing(true)}
          >
            <Pencil />
          </button>
          <button
            type="button"
            className="message-icon-button danger-button"
            title="Delete"
            aria-label="Delete"
            disabled={isBusy || message.status === "streaming"}
            onClick={() => onDelete(message)}
          >
            <Trash2 />
          </button>
        </div>
      )}
    </article>
  );
}

const markdownComponents = {
  pre({ children }) {
    return <CodeBlock>{children}</CodeBlock>;
  }
} satisfies Components;

function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="message-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={markdownComponents}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function CodeBlock({ children }: { children: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const resetCopiedRef = useRef<number | null>(null);
  const codeText = useMemo(() => textFromReactNode(children).replace(/\n$/, ""), [children]);

  useEffect(() => {
    return () => {
      if (resetCopiedRef.current) {
        window.clearTimeout(resetCopiedRef.current);
      }
    };
  }, []);

  async function copyCode() {
    if (!codeText) {
      return;
    }

    await navigator.clipboard.writeText(codeText);
    setCopied(true);

    if (resetCopiedRef.current) {
      window.clearTimeout(resetCopiedRef.current);
    }

    resetCopiedRef.current = window.setTimeout(() => {
      setCopied(false);
      resetCopiedRef.current = null;
    }, 1400);
  }

  return (
    <div className="code-block">
      <button
        type="button"
        className="code-copy-button"
        title="Copy code"
        aria-label="Copy code"
        disabled={!codeText}
        onClick={() => void copyCode()}
      >
        <Copy />
        <span>{copied ? "Copied" : "Copy"}</span>
      </button>
      <pre>{children}</pre>
    </div>
  );
}

function textFromReactNode(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }

  if (Array.isArray(node)) {
    return node.map(textFromReactNode).join("");
  }

  if (isValidElement<{ children?: ReactNode }>(node)) {
    return textFromReactNode(node.props.children);
  }

  return "";
}

function VersionSwitcher({
  versionInfo,
  isBusy,
  isGenerating
}: {
  versionInfo: VersionInfo;
  isBusy: boolean;
  isGenerating: boolean;
}) {
  const hidePrevious = !versionInfo.canPrevious;
  const hideNext = !versionInfo.canNext;

  return (
    <div className="version-switcher" aria-label="Message version selector">
      <button
        type="button"
        className={
          hidePrevious
            ? "message-icon-button version-arrow-hidden"
            : "message-icon-button"
        }
        title={hidePrevious ? undefined : "Previous version"}
        aria-label="Previous version"
        aria-hidden={hidePrevious}
        tabIndex={hidePrevious ? -1 : undefined}
        disabled={isBusy || isGenerating || hidePrevious}
        onClick={versionInfo.onPrevious}
      >
        <ChevronLeft />
      </button>
      <span className="version-count">
        {versionInfo.index + 1}/{versionInfo.total}
      </span>
      <button
        type="button"
        className={
          hideNext ? "message-icon-button version-arrow-hidden" : "message-icon-button"
        }
        title={hideNext ? undefined : "Next version"}
        aria-label="Next version"
        aria-hidden={hideNext}
        tabIndex={hideNext ? -1 : undefined}
        disabled={isBusy || isGenerating || hideNext}
        onClick={versionInfo.onNext}
      >
        <ChevronRight />
      </button>
    </div>
  );
}

function messageLabel(message: ChatMessage) {
  if (message.role === "assistant") {
    return message.model_name ?? "assistant";
  }

  return message.role;
}

function thinkingSummary(
  message: ChatMessage,
  thinkingDurationSeconds: number | null
): ReactNode {
  if (message.status === "streaming" && thinkingDurationSeconds === null) {
    return <ThinkingLoader />;
  }

  const durationSeconds = thinkingDurationSeconds ?? estimatedThinkingDurationSeconds(message);
  return `Thought for ${formatThoughtDuration(durationSeconds)}`;
}

function estimatedThinkingDurationSeconds(message: ChatMessage) {
  const startedAt = message.started_at ?? message.created_at;
  const endedAt = message.completed_at ?? message.updated_at;
  return Math.max(1, endedAt - startedAt);
}

function formatThoughtDuration(seconds: number) {
  const rounded = Math.max(1, Math.round(seconds));
  if (rounded < 60) {
    return `${rounded}s`;
  }

  const minutes = Math.floor(rounded / 60);
  const remainingSeconds = rounded % 60;
  return remainingSeconds === 0 ? `${minutes}m` : `${minutes}m ${remainingSeconds}s`;
}

function latestAssistantModelValue(messages: ChatMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && message.backend_id && message.model_name) {
      return modelValue(message.backend_id, message.model_name);
    }
  }

  return null;
}

function revisionsForMessage(message: ChatMessage) {
  const revisions = message.revisions.length
    ? message.revisions
    : message.active_revision
      ? [message.active_revision]
      : [];

  return [...revisions].sort(compareRevisionsByCreatedAt);
}

function updateRevisionList(
  revisions: ChatMessageRevision[],
  nextRevision: ChatMessageRevision
) {
  const nextRevisions = revisions.length ? [...revisions] : [];
  const revisionIndex = nextRevisions.findIndex((revision) => revision.id === nextRevision.id);
  if (revisionIndex >= 0) {
    nextRevisions[revisionIndex] = nextRevision;
  } else {
    nextRevisions.push(nextRevision);
  }

  return nextRevisions.sort(compareRevisionsByCreatedAt);
}

function activePathMessages(messages: ChatMessage[], activeRootMessageId: string | null) {
  const messagesById = new Map(messages.map((message) => [message.id, message]));
  const rootMessages = messages
    .filter((message) => !message.parent_message_id)
    .sort(compareMessagesByCreatedAt);
  const path: ChatMessage[] = [];
  const seen = new Set<string>();
  let currentId: string | null = activeRootMessageId ?? rootMessages[0]?.id ?? null;

  while (currentId && !seen.has(currentId)) {
    const message = messagesById.get(currentId);
    if (!message) {
      break;
    }

    path.push(message);
    seen.add(currentId);
    currentId = message.active_child_message_id;
  }

  return path;
}

function groupMessagesByParent(messages: ChatMessage[]) {
  const groups = new Map<string, ChatMessage[]>();

  for (const message of messages) {
    const key = parentGroupKey(message.parent_message_id);
    const siblings = groups.get(key) ?? [];
    siblings.push(message);
    groups.set(key, siblings);
  }

  for (const siblings of groups.values()) {
    siblings.sort(compareMessagesByCreatedAt);
  }

  return groups;
}

function parentGroupKey(parentMessageId: string | null) {
  return parentMessageId ?? rootSiblingGroupKey;
}

function compareMessagesByCreatedAt(left: ChatMessage, right: ChatMessage) {
  return left.created_at - right.created_at || left.id.localeCompare(right.id);
}

function compareRevisionsByCreatedAt(left: ChatMessageRevision, right: ChatMessageRevision) {
  return left.created_at - right.created_at || left.id.localeCompare(right.id);
}

function compareVersionsByCreatedAt(left: MessageVersion, right: MessageVersion) {
  return (
    left.revision.created_at - right.revision.created_at ||
    left.message.created_at - right.message.created_at ||
    left.revision.id.localeCompare(right.revision.id)
  );
}

function StartChatComposer({
  isBusy,
  isDisabled,
  isGenerating = false,
  placeholder,
  onStop,
  onSubmit
}: {
  isBusy: boolean;
  isDisabled: boolean;
  isGenerating?: boolean;
  placeholder: string;
  onStop?: () => void;
  onSubmit: (prompt: string) => Promise<void>;
}) {
  const [prompt, setPrompt] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const canSubmit = prompt.trim().length > 0 && !isDisabled && (!isBusy || isGenerating);

  useEffect(() => {
    textareaRef.current?.focus();
  }, [isGenerating]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }

    const submittedPrompt = prompt;
    setPrompt("");
    await onSubmit(submittedPrompt);
    window.requestAnimationFrame(() => textareaRef.current?.focus());
  }

  return (
    <form className="chat-composer" onSubmit={submit}>
      <textarea
        ref={textareaRef}
        rows={3}
        value={prompt}
        disabled={isDisabled || (isBusy && !isGenerating)}
        placeholder={placeholder}
        onChange={(event) => setPrompt(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            event.currentTarget.form?.requestSubmit();
          }
        }}
      />
      <div className="composer-actions">
        {isGenerating && (
          <button type="button" aria-label="Stop generation" onClick={onStop}>
            <Square />
          </button>
        )}
        <button type="submit" aria-label="Send message" disabled={!canSubmit}>
          {isBusy && !isGenerating ? <RetroLoader /> : <SendHorizontal />}
        </button>
      </div>
    </form>
  );
}

function ModelPicker({
  groups,
  isLoading,
  error,
  value,
  onChange
}: {
  groups: BackendModelGroup[];
  isLoading: boolean;
  error: string | null;
  value: string;
  onChange: (value: string) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const wrapRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const hasModels = groups.some((group) => group.models.length > 0);
  const selected = groups
    .flatMap((group) =>
      group.models.map((model) => ({
        backendId: group.backend.id,
        backendName: group.backend.name,
        model
      }))
    )
    .find((option) => modelValue(option.backendId, option.model.name) === value);
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const filteredGroups = groups
    .map((group) => ({
      backend: group.backend,
      models: group.models.filter((model) => {
        if (!normalizedQuery) {
          return true;
        }

        return (
          model.name.toLocaleLowerCase().includes(normalizedQuery) ||
          group.backend.name.toLocaleLowerCase().includes(normalizedQuery)
        );
      })
    }))
    .filter((group) => group.models.length > 0);

  useEffect(() => {
    if (isOpen) {
      window.setTimeout(() => inputRef.current?.focus(), 0);
    } else {
      setQuery("");
    }
  }, [isOpen]);

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

  function buttonLabel() {
    if (isLoading) {
      return "Loading models...";
    }

    if (error) {
      return "Models unavailable";
    }

    if (!hasModels) {
      return "No models";
    }

    return selected ? selected.model.name : "Select model";
  }

  return (
    <div className="model-picker" ref={wrapRef}>
      <button
        type="button"
        className="model-picker-button"
        disabled={isLoading || !hasModels}
        title={error ?? selected?.backendName}
        aria-expanded={isOpen}
        onClick={() => setIsOpen((open) => !open)}
      >
        <span>{buttonLabel()}</span>
      </button>
      {isOpen && (
        <div className="model-menu">
          <label className="model-search">
            <Search />
            <input
              ref={inputRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  setIsOpen(false);
                }
              }}
              placeholder="Search models"
            />
          </label>
          <div className="model-options">
            {filteredGroups.length === 0 ? (
              <p className="model-empty">No matching models</p>
            ) : (
              filteredGroups.map((group) => (
                <section key={group.backend.id} className="model-group">
                  <p>{group.backend.name}</p>
                  {group.models.map((model) => {
                    const optionValue = modelValue(group.backend.id, model.name);
                    return (
                      <button
                        type="button"
                        key={optionValue}
                        className={
                          optionValue === value
                            ? "model-option model-option-active"
                            : "model-option"
                        }
                        onClick={() => {
                          onChange(optionValue);
                          setIsOpen(false);
                        }}
                      >
                        <span>{model.name}</span>
                        {model.supports_images && <span className="model-vision">vision</span>}
                      </button>
                    );
                  })}
                </section>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function SettingsPage({
  currentUser,
  activeSection,
  onBackendsChanged,
  onAppSettingsGuardChange,
  onSelectSection,
  isAdmin
}: {
  currentUser: User;
  activeSection: SettingsSection;
  onBackendsChanged: () => Promise<void>;
  onAppSettingsGuardChange: (guard: AppSettingsGuard | null) => void;
  onSelectSection: (section: SettingsSection) => void;
  isAdmin: boolean;
}) {
  const sections: Array<{
    id: SettingsSection;
    label: string;
    icon: ReactNode;
    adminOnly?: boolean;
  }> = [
    { id: "profile", label: "Profile", icon: <UserRound /> },
    { id: "users", label: "Users", icon: <Users />, adminOnly: true },
    { id: "backends", label: "Backends", icon: <Server />, adminOnly: true },
    { id: "app", label: "App", icon: <SettingsIcon />, adminOnly: true }
  ];
  const visibleSections = sections.filter((section) => !section.adminOnly || isAdmin);
  const selectedSection =
    visibleSections.find((section) => section.id === activeSection)?.id ?? "profile";

  return (
    <div className="settings-page">
      <nav className="settings-nav" aria-label="Settings sections">
        {visibleSections.map((section) => (
          <button
            type="button"
            key={section.id}
            className={
              selectedSection === section.id ? "settings-tab settings-tab-active" : "settings-tab"
            }
            onClick={() => onSelectSection(section.id)}
          >
            {section.icon}
            <span>{section.label}</span>
          </button>
        ))}
      </nav>
      <section className="settings-content">
        {selectedSection === "profile" && <ProfileSettings user={currentUser} />}
        {selectedSection === "users" && isAdmin && <AdminUsersPanel currentUserId={currentUser.id} />}
        {selectedSection === "backends" && isAdmin && (
          <BackendsPanel onBackendsChanged={onBackendsChanged} />
        )}
        {selectedSection === "app" && <AppSettingsPanel onGuardChange={onAppSettingsGuardChange} />}
      </section>
    </div>
  );
}

function ProfileSettings({ user }: { user: User }) {
  return (
    <SettingsPlaceholder
      eyebrow="Account"
      title="Profile"
      text={`${user.username} is signed in as ${user.role}.`}
    />
  );
}

function SettingsPlaceholder({
  eyebrow,
  title,
  text
}: {
  eyebrow: string;
  title: string;
  text: string;
}) {
  return (
    <div className="settings-section">
      <p className="eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      <p className="status-message">{text}</p>
    </div>
  );
}

function AdminUsersPanel({ currentUserId }: { currentUserId: string }) {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busyUserId, setBusyUserId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<AdminUser | null>(null);

  const loadUsers = useCallback(async () => {
    setIsRefreshing(true);
    setError(null);

    try {
      const response = await requestJson<AdminUsersResponse>("/api/admin/users");
      setUsers(response.users);
      setHasLoaded(true);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load users");
      setHasLoaded(true);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadUsers();
  }, [loadUsers]);

  async function patchUser(userId: string, body: Partial<Pick<AdminUser, "role" | "is_disabled">>) {
    setBusyUserId(userId);
    setError(null);

    try {
      await requestJson(`/api/admin/users/${userId}`, {
        method: "PATCH",
        body: JSON.stringify(body)
      });
      await loadUsers();
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : "Failed to update user");
    } finally {
      setBusyUserId(null);
    }
  }

  async function deleteUser(userId: string) {
    setBusyUserId(userId);
    setError(null);

    try {
      await requestJson(`/api/admin/users/${userId}`, { method: "DELETE" });
      await loadUsers();
      setDeleteTarget(null);
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Failed to delete user");
      setDeleteTarget(null);
    } finally {
      setBusyUserId(null);
    }
  }

  const pendingUsers = users.filter((listedUser) => listedUser.is_disabled);
  const approvedUsers = users.filter((listedUser) => !listedUser.is_disabled);

  return (
    <div className="settings-section">
      <div className="section-header">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>Users</h1>
        </div>
        <button
          type="button"
          className="secondary-button refresh-button"
          onClick={() => void loadUsers()}
          disabled={isRefreshing}
        >
          {isRefreshing ? <RetroLoader /> : "Refresh"}
        </button>
      </div>

      {!hasLoaded && <p className="status-message">Loading users...</p>}
      {error && <p className="error">{error}</p>}
      {hasLoaded && users.length === 0 && <p className="status-message">No users found.</p>}

      <div className="user-list">
        <UserGroup
          title="Pending Approval"
          users={pendingUsers}
          emptyText="No pending users."
          currentUserId={currentUserId}
          busyUserId={busyUserId}
          onPatchUser={patchUser}
          onDeleteUser={setDeleteTarget}
        />
        <div className="user-divider" />
        <UserGroup
          title="Approved Users"
          users={approvedUsers}
          emptyText="No approved users."
          currentUserId={currentUserId}
          busyUserId={busyUserId}
          onPatchUser={patchUser}
          onDeleteUser={setDeleteTarget}
        />
      </div>
      {deleteTarget && (
        <ConfirmDialog
          title="Delete User"
          message={`Delete ${deleteTarget.username}? This removes the user, their sessions, and their settings.`}
          confirmLabel="Delete"
          isBusy={busyUserId === deleteTarget.id}
          onCancel={() => setDeleteTarget(null)}
          onConfirm={() => void deleteUser(deleteTarget.id)}
        />
      )}
    </div>
  );
}

function UserGroup({
  title,
  users,
  emptyText,
  currentUserId,
  busyUserId,
  onPatchUser,
  onDeleteUser
}: {
  title: string;
  users: AdminUser[];
  emptyText: string;
  currentUserId: string;
  busyUserId: string | null;
  onPatchUser: (
    userId: string,
    body: Partial<Pick<AdminUser, "role" | "is_disabled">>
  ) => Promise<void>;
  onDeleteUser: (user: AdminUser) => void;
}) {
  return (
    <section className="user-group">
      <div className="user-group-header">
        <h2>{title}</h2>
        <span>{users.length}</span>
      </div>
      {users.length === 0 ? (
        <p className="status-message">{emptyText}</p>
      ) : (
        users.map((listedUser) => (
          <UserRow
            key={listedUser.id}
            user={listedUser}
            isSelf={listedUser.id === currentUserId}
            isBusy={busyUserId === listedUser.id}
            onPatchUser={onPatchUser}
            onDeleteUser={onDeleteUser}
          />
        ))
      )}
    </section>
  );
}

function UserRow({
  user,
  isSelf,
  isBusy,
  onPatchUser,
  onDeleteUser
}: {
  user: AdminUser;
  isSelf: boolean;
  isBusy: boolean;
  onPatchUser: (
    userId: string,
    body: Partial<Pick<AdminUser, "role" | "is_disabled">>
  ) => Promise<void>;
  onDeleteUser: (user: AdminUser) => void;
}) {
  return (
    <article className="user-row">
      <div className="user-main">
        <div>
          <h2>{user.username}</h2>
          <p>{user.email ?? "No email"}</p>
        </div>
        <div className="badges">
          <span className="badge">{user.role}</span>
          <span className={user.is_disabled ? "badge badge-warning" : "badge"}>
            {user.is_disabled ? "pending" : "enabled"}
          </span>
          {isSelf && <span className="badge">you</span>}
        </div>
      </div>
      <div className="user-actions">
        {user.is_disabled ? (
          <button
            type="button"
            disabled={isBusy}
            onClick={() => void onPatchUser(user.id, { is_disabled: false })}
          >
            Approve
          </button>
        ) : (
          <button
            type="button"
            className="secondary-button"
            disabled={isBusy || isSelf}
            onClick={() => void onPatchUser(user.id, { is_disabled: true })}
          >
            Disable
          </button>
        )}
        {user.role === "admin" ? (
          <button
            type="button"
            className="secondary-button"
            disabled={isBusy || isSelf}
            onClick={() => void onPatchUser(user.id, { role: "user" })}
          >
            Make User
          </button>
        ) : (
          <button
            type="button"
            className="secondary-button"
            disabled={isBusy}
            onClick={() => void onPatchUser(user.id, { role: "admin" })}
          >
            Make Admin
          </button>
        )}
        <button
          type="button"
          className="danger-button"
          disabled={isBusy || isSelf}
          onClick={() => onDeleteUser(user)}
        >
          <Trash2 />
          <span>Delete</span>
        </button>
      </div>
    </article>
  );
}

function BackendsPanel({ onBackendsChanged }: { onBackendsChanged: () => Promise<void> }) {
  const [backends, setBackends] = useState<Backend[]>([]);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [busyBackendId, setBusyBackendId] = useState<string | null>(null);
  const [editingBackend, setEditingBackend] = useState<Backend | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Backend | null>(null);
  const [newName, setNewName] = useState("");
  const [newBaseUrl, setNewBaseUrl] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadBackends = useCallback(async () => {
    setIsRefreshing(true);
    setError(null);

    try {
      const response = await requestJson<BackendsResponse>("/api/backends");
      setBackends(response.backends);
      setHasLoaded(true);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load backends");
      setHasLoaded(true);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadBackends();
  }, [loadBackends]);

  async function createBackend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsCreating(true);
    setError(null);
    setStatus(null);

    try {
      await requestJson("/api/backends", {
        method: "POST",
        body: JSON.stringify({ name: newName, base_url: newBaseUrl })
      });
      setNewName("");
      setNewBaseUrl("");
      setStatus("Backend added.");
      await loadBackends();
      await onBackendsChanged();
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "Failed to add backend");
    } finally {
      setIsCreating(false);
    }
  }

  async function updateBackend(
    backendId: string,
    body: Partial<Pick<Backend, "name" | "base_url" | "is_enabled">>
  ) {
    setBusyBackendId(backendId);
    setError(null);
    setStatus(null);

    try {
      await requestJson(`/api/backends/${backendId}`, {
        method: "PATCH",
        body: JSON.stringify(body)
      });
      setEditingBackend(null);
      setStatus("Backend updated.");
      await loadBackends();
      await onBackendsChanged();
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : "Failed to update backend");
    } finally {
      setBusyBackendId(null);
    }
  }

  async function deleteBackend(backendId: string) {
    setBusyBackendId(backendId);
    setError(null);
    setStatus(null);

    try {
      await requestJson(`/api/backends/${backendId}`, { method: "DELETE" });
      setDeleteTarget(null);
      setStatus("Backend deleted.");
      await loadBackends();
      await onBackendsChanged();
    } catch (deleteError) {
      setDeleteTarget(null);
      setError(deleteError instanceof Error ? deleteError.message : "Failed to delete backend");
    } finally {
      setBusyBackendId(null);
    }
  }

  async function detectLocalhost() {
    setIsDetecting(true);
    setError(null);
    setStatus(null);

    try {
      const response = await requestJson<DetectLocalhostResponse>("/api/backends/detect-localhost", {
        method: "POST"
      });
      setStatus(
        response.detected.length === 0
          ? "No local Ollama backend found."
          : `Detected ${response.detected.map((backend) => backend.name).join(", ")}.`
      );
      await loadBackends();
      await onBackendsChanged();
    } catch (detectError) {
      setError(detectError instanceof Error ? detectError.message : "Localhost detection failed");
    } finally {
      setIsDetecting(false);
    }
  }

  async function scanLocalNetwork() {
    setIsScanning(true);
    setError(null);
    setStatus(null);

    try {
      const response = await requestJson<DetectLocalhostResponse>(
        "/api/backends/scan-local-network",
        {
          method: "POST"
        }
      );
      setStatus(
        response.detected.length === 0
          ? "No Ollama backends found on the local network."
          : `Detected ${response.detected.map((backend) => backend.name).join(", ")}.`
      );
      await loadBackends();
      await onBackendsChanged();
    } catch (scanError) {
      setError(scanError instanceof Error ? scanError.message : "Local network scan failed");
    } finally {
      setIsScanning(false);
    }
  }

  const hasLocalBackend = backends.some((backend) => isLocalBackend(backend.base_url));

  return (
    <div className="settings-section">
      <div className="section-header">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>Backends</h1>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="secondary-button refresh-button"
            onClick={() => void loadBackends()}
            disabled={isRefreshing}
          >
            {isRefreshing ? <RetroLoader /> : <RefreshCw />}
            <span>{isRefreshing ? "Loading" : "Refresh"}</span>
          </button>
          {!hasLocalBackend && (
            <button
              type="button"
              className="secondary-button"
              onClick={() => void detectLocalhost()}
              disabled={isDetecting}
            >
              {isDetecting ? <RetroLoader /> : <Search />}
              <span>{isDetecting ? "Detecting" : "Detect Localhost"}</span>
            </button>
          )}
          <button
            type="button"
            className="secondary-button"
            onClick={() => void scanLocalNetwork()}
            disabled={isScanning}
          >
            {isScanning ? <RetroLoader /> : <Server />}
            <span>{isScanning ? "Scanning" : "Scan Network"}</span>
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}
      {status && <p className="status-message">{status}</p>}

      <form className="settings-form backend-create-form" onSubmit={createBackend}>
        <label>
          <span>Name</span>
          <input
            required
            value={newName}
            onChange={(event) => setNewName(event.target.value)}
            placeholder="hostname"
          />
        </label>
        <label>
          <span>Base URL</span>
          <input
            required
            value={newBaseUrl}
            onChange={(event) => setNewBaseUrl(event.target.value)}
            placeholder="http://127.0.0.1:11434"
          />
        </label>
        <button type="submit" disabled={isCreating}>
          <Plus />
          <span>{isCreating ? "Adding..." : "Add Backend"}</span>
        </button>
      </form>

      {!hasLoaded && <p className="status-message">Loading backends...</p>}
      {hasLoaded && backends.length === 0 && (
        <p className="status-message">No Ollama backends configured.</p>
      )}

      <div className="backend-list">
        {backends.map((backend) =>
          editingBackend?.id === backend.id ? (
            <BackendEditRow
              key={backend.id}
              backend={editingBackend}
              isBusy={busyBackendId === backend.id}
              onCancel={() => setEditingBackend(null)}
              onSave={updateBackend}
            />
          ) : (
            <BackendRow
              key={backend.id}
              backend={backend}
              isBusy={busyBackendId === backend.id}
              onDelete={setDeleteTarget}
              onEdit={setEditingBackend}
              onToggle={(nextEnabled) =>
                void updateBackend(backend.id, { is_enabled: nextEnabled })
              }
            />
          )
        )}
      </div>

      {deleteTarget && (
        <ConfirmDialog
          title="Delete Backend"
          message={`Delete ${deleteTarget.name}? This cannot be undone.`}
          confirmLabel="Delete"
          isBusy={busyBackendId === deleteTarget.id}
          onCancel={() => setDeleteTarget(null)}
          onConfirm={() => void deleteBackend(deleteTarget.id)}
        />
      )}
    </div>
  );
}

function BackendRow({
  backend,
  isBusy,
  onDelete,
  onEdit,
  onToggle
}: {
  backend: Backend;
  isBusy: boolean;
  onDelete: (backend: Backend) => void;
  onEdit: (backend: Backend) => void;
  onToggle: (nextEnabled: boolean) => void;
}) {
  return (
    <article className="backend-row">
      <div className="backend-main">
        <div>
          <h2>{backend.name}</h2>
          <p>{backend.base_url}</p>
          {backend.last_error && <p className="backend-error">{backend.last_error}</p>}
        </div>
        <div className="badges">
          <span className={backend.is_enabled ? "badge" : "badge badge-warning"}>
            {backend.is_enabled ? "enabled" : "disabled"}
          </span>
          <span className={backend.last_health_status === "error" ? "badge badge-warning" : "badge"}>
            {backend.last_health_status ?? "unknown"}
          </span>
        </div>
      </div>
      <div className="backend-actions">
        <button
          type="button"
          className="secondary-button"
          disabled={isBusy}
          onClick={() => onToggle(!backend.is_enabled)}
        >
          <Power />
          <span>{backend.is_enabled ? "Disable" : "Enable"}</span>
        </button>
        <button
          type="button"
          className="secondary-button"
          disabled={isBusy}
          onClick={() => onEdit(backend)}
        >
          <Pencil />
          <span>Edit</span>
        </button>
        <button
          type="button"
          className="danger-button"
          disabled={isBusy}
          onClick={() => onDelete(backend)}
        >
          <Trash2 />
          <span>Delete</span>
        </button>
      </div>
    </article>
  );
}

function BackendEditRow({
  backend,
  isBusy,
  onCancel,
  onSave
}: {
  backend: Backend;
  isBusy: boolean;
  onCancel: () => void;
  onSave: (
    backendId: string,
    body: Partial<Pick<Backend, "name" | "base_url" | "is_enabled">>
  ) => Promise<void>;
}) {
  const [name, setName] = useState(backend.name);
  const [baseUrl, setBaseUrl] = useState(backend.base_url);
  const [isEnabled, setIsEnabled] = useState(backend.is_enabled);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await onSave(backend.id, { name, base_url: baseUrl, is_enabled: isEnabled });
  }

  return (
    <article className="backend-row">
      <form className="backend-edit-form" onSubmit={submit}>
        <div className="backend-edit-grid">
          <label>
            <span>Name</span>
            <input required value={name} onChange={(event) => setName(event.target.value)} />
          </label>
          <label>
            <span>Base URL</span>
            <input
              required
              value={baseUrl}
              onChange={(event) => setBaseUrl(event.target.value)}
            />
          </label>
        </div>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={isEnabled}
            onChange={(event) => setIsEnabled(event.target.checked)}
          />
          <span>Enabled</span>
        </label>
        <div className="backend-actions">
          <button type="button" className="secondary-button" disabled={isBusy} onClick={onCancel}>
            Cancel
          </button>
          <button type="submit" disabled={isBusy}>
            <Save />
            <span>{isBusy ? "Saving..." : "Save Backend"}</span>
          </button>
        </div>
      </form>
    </article>
  );
}

function AppSettingsPanel({
  onGuardChange
}: {
  onGuardChange: (guard: AppSettingsGuard | null) => void;
}) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [allowSignup, setAllowSignup] = useState(true);
  const [signupLimit, setSignupLimit] = useState(25);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSettings = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await requestJson<AppSettings>("/api/settings");
      setSettings(response);
      setAllowSignup(response.allow_signup);
      setSignupLimit(response.signup_limit);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load settings");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSettings();
  }, [loadSettings]);

  const isDirty = Boolean(
    settings && (settings.allow_signup !== allowSignup || settings.signup_limit !== signupLimit)
  );

  const saveSettingsDraft = useCallback(async () => {
    setIsSaving(true);
    setError(null);

    try {
      const response = await requestJson<AppSettings>("/api/settings", {
        method: "PATCH",
        body: JSON.stringify({
          allow_signup: allowSignup,
          signup_limit: signupLimit
        })
      });
      setSettings(response);
      setAllowSignup(response.allow_signup);
      setSignupLimit(response.signup_limit);
      return true;
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save settings");
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [allowSignup, signupLimit]);

  const discardSettingsDraft = useCallback(() => {
    if (!settings) {
      return;
    }

    setAllowSignup(settings.allow_signup);
    setSignupLimit(settings.signup_limit);
    setError(null);
  }, [settings]);

  useEffect(() => {
    onGuardChange({
      isDirty,
      save: saveSettingsDraft,
      discard: discardSettingsDraft
    });
  }, [discardSettingsDraft, isDirty, onGuardChange, saveSettingsDraft]);

  useEffect(() => () => onGuardChange(null), [onGuardChange]);

  async function saveSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await saveSettingsDraft();
  }

  return (
    <div className="settings-section">
      <div className="section-header">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>App Settings</h1>
        </div>
        <button
          type="button"
          className="secondary-button refresh-button"
          onClick={() => void loadSettings()}
          disabled={isLoading}
        >
          {isLoading ? <RetroLoader /> : "Refresh"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}
      {!settings && isLoading && <p className="status-message">Loading settings...</p>}
      {settings && (
        <form className="settings-form" onSubmit={saveSettings}>
          {isDirty && (
            <div className="info-box" role="status">
              <p className="eyebrow">Unsaved</p>
              <p>You have unsaved app settings.</p>
            </div>
          )}
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={allowSignup}
              onChange={(event) => setAllowSignup(event.target.checked)}
            />
            <span>Allow public account creation</span>
          </label>
          <label>
            <span>Signup limit</span>
            <input
              min="0"
              type="number"
              value={signupLimit}
              onChange={(event) => setSignupLimit(Number(event.target.value))}
            />
          </label>
          <p className="status-message">
            {settings.signup_count} public account signup
            {settings.signup_count === 1 ? "" : "s"} used.
          </p>
          <button type="submit" disabled={isSaving}>
            <Save />
            <span>{isSaving ? "Saving..." : "Save Settings"}</span>
          </button>
        </form>
      )}
    </div>
  );
}

function RetroLoader() {
  const frames = ["-______", "_-_____", "__-____", "___-___", "____-__", "_____-_", "______-"];
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const interval = window.setInterval(() => {
      setFrame((current) => (current + 1) % frames.length);
    }, 120);

    return () => window.clearInterval(interval);
  }, [frames.length]);

  return <span className="retro-loader">{frames[frame]}</span>;
}

function ThinkingLoader() {
  const frames = ["Thinking", "Thinking.", "Thinking..", "Thinking..."];
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const interval = window.setInterval(() => {
      setFrame((current) => (current + 1) % frames.length);
    }, 140);

    return () => window.clearInterval(interval);
  }, [frames.length]);

  return <span className="thinking-loader">{frames[frame]}</span>;
}

function ConfirmDialog({
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  isBusy = false,
  onCancel,
  onConfirm
}: {
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  isBusy?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && !isBusy) {
        onCancel();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isBusy, onCancel]);

  return (
    <div className="confirm-backdrop" role="presentation">
      <section
        className="confirm-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
      >
        <h2 id="confirm-dialog-title">{title}</h2>
        <p>{message}</p>
        <div className="dialog-actions">
          <button type="button" className="secondary-button" disabled={isBusy} onClick={onCancel}>
            {cancelLabel}
          </button>
          <button type="button" className="danger-button" disabled={isBusy} onClick={onConfirm}>
            {isBusy ? <RetroLoader /> : confirmLabel}
          </button>
        </div>
      </section>
    </div>
  );
}

function UnsavedSettingsDialog({
  isSaving,
  onCancel,
  onDiscard,
  onSave
}: {
  isSaving: boolean;
  onCancel: () => void;
  onDiscard: () => void;
  onSave: () => void;
}) {
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && !isSaving) {
        onCancel();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isSaving, onCancel]);

  return (
    <div className="confirm-backdrop" role="presentation">
      <section
        className="confirm-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="unsaved-settings-title"
      >
        <h2 id="unsaved-settings-title">Unsaved Settings</h2>
        <p>Save your app settings before leaving, or discard the unsaved changes.</p>
        <div className="dialog-actions">
          <button type="button" className="secondary-button" disabled={isSaving} onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="danger-button" disabled={isSaving} onClick={onDiscard}>
            Discard
          </button>
          <button type="button" disabled={isSaving} onClick={onSave}>
            {isSaving ? <RetroLoader /> : "Save"}
          </button>
        </div>
      </section>
    </div>
  );
}

function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <img
      className={compact ? "brand-logo brand-logo-compact" : "brand-logo"}
      src="/brand/vashti-logo.png"
      alt="Vashti"
    />
  );
}

async function requestJson<T = unknown>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    credentials: "include",
    headers: {
      "content-type": "application/json",
      ...options.headers
    },
    ...options
  });

  if (!response.ok) {
    let message = `Request failed with ${response.status}`;
    try {
      const payload = (await response.json()) as ApiError;
      message = payload.error?.message ?? message;
    } catch {
      // Keep the status-derived message when the body is not JSON.
    }
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

async function responseErrorMessage(response: Response) {
  let message = `Request failed with ${response.status}`;
  try {
    const payload = (await response.json()) as ApiError;
    message = payload.error?.message ?? message;
  } catch {
    // Keep the status-derived message when the body is not JSON.
  }
  return message;
}
