import {
  ChangeEvent,
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
  Brain,
  ChevronLeft,
  ChevronRight,
  Cog,
  Copy,
  FileText,
  Image as ImageIcon,
  LogOut,
  Lock,
  Menu,
  MessageSquare,
  MessageSquarePlus,
  MoreHorizontal,
  Paperclip,
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
  Volume2,
  Wrench,
  Users,
  X
} from "lucide-react";
import {
  createPrivatePersona,
  createPrivateChat,
  createPrivateMessage,
  deletePrivatePersona,
  deletePrivateChat,
  getCachedHostedChat,
  getCachedModelState,
  getPrivateChat,
  listPrivatePersonas,
  listPrivateChats,
  listPrivateMessages,
  privateId,
  renamePrivateChat,
  resetPrivateStorageUser,
  savePrivateChat,
  saveCachedHostedChat,
  saveCachedModelState,
  savePrivateMessage,
  savePrivateMessages,
  setPrivateStorageUser,
  updatePrivatePersona,
  unixTimestamp,
  type PrivateChatDetail,
  type PrivateChatMessage,
  type PrivateChatSummary,
  type PrivateChatMessageRevision,
  type PrivatePersona
} from "./privateChatStore";

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

type AdminUserMutationResponse = {
  user: AdminUser;
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
  supports_thinking?: boolean;
  capabilities?: string[];
};

type AdminModelInfo = ModelInfo & {
  is_enabled: boolean;
};

type BackendModelGroup = {
  backend: {
    id: string;
    name: string;
  };
  models: ModelInfo[];
};

type AdminBackendModelGroup = {
  backend: {
    id: string;
    name: string;
  };
  models: AdminModelInfo[];
};

type ModelsResponse = {
  backends: BackendModelGroup[];
};

type AdminModelsResponse = {
  backends: AdminBackendModelGroup[];
};

type PersonaVersion = {
  id: string;
  persona_id: string;
  version_number: number;
  display_name: string;
  avatar_attachment_id: string | null;
  base_backend_id: string;
  base_model_name: string;
  system_prompt: string;
  tool_policy_json: string | null;
  created_by_user_id: string | null;
  created_at: number;
};

type Persona = {
  id: string;
  owner_user_id: string | null;
  owner_username: string | null;
  visibility: string;
  lifecycle_state: string;
  current_version: PersonaVersion;
  is_owner: boolean;
  is_member: boolean;
  created_at: number;
  updated_at: number;
};

type PersonasResponse = {
  personas: Persona[];
};

type ModelPickerCache = {
  models: ModelsResponse;
  personas: PersonasResponse;
};

type PersonaMutationResponse = {
  persona: Persona;
};

type ChatSummary = {
  id: string;
  title: string;
  default_backend_id: string;
  backend_name: string;
  default_model_name: string;
  persona_id?: string | null;
  persona_version_id?: string | null;
  persona_name?: string | null;
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
  persona_id?: string | null;
  persona_version_id?: string | null;
  persona_name?: string | null;
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

type AttachmentInfo = {
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

type ImageOpenHandler = (attachment: AttachmentInfo, attachments?: AttachmentInfo[]) => void;

type ImageViewerState = {
  attachments: AttachmentInfo[];
  index: number;
};

type ComposerAttachment = AttachmentInfo & {
  status: "ready" | "uploaded" | "uploading" | "error";
  error?: string;
  file?: File;
  isExisting?: boolean;
};

type ComposerSubmitPayload = {
  prompt: string;
  attachments: ComposerAttachment[];
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
  active_revision: ChatMessageRevision | null;
  revisions: ChatMessageRevision[];
  revision_count: number;
  attachments: AttachmentInfo[];
};

type ListMessagesResponse = {
  active_root_message_id: string | null;
  messages: ChatMessage[];
};

type ChatSyncResponse = {
  changed: boolean;
  chat: ChatDetail;
  active_root_message_id: string | null;
  messages: ChatMessage[] | null;
};

type MessageResponse = {
  message: ChatMessage;
};

type AttachmentResponse = {
  attachment: AttachmentInfo;
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

type Page = "chat" | "private-chat" | "settings";
type SettingsSection = "profile" | "personas" | "users" | "models" | "app" | "backends";
type NewChatMode = "standard" | "private";
type AppRoute =
  | { page: "chat"; chatId?: string }
  | { page: "private-chat"; chatId: string }
  | { page: "settings"; section: SettingsSection };
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

const settingsSections: SettingsSection[] = [
  "profile",
  "personas",
  "users",
  "backends",
  "models",
  "app"
];
const rootSiblingGroupKey = "__root__";
const newChatModeStorageKey = "vashti:new-chat-mode";

function isSettingsSection(value: string | undefined): value is SettingsSection {
  return settingsSections.includes(value as SettingsSection);
}

function storedNewChatMode(): NewChatMode {
  try {
    return window.localStorage.getItem(newChatModeStorageKey) === "private"
      ? "private"
      : "standard";
  } catch {
    return "standard";
  }
}

function privateStreamTestEnabled() {
  try {
    return (
      new URLSearchParams(window.location.search).has("privateStreamTest") ||
      window.localStorage.getItem("vashti:private-stream-test") === "1"
    );
  } catch {
    return false;
  }
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

  if (path.startsWith("/app/private-chats/")) {
    const chatId = path.split("/")[3];
    if (chatId) {
      return { page: "private-chat", chatId };
    }
  }

  return { page: "chat" };
}

function pathForRoute(route: AppRoute) {
  if (route.page === "settings") {
    return `/app/settings/${route.section}`;
  }

  if (route.page === "private-chat") {
    return `/app/private-chats/${route.chatId}`;
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

function enabledModelValueSet(groups: BackendModelGroup[]) {
  return new Set(
    groups.flatMap((group) => group.models.map((model) => modelValue(group.backend.id, model.name)))
  );
}

function personaBaseModelValue(persona: {
  current_version: { base_backend_id: string; base_model_name: string };
}) {
  return modelValue(
    persona.current_version.base_backend_id,
    persona.current_version.base_model_name
  );
}

function personaModelValue(personaVersionId: string) {
  return `persona:${personaVersionId}`;
}

function privatePersonaModelValue(personaVersionId: string) {
  return `private-persona:${personaVersionId}`;
}

function personaVersionIdFromValue(value: string) {
  return value.startsWith("persona:") ? value.slice("persona:".length) : null;
}

function privatePersonaVersionIdFromValue(value: string) {
  return value.startsWith("private-persona:")
    ? value.slice("private-persona:".length)
    : null;
}

function isPrivatePersonaVersionId(value: string) {
  return value.startsWith("private-persona-version-");
}

function modelParts(value: string) {
  if (personaVersionIdFromValue(value) || privatePersonaVersionIdFromValue(value)) {
    return null;
  }

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
  setPrivateStorageUser(user.id);

  const [route, setRoute] = useState<AppRoute>(() => routeFromLocation());
  const routeRef = useRef(route);
  const appSettingsGuardRef = useRef<AppSettingsGuard | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSettingsMenuOpen, setIsSettingsMenuOpen] = useState(false);
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const [pendingNavigation, setPendingNavigation] = useState<AppRoute | null>(null);
  const [isSavingPendingNavigation, setIsSavingPendingNavigation] = useState(false);
  const [modelGroups, setModelGroups] = useState<BackendModelGroup[]>([]);
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [privatePersonas, setPrivatePersonas] = useState<PrivatePersona[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [privateChats, setPrivateChats] = useState<PrivateChatSummary[]>([]);
  const [isLoadingChats, setIsLoadingChats] = useState(false);
  const [isLoadingPrivateChats, setIsLoadingPrivateChats] = useState(false);
  const [newChatMode, setNewChatModeState] = useState<NewChatMode>(() => storedNewChatMode());
  const [isCreatingChat, setIsCreatingChat] = useState(false);
  const [isCreatingPrivateChat, setIsCreatingPrivateChat] = useState(false);
  const [chatDeleteTarget, setChatDeleteTarget] = useState<ChatSummary | null>(null);
  const [privateChatDeleteTarget, setPrivateChatDeleteTarget] =
    useState<PrivateChatSummary | null>(null);
  const [isDeletingChat, setIsDeletingChat] = useState(false);
  const [isDeletingPrivateChat, setIsDeletingPrivateChat] = useState(false);
  const [imageViewer, setImageViewerState] = useState<ImageViewerState | null>(null);
  const [queuedPrompt, setQueuedPrompt] = useState<
    ({ chatId: string } & ComposerSubmitPayload) | null
  >(null);
  const [queuedPrivatePrompt, setQueuedPrivatePrompt] = useState<
    ({ chatId: string } & ComposerSubmitPayload) | null
  >(null);
  const [error, setError] = useState<string | null>(null);
  const isAdmin = user.role === "admin";
  const page = route.page;
  const isSettingsPage = page === "settings";
  const settingsSection = route.page === "settings" ? route.section : "profile";
  const currentChatId = route.page === "chat" ? route.chatId ?? null : null;
  const currentPrivateChatId = route.page === "private-chat" ? route.chatId : null;
  const allowPrivatePersonaSelection =
    page === "private-chat" || (page === "chat" && !currentChatId && newChatMode === "private");
  const imageViewerRef = useRef<ImageViewerState | null>(null);

  useEffect(() => {
    routeRef.current = route;
  }, [route]);

  useEffect(() => {
    function updateAppHeight() {
      const nextHeight = window.visualViewport?.height ?? window.innerHeight;
      document.documentElement.style.setProperty("--app-height", `${Math.round(nextHeight)}px`);
    }

    updateAppHeight();
    window.addEventListener("resize", updateAppHeight);
    window.addEventListener("orientationchange", updateAppHeight);
    window.visualViewport?.addEventListener("resize", updateAppHeight);
    window.visualViewport?.addEventListener("scroll", updateAppHeight);

    return () => {
      window.removeEventListener("resize", updateAppHeight);
      window.removeEventListener("orientationchange", updateAppHeight);
      window.visualViewport?.removeEventListener("resize", updateAppHeight);
      window.visualViewport?.removeEventListener("scroll", updateAppHeight);
    };
  }, []);

  function setImageViewer(viewer: ImageViewerState | null) {
    imageViewerRef.current = viewer;
    setImageViewerState(viewer);
  }

  useEffect(() => {
    if (route.page === "private-chat") {
      setNewChatMode("private");
    } else if (route.page === "chat" && route.chatId) {
      setNewChatMode("standard");
    }
  }, [route]);

  const updateAppSettingsGuard = useCallback((guard: AppSettingsGuard | null) => {
    appSettingsGuardRef.current = guard;
  }, []);

  function setNewChatMode(mode: NewChatMode) {
    setNewChatModeState(mode);
    try {
      window.localStorage.setItem(newChatModeStorageKey, mode);
    } catch {
      // The toggle still works for this session if browser storage is unavailable.
    }
  }

  const applyModelPickerData = useCallback(
    (modelsResponse: ModelsResponse, personasResponse: PersonasResponse) => {
      const enabledValues = enabledModelValueSet(modelsResponse.backends);
      const visiblePersonas = personasResponse.personas.filter((persona) =>
        enabledValues.has(personaBaseModelValue(persona))
      );
      setModelGroups(modelsResponse.backends);
      setPersonas(visiblePersonas);
      setSelectedModel((current) => {
        const values = [
          ...modelsResponse.backends.flatMap((group) =>
            group.models.map((model) => modelValue(group.backend.id, model.name))
          ),
          ...visiblePersonas.map((persona) => personaModelValue(persona.current_version.id))
        ];

        return current &&
          (values.includes(current) || Boolean(privatePersonaVersionIdFromValue(current)))
          ? current
          : values[0] ?? "";
      });
    },
    []
  );

  const loadModels = useCallback(async () => {
    setIsLoadingModels(true);
    setModelError(null);
    let displayedCachedModels = false;

    try {
      const cached = await getCachedModelState<ModelPickerCache>();
      if (cached) {
        applyModelPickerData(cached.models, cached.personas);
        displayedCachedModels = true;
        setIsLoadingModels(false);
      }
    } catch {
      // Model cache is an optimization. The live server fetch below remains authoritative.
    }

    try {
      const [modelsResponse, personasResponse] = await Promise.all([
        requestJson<ModelsResponse>("/api/models"),
        requestJson<PersonasResponse>("/api/personas")
      ]);
      applyModelPickerData(modelsResponse, personasResponse);
      await saveCachedModelState<ModelPickerCache>({
        models: modelsResponse,
        personas: personasResponse
      }).catch(() => undefined);
    } catch (loadError) {
      if (!displayedCachedModels) {
        setModelGroups([]);
        setPersonas([]);
        setSelectedModel("");
      }
      setModelError(loadError instanceof Error ? loadError.message : "Failed to load models");
    } finally {
      setIsLoadingModels(false);
    }
  }, [applyModelPickerData]);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  const loadPrivatePersonas = useCallback(async () => {
    try {
      setPrivatePersonas(await listPrivatePersonas());
    } catch (loadError) {
      setError(
        loadError instanceof Error
          ? loadError.message
          : "Failed to load private personas"
      );
    }
  }, []);

  useEffect(() => {
    void loadPrivatePersonas();
  }, [loadPrivatePersonas]);

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

  const loadPrivateChats = useCallback(async () => {
    setIsLoadingPrivateChats(true);

    try {
      setPrivateChats(await listPrivateChats());
    } catch (loadError) {
      setError(
        loadError instanceof Error ? loadError.message : "Failed to load private chats"
      );
    } finally {
      setIsLoadingPrivateChats(false);
    }
  }, []);

  useEffect(() => {
    void loadPrivateChats();
  }, [loadPrivateChats]);

  useEffect(() => {
    function handlePopState() {
      if (imageViewerRef.current) {
        setImageViewer(null);
        return;
      }

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

  function openImageViewer(attachment: AttachmentInfo, attachments: AttachmentInfo[] = [attachment]) {
    const imageAttachments = attachments.filter(isImageAttachment);
    const viewerAttachments = imageAttachments.length > 0 ? imageAttachments : [attachment];
    const index = Math.max(
      0,
      viewerAttachments.findIndex((viewerAttachment) => viewerAttachment.id === attachment.id)
    );
    setImageViewer({ attachments: viewerAttachments, index });
    if (!window.history.state?.vashtiImageViewer) {
      window.history.pushState(
        {
          ...(typeof window.history.state === "object" && window.history.state !== null
            ? window.history.state
            : {}),
          vashtiImageViewer: true
        },
        "",
        window.location.href
      );
    }
  }

  function closeImageViewer() {
    const shouldStepBack = Boolean(window.history.state?.vashtiImageViewer);
    setImageViewer(null);
    if (shouldStepBack) {
      window.history.back();
    }
  }

  function setImageViewerIndex(index: number) {
    const current = imageViewerRef.current;
    if (!current) {
      return;
    }

    setImageViewer({
      ...current,
      index: Math.min(Math.max(index, 0), current.attachments.length - 1)
    });
  }

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

  function openPrivateChat(chatId: string) {
    navigate({ page: "private-chat", chatId });
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
      resetPrivateStorageUser();
      applyNavigation({ page: "chat" });
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

  async function createChatFromPrompt(prompt: string, attachments: ComposerAttachment[] = []) {
    if (!prompt.trim()) {
      openChat();
      return;
    }

    const selected = selectedModelBaseParts(modelGroups, personas, [], selectedModel);
    if (!selected) {
      setError("Select a model before starting a chat");
      return;
    }
    const selectedPersonaVersionId = personaVersionIdFromValue(selectedModel);

    setIsCreatingChat(true);
    setError(null);

    try {
      const response = await requestJson<ChatResponse>("/api/chats", {
        method: "POST",
        body: JSON.stringify({
          title: "New Chat",
          default_backend_id: selected.backendId,
          default_model_name: selected.modelName,
          persona_version_id: selectedPersonaVersionId
        })
      });

      if (prompt.trim()) {
        const uploadedAttachments = await uploadComposerAttachments(response.chat.id, attachments);
        setQueuedPrompt({ chatId: response.chat.id, prompt, attachments: uploadedAttachments });
      }

      await loadChats();
      openChat(response.chat.id);
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "Failed to create chat");
    } finally {
      setIsCreatingChat(false);
    }
  }

  function currentSelectedModel() {
    const selectedPrivatePersona = privatePersonaForValue(privatePersonas, selectedModel);
    if (selectedPrivatePersona) {
      return {
        backendId: selectedPrivatePersona.current_version.base_backend_id,
        backendName: selectedPrivatePersona.current_version.base_backend_name,
        modelName: selectedPrivatePersona.current_version.base_model_name
      };
    }

    if (personaVersionIdFromValue(selectedModel)) {
      return null;
    }

    const selected = modelParts(selectedModel);
    if (!selected) {
      return null;
    }

    const backend = modelGroups.find((group) => group.backend.id === selected.backendId);
    if (!backend) {
      return null;
    }

    return {
      backendId: selected.backendId,
      backendName: backend.backend.name,
      modelName: selected.modelName
    };
  }

  function selectedModelInfo() {
    return modelInfoForValue(modelGroups, personas, privatePersonas, selectedModel);
  }

  async function createPrivateChatFromPrompt(prompt: string, attachments: ComposerAttachment[] = []) {
    if (!prompt.trim()) {
      openChat();
      return;
    }

    if (personaVersionIdFromValue(selectedModel)) {
      setError("Copy this hosted persona to your device before using it in a private chat");
      return;
    }

    const selected = currentSelectedModel();
    const selectedPrivatePersona = privatePersonaForValue(privatePersonas, selectedModel);
    if (!selected) {
      setError("Select a model before starting a private chat");
      return;
    }

    setIsCreatingPrivateChat(true);
    setError(null);

    try {
      const chat = await createPrivateChat({
        title: "Private Chat",
        backendId: selected.backendId,
        backendName: selected.backendName,
        modelName: selected.modelName,
        personaId: selectedPrivatePersona?.id ?? null,
        personaVersionId: selectedPrivatePersona?.current_version.id ?? null,
        personaName: selectedPrivatePersona?.current_version.display_name ?? null
      });

      if (prompt.trim()) {
        setQueuedPrivatePrompt({ chatId: chat.id, prompt, attachments });
      }

      await loadPrivateChats();
      openPrivateChat(chat.id);
    } catch (createError) {
      setError(
        createError instanceof Error ? createError.message : "Failed to create private chat"
      );
    } finally {
      setIsCreatingPrivateChat(false);
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

  async function renameLocalPrivateChat(chatId: string, title: string) {
    setError(null);
    try {
      const chat = await renamePrivateChat(chatId, title);
      setPrivateChats((current) =>
        current.map((item) => (item.id === chatId ? { ...item, title: chat.title } : item))
      );
    } catch (renameError) {
      setError(
        renameError instanceof Error ? renameError.message : "Failed to rename private chat"
      );
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

  async function deleteSelectedPrivateChat() {
    const chat = privateChatDeleteTarget;
    if (!chat) {
      return;
    }

    setIsDeletingPrivateChat(true);
    setError(null);
    try {
      await deletePrivateChat(chat.id);
      setPrivateChats((current) => current.filter((item) => item.id !== chat.id));
      setPrivateChatDeleteTarget(null);
      if (currentPrivateChatId === chat.id) {
        openChat();
      }
    } catch (deleteError) {
      setError(
        deleteError instanceof Error ? deleteError.message : "Failed to delete private chat"
      );
    } finally {
      setIsDeletingPrivateChat(false);
    }
  }

  return (
    <main className={isSidebarOpen ? "app-shell sidebar-open" : "app-shell"}>
      <Sidebar
        chats={chats}
        privateChats={privateChats}
        currentChatId={currentChatId}
        currentPrivateChatId={currentPrivateChatId}
        currentPage={page}
        isOpen={isSidebarOpen}
        isLoading={isLoadingChats}
        isLoadingPrivateChats={isLoadingPrivateChats}
        onClose={() => setIsSidebarOpen(false)}
        onDeleteChat={setChatDeleteTarget}
        onDeletePrivateChat={setPrivateChatDeleteTarget}
        onOpenChat={openChat}
        onOpenPrivateChat={openPrivateChat}
        onRenameChat={renameChat}
        onRenamePrivateChat={renameLocalPrivateChat}
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
              personas={allowPrivatePersonaSelection ? [] : personas}
              privatePersonas={allowPrivatePersonaSelection ? privatePersonas : []}
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
              onClick={() => openChat()}
            >
              <MessageSquarePlus />
              <span>New Chat</span>
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
            onPersonasChanged={loadModels}
            onPrivatePersonasChanged={loadPrivatePersonas}
            onAppSettingsGuardChange={updateAppSettingsGuard}
            onSelectSection={(section) => openSettings(section)}
            isAdmin={isAdmin}
          />
        ) : page === "private-chat" && currentPrivateChatId ? (
          <PrivateChatView
            chatId={currentPrivateChatId}
            error={error}
            queuedPrompt={
              queuedPrivatePrompt?.chatId === currentPrivateChatId
                ? queuedPrivatePrompt.prompt
                : null
            }
            queuedAttachments={
              queuedPrivatePrompt?.chatId === currentPrivateChatId
                ? queuedPrivatePrompt.attachments
                : []
            }
            selectedModel={selectedModel}
            selectedModelInfo={selectedModelInfo()}
            privatePersonas={privatePersonas}
            onImageOpen={openImageViewer}
            onModelSelected={setSelectedModel}
            onPrivateChatsChanged={loadPrivateChats}
            onQueuedPromptConsumed={() => setQueuedPrivatePrompt(null)}
          />
        ) : (
          currentChatId ? (
            <ChatView
              chatId={currentChatId}
              error={error}
              queuedPrompt={queuedPrompt?.chatId === currentChatId ? queuedPrompt.prompt : null}
              queuedAttachments={
                queuedPrompt?.chatId === currentChatId ? queuedPrompt.attachments : []
              }
              selectedModel={selectedModel}
              selectedModelInfo={selectedModelInfo()}
              personas={personas}
              onChatsChanged={loadChats}
              onImageOpen={openImageViewer}
              onModelSelected={setSelectedModel}
              onQueuedPromptConsumed={() => setQueuedPrompt(null)}
            />
          ) : (
            <ChatHome
              error={error}
              isCreating={isCreatingChat}
              isCreatingPrivate={isCreatingPrivateChat}
              mode={newChatMode}
              selectedModel={
                newChatMode === "private" && personaVersionIdFromValue(selectedModel)
                  ? ""
                  : selectedModel
              }
              selectedModelInfo={
                newChatMode === "private" && personaVersionIdFromValue(selectedModel)
                  ? null
                  : selectedModelInfo()
              }
              onModeChange={setNewChatMode}
              onCreateChat={createChatFromPrompt}
              onCreatePrivateChat={createPrivateChatFromPrompt}
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
      {privateChatDeleteTarget && (
        <ConfirmDialog
          title="Delete Private Chat"
          message={`Delete "${privateChatDeleteTarget.title}" from this device? This cannot be recovered from the server.`}
          confirmLabel="Delete"
          isBusy={isDeletingPrivateChat}
          onCancel={() => setPrivateChatDeleteTarget(null)}
          onConfirm={() => void deleteSelectedPrivateChat()}
        />
      )}
      {imageViewer && (
        <ImageViewer
          attachments={imageViewer.attachments}
          index={imageViewer.index}
          onClose={closeImageViewer}
          onIndexChange={setImageViewerIndex}
        />
      )}
    </main>
  );
}

function Sidebar({
  chats,
  privateChats,
  currentChatId,
  currentPrivateChatId,
  currentPage,
  isOpen,
  isLoading,
  isLoadingPrivateChats,
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
            {combinedChats.map(({ chat, isPrivate }) => (
              <ChatListItem
                key={chat.id}
                chat={chat}
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
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}

function ChatListItem({
  chat,
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
  isCreatingPrivate,
  mode,
  selectedModel,
  selectedModelInfo,
  onModeChange,
  onCreateChat,
  onCreatePrivateChat
}: {
  error: string | null;
  isCreating: boolean;
  isCreatingPrivate: boolean;
  mode: NewChatMode;
  selectedModel: string;
  selectedModelInfo: ModelInfo | null;
  onModeChange: (mode: NewChatMode) => void;
  onCreateChat: (prompt: string, attachments?: ComposerAttachment[]) => Promise<void>;
  onCreatePrivateChat: (prompt: string, attachments?: ComposerAttachment[]) => Promise<void>;
}) {
  const isPrivate = mode === "private";
  const isCreatingSelectedMode = isPrivate ? isCreatingPrivate : isCreating;

  return (
    <div className="chat-home">
      <div className="chat-home-inner">
        <BrandMark compact />
        <div className="new-chat-mode" aria-label="New chat mode">
          <button
            type="button"
            className={!isPrivate ? "new-chat-mode-option active" : "new-chat-mode-option"}
            onClick={() => onModeChange("standard")}
          >
            <MessageSquare />
            <span>Standard</span>
          </button>
          <button
            type="button"
            className={isPrivate ? "new-chat-mode-option active" : "new-chat-mode-option"}
            onClick={() => onModeChange("private")}
          >
            <Lock />
            <span>Private</span>
          </button>
        </div>
        <StartChatComposer
          isBusy={isCreatingSelectedMode}
          isDisabled={!selectedModel}
          placeholder={
            selectedModel
              ? isPrivate
                ? "Message private chat"
                : "Message Vashti"
              : "Select a model to start"
          }
          selectedModelInfo={selectedModelInfo}
          onUploadAttachment={isPrivate ? preparePrivateAttachment : prepareLocalAttachment}
          onSubmit={isPrivate ? onCreatePrivateChat : onCreateChat}
        />
        <p className={isPrivate ? "chat-mode-note private" : "chat-mode-note"}>
          {isPrivate ? <Lock /> : <MessageSquare />}
          <span>
            {isPrivate
              ? "Private chats are stored only on this device and are not synced."
              : "Standard chats are saved on the server and available when you sign in."}
          </span>
        </p>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}

function ChatView({
  chatId,
  error,
  queuedPrompt,
  queuedAttachments,
  selectedModel,
  selectedModelInfo,
  personas,
  onChatsChanged,
  onImageOpen,
  onModelSelected,
  onQueuedPromptConsumed
}: {
  chatId: string;
  error: string | null;
  queuedPrompt: string | null;
  queuedAttachments: ComposerAttachment[];
  selectedModel: string;
  selectedModelInfo: ModelInfo | null;
  personas: Persona[];
  onChatsChanged: () => Promise<void>;
  onImageOpen: ImageOpenHandler;
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
  const [pendingPrompt, setPendingPrompt] = useState<ComposerSubmitPayload | null>(null);
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
  const chatContainsImages = useMemo(
    () => messages.some((message) => activeMessageAttachments(message).some(isImageAttachment)),
    [messages]
  );
  const modelImageWarning =
    selectedModelInfo && !selectedModelInfo.supports_images && chatContainsImages
      ? "This chat includes images. Images may not be supported by this model."
      : null;

  const applyLoadedChat = useCallback(
    (nextChat: ChatDetail, activeRootMessageId: string | null, nextMessages: ChatMessage[]) => {
      setChat({
        ...nextChat,
        active_root_message_id: activeRootMessageId
      });
      thinkingStartedAtRef.current.clear();
      setThinkingDurations({});
      setMessages(nextMessages);
      const streamingAssistantId = streamingAssistantIdFromMessages(nextMessages);
      setActiveAssistantId(streamingAssistantId);
      setIsGenerating(Boolean(streamingAssistantId));
      const latestModel = latestAssistantModelValue(
        activePathMessages(nextMessages, activeRootMessageId)
      );
      onModelSelected(
        latestModel ??
          (nextChat.persona_version_id ? personaModelValue(nextChat.persona_version_id) : null) ??
          modelValue(nextChat.default_backend_id, nextChat.default_model_name)
      );
    },
    [onModelSelected]
  );

  const loadChat = useCallback(async () => {
    scrollToBottomAfterLoadRef.current = true;
    setIsLoading(true);
    setLoadError(null);
    let cachedChat: {
      chat: ChatDetail;
      active_root_message_id: string | null;
      messages: ChatMessage[];
      updated_at: number;
    } | null = null;
    let displayedCachedChat = false;

    try {
      cachedChat = await getCachedHostedChat<ChatDetail, ChatMessage>(chatId);
      if (cachedChat) {
        applyLoadedChat(
          cachedChat.chat,
          cachedChat.active_root_message_id,
          cachedChat.messages
        );
        displayedCachedChat = true;
        setIsLoading(false);
      }
    } catch {
      // Hosted chat cache is best-effort. The server remains authoritative.
    }

    try {
      const syncUrl =
        cachedChat?.updated_at === undefined
          ? `/api/chats/${chatId}/sync`
          : `/api/chats/${chatId}/sync?known_updated_at=${cachedChat.updated_at}`;
      const syncResponse = await requestJson<ChatSyncResponse>(syncUrl);
      if (!syncResponse.changed && cachedChat) {
        applyLoadedChat(
          syncResponse.chat,
          syncResponse.active_root_message_id,
          cachedChat.messages
        );
        return;
      }

      const nextMessages = syncResponse.messages;
      if (!nextMessages) {
        throw new Error("Chat sync did not include messages");
      }

      scrollToBottomAfterLoadRef.current = true;
      applyLoadedChat(
        syncResponse.chat,
        syncResponse.active_root_message_id,
        nextMessages
      );
      await saveCachedHostedChat<ChatDetail, ChatMessage>({
        chat: {
          ...syncResponse.chat,
          active_root_message_id: syncResponse.active_root_message_id
        },
        active_root_message_id: syncResponse.active_root_message_id,
        messages: nextMessages
      }).catch(() => undefined);
    } catch (chatError) {
      if (!displayedCachedChat) {
        setLoadError(chatError instanceof Error ? chatError.message : "Failed to load chat");
      }
    } finally {
      setIsLoading(false);
    }
  }, [applyLoadedChat, chatId]);

  const refreshStreamingMessages = useCallback(async () => {
    try {
      const messageResponse = await requestJson<ListMessagesResponse>(
        `/api/chats/${chatId}/messages`
      );
      setChat((current) =>
        current
          ? { ...current, active_root_message_id: messageResponse.active_root_message_id }
          : current
      );
      setMessages(messageResponse.messages);

      const streamingAssistantId = streamingAssistantIdFromMessages(messageResponse.messages);
      setActiveAssistantId(streamingAssistantId);
      setIsGenerating(Boolean(streamingAssistantId));

      if (!streamingAssistantId) {
        const chatResponse = await requestJson<ChatResponse>(`/api/chats/${chatId}`);
        const nextChat = {
          ...chatResponse.chat,
          active_root_message_id: messageResponse.active_root_message_id
        };
        setChat(nextChat);
        await saveCachedHostedChat<ChatDetail, ChatMessage>({
          chat: nextChat,
          active_root_message_id: messageResponse.active_root_message_id,
          messages: messageResponse.messages
        }).catch(() => undefined);
        await onChatsChanged();
      }
    } catch (refreshError) {
      setGenerationError(
        refreshError instanceof Error ? refreshError.message : "Failed to refresh generation"
      );
    }
  }, [chatId, onChatsChanged]);

  useEffect(() => {
    if (!chat || isLoading || isGenerating) {
      return;
    }

    void saveCachedHostedChat<ChatDetail, ChatMessage>({
      chat,
      active_root_message_id: chat.active_root_message_id,
      messages
    }).catch(() => undefined);
  }, [chat, isGenerating, isLoading, messages]);

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
    async (prompt: string, attachments: ComposerAttachment[] = []) => {
      if (isGenerating) {
        return;
      }

      const selected =
        selectedModelBaseParts([], personas, [], selectedModel) ?? modelParts(selectedModel);
      const personaVersionId = personaVersionIdFromValue(selectedModel);
      await streamAssistantResponse(`/api/chats/${chatId}/generate`, {
        user_message: { content_text: prompt },
        backend_id: selected?.backendId ?? null,
        model_name: selected?.modelName ?? null,
        persona_version_id: personaVersionId,
        think_mode: null,
        attachments: attachmentReferences(attachments)
      });
    },
    [chatId, isGenerating, personas, selectedModel, streamAssistantResponse]
  );

  useEffect(() => {
    void loadChat();
  }, [loadChat]);

  useEffect(() => {
    const latestModel = latestAssistantModelValue(visibleMessages);
    if (latestModel) {
      onModelSelected(latestModel);
    }
  }, [onModelSelected, visibleMessages]);

  useEffect(() => {
    return () => {
      if (userScrollIntentTimeoutRef.current) {
        window.clearTimeout(userScrollIntentTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const hasStreamingMessage = messages.some(
      (message) => message.role === "assistant" && message.status === "streaming"
    );
    if (!hasStreamingMessage || abortRef.current) {
      return;
    }

    const interval = window.setInterval(() => {
      void refreshStreamingMessages();
    }, 1000);

    return () => window.clearInterval(interval);
  }, [messages, refreshStreamingMessages]);

  useEffect(() => {
    if (!queuedPrompt || isLoading || !chat || isGenerating) {
      return;
    }

    onQueuedPromptConsumed();
    void generate(queuedPrompt, queuedAttachments);
  }, [
    chat,
    generate,
    isGenerating,
    isLoading,
    onQueuedPromptConsumed,
    queuedAttachments,
    queuedPrompt
  ]);

  useEffect(() => {
    if (!pendingPrompt || isGenerating || isLoading || !chat) {
      return;
    }

    const prompt = pendingPrompt;
    setPendingPrompt(null);
    void generate(prompt.prompt, prompt.attachments);
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

    const messageList = list;
    const lastMessage = visibleMessages[visibleMessages.length - 1];
    const lastMessageElement = lastMessage
      ? messageList.querySelector<HTMLElement>(`[data-message-id="${lastMessage.id}"]`)
      : null;
    function scrollToLatestMessage() {
      if (lastMessageElement) {
        scrollMessageTopIntoListView(messageList, lastMessageElement);
      } else {
        messageList.scrollTop = 0;
      }
    }

    scrollToLatestMessage();
    const frame = window.requestAnimationFrame(scrollToLatestMessage);
    const settleTimeout = window.setTimeout(scrollToLatestMessage, 120);
    const lateSettleTimeout = window.setTimeout(scrollToLatestMessage, 420);
    scrollToBottomAfterLoadRef.current = false;

    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(settleTimeout);
      window.clearTimeout(lateSettleTimeout);
    };
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

  async function uploadAttachment(file: File) {
    return { ...(await uploadAttachmentToChat(chatId, file)), status: "uploaded" as const };
  }

  async function removeAttachment(attachment: ComposerAttachment) {
    if (
      attachment.isExisting ||
      attachment.status !== "uploaded" ||
      attachment.id.startsWith("pending-")
    ) {
      return;
    }

    await requestJson(`/api/attachments/${attachment.id}`, { method: "DELETE" });
  }

  async function submitPrompt(prompt: string, attachments: ComposerAttachment[] = []) {
    if (isGenerating) {
      setPendingPrompt({ prompt, attachments });
      await stopGeneration();
      return;
    }

    await generate(prompt, attachments);
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

  async function editMessage(
    message: ChatMessage,
    contentText: string,
    attachments: ComposerAttachment[] = []
  ) {
    setBusyMessageId(message.id);
    setGenerationError(null);

    try {
      const response = await requestJson<MessageResponse>(
        `/api/chats/${chatId}/messages/${message.id}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            content_text: contentText,
            attachments: attachmentReferences(attachments)
          })
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

  async function branchMessage(
    message: ChatMessage,
    contentText: string,
    attachments: ComposerAttachment[] = []
  ) {
    if (isGenerating || message.role !== "user") {
      return;
    }

    const selected =
      selectedModelBaseParts([], personas, [], selectedModel) ?? modelParts(selectedModel);
    const personaVersionId = personaVersionIdFromValue(selectedModel);
    await streamAssistantResponse(`/api/chats/${chatId}/messages/${message.id}/branch`, {
      content_text: contentText,
      backend_id: selected?.backendId ?? null,
      model_name: selected?.modelName ?? null,
      persona_version_id: personaVersionId,
      think_mode: null,
      attachments: attachmentReferences(attachments)
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

    const selected =
      selectedModelBaseParts([], personas, [], selectedModel) ?? modelParts(selectedModel);
    const personaVersionId = personaVersionIdFromValue(selectedModel);
    await streamAssistantResponse(`/api/chats/${chatId}/messages/${message.id}/regenerate`, {
      backend_id: selected?.backendId ?? message.backend_id,
      model_name: selected?.modelName ?? message.model_name,
      persona_version_id: personaVersionId,
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

        if (nextMessage.role === "assistant") {
          if (nextMessage.persona_version_id) {
            onModelSelected(personaModelValue(nextMessage.persona_version_id));
          } else if (nextMessage.backend_id && nextMessage.model_name) {
            onModelSelected(modelValue(nextMessage.backend_id, nextMessage.model_name));
          }
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
                  onImageOpen={onImageOpen}
                  onRemoveAttachment={removeAttachment}
                  onUploadAttachment={uploadAttachment}
                  onRegenerate={regenerateMessage}
                  selectedModelInfo={selectedModelInfo}
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
              selectedModelInfo={selectedModelInfo}
              warning={modelImageWarning}
              onStop={stopGeneration}
              onUploadAttachment={uploadAttachment}
              onRemoveAttachment={removeAttachment}
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

function PrivateChatView({
  chatId,
  error,
  queuedPrompt,
  queuedAttachments,
  selectedModel,
  selectedModelInfo,
  privatePersonas,
  onImageOpen,
  onModelSelected,
  onPrivateChatsChanged,
  onQueuedPromptConsumed
}: {
  chatId: string;
  error: string | null;
  queuedPrompt: string | null;
  queuedAttachments: ComposerAttachment[];
  selectedModel: string;
  selectedModelInfo: ModelInfo | null;
  privatePersonas: PrivatePersona[];
  onImageOpen: ImageOpenHandler;
  onModelSelected: (value: string) => void;
  onPrivateChatsChanged: () => Promise<void>;
  onQueuedPromptConsumed: () => void;
}) {
  const [chat, setChat] = useState<PrivateChatDetail | null>(null);
  const [messages, setMessages] = useState<PrivateChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeAssistantId, setActiveAssistantId] = useState<string | null>(null);
  const [pendingPrompt, setPendingPrompt] = useState<ComposerSubmitPayload | null>(null);
  const [busyMessageId, setBusyMessageId] = useState<string | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<PrivateChatMessage | null>(null);
  const [streamTestStatus, setStreamTestStatus] = useState<string | null>(null);
  const [isPrivateBannerExpanded, setIsPrivateBannerExpanded] = useState(true);
  const messageListRef = useRef<HTMLElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const generationRunRef = useRef(0);
  const autoScrollModeRef = useRef<AutoScrollMode>("top");
  const userScrollIntentRef = useRef(false);
  const userScrollIntentTimeoutRef = useRef<number | null>(null);
  const scrollToBottomAfterLoadRef = useRef(false);
  const branchScrollAnchorRef = useRef<BranchScrollAnchor | null>(null);
  const messagesRef = useRef<PrivateChatMessage[]>([]);
  const privateSaveChainRef = useRef<Promise<void>>(Promise.resolve());
  const privateBannerTimerRef = useRef<number | null>(null);
  const thinkingStartedAtRef = useRef(new Map<string, number>());
  const [thinkingDurations, setThinkingDurations] = useState<Record<string, number>>({});
  const visibleMessages = useMemo(
    () => activePathMessages(messages, chat?.active_root_message_id ?? null),
    [chat?.active_root_message_id, messages]
  );
  const siblingGroups = useMemo(() => groupMessagesByParent(messages), [messages]);
  const chatContainsImages = useMemo(
    () => messages.some((message) => activeMessageAttachments(message).some(isImageAttachment)),
    [messages]
  );
  const modelImageWarning =
    selectedModelInfo && !selectedModelInfo.supports_images && chatContainsImages
      ? "This chat includes images. Images may not be supported by this model."
      : null;
  const showStreamTest = privateStreamTestEnabled();
  const hasChat = Boolean(chat);

  const loadPrivateChat = useCallback(async () => {
    scrollToBottomAfterLoadRef.current = true;
    setIsLoading(true);
    setLoadError(null);

    try {
      const [nextChat, nextMessages] = await Promise.all([
        getPrivateChat(chatId),
        listPrivateMessages(chatId)
      ]);
      if (!nextChat) {
        throw new Error("Private chat not found on this device");
      }

      setChat(nextChat);
      thinkingStartedAtRef.current.clear();
      setThinkingDurations({});
      messagesRef.current = nextMessages;
      setMessages(nextMessages);
      const streamingAssistantId = streamingAssistantIdFromMessages(nextMessages);
      setActiveAssistantId(streamingAssistantId);
      setIsGenerating(Boolean(streamingAssistantId));
      const latestModel = latestAssistantModelValue(
        activePathMessages(nextMessages, nextChat.active_root_message_id)
      );
      const defaultModel =
        nextChat.persona_version_id &&
        privatePersonaForVersionId(privatePersonas, nextChat.persona_version_id)
          ? privatePersonaModelValue(nextChat.persona_version_id)
          : modelValue(nextChat.default_backend_id, nextChat.default_model_name);
      onModelSelected(latestModel ?? defaultModel);
    } catch (chatError) {
      setLoadError(
        chatError instanceof Error ? chatError.message : "Failed to load private chat"
      );
    } finally {
      setIsLoading(false);
    }
  }, [chatId, onModelSelected, privatePersonas]);

  useEffect(() => {
    void loadPrivateChat();
  }, [loadPrivateChat]);

  useEffect(() => {
    setIsPrivateBannerExpanded(true);
  }, [chatId]);

  useEffect(() => {
    if (privateBannerTimerRef.current) {
      window.clearTimeout(privateBannerTimerRef.current);
      privateBannerTimerRef.current = null;
    }

    if (!isPrivateBannerExpanded || !hasChat) {
      return;
    }

    privateBannerTimerRef.current = window.setTimeout(() => {
      setIsPrivateBannerExpanded(false);
      privateBannerTimerRef.current = null;
    }, 3000);

    return () => {
      if (privateBannerTimerRef.current) {
        window.clearTimeout(privateBannerTimerRef.current);
        privateBannerTimerRef.current = null;
      }
    };
  }, [chatId, hasChat, isPrivateBannerExpanded]);

  useEffect(() => {
    return () => {
      // Do not abort here. The stream continues writing into IndexedDB during app navigation.
      if (userScrollIntentTimeoutRef.current) {
        window.clearTimeout(userScrollIntentTimeoutRef.current);
      }
      if (privateBannerTimerRef.current) {
        window.clearTimeout(privateBannerTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const hasStreamingMessage = messages.some(
      (message) => message.role === "assistant" && message.status === "streaming"
    );
    if (!hasStreamingMessage || abortRef.current) {
      return;
    }

    const interval = window.setInterval(() => {
      void listPrivateMessages(chatId).then((nextMessages) => {
        messagesRef.current = nextMessages;
        setMessages(nextMessages);
        const streamingAssistantId = streamingAssistantIdFromMessages(nextMessages);
        setActiveAssistantId(streamingAssistantId);
        setIsGenerating(Boolean(streamingAssistantId));
      });
    }, 1000);

    return () => window.clearInterval(interval);
  }, [chatId, messages]);

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

    const messageList = list;
    const lastMessage = visibleMessages[visibleMessages.length - 1];
    const lastMessageElement = lastMessage
      ? messageList.querySelector<HTMLElement>(`[data-message-id="${lastMessage.id}"]`)
      : null;
    function scrollToLatestMessage() {
      if (lastMessageElement) {
        scrollMessageTopIntoListView(messageList, lastMessageElement);
      } else {
        messageList.scrollTop = 0;
      }
    }

    scrollToLatestMessage();
    const frame = window.requestAnimationFrame(scrollToLatestMessage);
    const settleTimeout = window.setTimeout(scrollToLatestMessage, 120);
    const lateSettleTimeout = window.setTimeout(scrollToLatestMessage, 420);
    scrollToBottomAfterLoadRef.current = false;

    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(settleTimeout);
      window.clearTimeout(lateSettleTimeout);
    };
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

  useEffect(() => {
    if (!queuedPrompt || isLoading || !chat || isGenerating) {
      return;
    }

    onQueuedPromptConsumed();
    void submitPrompt(queuedPrompt, queuedAttachments);
  }, [chat, isGenerating, isLoading, onQueuedPromptConsumed, queuedAttachments, queuedPrompt]);

  useEffect(() => {
    if (!pendingPrompt || isGenerating || isLoading || !chat) {
      return;
    }

    const prompt = pendingPrompt;
    setPendingPrompt(null);
    void generate(prompt.prompt, prompt.attachments);
  }, [chat, isGenerating, isLoading, pendingPrompt]);

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

  function updateMessage(messageId: string, updater: (message: PrivateChatMessage) => PrivateChatMessage) {
    let changedMessage: PrivateChatMessage | null = null;
    const next = messagesRef.current.map((message) => {
      if (message.id !== messageId) {
        return message;
      }

      changedMessage = updater(message);
      return changedMessage;
    });

    messagesRef.current = next;
    if (changedMessage) {
      queuePrivateMessageSave(changedMessage);
    }
    setMessages(next);
  }

  function replacePrivateMessages(nextMessages: PrivateChatMessage[]) {
    messagesRef.current = nextMessages;
    setMessages(nextMessages);
  }

  function queuePrivateMessageSave(message: PrivateChatMessage) {
    privateSaveChainRef.current = privateSaveChainRef.current
      .catch(() => undefined)
      .then(() => savePrivateMessage(message))
      .catch((saveError) => {
        setGenerationError(
          saveError instanceof Error ? saveError.message : "Failed to save private message"
        );
      });
  }

  async function updateChat(nextChat: PrivateChatDetail) {
    setChat(nextChat);
    await savePrivateChat(nextChat);
    await onPrivateChatsChanged();
  }

  async function streamPrivateAssistantResponse(
    assistantId: string,
    body: unknown,
    path = "/api/private/generate"
  ) {
    const runId = generationRunRef.current + 1;
    generationRunRef.current = runId;
    const controller = new AbortController();
    abortRef.current = controller;
    autoScrollModeRef.current = "top";
    setIsGenerating(true);
    setActiveAssistantId(assistantId);
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
          if (trimmed) {
            applyPrivateGenerateEvent(JSON.parse(trimmed) as GenerateEvent, runId);
          }
        }
      }

      buffer += decoder.decode();
      const trailing = buffer.trim();
      if (trailing) {
        applyPrivateGenerateEvent(JSON.parse(trailing) as GenerateEvent, runId);
      }

      await privateSaveChainRef.current;
      await onPrivateChatsChanged();
    } catch (generateError) {
      if (generateError instanceof DOMException && generateError.name === "AbortError") {
        return;
      }

      const message = generateError instanceof Error ? generateError.message : "Generation failed";
      setGenerationError(message);
      updateMessage(assistantId, (current) => ({
        ...current,
        status: "error",
        error_text: message,
        completed_at: unixTimestamp(),
        updated_at: unixTimestamp()
      }));
    } finally {
      if (generationRunRef.current === runId) {
        setIsGenerating(false);
        setActiveAssistantId(null);
        abortRef.current = null;
      }
    }
  }

  async function generate(prompt: string, attachments: ComposerAttachment[] = []) {
    if (!chat || isGenerating) {
      return;
    }

    if (personaVersionIdFromValue(selectedModel)) {
      setGenerationError("Copy this hosted persona to your device before using it in a private chat");
      return;
    }

    const selectedPrivatePersona =
      privatePersonaForValue(privatePersonas, selectedModel) ??
      (messagesRef.current.length === 0 && chat.persona_version_id
        ? privatePersonaForVersionId(privatePersonas, chat.persona_version_id)
        : null);
    const selected = selectedPrivatePersona
      ? {
          backendId: selectedPrivatePersona.current_version.base_backend_id,
          modelName: selectedPrivatePersona.current_version.base_model_name
        }
      : modelParts(selectedModel);
    if (!selected) {
      setGenerationError("Select a model before sending a private message");
      return;
    }

    const now = unixTimestamp();
    const pathMessages = activePathMessages(messages, chat.active_root_message_id);
    const parent = pathMessages[pathMessages.length - 1] as PrivateChatMessage | undefined;
    const userMessage = createPrivateMessage({
      chatId: chat.id,
      parentMessageId: parent?.id ?? null,
      role: "user",
      contentText: prompt,
      createdAt: now
    });
    userMessage.attachments = privateAttachmentsForMessage(userMessage, attachments);
    const assistantMessage = createPrivateMessage({
      chatId: chat.id,
      parentMessageId: userMessage.id,
      role: "assistant",
      contentText: "",
      status: "streaming",
      backendId: selected.backendId,
      modelName: selected.modelName,
      personaId: selectedPrivatePersona?.id ?? null,
      personaVersionId: selectedPrivatePersona?.current_version.id ?? null,
      personaNameSnapshot: selectedPrivatePersona?.current_version.display_name ?? null,
      createdAt: now
    });
    userMessage.active_child_message_id = assistantMessage.id;

    const nextMessages = [
      ...messages.map((message) =>
        parent && message.id === parent.id
          ? { ...message, active_child_message_id: userMessage.id, updated_at: now }
          : message
      ),
      userMessage,
      assistantMessage
    ];
    const title =
      chat.title === "Private Chat" && messages.length === 0
        ? fallbackTitleFromPrompt(prompt, "Private Chat")
        : chat.title;
    const nextChat = {
      ...chat,
      title,
      default_backend_id: selected.backendId,
      default_model_name: selected.modelName,
      persona_id: selectedPrivatePersona?.id ?? null,
      persona_version_id: selectedPrivatePersona?.current_version.id ?? null,
      persona_name: selectedPrivatePersona?.current_version.display_name ?? null,
      active_root_message_id: chat.active_root_message_id ?? userMessage.id,
      updated_at: now,
      last_message_at: now
    };

    setChat(nextChat);
    messagesRef.current = nextMessages;
    setMessages(nextMessages);
    await Promise.all([savePrivateChat(nextChat), savePrivateMessages(nextMessages)]);
    await onPrivateChatsChanged();

    await streamPrivateAssistantResponse(assistantMessage.id, {
      assistant_message_id: assistantMessage.id,
      backend_id: selected.backendId,
      model_name: selected.modelName,
      think_mode: null,
      messages: privatePromptMessagesWithPersona(
        nextMessages,
        nextChat.active_root_message_id,
        assistantMessage.id,
        selectedPrivatePersona
      ),
      attachments: []
    });
  }

  async function runPrivateStreamTest() {
    if (!chat || isGenerating) {
      return;
    }

    const contentTokens = 1600;
    const thinkingTokens = 220;
    const now = unixTimestamp();
    const pathMessages = activePathMessages(messagesRef.current, chat.active_root_message_id);
    const parent = pathMessages[pathMessages.length - 1] as PrivateChatMessage | undefined;
    const userMessage = createPrivateMessage({
      chatId: chat.id,
      parentMessageId: parent?.id ?? null,
      role: "user",
      contentText: `Synthetic private stream test: ${thinkingTokens} thinking chunks, ${contentTokens} content chunks.`,
      createdAt: now
    });
    const assistantMessage = createPrivateMessage({
      chatId: chat.id,
      parentMessageId: userMessage.id,
      role: "assistant",
      contentText: "",
      status: "streaming",
      modelName: "synthetic-stream-test",
      createdAt: now
    });
    userMessage.active_child_message_id = assistantMessage.id;

    const nextMessages = [
      ...messagesRef.current.map((message) =>
        parent && message.id === parent.id
          ? { ...message, active_child_message_id: userMessage.id, updated_at: now }
          : message
      ),
      userMessage,
      assistantMessage
    ];
    const nextChat = {
      ...chat,
      title: chat.title === "Private Chat" ? "Private Stream Test" : chat.title,
      active_root_message_id: chat.active_root_message_id ?? userMessage.id,
      updated_at: now,
      last_message_at: now
    };

    setStreamTestStatus("Running synthetic stream test...");
    setChat(nextChat);
    messagesRef.current = nextMessages;
    setMessages(nextMessages);
    await Promise.all([savePrivateChat(nextChat), savePrivateMessages(nextMessages)]);
    await onPrivateChatsChanged();

    await streamPrivateAssistantResponse(
      assistantMessage.id,
      {
        assistant_message_id: assistantMessage.id,
        content_tokens: contentTokens,
        thinking_tokens: thinkingTokens,
        delay_ms: 0
      },
      "/api/dev/private-stream-test"
    );
    await privateSaveChainRef.current;

    const saved = messagesRef.current.find((message) => message.id === assistantMessage.id);
    const content = saved?.active_revision?.content_text ?? "";
    const thinking = saved?.active_revision?.thinking_text ?? "";
    const expectedContent = syntheticStreamExpectedContent(contentTokens);
    const expectedThinking = syntheticStreamExpectedThinking(thinkingTokens);
    if (content === expectedContent && thinking === expectedThinking) {
      setStreamTestStatus(
        `Passed synthetic stream test: ${thinkingTokens + contentTokens} chunks preserved.`
      );
      return;
    }

    setStreamTestStatus(
      `Failed synthetic stream test: content ${content.length}/${expectedContent.length}, thinking ${thinking.length}/${expectedThinking.length}.`
    );
  }

  async function submitPrompt(prompt: string, attachments: ComposerAttachment[] = []) {
    if (isGenerating) {
      setPendingPrompt({ prompt, attachments });
      await stopGeneration();
      return;
    }

    await generate(prompt, attachments);
  }

  async function stopGeneration() {
    const assistantId = activeAssistantId;
    generationRunRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    setIsGenerating(false);
    setActiveAssistantId(null);

    if (assistantId) {
      updateMessage(assistantId, (current) => ({
        ...current,
        status: "stopped",
        done_reason: "stopped",
        completed_at: unixTimestamp(),
        updated_at: unixTimestamp()
      }));
    }
  }

  function applyPrivateGenerateEvent(event: GenerateEvent, runId: number) {
    if (generationRunRef.current !== runId) {
      return;
    }

    switch (event.type) {
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
        updateMessage(event.assistant_message_id, (current) => ({
          ...current,
          status: "complete",
          done_reason: event.done_reason,
          completed_at: unixTimestamp(),
          updated_at: unixTimestamp()
        }));
        setIsGenerating(false);
        setActiveAssistantId(null);
        abortRef.current = null;
        break;
      case "message_stopped":
        updateMessage(event.assistant_message_id, (current) => ({
          ...current,
          status: "stopped",
          done_reason: "stopped",
          completed_at: unixTimestamp(),
          updated_at: unixTimestamp()
        }));
        break;
      case "error":
        setGenerationError(event.message);
        if (event.assistant_message_id) {
          updateMessage(event.assistant_message_id, (current) => ({
            ...current,
            status: "error",
            error_text: event.message,
            completed_at: unixTimestamp(),
            updated_at: unixTimestamp()
          }));
        }
        break;
      case "message_start":
      case "chat_title":
        break;
    }
  }

  function appendMessageText(
    messageId: string,
    field: "content_text" | "thinking_text",
    delta: string
  ) {
    updateMessage(messageId, (message) => {
      const activeRevision = message.active_revision ?? {
        id: message.active_revision_id ?? privateId("private-revision"),
        content_text: "",
        thinking_text: "",
        source: "original",
        created_at: message.created_at
      };
      const nextRevision = {
        ...activeRevision,
        [field]: activeRevision[field] + delta
      };
      const revisions = updateRevisionList(message.revisions, nextRevision);

      return {
        ...message,
        status: "streaming",
        active_revision_id: nextRevision.id,
        active_revision: nextRevision,
        revisions,
        revision_count: revisions.length,
        updated_at: unixTimestamp()
      };
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

  async function editMessage(
    message: ChatMessage,
    contentText: string,
    attachments: ComposerAttachment[] = []
  ) {
    setBusyMessageId(message.id);
    try {
      const now = unixTimestamp();
      const revision: PrivateChatMessageRevision = {
        id: privateId("private-revision"),
        content_text: contentText,
        thinking_text: message.active_revision?.thinking_text ?? "",
        source: "edit",
        created_at: now
      };
      updateMessage(message.id, (current) => {
        const revisions = updateRevisionList(current.revisions, revision);
        const nextAttachments = privateAttachmentsForMessage(current, attachments, revision.id);
        return {
          ...current,
          active_revision_id: revision.id,
          active_revision: revision,
          revisions,
          revision_count: revisions.length,
          attachments: [...(current.attachments ?? []), ...nextAttachments],
          updated_at: now
        };
      });
    } finally {
      setBusyMessageId(null);
    }
  }

  async function branchMessage(
    message: ChatMessage,
    contentText: string,
    attachments: ComposerAttachment[] = []
  ) {
    if (!chat || isGenerating || message.role !== "user") {
      return;
    }

    if (personaVersionIdFromValue(selectedModel)) {
      setGenerationError("Copy this hosted persona to your device before using it in a private chat");
      return;
    }

    const selectedPrivatePersona = privatePersonaForValue(privatePersonas, selectedModel);
    const selected = selectedPrivatePersona
      ? {
          backendId: selectedPrivatePersona.current_version.base_backend_id,
          modelName: selectedPrivatePersona.current_version.base_model_name
        }
      : modelParts(selectedModel);
    if (!selected) {
      setGenerationError("Select a model before sending a private branch");
      return;
    }

    setBusyMessageId(message.id);
    setGenerationError(null);

    try {
      const now = unixTimestamp();
      const userMessage = createPrivateMessage({
        chatId: chat.id,
        parentMessageId: message.parent_message_id,
        role: "user",
        contentText,
        createdAt: now
      });
      userMessage.attachments = privateAttachmentsForMessage(userMessage, attachments);
      const assistantMessage = createPrivateMessage({
        chatId: chat.id,
        parentMessageId: userMessage.id,
        role: "assistant",
        contentText: "",
        status: "streaming",
        backendId: selected.backendId,
        modelName: selected.modelName,
        personaId: selectedPrivatePersona?.id ?? null,
        personaVersionId: selectedPrivatePersona?.current_version.id ?? null,
        personaNameSnapshot: selectedPrivatePersona?.current_version.display_name ?? null,
        createdAt: now
      });
      userMessage.active_child_message_id = assistantMessage.id;

      const nextMessages = [
        ...messagesRef.current.map((current) =>
          message.parent_message_id && current.id === message.parent_message_id
            ? { ...current, active_child_message_id: userMessage.id, updated_at: now }
            : current
        ),
        userMessage,
        assistantMessage
      ];
      const nextChat = {
        ...chat,
        default_backend_id: selected.backendId,
        default_model_name: selected.modelName,
        persona_id: selectedPrivatePersona?.id ?? null,
        persona_version_id: selectedPrivatePersona?.current_version.id ?? null,
        persona_name: selectedPrivatePersona?.current_version.display_name ?? null,
        active_root_message_id: message.parent_message_id
          ? chat.active_root_message_id
          : userMessage.id,
        updated_at: now,
        last_message_at: now
      };

      setChat(nextChat);
      replacePrivateMessages(nextMessages);
      await privateSaveChainRef.current;
      await Promise.all([savePrivateChat(nextChat), savePrivateMessages(nextMessages)]);
      await onPrivateChatsChanged();

      await streamPrivateAssistantResponse(assistantMessage.id, {
        assistant_message_id: assistantMessage.id,
        backend_id: selected.backendId,
        model_name: selected.modelName,
        think_mode: null,
        messages: privatePromptMessagesWithPersona(
          nextMessages,
          nextChat.active_root_message_id,
          assistantMessage.id,
          selectedPrivatePersona
        ),
        attachments: []
      });
    } finally {
      setBusyMessageId(null);
    }
  }

  async function deleteMessage(message: ChatMessage) {
    setBusyMessageId(message.id);
    try {
      const now = unixTimestamp();
      updateMessage(message.id, (current) => ({
        ...current,
        is_deleted: true,
        status: "complete",
        active_revision: current.active_revision
          ? { ...current.active_revision, content_text: "", thinking_text: "" }
          : null,
        revisions: current.revisions.map((revision) => ({
          ...revision,
          content_text: "",
          thinking_text: ""
        })),
        updated_at: now
      }));
      setDeleteTarget(null);
    } finally {
      setBusyMessageId(null);
    }
  }

  async function regenerateMessage(message: ChatMessage) {
    if (!chat || isGenerating || message.role !== "assistant") {
      return;
    }

    if (personaVersionIdFromValue(selectedModel)) {
      setGenerationError("Copy this hosted persona to your device before using it in a private chat");
      return;
    }

    const selectedPrivatePersona =
      privatePersonaForValue(privatePersonas, selectedModel) ??
      privatePersonaForVersionId(privatePersonas, message.persona_version_id);
    const selected = selectedPrivatePersona
      ? {
          backendId: selectedPrivatePersona.current_version.base_backend_id,
          modelName: selectedPrivatePersona.current_version.base_model_name
        }
      : modelParts(selectedModel);
    const backendId = selected?.backendId ?? message.backend_id;
    const modelName = selected?.modelName ?? message.model_name;
    if (!backendId || !modelName) {
      setGenerationError("Select a model before regenerating this private response");
      return;
    }

    setBusyMessageId(message.id);
    setGenerationError(null);

    try {
      const now = unixTimestamp();
      const assistantMessage = createPrivateMessage({
        chatId: chat.id,
        parentMessageId: message.parent_message_id,
        role: "assistant",
        contentText: "",
        status: "streaming",
        backendId,
        modelName,
        personaId: selectedPrivatePersona?.id ?? null,
        personaVersionId: selectedPrivatePersona?.current_version.id ?? null,
        personaNameSnapshot: selectedPrivatePersona?.current_version.display_name ?? null,
        createdAt: now
      });

      const nextMessages = [
        ...messagesRef.current.map((current) =>
          message.parent_message_id && current.id === message.parent_message_id
            ? { ...current, active_child_message_id: assistantMessage.id, updated_at: now }
            : current
        ),
        assistantMessage
      ];
      const nextChat = {
        ...chat,
        default_backend_id: backendId,
        default_model_name: modelName,
        persona_id: selectedPrivatePersona?.id ?? null,
        persona_version_id: selectedPrivatePersona?.current_version.id ?? null,
        persona_name: selectedPrivatePersona?.current_version.display_name ?? null,
        active_root_message_id: message.parent_message_id
          ? chat.active_root_message_id
          : assistantMessage.id,
        updated_at: now,
        last_message_at: now
      };

      setChat(nextChat);
      replacePrivateMessages(nextMessages);
      await privateSaveChainRef.current;
      await Promise.all([savePrivateChat(nextChat), savePrivateMessages(nextMessages)]);
      await onPrivateChatsChanged();

      await streamPrivateAssistantResponse(assistantMessage.id, {
        assistant_message_id: assistantMessage.id,
        backend_id: backendId,
        model_name: modelName,
        think_mode: null,
        messages: privatePromptMessagesWithPersona(
          nextMessages,
          nextChat.active_root_message_id,
          assistantMessage.id,
          selectedPrivatePersona
        ),
        attachments: []
      });
    } finally {
      setBusyMessageId(null);
    }
  }

  async function selectVersion(currentMessage: ChatMessage, version: MessageVersion) {
    const nextMessage = version.message as PrivateChatMessage;
    const nextRevision = version.revision as PrivateChatMessageRevision;
    const isSameMessage = currentMessage.id === nextMessage.id;
    const isSameRevision = nextMessage.active_revision_id === nextRevision.id;
    if ((isSameMessage && isSameRevision) || isGenerating || !chat) {
      return;
    }

    setBusyMessageId(currentMessage.id);
    setGenerationError(null);

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

    try {
      const now = unixTimestamp();
      const nextMessages = messagesRef.current.map((message) => {
        if (!isSameMessage) {
          if (currentMessage.parent_message_id && message.id === currentMessage.parent_message_id) {
            return { ...message, active_child_message_id: nextMessage.id, updated_at: now };
          }
        }

        if (message.id === nextMessage.id && !isSameRevision) {
          return {
            ...message,
            active_revision_id: nextRevision.id,
            active_revision: nextRevision,
            updated_at: now
          };
        }

        return message;
      });
      const nextChat =
        !isSameMessage && !currentMessage.parent_message_id
          ? { ...chat, active_root_message_id: nextMessage.id, updated_at: now }
          : chat;

      setChat(nextChat);
      replacePrivateMessages(nextMessages);
      if (nextMessage.role === "assistant" && nextMessage.backend_id && nextMessage.model_name) {
        onModelSelected(modelValue(nextMessage.backend_id, nextMessage.model_name));
      }

      await privateSaveChainRef.current;
      await Promise.all([
        nextChat === chat ? Promise.resolve() : savePrivateChat(nextChat),
        savePrivateMessages(nextMessages)
      ]);
      await onPrivateChatsChanged();
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
    <div className="chat-view private-chat-view">
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
          <div
            className={
              isPrivateBannerExpanded
                ? "private-local-banner"
                : "private-local-banner private-local-banner-collapsed"
            }
          >
            <button
              type="button"
              className="private-local-toggle"
              aria-label={
                isPrivateBannerExpanded
                  ? "Collapse private chat notice"
                  : "Show private chat notice"
              }
              aria-expanded={isPrivateBannerExpanded}
              onClick={() => setIsPrivateBannerExpanded((expanded) => !expanded)}
            >
              <Lock />
            </button>
            {isPrivateBannerExpanded && (
              <>
                <span className="private-local-text">
                  Stored only in this browser. Clearing browser data can delete this chat.
                </span>
                {showStreamTest && (
                  <button
                    type="button"
                    className="stream-test-button"
                    disabled={isGenerating}
                    onClick={() => void runPrivateStreamTest()}
                  >
                    Run Stream Test
                  </button>
                )}
                {streamTestStatus && <small>{streamTestStatus}</small>}
              </>
            )}
          </div>
          <section
            className={
              visibleMessages.length > 0 ? "message-list message-list-buffered" : "message-list"
            }
            aria-label="Private chat messages"
            ref={messageListRef}
            onScroll={handleMessageListScroll}
            onTouchMove={noteUserScrollIntent}
            onWheel={noteUserScrollIntent}
          >
            {visibleMessages.length === 0 ? (
              <div className="message-list-empty">
                <p>Private chat ready.</p>
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
                  onDelete={(message) => setDeleteTarget(message as PrivateChatMessage)}
                  onBranch={branchMessage}
                  onEdit={editMessage}
                  onImageOpen={onImageOpen}
                  onUploadAttachment={preparePrivateAttachment}
                  onRegenerate={regenerateMessage}
                  selectedModelInfo={selectedModelInfo}
                />
              ))
            )}
          </section>
          <div className="chat-composer-wrap">
            <StartChatComposer
              isBusy={isGenerating}
              isDisabled={!selectedModel}
              isGenerating={isGenerating}
              placeholder={selectedModel ? "Message private chat" : "Select a model to continue"}
              selectedModelInfo={selectedModelInfo}
              warning={modelImageWarning}
              onStop={stopGeneration}
              onUploadAttachment={preparePrivateAttachment}
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
          title="Delete Private Message"
          message="Delete this local message? Its text and thinking content will be scrubbed from this browser."
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
  canBranch = true,
  canRegenerate = true,
  canEdit = true,
  thinkingDurationSeconds,
  onCopy,
  onDelete,
  onBranch,
  onEdit,
  onImageOpen,
  onRemoveAttachment,
  onUploadAttachment,
  onRegenerate,
  selectedModelInfo
}: {
  message: ChatMessage;
  versionInfo: VersionInfo | null;
  copied: boolean;
  isBusy: boolean;
  isGenerating: boolean;
  canBranch?: boolean;
  canRegenerate?: boolean;
  canEdit?: boolean;
  thinkingDurationSeconds: number | null;
  onCopy: (message: ChatMessage) => Promise<void>;
  onDelete: (message: ChatMessage) => void;
  onBranch: (
    message: ChatMessage,
    contentText: string,
    attachments?: ComposerAttachment[]
  ) => Promise<void>;
  onEdit: (
    message: ChatMessage,
    contentText: string,
    attachments?: ComposerAttachment[]
  ) => Promise<void>;
  onImageOpen?: ImageOpenHandler;
  onRemoveAttachment?: (attachment: ComposerAttachment) => Promise<void>;
  onUploadAttachment?: (file: File) => Promise<ComposerAttachment> | ComposerAttachment;
  onRegenerate: (message: ChatMessage) => Promise<void>;
  selectedModelInfo?: ModelInfo | null;
}) {
  const content = message.is_deleted
    ? "Message deleted"
    : message.active_revision?.content_text.trim() || "";
  const thinking = message.active_revision?.thinking_text.trim();
  const attachments = activeMessageAttachments(message);
  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState(content);
  const [draftAttachments, setDraftAttachments] = useState<ComposerAttachment[]>([]);
  const editFileInputRef = useRef<HTMLInputElement | null>(null);
  const hasUnsupportedImageWarning =
    selectedModelInfo !== undefined &&
    selectedModelInfo !== null &&
    !selectedModelInfo.supports_images &&
    draftAttachments.some(isImageAttachment);
  const hasUploadingAttachment = draftAttachments.some(
    (attachment) => attachment.status === "uploading"
  );

  useEffect(() => {
    if (!isEditing) {
      setDraft(content);
    }
  }, [content, isEditing]);

  async function saveEdit() {
    dismissMobileKeyboard();
    await onEdit(message, draft, submittableAttachments(draftAttachments));
    setIsEditing(false);
  }

  async function sendEdit() {
    dismissMobileKeyboard();
    setIsEditing(false);
    await onBranch(message, draft, submittableAttachments(draftAttachments));
  }

  function startEditing() {
    setDraft(content);
    setDraftAttachments(attachments.map(composerAttachmentFromExisting));
    setIsEditing(true);
  }

  async function cancelEdit() {
    dismissMobileKeyboard();
    await cleanupDraftUploads(draftAttachments, onRemoveAttachment);
    setIsEditing(false);
  }

  async function addEditFiles(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (!onUploadAttachment || files.length === 0) {
      return;
    }

    for (const file of files) {
      const pendingId = newPendingAttachmentId();
      const pendingAttachment = pendingComposerAttachment(pendingId, file);
      setDraftAttachments((current) => [...current, pendingAttachment]);

      try {
        const attachment = await onUploadAttachment(file);
        setDraftAttachments((current) =>
          current.map((currentAttachment) =>
            currentAttachment.id === pendingId ? attachment : currentAttachment
          )
        );
      } catch (uploadError) {
        setDraftAttachments((current) =>
          current.map((currentAttachment) =>
            currentAttachment.id === pendingId
              ? {
                  ...currentAttachment,
                  status: "error",
                  error: uploadError instanceof Error ? uploadError.message : "Upload failed"
                }
              : currentAttachment
          )
        );
      }
    }
  }

  async function removeDraftAttachment(attachment: ComposerAttachment) {
    if (attachment.status === "uploaded" && !attachment.isExisting && onRemoveAttachment) {
      try {
        await onRemoveAttachment(attachment);
      } catch (removeError) {
        setDraftAttachments((current) =>
          current.map((currentAttachment) =>
            currentAttachment.id === attachment.id
              ? {
                  ...currentAttachment,
                  error: removeError instanceof Error ? removeError.message : "Delete failed"
                }
              : currentAttachment
          )
        );
        return;
      }
    }

    setDraftAttachments((current) =>
      current.filter((currentAttachment) => currentAttachment.id !== attachment.id)
    );
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
      {!isEditing && !message.is_deleted && attachments.length > 0 && (
        <MessageAttachments attachments={attachments} onImageOpen={onImageOpen} />
      )}
      {isEditing ? (
        <div className="message-edit">
          {hasUnsupportedImageWarning && (
            <p className="composer-warning">Images may not be supported by this model.</p>
          )}
          {draftAttachments.length > 0 && (
            <AttachmentChipList
              attachments={draftAttachments}
              onRemove={(attachment) => void removeDraftAttachment(attachment)}
            />
          )}
          {onUploadAttachment && (
            <div className="message-edit-attach">
              <input
                ref={editFileInputRef}
                className="visually-hidden"
                type="file"
                multiple
                accept={attachmentAcceptTypes}
                onChange={addEditFiles}
              />
              <button
                type="button"
                className="secondary-button"
                disabled={isBusy}
                onClick={() => editFileInputRef.current?.click()}
              >
                <Paperclip />
                <span>Attach</span>
              </button>
            </div>
          )}
          <textarea value={draft} onChange={(event) => setDraft(event.target.value)} rows={5} />
          <div className="message-actions">
            <button type="button" className="secondary-button" disabled={isBusy} onClick={() => void cancelEdit()}>
              <X />
              <span>Cancel</span>
            </button>
            <button
              type="button"
              disabled={isBusy || hasUploadingAttachment || draft.trim() === ""}
              onClick={() => void saveEdit()}
            >
              <Save />
              <span>{isBusy ? "Saving..." : "Save"}</span>
            </button>
            {message.role === "user" && canBranch && (
              <button
                type="button"
                disabled={isBusy || isGenerating || hasUploadingAttachment || draft.trim() === ""}
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
          {message.role === "assistant" && canRegenerate && (
            <button
              type="button"
              className="message-icon-button"
              title="Regenerate"
              aria-label="Regenerate"
              disabled={isBusy || isGenerating || message.status === "streaming"}
              onClick={() => {
                dismissMobileKeyboard();
                void onRegenerate(message);
              }}
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
            disabled={isBusy || !canEdit || message.is_deleted || message.status === "streaming"}
            onClick={startEditing}
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
    return message.persona_name_snapshot ?? message.model_name ?? "assistant";
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
    if (message.role === "assistant" && message.persona_version_id) {
      return isPrivatePersonaVersionId(message.persona_version_id)
        ? privatePersonaModelValue(message.persona_version_id)
        : personaModelValue(message.persona_version_id);
    }
    if (message.role === "assistant" && message.backend_id && message.model_name) {
      return modelValue(message.backend_id, message.model_name);
    }
  }

  return null;
}

function personaForValue(personas: Persona[], value: string) {
  const personaVersionId = personaVersionIdFromValue(value);
  if (!personaVersionId) {
    return null;
  }

  return personas.find((persona) => persona.current_version.id === personaVersionId) ?? null;
}

function privatePersonaForValue(personas: PrivatePersona[], value: string) {
  const personaVersionId = privatePersonaVersionIdFromValue(value);
  if (!personaVersionId) {
    return null;
  }

  return privatePersonaForVersionId(personas, personaVersionId);
}

function privatePersonaForVersionId(
  personas: PrivatePersona[],
  personaVersionId: string | null | undefined
) {
  if (!personaVersionId) {
    return null;
  }

  return personas.find((persona) => persona.current_version.id === personaVersionId) ?? null;
}

function selectedModelBaseParts(
  groups: BackendModelGroup[],
  personas: Persona[],
  privatePersonas: PrivatePersona[],
  value: string
) {
  const selectedPrivatePersona = privatePersonaForValue(privatePersonas, value);
  if (selectedPrivatePersona) {
    return {
      backendId: selectedPrivatePersona.current_version.base_backend_id,
      modelName: selectedPrivatePersona.current_version.base_model_name
    };
  }

  const selectedPersona = personaForValue(personas, value);
  if (selectedPersona) {
    return {
      backendId: selectedPersona.current_version.base_backend_id,
      modelName: selectedPersona.current_version.base_model_name
    };
  }

  const selected = modelParts(value);
  if (!selected) {
    return null;
  }

  const modelInfo = modelInfoForBase(groups, selected.backendId, selected.modelName);
  return modelInfo ? selected : null;
}

function modelInfoForValue(
  groups: BackendModelGroup[],
  personas: Persona[],
  privatePersonas: PrivatePersona[],
  value: string
) {
  const selectedPrivatePersona = privatePersonaForValue(privatePersonas, value);
  if (selectedPrivatePersona) {
    return modelInfoForBase(
      groups,
      selectedPrivatePersona.current_version.base_backend_id,
      selectedPrivatePersona.current_version.base_model_name
    );
  }

  const selectedPersona = personaForValue(personas, value);
  if (selectedPersona) {
    return modelInfoForBase(
      groups,
      selectedPersona.current_version.base_backend_id,
      selectedPersona.current_version.base_model_name
    );
  }

  const selected = modelParts(value);
  if (!selected) {
    return null;
  }

  return modelInfoForBase(groups, selected.backendId, selected.modelName);
}

function modelInfoForBase(groups: BackendModelGroup[], backendId: string, modelName: string) {
  const group = groups.find((modelGroup) => modelGroup.backend.id === backendId);
  return group?.models.find((model) => model.name === modelName) ?? null;
}

function streamingAssistantIdFromMessages(messages: ChatMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && message.status === "streaming") {
      return message.id;
    }
  }

  return null;
}

function privatePromptMessages(
  messages: ChatMessage[],
  activeRootMessageId: string | null,
  stopBeforeMessageId: string
) {
  return activePathMessages(messages, activeRootMessageId)
    .filter((message) => message.id !== stopBeforeMessageId)
    .filter((message) => !message.is_deleted)
    .map((message) => {
      const attachmentPayload = privateAttachmentPromptPayload(activeMessageAttachments(message));
      return {
        role: message.role,
        content_text: withPrivateAttachmentText(
          message.active_revision?.content_text ?? "",
          attachmentPayload.text
        ),
        thinking_text: message.active_revision?.thinking_text || null,
        images: attachmentPayload.images
      };
    })
    .filter((message) => message.content_text.trim() !== "" || message.images.length > 0);
}

function privatePromptMessagesWithPersona(
  messages: ChatMessage[],
  activeRootMessageId: string | null,
  stopBeforeMessageId: string,
  persona: PrivatePersona | null
) {
  const promptMessages = privatePromptMessages(messages, activeRootMessageId, stopBeforeMessageId);
  const systemPrompt = persona?.current_version.system_prompt.trim();
  if (!systemPrompt) {
    return promptMessages;
  }

  return [
    {
      role: "system",
      content_text: systemPrompt,
      thinking_text: null,
      images: []
    },
    ...promptMessages
  ];
}

function privateAttachmentPromptPayload(attachments: AttachmentInfo[]) {
  const textParts: string[] = [];
  const images: string[] = [];

  for (const attachment of attachments) {
    if (isImageAttachment(attachment)) {
      const imageBase64 = imageBase64FromDataUrl(attachment.data_url);
      if (imageBase64) {
        images.push(imageBase64);
      }
      continue;
    }

    if (attachment.text_content) {
      textParts.push(
        `Attachment: ${attachment.original_filename}\n\n${attachment.text_content}`
      );
    }
  }

  return {
    text: textParts.join("\n\n---\n\n"),
    images
  };
}

function withPrivateAttachmentText(content: string, attachmentText: string) {
  if (!attachmentText) {
    return content;
  }

  return `${content.trim()}\n\n${attachmentText}`.trim();
}

function imageBase64FromDataUrl(dataUrl: string | undefined) {
  const commaIndex = dataUrl?.indexOf(",") ?? -1;
  return commaIndex >= 0 ? dataUrl?.slice(commaIndex + 1) ?? "" : "";
}

function activeMessageAttachments(message: ChatMessage) {
  const attachments = message.attachments ?? [];
  if (!message.active_revision_id) {
    return attachments;
  }

  return attachments.filter(
    (attachment) =>
      !attachment.revision_id || attachment.revision_id === message.active_revision_id
  );
}

function composerAttachmentFromExisting(attachment: AttachmentInfo): ComposerAttachment {
  return {
    ...attachment,
    status: "uploaded",
    isExisting: true
  };
}

function privateAttachmentsForMessage(
  message: PrivateChatMessage,
  attachments: ComposerAttachment[],
  revisionId = message.active_revision_id
) {
  return attachments
    .filter((attachment) => attachment.status === "ready" || attachment.status === "uploaded")
    .map((attachment) =>
      privateAttachmentForMessage(
        message,
        attachment,
        attachment.isExisting ? privateId("private-attachment") : attachment.id,
        revisionId
      )
    );
}

function privateAttachmentForMessage(
  message: PrivateChatMessage,
  attachment: AttachmentInfo,
  id: string,
  revisionId: string | null
) {
  return {
    id,
    chat_id: message.chat_id,
    message_id: message.id,
    revision_id: revisionId,
    original_filename: attachment.original_filename,
    mime_type: attachment.mime_type,
    size_bytes: attachment.size_bytes,
    attachment_kind: attachment.attachment_kind,
    created_at: attachment.created_at ?? unixTimestamp(),
    data_url: attachment.data_url,
    text_content: attachment.text_content
  };
}

function fallbackTitleFromPrompt(prompt: string, fallback: string) {
  const title = prompt
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 5)
    .join(" ")
    .trim()
    .replace(/^["'`*#.,:;\s]+|["'`*#.,:;\s]+$/g, "");

  return title || fallback;
}

function syntheticStreamExpectedThinking(count: number) {
  return Array.from({ length: count }, (_, index) => `think-${String(index + 1).padStart(5, "0")} `).join("");
}

function syntheticStreamExpectedContent(count: number) {
  return Array.from({ length: count }, (_, index) => {
    const token = index + 1;
    return token % 17 === 0
      ? `\nchunk-${String(token).padStart(5, "0")};`
      : `tok-${String(token).padStart(5, "0")} `;
  }).join("");
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

function scrollMessageTopIntoListView(list: HTMLElement, messageElement: HTMLElement) {
  const styles = window.getComputedStyle(list);
  const topPadding = Number.parseFloat(styles.paddingTop) || 0;
  const topOffset =
    messageElement.getBoundingClientRect().top - list.getBoundingClientRect().top - topPadding;
  list.scrollTop += topOffset;
}

function usesTouchViewport() {
  return (
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse), (max-width: 720px)").matches
  );
}

function dismissMobileKeyboard() {
  if (!usesTouchViewport()) {
    return;
  }

  const activeElement = document.activeElement;
  if (activeElement instanceof HTMLElement) {
    activeElement.blur();
  }
}

function StartChatComposer({
  isBusy,
  isDisabled,
  isGenerating = false,
  placeholder,
  selectedModelInfo,
  warning,
  onStop,
  onUploadAttachment,
  onRemoveAttachment,
  onSubmit,
  autoFocusOnReady = true
}: {
  isBusy: boolean;
  isDisabled: boolean;
  isGenerating?: boolean;
  placeholder: string;
  selectedModelInfo?: ModelInfo | null;
  warning?: string | null;
  onStop?: () => void;
  onUploadAttachment?: (file: File) => Promise<ComposerAttachment> | ComposerAttachment;
  onRemoveAttachment?: (attachment: ComposerAttachment) => Promise<void>;
  onSubmit: (prompt: string, attachments?: ComposerAttachment[]) => Promise<void>;
  autoFocusOnReady?: boolean;
}) {
  const [prompt, setPrompt] = useState("");
  const [attachments, setAttachments] = useState<ComposerAttachment[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const canAttach = Boolean(onUploadAttachment);
  const hasUploadingAttachment = attachments.some((attachment) => attachment.status === "uploading");
  const hasUnsupportedImageWarning =
    selectedModelInfo !== undefined &&
    selectedModelInfo !== null &&
    !selectedModelInfo.supports_images &&
    attachments.some(isImageAttachment);
  const visibleWarning =
    warning ?? (hasUnsupportedImageWarning ? "Images may not be supported by this model." : null);
  const canSubmit =
    prompt.trim().length > 0 && !isDisabled && !hasUploadingAttachment && (!isBusy || isGenerating);

  useEffect(() => {
    if (autoFocusOnReady && !isGenerating && !usesTouchViewport()) {
      textareaRef.current?.focus();
    }
  }, [autoFocusOnReady, isGenerating]);

  useEffect(() => {
    if (!canAttach) {
      setAttachments([]);
    }
  }, [canAttach]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }

    const submittedPrompt = prompt;
    const submittedAttachments = attachments.filter(
      (attachment) => attachment.status === "ready" || attachment.status === "uploaded"
    );
    setPrompt("");
    setAttachments([]);
    const shouldRestoreFocus = !usesTouchViewport();
    if (!shouldRestoreFocus) {
      textareaRef.current?.blur();
      dismissMobileKeyboard();
    }
    await onSubmit(submittedPrompt, submittedAttachments);
    if (shouldRestoreFocus) {
      window.requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }

  async function uploadFiles(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (!onUploadAttachment || files.length === 0) {
      return;
    }

    for (const file of files) {
      const pendingId = newPendingAttachmentId();
      const pendingAttachment: ComposerAttachment = {
        id: pendingId,
        chat_id: undefined,
        message_id: null,
        revision_id: null,
        original_filename: file.name,
        mime_type: file.type || "application/octet-stream",
        size_bytes: file.size,
        attachment_kind: file.type.startsWith("image/") ? "image" : "text",
        status: "uploading",
        file
      };

      setAttachments((current) => [...current, pendingAttachment]);

      try {
        const attachment = await onUploadAttachment(file);
        setAttachments((current) =>
          current.map((currentAttachment) =>
            currentAttachment.id === pendingId ? attachment : currentAttachment
          )
        );
      } catch (uploadError) {
        setAttachments((current) =>
          current.map((currentAttachment) =>
            currentAttachment.id === pendingId
              ? {
                  ...currentAttachment,
                  status: "error",
                  error: uploadError instanceof Error ? uploadError.message : "Upload failed"
                }
              : currentAttachment
          )
        );
      }
    }
  }

  async function removeAttachment(attachment: ComposerAttachment) {
    if (attachment.status === "uploaded" && !attachment.isExisting && onRemoveAttachment) {
      try {
        await onRemoveAttachment(attachment);
      } catch (removeError) {
        setAttachments((current) =>
          current.map((currentAttachment) =>
            currentAttachment.id === attachment.id
              ? {
                  ...currentAttachment,
                  error: removeError instanceof Error ? removeError.message : "Delete failed"
                }
              : currentAttachment
          )
        );
        return;
      }
    }

    setAttachments((current) =>
      current.filter((currentAttachment) => currentAttachment.id !== attachment.id)
    );
  }

  return (
    <>
      {visibleWarning && <p className="composer-warning">{visibleWarning}</p>}
      <form
        className={canAttach ? "chat-composer" : "chat-composer chat-composer-no-attach"}
        onSubmit={submit}
      >
        {attachments.length > 0 && (
          <AttachmentChipList
            attachments={attachments}
            onRemove={(attachment) => void removeAttachment(attachment)}
          />
        )}
        {canAttach && (
          <div className="composer-attach">
            <input
              ref={fileInputRef}
              className="visually-hidden"
              type="file"
              multiple
              accept={attachmentAcceptTypes}
              onChange={uploadFiles}
            />
            <button
              type="button"
              aria-label="Attach files"
              title="Attach files"
              disabled={isDisabled}
              onClick={() => fileInputRef.current?.click()}
            >
              <Paperclip />
            </button>
          </div>
        )}
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
        {selectedModelInfo && (
          <div className="composer-model-capabilities">
            <CompactModelCapabilityBadges model={selectedModelInfo} />
          </div>
        )}
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
    </>
  );
}

const attachmentAcceptTypes = [
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
  "text/*",
  ".txt",
  ".md",
  ".markdown",
  ".json",
  ".csv",
  ".log",
  ".toml",
  ".yaml",
  ".yml",
  ".js",
  ".jsx",
  ".ts",
  ".tsx",
  ".css",
  ".html",
  ".rs",
  ".py",
  ".go",
  ".java",
  ".c",
  ".cpp",
  ".h",
  ".hpp",
  ".php",
  ".rb",
  ".sh"
].join(",");

function AttachmentChipList({
  attachments,
  onRemove
}: {
  attachments: ComposerAttachment[];
  onRemove: (attachment: ComposerAttachment) => void;
}) {
  return (
    <div className="composer-attachments" aria-label="Attached files">
      {attachments.map((attachment) => (
        <ComposerAttachmentItem
          key={attachment.id}
          attachment={attachment}
          onRemove={onRemove}
        />
      ))}
    </div>
  );
}

function ComposerAttachmentItem({
  attachment,
  onRemove
}: {
  attachment: ComposerAttachment;
  onRemove: (attachment: ComposerAttachment) => void;
}) {
  const imageUrl = useAttachmentImageUrl(attachment);
  const isImage = isImageAttachment(attachment);
  const className = [
    "attachment-preview",
    attachment.status === "error" ? "attachment-chip-error" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={className} title={attachment.error ?? attachment.original_filename}>
      <div className={isImage ? "attachment-preview-image" : "attachment-preview-image attachment-preview-document"}>
        {isImage && imageUrl ? (
          <img src={imageUrl} alt="" />
        ) : (
          <>
            {isImage ? <ImageIcon /> : <FileText />}
            {!isImage && <span className="attachment-type-label">{attachmentTypeLabel(attachment)}</span>}
          </>
        )}
      </div>
      <span>{attachment.original_filename}</span>
      <small>{attachment.status === "uploading" ? <RetroLoader /> : formatBytes(attachment.size_bytes)}</small>
      <button
        type="button"
        className="message-icon-button"
        aria-label={`Remove ${attachment.original_filename}`}
        onClick={() => onRemove(attachment)}
      >
        <X />
      </button>
    </div>
  );
}

function MessageAttachments({
  attachments,
  onImageOpen
}: {
  attachments: AttachmentInfo[];
  onImageOpen?: ImageOpenHandler;
}) {
  if (attachments.length === 0) {
    return null;
  }

  const imageAttachments = attachments.filter(isImageAttachment);
  const fileAttachments = attachments.filter((attachment) => !isImageAttachment(attachment));

  return (
    <>
      <div className="message-attachments" aria-label="Message attachments">
        {imageAttachments.length > 0 && (
          <MessageImageCarousel attachments={imageAttachments} onImageOpen={onImageOpen} />
        )}
        {fileAttachments.map((attachment) => (
          <a
            key={attachment.id}
            className="attachment-chip"
            href={attachmentDownloadUrl(attachment)}
            download={attachment.original_filename}
            title={attachment.original_filename}
          >
            {attachmentIcon(attachment)}
            <span>{attachment.original_filename}</span>
            <small>{formatBytes(attachment.size_bytes)}</small>
          </a>
        ))}
      </div>
    </>
  );
}

function MessageImageCarousel({
  attachments,
  onImageOpen
}: {
  attachments: AttachmentInfo[];
  onImageOpen?: ImageOpenHandler;
}) {
  const [carouselMetrics, setCarouselMetrics] = useState({
    hasOverflow: false,
    canPrevious: false,
    canNext: false,
    hiddenRightCount: 0
  });
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const pointerStartXRef = useRef<number | null>(null);
  const didSwipeRef = useRef(false);
  const hasMultipleImages = attachments.length > 1;

  const updateCarouselMetrics = useCallback(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }

    const viewRight = viewport.scrollLeft + viewport.clientWidth;
    const children = Array.from(viewport.children) as HTMLElement[];
    const nextMetrics = {
      hasOverflow: viewport.scrollWidth > viewport.clientWidth + 1,
      canPrevious: viewport.scrollLeft > 1,
      canNext: viewport.scrollWidth - viewRight > 1,
      hiddenRightCount: children.filter(
        (child) => child.offsetLeft + child.offsetWidth > viewRight + 1
      ).length
    };
    setCarouselMetrics((current) =>
      current.hasOverflow === nextMetrics.hasOverflow &&
      current.canPrevious === nextMetrics.canPrevious &&
      current.canNext === nextMetrics.canNext &&
      current.hiddenRightCount === nextMetrics.hiddenRightCount
        ? current
        : nextMetrics
    );
  }, []);

  useLayoutEffect(() => {
    updateCarouselMetrics();

    const viewport = viewportRef.current;
    if (!viewport || typeof ResizeObserver === "undefined") {
      return;
    }

    const resizeObserver = new ResizeObserver(updateCarouselMetrics);
    resizeObserver.observe(viewport);
    for (const child of Array.from(viewport.children)) {
      resizeObserver.observe(child);
    }

    return () => resizeObserver.disconnect();
  }, [attachments, updateCarouselMetrics]);

  function showPrevious() {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }

    const children = Array.from(viewport.children) as HTMLElement[];
    const currentFirstIndex = children.findIndex(
      (child) => child.offsetLeft + child.offsetWidth > viewport.scrollLeft + 1
    );
    const previous = children[Math.max((currentFirstIndex < 0 ? 0 : currentFirstIndex) - 1, 0)];
    viewport.scrollTo({ left: previous?.offsetLeft ?? 0, behavior: "smooth" });
  }

  function showNext() {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }

    const children = Array.from(viewport.children) as HTMLElement[];
    const next = children.find((child) => child.offsetLeft > viewport.scrollLeft + 1);
    viewport.scrollTo({
      left: next?.offsetLeft ?? viewport.scrollWidth - viewport.clientWidth,
      behavior: "smooth"
    });
  }

  function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    pointerStartXRef.current = event.clientX;
    didSwipeRef.current = false;
  }

  function handlePointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    const startX = pointerStartXRef.current;
    pointerStartXRef.current = null;
    if (startX === null || !hasMultipleImages || !carouselMetrics.hasOverflow) {
      return;
    }

    const deltaX = event.clientX - startX;
    if (Math.abs(deltaX) < 36) {
      return;
    }

    didSwipeRef.current = true;
    if (deltaX > 0) {
      showPrevious();
    } else {
      showNext();
    }
  }

  if (attachments.length === 0) {
    return null;
  }

  return (
    <div
      className="message-image-carousel"
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerCancel={() => {
        pointerStartXRef.current = null;
      }}
    >
      <div className="message-image-track" ref={viewportRef} onScroll={updateCarouselMetrics}>
        {attachments.map((attachment) => (
          <button
            key={attachment.id}
            type="button"
            className="message-image-button"
            aria-label={`Open ${attachment.original_filename}`}
            onClick={() => {
              if (didSwipeRef.current) {
                didSwipeRef.current = false;
                return;
              }
              onImageOpen?.(attachment, attachments);
            }}
          >
            <img
              src={attachmentDisplayUrl(attachment)}
              alt={attachment.original_filename}
              loading="lazy"
              onLoad={updateCarouselMetrics}
            />
          </button>
        ))}
      </div>
      {hasMultipleImages && carouselMetrics.hasOverflow && (
        <>
          {carouselMetrics.canPrevious && (
            <button
              type="button"
              className="message-image-nav message-image-nav-previous"
              aria-label="Previous image"
              onClick={showPrevious}
            >
              <ChevronLeft />
            </button>
          )}
          {carouselMetrics.canNext && (
            <button
              type="button"
              className="message-image-nav message-image-nav-next"
              aria-label="Next image"
              onClick={showNext}
            >
              <ChevronRight />
            </button>
          )}
          {carouselMetrics.hiddenRightCount > 0 && (
            <span className="message-image-count" aria-label={`${attachments.length} images`}>
              <ImageIcon />
              {attachments.length}
            </span>
          )}
        </>
      )}
    </div>
  );
}

function ImageViewer({
  attachments,
  index,
  onClose,
  onIndexChange
}: {
  attachments: AttachmentInfo[];
  index: number;
  onClose: () => void;
  onIndexChange: (index: number) => void;
}) {
  const attachment = attachments[index] ?? attachments[0];
  const canPrevious = index > 0;
  const canNext = index < attachments.length - 1;

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      } else if (event.key === "ArrowLeft" && canPrevious) {
        onIndexChange(index - 1);
      } else if (event.key === "ArrowRight" && canNext) {
        onIndexChange(index + 1);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [canNext, canPrevious, index, onClose, onIndexChange]);

  function closeViewer() {
    onClose();
  }

  if (!attachment) {
    return null;
  }

  return (
    <div className="image-viewer-backdrop" role="presentation" onClick={closeViewer}>
      <div className="image-viewer-top">
        {attachments.length > 1 && (
          <span className="image-viewer-count">
            {index + 1}/{attachments.length}
          </span>
        )}
        <button
          type="button"
          aria-label="Close image"
          onClick={(event) => {
            event.stopPropagation();
            closeViewer();
          }}
        >
          <X />
        </button>
      </div>
      <div className="image-viewer-body">
        <img
          className="image-viewer-image"
          src={attachmentDisplayUrl(attachment)}
          alt={attachment.original_filename}
          onClick={(event) => event.stopPropagation()}
        />
        {canPrevious && (
          <button
            type="button"
            className="image-viewer-nav image-viewer-nav-previous"
            aria-label="Previous image"
            onClick={(event) => {
              event.stopPropagation();
              onIndexChange(index - 1);
            }}
          >
            <ChevronLeft />
          </button>
        )}
        {canNext && (
          <button
            type="button"
            className="image-viewer-nav image-viewer-nav-next"
            aria-label="Next image"
            onClick={(event) => {
              event.stopPropagation();
              onIndexChange(index + 1);
            }}
          >
            <ChevronRight />
          </button>
        )}
      </div>
    </div>
  );
}

function attachmentIcon(attachment: Pick<AttachmentInfo, "attachment_kind" | "mime_type">) {
  return attachment.attachment_kind === "image" || attachment.mime_type.startsWith("image/") ? (
    <ImageIcon />
  ) : (
    <FileText />
  );
}

function attachmentTypeLabel(attachment: Pick<AttachmentInfo, "original_filename" | "mime_type">) {
  const extension = attachment.original_filename.split(".").pop()?.trim();
  if (extension && extension !== attachment.original_filename && extension.length <= 5) {
    return extension.toLocaleUpperCase();
  }

  if (attachment.mime_type.includes("json")) {
    return "JSON";
  }
  if (attachment.mime_type.includes("markdown")) {
    return "MD";
  }
  if (attachment.mime_type.startsWith("text/")) {
    return "TXT";
  }

  return "DOC";
}

function useAttachmentImageUrl(attachment: ComposerAttachment) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!isImageAttachment(attachment)) {
      setUrl(null);
      return;
    }

    if (attachment.data_url) {
      setUrl(attachment.data_url);
      return;
    }

    if (attachment.file) {
      const objectUrl = URL.createObjectURL(attachment.file);
      setUrl(objectUrl);
      return () => URL.revokeObjectURL(objectUrl);
    }

    if (attachment.status === "uploaded" && !attachment.id.startsWith("pending-")) {
      setUrl(attachmentUrl(attachment.id));
      return;
    }

    setUrl(null);
  }, [attachment]);

  return url;
}

function attachmentUrl(attachmentId: string) {
  return `/api/attachments/${attachmentId}`;
}

function attachmentDisplayUrl(attachment: AttachmentInfo) {
  return attachment.data_url ?? attachmentUrl(attachment.id);
}

function attachmentDownloadUrl(attachment: AttachmentInfo) {
  if (attachment.data_url) {
    return attachment.data_url;
  }

  if (attachment.text_content !== undefined) {
    return `data:${attachment.mime_type || "text/plain"};charset=utf-8,${encodeURIComponent(
      attachment.text_content
    )}`;
  }

  return attachmentUrl(attachment.id);
}

function prepareLocalAttachment(file: File): ComposerAttachment {
  return {
    id: newPendingAttachmentId(),
    chat_id: undefined,
    message_id: null,
    revision_id: null,
    original_filename: file.name,
    mime_type: file.type || "application/octet-stream",
    size_bytes: file.size,
    attachment_kind: file.type.startsWith("image/") ? "image" : "text",
    status: "ready",
    file
  };
}

async function preparePrivateAttachment(file: File): Promise<ComposerAttachment> {
  const mimeType = file.type || mimeTypeFromFilename(file.name) || "application/octet-stream";
  const isImage =
    mimeType.startsWith("image/") &&
    ["image/png", "image/jpeg", "image/gif", "image/webp"].includes(mimeType);
  const baseAttachment = {
    id: newPendingAttachmentId(),
    chat_id: undefined,
    message_id: null,
    revision_id: null,
    original_filename: file.name,
    mime_type: mimeType,
    size_bytes: file.size,
    created_at: unixTimestamp(),
    status: "ready" as const,
    file
  };

  if (isImage) {
    return {
      ...baseAttachment,
      attachment_kind: "image",
      data_url: await readFileAsDataUrl(file)
    };
  }

  return {
    ...baseAttachment,
    attachment_kind: "text",
    text_content: await readFileAsUtf8(file)
  };
}

function mimeTypeFromFilename(filename: string) {
  const extension = filename.split(".").pop()?.toLocaleLowerCase();
  switch (extension) {
    case "md":
    case "markdown":
      return "text/markdown";
    case "json":
      return "application/json";
    case "csv":
      return "text/csv";
    case "txt":
    case "log":
    case "toml":
    case "yaml":
    case "yml":
    case "js":
    case "jsx":
    case "ts":
    case "tsx":
    case "css":
    case "html":
    case "rs":
    case "py":
    case "go":
    case "java":
    case "c":
    case "cpp":
    case "h":
    case "hpp":
    case "php":
    case "rb":
    case "sh":
      return "text/plain";
    case "png":
      return "image/png";
    case "jpg":
    case "jpeg":
      return "image/jpeg";
    case "gif":
      return "image/gif";
    case "webp":
      return "image/webp";
    default:
      return null;
  }
}

function readFileAsDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read image"));
    reader.onload = () => {
      if (typeof reader.result !== "string") {
        reject(new Error("Failed to read image"));
        return;
      }
      resolve(reader.result);
    };
    reader.readAsDataURL(file);
  });
}

async function readFileAsUtf8(file: File) {
  try {
    const decoder = new TextDecoder("utf-8", { fatal: true });
    const text = decoder.decode(await file.arrayBuffer());
    if (text.includes("\u0000")) {
      throw new Error("Unsupported binary file");
    }
    return text;
  } catch (error) {
    throw new Error(
      error instanceof Error && error.message === "Unsupported binary file"
        ? "Only image files and UTF-8 text files are supported."
        : "Only image files and UTF-8 text files are supported."
    );
  }
}

async function uploadAttachmentToChat(chatId: string, file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`/api/chats/${chatId}/attachments`, {
    method: "POST",
    credentials: "include",
    body: formData
  });

  if (!response.ok) {
    throw new Error(await responseErrorMessage(response));
  }

  return ((await response.json()) as AttachmentResponse).attachment;
}

async function uploadComposerAttachments(chatId: string, attachments: ComposerAttachment[]) {
  const uploadedAttachments: ComposerAttachment[] = [];

  for (const attachment of attachments) {
    if (attachment.status === "uploaded") {
      uploadedAttachments.push(attachment);
      continue;
    }

    if (attachment.status === "ready" && attachment.file) {
      uploadedAttachments.push({
        ...(await uploadAttachmentToChat(chatId, attachment.file)),
        status: "uploaded" as const
      });
    }
  }

  return uploadedAttachments;
}

function attachmentReferences(attachments: ComposerAttachment[]) {
  return attachments
    .filter((attachment) => attachment.status === "uploaded")
    .map((attachment) => ({ id: attachment.id }));
}

function submittableAttachments(attachments: ComposerAttachment[]) {
  return attachments.filter(
    (attachment) => attachment.status === "ready" || attachment.status === "uploaded"
  );
}

function isImageAttachment(attachment: Pick<AttachmentInfo, "attachment_kind" | "mime_type">) {
  return attachment.attachment_kind === "image" || attachment.mime_type.startsWith("image/");
}

function newPendingAttachmentId() {
  return `pending-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function pendingComposerAttachment(id: string, file: File): ComposerAttachment {
  return {
    id,
    chat_id: undefined,
    message_id: null,
    revision_id: null,
    original_filename: file.name,
    mime_type: file.type || "application/octet-stream",
    size_bytes: file.size,
    attachment_kind: file.type.startsWith("image/") ? "image" : "text",
    status: "uploading",
    file
  };
}

async function cleanupDraftUploads(
  attachments: ComposerAttachment[],
  onRemoveAttachment: ((attachment: ComposerAttachment) => Promise<void>) | undefined
) {
  if (!onRemoveAttachment) {
    return;
  }

  await Promise.allSettled(
    attachments
      .filter((attachment) => attachment.status === "uploaded" && !attachment.isExisting)
      .map((attachment) => onRemoveAttachment(attachment))
  );
}

function formatBytes(bytes: number) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  const kib = bytes / 1024;
  if (kib < 1024) {
    return `${kib.toFixed(kib < 10 ? 1 : 0)} KB`;
  }

  const mib = kib / 1024;
  return `${mib.toFixed(mib < 10 ? 1 : 0)} MB`;
}

function ModelPicker({
  groups,
  personas,
  privatePersonas,
  isLoading,
  error,
  value,
  onChange
}: {
  groups: BackendModelGroup[];
  personas: Persona[];
  privatePersonas: PrivatePersona[];
  isLoading: boolean;
  error: string | null;
  value: string;
  onChange: (value: string) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const wrapRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const hasModels =
    groups.some((group) => group.models.length > 0) ||
    personas.length > 0 ||
    privatePersonas.length > 0;
  const selected = groups
    .flatMap((group) =>
      group.models.map((model) => ({
        backendId: group.backend.id,
        backendName: group.backend.name,
        model
      }))
    )
    .find((option) => modelValue(option.backendId, option.model.name) === value);
  const selectedPersona = personaForValue(personas, value);
  const selectedPrivatePersona = privatePersonaForValue(privatePersonas, value);
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const enabledValues = enabledModelValueSet(groups);
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
      }),
      personas: personas.filter((persona) => {
        if (!enabledValues.has(personaBaseModelValue(persona))) {
          return false;
        }
        if (persona.current_version.base_backend_id !== group.backend.id) {
          return false;
        }
        if (!normalizedQuery) {
          return true;
        }

        return (
          persona.current_version.display_name.toLocaleLowerCase().includes(normalizedQuery) ||
          persona.current_version.base_model_name.toLocaleLowerCase().includes(normalizedQuery) ||
          group.backend.name.toLocaleLowerCase().includes(normalizedQuery) ||
          (persona.owner_username ?? "").toLocaleLowerCase().includes(normalizedQuery)
        );
      }),
      privatePersonas: privatePersonas.filter((persona) => {
        if (!enabledValues.has(personaBaseModelValue(persona))) {
          return false;
        }
        if (persona.current_version.base_backend_id !== group.backend.id) {
          return false;
        }
        if (!normalizedQuery) {
          return true;
        }

        return (
          persona.current_version.display_name.toLocaleLowerCase().includes(normalizedQuery) ||
          persona.current_version.base_model_name.toLocaleLowerCase().includes(normalizedQuery) ||
          group.backend.name.toLocaleLowerCase().includes(normalizedQuery)
        );
      })
    }))
    .filter(
      (group) =>
        group.models.length > 0 || group.personas.length > 0 || group.privatePersonas.length > 0
    );

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

    return selectedPrivatePersona
      ? selectedPrivatePersona.current_version.display_name
      : selectedPersona
      ? selectedPersona.current_version.display_name
      : selected
        ? selected.model.name
        : "Select model";
  }

  return (
    <div className="model-picker" ref={wrapRef}>
      <button
        type="button"
        className="model-picker-button"
        disabled={isLoading || !hasModels}
        title={
          error ??
          selectedPrivatePersona?.current_version.base_model_name ??
          selectedPersona?.current_version.base_model_name ??
          selected?.backendName
        }
        aria-expanded={isOpen}
        onClick={() => setIsOpen((open) => !open)}
      >
        <span className="model-name">{buttonLabel()}</span>
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
                  {group.privatePersonas.map((persona) => {
                    const optionValue = privatePersonaModelValue(persona.current_version.id);
                    const baseModel = modelInfoForBase(
                      groups,
                      persona.current_version.base_backend_id,
                      persona.current_version.base_model_name
                    );
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
                        <span className="model-option-content">
                          <span className="model-name">{persona.current_version.display_name}</span>
                          <span className="model-subtitle">
                            Device · {persona.current_version.base_model_name}
                          </span>
                          <span className="model-capabilities">
                            <span className="model-capability" title="custom persona">
                              <Brain />
                              <span className="model-capability-label">custom</span>
                            </span>
                            <span className="model-capability model-capability-warning" title="device only">
                              <Lock />
                              <span className="model-capability-label">device</span>
                            </span>
                          </span>
                          {baseModel && <ModelCapabilityBadges model={baseModel} />}
                        </span>
                      </button>
                    );
                  })}
                  {group.personas.map((persona) => {
                    const optionValue = personaModelValue(persona.current_version.id);
                    const baseModel = modelInfoForBase(
                      groups,
                      persona.current_version.base_backend_id,
                      persona.current_version.base_model_name
                    );
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
                        <span className="model-option-content">
                          <span className="model-name">{persona.current_version.display_name}</span>
                          <span className="model-subtitle">
                            Custom · {persona.current_version.base_model_name}
                            {persona.owner_username ? ` · by ${persona.owner_username}` : ""}
                          </span>
                          <span className="model-capabilities">
                            <span className="model-capability" title="custom persona">
                              <Brain />
                              <span className="model-capability-label">custom</span>
                            </span>
                            <span
                              className={
                                persona.visibility === "public"
                                  ? "model-capability"
                                  : "model-capability model-capability-warning"
                              }
                              title={persona.visibility}
                            >
                              {persona.visibility === "public" ? <Users /> : <Lock />}
                              <span className="model-capability-label">
                                {persona.visibility}
                              </span>
                            </span>
                          </span>
                          {baseModel && <ModelCapabilityBadges model={baseModel} />}
                        </span>
                      </button>
                    );
                  })}
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
                        <span className="model-option-content">
                          <span className="model-name">{model.name}</span>
                          <ModelCapabilityBadges model={model} />
                        </span>
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

function ModelCapabilityBadges({ model }: { model: ModelInfo }) {
  const capabilities = modelCapabilityBadges(model);
  if (capabilities.length === 0) {
    return null;
  }

  return (
    <span className="model-capabilities" aria-label={`Capabilities: ${capabilities.join(", ")}`}>
      {capabilities.map((capability) => (
        <span key={capability} className="model-capability" title={capability}>
          {capabilityIcon(capability)}
          <span className="model-capability-label">{capabilityLabel(capability)}</span>
        </span>
      ))}
    </span>
  );
}

function CompactModelCapabilityBadges({ model }: { model: ModelInfo }) {
  const [activeCapability, setActiveCapability] = useState<string | null>(null);
  const capabilities = modelCapabilityBadges(model);
  if (capabilities.length === 0) {
    return null;
  }

  return (
    <span
      className="model-capabilities model-capabilities-icon-only"
      aria-label={`Capabilities: ${capabilities.join(", ")}`}
      onMouseLeave={() => setActiveCapability(null)}
    >
      {capabilities.map((capability) => (
        <button
          key={capability}
          type="button"
          className="model-capability model-capability-icon-only"
          title={capabilityLabel(capability)}
          aria-label={capabilityLabel(capability)}
          onBlur={() => window.setTimeout(() => setActiveCapability(null), 120)}
          onClick={() =>
            setActiveCapability((current) => (current === capability ? null : capability))
          }
        >
          {capabilityIcon(capability)}
        </button>
      ))}
      {activeCapability && (
        <span className="model-capability-popover">{capabilityLabel(activeCapability)}</span>
      )}
    </span>
  );
}

function modelCapabilityBadges(model: ModelInfo) {
  const capabilities = new Set(
    (model.capabilities ?? [])
      .map((capability) => capability.trim().toLocaleLowerCase())
      .filter(Boolean)
  );

  if (model.supports_images) {
    capabilities.add("vision");
  }
  if (model.supports_thinking) {
    capabilities.add("thinking");
  }

  capabilities.delete("completion");
  return [...capabilities].sort((left, right) => capabilityOrder(left) - capabilityOrder(right));
}

function capabilityOrder(capability: string) {
  const order = ["vision", "image", "thinking", "tools", "audio"];
  const index = order.indexOf(capability);
  return index === -1 ? order.length : index;
}

function capabilityIcon(capability: string) {
  switch (capability) {
    case "vision":
    case "image":
      return <ImageIcon />;
    case "thinking":
      return <Brain />;
    case "tools":
      return <Wrench />;
    case "audio":
      return <Volume2 />;
    default:
      return null;
  }
}

function capabilityLabel(capability: string) {
  switch (capability) {
    case "vision":
      return "vision";
    case "image":
      return "image";
    case "thinking":
      return "think";
    case "tools":
      return "tools";
    case "audio":
      return "audio";
    default:
      return capability;
  }
}

function SettingsPage({
  currentUser,
  activeSection,
  onBackendsChanged,
  onPersonasChanged,
  onPrivatePersonasChanged,
  onAppSettingsGuardChange,
  onSelectSection,
  isAdmin
}: {
  currentUser: User;
  activeSection: SettingsSection;
  onBackendsChanged: () => Promise<void>;
  onPersonasChanged: () => Promise<void>;
  onPrivatePersonasChanged: () => Promise<void>;
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
    { id: "personas", label: "Personas", icon: <Brain /> },
    { id: "users", label: "Users", icon: <Users />, adminOnly: true },
    { id: "backends", label: "Backends", icon: <Server />, adminOnly: true },
    { id: "models", label: "Models", icon: <Wrench />, adminOnly: true },
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
        {selectedSection === "personas" && (
          <PersonasPanel
            onPersonasChanged={onPersonasChanged}
            onPrivatePersonasChanged={onPrivatePersonasChanged}
          />
        )}
        {selectedSection === "users" && isAdmin && <AdminUsersPanel currentUserId={currentUser.id} />}
        {selectedSection === "backends" && isAdmin && (
          <BackendsPanel onBackendsChanged={onBackendsChanged} />
        )}
        {selectedSection === "models" && isAdmin && (
          <ModelsAccessPanel onModelsChanged={onBackendsChanged} />
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

function PersonasPanel({
  onPersonasChanged,
  onPrivatePersonasChanged
}: {
  onPersonasChanged: () => Promise<void>;
  onPrivatePersonasChanged: () => Promise<void>;
}) {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [privatePersonas, setPrivatePersonas] = useState<PrivatePersona[]>([]);
  const [modelGroups, setModelGroups] = useState<BackendModelGroup[]>([]);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [busyPersonaId, setBusyPersonaId] = useState<string | null>(null);
  const [editingPersona, setEditingPersona] = useState<Persona | null>(null);
  const [editingPrivatePersona, setEditingPrivatePersona] = useState<PrivatePersona | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Persona | null>(null);
  const [deletePrivateTarget, setDeletePrivateTarget] = useState<PrivatePersona | null>(null);
  const [displayName, setDisplayName] = useState("");
  const [storageMode, setStorageMode] = useState("local");
  const [selectedBaseModel, setSelectedBaseModel] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadPersonas = useCallback(async () => {
    setIsRefreshing(true);
    setError(null);

    try {
      const [personaResponse, privatePersonaResponse, modelsResponse] = await Promise.all([
        requestJson<PersonasResponse>("/api/personas"),
        listPrivatePersonas(),
        requestJson<ModelsResponse>("/api/models")
      ]);
      setPersonas(personaResponse.personas);
      setPrivatePersonas(privatePersonaResponse);
      setModelGroups(modelsResponse.backends);
      setHasLoaded(true);
      setSelectedBaseModel((current) => current || firstModelValue(modelsResponse.backends));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load personas");
      setHasLoaded(true);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadPersonas();
  }, [loadPersonas]);

  function resetDraft() {
    setEditingPersona(null);
    setEditingPrivatePersona(null);
    setDisplayName("");
    setStorageMode("local");
    setSystemPrompt("");
    setSelectedBaseModel(firstModelValue(modelGroups));
    setError(null);
    setStatus(null);
  }

  function startEditingPersona(persona: Persona) {
    const version = persona.current_version;
    setEditingPersona(persona);
    setEditingPrivatePersona(null);
    setDisplayName(version.display_name);
    setStorageMode(persona.visibility);
    setSelectedBaseModel(modelValue(version.base_backend_id, version.base_model_name));
    setSystemPrompt(version.system_prompt);
    setError(null);
    setStatus(null);
  }

  function startEditingPrivatePersona(persona: PrivatePersona) {
    const version = persona.current_version;
    setEditingPrivatePersona(persona);
    setEditingPersona(null);
    setDisplayName(version.display_name);
    setStorageMode("local");
    setSelectedBaseModel(modelValue(version.base_backend_id, version.base_model_name));
    setSystemPrompt(version.system_prompt);
    setError(null);
    setStatus(null);
  }

  async function savePersona(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const selected = modelParts(selectedBaseModel);
    if (!selected) {
      setError("Select a base model for this persona");
      return;
    }

    setIsSaving(true);
    setError(null);
    setStatus(null);

    try {
      const backendName = backendNameFor(modelGroups, selected.backendId);
      if (storageMode === "local") {
        const body = {
          displayName,
          baseBackendId: selected.backendId,
          baseBackendName: backendName,
          baseModelName: selected.modelName,
          systemPrompt
        };
        if (editingPrivatePersona) {
          await updatePrivatePersona(editingPrivatePersona.id, body);
          setStatus("Private persona updated on this device.");
        } else {
          await createPrivatePersona(body);
          setStatus("Private persona created on this device.");
        }
        resetDraft();
        await loadPersonas();
        await onPrivatePersonasChanged();
        return;
      }

      const body = {
        visibility: storageMode,
        display_name: displayName,
        avatar_attachment_id: null,
        base_backend_id: selected.backendId,
        base_model_name: selected.modelName,
        system_prompt: systemPrompt,
        tool_policy_json: null
      };
      if (editingPersona) {
        await requestJson<PersonaMutationResponse>(`/api/personas/${editingPersona.id}`, {
          method: "PATCH",
          body: JSON.stringify(body)
        });
        setStatus("Persona updated.");
      } else {
        await requestJson<PersonaMutationResponse>("/api/personas", {
          method: "POST",
          body: JSON.stringify(body)
        });
        setStatus("Persona created.");
      }
      resetDraft();
      await loadPersonas();
      await onPersonasChanged();
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save persona");
    } finally {
      setIsSaving(false);
    }
  }

  async function copyPersona(persona: Persona) {
    setBusyPersonaId(persona.id);
    setError(null);
    setStatus(null);

    try {
      await requestJson<PersonaMutationResponse>(`/api/personas/${persona.id}/copy`, {
        method: "POST",
        body: JSON.stringify({
          persona_version_id: persona.current_version.id,
          visibility: "private"
        })
      });
      setStatus("Persona copied to your private server personas.");
      await loadPersonas();
      await onPersonasChanged();
    } catch (copyError) {
      setError(copyError instanceof Error ? copyError.message : "Failed to copy persona");
    } finally {
      setBusyPersonaId(null);
    }
  }

  async function copyPersonaToDevice(persona: Persona) {
    const version = persona.current_version;
    setBusyPersonaId(persona.id);
    setError(null);
    setStatus(null);

    try {
      await createPrivatePersona({
        displayName: version.display_name,
        baseBackendId: version.base_backend_id,
        baseBackendName: backendNameFor(modelGroups, version.base_backend_id),
        baseModelName: version.base_model_name,
        systemPrompt: version.system_prompt,
        sourcePersonaId: persona.id,
        sourcePersonaVersionId: version.id
      });
      setStatus("Persona copied to this device for private chats.");
      await loadPersonas();
      await onPrivatePersonasChanged();
    } catch (copyError) {
      setError(copyError instanceof Error ? copyError.message : "Failed to copy persona to device");
    } finally {
      setBusyPersonaId(null);
    }
  }

  async function disownPersona(persona: Persona) {
    setBusyPersonaId(persona.id);
    setError(null);
    setStatus(null);

    try {
      await requestJson(`/api/personas/${persona.id}/disown`, { method: "POST" });
      setDeleteTarget(null);
      setStatus(persona.visibility === "public" ? "Persona removed." : "Persona deleted.");
      if (editingPersona?.id === persona.id) {
        resetDraft();
      }
      await loadPersonas();
      await onPersonasChanged();
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Failed to remove persona");
      setDeleteTarget(null);
    } finally {
      setBusyPersonaId(null);
    }
  }

  async function deletePrivatePersonaTarget(persona: PrivatePersona) {
    setBusyPersonaId(persona.id);
    setError(null);
    setStatus(null);

    try {
      await deletePrivatePersona(persona.id);
      setDeletePrivateTarget(null);
      setStatus("Private persona deleted from this device.");
      if (editingPrivatePersona?.id === persona.id) {
        resetDraft();
      }
      await loadPersonas();
      await onPrivatePersonasChanged();
    } catch (deleteError) {
      setError(
        deleteError instanceof Error ? deleteError.message : "Failed to delete private persona"
      );
      setDeletePrivateTarget(null);
    } finally {
      setBusyPersonaId(null);
    }
  }

  const canSave =
    displayName.trim() !== "" &&
    selectedBaseModel !== "" &&
    !(editingPersona && storageMode === "local") &&
    !(editingPrivatePersona && storageMode !== "local");

  return (
    <div className="settings-section">
      <div className="section-header">
        <div>
          <p className="eyebrow">Custom Models</p>
          <h1>Personas</h1>
        </div>
        <button
          type="button"
          className="secondary-button refresh-button"
          onClick={() => void loadPersonas()}
          disabled={isRefreshing}
        >
          {isRefreshing ? <RetroLoader /> : <RefreshCw />}
          <span>{isRefreshing ? "Loading" : "Refresh"}</span>
        </button>
      </div>

      {error && <p className="error">{error}</p>}
      {status && <p className="status-message">{status}</p>}

      <form className="settings-form persona-form" onSubmit={savePersona}>
        <label>
          <span>Name</span>
          <input
            required
            maxLength={80}
            value={displayName}
            onChange={(event) => setDisplayName(event.target.value)}
            placeholder="Careful Researcher"
          />
        </label>
        <label>
          <span>Storage</span>
          <select value={storageMode} onChange={(event) => setStorageMode(event.target.value)}>
            <option value="local">This device only</option>
            <option value="private">Server private</option>
            <option value="public">Server public</option>
          </select>
        </label>
        <label>
          <span>Base Model</span>
          <select
            required
            value={selectedBaseModel}
            onChange={(event) => setSelectedBaseModel(event.target.value)}
          >
            <option value="">Select a model</option>
            {modelGroups.map((group) => (
              <optgroup key={group.backend.id} label={group.backend.name}>
                {group.models.map((model) => (
                  <option
                    key={modelValue(group.backend.id, model.name)}
                    value={modelValue(group.backend.id, model.name)}
                  >
                    {model.name}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>
        <label>
          <span>System Prompt</span>
          <textarea
            rows={8}
            value={systemPrompt}
            onChange={(event) => setSystemPrompt(event.target.value)}
            placeholder="Describe how this persona should behave."
          />
        </label>
        <div className="persona-form-actions">
          {(editingPersona || editingPrivatePersona) && (
            <button
              type="button"
              className="secondary-button"
              disabled={isSaving}
              onClick={resetDraft}
            >
              <X />
              <span>Cancel</span>
            </button>
          )}
          <button type="submit" disabled={isSaving || !canSave}>
            {editingPersona ? <Save /> : <Plus />}
            <span>
              {isSaving
                ? "Saving..."
                : editingPersona || editingPrivatePersona
                  ? "Save Persona"
                  : "Create Persona"}
            </span>
          </button>
        </div>
        <p className="status-message">
          Device personas can be used in private chats without storing their name or prompt on the server.
          Server personas are available from signed-in standard chats.
        </p>
      </form>

      {!hasLoaded && <p className="status-message">Loading personas...</p>}
      {hasLoaded && personas.length === 0 && privatePersonas.length === 0 && (
        <p className="status-message">No personas created yet.</p>
      )}

      <div className="persona-list">
        {privatePersonas.map((persona) => (
          <PrivatePersonaRow
            key={persona.id}
            persona={persona}
            isBusy={busyPersonaId === persona.id}
            onDelete={() => setDeletePrivateTarget(persona)}
            onEdit={() => startEditingPrivatePersona(persona)}
          />
        ))}
        {personas.map((persona) => (
          <PersonaRow
            key={persona.id}
            persona={persona}
            backendName={backendNameFor(modelGroups, persona.current_version.base_backend_id)}
            isBusy={busyPersonaId === persona.id}
            canEdit={persona.is_owner && persona.lifecycle_state === "active"}
            onCopy={() => void copyPersona(persona)}
            onCopyToDevice={() => void copyPersonaToDevice(persona)}
            onDelete={() => setDeleteTarget(persona)}
            onEdit={() => startEditingPersona(persona)}
          />
        ))}
      </div>

      {deleteTarget && (
        <ConfirmDialog
          title={deleteTarget.visibility === "public" ? "Remove Persona" : "Delete Persona"}
          message={
            deleteTarget.visibility === "public"
              ? `Remove "${deleteTarget.current_version.display_name}" from your available personas? If other users have used it, their existing chats keep working.`
              : `Delete "${deleteTarget.current_version.display_name}"? Existing chats keep their stored message labels, but this persona will disappear from your picker.`
          }
          confirmLabel={deleteTarget.visibility === "public" ? "Remove" : "Delete"}
          isBusy={busyPersonaId === deleteTarget.id}
          onCancel={() => setDeleteTarget(null)}
          onConfirm={() => void disownPersona(deleteTarget)}
        />
      )}
      {deletePrivateTarget && (
        <ConfirmDialog
          title="Delete Private Persona"
          message={`Delete "${deletePrivateTarget.current_version.display_name}" from this device? Server chats and hosted personas are not affected.`}
          confirmLabel="Delete"
          isBusy={busyPersonaId === deletePrivateTarget.id}
          onCancel={() => setDeletePrivateTarget(null)}
          onConfirm={() => void deletePrivatePersonaTarget(deletePrivateTarget)}
        />
      )}
    </div>
  );
}

function PersonaRow({
  persona,
  backendName,
  isBusy,
  canEdit,
  onCopy,
  onCopyToDevice,
  onDelete,
  onEdit
}: {
  persona: Persona;
  backendName: string;
  isBusy: boolean;
  canEdit: boolean;
  onCopy: () => void;
  onCopyToDevice: () => void;
  onDelete: () => void;
  onEdit: () => void;
}) {
  const version = persona.current_version;
  return (
    <article className="persona-row">
      <div className="persona-main">
        <div>
          <h2>{version.display_name}</h2>
          <p>
            {backendName} / {version.base_model_name}
          </p>
          {persona.owner_username && <p>by {persona.owner_username}</p>}
        </div>
        <div className="badges">
          <span className="badge">custom</span>
          <span className={persona.visibility === "public" ? "badge" : "badge badge-warning"}>
            {persona.visibility}
          </span>
          <span className="badge">v{version.version_number}</span>
          {persona.is_owner && <span className="badge">yours</span>}
        </div>
      </div>
      <details className="persona-prompt">
        <summary>System prompt</summary>
        <pre>{version.system_prompt || "No system prompt."}</pre>
      </details>
      <div className="persona-actions">
        {canEdit && (
          <button type="button" className="secondary-button" disabled={isBusy} onClick={onEdit}>
            <Pencil />
            <span>Edit</span>
          </button>
        )}
        <button type="button" className="secondary-button" disabled={isBusy} onClick={onCopy}>
          <Copy />
          <span>Copy</span>
        </button>
        <button
          type="button"
          className="secondary-button"
          disabled={isBusy}
          onClick={onCopyToDevice}
        >
          <Lock />
          <span>Device</span>
        </button>
        {(persona.is_owner || persona.is_member) && (
          <button type="button" className="danger-button" disabled={isBusy} onClick={onDelete}>
            <Trash2 />
            <span>{persona.visibility === "public" ? "Remove" : "Delete"}</span>
          </button>
        )}
      </div>
    </article>
  );
}

function PrivatePersonaRow({
  persona,
  isBusy,
  onDelete,
  onEdit
}: {
  persona: PrivatePersona;
  isBusy: boolean;
  onDelete: () => void;
  onEdit: () => void;
}) {
  const version = persona.current_version;
  return (
    <article className="persona-row">
      <div className="persona-main">
        <div>
          <h2>{version.display_name}</h2>
          <p>
            {version.base_backend_name} / {version.base_model_name}
          </p>
          <p>stored on this device</p>
        </div>
        <div className="badges">
          <span className="badge">custom</span>
          <span className="badge badge-warning">device</span>
          <span className="badge">v{version.version_number}</span>
        </div>
      </div>
      <details className="persona-prompt">
        <summary>System prompt</summary>
        <pre>{version.system_prompt || "No system prompt."}</pre>
      </details>
      <div className="persona-actions">
        <button type="button" className="secondary-button" disabled={isBusy} onClick={onEdit}>
          <Pencil />
          <span>Edit</span>
        </button>
        <button type="button" className="danger-button" disabled={isBusy} onClick={onDelete}>
          <Trash2 />
          <span>Delete</span>
        </button>
      </div>
    </article>
  );
}

function firstModelValue(groups: BackendModelGroup[]) {
  const firstGroup = groups.find((group) => group.models.length > 0);
  const firstModel = firstGroup?.models[0];
  return firstGroup && firstModel ? modelValue(firstGroup.backend.id, firstModel.name) : "";
}

function backendNameFor(groups: BackendModelGroup[], backendId: string) {
  return groups.find((group) => group.backend.id === backendId)?.backend.name ?? backendId;
}

function AdminUsersPanel({ currentUserId }: { currentUserId: string }) {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isCreatingUser, setIsCreatingUser] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busyUserId, setBusyUserId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<AdminUser | null>(null);
  const [newUsername, setNewUsername] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newRole, setNewRole] = useState("user");
  const [newDisabled, setNewDisabled] = useState(false);

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

  async function createUser(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsCreatingUser(true);
    setError(null);

    try {
      await requestJson<AdminUserMutationResponse>("/api/admin/users", {
        method: "POST",
        body: JSON.stringify({
          username: newUsername,
          email: newEmail.trim() === "" ? null : newEmail,
          password: newPassword,
          role: newRole,
          is_disabled: newDisabled
        })
      });
      setNewUsername("");
      setNewEmail("");
      setNewPassword("");
      setNewRole("user");
      setNewDisabled(false);
      await loadUsers();
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "Failed to create user");
    } finally {
      setIsCreatingUser(false);
    }
  }

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

      <form className="settings-form user-create-form" onSubmit={createUser}>
        <label>
          <span>Username</span>
          <input
            required
            value={newUsername}
            onChange={(event) => setNewUsername(event.target.value)}
          />
        </label>
        <label>
          <span>Email</span>
          <input
            type="email"
            value={newEmail}
            onChange={(event) => setNewEmail(event.target.value)}
          />
        </label>
        <label>
          <span>Password</span>
          <input
            required
            type="password"
            value={newPassword}
            onChange={(event) => setNewPassword(event.target.value)}
          />
        </label>
        <label>
          <span>Role</span>
          <select value={newRole} onChange={(event) => setNewRole(event.target.value)}>
            <option value="user">User</option>
            <option value="admin">Admin</option>
          </select>
        </label>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={newDisabled}
            onChange={(event) => setNewDisabled(event.target.checked)}
          />
          <span>Create disabled</span>
        </label>
        <button
          type="submit"
          disabled={isCreatingUser || newUsername.trim() === "" || newPassword === ""}
        >
          <Plus />
          <span>{isCreatingUser ? "Creating..." : "Create User"}</span>
        </button>
      </form>

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

function ModelsAccessPanel({ onModelsChanged }: { onModelsChanged: () => Promise<void> }) {
  const [groups, setGroups] = useState<AdminBackendModelGroup[]>([]);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [busyModelKey, setBusyModelKey] = useState<string | null>(null);
  const [busyBackendId, setBusyBackendId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadAdminModels = useCallback(async () => {
    setIsRefreshing(true);
    setError(null);

    try {
      const response = await requestJson<AdminModelsResponse>("/api/admin/models");
      setGroups(response.backends);
      setHasLoaded(true);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load models");
      setHasLoaded(true);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadAdminModels();
  }, [loadAdminModels]);

  async function toggleModel(backendId: string, modelName: string, isEnabled: boolean) {
    const key = modelValue(backendId, modelName);
    setBusyModelKey(key);
    setError(null);
    setStatus(null);

    try {
      await requestJson("/api/admin/models", {
        method: "PATCH",
        body: JSON.stringify({
          backend_id: backendId,
          model_name: modelName,
          is_enabled: isEnabled
        })
      });
      setGroups((current) =>
        current.map((group) =>
          group.backend.id === backendId
            ? {
                ...group,
                models: group.models.map((model) =>
                  model.name === modelName ? { ...model, is_enabled: isEnabled } : model
                )
              }
            : group
        )
      );
      setStatus(isEnabled ? "Model enabled." : "Model disabled.");
      await onModelsChanged();
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : "Failed to update model");
    } finally {
      setBusyModelKey(null);
    }
  }

  async function toggleBackend(group: AdminBackendModelGroup, isEnabled: boolean) {
    setBusyBackendId(group.backend.id);
    setError(null);
    setStatus(null);

    try {
      await requestJson("/api/admin/models/backend", {
        method: "PATCH",
        body: JSON.stringify({
          backend_id: group.backend.id,
          model_names: group.models.map((model) => model.name),
          is_enabled: isEnabled
        })
      });
      setGroups((current) =>
        current.map((currentGroup) =>
          currentGroup.backend.id === group.backend.id
            ? {
                ...currentGroup,
                models: currentGroup.models.map((model) => ({
                  ...model,
                  is_enabled: isEnabled
                }))
              }
            : currentGroup
        )
      );
      setStatus(isEnabled ? "Backend models enabled." : "Backend models disabled.");
      await onModelsChanged();
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : "Failed to update models");
    } finally {
      setBusyBackendId(null);
    }
  }

  return (
    <div className="settings-section">
      <div className="section-header">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>Models</h1>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="secondary-button refresh-button"
            onClick={() => void loadAdminModels()}
            disabled={isRefreshing}
          >
            {isRefreshing ? <RetroLoader /> : <RefreshCw />}
            <span>{isRefreshing ? "Loading" : "Refresh"}</span>
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}
      {status && <p className="status-message">{status}</p>}
      {!hasLoaded && <p className="status-message">Loading models...</p>}
      {hasLoaded && groups.length === 0 && (
        <p className="status-message">No enabled Ollama backends are configured.</p>
      )}

      <div className="model-access-list">
        {groups.map((group) => {
          const enabledCount = group.models.filter((model) => model.is_enabled).length;
          const isBackendBusy = busyBackendId === group.backend.id;

          return (
            <section key={group.backend.id} className="model-access-group">
              <div className="model-access-header">
                <div>
                  <h2>{group.backend.name}</h2>
                  <p>
                    {enabledCount} of {group.models.length} enabled
                  </p>
                </div>
                <div className="model-access-actions">
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={isBackendBusy || group.models.length === 0}
                    onClick={() => void toggleBackend(group, true)}
                  >
                    Enable All
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={isBackendBusy || group.models.length === 0}
                    onClick={() => void toggleBackend(group, false)}
                  >
                    Disable All
                  </button>
                </div>
              </div>

              {group.models.length === 0 ? (
                <p className="status-message">No models returned by this backend.</p>
              ) : (
                <div className="model-access-models">
                  {group.models.map((model) => {
                    const key = modelValue(group.backend.id, model.name);
                    const isBusy = busyModelKey === key || isBackendBusy;

                    return (
                      <label key={key} className="model-access-row">
                        <span className="model-access-main">
                          <span className="model-name">{model.name}</span>
                          <ModelCapabilityBadges model={model} />
                        </span>
                        <span className="model-access-toggle">
                          <input
                            type="checkbox"
                            checked={model.is_enabled}
                            disabled={isBusy}
                            onChange={(event) =>
                              void toggleModel(
                                group.backend.id,
                                model.name,
                                event.target.checked
                              )
                            }
                          />
                          <span>{model.is_enabled ? "On" : "Off"}</span>
                        </span>
                      </label>
                    );
                  })}
                </div>
              )}
            </section>
          );
        })}
      </div>
    </div>
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
