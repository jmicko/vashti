import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import {
  Cog,
  Download,
  GitFork,
  LogOut,
  Menu,
  RefreshCw,
  Settings as SettingsIcon,
  X
} from "lucide-react";
import { requestJson } from "./api";
import {
  ImageViewer,
  isImageAttachment,
  uploadComposerAttachments
} from "./attachments";
import { ChatHome } from "./ChatHome";
import { ConfirmDialog, RetroLoader } from "./common";
import { ModelPicker } from "./ModelPicker";
import { ModelSettingsMenu } from "./ModelSettingsMenu";
import { storeCustomModelDraft, type CustomModelDraft } from "./customModelDraft";
import {
  enabledModelValueSet,
  modelInfoForValue,
  modelParts,
  modelValue,
  personaBaseModelValue,
  personaModelValue,
  personaVersionIdFromValue,
  privatePersonaVersionForValue,
  privatePersonaVersionIdFromValue,
  privatePersonaWithVersionForValue,
  selectedModelBaseParts
} from "./modelSelection";
import { Sidebar } from "./Sidebar";
import { UnsavedSettingsDialog } from "./settingsControls";
import { SettingsPage } from "./settingsPage";
import { markPerformance, measurePerformance } from "./performance";
import {
  pathForRoute,
  routeFromLocation,
  routesEqual,
  storedNewChatMode,
  storeNewChatMode
} from "./routing";
import {
  defaultToolPreferences,
} from "./toolPreferences";
import { applyTheme, normalizeTheme, storeAndApplyTheme, storedTheme } from "./theme";
import { usePwa } from "./pwa";
import {
  createPrivateChat,
  deleteCachedHostedChat,
  deleteHostedPendingSend,
  deletePrivateChat,
  getCachedHostedChatList,
  getCachedModelState,
  getPrivateChat,
  listPrivatePersonas,
  listPrivateChats,
  listPrivateContextLibrary,
  renamePrivateChat,
  resetPrivateStorageUser,
  saveCachedModelState,
  saveCachedHostedChatList,
  savePrivateChat,
  unixTimestamp,
  type PrivateChatSummary,
  type PrivatePersona,
  type PrivatePersonaVersion
} from "./privateChatStore";
import type {
  AppRoute,
  AppSettingsGuard,
  AttachmentInfo,
  AvailableTool,
  AvailableToolsResponse,
  BackendModelGroup,
  ChatResponse,
  ChatInferenceSettings,
  ContextBlockSelection,
  ContextLibraryResponse,
  ChatSummary,
  ChatToolPreferences,
  ComposerAttachment,
  ComposerSubmitPayload,
  ImageViewerState,
  ListChatsResponse,
  ModelPickerCache,
  ModelsResponse,
  NewChatMode,
  Persona,
  PersonaVersion,
  PersonasResponse,
  SettingsSection,
  ThinkingMode,
  User,
  UserSettings,
  UpdateStatusResponse,
  VersionResponse
} from "./types";

const UPDATE_STATUS_POLL_MS = 15 * 60 * 1000;

const ChatView = lazy(() =>
  import("./ChatView").then((module) => ({ default: module.ChatView }))
);
const PrivateChatView = lazy(() =>
  import("./PrivateChatView").then((module) => ({ default: module.PrivateChatView }))
);

export function AppShell({
  user,
  onSessionChanged,
  onUserChanged
}: {
  user: User;
  onSessionChanged: () => Promise<void>;
  onUserChanged: (user: User) => void;
}) {
  const [route, setRoute] = useState<AppRoute>(() => routeFromLocation());
  const routeRef = useRef(route);
  const appSettingsGuardRef = useRef<AppSettingsGuard | null>(null);
  const settingsMenuRef = useRef<HTMLDivElement>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isMessageTreeOpen, setIsMessageTreeOpen] = useState(false);
  const [isSettingsMenuOpen, setIsSettingsMenuOpen] = useState(false);
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const [pendingNavigation, setPendingNavigation] = useState<AppRoute | null>(null);
  const [isSavingPendingNavigation, setIsSavingPendingNavigation] = useState(false);
  const [modelGroups, setModelGroups] = useState<BackendModelGroup[]>([]);
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [privatePersonas, setPrivatePersonas] = useState<PrivatePersona[]>([]);
  const [personaVersions, setPersonaVersions] = useState<Record<string, PersonaVersion>>({});
  const [privatePersonaVersions, setPrivatePersonaVersions] = useState<
    Record<string, PrivatePersonaVersion>
  >({});
  const knownPersonaVersions = useMemo(() => Object.values(personaVersions), [personaVersions]);
  const knownPrivatePersonaVersions = useMemo(
    () => Object.values(privatePersonaVersions),
    [privatePersonaVersions]
  );
  const [availableTools, setAvailableTools] = useState<AvailableTool[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const hasHydratedModelCacheRef = useRef(false);
  const hasModelPickerDataRef = useRef(false);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const chatsRef = useRef<ChatSummary[]>([]);
  const hasHydratedChatListCacheRef = useRef(false);
  const [privateChats, setPrivateChats] = useState<PrivateChatSummary[]>([]);

  useEffect(() => {
    markPerformance("vashti:shell-mounted");
    measurePerformance(
      "vashti:startup-to-shell",
      "vashti:app-start",
      "vashti:shell-mounted"
    );
  }, []);
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
  const [chatSystemPromptOverride, setChatSystemPromptOverride] = useState<string | null>(null);
  const [chatInferenceSettings, setChatInferenceSettings] = useState<ChatInferenceSettings>({});
  const [chatContextBlocks, setChatContextBlocks] = useState<ContextBlockSelection[]>([]);
  const [serverContextLibrary, setServerContextLibrary] = useState<ContextLibraryResponse>({
    categories: [],
    blocks: []
  });
  const [deviceContextLibrary, setDeviceContextLibrary] = useState<ContextLibraryResponse>({
    categories: [],
    blocks: []
  });
  const [error, setError] = useState<string | null>(null);
  const [updateStatus, setUpdateStatus] = useState<UpdateStatusResponse | null>(null);
  const [updateStatusError, setUpdateStatusError] = useState<string | null>(null);
  const { reloadLatestFrontend } = usePwa();
  const isAdmin = user.role === "admin";
  const page = route.page;
  const isSettingsPage = page === "settings";
  const settingsSection = route.page === "settings" ? route.section : "profile";
  const currentChatId = route.page === "chat" ? route.chatId ?? null : null;
  const currentPrivateChatId = route.page === "private-chat" ? route.chatId : null;
  const isNewChatDraft = page === "chat" && !currentChatId;
  const allowPrivatePersonaSelection =
    page === "private-chat" || (page === "chat" && !currentChatId && newChatMode === "private");
  const isHostedPersonaUnavailableForPrivateDraft =
    isNewChatDraft && newChatMode === "private" && Boolean(personaVersionIdFromValue(selectedModel));
  const activeSelectedModel = isHostedPersonaUnavailableForPrivateDraft ? "" : selectedModel;
  const imageViewerRef = useRef<ImageViewerState | null>(null);
  const activeUpdateVersion =
    updateStatus?.operation.state === "requested" ||
    updateStatus?.operation.state === "installing"
      ? updateStatus.operation.version
      : null;

  const refreshUpdateStatus = useCallback(async () => {
    if (!isAdmin) {
      return null;
    }

    try {
      const response = await requestJson<UpdateStatusResponse>("/api/admin/update");
      setUpdateStatus(response);
      setUpdateStatusError(null);
      return response;
    } catch (statusError) {
      setUpdateStatusError(
        statusError instanceof Error ? statusError.message : "Failed to load update status"
      );
      return null;
    }
  }, [isAdmin]);

  useEffect(() => {
    if (!isAdmin) {
      setUpdateStatus(null);
      setUpdateStatusError(null);
      return;
    }

    void refreshUpdateStatus();
    const startupRefresh = window.setTimeout(() => void refreshUpdateStatus(), 5000);
    const interval = window.setInterval(() => void refreshUpdateStatus(), UPDATE_STATUS_POLL_MS);
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        void refreshUpdateStatus();
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      window.clearTimeout(startupRefresh);
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [isAdmin, refreshUpdateStatus]);

  useEffect(() => {
    if (!activeUpdateVersion) {
      return;
    }

    let cancelled = false;
    let timeout: number | null = null;

    async function pollUpdate() {
      try {
        const version = await requestJson<VersionResponse>("/api/version");
        if (`v${version.version.replace(/^v/, "")}` === activeUpdateVersion) {
          await reloadLatestFrontend();
          return;
        }
      } catch {
        // The server is expected to disappear briefly while systemd replaces it.
      }

      try {
        const status = await requestJson<UpdateStatusResponse>("/api/admin/update");
        if (!cancelled) {
          setUpdateStatus(status);
          setUpdateStatusError(null);
        }
        if (
          status.operation.version === activeUpdateVersion &&
          status.operation.state === "succeeded"
        ) {
          await reloadLatestFrontend();
          return;
        }
        if (
          status.operation.version === activeUpdateVersion &&
          ["failed", "rolled_back"].includes(status.operation.state)
        ) {
          return;
        }
      } catch {
        // Continue polling until either the new server or a rolled-back server responds.
      }

      if (!cancelled) {
        timeout = window.setTimeout(() => void pollUpdate(), 1000);
      }
    }

    void pollUpdate();
    return () => {
      cancelled = true;
      if (timeout !== null) {
        window.clearTimeout(timeout);
      }
    };
  }, [activeUpdateVersion, reloadLatestFrontend]);

  useEffect(() => {
    routeRef.current = route;
  }, [route]);

  useEffect(() => {
    setIsMessageTreeOpen(false);
  }, [currentChatId, currentPrivateChatId]);

  useEffect(() => {
    if (!isSettingsMenuOpen) {
      return;
    }

    function closeSettingsMenuOnOutsidePointer(event: PointerEvent) {
      const target = event.target;
      if (target instanceof Node && settingsMenuRef.current?.contains(target)) {
        return;
      }

      setIsSettingsMenuOpen(false);
    }

    function closeSettingsMenuOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsSettingsMenuOpen(false);
      }
    }

    window.addEventListener("pointerdown", closeSettingsMenuOnOutsidePointer);
    window.addEventListener("keydown", closeSettingsMenuOnEscape);
    return () => {
      window.removeEventListener("pointerdown", closeSettingsMenuOnOutsidePointer);
      window.removeEventListener("keydown", closeSettingsMenuOnEscape);
    };
  }, [isSettingsMenuOpen]);

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
    setChatSystemPromptOverride(null);
    setChatInferenceSettings({});
    setChatContextBlocks([]);
  }, [route]);

  const updateAppSettingsGuard = useCallback((guard: AppSettingsGuard | null) => {
    appSettingsGuardRef.current = guard;
  }, []);

  function setNewChatMode(mode: NewChatMode) {
    setNewChatModeState(mode);
    storeNewChatMode(mode);
    setChatContextBlocks([]);
  }

  const rememberPersonaVersions = useCallback((versions: PersonaVersion[]) => {
    if (versions.length === 0) {
      return;
    }

    setPersonaVersions((current) => {
      const next = { ...current };
      for (const version of versions) {
        next[version.id] = version;
      }
      return next;
    });
  }, []);

  const rememberPrivatePersonaVersions = useCallback((versions: PrivatePersonaVersion[]) => {
    if (versions.length === 0) {
      return;
    }

    setPrivatePersonaVersions((current) => {
      const next = { ...current };
      for (const version of versions) {
        next[version.id] = version;
      }
      return next;
    });
  }, []);

  const applyModelPickerData = useCallback(
    (modelsResponse: ModelsResponse, personasResponse: PersonasResponse) => {
      hasModelPickerDataRef.current = true;
      const enabledValues = enabledModelValueSet(modelsResponse.backends);
      const visiblePersonas = personasResponse.personas.filter((persona) =>
        enabledValues.has(personaBaseModelValue(persona))
      );
      setModelGroups(modelsResponse.backends);
      setPersonas(visiblePersonas);
      rememberPersonaVersions(visiblePersonas.map((persona) => persona.current_version));
      setSelectedModel((current) => {
        const values = [
          ...modelsResponse.backends.flatMap((group) =>
            group.models.map((model) => modelValue(group.backend.id, model.name))
          ),
          ...visiblePersonas.map((persona) => personaModelValue(persona.current_version.id))
        ];
        const defaultModel =
          modelsResponse.backends
            .flatMap((group) =>
              group.models.map((model) => ({
                backendId: group.backend.id,
                model
              }))
            )
            .find((option) => option.model.is_default) ?? null;
        const defaultValue = defaultModel
          ? modelValue(defaultModel.backendId, defaultModel.model.name)
          : "";

        return current &&
          (values.includes(current) || Boolean(privatePersonaVersionIdFromValue(current)))
          ? current
          : defaultValue;
      });
    },
    [rememberPersonaVersions]
  );

  const loadModels = useCallback(async () => {
    setIsLoadingModels(true);
    setModelError(null);

    if (!hasHydratedModelCacheRef.current) {
      hasHydratedModelCacheRef.current = true;
      try {
        const cached = await getCachedModelState<ModelPickerCache>();
        if (cached) {
          applyModelPickerData(cached.models, cached.personas);
          setIsLoadingModels(false);
        }
      } catch {
        // Model cache is an optimization. The live server fetch below remains authoritative.
      }
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
      if (!hasModelPickerDataRef.current) {
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

  useEffect(() => {
    void (async () => {
      try {
        const settings = await requestJson<UserSettings>("/api/user-settings");
        if (settings.theme) {
          storeAndApplyTheme(normalizeTheme(settings.theme));
        } else {
          applyTheme(storedTheme());
        }
      } catch {
        applyTheme(storedTheme());
      }
    })();
  }, [user.id]);

  const loadPrivatePersonas = useCallback(async () => {
    try {
      const nextPrivatePersonas = await listPrivatePersonas();
      setPrivatePersonas(nextPrivatePersonas);
      rememberPrivatePersonaVersions(
        nextPrivatePersonas.map((persona) => persona.current_version)
      );
    } catch (loadError) {
      setError(
        loadError instanceof Error
          ? loadError.message
          : "Failed to load private personas"
      );
    }
  }, [rememberPrivatePersonaVersions]);

  useEffect(() => {
    void loadPrivatePersonas();
  }, [loadPrivatePersonas]);

  const loadAvailableTools = useCallback(async () => {
    try {
      const response = await requestJson<AvailableToolsResponse>("/api/tools");
      setAvailableTools(response.tools_enabled ? response.tools : []);
    } catch {
      setAvailableTools([]);
    }
  }, []);

  useEffect(() => {
    void loadAvailableTools();
  }, [loadAvailableTools]);

  const loadContextLibraries = useCallback(async () => {
    const [serverResult, deviceResult] = await Promise.allSettled([
      requestJson<ContextLibraryResponse>("/api/context-library"),
      listPrivateContextLibrary()
    ]);

    if (serverResult.status === "fulfilled") {
      setServerContextLibrary(serverResult.value);
    }
    if (deviceResult.status === "fulfilled") {
      setDeviceContextLibrary(deviceResult.value);
    }
    if (serverResult.status === "rejected" && deviceResult.status === "rejected") {
      const reason = serverResult.reason;
      throw reason instanceof Error ? reason : new Error("Failed to load context libraries");
    }
  }, []);

  useEffect(() => {
    void loadContextLibraries().catch((loadError) => {
      setError(loadError instanceof Error ? loadError.message : "Failed to load context blocks");
    });
  }, [loadContextLibraries]);

  const loadChats = useCallback(async () => {
    setIsLoadingChats(true);
    let displayedCachedChats = false;

    if (!hasHydratedChatListCacheRef.current) {
      hasHydratedChatListCacheRef.current = true;
      try {
        const cachedChats = await getCachedHostedChatList<ChatSummary>();
        if (cachedChats) {
          chatsRef.current = cachedChats;
          setChats(cachedChats);
          displayedCachedChats = true;
          setIsLoadingChats(false);
          markPerformance("vashti:chat-list-cache-ready");
          measurePerformance(
            "vashti:startup-to-cached-chat-list",
            "vashti:app-start",
            "vashti:chat-list-cache-ready"
          );
        }
      } catch {
        // Chat summary caching is best-effort. The server remains authoritative.
      }
    }

    try {
      const response = await requestJson<ListChatsResponse>("/api/chats");
      chatsRef.current = response.chats;
      setChats(response.chats);
      markPerformance("vashti:chat-list-live-ready");
      measurePerformance(
        "vashti:startup-to-live-chat-list",
        "vashti:app-start",
        "vashti:chat-list-live-ready"
      );
      await saveCachedHostedChatList(response.chats).catch(() => undefined);
    } catch (loadError) {
      if (!displayedCachedChats) {
        setError(loadError instanceof Error ? loadError.message : "Failed to load chats");
      }
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

    setIsSidebarOpen(false);
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

  async function createChatFromPrompt(
    prompt: string,
    attachments: ComposerAttachment[] = [],
    toolPreferences: ChatToolPreferences = defaultToolPreferences,
    thinkMode: ThinkingMode = "auto"
  ) {
    if (!prompt.trim()) {
      openChat();
      return true;
    }

    const selected = selectedModelBaseParts(
      modelGroups,
      personas,
      [],
      selectedModel,
      knownPersonaVersions
    );
    if (!selected) {
      setError("Select a model before starting a chat");
      return false;
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
          persona_version_id: selectedPersonaVersionId,
          tool_preferences: toolPreferences,
          system_prompt_override: chatSystemPromptOverride,
          inference_settings: chatInferenceSettings,
          context_block_version_ids: chatContextBlocks.map(
            (selection) => selection.block_version_id
          )
        })
      });

      if (prompt.trim()) {
        const uploadedAttachments = await uploadComposerAttachments(response.chat.id, attachments);
        setQueuedPrompt({
          chatId: response.chat.id,
          prompt,
          attachments: uploadedAttachments,
          toolPreferences,
          thinkMode,
          inferenceSettings: chatInferenceSettings,
          contextBlocks: chatContextBlocks
        });
      }

      await loadChats();
      openChat(response.chat.id);
      return true;
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "Failed to create chat");
      return false;
    } finally {
      setIsCreatingChat(false);
    }
  }

  function currentSelectedModel() {
    const selectedPrivatePersonaVersion = privatePersonaVersionForValue(
      privatePersonas,
      knownPrivatePersonaVersions,
      selectedModel
    );
    if (selectedPrivatePersonaVersion) {
      return {
        backendId: selectedPrivatePersonaVersion.base_backend_id,
        backendName: selectedPrivatePersonaVersion.base_backend_name,
        modelName: selectedPrivatePersonaVersion.base_model_name
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
    return modelInfoForValue(
      modelGroups,
      personas,
      privatePersonas,
      selectedModel,
      knownPersonaVersions,
      knownPrivatePersonaVersions
    );
  }

  async function persistChatConversationSettings() {
    if (currentChatId) {
      const response = await requestJson<ChatResponse>(`/api/chats/${currentChatId}`, {
        method: "PATCH",
        body: JSON.stringify({
          system_prompt_override: chatSystemPromptOverride,
          inference_settings: chatInferenceSettings,
          context_block_version_ids: chatContextBlocks.map(
            (selection) => selection.block_version_id
          )
        })
      });
      setChatSystemPromptOverride(response.chat.system_prompt_override ?? null);
      setChatInferenceSettings(response.chat.inference_settings ?? {});
      setChatContextBlocks(response.chat.context_blocks ?? []);
      return;
    }

    if (currentPrivateChatId) {
      const chat = await getPrivateChat(currentPrivateChatId);
      if (!chat) {
        throw new Error("Private chat not found on this device");
      }

      const nextChat = {
        ...chat,
        system_prompt_override: chatSystemPromptOverride,
        inference_settings: chatInferenceSettings,
        context_blocks: chatContextBlocks.map((selection, position) => ({
          ...selection,
          position
        })),
        updated_at: unixTimestamp()
      };
      await savePrivateChat(nextChat);
      setChatSystemPromptOverride(chatSystemPromptOverride);
      setChatInferenceSettings(chatInferenceSettings);
      setChatContextBlocks(nextChat.context_blocks);
      await loadPrivateChats();
      return;
    }
  }

  const handleChatSettingsLoaded = useCallback((
    override: string | null | undefined,
    inferenceSettings?: ChatInferenceSettings,
    contextBlocks?: ContextBlockSelection[]
  ) => {
    setChatSystemPromptOverride(override ?? null);
    setChatInferenceSettings(inferenceSettings ?? {});
    setChatContextBlocks(contextBlocks ?? []);
  }, []);

  function createCustomModelFromSettings(draft: CustomModelDraft) {
    storeCustomModelDraft(draft);
    openSettings("models");
  }

  async function createPrivateChatFromPrompt(
    prompt: string,
    attachments: ComposerAttachment[] = [],
    _toolPreferences: ChatToolPreferences = defaultToolPreferences,
    thinkMode: ThinkingMode = "auto"
  ) {
    if (!prompt.trim()) {
      openChat();
      return true;
    }

    if (personaVersionIdFromValue(selectedModel)) {
      setError("Copy this hosted persona to your device before using it in a private chat");
      return false;
    }

    const selected = currentSelectedModel();
    const selectedPrivatePersona = privatePersonaWithVersionForValue(
      privatePersonas,
      knownPrivatePersonaVersions,
      selectedModel
    );
    if (!selected) {
      setError("Select a model before starting a private chat");
      return false;
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
        personaName: selectedPrivatePersona?.current_version.display_name ?? null,
        systemPromptOverride: chatSystemPromptOverride,
        inferenceSettings: chatInferenceSettings,
        contextBlocks: chatContextBlocks
      });

      if (prompt.trim()) {
        setQueuedPrivatePrompt({
          chatId: chat.id,
          prompt,
          attachments,
          thinkMode,
          systemPromptOverride: chatSystemPromptOverride,
          inferenceSettings: chatInferenceSettings,
          contextBlocks: chatContextBlocks
        });
      }

      await loadPrivateChats();
      openPrivateChat(chat.id);
      return true;
    } catch (createError) {
      setError(
        createError instanceof Error ? createError.message : "Failed to create private chat"
      );
      return false;
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
      const nextChats = chatsRef.current.map((chat) =>
        chat.id === chatId ? { ...chat, title: response.chat.title } : chat
      );
      chatsRef.current = nextChats;
      setChats(nextChats);
      await saveCachedHostedChatList(nextChats).catch(() => undefined);
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
      const nextChats = chatsRef.current.filter((item) => item.id !== chat.id);
      chatsRef.current = nextChats;
      setChats(nextChats);
      await Promise.all([
        saveCachedHostedChatList(nextChats),
        deleteCachedHostedChat(chat.id),
        deleteHostedPendingSend(chat.id)
      ]).catch(() => undefined);
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

  const shellClassName = [
    "app-shell",
    isSidebarOpen ? "sidebar-open" : "",
    isSettingsPage ? "settings-shell" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <main className={shellClassName}>
      {!isSettingsPage && (
        <>
          <Sidebar
            chats={chats}
            privateChats={privateChats}
            currentChatId={currentChatId}
            currentPrivateChatId={currentPrivateChatId}
            isOpen={isSidebarOpen}
            isLoading={isLoadingChats}
            isLoadingPrivateChats={isLoadingPrivateChats}
            personaVersions={knownPersonaVersions}
            privatePersonaVersions={knownPrivatePersonaVersions}
            modelGroups={modelGroups}
            onClose={() => setIsSidebarOpen(false)}
            onDeleteChat={setChatDeleteTarget}
            onDeletePrivateChat={setPrivateChatDeleteTarget}
            onNewChat={() => openChat()}
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
        </>
      )}
      <section className="main-pane">
        <header className="topbar">
          <div className="topbar-left">
            {isSettingsPage ? (
              <div className="settings-topbar-title">
                <SettingsIcon />
                <span>Settings</span>
              </div>
            ) : (
              <>
                <button
                  type="button"
                  className="icon-button mobile-only"
                  aria-label={isSidebarOpen ? "Close sidebar" : "Open sidebar"}
                  aria-expanded={isSidebarOpen}
                  onClick={() => {
                    setIsSettingsMenuOpen(false);
                    setIsSidebarOpen((open) => !open);
                  }}
                >
                  <Menu />
                </button>
                <ModelPicker
                  groups={modelGroups}
                  personas={allowPrivatePersonaSelection ? [] : personas}
                  privatePersonas={allowPrivatePersonaSelection ? privatePersonas : []}
                  personaVersions={knownPersonaVersions}
                  privatePersonaVersions={knownPrivatePersonaVersions}
                  isLoading={isLoadingModels}
                  error={modelError}
                  value={activeSelectedModel}
                  onChange={setSelectedModel}
                />
                <ModelSettingsMenu
                  groups={modelGroups}
                  personas={allowPrivatePersonaSelection ? [] : personas}
                  privatePersonas={allowPrivatePersonaSelection ? privatePersonas : []}
                  personaVersions={knownPersonaVersions}
                  privatePersonaVersions={knownPrivatePersonaVersions}
                  selectedModel={activeSelectedModel}
                  selectedModelInfo={activeSelectedModel ? selectedModelInfo() : null}
                  systemPromptOverride={chatSystemPromptOverride}
                  inferenceSettings={chatInferenceSettings}
                  contextLibrary={
                    allowPrivatePersonaSelection ? deviceContextLibrary : serverContextLibrary
                  }
                  contextBlocks={chatContextBlocks}
                  canSaveConversationSettings={page === "chat" || Boolean(currentPrivateChatId)}
                  disabled={!activeSelectedModel || isLoadingModels}
                  onModelSelected={setSelectedModel}
                  onCreateCustomModelFromSettings={createCustomModelFromSettings}
                  onPersonaVersionsLoaded={rememberPersonaVersions}
                  onPrivatePersonaVersionsLoaded={rememberPrivatePersonaVersions}
                  onSystemPromptOverrideChange={setChatSystemPromptOverride}
                  onInferenceSettingsChange={setChatInferenceSettings}
                  onContextBlocksChange={setChatContextBlocks}
                  onOpenContextSettings={() => openSettings("context")}
                />
              </>
            )}
          </div>
          <div className="topbar-right">
            {(currentChatId || currentPrivateChatId) && (
              <button
                type="button"
                className="icon-button"
                aria-label={isMessageTreeOpen ? "Close message tree" : "Explore message tree"}
                aria-pressed={isMessageTreeOpen}
                title={isMessageTreeOpen ? "Close message tree" : "Explore message tree"}
                onClick={() => {
                  setIsSidebarOpen(false);
                  setIsSettingsMenuOpen(false);
                  setIsMessageTreeOpen((open) => !open);
                }}
              >
                <GitFork />
              </button>
            )}
            <div className="settings-menu-wrap" ref={settingsMenuRef}>
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
                title={updateStatus?.available ? `${updateStatus.available.version} is available` : undefined}
                onClick={activateSettingsControl}
              >
                {isSettingsPage || isSettingsMenuOpen ? <X /> : <Cog />}
                {updateStatus?.available && !isSettingsPage && (
                  <span className="update-available-dot" aria-hidden="true" />
                )}
              </button>
              {isSettingsMenuOpen && !isSettingsPage && (
                <div className="settings-menu">
                  {isAdmin && updateStatus?.available && (
                    <button
                      type="button"
                      className="menu-item update-menu-item"
                      onClick={() => openSettings("app")}
                    >
                      <Download />
                      <span>Update to {updateStatus.available.version}</span>
                    </button>
                  )}
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
            onToolsChanged={loadAvailableTools}
            onPersonasChanged={loadModels}
            onPrivatePersonasChanged={loadPrivatePersonas}
            onContextChanged={loadContextLibraries}
            onAppSettingsGuardChange={updateAppSettingsGuard}
            updateStatus={updateStatus}
            updateStatusError={updateStatusError}
            onUpdateStatusChange={(status) => {
              setUpdateStatus(status);
              setUpdateStatusError(null);
            }}
            onRefreshUpdateStatus={refreshUpdateStatus}
            onSelectSection={(section) => openSettings(section)}
            onUserChanged={onUserChanged}
            isAdmin={isAdmin}
          />
        ) : page === "private-chat" && currentPrivateChatId ? (
          <Suspense fallback={<ChatInterfaceLoader />}>
            <PrivateChatView
              chatId={currentPrivateChatId}
              error={error}
              queuedPrompt={
                queuedPrivatePrompt?.chatId === currentPrivateChatId ? queuedPrivatePrompt : null
              }
              selectedModel={selectedModel}
              selectedModelInfo={selectedModelInfo()}
              modelGroups={modelGroups}
              privatePersonas={privatePersonas}
              privatePersonaVersions={knownPrivatePersonaVersions}
              systemPromptOverride={chatSystemPromptOverride}
              inferenceSettings={chatInferenceSettings}
              contextBlocks={chatContextBlocks}
              isTreeOpen={isMessageTreeOpen}
              onTreeClose={() => setIsMessageTreeOpen(false)}
              onImageOpen={openImageViewer}
              onChatSettingsLoaded={handleChatSettingsLoaded}
              onModelSelected={setSelectedModel}
              onConversationSettingsSave={persistChatConversationSettings}
              onPrivatePersonaVersionsLoaded={rememberPrivatePersonaVersions}
              onPrivateChatsChanged={loadPrivateChats}
              onQueuedPromptConsumed={() => setQueuedPrivatePrompt(null)}
            />
          </Suspense>
        ) : (
          currentChatId ? (
            <Suspense fallback={<ChatInterfaceLoader />}>
              <ChatView
                chatId={currentChatId}
                error={error}
                queuedPrompt={queuedPrompt?.chatId === currentChatId ? queuedPrompt : null}
                selectedModel={selectedModel}
                selectedModelInfo={selectedModelInfo()}
                modelGroups={modelGroups}
                inferenceSettings={chatInferenceSettings}
                availableTools={availableTools}
                personas={personas}
                personaVersions={knownPersonaVersions}
                isTreeOpen={isMessageTreeOpen}
                onTreeClose={() => setIsMessageTreeOpen(false)}
                onChatSettingsLoaded={handleChatSettingsLoaded}
                onChatsChanged={loadChats}
                onConversationSettingsSave={persistChatConversationSettings}
                onPersonaVersionsLoaded={rememberPersonaVersions}
                onImageOpen={openImageViewer}
                onModelSelected={setSelectedModel}
                onQueuedPromptConsumed={() => setQueuedPrompt(null)}
              />
            </Suspense>
          ) : (
            <ChatHome
              error={error}
              isCreating={isCreatingChat}
              isCreatingPrivate={isCreatingPrivateChat}
              mode={newChatMode}
              selectedModel={
                activeSelectedModel
              }
              selectedModelInfo={activeSelectedModel ? selectedModelInfo() : null}
              availableTools={newChatMode === "standard" ? availableTools : []}
              onModeChange={setNewChatMode}
              onCreateChat={createChatFromPrompt}
              onCreatePrivateChat={createPrivateChatFromPrompt}
            />
          )
        )}
      </section>
      {activeUpdateVersion && (
        <aside className="server-update-notice" role="status" aria-live="polite">
          <RefreshCw aria-hidden="true" />
          <div>
            <strong>Installing {activeUpdateVersion}</strong>
            <span>Vashti may disconnect briefly, then this page will reload.</span>
          </div>
        </aside>
      )}
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

function ChatInterfaceLoader() {
  return (
    <section className="chat-view" role="status" aria-label="Loading chat interface">
      <div className="empty-state">
        <RetroLoader />
      </div>
    </section>
  );
}
