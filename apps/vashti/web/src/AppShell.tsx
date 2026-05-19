import {
  useCallback,
  useEffect,
  useRef,
  useState
} from "react";
import {
  Cog,
  LogOut,
  Menu,
  MessageSquarePlus,
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
import { ConfirmDialog } from "./common";
import { ModelPicker } from "./ModelPicker";
import {
  enabledModelValueSet,
  modelInfoForValue,
  modelParts,
  modelValue,
  personaBaseModelValue,
  personaModelValue,
  personaVersionIdFromValue,
  privatePersonaForValue,
  privatePersonaVersionIdFromValue,
  selectedModelBaseParts
} from "./modelSelection";
import { Sidebar } from "./Sidebar";
import { UnsavedSettingsDialog } from "./settingsControls";
import { SettingsPage } from "./settingsPage";
import { ChatView } from "./ChatView";
import { PrivateChatView } from "./PrivateChatView";
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
import {
  createPrivateChat,
  deletePrivateChat,
  getCachedModelState,
  listPrivatePersonas,
  listPrivateChats,
  renamePrivateChat,
  resetPrivateStorageUser,
  saveCachedModelState,
  setPrivateStorageUser,
  type PrivateChatSummary,
  type PrivatePersona
} from "./privateChatStore";
import type {
  AppRoute,
  AppSettingsGuard,
  AttachmentInfo,
  AvailableTool,
  AvailableToolsResponse,
  BackendModelGroup,
  ChatResponse,
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
  PersonasResponse,
  SettingsSection,
  ThinkingMode,
  User,
  UserSettings
} from "./types";

export function AppShell({
  user,
  onSessionChanged,
  onUserChanged
}: {
  user: User;
  onSessionChanged: () => Promise<void>;
  onUserChanged: (user: User) => void;
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
  const [availableTools, setAvailableTools] = useState<AvailableTool[]>([]);
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
    storeNewChatMode(mode);
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

  async function createChatFromPrompt(
    prompt: string,
    attachments: ComposerAttachment[] = [],
    toolPreferences: ChatToolPreferences = defaultToolPreferences,
    thinkMode: ThinkingMode = "auto"
  ) {
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
          persona_version_id: selectedPersonaVersionId,
          tool_preferences: toolPreferences
        })
      });

      if (prompt.trim()) {
        const uploadedAttachments = await uploadComposerAttachments(response.chat.id, attachments);
        setQueuedPrompt({
          chatId: response.chat.id,
          prompt,
          attachments: uploadedAttachments,
          toolPreferences,
          thinkMode
        });
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

  async function createPrivateChatFromPrompt(
    prompt: string,
    attachments: ComposerAttachment[] = [],
    _toolPreferences: ChatToolPreferences = defaultToolPreferences,
    thinkMode: ThinkingMode = "auto"
  ) {
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
        setQueuedPrivatePrompt({ chatId: chat.id, prompt, attachments, thinkMode });
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
              </>
            )}
          </div>
          <div className="topbar-right">
            {!isSettingsPage && (
              <button
                type="button"
                className="primary-action"
                onClick={() => openChat()}
              >
                <MessageSquarePlus />
                <span>New Chat</span>
              </button>
            )}
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
            onToolsChanged={loadAvailableTools}
            onPersonasChanged={loadModels}
            onPrivatePersonasChanged={loadPrivatePersonas}
            onAppSettingsGuardChange={updateAppSettingsGuard}
            onSelectSection={(section) => openSettings(section)}
            onUserChanged={onUserChanged}
            isAdmin={isAdmin}
          />
        ) : page === "private-chat" && currentPrivateChatId ? (
          <PrivateChatView
            chatId={currentPrivateChatId}
            error={error}
            queuedPrompt={
              queuedPrivatePrompt?.chatId === currentPrivateChatId ? queuedPrivatePrompt : null
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
              queuedPrompt={queuedPrompt?.chatId === currentChatId ? queuedPrompt : null}
              selectedModel={selectedModel}
              selectedModelInfo={selectedModelInfo()}
              availableTools={availableTools}
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
              availableTools={newChatMode === "standard" ? availableTools : []}
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
