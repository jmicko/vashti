import { FormEvent, ReactNode, useCallback, useEffect, useRef, useState } from "react";
import {
  Cog,
  LogOut,
  Menu,
  MessageSquare,
  MessageSquarePlus,
  Server,
  Save,
  Settings as SettingsIcon,
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
type AppRoute = { page: "chat" } | { page: "settings"; section: SettingsSection };
type AppSettingsGuard = {
  isDirty: boolean;
  save: () => Promise<boolean>;
  discard: () => void;
};

const settingsSections: SettingsSection[] = ["profile", "users", "backends", "app"];

function isSettingsSection(value: string | undefined): value is SettingsSection {
  return settingsSections.includes(value as SettingsSection);
}

function routeFromLocation(): AppRoute {
  const path = window.location.pathname;

  if (path.startsWith("/app/settings")) {
    const section = path.split("/")[3];
    return { page: "settings", section: isSettingsSection(section) ? section : "profile" };
  }

  return { page: "chat" };
}

function pathForRoute(route: AppRoute) {
  if (route.page === "settings") {
    return `/app/settings/${route.section}`;
  }

  return "/app";
}

function routesEqual(left: AppRoute, right: AppRoute) {
  return pathForRoute(left) === pathForRoute(right);
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
  const [error, setError] = useState<string | null>(null);
  const isAdmin = user.role === "admin";
  const page = route.page;
  const isSettingsPage = page === "settings";
  const settingsSection = route.page === "settings" ? route.section : "profile";

  useEffect(() => {
    routeRef.current = route;
  }, [route]);

  const updateAppSettingsGuard = useCallback((guard: AppSettingsGuard | null) => {
    appSettingsGuardRef.current = guard;
  }, []);

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

  function openChat() {
    navigate({ page: "chat" });
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

  return (
    <main className={isSidebarOpen ? "app-shell sidebar-open" : "app-shell"}>
      <Sidebar currentPage={page} onClose={() => setIsSidebarOpen(false)} onOpenChat={openChat} />
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
            <select className="model-picker" aria-label="Model" disabled>
              <option>Model</option>
            </select>
          </div>
          <div className="topbar-right">
            <button type="button" className="primary-action" disabled>
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
            onAppSettingsGuardChange={updateAppSettingsGuard}
            onSelectSection={(section) => openSettings(section)}
            isAdmin={isAdmin}
          />
        ) : (
          <ChatHome error={error} />
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
    </main>
  );
}

function Sidebar({
  currentPage,
  onClose,
  onOpenChat
}: {
  currentPage: Page;
  onClose: () => void;
  onOpenChat: () => void;
}) {
  return (
    <aside className="sidebar">
      <div>
        <div className="sidebar-header">
          <BrandMark compact />
          <button type="button" className="icon-button mobile-only" aria-label="Close sidebar" onClick={onClose}>
            <X />
          </button>
        </div>
        <button
          type="button"
          className={currentPage === "chat" ? "nav-button nav-button-active" : "nav-button"}
          onClick={onOpenChat}
        >
          <MessageSquare />
          <span>Chats</span>
        </button>
        <div className="chat-history">
          <p className="eyebrow">Previous Chats</p>
          <p>No chats yet</p>
        </div>
      </div>
    </aside>
  );
}

function ChatHome({ error }: { error: string | null }) {
  return (
    <div className="empty-state">
      <p>No chat selected</p>
      {error && <p className="error">{error}</p>}
    </div>
  );
}

function SettingsPage({
  currentUser,
  activeSection,
  onAppSettingsGuardChange,
  onSelectSection,
  isAdmin
}: {
  currentUser: User;
  activeSection: SettingsSection;
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
        {selectedSection === "backends" && (
          <SettingsPlaceholder
            eyebrow="Admin"
            title="Backends"
            text="Ollama backend management starts in the next slice."
          />
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
