import {
  FormEvent,
  useCallback,
  useEffect,
  useState
} from "react";
import { Plus, Trash2, X } from "lucide-react";
import { requestJson } from "./api";
import { ConfirmDialog, RetroLoader } from "./common";
import { PermissionTagEditor } from "./permissionTags";
import { SettingsPanel } from "./settingsControls";
import { permissionTagPayload } from "./settingsModelHelpers";
import type {
  AdminUser,
  AdminUserMutationResponse,
  AdminUsersResponse,
  PermissionTag
} from "./types";

export function AdminUsersPanel({ currentUserId }: { currentUserId: string }) {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [availableTags, setAvailableTags] = useState<PermissionTag[]>([]);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isCreatingUser, setIsCreatingUser] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busyUserId, setBusyUserId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<AdminUser | null>(null);
  const [showCreateUser, setShowCreateUser] = useState(false);
  const [newUsername, setNewUsername] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newRole, setNewRole] = useState("user");
  const [newDisabled, setNewDisabled] = useState(false);

  function resetCreateUserDraft() {
    setNewUsername("");
    setNewEmail("");
    setNewPassword("");
    setNewRole("user");
    setNewDisabled(false);
  }

  function cancelCreateUser() {
    resetCreateUserDraft();
    setShowCreateUser(false);
    setError(null);
  }

  const loadUsers = useCallback(async () => {
    setIsRefreshing(true);
    setError(null);

    try {
      const response = await requestJson<AdminUsersResponse>("/api/admin/users");
      setUsers(response.users);
      setAvailableTags(response.available_tags);
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
      resetCreateUserDraft();
      setShowCreateUser(false);
      await loadUsers();
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "Failed to create user");
    } finally {
      setIsCreatingUser(false);
    }
  }

  async function patchUser(
    userId: string,
    body: Partial<Pick<AdminUser, "role" | "is_disabled">> & { permission_tags?: string[] }
  ) {
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
  const currentUser = users.find((listedUser) => listedUser.id === currentUserId) ?? null;
  const approvedUsers = users.filter(
    (listedUser) => !listedUser.is_disabled && listedUser.id !== currentUserId
  );

  return (
    <SettingsPanel
      eyebrow="Admin"
      title="Users"
      width="wide"
      actions={
        <div className="header-actions">
          <button
            type="button"
            className="secondary-button refresh-button"
            onClick={() => void loadUsers()}
            disabled={isRefreshing}
          >
            {isRefreshing ? <RetroLoader /> : "Refresh"}
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() => setShowCreateUser(true)}
            disabled={showCreateUser}
          >
            <Plus />
            <span>Create User</span>
          </button>
        </div>
      }
    >

      {!hasLoaded && <p className="status-message">Loading users...</p>}
      {error && <p className="error">{error}</p>}

      {showCreateUser && (
        <form className="settings-form settings-create-card user-create-form" onSubmit={createUser}>
          <div className="settings-create-card-header">
            <p className="eyebrow">Manual Creation</p>
            <h2>Create User</h2>
            <p>
              Creates an account immediately. Invite links can replace this flow later so admins
              never handle user passwords.
            </p>
          </div>
          <div className="settings-create-card-grid">
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
          </div>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={newDisabled}
              onChange={(event) => setNewDisabled(event.target.checked)}
            />
            <span>Create disabled</span>
          </label>
          <div className="settings-create-card-actions">
            <button
              type="button"
              className="secondary-button"
              onClick={cancelCreateUser}
              disabled={isCreatingUser}
            >
              <X />
              <span>Cancel</span>
            </button>
            <button
              type="submit"
              disabled={isCreatingUser || newUsername.trim() === "" || newPassword === ""}
            >
              <Plus />
              <span>{isCreatingUser ? "Creating..." : "Create User"}</span>
            </button>
          </div>
        </form>
      )}

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
          availableTags={availableTags}
        />
        <div className="user-divider" />
        {currentUser && (
          <>
            <section className="user-group user-self-group">
              <div className="user-group-header">
                <h2>Signed In</h2>
              </div>
              <UserRow
                user={currentUser}
                isSelf
                isBusy={busyUserId === currentUser.id}
                onPatchUser={patchUser}
                onDeleteUser={setDeleteTarget}
                availableTags={availableTags}
              />
            </section>
            <div className="user-divider" />
          </>
        )}
        <UserGroup
          title="Approved Users"
          users={approvedUsers}
          emptyText="No approved users."
          currentUserId={currentUserId}
          busyUserId={busyUserId}
          onPatchUser={patchUser}
          onDeleteUser={setDeleteTarget}
          availableTags={availableTags}
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
    </SettingsPanel>
  );
}

function UserGroup({
  title,
  users,
  emptyText,
  currentUserId,
  busyUserId,
  onPatchUser,
  onDeleteUser,
  availableTags
}: {
  title: string;
  users: AdminUser[];
  emptyText: string;
  currentUserId: string;
  busyUserId: string | null;
  onPatchUser: (
    userId: string,
    body: Partial<Pick<AdminUser, "role" | "is_disabled">> & { permission_tags?: string[] }
  ) => Promise<void>;
  onDeleteUser: (user: AdminUser) => void;
  availableTags: PermissionTag[];
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
            availableTags={availableTags}
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
  onDeleteUser,
  availableTags
}: {
  user: AdminUser;
  isSelf: boolean;
  isBusy: boolean;
  onPatchUser: (
    userId: string,
    body: Partial<Pick<AdminUser, "role" | "is_disabled">> & { permission_tags?: string[] }
  ) => Promise<void>;
  onDeleteUser: (user: AdminUser) => void;
  availableTags: PermissionTag[];
}) {
  return (
    <article className={isSelf ? "user-row user-row-self" : "user-row"}>
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
          {isSelf && <span className="badge badge-self">you</span>}
        </div>
      </div>
      <PermissionTagEditor
        label="Group tags"
        tags={user.permission_tags}
        availableTags={availableTags}
        suggestionsKind="group"
        disabled={isBusy}
        onChange={(tags) =>
          void onPatchUser(user.id, { permission_tags: permissionTagPayload(tags) })
        }
      />
      {!isSelf && (
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
              disabled={isBusy}
              onClick={() => void onPatchUser(user.id, { is_disabled: true })}
            >
              Disable
            </button>
          )}
          {user.role === "admin" ? (
            <button
              type="button"
              className="secondary-button"
              disabled={isBusy}
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
            disabled={isBusy}
            onClick={() => onDeleteUser(user)}
          >
            <Trash2 />
            <span>Delete</span>
          </button>
        </div>
      )}
    </article>
  );
}
