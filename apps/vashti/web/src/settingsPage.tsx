import { Fragment, lazy, ReactNode, Suspense } from "react";
import {
  Server,
  Library,
  SlidersHorizontal,
  Sparkles,
  UserRound,
  Users,
  Wrench
} from "lucide-react";
import { RetroLoader } from "./common";
import type { AppSettingsGuard, SettingsSection, User } from "./types";

const AppSettingsPanel = lazy(() =>
  import("./settingsApp").then((module) => ({ default: module.AppSettingsPanel }))
);
const BackendsPanel = lazy(() =>
  import("./settingsBackends").then((module) => ({ default: module.BackendsPanel }))
);
const UserModelsPanel = lazy(() =>
  import("./settingsModels").then((module) => ({ default: module.UserModelsPanel }))
);
const ProfileSettings = lazy(() =>
  import("./settingsProfile").then((module) => ({ default: module.ProfileSettings }))
);
const ToolsSettingsPanel = lazy(() =>
  import("./settingsTools").then((module) => ({ default: module.ToolsSettingsPanel }))
);
const AdminUsersPanel = lazy(() =>
  import("./settingsUsers").then((module) => ({ default: module.AdminUsersPanel }))
);
const ContextSettingsPanel = lazy(() =>
  import("./settingsContext").then((module) => ({ default: module.ContextSettingsPanel }))
);

export function SettingsPage({
  currentUser,
  activeSection,
  onBackendsChanged,
  onToolsChanged,
  onPersonasChanged,
  onPrivatePersonasChanged,
  onContextChanged,
  onAppSettingsGuardChange,
  onSelectSection,
  onUserChanged,
  isAdmin
}: {
  currentUser: User;
  activeSection: SettingsSection;
  onBackendsChanged: () => Promise<void>;
  onToolsChanged: () => Promise<void>;
  onPersonasChanged: () => Promise<void>;
  onPrivatePersonasChanged: () => Promise<void>;
  onContextChanged: () => Promise<void>;
  onAppSettingsGuardChange: (guard: AppSettingsGuard | null) => void;
  onSelectSection: (section: SettingsSection) => void;
  onUserChanged: (user: User) => void;
  isAdmin: boolean;
}) {
  const sections: Array<{
    id: SettingsSection;
    label: string;
    icon: ReactNode;
    adminOnly?: boolean;
    group: "personal" | "admin";
  }> = [
    { id: "profile", label: "Profile", icon: <UserRound />, group: "personal" },
    { id: "models", label: "Models", icon: <Sparkles />, group: "personal" },
    { id: "context", label: "Context", icon: <Library />, group: "personal" },
    { id: "users", label: "Users", icon: <Users />, adminOnly: true, group: "admin" },
    { id: "backends", label: "Backends", icon: <Server />, adminOnly: true, group: "admin" },
    { id: "tools", label: "Tools", icon: <Wrench />, adminOnly: true, group: "admin" },
    { id: "app", label: "App", icon: <SlidersHorizontal />, adminOnly: true, group: "admin" }
  ];
  const visibleSections = sections.filter((section) => !section.adminOnly || isAdmin);
  const selectedSection =
    visibleSections.find((section) => section.id === activeSection)?.id ?? "profile";

  return (
    <div className="settings-page">
      <nav className="settings-nav" aria-label="Settings sections">
        {visibleSections.map((section, index) => (
          <Fragment key={section.id}>
            {section.group === "admin" && visibleSections[index - 1]?.group !== "admin" && (
              <div className="settings-nav-divider" />
            )}
            <button
              type="button"
              className={
                selectedSection === section.id
                  ? "settings-tab settings-tab-active"
                  : "settings-tab"
              }
              onClick={() => onSelectSection(section.id)}
            >
              {section.icon}
              <span>{section.label}</span>
            </button>
          </Fragment>
        ))}
      </nav>
      <section className="settings-content">
        <Suspense
          fallback={
            <div className="settings-section" role="status" aria-label="Loading settings">
              <RetroLoader />
            </div>
          }
        >
          {selectedSection === "profile" && (
            <ProfileSettings user={currentUser} onUserChanged={onUserChanged} />
          )}
          {selectedSection === "users" && isAdmin && (
            <AdminUsersPanel currentUserId={currentUser.id} />
          )}
          {selectedSection === "backends" && isAdmin && (
            <BackendsPanel onBackendsChanged={onBackendsChanged} />
          )}
          {selectedSection === "models" && (
            <UserModelsPanel
              onModelsChanged={onBackendsChanged}
              onPersonasChanged={onPersonasChanged}
              onPrivatePersonasChanged={onPrivatePersonasChanged}
            />
          )}
          {selectedSection === "context" && (
            <ContextSettingsPanel onContextChanged={onContextChanged} />
          )}
          {selectedSection === "tools" && isAdmin && (
            <ToolsSettingsPanel onToolsChanged={onToolsChanged} />
          )}
          {selectedSection === "app" && (
            <AppSettingsPanel onGuardChange={onAppSettingsGuardChange} />
          )}
        </Suspense>
      </section>
    </div>
  );
}
