import { Fragment, ReactNode } from "react";
import {
  Bot,
  Server,
  SlidersHorizontal,
  Sparkles,
  UserRound,
  Users,
  Wrench
} from "lucide-react";
import { SettingsPlaceholder } from "./settingsControls";
import { AppSettingsPanel } from "./settingsApp";
import { BackendsPanel } from "./settingsBackends";
import { UserModelsPanel } from "./settingsModels";
import { PersonasPanel } from "./settingsPersonas";
import { ToolsSettingsPanel } from "./settingsTools";
import { AdminUsersPanel } from "./settingsUsers";
import type { AppSettingsGuard, SettingsSection, User } from "./types";

export function SettingsPage({
  currentUser,
  activeSection,
  onBackendsChanged,
  onToolsChanged,
  onPersonasChanged,
  onPrivatePersonasChanged,
  onAppSettingsGuardChange,
  onSelectSection,
  isAdmin
}: {
  currentUser: User;
  activeSection: SettingsSection;
  onBackendsChanged: () => Promise<void>;
  onToolsChanged: () => Promise<void>;
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
    group: "personal" | "admin";
  }> = [
    { id: "profile", label: "Profile", icon: <UserRound />, group: "personal" },
    { id: "personas", label: "Personas", icon: <Bot />, group: "personal" },
    { id: "models", label: "Models", icon: <Sparkles />, group: "personal" },
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
        {selectedSection === "models" && <UserModelsPanel onModelsChanged={onBackendsChanged} />}
        {selectedSection === "tools" && isAdmin && (
          <ToolsSettingsPanel onToolsChanged={onToolsChanged} />
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
