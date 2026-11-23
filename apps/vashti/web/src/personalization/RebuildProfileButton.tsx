import { useState } from "react";
import { RotateCcw } from "lucide-react";
import { ConfirmDialog } from "../common";
import { getProfile, rebuildProfile } from "./api";
import type { PersonalProfile, ProfileMode } from "./types";

export function RebuildProfileButton({ mode, profile, disabled, onQueued }: {
  mode: ProfileMode;
  profile?: PersonalProfile;
  disabled?: boolean;
  onQueued?: (value: PersonalProfile) => void;
}) {
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [queued, setQueued] = useState(false);
  const rebuilding = profile ? Boolean(profile.rebuild_id) : queued;
  async function start() {
    setBusy(true);
    setError("");
    try {
      const current = profile ?? await getProfile(mode);
      const value = await rebuildProfile(mode, current);
      onQueued?.(value);
      setQueued(true);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not queue the profile rebuild.");
    } finally {
      setConfirm(false);
      setBusy(false);
    }
  }
  return <div className="profile-rebuild-action">
    <button type="button" className="secondary-button" disabled={disabled || busy || Boolean(profile?.rebuild_id)} onClick={() => { setError(""); setConfirm(true); }}>
      <RotateCcw aria-hidden="true" />Rebuild profile
    </button>
    {rebuilding && <small role="status" className="muted">{profile ? "Rebuild queued or running." : "Rebuild requested."} Completion will appear in Assistant.</small>}
    {error && <small role="alert" className="error">{error}</small>}
    {confirm && <ConfirmDialog
      title={mode === "server" ? "Rebuild server profile?" : "Rebuild this device’s profile?"}
      message={`Vashti replays eligible history from oldest to newest. Your current profile stays in use until the rebuild finishes and replaces its automatically learned facts and history. Manual facts and forgotten topics are kept. This can take a long time. ${mode === "device" ? "Keep Vashti open to let it run. " : "It runs in the background. "}You’ll see a notice in Assistant when it finishes.`}
      confirmLabel="Rebuild profile" isBusy={busy} onCancel={() => setConfirm(false)} onConfirm={() => void start()}
    />}
  </div>;
}
