import { useEffect, useRef, useState } from "react";
import { RotateCcw, Square } from "lucide-react";
import { ConfirmDialog } from "../common";
import { cancelRebuild, getProfile, getRebuildProgress, rebuildProfile } from "./api";
import type { PersonalProfile, ProfileMode, RebuildProgress } from "./types";

const phaseLabels: Record<RebuildProgress["phase"], string> = {
  queued: "Queued",
  replaying: "Replaying history",
  waiting_for_idle: "Waiting for idle time",
  waiting_for_generation: "Waiting for generation to finish",
  paused: "Paused. Check personalization settings",
  needs_attention: "Processing needs attention. Check Memories"
};

export function RebuildProfileButton({ mode, profile, disabled, onChanged }: {
  mode: ProfileMode;
  profile?: PersonalProfile;
  disabled?: boolean;
  onChanged?: (value: PersonalProfile) => void;
}) {
  const [confirm, setConfirm] = useState<"start" | { cancelId: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [progress, setProgress] = useState<RebuildProgress | null>();
  const request = useRef(0);
  const rebuildId = progress === undefined ? profile?.rebuild_id : progress?.id;

  useEffect(() => {
    if (busy) return;
    let stopped = false;
    let timer = 0;
    const poll = async () => {
      const id = ++request.current;
      let active = true;
      try {
        const value = await getRebuildProgress(mode);
        active = value !== null;
        if (!stopped && id === request.current) setProgress(value);
      } catch {
        // Keep the last confirmed status when temporarily offline.
      } finally {
        if (!stopped && id === request.current)
          timer = window.setTimeout(poll, active && !document.hidden ? 3000 : 15000);
      }
    };
    void poll();
    return () => { stopped = true; window.clearTimeout(timer); };
  }, [busy, mode]);

  async function submit() {
    if (!confirm) return;
    request.current++;
    setBusy(true);
    setError("");
    setNotice("");
    const canceling = confirm !== "start";
    try {
      const value = canceling
        ? await cancelRebuild(mode, confirm.cancelId)
        : await rebuildProfile(mode, profile ?? await getProfile(mode));
      setProgress(value.rebuild_id ? { id: value.rebuild_id, processed: 0, phase: "queued" } : null);
      if (canceling) setNotice("Rebuild canceled. Current profile kept.");
      onChanged?.(value);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : canceling ? "Could not cancel the rebuild." : "Could not queue the profile rebuild.");
    } finally {
      setConfirm(null);
      setBusy(false);
    }
  }

  return <div className="profile-rebuild-action">
    {rebuildId ? <>
      <button type="button" className="secondary-button" disabled={busy} onClick={() => { setError(""); setConfirm({ cancelId: rebuildId }); }}>
        <Square aria-hidden="true" />Cancel rebuild
      </button>
      <small role="status" className="muted">{progress
        ? `${phaseLabels[progress.phase]} · ${progress.processed.toLocaleString()} ${progress.processed === 1 ? "message" : "messages"} processed`
        : "Checking rebuild progress..."}</small>
    </> : <button type="button" className="secondary-button" disabled={disabled || busy} onClick={() => { setError(""); setConfirm("start"); }}>
      <RotateCcw aria-hidden="true" />Rebuild profile
    </button>}
    {notice && !rebuildId && <small role="status" className="muted">{notice}</small>}
    {error && <small role="alert" className="error">{error}</small>}
    {confirm && <ConfirmDialog
      title={confirm !== "start" ? "Cancel profile rebuild?" : mode === "server" ? "Rebuild server profile?" : "Rebuild this device’s profile?"}
      message={confirm !== "start" ? "The rebuild draft will be discarded. Your current profile and its history won’t change."
        : `Vashti replays eligible history from oldest to newest. Your current profile stays in use until the rebuild finishes and replaces its automatically learned facts and history. Manual facts and forgotten topics are kept. This can take a long time. ${mode === "device" ? "Keep Vashti open to let it run. " : "It runs in the background. "}You’ll see a notice in Assistant when it finishes.`}
      confirmLabel={confirm !== "start" ? "Discard rebuild" : "Rebuild profile"} isBusy={busy} onCancel={() => setConfirm(null)} onConfirm={() => void submit()}
    />}
  </div>;
}
