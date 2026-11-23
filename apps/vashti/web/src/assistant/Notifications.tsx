import { useEffect, useState } from "react";
import { ArrowLeft, BrainCircuit, Check } from "lucide-react";
import { requestJson } from "../api";
import { profileNotices, readNotice } from "../personalization/device";
import type { ProfileMode, ProfileNotice } from "../personalization/types";

type Notice = ProfileNotice & { mode: ProfileMode };

export function Notifications({ onBack }: { onBack: () => void }) {
  const [rows, setRows] = useState<Notice[] | null>(null);
  const [error, setError] = useState("");
  const [reading, setReading] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    let epoch = 0;
    async function load() {
      const attempt = ++epoch;
      const [server, device] = await Promise.allSettled([
        requestJson<{ notices: ProfileNotice[] }>("/api/assistant/notices"),
        profileNotices()
      ]);
      if (!active || epoch !== attempt) return;
      const next: Notice[] = [
        ...(server.status === "fulfilled" ? server.value.notices.map((item) => ({ ...item, mode: "server" as const })) : []),
        ...(device.status === "fulfilled" ? device.value.map((item) => ({ ...item, mode: "device" as const })) : [])
      ];
      setRows(next.sort((a, b) => b.created_at - a.created_at || b.id.localeCompare(a.id)));
      setError(server.status === "rejected" || device.status === "rejected" ? "Some notifications couldn’t be loaded." : "");
    }
    void load();
    const timer = window.setInterval(load, 10000);
    window.addEventListener("vashti:personalization-changed", load);
    window.addEventListener("vashti:assistant-notices", load);
    return () => {
      active = false;
      window.clearInterval(timer);
      window.removeEventListener("vashti:personalization-changed", load);
      window.removeEventListener("vashti:assistant-notices", load);
    };
  }, []);
  async function markRead(notice: Notice) {
    setReading(notice.id);
    setError("");
    try {
      if (notice.mode === "device") await readNotice(notice.id);
      else await requestJson(`/api/assistant/notices/${encodeURIComponent(notice.id)}/read`, { method: "POST" });
      setRows((current) => current?.map((item) => item.mode === notice.mode && item.id === notice.id
        ? { ...item, read_at: Math.floor(Date.now() / 1000) } : item) ?? null);
      window.dispatchEvent(new Event("vashti:assistant-notices"));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not mark the notification as read.");
    } finally {
      setReading(null);
    }
  }
  return <section className="assistant-workspace assistant-scroll"><div className="assistant-inbox">
    <div className="assistant-page-heading">
      <button type="button" className="icon-button" title="Back to Assistant" aria-label="Back to Assistant" onClick={onBack}><ArrowLeft /></button>
      <h1>Notifications</h1>
    </div>
    {error && <p className="error-message" role="alert">{error}</p>}
    {!rows ? <p role="status">Loading notifications...</p> : rows.length === 0 ? <p className="assistant-empty">No notifications yet</p> :
      <ul className="assistant-task-list assistant-notice-list">{rows.map((notice) => <li key={`${notice.mode}:${notice.id}`} className="assistant-notice-row">
        <BrainCircuit aria-hidden="true" />
        <div className="assistant-notice-text">
          <strong>{notice.mode === "server" ? "Server profile rebuilt" : "Device profile rebuilt"}</strong>
          <span className="assistant-caption">{new Date(notice.created_at * 1000).toLocaleString()}</span>
        </div>
        {notice.read_at === null && <button type="button" className="icon-button" disabled={reading !== null} title="Mark as read" aria-label="Mark as read" onClick={() => void markRead(notice)}><Check /></button>}
      </li>)}</ul>}
  </div></section>;
}
