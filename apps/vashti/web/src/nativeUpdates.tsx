import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { Download, RefreshCw } from "lucide-react";
import { invokeNative, isNativeRuntime } from "./runtime";

type Channel = "stable" | "prerelease";
type AppUpdate = { version: string };
const CHANNEL_KEY = "vashti:android-update-channel";
const CHECK_INTERVAL = 6 * 60 * 60 * 1000;
const RETRY_INTERVAL = 15 * 60 * 1000;

function storedChannel(): Channel {
  try {
    return localStorage.getItem(CHANNEL_KEY) === "prerelease" ? "prerelease" : "stable";
  } catch {
    return "stable";
  }
}

function useUpdateController() {
  const native = isNativeRuntime();
  const [channel, setChannelState] = useState<Channel>(storedChannel);
  const [version, setVersion] = useState<string | null>(null);
  const [available, setAvailable] = useState<AppUpdate | null>(null);
  const [checking, setChecking] = useState(false);
  const [opening, setOpening] = useState(false);
  const [checkedAt, setCheckedAt] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sequence = useRef(0);
  const busy = useRef(false);
  const nextCheck = useRef(0);

  const check = useCallback(async () => {
    if (!native || busy.current) return;
    const request = ++sequence.current;
    busy.current = true;
    setChecking(true);
    setCheckedAt(null);
    setError(null);
    try {
      const update = await invokeNative<AppUpdate | null>("native_check_app_update", { channel });
      if (request !== sequence.current) return;
      setAvailable(update);
      setCheckedAt(Date.now());
      nextCheck.current = Date.now() + CHECK_INTERVAL;
    } catch (reason) {
      if (request !== sequence.current) return;
      setError(reason instanceof Error ? reason.message : String(reason));
      nextCheck.current = Date.now() + RETRY_INTERVAL;
    } finally {
      if (request === sequence.current) {
        busy.current = false;
        setChecking(false);
      }
    }
  }, [channel, native]);

  useEffect(() => {
    if (!native) return;
    let active = true;
    void import("@tauri-apps/api/app").then(({ getVersion }) => getVersion()).then((value) => {
      if (active) setVersion(value);
    }).catch(() => {});
    return () => { active = false; };
  }, [native]);

  useEffect(() => {
    if (!native) return;
    setAvailable(null);
    setCheckedAt(null);
    setChecking(false);
    setError(null);
    nextCheck.current = 0;
    const checkWhenVisible = () => {
      if (document.visibilityState === "visible" && Date.now() >= nextCheck.current) void check();
    };
    const initial = window.setTimeout(checkWhenVisible, 5000);
    const interval = window.setInterval(checkWhenVisible, 60_000);
    document.addEventListener("visibilitychange", checkWhenVisible);
    window.addEventListener("online", checkWhenVisible);
    return () => {
      ++sequence.current;
      busy.current = false;
      window.clearTimeout(initial);
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", checkWhenVisible);
      window.removeEventListener("online", checkWhenVisible);
    };
  }, [check, native]);

  function setChannel(value: Channel) {
    try {
      localStorage.setItem(CHANNEL_KEY, value);
      setChannelState(value);
    } catch {
      setError("Could not save the app update channel on this device.");
    }
  }

  async function download() {
    if (!available || opening) return;
    setOpening(true);
    setError(null);
    try {
      await invokeNative("native_open_app_update", { version: available.version });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setOpening(false);
    }
  }

  return { native, channel, setChannel, version, available, checking, opening, checkedAt, error, check, download };
}

const NativeUpdatesContext = createContext<ReturnType<typeof useUpdateController> | null>(null);

export function NativeUpdatesProvider({ children }: { children: ReactNode }) {
  const value = useUpdateController();
  return <NativeUpdatesContext.Provider value={value}>{children}</NativeUpdatesContext.Provider>;
}

export function useNativeUpdates() {
  const value = useContext(NativeUpdatesContext);
  if (!value) throw new Error("NativeUpdatesProvider is missing");
  return value;
}

export function NativeUpdateSettings() {
  const updates = useNativeUpdates();
  if (!updates.native) return null;
  return (
    <section className="settings-subsection">
      <div><p className="eyebrow">This device</p><h2>Android App Updates</h2></div>
      <p className="status-message">Installed version: {updates.version ? `v${updates.version}` : "Unavailable"}</p>
      <div className="segmented-control" role="group" aria-label="Android app release channel">
        {(["stable", "prerelease"] as const).map((channel) => (
          <button key={channel} type="button" className={updates.channel === channel ? "active" : undefined}
            aria-pressed={updates.channel === channel} onClick={() => updates.setChannel(channel)}>
            {channel === "stable" ? "Stable" : "Prerelease"}
          </button>
        ))}
      </div>
      <p className="status-message">{updates.channel === "stable" ? "Stable app releases only." : "Includes prerelease app builds for testing."} This does not change the server's update channel.</p>
      {updates.available ? <p role="status">Android app {updates.available.version} is available.</p>
        : updates.checkedAt && <p role="status">No newer Android app release is available on this channel.</p>}
      <div className="native-update-actions">
        <button type="button" className="secondary-button" disabled={updates.checking} onClick={() => void updates.check()}>
          <RefreshCw /><span>{updates.checking ? "Checking..." : "Check for updates"}</span>
        </button>
        {updates.available && <button type="button" className="secondary-button" disabled={updates.opening} onClick={() => void updates.download()}>
          <Download /><span>{updates.opening ? "Opening..." : `Download ${updates.available.version}`}</span>
        </button>}
      </div>
      {updates.error && <p className="error" role="alert">{updates.error}</p>}
    </section>
  );
}
