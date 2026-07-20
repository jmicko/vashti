import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState
} from "react";
import { RefreshCw } from "lucide-react";
import { useRegisterSW } from "virtual:pwa-register/react";

const UPDATE_INTERVAL_MS = 60 * 60 * 1000;
const VISIBILITY_UPDATE_THROTTLE_MS = 15 * 60 * 1000;

interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{
    outcome: "accepted" | "dismissed";
    platform: string;
  }>;
}

type InstallState =
  | "available"
  | "browser"
  | "development"
  | "insecure"
  | "installed"
  | "unsupported";

interface PwaContextValue {
  installState: InstallState;
  installError: string | null;
  isInstalling: boolean;
  install(): Promise<void>;
}

const PwaContext = createContext<PwaContextValue | null>(null);

function isStandaloneDisplay() {
  const navigatorWithStandalone = navigator as Navigator & { standalone?: boolean };
  return (
    window.matchMedia?.("(display-mode: standalone)").matches === true ||
    navigatorWithStandalone.standalone === true
  );
}

export function PwaProvider({ children }: { children: ReactNode }) {
  const [registration, setRegistration] = useState<ServiceWorkerRegistration | null>(null);
  const [installPrompt, setInstallPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [isInstalled, setIsInstalled] = useState(isStandaloneDisplay);
  const [isInstalling, setIsInstalling] = useState(false);
  const [installError, setInstallError] = useState<string | null>(null);
  const [updateError, setUpdateError] = useState<string | null>(null);
  const [isApplyingUpdate, setIsApplyingUpdate] = useState(false);

  const {
    needRefresh: [needRefresh, setNeedRefresh],
    updateServiceWorker
  } = useRegisterSW({
    immediate: true,
    onRegisteredSW(_serviceWorkerUrl, nextRegistration) {
      setRegistration(nextRegistration ?? null);
    },
    onRegisterError(error) {
      setUpdateError(
        error instanceof Error ? error.message : "The app update service could not start."
      );
    }
  });

  useEffect(() => {
    const displayMode = window.matchMedia?.("(display-mode: standalone)");
    const syncInstalledState = () => setIsInstalled(isStandaloneDisplay());
    const handleInstallPrompt = (event: Event) => {
      event.preventDefault();
      setInstallPrompt(event as BeforeInstallPromptEvent);
      setInstallError(null);
    };
    const handleInstalled = () => {
      setInstallPrompt(null);
      setIsInstalled(true);
      setInstallError(null);
    };

    window.addEventListener("beforeinstallprompt", handleInstallPrompt);
    window.addEventListener("appinstalled", handleInstalled);
    displayMode?.addEventListener("change", syncInstalledState);

    return () => {
      window.removeEventListener("beforeinstallprompt", handleInstallPrompt);
      window.removeEventListener("appinstalled", handleInstalled);
      displayMode?.removeEventListener("change", syncInstalledState);
    };
  }, []);

  useEffect(() => {
    if (!registration) {
      return;
    }

    const activeRegistration = registration;
    let lastUpdateCheck = Date.now();

    async function checkForUpdate(ignoreThrottle: boolean) {
      if (
        activeRegistration.installing ||
        !navigator.onLine ||
        document.visibilityState !== "visible" ||
        (!ignoreThrottle && Date.now() - lastUpdateCheck < VISIBILITY_UPDATE_THROTTLE_MS)
      ) {
        return;
      }

      lastUpdateCheck = Date.now();
      try {
        await activeRegistration.update();
      } catch {
        // A transient network failure should not interrupt the running app.
      }
    }

    const interval = window.setInterval(
      () => void checkForUpdate(true),
      UPDATE_INTERVAL_MS
    );
    const handleVisibilityChange = () => void checkForUpdate(false);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    void checkForUpdate(true);

    return () => {
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [registration]);

  const installState: InstallState = useMemo(() => {
    if (isInstalled) {
      return "installed";
    }
    if (!window.isSecureContext) {
      return "insecure";
    }
    if (!("serviceWorker" in navigator)) {
      return "unsupported";
    }
    if (!import.meta.env.PROD) {
      return "development";
    }
    if (installPrompt) {
      return "available";
    }
    return "browser";
  }, [installPrompt, isInstalled]);

  const install = useCallback(async () => {
    if (!installPrompt || isInstalling) {
      return;
    }

    setIsInstalling(true);
    setInstallError(null);
    try {
      await installPrompt.prompt();
      const choice = await installPrompt.userChoice;
      setInstallPrompt(null);
      if (choice.outcome === "accepted") {
        setIsInstalled(true);
      }
    } catch (error) {
      setInstallError(error instanceof Error ? error.message : "Vashti could not be installed.");
    } finally {
      setIsInstalling(false);
    }
  }, [installPrompt, isInstalling]);

  async function applyUpdate() {
    setIsApplyingUpdate(true);
    setUpdateError(null);
    try {
      await updateServiceWorker(true);
    } catch (error) {
      setUpdateError(error instanceof Error ? error.message : "The update could not be applied.");
      setIsApplyingUpdate(false);
    }
  }

  const contextValue = useMemo<PwaContextValue>(
    () => ({
      installState,
      installError,
      isInstalling,
      install
    }),
    [install, installError, installState, isInstalling]
  );

  return (
    <PwaContext.Provider value={contextValue}>
      {children}
      {needRefresh && (
        <aside className="pwa-update-notice" role="status" aria-live="polite">
          <RefreshCw aria-hidden="true" />
          <div className="pwa-update-copy">
            <strong>Vashti update ready</strong>
            <span>{updateError ?? "Reload when you are ready to use the new version."}</span>
          </div>
          <div className="pwa-update-actions">
            <button
              type="button"
              className="secondary-button"
              disabled={isApplyingUpdate}
              onClick={() => {
                setNeedRefresh(false);
                setUpdateError(null);
              }}
            >
              Later
            </button>
            <button type="button" disabled={isApplyingUpdate} onClick={() => void applyUpdate()}>
              {isApplyingUpdate ? "Updating..." : "Reload"}
            </button>
          </div>
        </aside>
      )}
    </PwaContext.Provider>
  );
}

export function usePwa() {
  const value = useContext(PwaContext);
  if (!value) {
    throw new Error("usePwa must be used within PwaProvider");
  }
  return value;
}
