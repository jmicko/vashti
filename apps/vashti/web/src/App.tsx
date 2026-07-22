import { useCallback, useEffect, useState } from "react";
import { requestJson } from "./api";
import { AppShell } from "./AppShell";
import { AuthScreen } from "./auth";
import { BrandMark } from "./common";
import { resetPrivateStorageUser, setPrivateStorageUser } from "./privateChatStore";
import { markPerformance, measurePerformance } from "./performance";
import { useNativeConnections } from "./nativeConnections";
import type { LoadState, SessionResponse, User } from "./types";

export default function App() {
  const { activeConnection, syncActiveIdentity } = useNativeConnections();
  const [state, setState] = useState<LoadState>({ status: "loading" });

  const loadSession = useCallback(async () => {
    setState({ status: "loading" });
    markPerformance("vashti:session-request");
    try {
      const session = await requestJson<SessionResponse>("/api/auth/session");
      if (
        activeConnection &&
        (await syncActiveIdentity(session.instance_id, session.api_version))
      ) {
        window.location.reload();
        return;
      }
      if (session.is_authenticated && session.user) {
        setPrivateStorageUser(
          session.user.id,
          session.private_vault_key ?? undefined,
          activeConnection?.instance_id
        );
      } else {
        resetPrivateStorageUser();
      }
      markPerformance("vashti:session-ready");
      measurePerformance(
        "vashti:session-duration",
        "vashti:session-request",
        "vashti:session-ready"
      );
      setState({
        status: "loaded",
        session: { ...session, private_vault_key: null }
      });
    } catch (error) {
      setState({
        status: "error",
        message: error instanceof Error ? error.message : "Session request failed"
      });
    }
  }, [activeConnection, syncActiveIdentity]);

  useEffect(() => {
    void loadSession();
  }, [loadSession]);

  function updateSessionUser(user: User) {
    setState((current) =>
      current.status === "loaded" && current.session.user
        ? {
            status: "loaded",
            session: {
              ...current.session,
              user
            }
          }
        : current
    );
  }

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

  return (
    <AppShell
      user={state.session.user}
      onSessionChanged={loadSession}
      onUserChanged={updateSessionUser}
    />
  );
}
