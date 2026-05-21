import { useCallback, useEffect, useState } from "react";
import { requestJson } from "./api";
import { AppShell } from "./AppShell";
import { AuthScreen } from "./auth";
import { BrandMark } from "./common";
import type { LoadState, SessionResponse, User } from "./types";

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
