import { FormEvent, useEffect, useState } from "react";
import { requestJson } from "./api";
import { BrandMark } from "./common";
import type { FormState, RegisteredUser, RegisterResponse } from "./types";

export function AuthScreen({
  canCreateAccount,
  onSessionChanged
}: {
  canCreateAccount: boolean;
  onSessionChanged: () => Promise<void>;
}) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [pendingUser, setPendingUser] = useState<RegisteredUser | null>(null);

  useEffect(() => {
    if (!canCreateAccount && mode === "register") {
      setMode("login");
    }
  }, [canCreateAccount, mode]);

  return (
    <main className="auth-page">
      <section className="auth-panel">
        <BrandMark />
        {pendingUser ? (
          <PendingApproval
            username={pendingUser.username}
            onBack={() => {
              setPendingUser(null);
              setMode("login");
            }}
          />
        ) : mode === "login" ? (
          <LoginForm
            canCreateAccount={canCreateAccount}
            onCreateAccount={() => setMode("register")}
            onSessionChanged={onSessionChanged}
          />
        ) : canCreateAccount ? (
          <RegisterForm
            onBackToLogin={() => setMode("login")}
            onPendingApproval={setPendingUser}
            onSessionChanged={onSessionChanged}
          />
        ) : (
          <LoginForm
            canCreateAccount={false}
            onCreateAccount={() => setMode("register")}
            onSessionChanged={onSessionChanged}
          />
        )}
      </section>
    </main>
  );
}

function LoginForm({
  canCreateAccount,
  onCreateAccount,
  onSessionChanged
}: {
  canCreateAccount: boolean;
  onCreateAccount: () => void;
  onSessionChanged: () => Promise<void>;
}) {
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [formState, setFormState] = useState<FormState>({
    isSubmitting: false,
    error: null
  });

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormState({ isSubmitting: true, error: null });

    try {
      await requestJson("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ identifier, password })
      });
      await onSessionChanged();
    } catch (error) {
      setFormState({
        isSubmitting: false,
        error: error instanceof Error ? error.message : "Login failed"
      });
    }
  }

  return (
    <>
      <h1>Sign In</h1>
      <form onSubmit={submit}>
        <label>
          <span>Username or Email</span>
          <input
            autoComplete="username"
            autoFocus
            required
            value={identifier}
            onChange={(event) => setIdentifier(event.target.value)}
          />
        </label>
        <label>
          <span>Password</span>
          <input
            autoComplete="current-password"
            required
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        {formState.error && <p className="error">{formState.error}</p>}
        <button type="submit" disabled={formState.isSubmitting}>
          {formState.isSubmitting ? "Signing In..." : "Sign In"}
        </button>
      </form>
      {canCreateAccount && (
        <button type="button" className="text-button" onClick={onCreateAccount}>
          Create Account
        </button>
      )}
    </>
  );
}

function RegisterForm({
  onBackToLogin,
  onPendingApproval,
  onSessionChanged
}: {
  onBackToLogin: () => void;
  onPendingApproval: (user: RegisteredUser) => void;
  onSessionChanged: () => Promise<void>;
}) {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [formState, setFormState] = useState<FormState>({
    isSubmitting: false,
    error: null
  });

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormState({ isSubmitting: true, error: null });

    try {
      const response = await requestJson<RegisterResponse>("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({
          username,
          email: email.trim() === "" ? null : email,
          password
        })
      });

      if (response.requires_approval) {
        onPendingApproval(response.user);
        return;
      }

      await onSessionChanged();
    } catch (error) {
      setFormState({
        isSubmitting: false,
        error: error instanceof Error ? error.message : "Account creation failed"
      });
    }
  }

  return (
    <>
      <h1>Create Account</h1>
      <form onSubmit={submit}>
        <label>
          <span>Username</span>
          <input
            autoComplete="username"
            autoFocus
            required
            value={username}
            onChange={(event) => setUsername(event.target.value)}
          />
        </label>
        <label>
          <span>Email</span>
          <input
            autoComplete="email"
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
        </label>
        <label>
          <span>Password</span>
          <input
            autoComplete="new-password"
            required
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        {formState.error && <p className="error">{formState.error}</p>}
        <button type="submit" disabled={formState.isSubmitting}>
          {formState.isSubmitting ? "Creating..." : "Create Account"}
        </button>
      </form>
      <button type="button" className="text-button" onClick={onBackToLogin}>
        Back to Sign In
      </button>
    </>
  );
}

function PendingApproval({ username, onBack }: { username: string; onBack: () => void }) {
  return (
    <>
      <h1>Pending Approval</h1>
      <p className="status-message">{username} is waiting for admin approval.</p>
      <button type="button" onClick={onBack}>
        Back to Sign In
      </button>
    </>
  );
}
