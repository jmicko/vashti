import { useEffect, useState } from "react";

export function RetroLoader() {
  const frames = ["-______", "_-_____", "__-____", "___-___", "____-__", "_____-_", "______-"];
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const interval = window.setInterval(() => {
      setFrame((current) => (current + 1) % frames.length);
    }, 120);

    return () => window.clearInterval(interval);
  }, [frames.length]);

  return <span className="retro-loader">{frames[frame]}</span>;
}

export function ThinkingLoader() {
  const frames = ["Thinking", "Thinking.", "Thinking..", "Thinking..."];
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const interval = window.setInterval(() => {
      setFrame((current) => (current + 1) % frames.length);
    }, 140);

    return () => window.clearInterval(interval);
  }, [frames.length]);

  return <span className="thinking-loader">{frames[frame]}</span>;
}

export function ConfirmDialog({
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  isBusy = false,
  onCancel,
  onConfirm
}: {
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  isBusy?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && !isBusy) {
        onCancel();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isBusy, onCancel]);

  return (
    <div className="confirm-backdrop" role="presentation">
      <section
        className="confirm-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
      >
        <h2 id="confirm-dialog-title">{title}</h2>
        <p>{message}</p>
        <div className="dialog-actions">
          <button type="button" className="secondary-button" disabled={isBusy} onClick={onCancel}>
            {cancelLabel}
          </button>
          <button type="button" className="danger-button" disabled={isBusy} onClick={onConfirm}>
            {isBusy ? <RetroLoader /> : confirmLabel}
          </button>
        </div>
      </section>
    </div>
  );
}

export function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <img
      className={compact ? "brand-logo brand-logo-compact" : "brand-logo"}
      src="/brand/vashti-logo.png"
      alt="Vashti"
    />
  );
}
