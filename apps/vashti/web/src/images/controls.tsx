import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { X } from "lucide-react";

export function ImagePanel({ title, onClose, children, className = "" }: { title: string; onClose: () => void; children: ReactNode; className?: string }) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  useEffect(() => {
    const dialog = ref.current!;
    const opener = document.activeElement;
    dialog.showModal();
    return () => {
      dialog.close();
      if (opener instanceof HTMLElement && opener.isConnected)
        opener.focus({ preventScroll: true });
    };
  }, []);
  return <dialog ref={ref} className={`images-panel ${className}`} aria-labelledby={titleId}
    onCancel={event => { event.preventDefault(); onClose(); }}
    onClick={event => { if (event.target === ref.current) { const rect = ref.current.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) onClose(); } }}>
    <header><h2 id={titleId}>{title}</h2><button type="button" className="images-icon" onClick={onClose} title="Close panel" aria-label="Close panel"><X size={20} /></button></header>
    <div className="images-panel-content">{children}</div>
  </dialog>;
}

export function NumberField({ label, value, min, max, step = 1, onChange }: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (value: number) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  return <label>{label}<input type="number" inputMode={step < 1 ? "decimal" : "numeric"}
    min={min} max={max} step={step} value={text}
    onChange={event => {
      setText(event.target.value);
      if (event.target.value !== "" && event.target.validity.valid) onChange(Number(event.target.value));
    }}
    onBlur={() => setText(String(value))} /></label>;
}
