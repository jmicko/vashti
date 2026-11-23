import { useEffect, useId, useRef, type ReactNode } from "react";
import { MoreHorizontal } from "lucide-react";

export function ActionMenu({
  label,
  children,
  icon,
  disabled = false,
}: {
  label: string;
  children: ReactNode;
  icon?: ReactNode;
  disabled?: boolean;
}) {
  const id = useId();
  const panel = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  function position() {
    const menu = panel.current;
    const rect = trigger.current?.getBoundingClientRect();
    if (!menu?.matches(":popover-open") || !rect) return;
    const size = menu.getBoundingClientRect();
    menu.style.left = `${Math.max(8, Math.min(window.innerWidth - size.width - 8, rect.left))}px`;
    menu.style.top = `${rect.bottom + 8 + size.height <= window.innerHeight - 8
      ? rect.bottom + 8 : Math.max(8, rect.top - size.height - 8)}px`;
  }
  useEffect(() => {
    window.addEventListener("resize", position);
    return () => window.removeEventListener("resize", position);
  }, []);
  return (
    <>
      <button
        ref={trigger}
        type="button"
        className="message-icon-button"
        aria-label={label}
        title={label}
        popoverTarget={id}
        disabled={disabled}
      >
        {icon ?? <MoreHorizontal />}
      </button>
      <div
        ref={panel}
        id={id}
        popover="auto"
        className="action-menu"
        role="group"
        aria-label={label}
        onToggle={position}
        onClick={(event) => {
          if (
            event.target instanceof Element &&
            event.target.closest("button:not(:disabled)")
          )
            panel.current?.hidePopover();
        }}
      >
        {children}
      </div>
    </>
  );
}
