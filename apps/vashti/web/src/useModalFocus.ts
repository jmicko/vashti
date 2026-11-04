import { useEffect, useRef, type RefObject } from "react";

const scopes: HTMLElement[] = [];
const selector =
  'button:not(:disabled),a[href],input:not(:disabled),textarea:not(:disabled),select:not(:disabled),[tabindex]:not([tabindex="-1"])';

export function useModalFocus(
  ref: RefObject<HTMLElement | null>,
  onClose: () => void,
  disabled = false,
) {
  const current = useRef({ onClose, disabled });
  current.current = { onClose, disabled };
  useEffect(() => {
    const scope = ref.current;
    if (!scope) return;
    const opener = document.activeElement;
    const hidden: Array<[HTMLElement, boolean]> = [];
    // Inert siblings at each level, not the dialog's own ancestor.
    for (
      let node: HTMLElement | null = scope;
      node?.parentElement;
      node = node.parentElement
    ) {
      for (const sibling of node.parentElement.children) {
        if (sibling !== node && sibling instanceof HTMLElement) {
          hidden.push([sibling, sibling.inert]);
          sibling.inert = true;
        }
      }
      if (node.parentElement === document.body) break;
    }
    scopes.push(scope);
    const controls = () =>
      Array.from(scope.querySelectorAll<HTMLElement>(selector)).filter(
        (element) =>
          !element.closest("[inert]") && element.getClientRects().length > 0,
      );
    const first = controls()[0];
    first?.focus({ preventScroll: true });
    const key = (event: KeyboardEvent) => {
      const nativeDialog = document.querySelector("dialog[open]");
      if (
        scopes[scopes.length - 1] !== scope ||
        (nativeDialog && !scope.contains(nativeDialog))
      )
        return;
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopImmediatePropagation();
        if (!current.current.disabled) current.current.onClose();
      } else if (event.key === "Tab") {
        const items = controls();
        if (!items.length) {
          event.preventDefault();
          return;
        }
        const index = items.indexOf(document.activeElement as HTMLElement);
        if (
          index < 0 ||
          (!event.shiftKey && index === items.length - 1) ||
          (event.shiftKey && index === 0)
        ) {
          event.preventDefault();
          items[event.shiftKey ? items.length - 1 : 0].focus({
            preventScroll: true,
          });
        }
      }
    };
    document.addEventListener("keydown", key, true);
    return () => {
      document.removeEventListener("keydown", key, true);
      scopes.splice(scopes.indexOf(scope), 1);
      for (const [element, inert] of hidden.reverse()) element.inert = inert;
      if (
        opener instanceof HTMLElement &&
        opener.isConnected &&
        !opener.closest("[inert]")
      )
        opener.focus({ preventScroll: true });
    };
  }, [ref]);
}
