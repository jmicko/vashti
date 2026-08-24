import {
  useCallback,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent
} from "react";
import { usesMobileInputBehavior } from "./viewport";

const ARM_DISTANCE_REM = 3;
const MAX_DISTANCE_REM = 4;
const DRAG_START_REM = 0.35;
const PULL_RESISTANCE = 0.5;

type PullState = "idle" | "pulling" | "armed" | "refreshing";

type PullGesture = {
  pointerId: number;
  startX: number;
  startY: number;
  isDragging: boolean;
  isArmed: boolean;
};

function rootFontSize() {
  const parsed = Number.parseFloat(window.getComputedStyle(document.documentElement).fontSize);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 16;
}

export function useTopbarPullToRefresh(onRefresh: () => Promise<void>) {
  const [pullState, setPullState] = useState<PullState>("idle");
  const [pullDistance, setPullDistance] = useState(0);
  const gestureRef = useRef<PullGesture | null>(null);
  const suppressClickRef = useRef(false);
  const isEnabled = usesMobileInputBehavior();

  const resetPull = useCallback(() => {
    gestureRef.current = null;
    setPullState("idle");
    setPullDistance(0);
  }, []);

  function handlePointerDown(event: ReactPointerEvent<HTMLElement>) {
    if (
      !isEnabled ||
      pullState === "refreshing" ||
      event.pointerType !== "touch" ||
      !event.isPrimary
    ) {
      return;
    }

    gestureRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      isDragging: false,
      isArmed: false
    };
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLElement>) {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) {
      return;
    }

    const deltaX = event.clientX - gesture.startX;
    const deltaY = event.clientY - gesture.startY;
    const rem = rootFontSize();

    if (!gesture.isDragging) {
      if (deltaY <= 0 || Math.abs(deltaX) > deltaY) {
        if (Math.abs(deltaX) > DRAG_START_REM * rem || deltaY < -DRAG_START_REM * rem) {
          gestureRef.current = null;
        }
        return;
      }

      if (deltaY < DRAG_START_REM * rem) {
        return;
      }

      gesture.isDragging = true;
      event.currentTarget.setPointerCapture(event.pointerId);
    }

    event.preventDefault();
    const nextDistance = Math.min(deltaY * PULL_RESISTANCE, MAX_DISTANCE_REM * rem);
    const isArmed = nextDistance >= ARM_DISTANCE_REM * rem;
    gesture.isArmed = isArmed;
    setPullDistance(nextDistance);
    setPullState(isArmed ? "armed" : "pulling");
  }

  function finishPointer(event: ReactPointerEvent<HTMLElement>) {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) {
      return;
    }

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }

    gestureRef.current = null;
    if (!gesture.isDragging) {
      return;
    }

    suppressClickRef.current = true;
    window.setTimeout(() => {
      suppressClickRef.current = false;
    }, 350);

    if (!gesture.isArmed) {
      resetPull();
      return;
    }

    setPullState("refreshing");
    setPullDistance(ARM_DISTANCE_REM * rootFontSize());
    void Promise.resolve().then(onRefresh).catch(resetPull);
  }

  function cancelPointer(event: ReactPointerEvent<HTMLElement>) {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) {
      return;
    }

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    resetPull();
  }

  function suppressDraggedClick(event: ReactMouseEvent<HTMLElement>) {
    if (!suppressClickRef.current) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
  }

  return {
    isEnabled,
    pullState,
    pullStyle: { "--topbar-pull-distance": `${pullDistance}px` } as CSSProperties,
    handlers: {
      onPointerDown: handlePointerDown,
      onPointerMove: handlePointerMove,
      onPointerUp: finishPointer,
      onPointerCancel: cancelPointer,
      onClickCapture: suppressDraggedClick
    }
  };
}
