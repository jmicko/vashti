import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type ReactNode
} from "react";
import type { ChatMessage, MessageVersion, VersionInfo } from "./types";
import { usesMobileInputBehavior } from "./viewport";

const cardGap = 8;
const maximumFlingSteps = 4;
const minimumVisibleShelfHeight = 160;

type GestureState = {
  pointerId: number;
  startX: number;
  startY: number;
  lastX: number;
  lastTime: number;
  velocityX: number;
  axis: "horizontal" | "vertical" | null;
};

export function MessageVersionCarousel({
  isBusy,
  renderVersion,
  role,
  versionInfo
}: {
  isBusy: boolean;
  renderVersion: (version: MessageVersion, index: number, isCurrent: boolean) => ReactNode;
  role: ChatMessage["role"];
  versionInfo: VersionInfo;
}) {
  const swipeEnabled = useMobileSwipeInput();
  const carouselRef = useRef<HTMLDivElement | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const cardElementsRef = useRef(new Map<number, HTMLDivElement>());
  const interactionRef = useRef<"idle" | "dragging" | "settling">("idle");
  const gestureRef = useRef<GestureState | null>(null);
  const settleTimeoutRef = useRef<number | null>(null);
  const suppressClickRef = useRef(false);
  const suppressClickTimeoutRef = useRef<number | null>(null);

  useLayoutEffect(() => {
    clearSettleTimeout();
    interactionRef.current = "idle";
    updateInteractionClasses(carouselRef.current, "idle");
    updateCarouselPosition(viewportRef.current, 0, 0);
    updateShelfOffset(viewportRef.current, 0);
  }, [versionInfo.index]);

  useEffect(() => {
    return () => {
      clearSettleTimeout();
      if (suppressClickTimeoutRef.current !== null) {
        window.clearTimeout(suppressClickTimeoutRef.current);
      }
    };
  }, []);

  if (!swipeEnabled) {
    const currentVersion = versionInfo.versions[versionInfo.index];
    return currentVersion
      ? renderVersion(currentVersion, versionInfo.index, true)
      : null;
  }

  function clearSettleTimeout() {
    if (settleTimeoutRef.current !== null) {
      window.clearTimeout(settleTimeoutRef.current);
      settleTimeoutRef.current = null;
    }
  }

  function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (
      isBusy ||
      interactionRef.current === "settling" ||
      !event.isPrimary ||
      (event.pointerType === "mouse" && event.button !== 0) ||
      isInteractiveTarget(event.target)
    ) {
      return;
    }

    const timestamp = performance.now();
    gestureRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      lastX: event.clientX,
      lastTime: timestamp,
      velocityX: 0,
      axis: null
    };
    updateCarouselTransition(viewportRef.current, 0);
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) {
      return;
    }

    const deltaX = event.clientX - gesture.startX;
    const deltaY = event.clientY - gesture.startY;

    if (!gesture.axis) {
      if (Math.max(Math.abs(deltaX), Math.abs(deltaY)) < 7) {
        return;
      }
      gesture.axis =
        Math.abs(deltaX) > Math.abs(deltaY) * 1.12 ? "horizontal" : "vertical";
      if (gesture.axis === "vertical") {
        gestureRef.current = null;
        return;
      }
      event.currentTarget.setPointerCapture(event.pointerId);
      interactionRef.current = "dragging";
      updateInteractionClasses(carouselRef.current, "dragging");
      updateShelfOffset(viewportRef.current, visibleShelfOffset(viewportRef.current));
    }

    if (gesture.axis !== "horizontal") {
      return;
    }

    event.preventDefault();
    const timestamp = performance.now();
    const elapsed = Math.max(timestamp - gesture.lastTime, 1);
    const instantaneousVelocity = (event.clientX - gesture.lastX) / elapsed;
    gesture.velocityX = gesture.velocityX * 0.58 + instantaneousVelocity * 0.42;
    gesture.lastX = event.clientX;
    gesture.lastTime = timestamp;

    const isPastStart = versionInfo.index === 0 && deltaX > 0;
    const isPastEnd = versionInfo.index === versionInfo.total - 1 && deltaX < 0;
    const offset = isPastStart || isPastEnd ? deltaX * 0.18 : deltaX;
    updateCarouselPosition(viewportRef.current, offset, 0);
    if (Math.abs(deltaX) > 9) {
      suppressClickRef.current = true;
    }
  }

  function handlePointerEnd(event: ReactPointerEvent<HTMLDivElement>) {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) {
      return;
    }

    gestureRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }

    if (gesture.axis !== "horizontal") {
      resetPosition();
      return;
    }

    finishHorizontalGesture(gesture);
  }

  function finishHorizontalGesture(gesture: GestureState) {
    const deltaX = gesture.lastX - gesture.startX;
    const cardDistance = cardDistanceFor(viewportRef.current);
    const displacementThreshold = Math.min(72, Math.max(42, cardDistance * 0.16));
    const hasIntent =
      Math.abs(deltaX) >= displacementThreshold || Math.abs(gesture.velocityX) >= 0.52;

    if (!hasIntent) {
      resetPosition();
      releaseClickSuppression();
      return;
    }

    const indexDirection = deltaX < 0 ? 1 : -1;
    const availableSteps =
      indexDirection > 0 ? versionInfo.total - versionInfo.index - 1 : versionInfo.index;
    if (availableSteps <= 0) {
      resetPosition();
      releaseClickSuppression();
      return;
    }

    const requestedSteps = flingStepCount(deltaX, gesture.velocityX, cardDistance);
    const steps = Math.min(requestedSteps, maximumFlingSteps, availableSteps);
    settleToIndex(versionInfo.index + indexDirection * steps, cardDistance);
    releaseClickSuppression();
  }

  function handlePointerCancel(event: ReactPointerEvent<HTMLDivElement>) {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) {
      return;
    }
    gestureRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (gesture.axis === "horizontal") {
      finishHorizontalGesture(gesture);
    } else {
      resetPosition();
      releaseClickSuppression();
    }
  }

  function resetPosition() {
    interactionRef.current = "idle";
    updateInteractionClasses(carouselRef.current, "idle");
    updateCarouselPosition(viewportRef.current, 0, prefersReducedMotion() ? 0 : 160);
    updateShelfOffset(viewportRef.current, 0);
  }

  function settleToIndex(targetIndex: number, cardDistance: number) {
    const steps = targetIndex - versionInfo.index;
    const duration = prefersReducedMotion() ? 0 : Math.min(280, 150 + Math.abs(steps) * 28);
    interactionRef.current = "settling";
    updateInteractionClasses(carouselRef.current, "settling");
    updateCarouselPosition(viewportRef.current, -steps * cardDistance, duration);

    clearSettleTimeout();
    const selectTarget = () => {
      settleTimeoutRef.current = null;
      const anchorTopOffset = selectionAnchorTopOffset(
        cardElementsRef.current.get(targetIndex),
        viewportRef.current
      );
      versionInfo.onSelectIndex(
        targetIndex,
        anchorTopOffset === undefined ? undefined : { topOffset: anchorTopOffset }
      );
    };
    if (duration === 0) {
      selectTarget();
      return;
    }
    settleTimeoutRef.current = window.setTimeout(selectTarget, duration);
  }

  function releaseClickSuppression() {
    if (!suppressClickRef.current) {
      return;
    }
    if (suppressClickTimeoutRef.current !== null) {
      window.clearTimeout(suppressClickTimeoutRef.current);
    }
    suppressClickTimeoutRef.current = window.setTimeout(() => {
      suppressClickRef.current = false;
      suppressClickTimeoutRef.current = null;
    }, 0);
  }

  function registerCard(index: number, element: HTMLDivElement | null) {
    if (element) {
      cardElementsRef.current.set(index, element);
    } else {
      cardElementsRef.current.delete(index);
    }
  }

  return (
    <div
      className={`message-version-carousel message-version-carousel-${role}`}
      ref={carouselRef}
      aria-label={`Message version ${versionInfo.index + 1} of ${versionInfo.total}`}
      aria-roledescription="carousel"
      role="group"
      onClickCapture={(event) => {
        if (!suppressClickRef.current) {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        suppressClickRef.current = false;
      }}
    >
      <div
        className="message-version-viewport"
        ref={viewportRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerEnd}
        onPointerCancel={handlePointerCancel}
      >
        {versionInfo.versions.map((version, index) => {
          const isCurrent = index === versionInfo.index;
          return (
            <VersionCard
              key={`${version.message.id}:${version.revision.id}`}
              distance={index - versionInfo.index}
              index={index}
              isCurrent={isCurrent}
              ref={(element) => registerCard(index, element)}
            >
              {renderVersion(version, index, isCurrent)}
            </VersionCard>
          );
        })}
      </div>
    </div>
  );
}

function VersionCard({
  children,
  distance,
  index,
  isCurrent,
  ref
}: {
  children: ReactNode;
  distance: number;
  index: number;
  isCurrent: boolean;
  ref: (element: HTMLDivElement | null) => void;
}) {
  const style = {
    "--message-version-distance-percent": `${distance * 100}%`,
    "--message-version-gap-offset": `${distance * cardGap}px`
  } as CSSProperties;

  return (
    <div
      className={
        isCurrent
          ? "message-version-card message-version-card-current"
          : "message-version-card message-version-card-adjacent"
      }
      data-version-index={index}
      ref={ref}
      style={style}
      aria-hidden={isCurrent ? undefined : "true"}
      inert={isCurrent ? undefined : true}
    >
      {children}
    </div>
  );
}

function isInteractiveTarget(target: EventTarget | null) {
  return (
    target instanceof Element &&
    Boolean(
      target.closest(
        "button, a, input, textarea, select, summary, label, [contenteditable='true']"
      )
    )
  );
}

function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function carouselPeekWidth(viewport: HTMLDivElement | null) {
  if (!viewport) {
    return 0;
  }
  return Number.parseFloat(window.getComputedStyle(viewport).paddingLeft) || 0;
}

function cardDistanceFor(viewport: HTMLDivElement | null) {
  return Math.max(
    (viewport?.clientWidth ?? 0) - carouselPeekWidth(viewport) * 2 + cardGap,
    1
  );
}

function flingStepCount(deltaX: number, velocityX: number, cardDistance: number) {
  const projectedDistance =
    Math.abs(deltaX) + Math.min(Math.abs(velocityX), 2.4) * 280;
  return Math.max(1, Math.round(projectedDistance / Math.max(cardDistance * 0.72, 1)));
}

function visibleShelfOffset(viewport: HTMLDivElement | null) {
  const list = viewport?.closest<HTMLElement>(".message-list");
  if (!viewport || !list) {
    return 0;
  }

  const viewportRect = viewport.getBoundingClientRect();
  const listRect = list.getBoundingClientRect();
  const desiredOffset = listRect.top + 12 - viewportRect.top;
  const maximumOffset = Math.max(0, viewportRect.height - minimumVisibleShelfHeight);
  return Math.max(0, Math.min(desiredOffset, maximumOffset));
}

function selectionAnchorTopOffset(
  card: HTMLDivElement | undefined,
  viewport: HTMLDivElement | null
) {
  const list = viewport?.closest<HTMLElement>(".message-list");
  const anchor = card?.querySelector<HTMLElement>(".version-switcher") ?? card;
  if (!list || !anchor) {
    return undefined;
  }
  return anchor.getBoundingClientRect().top - list.getBoundingClientRect().top;
}

function updateCarouselTransition(viewport: HTMLDivElement | null, transitionMs: number) {
  viewport?.style.setProperty("--message-version-transition", `${transitionMs}ms`);
}

function updateInteractionClasses(
  carousel: HTMLDivElement | null,
  interaction: "idle" | "dragging" | "settling"
) {
  if (!carousel) return;
  carousel.classList.toggle("is-dragging", interaction === "dragging");
  carousel.classList.toggle("is-settling", interaction === "settling");
}

function updateCarouselPosition(
  viewport: HTMLDivElement | null,
  offset: number,
  transitionMs: number
) {
  if (!viewport) return;
  viewport.style.setProperty("--message-version-offset", `${offset}px`);
  updateCarouselTransition(viewport, transitionMs);
}

function updateShelfOffset(viewport: HTMLDivElement | null, offset: number) {
  viewport?.style.setProperty("--message-version-shelf-offset", `${offset}px`);
}

function useMobileSwipeInput() {
  const [isMobileInput, setIsMobileInput] = useState(usesMobileInputBehavior);

  useEffect(() => {
    const queries = [
      window.matchMedia("(hover: hover) and (pointer: fine)"),
      window.matchMedia("(hover: none) and (pointer: coarse)")
    ];
    const update = () => setIsMobileInput(usesMobileInputBehavior());
    for (const query of queries) {
      query.addEventListener("change", update);
    }
    window.addEventListener("resize", update);
    update();
    return () => {
      for (const query of queries) {
        query.removeEventListener("change", update);
      }
      window.removeEventListener("resize", update);
    };
  }, []);

  return isMobileInput;
}
