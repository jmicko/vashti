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

const cardGap = 8;
const maximumFlingSteps = 3;
const previewRadius = 3;

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
  children,
  isBusy,
  renderVersion,
  role,
  versionInfo
}: {
  children: ReactNode;
  isBusy: boolean;
  renderVersion: (version: MessageVersion, index: number) => ReactNode;
  role: ChatMessage["role"];
  versionInfo: VersionInfo;
}) {
  const swipeEnabled = useCoarsePointer();
  const carouselRef = useRef<HTMLDivElement | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const cardElementsRef = useRef(new Map<number, HTMLDivElement>());
  const cardHeightsRef = useRef<Record<number, number>>({});
  const interactionRef = useRef<"idle" | "dragging" | "settling">("idle");
  const gestureRef = useRef<GestureState | null>(null);
  const settleTimeoutRef = useRef<number | null>(null);
  const suppressClickRef = useRef(false);
  const suppressClickTimeoutRef = useRef<number | null>(null);

  useLayoutEffect(() => {
    clearSettleTimeout();
    interactionRef.current = "idle";
    updateInteractionClasses(carouselRef.current, "idle");
    updateCarouselPosition(viewportRef.current, 0, 0, cardHeightsRef.current[versionInfo.index]);
  }, [versionInfo.index]);

  useEffect(() => {
    return () => {
      clearSettleTimeout();
      if (suppressClickTimeoutRef.current !== null) {
        window.clearTimeout(suppressClickTimeoutRef.current);
      }
    };
  }, []);

  const renderedIndices = [
    versionInfo.index,
    ...versionInfo.versions
      .map((_, index) => index)
      .filter(
        (index) =>
          index !== versionInfo.index &&
          Math.abs(index - versionInfo.index) <= previewRadius
      )
  ];
  const renderedIndicesKey = renderedIndices.join(",");

  useLayoutEffect(() => {
    if (!swipeEnabled) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      let changed = false;
      const next = { ...cardHeightsRef.current };

      for (const entry of entries) {
        const index = Number((entry.target as HTMLElement).dataset.versionIndex);
        const height = entry.borderBoxSize[0]?.blockSize ?? entry.contentRect.height;
        if (Number.isFinite(index) && Math.abs((next[index] ?? 0) - height) > 0.5) {
          next[index] = height;
          changed = true;
        }
      }

      if (!changed) return;
      cardHeightsRef.current = next;
      if (interactionRef.current === "idle") {
        updateCarouselPosition(viewportRef.current, 0, 0, next[versionInfo.index]);
      }
    });

    for (const index of renderedIndices) {
      const element = cardElementsRef.current.get(index);
      if (element) {
        observer.observe(element);
      }
    }

    return () => observer.disconnect();
  }, [renderedIndicesKey, swipeEnabled]);

  if (!swipeEnabled) {
    return children;
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
      isInteractiveTarget(event.target) ||
      (event.pointerType === "mouse" && isSelectableMessageContent(event.target))
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
    }

    if (gesture.axis !== "horizontal") {
      return;
    }

    event.preventDefault();
    const timestamp = performance.now();
    const elapsed = Math.max(timestamp - gesture.lastTime, 1);
    const instantaneousVelocity = (event.clientX - gesture.lastX) / elapsed;
    gesture.velocityX = gesture.velocityX * 0.55 + instantaneousVelocity * 0.45;
    gesture.lastX = event.clientX;
    gesture.lastTime = timestamp;

    const isPastStart = versionInfo.index === 0 && deltaX > 0;
    const isPastEnd = versionInfo.index === versionInfo.total - 1 && deltaX < 0;
    const offset = isPastStart || isPastEnd ? deltaX * 0.18 : deltaX;
    const targetIndex =
      isPastStart || isPastEnd
        ? versionInfo.index
        : clampIndex(versionInfo.index + (deltaX < 0 ? 1 : -1), versionInfo.total);
    updateDragPosition(
      viewportRef.current,
      offset,
      versionInfo.index,
      targetIndex,
      cardDistanceFor(viewportRef.current),
      cardHeightsRef.current
    );
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

    const deltaX = event.clientX - gesture.startX;
    const width = viewportRef.current?.clientWidth ?? 0;
    const cardDistance = Math.max(
      width - carouselPeekWidth(viewportRef.current) * 2 + cardGap,
      1
    );
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

    const projectedDistance = Math.abs(deltaX) + Math.abs(gesture.velocityX) * 220;
    const requestedSteps = Math.max(1, Math.round(projectedDistance / cardDistance));
    const steps = Math.min(requestedSteps, maximumFlingSteps, availableSteps);
    settleToIndex(versionInfo.index + indexDirection * steps, cardDistance);
    releaseClickSuppression();
  }

  function handlePointerCancel(event: ReactPointerEvent<HTMLDivElement>) {
    if (gestureRef.current?.pointerId !== event.pointerId) {
      return;
    }
    gestureRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    resetPosition();
    releaseClickSuppression();
  }

  function resetPosition() {
    interactionRef.current = "idle";
    updateInteractionClasses(carouselRef.current, "idle");
    updateCarouselPosition(
      viewportRef.current,
      0,
      prefersReducedMotion() ? 0 : 180,
      cardHeightsRef.current[versionInfo.index]
    );
  }

  function settleToIndex(targetIndex: number, cardDistance: number) {
    const steps = targetIndex - versionInfo.index;
    const duration = prefersReducedMotion() ? 0 : Math.min(320, 180 + Math.abs(steps) * 42);
    interactionRef.current = "settling";
    updateInteractionClasses(carouselRef.current, "settling");
    updateCarouselPosition(
      viewportRef.current,
      -steps * cardDistance,
      duration,
      cardHeightsRef.current[targetIndex] ?? cardHeightsRef.current[versionInfo.index]
    );

    clearSettleTimeout();
    if (duration === 0) {
      versionInfo.onSelectIndex(targetIndex);
      return;
    }
    settleTimeoutRef.current = window.setTimeout(() => {
      settleTimeoutRef.current = null;
      versionInfo.onSelectIndex(targetIndex);
    }, duration);
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

  const previewVersions = versionInfo.versions
    .map((version, index) => ({ index, version }))
    .filter(
      ({ index }) =>
        index !== versionInfo.index &&
        Math.abs(index - versionInfo.index) <= previewRadius
    );
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
        {previewVersions.map(({ index, version }) => (
          <VersionPreview
            key={`${version.message.id}:${version.revision.id}`}
            distance={index - versionInfo.index}
            index={index}
            ref={(element) => registerCard(index, element)}
          >
            {renderVersion(version, index)}
          </VersionPreview>
        ))}
        <div
          className="message-version-current"
          data-version-index={versionInfo.index}
          ref={(element) => registerCard(versionInfo.index, element)}
        >
          {children}
        </div>
      </div>
    </div>
  );
}

function VersionPreview({
  children,
  distance,
  index,
  ref
}: {
  children: ReactNode;
  distance: number;
  index: number;
  ref: (element: HTMLDivElement | null) => void;
}) {
  const previewStyle = {
    "--message-version-distance-percent": `${distance * 100}%`,
    "--message-version-gap-offset": `${distance * cardGap}px`
  } as CSSProperties;

  return (
    <div
      className="message-version-preview"
      data-version-index={index}
      ref={ref}
      style={previewStyle}
      aria-hidden="true"
      inert
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
        "button, a, input, textarea, select, summary, details, label, [contenteditable='true']"
      )
    )
  );
}

function isSelectableMessageContent(target: EventTarget | null) {
  return (
    target instanceof Element &&
    Boolean(
      target.closest(
        ".message-markdown, .message-stream-content, .message-attachments, .message-stats-panel"
      )
    )
  );
}

function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function clampIndex(index: number, total: number) {
  return Math.max(0, Math.min(index, total - 1));
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
  transitionMs: number,
  height?: number
) {
  if (!viewport) return;
  viewport.style.setProperty("--message-version-offset", `${offset}px`);
  updateCarouselTransition(viewport, transitionMs);
  if (height && height > 0) {
    viewport.style.height = `${height}px`;
  }
}

function updateDragPosition(
  viewport: HTMLDivElement | null,
  offset: number,
  currentIndex: number,
  targetIndex: number,
  cardDistance: number,
  heights: Record<number, number>
) {
  if (!viewport) return;
  const currentHeight = heights[currentIndex] ?? 0;
  const targetHeight = heights[targetIndex] ?? currentHeight;
  const progress = Math.min(Math.abs(offset) / cardDistance, 1);
  const height = currentHeight + (targetHeight - currentHeight) * progress;
  updateCarouselPosition(viewport, offset, 0, height);
}

function useCoarsePointer() {
  const [isCoarsePointer, setIsCoarsePointer] = useState(() =>
    typeof window === "undefined" ? false : window.matchMedia("(pointer: coarse)").matches
  );

  useEffect(() => {
    const query = window.matchMedia("(pointer: coarse)");
    const update = () => setIsCoarsePointer(query.matches);
    query.addEventListener("change", update);
    update();
    return () => query.removeEventListener("change", update);
  }, []);

  return isCoarsePointer;
}
