import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { useGesture } from "@use-gesture/react";
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Minimize2,
  MoreHorizontal,
  X,
  ZoomIn,
} from "lucide-react";
import {
  boundPhoto,
  fittedView,
  photoSwipe,
  zoomPhoto,
  type PhotoSize,
  type PhotoView,
} from "./photoGeometry";
import "./photoViewer.css";

export type PhotoViewerOptions = {
  title: string;
  onClose: () => void;
  canPrevious: boolean;
  canNext: boolean;
};

export function PhotoViewer({
  id,
  title,
  onClose,
  onStep,
  canPrevious,
  canNext,
  children,
  actions,
  more,
  versions,
  previous,
  next,
  notice,
}: PhotoViewerOptions & {
  id: string;
  onStep: (direction: number) => void;
  children: ReactNode;
  actions?: ReactNode;
  more?: ReactNode;
  versions?: ReactNode;
  previous?: ReactNode;
  next?: ReactNode;
  notice?: ReactNode;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const track = useRef<HTMLDivElement>(null);
  const media = useRef<HTMLDivElement>(null);
  const view = useRef<PhotoView>({ ...fittedView });
  const imageSize = useRef<PhotoSize>({ width: 1, height: 1 });
  const viewport = useRef<PhotoSize>({ width: 1, height: 1 });
  const [chrome, setChrome] = useState(true);
  const [menu, setMenu] = useState(false);
  const [zoomed, setZoomed] = useState(false);
  const mounted = useRef(false);
  const menuRef = useRef(false);
  menuRef.current = menu;
  const callbacks = useRef({ onClose, onStep, canPrevious, canNext });
  callbacks.current = { onClose, onStep, canPrevious, canNext };
  const token = useRef(crypto.randomUUID());
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const settling = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const gestures = useRef(false);
  const moved = useRef(false);
  const dragging = useRef(false);
  const pendingStep = useRef(0);

  function paint(value: PhotoView, animate = false) {
    view.current = boundPhoto(value, imageSize.current, viewport.current);
    if (media.current) {
      media.current.style.transition = animate
        ? "transform 180ms ease-out"
        : "none";
      media.current.style.transform = `translate3d(${view.current.x}px,${view.current.y}px,0) scale(${view.current.scale})`;
    }
  }
  function slide(x: number, animate = false) {
    if (!track.current) return;
    track.current.style.transition = animate
      ? "transform 180ms ease-out"
      : "none";
    track.current.style.transform = `translate3d(${x}px,0,0)`;
  }
  function scheduleHide() {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      const active = document.activeElement;
      if (
        gestures.current ||
        menuRef.current ||
        dialog.current?.querySelector("details[open]") ||
        (active instanceof HTMLElement &&
          active.matches(":focus-visible") &&
          active.closest(".photo-viewer-controls,.photo-viewer-header"))
      ) {
        scheduleHide();
        return;
      }
      setChrome(false);
    }, 3000);
  }
  function reveal() {
    setChrome(true);
    scheduleHide();
  }
  function finishSlide() {
    clearTimeout(settling.current);
    if (pendingStep.current) step(pendingStep.current);
  }
  function close() {
    if (window.history.state?.vashtiMediaViewer === token.current)
      window.history.back();
    else callbacks.current.onClose();
  }
  function toggleMenu() {
    reveal();
    if (menuRef.current) {
      window.history.back();
      return;
    }
    window.history.pushState(
      { ...window.history.state, vashtiMediaPanel: true },
      "",
    );
    setMenu(true);
  }
  function step(direction: number) {
    if (
      direction < 0
        ? !callbacks.current.canPrevious
        : !callbacks.current.canNext
    ) {
      slide(0, true);
      return;
    }
    clearTimeout(settling.current);
    pendingStep.current = 0;
    slide(0);
    paint({ ...fittedView });
    setZoomed(false);
    callbacks.current.onStep(direction);
    reveal();
  }
  function zoom(scale: number, origin: [number, number] = [0, 0]) {
    paint(
      zoomPhoto(view.current, Math.max(1, Math.min(8, scale)), origin),
      true,
    );
    setZoomed(view.current.scale > 1.01);
    reveal();
  }
  useLayoutEffect(() => {
    paint({ ...fittedView });
    slide(0);
    setZoomed(false);
    reveal();
  }, [id]);
  useEffect(() => {
    if (menu)
      dialog.current
        ?.querySelector<HTMLButtonElement>(".photo-viewer-menu button")
        ?.focus({ preventScroll: true });
    else stage.current?.focus({ preventScroll: true });
  }, [menu]);
  useEffect(() => {
    const element = stage.current!;
    const resize = () => {
      viewport.current = {
        width: element.clientWidth,
        height: element.clientHeight,
      };
      paint(view.current);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    mounted.current = true;
    const element = dialog.current!;
    const opener = document.activeElement;
    element.showModal();
    stage.current?.focus({ preventScroll: true });
    if (window.history.state?.vashtiMediaViewer !== token.current) {
      window.history.pushState(
        { ...window.history.state, vashtiMediaViewer: token.current },
        "",
      );
    }
    const pop = () => {
      if (window.history.state?.vashtiMediaViewer !== token.current)
        callbacks.current.onClose();
      else setMenu(Boolean(window.history.state?.vashtiMediaPanel));
    };
    const key = (event: KeyboardEvent) => {
      if (event.key === "Tab") {
        reveal();
        return;
      }
      if (
        event.target instanceof HTMLElement &&
        event.target.closest("input,textarea,select")
      )
        return;
      if (menuRef.current) return;
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        step(event.key === "ArrowLeft" ? -1 : 1);
      } else if (["+", "=", "-", "0"].includes(event.key)) {
        event.preventDefault();
        zoom(
          event.key === "0"
            ? 1
            : view.current.scale * (event.key === "-" ? 0.5 : 2),
        );
      }
    };
    window.addEventListener("popstate", pop);
    window.addEventListener("keydown", key);
    return () => {
      mounted.current = false;
      element.close();
      window.removeEventListener("popstate", pop);
      window.removeEventListener("keydown", key);
      clearTimeout(timer.current);
      clearTimeout(settling.current);
      // Defer history cleanup so StrictMode's effect replay can reuse its entry.
      queueMicrotask(() => {
        if (
          !mounted.current &&
          window.history.state?.vashtiMediaViewer === token.current
        ) {
          window.history.go(window.history.state?.vashtiMediaPanel ? -2 : -1);
        }
      });
      if (opener instanceof HTMLElement && opener.isConnected)
        opener.focus({ preventScroll: true });
    };
  }, []);

  useGesture(
    {
      onDrag: ({
        first,
        last,
        canceled,
        pinching,
        cancel,
        movement: [x, y],
        velocity: [velocity],
        direction: [direction],
        tap,
        memo,
      }) => {
        if (pinching || (gestures.current && !dragging.current)) {
          cancel();
          return;
        }
        if (first) {
          finishSlide();
          slide(0);
          gestures.current = true;
          dragging.current = true;
          moved.current = false;
          memo = { ...view.current };
        }
        if (tap) {
          gestures.current = false;
          dragging.current = false;
          return memo;
        }
        if (!memo) return;
        moved.current = true;
        if (memo.scale > 1.01)
          paint({ scale: memo.scale, x: memo.x + x, y: memo.y + y });
        else if (Math.abs(x) > Math.abs(y) * 1.15) {
          const available =
            x > 0 ? callbacks.current.canPrevious : callbacks.current.canNext;
          slide(available ? x : x * 0.16);
        }
        if (last) {
          gestures.current = false;
          dragging.current = false;
          if (
            !canceled &&
            memo.scale <= 1.01 &&
            Math.abs(x) > Math.abs(y) * 1.15
          ) {
            const target = photoSwipe(
              x,
              velocity * direction,
              viewport.current.width,
            );
            const available =
              target < 0
                ? callbacks.current.canPrevious
                : callbacks.current.canNext;
            if (target && available) {
              const reduced = window.matchMedia(
                "(prefers-reduced-motion: reduce)",
              ).matches;
              slide(-target * (viewport.current.width + 24), !reduced);
              pendingStep.current = target;
              settling.current = setTimeout(
                () => step(target),
                reduced ? 0 : 150,
              );
            } else slide(0, true);
          } else slide(0, true);
          reveal();
        }
        return memo;
      },
      onDragEnd: () => {
        dragging.current = false;
        gestures.current = false;
      },
      onPinch: ({ first, last, origin, offset: [scale], memo, event }) => {
        event.preventDefault();
        const rect = stage.current!.getBoundingClientRect();
        const point: [number, number] = [
          origin[0] - rect.left - rect.width / 2,
          origin[1] - rect.top - rect.height / 2,
        ];
        if (first) {
          finishSlide();
          slide(0);
          gestures.current = true;
          dragging.current = false;
          memo = { view: { ...view.current }, point };
        }
        if (memo) paint(zoomPhoto(memo.view, scale, memo.point, point));
        moved.current = true;
        if (last) {
          gestures.current = false;
          setZoomed(view.current.scale > 1.01);
          reveal();
        }
        return memo;
      },
    },
    {
      target: stage,
      eventOptions: { passive: false },
      drag: { threshold: 6, filterTaps: true, pointer: { keys: false } },
      pinch: {
        from: () => [view.current.scale, 0],
        scaleBounds: { min: 1, max: 8 },
        rubberband: false,
      },
    },
  );

  return createPortal(
    <dialog
      ref={dialog}
      className="photo-viewer"
      aria-label={title}
      onCancel={(event) => {
        event.preventDefault();
        close();
      }}
      onPointerDownCapture={(event) => {
        if (
          !(event.target instanceof Element) ||
          !event.target.closest(".photo-viewer-stage")
        )
          reveal();
      }}
      onFocusCapture={(event) => {
        if (event.target instanceof Element &&
          (event.target.matches(":focus-visible") || !event.target.closest(".photo-viewer-stage"))) reveal();
      }}
    >
      <div
        ref={stage}
        className="photo-viewer-stage"
        tabIndex={0}
        aria-label="Image viewer"
        inert={menu || undefined}
        onPointerDown={() => {
          moved.current = false;
        }}
        onClick={() => {
          if (!moved.current) {
            setChrome((value) => !value);
            scheduleHide();
          }
        }}
        onDoubleClick={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          zoom(view.current.scale > 1.01 ? 1 : 2.5, [
            event.clientX - rect.left - rect.width / 2,
            event.clientY - rect.top - rect.height / 2,
          ]);
        }}
      >
        <div ref={track} className="photo-viewer-track">
          {previous && (
            <div
              className="photo-viewer-page photo-viewer-previous"
              aria-hidden="true"
              inert
            >
              {previous}
            </div>
          )}
          <div className="photo-viewer-page">
            <div
              ref={media}
              className="photo-viewer-media"
              onLoadCapture={(event) => {
                if (
                  event.target instanceof HTMLImageElement &&
                  !event.target.closest(".image-compare-reference")
                ) {
                  imageSize.current = {
                    width: event.target.naturalWidth,
                    height: event.target.naturalHeight,
                  };
                  paint(view.current);
                }
              }}
            >
              {children}
            </div>
          </div>
          {next && (
            <div
              className="photo-viewer-page photo-viewer-next"
              aria-hidden="true"
              inert
            >
              {next}
            </div>
          )}
        </div>
      </div>
      <header
        className="photo-viewer-header"
        data-visible={chrome || menu}
        inert={(!chrome && !menu) || undefined}
      >
        <button
          type="button"
          aria-label="Close image"
          title="Close image"
          onClick={close}
        >
          <ArrowLeft />
        </button>
        <span>{title}</span>
        {more ? (
          <button
            type="button"
            aria-label="More image actions"
            title="More image actions"
            aria-expanded={menu}
            onClick={toggleMenu}
          >
            <MoreHorizontal />
          </button>
        ) : (
          <span />
        )}
      </header>
      {!menu && (
        <>
          {canPrevious && (
            <button
              type="button"
              className="photo-viewer-nav photo-viewer-nav-previous"
              data-visible={chrome}
              inert={!chrome || undefined}
              aria-label="Previous image"
              title="Previous image"
              onClick={() => step(-1)}
            >
              <ChevronLeft />
            </button>
          )}
          {canNext && (
            <button
              type="button"
              className="photo-viewer-nav photo-viewer-nav-next"
              data-visible={chrome}
              inert={!chrome || undefined}
              aria-label="Next image"
              title="Next image"
              onClick={() => step(1)}
            >
              <ChevronRight />
            </button>
          )}
        </>
      )}
      <footer
        className="photo-viewer-controls"
        data-visible={chrome && !menu}
        inert={!chrome || menu || undefined}
      >
        {versions}
        <div className="photo-viewer-actions">
          {actions}
          <button
            type="button"
            aria-label={zoomed ? "Fit image" : "Zoom in"}
            title={zoomed ? "Fit image" : "Zoom in"}
            onClick={() => zoom(zoomed ? 1 : 2.5)}
          >
            {zoomed ? <Minimize2 /> : <ZoomIn />}
          </button>
        </div>
        {notice}
      </footer>
      {menu && (
        <section className="photo-viewer-menu" aria-label="Image actions">
          <header>
            <strong>Image actions</strong>
            <button
              type="button"
              aria-label="Close image actions"
              onClick={close}
            >
              <X />
            </button>
          </header>
          {more}
          <div className="photo-viewer-zoom-controls">
            <button type="button" onClick={() => zoom(view.current.scale / 2)}>
              Zoom out
            </button>
            <button type="button" onClick={() => zoom(1)}>
              Fit
            </button>
            <button type="button" onClick={() => zoom(view.current.scale * 2)}>
              Zoom in
            </button>
          </div>
        </section>
      )}
    </dialog>,
    document.body,
  );
}
