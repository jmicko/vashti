import { useRef, type ReactNode } from "react";

export function SwipeSurface({
  children,
  onStep,
}: {
  children: ReactNode;
  onStep: (direction: number) => void;
}) {
  const gesture = useRef<{
    id: number;
    x: number;
    y: number;
    dragging: boolean;
  } | null>(null);
  const suppressClick = useRef(false);
  return (
    <div
      className="images-swipe"
      onPointerDown={(event) => {
        if (
          !event.isPrimary ||
          (event.pointerType === "mouse" && event.button !== 0)
        )
          return;
        suppressClick.current = false;
        gesture.current = {
          id: event.pointerId,
          x: event.clientX,
          y: event.clientY,
          dragging: false,
        };
      }}
      onPointerMove={(event) => {
        const start = gesture.current;
        if (!start || start.id !== event.pointerId) return;
        const dx = event.clientX - start.x,
          dy = event.clientY - start.y;
        if (
          !start.dragging &&
          Math.abs(dx) > 14 &&
          Math.abs(dx) > Math.abs(dy) * 1.5
        ) {
          start.dragging = true;
          event.currentTarget.setPointerCapture(event.pointerId);
        }
        if (start.dragging) {
          event.preventDefault();
          suppressClick.current = true;
        }
      }}
      onPointerCancel={() => {
        gesture.current = null;
      }}
      onPointerUp={(event) => {
        const start = gesture.current;
        gesture.current = null;
        if (
          start?.id === event.pointerId &&
          start.dragging &&
          Math.abs(event.clientX - start.x) > 60
        )
          onStep(event.clientX < start.x ? 1 : -1);
      }}
      onClickCapture={(event) => {
        if (suppressClick.current) {
          event.preventDefault();
          event.stopPropagation();
          suppressClick.current = false;
        }
      }}
    >
      {children}
    </div>
  );
}
