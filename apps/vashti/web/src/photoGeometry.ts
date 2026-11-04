export type PhotoView = { scale: number; x: number; y: number };
export type PhotoSize = { width: number; height: number };
export const fittedView: PhotoView = { scale: 1, x: 0, y: 0 };

export function fitPhoto(image: PhotoSize, viewport: PhotoSize): PhotoSize {
  const ratio = Math.min(
    viewport.width / Math.max(1, image.width),
    viewport.height / Math.max(1, image.height),
  );
  return { width: image.width * ratio, height: image.height * ratio };
}

export function boundPhoto(
  view: PhotoView,
  image: PhotoSize,
  viewport: PhotoSize,
): PhotoView {
  const scale = Math.max(
    1,
    Math.min(8, Number.isFinite(view.scale) ? view.scale : 1),
  );
  const fit = fitPhoto(image, viewport);
  const x = Math.max(0, (fit.width * scale - viewport.width) / 2);
  const y = Math.max(0, (fit.height * scale - viewport.height) / 2);
  return {
    scale,
    x: x ? Math.max(-x, Math.min(x, view.x)) : 0,
    y: y ? Math.max(-y, Math.min(y, view.y)) : 0,
  };
}

export function zoomPhoto(
  view: PhotoView,
  scale: number,
  origin: [number, number],
  nextOrigin = origin,
): PhotoView {
  const ratio = scale / view.scale;
  return {
    scale,
    x: nextOrigin[0] - (origin[0] - view.x) * ratio,
    y: nextOrigin[1] - (origin[1] - view.y) * ratio,
  };
}

export function photoSwipe(
  distance: number,
  velocity: number,
  width: number,
): number {
  const threshold = Math.max(42, Math.min(90, width * 0.22));
  if (
    Math.abs(distance) < threshold &&
    !(Math.abs(distance) > 24 && Math.abs(velocity) > 0.55)
  )
    return 0;
  return distance < 0 ? 1 : -1;
}
