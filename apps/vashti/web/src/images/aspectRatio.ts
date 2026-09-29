export const aspectRatios = ["1:1", "5:4", "4:5", "4:3", "3:4", "3:2", "2:3", "16:10", "10:16", "16:9", "9:16", "2:1", "1:2", "21:9", "9:21", "3:1", "1:3"];

// Preserve the working pixel budget while respecting the engine's size and grid limits.
export function aspectSize(width: number, height: number, pixels: number) {
  if (![width, height, pixels].every(n => Number.isFinite(n) && n > 0)) throw new Error("Invalid image dimensions");
  const scale = Math.min(Math.sqrt(pixels / (width * height)), 2048 / Math.max(width, height));
  const grid = (n: number) => Math.max(64, Math.min(2048, Math.round(n / 16) * 16));
  return { width: grid(width * scale), height: grid(height * scale) };
}

export function closestAspect(width: number, height: number) {
  const ratio = width / height;
  return aspectRatios.find(item => {
    const [w, h] = item.split(":").map(Number);
    return Math.abs(ratio / (w / h) - 1) < 0.012;
  }) ?? "Custom";
}
