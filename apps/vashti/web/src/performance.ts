export function markPerformance(name: string) {
  if (typeof performance === "undefined") {
    return;
  }

  performance.clearMarks(name);
  performance.mark(name);
}

export function measurePerformance(name: string, startMark: string, endMark: string) {
  if (typeof performance === "undefined") {
    return;
  }

  try {
    performance.clearMeasures(name);
    performance.measure(name, startMark, endMark);
  } catch {
    // Performance marks are diagnostic only and must never affect startup.
  }
}
