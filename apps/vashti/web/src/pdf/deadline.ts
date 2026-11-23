export function withinDeadline<T>(work: Promise<T>, milliseconds: number, signal?: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      void work.catch(() => undefined);
      reject(signal.reason);
      return;
    }
    const abort = () => { finish(); reject(signal?.reason); };
    const timer = setTimeout(() => { finish(); reject(new DOMException("PDF processing timed out.", "TimeoutError")); }, milliseconds);
    function finish() {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    }
    signal?.addEventListener("abort", abort, { once: true });
    void work.then(value => { finish(); resolve(value); }, error => { finish(); reject(error); });
  });
}
