export function privateStreamTestEnabled() {
  try {
    return (
      new URLSearchParams(window.location.search).has("privateStreamTest") ||
      window.localStorage.getItem("vashti:private-stream-test") === "1"
    );
  } catch {
    return false;
  }
}
