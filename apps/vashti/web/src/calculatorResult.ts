import type { ToolUsageRecord } from "./types";

export function calculatorResult(usage: ToolUsageRecord): { expression: string; result: string } | null {
  if (usage.name !== "calculate" && usage.name !== "convert_units") return null;
  try {
    const value = JSON.parse(usage.result);
    if (!value || value.error || typeof value.result !== "number" || !Number.isFinite(value.result)) return null;
    if (usage.name === "calculate" && typeof value.expression === "string") {
      return { expression: value.expression, result: displayNumber(value.result) };
    }
    if (usage.name === "convert_units" && typeof value.value === "number" && Number.isFinite(value.value) &&
      typeof value.from_unit === "string" && typeof value.to_unit === "string") {
      return { expression: `${value.value} ${value.from_unit}`, result: `${displayNumber(value.result)} ${value.to_unit}` };
    }
  } catch {
    // Older or failed tool results still use the generic inspector.
  }
  return null;
}

function displayNumber(value: number): string {
  // Keep binary rounding noise out of the compact display; the inspector retains
  // the original result, including all digits returned by the server.
  return String(Number(value.toPrecision(15)));
}
