import type {
  ContextBlock,
  ContextBlockSelection,
  ContextCategory,
  ContextLibraryResponse
} from "./types";

export const MAX_CONTEXT_BLOCKS_PER_CHAT = 32;
export const MAX_COMPILED_SYSTEM_PROMPT_BYTES = 240_000;

export function contextSelectionFromBlock(
  block: ContextBlock,
  category: ContextCategory | null,
  position: number
): ContextBlockSelection {
  return {
    block_id: block.id,
    block_version_id: block.current_version.id,
    category_id: category?.id ?? null,
    category_name: category?.name ?? null,
    category_selection_mode: category?.selection_mode ?? null,
    version_number: block.current_version.version_number,
    name: block.current_version.name,
    content: block.current_version.content,
    position
  };
}

export function normalizeContextSelections(
  selections: ContextBlockSelection[] | null | undefined
) {
  return (selections ?? [])
    .slice()
    .sort((left, right) => left.position - right.position)
    .map((selection, position) => ({ ...selection, position }));
}

export function toggleContextBlock(
  selections: ContextBlockSelection[],
  block: ContextBlock,
  category: ContextCategory | null
) {
  const normalized = normalizeContextSelections(selections);
  const selected = normalized.some((selection) => selection.block_id === block.id);
  if (selected) {
    return normalizeContextSelections(
      normalized.filter((selection) => selection.block_id !== block.id)
    );
  }

  const withoutCategoryConflict = category?.selection_mode === "single"
    ? normalized.filter((selection) => selection.category_id !== category.id)
    : normalized;
  if (withoutCategoryConflict.length >= MAX_CONTEXT_BLOCKS_PER_CHAT) {
    throw new Error(`Select at most ${MAX_CONTEXT_BLOCKS_PER_CHAT} context blocks`);
  }
  return normalizeContextSelections([
    ...withoutCategoryConflict,
    contextSelectionFromBlock(block, category, withoutCategoryConflict.length)
  ]);
}

export function moveContextSelection(
  selections: ContextBlockSelection[],
  blockVersionId: string,
  direction: -1 | 1
) {
  const next = normalizeContextSelections(selections);
  const index = next.findIndex((selection) => selection.block_version_id === blockVersionId);
  const target = index + direction;
  if (index < 0 || target < 0 || target >= next.length) {
    return next;
  }
  [next[index], next[target]] = [next[target], next[index]];
  return normalizeContextSelections(next);
}

export function updateContextSelectionVersion(
  selections: ContextBlockSelection[],
  block: ContextBlock,
  library: ContextLibraryResponse
) {
  const category = block.category_id
    ? library.categories.find((candidate) => candidate.id === block.category_id) ?? null
    : null;
  const normalized = normalizeContextSelections(selections);
  const current = normalized.find((selection) => selection.block_id === block.id);
  if (!current) {
    return normalized;
  }
  const withoutCurrentOrConflict = normalized.filter(
    (selection) =>
      selection.block_id !== block.id &&
      !(category?.selection_mode === "single" && selection.category_id === category.id)
  );
  return normalizeContextSelections([
    ...withoutCurrentOrConflict,
    contextSelectionFromBlock(block, category, current.position)
  ]);
}

export function compileContextSystemPrompt(
  basePrompt: string | null | undefined,
  selections: ContextBlockSelection[]
) {
  const parts: string[] = [];
  const normalizedBase = basePrompt?.trim();
  if (normalizedBase) {
    parts.push(normalizedBase);
  }
  for (const selection of normalizeContextSelections(selections)) {
    parts.push(`[Context: ${selection.name.trim()}]\n${selection.content.trim()}`);
  }

  const compiled = parts.join("\n\n");
  if (new TextEncoder().encode(compiled).byteLength > MAX_COMPILED_SYSTEM_PROMPT_BYTES) {
    throw new Error(
      `The combined system prompt exceeds ${MAX_COMPILED_SYSTEM_PROMPT_BYTES.toLocaleString()} bytes`
    );
  }
  return compiled;
}
