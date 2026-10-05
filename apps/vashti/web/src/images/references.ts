import type { Draft, Profile, ImageWorkflow } from "./types";

export function referenceWorkflow(
  workflows: ImageWorkflow[],
  selected: string,
) {
  const ready = workflows.filter(
    (workflow) =>
      workflow.reference_images_max > 0 && workflow.state === "ready",
  );
  return ready.find((workflow) => workflow.id === selected) ?? ready[0];
}

export function referencePatch(
  draft: Draft,
  added: string[],
  profile: Profile,
  edit?: ImageWorkflow,
): Partial<Draft> {
  const input_ids = [...draft.input_ids];
  for (const id of added) if (!input_ids.includes(id)) input_ids.push(id);
  const limit = Math.min(10, edit?.reference_images_max || 10);
  if (input_ids.length > limit)
    throw new Error(`Choose at most ${limit} reference images.`);
  return {
    input_ids,
    original_input_ids: input_ids,
    parent_result_id: draft.input_ids.length ? draft.parent_result_id : null,
    ...(edit && draft.recipe.workflow !== edit.id
      ? {
          recipe: {
            ...profile.recipe,
            prompt: draft.recipe.prompt,
            negative_prompt: draft.recipe.negative_prompt,
            workflow: edit.id,
          },
          parent_result_id: null,
        }
      : {}),
  };
}

export function toggleReference(ids: string[], id: string, limit: number) {
  return ids.includes(id)
    ? ids.filter((value) => value !== id)
    : ids.length < limit
      ? [...ids, id]
      : ids;
}
