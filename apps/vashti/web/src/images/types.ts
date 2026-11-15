export type Recipe = {
  workflow: string; prompt: string; negative_prompt: string; seed: string;
  width: number; height: number; steps: number; cfg: number;
  attention: "standard" | "fast"; flow_shift?: number | null;
};
export type Profile = {
  recipe: Recipe; model_may_change: string[]; seed_mode?: "random" | "increment" | "fixed";
  backend_id?: string | null; count?: number | null; auto_improve?: boolean; prompt_help?: PromptHelp | null;
  edit_mode?: "iterative" | "original";
};
export type Preferences = { generate: Profile; edit: Profile; hidden_controls: string[] };
export type PromptHelp = { backend_id: string; model_name: string; guidance: string };
export type Draft = {
  backend_id?: string | null;
  recipe: Recipe; input_ids: string[]; original_input_ids: string[];
  parent_result_id: string | null; prompt_version_id: string | null;
  seed_mode: "random" | "increment" | "fixed"; count: number | null;
  auto_improve: boolean; prompt_help: PromptHelp | null;
};
export type ImageSession = { id: string; title: string; mode: "generate" | "iterative" | "original"; draft: Draft; revision: number; updated_at: number; active_result_id: string | null };
export type RunRequest = { client_request_id: string; draft: Draft; improve_only: boolean; retry_result_id: string | null; fixed_endless_confirmed: boolean };
export type Preview = { step:number; steps:number; width:number; height:number; png_base64:string };
export type Run = { preview?: Preview | null; id: string; session_id: string; status: string; completed_count: number; effective_prompt: string | null; prompt_version_id: string | null; error: string | null; request: RunRequest; created_at: number };
export type ImageResult = { id: string; run_id: string; origin_run_id?: string; session_id: string; group_id: string; asset_id: string; parent_result_id: string | null; recipe: Recipe; input_ids: string[]; duration_ms: number; created_at: number; selected: boolean };
export type PromptVersion = { id: string; parent_id: string | null; prompt: string; guidance: string; source: string; model_name: string | null; created_at: number };
export type GalleryItem = { width?: number; height?: number; id: string; asset_id: string; group_id: string | null; session_id: string | null; variants: number; created_at: number };
export type ImageView = "home" | "flow" | "gallery" | "generate" | "edit";
export type ImageHost = { backend: { id: string; name: string }; service: ImageService };
export type ImageWorkflow = { id: string; state: "ready" | "unavailable"; reason: string | null; reference_images_min: number; reference_images_max: number; attention: string[] };
export type ImageService = { state: string; message: string; capabilities: null | {
  service: "kuno"; discovery_version: number; version: string; inference_auth: "none";
  images: { api_version: number; access: string; state: string; reason: string | null; generate_path: string; response_formats: string[]; workflows: ImageWorkflow[] };
} };
export type ServiceSettings = { backend_id: string | null; max_user_bytes: number; service: ImageService };
export function workflowLabel(id: string): string {
  return ({ "qwen-image-2.1-text-to-image/v1": "Qwen Image 2.1", "qwen-image-2.1-edit/v1": "Qwen Image 2.1 Edit", "qwen-image-2512-text-to-image/v1": "Qwen Image 2512" } as Record<string, string>)[id] ?? id;
}
export function canGenerate(service: ImageService | undefined, recipe: Recipe, references: number): boolean {
  if (!service || !["ready", "partial"].includes(service.state)) return false;
  const workflow = service.capabilities?.images.workflows.find(w => w.id === recipe.workflow && w.state === "ready");
  return Boolean(workflow && workflow.attention.includes(recipe.attention) && references >= workflow.reference_images_min && references <= workflow.reference_images_max);
}
export const activeRun = (run: Run) => ["queued", "improving", "generating", "saving", "stopping"].includes(run.status);
export function newDraft(profile: Profile): Draft {
  return { recipe: { ...profile.recipe, prompt: "" }, input_ids: [], original_input_ids: [], parent_result_id: null,
    backend_id: profile.backend_id ?? null, prompt_version_id: null, seed_mode: profile.seed_mode ?? "increment",
    count: profile.count === undefined ? 1 : profile.count, auto_improve: profile.auto_improve ?? false,
    prompt_help: profile.prompt_help ? { ...profile.prompt_help } : null };
}
export function profileFromDraft(profile: Profile, draft: Draft, editMode: "iterative" | "original" = "iterative"): Profile {
  return { ...profile, recipe: { ...draft.recipe, prompt: "" }, backend_id: draft.backend_id ?? null,
    seed_mode: draft.seed_mode, count: draft.count, auto_improve: draft.auto_improve,
    prompt_help: draft.prompt_help ? { ...draft.prompt_help } : null, edit_mode: editMode };
}
export function resetDraftSettings(draft: Draft, profile: Profile): Draft {
  const defaults = newDraft(profile);
  const editing = defaults.recipe.workflow === "qwen-image-2.1-edit/v1";
  return { ...draft, ...defaults, recipe: { ...defaults.recipe, prompt: draft.recipe.prompt },
    input_ids: editing ? [...draft.input_ids] : [], original_input_ids: editing ? [...draft.original_input_ids] : [],
    parent_result_id: draft.parent_result_id, prompt_version_id: draft.prompt_version_id };
}
export function newFlowWithSettings(draft: Draft): Draft {
  return { ...structuredClone(draft), recipe: { ...draft.recipe, prompt: "" }, input_ids: [], original_input_ids: [],
    parent_result_id: null, prompt_version_id: null };
}
/** Exact parent result IDs preserve branches when a different retry is selected. */
export function imageLineage(results: ImageResult[], leaf: string | null): ImageResult[] {
  const map = new Map(results.map(r => [r.id, r]));
  const visited = new Set<string>(); const path: ImageResult[] = [];
  while (leaf && !visited.has(leaf)) {
    visited.add(leaf); const result = map.get(leaf); if (!result) break;
    path.unshift(result); leaf = result.parent_result_id;
  }
  return path;
}
