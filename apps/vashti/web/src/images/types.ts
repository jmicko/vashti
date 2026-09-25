export type Recipe = {
  workflow: string; prompt: string; negative_prompt: string; seed: string;
  width: number; height: number; steps: number; cfg: number;
  attention: "standard" | "fast"; flow_shift?: number | null;
};
export type Profile = { recipe: Recipe; model_may_change: string[]; seed_mode?: "random" | "increment" | "fixed" };
export type Preferences = { generate: Profile; edit: Profile; hidden_controls: string[] };
export type PromptHelp = { backend_id: string; model_name: string; guidance: string };
export type Draft = {
  recipe: Recipe; input_ids: string[]; original_input_ids: string[];
  parent_result_id: string | null; prompt_version_id: string | null;
  seed_mode: "random" | "increment" | "fixed"; count: number | null;
  auto_improve: boolean; prompt_help: PromptHelp | null;
};
export type ImageSession = { id: string; title: string; mode: "generate" | "iterative" | "original"; draft: Draft; revision: number; updated_at: number };
export type RunRequest = { client_request_id: string; draft: Draft; improve_only: boolean; retry_result_id: string | null; fixed_endless_confirmed: boolean };
export type Preview = { step:number; steps:number; width:number; height:number; png_base64:string };
export type Run = { preview?: Preview | null; id: string; session_id: string; status: string; completed_count: number; effective_prompt: string | null; prompt_version_id: string | null; error: string | null; request: RunRequest };
export type ImageResult = { id: string; run_id: string; session_id: string; group_id: string; asset_id: string; parent_result_id: string | null; recipe: Recipe; input_ids: string[]; duration_ms: number; created_at: number; selected: boolean };
export type PromptVersion = { id: string; parent_id: string | null; prompt: string; guidance: string; source: string; model_name: string | null; created_at: number };
export type GalleryItem = { id: string; asset_id: string; group_id: string | null; session_id: string | null; variants: number; created_at: number };
export type ImageView = "gallery" | "generate" | "edit";
export type ServiceSettings = { backend_id: string | null; token_configured: boolean; max_user_bytes: number };
export const activeRun = (run: Run) => ["queued", "improving", "generating", "saving", "stopping"].includes(run.status);
export function newDraft(profile: Profile): Draft {
  return { recipe: { ...profile.recipe }, input_ids: [], original_input_ids: [], parent_result_id: null,
    prompt_version_id: null, seed_mode: profile.seed_mode ?? "random", count: 1, auto_improve: false, prompt_help: null };
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
