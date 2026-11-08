export type VideoDraft = {
  backend_id: string;
  workflow: string;
  prompt: string;
  seed: string;
  seed_mode: "increment" | "random" | "fixed";
  width: number;
  height: number;
  frames: number;
  audio: boolean;
  first_frame_id: string | null;
};
export type VideoWorkflow = {
  id: string;
  state: string;
  first_frame: string;
  fps: number;
  frames: number[];
  max_pixels: number;
  dimension_multiple: number;
  min_dimension: number;
  max_dimension: number;
  audio: boolean;
  output_mime_type: string;
};
export type VideoHost = {
  backend: { id: string; name: string };
  capabilities: { state: string; workflows: VideoWorkflow[] };
};
export type VideoSession = {
  id: string;
  title: string;
  draft_json: string;
  revision: number;
  updated_at: number;
};
export type VideoRun = {
  id: string;
  session_id: string;
  group_id: string;
  draft_json: string;
  status: string;
  error: string | null;
  size_bytes: number;
  duration_ms: number;
  created_at: number;
};
export type VideoPage = { session: VideoSession; runs: VideoRun[] };
export type VideoStatus = {
  status: string;
  progress: {
    phase?: string;
    step?: number;
    steps?: number;
    preview?: { png_base64: string; width: number; height: number };
    error?: string;
  };
};
export const videoDefaults: VideoDraft = {
  backend_id: "",
  workflow: "minimax-h3-image-to-video/v1",
  prompt: "",
  seed: "0",
  seed_mode: "increment",
  width: 832,
  height: 544,
  frames: 56,
  audio: true,
  first_frame_id: null,
};
export const videoActive = (status: string) =>
  ["queued", "generating", "saving"].includes(status);

export function videoSize(ratio: number, workflow: VideoWorkflow) {
  if (!Number.isFinite(ratio) || ratio <= 0)
    throw new Error("Invalid aspect ratio");
  if (
    workflow.dimension_multiple !== 32 ||
    workflow.min_dimension < 256 ||
    workflow.max_dimension > 1280 ||
    workflow.min_dimension > workflow.max_dimension ||
    workflow.min_dimension % 32 !== 0 ||
    !Number.isFinite(workflow.max_pixels)
  )
    throw new Error("Unsupported video model dimensions");
  let best = { width: workflow.min_dimension, height: workflow.min_dimension },
    score = Infinity;
  for (
    let width = workflow.min_dimension;
    width <= workflow.max_dimension;
    width += workflow.dimension_multiple
  )
    for (
      let height = workflow.min_dimension;
      height <= workflow.max_dimension;
      height += workflow.dimension_multiple
    ) {
      const pixels = width * height;
      if (pixels > workflow.max_pixels) continue;
      const rank =
        Math.abs(Math.log(width / height / ratio)) * 10 +
        (1 - pixels / workflow.max_pixels) * 0.15;
      if (rank < score) {
        score = rank;
        best = { width, height };
      }
    }
  if (!Number.isFinite(score))
    throw new Error("No supported size for this video model");
  return best;
}
export function nextVideoSeed(draft: VideoDraft, index = 1): string {
  if (draft.seed_mode === "fixed") return draft.seed;
  if (draft.seed_mode === "random") {
    const a = crypto.getRandomValues(new Uint32Array(2));
    return ((BigInt(a[0] & 0x7fffffff) << 32n) | BigInt(a[1])).toString();
  }
  const n = BigInt(draft.seed) + BigInt(index);
  if (n > 9223372036854775807n)
    throw new Error("Seed limit reached. Choose another seed.");
  return n.toString();
}
