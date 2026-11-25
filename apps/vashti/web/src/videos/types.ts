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
  last_frame_id?: string | null;
  reference_image_ids?: string[];
  input_mode?: VideoInputMode | null;
};
export type VideoInputMode = "text" | "keyframes" | "references";
export const VIDEO_I2V = "minimax-h3-image-to-video/v1";
export const VIDEO_GENERAL = "minimax-h3-video/v1";
export const VIDEO_REFERENCE = "minimax-h3-reference-video/v1";
export type VideoWorkflow = {
  id: string;
  state: string;
  first_frame: string;
  last_frame?: string;
  max_reference_images?: number;
  steps?: number;
  fps: number;
  frames: number[];
  limits?: string;
  frame_base?: number;
  frame_step?: number;
  max_pixels: number;
  dimension_multiple: number;
  min_dimension: number;
  max_dimension: number;
  audio: boolean;
  output_mime_type: string;
};
export type VideoResolution = "standard" | "high";
const MIN_SIDE = 256, MAX_SIDE = 1280;
const STANDARD_PIXELS = 832 * 544, HIGH_PIXELS = 1280 * 704;
const MIN_FRAMES = 22, STANDARD_MAX_FRAMES = 243, HIGH_MAX_FRAMES = 124;

export function usesVideoMemoryLimits(workflow?: VideoWorkflow): workflow is VideoWorkflow & { frame_base: number; frame_step: number } {
  return workflow?.limits === "memory"
    && typeof workflow.frame_base === "number" && Number.isSafeInteger(workflow.frame_base) && workflow.frame_base > 0
    && typeof workflow.frame_step === "number" && Number.isSafeInteger(workflow.frame_step) && workflow.frame_step > 0;
}

export function videoResolution({ width, height }: Pick<VideoDraft, "width" | "height">): VideoResolution {
  return width * height > STANDARD_PIXELS ? "high" : "standard";
}

export function videoFrameOptions(size: Pick<VideoDraft, "width" | "height">, workflow: VideoWorkflow) {
  const max = videoResolution(size) === "high" ? HIGH_MAX_FRAMES : STANDARD_MAX_FRAMES;
  if (!usesVideoMemoryLimits(workflow)) return workflow.frames.filter((n) => n >= MIN_FRAMES && n <= max);
  const base = workflow.frame_base, step = workflow.frame_step;
  const start = base + Math.max(0, Math.ceil((MIN_FRAMES - base) / step)) * step;
  const frames: number[] = [];
  for (let n = start; n <= max; n += step) frames.push(n);
  return frames;
}

function sizeSupported({ width, height }: Pick<VideoDraft, "width" | "height">, workflow: VideoWorkflow) {
  if (width * height > HIGH_PIXELS || ![width, height].every((side) => side >= MIN_SIDE && side <= MAX_SIDE && side % 32 === 0 && side % workflow.dimension_multiple === 0)) return false;
  return usesVideoMemoryLimits(workflow) || (width * height <= workflow.max_pixels
    && [width, height].every((side) => side >= workflow.min_dimension && side <= workflow.max_dimension));
}

export function fitVideoSettings(draft: Pick<VideoDraft, "width" | "height" | "frames">, workflow: VideoWorkflow) {
  const size = sizeSupported(draft, workflow) ? { width: draft.width, height: draft.height }
    : videoSize(draft.width / draft.height, workflow, videoResolution(draft));
  const options = videoFrameOptions(size, workflow);
  if (!options.length) throw new Error("No supported clip lengths for this video model");
  const frames = options.reduce((best, n) => Math.abs(n - draft.frames) < Math.abs(best - draft.frames) ? n : best);
  return { ...size, frames };
}

export function videoGeometry(draft: Pick<VideoDraft, "width" | "height" | "frames">, ratio: number, workflow: VideoWorkflow, resolution = videoResolution(draft)) {
  return fitVideoSettings({ ...videoSize(ratio, workflow, resolution), frames: draft.frames }, workflow);
}
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
  workflow: VIDEO_GENERAL,
  prompt: "",
  seed: "0",
  seed_mode: "increment",
  width: 832,
  height: 544,
  frames: 56,
  audio: true,
  first_frame_id: null,
  last_frame_id: null,
  reference_image_ids: [],
  input_mode: null,
};
export function videoInputMode(draft: VideoDraft): VideoInputMode {
  return (
    draft.input_mode ??
    (draft.workflow === VIDEO_REFERENCE
      ? "references"
      : draft.first_frame_id ||
          draft.last_frame_id ||
          draft.workflow === VIDEO_I2V
        ? "keyframes"
        : "text")
  );
}
export function videoInputs(draft: VideoDraft) {
  return [
    ...(draft.first_frame_id
      ? [{ id: draft.first_frame_id, label: "Start frame" }]
      : []),
    ...(draft.last_frame_id
      ? [{ id: draft.last_frame_id, label: "End frame" }]
      : []),
    ...(draft.reference_image_ids ?? []).map((id, index) => ({
      id,
      label: `Picture ${index + 1}`,
    })),
  ];
}
export function videoWorkflowName(workflow: string) {
  return workflow === VIDEO_REFERENCE
    ? "MiniMax H3 · References"
    : workflow === VIDEO_I2V
      ? "MiniMax H3 · Image to video"
      : "MiniMax H3";
}
export function videoModeDraft(
  draft: VideoDraft,
  mode: VideoInputMode,
  workflows: VideoWorkflow[],
): VideoDraft {
  const workflow =
    workflows.find(
      (w) =>
        w.state === "ready" &&
        w.id === (mode === "references" ? VIDEO_REFERENCE : VIDEO_GENERAL),
    ) ??
    (mode === "keyframes"
      ? workflows.find((w) => w.state === "ready" && w.id === VIDEO_I2V)
      : undefined);
  if (!workflow)
    throw new Error("This input mode is unavailable on the selected host");
  return {
    ...draft,
    workflow: workflow.id,
    input_mode: mode,
    first_frame_id: mode === "keyframes" ? draft.first_frame_id : null,
    last_frame_id:
      mode === "keyframes" && workflow.last_frame === "optional"
        ? (draft.last_frame_id ?? null)
        : null,
    reference_image_ids:
      mode === "references" ? (draft.reference_image_ids ?? []) : [],
    audio: workflow.audio && draft.audio,
    ...fitVideoSettings(draft, workflow),
  };
}
// Keep first-frame-only flows usable with older Vashti servers as well as old Kuno.
export function wireVideoDraft(draft: VideoDraft) {
  if (
    draft.workflow === VIDEO_I2V &&
    !draft.last_frame_id &&
    !draft.reference_image_ids?.length
  ) {
    const {
      last_frame_id: _last,
      reference_image_ids: _references,
      input_mode: _mode,
      ...legacy
    } = draft;
    return legacy;
  }
  return draft;
}
export function canSendVideo(draft: VideoDraft, workflow?: VideoWorkflow) {
  if (
    !workflow ||
    workflow.state !== "ready" ||
    !sizeSupported(draft, workflow) ||
    !videoFrameOptions(draft, workflow).includes(draft.frames)
  )
    return false;
  const references = draft.reference_image_ids ?? [];
  if (workflow.first_frame === "required" && !draft.first_frame_id)
    return false;
  if (workflow.first_frame === "unsupported" && draft.first_frame_id)
    return false;
  if (draft.last_frame_id && workflow.last_frame !== "optional") return false;
  if (references.length > (workflow.max_reference_images ?? 0)) return false;
  if (
    draft.workflow === VIDEO_REFERENCE &&
    (!references.length || draft.first_frame_id || draft.last_frame_id)
  )
    return false;
  return !!draft.prompt.trim() || videoInputs(draft).length > 0;
}
export function closestVideoAspect(
  width: number,
  height: number,
  workflow: VideoWorkflow,
  ratios: string[],
) {
  return (
    ratios.find((ratio) => {
      const [w, h] = ratio.split(":").map(Number);
      const size = videoSize(w / h, workflow, videoResolution({ width, height }));
      return size.width === width && size.height === height;
    }) ?? "Custom"
  );
}
export const videoActive = (status: string) =>
  ["queued", "generating", "saving"].includes(status);

export function videoSize(ratio: number, workflow: VideoWorkflow, resolution: VideoResolution = "standard") {
  if (!Number.isFinite(ratio) || ratio <= 0)
    throw new Error("Invalid aspect ratio");
  const memory = usesVideoMemoryLimits(workflow);
  const high = memory && resolution === "high";
  const min = memory ? MIN_SIDE : workflow.min_dimension;
  const max = memory ? MAX_SIDE : workflow.max_dimension;
  const budget = memory ? (high ? HIGH_PIXELS : STANDARD_PIXELS) : Math.min(workflow.max_pixels, HIGH_PIXELS);
  const multiple = workflow.dimension_multiple;
  if (multiple !== 32 || min < MIN_SIDE || max > MAX_SIDE || min > max || min % multiple !== 0 || !Number.isFinite(budget))
    throw new Error("Unsupported video model dimensions");
  let best = { width: min, height: min }, score = Infinity;
  for (let width = min; width <= max; width += multiple)
    for (let height = min; height <= max; height += multiple) {
      const pixels = width * height;
      if (pixels > budget || (high && pixels <= STANDARD_PIXELS)) continue;
      const rank =
        Math.abs(Math.log(width / height / ratio)) * 10 +
        (1 - pixels / budget) * (high ? 1 : 0.15);
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
