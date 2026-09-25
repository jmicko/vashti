import { useEffect, useState } from "react";
import { requestJson } from "./api";

type Recipe = {
  workflow: string;
  prompt: string;
  negative_prompt: string;
  seed: string;
  width: number;
  height: number;
  steps: number;
  cfg: number;
  flow_shift?: number | null;
  reference_images?: string[];
};
type Imported = { recipe: Recipe; selected_output: string; warnings: string[]; required_reference_images: number };

export function ImageGenerationSettings({ attachmentId }: { attachmentId: string }) {
  const [open, setOpen] = useState(false);
  const [result, setResult] = useState<Imported | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open) return;
    let current = true;
    setLoading(true);
    setError("");
    setResult(null);
    requestJson<Imported>(`/api/attachments/${encodeURIComponent(attachmentId)}/image-settings`)
      .then(value => {
        if (current) setResult(value);
      })
      .catch(reason => {
        if (current) {
          setError(reason instanceof Error ? reason.message : "Could not read image settings.");
        }
      })
      .finally(() => {
        if (current) setLoading(false);
      });
    return () => { current = false; };
  }, [attachmentId, open]);

  function download() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result.recipe, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "qwen-image-settings.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <div className="image-generation-settings" onClick={event => event.stopPropagation()}>
      <button type="button" aria-expanded={open} onClick={() => setOpen(value => !value)}>
        Generation settings
      </button>
      {open && (
        <section className="image-generation-settings-panel" aria-label="Image generation settings" aria-live="polite">
          {loading && <p>Reading image settings…</p>}
          {error && <p role="status">{error}</p>}
          {result && <>
            <h3>{result.recipe.workflow.includes("2.1") ? "Qwen Image 2.1" : "Qwen Image 2512"}</h3>
            <dl>
              <dt>Size</dt><dd>{result.recipe.width} × {result.recipe.height}</dd>
              <dt>Steps</dt><dd>{result.recipe.steps}</dd>
              <dt>CFG</dt><dd>{result.recipe.cfg}</dd>
              <dt>Flow shift</dt><dd>{result.recipe.flow_shift == null ? "Automatic, based on resolution" : Number(result.recipe.flow_shift.toPrecision(8))}</dd>
              <dt>Sampler</dt><dd>Euler / simple</dd>
              <dt>Seed</dt><dd>{result.recipe.seed}</dd>
            </dl>
            <label>Prompt<textarea readOnly value={result.recipe.prompt} rows={4} /></label>
            <label>Negative prompt<textarea readOnly value={result.recipe.negative_prompt} rows={3} /></label>
            <p className="muted">
              Settings are from the image. Original model hashes are unavailable;
              another engine may produce different pixels.
            </p>
            {result.required_reference_images > 0 && <p>
              This edit needs {result.required_reference_images} original reference {result.required_reference_images === 1 ? "image" : "images"}, in the same order.
              They are not embedded in this output; select them again when generating.
            </p>}
            {result.warnings.includes("native_output_is_8_bit") && <p>The source used 16-bit output. Kuno currently returns 8-bit PNGs, including transparency.</p>}
            {result.warnings.includes("native_dimensions_require_multiple_of_32") && <p>These dimensions are preserved for reference. Native Qwen 2.1 generation requires both dimensions to be divisible by 32.</p>}
            <button type="button" onClick={download}>Download settings</button>
          </>}
        </section>
      )}
    </div>
  );
}
