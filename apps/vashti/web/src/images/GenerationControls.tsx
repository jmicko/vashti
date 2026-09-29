import { useState } from "react";
import { Scan } from "lucide-react";
import { ImageAsset } from "./ImageAsset";
import { NumberField } from "./controls";
import { aspectRatios, aspectSize, closestAspect } from "./aspectRatio";
import type { Draft, Recipe } from "./types";

export const imageControls = [["negative_prompt", "Negative prompt"], ["size", "Image size"], ["steps", "Steps"], ["cfg", "CFG"], ["seed", "Seed"], ["count", "Image count"], ["attention", "Attention"]] as const;

export function GenerationControls({ draft, hidden, patchRecipe, patchDraft }: {
  draft: Draft; hidden: string[]; patchRecipe: (patch: Partial<Recipe>) => void; patchDraft: (patch: Partial<Draft>) => void;
}) {
  const [source, setSource] = useState<{ id: string; width: number; height: number } | null>(null);
  const original = draft.original_input_ids[0];
  const recipe = draft.recipe;
  function chooseRatio(width: number, height: number) {
    patchRecipe(aspectSize(width, height, recipe.width * recipe.height));
  }
  return <div className="images-controls">
    {!hidden.includes("size") && <fieldset className="images-wide images-size">
      <legend>Aspect ratio <span>{closestAspect(recipe.width, recipe.height)} · {recipe.width} × {recipe.height}</span></legend>
      <div className="images-ratios">{aspectRatios.map(ratio => {
        const [width, height] = ratio.split(":").map(Number);
        return <button type="button" key={ratio} aria-pressed={closestAspect(recipe.width, recipe.height) === ratio}
          onClick={() => chooseRatio(width, height)}><span className="images-ratio-shape" style={{ width: 24 * Math.min(1, width / height), height: 24 * Math.min(1, height / width) }} />{ratio}</button>;
      })}</div>
      {original && <div className="images-original-size">
        <ImageAsset id={original} alt="Original reference" onDimensions={size => setSource({ id: original, ...size })} />
        <button type="button" disabled={source?.id !== original} onClick={() => { if (source?.id === original) chooseRatio(source.width, source.height); }}><Scan size={18} />Match original</button>
        {source?.id === original && <small>{source.width} × {source.height}</small>}
      </div>}
      <div className="images-controls"><NumberField label="Width" min={64} max={2048} step={16} value={recipe.width} onChange={width => patchRecipe({ width })} />
        <NumberField label="Height" min={64} max={2048} step={16} value={recipe.height} onChange={height => patchRecipe({ height })} /></div>
    </fieldset>}
    {!hidden.includes("negative_prompt") && <label className="images-wide">Negative prompt<textarea rows={2} value={recipe.negative_prompt} onChange={e => patchRecipe({ negative_prompt: e.target.value })} /></label>}
    {!hidden.includes("steps") && <NumberField label="Steps" min={1} max={200} value={recipe.steps} onChange={steps => patchRecipe({ steps })} />}
    {!hidden.includes("cfg") && <NumberField label="CFG" min={0} max={30} step={0.1} value={recipe.cfg} onChange={cfg => patchRecipe({ cfg })} />}
    {recipe.flow_shift != null && <NumberField label="Flow shift" min={0.1} max={10} step={0.1} value={recipe.flow_shift} onChange={flow_shift => patchRecipe({ flow_shift })} />}
    {!hidden.includes("seed") && <><fieldset className="images-wide"><legend>Seed</legend><div className="images-segments">{(["random", "increment", "fixed"] as const).map(value => <button type="button" key={value} aria-pressed={draft.seed_mode === value} onClick={() => patchDraft({ seed_mode: value })}>{value[0].toUpperCase() + value.slice(1)}</button>)}</div></fieldset>
      {draft.seed_mode !== "random" && <label className="images-wide">Starting seed<input inputMode="numeric" value={recipe.seed} onChange={e => patchRecipe({ seed: e.target.value })} /></label>}</>}
    {!hidden.includes("count") && <><label className="images-check"><input type="checkbox" checked={draft.count === null} onChange={e => patchDraft({ count: e.target.checked ? null : 1 })} />Until stopped</label>{draft.count !== null && <NumberField label="Images" min={1} max={100} value={draft.count} onChange={count => patchDraft({ count })} />}</>}
    {!hidden.includes("attention") && <fieldset className="images-wide"><legend>Attention</legend><div className="images-segments">{(["standard", "fast"] as const).map(attention => <button type="button" key={attention} aria-pressed={recipe.attention === attention} title={attention === "fast" ? "Qwen 2.1 on CUDA" : "Standard attention"} onClick={() => patchRecipe({ attention })}>{attention === "fast" ? "Fast" : "Standard"}</button>)}</div></fieldset>}
  </div>;
}
