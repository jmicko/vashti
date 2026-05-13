import { useState } from "react";
import { Brain, Image as ImageIcon, Volume2, Wrench } from "lucide-react";

export type ModelCapabilityInfo = {
  supports_images: boolean;
  supports_thinking?: boolean;
  capabilities?: string[];
};

export function ModelCapabilityBadges({ model }: { model: ModelCapabilityInfo }) {
  const capabilities = modelCapabilityBadges(model);
  if (capabilities.length === 0) {
    return null;
  }

  return (
    <span className="model-capabilities" aria-label={`Capabilities: ${capabilities.join(", ")}`}>
      {capabilities.map((capability) => (
        <span key={capability} className="model-capability" title={capability}>
          {capabilityIcon(capability)}
          <span className="model-capability-label">{capabilityLabel(capability)}</span>
        </span>
      ))}
    </span>
  );
}

export function CompactModelCapabilityBadges({
  model,
  hideTools = false
}: {
  model: ModelCapabilityInfo;
  hideTools?: boolean;
}) {
  const [activeCapability, setActiveCapability] = useState<string | null>(null);
  const capabilities = modelCapabilityBadges(model).filter(
    (capability) => !(hideTools && capability === "tools")
  );
  if (capabilities.length === 0) {
    return null;
  }

  return (
    <span
      className="model-capabilities model-capabilities-icon-only"
      aria-label={`Capabilities: ${capabilities.join(", ")}`}
      onMouseLeave={() => setActiveCapability(null)}
    >
      {capabilities.map((capability) => (
        <button
          key={capability}
          type="button"
          className="model-capability model-capability-icon-only"
          title={capabilityLabel(capability)}
          aria-label={capabilityLabel(capability)}
          onBlur={() => window.setTimeout(() => setActiveCapability(null), 120)}
          onClick={() =>
            setActiveCapability((current) => (current === capability ? null : capability))
          }
        >
          {capabilityIcon(capability)}
        </button>
      ))}
      {activeCapability && (
        <span className="model-capability-popover">{capabilityLabel(activeCapability)}</span>
      )}
    </span>
  );
}

export function modelCapabilityBadges(model: ModelCapabilityInfo) {
  const capabilities = new Set(
    (model.capabilities ?? [])
      .map((capability) => capability.trim().toLocaleLowerCase())
      .filter(Boolean)
  );

  if (model.supports_images) {
    capabilities.add("vision");
  }
  if (model.supports_thinking) {
    capabilities.add("thinking");
  }

  capabilities.delete("completion");
  return [...capabilities].sort((left, right) => capabilityOrder(left) - capabilityOrder(right));
}

function capabilityOrder(capability: string) {
  const order = ["vision", "image", "thinking", "tools", "audio"];
  const index = order.indexOf(capability);
  return index === -1 ? order.length : index;
}

function capabilityIcon(capability: string) {
  switch (capability) {
    case "vision":
    case "image":
      return <ImageIcon />;
    case "thinking":
      return <Brain />;
    case "tools":
      return <Wrench />;
    case "audio":
      return <Volume2 />;
    default:
      return null;
  }
}

function capabilityLabel(capability: string) {
  switch (capability) {
    case "vision":
      return "vision";
    case "image":
      return "image";
    case "thinking":
      return "think";
    case "tools":
      return "tools";
    case "audio":
      return "audio";
    default:
      return capability;
  }
}
