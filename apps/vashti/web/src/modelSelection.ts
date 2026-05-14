import type { BackendModelGroup, ChatMessage, Persona } from "./types";
import type { PrivatePersona } from "./privateChatStore";

export function compactModelName(name: string) {
  const huggingFaceMatch = /^hf\.co\/[^/]+\/(.+)$/i.exec(name);
  return huggingFaceMatch?.[1] ?? name;
}

export function modelValue(backendId: string, modelName: string) {
  return `${backendId}:${modelName}`;
}

export function enabledModelValueSet(groups: BackendModelGroup[]) {
  return new Set(
    groups.flatMap((group) => group.models.map((model) => modelValue(group.backend.id, model.name)))
  );
}

export function personaBaseModelValue(persona: {
  current_version: { base_backend_id: string; base_model_name: string };
}) {
  return modelValue(
    persona.current_version.base_backend_id,
    persona.current_version.base_model_name
  );
}

export function personaModelValue(personaVersionId: string) {
  return `persona:${personaVersionId}`;
}

export function privatePersonaModelValue(personaVersionId: string) {
  return `private-persona:${personaVersionId}`;
}

export function personaVersionIdFromValue(value: string) {
  return value.startsWith("persona:") ? value.slice("persona:".length) : null;
}

export function privatePersonaVersionIdFromValue(value: string) {
  return value.startsWith("private-persona:")
    ? value.slice("private-persona:".length)
    : null;
}

export function isPrivatePersonaVersionId(value: string) {
  return value.startsWith("private-persona-version-");
}

export function modelParts(value: string) {
  if (personaVersionIdFromValue(value) || privatePersonaVersionIdFromValue(value)) {
    return null;
  }

  const separator = value.indexOf(":");
  if (separator < 1) {
    return null;
  }

  return {
    backendId: value.slice(0, separator),
    modelName: value.slice(separator + 1)
  };
}

export function isLocalBackend(baseUrl: string) {
  try {
    const url = new URL(baseUrl);
    return (
      (url.hostname === "localhost" || url.hostname === "127.0.0.1") &&
      url.port === "11434"
    );
  } catch {
    return false;
  }
}

export function messageModelValue(message: ChatMessage) {
  if (message.role !== "assistant") {
    return null;
  }

  if (message.persona_version_id) {
    return isPrivatePersonaVersionId(message.persona_version_id)
      ? privatePersonaModelValue(message.persona_version_id)
      : personaModelValue(message.persona_version_id);
  }

  if (message.backend_id && message.model_name) {
    return modelValue(message.backend_id, message.model_name);
  }

  return null;
}

export function personaForValue(personas: Persona[], value: string) {
  const personaVersionId = personaVersionIdFromValue(value);
  if (!personaVersionId) {
    return null;
  }

  return personas.find((persona) => persona.current_version.id === personaVersionId) ?? null;
}

export function privatePersonaForValue(personas: PrivatePersona[], value: string) {
  const personaVersionId = privatePersonaVersionIdFromValue(value);
  if (!personaVersionId) {
    return null;
  }

  return privatePersonaForVersionId(personas, personaVersionId);
}

export function privatePersonaForVersionId(
  personas: PrivatePersona[],
  personaVersionId: string | null | undefined
) {
  if (!personaVersionId) {
    return null;
  }

  return personas.find((persona) => persona.current_version.id === personaVersionId) ?? null;
}

export function selectedModelBaseParts(
  groups: BackendModelGroup[],
  personas: Persona[],
  privatePersonas: PrivatePersona[],
  value: string
) {
  const selectedPrivatePersona = privatePersonaForValue(privatePersonas, value);
  if (selectedPrivatePersona) {
    return {
      backendId: selectedPrivatePersona.current_version.base_backend_id,
      modelName: selectedPrivatePersona.current_version.base_model_name
    };
  }

  const selectedPersona = personaForValue(personas, value);
  if (selectedPersona) {
    return {
      backendId: selectedPersona.current_version.base_backend_id,
      modelName: selectedPersona.current_version.base_model_name
    };
  }

  const selected = modelParts(value);
  if (!selected) {
    return null;
  }

  const modelInfo = modelInfoForBase(groups, selected.backendId, selected.modelName);
  return modelInfo ? selected : null;
}

export function modelInfoForValue(
  groups: BackendModelGroup[],
  personas: Persona[],
  privatePersonas: PrivatePersona[],
  value: string
) {
  const selectedPrivatePersona = privatePersonaForValue(privatePersonas, value);
  if (selectedPrivatePersona) {
    return modelInfoForBase(
      groups,
      selectedPrivatePersona.current_version.base_backend_id,
      selectedPrivatePersona.current_version.base_model_name
    );
  }

  const selectedPersona = personaForValue(personas, value);
  if (selectedPersona) {
    return modelInfoForBase(
      groups,
      selectedPersona.current_version.base_backend_id,
      selectedPersona.current_version.base_model_name
    );
  }

  const selected = modelParts(value);
  if (!selected) {
    return null;
  }

  return modelInfoForBase(groups, selected.backendId, selected.modelName);
}

export function modelInfoForBase(groups: BackendModelGroup[], backendId: string, modelName: string) {
  const group = groups.find((modelGroup) => modelGroup.backend.id === backendId);
  return group?.models.find((model) => model.name === modelName) ?? null;
}
