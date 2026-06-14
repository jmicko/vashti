import type { BackendModelGroup, ChatMessage, Persona, PersonaVersion } from "./types";
import type { PrivatePersona, PrivatePersonaVersion } from "./privateChatStore";

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

export function personaVersionForId(
  personas: Persona[],
  personaVersions: PersonaVersion[],
  personaVersionId: string | null | undefined
) {
  if (!personaVersionId) {
    return null;
  }

  return (
    personas.find((persona) => persona.current_version.id === personaVersionId)?.current_version ??
    personaVersions.find((version) => version.id === personaVersionId) ??
    null
  );
}

export function personaVersionForValue(
  personas: Persona[],
  personaVersions: PersonaVersion[],
  value: string
) {
  return personaVersionForId(personas, personaVersions, personaVersionIdFromValue(value));
}

export function privatePersonaForValue(personas: PrivatePersona[], value: string) {
  const personaVersionId = privatePersonaVersionIdFromValue(value);
  if (!personaVersionId) {
    return null;
  }

  return privatePersonaForVersionId(personas, personaVersionId);
}

export function privatePersonaVersionForId(
  personas: PrivatePersona[],
  personaVersions: PrivatePersonaVersion[],
  personaVersionId: string | null | undefined
) {
  if (!personaVersionId) {
    return null;
  }

  return (
    personas.find((persona) => persona.current_version.id === personaVersionId)?.current_version ??
    personaVersions.find((version) => version.id === personaVersionId) ??
    null
  );
}

export function privatePersonaVersionForValue(
  personas: PrivatePersona[],
  personaVersions: PrivatePersonaVersion[],
  value: string
) {
  return privatePersonaVersionForId(
    personas,
    personaVersions,
    privatePersonaVersionIdFromValue(value)
  );
}

export function privatePersonaWithVersionForId(
  personas: PrivatePersona[],
  personaVersions: PrivatePersonaVersion[],
  personaVersionId: string | null | undefined
): PrivatePersona | null {
  const version = privatePersonaVersionForId(personas, personaVersions, personaVersionId);
  if (!version) {
    return null;
  }

  const persona = personas.find((candidate) => candidate.id === version.persona_id);
  return persona ? { ...persona, current_version: version } : null;
}

export function privatePersonaWithVersionForValue(
  personas: PrivatePersona[],
  personaVersions: PrivatePersonaVersion[],
  value: string
) {
  return privatePersonaWithVersionForId(
    personas,
    personaVersions,
    privatePersonaVersionIdFromValue(value)
  );
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
  value: string,
  personaVersions: PersonaVersion[] = [],
  privatePersonaVersions: PrivatePersonaVersion[] = []
) {
  const selectedPrivatePersonaVersion = privatePersonaVersionForValue(
    privatePersonas,
    privatePersonaVersions,
    value
  );
  if (selectedPrivatePersonaVersion) {
    return {
      backendId: selectedPrivatePersonaVersion.base_backend_id,
      modelName: selectedPrivatePersonaVersion.base_model_name
    };
  }

  const selectedPersonaVersion = personaVersionForValue(personas, personaVersions, value);
  if (selectedPersonaVersion) {
    return {
      backendId: selectedPersonaVersion.base_backend_id,
      modelName: selectedPersonaVersion.base_model_name
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
  value: string,
  personaVersions: PersonaVersion[] = [],
  privatePersonaVersions: PrivatePersonaVersion[] = []
) {
  const selectedPrivatePersonaVersion = privatePersonaVersionForValue(
    privatePersonas,
    privatePersonaVersions,
    value
  );
  if (selectedPrivatePersonaVersion) {
    const base = modelInfoForBase(
      groups,
      selectedPrivatePersonaVersion.base_backend_id,
      selectedPrivatePersonaVersion.base_model_name
    );
    return personaModelInfo(base, selectedPrivatePersonaVersion, true);
  }

  const selectedPersonaVersion = personaVersionForValue(personas, personaVersions, value);
  if (selectedPersonaVersion) {
    const base = modelInfoForBase(
      groups,
      selectedPersonaVersion.base_backend_id,
      selectedPersonaVersion.base_model_name
    );
    return personaModelInfo(base, selectedPersonaVersion, false);
  }

  const selected = modelParts(value);
  if (!selected) {
    return null;
  }

  return modelInfoForBase(groups, selected.backendId, selected.modelName);
}

function personaModelInfo(
  base: ReturnType<typeof modelInfoForBase>,
  version: PersonaVersion | PrivatePersonaVersion,
  isPrivate: boolean
) {
  if (!base || !version.background_asset_id) return base;
  return {
    ...base,
    background_asset_id: version.background_asset_id,
    background_dim: version.background_dim ?? 0.72,
    background_message_dim: version.background_message_dim ?? 0.82,
    background_landscape_mode: version.background_landscape_mode ?? "fill",
    background_landscape_x: version.background_landscape_x ?? 50,
    background_landscape_y: version.background_landscape_y ?? 50,
    background_landscape_scale: version.background_landscape_scale ?? 35,
    background_portrait_mode: version.background_portrait_mode ?? "fill",
    background_portrait_x: version.background_portrait_x ?? 50,
    background_portrait_y: version.background_portrait_y ?? 50,
    background_portrait_scale: version.background_portrait_scale ?? 35,
    background_is_private: isPrivate
  };
}

export function modelInfoForBase(groups: BackendModelGroup[], backendId: string, modelName: string) {
  const group = groups.find((modelGroup) => modelGroup.backend.id === backendId);
  return group?.models.find((model) => model.name === modelName) ?? null;
}
