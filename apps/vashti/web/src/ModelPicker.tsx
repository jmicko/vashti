import { useEffect, useRef, useState } from "react";
import { Brain, Lock, Search, Star, Users } from "lucide-react";
import { ModelCapabilityBadges } from "./modelCapabilities";
import {
  compactModelName,
  enabledModelValueSet,
  modelInfoForBase,
  modelValue,
  personaBaseModelValue,
  personaForValue,
  personaVersionForValue,
  personaModelValue,
  privatePersonaForValue,
  privatePersonaVersionForValue,
  privatePersonaModelValue
} from "./modelSelection";
import type { PrivatePersona, PrivatePersonaVersion } from "./privateChatStore";
import type { BackendModelGroup, ModelInfo, Persona, PersonaVersion } from "./types";

export function ModelPicker({
  groups,
  personas,
  privatePersonas,
  personaVersions = [],
  privatePersonaVersions = [],
  isLoading,
  error,
  value,
  onChange
}: {
  groups: BackendModelGroup[];
  personas: Persona[];
  privatePersonas: PrivatePersona[];
  personaVersions?: PersonaVersion[];
  privatePersonaVersions?: PrivatePersonaVersion[];
  isLoading: boolean;
  error: string | null;
  value: string;
  onChange: (value: string) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const wrapRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const hasModels =
    groups.some((group) => group.models.length > 0) ||
    personas.length > 0 ||
    privatePersonas.length > 0;
  const selected = groups
    .flatMap((group) =>
      group.models.map((model) => ({
        backendId: group.backend.id,
        backendName: group.backend.name,
        model
      }))
    )
    .find((option) => modelValue(option.backendId, option.model.name) === value);
  const selectedPersona = personaForValue(personas, value);
  const selectedPrivatePersona = privatePersonaForValue(privatePersonas, value);
  const selectedPersonaVersion = personaVersionForValue(personas, personaVersions, value);
  const selectedPrivatePersonaVersion = privatePersonaVersionForValue(
    privatePersonas,
    privatePersonaVersions,
    value
  );
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const enabledValues = enabledModelValueSet(groups);
  const favoriteModels = groups
    .flatMap((group) =>
      group.models
        .filter((model) => model.is_favorite)
        .map((model) => ({
          backend: group.backend,
          model
        }))
    )
    .filter((option) => {
      if (!normalizedQuery) {
        return true;
      }

      return (
        option.model.name.toLocaleLowerCase().includes(normalizedQuery) ||
        compactModelName(option.model.name).toLocaleLowerCase().includes(normalizedQuery) ||
        option.backend.name.toLocaleLowerCase().includes(normalizedQuery)
      );
    });
  const filteredGroups = groups
    .map((group) => ({
      backend: group.backend,
      models: group.models.filter((model) => {
        if (!normalizedQuery) {
          return true;
        }

        return (
          model.name.toLocaleLowerCase().includes(normalizedQuery) ||
          compactModelName(model.name).toLocaleLowerCase().includes(normalizedQuery) ||
          group.backend.name.toLocaleLowerCase().includes(normalizedQuery)
        );
      }),
      personas: personas.filter((persona) => {
        if (!enabledValues.has(personaBaseModelValue(persona))) {
          return false;
        }
        if (persona.current_version.base_backend_id !== group.backend.id) {
          return false;
        }
        if (!normalizedQuery) {
          return true;
        }

        return (
          persona.current_version.display_name.toLocaleLowerCase().includes(normalizedQuery) ||
          persona.current_version.base_model_name.toLocaleLowerCase().includes(normalizedQuery) ||
          compactModelName(persona.current_version.base_model_name)
            .toLocaleLowerCase()
            .includes(normalizedQuery) ||
          group.backend.name.toLocaleLowerCase().includes(normalizedQuery) ||
          (persona.owner_username ?? "").toLocaleLowerCase().includes(normalizedQuery)
        );
      }),
      privatePersonas: privatePersonas.filter((persona) => {
        if (!enabledValues.has(personaBaseModelValue(persona))) {
          return false;
        }
        if (persona.current_version.base_backend_id !== group.backend.id) {
          return false;
        }
        if (!normalizedQuery) {
          return true;
        }

        return (
          persona.current_version.display_name.toLocaleLowerCase().includes(normalizedQuery) ||
          persona.current_version.base_model_name.toLocaleLowerCase().includes(normalizedQuery) ||
          compactModelName(persona.current_version.base_model_name)
            .toLocaleLowerCase()
            .includes(normalizedQuery) ||
          group.backend.name.toLocaleLowerCase().includes(normalizedQuery)
        );
      })
    }))
    .filter(
      (group) =>
        group.models.length > 0 || group.personas.length > 0 || group.privatePersonas.length > 0
    );

  useEffect(() => {
    if (isOpen) {
      window.setTimeout(() => inputRef.current?.focus(), 0);
    } else {
      setQuery("");
    }
  }, [isOpen]);

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

  function buttonLabel() {
    if (isLoading) {
      return "Loading models...";
    }

    if (error) {
      return "Models unavailable";
    }

    if (!hasModels) {
      return "No models";
    }

    return selectedPrivatePersona
      ? selectedPrivatePersona.current_version.display_name
      : selectedPrivatePersonaVersion
        ? selectedPrivatePersonaVersion.display_name
      : selectedPersona
        ? selectedPersona.current_version.display_name
        : selectedPersonaVersion
          ? selectedPersonaVersion.display_name
        : selected
          ? compactModelName(selected.model.name)
          : "Select model";
  }

  function renderModelOption(
    backendId: string,
    backendName: string,
    model: ModelInfo,
    keyPrefix = "model"
  ) {
    const optionValue = modelValue(backendId, model.name);
    return (
      <button
        type="button"
        key={`${keyPrefix}:${optionValue}`}
        className={optionValue === value ? "model-option model-option-active" : "model-option"}
        title={model.name}
        onClick={() => {
          onChange(optionValue);
          setIsOpen(false);
        }}
      >
        <span className="model-option-content">
          <span className="model-option-title-row">
            <span className="model-name">{compactModelName(model.name)}</span>
            {model.is_favorite && (
              <span className="model-option-star" title="Favorite">
                <Star />
              </span>
            )}
          </span>
          {keyPrefix === "favorite" && <span className="model-subtitle">{backendName}</span>}
          <ModelCapabilityBadges model={model} />
        </span>
      </button>
    );
  }

  return (
    <div className="model-picker" ref={wrapRef}>
      <button
        type="button"
        className="model-picker-button"
        disabled={isLoading || !hasModels}
        title={
          error ??
          selectedPrivatePersona?.current_version.base_model_name ??
          selectedPrivatePersonaVersion?.base_model_name ??
          selectedPersona?.current_version.base_model_name ??
          selectedPersonaVersion?.base_model_name ??
          selected?.model.name ??
          selected?.backendName
        }
        aria-expanded={isOpen}
        onClick={() => setIsOpen((open) => !open)}
      >
        <span className="model-name">{buttonLabel()}</span>
      </button>
      {isOpen && (
        <div className="model-menu">
          <label className="model-search">
            <Search />
            <input
              ref={inputRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  setIsOpen(false);
                }
              }}
              placeholder="Search models"
            />
          </label>
          <div className="model-options">
            {filteredGroups.length === 0 && favoriteModels.length === 0 ? (
              <p className="model-empty">No matching models</p>
            ) : (
              <>
                {favoriteModels.length > 0 && (
                  <section className="model-group model-group-favorites">
                    <p>
                      <Star />
                      <span>Favorites</span>
                    </p>
                    {favoriteModels.map((option) =>
                      renderModelOption(
                        option.backend.id,
                        option.backend.name,
                        option.model,
                        "favorite"
                      )
                    )}
                  </section>
                )}
                {filteredGroups.map((group) => (
                  <section key={group.backend.id} className="model-group">
                    <p>{group.backend.name}</p>
                    {group.privatePersonas.map((persona) => {
                      const optionValue = privatePersonaModelValue(persona.current_version.id);
                      const baseModel = modelInfoForBase(
                        groups,
                        persona.current_version.base_backend_id,
                        persona.current_version.base_model_name
                      );
                      return (
                        <button
                          type="button"
                          key={optionValue}
                          className={
                            optionValue === value
                              ? "model-option model-option-active"
                              : "model-option"
                          }
                          onClick={() => {
                            onChange(optionValue);
                            setIsOpen(false);
                          }}
                        >
                          <span className="model-option-content">
                            <span className="model-name">
                              {persona.current_version.display_name}
                            </span>
                            <span className="model-subtitle">
                              Device · {compactModelName(persona.current_version.base_model_name)}
                            </span>
                            <span className="model-capabilities">
                              <span className="model-capability" title="custom persona">
                                <Brain />
                                <span className="model-capability-label">custom</span>
                              </span>
                              <span
                                className="model-capability model-capability-warning"
                                title="device only"
                              >
                                <Lock />
                                <span className="model-capability-label">device</span>
                              </span>
                            </span>
                            {baseModel && <ModelCapabilityBadges model={baseModel} />}
                          </span>
                        </button>
                      );
                    })}
                    {group.personas.map((persona) => {
                      const optionValue = personaModelValue(persona.current_version.id);
                      const baseModel = modelInfoForBase(
                        groups,
                        persona.current_version.base_backend_id,
                        persona.current_version.base_model_name
                      );
                      return (
                        <button
                          type="button"
                          key={optionValue}
                          className={
                            optionValue === value
                              ? "model-option model-option-active"
                              : "model-option"
                          }
                          onClick={() => {
                            onChange(optionValue);
                            setIsOpen(false);
                          }}
                        >
                          <span className="model-option-content">
                            <span className="model-name">
                              {persona.current_version.display_name}
                            </span>
                            <span className="model-subtitle">
                              Custom · {compactModelName(persona.current_version.base_model_name)}
                              {persona.owner_username ? ` · by ${persona.owner_username}` : ""}
                            </span>
                            <span className="model-capabilities">
                              <span className="model-capability" title="custom persona">
                                <Brain />
                                <span className="model-capability-label">custom</span>
                              </span>
                              <span
                                className={
                                  persona.visibility === "public"
                                    ? "model-capability"
                                    : "model-capability model-capability-warning"
                                }
                                title={persona.visibility}
                              >
                                {persona.visibility === "public" ? <Users /> : <Lock />}
                                <span className="model-capability-label">
                                  {persona.visibility}
                                </span>
                              </span>
                            </span>
                            {baseModel && <ModelCapabilityBadges model={baseModel} />}
                          </span>
                        </button>
                      );
                    })}
                    {group.models.map((model) =>
                      renderModelOption(group.backend.id, group.backend.name, model)
                    )}
                  </section>
                ))}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
