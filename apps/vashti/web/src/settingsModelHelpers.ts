import { modelValue } from "./modelSelection";
import type {
  AdminBackendModelGroup,
  AdminModelInfo,
  AdminModelTagPatchPayload,
  BackendModelGroup,
  PermissionTag
} from "./types";

export function firstModelValue(groups: BackendModelGroup[]) {
  const firstGroup = groups.find((group) => group.models.length > 0);
  const firstModel = firstGroup?.models[0];
  return firstGroup && firstModel ? modelValue(firstGroup.backend.id, firstModel.name) : "";
}

export function backendNameFor(groups: BackendModelGroup[], backendId: string) {
  return groups.find((group) => group.backend.id === backendId)?.backend.name ?? backendId;
}

export function permissionTagPayload(tags: PermissionTag[]) {
  return tags.map((tag) => tag.id);
}

export function sortedPermissionTagPayload(tags: PermissionTag[]) {
  return permissionTagPayload(tags).slice().sort();
}

export function permissionTagSetsEqual(left: PermissionTag[], right: PermissionTag[]) {
  const leftIds = sortedPermissionTagPayload(left);
  const rightIds = sortedPermissionTagPayload(right);
  if (leftIds.length !== rightIds.length) {
    return false;
  }

  return leftIds.every((id, index) => id === rightIds[index]);
}

export function updateAdminModelTagsInGroups(
  groups: AdminBackendModelGroup[],
  backendId: string,
  modelName: string,
  patch: Partial<Pick<AdminModelInfo, "permission_tags" | "default_permission_tags">>
) {
  return groups.map((group) =>
    group.backend.id === backendId
      ? {
          ...group,
          models: group.models.map((model) =>
            model.name === modelName ? { ...model, ...patch } : model
          )
        }
      : group
  );
}

function modelTagSnapshot(groups: AdminBackendModelGroup[]) {
  return groups
    .flatMap((group) =>
      group.models.map((model) => ({
        key: `${group.backend.id}\u0000${model.name}`,
        permissionTags: sortedPermissionTagPayload(model.permission_tags),
        defaultPermissionTags: sortedPermissionTagPayload(model.default_permission_tags)
      }))
    )
    .sort((left, right) => left.key.localeCompare(right.key));
}

export function adminModelGroupTagsEqual(
  leftGroups: AdminBackendModelGroup[],
  rightGroups: AdminBackendModelGroup[]
) {
  const left = modelTagSnapshot(leftGroups);
  const right = modelTagSnapshot(rightGroups);
  return JSON.stringify(left) === JSON.stringify(right);
}

export function collectChangedAdminModelTagPatches(
  groups: AdminBackendModelGroup[],
  savedGroups: AdminBackendModelGroup[],
  includeDefaultTags: boolean
) {
  const savedModels = new Map<string, AdminModelInfo>();
  for (const group of savedGroups) {
    for (const model of group.models) {
      savedModels.set(`${group.backend.id}\u0000${model.name}`, model);
    }
  }

  const patches: AdminModelTagPatchPayload[] = [];
  for (const group of groups) {
    for (const model of group.models) {
      const savedModel = savedModels.get(`${group.backend.id}\u0000${model.name}`);
      if (!savedModel) {
        continue;
      }

      const patch: AdminModelTagPatchPayload = {
        backend_id: group.backend.id,
        model_name: model.name
      };
      let hasChanges = false;

      if (!permissionTagSetsEqual(model.permission_tags, savedModel.permission_tags)) {
        patch.permission_tags = permissionTagPayload(model.permission_tags);
        hasChanges = true;
      }

      if (
        includeDefaultTags &&
        !permissionTagSetsEqual(model.default_permission_tags, savedModel.default_permission_tags)
      ) {
        patch.default_permission_tags = permissionTagPayload(model.default_permission_tags);
        hasChanges = true;
      }

      if (hasChanges) {
        patches.push(patch);
      }
    }
  }

  return patches;
}
