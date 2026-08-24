import { requestJson } from "../api";
import type {
  Memory,
  MemoryModelScope,
  MemorySettings,
  MemorySummary,
  MemoryVersion
} from "./types";

type MemoryMutationResponse = { memory: Memory };

export async function listMemories({
  query = "",
  status = "active",
  limit = 100,
  offset = 0
}: {
  query?: string;
  status?: "active" | "forgotten" | "all";
  limit?: number;
  offset?: number;
} = {}) {
  const params = new URLSearchParams({
    status,
    limit: String(limit),
    offset: String(offset)
  });
  if (query.trim()) params.set("query", query.trim());
  return requestJson<{ memories: MemorySummary[]; total: number }>(`/api/memories?${params}`);
}

export async function getMemory(memoryId: string) {
  return (await requestJson<MemoryMutationResponse>(`/api/memories/${memoryId}`)).memory;
}

export async function createMemory(content: string, modelScope: MemoryModelScope) {
  return (
    await requestJson<MemoryMutationResponse>("/api/memories", {
      method: "POST",
      body: JSON.stringify({ content, model_scope: modelScope })
    })
  ).memory;
}

export async function updateMemory(
  memoryId: string,
  expectedVersion: number,
  content: string,
  modelScope: MemoryModelScope
) {
  return (
    await requestJson<MemoryMutationResponse>(`/api/memories/${memoryId}`, {
      method: "PATCH",
      body: JSON.stringify({
        expected_version: expectedVersion,
        content,
        model_scope: modelScope
      })
    })
  ).memory;
}

export async function forgetMemory(memoryId: string, expectedVersion: number) {
  return versionedMutation(`/api/memories/${memoryId}/forget`, expectedVersion);
}

export async function restoreMemory(memoryId: string, expectedVersion: number) {
  return versionedMutation(`/api/memories/${memoryId}/restore`, expectedVersion);
}

export async function permanentlyDeleteMemory(memoryId: string) {
  return requestJson<{ ok: boolean }>(`/api/memories/${memoryId}`, { method: "DELETE" });
}

export async function listMemoryVersions(memoryId: string) {
  return (
    await requestJson<{ versions: MemoryVersion[] }>(`/api/memories/${memoryId}/versions`)
  ).versions;
}

export async function restoreMemoryVersion(
  memoryId: string,
  versionId: string,
  expectedVersion: number
) {
  return versionedMutation(
    `/api/memories/${memoryId}/versions/${versionId}/restore`,
    expectedVersion
  );
}

export async function getMemorySettings() {
  return requestJson<MemorySettings>("/api/memories/settings");
}

export async function updateMemorySettings(settings: MemorySettings) {
  return requestJson<MemorySettings>("/api/memories/settings", {
    method: "PATCH",
    body: JSON.stringify(settings)
  });
}

async function versionedMutation(path: string, expectedVersion: number) {
  return (
    await requestJson<MemoryMutationResponse>(path, {
      method: "POST",
      body: JSON.stringify({ expected_version: expectedVersion })
    })
  ).memory;
}
