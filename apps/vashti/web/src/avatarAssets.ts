import { responseErrorMessage } from "./api";

export type HostedAvatarAsset = {
  id: string;
  original_filename: string;
  mime_type: string;
  size_bytes: number;
  created_at: number;
};

export async function uploadHostedAvatar(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch("/api/persona-avatars", {
    method: "POST",
    credentials: "include",
    body: formData
  });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response));
  }

  return ((await response.json()) as { asset: HostedAvatarAsset }).asset;
}

export async function deleteHostedAvatar(assetId: string) {
  const response = await fetch(`/api/persona-avatars/${encodeURIComponent(assetId)}`, {
    method: "DELETE",
    credentials: "include"
  });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response));
  }
}

export async function hostedAvatarFile(assetId: string) {
  const response = await fetch(`/api/persona-avatars/${encodeURIComponent(assetId)}`, {
    credentials: "include"
  });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response));
  }
  const blob = await response.blob();
  return new File([blob], "profile-image", { type: blob.type });
}
