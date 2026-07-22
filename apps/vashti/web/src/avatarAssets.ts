import { requestBlob, requestJson, requestMultipartJson } from "./api";

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
  const response = await requestMultipartJson<{ asset: HostedAvatarAsset }>(
    "/api/persona-avatars",
    formData,
    {
    method: "POST",
    }
  );
  return response.asset;
}

export async function deleteHostedAvatar(assetId: string) {
  await requestJson(`/api/persona-avatars/${encodeURIComponent(assetId)}`, {
    method: "DELETE"
  });
}

export async function hostedAvatarFile(assetId: string) {
  const blob = await requestBlob(`/api/persona-avatars/${encodeURIComponent(assetId)}`);
  return new File([blob], "profile-image", { type: blob.type });
}
