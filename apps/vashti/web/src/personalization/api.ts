import { requestJson } from "../api";
import type { PersonalProfile, ProfileModel, ProfileSettings, ProfileVersion } from "./types";
import * as device from "./device";
import type { ProfileMode } from "./types";

export const profileModels = () => requestJson<ProfileModel[]>("/api/personalization/models");
export const getProfile = (mode: ProfileMode): Promise<PersonalProfile> =>
  mode === "device" ? device.getProfile() : requestJson("/api/personalization");

const post = <T>(path: string, body: unknown) =>
  requestJson<T>(`/api/personalization/${path}`, { method: "POST", body: JSON.stringify(body) });

export function saveProfileSettings(
  mode: ProfileMode,
  profile: PersonalProfile,
  settings: ProfileSettings
): Promise<PersonalProfile> {
  return mode === "device"
    ? device.saveSettings(profile.revision, settings)
    : post("settings", { expected_revision: profile.revision, settings });
}
export function saveProfileFact(
  mode: ProfileMode,
  profile: PersonalProfile,
  id: string | null,
  topic: string,
  content: string
): Promise<PersonalProfile> {
  return mode === "device"
    ? device.editFact(profile.revision, id, topic, content)
    : post(id ? `facts/${encodeURIComponent(id)}` : "facts", {
        expected_revision: profile.revision,
        topic,
        content
      });
}
export function forgetProfile(
  mode: ProfileMode,
  profile: PersonalProfile,
  id: string | null
): Promise<PersonalProfile> {
  return mode === "device"
    ? device.forget(profile.revision, id)
    : post(id ? `facts/${encodeURIComponent(id)}/forget` : "forget", {
        expected_revision: profile.revision
      });
}
export function processProfile(
  mode: ProfileMode,
  profile: PersonalProfile
): Promise<PersonalProfile> {
  return mode === "device"
    ? device.processNow()
    : post("process", { expected_revision: profile.revision });
}
export function profileHistory(mode: ProfileMode, id: string): Promise<ProfileVersion[]> {
  return mode === "device"
    ? device.history(id)
    : requestJson(`/api/personalization/facts/${encodeURIComponent(id)}/history`);
}
