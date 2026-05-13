import { useState } from "react";
import { Lock, MessageSquare } from "lucide-react";
import { prepareLocalAttachment, preparePrivateAttachment } from "./attachments";
import { BrandMark } from "./common";
import { StartChatComposer } from "./Composer";
import { defaultToolPreferences } from "./toolPreferences";
import type {
  AvailableTool,
  ChatToolPreferences,
  ComposerAttachment,
  ModelInfo,
  NewChatMode
} from "./types";

export function ChatHome({
  error,
  isCreating,
  isCreatingPrivate,
  mode,
  selectedModel,
  selectedModelInfo,
  availableTools,
  onModeChange,
  onCreateChat,
  onCreatePrivateChat
}: {
  error: string | null;
  isCreating: boolean;
  isCreatingPrivate: boolean;
  mode: NewChatMode;
  selectedModel: string;
  selectedModelInfo: ModelInfo | null;
  availableTools: AvailableTool[];
  onModeChange: (mode: NewChatMode) => void;
  onCreateChat: (
    prompt: string,
    attachments?: ComposerAttachment[],
    toolPreferences?: ChatToolPreferences
  ) => Promise<void>;
  onCreatePrivateChat: (prompt: string, attachments?: ComposerAttachment[]) => Promise<void>;
}) {
  const isPrivate = mode === "private";
  const isCreatingSelectedMode = isPrivate ? isCreatingPrivate : isCreating;
  const [toolPreferences, setToolPreferences] =
    useState<ChatToolPreferences>(defaultToolPreferences);

  return (
    <div className="chat-home">
      <div className="chat-home-inner">
        <BrandMark compact />
        <div className="new-chat-mode" aria-label="New chat mode">
          <button
            type="button"
            className={!isPrivate ? "new-chat-mode-option active" : "new-chat-mode-option"}
            onClick={() => onModeChange("standard")}
          >
            <MessageSquare />
            <span>Standard</span>
          </button>
          <button
            type="button"
            className={isPrivate ? "new-chat-mode-option active" : "new-chat-mode-option"}
            onClick={() => onModeChange("private")}
          >
            <Lock />
            <span>Private</span>
          </button>
        </div>
        <StartChatComposer
          isBusy={isCreatingSelectedMode}
          isDisabled={!selectedModel}
          placeholder={
            selectedModel
              ? isPrivate
                ? "Message private chat"
                : "Message Vashti"
              : "Select a model to start"
          }
          selectedModelInfo={selectedModelInfo}
          availableTools={isPrivate ? [] : availableTools}
          toolPreferences={toolPreferences}
          onToolPreferencesChange={setToolPreferences}
          onUploadAttachment={isPrivate ? preparePrivateAttachment : prepareLocalAttachment}
          onSubmit={isPrivate ? onCreatePrivateChat : onCreateChat}
        />
        <p className={isPrivate ? "chat-mode-note private" : "chat-mode-note"}>
          {isPrivate ? <Lock /> : <MessageSquare />}
          <span>
            {isPrivate
              ? "Private chats are stored only on this device and are not synced."
              : "Standard chats are saved on the server and available when you sign in."}
          </span>
        </p>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
