import type { ReactNode, SyntheticEvent } from "react";
import { RetroLoader, ThinkingLoader } from "./common";
import { MarkdownContent } from "./MarkdownContent";
import { toolIcon } from "./toolUi";
import type { ChatMessage, MessageStreamSegment, ThinkingSegment, ToolUsageRecord } from "./types";

export function MessageStreamContent({
  message,
  segments,
  thinkingDurationSeconds,
  isThinkingOpen,
  onThinkingToggle
}: {
  message: ChatMessage;
  segments: MessageStreamSegment[];
  thinkingDurationSeconds: number | null;
  isThinkingOpen: boolean;
  onThinkingToggle: (event: SyntheticEvent<HTMLDetailsElement>) => void;
}) {
  const visibleSegments = segments.filter(
    (segment) => segment.type === "tool" || segment.text.trim() !== ""
  );

  if (visibleSegments.length === 0) {
    return <p>{message.status === "streaming" ? <RetroLoader /> : "No content"}</p>;
  }

  return (
    <div className="message-stream-content">
      {visibleSegments.map((segment, index) => {
        if (segment.type === "content") {
          return <MarkdownContent key={`content-${index}`} content={segment.text} />;
        }

        if (segment.type === "tool") {
          return <ToolUsageCard key={`tool-${index}`} usage={segment.usage} />;
        }

        const segmentThinkingDuration =
          message.status === "streaming" && index === visibleSegments.length - 1
            ? null
            : thinkingDurationSeconds;

        return (
          <details
            key={`thinking-${index}`}
            className="message-thinking"
            open={isThinkingOpen}
            onToggle={onThinkingToggle}
          >
            <summary>{thinkingSummary(message, segmentThinkingDuration)}</summary>
            <ThinkingContent segments={[{ type: "text", text: segment.text.trim() }]} />
          </details>
        );
      })}
    </div>
  );
}

export function ThinkingContent({ segments }: { segments: ThinkingSegment[] }) {
  return (
    <div className="message-thinking-content">
      {segments.map((segment, index) => (
        segment.type === "text" ? (
          <p key={`text-${index}`}>{segment.text}</p>
        ) : (
          <ToolUsageCard key={`tool-${index}`} usage={segment.usage} />
        )
      ))}
    </div>
  );
}

function ToolUsageCard({ usage }: { usage: ToolUsageRecord }) {
  return (
    <details className="message-tool-card">
      <summary>
        {toolIcon(usage.name)}
        <span>{usage.summary}</span>
      </summary>
      <div className="message-tool-details">
        <label>
          <span>Arguments</span>
          <pre>{formatToolValue(usage.arguments)}</pre>
        </label>
        <label>
          <span>Result</span>
          <pre>{formatToolResult(usage.result)}</pre>
        </label>
      </div>
    </details>
  );
}

function formatToolValue(value: unknown) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function formatToolResult(result: string) {
  try {
    return JSON.stringify(JSON.parse(result), null, 2);
  } catch {
    return result;
  }
}

export function thinkingSummary(
  message: ChatMessage,
  thinkingDurationSeconds: number | null
): ReactNode {
  if (message.status === "streaming" && thinkingDurationSeconds === null) {
    return <ThinkingLoader />;
  }

  const durationSeconds = thinkingDurationSeconds ?? estimatedThinkingDurationSeconds(message);
  return `Thought for ${formatThoughtDuration(durationSeconds)}`;
}

function estimatedThinkingDurationSeconds(message: ChatMessage) {
  const startedAt = message.started_at ?? message.created_at;
  const endedAt = message.completed_at ?? message.updated_at;
  return Math.max(1, endedAt - startedAt);
}

function formatThoughtDuration(seconds: number) {
  const rounded = Math.max(1, Math.round(seconds));
  if (rounded < 60) {
    return `${rounded}s`;
  }

  const minutes = Math.floor(rounded / 60);
  const remainingSeconds = rounded % 60;
  return remainingSeconds === 0 ? `${minutes}m` : `${minutes}m ${remainingSeconds}s`;
}
