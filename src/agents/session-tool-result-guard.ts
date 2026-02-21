import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, basename } from "node:path";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { TextContent } from "@mariozechner/pi-ai";
import type { SessionManager } from "@mariozechner/pi-coding-agent";
import { emitSessionTranscriptUpdate } from "../sessions/transcript-events.js";
import { HARD_MAX_TOOL_RESULT_CHARS } from "./pi-embedded-runner/tool-result-truncation.js";
import { makeMissingToolResult, sanitizeToolCallInputs } from "./session-transcript-repair.js";

/**
 * Soft character limit for tool results persisted to the session transcript.
 * Results exceeding this are truncated in-context and spilled to a file.
 * Keeps the first and last SPILL_KEEP_CHARS characters inline.
 */
const SOFT_MAX_TOOL_RESULT_CHARS = 4_000;

/**
 * Characters to keep from the beginning and end of a spilled tool result.
 */
const SPILL_KEEP_CHARS = 1_000;

const GUARD_TRUNCATION_SUFFIX =
  "\n\n⚠️ [Content truncated during persistence — original exceeded size limit. " +
  "Use offset/limit parameters or request specific sections for large content.]";

/**
 * Write the full tool result text to a spill file alongside the session.
 * Returns the spill file path, or null if writing failed.
 */
function writeSpillFile(
  sessionFile: string | null,
  toolCallId: string | undefined,
  fullText: string,
): string | null {
  if (!sessionFile) {
    return null;
  }
  try {
    const dir = dirname(sessionFile);
    const sessionBase = basename(sessionFile, ".jsonl");
    const suffix = toolCallId ? `.${toolCallId}` : `.${Date.now()}`;
    const spillPath = `${dir}/${sessionBase}.tool_result${suffix}.txt`;
    mkdirSync(dir, { recursive: true });
    writeFileSync(spillPath, fullText, "utf-8");
    return spillPath;
  } catch {
    return null;
  }
}

/**
 * Build the truncation notice that replaces oversized content inline.
 */
function buildSpillNotice(
  headText: string,
  tailText: string,
  originalLength: number,
  spillPath: string | null,
  headChars: number,
  tailChars: number,
): string {
  const fileInstruction = spillPath
    ? `The full content (${originalLength} characters) has been saved to:\n  ${spillPath}\n\n` +
      `To access the full content, read the file in chunks of ~2000 characters using offset/limit. ` +
      `Consider delegating extraction to a sub-agent if only specific data is needed.`
    : `The full content (${originalLength} characters) was too large to persist. ` +
      `Re-run the tool with offset/limit parameters to read smaller sections.`;

  return (
    headText +
    `\n\n⚠️ [TRUNCATED — showing first ${headChars} and last ${tailChars} of ${originalLength} characters]\n` +
    `${fileInstruction}\n\n` +
    `--- END OF FILE (last ${tailChars} chars) ---\n` +
    tailText
  );
}

/**
 * Truncate oversized text content blocks in a tool result message.
 *
 * Two-tier truncation:
 * 1. Soft limit (SOFT_MAX_TOOL_RESULT_CHARS): spills full content to a file,
 *    keeps first+last SPILL_KEEP_CHARS inline with a read instruction.
 * 2. Hard limit (HARD_MAX_TOOL_RESULT_CHARS): legacy fallback — truncates
 *    to the beginning only (no spill file).
 *
 * Returns the original message if under the soft limit.
 */
function capToolResultSize(
  msg: AgentMessage,
  meta?: { sessionFile?: string | null; toolCallId?: string },
): AgentMessage {
  const role = (msg as { role?: string }).role;
  if (role !== "toolResult") {
    return msg;
  }
  const content = (msg as { content?: unknown }).content;
  if (!Array.isArray(content)) {
    return msg;
  }

  // Calculate total text size
  let totalTextChars = 0;
  for (const block of content) {
    if (block && typeof block === "object" && (block as { type?: string }).type === "text") {
      const text = (block as TextContent).text;
      if (typeof text === "string") {
        totalTextChars += text.length;
      }
    }
  }

  if (totalTextChars <= SOFT_MAX_TOOL_RESULT_CHARS) {
    return msg;
  }

  // Collect all text for spill file
  const allTexts: string[] = [];
  for (const block of content) {
    if (block && typeof block === "object" && (block as { type?: string }).type === "text") {
      const text = (block as TextContent).text;
      if (typeof text === "string") {
        allTexts.push(text);
      }
    }
  }
  const fullText = allTexts.join("\n");

  // Write spill file
  const spillPath = writeSpillFile(meta?.sessionFile ?? null, meta?.toolCallId, fullText);

  // Build truncated content: first SPILL_KEEP_CHARS + notice + last SPILL_KEEP_CHARS
  const newContent = content.map((block: unknown) => {
    if (!block || typeof block !== "object" || (block as { type?: string }).type !== "text") {
      return block;
    }
    const textBlock = block as TextContent;
    if (typeof textBlock.text !== "string") {
      return block;
    }
    if (textBlock.text.length <= SOFT_MAX_TOOL_RESULT_CHARS) {
      return block;
    }

    const headChars = Math.min(SPILL_KEEP_CHARS, textBlock.text.length);
    const tailChars = Math.min(SPILL_KEEP_CHARS, Math.max(0, textBlock.text.length - headChars));

    // Try to break at newline boundaries
    let headEnd = headChars;
    const headNewline = textBlock.text.lastIndexOf("\n", headChars);
    if (headNewline > headChars * 0.8) {
      headEnd = headNewline;
    }

    let tailStart = textBlock.text.length - tailChars;
    const tailNewline = textBlock.text.indexOf("\n", tailStart);
    if (tailNewline !== -1 && tailNewline < tailStart + tailChars * 0.2) {
      tailStart = tailNewline + 1;
    }

    const headText = textBlock.text.slice(0, headEnd);
    const tailText = textBlock.text.slice(tailStart);
    const actualHeadChars = headText.length;
    const actualTailChars = tailText.length;

    return {
      ...textBlock,
      text: buildSpillNotice(
        headText,
        tailText,
        textBlock.text.length,
        spillPath,
        actualHeadChars,
        actualTailChars,
      ),
    };
  });

  return { ...msg, content: newContent } as AgentMessage;
}

type ToolCall = { id: string; name?: string };

function extractAssistantToolCalls(msg: Extract<AgentMessage, { role: "assistant" }>): ToolCall[] {
  const content = msg.content;
  if (!Array.isArray(content)) {
    return [];
  }

  const toolCalls: ToolCall[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const rec = block as { type?: unknown; id?: unknown; name?: unknown };
    if (typeof rec.id !== "string" || !rec.id) {
      continue;
    }
    if (rec.type === "toolCall" || rec.type === "toolUse" || rec.type === "functionCall") {
      toolCalls.push({
        id: rec.id,
        name: typeof rec.name === "string" ? rec.name : undefined,
      });
    }
  }
  return toolCalls;
}

function extractToolResultId(msg: Extract<AgentMessage, { role: "toolResult" }>): string | null {
  const toolCallId = (msg as { toolCallId?: unknown }).toolCallId;
  if (typeof toolCallId === "string" && toolCallId) {
    return toolCallId;
  }
  const toolUseId = (msg as { toolUseId?: unknown }).toolUseId;
  if (typeof toolUseId === "string" && toolUseId) {
    return toolUseId;
  }
  return null;
}

export function installSessionToolResultGuard(
  sessionManager: SessionManager,
  opts?: {
    /**
     * Optional transform applied to any message before persistence.
     */
    transformMessageForPersistence?: (message: AgentMessage) => AgentMessage;
    /**
     * Optional, synchronous transform applied to toolResult messages *before* they are
     * persisted to the session transcript.
     */
    transformToolResultForPersistence?: (
      message: AgentMessage,
      meta: { toolCallId?: string; toolName?: string; isSynthetic?: boolean },
    ) => AgentMessage;
    /**
     * Whether to synthesize missing tool results to satisfy strict providers.
     * Defaults to true.
     */
    allowSyntheticToolResults?: boolean;
  },
): {
  flushPendingToolResults: () => void;
  getPendingIds: () => string[];
} {
  const originalAppend = sessionManager.appendMessage.bind(sessionManager);
  const pending = new Map<string, string | undefined>();
  const persistMessage = (message: AgentMessage) => {
    const transformer = opts?.transformMessageForPersistence;
    return transformer ? transformer(message) : message;
  };

  const persistToolResult = (
    message: AgentMessage,
    meta: { toolCallId?: string; toolName?: string; isSynthetic?: boolean },
  ) => {
    const transformer = opts?.transformToolResultForPersistence;
    return transformer ? transformer(message, meta) : message;
  };

  const allowSyntheticToolResults = opts?.allowSyntheticToolResults ?? true;

  const flushPendingToolResults = () => {
    if (pending.size === 0) {
      return;
    }
    if (allowSyntheticToolResults) {
      for (const [id, name] of pending.entries()) {
        const synthetic = makeMissingToolResult({ toolCallId: id, toolName: name });
        originalAppend(
          persistToolResult(persistMessage(synthetic), {
            toolCallId: id,
            toolName: name,
            isSynthetic: true,
          }) as never,
        );
      }
    }
    pending.clear();
  };

  const guardedAppend = (message: AgentMessage) => {
    let nextMessage = message;
    const role = (message as { role?: unknown }).role;
    if (role === "assistant") {
      const sanitized = sanitizeToolCallInputs([message]);
      if (sanitized.length === 0) {
        if (allowSyntheticToolResults && pending.size > 0) {
          flushPendingToolResults();
        }
        return undefined;
      }
      nextMessage = sanitized[0];
    }
    const nextRole = (nextMessage as { role?: unknown }).role;

    if (nextRole === "toolResult") {
      const id = extractToolResultId(nextMessage as Extract<AgentMessage, { role: "toolResult" }>);
      const toolName = id ? pending.get(id) : undefined;
      if (id) {
        pending.delete(id);
      }
      // Apply size cap before persistence to prevent oversized tool results
      // from consuming the entire context window on subsequent LLM calls.
      // Results exceeding the soft limit are spilled to a file alongside the session.
      const sessionFile = (
        sessionManager as { getSessionFile?: () => string | null }
      ).getSessionFile?.() ?? null;
      const capped = capToolResultSize(persistMessage(nextMessage), {
        sessionFile,
        toolCallId: id ?? undefined,
      });
      return originalAppend(
        persistToolResult(capped, {
          toolCallId: id ?? undefined,
          toolName,
          isSynthetic: false,
        }) as never,
      );
    }

    const toolCalls =
      nextRole === "assistant"
        ? extractAssistantToolCalls(nextMessage as Extract<AgentMessage, { role: "assistant" }>)
        : [];

    if (allowSyntheticToolResults) {
      // If previous tool calls are still pending, flush before non-tool results.
      if (pending.size > 0 && (toolCalls.length === 0 || nextRole !== "assistant")) {
        flushPendingToolResults();
      }
      // If new tool calls arrive while older ones are pending, flush the old ones first.
      if (pending.size > 0 && toolCalls.length > 0) {
        flushPendingToolResults();
      }
    }

    const result = originalAppend(persistMessage(nextMessage) as never);

    const sessionFile = (
      sessionManager as { getSessionFile?: () => string | null }
    ).getSessionFile?.();
    if (sessionFile) {
      emitSessionTranscriptUpdate(sessionFile);
    }

    if (toolCalls.length > 0) {
      for (const call of toolCalls) {
        pending.set(call.id, call.name);
      }
    }

    return result;
  };

  // Monkey-patch appendMessage with our guarded version.
  sessionManager.appendMessage = guardedAppend as SessionManager["appendMessage"];

  return {
    flushPendingToolResults,
    getPendingIds: () => Array.from(pending.keys()),
  };
}
