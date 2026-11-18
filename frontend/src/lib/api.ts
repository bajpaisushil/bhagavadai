import { SSEParser } from "./sse";
import type { ChatMessage, Health, ModelInfo, Reference, Usage } from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly retryAfterSeconds?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function errorFrom(response: Response): Promise<ApiError> {
  let message = `Request failed (HTTP ${response.status})`;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") message = body.detail;
    else if (Array.isArray(body.detail) && body.detail.length > 0) {
      const first = body.detail[0] as { msg?: string };
      if (first.msg) message = first.msg;
    }
  } catch {
    // non-JSON error body
  }
  const retry = Number(response.headers.get("retry-after"));
  return new ApiError(message, response.status, Number.isFinite(retry) && retry > 0 ? retry : undefined);
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal, cache: "no-store" });
  if (!response.ok) throw await errorFrom(response);
  return (await response.json()) as T;
}

export const getHealth = (signal?: AbortSignal) => getJson<Health>("/api/health", signal);
export const getInfo = (signal?: AbortSignal) => getJson<ModelInfo>("/api/info", signal);

export interface StreamHandlers {
  onDelta: (text: string) => void;
  onDone: (result: { references: Reference[]; usage: Usage; finishReason: string }) => void;
}

/** POST the conversation and stream the reply as server-sent events. */
export async function streamChat(messages: ChatMessage[], handlers: StreamHandlers, signal: AbortSignal): Promise<void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify({ messages, max_new_tokens: 320, temperature: 0.7, top_p: 0.9 }),
    signal,
  });
  if (!response.ok || !response.body) throw await errorFrom(response);
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = new SSEParser();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const event of parser.push(value)) {
      if (event.type === "delta") handlers.onDelta(event.text);
      else if (event.type === "done")
        handlers.onDone({ references: event.references, usage: event.usage, finishReason: event.finish_reason });
      else if (event.type === "error") throw new ApiError(event.message, 500);
    }
  }
}
