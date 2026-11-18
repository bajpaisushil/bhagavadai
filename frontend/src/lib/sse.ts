import type { StreamEvent } from "./types";

const TYPES = new Set(["start", "delta", "done", "error"]);

function isStreamEvent(value: unknown): value is StreamEvent {
  return typeof value === "object" && value !== null && TYPES.has((value as { type?: unknown }).type as string);
}

/**
 * Incremental parser for server-sent events whose `data:` lines carry JSON.
 * Network chunks can split an event anywhere; complete events (terminated by a blank line) are returned
 * as soon as they are available, and unknown or malformed events are skipped.
 */
export class SSEParser {
  private buffer = "";

  push(chunk: string): StreamEvent[] {
    this.buffer = (this.buffer + chunk).replace(/\r\n?/g, "\n");
    const events: StreamEvent[] = [];
    let boundary = this.buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n");
      if (data) {
        try {
          const parsed: unknown = JSON.parse(data);
          if (isStreamEvent(parsed)) events.push(parsed);
        } catch {
          // ignore malformed payloads
        }
      }
      boundary = this.buffer.indexOf("\n\n");
    }
    return events;
  }
}
