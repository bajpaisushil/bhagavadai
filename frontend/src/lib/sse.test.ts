import { describe, expect, it } from "vitest";

import { SSEParser } from "./sse";

describe("SSEParser", () => {
  it("parses events split across arbitrary chunk boundaries", () => {
    const parser = new SSEParser();
    const wire =
      'data: {"type": "start"}\n\ndata: {"type": "delta", "text": "Śuka"}\n\n' +
      'data: {"type": "done", "references": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "seconds": 1, "tokens_per_s": 2}, "finish_reason": "stop"}\n\n';
    const events = [];
    for (let i = 0; i < wire.length; i += 7) events.push(...parser.push(wire.slice(i, i + 7)));
    expect(events.map((e) => e.type)).toEqual(["start", "delta", "done"]);
    expect(events[1]).toEqual({ type: "delta", text: "Śuka" });
  });

  it("handles CRLF line endings, including a CR/LF pair split across chunks", () => {
    const parser = new SSEParser();
    expect(parser.push('data: {"type": "start"}\r')).toEqual([]);
    expect(parser.push("\n\r\n")).toEqual([{ type: "start" }]);
  });

  it("skips malformed and unknown events", () => {
    const parser = new SSEParser();
    expect(parser.push('data: not json\n\ndata: {"type": "mystery"}\n\n: comment\n\n')).toEqual([]);
  });
});
